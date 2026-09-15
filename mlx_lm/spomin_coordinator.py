"""Default-off scheduling for pressure-driven Spomin compaction.

The coordinator owns policy and lifecycle only. Segment scoring may run on an
asynchronous CPU or ANE adapter, but cache mutation remains a backend operation
and is allowed only at a scheduler-declared safe point.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
import math
import os
from threading import Lock
from typing import Any, Mapping, Optional, Protocol, Sequence

from .adaptive_work_coordinator import (
    AsyncWorkItem,
    ComputeTopologySnapshot,
    OperationCostBook,
    WorkPlacement,
)
from .heterogeneous_execution import Engine
from .spomin_layer import (
    SegmentImportanceScorer,
    SpominBackend,
    SpominConfig,
    SpominLayer,
    SpominPlan,
    SpominTargetState,
    StaticSegmentScorer,
)


def spomin_coordinator_enabled(value: Optional[bool] = None) -> bool:
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_SPOMIN_COORDINATOR", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ratio(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


class SpominCoordinatorError(RuntimeError):
    pass


class SpominSafePointError(SpominCoordinatorError):
    pass


class SpominMethodKind(str, Enum):
    ANE_SEMANTIC = "ane_semantic"
    CPU_SEMANTIC = "cpu_semantic"
    HEAD_HOTSPOT = "head_hotspot"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class SpominPressureSignals:
    """Host scheduler observation with no device readback."""

    system_memory_pressure: float
    batch_pressure: float
    available_overlap_ms: float = 0.0

    def __post_init__(self) -> None:
        _ratio("system memory pressure", self.system_memory_pressure)
        _ratio("batch pressure", self.batch_pressure)
        _nonnegative("available overlap", self.available_overlap_ms)


@dataclass(frozen=True)
class SpominCoordinatorConfig:
    enabled: bool = False
    memory_pressure_threshold: float = 0.70
    batch_pressure_threshold: float = 0.70
    max_quality_risk: float = 0.20
    max_exposed_latency_ms: float = 5.0
    min_reclaim_tokens: int = 1
    min_estimate_confidence: float = 0.20

    def __post_init__(self) -> None:
        _ratio("memory pressure threshold", self.memory_pressure_threshold)
        _ratio("batch pressure threshold", self.batch_pressure_threshold)
        _ratio("maximum quality risk", self.max_quality_risk)
        _ratio("minimum estimate confidence", self.min_estimate_confidence)
        _nonnegative("maximum exposed latency", self.max_exposed_latency_ms)
        if (
            isinstance(self.min_reclaim_tokens, bool)
            or not isinstance(self.min_reclaim_tokens, int)
            or self.min_reclaim_tokens < 1
        ):
            raise ValueError("minimum reclaim tokens must be positive")


@dataclass(frozen=True)
class SegmentScoreStamp:
    target_revision: str
    transcript_digest: str


@dataclass(frozen=True)
class SegmentScoreResult:
    stamp: SegmentScoreStamp
    scores: Mapping[str, float]
    receipt: Mapping[str, Any] = field(default_factory=dict)


class AsyncSegmentScoringAdapter(Protocol):
    """Device-neutral boundary for disposable asynchronous scoring work."""

    engine: Engine

    def submit(self, transcript, stamp: SegmentScoreStamp) -> object | None: ...

    def resolve(self, ticket: object) -> SegmentScoreResult | None: ...

    def abandon(self, ticket: object) -> None: ...


class ANESemanticScoringAdapter(AsyncSegmentScoringAdapter, Protocol):
    """ANE boundary: score transcript segments, never edit model caches."""


@dataclass(frozen=True)
class SpominMethod:
    """One measured compaction option supplied by the serving scheduler."""

    method_id: str
    kind: SpominMethodKind
    strategy: str
    domain_id: str
    engine: Engine
    estimated_reclaim_tokens: int
    ready: bool = True
    scorer: SegmentImportanceScorer | None = None
    async_adapter: AsyncSegmentScoringAdapter | None = None

    def __post_init__(self) -> None:
        if not self.method_id or not self.domain_id:
            raise ValueError("Spomin method and domain ids are required")
        if not isinstance(self.kind, SpominMethodKind):
            raise ValueError("Spomin method kind is invalid")
        if not isinstance(self.engine, Engine):
            raise ValueError("Spomin method engine is invalid")
        if (
            isinstance(self.estimated_reclaim_tokens, bool)
            or not isinstance(self.estimated_reclaim_tokens, int)
            or self.estimated_reclaim_tokens < 0
        ):
            raise ValueError("estimated reclaim tokens must be non-negative")
        if self.async_adapter is not None:
            if self.async_adapter.engine is not self.engine:
                raise ValueError("async adapter engine disagrees with method")
            if self.scorer is not None:
                raise ValueError("a method cannot have sync and async scorers")
        if self.kind is SpominMethodKind.ANE_SEMANTIC:
            if self.engine is not Engine.ANE or self.async_adapter is None:
                raise ValueError("ANE semantic methods require an ANE adapter")
        if self.kind is SpominMethodKind.CPU_SEMANTIC and self.engine is not Engine.CPU:
            raise ValueError("CPU semantic methods require the CPU engine")
        if self.strategy == "lowest_importance" and (
            self.scorer is None and self.async_adapter is None
        ):
            raise ValueError("lowest_importance methods require a scorer")

    @property
    def asynchronous(self) -> bool:
        return self.async_adapter is not None

    @property
    def operation(self) -> str:
        return f"spomin/{self.method_id}"

    @property
    def work_id(self) -> str:
        return f"spomin:{self.method_id}"

    def as_work_item(
        self, owner_id: str, *, weight: int = 1, queued_ticks: int = 0
    ) -> AsyncWorkItem:
        return AsyncWorkItem(
            self.work_id,
            self.operation,
            owner_id,
            weight,
            queued_ticks,
            (self.domain_id,),
        )


@dataclass(frozen=True)
class SpominMethodDecision:
    method: SpominMethod | None
    reason: str
    considered_method_ids: tuple[str, ...]


@dataclass(frozen=True)
class SpominCoordinationTicket:
    stamp: SegmentScoreStamp
    method: SpominMethod
    adapter_ticket: object | None = None


@dataclass(frozen=True)
class SpominSafePoint:
    """Scheduler attestation required before publishing a cache replacement."""

    target_revision: str
    request_quiescent: bool
    device_work_drained: bool


class SpominCoordinatorMetrics:
    """Bounded host counters with no timers or device synchronization."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._lock = Lock()

    def add(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
        return {
            "counts": counts,
            "device_synchronizations": 0,
            "timed_hot_path_sections": 0,
        }


class SpominCoordinator:
    """Select, score, plan, and safely publish one Spomin operation."""

    def __init__(
        self,
        layer_config: SpominConfig,
        costs: OperationCostBook,
        *,
        config: SpominCoordinatorConfig | None = None,
        metrics: SpominCoordinatorMetrics | None = None,
    ) -> None:
        self.layer_config = layer_config
        self.costs = costs
        self.config = config or SpominCoordinatorConfig(
            enabled=spomin_coordinator_enabled()
        )
        self.metrics = metrics or SpominCoordinatorMetrics()

    def select_method(
        self,
        signals: SpominPressureSignals,
        methods: Sequence[SpominMethod],
        topology: ComputeTopologySnapshot,
        placements: Sequence[WorkPlacement],
    ) -> SpominMethodDecision:
        self.metrics.add("selection_calls")
        ordered_ids = tuple(method.method_id for method in methods)
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("Spomin method ids must be unique")
        if not self.config.enabled:
            self.metrics.add("declined_disabled")
            return SpominMethodDecision(None, "disabled", ordered_ids)

        memory_pressure = (
            signals.system_memory_pressure >= self.config.memory_pressure_threshold
        )
        batch_pressure = signals.batch_pressure >= self.config.batch_pressure_threshold
        if not memory_pressure and not batch_pressure:
            self.metrics.add("declined_below_pressure")
            return SpominMethodDecision(None, "below_pressure", ordered_ids)

        eligible = []
        domains = topology.by_id
        placements_by_work = {placement.work_id: placement for placement in placements}
        if len(placements_by_work) != len(placements):
            raise ValueError("work placements must be unique")
        for method in methods:
            domain = domains.get(method.domain_id)
            estimate = self.costs.get(method.operation, method.domain_id)
            placement = placements_by_work.get(method.work_id)
            if not method.ready:
                continue
            if (
                domain is None
                or not domain.available
                or domain.engine is not method.engine
                or method.operation not in domain.operations
                or estimate is None
                or placement is None
                or placement.topology_revision != topology.revision
                or placement.domain_id != method.domain_id
                or placement.engine is not method.engine
            ):
                continue
            if method.estimated_reclaim_tokens < self.config.min_reclaim_tokens:
                continue
            if estimate.confidence < self.config.min_estimate_confidence:
                continue
            if estimate.quality_risk > self.config.max_quality_risk:
                continue
            if estimate.working_set_bytes > domain.memory_headroom_bytes:
                continue
            overlap = (
                signals.available_overlap_ms
                if method.asynchronous and domain.supports_overlap
                else 0.0
            )
            exposed = max(0.0, estimate.service_ms - overlap)
            if exposed > self.config.max_exposed_latency_ms:
                continue
            eligible.append((method, estimate, exposed))
        if not eligible:
            self.metrics.add("declined_no_eligible_method")
            return SpominMethodDecision(None, "no_eligible_method", ordered_ids)

        def rank(candidate) -> tuple[Any, ...]:
            method, estimate, exposed = candidate
            if memory_pressure:
                return (
                    -method.estimated_reclaim_tokens,
                    estimate.quality_risk,
                    exposed,
                    -estimate.confidence,
                    method.method_id,
                )
            asynchronous_penalty = 0 if method.asynchronous else 1
            ane_penalty = 0 if method.engine is Engine.ANE else 1
            return (
                asynchronous_penalty,
                ane_penalty,
                exposed,
                estimate.quality_risk,
                -method.estimated_reclaim_tokens,
                -estimate.confidence,
                method.method_id,
            )

        selected, _, _ = min(eligible, key=rank)
        self.metrics.add(f"selected_{selected.kind.value}")
        reason = "memory_pressure" if memory_pressure else "batch_pressure"
        return SpominMethodDecision(selected, reason, ordered_ids)

    def begin(
        self,
        state: SpominTargetState,
        signals: SpominPressureSignals,
        methods: Sequence[SpominMethod],
        topology: ComputeTopologySnapshot,
        placements: Sequence[WorkPlacement],
    ) -> tuple[SpominMethodDecision, SpominCoordinationTicket | None]:
        decision = self.select_method(signals, methods, topology, placements)
        method = decision.method
        if method is None:
            return decision, None
        stamp = SegmentScoreStamp(
            target_revision=state.revision,
            transcript_digest=state.transcript.fingerprint.digest,
        )
        adapter_ticket = None
        if method.async_adapter is not None:
            adapter_ticket = method.async_adapter.submit(state.transcript, stamp)
            if adapter_ticket is None:
                self.metrics.add("async_submit_declined")
                return (
                    SpominMethodDecision(
                        None, "async_submit_declined", decision.considered_method_ids
                    ),
                    None,
                )
            self.metrics.add("async_submitted")
        return decision, SpominCoordinationTicket(stamp, method, adapter_ticket)

    def plan(
        self,
        ticket: SpominCoordinationTicket,
        state: SpominTargetState,
        *,
        protected_segment_ids: Sequence[str] = (),
        replacement_token_ids: Sequence[int] = (),
        target_limit_tokens: int | None = None,
    ) -> SpominPlan | None:
        current_stamp = SegmentScoreStamp(
            target_revision=state.revision,
            transcript_digest=state.transcript.fingerprint.digest,
        )
        if current_stamp != ticket.stamp:
            if ticket.adapter_ticket is not None:
                ticket.method.async_adapter.abandon(ticket.adapter_ticket)
            self.metrics.add("stale_before_plan")
            return None

        scorer = ticket.method.scorer
        if ticket.adapter_ticket is not None:
            result = ticket.method.async_adapter.resolve(ticket.adapter_ticket)
            if result is None:
                self.metrics.add("async_not_ready")
                return None
            if result.stamp != ticket.stamp:
                self.metrics.add("async_stale_result")
                return None
            scorer = StaticSegmentScorer(result.scores)
            self.metrics.add("async_resolved")

        layer = SpominLayer(
            replace(self.layer_config, strategy=ticket.method.strategy),
            scorer=scorer,
        )
        plan = layer.plan(
            state,
            protected_segment_ids=protected_segment_ids,
            replacement_token_ids=replacement_token_ids,
            target_limit_tokens=target_limit_tokens,
            force=True,
        )
        if plan is not None:
            self.metrics.add("plans_ready")
        return plan

    def abandon(self, ticket: SpominCoordinationTicket) -> bool:
        """Discard asynchronous scoring when its compaction opportunity closes."""

        if ticket.adapter_ticket is None:
            return False
        ticket.method.async_adapter.abandon(ticket.adapter_ticket)
        self.metrics.add("async_abandoned")
        return True

    def commit(
        self,
        state: SpominTargetState,
        plan: SpominPlan,
        backend: SpominBackend,
        safe_point: SpominSafePoint,
    ) -> SpominTargetState:
        if safe_point.target_revision != state.revision:
            self.metrics.add("commit_refused_revision")
            raise SpominSafePointError("safe point does not match target revision")
        if not safe_point.request_quiescent:
            self.metrics.add("commit_refused_active_request")
            raise SpominSafePointError("request is not quiescent")
        if not safe_point.device_work_drained:
            self.metrics.add("commit_refused_device_work")
            raise SpominSafePointError("device work has not drained")
        layer = SpominLayer(
            replace(self.layer_config, strategy=plan.selection.strategy)
        )
        updated = layer.apply(state, plan, backend)
        self.metrics.add("commits")
        return updated


__all__ = [
    "ANESemanticScoringAdapter",
    "AsyncSegmentScoringAdapter",
    "SegmentScoreResult",
    "SegmentScoreStamp",
    "SpominCoordinationTicket",
    "SpominCoordinator",
    "SpominCoordinatorConfig",
    "SpominCoordinatorError",
    "SpominCoordinatorMetrics",
    "SpominMethod",
    "SpominMethodDecision",
    "SpominMethodKind",
    "SpominPressureSignals",
    "SpominSafePoint",
    "SpominSafePointError",
    "spomin_coordinator_enabled",
]
