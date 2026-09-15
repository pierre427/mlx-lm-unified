"""Topology-first host policy for heterogeneous asynchronous work.

The coordinator consumes observations and emits placements or serving-knob
decisions. It does not probe devices, dispatch work, or synchronize a runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from threading import Lock
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

from .heterogeneous_execution import Engine, EngineCapability, OperationMeasurement


def _finite_nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _confidence(value: float) -> None:
    _finite_nonnegative("confidence", value)
    if value > 1:
        raise ValueError("confidence must be in [0, 1]")


class EstimateProvenance(str, Enum):
    MEASURED = "measured"
    CALIBRATED = "calibrated"
    MODELED = "modeled"


@dataclass(frozen=True)
class ComputeDomain:
    domain_id: str
    engine: Engine
    operations: frozenset[str]
    memory_headroom_bytes: int
    max_concurrency: int = 1
    available: bool = True
    supports_overlap: bool = False

    @classmethod
    def from_engine_capability(
        cls,
        domain_id: str,
        capability: EngineCapability,
        *,
        memory_headroom_bytes: int,
        max_concurrency: int = 1,
    ) -> "ComputeDomain":
        return cls(
            domain_id,
            capability.engine,
            capability.operations,
            memory_headroom_bytes,
            max_concurrency,
            capability.available,
            capability.supports_overlap,
        )

    def __post_init__(self) -> None:
        if not self.domain_id or not isinstance(self.engine, Engine):
            raise ValueError("compute domain identity is incomplete")
        if not self.operations or not all(self.operations):
            raise ValueError("compute domain operations are required")
        if (
            isinstance(self.memory_headroom_bytes, bool)
            or not isinstance(self.memory_headroom_bytes, int)
            or self.memory_headroom_bytes < 0
        ):
            raise ValueError("memory headroom must be a non-negative integer")
        if (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency < 1
        ):
            raise ValueError("maximum concurrency must be positive")


@dataclass(frozen=True)
class ComputeTopologySnapshot:
    revision: str
    sequence: int
    domains: tuple[ComputeDomain, ...]

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("topology revision is required")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("topology sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("topology sequence must be non-negative")
        ids = tuple(domain.domain_id for domain in self.domains)
        if len(set(ids)) != len(ids):
            raise ValueError("compute domain ids must be unique")

    @property
    def by_id(self) -> Mapping[str, ComputeDomain]:
        return MappingProxyType({domain.domain_id: domain for domain in self.domains})


@dataclass(frozen=True)
class OperationCostEstimate:
    operation: str
    domain_id: str
    service_ms: float
    working_set_bytes: int
    quality_risk: float
    confidence: float
    provenance: EstimateProvenance
    sample_count: int = 0

    def __post_init__(self) -> None:
        if not self.operation or not self.domain_id:
            raise ValueError("operation and domain are required")
        _finite_nonnegative("service time", self.service_ms)
        if (
            isinstance(self.working_set_bytes, bool)
            or not isinstance(self.working_set_bytes, int)
            or self.working_set_bytes < 0
        ):
            raise ValueError("working set must be a non-negative integer")
        _finite_nonnegative("quality risk", self.quality_risk)
        if self.quality_risk > 1:
            raise ValueError("quality risk must be in [0, 1]")
        _confidence(self.confidence)
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise ValueError("sample count must be non-negative")
        if not isinstance(self.provenance, EstimateProvenance):
            raise ValueError("estimate provenance is invalid")


@dataclass(frozen=True)
class OperationObservation:
    operation: str
    domain_id: str
    service_ms: float
    working_set_bytes: int
    quality_risk: float = 0.0
    provenance: EstimateProvenance = EstimateProvenance.MEASURED

    @classmethod
    def from_measurement(
        cls, measurement: OperationMeasurement, *, domain_id: str
    ) -> "OperationObservation":
        return cls(
            measurement.operation,
            domain_id,
            measurement.total_us / 1000.0,
            measurement.geometry.max_bytes,
            0.0,
            EstimateProvenance.MEASURED,
        )

    def __post_init__(self) -> None:
        OperationCostEstimate(
            self.operation,
            self.domain_id,
            self.service_ms,
            self.working_set_bytes,
            self.quality_risk,
            0.0,
            self.provenance,
        )


class OperationCostBook:
    """Deterministic online means with explicit source and confidence."""

    def __init__(
        self,
        seeds: Iterable[OperationCostEstimate] = (),
        *,
        confidence_step: float = 0.2,
    ) -> None:
        _confidence(confidence_step)
        if confidence_step == 0:
            raise ValueError("confidence step must be positive")
        self.confidence_step = float(confidence_step)
        self._estimates: dict[tuple[str, str], OperationCostEstimate] = {}
        for estimate in seeds:
            key = (estimate.operation, estimate.domain_id)
            if key in self._estimates:
                raise ValueError("cost estimate keys must be unique")
            self._estimates[key] = estimate
        self._lock = Lock()

    def get(self, operation: str, domain_id: str) -> OperationCostEstimate | None:
        with self._lock:
            return self._estimates.get((operation, domain_id))

    def observe(self, observation: OperationObservation) -> OperationCostEstimate:
        key = (observation.operation, observation.domain_id)
        with self._lock:
            previous = self._estimates.get(key)
            count = 1 if previous is None else previous.sample_count + 1
            if previous is None or previous.sample_count == 0:
                service_ms = observation.service_ms
                working_set = observation.working_set_bytes
                quality_risk = observation.quality_risk
            else:
                prior_count = previous.sample_count
                service_ms = (
                    previous.service_ms * prior_count + observation.service_ms
                ) / count
                working_set = round(
                    (
                        previous.working_set_bytes * prior_count
                        + observation.working_set_bytes
                    )
                    / count
                )
                quality_risk = (
                    previous.quality_risk * prior_count + observation.quality_risk
                ) / count
            estimate = OperationCostEstimate(
                observation.operation,
                observation.domain_id,
                service_ms,
                working_set,
                quality_risk,
                min(1.0, count * self.confidence_step),
                observation.provenance,
                count,
            )
            self._estimates[key] = estimate
            return estimate

    def snapshot(self) -> tuple[OperationCostEstimate, ...]:
        with self._lock:
            return tuple(self._estimates[key] for key in sorted(self._estimates))


@dataclass(frozen=True)
class AsyncWorkItem:
    work_id: str
    operation: str
    owner_id: str
    weight: int = 1
    queued_ticks: int = 0
    eligible_domain_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.work_id or not self.operation or not self.owner_id:
            raise ValueError("work identity is incomplete")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, int)
            or self.weight < 1
        ):
            raise ValueError("work weight must be positive")
        if (
            isinstance(self.queued_ticks, bool)
            or not isinstance(self.queued_ticks, int)
            or self.queued_ticks < 0
        ):
            raise ValueError("queued ticks must be non-negative")


@dataclass(frozen=True)
class WorkPlacement:
    work_id: str
    topology_revision: str
    domain_id: str
    engine: Engine
    projected_service_ms: float
    fairness_credit: float
    estimate_confidence: float
    estimate_provenance: EstimateProvenance

    def __post_init__(self) -> None:
        if not self.work_id or not self.topology_revision or not self.domain_id:
            raise ValueError("placement identity is incomplete")
        if not isinstance(self.engine, Engine):
            raise ValueError("placement engine is invalid")
        _finite_nonnegative("projected service", self.projected_service_ms)
        if not math.isfinite(self.fairness_credit):
            raise ValueError("fairness credit must be finite")
        _confidence(self.estimate_confidence)
        if not isinstance(self.estimate_provenance, EstimateProvenance):
            raise ValueError("placement provenance is invalid")


@dataclass(frozen=True)
class AdaptiveCoordinatorConfig:
    enabled: bool = False
    min_estimate_confidence: float = 0.2
    max_quality_risk: float = 0.2
    aging_credit_per_tick: float = 0.25

    def __post_init__(self) -> None:
        _confidence(self.min_estimate_confidence)
        _finite_nonnegative("maximum quality risk", self.max_quality_risk)
        if self.max_quality_risk > 1:
            raise ValueError("maximum quality risk must be in [0, 1]")
        _finite_nonnegative("aging credit", self.aging_credit_per_tick)


class AdaptiveWorkCoordinator:
    """Place queued work with measured eligibility and deficit fairness."""

    def __init__(
        self,
        costs: OperationCostBook,
        *,
        config: AdaptiveCoordinatorConfig | None = None,
    ) -> None:
        self.costs = costs
        self.config = config or AdaptiveCoordinatorConfig()
        self._deficits: dict[str, float] = {}
        self._lock = Lock()
        self._last_topology_sequence: int | None = None
        self._last_topology_revision: str | None = None

    def allocate(
        self,
        topology: ComputeTopologySnapshot,
        pending: Iterable[AsyncWorkItem],
        *,
        max_placements: Optional[int] = None,
    ) -> tuple[WorkPlacement, ...]:
        items = tuple(pending)
        if len({item.work_id for item in items}) != len(items):
            raise ValueError("work ids must be unique")
        limit = (
            sum(domain.max_concurrency for domain in topology.domains)
            if max_placements is None
            else max_placements
        )
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("maximum placements must be a non-negative integer")
        if not self.config.enabled or not items or limit == 0:
            return ()
        with self._lock:
            if (
                self._last_topology_sequence is not None
                and topology.sequence < self._last_topology_sequence
            ):
                return ()
            if (
                topology.sequence == self._last_topology_sequence
                and topology.revision != self._last_topology_revision
            ):
                return ()
            self._last_topology_sequence = topology.sequence
            self._last_topology_revision = topology.revision

            domains = topology.by_id
            remaining = {
                domain.domain_id: domain.max_concurrency
                for domain in topology.domains
            }
            memory_remaining = {
                domain.domain_id: domain.memory_headroom_bytes
                for domain in topology.domains
            }

            def routes(item):
                allowed = (
                    set(item.eligible_domain_ids)
                    if item.eligible_domain_ids
                    else set(domains)
                )
                result = []
                for domain_id in sorted(allowed):
                    domain = domains.get(domain_id)
                    if (
                        domain is None
                        or not domain.available
                        or remaining[domain_id] == 0
                        or item.operation not in domain.operations
                    ):
                        continue
                    estimate = self.costs.get(item.operation, domain_id)
                    if (
                        estimate is None
                        or estimate.confidence < self.config.min_estimate_confidence
                        or estimate.quality_risk > self.config.max_quality_risk
                        or estimate.working_set_bytes > memory_remaining[domain_id]
                    ):
                        continue
                    result.append((domain, estimate))
                return result

            eligible = [item for item in items if routes(item)]
            owner_weights: dict[str, int] = {}
            for item in eligible:
                owner_weights[item.owner_id] = max(
                    owner_weights.get(item.owner_id, 0), item.weight
                )
            for owner_id, weight in owner_weights.items():
                self._deficits[owner_id] = self._deficits.get(owner_id, 0.0) + weight
            fairness_quantum = sum(owner_weights.values())

            placements = []
            unplaced = list(items)
            while unplaced and len(placements) < limit:
                candidates = []
                for item in unplaced:
                    item_routes = routes(item)
                    credit = self._deficits.get(item.owner_id, 0.0) + (
                        item.queued_ticks * self.config.aging_credit_per_tick
                    )
                    for domain, estimate in item_routes:
                        candidates.append(
                            (item, domain, estimate, credit, len(item_routes))
                        )
                if not candidates:
                    break
                item, domain, estimate, credit, _ = min(
                    candidates,
                    key=lambda value: (
                        -value[3],
                        value[4],
                        value[2].service_ms,
                        value[0].work_id,
                        value[1].domain_id,
                    ),
                )
                placements.append(
                    WorkPlacement(
                        item.work_id,
                        topology.revision,
                        domain.domain_id,
                        domain.engine,
                        estimate.service_ms,
                        credit,
                        estimate.confidence,
                        estimate.provenance,
                    )
                )
                remaining[domain.domain_id] -= 1
                memory_remaining[domain.domain_id] -= estimate.working_set_bytes
                self._deficits[item.owner_id] -= fairness_quantum
                unplaced.remove(item)
            return tuple(placements)


@dataclass(frozen=True)
class ServingKnobState:
    batch_size: int
    concurrency: int
    mtp_draft_length: int

    def __post_init__(self) -> None:
        for name, value in (
            ("batch size", self.batch_size),
            ("concurrency", self.concurrency),
            ("MTP draft length", self.mtp_draft_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ServingKnobBounds:
    min_batch_size: int
    max_batch_size: int
    min_concurrency: int
    max_concurrency: int
    min_mtp_draft_length: int
    max_mtp_draft_length: int

    def __post_init__(self) -> None:
        pairs = (
            ("batch size", self.min_batch_size, self.max_batch_size),
            ("concurrency", self.min_concurrency, self.max_concurrency),
            (
                "MTP draft length",
                self.min_mtp_draft_length,
                self.max_mtp_draft_length,
            ),
        )
        for name, minimum, maximum in pairs:
            if (
                any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (minimum, maximum)
                )
                or not 1 <= minimum <= maximum
            ):
                raise ValueError(f"{name} bounds are invalid")

    def accepts(self, state: ServingKnobState) -> bool:
        return (
            self.min_batch_size <= state.batch_size <= self.max_batch_size
            and self.min_concurrency <= state.concurrency <= self.max_concurrency
            and self.min_mtp_draft_length
            <= state.mtp_draft_length
            <= self.max_mtp_draft_length
        )


@dataclass(frozen=True)
class ServingKnobOption:
    option_id: str
    state: ServingKnobState
    memory_delta_bytes: int
    latency_delta_ms: float
    throughput_delta: float
    quality_risk: float
    confidence: float
    provenance: EstimateProvenance

    def __post_init__(self) -> None:
        if not self.option_id:
            raise ValueError("knob option id is required")
        if isinstance(self.memory_delta_bytes, bool) or not isinstance(
            self.memory_delta_bytes, int
        ):
            raise ValueError("memory delta must be an integer")
        for name, value in (
            ("latency delta", self.latency_delta_ms),
            ("throughput delta", self.throughput_delta),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        _finite_nonnegative("quality risk", self.quality_risk)
        if self.quality_risk > 1:
            raise ValueError("quality risk must be in [0, 1]")
        _confidence(self.confidence)
        if not isinstance(self.provenance, EstimateProvenance):
            raise ValueError("option provenance is invalid")


@dataclass(frozen=True)
class SchedulerBoundary:
    sequence: int
    batch_formation_open: bool
    decode_round_complete: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("scheduler boundary sequence must be non-negative")


@dataclass(frozen=True)
class ServingKnobDecision:
    state: ServingKnobState
    option_id: str | None
    reason: str


class ServingKnobController:
    """Choose evidence-backed knobs with hysteresis and safe boundaries."""

    def __init__(
        self,
        bounds: ServingKnobBounds,
        *,
        enabled: bool = False,
        min_confidence: float = 0.4,
        max_quality_risk: float = 0.2,
        cooldown_ticks: int = 4,
        min_objective_gain: float = 0.1,
    ) -> None:
        self.bounds = bounds
        self.enabled = bool(enabled)
        _confidence(min_confidence)
        _confidence(max_quality_risk)
        _finite_nonnegative("minimum objective gain", min_objective_gain)
        if (
            isinstance(cooldown_ticks, bool)
            or not isinstance(cooldown_ticks, int)
            or cooldown_ticks < 0
        ):
            raise ValueError("cooldown ticks must be non-negative")
        self.min_confidence = min_confidence
        self.max_quality_risk = max_quality_risk
        self.cooldown_ticks = cooldown_ticks
        self.min_objective_gain = min_objective_gain
        self._last_change_sequence: int | None = None
        self._last_boundary_sequence: int | None = None
        self._lock = Lock()

    def decide(
        self,
        current: ServingKnobState,
        options: Iterable[ServingKnobOption],
        boundary: SchedulerBoundary,
        *,
        memory_pressure: float,
        latency_pressure: float,
        batch_pressure: float,
    ) -> ServingKnobDecision:
        for name, value in (
            ("memory pressure", memory_pressure),
            ("latency pressure", latency_pressure),
            ("batch pressure", batch_pressure),
        ):
            _confidence(value)
        if not self.bounds.accepts(current):
            raise ValueError("current serving knobs are outside controller bounds")
        choices = tuple(options)
        if len({option.option_id for option in choices}) != len(choices):
            raise ValueError("knob option ids must be unique")
        with self._lock:
            if (
                self._last_boundary_sequence is not None
                and boundary.sequence <= self._last_boundary_sequence
            ):
                return ServingKnobDecision(current, None, "stale_boundary")
            self._last_boundary_sequence = boundary.sequence
            if not self.enabled:
                return ServingKnobDecision(current, None, "disabled")
            if self._last_change_sequence is not None and (
                boundary.sequence - self._last_change_sequence < self.cooldown_ticks
            ):
                return ServingKnobDecision(current, None, "cooldown")

            candidates = []
            for option in choices:
                if (
                    option.state == current
                    or not self.bounds.accepts(option.state)
                    or option.confidence < self.min_confidence
                    or option.quality_risk > self.max_quality_risk
                ):
                    continue
                batch_changed = (
                    option.state.batch_size != current.batch_size
                    or option.state.concurrency != current.concurrency
                )
                mtp_changed = option.state.mtp_draft_length != current.mtp_draft_length
                if batch_changed and not boundary.batch_formation_open:
                    continue
                if mtp_changed and not boundary.decode_round_complete:
                    continue
                memory_relief = max(0, -option.memory_delta_bytes) / (1024**3)
                memory_cost = max(0, option.memory_delta_bytes) / (1024**3)
                latency_relief = max(0.0, -option.latency_delta_ms)
                latency_cost = max(0.0, option.latency_delta_ms)
                score = (
                    memory_pressure * (memory_relief - memory_cost)
                    + latency_pressure * (latency_relief - latency_cost)
                    + batch_pressure * option.throughput_delta
                    - option.quality_risk
                )
                if score >= self.min_objective_gain:
                    candidates.append((score, option))
            if not candidates:
                return ServingKnobDecision(current, None, "hysteresis")
            score, selected = min(
                candidates, key=lambda value: (-value[0], value[1].option_id)
            )
            self._last_change_sequence = boundary.sequence
            return ServingKnobDecision(
                selected.state, selected.option_id, f"objective_gain={score:.6f}"
            )


__all__ = [
    "AdaptiveCoordinatorConfig",
    "AdaptiveWorkCoordinator",
    "AsyncWorkItem",
    "ComputeDomain",
    "ComputeTopologySnapshot",
    "EstimateProvenance",
    "OperationCostBook",
    "OperationCostEstimate",
    "OperationObservation",
    "SchedulerBoundary",
    "ServingKnobBounds",
    "ServingKnobController",
    "ServingKnobDecision",
    "ServingKnobOption",
    "ServingKnobState",
    "WorkPlacement",
]
