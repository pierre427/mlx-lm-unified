# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""Model-free contracts for measured heterogeneous execution planning.

The planner consumes exact-geometry measurements. It does not probe devices or
dispatch work. An execution island is the smallest placement unit; model layers
cannot use cross-engine boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from itertools import product
import json
import math
from statistics import median
from types import MappingProxyType
from typing import Iterable, Mapping, Optional


class Engine(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    ANE = "ane"
    NAX = "nax"


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class HostRuntimeFingerprint:
    """Stable host, model, and runtime identity for measurement reuse."""

    host: str
    chip: str
    os_build: str
    model_id: str
    model_revision: str
    quantization: str
    runtimes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        required = (
            self.host,
            self.chip,
            self.os_build,
            self.model_id,
            self.model_revision,
            self.quantization,
        )
        if not all(value.strip() for value in required):
            raise ValueError("fingerprint fields must be non-empty")
        if tuple(sorted(self.runtimes)) != self.runtimes:
            raise ValueError("runtime versions must be sorted")

    @property
    def digest(self) -> str:
        payload = {
            "host": self.host,
            "chip": self.chip,
            "os_build": self.os_build,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantization": self.quantization,
            "runtimes": self.runtimes,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class OperationGeometry:
    """Exact tensor and implementation geometry for one measured operation."""

    input_bytes: int
    output_bytes: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str
    layout: str
    operation_fingerprint: str

    def __post_init__(self) -> None:
        _require_nonnegative_int("input bytes", self.input_bytes)
        _require_nonnegative_int("output bytes", self.output_bytes)
        for dimension in self.input_shape + self.output_shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise ValueError("shape dimensions must be positive integers")
            if dimension <= 0:
                raise ValueError("shape dimensions must be positive integers")
        if not self.dtype or not self.layout or not self.operation_fingerprint:
            raise ValueError("dtype, layout, and operation fingerprint are required")

    @property
    def max_bytes(self) -> int:
        return max(self.input_bytes, self.output_bytes)


@dataclass(frozen=True)
class OperationMeasurement:
    """One observed operation cost, split at synchronization boundaries."""

    operation: str
    engine: Engine
    geometry: OperationGeometry
    fingerprint_digest: str
    dispatch_us: float
    copy_in_us: float
    compute_us: float
    sync_us: float
    copy_out_us: float
    owner_lifetime_us: float

    def __post_init__(self) -> None:
        if not isinstance(self.engine, Engine):
            raise ValueError("measurement engine is invalid")
        if not self.operation or not self.fingerprint_digest:
            raise ValueError("operation and fingerprint are required")
        for name in (
            "dispatch_us",
            "copy_in_us",
            "compute_us",
            "sync_us",
            "copy_out_us",
            "owner_lifetime_us",
        ):
            _require_finite(name, getattr(self, name))
        if self.owner_lifetime_us < self.total_us:
            raise ValueError("owner lifetime must cover the measured operation")

    @property
    def total_us(self) -> float:
        return (
            self.dispatch_us
            + self.copy_in_us
            + self.compute_us
            + self.sync_us
            + self.copy_out_us
        )


@dataclass(frozen=True)
class TransferMeasurement:
    """One measured cross-engine edge for an exact payload geometry."""

    source_engine: Engine
    target_engine: Engine
    geometry: OperationGeometry
    fingerprint_digest: str
    copy_us: float
    sync_us: float
    overlap_credit_us: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.source_engine, Engine) or not isinstance(
            self.target_engine, Engine
        ):
            raise ValueError("transfer engines are invalid")
        if self.source_engine == self.target_engine:
            raise ValueError("a transfer measurement must cross engines")
        if not self.fingerprint_digest:
            raise ValueError("transfer fingerprint is required")
        for name in ("copy_us", "sync_us", "overlap_credit_us"):
            _require_finite(name, getattr(self, name))
        if self.overlap_credit_us > self.copy_us + self.sync_us:
            raise ValueError("overlap credit exceeds transfer work")

    def effective_us(self, *, overlap_supported: bool) -> float:
        credit = self.overlap_credit_us if overlap_supported else 0.0
        return self.copy_us + self.sync_us - credit


@dataclass(frozen=True)
class EngineCapability:
    engine: Engine
    operations: frozenset[str]
    max_bytes: int
    available: bool = True
    supports_overlap: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.engine, Engine):
            raise ValueError("capability engine is invalid")
        _require_nonnegative_int("max bytes", self.max_bytes)
        if self.max_bytes == 0:
            raise ValueError("max bytes must be positive")
        if not all(isinstance(name, str) and name for name in self.operations):
            raise ValueError("capability operations must be non-empty strings")

    def supports(self, operation: str, geometry: OperationGeometry) -> bool:
        return (
            self.available
            and operation in self.operations
            and geometry.max_bytes <= self.max_bytes
        )


@dataclass(frozen=True)
class TransferBoundary:
    dependency_id: str
    geometry: OperationGeometry

    def __post_init__(self) -> None:
        if not self.dependency_id:
            raise ValueError("transfer dependency id is required")


@dataclass(frozen=True)
class OperationNode:
    """One DAG operation assigned as part of a whole execution island."""

    node_id: str
    operation: str
    geometry: OperationGeometry
    island: str
    dependencies: tuple[str, ...] = ()
    deadline_us: Optional[float] = None
    transfer_boundaries: tuple[TransferBoundary, ...] = ()
    model_layer: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.node_id or not self.operation or not self.island:
            raise ValueError("node id, operation, and island are required")
        if self.deadline_us is not None:
            _require_finite("deadline", self.deadline_us)
            if self.deadline_us == 0:
                raise ValueError("deadline must be positive")
        if self.model_layer is not None:
            _require_nonnegative_int("model layer", self.model_layer)
        if self.node_id in self.dependencies:
            raise ValueError("a node cannot depend on itself")
        boundary_ids = tuple(item.dependency_id for item in self.transfer_boundaries)
        if len(set(boundary_ids)) != len(boundary_ids):
            raise ValueError("transfer boundary dependencies must be unique")
        if any(dependency not in self.dependencies for dependency in boundary_ids):
            raise ValueError("transfer boundary is not a node dependency")


@dataclass(frozen=True)
class PlacementDecision:
    node_id: str
    engine: Engine
    projected_us: float
    ready_us: float
    completion_us: float
    meets_deadline: bool
    fallback_engines: tuple[Engine, ...]
    reason: str


@dataclass(frozen=True)
class TransferDecision:
    source_id: str
    target_id: str
    source_engine: Engine
    target_engine: Engine
    projected_us: float


@dataclass(frozen=True)
class ExecutionPlan:
    fingerprint_digest: str
    placements: tuple[PlacementDecision, ...]
    transfers: tuple[TransferDecision, ...]
    critical_path_us: float


class PlanningRefused(RuntimeError):
    """The available evidence cannot produce a safe placement plan."""


@dataclass(frozen=True)
class HeterogeneousExecutionProfile:
    """Exact-evidence engine selector with fail-closed reuse rules."""

    fingerprint: HostRuntimeFingerprint
    capabilities: Mapping[Engine, EngineCapability]
    max_plan_combinations: int = 4096
    _samples: dict[
        tuple[str, Engine, OperationGeometry], list[OperationMeasurement]
    ] = field(default_factory=dict)
    _transfers: dict[
        tuple[Engine, Engine, OperationGeometry], list[TransferMeasurement]
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonnegative_int("max plan combinations", self.max_plan_combinations)
        if self.max_plan_combinations == 0:
            raise ValueError("max plan combinations must be positive")
        if not self.capabilities:
            raise ValueError("at least one capability is required")
        for key, capability in self.capabilities.items():
            if not isinstance(key, Engine) or key != capability.engine:
                raise ValueError("capability map key does not match its engine")
        object.__setattr__(
            self, "capabilities", MappingProxyType(dict(self.capabilities))
        )

    def record(self, sample: OperationMeasurement) -> None:
        if sample.fingerprint_digest != self.fingerprint.digest:
            raise ValueError("measurement fingerprint does not match this host")
        capability = self.capabilities.get(sample.engine)
        if capability is None or not capability.supports(
            sample.operation, sample.geometry
        ):
            raise ValueError("measurement is outside the engine capability")
        key = (sample.operation, sample.engine, sample.geometry)
        self._samples.setdefault(key, []).append(sample)

    def record_transfer(self, sample: TransferMeasurement) -> None:
        if sample.fingerprint_digest != self.fingerprint.digest:
            raise ValueError("transfer fingerprint does not match this host")
        source = self.capabilities.get(sample.source_engine)
        target = self.capabilities.get(sample.target_engine)
        if source is None or target is None:
            raise ValueError("transfer engine has no capability")
        if not source.available or not target.available:
            raise ValueError("transfer engine is unavailable")
        if sample.geometry.max_bytes > min(source.max_bytes, target.max_bytes):
            raise ValueError("transfer is outside an engine capability")
        key = (sample.source_engine, sample.target_engine, sample.geometry)
        self._transfers.setdefault(key, []).append(sample)

    def projected_us(
        self, operation: str, engine: Engine, geometry: OperationGeometry
    ) -> Optional[float]:
        samples = self._samples.get((operation, engine, geometry), ())
        if not samples:
            return None
        return float(median(sample.total_us for sample in samples))

    def projected_transfer_us(
        self,
        source: Engine,
        target: Engine,
        geometry: OperationGeometry,
    ) -> Optional[float]:
        samples = self._transfers.get((source, target, geometry), ())
        if not samples:
            return None
        overlap = (
            self.capabilities[source].supports_overlap
            and self.capabilities[target].supports_overlap
        )
        return float(
            median(
                sample.effective_us(overlap_supported=overlap)
                for sample in samples
            )
        )

    def _island_candidates(
        self, nodes: list[OperationNode]
    ) -> tuple[Engine, ...]:
        candidates: list[tuple[float, Engine]] = []
        for engine, capability in self.capabilities.items():
            costs: list[float] = []
            for node in nodes:
                if not capability.supports(node.operation, node.geometry):
                    break
                cost = self.projected_us(node.operation, engine, node.geometry)
                if cost is None:
                    break
                costs.append(cost)
            else:
                candidates.append((sum(costs), engine))
        return tuple(
            engine
            for _, engine in sorted(
                candidates, key=lambda item: (item[0], item[1].value)
            )
        )

    def plan(self, nodes: Iterable[OperationNode]) -> ExecutionPlan:
        ordered = _topological_order(tuple(nodes))
        by_id = {node.node_id: node for node in ordered}
        islands: dict[str, list[OperationNode]] = {}
        for node in ordered:
            islands.setdefault(node.island, []).append(node)
        island_names = tuple(sorted(islands))
        candidates = {
            island: self._island_candidates(islands[island])
            for island in island_names
        }
        for island, engines in candidates.items():
            if not engines:
                raise PlanningRefused(
                    f"island {island!r} has no exact-geometry measured engine"
                )
        combination_count = math.prod(
            len(candidates[name]) for name in island_names
        )
        if combination_count > self.max_plan_combinations:
            raise PlanningRefused("engine-placement search exceeds its bound")

        feasible = []
        for choices in product(*(candidates[name] for name in island_names)):
            assignment = dict(zip(island_names, choices))
            evaluated = self._evaluate_assignment(ordered, by_id, assignment)
            if evaluated is None:
                continue
            timings, transfers, critical_path = evaluated
            tie_break = tuple(assignment[name].value for name in island_names)
            feasible.append(
                (critical_path, tie_break, assignment, timings, transfers)
            )
        if not feasible:
            raise PlanningRefused(
                "no placement has exact transfer evidence and meets all deadlines"
            )

        critical_path, _, assignment, timings, transfers = min(
            feasible, key=lambda row: (row[0], row[1])
        )
        alternatives: dict[str, tuple[Engine, ...]] = {}
        for island in island_names:
            selected = assignment[island]
            alternatives[island] = tuple(
                engine
                for engine in candidates[island]
                if engine != selected
                and any(row[2][island] == engine for row in feasible)
            )
        placements = tuple(
            PlacementDecision(
                node_id=node.node_id,
                engine=assignment[node.island],
                projected_us=timings[node.node_id][0],
                ready_us=timings[node.node_id][1],
                completion_us=timings[node.node_id][2],
                meets_deadline=(
                    node.deadline_us is None
                    or timings[node.node_id][2] <= node.deadline_us
                ),
                fallback_engines=alternatives[node.island],
                reason="minimum measured critical path",
            )
            for node in ordered
        )
        return ExecutionPlan(
            self.fingerprint.digest, placements, transfers, critical_path
        )

    def _evaluate_assignment(
        self,
        ordered: tuple[OperationNode, ...],
        by_id: Mapping[str, OperationNode],
        assignment: Mapping[str, Engine],
    ):
        timings: dict[str, tuple[float, float, float]] = {}
        transfers: list[TransferDecision] = []
        for node in ordered:
            engine = assignment[node.island]
            operation_us = self.projected_us(
                node.operation, engine, node.geometry
            )
            assert operation_us is not None
            ready_us = 0.0
            boundaries = {
                boundary.dependency_id: boundary
                for boundary in node.transfer_boundaries
            }
            for dependency in node.dependencies:
                parent = by_id[dependency]
                parent_engine = assignment[parent.island]
                edge_ready = timings[dependency][2]
                if parent_engine != engine:
                    if (
                        parent.model_layer is not None
                        and node.model_layer is not None
                    ):
                        return None
                    boundary = boundaries.get(dependency)
                    if boundary is None:
                        return None
                    transfer_us = self.projected_transfer_us(
                        parent_engine, engine, boundary.geometry
                    )
                    if transfer_us is None:
                        return None
                    edge_ready += transfer_us
                    transfers.append(
                        TransferDecision(
                            dependency,
                            node.node_id,
                            parent_engine,
                            engine,
                            transfer_us,
                        )
                    )
                ready_us = max(ready_us, edge_ready)
            completion_us = ready_us + operation_us
            if node.deadline_us is not None and completion_us > node.deadline_us:
                return None
            timings[node.node_id] = (operation_us, ready_us, completion_us)
        critical_path = max((row[2] for row in timings.values()), default=0.0)
        return timings, tuple(transfers), critical_path


def _topological_order(nodes: tuple[OperationNode, ...]) -> tuple[OperationNode, ...]:
    if not nodes:
        raise PlanningRefused("operation graph is empty")
    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise PlanningRefused("node ids must be unique")
    missing = {
        dependency
        for node in nodes
        for dependency in node.dependencies
        if dependency not in by_id
    }
    if missing:
        raise PlanningRefused(f"missing dependencies: {sorted(missing)!r}")
    pending = {node.node_id: set(node.dependencies) for node in nodes}
    ordered: list[OperationNode] = []
    while pending:
        ready = sorted(node_id for node_id, deps in pending.items() if not deps)
        if not ready:
            raise PlanningRefused("operation graph contains a cycle")
        for node_id in ready:
            ordered.append(by_id[node_id])
            del pending[node_id]
        for dependencies in pending.values():
            dependencies.difference_update(ready)
    return tuple(ordered)
