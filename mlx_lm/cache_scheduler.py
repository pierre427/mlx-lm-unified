# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""Default-off lineage batching and per-plane retention policies.

This module contains host-only metadata decisions. It never reads a device
array, measures a hot path, or synchronizes an accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from threading import Lock, RLock
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

from .cache_planes import CachePlaneKind


def layered_cache_scheduler_enabled(value: Optional[bool] = None) -> bool:
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_LAYERED_CACHE_SCHEDULER", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class PlaneCompatibility:
    kind: CachePlaneKind
    fingerprint_digest: str
    generation: int
    hit: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CachePlaneKind):
            raise ValueError("cache plane kind is invalid")
        if not self.fingerprint_digest:
            raise ValueError("cache plane fingerprint is required")
        _nonnegative_int("cache plane generation", self.generation)


@dataclass(frozen=True)
class CacheBatchRequest:
    request_id: str
    lineage_id: str
    arrival_sequence: int
    compatibility: tuple[PlaneCompatibility, ...]

    def __post_init__(self) -> None:
        if not self.request_id or not self.lineage_id:
            raise ValueError("request and lineage ids are required")
        _nonnegative_int("arrival sequence", self.arrival_sequence)
        kinds = tuple(item.kind for item in self.compatibility)
        if len(set(kinds)) != len(kinds):
            raise ValueError("compatibility vector contains duplicate planes")
        if tuple(sorted(kinds, key=lambda item: item.value)) != kinds:
            raise ValueError("compatibility vector must be sorted by plane")

    @property
    def hit_vector(self) -> tuple[tuple[CachePlaneKind, str, int], ...]:
        return tuple(
            (item.kind, item.fingerprint_digest, item.generation)
            for item in self.compatibility
            if item.hit
        )


@dataclass(frozen=True)
class CacheBatchGroup:
    lineage_id: str
    requests: tuple[CacheBatchRequest, ...]
    shared_planes: tuple[tuple[CachePlaneKind, str, int], ...]

    @property
    def is_partial_hit(self) -> bool:
        vectors = {request.hit_vector for request in self.requests}
        return len(self.requests) > 1 and len(vectors) > 1


class CacheSchedulerMetrics:
    """Bounded host counters; no timers, allocations, or device readbacks."""

    _KEYS = (
        "schedule_calls",
        "requests",
        "groups",
        "partial_hit_groups",
        "lineage_splits",
        "admissions",
        "admission_rejections",
        "evictions",
        "evicted_bytes",
    )

    def __init__(self) -> None:
        self._values = {key: 0 for key in self._KEYS}
        self._lock = Lock()

    def add(self, key: str, amount: int = 1) -> None:
        if key not in self._values:
            raise KeyError(f"unknown cache scheduler metric: {key}")
        _nonnegative_int("metric amount", amount)
        with self._lock:
            self._values[key] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            result = dict(self._values)
        result["device_synchronizations"] = 0
        result["timed_hot_path_sections"] = 0
        return result


class LineageBatchScheduler:
    """Group FIFO requests by lineage and their common compatible planes."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        min_shared_planes: int = 1,
        metrics: CacheSchedulerMetrics | None = None,
    ) -> None:
        if min_shared_planes < 1:
            raise ValueError("minimum shared planes must be positive")
        self.enabled = bool(enabled)
        self.min_shared_planes = int(min_shared_planes)
        self.metrics = metrics or CacheSchedulerMetrics()

    def group(
        self, requests: Iterable[CacheBatchRequest]
    ) -> tuple[CacheBatchGroup, ...]:
        ordered = tuple(
            sorted(requests, key=lambda item: (item.arrival_sequence, item.request_id))
        )
        if len({item.request_id for item in ordered}) != len(ordered):
            raise ValueError("request ids must be unique")
        self.metrics.add("schedule_calls")
        self.metrics.add("requests", len(ordered))
        if not self.enabled:
            groups = tuple(
                CacheBatchGroup(request.lineage_id, (request,), ())
                for request in ordered
            )
            self.metrics.add("groups", len(groups))
            return groups

        by_lineage: dict[str, list[CacheBatchRequest]] = {}
        for request in ordered:
            by_lineage.setdefault(request.lineage_id, []).append(request)
        if by_lineage:
            self.metrics.add("lineage_splits", max(0, len(by_lineage) - 1))

        groups: list[CacheBatchGroup] = []
        for lineage_id in sorted(by_lineage):
            pending = list(by_lineage[lineage_id])
            while pending:
                anchor = pending.pop(0)
                members = [anchor]
                shared = set(anchor.hit_vector)
                retained = []
                for candidate in pending:
                    intersection = shared.intersection(candidate.hit_vector)
                    if len(intersection) >= self.min_shared_planes:
                        members.append(candidate)
                        shared = intersection
                    else:
                        retained.append(candidate)
                pending = retained
                group = CacheBatchGroup(
                    lineage_id,
                    tuple(members),
                    tuple(sorted(shared, key=lambda item: item[0].value)),
                )
                groups.append(group)
                if group.is_partial_hit:
                    self.metrics.add("partial_hit_groups")

        groups.sort(
            key=lambda group: (
                group.requests[0].arrival_sequence,
                group.requests[0].request_id,
            )
        )
        self.metrics.add("groups", len(groups))
        return tuple(groups)


@dataclass(frozen=True)
class PlaneRetentionRecord:
    key: str
    lineage_id: str
    kind: CachePlaneKind
    size_bytes: int
    recompute_cost_us: float
    last_access_sequence: int
    pinned: bool = False
    contains_mutable_request_data: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.lineage_id:
            raise ValueError("retention key and lineage are required")
        if not isinstance(self.kind, CachePlaneKind):
            raise ValueError("retention plane kind is invalid")
        _nonnegative_int("retention size", self.size_bytes)
        _nonnegative_int("last access sequence", self.last_access_sequence)
        if not math.isfinite(self.recompute_cost_us) or self.recompute_cost_us < 0:
            raise ValueError("recompute cost must be finite and non-negative")

    @property
    def recompute_value(self) -> float:
        return self.recompute_cost_us / max(1, self.size_bytes)


@dataclass(frozen=True)
class RetentionDecision:
    admitted: bool
    evicted_keys: tuple[str, ...]
    reason: str
    candidate_value: float


class PlaneRetentionManager:
    """Apply independent recomputation-value budgets to cache planes."""

    def __init__(
        self,
        plane_budgets: Mapping[CachePlaneKind, int],
        *,
        enabled: bool = False,
        metrics: CacheSchedulerMetrics | None = None,
    ) -> None:
        budgets = {}
        for kind, budget in plane_budgets.items():
            if not isinstance(kind, CachePlaneKind):
                raise ValueError("retention budget plane kind is invalid")
            _nonnegative_int("retention budget", budget)
            budgets[kind] = budget
        self.plane_budgets = MappingProxyType(budgets)
        self.enabled = bool(enabled)
        self.metrics = metrics or CacheSchedulerMetrics()
        self._records: dict[str, PlaneRetentionRecord] = {}
        self._lock = RLock()

    def consider(self, candidate: PlaneRetentionRecord) -> RetentionDecision:
        with self._lock:
            if not self.enabled:
                return self._reject(candidate, "disabled")
            if candidate.contains_mutable_request_data:
                return self._reject(candidate, "mutable_request_data")
            budget = self.plane_budgets.get(candidate.kind, 0)
            if candidate.size_bytes > budget:
                return self._reject(candidate, "over_plane_budget")
            current = self._records.get(candidate.key)
            if current is not None and current.pinned and current != candidate:
                return self._reject(candidate, "pinned_replacement")
            occupied = sum(
                record.size_bytes
                for key, record in self._records.items()
                if record.kind == candidate.kind and key != candidate.key
            )
            required = max(0, occupied + candidate.size_bytes - budget)
            victims = sorted(
                (
                    record
                    for key, record in self._records.items()
                    if key != candidate.key
                    and record.kind == candidate.kind
                    and not record.pinned
                    and record.recompute_value < candidate.recompute_value
                ),
                key=lambda item: (
                    item.recompute_value,
                    item.last_access_sequence,
                    item.key,
                ),
            )
            evicted = []
            recovered = 0
            for victim in victims:
                if recovered >= required:
                    break
                evicted.append(victim)
                recovered += victim.size_bytes
            if recovered < required:
                return self._reject(candidate, "lower_recompute_value")
            for victim in evicted:
                del self._records[victim.key]
            self._records[candidate.key] = candidate
            self.metrics.add("admissions")
            self.metrics.add("evictions", len(evicted))
            self.metrics.add(
                "evicted_bytes", sum(item.size_bytes for item in evicted)
            )
            reason = "replaced" if current is not None else "admitted"
            return RetentionDecision(
                True,
                tuple(item.key for item in evicted),
                reason,
                candidate.recompute_value,
            )

    def _reject(
        self, candidate: PlaneRetentionRecord, reason: str
    ) -> RetentionDecision:
        self.metrics.add("admission_rejections")
        return RetentionDecision(False, (), reason, candidate.recompute_value)

    def records(self) -> tuple[PlaneRetentionRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._records.values(),
                    key=lambda item: (item.kind.value, item.key),
                )
            )

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._records

    def touch(self, key: str, access_sequence: int) -> bool:
        _nonnegative_int("access sequence", access_sequence)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return False
            if access_sequence < record.last_access_sequence:
                raise ValueError("access sequence must be monotone")
            self._records[key] = replace(
                record, last_access_sequence=access_sequence
            )
            return True

    def set_pinned(self, key: str, pinned: bool) -> bool:
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return False
            self._records[key] = replace(record, pinned=bool(pinned))
            return True


__all__ = [
    "CacheBatchGroup",
    "CacheBatchRequest",
    "CacheSchedulerMetrics",
    "LineageBatchScheduler",
    "PlaneCompatibility",
    "PlaneRetentionManager",
    "PlaneRetentionRecord",
    "RetentionDecision",
    "layered_cache_scheduler_enabled",
]
