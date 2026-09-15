"""Adaptive byte budgeting for Spomin context compaction.

The policy is deliberately separate from segment selection and cache surgery.
It turns host-memory and batching observations into a per-session context
limit.  A caller may then ask ``SpominLayer`` to select segments for that
limit and choose an exact-rebuild or supported surgery backend.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Callable


def system_available_memory_bytes() -> int | None:
    """Return reclaimable host memory without reading or synchronizing the GPU."""

    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["/usr/bin/vm_stat"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            match = re.search(r"page size of (\d+) bytes", result.stdout)
            if match is None:
                return None
            page_size = int(match.group(1))
            counts = {}
            for line in result.stdout.splitlines()[1:]:
                if ":" not in line:
                    continue
                name, value = line.split(":", 1)
                counts[name] = int(value.strip().rstrip("."))
            pages = sum(
                counts.get(name, 0)
                for name in ("Pages free", "Pages inactive", "Pages speculative")
            )
            return pages * page_size if pages > 0 else None
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size if pages > 0 and page_size > 0 else None
    except (
        KeyError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


@dataclass(frozen=True)
class SpominPressureConfig:
    max_context_tokens: int
    min_context_tokens: int
    bytes_per_token_per_session: int
    max_total_active_kv_bytes: int
    system_reserve_bytes: int
    critical_available_bytes: int
    queued_lane_weight: float = 0.5
    pressure_samples: int = 2
    recovery_samples: int = 3

    def __post_init__(self):
        positive = (
            self.max_context_tokens,
            self.min_context_tokens,
            self.bytes_per_token_per_session,
            self.max_total_active_kv_bytes,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive):
            raise ValueError("Spomin pressure capacities must be positive")
        if self.min_context_tokens > self.max_context_tokens:
            raise ValueError("minimum context cannot exceed maximum context")
        if self.system_reserve_bytes < 0 or self.critical_available_bytes < 0:
            raise ValueError("memory thresholds must be non-negative")
        if not 0 <= self.queued_lane_weight <= 1:
            raise ValueError("queued_lane_weight must be between zero and one")
        if self.pressure_samples < 1 or self.recovery_samples < 1:
            raise ValueError("hysteresis sample counts must be positive")


@dataclass(frozen=True)
class SpominPressureSample:
    current_context_tokens: int
    active_sessions: int
    queued_sessions: int
    active_kv_bytes: int
    idle_apc_reclaimable_bytes: int = 0
    available_system_bytes: int | None = None

    def __post_init__(self):
        values = (
            self.current_context_tokens,
            self.active_sessions,
            self.queued_sessions,
            self.active_kv_bytes,
            self.idle_apc_reclaimable_bytes,
        )
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("Spomin pressure observations must be non-negative")
        if self.active_sessions < 1:
            raise ValueError("at least one active session is required")
        if self.available_system_bytes is not None and self.available_system_bytes < 0:
            raise ValueError("available_system_bytes must be non-negative")


@dataclass(frozen=True)
class SpominBudgetDecision:
    context_limit_tokens: int
    desired_context_tokens: int
    reclaim_idle_apc_bytes: int
    effective_lanes: float
    action: str
    reasons: tuple[str, ...]

    @property
    def compact_to_tokens(self) -> int | None:
        return self.context_limit_tokens if self.action == "compact" else None


class AdaptiveSpominPressurePolicy:
    """Stateful, hysteretic context budget for one serving cohort."""

    def __init__(
        self,
        config: SpominPressureConfig,
        *,
        memory_reader: Callable[[], int | None] = system_available_memory_bytes,
    ):
        self.config = config
        self.memory_reader = memory_reader
        self._limit = config.max_context_tokens
        self._pressure_streak = 0
        self._recovery_streak = 0

    @property
    def context_limit_tokens(self) -> int:
        return self._limit

    def observe(self, sample: SpominPressureSample) -> SpominBudgetDecision:
        cfg = self.config
        available = (
            self.memory_reader()
            if sample.available_system_bytes is None
            else sample.available_system_bytes
        )
        reclaim = 0
        if available is not None and available < cfg.system_reserve_bytes:
            reclaim = min(
                sample.idle_apc_reclaimable_bytes,
                cfg.system_reserve_bytes - available,
            )
        projected_available = None if available is None else available + reclaim

        # Queued work is discounted because it does not own KV yet.  Hysteresis
        # below prevents one arrival from immediately shrinking live context.
        effective_lanes = sample.active_sessions + (
            sample.queued_sessions * cfg.queued_lane_weight
        )
        total_budget = cfg.max_total_active_kv_bytes
        reasons = []
        if projected_available is not None:
            pressure_budget = max(
                0,
                sample.active_kv_bytes
                + projected_available
                - cfg.system_reserve_bytes,
            )
            if pressure_budget < total_budget:
                total_budget = pressure_budget
                reasons.append("system_memory")
        if effective_lanes > 1:
            reasons.append("batch")
        if reclaim:
            reasons.insert(0, "idle_apc_first")

        desired = int(
            total_budget
            / (cfg.bytes_per_token_per_session * effective_lanes)
        )
        desired = min(cfg.max_context_tokens, max(cfg.min_context_tokens, desired))
        critical = bool(
            available is not None and available <= cfg.critical_available_bytes
        )

        if desired < self._limit:
            self._pressure_streak += 1
            self._recovery_streak = 0
            if critical or self._pressure_streak >= cfg.pressure_samples:
                self._limit = desired
                self._pressure_streak = 0
        elif desired > self._limit:
            self._recovery_streak += 1
            self._pressure_streak = 0
            if self._recovery_streak >= cfg.recovery_samples:
                self._limit = desired
                self._recovery_streak = 0
        else:
            self._pressure_streak = 0
            self._recovery_streak = 0

        if sample.current_context_tokens > self._limit:
            action = "compact"
        elif reclaim:
            action = "evict_idle_apc"
        elif self._limit > sample.current_context_tokens:
            action = "headroom"
        else:
            action = "hold"
        if critical:
            reasons.append("critical")
        if not reasons:
            reasons.append("healthy_single_session")
        return SpominBudgetDecision(
            context_limit_tokens=self._limit,
            desired_context_tokens=desired,
            reclaim_idle_apc_bytes=reclaim,
            effective_lanes=effective_lanes,
            action=action,
            reasons=tuple(reasons),
        )
