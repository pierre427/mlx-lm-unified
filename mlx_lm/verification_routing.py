"""Host policy for a future exact delayed-verification backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class PendingVerification:
    batch_id: str
    acceptance_probability: float
    queued_ms: float
    projected_gpu_gib: float
    exact_ane_eligible: bool

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id is required")
        if not 0.0 <= self.acceptance_probability <= 1.0:
            raise ValueError("acceptance_probability must be in [0, 1]")
        for name, value in (
            ("queued_ms", self.queued_ms),
            ("projected_gpu_gib", self.projected_gpu_gib),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ANEVerificationDecision:
    selected_batch_id: str | None
    reason: str
    ranked_batch_ids: tuple[str, ...]


def select_ane_verification(
    pending: Iterable[PendingVerification],
    *,
    enabled: bool,
    memory_pressure: bool,
    gpu_verify_delay_ms: float,
    min_gpu_delay_ms: float = 10.0,
    starvation_ms: float = 250.0,
) -> ANEVerificationDecision:
    """Select one exact-eligible batch without claiming an ANE implementation.

    Starved work wins first. Otherwise the highest expected acceptance wins,
    with age as a bounded tie-breaker. The caller must supply an independently
    qualified exact ANE backend before acting on the selection.
    """
    candidates = tuple(item for item in pending if item.exact_ane_eligible)
    ranked = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.queued_ms < starvation_ms,
                -(
                    item.acceptance_probability
                    + 0.25 * min(item.queued_ms / max(starvation_ms, 1.0), 1.0)
                ),
                -item.queued_ms,
                item.batch_id,
            ),
        )
    )
    ids = tuple(item.batch_id for item in ranked)
    if not enabled:
        return ANEVerificationDecision(None, "disabled", ids)
    if not ranked:
        return ANEVerificationDecision(None, "no_exact_eligible_batch", ids)
    if not memory_pressure and gpu_verify_delay_ms < min_gpu_delay_ms:
        return ANEVerificationDecision(None, "gpu_service_is_timely", ids)
    return ANEVerificationDecision(ranked[0].batch_id, "eligible", ids)
