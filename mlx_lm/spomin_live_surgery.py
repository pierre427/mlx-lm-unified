"""Default-on serving bridge for revision-bound Spomin KV surgery."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import os
from threading import Lock
from typing import Any, Mapping, Sequence

from .spomin_layer import (
    SpominCapabilityError,
    SpominConfig,
    SpominLayer,
    SpominPlan,
    SpominTargetState,
)
from .spomin_qwen4_surgery import Qwen4SpominSurgeryBackend


def live_surgery_enabled(value: bool | None = None) -> bool:
    """Return the process policy. Live surgery is wired on unless opted out."""

    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_SPOMIN_LIVE_SURGERY", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@dataclass(frozen=True)
class LiveSurgeryEpoch:
    request_id: str
    generation: int


@dataclass
class LiveSurgeryTransaction:
    """One request-private edit prepared before entering the generation loop."""

    manager: "SpominLiveSurgeryManager"
    epoch: LiveSurgeryEpoch
    state: SpominTargetState
    plan: SpominPlan
    prompt_token_ids: tuple[int, ...]
    receipt: dict[str, Any]
    retained_token_ids: tuple[int, ...] | None = None

    def apply(
        self,
        model,
        prompt_cache: Sequence[object],
        *,
        request_quiescent: bool,
        device_work_drained: bool,
    ) -> Mapping[str, Any]:
        return self.manager.apply(
            self,
            model,
            prompt_cache,
            request_quiescent=request_quiescent,
            device_work_drained=device_work_drained,
        )

    def close(self) -> None:
        self.manager.close(self.epoch)


class SpominLiveSurgeryManager:
    """Own request epochs and publish live edits only at a drained barrier."""

    def __init__(self, *, enabled: bool | None = None, history_size: int = 64):
        if history_size < 1:
            raise ValueError("live-surgery history size must be positive")
        self.enabled = live_surgery_enabled(enabled)
        self._lock = Lock()
        self._next_generation = 0
        self._epochs: dict[str, int] = {}
        self._counts: Counter[str] = Counter()
        self._recent = deque(maxlen=history_size)

    def _record(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(receipt)
        status = str(result.get("status", "unknown"))
        reason = str(result.get("reason", status))
        with self._lock:
            self._counts[status] += 1
            self._counts[f"reason:{reason}"] += 1
            self._recent.append(result)
        return result

    def decline(self, request_id: str, reason: str) -> dict[str, Any]:
        return self._record(
            {"request_id": request_id, "status": "declined", "reason": reason}
        )

    def prepare(
        self,
        *,
        request_id: str,
        prompt_token_ids: Sequence[int],
        transcript,
        capacity_tokens: int,
        strategy: str,
        has_mtp_state: bool,
        has_recurrent_state: bool,
        cache_is_request_private: bool,
    ) -> LiveSurgeryTransaction | None:
        if not self.enabled:
            self.decline(request_id, "disabled")
            return None
        if transcript is None:
            return None
        if not cache_is_request_private:
            self.decline(request_id, "cache_not_request_private")
            return None
        prompt = tuple(int(token) for token in prompt_token_ids)
        if tuple(transcript.token_ids) != prompt:
            self.decline(request_id, "transcript_prompt_mismatch")
            return None
        if has_mtp_state:
            self.decline(request_id, "mtp_state_active")
            return None
        if strategy == "lowest_importance":
            self.decline(request_id, "importance_scores_unavailable")
            return None

        state = SpominTargetState(
            revision=f"request:{request_id}:{transcript.fingerprint.digest}",
            target_tokens=len(prompt),
            transcript=transcript,
            visible_segment_ids=tuple(
                segment.segment_id for segment in transcript.segments
            ),
            has_mtp_state=False,
            has_recurrent_state=has_recurrent_state,
        )
        layer = SpominLayer(
            SpominConfig(capacity_tokens=capacity_tokens, strategy=strategy)
        )
        plan = layer.plan(state)
        if plan is None:
            self.decline(request_id, "below_pressure")
            return None
        if not plan.ready:
            self.decline(request_id, "target_unreachable")
            return None

        with self._lock:
            self._next_generation += 1
            epoch = LiveSurgeryEpoch(request_id, self._next_generation)
            self._epochs[request_id] = epoch.generation
        receipt = {
            "request_id": request_id,
            "status": "prepared",
            "reason": "pressure",
            "source_tokens": len(prompt),
            "target_tokens": plan.projected_target_tokens,
            "strategy": strategy,
            "epoch": epoch.generation,
        }
        return LiveSurgeryTransaction(self, epoch, state, plan, prompt, receipt)

    def apply(
        self,
        transaction: LiveSurgeryTransaction,
        model,
        prompt_cache: Sequence[object],
        *,
        request_quiescent: bool,
        device_work_drained: bool,
    ) -> Mapping[str, Any]:
        epoch = transaction.epoch
        with self._lock:
            current = self._epochs.get(epoch.request_id)
            if current != epoch.generation:
                return self._record_unlocked(transaction, "declined", "stale_epoch")
            if not request_quiescent:
                return self._record_unlocked(
                    transaction, "declined", "request_not_quiescent"
                )
            if not device_work_drained:
                return self._record_unlocked(
                    transaction, "declined", "device_work_not_drained"
                )
            del self._epochs[epoch.request_id]
            try:
                updated = SpominLayer(
                    SpominConfig(
                        capacity_tokens=max(transaction.state.target_tokens, 1),
                        strategy=transaction.plan.selection.strategy,
                    )
                ).apply(
                    transaction.state,
                    transaction.plan,
                    Qwen4SpominSurgeryBackend(model, prompt_cache),
                )
            except SpominCapabilityError as exc:
                return self._record_unlocked(
                    transaction, "declined", "capability_refused", detail=str(exc)
                )

            removed = set(transaction.plan.selection.segment_ids)
            retained = tuple(
                token
                for segment in transaction.state.transcript.segments
                if segment.segment_id not in removed
                for token in segment.token_ids
            )
            transaction.retained_token_ids = retained
            transaction.state = updated
            return self._record_unlocked(transaction, "applied", "committed")

    def _record_unlocked(
        self,
        transaction: LiveSurgeryTransaction,
        status: str,
        reason: str,
        *,
        detail: str | None = None,
    ) -> dict[str, Any]:
        receipt = dict(transaction.receipt, status=status, reason=reason)
        if detail is not None:
            receipt["detail"] = detail
        transaction.receipt = receipt
        self._counts[status] += 1
        self._counts[f"reason:{reason}"] += 1
        self._recent.append(receipt)
        return receipt

    def close(self, epoch: LiveSurgeryEpoch) -> None:
        with self._lock:
            if self._epochs.get(epoch.request_id) == epoch.generation:
                del self._epochs[epoch.request_id]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "active_epochs": len(self._epochs),
                "counts": dict(self._counts),
                "recent": list(self._recent),
            }


__all__ = [
    "LiveSurgeryEpoch",
    "LiveSurgeryTransaction",
    "SpominLiveSurgeryManager",
    "live_surgery_enabled",
]
