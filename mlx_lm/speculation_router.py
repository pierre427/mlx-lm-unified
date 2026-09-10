"""Pure-Python request-local router for measured speculative decoding.

The router composes the static verify-cost recommendation with marginal
pre-verification truncation, then adds an abstaining hysteretic controller.
It never touches model state: ``num_draft=0`` means take the plain decode floor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .verify_cost_policy import (
    VerifyCostModel,
    recommend_num_draft,
    truncate_draft_by_benefit,
)


@dataclass(frozen=True)
class SpeculationDecision:
    num_draft: int
    reason: str
    accept_prob: float
    latched_plain: bool
    cooldown_remaining: int


class RoutedSpeculationPolicy:
    """Hysteretic measured router with a periodic one-token re-probe."""

    def __init__(
        self,
        *,
        max_draft: int = 16,
        initial_accept_prob: float = 0.5,
        ewma_alpha: float = 0.25,
        min_proposals: int = 8,
        disable_below: float = 0.35,
        bad_cycle_patience: int = 2,
        plain_cooldown_cycles: int = 8,
        token_value_us: Optional[float] = None,
    ):
        if max_draft < 1:
            raise ValueError("max_draft must be positive")
        if not 0.0 <= initial_accept_prob <= 1.0:
            raise ValueError("initial_accept_prob must be in [0, 1]")
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if min_proposals < 1 or bad_cycle_patience < 1:
            raise ValueError("sample and patience counts must be positive")
        if plain_cooldown_cycles < 1:
            raise ValueError("plain_cooldown_cycles must be positive")
        self.max_draft = int(max_draft)
        self.accept_prob = float(initial_accept_prob)
        self.ewma_alpha = float(ewma_alpha)
        self.min_proposals = int(min_proposals)
        self.disable_below = float(disable_below)
        self.bad_cycle_patience = int(bad_cycle_patience)
        self.plain_cooldown_cycles = int(plain_cooldown_cycles)
        self.token_value_us = token_value_us
        self.verify_cost_model = VerifyCostModel.from_measured()
        self.total_proposed = 0
        self.total_accepted = 0
        self.bad_cycles = 0
        self.latched_plain = False
        self.cooldown_remaining = 0
        self.decisions = 0
        self.plain_decisions = 0
        self.reengagements = 0
        self.last_decision = SpeculationDecision(
            0, "not_started", self.accept_prob, False, 0
        )

    def _decision(self, num_draft: int, reason: str) -> SpeculationDecision:
        self.decisions += 1
        if num_draft == 0:
            self.plain_decisions += 1
        self.last_decision = SpeculationDecision(
            num_draft=num_draft,
            reason=reason,
            accept_prob=self.accept_prob,
            latched_plain=self.latched_plain,
            cooldown_remaining=self.cooldown_remaining,
        )
        return self.last_decision

    def decide(
        self,
        *,
        max_draft: Optional[int] = None,
        remaining: Optional[int] = None,
    ) -> SpeculationDecision:
        requested_cap = self.max_draft if max_draft is None else int(max_draft)
        cap = min(self.max_draft, requested_cap)
        if remaining is not None:
            cap = min(cap, max(0, int(remaining)))
        if cap <= 0:
            return self._decision(0, "no_remaining_budget")
        if self.latched_plain:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                return self._decision(0, "plain_cooldown")
            self.latched_plain = False
            self.bad_cycles = 0
            self.reengagements += 1
            return self._decision(1, "periodic_reprobe")

        recommended = recommend_num_draft(
            {
                "accept_prob": self.accept_prob,
                "max_draft": cap,
                **(
                    {"token_value_us": self.token_value_us}
                    if self.token_value_us is not None
                    else {}
                ),
            }
        )
        trimmed = truncate_draft_by_benefit(
            [self.accept_prob] * recommended,
            self.verify_cost_model,
            accept_prob_fn=lambda probability: probability,
            token_value_us=self.token_value_us,
        )
        selected = min(cap, int(trimmed))
        if selected <= 0:
            self.latched_plain = True
            self.cooldown_remaining = self.plain_cooldown_cycles
            return self._decision(0, "marginal_verify_cost")
        return self._decision(selected, "measured_verify_cost")

    def observe(self, proposed: int, accepted: int) -> None:
        proposed, accepted = int(proposed), int(accepted)
        if proposed <= 0 or not 0 <= accepted <= proposed:
            raise ValueError("require proposed > 0 and 0 <= accepted <= proposed")
        observed = accepted / proposed
        self.accept_prob = (
            self.ewma_alpha * observed
            + (1.0 - self.ewma_alpha) * self.accept_prob
        )
        self.total_proposed += proposed
        self.total_accepted += accepted
        if (
            self.total_proposed >= self.min_proposals
            and self.accept_prob < self.disable_below
        ):
            self.bad_cycles += 1
        else:
            self.bad_cycles = 0
        if self.bad_cycles >= self.bad_cycle_patience:
            self.latched_plain = True
            self.cooldown_remaining = self.plain_cooldown_cycles

    def snapshot(self) -> Dict[str, Any]:
        return {
            "accept_prob": round(self.accept_prob, 6),
            "total_proposed": self.total_proposed,
            "total_accepted": self.total_accepted,
            "decisions": self.decisions,
            "plain_decisions": self.plain_decisions,
            "reengagements": self.reengagements,
            "last_decision": asdict(self.last_decision),
        }


class DepthCeilingController:
    """Adaptive draft-depth ceiling: the caller's depth is a floor, not a fix.

    mlx-vlm PR #2046 semantics: start at the native depth (``num_draft``, the
    trained regime), expand toward ``ceiling`` only after the full native
    prefix has been accepted in at least ``expand_threshold`` of the last
    ``window`` draft rounds, and back off when that fraction falls below
    ``backoff_threshold``. The window is cleared after every depth change, so
    each step needs ``window`` fresh rounds of evidence (hysteretic dwell —
    no per-cycle thrashing).

    Unlike :class:`RoutedSpeculationPolicy` this controller never returns 0
    and never latches plain: the measured rate gate in the MTP loop keeps
    sole authority over WHETHER to speculate; this controller only decides
    HOW DEEP. ``decide`` returns a depth in ``[1, ceiling]``: normally
    within ``[floor, ceiling]``, below the floor only when the caller passes
    a harder external cap (``max_draft``/``remaining``) below it — the
    harder cap wins but the result stays positive. A nonpositive cap is a
    caller bug and raises ``ValueError`` (the driving loop only consults the
    controller while budget remains). It is request-local state — create one
    per admitted request and never persist it into APC sidecars.

    Duck-type compatible with the ``speculation_router=`` seam of
    ``self_mtp_generate_step``: ``decide``/``observe``/``accept_prob``/
    ``reengagements``.
    """

    def __init__(
        self,
        num_draft: int,
        ceiling: int,
        *,
        window: int = 8,
        expand_threshold: float = 0.65,
        backoff_threshold: float = 0.50,
    ):
        if num_draft < 1:
            raise ValueError("num_draft (the floor depth) must be >= 1")
        if ceiling < num_draft:
            raise ValueError(
                f"ceiling {ceiling} must be >= the floor depth {num_draft}"
            )
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 0.0 < backoff_threshold <= expand_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < backoff <= expand <= 1"
            )
        self.floor = int(num_draft)
        self.ceiling = int(ceiling)
        self.depth = self.floor
        self.window = int(window)
        self.expand_threshold = float(expand_threshold)
        self.backoff_threshold = float(backoff_threshold)
        self._full_rounds: deque = deque(maxlen=self.window)
        self.accept_prob = 0.0
        self.total_proposed = 0
        self.total_accepted = 0
        self.expansions = 0
        self.backoffs = 0
        self.decisions = 0
        # Seam compatibility: this controller never latches plain, so it
        # never re-engages either.
        self.reengagements = 0
        self.last_decision = SpeculationDecision(
            0, "not_started", self.accept_prob, False, 0
        )

    def _decision(self, num_draft: int, reason: str) -> SpeculationDecision:
        self.decisions += 1
        self.last_decision = SpeculationDecision(
            num_draft=num_draft,
            reason=reason,
            accept_prob=self.accept_prob,
            latched_plain=False,
            cooldown_remaining=0,
        )
        return self.last_decision

    def decide(
        self,
        *,
        max_draft: Optional[int] = None,
        remaining: Optional[int] = None,
    ) -> SpeculationDecision:
        cap = self.ceiling if max_draft is None else min(self.ceiling, int(max_draft))
        if remaining is not None:
            cap = min(cap, int(remaining))
        if cap <= 0:
            # Never 0: plain-vs-spec is the rate gate's axis, not ours. The
            # loop only consults the controller while budget remains, so a
            # nonpositive cap is a caller bug — fail loud.
            raise ValueError(
                f"decide() needs a positive draft cap; got {cap} "
                f"(max_draft={max_draft!r}, remaining={remaining!r})"
            )
        return self._decision(min(self.depth, cap), "depth_ceiling")

    def observe(self, proposed: int, accepted: int) -> None:
        proposed, accepted = int(proposed), int(accepted)
        if proposed <= 0 or not 0 <= accepted <= proposed:
            raise ValueError("require proposed > 0 and 0 <= accepted <= proposed")
        self.total_proposed += proposed
        self.total_accepted += accepted
        if proposed < self.floor:
            # A budget-truncated round never tested the FULL native prefix:
            # it is evidence of nothing, so it must not enter the window in
            # either direction (it still counts in the raw totals above).
            return
        self._full_rounds.append(accepted >= self.floor)
        rate = sum(self._full_rounds) / len(self._full_rounds)
        self.accept_prob = rate
        if len(self._full_rounds) < self.window:
            return
        if rate >= self.expand_threshold and self.depth < self.ceiling:
            self.depth += 1
            self.expansions += 1
            self._full_rounds.clear()
        elif rate < self.backoff_threshold and self.depth > self.floor:
            self.depth -= 1
            self.backoffs += 1
            self._full_rounds.clear()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "floor": self.floor,
            "ceiling": self.ceiling,
            "depth": self.depth,
            "accept_prob": round(self.accept_prob, 6),
            "total_proposed": self.total_proposed,
            "total_accepted": self.total_accepted,
            "expansions": self.expansions,
            "backoffs": self.backoffs,
            "decisions": self.decisions,
            "last_decision": asdict(self.last_decision),
        }
