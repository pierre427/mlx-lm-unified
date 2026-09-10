"""Verify-cost-aware speculative-decoding policy (EVICT / BASTION style).

This is a *policy layer* prototype: it decides **how long a draft chain is worth
verifying** given the empirically measured, non-monotone cost of the target
verify forward on Apple-Silicon (M5). It contains **no model, no MLX arrays and
no GPU work** — it is a pure-Python cost/benefit calculator that a speculative
loop consults.

Why this exists
---------------
On a bandwidth-bound Mac the target forward is already cheap per token, so the
verify batch's *shape* — not the draft cost — is what erases the paper spec
multiplier (see ``speculative-decoding-m-series-ceiling``). The measured
``vocab_proj`` quantized-matmul curve (A-tier GPU battery §3, 2026-07-13) shows
the verify batch ``B = 1 + n_draft`` paying a **flat expensive plateau across
B = 5..14** and only recovering at **B >= 16**. EVICT (arXiv:2605.00342) and
BASTION (arXiv:2605.29727) publish the policy idea (spend verify compute only
where it pays); this module encodes *our* measured curve into that objective.

The three pieces
----------------
* ``VerifyCostModel`` — profiled per-batch verify cost from the measured curve.
* ``truncate_draft_by_benefit`` — EVICT-style greedy pre-verification trim: keep
  extending the draft only while the marginal accepted-token benefit beats the
  marginal profiled verify cost at that ``B``; cut where the objective turns
  negative.
* ``recommend_num_draft`` — a *static* measured gate (not an online controller)
  that returns the ``num_draft`` predicted net-positive, structurally refusing
  to pick a draft length whose verify batch lands in the B=5..14 tax zone.

Integration point (documented, not wired)
------------------------------------------
The natural consumer is ``hybrid_speculative._mtp_draft_verify_loop`` /
``_draft_chain``: call ``recommend_num_draft`` once per context to size the draft
cap, and ``truncate_draft_by_benefit`` on the drafted confidences before issuing
the verify forward at ``hybrid_speculative.py`` step 2 (the
``logits = model(y_verify[None], ...)`` call).

**Fable finding L1 — dtype discipline at the seam.** This module deliberately
operates on scalar confidences and a cost table; it never touches logits. The
integration point MUST keep logprob work in the model dtype and **slice the
verify rows before any upcast** — exactly the existing
``rel = logits[0, -(n_prop + 1):, :]`` *then* ``logsumexp`` order in
``_mtp_draft_verify_loop``. Do **not** materialise the full ``(k+1) x V`` logits
tensor in fp32 per cycle: that upcast is the cost this policy is trying to spend
sparingly, and doing it eagerly reintroduces the very seam the plateau describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Provenance-carrying measured curve.
# ---------------------------------------------------------------------------
#
# Source: A-tier GPU battery §3 (wiki/docs/experiments/a-tier-gpu-battery-2026-07-13.md).
# Shape: `vocab_proj` (4096 x 151936) q4 quantized-matmul — the exact verify-qmm
# shape behind mlx #3791. Values below are TOTAL microseconds for a batch of `B`
# rows, averaged over the recorded cold/hot columns (they agree to <1%):
#
#   B      |  4   |  5   |  6   |  7   |  8   |  10  |  12  |  14  |  16
#   cold   | 825  | 960  | 1253 | 1481 | 1494 | 1735 | 2152 | 2531 | 1773
#   hot    | 838  | 958  | 1243 | 1477 | 1508 | 1709 | 2157 | 2546 | 1800
#
# Per-token (total / B) plateaus at ~180-210 us across B=5..14 instead of
# falling, then recovers to ~112 us at B=16. Because the verify batch is
# B = 1 + n_draft, draft lengths k = 4..13 land squarely in this tax band.
#
# Small-B (< 4) values were directly measured on this M5 (mlx 0.32.0) in the
# 2026-07-15 GPU-validation battery (T4) — see below; they revealed a fixed
# ~800 us kernel-launch floor, so B=1 sits far above the old extrapolated ramp.
MEASURED_VERIFY_US: Dict[int, float] = {
    # --- directly measured (A-tier battery §3, cold/hot averaged) ---
    4: 831.5,
    5: 959.0,
    6: 1248.0,
    7: 1479.0,
    8: 1501.0,
    10: 1722.0,
    12: 2154.5,
    14: 2538.5,
    16: 1786.5,
}

# Small-B (B < 4) totals, DIRECTLY MEASURED on M5 (mlx 0.32.0) in the
# 2026-07-15 GPU-validation battery (T4): a fixed ~800 us kernel-launch floor
# dominates below B=4, so per-token cost is still highest at B=1 (~807 us/tok)
# but the total is near-flat, not a linear ramp. Supersedes the earlier
# shape-calibrated 250/450/650 extrapolation (B=1 was 3.2x too low).
_MEASURED_SMALL_B_US: Dict[int, float] = {
    1: 807.0,   # 807 us/tok — launch-floor dominated (measured 806.8)
    2: 812.0,   # 406 us/tok
    3: 829.0,   # 276 us/tok (measured 829.4)
}

# The measured tax plateau, in verify-batch terms: per-token cost is flat/high
# and does not fall. Draft length k maps to B = k + 1, so k in [4, 13] is taxed.
TAX_ZONE_B_LO = 5
TAX_ZONE_B_HI = 14

# Batch at which per-token cost recovers (B >= this is "cheap again").
RECOVERY_B = 16


@dataclass(frozen=True)
class VerifyCostModel:
    """Profiled verify cost as a function of the verify batch ``B = 1 + k``.

    The model interpolates the measured ``MEASURED_VERIFY_US`` curve for batches
    between recorded points and extrapolates the small-``B`` ramp. It exposes
    both total and per-token costs plus the *marginal* cost of adding one more
    draft token — the quantity the EVICT-style truncation trades against benefit.

    All costs are in microseconds and carry the provenance of
    ``MEASURED_VERIFY_US`` (A-tier battery §3, ``vocab_proj`` q4).
    """

    # Merged, sorted table of (B -> total us). Frozen at construction.
    _table: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_measured(cls) -> "VerifyCostModel":
        table = dict(_MEASURED_SMALL_B_US)
        table.update(MEASURED_VERIFY_US)
        return cls(_table=dict(sorted(table.items())))

    def __post_init__(self) -> None:
        if not self._table:
            # Allow VerifyCostModel() to mean "the measured curve".
            merged = dict(_MEASURED_SMALL_B_US)
            merged.update(MEASURED_VERIFY_US)
            object.__setattr__(self, "_table", dict(sorted(merged.items())))

    # -- core lookups --------------------------------------------------------

    def total_verify_us(self, B: int) -> float:
        """Total verify cost (us) for a batch of ``B`` rows.

        Piecewise-linear interpolation between measured/extrapolated knots;
        clamps below the smallest and above the largest knot.
        """
        if B < 1:
            raise ValueError("verify batch B must be >= 1")
        knots = sorted(self._table)
        if B in self._table:
            return self._table[B]
        if B <= knots[0]:
            return self._table[knots[0]]
        if B >= knots[-1]:
            return self._table[knots[-1]]
        # find bracketing knots
        lo = max(k for k in knots if k < B)
        hi = min(k for k in knots if k > B)
        y0, y1 = self._table[lo], self._table[hi]
        return y0 + (y1 - y0) * (B - lo) / (hi - lo)

    def per_token_verify_us(self, B: int) -> float:
        """Total verify cost amortised over the ``B`` rows in the batch."""
        return self.total_verify_us(B) / B

    def verify_batch(self, num_draft: int) -> int:
        """Verify batch for a draft chain of length ``k``: ``B = k + 1``."""
        if num_draft < 0:
            raise ValueError("num_draft must be >= 0")
        return num_draft + 1

    def total_us_for_draft(self, num_draft: int) -> float:
        """Total verify cost for a draft chain of length ``k`` (batch ``k+1``)."""
        return self.total_verify_us(self.verify_batch(num_draft))

    def marginal_verify_us(self, num_draft: int) -> float:
        """Cost of *adding* the ``num_draft``-th draft token.

        Extending the chain from ``k-1`` to ``k`` grows the verify batch from
        ``k`` to ``k+1`` rows, so the marginal cost is
        ``total_verify_us(k+1) - total_verify_us(k)``. On the plateau this is a
        large positive number; across the B=15->16 recovery it can be *negative*
        (the batch got cheaper), which is exactly the "jump past the tax" signal.
        """
        if num_draft < 1:
            raise ValueError("num_draft must be >= 1 to have a marginal cost")
        return self.total_verify_us(num_draft + 1) - self.total_verify_us(num_draft)

    def in_tax_zone(self, num_draft: int) -> bool:
        """True if a draft length ``k`` verifies inside the B=5..14 plateau."""
        B = self.verify_batch(num_draft)
        return TAX_ZONE_B_LO <= B <= TAX_ZONE_B_HI

    def token_value_us(self) -> float:
        """Shape-consistent value of one committed token.

        A committed token saves one plain (B=1) target forward; we use the
        profiled ``total_verify_us(1)`` on the same ``vocab_proj`` shape as the
        proxy so benefit and cost live in one measured unit. In production this
        should be the measured *full* target decode-step time (larger, which only
        makes speculation more attractive); the inflection structure is what this
        model preserves.
        """
        return self.total_verify_us(1)


# ---------------------------------------------------------------------------
# Acceptance model.
# ---------------------------------------------------------------------------

def default_accept_prob_fn(confidence: float) -> float:
    """Map a draft-model confidence (its top-token prob) to an accept prob.

    Greedy target verification accepts iff the target's argmax equals the
    draft's proposed token; the draft's own top-token probability is a
    well-calibrated proxy for that agreement, so the identity (clamped to a
    valid probability) is the honest default. Swap in a fitted calibration when
    one is measured.
    """
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return float(confidence)


# ---------------------------------------------------------------------------
# EVICT-style pre-verification truncation.
# ---------------------------------------------------------------------------

@dataclass
class _MarginalStep:
    """One position's marginal objective terms (for inspection / tests)."""

    position: int          # 1-based draft position i
    verify_batch: int      # B = i + 1
    cum_accept_prob: float  # prod_{j<=i} accept_prob(conf_j)
    marginal_benefit_us: float
    marginal_cost_us: float

    @property
    def marginal_net_us(self) -> float:
        return self.marginal_benefit_us - self.marginal_cost_us


@dataclass
class TruncationResult:
    """Result of ``truncate_draft_by_benefit``."""

    length: int
    steps: List[_MarginalStep]

    def __int__(self) -> int:  # let callers use it directly as a length
        return self.length


def marginal_objective(
    draft_confidences: Sequence[float],
    verify_cost_model: VerifyCostModel,
    accept_prob_fn: Optional[Callable[[float], float]] = None,
    token_value_us: Optional[float] = None,
) -> List[_MarginalStep]:
    """Per-position marginal benefit vs marginal profiled verify cost.

    For draft position ``i`` (1-based), the marginal *benefit* of including it is
    ``P(all of 1..i accepted) * token_value`` — the expected extra committed
    token it buys — and the marginal *cost* is the profiled
    ``marginal_verify_us(i)`` of growing the verify batch by one row. Returned as
    a list so callers/tests can inspect the whole objective trace.
    """
    accept_prob_fn = accept_prob_fn or default_accept_prob_fn
    tv = verify_cost_model.token_value_us() if token_value_us is None else token_value_us
    steps: List[_MarginalStep] = []
    cum = 1.0
    for i, conf in enumerate(draft_confidences, start=1):
        cum *= accept_prob_fn(conf)
        steps.append(
            _MarginalStep(
                position=i,
                verify_batch=verify_cost_model.verify_batch(i),
                cum_accept_prob=cum,
                marginal_benefit_us=cum * tv,
                marginal_cost_us=verify_cost_model.marginal_verify_us(i),
            )
        )
    return steps


def truncate_draft_by_benefit(
    draft_confidences: Sequence[float],
    verify_cost_model: VerifyCostModel,
    accept_prob_fn: Optional[Callable[[float], float]] = None,
    token_value_us: Optional[float] = None,
) -> TruncationResult:
    """EVICT-style greedy pre-verification truncation.

    Walk the drafted chain position by position. Keep extending while the
    marginal accepted-token benefit at least covers the marginal profiled verify
    cost of enlarging the batch by that row; **cut at the first position where
    the marginal objective turns negative**. This spends verify compute only
    where it pays on the measured curve: cheap early rows are kept, and the
    steep B=5..14 plateau is where a low-acceptance chain gets cut.

    Returns a :class:`TruncationResult` (usable directly as an ``int`` length),
    with the full per-step objective attached for inspection.
    """
    steps = marginal_objective(
        draft_confidences, verify_cost_model, accept_prob_fn, token_value_us
    )
    kept = 0
    for step in steps:
        if step.marginal_net_us >= 0.0:
            kept += 1
        else:
            break
    return TruncationResult(length=kept, steps=steps)


# ---------------------------------------------------------------------------
# Static measured gate: recommend num_draft.
# ---------------------------------------------------------------------------

def _net_saving_us(
    num_draft: int,
    accept_prob: float,
    verify_cost_model: VerifyCostModel,
    token_value_us: float,
) -> float:
    """Whole-cycle net saving of drafting ``k`` tokens vs plain decode.

    Under a constant per-token accept probability ``p`` the expected accepted
    length of a length-``k`` chain is ``sum_{i=1..k} p**i``; the cycle commits
    ``1 + that`` tokens for the price of ONE verify forward of batch ``k+1``.
    Plain decode would pay ``token_value`` per committed token, so::

        net(k) = (1 + E[accepted]) * token_value - total_verify_us(k+1)
    """
    e_accepted = 0.0
    term = 1.0
    for _ in range(num_draft):
        term *= accept_prob
        e_accepted += term
    committed = 1.0 + e_accepted
    return committed * token_value_us - verify_cost_model.total_us_for_draft(num_draft)


def recommend_num_draft(context_features: Dict[str, object]) -> int:
    """Static measured gate: the ``num_draft`` the cost model predicts pays off.

    This is a *gate*, not an online controller — a pure function of the supplied
    features. It maximises the whole-cycle net saving over an **efficient**
    candidate set that structurally **excludes the B=5..14 tax zone**: draft
    lengths ``k in {1..4}`` (cheap, pre-plateau) and ``k in {15..max_draft}``
    (past the B>=16 recovery). Because the plateau is never a candidate, the gate
    returns ``k <= 4`` under low acceptance and only jumps to ``k >= 15`` when
    acceptance is high enough to amortise the recovered batch.

    ``context_features`` keys:
      * ``accept_prob`` (float, required) — expected per-token draft acceptance.
      * ``max_draft`` (int, optional, default 16) — hard cap on ``k``.
      * ``token_value_us`` (float, optional) — value of a committed token;
        defaults to the model's shape-consistent ``token_value_us()``.
    """
    if "accept_prob" not in context_features:
        raise KeyError("context_features must include 'accept_prob'")
    accept_prob = float(context_features["accept_prob"])  # type: ignore[arg-type]
    if not (0.0 <= accept_prob <= 1.0):
        raise ValueError("accept_prob must be in [0, 1]")
    max_draft = int(context_features.get("max_draft", RECOVERY_B))  # type: ignore[arg-type]
    if max_draft < 1:
        raise ValueError("max_draft must be >= 1")

    model = VerifyCostModel.from_measured()
    token_value = context_features.get("token_value_us")
    tv = model.token_value_us() if token_value is None else float(token_value)  # type: ignore[arg-type]

    # Efficient candidate set: cheap pre-plateau lengths, plus past-recovery
    # lengths. k with B in [5, 14] (i.e. k in [4, 13]) is the tax zone and is
    # deliberately never offered to the gate. The pre-plateau band is therefore
    # k <= 3 (B <= 4) — the first taxed length is k=4 (B=5).
    max_cheap_k = TAX_ZONE_B_LO - 2  # B = k+1 <= 4  =>  k <= 3
    cheap = [k for k in range(1, min(max_cheap_k, max_draft) + 1)]
    recovered = [k for k in range(RECOVERY_B - 1, max_draft + 1)]  # k>=15 => B>=16
    candidates = cheap + recovered
    if not candidates:
        candidates = [1]

    best_k = candidates[0]
    best_net = _net_saving_us(best_k, accept_prob, model, tv)
    for k in candidates[1:]:
        net = _net_saving_us(k, accept_prob, model, tv)
        # Strictly greater so the smallest/cheapest k wins ties (prefer less
        # verify pressure when the objective is flat).
        if net > best_net:
            best_net = net
            best_k = k
    return best_k
