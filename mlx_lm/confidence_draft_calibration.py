"""Offline calibration and cost screening for confidence-gated drafts.

This module deliberately has no inference-loop integration.  It consumes
hosted scalar traces after the fact and asks whether shortening a speculative
verify window could pay.  Every selected proposal position remains covered by
the target verify window; confidence never authorizes a token.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

from .verify_cost_policy import VerifyCostModel


@dataclass(frozen=True)
class DraftConfidenceTrace:
    """One completed draft/verify cycle represented by hosted scalars."""

    probability: tuple[float, ...]
    margin: tuple[float, ...]
    entropy: tuple[float, ...]
    accepted_length: int

    def __post_init__(self) -> None:
        depth = len(self.probability)
        if len(self.margin) != depth or len(self.entropy) != depth:
            raise ValueError("confidence vectors must have the same length")
        if not 0 <= self.accepted_length <= depth:
            raise ValueError("accepted_length must be within the draft depth")
        for name, values in (
            ("probability", self.probability),
            ("margin", self.margin),
            ("entropy", self.entropy),
        ):
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"{name} values must be finite")
        if any(not 0.0 <= value <= 1.0 for value in self.probability):
            raise ValueError("probability values must be in [0, 1]")
        if any(value < 0.0 for value in self.margin):
            raise ValueError("margin values must be non-negative")
        if any(not 0.0 <= value <= 1.0 for value in self.entropy):
            raise ValueError("normalized entropy values must be in [0, 1]")

    @property
    def depth(self) -> int:
        return len(self.probability)

    @classmethod
    def from_mapping(cls, item: dict) -> "DraftConfidenceTrace":
        return cls(
            probability=tuple(float(value) for value in item["probability"]),
            margin=tuple(float(value) for value in item["margin"]),
            entropy=tuple(float(value) for value in item["entropy"]),
            accepted_length=int(item["accepted_length"]),
        )


@dataclass(frozen=True)
class ConfidenceGate:
    """Default-off fixed confidence gate.

    ``enabled=False`` returns the entire proposal.  When enabled, it returns
    the longest consecutive prefix whose prefix score clears ``threshold``.
    This object only chooses target verification effort.
    """

    metric: str = "probability"
    threshold: float = 0.0
    enabled: bool = False

    def select(self, trace: DraftConfidenceTrace) -> int:
        if not self.enabled:
            return trace.depth
        kept = 0
        for score in prefix_scores(trace, self.metric):
            if score < self.threshold:
                break
            kept += 1
        return kept


@dataclass(frozen=True)
class PolicyEvaluation:
    cycles: int
    committed_tokens: int
    target_verified_rows: int
    accepted_tokens_discarded: int
    verification_rows_saved: int
    target_verify_us: float
    total_cost_us: float
    token_rate_per_us: float
    unverified_committed_tokens: int = 0


@dataclass(frozen=True)
class AggregateOracleBound:
    cycles: int
    proposed: int
    accepted: int
    min_saving_us: float
    max_saving_us: float
    min_saving_fraction: float
    max_saving_fraction: float


def prefix_scores(trace: DraftConfidenceTrace, metric: str) -> tuple[float, ...]:
    """Return higher-is-better scores for every proposal prefix."""
    if metric not in {"probability", "margin", "entropy", "joint_probability"}:
        raise ValueError(f"unsupported confidence metric: {metric}")
    scores = []
    running = 1.0
    running_min = math.inf
    running_max = 0.0
    for probability, margin, entropy in zip(
        trace.probability, trace.margin, trace.entropy
    ):
        if metric == "probability":
            running_min = min(running_min, probability)
            score = running_min
        elif metric == "margin":
            running_min = min(running_min, margin)
            score = running_min
        elif metric == "entropy":
            running_max = max(running_max, entropy)
            score = 1.0 - running_max
        else:
            running *= probability
            score = running
        scores.append(float(score))
    return tuple(scores)


class IsotonicCalibrator:
    """Small dependency-free monotone calibrator for offline trace analysis."""

    def __init__(self, upper_bounds: Sequence[float], probabilities: Sequence[float]):
        if not upper_bounds or len(upper_bounds) != len(probabilities):
            raise ValueError("calibrator requires aligned non-empty blocks")
        self.upper_bounds = tuple(float(value) for value in upper_bounds)
        self.probabilities = tuple(float(value) for value in probabilities)

    @classmethod
    def fit(cls, scores: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        if not scores or len(scores) != len(labels):
            raise ValueError("fit requires aligned non-empty samples")
        grouped: list[list[float]] = []
        for score, label in sorted(zip(scores, labels)):
            if label not in (0, 1):
                raise ValueError("calibration labels must be binary")
            if grouped and score == grouped[-1][0]:
                grouped[-1][1] += label
                grouped[-1][2] += 1
            else:
                grouped.append([float(score), float(label), 1.0])

        blocks: list[list[float]] = []
        for upper, positives, count in grouped:
            blocks.append([upper, positives, count])
            while len(blocks) >= 2:
                left, right = blocks[-2], blocks[-1]
                if left[1] / left[2] <= right[1] / right[2]:
                    break
                blocks[-2:] = [
                    [right[0], left[1] + right[1], left[2] + right[2]]
                ]
        return cls(
            [block[0] for block in blocks],
            [block[1] / block[2] for block in blocks],
        )

    def predict(self, score: float) -> float:
        index = min(bisect_left(self.upper_bounds, float(score)), len(self.upper_bounds) - 1)
        return self.probabilities[index]


def calibration_samples(
    traces: Iterable[DraftConfidenceTrace], metric: str
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    for trace in traces:
        for position, score in enumerate(prefix_scores(trace, metric), start=1):
            scores.append(score)
            labels.append(int(trace.accepted_length >= position))
    return scores, labels


def calibration_metrics(
    scores: Sequence[float], labels: Sequence[int], calibrator: IsotonicCalibrator
) -> dict[str, float]:
    predictions = [calibrator.predict(score) for score in scores]
    count = len(labels)
    brier = sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels)) / count
    # Equal-width ECE is sufficient for this screening harness; the raw sample
    # is also retained by callers for more elaborate follow-up analysis.
    ece = 0.0
    for bucket in range(10):
        lo, hi = bucket / 10.0, (bucket + 1) / 10.0
        members = [
            index
            for index, prediction in enumerate(predictions)
            if lo <= prediction < hi or (bucket == 9 and prediction == 1.0)
        ]
        if members:
            p_mean = sum(predictions[index] for index in members) / len(members)
            y_mean = sum(labels[index] for index in members) / len(members)
            ece += len(members) / count * abs(p_mean - y_mean)
    return {"brier": brier, "ece": ece}


def choose_expected_value_prefix(
    trace: DraftConfidenceTrace,
    calibrator: IsotonicCalibrator,
    *,
    metric: str,
    verify_cost_model: VerifyCostModel,
    token_value_us: float | None = None,
) -> int:
    """Choose target width by calibrated expected committed work.

    The selected proposal length ``k`` always implies a target verify batch of
    ``k + 1`` rows.  Zero means a normal one-row target step, never accepting a
    draft token on confidence alone.
    """
    token_value = (
        verify_cost_model.token_value_us()
        if token_value_us is None
        else float(token_value_us)
    )
    prefix_probabilities = []
    previous = 1.0
    for score in prefix_scores(trace, metric):
        previous = min(previous, calibrator.predict(score))
        prefix_probabilities.append(previous)
    best_k = 0
    best_value = token_value - verify_cost_model.total_us_for_draft(0)
    expected_accepted = 0.0
    for k, probability in enumerate(prefix_probabilities, start=1):
        expected_accepted += probability
        value = (
            (1.0 + expected_accepted) * token_value
            - verify_cost_model.total_us_for_draft(k)
        )
        if value > best_value:
            best_k, best_value = k, value
    return best_k


def evaluate_policy(
    traces: Sequence[DraftConfidenceTrace],
    selector: Callable[[DraftConfidenceTrace], int],
    *,
    verify_cost_model: VerifyCostModel | None = None,
    draft_cycle_us: float = 0.0,
    router_us: float = 0.0,
) -> PolicyEvaluation:
    """Evaluate scheduling work; authoritative target verification is invariant."""
    model = verify_cost_model or VerifyCostModel.from_measured()
    committed = verified_rows = discarded = rows_saved = 0
    verify_us = total_us = 0.0
    for trace in traces:
        selected = int(selector(trace))
        if not 0 <= selected <= trace.depth:
            raise ValueError("selector returned a prefix outside the proposal")
        accepted = min(trace.accepted_length, selected)
        committed += accepted + 1  # accepted prefix plus target-produced bonus
        verified_rows += selected + 1
        discarded += trace.accepted_length - accepted
        rows_saved += trace.depth - selected
        cost = model.total_us_for_draft(selected)
        verify_us += cost
        total_us += cost + float(draft_cycle_us) + float(router_us)
    return PolicyEvaluation(
        cycles=len(traces),
        committed_tokens=committed,
        target_verified_rows=verified_rows,
        accepted_tokens_discarded=discarded,
        verification_rows_saved=rows_saved,
        target_verify_us=verify_us,
        total_cost_us=total_us,
        token_rate_per_us=committed / total_us if total_us else 0.0,
        unverified_committed_tokens=0,
    )


def aggregate_oracle_bound(
    *,
    proposed: int,
    accepted: int,
    max_draft: int,
    verify_cost_model: VerifyCostModel | None = None,
) -> AggregateOracleBound:
    """Bound a perfect confidence gate using aggregate prefix-acceptance counts.

    The calculation enumerates all accepted-length histograms compatible with
    ``proposed`` and ``accepted``.  The oracle chooses ``k = accepted_length``:
    the shortest verify window that preserves the cycle's committed progress.
    """
    if max_draft < 1 or proposed < 0 or accepted < 0 or accepted > proposed:
        raise ValueError("invalid aggregate speculative counters")
    if proposed % max_draft:
        raise ValueError("proposed must describe fixed-width complete cycles")
    cycles = proposed // max_draft
    model = verify_cost_model or VerifyCostModel.from_measured()
    full_cost = model.total_us_for_draft(max_draft)
    savings_by_accept = [
        full_cost - model.total_us_for_draft(length)
        for length in range(max_draft + 1)
    ]

    possible = {(0, 0): (0.0, 0.0)}
    for _ in range(cycles):
        updated: dict[tuple[int, int], tuple[float, float]] = {}
        for (used, total_accept), (minimum, maximum) in possible.items():
            for length, saving in enumerate(savings_by_accept):
                key = (used + 1, total_accept + length)
                candidate = (minimum + saving, maximum + saving)
                if key in updated:
                    old_min, old_max = updated[key]
                    updated[key] = (min(old_min, candidate[0]), max(old_max, candidate[1]))
                else:
                    updated[key] = candidate
        possible = updated
    key = (cycles, accepted)
    if key not in possible:
        raise ValueError("aggregate counters have no valid prefix histogram")
    minimum, maximum = possible[key]
    baseline = cycles * full_cost
    return AggregateOracleBound(
        cycles=cycles,
        proposed=proposed,
        accepted=accepted,
        min_saving_us=minimum,
        max_saving_us=maximum,
        min_saving_fraction=minimum / baseline if baseline else 0.0,
        max_saving_fraction=maximum / baseline if baseline else 0.0,
    )
