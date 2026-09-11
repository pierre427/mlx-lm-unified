#!/usr/bin/env python3
"""Offline calibration/cost screen for confidence-gated speculative width."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path

from mlx_lm.confidence_draft_calibration import (
    ConfidenceGate,
    DraftConfidenceTrace,
    IsotonicCalibrator,
    aggregate_oracle_bound,
    calibration_metrics,
    calibration_samples,
    choose_expected_value_prefix,
    evaluate_policy,
    prefix_scores,
)
from mlx_lm.verify_cost_policy import VerifyCostModel


METRICS = ("probability", "joint_probability", "margin", "entropy")


def synthetic_traces(count: int, depth: int, seed: int) -> list[DraftConfidenceTrace]:
    rng = random.Random(seed)
    traces = []
    for _ in range(count):
        quality = rng.betavariate(3.0, 2.2)
        probabilities = []
        margins = []
        entropies = []
        accepted = 0
        prefix_alive = True
        for position in range(depth):
            probability = min(
                0.995,
                max(0.05, 0.34 + 0.64 * quality - 0.055 * position + rng.gauss(0.0, 0.07)),
            )
            margin = max(0.0, probability - (1.0 - probability) * rng.uniform(0.3, 1.2))
            entropy = min(1.0, max(0.0, 1.04 - probability + rng.gauss(0.0, 0.06)))
            probabilities.append(probability)
            margins.append(margin)
            entropies.append(entropy)
            agreement_probability = min(
                0.995,
                max(0.01, 0.08 + 0.84 * probability + 0.08 * margin - 0.08 * entropy),
            )
            if prefix_alive and rng.random() < agreement_probability:
                accepted += 1
            else:
                prefix_alive = False
        traces.append(
            DraftConfidenceTrace(
                tuple(probabilities), tuple(margins), tuple(entropies), accepted
            )
        )
    return traces


def read_jsonl(path: Path) -> list[DraftConfidenceTrace]:
    traces = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    traces.append(DraftConfidenceTrace.from_mapping(json.loads(line)))
                except Exception as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    return traces


def tune_fixed_gate(
    traces: list[DraftConfidenceTrace],
    metric: str,
    model: VerifyCostModel,
    *,
    draft_cycle_us: float,
    router_us: float,
) -> tuple[ConfidenceGate, dict]:
    scores = sorted({score for trace in traces for score in prefix_scores(trace, metric)})
    # This is an offline screening tool, not a threshold optimizer.  Bound the
    # grid so arbitrary floating-point traces cannot turn it quadratic.
    if len(scores) > 256:
        scores = [
            scores[round(index * (len(scores) - 1) / 255)]
            for index in range(256)
        ]
        scores = sorted(set(scores))
    candidates = [scores[0] - 1e-12] + scores
    best = None
    for threshold in candidates:
        gate = ConfidenceGate(metric=metric, threshold=threshold, enabled=True)
        evaluation = evaluate_policy(
            traces,
            gate.select,
            verify_cost_model=model,
            draft_cycle_us=draft_cycle_us,
            router_us=router_us,
        )
        key = (evaluation.token_rate_per_us, -evaluation.target_verified_rows, threshold)
        if best is None or key > best[0]:
            best = (key, gate, evaluation)
    assert best is not None
    return best[1], asdict(best[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-jsonl", type=Path)
    parser.add_argument("--synthetic-cycles", type=int, default=2000)
    parser.add_argument("--max-draft", type=int, default=2)
    parser.add_argument("--seed", type=int, default=427)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--draft-cycle-us", type=float, default=0.0)
    parser.add_argument("--router-us", type=float, default=0.0)
    parser.add_argument("--aggregate-proposed", type=int, default=196)
    parser.add_argument("--aggregate-accepted", type=int, default=157)
    parser.add_argument(
        "--aggregate-source",
        default="results/qwen4-mtp-policy-gate-16k-cool30-20260910.json fixed K=2 arm",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 0.1 <= args.train_fraction <= 0.9:
        parser.error("--train-fraction must be in [0.1, 0.9]")
    if args.max_draft < 1:
        parser.error("--max-draft must be >= 1")

    real_trace = args.trace_jsonl is not None
    traces = (
        read_jsonl(args.trace_jsonl)
        if real_trace
        else synthetic_traces(args.synthetic_cycles, args.max_draft, args.seed)
    )
    if not traces or any(trace.depth != args.max_draft for trace in traces):
        parser.error("trace must be non-empty and fixed at --max-draft")
    split = max(1, min(len(traces) - 1, int(len(traces) * args.train_fraction)))
    train, holdout = traces[:split], traces[split:]
    model = VerifyCostModel.from_measured()
    baseline = evaluate_policy(
        holdout,
        lambda trace: trace.depth,
        verify_cost_model=model,
        draft_cycle_us=args.draft_cycle_us,
    )
    arms = {}
    calibrators = {}
    for metric in METRICS:
        gate, training = tune_fixed_gate(
            train,
            metric,
            model,
            draft_cycle_us=args.draft_cycle_us,
            router_us=args.router_us,
        )
        evaluation = evaluate_policy(
            holdout,
            gate.select,
            verify_cost_model=model,
            draft_cycle_us=args.draft_cycle_us,
            router_us=args.router_us,
        )
        train_scores, train_labels = calibration_samples(train, metric)
        holdout_scores, holdout_labels = calibration_samples(holdout, metric)
        calibrator = IsotonicCalibrator.fit(train_scores, train_labels)
        calibrators[metric] = calibrator
        arms[f"fixed_{metric}"] = {
            "threshold": gate.threshold,
            "training": training,
            "holdout": asdict(evaluation),
            "rate_ratio_vs_fixed_full": evaluation.token_rate_per_us / baseline.token_rate_per_us,
            "calibration": calibration_metrics(holdout_scores, holdout_labels, calibrator),
        }

    # Pick the confidence representation with the best held-out Brier score,
    # then let calibrated expected value choose the target verify width.
    ev_metric = min(arms, key=lambda name: arms[name]["calibration"]["brier"])[6:]
    ev_calibrator = calibrators[ev_metric]
    ev_selector = lambda trace: choose_expected_value_prefix(
        trace,
        ev_calibrator,
        metric=ev_metric,
        verify_cost_model=model,
    )
    ev = evaluate_policy(
        holdout,
        ev_selector,
        verify_cost_model=model,
        draft_cycle_us=args.draft_cycle_us,
        router_us=args.router_us,
    )
    arms["expected_value"] = {
        "metric": ev_metric,
        "holdout": asdict(ev),
        "rate_ratio_vs_fixed_full": ev.token_rate_per_us / baseline.token_rate_per_us,
    }

    oracle = aggregate_oracle_bound(
        proposed=args.aggregate_proposed,
        accepted=args.aggregate_accepted,
        max_draft=args.max_draft,
        verify_cost_model=model,
    )
    base_us = model.total_us_for_draft(0)
    full_us = model.total_us_for_draft(args.max_draft)
    marginal_us = full_us - base_us

    def scale_for_fraction(saving_us: float, fraction: float = 0.02) -> float | None:
        # Scale only the measured B1->B(k+1) width increment.  This asks how
        # much steeper the full-model curve must be than vocab projection for
        # a perfect gate to reach a 2% target-verify saving.
        denominator = saving_us - fraction * oracle.cycles * marginal_us
        if denominator <= 0.0:
            return None
        return fraction * oracle.cycles * base_us / denominator

    best_arm = max(arms, key=lambda name: arms[name]["rate_ratio_vs_fixed_full"])
    result = {
        "schema": "mlx-uag-confidence-gated-draft-screen-v1",
        "source": "real_trace" if real_trace else "synthetic_fixture",
        "trace_schema": {
            "probability": "per-position draft top-1 probability",
            "margin": "per-position top-1 minus top-2 probability",
            "entropy": "per-position normalized draft entropy",
            "accepted_length": "target-verified accepted prefix length",
        },
        "acceptance_authority": "target verification for every committed token",
        "first_divergence": (
            "not observable from scalar offline traces; no acceptance path is changed"
        ),
        "fixed_full": asdict(baseline),
        "arms": arms,
        "best_arm": best_arm,
        "aggregate_oracle_bound": asdict(oracle),
        "aggregate_source": args.aggregate_source,
        "oracle_overhead_budget_us_per_cycle": {
            "distribution_min": oracle.min_saving_us / oracle.cycles,
            "distribution_max": oracle.max_saving_us / oracle.cycles,
        },
        "marginal_width_scale_needed_for_2pct_oracle": {
            "distribution_best_case": scale_for_fraction(oracle.max_saving_us),
            "distribution_worst_case": scale_for_fraction(oracle.min_saving_us),
        },
        "current_telemetry_gap": None if real_trace else "no per-cycle confidence trace",
        "worth_dedicated_gpu_trace_capture": bool(
            real_trace
            or oracle.max_saving_fraction >= 0.02
            or args.max_draft >= 4
        ),
        "verdict": (
            "CALIBRATE_ON_REAL_TRACE"
            if real_trace
            else (
                "CAPTURE_IF_PIGGYBACKED_OR_WIDTH_COST_GROWS"
                if oracle.max_saving_fraction < 0.02
                else "CAPTURE_REAL_GPU_TRACE"
            )
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
