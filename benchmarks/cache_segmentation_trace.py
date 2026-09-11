#!/usr/bin/env python3
"""Replay a deterministic long-serving trace through cache segment policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import median
import time

from mlx_lm.cache_planes import CachePlaneKind
from mlx_lm.cache_segmentation import (
    PerPlaneSegmentationPolicy,
    PlaneSegmentCandidate,
    Qwen4CacheGeometry,
    SegmentationPolicyConfig,
    SyntheticServingTrace,
    TraceCalibrator,
    pareto_front,
    qsa_window,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=427)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--memory-budget-mb", type=int, default=768)
    parser.add_argument("--trace-input", type=Path)
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=Path("results/qwen4-cache-segmentation-trace-20260910.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/qwen4-cache-segmentation-calibration-20260910.json"
        ),
    )
    return parser.parse_args()


def plane_budgets(total_bytes):
    shares = {
        CachePlaneKind.PROMPT_HOST: 0.05,
        CachePlaneKind.ATTENTION_KV: 0.45,
        CachePlaneKind.ATTENTION_RING: 0.10,
        CachePlaneKind.QSA_SUMMARY: 0.10,
        CachePlaneKind.GDN_RECURRENT: 0.25,
        CachePlaneKind.MTP_DRAFT: 0.05,
    }
    return {kind: int(total_bytes * share) for kind, share in shares.items()}


def configurations():
    for blocks in (16, 64, 128, 256):
        for inline_bytes in (2048, 8192):
            for delta_depth in (4, 8):
                label = (
                    f"blocks={blocks},inline={inline_bytes},depth={delta_depth}"
                )
                config = SegmentationPolicyConfig.uniform(
                    blocks,
                    enabled=True,
                    inline_threshold_bytes=inline_bytes,
                    max_delta_depth=delta_depth,
                )
                yield label, config


def aggregate(repetitions):
    first = repetitions[0]
    deterministic_fields = (
        "trace_digest",
        "events",
        "bytes_before",
        "bytes_after",
        "segment_hits",
        "segment_misses",
        "partial_hit_events",
        "recomputation_avoided_us",
        "fragmentation_bytes",
        "compactions",
        "shared_branch_bytes_avoided",
        "retained_bytes",
        "mean_reuse_distance",
        "p95_reuse_distance",
    )
    for row in repetitions[1:]:
        if any(
            getattr(row, field) != getattr(first, field)
            for field in deterministic_fields
        ):
            raise RuntimeError("trace replay produced non-deterministic metrics")
    return replace(
        first,
        host_latency_ms=median(row.host_latency_ms for row in repetitions),
        compression_cost_ms=median(
            row.compression_cost_ms for row in repetitions
        ),
    )


def boundary_receipts(geometry):
    policy = PerPlaneSegmentationPolicy(
        SegmentationPolicyConfig.uniform(64, enabled=True), geometry
    )
    receipts = {}
    for tokens in (2048, 2051, 2052):
        window = qsa_window(tokens, geometry)
        plan = policy.decide(
            PlaneSegmentCandidate(
                CachePlaneKind.QSA_SUMMARY,
                0,
                tokens,
                geometry.plane_bytes(CachePlaneKind.QSA_SUMMARY, tokens),
                2.0,
                1,
                2,
                1000.0,
                1,
                True,
                False,
            )
        )
        receipts[str(tokens)] = {
            "qsa": asdict(window),
            "segment_stops": [segment.token_stop for segment in plan.segments],
            "incomplete_tail_tokens": plan.incomplete_tail_tokens,
        }
    return receipts


def metric_winners(rows, field, *, maximum):
    value = (max if maximum else min)(getattr(row, field) for row in rows)
    return sorted(row.label for row in rows if getattr(row, field) == value)


def select_candidate(rows, *, latency_multiplier):
    latency_limit = min(row.host_latency_ms for row in rows) * latency_multiplier
    eligible = [row for row in rows if row.host_latency_ms <= latency_limit]
    return min(
        eligible,
        key=lambda row: (
            -row.hit_rate,
            row.compactions,
            row.retained_bytes,
            row.host_latency_ms,
            row.label,
        ),
    ).label


def main():
    args = parse_args()
    if args.sessions < 1 or args.turns < 1 or args.repetitions < 1:
        raise SystemExit("sessions, turns, and repetitions must be positive")
    geometry = Qwen4CacheGeometry()
    if args.trace_input:
        trace = SyntheticServingTrace.from_json(args.trace_input.read_text())
    else:
        trace = SyntheticServingTrace.generate(
            seed=args.seed,
            sessions=args.sessions,
            turns=args.turns,
            geometry=geometry,
        )
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(trace.to_json() + "\n")

    budgets = plane_budgets(args.memory_budget_mb << 20)
    started = time.perf_counter()
    rows = []
    raw_timings = {}
    for label, config in configurations():
        runs = []
        for _ in range(args.repetitions):
            policy = PerPlaneSegmentationPolicy(config, geometry)
            runs.append(TraceCalibrator(policy, budgets).run(trace, label=label))
        rows.append(aggregate(runs))
        raw_timings[label] = {
            "host_latency_ms": [row.host_latency_ms for row in runs],
            "compression_cost_ms": [row.compression_cost_ms for row in runs],
        }
    front_labels = pareto_front(rows)
    front = [row for row in rows if row.label in front_labels]
    artifact = {
        "schema": "qwen4-cache-segmentation-calibration-v1",
        "scope": "CPU/model-free synthetic replay; no production default change",
        "trace": {
            "digest": trace.digest,
            "seed": trace.seed,
            "sessions": trace.sessions,
            "turns": trace.turns,
            "events": len(trace.events),
            "path": str(args.trace_output),
        },
        "qwen4_geometry": asdict(geometry),
        "boundary_receipts": boundary_receipts(geometry),
        "memory_budget_bytes": args.memory_budget_mb << 20,
        "plane_budgets": {kind.value: value for kind, value in budgets.items()},
        "recompute_cost_model": {
            "kind": "model-free eviction proxy",
            "memory_bandwidth_bytes_per_us": 50_000,
            "warning": "not a measured GPU recomputation time",
        },
        "repetitions": args.repetitions,
        "rows": [asdict(row) for row in rows],
        "raw_timings": raw_timings,
        "pareto_front": front_labels,
        "pareto_recommendations": {
            "lowest_host_latency": metric_winners(
                front, "host_latency_ms", maximum=False
            ),
            "highest_hit_rate": metric_winners(front, "hit_rate", maximum=True),
            "most_recomputation_avoided": metric_winners(
                front, "recomputation_avoided_us", maximum=True
            ),
            "lowest_retained_bytes": metric_winners(
                front, "retained_bytes", maximum=False
            ),
            "lowest_fragmentation": metric_winners(
                front, "fragmentation_bytes", maximum=False
            ),
            "balanced_low_latency_candidate": select_candidate(
                rows, latency_multiplier=1.05
            ),
            "higher_hit_knee_candidate": select_candidate(
                rows, latency_multiplier=2.0
            ),
            "selection_rule": (
                "within 1.05x/2x minimum measured host latency, maximize hit "
                "rate, then minimize compactions and retained bytes"
            ),
        },
        "decision": "research candidates only; retain all production defaults",
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
