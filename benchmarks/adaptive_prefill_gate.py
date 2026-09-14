#!/usr/bin/env python3
"""Score matched mixed-load runs for adaptive prefill scheduling."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def jain(values):
    if not values:
        return None
    denominator = len(values) * sum(value * value for value in values)
    return 1.0 if denominator == 0 else sum(values) ** 2 / denominator


def scheduler_delta(run, key):
    before = (run.get("server_metrics_before") or {}).get("scheduler", {})
    after = (run.get("server_metrics_after") or {}).get("scheduler", {})
    before_value = before.get(key, 0)
    after_value = after.get(key, 0)
    if isinstance(after_value, dict):
        return {
            name: int(value) - int((before_value or {}).get(name, 0))
            for name, value in after_value.items()
        }
    return int(after_value) - int(before_value)


def summarize(run):
    rows = [
        row
        for row in run["requests"]
        if row.get("action", "complete")
        in {"complete", "cache_evict", "cache_reallocate"}
    ]
    ttft = [row["ttft_ms"] for row in rows if row.get("ttft_ms") is not None]
    itl = [value for row in rows for value in row.get("itl_ms", [])]
    starts = [row["started_offset_ms"] for row in rows]
    ends = [row["started_offset_ms"] + row["wall_ms"] for row in rows]
    tokens = [
        int(row.get("completion_tokens") or row.get("stream_events") or 0)
        for row in rows
    ]
    by_tenant = defaultdict(int)
    service_by_tenant = defaultdict(float)
    for row, count in zip(rows, tokens):
        tenant = row["tenant_id"]
        by_tenant[tenant] += count
        service_by_tenant[tenant] += row["wall_ms"] / 1000.0
    rates = [
        by_tenant[tenant] / seconds
        for tenant, seconds in service_by_tenant.items()
        if seconds > 0
    ]
    duration_s = (max(ends) - min(starts)) / 1000.0 if rows else 0.0
    configured = (run.get("server_metrics_after") or {}).get("configured", {})
    engagement = {
        key: scheduler_delta(run, key)
        for key in (
            "prefill_rounds",
            "adaptive_prefill_release_rounds",
            "adaptive_prefill_slack_deferred_rounds",
            "adaptive_prefill_deadline_forced_rounds",
            "adaptive_prefill_apc_priority_admissions",
            "adaptive_prefill_chunk_histogram",
        )
    }
    adaptive = bool(configured.get("adaptive_prefill", False))
    engagement_ok = engagement["prefill_rounds"] > 0
    if adaptive:
        engagement_ok = (
            engagement_ok
            and engagement["adaptive_prefill_release_rounds"] > 0
            and engagement["adaptive_prefill_slack_deferred_rounds"] > 0
            and bool(engagement["adaptive_prefill_chunk_histogram"])
        )
    return {
        "adaptive_prefill": adaptive,
        "requests": len(rows),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "itl_p90_ms": percentile(itl, 0.90),
        "itl_p99_ms": percentile(itl, 0.99),
        "aggregate_output_tps": sum(tokens) / duration_s if duration_s > 0 else None,
        "jain_tenant_token_rate": jain(rates),
        "engagement": engagement,
        "engagement_pass": engagement_ok,
    }


def compare(baseline_run, candidate_run):
    baseline = summarize(baseline_run)
    candidate = summarize(candidate_run)
    if baseline["adaptive_prefill"]:
        raise ValueError("baseline must have adaptive_prefill disabled")
    if not candidate["adaptive_prefill"]:
        raise ValueError("candidate must have adaptive_prefill enabled")
    for name in ("ttft_p95_ms", "itl_p99_ms", "aggregate_output_tps"):
        if baseline[name] is None or candidate[name] is None:
            raise ValueError(f"both runs need {name}")
    gates = {
        "baseline_engaged": baseline["engagement_pass"],
        "candidate_engaged": candidate["engagement_pass"],
        "itl_p99_improved_20pct": (
            candidate["itl_p99_ms"] <= baseline["itl_p99_ms"] * 0.80
        ),
        "throughput_within_5pct": (
            candidate["aggregate_output_tps"] >= baseline["aggregate_output_tps"] * 0.95
        ),
        "ttft_p95_within_25pct": (
            candidate["ttft_p95_ms"] <= baseline["ttft_p95_ms"] * 1.25
        ),
        "fairness_at_least_090": ((candidate["jain_tenant_token_rate"] or 0.0) >= 0.90),
    }
    return {
        "schema": "mlx-lm.adaptive-prefill-gate.v1",
        "baseline": baseline,
        "candidate": candidate,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.out}")


if __name__ == "__main__":
    main()
