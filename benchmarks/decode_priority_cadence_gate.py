#!/usr/bin/env python3
"""Score matched mixed-load runs for decode-priority cadence."""

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
    return int(after.get(key, 0)) - int(before.get(key, 0))


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
    cadence = int(configured.get("decode_priority_cadence", 1))
    engagement = {
        key: scheduler_delta(run, key)
        for key in (
            "prefill_rounds",
            "decode_priority_release_rounds",
            "decode_priority_deferred_rounds",
        )
    }
    quality = all(
        not row.get("error")
        and row.get("quality", {}).get("expected_contains_pass") is True
        and row.get("quality", {}).get("cross_request_canary_pass") is True
        and row.get("quality", {}).get("own_canary_pass") is True
        and row.get("quality", {}).get("reference_exact") is not False
        for row in rows
    )
    engagement_ok = engagement["prefill_rounds"] > 0
    if cadence > 1:
        engagement_ok = (
            engagement_ok
            and engagement["decode_priority_release_rounds"] > 0
            and engagement["decode_priority_deferred_rounds"] > 0
        )
    return {
        "cadence": cadence,
        "requests": len(rows),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "itl_p99_ms": percentile(itl, 0.99),
        "aggregate_output_tps": sum(tokens) / duration_s if duration_s > 0 else None,
        "jain_tenant_token_rate": jain(rates),
        "quality_pass": quality,
        "engagement": engagement,
        "engagement_pass": engagement_ok,
    }


def compare(baseline_run, candidate_run):
    baseline = summarize(baseline_run)
    candidate = summarize(candidate_run)
    if baseline["cadence"] != 1:
        raise ValueError("baseline must report decode_priority_cadence=1")
    if candidate["cadence"] <= 1:
        raise ValueError("candidate must report decode_priority_cadence>1")
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
        "quality": baseline["quality_pass"] and candidate["quality_pass"],
    }
    return {
        "schema": "mlx-lm.decode-priority-cadence-gate.v1",
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
