#!/usr/bin/env python3
"""Test whether fused SDPA consumes a zero-stride shared B1 prefix at B2.

Cheap B1-to-B2 broadcasting only helps if the attention kernel accepts the
view without first materializing a two-row prefix.  This benchmark prepares
both a shared broadcast view and a physical B2 control, then interleaves their
synchronized decode-shaped SDPA calls and records time and peak allocation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx


def _measure(fn):
    mx.reset_peak_memory()
    active = int(mx.get_active_memory())
    started = time.perf_counter_ns()
    output = fn()
    mx.eval(output)
    mx.synchronize()
    return {
        "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
        "peak_over_before_bytes": int(mx.get_peak_memory()) - active,
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mx.random.seed(20260910)
    q = mx.random.normal(
        (2, args.query_heads, 1, args.head_dim), dtype=mx.float16
    )
    k1 = mx.random.normal(
        (1, args.kv_heads, args.tokens, args.head_dim), dtype=mx.float16
    )
    v1 = mx.random.normal(
        (1, args.kv_heads, args.tokens, args.head_dim), dtype=mx.float16
    )
    kb = mx.broadcast_to(k1, (2, *k1.shape[1:]))
    vb = mx.broadcast_to(v1, (2, *v1.shape[1:]))
    kp = mx.concatenate([k1, k1], axis=0)
    vp = mx.concatenate([v1, v1], axis=0)
    mx.eval(q, k1, v1, kb, vb, kp, vp)

    def shared():
        return mx.fast.scaled_dot_product_attention(
            q, kb, vb, scale=args.head_dim ** -0.5, force_fused=True
        )

    def physical():
        return mx.fast.scaled_dot_product_attention(
            q, kp, vp, scale=args.head_dim ** -0.5, force_fused=True
        )

    for _ in range(args.warmup):
        mx.eval(shared(), physical())
    mx.synchronize()

    rows = {"shared_broadcast": [], "physical_b2": []}
    last = {}
    for repetition in range(args.repetitions):
        order = (
            (("shared_broadcast", shared), ("physical_b2", physical))
            if repetition % 2 == 0
            else (("physical_b2", physical), ("shared_broadcast", shared))
        )
        for label, fn in order:
            measured = _measure(fn)
            last[label] = measured.pop("output")
            rows[label].append(measured)

    exact = bool(mx.array_equal(last["shared_broadcast"], last["physical_b2"]))
    result = {
        "passed": exact,
        "exact": exact,
        "geometry": {
            "tokens": args.tokens,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "b1_kv_bytes": int(k1.nbytes + v1.nbytes),
            "physical_b2_kv_bytes": int(kp.nbytes + vp.nbytes),
        },
        "arms": {
            label: {
                "median_ms": statistics.median(
                    row["elapsed_ms"] for row in measurements
                ),
                "median_peak_over_before_bytes": statistics.median(
                    row["peak_over_before_bytes"] for row in measurements
                ),
                "rows": measurements,
            }
            for label, measurements in rows.items()
        },
    }
    result["shared_over_physical_speedup"] = (
        result["arms"]["physical_b2"]["median_ms"]
        / result["arms"]["shared_broadcast"]["median_ms"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not exact:
        raise AssertionError("broadcast and physical B2 attention differ")


if __name__ == "__main__":
    main()
