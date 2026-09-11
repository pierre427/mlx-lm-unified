#!/usr/bin/env python3
"""Synthetic component gate for one real BF16 ``KVCache`` object.

This is not a Qwen4, APC-hit, ANE-overlap, model, or whole-request result. It measures
construction, handoff, first-consumer synchronization, ownership, and raw-bit
exactness for the cache object shape used by APC.  Run it only under the lab's
GPU lease and thermal protocol; CPU construction still reads an MLX source.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.cache_capsule import (
    CacheCapsuleGeneration,
    CacheCapsulePool,
    capture_kv_cache_plane,
)
from mlx_lm.models.cache import KVCache


def _thermal_snapshot():
    run = subprocess.run(
        ["/usr/bin/pmset", "-g", "therm"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": run.returncode,
        "stdout": run.stdout.splitlines(),
        "stderr": run.stderr.splitlines(),
    }


def _make_cache(tokens, heads, dim):
    cache = KVCache()
    values = mx.arange(tokens * heads * dim, dtype=mx.uint32).reshape(
        1, heads, tokens, dim
    )
    # Stay inside finite BF16 values but keep enough mantissa variation for a
    # raw-bit comparison to catch conversion instead of transport.
    keys = ((values % 4093).astype(mx.float32) / 32).astype(mx.bfloat16)
    vals = (((values + 17) % 4093).astype(mx.float32) / 64).astype(mx.bfloat16)
    cache.update_and_fetch(keys, vals)
    mx.eval(cache.keys, cache.values)
    mx.synchronize()
    return cache


def _exact(source, restored):
    expected_keys = mx.concatenate([source.keys] * source.target_batch, axis=0)
    expected_values = mx.concatenate(
        [source.values] * source.target_batch, axis=0
    )
    mx.eval(expected_keys, expected_values)
    return bool(
        mx.array_equal(
            restored.keys.view(mx.uint16), expected_keys.view(mx.uint16)
        ).item()
        and mx.array_equal(
            restored.values.view(mx.uint16), expected_values.view(mx.uint16)
        ).item()
        and restored.offset == source.offset
    )


def _run_arm(pool, source, backend):
    started = time.perf_counter_ns()
    receipt = pool.prepare(source, primary=backend, fallback=None)
    build_ms = (time.perf_counter_ns() - started) / 1e6
    with receipt.owner.lease() as lease:
        started = time.perf_counter_ns()
        restored = lease.restore_kv_cache(
            lambda payload: (
                mx.eval(payload.keys, payload.values),
                mx.synchronize(),
            )
        )
        first_consumer_ms = (time.perf_counter_ns() - started) / 1e6
        exact = _exact(source, restored)
    receipt.owner.release()
    return {
        "backend": backend,
        "build_ms": build_ms,
        "first_consumer_ms": first_consumer_ms,
        "total_ms": build_ms + first_consumer_ms,
        "exact_bf16_bits": exact,
        "owner_released": receipt.owner.released,
    }


def _summary(rows, backend):
    selected = [row for row in rows if row["backend"] == backend]
    return {
        "backend": backend,
        "repetitions": len(selected),
        "build_median_ms": statistics.median(row["build_ms"] for row in selected),
        "first_consumer_median_ms": statistics.median(
            row["first_consumer_ms"] for row in selected
        ),
        "total_median_ms": statistics.median(row["total_ms"] for row in selected),
        "all_exact": all(row["exact_bf16_bits"] for row in selected),
        "all_owners_released": all(row["owner_released"] for row in selected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16376)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--target-batch", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.tokens,
        args.heads,
        args.dim,
        args.target_batch,
        args.repetitions,
    ) < 1:
        parser.error("all geometry and repetition values must be positive")
    if args.target_batch < 2 or args.warmup < 0:
        parser.error("target-batch must be >=2 and warmup cannot be negative")

    thermal_before = _thermal_snapshot()
    cache = _make_cache(args.tokens, args.heads, args.dim)
    clock = CacheCapsuleGeneration()
    source = capture_kv_cache_plane(
        cache,
        generation=clock.current,
        source_id="qwen4-apc-kv-plane-0",
        target_batch=args.target_batch,
    )
    rows = []
    with CacheCapsulePool(clock, enabled=True) as pool:
        for index in range(args.warmup + args.repetitions):
            # Interleave the two constructors to make drift visible.  This
            # component gate is not a substitute for the full thermal ABBA
            # request bracket required for promotion.
            order = ("gpu", "cpu") if index % 2 == 0 else ("cpu", "gpu")
            for backend in order:
                row = _run_arm(pool, source, backend)
                row["iteration"] = index
                if index >= args.warmup:
                    rows.append(row)
                    print(json.dumps({"event": "cache_capsule_arm", **row}), flush=True)
        counters = pool.counters

    summaries = [_summary(rows, backend) for backend in ("cpu", "gpu")]
    result = {
        "passed": all(row["all_exact"] for row in summaries)
        and all(row["all_owners_released"] for row in summaries),
        "scope": (
            "synthetic values in one real KVCache object; sequential CPU/GPU "
            "component construction only"
        ),
        "geometry": {
            "tokens": args.tokens,
            "capacity_tokens": int(cache.keys.shape[2]),
            "heads": args.heads,
            "dim": args.dim,
            "target_batch": args.target_batch,
            "source_bytes": int(cache.nbytes),
        },
        "thermal_before": thermal_before,
        "thermal_after": _thermal_snapshot(),
        "pool_counters": counters,
        "summaries": summaries,
        "rows": rows,
        "not_qualified": [
            "ANE/e5rt execution",
            "an AutomaticPrefixCache lookup",
            "a Qwen4 model forward or attention/state consumer",
            "independent GPU overlap",
            "logit/token equality",
            "whole-request wall time",
        ],
        "separate_real_model_gate": (
            "A future qwen4_real_apc_capsule_gate must start from an actual APC "
            "hit, submit e5rt work, run independent GPU work before await_adopt, "
            "then compare first-consumer logits/tokens and ABBA request wall time."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered + "\n")
    if not result["passed"]:
        raise AssertionError("real cache-capsule component gate failed")


if __name__ == "__main__":
    main()
