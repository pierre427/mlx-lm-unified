#!/usr/bin/env python3
"""Measure APC-hit batch materialization through a real attention consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.apc import APCKey, AutomaticPrefixCache
from mlx_lm.cache_capsule import CacheCapsulePool, prepare_prompt_cache_capsules
from mlx_lm.models.cache import BatchKVCache, KVCache


def _thermal():
    result = subprocess.run(
        ["/usr/bin/pmset", "-g", "therm"], capture_output=True, text=True
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.splitlines(),
        "stderr": result.stderr.splitlines(),
    }


def _source(tokens: int, heads: int, dim: int) -> KVCache:
    cache = KVCache()
    values = mx.arange(tokens * heads * dim, dtype=mx.uint32).reshape(
        1, heads, tokens, dim
    )
    keys = ((values % 4093).astype(mx.float32) / 64).astype(mx.bfloat16)
    vals = (((values + 31) % 4093).astype(mx.float32) / 96).astype(mx.bfloat16)
    cache.update_and_fetch(keys, vals)
    mx.eval(cache.state)
    return cache


def _consume(cache: BatchKVCache, batch: int, heads: int, dim: int):
    query = mx.sin(mx.arange(batch * heads * dim, dtype=mx.float32)).reshape(
        batch, heads, 1, dim
    ).astype(mx.bfloat16)
    update = mx.cos(mx.arange(batch * heads * dim, dtype=mx.float32)).reshape(
        batch, heads, 1, dim
    ).astype(mx.bfloat16)
    keys, values = cache.update_and_fetch(update, update)
    output = mx.fast.scaled_dot_product_attention(
        query, keys, values, scale=dim ** -0.5
    )
    mx.eval(output, cache.state)
    mx.synchronize()
    return output


def _digest(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _run(backend: str, tokens: int, heads: int, dim: int, batch: int):
    apc = AutomaticPrefixCache()
    key = APCKey("capsule-serving-gate", cache_layout_fingerprint="plain-bf16")
    token_ids = list(range(tokens))
    apc.store(key, token_ids, [_source(tokens, heads, dim)])
    lookup = apc.lookup(key, token_ids + [tokens])
    if not lookup.hit or lookup.capsule_generation is None:
        raise AssertionError("the serving gate requires a real APC hit")
    restored = lookup.cache
    pool = CacheCapsulePool(apc.capsule_generation, enabled=True)
    owner = None
    started = time.perf_counter_ns()
    if backend == "ordinary":
        batch_cache = [BatchKVCache.merge([restored[0]] * batch)]
    else:
        owner = prepare_prompt_cache_capsules(
            restored,
            target_batch=batch,
            generation=lookup.capsule_generation,
            pool=pool,
            backend=backend,
            fallback=None,
            source_prefix="real-apc-hit",
            synchronize=lambda payload: mx.eval(payload.keys, payload.values),
        )
        if owner is None or owner.capsule_planes != 1:
            raise AssertionError("the cache-capsule plane did not engage")
        batch_cache = owner.prompt_cache
    mx.eval(batch_cache[0].state)
    mx.synchronize()
    prepare_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    output = _consume(batch_cache[0], batch, heads, dim)
    consumer_ms = (time.perf_counter_ns() - started) / 1e6
    result = {
        "backend": backend,
        "prepare_ms": prepare_ms,
        "first_attention_ms": consumer_ms,
        "total_ms": prepare_ms + consumer_ms,
        "output_digest": _digest(output.view(mx.uint16)),
        "cache_digest": _digest(batch_cache[0].keys.view(mx.uint16)),
        "capsule_planes": 0 if owner is None else owner.capsule_planes,
        "pool_counters": pool.counters,
    }
    mx.synchronize()
    batch_cache.clear()
    if owner is not None:
        owner.close(synchronize=False)
        result["owners_released"] = all(
            receipt.owner.released for receipt in owner.receipts
        )
    pool.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.dim, args.batch, args.repetitions) < 1:
        parser.error("all geometry and repetition values must be positive")
    if args.batch < 2 or args.warmup < 0:
        parser.error("batch must be >=2 and warmup cannot be negative")

    rows = []
    backends = ("ordinary", "gpu", "cpu")
    thermal_before = _thermal()
    for iteration in range(args.warmup + args.repetitions):
        order = backends if iteration % 2 == 0 else tuple(reversed(backends))
        for backend in order:
            row = _run(backend, args.tokens, args.heads, args.dim, args.batch)
            row["iteration"] = iteration
            if iteration >= args.warmup:
                rows.append(row)
                print(json.dumps({"event": "apc_capsule_arm", **row}), flush=True)

    summaries = []
    for backend in backends:
        selected = [row for row in rows if row["backend"] == backend]
        summaries.append(
            {
                "backend": backend,
                "prepare_median_ms": statistics.median(
                    row["prepare_ms"] for row in selected
                ),
                "first_attention_median_ms": statistics.median(
                    row["first_attention_ms"] for row in selected
                ),
                "total_median_ms": statistics.median(
                    row["total_ms"] for row in selected
                ),
            }
        )
    digests = {
        (row["output_digest"], row["cache_digest"]) for row in rows
    }
    result = {
        "schema": "mlx-uag.apc-cache-capsule-serving-gate.v1",
        "passed": len(digests) == 1
        and all(row.get("owners_released", True) for row in rows),
        "scope": (
            "real AutomaticPrefixCache hit, physical B2 cache, and first "
            "MLX scaled-dot-product-attention consumer; one plain KV plane"
        ),
        "geometry": {
            "tokens": args.tokens,
            "heads": args.heads,
            "dim": args.dim,
            "batch": args.batch,
        },
        "summaries": summaries,
        "rows": rows,
        "thermal_before": thermal_before,
        "thermal_after": _thermal(),
        "not_qualified": [
            "Qwen4 QSA or recurrent cache planes",
            "e5rt/ANE construction",
            "overlap with independent GPU work",
            "whole-model or whole-request throughput",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise AssertionError("APC cache-capsule serving gate failed")


if __name__ == "__main__":
    main()
