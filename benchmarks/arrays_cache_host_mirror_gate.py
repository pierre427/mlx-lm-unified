#!/usr/bin/env python3
"""ABBA gate for ArraysCache host-mirror preservation at membership churn.

The candidate arm uses the implementation's mirror carried by ``filter`` or
``extend``.  The control invalidates that mirror immediately after the same
membership operation, reproducing the old next-``advance`` fallback read.
Model weights are not loaded; timings include GPU evaluation and synchronize.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache


def build(layers: int, batch: int, transition: str):
    def populated(rows: int):
        host = list(range(rows))
        caches = []
        for _ in range(layers):
            cache = ArraysCache(1, left_padding=host)
            cache[0] = mx.zeros((rows, 1), dtype=mx.float32)
            mx.eval(cache.left_padding, cache[0])
            caches.append(cache)
        return caches

    if transition == "filter":
        caches = populated(batch)
        additions = None
    else:
        caches = populated(batch - 1)
        additions = populated(1)
    return caches, additions


def transaction(layers: int, batch: int, transition: str, fallback: bool):
    caches, additions = build(layers, batch, transition)
    if transition == "filter":
        keep = list(range(batch - 1))
        for cache in caches:
            cache.filter(keep)
    else:
        for cache, addition in zip(caches, additions):
            cache.extend(addition)
    if fallback:
        for cache in caches:
            cache._host_left_padding = None
    for cache in caches:
        cache.advance(1)
    mx.eval([cache.state for cache in caches])
    mx.synchronize()
    return caches


def timed(layers: int, batch: int, transition: str, fallback: bool, inner: int):
    started = time.perf_counter_ns()
    for _ in range(inner):
        transaction(layers, batch, transition, fallback)
    return (time.perf_counter_ns() - started) / 1e6 / inner


def exact(layers: int, batch: int, transition: str) -> bool:
    fast = transaction(layers, batch, transition, False)
    fallback = transaction(layers, batch, transition, True)
    return all(
        left.left_padding.tolist() == right.left_padding.tolist()
        and all(bool(mx.array_equal(a, b).item()) for a, b in zip(left.cache, right.cache))
        for left, right in zip(fast, fallback)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.layers, args.batch, args.blocks, args.inner, args.warmup) < 1:
        parser.error("all numeric arguments must be positive")
    mx.set_default_device(mx.gpu)

    rows = []
    for transition in ("filter", "extend"):
        for _ in range(args.warmup):
            timed(args.layers, args.batch, transition, True, 1)
            timed(args.layers, args.batch, transition, False, 1)
        blocks = []
        for block in range(args.blocks):
            a1 = timed(args.layers, args.batch, transition, True, args.inner)
            b1 = timed(args.layers, args.batch, transition, False, args.inner)
            b2 = timed(args.layers, args.batch, transition, False, args.inner)
            a2 = timed(args.layers, args.batch, transition, True, args.inner)
            baseline = (a1 + a2) / 2
            candidate = (b1 + b2) / 2
            blocks.append(
                {
                    "block": block,
                    "fallback_ms": baseline,
                    "preserved_ms": candidate,
                    "speedup": baseline / candidate,
                    "saved_ms": baseline - candidate,
                    "a_bracket_drift_fraction": abs(a2 - a1) / baseline,
                }
            )
        rows.append(
            {
                "transition": transition,
                "exact": exact(args.layers, args.batch, transition),
                "median_fallback_ms": statistics.median(x["fallback_ms"] for x in blocks),
                "median_preserved_ms": statistics.median(x["preserved_ms"] for x in blocks),
                "median_speedup": statistics.median(x["speedup"] for x in blocks),
                "median_saved_ms": statistics.median(x["saved_ms"] for x in blocks),
                "max_a_bracket_drift_fraction": max(x["a_bracket_drift_fraction"] for x in blocks),
                "blocks": blocks,
            }
        )
    payload = {
        "scope": "metadata-only membership transition; no steady-decode claim",
        "device": "gpu",
        "layers": args.layers,
        "batch": args.batch,
        "blocks": args.blocks,
        "inner": args.inner,
        "rows": rows,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
