#!/usr/bin/env python3
"""Synthetic APC deepcopy-vs-COW component gate.

This does not load a model, exercise attention, or qualify the physical B2
consumer.  It measures only APC publication, branch preparation and first KV
write at a caller-selected cache geometry.  Run it under the lab GPU lease and
thermal protocol before using the timings as performance evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

from mlx_lm.apc import APCKey, AutomaticPrefixCache
from mlx_lm.models.cache import KVCache


def _cache(tokens: int, heads: int, head_dim: int) -> KVCache:
    cache = KVCache()
    values = mx.arange(tokens * heads * head_dim, dtype=mx.float32).reshape(
        1, heads, tokens, head_dim
    ).astype(mx.bfloat16)
    cache.update_and_fetch(values, values)
    mx.eval(cache.keys, cache.values)
    return cache


def _median_ms(values):
    return statistics.median(values) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.head_dim, args.repeats) < 1:
        parser.error("all numeric arguments must be positive")

    key = APCKey("synthetic-cow-component")
    tokens = list(range(args.tokens))
    source = [_cache(args.tokens, args.heads, args.head_dim)]
    legacy = AutomaticPrefixCache(cow_branching=False)
    cow = AutomaticPrefixCache(cow_branching=True)
    legacy.store(key, tokens, source)
    cow.store(key, tokens, source)

    legacy_ns = []
    cow_ns = []
    write_ns = []
    exact = True
    for repeat in range(args.repeats):
        query = tokens + [10_000 + repeat]
        started = time.perf_counter_ns()
        legacy_hit = legacy.lookup(key, query)
        legacy_ns.append(time.perf_counter_ns() - started)

        started = time.perf_counter_ns()
        cow_hit = cow.lookup(key, query)
        cow_ns.append(time.perf_counter_ns() - started)

        sibling = cow_hit.cache.fork()
        before = sibling[0].keys
        value = mx.full(
            (1, args.heads, 1, args.head_dim),
            repeat + 1,
            dtype=mx.bfloat16,
        )
        started = time.perf_counter_ns()
        cow_hit.cache[0].update_and_fetch(value, value)
        mx.eval(cow_hit.cache[0].keys)
        write_ns.append(time.perf_counter_ns() - started)
        exact = exact and bool(mx.array_equal(before, sibling[0].keys).item())
        exact = exact and sibling[0].offset == args.tokens
        sibling.close()
        cow_hit.cache.close()

    legacy_ms = _median_ms(legacy_ns)
    cow_ms = _median_ms(cow_ns)
    report = {
        "scope": "synthetic_apc_component_only",
        "qualifies_model_or_b2_consumer": False,
        "geometry": {
            "tokens": args.tokens,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": "bfloat16",
            "logical_cache_bytes": source[0].nbytes,
        },
        "legacy_lookup_median_ms": legacy_ms,
        "cow_lookup_median_ms": cow_ms,
        "lookup_speedup": legacy_ms / cow_ms if cow_ms else None,
        "cow_first_write_median_ms": _median_ms(write_ns),
        "source_and_sibling_exact": exact,
        "cow": cow.apc_stats["cow"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not exact:
        raise SystemExit("COW branch isolation failed")


if __name__ == "__main__":
    main()
