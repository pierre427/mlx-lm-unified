"""A/B exact GDN N=2 prefix fan-out at one recurrent layer.

The baseline materializes the same boundary twice and runs two suffixes
serially.  The candidate materializes once, forks two rows, and runs the
suffixes as one batch.  This is a component gate, not a full-model result.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

# Pin the arithmetic contract before importing MLX.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx

from mlx_lm.gdn_prefix_fanout import (
    GDNPrefixFanout,
    gdn_prefix_fanout_stats,
)
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs


def _config(args):
    return TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=args.hidden,
        intermediate_size=args.hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=max(1, args.hidden // args.head_dim),
        num_key_value_heads=args.key_heads,
        head_dim=args.head_dim,
        linear_num_value_heads=args.value_heads,
        linear_num_key_heads=args.key_heads,
        linear_key_head_dim=args.head_dim,
        linear_value_head_dim=args.head_dim,
        linear_conv_kernel_dim=4,
        vocab_size=128,
    )


def _arrays(values):
    return [value for value in values if value is not None]


def _eval_result(result):
    output, caches = result
    mx.eval(output, *[value for cache in caches for value in _arrays(cache.cache)])


def _time_once(fn, inner):
    started = time.perf_counter_ns()
    results = [fn() for _ in range(inner)]
    arrays = []
    for output, caches in results:
        arrays.append(output)
        arrays.extend(
            value for cache in caches for value in _arrays(cache.cache)
        )
    mx.eval(*arrays)
    mx.synchronize()
    return (time.perf_counter_ns() - started) / 1e6 / inner


def _timing(samples):
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "samples_ms": samples,
    }


def _abba(serial, fanout, blocks, inner):
    rows = []
    serial_samples = []
    fanout_samples = []
    for index in range(blocks):
        a1 = _time_once(serial, inner)
        b1 = _time_once(fanout, inner)
        b2 = _time_once(fanout, inner)
        a2 = _time_once(serial, inner)
        a_ms = (a1 + a2) / 2.0
        b_ms = (b1 + b2) / 2.0
        drift = abs(a2 - a1) / max(min(a1, a2), 1e-12)
        rows.append(
            {
                "block": index,
                "a1_serial_ms": a1,
                "b1_fanout_ms": b1,
                "b2_fanout_ms": b2,
                "a2_serial_ms": a2,
                "serial_bracket_ms": a_ms,
                "fanout_pair_ms": b_ms,
                "speedup": a_ms / b_ms,
                "a_bracket_drift_fraction": drift,
            }
        )
        serial_samples.extend((a1, a2))
        fanout_samples.extend((b1, b2))
    return rows, _timing(serial_samples), _timing(fanout_samples)


def _relative_max(left, right):
    delta = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    scale = mx.maximum(mx.max(mx.abs(right.astype(mx.float32))), 1e-8)
    return float((delta / scale).item())


def _memory(fn):
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    before = int(mx.get_active_memory())
    result = fn()
    _eval_result(result)
    mx.synchronize()
    after = int(mx.get_active_memory())
    peak = int(mx.get_peak_memory())
    del result
    gc.collect()
    mx.clear_cache()
    return {
        "active_before_bytes": before,
        "active_after_bytes": after,
        "active_delta_bytes": after - before,
        "peak_bytes": peak,
        "peak_delta_bytes": peak - before,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--key-heads", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--prefix", type=int, default=5)
    parser.add_argument("--ring", type=int, default=3)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--suffix", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 0 <= args.boundary <= args.ring:
        parser.error("--boundary must be in 0..--ring")
    if args.inner < 1:
        parser.error("--inner must be positive")
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    mx.random.seed(args.seed)

    layer = GatedDeltaNet(_config(args))
    layer.eval()
    prefix = mx.random.normal((1, args.prefix, args.hidden)).astype(mx.float32)
    ring = mx.random.normal((1, args.ring, args.hidden)).astype(mx.float32)
    suffix = mx.random.normal((2, args.suffix, args.hidden)).astype(mx.float32)
    mx.eval(layer.parameters(), prefix, ring, suffix)

    source = ArraysCache(2)
    mx.eval(layer(prefix, cache=source))
    source.start_speculation()
    mx.eval(layer(ring, cache=source))
    record = source._rollbacks[-1]
    defaults = record.fn.__defaults__
    retained = sum(
        int(getattr(value, "nbytes", 0))
        for value in list(defaults[:5]) + [defaults[6]]
    )
    gdn_prefix_fanout_stats(reset=True)
    owner = GDNPrefixFanout.from_latest_record(
        source, enabled=True, retained_input_bytes=retained
    )
    parent_before = [
        None if value is None else mx.array(value) for value in owner._parent
    ]
    mx.eval(*_arrays(parent_before))

    def serial():
        outputs = []
        caches = []
        for row in range(2):
            state = (
                list(record.snapshot)
                if args.boundary == 0
                else list(record.fn(args.boundary))
            )
            cache = ArraysCache(len(state))
            cache.cache = state
            outputs.append(layer(suffix[row : row + 1], cache=cache))
            caches.append(cache)
        return mx.concatenate(outputs, axis=0), caches

    def fanout():
        lease = owner.fork(args.boundary)
        output = layer(suffix, cache=lease.cache)
        caches = [lease.cache.extract(row) for row in range(2)]
        lease.abort()
        return output, caches

    for _ in range(args.warmup):
        _eval_result(serial())
        _eval_result(fanout())
        _eval_result(fanout())
        _eval_result(serial())
    mx.synchronize()

    blocks, serial_timing, fanout_timing = _abba(
        serial, fanout, args.reps, args.inner
    )
    speedups = [row["speedup"] for row in blocks]
    speedup = statistics.median(speedups)
    serial_memory = _memory(serial)
    fanout_memory = _memory(fanout)
    serial_result = serial()
    fanout_result = fanout()
    _eval_result(serial_result)
    _eval_result(fanout_result)
    output_rel = _relative_max(fanout_result[0], serial_result[0])
    state_rel = max(
        _relative_max(candidate, reference)
        for candidate_cache, reference_cache in zip(
            fanout_result[1], serial_result[1]
        )
        for candidate, reference in zip(
            candidate_cache.cache, reference_cache.cache
        )
        if candidate is not None
    )
    parent_unchanged = all(
        before is None or bool(mx.array_equal(before, after))
        for before, after in zip(parent_before, owner._parent)
    )
    proof = owner.fork(args.boundary)
    boundary_state = (
        list(record.snapshot)
        if args.boundary == 0
        else list(record.fn(args.boundary))
    )
    mx.eval(*_arrays(boundary_state))
    boundary_rows_bit_exact = all(
        bool(mx.array_equal(value[row : row + 1], reference))
        for value, reference in zip(proof.cache.cache, boundary_state)
        if value is not None
        for row in range(2)
    )
    copied_state_bytes_per_fanout = proof.copied_state_bytes
    proof.abort()
    stats = gdn_prefix_fanout_stats()
    row_tokens = 2 * args.suffix
    payload = {
        "scope": "one-layer component gate; no full-model claim",
        "device": args.device,
        "timing": {
            "design": "ABBA",
            "blocks": args.reps,
            "inner_transactions_per_sample": args.inner,
            "reported_ms": "per transaction",
        },
        "geometry": {
            "hidden": args.hidden,
            "key_heads": args.key_heads,
            "value_heads": args.value_heads,
            "head_dim": args.head_dim,
            "prefix": args.prefix,
            "ring": args.ring,
            "boundary": args.boundary,
            "suffix": args.suffix,
            "rows": 2,
        },
        "serial": serial_timing,
        "fanout": fanout_timing,
        "abba_blocks": blocks,
        "speedup": speedup,
        "speedup_mean": statistics.mean(speedups),
        "max_a_bracket_drift_fraction": max(
            row["a_bracket_drift_fraction"] for row in blocks
        ),
        "serial_row_tokens_per_second": row_tokens
        / (serial_timing["median_ms"] / 1000.0),
        "fanout_row_tokens_per_second": row_tokens
        / (fanout_timing["median_ms"] / 1000.0),
        "memory": {"serial": serial_memory, "fanout": fanout_memory},
        "boundary_rows_bit_exact": boundary_rows_bit_exact,
        "descendant_output_bit_exact": bool(
            mx.array_equal(fanout_result[0], serial_result[0])
        ),
        "descendant_state_bit_exact": all(
            bool(mx.array_equal(candidate, reference))
            for candidate_cache, reference_cache in zip(
                fanout_result[1], serial_result[1]
            )
            for candidate, reference in zip(
                candidate_cache.cache, reference_cache.cache
            )
            if candidate is not None
        ),
        "output_relative_max": output_rel,
        "state_relative_max": state_rel,
        "parent_bit_exact": parent_unchanged,
        "retained_input_bytes": retained,
        "copied_state_bytes_per_fanout": copied_state_bytes_per_fanout,
        "replay_tokens_per_fanout": args.boundary,
        "counters": stats,
        "engaged": stats["fanout_batches"] > 0,
        "component_gate": (
            parent_unchanged
            and boundary_rows_bit_exact
            and output_rel < 3e-3
            and state_rel < 3e-3
            and stats["fanout_batches"] > 0
        ),
    }
    owner.close()
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
