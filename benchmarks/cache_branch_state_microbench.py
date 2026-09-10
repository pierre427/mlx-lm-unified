"""Measure MLX cache-branch descriptor, materialization, and first-write costs.

The benchmark isolates the storage operations behind a cached-prefix hit.  It
does not load a model.  A configurable set of bf16 tensors stands in for the
aggregate KV/QSA sequence state and compares cheap shared descriptors with the
first private write that makes them diverge.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)


def _measure(label, build, reps):
    rows = []
    for rep in range(reps):
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        active_before = int(mx.get_active_memory())
        started = time.perf_counter_ns()
        output = build()
        values = list(_arrays(output))
        if values:
            mx.eval(*values)
        mx.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        active_after = int(mx.get_active_memory())
        peak = int(mx.get_peak_memory())
        row = {
            "rep": rep,
            "elapsed_ms": elapsed_ms,
            "active_delta_bytes": active_after - active_before,
            "peak_over_before_bytes": peak - active_before,
        }
        rows.append(row)
        print(json.dumps({"event": "cache_branch_arm", "label": label, **row}), flush=True)
        del values, output
    return {
        "label": label,
        "rows": rows,
        "median_ms": statistics.median(row["elapsed_ms"] for row in rows),
        "median_active_delta_bytes": statistics.median(
            row["active_delta_bytes"] for row in rows
        ),
        "median_peak_over_before_bytes": statistics.median(
            row["peak_over_before_bytes"] for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16376)
    parser.add_argument("--tail", type=int, default=8)
    parser.add_argument("--slack", type=int, default=256)
    parser.add_argument("--tensors", type=int, default=26)
    parser.add_argument(
        "--total-mib",
        type=float,
        default=512.0,
        help="Aggregate size of the synthetic B=1 prefix state.",
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.tokens, args.tail, args.slack, args.tensors, args.reps) < 1:
        parser.error("tokens, tail, slack, tensors, and reps must be positive")

    target_elements = int(args.total_mib * (1 << 20) / 2)
    dim = max(1, target_elements // (args.tensors * args.tokens))
    base = [
        mx.zeros((1, 1, args.tokens, dim), dtype=mx.bfloat16)
        for _ in range(args.tensors)
    ]
    mx.eval(*base)
    mx.synchronize()
    actual_bytes = sum(array.nbytes for array in base)

    tail_b1 = [
        mx.full((1, 1, args.tail, dim), index + 1, dtype=mx.bfloat16)
        for index in range(args.tensors)
    ]
    tail_b2 = [mx.broadcast_to(value, (2, *value.shape[1:])) for value in tail_b1]
    mx.eval(*tail_b1, *tail_b2)

    def descriptor_branch():
        return [mx.stop_gradient(value) for value in base]

    def explicit_array_branch():
        return [mx.array(value) for value in base]

    def broadcast_b2():
        return [mx.broadcast_to(value, (2, *value.shape[1:])) for value in base]

    def physical_b2():
        return [mx.concatenate([value, value], axis=0) for value in base]

    def descriptor_append_b1():
        return [
            mx.concatenate([mx.stop_gradient(value), update], axis=2)
            for value, update in zip(base, tail_b1)
        ]

    def broadcast_append_b2():
        return [
            mx.concatenate(
                [mx.broadcast_to(value, (2, *value.shape[1:])), update], axis=2
            )
            for value, update in zip(base, tail_b2)
        ]

    def private_delta_only():
        return [mx.array(value) for value in tail_b2]

    measurements = []
    for label, build in (
        ("stop_gradient_descriptor_b1", descriptor_branch),
        ("mx_array_descriptor_b1", explicit_array_branch),
        ("zero_stride_broadcast_b2", broadcast_b2),
        ("physical_concat_b2", physical_b2),
        ("descriptor_first_append_b1", descriptor_append_b1),
        ("broadcast_first_append_b2", broadcast_append_b2),
        ("private_delta_only_b2", private_delta_only),
    ):
        measurements.append(_measure(label, build, args.reps))

    # A serving-ready capsule pays this once outside the request critical path.
    capsule_build = _measure(
        "ready_capsule_build_b2",
        lambda: [
            mx.concatenate(
                [
                    mx.broadcast_to(value, (2, *value.shape[1:])),
                    mx.zeros(
                        (2, 1, args.slack, dim), dtype=value.dtype
                    ),
                ],
                axis=2,
            )
            for value in base
        ],
        args.reps,
    )
    measurements.append(capsule_build)

    # Keep one private capsule alive and measure its first request write.  MLX
    # may donate this unique backing buffer to slice_update instead of copying.
    capsule = [
        mx.concatenate(
            [
                mx.broadcast_to(value, (2, *value.shape[1:])),
                mx.zeros((2, 1, args.slack, dim), dtype=value.dtype),
            ],
            axis=2,
        )
        for value in base
    ]
    mx.eval(*capsule)
    mx.synchronize()

    def patch_capsule():
        nonlocal capsule
        capsule = [
            mx.slice_update(
                value,
                update,
                mx.array([args.tokens], dtype=mx.int32),
                axes=(2,),
            )
            for value, update in zip(capsule, tail_b2)
        ]
        return capsule

    measurements.append(_measure("ready_capsule_first_patch_b2", patch_capsule, args.reps))

    # Functional updates must leave the shared APC prefix untouched.
    patched = broadcast_append_b2()
    mx.eval(*patched)
    source_unchanged = all(bool(mx.all(value == 0).item()) for value in base)
    tail_exact = all(
        bool(mx.array_equal(value[:, :, -args.tail :, :], update).item())
        for value, update in zip(patched, tail_b2)
    )
    result = {
        "passed": source_unchanged and tail_exact,
        "geometry": {
            "tokens": args.tokens,
            "tail": args.tail,
            "slack": args.slack,
            "tensors": args.tensors,
            "dim": dim,
            "b1_prefix_bytes": actual_bytes,
        },
        "source_unchanged": source_unchanged,
        "tail_exact": tail_exact,
        "measurements": measurements,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    if not result["passed"]:
        raise AssertionError("cache branch isolation failed")


if __name__ == "__main__":
    main()
