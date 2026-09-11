#!/usr/bin/env python3
"""Gate direct shared-base/private-delta QSA against physical B2 controls.

The benchmark keeps the 16K production geometry and interleaves four arms:

* ``physical_prebuilt``: current indexed QSA over an already-materialized B2 KV;
* ``physical_build``: broadcast/concatenate the B2 KV in the timed graph;
* ``serial_b1``: two independent B1 indexed-QSA dispatches;
* ``private_delta``: one B2 dispatch whose single accumulator traverses a
  straight-line B1-base phase followed by a straight-line private-suffix phase.

All arms use the same compact block selection, split count, and merge path.
The receipt includes raw samples, allocation deltas, exactness, immutable-base
digests, swap growth, and macOS thermal-pressure snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.qwen4_exp import QSACompactBlocks
from mlx_lm.models.qwen4_qsa_indexed import (
    qwen4_qsa_indexed_attention,
    qwen4_qsa_indexed_private_delta_attention,
)


def _command(argv: list[str]) -> str:
    return subprocess.run(
        argv, capture_output=True, check=False, text=True
    ).stdout.strip()


def _swap_used_bytes() -> int | None:
    output = _command(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    match = re.search(r"used\s*=\s*([0-9.]+)([MG])", output)
    if match is None:
        return None
    multiplier = 1024**2 if match.group(2) == "M" else 1024**3
    return int(float(match.group(1)) * multiplier)


def _host_snapshot() -> dict:
    return {
        "thermal": _command(["/usr/bin/pmset", "-g", "therm"]),
        "swap_used_bytes": _swap_used_bytes(),
    }


def _digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        raw = array.view(mx.uint16) if array.dtype == mx.bfloat16 else array
        digest.update(np.asarray(raw).view(np.uint8).tobytes())
    return digest.hexdigest()


def _row_compact(compact: QSACompactBlocks, row: int) -> QSACompactBlocks:
    causal = compact.causal_mask
    if causal is not None and int(causal.shape[0]) != 1:
        causal = causal[row : row + 1]
    left = compact.left_padding
    if left is not None:
        left = left[row : row + 1]
    return replace(
        compact,
        block_ids=compact.block_ids[row : row + 1],
        block_counts=compact.block_counts[row : row + 1],
        tail_start=compact.tail_start[row : row + 1],
        tail_stop=compact.tail_stop[row : row + 1],
        left_padding=left,
        causal_mask=causal,
    )


def _make_compact(
    *, batch: int, query: int, base: int, delta_lengths: list[int], selected: int
) -> QSACompactBlocks:
    if selected < 2:
        raise ValueError("selected width must be at least two")
    base_blocks = base // 4
    # Cover the whole immutable prefix, then include the first private block.
    # For early rows that block is also the causal tail; the existing compact
    # converter deduplicates it exactly as it does in production.
    prefix = np.linspace(0, base_blocks - 1, selected - 1, dtype=np.uint32)
    prefix = np.unique(prefix)
    if len(prefix) != selected - 1:
        raise ValueError("selected width is too large for the base")
    selected_ids = np.concatenate([prefix, np.array([base_blocks], np.uint32)])
    ids = np.broadcast_to(selected_ids, (batch, query, selected)).copy()
    counts = np.full((batch, query), selected, dtype=np.int32)
    q_pos = np.empty((batch, query), dtype=np.int32)
    for row, delta_length in enumerate(delta_lengths):
        if delta_length < query:
            raise ValueError("each private suffix must cover every query row")
        q_pos[row] = base + np.arange(delta_length - query, delta_length)
    tail_stop = q_pos + 1
    tail_start = tail_stop // 4 * 4
    return QSACompactBlocks(
        block_ids=mx.array(ids),
        block_counts=mx.array(counts),
        tail_start=mx.array(tail_start),
        tail_stop=mx.array(tail_stop),
        left_padding=None,
        block_size=4,
        physical_width=base + max(delta_lengths),
        causal_mask=None,
    )


def _measure(fn) -> tuple[dict, mx.array]:
    mx.reset_peak_memory()
    active_before = int(mx.get_active_memory())
    started = time.perf_counter_ns()
    output = fn()
    mx.eval(output)
    mx.synchronize()
    elapsed = time.perf_counter_ns() - started
    return (
        {
            "elapsed_ms": elapsed / 1.0e6,
            "active_before_bytes": active_before,
            "active_after_bytes": int(mx.get_active_memory()),
            "peak_over_before_bytes": max(
                0, int(mx.get_peak_memory()) - active_before
            ),
        },
        output,
    )


def _summary(rows: list[dict]) -> dict:
    elapsed = [row["elapsed_ms"] for row in rows]
    return {
        "median_ms": statistics.median(elapsed),
        "mean_ms": statistics.fmean(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "relative_spread": (max(elapsed) - min(elapsed)) / statistics.median(elapsed),
        "median_peak_over_before_bytes": statistics.median(
            row["peak_over_before_bytes"] for row in rows
        ),
        "rows": rows,
    }


def _run_width(args, query: int) -> dict:
    batch = args.batch
    delta_lengths = [
        args.delta if row % 2 == 0 else args.delta - 3
        for row in range(batch)
    ]
    compact = _make_compact(
        batch=batch,
        query=query,
        base=args.base,
        delta_lengths=delta_lengths,
        selected=args.selected,
    )
    mx.random.seed(args.seed + query)
    q = mx.random.normal(
        (batch, args.query_heads, query, args.head_dim)
    ).astype(mx.bfloat16)
    base_k = mx.random.normal(
        (1, args.kv_heads, args.base, args.head_dim)
    ).astype(mx.bfloat16)
    base_v = mx.random.normal(
        (1, args.kv_heads, args.base, args.head_dim)
    ).astype(mx.bfloat16)
    delta_k = mx.random.normal(
        (batch, args.kv_heads, args.delta, args.head_dim)
    ).astype(mx.bfloat16)
    delta_v = mx.random.normal(
        (batch, args.kv_heads, args.delta, args.head_dim)
    ).astype(mx.bfloat16)
    lengths = mx.array(delta_lengths, dtype=mx.uint32)
    physical_k = mx.concatenate(
        [mx.broadcast_to(base_k, (batch, *base_k.shape[1:])), delta_k], axis=2
    )
    physical_v = mx.concatenate(
        [mx.broadcast_to(base_v, (batch, *base_v.shape[1:])), delta_v], axis=2
    )
    mx.eval(
        q,
        base_k,
        base_v,
        delta_k,
        delta_v,
        lengths,
        physical_k,
        physical_v,
        compact.block_ids,
        compact.block_counts,
        compact.tail_start,
        compact.tail_stop,
    )
    base_digest_before = _digest(base_k, base_v)
    row_compacts = [_row_compact(compact, row) for row in range(batch)]

    def physical_prebuilt():
        return qwen4_qsa_indexed_attention(
            q,
            physical_k,
            physical_v,
            compact,
            scale=args.head_dim**-0.5,
            splits=args.splits,
            hpt=args.hpt,
        )

    def physical_build():
        keys = mx.concatenate(
            [mx.broadcast_to(base_k, (batch, *base_k.shape[1:])), delta_k],
            axis=2,
        )
        values = mx.concatenate(
            [mx.broadcast_to(base_v, (batch, *base_v.shape[1:])), delta_v],
            axis=2,
        )
        return qwen4_qsa_indexed_attention(
            q,
            keys,
            values,
            compact,
            scale=args.head_dim**-0.5,
            splits=args.splits,
            hpt=args.hpt,
        )

    def serial_b1():
        outputs = []
        for row in range(batch):
            outputs.append(
                qwen4_qsa_indexed_attention(
                    q[row : row + 1],
                    physical_k[row : row + 1],
                    physical_v[row : row + 1],
                    row_compacts[row],
                    scale=args.head_dim**-0.5,
                    splits=args.splits,
                    hpt=args.hpt,
                )
            )
        return mx.concatenate(outputs, axis=0)

    def private_delta():
        return qwen4_qsa_indexed_private_delta_attention(
            q,
            base_k,
            base_v,
            delta_k,
            delta_v,
            lengths,
            compact,
            scale=args.head_dim**-0.5,
            splits=args.splits,
            hpt=args.hpt,
        )

    arms = {
        "physical_prebuilt": physical_prebuilt,
        "physical_build": physical_build,
        "serial_b1": serial_b1,
        "private_delta": private_delta,
    }
    for _ in range(args.warmup):
        for fn in arms.values():
            mx.eval(fn())
        mx.synchronize()

    rows = {name: [] for name in arms}
    last = {}
    order = list(arms)
    for repetition in range(args.repetitions):
        labels = order if repetition % 2 == 0 else list(reversed(order))
        for label in labels:
            measurement, output = _measure(arms[label])
            rows[label].append(measurement)
            last[label] = output

    reference = last["physical_prebuilt"]
    exact = {}
    max_abs = {}
    for label, output in last.items():
        mx.eval(output, reference)
        exact[label] = bool(mx.array_equal(output, reference).item())
        max_abs[label] = float(
            mx.max(mx.abs(output.astype(mx.float32) - reference.astype(mx.float32))).item()
        )
    summaries = {name: _summary(values) for name, values in rows.items()}
    base_digest_after = _digest(base_k, base_v)
    return {
        "query_width": query,
        "private_delta_layout": "source_homogeneous_single_accumulator",
        "geometry": {
            "batch": batch,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "base_tokens": args.base,
            "delta_width": args.delta,
            "delta_lengths": delta_lengths,
            "selected_blocks": args.selected,
            "splits": args.splits,
            "heads_per_threadgroup": args.hpt,
            "base_kv_bytes": int(base_k.nbytes + base_v.nbytes),
            "physical_b2_kv_bytes": int(physical_k.nbytes + physical_v.nbytes),
        },
        "arms": summaries,
        "exact_against_physical_prebuilt": exact,
        "max_abs_against_physical_prebuilt": max_abs,
        "immutable_base_digest_before": base_digest_before,
        "immutable_base_digest_after": base_digest_after,
        "immutable_base_unchanged": base_digest_before == base_digest_after,
        "private_delta_over_physical_prebuilt_speedup": (
            summaries["physical_prebuilt"]["median_ms"]
            / summaries["private_delta"]["median_ms"]
        ),
        "private_delta_over_physical_build_speedup": (
            summaries["physical_build"]["median_ms"]
            / summaries["private_delta"]["median_ms"]
        ),
        "private_delta_over_serial_b1_speedup": (
            summaries["serial_b1"]["median_ms"]
            / summaries["private_delta"]["median_ms"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--delta", type=int, default=8)
    parser.add_argument("--selected", type=int, default=512)
    parser.add_argument("--query-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--splits", type=int, default=128)
    parser.add_argument("--hpt", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.base % 4:
        raise ValueError("base must end on a four-token block")
    if args.batch < 2:
        raise ValueError("batch must be at least two")
    if args.delta < 4:
        raise ValueError("delta must be at least four for the ragged B2 gate")
    before = _host_snapshot()
    cells = [_run_width(args, query) for query in (1, 3)]
    after = _host_snapshot()
    swap_growth = None
    if before["swap_used_bytes"] is not None and after["swap_used_bytes"] is not None:
        swap_growth = after["swap_used_bytes"] - before["swap_used_bytes"]
    passed = all(
        all(cell["exact_against_physical_prebuilt"].values())
        and cell["immutable_base_unchanged"]
        for cell in cells
    )
    result = {
        "passed": passed,
        "mlx_version": str(mx.__version__),
        "device": str(mx.default_device()),
        "host_before": before,
        "host_after": after,
        "swap_growth_bytes": swap_growth,
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("private-delta QSA gate failed exactness or base integrity")
    if swap_growth is not None and swap_growth > 256 * 1024**2:
        raise RuntimeError(f"swap grew by {swap_growth} bytes during QSA gate")


if __name__ == "__main__":
    main()
