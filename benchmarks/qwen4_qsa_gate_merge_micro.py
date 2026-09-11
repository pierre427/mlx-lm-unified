"""Exactness and seam timing for QSA merge-epilogue output gating."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models import qwen4_qsa_indexed_merge as merge


def _inputs(length: int, seed: int):
    rng = np.random.default_rng(seed)
    shape = (1, 24, length, 128)
    m = mx.array(rng.normal(0, 4, shape).astype(np.float32))
    l = mx.array(np.exp(rng.normal(0, 1, shape)).astype(np.float32))
    o = mx.array(
        rng.normal(0, 2, shape + (256,)).astype(np.float32)
    ).astype(mx.bfloat16)
    gate = mx.array(
        rng.normal(0, 3, (1, length, 24 * 256)).astype(np.float32)
    ).astype(mx.bfloat16)
    mx.eval(m, l, o, gate)
    return m, l, o, gate


def _run(inputs, fused_gate: bool):
    m, l, o, gate = inputs
    os.environ["MLX_QWEN4_QSA_INDEXED_FUSED_MERGE"] = "1"
    os.environ["MLX_QWEN4_QSA_INDEXED_FUSED_GATE"] = "1" if fused_gate else "0"
    return merge.combine_indexed_partials(
        m,
        l,
        o,
        output_dtype=mx.bfloat16,
        output_gate=gate,
    )


def _time(inputs, fused_gate: bool, calls: int) -> float:
    started = time.perf_counter_ns()
    for _ in range(calls):
        mx.eval(_run(inputs, fused_gate))
    mx.synchronize()
    return (time.perf_counter_ns() - started) / calls / 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=(1, 3))
    parser.add_argument("--exactness-seeds", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--calls", type=int, default=12)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cells = []
    all_exact = True
    for length in args.lengths:
        exact = 0
        max_abs = 0.0
        for seed in range(args.exactness_seeds):
            inputs = _inputs(length, seed)
            baseline = _run(inputs, False)
            candidate = _run(inputs, True)
            mx.eval(baseline, candidate)
            equal = bool(mx.array_equal(baseline, candidate).item())
            exact += int(equal)
            max_abs = max(
                max_abs,
                float(
                    mx.max(
                        mx.abs(baseline.astype(mx.float32) - candidate)
                    ).item()
                ),
            )
        all_exact &= exact == args.exactness_seeds

        inputs = _inputs(length, 20260911 + length)
        _time(inputs, False, 2)
        _time(inputs, True, 2)
        baseline_samples = []
        candidate_samples = []
        for block in range(args.blocks):
            order = (
                (False, True, True, False)
                if block % 2 == 0
                else (True, False, False, True)
            )
            for fused_gate in order:
                value = _time(inputs, fused_gate, args.calls)
                (candidate_samples if fused_gate else baseline_samples).append(value)
        baseline_us = statistics.median(baseline_samples)
        candidate_us = statistics.median(candidate_samples)
        cells.append(
            {
                "length": length,
                "exact": exact,
                "comparisons": args.exactness_seeds,
                "max_abs": max_abs,
                "baseline_us_per_layer": baseline_us,
                "candidate_us_per_layer": candidate_us,
                "speedup": baseline_us / candidate_us,
                "saving_us_per_layer": baseline_us - candidate_us,
                "projected_saving_us_12_layers": 12 * (baseline_us - candidate_us),
                "baseline_samples": baseline_samples,
                "candidate_samples": candidate_samples,
            }
        )

    result = {
        "schema": "mlx-uag.qwen4-qsa-merge-gate-micro.v1",
        "device": mx.device_info(),
        "all_exact": all_exact,
        "status": merge.fused_merge_status(),
        "cells": cells,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    raise SystemExit(0 if all_exact else 2)


if __name__ == "__main__":
    main()
