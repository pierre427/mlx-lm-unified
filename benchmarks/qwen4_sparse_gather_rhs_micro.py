#!/usr/bin/env python3
"""Real-weight gate for MLX's grouped sorted-RHS gather at Qwen4 S=7.

Run this driver in a fresh process with either the stock density gate or
``MLX_GATHER_QMM_RHS_MIN_DENSITY=0``.  It maps one deployed layer's fused
gate/up and down expert tables, then times sorted 70-assignment gathers over
the observed S=7 unique-expert range.  The override exists only in the
rejected experimental MLX wheel recorded by the 2026-09-11 lab artifact; stock
MLX intentionally ignores it.  The driver never loads the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


MODEL = Path(
    "/System/Volumes/Data/Users/pierrelamy/mlx-models/"
    "Qwen3.8-Flash-Next-MLX-4bit-MTP"
)
PREFIX = "language_model.model.layers.0.mlp.switch_mlp"
UNIQUE_COUNTS = (33, 49, 66, 70)


def _require_idle_service() -> None:
    with socket.socket() as sock:
        sock.settimeout(0.1)
        if sock.connect_ex(("127.0.0.1", 8282)) == 0:
            raise RuntimeError("port 8282 is live; stop the resident model first")


def _routes(unique_count: int, offset: int = 0) -> np.ndarray:
    if not 10 <= unique_count <= 70:
        raise ValueError(unique_count)
    rows: list[list[int]] = [list(range(10))]
    next_new = 10
    remaining_new = unique_count - 10
    for row_id in range(1, 7):
        slots_left = (7 - row_id) * 10
        introduce = min(10, max(0, remaining_new - max(0, slots_left - 10)))
        row = list(range(next_new, next_new + introduce))
        next_new += introduce
        remaining_new -= introduce
        candidate = row_id
        while len(row) < 10:
            expert = candidate % max(next_new, 10)
            candidate += 7
            if expert not in row:
                row.append(expert)
        rows.append(row)
    routes = (np.asarray(rows, dtype=np.uint32) + offset) % 512
    if len(np.unique(routes)) != unique_count:
        raise AssertionError((unique_count, len(np.unique(routes)), routes.tolist()))
    if any(len(np.unique(row)) != 10 for row in routes):
        raise AssertionError("a token route contains a duplicate expert")
    return routes


def _load_tables(model: Path) -> dict[str, tuple[mx.array, mx.array, mx.array]]:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    required = {
        f"{PREFIX}.{proj}.{suffix}"
        for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "scales", "biases")
    }
    shards = {weight_map[name] for name in required}
    if len(shards) != 1:
        raise RuntimeError(f"layer-0 routed tables span shards: {sorted(shards)}")
    shard = model / shards.pop()
    mapped = mx.load(str(shard))

    def trio(proj: str) -> tuple[mx.array, mx.array, mx.array]:
        return tuple(
            mapped[f"{PREFIX}.{proj}.{suffix}"]
            for suffix in ("weight", "scales", "biases")
        )

    gate = trio("gate_proj")
    up = trio("up_proj")
    gate_up = tuple(mx.concatenate([g, u], axis=1) for g, u in zip(gate, up))
    down = trio("down_proj")
    mx.eval(*gate_up, *down)
    del mapped, gate, up
    return {"gate_up": gate_up, "down": down}


def _inputs(routes: np.ndarray, projection: str) -> tuple[mx.array, mx.array]:
    route_array = mx.array(routes, dtype=mx.uint32)
    flat = route_array.flatten()
    order = mx.argsort(flat)
    sorted_indices = flat[order]
    if projection == "gate_up":
        base = mx.arange(7 * 2560, dtype=mx.float32).reshape(7, 2560)
        tokens = (mx.sin(base * 0.0013) + 0.25 * mx.cos(base * 0.0007)).astype(
            mx.bfloat16
        )
        x = tokens[order // 10]
    else:
        base = mx.arange(70 * 640, dtype=mx.float32).reshape(70, 640)
        assignments = (
            mx.sin(base * 0.013) * 0.75 + mx.cos(base * 0.007) * 0.25
        ).astype(mx.bfloat16)
        x = assignments[order]
    mx.eval(x, sorted_indices)
    return x[:, None, :], sorted_indices


def _gather(
    x: mx.array,
    indices: mx.array,
    table: tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    return mx.gather_qmm(
        x,
        *table,
        rhs_indices=indices,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
        sorted_indices=True,
    ).squeeze(-2)


def _digest(value: mx.array) -> tuple[str, np.ndarray]:
    widened = np.asarray(value.astype(mx.float32), dtype=np.float32)
    return hashlib.sha256(widened.tobytes()).hexdigest(), widened


def _summary(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": values,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_idle_service()
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("Metal GPU is unavailable")
    mx.reset_peak_memory()
    tables = _load_tables(args.model)
    arrays: dict[str, np.ndarray] = {}
    cells: dict[str, object] = {}

    for unique_count in UNIQUE_COUNTS:
        for projection in ("gate_up", "down"):
            key = f"u{unique_count}_{projection}"
            base_routes = _routes(unique_count)
            x, indices = _inputs(base_routes, projection)
            value = _gather(x, indices, tables[projection])
            mx.eval(value)
            digest, widened = _digest(value)
            arrays[key] = widened

            variants = []
            for shift in (0, 73, 149, 227, 311, 397):
                vx, vi = _inputs(_routes(unique_count, shift), projection)
                variants.append((vx, vi))
            for warmup in range(args.warmups):
                vx, vi = variants[warmup % len(variants)]
                mx.eval(_gather(vx, vi, tables[projection]))
            samples = []
            for trial in range(args.trials):
                started = time.perf_counter_ns()
                for inner in range(args.inner):
                    vx, vi = variants[(trial * args.inner + inner) % len(variants)]
                    mx.eval(_gather(vx, vi, tables[projection]))
                mx.synchronize()
                samples.append(
                    (time.perf_counter_ns() - started) / 1e6 / args.inner
                )
            cells[key] = {
                "unique_experts": unique_count,
                "duplicate_fraction": (70 - unique_count) / 70,
                "projection": projection,
                "output_shape": list(widened.shape),
                "output_sha256_f32": digest,
                "timing": _summary(samples),
            }

    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    return {
        "schema": "mlx-uag.qwen4-sparse-gather-rhs-micro.v1",
        "arm": args.arm,
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "mlx_version": mx.__version__,
            "rhs_min_density": os.environ.get(
                "MLX_GATHER_QMM_RHS_MIN_DENSITY", "<default:4>"
            ),
        },
        "geometry": {"tokens": 7, "top_k": 10, "experts": 512},
        "sampling": {
            "warmups": args.warmups,
            "trials": args.trials,
            "inner": args.inner,
            "expert_offsets": [0, 73, 149, 227, 311, 397],
        },
        "cells": cells,
        "arrays": str(args.arrays.resolve()),
        "memory": {
            "active_gb": mx.get_active_memory() / 1e9,
            "peak_gb": mx.get_peak_memory() / 1e9,
            "cache_gb": mx.get_cache_memory() / 1e9,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--inner", type=int, default=4)
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
