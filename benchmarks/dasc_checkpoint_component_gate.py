#!/usr/bin/env python3
"""Component gate for decay-aware GDN checkpoint packing.

This deliberately does not load the model.  It reads the tiny BF16 decay
parameters directly from their safetensors byte ranges, derives the DASC keep
mask, and benchmarks the exact checkpoint geometry used by Qwen4 ArraysCache.
The packed restore is approximate because omitted heads are zero-filled.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


TENSOR_RE = re.compile(
    r"language_model\.model\.layers\.(\d+)\.linear_attn\.(A_log|dt_bias)$"
)


def _bf16_tensor(path: Path, name: str) -> np.ndarray:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        spec = header[name]
        if spec["dtype"] != "BF16":
            raise ValueError(f"{name} is {spec['dtype']}, expected BF16")
        lo, hi = spec["data_offsets"]
        handle.seek(8 + header_len + lo)
        raw = handle.read(hi - lo)
    words = np.frombuffer(raw, dtype="<u2")
    return (words.astype(np.uint32) << 16).view(np.float32).reshape(spec["shape"])


def _keep_masks(model: Path, horizon: int, gate_logit: float, epsilon: float):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    by_layer: dict[int, dict[str, np.ndarray]] = {}
    for name, shard in weight_map.items():
        match = TENSOR_RE.fullmatch(name)
        if match is None:
            continue
        layer, field = int(match.group(1)), match.group(2)
        by_layer.setdefault(layer, {})[field] = _bf16_tensor(model / shard, name)
    masks = []
    horizons = []
    for layer in sorted(by_layer):
        values = by_layer[layer]
        if set(values) != {"A_log", "dt_bias"}:
            raise ValueError(f"layer {layer} has incomplete decay parameters")
        softplus = np.logaddexp(0.0, gate_logit + values["dt_bias"])
        g = -np.exp(values["A_log"]) * softplus
        hs = math.log(epsilon) / g
        keep = np.flatnonzero(hs > horizon).astype(np.int32)
        masks.append((layer, keep))
        horizons.extend(float(v) for v in hs)
    if not masks:
        raise ValueError("no Qwen GDN decay tensors found")
    return masks, horizons


def _measure(fn, warmups: int, repetitions: int):
    times = []
    result = None
    for index in range(warmups + repetitions):
        start = time.perf_counter_ns()
        result = fn()
        mx.eval(*result)
        mx.synchronize()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if index >= warmups:
            times.append(elapsed_ms)
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "samples_ms": times,
    }, result


def _time_call(fn):
    start = time.perf_counter_ns()
    result = fn()
    mx.eval(*result)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def _measure_abba(dense_fn, dasc_fn, warmups: int, repetitions: int):
    """Interleave whole round trips so drift cannot choose the winner."""
    rows = []
    for bracket in range(warmups + repetitions):
        order = ("dense", "dasc", "dasc", "dense")
        samples = {"dense": [], "dasc": []}
        for lane in order:
            samples[lane].append(_time_call(dense_fn if lane == "dense" else dasc_fn))
        if bracket >= warmups:
            rows.append(
                {
                    "dense_ms": statistics.median(samples["dense"]),
                    "dasc_ms": statistics.median(samples["dasc"]),
                }
            )
    dense = [row["dense_ms"] for row in rows]
    dasc = [row["dasc_ms"] for row in rows]
    return {
        "order": ["dense", "dasc", "dasc", "dense"],
        "brackets": rows,
        "dense_median_ms": statistics.median(dense),
        "dasc_median_ms": statistics.median(dasc),
        "dasc_over_dense": statistics.median(dense) / statistics.median(dasc),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--gate-logit", type=float, default=-0.3)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=15)
    args = parser.parse_args()

    masks, horizons = _keep_masks(
        args.model, args.horizon, args.gate_logit, args.epsilon
    )
    # Qwen4 GDN recurrent state: B=1, Hv=48, Dv=Dk=128, fp32.
    states = [
        mx.full((1, 48, 128, 128), float(index + 1), dtype=mx.float32)
        for index, _ in enumerate(masks)
    ]
    mx.eval(*states)
    indices = [mx.array(keep) for _, keep in masks]

    dense_pack, dense = _measure(
        lambda: [mx.array(state) for state in states],
        args.warmups,
        args.repetitions,
    )
    packed_pack, packed = _measure(
        lambda: [mx.take(state, idx, axis=1) for state, idx in zip(states, indices)],
        args.warmups,
        args.repetitions,
    )
    dense_restore, restored_dense = _measure(
        lambda: [mx.array(state) for state in dense],
        args.warmups,
        args.repetitions,
    )
    packed_restore, restored_packed = _measure(
        lambda: [
            mx.zeros_like(state).at[:, idx].add(part)
            for state, idx, part in zip(states, indices, packed)
        ],
        args.warmups,
        args.repetitions,
    )

    def dense_roundtrip():
        snapshot = [mx.array(state) for state in states]
        mx.eval(*snapshot)
        return [mx.array(state) for state in snapshot]

    def dasc_roundtrip():
        parts = [
            mx.take(state, idx, axis=1) for state, idx in zip(states, indices)
        ]
        mx.eval(*parts)
        return [
            mx.zeros_like(state).at[:, idx].add(part)
            for state, idx, part in zip(states, indices, parts)
        ]

    abba = _measure_abba(
        dense_roundtrip,
        dasc_roundtrip,
        args.warmups,
        args.repetitions,
    )

    retained = sum(int(idx.size) for idx in indices)
    total = len(indices) * 48
    dense_bytes = sum(state.nbytes for state in states)
    packed_bytes = sum(part.nbytes for part in packed)
    # Retained heads must be exact; omitted heads are deliberately zero-filled.
    retained_exact = True
    omitted_zero = True
    for original, idx, restored in zip(states, indices, restored_packed):
        mx.eval(restored)
        if idx.size:
            retained_exact &= bool(mx.all(mx.take(restored, idx, axis=1) == mx.take(original, idx, axis=1)).item())
        keep_host = set(int(v) for v in idx.tolist())
        omitted = mx.array([i for i in range(48) if i not in keep_host])
        if omitted.size:
            omitted_zero &= bool(mx.all(mx.take(restored, omitted, axis=1) == 0).item())
    dense_exact = all(
        bool(mx.all(a == b).item()) for a, b in zip(states, restored_dense)
    )

    report = {
        "schema": "mlx-uag.dasc-checkpoint-component.v1",
        "device": str(mx.default_device()),
        "model": str(args.model),
        "horizon": args.horizon,
        "gate_logit": args.gate_logit,
        "epsilon": args.epsilon,
        "geometry": {
            "layers": len(states),
            "heads_per_layer": 48,
            "head_shape": [128, 128],
            "dtype": "float32",
            "dense_bytes": dense_bytes,
            "packed_bytes": packed_bytes,
            "retained_heads": retained,
            "total_heads": total,
            "retained_fraction": retained / total,
            "max_checkpoints_default": 4,
            "dense_default_capacity_bytes": dense_bytes * 4,
            "packed_default_capacity_bytes": packed_bytes * 4,
        },
        "derived_horizons": {
            "minimum": min(horizons),
            "median": statistics.median(horizons),
            "maximum": max(horizons),
        },
        "timing": {
            "dense_pack": dense_pack,
            "dasc_pack": packed_pack,
            "dense_restore": dense_restore,
            "dasc_zero_fill_restore": packed_restore,
            "dense_roundtrip_median_ms": dense_pack["median_ms"] + dense_restore["median_ms"],
            "dasc_roundtrip_median_ms": packed_pack["median_ms"] + packed_restore["median_ms"],
            "interleaved_abba_roundtrip": abba,
        },
        "correctness": {
            "dense_exact": dense_exact,
            "dasc_retained_heads_exact": retained_exact,
            "dasc_omitted_heads_zero": omitted_zero,
            "dasc_is_lossless": retained == total,
        },
        "verdict_scope": "component-only; zero-filled DASC restore is approximate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
