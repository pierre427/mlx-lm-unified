#!/usr/bin/env python3
"""Real-layer rejection gate for the default-off Qwen4 S=7 expert union.

The driver lazily loads one deployed MoE layer, then compares the isolated
two-dispatch union component with the stock sorted gather-QMV graph over the
same S=7 routes.  It requires raw-bit output parity before reporting timing.
It does not modify model dispatch or production defaults.  ``--allow-nonexact``
exists only to time a rejected candidate and never changes the failed verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models import qwen4_s7_expert_union as s7_union
from mlx_lm.models.qwen3_next import FusedGateUpSwitchGLU
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
from mlx_lm.utils import load


MODEL = Path(
    "/System/Volumes/Data/Users/pierrelamy/mlx-models/"
    "Qwen3.8-Flash-Next-MLX-4bit-MTP"
)


def _require_idle_service() -> None:
    with socket.socket() as sock:
        sock.settimeout(0.1)
        if sock.connect_ex(("127.0.0.1", 8282)) == 0:
            raise RuntimeError("port 8282 is live; stop the resident model first")


def _routes(unique_count: int, offset: int = 0) -> np.ndarray:
    """Deterministic seven-row routes with exactly ``unique_count`` experts."""

    if not 10 <= unique_count <= 70:
        raise ValueError(unique_count)
    rows: list[list[int]] = [list(range(10))]
    next_new = 10
    remaining = unique_count - 10
    for query in range(1, 7):
        later_capacity = (6 - query) * 10
        introduce = min(10, max(0, remaining - later_capacity))
        row = list(range(next_new, next_new + introduce))
        next_new += introduce
        remaining -= introduce
        candidate = query
        while len(row) < 10:
            expert = candidate % max(next_new, 10)
            candidate += 7
            if expert not in row:
                row.append(expert)
        rows.append(row)
    routes = (np.asarray(rows, dtype=np.uint32) + offset) % 512
    if len(np.unique(routes)) != unique_count:
        raise AssertionError((unique_count, len(np.unique(routes))))
    if any(len(np.unique(row)) != 10 for row in routes):
        raise AssertionError("one query contains a duplicate expert")
    return routes


def _strided_routes(routes: np.ndarray) -> tuple[mx.array, mx.array]:
    """Reproduce router views whose row stride is 512 rather than top_k."""

    index_pad = np.zeros((1, 7, 512), dtype=np.uint32)
    index_pad[..., -10:] = routes[None]
    score_pad = np.zeros((1, 7, 512), dtype=np.float32)
    base = np.arange(1, 11, dtype=np.float32)
    base /= base.sum()
    score_pad[..., -10:] = base[None, None]
    return (
        mx.array(index_pad)[..., -10:],
        mx.array(score_pad).astype(mx.bfloat16)[..., -10:],
    )


def _input(offset: int) -> mx.array:
    base = mx.arange(7 * 2560, dtype=mx.float32).reshape(1, 7, 2560)
    return (
        mx.sin(base * (0.0011 + offset * 1e-7))
        + 0.25 * mx.cos(base * (0.0007 + offset * 1e-7))
    ).astype(mx.bfloat16)


def _parts(projection):
    return (
        projection["weight"],
        projection["scales"],
        projection.biases,
    )


def _stock(block, x, indices, scores):
    """The stock sorted gate-up, SwiGLU, down, unsort, and slot reduction."""

    switch = block.switch_mlp
    expanded = mx.expand_dims(x, (-2, -3))
    routed_shape = indices.shape
    expanded, sorted_indices, inverse = _gather_sort(expanded, indices)
    gate_up = switch.gate_up_proj(
        expanded, sorted_indices, sorted_indices=True
    )
    half = switch.hidden_dims
    hidden = switch.activation(gate_up[..., half:], gate_up[..., :half])
    rows = switch.down_proj(hidden, sorted_indices, sorted_indices=True)
    rows = _scatter_unsort(rows, inverse, routed_shape).squeeze(-2)
    return (rows * scores[..., None]).sum(axis=-2)


def _stock_hidden(block, x, indices):
    """Return stock SwiGLU rows restored to original query/slot order."""

    switch = block.switch_mlp
    expanded = mx.expand_dims(x, (-2, -3))
    routed_shape = indices.shape
    expanded, sorted_indices, inverse = _gather_sort(expanded, indices)
    gate_up = switch.gate_up_proj(
        expanded, sorted_indices, sorted_indices=True
    )
    half = switch.hidden_dims
    hidden = switch.activation(gate_up[..., half:], gate_up[..., :half])
    return _scatter_unsort(hidden, inverse, routed_shape).squeeze(-2)


def _stock_gate_up(block, x, indices):
    switch = block.switch_mlp
    expanded = mx.expand_dims(x, (-2, -3))
    routed_shape = indices.shape
    expanded, sorted_indices, inverse = _gather_sort(expanded, indices)
    gate_up = switch.gate_up_proj(
        expanded, sorted_indices, sorted_indices=True
    )
    return _scatter_unsort(gate_up, inverse, routed_shape).squeeze(-2)


def _union_hidden(block, x, indices):
    switch = block.switch_mlp
    weight, scales, biases = _parts(switch.gate_up_proj)
    return s7_union._get_e1_kernel()(
        inputs=[x, indices, weight, scales, biases],
        template=[("T", x.dtype)],
        grid=(32, 640, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(7, 10, 640)],
        output_dtypes=[x.dtype],
    )[0]


def _union_affine_half(block, x, indices, half):
    """Compile a diagnostic E1 that exposes one pre-activation affine half."""

    needle = "hidden[(query * TOPK + (slot1 - 1)) * EH + row] = value;"
    replacement = (
        "hidden[(query * TOPK + (slot1 - 1)) * EH + row] = "
        f"{half}_value;"
    )
    source = s7_union._E1_SOURCE.replace(needle, replacement)
    if source == s7_union._E1_SOURCE:
        raise AssertionError("diagnostic E1 source substitution did not match")
    kernel = mx.fast.metal_kernel(
        name=f"qwen4_s7_expert_union_e1_debug_{half}",
        input_names=[
            "x", "indices", "gate_up_weight", "gate_up_scales", "gate_up_biases"
        ],
        output_names=["hidden"],
        source=source,
        header=s7_union._QMV_HEADER,
        ensure_row_contiguous=False,
    )
    switch = block.switch_mlp
    weight, scales, biases = _parts(switch.gate_up_proj)
    return kernel(
        inputs=[x, indices, weight, scales, biases],
        template=[("T", x.dtype)],
        grid=(32, 640, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(7, 10, 640)],
        output_dtypes=[x.dtype],
    )[0]


def _union(block, x, indices, scores):
    switch = block.switch_mlp
    return s7_union.qwen4_s7_expert_union(
        x,
        indices,
        scores,
        *_parts(switch.gate_up_proj),
        *_parts(switch.down_proj),
        enabled=True,
    )


def _digest(value: mx.array) -> str:
    widened = np.asarray(value.astype(mx.float32), dtype=np.float32)
    return hashlib.sha256(widened.tobytes()).hexdigest()


def _summary(values: list[float]) -> dict:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def _time(fn, variants, warmups: int, trials: int, inner: int) -> list[float]:
    for i in range(warmups):
        mx.eval(fn(*variants[i % len(variants)]))
    mx.synchronize()
    samples = []
    for trial in range(trials):
        started = time.perf_counter_ns()
        for j in range(inner):
            mx.eval(fn(*variants[(trial * inner + j) % len(variants)]))
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6 / inner)
    return samples


def run(args: argparse.Namespace) -> dict:
    _require_idle_service()
    if not mx.metal.is_available():
        raise RuntimeError("Metal GPU is unavailable")
    mx.set_default_device(mx.gpu)
    mx.reset_peak_memory()

    model, _ = load(str(args.model), lazy=True)
    model.eval()
    block = model.language_model.model.layers[args.layer].mlp
    if not isinstance(block.switch_mlp, FusedGateUpSwitchGLU):
        raise RuntimeError(
            "the layer does not expose the resident fused gate/up layout; "
            "run with MLX_QWEN4_MOE_FUSED_GATE_UP=1"
        )

    offsets = (0, 73, 149, 227)
    variants = []
    for offset in offsets:
        x = _input(offset)
        indices, scores = _strided_routes(_routes(args.unique_experts, offset))
        mx.eval(x, indices, scores)
        variants.append((block, x, indices, scores))

    reference = _stock(*variants[0])
    candidate = _union(*variants[0])
    mx.eval(reference, candidate)
    bit_exact = bool(mx.array_equal(reference, candidate))
    max_abs = float(
        mx.max(mx.abs(reference.astype(mx.float32) - candidate.astype(mx.float32))).item()
    )
    exactness_diagnostic = None
    if not bit_exact:
        stock_hidden = _stock_hidden(block, variants[0][1], variants[0][2])
        union_hidden = _union_hidden(block, variants[0][1], variants[0][2])
        mx.eval(stock_hidden, union_hidden)
        hidden_max_abs = float(
            mx.max(
                mx.abs(
                    stock_hidden.reshape(union_hidden.shape).astype(mx.float32)
                    - union_hidden.astype(mx.float32)
                )
            ).item()
        )
        hidden_bit_exact = bool(
            mx.array_equal(stock_hidden.reshape(union_hidden.shape), union_hidden)
        )
        stock_gate_up = _stock_gate_up(block, variants[0][1], variants[0][2])
        half = block.switch_mlp.hidden_dims
        union_gate = _union_affine_half(
            block, variants[0][1], variants[0][2], "gate"
        )
        union_up = _union_affine_half(
            block, variants[0][1], variants[0][2], "up"
        )
        stock_gate = stock_gate_up[..., :half].reshape(union_gate.shape)
        stock_up = stock_gate_up[..., half:].reshape(union_up.shape)
        mx.eval(stock_gate, stock_up, union_gate, union_up)
        gate_max_abs = float(
            mx.max(mx.abs(stock_gate.astype(mx.float32) - union_gate.astype(mx.float32))).item()
        )
        up_max_abs = float(
            mx.max(mx.abs(stock_up.astype(mx.float32) - union_up.astype(mx.float32))).item()
        )
        exactness_diagnostic = {
            "hidden_bit_exact": hidden_bit_exact,
            "hidden_max_abs": hidden_max_abs,
            "gate_bit_exact": bool(mx.array_equal(stock_gate, union_gate)),
            "gate_max_abs": gate_max_abs,
            "up_bit_exact": bool(mx.array_equal(stock_up, union_up)),
            "up_max_abs": up_max_abs,
        }
        if not args.allow_nonexact:
            raise AssertionError(
                "S=7 union changed the stock component output: "
                f"max_abs={max_abs}, diagnostic={exactness_diagnostic}"
            )

    pattern = ("stock", "union", "union", "stock")
    timings: dict[str, list[float]] = {"stock": [], "union": []}
    functions = {"stock": _stock, "union": _union}
    for arm in pattern:
        timings[arm].extend(
            _time(
                functions[arm],
                variants,
                args.warmups,
                args.trials,
                args.inner,
            )
        )
    stock_ms = statistics.median(timings["stock"])
    union_ms = statistics.median(timings["union"])
    return {
        "schema": "mlx-uag.qwen4-s7-expert-union-micro.v1",
        "passed": bit_exact,
        "model": str(args.model.resolve()),
        "layer": args.layer,
        "geometry": {
            "tokens": 7,
            "top_k": 10,
            "experts": 512,
            "hidden": 2560,
            "expert_hidden": 640,
            "unique_experts": args.unique_experts,
        },
        "exactness": {
            "raw_bit_equal": bit_exact,
            "max_abs": max_abs,
            "diagnostic": exactness_diagnostic,
            "stock_sha256_f32": _digest(reference),
            "union_sha256_f32": _digest(candidate),
        },
        "timing": {
            "pattern": list(pattern),
            "stock": _summary(timings["stock"]),
            "union": _summary(timings["union"]),
            "speedup": stock_ms / union_ms,
        },
        "receipts": s7_union.qwen4_s7_expert_union_status(),
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "mlx_version": mx.__version__,
        },
        "memory": {
            "active_gb": mx.get_active_memory() / 1e9,
            "peak_gb": mx.get_peak_memory() / 1e9,
            "cache_gb": mx.get_cache_memory() / 1e9,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--unique-experts", type=int, default=49)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--inner", type=int, default=2)
    parser.add_argument(
        "--allow-nonexact",
        action="store_true",
        help="Diagnostic timing only; result remains failed when parity is not raw-bit exact.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
