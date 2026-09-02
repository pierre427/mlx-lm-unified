"""Optional fused second-pass merge for indexed QSA partials."""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Callable

import mlx.core as mx

_ENV_NAME = "MLX_QWEN4_QSA_INDEXED_FUSED_MERGE"
_THREAD_CANDIDATES = (256, 128)

_STATUS_LOCK = threading.Lock()
_STATUS_ENGAGED = False
_STATUS_FALLBACKS = 0
_STATUS_CANDIDATE = None

_PROBE_LOCK = threading.Lock()
_PROBE_RESULTS = {}
_MISSING = object()


def fused_merge_enabled() -> bool:
    """Return the process environment switch for the fused merge."""

    raw = os.environ.get(_ENV_NAME)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{_ENV_NAME} must be 0/off or 1/on; got {raw!r}")


def fused_merge_available() -> bool:
    """Return whether the current device can dispatch the merge kernel."""

    return bool(
        hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
        and hasattr(mx, "metal")
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    )


def fused_merge_status(*, reset: bool = False) -> dict:
    """Return bounded process evidence for the optional merge."""

    global _STATUS_CANDIDATE, _STATUS_ENGAGED, _STATUS_FALLBACKS
    with _STATUS_LOCK:
        report = {
            "engaged": bool(_STATUS_ENGAGED),
            "fallbacks": int(_STATUS_FALLBACKS),
            "candidate": _STATUS_CANDIDATE,
        }
        if reset:
            _STATUS_ENGAGED = False
            _STATUS_FALLBACKS = 0
            _STATUS_CANDIDATE = None
    if reset:
        with _PROBE_LOCK:
            _PROBE_RESULTS.clear()
    return report


def _record_engaged(candidate: int) -> None:
    global _STATUS_CANDIDATE, _STATUS_ENGAGED
    with _STATUS_LOCK:
        _STATUS_ENGAGED = True
        _STATUS_CANDIDATE = int(candidate)


def _record_fallback() -> None:
    global _STATUS_ENGAGED, _STATUS_FALLBACKS
    with _STATUS_LOCK:
        _STATUS_ENGAGED = False
        _STATUS_FALLBACKS += 1


def _partial_geometry(m, l, o) -> tuple[int, int, int, int, int]:
    if m.ndim < 4 or l.shape != m.shape:
        raise ValueError("indexed QSA merge wants matching fp32 m/l partials")
    if o.ndim != m.ndim + 1 or o.shape[:-1] != m.shape:
        raise ValueError("indexed QSA merge O partials do not match m/l")
    if m.dtype != mx.float32 or l.dtype != mx.float32:
        raise ValueError("indexed QSA merge m/l partials must be fp32")
    batch, heads, length = map(int, m.shape[:3])
    partials = 1
    for width in m.shape[3:]:
        partials *= int(width)
    dim = int(o.shape[-1])
    if partials < 1 or dim < 1:
        raise ValueError("indexed QSA merge partial geometry must be non-empty")
    return batch, heads, length, partials, dim


def mlx_sequential_merge(m, l, o, *, output_dtype):
    """Merge partials in fixed row-major order with explicit fp32 MLX ops."""

    batch, heads, length, partials, dim = _partial_geometry(m, l, o)
    flat_m = m.reshape(batch, heads, length, partials)
    flat_l = l.reshape(batch, heads, length, partials)
    flat_o = o.reshape(batch, heads, length, partials, dim).astype(mx.float32)
    state_m = mx.full((batch, heads, length), -mx.inf, dtype=mx.float32)
    state_l = mx.zeros_like(state_m)
    state_o = mx.zeros((batch, heads, length, dim), dtype=mx.float32)
    for partial in range(partials):
        chunk_m = flat_m[..., partial]
        chunk_l = flat_l[..., partial]
        chunk_o = flat_o[..., partial, :]
        chunk_live = mx.isfinite(chunk_m)
        state_live = mx.isfinite(state_m)
        merged_m = mx.maximum(state_m, chunk_m)
        safe_m = mx.where(chunk_live, merged_m, mx.zeros_like(merged_m))
        alpha = mx.where(
            state_live,
            mx.exp(state_m - safe_m),
            mx.zeros_like(state_l),
        )
        beta = mx.where(
            chunk_live,
            mx.exp(chunk_m - safe_m),
            mx.zeros_like(chunk_l),
        )
        merged_l = state_l * alpha + chunk_l * beta
        merged_o = state_o * alpha[..., None] + chunk_o * beta[..., None]
        state_m = mx.where(chunk_live, merged_m, state_m)
        state_l = mx.where(chunk_live, merged_l, state_l)
        state_o = mx.where(chunk_live[..., None], merged_o, state_o)
    out = mx.where(
        (state_l > 0)[..., None],
        state_o / mx.maximum(state_l[..., None], mx.array(1.0e-30, mx.float32)),
        mx.zeros_like(state_o),
    )
    return out.astype(output_dtype)


_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
"""


_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const device float* row_m = part_m + (size_t)row * P;
    const device float* row_l = part_l + (size_t)row * P;
    const device O* row_o = part_o + (size_t)row * P * D;
    device TO* row_out = out + (size_t)row * D;

    threadgroup float alphas[P];
    threadgroup float betas[P];
    threadgroup uchar live[P];
    threadgroup float denominator;

    if (tid == 0) {
        float state_m = -INFINITY;
        float state_l = 0.0f;
        for (uint partial = 0; partial < P; ++partial) {
            const float chunk_m = row_m[partial];
            const float chunk_l = row_l[partial];
            const bool chunk_live = metal::isfinite(chunk_m);
            const bool state_live = metal::isfinite(state_m);
            const float merged_m = metal::max(state_m, chunk_m);
            const float safe_m = chunk_live ? merged_m : 0.0f;
            const float alpha = state_live
                ? metal::precise::exp(state_m - safe_m) : 0.0f;
            const float beta = chunk_live
                ? metal::precise::exp(chunk_m - safe_m) : 0.0f;
            // Preserve the separate MLX multiply and add rounding boundaries.
            volatile float state_l_product = state_l * alpha;
            volatile float chunk_l_product = chunk_l * beta;
            const float merged_l = state_l_product + chunk_l_product;
            alphas[partial] = alpha;
            betas[partial] = beta;
            live[partial] = chunk_live ? 1 : 0;
            if (chunk_live) {
                state_m = merged_m;
                state_l = merged_l;
            }
        }
        denominator = state_l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < D; d += THREADS) {
        float state_o = 0.0f;
        for (uint partial = 0; partial < P; ++partial) {
            volatile float state_o_product = state_o * alphas[partial];
            volatile float chunk_o_product =
                float(row_o[(size_t)partial * D + d]) * betas[partial];
            const float merged_o = state_o_product + chunk_o_product;
            if (live[partial])
                state_o = merged_o;
        }
        row_out[d] = denominator > 0.0f
            ? TO(state_o / metal::max(denominator, 1.0e-30f)) : TO(0.0f);
    }
"""


@lru_cache(maxsize=None)
def _fused_merge_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_fused_merge_v1",
        input_names=["part_m", "part_l", "part_o"],
        output_names=["out"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def _dispatch_candidate(m, l, o, *, output_dtype, threads: int):
    batch, heads, length, partials, dim = _partial_geometry(m, l, o)
    rows = batch * heads * length
    return _fused_merge_kernel()(
        inputs=[mx.contiguous(m), mx.contiguous(l), mx.contiguous(o)],
        template=[
            ("O", o.dtype),
            ("TO", output_dtype),
            ("D", dim),
            ("P", partials),
            ("THREADS", int(threads)),
        ],
        grid=(threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(batch, heads, length, dim)],
        output_dtypes=[output_dtype],
    )[0]


def _fused_merge(m, l, o, *, output_dtype):
    _, _, _, partials, dim = _partial_geometry(m, l, o)
    key = (str(o.dtype), str(output_dtype), partials, dim)
    candidate = _PROBE_RESULTS.get(key, _MISSING)
    if candidate is False:
        raise RuntimeError("indexed QSA fused merge candidate ladder declined")
    if candidate is _MISSING:
        with _PROBE_LOCK:
            candidate = _PROBE_RESULTS.get(key, _MISSING)
            if candidate is _MISSING:
                candidate = None
                for threads in _THREAD_CANDIDATES:
                    try:
                        output = _dispatch_candidate(
                            m,
                            l,
                            o,
                            output_dtype=output_dtype,
                            threads=threads,
                        )
                        mx.eval(output)
                        candidate = threads
                        _PROBE_RESULTS[key] = threads
                        break
                    except RuntimeError:
                        continue
                if candidate is None:
                    _PROBE_RESULTS[key] = False
                    raise RuntimeError(
                        "indexed QSA fused merge candidate ladder declined"
                    )
                _record_engaged(candidate)
                return output
    _record_engaged(candidate)
    return _dispatch_candidate(m, l, o, output_dtype=output_dtype, threads=candidate)


def combine_indexed_partials(
    m,
    l,
    o,
    *,
    output_dtype,
    on_fallback: Callable[[], None] | None = None,
):
    """Use the optional fused pass, with a sequential MLX fallback."""

    if not fused_merge_enabled() or not fused_merge_available():
        return mlx_sequential_merge(m, l, o, output_dtype=output_dtype)
    try:
        return _fused_merge(m, l, o, output_dtype=output_dtype)
    except RuntimeError:
        _record_fallback()
        if on_fallback is not None:
            on_fallback()
        return mlx_sequential_merge(m, l, o, output_dtype=output_dtype)


__all__ = [
    "combine_indexed_partials",
    "fused_merge_available",
    "fused_merge_enabled",
    "fused_merge_status",
    "mlx_sequential_merge",
]
