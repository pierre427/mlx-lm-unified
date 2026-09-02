"""Auto-default indexed split-K QSA attention for short verify widths."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import Counter
from functools import lru_cache
from typing import Any

import mlx.core as mx
import numpy as np

from .qwen4_qsa_nax import compact_blocks_to_kernel_inputs


_BLOCK_SIZE = 4
_CHUNK_SLOTS = 64
_CHUNK_TOKENS = _CHUNK_SLOTS * _BLOCK_SIZE
_SDPA_BLOCKS = 128
_SPLIT_CANDIDATES = (128, 64, 32, 16, 8)
_EXACT_MLX_BUILDS = frozenset({"0.32.2.dev20260829+334084ce9"})
_SDPA_VECTOR_HEADER_SHA256 = (
    "2100a4d1eaa8a524c5147c82c771cad75197495c72daffa03e7ea4c259aebf10"
)


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{name} must be 0/off or 1/on; got {raw!r}")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_mode(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value == "auto":
        return None
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{name} must be 0/off, 1/on, or auto; got {raw!r}")


_QSA_INDEXED_ENABLED = _env_mode("MLX_QWEN4_QSA_INDEXED")
_MIN_QUERY = _env_int("MLX_QWEN4_QSA_INDEXED_MIN_QUERY", 2, minimum=1)
_MAX_QUERY = _env_int("MLX_QWEN4_QSA_INDEXED_MAX_QUERY", 8, minimum=1)
_MIN_CONTEXT = _env_int("MLX_QWEN4_QSA_INDEXED_MIN_CONTEXT", 16384)
_MAX_CONTEXT = _env_int("MLX_QWEN4_QSA_INDEXED_MAX_CONTEXT", 0)
_AUTO_MIN_CONTEXT_M3 = _env_int(
    "MLX_QWEN4_QSA_INDEXED_AUTO_MIN_CONTEXT_M3", 16384
)
_AUTO_MIN_CONTEXT_M1 = _env_int(
    "MLX_QWEN4_QSA_INDEXED_AUTO_MIN_CONTEXT_M1", 65536
)
_SPLITS_OVERRIDE = _env_int("MLX_QWEN4_QSA_INDEXED_SPLITS", 0)


def _qsa_indexed_mode() -> str:
    if _QSA_INDEXED_ENABLED is None:
        return "auto"
    return "on" if _QSA_INDEXED_ENABLED else "off"


def _mlx_build_hash() -> str | None:
    version = str(getattr(mx, "__version__", "unknown"))
    marker = version.rsplit("+", 1)
    if len(marker) == 2 and marker[1]:
        return marker[1]
    return None


def indexed_splits_for(u_width: int) -> int:
    """Return the requested split ceiling for a compact block-slot width."""

    width = int(u_width)
    if width < 1:
        raise ValueError("u_width must be positive")
    requested = _SPLITS_OVERRIDE
    if requested == 0:
        requested = _SPLIT_CANDIDATES[0]
    if requested not in _SPLIT_CANDIDATES:
        allowed = ", ".join(map(str, reversed(_SPLIT_CANDIDATES)))
        raise ValueError(f"indexed QSA splits must be one of {allowed}")
    return requested


def indexed_chunk_ranges(u_width: int) -> tuple[tuple[int, int], ...]:
    """Return fixed slot chunks independent of the split count."""

    width = int(u_width)
    if width < 1:
        raise ValueError("u_width must be positive")
    return tuple(
        (start, min(start + _CHUNK_SLOTS, width))
        for start in range(0, width, _CHUNK_SLOTS)
    )


def indexed_split_chunk_ranges(
    u_width: int, splits: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Distribute fixed chunks over splits, with remainders first."""

    chunks = indexed_chunk_ranges(u_width)
    count = int(splits)
    if count < 1 or count > min(_SDPA_BLOCKS, int(u_width)):
        raise ValueError("splits must be in [1, min(128, u_width)]")
    base, remainder = divmod(len(chunks), count)
    groups = []
    start = 0
    for split in range(count):
        stop = start + base + (1 if split < remainder else 0)
        groups.append(chunks[start:stop])
        start = stop
    return tuple(groups)


def indexed_kernel_available() -> bool:
    """Return whether the current device can dispatch a Metal custom kernel."""

    return bool(
        hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
        and hasattr(mx, "metal")
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    )


def _selection_topk_width(selection) -> int:
    ids = getattr(selection, "raw_block_ids", None)
    if ids is None:
        return 0
    return int(ids.shape[-1])


def decide_qsa_indexed_admission(
    selection, *, length: int, training: bool, layout_ok: bool
) -> tuple[bool, str]:
    """Resolve the indexed route without evaluating arrays or changing state."""

    mode = _qsa_indexed_mode()
    if mode == "off":
        return False, "disabled"
    if training:
        return False, "training"
    if selection.kind != "explicit":
        return False, "selection_not_explicit"
    topk = _selection_topk_width(selection)
    if topk and int(selection.n_blocks) <= topk:
        return False, "dense_by_construction"
    length = int(length)
    context = int(selection.physical_width)
    if mode == "auto":
        if length < 1 or length > _MAX_QUERY:
            return False, "width_out_of_range"
        threshold = (
            _AUTO_MIN_CONTEXT_M1 if length == 1 else _AUTO_MIN_CONTEXT_M3
        )
        if context < threshold:
            return False, "auto_context_out_of_range"
        if _MAX_CONTEXT and context > _MAX_CONTEXT:
            return False, "context_out_of_range"
    else:
        if length < _MIN_QUERY or length > _MAX_QUERY:
            return False, "width_out_of_range"
        if context < _MIN_CONTEXT or (_MAX_CONTEXT and context > _MAX_CONTEXT):
            return False, "context_out_of_range"
    if not layout_ok:
        return False, "unsupported_layout"
    if not indexed_kernel_available():
        return False, "kernel_unavailable"
    return True, "engaged"


_STATUS_LOCK = threading.Lock()
_STATUS_COUNTS = Counter()
_STATUS_WIDTHS = {
    "1": Counter(),
    "2-8": Counter(),
    ">8": Counter(),
}
_STATUS_LAST = None
_STATUS_CANDIDATE = None
_STATUS_GEOMETRIES = {}
_STATUS_FALLBACKS = 0


def _width_bucket(width: int) -> str:
    if width == 1:
        return "1"
    if width <= 8:
        return "2-8"
    return ">8"


def record_qsa_indexed_receipt(
    *,
    engaged: bool,
    reason: str,
    length: int,
    context: int,
    splits: int | None = None,
    candidate: tuple[int, int] | None = None,
    exception_class: str | None = None,
    geometry_key: str | None = None,
    candidate_timings_ms: dict[int, float] | None = None,
) -> None:
    """Record bounded process evidence without evaluating device arrays."""

    global _STATUS_CANDIDATE, _STATUS_FALLBACKS, _STATUS_LAST
    outcome = "engaged" if engaged else "declined"
    receipt = {
        "engaged": bool(engaged),
        "reason": str(reason),
        "query_width": int(length),
        "physical_kv": int(context),
        "splits": None if splits is None else int(splits),
        "candidate": None if candidate is None else list(candidate),
        "exception_class": exception_class,
        "geometry_key": geometry_key,
        "candidate_timings_ms": (
            None
            if candidate_timings_ms is None
            else {
                str(split): float(elapsed)
                for split, elapsed in candidate_timings_ms.items()
            }
        ),
        "fully_masked_output": "zero",
    }
    with _STATUS_LOCK:
        _STATUS_COUNTS[reason] += 1
        _STATUS_WIDTHS[_width_bucket(int(length))][outcome] += 1
        if reason in {
            "mlx_build_unverified",
            "probe_declined",
            "dispatch_raised",
        }:
            _STATUS_FALLBACKS += 1
        if candidate is not None:
            _STATUS_CANDIDATE = tuple(candidate)
            if geometry_key is not None:
                _STATUS_GEOMETRIES[geometry_key] = {
                    "candidate": list(candidate),
                    "candidate_timings_ms": receipt["candidate_timings_ms"],
                }
        _STATUS_LAST = receipt


def qsa_indexed_status(*, reset: bool = False) -> dict[str, Any]:
    """Return indexed-QSA admission, candidate, and fallback evidence."""

    global _STATUS_CANDIDATE, _STATUS_FALLBACKS, _STATUS_LAST
    with _STATUS_LOCK:
        report = {
            "enabled": _qsa_indexed_mode() != "off",
            "mode": _qsa_indexed_mode(),
            "min_query_width": _MIN_QUERY,
            "max_query_width": _MAX_QUERY,
            "min_context": _MIN_CONTEXT,
            "max_context": _MAX_CONTEXT,
            "auto_min_context_m3": _AUTO_MIN_CONTEXT_M3,
            "auto_min_context_m1": _AUTO_MIN_CONTEXT_M1,
            "splits_override": _SPLITS_OVERRIDE,
            "split_candidates": list(_SPLIT_CANDIDATES),
            "mlx_version": str(getattr(mx, "__version__", "unknown")),
            "mlx_build_hash": _mlx_build_hash(),
            "mlx_build_verified": (
                str(getattr(mx, "__version__", "unknown"))
                in _EXACT_MLX_BUILDS
            ),
            "mlx_build_allow_unverified": (
                os.environ.get(
                    "MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX"
                )
                == "1"
            ),
            "counts": dict(_STATUS_COUNTS),
            "query_width_counts": {
                key: dict(value) for key, value in _STATUS_WIDTHS.items()
            },
            "candidate": (
                None if _STATUS_CANDIDATE is None else list(_STATUS_CANDIDATE)
            ),
            "geometry_candidates": dict(_STATUS_GEOMETRIES),
            "fallbacks": _STATUS_FALLBACKS,
            "fully_masked_output": "zero",
            "last_decision": _STATUS_LAST,
        }
        if reset:
            _STATUS_COUNTS.clear()
            for value in _STATUS_WIDTHS.values():
                value.clear()
            _STATUS_CANDIDATE = None
            _STATUS_GEOMETRIES.clear()
            _STATUS_FALLBACKS = 0
            _STATUS_LAST = None
    return report


def set_qwen4_qsa_indexed(enabled: bool | None) -> bool | None:
    """Live-toggle indexed QSA without changing resident arrays."""

    global _QSA_INDEXED_ENABLED
    _QSA_INDEXED_ENABLED = None if enabled is None else bool(enabled)
    return _QSA_INDEXED_ENABLED


def qsa_indexed_enabled() -> bool:
    """Return the live indexed-QSA switch."""

    return _qsa_indexed_mode() != "off"


def _compact_token_inputs(compact):
    ids, counts, n_sel, u_width, q_pos, left_pad, total = (
        compact_blocks_to_kernel_inputs(compact)
    )
    block_size = int(compact.block_size)
    logical = (
        ids.astype(mx.int32)[..., None] * block_size
        + mx.arange(block_size, dtype=mx.int32)
    )
    slots = mx.arange(u_width, dtype=mx.int32)[None, None, :, None]
    present = slots < counts.astype(mx.int32)[..., None, None]
    selected = slots < n_sel.astype(mx.int32)[..., None, None]
    tail = (logical >= compact.tail_start[..., None, None]) & (
        logical < compact.tail_stop[..., None, None]
    )
    valid = present & (selected | tail)
    physical = logical + left_pad[:, None, None, None]
    valid = valid & (physical >= 0) & (physical < total)
    valid = valid & (logical <= q_pos[..., None, None])
    physical = mx.clip(physical, 0, total - 1)
    if compact.causal_mask is not None:
        batch, length = ids.shape[:2]
        causal = mx.broadcast_to(
            compact.causal_mask, (batch, 1, length, total)
        )[:, 0]
        gathered = mx.take_along_axis(
            causal,
            physical.reshape(batch, length, -1),
            axis=-1,
        ).reshape(physical.shape)
        valid = valid & gathered
    return (
        ids,
        counts,
        n_sel,
        u_width,
        q_pos,
        left_pad,
        total,
        physical,
        valid,
    )


def _validate_no_duplicate_blocks(ids, counts) -> None:
    """Reject a malformed compact producer before it can double-count keys."""

    ids_np = np.asarray(ids)
    counts_np = np.asarray(counts)
    for index in np.ndindex(counts_np.shape):
        count = int(counts_np[index])
        row = ids_np[index][:count].tolist()
        if len(row) != len(set(row)):
            raise ValueError("compact QSA block ids must be unique per row")


def _reference_partials(q, k, v, compact, *, scale: float, splits: int):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError("indexed QSA wants matching rank-4 q/k/v tensors")
    batch, nqh, length, dim = map(int, q.shape)
    if k.shape[0] != batch or int(k.shape[2]) != int(compact.physical_width):
        raise ValueError("indexed QSA tensors do not match compact selection")
    nkh = int(k.shape[1])
    if nkh < 1 or nqh % nkh:
        raise ValueError("indexed QSA requires integral GQA")

    ids, counts, _, u_width, _, _, _, physical, valid = _compact_token_inputs(
        compact
    )
    _validate_no_duplicate_blocks(ids, counts)
    if splits < 1 or splits > u_width:
        raise ValueError("splits must be in [1, u_width]")

    token_width = u_width * int(compact.block_size)
    steps = math.ceil(token_width / _SDPA_BLOCKS)
    padded_width = steps * _SDPA_BLOCKS
    padding = padded_width - token_width
    physical = physical.reshape(batch, length, token_width)
    valid = valid.reshape(batch, length, token_width)
    if padding:
        physical = mx.pad(physical, [(0, 0), (0, 0), (0, padding)])
        valid = mx.pad(
            valid,
            [(0, 0), (0, 0), (0, padding)],
            constant_values=False,
        )
    physical = physical.reshape(
        batch, length, steps, _SDPA_BLOCKS
    ).transpose(0, 1, 3, 2)
    valid = valid.reshape(batch, length, steps, _SDPA_BLOCKS).transpose(
        0, 1, 3, 2
    )

    q_rows = q.transpose(0, 2, 1, 3).astype(mx.float32) * float(scale)
    k_by_token = k.transpose(0, 2, 1, 3)
    v_by_token = v.transpose(0, 2, 1, 3)
    gqa = nqh // nkh
    head_map = mx.arange(nqh, dtype=mx.int32) // gqa
    gather_index = physical[..., None, None]
    gathered_k = mx.take_along_axis(
        k_by_token[:, None, None], gather_index, axis=3
    )
    gathered_v = mx.take_along_axis(
        v_by_token[:, None, None], gather_index, axis=3
    )
    gathered_k = mx.take(gathered_k, head_map, axis=4).transpose(
        0, 1, 4, 2, 3, 5
    )
    gathered_v = mx.take(gathered_v, head_map, axis=4).transpose(
        0, 1, 4, 2, 3, 5
    )
    scores = mx.sum(
        q_rows[..., None, None, :] * gathered_k.astype(mx.float32), axis=-1
    )
    head_valid = valid[:, :, None]
    scores = mx.where(head_valid, scores, -mx.inf)
    part_m = mx.max(scores, axis=-1)
    live = mx.isfinite(part_m)
    safe_m = mx.where(live, part_m, mx.zeros_like(part_m))
    probabilities = mx.where(
        head_valid,
        mx.exp(scores - safe_m[..., None]),
        mx.zeros_like(scores),
    )
    part_l = mx.sum(probabilities, axis=-1)
    part_o = mx.sum(
        probabilities[..., None] * gathered_v.astype(mx.float32), axis=-2
    ).astype(q.dtype)
    return (
        part_m.transpose(0, 2, 1, 3),
        part_l.transpose(0, 2, 1, 3),
        part_o.transpose(0, 2, 1, 3, 4),
    )


def _combine_reference_sdpa_partials(m, l, o, *, output_dtype):
    live = mx.isfinite(m)
    state_m = mx.max(m, axis=-1)
    safe_m = mx.where(mx.isfinite(state_m), state_m, mx.zeros_like(state_m))
    factors = mx.where(
        live,
        mx.exp(m - safe_m[..., None]),
        mx.zeros_like(m),
    )
    denom = mx.sum(l * factors, axis=-1)
    numer = mx.sum(o.astype(mx.float32) * factors[..., None], axis=-2)
    return mx.where(
        (denom > 0)[..., None],
        numer / mx.maximum(denom[..., None], mx.array(1.0e-30, mx.float32)),
        mx.zeros_like(numer),
    ).astype(output_dtype)


def qwen4_qsa_indexed_reference(
    q, k, v, compact, *, scale: float, splits: int
):
    """MLX-ops mirror of fixed-chunk two-pass indexed attention."""

    m, l, o = _reference_partials(
        q, k, v, compact, scale=scale, splits=int(splits)
    )
    return _combine_reference_sdpa_partials(m, l, o, output_dtype=q.dtype)


_HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""


_SOURCE = r"""
    // Match MLX sdpa_vector_2pass_1 on the compact token order. Splits only
    // distribute the fixed 128 blocks; every block keeps its global index.
    const uint lane = thread_index_in_simdgroup;
    const uint head = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint unit = threadgroup_position_in_grid.z;
    const uint split = unit % S;
    const uint bkv = unit / S;
    const uint b = bkv / NKVH;
    const uint hkv = bkv % NKVH;

    const int L = dims[0];
    const int TOT = dims[1];
    const int U = dims[2];
    const uint count = counts[b * L + row];
    const uint selected = n_sel[b * L + row];
    const int qp = qpos[b * L + row];
    const int complete = ((qp + 1) / BS) * BS;
    const int lpad = left_pad[b];
    const uint base = BLOCKS / S;
    const uint remainder = BLOCKS % S;
    const uint block_begin = split * base + metal::min(split, remainder);
    const uint block_count = base + (split < remainder ? 1u : 0u);
    const uint token_width = uint(U) * BS;
    const uint qh = hkv * GQA + head;
    const uint elements = D / 32;

    float q_values[D / 32];
    for (uint part = 0; part < elements; ++part) {
        const uint d = lane * elements + part;
        const size_t q_index =
            (size_t)b * q_strides[0] +
            (size_t)qh * q_strides[1] +
            (size_t)row * q_strides[2] +
            (size_t)d * q_strides[3];
        q_values[part] = float(scale[0]) * float(q[q_index]);
    }

    const uint slot_base = (b * L + row) * uint(U);
    const size_t k_head =
        (size_t)b * k_strides[0] + (size_t)hkv * k_strides[1];
    const size_t v_head =
        (size_t)b * v_strides[0] + (size_t)hkv * v_strides[1];
    for (uint local_block = 0; local_block < block_count; ++local_block) {
        const uint block_idx = block_begin + local_block;
        float out_values[D / 32] = {0};
        float maximum = -3.402823466e+38F;
        float sum = 0.0f;

        for (uint token = block_idx; token < token_width; token += BLOCKS) {
            const uint slot = token / BS;
            const uint tail = token % BS;
            int logical = 0;
            int physical = 0;
            bool live = slot < count;
            if (live) {
                const int block = int(ids[slot_base + slot]);
                logical = block * BS + int(tail);
                physical = lpad + logical;
                live = physical >= 0 && physical < TOT && logical <= qp;
                live = live && (slot < selected || logical >= complete);
                if (HAS_MASK && live) {
                    const uint mask_batch = mask_shape[0] == 1 ? 0 : b;
                    const size_t mask_index =
                        (size_t)mask_batch * mask_strides[0] +
                        (size_t)row * mask_strides[2] +
                        (size_t)physical * mask_strides[3];
                    live = mask[mask_index];
                }
            }
            if (!live) continue;

            float score = 0.0f;
            for (uint part = 0; part < elements; ++part) {
                const uint d = lane * elements + part;
                const size_t k_index =
                    k_head + (size_t)physical * k_strides[2] +
                    (size_t)d * k_strides[3];
                score += q_values[part] * float(k[k_index]);
            }
            score = simd_sum(score);
            const float new_max = metal::max(maximum, score);
            const float factor = fast::exp(maximum - new_max);
            const float probability = fast::exp(score - new_max);
            maximum = new_max;
            sum = sum * factor + probability;
            for (uint part = 0; part < elements; ++part) {
                const uint d = lane * elements + part;
                out_values[part] = out_values[part] * factor
                    + probability * float(
                        v[v_head + (size_t)physical * v_strides[2] +
                          (size_t)d * v_strides[3]]
                    );
            }
        }

        const size_t state = (
            ((size_t)(b * NQH + qh) * L + row) * BLOCKS + block_idx
        );
        if (lane == 0) {
            part_m[state] = maximum;
            part_l[state] = sum;
        }
        for (uint part = 0; part < elements; ++part) {
            const uint d = lane * elements + part;
            part_o[state * D + d] = T(out_values[part]);
        }
    }
"""


_COMBINE_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint bh = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const uint elements = D / 32;
    const size_t state = ((size_t)bh * L + row) * BLOCKS;
    const device float* row_m = part_m + state;
    const device float* row_l = part_l + state;
    const device T* row_o = part_o + state * D;

    float maximum = -3.402823466e+38F;
    for (uint group = 0; group < BLOCKS / 32; ++group)
        maximum = metal::max(maximum, row_m[lane + 32 * group]);
    maximum = simd_max(maximum);

    float sum = 0.0f;
    for (uint group = 0; group < BLOCKS / 32; ++group) {
        const uint block = lane + 32 * group;
        sum += fast::exp(row_m[block] - maximum) * row_l[block];
    }
    sum = simd_sum(sum);

    float values[D / 32] = {0};
    for (uint group = 0; group < BLOCKS / 32; ++group) {
        const uint block = sg + 32 * group;
        const float factor = fast::exp(row_m[block] - maximum);
        for (uint part = 0; part < elements; ++part) {
            const uint d = lane * elements + part;
            values[part] += factor * float(row_o[(size_t)block * D + d]);
        }
    }

    threadgroup float transposed[32 * 32];
    for (uint part = 0; part < elements; ++part) {
        transposed[lane * 32 + sg] = values[part];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[part] = simd_sum(transposed[sg * 32 + lane]);
        values[part] = sum == 0.0f ? values[part] : values[part] / sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        device T* row_out = out + ((size_t)bh * L + row) * D + sg * elements;
        for (uint part = 0; part < elements; ++part)
            row_out[part] = T(values[part]);
    }
"""


@lru_cache(maxsize=None)
def _partition_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_sdpa_pass1_v3",
        input_names=[
            "q",
            "k",
            "v",
            "ids",
            "counts",
            "n_sel",
            "qpos",
            "left_pad",
            "mask",
            "scale",
            "dims",
        ],
        output_names=["part_m", "part_l", "part_o"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=None)
def _combine_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_sdpa_pass2_v3",
        input_names=["part_m", "part_l", "part_o", "dims"],
        output_names=["out"],
        header=_HEADER,
        source=_COMBINE_SOURCE,
        ensure_row_contiguous=True,
    )


class QSAIndexedProbeDeclined(RuntimeError):
    """No candidate in the indexed Metal ladder could dispatch."""

    def __init__(self, message: str, *, reason: str = "probe_declined"):
        super().__init__(message)
        self.reason = reason


_PROBE_LOCK = threading.Lock()
_PROBE_RESULTS = {}
_PROBE_TIMINGS = {}
_MISSING = object()


def _candidate_ladder(splits: int | None, threads: int):
    split_candidates = (
        _SPLIT_CANDIDATES if splits is None else (int(splits),)
    )
    return tuple((int(threads), value) for value in split_candidates)


def _measure_candidates(candidates, dispatch):
    """Compile viable candidates, then time one real dispatch for each."""

    viable = []
    for candidate in candidates:
        try:
            output = dispatch(candidate)
            mx.eval(output)
            viable.append(candidate)
        except RuntimeError:
            continue

    timings = {}
    outputs = {}
    for candidate in viable:
        try:
            started = time.perf_counter_ns()
            output = dispatch(candidate)
            mx.eval(output)
            elapsed = time.perf_counter_ns() - started
        except RuntimeError:
            continue
        timings[candidate[1]] = elapsed / 1.0e6
        outputs[candidate] = output
    if not timings:
        return None, None, {}
    selected = min(
        outputs,
        key=lambda candidate: (timings[candidate[1]], -candidate[1]),
    )
    return selected, outputs[selected], timings


def _partition_dispatch(
    q,
    k,
    v,
    compact,
    *,
    scale: float,
    threads: int,
    splits: int,
):
    batch, nqh, length, dim = map(int, q.shape)
    nkh = int(k.shape[1])
    gqa = nqh // nkh
    ids, counts, n_sel, u_width, q_pos, left_pad, total = (
        compact_blocks_to_kernel_inputs(compact)
    )
    required_threads = gqa * 32
    if threads != required_threads:
        raise ValueError("indexed QSA pass 1 requires one SIMD group per GQA head")
    if compact.causal_mask is None:
        mask = mx.ones((1, 1, 1, 1), dtype=mx.bool_)
        has_mask = False
    else:
        mask = compact.causal_mask
        if int(mask.shape[0]) == 1 and batch > 1:
            mask = mx.broadcast_to(mask, (batch, 1, length, total))
        has_mask = True
    return _partition_kernel()(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.astype(mx.uint32)),
            mx.contiguous(n_sel.astype(mx.uint32)),
            mx.contiguous(q_pos.astype(mx.int32)),
            mx.contiguous(left_pad.astype(mx.int32)),
            mask,
            mx.array([scale], dtype=mx.float32),
            mx.array([length, total, u_width], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("D", dim),
            ("NQH", nqh),
            ("NKVH", nkh),
            ("GQA", gqa),
            ("BS", int(compact.block_size)),
            ("S", int(splits)),
            ("BLOCKS", _SDPA_BLOCKS),
            ("HAS_MASK", int(has_mask)),
        ],
        grid=(threads, length, batch * nkh * splits),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS),
            (batch, nqh, length, _SDPA_BLOCKS, dim),
        ],
        output_dtypes=[mx.float32, mx.float32, q.dtype],
    )


def _combine_sdpa_partials(m, l, o, *, output_dtype):
    batch, nqh, length, blocks = map(int, m.shape)
    dim = int(o.shape[-1])
    if blocks != _SDPA_BLOCKS or dim % 32:
        raise ValueError("indexed QSA pass 2 has unsupported partial geometry")
    return _combine_kernel()(
        inputs=[m, l, o, mx.array([length], dtype=mx.int32)],
        template=[
            ("T", output_dtype),
            ("D", dim),
            ("BLOCKS", _SDPA_BLOCKS),
        ],
        grid=(1024, length, batch * nqh),
        threadgroup=(1024, 1, 1),
        output_shapes=[(batch, nqh, length, dim)],
        output_dtypes=[output_dtype],
    )[0]


def qwen4_qsa_indexed_attention(
    q, k, v, compact, *, scale: float, splits: int | None = None
):
    """Dispatch indexed attention with MLX SDPA's two-pass reduction tree."""

    mlx_version = str(getattr(mx, "__version__", "unknown"))
    allow_unverified = (
        os.environ.get("MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX") == "1"
    )
    if mlx_version not in _EXACT_MLX_BUILDS and not allow_unverified:
        raise QSAIndexedProbeDeclined(
            f"indexed QSA exactness is unproven on mlx {mlx_version}",
            reason="mlx_build_unverified",
        )
    if not indexed_kernel_available():
        raise QSAIndexedProbeDeclined("indexed QSA Metal runtime is unavailable")
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError("indexed QSA wants matching rank-4 q/k/v tensors")
    if q.shape[0] != k.shape[0] or q.shape[1] % k.shape[1]:
        raise ValueError("indexed QSA requires matching batch and integral GQA")
    if int(k.shape[2]) != int(compact.physical_width):
        raise ValueError("indexed QSA tensors do not match compact selection")
    if int(compact.block_size) != _BLOCK_SIZE:
        raise ValueError("indexed QSA requires block size 4")
    if int(q.shape[-1]) != 256:
        raise QSAIndexedProbeDeclined("indexed QSA exact mode requires D=256")
    gqa = int(q.shape[1]) // int(k.shape[1])
    if gqa != 12:
        raise QSAIndexedProbeDeclined("indexed QSA exact mode requires GQA=12")
    threads = gqa * 32
    if threads > 1024:
        raise ValueError("indexed QSA GQA exceeds the Metal threadgroup limit")
    sdpa_blocks = os.environ.get("MLX_SDPA_BLOCKS")
    if sdpa_blocks not in (None, "", str(_SDPA_BLOCKS)):
        raise QSAIndexedProbeDeclined(
            "indexed QSA requires MLX_SDPA_BLOCKS=128"
        )

    _, _, _, u_width, _, _, _ = compact_blocks_to_kernel_inputs(compact)
    token_width = u_width * _BLOCK_SIZE
    if token_width <= 1024 or token_width > 8192:
        raise QSAIndexedProbeDeclined(
            "indexed QSA requires the MLX two-pass SDPA geometry"
        )
    architecture = str(mx.device_info().get("architecture", ""))
    if architecture[-1:] not in {"s", "d"}:
        raise QSAIndexedProbeDeclined(
            "indexed QSA exact mode requires a 128-block MLX SDPA device"
        )
    requested = (
        None
        if splits is None and _SPLITS_OVERRIDE == 0
        else indexed_splits_for(u_width) if splits is None else int(splits)
    )
    if requested is not None and requested not in _SPLIT_CANDIDATES:
        allowed = ", ".join(map(str, reversed(_SPLIT_CANDIDATES)))
        raise ValueError(f"indexed QSA splits must be one of {allowed}")
    geometry_key = (
        f"B{int(q.shape[0])}-L{int(q.shape[2])}"
        f"-T{int(compact.physical_width)}-U{int(u_width)}"
        f"-mask{int(compact.causal_mask is not None)}"
    )
    key = (
        mlx_version,
        str(q.dtype),
        int(q.shape[-1]),
        int(q.shape[1]),
        int(k.shape[1]),
        int(compact.block_size),
        int(u_width),
        int(q.shape[0]),
        int(q.shape[2]),
        int(compact.physical_width),
        int(compact.causal_mask is not None),
        requested,
    )

    candidate = _PROBE_RESULTS.get(key, _MISSING)
    if candidate is False:
        raise QSAIndexedProbeDeclined("indexed QSA candidate ladder was declined")
    if candidate is _MISSING:
        with _PROBE_LOCK:
            candidate = _PROBE_RESULTS.get(key, _MISSING)
            if candidate is False:
                raise QSAIndexedProbeDeclined(
                    "indexed QSA candidate ladder was declined"
                )
            if candidate is _MISSING:
                def dispatch(attempted):
                    partials = _partition_dispatch(
                        q,
                        k,
                        v,
                        compact,
                        scale=scale,
                        threads=attempted[0],
                        splits=attempted[1],
                    )
                    return _combine_sdpa_partials(
                        *partials, output_dtype=q.dtype
                    )

                candidate, combined, timings = _measure_candidates(
                    _candidate_ladder(requested, threads), dispatch
                )
                if candidate is None:
                    _PROBE_RESULTS[key] = False
                    raise QSAIndexedProbeDeclined(
                        "indexed QSA candidate ladder was declined"
                    )
                _PROBE_RESULTS[key] = candidate
                _PROBE_TIMINGS[key] = timings
                record_qsa_indexed_receipt(
                    engaged=True,
                    reason="engaged",
                    length=int(q.shape[2]),
                    context=int(compact.physical_width),
                    splits=candidate[1],
                    candidate=candidate,
                    geometry_key=geometry_key,
                    candidate_timings_ms=timings,
                )
                return combined

    m, l, o = _partition_dispatch(
        q,
        k,
        v,
        compact,
        scale=scale,
        threads=candidate[0],
        splits=candidate[1],
    )
    record_qsa_indexed_receipt(
        engaged=True,
        reason="engaged",
        length=int(q.shape[2]),
        context=int(compact.physical_width),
        splits=candidate[1],
        candidate=candidate,
        geometry_key=geometry_key,
        candidate_timings_ms=_PROBE_TIMINGS.get(key),
    )
    return _combine_sdpa_partials(m, l, o, output_dtype=q.dtype)


__all__ = [
    "QSAIndexedProbeDeclined",
    "decide_qsa_indexed_admission",
    "indexed_chunk_ranges",
    "indexed_kernel_available",
    "indexed_split_chunk_ranges",
    "indexed_splits_for",
    "qsa_indexed_status",
    "qsa_indexed_enabled",
    "qwen4_qsa_indexed_attention",
    "qwen4_qsa_indexed_reference",
    "record_qsa_indexed_receipt",
    "set_qwen4_qsa_indexed",
]
