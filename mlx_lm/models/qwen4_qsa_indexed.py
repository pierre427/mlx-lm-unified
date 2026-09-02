"""Default-off indexed split-K QSA attention for short verify widths."""

from __future__ import annotations

import math
import os
import threading
from collections import Counter
from functools import lru_cache
from typing import Any

import mlx.core as mx
import numpy as np

from .qwen4_qsa_nax import compact_blocks_to_kernel_inputs


_BLOCK_SIZE = 4
_CHUNK_SLOTS = 64
_CHUNK_TOKENS = _CHUNK_SLOTS * _BLOCK_SIZE
_THREAD_CANDIDATES = (256, 128, 64)


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


_QSA_INDEXED_ENABLED = _env_flag("MLX_QWEN4_QSA_INDEXED")
_MIN_QUERY = _env_int("MLX_QWEN4_QSA_INDEXED_MIN_QUERY", 2, minimum=1)
_MAX_QUERY = _env_int("MLX_QWEN4_QSA_INDEXED_MAX_QUERY", 8, minimum=1)
_MIN_CONTEXT = _env_int("MLX_QWEN4_QSA_INDEXED_MIN_CONTEXT", 16384)
_MAX_CONTEXT = _env_int("MLX_QWEN4_QSA_INDEXED_MAX_CONTEXT", 0)
_SPLITS_OVERRIDE = _env_int("MLX_QWEN4_QSA_INDEXED_SPLITS", 0)


def indexed_splits_for(u_width: int) -> int:
    """Return the static split count for a compact block-slot width."""

    width = int(u_width)
    if width < 1:
        raise ValueError("u_width must be positive")
    requested = _SPLITS_OVERRIDE
    if requested == 0:
        requested = min(8, max(1, math.ceil(width * _BLOCK_SIZE / 256)))
    if width * _BLOCK_SIZE >= 512:
        requested = max(4, requested)
    return min(width, 8, requested)


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
    if count < 1 or count > min(8, int(u_width)):
        raise ValueError("splits must be in [1, min(8, u_width)]")
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

    if not _QSA_INDEXED_ENABLED:
        return False, "disabled"
    if training:
        return False, "training"
    if selection.kind != "explicit":
        return False, "selection_not_explicit"
    topk = _selection_topk_width(selection)
    if topk and int(selection.n_blocks) <= topk:
        return False, "dense_by_construction"
    if int(length) < _MIN_QUERY or int(length) > _MAX_QUERY:
        return False, "width_out_of_range"
    context = int(selection.physical_width)
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
        "fully_masked_output": "zero",
    }
    with _STATUS_LOCK:
        _STATUS_COUNTS[reason] += 1
        _STATUS_WIDTHS[_width_bucket(int(length))][outcome] += 1
        if reason in {"probe_declined", "dispatch_raised"}:
            _STATUS_FALLBACKS += 1
        if candidate is not None:
            _STATUS_CANDIDATE = tuple(candidate)
        _STATUS_LAST = receipt


def qsa_indexed_status(*, reset: bool = False) -> dict[str, Any]:
    """Return indexed-QSA admission, candidate, and fallback evidence."""

    global _STATUS_CANDIDATE, _STATUS_FALLBACKS, _STATUS_LAST
    with _STATUS_LOCK:
        report = {
            "enabled": bool(_QSA_INDEXED_ENABLED),
            "min_query_width": _MIN_QUERY,
            "max_query_width": _MAX_QUERY,
            "min_context": _MIN_CONTEXT,
            "max_context": _MAX_CONTEXT,
            "splits_override": _SPLITS_OVERRIDE,
            "counts": dict(_STATUS_COUNTS),
            "query_width_counts": {
                key: dict(value) for key, value in _STATUS_WIDTHS.items()
            },
            "candidate": (
                None if _STATUS_CANDIDATE is None else list(_STATUS_CANDIDATE)
            ),
            "fallbacks": _STATUS_FALLBACKS,
            "fully_masked_output": "zero",
            "last_decision": _STATUS_LAST,
        }
        if reset:
            _STATUS_COUNTS.clear()
            for value in _STATUS_WIDTHS.values():
                value.clear()
            _STATUS_CANDIDATE = None
            _STATUS_FALLBACKS = 0
            _STATUS_LAST = None
    return report


def set_qwen4_qsa_indexed(enabled: bool) -> bool:
    """Live-toggle indexed QSA without changing resident arrays."""

    global _QSA_INDEXED_ENABLED
    _QSA_INDEXED_ENABLED = bool(enabled)
    return _QSA_INDEXED_ENABLED


def qsa_indexed_enabled() -> bool:
    """Return the live indexed-QSA switch."""

    return bool(_QSA_INDEXED_ENABLED)


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


# TODO: add an optional fused second-pass hook after an isolated A/B.
def _combine_indexed_partials(m, l, o, *, output_dtype):
    """Merge fixed chunks in ascending split then chunk order.

    This is the hook point for a future fused second pass.
    """

    state_m = mx.full(m.shape[:3], -mx.inf, dtype=mx.float32)
    state_l = mx.zeros_like(state_m)
    state_o = mx.zeros(o.shape[:3] + (o.shape[-1],), dtype=mx.float32)
    for split in range(m.shape[-2]):
        for chunk in range(m.shape[-1]):
            chunk_m = m[..., split, chunk]
            chunk_l = l[..., split, chunk]
            chunk_o = o[..., split, chunk, :]
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
            merged_o = (
                state_o * alpha[..., None] + chunk_o * beta[..., None]
            )
            state_m = mx.where(chunk_live, merged_m, state_m)
            state_l = mx.where(chunk_live, merged_l, state_l)
            state_o = mx.where(chunk_live[..., None], merged_o, state_o)
    out = mx.where(
        (state_l > 0)[..., None],
        state_o
        / mx.maximum(state_l[..., None], mx.array(1.0e-30, mx.float32)),
        mx.zeros_like(state_o),
    )
    return out.astype(output_dtype)


def _reference_partials(q, k, v, compact, *, scale: float, splits: int):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError("indexed QSA wants matching rank-4 q/k/v tensors")
    batch, nqh, length, dim = map(int, q.shape)
    if k.shape[0] != batch or int(k.shape[2]) != int(compact.physical_width):
        raise ValueError("indexed QSA tensors do not match compact selection")
    nkh = int(k.shape[1])
    if nkh < 1 or nqh % nkh:
        raise ValueError("indexed QSA requires integral GQA")

    (
        ids,
        counts,
        _,
        u_width,
        _,
        _,
        _,
        physical,
        valid,
    ) = _compact_token_inputs(compact)
    _validate_no_duplicate_blocks(ids, counts)
    if splits < 1 or splits > u_width:
        raise ValueError("splits must be in [1, u_width]")

    q_rows = q.transpose(0, 2, 1, 3).astype(mx.float32)
    k_by_token = k.transpose(0, 2, 1, 3)
    v_by_token = v.transpose(0, 2, 1, 3)
    gqa = nqh // nkh
    head_map = mx.arange(nqh, dtype=mx.int32) // gqa
    split_chunks = indexed_split_chunk_ranges(u_width, splits)
    chunks_per_split = max(1, max(map(len, split_chunks)))
    split_ms = []
    split_ls = []
    split_os = []
    empty_m = mx.full((batch, nqh, length), -mx.inf, dtype=mx.float32)
    empty_l = mx.zeros_like(empty_m)
    empty_o = mx.zeros((batch, nqh, length, dim), dtype=mx.float32)
    for chunks in split_chunks:
        chunk_ms = []
        chunk_ls = []
        chunk_os = []
        for start, stop in chunks:
            token_index = physical[:, :, start:stop].reshape(batch, length, -1)
            token_valid = valid[:, :, start:stop].reshape(batch, length, -1)
            gather_index = token_index[..., None, None]
            gathered_k = mx.take_along_axis(
                k_by_token[:, None], gather_index, axis=2
            ).transpose(0, 1, 3, 2, 4)
            gathered_v = mx.take_along_axis(
                v_by_token[:, None], gather_index, axis=2
            ).transpose(0, 1, 3, 2, 4)
            gathered_k = mx.take(gathered_k, head_map, axis=2).astype(mx.float32)
            gathered_v = mx.take(gathered_v, head_map, axis=2).astype(mx.float32)
            scores = (
                mx.sum(q_rows[..., None, :] * gathered_k, axis=-1)
                * float(scale)
            )
            head_valid = token_valid[:, :, None, :]
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
                probabilities[..., None] * gathered_v, axis=-2
            )
            chunk_ms.append(part_m.transpose(0, 2, 1))
            chunk_ls.append(part_l.transpose(0, 2, 1))
            chunk_os.append(part_o.transpose(0, 2, 1, 3))
        while len(chunk_ms) < chunks_per_split:
            chunk_ms.append(empty_m)
            chunk_ls.append(empty_l)
            chunk_os.append(empty_o)
        split_ms.append(mx.stack(chunk_ms, axis=-1))
        split_ls.append(mx.stack(chunk_ls, axis=-1))
        split_os.append(mx.stack(chunk_os, axis=-2))
    return (
        mx.stack(split_ms, axis=-2),
        mx.stack(split_ls, axis=-2),
        mx.stack(split_os, axis=-3),
    )


def qwen4_qsa_indexed_reference(
    q, k, v, compact, *, scale: float, splits: int
):
    """MLX-ops mirror of fixed-chunk two-pass indexed attention."""

    m, l, o = _reference_partials(
        q, k, v, compact, scale=scale, splits=int(splits)
    )
    return _combine_indexed_partials(m, l, o, output_dtype=q.dtype)


_HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""


_SOURCE = r"""
    // Each group owns (batch, query row, KV head, split). Fixed chunks do
    // not depend on S. The MLX merge scans split then chunk in slot order.
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
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
    const uint chunks = (uint(U) + CHUNK_SLOTS - 1) / CHUNK_SLOTS;
    const uint base = chunks / S;
    const uint remainder = chunks % S;
    const uint chunk_begin = split * base + metal::min(split, remainder);
    const uint chunk_count = base + (split < remainder ? 1u : 0u);
    const uint nsg = THREADS / 32;

    threadgroup float dot_parts[GQA * (THREADS / 32)];
    threadgroup float scores[GQA * CHUNK_TOKENS];
    threadgroup float probabilities[GQA];
    threadgroup float maxima[GQA];
    threadgroup float sums[GQA];

    float q_values[GQA][D / THREADS + 1];
    float out[GQA][D / THREADS + 1];
    for (uint head = 0; head < GQA; ++head) {
        const uint qh = hkv * GQA + head;
        for (uint d = tid; d < D; d += THREADS) {
            const uint part = d / THREADS;
            q_values[head][part] = float(
                q[((size_t)(b * NQH + qh) * L + row) * D + d]
            );
        }
    }

    const uint slot_base = (b * L + row) * uint(U);
    const device T* kb = k + (size_t)(b * NKVH + hkv) * TOT * D;
    const device T* vb = v + (size_t)(b * NKVH + hkv) * TOT * D;
    for (uint local_chunk = 0; local_chunk < CPS; ++local_chunk) {
        for (uint head = 0; head < GQA; ++head)
            for (uint part = 0; part < D / THREADS + 1; ++part)
                out[head][part] = 0.0f;
        if (tid < GQA) {
            maxima[tid] = -INFINITY;
            sums[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_chunk < chunk_count) {
            const uint chunk = chunk_begin + local_chunk;
            const uint first_slot = chunk * CHUNK_SLOTS;
            const uint last_slot = metal::min(first_slot + CHUNK_SLOTS, uint(U));

            // Pass 1 stores every score and finds the chunk maximum.
            for (uint token = 0; token < CHUNK_TOKENS; ++token) {
                const uint slot = first_slot + token / BS;
                const uint tail = token % BS;
                int logical = 0;
                int physical = 0;
                bool live = slot < last_slot && slot < count;
                if (live) {
                    const int block = int(ids[slot_base + slot]);
                    logical = block * BS + int(tail);
                    physical = lpad + logical;
                    live = physical >= 0 && physical < TOT && logical <= qp;
                    live = live && (slot < selected || logical >= complete);
                    if (HAS_MASK && live)
                        live = mask[(size_t)(b * L + row) * TOT + physical];
                }
                if (!live) {
                    if (tid < GQA)
                        scores[tid * CHUNK_TOKENS + token] = -INFINITY;
                    continue;
                }

                float key_values[D / THREADS + 1];
                for (uint d = tid; d < D; d += THREADS) {
                    const uint part = d / THREADS;
                    key_values[part] = float(kb[(size_t)physical * D + d]);
                }
                for (uint head = 0; head < GQA; ++head) {
                    float local = 0.0f;
                    for (uint d = tid; d < D; d += THREADS) {
                        const uint part = d / THREADS;
                        local += q_values[head][part] * key_values[part];
                    }
                    local = simd_sum(local);
                    if (lane == 0)
                        dot_parts[head * nsg + sg] = local;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < GQA) {
                    float score = 0.0f;
                    for (uint part = 0; part < nsg; ++part)
                        score += dot_parts[tid * nsg + part];
                    score *= scale[0];
                    scores[tid * CHUNK_TOKENS + token] = score;
                    maxima[tid] = metal::max(maxima[tid], score);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Pass 2 accumulates in fixed slot-major, token-minor order.
            for (uint token = 0; token < CHUNK_TOKENS; ++token) {
                const uint slot = first_slot + token / BS;
                const uint tail = token % BS;
                int logical = 0;
                int physical = 0;
                bool live = slot < last_slot && slot < count;
                if (live) {
                    const int block = int(ids[slot_base + slot]);
                    logical = block * BS + int(tail);
                    physical = lpad + logical;
                    live = physical >= 0 && physical < TOT && logical <= qp;
                    live = live && (slot < selected || logical >= complete);
                    if (HAS_MASK && live)
                        live = mask[(size_t)(b * L + row) * TOT + physical];
                }
                if (!live) continue;

                float value_values[D / THREADS + 1];
                for (uint d = tid; d < D; d += THREADS) {
                    const uint part = d / THREADS;
                    value_values[part] = float(vb[(size_t)physical * D + d]);
                }
                if (tid < GQA) {
                    const float probability = metal::precise::exp(
                        scores[tid * CHUNK_TOKENS + token] - maxima[tid]
                    );
                    probabilities[tid] = probability;
                    sums[tid] += probability;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint d = tid; d < D; d += THREADS) {
                    const uint part = d / THREADS;
                    for (uint head = 0; head < GQA; ++head)
                        out[head][part] += probabilities[head] * value_values[part];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        for (uint head = 0; head < GQA; ++head) {
            const uint qh = hkv * GQA + head;
            const size_t state = (
                (((size_t)b * NQH + qh) * L + row) * S + split
            ) * CPS + local_chunk;
            if (tid == head) {
                part_m[state] = maxima[head];
                part_l[state] = sums[head];
            }
            for (uint d = tid; d < D; d += THREADS) {
                const uint part = d / THREADS;
                part_o[state * D + d] = out[head][part];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


@lru_cache(maxsize=None)
def _partition_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_qsa_indexed_splitk_v2",
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
        ensure_row_contiguous=True,
    )


class QSAIndexedProbeDeclined(RuntimeError):
    """No candidate in the indexed Metal ladder could dispatch."""


_PROBE_LOCK = threading.Lock()
_PROBE_RESULTS = {}
_MISSING = object()


def _candidate_ladder(splits: int):
    split_candidates = [int(splits)]
    if splits > 4:
        split_candidates.append(max(4, splits // 2))
    return tuple(
        (threads, candidate_splits)
        for candidate_splits in split_candidates
        for threads in _THREAD_CANDIDATES
    )


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
    chunks_per_split = max(
        1, max(map(len, indexed_split_chunk_ranges(u_width, splits)))
    )
    if compact.causal_mask is None:
        mask = mx.ones((1,), dtype=mx.bool_)
        has_mask = False
    else:
        mask = mx.broadcast_to(
            compact.causal_mask, (batch, 1, length, total)
        )[:, 0]
        has_mask = True
    return _partition_kernel()(
        inputs=[
            mx.contiguous(q),
            mx.contiguous(k),
            mx.contiguous(v),
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.astype(mx.uint32)),
            mx.contiguous(n_sel.astype(mx.uint32)),
            mx.contiguous(q_pos.astype(mx.int32)),
            mx.contiguous(left_pad.astype(mx.int32)),
            mx.contiguous(mask),
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
            ("CHUNK_SLOTS", _CHUNK_SLOTS),
            ("CHUNK_TOKENS", _CHUNK_TOKENS),
            ("CPS", chunks_per_split),
            ("THREADS", int(threads)),
            ("HAS_MASK", int(has_mask)),
        ],
        grid=(threads, length, batch * nkh * splits),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, nqh, length, splits, chunks_per_split),
            (batch, nqh, length, splits, chunks_per_split),
            (batch, nqh, length, splits, chunks_per_split, dim),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


def qwen4_qsa_indexed_attention(
    q, k, v, compact, *, scale: float, splits: int | None = None
):
    """Dispatch indexed split-K Metal attention and merge partials in MLX."""

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

    _, _, _, u_width, _, _, _ = compact_blocks_to_kernel_inputs(compact)
    requested = indexed_splits_for(u_width) if splits is None else int(splits)
    if requested < 1 or requested > min(8, u_width):
        raise ValueError("splits must be in [1, min(8, u_width)]")
    key = (
        str(q.dtype),
        int(q.shape[-1]),
        int(q.shape[1]),
        int(k.shape[1]),
        int(compact.block_size),
        int(u_width),
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
                candidate = None
                for attempted in _candidate_ladder(requested):
                    try:
                        partials = _partition_dispatch(
                            q,
                            k,
                            v,
                            compact,
                            scale=scale,
                            threads=attempted[0],
                            splits=attempted[1],
                        )
                        mx.eval(*partials)
                        candidate = attempted
                        _PROBE_RESULTS[key] = attempted
                        break
                    except (RuntimeError, ValueError):
                        continue
                if candidate is None:
                    _PROBE_RESULTS[key] = False
                    raise QSAIndexedProbeDeclined(
                        "indexed QSA candidate ladder was declined"
                    )
                m, l, o = partials
                record_qsa_indexed_receipt(
                    engaged=True,
                    reason="engaged",
                    length=int(q.shape[2]),
                    context=int(compact.physical_width),
                    splits=candidate[1],
                    candidate=candidate,
                )
                return _combine_indexed_partials(m, l, o, output_dtype=q.dtype)

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
    )
    return _combine_indexed_partials(m, l, o, output_dtype=q.dtype)


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
