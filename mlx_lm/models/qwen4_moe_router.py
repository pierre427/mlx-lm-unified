"""One-dispatch direct-decode router for production Qwen4 sparse MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


NUM_EXPERTS = 512
TOP_K = 10


@dataclass(frozen=True)
class RouterAdmission:
    accepted: bool
    reason: str


def admit_qwen4_moe_router(gates, *, top_k: int, norm_topk_prob: bool):
    if gates.shape != (1, 1, NUM_EXPERTS):
        return RouterAdmission(False, "only B1/M1/512 experts")
    if gates.dtype != mx.bfloat16:
        return RouterAdmission(False, "gates must be bfloat16")
    if top_k != TOP_K or not norm_topk_prob:
        return RouterAdmission(False, "only normalized top-10 routing")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return RouterAdmission(False, "Metal GPU unavailable")
    return RouterAdmission(True, "eligible")


_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float local_max[32];
    threadgroup float local_sum[32];
    threadgroup T probabilities[512];

    float values[4];
    float vmax = -INFINITY;
    for (uint i = 0; i < 4; ++i) {
        values[i] = float(gates[tid * 4 + i]);
        vmax = metal::max(vmax, values[i]);
    }
    if (sg == 0) {
        local_max[lane] = -INFINITY;
        local_sum[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vmax = simd_max(vmax);
    if (lane == 0) local_max[sg] = vmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        vmax = simd_max(local_max[lane]);
        if (lane == 0) local_max[0] = vmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vmax = local_max[0];

    float normalizer = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        values[i] = metal::fast::exp(values[i] - vmax);
        normalizer += values[i];
    }
    normalizer = simd_sum(normalizer);
    if (lane == 0) local_sum[sg] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        normalizer = simd_sum(local_sum[lane]);
        if (lane == 0) local_sum[0] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1.0f / local_sum[0];
    for (uint i = 0; i < 4; ++i)
        probabilities[tid * 4 + i] = T(values[i] * normalizer);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) return;

    float topv[10];
    uint topi[10];
    for (uint j = 0; j < 10; ++j) {
        topv[j] = -INFINITY;
        topi[j] = 0;
    }

    // Maintain the ten largest values in ascending order. MLX argpartition's
    // selected suffix uses that same order for this production geometry.
    for (uint e = 0; e < 512; ++e) {
        float value = float(probabilities[e]);
        if (value < topv[0]) continue;
        uint pos = 0;
        while (pos < 10 && (value > topv[pos] ||
               (value == topv[pos] && e > topi[pos]))) ++pos;
        if (pos == 0) continue;
        for (uint j = 0; j + 1 < pos; ++j) {
            topv[j] = topv[j + 1];
            topi[j] = topi[j + 1];
        }
        topv[pos - 1] = value;
        topi[pos - 1] = e;
    }

    T probs[10];
    // MLX's length-10 BF16 row reduction accumulates in BF16, rounding after
    // every add. Mirror that order so the normalized router scores are exact.
    T selected_sum = T(0.0f);
    for (uint j = 0; j < 10; ++j) {
        probs[j] = probabilities[topi[j]];
        selected_sum = T(probs[j] + selected_sum);
    }
    for (uint j = 0; j < 10; ++j) {
        indices[j] = topi[j];
        scores[j] = T(float(probs[j]) / float(selected_sum));
    }
"""


_KERNEL = mx.fast.metal_kernel(
    name="qwen4_moe_router_b1_m1",
    input_names=["gates"],
    output_names=["indices", "scores"],
    header=_HEADER,
    source=_SOURCE,
    ensure_row_contiguous=True,
)


def qwen4_moe_router(gates):
    """Return ascending top-10 expert ids and normalized bf16 scores."""
    indices, scores = _KERNEL(
        inputs=[gates],
        template=[("T", gates.dtype)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, TOP_K), (1, 1, TOP_K)],
        output_dtypes=[mx.uint32, gates.dtype],
    )
    return indices, scores


_PROBE_COMPLETE = False
_PROBE_OK = False


def probe_qwen4_moe_router(dtype=mx.bfloat16) -> bool:
    global _PROBE_COMPLETE, _PROBE_OK
    if _PROBE_COMPLETE:
        return _PROBE_OK
    _PROBE_COMPLETE = True
    if dtype != mx.bfloat16 or not mx.metal.is_available():
        return False
    try:
        gates = mx.arange(NUM_EXPERTS, dtype=dtype)[None, None, :]
        indices, scores = qwen4_moe_router(gates)
        mx.eval(indices, scores)
        _PROBE_OK = indices.tolist() == [[list(range(502, 512))]]
    except (RuntimeError, ValueError):
        _PROBE_OK = False
    return _PROBE_OK


__all__ = [
    "RouterAdmission",
    "admit_qwen4_moe_router",
    "probe_qwen4_moe_router",
    "qwen4_moe_router",
]
