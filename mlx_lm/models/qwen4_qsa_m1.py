"""Direct selected-block QSA attention for Qwen4 B1/M1 decode."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""


_SOURCE = r"""
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint bh = threadgroup_position_in_grid.z;
    const uint b = bh / NQH;
    const uint h = bh % NQH;
    const uint hkv = h / GQA;

    const int TOT = dims[0];
    const int U = dims[1];
    const uint cnt = counts[b];
    const uint selected = n_sel[b];
    const int qp = qpos[b];
    const int lpad = left_pad[b];
    const int complete = ((qp + 1) / BS) * BS;
    const uint tile_base = b * U;

    threadgroup float scores[8];
    threadgroup float probs[8];
    threadgroup int physical[8];
    threadgroup float shared_alpha;
    threadgroup float shared_sum;
    threadgroup float shared_max;

    float out = 0.0f;
    if (tid == 0u) {
        shared_sum = 0.0f;
        shared_max = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device T* qh = q + (size_t)(b * NQH + h) * D;
    const device T* kh = k + (size_t)(b * NKVH + hkv) * TOT * D;
    const device T* vh = v + (size_t)(b * NKVH + hkv) * TOT * D;

    const uint ntokens = cnt * BS;
    for (uint token0 = 0; token0 < ntokens; token0 += 8u) {
        const uint token = token0 + sg;
        const uint u = token / BS;
        const uint within = token % BS;
        const int bid = u < cnt ? int(ids[tile_base + u]) : 0;
        const int logical = bid * BS + int(within);
        const int phys = metal::clamp(lpad + logical, 0, TOT - 1);
        const bool live = token < ntokens && logical <= qp
            && ((u < selected) || (logical >= complete));

        float dot = 0.0f;
        for (uint d = lane; d < D; d += 32u)
            dot += float(qh[d]) * float(kh[(size_t)phys * D + d]);
        dot = simd_sum(dot);
        if (lane == 0u) {
            scores[sg] = live ? dot * scale[0] : -INFINITY;
            physical[sg] = phys;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float tile_max = -INFINITY;
            for (uint j = 0; j < 8u; ++j)
                tile_max = metal::max(tile_max, scores[j]);
            float next_max = metal::max(shared_max, tile_max);
            bool any = next_max > -INFINITY;
            float alpha = any ? metal::fast::exp(shared_max - next_max) : 1.0f;
            float tile_sum = 0.0f;
            for (uint j = 0; j < 8u; ++j) {
                float p = any ? metal::fast::exp(scores[j] - next_max) : 0.0f;
                probs[j] = p;
                tile_sum += p;
            }
            shared_alpha = alpha;
            shared_sum = shared_sum * alpha + tile_sum;
            shared_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < D) {
            out *= shared_alpha;
            for (uint j = 0; j < 8u; ++j)
                out += probs[j] * float(vh[(size_t)physical[j] * D + tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < D) {
        float value = shared_sum > 0.0f ? out / shared_sum : 0.0f;
        output[(size_t)(b * NQH + h) * D + tid] = T(value);
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="qwen4_qsa_direct_m1",
        input_names=[
            "q", "k", "v", "ids", "counts", "n_sel", "qpos",
            "left_pad", "scale", "dims",
        ],
        output_names=["output"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def qwen4_qsa_direct_m1(
    q, k, v, ids, counts, n_sel, q_pos, left_pad, *, scale, u_width, total,
    n_kv_heads,
):
    batch, nqh, length, dim = q.shape
    if length != 1 or dim != 256 or nqh % n_kv_heads:
        raise ValueError("direct QSA kernel requires M1, D256, integral GQA")
    gqa = nqh // n_kv_heads
    (output,) = _kernel()(
        inputs=[
            mx.contiguous(q), mx.contiguous(k), mx.contiguous(v),
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.reshape(batch).astype(mx.uint32)),
            mx.contiguous(n_sel.reshape(batch).astype(mx.uint32)),
            mx.contiguous(q_pos.reshape(batch).astype(mx.int32)),
            mx.contiguous(left_pad.astype(mx.int32)),
            mx.array([scale], dtype=mx.float32),
            mx.array([total, u_width], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype), ("D", dim), ("NQH", nqh),
            ("NKVH", n_kv_heads), ("GQA", gqa), ("BS", 4),
        ],
        grid=(256, 1, batch * nqh),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, nqh, 1, dim)],
        output_dtypes=[q.dtype],
    )
    return output


__all__ = ["qwen4_qsa_direct_m1"]
