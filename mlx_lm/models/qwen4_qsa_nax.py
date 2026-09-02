# Copyright © 2026 Apple Inc.
#
# NAX (Metal Performance Primitives matmul2d) block-sparse QSA attention for
# Qwen4-Exp / Qwen3.8-Flash-Next.  This is the "variant B" prototype kernel
# (1 query token x 12 query heads per NAX tile) validated at the true 24-head
# geometry (worst ~1.4e-3 rel vs fp32, ~7x better than MLX bf16 SDPA), wired
# behind ``MLX_QWEN4_QSA_NAX_KERNEL``.  It keeps fp32 accumulation and rounds
# only P, so it is MORE accurate than the bf16 masked-SDPA path it replaces on
# the 12 full-attention layers plus the MTP head at prefill.
#
# The kernel consumes the sorted, prefix-packed block ids that
# ``QSASelection.compact_blocks()`` already produces, so the dense
# [B, 1, L, T] selection mask is never materialized.

import mlx.core as mx

BLOCK = 4
MTILE = 16

HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
"""

SOURCE = r"""
    constexpr int NSG = 4;
    constexpr int DS  = D / NSG;
    constexpr int NKS = DS / 16;
    constexpr int NNT = DS / 32;

    // Query length, KV length and ids stride are runtime uniforms, not template
    // constants: every prefill-chunk width then shares ONE compiled pipeline.
    const int L   = dims[0];   // query length
    const int TOT = dims[1];   // KV length (physical_width)
    const int U   = dims[2];   // per-token ids stride (u_width)

    const ushort sg   = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const uint  tok   = threadgroup_position_in_grid.y;   // query token
    const uint  bkv   = threadgroup_position_in_grid.z;   // b * NKVH + hkv
    const uint  b     = bkv / NKVH;
    const uint  hkv   = bkv % NKVH;

    const short qid = lane >> 2;
    const short fm0 = ((qid & 4) | ((lane >> 1) & 3));    // row = query head
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;

    threadgroup float sS[NSG * 32 * 16];

    const device T* kb = k + (size_t)(b * NKVH + hkv) * TOT * D;
    const device T* vb = v + (size_t)(b * NKVH + hkv) * TOT * D;

    const int  lpad      = left_pad[b];
    const uint tile_base = (b * L + tok) * U;
    const uint cnt       = counts[b * L + tok];
    const uint nsel      = n_sel[b * L + tok];

    const int qp       = qpos[b * L + tok];
    const int complete = ((qp + 1) / BS) * BS;

    // Row r is query head hkv*GQA + r.  Rows GQA..15 are idle.
    int head_of[2];
    bool row_live[2];
    for (short rr = 0; rr < 2; rr++) {
        int r = fm0 + rr * 8;
        row_live[rr] = (r < GQA);
        head_of[rr] = (int)hkv * GQA + r;
    }

    typedef metal::vec<T, 8> frag_t;
    frag_t qf[NKS];
    for (short ks = 0; ks < NKS; ks++) {
        for (short i = 0; i < 8; i++) {
            short rr = (i >> 2);
            int c = sg * DS + ks * 16 + fn0 + (i % 4);
            qf[ks][i] = row_live[rr]
                ? q[((size_t)(b * NQH + head_of[rr]) * L + tok) * D + c]
                : T(0);
        }
    }

    float of[NNT][16];
    for (short t = 0; t < NNT; t++)
        for (short i = 0; i < 16; i++) of[t][i] = 0.0f;
    float rmax_r[2] = {-INFINITY, -INFINITY};
    float rsum_r[2] = {0.0f, 0.0f};

    constexpr auto desc_qk = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_pv = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    const float qk_scale = scale[0];

    for (uint u0 = 0; u0 < cnt; u0 += 8) {

        size_t koff[4];
        for (short qq = 0; qq < 4; qq++) {
            int n = fm0 + qq * 8;
            uint u = u0 + (uint)(n >> 2);
            int bid = (u < cnt) ? (int)ids[tile_base + u] : 0;
            int phys = lpad + bid * BS + (n & 3);
            koff[qq] = (size_t)metal::clamp(phys, 0, (int)TOT - 1) * D;
        }

        {
            mpp::tensor_ops::matmul2d<desc_qk, metal::execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = 0.0f;
            for (short ks = 0; ks < NKS; ks++) {
                for (short i = 0; i < 8; i++) ca[i] = qf[ks][i];
                for (short i = 0; i < 8; i++) {
                    int c = sg * DS + ks * 16 + fn0 + (i % 4);
                    cb[i]     = kb[koff[(i >> 2)] + c];
                    cb[8 + i] = kb[koff[2 + (i >> 2)] + c];
                }
                op.run(ca, cb, cc);
            }
            for (short i = 0; i < 16; i++)
                sS[(sg * 32 + lane) * 16 + i] = cc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[16];
        for (short i = 0; i < 16; i++) {
            s[i] = (sS[(0 * 32 + lane) * 16 + i] + sS[(1 * 32 + lane) * 16 + i]
                  + sS[(2 * 32 + lane) * 16 + i] + sS[(3 * 32 + lane) * 16 + i])
                 * qk_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // One selection for the whole tile: the mask has NO row dependence.
        bool colok[8];
        for (short f = 0; f < 2; f++) {
            for (short co = 0; co < 4; co++) {
                int n = f * 16 + fn0 + co;
                uint u = u0 + (uint)(n >> 2);
                bool inb = (u < cnt);
                int bid = inb ? (int)ids[tile_base + u] : 0;
                int kl = bid * BS + (n & 3);
                colok[f * 4 + co] = inb && (kl >= 0) && (kl <= qp)
                                    && ((u < nsel) || (kl >= complete));
            }
        }
        for (short f = 0; f < 2; f++)
            for (short rr = 0; rr < 2; rr++)
                for (short co = 0; co < 4; co++)
                    if (!colok[f * 4 + co]) s[f * 8 + rr * 4 + co] = -INFINITY;

        // The mask is shared across rows; the scores are not.  Max and sum
        // stay per-row.
        float rowmax[2];
        for (short rr = 0; rr < 2; rr++) {
            float m = -INFINITY;
            for (short f = 0; f < 2; f++)
                for (short co = 0; co < 4; co++)
                    m = metal::max(m, s[f * 8 + rr * 4 + co]);
            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(1)));
            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(8)));
            rowmax[rr] = m;
        }

        float alpha[2], p[16], newmax[2], newsum[2];
        for (short rr = 0; rr < 2; rr++) {
            float mn = metal::max(rmax_r[rr], rowmax[rr]);
            bool live = mn > -INFINITY;
            alpha[rr] = live ? metal::fast::exp(rmax_r[rr] - mn) : 1.0f;
            float sm = 0.0f;
            for (short f = 0; f < 2; f++) {
                for (short co = 0; co < 4; co++) {
                    short idx = f * 8 + rr * 4 + co;
                    float pv = live ? metal::fast::exp(s[idx] - mn) : 0.0f;
                    p[idx] = pv;
                    sm += pv;
                }
            }
            sm += metal::simd_shuffle_xor(sm, ushort(1));
            sm += metal::simd_shuffle_xor(sm, ushort(8));
            newmax[rr] = mn;
            newsum[rr] = rsum_r[rr] * alpha[rr] + sm;
        }
        for (short rr = 0; rr < 2; rr++) {
            rmax_r[rr] = newmax[rr];
            rsum_r[rr] = newsum[rr];
        }

        for (short t = 0; t < NNT; t++) {
            for (short f = 0; f < 2; f++)
                for (short rr = 0; rr < 2; rr++)
                    for (short co = 0; co < 4; co++)
                        of[t][f * 8 + rr * 4 + co] *= alpha[rr];

            int nbase = sg * DS + t * 32;
            mpp::tensor_ops::matmul2d<desc_pv, metal::execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = of[t][i];
            for (short ks = 0; ks < 2; ks++) {
                for (short i = 0; i < 8; i++) ca[i] = (T)p[ks * 8 + i];
                for (short i = 0; i < 8; i++) {
                    size_t ko = koff[ks * 2 + (i >> 2)];
                    int co = nbase + fn0 + (i % 4);
                    cb[i]     = vb[ko + co];
                    cb[8 + i] = vb[ko + co + 16];
                }
                op.run(ca, cb, cc);
            }
            for (short i = 0; i < 16; i++) of[t][i] = cc[i];
        }
    }

    for (short t = 0; t < NNT; t++) {
        for (short f = 0; f < 2; f++) {
            for (short rr = 0; rr < 2; rr++) {
                if (!row_live[rr]) continue;
                float inv = (rsum_r[rr] > 0.0f) ? (1.0f / rsum_r[rr]) : 0.0f;
                for (short co = 0; co < 4; co++) {
                    int c = sg * DS + t * 32 + f * 16 + fn0 + co;
                    out[((size_t)(b * NQH + head_of[rr]) * L + tok) * D + c] =
                        of[t][f * 8 + rr * 4 + co] * inv;
                }
            }
        }
    }
"""

_KERNEL = mx.fast.metal_kernel(
    name="nax_qsa_attn_b",
    input_names=["q", "k", "v", "ids", "counts", "n_sel", "qpos", "left_pad",
                 "scale", "dims"],
    output_names=["out"],
    header=HEADER,
    source=SOURCE,
)


def block_sparse_layout_supported(head_dim, n_heads, n_kv_heads, block_size):
    """Whether the head-tiled NAX kernel can serve this attention geometry.

    Constraints come from the tile shape, not the KV length: the D axis is
    split across NSG=4 simdgroups then 16/32-wide NAX tiles (so DS = D/4 must
    be a multiple of 32), one KV head's query heads must fit a 16-row tile,
    and the block size is fixed at 4 (the QSA compression ratio).
    """
    if n_kv_heads <= 0 or n_heads % n_kv_heads:
        return False
    ds = head_dim // 4
    return (
        head_dim % 4 == 0
        and ds % 32 == 0
        and block_size == BLOCK
        and (n_heads // n_kv_heads) <= MTILE
    )


_NAX_AVAILABLE = None


def nax_kernel_available():
    """True iff the MPP/NAX kernel compiles and runs on this device.

    Memoized: the probe compiles the kernel once on a minimal shape and runs
    it.  A device without Metal 4 / MetalPerformancePrimitives fails to build
    the ``mpp::tensor_ops::matmul2d`` op, which this catches so the caller
    falls back to dense SDPA instead of crashing a forward pass.
    """
    global _NAX_AVAILABLE
    if _NAX_AVAILABLE is not None:
        return _NAX_AVAILABLE
    if not mx.metal.is_available():
        _NAX_AVAILABLE = False
        return False
    try:
        total, dim = 8, 256
        q = mx.zeros((1, 2, 2, dim), dtype=mx.bfloat16)
        k = mx.zeros((1, 1, total, dim), dtype=mx.bfloat16)
        v = mx.zeros((1, 1, total, dim), dtype=mx.bfloat16)
        ids = mx.zeros((1, 2, 8), dtype=mx.uint32)
        counts = mx.ones((1, 2), dtype=mx.uint32)
        n_sel = mx.zeros((1, 2), dtype=mx.uint32)
        qpos = mx.array([[0, 1]], dtype=mx.int32)
        left_pad = mx.zeros((1,), dtype=mx.int32)
        out = nax_qsa_attention(
            q, k, v, ids, counts, n_sel, qpos, left_pad,
            scale=1.0 / 16.0, u_width=8, total=total, n_kv_heads=1,
        )
        mx.eval(out)
        _NAX_AVAILABLE = True
    except Exception:
        _NAX_AVAILABLE = False
    return _NAX_AVAILABLE


def compact_blocks_to_kernel_inputs(cb):
    """Turn a ``QSACompactBlocks`` into the kernel's per-token block list.

    ``cb.block_ids`` is the sorted, prefix-packed set of genuinely SELECTED
    logical blocks (``cb.block_counts`` long, suffix zeroed).  The kernel also
    needs the incomplete TAIL block appended at slot ``n_sel``; a slot at or
    past ``n_sel`` is admitted only by ``kl >= complete`` (the causal tail),
    never by membership.  The tail block is ``q_pos // block_size`` and is
    dropped from ``counts`` when it duplicates the last selected block
    (``q_pos == block_size - 1 mod block_size`` with that block selected), so
    the block list stays a set.

    Returns ``(ids, counts, n_sel, u_width, q_pos, left_pad, total)`` ready for
    ``nax_qsa_attention``.  The width is derived from the static block-id
    width, so this never syncs the device.
    """
    block_ids = cb.block_ids.astype(mx.int32)  # [B, L, K], suffix zeroed
    n_sel = cb.block_counts.astype(mx.int32)  # [B, L]
    bs = cb.block_size
    total = cb.physical_width
    batch, length, k_width = block_ids.shape
    n_ext = -(-total // bs)  # ceil: the incomplete tail block is block n_ext-1

    q_pos = (cb.tail_stop - 1).astype(mx.int32)  # tail_stop = q_pos + 1
    if q_pos.shape != (batch, length):
        q_pos = mx.broadcast_to(q_pos, (batch, length))
    tail = mx.clip(q_pos // bs, 0, n_ext - 1)

    last = mx.take_along_axis(
        block_ids, mx.clip(n_sel - 1, 0, k_width - 1)[..., None], axis=-1
    )[..., 0]
    dup = (n_sel > 0) & (last == tail)
    counts = n_sel + mx.where(dup, 0, 1)

    # Static width: at most every selected block plus the tail, rounded to the
    # kernel's 8-wide u-loop stride.  Extra width costs a wider ids buffer but
    # no kernel iterations (the u-loop bounds on ``counts``, not U).
    u_width = max(8, ((k_width + 1 + 7) // 8) * 8)
    pad = u_width - k_width
    ids_full = mx.concatenate(
        [block_ids, mx.full((batch, length, pad), n_ext, dtype=mx.int32)],
        axis=-1,
    )
    ids_full = mx.put_along_axis(
        ids_full, n_sel[..., None], tail[..., None], axis=-1
    )
    ids = mx.where(ids_full < n_ext, ids_full, 0).astype(mx.uint32)

    left_pad = cb.left_padding
    if left_pad is None:
        left_pad = mx.zeros((batch,), dtype=mx.int32)
    else:
        left_pad = left_pad.astype(mx.int32)
    return (
        ids,
        counts.astype(mx.uint32),
        n_sel.astype(mx.uint32),
        u_width,
        q_pos,
        left_pad,
        total,
    )


def compact_token_validity(cb):
    """Return compact token coordinates and the shared validity predicate."""

    ids, counts, n_sel, u_width, q_pos, left_pad, total = (
        compact_blocks_to_kernel_inputs(cb)
    )
    block_size = int(cb.block_size)
    logical = (
        ids.astype(mx.int32)[..., None] * block_size
        + mx.arange(block_size, dtype=mx.int32)
    )
    slots = mx.arange(u_width, dtype=mx.int32)[None, None, :, None]
    present = slots < counts.astype(mx.int32)[..., None, None]
    selected = slots < n_sel.astype(mx.int32)[..., None, None]
    tail = (logical >= cb.tail_start[..., None, None]) & (
        logical < cb.tail_stop[..., None, None]
    )
    valid = present & (selected | tail)
    physical = logical + left_pad[:, None, None, None]
    valid = valid & (physical >= 0) & (physical < total)
    valid = valid & (logical <= q_pos[..., None, None])
    physical = mx.clip(physical, 0, total - 1)
    if cb.causal_mask is not None:
        batch, length = ids.shape[:2]
        causal = mx.broadcast_to(
            cb.causal_mask, (batch, 1, length, total)
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


def nax_qsa_attention(q, k, v, ids, counts, n_sel, q_pos, left_pad, *,
                      scale, u_width, total, n_kv_heads):
    """Block-sparse QSA attention on NAX.  q [B,H,L,D], k/v [B,HKV,T,D] -> fp32.

    ``ids``/``counts``/``n_sel``/``q_pos``/``left_pad`` come from
    ``compact_blocks_to_kernel_inputs``.  Output is fp32 [B,H,L,D]; the caller
    casts to the attention dtype.
    """
    batch, nqh, length, dim = q.shape
    gqa = nqh // n_kv_heads
    assert gqa <= MTILE, "one token's heads must fit a 16-row NAX tile"
    (out,) = _KERNEL(
        inputs=[
            mx.contiguous(q), mx.contiguous(k), mx.contiguous(v),
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.astype(mx.uint32)),
            mx.contiguous(n_sel.astype(mx.uint32)),
            mx.contiguous(
                mx.broadcast_to(q_pos, (batch, length)).astype(mx.int32)
            ),
            mx.contiguous(left_pad.astype(mx.int32)),
            mx.array([scale], dtype=mx.float32),
            mx.array([length, total, u_width], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype), ("D", dim), ("NQH", nqh), ("NKVH", n_kv_heads),
            ("GQA", gqa), ("BS", BLOCK),
        ],
        grid=(128, length, batch * n_kv_heads),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch, nqh, length, dim)],
        output_dtypes=[mx.float32],
    )
    return out
