# Copyright © 2026 Apple Inc.
#
# The radix-selection design is adapted from MTPLX PR #397 (Apache-2.0).

"""Fused Metal scorer and exact selector for Qwen4 QSA stage one."""

from __future__ import annotations

import math
from functools import lru_cache
from types import SimpleNamespace

import mlx.core as mx

from .qwen4_qsa_nax import nax_kernel_available

_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16, mx.float32)
_MAX_TOPK = 512
_EXACT_BAND_EXTRA = 32


def qsa_stage1_kernel_available() -> bool:
    """Return whether a Metal custom kernel can run in this process."""

    return mx.metal.is_available() and mx.default_device() == mx.gpu


def qsa_stage1_supported(
    q: mx.array,
    pooled: mx.array,
    q_positions: mx.array,
    *,
    block_topk: int,
    compress_ratio: int,
) -> bool:
    """Check static geometry without evaluating device arrays."""

    if not qsa_stage1_kernel_available():
        return False
    if q.ndim != 4 or pooled.ndim != 3 or q_positions.ndim != 2:
        return False
    if q.shape[:2] != q_positions.shape:
        return False
    if q.shape[0] != pooled.shape[0] or q.shape[-1] != pooled.shape[-1]:
        return False
    if q.dtype not in _SUPPORTED_DTYPES or pooled.dtype not in _SUPPORTED_DTYPES:
        return False
    if q_positions.dtype not in (mx.int32, mx.int64):
        return False
    if int(q.shape[1]) <= 0 or int(pooled.shape[1]) <= int(block_topk):
        return False
    return 1 <= int(block_topk) <= _MAX_TOPK and int(compress_ratio) > 0


def qsa_stage1_score_producer(
    q: mx.array, pooled: mx.array, *, block_topk: int = 512
) -> str:
    """Return the selected score producer without dispatching work."""

    if (
        q.shape[0] == pooled.shape[0] == 1
        and q.shape[2:] == (4, 128)
        and pooled.shape[2] == 128
        and q.dtype == pooled.dtype
        and q.dtype in (mx.float16, mx.bfloat16)
        and pooled.shape[1] > int(block_topk) + _EXACT_BAND_EXTRA
        and nax_kernel_available()
    ):
        return "mpp_exact_band"
    return "mlx"


_MPP_SCORE_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant constexpr uint QSA_SCORE_HEADS = 4;
constant constexpr uint QSA_SCORE_HEAD_DIM = 128;
constant constexpr uint QSA_SCORE_QUERY_TILE = 16;
constant constexpr uint QSA_SCORE_KEY_TILE = 32;
constant constexpr uint QSA_SCORE_QUERIES_PER_SIMDGROUP = 4;
constant constexpr uint QSA_SCORE_SIMDGROUPS = 4;
constant constexpr uint QSA_SCORE_THREADS = 128;
constant constexpr uint QSA_SCORE_K_FRAGMENTS = 8;
constant constexpr float QSA_SCORE_SQRT_HEAD_DIM = 11.313708498984761f;

inline short2 qsa_score_nax_coord(ushort lane) {
    const short qid = short(lane >> 2);
    const short fragment_row = ((qid & 4) | ((short(lane) >> 1) & 3));
    const short fragment_col = ((qid & 2) | (short(lane) & 1)) * 4;
    return short2{fragment_col, fragment_row};
}
"""


_MPP_SCORE_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint simdgroup = tid >> 5;
    const ushort lane = ushort(tid & 31u);
    const uint rows = q_shape[1];
    const uint blocks = pooled_shape[1];
    const uint key_tiles = (blocks + QSA_SCORE_KEY_TILE - 1u) /
        QSA_SCORE_KEY_TILE;
    const uint tile = threadgroup_position_in_grid.x;
    const uint query_tile = tile / key_tiles;
    const uint key_tile = tile - query_tile * key_tiles;
    const uint query0 =
        query_tile * QSA_SCORE_QUERY_TILE +
        simdgroup * QSA_SCORE_QUERIES_PER_SIMDGROUP;
    const uint block0 = key_tile * QSA_SCORE_KEY_TILE;

    threadgroup InT pooled_tile[QSA_SCORE_KEY_TILE * QSA_SCORE_HEAD_DIM];
    constexpr uint VECTORS_PER_KEY = QSA_SCORE_HEAD_DIM / 4u;
    constexpr uint TILE_VECTORS = QSA_SCORE_KEY_TILE * VECTORS_PER_KEY;
    for (uint item = tid; item < TILE_VECTORS; item += QSA_SCORE_THREADS) {
        const uint key_local = item / VECTORS_PER_KEY;
        const uint dim4 = item - key_local * VECTORS_PER_KEY;
        const uint block = block0 + key_local;
        vec<InT, 4> values = vec<InT, 4>(InT(0));
        if (block < blocks) {
            const int64_t source =
                int64_t(block) * pooled_strides[1] +
                int64_t(dim4 * 4u) * pooled_strides[2];
            for (uint elem = 0u; elem < 4u; ++elem) {
                values[elem] = pooled[
                    source + int64_t(elem) * pooled_strides[2]];
            }
        }
        const uint destination =
            key_local * QSA_SCORE_HEAD_DIM + dim4 * 4u;
        for (uint elem = 0u; elem < 4u; ++elem) {
            pooled_tile[destination + elem] = values[elem];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul;
    auto left = matmul.get_left_input_cooperative_tensor<InT, InT, float>();
    auto right = matmul.get_right_input_cooperative_tensor<InT, InT, float>();
    auto accumulator = matmul.get_destination_cooperative_tensor<
        decltype(left), decltype(right), float>();

    constexpr short ELEMENTS_PER_FRAGMENT = 8;
    constexpr short ELEMENT_COLUMNS = 4;
    constexpr short ELEMENT_ROW_JUMP = 8;
    const short2 coordinate = qsa_score_nax_coord(lane);
    for (short item = 0; item < 2 * ELEMENTS_PER_FRAGMENT; ++item) {
        accumulator[item] = 0.0f;
    }

    for (uint k_frag = 0u; k_frag < QSA_SCORE_K_FRAGMENTS; ++k_frag) {
        for (short row_part = 0; row_part < 2; ++row_part) {
            const uint matrix_row =
                uint(coordinate.y + row_part * ELEMENT_ROW_JUMP);
            const uint head = matrix_row / QSA_SCORE_QUERIES_PER_SIMDGROUP;
            const uint query_local =
                matrix_row - head * QSA_SCORE_QUERIES_PER_SIMDGROUP;
            const uint query = query0 + query_local;
            vec<InT, 4> values = vec<InT, 4>(InT(0));
            if (query < rows) {
                const int64_t source =
                    int64_t(query) * q_strides[1] +
                    int64_t(head) * q_strides[2] +
                    int64_t(k_frag * 16u + uint(coordinate.x)) * q_strides[3];
                for (uint elem = 0u; elem < 4u; ++elem) {
                    values[elem] = q[
                        source + int64_t(elem) * q_strides[3]];
                }
            }
            for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {
                left[row_part * ELEMENT_COLUMNS + elem] = values[elem];
            }
        }

        for (short key_half = 0; key_half < 2; ++key_half) {
            for (short row_part = 0; row_part < 2; ++row_part) {
                const uint key_local = uint(
                    key_half * 16 + coordinate.y +
                    row_part * ELEMENT_ROW_JUMP);
                const uint source =
                    key_local * QSA_SCORE_HEAD_DIM +
                    k_frag * 16u + uint(coordinate.x);
                const threadgroup vec<InT, 4>* source4 =
                    reinterpret_cast<const threadgroup vec<InT, 4>*>(
                        pooled_tile + source);
                const vec<InT, 4> values = source4[0];
                for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {
                    right[
                        key_half * ELEMENTS_PER_FRAGMENT +
                        row_part * ELEMENT_COLUMNS + elem] = values[elem];
                }
            }
        }
        matmul.run(left, right, accumulator);
    }

    for (short key_half = 0; key_half < 2; ++key_half) {
        for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {
            const float head0_or_1 = metal::max(
                accumulator[key_half * ELEMENTS_PER_FRAGMENT + elem], 0.0f);
            const float head2_or_3 = metal::max(
                accumulator[
                    key_half * ELEMENTS_PER_FRAGMENT +
                    ELEMENT_COLUMNS + elem],
                0.0f);
            const float paired_head1_or_0 =
                simd_shuffle_xor(head0_or_1, ushort(16));
            const float paired_head3_or_2 =
                simd_shuffle_xor(head2_or_3, ushort(16));
            if ((lane & 16u) == 0u) {
                const uint query = query0 + uint(coordinate.y);
                const uint block =
                    block0 + uint(key_half * 16 + coordinate.x + elem);
                if (query < rows && block < blocks) {
                    const float head_sum =
                        ((head0_or_1 + paired_head1_or_0) + head2_or_3) +
                        paired_head3_or_2;
                    scores[size_t(query) * blocks + block] =
                        head_sum / QSA_SCORE_SQRT_HEAD_DIM;
                }
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _mpp_score_kernel():
    return mx.fast.metal_kernel(
        name="mlx_lm_qwen4_qsa_stage1_mpp_h4d128",
        input_names=["q", "pooled"],
        output_names=["scores"],
        header=_MPP_SCORE_HEADER,
        source=_MPP_SCORE_SOURCE,
        ensure_row_contiguous=False,
    )


def _mpp_scores(q: mx.array, pooled: mx.array) -> mx.array:
    rows = int(q.shape[1])
    blocks = int(pooled.shape[1])
    query_tiles = (rows + 15) // 16
    key_tiles = (blocks + 31) // 32
    (scores,) = _mpp_score_kernel()(
        inputs=[q, pooled],
        template=[("InT", q.dtype)],
        grid=(query_tiles * key_tiles * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, blocks)],
        output_dtypes=[mx.float32],
    )
    return scores


_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

inline uint qsa_float_order_key(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

inline ulong qsa_composite_key(float score, uint block_id) {
    return (ulong(qsa_float_order_key(score)) << 32) | ulong(block_id);
}

inline bool qsa_id_before(
    uint a_index, bool a_valid, uint b_index, bool b_valid) {
    if (a_valid != b_valid) {
        return a_valid;
    }
    return a_index < b_index;
}
"""


@lru_cache(maxsize=32)
def _stage1_kernel(
    heads: int,
    head_dim: int,
    topk: int,
    ratio: int,
    q_dtype: mx.Dtype,
    pooled_dtype: mx.Dtype,
):
    width = 1 << (max(256, topk) - 1).bit_length()
    if width > 1024:
        raise ValueError(f"QSA stage-one threadgroup width {width} is unsupported")

    header = _HEADER + f"""
constant constexpr uint HEADS = {heads};
constant constexpr uint HEAD_DIM = {head_dim};
constant constexpr uint TOP_K = {topk};
constant constexpr uint RATIO = {ratio};
constant constexpr uint WIDTH = {width};
constant constexpr uint RADIX_BINS = 256;
constant constexpr float SQRT_HEAD_DIM = {math.sqrt(head_dim)!r}f;
"""
    source = r"""
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const uint blocks = uint(dims[0]);
        const uint query_rows = uint(dims[1]);
        const uint batch = row / query_rows;
        const int qpos = int(q_positions[row]);
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(blocks, complete);
        const uint selected_valid = metal::min(TOP_K, valid_count);
        const size_t scratch_base = size_t(row) * blocks;

        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup atomic_uint selected_count;
        threadgroup ulong radix_prefix;
        threadgroup uint radix_rank;
        threadgroup ulong threshold_key;

        for (uint block = lane; block < valid_count; block += WIDTH) {
            float score_sum = 0.0f;
            for (uint head = 0; head < HEADS; ++head) {
                float dot = 0.0f;
                const size_t q_base =
                    (size_t(row) * HEADS + head) * HEAD_DIM;
                const size_t pooled_base =
                    (size_t(batch) * blocks + block) * HEAD_DIM;
                for (uint dim = 0; dim < HEAD_DIM; ++dim) {
                    dot += float(q[q_base + dim]) *
                           float(pooled[pooled_base + dim]);
                }
                score_sum += metal::max(dot, 0.0f);
            }
            score_scratch[scratch_base + block] = score_sum / SQRT_HEAD_DIM;
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup | mem_flags::mem_device);

        if (lane == 0) {
            radix_prefix = 0ul;
            radix_rank = selected_valid > 0 ? selected_valid - 1 : 0;
            threshold_key = 0xfffffffffffffffful;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_valid > 0) {
            for (uint pass = 0; pass < 8; ++pass) {
                if (lane < RADIX_BINS) {
                    atomic_store_explicit(
                        &radix_histogram[lane], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint shift = 56u - pass * 8u;
                const ulong prefix = radix_prefix;
                for (uint block = lane; block < valid_count; block += WIDTH) {
                    const float score = score_scratch[scratch_base + block];
                    const ulong key = qsa_composite_key(score, block);
                    const bool prefix_matches = pass == 0 ||
                        (key >> (shift + 8u)) == prefix;
                    if (prefix_matches) {
                        atomic_fetch_add_explicit(
                            &radix_histogram[uint((key >> shift) & 0xfful)],
                            1u,
                            memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lane == 0) {
                    uint rank = radix_rank;
                    uint chosen = 0;
                    for (int digit = 255; digit >= 0; --digit) {
                        const uint count = atomic_load_explicit(
                            &radix_histogram[uint(digit)], memory_order_relaxed);
                        if (rank < count) {
                            chosen = uint(digit);
                            break;
                        }
                        rank -= count;
                    }
                    radix_prefix = (radix_prefix << 8) | ulong(chosen);
                    radix_rank = rank;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                threshold_key = radix_prefix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0;
        if (lane == 0) {
            atomic_store_explicit(&selected_count, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (selected_valid > 0) {
            const ulong threshold = threshold_key;
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const float score = score_scratch[scratch_base + block];
                if (qsa_composite_key(score, block) >= threshold) {
                    const uint slot = atomic_fetch_add_explicit(
                        &selected_count, 1u, memory_order_relaxed);
                    if (slot < TOP_K) {
                        exchange_indices[slot] = block;
                        exchange_valid[slot] = 1;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_index = exchange_indices[lane];
        bool my_valid = exchange_valid[lane] != 0;
        for (uint sequence = 2; sequence <= WIDTH; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                exchange_indices[lane] = my_index;
                exchange_valid[lane] = my_valid ? 1 : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint partner = lane ^ stride;
                const uint other_index = exchange_indices[partner];
                const bool other_valid = exchange_valid[partner] != 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const bool is_lower = (lane & stride) == 0;
                const uint a_index = is_lower ? my_index : other_index;
                const bool a_valid = is_lower ? my_valid : other_valid;
                const uint b_index = is_lower ? other_index : my_index;
                const bool b_valid = is_lower ? other_valid : my_valid;
                const bool lower_wants_before = (lane & sequence) == 0;
                const bool b_before_a = qsa_id_before(
                    b_index, b_valid, a_index, a_valid);
                const bool a_before_b = qsa_id_before(
                    a_index, a_valid, b_index, b_valid);
                const bool swap = lower_wants_before ? b_before_a : a_before_b;
                if (swap) {
                    my_index = is_lower ? b_index : a_index;
                    my_valid = is_lower ? b_valid : a_valid;
                }
            }
        }

        if (lane < TOP_K) {
            uint output_id = my_index;
            if (lane >= selected_valid) {
                const uint invalid_count = TOP_K - selected_valid;
                output_id = blocks - invalid_count + (lane - selected_valid);
            }
            block_ids[size_t(row) * TOP_K + lane] = output_id;
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mlx_lm_qwen4_qsa_stage1_h{heads}_d{head_dim}_k{topk}_r{ratio}_"
            f"{str(q_dtype).replace('.', '_')}_{str(pooled_dtype).replace('.', '_')}"
        ),
        input_names=["q", "pooled", "q_positions", "dims"],
        output_names=["block_ids", "score_scratch"],
        header=header,
        source=source,
    )


def _select_scores(
    scores: mx.array,
    q_positions: mx.array,
    *,
    topk: int,
    compress_ratio: int,
) -> mx.array:
    """Select score-column IDs with the exact radix kernel."""

    rows, blocks = map(int, scores.shape)
    score_q = mx.ones((rows, 1, 1, 1), dtype=mx.float32)
    score_keys = scores.reshape(rows, blocks, 1)
    score_positions = q_positions.reshape(rows, 1)
    kernel = _stage1_kernel(
        1,
        1,
        int(topk),
        int(compress_ratio),
        mx.float32,
        mx.float32,
    )
    width = 1 << (max(256, int(topk)) - 1).bit_length()
    outputs = kernel(
        inputs=[
            score_q,
            score_keys,
            score_positions.astype(mx.int32),
            mx.array([blocks, 1], dtype=mx.int32),
        ],
        grid=(rows * width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=[(rows, 1, int(topk)), (rows, blocks)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return outputs[0].reshape(rows, int(topk))


def qsa_stage1_select(
    q: mx.array,
    pooled: mx.array,
    q_positions: mx.array,
    *,
    block_topk: int,
    compress_ratio: int,
) -> mx.array:
    """Return deterministic selected block IDs with shape ``[B,L,K]``."""

    if not qsa_stage1_supported(
        q,
        pooled,
        q_positions,
        block_topk=block_topk,
        compress_ratio=compress_ratio,
    ):
        raise ValueError("unsupported QSA stage-one kernel geometry")

    batch, length, _, _ = map(int, q.shape)
    blocks = int(pooled.shape[1])
    topk = int(block_topk)
    rows = batch * length
    positions = q_positions.reshape(rows)
    producer = qsa_stage1_score_producer(q, pooled, block_topk=topk)
    if producer == "mpp_exact_band":
        approximate = _mpp_scores(q, pooled)
        candidate_count = topk + _EXACT_BAND_EXTRA
        candidate_ids = _select_scores(
            approximate,
            positions,
            topk=candidate_count,
            compress_ratio=compress_ratio,
        )
        candidate_keys = mx.take(pooled[0], candidate_ids, axis=0)
        exact = mx.einsum(
            "lhd,lcd->lch",
            q[0].astype(mx.float32),
            candidate_keys.astype(mx.float32),
        )
        exact = mx.sum(mx.maximum(exact, 0), axis=-1) / math.sqrt(q.shape[-1])
        candidate_valid = (
            candidate_ids.astype(mx.int32) * int(compress_ratio)
            + int(compress_ratio)
            - 1
        ) <= positions[:, None]
        exact = mx.where(candidate_valid, exact, -mx.inf)
        all_candidate_positions = mx.full(
            (rows,),
            candidate_count * int(compress_ratio) - 1,
            dtype=mx.int32,
        )
        selected_slots = _select_scores(
            exact,
            all_candidate_positions,
            topk=topk,
            compress_ratio=compress_ratio,
        )
        selected = mx.take_along_axis(candidate_ids, selected_slots, axis=-1)
    else:
        scores = mx.einsum(
            "blhd,bnd->blnh",
            q.astype(mx.float32),
            pooled.astype(mx.float32),
        )
        scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(q.shape[-1])
        selected = _select_scores(
            scores.reshape(rows, blocks),
            positions,
            topk=topk,
            compress_ratio=compress_ratio,
        )
    return selected.reshape(batch, length, topk)


def qsa_stage1_kernel_cache_info():
    """Return the bounded template-cache receipt."""

    selector = _stage1_kernel.cache_info()
    scorer = _mpp_score_kernel.cache_info()
    return SimpleNamespace(
        currsize=selector.currsize + scorer.currsize,
        maxsize=selector.maxsize + scorer.maxsize,
    )
