"""Fixed-shape Qwen4-Exp selected-expert MoE prototype.

The stock path already coalesces selected assignments in one fused gate-up
``gather_qmm``. Its remaining large boundary is the down projection: it writes
``[M, 10, 2560]`` before router weighting and reduction.

This module provides an isolated Metal kernel that consumes the resident
affine-q4 down table and the stock ``[M, 10, 640]`` SwiGLU activations. Each
SIMD group computes one output channel across all ten selected experts, applies
the router scores in registers, and writes only ``[M, 2560]``. No weight table
is copied or transformed. Qwen3-Next-family MoE blocks use it through the
default-``auto`` ``MLX_QWEN4_FUSED_EXPERT_KERNEL`` policy and retain the stock
path for every geometry or layout outside the narrow admission contract.
The ``scalar`` variant is the conservative baseline. ``tile4`` reuses each
hidden word across four output rows and splits the ten experts over five SIMD
groups, then reduces router-weighted values in the original slot order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx

HIDDEN_SIZE = 2560
EXPERT_HIDDEN_SIZE = 640
TOP_K = 10
NUM_EXPERTS = 512
GROUP_SIZE = 64
BITS = 4
PACK_FACTOR = 8
_SIMD_WIDTH = 32
_TILE4_SIMDGROUPS = 5
_TILE4_THREADS = _SIMD_WIDTH * _TILE4_SIMDGROUPS
_VARIANTS = ("scalar", "tile4")

_INDEX_DTYPES = (mx.int32, mx.uint32)


@dataclass(frozen=True)
class FusedMoeAdmission:
    """Result of the exact-shape admission check."""

    accepted: bool
    reason: str
    tokens: int = 0


def _shape(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(shape)


def _dtype_in(value, allowed: Sequence) -> bool:
    return getattr(value, "dtype", None) in allowed


def admit_qwen4_fused_down(
    hidden: mx.array,
    indices: mx.array,
    scores: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array | None,
    *,
    num_experts: int = NUM_EXPERTS,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    mode: str = "affine",
) -> FusedMoeAdmission:
    """Admit only the production Qwen4-Exp q4 decode geometries.

    The check is structural. It does not evaluate an MLX array or inspect
    index values, so it cannot add a device synchronization to decode.
    """

    hidden_shape = _shape(hidden)
    if hidden_shape is None or len(hidden_shape) < 3:
        return FusedMoeAdmission(False, "hidden must end in [top_k, 640]")
    if hidden_shape[-2:] != (TOP_K, EXPERT_HIDDEN_SIZE):
        return FusedMoeAdmission(False, "hidden must end in [10, 640]")

    tokens = 1
    for extent in hidden_shape[:-2]:
        tokens *= extent
    if tokens not in (1, 3):
        return FusedMoeAdmission(False, "only flattened token widths M=1 or M=3")

    routed_shape = hidden_shape[:-2] + (TOP_K,)
    if _shape(indices) != routed_shape or _shape(scores) != routed_shape:
        return FusedMoeAdmission(
            False,
            "indices and scores must match hidden prefix plus top_k=10",
            tokens,
        )
    if hidden.dtype != mx.bfloat16:
        return FusedMoeAdmission(False, "hidden must be bfloat16", tokens)
    if scores.dtype != hidden.dtype:
        return FusedMoeAdmission(
            False, "scores dtype must match hidden dtype", tokens
        )
    if not _dtype_in(indices, _INDEX_DTYPES):
        return FusedMoeAdmission(False, "indices must be int32 or uint32", tokens)
    if num_experts != NUM_EXPERTS:
        return FusedMoeAdmission(
            False, "only 512 routed experts are supported", tokens
        )
    if (group_size, bits, mode) != (GROUP_SIZE, BITS, "affine"):
        return FusedMoeAdmission(
            False, "only affine q4 with group_size=64 is supported", tokens
        )
    if down_biases is None:
        return FusedMoeAdmission(False, "affine q4 requires a bias table", tokens)

    packed_width = EXPERT_HIDDEN_SIZE // PACK_FACTOR
    group_width = EXPERT_HIDDEN_SIZE // GROUP_SIZE
    expected = (
        (down_weight, (NUM_EXPERTS, HIDDEN_SIZE, packed_width)),
        (down_scales, (NUM_EXPERTS, HIDDEN_SIZE, group_width)),
        (down_biases, (NUM_EXPERTS, HIDDEN_SIZE, group_width)),
    )
    for value, wanted in expected:
        if _shape(value) != wanted:
            return FusedMoeAdmission(
                False,
                f"packed table shape {_shape(value)} does not match {wanted}",
                tokens,
            )
    if down_weight.dtype != mx.uint32:
        return FusedMoeAdmission(False, "packed q4 weights must be uint32", tokens)
    if down_scales.dtype != mx.bfloat16:
        return FusedMoeAdmission(
            False, "q4 scales and biases must be bfloat16", tokens
        )
    if down_biases.dtype != down_scales.dtype:
        return FusedMoeAdmission(
            False, "q4 scales and biases must have one dtype", tokens
        )
    if not hasattr(mx.fast, "metal_kernel") or not mx.metal.is_available():
        return FusedMoeAdmission(False, "MLX Metal kernels are unavailable", tokens)
    if mx.default_device() != mx.gpu:
        return FusedMoeAdmission(False, "the default MLX device is not GPU", tokens)
    return FusedMoeAdmission(True, "eligible", tokens)


_DOWN_REDUCE_SOURCE = r"""
    constexpr uint H = 2560;
    constexpr uint EH = 640;
    constexpr uint TOPK = 10;
    constexpr uint DOWN_WORDS = 80;
    constexpr uint DOWN_GROUPS = 10;

    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.y;
    uint token = thread_position_in_grid.z;
    const device uint32_t* packed = down_weight;

    float routed = 0.0f;
#pragma unroll
    for (uint slot = 0; slot < TOPK; ++slot) {
        uint expert = uint(indices[token * TOPK + slot]);
        size_t wrow = size_t(expert) * H + row;
        const device uint32_t* dw = packed + wrow * DOWN_WORDS;
        const device W* ds = down_scales + wrow * DOWN_GROUPS;
        const device W* db = down_biases + wrow * DOWN_GROUPS;
        const device T* hrow = hidden + (size_t(token) * TOPK + slot) * EH;

        float value = 0.0f;
        for (uint word = lane; word < DOWN_WORDS; word += 32) {
            uint32_t p = dw[word];
            uint group = word >> 3;
            float scale = float(ds[group]);
            float bias = float(db[group]);
            size_t hbase = size_t(word) * 8;
            float accum_q = 0.0f;
            float accum_x = 0.0f;
#pragma unroll
            for (uint nibble = 0; nibble < 8; ++nibble) {
                float xv = float(hrow[hbase + nibble]);
                accum_x += xv;
                accum_q += xv * float((p >> (4 * nibble)) & 0xFu);
            }
            value += scale * accum_q + bias * accum_x;
        }
        value = simd_sum(value);
        if (lane == 0) {
            // Match the two T-valued boundaries in the stock graph.
            T expert_value = static_cast<T>(value);
            T weighted_value = static_cast<T>(
                float(expert_value) * float(scores[token * TOPK + slot]));
            routed += float(weighted_value);
        }
    }
    if (lane == 0) {
        out[size_t(token) * H + row] = static_cast<T>(routed);
    }
"""


_DOWN_REDUCE_TILE4_SOURCE = r"""
    constexpr uint H = 2560;
    constexpr uint EH = 640;
    constexpr uint TOPK = 10;
    constexpr uint DOWN_WORDS = 80;
    constexpr uint DOWN_GROUPS = 10;

    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row_base = thread_position_in_grid.y * 4;
    uint token = thread_position_in_grid.z;
    uint slot_base = sg * 2;
    const device uint32_t* packed = down_weight;
    threadgroup float partials[TOPK * 4];

    float values[8];
#pragma unroll
    for (uint i = 0; i < 8; ++i) {
        values[i] = 0.0f;
    }

#pragma unroll
    for (uint local_slot = 0; local_slot < 2; ++local_slot) {
        uint slot = slot_base + local_slot;
        uint expert = uint(indices[token * TOPK + slot]);
        const device T* hrow = hidden + (size_t(token) * TOPK + slot) * EH;

        for (uint word = lane; word < DOWN_WORDS; word += 32) {
            size_t hbase = size_t(word) * 8;
            float xv[8];
            float accum_x = 0.0f;
#pragma unroll
            for (uint nibble = 0; nibble < 8; ++nibble) {
                xv[nibble] = float(hrow[hbase + nibble]);
                accum_x += xv[nibble];
            }

#pragma unroll
            for (uint local_row = 0; local_row < 4; ++local_row) {
                uint row = row_base + local_row;
                size_t wrow = size_t(expert) * H + row;
                uint32_t p = packed[wrow * DOWN_WORDS + word];
                uint group = word >> 3;
                float scale = float(
                    down_scales[wrow * DOWN_GROUPS + group]);
                float bias = float(
                    down_biases[wrow * DOWN_GROUPS + group]);
                float accum_q = 0.0f;
#pragma unroll
                for (uint nibble = 0; nibble < 8; ++nibble) {
                    accum_q += xv[nibble] *
                        float((p >> (4 * nibble)) & 0xFu);
                }
                values[local_slot * 4 + local_row] +=
                    scale * accum_q + bias * accum_x;
            }
        }
    }

#pragma unroll
    for (uint i = 0; i < 8; ++i) {
        values[i] = simd_sum(values[i]);
    }
    if (lane == 0) {
#pragma unroll
        for (uint local_slot = 0; local_slot < 2; ++local_slot) {
            uint slot = slot_base + local_slot;
            float score = float(scores[token * TOPK + slot]);
#pragma unroll
            for (uint local_row = 0; local_row < 4; ++local_row) {
                T expert_value = static_cast<T>(
                    values[local_slot * 4 + local_row]);
                T weighted_value = static_cast<T>(
                    float(expert_value) * score);
                partials[slot * 4 + local_row] = float(weighted_value);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0 && lane < 4) {
        float routed = 0.0f;
#pragma unroll
        for (uint slot = 0; slot < TOPK; ++slot) {
            routed += partials[slot * 4 + lane];
        }
        out[size_t(token) * H + row_base + lane] = static_cast<T>(routed);
    }
"""


_scalar_down_reduce_kernel = None
_tile4_down_reduce_kernel = None


def _kernel(variant: str):
    global _scalar_down_reduce_kernel, _tile4_down_reduce_kernel
    if variant == "scalar" and _scalar_down_reduce_kernel is None:
        _scalar_down_reduce_kernel = mx.fast.metal_kernel(
            name="qwen4_q4_down_reduce_scalar",
            input_names=[
                "hidden",
                "down_weight",
                "down_scales",
                "down_biases",
                "indices",
                "scores",
            ],
            output_names=["out"],
            source=_DOWN_REDUCE_SOURCE,
            # Resident expert tables are row-contiguous. Do not copy a large
            # non-contiguous view implicitly; this kernel is a trusted path.
            ensure_row_contiguous=False,
        )
    if variant == "tile4" and _tile4_down_reduce_kernel is None:
        _tile4_down_reduce_kernel = mx.fast.metal_kernel(
            name="qwen4_q4_down_reduce_tile4",
            input_names=[
                "hidden",
                "down_weight",
                "down_scales",
                "down_biases",
                "indices",
                "scores",
            ],
            output_names=["out"],
            source=_DOWN_REDUCE_TILE4_SOURCE,
            ensure_row_contiguous=False,
        )
    return (
        _scalar_down_reduce_kernel
        if variant == "scalar"
        else _tile4_down_reduce_kernel
    )


def qwen4_fused_down(
    hidden: mx.array,
    indices: mx.array,
    scores: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
    *,
    num_experts: int = NUM_EXPERTS,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    mode: str = "affine",
    variant: str = "scalar",
) -> mx.array:
    """Fuse q4 down projection, router weighting, and top-10 reduction.

    Raise when the exact production geometry is not present. The integration
    layer must catch the rejected admission before this call and use the stock
    ``gather_qmm`` path instead. Indices must come from the model's trusted
    top-k router, and every input must use its standard row-contiguous layout.
    ``variant`` is selected per call, so serving can tune it without reload.
    """

    if variant not in _VARIANTS:
        raise ValueError(
            f"unknown Qwen4 fused-down variant {variant!r}; "
            f"expected one of {_VARIANTS}"
        )

    admission = admit_qwen4_fused_down(
        hidden,
        indices,
        scores,
        down_weight,
        down_scales,
        down_biases,
        num_experts=num_experts,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    if not admission.accepted:
        raise ValueError(f"Qwen4 fused down is not eligible: {admission.reason}")

    if variant == "scalar":
        grid = (_SIMD_WIDTH, HIDDEN_SIZE, admission.tokens)
        threadgroup = (_SIMD_WIDTH, 1, 1)
    else:
        grid = (_TILE4_THREADS, HIDDEN_SIZE // 4, admission.tokens)
        threadgroup = (_TILE4_THREADS, 1, 1)
    out = _kernel(variant)(
        inputs=[
            hidden,
            down_weight,
            down_scales,
            down_biases,
            indices,
            scores,
        ],
        template=[("T", hidden.dtype), ("W", down_scales.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(admission.tokens, HIDDEN_SIZE)],
        output_dtypes=[hidden.dtype],
    )[0]
    return out.reshape(hidden.shape[:-2] + (HIDDEN_SIZE,))


__all__ = [
    "FusedMoeAdmission",
    "admit_qwen4_fused_down",
    "qwen4_fused_down",
]
