"""Qwen4 decode-shape affine-q4 output projection building block.

The standalone kernel mirrors MLX's ``affine_qmv_fast`` reduction order.  It
exists so the same QMV epilogue can be embedded in the one-dispatch GDN kernel
without depending on MLX-private Metal headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx


INPUT_DIM = 6144
OUTPUT_DIM = 2560
GROUP_SIZE = 64
BITS = 4


@dataclass(frozen=True)
class GdnOutprojAdmission:
    accepted: bool
    reason: str


def admit_qwen4_gdn_outproj(module: Any, x: Any) -> GdnOutprojAdmission:
    if tuple(getattr(x, "shape", ())) != (1, 1, INPUT_DIM):
        return GdnOutprojAdmission(False, "only B1/M1/6144 input")
    if getattr(x, "dtype", None) != mx.bfloat16:
        return GdnOutprojAdmission(False, "input must be bfloat16")
    if not all(hasattr(module, name) for name in ("weight", "scales", "biases")):
        return GdnOutprojAdmission(False, "output projection is not affine-quantized")
    if getattr(module, "group_size", None) != GROUP_SIZE:
        return GdnOutprojAdmission(False, "requires affine group-size 64")
    if getattr(module, "bits", None) != BITS:
        return GdnOutprojAdmission(False, "requires affine q4")
    if tuple(module.weight.shape) != (OUTPUT_DIM, INPUT_DIM // 8):
        return GdnOutprojAdmission(False, f"weight shape {tuple(module.weight.shape)}")
    expected_groups = INPUT_DIM // GROUP_SIZE
    for name in ("scales", "biases"):
        value = getattr(module, name)
        if tuple(value.shape) != (OUTPUT_DIM, expected_groups):
            return GdnOutprojAdmission(False, f"{name} shape {tuple(value.shape)}")
        if value.dtype != mx.bfloat16:
            return GdnOutprojAdmission(False, f"{name} must be bfloat16")
    if module.weight.dtype != mx.uint32:
        return GdnOutprojAdmission(False, "weight must be packed uint32")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return GdnOutprojAdmission(False, "Metal GPU unavailable")
    return GdnOutprojAdmission(True, "eligible")


_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""


_SOURCE = r"""
    constexpr int VALUES = 16;
    constexpr int BLOCK = 512;
    constexpr int ROWS_PER_SIMD = 4;
    constexpr int ROWS_PER_GROUP = 8;

    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row0 = threadgroup_position_in_grid.y * ROWS_PER_GROUP
                    + sg * ROWS_PER_SIMD;

    float result[ROWS_PER_SIMD] = {0.0f};
    for (uint k = 0; k < K; k += BLOCK) {
        float xv[VALUES];
        float xsum = 0.0f;
        const uint xbase = k + lane * VALUES;
        for (uint i = 0; i < VALUES; i += 4) {
            T tx0 = x[xbase + i];
            T tx1 = x[xbase + i + 1];
            T tx2 = x[xbase + i + 2];
            T tx3 = x[xbase + i + 3];
            float x0 = float(tx0);
            float x1 = float(tx1);
            float x2 = float(tx2);
            float x3 = float(tx3);
            // Preserve the source-type arithmetic used by MLX load_vector.
            xsum += float(tx0 + tx1 + tx2 + tx3);
            xv[i] = x0;
            xv[i + 1] = x1 / 16.0f;
            xv[i + 2] = x2 / 256.0f;
            xv[i + 3] = x3 / 4096.0f;
        }

        for (uint r = 0; r < ROWS_PER_SIMD; ++r) {
            uint row = row0 + r;
            if (row >= N) continue;
            const device uchar* wb =
                reinterpret_cast<const device uchar*>(weight)
                + (size_t)row * (K / 2) + k / 2 + lane * 8;
            const device ushort* wp =
                reinterpret_cast<const device ushort*>(wb);
            float accum = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                ushort packed = wp[i];
                accum += xv[4 * i] * float(packed & 0x000f)
                       + xv[4 * i + 1] * float(packed & 0x00f0)
                       + xv[4 * i + 2] * float(packed & 0x0f00)
                       + xv[4 * i + 3] * float(packed & 0xf000);
            }
            uint group = k / GS + lane / 4;
            float scale = float(scales[(size_t)row * (K / GS) + group]);
            float bias = float(biases[(size_t)row * (K / GS) + group]);
            result[r] += scale * accum + xsum * bias;
        }
    }

    for (uint r = 0; r < ROWS_PER_SIMD; ++r) {
        float value = simd_sum(result[r]);
        uint row = row0 + r;
        if (lane == 0 && row < N) output[row] = T(value);
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="qwen4_affine_q4_qmv_fast",
        input_names=["weight", "scales", "biases", "x"],
        output_names=["output"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def qwen4_gdn_outproj_qmv(module: Any, x: mx.array) -> mx.array:
    """Apply the production 2560x6144 affine-q4 projection to one BF16 row."""
    (output,) = _kernel()(
        inputs=[module.weight, module.scales, module.biases, x],
        template=[
            ("T", x.dtype),
            ("K", INPUT_DIM),
            ("N", OUTPUT_DIM),
            ("GS", GROUP_SIZE),
        ],
        grid=(64, (OUTPUT_DIM + 7) // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, 1, OUTPUT_DIM)],
        output_dtypes=[x.dtype],
    )
    return output


__all__ = [
    "GdnOutprojAdmission",
    "admit_qwen4_gdn_outproj",
    "qwen4_gdn_outproj_qmv",
]
