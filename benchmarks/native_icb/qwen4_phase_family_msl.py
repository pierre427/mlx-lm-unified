"""Generate the exact-helper Metal source for the native Qwen4 P1 gate.

The Q4 arithmetic is not reimplemented here.  ``build_source`` extracts the
current qmv helper slice and activation function from ``BODY_HELPERS`` and the
current low-level dot/dequant helpers from ``MEGA_QMV``.  Digest binding in the
prepared artifact makes source drift a fail-closed event.
"""

from __future__ import annotations

import re
from typing import Any


PHASE_SOURCE = r"""
struct PhaseParams {
  uint op;
  uint entry;
  uint src;
  uint dst;
  uint arg0;
  uint arg1;
  uint arg2;
  uint barrier;
  uint command;
};

inline void p1_receipt(device atomic_uint* receipts, uint command,
                       uint tg, uint tid) {
  if (tg == 0u && tid == 0u)
    atomic_fetch_add_explicit(receipts + command, 1u, memory_order_relaxed);
}

kernel void p1_group_rmsnorm(
    const device uint* W [[buffer(0)]],
    const device uint* tbl [[buffer(1)]],
    device float* scratch [[buffer(2)]],
    constant PhaseParams& p [[buffer(3)]],
    device atomic_uint* receipts [[buffer(4)]],
    threadgroup float* tgmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
  const uint lane = tid & 31u;
  const uint sg = tid >> 5u;
  threadgroup float* red = tgmem;
  threadgroup float* gsc = tgmem + 16u;
  const device uint* T = tbl + p.entry * 12u;
  const device bfloat16_t* gw =
      reinterpret_cast<const device bfloat16_t*>(W + T[TBL_WOFF]);
  const uint dim = p.arg0, grp = p.arg1, ngrp = dim / grp;
  for (uint g = 0u; g < ngrp; ++g) {
    float sum = 0.0f;
    for (uint i = tid; i < grp; i += 512u) {
      const float value = scratch[p.src + g * grp + i];
      sum += value * value;
    }
    sum = simd_sum(sum);
    if (lane == 0u) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float total = 0.0f;
      for (uint j = 0u; j < 16u; ++j) total += red[j];
      gsc[g] = metal::precise::rsqrt(total / float(grp) + 9.9999999999999995e-07f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = tg * 512u + tid; i < dim; i += ntg * 512u)
    scratch[p.dst + i] =
        scratch[p.src + i] * gsc[i / grp] * float(gw[i]);
  p1_receipt(receipts, p.command, tg, tid);
}

kernel void p1_qmv(
    const device uint* W [[buffer(0)]],
    const device uint* tbl [[buffer(1)]],
    device float* scratch [[buffer(2)]],
    constant PhaseParams& p [[buffer(3)]],
    device atomic_uint* receipts [[buffer(4)]],
    threadgroup float* tgx [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
  const uint lane = tid & 31u;
  const uint sg = tid >> 5u;
  const uint grow = tg * 16u + sg;
  const uint nrow = ntg * 16u;
  const device uint* T = tbl + p.entry * 12u;
  const uint cols = T[TBL_COLS], rows = T[TBL_ROWS];
  const uint ng = cols / T[TBL_GSIZE];
  const uint R = (p.arg0 == 0u) ? 1u : p.arg0;
  const bool staged = cols <= 2560u;
  if (staged) {
    for (uint i = tid; i < cols; i += 512u) tgx[i] = scratch[p.src + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const threadgroup float4* tgx4 =
      reinterpret_cast<const threadgroup float4*>(tgx);
  const device float4* dx4 =
      reinterpret_cast<const device float4*>(scratch + p.src);
  for (uint r0 = grow * R; r0 < rows; r0 += nrow * R) {
    float acc[4];
    if (staged)
      qmv_any<1>(true, R, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                 r0, rows, 0u, tgx4, cols / 4u, lane, acc);
    else
      qmv_any<1>(true, R, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                 r0, rows, 0u, dx4, 0u, lane, acc);
    if (lane == 0u) {
      for (uint r = 0u; r < R; ++r)
        if (r0 + r < rows)
          scratch[p.dst + r0 + r] = apply_act(acc[r], p.arg1);
    }
  }
  p1_receipt(receipts, p.command, tg, tid);
}

kernel void p1_hc_mix(
    const device uint* W [[buffer(0)]],
    const device uint* tbl [[buffer(1)]],
    device float* scratch [[buffer(2)]],
    constant PhaseParams& p [[buffer(3)]],
    device atomic_uint* receipts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]) {
  (void)W;
  (void)tbl;
  const uint count = p.arg0, width = p.arg1;
  for (uint d = tg * 512u + tid; d < width; d += ntg * 512u) {
    float total = 0.0f;
    for (uint h = 0u; h < count; ++h)
      total += scratch[p.src + h * width + d]
             * scratch[20480u + h * width + d];
    scratch[p.dst + d] = total / float(count);
  }
  p1_receipt(receipts, p.command, tg, tid);
}
"""


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise ValueError(f"unterminated helper: {signature}")


def build_source(megakernel: Any, body: Any) -> tuple[str, dict[str, str]]:
    helper_stop = "// Build the union of the M queries' top-k expert lists"
    if helper_stop not in body.BODY_HELPERS:
        raise ValueError("BODY_HELPERS qmv extraction marker drifted")
    qmv_helpers = body.BODY_HELPERS.split(helper_stop, 1)[0]
    activation = _function(body.BODY_HELPERS, "inline float apply_act")
    low_level = megakernel.MEGA_QMV.replace("BF", "bfloat16_t")
    source = "\n".join(
        (
            "#include <metal_stdlib>",
            "#include <metal_atomic>",
            "using namespace metal;",
            "typedef bfloat bfloat16_t;",
            low_level,
            qmv_helpers,
            activation,
            PHASE_SOURCE,
        )
    )
    substitutions = {
        "BFT": "bfloat16_t",
        "RMAXN": "4",
        "MAXMW": "1",
        "HCN": "4",
    }
    for name in sorted(substitutions, key=len, reverse=True):
        source = re.sub(r"\b" + name + r"\b", substitutions[name], source)
    if any(re.search(r"\b" + name + r"\b", source) for name in substitutions):
        raise ValueError("phase source has unresolved qmv substitutions")
    provenance = {
        "low_level": "qwen4_megakernel.MEGA_QMV",
        "qmv_helpers": "qwen4_megakernel_body.BODY_HELPERS prefix through qmv_any",
        "activation": "qwen4_megakernel_body.BODY_HELPERS apply_act",
        "phase_wrappers": "benchmarks/native_icb/qwen4_phase_family_msl.py",
    }
    return source + "\n", provenance
