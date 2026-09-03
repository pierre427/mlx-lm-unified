"""The megakernel's kernel body: phase bodies behind the schedule dispatcher.

Phase A built three things that had never met: a weight pack with an offset
table, a schedule of 8-word phase records with a CPU mirror that defines each
opcode, and two *separately prototyped* Metal kernels -- the tuned GDN+MoE
chain from the feasibility spike and the hyper-connection mixer -- each with
its own ad-hoc bindings.  This module is where they meet: ONE persistent
dispatch that walks the schedule buffer and reads every weight through the
offset table, so a real multi-layer chain runs end to end without the host.

**Nothing here is compile-time specialised on the schedule.**  The step count,
the repetition count and the whole opcode stream arrive as buffers, so one
compiled binary serves a 2-layer probe, a 48-layer token and the MTP head.
The only substitutions are machine constants (threadgroup width, model dims,
scratch offsets), which is what keeps the shader cache from scaling with the
layer count.

**Rows per simdgroup lives in the schedule, not in the source.**  The spec
tuned a different value for almost every projection (2 on the GDN input
projection, 4 on the router, 2 pairs on gate+up), and a data-driven dispatcher
cannot specialise per call site.  ``Step.arg0`` therefore carries it, and the
kernel dispatches to a template on 1/2/4.  Metal allocates registers for the
whole kernel, so the budget is the R=4 path either way -- which is exactly why
R=8 is not offered: the spec measured 8 rows spilling on the GDN input
projection (217 GB/s against 340 at 2 rows).

**What differs from the spike, deliberately.**  The spike DEQUANTISED
``in_proj_b`` and ``in_proj_a`` into one dense bf16 table before launch, so its
phase 1 ran those two projections in bf16.  Here they are ordinary packed
4-bit entries and go through the same quantised matvec as everything else,
which is the arithmetic the stock module actually runs.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx

from .qwen4_megakernel import (
    CONV_DIM,
    CONV_KERNEL,
    FF,
    GDN_KEY_DIM,
    GDN_KEY_HEADS,
    GDN_RATIO,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HC_COUNT,
    HC_HIDDEN,
    HC_LOWRANK,
    HIDDEN,
    KEY_DIM,
    NUM_EXPERTS,
    RMS_EPS,
    SCRATCH,
    SCRATCH_FLOATS,
    STEP_STRIDE,
    TOPK,
    VALUE_DIM,
    _SPIN_CAP,
    _THREADGROUPS,
    _THREADS,
    kernel_header,
)
from .qwen4_megakernel_pack import TABLE_STRIDE

# ------------------------------------------------------------- threadgroup map
# One arena, carved into named blocks, exactly as the device scratch is.  The
# spec's hard cap is 16 KiB: crossing to 25.6 KiB is what made the spike's
# DNSTAGE variant lose 4--6%, because it dropped resident threadgroups per core.
MAX_SIMDGROUPS = 16              # NT = 512 is the widest geometry we build
RMAX = 4                         # rows per simdgroup the dispatcher offers

_TG_BLOCKS = (
    ("TGX", HIDDEN),             # staged source vector for a HIDDEN-wide matvec
    ("TLOG", NUM_EXPERTS),       # router logits, so top-k is threadgroup-local
    ("SQ", GDN_KEY_DIM),
    ("SK", GDN_KEY_DIM),
    ("SV", GDN_VALUE_DIM),
    ("SY", GDN_VALUE_DIM),
    ("TLR", HC_LOWRANK),         # hyper low-rank vector, re-read HIDDEN times
    ("RED", MAX_SIMDGROUPS),     # cross-simdgroup reduction slots
    ("PART", RMAX * MAX_SIMDGROUPS),   # split-K partials
    ("GSC", HC_COUNT),           # the four GroupRMSNorm scales
    ("SHR", 8),                  # GDN core's shared scalars
    ("TOPW", TOPK),
)

TG: dict[str, int] = {}
_off = 0
for _name, _size in _TG_BLOCKS:
    TG[_name] = _off
    _off += _size
TG_FLOATS = _off
del _off, _name, _size

# + the top-k expert ids, which are uint and live in their own array
THREADGROUP_BYTES = TG_FLOATS * 4 + TOPK * 4


# ---------------------------------------------------------------- MSL: helpers
BODY_HELPERS = r"""
// ---- offset-table access.  Field order mirrors TABLE_FIELDS exactly. -------
#define TBL_GROUP 0u
#define TBL_KIND 1u
#define TBL_ROWS 2u
#define TBL_COLS 3u
#define TBL_EXPERTS 4u
#define TBL_BITS 5u
#define TBL_GSIZE 6u
#define TBL_WOFF 7u
#define TBL_SBOFF 8u
#define TBL_NW 9u
#define TBL_NSB 10u

#define NO_ENTRY 0xFFFFFFFFu

// R rows of a packed quantised entry, sharing one pass over the activation.
//
// The scale/bias region is the ADOPTED interleaved layout, [rows][2][ngroups]
// in bf16: row r's scales start at r*2*ng and its biases ng elements later.
// (The split layout puts all scales before all biases; the pack keeps it
// selectable, and `sb_stride_rows` is the one line that differs.)
template <uint R, typename F4>
inline void qmv4(const device uint* W, uint w_off, uint sb_off, uint cols,
                 uint ng, uint row0, uint rows, uint estride,
                 F4 x4, uint lane, thread float* acc) {
  const device uint2* w2 =
      reinterpret_cast<const device uint2*>(W + w_off);
  const device BFT* sb = reinterpret_cast<const device BFT*>(W + sb_off);
  const uint nwords = cols >> 3u;          // uint32 words per row at 4 bits
  const uint n2 = nwords >> 1u;
  uint woff[R], soff[R];
  for (uint r = 0; r < R; ++r) {
    uint rr = (row0 + r < rows) ? (row0 + r) : (rows - 1u);
    woff[r] = (estride + rr) * n2;         // in uint2 blocks
    soff[r] = (estride + rr) * 2u * ng;
    acc[r] = 0.0f;
  }
  for (uint bl = lane; bl < n2; bl += 32u) {
    float4 a0 = x4[bl * 4u + 0u], a1 = x4[bl * 4u + 1u];
    float4 a2 = x4[bl * 4u + 2u], a3 = x4[bl * 4u + 3u];
    float xs = hsum4(a0) + hsum4(a1) + hsum4(a2) + hsum4(a3);
    uint g = bl >> 2u;
    for (uint r = 0; r < R; ++r) {
      uint2 p = w2[woff[r] + bl];
      float part = dot8(p.x, a0, a1) + dot8(p.y, a2, a3);
      acc[r] += float(sb[soff[r] + g]) * part
              + float(sb[soff[r] + ng + g]) * xs;
    }
  }
  for (uint r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);
}

template <uint R, typename F4>
inline void qmv8(const device uint* W, uint w_off, uint sb_off, uint cols,
                 uint ng, uint row0, uint rows, uint estride,
                 F4 x4, uint lane, thread float* acc) {
  const device uint2* w2 =
      reinterpret_cast<const device uint2*>(W + w_off);
  const device BFT* sb = reinterpret_cast<const device BFT*>(W + sb_off);
  const uint nwords = cols >> 2u;
  const uint n2 = nwords >> 1u;
  uint woff[R], soff[R];
  for (uint r = 0; r < R; ++r) {
    uint rr = (row0 + r < rows) ? (row0 + r) : (rows - 1u);
    woff[r] = (estride + rr) * n2;
    soff[r] = (estride + rr) * 2u * ng;
    acc[r] = 0.0f;
  }
  for (uint bl = lane; bl < n2; bl += 32u) {
    float4 a0 = x4[bl * 2u + 0u], a1 = x4[bl * 2u + 1u];
    float xs = hsum4(a0) + hsum4(a1);
    uint g = bl >> 3u;
    for (uint r = 0; r < R; ++r) {
      uint2 p = w2[woff[r] + bl];
      float part = dot4x8(p.x, a0) + dot4x8(p.y, a1);
      acc[r] += float(sb[soff[r] + g]) * part
              + float(sb[soff[r] + ng + g]) * xs;
    }
  }
  for (uint r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);
}

// The same, but K is split ACROSS THE SIMDGROUPS of one threadgroup and the
// partials reduce through threadgroup memory.  The spec's "cheap form" of
// split-K: it costs no grid barrier.  It lost on the MoE down projection and
// wins 1.37x on the hyper down-mix -- the difference is row count, not the
// technique.  324 rows over 640 simdgroups leaves 87% of them idle otherwise.
template <uint R, typename F4>
inline void qmv4_ksplit(const device uint* W, uint w_off, uint sb_off,
                        uint cols, uint ng, uint row0, uint rows,
                        F4 x4, uint lane, uint sg, uint nsg,
                        thread float* acc) {
  const device uint2* w2 =
      reinterpret_cast<const device uint2*>(W + w_off);
  const device BFT* sb = reinterpret_cast<const device BFT*>(W + sb_off);
  const uint n2 = (cols >> 3u) >> 1u;
  uint woff[R], soff[R];
  for (uint r = 0; r < R; ++r) {
    uint rr = (row0 + r < rows) ? (row0 + r) : (rows - 1u);
    woff[r] = rr * n2;
    soff[r] = rr * 2u * ng;
    acc[r] = 0.0f;
  }
  for (uint bl = sg * 32u + lane; bl < n2; bl += nsg * 32u) {
    float4 a0 = x4[bl * 4u + 0u], a1 = x4[bl * 4u + 1u];
    float4 a2 = x4[bl * 4u + 2u], a3 = x4[bl * 4u + 3u];
    float xs = hsum4(a0) + hsum4(a1) + hsum4(a2) + hsum4(a3);
    uint g = bl >> 2u;
    for (uint r = 0; r < R; ++r) {
      uint2 p = w2[woff[r] + bl];
      float part = dot8(p.x, a0, a1) + dot8(p.y, a2, a3);
      acc[r] += float(sb[soff[r] + g]) * part
              + float(sb[soff[r] + ng + g]) * xs;
    }
  }
  for (uint r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);
}

inline float apply_act(float v, uint kind) {
  if (kind == 1u) return silu_f(v);                       // ACT_SILU
  if (kind == 2u) return sigmoid_f(v);                    // ACT_SIGMOID
  if (kind == 3u) { float s = v * (1.0f / float(HCN));    // ACT_SILU_SCALED
                    return silu_f(s); }
  if (kind == 4u) return 2.0f * sigmoid_f(v * (1.0f / float(HCN)));
  return v;                                               // ACT_NONE
}
"""


# ------------------------------------------------------------------- MSL: body
BODY_SRC = r"""
  const uint tid  = thread_position_in_threadgroup.x;
  const uint tg   = threadgroup_position_in_grid.x;
  const uint ntg  = threadgroups_per_grid.x;
  const uint lane = tid & 31u;
  const uint sg   = tid >> 5u;
  const uint NSG  = NT / 32u;
  const uint nrow = ntg * NSG;              // simdgroups in the whole grid
  const uint grow = tg * NSG + sg;          // this simdgroup's global index

  device atomic_uint* ctr =
      reinterpret_cast<device atomic_uint*>(const_cast<device uint*>(ctrl));
  device atomic_uint* ab = ctr + 1;
  uint phase = base[0];

  const uint nsteps = meta[0];
  const uint reps   = meta[1];

  // Every packed group is a binding; the table's `group` field selects one.
  const device uint* WB[8] = {w0, w1, w2, w3, w4, w5, w6, w7};

  device float* sc = scratch;

  threadgroup float A[TGF];
  threadgroup uint  topi[TOPKN];
  threadgroup float* tgx  = A + TG_TGX;
  threadgroup float* tlog = A + TG_TLOG;
  threadgroup float* sq   = A + TG_SQ;
  threadgroup float* sk   = A + TG_SK;
  threadgroup float* sv   = A + TG_SV;
  threadgroup float* sy   = A + TG_SY;
  threadgroup float* tlr  = A + TG_TLR;
  threadgroup float* red  = A + TG_RED;
  threadgroup float* part = A + TG_PART;
  threadgroup float* gsc  = A + TG_GSC;
  threadgroup float* shr  = A + TG_SHR;
  threadgroup float* topw = A + TG_TOPW;
  const threadgroup float4* tgx4 =
      reinterpret_cast<const threadgroup float4*>(A + TG_TGX);
  const threadgroup float4* tlr4 =
      reinterpret_cast<const threadgroup float4*>(A + TG_TLR);

  bool live = true;
  for (uint rep = 0; rep < reps && live; ++rep) {
    // The residual streams arrive hoisted: PLE's n-gram gather and the
    // embedding depend only on the input token, so they are host work.
    for (uint i = tg * NT + tid; i < HCH; i += ntg * NT)
      sc[SC_RESID_A + i] = float(xin[(size_t)rep * HCH + i]);
    live = gbar(ctr, ab, ntg, tid, phase, SPINCAP);
    if (!live) break;

    uint gdn_slot = 0u;
    for (uint step = 0; step < nsteps && live; ++step) {
      const device uint* S = sched + step * 8u;
      const uint op = S[0], ent = S[1], src = S[2], dst = S[3];
      const uint a0 = S[4], a1 = S[5], a2 = S[6], bar = S[7];

      // ------------------------------------------------------- OP_QMV (1)
      if (op == 1u) {
        const device uint* T = tbl + ent * TSTRIDE;
        const device uint* W = WB[T[TBL_GROUP]];
        const uint cols = T[TBL_COLS], rows = T[TBL_ROWS];
        const uint ng = cols / T[TBL_GSIZE];
        const uint bits = T[TBL_BITS];
        const uint woff = T[TBL_WOFF], sboff = T[TBL_SBOFF];
        const uint R = (a0 == 0u) ? 1u : a0;
        // Stage the source when it fits: G threadgroups x NSG simdgroups all
        // stream the same vector, and a HIDDEN-wide one is 10 KiB.
        bool staged = cols <= HID;
        if (staged) {
          for (uint i = tid; i < cols; i += NT) tgx[i] = sc[src + i];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const device float4* dx4 =
            reinterpret_cast<const device float4*>(sc + src);
        for (uint r0 = grow * R; r0 < rows; r0 += nrow * R) {
          float acc[RMAXN];
          if (bits == 4u) {
            if (staged) {
              if (R == 4u)      qmv4<4>(W, woff, sboff, cols, ng, r0, rows, 0u, tgx4, lane, acc);
              else if (R == 2u) qmv4<2>(W, woff, sboff, cols, ng, r0, rows, 0u, tgx4, lane, acc);
              else              qmv4<1>(W, woff, sboff, cols, ng, r0, rows, 0u, tgx4, lane, acc);
            } else {
              if (R == 4u)      qmv4<4>(W, woff, sboff, cols, ng, r0, rows, 0u, dx4, lane, acc);
              else if (R == 2u) qmv4<2>(W, woff, sboff, cols, ng, r0, rows, 0u, dx4, lane, acc);
              else              qmv4<1>(W, woff, sboff, cols, ng, r0, rows, 0u, dx4, lane, acc);
            }
          } else {
            if (staged) {
              if (R == 4u)      qmv8<4>(W, woff, sboff, cols, ng, r0, rows, 0u, tgx4, lane, acc);
              else if (R == 2u) qmv8<2>(W, woff, sboff, cols, ng, r0, rows, 0u, tgx4, lane, acc);
              else              qmv8<1>(W, woff, sboff, cols, ng, r0, rows, 0u, tgx4, lane, acc);
            } else {
              if (R == 4u)      qmv8<4>(W, woff, sboff, cols, ng, r0, rows, 0u, dx4, lane, acc);
              else if (R == 2u) qmv8<2>(W, woff, sboff, cols, ng, r0, rows, 0u, dx4, lane, acc);
              else              qmv8<1>(W, woff, sboff, cols, ng, r0, rows, 0u, dx4, lane, acc);
            }
          }
          if (lane == 0u)
            for (uint r = 0; r < R; ++r)
              if (r0 + r < rows) sc[dst + r0 + r] = apply_act(acc[r], a1);
        }
      }

      // ------------------------------------------- OP_GROUP_RMSNORM (4)
      // Four groups of HID.  Every threadgroup computes all four scales for
      // itself -- 4 x HID reads, 40 KiB -- so the scale needs no publish and
      // only the normed vector crosses the grid.
      else if (op == 4u) {
        const device uint* T = tbl + ent * TSTRIDE;
        const device BFT* gw = reinterpret_cast<const device BFT*>(
            WB[T[TBL_GROUP]] + T[TBL_WOFF]);
        const uint dim = a0, grp = a1, ngrp = dim / grp;
        for (uint g = 0; g < ngrp; ++g) {
          float p = 0.0f;
          for (uint i = tid; i < grp; i += NT) {
            float v = sc[src + g * grp + i];
            p += v * v;
          }
          p = simd_sum(p);
          if (lane == 0u) red[sg] = p;
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (tid == 0u) {
            float total = 0.0f;
            for (uint j = 0; j < NSG; ++j) total += red[j];
            gsc[g] = metal::precise::rsqrt(total / float(grp) + NEPS);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = tg * NT + tid; i < dim; i += ntg * NT)
          sc[dst + i] = sc[src + i] * gsc[i / grp] * float(gw[i]);
      }

      // ------------------------------------------------- OP_HC_DOWN (16)
      // The 10,240 -> 320 mix-down and, from the SAME normed vector, the
      // 4-row block-inject gate.  Both are K-split across the simdgroups.
      else if (op == 16u) {
        const device uint* T = tbl + ent * TSTRIDE;
        const device uint* W = WB[T[TBL_GROUP]];
        const uint cols = T[TBL_COLS], rows = T[TBL_ROWS];
        const uint ng = cols / T[TBL_GSIZE];
        const device uint* TI = (a0 == NO_ENTRY) ? T : (tbl + a0 * TSTRIDE);
        const uint irows = (a0 == NO_ENTRY) ? 0u : TI[TBL_ROWS];
        const device float4* x4 =
            reinterpret_cast<const device float4*>(sc + src);
        for (uint r0 = tg * RDOWN; r0 < rows + irows; r0 += ntg * RDOWN) {
          bool inj = r0 >= rows;
          uint rr = inj ? (r0 - rows) : r0;
          uint cap = inj ? irows : rows;
          if (rr >= cap) continue;
          const device uint* WW = inj ? WB[TI[TBL_GROUP]] : W;
          uint wo = inj ? TI[TBL_WOFF] : T[TBL_WOFF];
          uint so = inj ? TI[TBL_SBOFF] : T[TBL_SBOFF];
          uint nn = inj ? (TI[TBL_COLS] / TI[TBL_GSIZE]) : ng;
          uint nc = inj ? TI[TBL_COLS] : cols;
          float acc[RMAXN];
          qmv4_ksplit<RDOWN>(WW, wo, so, nc, nn, rr, cap, x4,
                             lane, sg, NSG, acc);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (lane == 0u)
            for (uint r = 0; r < RDOWN; ++r) part[r * NSG + sg] = acc[r];
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (tid == 0u) {
            for (uint r = 0; r < RDOWN; ++r) {
              if (rr + r >= cap) break;
              float total = 0.0f;
              for (uint j = 0; j < NSG; ++j) total += part[r * NSG + j];
              if (inj) sc[a1 + rr + r] = apply_act(total, 4u);
              else     sc[dst + rr + r] = apply_act(total, 3u);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }

      // --------------------------------------------------- OP_HC_UP (17)
      // 320 -> 10,240 with sigmoid, taken d-MAJOR so one simdgroup holds all
      // four streams of a feature and mean(w * streams) stays local.
      else if (op == 17u) {
        const device uint* T = tbl + ent * TSTRIDE;
        const device uint* W = WB[T[TBL_GROUP]];
        const uint cols = T[TBL_COLS];
        const uint ng = cols / T[TBL_GSIZE];
        const uint hcn = a1, width = a2;
        for (uint i = tid; i < cols; i += NT) tlr[i] = sc[src + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = grow; d < width; d += nrow) {
          float acc[RMAXN];
          // rows h*width + d, h = 0..hcn-1: one simdgroup, four rows
          const device uint2* w2 =
              reinterpret_cast<const device uint2*>(W + T[TBL_WOFF]);
          const device BFT* sb =
              reinterpret_cast<const device BFT*>(W + T[TBL_SBOFF]);
          const uint n2 = (cols >> 3u) >> 1u;
          for (uint h = 0; h < hcn; ++h) acc[h] = 0.0f;
          for (uint bl = lane; bl < n2; bl += 32u) {
            float4 a0v = tlr4[bl * 4u + 0u], a1v = tlr4[bl * 4u + 1u];
            float4 a2v = tlr4[bl * 4u + 2u], a3v = tlr4[bl * 4u + 3u];
            float xs = hsum4(a0v) + hsum4(a1v) + hsum4(a2v) + hsum4(a3v);
            uint g = bl >> 2u;
            for (uint h = 0; h < hcn; ++h) {
              uint row = h * width + d;
              uint2 p = w2[row * n2 + bl];
              float pr = dot8(p.x, a0v, a1v) + dot8(p.y, a2v, a3v);
              acc[h] += float(sb[row * 2u * ng + g]) * pr
                      + float(sb[row * 2u * ng + ng + g]) * xs;
            }
          }
          for (uint h = 0; h < hcn; ++h) acc[h] = simd_sum(acc[h]);
          if (lane == 0u) {
            float total = 0.0f;
            for (uint h = 0; h < hcn; ++h)
              total += sigmoid_f(acc[h]) * sc[a0 + h * width + d];
            sc[dst + d] = total / float(hcn);
          }
        }
      }

      // ---------------------------------------------------- OP_INJECT (6)
      // residual + branch * inject, broadcast over the H streams.  Writes the
      // OTHER slab: the spike's U2 result is that a REUSED address is what
      // goes stale, and this kernel reuses every address 48 times.
      else if (op == 6u) {
        const uint hcn = a2;
        for (uint i = tg * NT + tid; i < hcn * HID; i += ntg * NT)
          sc[dst + i] = sc[src + i] + sc[a0 + (i % HID)] * sc[a1 + i / HID];
      }

      // ------------------------------------------------- OP_GDN_CORE (7)
      else if (op == 7u) {
        const device uint* TC = tbl + ent * TSTRIDE;
        const device BFT* convw = reinterpret_cast<const device BFT*>(
            WB[TC[TBL_GROUP]] + TC[TBL_WOFF]);
        const device uint* TA = tbl + a0 * TSTRIDE;
        const device BFT* alog = reinterpret_cast<const device BFT*>(
            WB[TA[TBL_GROUP]] + TA[TBL_WOFF]);
        const device uint* TD = tbl + a1 * TSTRIDE;
        const device BFT* dtb = reinterpret_cast<const device BFT*>(
            WB[TD[TBL_GROUP]] + TD[TBL_WOFF]);
        const device uint* TN = tbl + a2 * TSTRIDE;
        const device BFT* gnw = reinterpret_cast<const device BFT*>(
            WB[TN[TBL_GROUP]] + TN[TBL_WOFF]);
        device const BFT* cs_i = cs_in + (size_t)gdn_slot * (CK - 1u) * CD;
        device BFT* cs_o = cs_out + (size_t)gdn_slot * (CK - 1u) * CD;
        device const float* rec_i =
            rec_in + (size_t)gdn_slot * HV * DV * DK;
        device float* rec_o = rec_out + (size_t)gdn_slot * HV * DV * DK;
        device float* s_qkv = sc + SC_GDN_QKV;
        device float* s_z   = sc + SC_GDN_Z;
        device float* s_ba  = sc + SC_GDN_BA;

        // HV=48 value heads is this phase's WHOLE parallelism.  The strided
        // form is mandatory: the shipped G=40 is below 48, and the spike's
        // `if (tg < HV)` guard silently dropped heads 40..47 there.
        for (uint hv = tg; hv < HV; hv += ntg) {
          const uint hk = hv / RATIO;
          const uint ty = sg;
          const uint NDK = DK / 32u;
          const uint NDV = DV / NSGC;
          const uint KD = HK * DK;
          device const float* si = rec_i + (size_t)hv * DV * DK;
          device float* so = rec_o + (size_t)hv * DV * DK;
          float st[DV / NSGC][DK / 32u];
          for (uint j = 0; j < NDV; ++j) {
            uint dv = ty + NSGC * j;
            for (uint i = 0; i < NDK; ++i)
              st[j][i] = si[(size_t)dv * DK + NDK * lane + i];
          }
          for (uint idx = tid; idx < 2u * DK + DV; idx += NT) {
            uint p = idx / DK;
            uint d = idx - p * DK;
            uint c = p == 0u ? hk * DK + d
                   : (p == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);
            const device BFT* wc = convw + (size_t)c * CK;
            float a = 0.0f;
            for (uint tap = 0; tap + 1u < CK; ++tap)
              a += float(cs_i[(size_t)tap * CD + c]) * float(wc[tap]);
            a += s_qkv[c] * float(wc[CK - 1u]);
            float sl = silu_f(a);
            if (p == 0u) sq[d] = sl; else if (p == 1u) sk[d] = sl; else sv[d] = sl;
            if (p == 2u || (hv % RATIO) == 0u) {
              for (uint tap = 0; tap + 2u < CK; ++tap)
                cs_o[(size_t)tap * CD + c] = cs_i[(size_t)(tap + 1u) * CD + c];
              cs_o[(size_t)(CK - 2u) * CD + c] = static_cast<BFT>(s_qkv[c]);
            }
          }
          if (tid == 0u) {
            float av = s_ba[HV + hv] + float(dtb[hv]);
            shr[2] = metal::precise::exp(
                -metal::precise::exp(float(alog[hv])) * softplus_f(av));
            shr[3] = 1.0f / (1.0f + metal::precise::exp(-s_ba[hv]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (sg == 0u) {
            float pq = 0.0f, pk = 0.0f;
            uint b0 = 4u * lane;
            for (uint i = 0; i < 4u; ++i) {
              pq += sq[b0 + i] * sq[b0 + i];
              pk += sk[b0 + i] * sk[b0 + i];
            }
            pq = simd_sum(pq); pk = simd_sum(pk);
            if (lane == 0u) {
              shr[0] = metal::precise::rsqrt(pq + 1.0e-6f);
              shr[1] = metal::precise::rsqrt(pk + 1.0e-6f);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          for (uint d = tid; d < DK; d += NT) {
            sq[d] = sq[d] * shr[0] * QSCALE;
            sk[d] = sk[d] * shr[1];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          for (uint j = 0; j < NDV; ++j) {
            uint dv = ty + NSGC * j;
            float kv = 0.0f;
            for (uint i = 0; i < NDK; ++i) {
              uint s = NDK * lane + i;
              st[j][i] = st[j][i] * shr[2];
              kv += st[j][i] * sk[s];
            }
            kv = simd_sum(kv);
            float delta = (sv[dv] - kv) * shr[3];
            float o = 0.0f;
            for (uint i = 0; i < NDK; ++i) {
              uint s = NDK * lane + i;
              st[j][i] = st[j][i] + sk[s] * delta;
              o += st[j][i] * sq[s];
            }
            o = simd_sum(o);
            if (lane == 0u) sy[dv] = o;
            for (uint i = 0; i < NDK; ++i)
              so[(size_t)dv * DK + NDK * lane + i] = st[j][i];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (sg == 0u) {
            float po = 0.0f;
            uint b0 = 4u * lane;
            for (uint i = 0; i < 4u; ++i) po += sy[b0 + i] * sy[b0 + i];
            po = simd_sum(po);
            if (lane == 0u)
              shr[0] = metal::precise::rsqrt(po / float(DV) + NEPS);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          for (uint d = tid; d < DV; d += NT) {
            float n = sy[d] * shr[0] * float(gnw[d]);
            float zz = s_z[hv * DV + d];
            sc[dst + hv * DV + d] = n / (1.0f + metal::precise::exp(-zz));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        gdn_slot += 1u;
      }

      // -------------------------------------------------- OP_MOE_TOPK (8)
      // Recomputed in EVERY threadgroup, which is what makes it free: the
      // spec's phase 5 profiled at ~0.00 ms and needs no grid barrier.
      else if (op == 8u) {
        const uint ne = a1, kk = a0;
        for (uint e = tid; e < ne; e += NT) tlog[e] = sc[src + e];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
          const uint PER = ne / 32u;
          uint taken = 0u;
          float bestv[TOPKN]; uint besti[TOPKN];
          for (uint t = 0; t < kk; ++t) {
            float mv = -INFINITY; uint mi = 0u;
            for (uint j = 0; j < PER; ++j) {
              if (taken & (1u << j)) continue;
              float v = tlog[lane * PER + j];
              if (v > mv) { mv = v; mi = j; }
            }
            float gv = simd_max(mv);
            uint win = simd_min(mv == gv ? lane : 32u);
            uint gi = simd_shuffle(mi, win);
            if (lane == win) taken |= (1u << gi);
            bestv[t] = gv; besti[t] = win * PER + gi;
          }
          if (lane == 0u) {
            float m = bestv[0];
            for (uint t = 1; t < kk; ++t) m = metal::max(m, bestv[t]);
            float ssum = 0.0f; float ex[TOPKN];
            for (uint t = 0; t < kk; ++t) {
              ex[t] = metal::precise::exp(bestv[t] - m); ssum += ex[t];
            }
            for (uint t = 0; t < kk; ++t) {
              topi[t] = besti[t]; topw[t] = ex[t] / ssum;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Published to scratch too, purely so a test can read the routing
        // back.  Nothing in the kernel reads it from there, so it needs no
        // barrier and every threadgroup writes the same values.
        if (tg == 0u && tid == 0u)
          for (uint t = 0; t < kk; ++t) {
            sc[dst + t] = float(topi[t]);
            sc[SC_MOE_TOPW + t] = topw[t];
          }
      }

      // -------------------------------------------------- OP_MOE_E1 (9)
      // One fused [E, 2*FF, HID] table -- gate rows then up rows, the layout
      // transform_moe_weights already leaves resident, so this streams one
      // table and the pack does no concatenation.
      else if (op == 9u) {
        const device uint* T = tbl + ent * TSTRIDE;
        const device uint* W = WB[T[TBL_GROUP]];
        const uint cols = T[TBL_COLS], rows = T[TBL_ROWS];
        const uint ng = cols / T[TBL_GSIZE];
        const uint width = a1;
        for (uint i = tid; i < cols; i += NT) tgx[i] = sc[src + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r0 = grow * RGU; r0 < TOPKN * width; r0 += nrow * RGU) {
          uint e = r0 / width, j = r0 - e * width;
          uint eid = topi[e];
          float g_[RGU], u_[RGU];
          qmv4<RGU>(W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng, j, rows,
                    eid * rows, tgx4, lane, g_);
          qmv4<RGU>(W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng, width + j, rows,
                    eid * rows, tgx4, lane, u_);
          if (lane == 0u)
            for (uint r = 0; r < RGU; ++r)
              if (j + r < width) sc[dst + r0 + r] = silu_f(g_[r]) * u_[r];
        }
      }

      // -------------------------------------------------- OP_MOE_E2 (10)
      // The expert axis folded into the K axis: 10 experts x 40 uint2 blocks
      // = 400 over 32 lanes, so lanes idle 4% of steps instead of 37%.  The
      // spec's largest single win, 230 -> 325 GB/s.
      else if (op == 10u) {
        const device uint* T = tbl + ent * TSTRIDE;
        const device uint* W = WB[T[TBL_GROUP]];
        const uint cols = T[TBL_COLS], rows = T[TBL_ROWS];
        const uint ng = cols / T[TBL_GSIZE];
        const uint n2 = (cols >> 3u) >> 1u;
        const device uint2* w2 =
            reinterpret_cast<const device uint2*>(W + T[TBL_WOFF]);
        const device BFT* sb =
            reinterpret_cast<const device BFT*>(W + T[TBL_SBOFF]);
        const device float4* act4 =
            reinterpret_cast<const device float4*>(sc + src);
        for (uint r0 = grow * RDN; r0 < rows; r0 += nrow * RDN) {
          float acc[RDN];
          for (uint r = 0; r < RDN; ++r) acc[r] = 0.0f;
          for (uint gb = lane; gb < TOPKN * n2; gb += 32u) {
            uint e = gb / n2, bl = gb - e * n2;
            uint eid = topi[e]; float wg = topw[e];
            uint ab_ = e * (cols / 4u) + bl * 4u;
            float4 a0v = act4[ab_ + 0u], a1v = act4[ab_ + 1u];
            float4 a2v = act4[ab_ + 2u], a3v = act4[ab_ + 3u];
            float xs = hsum4(a0v) + hsum4(a1v) + hsum4(a2v) + hsum4(a3v);
            uint g = bl >> 2u;
            for (uint r = 0; r < RDN; ++r) {
              if (r0 + r >= rows) break;
              uint row = eid * rows + r0 + r;
              uint2 p = w2[row * n2 + bl];
              float pr = dot8(p.x, a0v, a1v) + dot8(p.y, a2v, a3v);
              acc[r] += wg * (float(sb[row * 2u * ng + g]) * pr
                            + float(sb[row * 2u * ng + ng + g]) * xs);
            }
          }
          for (uint r = 0; r < RDN; ++r) {
            float v = simd_sum(acc[r]);
            if (lane == 0u && r0 + r < rows) sc[dst + r0 + r] = v;
          }
        }
      }

      // --------------------------------------------- OP_SILU_MUL (15)
      else if (op == 15u) {
        for (uint i = tg * NT + tid; i < a1; i += ntg * NT)
          sc[dst + i] = sc[src + i] * sc[a0 + i];
      }

      // -------------------------------------------------- OP_ADD (14)
      else if (op == 14u) {
        float gate = sc[a1];
        for (uint i = tg * NT + tid; i < a0; i += ntg * NT)
          sc[dst + i] = sc[dst + i] + gate * sc[src + i];
      }

      // ------------------------------------------------- OP_COPY (13)
      else if (op == 13u) {
        for (uint i = tg * NT + tid; i < a0; i += ntg * NT)
          sc[dst + i] = sc[src + i];
      }

      if (bar == 2u) {
        live = gbar(ctr, ab, ntg, tid, phase, SPINCAP);
      } else if (bar == 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    if (!live) break;
    for (uint i = tg * NT + tid; i < HID; i += ntg * NT)
      out[(size_t)rep * HID + i] = static_cast<BFT>(sc[SC_MIXED + i]);
    live = gbar(ctr, ab, ntg, tid, phase, SPINCAP);
  }

  if (tg == 0u && tid == 0u) {
    status[0] = atomic_load_explicit(ab, memory_order_relaxed);
    status[1] = phase;
  }
"""

IN_NAMES = [
    "xin", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7",
    "tbl", "sched", "meta", "cs_in", "rec_in", "ctrl", "base",
]
OUT_NAMES = ["scratch", "out", "cs_out", "rec_out", "status"]

MAX_GROUPS = 8

_KERNEL_CACHE: dict[Any, Any] = {}


def build_body_kernel(*, threads: int = None, rdown: int = 4,
                      rgu: int = 2, rdn: int = 2, spin_cap: int = None):
    """Compile the schedule-walking kernel.  One binary for every schedule."""
    threads = _THREADS if threads is None else threads
    spin_cap = _SPIN_CAP if spin_cap is None else spin_cap
    nsg = threads // 32
    if threads % 32 or nsg > MAX_SIMDGROUPS:
        raise ValueError(f"threads={threads} must be a multiple of 32, <= 512")
    if GDN_VALUE_DIM % nsg:
        raise ValueError(
            f"GDN core splits {GDN_VALUE_DIM} value dims over {nsg} simdgroups"
        )
    key = (threads, rdown, rgu, rdn, spin_cap)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    subs = {
        "NT": threads, "NSGC": nsg, "SPINCAP": spin_cap,
        "HID": HIDDEN, "HCN": HC_COUNT, "HCH": HC_HIDDEN,
        "CD": CONV_DIM, "CK": CONV_KERNEL,
        "HV": GDN_VALUE_HEADS, "HK": GDN_KEY_HEADS, "RATIO": GDN_RATIO,
        "DK": GDN_KEY_DIM, "DV": GDN_VALUE_DIM,
        "TOPKN": TOPK, "TSTRIDE": TABLE_STRIDE,
        "RDOWN": rdown, "RGU": rgu, "RDN": rdn, "RMAXN": RMAX,
        "TGF": TG_FLOATS,
        "QSCALE": f"{GDN_KEY_DIM ** -0.5:.17g}f",
        "NEPS": f"{RMS_EPS:.17g}f",
        "BFT": "bfloat16_t",
    }
    for name, off in TG.items():
        subs[f"TG_{name}"] = off
    for name, off in SCRATCH.items():
        subs[f"SC_{name}"] = off

    import re as _re
    src, hdr = BODY_SRC, kernel_header() + BODY_HELPERS
    # Longest first, so HCH is not eaten by HC and NSGC not by NSG.
    for name in sorted(subs, key=len, reverse=True):
        pattern = _re.compile(r"\b" + name + r"\b")
        src = pattern.sub(str(subs[name]), src)
        hdr = pattern.sub(str(subs[name]), hdr)
    kernel = mx.fast.metal_kernel(
        name=f"qwen4_mega_t{threads}_d{rdown}_g{rgu}_n{rdn}",
        input_names=IN_NAMES, output_names=OUT_NAMES,
        source=src, header=hdr,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


class MegakernelBody:
    """Launches the persistent kernel over a packed model and a schedule."""

    def __init__(self, pack, schedule, *, gdn_layers: int,
                 threads: int = None, groups: int = None, **kw):
        if THREADGROUP_BYTES > 16 * 1024:
            raise ValueError(
                f"threadgroup arena {THREADGROUP_BYTES} B over the 16 KiB cap"
            )
        if len(pack.buffers) > MAX_GROUPS:
            raise ValueError(
                f"{len(pack.buffers)} packed groups over the {MAX_GROUPS} "
                "weight bindings this kernel declares"
            )
        self.pack = pack
        self.schedule = schedule
        self.gdn_layers = max(int(gdn_layers), 1)
        self.threads = _THREADS if threads is None else threads
        self.groups = _THREADGROUPS if groups is None else groups
        self.kernel = build_body_kernel(threads=self.threads, **kw)
        pad = mx.zeros((16,), mx.uint32)
        self.wbufs = list(pack.buffers) + [pad] * (MAX_GROUPS - len(pack.buffers))
        self.table = pack.table
        self.sched = schedule.to_array()
        mx.eval(self.wbufs, self.table, self.sched, pad)
        self.reset()

    def reset(self) -> None:
        self.ctrl = mx.zeros((64,), mx.uint32)
        mx.eval(self.ctrl)
        self.phase = 0

    def __call__(self, xin, cs_in, rec_in, *, reps: int = 1,
                 steps: Optional[int] = None):
        """``steps`` truncates the schedule, for cumulative phase profiling.

        The barrier map lives in the schedule, so a prefix is a real, running
        kernel with exactly the barriers its own phases need -- not a kernel
        with the tail compiled out.
        """
        nsteps = len(self.schedule) if steps is None else int(steps)
        meta = mx.array([nsteps, reps], mx.uint32)
        base = mx.array([self.phase], mx.uint32)
        outs = self.kernel(
            inputs=[xin, *self.wbufs, self.table, self.sched, meta,
                    cs_in, rec_in, self.ctrl, base],
            grid=(self.groups * self.threads, 1, 1),
            threadgroup=(self.threads, 1, 1),
            output_shapes=[
                (SCRATCH_FLOATS,), (reps, HIDDEN),
                (self.gdn_layers, CONV_KERNEL - 1, CONV_DIM),
                (self.gdn_layers, GDN_VALUE_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM),
                (4,),
            ],
            output_dtypes=[mx.float32, mx.bfloat16, mx.bfloat16,
                           mx.float32, mx.uint32],
        )
        # One grid barrier for the residual load, one after the output write,
        # plus the schedule's own device barriers, per repetition.
        bars = sum(1 for st in self.schedule.steps[:nsteps] if st.barrier == 2)
        self.phase += reps * (bars + 2)
        return outs
