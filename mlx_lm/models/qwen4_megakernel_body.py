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
import time
from typing import Any, Optional

import mlx.core as mx

from .qwen4_megakernel import (
    ACTL,
    ACTL_HEADER,
    ACTL_IDS,
    ACTL_M0,
    ACTL_M_STRIDE,
    BLOCK_TOPK,
    PLE_CONV_KERNEL,
    PLE_NGRAM,
    PLE_STATE_LEN,
    ROTARY_DIM,
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
    HEAD_DIM,
    HIDDEN,
    IDX_HEADS,
    IDX_HEAD_DIM,
    IDX_COMPRESS,
    SCORE_TILE_BLOCKS,
    KEY_DIM,
    N_KV_HEADS,
    N_Q_HEADS,
    Q_DIM,
    SDPA_BLOCKS,
    NUM_EXPERTS,
    RMS_EPS,
    MAX_QUERY_WIDTH,
    SCRATCH,
    SCRATCH_FLOATS,
    SCRATCH_STRIDE,
    score_tile_floats,
    scratch_floats,
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

def _tg_blocks(max_query_width: int):
    return (
        ("TGX", HIDDEN),          # staged source vector for a HIDDEN-wide matvec
        ("TLOG", NUM_EXPERTS),    # router logits, so top-k is threadgroup-local
        ("SQ", GDN_KEY_DIM),
        ("SK", GDN_KEY_DIM),
        ("SV", GDN_VALUE_DIM),
        ("SY", GDN_VALUE_DIM),
        ("TLR", HC_LOWRANK),      # hyper low-rank vector, re-read HIDDEN times
        ("RED", MAX_SIMDGROUPS),  # cross-simdgroup reduction slots
        ("PART", RMAX * MAX_SIMDGROUPS),   # split-K partials
        ("GSC", HC_COUNT),        # the four GroupRMSNorm scales
        ("SHR", 8),                # GDN core's shared scalars
        ("TOPW", max_query_width * TOPK),   # per-query routing weights
    )


def compute_tg_layout(max_query_width: int):
    """The threadgroup arena, sized for one build's query width.

    Every named block except ``TOPW`` is width-invariant, so the two builds a
    dual-width process holds share the same offsets for everything up to
    ``TOPW`` and differ only in the arena's tail and total size -- which is
    exactly what lets the narrow build's threadgroup footprint shrink instead
    of carrying the wide build's ``TOPW``/``topi`` allocation on every decode
    token.  Returns ``(tg_dict, tg_floats, threadgroup_bytes)``.
    """
    tg: dict[str, int] = {}
    off = 0
    for name, size in _tg_blocks(max_query_width):
        tg[name] = off
        off += size
    tg_floats = off
    # + the top-k expert ids, which are uint and live in their own array
    threadgroup_bytes = tg_floats * 4 + max_query_width * TOPK * 4
    return tg, tg_floats, threadgroup_bytes


# Module-level layout, at the process ceiling (``MAX_QUERY_WIDTH``).  Kept for
# every caller that references ``TG``/``TG_FLOATS``/``THREADGROUP_BYTES``
# directly -- a single-width build (the common case) never calls
# ``compute_tg_layout`` itself.  A dual-width build computes its own narrow
# layout separately; see ``build_body_kernel``.
TG, TG_FLOATS, THREADGROUP_BYTES = compute_tg_layout(MAX_QUERY_WIDTH)


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
// M is the QUERY WIDTH.  The weight is read ONCE and used for all M queries:
// `w2[woff[r] + bl]` is outside the m loop, and only the source words and the
// accumulators multiply.  That is the whole economics of a verify slab --
// weight traffic is what a decode matvec is made of, and it does not grow
// with the width.  `xs4` is the distance in float4s between one query's
// source vector and the next, so the same body serves a staged source (M
// vectors packed cols apart) and an unstaged one (M scratch planes SCSTRIDE
// apart).  At M = 1 every m loop is one iteration and the emitted code is
// what shipped.
template <uint R, uint M, typename F4>
inline void qmv4(const device uint* W, uint w_off, uint sb_off, uint cols,
                 uint ng, uint row0, uint rows, uint estride,
                 F4 x4, uint xs4, uint lane, thread float* acc) {
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
    for (uint m = 0; m < M; ++m) acc[r * M + m] = 0.0f;
  }
  for (uint bl = lane; bl < n2; bl += 32u) {
    float4 a[M * 4];
    float xs[M];
    for (uint m = 0; m < M; ++m) {
      const uint b = m * xs4 + bl * 4u;
      a[m*4+0] = x4[b + 0u]; a[m*4+1] = x4[b + 1u];
      a[m*4+2] = x4[b + 2u]; a[m*4+3] = x4[b + 3u];
      xs[m] = hsum4(a[m*4+0]) + hsum4(a[m*4+1])
            + hsum4(a[m*4+2]) + hsum4(a[m*4+3]);
    }
    uint g = bl >> 2u;
    for (uint r = 0; r < R; ++r) {
      uint2 p = w2[woff[r] + bl];
      float s0 = float(sb[soff[r] + g]), s1 = float(sb[soff[r] + ng + g]);
      for (uint m = 0; m < M; ++m) {
        float part = dot8(p.x, a[m*4+0], a[m*4+1])
                   + dot8(p.y, a[m*4+2], a[m*4+3]);
        acc[r * M + m] += s0 * part + s1 * xs[m];
      }
    }
  }
  for (uint r = 0; r < R * M; ++r) acc[r] = simd_sum(acc[r]);
}

template <uint R, uint M, typename F4>
inline void qmv8(const device uint* W, uint w_off, uint sb_off, uint cols,
                 uint ng, uint row0, uint rows, uint estride,
                 F4 x4, uint xs4, uint lane, thread float* acc) {
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
    for (uint m = 0; m < M; ++m) acc[r * M + m] = 0.0f;
  }
  for (uint bl = lane; bl < n2; bl += 32u) {
    float4 a[M * 2];
    float xs[M];
    for (uint m = 0; m < M; ++m) {
      const uint b = m * xs4 + bl * 2u;
      a[m*2+0] = x4[b + 0u]; a[m*2+1] = x4[b + 1u];
      xs[m] = hsum4(a[m*2+0]) + hsum4(a[m*2+1]);
    }
    uint g = bl >> 3u;
    for (uint r = 0; r < R; ++r) {
      uint2 p = w2[woff[r] + bl];
      float s0 = float(sb[soff[r] + g]), s1 = float(sb[soff[r] + ng + g]);
      for (uint m = 0; m < M; ++m) {
        float part = dot4x8(p.x, a[m*2+0]) + dot4x8(p.y, a[m*2+1]);
        acc[r * M + m] += s0 * part + s1 * xs[m];
      }
    }
  }
  for (uint r = 0; r < R * M; ++r) acc[r] = simd_sum(acc[r]);
}

// The same, but K is split ACROSS THE SIMDGROUPS of one threadgroup and the
// partials reduce through threadgroup memory.  The spec's "cheap form" of
// split-K: it costs no grid barrier.  It lost on the MoE down projection and
// wins 1.37x on the hyper down-mix -- the difference is row count, not the
// technique.  324 rows over 640 simdgroups leaves 87% of them idle otherwise.
template <uint R, uint M, typename F4>
inline void qmv4_ksplit(const device uint* W, uint w_off, uint sb_off,
                        uint cols, uint ng, uint row0, uint rows,
                        F4 x4, uint xs4, uint lane, uint sg, uint nsg,
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
    for (uint m = 0; m < M; ++m) acc[r * M + m] = 0.0f;
  }
  for (uint bl = sg * 32u + lane; bl < n2; bl += nsg * 32u) {
    float4 a[M * 4];
    float xs[M];
    for (uint m = 0; m < M; ++m) {
      const uint b = m * xs4 + bl * 4u;
      a[m*4+0] = x4[b + 0u]; a[m*4+1] = x4[b + 1u];
      a[m*4+2] = x4[b + 2u]; a[m*4+3] = x4[b + 3u];
      xs[m] = hsum4(a[m*4+0]) + hsum4(a[m*4+1])
            + hsum4(a[m*4+2]) + hsum4(a[m*4+3]);
    }
    uint g = bl >> 2u;
    for (uint r = 0; r < R; ++r) {
      uint2 p = w2[woff[r] + bl];
      float s0 = float(sb[soff[r] + g]), s1 = float(sb[soff[r] + ng + g]);
      for (uint m = 0; m < M; ++m) {
        float part = dot8(p.x, a[m*4+0], a[m*4+1])
                   + dot8(p.y, a[m*4+2], a[m*4+3]);
        acc[r * M + m] += s0 * part + s1 * xs[m];
      }
    }
  }
  for (uint r = 0; r < R * M; ++r) acc[r] = simd_sum(acc[r]);
}

// Which opcodes rerun per query, and which take the slab inside their own
// inner loop.  The distinction is whether the phase has a WEIGHT to amortise:
// a matvec reads its weight once for all M queries, a norm against a shared
// gain vector has nothing to share and simply runs M times.  Attention, the
// GDN core and the ledgers are amortising for the same reason -- they reuse
// every K/V read, read and write the recurrent state once, and append M
// columns in one pass.
//
// A bitmask rather than a switch: `op` is uniform across the threadgroup, so
// this is a scalar test in front of a loop, not a per-thread branch.
// One place where R and the bit width are resolved, templated on the query
// width.  Written once so every call site -- the generic matvec, the hyper
// mixers, the MoE -- widens by passing a different M rather than by growing
// its own if-chain.
template <uint M, typename F4>
inline void qmv_any(bool bits4, uint R, const device uint* W, uint w_off,
                    uint sb_off, uint cols, uint ng, uint row0, uint rows,
                    uint estride, F4 x4, uint xs4, uint lane,
                    thread float* acc) {
  if (bits4) {
    if (R == 4u)      qmv4<4, M>(W, w_off, sb_off, cols, ng, row0, rows,
                                 estride, x4, xs4, lane, acc);
    else if (R == 2u) qmv4<2, M>(W, w_off, sb_off, cols, ng, row0, rows,
                                 estride, x4, xs4, lane, acc);
    else              qmv4<1, M>(W, w_off, sb_off, cols, ng, row0, rows,
                                 estride, x4, xs4, lane, acc);
  } else {
    if (R == 4u)      qmv8<4, M>(W, w_off, sb_off, cols, ng, row0, rows,
                                 estride, x4, xs4, lane, acc);
    else if (R == 2u) qmv8<2, M>(W, w_off, sb_off, cols, ng, row0, rows,
                                 estride, x4, xs4, lane, acc);
    else              qmv8<1, M>(W, w_off, sb_off, cols, ng, row0, rows,
                                 estride, x4, xs4, lane, acc);
  }
}

// Build the union of the M queries' top-k expert lists, in a deterministic
// order: query 0's choices in its own order first, then whatever query 1 adds,
// and so on.  At M = 1 the union IS query 0's list in query 0's order, which
// is what makes the width-1 MoE path bit-identical to what shipped.
//
// `uslot[u]` packs, in a byte per query, that query's own slot for union
// expert `u` plus one; 0 means the query did not choose it.  The slot, not the
// union position, is what indexes the query's E1 output and its routing
// weight.
//
// One thread does it.  At M = 3 and k = 10 that is at most 30 insertions
// against a list of at most 30, i.e. ~450 comparisons once per phase, against
// the megabytes of expert weight the phase is about to read.
inline void build_expert_union(const threadgroup uint* topi, uint MWq,
                               uint tid, threadgroup uint* uni,
                               threadgroup uint* uslot,
                               threadgroup uint* unin) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint n = 0u;
    for (uint m = 0; m < MWq; ++m) {
      for (uint t = 0; t < TOPKN; ++t) {
        const uint e = topi[m * TOPKN + t];
        uint at = n;
        for (uint u = 0; u < n; ++u) if (uni[u] == e) { at = u; break; }
        if (at == n) { uni[n] = e; uslot[n] = 0u; n += 1u; }
        uslot[at] |= (t + 1u) << (8u * m);
      }
    }
    unin[0] = n;
  }
}

inline bool per_query_op(uint op) {
  return ((PERQMASK >> op) & 1u) != 0u;
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
  const uint nsteps = actl[19];
  const uint reps   = actl[20];
  // The barrier's generation base travels in `actl`, not its own binding: a
  // four-word buffer was worth as much of the 31-binding budget as an 8 GiB
  // weight group, and width 3 needs the slot.
  uint phase = actl[22];
  // QUERY WIDTH.  1 is a draft token; k+1 is a verify slab, whose queries sit
  // in M scratch planes SCSTRIDE floats apart and whose per-query control --
  // position, block count, and above all the incomplete TAIL block, which
  // ADVANCES mid-slab -- lives in the per-query blocks at ACTLM0.
  //
  // A MAXMW=1 build never receives mwidth > 1 (the host asserts it), so the
  // actl[23] read is provably 1 every launch -- but the compiler cannot see
  // through a device-memory load, and every mq-loop, mc offset and MW==1u/
  // MW==2u/else dispatch chain below stays generic runtime logic on top of a
  // narrowed register footprint.  #if turns MW into an actual compile-time
  // constant on the narrow build (the preprocessor drops the actl[23] read
  // entirely, before the compiler ever sees it), which is what lets every
  // site below fold: the mq-loops collapse to their mq=0 iteration, ACTLM0 +
  // 0*ACTLMS becomes the same address the header already read, and the
  // MW==2u/else specializations are dead code.  A MAXMW=3 build is untouched
  // -- MW is still the runtime read, same as before this change.
#if MAXMW == 1
  const uint MW = 1u;
#else
  const uint MW = actl[23] == 0u ? 1u : actl[23];
#endif
  // Host inputs per repetition.  The first HCH floats are the residual
  // streams; a surplus is the MTP head's embedding, which its `fuse` reads
  // from SC_BRANCH.  Anything the schedule reads must be WRITTEN here --
  // scratch is a fresh allocation every call, so a phase that reads an
  // uninitialised slot is grid-dependent garbage, which is exactly what the
  // G-sweep coverage test reports as "differs".
  const uint xw = actl[21] == 0u ? HCH : actl[21];

  // The two merged ledgers, split back into the four names the phase bodies
  // use.  Both offsets are products of control words the token already
  // carries -- no new binding, no new field.
  const device BFT* kbuf = kv;
  const device BFT* vbuf =
      kv + (size_t)actl[17] * NKVH * (size_t)actl[0] * HD;
  const device BFT* rawk = idxl;
  const device BFT* pooled =
      idxl + (size_t)actl[17] * (size_t)actl[0] * IDXD;

  // PLE state is transactional like the GDN state.  Copy it to an output
  // owned by the launch so a rejected verify slab leaves its input intact.
  // The same global thread owns a channel here and in OP_PLE_CONV, so no
  // device barrier is needed between the copy and that later phase.
  for (uint c = tg * NT + tid; c < HCH; c += ntg * NT)
    for (uint r = 0; r < PLES; ++r)
      pconv_out[(size_t)r * HCH + c] = pconv[(size_t)r * HCH + c];

  // Every packed group is a binding; the table's `group` field selects one.
  const device uint* WB[10] = {w0, w1, w2, w3, w4, w5, w6, w7, w8, w9};

  // Plane 0.  A phase body gets its own `sc` inside the per-query loop; this
  // one serves the residual load and the phases that address plane 0 by name.
  device float* sc = scratch;
  // Block ids for the attention phase. ``OP_INDEX_TOPB`` writes them into
  // scratch through a uint view; a standalone probe hands them in through
  // ``actl`` instead. Index scores use a separate, dynamically tile-aligned
  // output so activation scratch no longer imposes a context ceiling.
  const device uint* sel_ids = actl + ACTLIDS;
  const uint ids_from_scratch = actl[7];
  const uint score_stride = actl[24];

  threadgroup float A[TGF];
  // Routing is PER QUERY -- the M tokens of a slab pick their own top-10 --
  // so both arrays carry a query plane.  30 uints and 30 floats is 160 B of
  // the arena's 320 B of headroom.
  threadgroup uint  topi[MAXMW * TOPKN];
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
  // The routed-expert UNION, aliased over the GDN core's staging blocks.
  // `SQ`/`SK` are live only inside OP_GDN_CORE, which is a different layer
  // branch and never runs beside the MoE, so 128 floats there are free here
  // -- the same alias trade `tlog` and the combine's transpose already make.
  threadgroup uint* uni =
      reinterpret_cast<threadgroup uint*>(A + TG_SQ);          // MAXMW*TOPKN
  threadgroup uint* uslot = uni + MAXMW * TOPKN;               // packed slots
  threadgroup uint* unin = uslot + MAXMW * TOPKN;              // union size
  const threadgroup float4* tgx4 =
      reinterpret_cast<const threadgroup float4*>(A + TG_TGX);
  const threadgroup float4* tlr4 =
      reinterpret_cast<const threadgroup float4*>(A + TG_TLR);

  bool live = true;
  for (uint rep = 0; rep < reps && live; ++rep) {
    // The residual streams arrive hoisted: PLE's n-gram gather and the
    // embedding depend only on the input token, so they are host work.
    // M embeddings for a width-M slab, one per plane.  `xin` is laid out
    // (rep, query, xw): a verify slab hands in the embedding of each of its
    // k+1 tokens, and each lands in its own scratch plane.
    for (uint mq = 0; mq < MW; ++mq) {
      device float* scm = scratch + (size_t)mq * SCSTRIDE;
      const size_t xbase = ((size_t)rep * MW + mq) * xw;
      for (uint i = tg * NT + tid; i < xw; i += ntg * NT) {
        const float v = float(xin[xbase + i]);
        if (i < HCH)            scm[SC_RESID_A + i] = v;
        else if (i < HCH + HID) scm[SC_BRANCH + (i - HCH)] = v;
        else                    scm[SC_PLE_EMB + (i - HCH - HID)] = v;
      }
    }
    live = gbar(ctr, ab, ntg, tid, phase, SPINCAP);
    if (!live) break;

    uint gdn_slot = 0u;
    for (uint step = 0; step < nsteps && live; ++step) {
      const device uint* S = sched + step * 8u;
      const uint op = S[0], ent = S[1], src = S[2], dst = S[3];
      const uint a0 = S[4], a1 = S[5], a2 = S[6], bar = S[7];

      // ------------------------------------------------- the query loop
      // Two classes of phase, and the split is the whole economics of a
      // verify slab.
      //
      // AMORTISING phases read a weight and use it for every query in the
      // slab, so they take M inside their own inner loop and run ONCE: the
      // weight traffic, which is what a decode token is made of, is paid
      // once for the whole slab.  Those are the matvecs, attention (one
      // simdgroup owns a (head, block) unit and reuses every K/V read), the
      // GDN core (the 3.1 MB/layer state is read and written once and the
      // delta rule is applied sequentially over the queries), and the
      // ledgers.
      //
      // PER-QUERY phases have no weight to amortise -- norms against a
      // shared gain vector, elementwise work, the router's own selection --
      // so they simply run M times with `sc` rebased onto the query's
      // plane.  That is ~3x of a small term, which is the plan's estimate
      // and is what makes the slab worth having: 3x of 0.35 ms beside 1.05x
      // of the matvecs.
      const uint mloops = per_query_op(op) ? MW : 1u;
      for (uint mq = 0; mq < mloops; ++mq) {
      device float* sc = scratch + (size_t)mq * SCSTRIDE;
      // This query's control block.  Query 0's block is mirrored into the
      // shared header, so an un-widened body reading `actl[k]` reads query 0.
      const device uint* mc = actl + ACTLM0 + mq * ACTLMS;

      // ------------------------------------------------------- OP_QMV (1)
      if (op == 1u) {
        const device uint* T = tbl + ent * TSTRIDE;
        const device uint* W = WB[T[TBL_GROUP]];
        const uint cols = T[TBL_COLS], rows = T[TBL_ROWS];
        const uint ng = cols / T[TBL_GSIZE];
        const uint bits = T[TBL_BITS];
        const uint woff = T[TBL_WOFF], sboff = T[TBL_SBOFF];
        const uint R = (a0 == 0u) ? 1u : a0;
        // ------------------------------------------- E2: where M sources come from
        // Stage the source when it fits: G threadgroups x NSG simdgroups all
        // stream the same vector, and a HIDDEN-wide one is 10 KiB.  A width-M
        // slab wants M of them, and `TGX` is 10 KiB of a 16,064 B arena whose
        // hard cap is 16 KiB, so at M = 3 only sources of at most HID / 3
        // columns still fit.  Everything wider reads its M sources straight
        // from device scratch, one plane apart -- option (b) of the E2 fork,
        // measured against a K-tiled staging in
        // results/qwen4-megakernel-width3-20260903-qmv.py.  The unstaged path
        // already existed (`cols > HID` selected it) and the L2 absorbs the
        // re-read, since every threadgroup wants the same words.
        //
        // At M = 1 the predicate is `cols <= HID`, exactly as it shipped.
        const bool staged = (MW * cols) <= HID;
        if (staged) {
          for (uint i = tid; i < MW * cols; i += NT) {
            const uint m = i / cols, c = i - m * cols;
            tgx[i] = scratch[(size_t)m * SCSTRIDE + src + c];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Distance in float4s between one query's source and the next: the
        // staged copies are packed `cols` apart, the unstaged ones are a whole
        // scratch plane apart.
        const uint sxs4 = cols / 4u;
        const uint dxs4 = SCSTRIDE / 4u;
        const device float4* dx4 =
            reinterpret_cast<const device float4*>(scratch + src);
        for (uint r0 = grow * R; r0 < rows; r0 += nrow * R) {
          float acc[RMAXN * MAXMW];
          if (MW == 1u) {
            if (staged) qmv_any<1>(bits == 4u, R, W, woff, sboff, cols, ng,
                                   r0, rows, 0u, tgx4, sxs4, lane, acc);
            else        qmv_any<1>(bits == 4u, R, W, woff, sboff, cols, ng,
                                   r0, rows, 0u, dx4, dxs4, lane, acc);
          } else if (MW == 2u) {
            if (staged) qmv_any<2>(bits == 4u, R, W, woff, sboff, cols, ng,
                                   r0, rows, 0u, tgx4, sxs4, lane, acc);
            else        qmv_any<2>(bits == 4u, R, W, woff, sboff, cols, ng,
                                   r0, rows, 0u, dx4, dxs4, lane, acc);
          } else {
            if (staged) qmv_any<MAXMW>(bits == 4u, R, W, woff, sboff, cols, ng,
                                       r0, rows, 0u, tgx4, sxs4, lane, acc);
            else        qmv_any<MAXMW>(bits == 4u, R, W, woff, sboff, cols, ng,
                                       r0, rows, 0u, dx4, dxs4, lane, acc);
          }
          if (lane == 0u) {
            // arg2 == 1 is DST_OUT: 248,320 vocabulary rows do not fit the
            // scratch, so lm_head writes its own output buffer -- one row per
            // query, so a verify slab returns M logit vectors.
            if (a2 == 1u) {
              for (uint r = 0; r < R; ++r)
                if (r0 + r < rows)
                  for (uint m = 0; m < MW; ++m)
                    logits[((size_t)rep * MW + m) * rows + r0 + r] =
                        static_cast<BFT>(
                            apply_act(acc[r * MW + m], a1));
            } else {
              for (uint r = 0; r < R; ++r)
                if (r0 + r < rows)
                  for (uint m = 0; m < MW; ++m)
                    scratch[(size_t)m * SCSTRIDE + dst + r0 + r] =
                        apply_act(acc[r * MW + m], a1);
            }
          }
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
        // The M sources are whole scratch planes apart, and the hyper
        // down-mix is 10,240 wide -- three of those is 120 KiB, so staging is
        // not on the table here at any width.
        const device float4* x4 =
            reinterpret_cast<const device float4*>(scratch + src);
        const uint dxs4 = SCSTRIDE / 4u;
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
          float acc[RMAXN * MAXMW];
          // The weight is read ONCE for all M queries; only the split-K
          // partials are per query.  `PART` is RMAX x 16 floats and the
          // threadgroup arena has 320 B free, so it cannot be widened by M --
          // instead the M queries take turns through the SAME buffer, one
          // threadgroup barrier apiece.  Barriers are the cheap resource
          // here: the expensive one, the quantized weight read, is shared.
          if (MW == 1u)
            qmv4_ksplit<RDOWN, 1>(WW, wo, so, nc, nn, rr, cap, x4, dxs4,
                                  lane, sg, NSG, acc);
          else if (MW == 2u)
            qmv4_ksplit<RDOWN, 2>(WW, wo, so, nc, nn, rr, cap, x4, dxs4,
                                  lane, sg, NSG, acc);
          else
            qmv4_ksplit<RDOWN, MAXMW>(WW, wo, so, nc, nn, rr, cap, x4, dxs4,
                                      lane, sg, NSG, acc);
          for (uint m = 0; m < MW; ++m) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u)
              for (uint r = 0; r < RDOWN; ++r)
                part[r * NSG + sg] = acc[r * MW + m];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
              device float* scm = scratch + (size_t)m * SCSTRIDE;
              for (uint r = 0; r < RDOWN; ++r) {
                if (rr + r >= cap) break;
                float total = 0.0f;
                for (uint j = 0; j < NSG; ++j) total += part[r * NSG + j];
                if (inj) scm[a1 + rr + r] = apply_act(total, 4u);
                else     scm[dst + rr + r] = apply_act(total, 3u);
              }
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
        // `TLR` is HC_LOWRANK floats and cannot be tripled -- the arena has
        // 320 B free -- so a slab stages query 0 and reads the rest from
        // their planes.  Every threadgroup wants the same 320 floats, so the
        // re-read is L2-resident; the weight, which is the expensive read,
        // is shared across all M either way.
        for (uint i = tid; i < cols; i += NT) tlr[i] = scratch[src + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint pstride4 = SCSTRIDE / 4u;
        const device float4* dlr4 =
            reinterpret_cast<const device float4*>(scratch + src);
        for (uint d = grow; d < width; d += nrow) {
          float acc[RMAXN * MAXMW];
          // rows h*width + d, h = 0..hcn-1: one simdgroup, four rows
          const device uint2* w2 =
              reinterpret_cast<const device uint2*>(W + T[TBL_WOFF]);
          const device BFT* sb =
              reinterpret_cast<const device BFT*>(W + T[TBL_SBOFF]);
          const uint n2 = (cols >> 3u) >> 1u;
          for (uint i = 0; i < hcn * MW; ++i) acc[i] = 0.0f;
          for (uint bl = lane; bl < n2; bl += 32u) {
            float4 a[MAXMW * 4];
            float xs[MAXMW];
            for (uint m = 0; m < MW; ++m) {
              if (m == 0u) {
                a[0] = tlr4[bl * 4u + 0u]; a[1] = tlr4[bl * 4u + 1u];
                a[2] = tlr4[bl * 4u + 2u]; a[3] = tlr4[bl * 4u + 3u];
              } else {
                const device float4* p4 = dlr4 + (size_t)m * pstride4;
                a[m*4+0] = p4[bl * 4u + 0u]; a[m*4+1] = p4[bl * 4u + 1u];
                a[m*4+2] = p4[bl * 4u + 2u]; a[m*4+3] = p4[bl * 4u + 3u];
              }
              xs[m] = hsum4(a[m*4+0]) + hsum4(a[m*4+1])
                    + hsum4(a[m*4+2]) + hsum4(a[m*4+3]);
            }
            uint g = bl >> 2u;
            for (uint h = 0; h < hcn; ++h) {
              uint row = h * width + d;
              uint2 p = w2[row * n2 + bl];
              float s0 = float(sb[row * 2u * ng + g]);
              float s1 = float(sb[row * 2u * ng + ng + g]);
              for (uint m = 0; m < MW; ++m) {
                float pr = dot8(p.x, a[m*4+0], a[m*4+1])
                         + dot8(p.y, a[m*4+2], a[m*4+3]);
                acc[h * MW + m] += s0 * pr + s1 * xs[m];
              }
            }
          }
          for (uint i = 0; i < hcn * MW; ++i) acc[i] = simd_sum(acc[i]);
          if (lane == 0u) {
            for (uint m = 0; m < MW; ++m) {
              device float* scm = scratch + (size_t)m * SCSTRIDE;
              float total = 0.0f;
              for (uint h = 0; h < hcn; ++h)
                total += sigmoid_f(acc[h * MW + m])
                       * scm[a0 + h * width + d];
              scm[dst + d] = total / float(hcn);
            }
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
        // Per-query views come from the plane inside the query loop below.

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
          // ------------------------------------------- the slab, in order
          // The delta rule is SEQUENTIAL over the queries: query m updates
          // the state query m-1 left.  The state stays in registers across
          // the whole slab, so the 3.1 MB/layer recurrent state is read once
          // and written once no matter how wide the slab is -- which is why
          // this phase costs ~1.5x rather than M x.
          //
          // ONE state is written: the one after the LAST query.  Widening
          // `rec_out` to carry a restore point per query costs no binding but
          // +227 MB of write traffic per token (`rec_out` alone is
          // 36 x 48 x 128 x 128 x 4 = 113 MB, ~0.45 ms at 500 GB/s), which is
          // a third of the whole token budget spent on a rollback that mostly
          // does not happen.  The contract the unified fused GDN verify
          // kernel already settled is the other one: `cs_in`/`rec_in` are a
          // separate buffer from `cs_out`/`rec_out`, so the PRE-SLAB state
          // survives the launch untouched, and a partial accept re-launches a
          // narrower slab from it.  Re-launch on rejection, not restore
          // points.
          //
          // The conv window rolls with the slab.  `2*DK + DV` = 384 channels
          // and NT = 512 threads, so one thread owns at most one channel and
          // its CK-1 taps live in registers across the queries; only the
          // final window reaches `cs_o`.
          // Channels per thread.  384 channels over NT threads; at the
          // shipped NT = 512 that is one apiece, and the general form keeps
          // the narrow geometries the coverage tests build.
          const uint NCH = 2u * DK + DV;
          const uint CPT = (NCH + NT - 1u) / NT;
          uint cp[CPT], cd_[CPT], cc[CPT];
          float win[CPT][CK - 1u];
          for (uint q = 0; q < CPT; ++q) {
            const uint cidx = tid + q * NT;
            if (cidx >= NCH) { cp[q] = 3u; continue; }   // 3 = no channel
            cp[q] = cidx / DK; cd_[q] = cidx - cp[q] * DK;
            cc[q] = cp[q] == 0u ? hk * DK + cd_[q]
                  : (cp[q] == 1u ? KD + hk * DK + cd_[q]
                                 : 2u * KD + hv * DV + cd_[q]);
            for (uint tap = 0; tap + 1u < CK; ++tap)
              win[q][tap] = float(cs_i[(size_t)tap * CD + cc[q]]);
          }

          for (uint mq2 = 0; mq2 < MW; ++mq2) {
          device float* scm  = scratch + (size_t)mq2 * SCSTRIDE;
          device float* s_qkv = scm + SC_GDN_QKV;
          device float* s_z   = scm + SC_GDN_Z;
          device float* s_ba  = scm + SC_GDN_BA;
          threadgroup_barrier(mem_flags::mem_threadgroup);
          for (uint q = 0; q < CPT; ++q) {
            if (cp[q] == 3u) continue;
            const device BFT* wc = convw + (size_t)cc[q] * CK;
            const float xin_c = s_qkv[cc[q]];
            float a = 0.0f;
            for (uint tap = 0; tap + 1u < CK; ++tap)
              a += win[q][tap] * float(wc[tap]);
            a += xin_c * float(wc[CK - 1u]);
            float sl = silu_f(a);
            if (cp[q] == 0u) sq[cd_[q]] = sl;
            else if (cp[q] == 1u) sk[cd_[q]] = sl;
            else sv[cd_[q]] = sl;
            // Roll the window: the taps a bf16 ledger would have held.  The
            // round trip through bf16 is kept so the slab's later queries see
            // exactly what a sequence of single-token launches would have
            // left them.
            for (uint tap = 0; tap + 2u < CK; ++tap)
              win[q][tap] = win[q][tap + 1u];
            win[q][CK - 2u] = float(static_cast<BFT>(xin_c));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
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
            // The state stays in registers; only the state after the LAST
            // query reaches `rec_out`.  See the restore-point note above.
            if (mq2 + 1u == MW)
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
            scm[dst + hv * DV + d] = n / (1.0f + metal::precise::exp(-zz));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          }   // the slab, in order

          // The conv window AFTER the last query, written once.  The guard is
          // the original one: q/k channels are shared across the RATIO value
          // heads that map to one key head, so only the first writes them.
          for (uint q = 0; q < CPT; ++q) {
            if (cp[q] == 3u) continue;
            if (!(cp[q] == 2u || (hv % RATIO) == 0u)) continue;
            for (uint tap = 0; tap + 1u < CK; ++tap)
              cs_o[(size_t)tap * CD + cc[q]] = static_cast<BFT>(win[q][tap]);
          }
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
              topi[mq * TOPKN + t] = besti[t];
              topw[mq * TOPKN + t] = ex[t] / ssum;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Published to scratch too, purely so a test can read the routing
        // back.  Nothing in the kernel reads it from there, so it needs no
        // barrier and every threadgroup writes the same values.
        if (tg == 0u && tid == 0u)
          for (uint t = 0; t < kk; ++t) {
            sc[dst + t] = float(topi[mq * TOPKN + t]);
            sc[SC_MOE_TOPW + t] = topw[mq * TOPKN + t];
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

        // ------------------------------------------------ the expert UNION
        // THE one term that does not amortise.  Every other phase's weight
        // traffic is width-invariant; here the M queries choose their own
        // top-10, so the read is the size of the UNION of their choices --
        // 10 if they route identically, 30 if disjoint.  Loop the union ONCE
        // and mask per query: one straight-line loop, no per-query branch to
        // serialise the simdgroup, and the expert weight is read once no
        // matter how many queries want it.
        //
        // `uslot[u]` packs, in a byte per query, that query's own slot for
        // union expert `u`, plus one -- 0 means "this query did not choose
        // it".  The slot matters because the E1 output and the E2 activation
        // are indexed by the query's OWN slot, not by the union position.
        build_expert_union(topi, MW, tid, uni, uslot, unin);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nu = unin[0];

        const bool staged = (MW * cols) <= HID;
        if (staged) {
          for (uint i = tid; i < MW * cols; i += NT) {
            const uint m = i / cols, c = i - m * cols;
            tgx[i] = scratch[(size_t)m * SCSTRIDE + src + c];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint sxs4 = cols / 4u, dxs4 = SCSTRIDE / 4u;
        const device float4* dx4 =
            reinterpret_cast<const device float4*>(scratch + src);

        for (uint r0 = grow * RGU; r0 < nu * width; r0 += nrow * RGU) {
          uint u = r0 / width, j = r0 - u * width;
          uint eid = uni[u], packed = uslot[u];
          float g_[RMAXN * MAXMW], u_[RMAXN * MAXMW];
          if (MW == 1u) {
            if (staged) {
              qmv_any<1>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                         j, rows, eid * rows, tgx4, sxs4, lane, g_);
              qmv_any<1>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                         width + j, rows, eid * rows, tgx4, sxs4, lane, u_);
            } else {
              qmv_any<1>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                         j, rows, eid * rows, dx4, dxs4, lane, g_);
              qmv_any<1>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                         width + j, rows, eid * rows, dx4, dxs4, lane, u_);
            }
          } else if (MW == 2u) {
            if (staged) {
              qmv_any<2>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                         j, rows, eid * rows, tgx4, sxs4, lane, g_);
              qmv_any<2>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                         width + j, rows, eid * rows, tgx4, sxs4, lane, u_);
            } else {
              qmv_any<2>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                         j, rows, eid * rows, dx4, dxs4, lane, g_);
              qmv_any<2>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                         width + j, rows, eid * rows, dx4, dxs4, lane, u_);
            }
          } else if (staged) {
            qmv_any<MAXMW>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                           j, rows, eid * rows, tgx4, sxs4, lane, g_);
            qmv_any<MAXMW>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                           width + j, rows, eid * rows, tgx4, sxs4, lane, u_);
          } else {
            qmv_any<MAXMW>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                           j, rows, eid * rows, dx4, dxs4, lane, g_);
            qmv_any<MAXMW>(true, RGU, W, T[TBL_WOFF], T[TBL_SBOFF], cols, ng,
                           width + j, rows, eid * rows, dx4, dxs4, lane, u_);
          }
          if (lane == 0u) {
            for (uint m = 0; m < MW; ++m) {
              const uint slot1 = (packed >> (8u * m)) & 255u;
              if (slot1 == 0u) continue;          // query m skipped this one
              device float* scm = scratch + (size_t)m * SCSTRIDE;
              const uint base = (slot1 - 1u) * width + j;
              for (uint r = 0; r < RGU; ++r)
                if (j + r < width)
                  scm[dst + base + r] =
                      silu_f(g_[r * MW + m]) * u_[r * MW + m];
            }
          }
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
            reinterpret_cast<const device float4*>(scratch + src);
        const uint pstride4 = SCSTRIDE / 4u;
        // The union again, and the same shape: the K fold now runs over the
        // UNION's experts and each query contributes only through the ones it
        // selected.  At M = 1 the union IS query 0's top-10 in its own order,
        // so both the loop bounds and the accumulation order are what shipped.
        build_expert_union(topi, MW, tid, uni, uslot, unin);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nu = unin[0];
        for (uint r0 = grow * RDN; r0 < rows; r0 += nrow * RDN) {
          float acc[RDN * MAXMW];
          for (uint i = 0; i < RDN * MW; ++i) acc[i] = 0.0f;
          for (uint gb = lane; gb < nu * n2; gb += 32u) {
            uint u = gb / n2, bl = gb - u * n2;
            uint eid = uni[u], packed = uslot[u];
            uint g = bl >> 2u;
            // The weight read is OUTSIDE the query loop: one expert row read
            // serves every query in the union that chose it.
            for (uint r = 0; r < RDN; ++r) {
              if (r0 + r >= rows) break;
              uint row = eid * rows + r0 + r;
              uint2 p = w2[row * n2 + bl];
              float s0 = float(sb[row * 2u * ng + g]);
              float s1 = float(sb[row * 2u * ng + ng + g]);
              for (uint m = 0; m < MW; ++m) {
                const uint slot1 = (packed >> (8u * m)) & 255u;
                if (slot1 == 0u) continue;
                const device float4* am4 = act4 + (size_t)m * pstride4;
                uint ab_ = (slot1 - 1u) * (cols / 4u) + bl * 4u;
                float4 a0v = am4[ab_ + 0u], a1v = am4[ab_ + 1u];
                float4 a2v = am4[ab_ + 2u], a3v = am4[ab_ + 3u];
                float xs = hsum4(a0v) + hsum4(a1v) + hsum4(a2v) + hsum4(a3v);
                float pr = dot8(p.x, a0v, a1v) + dot8(p.y, a2v, a3v);
                acc[r * MW + m] +=
                    topw[m * TOPKN + (slot1 - 1u)] * (s0 * pr + s1 * xs);
              }
            }
          }
          for (uint r = 0; r < RDN; ++r)
            for (uint m = 0; m < MW; ++m) {
              float v = simd_sum(acc[r * MW + m]);
              if (lane == 0u && r0 + r < rows)
                scratch[(size_t)m * SCSTRIDE + dst + r0 + r] = v;
            }
        }
      }


      // ------------------------------------------------------ OP_ATTN (11)
      // Pass 1 of indexed split-K attention, arithmetic-for-arithmetic with
      // qwen4_qsa_indexed's `_SOURCE` -- itself a clone of MLX's native
      // sdpa_vector_2pass_1 on the compact token order.  Everything here is
      // load-bearing for bit-identity: the lane -> d map (`lane*elements +
      // part`), the online-softmax rescale order, and `fast::exp` rather than
      // the precise one the rest of this kernel uses.
      //
      // The shipped kernel's `S` splits only distribute the fixed 128 blocks
      // over threadgroups; every block keeps its global index and its own
      // sequential token walk, so the per-block arithmetic does not depend on
      // S at all.  Here one SIMDGROUP owns one (head, block) unit -- 24 x 128
      // = 3072 units over the grid's simdgroups -- which is the S = 128 case.
      else if (op == 11u) {
        const uint TOT   = actl[0];
        const int  lpad  = int(actl[5]);
        const uint BS    = actl[6];
        const float qscale = as_type<float>(actl[8]);
        const uint elements = HD / 32u;
        const uint units = NQH * ABLK;
        // `a0` is the attention layer's slot; all twelve KV ledgers share one
        // binding pair, NKVH * TOT * HD elements apart.
        const size_t kvbase = (size_t)a0 * NKVH * (size_t)TOT * HD;

        // ------------------------------------------ per-query causal edge
        // Everything that bounds the walk is the QUERY's, not the slab's.
        // Query m sits at p + m, so its `q_pos`, its selected-block count,
        // its `n_sel` boundary and its `complete` -- the position at which
        // the incomplete tail begins -- all advance mid-slab.  Sharing any
        // of them silently drops attention: the newest positions of the
        // later queries fall outside `complete` and outside the scored
        // blocks at once, which is exactly the 2026-09-03 defect.
        uint mU[MAXMW], mcount[MAXMW], mnsel[MAXMW];
        int  mqp[MAXMW], mcomplete[MAXMW];
        uint token_width = 0u;
        for (uint m = 0; m < MW; ++m) {
          const device uint* MC = actl + ACTLM0 + m * ACTLMS;
          mU[m] = MC[0]; mcount[m] = MC[1]; mnsel[m] = MC[11];
          mqp[m] = int(MC[2]);
          mcomplete[m] = int(MC[10]);
          token_width = metal::max(token_width, mU[m] * BS);
        }

        for (uint unit = grow; unit < units; unit += nrow) {
          const uint qh = unit / ABLK;
          const uint block_idx = unit - qh * ABLK;
          const uint hkv = qh / AGQA;
          // M query vectors and M online-softmax states in registers.  At
          // HD = 256 that is 8 floats of q and 8 of accumulator per query
          // plus two scalars -- 18 registers a query, 54 at M = 3.
          float q_values[MAXMW * (HD / 32u)];
          float out_values[MAXMW * (HD / 32u)];
          float maximum[MAXMW], total[MAXMW];
          for (uint m = 0; m < MW; ++m) {
            const device float* scm = scratch + (size_t)m * SCSTRIDE;
            for (uint part = 0; part < elements; ++part) {
              q_values[m * elements + part] =
                  qscale * scm[src + qh * HD + lane * elements + part];
              out_values[m * elements + part] = 0.0f;
            }
            maximum[m] = -3.402823466e+38F;
            total[m] = 0.0f;
          }

          for (uint token = block_idx; token < token_width; token += ABLK) {
            const uint slot = token / BS;
            const uint tail = token - slot * BS;
            // The K/V row this slot names, per query.  -1 is "this query
            // does not attend this slot".
            int phys[MAXMW];
            for (uint m = 0; m < MW; ++m) {
              phys[m] = -1;
              if (slot >= mcount[m]) continue;
              const device float* scm = scratch + (size_t)m * SCSTRIDE;
              const device uint* sids = sel_ids + m * (BTK + 1u);
              const int block = int(ids_from_scratch
                  ? reinterpret_cast<const device uint*>(scm + SC_IDX_SEL)[slot]
                  : sids[slot]);
              const int logical = block * int(BS) + int(tail);
              const int physical = lpad + logical;
              if (physical < 0 || physical >= int(TOT) || logical > mqp[m])
                continue;
              if (!(slot < mnsel[m] || logical >= mcomplete[m])) continue;
              phys[m] = physical;
            }
            // ONE K/V read serves every query that names the same row.  The
            // queries of a slab differ by at most k positions, so their
            // top-512 selections agree at most slots and this is the whole
            // amortisation attention gets; when they disagree the load
            // simply happens again, and the arithmetic each query sees is
            // the same sequence in the same order either way.
            int loaded = -1;
            float kvals[HD / 32u], vvals[HD / 32u];
            for (uint m = 0; m < MW; ++m) {
              if (phys[m] < 0) continue;
              if (phys[m] != loaded) {
                const device BFT* krow = kbuf + kvbase
                    + ((size_t)hkv * TOT + (size_t)phys[m]) * HD;
                const device BFT* vrow = vbuf + kvbase
                    + ((size_t)hkv * TOT + (size_t)phys[m]) * HD;
                for (uint part = 0; part < elements; ++part) {
                  kvals[part] = float(krow[lane * elements + part]);
                  vvals[part] = float(vrow[lane * elements + part]);
                }
                loaded = phys[m];
              }
              float score = 0.0f;
              for (uint part = 0; part < elements; ++part)
                score += q_values[m * elements + part] * kvals[part];
              score = simd_sum(score);
              const float new_max = metal::max(maximum[m], score);
              const float factor = metal::fast::exp(maximum[m] - new_max);
              const float probability = metal::fast::exp(score - new_max);
              maximum[m] = new_max;
              total[m] = total[m] * factor + probability;
              for (uint part = 0; part < elements; ++part)
                out_values[m * elements + part] =
                    out_values[m * elements + part] * factor
                    + probability * vvals[part];
            }
          }

          const uint state = qh * ABLK + block_idx;
          for (uint m = 0; m < MW; ++m) {
            const size_t sbase = (size_t)m * 2u * NQH * ABLK;
            if (lane == 0u) {
              apm[sbase + state] = maximum[m];
              apm[sbase + NQH * ABLK + state] = total[m];
            }
            for (uint part = 0; part < elements; ++part)
              apo[((size_t)m * NQH * ABLK + state) * HD
                  + lane * elements + part] =
                  static_cast<BFT>(out_values[m * elements + part]);
          }
        }
      }

      // ---------------------------------------------- OP_ATTN_COMBINE (18)
      // Pass 2, arithmetic-for-arithmetic with `_COMBINE_SOURCE`.  That
      // kernel uses THIRTY-TWO simdgroups and a 32x32 threadgroup transpose;
      // this one has sixteen, so each takes two of the 32 roles.  The set
      // summed by each final `simd_sum` and the order inside every role's
      // accumulation are unchanged, so the result is unchanged -- the role
      // count is a schedule, not an arithmetic choice.
      else if (op == 18u) {
        const uint elements = HD / 32u;
        threadgroup float* tr = A + TG_TGX;      // 1024 floats, aliased
        const uint ROLES = 32u;
        const uint per = ROLES / NSGA;           // roles per simdgroup
        // Pass 1 wrote M planes of partials.  The combine has no weight to
        // share, so it is a per-query phase: `mq` selects the plane and `sc`
        // is already rebased onto the same query's scratch.
        const size_t apm_base = (size_t)mq * 2u * NQH * ABLK;
        const size_t apo_base = (size_t)mq * NQH * ABLK;
        for (uint qh = tg; qh < NQH; qh += ntg) {
          const uint state = qh * ABLK;
          const device float* row_m = apm + apm_base + state;
          const device float* row_l = apm + apm_base + NQH * ABLK + state;
          const device BFT* row_o = apo + (apo_base + state) * HD;
          float maximum = -3.402823466e+38F;
          for (uint g = 0; g < ABLK / 32u; ++g)
            maximum = metal::max(maximum, row_m[lane + 32u * g]);
          maximum = simd_max(maximum);
          float total = 0.0f;
          for (uint g = 0; g < ABLK / 32u; ++g) {
            const uint block = lane + 32u * g;
            total += metal::fast::exp(row_m[block] - maximum) * row_l[block];
          }
          total = simd_sum(total);
          float values[(HD / 32u) * (32u / NSGA)];
          for (uint r = 0; r < per; ++r) {
            const uint role = sg + r * NSGA;
            for (uint part = 0; part < elements; ++part)
              values[r * elements + part] = 0.0f;
            for (uint g = 0; g < ABLK / 32u; ++g) {
              const uint block = role + 32u * g;
              const float factor =
                  metal::fast::exp(row_m[block] - maximum);
              for (uint part = 0; part < elements; ++part)
                values[r * elements + part] += factor * float(
                    row_o[(size_t)block * HD + lane * elements + part]);
            }
          }
          for (uint part = 0; part < elements; ++part) {
            for (uint r = 0; r < per; ++r)
              tr[lane * 32u + sg + r * NSGA] = values[r * elements + part];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < per; ++r) {
              const uint role = sg + r * NSGA;
              float v = simd_sum(tr[role * 32u + lane]);
              v = total == 0.0f ? v : v / total;
              if (lane == 0u)
                sc[dst + qh * HD + role * elements + part] =
                    float(static_cast<BFT>(v));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
        }
      }


      // ---------------------------------------------- OP_INDEX_SCORE (19)
      // The indexer's block score: sum over its heads of relu(q_h . pooled_n),
      // divided by sqrt(head_dim), and -inf for a block the causal geometry
      // rejects.  That shape is why the selector's tie rule matters -- a block
      // no head likes is exactly 0.0 and an invalid one is exactly -inf, so
      // ties are common rather than exotic.
      //
      // The pooled-key ledger is a host binding, not device work: one new
      // block closes every `block_size` tokens, so pooling it is incremental
      // host state in the same class as the PLE n-gram gather.
      else if (op == 19u) {
        // Both are context-length dependent, so they arrive per token in the
        // control block rather than being baked into the schedule.
        // Per-query: query m has `(p + m + 1) // 4` closed blocks, so the
        // count advances mid-slab and the scores land in query m's plane.
        const uint nblocks = mc[4], nvalid = mc[5];
        const uint heads = IDXH, hd = IDXD;
        const float inv = INVIDXD;
        // `a0` is the attention layer's slot: the pooled ledgers of all
        // twelve live in one binding, `actl[15]` rows apart.
        const size_t pbase = (size_t)a0 * actl[15] * hd;
        device float* score_row = score_tiles + (size_t)mq * score_stride;
        for (uint n = grow; n < nblocks; n += nrow) {
          if (n >= nvalid) {
            if (lane == 0u) score_row[n] = -INFINITY;
            continue;
          }
          const device BFT* prow = pooled + pbase + (size_t)n * hd;
          float acc = 0.0f;
          for (uint h = 0; h < heads; ++h) {
            float d = 0.0f;
            for (uint i = lane; i < hd; i += 32u)
              d += sc[src + h * hd + i] * float(prow[i]);
            d = simd_sum(d);
            acc += metal::max(d, 0.0f);
          }
          if (lane == 0u) score_row[n] = acc * inv;
        }
      }

      // ----------------------------------------------- OP_INDEX_TOPB (12)
      // Top-BLOCK_TOPK over the block scores, in ONE threadgroup: four 8-bit
      // radix passes over a monotone float key narrow to the exact k-th key,
      // then one order-free emit places each element at
      // `gt_before + min(eq_before, need)`.  Every threadgroup runs the whole
      // selection for itself and writes the same answer. The score plane is
      // launch-sized and tile-aligned; the selector's global histogram and
      // final contiguous emit preserve the pre-tiling result exactly.
      else if (op == 12u) {
        threadgroup atomic_uint* hist =
            reinterpret_cast<threadgroup atomic_uint*>(A + TG_TGX);
        threadgroup uint* gtc =
            reinterpret_cast<threadgroup uint*>(A + TG_TGX) + 256u;
        threadgroup uint* eqc = gtc + NT;
        threadgroup uint* shared = eqc + NT;
        const device float* score_row =
            score_tiles + (size_t)mq * score_stride;
        select_top_blocks(score_row, mc[4], a0,
                          reinterpret_cast<device uint*>(sc + dst),
                          hist, gtc, eqc, shared, tid, NT);
        // The INCOMPLETE TAIL block, appended at slot `n_sel` exactly as
        // `compact_blocks_to_kernel_inputs` does for the shipped kernel.  The
        // indexer scores only closed blocks, so without this slot a query
        // whose length is not a multiple of the block size cannot attend its
        // own newest positions -- itself included.  `count > n_sel` is the
        // host's statement that the slot is live; OP_ATTN then admits it only
        // through `logical >= complete`, never by membership, so a block that
        // is both selected and the tail is not counted twice.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // THE TAIL BLOCK, per query.  `count > n_sel` says this query has
        // an incomplete tail, and the block it names is `(p + m) // BS`,
        // which ADVANCES mid-slab: for a slab at p = 3 the three queries
        // want blocks 0, 1, 1.  One shared tail would leave two of the
        // three queries unable to attend their own newest positions --
        // the 2026-09-03 defect, three times over.
        if (mc[1] > mc[11] && tid == 0u) {
          reinterpret_cast<device uint*>(sc + dst)[mc[11]] = mc[9];
        }
      }

      // -------------------------------------------- OP_QK_NORM_ROPE (21)
      // Per-head RMSNorm then partial RoPE, one SIMDGROUP per head.  Both
      // boundaries round to bfloat16, because both stock ops return the
      // activation dtype -- and the attention phase's bit-identity with the
      // shipped indexed kernel holds only if it reads exactly float(bf16 q).
      //
      // `a2` is the SOURCE stride per head: q_proj emits [q | gate] per head
      // (2 * head_dim) while k_proj and the indexer's q are packed.  The
      // destination is always head-major and packed.
      //
      // The rope pair is (i, i + half) with half = ROTD/2 = 32 = the SIMD
      // width, so lane `l` wrote both halves of its own pair in the norm pass
      // above.  The simdgroup barrier makes that ordering explicit rather
      // than relying on it.
      else if (op == 21u) {
        const device uint* T = tbl + ent * TSTRIDE;
        const device BFT* g = reinterpret_cast<const device BFT*>(
            WB[T[TBL_GROUP]] + T[TBL_WOFF]);
        const uint heads = a0, hd = a1;
        const uint sstride = (a2 == 0u) ? hd : a2;
        const uint hrot = ROTD / 2u;
        // Per-query: the RoPE angle is THIS query's position, and the
        // three queries of a verify slab sit at p, p+1, p+2.
        const float pos = float(mc[6]);
        const float logtheta = metal::precise::log(as_type<float>(actl[13]));
        for (uint h = grow; h < heads; h += nrow) {
          float ss = 0.0f;
          for (uint i = lane; i < hd; i += 32u) {
            const float v = sc[src + h * sstride + i];
            ss += v * v;
          }
          ss = simd_sum(ss);
          const float inv = metal::precise::rsqrt(ss / float(hd) + NEPS);
          for (uint i = lane; i < hd; i += 32u) {
            const float v = sc[src + h * sstride + i] * inv * float(g[i]);
            sc[dst + h * hd + i] = float(static_cast<BFT>(v));
          }
          simdgroup_barrier(mem_flags::mem_device);
          for (uint i = lane; i < hrot; i += 32u) {
            const float freq = metal::precise::exp(
                -logtheta * (2.0f * float(i)) / float(ROTD));
            const float ang = pos * freq;
            const float c = metal::precise::cos(ang);
            const float sn = metal::precise::sin(ang);
            const float l = sc[dst + h * hd + i];
            const float r = sc[dst + h * hd + hrot + i];
            sc[dst + h * hd + i] = float(static_cast<BFT>(l * c - r * sn));
            sc[dst + h * hd + hrot + i] =
                float(static_cast<BFT>(r * c + l * sn));
          }
        }
      }

      // ------------------------------------------------ OP_KV_APPEND (22)
      // `cache.update_and_fetch` and `cache.update_index_keys`, in place.
      // `src` is k, `a0` is v, `a1` is the raw index key, `a2` is the
      // attention layer's slot -- the ONE thing about attention that varies
      // with depth, so it travels in the schedule and the control block stays
      // one array for the whole token.
      else if (op == 22u) {
        const uint TOT = actl[0], li = a2;
        device BFT* kw = const_cast<device BFT*>(kbuf);
        device BFT* vw = const_cast<device BFT*>(vbuf);
        device BFT* rw = const_cast<device BFT*>(rawk);
        const size_t kb = (size_t)li * NKVH * (size_t)TOT * HD;
        // M columns, one per query, each at its OWN physical slot p + m.
        // INTRA-SLAB CAUSALITY LIVES HERE: this phase precedes the indexer
        // and the attention phases inside the layer, so by the time query
        // m + 1 scores and attends, query m's column is already in the
        // ledger and the tail block it names contains it.  The ordering was
        // already right for M = 1; what makes it right for a slab is that
        // ALL M columns land before ANY query attends.
        for (uint m = 0; m < MW; ++m) {
          const device float* scm = scratch + (size_t)m * SCSTRIDE;
          const uint slot = (actl + ACTLM0 + m * ACTLMS)[7];
          for (uint i = tg * NT + tid; i < NKVH * HD; i += ntg * NT) {
            const uint h = i / HD, d = i - h * HD;
            const size_t off = kb + ((size_t)h * TOT + slot) * HD + d;
            kw[off] = static_cast<BFT>(scm[src + i]);
            vw[off] = static_cast<BFT>(scm[a0 + i]);
          }
          const size_t rb =
              (size_t)li * (size_t)TOT * IDXD + (size_t)slot * IDXD;
          for (uint i = tg * NT + tid; i < IDXD; i += ntg * NT)
            rw[rb + i] = static_cast<BFT>(scm[a1 + i]);
        }
      }

      // ------------------------------------------------ OP_POOL_BLOCK (26)
      // The ONE block a decode token can close, pooled exactly as
      // `QSAIndexer._pool_blocks` does it: mean over the block's four raw
      // keys with the fp32 upcast stock takes, back to bf16, k_layernorm,
      // then RoPE at the block's START position.  `actl[16]` is 0 on the
      // three tokens in four that close nothing, and the phase is then a
      // no-op that still costs its barrier.
      else if (op == 26u) {
        // Over a slab, blocks close at most once per BS positions, so at
        // M = 3 and BS = 4 AT MOST ONE query closes a block -- the phase
        // becomes a count and a start rather than M pooling passes.  The
        // loop is written for the general case and costs nothing when M < BS
        // because the predicate is false for every other query.
        for (uint m = 0; m < MW; ++m) {
        const device uint* PC = actl + ACTLM0 + m * ACTLMS;
        if (PC[8] != 0u) {
          const uint TOT = actl[0], BSZ = actl[6], li = a2;
          const uint nb = PC[4];                 // blocks after this query
          const uint blk = nb - 1u;
          const uint start = blk * BSZ;
          const device uint* T = tbl + ent * TSTRIDE;
          const device BFT* g = reinterpret_cast<const device BFT*>(
              WB[T[TBL_GROUP]] + T[TBL_WOFF]);
          device BFT* pw = const_cast<device BFT*>(pooled);
          const size_t rb = (size_t)li * (size_t)TOT * IDXD;
          const size_t pb = ((size_t)li * actl[15] + blk) * IDXD;
          // One simdgroup does the whole 128-wide block; every threadgroup
          // computes the same answer and writes it, which is the same
          // replicate-rather-than-barrier trade OP_INDEX_TOPB makes.
          if (sg == 0u) {
            float ss = 0.0f;
            float mine[IDXD / 32];
            for (uint j = 0; j < IDXD / 32u; ++j) {
              const uint i = lane + j * 32u;
              float acc = 0.0f;
              for (uint t = 0; t < BSZ; ++t)
                acc += float(rawk[rb + (size_t)(start + t) * IDXD + i]);
              const float m = float(static_cast<BFT>(acc / float(BSZ)));
              mine[j] = m;
              ss += m * m;
            }
            ss = simd_sum(ss);
            const float inv = metal::precise::rsqrt(ss / float(IDXD) + NEPS);
            const uint hrot = ROTD / 2u;
            const float logtheta =
                metal::precise::log(as_type<float>(actl[13]));
            float normed[IDXD / 32];
            for (uint j = 0; j < IDXD / 32u; ++j) {
              const uint i = lane + j * 32u;
              normed[j] = float(static_cast<BFT>(
                  mine[j] * inv * float(g[i])));
            }
            // Lane `l` owns rope indices l and l + half of the FIRST 64 dims,
            // which are j = 0 (i = l, l < 32 == hrot) and j = 1 (i = l + 32).
            const float freq = metal::precise::exp(
                -logtheta * (2.0f * float(lane)) / float(ROTD));
            const float ang = float(start) * freq;
            const float c = metal::precise::cos(ang);
            const float sn = metal::precise::sin(ang);
            const float l0 = normed[0], r0 = normed[1];
            normed[0] = float(static_cast<BFT>(l0 * c - r0 * sn));
            normed[1] = float(static_cast<BFT>(r0 * c + l0 * sn));
            for (uint j = 0; j < IDXD / 32u; ++j)
              pw[pb + lane + j * 32u] = static_cast<BFT>(normed[j]);
          }
        }
        }   // per-query pooling
      }

      // -------------------------------------------------- OP_GATE_MUL (23)
      // `out * mx.sigmoid(gate)`.  The gate is the second half of q_proj's
      // doubled per-head output, so it is strided by `a2` while the attention
      // output is packed.
      else if (op == 23u) {
        const uint heads = a0, hd = a1, sstride = a2;
        for (uint i = tg * NT + tid; i < heads * hd; i += ntg * NT) {
          const uint h = i / hd, d = i - h * hd;
          sc[dst + i] = sc[dst + i]
              * sigmoid_f(sc[src + h * sstride + hd + d]);
        }
      }

      // -------------------------------------------------- OP_PLE_GATE (24)
      // `_chain_gate` plus the sigmoid and the gate product.  `src` is the
      // normed key (HCN x HID), `a0` the normed query (HCN x HID), `a1` the
      // value (HID); `dst` takes the gated result (HCN x HID).
      //
      // The sigmoid is `metal::precise::exp`, matching the STANDALONE
      // `mx.sigmoid` primitive.  That is the whole reason this is a phase of
      // its own: the engine's compiled PLE chain is exact only because the
      // sigmoid sits OUTSIDE the traced span, where a fused rewrite cannot
      // reach it.
      else if (op == 24u) {
        for (uint h = grow; h < HCN; h += nrow) {
          float acc = 0.0f;
          for (uint i = lane; i < HID; i += 32u)
            acc += sc[src + h * HID + i] * sc[a0 + h * HID + i];
          acc = simd_sum(acc);
          float gate = acc * PLEQS;
          gate = (gate < 0.0f ? -1.0f : (gate > 0.0f ? 1.0f : 0.0f))
               * metal::precise::sqrt(metal::max(metal::abs(gate), 1e-6f));
          gate = sigmoid_f(gate);
          for (uint i = lane; i < HID; i += 32u)
            sc[dst + h * HID + i] = gate * sc[a1 + i];
        }
      }

      // -------------------------------------------------- OP_PLE_CONV (25)
      // The dilated depthwise conv, its silu, the gated residual, and the
      // conv-state roll.  `src` is the conv-normed vector, `a0` the gated
      // vector, `dst` the hyper-connection residual the result adds into.
      //
      // At M = 1 the conv window is the 9 state rows plus the new one and the
      // taps are 3 apart, so the output reads rows 0, 3, 6 of the state and
      // the new row -- and the new state is rows 1..9 of that window, i.e. the
      // old state shifted by one with the new row appended.  The shift is
      // deliberately done AFTER the whole output is read, behind a
      // threadgroup barrier per element rather than a copy: every channel is
      // independent, so one thread owning a channel does read-then-write with
      // no cross-thread hazard at all.
      else if (op == 25u) {
        const device uint* T = tbl + ent * TSTRIDE;
        const device BFT* w = reinterpret_cast<const device BFT*>(
            WB[T[TBL_GROUP]] + T[TBL_WOFF]);
        device BFT* st = pconv_out;
        // The dilated window rolls M steps.  One thread owns a channel for
        // the whole slab, so the queries simply run in order inside it: no
        // cross-thread hazard, and the state is read and written once even
        // though it advances M times.
        for (uint c = tg * NT + tid; c < HCH; c += ntg * NT) {
          for (uint m = 0; m < MW; ++m) {
            device float* scm = scratch + (size_t)m * SCSTRIDE;
            const float xnew = scm[src + c];
            float acc = 0.0f;
            for (uint kk = 0; kk + 1u < PLEK; ++kk)
              acc += float(w[c * PLEK + kk])
                   * float(st[(size_t)(kk * PLEN) * HCH + c]);
            acc += float(w[c * PLEK + (PLEK - 1u)]) * xnew;
            scm[dst + c] = scm[dst + c] + scm[a0 + c] + silu_f(acc);
            // roll: state row r <- row r+1, last row <- the new vector
            for (uint r = 0; r + 1u < PLES; ++r)
              st[(size_t)r * HCH + c] = st[(size_t)(r + 1u) * HCH + c];
            st[(size_t)(PLES - 1u) * HCH + c] = static_cast<BFT>(xnew);
          }
        }
      }

      // ------------------------------------------------ OP_ADD_BCAST (20)
      // ``dst[h * width + d] += src[d]`` over the H streams.  The MTP head's
      // fuse adds one embedding vector into all four of them.
      else if (op == 20u) {
        const uint width = a0, count = a1;
        for (uint i = tg * NT + tid; i < count * width; i += ntg * NT)
          sc[dst + i] = sc[dst + i] + sc[src + (i % width)];
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

      // The passes write DIFFERENT scratch planes, so device memory needs no
      // barrier between them -- but they SHARE the threadgroup arena (`tgx`,
      // `tlog`, `red`, `part`, `gsc`), so pass m+1 would overwrite staging
      // that pass m's stragglers are still reading.  One threadgroup barrier
      // per pass, and only when there is more than one pass.
      if (mloops > 1u) threadgroup_barrier(mem_flags::mem_threadgroup);
      }   // per-query loop

      if (bar == 2u) {
        live = gbar(ctr, ab, ntg, tid, phase, SPINCAP);
      } else if (bar == 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    if (!live) break;
    for (uint i = tg * NT + tid; i < HID; i += ntg * NT)
      for (uint m = 0; m < MW; ++m)
        out[((size_t)rep * MW + m) * HID + i] =
            static_cast<BFT>(scratch[(size_t)m * SCSTRIDE + SC_MIXED + i]);
    live = gbar(ctr, ab, ntg, tid, phase, SPINCAP);
  }

  if (tg == 0u && tid == 0u) {
    status[0] = atomic_load_explicit(ab, memory_order_relaxed);
    status[1] = phase;
  }
"""

# Ten weight buffers, and the barrier's generation counter folded into `meta`.
# Both are BINDING arithmetic, not preference.  MLX binds inputs and outputs
# alike, so the budget is Metal's 31 for the pair: ten outputs leave 21, and
# the 63 GiB of experts needs eight groups on its own because a group is
# uint32-indexed and so caps at 2^31 words = 8 GiB.  Ten buffers + main +
# the remaining tables and state fit in 20 inputs.
# Phase E bought back THREE binding slots, each a pure addressing change with
# no new arithmetic, because width 3 needs slots and the kernel was AT the
# ceiling (23 inputs + 8 outputs = 31 of Metal's 31, not the 30 the build log
# recorded):
#   * ``kbuf`` + ``vbuf`` -> ``kv``.  Same dtype, same shape, one constant
#     offset apart; the attention phase already computed ``kvbase``.
#   * ``pooled`` + ``rawk`` -> ``idxl``, the one index ledger.  Same argument.
#   * ``meta`` -> folded into ``actl``.  A four-word uint32 control buffer was
#     worth as much of the budget as an 8 GiB weight group.
# 20 inputs + 10 outputs = 30 of 31. The tenth output is the dynamically
# tile-aligned score plane; moving it out of activation scratch removes the
# fixed context ceiling while keeping one binding in reserve. Transactional
# PLE state remains separate from its input so verify rollback is real.
IN_NAMES = [
    "xin", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9",
    "tbl", "sched", "cs_in", "rec_in",
    # The KV ledger (keys then values) and the index ledger (raw keys then
    # pooled block keys).  Both are LEDGERS the kernel appends to in place,
    # like ``ctrl``, not values it returns: the decode token that writes them
    # is the same launch that reads them back two phases later, so an output
    # would put the host in the middle of the dispatch.
    "kv", "idxl", "pconv",
    "actl", "ctrl",
]
OUT_NAMES = ["scratch", "score_tiles", "out", "cs_out", "rec_out",
             "pconv_out", "apm", "apo", "logits", "status"]

MAX_GROUPS = 10

_KERNEL_CACHE: dict[Any, Any] = {}


def build_body_kernel(*, threads: int = None, rdown: int = 4,
                      rgu: int = 2, rdn: int = 2, spin_cap: int = None,
                      max_query_width: int = None):
    """Compile the schedule-walking kernel.  One binary for every schedule.

    ``max_query_width`` is the REGISTER-FOOTPRINT ceiling this one binary is
    built for -- it controls ``MAXMW`` (register-array sizing, the
    threadgroup arena's ``TOPW`` block, and the ``#if MAXMW == 1`` compile-out
    in the body source) and nothing else.  It is deliberately independent of
    ``ACTLIDS``/``ACTLM0``/``ACTLMS``, which stay pinned to the PROCESS
    ceiling (``MAX_QUERY_WIDTH``) regardless -- those three constants are the
    ``actl`` control buffer's layout, shared with the weight pack and the
    scratch offset table (``SCRATCH``, width-invariant already) as the ONE
    addressing scheme every build agrees on.  A dual-width process therefore
    calls this twice, at ``max_query_width=1`` and ``max_query_width=3``, and
    gets two compiled pipelines that read the SAME ``actl``/scratch layout but
    carry different register/threadgroup footprints -- not two processes with
    two incompatible layouts.
    """
    threads = _THREADS if threads is None else threads
    spin_cap = _SPIN_CAP if spin_cap is None else spin_cap
    mqw = MAX_QUERY_WIDTH if max_query_width is None else int(max_query_width)
    nsg = threads // 32
    if threads % 32 or nsg > MAX_SIMDGROUPS:
        raise ValueError(f"threads={threads} must be a multiple of 32, <= 512")
    if GDN_VALUE_DIM % nsg:
        raise ValueError(
            f"GDN core splits {GDN_VALUE_DIM} value dims over {nsg} simdgroups"
        )
    # The M dispatch chains are written out for widths 1, 2 and the built
    # maximum.  A maximum above 3 would leave width 3 falling into the
    # MAXMW branch, instantiating a body that reads a source plane the slab
    # does not have -- silently.  Fail closed rather than generate that.
    if mqw > 3 or mqw < 1:
        raise ValueError(
            f"max_query_width={mqw}: the kernel's per-width dispatch chains "
            "cover 1, 2 and the maximum, so a build outside 1..3 needs the "
            "chains generated, not extended by hand"
        )
    if mqw > MAX_QUERY_WIDTH:
        raise ValueError(
            f"max_query_width={mqw} exceeds the process ceiling "
            f"MAX_QUERY_WIDTH={MAX_QUERY_WIDTH} that ACTLIDS/scratch are "
            "sized against -- raise MLX_QWEN4_MEGAKERNEL_MAX_WIDTH instead "
            "of building past it"
        )
    key = (threads, rdown, rgu, rdn, spin_cap, mqw)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    tg, tg_floats, tg_bytes = compute_tg_layout(mqw)
    need = 256 + 2 * threads + 2
    if need > tg["TLOG"]:
        raise ValueError(
            f"top-block selector needs {need} threadgroup words but only "
            f"{tg['TLOG']} are aliasable over TGX"
        )
    if tg_bytes > 16 * 1024:
        raise ValueError(
            f"threadgroup arena {tg_bytes} B over the 16 KiB cap at "
            f"max_query_width={mqw}"
        )

    subs = {
        "NT": threads, "NSGC": nsg, "SPINCAP": spin_cap,
        "HID": HIDDEN, "HCN": HC_COUNT, "HCH": HC_HIDDEN,
        "CD": CONV_DIM, "CK": CONV_KERNEL,
        "HV": GDN_VALUE_HEADS, "HK": GDN_KEY_HEADS, "RATIO": GDN_RATIO,
        "DK": GDN_KEY_DIM, "DV": GDN_VALUE_DIM,
        "TOPKN": TOPK, "TSTRIDE": TABLE_STRIDE,
        "HD": HEAD_DIM, "NQH": N_Q_HEADS, "NKVH": N_KV_HEADS,
        "AGQA": N_Q_HEADS // N_KV_HEADS, "ABLK": SDPA_BLOCKS, "NSGA": nsg,
        "IDXH": IDX_HEADS, "IDXD": IDX_HEAD_DIM,
        "INVIDXD": f"{IDX_HEAD_DIM ** -0.5:.17g}f",
        "ROTD": ROTARY_DIM, "ACTLH": ACTL_HEADER,
        # Pinned to the PROCESS ceiling -- see the docstring.  NOT ``mqw``.
        "ACTLIDS": ACTL_IDS, "ACTLM0": ACTL_M0, "ACTLMS": ACTL_M_STRIDE,
        "BTK": BLOCK_TOPK,
        "SCTILE": SCORE_TILE_BLOCKS,
        "SCSTRIDE": SCRATCH_STRIDE, "MAXMW": mqw,
        "PERQMASK": "29159800u",
        "PLEK": PLE_CONV_KERNEL, "PLEN": PLE_NGRAM, "PLES": PLE_STATE_LEN,
        "PLEQS": f"{HIDDEN ** -0.5:.17g}f",
        "RDOWN": rdown, "RGU": rgu, "RDN": rdn, "RMAXN": RMAX,
        "TGF": tg_floats,
        "QSCALE": f"{GDN_KEY_DIM ** -0.5:.17g}f",
        "NEPS": f"{RMS_EPS:.17g}f",
        "BFT": "bfloat16_t",
    }
    for name, off in tg.items():
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
        name=f"qwen4_mega_t{threads}_d{rdown}_g{rgu}_n{rdn}_w{mqw}",
        input_names=IN_NAMES, output_names=OUT_NAMES,
        source=src, header=hdr,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


class MegakernelBody:
    """Launches the persistent kernel over a packed model and a schedule."""

    def __init__(self, pack, schedule, *, gdn_layers: int, vocab: int = 1,
                 threads: int = None, groups: int = None,
                 max_query_width: int = None, **kw):
        self.max_query_width = (MAX_QUERY_WIDTH if max_query_width is None
                                else int(max_query_width))
        tg, _, tg_bytes = compute_tg_layout(self.max_query_width)
        need = (256 + 2 * (_THREADS if threads is None else threads) + 2)
        if need > tg["TLOG"]:
            raise ValueError(
                f"top-block selector needs {need} threadgroup words but only "
                f"{tg['TLOG']} are aliasable over TGX"
            )
        if tg_bytes > 16 * 1024:
            raise ValueError(
                f"threadgroup arena {tg_bytes} B over the 16 KiB cap at "
                f"max_query_width={self.max_query_width}"
            )
        if len(pack.buffers) > MAX_GROUPS:
            raise ValueError(
                f"{len(pack.buffers)} packed groups over the {MAX_GROUPS} "
                "weight bindings this kernel declares"
            )
        self.pack = pack
        self.schedule = schedule
        self.gdn_layers = max(int(gdn_layers), 1)
        self.vocab = max(int(vocab), 1)
        self.threads = _THREADS if threads is None else threads
        self.groups = _THREADGROUPS if groups is None else groups
        self.spin_cap = (_SPIN_CAP if kw.get("spin_cap") is None
                         else int(kw["spin_cap"]))
        self.kernel = build_body_kernel(
            threads=self.threads, max_query_width=self.max_query_width, **kw)
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
                 steps: Optional[int] = None, kbuf=None, vbuf=None,
                 pooled=None, actl=None, total: int = 1, rawk=None,
                 pconv=None, kv=None, idxl=None, mwidth: int = 1,
                 score_blocks: Optional[int] = None):
        """``steps`` truncates the schedule, for cumulative phase profiling.

        The barrier map lives in the schedule, so a prefix is a real, running
        kernel with exactly the barriers its own phases need -- not a kernel
        with the tail compiled out.

        ``mwidth`` is the QUERY width of the slab: 1 for a draft token, k+1
        for a verify slab.  The scratch is allocated as that many planes and
        ``actl[23]`` carries it into the kernel.

        The ledgers arrive merged (``kv``, ``idxl``).  The separate
        ``kbuf``/``vbuf``/``rawk``/``pooled`` kwargs are still accepted, for
        the standalone phase probes that build tiny ones, and are concatenated
        here; the decode path allocates them merged and never pays that copy.
        """
        nsteps = len(self.schedule) if steps is None else int(steps)
        width = int(xin.shape[-1])
        mwidth = max(int(mwidth), 1)
        if mwidth > self.max_query_width:
            raise ValueError(
                f"query width {mwidth} over the built maximum "
                f"{self.max_query_width}"
            )
        if kv is None:
            if kbuf is None:
                kbuf = mx.zeros((N_KV_HEADS, 1, HEAD_DIM), mx.bfloat16)
                vbuf = kbuf
            kv = mx.concatenate(
                [kbuf.reshape(-1), vbuf.reshape(-1)]).reshape(-1, HEAD_DIM)
        if idxl is None:
            if rawk is None:
                rawk = mx.zeros((1, IDX_HEAD_DIM), mx.bfloat16)
            if pooled is None:
                pooled = mx.zeros((1, IDX_HEAD_DIM), mx.bfloat16)
            idxl = mx.concatenate(
                [rawk.reshape(-1, IDX_HEAD_DIM),
                 pooled.reshape(-1, IDX_HEAD_DIM)])
        if actl is None:
            actl = mx.zeros((ACTL_IDS + BLOCK_TOPK + 1,), mx.uint32)
        if pconv is None:
            pconv = mx.zeros((PLE_STATE_LEN, HC_HIDDEN), mx.bfloat16)
        # `meta` used to be its own binding.  Its four words now ride in the
        # `actl` header, which means they are per-LAUNCH host state written
        # into an array the caller owns -- so patch a copy, never the caller's.
        score_blocks = (
            max(int(total) // IDX_COMPRESS, 0)
            if score_blocks is None else int(score_blocks)
        )
        score_floats = score_tile_floats(score_blocks, mwidth)
        score_stride = score_floats // mwidth
        actl = mx.concatenate([
            actl[:ACTL["nsteps"]],
            mx.array(
                [nsteps, reps, width, self.phase, mwidth, score_stride],
                mx.uint32,
            ),
            actl[ACTL["score_stride"] + 1:],
        ])
        outs = self.kernel(
            inputs=[xin, *self.wbufs, self.table, self.sched,
                    cs_in, rec_in, kv, idxl, pconv, actl,
                    self.ctrl],
            grid=(self.groups * self.threads, 1, 1),
            threadgroup=(self.threads, 1, 1),
            output_shapes=[
                (scratch_floats(mwidth),), (score_floats,),
                (reps * mwidth, HIDDEN),
                (self.gdn_layers, CONV_KERNEL - 1, CONV_DIM),
                (self.gdn_layers, GDN_VALUE_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM),
                (PLE_STATE_LEN, HC_HIDDEN),
                (mwidth * 2 * N_Q_HEADS * SDPA_BLOCKS,),
                (mwidth * N_Q_HEADS * SDPA_BLOCKS * HEAD_DIM,),
                (reps * mwidth, self.vocab),
                (4,),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.bfloat16, mx.bfloat16,
                           mx.float32, mx.bfloat16, mx.float32, mx.bfloat16,
                           mx.bfloat16, mx.uint32],
        )
        # One grid barrier for the residual load, one after the output write,
        # plus the schedule's own device barriers, per repetition.
        bars = sum(1 for st in self.schedule.steps[:nsteps] if st.barrier == 2)
        self.phase += reps * (bars + 2)
        return outs


class DualWidthMegakernelBody:
    """Two compiled pipelines, one pack, one call surface.

    Phase F's knob (``MLX_QWEN4_MEGAKERNEL_MAX_WIDTH``) picks ONE
    ``MAX_QUERY_WIDTH`` at import and the process is stuck with it -- a
    draft-narrow/verify-wide k=2 round needs both in the SAME process, one
    dispatched per call by the caller's actual query width.  This class is
    that dispatch.

    **What is genuinely shared, not duplicated.**  Both ``MegakernelBody``
    instances are constructed from the SAME ``pack``/``schedule`` objects, so
    ``list(pack.buffers)`` in each instance's ``__init__`` copies the PYTHON
    list, never the underlying ~70 GiB of GPU buffers -- ``narrow.wbufs[i] is
    wide.wbufs[i]`` for every packed group, asserted below rather than
    assumed.  ``ACTLIDS``/``ACTLM0``/``ACTLMS`` and the ``SCRATCH`` offset
    table are pinned to the process ceiling for both builds (see
    ``build_body_kernel``'s docstring), so the ``actl``/scratch layout is the
    ONE addressing scheme both pipelines read -- a narrow call never differs
    from a wide call in anything but which planes it touches.  The scratch
    buffer itself is not a hand-managed persistent allocation: it is an MLX
    kernel OUTPUT, freshly sized to ``scratch_floats(mwidth)`` every launch,
    exactly as a single-width build already does -- dual-width adds no new
    scratch duplication risk, because there was never a persistent scratch
    buffer to duplicate.  What dual-width DOES duplicate, deliberately, is
    the threadgroup arena and the register file: two different compiled
    pipelines, which is the entire point.

    **Lazy wide wrapper.** The narrow Python wrapper is built in ``__init__``;
    the wide wrapper is prepared explicitly before verification. Metal library
    and pipeline compilation are deferred until the first evaluated launch.

    **One barrier generation, shared.**  ``ctrl``'s atomic counters and the
    ``phase`` value that seeds them are a running sequence across a decode
    session; a draft dispatch and the verify dispatch that follows it must
    agree on that sequence regardless of which pipeline ran.  This class owns
    ``ctrl``/``phase`` itself and stamps them onto whichever sub-body is about
    to launch, then reads the advanced ``phase`` back -- the two
    ``MegakernelBody`` instances never disagree about how many barriers have
    already happened.
    """

    def __init__(self, pack, schedule, *, gdn_layers: int, vocab: int = 1,
                 threads: int = None, groups: int = None,
                 narrow_width: int = 1, wide_width: int = None, **kw):
        self.pack = pack
        self.schedule = schedule
        self.wide_width = MAX_QUERY_WIDTH if wide_width is None else int(wide_width)
        self.narrow_width = int(narrow_width)
        if self.narrow_width >= self.wide_width:
            raise ValueError(
                f"narrow_width={self.narrow_width} must be < "
                f"wide_width={self.wide_width}"
            )
        self._kw = dict(gdn_layers=gdn_layers, vocab=vocab, threads=threads,
                        groups=groups, **kw)
        self.narrow = MegakernelBody(
            pack, schedule, max_query_width=self.narrow_width, **self._kw)
        self._wide: Optional[MegakernelBody] = None
        self.threads = self.narrow.threads
        self.groups = self.narrow.groups
        self.spin_cap = self.narrow.spin_cap
        self.wide_wrapper_prepare_seconds: Optional[float] = None
        self.calls_narrow = 0
        self.calls_wide = 0
        self.reset()

    def reset(self) -> None:
        self.ctrl = mx.zeros((64,), mx.uint32)
        mx.eval(self.ctrl)
        self.phase = 0
        self.narrow.ctrl = self.ctrl
        self.narrow.phase = 0
        if self._wide is not None:
            self._wide.ctrl = self.ctrl
            self._wide.phase = 0

    def _ensure_wide(self) -> MegakernelBody:
        if self._wide is None:
            t0 = time.perf_counter()
            wide = MegakernelBody(
                self.pack, self.schedule, max_query_width=self.wide_width,
                **self._kw)
            mx.eval(wide.wbufs, wide.table, wide.sched)
            self.wide_wrapper_prepare_seconds = time.perf_counter() - t0
            # Weight-pack sharing is a claim about object identity, not a
            # hope -- assert it rather than infer it from a passing gate.
            for a, b in zip(self.narrow.wbufs, wide.wbufs):
                if a is not b and not (a.size <= 16 and b.size <= 16):
                    raise AssertionError(
                        "wide body's weight buffers are not the SAME arrays "
                        "as the narrow body's -- the pack was copied"
                    )
            assert self.narrow.table is wide.table, "offset table copied"
            assert self.narrow.sched is wide.sched, "schedule buffer copied"
            wide.ctrl = self.ctrl
            wide.phase = self.phase
            self._wide = wide
        return self._wide

    def is_width_wrapper_prepared(self, width: int) -> bool:
        """Whether the Python wrapper exists; not a Metal compilation check."""
        width = int(width)
        if width < 1 or width > self.wide_width:
            return False
        return width <= self.narrow_width or self._wide is not None

    def prepare_width_wrapper(self, width: int) -> dict:
        """Prepare the selected wrapper without launching the decode kernel.

        The first evaluated output still compiles the Metal library/pipeline.
        A successful receipt does not qualify that cold launch or its latency.
        """
        width = int(width)
        if width < 1 or width > self.wide_width:
            raise ValueError(
                f"query width {width} outside 1..{self.wide_width}"
            )
        was_built = self._wide is not None
        if width > self.narrow_width:
            self._ensure_wide()
        return {
            **self.receipt(),
            "requested_width": width,
            "wrapper_prepared": self.is_width_wrapper_prepared(width),
            "wrapper_built_now": width > self.narrow_width and not was_built,
            "launched": False,
        }

    def prepare_width(self, width: int) -> dict:
        """Compatibility alias; this prepares a wrapper, not a Metal pipeline."""
        return self.prepare_width_wrapper(width)

    def __call__(self, *args, mwidth: int = 1, **kwargs):
        mwidth = max(int(mwidth), 1)
        if mwidth <= self.narrow_width:
            body, self.calls_narrow = self.narrow, self.calls_narrow + 1
            variant = "narrow"
        else:
            body = self._ensure_wide()
            self.calls_wide += 1
            variant = "wide"
        body.ctrl = self.ctrl
        body.phase = self.phase
        outs = body(*args, mwidth=mwidth, **kwargs)
        self.phase = body.phase
        self.last_variant = variant
        return outs

    def receipt(self) -> dict:
        return {
            "narrow_width": self.narrow_width,
            "wide_width": self.wide_width,
            "wide_wrapper_prepared": self._wide is not None,
            "wide_wrapper_prepare_seconds": self.wide_wrapper_prepare_seconds,
            "pipeline_compilation": "deferred_to_first_evaluation",
            "calls_narrow": self.calls_narrow,
            "calls_wide": self.calls_wide,
            "last_variant": getattr(self, "last_variant", None),
            "threads": self.threads,
            "threadgroups": self.groups,
            "spin_cap": self.spin_cap,
        }
