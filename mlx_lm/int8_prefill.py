"""Selective W8A8 int8 prefill on Apple M5 neural accelerators (NAX).

Ported from the lab's mlx-vlm tree (mlx_vlm/int8_prefill.py, branch
codex/int8-prefill-overlay) with MLX_VLM_INT8_* env vars renamed to
MLX_LM_INT8_*.

For prefill-sized calls on selected large language-model projections (MLP by
default, or attention/linear-attention projections with the opt-in "all"
scope),
replaces the 4-bit quantized matmul with an int8 x int8 -> int32 GEMM running
on the M5 GPU neural accelerators via Metal Performance Primitives tensor ops:

  - activations: per-token (per-row) dynamic symmetric int8, custom kernel
  - weights: per-output-channel symmetric int8, produced by a fused Metal
    kernel straight from the resident packed 4-bit weights (no bf16
    intermediate). By default (MLX_LM_INT8_CACHE=none) each layer's int8
    tensor is built per prefill call and freed by the MLX executor right
    after its GEMM consumes it, so peak extra memory is a few hundred MB
    (one layer), not the ~24 GB of a full copy. MLX_LM_INT8_CACHE=ttl
    instead caches all copies and evicts them after MLX_LM_INT8_TTL_S idle.
    Only the per-channel scales (~10 MB total) are kept across calls; the
    reaper evicts them after the same idle TTL and drops entries whose
    module has been freed.
  - accumulation int32, scales applied in-register, bf16 output

Decode-sized calls (rows < ROW_THRESHOLD) keep the 4-bit quantized kernels,
so decode speed and numerics are completely unchanged. The default MLP scope
also leaves attention, lm_head, embeddings and the vision tower untouched.

Measured on M5 Max (research/int8-nax/): int8 GEMM ~91 TOPS-eq vs ~58 TF for
MLX's bf16 NAX GEMM at the MLP shapes; fused (quant + GEMM) 1.49x over bf16
and 1.66x over 4-bit qmm at M=2048.

Requires an M5-class GPU (Metal 4 tensor ops). Usage: call apply() any time
before serving traffic (weight init is lazy, or call warmup(model) to
pre-build). The server applies it at startup with --int8-prefill
(MLX_LM_INT8_PREFILL=1).
"""

import logging
import os
import threading
import time
import weakref
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Only calls with at least this many rows (tokens) take the int8 path; below
# it the 4-bit qmm kernels win (they are ~2x faster than bf16 at decode
# sizes). This is also what keeps generation on the current kernels: decode
# calls have 1..O(draft block) rows, far below the threshold.
ROW_THRESHOLD = 512

# Scope of layers routed to W8A8 (env MLX_LM_INT8_SCOPE):
#   "all": every large projection — MLP plus
#       attention/linear-attention (q/k/v/o, in_proj_qkv/z, out_proj).
#   "mlp": only the MLP projections (gate/up 5120->17408, down 17408->5120),
#       the more conservative choice if a quality eval flags "all".
# Either way lm_head is excluded (N > MAX_OUT) and tiny projections such as
# linear_attn.in_proj_a/b (N=48) fail the N % 128 tile requirement.
# Default to the validated Qwen3.6 MLP shapes. ``all`` is a broader research
# mode: the process-wide QuantizedLinear hook cannot distinguish a language
# projection from a same-shaped vision projection.
SCOPE = os.environ.get("MLX_LM_INT8_SCOPE", "mlp")
MLP_SHAPES = {(17408, 5120), (5120, 17408)}
MAX_OUT = 32768
MIN_DIM = 1024

# int8 weight-copy lifecycle (env MLX_LM_INT8_CACHE):
#   "none" (default): build each layer's int8 tensor per prefill call with
#       the fused requant kernel and let the MLX executor free it after its
#       GEMM. Peak extra memory ~ one layer; the rebuild is cheap (packed
#       4-bit read + int8 write, no bf16 intermediate), and with a large
#       --prefill-step-size it costs a few percent of chunk compute.
#   "ttl": cache all copies (~24 GB at scope=all) and evict after TTL_S
#       seconds without an int8-path call; fastest, highest peak memory.
CACHE = os.environ.get("MLX_LM_INT8_CACHE", "none")
TTL_S = float(os.environ.get("MLX_LM_INT8_TTL_S", "120"))
ACT_TTL_S = float(os.environ.get("MLX_LM_INT8_ACT_TTL_S", "5"))

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""

# One threadgroup (256 threads) per row: absmax reduce, then quantize.
_QUANT_SRC = """
    constexpr int K = {K};
    constexpr int NTH = 256;

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg = tid / 32;

    const device {T}* xrow = x + size_t(row) * K;

    float amax = 0.0f;
    for (int i = tid; i < K; i += NTH) {{
        amax = max(amax, fabs(float(xrow[i])));
    }}
    amax = simd_max(amax);

    threadgroup float tg_max[NTH / 32];
    if (lane == 0) tg_max[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = tg_max[lane % (NTH / 32)];
    amax = simd_max(amax);

    float scale = max(amax, 1e-8f) / 127.0f;
    float inv = 1.0f / scale;
    if (tid == 0) xs[row] = scale;

    device int8_t* qrow = xq + size_t(row) * K;
    for (int i = tid; i < K; i += NTH) {{
        qrow[i] = int8_t(clamp(rint(float(xrow[i]) * inv), -127.0f, 127.0f));
    }}
"""

# Threadgroup computes a 128x128 output tile with 8 simdgroups; matmul2d
# loops over K internally (dynamic_extent). Edge tiles are bounds-checked by
# the tensor extents and the epilogue guard, so any M works.
_GEMM_SRC = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int TM = 128;
    constexpr int TN = 128;

    uint2 tgid = threadgroup_position_in_grid.xy;
    const int M = m_dim[0];

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/true,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<8>> op;

    // Row-major X[M,K] -> extents (K, M); row-major W[N,K] used as the
    // transposed right operand -> extents (K, N).
    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)xq, dextents<int32_t, 2>(K, M));
    auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)wq, dextents<int32_t, 2>(K, N));

    auto tA = A.slice(0, int(tgid.y) * TM);
    auto tB = B.slice(0, int(tgid.x) * TN);

    auto cT = op.get_destination_cooperative_tensor<
        decltype(tA), decltype(tB), int32_t>();

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) cT[i] = 0;
    }}

    op.run(tA, tB, cT);

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int n = int(tgid.x) * TN + idx[0];
            int m = int(tgid.y) * TM + idx[1];
            if (m < M && n < N) {{
                float v = float(cT[i]) * xs[m] * ws[n];
                {BIAS_LINE}
                out[size_t(m) * N + n] = bfloat(v);
            }}
        }}
    }}
"""

# Fused requantization: packed affine-4bit (group_size 64) -> per-channel
# symmetric int8, one pass, no bf16 intermediate. One threadgroup (256
# threads) per output channel; each uint32 word holds 8 nibbles, and a
# 64-value group spans exactly 8 words, so a word never crosses groups.
_REQUANT_SRC = """
    constexpr int KW = {KW};   // packed words per row (K / 8)

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

    const device uint32_t* prow = packed + size_t(row) * KW;
    const device {T}* srow = scales + size_t(row) * (KW / 8);
    const device {T}* brow = biases + size_t(row) * (KW / 8);
    device int8_t* orow = out + size_t(row) * KW * 8;

    float inv = 1.0f / ws[row];

    for (int i = tid; i < KW; i += 256) {{
        uint32_t wrd = prow[i];
        int g = i / 8;
        float s = float(srow[g]);
        float b = float(brow[g]);
        device int8_t* o = orow + i * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {{
            float v = float((wrd >> (4 * j)) & 0xF) * s + b;
            o[j] = int8_t(clamp(rint(v * inv), -127.0f, 127.0f));
        }}
    }}
"""

_TM, _TN, _NSIMD = 128, 128, 8
_quant_kernels = {}
_gemm_kernels = {}
_requant_kernels = {}
# id(module) -> (weakref.ref(module), wq int8 [N,K], ws fp32 [N]). id() alone
# is not a safe key: after a model swap CPython can hand a new module a freed
# module's id(), so entries are only served when the stored referent is the
# calling module, and the reaper drops entries whose module has been freed
# (plus the TTL_S idle eviction).
_int8_weights = {}
_weights_lock = threading.Lock()
_last_use = 0.0
_reaper_started = False
_reaper_stop = None
_original_quantized_linear_call = None


def _touch():
    global _last_use
    _last_use = time.monotonic()


def _reap_once():
    with _weights_lock:
        idle = time.monotonic() - _last_use
        # Entries whose module has been freed go every pass, idle or not: a
        # later model's module can reuse the dead module's id(), and a stale
        # entry must never outlive its module.
        for cache in (_int8_weights, _ws_cache):
            for key, entry in list(cache.items()):
                if entry[0]() is None:
                    cache.pop(key, None)
        if idle > TTL_S and (_int8_weights or _ws_cache):
            if _int8_weights:
                logger.info(
                    "int8 NAX prefill: evicted %d int8 weight copies after "
                    "%.0fs idle",
                    len(_int8_weights),
                    TTL_S,
                )
            _int8_weights.clear()
            _ws_cache.clear()
            mx.clear_cache()
        if _act_cache and idle > ACT_TTL_S:
            _act_cache.clear()
            mx.clear_cache()


def _reaper(stop):
    interval = max(min(TTL_S, ACT_TTL_S) / 4.0, 1.0)
    while not stop.wait(interval):
        _reap_once()


def _start_reaper():
    global _reaper_started, _reaper_stop
    if (TTL_S > 0 or ACT_TTL_S > 0) and not _reaper_started:
        _reaper_started = True
        _reaper_stop = threading.Event()
        threading.Thread(
            target=_reaper,
            args=(_reaper_stop,),
            name="int8-prefill-reaper",
            daemon=True,
        ).start()


def _quantize_rows(x):
    M, K = x.shape
    tname = {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}[
        x.dtype
    ]
    key = (K, tname)
    if key not in _quant_kernels:
        _quant_kernels[key] = mx.fast.metal_kernel(
            name=f"i8p_rowquant_{K}_{tname}",
            input_names=["x"],
            output_names=["xq", "xs"],
            header=_HEADER,
            source=_QUANT_SRC.format(K=K, T=tname),
        )
    return _quant_kernels[key](
        inputs=[x],
        grid=(M * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(M, K), (M,)],
        output_dtypes=[mx.int8, mx.float32],
    )


def _int8_gemm(xq, xs, wq, ws, bias=None):
    M, K = xq.shape
    N = wq.shape[0]
    key = (N, K, bias is not None)
    if key not in _gemm_kernels:
        names = ["xq", "wq", "xs", "ws", "m_dim"]
        bias_line = ""
        if bias is not None:
            names.append("bias")
            bias_line = "v += float(bias[n]);"
        _gemm_kernels[key] = mx.fast.metal_kernel(
            name=f"i8p_gemm_{N}x{K}{'_b' if bias is not None else ''}",
            input_names=names,
            output_names=["out"],
            header=_HEADER,
            source=_GEMM_SRC.format(N=N, K=K, BIAS_LINE=bias_line),
        )
    inputs = [xq, wq, xs, ws, mx.array([M], dtype=mx.int32)]
    if bias is not None:
        inputs.append(bias)
    return _gemm_kernels[key](
        inputs=inputs,
        grid=(N // _TN * 32 * _NSIMD, (M + _TM - 1) // _TM, 1),
        threadgroup=(32 * _NSIMD, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


# id(module) -> (weakref.ref(module), ws fp32 [N]); tiny (~10 MB total).
# Entries are validated against the live module on lookup (id() reuse after a
# model swap must not resurrect the previous model's scales) and evicted by
# the reaper (dead modules every pass, everything after TTL_S idle) and by
# remove().
_ws_cache = {}


def _ws_for(m: nn.Module):
    """Per-channel int8 scale: an upper bound on |w| per output channel,
    computed from the affine group scales/biases alone (no dequantization).
    Within a group max|w| <= max(|bias|, |15*scale + bias|); slightly coarser
    than the exact absmax, but safe and essentially free to compute."""
    entry = _ws_cache.get(id(m))
    if entry is not None and entry[0]() is m:
        return entry[1]
    s = m["scales"].astype(mx.float32)
    b = m["biases"].astype(mx.float32)
    bound = mx.maximum(mx.abs(b), mx.abs(15.0 * s + b))
    ws = mx.maximum(bound.max(axis=1), 1e-8) / 127.0
    mx.eval(ws)
    _ws_cache[id(m)] = (weakref.ref(m), ws)
    return ws


def _requant(m: nn.Module, ws):
    """int8 [N, K] weights from the resident packed 4-bit tensor, fused."""
    w = m["weight"]  # uint32 [N, K/8]
    N, KW = w.shape
    tname = {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}[
        m["scales"].dtype
    ]
    key = (KW, tname)
    if key not in _requant_kernels:
        _requant_kernels[key] = mx.fast.metal_kernel(
            name=f"i8p_requant_{KW}_{tname}",
            input_names=["packed", "scales", "biases", "ws"],
            output_names=["out"],
            header=_HEADER,
            source=_REQUANT_SRC.format(KW=KW, T=tname),
        )
    return _requant_kernels[key](
        inputs=[w, m["scales"], m["biases"], ws],
        grid=(N * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(N, KW * 8)],
        output_dtypes=[mx.int8],
    )[0]


def _weights_for(m: nn.Module):
    """(wq int8 [N,K], ws fp32 [N]) for a QuantizedLinear.

    CACHE="none": wq is rebuilt per call by the fused kernel and freed by
    the MLX executor once its GEMM has consumed it (peak ~ one layer).
    CACHE="ttl": wq is cached and evicted after TTL_S idle.
    """
    ws = _ws_for(m)
    if CACHE != "ttl":
        return _requant(m, ws), ws
    _touch()
    with _weights_lock:
        entry = _int8_weights.get(id(m))
        if entry is None or entry[0]() is not m:
            wq = _requant(m, ws)
            mx.eval(wq)
            entry = (weakref.ref(m), wq, ws)
            _int8_weights[id(m)] = entry
    return entry[1], entry[2]


def _eligible(m: nn.Module, k_dim: int) -> bool:
    n = m["weight"].shape[0]
    if n % _TN or k_dim % 32:
        return False
    # The fused requant kernel assumes the standard mlx-community layout.
    if (
        m.bits != 4
        or m.group_size != 64
        or getattr(m, "mode", "affine") != "affine"
        or "biases" not in m
    ):
        return False
    if SCOPE == "mlp":
        return (n, k_dim) in MLP_SHAPES
    return n <= MAX_OUT and min(n, k_dim) >= MIN_DIM


# q/k/v (and gate/up) are called with the *same* activation tensor; quantize
# it once and reuse. Entries hold a strong reference to the input, so the
# id() key stays valid for the entry's lifetime.
_act_cache = OrderedDict()
# One entry is enough to reuse a shared activation across q/k/v or gate/up,
# while bounding the strong-reference tail to one prefill tensor.
_ACT_CACHE_SIZE = 1


def _quantize_rows_cached(x, k_dim):
    key = id(x)
    entry = _act_cache.get(key)
    if entry is not None and entry[0] is x:
        _act_cache.move_to_end(key)
        return entry[1], entry[2]
    xq, xs = _quantize_rows(x.reshape(-1, k_dim))
    _act_cache[key] = (x, xq, xs)
    if len(_act_cache) > _ACT_CACHE_SIZE:
        _act_cache.popitem(last=False)
    return xq, xs


def apply():
    """Patch nn.QuantizedLinear to route eligible prefill calls to W8A8."""
    global _original_quantized_linear_call
    if _original_quantized_linear_call is not None:
        return False
    device_name = str(mx.device_info().get("device_name", ""))
    if "M5" not in device_name:
        raise RuntimeError(
            "int8 NAX prefill requires an Apple M5-class GPU; "
            f"found {device_name or 'unknown device'}"
        )
    _original_quantized_linear_call = nn.QuantizedLinear.__call__

    def ql_call(self, x):
        k_dim = x.shape[-1]
        rows = x.size // k_dim
        # The custom epilogue writes bfloat16. Fall back rather than silently
        # changing fp16/fp32 model semantics.
        if (
            x.dtype != mx.bfloat16
            or rows < ROW_THRESHOLD
            or not _eligible(self, k_dim)
        ):
            return _original_quantized_linear_call(self, x)
        _touch()
        wq, ws = _weights_for(self)
        xq, xs = _quantize_rows_cached(x, k_dim)
        bias = self["bias"] if "bias" in self else None
        y = _int8_gemm(xq, xs, wq, ws, bias=bias)
        return y.reshape(*x.shape[:-1], wq.shape[0])

    nn.QuantizedLinear.__call__ = ql_call
    _start_reaper()
    logger.info(
        "int8 NAX prefill patch applied (row threshold %d, scope %s, "
        "weight cache %s%s)",
        ROW_THRESHOLD,
        SCOPE,
        CACHE,
        f", TTL {TTL_S:.0f}s" if CACHE == "ttl" else "",
    )
    return True


def remove():
    """Restore the original QuantizedLinear call, release overlay caches and
    stop the reaper thread (a later apply() starts a fresh one)."""
    global _original_quantized_linear_call, _reaper_started, _reaper_stop
    if _original_quantized_linear_call is None:
        return False
    nn.QuantizedLinear.__call__ = _original_quantized_linear_call
    _original_quantized_linear_call = None
    if _reaper_stop is not None:
        _reaper_stop.set()
        _reaper_stop = None
    _reaper_started = False
    with _weights_lock:
        _int8_weights.clear()
        _ws_cache.clear()
        _act_cache.clear()
    mx.clear_cache()
    return True


def release(reapply: bool = True):
    """Model-unload boundary hook for the server.

    Runs remove() to restore nn.QuantizedLinear.__call__ and flush every
    id()-keyed overlay cache — after a swap, a new model's module can reuse a
    freed module's id() and must never inherit the previous model's scales or
    int8 weights. Unless ``reapply`` is False (process shutdown), the patch is
    then reinstalled with empty caches so the next model keeps int8 prefill.
    A no-op returning False when the overlay was never applied.
    """
    removed = remove()
    if removed and reapply:
        apply()
    return removed


def warmup(model: nn.Module):
    """Pre-build per-channel scales (and, with CACHE=ttl, the int8 weights)
    for all eligible modules (optional)."""
    count = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            n, kp = m["weight"].shape
            k = kp * 32 // m.bits
            if _eligible(m, k):
                if CACHE == "ttl":
                    _weights_for(m)
                else:
                    _ws_for(m)
                count += 1
    logger.info("int8 NAX prefill: %d modules prepared (cache=%s)", count, CACHE)
