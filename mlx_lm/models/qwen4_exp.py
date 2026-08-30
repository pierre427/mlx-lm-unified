# Copyright © 2026 Apple Inc.
#
# Qwen4-Exp / Qwen3.8-Flash-Next text-core support.  The architecture was
# derived from the Apache-2.0 Transformers Qwen4Exp implementation and the
# release checkpoint at Qwen/Qwen3.8-Flash-Next.

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask, create_ssm_mask, scaled_dot_product_attention
from .cache import ArraysCache, BatchKVCache, KVCache, SinkWindowKVCache, dynamic_roll
from .pipeline import PipelineMixin
from .qwen4_qsa_nax import (
    block_sparse_layout_supported,
    compact_blocks_to_kernel_inputs,
    nax_kernel_available,
    nax_qsa_attention,
)
from .qwen3_5 import GatedDeltaNet as Qwen35GatedDeltaNet
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock
from . import qwen3_next
from .qwen3_next import (
    _concat_tables,
    _env_flag,
    _proj_identity,
    _proj_signature,
    _proj_table,
    check_materialization_budget,
    table_bytes,
    transform_moe_weights,
)
from .rope_utils import initialize_rope


# Opt-in micro-levers, each read once at import.  Off keeps the stock path,
# EXCEPT where a lever has been promoted (``default=True``) -- see below.
_RMSNORM_FAST = _env_flag("MLX_QWEN4_RMSNORM_FAST")
_QSA_POOLED_KEY_CACHE = _env_flag("MLX_QWEN4_QSA_POOLED_KEY_CACHE")

# PROMOTED to default-on 2026-08-28; set MLX_QWEN4_QSA_SCATTER_CHOSEN=0 to
# revert (the server also exposes it live as ``qwen4_qsa_scatter_chosen``).
#
# ``dense_mask()`` below builds the chosen-block indicator either by scatter
# or by a broadcast equality reduction.  The two are BIT-IDENTICAL -- proven
# by ``argpartition`` returning no duplicate indices, and measured equal by
# ``mx.array_equal`` over decode, prefill, ragged and left-padded-batch
# shapes -- so this changes cost only, never a number.
#
# The broadcast form materializes a [B, L, K, n_blocks] boolean intermediate.
# That is quadratic in context (K and n_blocks both grow), and it dominates:
#
#   isolated mask build, L=2048, M5 Max, mlx 0.32.2 (per QSA layer-call).
#   The mask is [B, 1, L, T] and broadcasts over heads, so these are
#   head-independent: measured 89.9 ms at 16 query heads and 89.85 ms at the
#   production 24, from separate runs.
#     KV  8192   broadcast 20.7 ms   scatter 0.52 ms
#     KV 16384   broadcast 42.3 ms   scatter 1.14 ms
#     KV 32768   broadcast 89.9 ms   scatter 2.22 ms
#   At 32K that build is 1.6x the 24-head SDPA it feeds (55.37 ms).
#
#   peak allocation, L=512 (production prefill chunk)
#     KV 16384   broadcast 1.12 GB   scatter 0.05 GB
#     KV 32768   broadcast 2.24 GB   scatter 0.09 GB
#     KV 65536   broadcast 4.48 GB   scatter 0.19 GB
#
# The memory ratio is the reason this is a default and not a tuning knob: at
# 24x, the broadcast form is a long-context OOM hazard on a host already
# holding a large resident model, and it bought nothing.
_QSA_SCATTER_CHOSEN = _env_flag("MLX_QWEN4_QSA_SCATTER_CHOSEN", default=True)
_PLE_VECTOR_SHIFT = _env_flag("MLX_QWEN4_PLE_VECTOR_SHIFT")
_PLE_GATHER_CONCAT = _env_flag("MLX_QWEN4_PLE_GATHER_CONCAT")
# Diagnostic only: make the Qwen4 hyper-connection mixer and each token's four
# GDN input projections use the same M=1 kernel family as ordinary decode. A
# short speculative slab normally projects B*S rows together, and MLX may
# choose a different quantized-matmul kernel whose bf16 rounding changes the
# cached convolution input. Keeping this opt-in lets the transactional-state
# oracle walk that width dependence upstream without changing production.
_GDN_SHAPE_STABLE_PROJECTIONS = _env_flag(
    "MLX_QWEN4_GDN_SHAPE_STABLE_PROJECTIONS"
)
_SHAPE_STABLE_SHORT_FORWARD = _env_flag(
    "MLX_QWEN4_SHAPE_STABLE_SHORT_FORWARD"
)

# MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT (2026-08-27): skip the whole indexer
# selection while the QSA mask is dense BY CONSTRUCTION, i.e. while
# ``n_blocks <= block_topk`` (block_topk = indexer_budget // compress_ratio =
# 512, so total <= indexer_budget + compress_ratio - 1 = 2051 cached tokens).
# Rapid-MLX ships the same guard; llama.cpp documents the property as a test
# oracle.  Gate class: BITWISE.
#
# Proof that ``causal_mask & sparse == causal_mask`` in that regime.  Write
# r = compress_ratio, and let p be a query's position and t a key position.
#  (0) k = min(block_topk, n_blocks) = n_blocks, and ``argpartition`` returns
#      distinct indices, so ALL n_blocks blocks are selected.  The -inf scores
#      of the invalid blocks change nothing, so chosen == valid_blocks and the
#      pooled keys, the scores and the top-k never influence the result.
#  (1) valid_blocks[n] = (n*r + r - 1 <= p): block n is picked iff it closed
#      at or before p.  Blocks are complete by construction, so this is the
#      only causality term the block half carries.
#  (2) tail covers complete = ((p+1)//r)*r <= t <= p.
#  (3) Take any t <= p.  If t >= complete, (2) gives it.  Otherwise t <
#      complete <= n_blocks*r (complete is a multiple of r and <= p+1 <=
#      total), so token_block[t] = t//r is unclamped, and that block ends at
#      (t//r)*r + r - 1 <= complete - 1 <= p, hence valid by (1).
#      So every causal (b, l, t) is set in ``sparse`` and the AND is a no-op.
# The identity is one-directional: ``sparse`` is never NARROWER than causal on
# the valid cells, but it can be WIDER.  ``token_block`` clamps a logical key
# past the last closed block down into that block, so a future t > p can ride
# in on a selected block (r=4, total=10, left_pad=1, p=7, t=8).  That is
# exactly why the short-circuit returns ``causal_mask`` itself and never an
# all-true mask (None included: ``create_attention_mask`` returns None only
# for a 1-token decode on an unpadded cache, where no future column exists at
# all).  For a batch cache, which keeps the left padding out of its mask,
# returning ``causal_mask`` preserves that padding term exactly.
#
# One guard, provable: MTP shared top-k reuses a k-wide index set from an
# earlier step, so it is dense only when that set still covers every block:
# shared.shape[-1] == n_blocks (which already implies n_blocks <= block_topk).
# A block closed mid-cycle is NOT in the shared set and NOT in the tail, so
# the stale mask is genuinely sparser than causal there.
#
# The proof is written in the indexer's LOGICAL coordinates, so it covers a
# left-padded batch row for row.  Physical column j of row b holds logical
# position ``j - left_padding[b]``, the causal mask admits exactly
# ``0 <= t <= p``, and step (3)'s ``complete <= n_blocks*r`` still holds
# because that row's own block count is at most the shared ``n_blocks``.
# (Before 2026-08-27 it did not: ``q_pos`` was logical while ``starts`` and
# ``token_pos`` were physical, which made the stock sparse mask itself wrong
# for a left-padded row -- see QSAIndexer.__call__.)
_QSA_DENSE_SHORTCIRCUIT = _env_flag("MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT")

# MLX_QWEN4_QSA_FUSED_PROJ (2026-08-27 decode-decomposition lever): run every
# same-input QSA-layer projection — q_proj incl. its gate half, k_proj,
# v_proj, and the indexer's index_qk_proj — as ONE wide quantized matmul over
# x, split after.  Honesty note: North measured plain qkv-concat at -3..-4%
# (bit-exact but slower; "MLX already overlaps independent kernels").  The
# reopen rationale is the decomposition finding (GPU window 86% of the decode
# step at 22% bandwidth at M=1): the overlap defense does not hold at this
# occupancy-bound operating point, and this lever settles it empirically.
# Trade documented: MTP shared-top-k steps and sink-window MTP skip the
# separate indexer projection; the fused matmul always includes that slice.
# Gate class: TOLERANCE, not bitwise (demoted 2026-08-27 review).  Tiny-
# scale outputs measured bit-identical, but mlx's qmm dispatch is width-
# dependent (mlx-src/mlx/backend/metal/quantized.cpp:102 thresholds,
# :907 split-K selection): at M=512 the stock 512-wide K/V projections take
# split-K=2 while the fused 13952-wide op takes non-split qmm, and the
# qmv/qmm crossover differs at M=12-15 — regimes a toy-shape test cannot
# see, so production shapes may not be bit-identical.
_QSA_FUSED_PROJ = _env_flag("MLX_QWEN4_QSA_FUSED_PROJ")

# MLX_QWEN4_QSA_NAX_KERNEL (2026-08-28): route the sparse QSA attention through
# the hand-written NAX (MPP matmul2d) block-sparse kernel instead of building
# the dense [B, 1, L, T] selection mask and calling bf16 masked SDPA.  Off by
# default; live-toggleable as ``qwen4_qsa_nax_kernel``.  It engages ONLY on an
# ``explicit`` selection (the sparse path, total > ~2051 cached tokens), a
# multi-token query (prefill/verify, NOT M=1 decode), a supported layout, and a
# device where the kernel compiles.  Every other path -- decode, implicit_all,
# mask_only, unsupported shape, NAX unavailable -- keeps the dense SDPA path
# bit-identical.  Gate class: TOLERANCE, not bitwise: the kernel keeps fp32
# accumulation and rounds only P, so it is ~7x MORE accurate than bf16 SDPA
# (~1.4e-3 rel vs fp32 vs bf16 SDPA's ~1.1e-2).  Near-tie greedy flips vs the
# OFF path are expected and are the kernel being more accurate, not less.
# Kept OFF by default 2026-08-28, cause NOT root-caused. A default-on restart
# served one 200 then the service went down WITHOUT a Python traceback (a kill,
# not a crash) -- likely a transient over-budget while the kernel's first-use
# Metal-library compile ran on top of the 108 GB model + the fused-gate-up
# load-time rebuild. An initial "stream-affinity" diagnosis was WRONG: the
# cited "no Stream(gpu,0)" errors were stale from an unrelated morning incident;
# a Codex review showed metal_kernel() captures no stream and the served ladder
# did cross the HTTP->worker boundary and passed. So this is memory/lifecycle,
# not threading. Re-test on an isolated server watching memory before default-on.
# Flag still works for isolated benches.
_QSA_NAX_KERNEL = _env_flag("MLX_QWEN4_QSA_NAX_KERNEL")

# Minimum query length for the kernel to engage. It tiles M by query heads, so
# it needs many tokens to amortize its launch: it wins on the 512-wide prefill
# chunks but LOSES on the small-M speculative-verify shapes at decode (self-MTP
# verify is num_draft+1 tokens). Measured 2026-08-28, engaging on the M=3
# verify halved decode (54->16 t/s at 1K). Above this, prefill only; below it,
# every decode-time shape falls back to dense SDPA, bit-identical to OFF.
_QSA_NAX_MIN_QUERY = int(os.environ.get("MLX_QWEN4_QSA_NAX_MIN_QUERY", "64"))


def _table_matmul(table, x: mx.array) -> mx.array:
    weight, scales, biases, group_size, bits, mode = table
    if scales is None:
        return x @ weight.T
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def _valid_span_end(mask):
    """One past each row's last valid position, as a ``[B]`` vector.

    ``ArraysCache.make_mask`` builds either ``pos >= left_padding`` (leading
    pads, from a left-padded batched prefill) or ``pos < lengths`` (trailing
    pads, from a right-padded continuation or a ragged verify), so a row's
    valid positions are always ONE contiguous run and its end is all the PLE
    state updates need.  An all-pad row returns 0 and keeps its prior state.
    """
    length = mask.shape[1]
    if isinstance(mask, np.ndarray):
        return np.max(np.where(mask, np.arange(1, length + 1), 0), axis=1)
    return mx.max(mx.where(mask, mx.arange(1, length + 1), 0), axis=1)


def _row_tail(values, end, width):
    """Per-row trailing window of ``[prev(width), new]`` ending at ``end``.

    Both PLE states are stored as the last ``width`` entries of a
    ``width``-prefixed buffer, so row ``b``'s window is
    ``values[b, end[b] : end[b] + width]``.  With ``end == values.shape[1] -
    width`` this is exactly ``values[:, -width:]``, the unpadded update.
    """
    positions = end[:, None] + (
        np.arange(width) if isinstance(values, np.ndarray) else mx.arange(width)
    )
    if isinstance(values, np.ndarray):
        return np.take_along_axis(values, positions, axis=1)
    if values.ndim == 3:
        positions = mx.broadcast_to(
            positions[..., None], (values.shape[0], width, values.shape[2])
        )
    return mx.take_along_axis(values, positions, axis=1)


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return [
        2
        * (
            _splitmix64(
                (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
            )
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    intermediate_size: int = 0
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    output_gate_type: Optional[str] = "sigmoid"

    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    layer_types: Optional[List[str]] = None

    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    decoder_sparse_step: int = 1
    norm_topk_prob: Optional[bool] = True

    hc_count: int = 4
    hc_lowrank: int = 320
    ple_layer_ids: List[int] = field(default_factory=list)
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: Optional[int] = 1234
    split_ngram_parts: int = 128
    eos_token_id: Union[int, List[int]] = 248044

    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    mtp_num_hidden_layers: int = 1

    rope_parameters: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        }
    )
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.seed = 1234 if self.seed is None else self.seed
        self.norm_topk_prob = (
            True if self.norm_topk_prob is None else self.norm_topk_prob
        )
        self.ple_embed_dim = self.hidden_size if self.ple_embed_dim is None else self.ple_embed_dim
        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if (i + 1) % self.full_attention_interval
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters:
            rope = dict(self.rope_parameters)
            if "type" not in rope and "rope_type" in rope:
                rope["type"] = rope["rope_type"]
            self.partial_rotary_factor = rope.get("partial_rotary_factor", 0.25)
            self.rope_theta = rope.get("rope_theta", 10_000_000.0)
            self.rope_scaling = rope
        self._validate()

    def _validate(self):
        if self.hc_count <= 1:
            raise ValueError("Qwen4-Exp requires more than one hyper-connection stream")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ple_embed_dim % ngram_heads:
            raise ValueError("ple_embed_dim must be divisible by the n-gram head count")
        for layer_id in self.ple_layer_ids:
            if not 1 <= layer_id <= self.num_hidden_layers:
                raise ValueError(f"invalid one-indexed PLE layer id {layer_id}")
            if self.layer_types[layer_id - 1] != "linear_attention":
                raise ValueError("PLE is only defined on linear-attention layers")


class GroupRMSNorm(nn.Module):
    """Zero-centred checkpoint RMSNorm, optionally normalised per H stream."""

    def __init__(self, dim: int, group_size: Optional[int], eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        if _RMSNORM_FAST:
            # The stock path is a pure (not mean-centred) RMS norm, so
            # ``mx.fast.rms_norm`` matches it up to fp32 accumulation order.
            # Per-group weights differ, so the weight is applied outside.
            if self.group_size is not None:
                grouped = x.reshape(*x.shape[:-1], -1, self.group_size)
                out = mx.fast.rms_norm(grouped, None, self.eps).reshape(x.shape)
            else:
                out = mx.fast.rms_norm(x, None, self.eps)
            return (
                out.astype(mx.float32) * self.weight.astype(mx.float32)
            ).astype(dtype)
        xf = x.astype(mx.float32)
        if self.group_size is not None:
            xf = xf.reshape(*xf.shape[:-1], -1, self.group_size)
        out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        if self.group_size is not None:
            out = out.reshape(*x.shape)
        return (out * self.weight.astype(mx.float32)).astype(dtype)


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float, activation: str):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps
        self.activation = activation

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        dtype = hidden_states.dtype
        x = hidden_states.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        x = x * self.weight.astype(mx.float32)
        g = gate.astype(mx.float32)
        g = mx.sigmoid(g) if self.activation == "sigmoid" else nn.silu(g)
        return (x * g).astype(dtype)


class GatedDeltaNet(Qwen35GatedDeltaNet):
    def __init__(self, args: TextModelArgs):
        super().__init__(args)
        self.norm = RMSNormGated(
            args.linear_value_head_dim,
            args.rms_norm_eps,
            args.output_gate_type or args.hidden_act,
        )

    def _input_projections(self, inputs: mx.array):
        if not _GDN_SHAPE_STABLE_PROJECTIONS or inputs.shape[1] <= 1:
            return super()._input_projections(inputs)

        per_token = [
            super(GatedDeltaNet, self)._input_projections(inputs[:, i : i + 1])
            for i in range(inputs.shape[1])
        ]
        return tuple(
            mx.concatenate([token[projection] for token in per_token], axis=1)
            for projection in range(4)
        )


class Qwen4ArraysCache(ArraysCache):
    """Four-state PLE+GDN cache with one atomic speculative rollback."""

    def __new__(cls, *args, **kwargs):
        # __init__ never runs on the from_state path, so anything a method
        # reachable from there touches has to be set here.
        instance = super().__new__(cls, *args, **kwargs)
        instance._ple_rollback = None
        return instance

    def start_speculation(self, rollback_window=None):
        self._ple_rollback = None
        super().start_speculation(rollback_window)

    def stop_speculation(self):
        self._ple_rollback = None
        super().stop_speculation()

    def _clear_staged_rollback(self):
        """Drop a PLE half staged by a forward that never reached GDN.

        ``ArraysCache._invalidate_rollbacks`` calls this on every membership
        change; without it the stale closure would restore old-batch tensors
        into the new membership on the next trim.
        """
        self._ple_rollback = None

    def _refuse_pending_ple(self, who: str):
        """A staged-but-unrecorded PLE half makes the span unrestorable.

        The rewind walks RECORDS, so a forward whose PLE half never reached
        ``record_rollback`` is invisible to it: with older records on the
        stack a trim would silently take its tokens out of those instead of
        this forward, restoring the wrong state. Fail here instead.
        """
        if self._ple_rollback is not None:
            raise RuntimeError(
                f"{who}: a Qwen4 forward staged a PLE rollback that "
                "GatedDeltaNet never recorded, so that span's PLE and GDN "
                "halves cannot be restored together. The two stage on the "
                "same geometry test, so this means the forward was "
                "interrupted between them (qwen4_exp.py, qwen3_5.py)."
            )

    def is_trimmable(self):
        return super().is_trimmable() and self._ple_rollback is None

    def trim(self, n):
        self._refuse_pending_ple("Qwen4ArraysCache.trim")
        return super().trim(n)

    def preflight_ragged_trim(self, n, *, validate: bool = True):
        # trim_ragged() routes through preflight, so this covers both.
        self._refuse_pending_ple("Qwen4ArraysCache.trim_ragged")
        return super().preflight_ragged_trim(n, validate=validate)

    def stage_ple_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None):
        if self._ple_rollback is not None:
            raise RuntimeError(
                "Qwen4 PLE rollback was staged twice without a GDN record in "
                "between. PLE and GDN roll back as ONE record, so a forward "
                "that stages the PLE half must reach GatedDeltaNet's "
                "record_rollback in the same forward."
            )
        self._ple_rollback = (num_tokens, fn, snapshot, per_row_fn)

    def record_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None):
        staged = self._ple_rollback
        self._ple_rollback = None
        if staged is None:
            return super().record_rollback(
                num_tokens, fn, snapshot, per_row_fn=per_row_fn
            )
        ple_tokens, ple_fn, ple_snapshot, ple_per_row = staged
        if ple_tokens != num_tokens:
            raise RuntimeError(
                "Qwen4 PLE/GDN rollback span mismatch: "
                f"{ple_tokens} != {num_tokens}"
            )

        def combined(m):
            return list(fn(m)) + list(ple_fn(m))

        # One record, so the vectorized per-row replay is available only if
        # BOTH halves stage one; a partial form would rewind the pair by
        # different rules.
        rows = None
        if per_row_fn is not None and ple_per_row is not None:

            def rows(lengths):
                return list(per_row_fn(lengths)) + list(ple_per_row(lengths))

        # The per-row depths are ArraysCache's to derive: it reads the same
        # padding metadata this forward staged against, and both halves record
        # before advance(), so there is one implementation, not two.
        return super().record_rollback(
            num_tokens,
            combined,
            list(snapshot) + list(ple_snapshot),
            per_row_fn=rows,
        )

    def extract(self, idx):
        cache = type(self)(len(self.cache))
        cache.cache = [
            None if value is None else mx.contiguous(value[idx : idx + 1])
            for value in self.cache
        ]
        if idx < len(self._checkpoints):
            cache._checkpoints = [list(self._checkpoints[idx])]
        return cache


class GatedResidual(nn.Module):
    def __init__(self, args: TextModelArgs, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = GroupRMSNorm(hc_hidden, args.hidden_size, args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden, self.hc_count, bias=False)

    def __call__(self, hyper_input: mx.array):
        if _GDN_SHAPE_STABLE_PROJECTIONS and hyper_input.shape[1] > 1:
            tokens = [
                self(hyper_input[:, index : index + 1])
                for index in range(hyper_input.shape[1])
            ]
            if isinstance(tokens[0], tuple):
                return tuple(
                    mx.concatenate([token[field] for token in tokens], axis=1)
                    for field in range(len(tokens[0]))
                )
            return mx.concatenate(tokens, axis=1)

        normed = self.hc_norm(hyper_input)
        weights = nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
        weights = mx.sigmoid(self.input_mix_weight_up(weights))
        streams = normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        weights = weights.reshape(*weights.shape[:-1], self.hc_count, self.hidden_size)
        mixed = mx.mean(weights * streams, axis=-2)
        if not hasattr(self, "block_inject_weight"):
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc_count)
        return mixed, hyper_input, inject


class ShardedEmbedding(nn.Module):
    """Row-sharded embedding that never concatenates the 128 PLE tensors.

    MLX safetensor arrays remain file-backed until selected.  Grouping the row
    ids by checkpoint shard therefore transfers only requested rows instead of
    materialising the ~102.4 GB BF16 table.
    """

    def __init__(self, vocab_size: int, dims: int, num_shards: int):
        super().__init__()
        if vocab_size % num_shards:
            raise ValueError("Qwen4-Exp PLE vocabulary must split evenly")
        self.vocab_size = vocab_size
        self.dims = dims
        self.num_shards = num_shards
        self.rows_per_shard = vocab_size // num_shards
        for index in range(num_shards):
            setattr(self, f"shard_{index}", nn.Embedding(self.rows_per_shard, dims))

    def lookup_numpy(self, indices: np.ndarray) -> mx.array:
        """Gather already-hosted row ids without a redundant MLX sync."""
        shape = indices.shape
        flat = np.asarray(indices, dtype=np.int64).reshape(-1)
        shard_ids = flat // self.rows_per_shard
        if _PLE_GATHER_CONCAT:
            # One concatenate plus one take replaces the serial at[].add
            # chain; the inverse permutation restores the request order.
            # Resident-path lever only: in NVMe mode (MLX_QWEN4_PLE_NVME)
            # FileBackedShardedEmbedding replaces this module, gathers a
            # deduplicated selection in one pass, and supersedes this flag.
            pieces = []
            ordering = []
            for shard_index in np.unique(shard_ids):
                positions = np.flatnonzero(shard_ids == shard_index)
                local = flat[positions] - int(shard_index) * self.rows_per_shard
                pieces.append(
                    getattr(self, f"shard_{int(shard_index)}")(
                        mx.array(local, dtype=mx.int64)
                    )
                )
                ordering.append(positions)
            permutation = np.concatenate(ordering)
            inverse = np.empty_like(permutation)
            inverse[permutation] = np.arange(permutation.size)
            output = mx.take(
                mx.concatenate(pieces, axis=0), mx.array(inverse), axis=0
            )
            return output.reshape(*shape, self.dims)
        output = None
        for shard_index in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_index)
            local = flat[positions] - int(shard_index) * self.rows_per_shard
            values = getattr(self, f"shard_{int(shard_index)}")(
                mx.array(local, dtype=mx.int64)
            )
            if output is None:
                output = mx.zeros((flat.size, self.dims), dtype=values.dtype)
            output = output.at[mx.array(positions)].add(values)
        return output.reshape(*shape, self.dims)

    def __call__(self, indices: mx.array) -> mx.array:
        mx.eval(indices)
        return self.lookup_numpy(np.asarray(indices, dtype=np.int64))


class NGramEmbedding(nn.Module):
    def __init__(
        self,
        args: TextModelArgs,
        embedding_dim: int,
        layer_idx: int,
        ple_layer_index: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0] if isinstance(args.eos_token_id, list) else args.eos_token_id
        )
        sizes = [
            _find_nth_prime_after(args.ngram_vocab_size_base - 1, i + 1)
            for i in range(self.ngram_heads)
        ]
        offsets = np.cumsum([0] + sizes[:-1], dtype=np.int64)
        total = sum(sizes)
        divisor = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / divisor) * divisor
        multipliers = _build_layer_multipliers(
            args.vocab_size, args.ngram_size, ple_layer_index, args.seed
        )
        self.layer_multipliers = mx.array(multipliers, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        # Host-side copies of the hash constants save one device round-trip
        # per forward.  Snapshotted lazily and keyed by the source array
        # objects: ``load_weights``/``update`` REPLACE the mx attributes with
        # checkpoint values after construction, and the CPU hash must track
        # them exactly as the Metal path does.
        self._np_constants = None
        self._np_constants_src = None
        self.ngram_embedding = ShardedEmbedding(
            padded, embedding_dim // self.ngram_heads, args.split_ngram_parts
        )
        self.hash_backend = os.getenv("MLX_QWEN4_PLE_HASH_BACKEND", "cpu")
        if self.hash_backend not in {"cpu", "routed_cpu", "metal", "metal_prefill"}:
            raise ValueError(
                "MLX_QWEN4_PLE_HASH_BACKEND must be cpu, routed_cpu, metal, "
                "or metal_prefill"
            )
        self.metal_hash_min_tokens = int(
            os.getenv("MLX_QWEN4_PLE_METAL_MIN_TOKENS", "1024")
        )
        if self.metal_hash_min_tokens < 1:
            raise ValueError("MLX_QWEN4_PLE_METAL_MIN_TOKENS must be positive")
        self._metal_hash_kernel = None

    def _hash_constants_numpy(self):
        """Host copies of the hash constants, tracking the live mx arrays.

        Keyed by object identity (the sources are kept referenced, so an id
        can never be recycled): any ``load_weights``/``update`` swap of the
        underlying arrays invalidates the snapshot on the next call.
        Concurrent prefetch snapshots are benign under the supported
        lifecycle (constants are immutable after load); a concurrent live
        ``load_weights`` would need synchronization - the two cache
        assignments below are not jointly atomic.
        """
        src = (
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
        )
        cached_src = self._np_constants_src
        if cached_src is None or any(
            new is not old for new, old in zip(src, cached_src)
        ):
            self._np_constants = tuple(
                np.asarray(value, dtype=np.int64) for value in src
            )
            self._np_constants_src = src
        return self._np_constants

    def _shift_history_vectorized(self, history: np.ndarray) -> list:
        """Vectorized per-token history shift with per-segment EOS resets."""
        batch, length = history.shape
        positions = np.arange(length, dtype=np.int64)
        eos_at = np.where(history == self.eos_token_id, positions[None, :], -1)
        # Latest EOS at a position strictly before each token; the token at
        # ``pos - shift`` is in the same segment iff it sits after that EOS.
        previous_eos = np.concatenate(
            [
                np.full((batch, 1), -1, dtype=np.int64),
                np.maximum.accumulate(eos_at, axis=1)[:, :-1],
            ],
            axis=1,
        )
        shifted = [history.copy()]
        for shift in range(1, self.ngram_size):
            rolled = np.full_like(history, self.eos_token_id)
            rolled[:, shift:] = history[:, :-shift]
            valid = (positions[None, :] - shift) > previous_eos
            shifted.append(np.where(valid, rolled, self.eos_token_id))
        return shifted

    def _ngram_ids_numpy(
        self,
        input_ids: mx.array,
        cache: Optional[ArraysCache] = None,
        mask: Optional[mx.array] = None,
        previous: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mx.eval(input_ids, mask)
        tokens = np.asarray(input_ids, dtype=np.int64)
        batch, seq_len = tokens.shape
        if previous is not None:
            previous = np.asarray(previous, dtype=np.int64)
            if previous.shape[-1] < self.context_len:
                pad = np.full(
                    (batch, self.context_len - previous.shape[-1]),
                    self.eos_token_id,
                    dtype=np.int64,
                )
                previous = np.concatenate([pad, previous], axis=-1)
        elif cache is not None and cache[3] is not None:
            previous = np.asarray(cache[3], dtype=np.int64)
        else:
            previous = np.full((batch, self.context_len), self.eos_token_id, dtype=np.int64)
        if mask is not None:
            # A pad id is not a token.  EOS is the segment sentinel the shift
            # already resets on, so substituting it makes a padded row hash
            # and store exactly what that row hashes and stores alone.
            tokens = np.where(np.asarray(mask), tokens, self.eos_token_id)
        history = np.concatenate([previous, tokens], axis=-1)
        if cache is not None:
            tail = (
                history[:, -self.context_len :]
                if mask is None
                else _row_tail(
                    history, _valid_span_end(np.asarray(mask)), self.context_len
                )
            )
            cache[3] = mx.array(tail, dtype=mx.int64)

        if _PLE_VECTOR_SHIFT:
            shifted = self._shift_history_vectorized(history)
        else:
            shifted = []
            for shift in range(self.ngram_size):
                out = np.full_like(history, self.eos_token_id)
                if shift == 0:
                    out = history.copy()
                else:
                    for b in range(batch):
                        segment_start = 0
                        for pos in range(history.shape[1]):
                            if pos - segment_start >= shift:
                                out[b, pos] = history[b, pos - shift]
                            if history[b, pos] == self.eos_token_id:
                                segment_start = pos + 1
                shifted.append(out)

        multipliers, sizes, offsets = self._hash_constants_numpy()
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            with np.errstate(over="ignore"):
                mixed = shifted[0] * multipliers[0]
                for position in range(1, ngram):
                    mixed = np.bitwise_xor(
                        mixed, shifted[position] * multipliers[position]
                    )
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            blocks.append(
                np.remainder(mixed[..., None], sizes[start:end]) + offsets[start:end]
            )
        return np.concatenate(blocks, axis=-1)[:, -seq_len:]

    def _ngram_ids_metal(
        self,
        input_ids: mx.array,
        cache: Optional[ArraysCache] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if self.ngram_size != 3 or not mx.metal.is_available():
            return mx.array(
                self._ngram_ids_numpy(input_ids, cache, mask), dtype=mx.int64
            )

        batch, seq_len = input_ids.shape
        if cache is not None and cache[3] is not None:
            previous = cache[3]
        else:
            previous = mx.full(
                (batch, self.context_len), self.eos_token_id, dtype=mx.int64
            )
        tokens = input_ids.astype(mx.int64)
        if mask is not None:
            tokens = mx.where(mask, tokens, self.eos_token_id)
        history = mx.concatenate([previous, tokens], axis=-1)
        if cache is not None:
            cache[3] = mx.contiguous(
                history[:, -self.context_len :]
                if mask is None
                else _row_tail(history, _valid_span_end(mask), self.context_len)
            )

        if self._metal_hash_kernel is None:
            self._metal_hash_kernel = mx.fast.metal_kernel(
                name="qwen4_ple_ngram3_hash",
                input_names=["history", "multipliers", "sizes", "offsets"],
                output_names=["out"],
                source=r"""
                    uint elem = thread_position_in_grid.x;
                    uint head = elem % HEADS;
                    uint token = (elem / HEADS) % SEQ_LEN;
                    uint batch = elem / (HEADS * SEQ_LEN);
                    uint history_pos = token + 2;
                    uint history_base = batch * (SEQ_LEN + 2);

                    long current = history[history_base + history_pos];
                    long previous_1 = history[history_base + history_pos - 1];
                    long previous_2 = history[history_base + history_pos - 2];
                    if (previous_1 == EOS_TOKEN) {
                        previous_2 = EOS_TOKEN;
                    }

                    ulong mixed = ulong(current) * ulong(multipliers[0]);
                    mixed ^= ulong(previous_1) * ulong(multipliers[1]);
                    if (head >= HEADS_PER_NGRAM) {
                        mixed ^= ulong(previous_2) * ulong(multipliers[2]);
                    }
                    long remainder = long(mixed) % sizes[head];
                    if (remainder < 0) {
                        remainder += sizes[head];
                    }
                    out[elem] = remainder + offsets[head];
                """,
            )
        total = batch * seq_len * self.ngram_heads
        return self._metal_hash_kernel(
            inputs=[
                history,
                self.layer_multipliers,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
            ],
            template=[
                ("HEADS", self.ngram_heads),
                ("HEADS_PER_NGRAM", self.heads_per_ngram),
                ("SEQ_LEN", seq_len),
                ("EOS_TOKEN", self.eos_token_id),
            ],
            grid=(total, 1, 1),
            threadgroup=(min(256, total), 1, 1),
            output_shapes=[(batch, seq_len, self.ngram_heads)],
            output_dtypes=[mx.int64],
            stream=mx.gpu,
        )[0]

    @property
    def file_backed(self) -> bool:
        return getattr(self.ngram_embedding, "is_file_backed", False)

    def ngram_ids(
        self,
        input_ids: mx.array,
        cache: Optional[ArraysCache] = None,
        mask: Optional[mx.array] = None,
    ):
        if self.hash_backend == "metal" or (
            self.hash_backend == "metal_prefill"
            and input_ids.shape[1] >= self.metal_hash_min_tokens
        ):
            return self._ngram_ids_metal(input_ids, cache, mask)
        return mx.array(
            self._ngram_ids_numpy(input_ids, cache, mask), dtype=mx.int64
        )

    def prefetch_prompt_chunk(
        self, chunk_tokens: np.ndarray, previous: np.ndarray
    ) -> None:
        """Warm the NVMe rows an upcoming prompt chunk will gather.

        ``previous`` holds the up-to-context_len prompt tokens right before
        the chunk. Hashing and preads run on the embedding's prefetch pool;
        the call returns immediately and mutates no cache state. No-op for
        resident embeddings.

        Stage 1: this warms the page cache, so the chunk's foreground
        lookup re-reads the same rows warm (~1 us/row). TODO: hand the
        prefetched row bytes to the next chunk's lookup directly to also
        skip the warm re-read; needs a keyed handoff between the generate
        loop and the forward pass.
        """
        if not self.file_backed:
            return
        chunk = np.asarray(chunk_tokens, dtype=np.int64)
        prev = np.asarray(previous, dtype=np.int64)
        if chunk.ndim == 1:
            chunk = chunk[None]
        if prev.ndim == 1:
            prev = prev[None]
        if chunk.size == 0:
            return

        def hash_and_warm():
            ids = self._ngram_ids_numpy(mx.array(chunk), None, previous=prev)
            self.ngram_embedding.prefetch_rows(ids)

        self.ngram_embedding.submit_prefetch(hash_and_warm)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[ArraysCache] = None,
        mask: Optional[mx.array] = None,
    ):
        # File-backed embeddings force the CPU id path: ids are hashed and
        # deduplicated on CPU and the rows are pread from NVMe, so a Metal
        # hash round-trip would only add a sync.
        if (
            self.file_backed
            or self.hash_backend == "routed_cpu"
            or (
                self.hash_backend == "metal_prefill"
                and input_ids.shape[1] < self.metal_hash_min_tokens
            )
        ):
            ids = self._ngram_ids_numpy(input_ids, cache, mask)
            return self.ngram_embedding.lookup_numpy(ids).reshape(
                *input_ids.shape, -1
            )
        return self.ngram_embedding(
            self.ngram_ids(input_ids, cache, mask)
        ).reshape(*input_ids.shape, -1)


class PLELayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hidden_size * args.hc_count
        self.ple_embedding = NGramEmbedding(
            args, args.ple_embed_dim, layer_idx, ple_layer_index
        )
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = GroupRMSNorm(hc_hidden, args.hidden_size, args.rms_norm_eps)
        self.norm_query = GroupRMSNorm(hc_hidden, args.hidden_size, args.rms_norm_eps)
        self.norm_conv = GroupRMSNorm(hc_hidden, args.hidden_size, args.rms_norm_eps)
        self.short_conv_state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        self.conv1d = nn.Conv1d(
            hc_hidden,
            hc_hidden,
            args.ple_conv_kernel_size,
            dilation=args.ngram_size,
            groups=hc_hidden,
            bias=False,
        )

    def _short_conv(self, x: mx.array, cache: Optional[ArraysCache], mask=None):
        state = cache[2] if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], self.short_conv_state_len, x.shape[-1]), x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            # The conv is causal, so the branch OUTPUT at a valid position is
            # already pad-free; the persistent tail is not.  Cut each row's
            # window at its own last valid position instead of at the padded
            # width, or the row carries pads in its conv state forever.
            cache[2] = mx.contiguous(
                conv_input[:, -self.short_conv_state_len :, :]
                if mask is None
                else _row_tail(
                    conv_input, _valid_span_end(mask), self.short_conv_state_len
                )
            )
        return nn.silu(self.conv1d(conv_input))[:, -x.shape[1] :, :]

    def __call__(self, hidden: mx.array, input_ids: mx.array, cache=None, mask=None):
        if mask is None and isinstance(cache, ArraysCache):
            # A model with no linear layer builds no ssm mask, so read the
            # padding geometry off the cache rather than trust the caller.
            if cache.lengths is not None or cache.left_padding is not None:
                mask = cache.make_mask(input_ids.shape[1])
        previous_conv = cache[2] if cache is not None else None
        previous_tokens = cache[3] if cache is not None else None
        embeddings = self.ple_embedding(input_ids, cache, mask)
        key = self.norm_key(self.key_proj(embeddings)).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        value = self.value_proj(embeddings)
        query = self.norm_query(hidden).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        gate = mx.sum(key * query, axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * value[..., None, :]).reshape(*hidden.shape)
        normed = self.norm_conv(gated)
        if mask is not None:
            gated = mx.where(mask[..., None], gated, 0)
            normed = mx.where(mask[..., None], normed, 0)
        conv = self._short_conv(normed, cache, mask)
        spans = (
            cache.rollback_spans(input_ids.shape[1], mask)
            if isinstance(cache, Qwen4ArraysCache) and cache.speculating
            else None
        )
        if spans is not None:
            # Staging is gated on the geometry being DESCRIBABLE per row, not
            # on it being unpadded: a right-padded speculative slab (the
            # uniform-width verify, where rows propose different draft counts)
            # is exactly the case that has to roll back, and the old
            # ``mask is None`` predicate disarmed precisely there.
            state_len = self.short_conv_state_len
            conv_base = previous_conv
            if conv_base is None:
                conv_base = mx.zeros(
                    (normed.shape[0], state_len, normed.shape[-1]), normed.dtype
                )
            conv_input = mx.concatenate([conv_base, normed], axis=1)
            context_len = self.ple_embedding.context_len
            token_base = previous_tokens
            if token_base is None:
                token_base = mx.full(
                    (input_ids.shape[0], context_len),
                    self.ple_embedding.eos_token_id,
                    dtype=mx.int64,
                )
            staged_ids = input_ids.astype(mx.int64)
            if mask is not None:
                # The same substitution the live history update makes, so a
                # replayed window is the window that was stored.
                staged_ids = mx.where(
                    mask, staged_ids, self.ple_embedding.eos_token_id
                )
            token_history = mx.concatenate([token_base, staged_ids], axis=1)

            def _ple_rollback(
                m, ci=conv_input, th=token_history, sl=state_len, cl=context_len
            ):
                return [
                    mx.contiguous(ci[:, m : m + sl, :]),
                    mx.contiguous(th[:, m : m + cl]),
                ]

            def _ple_rollback_rows(
                lengths,
                ci=conv_input,
                th=token_history,
                sl=state_len,
                cl=context_len,
            ):
                # Both states are the window ending at the row's own length,
                # so the ragged replay is the same gather as the pad-safe
                # update, in one graph.
                ends = mx.array(list(lengths))
                return [
                    mx.contiguous(_row_tail(ci, ends, sl)),
                    mx.contiguous(_row_tail(th, ends, cl)),
                ]

            cache.stage_ple_rollback(
                input_ids.shape[1],
                _ple_rollback,
                [previous_conv, previous_tokens],
                per_row_fn=_ple_rollback_rows,
            )
        return gated + conv


_ROPE_POSITION_FREQS: Dict[tuple, mx.array] = {}


def _apply_rope_positions(x: mx.array, positions: mx.array, dims: int, base: float):
    """Transformers-compatible non-traditional partial RoPE at arbitrary positions."""
    if dims == 0:
        return x
    freqs = _ROPE_POSITION_FREQS.get((dims, base))
    if freqs is None:
        freqs = mx.exp(-math.log(base) * mx.arange(0, dims, 2) / dims)
        _ROPE_POSITION_FREQS[(dims, base)] = freqs
    angles = positions[..., None].astype(mx.float32) * freqs
    cos, sin = mx.cos(angles), mx.sin(angles)
    rope, tail = x[..., :dims], x[..., dims:]
    half = dims // 2
    left, right = rope[..., :half], rope[..., half:]
    rotated = mx.concatenate([left * cos - right * sin, right * cos + left * sin], axis=-1)
    return mx.concatenate([rotated.astype(x.dtype), tail], axis=-1)


# THE armed cross-call QSA state, shared by both cache types. Everything here
# is derived from the block grid and the row set, so nothing here may outlive a
# change to either -- see release_qsa_cycle.
_QSA_CYCLE_STATE = (
    ("_mtp_share_topk", False),
    ("_mtp_shared_topk", None),
    ("_qsa_pooled_keys", None),
    ("_qsa_pooled_ratio", None),
)


# QSA caches are NOT quantizable, and both classes have to say so the same way.
#
# The attention KV is only half of a QSA cache. Beside it sits ``index_keys``,
# the raw pre-pooling indexer-key ledger that ``QSAIndexer.__call__`` appends to
# every forward and pools its block grid from, plus the cross-call state in
# ``_QSA_CYCLE_STATE``. No quantized cache class carries either, and the two
# classes used to fail in OPPOSITE directions because of it:
#
#   * ``QSAKVCache`` inherited ``KVCache.to_quantized`` and converted into a
#     plain ``QuantizedKVCache``, DROPPING the ledger and every cycle field --
#     so the indexer's ``cache.update_index_keys(raw)`` had nothing to call.
#   * ``BatchQSAKVCache`` had no ``to_quantized`` at all, so the ``hasattr``
#     gate in ``maybe_quantize_kv_cache`` skipped it and the batched path
#     ignored the user's ``--kv-bits`` in SILENCE.
#
# Both are closed the same way: refuse, out loud, on both classes. Carrying the
# QSA side state through quantization is a feature, not a bug fix -- it needs
# quantized twins of the whole batch ledger protocol (merge / filter / extend /
# extract / ragged trim, all written against unquantized ``keys``) and its own
# equivalence battery, and nothing serves QSA with quantized KV today.
_QSA_KV_QUANT_UNSUPPORTED = (
    "QSA attention caches keep a raw indexer-key ledger (index_keys) and "
    "cross-call cycle state beside the attention KV, and no quantized cache "
    "class carries them, so quantizing would leave the indexer without the "
    "ledger it reads every forward. Serve QSA models with unquantized KV "
    "(drop --kv-bits)."
)


def _qsa_to_quantized(self, group_size: int = 64, bits: int = 4, **kwargs):
    """The refusal itself, for anyone who calls ``to_quantized`` directly.

    Both QSA cache classes bind THIS function object rather than each defining
    their own, so they cannot drift apart again; the regression test asserts
    that identity. ``**kwargs`` swallows the asymmetric/rotated extension
    (``key_bits``/``value_bits``/``rotate``) so the refusal is the same on
    every call shape ``maybe_quantize_kv_cache`` uses.
    """
    raise NotImplementedError(_QSA_KV_QUANT_UNSUPPORTED)


class BatchQSAKVCache(BatchKVCache):
    """Batched QSA cache retaining raw indexer keys beside attention KV."""

    # Refused identically on both QSA classes -- see _QSA_KV_QUANT_UNSUPPORTED.
    # The attribute is what ``maybe_quantize_kv_cache`` reads so it can refuse
    # at setup instead of when ``offset`` first crosses ``quantized_kv_start``;
    # the method is what a direct caller gets. Defining ``to_quantized`` at all
    # is also what stops the ``hasattr`` gate from skipping this class quietly.
    kv_quantization_unsupported = _QSA_KV_QUANT_UNSUPPORTED
    to_quantized = _qsa_to_quantized

    # ``index_keys`` is a per-row ledger parallel to the KV columns, so a
    # ragged trim rolls it with the same per-row shifts.  The base check that
    # it is at least as wide as the cursor is wanted: only an MTP draft cycle
    # may run it short (a shared-top-k step appends no key), and that cache is
    # rewound uniformly, never raggedly.
    _RAGGED_TRIM_AUX_ARRAYS = (("index_keys", 1),)
    _QSA_CYCLE_FIELDS = _QSA_CYCLE_STATE

    def __new__(cls, *args, **kwargs):
        # ``from_state`` builds through ``cls.__new__(cls)`` and assigns
        # ``state``, so __init__ never runs on the prompt-cache load path.
        # Every attribute a method reachable from there reads has to be set
        # here: the state setter alone routes through release_qsa_cycle, which
        # reads all four cycle fields, and through max_left_padding.
        instance = super().__new__(cls)
        instance.index_keys = None
        instance._max_left_pad = None
        for name, blank in cls._QSA_CYCLE_FIELDS:
            setattr(instance, name, blank)
        return instance

    def __init__(self, left_padding: List[int], attention_backend=None):
        # __new__ owns the QSA fields; it runs on both construction paths.
        super().__init__(left_padding, attention_backend=attention_backend)

    def max_left_padding(self) -> int:
        """Host copy of ``left_padding.max()``, keyed by array identity.

        ``left_padding`` is rebound only at membership boundaries (merge,
        filter, extend, finalize) and mx arrays are immutable, so an identity
        miss is exactly the set of events that can change the maximum.  The
        pooled-key bound below needs this every forward and must not sync.
        """
        padding = self.left_padding
        cached = self._max_left_pad
        if cached is None or cached[0] is not padding:
            self._max_left_pad = (padding, int(padding.max().item()))
        return self._max_left_pad[1]

    def release_qsa_cycle(
        self,
        who: str,
        *,
        rows=None,
        keep_pooled=True,
        cursor_final=True,
        keep_shared=False,
    ):
        """The ONE exit for armed QSA state. Every lifecycle method routes here.

        This class has now been bitten three times by armed state outliving
        the geometry it was computed against (the live ``trim()`` desync, the
        aborted draft cycle, the filtered block ids), so the rules below are
        stated once and derived from the CURRENT geometry -- never from which
        method called:

        * **The MTP shared top-k dies at every membership or cycle boundary.**
          It is a cycle-local set of block ids with no geometry-independent
          meaning: ``filter`` can shrink the physical grid under it (dropping
          the common left padding), which leaves ids past ``n_blocks`` that
          ``_QSA_SCATTER_CHOSEN`` would scatter out of bounds. The only
          exception is a prepare/finalize pair inside the same ragged MTP
          draft cycle: that pair changes physical padding but not the rows or
          their logical block ids, so ``keep_shared=True`` preserves the set.
        * **The pooled keys are re-bounded, not dropped.** They are indexed by
          LOGICAL block per row, so they survive anything that does not change
          a row's own logical content -- and the bound that expresses this,
          ``(cursor - max left padding) // ratio``, is read off the live state
          after the caller has updated it. That covers a rewind (offsets
          shrink), a filter (rows subset, then re-bound) and ``finalize`` (the
          right-padding roll only ever invalidates blocks past the shortest
          row). ``keep_pooled=False`` is for the one case where the cache stops
          being the same cache at all: a ``state`` restore.

        Post-condition: with the cycle released the raw-key ledger must span
        exactly the cursor. Only a live shared-top-k cycle may run it short.
        ``cursor_final=False`` defers that check either while such a cycle is
        still live, or for the ``state`` setter where ``_idx`` remains the
        allocated buffer width until ``meta_state`` lands the real one.
        """
        pooled, ratio = self._qsa_pooled_keys, self._qsa_pooled_ratio
        share_topk = self._mtp_share_topk
        shared_topk = self._mtp_shared_topk
        # Blank everything first, so a field added to _QSA_CYCLE_FIELDS later
        # is cleared by default and only what is re-derived below survives.
        for name, blank in self._QSA_CYCLE_FIELDS:
            setattr(self, name, blank)
        if keep_shared:
            self._mtp_share_topk = share_topk
            self._mtp_shared_topk = shared_topk
        if keep_pooled and pooled is not None:
            if rows is not None:
                pooled = mx.contiguous(pooled[rows])
            if not ratio:
                raise RuntimeError("QSA pooled keys are cached without a ratio")
            keep = max(0, self._idx - self.max_left_padding()) // ratio
            if keep and pooled.shape[0] == self.offset.shape[0]:
                self._qsa_pooled_keys = (
                    pooled if keep >= pooled.shape[1]
                    else mx.contiguous(pooled[:, :keep])
                )
                self._qsa_pooled_ratio = ratio
        if cursor_final:
            self._reconcile_index_ledger(who)

    def _reconcile_index_ledger(self, who: str):
        """``len(index_keys) == cursor``, or say exactly why not.

        A ledger WIDER than the cursor is the normal aftermath of a rewind and
        is simply cut. A ledger NARROWER than the cursor means a shared-top-k
        cycle skipped raw-key appends for KV that is still in the cache: the
        13-vs-12 shape of the live desync this class shipped once already.
        """
        if self.index_keys is None:
            return
        width = self.index_keys.shape[1]
        if width > self._idx:
            self.index_keys = mx.contiguous(self.index_keys[:, : self._idx])
        elif width < self._idx:
            raise RuntimeError(
                f"{who}: the QSA raw-key ledger holds {width} positions but "
                f"the cursor is at {self._idx}. A shared-top-k draft cycle "
                "left un-ledgered KV behind and was ended without rewinding "
                "the drafted span."
            )

    def trim(self, n):
        n = super().trim(n)
        self.release_qsa_cycle("BatchQSAKVCache.trim")
        return n

    def trim_ragged(self, n, *, validate: bool = True):
        drops = super().trim_ragged(n, validate=validate)
        self.release_qsa_cycle("BatchQSAKVCache.trim_ragged")
        return drops

    def prepare(self, *args, **kwargs):
        super().prepare(*args, **kwargs)
        self.release_qsa_cycle("BatchQSAKVCache.prepare")

    def prepare_self_mtp_step(self, *args, **kwargs):
        if not self._mtp_share_topk:
            return self.prepare(*args, **kwargs)
        super().prepare(*args, **kwargs)
        self.release_qsa_cycle(
            "BatchQSAKVCache.prepare_self_mtp_step",
            cursor_final=False,
            keep_shared=True,
        )

    def last_valid_query(self, values: mx.array) -> mx.array:
        """Gather each row's final non-padding query from ``[B, L, ...]``."""
        if values.ndim < 2 or values.shape[0] != self.offset.shape[0]:
            raise ValueError("QSA query values must have shape [batch, length, ...]")
        length = values.shape[1]
        if length == 0:
            raise ValueError("QSA query values cannot have zero length")
        padding = self._right_padding
        if padding is None:
            return values[:, -1]
        invalid = mx.any((padding < 0) | (padding >= length))
        if bool(invalid.item()):
            raise ValueError("QSA right padding must leave one valid query per row")
        rows = mx.arange(values.shape[0], dtype=mx.int32)
        positions = length - padding.astype(mx.int32) - 1
        return values[rows, positions]

    def update_index_keys(self, keys: mx.array):
        self.index_keys = (
            keys
            if self.index_keys is None
            else mx.concatenate([self.index_keys[:, : self._idx], keys], axis=1)
        )
        return self.index_keys

    @property
    def state(self):
        return (*BatchKVCache.state.fget(self), self.index_keys)

    @state.setter
    def state(self, value):
        BatchKVCache.state.fset(self, value[:4])
        self.index_keys = value[4]
        self._max_left_pad = None
        # A restore replaces the contents wholesale, so the pooled keys are
        # not this cache's any more even at an unchanged row count. The ledger
        # check waits for meta_state, which lands the real cursor.
        self.release_qsa_cycle(
            "BatchQSAKVCache.state", keep_pooled=False, cursor_final=False
        )

    @property
    def meta_state(self):
        return BatchKVCache.meta_state.fget(self)

    @meta_state.setter
    def meta_state(self, value):
        BatchKVCache.meta_state.fset(self, value)
        if value:
            # ``_idx`` is only now the real cursor, so this is where a restored
            # ledger can be checked against it at all.
            self._reconcile_index_ledger("BatchQSAKVCache.meta_state")

    @property
    def nbytes(self):
        return super().nbytes + (
            0 if self.index_keys is None else self.index_keys.nbytes
        )

    def _finalize(self, *, keep_shared=False):
        padding = self._right_padding
        if padding is not None and self.index_keys is not None:
            self.index_keys = dynamic_roll(self.index_keys, padding, axis=1)
        super().finalize()
        self.release_qsa_cycle(
            "BatchQSAKVCache.finalize",
            cursor_final=not keep_shared,
            keep_shared=keep_shared,
        )

    def finalize(self):
        self._finalize()

    def finalize_self_mtp_step(self):
        if not self._mtp_share_topk:
            return self.finalize()
        self._finalize(keep_shared=True)

    def filter(self, batch_indices):
        min_left_pad = self.left_padding[batch_indices].min().item()
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if min_left_pad > 0:
                self.index_keys = self.index_keys[:, min_left_pad:]
        super().filter(batch_indices)
        # After super(), so the bound is read off the new cursor and padding.
        self.release_qsa_cycle("BatchQSAKVCache.filter", rows=batch_indices)

    def extend(self, other):
        index_a, index_b = self.index_keys, other.index_keys
        idx_a, idx_b = self._idx, other._idx
        if index_a is None and index_b is None:
            merged_index = None
        else:
            populated = index_a if index_a is not None else index_b
            dims, dtype = populated.shape[-1], populated.dtype
            max_idx = max(idx_a, idx_b)

            def pad(index, idx, batch):
                if index is None:
                    index = mx.zeros((batch, 0, dims), dtype=dtype)
                else:
                    index = index[:, :idx]
                return mx.pad(index, [(0, 0), (max_idx - idx, 0), (0, 0)])

            merged_index = mx.concatenate(
                [
                    pad(index_a, idx_a, self.offset.shape[0]),
                    pad(index_b, idx_b, other.offset.shape[0]),
                ]
            )
        super().extend(other)
        self.index_keys = merged_index
        # The joining rows carry neither, and a partial set has no meaning as
        # a batch tensor: a join ends the cycle for every lane. The row-count
        # change makes the release drop the pooled keys on its own.
        self._max_left_pad = None
        self.release_qsa_cycle("BatchQSAKVCache.extend")

    def extract(self, idx):
        # A row leaves as a standalone sequence, so its ledger must be whole.
        self._reconcile_index_ledger("BatchQSAKVCache.extract")
        cache = QSAKVCache()
        padding = self.left_padding[idx].item()
        end = self._idx
        if self._right_padding is not None:
            end -= int(self._right_padding[idx].item())
        if self.keys is not None:
            cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding:end])
            cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding:end])
            cache.offset = cache.keys.shape[2]
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(
                self.index_keys[idx : idx + 1, padding:end]
            )
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [cache.size() for cache in caches]
        width = max(lengths)
        padding = [width - length for length in lengths]
        batch = cls(padding)
        if width == 0:
            return batch

        base = BatchKVCache.merge(caches)
        batch.keys = base.keys
        batch.values = base.values
        batch.offset = base.offset
        batch.left_padding = base.left_padding
        batch._idx = base._idx

        populated = next(
            (cache.index_keys for cache in caches if cache.index_keys is not None),
            None,
        )
        if populated is not None:
            dims, dtype = populated.shape[-1], populated.dtype
            rows = []
            for cache, length, left in zip(caches, lengths, padding):
                values = cache.index_keys
                if values is None:
                    values = mx.zeros((1, 0, dims), dtype=dtype)
                else:
                    values = values[:, :length]
                rows.append(mx.pad(values, [(0, 0), (left, 0), (0, 0)]))
            batch.index_keys = mx.concatenate(rows)
        return batch


class QSAKVCache(KVCache):
    """KV cache with the raw, pre-pooling indexer keys QSA also requires."""

    _QSA_CYCLE_FIELDS = _QSA_CYCLE_STATE

    # The single-sequence twin of the refusal on BatchQSAKVCache: the same
    # attribute and the same function object, so the two classes agree.
    kv_quantization_unsupported = _QSA_KV_QUANT_UNSUPPORTED
    to_quantized = _qsa_to_quantized

    def __new__(cls, *args, **kwargs):
        # Same from_state contract as BatchQSAKVCache: __init__ does not run
        # on the prompt-cache load path.
        instance = super().__new__(cls)
        instance.index_keys = None
        for name, blank in cls._QSA_CYCLE_FIELDS:
            setattr(instance, name, blank)
        return instance

    def __init__(self):
        # __new__ owns the QSA fields; it runs on both construction paths.
        super().__init__()

    def update_index_keys(self, keys: mx.array):
        self.index_keys = keys if self.index_keys is None else mx.concatenate([self.index_keys[:, : self.offset], keys], axis=1)
        return self.index_keys

    def release_qsa_cycle(self, who: str, *, keep_pooled: bool = True):
        """Single-sequence twin of ``BatchQSAKVCache.release_qsa_cycle``.

        A rewind ends any MTP draft cycle.  A stale shared top-k would make
        the next uncycled ``mtp_step`` skip its raw-key append and desync
        ``index_keys`` from the KV offset; ``mtp_start_cycle`` re-arms it.
        A block mean is a closed window over ``ratio`` tokens, so every block
        fully inside the trimmed offset stays exact and only the tail is cut.
        """
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        if self.index_keys is not None:
            width = self.index_keys.shape[1]
            if width < self.offset:
                raise RuntimeError(
                    f"{who}: the QSA raw-key ledger holds {width} positions "
                    f"but the offset is {self.offset}. A shared-top-k draft "
                    "cycle left un-ledgered KV behind and was ended without "
                    "rewinding the drafted span."
                )
            if width > self.offset:
                self.index_keys = mx.contiguous(
                    self.index_keys[:, : self.offset]
                )
        if self._qsa_pooled_keys is not None:
            keep = 0 if not keep_pooled else self.offset // self._qsa_pooled_ratio
            if keep == 0:
                self._qsa_pooled_keys = None
                self._qsa_pooled_ratio = None
            elif keep < self._qsa_pooled_keys.shape[1]:
                self._qsa_pooled_keys = mx.contiguous(
                    self._qsa_pooled_keys[:, :keep]
                )

    def trim(self, n):
        n = super().trim(n)
        self.release_qsa_cycle("QSAKVCache.trim")
        return n

    @classmethod
    def merge(cls, caches):
        return BatchQSAKVCache.merge(caches)

    @property
    def state(self):
        return self.keys, self.values, self.index_keys

    @state.setter
    def state(self, value):
        self.keys, self.values, self.index_keys = value
        self.offset = 0 if self.keys is None else self.keys.shape[2]
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        self._qsa_pooled_keys = None
        self._qsa_pooled_ratio = None

    @property
    def nbytes(self):
        return super().nbytes + (0 if self.index_keys is None else self.index_keys.nbytes)


@dataclass(frozen=True)
class QSACompactBlocks:
    """Sorted, prefix-packed block ids, the shape a gather kernel wants.

    Ids are LOGICAL.  Row ``b``'s physical block start is
    ``left_padding[b] + block_id * block_size``, or just ``block_id *
    block_size`` without left padding.  ``[tail_start, tail_stop)`` is the
    incomplete block no selection names, and may be empty.  ``causal_mask``
    stays attached because blocks alone are NOT causally complete; see
    ``QSASelection.dense_mask``.
    """

    block_ids: mx.array  # [B, L, K], valid prefix ascending, suffix zeroed
    block_counts: mx.array  # [B, L]
    tail_start: mx.array  # [B, L], logical, inclusive
    tail_stop: mx.array  # [B, L], logical, exclusive
    left_padding: Optional[mx.array]  # [B]
    block_size: int
    physical_width: int
    causal_mask: Optional[mx.array]

    @property
    def block_valid(self) -> mx.array:
        """Derived, never stored: a second tensor could desynchronize."""
        return mx.arange(self.block_ids.shape[-1]) < self.block_counts[..., None]


def _compact_qsa_block_ids(
    selected_block_ids: mx.array,
    selected_is_valid: mx.array,
    *,
    n_blocks: int,
):
    """Return sorted, prefix-packed logical ids and their counts.

    ``argpartition`` gives no order, and invalid slots sit anywhere -- with
    one valid block the first eight slots are invalid -- so a prefix scan of
    the raw ids would be wrong.  Keying invalid slots at ``n_blocks``, past
    every real id, sorts them to the end instead.
    """
    if selected_block_ids.ndim != 3 or selected_is_valid.ndim != 3:
        raise ValueError("compaction wants [B, L, K] ids and validity")
    if selected_block_ids.shape != selected_is_valid.shape:
        raise ValueError(
            "compaction shape mismatch: "
            f"{selected_block_ids.shape} ids vs {selected_is_valid.shape} validity"
        )
    width = selected_block_ids.shape[-1]
    keys = mx.where(
        selected_is_valid,
        selected_block_ids.astype(mx.int32),
        mx.array(n_blocks, dtype=mx.int32),
    )
    ids = mx.take_along_axis(
        selected_block_ids, mx.argsort(keys, axis=-1), axis=-1
    )
    counts = mx.sum(selected_is_valid.astype(mx.int32), axis=-1)
    packed = mx.arange(width) < counts[..., None]
    return mx.where(packed, ids, mx.zeros_like(ids)), counts


@dataclass(frozen=True)
class QSASelection:
    """What the indexer chose, before it becomes an attention mask.

    ``dense_mask()`` rebuilds today's mask array operation for operation, so
    nothing observable changes.  ``compact_blocks()`` is the gather-shaped
    view and is LAZY: the mask path never calls it and pays nothing.

    Kinds: ``explicit`` is the normal sparse selection, ``implicit_all`` is a
    dense step whose mask IS the causal mask, and ``mask_only`` is the
    ``SinkWindowKVCache`` path, which has no QSA blocks at all.
    """

    kind: str
    batch: int
    length: int
    block_size: int
    raw_block_ids: Optional[mx.array] = None  # [B, L, K], unsorted, ragged
    valid_blocks: Optional[mx.array] = None  # [B or 1, L, N]
    q_positions: Optional[mx.array] = None  # [B or 1, L], logical
    token_positions: Optional[mx.array] = None  # [B or 1, T], logical
    causal_mask: Optional[mx.array] = None
    passthrough_mask: Optional[mx.array] = None
    left_padding: Optional[mx.array] = None  # [B]
    offset: Union[int, mx.array] = 0
    physical_width: int = 0
    n_blocks: int = 0
    # Snapshot, so a delayed dense_mask() cannot read a different global.
    scatter_chosen: bool = False

    def __post_init__(self):
        # Structural only.  An .item() here would sync the device every step.
        if self.kind not in ("explicit", "implicit_all", "mask_only"):
            raise ValueError(f"unknown QSA selection kind {self.kind!r}")
        if self.kind == "mask_only":
            return
        if self.physical_width // self.block_size != self.n_blocks:
            raise ValueError("block grid does not match the physical width")
        if self.left_padding is not None and self.left_padding.shape != (
            self.batch,
        ):
            raise ValueError("left padding wants one entry per row")
        if self.causal_mask is not None and self.causal_mask.shape[-2:] != (
            self.length,
            self.physical_width,
        ):
            raise ValueError("causal mask must end in [L, physical width]")
        if self.kind == "implicit_all":
            return
        if self.raw_block_ids.ndim != 3 or self.valid_blocks.ndim != 3:
            raise ValueError("explicit selection wants rank-3 ids and validity")
        if self.raw_block_ids.shape[:2] != (self.batch, self.length):
            raise ValueError("selected ids must be [B, L, K]")
        if self.raw_block_ids.shape[-1] > self.n_blocks:
            raise ValueError("more selected ids than blocks")
        if self.valid_blocks.shape[1:] != (self.length, self.n_blocks):
            raise ValueError("block validity must be [B or 1, L, N]")
        if self.q_positions.shape[-1] != self.length:
            raise ValueError("query positions must be [B or 1, L]")
        if self.token_positions.shape[-1] != self.physical_width:
            raise ValueError("token positions must span the physical width")

    def dense_mask(self) -> Optional[mx.array]:
        """Today's QSA mask, rebuilt operation for operation."""
        if self.kind == "mask_only":
            return self.passthrough_mask
        if self.kind == "implicit_all":
            return self.causal_mask
        batch, length = self.batch, self.length
        n_blocks, total = self.n_blocks, self.physical_width
        selected, valid_blocks = self.raw_block_ids, self.valid_blocks
        q_pos, token_logical = self.q_positions, self.token_positions
        if self.scatter_chosen:
            # ``argpartition`` output has no duplicate indices, so a scatter
            # of ones is equivalent to the one-hot broadcast reduction.
            # Default since 2026-08-28.
            chosen = mx.put_along_axis(
                mx.zeros((batch, length, n_blocks), dtype=mx.bool_),
                selected,
                mx.array(True),
                axis=-1,
            )
        else:
            # Retained as the reference form the scatter is verified against
            # (the equality tests pin BOTH arms, so neither is the default),
            # and as the ``=0`` escape hatch.  Not the shipped path: it costs
            # a [B, L, K, n_blocks] boolean intermediate for the same bits.
            block_ids = mx.arange(n_blocks)
            chosen = mx.any(
                selected[..., None] == block_ids[None, None, None, :], axis=-2
            )
        chosen = chosen & valid_blocks
        # ``clip`` where the unpadded path clamped: the lower bound only bites
        # on left padding, whose columns the ``>= 0`` term below removes.
        token_block = mx.clip(token_logical // self.block_size, 0, n_blocks - 1)
        selected_tokens = mx.take_along_axis(
            chosen,
            mx.broadcast_to(token_block[:, None, :], (batch, length, total)),
            axis=-1,
        )
        complete = ((q_pos + 1) // self.block_size) * self.block_size
        tail = (token_logical[:, None, :] >= complete[..., None]) & (
            token_logical[:, None, :] <= q_pos[..., None]
        )
        sparse = selected_tokens | tail
        if self.left_padding is not None:
            sparse = sparse & (token_logical[:, None, :] >= 0)
        sparse = sparse[:, None, :, :]
        # ``create_attention_mask`` deliberately returns ``None`` for a
        # single-token decode because every cached position is causal.  QSA
        # still needs its sparse selection mask in that case.
        #
        # This conjunction MUST stay last and MUST stay here.  The clip above
        # can name a block holding future tokens -- ratio 4, total 10, left
        # padding 1, query 7 clips logical token 8 from block 2 down to the
        # selected block 1 -- and the tail does not remove it.  Only this
        # term does, so the selection is not itself an attention mask.
        return sparse if self.causal_mask is None else self.causal_mask & sparse

    def _logical_coordinates(self):
        """Rebuild ``(q_pos, token_logical)`` for a kind that never derived them.

        Deliberately NOT shared with ``QSAIndexer.__call__``: the dense path
        returns before it needs coordinates, and making it compute them would
        put new work on the hot path.
        """
        if self.left_padding is None:
            return (
                mx.arange(self.offset, self.offset + self.length)[None, :],
                mx.arange(self.physical_width)[None, :],
            )
        return (
            self.offset[:, None] + mx.arange(self.length)[None, :],
            mx.arange(self.physical_width)[None, :] - self.left_padding[:, None],
        )

    def compact_blocks(self) -> Optional[QSACompactBlocks]:
        """Sorted, prefix-packed blocks for a future gather kernel.

        Lazy on purpose: sorting up to ``block_topk`` ids on every masked SDPA
        call would be new hot-path work for no change in output.
        """
        if self.kind == "mask_only":
            # Windowed MTP replaces global QSA outright; it has no blocks.
            return None
        if self.kind == "implicit_all":
            q_pos, _ = self._logical_coordinates()
            starts = mx.arange(self.n_blocks) * self.block_size
            valid_blocks = (starts + self.block_size - 1)[
                None, None, :
            ] <= q_pos[..., None]
            ids = mx.broadcast_to(
                mx.arange(self.n_blocks, dtype=mx.uint32)[None, None, :],
                (self.batch, self.length, self.n_blocks),
            )
        else:
            q_pos, valid_blocks, ids = (
                self.q_positions,
                self.valid_blocks,
                self.raw_block_ids,
            )
        if valid_blocks.shape[0] != ids.shape[0]:
            valid_blocks = mx.broadcast_to(
                valid_blocks, (ids.shape[0],) + valid_blocks.shape[1:]
            )
        block_ids, counts = _compact_qsa_block_ids(
            ids,
            mx.take_along_axis(valid_blocks, ids, axis=-1),
            n_blocks=self.n_blocks,
        )
        tail_stop = q_pos + 1
        tail_start = (tail_stop // self.block_size) * self.block_size
        if tail_stop.shape[0] != self.batch:
            shape = (self.batch, self.length)
            tail_stop = mx.broadcast_to(tail_stop, shape)
            tail_start = mx.broadcast_to(tail_start, shape)
        return QSACompactBlocks(
            block_ids=block_ids,
            block_counts=counts,
            tail_start=tail_start,
            tail_stop=tail_stop,
            left_padding=self.left_padding,
            block_size=self.block_size,
            physical_width=self.physical_width,
            causal_mask=self.causal_mask,
        )


class QSAIndexer(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = args.indexer_budget // args.indexer_compress_ratio
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (args.indexer_n_heads + args.indexer_kv_heads) * args.indexer_head_dim,
            bias=False,
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def _pool_blocks(self, raw: mx.array, starts: mx.array) -> mx.array:
        """Mean-pool, layernorm, and rope closed key blocks.

        ``starts`` carries absolute block-start positions, so a partial
        recompute matches the full recompute value for value.
        """
        batch = raw.shape[0]
        pooled = (
            raw.reshape(batch, starts.shape[0], self.compress_ratio, self.head_dim)
            .astype(mx.float32)
            .mean(axis=2)
            .astype(raw.dtype)
        )
        pooled = self.k_layernorm(pooled)
        return _apply_rope_positions(
            pooled, starts[None, :], self.rotary_dim, self.rope_theta
        )

    def _pool_blocks_left_padded(
        self, all_raw: mx.array, starts: mx.array, left_pad
    ) -> mx.array:
        """Pool the logical blocks ``starts`` out of a left-padded ledger.

        Row ``b``'s logical block at ``start`` occupies physical columns
        ``[left_pad[b] + start, left_pad[b] + start + r)``, so gather each
        row's own columns before pooling.  The block grid is sized off the
        physical width, an upper bound on any row's own block count, so a
        padded row's trailing gathers run past the ledger and are clamped here.

        A clamped block IS pooled and scored -- what it can never do is reach
        the returned mask.  Row ``b``'s deepest query sits at logical
        ``total - 1 - left_pad[b]``, and a block clamps exactly when
        ``left_pad[b] + block_end > total - 1``, i.e. when
        ``block_end > total - 1 - left_pad[b] >= q_pos``: precisely the
        condition under which ``valid_blocks`` rejects it.  A NaN or Inf from
        garbage keys is block-local and is overwritten by the ``-inf`` in
        ``mx.where(valid_blocks, ...)``; an invalid id that ``argpartition``
        still returns (there can be fewer than ``k`` valid blocks) is dropped
        by ``chosen & valid_blocks``.
        """
        batch, total, _ = all_raw.shape
        columns = mx.minimum(
            left_pad[:, None, None]
            + starts[None, :, None]
            + mx.arange(self.compress_ratio)[None, None, :],
            total - 1,
        )
        gathered = mx.take_along_axis(
            all_raw, columns.reshape(batch, -1)[..., None], axis=1
        )
        return self._pool_blocks(gathered, starts)

    def _dense_by_construction(self, n_blocks, shared_topk) -> bool:
        """True when ``causal_mask & sparse == causal_mask`` for the mask this
        call would build, i.e. when the selection removes no causal cell -- so
        returning ``causal_mask`` is exact.  Raw ``sparse`` may still be wider
        than causal; see the MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT proof.
        """
        if shared_topk is None:
            return n_blocks <= self.block_topk
        # A reused index set covers every block only if it is as wide as the
        # block count; that already implies n_blocks <= block_topk.
        return shared_topk.shape[-1] == n_blocks

    def _pooled_keys(self, all_raw, n_blocks, starts, cache, length, left_pad):
        ratio = self.compress_ratio

        def pool(first, last):
            if left_pad is None:
                return self._pool_blocks(
                    all_raw[:, first * ratio : last * ratio], starts[first:last]
                )
            return self._pool_blocks_left_padded(
                all_raw, starts[first:last], left_pad
            )

        if not (
            _QSA_POOLED_KEY_CACHE and type(cache) in (QSAKVCache, BatchQSAKVCache)
        ):
            return pool(0, n_blocks)
        if left_pad is None and all_raw.shape[1] != cache.offset + length:
            # A raw-key ledger out of step with the KV offset means block
            # positions no longer match token positions; fail loudly instead
            # of pooling from a shifted history.  (The batch path runs the
            # same check against the PHYSICAL write index in __call__.)
            raise RuntimeError(
                "QSA index_keys desync: "
                f"{all_raw.shape[1]} raw keys != offset {cache.offset} "
                f"+ {length} new"
            )
        # A padded row has not closed the block grid's trailing blocks, so
        # those were pooled from CLAMPED columns and would be reused as real
        # values once the row does close them.  Retain only the blocks every
        # row has closed, which the row with the most left padding bounds.
        closed = n_blocks
        if left_pad is not None:
            closed = (all_raw.shape[1] - cache.max_left_padding()) // ratio
        cached = cache._qsa_pooled_keys
        count = 0 if cached is None else cached.shape[1]
        if cached is not None and (
            count > closed
            or cache._qsa_pooled_ratio != ratio
            or cached.shape[0] != all_raw.shape[0]
        ):
            cached, count = None, 0
        if count == n_blocks:
            pooled = cached
        else:
            # A block is final once its last token is written; only blocks
            # closed since the previous call need computing.
            new = pool(count, n_blocks)
            pooled = new if cached is None else mx.concatenate([cached, new], axis=1)
        cache._qsa_pooled_keys = (
            pooled if closed >= n_blocks else mx.contiguous(pooled[:, :closed])
        )
        cache._qsa_pooled_ratio = ratio
        return pooled

    def __call__(
        self,
        hidden: mx.array,
        causal_mask: mx.array,
        cache: QSAKVCache,
        projected_qk: Optional[mx.array] = None,
    ):
        batch, length, _ = hidden.shape
        if isinstance(cache, SinkWindowKVCache):
            # Windowed MTP deliberately replaces the draft head's global QSA
            # lookup with dense sink+recent attention. The full target keeps
            # native QSA and remains the sole verifier.
            mask = cache.make_mask(length, return_array=True)
            return QSASelection(
                kind="mask_only",
                batch=batch,
                length=length,
                block_size=self.compress_ratio,
                passthrough_mask=None if mask is None else mask[None, None, :, :],
            )
        offset = 0 if cache is None else cache.offset
        # This indexer straddles two coordinate systems and ``left_pad`` is the
        # only bridge between them.  A BatchKVCache reports a LOGICAL, per-row
        # ``offset`` (its physical write index minus that row's left padding)
        # while ``index_keys`` and the KV columns are PHYSICAL and shared
        # across the batch: physical column ``j`` of row ``b`` holds logical
        # position ``j - left_pad[b]``.  Everything below -- block starts,
        # block validity, the incomplete tail, both RoPE position sets -- is
        # LOGICAL, so an unequal-length merge reproduces each row's own
        # single-sequence geometry exactly.  Mixing the two emitted an
        # ALL-FALSE mask row for a shorter row (2026-08-27).
        left_pad = None
        if isinstance(offset, mx.array):
            left_pad = cache.left_padding.astype(offset.dtype)
        shared_topk = (
            getattr(cache, "_mtp_shared_topk", None) if cache is not None else None
        )
        if shared_topk is None:
            qk = (
                projected_qk
                if projected_qk is not None
                else self.index_qk_proj(hidden)
            )
            q, raw = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
            raw = raw.reshape(batch, length, self.head_dim)
            # The raw-key append is the ONE step the dense short-circuit below
            # may not skip: the ledger must stay in step with the KV offset so
            # a later step that does cross the budget can pool every block.
            all_raw = raw if cache is None else cache.update_index_keys(raw)
            total = all_raw.shape[1]
            if left_pad is not None and total != cache._idx + length:
                # Same contract as the pooled-key cache check below, for the
                # batch ledger: raw keys out of step with the PHYSICAL write
                # index mean block columns no longer address the keys they
                # name.  Fail loudly rather than pool a shifted history.
                raise RuntimeError(
                    "QSA index_keys desync: "
                    f"{total} raw keys != physical index {cache._idx} "
                    f"+ {length} new"
                )
        else:
            # The current draft token is transient and will be rewound before
            # any accepted span is teacher-forced next cycle.  Skipping its raw
            # index key is therefore safe and is what removes the indexer work.
            # ``total`` counts PHYSICAL columns, so a batch cache reads its
            # write index, not its (per-row, left-padding-adjusted) offset.
            total = (cache._idx if left_pad is not None else offset) + length

        # One logical block grid, shared by every row and read off the
        # PHYSICAL width so it costs no host sync.  It is an upper bound, not
        # a per-row count: row b closes (total - left_padding[b]) // r blocks.
        # merge() and filter() do leave min(left_padding) at 0, but finalize()
        # need not -- the row holding the zero left padding and the row holding
        # the zero right padding can differ, so a right-padded continuation of
        # an already-left-padded batch (histories [10, 5], continuations
        # [1, 5]) lands on left_padding [4, 5].  Over-counting is contained:
        # a surplus block ends past every row's deepest query, so valid_blocks
        # rejects it for every row (see _pool_blocks_left_padded), and the
        # dense short-circuit below only declines more often.  The cost is
        # pooling and scoring up to min(left_padding) // r blocks nothing
        # reads.
        n_blocks = total // self.compress_ratio
        if n_blocks == 0:
            return QSASelection(
                kind="implicit_all",
                batch=batch,
                length=length,
                block_size=self.compress_ratio,
                causal_mask=causal_mask,
                left_padding=left_pad,
                offset=offset,
                physical_width=total,
                n_blocks=n_blocks,
            )
        if _QSA_DENSE_SHORTCIRCUIT and self._dense_by_construction(
            n_blocks, shared_topk
        ):
            if shared_topk is None and getattr(cache, "_mtp_share_topk", False):
                # An MTP cycle opening on a dense step must still hand the
                # later steps a set: top-k over all blocks IS every block, and
                # the mask only ever uses ``selected`` as a set, so arange is
                # the same selection the stock path would have stored.
                cache._mtp_shared_topk = mx.contiguous(
                    mx.broadcast_to(
                        mx.arange(n_blocks, dtype=mx.uint32), (batch, n_blocks)
                    )
                )
            return QSASelection(
                kind="implicit_all",
                batch=batch,
                length=length,
                block_size=self.compress_ratio,
                causal_mask=causal_mask,
                left_padding=left_pad,
                offset=offset,
                physical_width=total,
                n_blocks=n_blocks,
            )

        if left_pad is not None:
            q_pos = offset[:, None] + mx.arange(length)[None, :]
            # Negative for a row's left padding, which is not a key of that
            # row at all; the ``>= 0`` term at the bottom drops those columns.
            token_logical = mx.arange(total)[None, :] - left_pad[:, None]
        else:
            q_pos = mx.arange(offset, offset + length)[None, :]
            token_logical = mx.arange(total)[None, :]
        if shared_topk is None:
            q = self.q_layernorm(
                q.reshape(batch, length, self.n_heads, self.head_dim)
            )
            q = _apply_rope_positions(
                q, q_pos[..., None], self.rotary_dim, self.rope_theta
            )
        starts = mx.arange(n_blocks) * self.compress_ratio
        valid_blocks = (
            (starts + self.compress_ratio - 1)[None, None, :] <= q_pos[..., None]
        )
        if shared_topk is None:
            pooled = self._pooled_keys(
                all_raw, n_blocks, starts, cache, length, left_pad
            )
            scores = mx.einsum(
                "blhd,bnd->blnh", q.astype(mx.float32), pooled.astype(mx.float32)
            )
            scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(self.head_dim)
            scores = mx.where(valid_blocks, scores, -mx.inf)
            k = min(self.block_topk, n_blocks)
            selected = mx.argpartition(scores, kth=n_blocks - k, axis=-1)[..., -k:]
            if cache is not None and getattr(cache, "_mtp_share_topk", False):
                shared = (
                    cache.last_valid_query(selected)
                    if isinstance(cache, BatchQSAKVCache)
                    else selected[:, -1]
                )
                cache._mtp_shared_topk = mx.contiguous(shared)
        else:
            selected = mx.broadcast_to(
                shared_topk[:, None, :], (batch, length, shared_topk.shape[-1])
            )
        # Mask assembly lives in ``QSASelection.dense_mask``.  Selection and
        # every cache side effect stay here, so the caller sees the same
        # ledger, pooled-key and shared-top-k state it always did.
        return QSASelection(
            kind="explicit",
            batch=batch,
            length=length,
            block_size=self.compress_ratio,
            raw_block_ids=selected,
            valid_blocks=valid_blocks,
            q_positions=q_pos,
            token_positions=token_logical,
            causal_mask=causal_mask,
            left_padding=left_pad,
            offset=offset,
            physical_width=total,
            n_blocks=n_blocks,
            scatter_chosen=_QSA_SCATTER_CHOSEN,
        )


class Attention(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_kv_heads = args.num_key_value_heads
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.q_proj = nn.Linear(args.hidden_size, self.num_heads * self.head_dim * 2, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, args.hidden_size, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args)
        self.rope = initialize_rope(
            int(args.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        # Lazy MLX_QWEN4_QSA_FUSED_PROJ table, identity-keyed by the source
        # weight arrays (invalidated by load_weights/update); in __dict__ so
        # it never reaches parameters()/state.
        object.__setattr__(self, "_qsa_fused_cache", None)
        # Static: whether this attention geometry fits the NAX block-sparse
        # kernel's tile (head_dim, gqa, block size).  Availability of the
        # kernel itself is a device probe, checked per forward only when armed.
        self._nax_layout_ok = block_sparse_layout_supported(
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            block_size=args.indexer_compress_ratio,
        )

    def _fused_projection_table(self):
        modules = (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.indexer.index_qk_proj,
        )
        key = tuple(part for m in modules for part in _proj_identity(m))
        cached = self._qsa_fused_cache
        if cached is not None and all(
            new is old for new, old in zip(key, cached[0])
        ):
            return cached[1]
        signatures = [_proj_signature(m) for m in modules]
        table = None
        if signatures[0] is not None and all(
            s == signatures[0] for s in signatures
        ):
            parts = [_proj_table(m) for m in modules]
            # This IS a runtime second copy, so size it first.
            check_materialization_budget(
                sum(table_bytes(part) for part in parts), "QSA fused projection"
            )
            table = _concat_tables(parts, axis=0)
        object.__setattr__(self, "_qsa_fused_cache", (key, table))
        return table

    def __call__(self, x: mx.array, mask: mx.array, cache: Optional[QSAKVCache]):
        batch, length, _ = x.shape
        fused_index_qk = None
        if _QSA_FUSED_PROJ and not self.training:
            table = self._fused_projection_table()
            if table is not None:
                width_q = self.num_heads * self.head_dim * 2
                width_kv = self.num_kv_heads * self.head_dim
                qg, k_flat, v_flat, fused_index_qk = mx.split(
                    _table_matmul(table, x),
                    [width_q, width_q + width_kv, width_q + 2 * width_kv],
                    axis=-1,
                )
        selection = self.indexer(x, mask, cache, projected_qk=fused_index_qk)
        # Route the sparse path through the NAX block-sparse kernel when armed.
        # Only an ``explicit`` (sparse) selection on a multi-token query with a
        # supported layout on a NAX-capable device qualifies; everything else
        # keeps the dense masked-SDPA path and its mask, bit-identical.
        use_nax = (
            _QSA_NAX_KERNEL
            # The kernel is an MLX CustomKernel with no VJP, so a backward pass
            # raises "Primitive::vjp Not implemented". Never route training
            # through it (mirrors the _QSA_FUSED_PROJ guard); inference only.
            and not self.training
            and selection.kind == "explicit"
            and length >= _QSA_NAX_MIN_QUERY
            and self._nax_layout_ok
            and nax_kernel_available()
        )
        # Do NOT build the dense mask when the kernel is engaged: not
        # materializing that [B, 1, L, T] array is the point.
        sparse_mask = None if use_nax else selection.dense_mask()
        if fused_index_qk is None:
            qg = self.q_proj(x)
            k_flat = self.k_proj(x)
            v_flat = self.v_proj(x)
        q, gate = mx.split(
            qg.reshape(batch, length, self.num_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(batch, length, -1)
        k = k_flat.reshape(batch, length, self.num_kv_heads, self.head_dim)
        v = v_flat.reshape(batch, length, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        if use_nax:
            ids, counts, n_sel, u_width, q_pos, left_pad, total = (
                compact_blocks_to_kernel_inputs(selection.compact_blocks())
            )
            out = nax_qsa_attention(
                q, k, v, ids, counts, n_sel, q_pos, left_pad,
                scale=self.scale, u_width=u_width, total=total,
                n_kv_heads=self.num_kv_heads,
            ).astype(q.dtype)
        else:
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=sparse_mask
            )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out * mx.sigmoid(gate))


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        self.linear_attn = GatedDeltaNet(args) if self.is_linear else None
        self.self_attn = None if self.is_linear else Attention(args)
        self.mlp = SparseMoeBlock(args)
        ple_index = args.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in args.ple_layer_ids else None
        self.ple = PLELayer(args, layer_idx, ple_index) if ple_index is not None else None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(self, x, input_ids, mask=None, cache=None, ssm_mask=None):
        if self.ple is not None:
            x = x + self.ple(x, input_ids, cache, ssm_mask)
        mixed, residual, inject = self.attn_hyper_connection(x)
        if self.is_linear:
            branch = self.linear_attn(mixed, ssm_mask, cache)
        else:
            branch = self.self_attn(mixed, mask, cache)
        x = residual + (branch[..., None, :] * inject[..., None]).reshape(*residual.shape)
        mixed, residual, inject = self.mlp_hyper_connection(x)
        branch = self.mlp(mixed)
        return residual + (branch[..., None, :] * inject[..., None]).reshape(*residual.shape)


class Qwen4ExpTextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ssm_idx = next((i for i, layer in enumerate(self.layers) if layer.is_linear), None)
        self.fa_idx = next((i for i, layer in enumerate(self.layers) if not layer.is_linear), None)

    def __call__(self, inputs, cache=None, input_embeddings=None, return_hyper=False):
        if (
            _SHAPE_STABLE_SHORT_FORWARD
            and cache is not None
            and inputs.shape[1] > 1
        ):
            outputs = [
                self(
                    inputs[:, index : index + 1],
                    cache,
                    None
                    if input_embeddings is None
                    else input_embeddings[:, index : index + 1],
                    return_hyper,
                )
                for index in range(inputs.shape[1])
            ]
            if return_hyper:
                return tuple(
                    mx.concatenate([output[field] for output in outputs], axis=1)
                    for field in range(2)
                )
            return mx.concatenate(outputs, axis=1)

        hidden = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        hidden = mx.tile(hidden, (1, 1, self.args.hc_count))
        cache = [None] * len(self.layers) if cache is None else cache
        fa_mask = None
        if self.fa_idx is not None:
            fa_cache = cache[self.fa_idx]
            fa_mask = create_attention_mask(hidden, fa_cache, return_array=True)
            if fa_mask is not None and fa_mask.ndim == 2:
                fa_mask = fa_mask[None, None, :, :]
        ssm_mask = create_ssm_mask(hidden, cache[self.ssm_idx]) if self.ssm_idx is not None else None
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(hidden, inputs, fa_mask, layer_cache, ssm_mask)
        mixed = self.hyper_connection_mixer(hidden)
        return (mixed, hidden) if return_hyper else mixed


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpTextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.pipeline_layers

    def __call__(self, inputs, cache=None, input_embeddings=None):
        hidden = self.model(inputs, cache, input_embeddings)
        return self.model.embed_tokens.as_linear(hidden) if self.args.tie_word_embeddings else self.lm_head(hidden)

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                cache_type = Qwen4ArraysCache if layer.ple is not None else ArraysCache
                caches.append(cache_type(size=4 if layer.ple is not None else 2))
            else:
                caches.append(QSAKVCache())
        return caches

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        override = os.environ.get("MLX_QWEN4_NORM_CONVENTION")
        if override not in (None, "", "raw", "converted"):
            raise ValueError(
                "MLX_QWEN4_NORM_CONVENTION must be 'raw' or 'converted', "
                f"got {override!r}"
            )
        raw = any("conv1d.weight" in key and value.shape[-1] != 1 for key, value in weights.items())
        if override:
            raw = override == "raw"
        zero_centered = (
            ".hc_norm.weight",
            ".norm_key.weight",
            ".norm_query.weight",
            ".norm_conv.weight",
            ".q_layernorm.weight",
            ".k_layernorm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            "hyper_connection_mixer.hc_norm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        )
        for key, value in list(weights.items()):
            if "conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
            if raw and any(key.endswith(suffix) for suffix in zero_centered):
                weights[key] = value + 1.0
        if not override:
            self._check_norm_convention(weights, zero_centered, raw)
        return weights

    @staticmethod
    def _check_norm_convention(weights, zero_centered, raw):
        """Refuse the exact +-1 signature of a wrong convention guess.

        A wrong zero-vs-ones-centered guess loads cleanly and produces
        deterministic garbage (mlx-vlm #2041/#2045 class). The check is
        comparative, not a legitimacy window on learned gains: it refuses
        only when the opposite convention fits gains-near-1 decisively
        better across the per-family aggregates.
        """
        families = {}
        for key, value in weights.items():
            for suffix in zero_centered:
                if key.endswith(suffix):
                    families.setdefault(suffix, []).append(
                        (key, value.astype(mx.float32).mean().item())
                    )
                    break
        if not families:
            return
        # Post-sanitize means; the alternative convention differs by -1
        # (raw applied +1 that converted would not) or +1 (the reverse).
        shift = -1.0 if raw else 1.0
        total = chosen = alternative = 0.0
        rows = []
        for suffix, entries in families.items():
            count = len(entries)
            mean = sum(value for _, value in entries) / count
            chosen += count * abs(mean - 1.0)
            alternative += count * abs(mean + shift - 1.0)
            total += count
            rows.append((abs(mean - 1.0), suffix, count, mean))
        if alternative / total + 0.25 >= chosen / total:
            return
        rows.sort(reverse=True)
        worst = ", ".join(
            f"{suffix} (n={count}, mean {mean:.3f})"
            for _, suffix, count, mean in rows[:4]
        )
        applied, other = "raw (+1 offset)", "converted (no offset)"
        if not raw:
            applied, other = other, applied
        raise ValueError(
            "norm convention mismatch: the conv1d layout proxy chose the "
            f"{applied} convention, but the {other} convention fits the "
            f"stored RMSNorm gains decisively better (mean |gain-1| "
            f"{chosen / total:.3f} vs {alternative / total:.3f}). Worst "
            f"families: {worst}. If the proxy misreads this checkpoint, set "
            "MLX_QWEN4_NORM_CONVENTION=raw|converted to force the "
            "convention and skip this check."
        )

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Quantize each 160-wide PLE shard independently.  Group 32 is
            # required because 160 is not divisible by the global group 64;
            # the 128 modules remain separately file-backed at lookup time.
            if ".ple_embedding.ngram_embedding.shard_" in path:
                return {"group_size": 32, "bits": 4, "mode": "affine"}
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate


class Qwen4ExpMTP(nn.Module):
    """Depth-1 residual-linear-shared MTP head with HC scheme-A state."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hc_count * args.hidden_size
        self.pre_fc_norm_embedding = GroupRMSNorm(
            args.hidden_size, None, args.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GroupRMSNorm(
            hc_hidden, args.hidden_size, args.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(
            args.hidden_size, args.hidden_size, bias=False
        )
        self.fc_hidden = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        mtp_args = replace(
            args,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            ple_layer_ids=[],
        )
        self.layers = [DecoderLayer(mtp_args, 0)]
        self.hyper_connection_mixer = GatedResidual(mtp_args, use_combine=False)

    def fuse(self, embeddings: mx.array, hidden: mx.array) -> mx.array:
        embeddings = self.fc_embedding(self.pre_fc_norm_embedding(embeddings))
        hidden = self.pre_fc_norm_hidden(hidden).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        hidden = self.fc_hidden(hidden)
        return (embeddings[..., None, :] + hidden).reshape(
            *hidden.shape[:-2], self.hc_count * self.hidden_size
        )


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    # Qwen4ArraysCache restores PLE token history + ShortConv and GDN
    # convolution + recurrence as one atomic record.
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = TextModelArgs.from_dict(args.text_config)
        self.language_model = TextModel(text_args)
        if text_args.mtp_num_hidden_layers > 0:
            self.mtp = Qwen4ExpMTP(text_args)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.language_model(inputs, cache, input_embeddings)

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def logits(self, hidden):
        return (
            self.language_model.model.embed_tokens.as_linear(hidden)
            if self.language_model.args.tie_word_embeddings
            else self.language_model.lm_head(hidden)
        )

    def mtp_backbone(self, inputs, cache=None):
        """Return LM-head and scheme-A HC hiddens from one trunk forward."""
        return self.language_model.model(inputs, cache, return_hyper=True)

    def prefill_prefetch_hook(self):
        """Return a chunk prefetcher when any PLE table is NVMe-backed.

        The callable takes ``(chunk_tokens, previous_tokens)`` as numpy or
        mx int arrays ([T] or [B, T]) and warms the sidecar rows the chunk
        will gather, asynchronously. ``None`` when every table is resident.
        """
        embeddings = [
            layer.ple.ple_embedding
            for layer in self.language_model.model.layers
            if layer.ple is not None and layer.ple.ple_embedding.file_backed
        ]
        if not embeddings:
            return None

        def hook(chunk_tokens, previous):
            for embedding in embeddings:
                embedding.prefetch_prompt_chunk(chunk_tokens, previous)

        hook.context_len = max(e.context_len for e in embeddings)
        return hook

    def make_mtp_cache(self, window_size: Optional[int] = None, sink_size: int = 4):
        if window_size is None:
            return [QSAKVCache() for _ in self.mtp.layers]
        return [SinkWindowKVCache(window_size, sink_size) for _ in self.mtp.layers]

    def mtp_end_cycle(self, mtp_cache):
        """Disarm QSA top-k sharing at the end of an MTP draft cycle.

        The paired hook for ``mtp_start_cycle``. Arming has to be undone by
        SOMETHING even when the cycle is abandoned -- an abort, an exception,
        a stream that disconnects mid-draft -- because a surviving armed flag
        makes the next forward reuse a stale index set and build a silently
        wrong mask, with no desync check on that branch to catch it. Call it
        AFTER the drafted span is rewound; called before, it reports the
        un-ledgered KV the cycle left behind instead of hiding it.
        """
        for cache in mtp_cache:
            if isinstance(cache, BatchQSAKVCache):
                cache.release_qsa_cycle("Model.mtp_end_cycle")
            elif isinstance(cache, QSAKVCache):
                cache.release_qsa_cycle("Model.mtp_end_cycle")

    def mtp_start_cycle(self, mtp_cache, share_qsa_indices: bool = False):
        """Reset optional QSA top-k sharing at an MTP draft-cycle boundary.

        Sharing is per lane under a batched head cache: the stored index set
        is ``[B, k]`` and each row reuses its own blocks.

        Ending the previous cycle first makes arming idempotent and self
        healing: a cycle that was abandoned without reaching a rewind cannot
        leak its index set into this one, and its un-ledgered KV is reported
        here rather than several forwards later.
        """
        self.mtp_end_cycle(mtp_cache)
        for cache in mtp_cache:
            if isinstance(cache, (QSAKVCache, BatchQSAKVCache)):
                cache._mtp_share_topk = bool(share_qsa_indices)
                cache._mtp_shared_topk = None

    def mtp_step(self, hidden, tokens, mtp_cache):
        embeddings = self.language_model.model.embed_tokens(tokens)
        multi = self.mtp.fuse(embeddings, hidden)
        cache = mtp_cache[0]
        mask = create_attention_mask(multi, cache, return_array=True)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None, :, :]
        multi = self.mtp.layers[0](multi, tokens, mask, cache, None)
        sample = self.mtp.hyper_connection_mixer(multi)
        return self.logits(sample), multi

    def sanitize(self, weights):
        has_mtp_weights = any(
            key.startswith("mtp.")
            or key.startswith("model.mtp.")
            or key.startswith("model.language_model.mtp.")
            for key in weights
        )
        if not (has_mtp_weights and getattr(self, "mtp", None) is not None):
            if getattr(self, "mtp", None) is not None:
                self.mtp = None
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.visual") or key.startswith("vision_tower"):
                continue
            if key.startswith("model.language_model.mtp."):
                key = key.replace("model.language_model.mtp.", "mtp.", 1)
            elif key.startswith("model.mtp."):
                key = key.removeprefix("model.")
            elif key.startswith("mtp."):
                if getattr(self, "mtp", None) is None:
                    continue
            elif key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value

        mlp_prefixes = [
            f"language_model.model.layers.{layer_idx}.mlp"
            for layer_idx in range(self.language_model.args.num_hidden_layers)
        ]
        if getattr(self, "mtp", None) is not None:
            mlp_prefixes.extend(
                f"mtp.layers.{layer_idx}.mlp"
                for layer_idx in range(self.language_model.args.mtp_num_hidden_layers)
            )
        for prefix in mlp_prefixes:
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key not in sanitized:
                continue
            gate_up = sanitized.pop(gate_up_key)
            if qwen3_next._MOE_FUSED_GATE_UP:
                # The lever consumes the shipped layout: never split it.
                sanitized[f"{prefix}.switch_mlp.gate_up_proj.weight"] = gate_up
            else:
                midpoint = gate_up.shape[-2] // 2
                sanitized[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :midpoint, :
                ]
                sanitized[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., midpoint:, :
                ]
            sanitized[f"{prefix}.switch_mlp.down_proj.weight"] = sanitized.pop(
                f"{prefix}.experts.down_proj"
            )
        # Load-time MoE lever transforms. They consume the file-backed
        # checkpoint arrays, so the transformed tensor is the only resident
        # copy (a runtime re-fusion cost 45 GB and was OOM-killed).
        transform_moe_weights(
            sanitized,
            mlp_prefixes,
            fuse_gate_up=qwen3_next._MOE_FUSED_GATE_UP,
            fold_shared=qwen3_next._MOE_SHARED_IN_GATHER,
        )
        return self.language_model.sanitize(sanitized)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate
