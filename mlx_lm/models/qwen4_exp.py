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
from .qwen3_5 import GatedDeltaNet as Qwen35GatedDeltaNet
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock
from .qwen3_next import _env_flag
from .rope_utils import initialize_rope


# Opt-in micro-levers, each read once at import.  Off keeps the stock path.
_RMSNORM_FAST = _env_flag("MLX_QWEN4_RMSNORM_FAST")
_QSA_POOLED_KEY_CACHE = _env_flag("MLX_QWEN4_QSA_POOLED_KEY_CACHE")
_QSA_SCATTER_CHOSEN = _env_flag("MLX_QWEN4_QSA_SCATTER_CHOSEN")
_PLE_VECTOR_SHIFT = _env_flag("MLX_QWEN4_PLE_VECTOR_SHIFT")
_PLE_GATHER_CONCAT = _env_flag("MLX_QWEN4_PLE_GATHER_CONCAT")


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


class Qwen4ArraysCache(ArraysCache):
    """Four-state PLE+GDN cache with one atomic speculative rollback."""

    def start_speculation(self, rollback_window=None):
        self._ple_rollback = None
        super().start_speculation(rollback_window)

    def stop_speculation(self):
        self._ple_rollback = None
        super().stop_speculation()

    def stage_ple_rollback(self, num_tokens, fn, snapshot):
        if self._ple_rollback is not None:
            raise RuntimeError("Qwen4 PLE rollback was staged twice")
        self._ple_rollback = (num_tokens, fn, snapshot)

    def record_rollback(self, num_tokens, fn, snapshot):
        staged = self._ple_rollback
        self._ple_rollback = None
        if staged is None:
            return super().record_rollback(num_tokens, fn, snapshot)
        ple_tokens, ple_fn, ple_snapshot = staged
        if ple_tokens != num_tokens:
            raise RuntimeError(
                "Qwen4 PLE/GDN rollback span mismatch: "
                f"{ple_tokens} != {num_tokens}"
            )

        def combined(m):
            return list(fn(m)) + list(ple_fn(m))

        return super().record_rollback(
            num_tokens, combined, list(snapshot) + list(ple_snapshot)
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
        self, input_ids: mx.array, cache: Optional[ArraysCache] = None
    ) -> np.ndarray:
        mx.eval(input_ids)
        tokens = np.asarray(input_ids, dtype=np.int64)
        batch, seq_len = tokens.shape
        if cache is not None and cache[3] is not None:
            previous = np.asarray(cache[3], dtype=np.int64)
        else:
            previous = np.full((batch, self.context_len), self.eos_token_id, dtype=np.int64)
        history = np.concatenate([previous, tokens], axis=-1)
        if cache is not None:
            cache[3] = mx.array(history[:, -self.context_len :], dtype=mx.int64)

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
        self, input_ids: mx.array, cache: Optional[ArraysCache] = None
    ) -> mx.array:
        if self.ngram_size != 3 or not mx.metal.is_available():
            return mx.array(self._ngram_ids_numpy(input_ids, cache), dtype=mx.int64)

        batch, seq_len = input_ids.shape
        if cache is not None and cache[3] is not None:
            previous = cache[3]
        else:
            previous = mx.full(
                (batch, self.context_len), self.eos_token_id, dtype=mx.int64
            )
        history = mx.concatenate([previous, input_ids.astype(mx.int64)], axis=-1)
        if cache is not None:
            cache[3] = mx.contiguous(history[:, -self.context_len :])

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

    def ngram_ids(self, input_ids: mx.array, cache: Optional[ArraysCache] = None):
        if self.hash_backend == "metal" or (
            self.hash_backend == "metal_prefill"
            and input_ids.shape[1] >= self.metal_hash_min_tokens
        ):
            return self._ngram_ids_metal(input_ids, cache)
        return mx.array(self._ngram_ids_numpy(input_ids, cache), dtype=mx.int64)

    def __call__(self, input_ids: mx.array, cache: Optional[ArraysCache] = None):
        if self.hash_backend == "routed_cpu" or (
            self.hash_backend == "metal_prefill"
            and input_ids.shape[1] < self.metal_hash_min_tokens
        ):
            ids = self._ngram_ids_numpy(input_ids, cache)
            return self.ngram_embedding.lookup_numpy(ids).reshape(
                *input_ids.shape, -1
            )
        return self.ngram_embedding(self.ngram_ids(input_ids, cache)).reshape(
            *input_ids.shape, -1
        )


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

    def _short_conv(self, x: mx.array, cache: Optional[ArraysCache]):
        state = cache[2] if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], self.short_conv_state_len, x.shape[-1]), x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.short_conv_state_len :, :])
        return nn.silu(self.conv1d(conv_input))[:, -x.shape[1] :, :]

    def __call__(self, hidden: mx.array, input_ids: mx.array, cache=None, mask=None):
        previous_conv = cache[2] if cache is not None else None
        previous_tokens = cache[3] if cache is not None else None
        embeddings = self.ple_embedding(input_ids, cache)
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
        conv = self._short_conv(normed, cache)
        if (
            isinstance(cache, Qwen4ArraysCache)
            and cache.speculating
            and mask is None
            and cache.lengths is None
            and cache.left_padding is None
        ):
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
            token_history = mx.concatenate(
                [token_base, input_ids.astype(mx.int64)], axis=1
            )

            def _ple_rollback(
                m, ci=conv_input, th=token_history, sl=state_len, cl=context_len
            ):
                return [
                    mx.contiguous(ci[:, m : m + sl, :]),
                    mx.contiguous(th[:, m : m + cl]),
                ]

            cache.stage_ple_rollback(
                input_ids.shape[1],
                _ple_rollback,
                [previous_conv, previous_tokens],
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


class BatchQSAKVCache(BatchKVCache):
    """Batched QSA cache retaining raw indexer keys beside attention KV."""

    def __init__(self, left_padding: List[int], attention_backend=None):
        super().__init__(left_padding, attention_backend=attention_backend)
        self.index_keys = None

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

    @property
    def nbytes(self):
        return super().nbytes + (
            0 if self.index_keys is None else self.index_keys.nbytes
        )

    def finalize(self):
        padding = self._right_padding
        if padding is not None and self.index_keys is not None:
            self.index_keys = dynamic_roll(self.index_keys, padding, axis=1)
        super().finalize()

    def filter(self, batch_indices):
        min_left_pad = self.left_padding[batch_indices].min().item()
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if min_left_pad > 0:
                self.index_keys = self.index_keys[:, min_left_pad:]
        super().filter(batch_indices)

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

    def extract(self, idx):
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

    def __init__(self):
        super().__init__()
        self.index_keys = None
        # Ephemeral MTP-cycle state.  Step zero computes QSA top-k normally;
        # later chained draft steps may reuse those block indices and skip the
        # index projection.  This is deliberately absent from ``state``: a
        # cache snapshot is a sequence snapshot, not an in-flight draft cycle.
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        # Ephemeral pooled+layernormed+roped block keys, derived from
        # ``index_keys`` (MLX_QWEN4_QSA_POOLED_KEY_CACHE).  Also absent from
        # ``state``: a restore simply recomputes them.
        self._qsa_pooled_keys = None
        self._qsa_pooled_ratio = None

    def update_index_keys(self, keys: mx.array):
        self.index_keys = keys if self.index_keys is None else mx.concatenate([self.index_keys[:, : self.offset], keys], axis=1)
        return self.index_keys

    def trim(self, n):
        n = super().trim(n)
        # A rewind ends any MTP draft cycle.  A stale shared top-k would make
        # the next uncycled ``mtp_step`` skip its raw-key append and desync
        # ``index_keys`` from the KV offset; ``mtp_start_cycle`` re-arms it.
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        if self.index_keys is not None:
            self.index_keys = mx.contiguous(self.index_keys[:, : self.offset])
        if self._qsa_pooled_keys is not None:
            # A block mean is a closed window over ``ratio`` tokens, so every
            # block fully inside the trimmed offset stays exact.
            keep = self.offset // self._qsa_pooled_ratio
            if keep == 0:
                self._qsa_pooled_keys = None
            elif keep < self._qsa_pooled_keys.shape[1]:
                self._qsa_pooled_keys = mx.contiguous(
                    self._qsa_pooled_keys[:, :keep]
                )
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

    def _pooled_keys(self, all_raw, n_blocks, starts, cache, length) -> mx.array:
        ratio = self.compress_ratio
        if not (_QSA_POOLED_KEY_CACHE and type(cache) is QSAKVCache):
            return self._pool_blocks(all_raw[:, : n_blocks * ratio], starts)
        if all_raw.shape[1] != cache.offset + length:
            # A raw-key ledger out of step with the KV offset means block
            # positions no longer match token positions; fail loudly instead
            # of pooling from a shifted history.
            raise RuntimeError(
                "QSA index_keys desync: "
                f"{all_raw.shape[1]} raw keys != offset {cache.offset} "
                f"+ {length} new"
            )
        cached = cache._qsa_pooled_keys
        count = 0 if cached is None else cached.shape[1]
        if cached is not None and (
            count > n_blocks
            or cache._qsa_pooled_ratio != ratio
            or cached.shape[0] != all_raw.shape[0]
        ):
            cached, count = None, 0
        if count == n_blocks:
            pooled = cached
        else:
            # A block is final once its last token is written; only blocks
            # closed since the previous call need computing.
            new = self._pool_blocks(
                all_raw[:, count * ratio : n_blocks * ratio], starts[count:]
            )
            pooled = new if cached is None else mx.concatenate([cached, new], axis=1)
        cache._qsa_pooled_keys = pooled
        cache._qsa_pooled_ratio = ratio
        return pooled

    def __call__(self, hidden: mx.array, causal_mask: mx.array, cache: QSAKVCache):
        batch, length, _ = hidden.shape
        if isinstance(cache, SinkWindowKVCache):
            # Windowed MTP deliberately replaces the draft head's global QSA
            # lookup with dense sink+recent attention. The full target keeps
            # native QSA and remains the sole verifier.
            mask = cache.make_mask(length, return_array=True)
            return None if mask is None else mask[None, None, :, :]
        offset = 0 if cache is None else cache.offset
        if isinstance(offset, mx.array):
            q_pos = offset[:, None] + mx.arange(length)[None, :]
        else:
            q_pos = mx.arange(offset, offset + length)[None, :]

        shared_topk = (
            getattr(cache, "_mtp_shared_topk", None) if cache is not None else None
        )
        if shared_topk is None:
            qk = self.index_qk_proj(hidden)
            q, raw = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
            q = self.q_layernorm(
                q.reshape(batch, length, self.n_heads, self.head_dim)
            )
            raw = raw.reshape(batch, length, self.head_dim)
            all_raw = raw if cache is None else cache.update_index_keys(raw)
            total = all_raw.shape[1]
            q = _apply_rope_positions(
                q, q_pos[..., None], self.rotary_dim, self.rope_theta
            )
        else:
            # The current draft token is transient and will be rewound before
            # any accepted span is teacher-forced next cycle.  Skipping its raw
            # index key is therefore safe and is what removes the indexer work.
            total = (
                int(offset.max().item()) + length
                if isinstance(offset, mx.array)
                else offset + length
            )

        n_blocks = total // self.compress_ratio
        if n_blocks == 0:
            return causal_mask
        starts = mx.arange(n_blocks) * self.compress_ratio
        valid_blocks = (
            (starts + self.compress_ratio - 1)[None, None, :] <= q_pos[..., None]
        )
        if shared_topk is None:
            pooled = self._pooled_keys(all_raw, n_blocks, starts, cache, length)
            scores = mx.einsum(
                "blhd,bnd->blnh", q.astype(mx.float32), pooled.astype(mx.float32)
            )
            scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(self.head_dim)
            scores = mx.where(valid_blocks, scores, -mx.inf)
            k = min(self.block_topk, n_blocks)
            selected = mx.argpartition(scores, kth=n_blocks - k, axis=-1)[..., -k:]
            if cache is not None and getattr(cache, "_mtp_share_topk", False):
                cache._mtp_shared_topk = mx.contiguous(selected[:, -1])
        else:
            selected = mx.broadcast_to(
                shared_topk[:, None, :], (batch, length, shared_topk.shape[-1])
            )
        if _QSA_SCATTER_CHOSEN:
            # ``argpartition`` output has no duplicate indices, so a scatter
            # of ones is equivalent to the one-hot broadcast reduction.
            chosen = mx.put_along_axis(
                mx.zeros((batch, length, n_blocks), dtype=mx.bool_),
                selected,
                mx.array(True),
                axis=-1,
            )
        else:
            block_ids = mx.arange(n_blocks)
            chosen = mx.any(
                selected[..., None] == block_ids[None, None, None, :], axis=-2
            )
        chosen = chosen & valid_blocks
        token_pos = mx.arange(total)
        token_block = mx.minimum(token_pos // self.compress_ratio, n_blocks - 1)
        selected_tokens = mx.take_along_axis(
            chosen, mx.broadcast_to(token_block[None, None, :], (batch, length, total)), axis=-1
        )
        complete = ((q_pos + 1) // self.compress_ratio) * self.compress_ratio
        tail = (token_pos[None, None, :] >= complete[..., None]) & (
            token_pos[None, None, :] <= q_pos[..., None]
        )
        sparse = selected_tokens | tail
        sparse = sparse[:, None, :, :]
        # ``create_attention_mask`` deliberately returns ``None`` for a
        # single-token decode because every cached position is causal.  QSA
        # still needs its sparse selection mask in that case.
        return sparse if causal_mask is None else causal_mask & sparse


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

    def __call__(self, x: mx.array, mask: mx.array, cache: Optional[QSAKVCache]):
        batch, length, _ = x.shape
        sparse_mask = self.indexer(x, mask, cache)
        q, gate = mx.split(
            self.q_proj(x).reshape(batch, length, self.num_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(batch, length, -1)
        k = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=sparse_mask)
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
        raw = any("conv1d.weight" in key and value.shape[-1] != 1 for key, value in weights.items())
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
        # Convention sanity check, independent of the conv1d layout proxy
        # (mlx-vlm #2041/#2045 class: a wrong zero-vs-ones-centered guess
        # loads cleanly and produces deterministic garbage). Post-sanitize
        # gains must center near 1; a wrong guess shifts every family by
        # exactly +-1, so the aggregate lands near 0 or 2.
        norm_means = [
            weights[key].astype(mx.float32).mean()
            for key in weights
            if any(key.endswith(suffix) for suffix in zero_centered)
        ]
        if norm_means:
            center = mx.mean(mx.stack(norm_means)).item()
            if not 0.5 < center < 1.5:
                raise ValueError(
                    "norm convention mismatch: zero-centered norm families "
                    f"average {center:.3f} after sanitize, expected ~1. The "
                    "conv1d layout proxy disagrees with how this checkpoint "
                    "stores its RMSNorm gains; refusing to load garbage."
                )
        return weights

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

    def make_mtp_cache(self, window_size: Optional[int] = None, sink_size: int = 4):
        if window_size is None:
            return [QSAKVCache() for _ in self.mtp.layers]
        return [SinkWindowKVCache(window_size, sink_size) for _ in self.mtp.layers]

    def mtp_start_cycle(self, mtp_cache, share_qsa_indices: bool = False):
        """Reset optional QSA top-k sharing at an MTP draft-cycle boundary."""
        for cache in mtp_cache:
            if isinstance(cache, QSAKVCache):
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
        return self.language_model.sanitize(sanitized)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate
