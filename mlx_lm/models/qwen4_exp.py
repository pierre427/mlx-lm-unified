# Copyright © 2026 Apple Inc.
#
# Qwen4-Exp / Qwen3.8-Flash-Next text-core support.  The architecture was
# derived from the Apache-2.0 Transformers Qwen4Exp implementation and the
# release checkpoint at Qwen/Qwen3.8-Flash-Next.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask, create_ssm_mask, scaled_dot_product_attention
from .cache import ArraysCache, BatchKVCache, KVCache, dynamic_roll
from .pipeline import PipelineMixin
from .qwen3_5 import GatedDeltaNet as Qwen35GatedDeltaNet
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock
from .rope_utils import initialize_rope


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

    def __call__(self, indices: mx.array) -> mx.array:
        mx.eval(indices)
        shape = indices.shape
        flat = np.asarray(indices, dtype=np.int64).reshape(-1)
        output = None
        shard_ids = flat // self.rows_per_shard
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
        self.layer_multipliers = mx.array(
            _build_layer_multipliers(
                args.vocab_size, args.ngram_size, ple_layer_index, args.seed
            ),
            dtype=mx.int64,
        )
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self.ngram_embedding = ShardedEmbedding(
            padded, embedding_dim // self.ngram_heads, args.split_ngram_parts
        )

    def ngram_ids(self, input_ids: mx.array, cache: Optional[ArraysCache] = None):
        mx.eval(input_ids, self.layer_multipliers, self.ngram_heads_vocab_sizes, self.ngram_heads_offsets)
        tokens = np.asarray(input_ids, dtype=np.int64)
        batch, seq_len = tokens.shape
        if cache is not None and cache[3] is not None:
            previous = np.asarray(cache[3], dtype=np.int64)
        else:
            previous = np.full((batch, self.context_len), self.eos_token_id, dtype=np.int64)
        history = np.concatenate([previous, tokens], axis=-1)
        if cache is not None:
            cache[3] = mx.array(history[:, -self.context_len :], dtype=mx.int64)

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

        multipliers = np.asarray(self.layer_multipliers, dtype=np.int64)
        sizes = np.asarray(self.ngram_heads_vocab_sizes, dtype=np.int64)
        offsets = np.asarray(self.ngram_heads_offsets, dtype=np.int64)
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
        return mx.array(np.concatenate(blocks, axis=-1)[:, -seq_len:], dtype=mx.int64)

    def __call__(self, input_ids: mx.array, cache: Optional[ArraysCache] = None):
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
        return gated + self._short_conv(normed, cache)


def _apply_rope_positions(x: mx.array, positions: mx.array, dims: int, base: float):
    """Transformers-compatible non-traditional partial RoPE at arbitrary positions."""
    if dims == 0:
        return x
    freqs = mx.exp(-math.log(base) * mx.arange(0, dims, 2) / dims)
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

    def update_index_keys(self, keys: mx.array):
        self.index_keys = keys if self.index_keys is None else mx.concatenate([self.index_keys[:, : self.offset], keys], axis=1)
        return self.index_keys

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

    def __call__(self, hidden: mx.array, causal_mask: mx.array, cache: QSAKVCache):
        batch, length, _ = hidden.shape
        qk = self.index_qk_proj(hidden)
        q, raw = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
        q = self.q_layernorm(q.reshape(batch, length, self.n_heads, self.head_dim))
        raw = raw.reshape(batch, length, self.head_dim)
        offset = 0 if cache is None else cache.offset
        all_raw = raw if cache is None else cache.update_index_keys(raw)
        total = all_raw.shape[1]
        if isinstance(offset, mx.array):
            q_pos = offset[:, None] + mx.arange(length)[None, :]
        else:
            q_pos = mx.arange(offset, offset + length)[None, :]
        q = _apply_rope_positions(q, q_pos[..., None], self.rotary_dim, self.rope_theta)

        n_blocks = total // self.compress_ratio
        if n_blocks == 0:
            return causal_mask
        pooled = all_raw[:, : n_blocks * self.compress_ratio].reshape(
            batch, n_blocks, self.compress_ratio, self.head_dim
        ).astype(mx.float32).mean(axis=2).astype(all_raw.dtype)
        pooled = self.k_layernorm(pooled)
        starts = mx.arange(n_blocks) * self.compress_ratio
        pooled = _apply_rope_positions(
            pooled, starts[None, :], self.rotary_dim, self.rope_theta
        )
        scores = mx.einsum("blhd,bnd->blnh", q.astype(mx.float32), pooled.astype(mx.float32))
        scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(self.head_dim)
        valid_blocks = (
            (starts + self.compress_ratio - 1)[None, None, :] <= q_pos[..., None]
        )
        scores = mx.where(valid_blocks, scores, -mx.inf)
        k = min(self.block_topk, n_blocks)
        selected = mx.argpartition(scores, kth=n_blocks - k, axis=-1)[..., -k:]
        block_ids = mx.arange(n_blocks)
        chosen = mx.any(selected[..., None] == block_ids[None, None, None, :], axis=-2)
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
        return causal_mask & sparse[:, None, :, :]


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
            if fa_mask.ndim == 2:
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
            caches.append(ArraysCache(size=4 if layer.ple is not None else 2) if layer.is_linear else QSAKVCache())
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
    # PLE has two additional recurrent states. Exact speculative rollback is
    # deliberately disabled until all four states are restored as one unit.
    supports_speculative_rollback = False

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

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

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.visual") or key.startswith("vision_tower"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif key.startswith("mtp."):
                # MTP weights are retained in the release artifact but are not
                # loaded until exact four-state speculative rollback lands.
                continue
            elif not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value

        for layer_idx in range(self.language_model.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{layer_idx}.mlp"
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
