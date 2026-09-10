# DFlash block-diffusion speculator for Qwen3-Coder-30B-A3B (z-lab/Qwen3-Coder-
# 30B-A3B-DFlash) — MLX port of the authoritative torch `dflash.py`.
#
# A lightweight BLOCK speculator: it has NO embedding / lm_head of its own and
# must be paired with the TARGET Qwen3-Coder-30B-A3B model, reusing the target's
# `embed_tokens` and `lm_head`. Per block it embeds [anchor, MASK*(block-1)] via
# the TARGET embedding; each of its 8 draft layers runs Qwen3 attention where the
# K/V are concat(proj(fused_target_hidden), proj(block)) — i.e. the target's fused
# aux hidden states are INJECTED as attention context. One parallel forward
# predicts the whole block; block position k predicts the token AT anchor_pos+k
# (position 0 reproduces the anchor, [1:] are the speculative tokens).
#
# Faithful to z-lab dflash.py (differs from the Laguna DFlash variant):
#   * plain Qwen3 attention — NO per-head softplus gate (Laguna has g_proj)
#   * fuse = hidden_norm(fc(concat(raw aux)))  — NO per-aux RMSNorm (Laguna has
#     aux_hidden_norms applied before concat)
#   * dims 32 heads / 4 KV / head_dim 128, hidden 2048, intermediate 6144, 8 layers
#   * mask_token_id 151669, target aux layers [1,12,23,34,45], rope_theta 1e7
#   * the anchor sits ONE PAST the fused context (blk_off = context length): the
#     context covers positions 0..C-1 and the block covers C..C+B-1.
#
# Weights (91 tensors) load strict=True. The torch tensor names already match this
# module tree, so `sanitize` is an identity pass. Reference driver + verify/accept
# loop: ../../dflash_spec_qwen3.py.
from dataclasses import dataclass, field
from typing import List

import mlx.core as mx
import mlx.nn as nn

# z-lab/Qwen3-Coder-30B-A3B-DFlash config constants
NH, NKV, HD = 32, 4, 128
HIDDEN, INTERMEDIATE = 2048, 6144
NUM_LAYERS = 8
NUM_AUX = 5
BLOCK_SIZE = 16
MASK_TOKEN_ID = 151669
THETA = 10000000.0
TARGET_LAYER_IDS = [1, 12, 23, 34, 45]


def _rms(x, w, eps=1e-6):
    x = x.astype(mx.float32)
    return (x * mx.rsqrt(mx.mean(x * x, -1, keepdims=True) + eps)) * w.astype(mx.float32)


def _rope(x, offset):
    return mx.fast.rope(x, HD, traditional=False, base=THETA, scale=1.0, offset=offset)


@dataclass
class ModelArgs:
    model_type: str = "dflash_qwen3"
    hidden_size: int = HIDDEN
    intermediate_size: int = INTERMEDIATE
    num_hidden_layers: int = NUM_LAYERS
    num_aux_hidden_states: int = NUM_AUX
    block_size: int = BLOCK_SIZE
    mask_token_id: int = MASK_TOKEN_ID
    rms_norm_eps: float = 1e-6
    target_layer_ids: List[int] = field(default_factory=lambda: list(TARGET_LAYER_IDS))

    @classmethod
    def from_dict(cls, d):
        dc = d.get("dflash_config", {})
        tids = dc.get("target_layer_ids", list(TARGET_LAYER_IDS))
        return cls(
            hidden_size=d.get("hidden_size", HIDDEN),
            intermediate_size=d.get("intermediate_size", INTERMEDIATE),
            num_hidden_layers=d.get("num_hidden_layers", NUM_LAYERS),
            num_aux_hidden_states=len(tids),
            block_size=d.get("block_size", BLOCK_SIZE),
            mask_token_id=dc.get("mask_token_id", MASK_TOKEN_ID),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            target_layer_ids=tids,
        )


class DFlashAttention(nn.Module):
    """q from the mask block; K/V = concat(proj(fused target_hidden), proj(block))."""

    def __init__(self, eps):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, NH * HD, bias=False)
        self.k_proj = nn.Linear(HIDDEN, NKV * HD, bias=False)
        self.v_proj = nn.Linear(HIDDEN, NKV * HD, bias=False)
        self.o_proj = nn.Linear(NH * HD, HIDDEN, bias=False)
        self.q_norm = nn.RMSNorm(HD, eps=eps)
        self.k_norm = nn.RMSNorm(HD, eps=eps)

    def __call__(self, h, target_hidden, blk_off, block_mask):
        C, B = target_hidden.shape[1], h.shape[1]
        q = _rms((h @ self.q_proj.weight.T).reshape(1, B, NH, HD), self.q_norm.weight)
        kc = (target_hidden @ self.k_proj.weight.T).reshape(1, C, NKV, HD)
        kn = (h @ self.k_proj.weight.T).reshape(1, B, NKV, HD)
        # k_norm is applied over the WHOLE concatenated K (context + block), per z-lab.
        k = _rms(mx.concatenate([kc, kn], axis=1), self.k_norm.weight)
        v = mx.concatenate(
            [
                (target_hidden @ self.v_proj.weight.T).reshape(1, C, NKV, HD),
                (h @ self.v_proj.weight.T).reshape(1, B, NKV, HD),
            ],
            axis=1,
        )
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        # rope: context positions 0..C-1 ; block positions blk_off..blk_off+B-1
        k = mx.concatenate([_rope(k[:, :, :C], 0), _rope(k[:, :, C:], blk_off)], axis=2)
        q = _rope(q, blk_off)
        k = mx.repeat(k, NH // NKV, axis=1)
        v = mx.repeat(v, NH // NKV, axis=1)
        s = (q @ k.transpose(0, 1, 3, 2)) * (HD ** -0.5) + block_mask
        o = (mx.softmax(s, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(1, B, NH * HD)
        return o @ self.o_proj.weight.T


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def __call__(self, x):
        return (nn.silu(x @ self.gate_proj.weight.T) * (x @ self.up_proj.weight.T)) @ self.down_proj.weight.T


class DFlashLayer(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.self_attn = DFlashAttention(eps)
        self.mlp = _MLP()
        self.input_layernorm = nn.RMSNorm(HIDDEN, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN, eps=eps)

    def __call__(self, x, target_hidden, blk_off, block_mask):
        x = x + self.self_attn(
            _rms(x, self.input_layernorm.weight), target_hidden, blk_off, block_mask
        )
        return x + self.mlp(_rms(x, self.post_attention_layernorm.weight))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        eps = args.rms_norm_eps
        self.layers = [DFlashLayer(eps) for _ in range(args.num_hidden_layers)]
        self.fc = nn.Linear(args.num_aux_hidden_states * HIDDEN, HIDDEN, bias=False)
        self.hidden_norm = nn.RMSNorm(HIDDEN, eps=eps)
        self.norm = nn.RMSNorm(HIDDEN, eps=eps)

    def fuse(self, aux: List[mx.array]) -> mx.array:
        """aux: list of raw target hidden states [1,C,H]. Returns fused [1,C,H].

        z-lab: target_hidden = hidden_norm(fc(concat(raw aux)))  (no per-aux norm)."""
        cat = mx.concatenate([a.astype(mx.float32) for a in aux], axis=-1)
        return _rms(cat @ self.fc.weight.T, self.hidden_norm.weight)

    def draft_block(self, target, target_hidden, anchor_tok, blk_off, block_size=None):
        """target: the paired Qwen3-Coder model (reuse embed_tokens + lm_head).
        target_hidden: [1,C,H] fused aux context (positions 0..C-1).
        anchor_tok: the known anchor token id (its absolute position == blk_off == C).
        Returns block logits [B, vocab] over the vocabulary; row k predicts pos blk_off+k
        (row 0 reproduces the anchor, rows [1:] are the drafted tokens)."""
        B = block_size or self.args.block_size
        ids = mx.array([[anchor_tok] + [self.args.mask_token_id] * (B - 1)])
        x = target.model.embed_tokens(ids).astype(mx.float32)  # reuse TARGET embed
        C = target_hidden.shape[1]
        # z-lab block diffusion: FULL bidirectional attention within the block and
        # over the injected context (spec_generate feeds attention_mask=None /
        # is_causal=False). A causal within-block mask collapses acceptance to ~0.
        block_mask = mx.zeros((1, 1, B, C + B))
        for layer in self.layers:
            x = layer(x, target_hidden, blk_off, block_mask)
        x = _rms(x, self.norm.weight)
        return target.lm_head(x.astype(mx.bfloat16))[0]  # [B, vocab]

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "dflash_qwen3 is a target-coupled DFlash speculator, not a standalone "
            "causal LM. Use dflash_spec_qwen3.py so it can reuse the target "
            "Qwen3-Coder-30B-A3B embedding and LM head."
        )

    def sanitize(self, weights):
        # z-lab torch tensor names already match this module tree (identity map).
        return weights
