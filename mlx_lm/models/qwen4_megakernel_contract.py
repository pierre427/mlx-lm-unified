"""Pure, import-safe contracts for the Qwen4 megakernel runtime.

This module deliberately has no MLX dependency.  The persistent kernel is
specialised to one checkpoint geometry; validating that geometry only after
packing or launching turns a configuration error into memory corruption.
"""

from __future__ import annotations

from typing import Any, Iterable


METAL_UINT32_MAX = (1 << 32) - 1
METAL_INT32_MAX = (1 << 31) - 1
SCORE_TILE_BLOCKS = 4096
MAX_QUERY_WIDTH = 3


MODEL_CONTRACT = {
    "model_type": "qwen4_exp_text",
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "hc_count": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_value_heads": 48,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "output_gate_type": "sigmoid",
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "ple_embed_dim": 2560,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "seed": 1234,
    "hc_lowrank": 320,
    "mtp_num_hidden_layers": 1,
}

EXPECTED_LAYER_TYPES = tuple(
    "linear_attention" if (index + 1) % 4 else "full_attention"
    for index in range(48)
)


def rounded_ledger_capacity(max_context: int, compress_ratio: int = 4) -> int:
    """Return the Metal-safe ledger width rounded to a compression block.

    The kernel stores positions and widths in uint32 control words but casts
    them to signed ``int`` in its causal-address checks.  Enforce that narrower
    limit before allocation so rounding cannot wrap either representation.
    """
    if isinstance(max_context, bool) or not isinstance(max_context, int):
        raise TypeError("max_context must be an integer")
    if isinstance(compress_ratio, bool) or not isinstance(compress_ratio, int):
        raise TypeError("compress_ratio must be an integer")
    if max_context < 1:
        raise ValueError(f"max_context must be positive; got {max_context}")
    if compress_ratio < 1:
        raise ValueError("compress_ratio must be positive")
    capacity = ((max_context + compress_ratio - 1) // compress_ratio
                * compress_ratio)
    if capacity > METAL_INT32_MAX:
        raise OverflowError(
            f"rounded ledger capacity {capacity} exceeds Metal int32 addressing"
        )
    return capacity


def score_tile_layout(
    block_count: int,
    width: int = 1,
    tile_blocks: int = SCORE_TILE_BLOCKS,
) -> dict[str, int]:
    """Describe the dynamically sized, tile-aligned index-score planes.

    Scores are still globally selected in their original block order.  Tiling
    is an allocation/addressing contract: it removes the old fixed 16K-block
    plane while keeping each query in a disjoint, padded plane.  The returned
    values are safe to place in Metal uint32 control words and host shape
    products.
    """
    for name, value in (
        ("block_count", block_count),
        ("width", width),
        ("tile_blocks", tile_blocks),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if block_count < 0:
        raise ValueError("block_count must be non-negative")
    if not 1 <= width <= MAX_QUERY_WIDTH:
        raise ValueError(
            f"width must be between 1 and {MAX_QUERY_WIDTH}; got {width}"
        )
    if tile_blocks < 1 or tile_blocks & (tile_blocks - 1):
        raise ValueError("tile_blocks must be a positive power of two")
    tiles = max(1, (block_count + tile_blocks - 1) // tile_blocks)
    stride = tiles * tile_blocks
    if stride > METAL_UINT32_MAX:
        raise OverflowError(
            f"score stride {stride} exceeds Metal uint32 addressing"
        )
    elements = width * stride
    if elements > METAL_INT32_MAX:
        raise OverflowError(
            f"score allocation has {elements} elements, over the host shape limit"
        )
    return {
        "tile_blocks": tile_blocks,
        "tiles": tiles,
        "block_count": block_count,
        "stride": stride,
        "width": width,
        "elements": elements,
        "bytes": elements * 4,
    }


def validate_model_contract(args: Any) -> None:
    """Refuse any model geometry the generated Metal body does not encode."""
    for name, expected in MODEL_CONTRACT.items():
        actual = getattr(args, name, None)
        if actual != expected:
            raise ValueError(
                f"megakernel requires {name}={expected!r}; got {actual!r}"
            )
    layers = list(getattr(args, "layer_types", ()) or ())
    n_layers = int(getattr(args, "num_hidden_layers", len(layers)))
    if len(layers) != n_layers:
        raise ValueError(
            f"layer_types has {len(layers)} entries, expected {n_layers}"
        )
    unsupported = [value for value in layers
                   if value not in {"linear_attention", "full_attention"}]
    if unsupported:
        raise ValueError(f"unsupported layer type {unsupported[0]!r}")
    if tuple(layers) != EXPECTED_LAYER_TYPES:
        raise ValueError("megakernel requires the target's exact layer pattern")
    partial = float(getattr(args, "partial_rotary_factor", 0.0))
    if int(getattr(args, "head_dim", 0) * partial) != 64:
        raise ValueError(
            "megakernel requires a 64-dimensional partial rotary embedding"
        )
    scaling = getattr(args, "rope_scaling", None)
    if scaling is not None and dict(scaling).get("type", "default") != "default":
        raise ValueError("megakernel supports only default RoPE scaling")


def validate_model_binding(model: Any, args: Any) -> None:
    """Prove the supplied contract describes the model that will be packed."""
    language_model = getattr(model, "language_model", None)
    live = getattr(language_model, "args", None)
    if live is None:
        raise ValueError("megakernel requires the Qwen4 Model wrapper as model")
    fields = (*MODEL_CONTRACT, "rope_theta", "partial_rotary_factor",
              "num_hidden_layers", "layer_types", "ple_layer_ids")
    for name in fields:
        expected, actual = getattr(args, name, None), getattr(live, name, None)
        if actual != expected:
            raise ValueError(
                f"model/args mismatch for {name}: {actual!r} != {expected!r}"
            )


def validate_launch_shapes(
    embedding_shape: Iterable[int],
    ple_shape: Iterable[int] | None,
    *,
    width: int,
    has_ple: bool,
    hidden: int = 2560,
) -> None:
    """Validate the batch-one host inputs before any device allocation."""
    shape = tuple(int(v) for v in embedding_shape)
    expected = (hidden,) if width == 1 else (width, hidden)
    if shape != expected:
        raise ValueError(f"embedding shape must be {expected}; got {shape}")
    pshape = None if ple_shape is None else tuple(int(v) for v in ple_shape)
    if has_ple and pshape is None:
        raise ValueError("PLE embedding is required for the configured PLE layer")
    if not has_ple and pshape is not None:
        raise ValueError("PLE embedding was supplied but no PLE layer is configured")
    if pshape is not None and pshape != expected:
        raise ValueError(f"PLE embedding shape must be {expected}; got {pshape}")


def validate_position(current: int, requested: int, width: int, total: int) -> None:
    """Enforce the decoder's single monotonic transaction cursor."""
    current, requested, width, total = map(int, (current, requested, width, total))
    if requested != current:
        raise ValueError(
            f"launch position {requested} does not match decoder position {current}"
        )
    if width < 1:
        raise ValueError(f"width must be positive; got {width}")
    if requested < 0 or requested + width > total:
        raise ValueError(
            f"megakernel span [{requested}, {requested + width}) exceeds "
            f"ledger capacity {total}"
        )
