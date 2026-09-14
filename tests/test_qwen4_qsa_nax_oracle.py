"""M5 numerical oracle for the production QSA NAX attention geometry."""

import os

os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import pytest

from mlx_lm.models.qwen4_exp import QSASelection
from mlx_lm.models.qwen4_qsa_nax import (
    compact_blocks_to_kernel_inputs,
    nax_qsa_attention,
)


BLOCK_SIZE = 4
N_QUERY_HEADS = 24
N_KV_HEADS = 2
HEAD_DIM = 256


def _is_m5_metal() -> bool:
    try:
        return mx.metal.is_available() and "M5" in str(
            mx.device_info().get("device_name", "")
        )
    except Exception:
        return False


def _production_selection(*, length: int, total: int, topk: int) -> QSASelection:
    n_blocks = total // BLOCK_SIZE
    offset = total - length
    q_positions = mx.arange(offset, total, dtype=mx.int32)[None, :]
    token_positions = mx.arange(total, dtype=mx.int32)[None, :]
    block_ends = mx.arange(n_blocks, dtype=mx.int32) * BLOCK_SIZE + (
        BLOCK_SIZE - 1
    )
    valid_blocks = block_ends[None, None, :] <= q_positions[..., None]

    mx.random.seed(3440)
    scores = mx.random.normal((1, length, n_blocks))
    scores = mx.where(valid_blocks, scores, -mx.inf)
    selected = mx.argpartition(
        scores, kth=n_blocks - topk, axis=-1
    )[..., -topk:].astype(mx.uint32)
    causal_mask = (
        token_positions[:, None, :] <= q_positions[..., None]
    )[:, None, :, :]

    return QSASelection(
        kind="explicit",
        batch=1,
        length=length,
        block_size=BLOCK_SIZE,
        raw_block_ids=selected,
        valid_blocks=valid_blocks,
        q_positions=q_positions,
        token_positions=token_positions,
        causal_mask=causal_mask,
        offset=offset,
        physical_width=total,
        n_blocks=n_blocks,
        scatter_chosen=True,
    )


@pytest.mark.skipif(
    not _is_m5_metal(),
    reason="the QSA NAX numerical oracle requires an Apple M5-class GPU",
)
def test_qsa_nax_production_geometry_matches_fp32_dense_reference():
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        length, total, topk = 64, 1024, 32
        selection = _production_selection(
            length=length, total=total, topk=topk
        )
        dense_mask = selection.dense_mask()
        compact = selection.compact_blocks()
        ids, counts, n_sel, u_width, q_pos, left_pad, kernel_total = (
            compact_blocks_to_kernel_inputs(compact)
        )

        mx.random.seed(3441)
        q = mx.random.normal(
            (1, N_QUERY_HEADS, length, HEAD_DIM)
        ).astype(mx.bfloat16)
        k = mx.random.normal(
            (1, N_KV_HEADS, total, HEAD_DIM)
        ).astype(mx.bfloat16)
        v = mx.random.normal(
            (1, N_KV_HEADS, total, HEAD_DIM)
        ).astype(mx.bfloat16)
        scale = HEAD_DIM**-0.5
        repeats = N_QUERY_HEADS // N_KV_HEADS

        reference = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32),
            mx.repeat(k, repeats, axis=1).astype(mx.float32),
            mx.repeat(v, repeats, axis=1).astype(mx.float32),
            scale=scale,
            mask=dense_mask,
        )
        actual = nax_qsa_attention(
            q,
            k,
            v,
            ids,
            counts,
            n_sel,
            q_pos,
            left_pad,
            scale=scale,
            u_width=u_width,
            total=kernel_total,
            n_kv_heads=N_KV_HEADS,
        )
        mx.eval(reference, actual)

        max_error = float(mx.max(mx.abs(actual - reference)).item())
        output_scale = float(mx.max(mx.abs(reference)).item())
        relative_error = max_error / max(output_scale, 1e-30)
        assert bool(mx.all(mx.isfinite(actual)).item())
        assert relative_error <= 2.0e-3, (
            f"24Q/2KV/D256 NAX relative error {relative_error:.3e} "
            f"exceeds 2.0e-3 (max abs {max_error:.3e})"
        )
    finally:
        mx.set_default_device(previous_device)
