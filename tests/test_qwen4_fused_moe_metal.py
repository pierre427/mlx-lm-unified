import os

import mlx.core as mx
import pytest

from mlx_lm.models import qwen4_fused_moe


pytestmark = pytest.mark.skipif(
    os.environ.get("MLX_QWEN4_RUN_REAL_METAL_TEST") != "1",
    reason="set MLX_QWEN4_RUN_REAL_METAL_TEST=1 for the real-Metal gate",
)


def _router_shaped_indices(tokens, top_k=10, num_experts=512):
    """Reproduce the production router layout: the last top_k columns."""
    gates = mx.random.normal((1, tokens, num_experts), key=mx.random.key(0))
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(mx.bfloat16)
    return inds.astype(mx.uint32), scores


def _flat_buffer_read(indices, count):
    """Read `indices` the way the kernels do: flat, ignoring strides."""
    probe = mx.fast.metal_kernel(
        name="qwen4_index_buffer_probe_test",
        input_names=["indices"],
        output_names=["out"],
        source=(
            "    uint i = thread_position_in_grid.x;\n"
            "    out[i] = uint(indices[i]);\n"
        ),
        ensure_row_contiguous=False,
    )
    return probe(
        inputs=[indices],
        template=[("T", indices.dtype)],
        grid=(count, 1, 1),
        threadgroup=(min(count, 32), 1, 1),
        output_shapes=[(count,)],
        output_dtypes=[mx.uint32],
    )[0]


def test_router_index_view_is_misread_flat_and_contiguous_repairs_it():
    """The defect that made width>1 wrong, isolated from the MoE kernels."""
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")
    mx.set_default_device(mx.gpu)

    # A single token is row-contiguous by construction, so the flat read is
    # correct at M=1 by accident. That is why the bug never showed at decode.
    inds, _ = _router_shaped_indices(1)
    flat = inds.reshape(-1)
    assert mx.array_equal(_flat_buffer_read(inds, 10), flat)

    for tokens in (2, 3, 8):
        inds, _ = _router_shaped_indices(tokens)
        flat = inds.reshape(-1)
        seen = _flat_buffer_read(inds, tokens * 10)
        assert not mx.array_equal(seen, flat), tokens
        repaired = _flat_buffer_read(mx.contiguous(inds), tokens * 10)
        assert mx.array_equal(repaired, flat), tokens


def _reference(hidden, indices, scores, weight, scales, biases):
    tokens, top_k, _ = hidden.shape
    flat = indices.reshape(-1).astype(mx.uint32)
    table = mx.dequantize(
        mx.take(weight, flat, axis=0),
        mx.take(scales, flat, axis=0).astype(mx.float32),
        mx.take(biases, flat, axis=0).astype(mx.float32),
        group_size=qwen4_fused_moe.GROUP_SIZE,
        bits=qwen4_fused_moe.BITS,
        mode="affine",
    )
    rows = (table @ hidden.reshape(tokens * top_k, -1, 1).astype(mx.float32))
    rows = rows.squeeze(-1).reshape(tokens, top_k, -1)
    return (rows * scores.reshape(tokens, top_k, 1).astype(mx.float32)).sum(1)


@pytest.mark.parametrize("variant", ["scalar", "tile4"])
@pytest.mark.parametrize("tokens", [1, 2, 3, 8])
def test_fused_down_matches_an_fp32_reference(variant, tokens, monkeypatch):
    """End-to-end: production-shaped tables, production router layout.

    The admission cap is lifted here on purpose: this asserts the kernel's
    arithmetic at each width. Which widths production admits is a separate
    contract, covered in test_qwen4_fused_moe_contract.py.
    """
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")
    mx.set_default_device(mx.gpu)
    monkeypatch.setattr(qwen4_fused_moe, "QUALIFIED_TOKEN_WIDTHS", (tokens,))

    experts = qwen4_fused_moe.NUM_EXPERTS
    rows = qwen4_fused_moe.HIDDEN_SIZE
    groups = qwen4_fused_moe.EXPERT_HIDDEN_SIZE // qwen4_fused_moe.GROUP_SIZE
    words = qwen4_fused_moe.EXPERT_HIDDEN_SIZE // qwen4_fused_moe.PACK_FACTOR

    weight = mx.random.randint(
        0, 2**31, (experts, rows, words), key=mx.random.key(1)
    ).astype(mx.uint32)
    scales = (
        mx.random.normal((experts, rows, groups), key=mx.random.key(2)) * 0.01
    ).astype(mx.bfloat16)
    biases = (
        mx.random.normal((experts, rows, groups), key=mx.random.key(3)) * 0.01
    ).astype(mx.bfloat16)
    hidden = (
        mx.random.normal(
            (1, tokens, qwen4_fused_moe.TOP_K, qwen4_fused_moe.EXPERT_HIDDEN_SIZE),
            key=mx.random.key(4),
        )
        * 0.5
    ).astype(mx.bfloat16)
    indices, scores = _router_shaped_indices(tokens)

    reference = _reference(
        hidden.reshape(tokens, qwen4_fused_moe.TOP_K, -1),
        indices,
        scores,
        weight,
        scales,
        biases,
    )
    peak = float(mx.max(mx.abs(reference)).item())

    fused = qwen4_fused_moe.qwen4_fused_down(
        hidden, indices, scores, weight, scales, biases, variant=variant
    ).reshape(tokens, rows)
    error = float(
        mx.max(mx.abs(fused.astype(mx.float32) - reference)).item()
    ) / peak
    # One bf16 ULP class; the pre-fix kernel was above 1.0 here at width > 1.
    assert error < 2.0**-7, (variant, tokens, error)
