import os

import mlx.core as mx
import pytest
from mlx import nn

from mlx_lm.models import qwen4_fused_gdn as fused_gdn
from mlx_lm.models.gated_delta import gated_delta_update, normalize_gdn_qk

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX_AGNES_RUN_REAL_METAL_TEST") != "1",
    reason="set MLX_AGNES_RUN_REAL_METAL_TEST=1 for the real-Metal gate",
)


def _assert_real_metal_kernel_matches_stock(threadgroup_y):
    """Guard every Agnes BF16 boundary replaced by the fused dispatch."""
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    dtype = mx.bfloat16
    conv_weight = (
        mx.random.normal(
            (fused_gdn.CONV_DIM, fused_gdn.CONV_KERNEL, 1),
            key=mx.random.key(1),
        )
        * 0.02
    ).astype(dtype)
    a_log = (
        mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(2)) * 0.2
    ).astype(mx.float32)
    dt_bias = (
        mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(3)) * 0.2
    ).astype(dtype)
    norm_weight = (
        mx.random.normal((fused_gdn.VALUE_HEAD_DIM,), key=mx.random.key(4)) * 0.05 + 1
    ).astype(dtype)
    stock_conv = mx.zeros(
        (1, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM), dtype=dtype
    )
    fused_conv = mx.array(stock_conv)
    stock_state = mx.zeros(
        (
            1,
            fused_gdn.NUM_VALUE_HEADS,
            fused_gdn.VALUE_HEAD_DIM,
            fused_gdn.KEY_HEAD_DIM,
        ),
        dtype=mx.float32,
    )
    fused_state = mx.array(stock_state)

    try:
        for step in range(32):
            qkv = (
                mx.random.normal(
                    (1, 1, fused_gdn.CONV_DIM), key=mx.random.key(100 + step)
                )
                * 0.2
            ).astype(dtype)
            z = (
                mx.random.normal(
                    (1, 1, fused_gdn.VALUE_DIM), key=mx.random.key(200 + step)
                )
                * 0.2
            ).astype(dtype)
            beta = (
                mx.random.normal(
                    (1, 1, fused_gdn.NUM_VALUE_HEADS),
                    key=mx.random.key(300 + step),
                )
                * 0.2
            ).astype(dtype)
            alpha = (
                mx.random.normal(
                    (1, 1, fused_gdn.NUM_VALUE_HEADS),
                    key=mx.random.key(400 + step),
                )
                * 0.2
            ).astype(dtype)

            conv_input = mx.concatenate([stock_conv, qkv], axis=1)
            next_stock_conv = mx.contiguous(
                conv_input[:, -(fused_gdn.CONV_KERNEL - 1) :, :]
            )
            convolved = nn.silu(
                mx.conv1d(
                    conv_input,
                    conv_weight,
                    groups=fused_gdn.CONV_DIM,
                )
            )
            query, key, value = [
                item.reshape(1, 1, heads, dim)
                for item, heads, dim in zip(
                    mx.split(
                        convolved,
                        [fused_gdn.KEY_DIM, 2 * fused_gdn.KEY_DIM],
                        axis=-1,
                    ),
                    [
                        fused_gdn.NUM_KEY_HEADS,
                        fused_gdn.NUM_KEY_HEADS,
                        fused_gdn.NUM_VALUE_HEADS,
                    ],
                    [
                        fused_gdn.KEY_HEAD_DIM,
                        fused_gdn.KEY_HEAD_DIM,
                        fused_gdn.VALUE_HEAD_DIM,
                    ],
                )
            ]
            query, key = normalize_gdn_qk(query, key)
            stock_output, next_stock_state = gated_delta_update(
                query,
                key,
                value,
                alpha,
                beta,
                a_log,
                dt_bias,
                stock_state,
                use_kernel=True,
            )
            stock_output = mx.fast.rms_norm(stock_output, norm_weight, 1e-6).astype(
                mx.float32
            )
            stock_gate = nn.silu(
                z.reshape(
                    1,
                    1,
                    fused_gdn.NUM_VALUE_HEADS,
                    fused_gdn.VALUE_HEAD_DIM,
                ).astype(mx.float32)
            )
            stock_output = (stock_output * stock_gate).astype(dtype)
            stock_output = stock_output.reshape(1, 1, fused_gdn.VALUE_DIM)

            fused_output, next_fused_conv, next_fused_state = (
                fused_gdn.qwen4_fused_gdn_decode(
                    qkv,
                    z,
                    beta,
                    alpha,
                    fused_conv,
                    conv_weight,
                    a_log,
                    dt_bias,
                    fused_state,
                    norm_weight,
                    1e-6,
                    threadgroup_y=threadgroup_y,
                    architecture="agnes",
                )
            )
            mx.eval(
                stock_output,
                fused_output,
                next_stock_conv,
                next_fused_conv,
                next_stock_state,
                next_fused_state,
            )
            assert mx.array_equal(stock_output, fused_output).item(), step
            assert mx.array_equal(next_stock_conv, next_fused_conv).item(), step
            assert mx.array_equal(next_stock_state, next_fused_state).item(), step
            stock_conv, fused_conv = next_stock_conv, next_fused_conv
            stock_state, fused_state = next_stock_state, next_fused_state
    finally:
        mx.set_default_device(previous_device)


def test_real_metal_kernel_matches_stock_for_every_supported_threadgroup():
    supported = []
    for threadgroup_y in fused_gdn._THREADGROUP_Y_CANDIDATES:
        try:
            _assert_real_metal_kernel_matches_stock(threadgroup_y)
        except ValueError as exc:
            if "threads per threadgroup" not in str(exc):
                raise
        except RuntimeError:
            continue
        else:
            supported.append(threadgroup_y)
    assert supported, "Metal rejected every Agnes fused GDN threadgroup candidate"
