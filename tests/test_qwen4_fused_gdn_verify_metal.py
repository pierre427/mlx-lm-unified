# Copyright © 2025 Apple Inc.

"""Real-Metal gates for the fused Qwen4 GDN speculative-verify kernel."""

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_lm.models import qwen4_exp
from mlx_lm.models import qwen4_fused_gdn as fused_gdn
from mlx_lm.models import qwen4_fused_gdn_verify as fused_verify
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_update

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX_QWEN4_RUN_REAL_METAL_TEST") != "1",
    reason="set MLX_QWEN4_RUN_REAL_METAL_TEST=1 for the real-Metal gate",
)


def _require_metal():
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")


def _stock_verify_block(
    qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias, norm_weight, state
):
    """The stock Qwen4 verify path (qwen3_5.GatedDeltaNet.__call__ with the
    qwen4_exp overrides), including the replay closure's restore points."""
    steps = qkv.shape[1]
    keep = fused_gdn.CONV_KERNEL - 1
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    next_conv = mx.contiguous(conv_input[:, -keep:, :])
    convolved = nn.silu(mx.conv1d(conv_input, conv_weight, groups=fused_gdn.CONV_DIM))
    q, k, v = [
        t.reshape(1, steps, h, d)
        for t, h, d in zip(
            mx.split(convolved, [fused_gdn.KEY_DIM, 2 * fused_gdn.KEY_DIM], axis=-1),
            [
                fused_gdn.NUM_KEY_HEADS,
                fused_gdn.NUM_KEY_HEADS,
                fused_gdn.NUM_VALUE_HEADS,
            ],
            [fused_gdn.KEY_HEAD_DIM, fused_gdn.KEY_HEAD_DIM, fused_gdn.VALUE_HEAD_DIM],
        )
    ]
    q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1.0e-6)
    k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1.0e-6)
    q = q * (fused_gdn.KEY_HEAD_DIM**-0.5)

    def update(m):
        return gated_delta_update(
            q[:, :m],
            k[:, :m],
            v[:, :m],
            a[:, :m],
            b[:, :m],
            A_log,
            dt_bias,
            state,
            None,
            use_kernel=True,
            beta_input_dtype=True,
        )

    out, next_state = update(steps)
    restore_states = [update(m)[1] for m in range(1, steps)]
    restore_convs = [
        mx.contiguous(conv_input[:, m : m + keep, :]) for m in range(1, steps)
    ]
    gate = mx.sigmoid(
        z.reshape(1, steps, fused_gdn.NUM_VALUE_HEADS, -1).astype(mx.float32)
    )
    output = (
        mx.fast.rms_norm(out, norm_weight, 1.0e-6).astype(mx.float32) * gate
    ).astype(qkv.dtype)
    return (
        output.reshape(1, steps, fused_gdn.VALUE_DIM),
        next_conv,
        next_state,
        restore_states,
        restore_convs,
    )


def _random_layer_weights(dtype=mx.bfloat16):
    conv_weight = (
        mx.random.normal(
            (fused_gdn.CONV_DIM, fused_gdn.CONV_KERNEL, 1), key=mx.random.key(1)
        )
        * 0.02
    ).astype(dtype)
    A_log = (
        mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(2)) * 0.2
    ).astype(mx.float32)
    dt_bias = (
        mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(3)) * 0.2
    ).astype(dtype)
    norm_weight = (
        mx.random.normal((fused_gdn.VALUE_HEAD_DIM,), key=mx.random.key(4)) * 0.05 + 1
    ).astype(dtype)
    return conv_weight, A_log, dt_bias, norm_weight


def _assert_verify_kernel_matches_stock(steps, threadgroup_y, blocks=6):
    _require_metal()
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    dtype = mx.bfloat16
    conv_weight, A_log, dt_bias, norm_weight = _random_layer_weights(dtype)
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
        for block in range(blocks):
            seed = 1000 * steps + 10 * block

            def draw(shape, offset, seed=seed, scale=0.2):
                return (
                    mx.random.normal(shape, key=mx.random.key(seed + offset)) * scale
                ).astype(dtype)

            qkv = draw((1, steps, fused_gdn.CONV_DIM), 1)
            z = draw((1, steps, fused_gdn.VALUE_DIM), 2)
            b = draw((1, steps, fused_gdn.NUM_VALUE_HEADS), 3)
            a = draw((1, steps, fused_gdn.NUM_VALUE_HEADS), 4)
            stock = _stock_verify_block(
                qkv,
                z,
                b,
                a,
                stock_conv,
                conv_weight,
                A_log,
                dt_bias,
                norm_weight,
                stock_state,
            )
            fused = fused_verify.qwen4_fused_gdn_verify(
                qkv,
                z,
                b,
                a,
                fused_conv,
                conv_weight,
                A_log,
                dt_bias,
                fused_state,
                norm_weight,
                1.0e-6,
                threadgroup_y=threadgroup_y,
            )
            mx.eval(*stock[:3], *stock[3], *stock[4], *fused)
            assert mx.array_equal(stock[0], fused[0]).item(), ("output", block)
            assert mx.array_equal(stock[1], fused[1]).item(), ("conv", block)
            assert mx.array_equal(stock[2], fused[2]).item(), ("state", block)
            for p in range(steps - 1):
                assert mx.array_equal(stock[3][p], fused[3][:, p]).item(), (
                    "state point",
                    block,
                    p,
                )
                assert mx.array_equal(stock[4][p], fused[4][:, p]).item(), (
                    "conv point",
                    block,
                    p,
                )
            keep = block % steps  # 0 commits the block; k > 0 restores to k tokens
            if keep == 0:
                stock_conv, stock_state, fused_conv, fused_state = (
                    stock[1],
                    stock[2],
                    fused[1],
                    fused[2],
                )
            else:
                stock_conv, stock_state = stock[4][keep - 1], stock[3][keep - 1]
                fused_conv, fused_state = fused[4][:, keep - 1], fused[3][:, keep - 1]
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("steps", [2, 3, 5, fused_verify.MAX_VERIFY_STEPS])
def test_real_metal_verify_matches_stock_for_every_supported_threadgroup(steps):
    supported = []
    for threadgroup_y in fused_gdn._THREADGROUP_Y_CANDIDATES:
        try:
            _assert_verify_kernel_matches_stock(steps, threadgroup_y)
        except ValueError as exc:
            if "threads per threadgroup" not in str(exc):
                raise
        except RuntimeError:
            continue
        else:
            supported.append(threadgroup_y)
    assert supported, "Metal rejected every fused GDN verify threadgroup candidate"


def _production_layer():
    args = SimpleNamespace(
        hidden_size=256,
        linear_num_value_heads=fused_gdn.NUM_VALUE_HEADS,
        linear_num_key_heads=fused_gdn.NUM_KEY_HEADS,
        linear_key_head_dim=fused_gdn.KEY_HEAD_DIM,
        linear_value_head_dim=fused_gdn.VALUE_HEAD_DIM,
        linear_conv_kernel_dim=fused_gdn.CONV_KERNEL,
        rms_norm_eps=1.0e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )
    mx.random.seed(7)
    layer = qwen4_exp.GatedDeltaNet(args)
    layer.eval()
    layer.set_dtype(mx.bfloat16)
    layer.A_log = layer.A_log.astype(mx.float32)
    mx.eval(layer.parameters())
    layer.set_fused_gdn_decode_mode("stock")
    return layer, args


@pytest.mark.parametrize("slots", [2, 4])
def test_real_metal_layer_verify_records_exact_restore_points_and_trims(slots):
    """Drive GatedDeltaNet under a speculating cache: the fused record must
    replay identically to the stock replay closure, and ``trim`` must land on
    the same state. ``slots=4`` mimics a PLE layer whose half is staged first."""
    _require_metal()
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        layer, args = _production_layer()
        steps = 3
        cache_type = qwen4_exp.Qwen4ArraysCache if slots == 4 else ArraysCache
        stock_cache, fused_cache = cache_type(slots), cache_type(slots)
        warm = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)
        layer.set_fused_gdn_verify_mode("stock")
        mx.eval(layer(warm, cache=stock_cache), layer(warm, cache=fused_cache))
        for cache in (stock_cache, fused_cache):
            for slot in range(2, slots):
                cache.cache[slot] = mx.array([float(100 + slot)])
            cache.start_speculation()

        def stage_ple(cache, block):
            if slots == 2:
                return
            snapshot = [mx.array(cache.cache[2]), mx.array(cache.cache[3])]

            def ple_fn(m, block=block):
                return [
                    mx.array([block * 10.0 + m]),
                    mx.array([block * 10.0 + m + 0.5]),
                ]

            cache.stage_ple_rollback(steps, ple_fn, snapshot)

        for block in range(6):
            hidden = (mx.random.normal((1, steps, args.hidden_size)) * 0.5).astype(
                mx.bfloat16
            )
            layer.set_fused_gdn_verify_mode("stock")
            stage_ple(stock_cache, block)
            stock_out = layer(hidden, cache=stock_cache)
            layer.set_fused_gdn_verify_mode("fused")
            stage_ple(fused_cache, block)
            fused_out = layer(hidden, cache=fused_cache)
            mx.eval(stock_out, fused_out, *stock_cache.cache, *fused_cache.cache)
            assert mx.array_equal(stock_out, fused_out).item(), block
            for slot in range(slots):
                assert mx.array_equal(
                    stock_cache.cache[slot], fused_cache.cache[slot]
                ).item(), (block, slot)
            stock_record, fused_record = (
                stock_cache._rollbacks[-1],
                fused_cache._rollbacks[-1],
            )
            assert stock_record.num_tokens == fused_record.num_tokens == steps
            for m in range(1, steps):
                left, right = stock_record.fn(m), fused_record.fn(m)
                assert len(left) == len(right) == slots
                mx.eval(*left, *right)
                for slot, (x, y) in enumerate(zip(left, right)):
                    assert mx.array_equal(x, y).item(), (block, m, slot)
            n_to_drop = block % steps
            if n_to_drop:
                assert stock_cache.trim(n_to_drop) == n_to_drop
                assert fused_cache.trim(n_to_drop) == n_to_drop
                mx.eval(*stock_cache.cache, *fused_cache.cache)
                for slot in range(slots):
                    assert mx.array_equal(
                        stock_cache.cache[slot], fused_cache.cache[slot]
                    ).item(), ("trim", block, slot)
        assert layer.fused_gdn_verify_calls == 6
        assert layer.fused_gdn_verify_fallbacks == 0
        assert layer.fused_gdn_decode_calls == 0
    finally:
        mx.set_default_device(previous)


def _every_finite_bf16():
    bits = np.arange(0, 65536, dtype=np.uint32)
    values = (bits << 16).view(np.float32)
    return mx.array(values[np.isfinite(values)]).astype(mx.bfloat16)


def _elementwise(name, body, out_dtype, x):
    kernel = mx.fast.metal_kernel(
        name=f"sweep_{name}",
        input_names=["x"],
        output_names=["out"],
        header=fused_gdn._HEADER,
        source="uint i = thread_position_in_grid.x; " + body,
    )
    (out,) = kernel(
        inputs=[x],
        template=[("T", mx.bfloat16)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[out_dtype],
    )
    return out


def test_real_metal_unary_boundaries_match_mlx_on_every_finite_bf16_value():
    """Pin the sigmoid forms both fused kernels use, over the whole bf16 domain.

    ``mlx_sigmoid_fast<bf16>`` differs from ``mx.sigmoid`` on exactly one
    finite bf16 input (x ~ -6.85), which real Qwen4 activations reach; sampled
    trajectories cannot see a one-value boundary, an exhaustive sweep can.
    """
    _require_metal()
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        x = _every_finite_bf16()
        assert x.size == 65280
        beta = _elementwise(
            "beta", "out[i] = float(mlx_sigmoid_precise(x[i]));", mx.float32, x
        )
        silu = _elementwise(
            "silu",
            "{ T v = x[i]; T s = mlx_sigmoid_fast(v); out[i] = v * s; }",
            mx.bfloat16,
            x,
        )
        gate = _elementwise(
            "gate", "out[i] = mlx_sigmoid_precise<float>(float(x[i]));", mx.float32, x
        )
        decay = _elementwise(
            "decay",
            "{ T sp = mlx_softplus_fast(x[i]); out[i] = metal::precise::exp("
            "-metal::precise::exp(0.7f) * float(sp)); }",
            mx.float32,
            x,
        )
        fast_beta = _elementwise(
            "beta_fast", "out[i] = float(mlx_sigmoid_fast(x[i]));", mx.float32, x
        )
        ref_decay = compute_g(mx.array([0.7], mx.float32), x, mx.zeros_like(x))
        mx.eval(beta, silu, gate, decay, fast_beta, ref_decay)
        assert int(mx.sum(beta != mx.sigmoid(x).astype(mx.float32)).item()) == 0
        assert int(mx.sum(silu != nn.silu(x)).item()) == 0
        assert int(mx.sum(gate != mx.sigmoid(x.astype(mx.float32))).item()) == 0
        assert int(mx.sum(decay != ref_decay).item()) == 0
        # The rejected form really differs, so the sweep is not vacuous.
        assert int(mx.sum(fast_beta != mx.sigmoid(x).astype(mx.float32)).item()) == 1
    finally:
        mx.set_default_device(previous)
