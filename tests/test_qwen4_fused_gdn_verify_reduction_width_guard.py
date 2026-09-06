# Copyright © 2025 Apple Inc.

"""Class-C reduction-width guard for the fused Qwen4 GDN speculative-verify kernel.

Background (vllm#54506, memory ``qwen38-k2-divergence-is-near-tie-numeric``): on
some Qwen3.5/3.6 GDN/Mamba hybrids the reduction *width* used by the scan/norm
differs between the ``M=1`` decode path and the ``M=k+1`` verify path. Where a
top-2 decision sits on a ~1-ULP (near-tie) margin, that width change flips which
draft token is accepted -- a silent acceptance-rate regression (~100% -> 72% was
reported) that no bulk-tolerance check catches.

Our fused verify kernel (:mod:`mlx_lm.models.qwen4_fused_gdn_verify`) is on this
axis: it is the ``S = k + 1`` verify slab. This test forces the two arms that
Class-C would separate and asserts they do NOT separate:

* **Cross-width equivalence.** Position ``p`` of the width-``S`` verify slab is
  compared, bit-for-bit, to the width-``1`` decode kernel chained ``p + 1`` times
  from the same state, for ``S`` in ``{2, 3, 4}`` (``k`` in ``{1, 2, 3}``). If any
  reduction depended on ``S``, position ``p``'s output would move with ``S``; the
  recurrence is sequential over tokens, so it must not.

* **Forced trip-count divergence.** The same two-token prefix is run at ``S=2``
  and again as the prefix of an ``S=4`` slab. ``S`` sets only the token-loop trip
  count, so the shared positions must be bit-identical across the two widths.

* **Near-tie argmax stability.** Over many random blocks the smallest top-2 gap
  in the stock output is tracked, and the argmax of the fused verify, the chained
  decode, and the stock path are required to agree at every position. The verdict
  printed is the margin at which any flip first occurs (``inf`` = no flip).

The kernel's reductions are all over a fixed head dimension (``DK`` / ``DV``),
never over ``S`` -- see ``qwen4_fused_gdn_verify.py`` L314-374 vs the decode body
``qwen4_fused_gdn.py`` L303-378 -- so the expected verdict is PASS. This file is
the regression that keeps that true. It exercises real Metal, so the coordinator
runs it on the GPU:

    MLX_QWEN4_RUN_REAL_METAL_TEST=1 \
      /Users/pierrelamy/Desktop/mlx-uag/.venv/bin/python -m pytest -s \
      tests/test_qwen4_fused_gdn_verify_reduction_width_guard.py

``-s`` surfaces the printed PASS/FLIP verdict and the minimum near-tie margin.
"""

import math
import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.models import qwen4_fused_gdn as fused_gdn
from mlx_lm.models import qwen4_fused_gdn_verify as fused_verify
from mlx_lm.models.gated_delta import gated_delta_update

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX_QWEN4_RUN_REAL_METAL_TEST") != "1",
    reason="set MLX_QWEN4_RUN_REAL_METAL_TEST=1 for the real-Metal gate",
)


def _require_metal():
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")


def _pick_threadgroup_y():
    """First threadgroup_y Metal accepts for both kernels at this geometry."""
    dtype = mx.bfloat16
    conv_weight = mx.zeros((fused_gdn.CONV_DIM, fused_gdn.CONV_KERNEL, 1), dtype=dtype)
    A_log = mx.zeros((fused_gdn.NUM_VALUE_HEADS,), dtype=mx.float32)
    dt_bias = mx.zeros((fused_gdn.NUM_VALUE_HEADS,), dtype=dtype)
    norm_weight = mx.ones((fused_gdn.VALUE_HEAD_DIM,), dtype=dtype)
    conv_state = mx.zeros((1, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM), dtype=dtype)
    state = mx.zeros(
        (1, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM, fused_gdn.KEY_HEAD_DIM),
        dtype=mx.float32,
    )
    for ty in fused_gdn._THREADGROUP_Y_CANDIDATES:
        try:
            dec = fused_gdn.qwen4_fused_gdn_decode(
                mx.zeros((1, 1, fused_gdn.CONV_DIM), dtype=dtype),
                mx.zeros((1, 1, fused_gdn.VALUE_DIM), dtype=dtype),
                mx.zeros((1, 1, fused_gdn.NUM_VALUE_HEADS), dtype=dtype),
                mx.zeros((1, 1, fused_gdn.NUM_VALUE_HEADS), dtype=dtype),
                conv_state,
                conv_weight,
                A_log,
                dt_bias,
                state,
                norm_weight,
                1.0e-6,
                threadgroup_y=ty,
            )
            ver = fused_verify.qwen4_fused_gdn_verify(
                mx.zeros((1, 2, fused_gdn.CONV_DIM), dtype=dtype),
                mx.zeros((1, 2, fused_gdn.VALUE_DIM), dtype=dtype),
                mx.zeros((1, 2, fused_gdn.NUM_VALUE_HEADS), dtype=dtype),
                mx.zeros((1, 2, fused_gdn.NUM_VALUE_HEADS), dtype=dtype),
                conv_state,
                conv_weight,
                A_log,
                dt_bias,
                state,
                norm_weight,
                1.0e-6,
                threadgroup_y=ty,
            )
            mx.eval(*dec, *ver)
            return ty
        except ValueError as exc:
            if "threads per threadgroup" in str(exc):
                continue
            raise
        except RuntimeError:
            continue
    pytest.skip("Metal rejected every fused GDN threadgroup candidate")


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


def _stock_verify_output(qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias, norm_weight, state):
    """Stock Qwen4 verify output (no snapshots); mirrors the metal-gate reference."""
    steps = qkv.shape[1]
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    convolved = nn.silu(mx.conv1d(conv_input, conv_weight, groups=fused_gdn.CONV_DIM))
    q, k, v = [
        t.reshape(1, steps, h, d)
        for t, h, d in zip(
            mx.split(convolved, [fused_gdn.KEY_DIM, 2 * fused_gdn.KEY_DIM], axis=-1),
            [fused_gdn.NUM_KEY_HEADS, fused_gdn.NUM_KEY_HEADS, fused_gdn.NUM_VALUE_HEADS],
            [fused_gdn.KEY_HEAD_DIM, fused_gdn.KEY_HEAD_DIM, fused_gdn.VALUE_HEAD_DIM],
        )
    ]
    q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1.0e-6)
    k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1.0e-6)
    q = q * (fused_gdn.KEY_HEAD_DIM**-0.5)
    out, _ = gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state, None,
        use_kernel=True, beta_input_dtype=True,
    )
    gate = mx.sigmoid(z.reshape(1, steps, fused_gdn.NUM_VALUE_HEADS, -1).astype(mx.float32))
    output = (mx.fast.rms_norm(out, norm_weight, 1.0e-6).astype(mx.float32) * gate).astype(
        qkv.dtype
    )
    return output.reshape(1, steps, fused_gdn.VALUE_DIM)


def _decode_step(qkv_t, z_t, b_t, a_t, conv_state, conv_weight, A_log, dt_bias, state, norm_weight, ty):
    """One M=1 decode: returns (output(1,1,VD), next_conv, next_state)."""
    out, next_conv, next_state = fused_gdn.qwen4_fused_gdn_decode(
        qkv_t, z_t, b_t, a_t, conv_state, conv_weight, A_log, dt_bias, state,
        norm_weight, 1.0e-6, threadgroup_y=ty,
    )
    return out, next_conv, next_state


def _top2_gap(vec):
    """Smallest top-2 gap of a 1-D mx array (the near-tie decision margin)."""
    v = mx.sort(vec.astype(mx.float32))
    return float((v[-1] - v[-2]).item())


def test_reduction_width_is_M_invariant_and_no_near_tie_flip():
    _require_metal()
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    dtype = mx.bfloat16
    try:
        ty = _pick_threadgroup_y()
        conv_weight, A_log, dt_bias, norm_weight = _random_layer_weights(dtype)

        init_conv = mx.zeros(
            (1, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM), dtype=dtype
        )
        init_state = mx.zeros(
            (1, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM, fused_gdn.KEY_HEAD_DIM),
            dtype=mx.float32,
        )

        blocks = 24
        widths = (2, 3, 4)  # k + 1 for k in {1, 2, 3}
        min_flip_margin = math.inf   # smallest top-2 gap at which ANY argmax flips
        min_gap_seen = math.inf      # smallest top-2 gap observed at all (canary)
        cross_width_exact = True     # position p bit-identical across S and vs decode
        forced_prefix_exact = True   # S=2 prefix == first 2 positions of S=4
        argmax_stable = True

        for block in range(blocks):
            seed = 7000 + 37 * block

            def draw(shape, offset, seed=seed, scale=0.35):
                return (
                    mx.random.normal(shape, key=mx.random.key(seed + offset)) * scale
                ).astype(dtype)

            max_w = max(widths)
            qkv = draw((1, max_w, fused_gdn.CONV_DIM), 1)
            z = draw((1, max_w, fused_gdn.VALUE_DIM), 2)
            b = draw((1, max_w, fused_gdn.NUM_VALUE_HEADS), 3)
            a = draw((1, max_w, fused_gdn.NUM_VALUE_HEADS), 4)

            # Chained M=1 decode: the width-invariant per-position reference.
            decode_positions = []
            conv_state, state = init_conv, init_state
            for t in range(max_w):
                out_t, conv_state, state = _decode_step(
                    mx.contiguous(qkv[:, t : t + 1, :]),
                    mx.contiguous(z[:, t : t + 1, :]),
                    mx.contiguous(b[:, t : t + 1, :]),
                    mx.contiguous(a[:, t : t + 1, :]),
                    conv_state, conv_weight, A_log, dt_bias, state, norm_weight, ty,
                )
                decode_positions.append(out_t.reshape(fused_gdn.VALUE_DIM))
            mx.eval(*decode_positions)

            verify_by_width = {}
            for S in widths:
                v_out = fused_verify.qwen4_fused_gdn_verify(
                    mx.contiguous(qkv[:, :S, :]),
                    mx.contiguous(z[:, :S, :]),
                    mx.contiguous(b[:, :S, :]),
                    mx.contiguous(a[:, :S, :]),
                    init_conv, conv_weight, A_log, dt_bias, init_state,
                    norm_weight, 1.0e-6, threadgroup_y=ty,
                )[0]
                mx.eval(v_out)
                verify_by_width[S] = v_out

                # Cross-width equivalence: verify pos p == chained decode pos p.
                for p in range(S):
                    if not mx.array_equal(
                        v_out[:, p].reshape(fused_gdn.VALUE_DIM), decode_positions[p]
                    ).item():
                        cross_width_exact = False

            # Forced trip-count divergence: S=2 prefix == first 2 of S=4.
            for p in range(2):
                if not mx.array_equal(
                    verify_by_width[2][:, p], verify_by_width[4][:, p]
                ).item():
                    forced_prefix_exact = False

            # Near-tie argmax stability against the stock path, per position.
            stock = _stock_verify_output(
                mx.contiguous(qkv[:, :max_w, :]),
                mx.contiguous(z[:, :max_w, :]),
                mx.contiguous(b[:, :max_w, :]),
                mx.contiguous(a[:, :max_w, :]),
                init_conv, conv_weight, A_log, dt_bias, norm_weight, init_state,
            )
            mx.eval(stock)
            v_out = verify_by_width[max_w]
            for p in range(max_w):
                svec = stock[:, p].reshape(fused_gdn.VALUE_DIM)
                gap = _top2_gap(svec)
                min_gap_seen = min(min_gap_seen, gap)
                s_arg = int(mx.argmax(svec).item())
                f_arg = int(mx.argmax(v_out[:, p].reshape(fused_gdn.VALUE_DIM)).item())
                d_arg = int(mx.argmax(decode_positions[p]).item())
                if f_arg != s_arg or d_arg != s_arg:
                    argmax_stable = False
                    min_flip_margin = min(min_flip_margin, gap)

        verdict = "PASS" if (cross_width_exact and forced_prefix_exact and argmax_stable) else "FLIP"
        print(
            "\n[qwen4 fused GDN verify reduction-width guard]"
            f"\n  threadgroup_y            : {ty}"
            f"\n  blocks x widths          : {blocks} x {widths}"
            f"\n  cross-width bit-exact    : {cross_width_exact}"
            f"\n  forced-prefix bit-exact  : {forced_prefix_exact}"
            f"\n  argmax stable vs stock   : {argmax_stable}"
            f"\n  min top-2 margin seen    : {min_gap_seen:.3e}"
            f"\n  margin at first flip     : "
            f"{'none' if math.isinf(min_flip_margin) else f'{min_flip_margin:.3e}'}"
            f"\n  VERDICT                  : {verdict}"
        )

        assert cross_width_exact, (
            "reduction width is M-dependent: a verify position moved with S "
            "(Class-C exposure) -- see module docstring"
        )
        assert forced_prefix_exact, (
            "S=2 prefix diverged from the S=4 prefix: the token-loop trip count "
            "changed the arithmetic of an earlier position"
        )
        assert argmax_stable, (
            "a near-tie top-2 decision flipped between fused/decode/stock; "
            f"first flip at top-2 margin {min_flip_margin:.3e}"
        )
    finally:
        mx.set_default_device(previous)
