import unittest

import mlx.core as mx

import mlx_lm.models.gated_delta as gated_delta
from mlx_lm.models.gated_delta import (
    gated_delta_kernel,
    gated_delta_kernel_unpacked,
    gated_delta_kernel_xtree,
    gated_delta_ops,
)


def _normed(shape, D, dtype):
    x = mx.random.normal(shape)
    return (mx.fast.rms_norm(x, None, 1e-6) * D**-0.5).astype(dtype)


def _rel_l2(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-9)).item()


class TestGatedDelta(unittest.TestCase):
    def setUp(self):
        self._packed = gated_delta._ENABLE_GDN_PACKED
        self._core = gated_delta._ENABLE_GDN_CORE
        self._core_fn = gated_delta._core_gated_delta_update
        gated_delta._ENABLE_GDN_PACKED = True
        gated_delta._ENABLE_GDN_CORE = False

    def tearDown(self):
        gated_delta._ENABLE_GDN_PACKED = self._packed
        gated_delta._ENABLE_GDN_CORE = self._core
        gated_delta._core_gated_delta_update = self._core_fn

    def test_core_adapter_eligibility_is_narrow(self):
        gated_delta._ENABLE_GDN_CORE = True
        gated_delta._core_gated_delta_update = lambda *args, **kwargs: None
        q, k, v, g, _, state = self._inputs(
            1, 64, 16, 32, 128, 128, mx.bfloat16
        )
        self.assertTrue(
            gated_delta._can_use_core_gated_delta(q, k, v, g, state, None)
        )
        self.assertFalse(
            gated_delta._can_use_core_gated_delta(
                q, k, v, g[..., None], state, None
            )
        )
        self.assertFalse(
            gated_delta._can_use_core_gated_delta(
                q, k, v, g, state, mx.ones((1, 64), dtype=mx.bool_)
            )
        )
        short = tuple(x[:, :1] if x.ndim >= 2 else x for x in (q, k, v, g))
        self.assertFalse(
            gated_delta._can_use_core_gated_delta(*short, state, None)
        )
        long = tuple(
            mx.repeat(x, 5, axis=1) if x.ndim >= 2 else x
            for x in (q, k, v, g)
        )
        self.assertFalse(
            gated_delta._can_use_core_gated_delta(*long, state, None)
        )

    def test_core_adapter_passes_explicit_chunk_policy(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        calls = []

        def fake_core(q, k, v, g, beta, **kwargs):
            calls.append(kwargs)
            return mx.zeros_like(v), kwargs["initial_state"]

        gated_delta._ENABLE_GDN_CORE = True
        gated_delta._core_gated_delta_update = fake_core
        q, k, v, _, _, state = self._inputs(
            1, 64, 16, 32, 128, 128, mx.bfloat16
        )
        a = mx.zeros((1, 64, 32))
        b = mx.zeros_like(a)
        A_log = mx.zeros((32,))
        dt_bias = mx.zeros((32,))
        gated_delta.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["chunk_size"], 8)

    def test_update_keeps_bf16_sigmoid_in_float32(self):
        previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            T = 1000
            mx.random.seed(53877)
            q = (mx.random.normal((1, T, 1, 1)) * 0.2).astype(mx.bfloat16)
            k = (mx.random.normal((1, T, 1, 1)) * 0.2).astype(mx.bfloat16)
            v = (mx.random.normal((1, T, 1, 1)) * 0.2).astype(mx.bfloat16)
            a = (mx.random.normal((1, T, 1)) * 0.2 - 2.0).astype(mx.bfloat16)
            b = (mx.random.normal((1, T, 1)) * 0.7 + 0.5).astype(mx.bfloat16)
            A_log = mx.zeros((1,), dtype=mx.float32)
            dt_bias = mx.zeros((1,), dtype=mx.float32)
            state = mx.zeros((1, 1, 1, 1), dtype=mx.float32)

            y, final_state = gated_delta.gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                use_kernel=False,
            )
            g = gated_delta.compute_g(A_log, a, dt_bias)
            y_ref, state_ref = gated_delta_ops(
                q, k, v, g, mx.sigmoid(b.astype(mx.float32)), state
            )
            mx.eval(y, final_state, y_ref, state_ref)

            self.assertLess(_rel_l2(y, y_ref), 1e-7)
            self.assertLess(_rel_l2(final_state, state_ref), 1e-7)
        finally:
            mx.set_default_device(previous_device)

    def test_kill_switch_restores_original_kernel(self):
        # MLX_GDN_PACKED=0 routes back to the pre-existing simd_sum kernel.
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        gated_delta._ENABLE_GDN_PACKED = False
        args = self._inputs(1, 64, 16, 32, 128, 128, mx.bfloat16)
        y1, s1 = gated_delta_kernel(*args, None)
        y2, s2 = gated_delta_kernel_unpacked(*args, None)
        mx.eval(y1, s1, y2, s2)
        self.assertTrue(mx.array_equal(y1, y2))
        self.assertTrue(mx.array_equal(s1, s2))

    def _inputs(self, B, T, Hk, Hv, Dk, Dv, dtype):
        mx.random.seed(3)
        q = _normed((B, T, Hk, Dk), Dk, dtype)
        k = _normed((B, T, Hk, Dk), Dk, dtype)
        v = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
        # decays in (0, 1], as produced by compute_g
        g = mx.exp(-mx.random.uniform(shape=(B, T, Hv)) * 0.2).astype(mx.float32)
        beta = mx.random.uniform(shape=(B, T, Hv)).astype(dtype)
        state = (mx.random.normal((B, Hv, Dv, Dk)) * 0.3).astype(mx.float32)
        mx.eval(q, k, v, g, beta, state)
        return q, k, v, g, beta, state

    def test_packed_matches_unpacked(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")

        cases = [
            # B, Hk, Hv, Dk, Dv, dtype
            (1, 16, 32, 128, 128, mx.bfloat16),  # Qwen3.5/3.6 shape
            (2, 16, 32, 128, 128, mx.bfloat16),  # batched
            (1, 4, 8, 128, 128, mx.bfloat16),  # fewer heads
            (1, 8, 8, 128, 128, mx.bfloat16),  # Hv == Hk
            (1, 16, 32, 128, 128, mx.float16),
            (1, 16, 32, 128, 128, mx.float32),
            (1, 8, 16, 128, 64, mx.bfloat16),  # Dv != Dk
            (3, 2, 8, 128, 256, mx.bfloat16),  # larger Dv
        ]
        for B, Hk, Hv, Dk, Dv, dtype in cases:
            for T in (1, 7, 64, 257):  # decode, ragged, aligned, spilling
                with self.subTest(B=B, Hk=Hk, Hv=Hv, Dv=Dv, dtype=dtype, T=T):
                    args = self._inputs(B, T, Hk, Hv, Dk, Dv, dtype)
                    y_p, s_p = gated_delta_kernel(*args, None)
                    y_x, s_x = gated_delta_kernel_xtree(*args, None)
                    mx.eval(y_p, s_p, y_x, s_x)
                    # The packed kernel reproduces the comparator's explicit
                    # reduction tree, so this holds on any device
                    # independently of how simd_sum lowers.
                    self.assertTrue(mx.array_equal(y_p, y_x))
                    self.assertTrue(mx.array_equal(s_p, s_x))

    def test_packed_matches_ops_reference(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")

        q, k, v, g, beta, state = self._inputs(1, 64, 16, 32, 128, 128, mx.bfloat16)
        y_p, s_p = gated_delta_kernel(q, k, v, g, beta, state, None)
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, None)
        mx.eval(y_p, s_p, y_r, s_r)
        self.assertLess(_rel_l2(y_p, y_r), 2e-3)
        self.assertLess(_rel_l2(s_p, s_r), 2e-3)

    def test_explicit_tree_matches_simd_sum_kernel(self):
        # On all current Apple GPUs simd_sum lowers to the same ascending
        # butterfly the comparator writes out, so the packed kernel is also
        # bit-identical to the pre-existing kernel. If this ever fails on a
        # new device or toolchain, the packed default should be revisited
        # (its contract vs the comparator still holds).
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        args = self._inputs(1, 257, 16, 32, 128, 128, mx.bfloat16)
        y_x, s_x = gated_delta_kernel_xtree(*args, None)
        y_u, s_u = gated_delta_kernel_unpacked(*args, None)
        mx.eval(y_x, s_x, y_u, s_u)
        self.assertTrue(mx.array_equal(y_x, y_u))
        self.assertTrue(mx.array_equal(s_x, s_u))

    def test_masked_generic_matches_ops_reference(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, g, beta, state = self._inputs(2, 33, 8, 16, 128, 128, mx.bfloat16)
        mask = mx.arange(33)[None] < mx.array([[29], [17]])
        y_k, s_k = gated_delta_kernel(q, k, v, g, beta, state, mask)
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, mask)
        mx.eval(y_k, s_k, y_r, s_r)
        # Outputs at padded positions are unspecified (the kernel zeros
        # them, the ops reference does not); compare valid positions only.
        valid = mask[..., None, None]
        y_k = mx.where(valid, y_k, 0)
        y_r = mx.where(valid, y_r, 0)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_vector_gate_generic_matches_ops_reference(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, _, beta, state = self._inputs(1, 65, 4, 8, 128, 128, mx.bfloat16)
        g = mx.exp(-mx.random.uniform(shape=(1, 65, 8, 128)) * 0.2).astype(mx.float32)
        mx.eval(g)
        y_k, s_k = gated_delta_kernel(q, k, v, g, beta, state, None)
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, None)
        mx.eval(y_k, s_k, y_r, s_r)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_small_head_dim_generic_matches_ops_reference(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, g, beta, state = self._inputs(1, 65, 4, 8, 64, 64, mx.bfloat16)
        y_k, s_k = gated_delta_kernel(q, k, v, g, beta, state, None)
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, None)
        mx.eval(y_k, s_k, y_r, s_r)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_unsupported_shapes_fall_back(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")

        # Dk != 128 must take the general kernel and stay exact against it.
        q, k, v, g, beta, state = self._inputs(1, 32, 4, 8, 64, 64, mx.bfloat16)
        y_p, s_p = gated_delta_kernel(q, k, v, g, beta, state, None)
        y_u, s_u = gated_delta_kernel_unpacked(q, k, v, g, beta, state, None)
        mx.eval(y_p, s_p, y_u, s_u)
        self.assertTrue(mx.array_equal(y_p, y_u))
        self.assertTrue(mx.array_equal(s_p, s_u))

        # A padding mask also forces the general kernel.
        q, k, v, g, beta, state = self._inputs(1, 16, 16, 32, 128, 128, mx.bfloat16)
        mask = mx.ones((1, 16), dtype=mx.bool_)
        y_p, s_p = gated_delta_kernel(q, k, v, g, beta, state, mask)
        y_u, s_u = gated_delta_kernel_unpacked(q, k, v, g, beta, state, mask)
        mx.eval(y_p, s_p, y_u, s_u)
        self.assertTrue(mx.array_equal(y_p, y_u))
        self.assertTrue(mx.array_equal(s_p, s_u))


class TestGatedDeltaHeadMapping(unittest.TestCase):
    """Pin the value-head -> key-head mapping used when Hv > Hk.

    Every GDN model we serve has Hv > Hk (16/32, 16/48, 16/64), so the
    mapping is live. The HuggingFace reference for both qwen3_next and
    qwen3_5 expands q/k with ``repeat_interleave``, i.e. value head hv
    reads key head ``hv // (Hv // Hk)`` (blocked). The alternative tiled
    mapping ``hv % Hk`` gives a completely different result.

    The kernel-vs-ops tests cannot see a flip of this convention because
    both paths share it. These cases compare each path against an
    explicit pre-expansion instead, so a flip fails here.
    """

    def _inputs(self, B, T, Hk, Hv, Dk, Dv, dtype):
        mx.random.seed(11)
        q = _normed((B, T, Hk, Dk), Dk, dtype)
        k = _normed((B, T, Hk, Dk), Dk, dtype)
        v = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
        g = mx.exp(-mx.random.uniform(shape=(B, T, Hv)) * 0.2).astype(mx.float32)
        beta = mx.random.uniform(shape=(B, T, Hv)).astype(dtype)
        state = (mx.random.normal((B, Hv, Dv, Dk)) * 0.3).astype(mx.float32)
        mx.eval(q, k, v, g, beta, state)
        return q, k, v, g, beta, state

    @staticmethod
    def _blocked(x, Hk, Hv):
        return mx.repeat(x, Hv // Hk, -2)

    @staticmethod
    def _tiled(x, Hk, Hv):
        return mx.take(x, mx.array([hv % Hk for hv in range(Hv)]), axis=2)

    def _check(self, fn, Hk, Hv):
        B, T, Dk, Dv = 1, 24, 128, 128
        q, k, v, g, beta, state = self._inputs(B, T, Hk, Hv, Dk, Dv, mx.float32)

        y, s = fn(q, k, v, g, beta, state, None)
        # Pre-expanded calls pass Hk == Hv, so no in-path mapping applies.
        y_b, s_b = fn(
            self._blocked(q, Hk, Hv), self._blocked(k, Hk, Hv), v, g, beta, state, None
        )
        y_t, _ = fn(
            self._tiled(q, Hk, Hv), self._tiled(k, Hk, Hv), v, g, beta, state, None
        )
        mx.eval(y, s, y_b, s_b, y_t)

        self.assertLess(_rel_l2(y, y_b), 1e-5)
        self.assertLess(_rel_l2(s, s_b), 1e-5)
        # The two conventions must be far apart, or the case proves nothing.
        self.assertGreater(_rel_l2(y_b, y_t), 0.1)

    def test_ops_reference_uses_blocked_mapping(self):
        for Hk, Hv in ((16, 32), (16, 48), (16, 64)):
            with self.subTest(Hk=Hk, Hv=Hv):
                self._check(gated_delta_ops, Hk, Hv)

    def test_kernels_use_blocked_mapping(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        for fn in (gated_delta_kernel, gated_delta_kernel_unpacked):
            for Hk, Hv in ((16, 32), (16, 48), (16, 64)):
                with self.subTest(fn=fn.__name__, Hk=Hk, Hv=Hv):
                    self._check(fn, Hk, Hv)


class TestGatedDeltaReadoutRange(unittest.TestCase):
    """Pin the readout against the activation dtype's range.

    The recurrent state is fp32 but the readout leaves in the activation
    dtype. float16 tops out at 65504, and an infinite readout becomes NaN in
    the RMSNormGated that every GDN caller applies next. bfloat16 shares
    float32's exponent, so it must stay untouched.
    """

    B, T, H, D = 1, 8, 1, 128
    # k_t = e_t gives orthogonal keys, so the delta rule never self-corrects
    # and the state keeps one large row per step; q = 1 sums all 8 of them.
    EXPECTED = 8 * 20000.0

    def _inputs(self, dtype):
        k = mx.zeros((self.B, self.T, self.H, self.D))
        for t in range(self.T):
            k[0, t, 0, t] = 1.0
        v = mx.full((self.B, self.T, self.H, self.D), 20000.0)
        q = mx.ones((self.B, self.T, self.H, self.D))
        g = mx.ones((self.B, self.T, self.H), dtype=mx.float32)
        beta = mx.ones((self.B, self.T, self.H), dtype=mx.float32)
        state = mx.zeros((self.B, self.H, self.D, self.D), dtype=mx.float32)
        return q.astype(dtype), k.astype(dtype), v.astype(dtype), g, beta, state

    def _paths(self):
        paths = {"ops": gated_delta_ops}
        if mx.default_device() == mx.gpu:
            paths.update(
                packed=gated_delta_kernel,
                unpacked=gated_delta_kernel_unpacked,
                xtree=gated_delta_kernel_xtree,
            )
        return paths

    def test_construction_exceeds_the_float16_ceiling(self):
        # Without this the fp16 cases below would prove nothing.
        args = self._inputs(mx.float32)
        y, _ = gated_delta_ops(*args, None)
        mx.eval(y)
        self.assertAlmostEqual(mx.max(y).item(), self.EXPECTED, delta=1.0)
        self.assertGreater(self.EXPECTED, mx.finfo(mx.float16).max)

    def test_float16_readout_saturates_instead_of_overflowing(self):
        for name, fn in self._paths().items():
            with self.subTest(path=name):
                y, _ = fn(*self._inputs(mx.float16), None)
                mx.eval(y)
                self.assertEqual(y.dtype, mx.float16)
                self.assertEqual(int(mx.isinf(y).sum().item()), 0)
                self.assertEqual(int(mx.isnan(y).sum().item()), 0)
                self.assertEqual(mx.max(y).item(), mx.finfo(mx.float16).max)

    def test_float16_saturated_readout_survives_the_gated_norm(self):
        # The failure this guards against is a NaN layer output, not the inf.
        from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated

        norm = Qwen3NextRMSNormGated(self.D, eps=1e-6)
        for name, fn in self._paths().items():
            with self.subTest(path=name):
                y, _ = fn(*self._inputs(mx.float16), None)
                out = norm(y, mx.ones_like(y))
                mx.eval(out)
                self.assertEqual(int(mx.isnan(out).sum().item()), 0)
                self.assertEqual(int(mx.isinf(out).sum().item()), 0)

    def test_bfloat16_readout_is_not_clamped(self):
        # bf16 reaches 1.6e5 unaided, so it must take the untouched path.
        for name, fn in self._paths().items():
            with self.subTest(path=name):
                y, _ = fn(*self._inputs(mx.bfloat16), None)
                mx.eval(y)
                self.assertEqual(y.dtype, mx.bfloat16)
                self.assertEqual(int(mx.isinf(y).sum().item()), 0)
                self.assertAlmostEqual(
                    mx.max(y).astype(mx.float32).item(), self.EXPECTED, delta=512.0
                )

    def test_widening_predicate_tracks_exponent_width(self):
        needs = gated_delta._readout_needs_widening
        self.assertTrue(needs(mx.float16, mx.float32))
        self.assertFalse(needs(mx.bfloat16, mx.float32))
        self.assertFalse(needs(mx.float32, mx.float32))
        self.assertFalse(needs(mx.float16, mx.float16))

    def test_saturating_cast_keeps_nan(self):
        # Range clamping must not hide a genuinely diverged state.
        y = mx.array([float("nan"), float("inf"), 1.0], dtype=mx.float32)
        out = gated_delta._cast_readout(y, mx.float16, mx.float32)
        mx.eval(out)
        self.assertTrue(mx.isnan(out[0]).item())
        self.assertEqual(out[1].item(), mx.finfo(mx.float16).max)

    def test_core_adapter_declines_narrow_readout_dtypes(self):
        # mx.fast.gated_delta_update writes its readout without saturating.
        gated_delta._ENABLE_GDN_CORE = True
        previous = gated_delta._core_gated_delta_update
        gated_delta._core_gated_delta_update = lambda *a, **kw: None
        try:
            B, T, Hk, Hv, D = 1, 64, 16, 32, 128
            state = mx.zeros((B, Hv, D, D), dtype=mx.float32)
            g = mx.ones((B, T, Hv), dtype=mx.float32)
            for dtype, expected in ((mx.bfloat16, True), (mx.float16, False)):
                q = mx.zeros((B, T, Hk, D), dtype=dtype)
                v = mx.zeros((B, T, Hv, D), dtype=dtype)
                with self.subTest(dtype=dtype):
                    self.assertEqual(
                        gated_delta._can_use_core_gated_delta(q, q, v, g, state, None),
                        expected,
                    )
        finally:
            gated_delta._ENABLE_GDN_CORE = False
            gated_delta._core_gated_delta_update = previous


if __name__ == "__main__":
    unittest.main()
