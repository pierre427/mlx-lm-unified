import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models import switch_layers as sl


class TestSortedGatherTailGuard(unittest.TestCase):
    """The >32768-row misaligned sorted-gather corruption (mlx#3922 family).

    Metal's affine gather_qmm NAX path drops tail-tile accumulations when the
    flattened row count exceeds 32768 and is not a multiple of 64. Only the
    mxfp4 manifestation was fixed upstream in 0.32.0; the affine variant
    reproduces on 0.32.0.dev20260708 on M5, so the pad guard is unconditional.
    """

    def _dispatch(self, tokens, top_k, E=8, K=256, N=256):
        mx.random.seed(0)
        w = mx.random.normal((E, N, K)).astype(mx.float16)
        qw, sc, bi = mx.quantize(w, group_size=64, bits=4)
        mx.random.seed(1)
        x = mx.random.normal((tokens, 1, 1, K)).astype(mx.float16)
        idx = mx.sort(mx.random.randint(0, E, (tokens, top_k)))
        xs, iss, inv = sl._gather_sort(x, idx)
        out = mx.gather_qmm(
            xs,
            qw,
            sc,
            bi,
            rhs_indices=iss,
            transpose=True,
            group_size=64,
            bits=4,
            sorted_indices=True,
        )
        out = sl._scatter_unsort(out, inv, idx.shape)
        mx.eval(out)

        wd = mx.dequantize(qw, sc, bi, group_size=64, bits=4).astype(mx.float32)
        xf = x.astype(mx.float32).reshape(tokens, K)
        stride = 997
        ref = mx.stack(
            [
                mx.stack([xf[t] @ wd[int(idx[t, j])].T for j in range(top_k)])
                for t in range(0, tokens, stride)
            ]
        )
        got = out.reshape(tokens, top_k, N).astype(mx.float32)[::stride]
        return mx.abs(got - ref).max().item()

    def test_guard_is_unconditional(self):
        self.assertTrue(sl._SORTED_GATHER_TAIL_BUG)

    @unittest.skipUnless(mx.metal.is_available(), "requires Metal")
    def test_misaligned_rows_past_32768_are_exact(self):
        # 8193 tokens x top-4 = 32772 rows: > 32768 and % 64 != 0.
        err = self._dispatch(8193, 4)
        self.assertLess(err, 0.1)

    @unittest.skipUnless(mx.metal.is_available(), "requires Metal")
    def test_aligned_rows_stay_exact(self):
        err = self._dispatch(8192, 4)
        self.assertLess(err, 0.1)


class TestQuantizedGatherKTailGuard(unittest.TestCase):
    """Lossless mlx#3912/#4009 fallbacks at the unified MoE boundary."""

    def test_policy_covers_both_upstream_tail_families(self):
        policy = sl._quantized_gather_tail_policy
        self.assertEqual(policy("nvfp4", 80, False), "dense")
        self.assertEqual(policy("nvfp4", 80, True), "dense")
        self.assertEqual(policy("affine", 160, True), "unsorted")
        self.assertEqual(policy("mxfp4", 160, True), "unsorted")
        self.assertEqual(policy("affine", 160, False), "native")
        self.assertEqual(policy("nvfp4", 96, False), "native")
        self.assertEqual(policy("affine", 192, True), "native")

    def test_sorted_unaligned_k_declines_only_the_sorted_optimization(self):
        layer = sl.QuantizedSwitchLinear(
            160, 64, 4, bias=False, group_size=32, bits=4, mode="affine"
        )
        mx.random.seed(11)
        x = mx.random.normal((20, 1, 160))
        indices = mx.sort(mx.arange(20, dtype=mx.uint32) % 4)

        with patch.object(mx, "gather_qmm", wraps=mx.gather_qmm) as gather_qmm:
            out = layer(x, indices, sorted_indices=True)
            mx.eval(out)

        self.assertFalse(gather_qmm.call_args.kwargs["sorted_indices"])
        self.assertEqual(out.shape, (20, 1, 64))
        reference = mx.gather_qmm(
            x,
            layer.weight,
            layer.scales,
            layer.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
            sorted_indices=False,
        )
        self.assertTrue(mx.array_equal(out, reference))

    def test_nvfp4_16_wide_tail_uses_dense_reference(self):
        layer = sl.QuantizedSwitchLinear(
            80, 64, 4, bias=False, group_size=16, bits=4, mode="nvfp4"
        )
        mx.random.seed(12)
        x = mx.random.normal((8, 1, 80))
        indices = mx.arange(8, dtype=mx.uint32) % 4

        with (
            patch.object(mx, "gather_qmm", wraps=mx.gather_qmm) as gather_qmm,
            patch.object(mx, "gather_mm", wraps=mx.gather_mm) as gather_mm,
        ):
            out = layer(x, indices, sorted_indices=False)
            mx.eval(out)

        gather_qmm.assert_not_called()
        gather_mm.assert_called_once()
        self.assertEqual(out.shape, (8, 1, 64))
        weight = mx.dequantize(
            layer.weight,
            layer.scales,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )
        reference = mx.gather_mm(
            x,
            weight.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=False,
        )
        self.assertTrue(mx.array_equal(out, reference))


if __name__ == "__main__":
    unittest.main()
