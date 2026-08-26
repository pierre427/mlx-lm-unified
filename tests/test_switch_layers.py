import unittest

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


if __name__ == "__main__":
    unittest.main()
