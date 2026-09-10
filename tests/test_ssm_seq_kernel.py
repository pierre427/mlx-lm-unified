import unittest

import mlx.core as mx

from mlx_lm.models.ssm import compute_dt, ssm_update_seq_kernel


def exact_ref(x, A_log, B, C, D, dt, dt_bias, state, tsl):
    """Position-at-a-time fp32 recurrence, the same math as the S=1 step
    kernel; the sequential kernel must match it to float rounding."""
    b, S, h, dh = x.shape
    g = B.shape[2]
    dtc = compute_dt(dt, dt_bias, tsl)
    A = -mx.exp(A_log)
    rep = h // g
    ys = []
    st = state
    for s in range(S):
        dA = mx.exp(A * dtc[:, s])
        Bs = mx.repeat(B[:, s], rep, axis=1)
        Cs = mx.repeat(C[:, s], rep, axis=1)
        xs = x[:, s]
        dBx = dtc[:, s][..., None, None] * xs[..., None] * Bs[:, :, None, :]
        st = dA[..., None, None] * st + dBx
        ys.append((st * Cs[:, :, None, :]).sum(-1) + xs * D[None, :, None])
    return mx.stack(ys, axis=1), st


@unittest.skipUnless(mx.metal.is_available(), "metal only")
class TestSSMSeqKernel(unittest.TestCase):
    def test_matches_exact_recurrence(self):
        mx.random.seed(0)
        b, h, dh, g, ds = 1, 8, 16, 2, 32
        tsl = (0.001, 100.0)
        for S in (2, 3, 4, 5, 8):
            x = mx.random.normal((b, S, h, dh)) * 0.5
            B = mx.random.normal((b, S, g, ds)) * 0.3
            C = mx.random.normal((b, S, g, ds)) * 0.3
            dt = mx.random.normal((b, S, h)) * 0.1
            A_log = mx.random.normal((h,)) * 0.5
            D = mx.abs(mx.random.normal((h,)))
            dt_bias = mx.random.normal((h,)) * 0.1
            state = mx.random.normal((b, h, dh, ds)) * 0.2

            y_e, s_e = exact_ref(x, A_log, B, C, D, dt, dt_bias, state, tsl)
            y_k, s_k = ssm_update_seq_kernel(
                x, A_log, B, C, D, dt, dt_bias, state, tsl
            )
            self.assertTrue(
                mx.allclose(y_e, y_k, atol=1e-5).item(), f"y mismatch at S={S}"
            )
            self.assertTrue(
                mx.allclose(s_e, s_k, atol=1e-5).item(), f"state mismatch at S={S}"
            )


if __name__ == "__main__":
    unittest.main()
