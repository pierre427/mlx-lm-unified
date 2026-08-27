# Copyright © 2026 Apple Inc.

"""GDN q/k l2norm parity against the HF/FLA reference.

The reference (transformers ``models/qwen3_5``, ``qwen3_next``, ``qwen4_exp``,
all identical) normalizes with::

    inv_norm = rsqrt((x * x).sum(-1, keepdims=True) + 1e-6)
    x * inv_norm

and then applies ``scale = head_dim ** -0.5`` to the QUERY ONLY, inside the
delta-rule call. ``mx.fast.rms_norm(x, None, eps)`` adds eps to the MEAN of
squares, so the epsilon handed to it must be ``1e-6 / head_dim``. Passing
``1e-6`` straight through inflated the effective epsilon by head_dim.
"""

import unittest
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

import mlx_lm.models.gated_delta as gated_delta
from mlx_lm.models.gated_delta import normalize_gdn_qk
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs


@contextmanager
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def reference_l2norm(x, eps=1e-6):
    """Verbatim port of the reference ``l2norm``."""
    x = x.astype(mx.float32)
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def reference_normalize_qk(q, k):
    """Reference l2norm + the delta rule's query-only ``scale``."""
    scale = q.shape[-1] ** -0.5
    return reference_l2norm(q) * scale, reference_l2norm(k)


def _at_scale(shape, target_sq_sum, seed):
    """Random vectors whose per-row sum(x^2) is exactly ``target_sq_sum``."""
    mx.random.seed(seed)
    x = mx.random.normal(shape).astype(mx.float32)
    return x * mx.sqrt(target_sq_sum / (x * x).sum(axis=-1, keepdims=True))


# Activation scales spanning the regime where the inflated epsilon bit hardest.
SQ_SUMS = [130.0, 13.0, 1.3, 0.13, 0.014, 0.001]


class TestGDNQKL2Norm(unittest.TestCase):
    def test_matches_reference_across_activation_scales(self):
        """Direct numeric parity with the reference at every scale."""
        with cpu_device():
            for head_dim in (32, 64, 128):
                for sq_sum in SQ_SUMS:
                    with self.subTest(head_dim=head_dim, sq_sum=sq_sum):
                        q = _at_scale((2, 5, 4, head_dim), sq_sum, seed=0)
                        k = _at_scale((2, 5, 4, head_dim), sq_sum, seed=1)
                        got_q, got_k = normalize_gdn_qk(q, k)
                        want_q, want_k = reference_normalize_qk(q, k)
                        # Algebraically exact; only fp32 rounding separates the
                        # rms_norm route from rsqrt-on-sum (a few ulp).
                        self.assertLess(_max_rel(got_q, want_q), 1e-6)
                        self.assertLess(_max_rel(got_k, want_k), 1e-6)

    def test_query_only_scale_folding(self):
        """The extra inv_scale on q reproduces the reference's separate scale.

        The reference applies ``scale`` to the query only, never the key, and
        our kernel applies no scale of its own -- so q carries ``inv_scale**2``
        (one factor of normalization, one of scale) and k carries ``inv_scale``.
        """
        with cpu_device():
            head_dim = 128
            q = _at_scale((1, 3, 2, head_dim), 4.0, seed=2)
            k = _at_scale((1, 3, 2, head_dim), 4.0, seed=3)
            got_q, got_k = normalize_gdn_qk(q, k)
            # q is exactly the key-side normalization times scale.
            ratio = got_q / (reference_l2norm(q) * (head_dim**-0.5))
            self.assertLess(float(mx.max(mx.abs(ratio - 1.0))), 1e-6)
            # k carries NO scale factor: it is unit-norm.
            norms = mx.sqrt((got_k * got_k).sum(axis=-1))
            self.assertLess(float(mx.max(mx.abs(norms - 1.0))), 1e-5)

    def test_effective_epsilon_is_1e6_on_the_sum(self):
        """Regression guard: the head_dim-scaling trap cannot silently return.

        Recovers the epsilon empirically from a vector of known norm and
        asserts it is 1e-6 on the SUM of squares -- not head_dim * 1e-6.
        """
        with cpu_device():
            for head_dim in (32, 64, 128):
                with self.subTest(head_dim=head_dim):
                    # A tiny-norm vector makes epsilon dominate and observable.
                    val = 1e-3
                    k = mx.full((1, 1, 1, head_dim), val, dtype=mx.float32)
                    _, got_k = normalize_gdn_qk(k, k)
                    out = float(got_k[0, 0, 0, 0])
                    sq_sum = val * val * head_dim
                    # out = val / sqrt(sq_sum + eps)  =>  solve for eps.
                    eps_eff = (val / out) ** 2 - sq_sum
                    self.assertAlmostEqual(eps_eff, 1e-6, delta=2e-8)
                    # Explicitly exclude the old, head_dim-inflated value.
                    self.assertLess(eps_eff, head_dim * 1e-6 / 2)


class TestGDNLayerParity(unittest.TestCase):
    """Whole-layer parity where the old form deviated most."""

    def _tiny_args(self, head_dim=32):
        return TextModelArgs(
            model_type="qwen3_5_text",
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=head_dim,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
            linear_conv_kernel_dim=4,
            vocab_size=128,
        )

    @contextmanager
    def _patched_norm(self, fn):
        """Swap the normalization the layer calls, via the shared helper."""
        import mlx_lm.models.qwen3_5 as qwen3_5

        original = qwen3_5.normalize_gdn_qk
        qwen3_5.normalize_gdn_qk = fn
        try:
            yield
        finally:
            qwen3_5.normalize_gdn_qk = original

    def test_layer_output_matches_reference_at_small_magnitudes(self):
        with cpu_device():
            layer = GatedDeltaNet(self._tiny_args())
            mx.eval(layer.parameters())
            mx.random.seed(7)
            # Small activations -> small q/k norms, the worst case for the bug.
            x = (mx.random.normal((1, 12, 128)) * 1e-2).astype(mx.float32)

            ours = layer(x)
            with self._patched_norm(reference_normalize_qk):
                want = layer(x)
            mx.eval(ours, want)
            self.assertLess(_rel_l2(ours, want), 1e-5)

    def test_layer_detects_the_old_inflated_epsilon(self):
        """The test above is load-bearing: the OLD form fails it."""

        def old_form(q, k):
            inv_scale = k.shape[-1] ** -0.5
            return (
                (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6),
                inv_scale * mx.fast.rms_norm(k, None, 1e-6),
            )

        with cpu_device():
            layer = GatedDeltaNet(self._tiny_args())
            mx.eval(layer.parameters())
            mx.random.seed(7)
            x = (mx.random.normal((1, 12, 128)) * 1e-2).astype(mx.float32)

            with self._patched_norm(reference_normalize_qk):
                want = layer(x)
            with self._patched_norm(old_form):
                old = layer(x)
            mx.eval(want, old)
            # The old form is orders of magnitude further from the reference.
            self.assertGreater(_rel_l2(old, want), 1e-3)


def _max_rel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return float(mx.max(mx.abs(a - b) / (mx.abs(b) + 1e-30)))


def _rel_l2(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-9)).item()


if __name__ == "__main__":
    unittest.main()
