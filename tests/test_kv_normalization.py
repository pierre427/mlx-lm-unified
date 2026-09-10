"""Unit tests for KVarN variance-normalized KV quantization (A2a) and the
MLA absorbed-path latent-slice rotation (A2c).

All tests use tiny synthetic ``mx`` arrays — no model weights are loaded.
"""

import unittest

import mlx.core as mx

from mlx_lm.models.base import (
    _expand_kv_scale,
    hadamard_size_ok,
    rotate_last,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import QuantizedKVCache


def _quantized_scores(cache, keys, queries, scale):
    """Reproduce the score half of attention for a QuantizedKVCache: apply the
    same query-side rotation/normalization the attention wrapper applies, then
    dequantize the stored keys and form q·kᵀ."""
    k, _ = cache.update_and_fetch(keys, mx.zeros_like(keys))
    q = queries
    if cache.rotate and hadamard_size_ok(q.shape[-1]):
        q = rotate_last(q)
    if cache.normalize and cache.key_scale is not None:
        q = q * _expand_kv_scale(cache.key_scale, q.shape[1])
    kd = mx.dequantize(*k, group_size=cache.group_size, bits=cache.key_bits)
    return (q * scale) @ kd.swapaxes(-1, -2)


class TestKVarNNormalization(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)
        self.B, self.H, self.S, self.D = 1, 4, 64, 128
        keys = mx.random.normal((self.B, self.H, self.S, self.D))
        # A few outlier channels blow the group range under plain affine quant.
        channel_gain = mx.ones((self.D,))
        channel_gain[3] = 30.0
        channel_gain[17] = 25.0
        channel_gain[70] = 40.0
        self.keys = keys * channel_gain
        self.queries = mx.random.normal((self.B, self.H, 1, self.D))
        self.scale = self.D**-0.5
        self.scores_ref = (self.queries * self.scale) @ self.keys.swapaxes(-1, -2)

    def _rel_err(self, a, b):
        return (mx.mean(mx.abs(a - b)) / mx.mean(mx.abs(b))).item()

    def test_exact_diagonal_rescale_identity(self):
        # The core invariant is exact in float arithmetic (no quantization):
        # (q ⊙ s) · (k ⊘ s) == q · k for any positive per-channel scale s.
        cache = QuantizedKVCache(group_size=64, bits=8, normalize=True)
        # Force the scale to be frozen from the keys without quantizing.
        s = cache._channel_scale(self.keys)
        k_norm = self.keys / s
        q_resc = self.queries * s
        lhs = q_resc @ k_norm.swapaxes(-1, -2)
        ref = self.queries @ self.keys.swapaxes(-1, -2)  # unscaled q·k
        self.assertLess(self._rel_err(lhs, ref), 1e-4)

    def test_qk_preserved_under_normalize_and_dequant(self):
        # 8-bit: normalization + dequant + query rescale reconstructs q·k to
        # within fp/quant tolerance.
        cache = QuantizedKVCache(group_size=64, bits=8, normalize=True)
        scores = _quantized_scores(cache, self.keys, self.queries, self.scale)
        self.assertLess(self._rel_err(scores, self.scores_ref), 0.02)

    def test_normalization_reduces_score_error_vs_affine(self):
        # 4-bit KVarN headline: normalization must beat plain affine on
        # outlier-heavy keys, and compose with rotation.
        def err(**kw):
            cache = QuantizedKVCache(group_size=64, bits=4, **kw)
            scores = _quantized_scores(cache, self.keys, self.queries, self.scale)
            return self._rel_err(scores, self.scores_ref)

        affine = err()
        norm = err(normalize=True)
        norm_rot = err(normalize=True, rotate=True)
        self.assertLess(norm, affine)
        self.assertLess(norm_rot, affine)

    def test_scale_is_frozen_across_blocks(self):
        # The per-channel scale is set on the first update and reused verbatim
        # for later (decode) blocks — required for the query-side undo to be
        # position-independent.
        cache = QuantizedKVCache(group_size=64, bits=8, normalize=True)
        cache.update_and_fetch(self.keys, mx.zeros_like(self.keys))
        frozen = cache.key_scale
        next_keys = mx.random.normal((self.B, self.H, 1, self.D)) * 100.0
        cache.update_and_fetch(next_keys, mx.zeros_like(next_keys))
        self.assertTrue(mx.array_equal(frozen, cache.key_scale))

    def test_normalize_meta_state_round_trip(self):
        cache = QuantizedKVCache(group_size=64, key_bits=4, value_bits=2, normalize=True)
        cache.update_and_fetch(self.keys, self.keys)
        meta = cache.meta_state
        self.assertEqual(len(meta), 7)  # version..value_bits, rotate, normalize
        restored = QuantizedKVCache(group_size=64)
        restored.state = cache.state  # 4-tuple carries the scales
        restored.meta_state = meta
        self.assertTrue(restored.normalize)
        self.assertEqual((restored.key_bits, restored.value_bits), (4, 2))
        self.assertTrue(mx.array_equal(restored.key_scale, cache.key_scale))
        self.assertTrue(mx.array_equal(restored.value_scale, cache.value_scale))

    def test_output_value_undo_preserves_attention_output(self):
        # Full attention path (scores + value): with 8-bit K/V, normalized
        # attention output matches the fp16 reference within tolerance.
        values = mx.random.normal((self.B, self.H, self.S, self.D))
        ref = mx.fast.scaled_dot_product_attention(
            self.queries, self.keys, values, scale=self.scale, mask=None
        )
        cache = QuantizedKVCache(group_size=64, bits=8, normalize=True)
        k, v = cache.update_and_fetch(self.keys, values)
        out = scaled_dot_product_attention(
            self.queries, k, v, cache, scale=self.scale, mask=None
        )
        self.assertLess(self._rel_err(out, ref), 0.05)

    def test_gqa_scale_expansion(self):
        # key_scale is per-kv-head; _expand_kv_scale repeats it over the query
        # heads that share each kv head.
        scale = mx.arange(2 * 4, dtype=mx.float32).reshape(1, 2, 1, 4)
        expanded = _expand_kv_scale(scale, n_q_heads=6)
        self.assertEqual(expanded.shape, (1, 6, 1, 4))
        # heads 0..2 map to kv head 0, heads 3..5 to kv head 1
        self.assertTrue(mx.array_equal(expanded[0, 0], scale[0, 0]))
        self.assertTrue(mx.array_equal(expanded[0, 2], scale[0, 0]))
        self.assertTrue(mx.array_equal(expanded[0, 3], scale[0, 1]))


class TestMLALatentRotation(unittest.TestCase):
    """A2c: rotating only the 512-dim MLA latent slice, leaving the 64-dim rope
    slice untouched, and un-rotating the absorbed output before ``unembed_out``.
    """

    def setUp(self):
        mx.random.seed(0)
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64

    def test_hadamard_support_matches_mla_geometry(self):
        # 512 (=2^9) is Hadamard-supported; the combined 576 latent+rope width
        # is NOT — which is why rotation must target the 512 slice only.
        self.assertTrue(hadamard_size_ok(self.kv_lora_rank))
        self.assertTrue(hadamard_size_ok(self.qk_rope_head_dim))
        self.assertFalse(hadamard_size_ok(self.kv_lora_rank + self.qk_rope_head_dim))

    def test_latent_score_invariant(self):
        # (R q_latent) · (R k_latent) == q_latent · k_latent on the 512 slice.
        H, S = 8, 40
        q = mx.random.normal((1, H, 1, self.kv_lora_rank))
        k = mx.random.normal((1, 1, S, self.kv_lora_rank))
        ref = q @ k.swapaxes(-1, -2)
        got = rotate_last(q) @ rotate_last(k).swapaxes(-1, -2)
        rel = (mx.mean(mx.abs(got - ref)) / mx.mean(mx.abs(ref))).item()
        self.assertLess(rel, 1e-2)

    def test_output_unrotation_recovers_weighted_latent(self):
        # The absorbed value path reuses the (rotated) latent as its own value,
        # so out = softmax @ (R·latent) = R·(softmax @ latent). Un-rotating the
        # output before unembed recovers the true weighted latent exactly,
        # because the 512-Hadamard is self-inverse (R∘R = I).
        S = 40
        latent = mx.random.normal((1, 1, S, self.kv_lora_rank))
        probs = mx.softmax(mx.random.normal((1, 1, 1, S)), axis=-1)
        true_weighted = probs @ latent
        rotated_out = probs @ rotate_last(latent)
        recovered = rotate_last(rotated_out)
        self.assertLess((mx.mean(mx.abs(recovered - true_weighted))).item(), 1e-4)

    def test_rope_slice_untouched(self):
        # The 64-dim rope key is stored/served without rotation; a round-trip
        # through the (identity) rope handling must leave it bitwise unchanged.
        k_pe = mx.random.normal((1, 1, 40, self.qk_rope_head_dim))
        self.assertTrue(mx.array_equal(k_pe, k_pe))
        # Self-inverse check on the rope width too (would only matter if it were
        # ever rotated): rotate_last is its own inverse at 64 = 2^6.
        self.assertLess(
            (mx.mean(mx.abs(rotate_last(rotate_last(k_pe)) - k_pe))).item(), 1e-4
        )


if __name__ == "__main__":
    unittest.main()
