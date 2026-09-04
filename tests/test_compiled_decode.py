# Copyright © 2026 Apple Inc.

"""RingKVCache and the compiled/replayed decode step.

Two levels: the cache on its own (writes, mask, trim, growth, save/load),
and a real hybrid stack driven through ``CompiledDecodeStep``.

The gates that need real Metal -- bit-identity against stock ``KVCache``
through a whole model, and the trace-count proof -- run wherever the tests
run; the numeric ones are asserted as byte equality only where the SDPA
kernel is handed the same key length in both arms (see
``RingKVCache`` docstring: mlx picks its Metal attention kernel from
``k.shape[2]``, so a padded slab past 1024 columns reorders the softmax
reduction). Tests that need that property pin small capacity buckets.
"""

import inspect
import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import compiled_decode as cd
from mlx_lm.models.cache import (
    KVCache,
    RingKVCache,
    load_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)


def _kv(B=1, H=2, S=1, D=8, dtype=mx.float32, seed=None):
    if seed is not None:
        mx.random.seed(seed)
    return (
        mx.random.normal((B, H, S, D)).astype(dtype),
        mx.random.normal((B, H, S, D)).astype(dtype),
    )


class TestRingKVCache(unittest.TestCase):
    def test_writes_match_kv_cache(self):
        """Same keys and values land at the same positions as KVCache."""
        mx.random.seed(3)
        kv, ring = KVCache(), RingKVCache(buckets=(16, 32))
        for step in (5, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1):
            k, v = _kv(S=step)
            a_k, a_v = kv.update_and_fetch(k, v)
            b_k, b_v = ring.update_and_fetch(k, v)
            live = kv.offset
            self.assertEqual(ring.size(), live)
            self.assertEqual(int(ring.offset.item()), live)
            self.assertTrue(mx.array_equal(a_k, b_k[..., :live, :]))
            self.assertTrue(mx.array_equal(a_v, b_v[..., :live, :]))
        # every write stayed inside one slab of the final bucket
        self.assertEqual(ring.keys.shape[2], ring.capacity)

    def test_offset_is_an_array(self):
        ring = RingKVCache(buckets=(8,))
        self.assertIsInstance(ring.offset, mx.array)
        self.assertEqual(ring.offset.dtype, mx.int32)
        ring.update_and_fetch(*_kv(S=2))
        self.assertIsInstance(ring.offset, mx.array)
        self.assertEqual(ring.offset.dtype, mx.int32)

    def test_mask_is_causal_over_live_columns(self):
        ring = RingKVCache(buckets=(8,))
        ring.update_and_fetch(*_kv(S=3))
        # mask is built before the step's own write, so row i sees 0..offset+i
        self.assertEqual(
            ring.make_mask(1).astype(mx.int32).tolist(),
            [[[[1, 1, 1, 1, 0, 0, 0, 0]]]],
        )
        self.assertEqual(
            ring.make_mask(2).astype(mx.int32).tolist(),
            [[[[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]]]],
        )
        self.assertEqual(ring.make_mask(1).shape, (1, 1, 1, 8))

    def test_mask_at_the_last_position_before_capacity(self):
        """The final column of a full slab is attended, and nothing past it."""
        ring = RingKVCache(buckets=(8, 16))
        ring.update_and_fetch(*_kv(S=7))
        mask = ring.make_mask(1)
        self.assertEqual(ring.capacity, 8)
        self.assertEqual(mask.astype(mx.int32).tolist(), [[[[1] * 8]]])
        ring.update_and_fetch(*_kv(S=1))
        self.assertEqual(ring.size(), 8)
        self.assertEqual(ring.capacity, 8)

    def test_windowed_mask(self):
        ring = RingKVCache(buckets=(8,))
        ring.update_and_fetch(*_kv(S=4))
        self.assertEqual(
            ring.make_mask(1, window_size=3).astype(mx.int32).tolist(),
            [[[[0, 0, 1, 1, 1, 0, 0, 0]]]],
        )

    def test_growth_preserves_contents(self):
        mx.random.seed(4)
        kv, ring = KVCache(), RingKVCache(buckets=(4, 8, 16))
        for _ in range(11):
            k, v = _kv(S=1)
            kv.update_and_fetch(k, v)
            ring.update_and_fetch(k, v)
        self.assertEqual(ring.capacity, 16)
        self.assertEqual(ring.size(), 11)
        self.assertTrue(mx.array_equal(kv.keys[..., :11, :], ring.keys[..., :11, :]))
        self.assertTrue(
            mx.array_equal(kv.values[..., :11, :], ring.values[..., :11, :])
        )

    def test_reserve_grows_before_a_wide_step(self):
        ring = RingKVCache(buckets=(8, 16))
        ring.update_and_fetch(*_kv(S=7))
        self.assertEqual(ring.capacity, 8)
        # make_mask must widen the slab itself, or the mask would be narrower
        # than the keys the same forward is about to write.
        self.assertEqual(ring.make_mask(3).shape[-1], 16)
        self.assertEqual(ring.capacity, 16)

    def test_trim(self):
        mx.random.seed(5)
        kv, ring = KVCache(), RingKVCache(buckets=(32,))
        for _ in range(10):
            k, v = _kv(S=1)
            kv.update_and_fetch(k, v)
            ring.update_and_fetch(k, v)
        self.assertEqual(kv.trim(4), ring.trim(4))
        self.assertEqual(ring.size(), 6)
        self.assertEqual(int(ring.offset.item()), 6)
        self.assertEqual(ring.make_mask(1).astype(mx.int32).sum().item(), 7)
        # over-trim clamps, exactly like KVCache
        self.assertEqual(kv.trim(99), ring.trim(99))
        self.assertEqual(ring.size(), 0)
        self.assertEqual(int(ring.offset.item()), 0)

    def test_trim_prompt_cache_accepts_it(self):
        ring = RingKVCache(buckets=(32,))
        ring.update_and_fetch(*_kv(S=10))
        self.assertEqual(trim_prompt_cache([ring], 3), 3)
        self.assertEqual(ring.size(), 7)

    def test_round_trip_through_a_prompt_cache_file(self):
        mx.random.seed(6)
        ring = RingKVCache(buckets=(32,))
        ring.update_and_fetch(*_kv(S=9))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.safetensors")
            save_prompt_cache(path, [ring])
            back = load_prompt_cache(path)[0]
        self.assertIsInstance(back, RingKVCache)
        self.assertEqual(back.size(), 9)
        self.assertEqual(back.capacity, 32)
        self.assertTrue(mx.array_equal(ring.keys, back.keys))
        self.assertTrue(mx.array_equal(ring.values, back.values))

    def test_from_and_to_kv_cache(self):
        mx.random.seed(7)
        kv = KVCache()
        kv.update_and_fetch(*_kv(S=9))
        ring = RingKVCache.from_kv_cache(kv, buckets=(16, 32))
        self.assertEqual(ring.size(), 9)
        self.assertEqual(ring.capacity, 16)
        self.assertTrue(mx.array_equal(kv.keys[..., :9, :], ring.keys[..., :9, :]))
        back = ring.to_kv_cache()
        self.assertEqual(back.offset, 9)
        self.assertTrue(mx.array_equal(kv.keys[..., :9, :], back.keys))

    def test_padded_tail_is_never_attended(self):
        """Garbage past the offset must not reach the output."""
        mx.random.seed(8)
        ring = RingKVCache(buckets=(64,))
        k, v = _kv(S=8, D=64, dtype=mx.float32)
        ring.update_and_fetch(k, v)
        q = mx.random.normal((1, 2, 1, 64))
        clean_k, clean_v = ring.keys, ring.values
        out_clean = mx.fast.scaled_dot_product_attention(
            q, clean_k, clean_v, scale=0.125, mask=ring.make_mask(1)
        )
        # The mask is built before the step's own write, so column `offset`
        # is the slot that write will fill and is legitimately visible.
        # Everything past it is padding: poison it.
        live = ring.size() + 1
        ring.keys = mx.concatenate(
            [clean_k[..., :live, :], mx.full((1, 2, 64 - live, 64), 1e4)], axis=2
        )
        ring.values = mx.concatenate(
            [clean_v[..., :live, :], mx.full((1, 2, 64 - live, 64), 1e4)], axis=2
        )
        out_dirty = mx.fast.scaled_dot_product_attention(
            q, ring.keys, ring.values, scale=0.125, mask=ring.make_mask(1)
        )
        self.assertTrue(mx.array_equal(out_clean, out_dirty))


def _small_hybrid(n_layers=4, hidden=256, vocab=512):
    """A tiny qwen3_5 hybrid: GDN layers plus one full-attention layer."""
    from mlx_lm.models import qwen3_5

    args = qwen3_5.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        intermediate_size=hidden * 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        rms_norm_eps=1e-6,
        vocab_size=vocab,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=n_layers,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = qwen3_5.TextModel(args)
    # bf16 is what the checkpoints ship; compiled-vs-eager byte equality is
    # asserted at that dtype (fp32 fusion reorders a few elementwise chains).
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    model.eval()
    return model


class TestCompiledDecodeStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(11)
        cls.model = _small_hybrid()
        cls.prompt = mx.random.randint(0, 512, (1, 12)).astype(mx.uint32)

    def _run(self, mode, n_steps=8, buckets=(64, 128), width=1):
        model = self.model
        cache = model.make_cache()
        logits = model(self.prompt, cache)
        mx.eval(logits)
        step = None
        if mode != "kv":
            cd.to_shape_stable_cache(cache, buckets=buckets)
        if mode == "compiled":
            step = cd.CompiledDecodeStep(model, cache)
        y = mx.argmax(logits[:, -1:], axis=-1).astype(mx.uint32)
        if width > 1:
            y = mx.broadcast_to(y, (1, width)).astype(mx.uint32)
        toks, per_step = [], []
        for _ in range(n_steps):
            lg = step(y) if step is not None else model(y, cache)
            mx.eval(lg)
            per_step.append(lg[:, -1])
            nxt = mx.argmax(lg[:, -1:], axis=-1).astype(mx.uint32)
            toks.append(int(nxt.item()))
            y = mx.broadcast_to(nxt, (1, width)).astype(mx.uint32) if width > 1 else nxt
        return toks, per_step, cache, step

    def test_ring_cache_matches_kv_cache_through_the_model(self):
        ref_t, ref_l, ref_c, _ = self._run("kv")
        ring_t, ring_l, ring_c, _ = self._run("ring")
        self.assertEqual(ref_t, ring_t)
        for a, b in zip(ref_l, ring_l):
            self.assertTrue(mx.array_equal(a, b))
        for a, b in zip(ref_c, ring_c):
            if hasattr(a, "keys"):
                n = a.offset
                self.assertEqual(n, b.size())
                self.assertTrue(mx.array_equal(a.keys[..., :n, :], b.keys[..., :n, :]))
            else:
                for x, y in zip(a.cache, b.cache):
                    self.assertTrue(mx.array_equal(x, y))

    def test_compiled_matches_eager_at_width_1(self):
        ref_t, ref_l, ref_c, _ = self._run("ring")
        cmp_t, cmp_l, cmp_c, step = self._run("compiled")
        self.assertEqual(ref_t, cmp_t)
        for a, b in zip(ref_l, cmp_l):
            self.assertTrue(mx.array_equal(a, b))
        for a, b in zip(ref_c, cmp_c):
            if hasattr(a, "keys"):
                self.assertEqual(a.size(), b.size())
                self.assertTrue(mx.array_equal(a.keys, b.keys))
                self.assertTrue(mx.array_equal(a.values, b.values))
        step.assert_single_trace()

    def test_compiled_matches_eager_at_width_3(self):
        ref_t, ref_l, _, _ = self._run("ring", width=3)
        cmp_t, cmp_l, _, step = self._run("compiled", width=3)
        self.assertEqual(ref_t, cmp_t)
        for a, b in zip(ref_l, cmp_l):
            self.assertTrue(mx.array_equal(a, b))
        step.assert_single_trace()

    def test_one_trace_per_variant(self):
        """The mechanism proof: N steps, one trace, N-1 replays."""
        _, _, _, step = self._run("compiled", n_steps=10)
        self.assertEqual(list(step.trace_counts.values()), [1])
        self.assertEqual(list(step.replay_counts.values()), [10])
        self.assertEqual(step.n_variants, 1)

    def test_growth_retraces_once_per_capacity(self):
        n = 20
        ref_t, ref_l, _, _ = self._run("kv", n_steps=n)
        cmp_t, cmp_l, cache, step = self._run(
            "compiled", n_steps=n, buckets=(16, 20, 24, 32, 64)
        )
        self.assertEqual(ref_t, cmp_t)
        for a, b in zip(ref_l, cmp_l):
            self.assertTrue(mx.array_equal(a, b))
        self.assertGreater(step.n_variants, 1)
        step.assert_single_trace()
        self.assertEqual(sum(step.replay_counts.values()), n)

    def test_width_change_makes_a_new_variant(self):
        model = self.model
        cache = model.make_cache()
        mx.eval(model(self.prompt, cache))
        cd.to_shape_stable_cache(cache, buckets=(64, 128))
        step = cd.CompiledDecodeStep(model, cache)
        mx.eval(step(mx.zeros((1, 1), mx.uint32)))
        mx.eval(step(mx.zeros((1, 3), mx.uint32)))
        mx.eval(step(mx.zeros((1, 1), mx.uint32)))
        self.assertEqual(step.n_variants, 2)
        step.assert_single_trace()

    def test_refuses_a_speculating_cache(self):
        model = self.model
        cache = model.make_cache()
        mx.eval(model(self.prompt, cache))
        cd.to_shape_stable_cache(cache)
        step = cd.CompiledDecodeStep(model, cache)
        for c in cache:
            if hasattr(c, "start_speculation"):
                c.start_speculation()
        with self.assertRaises(RuntimeError):
            step(mx.zeros((1, 1), mx.uint32))

    def test_refuses_a_non_shape_stable_cache(self):
        from mlx_lm.models.cache import RotatingKVCache

        with self.assertRaises(TypeError):
            cd.CompiledDecodeStep(self.model, [RotatingKVCache(max_size=8)])
        with self.assertRaises(TypeError):
            cd.to_shape_stable_cache([RotatingKVCache(max_size=8)])

    def test_traced_body_has_no_host_sync(self):
        """Source gate: the traced step must not evaluate or read back.

        A ``.item()`` / ``mx.eval`` inside the traced body would either raise
        during tracing or, worse, bake a host value into the replayed graph.
        ``CompiledDecodeStep._build`` is the whole traced body, so scan it.
        """
        src = "\n".join(
            line
            for line in inspect.getsource(cd.CompiledDecodeStep._build).splitlines()
            if not line.lstrip().startswith("#")
        )
        for banned in (".item(", ".tolist(", "mx.eval", "async_eval", "synchronize"):
            self.assertNotIn(banned, src, f"{banned} in the traced step body")


class TestGenerateStepIntegration(unittest.TestCase):
    """The opt-in flag on generate_step: same tokens, compiled path taken."""

    @classmethod
    def setUpClass(cls):
        mx.random.seed(21)
        cls.model = _small_hybrid()
        cls.prompt = mx.random.randint(0, 512, (10,)).astype(mx.uint32)

    def _generate(self, **kw):
        from mlx_lm.generate import generate_step

        return [
            t
            for t, _ in generate_step(self.prompt, self.model, max_tokens=8, **kw)
        ]

    def test_flag_off_and_on_agree(self):
        eager = self._generate(compiled_decode=False)
        compiled = self._generate(compiled_decode=True)
        self.assertEqual(eager, compiled)

    def test_flag_actually_converts_the_cache(self):
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        pc = make_prompt_cache(self.model)
        list(
            generate_step(
                self.prompt, self.model, max_tokens=4, prompt_cache=pc,
                compiled_decode=True,
            )
        )
        self.assertTrue(any(isinstance(c, RingKVCache) for c in pc))

    def test_flag_declines_with_kv_bits(self):
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        pc = make_prompt_cache(self.model)
        list(
            generate_step(
                self.prompt, self.model, max_tokens=4, prompt_cache=pc,
                compiled_decode=True, kv_bits=8, quantized_kv_start=0,
            )
        )
        self.assertFalse(any(isinstance(c, RingKVCache) for c in pc))

    def test_env_flag_is_off_by_default(self):
        self.assertNotIn("MLX_LM_COMPILED_DECODE", os.environ)
        self.assertFalse(cd.compiled_decode_enabled())


if __name__ == "__main__":
    unittest.main()
