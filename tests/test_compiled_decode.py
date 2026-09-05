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
import importlib
import os
import tempfile
import threading
import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import compiled_decode as cd
from mlx_lm.models import precise_ops
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
    def test_capacity_is_the_minimum_first_bucket(self):
        ring = RingKVCache(capacity=10, buckets=(4, 8, 16, 32))
        self.assertEqual(ring.buckets, (10, 16, 32))
        self.assertEqual(ring._bucket_for(1), 10)

    def test_invalid_capacity_is_rejected(self):
        for capacity in (0, -1, True, 1.5):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                RingKVCache(capacity=capacity)

    def test_invalid_buckets_and_empty_reserve_are_rejected(self):
        with self.assertRaises(ValueError):
            RingKVCache(buckets=(True,))
        ring = RingKVCache(capacity=8)
        with self.assertRaisesRegex(ValueError, "geometry is known"):
            ring.reserve(1)

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


class TestPreciseSigmoid(unittest.TestCase):
    """MLX's Sigmoid struct spells an unqualified metal::exp, so a fused
    chain gets the fast approximation while eager gets the precise one
    (wiki research/mlx-compile-fused-sigmoid-rca-2026-09-03.md). These pin
    both halves: the defect is real, and our replacement is immune to it."""

    SWEEP = None

    @classmethod
    def setUpClass(cls):
        cls.SWEEP = mx.linspace(-12, 12, 4001)

    def test_precise_sigmoid_is_bit_identical_to_eager(self):
        for dtype in (mx.float32, mx.bfloat16):
            x = self.SWEEP.astype(dtype)
            self.assertTrue(
                mx.array_equal(mx.sigmoid(x), precise_ops.sigmoid(x)),
                f"precise sigmoid differs from eager at {dtype}",
            )

    def test_precise_sigmoid_survives_fusion(self):
        """The point of the custom primitive: mx.compile cannot fuse it."""
        for dtype in (mx.float32, mx.bfloat16):
            x = self.SWEEP.astype(dtype)
            fused = mx.compile(lambda z: precise_ops.sigmoid(z) * 1)(x)
            self.assertTrue(
                mx.array_equal(mx.sigmoid(x), fused),
                f"precise sigmoid moved inside a compiled span at {dtype}",
            )

    def test_the_defect_it_works_around_is_real(self):
        """Assert the mechanism: a plain fused sigmoid DOES move. If this
        ever stops failing, MLX was fixed and the workaround can go."""
        moved = []
        for dtype in (mx.float32, mx.bfloat16):
            x = self.SWEEP.astype(dtype)
            fused = mx.compile(lambda z: mx.sigmoid(z) * 1)(x)
            moved.append(not mx.array_equal(mx.sigmoid(x), fused).item())
        self.assertTrue(
            any(moved),
            "a fused mx.sigmoid no longer differs from eager -- MLX may have "
            "qualified metal::exp; re-check precise_ops before keeping it",
        )

    def test_float16_falls_back(self):
        """At fp16 the eager metallib sigmoid matches the FAST form, so the
        precise kernel would be the one that diverges."""
        x = self.SWEEP.astype(mx.float16)
        self.assertTrue(mx.array_equal(mx.sigmoid(x), precise_ops.sigmoid(x)))

    def test_gate_sigmoid_is_eager_outside_a_span(self):
        x = self.SWEEP.astype(mx.float32)
        self.assertFalse(precise_ops.in_precise_span())
        self.assertTrue(
            mx.array_equal(mx.sigmoid(x), precise_ops.gate_sigmoid(x))
        )
        with precise_ops.precise_span():
            self.assertTrue(precise_ops.in_precise_span())
            inside = precise_ops.gate_sigmoid(x)
        self.assertFalse(precise_ops.in_precise_span())
        self.assertTrue(mx.array_equal(mx.sigmoid(x), inside))

    def test_span_is_nested_and_context_local(self):
        seen = []
        with precise_ops.precise_span():
            self.assertTrue(precise_ops.in_precise_span())
            with precise_ops.precise_span():
                self.assertTrue(precise_ops.in_precise_span())
            self.assertTrue(precise_ops.in_precise_span())
            thread = threading.Thread(
                target=lambda: seen.append(precise_ops.in_precise_span())
            )
            thread.start()
            thread.join()
        self.assertFalse(precise_ops.in_precise_span())
        self.assertEqual(seen, [False])


def _small_hybrid(n_layers=4, hidden=256, vocab=512):
    """A tiny qwen3_5 hybrid: GDN layers plus one full-attention layer."""
    from mlx_lm.models import qwen3_5

    args = qwen3_5.TextModelArgs(
        model_type="qwen3_5_moe",
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
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=hidden,
        shared_expert_intermediate_size=hidden,
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
        cls.bucket_env = mock.patch.dict(
            os.environ, {"MLX_LM_RING_KV_BUCKETS": "64,128"}
        )
        cls.bucket_env.start()
        mx.random.seed(11)
        cls.model = _small_hybrid()
        cls.prompt = mx.random.randint(0, 512, (1, 12)).astype(mx.uint32)

    @classmethod
    def tearDownClass(cls):
        cls.bucket_env.stop()

    def _run(self, mode, n_steps=8, buckets=(64, 128), width=1):
        model = self.model
        cache = model.make_cache()
        logits = model(self.prompt, cache)
        mx.eval(logits)
        step = None
        if mode != "kv":
            cd.to_shape_stable_cache(cache, buckets=buckets)
        if mode == "compiled":
            bucket_text = ",".join(str(b) for b in buckets)
            with mock.patch.dict(
                os.environ, {"MLX_LM_RING_KV_BUCKETS": bucket_text}
            ):
                step = cd.CompiledDecodeStep(model, cache)
        y = mx.argmax(logits[:, -1:], axis=-1).astype(mx.uint32)
        if width > 1:
            y = mx.broadcast_to(y, (1, width)).astype(mx.uint32)
        toks, per_step = [], []
        for _ in range(n_steps):
            lg = step(y) if step is not None else model(y, cache)
            if step is not None:
                step.materialize_and_confirm(lg)
            else:
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

    def test_refuses_width_3(self):
        model = self.model
        cache = model.make_cache()
        mx.eval(model(self.prompt, cache))
        cd.to_shape_stable_cache(cache, buckets=(64, 128))
        step = cd.CompiledDecodeStep(model, cache)
        with self.assertRaisesRegex(ValueError, "batch 1, width 1"):
            step(mx.zeros((1, 3), mx.uint32))

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

    def test_refuses_batch_2(self):
        model = self.model
        cache = model.make_cache()
        mx.eval(model(self.prompt, cache))
        cd.to_shape_stable_cache(cache, buckets=(64, 128))
        step = cd.CompiledDecodeStep(model, cache)
        with self.assertRaisesRegex(ValueError, "batch 1, width 1"):
            step(mx.zeros((2, 1), mx.uint32))

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

    def test_mixed_cache_decline_is_atomic(self):
        from mlx_lm.models.cache import RotatingKVCache

        kv = KVCache()
        rotating = RotatingKVCache(max_size=8)
        caches = [kv, rotating]
        with self.assertRaises(TypeError):
            cd.to_shape_stable_cache(caches)
        self.assertIs(caches[0], kv)
        self.assertIs(caches[1], rotating)

    def test_unqualified_model_is_rejected_before_cache_conversion(self):
        class Unsupported:
            model_type = "gpt_oss"

        class UnprovenQwen:
            model_type = "qwen3_5"

        for model in (Unsupported(), UnprovenQwen()):
            why = cd.model_is_compilable(model, [KVCache()])
            self.assertIn("has not been qualified", why)

    def test_max_variants_is_bounded_and_eviction_preserves_receipts(self):
        for value in (0, 65, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cd.CompiledDecodeStep(self.model, [], max_variants=value)

        step = cd.CompiledDecodeStep.__new__(cd.CompiledDecodeStep)
        step._variants = {"old": object()}
        step.trace_counts = {"old": 1}
        step.submission_counts = {"old": 9}
        step.replay_counts = {"old": 9}
        step.failure_counts = {}
        step._evict_all()
        self.assertEqual(step._variants, {})
        self.assertEqual(step.trace_counts, {"old": 1})
        self.assertEqual(step.submission_counts, {"old": 9})
        self.assertEqual(step.replay_counts, {"old": 9})

    def test_context_policy_keeps_the_16k_tradeoff_explicit(self):
        default_buckets = (2048, 4096, 8192, 16384, 32768, 65536)
        with mock.patch.object(cd, "_ring_buckets", return_value=default_buckets):
            why, policy = cd.compiled_decode_context_policy(4000, 96, "short")
            self.assertIsNone(why)
            self.assertEqual(policy.max_context, 4096)
            self.assertIn(4096, policy.buckets)

            why, _ = cd.compiled_decode_context_policy(4000, 97, "short")
            self.assertIn("exceeds", why)
            why, _ = cd.compiled_decode_context_policy(1, -1, "short")
            self.assertIn("unbounded", why)

            why, memory = cd.compiled_decode_context_policy(
                8000, 8000, "memory"
            )
            self.assertIsNone(why)
            self.assertEqual(memory.max_context, 16384)
            self.assertIn(16384, memory.buckets)
            why, latency = cd.compiled_decode_context_policy(
                8000, 8000, "latency"
            )
            self.assertIsNone(why)
            self.assertNotIn(16384, latency.buckets)
            self.assertIn(32768, latency.buckets)

    def test_invalid_context_policy_declines(self):
        with mock.patch.dict(
            os.environ,
            {"MLX_LM_COMPILED_DECODE_CONTEXT_POLICY": "surprise"},
        ):
            why, buckets = cd.compiled_decode_context_policy(1, 1)
        self.assertIn("must be one of", why)
        self.assertIsNone(buckets)

    def test_padded_sdpa_acceptance_is_explicit_and_exact(self):
        with mock.patch.dict(
            os.environ, {"MLX_LM_COMPILED_DECODE_ACCEPTANCE": ""}
        ):
            why, policy = cd.compiled_decode_context_policy(1, 1)
            self.assertIsNone(why)
            self.assertFalse(cd.compiled_decode_numerics_accepted(policy))
        with mock.patch.dict(
            os.environ,
            {"MLX_LM_COMPILED_DECODE_ACCEPTANCE": "class3-padded-sdpa-v1"},
        ):
            why, policy = cd.compiled_decode_context_policy(1, 1)
            self.assertIsNone(why)
            self.assertTrue(cd.compiled_decode_numerics_accepted(policy))
        with mock.patch.dict(
            os.environ, {"MLX_LM_COMPILED_DECODE_ACCEPTANCE": "yes"}
        ):
            why, policy = cd.compiled_decode_context_policy(1, 1)
            self.assertIn("must be", why)
            self.assertIsNone(policy)

    def test_forged_context_policy_is_rejected(self):
        cache = self.model.make_cache()
        mx.eval(self.model(self.prompt, cache))
        cd.to_shape_stable_cache(cache)
        forged = cd.CompiledDecodePolicy("short", 1_000_000, cache[-1].buckets)
        with self.assertRaisesRegex(ValueError, "limit must be"):
            cd.CompiledDecodeStep(self.model, cache, context_policy=forged)

    def test_context_policy_must_match_ring_buckets(self):
        cache = self.model.make_cache()
        mx.eval(self.model(self.prompt, cache))
        cd.to_shape_stable_cache(cache, buckets=(64, 128))
        policy = cd.CompiledDecodePolicy("short", 4096, (32, 64))
        with self.assertRaisesRegex(ValueError, "buckets do not match"):
            cd.CompiledDecodeStep(self.model, cache, context_policy=policy)

    def test_empty_ring_is_rejected_at_setup(self):
        why = cd.model_is_compilable(self.model, [RingKVCache()])
        self.assertIn("has not been filled", why)

    def test_cache_batch_geometry_is_checked(self):
        ring = RingKVCache(buckets=(16,))
        ring.update_and_fetch(*_kv(B=2, S=3))
        self.assertIn("not batch 1", cd.model_is_compilable(self.model, [ring]))

        from mlx_lm.models.cache import ArraysCache

        arrays = ArraysCache(1)
        arrays.cache[0] = mx.zeros((2, 4), mx.float32)
        one = RingKVCache(buckets=(16,))
        one.update_and_fetch(*_kv(B=1, S=3))
        self.assertIn(
            "not batch 1", cd.model_is_compilable(self.model, [one, arrays])
        )

    def test_full_attention_positions_must_be_synchronized(self):
        first = RingKVCache(buckets=(16,))
        second = RingKVCache(buckets=(16,))
        first.update_and_fetch(*_kv(S=2))
        second.update_and_fetch(*_kv(S=3))
        self.assertIn(
            "not synchronized",
            cd.model_is_compilable(self.model, [first, second]),
        )

    def test_family_and_distributed_modes_are_not_inherited_as_qualified(self):
        from mlx_lm.models import qwen3_next

        self.assertFalse(qwen3_next.Model.supports_compiled_decode_replay)
        original_experts = self.model.args.num_experts
        self.model.args.num_experts = 0
        try:
            why = cd.model_is_compilable(self.model, self.model.make_cache())
        finally:
            self.model.args.num_experts = original_experts
        self.assertIn("only the qwen3_5 MoE", why)

        original = self.model.model.pipeline_size
        self.model.model.pipeline_size = 2
        try:
            why = cd.model_is_compilable(self.model, self.model.make_cache())
        finally:
            self.model.model.pipeline_size = original
        self.assertIn("pipeline-parallel", why)

    def test_completion_receipt_requires_materialized_ack(self):
        model = self.model
        cache = model.make_cache()
        mx.eval(model(self.prompt, cache))
        cd.to_shape_stable_cache(cache, buckets=(64, 128))
        step = cd.CompiledDecodeStep(model, cache)
        logits = step(mx.zeros((1, 1), mx.uint32))
        with self.assertRaisesRegex(AssertionError, "unconfirmed"):
            step.assert_single_trace()
        with self.assertRaisesRegex(RuntimeError, "oldest compiled submission"):
            step.materialize_and_confirm(mx.zeros_like(logits))
        self.assertEqual(step.materialize_and_confirm(logits), 1)
        self.assertTrue(step.assert_single_trace())
        receipt = step.receipt()
        self.assertEqual(sum(receipt["submission_counts"].values()), 1)
        self.assertEqual(sum(receipt["completed_counts"].values()), 1)
        self.assertEqual(receipt["pending"], 0)

    def test_trace_failure_restores_state_and_poisons_the_step(self):
        model = self.model
        cache = model.make_cache()
        mx.eval(model(self.prompt, cache))
        cd.to_shape_stable_cache(cache, buckets=(64, 128))
        step = cd.CompiledDecodeStep(model, cache)
        ring = next(c for c in cache if isinstance(c, RingKVCache))
        before = (ring.keys, ring.values, ring.offset, ring._host_offset)

        class FailingModel:
            def __call__(self, x, cache):
                del x
                target = next(c for c in cache if isinstance(c, RingKVCache))
                target._host_offset += 1
                raise RuntimeError("synthetic trace failure")

        step.model = FailingModel()
        with mock.patch.object(mx, "compile", side_effect=lambda fn: fn):
            with self.assertRaises(cd.CompiledDecodePoisoned):
                step(mx.zeros((1, 1), mx.uint32))
        self.assertIs(ring.keys, before[0])
        self.assertIs(ring.values, before[1])
        self.assertIs(ring.offset, before[2])
        self.assertEqual(ring._host_offset, before[3])
        with self.assertRaises(cd.CompiledDecodePoisoned):
            step(mx.zeros((1, 1), mx.uint32))

    def test_materialization_failure_poisons_pending_receipts(self):
        model = self.model
        cache = model.make_cache()
        mx.eval(model(self.prompt, cache))
        cd.to_shape_stable_cache(cache, buckets=(64, 128))
        step = cd.CompiledDecodeStep(model, cache)
        logits = step(mx.zeros((1, 1), mx.uint32))
        with mock.patch.object(
            mx, "eval", side_effect=RuntimeError("synthetic async failure")
        ):
            with self.assertRaises(cd.CompiledDecodePoisoned):
                step.materialize_and_confirm(logits)
        receipt = step.receipt()
        self.assertTrue(receipt["poisoned"])
        self.assertEqual(receipt["pending"], 0)
        self.assertEqual(sum(receipt["failure_counts"].values()), 1)
        with self.assertRaisesRegex(AssertionError, "poisoned"):
            step.assert_single_trace()

    def test_the_trace_runs_inside_a_precise_span(self):
        """Mechanism proof for the sigmoid cut: the flag must be set while
        the step is traced, or every eager sigmoid the step swallows would
        silently take the fast-exp path."""
        model = self.model
        cache = model.make_cache()
        mx.eval(model(self.prompt, cache))
        cd.to_shape_stable_cache(cache, buckets=(64, 128))
        step = cd.CompiledDecodeStep(model, cache)
        seen = []
        orig = precise_ops.gate_sigmoid

        def spy(x):
            seen.append(precise_ops.in_precise_span())
            return orig(x)

        import mlx_lm.models.qwen3_next as qn

        qn.gate_sigmoid = spy
        try:
            logits = step(mx.zeros((1, 1), mx.uint32))
            step.materialize_and_confirm(logits)
        finally:
            qn.gate_sigmoid = orig
        self.assertTrue(seen, "no gated sigmoid ran during the trace")
        self.assertTrue(all(seen), "a sigmoid was traced outside the span")

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
        cls.bucket_env = mock.patch.dict(
            os.environ, {"MLX_LM_RING_KV_BUCKETS": "64,128"}
        )
        cls.bucket_env.start()
        mx.random.seed(21)
        cls.model = _small_hybrid()
        cls.prompt = mx.random.randint(0, 512, (10,)).astype(mx.uint32)

    def setUp(self):
        # These tiny-model integration tests exercise research mechanics,
        # not a checkpoint serving approval (covered by CPU admission tests).
        self.qualification = mock.patch(
            "mlx_lm.generate.compiled_decode_serving_reason", return_value=None
        )
        self.qualification.start()
        self.addCleanup(self.qualification.stop)

    @classmethod
    def tearDownClass(cls):
        cls.bucket_env.stop()

    def _generate(self, **kw):
        from mlx_lm.generate import generate_step

        with mock.patch.dict(
            os.environ,
            {"MLX_LM_COMPILED_DECODE_ACCEPTANCE": "class3-padded-sdpa-v1"},
        ):
            return [
                t
                for t, _ in generate_step(
                    self.prompt, self.model, max_tokens=8, **kw
                )
            ]

    def test_flag_off_and_on_agree(self):
        eager = self._generate(compiled_decode=False)
        compiled = self._generate(compiled_decode=True)
        self.assertEqual(eager, compiled)

    def test_private_cache_path_constructs_the_compiled_step(self):
        generate_module = importlib.import_module("mlx_lm.generate")

        real = generate_module.CompiledDecodeStep
        with mock.patch.object(
            generate_module, "CompiledDecodeStep", wraps=real
        ) as construct:
            self._generate(compiled_decode=True)
        construct.assert_called_once()

    def test_caller_owned_cache_is_not_converted(self):
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        pc = make_prompt_cache(self.model)
        status = {}
        with mock.patch.dict(
            os.environ,
            {"MLX_LM_COMPILED_DECODE_ACCEPTANCE": "class3-padded-sdpa-v1"},
        ):
            list(
                generate_step(
                    self.prompt,
                    self.model,
                    max_tokens=4,
                    prompt_cache=pc,
                    compiled_decode=True,
                    _compiled_decode_status=status,
                )
            )
        self.assertFalse(any(isinstance(c, RingKVCache) for c in pc))
        self.assertTrue(any(type(c) is KVCache for c in pc))
        self.assertFalse(status["used"])
        self.assertIn("caller-owned", status["decline_reason"])

    def test_server_request_private_cache_is_eligible_but_not_implicit(self):
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        pc = make_prompt_cache(self.model)
        status = {}
        with mock.patch.dict(
            os.environ,
            {"MLX_LM_COMPILED_DECODE_ACCEPTANCE": "class3-padded-sdpa-v1"},
        ):
            list(
                generate_step(
                    self.prompt,
                    self.model,
                    max_tokens=4,
                    prompt_cache=pc,
                    compiled_decode=True,
                    _prompt_cache_is_request_private=True,
                    _compiled_decode_status=status,
                )
            )
        self.assertTrue(status["used"])
        self.assertTrue(any(type(c) is RingKVCache for c in pc))

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

    def test_setup_failure_leaves_prompt_cache_eager(self):
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        pc = make_prompt_cache(self.model)
        with (
            mock.patch("mlx_lm.generate.make_prompt_cache", return_value=pc),
            mock.patch(
                "mlx_lm.generate.CompiledDecodeStep",
                side_effect=ValueError("synthetic setup failure"),
            ),
            mock.patch.dict(
                os.environ,
                {"MLX_LM_COMPILED_DECODE_ACCEPTANCE": "class3-padded-sdpa-v1"},
            ),
        ):
            list(
                generate_step(
                    self.prompt, self.model, max_tokens=4, compiled_decode=True
                )
            )
        self.assertTrue(any(type(c) is KVCache for c in pc))
        self.assertFalse(any(isinstance(c, RingKVCache) for c in pc))

    def test_zero_token_request_does_not_construct_compiled_step(self):
        from mlx_lm.generate import generate_step

        with mock.patch(
            "mlx_lm.generate.CompiledDecodeStep",
            side_effect=AssertionError("compiled setup must be skipped"),
        ):
            self.assertEqual(
                list(
                    generate_step(
                        self.prompt,
                        self.model,
                        max_tokens=0,
                        compiled_decode=True,
                    )
                ),
                [],
            )

    def test_env_flag_is_on_by_default_and_zero_opts_out(self):
        self.assertNotIn("MLX_LM_COMPILED_DECODE", os.environ)
        self.assertTrue(cd.compiled_decode_enabled())
        for off in ("0", "false", "no", "off", ""):
            with mock.patch.dict(os.environ, {"MLX_LM_COMPILED_DECODE": off}):
                self.assertFalse(cd.compiled_decode_enabled(), off)
        with mock.patch.dict(os.environ, {"MLX_LM_COMPILED_DECODE": "1"}):
            self.assertTrue(cd.compiled_decode_enabled())

    def test_default_ladder_is_class1_and_needs_no_acceptance(self):
        with mock.patch.dict(os.environ):
            # The class fixture pins a tiny test ladder; the default ladder is
            # what production resolves when nothing is set.
            os.environ.pop("MLX_LM_RING_KV_BUCKETS", None)
            os.environ.pop("MLX_LM_COMPILED_DECODE_ACCEPTANCE", None)
            why, policy = cd.compiled_decode_context_policy(11, 128, "short")
            self.assertIsNone(why)
            self.assertIn(1023, policy.buckets)
            self.assertIn(1024, policy.buckets)
            self.assertEqual(policy.numerical_acceptance, "class1-bucketed-v1")
            self.assertTrue(cd.compiled_decode_numerics_accepted(policy))
        with mock.patch.dict(os.environ, {"MLX_LM_RING_KV_BUCKETS": "2048,4096"}):
            os.environ.pop("MLX_LM_COMPILED_DECODE_ACCEPTANCE", None)
            why, policy = cd.compiled_decode_context_policy(11, 128, "short")
            self.assertIsNone(why)
            self.assertIsNone(policy.numerical_acceptance)
            self.assertFalse(cd.compiled_decode_numerics_accepted(policy))


if __name__ == "__main__":
    unittest.main()
