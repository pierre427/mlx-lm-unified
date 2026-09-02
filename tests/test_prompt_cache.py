# Copyright © 2024 Apple Inc.

import copy
import json
import os
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.base import create_attention_mask, create_causal_mask
from mlx_lm.models.cache import (
    ArraysCache,
    BatchQuantizedKVCache,
    BatchKVCache,
    BatchRotatingKVCache,
    BatchRotatingQuantizedKVCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    RotatingQuantizedKVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.utils import load

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestPromptCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        cls.model, cls.tokenizer = load(HF_MODEL_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_save_load(self):
        cache = [KVCache() for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Test with metadata
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        metadata = {"a": "b", "c": "d"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_metadata = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_metadata)

    def test_save_load_rotating_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        # Test with rotating cache
        cache = [RotatingKVCache(max_size=8, keep=2) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertEqual(c.keep, lc.keep)
            self.assertEqual(c.max_size, lc.max_size)
            self.assertEqual(c.step, lc.step)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Do a couple single token updates to get a rotation
        for _ in range(2):
            for c in cache:
                x = mx.random.uniform(shape=(1, 8, 1, 4))
                c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        for c, lc in zip(cache, loaded_cache):
            x = mx.random.uniform(shape=(1, 8, 1, 4))
            k, v = c.update_and_fetch(x, x)
            lk, lv = lc.update_and_fetch(x, x)
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(k, lk))
            self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_mixed_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2),
            KVCache(),
            RotatingKVCache(8),
            ArraysCache(size=2),
            ChunkedKVCache(256),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        for c, lc in zip(cache, loaded_cache):
            if isinstance(c, ArraysCache):
                self.assertTrue(mx.array_equal(c[0], lc[0]))
                self.assertTrue(mx.array_equal(c[1], lc[1]))
            else:
                x = mx.random.uniform(shape=(4, 4, 1, 4))
                y = mx.random.uniform(shape=(4, 4, 1, 4))
                k, v = c.update_and_fetch(x, y)
                lk, lv = lc.update_and_fetch(x, y)
                self.assertEqual(c.offset, lc.offset)
                self.assertTrue(mx.array_equal(k, lk))
                self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_cache_list(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2),
            KVCache(),
            RotatingKVCache(8),
            ArraysCache(size=2),
            ChunkedKVCache(256),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)
        cache = [CacheList(*cache)]

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        for c, lc in zip(cache[0].caches, loaded_cache[0].caches):
            if isinstance(c, ArraysCache):
                self.assertTrue(mx.array_equal(c[0], lc[0]))
                self.assertTrue(mx.array_equal(c[1], lc[1]))
            else:
                x = mx.random.uniform(shape=(4, 4, 1, 4))
                y = mx.random.uniform(shape=(4, 4, 1, 4))
                k, v = c.update_and_fetch(x, y)
                lk, lv = lc.update_and_fetch(x, y)
                self.assertEqual(c.offset, lc.offset)
                self.assertTrue(mx.array_equal(k, lk))
                self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_arrays_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [ArraysCache(size=2)]
        cache[0][0] = mx.zeros((1, 4, 4))
        cache[0][1] = mx.zeros((1, 4, 4))

        save_prompt_cache(cache_file, cache)
        loaded = load_prompt_cache(cache_file)

        # Try to make a mask
        mask = loaded[0].make_mask(4)

    def test_cache_with_generate(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]
        results = list(generate_step(prompt, model, max_tokens=4))
        toks, all_logits = zip(*results)

        prompt_cache = make_prompt_cache(model)
        i = 0
        for tok, logits in generate_step(
            prompt, model, prompt_cache=prompt_cache, max_tokens=2
        ):
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))
            i += 1

        for tok, logits in generate_step(
            mx.array([toks[i]]), model, prompt_cache=prompt_cache, max_tokens=1
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))

    def test_trim_cache(self):
        cache = [KVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        # Trim
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 7)

        # Trim more tokens than remain
        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 3)

        # Can't trim arrays cache
        cache = [ArraysCache(size=2) for _ in range(2)]
        for c in cache:
            c[0] = mx.zeros((5, 5))
            c[1] = mx.zeros((5, 5))
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 0)

        # All cache's have to be trimmable
        cache = [ArraysCache(size=2), KVCache()]
        cache[0][0] = mx.zeros((5, 5))
        cache[0][1] = mx.zeros((5, 5))
        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[1].update_and_fetch(x, x)
        num_trimmed = trim_prompt_cache(cache, 1)
        self.assertEqual(num_trimmed, 0)

        cache = [RotatingKVCache(max_size=6) for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 5, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 4)

        # Can't trim fixed-size KV cache after processing
        # more than max_kv_size tokens
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 0)

        cache = [QuantizedKVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 64))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 7)

        # Trim more tokens than remain
        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 3)

    def test_trim_cache_with_generate(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]

        prompt_cache = make_prompt_cache(model)

        # Generate one token so we process the full prompt
        last_tok, _ = next(generate_step(prompt, model, prompt_cache=prompt_cache))
        last_tok = mx.array([last_tok])

        # Generate two more tokens
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        toks, all_logits = zip(*(r[1] for r in results))

        # To get back to the cache just after processing the prompt,
        # trim by 3 tokens
        trim_prompt_cache(prompt_cache, 3)

        # Generate the same thing again
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        second_toks, second_all_logits = zip(*(r[1] for r in results))
        self.assertEqual(toks, second_toks)
        self.assertTrue(
            all(mx.allclose(l, l2) for l, l2 in zip(all_logits, second_all_logits))
        )

    def test_cache_copying(self):
        cache = [KVCache()]

        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[0].update_and_fetch(x, x)

        y = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(y, y)

        old_cache = copy.deepcopy(cache)

        trim_prompt_cache(cache, 1)

        self.assertTrue(old_cache[0].offset, 11)
        self.assertTrue(cache[0].offset, 10)

        z = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(z, z)

        self.assertTrue(mx.allclose(old_cache[0].keys[..., 10:11, :], y))
        self.assertTrue(mx.allclose(cache[0].keys[..., 10:11, :], z))

    def test_save_load_quantized_cache(self):
        cache = [QuantizedKVCache(bits=4, group_size=32) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 32))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(loaded_cache[0].bits == cache[0].bits)
        self.assertTrue(loaded_cache[0].group_size == cache[0].group_size)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            # Loop over quantized tuple
            for i in range(3):
                self.assertTrue(mx.array_equal(c.state[0][i], lc.state[0][i]))
                self.assertTrue(mx.array_equal(c.state[1][i], lc.state[1][i]))

        # Test with metadata
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        metadata = {"a": "b", "c": "d"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_metadata = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_metadata)

    def test_cache_to_quantized(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]
        results = zip(range(4), generate_step(prompt, model))
        toks, all_logits = zip(*(r[1] for r in results))

        prompt_cache = make_prompt_cache(model)
        i = 0
        for _, (tok, logits) in zip(
            range(2), generate_step(prompt, model, prompt_cache=prompt_cache)
        ):
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))
            i += 1

        prompt_cache = [c.to_quantized(bits=8, group_size=32) for c in prompt_cache]

        for _, (tok, logits) in zip(
            range(1),
            generate_step(mx.array([toks[i]]), model, prompt_cache=prompt_cache),
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i], rtol=4e-2))

    def test_rotating_cache_to_quantized(self):
        """RotatingKVCache.to_quantized() — previously NotImplementedError.
        Build up a rotating cache past its max_size (forcing rotation) with a
        real model, convert to quantized mid-stream, and confirm generation
        continues producing sane (numerically close) logits."""
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode(
            "Once upon a time in a small town", return_tensors="mlx"
        )[0]

        prompt_cache = [RotatingKVCache(max_size=4) for _ in model.layers]
        toks = []
        for _, (tok, logits) in zip(
            range(8), generate_step(prompt, model, prompt_cache=prompt_cache)
        ):
            toks.append(tok)
        self.assertIsInstance(prompt_cache[0], RotatingKVCache)

        quant_cache = [c.to_quantized(bits=8, group_size=32) for c in prompt_cache]
        for c in quant_cache:
            self.assertEqual(c.offset, prompt_cache[0].offset)

        # Generation must continue without error after the mid-stream swap.
        next_toks = []
        for _, (tok, logits) in zip(
            range(3),
            generate_step(mx.array([toks[-1]]), model, prompt_cache=quant_cache),
        ):
            self.assertFalse(bool(mx.any(mx.isnan(logits))))
            next_toks.append(tok)
        self.assertEqual(len(next_toks), 3)

    def test_rotating_cache_to_quantized_matches_unquantized(self):
        """The rotating-quantized path must stay numerically faithful, not just
        non-NaN: after rotation, a quantized rotating cache should track the
        unquantized rotating reference within the same tolerance QuantizedKVCache
        gets in test_cache_to_quantized (rtol=4e-2). Guards against a silent
        numerical regression in the (packed, scales, biases) bookkeeping."""
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode(
            "Once upon a time in a small town", return_tensors="mlx"
        )[0]

        # Two identical warmups (argmax is deterministic, so the KV state is the
        # same in both) run long enough to force rotation past max_size.
        def warm():
            c = [RotatingKVCache(max_size=8) for _ in model.layers]
            last = None
            for _, (tok, _) in zip(
                range(12), generate_step(prompt, model, prompt_cache=c)
            ):
                last = tok
            return c, last

        ref_cache, last_ref = warm()
        base_cache, last_base = warm()
        self.assertEqual(last_ref, last_base)  # warmups agree
        self.assertGreater(ref_cache[0].offset, ref_cache[0].max_size)  # rotated

        # Keep one continuation unquantized (reference), quantize the other, then
        # feed both the same next token and compare the resulting logits.
        quant_cache = [c.to_quantized(bits=8, group_size=32) for c in base_cache]
        for c in quant_cache:
            self.assertIsInstance(c, RotatingQuantizedKVCache)

        x = mx.array([last_ref])
        ref_logits = next(generate_step(x, model, prompt_cache=ref_cache))[1]
        quant_logits = next(generate_step(x, model, prompt_cache=quant_cache))[1]
        self.assertFalse(bool(mx.any(mx.isnan(quant_logits))))
        self.assertTrue(mx.allclose(ref_logits, quant_logits, rtol=4e-2))

    def test_rotating_quantized_cache_save_load(self):
        cache = [
            RotatingKVCache(max_size=4).to_quantized(bits=8, group_size=32)
            for _ in range(2)
        ]
        for c in cache:
            for _ in range(10):  # forces rotation
                x = mx.random.uniform(shape=(1, 4, 1, 32))
                c.update_and_fetch(x, x)

        cache_file = os.path.join(self.test_dir, "rotating_quant_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertEqual(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertIsInstance(lc, RotatingQuantizedKVCache)
            self.assertEqual(c.offset, lc.offset)
            self.assertEqual(c.max_size, lc.max_size)
            self.assertEqual(c.group_size, lc.group_size)
            self.assertEqual(c.bits, lc.bits)
            for i in range(3):
                self.assertTrue(mx.array_equal(c.state[0][i], lc.state[0][i]))
                self.assertTrue(mx.array_equal(c.state[1][i], lc.state[1][i]))

    def test_batch_rotating_quantized_kv_cache_merge_filter_extend_extract(self):
        """The batching-facing surface BatchGenerator actually relies on."""
        jobs = []
        for j in range(3):
            c = RotatingQuantizedKVCache(max_size=8, group_size=32, bits=8)
            for _ in range(3 + j * 4):  # some jobs exceed max_size, forcing rotation
                x = mx.random.uniform(shape=(1, 4, 1, 32))
                c.update_and_fetch(x, x)
            jobs.append(c)

        batch = RotatingQuantizedKVCache.merge(jobs)
        self.assertIsInstance(batch, BatchRotatingQuantizedKVCache)
        self.assertEqual(batch.keys[0].shape[0], 3)
        for j, c in enumerate(jobs):
            self.assertEqual(int(batch.offset[j]), c.offset)

        k = mx.random.uniform(shape=(3, 4, 1, 32))
        bk, bv = batch.update_and_fetch(k, k)
        self.assertEqual(bk[0].shape[0], 3)

        batch.filter(mx.array([0, 2]))
        self.assertEqual(batch.keys[0].shape[0], 2)

        extracted = batch.extract(0)
        self.assertIsInstance(extracted, RotatingQuantizedKVCache)
        self.assertEqual(extracted.offset, int(batch.offset[0]))

        new_job = RotatingQuantizedKVCache(max_size=8, group_size=32, bits=8)
        for _ in range(2):
            x = mx.random.uniform(shape=(1, 4, 1, 32))
            new_job.update_and_fetch(x, x)
        batch.extend(RotatingQuantizedKVCache.merge([new_job]))
        self.assertEqual(batch.keys[0].shape[0], 3)

    def test_batch_quantized_kv_cache_asymmetric_protocol(self):
        jobs = []
        for length in (3, 7, 5):
            cache = QuantizedKVCache(
                group_size=32, key_bits=8, value_bits=4, rotate=True
            )
            x = mx.random.normal((1, 2, length, 32))
            cache.update_and_fetch(x, x)
            jobs.append(cache)

        batch = QuantizedKVCache.merge(jobs)
        self.assertIsInstance(batch, BatchQuantizedKVCache)
        self.assertEqual(batch.key_bits, 8)
        self.assertEqual(batch.value_bits, 4)
        self.assertEqual(batch.offset.tolist(), [3, 7, 5])

        x = mx.random.normal((3, 2, 1, 32))
        keys, values = batch.update_and_fetch(x, x)
        self.assertEqual(keys[0].shape[:3], (3, 2, 8))
        self.assertEqual(values[0].shape[:3], (3, 2, 8))
        self.assertEqual(batch.trim(1), 1)
        self.assertEqual(batch.offset.tolist(), [3, 7, 5])

        batch.filter(mx.array([0, 2]))
        self.assertEqual(batch.offset.tolist(), [3, 5])
        extracted = batch.extract(1)
        self.assertIsInstance(extracted, QuantizedKVCache)
        self.assertEqual(extracted.offset, 5)
        self.assertEqual(extracted.key_bits, 8)
        self.assertEqual(extracted.value_bits, 4)

        other = QuantizedKVCache.merge([jobs[0]])
        batch.extend(other)
        self.assertEqual(batch.offset.tolist(), [3, 5, 3])

    def test_batch_quantized_kv_cache_normalized_fails_closed(self):
        cache = QuantizedKVCache(group_size=32, bits=8, normalize=True)
        x = mx.random.normal((1, 2, 3, 32))
        cache.update_and_fetch(x, x)
        with self.assertRaisesRegex(ValueError, "normalized"):
            QuantizedKVCache.merge([cache])

    def test_cache_list(self):
        c = CacheList(KVCache(), KVCache())
        self.assertTrue(c.is_trimmable())
        k = mx.zeros((1, 2, 8, 8))
        v = mx.zeros((1, 2, 8, 8))
        c[0].update_and_fetch(k, v)
        c[1].update_and_fetch(k, v)
        m = c.trim(5)
        self.assertEqual(m, 5)

        c = CacheList(ArraysCache(size=2), KVCache())
        self.assertFalse(c.is_trimmable())

        c1 = CacheList(ArraysCache(size=1), KVCache())
        c1[0][0] = mx.random.normal(shape=(1, 2, 4, 4))
        c1[1].update_and_fetch(
            mx.random.normal(shape=(1, 2, 5, 4)), mx.random.normal(shape=(1, 2, 5, 4))
        )

        c2 = CacheList(ArraysCache(size=1), KVCache())
        c2[0][0] = mx.random.normal(shape=(1, 2, 4, 4))
        c2[1].update_and_fetch(
            mx.random.normal(shape=(1, 2, 7, 4)), mx.random.normal(shape=(1, 2, 7, 4))
        )

        merged_cache = CacheList.merge((c1, c2))
        c1_ex = merged_cache.extract(0)
        self.assertTrue(mx.array_equal(c1_ex[0][0], c1[0][0]))
        self.assertTrue(
            mx.array_equal(
                c1_ex[1].keys_and_values()[0], c1[1].keys_and_values()[0]
            )
        )
        c2_ex = merged_cache.extract(1)
        self.assertTrue(mx.array_equal(c2_ex[0][0], c2[0][0]))
        self.assertTrue(
            mx.array_equal(
                c2_ex[1].keys_and_values()[0], c2[1].keys_and_values()[0]
            )
        )

    def test_make_mask_with_cache(self):
        # For 1 time step with no cache, don't need a mask
        mask = create_attention_mask(mx.zeros((1, 1)), cache=None, return_array=False)
        self.assertEqual(mask, None)

        mask = create_attention_mask(mx.zeros((1, 1)), cache=None, return_array=True)
        self.assertEqual(mask, None)

        # Regular causal mask
        mask = create_attention_mask(mx.zeros((1, 4)), cache=None, return_array=False)
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(mx.zeros((1, 4)), cache=None, return_array=True)
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4)))

        # With a window size
        mask = create_attention_mask(
            mx.zeros((1, 4)), cache=None, window_size=4, return_array=False
        )
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(
            mx.zeros((1, 4)), cache=None, window_size=3, return_array=False
        )
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4, window_size=3)))

        # With a regular KV cache
        cache = KVCache()
        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=False)
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=True)
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4)))

        k = v = mx.zeros((1, 2, 16, 8))
        cache.update_and_fetch(k, v)
        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=True)
        self.assertEqual(mask.shape, (4, 20))

    def test_kv_state_preserves_allocated_capacity(self):
        cache = KVCache()
        x = mx.arange(24).reshape(1, 1, 3, 8)
        cache.update_and_fetch(x, x)

        self.assertEqual(cache.keys_and_values()[0].shape[-2], 3)
        self.assertEqual(cache.state[0].shape[-2], cache.step)

        restored = KVCache.from_state(cache.state, cache.meta_state)
        self.assertEqual(restored.offset, 3)
        self.assertEqual(restored.state[0].shape[-2], cache.step)
        self.assertTrue(mx.array_equal(restored.keys_and_values()[0], x))

    def test_rotating_cache_mask(self):
        cache = RotatingKVCache(max_size=8)

        mask = cache.make_mask(4, window_size=5)
        self.assertEqual(mask, "causal")
        mask = create_attention_mask(mx.zeros((1, 4, 32)), cache, window_size=5)
        self.assertEqual(mask, "causal")
        mask = create_attention_mask(
            mx.zeros((1, 4, 32)), cache, window_size=5, return_array=True
        )
        self.assertEqual(mask.dtype, mx.bool_)
        self.assertEqual(mask.shape, (4, 4))

        mask = cache.make_mask(6, window_size=5)
        self.assertEqual(mask.dtype, mx.bool_)
        self.assertEqual(mask.sum(axis=-1).max(), 5)
        cmask = create_attention_mask(mx.zeros((1, 6, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask, None)
        mask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertEqual(mask, None)

        kv = mx.zeros((1, 1, 10, 32))
        cache.update_and_fetch(kv, kv)
        mask = cache.make_mask(3, window_size=5)
        self.assertEqual(mask.shape, (3, 10))
        self.assertTrue(mx.all(mask.sum(axis=-1) == 5))
        for i in range(3):
            s = 11 - 3 + i
            self.assertTrue(mx.all(mask[s - 5 : s]))
        cmask = create_attention_mask(mx.zeros((1, 3, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        mask = cache.make_mask(1)
        self.assertEqual(mask, None)
        mask = create_attention_mask(mx.zeros((1, 1, 32)), cache)
        self.assertEqual(mask, None)

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask.tolist(), [True] + [False] * 3 + [True] * 4)
        cmask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        kv = mx.zeros((1, 1, 1, 32))
        cache.update_and_fetch(kv, kv)

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask.tolist(), [True] * 2 + [False] * 3 + [True] * 3)
        cmask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

    def test_batch_kv_cache(self):
        cache = BatchKVCache(left_padding=[2, 3, 4])
        k, v = mx.zeros((3, 1, 4, 8)), mx.zeros((3, 1, 4, 8))
        # Update works
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(k.shape, (3, 1, 4, 8))

        # State can be evaluated
        mx.eval(cache.state)

        # State can be set
        cache.state = cache.state

        # Test filtering
        cache.filter([0, 1])

        # In this case filtering left shifts the cache so it has zero padding
        self.assertEqual(cache.keys_and_values()[0].shape, (2, 1, 2, 8))

        mask = cache.make_mask(1)
        self.assertEqual(mask[0].squeeze().tolist(), [True, True, True])
        self.assertEqual(mask[1].squeeze().tolist(), [False, True, True])

        # Test extension
        cache_a = BatchKVCache(left_padding=[2, 1, 2])
        cache_b = BatchKVCache(left_padding=[3, 0])

        k = mx.zeros((3, 1, 8, 1))
        v = mx.zeros((3, 1, 8, 1))
        cache_a.update_and_fetch(k, v)

        k = mx.zeros((2, 1, 4, 1))
        v = mx.zeros((2, 1, 4, 1))
        cache_b.update_and_fetch(k, v)

        cache_a.extend(cache_b)
        self.assertEqual(cache_a.keys.shape[0], 5)
        self.assertEqual(cache_a.values.shape[0], 5)
        self.assertEqual(cache_a.offset.tolist(), [6, 7, 6, 1, 4])
        self.assertEqual(cache_a.left_padding.tolist(), [2, 1, 2, 7, 4])

    def test_batch_kv_state_preserves_capacity_and_live_index(self):
        cache = BatchKVCache(left_padding=[1, 0])
        x = mx.arange(48).reshape(2, 1, 3, 8)
        cache.update_and_fetch(x, x)

        self.assertEqual(cache.keys_and_values()[0].shape[-2], 3)
        self.assertEqual(cache.state[0].shape[-2], cache.step)

        restored = BatchKVCache.from_state(cache.state, cache.meta_state)
        self.assertEqual(restored._idx, 3)
        self.assertEqual(restored.state[0].shape[-2], cache.step)
        self.assertTrue(mx.array_equal(restored.keys_and_values()[0], x))

    def test_batch_rotating_kv_cache(self):
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0])
        mask = cache.make_mask(4)
        self.assertFalse(mx.any(mask[0, 0, 0, :]))
        self.assertTrue(
            mx.array_equal(mask[1, 0, 0, :], mx.array([True, False, False, False]))
        )

        # Batch update works
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)

        mask = cache.make_mask(4)
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(mask.shape[-2:], (4, k.shape[2]))
        self.assertEqual(
            mask[0, 0, 0, :].tolist(), [False, True, True, True, False, False, False]
        )

        # Single query update works
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0])
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)

        mask = cache.make_mask(1)
        k, v = mx.zeros((2, 1, 1, 8)), mx.zeros((2, 1, 1, 8))

        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(mask.shape[-2:], (1, k.shape[2]))
        self.assertEqual(mask[0, 0, 0].tolist(), [True, False, True, True])
        self.assertEqual(mask[1, 0, 0].tolist(), [True, True, True, True])

        # Check filtering
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0, 3])
        k, v = mx.zeros((3, 1, 3, 8)), mx.zeros((3, 1, 3, 8))
        cache.update_and_fetch(k, v)
        cache.filter(mx.array([1]))
        self.assertEqual(cache.keys.shape, (1, 1, 3, 8))

        # Check extend
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 1])
        other = BatchRotatingKVCache(max_size=4, left_padding=[2, 2])
        k, v = mx.zeros((2, 1, 5, 8)), mx.zeros((2, 1, 5, 8))
        cache.update_and_fetch(k, v)
        other.update_and_fetch(k, v)
        k, v = mx.zeros((2, 1, 1, 8)), mx.zeros((2, 1, 1, 8))
        cache.update_and_fetch(k, v)
        cache.extend(other)

        # Check mask when going from prompt -> extend -> prompt
        cache = BatchRotatingKVCache(max_size=8, left_padding=[4])
        k, v = mx.zeros((1, 1, 8, 8)), mx.zeros((1, 1, 8, 8))
        cache.update_and_fetch(k, v)

        mask = cache.make_mask(1)
        self.assertEqual(
            mask.squeeze().tolist(), [True, False, False, False, True, True, True, True]
        )

        k, v = mx.zeros((1, 1, 1, 8)), mx.zeros((1, 1, 1, 8))
        cache.update_and_fetch(k, v)

        mask = cache.make_mask(2)
        expected = mx.array(
            [
                [False, False, False, True, True, True, True, True, False],
                [False, False, False, True, True, True, True, True, True],
            ]
        )
        self.assertTrue(mx.array_equal(mask.squeeze(), expected))

    def test_save_load_batch_caches(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2, left_padding=[1, 2]),
            BatchKVCache(left_padding=[1, 2]),
            BatchRotatingKVCache(max_size=10, left_padding=[1, 2]),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        left_padding = mx.array([1, 2])
        for c, lc in zip(cache, loaded_cache):
            self.assertTrue(mx.array_equal(c.left_padding, left_padding))

    def test_rotating_cache_updates(self):
        cache = RotatingKVCache(max_size=8)
        k = v = mx.zeros((1, 1, 10, 1))
        cache.update_and_fetch(k, v)

        for _ in range(3):
            k = v = mx.zeros((1, 1, 1, 1))
            cache.update_and_fetch(k, v)

        k = v = mx.zeros((1, 1, 3, 1))
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(k.shape[2], 10)
        self.assertEqual(v.shape[2], 10)

    def test_merge_with_empty_caches(self):
        c1 = ArraysCache(2)
        c2 = ArraysCache(2)
        c2[0] = mx.zeros((1, 4))
        c2[1] = mx.zeros((1, 4))
        c_out = ArraysCache.merge((c1, c2))
        self.assertEqual(c_out[0].shape, (2, 4))
        self.assertEqual(c_out[1].shape, (2, 4))

        c1 = KVCache()
        c2 = KVCache()
        kv = mx.zeros((1, 4, 4, 4))
        c2.update_and_fetch(kv, kv)
        c_out = KVCache.merge((c1, c2))
        self.assertEqual(c_out.keys.shape, (2, 4, 4, 4))

        c1 = RotatingKVCache(max_size=4)
        c2 = RotatingKVCache(max_size=4)
        kv = mx.zeros((1, 4, 4, 4))
        c2.update_and_fetch(kv, kv)
        c_out = KVCache.merge((c1, c2))
        self.assertEqual(c_out.keys.shape, (2, 4, 4, 4))

    def test_extend_with_empty_and_nonempty_batch_caches(self):
        """Extending a batch cache when one side has keys=None should use the
        correct batch size for the placeholder, not the batch size from the
        non-None side. Regression test for broadcast error in dynamic_roll."""
        H, D = 8, 64
        max_size = 512

        # -- BatchRotatingKVCache --
        # Create 2 caches with content and 3 empty caches
        c1 = RotatingKVCache(max_size=max_size)
        c2 = RotatingKVCache(max_size=max_size)
        c1.update_and_fetch(mx.ones((1, H, 5, D)), mx.ones((1, H, 5, D)))
        c2.update_and_fetch(mx.ones((1, H, 3, D)), mx.ones((1, H, 3, D)))
        batch_full = BatchRotatingKVCache.merge([c1, c2])

        empty_caches = [RotatingKVCache(max_size=max_size) for _ in range(3)]
        batch_empty = BatchRotatingKVCache.merge(empty_caches)

        # Extend non-empty with empty (different batch sizes)
        batch_full.extend(batch_empty)
        self.assertEqual(batch_full.keys.shape[0], 5)
        self.assertEqual(batch_full.offset.shape[0], 5)

        # Prompt processing with right padding should not crash
        batch_full.prepare(lengths=[10, 8, 12, 7, 11], right_padding=[2, 4, 0, 5, 1])
        new_kv = mx.ones((5, H, 12, D))
        batch_full.update_and_fetch(new_kv, new_kv)

        # Also test empty extending non-empty
        batch_full2 = BatchRotatingKVCache.merge(
            [RotatingKVCache(max_size=max_size) for _ in range(3)]
        )
        c3 = RotatingKVCache(max_size=max_size)
        c4 = RotatingKVCache(max_size=max_size)
        c3.update_and_fetch(mx.ones((1, H, 4, D)), mx.ones((1, H, 4, D)))
        c4.update_and_fetch(mx.ones((1, H, 6, D)), mx.ones((1, H, 6, D)))
        batch_content = BatchRotatingKVCache.merge([c3, c4])
        batch_full2.extend(batch_content)
        self.assertEqual(batch_full2.keys.shape[0], 5)
        self.assertEqual(batch_full2.offset.shape[0], 5)

        # -- BatchKVCache --
        c1 = KVCache()
        c2 = KVCache()
        c1.update_and_fetch(mx.ones((1, H, 5, D)), mx.ones((1, H, 5, D)))
        c2.update_and_fetch(mx.ones((1, H, 3, D)), mx.ones((1, H, 3, D)))
        batch_full = BatchKVCache.merge([c1, c2])

        empty_caches = [KVCache() for _ in range(3)]
        batch_empty = BatchKVCache.merge(empty_caches)

        batch_full.extend(batch_empty)
        self.assertEqual(batch_full.keys.shape[0], 5)
        self.assertEqual(batch_full.offset.shape[0], 5)

    def test_extend_with_empty_batch_cache_preserves_kv_dtypes(self):
        """Empty placeholders must not promote either K or V to float32."""
        H, Dk, Dv = 2, 8, 6

        def make_batch(base_cls, batch_cls, n, with_content, **kwargs):
            caches = [base_cls(**kwargs) for _ in range(n)]
            if with_content:
                for cache in caches:
                    keys = mx.ones((1, H, 5, Dk), dtype=mx.bfloat16)
                    values = mx.ones((1, H, 5, Dv), dtype=mx.float16)
                    cache.update_and_fetch(keys, values)
            batch = batch_cls.merge(caches)
            if with_content:
                # Batch merge currently follows the key dtype for both
                # buffers. Restore a distinct value dtype here to isolate the
                # extend() placeholder contract under test.
                batch.values = batch.values.astype(mx.float16)
            return batch

        cases = (
            (KVCache, BatchKVCache, {}),
            (RotatingKVCache, BatchRotatingKVCache, {"max_size": 16}),
        )
        for base_cls, batch_cls, kwargs in cases:
            for direction in ("append", "prepend"):
                populated = make_batch(base_cls, batch_cls, 2, True, **kwargs)
                empty = make_batch(base_cls, batch_cls, 1, False, **kwargs)
                if direction == "append":
                    populated.extend(empty)
                    result = populated
                else:
                    empty.extend(populated)
                    result = empty
                message = f"{batch_cls.__name__} {direction}"
                self.assertEqual(result.keys.dtype, mx.bfloat16, message)
                self.assertEqual(result.values.dtype, mx.float16, message)

    def test_arrays_cache_extend_with_empty(self):
        # test simple merge
        c1 = ArraysCache(2)
        c2 = ArraysCache(2)
        c1[0] = mx.zeros((1, 4, 8))
        c1[1] = mx.zeros((1, 4))
        c2[0] = mx.zeros((1, 4, 8))
        c2[1] = mx.zeros((1, 4))
        full = ArraysCache.merge((c1, c2))
        self.assertEqual(full[0].shape, (2, 4, 8))

        # extend with empty
        empty = ArraysCache.merge((ArraysCache(2),))
        full.extend(empty)
        self.assertEqual(full[0].shape, (3, 4, 8))
        self.assertEqual(full[1].shape, (3, 4))
        self.assertTrue(mx.all(full[0][2:] == 0))

        # making an empty cache with 2 sequences and merging it with
        # another one with 2 sequences
        empty2 = ArraysCache.merge((ArraysCache(2), ArraysCache(2)))
        content = ArraysCache.merge((c1, c2))
        empty2.extend(content)
        self.assertEqual(empty2[0].shape, (4, 4, 8))
        self.assertEqual(empty2[1].shape, (4, 4))

        # Extend content with empty
        content = ArraysCache.merge((c1, c2))
        empty2 = ArraysCache.merge((ArraysCache(2), ArraysCache(2)))
        content.extend(empty2)
        self.assertEqual(content[0].shape, (4, 4, 8))
        self.assertEqual(content[1].shape, (4, 4))
        self.assertEqual(content.make_mask(10).shape, (4, 10))

        # multiple empty extensions accumulate correctly
        stepwise = ArraysCache.merge((c1,))
        stepwise.extend(ArraysCache(2))
        stepwise.extend(ArraysCache.merge((ArraysCache(2), ArraysCache(2))))
        self.assertEqual(stepwise[0].shape, (4, 4, 8))
        self.assertEqual(stepwise[1].shape, (4, 4))

    def test_batch_rotating_meta_state_rotated_roundtrip(self):
        # bool("False") is True — a persisted un-rotated batch cache must not
        # restore as rotated (silent ring-order corruption on restore).
        for cls, kwargs in (
            (BatchRotatingKVCache, {}),
            (BatchRotatingQuantizedKVCache, {"group_size": 64, "bits": 4}),
        ):
            for rotated in (False, True):
                c = cls(8, [0, 0], **kwargs)
                c.rotated = rotated
                restored = cls(8, [0, 0], **kwargs)
                restored.meta_state = c.meta_state
                self.assertIs(restored.rotated, rotated, cls.__name__)
                # Tolerate legacy string-encoded payloads from older files.
                legacy = list(c.meta_state)
                legacy[3] = str(rotated)
                restored.meta_state = tuple(legacy)
                self.assertIs(restored.rotated, rotated, cls.__name__)

    def test_arrays_cache_advance_evaluates_metadata_with_state(self):
        # mlx-lm#1632/#1642: without tying metadata into the state graph,
        # every advance() leaves one dead lazy node per un-masked layer per
        # token, pinning a Metal buffer object each (499k/process limit).
        cache = ArraysCache(2, left_padding=[2])
        cache.prepare(lengths=[3])
        cache[0] = mx.array([0])
        cache[1] = mx.array([0])

        for _ in range(256):
            cache[0] = cache[0] + 1
            cache.advance(1)
            mx.eval(cache[0])

        for name, metadata in (
            ("lengths", cache.lengths),
            ("left-padding", cache.left_padding),
        ):
            dot_path = os.path.join(self.test_dir, f"arrays-cache-{name}.dot")
            mx.export_to_dot(dot_path, metadata)
            with open(dot_path, encoding="utf-8") as graph:
                self.assertLessEqual(graph.read().count("->"), 8)
        self.assertEqual(cache[0].item(), 256)
        self.assertEqual(cache.lengths.item(), 3 - 256)
        self.assertEqual(cache.left_padding.item(), 2 - 256)

    def test_arrays_cache_extract_copies(self):
        # mlx-lm#1701: a view slice pins the whole batched parent buffer;
        # extract must copy so re-inserted lanes don't transitively retain
        # every previous batch's recurrent state.
        cache = ArraysCache(1)
        big = mx.zeros((8, 4, 512, 512))  # 8 lanes x 4 MB
        mx.eval(big)
        cache[0] = big
        child = cache.extract(0)
        mx.eval(child.cache[0])
        self.assertEqual(child.cache[0].shape, (1, 4, 512, 512))
        before = mx.get_active_memory()
        del cache, big
        import gc

        gc.collect()
        after = mx.get_active_memory()
        # The 32 MB parent must actually free; the child holds ~4 MB.
        self.assertLess(after, before - 16 * 1024 * 1024)
        # None entries (post-filter) must pass through extract untouched.
        holey = ArraysCache(2)
        holey[0] = mx.zeros((2, 3))
        lane = holey.extract(1)
        self.assertIsNone(lane.cache[1])

    def test_window_mask_with_full_kv_cache(self):
        c = KVCache()
        kv = mx.zeros((1, 1, 32, 128))
        c.update_and_fetch(kv, kv)

        h = mx.zeros((1, 1, 1, 128))
        mask = create_attention_mask(h, c, window_size=4)
        expected = create_causal_mask(1, offset=32, window_size=4)
        self.assertTrue(mx.array_equal(mask, expected))


class TestModelLocalCacheClasses(unittest.TestCase):
    """Cache subclasses defined in model files (not in models/cache.py) must
    round-trip through save_prompt_cache / load_prompt_cache. They resolve
    through the cache-class registry and module-qualified class tokens rather
    than cache.py globals."""

    def setUp(self):
        self.test_dir_fid = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_fid.name

    def tearDown(self):
        self.test_dir_fid.cleanup()

    def _make_qwen4_exp_caches(self):
        from mlx_lm.models.qwen4_exp import (
            BatchQSAKVCache,
            QSAKVCache,
            Qwen4ArraysCache,
        )

        qsa = QSAKVCache()
        x = mx.random.uniform(shape=(1, 8, 10, 4))
        qsa.update_and_fetch(x, x)
        qsa.update_index_keys(mx.random.uniform(shape=(1, 10, 16)))

        arrays = Qwen4ArraysCache(size=4)
        for i in range(4):
            arrays[i] = mx.random.uniform(shape=(1, 3, 5))

        batch = BatchQSAKVCache([1, 0])
        xb = mx.random.uniform(shape=(2, 8, 10, 4))
        batch.update_and_fetch(xb, xb)
        batch.update_index_keys(mx.random.uniform(shape=(2, 10, 16)))
        return [qsa, arrays, batch]

    def _assert_caches_equal(self, cache, loaded_cache):
        from mlx.utils import tree_flatten

        self.assertEqual(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertIs(type(lc), type(c))
            self.assertEqual(c.meta_state, lc.meta_state)
            state, loaded_state = tree_flatten(c.state), tree_flatten(lc.state)
            self.assertEqual([k for k, _ in state], [k for k, _ in loaded_state])
            for (k, a), (_, b) in zip(state, loaded_state):
                self.assertTrue(mx.array_equal(a, b), f"state leaf {k} differs")

    def test_qwen4_exp_cache_round_trip(self):
        cache = self._make_qwen4_exp_caches()
        cache_file = os.path.join(self.test_dir, "qwen4_exp_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self._assert_caches_equal(cache, loaded_cache)

        # Generation can continue on the reloaded caches.
        qsa = loaded_cache[0]
        x = mx.random.uniform(shape=(1, 8, 1, 4))
        keys, _ = qsa.update_and_fetch(x, x)
        self.assertEqual(qsa.offset, 11)
        self.assertEqual(keys.shape[2], 11)

    def test_qsa_summary_and_provenance_round_trip(self):
        from mlx_lm.models import qwen4_exp
        from mlx_lm.models.qwen4_exp import QSAKVCache

        previous = qwen4_exp._QSA_APC_SUMMARIES
        qwen4_exp._QSA_APC_SUMMARIES = True
        try:
            qsa = QSAKVCache()
            values = mx.zeros((1, 1, 8, 4))
            qsa.update_and_fetch(values, values)
            qsa.update_index_keys(mx.zeros((1, 8, 16)))
            qsa._qsa_pooled_keys = mx.arange(32).reshape(1, 2, 16)
            qsa._qsa_pooled_ratio = 4
            qsa._qsa_summary_identity = {
                "format_version": 1,
                "model_config_hash": "tiny-model-config",
                "block_size": 4,
                "compress_ratio": 4,
                "producer_version": "qwen4-pooled-key-v1",
                "layer_id": "3",
                "complete_blocks": 2,
            }
            path = os.path.join(self.test_dir, "qsa_summary.safetensors")
            save_prompt_cache(path, [qsa], {"user": "metadata"})
            loaded, user_metadata = load_prompt_cache(
                path, return_metadata=True
            )
            self.assertEqual(user_metadata, {"user": "metadata"})
            self.assertTrue(
                mx.array_equal(
                    loaded[0]._qsa_pooled_keys, qsa._qsa_pooled_keys
                ).item()
            )
            _, raw_metadata = mx.load(path, return_metadata=True)
            values = list(raw_metadata.values())
            self.assertIn("qsa_summary_v1", values)
            self.assertTrue(
                any('"format": "qsa_apc_summaries"' in v for v in values)
            )

            nested_path = os.path.join(
                self.test_dir, "qsa_summary_cache_list.safetensors"
            )
            save_prompt_cache(nested_path, [CacheList(qsa)])
            _, nested_metadata = mx.load(nested_path, return_metadata=True)
            self.assertTrue(
                any(
                    '"cache_path": [0, 0]' in value
                    for value in nested_metadata.values()
                )
            )
        finally:
            qwen4_exp._QSA_APC_SUMMARIES = previous

    def test_qsa_summary_env_unset_keeps_legacy_state(self):
        import subprocess
        import sys

        env = os.environ.copy()
        env.pop("MLX_QWEN4_QSA_APC_SUMMARIES", None)
        script = (
            "from mlx_lm.models import qwen4_exp\n"
            "from mlx_lm.models.qwen4_exp import QSAKVCache\n"
            "assert qwen4_exp._QSA_APC_SUMMARIES is False\n"
            "assert len(QSAKVCache().state) == 3\n"
            "status = qwen4_exp.qsa_apc_summary_status()\n"
            "assert status['enabled'] is False\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cache_list_with_model_local_members(self):
        # CacheList serializes its members' classes itself; model-local
        # members must survive that path too.
        cache = [CacheList(*self._make_qwen4_exp_caches())]
        cache_file = os.path.join(self.test_dir, "qwen4_exp_cache_list.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertIs(type(loaded_cache[0]), CacheList)
        self._assert_caches_equal(cache[0].caches, loaded_cache[0].caches)

    def test_load_without_importing_model_module(self):
        # A fresh process that never imported qwen4_exp must still reload the
        # cache: the saved module-qualified class tokens carry the import.
        import subprocess
        import sys

        cache_file = os.path.join(self.test_dir, "qwen4_exp_cache.safetensors")
        save_prompt_cache(cache_file, self._make_qwen4_exp_caches())
        script = (
            "import sys\n"
            "from mlx_lm.models.cache import load_prompt_cache\n"
            "assert 'mlx_lm.models.qwen4_exp' not in sys.modules\n"
            f"cache = load_prompt_cache({cache_file!r})\n"
            "print(','.join(type(c).__name__ for c in cache))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(), "QSAKVCache,Qwen4ArraysCache,BatchQSAKVCache"
        )

    def test_legacy_bare_name_resolves_via_registry(self):
        # Files saved before class tokens were module-qualified hold bare
        # names; once the model module is imported, the registry filled by
        # _BaseCache.__init_subclass__ resolves them.
        from unittest import mock

        from mlx_lm.models import cache as cache_module

        cache = self._make_qwen4_exp_caches()
        cache_file = os.path.join(self.test_dir, "legacy_cache.safetensors")
        with mock.patch.object(
            cache_module, "_cache_class_token", lambda cls: cls.__name__
        ):
            save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self._assert_caches_equal(cache, loaded_cache)


class TestEmptyArrayRoundTrip(unittest.TestCase):
    """Zero-sized state entries must survive save/load.

    safetensors cannot hold a zero-sized array -- mlx < 0.32.1 refuses one
    outright -- and cache states use them as "no value" sentinels, so saving
    any cache with an unfilled state slot used to fail. They are carried in
    the metadata now and rebuilt on load.
    """

    def setUp(self):
        self.test_dir_fid = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_fid.name
        self.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self.device)
        self.test_dir_fid.cleanup()

    def _round_trip(self, cache, name="cache", metadata={}):
        path = os.path.join(self.test_dir, f"{name}.safetensors")
        save_prompt_cache(path, cache, metadata)
        return load_prompt_cache(path, return_metadata=bool(metadata))

    def _assert_entries_match(self, original, loaded):
        from mlx.utils import tree_flatten

        self.assertEqual(len(original), len(loaded))
        for entry, (want, got) in enumerate(zip(original, loaded)):
            self.assertIs(type(got), type(want))
            flat_want = tree_flatten(want.state)
            flat_got = tree_flatten(got.state)
            self.assertEqual(
                [k for k, _ in flat_want], [k for k, _ in flat_got], f"entry {entry}"
            )
            for (key, a), (_, b) in zip(flat_want, flat_got):
                self.assertEqual(a.shape, b.shape, f"entry {entry} {key} shape")
                self.assertEqual(a.dtype, b.dtype, f"entry {entry} {key} dtype")
                self.assertTrue(
                    mx.array_equal(a, b).item(), f"entry {entry} {key} values"
                )

    def _arrays_cache(self, entries):
        cache = ArraysCache(len(entries))
        cache.cache = list(entries)
        return cache

    def test_empty_state_slot_round_trips(self):
        cache = [self._arrays_cache([mx.zeros((1, 0, 4)), mx.arange(6.0).reshape(1, 6)])]
        loaded = self._round_trip(cache, "empty_slot")
        self._assert_entries_match(cache, loaded)
        # The zero dimension is preserved, not collapsed away.
        self.assertEqual(loaded[0][0].shape, (1, 0, 4))
        self.assertIsNone(loaded[0].left_padding)
        self.assertIsNone(loaded[0].lengths)

    def test_no_zero_sized_tensor_is_written(self):
        """The portability property, independent of the mlx in use.

        mlx >= 0.32.1 tolerates a zero-sized tensor, so a round-trip alone
        passes here whether or not the sentinel is stripped. Assert on the
        file: nothing zero-sized reaches safetensors, and the spec that
        rebuilds it is present.
        """
        cache = [
            self._arrays_cache([mx.zeros((1, 0, 4)), mx.arange(6.0).reshape(1, 6)])
        ]
        path = os.path.join(self.test_dir, "no_zero_sized.safetensors")
        save_prompt_cache(path, cache)
        arrays, raw = mx.load(path, return_metadata=True)

        self.assertTrue(arrays, "everything was stripped")
        for key, value in arrays.items():
            self.assertGreater(value.size, 0, f"{key} was written zero-sized")
        spec = [raw[key] for key in raw if key.split(".")[0] == "3"]
        self.assertEqual(len(spec), 1, "no empty-array spec was recorded")
        self.assertIn("0.0.0", json.loads(spec[0]))

    def test_every_entry_empty_round_trips(self):
        cache = [self._arrays_cache([mx.zeros((0,)), mx.zeros((2, 0))])]
        loaded = self._round_trip(cache, "all_empty")
        self._assert_entries_match(cache, loaded)

    def test_partially_populated_cache_round_trips(self):
        keys = mx.random.uniform(shape=(1, 4, 6, 8))
        kv = KVCache()
        kv.update_and_fetch(keys, keys)
        arrays = self._arrays_cache(
            [mx.arange(4.0).reshape(1, 4), mx.zeros((1, 0, 8)), mx.zeros((1, 3))]
        )
        cache = [kv, arrays]
        loaded = self._round_trip(cache, "partial")
        self._assert_entries_match(cache, loaded)
        self.assertEqual(loaded[0].offset, kv.offset)

    def test_cache_list_with_mixed_members_round_trips(self):
        keys = mx.random.uniform(shape=(1, 2, 5, 4))
        kv = KVCache()
        kv.update_and_fetch(keys, keys)
        arrays = self._arrays_cache([mx.zeros((1, 0, 4)), mx.arange(3.0).reshape(1, 3)])
        cache = [CacheList(kv, arrays)]
        loaded = self._round_trip(cache, "cache_list")

        self.assertIsInstance(loaded[0], CacheList)
        self.assertEqual(len(loaded[0].caches), 2)
        self._assert_entries_match(cache[0].caches, loaded[0].caches)

    def test_dtypes_and_zero_dims_are_exact(self):
        cache = [
            self._arrays_cache(
                [
                    mx.zeros((1, 0, 4), mx.bfloat16),
                    mx.zeros((0,), mx.bool_),
                    mx.arange(6).reshape(1, 6).astype(mx.uint32),
                    mx.zeros((0, 3, 0), mx.float16),
                ]
            )
        ]
        loaded = self._round_trip(cache, "dtypes")
        self._assert_entries_match(cache, loaded)

    def test_reloaded_cache_keeps_working(self):
        arrays = self._arrays_cache([mx.zeros((1, 0, 4)), mx.arange(3.0).reshape(1, 3)])
        loaded = self._round_trip([arrays], "usable")[0]
        # A restored cache is a live cache: its slots take new state.
        loaded[0] = mx.ones((1, 2, 4))
        self.assertEqual(loaded[0].shape, (1, 2, 4))
        self.assertEqual(loaded.rollback_spans(3), ())

    def test_user_metadata_is_not_polluted(self):
        cache = [self._arrays_cache([mx.zeros((1, 0, 4))])]
        loaded, metadata = self._round_trip(
            cache, "metadata", metadata={"model": "tiny", "n": "3"}
        )
        self.assertEqual(metadata, {"model": "tiny", "n": "3"})
        self._assert_entries_match(cache, loaded)

    def test_cache_without_empty_arrays_keeps_the_old_layout(self):
        keys = mx.random.uniform(shape=(1, 2, 4, 4))
        kv = KVCache()
        kv.update_and_fetch(keys, keys)
        path = os.path.join(self.test_dir, "no_empties.safetensors")
        save_prompt_cache(path, [kv])
        _, raw = mx.load(path, return_metadata=True)
        # Nothing is appended, so a reader that predates this stays correct.
        self.assertFalse(any(key.split(".")[0] == "3" for key in raw))
        self._assert_entries_match([kv], load_prompt_cache(path))


if __name__ == "__main__":
    unittest.main()
