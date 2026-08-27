# Copyright © 2026 Apple Inc.
#
# Tests for the 2026-08-26 model-code micro-lever bundle.  Every lever is an
# import-time env flag whose module-level constant is toggled directly here.
#
# Flag-off residue (review item f, accepted): with every flag unset the code
# adds no MLX tensor ops, but it does add Python-level residue — two pooled
# cache attributes on QSAKVCache, a per-trim truncation check, and dispatch
# through _pooled_keys() — plus the unconditional (byte-identical) RoPE
# frequency memo and host-side hash-constant copies.

import math
import unittest
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.generate import _merge_caches
from mlx_lm.models import qwen3_5, qwen3_next, qwen4_exp
from mlx_lm.models.cache import ArraysCache, trim_prompt_cache
from mlx_lm.models.qwen3_5 import (
    _array_bytes,
    _can_fuse_gdn_projections,
    _fuse_gdn_projection_layer,
    fuse_gated_delta_net_projections,
)
from mlx_lm.models.qwen4_exp import (
    GroupRMSNorm,
    Model,
    ModelArgs,
    NGramEmbedding,
    QSAIndexer,
    QSAKVCache,
    ShardedEmbedding,
    TextModelArgs,
    _apply_rope_positions,
)


@contextmanager
def lever(module, name, value=True):
    previous = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, previous)


def tiny_args(**overrides):
    values = dict(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
        },
    )
    values.update(overrides)
    return TextModelArgs(**values)


def _bytes_equal(test, left, right):
    mx.eval(left, right)
    test.assertEqual(left.dtype, right.dtype)
    test.assertEqual(left.shape, right.shape)
    test.assertEqual(_array_bytes(left), _array_bytes(right))


class TestRMSNormFast(unittest.TestCase):
    def _reference(self, norm, x):
        dtype = x.dtype
        xf = x.astype(mx.float32)
        if norm.group_size is not None:
            xf = xf.reshape(*xf.shape[:-1], -1, norm.group_size)
        out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + norm.eps)
        if norm.group_size is not None:
            out = out.reshape(*x.shape)
        return (out * norm.weight.astype(mx.float32)).astype(dtype)

    def _norms(self):
        grouped = GroupRMSNorm(32, 8, 1e-6)
        flat = GroupRMSNorm(32, None, 1e-6)
        for norm in (grouped, flat):
            norm.weight = mx.random.uniform(0.5, 1.5, (32,), key=mx.random.key(7))
        return grouped, flat

    def test_flag_off_is_byte_identical_to_reference(self):
        self.assertFalse(qwen4_exp._RMSNORM_FAST)
        x = mx.random.normal((2, 3, 32), key=mx.random.key(0)).astype(mx.float16)
        for norm in self._norms():
            _bytes_equal(self, norm(x), self._reference(norm, x))

    def test_flag_on_matches_within_fp16_tolerance(self):
        x = mx.random.normal((2, 5, 32), key=mx.random.key(1)).astype(mx.float16)
        for norm in self._norms():
            expected = self._reference(norm, x).astype(mx.float32)
            with lever(qwen4_exp, "_RMSNORM_FAST"):
                actual = norm(x).astype(mx.float32)
            mx.eval(expected, actual)
            rel = mx.abs(actual - expected) / (mx.abs(expected) + 1e-6)
            self.assertLess(rel.max().item(), 1e-3)


class TestQSAPooledKeyCache(unittest.TestCase):
    def _drive(self, indexer, chunks, trim_at=None, trim_n=0):
        cache = QSAKVCache()
        outputs = []
        for index, chunk in enumerate(chunks):
            length = chunk.shape[1]
            if length > 1 and cache.offset == 0:
                mask = (
                    mx.arange(length)[:, None] >= mx.arange(length)[None, :]
                )[None, None]
            else:
                mask = None
            sparse = indexer(chunk, mask, cache)
            if sparse is not None:
                mx.eval(sparse)
                outputs.append(np.asarray(sparse))
            else:
                outputs.append(None)
            cache.offset += length
            if trim_at is not None and index == trim_at:
                cache.trim(trim_n)
        return outputs

    def _compare_runs(self, flag_name, trim_at=None, trim_n=0):
        args = tiny_args()
        indexer = QSAIndexer(args)
        chunks = [mx.random.normal((1, 9, args.hidden_size), key=mx.random.key(0))]
        chunks += [
            mx.random.normal((1, 1, args.hidden_size), key=mx.random.key(step))
            for step in range(1, 24)
        ]
        stock = self._drive(indexer, chunks, trim_at, trim_n)
        with lever(qwen4_exp, flag_name):
            fast = self._drive(indexer, chunks, trim_at, trim_n)
        self.assertEqual(len(stock), len(fast))
        for step, (expected, actual) in enumerate(zip(stock, fast)):
            if expected is None:
                self.assertIsNone(actual, f"step {step}")
            else:
                np.testing.assert_array_equal(actual, expected, f"step {step}")

    def test_selected_blocks_identical_over_long_decode(self):
        self._compare_runs("_QSA_POOLED_KEY_CACHE")

    def test_selected_blocks_identical_across_mid_block_trim(self):
        self._compare_runs("_QSA_POOLED_KEY_CACHE", trim_at=14, trim_n=3)

    def test_shared_topk_cycle_matches(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        chunks = [mx.random.normal((1, 9, args.hidden_size), key=mx.random.key(0))]
        chunks += [
            mx.random.normal((1, 1, args.hidden_size), key=mx.random.key(step))
            for step in range(1, 4)
        ]

        def drive():
            cache = QSAKVCache()
            outputs = []
            for index, chunk in enumerate(chunks):
                length = chunk.shape[1]
                mask = None
                if length > 1:
                    mask = (
                        mx.arange(length)[:, None] >= mx.arange(length)[None, :]
                    )[None, None]
                if index == 2:
                    cache._mtp_share_topk = True
                    cache._mtp_shared_topk = None
                sparse = indexer(chunk, mask, cache)
                mx.eval(sparse)
                outputs.append(np.asarray(sparse))
                cache.offset += length
            return outputs

        stock = drive()
        with lever(qwen4_exp, "_QSA_POOLED_KEY_CACHE"):
            fast = drive()
        for expected, actual in zip(stock, fast):
            np.testing.assert_array_equal(actual, expected)

    def _run_model_sequence(self, model):
        cache = model.make_cache()
        outputs = []

        def step(tokens):
            logits = model(mx.array([tokens], dtype=mx.int32), cache=cache)
            mx.eval(logits)
            outputs.append(np.asarray(logits))

        step([1, 2, 3, 4, 5])
        for token in range(6, 18):
            step([token % 60])
        for layer_cache in cache:
            layer_cache.start_speculation()
        step([7, 8, 9])
        trim_prompt_cache(cache, 2)
        for token in range(3):
            step([10 + token])
        return outputs

    def test_full_model_logits_bitwise_identical_with_rollback(self):
        args = tiny_args(ple_layer_ids=[2])
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        stock = self._run_model_sequence(model)
        with lever(qwen4_exp, "_QSA_POOLED_KEY_CACHE"):
            fast = self._run_model_sequence(model)
        for step, (expected, actual) in enumerate(zip(stock, fast)):
            np.testing.assert_array_equal(actual, expected, f"step {step}")

    def _run_batch_sequence(self, model):
        cache = _merge_caches([model.make_cache()])
        outputs = []
        for tokens in ([[1, 2, 3]], [[4]], [[5]]):
            logits = model(mx.array(tokens, dtype=mx.int32), cache=cache)
            mx.eval(logits)
            outputs.append(np.asarray(logits))
        extracted = [layer_cache.extract(0) for layer_cache in cache]
        remerged = _merge_caches([extracted])
        logits = model(mx.array([[6]], dtype=mx.int32), cache=remerged)
        mx.eval(logits)
        outputs.append(np.asarray(logits))
        return outputs

    def test_batch_merge_and_extract_match_with_flag_on(self):
        args = tiny_args(ple_layer_ids=[2])
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        model = model.language_model
        stock = self._run_batch_sequence(model)
        with lever(qwen4_exp, "_QSA_POOLED_KEY_CACHE"):
            fast = self._run_batch_sequence(model)
        for expected, actual in zip(stock, fast):
            np.testing.assert_array_equal(actual, expected)


class TestQSAScatterChosen(unittest.TestCase):
    def test_masks_identical_including_shared_topk(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        chunks = [mx.random.normal((1, 9, args.hidden_size), key=mx.random.key(0))]
        chunks += [
            mx.random.normal((1, 1, args.hidden_size), key=mx.random.key(step))
            for step in range(1, 8)
        ]

        def drive():
            cache = QSAKVCache()
            outputs = []
            for index, chunk in enumerate(chunks):
                length = chunk.shape[1]
                mask = None
                if length > 1:
                    mask = (
                        mx.arange(length)[:, None] >= mx.arange(length)[None, :]
                    )[None, None]
                if index == 4:
                    cache._mtp_share_topk = True
                    cache._mtp_shared_topk = None
                sparse = indexer(chunk, mask, cache)
                mx.eval(sparse)
                outputs.append(np.asarray(sparse))
                cache.offset += length
            return outputs

        stock = drive()
        with lever(qwen4_exp, "_QSA_SCATTER_CHOSEN"):
            scattered = drive()
        for step, (expected, actual) in enumerate(zip(stock, scattered)):
            np.testing.assert_array_equal(actual, expected, f"step {step}")


class TestPLEVectorShift(unittest.TestCase):
    def _embedding(self):
        return NGramEmbedding(tiny_args(), 16, layer_idx=1, ple_layer_index=0)

    def test_matches_loop_on_eos_edge_cases(self):
        emb = self._embedding()
        eos = 63
        cases = [
            [[1, 2, 3, 4, 5]],
            [[eos, 1, 2, 3]],
            [[1, 2, 3, eos]],
            [[eos, eos, 1, eos, 2, 3, eos]],
            [[eos], [1]],
            [[1, eos, eos, 2], [eos, 3, 4, eos]],
        ]
        rng = np.random.default_rng(0)
        for _ in range(8):
            tokens = rng.integers(0, 63, size=(2, 11))
            tokens[rng.random(tokens.shape) < 0.2] = eos
            cases.append(tokens.tolist())
        for case in cases:
            tokens = mx.array(case, dtype=mx.int64)
            expected = emb._ngram_ids_numpy(tokens, None)
            with lever(qwen4_exp, "_PLE_VECTOR_SHIFT"):
                actual = emb._ngram_ids_numpy(tokens, None)
            np.testing.assert_array_equal(actual, expected)

    def test_matches_loop_with_cache_continuation(self):
        emb = self._embedding()
        chunks = [
            mx.array([[1, 2, 63, 3], [63, 4, 5, 6]], dtype=mx.int64),
            mx.array([[4, 63], [7, 63]], dtype=mx.int64),
            mx.array([[8], [63]], dtype=mx.int64),
        ]
        stock_cache, fast_cache = ArraysCache(4), ArraysCache(4)
        for chunk in chunks:
            expected = emb._ngram_ids_numpy(chunk, stock_cache)
            with lever(qwen4_exp, "_PLE_VECTOR_SHIFT"):
                actual = emb._ngram_ids_numpy(chunk, fast_cache)
            np.testing.assert_array_equal(actual, expected)


class TestPLEGatherConcat(unittest.TestCase):
    def test_sharded_lookup_matches(self):
        emb = ShardedEmbedding(32, 8, 4)
        cases = [
            np.array([[0, 31, 8, 8, 15, 16, 7, 24]]),
            np.array([[1, 2], [3, 1]]),
            np.array([[9, 9, 9]]),  # single shard, duplicates
        ]
        for indices in cases:
            expected = emb.lookup_numpy(indices)
            with lever(qwen4_exp, "_PLE_GATHER_CONCAT"):
                actual = emb.lookup_numpy(indices)
            _bytes_equal(self, actual, expected)

    def test_ngram_embedding_forward_matches(self):
        emb = NGramEmbedding(tiny_args(), 16, layer_idx=1, ple_layer_index=0)
        tokens = mx.array([[1, 2, 63, 3, 4]], dtype=mx.int64)
        expected = emb(tokens)
        with lever(qwen4_exp, "_PLE_GATHER_CONCAT"):
            actual = emb(tokens)
        _bytes_equal(self, actual, expected)


class TestMoEGateCompile(unittest.TestCase):
    """MLX_QWEN4_MOE_GATE_COMPILE (shared: also compiles the qwen3_next and
    qwen3_5 MoE blocks).  Shaped traces are width-keyed, so only stable
    narrow shapes (decode / MTP verify) compile; prefill widths stay eager."""

    def test_moe_block_output_bitwise_identical(self):
        for norm_topk in (True, False):
            block = qwen3_next.Qwen3NextSparseMoeBlock(
                tiny_args(norm_topk_prob=norm_topk)
            )
            # Below and above the compile-width boundary.
            for shape in ((1, 1, 16), (1, 3, 16), (2, 4, 16), (2, 5, 16), (1, 40, 16)):
                for dtype in (mx.float32, mx.bfloat16):
                    x = mx.random.normal(shape, key=mx.random.key(3)).astype(
                        dtype
                    )
                    expected = block(x)
                    with lever(qwen3_next, "_MOE_GATE_COMPILE"):
                        actual = block(x)
                    _bytes_equal(self, actual, expected)

    def test_compiled_path_restricted_to_stable_narrow_widths(self):
        block = qwen3_next.Qwen3NextSparseMoeBlock(tiny_args())
        calls = []
        original = qwen3_next._select_experts

        def counting(gates, top_k, norm_topk_prob):
            calls.append(gates.shape)
            return original(gates, top_k, norm_topk_prob)

        with lever(qwen3_next, "_MOE_GATE_COMPILE"), lever(
            qwen3_next, "_select_experts", counting
        ):
            block(mx.random.normal((1, 1, 16)))  # decode
            block(mx.random.normal((1, 3, 16)))  # k=2 verify
            block(mx.random.normal((1, 40, 16)))  # prefill chunk: eager
            block(mx.random.normal((2, 5, 16)))  # 10 tokens: eager
        self.assertEqual(calls, [(1, 1, 4), (1, 3, 4)])


def _gdn_args():
    return tiny_args(
        hidden_size=256,
        ple_embed_dim=256,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
    )


class GDNHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = qwen4_exp.GatedDeltaNet(_gdn_args())
        nn.quantize(self, group_size=32, bits=4)
        mx.eval(self.parameters())


class TestGDNFusionSubclass(unittest.TestCase):
    def setUp(self):
        self._previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.clear_cache()
        mx.set_default_device(self._previous_device)

    def test_scan_matches_qwen4_subclass_only_with_flag(self):
        model = GDNHolder()
        self.assertTrue(_can_fuse_gdn_projections(model.layer))
        with lever(
            qwen3_5,
            "_probe_gdn_projection_parity",
            lambda layer: frozenset({mx.bfloat16}),
        ):
            self.assertEqual(
                fuse_gated_delta_net_projections(model, enabled=True), 0
            )
            with lever(qwen3_5, "_GDN_FUSION_SUBCLASS"):
                self.assertEqual(
                    fuse_gated_delta_net_projections(model, enabled=True), 1
                )
        self.assertTrue(hasattr(model.layer, "in_proj_fused"))

    def test_fused_qwen4_layer_is_byte_exact(self):
        stock = qwen4_exp.GatedDeltaNet(_gdn_args())
        fused = qwen4_exp.GatedDeltaNet(_gdn_args())
        fused.update(stock.parameters())
        for layer in (stock, fused):
            nn.quantize(layer, group_size=32, bits=4)
            layer.train()
        fused.update(stock.parameters())
        mx.eval(stock.parameters(), fused.parameters())
        _fuse_gdn_projection_layer(fused, frozenset({mx.bfloat16}))

        stock_cache, fused_cache = ArraysCache(size=2), ArraysCache(size=2)
        for rows, seed in ((16, 41), (1, 42)):
            inputs = (
                mx.random.normal((1, rows, 256), key=mx.random.key(seed)) * 0.3
            ).astype(mx.bfloat16)
            _bytes_equal(
                self,
                fused(inputs, cache=fused_cache),
                stock(inputs, cache=stock_cache),
            )
            _bytes_equal(self, fused_cache[0], stock_cache[0])
            _bytes_equal(self, fused_cache[1], stock_cache[1])


class TestSharedTopkTrimFix(unittest.TestCase):
    """Regression tests for the pre-existing shared-top-k desync.

    A QSA rewind (``QSAKVCache.trim``) used to leave ``_mtp_shared_topk``
    armed; the sidecar flush in ``_mtp_draft_verify_loop`` then ran
    ``mtp_step`` without ``mtp_start_cycle`` and took the skip branch,
    advancing KV without appending raw index keys.  The desynced cache was
    deepcopied into APC sidecars and crashed (or mis-attended) on reuse.
    """

    def test_trim_clears_shared_topk_state(self):
        cache = QSAKVCache()
        keys = mx.zeros((1, 1, 8, 4))
        cache.update_and_fetch(keys, keys)
        cache.update_index_keys(mx.zeros((1, 8, 4)))
        cache._mtp_share_topk = True
        cache._mtp_shared_topk = mx.array([[1, 2]])
        cache.trim(2)
        self.assertFalse(cache._mtp_share_topk)
        self.assertIsNone(cache._mtp_shared_topk)

    def _run_selfmtp(self, share):
        from mlx_lm.hybrid_speculative import HybridStats, _mtp_draft_verify_loop

        mx.random.seed(0)
        args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        cache = model.make_cache()
        mtp_cache = model.make_mtp_cache()

        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.uint32)
        logit_hidden, hidden = model.mtp_backbone(prompt, cache)
        model.mtp_step(hidden[:, :-1], prompt[:, 1:], mtp_cache)
        seed_h = hidden[:, -1:, :]
        cur = int(mx.argmax(model.logits(logit_hidden[:, -1:]), axis=-1).item())
        mx.eval([c.state for c in cache], [c.state for c in mtp_cache])
        for layer_cache in cache:
            layer_cache.start_speculation()

        out = {}
        gen = _mtp_draft_verify_loop(
            model,
            cache,
            cur,
            seed_h,
            1,
            64,
            2,  # k=2 arms sharing (share_qsa_indices and k > 1)
            HybridStats(),
            0.0,
            mtp_cache=mtp_cache,
            share_qsa_indices=share,
            mtp_state_out=out,
        )
        for _ in range(6):
            next(gen)
        gen.close()  # finalization flushes pending pairs into mtp_cache
        return model, cache, mtp_cache, out

    def test_selfmtp_sidecar_flush_keeps_index_keys_synced(self):
        for share in (False, True):
            model, cache, mtp_cache, out = self._run_selfmtp(share)
            for group in (cache, mtp_cache):
                for layer_cache in group:
                    if isinstance(layer_cache, QSAKVCache):
                        self.assertEqual(
                            layer_cache.index_keys.shape[1],
                            layer_cache.offset,
                            f"share={share}",
                        )
                        self.assertFalse(layer_cache._mtp_share_topk)
                        self.assertIsNone(layer_cache._mtp_shared_topk)
            # A resumed draft step over the sidecar state must run cleanly.
            model.mtp_start_cycle(mtp_cache, share_qsa_indices=share)
            logits, _ = model.mtp_step(
                out["state"][1], mx.array([[5]], mx.uint32), mtp_cache
            )
            mx.eval(logits)

    def test_chained_draft_steps_advance_positions_by_one(self):
        # llama.cpp #27781 class: draft tokens placed at stale or pinned
        # positions present only as quiet accept-rate loss. Pin per-step
        # advancement: each chained draft call must start at the previous
        # call's end offset, advance it by exactly 1, and (with the index
        # projection active) rope its indexer query at that same position.
        # Test-only pin (no local red revision): this asserts behavior the
        # path already had; the red arm is the external engine's bug.
        mx.random.seed(0)
        args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        cache = model.make_cache()
        mtp_cache = model.make_mtp_cache()
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.uint32)
        _, hidden = model.mtp_backbone(prompt, cache)
        model.mtp_step(hidden[:, :-1], prompt[:, 1:], mtp_cache)

        for share in (False, True):
            positions = []
            original = qwen4_exp._apply_rope_positions

            def spy(x, pos, dims, base, _record=positions):
                mx.eval(pos)
                flat = pos.reshape(-1).tolist()
                if len(flat) == 1:
                    _record.append(flat[0])
                return original(x, pos, dims, base)

            start = mtp_cache[0].offset
            model.mtp_start_cycle(mtp_cache, share_qsa_indices=share)
            h = hidden[:, -1:, :]
            tok = mx.array([[5]], mx.uint32)
            offsets = []
            qwen4_exp._apply_rope_positions = spy
            try:
                for _ in range(3):
                    offsets.append(mtp_cache[0].offset)
                    _, post = model.mtp_step(h, tok, mtp_cache)
                    h = post[:, -1:, :]
            finally:
                qwen4_exp._apply_rope_positions = original
            self.assertEqual(offsets, [start, start + 1, start + 2], share)
            if share:
                # Sharing runs the index projection EXACTLY once per cycle
                # (step 0); later chained steps reuse its block selection.
                self.assertEqual(positions, [start])
            else:
                self.assertEqual(positions, [start, start + 1, start + 2])
            trim_prompt_cache(mtp_cache, 3)
            self.assertEqual(mtp_cache[0].offset, start)

    def test_pooled_keys_raise_on_index_desync(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        cache = QSAKVCache()
        hidden = mx.random.normal((1, 9, args.hidden_size), key=mx.random.key(0))
        mask = (mx.arange(9)[:, None] >= mx.arange(9)[None, :])[None, None]
        with lever(qwen4_exp, "_QSA_POOLED_KEY_CACHE"):
            mx.eval(indexer(hidden, mask, cache))
            cache.offset += 9
            cache.offset += 1  # simulate a raw-key/KV desync
            with self.assertRaises(RuntimeError):
                indexer(
                    mx.random.normal((1, 1, args.hidden_size)), None, cache
                )


# Lever C (MLX_QWEN4_QSA_FUSED_PROJ, 2026-08-27 decode-decomposition
# addendum) lives here beside the other qwen4_exp levers; its args helper
# comes from the MoE lever test file.
from test_qwen4_moe_levers import moe_args  # noqa: E402

from mlx_lm.models.qwen4_exp import Attention  # noqa: E402


class TestQSAFusedProj(unittest.TestCase):
    def _run_sequence(self, attn, chunks, share_at=None):
        cache = QSAKVCache()
        outputs = []
        for index, chunk in enumerate(chunks):
            length = chunk.shape[1]
            mask = None
            if length > 1 and cache.offset == 0:
                mask = (
                    mx.arange(length)[:, None] >= mx.arange(length)[None, :]
                )[None, None]
            if share_at is not None and index == share_at:
                cache._mtp_share_topk = True
                cache._mtp_shared_topk = None
            out = attn(chunk, mask, cache)
            mx.eval(out, cache.state)
            outputs.append(out)
        return outputs

    def _attention(self, quantized, **overrides):
        attn = Attention(moe_args(**overrides))
        if quantized:
            nn.quantize(attn, group_size=32, bits=4)
        attn.eval()
        mx.eval(attn.parameters())
        return attn

    def test_fused_projections_bitwise_over_prefill_decode_and_share(self):
        for quantized in (False, True):
            for dtype in (mx.float32, mx.bfloat16):
                attn = self._attention(quantized)
                chunks = [
                    mx.random.normal((1, 9, 64), key=mx.random.key(0)).astype(
                        dtype
                    )
                ]
                chunks += [
                    mx.random.normal(
                        (1, 1, 64), key=mx.random.key(step)
                    ).astype(dtype)
                    for step in range(1, 5)
                ]
                expected = self._run_sequence(attn, chunks, share_at=2)
                with lever(qwen4_exp, "_QSA_FUSED_PROJ"):
                    actual = self._run_sequence(attn, chunks, share_at=2)
                self.assertIsNotNone(attn._qsa_fused_cache[1])  # engaged
                for left, right in zip(actual, expected):
                    _bytes_equal(self, left, right)

    def test_fused_table_tracks_weight_replacement(self):
        attn = self._attention(False)
        donor = self._attention(False)
        chunk = mx.random.normal((1, 4, 64), key=mx.random.key(7))
        mask = (mx.arange(4)[:, None] >= mx.arange(4)[None, :])[None, None]
        with lever(qwen4_exp, "_QSA_FUSED_PROJ"):
            self._run_sequence(attn, [chunk])  # build the fused table
            attn.update(donor.parameters())
            actual = attn(chunk, mask, QSAKVCache())
        expected = donor(chunk, mask, QSAKVCache())
        _bytes_equal(self, actual, expected)

    def test_bias_attention_falls_back_to_stock(self):
        attn = self._attention(False, attention_bias=True)
        chunk = mx.random.normal((1, 4, 64), key=mx.random.key(8))
        mask = (mx.arange(4)[:, None] >= mx.arange(4)[None, :])[None, None]
        expected = attn(chunk, mask, QSAKVCache())
        with lever(qwen4_exp, "_QSA_FUSED_PROJ"):
            actual = attn(chunk, mask, QSAKVCache())
        _bytes_equal(self, actual, expected)
        self.assertIsNone(attn._qsa_fused_cache[1])


class TestUnconditionalCaches(unittest.TestCase):
    def test_rope_position_freq_cache_is_byte_identical(self):
        def reference(x, positions, dims, base):
            freqs = mx.exp(-math.log(base) * mx.arange(0, dims, 2) / dims)
            angles = positions[..., None].astype(mx.float32) * freqs
            cos, sin = mx.cos(angles), mx.sin(angles)
            rope, tail = x[..., :dims], x[..., dims:]
            half = dims // 2
            left, right = rope[..., :half], rope[..., half:]
            rotated = mx.concatenate(
                [left * cos - right * sin, right * cos + left * sin], axis=-1
            )
            return mx.concatenate([rotated.astype(x.dtype), tail], axis=-1)

        x = mx.random.normal((1, 5, 2, 8), key=mx.random.key(11)).astype(
            mx.float16
        )
        positions = mx.arange(3, 8)[None, :, None]
        for dims, base in ((4, 10000.0), (8, 10000.0), (4, 500.0)):
            for _ in range(2):  # second call hits the cache
                _bytes_equal(
                    self,
                    _apply_rope_positions(x, positions, dims, base),
                    reference(x, positions, dims, base),
                )

    def test_ngram_numpy_constants_match_device_arrays(self):
        emb = NGramEmbedding(tiny_args(), 16, layer_idx=1, ple_layer_index=0)
        multipliers, sizes, offsets = emb._hash_constants_numpy()
        np.testing.assert_array_equal(
            multipliers, np.asarray(emb.layer_multipliers)
        )
        np.testing.assert_array_equal(
            sizes, np.asarray(emb.ngram_heads_vocab_sizes)
        )
        np.testing.assert_array_equal(
            offsets, np.asarray(emb.ngram_heads_offsets)
        )
        # Same sources: the snapshot is reused, not rebuilt.
        self.assertIs(emb._hash_constants_numpy(), emb._np_constants)

    def test_cpu_hash_tracks_load_weights_replacement(self):
        """load_weights replaces the mx hash constants after construction;
        the CPU path must use the loaded values, matching the Metal path."""
        emb = NGramEmbedding(tiny_args(), 16, layer_idx=1, ple_layer_index=0)
        donor = NGramEmbedding(tiny_args(), 16, layer_idx=1, ple_layer_index=1)
        tokens = mx.array([[1, 2, 63, 3, 4]], dtype=mx.int64)
        stale = emb._ngram_ids_numpy(tokens, None)  # snapshot taken pre-load
        self.assertFalse(
            np.array_equal(
                np.asarray(emb.layer_multipliers),
                np.asarray(donor.layer_multipliers),
            )
        )
        emb.load_weights(
            [("layer_multipliers", donor.layer_multipliers)], strict=False
        )
        expected = donor._ngram_ids_numpy(tokens, None)
        actual_cpu = emb._ngram_ids_numpy(tokens, None)
        np.testing.assert_array_equal(actual_cpu, expected)
        self.assertFalse(np.array_equal(actual_cpu, stale))
        if mx.metal.is_available():
            actual_metal = np.asarray(emb._ngram_ids_metal(tokens, None))
            np.testing.assert_array_equal(actual_metal, actual_cpu)


if __name__ == "__main__":
    unittest.main()
