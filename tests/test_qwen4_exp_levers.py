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
from contextlib import ExitStack, contextmanager

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
    """MLX_QWEN4_QSA_FUSED_PROJ — TOLERANCE-class lever (demoted from
    bitwise, 2026-08-27 review).  The bitwise assertions below hold at THIS
    file's tiny shapes and guard the lever's structure, but they cannot
    establish bitwise at production shapes: mlx's qmm dispatch is width-
    dependent (mlx-src/mlx/backend/metal/quantized.cpp:102 thresholds, :907
    split-K selection) — at M=512 the stock 512-wide K/V projections take
    split-K=2 while the fused 13952-wide op takes non-split qmm, and the
    qmv/qmm crossover differs at M=12-15.  The bench gates this arm with
    the tolerance (digest + chosen-logprob) machinery."""

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


@contextmanager
def count_pooling():
    """Count every entry into block pooling, by cache shape."""
    calls = []
    originals = {
        name: getattr(QSAIndexer, name)
        for name in ("_pooled_keys", "_pool_blocks_left_padded")
    }

    def spy(name, inner):
        def wrapper(indexer, *args, **kwargs):
            calls.append(name)
            return inner(indexer, *args, **kwargs)

        return wrapper

    for name, inner in originals.items():
        setattr(QSAIndexer, name, spy(name, inner))
    try:
        yield calls
    finally:
        for name, inner in originals.items():
            setattr(QSAIndexer, name, inner)


class TestQSALeftPaddedBatchComposition(unittest.TestCase):
    """The other QSA levers over the left-padded batch geometry.

    The 2026-08-27 coordinate fix rebuilt the pooling, the token->block map
    and the tail term for that path, so each lever is re-checked there rather
    than only on the single-sequence path the rest of this file drives.

    ``_QSA_POOLED_KEY_CACHE`` engages here as of 2026-08-27.  Its incremental
    append keys blocks by index on the assumption that a block, once closed,
    is final -- true for one sequence, and true per row for a batch too, since
    the blocks are LOGICAL.  What is NOT true for a batch is that the shared
    block grid is closed: it is the widest row's, so a padded row's
    same-indexed trailing block is still open and was pooled from clamped
    columns.  The cache therefore retains only the blocks EVERY row has
    closed, which the row with the most left padding bounds, and recomputes
    the rest each call.  The tests below pin both halves.

    The schedule ends on a four-token chunk so the PADDED row's own logical
    total crosses the 11-token dense boundary (4, 5, 8, 9, 13) instead of
    leaving only the unpadded row genuinely sparse.  What this file asserts is
    that pooling ran on the padded geometry; per-row mask sparsity itself is
    pinned in ``tests/test_qwen4_exp.py::TestQSALeftPaddedBatch``.
    """

    PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17]]
    CHUNKS = ([20], [21], [22, 23, 24], [25], [26, 27, 28, 29])

    def _merged(self):
        """Prefill each prompt on its own cache and merge.  Single-sequence
        work, so it is deliberately outside the pooling counter below."""
        mx.random.seed(11)
        args = tiny_args(ple_layer_ids=[2])
        model = Model(
            ModelArgs(model_type="qwen4_exp", text_config=args.__dict__)
        ).language_model
        caches = []
        for prompt in self.PROMPTS:
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
            caches.append(cache)
        batch = _merge_caches(caches)
        padding = next(
            layer.left_padding.tolist()
            for layer in batch
            if isinstance(layer, qwen4_exp.BatchQSAKVCache)
        )
        return model, batch, padding

    def _decode(self, model, batch):
        outputs = []
        for chunk in self.CHUNKS:
            logits = model(
                mx.array([list(chunk)] * len(self.PROMPTS), dtype=mx.int32),
                cache=batch,
            )
            mx.eval(logits)
            outputs.append(np.asarray(logits))
        return outputs

    def _run(self, count=False):
        model, batch, padding = self._merged()
        if not count:
            return padding, self._decode(model, batch), None
        with count_pooling() as pooled:
            outputs = self._decode(model, batch)
            return padding, outputs, list(pooled)

    def test_levers_are_bitwise_over_a_left_padded_batch(self):
        padding, stock, pooled = self._run(count=True)
        self.assertEqual(padding, [0, 11])  # 14 vs 3 tokens
        # The comparison is only meaningful where the padded geometry is
        # actually pooled, i.e. where the schedule really does go sparse.
        self.assertEqual(
            set(pooled), {"_pooled_keys", "_pool_blocks_left_padded"}
        )
        # _RMSNORM_FAST is deliberately NOT in this list: it is the one
        # tolerance-class lever in the bundle (mx.fast.rms_norm rounds the
        # normalized value once before the per-stream weight multiply, a
        # documented <=1 fp16 ulp deviation), so asserting bitwise equality
        # for it asserts something the lever never claimed. It is gated by
        # tolerance in TestRMSNormFast and disqualified-on-mismatch in the
        # lever bench, not here.
        for flag in ("_QSA_SCATTER_CHOSEN",):
            with self.subTest(flag=flag):
                with lever(qwen4_exp, flag):
                    _, fast, _ = self._run()
                for step, (expected, actual) in enumerate(zip(stock, fast)):
                    np.testing.assert_array_equal(actual, expected, f"step {step}")

        # ...but it must still be within its tolerance class over the same
        # left-padded geometry, which is the property that actually matters.
        with lever(qwen4_exp, "_RMSNORM_FAST"):
            _, fast, _ = self._run()
        scale = max(float(np.abs(np.asarray(s)).max()) for s in stock)
        for step, (expected, actual) in enumerate(zip(stock, fast)):
            deviation = float(
                np.abs(np.asarray(actual) - np.asarray(expected)).max()
            )
            self.assertLess(
                deviation,
                3e-3 * scale,
                f"_RMSNORM_FAST step {step}: {deviation:.3e} vs scale {scale:.3e}",
            )

    def _qsa_caches(self, batch):
        return [
            layer
            for layer in batch
            if isinstance(layer, qwen4_exp.BatchQSAKVCache)
        ]

    def test_pooled_key_cache_is_bitwise_on_a_batch_cache(self):
        _, stock, _ = self._run()
        model, batch, padding = self._merged()
        with lever(qwen4_exp, "_QSA_POOLED_KEY_CACHE"):
            with count_pooling() as pooled:
                fast = self._decode(model, batch)
        for step, (expected, actual) in enumerate(zip(stock, fast)):
            np.testing.assert_array_equal(actual, expected, f"step {step}")
        self.assertIn("_pooled_keys", pooled)
        # Engagement, not just routing: something was retained.
        caches = self._qsa_caches(batch)
        self.assertTrue(caches)
        for cache in caches:
            self.assertIsNotNone(cache._qsa_pooled_keys)

    def test_batch_pooled_cache_retains_only_blocks_every_row_closed(self):
        # The bound that makes the reuse sound.  A block the padded row has
        # not closed was pooled from CLAMPED columns; retaining it would hand
        # that garbage back as a real value once the row does close it.
        model, batch, padding = self._merged()
        self.assertEqual(padding, [0, 11])
        ratio = tiny_args(ple_layer_ids=[2]).indexer_compress_ratio
        with lever(qwen4_exp, "_QSA_POOLED_KEY_CACHE"):
            for chunk in self.CHUNKS:
                mx.eval(
                    model(
                        mx.array(
                            [list(chunk)] * len(self.PROMPTS), dtype=mx.int32
                        ),
                        cache=batch,
                    )
                )
                for cache in self._qsa_caches(batch):
                    shortest = cache._idx - max(padding)
                    self.assertEqual(
                        cache._qsa_pooled_keys.shape[1],
                        shortest // ratio,
                        f"retained past the shortest row at idx {cache._idx}",
                    )
                    # Non-vacuous: the shared grid really is wider.
                    self.assertGreater(cache._idx // ratio, shortest // ratio)

    def test_batch_qsa_cycle_state_survives_filter_and_clears_on_membership(self):
        model, batch, _ = self._merged()
        with lever(qwen4_exp, "_QSA_POOLED_KEY_CACHE"):
            self._decode(model, batch)
        caches = self._qsa_caches(batch)
        for cache in caches:
            cache._mtp_shared_topk = mx.zeros((2, 3), dtype=mx.uint32)

        for cache in caches:
            cache.filter(mx.array([1]))
            self.assertEqual(cache._qsa_pooled_keys.shape[0], 1)
            self.assertEqual(cache._mtp_shared_topk.shape[0], 1)

        for cache in caches:
            cache.prepare(lengths=[2], right_padding=[0])
            self.assertIsNone(cache._qsa_pooled_keys)
            self.assertIsNone(cache._mtp_shared_topk)


class TestQSADenseShortCircuit(unittest.TestCase):
    """MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT.

    ``tiny_args`` puts the dense boundary at ``indexer_budget +
    compress_ratio - 1`` = 8 + 4 - 1 = 11 cached tokens (block_topk 2, ratio
    4), so these tests straddle 11 instead of the production 2051.  The mask
    a short-circuited call returns is ``causal_mask`` itself, which for a
    single-token decode is ``None`` where the stock path builds an all-true
    array; ``_dense`` normalizes that so the two are compared as masks.
    """

    BOUNDARY = 11  # largest total that is dense by construction

    def _dense(self, mask, length, total):
        if mask is None:
            return np.ones((1, 1, length, total), dtype=bool)
        return np.asarray(mask)

    def _drive(self, indexer, chunks, cache=None):
        """Run ``chunks`` through one QSAKVCache, returning per-step masks."""
        cache = QSAKVCache() if cache is None else cache
        masks, ledger = [], []
        for chunk in chunks:
            length = chunk.shape[1]
            total = cache.offset + length
            if length > 1:
                pos = mx.arange(cache.offset, total)
                mask = (pos[:, None] >= mx.arange(total)[None, :])[None, None]
            else:
                mask = None
            sparse = indexer(chunk, mask, cache)
            mx.eval(sparse)
            masks.append(self._dense(sparse, length, total))
            cache.offset += length
            ledger.append(
                (
                    cache.offset,
                    0 if cache.index_keys is None else cache.index_keys.shape[1],
                )
            )
        return masks, ledger

    def _chunks(self, args, prefill, steps):
        chunks = [
            mx.random.normal((1, prefill, args.hidden_size), key=mx.random.key(0))
        ]
        chunks += [
            mx.random.normal((1, 1, args.hidden_size), key=mx.random.key(step))
            for step in range(1, steps + 1)
        ]
        return chunks

    def test_flag_defaults_off(self):
        self.assertFalse(qwen4_exp._QSA_DENSE_SHORTCIRCUIT)

    def test_stock_mask_is_dense_exactly_below_the_boundary(self):
        # The oracle the lever rests on, measured on the STOCK path: the
        # sparse mask equals the causal mask iff n_blocks <= block_topk.
        args = tiny_args()
        indexer = QSAIndexer(args)
        for total in (4, 7, 8, 10, 11, 12, 16, 20):
            hidden = mx.random.normal(
                (1, total, args.hidden_size), key=mx.random.key(total)
            )
            causal = (
                mx.arange(total)[:, None] >= mx.arange(total)[None, :]
            )[None, None]
            sparse = indexer(hidden, causal, QSAKVCache())
            mx.eval(sparse)
            equal = np.array_equal(np.asarray(sparse), np.asarray(causal))
            self.assertEqual(
                equal, total <= self.BOUNDARY, f"total={total}"
            )

    def test_prefill_masks_bitwise_identical_across_boundary(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        for total in (4, 7, 8, 10, 11, 12, 16, 20):
            chunks = [
                mx.random.normal(
                    (1, total, args.hidden_size), key=mx.random.key(total)
                )
            ]
            stock, _ = self._drive(indexer, chunks)
            with lever(qwen4_exp, "_QSA_DENSE_SHORTCIRCUIT"):
                fast, _ = self._drive(indexer, chunks)
            np.testing.assert_array_equal(fast[0], stock[0], f"total={total}")

    def test_decode_across_boundary_keeps_masks_and_ledger(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        # 5 prompt tokens + 12 decode steps walks total from 6 to 17, so the
        # 11 -> 12 crossing happens mid-decode.
        chunks = self._chunks(args, 5, 12)
        stock, stock_ledger = self._drive(indexer, chunks)
        with lever(qwen4_exp, "_QSA_DENSE_SHORTCIRCUIT"):
            fast, fast_ledger = self._drive(indexer, chunks)
        for step, (expected, actual) in enumerate(zip(stock, fast)):
            np.testing.assert_array_equal(actual, expected, f"step {step}")
        self.assertEqual(fast_ledger, stock_ledger)
        for offset, keys in fast_ledger:
            self.assertEqual(keys, offset)

    def test_short_circuit_skips_pooling_and_query_rope(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        chunks = self._chunks(args, 5, 8)  # totals 5..13, crosses at 12
        pooled, roped = [], []
        original = qwen4_exp._apply_rope_positions
        inner = QSAIndexer._pooled_keys

        def rope_spy(x, pos, dims, base):
            roped.append(x.shape)
            return original(x, pos, dims, base)

        def pool_spy(self, *args, **kwargs):
            pooled.append(True)
            return inner(self, *args, **kwargs)

        qwen4_exp._apply_rope_positions = rope_spy
        QSAIndexer._pooled_keys = pool_spy
        try:
            with lever(qwen4_exp, "_QSA_DENSE_SHORTCIRCUIT"):
                self._drive(indexer, chunks)
        finally:
            qwen4_exp._apply_rope_positions = original
            QSAIndexer._pooled_keys = inner
        # Only the two steps at total 12 and 13 do indexer work; the nine
        # dense steps do the projection and the raw-key append and nothing
        # else (one query rope + one pooled-key rope per working step).
        self.assertEqual(len(pooled), 2)
        self.assertEqual(len(roped), 4)

    def test_composes_with_pooled_key_cache_and_scatter_chosen(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        chunks = self._chunks(args, 5, 12)
        for extra in (
            (),
            ("_QSA_POOLED_KEY_CACHE",),
            ("_QSA_SCATTER_CHOSEN",),
            ("_QSA_POOLED_KEY_CACHE", "_QSA_SCATTER_CHOSEN"),
        ):
            with ExitStack() as stack:
                for flag in extra:
                    stack.enter_context(lever(qwen4_exp, flag))
                stock, _ = self._drive(indexer, chunks)
                stack.enter_context(lever(qwen4_exp, "_QSA_DENSE_SHORTCIRCUIT"))
                fast, ledger = self._drive(indexer, chunks)
            for step, (expected, actual) in enumerate(zip(stock, fast)):
                np.testing.assert_array_equal(actual, expected, f"{extra} {step}")
            for offset, keys in ledger:
                self.assertEqual(keys, offset, str(extra))

    def _drive_shared_cycle(self, indexer, chunks, share_at):
        cache = QSAKVCache()
        masks = []
        for index, chunk in enumerate(chunks):
            length = chunk.shape[1]
            total = cache.offset + length
            mask = None
            if length > 1:
                pos = mx.arange(cache.offset, total)
                mask = (pos[:, None] >= mx.arange(total)[None, :])[None, None]
            if index == share_at:
                cache._mtp_share_topk = True
                cache._mtp_shared_topk = None
            sparse = indexer(chunk, mask, cache)
            mx.eval(sparse)
            masks.append(self._dense(sparse, length, total))
            cache.offset += length
        return masks

    def test_shared_topk_cycle_is_bitwise_identical(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        # Cycle opens at total 13 (past the boundary) so the shared index set
        # is narrower than the block count and must NOT be short-circuited.
        chunks = self._chunks(args, 9, 8)
        for share_at in (0, 1, 2, 4, 6):
            stock = self._drive_shared_cycle(indexer, chunks, share_at)
            with lever(qwen4_exp, "_QSA_DENSE_SHORTCIRCUIT"):
                fast = self._drive_shared_cycle(indexer, chunks, share_at)
            for step, (expected, actual) in enumerate(zip(stock, fast)):
                np.testing.assert_array_equal(
                    actual, expected, f"share_at={share_at} step={step}"
                )

    def test_shared_topk_below_boundary_is_dense_and_short_circuits(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        cache = QSAKVCache()
        hidden = mx.random.normal((1, 8, args.hidden_size), key=mx.random.key(3))
        mask = (mx.arange(8)[:, None] >= mx.arange(8)[None, :])[None, None]
        mx.eval(indexer(hidden, mask, cache))
        cache.offset += 8
        cache._mtp_share_topk = True
        cache._mtp_shared_topk = None
        step = mx.random.normal((1, 1, args.hidden_size), key=mx.random.key(4))
        mx.eval(indexer(step, None, cache))  # total 9, records the shared set
        cache.offset += 1
        self.assertEqual(cache._mtp_shared_topk.shape[-1], 2)
        # total 10 still has 2 blocks, so the shared set covers them all.
        self.assertTrue(indexer._dense_by_construction(2, cache._mtp_shared_topk))
        # total 12 closes a third block the shared set does not name.
        self.assertFalse(indexer._dense_by_construction(3, cache._mtp_shared_topk))

    def test_dense_cycle_start_still_records_the_shared_set(self):
        # A cycle that opens while the mask is dense must hand later steps
        # the full block set, or they silently recompute (mask-equal but a
        # different code path, and a divergence from the stock run).
        args = tiny_args()
        indexer = QSAIndexer(args)
        cache = QSAKVCache()
        hidden = mx.random.normal((1, 8, args.hidden_size), key=mx.random.key(3))
        mask = (mx.arange(8)[:, None] >= mx.arange(8)[None, :])[None, None]
        cache._mtp_share_topk = True
        with lever(qwen4_exp, "_QSA_DENSE_SHORTCIRCUIT"):
            self.assertIs(indexer(hidden, mask, cache), mask)
        cache.offset += 8
        self.assertIsNotNone(cache._mtp_shared_topk)
        np.testing.assert_array_equal(
            np.asarray(cache._mtp_shared_topk), np.array([[0, 1]])
        )
        self.assertEqual(cache.index_keys.shape[1], cache.offset)

    def _model_sequence(self, model):
        cache = model.make_cache()
        outputs = []

        def step(tokens):
            logits = model(mx.array([tokens], dtype=mx.int32), cache=cache)
            mx.eval(logits)
            outputs.append(np.asarray(logits))

        step([1, 2, 3, 4, 5])
        for token in range(6, 24):  # walks the 11 -> 12 boundary mid-decode
            step([token % 60])
        for layer_cache in cache:
            layer_cache.start_speculation()
        step([7, 8, 9])
        trim_prompt_cache(cache, 2)
        for token in range(3):
            step([10 + token])
        for layer_cache in cache:
            if isinstance(layer_cache, QSAKVCache):
                self.assertEqual(
                    layer_cache.index_keys.shape[1], layer_cache.offset
                )
        return outputs

    def test_full_model_logits_bitwise_identical_with_rollback(self):
        mx.random.seed(11)
        args = tiny_args(ple_layer_ids=[2])
        model = Model(
            ModelArgs(model_type="qwen4_exp", text_config=args.__dict__)
        ).language_model
        stock = self._model_sequence(model)
        with lever(qwen4_exp, "_QSA_DENSE_SHORTCIRCUIT"):
            fast = self._model_sequence(model)
        for step, (expected, actual) in enumerate(zip(stock, fast)):
            np.testing.assert_array_equal(actual, expected, f"step {step}")

    def _batch_sequence(self, model, prompts):
        caches = []
        for prompt in prompts:
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
            caches.append(cache)
        batch = _merge_caches(caches)
        outputs = []
        for token in (5, 6, 7):
            logits = model(
                mx.array([[token]] * len(prompts), dtype=mx.int32), cache=batch
            )
            mx.eval(logits)
            outputs.append(np.asarray(logits))
        return batch, outputs

    def test_batch_without_left_padding_is_bitwise_identical(self):
        mx.random.seed(11)
        args = tiny_args(ple_layer_ids=[2])
        model = Model(
            ModelArgs(model_type="qwen4_exp", text_config=args.__dict__)
        ).language_model
        prompts = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]
        batch, stock = self._batch_sequence(model, prompts)
        for layer_cache in batch:
            if isinstance(layer_cache, qwen4_exp.BatchQSAKVCache):
                self.assertEqual(layer_cache.left_padding.max().item(), 0)
        with lever(qwen4_exp, "_QSA_DENSE_SHORTCIRCUIT"):
            _, fast = self._batch_sequence(model, prompts)
        for step, (expected, actual) in enumerate(zip(stock, fast)):
            np.testing.assert_array_equal(actual, expected, f"step {step}")

    def test_left_padded_batch_is_bitwise_identical(self):
        # The indexer's geometry is per-row LOGICAL as of 2026-08-27, so the
        # dense identity holds row for row and the lever covers a left-padded
        # batch too.  Before that fix ``q_pos`` was logical while the block
        # starts and token positions stayed physical, the stock mask itself
        # was wrong for a padded row, and this path was excluded by a
        # ``_qsa_positions_are_physical`` guard.
        mx.random.seed(11)
        args = tiny_args(ple_layer_ids=[2])
        model = Model(
            ModelArgs(model_type="qwen4_exp", text_config=args.__dict__)
        ).language_model
        prompts = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10]]
        batch, stock = self._batch_sequence(model, prompts)
        padded = [
            layer_cache
            for layer_cache in batch
            if isinstance(layer_cache, qwen4_exp.BatchQSAKVCache)
        ]
        self.assertTrue(padded)
        for layer_cache in padded:
            self.assertEqual(layer_cache.left_padding.max().item(), 4)
        with count_pooling() as pooled:
            with lever(qwen4_exp, "_QSA_DENSE_SHORTCIRCUIT"):
                _, fast = self._batch_sequence(model, prompts)
            short_circuited = list(pooled)
            del pooled[:]
            self._batch_sequence(model, prompts)
            stock_pooling = list(pooled)
        for step, (expected, actual) in enumerate(zip(stock, fast)):
            np.testing.assert_array_equal(actual, expected, f"step {step}")
        # Equality would be vacuous if the lever simply declined this path:
        # every step here stays at or below the 11-token dense boundary, so
        # the stock run pools and the lever run must not.
        self.assertTrue(
            any(name == "_pool_blocks_left_padded" for name in stock_pooling)
        )
        self.assertEqual(short_circuited, [])


if __name__ == "__main__":
    unittest.main()
