# Copyright © 2026 Apple Inc.

import unittest
from contextlib import contextmanager
from dataclasses import replace
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

# mlx >= 0.32 runs fp32 GEMMs at TF32 precision on M5 unless this is 0, while
# M=1 gemv shapes stay exact -- so a batched row (GEMM) and the same sequence
# decoded singly (gemv) would be compared at two different precisions. The
# flag latches process-wide on first matmul, so a full-suite run inherits the
# pin from tests/test_models.py but a single-file run of this module did not.
# See wiki lessons/tf32-default-fp32-gemm-m5.md.
environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm import utils
from mlx_lm.generate import (
    _merge_caches,
    _right_pad_prompts,
    generate_step,
    maybe_quantize_kv_cache,
)
from mlx_lm.hybrid_speculative import _GREEDY, HybridStats, self_mtp_generate_step
from mlx_lm.models.cache import (
    CacheList,
    KVCache,
    SinkWindowKVCache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.models import qwen4_exp as qwen4_exp_module
from mlx_lm.models.qwen4_exp import (
    BatchQSAKVCache,
    GatedResidual,
    Model,
    ModelArgs,
    NGramEmbedding,
    PLELayer,
    QSAIndexer,
    QSAKVCache,
    QSASelection,
    Qwen4ArraysCache,
    TextModel,
    TextModelArgs,
    _compact_qsa_block_ids,
)


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


@contextmanager
def lever_flag(name, value=True):
    """Flip one import-time micro-lever for the body of a test."""
    previous = getattr(qwen4_exp_module, name)
    setattr(qwen4_exp_module, name, value)
    try:
        yield
    finally:
        setattr(qwen4_exp_module, name, previous)


@contextmanager
def stock_moe_layout():
    """Pin the stock split layout. MLX_QWEN4_MOE_FUSED_GATE_UP keeps the
    shipped fused tensor instead, and these tests assert the split."""
    from mlx_lm.models import qwen3_next

    previous = qwen3_next._MOE_FUSED_GATE_UP
    qwen3_next._MOE_FUSED_GATE_UP = False
    try:
        yield
    finally:
        qwen3_next._MOE_FUSED_GATE_UP = previous


@contextmanager
def ple_hash_backend(name):
    previous = environ.get("MLX_QWEN4_PLE_HASH_BACKEND")
    environ["MLX_QWEN4_PLE_HASH_BACKEND"] = name
    try:
        yield
    finally:
        if previous is None:
            environ.pop("MLX_QWEN4_PLE_HASH_BACKEND", None)
        else:
            environ["MLX_QWEN4_PLE_HASH_BACKEND"] = previous


class TestQwen4Exp(unittest.TestCase):
    def test_sink_window_cache_preserves_positions_and_rolls_back(self):
        cache = SinkWindowKVCache(window_size=4, sink_size=2, rollback_window=4)

        def update(length):
            values = mx.zeros((1, 2, length, 8))
            cache.update_and_fetch(values, values)

        update(12)
        self.assertEqual(cache._active_positions, [0, 1, 8, 9, 10, 11])
        cache.start_speculation()
        update(3)
        self.assertEqual(cache.trim(3), 3)
        cache.stop_speculation()
        self.assertEqual(cache.offset, 12)
        self.assertEqual(cache._active_positions, [0, 1, 8, 9, 10, 11])

    def test_ngram_ids_preserve_context_and_reset_at_eos(self):
        args = tiny_args()
        emb = NGramEmbedding(args, 16, layer_idx=1, ple_layer_index=0)
        from mlx_lm.models.cache import ArraysCache

        cache = ArraysCache(4)
        first = emb.ngram_ids(mx.array([[1, 2, 63, 3]], dtype=mx.int64), cache)
        second = emb.ngram_ids(mx.array([[4]], dtype=mx.int64), cache)
        joined = emb.ngram_ids(mx.array([[1, 2, 63, 3, 4]], dtype=mx.int64), None)
        mx.eval(first, second, joined)
        self.assertTrue(mx.array_equal(first, joined[:, :4]).item())
        self.assertTrue(mx.array_equal(second, joined[:, 4:]).item())
        self.assertEqual(first.shape[-1], 4)  # 2 heads each for bi- and trigrams

    def test_ngram_hash_backends_match_cpu_with_cache_and_eos(self):
        args = tiny_args()
        inputs = [
            mx.array([[1, 2, 63, 3], [63, 4, 5, 6]], dtype=mx.int64),
            mx.array([[4, 5], [7, 63]], dtype=mx.int64),
        ]
        results = {}
        from mlx_lm.models.cache import ArraysCache

        for backend in ("cpu", "routed_cpu", "metal", "metal_prefill"):
            with ple_hash_backend(backend):
                emb = NGramEmbedding(args, 16, layer_idx=1, ple_layer_index=0)
            cache = ArraysCache(4)
            chunks = [emb.ngram_ids(value, cache) for value in inputs]
            mx.eval(*chunks)
            results[backend] = [np.asarray(value) for value in chunks]

        for backend in ("routed_cpu", "metal", "metal_prefill"):
            for expected, actual in zip(results["cpu"], results[backend]):
                np.testing.assert_array_equal(actual, expected)

    def test_optimized_embedding_backends_match_default(self):
        args = tiny_args()
        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int64)
        with ple_hash_backend("cpu"):
            reference = NGramEmbedding(args, 16, layer_idx=1, ple_layer_index=0)
        expected = reference(tokens)
        for backend in ("routed_cpu", "metal", "metal_prefill"):
            with ple_hash_backend(backend):
                optimized = NGramEmbedding(args, 16, layer_idx=1, ple_layer_index=0)
            optimized.update(reference.parameters())
            if backend == "metal_prefill":
                optimized.metal_hash_min_tokens = 1
            actual = optimized(tokens)
            mx.eval(expected, actual)
            self.assertTrue(mx.array_equal(expected, actual).item())

    def test_hyper_connection_shapes(self):
        args = tiny_args()
        layer = GatedResidual(args)
        x = mx.random.normal((2, 3, args.hc_count * args.hidden_size))
        mixed, residual, inject = layer(x)
        self.assertEqual(mixed.shape, (2, 3, args.hidden_size))
        self.assertEqual(residual.shape, x.shape)
        self.assertEqual(inject.shape, (2, 3, args.hc_count))

    def test_qsa_keeps_tail_and_updates_raw_key_cache(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        cache = QSAKVCache()
        hidden = mx.random.normal((1, 9, args.hidden_size))
        causal = mx.arange(9)[:, None] >= mx.arange(9)[None, :]
        sparse = indexer(hidden, causal[None, None], cache).dense_mask()
        mx.eval(sparse)
        self.assertEqual(sparse.shape, (1, 1, 9, 9))
        self.assertEqual(cache.index_keys.shape, (1, 9, args.indexer_head_dim))
        # The incomplete block tail is always included for its own query.
        self.assertTrue(bool(np.asarray(sparse)[0, 0, 8, 8]))
        self.assertFalse(bool(np.asarray(sparse)[0, 0, 3, 4]))

    def test_qsa_cache_survives_server_batch_merge_and_extract(self):
        args = tiny_args(ple_layer_ids=[2])
        model = TextModel(args)
        batch_cache = _merge_caches([model.make_cache()])

        first = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=batch_cache)
        second = model(mx.array([[4]], dtype=mx.int32), cache=batch_cache)
        mx.eval(first, second)

        qsa = batch_cache[3]
        self.assertEqual(qsa.index_keys.shape, (1, 4, args.indexer_head_dim))
        extracted = [cache.extract(0) for cache in batch_cache]
        self.assertIsInstance(extracted[3], QSAKVCache)
        self.assertEqual(
            extracted[3].index_keys.shape, (1, 4, args.indexer_head_dim)
        )

        remerged = _merge_caches([extracted])
        third = model(mx.array([[5]], dtype=mx.int32), cache=remerged)
        mx.eval(third)
        self.assertEqual(
            remerged[3].index_keys.shape, (1, 5, args.indexer_head_dim)
        )

    def test_qsa_single_token_decode_without_causal_mask(self):
        args = tiny_args(ple_layer_ids=[2])
        model = TextModel(args)
        cache = model.make_cache()

        prefill = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
        decode = model(mx.array([[4]], dtype=mx.int32), cache=cache)
        mx.eval(prefill, decode)

        self.assertEqual(decode.shape, (1, 1, args.vocab_size))
        self.assertEqual(cache[3].offset, 4)
        self.assertEqual(cache[3].index_keys.shape, (1, 4, args.indexer_head_dim))

    def test_release_config_counts_sixteen_ngram_heads(self):
        args = TextModelArgs(
            ple_layer_ids=[2],
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            * 12,
        )
        self.assertEqual((args.ngram_size - 1) * args.heads_per_ngram, 16)
        self.assertEqual(args.ple_embed_dim // 16, 160)

    def test_tiny_hybrid_model_prefill_with_ple_and_qsa(self):
        args = tiny_args(ple_layer_ids=[2])
        model = TextModel(args)
        cache = model.make_cache()
        logits = model(mx.array([[1, 2, 3, 4, 5]]), cache=cache)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 5, args.vocab_size))
        self.assertEqual(cache[3].offset, 5)
        self.assertEqual(cache[1][3].shape, (1, args.ngram_size - 1))

    def test_ple_gdn_and_qsa_roll_back_as_one_exact_span(self):
        args = tiny_args(ple_layer_ids=[2])
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        speculative = model.make_cache()
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        mx.eval(model(prompt, cache=speculative))
        for cache in speculative:
            cache.start_speculation()

        mx.eval(model(mx.array([[4, 5, 6]]), cache=speculative))
        expected_recurrent = {}
        for index, cache in enumerate(speculative[:3]):
            _, rollback, _ = cache._rollbacks[-1]
            expected_recurrent[index] = rollback(1)
        expected_qsa_keys = [
            mx.array(value[..., :4, :])
            for value in speculative[3].keys_and_values()
        ]
        expected_index_keys = mx.array(speculative[3].index_keys[:, :4])
        mx.eval(
            expected_qsa_keys,
            expected_index_keys,
            list(expected_recurrent.values()),
        )
        trim_prompt_cache(speculative, 2)

        for index, expected_states in expected_recurrent.items():
            for got, expected in zip(speculative[index].cache, expected_states):
                self.assertTrue(mx.array_equal(got, expected).item())
        self.assertEqual(len(speculative[1].cache), 4)
        self.assertEqual(speculative[3].offset, 4)
        self.assertTrue(
            mx.array_equal(
                speculative[3].index_keys, expected_index_keys
            ).item()
        )
        for got, expected in zip(
            speculative[3].keys_and_values(), expected_qsa_keys
        ):
            self.assertTrue(mx.array_equal(got, expected).item())

        got = model(mx.array([[7]]), cache=speculative)
        mx.eval(got)
        self.assertEqual(got.shape, (1, 1, args.vocab_size))

    def test_depth_one_mtp_uses_scheme_a_hyper_hidden(self):
        args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        cache = model.make_cache()
        sample, hyper = model.mtp_backbone(
            mx.array([[1, 2, 3]], dtype=mx.int32), cache
        )
        self.assertEqual(sample.shape, (1, 3, args.hidden_size))
        self.assertEqual(
            hyper.shape, (1, 3, args.hc_count * args.hidden_size)
        )

        logits, next_hyper = model.mtp_step(
            hyper[:, :2],
            mx.array([[2, 3]], dtype=mx.int32),
            model.make_mtp_cache(),
        )
        mx.eval(logits, next_hyper)
        self.assertEqual(logits.shape, (1, 2, args.vocab_size))
        self.assertEqual(
            next_hyper.shape, (1, 2, args.hc_count * args.hidden_size)
        )

    def test_qwen4_mtp_supports_bounded_sink_window_attention(self):
        args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        trunk_cache = model.make_cache()
        _, hyper = model.mtp_backbone(
            mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32), trunk_cache
        )
        mtp_cache = model.make_mtp_cache(window_size=3, sink_size=2)
        logits, next_hyper = model.mtp_step(
            hyper[:, :-1], mx.array([[2, 3, 4, 5, 6]]), mtp_cache
        )
        mx.eval(logits, next_hyper)
        self.assertIsInstance(mtp_cache[0], SinkWindowKVCache)
        self.assertEqual(mtp_cache[0].offset, 5)
        self.assertEqual(mtp_cache[0]._active_positions, [0, 1, 2, 3, 4])

        token = mx.argmax(logits[:, -1:, :], axis=-1)
        model.mtp_step(next_hyper[:, -1:], token, mtp_cache)
        mx.eval(mtp_cache[0].state)
        self.assertEqual(mtp_cache[0].offset, 6)
        self.assertEqual(mtp_cache[0]._active_positions, [0, 1, 3, 4, 5])

    def test_qwen4_mtp_can_share_qsa_topk_within_one_draft_cycle(self):
        args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        trunk_cache = model.make_cache()
        _, hyper = model.mtp_backbone(
            mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=mx.int32),
            trunk_cache,
        )
        mtp_cache = model.make_mtp_cache()
        model.mtp_step(
            hyper[:, :-1], mx.array([[2, 3, 4, 5, 6, 7, 8, 9]]), mtp_cache
        )
        mx.eval(mtp_cache[0].state)
        self.assertEqual(mtp_cache[0].offset, 8)
        self.assertEqual(mtp_cache[0].index_keys.shape[1], 8)

        model.mtp_start_cycle(mtp_cache, share_qsa_indices=True)
        logits, post = model.mtp_step(
            hyper[:, -1:], mx.array([[10]], dtype=mx.int32), mtp_cache
        )
        mx.eval(logits, post, mtp_cache[0].state)
        self.assertIsNotNone(mtp_cache[0]._mtp_shared_topk)
        self.assertEqual(mtp_cache[0].offset, 9)
        self.assertEqual(mtp_cache[0].index_keys.shape[1], 9)

        token = mx.argmax(logits[:, -1:, :], axis=-1)
        model.mtp_step(post[:, -1:], token, mtp_cache)
        mx.eval(mtp_cache[0].state)
        self.assertEqual(mtp_cache[0].offset, 10)
        # Step 1 reused step 0's top-k and skipped the QSA index projection.
        self.assertEqual(mtp_cache[0].index_keys.shape[1], 9)

        trim_prompt_cache(mtp_cache, 2)
        self.assertEqual(mtp_cache[0].offset, 8)
        self.assertEqual(mtp_cache[0].index_keys.shape[1], 8)

    def test_q4_quantizes_ple_as_independent_group_32_shards(self):
        args = tiny_args(ple_layer_ids=[2], ple_embed_dim=128)
        config = {"model_type": "qwen4_exp", "text_config": args.__dict__}
        model = Model(ModelArgs.from_dict(config))
        model, config = utils.quantize_model(model, config, 64, 4)

        ngram_embedding = (
            model.language_model.model.layers[1].ple.ple_embedding.ngram_embedding
        )
        for index in range(args.split_ngram_parts):
            shard = getattr(ngram_embedding, f"shard_{index}")
            self.assertIsInstance(shard, nn.QuantizedEmbedding)
            self.assertEqual(shard.group_size, 32)
            self.assertEqual(shard.bits, 4)

        weights = dict(tree_flatten(model.parameters()))
        prefix = (
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
        )
        self.assertNotIn(f"{prefix}.weight", weights)
        for index in range(args.split_ngram_parts):
            self.assertIn(f"{prefix}.shard_{index}.weight", weights)
            self.assertIn(f"{prefix}.shard_{index}.scales", weights)
            self.assertIn(f"{prefix}.shard_{index}.biases", weights)

        self.assertEqual(
            config["quantization"][f"{prefix}.shard_0"]["group_size"], 32
        )

        with TemporaryDirectory() as directory:
            utils.save_model(directory, model)
            utils.save_config(config, Path(directory) / "config.json")
            reloaded, _ = utils.load_model(Path(directory))
            logits = reloaded(mx.array([[1, 2, 3]], dtype=mx.int32))
            mx.eval(logits)
            self.assertEqual(logits.shape, (1, 3, args.vocab_size))

    @staticmethod
    def _norm_gain_weights(center, seed=3):
        # Noisy trained-looking gains: per-family centers spread around
        # ``center``, including one strongly off-center family like the
        # production checkpoint's pre_fc_norm_embedding (mean 0.236 when
        # ones-centered). A legitimacy window would false-refuse that; the
        # comparative check must not.
        rng = np.random.default_rng(seed)
        offsets = {
            "model.layers.3.self_attn.q_norm.weight": 0.6,
            "model.layers.3.self_attn.k_norm.weight": 0.6,
            "model.layers.2.self_attn.q_layernorm.weight": -0.05,
            "model.layers.2.self_attn.k_layernorm.weight": -0.05,
            "model.layers.1.ple.norm_key.weight": -0.1,
            "model.mtp.pre_fc_norm_embedding.weight": -0.76,
        }
        return {
            key: mx.array(
                (center + off + rng.normal(0, 0.05, 16)).astype(np.float32)
            )
            for key, off in offsets.items()
        }

    def test_sanitize_refuses_norm_convention_mismatch(self):
        # mlx-vlm #2041/#2045 class: a wrong zero-vs-ones-centered guess
        # loads cleanly and produces deterministic garbage. The check must
        # not depend on the conv1d layout proxy that makes the guess.
        args = tiny_args()
        model = TextModel(args)
        raw_conv = mx.zeros((8, 1, 3))  # HF raw layout, shape[-1] != 1
        converted_conv = mx.zeros((8, 3, 1))
        ones_centered = self._norm_gain_weights(1.0)
        zero_centered = self._norm_gain_weights(0.0)

        # Ones-centered gains inside a raw-looking checkpoint: the +1
        # offset would shift them to ~2. Must refuse with an actionable
        # message, not load.
        with self.assertRaisesRegex(
            ValueError,
            r"norm convention mismatch(?s:.*)raw \(\+1 offset\)"
            r"(?s:.*)q_norm(?s:.*)MLX_QWEN4_NORM_CONVENTION",
        ):
            model.sanitize(
                {
                    "model.layers.0.linear_attn.conv1d.weight": raw_conv,
                    **ones_centered,
                }
            )

        # Zero-centered gains in a converted-layout checkpoint (the offset
        # would be skipped, gains stay ~0) must also refuse.
        with self.assertRaisesRegex(ValueError, "norm convention mismatch"):
            model.sanitize(
                {
                    "model.layers.0.linear_attn.conv1d.weight": converted_conv,
                    **zero_centered,
                }
            )

        # Both consistent pairings load, including the off-center family.
        raw_ok = model.sanitize(
            {
                "model.layers.0.linear_attn.conv1d.weight": raw_conv,
                **zero_centered,
            }
        )
        converted_ok = model.sanitize(
            {
                "model.layers.0.linear_attn.conv1d.weight": converted_conv,
                **ones_centered,
            }
        )
        for output in (raw_ok, converted_ok):
            gains = output["model.layers.3.self_attn.q_norm.weight"]
            self.assertAlmostEqual(gains.mean().item(), 1.6, places=1)

    def test_norm_convention_override_forces_and_skips_check(self):
        args = tiny_args()
        model = TextModel(args)
        raw_conv = mx.zeros((8, 1, 3))
        ones_centered = self._norm_gain_weights(1.0)

        environ["MLX_QWEN4_NORM_CONVENTION"] = "converted"
        try:
            # The proxy says raw; the override forces converted (no +1)
            # and skips the refusal.
            output = model.sanitize(
                {
                    "model.layers.0.linear_attn.conv1d.weight": raw_conv,
                    **ones_centered,
                }
            )
        finally:
            del environ["MLX_QWEN4_NORM_CONVENTION"]
        gains = output["model.layers.3.self_attn.q_norm.weight"]
        self.assertAlmostEqual(gains.mean().item(), 1.6, places=1)
        # The layout transform still runs under an override.
        self.assertEqual(
            output["model.layers.0.linear_attn.conv1d.weight"].shape,
            (8, 3, 1),
        )

        environ["MLX_QWEN4_NORM_CONVENTION"] = "bogus"
        try:
            with self.assertRaisesRegex(
                ValueError, "MLX_QWEN4_NORM_CONVENTION"
            ):
                model.sanitize(dict(ones_centered))
        finally:
            del environ["MLX_QWEN4_NORM_CONVENTION"]

    def test_raw_moe_weights_are_split_for_switch_glu(self):
        args = tiny_args()
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        gate_up = mx.arange(4 * 16 * 16).reshape(4, 16, 16)
        down = mx.zeros((4, 16, 8))
        with stock_moe_layout():
            output = model.sanitize(
                {
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
                "model.language_model.layers.0.mlp.experts.down_proj": down,
                }
            )
        prefix = "language_model.model.layers.0.mlp.switch_mlp"
        self.assertEqual(output[f"{prefix}.gate_proj.weight"].shape, (4, 8, 16))
        self.assertEqual(output[f"{prefix}.up_proj.weight"].shape, (4, 8, 16))
        self.assertEqual(output[f"{prefix}.down_proj.weight"].shape, down.shape)

    def test_raw_mtp_moe_weights_are_retained_and_split(self):
        args = tiny_args(mtp_num_hidden_layers=1)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        gate_up = mx.arange(4 * 16 * 16).reshape(4, 16, 16)
        down = mx.zeros((4, 16, 8))
        with stock_moe_layout():
            output = model.sanitize(
                {
                    "mtp.layers.0.mlp.experts.gate_up_proj": gate_up,
                "mtp.layers.0.mlp.experts.down_proj": down,
                }
            )
        prefix = "mtp.layers.0.mlp.switch_mlp"
        self.assertEqual(output[f"{prefix}.gate_proj.weight"].shape, (4, 8, 16))
        self.assertEqual(output[f"{prefix}.up_proj.weight"].shape, (4, 8, 16))
        self.assertEqual(output[f"{prefix}.down_proj.weight"].shape, down.shape)


class TestQSALeftPaddedBatch(unittest.TestCase):
    """QSA geometry for a merged batch of unequal-length prompts.

    ``BatchKVCache.offset`` is LOGICAL -- the physical write index minus that
    row's left padding -- while ``index_keys`` and the KV columns are PHYSICAL
    and shared across the batch.  Until 2026-08-27 the indexer mixed the two:
    ``q_pos`` was logical while the block starts and token positions stayed
    physical, so a padded row's blocks sat off its own keys.  Prompts of
    length 7 and 3, merged and decoded one token, produced a causal row of
    ``[F,F,F,F,T,T,T,T]`` and a QSA row of ``[F]*8`` -- a fully masked SDPA
    row for that request.  Latent while serving runs at concurrency 1; live
    the moment batch serving is enabled.

    ``tiny_args`` has compress_ratio 4 and indexer_budget 8, so block_topk is
    2 and the mask is dense by construction up to 11 cached tokens and
    genuinely sparse past it.  The cases below straddle that boundary, and
    each asserts the sparsity is real so the comparisons cannot go vacuous.
    """

    # Mixed widths so a left-padded cache is exercised by multi-token chunks
    # as well as single-token decodes.
    CHUNKS = ([20], [21], [22, 23, 24], [25], [26, 27], [28], [29, 30])

    PROMPTS = (
        [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10]],
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]],
        [[1, 2], [3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13]],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17]],
    )

    def _model(self):
        mx.random.seed(11)
        return TextModel(tiny_args(ple_layer_ids=[2]))

    @contextmanager
    def _capture(self):
        """Record every ``(causal_mask, qsa_mask)`` pair the indexer builds."""
        records = []
        original = QSAIndexer.__call__

        def spy(indexer, hidden, causal_mask, cache, projected_qk=None):
            out = original(
                indexer, hidden, causal_mask, cache, projected_qk=projected_qk
            )
            mask = out.dense_mask()
            records.append(
                (
                    None if causal_mask is None else np.asarray(causal_mask),
                    None if mask is None else np.asarray(mask),
                )
            )
            return out

        QSAIndexer.__call__ = spy
        try:
            yield records
        finally:
            QSAIndexer.__call__ = original

    def _steps(self, model, cache, chunks, records):
        """Run ``chunks`` through ``cache``, returning (masks, logits) each."""
        steps = []
        for chunk in chunks:
            del records[:]
            logits = model(mx.array(chunk, dtype=mx.int32), cache=cache)
            mx.eval(logits)
            steps.append((list(records), np.asarray(logits)))
        return steps

    def _batch(self, model, prompts, records):
        """Prefill each prompt on its own cache, merge, then run CHUNKS."""
        caches = []
        for prompt in prompts:
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
            caches.append(cache)
        batch = _merge_caches(caches)
        padding = [
            layer.left_padding.tolist()
            for layer in batch
            if isinstance(layer, BatchQSAKVCache)
        ]
        self.assertTrue(padding, "the merge produced no batched QSA cache")
        self.assertEqual(len(set(map(tuple, padding))), 1)
        rows = len(prompts)
        steps = self._steps(
            model, batch, [[list(c)] * rows for c in self.CHUNKS], records
        )
        return batch, padding[0], steps

    def _single(self, model, prompt, records):
        """The same chunks on one sequence; the prefill step is dropped."""
        cache = model.make_cache()
        chunks = [[prompt]] + [[list(c)] for c in self.CHUNKS]
        return self._steps(model, cache, chunks, records)[1:]

    def test_merged_prompts_leave_no_fully_masked_row(self):
        # The reported repro, asserted directly: prompts of length 7 and 3
        # merged, then decoded.  Row 1 carries four padding columns.
        model = self._model()
        with self._capture() as records:
            _, padding, steps = self._batch(model, self.PROMPTS[0], records)
        self.assertEqual(padding, [0, 4])
        queries = 0
        for index, (masks, _) in enumerate(steps):
            self.assertTrue(masks, f"step {index} ran no attention layer")
            for layer, (causal, sparse) in enumerate(masks):
                where = f"step {index} layer {layer}"
                # A batch cache always builds a mask, even for one token.
                self.assertIsNotNone(causal, where)
                self.assertIsNotNone(sparse, where)
                self.assertEqual(sparse.shape, causal.shape, where)
                # QSA only ever removes causal cells; it never adds one.
                self.assertFalse(bool((sparse & ~causal).any()), where)
                for row in range(sparse.shape[0]):
                    for query in range(sparse.shape[2]):
                        self.assertEqual(
                            bool(sparse[row, 0, query].any()),
                            bool(causal[row, 0, query].any()),
                            f"{where} row {row} query {query}",
                        )
                        queries += 1
        self.assertGreater(queries, 0)

    def test_merged_prompts_are_dense_below_the_block_boundary(self):
        # 7 cached + up to 4 more stays at or under the 11-token dense
        # boundary, where the QSA mask must equal the causal mask exactly --
        # including its left-padding term.
        model = self._model()
        with self._capture() as records:
            _, padding, steps = self._batch(model, self.PROMPTS[0], records)
        self.assertEqual(padding, [0, 4])
        total = max(len(p) for p in self.PROMPTS[0])
        compared = 0
        for index, (masks, _) in enumerate(steps):
            total += len(self.CHUNKS[index])
            if total > 11:
                break
            for layer, (causal, sparse) in enumerate(masks):
                np.testing.assert_array_equal(
                    sparse, causal, f"step {index} layer {layer}"
                )
                compared += 1
        self.assertGreater(compared, 0)

    def _assert_rows_match(self, padding, batch_steps, singles):
        """Every row's QSA mask equals its own single-sequence mask.

        Returns the causal cells QSA removed PER ROW, so a caller can assert
        the comparison was not made in the dense regime -- where any
        implementation that returns the causal mask would pass -- for the
        padded rows specifically, not just for the batch as a whole.
        """
        dropped = [0] * len(singles)
        layers = len(batch_steps[0][0])
        self.assertGreater(layers, 0, "no attention layer was recorded")
        for row, single_steps in enumerate(singles):
            pad = padding[row]
            # ``zip`` truncates, so a dropped trailing step or layer would
            # silently shrink the comparison instead of failing it.
            self.assertEqual(len(single_steps), len(batch_steps))
            for index, (batch_step, single_step) in enumerate(
                zip(batch_steps, single_steps)
            ):
                self.assertEqual(len(batch_step[0]), layers, f"step {index}")
                self.assertEqual(len(single_step[0]), layers, f"step {index}")
                for layer, (batched, reference) in enumerate(
                    zip(batch_step[0], single_step[0])
                ):
                    causal, sparse = batched
                    where = f"row {row} step {index} layer {layer}"
                    got = sparse[row]
                    want = reference[1]
                    if want is None:
                        # ``create_attention_mask`` returns None for a
                        # single-token decode on an unpadded cache and QSA
                        # passes that through when every cached position is
                        # selected: an all-true row.
                        want = np.ones(
                            got.shape[:-1] + (got.shape[-1] - pad,), dtype=bool
                        )
                    else:
                        want = want[0]
                    self.assertFalse(
                        bool(got[..., :pad].any()),
                        f"{where}: attended its own left padding",
                    )
                    np.testing.assert_array_equal(got[..., pad:], want, where)
                    dropped[row] += int((causal[row] & ~sparse[row]).sum())
        return dropped

    def test_batch_rows_match_their_single_sequence_masks(self):
        for prompts in self.PROMPTS:
            with self.subTest(lengths=[len(p) for p in prompts]):
                model = self._model()
                with self._capture() as records:
                    _, padding, batch_steps = self._batch(model, prompts, records)
                    singles = [self._single(model, p, records) for p in prompts]
                dropped = self._assert_rows_match(padding, batch_steps, singles)
                # Equality is only meaningful where QSA actually goes sparse,
                # and the padded rows are the ones the fix is about, so every
                # row's own logical total must cross the dense boundary.
                self.assertTrue(all(dropped), f"dense-only rows: {dropped}")

    def test_right_padded_continuation_makes_min_left_padding_positive(self):
        """``min(left_padding) == 0`` is NOT an invariant.

        ``merge()`` and ``filter()`` do leave a zero minimum, but the row
        carrying the zero left padding and the row carrying the zero right
        padding need not be the same one, so ``finalize()`` can lift the
        minimum: histories of 10 and 5 tokens merge to ``left_padding``
        ``[0, 5]``, and a right-padded continuation of 1 and 5 tokens
        finalizes to ``[4, 5]``.  The shared block grid is then an upper
        bound, not a per-row count -- surplus blocks must stay causally
        invalid for every row.  (Found by adversarial review, 2026-08-27.)

        This case uses an attention-only model on purpose.  A right-padded
        lane also advances the GDN and PLE recurrent state by its filler
        tokens, which the batch generator undoes with its own state
        checkpoints rather than with the cache roll; including those layers
        here would test that machinery, not QSA geometry.
        """
        mx.random.seed(11)
        model = TextModel(
            tiny_args(ple_layer_ids=[], layer_types=["full_attention"] * 4)
        )
        histories = [list(range(1, 11)), list(range(20, 25))]
        continuations = [[40], [41, 42, 43, 44, 45]]
        decode = [[[t]] * len(histories) for t in (50, 51, 52)]
        with self._capture() as records:
            caches = []
            for prompt in histories:
                cache = model.make_cache()
                mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
                caches.append(cache)
            batch = _merge_caches(caches)
            lengths = [len(c) for c in continuations]
            width = max(lengths)
            for layer in batch:
                layer.prepare(
                    lengths=lengths,
                    right_padding=[width - length for length in lengths],
                )
            mx.eval(
                model(
                    _right_pad_prompts(continuations, max_length=width),
                    cache=batch,
                )
            )
            for layer in batch:
                layer.finalize()
            padding = next(
                layer.left_padding.tolist()
                for layer in batch
                if isinstance(layer, BatchQSAKVCache)
            )
            self.assertEqual(padding, [4, 5])
            self.assertGreater(min(padding), 0)
            batch_steps = self._steps(model, batch, decode, records)
            singles = [
                self._steps(
                    model,
                    model.make_cache(),
                    [[history + continuation]]
                    + [[[t]] for t in (50, 51, 52)],
                    records,
                )[1:]
                for history, continuation in zip(histories, continuations)
            ]
        dropped = self._assert_rows_match(padding, batch_steps, singles)
        self.assertTrue(all(dropped), f"dense-only rows: {dropped}")
        self._assert_logits_match(batch_steps, singles)

    # Fraction of output scale a pure kernel-shape change is allowed to move
    # the logits.  NOT an invented number: qwen3_next.py documents a measured
    # <= 1.5e-3 of output scale on M5 for a kernel-family change alone
    # (MLX_QWEN4_MOE_SHARED_IN_GATHER), and
    # tests/test_qwen4_moe_levers.py::test_outputs_match_within_tolerance
    # already gates that class at 2x it.  Same class, same band.
    SHAPE_NOISE_BAND = 3e-3
    TOP_K = 5

    def _assert_logits_match(self, batch_steps, singles):
        """The batched row must decode like the single-sequence run.

        This is a CROSS-SHAPE comparison -- a batched, left-padded row against
        the same sequence at B=1 -- so it is not gated on equality:

        * ``wiki/docs/plans/qwen38-mtp-batch-composition.md`` section 3: "B=1
          inside the batch engine is not required to be byte-identical to the
          single-stream path ... The gate for that comparison is
          divergence-classification, not equality."
        * ``results/qwen38-neartie-shape-probe-v2-20260826.json``: row count
          alone shifts a top-token logprob (-1.125 at 1-2 rows, -1.000 at 5-9),
          bucketed by kernel dispatch tier and repeatable.

        So the gate is (a) magnitude inside ``SHAPE_NOISE_BAND`` of output
        scale, measured the way test_qwen4_moe_levers.py measures it -- a
        per-element relative metric explodes on near-zero logits -- and (b) the
        claim that actually matters, that the batched row picks the same
        tokens.  Positions whose reference top-1/top-2 margin is inside the
        band are exempt from (b): an ordering that loose cannot be asserted,
        and a near-tie flip is the probe-v2 class the plan doc permits.

        The exemptions are counted so the gate cannot pass by exempting
        everything.  Measured coverage over the four PROMPTS cases: 99/99
        positions argmax-checked, 91/99 top-5-set-checked, 0 exempt, observed
        magnitude ~1e-7 of scale.  The band is 2 orders of magnitude below the
        bug it guards: run against the pre-fix indexer (8db42d5) the padded
        row lands at 6.6e-2 to 1.6e-1 of output scale.
        """
        compared = ranked = exempt = 0
        for row, single_steps in enumerate(singles):
            self.assertEqual(len(single_steps), len(batch_steps))
            for index, (batch_step, single_step) in enumerate(
                zip(batch_steps, single_steps)
            ):
                got, want = batch_step[1][row], single_step[1][0]
                where = f"row {row} step {index}"
                # A NaN would sail through both the band and the argmax check,
                # and an all-masked SDPA row -- the bug this class guards --
                # produces exactly that.
                self.assertTrue(np.isfinite(got).all(), f"{where}: batch")
                self.assertTrue(np.isfinite(want).all(), f"{where}: single")
                self.assertEqual(got.shape, want.shape, where)
                scale = float(np.abs(want).max())
                self.assertGreater(scale, 0.0, f"{where}: logits are all zero")
                error = float(np.abs(got - want).max()) / scale
                self.assertLess(
                    error,
                    self.SHAPE_NOISE_BAND,
                    f"{where}: {error:.3e} of output scale",
                )
                margin = self.SHAPE_NOISE_BAND * scale
                for query in range(got.shape[0]):
                    compared += 1
                    order = np.argsort(want[query])[::-1]
                    ordered = want[query][order]
                    if ordered[0] - ordered[1] <= margin:
                        exempt += 1
                        continue
                    self.assertEqual(
                        int(got[query].argmax()),
                        int(order[0]),
                        f"{where} query {query}: argmax differs on a "
                        f"non-tie (margin {ordered[0] - ordered[1]:.3e})",
                    )
                    k = self.TOP_K
                    if ordered[k - 1] - ordered[k] <= margin:
                        continue
                    ranked += 1
                    self.assertEqual(
                        set(np.argsort(got[query])[::-1][:k].tolist()),
                        set(order[:k].tolist()),
                        f"{where} query {query}: top-{k} set differs",
                    )
        self.assertGreater(compared, 0, "no logits were compared")
        self.assertGreater(ranked, 0, "no top-k set was comparable")
        self.assertLess(exempt, compared // 2, "most positions were near ties")

    def test_batch_row_logits_match_single_sequence_decode(self):
        for prompts in self.PROMPTS:
            with self.subTest(lengths=[len(p) for p in prompts]):
                model = self._model()
                with self._capture() as records:
                    _, _, batch_steps = self._batch(model, prompts, records)
                    singles = [self._single(model, p, records) for p in prompts]
                self._assert_logits_match(batch_steps, singles)

    def test_batch_index_ledger_desync_is_refused(self):
        # The block columns address ``index_keys`` by PHYSICAL position, so a
        # ledger out of step with the write index would silently pool a
        # shifted history.  Fail loudly instead.
        model = self._model()
        caches = []
        for prompt in ([1, 2, 3, 4, 5, 6, 7], [8, 9, 10]):
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
            caches.append(cache)
        batch = _merge_caches(caches)
        for layer in batch:
            if isinstance(layer, BatchQSAKVCache):
                layer.index_keys = layer.index_keys[:, :-1]
        with self.assertRaisesRegex(RuntimeError, "index_keys desync"):
            mx.eval(model(mx.array([[9], [9]], dtype=mx.int32), cache=batch))


class TestPLERollbackPerRowReplay(unittest.TestCase):
    """The PLE half of the combined record offers a vectorized ragged replay.

    ``ArraysCache.trim_ragged`` takes ``per_row_fn(lengths)`` when a layer
    stages one and otherwise replays the scalar ``fn`` once per DISTINCT row
    length (the plan's interim form a0).  Both PLE states are the window
    ending at a row's own length, so the vectorized form is one gather.
    """

    def _staged(self, batch=3, length=4):
        args = tiny_args(ple_layer_ids=[2])
        mx.random.seed(5)
        layer = PLELayer(args, 1, 0)
        cache = Qwen4ArraysCache(4)
        cache.start_speculation()
        captured = {}
        original = Qwen4ArraysCache.stage_ple_rollback

        def spy(self, num_tokens, fn, snapshot, *, per_row_fn=None):
            captured.update(fn=fn, per_row_fn=per_row_fn, num_tokens=num_tokens)
            return original(self, num_tokens, fn, snapshot, per_row_fn=per_row_fn)

        Qwen4ArraysCache.stage_ple_rollback = spy
        try:
            hidden = mx.random.normal(
                (batch, length, args.hidden_size * args.hc_count)
            )
            ids = mx.array(
                [[1 + row * 10 + i for i in range(length)] for row in range(batch)],
                dtype=mx.int32,
            )
            mx.eval(layer(hidden, ids, cache))
        finally:
            Qwen4ArraysCache.stage_ple_rollback = original
        return captured

    def test_per_row_replay_equals_the_scalar_replay_row_by_row(self):
        captured = self._staged()
        self.assertIsNotNone(captured["per_row_fn"])
        lengths = [4, 2, 0]
        rows = captured["per_row_fn"](lengths)
        mx.eval(rows)
        for row, m in enumerate(lengths):
            scalar = captured["fn"](m)
            mx.eval(scalar)
            for slot, (got, want) in enumerate(zip(rows, scalar)):
                self.assertTrue(
                    mx.array_equal(
                        got[row : row + 1], want[row : row + 1]
                    ).item(),
                    f"row {row} slot {slot} at m={m}",
                )

    def test_combined_record_carries_a_per_row_form_only_when_both_halves_do(self):
        for gdn_rows in (None, lambda lengths: [mx.array(list(lengths))]):
            with self.subTest(gdn_per_row=gdn_rows is not None):
                cache = Qwen4ArraysCache(4)
                cache.start_speculation()
                cache.stage_ple_rollback(
                    2,
                    lambda m: [mx.array([m])],
                    [None],
                    per_row_fn=lambda lengths: [mx.array(list(lengths))],
                )
                cache.record_rollback(
                    2, lambda m: [mx.array([m])], [None], per_row_fn=gdn_rows
                )
                record = cache._rollbacks[-1]
                self.assertEqual(
                    record.per_row_fn is not None, gdn_rows is not None
                )


class TestQSACacheFromState(unittest.TestCase):
    """The prompt-cache load path builds through ``__new__``, not ``__init__``.

    ``_BaseCache.from_state`` does ``cls.__new__(cls)`` and then assigns
    ``state`` and ``meta_state``, so every attribute a method reachable from
    there reads has to be set in ``__new__``.  Both QSA caches route their
    ``state`` setter through ``release_qsa_cycle``, which reads all four cycle
    fields, so an ``__init__``-only default raised ``AttributeError`` on any
    saved cache.  The ledger check has to wait for ``meta_state``: until then
    ``_idx`` is the allocated buffer width, not the cursor.
    """

    def _batch(self, rows=2, width=4):
        cache = BatchQSAKVCache([0] * rows)
        values = mx.broadcast_to(
            mx.arange(width, dtype=mx.float32).reshape(1, 1, width, 1),
            (rows, 1, width, 2),
        )
        cache.update_and_fetch(values, values)
        cache.update_index_keys(
            mx.broadcast_to(
                mx.arange(width, dtype=mx.float32).reshape(1, width, 1),
                (rows, width, 2),
            )
        )
        return cache

    def test_new_sets_every_attribute_the_load_path_reads(self):
        for cls in (BatchQSAKVCache, QSAKVCache):
            with self.subTest(cache=cls.__name__):
                bare = cls.__new__(cls)
                for name, blank in cls._QSA_CYCLE_FIELDS:
                    self.assertEqual(getattr(bare, name), blank, name)
                self.assertIsNone(bare.index_keys)
        self.assertIsNone(Qwen4ArraysCache.__new__(Qwen4ArraysCache)._ple_rollback)

    def test_batch_cache_round_trips_and_still_takes_the_cycle_hooks(self):
        cache = self._batch()
        restored = BatchQSAKVCache.from_state(cache.state, cache.meta_state)
        self.assertEqual(restored._idx, cache._idx)
        self.assertEqual(restored.index_keys.shape[1], restored._idx)
        # The hooks are the part that broke: exercise them on the restored
        # object, not just the construction.
        restored.release_qsa_cycle("test")
        restored.trim(1)
        restored.filter(mx.array([0]))
        self.assertEqual(restored.index_keys.shape[1], restored._idx)
        self.assertIsNone(restored._mtp_shared_topk)

    def test_single_cache_round_trips_and_still_takes_the_cycle_hooks(self):
        cache = QSAKVCache()
        values = mx.zeros((1, 1, 4, 2))
        cache.update_and_fetch(values, values)
        cache.update_index_keys(mx.zeros((1, 4, 2)))
        restored = QSAKVCache.from_state(cache.state, cache.meta_state)
        self.assertEqual(restored.offset, 4)
        restored.release_qsa_cycle("test")
        restored.trim(1)
        self.assertEqual(restored.index_keys.shape[1], restored.offset)

    def test_restoring_a_short_ledger_is_reported(self):
        cache = self._batch()
        state = list(cache.state)
        state[4] = state[4][:, :-1]
        with self.assertRaisesRegex(RuntimeError, "un-ledgered KV"):
            BatchQSAKVCache.from_state(tuple(state), cache.meta_state)


class TestPaddedSpeculativeStaging(unittest.TestCase):
    """PLE rollback staging under the uniform-width verify geometry.

    The old predicate required ``mask is None and lengths is None and
    left_padding is None``, so it disarmed on exactly the forward that has to
    roll back: a right-padded speculative slab where rows propose different
    numbers of draft tokens.  PLE advanced every row and staged nothing.

    Staging is now gated on the geometry being DESCRIBABLE per row instead.
    The record has to carry each row's OWN span as its depth: crediting every
    row the slab width lets a later rewind take tokens out of this record that
    a short row never processed, and stop before the older record that holds
    them.
    """

    PROMPTS = ([1, 2, 3, 4, 5, 6, 7], [8, 9, 10])
    PROPOSALS = ([20, 21, 22], [30])  # Codex's [3, 1] in a width-3 slab
    PLE_LAYER = 1
    CONV, HISTORY = 2, 3

    def _model(self):
        mx.random.seed(11)
        return TextModel(tiny_args(ple_layer_ids=[2]))

    def _merged(self, model):
        caches = []
        for prompt in self.PROMPTS:
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
            caches.append(cache)
        return _merge_caches(caches)

    def _speculative_slab(self, model, batch, proposals=None):
        """Run one right-padded speculative slab, as the verify forward does."""
        proposals = proposals or self.PROPOSALS
        lengths = [len(p) for p in proposals]
        width = max(lengths)
        for layer in batch:
            layer.start_speculation()
            layer.prepare(
                lengths=lengths,
                right_padding=[width - length for length in lengths],
            )
        mx.eval(
            model(_right_pad_prompts(proposals, max_length=width), cache=batch)
        )
        return lengths

    def test_padded_slab_records_a_per_row_depth(self):
        model = self._model()
        batch = self._merged(model)
        lengths = self._speculative_slab(model, batch)
        cache = batch[self.PLE_LAYER]
        self.assertIsInstance(cache, Qwen4ArraysCache)
        # GDN now records under this geometry too, so the two halves pair up
        # in one record instead of leaving a staged half behind.
        self.assertIsNone(cache._ple_rollback)
        record = cache._rollbacks[-1]
        self.assertEqual(record.num_tokens, max(lengths))
        self.assertEqual(record.depths, lengths)
        self.assertEqual(cache._row_capacity(len(lengths)), lengths)

    def test_padded_record_rewinds_each_row_to_its_own_prefix(self):
        """The whole point: reject different draft counts and land right.

        GDN records under this geometry now, so the record under test is the
        real paired PLE+GDN one; the PLE slots are asserted against a
        single-row reference decode.
        """
        for accepted in ((2, 0), (0, 1), (3, 1), (1, 0)):
            with self.subTest(accepted=accepted):
                model = self._model()
                batch = self._merged(model)
                lengths = self._speculative_slab(model, batch)
                cache = batch[self.PLE_LAYER]
                self.assertEqual(cache._row_capacity(len(lengths)), list(lengths))

                for layer in batch:
                    layer.finalize()
                drops = [l - a for l, a in zip(lengths, accepted)]
                cache.trim_ragged(drops)

                for row, take in enumerate(accepted):
                    reference = model.make_cache()
                    mx.eval(
                        model(
                            mx.array([self.PROMPTS[row]], dtype=mx.int32),
                            cache=reference,
                        )
                    )
                    if take:
                        mx.eval(
                            model(
                                mx.array(
                                    [self.PROPOSALS[row][:take]], dtype=mx.int32
                                ),
                                cache=reference,
                            )
                        )
                    want = reference[self.PLE_LAYER]
                    np.testing.assert_array_equal(
                        np.asarray(cache[self.HISTORY][row : row + 1]),
                        np.asarray(want[self.HISTORY]),
                        f"row {row} token history",
                    )
                    # Slots 0 and 1 are GDN's: its replay is mask-free, so
                    # this is where crediting a row more than its own span
                    # would show up.
                    for slot, name in (
                        (0, "gdn conv"),
                        (1, "gdn recurrent state"),
                        (self.CONV, "ple conv"),
                    ):
                        got = np.asarray(cache[slot][row : row + 1])
                        ref = np.asarray(want[slot])
                        scale = float(np.abs(ref).max()) or 1.0
                        self.assertLess(
                            float(np.abs(got - ref).max()) / scale,
                            3e-3,
                            f"row {row} {name}",
                        )

    def test_a_pending_ple_half_makes_the_span_untrimmable(self):
        # A staged-but-unrecorded half is invisible to the rewind walker, so
        # with older records on the stack a trim would take this forward's
        # tokens out of THOSE. Refuse instead.
        model = self._model()
        batch = self._merged(model)
        self._speculative_slab(model, batch)
        cache = batch[self.PLE_LAYER]
        # GDN records under this geometry now, so a pending half needs an
        # interrupted forward: stage one without ever reaching record.
        gdn = [cache[0], cache[1]]
        cache.stage_ple_rollback(1, lambda m, s=gdn: list(s), list(gdn))
        self.assertIsNotNone(cache._ple_rollback)
        self.assertFalse(cache.is_trimmable())
        for call in (
            lambda: cache.trim(1),
            lambda: cache.trim_ragged([1, 0]),
            lambda: cache.preflight_ragged_trim([1, 0]),
        ):
            with self.assertRaisesRegex(RuntimeError, "qwen3_5.py"):
                call()

    def test_leading_pads_disarm_staging_rather_than_lie(self):
        """PLE must consult ``rollback_spans`` and skip when it refuses.

        The refusal rule itself is ``ArraysCache``'s and is covered in
        tests/test_ragged_trim.py; what is asserted here is that this layer
        obeys it, because a leading pad run would otherwise be staged with a
        replay closure that indexes the row from slab position 0.
        """
        args = tiny_args(ple_layer_ids=[2])
        mx.random.seed(5)
        layer = PLELayer(args, 1, 0)
        hidden = mx.random.normal((2, 4, args.hidden_size * args.hc_count))
        ids = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32)

        refuses = Qwen4ArraysCache(4)
        refuses.start_speculation()
        refuses.left_padding = mx.array([0, 2])
        self.assertIsNone(refuses.rollback_spans(4, refuses.make_mask(4)))
        mx.eval(layer(hidden, ids, refuses, refuses.make_mask(4)))
        self.assertIsNone(
            refuses._ple_rollback, "staged a span it cannot replay"
        )

        stages = Qwen4ArraysCache(4)
        stages.start_speculation()
        stages.prepare(lengths=[4, 2])
        mx.eval(layer(hidden, ids, stages, stages.make_mask(4)))
        self.assertIsNotNone(stages._ple_rollback)

    def test_membership_change_drops_a_staged_ple_half(self):
        model = self._model()
        batch = self._merged(model)
        self._speculative_slab(model, batch)
        cache = batch[self.PLE_LAYER]
        gdn = [cache[0], cache[1]]
        cache.stage_ple_rollback(1, lambda m, s=gdn: list(s), list(gdn))
        self.assertIsNotNone(cache._ple_rollback)
        for layer in batch:
            layer.finalize()
        cache.filter([0])
        self.assertIsNone(
            cache._ple_rollback,
            "a stale closure over old-batch tensors survived the filter",
        )
        self.assertTrue(cache.is_trimmable())


class TestMTPCycleAbort(unittest.TestCase):
    """Arming shared top-k needs a paired exit, including on the abort path."""

    def _armed(self):
        mx.random.seed(11)
        args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        trunk = model.make_cache()
        prompt = list(range(1, 11))
        _, hyper = model.mtp_backbone(mx.array([prompt], dtype=mx.int32), trunk)
        head = model.make_mtp_cache()
        logits, post = model.mtp_step(
            hyper[:, :-1], mx.array([prompt[1:]], dtype=mx.int32), head
        )
        model.mtp_start_cycle(head, share_qsa_indices=True)
        for _ in range(2):
            token = mx.argmax(logits[:, -1:, :], axis=-1)
            logits, post = model.mtp_step(post[:, -1:], token, head)
            mx.eval(logits, post, head[0].state)
        return model, head

    def test_ending_a_cycle_without_rewinding_reports_the_ledger(self):
        model, head = self._armed()
        cache = head[0]
        self.assertLess(cache.index_keys.shape[1], cache.offset)
        with self.assertRaisesRegex(RuntimeError, "un-ledgered KV"):
            model.mtp_end_cycle(head)

    def test_ending_a_cycle_after_the_rewind_is_clean(self):
        model, head = self._armed()
        cache = head[0]
        trim_prompt_cache(head, 2)
        model.mtp_end_cycle(head)
        self.assertIsNone(cache._mtp_shared_topk)
        self.assertFalse(cache._mtp_share_topk)
        self.assertEqual(cache.index_keys.shape[1], cache.offset)

    def _armed_batch(self):
        mx.random.seed(11)
        args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        heads, hiddens, tokens = [], [], []
        for prompt in (list(range(1, 11)), [11, 12, 13, 14, 15]):
            trunk = model.make_cache()
            _, hyper = model.mtp_backbone(
                mx.array([prompt], dtype=mx.int32), trunk
            )
            head = model.make_mtp_cache()
            logits, post = model.mtp_step(
                hyper[:, :-1], mx.array([prompt[1:]], dtype=mx.int32), head
            )
            mx.eval(logits, post, head[0].state)
            heads.append(head)
            hiddens.append(post[:, -1:])
            tokens.append(mx.argmax(logits[:, -1:, :], axis=-1))
        batch = _merge_caches(heads)
        model.mtp_start_cycle(batch, share_qsa_indices=True)
        hidden = mx.concatenate(hiddens)
        token = mx.concatenate(tokens)
        for _ in range(2):
            logits, hidden = model.mtp_step(hidden, token, batch)
            mx.eval(logits, hidden, batch[0].state)
            token = mx.argmax(logits[:, -1:, :], axis=-1)
            hidden = hidden[:, -1:]
        return model, batch

    def test_ending_a_batched_cycle_without_rewinding_reports_the_ledger(self):
        model, batch = self._armed_batch()
        cache = batch[0]
        self.assertIsInstance(cache, BatchQSAKVCache)
        self.assertLess(cache.index_keys.shape[1], cache._idx)
        with self.assertRaisesRegex(RuntimeError, "un-ledgered KV"):
            model.mtp_end_cycle(batch)

    def test_ending_a_batched_cycle_after_the_rewind_is_clean(self):
        model, batch = self._armed_batch()
        cache = batch[0]
        trim_prompt_cache(batch, 2)
        model.mtp_end_cycle(batch)
        self.assertIsNone(cache._mtp_shared_topk)
        self.assertEqual(cache.index_keys.shape[1], cache._idx)

    def test_starting_a_cycle_ends_the_previous_one(self):
        # Arming is the only hook the loop is guaranteed to reach, so it must
        # be idempotent: an abandoned cycle cannot leak its index set forward.
        model, head = self._armed()
        cache = head[0]
        trim_prompt_cache(head, 2)
        stale = cache._mtp_shared_topk
        cache._mtp_shared_topk = stale if stale is not None else mx.zeros(
            (1, 2), dtype=mx.uint32
        )
        model.mtp_start_cycle(head, share_qsa_indices=False)
        self.assertIsNone(cache._mtp_shared_topk)
        self.assertFalse(cache._mtp_share_topk)


class TestSelfMTPLeavesNoDraftResidue(unittest.TestCase):
    """A full self-MTP generation must leave no trace of rejected drafts.

    The per-trim rewind is covered elsewhere. This covers the accumulated
    state after many cycles: two runs commit the same tokens but draft
    different (always rejected) tails, so every trunk cache must end
    bit-identical. Both runs use the same forward shapes, so an inequality
    is residue, not reduction order.
    """

    PROMPT = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.uint32)
    MAX_TOKENS = 16

    def _model(self):
        mx.random.seed(23)
        args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        return Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))

    def _never_drafted(self, count):
        """Token ids the trunk never picks, so every forced draft is rejected."""
        model = self._model()
        picked = {
            int(token)
            for token, _ in generate_step(
                self.PROMPT, model, max_tokens=self.MAX_TOKENS, sampler=_GREEDY
            )
        }
        free = [t for t in range(model.args.text_config["vocab_size"]) if t not in picked]
        self.assertGreaterEqual(len(free), count)
        return free[-count:]

    def _run(self, draft_token):
        model = self._model()

        real_step = model.mtp_step

        def forced_step(hidden, tokens, mtp_cache):
            logits, post = real_step(hidden, tokens, mtp_cache)
            forced = mx.full(logits.shape, -30.0)
            forced[..., draft_token] = 30.0
            return forced, post

        model.mtp_step = forced_step
        cache = make_prompt_cache(model)
        stats = HybridStats()
        tokens = [
            int(token)
            for token, _, _ in self_mtp_generate_step(
                self.PROMPT,
                model,
                num_draft=2,
                max_tokens=self.MAX_TOKENS,
                persistent_mtp=True,
                prompt_cache=cache,
                stats=stats,
            )
        ]
        for entry in cache:
            entry.stop_speculation()
        mx.eval([entry.state for entry in cache])
        return tokens, cache, stats

    def test_rejected_tails_leave_the_trunk_cache_identical(self):
        first, second = self._never_drafted(2)
        tokens_a, cache_a, stats_a = self._run(draft_token=first)
        tokens_b, cache_b, stats_b = self._run(draft_token=second)

        # The forced drafts must all be rejected, or the two runs commit
        # different tokens and there is nothing to compare.
        self.assertEqual(stats_a.draft_accepted, 0)
        self.assertEqual(stats_b.draft_accepted, 0)
        self.assertGreater(stats_a.draft_proposed, 0)
        self.assertEqual(
            tokens_a,
            tokens_b,
            "the drafted tail changed the committed tokens, so a rejected "
            "span stayed in the trunk cache",
        )

        for index, (entry_a, entry_b) in enumerate(zip(cache_a, cache_b)):
            if isinstance(entry_a, Qwen4ArraysCache):
                self.assertIsNone(entry_a._ple_rollback)
            if hasattr(entry_a, "cache"):
                for slot, (got, want) in enumerate(
                    zip(entry_a.cache, entry_b.cache)
                ):
                    self.assertTrue(
                        mx.array_equal(got, want).item(),
                        f"layer {index} slot {slot} kept rejected-draft state",
                    )
            else:
                self.assertEqual(entry_a.offset, entry_b.offset)
                width = entry_a.offset
                for name in ("keys", "values", "index_keys"):
                    got = getattr(entry_a, name)
                    want = getattr(entry_b, name)
                    if got is None:
                        self.assertIsNone(want)
                        continue
                    if name == "index_keys":
                        got, want = got[:, :width], want[:, :width]
                    else:
                        got, want = got[..., :width, :], want[..., :width, :]
                    self.assertTrue(
                        mx.array_equal(got, want).item(),
                        f"layer {index} {name} kept rejected-draft state",
                    )
                # The raw indexer ledger must span exactly the cursor.
                if entry_a.index_keys is not None:
                    self.assertEqual(entry_a.index_keys.shape[1], entry_a.offset)


class TestRaggedRollbackComposition(unittest.TestCase):
    """One ragged verify rewind across all four Qwen4 state families.

    Rows accept different numbers of drafts, so the trunk must rewind the
    QSA KV, the raw indexer ledger, the GDN conv+recurrence and the PLE
    ShortConv+token history by DIFFERENT amounts in one call, and land every
    row where its own accepted prefix would have left it.  This is the model
    half of gate 2 in the batch-composition plan; ``tests/test_ragged_trim.py``
    owns the cache primitives.
    """

    PROMPTS = ([1, 2, 3, 4, 5, 6, 7], [8, 9, 10])
    DRAFTS = ([20, 21, 22], [30, 31, 32])
    ACCEPTED = (2, 0)
    NEXT = (40, 41)
    SHAPE_NOISE_BAND = 3e-3

    def _model(self):
        mx.random.seed(11)
        return TextModel(tiny_args(ple_layer_ids=[2]))

    def test_ragged_rewind_lands_every_row_on_its_own_accepted_prefix(self):
        from mlx_lm.models.cache import trim_ragged_prompt_cache

        model = self._model()
        caches = []
        for prompt in self.PROMPTS:
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
            caches.append(cache)
        batch = _merge_caches(caches)
        for layer in batch:
            layer.start_speculation()
        mx.eval(model(mx.array(self.DRAFTS, dtype=mx.int32), cache=batch))

        drops = [len(d) - a for d, a in zip(self.DRAFTS, self.ACCEPTED)]
        self.assertGreater(len(set(drops)), 1, "the rewind is not ragged")
        self.assertEqual(trim_ragged_prompt_cache(batch, drops), drops)
        for layer in batch:
            layer.stop_speculation()

        qsa = [layer for layer in batch if isinstance(layer, BatchQSAKVCache)]
        self.assertTrue(qsa)
        for cache in qsa:
            self.assertEqual(cache.index_keys.shape[1], cache._idx)
            self.assertEqual(
                cache.offset.tolist(),
                [
                    len(prompt) + accepted
                    for prompt, accepted in zip(self.PROMPTS, self.ACCEPTED)
                ],
            )

        got = np.asarray(
            model(
                mx.array([[token] for token in self.NEXT], dtype=mx.int32),
                cache=batch,
            )
        )
        for row, (prompt, drafts, accepted) in enumerate(
            zip(self.PROMPTS, self.DRAFTS, self.ACCEPTED)
        ):
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
            if accepted:
                mx.eval(
                    model(
                        mx.array([drafts[:accepted]], dtype=mx.int32), cache=cache
                    )
                )
            want = np.asarray(
                model(mx.array([[self.NEXT[row]]], dtype=mx.int32), cache=cache)
            )
            scale = float(np.abs(want).max())
            error = float(np.abs(got[row : row + 1] - want).max()) / scale
            self.assertLess(
                error,
                self.SHAPE_NOISE_BAND,
                f"row {row}: {error:.3e} of output scale",
            )
            self.assertEqual(
                int(got[row, -1].argmax()),
                int(want[0, -1].argmax()),
                f"row {row}: argmax differs after the ragged rewind",
            )


class TestBatchedMTPSharedTopK(unittest.TestCase):
    """QSA top-k sharing across an MTP draft cycle, on a batched head cache.

    The flag and the shared tensor lived only on the single-sequence
    ``QSAKVCache``, so ``mtp_start_cycle`` silently did nothing once the head
    cache was merged and every draft step paid the full index projection.
    Under a batch the shared set is per lane: ``selected[:, -1]`` is already
    ``[B, k]`` and each row reuses its own LOGICAL blocks.
    """

    PROMPTS = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15])
    SHAPE_NOISE_BAND = 3e-3

    def _model(self):
        mx.random.seed(11)
        args = tiny_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        return Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))

    def _seed_head(self, model, prompt):
        """Teacher-force one sequence into its own MTP head cache."""
        trunk = model.make_cache()
        _, hyper = model.mtp_backbone(mx.array([prompt], dtype=mx.int32), trunk)
        logits, post = model.mtp_step(
            hyper[:, :-1],
            mx.array([prompt[1:]], dtype=mx.int32),
            (head := model.make_mtp_cache()),
        )
        mx.eval(logits, post, head[0].state)
        return head, post[:, -1:], mx.argmax(logits[:, -1:, :], axis=-1)

    def _cycle(self, model, head, hidden, token, steps=2):
        model.mtp_start_cycle(head, share_qsa_indices=True)
        outputs = []
        for _ in range(steps):
            logits, hidden = model.mtp_step(hidden, token, head)
            mx.eval(logits, hidden, head[0].state)
            outputs.append(np.asarray(logits))
            token = mx.argmax(logits[:, -1:, :], axis=-1)
            hidden = hidden[:, -1:]
        return outputs

    def test_shared_topk_is_per_lane_and_skips_the_second_projection(self):
        model = self._model()
        heads, hiddens, tokens = zip(
            *(self._seed_head(model, prompt) for prompt in self.PROMPTS)
        )
        singles = [
            self._cycle(model, head, hidden, token)
            for head, hidden, token in zip(heads, hiddens, tokens)
        ]

        model = self._model()
        heads, hiddens, tokens = zip(
            *(self._seed_head(model, prompt) for prompt in self.PROMPTS)
        )
        batch = _merge_caches([list(head) for head in heads])
        cache = batch[0]
        self.assertIsInstance(cache, BatchQSAKVCache)
        self.assertEqual(cache.left_padding.tolist(), [0, 5])

        model.mtp_start_cycle(batch, share_qsa_indices=True)
        self.assertTrue(cache._mtp_share_topk)
        logits, hidden = model.mtp_step(
            mx.concatenate(list(hiddens)),
            mx.concatenate(list(tokens)),
            batch,
        )
        mx.eval(logits, hidden, cache.state)
        rows = len(self.PROMPTS)
        self.assertIsNotNone(cache._mtp_shared_topk)
        self.assertEqual(cache._mtp_shared_topk.shape[0], rows)
        width = cache.index_keys.shape[1]

        batched = [np.asarray(logits)]
        token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits, hidden = model.mtp_step(hidden[:, -1:], token, batch)
        mx.eval(logits, hidden, cache.state)
        batched.append(np.asarray(logits))
        # The point of sharing: step 1 reuses step 0's blocks and never
        # projects (or appends) an index key of its own.
        self.assertEqual(cache.index_keys.shape[1], width)

        for step, (got, wants) in enumerate(zip(batched, zip(*singles))):
            for row, want in enumerate(wants):
                scale = float(np.abs(want).max())
                error = float(np.abs(got[row : row + 1] - want).max()) / scale
                self.assertLess(
                    error,
                    self.SHAPE_NOISE_BAND,
                    f"step {step} row {row}: {error:.3e} of output scale",
                )
                self.assertEqual(
                    int(got[row, -1].argmax()),
                    int(want[0, -1].argmax()),
                    f"step {step} row {row}: argmax differs",
                )

        trim_prompt_cache(batch, 2)
        self.assertIsNone(cache._mtp_shared_topk)
        self.assertFalse(cache._mtp_share_topk)


class TestPLEPadSafety(unittest.TestCase):
    """PLE state under a padded batch row must equal that row decoded alone.

    Unlike logits, PLE state is deterministic bookkeeping -- an integer token
    history and a pure slice of the conv buffer -- so the bar here is exact
    equality, not a divergence band.  Two defects broke it (design review of
    ``wiki/docs/plans/qwen38-mtp-batch-composition.md`` section 2):

    * ``_short_conv`` cut the persistent tail at the PADDED width, so a short
      row's conv state advanced through its filler positions.  Only the branch
      output was masked, and the output is causal, so nothing downstream saw
      it in the same forward -- the damage landed in the next one.
    * The n-gram history took every input id, so pad id 0 entered ``cache[3]``
      as a real token and (with leading pads) also entered the hash of the
      row's FIRST real tokens, where EOS belongs.

    Both are prerequisites for the section-2 uniform-width verify, where rows
    with ``k_i < k_max`` carry trailing pads every cycle.
    """

    PLE_LAYER = 1  # ple_layer_ids=[2] is layer_idx + 1
    CONV, HISTORY = 2, 3

    def _layer(self, args):
        mx.random.seed(5)
        return PLELayer(args, self.PLE_LAYER, 0)

    @staticmethod
    def _prefix_mask(lengths, width):
        return mx.arange(width)[None, :] < mx.array(lengths)[:, None]

    @staticmethod
    def _suffix_mask(padding, width):
        return mx.arange(width)[None, :] >= mx.array(padding)[:, None]

    def test_short_conv_tail_stops_at_each_row_valid_end(self):
        args = tiny_args(ple_layer_ids=[2])
        layer = self._layer(args)
        state_len = layer.short_conv_state_len
        channels = args.hidden_size * args.hc_count
        lengths = [6, 4, 1]
        width = 6
        mx.random.seed(7)
        x = mx.random.normal((len(lengths), width, channels))
        previous = mx.random.normal((len(lengths), state_len, channels))

        batch = Qwen4ArraysCache(4)
        batch[self.CONV] = previous
        layer._short_conv(x, batch, self._prefix_mask(lengths, width))
        mx.eval(batch[self.CONV])
        naive = mx.contiguous(
            mx.concatenate([previous, x], axis=1)[:, -state_len:, :]
        )

        for row, length in enumerate(lengths):
            single = Qwen4ArraysCache(4)
            single[self.CONV] = previous[row : row + 1]
            layer._short_conv(x[row : row + 1, :length], single, None)
            mx.eval(single[self.CONV])
            self.assertTrue(
                mx.array_equal(
                    batch[self.CONV][row : row + 1], single[self.CONV]
                ).item(),
                f"row {row}: conv tail differs from the solo run",
            )
            # Non-vacuous: the padded rows are exactly the ones the old
            # width-based tail got wrong.
            padded = length < width
            self.assertEqual(
                padded,
                not mx.array_equal(
                    batch[self.CONV][row : row + 1], naive[row : row + 1]
                ).item(),
                f"row {row}: padding/naive-tail disagreement",
            )

    def test_short_conv_tail_is_unchanged_by_leading_pads(self):
        # ``make_mask`` also builds ``pos >= left_padding``.  Leading pads
        # already leave the tail at the padded width, so the fix must be a
        # no-op there -- the whole buffer, state included, is the same one.
        args = tiny_args(ple_layer_ids=[2])
        layer = self._layer(args)
        channels = args.hidden_size * args.hc_count
        mx.random.seed(8)
        x = mx.random.normal((2, 5, channels))
        masked = Qwen4ArraysCache(4)
        layer._short_conv(x, masked, self._suffix_mask([0, 3], 5))
        plain = Qwen4ArraysCache(4)
        layer._short_conv(x, plain, None)
        mx.eval(masked[self.CONV], plain[self.CONV])
        self.assertTrue(
            mx.array_equal(masked[self.CONV], plain[self.CONV]).item()
        )

    def _ngram(self, args):
        mx.random.seed(5)
        return NGramEmbedding(args, args.ple_embed_dim, self.PLE_LAYER, 0)

    def test_ngram_history_stops_at_each_row_valid_end(self):
        args = tiny_args(ple_layer_ids=[2])
        lengths = [6, 3, 1]
        width = 6
        rows = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0], [10, 0, 0, 0, 0, 0]]
        ids = mx.array(rows, dtype=mx.int32)
        for backend in ("cpu", "routed_cpu", "metal_prefill"):
            with self.subTest(backend=backend), ple_hash_backend(backend):
                embedding = self._ngram(args)
                batch = Qwen4ArraysCache(4)
                embedding.ngram_ids(ids, batch, self._prefix_mask(lengths, width))
                mx.eval(batch[self.HISTORY])
                history = np.asarray(batch[self.HISTORY])
                for row, length in enumerate(lengths):
                    single = Qwen4ArraysCache(4)
                    embedding.ngram_ids(
                        ids[row : row + 1, :length], single, None
                    )
                    mx.eval(single[self.HISTORY])
                    np.testing.assert_array_equal(
                        history[row : row + 1],
                        np.asarray(single[self.HISTORY]),
                        f"{backend} row {row}",
                    )
                # Pad id 0 is not a token of any of these rows, so its
                # presence anywhere in the history IS the defect.
                self.assertFalse(
                    bool((history == 0).any()),
                    f"{backend}: a pad id entered the token history",
                )

    def test_ngram_ids_with_leading_pads_hash_eos_not_pad(self):
        # A left-padded row's first real tokens hash against their PREVIOUS
        # tokens.  With pads left raw those are id 0; alone they are EOS, the
        # sentinel the segment shift resets on.
        args = tiny_args(ple_layer_ids=[2])
        padding = [0, 4]
        width = 7
        rows = [[1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 11, 12, 13]]
        ids = mx.array(rows, dtype=mx.int32)
        for backend in ("cpu", "routed_cpu", "metal_prefill"):
            with self.subTest(backend=backend), ple_hash_backend(backend):
                embedding = self._ngram(args)
                batch = Qwen4ArraysCache(4)
                batched = np.asarray(
                    embedding.ngram_ids(
                        ids, batch, self._suffix_mask(padding, width)
                    )
                )
                raw = np.asarray(
                    embedding.ngram_ids(ids, Qwen4ArraysCache(4), None)
                )
                mx.eval(batch[self.HISTORY])
                for row, pad in enumerate(padding):
                    single = Qwen4ArraysCache(4)
                    reference = np.asarray(
                        embedding.ngram_ids(ids[row : row + 1, pad:], single, None)
                    )
                    np.testing.assert_array_equal(
                        batched[row : row + 1, pad:],
                        reference,
                        f"{backend} row {row}: ids differ from the solo run",
                    )
                    np.testing.assert_array_equal(
                        np.asarray(batch[self.HISTORY])[row : row + 1],
                        np.asarray(single[self.HISTORY]),
                        f"{backend} row {row}: history differs",
                    )
                    if pad:
                        # Non-vacuous: pad ids really did change the hash.
                        self.assertTrue(
                            bool((batched[row, pad:] != raw[row, pad:]).any()),
                            f"{backend} row {row}: pads were already inert",
                        )


class TestRaggedBatchRecurrentState(unittest.TestCase):
    """A right-padded continuation must leave every row's recurrent state
    exactly where that row's own decode leaves it.

    This is the section-2 uniform-width verify geometry, and the one the
    batch generator already drives today (``PromptBatch.prompt`` right-pads a
    ragged prompt chunk, calls ``prepare(lengths=..., right_padding=...)``,
    then ``finalize()``).  Until the PLE pad-safety fix the PLE half of it was
    wrong and only the per-lane state checkpoints hid it.

    Precondition worth naming: this drives a merge of ALREADY-PREFILLED
    caches, where ``ArraysCache.left_padding`` is ``None`` and ``make_mask``
    therefore builds the ``pos < lengths`` mask.  A merge of EMPTY caches sets
    ``left_padding = [0] * B``, which takes precedence in ``make_mask`` and
    returns an all-true mask, so a fresh ragged batch masks nothing at all
    (``cache.py``; reported separately, not fixable from this module).
    """

    CASES = (
        ([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10]], [[40], [41, 42, 43, 44, 45]]),
        (
            [list(range(1, 15)), [15, 16, 17]],
            [[40, 41, 42], [43]],
        ),
        (
            [[1, 2], [3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13]],
            [[40, 41, 42, 43], [44], [45, 46]],
        ),
    )
    DECODE = (50, 51, 52)
    PLE_LAYER = 1
    # Same class and band as TestQSALeftPaddedBatch.SHAPE_NOISE_BAND: a
    # batched row and a solo row are different kernel shapes, so the float
    # halves of the state are compared at that tolerance.  The integer token
    # history is compared exactly.
    STATE_NOISE_BAND = 3e-3

    def _model(self):
        mx.random.seed(11)
        return TextModel(tiny_args(ple_layer_ids=[2]))

    def _batch(self, model, prompts, continuations):
        caches = []
        for prompt in prompts:
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
            caches.append(cache)
        batch = _merge_caches(caches)
        lengths = [len(c) for c in continuations]
        width = max(lengths)
        for layer in batch:
            layer.prepare(
                lengths=lengths,
                right_padding=[width - length for length in lengths],
            )
        mx.eval(
            model(_right_pad_prompts(continuations, max_length=width), cache=batch)
        )
        for layer in batch:
            layer.finalize()
        rows = len(prompts)
        for token in self.DECODE:
            mx.eval(
                model(
                    mx.array([[token]] * rows, dtype=mx.int32), cache=batch
                )
            )
        return batch

    def _single(self, model, prompt, continuation):
        cache = model.make_cache()
        mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
        mx.eval(model(mx.array([continuation], dtype=mx.int32), cache=cache))
        for token in self.DECODE:
            mx.eval(model(mx.array([[token]], dtype=mx.int32), cache=cache))
        return cache

    def _assert_state_matches(self, got, want, where):
        got, want = np.asarray(got), np.asarray(want)
        self.assertEqual(got.shape, want.shape, where)
        if np.issubdtype(want.dtype, np.integer):
            np.testing.assert_array_equal(got, want, where)
            return
        scale = float(np.abs(want).max())
        self.assertGreater(scale, 0.0, f"{where}: state is all zero")
        error = float(np.abs(got - want).max()) / scale
        self.assertLess(
            error, self.STATE_NOISE_BAND, f"{where}: {error:.3e} of state scale"
        )

    def test_ragged_continuation_leaves_each_row_where_solo_decode_does(self):
        for prompts, continuations in self.CASES:
            with self.subTest(
                prompts=[len(p) for p in prompts],
                continuations=[len(c) for c in continuations],
            ):
                model = self._model()
                batch = self._batch(model, prompts, continuations)
                singles = [
                    self._single(model, prompt, continuation)
                    for prompt, continuation in zip(prompts, continuations)
                ]
                ple = batch[self.PLE_LAYER]
                self.assertIsInstance(ple, Qwen4ArraysCache)
                self.assertEqual(len(ple.cache), 4)
                slots = ("gdn conv", "gdn state", "ple conv", "ple history")
                for row, single in enumerate(singles):
                    for slot, name in enumerate(slots):
                        self._assert_state_matches(
                            ple[slot][row : row + 1],
                            single[self.PLE_LAYER][slot],
                            f"row {row} {name}",
                        )
                # The history is the integer half, and pad id 0 is not a
                # token of any of these sequences.
                self.assertFalse(
                    bool((np.asarray(ple[3]) == 0).any()),
                    "a pad id survived in the token history",
                )

    def test_ragged_continuation_keeps_the_rows_decoding_alike(self):
        # State exactness is only useful if the next forward agrees, so gate
        # the logits too -- divergence CLASS, per the plan doc's section 3.
        model = self._model()
        prompts, continuations = self.CASES[0]
        batch = self._batch(model, prompts, continuations)
        rows = len(prompts)
        batched = np.asarray(
            model(mx.array([[60]] * rows, dtype=mx.int32), cache=batch)
        )
        for row, (prompt, continuation) in enumerate(zip(prompts, continuations)):
            single = self._single(model, prompt, continuation)
            want = np.asarray(model(mx.array([[60]], dtype=mx.int32), cache=single))
            scale = float(np.abs(want).max())
            error = float(np.abs(batched[row : row + 1] - want).max()) / scale
            self.assertLess(error, 3e-3, f"row {row}: {error:.3e} of scale")
            self.assertEqual(
                int(batched[row, -1].argmax()),
                int(want[0, -1].argmax()),
                f"row {row}: argmax differs",
            )


class TestQSANAXAdmission(unittest.TestCase):
    def setUp(self):
        self.mode = qwen4_exp_module._QSA_NAX_KERNEL
        self.min_query = qwen4_exp_module._QSA_NAX_MIN_QUERY
        self.min_context = qwen4_exp_module._QSA_NAX_AUTO_MIN_PHYSICAL_KV
        qwen4_exp_module._QSA_NAX_KERNEL = None
        qwen4_exp_module._QSA_NAX_MIN_QUERY = 64
        qwen4_exp_module._QSA_NAX_AUTO_MIN_PHYSICAL_KV = 16_384
        qwen4_exp_module.qsa_nax_admission_status(reset=True)

    def tearDown(self):
        qwen4_exp_module._QSA_NAX_KERNEL = self.mode
        qwen4_exp_module._QSA_NAX_MIN_QUERY = self.min_query
        qwen4_exp_module._QSA_NAX_AUTO_MIN_PHYSICAL_KV = self.min_context
        qwen4_exp_module.qsa_nax_admission_status(reset=True)

    @staticmethod
    def selection(**overrides):
        values = {
            "kind": "explicit",
            "batch": 1,
            "length": 512,
            "physical_width": 16_384,
            "left_padding": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def decide(self, selection=None, **overrides):
        values = {
            "training": False,
            "layout_ok": True,
            "device_supported": True,
            "kernel_available": True,
        }
        values.update(overrides)
        return qwen4_exp_module.decide_qsa_nax_admission(
            selection or self.selection(), **values
        )

    def test_unset_environment_defaults_to_guarded_auto(self):
        name = "MLX_QWEN4_QSA_NAX_KERNEL_TEST_UNSET"
        previous = environ.pop(name, None)
        try:
            self.assertIsNone(qwen4_exp_module._env_auto_flag(name))
        finally:
            if previous is not None:
                environ[name] = previous

    def test_auto_admits_only_measured_single_user_envelope(self):
        decision = self.decide()
        self.assertTrue(decision.engage)
        self.assertEqual(decision.reason, "engaged_auto")

        refusals = (
            (self.selection(batch=2), {}, "auto_batch_gt_one"),
            (
                self.selection(physical_width=16_383),
                {},
                "auto_context_below_crossover",
            ),
            (self.selection(length=63), {}, "query_below_min"),
            (self.selection(kind="mask_only"), {}, "selection_mask_only"),
            (self.selection(), {"training": True}, "training"),
            (self.selection(), {"layout_ok": False}, "unsupported_layout"),
            (
                self.selection(),
                {"device_supported": False},
                "unsupported_device",
            ),
            (
                self.selection(),
                {"kernel_available": False},
                "kernel_unavailable",
            ),
        )
        for selection, kwargs, reason in refusals:
            with self.subTest(reason=reason):
                decision = self.decide(selection, **kwargs)
                self.assertFalse(decision.engage)
                self.assertEqual(decision.reason, reason)

        # The serving B1 cache publishes a zero-valued left-padding array even
        # when no row is padded. Presence is therefore not a padding oracle;
        # B1 is the auto gate, and padded math has an independent correctness
        # receipt.
        self.assertTrue(self.decide(self.selection(left_padding=object())).engage)

    def test_explicit_modes_preserve_hard_off_and_checked_on(self):
        qwen4_exp_module._QSA_NAX_KERNEL = False
        self.assertEqual(self.decide().reason, "explicit_off")

        qwen4_exp_module._QSA_NAX_KERNEL = True
        wide_batch = self.selection(
            batch=4, left_padding=object(), physical_width=1024
        )
        decision = self.decide(wide_batch)
        self.assertTrue(decision.engage)
        self.assertEqual(decision.reason, "engaged_on")
        self.assertEqual(
            self.decide(wide_batch, device_supported=False).reason,
            "unsupported_device",
        )

    def test_status_receipt_is_bounded_and_resettable(self):
        selection = self.selection()
        decision = self.decide(selection)
        qwen4_exp_module._record_qsa_nax_admission(selection, decision)
        status = qwen4_exp_module.qsa_nax_admission_status()
        self.assertEqual(status["mode"], "auto")
        self.assertEqual(status["counts"], {"engaged_auto": 1})
        self.assertEqual(status["last_decision"]["physical_kv"], 16_384)
        qwen4_exp_module.qsa_nax_admission_status(reset=True)
        self.assertEqual(
            qwen4_exp_module.qsa_nax_admission_status()["counts"], {}
        )


if __name__ == "__main__":
    unittest.main()


class TestQSAKVQuantizationRefused(unittest.TestCase):
    """``--kv-bits`` over a QSA cache: both classes refuse, neither goes quiet.

    The two classes used to fail in OPPOSITE directions.  ``QSAKVCache``
    inherited ``KVCache.to_quantized`` and converted into a plain
    ``QuantizedKVCache``, dropping ``index_keys`` and every ``_QSA_CYCLE_STATE``
    field, so the indexer's ``cache.update_index_keys(raw)`` hit an object
    without the method.  ``BatchQSAKVCache`` had no ``to_quantized`` at all, so
    the ``hasattr`` gate in ``maybe_quantize_kv_cache`` skipped it and the
    batched path ignored the user's ``--kv-bits`` in silence -- the mode this
    class exists to keep closed, because silence is the one failure a serving
    run does not report.

    Refusal, not support: carrying the QSA side state through quantization
    needs quantized twins of the whole batch ledger protocol and its own
    equivalence battery.  These tests pin the refusal, and they are what a
    future implementation has to come back and rewrite deliberately.
    """

    GROUP = 64
    BITS = 8

    # A head dim the quantizer actually accepts (divisible by GROUP).  With a
    # narrower one the old inherited KVCache.to_quantized died on the group
    # size instead of converting, which would have made these tests pass
    # against the very bug they exist to pin.
    DIM = GROUP

    def _single(self, width=8):
        cache = QSAKVCache()
        values = mx.zeros((1, 1, width, self.DIM))
        cache.update_and_fetch(values, values)
        cache.update_index_keys(
            mx.zeros((1, width, self.DIM), dtype=mx.float32)
        )
        return cache

    def _batch(self, rows=2, width=8):
        cache = BatchQSAKVCache([0] * rows)
        values = mx.zeros((rows, 1, width, self.DIM))
        cache.update_and_fetch(values, values)
        cache.update_index_keys(mx.zeros((rows, width, self.DIM)))
        return cache

    def _caches(self):
        return (self._single(), self._batch())

    def test_the_two_classes_refuse_through_one_object(self):
        # Identity, not just equal behaviour: bound to one function, the two
        # classes cannot drift apart into opposite failure modes again.
        for cls in (QSAKVCache, BatchQSAKVCache):
            self.assertIn("to_quantized", cls.__dict__, cls.__name__)
            self.assertTrue(cls.kv_quantization_unsupported, cls.__name__)
        self.assertIs(
            QSAKVCache.__dict__["to_quantized"],
            BatchQSAKVCache.__dict__["to_quantized"],
        )
        self.assertEqual(
            QSAKVCache.kv_quantization_unsupported,
            BatchQSAKVCache.kv_quantization_unsupported,
        )
        # And it is NOT the inherited converter that dropped the ledger.
        self.assertIsNot(
            QSAKVCache.__dict__["to_quantized"], KVCache.__dict__["to_quantized"]
        )

    def test_to_quantized_refuses_on_every_call_shape(self):
        # maybe_quantize_kv_cache calls the symmetric shape positionally-ish
        # and the asymmetric/rotated shape with three more kwargs; a refusal
        # that only covered one would be a crash on the other.
        shapes = (
            {"group_size": 64, "bits": 8},
            {
                "group_size": 64,
                "bits": 8,
                "key_bits": 8,
                "value_bits": 4,
                "rotate": True,
            },
        )
        for cache in self._caches():
            for kwargs in shapes:
                with self.subTest(cache=type(cache).__name__, shape=len(kwargs)):
                    with self.assertRaises(NotImplementedError):
                        cache.to_quantized(**kwargs)

    def _assert_refused(self, cache, **kwargs):
        prompt_cache = [cache]
        with self.assertRaises(ValueError) as raised:
            maybe_quantize_kv_cache(
                prompt_cache, 0, self.GROUP, self.BITS, **kwargs
            )
        message = str(raised.exception)
        self.assertIn(type(cache).__name__, message)
        # The reason travels, not just the refusal.
        self.assertIn("index_keys", message)
        # Refused whole: not replaced, not half-converted, ledger intact.
        self.assertIs(prompt_cache[0], cache)
        self.assertIsNotNone(cache.index_keys)
        self.assertFalse(hasattr(cache, "bits"))

    def test_single_sequence_refuses_instead_of_dropping_the_ledger(self):
        cache = self._single()
        self._assert_refused(cache)
        # The method the indexer calls every forward is still there -- the old
        # QuantizedKVCache conversion is exactly what took it away.
        self.assertTrue(hasattr(cache, "update_index_keys"))

    def test_batched_path_is_not_silently_skipped(self):
        # THE regression.  Before ``to_quantized`` existed on this class the
        # hasattr gate skipped it and this call returned cleanly, leaving an
        # unquantized cache and a user who believed --kv-bits took effect.
        cache = self._batch()
        self.assertTrue(hasattr(cache, "to_quantized"))
        self._assert_refused(cache)

    def test_batched_refusal_is_not_a_stray_array_truthiness_error(self):
        # A B>1 batch cache reports ``offset`` as an mx.array, so reaching the
        # ``offset >= quantized_kv_start`` gate at all would raise a confusing
        # "[convert] Only length-1 arrays" ValueError instead of the reason.
        cache = self._batch(rows=3)
        self.assertIsInstance(cache.offset, mx.array)
        with self.assertRaises(ValueError) as raised:
            maybe_quantize_kv_cache([cache], 0, self.GROUP, self.BITS)
        self.assertNotIn("length-1", str(raised.exception))

    def test_asymmetric_and_rotated_requests_are_refused_too(self):
        for cache in self._caches():
            with self.subTest(cache=type(cache).__name__, mode="asymmetric"):
                self._assert_refused(cache, kv_key_bits=8, kv_value_bits=4)
        for cache in self._caches():
            with self.subTest(cache=type(cache).__name__, mode="rotated"):
                self._assert_refused(cache, kv_rotate=True)

    def test_refusal_survives_the_cachelist_recursion(self):
        # Hybrid models nest their KV leaves one level down, and
        # maybe_quantize_kv_cache recurses on purpose so nested leaves honor
        # kv_bits -- which is the recursion that reaches QSA leaves at all.
        for leaf in self._caches():
            with self.subTest(cache=type(leaf).__name__):
                wrapper = CacheList(leaf)
                with self.assertRaisesRegex(ValueError, "index_keys"):
                    maybe_quantize_kv_cache(
                        [wrapper], 0, self.GROUP, self.BITS
                    )
                self.assertIs(wrapper.caches[0], leaf)

    def test_refusal_does_not_wait_for_quantized_kv_start(self):
        # A cache that can NEVER be quantized fails at setup, not at the step
        # where its offset first crosses the threshold: a server that starts
        # and then dies mid-stream is strictly worse than one that will not
        # start.
        for cache in self._caches():
            with self.subTest(cache=type(cache).__name__):
                with self.assertRaisesRegex(ValueError, "index_keys"):
                    maybe_quantize_kv_cache(
                        [cache], 1 << 30, self.GROUP, self.BITS
                    )

    def test_unquantized_serving_is_left_alone(self):
        # The live bf16 profile: kv_bits=None returns early and must stay a
        # no-op, refusal or not.
        for cache in self._caches():
            with self.subTest(cache=type(cache).__name__):
                prompt_cache = [cache]
                maybe_quantize_kv_cache(prompt_cache, 0, self.GROUP, None)
                self.assertIs(prompt_cache[0], cache)
                self.assertIsNotNone(cache.index_keys)

    def test_the_cache_the_model_actually_builds_is_refused(self):
        # Against ``make_cache``'s own leaves, so a future make_cache that
        # hands back some other QSA cache class still has to refuse.
        model = Model(
            ModelArgs(model_type="qwen4_exp", text_config=tiny_args().__dict__)
        )
        prompt_cache = make_prompt_cache(model)
        self.assertTrue(
            any(isinstance(c, QSAKVCache) for c in prompt_cache),
            "the tiny config has no QSA layer, so this asserts nothing",
        )
        with self.assertRaisesRegex(ValueError, "index_keys"):
            maybe_quantize_kv_cache(prompt_cache, 0, self.GROUP, self.BITS)


def _oracle_dense_mask(
    selection, *, causal=True, ignore_validity=False, ignore_padding=False
):
    """Rebuild the QSA mask independently, in NumPy, with explicit loops.

    Deliberately shares NO code with ``QSASelection.dense_mask`` or with the
    production compactor: it re-derives the selected set, the clipped block
    lookup, the incomplete tail, the left-padding cut and the causal
    conjunction from the object's own fields.  The keyword switches drop one
    term at a time so a test can show that term is load-bearing.
    """
    ratio = selection.block_size
    n_blocks = selection.n_blocks
    batch, length = selection.batch, selection.length
    total = selection.physical_width
    ids = np.asarray(selection.raw_block_ids)
    valid = np.asarray(selection.valid_blocks)
    q_pos = np.asarray(selection.q_positions)
    tok_pos = np.asarray(selection.token_positions)
    padded = selection.left_padding is not None
    out = np.zeros((batch, 1, length, total), dtype=bool)
    for row in range(batch):
        for query in range(length):
            chosen = {
                int(i)
                for i in ids[row, query]
                if ignore_validity or valid[row % valid.shape[0], query, int(i)]
            }
            position = int(q_pos[row % q_pos.shape[0], query])
            tail_low = ((position + 1) // ratio) * ratio
            for column in range(total):
                logical = int(tok_pos[row % tok_pos.shape[0], column])
                block = min(max(logical // ratio, 0), n_blocks - 1)
                hit = block in chosen or tail_low <= logical <= position
                cut = padded and logical < 0 and not ignore_padding
                out[row, 0, query, column] = hit and not cut
    if causal and selection.causal_mask is not None:
        out = out & np.asarray(selection.causal_mask)
    return out


class TestQSASelectionObject(unittest.TestCase):
    """The gather-sparse step 1 seam: selection is an object, the mask is a
    method on it, and the compact view is lazy.  Nothing observable changes.
    """

    RATIO = 4  # tiny_args indexer_compress_ratio; block_topk is 2

    def _model(self):
        mx.random.seed(11)
        return TextModel(
            tiny_args(ple_layer_ids=[], layer_types=["full_attention"] * 4)
        )

    @contextmanager
    def _selections(self):
        """Collect every ``QSASelection`` the indexer returns, in call order."""
        records = []
        original = QSAIndexer.__call__

        def spy(indexer, hidden, causal_mask, cache, projected_qk=None):
            out = original(
                indexer, hidden, causal_mask, cache, projected_qk=projected_qk
            )
            records.append(out)
            return out

        QSAIndexer.__call__ = spy
        try:
            yield records
        finally:
            QSAIndexer.__call__ = original

    def _merged(self, model, prompts):
        caches = []
        for prompt in prompts:
            cache = model.make_cache()
            mx.eval(model(mx.array([prompt], dtype=mx.int32), cache=cache))
            caches.append(cache)
        return _merge_caches(caches)

    def _drive(self, model, records):
        """A schedule that reaches every selection kind and both geometries."""
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache))
        for chunk in ([4, 5, 6, 7, 8, 9], [10], [11, 12], [13], [14]):
            mx.eval(model(mx.array([chunk], dtype=mx.int32), cache=cache))
        batch = self._merged(model, [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]])
        for chunk in ([[20, 21], [20, 21]], [[22], [22]], [[23], [23]]):
            mx.eval(model(mx.array(chunk, dtype=mx.int32), cache=batch))
        return records

    # ---- 1. the mask is what it always was -----------------------------

    def test_dense_mask_matches_an_independent_numpy_oracle(self):
        model = self._model()
        with self._selections() as records:
            self._drive(model, records)
        explicit = [s for s in records if s.kind == "explicit"]
        self.assertTrue(explicit, "the schedule never went sparse")
        dropped = 0
        for index, selection in enumerate(explicit):
            mask = selection.dense_mask()
            self.assertIsNotNone(mask, f"selection {index}")
            np.testing.assert_array_equal(
                np.asarray(mask), _oracle_dense_mask(selection), f"selection {index}"
            )
            if selection.causal_mask is not None:
                causal = np.asarray(selection.causal_mask)
                dropped += int((causal & ~np.asarray(mask)).sum())
        # Without this the oracle could be agreeing on a dense mask only.
        self.assertGreater(dropped, 0, "no causal cell was ever removed")

    def test_every_kind_is_reached_and_tagged(self):
        model = self._model()
        with self._selections() as records:
            self._drive(model, records)
        kinds = {s.kind for s in records}
        self.assertIn("explicit", kinds)
        self.assertIn("implicit_all", kinds)
        for selection in records:
            if selection.kind == "implicit_all":
                # The dense return is the causal mask OBJECT, not a copy.
                self.assertIs(selection.dense_mask(), selection.causal_mask)

    def test_sink_window_cache_is_mask_only_and_has_no_blocks(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        cache = SinkWindowKVCache(window_size=4, sink_size=2, rollback_window=4)
        cache.offset = 6
        selection = indexer(
            mx.random.normal((1, 1, args.hidden_size)), None, cache
        )
        self.assertEqual(selection.kind, "mask_only")
        # Windowed MTP replaces global QSA; there is nothing to gather.
        self.assertIsNone(selection.compact_blocks())
        mask = selection.dense_mask()
        self.assertTrue(mask is None or mask.ndim == 4)

    # ---- 2. the causal conjunction is load-bearing ----------------------

    def test_clip_names_an_acausal_column_that_only_the_conjunction_removes(self):
        """The ratio-4 / total-10 / left-pad-1 geometry from the design.

        Row 1 holds one padding column, so physical column 9 is its logical
        position 8 -- FUTURE for the chunk's first query at logical 7.  The
        clip maps that column to block 1, which is selected and causally
        valid for query 7, and the tail range is empty.  Only the final
        ``causal_mask &`` removes the cell.
        """
        model = self._model()
        with self._selections() as records:
            batch = self._merged(
                model, [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]]
            )
            padding = next(
                layer.left_padding.tolist()
                for layer in batch
                if isinstance(layer, BatchQSAKVCache)
            )
            self.assertEqual(padding, [0, 1])
            del records[:]
            mx.eval(model(mx.array([[20, 21], [20, 21]], dtype=mx.int32), cache=batch))
        acausal = 0
        for selection in records:
            self.assertEqual(selection.kind, "explicit")
            self.assertEqual(selection.physical_width, 10)
            self.assertEqual(selection.n_blocks, 2)
            self.assertEqual(int(np.asarray(selection.q_positions)[1, 0]), 7)
            self.assertEqual(int(np.asarray(selection.token_positions)[1, 9]), 8)
            # The mask BEFORE the conjunction: same object, causal mask
            # dropped, so dense_mask() returns the raw sparse form.
            raw = np.asarray(replace(selection, causal_mask=None).dense_mask())
            self.assertTrue(
                bool(raw[1, 0, 0, 9]),
                "the clip no longer names the acausal column; this test is vacuous",
            )
            acausal += 1
            self.assertFalse(
                bool(np.asarray(selection.dense_mask())[1, 0, 0, 9]),
                "an acausal column survived into the QSA mask",
            )
        self.assertGreater(acausal, 0, "no selection was inspected")

    def test_the_raw_sparse_form_still_filters_by_block_validity(self):
        """``chosen & valid_blocks`` and the left-padding cut are invisible
        AFTER the causal conjunction -- it subsumes both -- so gate them on
        the pre-conjunction form, where they do real work.
        """
        model = self._model()
        with self._selections() as records:
            self._drive(model, records)
        explicit = [s for s in records if s.kind == "explicit"]
        self.assertTrue(explicit)
        loosened = padded = padding_cells = 0
        for index, selection in enumerate(explicit):
            raw = np.asarray(replace(selection, causal_mask=None).dense_mask())
            np.testing.assert_array_equal(
                raw,
                _oracle_dense_mask(selection, causal=False),
                f"selection {index}",
            )
            loosened += int(
                (
                    _oracle_dense_mask(
                        selection, causal=False, ignore_validity=True
                    )
                    != raw
                ).sum()
            )
            if selection.left_padding is not None:
                padded += 1
                # Padding is PER ROW; row 0 usually has none.
                for row, pad in enumerate(
                    np.asarray(selection.left_padding).tolist()
                ):
                    np.testing.assert_array_equal(
                        raw[row, ..., :pad],
                        False,
                        f"selection {index} row {row} attends its left padding",
                    )
                padding_cells += int(
                    (
                        _oracle_dense_mask(
                            selection, causal=False, ignore_padding=True
                        )
                        != raw
                    ).sum()
                )
        self.assertGreater(loosened, 0, "the validity filter removed nothing")
        self.assertGreater(padded, 0, "no left-padded selection was seen")
        self.assertGreater(padding_cells, 0, "the padding cut removed nothing")

    def test_mtp_shared_top_k_stores_the_raw_selection_not_the_compacted_one(self):
        """Compaction filters by the CURRENT query's validity.  Dropping
        invalid slots before caching would change which ids can become valid
        on a later query, so the cached set must be the raw one."""
        args = tiny_args()
        indexer = QSAIndexer(args)
        cache = QSAKVCache()
        hidden = mx.random.normal((1, 13, args.hidden_size), key=mx.random.key(8))
        causal = (mx.arange(13)[:, None] >= mx.arange(13)[None, :])[None, None]
        cache._mtp_share_topk = True
        cache._mtp_shared_topk = None
        selection = indexer(hidden, causal, cache)
        self.assertEqual(selection.kind, "explicit")
        raw_last = np.asarray(selection.raw_block_ids)[:, -1]
        np.testing.assert_array_equal(
            np.asarray(cache._mtp_shared_topk),
            raw_last,
            "the shared set is not the raw last-query selection",
        )
        compact_last = np.asarray(selection.compact_blocks().block_ids)[:, -1]
        self.assertFalse(
            np.array_equal(raw_last, compact_last),
            "raw and compacted ids coincide here, so this test is vacuous",
        )

    # ---- 3. the scatter lever is snapshotted, not reread -----------------

    def test_scatter_chosen_is_snapshot_at_selection_time(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        hidden = mx.random.normal((1, 13, args.hidden_size), key=mx.random.key(5))
        causal = (mx.arange(13)[:, None] >= mx.arange(13)[None, :])[None, None]
        previous = qwen4_exp_module._QSA_SCATTER_CHOSEN
        qwen4_exp_module._QSA_SCATTER_CHOSEN = True
        try:
            selection = indexer(hidden, causal, QSAKVCache())
            self.assertEqual(selection.kind, "explicit")
            self.assertTrue(selection.scatter_chosen)
        finally:
            qwen4_exp_module._QSA_SCATTER_CHOSEN = previous
        # The global is back to stock, but a mask built now must still take
        # the path the selection was made under.  Both paths give the same
        # mask, so watch WHICH one runs, not what it returns.
        self.assertTrue(selection.scatter_chosen)
        with self._count_scatters() as scatters:
            scattered = np.asarray(selection.dense_mask())
        self.assertEqual(scatters, [1], "the snapshot was ignored, the global was read")
        broadcast = replace(selection, scatter_chosen=False)
        qwen4_exp_module._QSA_SCATTER_CHOSEN = True
        try:
            with self._count_scatters() as scatters:
                stock = np.asarray(broadcast.dense_mask())
        finally:
            qwen4_exp_module._QSA_SCATTER_CHOSEN = previous
        self.assertEqual(scatters, [], "the global overrode a False snapshot")
        np.testing.assert_array_equal(scattered, stock)

    @contextmanager
    def _count_scatters(self):
        """Count ``mx.put_along_axis`` calls: the scatter path's fingerprint."""
        calls = []
        original = mx.put_along_axis

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        mx.put_along_axis = spy
        try:
            yield calls
        finally:
            mx.put_along_axis = original

    # ---- 4. compaction ---------------------------------------------------

    def _compact(self, ids, valid, n_blocks):
        packed, counts = _compact_qsa_block_ids(
            mx.array(ids, dtype=mx.uint32),
            mx.array(valid, dtype=mx.bool_),
            n_blocks=n_blocks,
        )
        return np.asarray(packed), np.asarray(counts)

    def test_compaction_packs_a_single_valid_id_out_of_the_last_slot(self):
        """``argpartition`` with one valid block leaves the first slots invalid;
        a prefix scan of the raw ids would read the wrong thing."""
        ids = [[[3, 1, 4, 1, 5, 9, 2, 6, 7]]]
        valid = [[[False] * 8 + [True]]]
        packed, counts = self._compact(ids, valid, n_blocks=16)
        self.assertEqual(counts.tolist(), [[1]])
        self.assertEqual(packed[0, 0, 0], 7)
        self.assertEqual(packed[0, 0, 1:].tolist(), [0] * 8)

    def test_compaction_sorts_a_mixed_unsorted_selection(self):
        packed, counts = self._compact(
            [[[5, 2, 7, 1]]], [[[True, False, True, True]]], n_blocks=8
        )
        self.assertEqual(counts.tolist(), [[3]])
        self.assertEqual(packed[0, 0, :3].tolist(), [1, 5, 7])
        self.assertEqual(packed[0, 0, 3], 0)

    def test_compaction_handles_empty_full_and_ragged_rows(self):
        ids = [[[2, 0, 1], [2, 0, 1]], [[1, 2, 0], [0, 2, 1]]]
        valid = [
            [[False, False, False], [True, True, True]],
            [[True, False, True], [False, True, False]],
        ]
        packed, counts = self._compact(ids, valid, n_blocks=3)
        self.assertEqual(counts.tolist(), [[0, 3], [2, 1]])
        self.assertEqual(packed[0, 0].tolist(), [0, 0, 0])
        self.assertEqual(packed[0, 1].tolist(), [0, 1, 2])
        self.assertEqual(packed[1, 0].tolist(), [0, 1, 0])
        self.assertEqual(packed[1, 1].tolist(), [2, 0, 0])

    def test_compaction_preserves_the_set_capacity_and_dtype(self):
        rng = np.random.default_rng(7)
        n_blocks, width = 12, 6
        ids = np.stack(
            [
                [rng.permutation(n_blocks)[:width] for _ in range(3)]
                for _ in range(2)
            ]
        ).astype(np.uint32)
        valid = rng.random((2, 3, width)) < 0.5
        source = mx.array(ids)
        packed, counts = _compact_qsa_block_ids(
            source, mx.array(valid), n_blocks=n_blocks
        )
        self.assertEqual(packed.dtype, source.dtype)
        self.assertEqual(packed.shape, source.shape)
        packed, counts = np.asarray(packed), np.asarray(counts)
        for row in range(2):
            for query in range(3):
                count = int(counts[row, query])
                prefix = packed[row, query, :count].tolist()
                self.assertEqual(
                    sorted(prefix), sorted(ids[row, query][valid[row, query]].tolist())
                )
                self.assertEqual(prefix, sorted(prefix))
                self.assertEqual(packed[row, query, count:].tolist(), [0] * (width - count))

    def test_compaction_rejects_mismatched_input(self):
        with self.assertRaises(ValueError):
            _compact_qsa_block_ids(
                mx.zeros((1, 2), dtype=mx.uint32),
                mx.zeros((1, 2), dtype=mx.bool_),
                n_blocks=4,
            )
        with self.assertRaises(ValueError):
            _compact_qsa_block_ids(
                mx.zeros((1, 2, 3), dtype=mx.uint32),
                mx.zeros((1, 2, 4), dtype=mx.bool_),
                n_blocks=4,
            )

    def test_compact_blocks_agrees_with_the_dense_mask(self):
        """Every compacted block must be one the mask actually attends, and
        the mask must attend nothing outside the compacted set plus tail."""
        model = self._model()
        with self._selections() as records:
            self._drive(model, records)
        checked = invalid_before_valid = 0
        for selection in records:
            if selection.kind != "explicit":
                continue
            compact = selection.compact_blocks()
            self.assertEqual(
                compact.block_ids.shape, selection.raw_block_ids.shape
            )
            packed = np.asarray(compact.block_ids)
            counts = np.asarray(compact.block_counts)
            valid = np.asarray(compact.block_valid)
            raw = np.asarray(selection.raw_block_ids)
            block_valid = np.asarray(selection.valid_blocks)
            q_pos = np.asarray(selection.q_positions)
            for row in range(selection.batch):
                for query in range(selection.length):
                    count = int(counts[row, query])
                    prefix = packed[row, query, :count].tolist()
                    expected = sorted(
                        {
                            int(i)
                            for i in raw[row, query]
                            if block_valid[row % block_valid.shape[0], query, int(i)]
                        }
                    )
                    self.assertEqual(prefix, expected)
                    self.assertEqual(prefix, sorted(prefix))
                    self.assertEqual(valid[row, query].tolist(), [True] * count + [False] * (len(packed[row, query]) - count))
                    position = int(q_pos[row % q_pos.shape[0], query])
                    self.assertEqual(int(np.asarray(compact.tail_stop)[row, query]), position + 1)
                    self.assertEqual(
                        int(np.asarray(compact.tail_start)[row, query]),
                        ((position + 1) // self.RATIO) * self.RATIO,
                    )
                    flags = [
                        bool(block_valid[row % block_valid.shape[0], query, int(i)])
                        for i in raw[row, query]
                    ]
                    if False in flags and True in flags and flags.index(False) < flags.index(True):
                        invalid_before_valid += 1
                    checked += 1
        self.assertGreater(checked, 0, "no explicit selection was compacted")
        self.assertGreater(
            invalid_before_valid,
            0,
            "compaction never saw an invalid slot before a valid one",
        )

    def test_compact_blocks_carries_the_causal_contract(self):
        model = self._model()
        with self._selections() as records:
            batch = self._merged(
                model, [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]]
            )
            del records[:]
            mx.eval(model(mx.array([[20, 21], [20, 21]], dtype=mx.int32), cache=batch))
        for selection in records:
            compact = selection.compact_blocks()
            # Blocks alone are not causally complete, so the contract travels
            # with them: left padding to reach physical columns, and the
            # causal mask the clip made necessary.
            self.assertIs(compact.causal_mask, selection.causal_mask)
            self.assertIs(compact.left_padding, selection.left_padding)
            self.assertEqual(compact.block_size, self.RATIO)
            self.assertEqual(compact.physical_width, 10)

    def test_implicit_all_compacts_to_every_causally_valid_block(self):
        args = tiny_args()
        indexer = QSAIndexer(args)
        cache = QSAKVCache()
        hidden = mx.random.normal((1, 9, args.hidden_size), key=mx.random.key(2))
        causal = (mx.arange(9)[:, None] >= mx.arange(9)[None, :])[None, None]
        with lever_flag("_QSA_DENSE_SHORTCIRCUIT"):
            selection = indexer(hidden, causal, cache)
        self.assertEqual(selection.kind, "implicit_all")
        compact = selection.compact_blocks()
        counts = np.asarray(compact.block_counts)[0].tolist()
        # Blocks are [0,4) and [4,8): query q closes block b when 4b+3 <= q.
        self.assertEqual(counts, [0, 0, 0, 1, 1, 1, 1, 2, 2])
        packed = np.asarray(compact.block_ids)
        self.assertEqual(packed[0, 7].tolist(), [0, 1])
        self.assertEqual(packed[0, 3].tolist(), [0, 0])

    # ---- 5. laziness -----------------------------------------------------

    def test_compaction_is_never_called_by_the_mask_path(self):
        """Sorting up to block_topk ids on every masked SDPA call would be new
        hot-path work.  The model and dense_mask() must not touch it."""
        calls = []
        original = qwen4_exp_module._compact_qsa_block_ids

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        qwen4_exp_module._compact_qsa_block_ids = spy
        try:
            model = self._model()
            with self._selections() as records:
                self._drive(model, records)
            for selection in records:
                mask = selection.dense_mask()
                if mask is not None:
                    mx.eval(mask)
            self.assertEqual(calls, [], "the mask path called the compactor")
            self.assertTrue(any(s.kind == "explicit" for s in records))
            # ... and it IS reachable, so the gate above is not vacuous.
            next(s for s in records if s.kind == "explicit").compact_blocks()
            self.assertEqual(len(calls), 1)
        finally:
            qwen4_exp_module._compact_qsa_block_ids = original

    # ---- 6. structural invariants ----------------------------------------

    def test_structural_invariants_are_asserted(self):
        base = dict(
            kind="explicit",
            batch=1,
            length=2,
            block_size=4,
            raw_block_ids=mx.zeros((1, 2, 2), dtype=mx.uint32),
            valid_blocks=mx.zeros((1, 2, 2), dtype=mx.bool_),
            q_positions=mx.zeros((1, 2), dtype=mx.int32),
            token_positions=mx.zeros((1, 8), dtype=mx.int32),
            physical_width=8,
            n_blocks=2,
        )
        QSASelection(**base)  # the good shape builds
        for field, value in (
            ("n_blocks", 3),
            ("valid_blocks", mx.zeros((1, 2, 3), dtype=mx.bool_)),
            ("raw_block_ids", mx.zeros((1, 2), dtype=mx.uint32)),
            ("token_positions", mx.zeros((1, 7), dtype=mx.int32)),
            ("left_padding", mx.zeros((2,), dtype=mx.int32)),
            ("causal_mask", mx.zeros((1, 1, 2, 7), dtype=mx.bool_)),
            ("kind", "gather"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    QSASelection(**{**base, field: value})


if __name__ == "__main__":
    unittest.main()
