# Copyright © 2026 Apple Inc.

import unittest
from contextlib import contextmanager
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory

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
from mlx_lm.generate import _merge_caches, _right_pad_prompts
from mlx_lm.models.cache import SinkWindowKVCache, trim_prompt_cache
from mlx_lm.models.qwen4_exp import (
    BatchQSAKVCache,
    GatedResidual,
    Model,
    ModelArgs,
    NGramEmbedding,
    QSAIndexer,
    QSAKVCache,
    TextModel,
    TextModelArgs,
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
        sparse = indexer(hidden, causal[None, None], cache)
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
            records.append(
                (
                    None if causal_mask is None else np.asarray(causal_mask),
                    None if out is None else np.asarray(out),
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


if __name__ == "__main__":
    unittest.main()
