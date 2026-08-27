# Copyright © 2026 Apple Inc.

import unittest
from contextlib import contextmanager
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm import utils
from mlx_lm.generate import _merge_caches
from mlx_lm.models.cache import SinkWindowKVCache, trim_prompt_cache
from mlx_lm.models.qwen4_exp import (
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

    def test_sanitize_refuses_norm_convention_mismatch(self):
        # mlx-vlm #2041/#2045 class: a wrong zero-vs-ones-centered guess
        # loads cleanly and produces deterministic garbage. The check must
        # not depend on the conv1d layout proxy that makes the guess.
        args = tiny_args()
        model = TextModel(args)
        raw_conv = mx.zeros((8, 1, 3))  # HF raw layout, shape[-1] != 1
        norms = {
            "model.layers.3.self_attn.q_norm.weight": mx.ones((8,)),
            "model.layers.3.self_attn.k_norm.weight": mx.ones((8,)),
        }

        # Ones-centered norms inside a raw-looking checkpoint: the +1
        # offset would shift gains to ~2. Must refuse, not load.
        with self.assertRaisesRegex(ValueError, "norm convention mismatch"):
            model.sanitize(
                {"model.layers.0.linear_attn.conv1d.weight": raw_conv, **norms}
            )

        # Zero-centered norms in a converted-layout checkpoint (offset
        # would be skipped, gains stay ~0) must also refuse.
        zero_norms = {key: mx.zeros((8,)) for key in norms}
        with self.assertRaisesRegex(ValueError, "norm convention mismatch"):
            model.sanitize(
                {
                    "model.layers.0.linear_attn.conv1d.weight": mx.zeros(
                        (8, 3, 1)
                    ),
                    **zero_norms,
                }
            )

        # The two consistent pairings convert to ~1-centered gains.
        raw_ok = model.sanitize(
            {"model.layers.0.linear_attn.conv1d.weight": raw_conv, **zero_norms}
        )
        converted_ok = model.sanitize(
            {
                "model.layers.0.linear_attn.conv1d.weight": mx.zeros((8, 3, 1)),
                **norms,
            }
        )
        for output in (raw_ok, converted_ok):
            gains = output["model.layers.3.self_attn.q_norm.weight"]
            self.assertAlmostEqual(gains.mean().item(), 1.0, places=5)

    def test_raw_moe_weights_are_split_for_switch_glu(self):
        args = tiny_args()
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        gate_up = mx.arange(4 * 16 * 16).reshape(4, 16, 16)
        down = mx.zeros((4, 16, 8))
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


if __name__ == "__main__":
    unittest.main()
