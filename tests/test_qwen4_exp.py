# Copyright © 2026 Apple Inc.

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm import utils
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


class TestQwen4Exp(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
