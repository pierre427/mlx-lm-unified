import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.apc import APCKey, AutomaticPrefixCache, AutomaticPrefixCacheV2
from mlx_lm.cache_planes import CachePlaneKind
from mlx_lm.models import agnes
from mlx_lm.models.cache import record_state_checkpoints
from mlx_lm.models.qwen3_5 import fuse_gated_delta_net_projections
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated
from mlx_lm.server import ResponseGenerator


class TestAgnes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.set_default_device(mx.cpu)

    def make_args(self, **overrides):
        text_config = {
            "model_type": "agnes_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 24,
            "parallel_ffn_intermediate_size": 8,
            "vocab_size": 32,
            "layer_types": [agnes.LAYER_DELTA, agnes.LAYER_GLOBAL],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 64,
            "mtp_num_hidden_layers": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            },
        }
        text_config.update(overrides)
        return agnes.ModelArgs.from_dict(
            {"model_type": "agnes", "text_config": text_config}
        )

    def test_exact_layer_plan_parallel_ffn_and_cached_decode(self):
        model = agnes.Model(self.make_args())
        self.assertIsInstance(model.layers[0].delta_attn, agnes.GatedDeltaNet)
        self.assertIsInstance(model.layers[1].global_attn, agnes.Qwen3NextAttention)
        self.assertIsNotNone(model.layers[0].mlp.parallel_ffn)

        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        full = model(tokens)
        cache = model.make_cache()
        steps = [model(tokens[:, i : i + 1], cache=cache) for i in range(4)]
        decoded = mx.concatenate(steps, axis=1)
        mx.eval(full, decoded)
        self.assertTrue(mx.allclose(full, decoded, rtol=2e-4, atol=2e-4))

    def test_apcv2_prefix_hit_miss_fork_and_cold_reuse_logits(self):
        model = agnes.Model(self.make_args())
        self.assertEqual(model.apc_v2_layout, "agnes-hybrid-layer-segments-v1")
        self.assertEqual(
            model.language_model.apc_v2_layout,
            "agnes-hybrid-layer-segments-v1",
        )
        apc = AutomaticPrefixCacheV2(max_size=2, layout_name=model.apc_v2_layout)
        key = APCKey(
            "agnes-tiny",
            revision="test-revision",
            cache_layout_fingerprint=model.apc_v2_layout,
        )
        prefix = mx.array([[1, 2, 3]], dtype=mx.int32)

        miss = apc.lookup(key, prefix[0].tolist())
        self.assertFalse(miss.hit)
        self.assertEqual(miss.miss_reason, "no_compatible_prefix")

        source = model.make_cache()
        source_logits = model(prefix, cache=source)
        mx.eval(source_logits, *[cache.state for cache in source])
        capabilities = apc.store(key, prefix[0].tolist(), source)
        self.assertEqual(capabilities.topology, "checkpointed_hybrid")

        incompatible = apc.lookup(
            APCKey(
                "agnes-tiny",
                revision="other-revision",
                cache_layout_fingerprint=model.apc_v2_layout,
            ),
            [1, 2, 3, 4],
        )
        self.assertFalse(incompatible.hit)

        suffix_a = mx.array([[4, 5]], dtype=mx.int32)
        suffix_b = mx.array([[6, 7]], dtype=mx.int32)
        hit = apc.lookup(key, [1, 2, 3, 4, 5])
        self.assertTrue(hit.hit)
        self.assertEqual(hit.cached_tokens, 3)
        self.assertEqual(hit.remaining_tokens, [4, 5])
        sibling = hit.cache.fork()

        reused_a = model(suffix_a, cache=hit.cache)
        reused_b = model(suffix_b, cache=sibling)
        cold_a_cache = model.make_cache()
        cold_b_cache = model.make_cache()
        model(prefix, cache=cold_a_cache)
        model(prefix, cache=cold_b_cache)
        cold_a = model(suffix_a, cache=cold_a_cache)
        cold_b = model(suffix_b, cache=cold_b_cache)
        mx.eval(reused_a, reused_b, cold_a, cold_b)

        # Compare the same prefill/decode split. A monolithic forward can use
        # different kernels and is not the cache-reuse fidelity authority.
        self.assertTrue(mx.array_equal(reused_a, cold_a))
        self.assertTrue(mx.array_equal(reused_b, cold_b))

        stats = apc.apc_stats
        self.assertEqual(stats["version"], 2)
        self.assertEqual(stats["layout_name"], model.apc_v2_layout)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 2)
        self.assertEqual(stats["cached_tokens"], 3)
        self.assertEqual(stats["layer_segments"]["fallback_entries"], 0)
        self.assertEqual(
            stats["layer_segments"]["by_plane"]["gdn_recurrent"]["layers"],
            1,
        )
        self.assertEqual(
            stats["layer_segments"]["by_plane"]["attention_kv"]["layers"],
            1,
        )
        self.assertEqual(
            stats["cow"]["planes"]["gdn_recurrent"]["materializations"],
            2,
        )
        self.assertEqual(
            stats["cow"]["planes"]["attention_kv"]["materializations"],
            2,
        )
        hit.cache.close()
        sibling.close()

    def test_server_selects_apcv2_for_native_agnes(self):
        response = ResponseGenerator.__new__(ResponseGenerator)
        response.prompt_cache = AutomaticPrefixCache(max_size=3, max_bytes=123456)
        response._cache_capsule_pool = None
        model = agnes.Model(self.make_args())

        response._configure_apc_for_model(model)

        self.assertIsInstance(response.prompt_cache, AutomaticPrefixCacheV2)
        self.assertEqual(response.prompt_cache.layout_name, model.apc_v2_layout)
        self.assertEqual(response.prompt_cache.max_size, 3)
        self.assertEqual(response.prompt_cache.max_bytes, 123456)
        response._cache_capsule_pool.close()

    def test_existing_gdn_projection_fusion_accepts_agnes_explicitly(self):
        model = agnes.Model(
            self.make_args(
                hidden_size=32,
                intermediate_size=64,
                parallel_ffn_intermediate_size=32,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
            )
        )
        projection_names = (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
        )
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda path, module: any(
                path.endswith(name) for name in projection_names
            ),
        )
        mx.eval(model.parameters())

        # The inherited optimization remains default-off and reports the
        # number of real Agnes delta layers it verified and rewrote.
        self.assertEqual(fuse_gated_delta_net_projections(model), 0)
        self.assertFalse(hasattr(model.layers[0].delta_attn, "in_proj_fused"))
        self.assertEqual(fuse_gated_delta_net_projections(model, enabled=True), 1)
        self.assertTrue(hasattr(model.layers[0].delta_attn, "in_proj_fused"))

    def test_apcv2_checkpoint_trim_and_atomic_invalidation(self):
        model = agnes.Model(self.make_args())
        prefix = mx.array([[1, 2, 3]], dtype=mx.int32)
        continuation = mx.array([[4, 5]], dtype=mx.int32)
        source = model.make_cache()
        prefix_logits = model(prefix, cache=source)
        mx.eval(prefix_logits, *[cache.state for cache in source])
        record_state_checkpoints(source, [3], force=True)
        long_logits = model(continuation, cache=source)
        mx.eval(long_logits, *[cache.state for cache in source])

        apc = AutomaticPrefixCacheV2(max_size=2, layout_name=model.apc_v2_layout)
        key = APCKey("agnes-trim")
        stored_tokens = [1, 2, 3, 4, 5]
        apc.store(key, stored_tokens, source)

        # The requested branch diverges inside the stored entry. APCv2 must
        # restore the GDN checkpoint and trim attention K/V to the same point.
        hit = apc.lookup(key, [1, 2, 3, 9])
        self.assertTrue(hit.hit)
        self.assertEqual(hit.hit_kind, "prefix")
        self.assertEqual(hit.cached_tokens, 3)
        self.assertEqual(hit.remaining_tokens, [9])
        self.assertEqual(
            hit.segment_manifest["by_plane"]["gdn_recurrent"]["segments"],
            1,
        )
        self.assertEqual(
            hit.segment_manifest["by_plane"]["attention_kv"]["segments"],
            1,
        )

        reused = model(
            mx.array([hit.remaining_tokens], dtype=mx.int32), cache=hit.cache
        )
        cold_cache = model.make_cache()
        model(prefix, cache=cold_cache)
        cold = model(mx.array([[9]], dtype=mx.int32), cache=cold_cache)
        mx.eval(reused, cold)
        self.assertTrue(mx.array_equal(reused, cold))
        hit.cache.close()

        entry = apc._trie.get(key, stored_tokens)
        owner = entry.prompt_cache.cow_owner
        recurrent_segment = next(
            segment
            for segment in owner.segment_manifest.segments
            if segment.key.kind == CachePlaneKind.GDN_RECURRENT
        )
        self.assertTrue(
            owner.invalidate_segment(recurrent_segment.key, "test-stale-gdn")
        )
        invalidated = apc.lookup(key, [1, 2, 3, 10])
        self.assertFalse(invalidated.hit)
        self.assertEqual(invalidated.miss_reason, "stale_cow_generation")

    def test_sanitize_raw_then_native_does_not_shift_norm_twice(self):
        raw_model = agnes.Model(self.make_args(mtp_num_hidden_layers=1))
        norm_key = "model.language_model.layers.0.input_layernorm.weight"
        conv_key = "model.language_model.layers.0.delta_attn.conv1d.weight"
        base = mx.arange(16, dtype=mx.float32)
        raw_conv = mx.zeros((16, 1, 4), dtype=mx.float32)
        converted = raw_model.sanitize(
            {
                norm_key: base,
                conv_key: raw_conv,
                "mtp.fc.weight": mx.zeros((16, 32)),
                "model.visual.stub": mx.zeros((1,)),
            }
        )

        mlx_norm_key = "language_model.model.layers.0.input_layernorm.weight"
        mlx_conv_key = "language_model.model.layers.0.delta_attn.conv1d.weight"
        self.assertTrue(mx.array_equal(converted[mlx_norm_key], base + 1.0))
        self.assertEqual(converted[mlx_conv_key].shape, (16, 4, 1))
        self.assertFalse(any("mtp." in key for key in converted))
        self.assertFalse(any("visual" in key for key in converted))

        native_model = agnes.Model(self.make_args(mtp_num_hidden_layers=1))
        loaded = native_model.sanitize(converted.copy())
        self.assertTrue(mx.array_equal(loaded[mlx_norm_key], base + 1.0))

    def test_bfloat16_gated_norm_preserves_reference_operation_order(self):
        norm = Qwen3NextRMSNormGated(8, 1e-6)
        norm.weight = (mx.arange(8) / 16 + 0.75).astype(mx.bfloat16)
        hidden = (mx.arange(16).reshape(1, 2, 8) / 13 - 0.4).astype(mx.bfloat16)
        gate = (mx.arange(16).reshape(1, 2, 8) / 17 - 0.3).astype(mx.bfloat16)
        actual = norm(hidden, gate)
        expected = (
            mx.fast.rms_norm(hidden, norm.weight, norm.eps).astype(mx.float32)
            * nn.silu(gate.astype(mx.float32))
        ).astype(mx.bfloat16)
        mx.eval(actual, expected)
        self.assertTrue(mx.array_equal(actual, expected))

    def test_native_vlm_weight_names_load_strictly(self):
        model = agnes.Model(self.make_args())
        native = dict(tree_flatten(model.parameters()))
        native["vision_tower.stub"] = mx.zeros((1,))

        reloaded = agnes.Model(self.make_args())
        sanitized = reloaded.sanitize(native)
        reloaded.load_weights(list(sanitized.items()), strict=True)
        mx.eval(reloaded.parameters())

    def test_invalid_layer_plan_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "layer_types"):
            agnes.Model(
                self.make_args(layer_types=["full_attention", agnes.LAYER_DELTA])
            )

    def test_pinned_public_config_parses_without_constructing_full_model(self):
        layer_types = [
            agnes.LAYER_GLOBAL if (i + 1) % 4 == 0 else agnes.LAYER_DELTA
            for i in range(72)
        ]
        args = agnes.ModelArgs.from_dict(
            {
                "model_type": "agnes",
                "text_config": {
                    "model_type": "agnes_text",
                    "hidden_size": 5120,
                    "num_hidden_layers": 72,
                    "intermediate_size": 17408,
                    "parallel_ffn_intermediate_size": 2048,
                    "vocab_size": 248320,
                    "layer_types": layer_types,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                },
            }
        )
        text = agnes.TextModelArgs.from_dict(args.text_config)
        self.assertEqual(args.model_type, "agnes")
        self.assertEqual(text.num_hidden_layers, 72)
        self.assertEqual(text.parallel_ffn_intermediate_size, 2048)
        self.assertEqual(text.layer_types.count(agnes.LAYER_GLOBAL), 18)
        self.assertEqual(text.layer_types.count(agnes.LAYER_DELTA), 54)


if __name__ == "__main__":
    unittest.main()
