import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import agnes
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated


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
        self.assertIsInstance(
            model.layers[1].global_attn, agnes.Qwen3NextAttention
        )
        self.assertIsNotNone(model.layers[0].mlp.parallel_ffn)

        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        full = model(tokens)
        cache = model.make_cache()
        steps = [model(tokens[:, i : i + 1], cache=cache) for i in range(4)]
        decoded = mx.concatenate(steps, axis=1)
        mx.eval(full, decoded)
        self.assertTrue(mx.allclose(full, decoded, rtol=2e-4, atol=2e-4))

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
        hidden = (mx.arange(16).reshape(1, 2, 8) / 13 - 0.4).astype(
            mx.bfloat16
        )
        gate = (mx.arange(16).reshape(1, 2, 8) / 17 - 0.3).astype(
            mx.bfloat16
        )
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
