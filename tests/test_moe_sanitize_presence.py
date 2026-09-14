import importlib
import unittest
from types import SimpleNamespace

import mlx.core as mx


class TestMoeSanitizePresence(unittest.TestCase):
    def _check(
        self,
        module_name,
        *,
        stacked_component="switch_mlp",
        suffixes=("weight",),
    ):
        module = importlib.import_module(f"mlx_lm.models.{module_name}")
        model = SimpleNamespace(
            args=SimpleNamespace(
                num_hidden_layers=3,
                num_experts=2,
                tie_word_embeddings=False,
            ),
            mtp=None,
        )
        layer = 2
        projections = ("gate_proj", "up_proj", "down_proj")
        weights = {
            f"model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}": mx.zeros(
                (1, 1)
            )
            for expert in range(model.args.num_experts)
            for projection in projections
            for suffix in suffixes
        }

        sanitized = module.Model.sanitize(model, weights)

        for projection in projections:
            for suffix in suffixes:
                self.assertIn(
                    f"model.layers.{layer}.mlp.{stacked_component}.{projection}.{suffix}",
                    sanitized,
                )
        self.assertFalse(any(".experts.0." in key for key in sanitized))
        return sanitized

    def test_qwen3_moe_nonzero_layer(self):
        self._check("qwen3_moe")

    def test_qwen3_next_nonzero_layer(self):
        self._check("qwen3_next")

    def test_qwen3_next_keeps_supported_mtp_weights(self):
        from mlx_lm.models import qwen3_next

        model = SimpleNamespace(
            args=SimpleNamespace(
                num_hidden_layers=3,
                num_experts=2,
                tie_word_embeddings=False,
            ),
            mtp=object(),
        )
        weights = {
            "mtp.fc.weight": mx.zeros((1, 1)),
            **{
                f"model.layers.2.mlp.experts.{expert}.{projection}.weight": mx.zeros(
                    (1, 1)
                )
                for expert in range(model.args.num_experts)
                for projection in ("gate_proj", "up_proj", "down_proj")
            },
        }

        sanitized = qwen3_next.Model.sanitize(model, weights)

        self.assertIn("mtp.fc.weight", sanitized)
        self.assertIsNotNone(model.mtp)

    def test_qwen3_next_drops_unsupported_mtp_weights(self):
        from mlx_lm.models import qwen3_next

        model = SimpleNamespace(
            args=SimpleNamespace(
                num_hidden_layers=3,
                num_experts=2,
                tie_word_embeddings=False,
            ),
            mtp=None,
        )
        weights = {
            "mtp.fc.weight": mx.zeros((1, 1)),
            **{
                f"model.layers.2.mlp.experts.{expert}.{projection}.weight": mx.zeros(
                    (1, 1)
                )
                for expert in range(model.args.num_experts)
                for projection in ("gate_proj", "up_proj", "down_proj")
            },
        }

        sanitized = qwen3_next.Model.sanitize(model, weights)

        self.assertNotIn("mtp.fc.weight", sanitized)
        self.assertIsNone(model.mtp)

    def test_klear_nonzero_layer(self):
        self._check("Klear", stacked_component="experts")

    def test_hunyuan_nonzero_layer(self):
        self._check("hunyuan", suffixes=("weight", "scales", "biases"))

    def test_olmoe_nonzero_layer(self):
        self._check("olmoe", suffixes=("weight", "scales", "biases"))

    def test_mellum_nonzero_layer(self):
        self._check("mellum")

    def test_qwen2_moe_nonzero_layer(self):
        self._check("qwen2_moe", suffixes=("weight", "scales", "biases"))


if __name__ == "__main__":
    unittest.main()
