"""qwen3_5_moe must load MLX checkpoints that ship separate
``switch_mlp.gate_proj`` / ``up_proj`` expert tables under the fused
gate-up lever (``MLX_QWEN4_MOE_FUSED_GATE_UP``, default on), the same way
qwen4_exp does. Before the fix every Qwen3.6-35B-A3B checkpoint failed
``load_weights(strict=True)`` with 240+ "parameters not in model"."""

import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import qwen3_5_moe, qwen3_next

TINY = {
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 256,
        "full_attention_interval": 2,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "attention_bias": False,
        "attn_output_gate": True,
        "tie_word_embeddings": False,
        "mtp_num_hidden_layers": 1,
    },
}


def _build(fused: bool):
    saved = qwen3_next._MOE_FUSED_GATE_UP
    qwen3_next._MOE_FUSED_GATE_UP = fused
    try:
        model = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(TINY))
    finally:
        qwen3_next._MOE_FUSED_GATE_UP = saved
    return model


class TestQwen35MoeSanitizeFusedGateUp(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        mx.random.seed(0)

    def _checkpoint_with_separate_tables(self):
        # An MLX checkpoint: separate gate/up, quantized triplets, 4-bit g32.
        model = _build(fused=False)
        nn.quantize(model, group_size=32, bits=4)
        weights = dict(tree_flatten(model.parameters()))
        self.assertTrue(
            any(k.endswith("switch_mlp.gate_proj.scales") for k in weights)
        )
        self.assertFalse(any("gate_up_proj" in k for k in weights))
        return model, weights

    def test_separate_tables_load_under_fused_lever(self):
        ref, weights = self._checkpoint_with_separate_tables()
        saved = qwen3_next._MOE_FUSED_GATE_UP
        qwen3_next._MOE_FUSED_GATE_UP = True
        try:
            fused = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(TINY))
            nn.quantize(fused, group_size=32, bits=4)
            sanitized = fused.sanitize(dict(weights))
            self.assertTrue(
                any(k.endswith("switch_mlp.gate_up_proj.weight") for k in sanitized)
            )
            self.assertFalse(
                any(k.endswith("switch_mlp.gate_proj.weight") for k in sanitized)
            )
            self.assertTrue(
                any(k.startswith("language_model.mtp.") and "gate_up_proj" in k
                    for k in sanitized),
                "the MTP layer's expert tables must be fused too",
            )
            fused.load_weights(list(sanitized.items()), strict=True)
        finally:
            qwen3_next._MOE_FUSED_GATE_UP = saved

        x = mx.array([[3, 17, 42, 7, 99]], dtype=mx.uint32)
        a = ref.language_model(x)
        b = fused.language_model(x)
        mx.eval(a, b)
        self.assertTrue(mx.allclose(a, b, atol=1e-4, rtol=1e-4).item())

    def test_separate_tables_still_load_with_lever_off(self):
        _, weights = self._checkpoint_with_separate_tables()
        saved = qwen3_next._MOE_FUSED_GATE_UP
        qwen3_next._MOE_FUSED_GATE_UP = False
        try:
            model = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(TINY))
            nn.quantize(model, group_size=32, bits=4)
            model.load_weights(list(model.sanitize(dict(weights)).items()), strict=True)
        finally:
            qwen3_next._MOE_FUSED_GATE_UP = saved


if __name__ == "__main__":
    unittest.main()
