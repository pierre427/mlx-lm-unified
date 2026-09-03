# Copyright © 2026 Apple Inc.

"""Load-path test: a qwen3_5_moe MLX checkpoint with per-table quantization
overrides (6-bit expert tables in one layer, 4-bit default) must load through
``load_model`` under the fused gate-up lever and produce the same logits as
the unfused load. Before the fix the fused table was built at the default
bits and load failed on shape (the default Qwen3.6-35B-A3B checkpoint)."""

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import qwen3_5_moe, qwen3_next
from mlx_lm.utils import _carry_fused_gate_up_overrides, load_model

from test_qwen3_5_moe_sanitize_fused_gate_up import TINY, _build


def _write_checkpoint(tmp: Path):
    """A separate-table checkpoint: layer 0 experts at 6-bit, rest 4-bit."""
    model = _build(fused=False)
    quant = {"group_size": 32, "bits": 4, "mode": "affine"}
    l0 = "language_model.model.layers.0.mlp.switch_mlp"
    for name in ("gate_proj", "up_proj", "down_proj"):
        quant[f"{l0}.{name}"] = {"group_size": 32, "bits": 6, "mode": "affine"}

    def predicate(path, module):
        if path in quant:
            return quant[path]
        return hasattr(module, "to_quantized")

    nn.quantize(model, group_size=32, bits=4, class_predicate=predicate)
    weights = dict(tree_flatten(model.parameters()))
    mx.eval(weights)
    mx.save_safetensors(str(tmp / "model.safetensors"), weights)
    cfg = json.loads(json.dumps(TINY))
    cfg["quantization"] = quant
    (tmp / "config.json").write_text(json.dumps(cfg))
    return model


class TestQwen35MoeLoadMixedPrecision(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        mx.random.seed(1)

    def test_helper_carries_and_refuses(self):
        weights = {"a.switch_mlp.gate_up_proj.weight": 0, "b.switch_mlp.gate_up_proj.weight": 0}
        cfg = {"quantization": {"bits": 4, "group_size": 64,
                                "a.switch_mlp.gate_proj": {"bits": 6, "group_size": 64},
                                "a.switch_mlp.up_proj": {"bits": 6, "group_size": 64}}}
        self.assertEqual(_carry_fused_gate_up_overrides(cfg, weights), 1)
        self.assertEqual(cfg["quantization"]["a.switch_mlp.gate_up_proj"]["bits"], 6)
        cfg["quantization"]["b.switch_mlp.gate_proj"] = {"bits": 6, "group_size": 64}
        cfg["quantization"]["b.switch_mlp.up_proj"] = {"bits": 4, "group_size": 64}
        with self.assertRaises(ValueError):
            _carry_fused_gate_up_overrides(cfg, weights)

    def test_load_model_fused_matches_unfused(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ref = _write_checkpoint(tmp)
            x = mx.array([[3, 17, 42, 7, 99]], dtype=mx.uint32)
            ref_logits = ref.language_model(x)

            saved = qwen3_next._MOE_FUSED_GATE_UP
            try:
                qwen3_next._MOE_FUSED_GATE_UP = True
                fused, _ = load_model(tmp)
                names = dict(tree_flatten(fused.parameters()))
                l0 = "language_model.model.layers.0.mlp.switch_mlp.gate_up_proj"
                self.assertIn(f"{l0}.weight", names)
                # 6-bit packs 64 rows into 12 uint32 per group of 32 cols... assert bits via module
                mod = fused.language_model.model.layers[0].mlp.switch_mlp.gate_up_proj
                self.assertEqual(mod.bits, 6)
                self.assertEqual(
                    fused.language_model.model.layers[1].mlp.switch_mlp.gate_up_proj.bits, 4
                )
                fused_logits = fused.language_model(x)

                qwen3_next._MOE_FUSED_GATE_UP = False
                plain, _ = load_model(tmp)
                plain_logits = plain.language_model(x)
            finally:
                qwen3_next._MOE_FUSED_GATE_UP = saved
            mx.eval(ref_logits, fused_logits, plain_logits)
            self.assertTrue(mx.allclose(plain_logits, ref_logits, atol=1e-4, rtol=1e-4).item())
            self.assertTrue(mx.allclose(fused_logits, ref_logits, atol=1e-4, rtol=1e-4).item())


if __name__ == "__main__":
    unittest.main()
