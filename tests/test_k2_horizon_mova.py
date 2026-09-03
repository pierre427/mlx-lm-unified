# Copyright © 2026 mlx-uag lab
#
# CPU-only gates for the native K2-Horizon-MoVA port. The parity tests build
# the same tiny model in the native module and in the checkpoint's vendored
# reference file and assert bit-identical logits.

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import k2_horizon_mova, switch_layers
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.utils import _get_classes, load_model

mx.set_default_device(mx.cpu)

CHECKPOINT = Path(
    os.environ.get(
        "K2_HORIZON_MOVA_DIR",
        "/Users/pierrelamy/mlx-models/K2-Horizon-MoVA-36B-A4B-MLX-4bit",
    )
)
REFERENCE_FILE = CHECKPOINT / "k2_horizon_mova_mlx.py"
REAL_CONFIG = CHECKPOINT / "config.json"

TINY_CONFIG = {
    "model_type": "k2_horizon_mova",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "moe_intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rope_head_dim": 16,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
    "mova_num_experts": 4,
    "mova_num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [0],
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "max_position_embeddings": 4096,
    "norm_topk_prob": True,
    "router_score_func": "sigmoid",
    "router_scaling_factor": 2.5,
    "tie_word_embeddings": False,
    "layernorm_num_groups": 2,
    "rope_theta": 10_000_000.0,
    "rope_parameters": {"rope_theta": 10_000_000.0, "rope_type": "default"},
    "attention_bias": False,
    "moe_gate_bias": True,
    "attention_gate_func": "softplus",
}


def _load_reference():
    spec = importlib.util.spec_from_file_location("k2_reference", REFERENCE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_weights(model, seed=0):
    mx.random.seed(seed)
    out = []
    for name, p in tree_flatten(model.parameters()):
        if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
            w = 1.0 + 0.1 * mx.random.normal(p.shape)
        elif name.endswith(".bias"):
            w = 0.5 * mx.random.normal(p.shape)
        else:
            w = 0.2 * mx.random.normal(p.shape)
        out.append((name, w.astype(mx.float32)))
    return out


def _build_native(config=TINY_CONFIG):
    args = k2_horizon_mova.ModelArgs.from_dict(config)
    model = k2_horizon_mova.Model(args)
    model.load_weights(_random_weights(model), strict=True)
    mx.eval(model.parameters())
    return model


class TestK2HorizonMoVAParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REFERENCE_FILE.exists():
            raise unittest.SkipTest(f"reference file missing: {REFERENCE_FILE}")
        cls.reference = _load_reference()

    def _build_pair(self):
        native = _build_native()
        ref_args = self.reference.ModelArgs.from_dict(TINY_CONFIG)
        ref = self.reference.Model(ref_args)
        native_names = sorted(k for k, _ in tree_flatten(native.parameters()))
        ref_names = sorted(k for k, _ in tree_flatten(ref.parameters()))
        # Same parameter tree: no name mapping between port and reference.
        self.assertEqual(native_names, ref_names)
        ref.load_weights(tree_flatten(native.parameters()), strict=True)
        mx.eval(ref.parameters())
        return native, ref

    def _assert_same(self, a, b, what):
        mx.eval(a, b)
        self.assertEqual(a.shape, b.shape, what)
        self.assertEqual(a.dtype, b.dtype, what)
        self.assertTrue(mx.all(mx.isfinite(a)).item(), f"{what}: non-finite")
        self.assertTrue(mx.array_equal(a, b).item(), f"{what}: logits differ")

    def test_structure(self):
        native = _build_native()
        layers = native.layers
        self.assertEqual(len(layers), 4)
        self.assertIsInstance(layers[0].mlp, k2_horizon_mova.MLP)
        self.assertFalse(layers[0].self_attn.mova)
        self.assertTrue("v_proj" in layers[0].self_attn)
        for layer in layers[1:]:
            self.assertIsInstance(layer.mlp, k2_horizon_mova.SparseMoE)
            self.assertTrue(layer.self_attn.mova)
            self.assertFalse("v_proj" in layer.self_attn)
            self.assertEqual(layer.self_attn.v_experts.weight.shape, (4, 32, 64))
        cache = make_prompt_cache(native)
        self.assertEqual(len(cache), 4)
        self.assertTrue(all(type(c) is KVCache for c in cache))

    def test_bit_identical_prefill_decode_chunk_and_trim(self):
        native, ref = self._build_pair()
        mx.random.seed(1)
        prompt = mx.random.randint(0, TINY_CONFIG["vocab_size"], (1, 17))
        tail = mx.random.randint(0, TINY_CONFIG["vocab_size"], (1, 4))

        # No-cache forward first.
        self._assert_same(native(prompt), ref(prompt), "no-cache prefill")

        nc, rc = make_prompt_cache(native), make_prompt_cache(ref)

        # (a) 17-token prefill.
        self._assert_same(native(prompt, cache=nc), ref(prompt, cache=rc), "prefill")
        self.assertEqual([c.offset for c in nc], [17] * 4)
        self.assertEqual([c.offset for c in nc], [c.offset for c in rc])

        # (b) 1-token decode step.
        step = tail[:, :1]
        self._assert_same(native(step, cache=nc), ref(step, cache=rc), "decode")
        self.assertEqual([c.offset for c in nc], [18] * 4)

        # (c) 3-token chunk (prompt-lookup verify shape).
        chunk = tail[:, 1:4]
        self._assert_same(native(chunk, cache=nc), ref(chunk, cache=rc), "chunk")
        self.assertEqual([c.offset for c in nc], [21] * 4)
        self.assertEqual([c.offset for c in nc], [c.offset for c in rc])

        # Trim (speculative rollback) and re-verify the same chunk.
        for c in nc + rc:
            self.assertTrue(c.is_trimmable())
            self.assertEqual(c.trim(2), 2)
        self.assertEqual([c.offset for c in nc], [19] * 4)
        redo = tail[:, 2:4]
        self._assert_same(native(redo, cache=nc), ref(redo, cache=rc), "post-trim")
        self.assertEqual([c.offset for c in nc], [21] * 4)

    def test_sorted_and_unsorted_value_gather_agree(self):
        # 17 tokens x top-2 = 34 assignments takes the sorted gather; force the
        # unsorted path and check both are bit-identical on the same weights.
        native = _build_native()
        mx.random.seed(2)
        prompt = mx.random.randint(0, TINY_CONFIG["vocab_size"], (1, 17))
        attn = native.layers[1].self_attn
        h = native.model.layers[0].input_layernorm(native.model.embed_tokens(prompt))
        sorted_values = attn._mova_values(h)
        threshold = switch_layers._GATHER_SORT_MIN_ASSIGNMENTS
        self.assertGreaterEqual(17 * 2, threshold)
        self.assertLess(3 * 2, threshold)
        try:
            switch_layers._GATHER_SORT_MIN_ASSIGNMENTS = 1 << 30
            unsorted_values = attn._mova_values(h)
        finally:
            switch_layers._GATHER_SORT_MIN_ASSIGNMENTS = threshold
        self._assert_same(sorted_values, unsorted_values, "sorted vs unsorted gather")

    def test_batch_and_dtypes(self):
        native = _build_native()
        inputs = mx.array([[0, 1, 2], [3, 4, 5]])
        out = native(inputs)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 3, TINY_CONFIG["vocab_size"]))
        copy.deepcopy(native)

        # The CPU gather_mm kernel is float32-only, so check half-precision
        # dtype propagation (grouped norm, softplus gate, rope) on the dense
        # layers alone; the routed paths run in fp16 on Metal.
        dense = _build_native(dict(TINY_CONFIG, mlp_only_layers=[0, 1, 2, 3]))
        self.assertTrue(all(not l.self_attn.mova for l in dense.layers))
        for dtype in (mx.float16, mx.bfloat16):
            dense.set_dtype(dtype)
            out_half = dense(inputs)
            mx.eval(out_half)
            self.assertEqual(out_half.dtype, dtype)
            self.assertTrue(mx.all(mx.isfinite(out_half)).item())


class TestK2HorizonMoVAQuantized(unittest.TestCase):
    def test_quantized_forward_with_router_overrides(self):
        native = _build_native()
        reference_out = native(mx.array([[1, 2, 3, 4, 5]]))
        mx.eval(reference_out)

        def class_predicate(path, module):
            if path.endswith("mlp.gate") or path.endswith("self_attn.v_router"):
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return hasattr(module, "to_quantized")

        nn.quantize(
            native,
            group_size=64,
            bits=4,
            mode="affine",
            class_predicate=class_predicate,
        )

        sparse = native.layers[1]
        self.assertIsInstance(sparse.self_attn.v_experts, QuantizedSwitchLinear)
        self.assertEqual(sparse.self_attn.v_experts.bits, 4)
        self.assertEqual(sparse.mlp.switch_mlp.up_proj.bits, 4)
        self.assertIsInstance(sparse.self_attn.v_router, nn.QuantizedLinear)
        self.assertEqual(sparse.self_attn.v_router.bits, 8)
        self.assertIsInstance(sparse.mlp.gate, nn.QuantizedLinear)
        self.assertEqual(sparse.mlp.gate.bits, 8)
        self.assertTrue("bias" in sparse.mlp.gate)
        self.assertIsInstance(native.lm_head, nn.QuantizedLinear)
        self.assertIsInstance(native.model.embed_tokens, nn.QuantizedEmbedding)
        self.assertEqual(native.layers[0].self_attn.v_proj.bits, 4)

        # The quantized value experts run through gather_qmm with the right shape.
        x = mx.random.normal((5, 1, 1, 64))
        idx = mx.array([[0, 1], [2, 3], [1, 2], [3, 0], [0, 2]])
        y = sparse.self_attn.v_experts(x, idx)
        mx.eval(y)
        self.assertEqual(y.shape, (5, 2, 1, 32))

        out = native(mx.array([[1, 2, 3, 4, 5]]))
        mx.eval(out)
        self.assertEqual(out.shape, reference_out.shape)
        self.assertTrue(mx.all(mx.isfinite(out)).item())

        # The model's own predicate selects the same 8-bit routers.
        fresh = _build_native()
        pred = fresh.quant_predicate
        self.assertEqual(
            pred("model.layers.1.mlp.gate", None), {"group_size": 64, "bits": 8}
        )
        self.assertEqual(
            pred("model.layers.1.self_attn.v_router", None),
            {"group_size": 64, "bits": 8},
        )
        self.assertIs(pred("model.layers.1.self_attn.q_proj", None), True)


class TestK2HorizonMoVALoader(unittest.TestCase):
    @unittest.skipUnless(REAL_CONFIG.exists(), "real config.json not available")
    def test_real_config_resolves_and_constructs(self):
        with open(REAL_CONFIG) as f:
            config = json.load(f)
        model_class, args_class = _get_classes(config)
        self.assertIs(model_class, k2_horizon_mova.Model)
        self.assertIs(args_class, k2_horizon_mova.ModelArgs)

        args = args_class.from_dict(config)
        self.assertEqual(args.num_hidden_layers, 48)
        self.assertEqual(args.hidden_size, 2560)
        self.assertEqual(args.head_dim, 128)
        self.assertEqual(args.rope_theta, 10_000_000.0)
        self.assertIsNone(args.rope_scaling)
        self.assertEqual(args.mlp_only_layers, [0, 1, 2])
        self.assertEqual(args.layernorm_num_groups, 2)
        self.assertEqual(args.router_scaling_factor, 2.5)
        self.assertEqual(args.attention_gate_func, "softplus")
        self.assertEqual(args.vocab_size, 250624)

        # Construct lazily with 4 layers (one MoE/MoVA layer) and every other
        # real field; nothing is evaluated so no 36B-parameter buffers exist.
        small = dict(config, num_hidden_layers=4)
        model = model_class(args_class.from_dict(small))
        self.assertEqual(len(model.layers), 4)
        for i in range(3):
            self.assertIsInstance(model.layers[i].mlp, k2_horizon_mova.MLP)
            self.assertFalse(model.layers[i].self_attn.mova)
        self.assertIsInstance(model.layers[3].mlp, k2_horizon_mova.SparseMoE)
        self.assertTrue(model.layers[3].self_attn.mova)
        self.assertEqual(
            model.layers[3].self_attn.v_experts.weight.shape, (64, 1024, 2560)
        )
        self.assertEqual(
            model.layers[3].mlp.switch_mlp.up_proj.weight.shape, (100, 768, 2560)
        )
        self.assertEqual(model.layers[3].mlp.gate.weight.shape, (100, 2560))
        self.assertTrue("bias" in model.layers[3].mlp.gate)
        self.assertEqual(model.layers[3].self_attn.gate_proj.weight.shape, (4096, 2560))
        self.assertEqual(model.lm_head.weight.shape, (250624, 2560))
        del model

        # Every quantization override path in config.json names a module.
        quant = config["quantization"]
        overrides = [k for k in quant if k.startswith("model.layers.")]
        self.assertEqual(len(overrides), 2 * 45)
        for path in overrides:
            self.assertTrue(
                path.endswith("mlp.gate") or path.endswith("self_attn.v_router")
            )
            self.assertEqual(quant[path]["bits"], 8)

    def test_native_module_preferred_over_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with open(path / "config.json", "w") as f:
                json.dump(dict(TINY_CONFIG, model_file="k2_horizon_mova_mlx.py"), f)
            model, config = load_model(path, strict=False)
            self.assertIsInstance(model, k2_horizon_mova.Model)
            self.assertEqual(config["model_file"], "k2_horizon_mova_mlx.py")

            with open(path / "config.json", "w") as f:
                json.dump(
                    dict(
                        TINY_CONFIG,
                        model_type="no_such_model_xyz",
                        model_file="k2_horizon_mova_mlx.py",
                    ),
                    f,
                )
            with self.assertRaises(ValueError) as ctx:
                load_model(path, strict=False)
            self.assertIn("trust_remote_code", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
