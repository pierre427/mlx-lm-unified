# Copyright © 2026 mlx-uag lab
#
# CPU-only gates for the native Ling-3.0-flash (bailing_hybrid) port. The
# parity tests build the same tiny model in the native module and in the
# checkpoint's vendored reference file and assert bit-identical logits.

import copy
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import bailing_hybrid, mla
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
from mlx_lm.models.mla import QuantizedMultiLinear
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.utils import _get_classes, load_model

_DEFAULT_DEVICE = None


def setUpModule():
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    mx.clear_cache()
    mx.set_default_device(_DEFAULT_DEVICE)

CHECKPOINT = Path(
    os.environ.get(
        "LING_3_FLASH_DIR", "/Users/pierrelamy/mlx-models/Ling-3.0-flash-oQ4"
    )
)
REFERENCE_FILE = CHECKPOINT / "bailing_hybrid.py"
REAL_CONFIG = CHECKPOINT / "config.json"

# 4 layers, layer_group_size 2, first_k_dense_replace 1:
#   0 KDA + dense MLP, 1 MLA + MoE, 2 KDA + MoE, 3 MLA + MoE.
# kda_lower_bound is None here: the vendored reference ignores the bound, so
# this is the arm where bit identity with it is defined.
TINY_CONFIG = {
    "model_type": "bailing_hybrid",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "moe_shared_expert_intermediate_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
    "n_group": 2,
    "topk_group": 1,
    "first_k_dense_replace": 1,
    "layer_group_size": 2,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000.0,
    "rope_interleave": True,
    "routed_scaling_factor": 2.5,
    "head_dim": 16,
    "kv_lora_rank": 32,
    "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 16,
    "v_head_dim": 16,
    "short_conv_kernel_size": 4,
    "no_kda_lora": True,
    "kda_safe_gate": True,
    "kda_lower_bound": None,
    "gated_attention_proj_granularity_type": "head_wise",
    "tie_word_embeddings": False,
}
BOUNDED_CONFIG = dict(TINY_CONFIG, kda_lower_bound=-5.0)
# Every quantizable input width a multiple of 32 (affine group sizes are
# 32/64/128): widen the MLA latent pair for the quantized forward.
QUANT_CONFIG = dict(
    TINY_CONFIG,
    qk_nope_head_dim=32,
    kv_lora_rank=64,
    moe_intermediate_size=64,
    moe_shared_expert_intermediate_size=64,
)


def _load_reference():
    spec = importlib.util.spec_from_file_location("bailing_reference", REFERENCE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_weights(model, seed=0):
    mx.random.seed(seed)
    out = []
    for name, p in tree_flatten(model.parameters()):
        if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
            w = 1.0 + 0.1 * mx.random.normal(p.shape)
        elif name.endswith("A_log"):
            w = mx.log(mx.random.uniform(low=1.0, high=16.0, shape=p.shape))
        elif name.endswith("dt_bias") or name.endswith("expert_bias"):
            w = 0.5 * mx.random.normal(p.shape)
        else:
            w = 0.2 * mx.random.normal(p.shape)
        out.append((name, w.astype(mx.float32)))
    return out


def _build_native(config=TINY_CONFIG, seed=0):
    args = bailing_hybrid.ModelArgs.from_dict(config)
    model = bailing_hybrid.Model(args)
    model.load_weights(_random_weights(model, seed), strict=True)
    mx.eval(model.parameters())
    return model


def _kv_offsets(caches):
    return [c.offset for c in caches if isinstance(c, KVCache)]


def _cache_state(c):
    if isinstance(c, KVCache):
        return [c.keys[..., : c.offset, :], c.values[..., : c.offset, :]]
    return list(c.cache)


def _run_history(model, cache, pieces):
    for piece in pieces:
        mx.eval(model(piece, cache=cache))
    return cache


class TestBailingHybridParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REFERENCE_FILE.exists():
            raise unittest.SkipTest(f"reference file missing: {REFERENCE_FILE}")
        cls.reference = _load_reference()

    def setUp(self):
        # The reference takes the absorbed MLA branch only at L == 1; pin the
        # tree's gate to the same dispatch so the comparison is op for op.
        mla.set_absorbed_max_query_override(1)
        bailing_hybrid.set_kda_gate_mode("config")

    def tearDown(self):
        mla.set_absorbed_max_query_override(None)
        bailing_hybrid.set_kda_gate_mode(None)

    def _build_pair(self, config=TINY_CONFIG):
        native = _build_native(config)
        ref_args = self.reference.ModelArgs.from_dict(config)
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
        self.assertEqual([l.is_global for l in layers], [False, True, False, True])
        self.assertIsInstance(layers[0].attention, bailing_hybrid.KimiDeltaAttention)
        self.assertIsInstance(layers[1].attention, bailing_hybrid.MultiLatentAttention)
        self.assertIsInstance(layers[0].mlp, bailing_hybrid.MLP)
        for layer in layers[1:]:
            self.assertIsInstance(layer.mlp, bailing_hybrid.SparseMoeBlock)
            self.assertEqual(layer.mlp.switch_mlp.up_proj.weight.shape, (8, 32, 64))
            self.assertEqual(layer.mlp.gate.gate_proj.weight.shape, (8, 64))
            self.assertEqual(layer.mlp.gate.expert_bias.dtype, mx.float32)
        kda = layers[0].attention
        self.assertEqual(kda.q_conv1d.conv.weight.shape, (64, 4, 1))
        self.assertEqual(kda.f_proj.weight.shape, (64, 64))
        self.assertEqual(kda.b_proj.weight.shape, (4, 64))
        self.assertEqual(kda.A_log.shape, (4,))
        self.assertEqual(kda.dt_bias.shape, (64,))
        attn = layers[1].attention
        self.assertEqual(attn.q_proj.weight.shape, (4 * 32, 64))
        self.assertEqual(attn.kv_a_proj_with_mqa.weight.shape, (32 + 16, 64))
        self.assertEqual(attn.embed_q.weight.shape, (4, 32, 16))
        self.assertEqual(attn.unembed_out.weight.shape, (4, 16, 32))
        self.assertEqual(attn.g_proj.weight.shape, (4, 64))
        self.assertTrue(attn.rope.traditional)
        cache = make_prompt_cache(native)
        self.assertEqual(
            [type(c) for c in cache], [ArraysCache, KVCache, ArraysCache, KVCache]
        )
        self.assertEqual(len(cache[0].cache), 4)

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
        self.assertEqual(_kv_offsets(nc), [17, 17])
        self.assertEqual(_kv_offsets(nc), _kv_offsets(rc))
        for c in nc:
            if isinstance(c, ArraysCache):
                self.assertEqual(c[3].shape, (1, 4, 16, 16))
                self.assertEqual(c[0].shape, (1, 3, 64))

        # (b) 1-token decode step.
        step = tail[:, :1]
        self._assert_same(native(step, cache=nc), ref(step, cache=rc), "decode")
        self.assertEqual(_kv_offsets(nc), [18, 18])

        # Recurrent caches are trimmable only while speculating (the generate
        # loops call start_speculation after prefill); the KDA layer then
        # records an exact rollback per forward.
        for c in nc:
            self.assertEqual(c.is_trimmable(), isinstance(c, KVCache))
        for c in nc:
            c.start_speculation()

        # (c) 3-token chunk (prompt-lookup verify shape).
        chunk = tail[:, 1:4]
        self._assert_same(native(chunk, cache=nc), ref(chunk, cache=rc), "chunk")
        self.assertEqual(_kv_offsets(nc), [21, 21])
        self.assertEqual(_kv_offsets(nc), _kv_offsets(rc))

        # (d) Trim (speculative rollback): KV caches move their offset; the
        # ArraysCache replays the recurrence over the kept prefix of the
        # chunk. It keeps no token counter, so only the KV layers report one.
        for c in nc:
            self.assertTrue(c.is_trimmable())
            self.assertEqual(c.trim(2), 2)
        self.assertEqual(_kv_offsets(nc), [19, 19])
        trimmed = [_cache_state(c) for c in nc]

        # The rewind depends only on the kept token: a chunk with the same
        # first token and a different rest rewinds to a bit-identical state.
        nc_alt = _run_history(native, make_prompt_cache(native), [prompt, tail[:, :1]])
        for c in nc_alt:
            c.start_speculation()
        other = mx.concatenate([tail[:, 1:2], (tail[:, 2:4] + 7) % 128], axis=1)
        mx.eval(native(other, cache=nc_alt))
        for c in nc_alt:
            self.assertEqual(c.trim(2), 2)
        for a, b in zip(trimmed, (_cache_state(c) for c in nc_alt)):
            for x, y in zip(a, b):
                self._assert_same(x, y, "rewound state")

        # (e) Re-verify after the trim. The reference cannot trim its
        # ArraysCache, so its history is rebuilt one token at a time. CPU BLAS
        # is not bit-stable across M (token 18's projections come from the
        # M=3 chunk on the trimmed side and an M=1 step on the reference; the
        # tree's 1-token-step history is bit-identical to the reference), so
        # this comparison is a tolerance on the same function, not a bit gate.
        rc = _run_history(ref, make_prompt_cache(ref), [prompt, tail[:, :1], tail[:, 1:2]])
        nc_steps = _run_history(
            native, make_prompt_cache(native), [prompt, tail[:, :1], tail[:, 1:2]]
        )
        self.assertEqual(_kv_offsets(rc), [19, 19])
        redo = tail[:, 2:4]
        ref_redo = ref(redo, cache=rc)
        self._assert_same(native(redo, cache=nc_steps), ref_redo, "1-token history")
        trim_redo = native(redo, cache=nc)
        mx.eval(trim_redo, ref_redo)
        self.assertEqual(_kv_offsets(nc), [21, 21])
        self.assertTrue(mx.allclose(trim_redo, ref_redo, atol=1e-4, rtol=1e-4).item())
        self.assertTrue(mx.array_equal(trim_redo.argmax(-1), ref_redo.argmax(-1)).item())

        # (f) A full rollback restores the pre-chunk snapshot: trimming the
        # rest of the chunk and re-decoding its first token matches the
        # reference's own 1-token decode bit for bit (both sides at M=1).
        for c in nc:
            self.assertEqual(c.trim(3), 3)
        self.assertEqual(_kv_offsets(nc), [18, 18])
        rc = _run_history(ref, make_prompt_cache(ref), [prompt, tail[:, :1]])
        self._assert_same(
            native(tail[:, 1:2], cache=nc), ref(tail[:, 1:2], cache=rc), "full rollback"
        )
        for c in nc:
            c.stop_speculation()
            self.assertEqual(c.is_trimmable(), isinstance(c, KVCache))

    def test_absorbed_gate_agrees_with_expanded_on_chunk(self):
        # With the tree's gate, a 3-token verify against a warm cache takes the
        # absorbed branch (the reference expands). The two are the same
        # function; pin closeness and the mechanism.
        mla.set_absorbed_max_query_override(None)
        native = _build_native()
        geometry = native.layers[1].attention.absorbed_geometry
        self.assertEqual(geometry, (32, 16, 16))
        self.assertTrue(mla.use_absorbed_path(3, 21, geometry))
        self.assertFalse(mla.use_absorbed_path(17, 17, geometry))
        mx.random.seed(1)
        prompt = mx.random.randint(0, TINY_CONFIG["vocab_size"], (1, 18))
        chunk = mx.random.randint(0, TINY_CONFIG["vocab_size"], (1, 3))

        def run(override):
            mla.set_absorbed_max_query_override(override)
            try:
                cache = make_prompt_cache(native)
                mx.eval(native(prompt, cache=cache))
                out = native(chunk, cache=cache)
                mx.eval(out)
                return out
            finally:
                mla.set_absorbed_max_query_override(None)

        absorbed, expanded = run(None), run(1)
        self.assertTrue(mx.allclose(absorbed, expanded, atol=1e-5, rtol=1e-5).item())
        self.assertTrue(
            mx.array_equal(absorbed.argmax(-1), expanded.argmax(-1)).item()
        )

    def test_kda_lower_bound_is_plumbed(self):
        # Instrument the recurrence entry point: the config's bound must reach
        # gated_delta_update on every KDA call, and the gate mode must be able
        # to force the reference's plain gate.
        seen = []
        original = bailing_hybrid.gated_delta_update

        def recording(*args, **kwargs):
            seen.append(kwargs.get("lower_bound"))
            return original(*args, **kwargs)

        bailing_hybrid.gated_delta_update = recording
        try:
            plain = _build_native(TINY_CONFIG)
            bounded = _build_native(BOUNDED_CONFIG)
            x = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])

            seen.clear()
            out_plain = plain(x)
            mx.eval(out_plain)
            self.assertEqual(seen, [None, None])

            seen.clear()
            out_bounded = bounded(x)
            mx.eval(out_bounded)
            self.assertEqual(seen, [-5.0, -5.0])
            self.assertTrue(mx.all(mx.isfinite(out_bounded)).item())
            self.assertFalse(mx.array_equal(out_plain, out_bounded).item())

            # Same weights, plain-gate mode: identical to the unbounded model.
            bailing_hybrid.set_kda_gate_mode("plain")
            seen.clear()
            out_forced = bounded(x)
            mx.eval(out_forced)
            self.assertEqual(seen, [None, None])
            self.assertTrue(mx.array_equal(out_plain, out_forced).item())
        finally:
            bailing_hybrid.gated_delta_update = original
            bailing_hybrid.set_kda_gate_mode(None)

        with self.assertRaises(ValueError):
            bailing_hybrid.set_kda_gate_mode("bogus")

    def test_bounded_gate_matches_reference_formula_in_ops(self):
        # exp(lower_bound * sigmoid(exp(A_log) * (a + dt_bias))) is the decay
        # gated_delta.compute_lower_bound_g applies (tests/test_models.py pins
        # it); check the KDA layer feeds it the same operands as the plain gate.
        from mlx_lm.models import gated_delta

        bounded = _build_native(BOUNDED_CONFIG)
        kda = bounded.layers[0].attention
        A_log = kda.A_log.reshape(4, 1)
        dt_bias = kda.dt_bias.reshape(4, 16)
        a = mx.random.normal((1, 5, 4, 16))
        want = mx.exp(-5.0 * mx.sigmoid(mx.exp(A_log) * (a + dt_bias)))
        got = gated_delta.compute_lower_bound_g(A_log, a, dt_bias, -5.0)
        mx.eval(want, got)
        self.assertTrue(mx.allclose(want, got, atol=1e-6).item())
        # sigmoid saturates to exactly 1.0 / 0.0 in fp32, so the bounds close.
        self.assertTrue(mx.all(got >= math.exp(-5.0) - 1e-7).item())
        self.assertTrue(mx.all(got <= 1.0).item())

    def test_batch_and_dtypes(self):
        native = _build_native()
        inputs = mx.array([[0, 1, 2], [3, 4, 5]])
        out = native(inputs)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 3, TINY_CONFIG["vocab_size"]))
        copy.deepcopy(native)

        # The CPU gather_mm kernel is float32-only, so check half-precision
        # dtype propagation (KDA conv/recurrence, MLA gate, rope) on a dense
        # config alone; the routed paths run in half precision on Metal.
        dense = _build_native(dict(TINY_CONFIG, first_k_dense_replace=4))
        self.assertTrue(all(isinstance(l.mlp, bailing_hybrid.MLP) for l in dense.layers))
        for dtype in (mx.float16, mx.bfloat16):
            dense.set_dtype(dtype)
            cache = make_prompt_cache(dense)
            out_half = dense(inputs, cache=cache)
            mx.eval(out_half)
            self.assertEqual(out_half.dtype, dtype)
            self.assertTrue(mx.all(mx.isfinite(out_half)).item())
            step = dense(inputs[:, :1], cache=cache)
            mx.eval(step)
            self.assertEqual(step.dtype, dtype)

    def test_sanitize_hf_names(self):
        # HF-style tensors (per-expert weights, gate.weight, fused kv_b_proj,
        # (C, 1, K) conv weights, bf16 gate parameters) load into the same
        # model and give the same logits as the MLX-named tree.
        native = _build_native()
        args = native.args
        hf = {}
        for name, value in tree_flatten(native.parameters()):
            layer = native.layers[int(name.split(".")[2])] if ".layers." in name else None
            if ".mlp.switch_mlp." in name:
                proj = name.split(".")[-2]
                prefix = name.split(".mlp.")[0]
                for e in range(args.num_experts):
                    hf[f"{prefix}.mlp.experts.{e}.{proj}.weight"] = value[e]
            elif name.endswith(".mlp.gate.gate_proj.weight"):
                hf[name.replace(".gate_proj.weight", ".weight")] = value
            elif "_conv1d.conv.weight" in name:
                hf[name.replace(".conv.weight", ".weight")] = value.moveaxis(1, 2)
            elif name.endswith(".attention.A_log") or name.endswith(".attention.dt_bias"):
                hf[name] = value.astype(mx.bfloat16)
            elif name.endswith(".attention.embed_q.weight"):
                attn = layer.attention
                wk = value.swapaxes(-1, -2)  # (H, nope, r)
                wv = native.model.layers[int(name.split(".")[2])].attention.unembed_out.weight
                fused = mx.concatenate([wk, wv], axis=1)  # (H, nope + v, r)
                hf[name.replace("embed_q.weight", "kv_b_proj.weight")] = fused.reshape(
                    -1, attn.kv_lora_rank
                )
            elif name.endswith(".attention.unembed_out.weight"):
                continue
            else:
                hf[name] = value
        hf["model.layers.99.attention.q_proj.weight"] = mx.zeros((1, 1))  # MTP junk
        loaded = bailing_hybrid.Model(args)
        loaded.load_weights(list(loaded.sanitize(hf).items()), strict=True)
        mx.eval(loaded.parameters())
        self.assertEqual(loaded.layers[0].attention.A_log.dtype, mx.float32)
        self.assertEqual(loaded.layers[0].attention.dt_bias.dtype, mx.float32)
        x = mx.array([[7, 8, 9, 10, 11]])
        a, b = native(x), loaded(x)
        mx.eval(a, b)
        # bf16 round trip of A_log/dt_bias moves the KDA gate; compare the
        # wiring with the exact fp32 values restored.
        for i, l in enumerate(loaded.layers):
            if not l.is_global:
                l.attention.A_log = native.layers[i].attention.A_log
                l.attention.dt_bias = native.layers[i].attention.dt_bias
        b = loaded(x)
        mx.eval(b)
        self.assertTrue(mx.array_equal(a, b).item())


class TestBailingHybridQuantized(unittest.TestCase):
    def test_quantized_forward_with_checkpoint_overrides(self):
        native = _build_native(QUANT_CONFIG)
        reference_out = native(mx.array([[1, 2, 3, 4, 5]]))
        mx.eval(reference_out)

        # The oQ4 checkpoint's shape: a 4-bit base, 8-bit routers / lm_head /
        # embeddings / MLA latent pair, wider-group 8-bit shared experts.
        def class_predicate(path, module):
            if path.endswith("mlp.gate.gate_proj") or path in (
                "lm_head",
                "model.word_embeddings",
            ):
                return {"group_size": 32, "bits": 8, "mode": "affine"}
            if path.endswith(("attention.embed_q", "attention.unembed_out")):
                return {"group_size": 32, "bits": 8, "mode": "affine"}
            if ".mlp.shared_experts." in path:
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return hasattr(module, "to_quantized")

        nn.quantize(
            native, group_size=32, bits=4, mode="affine", class_predicate=class_predicate
        )

        moe = native.layers[2]
        self.assertIsInstance(moe.mlp.switch_mlp.up_proj, QuantizedSwitchLinear)
        self.assertEqual(moe.mlp.switch_mlp.up_proj.bits, 4)
        self.assertIsInstance(moe.mlp.gate.gate_proj, nn.QuantizedLinear)
        self.assertEqual(moe.mlp.gate.gate_proj.bits, 8)
        self.assertEqual(moe.mlp.shared_experts.up_proj.bits, 8)
        self.assertEqual(moe.mlp.shared_experts.up_proj.group_size, 64)
        self.assertIsInstance(moe.attention.f_proj, nn.QuantizedLinear)
        self.assertIsInstance(moe.attention.b_proj, nn.QuantizedLinear)
        self.assertNotIsInstance(moe.attention.q_conv1d.conv, nn.QuantizedLinear)
        mla_attn = native.layers[1].attention
        self.assertIsInstance(mla_attn.embed_q, QuantizedMultiLinear)
        self.assertIsInstance(mla_attn.unembed_out, QuantizedMultiLinear)
        self.assertEqual(mla_attn.embed_q.bits, 8)
        self.assertIsInstance(mla_attn.g_proj, nn.QuantizedLinear)
        self.assertIsInstance(native.lm_head, nn.QuantizedLinear)
        self.assertEqual(native.lm_head.bits, 8)
        self.assertIsInstance(native.model.word_embeddings, nn.QuantizedEmbedding)

        out = native(mx.array([[1, 2, 3, 4, 5]]))
        mx.eval(out)
        self.assertEqual(out.shape, reference_out.shape)
        self.assertTrue(mx.all(mx.isfinite(out)).item())

        cache = make_prompt_cache(native)
        mx.eval(native(mx.array([[1, 2, 3, 4, 5]]), cache=cache))
        step = native(mx.array([[6]]), cache=cache)
        mx.eval(step)
        self.assertTrue(mx.all(mx.isfinite(step)).item())

        # The model's own predicate selects the 8-bit routers.
        fresh = _build_native()
        pred = fresh.quant_predicate
        self.assertEqual(
            pred("model.layers.1.mlp.gate.gate_proj", None),
            {"group_size": 64, "bits": 8},
        )
        self.assertIs(pred("model.layers.1.attention.q_proj", None), True)
        cast = fresh.cast_predicate
        self.assertFalse(cast("model.layers.0.attention.A_log"))
        self.assertFalse(cast("model.layers.0.attention.dt_bias"))
        self.assertFalse(cast("model.layers.1.mlp.gate.expert_bias"))
        self.assertTrue(cast("model.layers.1.mlp.gate.gate_proj.weight"))


class TestBailingHybridLoader(unittest.TestCase):
    @unittest.skipUnless(REAL_CONFIG.exists(), "real config.json not available")
    def test_real_config_resolves_and_constructs(self):
        with open(REAL_CONFIG) as f:
            config = json.load(f)
        self.assertEqual(config["model_type"], "bailing_hybrid")
        self.assertEqual(config["architectures"], ["BailingMoeV3ForCausalLM"])
        model_class, args_class = _get_classes(config)
        self.assertIs(model_class, bailing_hybrid.Model)
        self.assertIs(args_class, bailing_hybrid.ModelArgs)

        args = args_class.from_dict(config)
        self.assertEqual(args.num_hidden_layers, 42)
        self.assertEqual(args.hidden_size, 2560)
        self.assertEqual(args.num_attention_heads, 32)
        self.assertEqual(args.head_dim, 128)
        self.assertEqual(args.vocab_size, 157184)
        self.assertEqual(args.first_k_dense_replace, 2)
        self.assertEqual(args.layer_group_size, 6)
        self.assertEqual((args.num_experts, args.num_experts_per_tok), (512, 8))
        self.assertEqual((args.n_group, args.topk_group), (8, 4))
        self.assertEqual(args.routed_scaling_factor, 2.5)
        self.assertEqual(args.moe_intermediate_size, 768)
        self.assertEqual(args.moe_shared_expert_intermediate_size, 768)
        self.assertEqual(args.num_shared_experts, 1)
        self.assertEqual(args.kv_lora_rank, 512)
        self.assertEqual((args.qk_nope_head_dim, args.qk_rope_head_dim), (128, 64))
        self.assertEqual(args.v_head_dim, 128)
        self.assertIsNone(args.q_lora_rank)
        self.assertEqual(args.rope_theta, 6_000_000)
        self.assertTrue(args.rope_interleave)
        self.assertIsNone(args.rope_scaling)
        self.assertEqual(args.max_position_embeddings, 262144)
        self.assertEqual(args.score_function, "sigmoid")
        self.assertTrue(args.norm_topk_prob)
        self.assertTrue(args.moe_router_enable_expert_bias)
        self.assertTrue(args.no_kda_lora)
        self.assertTrue(args.kda_safe_gate)
        self.assertEqual(args.kda_lower_bound, -5.0)
        self.assertEqual(args.gated_attention_proj_granularity_type, "head_wise")
        self.assertEqual(args.short_conv_kernel_size, 4)
        self.assertEqual(args.num_nextn_predict_layers, 0)
        self.assertFalse(args.tie_word_embeddings)

        # Real schedule: layers 5, 11, ..., 41 are MLA (7), the other 35 KDA.
        globals_ = [
            i for i in range(42) if bailing_hybrid.is_global_layer(i, 6, 42)
        ]
        self.assertEqual(globals_, [5, 11, 17, 23, 29, 35, 41])

        # Two layers with every real width: layer_group_size 2 and no dense
        # layers put one KDA and one MLA MoE layer side by side. Nothing is
        # evaluated, so no 100B-parameter buffers exist.
        small = dict(config, num_hidden_layers=2, layer_group_size=2, first_k_dense_replace=0)
        model = model_class(args_class.from_dict(small))
        self.assertEqual([l.is_global for l in model.layers], [False, True])
        kda = model.layers[0].attention
        self.assertEqual(kda.q_proj.weight.shape, (4096, 2560))
        self.assertEqual(kda.q_conv1d.conv.weight.shape, (4096, 4, 1))
        self.assertEqual(kda.f_proj.weight.shape, (4096, 2560))
        self.assertEqual(kda.g_proj.weight.shape, (4096, 2560))
        self.assertEqual(kda.b_proj.weight.shape, (32, 2560))
        self.assertEqual(kda.A_log.shape, (32,))
        self.assertEqual(kda.dt_bias.shape, (4096,))
        self.assertEqual(kda.o_norm.weight.shape, (128,))
        self.assertEqual(kda.o_proj.weight.shape, (2560, 4096))
        attn = model.layers[1].attention
        self.assertEqual(attn.q_proj.weight.shape, (32 * 192, 2560))
        self.assertEqual(attn.kv_a_proj_with_mqa.weight.shape, (576, 2560))
        self.assertEqual(attn.kv_a_layernorm.weight.shape, (512,))
        self.assertEqual(attn.embed_q.weight.shape, (32, 512, 128))
        self.assertEqual(attn.unembed_out.weight.shape, (32, 128, 512))
        self.assertEqual(attn.g_proj.weight.shape, (32, 2560))
        self.assertEqual(attn.dense.weight.shape, (2560, 4096))
        self.assertEqual(attn.absorbed_max_query, 170)
        for layer in model.layers:
            self.assertEqual(layer.mlp.switch_mlp.up_proj.weight.shape, (512, 768, 2560))
            self.assertEqual(layer.mlp.switch_mlp.down_proj.weight.shape, (512, 2560, 768))
            self.assertEqual(layer.mlp.gate.gate_proj.weight.shape, (512, 2560))
            self.assertEqual(layer.mlp.gate.expert_bias.shape, (512,))
            self.assertEqual(layer.mlp.shared_experts.up_proj.weight.shape, (768, 2560))
        self.assertEqual(model.lm_head.weight.shape, (157184, 2560))
        cache = make_prompt_cache(model)
        self.assertEqual([type(c) for c in cache], [ArraysCache, KVCache])
        del model

        # Every quantization override in config.json names a module the port
        # builds, and the mixed-precision set matches the README (116 x 6-bit,
        # 339 x 8-bit).
        quant = config["quantization"]
        overrides = {k: v for k, v in quant.items() if isinstance(v, dict)}
        self.assertEqual(len(overrides), 455)
        bits = [v["bits"] for v in overrides.values()]
        self.assertEqual((bits.count(6), bits.count(8)), (116, 339))
        kda_suffixes = {"q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "b_proj", "o_proj"}
        mla_suffixes = {"q_proj", "kv_a_proj_with_mqa", "embed_q", "unembed_out", "g_proj", "dense"}
        for path in overrides:
            if path in ("lm_head", "model.word_embeddings"):
                continue
            parts = path.split(".")
            self.assertEqual(parts[:2], ["model", "layers"], path)
            idx = int(parts[2])
            is_global = bailing_hybrid.is_global_layer(idx, 6, 42)
            if parts[3] == "attention":
                self.assertIn(parts[4], mla_suffixes if is_global else kda_suffixes, path)
            else:
                self.assertEqual(parts[3], "mlp", path)
                if idx < 2:
                    self.assertIn(parts[4], {"gate_proj", "up_proj", "down_proj"}, path)
                elif parts[4] == "gate":
                    # Routers above the 4-bit base (6-bit in the 6-bit layers).
                    self.assertEqual(parts[5:], ["gate_proj"], path)
                    self.assertIn(overrides[path]["bits"], (6, 8), path)
                else:
                    self.assertEqual(parts[4:5], ["shared_experts"], path)
        routers = [p for p in overrides if p.endswith("mlp.gate.gate_proj")]
        self.assertEqual(len(routers), 40)

    def test_native_module_preferred_over_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with open(path / "config.json", "w") as f:
                json.dump(
                    dict(
                        TINY_CONFIG,
                        architectures=["BailingMoeV3ForCausalLM"],
                        model_file="bailing_hybrid.py",
                    ),
                    f,
                )
            model, config = load_model(path, strict=False)
            self.assertIsInstance(model, bailing_hybrid.Model)
            self.assertEqual(config["model_file"], "bailing_hybrid.py")

            with open(path / "config.json", "w") as f:
                json.dump(
                    dict(
                        TINY_CONFIG,
                        model_type="no_such_model_xyz",
                        model_file="bailing_hybrid.py",
                    ),
                    f,
                )
            with self.assertRaises(ValueError) as ctx:
                load_model(path, strict=False)
            self.assertIn("trust_remote_code", str(ctx.exception))


class TestBailingHybridEagerDispatch(unittest.TestCase):
    """MLX_LING_EAGER_DISPATCH is pure scheduling (async_eval) and must not
    change any value. The flag is a module global, resolved at import."""

    def _run(self, model):
        mx.random.seed(3)
        cache = make_prompt_cache(model)
        outs = [model(mx.random.randint(0, TINY_CONFIG["vocab_size"], (1, 6)), cache=cache)]
        for w in (1, 3):
            step = mx.random.randint(0, TINY_CONFIG["vocab_size"], (1, w))
            outs.append(model(step, cache=cache))
        mx.eval(outs)
        return outs

    def test_eager_dispatch_bit_identical(self):
        base = self._run(_build_native())
        saved = bailing_hybrid._EAGER_DISPATCH
        try:
            bailing_hybrid._EAGER_DISPATCH = True
            got = self._run(_build_native())
        finally:
            bailing_hybrid._EAGER_DISPATCH = saved
        for i, (a, b) in enumerate(zip(base, got)):
            self.assertTrue(mx.array_equal(a, b).item(), f"run {i}: logits differ")


if __name__ == "__main__":
    unittest.main()
