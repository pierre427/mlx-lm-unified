# Copyright © 2026 mlx-uag lab
"""Quantized KV caches on the absorbed-MLA twins.

``kimi_linear``, ``longcat_flash``, ``bailing_moe_v3`` and ``deepseek_v32``
scored the rope half of MLA attention with a plain ``@`` against the cached
rope keys, before any handling of a quantized cache.  ``maybe_quantize_kv_cache``
reaches all four (it recurses into ``CacheList``), so ``--kv-bits`` raised
``AttributeError: 'list' object has no attribute 'swapaxes'`` at every query
width, decode included.  ``deepseek_v2`` / ``deepseek_v3`` / ``glm4_moe_lite``
route through ``quantized_scaled_dot_product_attention`` instead; these tests
pin the four twins to that behavior:

  * a quantized forward runs and tracks the bf16 forward, at 8 and 4 bits, at
    L in {1, 3}, on both the absorbed and the expanded branch,
  * the branch under test is asserted to have actually run,
  * asymmetric key/value bits are refused rather than failing on ``bits=None``,
  * the non-quantized path is byte-identical to what it was before the fix.
"""

import hashlib
import os
import unittest

# fp32 allclose harness: MLX defaults fp32 matmul to TF32 on M-series NAX.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx

from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models import mla
from mlx_lm.models.cache import make_prompt_cache

PROMPT = [[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]]

# Tiny configs sized for a quantized KV cache: the latent (cache keys) and the
# rope half (cache values) are both quantized along their last axis, so both
# must be a multiple of the 64-element group size.


def _tiny_kimi_linear():
    from mlx_lm.models import kimi_linear

    return kimi_linear.Model(
        kimi_linear.ModelArgs(
            model_type="kimi_linear",
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=64,
            head_dim=64,
            rope_theta=100.0,
            rms_norm_eps=1e-6,
            linear_attn_config={"num_heads": 2, "head_dim": 32, "kda_layers": [1]},
            model_max_length=1000,
            num_experts=2,
            moe_intermediate_size=64,
            kv_lora_rank=64,
            qk_nope_head_dim=64,
            qk_rope_head_dim=64,
            v_head_dim=64,
        )
    )


def _tiny_longcat_flash():
    from mlx_lm.models import longcat_flash

    return longcat_flash.Model(
        longcat_flash.ModelArgs(
            model_type="longcat_flash",
            attention_method="MLA",
            zero_expert_type="identity",
            hidden_size=64,
            ffn_hidden_size=64,
            moe_topk=2,
            expert_ffn_hidden_size=64,
            n_routed_experts=2,
            zero_expert_num=2,
            num_layers=2,
            vocab_size=128,
            max_position_embeddings=1000,
            num_attention_heads=2,
            kv_lora_rank=64,
            q_lora_rank=32,
            qk_rope_head_dim=64,
            qk_nope_head_dim=32,
            v_head_dim=32,
            routed_scaling_factor=1.0,
            rms_norm_eps=1e-5,
            rope_theta=1000,
            mla_scale_q_lora=True,
            mla_scale_kv_lora=True,
            attention_bias=False,
        )
    )


def _tiny_bailing_moe_v3():
    from mlx_lm.models import bailing_moe_v3

    return bailing_moe_v3.Model(
        bailing_moe_v3.ModelArgs(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            moe_intermediate_size=64,
            moe_shared_expert_intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            n_group=2,
            topk_group=1,
            layer_group_size=4,
            head_dim=64,
            q_lora_rank=32,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=64,
            v_head_dim=32,
        )
    )


def _tiny_deepseek_v32():
    from mlx_lm.models import deepseek_v32

    # index_topk is small enough that the DSA indexer actually fires on a
    # 12-token prompt, so the sparse gather (L == 1) and the sparse mask
    # (L > 1) are both exercised against the quantized cache.
    return deepseek_v32.Model(
        deepseek_v32.ModelArgs(
            model_type="deepseek_v32",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=64,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=4,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            n_shared_experts=1,
            kv_lora_rank=64,
            q_lora_rank=32,
            qk_rope_head_dim=64,
            v_head_dim=32,
            qk_nope_head_dim=32,
            index_head_dim=64,
            index_n_heads=2,
            index_topk=8,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
        )
    )


BUILDERS = {
    "bailing_moe_v3": _tiny_bailing_moe_v3,
    "deepseek_v32": _tiny_deepseek_v32,
    "kimi_linear": _tiny_kimi_linear,
    "longcat_flash": _tiny_longcat_flash,
}

# Digests of the non-quantized forward, taken on `unified` before this fix.
# The bf16 path must stay byte-identical: the quantized handling is additive
# and sits behind `isinstance(k_pe, mx.array)`.
BF16_DIGESTS = {
    "bailing_moe_v3": "cd5da24d096a839b",
    "deepseek_v32": "2e27071968847701",
    "kimi_linear": "012103e113dd70eb",
    "longcat_flash": "da41c754ac1f2b2f",
}


# Relative (max-abs / max-abs) tolerance between a quantized and a bf16 run.
# Measured on these tiny configs: 8-bit is <= 0.006 everywhere and 4-bit <= 0.09,
# except on deepseek_v32.  Its DSA indexer picks the sparse key set by top-k over
# scores read from the (also quantized) index cache, and that discrete choice can
# land on a different set at 4 bits, moving the output far more than the
# arithmetic error.  That is a property of DSA under quantization, not of the
# dequant boundary under test -- which test_quantized_branches_agree_with_each_other
# pins directly.
_DEFAULT_TOLERANCE = {8: 0.02, 4: 0.10}
TOLERANCES = {name: _DEFAULT_TOLERANCE for name in BUILDERS}
TOLERANCES["deepseek_v32"] = {8: 0.02, 4: 0.40}


def _build(name):
    mx.random.seed(0)
    model = BUILDERS[name]()
    model.eval()
    return model


def _digest(x):
    return hashlib.sha256(
        x.astype(mx.float32).flatten().tolist().__repr__().encode()
    ).hexdigest()[:16]


def _run(model, lengths, bits=None):
    """Prefill PROMPT, then step `lengths` tokens at a time. Returns the logits
    of the final step. With `bits`, the cache is quantized after the prefill."""
    cache = make_prompt_cache(model)
    out = model(mx.array(PROMPT), cache=cache)
    mx.eval(out)
    if bits is not None:
        maybe_quantize_kv_cache(cache, 0, 64, bits)
    for length in lengths:
        out = model(mx.array([[2] * length]), cache=cache)
        mx.eval(out)
    return out


class _BranchRecorder:
    """Assert the branch under test actually ran (see the lab rule: an A/B
    whose arms are secretly identical looks exactly like a null)."""

    def __enter__(self):
        self.seen = set()
        self._real = mla.use_absorbed_path
        recorder = self

        def spy(query_len, cache_len, geometry):
            result = recorder._real(query_len, cache_len, geometry)
            recorder.seen.add(bool(result))
            return result

        for module in _patch_targets():
            module.use_absorbed_path = spy
        return self

    def __exit__(self, *exc):
        for module in _patch_targets():
            module.use_absorbed_path = self._real
        return False


def _patch_targets():
    from mlx_lm.models import (
        bailing_moe_v3,
        deepseek_v32,
        kimi_linear,
        longcat_flash,
    )

    return (mla, bailing_moe_v3, deepseek_v32, kimi_linear, longcat_flash)


class TestQuantizedKVCacheOnMLATwins(unittest.TestCase):
    def tearDown(self):
        mla.set_absorbed_max_query_override(None)

    def test_quantized_forward_tracks_bf16(self):
        for name in BUILDERS:
            tolerances = TOLERANCES[name]
            model = _build(name)
            for absorbed in (True, False):
                # 0 forces the expanded branch at every width; a huge limit
                # forces the absorbed branch.
                mla.set_absorbed_max_query_override(
                    mla.ABSORBED_UNBOUNDED if absorbed else 0
                )
                for length in (1, 3):
                    with self.subTest(model=name, absorbed=absorbed, L=length):
                        with _BranchRecorder() as rec:
                            reference = _run(model, [length])
                        self.assertEqual(rec.seen, {absorbed})
                        scale = float(mx.abs(reference).max())
                        for bits in (8, 4):
                            with _BranchRecorder() as rec:
                                got = _run(model, [length], bits=bits)
                            self.assertEqual(rec.seen, {absorbed})
                            self.assertEqual(got.shape, reference.shape)
                            error = float(mx.abs(got - reference).max()) / scale
                            self.assertLess(error, tolerances[bits])

    def test_quantized_branches_agree_with_each_other(self):
        # Both branches are the same function up to quantization error, so a
        # quantized absorbed run and a quantized expanded run must agree.
        for name in BUILDERS:
            with self.subTest(model=name):
                model = _build(name)
                mla.set_absorbed_max_query_override(mla.ABSORBED_UNBOUNDED)
                absorbed = _run(model, [3], bits=8)
                mla.set_absorbed_max_query_override(0)
                expanded = _run(model, [3], bits=8)
                scale = float(mx.abs(absorbed).max())
                error = float(mx.abs(absorbed - expanded).max()) / scale
                self.assertLess(error, 0.01)

    def test_multi_step_quantized_decode(self):
        # A quantized cache that keeps growing: verify-shaped steps followed by
        # single-token decode, which is where the bug bit in production.
        steps = [3, 1, 1, 2]
        for name in BUILDERS:
            with self.subTest(model=name):
                model = _build(name)
                if name == "deepseek_v32":
                    # Over this horizon the DSA indexer picks a different
                    # sparse key set on 2 of its 10 calls when it reads an
                    # 8-bit index cache, so a quantized run cannot track a
                    # bf16 one.  Pin what this fix owns instead: on the *same*
                    # quantized cache both MLA branches see the same selection
                    # and must agree.
                    mla.set_absorbed_max_query_override(mla.ABSORBED_UNBOUNDED)
                    absorbed = _run(model, steps, bits=8)
                    mla.set_absorbed_max_query_override(0)
                    expanded = _run(model, steps, bits=8)
                    scale = float(mx.abs(absorbed).max())
                    self.assertLess(
                        float(mx.abs(absorbed - expanded).max()) / scale, 0.01
                    )
                    continue
                reference = _run(model, steps)
                got = _run(model, steps, bits=8)
                scale = float(mx.abs(reference).max())
                self.assertLess(
                    float(mx.abs(got - reference).max()) / scale,
                    TOLERANCES[name][8],
                )

    def test_asymmetric_kv_bits_are_refused(self):
        for name in BUILDERS:
            with self.subTest(model=name):
                model = _build(name)
                cache = make_prompt_cache(model)
                mx.eval(model(mx.array(PROMPT), cache=cache))
                maybe_quantize_kv_cache(
                    cache, 0, 64, None, kv_key_bits=8, kv_value_bits=4
                )
                with self.assertRaises(NotImplementedError):
                    mx.eval(model(mx.array([[2]]), cache=cache))

    def test_bf16_path_is_byte_identical(self):
        if not BF16_DIGESTS:
            self.skipTest("no recorded pre-fix digests")
        for name, expected in BF16_DIGESTS.items():
            with self.subTest(model=name):
                model = _build(name)
                self.assertEqual(_digest(_run(model, [1, 3])), expected)


if __name__ == "__main__":
    unittest.main()
