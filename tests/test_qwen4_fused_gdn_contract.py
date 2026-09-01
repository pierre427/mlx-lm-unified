import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models import qwen4_exp, qwen4_fused_gdn


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeCache:
    def __init__(self, conv_state=None, recurrent_state=None, *, speculating=False):
        self.cache = [conv_state, recurrent_state]
        self.lengths = None
        self.speculating = speculating
        self.advanced = 0

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    def advance(self, amount):
        self.advanced += amount


def production_values(dtype=mx.bfloat16):
    return dict(
        qkv=FakeArray((1, 1, 10240), dtype),
        z=FakeArray((1, 1, 6144), dtype),
        b=FakeArray((1, 1, 48), dtype),
        a=FakeArray((1, 1, 48), dtype),
        conv_state=FakeArray((1, 3, 10240), dtype),
        recurrent_state=FakeArray((1, 48, 128, 128), mx.float32),
        conv_weight=FakeArray((10240, 4, 1), dtype),
        A_log=FakeArray((48,), mx.float32),
        dt_bias=FakeArray((48,), dtype),
        norm_weight=FakeArray((128,), dtype),
    )


def admission(**overrides):
    values = production_values()
    values.update(overrides)
    return qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
        **values,
        mask=None,
        cache_lengths=None,
        speculating=False,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="sigmoid",
    )


def tiny_args():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1.0e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )


class Identity:
    def __call__(self, value):
        return value


class TestFusedGdnAdmission(unittest.TestCase):
    def test_production_single_token_decode_is_admitted(self):
        result = admission()
        self.assertTrue(result.accepted, result.reason)

    def test_batch_prefill_mask_and_speculation_fall_back(self):
        result = admission(qkv=FakeArray((2, 1, 10240), mx.bfloat16))
        self.assertFalse(result.accepted)
        self.assertIn("qkv shape", result.reason)
        result = admission(qkv=FakeArray((1, 2, 10240), mx.bfloat16))
        self.assertFalse(result.accepted)
        self.assertIn("qkv shape", result.reason)

        values = production_values()
        result = qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
            **values,
            mask=object(),
            cache_lengths=None,
            speculating=False,
            training=False,
            sharded=False,
            num_key_heads=16,
            num_value_heads=48,
            key_head_dim=128,
            value_head_dim=128,
            conv_kernel=4,
            gate_activation="sigmoid",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "masked decode")

        result = qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
            **values,
            mask=None,
            cache_lengths=None,
            speculating=True,
            training=False,
            sharded=False,
            num_key_heads=16,
            num_value_heads=48,
            key_head_dim=128,
            value_head_dim=128,
            conv_kernel=4,
            gate_activation="sigmoid",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "speculative rollback")

    def test_dtype_and_geometry_are_strict(self):
        result = admission(A_log=FakeArray((48,), mx.float16))
        self.assertFalse(result.accepted)
        self.assertIn("A_log", result.reason)

        result = admission(A_log=FakeArray((48,), mx.bfloat16))
        self.assertTrue(result.accepted, result.reason)

        values = production_values()
        result = qwen4_fused_gdn.admit_qwen4_fused_gdn_decode(
            **values,
            mask=None,
            cache_lengths=None,
            speculating=False,
            training=False,
            sharded=False,
            num_key_heads=24,
            num_value_heads=48,
            key_head_dim=128,
            value_head_dim=128,
            conv_kernel=4,
            gate_activation="sigmoid",
        )
        self.assertFalse(result.accepted)
        self.assertIn("unsupported geometry", result.reason)

    def test_kernel_dispatch_geometry_is_one_threadgroup_per_value_head(self):
        calls = []

        def fake_kernel(**kwargs):
            calls.append(kwargs)
            return [
                FakeArray(shape, dtype)
                for shape, dtype in zip(
                    kwargs["output_shapes"], kwargs["output_dtypes"]
                )
            ]

        values = production_values()
        with patch.object(qwen4_fused_gdn, "_kernel", return_value=fake_kernel):
            outputs = qwen4_fused_gdn.qwen4_fused_gdn_decode(
                values["qkv"],
                values["z"],
                values["b"],
                values["a"],
                values["conv_state"],
                values["conv_weight"],
                values["A_log"],
                values["dt_bias"],
                values["recurrent_state"],
                values["norm_weight"],
                1.0e-6,
                threadgroup_y=16,
            )
        self.assertEqual(calls[0]["grid"], (32, 16, 48))
        self.assertEqual(calls[0]["threadgroup"], (32, 16, 1))
        self.assertEqual(outputs[0].shape, (1, 1, 6144))
        self.assertEqual(outputs[1].shape, (1, 3, 10240))
        self.assertEqual(outputs[2].shape, (1, 48, 128, 128))
        self.assertIn(("RATIO", 3), calls[0]["template"])
        with self.assertRaisesRegex(ValueError, "unsupported threadgroup_y"):
            qwen4_fused_gdn.qwen4_fused_gdn_decode(
                values["qkv"],
                values["z"],
                values["b"],
                values["a"],
                values["conv_state"],
                values["conv_weight"],
                values["A_log"],
                values["dt_bias"],
                values["recurrent_state"],
                values["norm_weight"],
                1.0e-6,
                threadgroup_y=3,
            )


class TestFusedGdnIntegration(unittest.TestCase):
    def test_resident_switch_preserves_weights(self):
        with patch.object(qwen4_exp, "_FUSED_GDN_DECODE", False):
            layer = qwen4_exp.GatedDeltaNet(tiny_args())
        weight = layer.conv1d.weight
        self.assertEqual(
            qwen4_exp.qwen4_fused_gdn_mode_counts(layer),
            {"stock": 1, "fused": 0, "fused_outproj": 0},
        )
        self.assertEqual(qwen4_exp.set_qwen4_fused_gdn_mode(layer, "fused"), 1)
        self.assertIs(layer.conv1d.weight, weight)
        self.assertEqual(layer.fused_gdn_decode_mode, "fused")
        self.assertEqual(
            qwen4_exp.set_qwen4_fused_gdn_mode(layer, "fused_outproj"), 1
        )
        self.assertIs(layer.conv1d.weight, weight)
        self.assertEqual(layer.fused_gdn_decode_mode, "fused_outproj")
        self.assertEqual(qwen4_exp.set_qwen4_fused_gdn_mode(layer, "stock"), 1)
        self.assertIs(layer.conv1d.weight, weight)
        with self.assertRaisesRegex(ValueError, "unknown fused GDN decode mode"):
            qwen4_exp.set_qwen4_fused_gdn_mode(layer, "bogus")

    def test_uninitialized_and_speculative_cache_do_not_reach_runtime(self):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        layer.eval()
        layer.set_fused_gdn_decode_mode("fused")
        values = production_values()
        with patch.object(qwen4_exp, "fused_gdn_runtime_supported") as runtime:
            result = layer._try_fused_decode(
                values["qkv"], values["z"], values["b"], values["a"], None, FakeCache()
            )
        self.assertIsNone(result)
        runtime.assert_not_called()
        self.assertEqual(layer.fused_gdn_decode_last_fallback, "uninitialized cache")

        cache = FakeCache(
            values["conv_state"], values["recurrent_state"], speculating=True
        )
        result = layer._try_fused_decode(
            values["qkv"], values["z"], values["b"], values["a"], None, cache
        )
        self.assertIsNone(result)
        self.assertEqual(layer.fused_gdn_decode_last_fallback, "speculative rollback")

    def test_admitted_path_updates_cache_and_counter_without_real_kernel(self):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
        layer.set_fused_gdn_decode_mode("fused")
        layer.out_proj = Identity()
        values = production_values()
        cache = FakeCache(values["conv_state"], values["recurrent_state"])
        fused_out = FakeArray((1, 1, 6144), mx.bfloat16)
        next_conv = object()
        next_state = object()
        accepted = qwen4_fused_gdn.FusedGdnAdmission(True, "eligible")
        with patch.object(
            qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted
        ), patch.object(
            qwen4_exp, "fused_gdn_runtime_supported", return_value=True
        ), patch.object(
            qwen4_exp, "probe_qwen4_fused_gdn_decode", return_value=8
        ), patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_decode",
            return_value=(fused_out, next_conv, next_state),
        ) as execute:
            result = layer._try_fused_decode(
                values["qkv"], values["z"], values["b"], values["a"], None, cache
            )
        self.assertIs(result, fused_out)
        self.assertIs(cache[0], next_conv)
        self.assertIs(cache[1], next_state)
        self.assertEqual(cache.advanced, 1)
        self.assertEqual(layer.fused_gdn_decode_calls, 1)
        self.assertEqual(execute.call_args.kwargs["threadgroup_y"], 8)


if __name__ == "__main__":
    unittest.main()
