import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_lm.models import qwen3_next, qwen4_fused_moe


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    def squeeze(self, axis):
        axis %= self.ndim
        if self.shape[axis] != 1:
            raise ValueError("cannot squeeze a non-singleton dimension")
        return FakeArray(self.shape[:axis] + self.shape[axis + 1 :], self.dtype)

    def reshape(self, shape):
        return FakeArray(shape, self.dtype)


class FakeQuantizedDown:
    training = False
    num_experts = 512
    group_size = 64
    bits = 4
    mode = "affine"

    def __init__(self):
        self.weight = FakeArray((512, 2560, 80), mx.uint32)
        self.scales = FakeArray((512, 2560, 10), mx.bfloat16)
        self.biases = FakeArray((512, 2560, 10), mx.bfloat16)

    def __contains__(self, name):
        return False

    def __getitem__(self, name):
        return getattr(self, name)


def tiny_args():
    return SimpleNamespace(
        hidden_size=16,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        norm_topk_prob=True,
        num_experts=4,
        num_experts_per_tok=2,
    )


class TestFusedMoeAdmission(unittest.TestCase):
    def production_inputs(self, tokens=1):
        prefix = (1, tokens)
        return (
            FakeArray(prefix + (10, 640), mx.bfloat16),
            FakeArray(prefix + (10,), mx.uint32),
            FakeArray(prefix + (10,), mx.bfloat16),
            FakeArray((512, 2560, 80), mx.uint32),
            FakeArray((512, 2560, 10), mx.bfloat16),
            FakeArray((512, 2560, 10), mx.bfloat16),
        )

    def test_exact_decode_and_mtp_shapes_are_admitted(self):
        for tokens in (1, 3):
            result = qwen4_fused_moe.admit_qwen4_fused_down(
                *self.production_inputs(tokens)
            )
            self.assertTrue(result.accepted, result.reason)
            self.assertEqual(result.tokens, tokens)

    def test_prefill_and_wrong_quantization_fall_back(self):
        result = qwen4_fused_moe.admit_qwen4_fused_down(
            *self.production_inputs(4)
        )
        self.assertFalse(result.accepted)
        self.assertIn("M=1 or M=3", result.reason)

        result = qwen4_fused_moe.admit_qwen4_fused_down(
            *self.production_inputs(1), bits=8
        )
        self.assertFalse(result.accepted)
        self.assertIn("affine q4", result.reason)

        values = list(self.production_inputs(1))
        values[2] = FakeArray((1, 1, 10), mx.float32)
        result = qwen4_fused_moe.admit_qwen4_fused_down(*values)
        self.assertFalse(result.accepted)
        self.assertIn("scores dtype", result.reason)

    def test_variant_dispatch_grids_are_static_and_selectable(self):
        calls = []

        def fake_kernel(**kwargs):
            calls.append(kwargs)
            return [FakeArray(kwargs["output_shapes"][0], mx.bfloat16)]

        values = self.production_inputs(3)
        with patch.object(
            qwen4_fused_moe, "_kernel", return_value=fake_kernel
        ) as select:
            for variant, grid, threadgroup in (
                ("scalar", (32, 2560, 3), (32, 1, 1)),
                ("tile4", (160, 640, 3), (160, 1, 1)),
            ):
                output = qwen4_fused_moe.qwen4_fused_down(
                    *values, variant=variant
                )
                self.assertEqual(output.shape, (1, 3, 2560))
                self.assertEqual(calls[-1]["grid"], grid)
                self.assertEqual(calls[-1]["threadgroup"], threadgroup)
                select.assert_called_with(variant)

        with self.assertRaisesRegex(ValueError, "unknown Qwen4 fused-down variant"):
            qwen4_fused_moe.qwen4_fused_down(*values, variant="bogus")


class TestFusedMoeIntegration(unittest.TestCase):
    def test_auto_selects_split_gate_up_fused_down_class(self):
        with patch.object(qwen3_next, "_MOE_FUSED_EXPERT_MODE", "auto"), patch.object(
            qwen3_next, "_MOE_FUSED_GATE_UP", False
        ), patch.object(qwen3_next, "_MOE_SHARED_IN_GATHER", False):
            block = qwen3_next.Qwen3NextSparseMoeBlock(tiny_args())
        self.assertTrue(block.fused_expert_kernel_enabled)
        self.assertEqual(block.fused_expert_kernel_mode, "auto")
        self.assertIsInstance(block.switch_mlp, qwen3_next.FusedDownSwitchGLU)

    def test_shared_fold_disables_incompatible_kernel(self):
        with patch.object(qwen3_next, "_MOE_FUSED_EXPERT_MODE", "auto"), patch.object(
            qwen3_next, "_MOE_FUSED_GATE_UP", False
        ), patch.object(qwen3_next, "_MOE_SHARED_IN_GATHER", True):
            block = qwen3_next.Qwen3NextSparseMoeBlock(tiny_args())
        self.assertFalse(block.fused_expert_kernel_enabled)

    def test_resident_model_switches_modes_without_replacing_weights(self):
        with patch.object(qwen3_next, "_MOE_FUSED_EXPERT_MODE", "stock"), patch.object(
            qwen3_next, "_MOE_FUSED_GATE_UP", False
        ), patch.object(qwen3_next, "_MOE_SHARED_IN_GATHER", False):
            block = qwen3_next.Qwen3NextSparseMoeBlock(tiny_args())
        weight = block.switch_mlp.down_proj.weight

        self.assertEqual(
            qwen3_next.qwen4_fused_expert_mode_counts(block),
            {"stock": 1, "auto": 0, "scalar": 0, "tile4": 0},
        )
        self.assertEqual(qwen3_next.set_qwen4_fused_expert_mode(block, "tile4"), 1)
        self.assertIs(block.switch_mlp.down_proj.weight, weight)
        self.assertEqual(block.fused_expert_kernel_mode, "tile4")
        self.assertEqual(qwen3_next.set_qwen4_fused_expert_mode(block, "stock"), 1)
        self.assertIs(block.switch_mlp.down_proj.weight, weight)
        with self.assertRaisesRegex(ValueError, "unknown fused expert mode"):
            qwen3_next.set_qwen4_fused_expert_mode(block, "bogus")
        self.assertEqual(block.fused_expert_kernel_mode, "stock")

    def test_environment_policy_defaults_auto_and_preserves_hard_off(self):
        with patch.dict(qwen3_next.os.environ, {}, clear=True):
            self.assertEqual(qwen3_next._fused_expert_mode_from_env(), "auto")
        for value in ("0", "off", "false", "stock"):
            with self.subTest(value=value), patch.dict(
                qwen3_next.os.environ,
                {"MLX_QWEN4_FUSED_EXPERT_KERNEL": value},
                clear=True,
            ):
                self.assertEqual(
                    qwen3_next._fused_expert_mode_from_env(), "stock"
                )
        for value, expected in (
            ("1", "auto"),
            ("scalar", "scalar"),
            ("tile4", "tile4"),
        ):
            with self.subTest(value=value), patch.dict(
                qwen3_next.os.environ,
                {"MLX_QWEN4_FUSED_EXPERT_KERNEL": value},
                clear=True,
            ):
                self.assertEqual(
                    qwen3_next._fused_expert_mode_from_env(), expected
                )

    def test_auto_selects_qualified_variant_by_token_width(self):
        down = FakeQuantizedDown()
        sentinel = object()
        for tokens, expected in ((1, "tile4"), (3, "scalar")):
            hidden = FakeArray((1, tokens, 10, 1, 640), mx.bfloat16)
            indices = FakeArray((1, tokens, 10), mx.uint32)
            scores = FakeArray((1, tokens, 10), mx.bfloat16)
            accepted = qwen4_fused_moe.FusedMoeAdmission(
                True, "eligible", tokens
            )
            with self.subTest(tokens=tokens), patch.object(
                qwen3_next, "QuantizedSwitchLinear", FakeQuantizedDown
            ), patch.object(
                qwen4_fused_moe,
                "admit_qwen4_fused_down",
                return_value=accepted,
            ), patch.object(
                qwen4_fused_moe,
                "qwen4_fused_down",
                return_value=sentinel,
            ) as execute:
                result = qwen3_next._try_qwen4_fused_down(
                    hidden, indices, scores, down, False, "auto"
                )
            self.assertIs(result, sentinel)
            self.assertEqual(execute.call_args.kwargs["variant"], expected)

    def test_switch_singleton_is_removed_before_kernel(self):
        hidden = FakeArray((1, 1, 10, 1, 640), mx.bfloat16)
        indices = FakeArray((1, 1, 10), mx.uint32)
        scores = FakeArray((1, 1, 10), mx.bfloat16)
        down = FakeQuantizedDown()
        sentinel = object()

        accepted = qwen4_fused_moe.FusedMoeAdmission(True, "eligible", 1)
        with patch.object(
            qwen3_next, "QuantizedSwitchLinear", FakeQuantizedDown
        ), patch.object(
            qwen4_fused_moe, "admit_qwen4_fused_down", return_value=accepted
        ) as admit, patch.object(
            qwen4_fused_moe, "qwen4_fused_down", return_value=sentinel
        ) as execute:
            result = qwen3_next._try_qwen4_fused_down(
                hidden, indices, scores, down, False
            )

        self.assertIs(result, sentinel)
        self.assertEqual(admit.call_args.args[0].shape, (1, 1, 10, 640))
        self.assertEqual(execute.call_args.args[0].shape, (1, 1, 10, 640))
        self.assertEqual(execute.call_args.kwargs["variant"], "scalar")

    def test_rejected_kernel_still_applies_router_reduction(self):
        rows = np.arange(16, dtype=np.float32).reshape(1, 1, 2, 1, 8)
        scores = np.array([[[0.25, 0.75]]], dtype=np.float32)

        class Projection:
            training = False

            def __init__(self, output):
                self.output = output

            def __call__(self, *args, **kwargs):
                return self.output

            def __contains__(self, name):
                return False

        fake = SimpleNamespace(
            training=False,
            up_proj=Projection(rows),
            gate_proj=Projection(rows),
            down_proj=Projection(rows),
            activation=lambda up, gate: up,
        )
        indices = np.array([[[0, 1]]], dtype=np.uint32)
        with patch.object(qwen3_next.mx, "expand_dims", return_value=rows):
            actual = qwen3_next.FusedDownSwitchGLU.__call__(
                fake, rows, indices, scores
            )
        expected = (rows.squeeze(-2) * scores[..., None]).sum(axis=-2)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
