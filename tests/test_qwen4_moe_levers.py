# Copyright © 2026 Apple Inc.
#
# Tests for the 2026-08-27 decode-decomposition MoE levers
# (results/qwen38-decode-decomposition-20260827.json: decode GPU window 86%
# of the step at 22% bandwidth — occupancy-bound, so tile aggregation is
# back on the table):
#
#   MLX_QWEN4_MOE_FUSED_GATE_UP    qwen3_next._MOE_FUSED_GATE_UP    (tolerance)
#   MLX_QWEN4_MOE_SHARED_IN_GATHER qwen3_next._MOE_SHARED_IN_GATHER (tolerance)
#
# Both are LOAD-TIME weight transforms. The first implementation built the
# tables at runtime and was OOM-killed on the 104 GB serving artifact (45 GB
# of extra fused tensors); see wiki lessons/moe-runtime-fusion-oom.md. The
# memory-shape test below is the regression guard tiny-model output tests
# could not provide: it asserts the transformed model holds no second copy.
#
# The third lever of the set, MLX_QWEN4_QSA_FUSED_PROJ, lives in
# qwen4_exp.py with its tests in tests/test_qwen4_exp_levers.py.

import unittest
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import qwen3_next
from mlx_lm.models.qwen3_next import (
    MaterializationTooLarge,
    check_materialization_budget,
    transform_moe_weights,
)
from mlx_lm.models.qwen4_exp import TextModelArgs


@contextmanager
def lever(module, name, value=True):
    previous = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, previous)


@contextmanager
def levers(fuse_gate_up=False, fold_shared=False):
    with lever(qwen3_next, "_MOE_FUSED_GATE_UP", fuse_gate_up), lever(
        qwen3_next, "_MOE_SHARED_IN_GATHER", fold_shared
    ):
        yield


def moe_args(**overrides):
    values = dict(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=64,
        max_position_embeddings=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[],
        ple_embed_dim=64,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
        },
    )
    values.update(overrides)
    return TextModelArgs(**values)


def _block(quantized, **overrides):
    """Build a block under the CURRENT lever globals (they are structural)."""
    block = qwen3_next.Qwen3NextSparseMoeBlock(moe_args(**overrides))
    if quantized:
        nn.quantize(block, group_size=32, bits=4)
    block.eval()
    mx.eval(block.parameters())
    return block


def _weights(block) -> dict:
    return {f"mlp.{k}": v for k, v in tree_flatten(block.parameters())}


def _lever_block_from(stock, fuse_gate_up, fold_shared, quantized, **overrides):
    """Transform a stock block's weights and load them into a lever block."""
    weights = _weights(stock)
    changed = transform_moe_weights(
        weights, ["mlp"], fuse_gate_up=fuse_gate_up, fold_shared=fold_shared
    )
    with levers(fuse_gate_up, fold_shared):
        block = _block(quantized, **overrides)
    block.load_weights(
        [(k[len("mlp.") :], v) for k, v in weights.items()], strict=True
    )
    block.eval()
    mx.eval(block.parameters())
    return block, changed


def _param_bytes(block) -> int:
    return sum(v.nbytes for _, v in tree_flatten(block.parameters()))


def _scaled_err(actual, expected):
    """Max deviation as a fraction of output scale (a per-element relative
    metric explodes on near-zero elements)."""
    actual = actual.astype(mx.float32)
    expected = expected.astype(mx.float32)
    return (mx.abs(actual - expected).max() / mx.abs(expected).max()).item()


# Widths straddling the sorted-gather threshold (>= 64 flat indices).
_SHAPES = ((1, 1, 64), (1, 8, 64), (1, 40, 64))
_COMBOS = ((True, False), (False, True), (True, True))


class TestLoadTimeTransforms(unittest.TestCase):
    def test_flags_default_off_and_layout_is_stock(self):
        self.assertFalse(qwen3_next._MOE_FUSED_GATE_UP)
        self.assertFalse(qwen3_next._MOE_SHARED_IN_GATHER)
        block = _block(False)
        self.assertFalse(block.fused_gate_up)
        self.assertFalse(block.shared_folded)
        self.assertTrue(hasattr(block.switch_mlp, "gate_proj"))
        self.assertTrue(hasattr(block, "shared_expert"))

    def test_transformed_layout_and_expert_counts(self):
        stock = _block(True)
        for fuse_gate_up, fold_shared in _COMBOS:
            block, changed = _lever_block_from(
                stock, fuse_gate_up, fold_shared, True
            )
            self.assertEqual(changed, 1)
            experts = 4 + (1 if fold_shared else 0)
            if fuse_gate_up:
                self.assertTrue(hasattr(block.switch_mlp, "gate_up_proj"))
                self.assertFalse(hasattr(block.switch_mlp, "gate_proj"))
                self.assertEqual(
                    block.switch_mlp.gate_up_proj.num_experts, experts
                )
                self.assertEqual(
                    block.switch_mlp.gate_up_proj.output_dims, 128
                )
            else:
                self.assertEqual(block.switch_mlp.gate_proj.num_experts, experts)
            self.assertEqual(block.switch_mlp.down_proj.num_experts, experts)
            self.assertEqual(hasattr(block, "shared_expert"), not fold_shared)

    def test_transform_adds_no_second_copy(self):
        """The OOM regression guard: a load-time transform must not grow
        resident parameters beyond the folded shared expert's own rows."""
        stock = _block(True)
        base = _param_bytes(stock)
        for fuse_gate_up, fold_shared in _COMBOS:
            block, _ = _lever_block_from(stock, fuse_gate_up, fold_shared, True)
            after = _param_bytes(block)
            # Folding moves the shared expert into the routed table; fusing
            # moves bytes between tensors. Neither may duplicate anything.
            self.assertLessEqual(
                after,
                base * 1.02,
                f"fuse={fuse_gate_up} fold={fold_shared}: "
                f"{after} bytes vs stock {base}",
            )

    def test_outputs_match_within_tolerance(self):
        for quantized in (False, True):
            stock = _block(quantized)
            for fuse_gate_up, fold_shared in _COMBOS:
                block, _ = _lever_block_from(
                    stock, fuse_gate_up, fold_shared, quantized
                )
                for shape in _SHAPES:
                    for dtype in (mx.float32, mx.bfloat16):
                        x = mx.random.normal(
                            shape, key=mx.random.key(shape[1])
                        ).astype(dtype)
                        expected = stock(x)
                        actual = block(x)
                        mx.eval(expected, actual)
                        self.assertLess(
                            _scaled_err(actual, expected),
                            3e-3,
                            f"fuse={fuse_gate_up} fold={fold_shared} "
                            f"shape={shape} dtype={dtype}",
                        )

    def test_fused_gate_up_halves_routed_dispatches(self):
        """Structural check that the lever does what it claims."""
        stock = _block(True)
        block, _ = _lever_block_from(stock, True, False, True)
        stock_projs = [
            name
            for name, _ in tree_flatten(stock.switch_mlp.parameters())
            if name.endswith(".weight")
        ]
        lever_projs = [
            name
            for name, _ in tree_flatten(block.switch_mlp.parameters())
            if name.endswith(".weight")
        ]
        self.assertEqual(len(stock_projs), 3)
        self.assertEqual(len(lever_projs), 2)

    def test_shared_fold_refuses_mismatched_width(self):
        """A shared expert of a different width cannot become expert E."""
        with levers(False, True):
            block = _block(False, shared_expert_intermediate_size=32)
        self.assertFalse(block.shared_folded)
        self.assertTrue(hasattr(block, "shared_expert"))
        weights = _weights(block)
        self.assertEqual(
            transform_moe_weights(
                weights, ["mlp"], fuse_gate_up=False, fold_shared=True
            ),
            0,
        )

    def test_transform_is_a_noop_without_flags(self):
        stock = _block(True)
        weights = _weights(stock)
        before = dict(weights)
        self.assertEqual(
            transform_moe_weights(
                weights, ["mlp"], fuse_gate_up=False, fold_shared=False
            ),
            0,
        )
        self.assertEqual(set(weights), set(before))

    def test_quantized_refusion_reproduces_shipped_layout(self):
        """concat(quantize(gate), quantize(up)) along N equals
        quantize(concat): affine groups run along K, so the transform
        rebuilds the checkpoint's fused tensor byte for byte."""
        from mlx_lm.models.qwen3_5 import _array_bytes

        gate = mx.random.normal((4, 64, 64), key=mx.random.key(0))
        up = mx.random.normal((4, 64, 64), key=mx.random.key(1))
        fused = mx.quantize(mx.concatenate([gate, up], axis=1), 64, 4)
        refused = [
            mx.concatenate(parts, axis=1)
            for parts in zip(mx.quantize(gate, 64, 4), mx.quantize(up, 64, 4))
        ]
        for expected, actual in zip(fused, refused):
            mx.eval(expected, actual)
            self.assertEqual(_array_bytes(actual), _array_bytes(expected))


class TestSanitizeSeamEndToEnd(unittest.TestCase):
    """The production path: checkpoint keys -> Model.sanitize -> load."""

    def _model(self):
        from mlx_lm.models.qwen4_exp import Model, ModelArgs

        args = moe_args(ple_layer_ids=[2], mtp_num_hidden_layers=1)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        model.eval()
        mx.eval(model.parameters())
        return model

    @staticmethod
    def _checkpoint_keys(model) -> dict:
        prefix = "language_model.model."
        return {
            ("model.language_model." + k[len(prefix) :])
            if k.startswith(prefix)
            else k: v
            for k, v in tree_flatten(model.parameters())
        }

    @staticmethod
    def _greedy(model, steps=10):
        cache = model.make_cache()
        tokens = []
        logits = model(mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32), cache=cache)
        for _ in range(steps):
            token = int(mx.argmax(logits[:, -1], axis=-1).item())
            tokens.append(token)
            logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        return tokens

    def test_sanitize_transforms_and_keeps_trajectory_and_size(self):
        stock = self._model()
        weights = self._checkpoint_keys(stock)
        expected = self._greedy(stock)
        stock_bytes = _param_bytes(stock)
        for fuse_gate_up, fold_shared in _COMBOS:
            with levers(fuse_gate_up, fold_shared):
                model = self._model()
                sanitized = model.sanitize(dict(weights))
                # strict=True proves the transformed keys match the layout.
                model.load_weights(list(sanitized.items()), strict=True)
                model.eval()
                mx.eval(model.parameters())
                actual = self._greedy(model)
                size = _param_bytes(model)
            self.assertEqual(
                actual, expected, f"fuse={fuse_gate_up} fold={fold_shared}"
            )
            self.assertLessEqual(size, stock_bytes * 1.02)

    def test_shipped_fused_tensor_is_never_split(self):
        """Raw upstream checkpoints ship gate_up_proj fused; the lever must
        keep it instead of splitting and re-fusing."""
        with levers(fuse_gate_up=True):
            model = self._model()
            gate_up = mx.zeros((4, 128, 64))
            out = model.sanitize(
                {
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
                    "model.language_model.layers.0.mlp.experts.down_proj": mx.zeros(
                        (4, 64, 64)
                    ),
                }
            )
        prefix = "language_model.model.layers.0.mlp.switch_mlp"
        self.assertIn(f"{prefix}.gate_up_proj.weight", out)
        self.assertNotIn(f"{prefix}.gate_proj.weight", out)
        self.assertEqual(out[f"{prefix}.gate_up_proj.weight"].shape, (4, 128, 64))


class TestMaterializationBudget(unittest.TestCase):
    def test_small_table_is_allowed_and_recorded(self):
        estimate = check_materialization_budget(1 << 20, "tiny table")
        self.assertEqual(estimate["bytes"], 1 << 20)
        if estimate["checked"]:
            self.assertTrue(estimate["fits"])
            self.assertGreater(estimate["budget_bytes"], 0)

    def test_oversized_table_is_refused(self):
        if not mx.metal.is_available():
            self.skipTest("no Metal device to size against")
        with self.assertRaises(MaterializationTooLarge):
            # The 45 GB the OOM-killed runtime fusion would have needed.
            check_materialization_budget(45 * 10**9 * 10, "oversized table")


if __name__ == "__main__":
    unittest.main()
