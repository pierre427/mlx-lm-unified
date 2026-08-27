# Copyright © 2026 Apple Inc.
#
# Tests for the 2026-08-27 decode-decomposition tile-aggregation levers
# (results/qwen38-decode-decomposition-20260827.json: decode GPU window 86%
# of the step at 22% bandwidth — occupancy-bound, reopening tile fusion):
#
#   MLX_QWEN4_MOE_FUSED_GATE_UP    qwen3_next._MOE_FUSED_GATE_UP    (tolerance)
#   MLX_QWEN4_MOE_SHARED_IN_GATHER qwen3_next._MOE_SHARED_IN_GATHER (tolerance)
#
# The third lever of the set, MLX_QWEN4_QSA_FUSED_PROJ, lives in
# qwen4_exp.py; its tests sit in tests/test_qwen4_exp_levers.py beside the
# other qwen4_exp levers.
#
# Measured 2026-08-27 on M5 (tiny models, fp32/bf16, plain and q4-g32,
# unsorted and sorted gather widths):
#   moe_fused_gate_up    BIT-IDENTICAL everywhere measured; still gated at
#                        tolerance per direction (the wide fused matmul may
#                        change accumulation grouping at production shapes).
#   moe_shared_in_gather NOT bitwise: composition order is preserved, but
#                        the shared row moves from the plain (q)mm kernel
#                        family to the gather family (fp32 gap = M5 NAX
#                        TF32 path; <= 2e-7 with MLX_ENABLE_TF32=0).
#                        Measured <= 1.5e-3 of output scale => tolerance.
#   qsa_fused_proj       BIT-IDENTICAL (plain qmm -> one wider plain qmm,
#                        same kernel family); asserted bitwise in
#                        tests/test_qwen4_exp_levers.py as its promotion
#                        basis for the bench's bitwise set.

import unittest
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import qwen3_next
from mlx_lm.models.qwen3_5 import _array_bytes
from mlx_lm.models.qwen4_exp import TextModelArgs


@contextmanager
def lever(module, name, value=True):
    previous = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, previous)


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


def _moe_block(quantized, **overrides):
    block = qwen3_next.Qwen3NextSparseMoeBlock(moe_args(**overrides))
    if quantized:
        nn.quantize(block, group_size=32, bits=4)
    # The lever path is inference-only (its tables are frozen copies).
    block.eval()
    mx.eval(block.parameters())
    return block


def _bytes_equal(test, left, right):
    mx.eval(left, right)
    test.assertEqual(left.dtype, right.dtype)
    test.assertEqual(left.shape, right.shape)
    test.assertEqual(_array_bytes(left), _array_bytes(right))


def _scaled_err(actual, expected):
    """Max abs deviation as a fraction of the output scale (a per-element
    relative metric explodes on near-zero output elements)."""
    actual = actual.astype(mx.float32)
    expected = expected.astype(mx.float32)
    return (mx.abs(actual - expected).max() / mx.abs(expected).max()).item()


# Widths straddling the sorted-gather threshold (>= 64 flat indices).
_MOE_SHAPES = ((1, 1, 64), (1, 8, 64), (1, 40, 64))


class TestMoEFusedGateUp(unittest.TestCase):
    def test_quantized_refusion_reproduces_checkpoint_layout(self):
        """concat(quantize(gate), quantize(up)) along N must equal
        quantize(concat) — groups run along K, so the re-fused tensors are
        byte-for-byte the checkpoint's fused [gate|up] quantization."""
        gate = mx.random.normal((4, 64, 64), key=mx.random.key(0))
        up = mx.random.normal((4, 64, 64), key=mx.random.key(1))
        fused = mx.quantize(mx.concatenate([gate, up], axis=1), 64, 4)
        refused = [
            mx.concatenate(parts, axis=1)
            for parts in zip(mx.quantize(gate, 64, 4), mx.quantize(up, 64, 4))
        ]
        for expected, actual in zip(fused, refused):
            _bytes_equal(self, actual, expected)

    def test_fused_gate_up_within_tolerance(self):
        for quantized in (False, True):
            block = _moe_block(quantized)
            for shape in _MOE_SHAPES:
                for dtype in (mx.float32, mx.bfloat16):
                    x = mx.random.normal(
                        shape, key=mx.random.key(shape[1])
                    ).astype(dtype)
                    expected = block(x)
                    with lever(qwen3_next, "_MOE_FUSED_GATE_UP"):
                        actual = block(x)
                    mx.eval(expected, actual)
                    self.assertLess(_scaled_err(actual, expected), 2e-3)
                    self.assertTrue(block._moe_lever_cache)  # engaged

    def test_flag_off_matches_inline_stock_formula(self):
        self.assertFalse(qwen3_next._MOE_FUSED_GATE_UP)
        self.assertFalse(qwen3_next._MOE_SHARED_IN_GATHER)
        block = _moe_block(False)
        x = mx.random.normal((2, 3, 64), key=mx.random.key(9))
        gates = mx.softmax(block.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-2, axis=-1)[..., -2:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        expected = (block.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
        expected = expected + mx.sigmoid(
            block.shared_expert_gate(x)
        ) * block.shared_expert(x)
        _bytes_equal(self, block(x), expected)


class TestMoESharedInGather(unittest.TestCase):
    def test_shared_fold_within_tolerance(self):
        """NOT a bitwise gate: the shared row's kernel family changes (plain
        (q)mm -> gather); measured <= 1.5e-3 of output scale on M5."""
        for quantized in (False, True):
            block = _moe_block(quantized)
            for shape in _MOE_SHAPES:
                for dtype in (mx.float32, mx.bfloat16):
                    x = mx.random.normal(
                        shape, key=mx.random.key(shape[1])
                    ).astype(dtype)
                    expected = block(x)
                    with lever(qwen3_next, "_MOE_SHARED_IN_GATHER"):
                        actual = block(x)
                    mx.eval(expected, actual)
                    self.assertLess(_scaled_err(actual, expected), 3e-3)
                    self.assertTrue(block._moe_lever_cache)  # engaged

    def test_combined_with_fused_gate_up_within_tolerance(self):
        for quantized in (False, True):
            block = _moe_block(quantized)
            for shape in _MOE_SHAPES:
                x = mx.random.normal(
                    shape, key=mx.random.key(shape[1])
                ).astype(mx.bfloat16)
                expected = block(x)
                with lever(qwen3_next, "_MOE_FUSED_GATE_UP"), lever(
                    qwen3_next, "_MOE_SHARED_IN_GATHER"
                ):
                    actual = block(x)
                mx.eval(expected, actual)
                self.assertLess(_scaled_err(actual, expected), 3e-3)

    def test_shape_mismatched_shared_expert_falls_back_to_stock(self):
        block = _moe_block(False, shared_expert_intermediate_size=32)
        x = mx.random.normal((1, 3, 64), key=mx.random.key(4))
        expected = block(x)
        with lever(qwen3_next, "_MOE_SHARED_IN_GATHER"):
            actual = block(x)
        _bytes_equal(self, actual, expected)
        self.assertIsNone(block._moe_lever_cache[(False, True)][1])

    def test_tables_track_weight_replacement(self):
        block = _moe_block(False)
        donor = _moe_block(False)
        x = mx.random.normal((1, 3, 64), key=mx.random.key(5))
        with lever(qwen3_next, "_MOE_SHARED_IN_GATHER"), lever(
            qwen3_next, "_MOE_FUSED_GATE_UP"
        ):
            mx.eval(block(x))  # build tables from the initial weights
            block.update(donor.parameters())
            actual = block(x)
            expected = donor(x)  # same lever path, donor's own weights
        _bytes_equal(self, actual, expected)


if __name__ == "__main__":
    unittest.main()
