import unittest
from contextlib import contextmanager
from os import environ

environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import qwen3_next as qwen3_next_module
from mlx_lm.models import qwen4_exp as qwen4_exp_module
from mlx_lm.models.qwen4_exp import (
    GatedResidual,
    TextModel,
    TextModelArgs,
    _apply_inject,
)


def tiny_args(**overrides):
    """Same tiny Flash-Next geometry the other qwen4 tests build."""
    values = dict(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
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


@contextmanager
def compile_glue(enabled: bool):
    previous = qwen3_next_module.compile_glue_enabled()
    qwen3_next_module.set_qwen4_compile_glue(enabled)
    qwen3_next_module.qwen4_compile_glue_status(reset=True)
    try:
        yield
    finally:
        qwen3_next_module.set_qwen4_compile_glue(previous)


def _randomize(module, dtype):
    module.apply(lambda p: (mx.random.normal(p.shape) * 0.1).astype(dtype))
    return module


def _both(callable_, *args):
    """Return (eager, compiled) results for one glue-carrying callable."""
    with compile_glue(False):
        eager = callable_(*args)
    with compile_glue(True):
        compiled = callable_(*args)
    return eager, compiled


class TestQwen4CompiledGlue(unittest.TestCase):
    """The compiled glue may change COST ONLY: every span is bit-identical."""

    widths = (1, 3, 17)
    exact_dtypes = (mx.bfloat16,)

    def assert_identical(self, eager, compiled):
        eager = [eager] if isinstance(eager, mx.array) else list(eager)
        compiled = [compiled] if isinstance(compiled, mx.array) else list(compiled)
        self.assertEqual(len(eager), len(compiled))
        for left, right in zip(eager, compiled):
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.dtype, right.dtype)
            self.assertTrue(bool(mx.all(left == right)))

    def test_gated_residual_is_bit_identical(self):
        args = tiny_args()
        for dtype in self.exact_dtypes:
            for use_combine in (True, False):
                mx.random.seed(0)
                mixer = _randomize(GatedResidual(args, use_combine=use_combine), dtype)
                for width in self.widths:
                    hyper = (
                        mx.random.normal((2, width, args.hc_count * args.hidden_size))
                    ).astype(dtype)
                    eager, compiled = _both(mixer, hyper)
                    with self.subTest(dtype=dtype, combine=use_combine, width=width):
                        self.assert_identical(eager, compiled)

    def test_inject_apply_is_bit_identical(self):
        args = tiny_args()
        for dtype in self.exact_dtypes:
            for width in self.widths:
                mx.random.seed(width)
                residual = (
                    mx.random.normal((2, width, args.hc_count * args.hidden_size))
                ).astype(dtype)
                branch = mx.random.normal((2, width, args.hidden_size)).astype(dtype)
                inject = mx.random.normal((2, width, args.hc_count)).astype(dtype)
                eager, compiled = _both(_apply_inject, residual, branch, inject)
                with self.subTest(dtype=dtype, width=width):
                    self.assert_identical(eager, compiled)

    def test_moe_combine_span_is_bit_identical(self):
        """The span takes the sigmoid OUTPUT: a fused sigmoid is not exact."""
        for dtype in self.exact_dtypes:
            for width in self.widths:
                mx.random.seed(width)
                routed = mx.random.normal((2, width, 16)).astype(dtype)
                gate = mx.sigmoid(mx.random.normal((2, width, 1)).astype(dtype))
                shared_y = mx.random.normal((2, width, 16)).astype(dtype)
                eager = routed + gate * shared_y
                with compile_glue(True):
                    compiled = qwen3_next_module._run_glue(
                        ("moe_combine",),
                        qwen3_next_module._build_moe_combine,
                        routed,
                        gate,
                        shared_y,
                    )
                with self.subTest(dtype=dtype, width=width):
                    self.assert_identical(eager, compiled)

    def test_no_span_contains_a_sigmoid(self):
        """A fused sigmoid takes the fast Metal variant and drifts at -6.85."""
        import inspect

        for builder, args in (
            (qwen3_next_module._build_hyper_gate(4), 1),
            (qwen3_next_module._build_hyper_mix(), 2),
            (qwen3_next_module._build_inject_apply(), 3),
            (qwen3_next_module._build_moe_combine(), 3),
        ):
            source = inspect.getsource(builder)
            with self.subTest(span=builder.__name__):
                self.assertNotIn("sigmoid", source)
                self.assertEqual(
                    builder.__code__.co_argcount, args, "span arity changed"
                )

    def test_bf16_boundary_of_record_stays_eager(self):
        """x = -6.85 is the input Rapid #2912 found; the spans must not see it."""
        x = mx.array([-6.84375], dtype=mx.bfloat16)
        with compile_glue(True):
            gated = qwen3_next_module._run_glue(
                ("hyper_gate", 4),
                lambda: qwen3_next_module._build_hyper_gate(4),
                x,
            )
        self.assert_identical(nn.silu(x / 4), gated)

    def test_non_bfloat16_activations_run_eager(self):
        """fp32 keeps the fused sigmoid's 1 ULP and fp16 drifts 2.0e-3."""
        args = tiny_args()
        for dtype in (mx.float16, mx.float32):
            mx.random.seed(0)
            mixer = _randomize(GatedResidual(args), dtype)
            hyper = mx.random.normal(
                (2, 3, args.hc_count * args.hidden_size)
            ).astype(dtype)
            eager, compiled = _both(mixer, hyper)
            with self.subTest(dtype=dtype):
                self.assert_identical(eager, compiled)
                status = qwen3_next_module.qwen4_compile_glue_status()
                self.assertEqual(status["counts"]["calls"], 0)
                self.assertGreater(status["counts"]["skips"], 0)

    def test_hyper_mix_span_is_bit_identical(self):
        """The mean stays inside the span; the sigmoid is the caller's."""
        for width in self.widths:
            mx.random.seed(width)
            weights = mx.sigmoid(
                mx.random.normal((2, width, 4, 64)).astype(mx.bfloat16)
            )
            streams = mx.random.normal((2, width, 4, 64)).astype(mx.bfloat16)
            with compile_glue(True):
                compiled = qwen3_next_module._run_glue(
                    ("hyper_mix",),
                    qwen3_next_module._build_hyper_mix,
                    weights,
                    streams,
                )
            with self.subTest(width=width):
                self.assert_identical(
                    mx.mean(weights * streams, axis=-2), compiled
                )

    def test_model_forward_is_bit_identical(self):
        args = tiny_args()
        mx.random.seed(0)
        model = _randomize(TextModel(args), mx.bfloat16)
        for width in self.widths:
            inputs = mx.random.randint(0, args.vocab_size, (1, width))
            eager, compiled = _both(model, inputs)
            with self.subTest(width=width):
                self.assert_identical(eager, compiled)

    def test_status_receipts_and_flag_off_path(self):
        args = tiny_args()
        mx.random.seed(0)
        model = _randomize(TextModel(args), mx.bfloat16)
        inputs = mx.random.randint(0, args.vocab_size, (1, 3))
        with compile_glue(False):
            model(inputs)
            status = qwen3_next_module.qwen4_compile_glue_status()
            self.assertFalse(status["enabled"])
            self.assertEqual(status["counts"]["calls"], 0)
            self.assertEqual(status["counts"]["builds"], 0)
        with compile_glue(True):
            model(inputs)
            status = qwen3_next_module.qwen4_compile_glue_status()
            self.assertTrue(status["enabled"])
            self.assertGreater(status["counts"]["calls"], 0)
            self.assertEqual(status["counts"]["fallbacks"], 0)

    def test_default_is_off(self):
        self.assertFalse(qwen3_next_module._COMPILE_GLUE_DEFAULT)


if __name__ == "__main__":
    unittest.main()
