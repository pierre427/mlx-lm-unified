import types
import unittest

import mlx.core as mx

from mlx_lm.models.dream import _select_transfer_positions, diffusion_generate


class _DreamStub:
    def __init__(self, mask_token_id=9, vocab_size=12):
        self.args = types.SimpleNamespace(mask_token_id=mask_token_id)
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        scores = mx.arange(self.vocab_size, dtype=mx.float32)
        return mx.broadcast_to(scores, (*x.shape, self.vocab_size))


class TestDreamScheduleBoundaries(unittest.TestCase):
    """Finding D4: schedule/stats must not fail open at zero work."""

    def test_zero_and_negative_steps_raise(self):
        model = _DreamStub()
        inputs = mx.array([[1, 9, 2, 9]])
        for bad_steps in (0, -1):
            with self.assertRaises(ValueError):
                diffusion_generate(model, inputs, max_length=5, steps=bad_steps)

    def test_no_new_token_input_early_returns_with_zero_stats(self):
        # max_length == prompt length AND no masked positions: nothing to
        # denoise. Must early-return BEFORE any model call with honest zeros
        # (no phantom forward, no clamped tokens_per_step_mean=1.0).
        model = _DreamStub()
        inputs = mx.array([[1, 2, 3, 4]])  # no mask id (9) present
        output, stats = diffusion_generate(
            model, inputs, max_length=4, steps=8, return_stats=True
        )
        self.assertEqual(model.calls, 0, "model was called on a no-work input")
        self.assertEqual(stats["forwards"], 0)
        self.assertEqual(stats["tokens_per_step_mean"], 0.0)
        self.assertEqual(output.tolist(), [[1, 2, 3, 4]])

    def test_stats_are_exact_not_clamped(self):
        # A one-step fill of a single masked position: forwards==1 and mean is
        # the true generated/forwards ratio, not a max(1, ...) fabrication.
        model = _DreamStub()
        inputs = mx.array([[1, 2, 3]])
        _, stats = diffusion_generate(
            model, inputs, max_length=4, steps=1, alg="maskgit_plus",
            return_stats=True,
        )
        self.assertEqual(stats["forwards"], 1)
        self.assertEqual(stats["tokens_per_step_mean"], 1.0)


class TestDreamRemasking(unittest.TestCase):
    def test_non_origin_algorithms_handle_different_mask_counts_per_row(self):
        model = _DreamStub()
        inputs = mx.array([[1, 9, 2, 9], [1, 9, 9, 9]])
        for alg in ("maskgit_plus", "topk_margin", "entropy"):
            output = diffusion_generate(
                model, inputs, max_length=5, steps=1, alg=alg
            )
            self.assertEqual(output.shape, (2, 5))
            self.assertFalse(mx.any(output == 9).item())
            self.assertEqual(output[:, 0].tolist(), [1, 1])
            self.assertEqual(output[0, 2].item(), 2)

    def test_temperature_selection_is_batched_and_without_replacement(self):
        mask = mx.array(
            [[True, True, False, False], [True, True, True, True]]
        )
        confidence = mx.array(
            [[0.1, 0.2, 100.0, 100.0], [0.1, 0.2, 0.3, 0.4]]
        )
        transfer = _select_transfer_positions(
            confidence, mask, mx.array([1, 3]), alg_temp=0.5
        )
        self.assertEqual(mx.sum(transfer, axis=-1).tolist(), [1, 3])
        self.assertFalse(mx.any(transfer & ~mask).item())

    def test_diffusion_temperature_path_runs_for_a_batch(self):
        output, stats = diffusion_generate(
            _DreamStub(),
            mx.array([[1, 9, 2, 9], [1, 9, 9, 9]]),
            max_length=5,
            steps=2,
            alg="maskgit_plus",
            alg_temp=0.5,
            return_stats=True,
        )
        self.assertFalse(mx.any(output == 9).item())
        self.assertEqual(stats["transferred_total"], 7)


if __name__ == "__main__":
    unittest.main()
