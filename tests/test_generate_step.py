import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generate_step


class DummyLM(nn.Module):
    def __init__(self, vocab_size=16):
        super().__init__()
        self.layers = []
        self.vocab_size = vocab_size

    def make_cache(self):
        return []

    def __call__(self, tokens, cache=None, input_embeddings=None):
        sequence = tokens if input_embeddings is None else input_embeddings
        return mx.zeros((sequence.shape[0], sequence.shape[1], self.vocab_size))


class TestGenerateStepLogitsProcessorHistory(unittest.TestCase):
    def _first_history(self, *, input_embeddings=None):
        histories = []

        def record_history(tokens, logits):
            histories.append(tokens.tolist())
            return logits

        list(
            generate_step(
                mx.array([2, 5, 7, 9]),
                DummyLM(),
                max_tokens=1,
                prefill_step_size=2,
                input_embeddings=input_embeddings,
                logits_processors=[record_history],
            )
        )
        self.assertTrue(histories)
        return histories[0]

    def test_processor_sees_prefilled_prompt_tokens(self):
        self.assertEqual(self._first_history(), [2, 5, 7, 9])

    def test_processor_history_uses_token_ids_with_input_embeddings(self):
        embeddings = mx.zeros((4, 3))
        self.assertEqual(
            self._first_history(input_embeddings=embeddings),
            [2, 5, 7, 9],
        )


if __name__ == "__main__":
    unittest.main()
