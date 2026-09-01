import inspect
import unittest

import mlx.core as mx

from mlx_lm.models._safe_reduce import sum_head_axis
from mlx_lm.models.deepseek_v32 import Indexer, ModelArgs


class TestSafeHeadReduction(unittest.TestCase):
    def test_matches_sum_with_and_without_keepdims(self):
        values = mx.arange(2 * 4 * 3 * 5, dtype=mx.float32).reshape(2, 4, 3, 5)

        reduced = sum_head_axis(values)
        kept = sum_head_axis(values, keepdims=True)
        mx.eval(reduced, kept)

        self.assertTrue(mx.array_equal(reduced, values.sum(axis=1)).item())
        self.assertTrue(
            mx.array_equal(kept, values.sum(axis=1, keepdims=True)).item()
        )

    def test_rejects_an_empty_head_axis(self):
        with self.assertRaisesRegex(ValueError, "non-empty axis 1"):
            sum_head_axis(mx.zeros((1, 0, 2, 3)))

    def test_deepseek_prefill_uses_safe_reduction(self):
        source = inspect.getsource(Indexer.__call__)
        self.assertIn("sum_head_axis(scores, keepdims=True)", source)
        self.assertIn("if s > 1", source)

    def test_deepseek_indexer_prefill_smoke(self):
        args = ModelArgs(
            hidden_size=8,
            index_head_dim=4,
            index_n_heads=2,
            index_topk=2,
            q_lora_rank=4,
            qk_rope_head_dim=2,
            max_position_embeddings=32,
        )
        indexer = Indexer(args)
        hidden = mx.random.normal((1, 4, args.hidden_size))
        query_residual = mx.random.normal((1, 4, args.q_lora_rank))

        selected = indexer(hidden, query_residual, mask=None)
        mx.eval(selected)

        self.assertEqual(selected.shape, (1, 1, 4, args.index_topk))


if __name__ == "__main__":
    unittest.main()
