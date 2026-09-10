import math
import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models.qwen4_qsa_stage1 import (
    qsa_stage1_kernel_cache_info,
    qsa_stage1_score_producer,
    qsa_stage1_select,
)


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class TestQSAStage1Metal(unittest.TestCase):
    @staticmethod
    def oracle(q, pooled, q_positions, *, topk, ratio):
        blocks = pooled.shape[1]
        scores = mx.einsum(
            "blhd,bnd->blnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(q.shape[-1])
        starts = mx.arange(blocks) * ratio
        valid = (starts + ratio - 1)[None, None, :] <= q_positions[..., None]
        scores = mx.where(valid, scores, -mx.inf)
        ids = mx.argpartition(scores, kth=blocks - topk, axis=-1)[..., -topk:]
        chosen = mx.put_along_axis(
            mx.zeros(valid.shape, dtype=mx.bool_), ids, mx.array(True), axis=-1
        )
        return chosen & valid, valid

    def assert_selection_equal(self, q, pooled, q_positions, *, topk=4, ratio=4):
        native_ids = qsa_stage1_select(
            q,
            pooled,
            q_positions,
            block_topk=topk,
            compress_ratio=ratio,
        )
        eager, valid = self.oracle(
            q, pooled, q_positions, topk=topk, ratio=ratio
        )
        native = mx.put_along_axis(
            mx.zeros(valid.shape, dtype=mx.bool_),
            native_ids,
            mx.array(True),
            axis=-1,
        ) & valid
        mx.eval(native_ids, native, eager)
        self.assertTrue(np.array_equal(np.asarray(native), np.asarray(eager)))
        return np.asarray(native_ids)

    def test_b1_b2_padding_tails_and_ties(self):
        rng = np.random.default_rng(7)
        q = mx.array(rng.normal(size=(2, 5, 3, 8)).astype(np.float16))
        pooled = mx.array(rng.normal(size=(2, 17, 8)).astype(np.float16))
        q_positions = mx.array(
            [[-1, 3, 5, 15, 67], [3, 7, 11, 31, 63]], dtype=mx.int32
        )
        self.assert_selection_equal(q, pooled, q_positions)

        zero_q = mx.zeros((2, 3, 3, 8), dtype=mx.float16)
        tie_positions = mx.array([[3, 19, 67], [7, 23, 63]], dtype=mx.int32)
        ids = self.assert_selection_equal(zero_q, pooled, tie_positions)
        self.assertTrue(np.array_equal(ids[0, -1], np.array([13, 14, 15, 16])))
        self.assertTrue(np.array_equal(ids[1, -1], np.array([12, 13, 14, 15])))

    def test_invisible_block_perturbation_is_neutral(self):
        rng = np.random.default_rng(11)
        q = mx.array(rng.normal(size=(1, 2, 4, 16)).astype(np.float16))
        pooled = mx.array(rng.normal(size=(1, 20, 16)).astype(np.float16))
        q_positions = mx.array([[19, 23]], dtype=mx.int32)
        base = qsa_stage1_select(
            q, pooled, q_positions, block_topk=4, compress_ratio=4
        )
        changed = mx.concatenate(
            [pooled[:, :6], mx.full_like(pooled[:, 6:], 4096)], axis=1
        )
        perturbed = qsa_stage1_select(
            q, changed, q_positions, block_topk=4, compress_ratio=4
        )
        mx.eval(base, perturbed)
        self.assertTrue(np.array_equal(np.asarray(base), np.asarray(perturbed)))

    def test_depth_ladder_keeps_one_template(self):
        rng = np.random.default_rng(13)
        q = mx.array(rng.normal(size=(1, 1, 2, 8)).astype(np.float16))
        template_count = None
        for tokens in (16_384, 32_768, 65_536, 98_304):
            blocks = tokens // 4
            pooled = mx.array(
                rng.normal(size=(1, blocks, 8)).astype(np.float16)
            )
            q_positions = mx.array([[tokens - 1]], dtype=mx.int32)
            self.assert_selection_equal(
                q, pooled, q_positions, topk=512, ratio=4
            )
            current = qsa_stage1_kernel_cache_info().currsize
            if template_count is None:
                template_count = current
            else:
                self.assertEqual(current, template_count)

    def test_production_mpp_scorer_matches_eager_selection(self):
        rng = np.random.default_rng(17)
        q = mx.array(rng.normal(size=(1, 16, 4, 128)).astype(np.float16))
        pooled = mx.array(rng.normal(size=(1, 1024, 128)).astype(np.float16))
        q_positions = mx.arange(4096 - 16, 4096, dtype=mx.int32)[None, :]
        self.assertEqual(qsa_stage1_score_producer(q, pooled), "mpp_exact_band")
        self.assert_selection_equal(
            q, pooled, q_positions, topk=512, ratio=4
        )


if __name__ == "__main__":
    unittest.main()
