import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_lm.qsa_segment_bounds import (
    build_qsa_segment_index,
    dense_qsa_topk,
    exact_segment_qsa_topk,
    profile_qsa_segment_value,
    qsa_scores,
    qsa_segment_upper_bounds,
)
from mlx_lm.models import qwen4_exp


class TestQSASegmentBounds(unittest.TestCase):
    def assert_matches_dense(self, query, keys, *, k, segment_size, valid=None):
        index = build_qsa_segment_index(keys, segment_size=segment_size)
        expected_ids, expected_scores = dense_qsa_topk(
            query, keys, k=k, valid_blocks=valid
        )
        actual = exact_segment_qsa_topk(
            query, index, k=k, valid_blocks=valid
        )
        np.testing.assert_array_equal(actual.block_ids, expected_ids)
        # BLAS can round the same dot product differently when the scored
        # matrix has a different width. Selection ids remain exact.
        np.testing.assert_allclose(actual.scores, expected_scores, rtol=1e-14, atol=1e-14)
        self.assertEqual(actual.segments_scored + actual.segments_pruned,
                         actual.segments_total)
        return actual

    def test_randomized_multihead_selection_is_exact(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            keys = rng.normal(size=(137, 16)).astype(np.float32)
            query = rng.normal(size=(4, 16)).astype(np.float32)
            for valid in (1, 17, 64, 136, 137):
                self.assert_matches_dense(
                    query, keys, k=11, segment_size=13, valid=valid
                )

    def test_clustered_keys_engage_pruning(self):
        rng = np.random.default_rng(23)
        centers = np.eye(8, dtype=np.float64)
        keys = np.concatenate(
            [center + rng.normal(scale=1e-3, size=(32, 8)) for center in centers]
        )
        query = np.repeat(centers[3:4], 3, axis=0)
        result = self.assert_matches_dense(
            query, keys, k=8, segment_size=32, valid=len(keys)
        )
        self.assertGreater(result.segments_pruned, 0)
        self.assertGreater(result.block_prune_fraction, 0.5)

    def test_zero_query_ties_disable_pruning(self):
        rng = np.random.default_rng(29)
        keys = rng.normal(size=(65, 8))
        query = np.zeros((4, 8))
        result = self.assert_matches_dense(
            query, keys, k=7, segment_size=8, valid=63
        )
        np.testing.assert_array_equal(result.block_ids, np.arange(7))
        self.assertEqual(result.blocks_scored, 63)
        self.assertEqual(result.segments_pruned, 0)

    def test_outlier_radius_prevents_unsafe_pruning(self):
        keys = np.zeros((16, 4), dtype=np.float64)
        keys[7, 0] = 100.0
        query = np.array([[1.0, 0.0, 0.0, 0.0]])
        result = self.assert_matches_dense(
            query, keys, k=1, segment_size=8, valid=16
        )
        self.assertEqual(result.block_ids.tolist(), [7])

    def test_every_segment_bound_covers_every_member_score(self):
        rng = np.random.default_rng(27)
        for _ in range(50):
            keys = rng.normal(size=(79, 17)) * 10 ** rng.uniform(-4, 4)
            query = rng.normal(size=(5, 17)) * 10 ** rng.uniform(-4, 4)
            index = build_qsa_segment_index(keys, segment_size=9)
            bounds = qsa_segment_upper_bounds(query, index)
            scores = qsa_scores(query, keys)
            for bound, start, stop in zip(bounds, index.starts, index.stops):
                self.assertTrue(np.all(scores[start:stop] <= bound))

    def test_prefix_mask_and_empty_inputs(self):
        rng = np.random.default_rng(31)
        keys = rng.normal(size=(19, 5))
        query = rng.normal(size=(2, 5))
        empty = self.assert_matches_dense(
            query, keys, k=4, segment_size=7, valid=0
        )
        self.assertEqual(empty.blocks_scored, 0)
        short = self.assert_matches_dense(
            query, keys, k=20, segment_size=7, valid=3
        )
        self.assertEqual(len(short.block_ids), 3)

    def test_score_matches_qwen4_relu_head_reduction(self):
        query = np.array([[[1.0, -2.0], [-3.0, 4.0]]]).reshape(2, 2)
        keys = np.array([[2.0, 1.0], [-1.0, 1.0]])
        expected = np.array([max(0.0, 0.0) + max(0.0, -2.0),
                             max(0.0, -3.0) + max(0.0, 7.0)]) / np.sqrt(2)
        np.testing.assert_allclose(qsa_scores(query, keys), expected)

    def test_invalid_shapes_and_nonfinite_values_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            build_qsa_segment_index(np.ones(4), segment_size=2)
        with self.assertRaisesRegex(ValueError, "positive"):
            build_qsa_segment_index(np.ones((4, 2)), segment_size=0)
        with self.assertRaisesRegex(ValueError, "finite"):
            build_qsa_segment_index([[np.nan, 0]], segment_size=1)

    def test_value_profile_reports_hot_segment_concentration(self):
        profile = profile_qsa_segment_value(
            [[0, 1, 4], [0, 2, 5], [1, 3, 6]],
            segment_size=4,
            segment_count=3,
            hot_fraction=1 / 3,
        )
        np.testing.assert_array_equal(profile.block_selection_counts, [6, 3, 0])
        np.testing.assert_array_equal(profile.query_touch_counts, [3, 3, 0])
        np.testing.assert_array_equal(profile.hot_segment_ids, [0])
        self.assertAlmostEqual(profile.hot_membership_coverage, 2 / 3)
        self.assertAlmostEqual(profile.hot_query_touch_coverage, 1 / 2)

    def test_opt_in_capture_is_bounded_and_replayable(self):
        q = mx.array(np.arange(24, dtype=np.float32).reshape(1, 1, 3, 8))
        pooled = mx.array(np.arange(80, dtype=np.float32).reshape(1, 10, 8))
        q_pos = mx.array([[39]], dtype=mx.int32)
        valid = mx.ones((1, 1, 10), dtype=mx.bool_)
        selected = mx.array([[[1, 3, 5]]], dtype=mx.uint32)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {
                "MLX_QWEN4_QSA_SEGMENT_CAPTURE_DIR": directory,
                "MLX_QWEN4_QSA_SEGMENT_CAPTURE_COUNT": "1",
            },
        ):
            qwen4_exp._QSA_SEGMENT_CAPTURE_COUNT = 0
            qwen4_exp._capture_qsa_segment_inputs(
                q, pooled, q_pos, valid, selected, layer_id=7
            )
            qwen4_exp._capture_qsa_segment_inputs(
                q, pooled, q_pos, valid, selected, layer_id=8
            )
            captures = list(Path(directory).glob("*.npz"))
            self.assertEqual(len(captures), 1)
            with np.load(captures[0]) as data:
                self.assertEqual(data["keys"].shape, (10, 8))
                self.assertEqual(data["queries"].shape, (1, 3, 8))
                self.assertEqual(data["valid_blocks"].tolist(), [10])
                self.assertEqual(data["production_selected"].tolist(), [[1, 3, 5]])
                self.assertEqual(data["layer_id"].tolist(), ["7"])

    def test_capture_layer_and_context_filters_avoid_syncing(self):
        q = mx.zeros((1, 1, 1, 2))
        pooled = mx.zeros((1, 4, 2))
        q_pos = mx.array([[15]])
        valid = mx.ones((1, 1, 4), dtype=mx.bool_)
        selected = mx.array([[[0]]])
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {
                "MLX_QWEN4_QSA_SEGMENT_CAPTURE_DIR": directory,
                "MLX_QWEN4_QSA_SEGMENT_CAPTURE_LAYERS": "3,7",
                "MLX_QWEN4_QSA_SEGMENT_CAPTURE_MIN_BLOCKS": "5",
            },
        ), patch.object(qwen4_exp.mx, "eval") as evaluate:
            qwen4_exp._QSA_SEGMENT_CAPTURE_COUNT = 0
            qwen4_exp._capture_qsa_segment_inputs(
                q, pooled, q_pos, valid, selected, layer_id=11
            )
            qwen4_exp._capture_qsa_segment_inputs(
                q, pooled, q_pos, valid, selected, layer_id=3
            )
            evaluate.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
