"""CPU-only contracts for confidence-gated draft scheduling."""

import unittest

from mlx_lm.confidence_draft_calibration import (
    ConfidenceGate,
    DraftConfidenceTrace,
    IsotonicCalibrator,
    aggregate_oracle_bound,
    calibration_samples,
    choose_expected_value_prefix,
    evaluate_policy,
    prefix_scores,
)
from mlx_lm.verify_cost_policy import VerifyCostModel


class TestConfidenceDraftCalibration(unittest.TestCase):
    def setUp(self):
        self.trace = DraftConfidenceTrace(
            probability=(0.95, 0.80, 0.30),
            margin=(0.70, 0.40, 0.05),
            entropy=(0.10, 0.25, 0.85),
            accepted_length=2,
        )
        self.model = VerifyCostModel.from_measured()

    def test_gate_is_default_off(self):
        gate = ConfidenceGate(metric="probability", threshold=0.99)
        self.assertFalse(gate.enabled)
        self.assertEqual(gate.select(self.trace), self.trace.depth)

    def test_fixed_gates_only_keep_consecutive_prefix(self):
        self.assertEqual(
            ConfidenceGate("probability", 0.75, True).select(self.trace), 2
        )
        self.assertEqual(ConfidenceGate("margin", 0.5, True).select(self.trace), 1)
        self.assertEqual(ConfidenceGate("entropy", 0.7, True).select(self.trace), 2)
        joint = prefix_scores(self.trace, "joint_probability")
        self.assertEqual(joint[:2], (0.95, 0.76))
        self.assertAlmostEqual(joint[2], 0.228)

    def test_isotonic_fit_is_monotone(self):
        calibrator = IsotonicCalibrator.fit(
            [0.1, 0.2, 0.3, 0.4, 0.8, 0.9],
            [0, 1, 0, 1, 1, 1],
        )
        predictions = [calibrator.predict(value / 100.0) for value in range(101)]
        self.assertEqual(predictions, sorted(predictions))

    def test_labels_are_prefix_acceptance_not_independent_tokens(self):
        scores, labels = calibration_samples([self.trace], "probability")
        self.assertEqual(len(scores), 3)
        self.assertEqual(labels, [1, 1, 0])

    def test_every_committed_token_has_a_target_verify_row(self):
        evaluation = evaluate_policy(
            [self.trace], ConfidenceGate("probability", 0.75, True).select
        )
        self.assertEqual(evaluation.committed_tokens, 3)
        self.assertEqual(evaluation.target_verified_rows, 3)
        self.assertEqual(evaluation.unverified_committed_tokens, 0)

    def test_shortening_discards_work_not_authority(self):
        evaluation = evaluate_policy(
            [self.trace], ConfidenceGate("probability", 0.9, True).select
        )
        self.assertEqual(evaluation.committed_tokens, 2)
        self.assertEqual(evaluation.accepted_tokens_discarded, 1)
        self.assertEqual(evaluation.unverified_committed_tokens, 0)

    def test_expected_value_can_choose_plain_or_full(self):
        low = IsotonicCalibrator([1.0], [0.0])
        high = IsotonicCalibrator([1.0], [1.0])
        self.assertEqual(
            choose_expected_value_prefix(
                self.trace, low, metric="probability", verify_cost_model=self.model
            ),
            0,
        )
        self.assertEqual(
            choose_expected_value_prefix(
                self.trace, high, metric="probability", verify_cost_model=self.model
            ),
            self.trace.depth,
        )

    def test_current_k2_aggregate_oracle_is_below_one_percent(self):
        bound = aggregate_oracle_bound(
            proposed=196, accepted=157, max_draft=2, verify_cost_model=self.model
        )
        self.assertEqual(bound.cycles, 98)
        self.assertAlmostEqual(bound.min_saving_us, 435.0)
        self.assertAlmostEqual(bound.max_saving_us, 663.0)
        self.assertLess(bound.max_saving_fraction, 0.01)

    def test_invalid_trace_and_selector_fail_closed(self):
        with self.assertRaises(ValueError):
            DraftConfidenceTrace((0.8,), (0.2,), (0.3,), 2)
        with self.assertRaises(ValueError):
            evaluate_policy([self.trace], lambda trace: trace.depth + 1)


if __name__ == "__main__":
    unittest.main()
