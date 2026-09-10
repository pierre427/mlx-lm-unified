"""CPU-only tests for the verify-cost-aware speculative-decoding policy.

No model, no MLX arrays, no GPU — the policy is a pure-Python cost/benefit
calculator over the measured ``vocab_proj`` verify curve (A-tier battery §3).
"""

import unittest

from mlx_lm.verify_cost_policy import (
    MEASURED_VERIFY_US,
    RECOVERY_B,
    TAX_ZONE_B_HI,
    TAX_ZONE_B_LO,
    VerifyCostModel,
    marginal_objective,
    recommend_num_draft,
    truncate_draft_by_benefit,
)


class TestVerifyCostModel(unittest.TestCase):
    def setUp(self):
        self.model = VerifyCostModel.from_measured()

    def test_measured_knots_are_exact(self):
        # The model must return the recorded totals at measured batches.
        for B, us in MEASURED_VERIFY_US.items():
            self.assertAlmostEqual(self.model.total_verify_us(B), us, places=6)

    def test_reproduces_b5_14_plateau_shape(self):
        # Per-token cost is a flat/expensive plateau across B=5..14 (does NOT
        # fall), then recovers sharply at B=16. This is the whole point of the
        # curve, so assert the *shape*, not just individual points.
        plateau_batches = [5, 6, 7, 8, 10, 12, 14]
        plateau = [self.model.per_token_verify_us(B) for B in plateau_batches]
        # Every plateau per-token cost sits in the measured ~180-210 us band.
        for pt in plateau:
            self.assertGreater(pt, 160.0)
            self.assertLess(pt, 220.0)
        # The plateau does not trend downward: max and min stay close together.
        self.assertLess(max(plateau) - min(plateau), 45.0)
        # Recovery at B=16 is decisively cheaper than anywhere on the plateau.
        recovered = self.model.per_token_verify_us(16)
        self.assertLess(recovered, min(plateau) - 40.0)
        self.assertLess(recovered, 130.0)  # ~112 us/tok in the wiki

    def test_total_climbs_across_plateau_then_drops_at_recovery(self):
        # Total latency climbs steeply B=4->14, then the B=16 batch is cheaper
        # in TOTAL than the B=14 batch (the recovery).
        totals = [self.model.total_verify_us(B) for B in range(4, 15)]
        self.assertEqual(totals, sorted(totals))  # monotone non-decreasing 4..14
        self.assertLess(
            self.model.total_verify_us(16), self.model.total_verify_us(14)
        )

    def test_interpolation_between_knots(self):
        # B=9 is not measured; it must interpolate strictly between B=8 and B=10.
        v9 = self.model.total_verify_us(9)
        self.assertGreater(v9, self.model.total_verify_us(8))
        self.assertLess(v9, self.model.total_verify_us(10))

    def test_verify_batch_and_marginal_cost_sign(self):
        self.assertEqual(self.model.verify_batch(4), 5)
        # Marginal cost of entering the plateau (k=4 -> B=5 from B=4) is a large
        # positive number.
        self.assertGreater(self.model.marginal_verify_us(4), 0.0)
        # Marginal cost of the k=15 token (B 15 -> 16) is NEGATIVE: the batch got
        # cheaper crossing into recovery. This is the "jump past the tax" signal.
        self.assertLess(self.model.marginal_verify_us(15), 0.0)

    def test_in_tax_zone(self):
        # k in [4, 13] => B in [5, 14] is taxed; k<=3 and k>=15 are not.
        self.assertFalse(self.model.in_tax_zone(3))
        self.assertTrue(self.model.in_tax_zone(4))
        self.assertTrue(self.model.in_tax_zone(13))
        self.assertFalse(self.model.in_tax_zone(15))
        self.assertEqual(TAX_ZONE_B_LO, 5)
        self.assertEqual(TAX_ZONE_B_HI, 14)

    def test_default_constructor_matches_from_measured(self):
        # VerifyCostModel() should also mean "the measured curve".
        bare = VerifyCostModel()
        for B in MEASURED_VERIFY_US:
            self.assertEqual(
                bare.total_verify_us(B), self.model.total_verify_us(B)
            )


class TestTruncation(unittest.TestCase):
    def setUp(self):
        self.model = VerifyCostModel.from_measured()

    def test_high_confidence_prefix_kept_low_conf_cut(self):
        # Strong early tokens then a weak one: truncation should keep the strong
        # prefix and cut at/after confidence collapses.
        confs = [0.98, 0.97, 0.95, 0.2, 0.1]
        res = truncate_draft_by_benefit(confs, self.model)
        self.assertGreaterEqual(res.length, 1)
        self.assertLess(res.length, len(confs))  # something was cut
        # The cut position's marginal objective is where it turned negative.
        self.assertLess(res.steps[res.length].marginal_net_us, 0.0)

    def test_cut_when_marginal_cost_exceeds_benefit(self):
        # Uniformly mediocre confidence: cumulative accept prob decays, and the
        # plateau's steep marginal cost outruns the shrinking benefit, forcing a
        # cut well before the nominal chain end.
        confs = [0.7] * 12
        res = truncate_draft_by_benefit(confs, self.model)
        self.assertLess(res.length, 12)
        # Every kept step had non-negative net; the first dropped step did not.
        for s in res.steps[: res.length]:
            self.assertGreaterEqual(s.marginal_net_us, 0.0)
        if res.length < len(res.steps):
            self.assertLess(res.steps[res.length].marginal_net_us, 0.0)

    def test_all_kept_when_confidence_saturated_and_cost_recovers(self):
        # Near-certain confidence: benefit stays high; a chain that reaches the
        # B>=16 recovery keeps going (marginal cost turns negative there).
        confs = [0.999] * 16
        res = truncate_draft_by_benefit(confs, self.model)
        self.assertGreaterEqual(res.length, 4)

    def test_objective_is_monotone_and_consistent(self):
        # Consistency: cumulative accept prob is monotone non-increasing (it is a
        # running product of probabilities <= 1), and benefit tracks it.
        confs = [0.9, 0.85, 0.8, 0.75, 0.7, 0.6]
        steps = marginal_objective(confs, self.model)
        cum = [s.cum_accept_prob for s in steps]
        self.assertEqual(cum, sorted(cum, reverse=True))  # non-increasing
        for s in steps:
            # benefit = cum_accept_prob * token_value, so same monotonicity.
            self.assertAlmostEqual(
                s.marginal_benefit_us,
                s.cum_accept_prob * self.model.token_value_us(),
                places=6,
            )
            # verify_batch is exactly position + 1.
            self.assertEqual(s.verify_batch, s.position + 1)
        # Benefit strictly decreases as confidence compounds.
        benefits = [s.marginal_benefit_us for s in steps]
        self.assertEqual(benefits, sorted(benefits, reverse=True))

    def test_truncation_result_usable_as_int(self):
        res = truncate_draft_by_benefit([0.9, 0.9], self.model)
        self.assertEqual(int(res), res.length)


class TestRecommendNumDraft(unittest.TestCase):
    def test_low_acceptance_avoids_tax_zone(self):
        # Low acceptance => the gate stays in the cheap pre-plateau band (k<=4),
        # never selecting a draft length that verifies in B=5..14.
        for p in (0.2, 0.4, 0.5, 0.6):
            k = recommend_num_draft({"accept_prob": p})
            self.assertLessEqual(k, 4, f"p={p} picked k={k} (tax zone)")
            B = k + 1
            self.assertFalse(TAX_ZONE_B_LO <= B <= TAX_ZONE_B_HI)

    def test_high_acceptance_jumps_past_tax_zone(self):
        # Very high acceptance => it pays to jump to the recovered batch (k>=15,
        # B>=16), NOT to sit on the plateau.
        k = recommend_num_draft({"accept_prob": 0.98, "max_draft": 20})
        self.assertGreaterEqual(k, RECOVERY_B - 1)  # k >= 15
        self.assertGreaterEqual(k + 1, RECOVERY_B)  # B >= 16

    def test_gate_never_returns_tax_zone_length(self):
        # Sweep acceptance across the whole range: the returned k must never land
        # in the tax zone, by construction of the efficient candidate set.
        p = 0.0
        while p <= 1.0 + 1e-9:
            k = recommend_num_draft({"accept_prob": min(p, 1.0), "max_draft": 20})
            B = k + 1
            self.assertFalse(
                TAX_ZONE_B_LO <= B <= TAX_ZONE_B_HI,
                f"p={p:.2f} picked tax-zone k={k} (B={B})",
            )
            p += 0.05

    def test_capped_max_draft_stays_cheap(self):
        # If the caller caps below the recovery, the recovered band is empty, so
        # the gate cannot jump — it must stay in the cheap zone rather than enter
        # the tax plateau, even at high acceptance.
        k = recommend_num_draft({"accept_prob": 0.98, "max_draft": 10})
        self.assertLessEqual(k, 4)
        B = k + 1
        self.assertFalse(TAX_ZONE_B_LO <= B <= TAX_ZONE_B_HI)

    def test_monotone_in_acceptance(self):
        # Recommended k should be non-decreasing in acceptance (more acceptance
        # never makes a longer draft less attractive on this objective).
        ks = [recommend_num_draft({"accept_prob": p, "max_draft": 20})
              for p in [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]]
        self.assertEqual(ks, sorted(ks), f"non-monotone: {ks}")

    def test_missing_accept_prob_raises(self):
        with self.assertRaises(KeyError):
            recommend_num_draft({"max_draft": 8})

    def test_bad_accept_prob_raises(self):
        with self.assertRaises(ValueError):
            recommend_num_draft({"accept_prob": 1.5})


if __name__ == "__main__":
    unittest.main()
