"""Unit tests for the ``proof_battery`` anti-rigging checks.

Exit criterion: every detector must FAIL CLOSED on a deliberately rigged input
*before* any expensive model run. So each check is exercised with both a
synthetic pass case and a synthetic fail case that proves the detector actually
fires. No model weights, downloads, or GPU — pure CPU fixtures.
"""

import math
import unittest

from mlx_lm.proof_battery import (
    ControlResult,
    ProofBatteryError,
    RobustnessResult,
    assert_finish_reason_and_token_count,
    assert_same_norm,
    assert_token_path_equal,
    batched_vs_stepwise_greedy,
    eviction_reinsertion_identity,
    interleaving_invariant,
    is_finite_number,
    l2_norm,
    logging_parity,
    same_norm_random_direction,
    state_accessor_mutation_probe,
    varied_filler_paraphrase_controls,
    wrong_number_and_alien_controls,
)


class TestNumericPrimitives(unittest.TestCase):
    def test_is_finite_number(self):
        self.assertTrue(is_finite_number(1))
        self.assertTrue(is_finite_number(-2.5))
        self.assertFalse(is_finite_number(True))  # bool rejected
        self.assertFalse(is_finite_number(float("nan")))
        self.assertFalse(is_finite_number(float("inf")))
        self.assertFalse(is_finite_number("3"))
        self.assertFalse(is_finite_number(None))

    def test_l2_norm_pass(self):
        self.assertAlmostEqual(l2_norm([3.0, 4.0]), 5.0)

    def test_l2_norm_fails_closed_empty(self):
        with self.assertRaises(ProofBatteryError):
            l2_norm([])

    def test_l2_norm_fails_closed_nan(self):
        with self.assertRaises(ProofBatteryError):
            l2_norm([1.0, float("nan")])


class TestSameNormRandomDirection(unittest.TestCase):
    def test_pass_norm_preserved_and_deterministic(self):
        vec = [1.0, -2.0, 3.0, 0.5]
        d1 = same_norm_random_direction(vec, seed=7)
        d2 = same_norm_random_direction(vec, seed=7)
        self.assertEqual(d1, d2)  # deterministic in seed
        assert_same_norm(vec, d1)  # same magnitude — the control invariant
        # A different seed gives a different direction.
        d3 = same_norm_random_direction(vec, seed=8)
        self.assertNotEqual(d1, d3)

    def test_fail_closed_zero_norm(self):
        # A rigged "control" for a zero vector has no magnitude to preserve.
        with self.assertRaises(ProofBatteryError):
            same_norm_random_direction([0.0, 0.0, 0.0], seed=1)

    def test_fail_closed_nonfinite_input(self):
        with self.assertRaises(ProofBatteryError):
            same_norm_random_direction([1.0, float("inf")], seed=1)

    def test_assert_same_norm_fires_on_mismatch(self):
        # DETECTOR FIRES: a control that is NOT magnitude-matched is rejected.
        with self.assertRaises(ProofBatteryError):
            assert_same_norm([3.0, 4.0], [30.0, 40.0])


class TestTokenPathEquality(unittest.TestCase):
    def test_pass_identical_paths(self):
        assert_token_path_equal([1, 2, 3], [1, 2, 3])

    def test_fail_closed_divergence(self):
        # DETECTOR FIRES: the Vegas FP-near-tie lesson — differing token ids.
        with self.assertRaises(ProofBatteryError):
            assert_token_path_equal([1, 2, 3], [1, 2, 4])

    def test_fail_closed_length_mismatch(self):
        with self.assertRaises(ProofBatteryError):
            assert_token_path_equal([1, 2, 3], [1, 2])

    def test_fail_closed_empty(self):
        with self.assertRaises(ProofBatteryError):
            assert_token_path_equal([], [])

    def test_fail_closed_non_int_tokens(self):
        # Float "tokens" (e.g. logits) must be rejected as ambiguous.
        with self.assertRaises(ProofBatteryError):
            assert_token_path_equal([1.0, 2.0], [1.0, 2.0])

    def test_batched_vs_stepwise_pass(self):
        path = [5, 9, 2, 7]
        batched_vs_stepwise_greedy(lambda: list(path), lambda: list(path))

    def test_batched_vs_stepwise_fires_on_divergence(self):
        # DETECTOR FIRES: a "lossless" claim where the paths actually differ.
        with self.assertRaises(ProofBatteryError):
            batched_vs_stepwise_greedy(lambda: [5, 9, 2, 7], lambda: [5, 9, 2, 8])

    def test_batched_vs_stepwise_fails_closed_on_exception(self):
        def boom():
            raise RuntimeError("decode crashed")

        with self.assertRaises(ProofBatteryError):
            batched_vs_stepwise_greedy(boom, lambda: [1, 2])


class TestFinishReasonAndTokenCount(unittest.TestCase):
    def test_pass_mapping_with_count(self):
        result = {"finish_reason": "stop", "n_tokens": 12}
        assert_finish_reason_and_token_count(result, "stop", 12)

    def test_pass_mapping_with_token_list(self):
        result = {"finish_reason": "stop", "tokens": [1, 2, 3]}
        assert_finish_reason_and_token_count(result, "stop", 3)

    def test_pass_object_attributes(self):
        class R:
            finish_reason = "length"
            n_tokens = 256

        assert_finish_reason_and_token_count(R(), "length", 256)

    def test_fail_closed_wrong_reason(self):
        # DETECTOR FIRES: hit the length cap instead of a natural stop.
        with self.assertRaises(ProofBatteryError):
            assert_finish_reason_and_token_count(
                {"finish_reason": "length", "n_tokens": 12}, "stop", 12
            )

    def test_fail_closed_wrong_count(self):
        with self.assertRaises(ProofBatteryError):
            assert_finish_reason_and_token_count(
                {"finish_reason": "stop", "n_tokens": 11}, "stop", 12
            )

    def test_fail_closed_missing_field(self):
        with self.assertRaises(ProofBatteryError):
            assert_finish_reason_and_token_count({"n_tokens": 12}, "stop", 12)

    def test_fail_closed_non_string_reason(self):
        with self.assertRaises(ProofBatteryError):
            assert_finish_reason_and_token_count(
                {"finish_reason": 0, "n_tokens": 12}, "stop", 12
            )


class TestWrongNumberAndAlienControls(unittest.TestCase):
    @staticmethod
    def _grounded_measure(payload):
        # A genuinely grounded system: high signal only for the correct fact.
        return {"correct": 0.9, "wrong": 0.1, "alien": 0.05}[payload]

    @staticmethod
    def _parroting_measure(payload):
        # A rigged system that answers the same regardless of context.
        return 0.9

    def test_pass_discriminates(self):
        res = wrong_number_and_alien_controls(
            self._grounded_measure, "correct", "wrong", "alien", margin=0.1
        )
        self.assertIsInstance(res, ControlResult)
        self.assertTrue(res.discriminates)

    def test_fires_on_parroting(self):
        # DETECTOR FIRES: no discrimination between correct/wrong/alien.
        res = wrong_number_and_alien_controls(
            self._parroting_measure, "correct", "wrong", "alien", margin=0.1
        )
        self.assertFalse(res.discriminates)
        self.assertIn("did not exceed", res.reason)

    def test_fails_closed_on_measure_exception(self):
        def boom(payload):
            raise ValueError("model error")

        res = wrong_number_and_alien_controls(boom, "c", "w", "a")
        self.assertFalse(res.discriminates)

    def test_fails_closed_on_nan(self):
        res = wrong_number_and_alien_controls(
            lambda p: float("nan"), "c", "w", "a"
        )
        self.assertFalse(res.discriminates)


class TestVariedFillerParaphraseControls(unittest.TestCase):
    def test_pass_robust_across_all_cells(self):
        # A real retrieval mechanism succeeds regardless of phrasing/filler.
        res = varied_filler_paraphrase_controls(
            lambda q, f: True,
            paraphrases=["where is the key?", "locate the key", "key location?"],
            fillers=["short", "medium filler text", "very long filler " * 20],
        )
        self.assertIsInstance(res, RobustnessResult)
        self.assertTrue(res.robust)
        self.assertEqual(res.n_cells, 9)
        self.assertEqual(res.n_passed, 9)

    def test_fires_on_rigged_needle(self):
        # DETECTOR FIRES: only the ORIGINAL phrasing on the SHORT filler passes
        # — the classic rigged-needle / lexical-overlap artifact.
        def measure(q, f):
            return q == "where is the key?" and f == "short"

        res = varied_filler_paraphrase_controls(
            measure,
            paraphrases=["where is the key?", "locate the key"],
            fillers=["short", "long filler"],
        )
        self.assertFalse(res.robust)
        self.assertEqual(res.n_passed, 1)
        self.assertEqual(len(res.failures), 3)

    def test_fails_closed_on_empty_axis(self):
        res = varied_filler_paraphrase_controls(
            lambda q, f: True, paraphrases=["q"], fillers=[]
        )
        self.assertFalse(res.robust)

    def test_fails_closed_on_measure_exception(self):
        def boom(q, f):
            raise RuntimeError("boom")

        res = varied_filler_paraphrase_controls(
            boom, paraphrases=["q"], fillers=["f"]
        )
        self.assertFalse(res.robust)

    def test_numeric_score_with_default_grader(self):
        # score >= 1.0 passes; a NaN score fails closed.
        res_ok = varied_filler_paraphrase_controls(
            lambda q, f: 1.0, paraphrases=["q"], fillers=["f"]
        )
        self.assertTrue(res_ok.robust)
        res_nan = varied_filler_paraphrase_controls(
            lambda q, f: float("nan"), paraphrases=["q"], fillers=["f"]
        )
        self.assertFalse(res_nan.robust)


class TestStateAccessorMutationProbe(unittest.TestCase):
    def test_pass_all_accessors_live(self):
        # Every accessor is a live view of the same underlying dict.
        state = {"offset": 0}
        accessors = [
            lambda: state["offset"],
            lambda: dict(state)["offset"],
        ]

        def mutate():
            state["offset"] += 1

        state_accessor_mutation_probe(accessors, mutate)

    def test_fires_on_detached_copy(self):
        # DETECTOR FIRES: one accessor captured a stale snapshot at build time
        # and never reflects the mutation.
        state = {"offset": 0}
        stale_snapshot = state["offset"]  # frozen copy
        accessors = [
            lambda: state["offset"],       # live
            lambda: stale_snapshot,        # detached — the bug
        ]

        def mutate():
            state["offset"] += 5

        with self.assertRaises(ProofBatteryError):
            state_accessor_mutation_probe(accessors, mutate)

    def test_fires_on_noop_mutate(self):
        # DETECTOR FIRES: mutate is a silent no-op, so state never changes.
        state = {"offset": 0}
        with self.assertRaises(ProofBatteryError):
            state_accessor_mutation_probe([lambda: state["offset"]], lambda: None)

    def test_fails_closed_empty_accessors(self):
        with self.assertRaises(ProofBatteryError):
            state_accessor_mutation_probe([], lambda: None)


class TestLoggingParity(unittest.TestCase):
    def test_pass_identical_output(self):
        logging_parity(lambda: [1, 2, 3], lambda: [1, 2, 3])

    def test_fires_when_logging_perturbs_output(self):
        # DETECTOR FIRES: enabling telemetry changes the token path.
        with self.assertRaises(ProofBatteryError):
            logging_parity(lambda: [1, 2, 3, 99], lambda: [1, 2, 3])

    def test_fails_closed_on_exception(self):
        def boom():
            raise RuntimeError("logger threw")

        with self.assertRaises(ProofBatteryError):
            logging_parity(boom, lambda: [1])

    def test_custom_equality(self):
        logging_parity(
            lambda: 1.0000001,
            lambda: 1.0,
            equal=lambda a, b: math.isclose(a, b, abs_tol=1e-3),
        )


class TestEvictionReinsertionIdentity(unittest.TestCase):
    def _make_cache(self):
        cache = {"k": [1, 2, 3]}

        def read(key):
            return cache.get(key)

        def evict(key):
            cache.pop(key, None)

        def reinsert(key, value):
            cache[key] = value

        return cache, read, evict, reinsert

    def test_pass_identity_restored(self):
        _, read, evict, reinsert = self._make_cache()
        eviction_reinsertion_identity(read, evict, reinsert, "k")

    def test_fires_on_corrupted_reinsert(self):
        # DETECTOR FIRES: reinsertion does not reconstruct the original state.
        cache, read, evict, _ = self._make_cache()

        def bad_reinsert(key, value):
            cache[key] = value + [999]  # corruption

        with self.assertRaises(ProofBatteryError):
            eviction_reinsertion_identity(read, evict, bad_reinsert, "k")

    def test_fires_on_fake_eviction(self):
        # DETECTOR FIRES: evict is a no-op, so the reinsertion test is hollow.
        cache, read, _, reinsert = self._make_cache()

        def fake_evict(key):
            pass  # entry remains

        with self.assertRaises(ProofBatteryError):
            eviction_reinsertion_identity(read, fake_evict, reinsert, "k")

    def test_fails_closed_absent_key(self):
        _, read, evict, reinsert = self._make_cache()
        with self.assertRaises(ProofBatteryError):
            eviction_reinsertion_identity(read, evict, reinsert, "missing")


class TestInterleavingInvariant(unittest.TestCase):
    def test_pass_balanced_claim_release(self):
        # A ledger where claims and releases interleave but never go negative.
        ledger = {"outstanding": 0}
        steps = [
            lambda: ledger.__setitem__("outstanding", ledger["outstanding"] + 1),
            lambda: ledger.__setitem__("outstanding", ledger["outstanding"] + 1),
            lambda: ledger.__setitem__("outstanding", ledger["outstanding"] - 1),
            lambda: ledger.__setitem__("outstanding", ledger["outstanding"] - 1),
        ]
        interleaving_invariant(steps, lambda: ledger["outstanding"] >= 0)

    def test_fires_on_invariant_violation(self):
        # DETECTOR FIRES: an error path double-releases and the count goes
        # negative — a corruption only this interleaving exposes.
        ledger = {"outstanding": 0}
        steps = [
            lambda: ledger.__setitem__("outstanding", ledger["outstanding"] + 1),
            lambda: ledger.__setitem__("outstanding", ledger["outstanding"] - 1),
            lambda: ledger.__setitem__("outstanding", ledger["outstanding"] - 1),
        ]
        with self.assertRaises(ProofBatteryError):
            interleaving_invariant(steps, lambda: ledger["outstanding"] >= 0)

    def test_fails_closed_on_step_exception(self):
        def boom():
            raise RuntimeError("claim failed")

        with self.assertRaises(ProofBatteryError):
            interleaving_invariant([boom], lambda: True)

    def test_fails_closed_on_empty_steps(self):
        with self.assertRaises(ProofBatteryError):
            interleaving_invariant([], lambda: True)

    def test_fails_closed_on_bad_starting_state(self):
        with self.assertRaises(ProofBatteryError):
            interleaving_invariant([lambda: None], lambda: False)


if __name__ == "__main__":
    unittest.main()
