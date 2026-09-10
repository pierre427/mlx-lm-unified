"""Per-batch-size draft-depth policy (spec_policy).

Covers table parsing (valid/invalid, loudly), depth selection at each band,
the hard M5 verify-width cap (drafts + 1 bonus <= 8) with its one-time
warning, env-var override precedence, and a scripted no-regression check that
the default single-stream configuration reproduces the previous fixed-K
depth decisions decision-for-decision (no model load)."""

import os
import unittest
import warnings

import mlx_lm.spec_policy as sp
from mlx_lm.generate import draft_tokens_for_budget
from mlx_lm.spec_policy import (
    DEFAULT_DEPTH_TABLE,
    DEPTH_TABLE_ENV,
    MAX_DRAFT_TOKENS,
    MAX_VERIFY_WIDTH,
    DepthBand,
    DepthTable,
    cap_draft_tokens,
    draft_depth_for,
    resolve_depth_table,
)


class _EnvIsolatedTest(unittest.TestCase):
    """Isolate the env override and the one-time cap-warning latch."""

    def setUp(self):
        self._saved_env = os.environ.pop(DEPTH_TABLE_ENV, None)
        self._saved_latch = sp._cap_warning_emitted
        sp._cap_warning_emitted = False

    def tearDown(self):
        if self._saved_env is not None:
            os.environ[DEPTH_TABLE_ENV] = self._saved_env
        else:
            os.environ.pop(DEPTH_TABLE_ENV, None)
        sp._cap_warning_emitted = self._saved_latch


class TestDepthTableParsing(_EnvIsolatedTest):
    def test_parse_example_spec(self):
        t = DepthTable.parse("1:6,2-4:3,5-8:2")
        self.assertEqual(
            [(b.lo, b.hi, b.k) for b in t.bands],
            [(1, 1, 6), (2, 4, 3), (5, 8, 2)],
        )

    def test_parse_open_ended_and_whitespace(self):
        t = DepthTable.parse(" 1 : 6 , 2-4 : 3 , 5+ : 2 ")
        self.assertEqual(
            [(b.lo, b.hi, b.k) for b in t.bands],
            [(1, 1, 6), (2, 4, 3), (5, None, 2)],
        )

    def test_parse_single_row(self):
        t = DepthTable.parse("1+:4")
        self.assertEqual([(b.lo, b.hi, b.k) for b in t.bands], [(1, None, 4)])

    def test_parse_unsorted_rows_are_sorted(self):
        t = DepthTable.parse("5+:2,1:6,2-4:3")
        self.assertEqual([b.lo for b in t.bands], [1, 2, 5])

    def test_parse_invalid_specs_raise(self):
        bad = [
            "",  # empty
            "   ",  # whitespace only
            "1",  # no colon
            "1:2:3",  # too many colons
            "1:x",  # non-integer depth
            "x:2",  # non-integer range
            "3-2:4",  # lo > hi
            "0:3",  # batch size below 1
            "1:-1",  # negative depth
            "1:3,,2:1",  # empty row
            "1-3:2,2-5:1",  # overlapping bands
            "1+:2,2:1",  # open-ended band overlaps a later one
            "2-4:3,2-4:1",  # duplicate band
        ]
        for spec in bad:
            with self.assertRaises(ValueError, msg=f"spec {spec!r} should fail"):
                DepthTable.parse(spec)

    def test_parse_errors_name_the_offending_row(self):
        with self.assertRaisesRegex(ValueError, "3-2:4"):
            DepthTable.parse("1:6,3-2:4")

    def test_empty_table_rejected(self):
        with self.assertRaises(ValueError):
            DepthTable([])


class TestDepthSelection(_EnvIsolatedTest):
    def test_default_table_bands(self):
        # bs1 -> base, bs2-4 -> min(base, 3), bs5+ -> min(base, 2)
        base = 6
        self.assertEqual(draft_depth_for(1, base), 6)
        for bs in (2, 3, 4):
            self.assertEqual(draft_depth_for(bs, base), 3)
        for bs in (5, 8, 100):
            self.assertEqual(draft_depth_for(bs, base), 2)

    def test_table_is_a_ceiling_not_a_floor(self):
        # min(base, K): a small base is never inflated by a larger band K.
        self.assertEqual(draft_depth_for(2, 1), 1)
        self.assertEqual(draft_depth_for(5, 0), 0)
        self.assertEqual(draft_depth_for(1, 2, table="1:6"), 2)

    def test_custom_table_bands(self):
        t = DepthTable.parse("1:6,2-4:3,5-8:2")
        self.assertEqual(draft_depth_for(1, 7, table=t), 6)
        self.assertEqual(draft_depth_for(3, 7, table=t), 3)
        self.assertEqual(draft_depth_for(8, 7, table=t), 2)

    def test_batch_size_above_table_falls_back_to_nearest_band_below(self):
        # bs9 is uncovered by "5-8:2"; higher concurrency must never
        # speculate deeper than the highest configured band.
        t = DepthTable.parse("1:6,2-4:3,5-8:2")
        self.assertEqual(draft_depth_for(9, 7, table=t), 2)
        self.assertEqual(draft_depth_for(64, 7, table=t), 2)

    def test_batch_size_below_table_uses_base(self):
        t = DepthTable.parse("4+:2")
        self.assertEqual(draft_depth_for(2, 5, table=t), 5)

    def test_spec_string_accepted_directly(self):
        self.assertEqual(draft_depth_for(3, 7, table="1:6,2-4:4"), 4)

    def test_invalid_batch_size_raises(self):
        with self.assertRaises(ValueError):
            draft_depth_for(0, 3)
        with self.assertRaises(ValueError):
            DEFAULT_DEPTH_TABLE.depth_for(-1, 3)


class TestVerifyWidthCap(_EnvIsolatedTest):
    def test_cap_constants(self):
        self.assertEqual(MAX_VERIFY_WIDTH, 8)
        self.assertEqual(MAX_DRAFT_TOKENS, 7)

    def test_configured_k10_caps_to_seven_drafts_plus_bonus(self):
        # K=10 configured -> 7 drafts; 7 drafts + 1 bonus = 8 verify rows.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = draft_depth_for(1, 10)
        self.assertEqual(k, 7)
        self.assertEqual(k + 1, MAX_VERIFY_WIDTH)

    def test_cap_applies_after_table_lookup(self):
        # The table itself may hold an over-cap depth; the cap wins.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(draft_depth_for(1, 12, table="1:12"), 7)

    def test_cap_warning_emitted_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            draft_depth_for(1, 10)
            draft_depth_for(1, 12)
        collisions = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(collisions), 1)
        self.assertIn("verify-width", str(collisions[0].message))

    def test_no_warning_below_cap(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(cap_draft_tokens(7), 7)
            self.assertEqual(cap_draft_tokens(0), 0)
        self.assertEqual(caught, [])


class TestEnvOverride(_EnvIsolatedTest):
    def test_env_table_used_when_set(self):
        os.environ[DEPTH_TABLE_ENV] = "1:6,2-4:4,5+:1"
        self.assertEqual(draft_depth_for(3, 7), 4)
        self.assertEqual(draft_depth_for(6, 7), 1)

    def test_explicit_table_beats_env(self):
        os.environ[DEPTH_TABLE_ENV] = "1+:1"
        self.assertEqual(draft_depth_for(3, 7, table="1+:5"), 5)

    def test_invalid_env_fails_loudly(self):
        os.environ[DEPTH_TABLE_ENV] = "garbage"
        with self.assertRaisesRegex(ValueError, DEPTH_TABLE_ENV):
            draft_depth_for(1, 3)

    def test_env_unset_gives_default_table(self):
        self.assertIs(resolve_depth_table(), DEFAULT_DEPTH_TABLE)


class TestNoRegressionSingleStream(_EnvIsolatedTest):
    """Default single-stream config must reproduce the previous fixed-K
    depth decisions decision-for-decision (scripted, no model load).

    Before this policy existed, the MTP loop chose per cycle
    ``k = draft_tokens_for_budget(num_draft, remaining)``. Now ``num_draft``
    first passes through ``draft_depth_for(1, num_draft)`` at generator
    entry. With no table configured and batch_size=1, the two rules must
    make the same choice at every cycle for every in-cap base depth.
    """

    def _script_cycles(self, pick_k, base, max_tokens, accepts):
        """Mimic _mtp_draft_verify_loop's k-selection/accounting: each cycle
        picks a depth, accepts a scripted number of drafts, and commits
        accepted + bonus tokens. Returns the (k, ntoks) decision trace."""
        trace = []
        ntoks, cycle = 1, 0  # first token is committed before the loop
        while ntoks < max_tokens:
            k = pick_k(base, max_tokens - ntoks)
            trace.append((k, ntoks))
            if k == 0:
                ntoks += 1  # plain step
                continue
            n_accept = min(accepts[cycle % len(accepts)], k)
            ntoks = min(ntoks + n_accept + 1, max_tokens)  # accepted + bonus
            cycle += 1
        return trace

    def test_decisions_identical_for_all_in_cap_depths(self):
        old_rule = draft_tokens_for_budget
        new_rule = lambda base, rem: draft_tokens_for_budget(
            draft_depth_for(1, base), rem
        )
        for base in range(0, MAX_DRAFT_TOKENS + 1):
            for accepts in ([0], [1, 0, 2], [7, 7, 3]):
                old = self._script_cycles(old_rule, base, 64, accepts)
                new = self._script_cycles(new_rule, base, 64, accepts)
                self.assertEqual(old, new, f"base={base} accepts={accepts}")

    def test_entry_resolution_is_identity_below_cap(self):
        for base in range(0, MAX_DRAFT_TOKENS + 1):
            self.assertEqual(draft_depth_for(1, base), base)


if __name__ == "__main__":
    unittest.main()
