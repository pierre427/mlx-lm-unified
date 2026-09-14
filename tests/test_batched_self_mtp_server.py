import math
import threading
import types
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.apc import AutomaticPrefixCache
from mlx_lm.generate import BatchGenerator, ParallelSampleGenerator
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.server import (
    SelfMTPLaneAdmissionController,
    _batch_kind_key,
    _batched_kv_quantization,
    _batched_prompt_cache_model_key,
    _batched_self_mtp_config,
    _current_self_mtp_free_memory_gib,
    _make_generation_thread_lane_rng_root,
    _make_lane_rng,
    _make_self_mtp_admission_callback,
    _parallel_sampling_route,
    _parallel_prompt_cache_key,
    _parallel_mtp_completion_cache,
    _parallel_self_mtp_required_gib,
    ResponseGenerator,
)


GIB = 1 << 30


class TestSelfMTPLaneAdmissionController(unittest.TestCase):
    def setUp(self):
        self.controller = SelfMTPLaneAdmissionController()
        # 128 GiB host - 72.5 GiB resident PLE-offload operating point.
        self.free = 55.5

    def test_measured_ceiling_at_one_k_is_sixteen_lanes(self):
        decision = self.controller.decide([1024] * 17, self.free)
        self.assertEqual(len(decision.mtp_indices), 16)
        self.assertEqual(decision.stage, "fewer_lanes")
        self.assertTrue(all(decision.draft_depths[i] == 2 for i in decision.mtp_indices))

    def test_saturation_cap_binds_when_memory_would_admit_more(self):
        # A dense 27B (~18 GiB resident) leaves far more free than Flash-Next's
        # PLE operating point, so the memory envelope alone would admit ~40
        # lanes at short context -- past the measured throughput knee.  The
        # compute-saturation cap holds N at 16 regardless of free memory.
        decision = self.controller.decide([256] * 64, 95.0)
        self.assertEqual(len(decision.mtp_indices), 16)
        self.assertEqual(decision.stage, "fewer_lanes")
        self.assertTrue(all(decision.draft_depths[i] == 2 for i in decision.mtp_indices))
        # Uncapped, the same budget admits many more than the cap.
        uncapped = SelfMTPLaneAdmissionController(saturation_lane_cap=None)
        self.assertGreater(len(uncapped.decide([256] * 64, 95.0).mtp_indices), 16)

    def test_saturation_cap_is_configurable(self):
        controller = SelfMTPLaneAdmissionController(saturation_lane_cap=8)
        self.assertEqual(len(controller.decide([256] * 64, 95.0).mtp_indices), 8)

    def test_verification_row_cap_tracks_primary_and_branch_width(self):
        controller = SelfMTPLaneAdmissionController(
            saturation_lane_cap=None, verification_row_cap=8
        )
        decision = controller.decide([256] * 4, 95.0, max_draft=2)
        self.assertEqual(decision.primary_rows, 4)
        self.assertEqual(decision.speculative_rows, 4)
        self.assertEqual(len(decision.mtp_indices), 2)
        self.assertEqual(decision.stage, "fewer_lanes")

        pressured = controller.decide([256] * 8, 95.0, max_draft=2)
        self.assertEqual(pressured.mtp_indices, ())
        self.assertEqual(pressured.stage, "plain")
        self.assertEqual(pressured.primary_rows, 8)

    def test_dense_transient_admits_fewer_lanes(self):
        # The MoE-calibrated 1.76 GiB/lane under-models a dense 27B (~3.1 GiB).
        # A dense-calibrated controller admits fewer lanes for the same budget.
        moe = SelfMTPLaneAdmissionController(saturation_lane_cap=None)
        dense = SelfMTPLaneAdmissionController(
            saturation_lane_cap=None, transient_gib_per_lane=3.1
        )
        self.assertLess(
            len(dense.decide([256] * 64, 60.0).mtp_indices),
            len(moe.decide([256] * 64, 60.0).mtp_indices),
        )

    def test_transient_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "transient"):
            SelfMTPLaneAdmissionController(transient_gib_per_lane=0.0)

    def test_saturation_cap_rejects_non_positive(self):
        with self.assertRaisesRegex(ValueError, "saturation lane cap"):
            SelfMTPLaneAdmissionController(saturation_lane_cap=0)

    def test_measured_ceiling_at_sixteen_k_is_four_lanes(self):
        decision = self.controller.decide([16 * 1024] * 5, self.free)
        self.assertEqual(len(decision.mtp_indices), 4)
        self.assertEqual(decision.stage, "fewer_lanes")

    def test_hard_reserve_is_never_spent(self):
        for free in (19.0, 20.0, 24.0, 40.0, 55.5, 80.0):
            for contexts in (
                [1024],
                [1024] * 20,
                [4 * 1024, 16 * 1024, 64 * 1024],
            ):
                decision = self.controller.decide(contexts, free)
                self.assertLessEqual(
                    decision.estimated_gib,
                    max(free - self.controller.hard_reserve_gib, 0.0) + 1e-12,
                )
                self.assertAlmostEqual(
                    decision.usable_gib,
                    max(free - self.controller.hard_reserve_gib, 0.0),
                )

    def test_reserve_cannot_be_configured_below_sixteen_gib(self):
        with self.assertRaisesRegex(ValueError, "at least 16"):
            SelfMTPLaneAdmissionController(service_reserve_gib=15.99)

    def test_each_call_re_evaluates_context_and_free_memory(self):
        first = self.controller.decide([1024] * 4, self.free)
        self.assertEqual(first.stage, "full")
        self.assertEqual(first.draft_depths, (2, 2, 2, 2))

        # Same controller, later boundary: contexts grew and free memory fell.
        # 28.6 GiB free leaves 8.6 usable: a k=2 lane needs 8.80 and does not
        # fit, a k=1 lane needs 8.448 and does. The figure moved with the
        # 2026-09-09 transient recalibration (k=1 reserve 0.50 -> 0.80 of k=2,
        # measured); what this test pins is the ladder ORDER, not the constant.
        second = self.controller.decide([16 * 1024] * 4, 28.6)
        self.assertEqual(second.stage, "lower_k")
        self.assertEqual(second.mtp_indices, (0,))
        self.assertEqual(second.draft_depths[0], 1)

        # Membership changes are reflected immediately too.
        third = self.controller.decide([1024] * 12, self.free)
        self.assertEqual(third.stage, "full")
        self.assertEqual(len(third.mtp_indices), 12)

    def test_degradation_order_is_exact(self):
        fewer = self.controller.decide([16 * 1024] * 5, self.free)
        self.assertEqual(fewer.stage, "fewer_lanes")
        self.assertEqual(len(fewer.mtp_indices), 4)
        self.assertTrue(all(fewer.draft_depths[i] == 2 for i in fewer.mtp_indices))

        lower_k = self.controller.decide([77 * 1024], self.free)
        self.assertEqual(lower_k.stage, "lower_k")
        self.assertEqual(lower_k.draft_depths, (1,))

        plain = self.controller.decide([79 * 1024], self.free)
        self.assertEqual(plain.stage, "plain")
        self.assertEqual(plain.modes, ("plain",))
        self.assertEqual(plain.draft_depths, (0,))

        queued = self.controller.decide([80 * 1024], self.free)
        self.assertEqual(queued.stage, "queue")
        self.assertEqual(queued.modes, ("queue",))
        self.assertEqual(queued.draft_depths, (None,))

    def test_long_lane_never_forces_unsafe_m_equals_three_n(self):
        decision = self.controller.decide(
            [16 * 1024, 80 * 1024, 1024], self.free
        )
        self.assertEqual(decision.stage, "fewer_lanes")
        self.assertEqual(decision.mtp_indices, (0, 2))
        self.assertEqual(decision.modes[1], "queue")
        self.assertLessEqual(decision.estimated_gib, decision.usable_gib)

    def test_excluded_lanes_route_plain_without_entering_mtp_width(self):
        decision = self.controller.decide(
            [1024, 1024, 1024], self.free, eligible=[True, False, True]
        )
        self.assertEqual(decision.modes, ("self_mtp", "plain", "self_mtp"))
        self.assertEqual(decision.draft_depths, (2, 0, 2))

    def test_uncertain_inputs_fail_closed(self):
        for free in (math.nan, math.inf, -1.0):
            decision = self.controller.decide([1024], free)
            self.assertEqual(decision.stage, "queue")
            self.assertEqual(decision.mtp_indices, ())
        decision = self.controller.decide([-1], self.free)
        self.assertEqual(decision.stage, "queue")
        self.assertEqual(decision.mtp_indices, ())

    def test_live_budget_uses_actual_system_available_memory(self):
        with mock.patch(
            "mlx_lm.server._system_available_memory_bytes", return_value=56 * GIB
        ):
            self.assertEqual(_current_self_mtp_free_memory_gib(), 56.0)

    def test_missing_live_budget_fails_closed(self):
        with mock.patch(
            "mlx_lm.server._system_available_memory_bytes", return_value=None
        ):
            self.assertIsNone(_current_self_mtp_free_memory_gib())

    def test_actual_cache_floor_and_joining_rows_are_budgeted(self):
        cheap = self.controller.decide([1024, 1024], 25.0)
        self.assertEqual(cheap.mtp_indices, (0, 1))
        retained = self.controller.decide(
            [1024, 1024], 25.0, cache_gib=[0.25, 4.0]
        )
        self.assertEqual(retained.mtp_indices, (0,))
        self.assertEqual(retained.modes[1], "queue")

    def test_parallel_projection_includes_source_and_joining_caches(self):
        cache = [types.SimpleNamespace(nbytes=2 * GIB, offset=1024)]
        required = _parallel_self_mtp_required_gib(
            self.controller, cache, 2, 1024, 2
        )
        # Three 2-GiB cache rows: retained source plus both joining lanes.
        self.assertGreaterEqual(required, 6.0 + 2 * 1.76)

    def test_parallel_projection_adds_actual_draft_cache_floor(self):
        cache = [
            types.SimpleNamespace(nbytes=GIB // 2, offset=1024) for _ in range(4)
        ]
        base = _parallel_self_mtp_required_gib(self.controller, cache, 2, 1024, 2)
        # Without a sidecar, each MTP lane still takes the single-MTP-layer
        # share of the measured target cost (2 GiB / 4 layers = 0.5 GiB):
        # (n+1) * 2.0 target rows + n * (0.5 draft + 1.76 transient).
        self.assertAlmostEqual(base, 3 * 2.0 + 2 * (0.5 + 1.76), places=6)
        # A restored sidecar's ACTUAL bytes replace the heuristic, scaled to
        # the full prompt: 1 GiB over 1023 pairs -> ~1.001 GiB per lane.
        draft_state = ([types.SimpleNamespace(nbytes=GIB, offset=1023)], object())
        with_draft = _parallel_self_mtp_required_gib(
            self.controller, cache, 2, 1024, 2, mtp_state=draft_state
        )
        self.assertAlmostEqual(
            with_draft - base, 2 * (1024.0 / 1023.0 - 0.5), places=6
        )
        # Plain projections allocate no draft rows and take no draft floor.
        self.assertEqual(
            _parallel_self_mtp_required_gib(
                self.controller, cache, 2, 1024, 0, mtp_state=draft_state
            ),
            _parallel_self_mtp_required_gib(self.controller, cache, 2, 1024, 0),
        )

    def test_parallel_projection_budgets_fresh_draft_on_cache_miss(self):
        # A cache miss measures 0 bytes but every MTP lane still allocates a
        # fresh draft cache; the floor falls back to the single-layer share of
        # the envelope-projected target cost instead of 0.
        cache = [types.SimpleNamespace(nbytes=0, offset=0) for _ in range(4)]
        required = _parallel_self_mtp_required_gib(
            self.controller, cache, 2, 1024, 2
        )
        envelope = self.controller.CACHE_GIB_PER_1K_TOKENS
        self.assertAlmostEqual(
            required, 3 * envelope + 2 * (envelope / 4 + 1.76), places=6
        )

    def test_parallel_mtp_apc_key_matches_the_advanced_cache_cursor(self):
        prompt = [1, 2, 3, 4]
        cache = [types.SimpleNamespace(nbytes=64, offset=4)]
        self.assertEqual(
            _parallel_prompt_cache_key(prompt, cache, {"persistent": True}),
            prompt,
        )
        self.assertEqual(_parallel_prompt_cache_key(prompt, cache, None), prompt[:-1])

    def test_generator_callback_rechecks_every_cycle_boundary(self):
        # See the note on the reserve recalibration above: 28.6 is the figure
        # at which exactly one k=1 lane still fits at 16K.
        samples = iter((55.5, 28.6, None))
        callback = _make_self_mtp_admission_callback(
            self.controller, lambda: next(samples)
        )
        rows = [(101, 1024, 2, True), (102, 1024, 2, True)]
        self.assertEqual(callback(rows), {101: 2, 102: 2})

        grown = [(101, 16 * 1024, 2, True), (102, 16 * 1024, 2, True)]
        # Only one k=1 lane fits after the pressure change; the other pauses.
        self.assertEqual(callback(grown), {101: 1, 102: "queue"})
        # Missing live memory is never converted into optimistic capacity.
        self.assertEqual(callback(grown), {101: "queue", 102: "queue"})

    def test_active_long_lane_does_not_recharge_its_resident_cache(self):
        # A 64.7K Hermes prompt was admitted with enough free memory, then
        # permanently queued itself after prefill because the cycle callback
        # charged its now-resident ~28 GiB cache against the already-reduced
        # free-memory reading.  The active lane needs only the next transient;
        # a joining lane with the same projected cache still pays full cost.
        callback = _make_self_mtp_admission_callback(
            self.controller, lambda: 44.0
        )
        active = [(101, 64680, 2, True, 28.0)]
        joining = [(102, 64680, 2, False, 28.0)]
        self.assertEqual(callback(active), {101: 2})
        self.assertEqual(callback(joining), {102: "queue"})

    def test_hybrid_long_lane_is_not_charged_envelope_gap_as_growth(self):
        # Flash-Next at 57K measured ~3.5 GiB of cache against a 24.6 GiB
        # linear envelope. With ~42 GiB free the sole lane was queued on 85%
        # of cycles (1122 of 1316) and decoded at 0.2 tok/s.
        callback = _make_self_mtp_admission_callback(
            self.controller, lambda: 42.5
        )
        self.assertEqual(callback([(101, 57290, 2, True, 3.5)]), {101: 2})
        self.assertEqual(callback([(101, 102171, 2, True, 6.0)]), {101: 2})
        # A joining lane has no resident cache and still pays the envelope.
        self.assertEqual(callback([(102, 57290, 2, False, 0.0)]), {102: "queue"})

    def test_resident_growth_uses_larger_of_measured_and_envelope_rate(self):
        c = self.controller
        horizon = c.RESIDENT_GROWTH_HORIZON_TOKENS
        transient = c.K2_TRANSIENT_GIB_PER_LANE
        floor = c.CACHE_GIB_PER_1K_TOKENS * horizon / 1024
        # A small hybrid rate is floored at the envelope rate.
        self.assertAlmostEqual(
            c.lane_gib(57290, 2, 3.5, resident_cache=True),
            floor + transient,
            places=6,
        )
        # A measured rate above the envelope is charged as measured.
        self.assertAlmostEqual(
            c.lane_gib(16 * 1024, 2, 8.0, resident_cache=True),
            8.0 / (16 * 1024) * horizon + transient,
            places=6,
        )
        # Unmeasured resident cache fails closed to the full envelope.
        envelope = c.CACHE_GIB_PER_1K_TOKENS * 16
        self.assertAlmostEqual(
            c.lane_gib(16 * 1024, 2, 0.0, resident_cache=True),
            envelope + transient,
            places=6,
        )
        # Non-resident projection is unchanged.
        self.assertAlmostEqual(
            c.lane_gib(16 * 1024, 2, 0.0), envelope + transient, places=6
        )

    def test_pending_memory_is_charged_in_full(self):
        c = self.controller
        base = c.lane_gib(57290, 2, 3.5, resident_cache=True)
        self.assertAlmostEqual(
            c.lane_gib(57290, 2, 3.5, resident_cache=True, pending_gib=5.0),
            base + 5.0,
            places=6,
        )
        usable = base + 1.0
        free = usable + c.hard_reserve_gib
        callback = _make_self_mtp_admission_callback(c, lambda: free)
        self.assertEqual(callback([(101, 57290, 2, True, 3.5, 0.0)]), {101: 2})
        self.assertNotEqual(callback([(101, 57290, 2, True, 3.5, 5.0)]), {101: 2})
        with self.assertRaisesRegex(ValueError, "pending_gib"):
            c.lane_gib(1024, 2, pending_gib=-1.0)

    def test_resident_growth_can_still_lower_admission(self):
        # 16K at 0.5 GiB/1K measured: horizon growth is real and is charged,
        # so k=2 does not fit in 2 GiB usable but k=1 does.
        c = self.controller
        growth = 8.0 / (16 * 1024) * c.RESIDENT_GROWTH_HORIZON_TOKENS
        usable = growth + c.K2_TRANSIENT_GIB_PER_LANE * 0.9
        callback = _make_self_mtp_admission_callback(
            c, lambda: usable + c.hard_reserve_gib
        )
        self.assertEqual(callback([(101, 16 * 1024, 2, True, 8.0)]), {101: 1})

    def test_measured_joining_lane_pays_its_cache_not_the_envelope(self):
        # A warm APC restore at 57K measured ~3.8 GiB. Charging the 24.6 GiB
        # envelope held the join for ~32 s before its first token.
        c = self.controller
        growth = c.CACHE_GIB_PER_1K_TOKENS * c.RESIDENT_GROWTH_HORIZON_TOKENS / 1024
        self.assertAlmostEqual(
            c.lane_gib(57272, 2, 3.8),
            3.8 + growth + c.K2_TRANSIENT_GIB_PER_LANE,
            places=6,
        )
        callback = _make_self_mtp_admission_callback(c, lambda: 42.5)
        self.assertEqual(callback([(102, 57272, 2, False, 3.8)]), {102: 2})

    def test_short_resident_lanes_keep_the_sixteen_lane_ceiling(self):
        # Production rows are resident and measured; the growth floor must
        # not cut the calibrated N=16 ceiling at 1K.
        rows = [(100 + i, 1024, 2, True, 0.05) for i in range(17)]
        callback = _make_self_mtp_admission_callback(
            self.controller, lambda: self.free
        )
        admitted = [a for a in callback(rows).values() if a == 2]
        self.assertEqual(len(admitted), 16)

    def _resident(self, uid, context=1024, cache=0.05):
        return (uid, context, 2, True, cache, 0.0)

    def _cost(self, row, depth=2):
        return self.controller.lane_gib(
            row[1], depth, row[4], resident_cache=True
        )

    def test_queued_lane_needs_margin_while_another_lane_decodes(self):
        c = self.controller
        a, b = self._resident(101), self._resident(102)
        cost = self._cost(a)
        one = c.hard_reserve_gib + cost + 0.01
        two = c.hard_reserve_gib + 2 * cost + 0.01
        samples = iter((one, two, two + c.READMIT_MARGIN_GIB, two))
        seen = []
        callback = _make_self_mtp_admission_callback(
            c, lambda: next(samples), observer=seen.append
        )
        self.assertEqual(callback([a, b]), {101: 2, 102: "queue"})
        # 102 fits again but not with the margin; 101 keeps decoding.
        self.assertEqual(callback([a, b]), {101: 2, 102: "queue"})
        self.assertEqual(seen[-1].stage, "fewer_lanes")
        self.assertEqual(seen[-1].speculative_rows, 2)
        self.assertEqual(callback([a, b]), {101: 2, 102: 2})
        # Once readmitted, no margin applies.
        self.assertEqual(callback([a, b]), {101: 2, 102: 2})

    def test_sole_queued_lane_rejoins_without_margin(self):
        # A paused sole lane keeps its cache, so waiting for a margin that
        # nothing will free would starve it.
        c = self.controller
        row = self._resident(101)
        fit = c.hard_reserve_gib + self._cost(row) + 0.01
        samples = iter((c.hard_reserve_gib, fit, fit))
        callback = _make_self_mtp_admission_callback(c, lambda: next(samples))
        self.assertEqual(callback([row]), {101: "queue"})
        self.assertEqual(callback([row]), {101: 2})
        self.assertEqual(callback([row]), {101: 2})

    def test_margin_never_mixes_draft_depths(self):
        from mlx_lm.server import SelfMTPLaneAdmission

        class Stub:
            READMIT_MARGIN_GIB = 4.0
            READMIT_MAX_HOLDS = 8

            def decide(self, contexts, free, **_):
                if free < 10:
                    return SelfMTPLaneAdmission(
                        ("self_mtp", "queue"), (2, None), "fewer_lanes", 1, 1, 2, 2
                    )
                if free < 20:
                    return SelfMTPLaneAdmission(
                        ("self_mtp", "self_mtp"), (1, 1), "lower_k", 1, 1, 2, 2
                    )
                return SelfMTPLaneAdmission(
                    ("self_mtp", "self_mtp"), (2, 2), "full", 1, 1, 2, 4
                )

        samples = iter((5.0, 22.0))
        callback = _make_self_mtp_admission_callback(Stub(), lambda: next(samples))
        rows = [self._resident(101), self._resident(102)]
        self.assertEqual(callback(rows), {101: 2, 102: "queue"})
        # The margin plan admits 102 only at k=1. Lowering one lane would mix
        # depths, which attach rejects, so 102 rejoins at the plan depth.
        self.assertEqual(callback(rows), {101: 2, 102: 2})

    def test_join_probe_does_not_touch_hysteresis_state(self):
        c = self.controller
        active = self._resident(101, 16 * 1024, 8.0)
        joiner = (201, 1024, 2, False, 0.0)
        a_cost = self._cost(active)
        j_cost = c.lane_gib(1024, 2)
        tight = c.hard_reserve_gib + max(a_cost, j_cost) + 0.01
        samples = iter((tight, tight + a_cost + j_cost))
        callback = _make_self_mtp_admission_callback(c, lambda: next(samples))
        # The probe may queue the active lane in its own plan...
        probe = callback([active, joiner])
        self.assertIn("queue", probe.values())
        # ...but that must not mark it demoted at the next real boundary.
        self.assertEqual(callback([active]), {101: 2})

    def test_hold_is_capped_so_arrivals_cannot_starve_a_lane(self):
        c = self.controller
        a, b = self._resident(101), self._resident(102)
        cost = self._cost(a)
        one = c.hard_reserve_gib + cost + 0.01
        two = c.hard_reserve_gib + 2 * cost + 0.01
        samples = iter([one] + [two] * (c.READMIT_MAX_HOLDS + 1))
        callback = _make_self_mtp_admission_callback(c, lambda: next(samples))
        self.assertEqual(callback([a, b]), {101: 2, 102: "queue"})
        for _ in range(c.READMIT_MAX_HOLDS):
            self.assertEqual(callback([a, b]), {101: 2, 102: "queue"})
        self.assertEqual(callback([a, b]), {101: 2, 102: 2})

    def test_departed_and_nan_boundaries_reset_hysteresis(self):
        c = self.controller
        a, b = self._resident(101), self._resident(102)
        cost = self._cost(a)
        one = c.hard_reserve_gib + cost + 0.01
        two = c.hard_reserve_gib + 2 * cost + 0.01
        samples = iter((one, two, None, two, two))
        callback = _make_self_mtp_admission_callback(c, lambda: next(samples))
        self.assertEqual(callback([a, b]), {101: 2, 102: "queue"})
        # 102 leaves; when it returns it is a fresh lane.
        self.assertEqual(callback([a]), {101: 2})
        # Unmeasured free memory queues everything.
        self.assertEqual(callback([a, b]), {101: "queue", 102: "queue"})
        # Both were queued: neither is held while no other lane decodes.
        self.assertEqual(callback([a, b]), {101: 2, 102: 2})
        # Rows without a cache field fail closed to the envelope.
        self.assertEqual(callback([(103, 64 * 1024, 2, True)]), {103: "queue"})


class TestBatchedSelfMTPRouting(unittest.TestCase):
    def args(self, **overrides):
        values = dict(
            model=types.SimpleNamespace(draft="default_model"),
            prompt_lookup_ngram=0,
            sampling=types.SimpleNamespace(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                xtc_probability=0.0,
            ),
            seed=None,
        )
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def cli(self, **overrides):
        values = dict(
            self_mtp=True,
            self_mtp_num_draft=2,
            self_mtp_persistent=True,
            self_mtp_rate_gate=False,
            self_mtp_transformed_verifier=True,
            self_mtp_share_qsa_indices=False,
            self_mtp_share_qsa_indices_min_prompt_tokens=0,
            self_mtp_adaptive_depth_ceiling=None,
            self_mtp_max_prompt_tokens=None,
            self_mtp_window_size=0,
            self_mtp_window_min_prompt_tokens=0,
            kv_bits=None,
        )
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def setUp(self):
        self.model = types.SimpleNamespace(mtp=object())

    def route(self, *, args=None, cli=None, cached=0, state=None):
        return _batched_self_mtp_config(
            args or self.args(),
            cli or self.cli(),
            self.model,
            prompt_tokens=4096,
            cached_prompt_tokens=cached,
            mtp_state=state,
        )

    def test_supported_request_gets_distinct_fixed_batch_kind(self):
        config = self.route()
        self.assertIsNotNone(config)
        plain = _batch_kind_key(("model", "adapter", None))
        mtp = _batch_kind_key(("model", "adapter", None), config)
        self.assertEqual(plain[-1], "plain")
        self.assertEqual(
            mtp,
            (("model", "adapter", None), "self_mtp", True, 2, "native", False),
        )
        self.assertNotEqual(plain, mtp)

    def test_quantized_generator_args_and_apc_namespace_are_opt_in(self):
        cli = self.cli(
            kv_bits=8,
            kv_group_size=32,
            self_mtp_allow_quantized_kv=True,
        )
        config = self.route(cli=cli)
        kwargs = _batched_kv_quantization(cli, config)
        self.assertEqual(kwargs, {"kv_bits": 8, "kv_group_size": 32})

        generator = BatchGenerator(object(), self_mtp=config, **kwargs)
        try:
            self.assertEqual(generator.kv_bits, 8)
            self.assertEqual(generator.kv_group_size, 32)
        finally:
            generator.close()

        model_key = ("model", "adapter", None)
        quantized_key = _batched_prompt_cache_model_key(model_key, cli, config)
        self.assertNotEqual(quantized_key, model_key)
        self.assertEqual(_batched_kv_quantization(cli, None), {})
        self.assertEqual(
            _batched_prompt_cache_model_key(model_key, cli, None), model_key
        )

    def test_frozen_exclusions_all_route_plain(self):
        cases = (
            (self.args(), self.cli(self_mtp_persistent=False)),
            (self.args(), self.cli(self_mtp_window_size=4096)),
            (self.args(), self.cli(kv_bits=4)),
            (self.args(prompt_lookup_ngram=4), self.cli()),
            (
                self.args(
                    sampling=types.SimpleNamespace(
                        temperature=0.8,
                        top_p=1.0,
                        top_k=0,
                        min_p=0.0,
                        xtc_probability=0.1,
                    )
                ),
                self.cli(),
            ),
            (self.args(), self.cli(self_mtp_rate_gate=True)),
            (self.args(), self.cli(self_mtp_adaptive_depth_ceiling=2)),
            (self.args(), self.cli(self_mtp_share_qsa_indices=True)),
        )
        for args, cli in cases:
            with self.subTest(args=args, cli=cli):
                self.assertIsNone(self.route(args=args, cli=cli))

    def test_sidecarless_target_cache_hit_routes_plain(self):
        self.assertIsNone(self.route(cached=128, state=None))
        self.assertIsNotNone(self.route(cached=128, state=([], object())))

    def test_lane_rng_is_materialized_where_it_is_created_or_restored(self):
        root = LaneRNG(7)
        with mock.patch("mlx_lm.server.mx.eval") as evaluate:
            lane = _make_lane_rng(types.SimpleNamespace(seed=11), root)
        evaluate.assert_called_once_with(lane.key)

    def test_unseeded_root_is_created_and_evaluated_on_generation_thread(self):
        observed = {}

        def worker():
            observed["thread"] = threading.get_ident()
            with mock.patch("mlx_lm.server.mx.eval") as evaluate:
                root = _make_generation_thread_lane_rng_root()
            observed["root"] = root
            observed["eval_arg"] = evaluate.call_args.args[0]

        origin = threading.get_ident()
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertNotEqual(observed["thread"], origin)
        self.assertIs(observed["eval_arg"], observed["root"].key)

    def test_cross_thread_lazy_root_cannot_reach_a_lane(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        # This is the old outage root: created lazily on the caller thread.
        generator._lane_rng_root = LaneRNG.from_key(
            mx.random.split(mx.random.state[0])[1]
        )
        generator._generation_error = None

        def generate():
            lane = _make_lane_rng(
                types.SimpleNamespace(seed=None), generator._lane_rng_root
            )
            draw = mx.random.uniform(shape=(1,), key=lane.next_key())
            mx.eval(lane.key, draw)

        generator._generate = generate
        thread = threading.Thread(target=generator._run_generation)
        thread.start()
        thread.join()
        if generator._generation_error is not None:
            raise generator._generation_error

    def test_logits_processors_route_parallel_mtp_to_plain(self):
        kind, note = _parallel_sampling_route(
            self.args(),
            self.cli(),
            self.model,
            prompt_tokens=4096,
            has_logits_processors=True,
        )
        self.assertEqual(kind, "plain")
        self.assertIn("logits processors", note)


def _tiny_mtp_model_args():
    return TextModelArgs(
        model_type="qwen3_5_moe_text",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        full_attention_interval=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        mtp_num_hidden_layers=1,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )


class TestParallelAPCLifecycle(unittest.TestCase):
    """MLX-backed: the REAL parallel prepare -> advance -> store -> restore.

    Runs on the production M5 at the gate. The retained source cache is
    advanced by lane preparation, decode runs to completion on per-row
    copies, and the APC key is derived at store time — exactly the server's
    ``_serve_parallel`` insert path — so the stored span and its key must
    agree and a later lookup must restore the exact cursor.
    """

    @classmethod
    def setUpClass(cls):
        cls._previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        mx.random.seed(11)
        cls.model = TextModel(_tiny_mtp_model_args())
        mx.eval(cls.model.parameters())

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._previous_device)

    def test_store_key_matches_advanced_cache_and_restores_exact_cursor(self):
        prompt = [1, 2, 3, 4, 5, 6]
        self_mtp = {
            "persistent": True,
            "num_draft": 2,
            "sampling_temp": 0.0,
            "accept_rule": "residual",
        }
        cache = make_prompt_cache(self.model)
        parallel = ParallelSampleGenerator(
            self.model,
            cache,
            prompt[-1],
            2,
            max_tokens=4,
            all_tokens=[],
            prefill_step_size=4,
            self_mtp=self_mtp,
            mtp_state=None,
            lane_rng=LaneRNG(5),
            mtp_prompt=prompt,
        )
        try:
            while len(parallel) > 0:
                parallel.next()
        finally:
            parallel.close()

        # Store-time key derivation (the server's insert path): the key must
        # cover exactly the tokens the advanced retained cache holds.
        key = _parallel_prompt_cache_key(prompt, cache, self_mtp)
        self.assertEqual(key, prompt)
        self.assertEqual(
            max((getattr(c, "offset", 0) for c in cache), default=0),
            len(prompt),
        )

        apc = AutomaticPrefixCache()
        apc.insert_cache("model", key, cache)
        lookup = apc.lookup("model", prompt + [7, 8])
        self.assertEqual(lookup.cached_tokens, len(prompt))
        self.assertEqual(lookup.remaining_tokens, [7, 8])
        self.assertEqual(
            max((getattr(c, "offset", 0) for c in lookup.cache), default=0),
            lookup.cached_tokens,
        )

    def test_completed_row_retains_exact_continuation_after_source_is_consumed(self):
        prompt = [1, 2, 3, 4, 5, 6]
        self_mtp = {
            "persistent": True,
            "num_draft": 2,
            "sampling_temp": 0.0,
            "accept_rule": "residual",
        }
        source = make_prompt_cache(self.model)
        parallel = ParallelSampleGenerator(
            self.model,
            source,
            prompt[-1],
            2,
            max_tokens=4,
            all_tokens=[],
            prefill_step_size=4,
            self_mtp=self_mtp,
            mtp_state=None,
            lane_rng=LaneRNG(7),
            mtp_prompt=prompt,
        )
        completed = None
        try:
            while len(parallel) > 0:
                for _, response in parallel.next():
                    if response.finish_reason is not None and completed is None:
                        completed = response
        finally:
            parallel.close()

        self.assertIsNotNone(completed)
        recovered = _parallel_mtp_completion_cache(prompt, completed)
        self.assertIsNotNone(recovered)
        key, cache, sidecar = recovered
        self.assertEqual(key, completed.all_tokens)
        self.assertEqual(sidecar.covered_tokens, len(key))
        self.assertIs(sidecar.state, completed.mtp_state)
        self.assertEqual(
            max((getattr(c, "offset", 0) for c in cache), default=0),
            len(key),
        )
        apc = AutomaticPrefixCache()
        apc.insert_cache("model", key, cache, sidecar=sidecar)
        lookup = apc.lookup("model", key + [7, 8])
        self.assertEqual(lookup.cached_tokens, len(key))
        self.assertEqual(lookup.remaining_tokens, [7, 8])
        self.assertIsNotNone(lookup.sidecar)
        self.assertEqual(lookup.sidecar.covered_tokens, len(key))


if __name__ == "__main__":
    unittest.main()
