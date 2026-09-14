import types
import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.generate import (
    MTPGenerationBatch,
    ParallelSampleGenerator,
    StopSequenceMatcher,
)
from mlx_lm.hybrid_speculative import MTPToken
from mlx_lm.hybrid_speculative import HybridStats


class _Lane:
    def __init__(self, uid, maximum=16, depth=2):
        self.uid = uid
        self.cur = 7
        self.seed_h = mx.zeros((1, 1, 1))
        self.token_prefix = mx.array([1, 2, 3], dtype=mx.uint32)
        self.rng = None
        self.ntoks = 1
        self.max_tokens = maximum
        self.num_draft = depth
        self.sampling_temp = 0.0
        self.accept_rule = "residual"
        self.logprob_transform = None
        self.logits_processors = []
        self.stats = HybridStats()


class _Detached:
    def __init__(self, lane):
        self.lane = lane
        self.caches = types.SimpleNamespace(
            target=[types.SimpleNamespace(uid=lane.uid, nbytes=0)],
            draft=[types.SimpleNamespace(uid=lane.uid, nbytes=0)],
        )


def _state(detached):
    return types.SimpleNamespace(
        lanes=[item.lane for item in detached],
        caches=types.SimpleNamespace(target=[], draft=[]),
        membership_epoch=1,
        proposal_open=False,
    )


def _detach(_model, state, indices):
    rows = [_Detached(state.lanes[index]) for index in indices]
    leaving = set(indices)
    state.lanes = [
        lane for index, lane in enumerate(state.lanes) if index not in leaving
    ]
    state.membership_epoch += 1
    return state, rows


class TestMTPGenerationBatch(unittest.TestCase):
    def _batch(self, lanes, initial=None, matchers=None, admission=None):
        detached = [_Detached(lane) for lane in lanes]
        initial = initial or [None] * len(lanes)
        matchers = matchers or [StopSequenceMatcher() for _ in lanes]
        with patch(
            "mlx_lm.hybrid_speculative.attach_self_mtp_lanes",
            side_effect=lambda _m, _b, joining: _state(joining),
        ):
            return MTPGenerationBatch(
                object(),
                detached,
                initial,
                matchers,
                mtp_admission=admission,
            )

    def test_stop_and_length_trim_each_burst_before_detach(self):
        lanes = [_Lane(10), _Lane(20, maximum=2)]
        batch = self._batch(
            lanes,
            matchers=[StopSequenceMatcher([[12]]), StopSequenceMatcher()],
        )
        batch._num_tokens = [1, 1]
        logprobs = [mx.array([-1.0, -0.5])] * 3
        proposal = types.SimpleNamespace(
            outputs=(
                (
                    MTPToken(11, logprobs[0], True),
                    MTPToken(12, logprobs[1], True),
                    MTPToken(13, logprobs[2], False),
                ),
                (
                    MTPToken(21, logprobs[0], True),
                    MTPToken(22, logprobs[1], False),
                ),
            )
        )
        committed = {}
        events = []

        def commit(_state, _proposal, *, emitted_counts, terminal):
            events.append("commit")
            committed["counts"] = list(emitted_counts)
            committed["terminal"] = list(terminal)

        def detach(*args):
            events.append("detach")
            return _detach(*args)

        with (
            patch(
                "mlx_lm.hybrid_speculative.propose_batched_self_mtp",
                return_value=proposal,
            ),
            patch(
                "mlx_lm.hybrid_speculative.commit_batched_self_mtp",
                side_effect=commit,
            ),
            patch(
                "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
                side_effect=detach,
            ),
        ):
            responses = batch.next()

        self.assertEqual([r.token for r in responses], [11, 12, 21])
        self.assertEqual(committed["counts"], [2, 1])
        self.assertEqual(committed["terminal"], [True, True])
        self.assertEqual(events, ["commit", "detach"])
        self.assertEqual(
            [r.finish_reason for r in responses],
            [None, "stop", "length"],
        )
        terminal = [r for r in responses if r.finish_reason is not None]
        self.assertTrue(all(r.prompt_cache is not None for r in terminal))
        self.assertTrue(all(r.mtp_state is not None for r in terminal))
        self.assertEqual(len(batch), 0)

    def test_nonterminal_lane_consumes_distribution_authorized_row(self):
        batch = self._batch([_Lane(9)])
        lp0 = mx.array([-0.1, -2.0])
        lp1 = mx.array([-1.5, -0.2])
        outputs = (MTPToken(0, lp0, True), MTPToken(1, lp1, False))
        proposal = types.SimpleNamespace(outputs=(outputs,))
        committed = {}

        def commit(_state, _proposal, *, emitted_counts, terminal):
            committed["counts"] = list(emitted_counts)
            committed["terminal"] = list(terminal)

        with (
            patch(
                "mlx_lm.hybrid_speculative.propose_batched_self_mtp",
                return_value=proposal,
            ),
            patch(
                "mlx_lm.hybrid_speculative.commit_batched_self_mtp",
                side_effect=commit,
            ),
        ):
            responses = batch.next()

        self.assertEqual([r.token for r in responses], [0, 1])
        self.assertIs(responses[0].logprobs, lp0)
        self.assertIs(responses[1].logprobs, lp1)
        self.assertEqual([r.from_draft for r in responses], [True, False])
        self.assertEqual(committed, {"counts": [2], "terminal": [False]})

    def test_admission_plain_detaches_before_proposal(self):
        batch = self._batch(
            [_Lane(4)], admission=lambda _rows: {4: "plain"}
        )
        with (
            patch(
                "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
                side_effect=_detach,
            ),
            patch(
                "mlx_lm.hybrid_speculative.propose_batched_self_mtp"
            ) as propose,
        ):
            self.assertEqual(batch.next(), [])
        propose.assert_not_called()
        ready = batch.take_plain_fallbacks()
        self.assertEqual([item.detached.lane.uid for item in ready], [4])

    def test_dynamic_depth_is_normalized_before_a_later_join(self):
        decisions = lambda rows: {row[0]: 1 for row in rows}
        active = self._batch([_Lane(1, depth=2)], admission=decisions)
        joining = self._batch([_Lane(2, depth=2)])

        def attach(_model, batch, packages):
            if batch is None or not batch.lanes:
                return _state(packages)
            batch.lanes.extend(package.lane for package in packages)
            batch.membership_epoch += 1
            return batch

        with (
            patch(
                "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
                side_effect=_detach,
            ),
            patch(
                "mlx_lm.hybrid_speculative.attach_self_mtp_lanes",
                side_effect=attach,
            ),
        ):
            active.extend(joining)

        self.assertEqual(active.uids, [1, 2])
        self.assertEqual([lane.num_draft for lane in active.state.lanes], [1, 1])

    def test_newly_admitted_lane_delivers_initial_before_speculative_cycle(self):
        decisions = {1: 3, 2: "queue"}

        def admit(rows):
            return {row[0]: decisions[row[0]] for row in rows}

        active = self._batch([_Lane(1, depth=3)], admission=admit)
        first_lp = mx.array([-2.0, -0.1])
        joining = self._batch(
            [_Lane(2, depth=3)],
            initial=[MTPToken(1, first_lp, False)],
        )

        def attach(_model, batch, packages):
            if batch is None or not batch.lanes:
                return _state(packages)
            batch.lanes.extend(package.lane for package in packages)
            batch.membership_epoch += 1
            return batch

        with (
            patch(
                "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
                side_effect=_detach,
            ),
            patch(
                "mlx_lm.hybrid_speculative.attach_self_mtp_lanes",
                side_effect=attach,
            ),
        ):
            active.extend(joining)
            self.assertIn(2, active._paused)
            decisions[2] = 3
            with patch(
                "mlx_lm.hybrid_speculative.propose_batched_self_mtp"
            ) as propose:
                responses = active.next()

        self.assertEqual([(r.uid, r.token) for r in responses], [(2, 1)])
        propose.assert_not_called()

    def test_paused_row_in_physical_batch_reports_merge_copy(self):
        short, long = _Lane(1), _Lane(2)
        short.token_prefix = mx.array([1] * 999, dtype=mx.uint32)
        long.token_prefix = mx.array([1] * 9_999, dtype=mx.uint32)
        batch = self._batch([short, long])
        batch.state.caches = types.SimpleNamespace(
            target=[types.SimpleNamespace(nbytes=2_000 << 20)], draft=[]
        )
        with patch(
            "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
            side_effect=_detach,
        ):
            for package in batch._detach_packages([1]):
                batch._paused[package.detached.lane.uid] = package
        rows = {row[0]: row for row in batch.mtp_cycle_state()}
        self.assertTrue(rows[2][3])
        # Rejoining rebuilds both rows at the paused 10K width.
        self.assertGreater(rows[2][5], 0.0)
        self.assertEqual(rows[1][5], 0.0)

    def test_padded_rows_report_one_shared_growth_rate(self):
        short, long = _Lane(1), _Lane(2)
        short.token_prefix = mx.array([1] * 99, dtype=mx.uint32)
        long.token_prefix = mx.array([1] * 9_999, dtype=mx.uint32)
        batch = self._batch([short, long])
        batch.state.caches = types.SimpleNamespace(
            target=[types.SimpleNamespace(nbytes=20_000 << 20)], draft=[]
        )
        rows = {row[0]: row for row in batch.mtp_cycle_state()}
        rate = [rows[uid][4] / rows[uid][1] for uid in (1, 2)]
        # Both padded rows grow at total / lanes / max context per token.
        expected = (20_000 << 20) / 2 / 10_000 / float(1 << 30)
        self.assertAlmostEqual(rate[0], expected, places=12)
        self.assertAlmostEqual(rate[1], expected, places=12)
        self.assertEqual(rows[1][5], 0.0)

    def test_paused_lane_reports_its_retained_cache_as_resident(self):
        decisions = {1: "queue", 2: 3}
        batch = self._batch(
            [_Lane(1, depth=3), _Lane(2, depth=3)],
            admission=lambda rows: {row[0]: decisions[row[0]] for row in rows},
        )

        with patch(
            "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
            side_effect=_detach,
        ):
            batch._apply_admission()

        self.assertEqual(batch.uids, [2])
        rows = {row[0]: row for row in batch.mtp_cycle_state()}
        self.assertTrue(rows[1][3])

    def test_paused_lane_decided_plain_moves_to_plain(self):
        decisions = {1: "queue", 2: 3}
        batch = self._batch(
            [_Lane(1, depth=3), _Lane(2, depth=3)],
            admission=lambda rows: {row[0]: decisions[row[0]] for row in rows},
        )
        with patch(
            "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
            side_effect=_detach,
        ):
            batch._apply_admission()
            self.assertIn(1, batch._paused)
            decisions[1] = "plain"
            batch._apply_admission()
        self.assertNotIn(1, batch._paused)
        ready = batch.take_plain_fallbacks()
        self.assertEqual([item.detached.lane.uid for item in ready], [1])

    def test_sole_queued_lane_stays_paused_inside_the_batch(self):
        # The batch cannot see the plain fallback batch, so it never demotes.
        batch = self._batch(
            [_Lane(1, depth=3)], admission=lambda rows: {row[0]: "queue" for row in rows}
        )
        with patch(
            "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
            side_effect=_detach,
        ):
            batch._apply_admission()
        self.assertIn(1, batch._paused)
        self.assertEqual(batch.take_plain_fallbacks(), [])

    def _starved_generator(self, batch, plain_lanes=0):
        from mlx_lm.generate import BatchGenerator

        generator = BatchGenerator.__new__(BatchGenerator)
        generator._generation_batch = batch
        generator._plain_fallback_batch = [object()] * plain_lanes
        generator._starved_mtp_boundaries = 0
        return generator

    def _paused_batch(self):
        batch = self._batch(
            [_Lane(1, depth=3), _Lane(2, depth=3)],
            admission=lambda rows: {row[0]: "queue" for row in rows},
        )
        with patch(
            "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
            side_effect=_detach,
        ):
            batch._apply_admission()
        return batch

    def test_starved_lane_continues_plain_after_consecutive_boundaries(self):
        from mlx_lm.generate import MTP_STARVED_BOUNDARIES_BEFORE_PLAIN

        batch = self._paused_batch()
        generator = self._starved_generator(batch)
        for _ in range(MTP_STARVED_BOUNDARIES_BEFORE_PLAIN - 1):
            self.assertIsNone(generator._demote_starved_mtp_lane())
        with self.assertLogs(level="WARNING"):
            self.assertEqual(generator._demote_starved_mtp_lane(), 1)
        self.assertEqual(list(batch._paused), [2])
        self.assertEqual(
            [p.detached.lane.uid for p in batch.take_plain_fallbacks()], [1]
        )

    def test_starvation_needs_an_unbroken_run_and_idle_plain_batch(self):
        from mlx_lm.generate import MTP_STARVED_BOUNDARIES_BEFORE_PLAIN

        batch = self._paused_batch()
        # A decoding plain lane will free memory: never demote.
        busy = self._starved_generator(batch, plain_lanes=1)
        for _ in range(MTP_STARVED_BOUNDARIES_BEFORE_PLAIN * 2):
            self.assertIsNone(busy._demote_starved_mtp_lane())
        # A boundary with an active lane resets the count.
        idle = self._starved_generator(batch)
        for _ in range(MTP_STARVED_BOUNDARIES_BEFORE_PLAIN - 1):
            idle._demote_starved_mtp_lane()
        batch.state.lanes.append(_Lane(9))
        self.assertIsNone(idle._demote_starved_mtp_lane())
        batch.state.lanes.pop()
        self.assertIsNone(idle._demote_starved_mtp_lane())
        self.assertEqual(sorted(batch._paused), [1, 2])

    def test_async_segmented_cohort_cancels_empties_and_reconstructs(self):
        """A membership churn boundary must retire and re-arm async work."""

        class Ticket:
            def __init__(self):
                self.stream = object()
                self.cancel_count = 0

            def cancel_and_drain(self):
                self.cancel_count += 1

        def shell(lanes, *, ticket=None, pending=True):
            batch = MTPGenerationBatch.__new__(MTPGenerationBatch)
            batch.model = object()
            batch.state = _state([_Detached(lane) for lane in lanes])
            batch.segmented_live_tip = True
            batch.async_qsa_promotion = True
            batch.stop_matchers = [StopSequenceMatcher() for _ in lanes]
            batch._matcher_states = [matcher.make_state() for matcher in batch.stop_matchers]
            batch._num_tokens = [0] * len(lanes)
            batch._initial_outputs = [None] * len(lanes)
            batch._paused = {}
            batch._plain_ready = []
            batch.mtp_admission = None
            batch._async_qsa_ticket = ticket
            batch._async_qsa_pending = pending
            batch._async_qsa_receipt = None
            batch._async_qsa_receipts_by_uid = {
                lane.uid: {"queued": True} for lane in lanes
            }
            return batch

        ticket = Ticket()
        active = shell([_Lane(1), _Lane(2)], ticket=ticket)
        joining = shell([_Lane(3)], pending=False)
        armed = []

        def attach(_model, state, packages):
            state.lanes.extend(package.lane for package in packages)
            state.membership_epoch += 1
            return state

        with (
            patch(
                "mlx_lm.hybrid_speculative.detach_self_mtp_lanes",
                side_effect=_detach,
            ),
            patch(
                "mlx_lm.hybrid_speculative.attach_segmented_self_mtp_lanes",
                side_effect=attach,
            ),
            patch("mlx_lm.generate._close_segmented_detached"),
            patch.object(active, "_arm_async_qsa_promotion", side_effect=lambda: armed.append(True)),
        ):
            # Cancellation/exit of the whole cohort drains the queued device
            # work before any cache row is released.
            active.remove_uids([1, 2])
            self.assertEqual(ticket.cancel_count, 1)
            self.assertEqual(active.uids, [])
            self.assertTrue(active.segmented_live_tip)
            self.assertTrue(active._async_qsa_pending)
            self.assertIsNone(active._async_qsa_ticket)
            self.assertEqual(active._async_qsa_receipts_by_uid, {})

            # A later join reconstructs the empty segmented cohort and queues
            # one fresh async promotion against the new membership epoch.
            active.extend(joining)

        self.assertEqual(active.uids, [3])
        self.assertEqual(joining.uids, [])
        self.assertEqual(len(armed), 1)

    def test_parallel_logits_processors_fail_closed_to_plain(self):
        # lane_rng=None: the processor route must be decided BEFORE the
        # lane-RNG check, config validation, and RNG fork can raise.
        fake = types.SimpleNamespace(insert=lambda **_kwargs: [0, 1])
        processor = lambda _tokens, logits: logits
        with patch("mlx_lm.generate.BatchGenerator", return_value=fake):
            generator = ParallelSampleGenerator(
                object(),
                [],
                7,
                2,
                max_tokens=4,
                logits_processors=[[processor], [processor]],
                self_mtp={"persistent": True, "num_draft": 2},
                lane_rng=None,
            )
        self.assertEqual(generator.active_samples, [0, 1])

    def test_migrated_plain_lane_emits_the_prepared_first_token(self):
        from mlx_lm.generate import BatchGenerator, _PausedMTPGenerationLane

        # max_tokens=1: the join->plain migration's ENTIRE output is the
        # prepared token; it must be emitted, terminal, and complete.
        lane = _Lane(6, maximum=1)
        lane.cur = 42
        matcher = StopSequenceMatcher()
        package = _PausedMTPGenerationLane(
            _Detached(lane),
            MTPToken(42, mx.array([-0.5, -1.0]), False),
            matcher,
            matcher.make_state(),
            0,
        )
        generator = BatchGenerator.__new__(BatchGenerator)
        generator.self_mtp = {"num_draft": 2}
        generator._generation_batch = types.SimpleNamespace(
            take_plain_fallbacks=lambda: [package]
        )

        def refuse(_batch):
            raise AssertionError("a terminal prepared token needs no plain row")

        generator._plain_fallback_batch = types.SimpleNamespace(extend=refuse)
        responses = generator._migrate_plain_fallbacks()
        self.assertEqual([r.token for r in responses], [42])
        self.assertEqual([r.finish_reason for r in responses], ["length"])
        self.assertIsNotNone(responses[0].prompt_cache)
        self.assertIsNotNone(responses[0].mtp_state)

        # With budget left, the prepared token is emitted AND the plain
        # continuation budget is max_tokens minus the one already-counted
        # token — never one fewer.
        lane2 = _Lane(9, maximum=3)
        lane2.cur = 5
        matcher2 = StopSequenceMatcher()
        package2 = _PausedMTPGenerationLane(
            _Detached(lane2),
            MTPToken(5, mx.array([-0.5, -1.0]), True),
            matcher2,
            matcher2.make_state(),
            0,
        )
        generator._generation_batch = types.SimpleNamespace(
            take_plain_fallbacks=lambda: [package2]
        )
        extended = []
        generator._plain_fallback_batch = types.SimpleNamespace(
            extend=extended.append
        )
        generator.model = object()
        generator.sampler = None
        captured = {}

        def fake_plain(_model, uids, _tokens, _caches, _prefixes, _samplers,
                       _default, _processors, _matchers, remaining):
            captured["uids"] = uids
            captured["remaining"] = remaining
            return types.SimpleNamespace(_matcher_states=[None])

        with (
            patch("mlx_lm.generate.GenerationBatch", side_effect=fake_plain),
            patch("mlx_lm.generate._merge_caches", return_value=[]),
        ):
            responses = generator._migrate_plain_fallbacks()
        self.assertEqual([r.token for r in responses], [5])
        self.assertIsNone(responses[0].finish_reason)
        self.assertEqual(captured["uids"], [9])
        self.assertEqual(captured["remaining"], [2])
        self.assertEqual(len(extended), 1)

        # A lane that already emitted its tokens (no pending initial) still
        # migrates exactly as before.
        lane3 = _Lane(11, maximum=4)
        lane3.ntoks = 2
        matcher3 = StopSequenceMatcher()
        package3 = _PausedMTPGenerationLane(
            _Detached(lane3),
            None,
            matcher3,
            matcher3.make_state(),
            2,
        )
        generator._generation_batch = types.SimpleNamespace(
            take_plain_fallbacks=lambda: [package3]
        )
        with (
            patch("mlx_lm.generate.GenerationBatch", side_effect=fake_plain),
            patch("mlx_lm.generate._merge_caches", return_value=[]),
        ):
            responses = generator._migrate_plain_fallbacks()
        self.assertEqual(responses, [])
        self.assertEqual(captured["remaining"], [2])

    def test_joining_lane_caches_are_budgeted_before_allocation(self):
        from collections import deque

        from mlx_lm.generate import BatchGenerator

        generator = BatchGenerator.__new__(BatchGenerator)
        generator.prefill_step_size = 2048
        generator.self_mtp = {"num_draft": 2}
        generator._mtp_configs = {7: {"num_draft": 1}}
        # uid 7 restores an APC draft sidecar: its actual bytes are part of
        # the pre-allocation budget, not only the later merge-boundary check.
        generator._mtp_states = {
            7: ([types.SimpleNamespace(nbytes=1 << 30)], object())
        }
        generator._generation_batch = types.SimpleNamespace(
            mtp_cycle_state=lambda: [(1, 1025, 2, True, 0.5)]
        )
        cache_row = [types.SimpleNamespace(nbytes=1 << 30)]
        generator._unprocessed_sequences = deque(
            (
                (7, [[1, 2], [3]], 8, cache_row, [9], None, [], None),
                (8, [[4, 5, 6]], 8, cache_row, [], None, [], None),
            )
        )
        seen = {}

        def admission(rows):
            seen["rows"] = tuple(rows)
            return {1: 2, 7: 1, 8: "queue"}

        generator.mtp_admission = admission
        # Live rows and both joining candidates reach the controller BEFORE
        # any allocation.  These cache leaves report no offset, so nothing is
        # known to be restored: both joins report 0 GiB and pay the envelope.
        self.assertEqual(generator._admit_mtp_joining(2), 1)
        self.assertEqual(
            seen["rows"],
            ((1, 1025, 2, True, 0.5), (7, 4, 1, False, 0.0), (8, 3, 2, False, 0.0)),
        )
        # A queued head keeps the FIFO prefix closed (fail closed, in order).
        generator.mtp_admission = lambda rows: {7: "queue", 8: 2}
        self.assertEqual(generator._admit_mtp_joining(2), 0)
        # With nothing running, waiting frees nothing: the head proceeds.
        generator._generation_batch = types.SimpleNamespace(
            mtp_cycle_state=lambda: []
        )
        generator._plain_fallback_batch = [object()]
        self.assertEqual(generator._admit_mtp_joining(2), 0)
        generator._plain_fallback_batch = []
        with self.assertLogs(level="WARNING"):
            self.assertEqual(generator._admit_mtp_joining(2), 1)
        generator._generation_batch = types.SimpleNamespace(
            mtp_cycle_state=lambda: [(1, 1025, 2, True, 0.5)]
        )
        # A plain-only approval still admits; the merge boundary migrates it.
        generator.mtp_admission = lambda rows: {7: "plain", 8: 2}
        self.assertEqual(generator._admit_mtp_joining(2), 2)
        # No callback keeps the pre-admission behavior.
        generator.mtp_admission = None
        self.assertEqual(generator._admit_mtp_joining(2), 2)

    def test_joining_budget_projects_suffix_growth_and_fresh_draft(self):
        from collections import deque

        from mlx_lm.generate import BatchGenerator

        generator = BatchGenerator.__new__(BatchGenerator)
        generator.prefill_step_size = 2048
        generator.self_mtp = {"num_draft": 2}
        generator._mtp_configs = {}
        generator._mtp_states = {}
        generator._generation_batch = types.SimpleNamespace(
            mtp_cycle_state=lambda: []
        )
        # 4-layer cache: 1 GiB restored covering 512 of 1024 context tokens.
        cache_row = [
            types.SimpleNamespace(nbytes=1 << 28, offset=512) for _ in range(4)
        ]
        generator._unprocessed_sequences = deque(
            ((5, [[1, 2, 3], [4]], 8, cache_row, [0] * 1020, None, [], None),)
        )
        seen = {}

        def admission(rows):
            seen["rows"] = tuple(rows)
            return {5: 2}

        generator.mtp_admission = admission
        self.assertEqual(generator._admit_mtp_joining(1), 1)
        # Target projected to the full context (1 GiB * 1024/512 = 2 GiB) plus
        # the fresh single-layer draft floor (2 GiB / 4 layers = 0.5 GiB).
        self.assertEqual(seen["rows"], ((5, 1024, 2, False, 2.5),))

        # A long uncached tail allocates prefill work it does not report.
        generator.prefill_step_size = 256
        generator._unprocessed_sequences = deque(
            ((5, [[1, 2, 3], [4]], 8, cache_row, [0] * 1020, None, [], None),)
        )
        self.assertEqual(generator._admit_mtp_joining(1), 1)
        self.assertEqual(seen["rows"], ((5, 1024, 2, False, 0.0),))

    def test_joining_budget_walks_nested_cache_lists(self):
        from collections import deque

        from mlx_lm.generate import BatchGenerator
        from mlx_lm.models.cache import CacheList

        generator = BatchGenerator.__new__(BatchGenerator)
        generator.prefill_step_size = 2048
        generator.self_mtp = {"num_draft": 2}
        generator._mtp_configs = {}
        generator._mtp_states = {}
        generator._generation_batch = types.SimpleNamespace(
            mtp_cycle_state=lambda: []
        )
        leaf = lambda: types.SimpleNamespace(nbytes=1 << 28, offset=512)
        nested = CacheList.__new__(CacheList)
        nested.caches = (leaf(), leaf())
        generator._unprocessed_sequences = deque(
            ((5, [[1, 2, 3], [4]], 8, [nested, nested], [0] * 1020, None, [], None),)
        )
        seen = {}
        generator.mtp_admission = lambda rows: seen.setdefault("rows", tuple(rows)) and {5: 2}
        self.assertEqual(generator._admit_mtp_joining(1), 1)
        # 4 leaves x 0.25 GiB covering 512 of 1024 tokens -> 2 GiB target,
        # plus the one-layer fresh draft share (2 GiB / 2 entries = 1 GiB).
        (row,) = seen["rows"]
        self.assertEqual(row[:4], (5, 1024, 2, False))
        self.assertGreaterEqual(row[4], 2.0)

    def test_prompt_cache_nbytes_counts_queued_draft_sidecars(self):
        from collections import deque

        from mlx_lm.generate import BatchGenerator

        generator = BatchGenerator.__new__(BatchGenerator)
        generator.self_mtp = {"num_draft": 1}
        generator._unprocessed_sequences = deque(
            ((0, [[1]], 8, [types.SimpleNamespace(nbytes=100)], [], None, [], None),)
        )
        # uid 0 retains a restored draft sidecar; a None placeholder must not
        # break the sum.
        generator._mtp_states = {
            0: ([types.SimpleNamespace(nbytes=40)], object()),
            1: None,
        }
        generator._prompt_batch = types.SimpleNamespace(prompt_cache=[])
        generator._generation_batch = types.SimpleNamespace(cache_nbytes=7)
        generator._plain_fallback_batch = types.SimpleNamespace(prompt_cache=[])
        self.assertEqual(generator.prompt_cache_nbytes, 100 + 40 + 7)

    def test_plain_generator_does_not_retain_mtp_maps(self):
        from collections import deque

        from mlx_lm.generate import BatchGenerator

        generator = BatchGenerator.__new__(BatchGenerator)
        generator.self_mtp = None
        generator.kv_bits = None
        generator.max_tokens = 8
        generator.logits_processors = []
        generator._default_stop_matcher = StopSequenceMatcher()
        generator._unprocessed_sequences = deque()
        generator._mtp_states = {}
        generator._mtp_lane_rngs = {}
        generator._mtp_configs = {}
        generator._uid_count = 0
        sidecar = ([types.SimpleNamespace(nbytes=4)], object())
        generator.insert(
            [[1, 2, 3]],
            caches=[[types.SimpleNamespace(nbytes=0)]],
            mtp_states=[sidecar],
        )
        # The plain path never consumes these maps, so they must stay empty.
        self.assertEqual(len(generator._unprocessed_sequences), 1)
        self.assertEqual(generator._mtp_states, {})
        self.assertEqual(generator._mtp_lane_rngs, {})
        self.assertEqual(generator._mtp_configs, {})
        # An MTP generator still retains them for its preparation path.
        generator.self_mtp = {"num_draft": 1}
        (uid,) = generator.insert(
            [[4, 5, 6]],
            caches=[[types.SimpleNamespace(nbytes=0)]],
            mtp_states=[sidecar],
        )
        self.assertIs(generator._mtp_states[uid], sidecar)
        self.assertIn(uid, generator._mtp_configs)

    def test_fail_closed_config_exclusions(self):
        from mlx_lm.generate import BatchGenerator

        for config in (
            {"persistent": False},
            {"window_size": 2048},
            {"rate_gate": True},
            {"speculation_router": object()},
            {"xtc_probability": 0.1},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                BatchGenerator._validate_mtp_config(config)


if __name__ == "__main__":
    unittest.main()
