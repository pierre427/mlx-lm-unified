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
        # Live rows and both joining candidates (context, configured depth,
        # retained target + draft cache GiB) reach the controller BEFORE any
        # allocation.  uid 8 has no sidecar, so its fresh single-layer draft
        # cache takes the one-layer share of the target (1 layer here: 1 GiB).
        self.assertEqual(generator._admit_mtp_joining(2), 1)
        self.assertEqual(
            seen["rows"],
            ((1, 1025, 2, True, 0.5), (7, 4, 1, False, 2.0), (8, 3, 2, False, 2.0)),
        )
        # A queued head keeps the FIFO prefix closed (fail closed, in order).
        generator.mtp_admission = lambda rows: {7: "queue", 8: 2}
        self.assertEqual(generator._admit_mtp_joining(2), 0)
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
