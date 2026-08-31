# Copyright © 2026 Apple Inc.

"""Regression contracts adopted from Rapid-MLX continuous MTP hardening."""

import types
import unittest
from unittest.mock import patch

from mlx_lm.hybrid_speculative import (
    BatchedSelfMTPState,
    DetachedSelfMTPLane,
    SelfMTPCachePair,
    SelfMTPCycleResult,
    abort_batched_self_mtp,
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    propose_batched_self_mtp,
)
from mlx_lm.generate import MTPGenerationBatch


class _Cache:
    def __init__(self, rows, *, fail_merge=False, fail_extract=None):
        self.rows = list(rows)
        self.fail_merge = fail_merge
        self.fail_extract = fail_extract
        self.speculating = True

    @classmethod
    def merge(cls, caches):
        if any(cache.fail_merge for cache in caches):
            raise MemoryError("late replacement merge failed")
        return cls([row for cache in caches for row in cache.rows])

    def extract(self, index):
        if index == self.fail_extract:
            raise MemoryError("late extract failed")
        return type(self)(
            [self.rows[index]],
            fail_merge=self.fail_merge,
            fail_extract=self.fail_extract,
        )

    def stop_speculation(self):
        self.speculating = False

    def start_speculation(self):
        self.speculating = True


def _lane(uid):
    return types.SimpleNamespace(
        uid=uid,
        num_draft=2,
        share_qsa_indices=False,
    )


def _start(caches, _required, _message):
    for cache in caches:
        cache.start_speculation()


class TestAtomicMembership(unittest.TestCase):
    def test_late_attach_failure_keeps_live_pair_and_membership(self):
        original = SelfMTPCachePair(
            target=[_Cache(["t0"]), _Cache(["t1"])],
            draft=[_Cache(["d0"]), _Cache(["d1"])],
        )
        batch = BatchedSelfMTPState([_lane(1)], original, membership_epoch=7)
        joining = DetachedSelfMTPLane(
            _lane(2),
            SelfMTPCachePair(
                target=[_Cache(["t2"]), _Cache(["t3"], fail_merge=True)],
                draft=[_Cache(["d2"]), _Cache(["d3"])],
            ),
        )

        with (
            patch("mlx_lm.hybrid_speculative._validate_detached_self_mtp"),
            patch(
                "mlx_lm.hybrid_speculative._start_speculation_or_cleanup",
                side_effect=_start,
            ),
            self.assertRaisesRegex(MemoryError, "replacement merge"),
        ):
            attach_self_mtp_lanes(object(), batch, [joining])

        self.assertIs(batch.caches, original)
        self.assertEqual([lane.uid for lane in batch.lanes], [1])
        self.assertEqual(batch.membership_epoch, 7)
        self.assertFalse(batch.poisoned)
        self.assertTrue(all(cache.speculating for cache in original.target))

    def test_late_detach_failure_keeps_live_pair_and_membership(self):
        original = SelfMTPCachePair(
            target=[
                _Cache(["t0", "t1"]),
                _Cache(["u0", "u1"], fail_extract=1),
            ],
            draft=[_Cache(["d0", "d1"]), _Cache(["e0", "e1"])],
        )
        batch = BatchedSelfMTPState(
            [_lane(1), _lane(2)],
            original,
            membership_epoch=11,
        )

        with (
            patch(
                "mlx_lm.hybrid_speculative._start_speculation_or_cleanup",
                side_effect=_start,
            ),
            self.assertRaisesRegex(MemoryError, "late extract"),
        ):
            detach_self_mtp_lanes(object(), batch, [1])

        self.assertIs(batch.caches, original)
        self.assertEqual([lane.uid for lane in batch.lanes], [1, 2])
        self.assertEqual(batch.membership_epoch, 11)
        self.assertFalse(batch.poisoned)
        self.assertTrue(all(cache.speculating for cache in original.target))


class TestPoisonedTransaction(unittest.TestCase):
    def test_proposal_failure_poisons_and_forbids_reuse(self):
        batch = BatchedSelfMTPState(
            [_lane(1)],
            SelfMTPCachePair([], []),
            membership_epoch=1,
        )
        with (
            patch(
                "mlx_lm.hybrid_speculative._propose_batched_self_mtp_impl",
                side_effect=RuntimeError("verify failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "verify failed"),
        ):
            propose_batched_self_mtp(object(), batch)

        self.assertTrue(batch.poisoned)
        self.assertIn("rollback unproved", batch.poison_reason)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            detach_self_mtp_lanes(object(), batch, [])

    def test_explicit_abort_closes_and_isolates_open_proposal(self):
        proposal = SelfMTPCycleResult(1, (1,), (0,), (0,), (0,), (0,), ((),))
        batch = BatchedSelfMTPState(
            [_lane(1)],
            SelfMTPCachePair([], []),
            membership_epoch=1,
            proposal_open=True,
            _open_proposal=proposal,
        )

        abort_batched_self_mtp(batch, proposal, cause=RuntimeError("cancelled"))

        self.assertFalse(batch.proposal_open)
        self.assertIsNone(batch._open_proposal)
        self.assertTrue(batch.poisoned)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            attach_self_mtp_lanes(object(), batch, [])

    def test_delivery_exception_invokes_abort_boundary(self):
        lane = _lane(1)
        lane.max_tokens = 4
        state = types.SimpleNamespace(lanes=[lane], proposal_open=True)
        proposal = types.SimpleNamespace(
            outputs=((types.SimpleNamespace(token=7),),),
        )
        generation = MTPGenerationBatch.__new__(MTPGenerationBatch)
        generation.model = object()
        generation.state = state
        generation._initial_outputs = [None]
        generation._apply_admission = lambda: None
        generation._num_tokens = [0]
        generation._matcher_states = [object()]
        generation.stop_matchers = [object()]

        with (
            patch(
                "mlx_lm.hybrid_speculative.propose_batched_self_mtp",
                return_value=proposal,
            ),
            patch("mlx_lm.hybrid_speculative.abort_batched_self_mtp") as abort,
            patch.object(
                generation,
                "_finish_reason",
                side_effect=RuntimeError("delivery failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "delivery failed"),
        ):
            generation.next()

        abort.assert_called_once()
        self.assertIs(abort.call_args.args[0], state)
        self.assertIs(abort.call_args.args[1], proposal)

    def test_commit_mutation_failure_poisons_and_closes(self):
        lane = _lane(1)
        proposal = SelfMTPCycleResult(
            1,
            (1,),
            (0,),
            (0,),
            (0,),
            (0,),
            ((),),
        )
        batch = BatchedSelfMTPState(
            [lane],
            SelfMTPCachePair([], []),
            membership_epoch=1,
            proposal_open=True,
            _open_proposal=proposal,
        )
        with (
            patch(
                "mlx_lm.hybrid_speculative.trim_ragged_prompt_cache",
                side_effect=RuntimeError("trim failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "trim failed"),
        ):
            commit_batched_self_mtp(
                batch,
                proposal,
                emitted_counts=[0],
                terminal=[True],
            )

        self.assertTrue(batch.poisoned)
        self.assertFalse(batch.proposal_open)
        self.assertIsNone(batch._open_proposal)


if __name__ == "__main__":
    unittest.main()
