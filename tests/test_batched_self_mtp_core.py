# Copyright © 2026 Apple Inc.

"""CPU contracts for the batched persistent self-MTP transaction core.

The ship gate is distributional: transformed residual verification must emit
the batched target distribution lane by lane. Cross-shape greedy identity is a
real-hardware diagnostic; documented near-tie flips are recorded, while every
non-near-tie flip remains a hard failure.
"""

import unittest
import hashlib
import os
from collections import Counter
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.hybrid_speculative import (
    BatchedSelfMTPState,
    MTPToken,
    SelfMTPCachePair,
    _batched_residual_verify,
    _draw_mtp_acceptance_uniforms,
    _make_sampling_transform,
    _sample_from_logprobs,
    attach_self_mtp_lanes,
    classify_greedy_batch_divergence,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
    require_only_near_tie_greedy_flips,
    self_mtp_generate_step,
)
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.utils import load


def _tiny_args():
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


def _tv(counts, probabilities):
    total = sum(counts.values())
    return 0.5 * sum(
        abs(counts.get(index, 0) / total - probability)
        for index, probability in enumerate(probabilities)
    )


class _CPUCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._previous_device)


class TestFrozenSurface(_CPUCase):
    def test_mtp_token_is_immutable_and_matches_generator_semantics(self):
        token = MTPToken(7, mx.array([0.0, -1.0]), True)
        self.assertEqual(token.token, 7)
        self.assertTrue(token.from_draft)
        with self.assertRaises(FrozenInstanceError):
            token.token = 8

    def test_transaction_guards_membership_while_open(self):
        batch = BatchedSelfMTPState(
            lanes=[],
            caches=SelfMTPCachePair(target=[], draft=[]),
            membership_epoch=4,
            proposal_open=True,
        )
        with self.assertRaisesRegex(RuntimeError, "proposal is open"):
            attach_self_mtp_lanes(object(), batch, [])
        with self.assertRaisesRegex(RuntimeError, "proposal is open"):
            detach_self_mtp_lanes(object(), batch, [])


class TestDistributionalContract(_CPUCase):
    def test_acceptance_uniforms_use_each_lanes_native_k(self):
        shapes = []
        draws = []
        for seed, k in ((11, 1), (12, 2), (13, 3)):
            lane = LaneRNG(seed)
            values = _draw_mtp_acceptance_uniforms(k, rng=lane)
            mx.eval(values)
            shapes.append(tuple(values.shape))
            draws.append(lane.draws)
        self.assertEqual(shapes, [(1,), (2,), (3,)])
        self.assertEqual(draws, [1, 1, 1])

    def test_transformed_residual_committed_marginal_is_target(self):
        transform = _make_sampling_transform(
            0.8, top_p=1.0, top_k=3, min_p=0.0
        )
        target = transform(mx.array([3.0, 2.4, 1.8, -1.0, -1.5, -2.0]))
        draft = transform(mx.array([-1.0, -1.5, 1.8, 3.0, 2.4, -2.0]))
        target_rows = mx.stack([target, target])
        counts = Counter()
        n = 3000
        for trial in range(n):
            lane = LaneRNG(50_000 + trial)
            proposal = _sample_from_logprobs(draft, 0.8, rng=lane)
            accepted, correction = _batched_residual_verify(
                target_rows,
                [draft],
                [proposal],
                0.8,
                rng=lane,
            )
            counts[proposal if accepted else correction] += 1
            # Draft categorical, native (1,) acceptance uniform, then the
            # correction/bonus categorical: exactly one lane's trace.
            self.assertEqual(lane.draws, 3)
        self.assertLess(_tv(counts, mx.exp(target).tolist()), 0.05)


class TestGreedyDigestDiagnostic(_CPUCase):
    def test_near_tie_flip_is_recorded_not_rejected(self):
        single = mx.array([[1.0, 0.9985, -2.0], [2.0, 0.0, -1.0]])
        batched = mx.array([[0.9984, 1.0001, -2.0], [2.0, 0.0, -1.0]])
        diagnostic = require_only_near_tie_greedy_flips(single, batched)
        self.assertEqual(diagnostic.compared, 2)
        self.assertEqual(diagnostic.flip_positions, (0,))
        self.assertEqual(diagnostic.near_tie_flip_positions, (0,))
        self.assertEqual(diagnostic.non_near_tie_flip_positions, ())

    def test_non_near_tie_flip_fails_loudly(self):
        single = mx.array([[3.0, 1.0, 0.0]])
        batched = mx.array([[1.0, 3.0, 0.0]])
        diagnostic = classify_greedy_batch_divergence(single, batched)
        self.assertEqual(diagnostic.non_near_tie_flip_positions, (0,))
        with self.assertRaisesRegex(AssertionError, "outside.*near-tie"):
            require_only_near_tie_greedy_flips(single, batched)


class TestBatchedCoreLifecycle(_CPUCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        mx.random.seed(7)
        cls.model = TextModel(_tiny_args())
        mx.eval(cls.model.parameters())

    def _lane(self, uid, prompt, *, maximum=9, temperature=0.0, top_k=0):
        return prepare_self_mtp_lane(
            mx.array(prompt, mx.uint32),
            self.model,
            uid=uid,
            max_tokens=maximum,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=LaneRNG(100 + uid),
            num_draft=2,
            sampling_temp=temperature,
            sampling_top_p=1.0,
            sampling_top_k=top_k,
            sampling_min_p=0.0,
            accept_rule="residual",
            logits_processors=[],
            prefill_step_size=4,
            share_qsa_indices=False,
        )

    @staticmethod
    def _digest(tokens):
        return hashlib.sha256(
            b"".join(int(token).to_bytes(4, "little") for token in tokens)
        ).hexdigest()

    def _finish(self, batch, traces):
        while batch.lanes:
            proposal = propose_batched_self_mtp(self.model, batch)
            mx.eval(
                [output.logprobs for row in proposal.outputs for output in row]
            )
            terminal = []
            for lane, outputs in zip(batch.lanes, proposal.outputs):
                traces[lane.uid].extend(
                    (output.token, output.logprobs) for output in outputs
                )
                terminal.append(lane.ntoks + len(outputs) >= lane.max_tokens)
            commit_batched_self_mtp(
                batch,
                proposal,
                emitted_counts=[len(row) for row in proposal.outputs],
                terminal=terminal,
            )
            leaving = [i for i, value in enumerate(terminal) if value]
            if leaving:
                batch, _ = detach_self_mtp_lanes(self.model, batch, leaving)
        return traces

    def _incumbent_trace(self, uid, prompt, *, maximum=8):
        """Authoritative oracle: the incumbent single-lane generator.

        The digest contract is that batched lane i matches single-lane
        self-MTP, so the reference trace comes from
        ``self_mtp_generate_step`` — never from the batched engine at B=1.
        """
        trace = []
        for token, logprobs, _from_draft in self_mtp_generate_step(
            mx.array(prompt, mx.uint32),
            self.model,
            num_draft=2,
            max_tokens=maximum,
            prefill_step_size=4,
            sampling_temp=0.0,
            sampling_top_p=1.0,
            sampling_top_k=0,
            sampling_min_p=0.0,
            accept_rule="residual",
            persistent_mtp=True,
            lane_rng=LaneRNG(100 + uid),
        ):
            trace.append((int(token), mx.reshape(logprobs, (-1,))))
        return trace

    def _run_prompts(self, rows, *, maximum=8):
        prepared = [
            (uid, *self._lane(uid, prompt, maximum=maximum))
            for uid, prompt in rows
        ]
        traces = {
            uid: [(first.token, first.logprobs)]
            for uid, _lane, first in prepared
        }
        batch = attach_self_mtp_lanes(
            self.model, None, [lane for _uid, lane, _first in prepared]
        )
        return self._finish(batch, traces)

    def assert_trace_diagnostic(self, reference, candidate):
        self.assertEqual(len(reference), len(candidate))
        reference_tokens = [token for token, _ in reference]
        candidate_tokens = [token for token, _ in candidate]
        reference_digest = self._digest(reference_tokens)
        candidate_digest = self._digest(candidate_tokens)
        if reference_tokens == candidate_tokens:
            self.assertEqual(reference_digest, candidate_digest)
            return
        # After the first allowed near-tie flip the two token histories
        # diverge, so later positions are no longer same-prefix comparable:
        # classify ONLY through the first divergence and stop there. The
        # trace-length equality above still holds for the whole trace.
        first = next(
            index
            for index, (expected, got) in enumerate(
                zip(reference_tokens, candidate_tokens)
            )
            if expected != got
        )
        diagnostic = require_only_near_tie_greedy_flips(
            mx.stack(
                [
                    mx.reshape(logprobs, (-1,))
                    for _, logprobs in reference[: first + 1]
                ]
            ),
            mx.stack(
                [
                    mx.reshape(logprobs, (-1,))
                    for _, logprobs in candidate[: first + 1]
                ]
            ),
        )
        self.assertEqual(diagnostic.near_tie_flip_positions, (first,))
        self.assertFalse(diagnostic.non_near_tie_flip_positions)

    def test_ragged_commit_detach_rejoin_and_empty_batch(self):
        lane0, first0 = self._lane(10, [1, 2, 3, 4, 5])
        lane1, first1 = self._lane(20, [6, 7, 8, 9, 10, 11, 12])
        self.assertIsInstance(first0, MTPToken)
        self.assertIsInstance(first1, MTPToken)

        batch = attach_self_mtp_lanes(self.model, None, [lane0, lane1])
        proposal = propose_batched_self_mtp(self.model, batch)
        self.assertEqual(proposal.lane_uids, (10, 20))
        self.assertEqual(proposal.draft_depths, (2, 2))
        self.assertEqual(
            tuple(len(row) for row in proposal.outputs),
            tuple(value + 1 for value in proposal.accepted_lengths),
        )
        commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[len(row) for row in proposal.outputs],
            terminal=[False, False],
        )
        self.assertFalse(batch.proposal_open)

        previous_epoch = batch.membership_epoch
        batch, detached = detach_self_mtp_lanes(self.model, batch, [1])
        self.assertEqual([item.lane.uid for item in detached], [20])
        self.assertEqual(batch.membership_epoch, previous_epoch + 1)
        batch = attach_self_mtp_lanes(self.model, batch, detached)
        self.assertEqual([lane.uid for lane in batch.lanes], [10, 20])

        # A terminal lane that consumes none of its burst forces the delivery
        # tail trim before extraction; the unaffected lane commits normally.
        proposal = propose_batched_self_mtp(self.model, batch)
        commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[len(proposal.outputs[0]), 0],
            terminal=[False, True],
        )
        batch, terminal_lane = detach_self_mtp_lanes(self.model, batch, [1])
        self.assertEqual(terminal_lane[0].lane.uid, 20)

        batch, last_lane = detach_self_mtp_lanes(self.model, batch, [0])
        self.assertEqual(last_lane[0].lane.uid, 10)
        self.assertEqual(batch.lanes, [])
        self.assertEqual(batch.caches.target, [])
        self.assertEqual(batch.caches.draft, [])

    def test_rng_draw_counts_and_acceptance_shapes_follow_each_lane_k(self):
        lane0, first0 = self._lane(
            50, [1, 2, 3, 4], maximum=3, temperature=0.8, top_k=8
        )
        lane1, first1 = self._lane(
            60, [5, 6, 7, 8, 9], maximum=4, temperature=0.8, top_k=8
        )
        self.assertIsInstance(first0, MTPToken)
        self.assertIsInstance(first1, MTPToken)
        batch = attach_self_mtp_lanes(self.model, None, [lane0, lane1])
        shapes = []
        original = _draw_mtp_acceptance_uniforms

        def observe(k, *, rng=None):
            before = rng.draws
            values = original(k, rng=rng)
            shapes.append((tuple(values.shape), before, rng.draws))
            return values

        with patch(
            "mlx_lm.hybrid_speculative._draw_mtp_acceptance_uniforms",
            side_effect=observe,
        ):
            proposal = propose_batched_self_mtp(self.model, batch)
        self.assertEqual(proposal.draft_depths, (1, 2))
        self.assertEqual([shape for shape, _, _ in shapes], [(1,), (2,)])
        self.assertTrue(all(after == before + 1 for _, before, after in shapes))
        self.assertEqual([lane.rng.draws for lane in batch.lanes], [4, 5])
        commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[len(row) for row in proposal.outputs],
            terminal=[True, True],
        )

    def test_full_digest_diagnostic_permutations_and_early_join(self):
        # Four simultaneous lanes at k=2: the production steady-state
        # M=(k+1)N = 12 verify shape is exercised, not just N<=3.
        rows = [
            (10, [1, 2, 3, 4, 5]),
            (20, [6, 7, 8, 9, 10, 11]),
            (30, [12, 13, 14, 15, 16, 17, 18]),
            (40, [19, 20, 21, 22, 23]),
        ]
        singles = {
            uid: self._incumbent_trace(uid, prompt, maximum=8)
            for uid, prompt in rows
        }
        for order in ((0, 1, 2, 3), (2, 0, 3, 1), (3, 1, 0, 2)):
            traces = self._run_prompts([rows[index] for index in order])
            for uid, _prompt in rows:
                self.assert_trace_diagnostic(singles[uid], traces[uid])

        prepared = [(uid, *self._lane(uid, prompt, maximum=8)) for uid, prompt in rows]
        traces = {
            uid: [(first.token, first.logprobs)]
            for uid, _lane, first in prepared
        }
        batch = attach_self_mtp_lanes(
            self.model, None, [lane for _uid, lane, _first in prepared]
        )
        self.assertEqual(len(batch.lanes), 4)
        proposal = propose_batched_self_mtp(self.model, batch)
        # N=4 at k=2: this cycle's verify forward is the M=12 shape.
        self.assertEqual(proposal.draft_depths, (2, 2, 2, 2))
        for lane, outputs in zip(batch.lanes, proposal.outputs):
            traces[lane.uid].extend((output.token, output.logprobs) for output in outputs)
        commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[len(row) for row in proposal.outputs],
            terminal=[False, False, False, False],
        )
        batch, _ = detach_self_mtp_lanes(self.model, batch, [3])

        join_uid, join_prompt = 50, [24, 25, 26, 27, 28, 29]
        joining, first = self._lane(join_uid, join_prompt, maximum=8)
        traces[join_uid] = [(first.token, first.logprobs)]
        batch = attach_self_mtp_lanes(self.model, batch, [joining])
        self.assertEqual(len(batch.lanes), 4)
        self._finish(batch, traces)

        for uid in (10, 20, 30):
            self.assert_trace_diagnostic(singles[uid], traces[uid])
        join_reference = self._incumbent_trace(join_uid, join_prompt, maximum=8)
        self.assert_trace_diagnostic(join_reference, traces[join_uid])


@unittest.skipUnless(
    os.environ.get("MLX_BATCHED_MTP_GATE_MODEL"),
    "set MLX_BATCHED_MTP_GATE_MODEL to the production Qwen3.8 artifact",
)
class TestProductionQwen38DigestGate(unittest.TestCase):
    """Real-M5 digest battery; near-tie-only flips remain diagnostic."""

    _digest = staticmethod(TestBatchedCoreLifecycle._digest)
    _lane = TestBatchedCoreLifecycle._lane
    _finish = TestBatchedCoreLifecycle._finish
    _run_prompts = TestBatchedCoreLifecycle._run_prompts
    _incumbent_trace = TestBatchedCoreLifecycle._incumbent_trace
    assert_trace_diagnostic = TestBatchedCoreLifecycle.assert_trace_diagnostic
    test_full_digest_diagnostic_permutations_and_early_join = (
        TestBatchedCoreLifecycle.test_full_digest_diagnostic_permutations_and_early_join
    )

    @classmethod
    def setUpClass(cls):
        cls.model, _ = load(os.environ["MLX_BATCHED_MTP_GATE_MODEL"])
        if getattr(cls.model, "mtp", None) is None:
            raise AssertionError("production digest gate model has no MTP head")


if __name__ == "__main__":
    unittest.main()
