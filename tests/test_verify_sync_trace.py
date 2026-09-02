# Copyright © 2026 Apple Inc.

import hashlib
import os
import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.hybrid_speculative import (
    HybridStats,
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
    self_mtp_generate_step,
    trace_verify_syncs,
    verify_sync_status,
)
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.verify_sync import record_verify_sync, verify_sync_round
from test_batched_self_mtp_qwen4 import _tiny_qwen4_model


class TestVerifySyncTrace(unittest.TestCase):
    BASELINE_SITES = {
        "hybrid.greedy.targets_tolist": 1,
        "hybrid.sample.argmax_item": 2,
        "qwen4.ple.ids_eval": 1,
        "qwen4.ple.mask_asarray": 1,
        "qwen4.ple.mask_tail_asarray": 1,
    }

    @classmethod
    def setUpClass(cls):
        cls.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        mx.random.seed(41)
        cls.model = _tiny_qwen4_model()

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.previous_device)

    def _batch(self, *, uid=0, maximum=8, return_first=False, rng_seed=None):
        lane, first = prepare_self_mtp_lane(
            mx.array([1, 2, 3, 4, 5], mx.uint32),
            self.model,
            uid=uid,
            max_tokens=maximum,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=LaneRNG(700 + uid if rng_seed is None else rng_seed),
            num_draft=2,
            sampling_temp=0.0,
            sampling_top_p=1.0,
            sampling_top_k=0,
            sampling_min_p=0.0,
            accept_rule="residual",
            logits_processors=[],
            prefill_step_size=4,
            share_qsa_indices=False,
        )
        batch = attach_self_mtp_lanes(self.model, None, [lane])
        return (batch, first) if return_first else batch

    def test_tiny_qwen4_greedy_syncs_drop_from_six_to_four(self):
        batch = self._batch()
        with trace_verify_syncs():
            propose_batched_self_mtp(self.model, batch)
        status = verify_sync_status()

        self.assertEqual(status["round_count"], 1)
        self.assertEqual(sum(self.BASELINE_SITES.values()), 6)
        self.assertEqual(status["total"], 4)
        self.assertEqual(
            status["rounds"][0]["sites"],
            {
                "hybrid.greedy.accept_boundary": 1,
                "qwen4.ple.ids_eval": 1,
                "qwen4.ple.mask_asarray": 1,
                "qwen4.ple.mask_tail_asarray": 1,
            },
        )

    def test_tiny_qwen4_256_token_greedy_digest_is_unchanged(self):
        batch, first = self._batch(
            uid=10, maximum=256, return_first=True, rng_seed=110
        )
        tokens = [first.token]
        while len(tokens) < 256:
            proposal = propose_batched_self_mtp(self.model, batch)
            outputs = proposal.outputs[0][: 256 - len(tokens)]
            tokens.extend(output.token for output in outputs)
            terminal = len(tokens) == 256
            commit_batched_self_mtp(
                batch,
                proposal,
                emitted_counts=[len(outputs)],
                terminal=[terminal],
            )
        payload = b"".join(
            int(token).to_bytes(4, "little") for token in tokens
        )
        self.assertEqual(len(tokens), 256)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "0ca4feeb31df9c6549334315a233011827087a1c7734924ce9341873b74b506c",
        )

    def test_legacy_b1_greedy_rounds_use_one_accept_read(self):
        stats = HybridStats()
        with trace_verify_syncs():
            tokens = list(
                self_mtp_generate_step(
                    mx.array([1, 2, 3, 4, 5], mx.uint32),
                    self.model,
                    num_draft=2,
                    max_tokens=8,
                    persistent_mtp=True,
                    stats=stats,
                    lane_rng=LaneRNG(110),
                )
            )
        status = verify_sync_status()
        self.assertEqual(len(tokens), 8)
        self.assertEqual(status["round_count"], stats.draft_cycles)
        self.assertTrue(status["rounds"])
        for row in status["rounds"]:
            self.assertEqual(
                row["sites"],
                {
                    "hybrid.greedy.accept_boundary": 1,
                    "qwen4.ple.ids_eval": 1,
                },
            )

    def test_legacy_b1_256_token_digest_is_unchanged(self):
        tokens = [
            int(token)
            for token, _logprobs, _from_draft in self_mtp_generate_step(
                mx.array([1, 2, 3, 4, 5], mx.uint32),
                self.model,
                num_draft=2,
                max_tokens=256,
                persistent_mtp=True,
                lane_rng=LaneRNG(110),
            )
        ]
        payload = b"".join(
            int(token).to_bytes(4, "little") for token in tokens
        )
        self.assertEqual(len(tokens), 256)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "0ca4feeb31df9c6549334315a233011827087a1c7734924ce9341873b74b506c",
        )

    def test_environment_switch_enables_round_collection(self):
        with patch.dict(os.environ, {"MLX_LM_SYNC_TRACE": "1"}):
            with trace_verify_syncs():
                pass
            with verify_sync_round():
                record_verify_sync("test.site")
            status = verify_sync_status()
        self.assertEqual(status["rounds"][-1]["sites"], {"test.site": 1})
