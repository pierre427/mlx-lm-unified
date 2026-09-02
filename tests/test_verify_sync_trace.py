# Copyright © 2026 Apple Inc.

import os
import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.hybrid_speculative import (
    attach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
    trace_verify_syncs,
    verify_sync_status,
)
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.verify_sync import record_verify_sync, verify_sync_round
from test_batched_self_mtp_qwen4 import _tiny_qwen4_model


class TestVerifySyncTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        mx.random.seed(41)
        cls.model = _tiny_qwen4_model()

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.previous_device)

    def _batch(self):
        lane, _ = prepare_self_mtp_lane(
            mx.array([1, 2, 3, 4, 5], mx.uint32),
            self.model,
            uid=0,
            max_tokens=8,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=LaneRNG(700),
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
        return attach_self_mtp_lanes(self.model, None, [lane])

    def test_tiny_qwen4_greedy_baseline_has_six_named_syncs(self):
        batch = self._batch()
        with trace_verify_syncs():
            propose_batched_self_mtp(self.model, batch)
        status = verify_sync_status()

        self.assertEqual(status["round_count"], 1)
        self.assertEqual(status["total"], 6)
        self.assertEqual(
            status["rounds"][0]["sites"],
            {
                "hybrid.greedy.targets_tolist": 1,
                "hybrid.sample.argmax_item": 2,
                "qwen4.ple.ids_eval": 1,
                "qwen4.ple.mask_asarray": 1,
                "qwen4.ple.mask_tail_asarray": 1,
            },
        )

    def test_environment_switch_enables_round_collection(self):
        with patch.dict(os.environ, {"MLX_LM_SYNC_TRACE": "1"}):
            with trace_verify_syncs():
                pass
            with verify_sync_round():
                record_verify_sync("test.site")
            status = verify_sync_status()
        self.assertEqual(status["rounds"][-1]["sites"], {"test.site": 1})
