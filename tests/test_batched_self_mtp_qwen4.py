# Copyright © 2026 Apple Inc.

import unittest
import os
from itertools import product
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_lm.hybrid_speculative import (
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from mlx_lm.models.qwen4_exp import (
    BatchQSAKVCache,
    Model,
    ModelArgs,
    QSAKVCache,
    Qwen4ArraysCache,
)
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.utils import load


def _tiny_qwen4_model():
    text_config = dict(
        model_type="qwen4_exp_text",
        hidden_size=32,
        intermediate_size=0,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=1,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        mtp_num_hidden_layers=1,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    mx.eval(model.parameters())
    return model


def _tree_arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tree_arrays(item)


def _prepare_lane(model, uid, prompt):
    return prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=uid,
        max_tokens=8,
        prompt_cache=None,
        mtp_state=None,
        lane_rng=LaneRNG(700 + uid),
        num_draft=2,
        sampling_temp=0.8,
        sampling_top_p=1.0,
        sampling_top_k=8,
        sampling_min_p=0.0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=4,
        share_qsa_indices=False,
    )[0]


def _forced_cycle(model, prompts, accepts, uids=None):
    # Each lane's first token is sampled from LaneRNG(700 + uid); a single-lane
    # oracle must reuse the batched lane's uid so the seeds — and thus the
    # sampled first token — match. Default keeps positional uids for the batch.
    uids = list(range(len(prompts))) if uids is None else uids
    lanes = [_prepare_lane(model, uid, prompt) for uid, prompt in zip(uids, prompts)]
    batch = attach_self_mtp_lanes(model, None, lanes)
    pending = iter(accepts)

    def force(logprobs, _draft_lps, _drafts, _temperature, *, rng=None):
        accepted = next(pending)
        return accepted, int(mx.argmax(logprobs[accepted]).item())

    with patch("mlx_lm.hybrid_speculative._batched_residual_verify", side_effect=force):
        proposal = propose_batched_self_mtp(model, batch)
    commit_batched_self_mtp(
        batch,
        proposal,
        emitted_counts=[len(row) for row in proposal.outputs],
        terminal=[False] * len(prompts),
    )
    batch, detached = detach_self_mtp_lanes(
        model, batch, list(range(len(prompts)))
    )
    return detached


class TestBatchedSelfMTPQSA(unittest.TestCase):
    def test_shared_topk_uses_each_rows_last_valid_query(self):
        cache = BatchQSAKVCache([0, 0, 0])
        cache.prepare(right_padding=[0, 2, 1])
        selected = mx.array(
            [
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                [[10, 11], [12, 13], [98, 98], [99, 99]],
                [[20, 21], [22, 23], [24, 25], [99, 99]],
            ],
            dtype=mx.uint32,
        )

        got = cache.last_valid_query(selected)
        mx.eval(got)

        np.testing.assert_array_equal(
            np.asarray(got),
            np.asarray([[6, 7], [12, 13], [24, 25]], dtype=np.uint32),
        )

    def test_shared_topk_refuses_a_fully_padded_row(self):
        cache = BatchQSAKVCache([0, 0])
        cache.prepare(right_padding=[0, 3])
        selected = mx.zeros((2, 3, 2), dtype=mx.uint32)

        with self.assertRaisesRegex(ValueError, "one valid query"):
            cache.last_valid_query(selected)

    def test_ragged_head_trim_releases_cycle_and_aligns_qsa_ledger(self):
        cache = BatchQSAKVCache([0, 0])
        cache.keys = mx.arange(2 * 1 * 6 * 2).reshape(2, 1, 6, 2)
        cache.values = cache.keys + 100
        cache._idx = 6
        cache.offset = mx.array([6, 6])
        cache.left_padding = mx.array([0, 0])
        # A two-step shared-QSA cycle appends the first raw key only.
        cache.index_keys = mx.arange(2 * 5 * 3).reshape(2, 5, 3)
        cache._mtp_share_topk = True
        cache._mtp_shared_topk = mx.array([[0, 1], [1, 2]], dtype=mx.uint32)

        drops = cache.trim_ragged([2, 1])
        mx.eval(cache.state)

        self.assertEqual(drops, [2, 1])
        self.assertEqual(cache._idx, 5)
        self.assertEqual(cache.offset.tolist(), [4, 5])
        self.assertEqual(cache.left_padding.tolist(), [1, 0])
        self.assertEqual(cache.index_keys.shape[1], cache._idx)
        self.assertIsNone(cache._mtp_shared_topk)
        self.assertFalse(cache._mtp_share_topk)
        first, second = cache.extract(0), cache.extract(1)
        self.assertEqual(first.offset, 4)
        self.assertEqual(first.index_keys.shape[1], first.offset)
        self.assertEqual(second.offset, 5)
        self.assertEqual(second.index_keys.shape[1], second.offset)


class TestQwen4ForcedAcceptanceCacheEquality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(41)
        cls.model = _tiny_qwen4_model()

    def assert_cache_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertIs(type(got), type(want))
            got_arrays = list(_tree_arrays(got.state))
            want_arrays = list(_tree_arrays(want.state))
            self.assertEqual(len(got_arrays), len(want_arrays))
            for left, right in zip(got_arrays, want_arrays):
                # numpy has no bfloat16 buffer format; the production caches are
                # bf16, so upcast to fp32 for the comparison (bit-identical bf16
                # values upcast identically — this does not loosen the check).
                np.testing.assert_allclose(
                    np.asarray(left.astype(mx.float32)),
                    np.asarray(right.astype(mx.float32)),
                    rtol=2e-4, atol=2e-5,
                )
            if isinstance(got, QSAKVCache):
                self.assertEqual(got.offset, want.offset)
                np.testing.assert_allclose(
                    np.asarray(got.index_keys.astype(mx.float32)),
                    np.asarray(want.index_keys.astype(mx.float32)),
                    rtol=2e-4,
                    atol=2e-5,
                )

    def test_all_forced_acceptance_vectors_match_independent_lanes(self):
        prompts = ([1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12])
        for accepts in product(range(3), repeat=2):
            with self.subTest(accepts=accepts):
                batched = _forced_cycle(self.model, prompts, accepts)
                singles = [
                    _forced_cycle(self.model, [prompt], [accepted], uids=[i])[0]
                    for i, (prompt, accepted) in enumerate(zip(prompts, accepts))
                ]
                for got, want in zip(batched, singles):
                    self.assert_cache_equal(got.caches.target, want.caches.target)
                    self.assert_cache_equal(got.caches.draft, want.caches.draft)
                    qwen4 = [
                        cache
                        for cache in got.caches.target
                        if isinstance(cache, Qwen4ArraysCache)
                    ]
                    self.assertEqual(len(qwen4), 1)
                    self.assertEqual(len(qwen4[0].cache), 4)
                    self.assertTrue(all(slot is not None for slot in qwen4[0].cache))
                    qsa = [
                        cache
                        for cache in got.caches.target + got.caches.draft
                        if isinstance(cache, QSAKVCache)
                    ]
                    self.assertTrue(qsa)
                    self.assertTrue(all(cache.index_keys is not None for cache in qsa))


@unittest.skipUnless(
    os.environ.get("MLX_BATCHED_MTP_GATE_MODEL"),
    "set MLX_BATCHED_MTP_GATE_MODEL to the production Qwen3.8 artifact",
)
class TestProductionQwen38ForcedCacheGate(unittest.TestCase):
    """Run the forced vector battery against the production Qwen4 caches."""

    assert_cache_equal = TestQwen4ForcedAcceptanceCacheEquality.assert_cache_equal
    test_all_forced_acceptance_vectors_match_independent_lanes = (
        TestQwen4ForcedAcceptanceCacheEquality.test_all_forced_acceptance_vectors_match_independent_lanes
    )

    @classmethod
    def setUpClass(cls):
        cls.model, _ = load(os.environ["MLX_BATCHED_MTP_GATE_MODEL"])
        if getattr(cls.model, "mtp", None) is None:
            raise AssertionError("production cache gate model has no MTP head")


if __name__ == "__main__":
    unittest.main()
