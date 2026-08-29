# Copyright © 2026 Apple Inc.

"""Quantized KV batched self-MTP (opt-in via allow_quantized_kv).

The batched self-MTP transaction (merge -> start_speculation -> per-row ragged
speculative rollback -> commit -> detach) is bit-exact on a quantized target
cache: at B=1 it must equal single-lane self-MTP token+digest, and on the fp32
synthetic model B>1 equals B1 exactly (no bf16 batch-shape shift). This locks
that in, plus the opt-in policy gate (quantized is refused unless the flag is
set; windowed stays unbatchable regardless).
"""

import hashlib
import os
import types
import unittest

import mlx.core as mx

from mlx_lm.apc import AutomaticPrefixCache, MTPAPCSidecar
from mlx_lm.generate import (
    BatchGenerator,
    maybe_quantize_kv_cache,
    prefill_prompt_cache,
)
from mlx_lm.hybrid_speculative import (
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
    self_mtp_generate_step,
)
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.sample_utils import LaneRNG

_DEVICE = None
GROUP, BITS = 32, 8


def setUpModule():
    global _DEVICE
    _DEVICE = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    mx.set_default_device(_DEVICE)


def _args(head_dim=32):
    # head_dim >= 32 so the full-attention layers' KV can be mx.quantize'd
    # (group_size 32); full_attention_interval=2 gives a hybrid GDN/KV stack.
    return TextModelArgs(
        model_type="qwen3_5_moe_text", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=head_dim, vocab_size=64, full_attention_interval=2,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=64, linear_value_head_dim=64,
        linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        mtp_num_hidden_layers=1,
        rope_parameters={"type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 0.25},
    )


def _digest(tokens):
    return hashlib.sha256(
        b"".join(int(t).to_bytes(4, "little") for t in tokens)
    ).hexdigest()


class TestQuantizedBatchedSelfMTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(7)
        cls.model = TextModel(_args())
        mx.eval(cls.model.parameters())
        cls.prompts = {
            1: [3, 9, 14, 2, 51, 7],
            2: [1, 40, 5, 23, 8],
            3: [11, 12, 13, 14, 15, 16, 17],
        }

    def _qcache(self):
        tc = make_prompt_cache(self.model)
        maybe_quantize_kv_cache(tc, 0, GROUP, BITS)
        return tc

    def _single_lane(self, prompt, uid, maximum=8):
        toks = []
        for tok, _lp, _fd in self_mtp_generate_step(
            mx.array(prompt, mx.uint32), self.model, num_draft=2,
            max_tokens=maximum, prefill_step_size=4, sampling_temp=0.0,
            sampling_top_p=1.0, sampling_top_k=0, sampling_min_p=0.0,
            accept_rule="residual", persistent_mtp=True,
            lane_rng=LaneRNG(100 + uid), prompt_cache=self._qcache(),
        ):
            toks.append(int(tok))
        return toks

    def _prepare(self, prompt, uid, maximum=8):
        return prepare_self_mtp_lane(
            mx.array(prompt, mx.uint32), self.model, uid=uid,
            max_tokens=maximum, prompt_cache=self._qcache(), mtp_state=None,
            lane_rng=LaneRNG(100 + uid), num_draft=2, sampling_temp=0.0,
            sampling_top_p=1.0, sampling_top_k=0, sampling_min_p=0.0,
            accept_rule="residual", logits_processors=[], prefill_step_size=4,
            share_qsa_indices=False,
        )

    def _run_batch(self, rows, maximum=8):
        prepared = [(uid, *self._prepare(p, uid, maximum)) for uid, p in rows]
        traces = {uid: [int(first.token)] for uid, _d, first in prepared}
        merged_types = None
        batch = attach_self_mtp_lanes(
            self.model, None, [d for _u, d, _f in prepared]
        )
        while batch.lanes:
            if merged_types is None:
                merged_types = [type(c).__name__ for c in batch.caches.target]
            proposal = propose_batched_self_mtp(self.model, batch)
            mx.eval([o.logprobs for row in proposal.outputs for o in row])
            terminal = []
            for lane, outputs in zip(batch.lanes, proposal.outputs):
                traces[lane.uid].extend(int(o.token) for o in outputs)
                terminal.append(lane.ntoks + len(outputs) >= lane.max_tokens)
            commit_batched_self_mtp(
                batch, proposal,
                emitted_counts=[len(r) for r in proposal.outputs],
                terminal=terminal,
            )
            leaving = [i for i, v in enumerate(terminal) if v]
            if leaving:
                batch, _ = detach_self_mtp_lanes(self.model, batch, leaving)
        return traces, merged_types

    def _run_generator(
        self,
        prompt,
        *,
        prompt_cache=None,
        mtp_state=None,
        history=None,
        maximum=6,
    ):
        config = {
            "persistent": True,
            "num_draft": 2,
            "sampling_temp": 0.0,
            "accept_rule": "residual",
            "allow_quantized_kv": True,
        }
        generator = BatchGenerator(
            self.model,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=4,
            self_mtp=config,
            kv_bits=BITS,
            kv_group_size=GROUP,
        )
        try:
            generator.insert(
                [list(prompt)],
                max_tokens=[maximum],
                caches=[prompt_cache],
                all_tokens=[list(history or [])],
                mtp_states=[mtp_state],
                lane_rngs=[LaneRNG(901)],
                self_mtp_configs=[config],
            )
            output = []
            terminal = None
            for _ in range(64):
                _, responses = generator.next()
                output.extend(int(response.token) for response in responses)
                finished = [
                    response
                    for response in responses
                    if response.finish_reason is not None
                ]
                if finished:
                    terminal = finished[-1]
                    break
            self.assertIsNotNone(terminal)
            return output, terminal
        finally:
            generator.close()

    def test_quantized_target_cache_is_built(self):
        types_ = [type(c).__name__ for c in self._qcache()]
        self.assertIn("QuantizedKVCache", types_)
        self.assertIn("ArraysCache", types_)  # GDN layers stay recurrent

    def test_batched_b1_matches_single_lane_exactly(self):
        """L1 blocking oracle on a quantized cache: batched-B1 == single-lane."""
        for uid, prompt in self.prompts.items():
            with self.subTest(uid=uid):
                ref = self._single_lane(prompt, uid)
                traces, merged = self._run_batch([(uid, prompt)])
                self.assertIn("BatchQuantizedKVCache", merged)
                self.assertEqual(
                    _digest(traces[uid]), _digest(ref),
                    msg="quantized batched-B1 must equal single-lane self-MTP",
                )

    def test_multilane_batch_matches_single_lane(self):
        """B=3 in one batch, staggered termination, heterogeneous lengths."""
        refs = {u: self._single_lane(p, u) for u, p in self.prompts.items()}
        traces, merged = self._run_batch(list(self.prompts.items()))
        self.assertIn("BatchQuantizedKVCache", merged)
        for uid in self.prompts:
            with self.subTest(uid=uid):
                self.assertEqual(_digest(traces[uid]), _digest(refs[uid]))

    def test_quantized_batched_lane_apc_restore_is_token_exact(self):
        _, stored = self._run_generator(self.prompts[1], maximum=5)
        self.assertTrue(
            any(
                type(cache).__name__ == "QuantizedKVCache"
                for cache in stored.prompt_cache
            )
        )

        apc = AutomaticPrefixCache()
        sidecar = MTPAPCSidecar(
            stored.mtp_state,
            covered_tokens=len(stored.all_tokens),
        )
        apc.insert_cache(
            "quantized", stored.all_tokens, stored.prompt_cache, sidecar=sidecar
        )
        continued_prompt = list(stored.all_tokens) + [19, 27]
        lookup = apc.lookup("quantized", continued_prompt)
        self.assertEqual(lookup.hit_kind, "mtp_sidecar")
        self.assertIsNot(lookup.sidecar, sidecar)
        self.assertTrue(
            any(type(cache).__name__ == "QuantizedKVCache" for cache in lookup.cache)
        )
        stored_quantized = [
            cache
            for cache in stored.prompt_cache
            if type(cache).__name__ == "QuantizedKVCache"
        ]
        restored_quantized = [
            cache
            for cache in lookup.cache
            if type(cache).__name__ == "QuantizedKVCache"
        ]
        self.assertEqual(len(restored_quantized), len(stored_quantized))
        for source, restored in zip(stored_quantized, restored_quantized):
            self.assertEqual(restored.meta_state, source.meta_state)
            for side in ("keys", "values"):
                for source_array, restored_array in zip(
                    getattr(source, side), getattr(restored, side)
                ):
                    self.assertEqual(restored_array.shape, source_array.shape)
                    self.assertEqual(restored_array.dtype, source_array.dtype)

        cold, _ = self._run_generator(continued_prompt)
        warm, _ = self._run_generator(
            lookup.remaining_tokens,
            prompt_cache=lookup.cache,
            mtp_state=lookup.sidecar.state,
            history=continued_prompt[: lookup.cached_tokens],
        )
        self.assertEqual(warm, cold)

    def test_quantized_hybrid_apc_trim_is_exact_or_fails_closed(self):
        stored_tokens = list(range(1, 13))
        branch = stored_tokens[:9] + [41, 42, 43]
        stored_cache = self._qcache()
        prefill_prompt_cache(
            self.model, stored_tokens[:8], stored_cache, prefill_step_size=4
        )
        prefill_prompt_cache(
            self.model, stored_tokens[8:], stored_cache, prefill_step_size=4
        )
        apc = AutomaticPrefixCache()
        apc.insert_cache("quantized-hybrid", stored_tokens, stored_cache)

        lookup = apc.lookup("quantized-hybrid", branch)
        self.assertTrue(lookup.hit)
        self.assertEqual(lookup.cached_tokens, 8)
        self.assertEqual(lookup.remaining_tokens, branch[8:])
        reused_logits = self.model(
            mx.array(lookup.remaining_tokens)[None], cache=lookup.cache
        )
        fresh_cache = self._qcache()
        prefill_prompt_cache(
            self.model, branch[:8], fresh_cache, prefill_step_size=4
        )
        fresh_logits = self.model(mx.array(branch[8:])[None], cache=fresh_cache)
        mx.eval(reused_logits, fresh_logits)
        self.assertTrue(mx.array_equal(reused_logits, fresh_logits).item())

        previous = os.environ.get("MLX_LM_STATE_CHECKPOINT_MAX")
        os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "0"
        try:
            uncheckpointed = self._qcache()
            prefill_prompt_cache(
                self.model,
                stored_tokens,
                uncheckpointed,
                prefill_step_size=4,
            )
        finally:
            if previous is None:
                os.environ.pop("MLX_LM_STATE_CHECKPOINT_MAX", None)
            else:
                os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = previous
        unsafe = AutomaticPrefixCache()
        unsafe.insert_cache("quantized-hybrid", stored_tokens, uncheckpointed)
        miss = unsafe.lookup("quantized-hybrid", branch)
        self.assertFalse(miss.hit)
        self.assertEqual(miss.miss_reason, "untrimmable_branch")


class TestQuantizedMTPPolicyGate(unittest.TestCase):
    """The opt-in guard: quantized KV is refused for batched self-MTP unless
    allow_quantized_kv is set; windowed stays refused regardless."""

    def _construct(self, self_mtp, *, kv_bits=None, max_kv_size=None):
        return BatchGenerator(
            object(), self_mtp=self_mtp, kv_bits=kv_bits,
            max_kv_size=max_kv_size, kv_group_size=GROUP,
        )

    def test_quantized_refused_without_flag(self):
        with self.assertRaises(ValueError) as ctx:
            self._construct({"persistent": True, "num_draft": 2}, kv_bits=8)
        self.assertIn("allow_quantized_kv", str(ctx.exception))

    def test_quantized_allowed_with_flag(self):
        # Reaches past the config guards; a bare object() model has no
        # attributes needed before the guard block, so no ValueError is raised
        # for the quantized-KV reason.
        try:
            self._construct(
                {"persistent": True, "num_draft": 2, "allow_quantized_kv": True},
                kv_bits=8,
            )
        except ValueError as exc:  # pragma: no cover - guards must not fire
            self.assertNotIn("quantized", str(exc).lower())

    def test_windowed_refused_even_with_quantized_flag(self):
        with self.assertRaises(ValueError) as ctx:
            self._construct(
                {"persistent": True, "num_draft": 2, "allow_quantized_kv": True},
                max_kv_size=2048,
            )
        self.assertIn("windowed", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
