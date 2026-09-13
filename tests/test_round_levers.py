"""Bit-identity and mechanism tests for the 2026-09-08 async round levers.

Every lever is default-off and must leave the emitted token stream, the
head-cache offsets and the APC sidecar exactly as the off path leaves them.
Tiny Qwen4 fixture; runs on CPU or Metal.
"""

import contextlib
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np

from mlx_lm import round_levers as lv
from mlx_lm.generate import prompt_lookup_generate_step
from mlx_lm.hybrid_speculative import HybridStats, self_mtp_generate_step
from mlx_lm.models import qwen4_exp
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen4_exp import QSAKVCache, Qwen4ArraysCache
from mlx_lm.models.qwen4_ple_nvme import (
    FileBackedShardedEmbedding,
    dequant_rows_numpy,
)
from test_batched_self_mtp_qwen4 import _tiny_qwen4_model

LEVERS = ("ple_verify_prefetch", "device_sampling", "hedge_draft", "eager_dispatch")


def _all_off():
    for name in LEVERS:
        lv.set_lever(name, False)
    qwen4_exp.set_qwen4_eager_dispatch_stride(1)
    lv.reset_counters()


@contextlib.contextmanager
def _biased_logits(model, token: int, rule: str):
    """Make ``model.logits`` favour ``token`` by a value-only rule.

    ``rule="always"`` gives 100% draft acceptance; ``rule="sign"`` biases only
    positions whose hidden sums positive, which is a property of the position,
    not of which arm computed it, so both arms see the same stream while the
    accept rate lands in the middle.
    """
    original = model.logits

    def biased(hidden):
        logits = original(hidden)
        onehot = mx.zeros((logits.shape[-1],), logits.dtype).at[token].add(100.0)
        if rule == "always":
            return logits + onehot
        gate = (hidden.sum(axis=-1, keepdims=True) > 0).astype(logits.dtype)
        return logits + gate * onehot

    model.logits = biased
    try:
        yield
    finally:
        model.logits = original


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(0)
        cls.model = _tiny_qwen4_model()
        cls.prompt = mx.random.randint(0, 64, (12,)).astype(mx.uint32)

    def setUp(self):
        _all_off()

    def tearDown(self):
        _all_off()

    def _run(self, seed=7, max_tokens=40, mtp_state_out=None, **kwargs):
        mx.random.seed(seed)
        stats = kwargs.pop("stats", None) or HybridStats()
        tokens = [
            int(t)
            for t, _lp, _fd in self_mtp_generate_step(
                self.prompt,
                self.model,
                max_tokens=max_tokens,
                stats=stats,
                mtp_state_out=mtp_state_out,
                **kwargs,
            )
        ]
        return tokens, stats


class TestHedgeDraft(_Base):
    def _compare(self, **kwargs):
        _all_off()
        off_out = {}
        off, s_off = self._run(mtp_state_out=off_out, **kwargs)
        lv.set_lever("hedge_draft", True)
        lv.reset_counters()
        on_out = {}
        on, s_on = self._run(mtp_state_out=on_out, **kwargs)
        c = lv.counters()
        self.assertEqual(off, on)
        self.assertEqual(s_off.draft_accepted, s_on.draft_accepted)
        self.assertEqual(off_out["covered_tokens"], on_out["covered_tokens"])
        self.assertEqual(off_out["reusable"], on_out["reusable"])
        self.assertEqual(
            [c_.offset for c_ in off_out["state"][0]],
            [c_.offset for c_ in on_out["state"][0]],
        )
        return c

    def test_natural_stream_identical_all_widths(self):
        for k in (1, 2, 3):
            with self.subTest(k=k):
                c = self._compare(num_draft=k, persistent_mtp=True)
                self.assertGreater(c["hedge_built"], 0)
                self.assertEqual(c["hedge_hit"] + c["hedge_miss"], c["hedge_built"])
                self.assertEqual(
                    c["hedge_consumed"] + c["hedge_discarded"], c["hedge_hit"]
                )

    def test_full_acceptance_exercises_hits(self):
        with _biased_logits(self.model, 5, "always"):
            c = self._compare(num_draft=2, persistent_mtp=True, max_tokens=30)
        self.assertGreater(c["hedge_hit"], 0)
        self.assertGreater(c["hedge_consumed"], 0)
        self.assertEqual(c["hedge_miss"], 0)

    def test_mixed_acceptance_exercises_both(self):
        with _biased_logits(self.model, 5, "sign"):
            c = self._compare(num_draft=1, persistent_mtp=True, max_tokens=64)
        self.assertGreater(c["hedge_hit"], 0)
        self.assertGreater(c["hedge_miss"], 0)

    def test_early_close_rewinds_pending_hedge(self):
        # Close the generator right after a full-accept round: the queued
        # chain must be rewound so the sidecar offsets stay exact.
        lv.set_lever("hedge_draft", True)
        lv.reset_counters()
        with _biased_logits(self.model, 5, "always"):
            mx.random.seed(7)
            out = {}
            gen = self_mtp_generate_step(
                self.prompt,
                self.model,
                num_draft=2,
                persistent_mtp=True,
                max_tokens=40,
                mtp_state_out=out,
            )
            for _ in range(6):
                next(gen)
            gen.close()
        c = lv.counters()
        self.assertGreater(c["hedge_hit"], 0)
        self.assertGreaterEqual(c["hedge_discarded"], 1)
        self.assertTrue(out["reusable"])
        # The sidecar must describe the delivered prefix, not merely agree
        # with a target cache that is also ahead of the consumer.  Six output
        # tokens leave the sixth token as the uncached continuation seed.
        self.assertEqual(out["covered_tokens"], int(self.prompt.size) + 6 - 1)
        mtp_cache, _seed = out["state"]
        self.assertEqual(
            max(c_.offset for c_ in mtp_cache), out["covered_tokens"] - 1
        )

    def test_early_close_sidecar_resumes_from_delivered_prefix(self):
        with _biased_logits(self.model, 5, "always"):
            for hedge in (False, True):
                lv.set_lever("hedge_draft", hedge)
                full, _ = self._run(
                    seed=7, num_draft=2, persistent_mtp=True, max_tokens=12
                )
                for stop_after in (2, 6):
                    with self.subTest(hedge=hedge, stop_after=stop_after):
                        lv.set_lever("hedge_draft", hedge)
                        cache = make_prompt_cache(self.model)
                        state_out = {}
                        gen = self_mtp_generate_step(
                            self.prompt,
                            self.model,
                            num_draft=2,
                            persistent_mtp=True,
                            max_tokens=12,
                            prompt_cache=cache,
                            mtp_state_out=state_out,
                        )
                        prefix = []
                        for _ in range(stop_after):
                            token, _lp, _fd = next(gen)
                            prefix.append(int(token))
                        gen.close()

                        self.assertEqual(
                            state_out["covered_tokens"],
                            int(self.prompt.size) + stop_after - 1,
                        )
                        resumed = [
                            int(token)
                            for token, _lp, _fd in self_mtp_generate_step(
                                mx.array([prefix[-1]], mx.uint32),
                                self.model,
                                num_draft=2,
                                persistent_mtp=True,
                                max_tokens=12 - stop_after,
                                prompt_cache=cache,
                                mtp_state=state_out["state"],
                            )
                        ]
                        self.assertEqual(prefix + resumed, full)

    def test_budget_clamped_final_round_does_not_build_a_hedge(self):
        lv.set_lever("hedge_draft", True)
        lv.reset_counters()
        state_out = {}
        self._run(
            num_draft=2,
            persistent_mtp=True,
            max_tokens=3,
            mtp_state_out=state_out,
        )
        counters = lv.counters()
        self.assertEqual(counters["hedge_built"], 0)
        self.assertGreaterEqual(counters["hedge_skipped"], 1)
        self.assertTrue(state_out["reusable"])
        self.assertEqual(
            state_out["covered_tokens"], int(self.prompt.size) + 3 - 1
        )

    def test_hedge_is_silent_when_off_or_non_persistent(self):
        lv.set_lever("hedge_draft", True)
        lv.reset_counters()
        self._run(num_draft=2, persistent_mtp=False)
        self.assertEqual(lv.counters()["hedge_built"], 0)
        _all_off()
        self._run(num_draft=2, persistent_mtp=True)
        self.assertEqual(lv.counters()["hedge_built"], 0)


class TestDeviceSampling(_Base):
    def test_sampled_stream_identical(self):
        for kwargs in (
            dict(sampling_temp=0.8),
            dict(sampling_temp=0.8, sampling_top_p=0.9),
            dict(sampling_temp=0.6, accept_rule="block"),
        ):
            with self.subTest(**kwargs):
                _all_off()
                off, s_off = self._run(
                    seed=41, num_draft=2, persistent_mtp=True, **kwargs
                )
                lv.set_lever("device_sampling", True)
                lv.reset_counters()
                on, s_on = self._run(
                    seed=41, num_draft=2, persistent_mtp=True, **kwargs
                )
                self.assertEqual(off, on)
                self.assertEqual(s_off.draft_accepted, s_on.draft_accepted)
                self.assertGreater(lv.counters()["device_sampled_drafts"], 0)

    def test_greedy_and_processors_untouched(self):
        lv.set_lever("device_sampling", True)
        lv.reset_counters()
        self._run(num_draft=2, persistent_mtp=True)
        self.assertEqual(lv.counters()["device_sampled_drafts"], 0)
        self._run(
            num_draft=2,
            persistent_mtp=True,
            sampling_temp=0.8,
            logits_processors=[lambda toks, logits: logits],
        )
        self.assertEqual(lv.counters()["device_sampled_drafts"], 0)


class TestEagerDispatchToggle(_Base):
    def _forward(self, width):
        text = self.model.language_model.model
        cache = make_prompt_cache(self.model)
        out = text(self.prompt[None, :width], cache)
        mx.eval(out)
        return out

    def test_forward_identical_and_counted(self):
        base = self._forward(3)
        n_layers = len(self.model.language_model.model.layers)
        for stride, expected in ((1, n_layers), (2, (n_layers + 1) // 2)):
            with self.subTest(stride=stride):
                lv.set_lever("eager_dispatch", True)
                qwen4_exp.set_qwen4_eager_dispatch_stride(stride)
                lv.reset_counters()
                on = self._forward(3)
                self.assertTrue(mx.array_equal(base, on))
                self.assertEqual(lv.counters()["eager_async_evals"], expected)
                _all_off()

    def test_setter_roundtrip(self):
        self.assertFalse(lv.set_lever("eager_dispatch", True))
        self.assertTrue(lv.levers()["eager_dispatch"])
        self.assertTrue(lv.set_lever("eager_dispatch", False))
        self.assertEqual(qwen4_exp.set_qwen4_eager_dispatch_stride(4), 1)
        self.assertEqual(qwen4_exp.set_qwen4_eager_dispatch_stride(1), 4)


class TestPeerRegressionGuards(_Base):
    def test_qwen4_text_path_builds_native_hybrid_cache(self):
        caches = make_prompt_cache(self.model)
        self.assertTrue(any(isinstance(c, Qwen4ArraysCache) for c in caches))
        self.assertTrue(any(isinstance(c, QSAKVCache) for c in caches))

    def test_qwen4_prompt_lookup_enters_speculation(self):
        stats = HybridStats()
        prompt = mx.array([1, 2, 3, 4, 1, 2, 3], mx.uint32)
        result = list(
            prompt_lookup_generate_step(
                prompt,
                self.model,
                max_tokens=8,
                num_draft=4,
                ngram_max=3,
                stats=stats,
            )
        )
        self.assertEqual(len(result), 8)
        self.assertGreater(stats.retrieval_proposed, 0)
        self.assertGreater(stats.retrieval_cycles, 0)

class TestPleVerifyPrefetch(_Base):
    def test_resident_table_is_a_noop_and_stream_identical(self):
        off, _ = self._run(num_draft=2, persistent_mtp=True)
        lv.set_lever("ple_verify_prefetch", True)
        lv.reset_counters()
        self.assertEqual(self.model.ple_prefetch_verify([1, 2], [3]), 0)
        on, _ = self._run(num_draft=2, persistent_mtp=True)
        self.assertEqual(off, on)
        self.assertEqual(lv.counters()["ple_prefetch_submitted"], 0)

    def test_staged_rows_are_bit_identical(self):
        dims, vocab = 32, 64
        row_bytes = dims // 2 + 2 * (dims // 32) * 2
        rng = np.random.default_rng(3)
        payload = rng.integers(0, 256, size=(vocab, row_bytes), dtype=np.uint8)
        for lru_mb in ("0", "1"):
            with self.subTest(lru_mb=lru_mb), TemporaryDirectory() as tmp:
                path = Path(tmp) / "ple_rows.bin"
                path.write_bytes(payload.tobytes())
                os.environ["MLX_QWEN4_PLE_NVME_LRU_MB"] = lru_mb
                try:
                    table = FileBackedShardedEmbedding(
                        str(path), vocab_size=vocab, dims=dims, num_shards=1
                    )
                finally:
                    os.environ.pop("MLX_QWEN4_PLE_NVME_LRU_MB", None)
                try:
                    ids = np.array([[3, 7, 7, 9, 12]], dtype=np.int64)
                    expected = dequant_rows_numpy(payload[ids.reshape(-1)], dims)[None]
                    lv.reset_counters()
                    before = np.asarray(table.lookup_numpy(ids).view(mx.uint16))
                    np.testing.assert_array_equal(before, expected)
                    self.assertEqual(lv.counters()["ple_dq_hits"], 0)
                    # Stage two of the four unique rows, then look up again.
                    self.assertEqual(table.prefetch_dequant_rows(np.array([7, 9])), 2)
                    deadline = time.time() + 5.0
                    while len(table._dq_cache) < 2 and time.time() < deadline:
                        time.sleep(0.01)
                    self.assertEqual(len(table._dq_cache), 2)
                    after = np.asarray(table.lookup_numpy(ids).view(mx.uint16))
                    np.testing.assert_array_equal(after, expected)
                    c = lv.counters()
                    self.assertEqual(c["ple_dq_hits"], 2)
                    self.assertEqual(c["ple_dq_misses"], 2)
                    self.assertEqual(c["ple_prefetch_rows"], 2)
                    self.assertEqual(len(table._dq_cache), 0)  # consumed
                    # Stage every row: the whole lookup comes from the stage.
                    table.prefetch_dequant_rows(ids)
                    deadline = time.time() + 5.0
                    while len(table._dq_cache) < 4 and time.time() < deadline:
                        time.sleep(0.01)
                    lv.reset_counters()
                    again = np.asarray(table.lookup_numpy(ids).view(mx.uint16))
                    np.testing.assert_array_equal(again, expected)
                    self.assertEqual(lv.counters()["ple_dq_hits"], 4)
                    self.assertEqual(lv.counters()["ple_dq_misses"], 0)
                finally:
                    close = getattr(table, "close", None)
                    if close is not None:
                        close()


if __name__ == "__main__":
    unittest.main()
