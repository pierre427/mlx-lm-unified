import unittest

import mlx.core as mx

from mlx_lm.generate import (
    draft_tokens_for_budget,
    generate_step,
    speculative_generate_step,
)
from mlx_lm.hybrid_speculative import (
    HybridStats,
    _start_speculation_or_cleanup,
    _stop_all_speculation,
    hybrid_generate_step,
    self_mtp_generate_step,
)
from mlx_lm.models import cache, llama, qwen3_next

QWEN3_NEXT_ARGS = {
    "model_type": "qwen3_next",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "vocab_size": 1000,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 3,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "shared_expert_intermediate_size": 128,
    "mlp_only_layers": [0],
    "moe_intermediate_size": 128,
    "rms_norm_eps": 1e-5,
    "head_dim": 64,
    "rope_theta": 1000.0,
    "partial_rotary_factor": 0.5,
    "max_position_embeddings": 1000,
}

LLAMA_ARGS = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-5,
    "vocab_size": 1000,
}


def make_hybrid(seed=0):
    mx.random.seed(seed)
    return qwen3_next.Model(qwen3_next.ModelArgs.from_dict(QWEN3_NEXT_ARGS))


class TestSpeculativeRollback(unittest.TestCase):

    def test_hybrid_setup_failure_cleans_every_cache(self):
        class LifecycleCache:
            def __init__(self, supports_trim=True, fail_start=False):
                self.supports_trim = supports_trim
                self.fail_start = fail_start
                self.speculating = False
                self.stop_calls = 0

            def start_speculation(self):
                self.speculating = True
                if self.fail_start:
                    raise RuntimeError("start failed")

            def stop_speculation(self):
                self.speculating = False
                self.stop_calls += 1

            def is_trimmable(self):
                return self.speculating and self.supports_trim

        target = LifecycleCache()
        unsupported_draft = LifecycleCache(supports_trim=False)
        with self.assertRaisesRegex(ValueError, "needs rollback"):
            _start_speculation_or_cleanup(
                [target, unsupported_draft],
                [target, unsupported_draft],
                "needs rollback",
            )
        self.assertTrue(all(not c.speculating for c in (target, unsupported_draft)))
        self.assertTrue(all(c.stop_calls == 1 for c in (target, unsupported_draft)))

        first = LifecycleCache()
        failing = LifecycleCache(fail_start=True)
        with self.assertRaisesRegex(RuntimeError, "start failed"):
            _start_speculation_or_cleanup(
                [first, failing], [first, failing], "needs rollback"
            )
        self.assertTrue(all(not c.speculating for c in (first, failing)))
        self.assertTrue(all(c.stop_calls == 1 for c in (first, failing)))

    def test_rollback_is_exact(self):
        # Tail independence: feed two verify chunks sharing the first m tokens
        # but with different tails, roll both back to m — every cache entry and
        # the next-token logits must be IDENTICAL (no numeric tolerance: same
        # starting state, same shapes, deterministic kernels). Any off-by-one
        # in the recorded rollback leaks the divergent tail.
        model = make_hybrid()
        prompt = mx.random.randint(0, 1000, (32,), dtype=mx.uint32)
        T = 8
        xs = mx.random.randint(0, 1000, (T,), dtype=mx.uint32)
        zs = mx.random.randint(0, 1000, (T,), dtype=mx.uint32)
        probe = mx.array([7], mx.uint32)

        def run(chunk, m):
            c = model.make_cache()
            model(prompt[None], cache=c)
            mx.eval([x.state for x in c])
            for x in c:
                x.start_speculation()
            model(chunk[None], cache=c)
            self.assertEqual(cache.trim_prompt_cache(c, chunk.size - m), chunk.size - m)
            logits = model(probe[None], cache=c)
            mx.eval(logits, [x.state for x in c])
            return c, logits

        for m in (1, 3, T - 1):
            cA, lA = run(xs, m)
            chunk_b = mx.concatenate([xs[:m], zs[: T - m]])
            cB, lB = run(chunk_b, m)
            self.assertTrue(mx.array_equal(lA, lB))
            for a, b in zip(cA, cB):
                if isinstance(a, cache.ArraysCache):
                    for ea, eb in zip(a.cache, b.cache):
                        self.assertTrue(mx.array_equal(ea, eb))
                else:
                    self.assertEqual(a.offset, b.offset)
                    self.assertTrue(
                        mx.array_equal(
                            a.keys[..., : a.offset, :], b.keys[..., : b.offset, :]
                        )
                    )
                    self.assertTrue(
                        mx.array_equal(
                            a.values[..., : a.offset, :], b.values[..., : b.offset, :]
                        )
                    )

    def test_speculative_matches_vanilla(self):
        # With an unrelated (random) draft model the acceptance rate is ~0, so
        # every round exercises the rollback path; greedy speculative decoding
        # must still reproduce the vanilla greedy generation exactly.
        model = make_hybrid()
        mx.random.seed(1)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)
        n = 24

        vanilla = [int(tok) for tok, _ in generate_step(prompt, model, max_tokens=n)]
        spec = [
            int(tok)
            for tok, _, _ in speculative_generate_step(
                prompt, model, draft, num_draft_tokens=3, max_tokens=n
            )
        ]
        self.assertEqual(vanilla, spec)

    def test_reasoning_budget_survives_speculative_rejections(self):
        # A stateful reasoning-budget processor consumes the draft tokens fed
        # during verify; forced rejections then rewind prev_tokens. Its
        # decisions (and hence the output) must match sequential generation
        # with an identical processor.
        from mlx_lm.sample_utils import make_reasoning_budget

        model = make_hybrid()
        mx.random.seed(5)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))  # ~all rejected
        prompt = mx.random.randint(10, 1000, (16,), dtype=mx.uint32)
        n = 32

        def procs():
            return [
                make_reasoning_budget(
                    think_close=5, max_think_tokens=12, check_every=10**6
                )
            ]

        vanilla = [
            int(tok)
            for tok, _ in generate_step(
                prompt, model, max_tokens=n, logits_processors=procs()
            )
        ]
        spec = [
            int(tok)
            for tok, _, _ in speculative_generate_step(
                prompt,
                model,
                draft,
                num_draft_tokens=3,
                max_tokens=n,
                logits_processors=procs(),
            )
        ]
        self.assertEqual(vanilla, spec)
        self.assertIn(5, vanilla)  # the budget genuinely tripped

    def test_cache_reusable_after_speculation(self):
        # A prompt cache used for speculative decoding must come back clean:
        # not trimmable, no dangling rollback, and usable for plain decoding
        # that matches a never-speculated run.
        model = make_hybrid()
        mx.random.seed(2)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)

        c = cache.make_prompt_cache(model) + cache.make_prompt_cache(draft)
        spec = [
            int(tok)
            for tok, _, _ in speculative_generate_step(
                prompt, model, draft, num_draft_tokens=3, max_tokens=8, prompt_cache=c
            )
        ]
        model_cache = c[: len(model.layers)]
        self.assertFalse(cache.can_trim_prompt_cache(model_cache))
        for x in model_cache:
            if isinstance(x, cache.ArraysCache):
                self.assertFalse(x.speculating)
                self.assertEqual(len(x._rollbacks), 0)

        # continue decoding on the same cache with plain steps
        cont = [
            int(tok)
            for tok, _ in generate_step(
                mx.array(spec[-1:], mx.uint32),
                model,
                max_tokens=8,
                prompt_cache=model_cache,
            )
        ]

        # reference: one uninterrupted vanilla run over the same tokens
        ref = [
            int(tok)
            for tok, _ in generate_step(prompt, model, max_tokens=8 + len(spec))
        ]
        self.assertEqual(ref, spec + cont)

    def test_early_generator_close(self):
        # Closing the speculative generator mid-stream must roll the cache back
        # cleanly (the finally path) without raising.
        model = make_hybrid()
        mx.random.seed(3)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)

        gen = speculative_generate_step(
            prompt, model, draft, num_draft_tokens=3, max_tokens=64
        )
        took = [next(gen) for _ in range(3)]
        gen.close()  # must not raise (double-rewind / consumed-rollback bugs)
        self.assertEqual(len(took), 3)

    def test_unsupported_hybrid_still_raises(self):
        # An ArraysCache whose layers never record a rollback must not be
        # silently trimmable outside of speculation.
        c = cache.ArraysCache(size=2)
        self.assertFalse(c.is_trimmable())
        c.start_speculation()
        self.assertTrue(c.is_trimmable())
        with self.assertRaises(RuntimeError):
            c.trim(1)
        c.stop_speculation()
        self.assertFalse(c.is_trimmable())

    def test_hybrid_draft_model(self):
        # Target AND draft are hybrid GDN models (the setup reported in #1446:
        # Qwen3.6 target + Qwen3.5 draft). The draft cache is rewound exactly
        # too — its rollback spans several T=1 records per round — and the
        # output must still equal vanilla greedy.
        model = make_hybrid(seed=0)
        draft = make_hybrid(seed=7)  # different weights, same arch
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)
        n = 24

        vanilla = [int(tok) for tok, _ in generate_step(prompt, model, max_tokens=n)]
        c = cache.make_prompt_cache(model) + cache.make_prompt_cache(draft)
        spec = [
            int(tok)
            for tok, _, _ in speculative_generate_step(
                prompt,
                model,
                draft,
                num_draft_tokens=3,
                max_tokens=n,
                prompt_cache=c,
            )
        ]
        self.assertEqual(vanilla, spec)
        for x in c:
            if isinstance(x, cache.ArraysCache):
                self.assertFalse(x.speculating)
                self.assertEqual(len(x._rollbacks), 0)

    def test_draft_exception_not_masked(self):
        # An exception raised mid-round (e.g. inside the draft model) must
        # surface as itself: the finally-block cache rewind must not replace it
        # with a rollback RuntimeError.
        model = make_hybrid()
        mx.random.seed(4)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)

        boom = ValueError("draft exploded")
        calls = {"n": 0}

        class Flaky:
            def __call__(self, *args, **kwargs):
                calls["n"] += 1
                if calls["n"] > 4:  # fail during the second round's draft steps
                    raise boom
                return draft(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(draft, name)

        gen = speculative_generate_step(
            prompt, model, Flaky(), num_draft_tokens=3, max_tokens=32
        )
        with self.assertRaises(ValueError) as ctx:
            for _ in gen:
                pass
        self.assertIs(ctx.exception, boom)

    def test_trim_beyond_window_raises(self):
        # Trimming more than the recorded rollback window must fail loudly and
        # consistently rather than silently clamping (which would desync the
        # ArraysCache layers from the KVCache layers).
        model = make_hybrid()
        prompt = mx.random.randint(0, 1000, (32,), dtype=mx.uint32)
        c = model.make_cache()
        model(prompt[None], cache=c)
        mx.eval([x.state for x in c])
        for x in c:
            x.start_speculation()
        chunk = mx.random.randint(0, 1000, (4,), dtype=mx.uint32)
        model(chunk[None], cache=c)
        arrays = next(x for x in c if isinstance(x, cache.ArraysCache))
        with self.assertRaises(RuntimeError):
            arrays.trim(10)  # only 4 tokens of rollback recorded


class _FakeSpecCache:
    """Minimal trimmable prompt-cache stand-in with lifecycle accounting."""

    def __init__(self, fail_stop=False):
        self.offset = 0
        self.speculating = False
        self.stop_calls = 0
        self.fail_stop = fail_stop

    def start_speculation(self):
        self.speculating = True

    def stop_speculation(self):
        self.stop_calls += 1
        self.speculating = False
        if self.fail_stop:
            raise RuntimeError("stop_speculation failed")

    def is_trimmable(self):
        return self.speculating

    def record_rollback(self, *args, **kwargs):
        pass

    def trim(self, n):
        self.offset -= n
        return n

    @property
    def state(self):
        return mx.zeros((1,))


class _ZeroLogitModel:
    """Fake target model: always predicts token 0, tracks forward calls."""

    VOCAB = 32
    supports_speculative_rollback = True

    def __init__(self, caches=None):
        self.calls = 0
        self.input_lengths = []
        self._caches = caches

    def __call__(self, x, cache=None):
        self.calls += 1
        self.input_lengths.append(x.shape[-1])
        for c in cache:
            c.offset += x.shape[-1]
        return mx.zeros((x.shape[0], x.shape[1], self.VOCAB))

    def make_cache(self):
        return self._caches if self._caches is not None else [_FakeSpecCache()]


class _ZeroMTPModel:
    """Fake MTP model: trunk + head always predict token 0."""

    VOCAB = 16
    HIDDEN = 4

    def __init__(self, caches=None):
        self.mtp = object()
        self.trunk_calls = 0
        self.mtp_calls = 0
        self._caches = caches if caches is not None else [_FakeSpecCache()]

    def model(self, x, cache=None):
        self.trunk_calls += 1
        for c in cache:
            c.offset += x.shape[-1]
        return mx.zeros((x.shape[0], x.shape[1], self.HIDDEN))

    def logits(self, h):
        return mx.zeros(h.shape[:-1] + (self.VOCAB,))

    def make_cache(self):
        return self._caches

    def make_mtp_cache(self):
        return []

    def mtp_step(self, h, tok, mtp_cache):
        self.mtp_calls += 1
        return mx.zeros((1, 1, self.VOCAB)), h


class TestGeneratorLifecycleSemantics(unittest.TestCase):
    """max_tokens=0, yield-boundary telemetry, and fail-safe cleanup."""

    def test_stop_all_speculation_runs_every_hook_and_surfaces_first_error(self):
        first = _FakeSpecCache(fail_stop=True)
        second = _FakeSpecCache(fail_stop=True)
        third = _FakeSpecCache()
        with self.assertRaisesRegex(RuntimeError, "stop_speculation failed"):
            _stop_all_speculation([first, second, third])
        self.assertEqual([c.stop_calls for c in (first, second, third)], [1, 1, 1])
        self.assertTrue(all(not c.speculating for c in (first, second, third)))

    def test_hybrid_max_tokens_zero_yields_nothing_and_does_no_work(self):
        model = _ZeroLogitModel()
        gen = hybrid_generate_step(
            mx.array([0] * 8, mx.uint32), model, max_tokens=0
        )
        self.assertEqual(list(gen), [])
        self.assertEqual(model.calls, 0)

    def test_draft_budget_reserves_target_bonus_slot(self):
        self.assertEqual(draft_tokens_for_budget(4, 0), 0)
        self.assertEqual(draft_tokens_for_budget(4, 1), 0)
        self.assertEqual(draft_tokens_for_budget(4, 2), 1)
        self.assertEqual(draft_tokens_for_budget(4, 5), 4)
        self.assertEqual(draft_tokens_for_budget(4, -1), 4)

    def test_speculative_zero_budget_does_no_model_work(self):
        target = _ZeroLogitModel()
        draft = _ZeroLogitModel()
        gen = speculative_generate_step(
            mx.array([1, 2, 3], mx.uint32), target, draft, max_tokens=0
        )
        self.assertEqual(list(gen), [])
        self.assertEqual(target.calls, 0)
        self.assertEqual(draft.calls, 0)

    def test_speculative_one_token_budget_skips_drafting(self):
        target = _ZeroLogitModel()
        draft = _ZeroLogitModel()
        result = list(speculative_generate_step(
            mx.array([1, 2, 3], mx.uint32),
            target,
            draft,
            num_draft_tokens=4,
            max_tokens=1,
        ))
        self.assertEqual(len(result), 1)
        # The draft sees the prompt prefill only; no autoregressive proposal.
        self.assertEqual(draft.input_lengths, [2])
        self.assertEqual(target.input_lengths, [2, 1])

    def test_self_mtp_max_tokens_zero_yields_nothing_and_does_no_work(self):
        model = _ZeroMTPModel()
        gen = self_mtp_generate_step(
            mx.array([1, 2, 3], mx.uint32), model, max_tokens=0
        )
        self.assertEqual(list(gen), [])
        self.assertEqual(model.trunk_calls, 0)

    def test_self_mtp_one_token_tail_uses_plain_target_step(self):
        model = _ZeroMTPModel()
        stats = HybridStats()
        result = list(self_mtp_generate_step(
            mx.array([1, 2, 3], mx.uint32),
            model,
            num_draft=4,
            max_tokens=2,
            stats=stats,
        ))
        self.assertEqual(len(result), 2)
        self.assertEqual([from_draft for _, _, from_draft in result], [False, False])
        self.assertEqual(model.mtp_calls, 0)
        self.assertEqual(stats.draft_proposed, 0)
        self.assertEqual(stats.plain_tokens, 2)

    def test_hybrid_early_close_counts_only_delivered_tokens(self):
        # The consumer closes after the FIRST delivered token of an accepted
        # retrieval batch (e.g. EOS): telemetry must count exactly one
        # accepted token and no bonus token. The suffix [1,2,3] recurs, so
        # retrieval proposes the zeros after its first occurrence — which the
        # zero-predicting model accepts.
        model = _ZeroLogitModel()
        stats = HybridStats()
        gen = hybrid_generate_step(
            mx.array([1, 2, 3] + [0] * 6 + [1, 2, 3], mx.uint32),
            model,
            max_tokens=16,
            min_match=2,
            max_span=8,
            stats=stats,
        )
        tok, _logprobs, from_draft = next(gen)
        self.assertEqual(tok, 0)
        self.assertTrue(from_draft)
        gen.close()

        self.assertGreaterEqual(stats.retrieval_proposed, 2)
        self.assertEqual(stats.retrieval_accepted, 1)
        self.assertEqual(stats.bonus_tokens, 0)
        self.assertEqual(stats.plain_tokens, 0)
        self.assertEqual(stats.total_emitted, 1)

    def test_self_mtp_early_close_counts_only_delivered_tokens(self):
        # Close immediately after the first (plain) token: it was delivered,
        # so it must be counted — and nothing else may be.
        model = _ZeroMTPModel()
        stats = HybridStats()
        gen = self_mtp_generate_step(
            mx.array([1, 2, 3], mx.uint32), model, max_tokens=8, stats=stats
        )
        next(gen)
        gen.close()
        self.assertEqual(stats.plain_tokens, 1)
        self.assertEqual(stats.total_emitted, 1)

        # Close after one accepted draft token: the cycle's bonus token was
        # never delivered and must not be counted.
        model = _ZeroMTPModel()
        stats = HybridStats()
        gen = self_mtp_generate_step(
            mx.array([1, 2, 3], mx.uint32), model, max_tokens=8, stats=stats
        )
        next(gen)  # first plain token
        tok, _lp, from_draft = next(gen)  # first accepted draft token
        self.assertTrue(from_draft)
        gen.close()
        self.assertEqual(stats.plain_tokens, 1)
        self.assertEqual(stats.draft_proposed, 1)
        self.assertEqual(stats.draft_accepted, 1)
        self.assertEqual(stats.bonus_tokens, 0)

    def test_hybrid_close_cleans_every_cache_even_if_one_stop_raises(self):
        failing = _FakeSpecCache(fail_stop=True)
        healthy = _FakeSpecCache()
        model = _ZeroLogitModel(caches=[failing, healthy])
        gen = hybrid_generate_step(
            mx.array([0] * 8, mx.uint32), model, max_tokens=16
        )
        next(gen)
        with self.assertRaisesRegex(RuntimeError, "stop_speculation failed"):
            gen.close()
        self.assertEqual(failing.stop_calls, 1)
        self.assertEqual(healthy.stop_calls, 1)
        self.assertFalse(healthy.speculating)

    def test_self_mtp_close_cleans_every_cache_even_if_one_stop_raises(self):
        failing = _FakeSpecCache(fail_stop=True)
        healthy = _FakeSpecCache()
        model = _ZeroMTPModel(caches=[failing, healthy])
        gen = self_mtp_generate_step(
            mx.array([1, 2, 3], mx.uint32), model, max_tokens=8
        )
        next(gen)
        with self.assertRaisesRegex(RuntimeError, "stop_speculation failed"):
            gen.close()
        self.assertEqual(failing.stop_calls, 1)
        self.assertEqual(healthy.stop_calls, 1)
        self.assertFalse(healthy.speculating)


if __name__ == "__main__":
    unittest.main()
