# Copyright © 2024 Apple Inc.

"""Tests for draft-free prompt-lookup (PLD) speculative decoding.

Covers the proposer backends, the cache-lifecycle helpers, and end-to-end
correctness — including the two cache-reconciliation regressions found in review:
cache REUSE (a non-empty incoming prompt_cache must not be trimmed) and CacheList
models (the reconcile must not assume a flat ``.offset``).
"""
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import (
    _pld_offset,
    _pld_rewind,
    _pld_snapshot,
    _pld_start_speculation,
    generate_step,
    prompt_lookup_generate_step,
)
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, make_prompt_cache
from mlx_lm.prompt_lookup import (
    NgramProposer,
    SuffixAutomaton,
    SuffixAutomatonProposer,
    make_proposer,
    plan_proposal_around_verify_cliff,
    snap_proposal_around_verify_cliff,
)
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)


class TestProposers(unittest.TestCase):
    def test_ngram_lookback_bound(self):
        # A match beyond max_lookback is not scanned (bounds the per-cycle
        # cost on novel text)...
        seq = [1, 2, 7, 8] + [3] * 64 + [1, 2]
        bounded = NgramProposer(ngram_max=2, ngram_min=2, max_lookback=16)
        self.assertEqual(bounded.propose(seq, 2, 0), [])
        # ...while the default window still finds it, and the same proposer
        # still finds an in-window match.
        full = NgramProposer(ngram_max=2, ngram_min=2)
        self.assertEqual(full.propose(seq, 2, 0), [7, 8])
        near = [9, 9, 1, 2, 7, 8, 1, 2]
        self.assertEqual(bounded.propose(near, 2, 0), [7, 8])

    def test_ngram_finds_earlier_continuation(self):
        # "a b c ... a b" -> proposes the continuation after the earlier "a b".
        seq = [1, 2, 3, 9, 9, 1, 2]
        p = NgramProposer(ngram_max=2, ngram_min=1)
        self.assertEqual(p.propose(seq, max_span=3, prompt_len=len(seq)), [3, 9, 9])

    def test_ngram_no_match(self):
        p = NgramProposer(ngram_max=3, ngram_min=2)
        self.assertEqual(p.propose([1, 2, 3, 4], max_span=4, prompt_len=4), [])

    def test_suffix_automaton_longest_repeat(self):
        sam = SuffixAutomaton([5, 6, 7, 8, 5, 6])
        mlen, nxt = sam.longest_suffix_match(max_len=16)
        self.assertEqual(mlen, 2)          # "5 6" repeats
        self.assertEqual([5, 6, 7, 8, 5, 6][nxt], 7)  # continuation is "7"

    def test_make_proposer_returns_empty(self):
        # The caller is the single seeding authority; make_proposer must not seed.
        p = make_proposer("suffix_automaton")
        self.assertIsInstance(p, SuffixAutomatonProposer)
        self.assertEqual(len(p.sam), 0)
        with self.assertRaises(ValueError):
            make_proposer("nope")

    def test_verify_cliff_span_snapping(self):
        proposal = list(range(32))
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:7]), proposal[:7]
        )
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:8]), proposal[:7]
        )
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:14]), proposal[:7]
        )
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:15]), proposal[:15]
        )
        # Two pending rows need a six-token proposal to stay at eight total.
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:7], pending_rows=2),
            proposal[:6],
        )
        with self.assertRaises(ValueError):
            snap_proposal_around_verify_cliff(proposal, pending_rows=0)
        self.assertEqual(plan_proposal_around_verify_cliff(8, 20), 15)
        self.assertEqual(plan_proposal_around_verify_cliff(10, 12), 7)
        self.assertEqual(plan_proposal_around_verify_cliff(16, 20), 16)


class TestCacheHelpers(unittest.TestCase):
    def test_pld_offset_flat(self):
        c = KVCache()
        c.offset = 11
        self.assertEqual(_pld_offset(c), 11)

    def test_pld_offset_cachelist(self):
        # CacheList has no .offset of its own; the helper must descend.
        cl = CacheList(KVCache(), KVCache())
        cl.caches[0].offset = 7
        cl.caches[1].offset = 7
        self.assertFalse(hasattr(cl, "offset"))
        self.assertEqual(_pld_offset(cl), 7)

    def test_snapshot_rejects_unsupported_cache(self):
        # ArraysCache cannot be snapshotted until speculation recording is on.
        with self.assertRaises(NotImplementedError):
            _pld_snapshot([ArraysCache(size=2)])

    def test_arrays_cache_snapshot_rewinds_recorded_delta(self):
        c = ArraysCache(size=1)
        c.cache = [mx.array([0])]
        c.start_speculation()
        c.record_rollback(2, lambda m: [mx.array([m])], list(c.cache))
        snap = _pld_snapshot([c])
        c.record_rollback(3, lambda m: [mx.array([2 + m])], [mx.array([2])])
        _pld_rewind([c], snap)
        self.assertEqual(sum(r[0] for r in c._rollbacks), 2)
        self.assertEqual(c.cache[0].tolist(), [2])


class _LifecycleCache:
    def __init__(self, fail_start=False):
        self.offset = 0
        self.speculating = False
        self.fail_start = fail_start
        self.start_offsets = []
        self.stop_calls = 0

    @property
    def state(self):
        return []

    def start_speculation(self, rollback_window=None):
        self.start_offsets.append(self.offset)
        self.speculating = True
        if self.fail_start:
            raise RuntimeError("start failed")

    def stop_speculation(self):
        self.stop_calls += 1
        self.speculating = False

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= n
        return n


class _LifecycleModel:
    def __init__(self, fail_calls=()):
        self.calls = []
        self.input_lengths = []
        self.fail_calls = set(fail_calls)

    def __call__(self, x, cache=None):
        call_number = len(self.calls) + 1
        self.calls.append(cache[0].speculating)
        self.input_lengths.append(x.shape[-1])
        if call_number in self.fail_calls:
            raise RuntimeError(f"model call {call_number} failed")
        cache[0].offset += x.shape[-1]
        return mx.zeros((x.shape[0], x.shape[1], 8))


class TestPromptLookupLifecycle(unittest.TestCase):
    def test_early_close_counts_only_yielded_tokens(self):
        class AcceptAllProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        cache = _LifecycleCache()
        stats = HybridStats()
        generator = prompt_lookup_generate_step(
            mx.array([1]),
            _LifecycleModel(),
            prompt_cache=[cache],
            max_tokens=16,
            num_draft=8,
            backend=AcceptAllProposer(),
            stats=stats,
        )
        token, _logprobs, from_draft = next(generator)
        self.assertEqual(token, 0)
        self.assertTrue(from_draft)
        generator.close()

        self.assertEqual(stats.retrieval_proposed, 8)
        self.assertEqual(stats.retrieval_accepted, 1)
        self.assertEqual(stats.bonus_tokens, 0)
        self.assertEqual(stats.total_emitted, 1)
        self.assertEqual(cache.offset, 2)  # one prompt + one delivered token

    def test_cliff_aware_span_is_opt_in_and_tracks_trimming(self):
        class FixedProposer:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return [0] * max_span

        from mlx_lm.prompt_lookup import HybridStats

        default_cache = _LifecycleCache()
        default_model = _LifecycleModel()
        default_gen = prompt_lookup_generate_step(
            mx.array([1]), default_model, prompt_cache=[default_cache],
            max_tokens=16, num_draft=10, backend=FixedProposer(),
        )
        next(default_gen)
        default_gen.close()
        self.assertEqual(default_model.input_lengths[0], 11)

        stats = HybridStats()
        snapped_cache = _LifecycleCache()
        snapped_model = _LifecycleModel()
        snapped_gen = prompt_lookup_generate_step(
            mx.array([1]), snapped_model, prompt_cache=[snapped_cache],
            max_tokens=16, num_draft=10, backend=FixedProposer(),
            cliff_aware_span=True, stats=stats,
        )
        next(snapped_gen)
        snapped_gen.close()
        self.assertEqual(snapped_model.input_lengths[0], 16)
        self.assertEqual(stats.span_extend_cycles, 1)
        self.assertEqual(stats.span_extend_tokens, 5)
        self.assertEqual(stats.verify_span_hist, {16: 1})

        class ShortProposer(FixedProposer):
            def propose(self, seq, max_span, prompt_len):
                return [0] * min(max_span, 10)

        short_stats = HybridStats()
        short_model = _LifecycleModel()
        short_gen = prompt_lookup_generate_step(
            mx.array([1]), short_model, prompt_cache=[_LifecycleCache()],
            max_tokens=16, num_draft=10, backend=ShortProposer(),
            cliff_aware_span=True, stats=short_stats,
        )
        next(short_gen)
        short_gen.close()
        self.assertEqual(short_model.input_lengths[0], 8)
        self.assertEqual(short_stats.span_snap_cycles, 1)
        self.assertEqual(short_stats.span_snap_tokens, 3)
        self.assertEqual(short_stats.verify_span_hist, {8: 1})

    def test_speculation_starts_after_prompt_prefill(self):
        cache = _LifecycleCache()
        model = _LifecycleModel()
        gen = prompt_lookup_generate_step(
            mx.array([1, 2, 3]), model, prompt_cache=[cache], max_tokens=2
        )
        next(gen)
        gen.close()
        self.assertEqual(model.calls[0], False)
        self.assertTrue(all(model.calls[i] for i in range(1, len(model.calls))))
        self.assertEqual(cache.start_offsets, [2])
        self.assertFalse(cache.speculating)

    def test_validation_and_proposer_errors_do_not_start_speculation(self):
        for prompt, backend in ((mx.array([]), "ngram"), (mx.array([1]), "bad")):
            cache = _LifecycleCache()
            with self.assertRaises(ValueError):
                list(
                    prompt_lookup_generate_step(
                        prompt, _LifecycleModel(), prompt_cache=[cache],
                        backend=backend,
                    )
                )
            self.assertEqual(cache.start_offsets, [])
            self.assertFalse(cache.speculating)

    def test_prefill_error_does_not_start_speculation(self):
        cache = _LifecycleCache()
        with self.assertRaises(RuntimeError):
            list(
                prompt_lookup_generate_step(
                    mx.array([1, 2]), _LifecycleModel(fail_calls={1}),
                    prompt_cache=[cache],
                )
            )
        self.assertEqual(cache.start_offsets, [])
        self.assertFalse(cache.speculating)

    def test_loop_and_reconciliation_errors_stop_speculation(self):
        loop_cache = _LifecycleCache()
        with self.assertRaises(RuntimeError):
            list(
                prompt_lookup_generate_step(
                    mx.array([1]), _LifecycleModel(fail_calls={1, 2}),
                    prompt_cache=[loop_cache],
                )
            )
        self.assertFalse(loop_cache.speculating)
        self.assertGreater(loop_cache.stop_calls, 0)

        reconcile_cache = _LifecycleCache()
        gen = prompt_lookup_generate_step(
            mx.array([1]), _LifecycleModel(fail_calls={2}),
            prompt_cache=[reconcile_cache], max_tokens=2,
        )
        next(gen)
        with self.assertRaises(RuntimeError):
            gen.close()
        self.assertFalse(reconcile_cache.speculating)
        self.assertGreater(reconcile_cache.stop_calls, 0)

    def test_partial_start_failure_stops_all_caches(self):
        caches = [_LifecycleCache(), _LifecycleCache(fail_start=True)]
        with self.assertRaises(RuntimeError):
            _pld_start_speculation(caches, 8)
        self.assertTrue(all(not c.speculating for c in caches))
        self.assertTrue(all(c.stop_calls > 0 for c in caches))


class _TinyCacheListModel(nn.Module):
    """Minimal model whose per-layer cache is ``CacheList(KVCache, KVCache)`` —
    mirroring deepseek_v32 / longcat_flash, which put two KV caches per layer.
    Lets the CacheList path (snapshot/rewind + the finally reconcile) be tested
    without downloading a large real model."""

    def __init__(self, vocab=48, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, vocab, bias=False)

    def make_cache(self):
        return [CacheList(KVCache(), KVCache())]

    def __call__(self, x, cache=None):
        h = self.embed(x)
        if cache is not None:
            B, S, D = h.shape
            kv = self.q(h).reshape(B, 1, S, D)  # [B, heads=1, S, D]
            for sub in cache[0].caches:
                sub.update_and_fetch(kv, kv)     # advance both sub-caches
        return self.out(h)


class TestCacheListModel(unittest.TestCase):
    def test_pld_runs_on_cachelist_model(self):
        # Regression for the finally-reconcile CacheList bug: prompt_cache[0] is a
        # CacheList (no flat .offset). PLD must run and leave the cache exact.
        mx.random.seed(0)
        model = _TinyCacheListModel()
        mx.eval(model.parameters())
        prompt = mx.array([3, 7, 1, 3, 7])  # repeated suffix -> retrieval fires
        cache = make_prompt_cache(model)
        self.assertIsInstance(cache[0], CacheList)
        out = [
            int(t)
            for t, _, _ in prompt_lookup_generate_step(
                prompt, model, max_tokens=24, sampler=GREEDY,
                prompt_cache=cache, backend="suffix_automaton",
            )
        ]
        self.assertEqual(len(out), 24)
        self.assertEqual(_pld_offset(cache[0]), prompt.size + len(out))


class TestPromptLookupGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")

    def _prompt(self, text):
        return mx.array(self.tokenizer.encode(text))

    def _run(self, prompt, cache, backend, max_tokens):
        return [
            int(t)
            for t, _, _ in prompt_lookup_generate_step(
                prompt, self.model, max_tokens=max_tokens,
                sampler=GREEDY, prompt_cache=cache, backend=backend,
            )
        ]

    def test_backends_run_and_cache_exact(self):
        prompt = self._prompt("def add(a, b):\n    return a + b\n# repeat: def add")
        for backend in ("ngram", "suffix_automaton"):
            cache = make_prompt_cache(self.model)
            out = self._run(prompt, cache, backend, max_tokens=48)
            self.assertGreater(len(out), 0)
            # Cache is left EXACTLY at prompt + emitted.
            self.assertEqual(_pld_offset(cache[0]), prompt.size + len(out))

    def test_ngram_lookup_composes_with_cached_prefix_history(self):
        full = self._prompt(
            "alpha beta gamma delta alpha beta gamma delta "
            "alpha beta gamma delta continue the sequence"
        )
        split = max(1, full.size // 2)
        prefix, tail = full[:split], full[split:]

        cold_cache = make_prompt_cache(self.model)
        cold = [
            int(token)
            for token, _, _ in prompt_lookup_generate_step(
                full,
                self.model,
                max_tokens=32,
                sampler=GREEDY,
                prompt_cache=cold_cache,
                backend="ngram",
                ngram_max=3,
            )
        ]

        cached = make_prompt_cache(self.model)
        self.model(prefix[None], cache=cached)
        mx.eval([entry.state for entry in cached])
        warm = [
            int(token)
            for token, _, _ in prompt_lookup_generate_step(
                tail,
                self.model,
                max_tokens=32,
                sampler=GREEDY,
                prompt_cache=cached,
                backend="ngram",
                ngram_max=3,
                history_prompt=full,
            )
        ]

        self.assertEqual(warm, cold)
        self.assertEqual(_pld_offset(cached[0]), full.size + len(warm))

    def test_matches_target_batched_greedy(self):
        # PLD output must equal the target's own greedy over prompt+output, i.e.
        # every emitted token is the batched argmax given its prefix. (This is the
        # correct losslessness bar; bit-identity with sequential generate_step is
        # NOT expected for any speculative decoder — batched verify != sequential.)
        prompt = self._prompt("List: apple, banana, apple, banana, apple,")
        cache = make_prompt_cache(self.model)
        out = self._run(prompt, cache, "suffix_automaton", max_tokens=40)
        full = mx.array(prompt.tolist() + out)[None]
        logits = self.model(full)
        L = prompt.size
        for i, tok in enumerate(out):
            self.assertEqual(int(mx.argmax(logits[0, L - 1 + i]).item()), tok)

    def test_max_tokens_boundary_cache_exact(self):
        prompt = self._prompt("Count: 1 2 3 1 2 3 1 2 3")
        for mt in (1, 7, 20):
            cache = make_prompt_cache(self.model)
            out = self._run(prompt, cache, "ngram", max_tokens=mt)
            self.assertEqual(len(out), mt)
            self.assertEqual(_pld_offset(cache[0]), prompt.size + mt)

    def test_cache_reuse_preserves_base(self):
        # Regression: a reused (non-empty) cache's existing prefix must survive
        # the end-of-run reconciliation.
        cache = make_prompt_cache(self.model)
        p1 = self._prompt("Write a haiku about the sea.")
        g1 = self._run(p1, cache, "suffix_automaton", max_tokens=32)
        base = _pld_offset(cache[0])
        self.assertEqual(base, p1.size + len(g1))
        # Second turn reuses the same cache.
        p2 = self._prompt("\nNow one about the mountains.\n")
        g2 = self._run(p2, cache, "suffix_automaton", max_tokens=32)
        self.assertEqual(_pld_offset(cache[0]), base + p2.size + len(g2))

    def test_adaptive_latch_runs(self):
        prompt = self._prompt("Explain why the sky is blue in one paragraph.")
        cache = make_prompt_cache(self.model)
        from mlx_lm.prompt_lookup import HybridStats
        stats = HybridStats()
        out = [
            int(t)
            for t, _, _ in prompt_lookup_generate_step(
                prompt, self.model, max_tokens=80, sampler=GREEDY,
                prompt_cache=cache, backend="suffix_automaton",
                adaptive=True, warmup=16, gate=0.5, stats=stats,
            )
        ]
        self.assertEqual(len(out), 80)
        self.assertEqual(_pld_offset(cache[0]), prompt.size + len(out))


if __name__ == "__main__":
    unittest.main()
