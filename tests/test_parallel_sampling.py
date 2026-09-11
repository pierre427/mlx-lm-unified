"""Tests for OpenAI ``n>1`` parallel sampling (CPU-only, no model weights).

Covers the things the feature has to get right:

  * the API surface -- ``n`` is parsed, bounded, and greedy ``n>1`` is refused
    because argmax makes every sample the same continuation;
  * admission -- the model has to be batchable and draft-free, the routing
    decision uses the prefix-cache result the n=1 path uses, and the count cap
    is backed by a projected state budget, refused before the request is
    accepted;
  * the composition with self-MTP -- eligible requests use batched MTP, while
    explicit compatibility policies still refuse or demote them to plain;
  * per-sample independence -- one shared prefill, but separate sampling draws,
    separate token histories and separate logits processors;
  * the wire format -- one choice per sample with its own index, usage summing
    the completions, and an n=1 response pinned byte for byte to what the
    server sent before parallel sampling existed.

Everything runs on tiny synthetic tensors and a fake model.
"""

import io
import json
import types
import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import (
    ParallelSampleGenerator,
    PromptProcessingBatch,
    StopSequenceMatcher,
    TextStateMachine,
    prefill_prompt_cache,
)
from mlx_lm.apc import APCLookup
from mlx_lm.cache_capsule import (
    CacheCapsuleGeneration,
    CacheCapsulePool,
    prepare_prompt_cache_capsules,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import LaneRNG, make_sampler
from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    GenerationContext,
    RequestCompositionError,
    ResponseGenerator,
    _parallel_sampling_route,
    _parallel_sampling_state_bytes,
)

VOCAB = 8
# Two equally likely continuations, so independent rows diverge quickly.
COIN_TOKENS = (3, 5)


class CoinModel(nn.Module):
    """Emits a uniform distribution over ``COIN_TOKENS`` and keeps a KV cache."""

    def __init__(self, dims: int = 4):
        super().__init__()
        self.dims = dims

    def __call__(self, inputs, cache=None):
        B, S = inputs.shape
        if cache is not None and cache[0] is not None:
            kv = mx.ones((B, 1, S, self.dims), dtype=mx.float32)
            cache[0].update_and_fetch(kv, kv)
        logits = mx.full((B, S, VOCAB), -1e9, dtype=mx.float32)
        for token in COIN_TOKENS:
            logits[:, :, token] = 0.0
        return logits


def _fresh_cache():
    return [KVCache()]


class TestPrefillPromptCache(unittest.TestCase):
    def test_every_token_is_processed(self):
        cache = _fresh_cache()
        prefill_prompt_cache(CoinModel(), [1, 2, 3, 4, 5], cache, prefill_step_size=2)
        self.assertEqual(cache[0].offset, 5)

    def test_empty_prefill_is_a_noop(self):
        cache = _fresh_cache()
        prefill_prompt_cache(CoinModel(), [], cache)
        self.assertEqual(cache[0].offset, 0)


class TestParallelSampleGenerator(unittest.TestCase):
    """The engine: one prefill, n rows, independent per-row state."""

    def setUp(self):
        self.model = CoinModel()
        mx.random.seed(0)

    def _run(self, n, steps=24, samplers=None, logits_processors=None):
        cache = _fresh_cache()
        prompt = [1, 2, 3, 4]
        prefill_prompt_cache(self.model, prompt[:-1], cache)
        sampler = make_sampler(temp=1.0)
        parallel = ParallelSampleGenerator(
            self.model,
            cache,
            prompt[-1],
            n,
            max_tokens=steps,
            samplers=samplers if samplers is not None else [sampler] * n,
            logits_processors=logits_processors,
            stop_matchers=[StopSequenceMatcher()] * n,
            all_tokens=prompt[:-1],
        )
        try:
            sequences = [[] for _ in range(n)]
            finished = [None] * n
            while len(parallel) > 0:
                for index, r in parallel.next():
                    sequences[index].append(r.token)
                    if r.finish_reason is not None:
                        finished[index] = r
            return sequences, finished
        finally:
            parallel.close()

    def test_prefill_is_shared_and_rows_are_replicated(self):
        cache = _fresh_cache()
        prefill_prompt_cache(self.model, [1, 2, 3], cache)
        forwards = []
        model = self.model

        class CountingModel(nn.Module):
            def __call__(self, inputs, cache=None):
                forwards.append(tuple(inputs.shape))
                return model(inputs, cache=cache)

        parallel = ParallelSampleGenerator(
            CountingModel(),
            cache,
            4,
            3,
            max_tokens=2,
            samplers=[make_sampler(temp=1.0)] * 3,
            stop_matchers=[StopSequenceMatcher()] * 3,
        )
        try:
            while len(parallel) > 0:
                parallel.next()
        finally:
            parallel.close()

        # Every decode forward carries all three rows and no forward re-runs
        # the prompt: the prefix was paid once, before the replication.
        self.assertTrue(forwards)
        self.assertTrue(all(shape[0] == 3 and shape[1] == 1 for shape in forwards))

    def test_prepared_apc_capsule_is_the_first_model_cache_consumer(self):
        cache = KVCache()
        prefix = mx.ones((1, 1, 3, 4), dtype=mx.bfloat16)
        cache.update_and_fetch(prefix, prefix)
        mx.eval(cache.state)
        clock = CacheCapsuleGeneration()
        pool = CacheCapsulePool(clock, enabled=True)
        prepared = prepare_prompt_cache_capsules(
            [cache],
            target_batch=2,
            generation=clock.current,
            pool=pool,
            backend="gpu",
            source_prefix="parallel-test-apc",
        )
        seen = []

        class PreparedCoinModel(CoinModel):
            def __call__(self, inputs, cache=None):
                seen.append(
                    (
                        tuple(inputs.shape),
                        tuple(cache[0].keys.shape),
                        tuple(cache[0].offset.tolist()),
                    )
                )
                B, S = inputs.shape
                kv = mx.ones((B, 1, S, self.dims), dtype=mx.bfloat16)
                cache[0].update_and_fetch(kv, kv)
                logits = mx.full((B, S, VOCAB), -1e9, dtype=mx.float32)
                logits[:, :, COIN_TOKENS[0]] = 0.0
                return logits

        parallel = ParallelSampleGenerator(
            PreparedCoinModel(),
            [cache],
            4,
            2,
            max_tokens=1,
            stop_matchers=[StopSequenceMatcher()] * 2,
            prepared_prompt_cache=prepared.prompt_cache,
            prepared_prompt_cache_owner=prepared,
        )
        try:
            rows = parallel.next()
            self.assertEqual(len(rows), 2)
            self.assertEqual(seen[0], ((2, 1), (2, 1, 256, 4), (3, 3)))
        finally:
            parallel.close()
            pool.close()
        self.assertTrue(all(r.owner.released for r in prepared.receipts))

    def test_samples_are_distinct_and_two_sided(self):
        sequences, _ = self._run(n=4, steps=32)
        self.assertEqual(len(sequences), 4)
        # Distinct continuations: a copied row would be byte-identical.
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertNotEqual(sequences[i], sequences[j])
        # Independent draws from the same two-sided distribution: both tokens
        # must appear, and the pooled split must not be degenerate.
        flat = [t for seq in sequences for t in seq]
        self.assertEqual(set(flat), set(COIN_TOKENS))
        share = sum(t == COIN_TOKENS[0] for t in flat) / len(flat)
        self.assertGreater(share, 0.2)
        self.assertLess(share, 0.8)

    def test_every_sample_runs_to_its_own_max_tokens(self):
        sequences, finished = self._run(n=2, steps=6)
        self.assertEqual([len(s) for s in sequences], [6, 6])
        self.assertTrue(all(r is not None for r in finished))
        self.assertTrue(all(r.finish_reason == "length" for r in finished))

    def test_token_histories_do_not_alias(self):
        _, finished = self._run(n=2, steps=6)
        histories = [r.all_tokens for r in finished]
        self.assertEqual(len(histories[0]), len(histories[1]))
        self.assertIsNot(histories[0], histories[1])
        # The shared prompt prefix is present in both, the continuations differ.
        self.assertEqual(histories[0][:3], histories[1][:3])

    def test_shared_logits_processor_list_is_rejected(self):
        shared = []
        with self.assertRaises(ValueError):
            self._run(n=2, logits_processors=[shared, shared])

    def test_n_must_be_positive(self):
        with self.assertRaises(ValueError):
            self._run(n=0)


class TestFullSplitHandsOverCache(unittest.TestCase):
    """A prompt batch whose rows all move to generation must not deep copy."""

    def test_split_of_every_row_moves_the_cache(self):
        caches = [_fresh_cache(), _fresh_cache()]
        batch = PromptProcessingBatch(
            model=CoinModel(),
            uids=[0, 1],
            caches=caches,
            tokens=[[1], [1]],
            samplers=[None, None],
            fallback_sampler=lambda x: mx.argmax(x, axis=-1),
            logits_processors=[[], []],
            stop_matchers=[StopSequenceMatcher()] * 2,
            max_tokens=[4, 4],
        )
        merged = batch.prompt_cache
        moved = batch.split([0, 1])

        self.assertEqual(moved.uids, [0, 1])
        self.assertIs(moved.prompt_cache, merged)
        self.assertEqual(batch.uids, [])
        self.assertEqual(batch.prompt_cache, [])
        self.assertEqual(len(moved.samplers), 2)
        self.assertEqual(len(moved.logits_processors), 2)

    def test_partial_split_still_separates_caches(self):
        caches = [_fresh_cache(), _fresh_cache()]
        batch = PromptProcessingBatch(
            model=CoinModel(),
            uids=[0, 1],
            caches=caches,
            tokens=[[1], [1]],
            samplers=[None, None],
            fallback_sampler=lambda x: mx.argmax(x, axis=-1),
            logits_processors=[[], []],
            stop_matchers=[StopSequenceMatcher()] * 2,
            max_tokens=[4, 4],
        )
        merged = batch.prompt_cache
        moved = batch.split([1])

        self.assertEqual(moved.uids, [1])
        self.assertEqual(batch.uids, [0])
        self.assertIsNot(moved.prompt_cache[0], merged[0])


class TestContinuousBatchingStillWorks(unittest.TestCase):
    """The split fast path must not disturb ordinary continuous batching."""

    def test_staggered_prompts_split_partially_then_fully(self):
        from mlx_lm.generate import BatchGenerator

        generator = BatchGenerator(
            CoinModel(),
            completion_batch_size=4,
            prefill_batch_size=4,
            prefill_step_size=2,
            max_tokens=3,
        )
        try:
            uids = generator.insert(
                prompts=[[1, 2, 3, 4], [1, 2]],
                max_tokens=[3, 3],
                caches=[_fresh_cache(), _fresh_cache()],
            )
            produced = {uid: [] for uid in uids}
            for _ in range(24):
                _, generated = generator.next()
                for r in generated:
                    produced[r.uid].append(r.token)
                if all(len(v) >= 3 for v in produced.values()):
                    break
            self.assertEqual([len(v) for v in produced.values()], [3, 3])
        finally:
            generator.close()


class TestParallelSamplingRoute(unittest.TestCase):
    """n>1 vs self-MTP: explicit, never silent."""

    def setUp(self):
        self.cli = types.SimpleNamespace(
            self_mtp=True,
            self_mtp_num_draft=1,
            self_mtp_persistent=True,
            self_mtp_rate_gate=False,
            self_mtp_share_qsa_indices=False,
            self_mtp_share_qsa_indices_min_prompt_tokens=16384,
            self_mtp_window_size=0,
            self_mtp_window_sink_size=4,
            self_mtp_window_min_prompt_tokens=32768,
            kv_bits=None,
            parallel_sampling_mtp="refuse",
        )
        self.model = types.SimpleNamespace(mtp=object())

    @staticmethod
    def args(*, temperature=0.7, top_p=1.0, top_k=0, min_p=0.0, xtc=0.0):
        return types.SimpleNamespace(
            model=types.SimpleNamespace(draft="default_model"),
            prompt_lookup_ngram=0,
            sampling=types.SimpleNamespace(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                xtc_probability=xtc,
            ),
        )

    def test_refuse_policy_rejects_an_mtp_admissible_request(self):
        with self.assertRaises(ValueError) as cm:
            _parallel_sampling_route(self.args(), self.cli, self.model)
        self.assertIn("self-MTP", str(cm.exception))

    def test_plain_mode_demotes_with_a_stated_reason(self):
        self.cli.parallel_sampling_mtp = "plain"
        route, note = _parallel_sampling_route(self.args(), self.cli, self.model)
        self.assertEqual(route, "plain")
        self.assertIn("self-MTP disabled", note)

    def test_default_rate_gate_demotes_refuse_policy_to_plain(self):
        # The default rate gate is a frozen exclusion from the batched MTP
        # path, so the request becomes plain before the refuse policy. 92576ce.
        self.cli.self_mtp_rate_gate = True
        self.assertEqual(
            _parallel_sampling_route(self.args(), self.cli, self.model),
            ("plain", None),
        )

    def test_server_without_mtp_needs_no_note(self):
        self.cli.self_mtp = False
        self.assertEqual(
            _parallel_sampling_route(self.args(), self.cli, self.model),
            ("plain", None),
        )

    def test_request_that_would_not_use_mtp_is_not_refused(self):
        # Transformed sampling without the transformed verifier already fails
        # closed to plain decoding, so n>1 costs it nothing.
        route, note = _parallel_sampling_route(
            self.args(temperature=0.7, top_p=0.8), self.cli, self.model
        )
        self.assertEqual((route, note), ("plain", None))

    def test_a_sidecarless_apc_hit_is_not_refused(self):
        # The n=1 path would NOT have used MTP here (target state, no draft
        # state), so n>1 costs the request nothing and must be admitted.
        route, note = _parallel_sampling_route(
            self.args(),
            self.cli,
            self.model,
            prompt_tokens=256,
            cached_prompt_tokens=200,
            mtp_state=None,
        )
        self.assertEqual((route, note), ("plain", None))

    def test_an_apc_hit_with_a_sidecar_is_still_refused(self):
        with self.assertRaises(ValueError):
            _parallel_sampling_route(
                self.args(),
                self.cli,
                self.model,
                prompt_tokens=256,
                cached_prompt_tokens=200,
                mtp_state=("mtp-cache", "hidden"),
            )

    def test_n_gt_1_never_joins_the_continuous_batch(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = types.SimpleNamespace(
            is_batchable=True,
            cli_args=types.SimpleNamespace(self_mtp=False),
        )
        single = types.SimpleNamespace(seed=None, prompt_lookup_ngram=0, n=1)
        many = types.SimpleNamespace(seed=None, prompt_lookup_ngram=0, n=2)
        self.assertTrue(generator._is_batchable(single))
        self.assertFalse(generator._is_batchable(many))


class _StubDetokenizer:
    def __init__(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = f"<{token}>"

    def finalize(self):
        pass


class _StubTokenizer:
    has_thinking = False
    has_tool_calling = False
    tool_parser = None
    eos_token_ids = [7]

    def encode(self, text, add_special_tokens=True):
        return [1, 2, 3, 4, 5, 6]

    @property
    def detokenizer(self):
        return _StubDetokenizer()


class _StubPromptCache:
    def __init__(self):
        self.inserted = []

    def fetch_nearest_cache(self, model_key, tokens):
        return None, list(tokens)

    def insert_cache(self, model_key, tokens, cache, **kwargs):
        self.inserted.append((list(tokens), cache))

    def __len__(self):
        return len(self.inserted)

    @property
    def nbytes(self):
        return 0

    def stats_by_type(self):
        return {}


class _StubAPC(_StubPromptCache):
    """A prefix cache that reports an APC hit, with no MTP sidecar."""

    def __init__(self, hit_tokens):
        super().__init__()
        self.hit_tokens = hit_tokens
        self.capsule_generation = CacheCapsuleGeneration()

    def lookup(self, model_key, tokens):
        cache = [KVCache()]
        prefill_prompt_cache(CoinModel(), list(tokens[: self.hit_tokens]), cache)
        cache[0].keys = cache[0].keys.astype(mx.bfloat16)
        cache[0].values = cache[0].values.astype(mx.bfloat16)
        return APCLookup(
            cache=cache,
            remaining_tokens=list(tokens[self.hit_tokens :]),
            cached_tokens=self.hit_tokens,
            hit=True,
            hit_kind="prefix",
            miss_reason=None,
            capsule_generation=self.capsule_generation.current,
        )


class CachedCoinModel(CoinModel):
    def make_cache(self):
        return [KVCache()]


class MTPCoinModel(CachedCoinModel):
    """A model that carries an MTP head, so self-MTP admission can engage."""

    mtp = object()


class TestServeParallelSamples(unittest.TestCase):
    """The server path end to end: one prefill, n queued sample streams."""

    def _generator(
        self,
        *,
        self_mtp=False,
        mtp_mode="refuse",
        is_batchable=True,
        draft_model=None,
        state_budget_gb=None,
    ):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = types.SimpleNamespace(
            model=MTPCoinModel() if self_mtp else CachedCoinModel(),
            tokenizer=_StubTokenizer(),
            model_key=("stub", None, None),
            is_batchable=is_batchable,
            draft_model=draft_model,
            cli_args=types.SimpleNamespace(
                parallel_sampling_state_budget_gb=state_budget_gb,
                self_mtp=self_mtp,
                self_mtp_num_draft=1,
                self_mtp_persistent=True,
                self_mtp_rate_gate=False,
                self_mtp_share_qsa_indices=False,
                self_mtp_share_qsa_indices_min_prompt_tokens=16384,
                self_mtp_window_size=0,
                self_mtp_window_sink_size=4,
                self_mtp_window_min_prompt_tokens=32768,
                kv_bits=None,
                prefill_step_size=2,
                parallel_sampling_mtp=mtp_mode,
            ),
        )
        generator.prompt_cache = _StubPromptCache()
        generator._state_machine_cache = {}
        generator._lane_rng_root = LaneRNG(11)
        return generator

    @staticmethod
    def _args(n=2, max_tokens=4):
        return types.SimpleNamespace(
            n=n,
            max_tokens=max_tokens,
            model=types.SimpleNamespace(draft="default_model", model=None, adapter=None),
            sampling=types.SimpleNamespace(
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                xtc_probability=0.0,
                xtc_threshold=0.1,
            ),
            logits=types.SimpleNamespace(
                logit_bias=None,
                repetition_penalty=0.0,
                repetition_context_size=20,
                presence_penalty=0.0,
                presence_context_size=20,
                frequency_penalty=0.0,
                frequency_context_size=20,
            ),
            stop_words=[],
            top_logprobs=0,
            logprobs=False,
            seed=None,
            prompt_lookup_ngram=0,
            chat_template_kwargs=None,
        )

    @staticmethod
    def _drain(queue):
        items = []
        while True:
            item = queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            items.append(item)
        return items

    def _serve(self, generator, args):
        from queue import Queue

        rqueue = Queue()
        request = CompletionRequest("text", "hello", [], None, None)
        generator._serve_parallel_samples((rqueue, request, args))
        queued = self._drain(rqueue)
        ctx = next(item for item in queued if isinstance(item, GenerationContext))
        return ctx, [item for item in queued if item is not ctx]

    def test_two_samples_stream_with_their_own_indices(self):
        mx.random.seed(1)
        generator = self._generator()
        ctx, items = self._serve(generator, self._args(n=2, max_tokens=4))

        responses = [i for i in items if not isinstance(i, tuple)]
        self.assertEqual(ctx.prompt_cache_count, 0)
        self.assertEqual(sorted({r.index for r in responses}), [0, 1])
        by_index = {0: [], 1: []}
        for r in responses:
            by_index[r.index].append(r.token)
        self.assertEqual([len(v) for v in by_index.values()], [4, 4])
        self.assertNotEqual(by_index[0], by_index[1])
        terminal = [r for r in responses if r.finish_reason is not None]
        self.assertEqual(len(terminal), 2)
        self.assertTrue(all(r.finish_reason == "length" for r in terminal))

    def test_shared_prefix_is_stored_once_and_samples_are_not(self):
        mx.random.seed(2)
        generator = self._generator()
        self._serve(generator, self._args(n=3, max_tokens=3))

        self.assertEqual(len(generator.prompt_cache.inserted), 1)
        stored_tokens, _ = generator.prompt_cache.inserted[0]
        # The prompt is six tokens; the cache covers all but the seed token.
        self.assertEqual(stored_tokens, [1, 2, 3, 4, 5])

    def test_progress_is_reported_against_the_whole_prompt(self):
        mx.random.seed(3)
        generator = self._generator()
        _, items = self._serve(generator, self._args(n=2, max_tokens=2))

        progress = [i for i in items if isinstance(i, tuple)]
        self.assertTrue(progress)
        self.assertTrue(all(total == 6 for _, total in progress))
        # The last report marks the prompt complete, seed token included.
        self.assertEqual(progress[-1], (6, 6))

    def test_mtp_server_honors_the_refuse_policy(self):
        generator = self._generator(self_mtp=True)
        with self.assertRaises(ValueError):
            self._serve(generator, self._args(n=2, max_tokens=2))

    def test_mtp_server_serves_plain_when_configured(self):
        mx.random.seed(4)
        generator = self._generator(self_mtp=True, mtp_mode="plain")
        _, items = self._serve(generator, self._args(n=2, max_tokens=2))
        responses = [i for i in items if not isinstance(i, tuple)]
        self.assertEqual(sorted({r.index for r in responses}), [0, 1])

    def test_an_unbatchable_model_is_refused(self):
        generator = self._generator(is_batchable=False)
        with self.assertRaises(RequestCompositionError) as cm:
            self._serve(generator, self._args(n=2))
        self.assertIn("batchable", str(cm.exception))

    def test_a_draft_model_is_refused(self):
        generator = self._generator(draft_model=object(), is_batchable=False)
        with self.assertRaises(RequestCompositionError) as cm:
            self._serve(generator, self._args(n=2))
        self.assertIn("draft model", str(cm.exception))

    def test_a_sidecarless_apc_hit_still_serves_n_gt_1(self):
        # The n=1 path would have decoded this plainly (target state without
        # matching MTP state), so the n>1 request must not be refused.
        generator = self._generator(self_mtp=True, mtp_mode="refuse")
        generator.prompt_cache = _StubAPC(hit_tokens=3)
        _, items = self._serve(generator, self._args(n=2, max_tokens=2))
        responses = [i for i in items if not isinstance(i, tuple)]
        self.assertEqual(sorted({r.index for r in responses}), [0, 1])

    def test_apc_hit_capsule_reaches_parallel_serving_consumer(self):
        generator = self._generator()
        apc = _StubAPC(hit_tokens=3)
        generator.prompt_cache = apc
        generator._cache_capsule_pool = CacheCapsulePool(
            apc.capsule_generation, enabled=True
        )
        with mock.patch.dict(
            "os.environ",
            {
                "MLX_LM_CACHE_CAPSULE": "1",
                "MLX_LM_CACHE_CAPSULE_BACKEND": "gpu",
            },
        ):
            _, items = self._serve(generator, self._args(n=2, max_tokens=2))
        responses = [i for i in items if not isinstance(i, tuple)]
        self.assertEqual(sorted({r.index for r in responses}), [0, 1])
        self.assertGreater(generator._cache_capsule_pool.counters["requests"], 0)
        generator._cache_capsule_pool.close()

    def test_a_request_over_the_state_budget_is_refused(self):
        # The count cap is not a memory bound: n rows replicate the whole
        # prefix cache, so the projected state has to be checked.
        generator = self._generator(state_budget_gb=1e-9)
        with self.assertRaises(RequestCompositionError) as cm:
            self._serve(generator, self._args(n=2, max_tokens=4))
        self.assertIn("budget", str(cm.exception))

    def test_a_request_inside_the_state_budget_is_served(self):
        generator = self._generator(state_budget_gb=1.0)
        _, items = self._serve(generator, self._args(n=2, max_tokens=2))
        responses = [i for i in items if not isinstance(i, tuple)]
        self.assertEqual(sorted({r.index for r in responses}), [0, 1])

    def test_the_refusal_happens_before_the_request_is_accepted(self):
        # A refusal that arrived after the context would reach the client as a
        # broken stream instead of an HTTP error.
        from queue import Queue

        generator = self._generator(state_budget_gb=1e-9)
        rqueue = Queue()
        request = CompletionRequest("text", "hello", [], None, None)
        generator._serve_parallel_samples((rqueue, request, self._args(n=2)))
        self.assertIsInstance(rqueue.get(), RequestCompositionError)


class TestStateBudgetProjection(unittest.TestCase):
    """The projection the n>1 admission bound is built on."""

    def _cache(self, tokens):
        cache = _fresh_cache()
        prefill_prompt_cache(CoinModel(), list(range(tokens)), cache)
        return cache

    def test_an_empty_cache_is_not_projected(self):
        self.assertIsNone(
            _parallel_sampling_state_bytes(_fresh_cache(), 4, 0, 16)
        )

    def test_more_samples_project_more_state(self):
        cache = self._cache(8)
        small = _parallel_sampling_state_bytes(cache, 2, 8, 16)
        large = _parallel_sampling_state_bytes(cache, 8, 8, 16)
        self.assertLess(small, large)

    def test_the_completion_length_counts(self):
        cache = self._cache(8)
        short = _parallel_sampling_state_bytes(cache, 2, 8, 0)
        long = _parallel_sampling_state_bytes(cache, 2, 8, 4096)
        self.assertLess(short, long)

    def test_the_uncached_prompt_tail_counts(self):
        # Only part of the prompt is prefilled when the check runs; the
        # projection has to cover the whole prompt.
        cache = self._cache(8)
        measured = _parallel_sampling_state_bytes(cache, 2, 8, 0)
        whole = _parallel_sampling_state_bytes(cache, 2, 64, 0)
        self.assertGreater(whole, measured)


class _ValidationHarness(APIHandler):
    """APIHandler with just enough state to run parameter validation."""

    def __init__(self, *, n=1, temperature=0.7, max_n=1):
        self.n = n
        self.temperature = temperature
        self.response_generator = types.SimpleNamespace(
            cli_args=types.SimpleNamespace(
                parallel_sampling_max_n=max_n,
                single_model=False,
                model=None,
            )
        )

    def validate(self):
        self.validate_parallel_sampling(self.response_generator.cli_args)


class TestParallelSamplingValidation(unittest.TestCase):
    def test_default_is_one_sample(self):
        _ValidationHarness().validate()

    def test_n_gt_1_needs_the_server_flag(self):
        with self.assertRaises(ValueError) as cm:
            _ValidationHarness(n=2).validate()
        self.assertIn("--parallel-sampling-max-n", str(cm.exception))

    def test_n_is_capped(self):
        _ValidationHarness(n=4, max_n=4).validate()
        with self.assertRaises(ValueError):
            _ValidationHarness(n=5, max_n=4).validate()

    def test_greedy_n_gt_1_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            _ValidationHarness(n=2, temperature=0.0, max_n=4).validate()
        self.assertIn("greedy", str(cm.exception))

    def test_n_must_be_an_int(self):
        for bad in (True, 1.5, "2"):
            with self.subTest(n=bad):
                with self.assertRaises(ValueError):
                    _ValidationHarness(n=bad, max_n=4).validate()

    def test_zero_and_negative_are_refused(self):
        for bad in (0, -1):
            with self.subTest(n=bad):
                with self.assertRaises(ValueError):
                    _ValidationHarness(n=bad, max_n=4).validate()


class _AssemblyHarness(APIHandler):
    """APIHandler wired to a scripted (ctx, response) so the response-assembly
    loop in ``handle_completion`` runs without a model or a socket."""

    def __init__(
        self,
        ctx,
        raw_stream,
        *,
        n=1,
        stream=False,
        stream_options=None,
        logprobs=False,
        top_logprobs=0,
    ):
        self.wfile = io.BytesIO()
        self.status_codes = []
        self.stream = stream
        self.created = 0
        self.system_fingerprint = "fp-test"
        self.request_id = "req-test"
        self.object_type = "chat.completion.chunk" if stream else "chat.completion"
        self.requested_model = "test-model"
        self.requested_draft_model = None
        self.adapter = None
        self.stream_options = stream_options
        self.n = n
        response = iter(raw_stream)
        self.response_generator = types.SimpleNamespace(
            generate=lambda *a, **k: (ctx, response),
            cli_args=types.SimpleNamespace(allowed_origins=["*"]),
        )
        self.__dict__.update(
            dict(
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                xtc_probability=0.0,
                xtc_threshold=0.0,
                logit_bias=None,
                repetition_penalty=1.0,
                repetition_context_size=20,
                presence_penalty=0.0,
                presence_context_size=20,
                frequency_penalty=0.0,
                frequency_context_size=20,
                max_tokens=16,
                num_draft_tokens=0,
                prompt_lookup_ngram=0,
                prompt_lookup_tokens=8,
                prompt_lookup_adaptive=True,
                prompt_lookup_rate_gate=True,
                prompt_lookup_warmup=48,
                prompt_lookup_gate=0.12,
                prompt_lookup_rate_gate_probe=32,
                prompt_lookup_rate_gate_margin=0.0,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                seed=None,
                chat_template_kwargs=None,
            )
        )

    def _set_completion_headers(self, status_code=200):
        self.status_codes.append(status_code)

    def _set_stream_headers(self, status_code=200):
        self.status_codes.append(status_code)

    def send_header(self, *args, **kwargs):
        pass

    def end_headers(self):
        pass

    def raw(self):
        """Run and return the exact bytes written to the wire."""
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[],
            tools=None,
            role_mapping=None,
        )
        self.handle_completion(request, stop_words=[])
        return self.wfile.getvalue()

    def run(self):
        raw = self.raw().decode()
        if not self.stream:
            return json.loads(raw)
        return [
            json.loads(line[len("data: ") :])
            for line in raw.strip().split("\n\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]


def _ctx(prompt_cache_count=0):
    return types.SimpleNamespace(
        tool_parser=None,
        text_sm=TextStateMachine({"normal": []}),
        initial_state="normal",
        prompt=[1, 2, 3],
        prompt_cache_count=prompt_cache_count,
        stop=lambda: None,
    )


def _r(text, token, finish_reason=None, index=0, top_tokens=(), logprob=0.0):
    from mlx_lm.server import Response

    return Response(text, token, logprob, finish_reason, top_tokens, index)


class TestMultiChoiceWireFormat(unittest.TestCase):
    def test_single_sample_response_is_unchanged(self):
        stream = [_r("hi", 1), _r(" there", 2, finish_reason="length")]
        resp = _AssemblyHarness(_ctx(), stream).run()

        self.assertEqual(len(resp["choices"]), 1)
        self.assertEqual(resp["choices"][0]["index"], 0)
        self.assertEqual(resp["choices"][0]["message"]["content"], "hi there")
        self.assertEqual(resp["choices"][0]["finish_reason"], "length")
        self.assertEqual(
            resp["usage"],
            {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        )

    def test_two_samples_become_two_choices(self):
        stream = [
            _r("red", 1, index=0),
            _r("blue", 2, index=1),
            _r(" apple", 3, finish_reason="length", index=0),
            _r(" sky", 4, finish_reason="stop", index=1),
        ]
        resp = _AssemblyHarness(_ctx(), stream, n=2).run()

        self.assertEqual([c["index"] for c in resp["choices"]], [0, 1])
        self.assertEqual(resp["choices"][0]["message"]["content"], "red apple")
        self.assertEqual(resp["choices"][1]["message"]["content"], "blue")
        self.assertEqual(resp["choices"][0]["finish_reason"], "length")
        self.assertEqual(resp["choices"][1]["finish_reason"], "stop")
        # The prompt is billed once, the completions of both samples are summed.
        self.assertEqual(
            resp["usage"],
            {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        )

    def test_single_sample_stream_shape_is_unchanged(self):
        stream = [_r("hi", 1), _r(" there", 2, finish_reason="length")]
        chunks = _AssemblyHarness(_ctx(), stream, stream=True).run()

        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            [(c["choices"][0].get("delta", {}).get("content"),
              c["choices"][0]["finish_reason"]) for c in chunks],
            [("hi", None), (" there", None), (None, "length")],
        )
        self.assertTrue(all(c["choices"][0]["index"] == 0 for c in chunks))

    def test_a_finished_sample_closes_without_waiting_for_the_others(self):
        stream = [
            _r("a", 1, index=0),
            _r("b", 2, index=1),
            _r("", 3, finish_reason="stop", index=0),
            _r("c", 4, index=1),
            _r("", 5, finish_reason="stop", index=1),
        ]
        chunks = _AssemblyHarness(_ctx(), stream, n=2, stream=True).run()

        shape = [
            (c["choices"][0]["index"], c["choices"][0]["finish_reason"])
            for c in chunks
        ]
        self.assertEqual(
            shape,
            [(0, None), (1, None), (0, "stop"), (1, None), (1, "stop")],
        )

    def test_streamed_chunks_carry_their_choice_index(self):
        stream = [
            _r("red", 1, index=0),
            _r("blue", 2, index=1),
            _r("", 3, finish_reason="stop", index=0),
            _r("", 4, finish_reason="stop", index=1),
        ]
        chunks = _AssemblyHarness(_ctx(), stream, n=2, stream=True).run()

        indices = [c["choices"][0]["index"] for c in chunks]
        self.assertEqual(set(indices), {0, 1})
        by_index = {0: [], 1: []}
        for chunk in chunks:
            choice = chunk["choices"][0]
            by_index[choice["index"]].append(choice)
        self.assertEqual(by_index[0][0]["delta"].get("content"), "red")
        self.assertEqual(by_index[1][0]["delta"].get("content"), "blue")
        # Exactly one terminal chunk per sample.
        for index in (0, 1):
            terminal = [c for c in by_index[index] if c["finish_reason"]]
            self.assertEqual(len(terminal), 1)


class TestStreamOptions(unittest.TestCase):
    """``stream_options`` is optional and its keys are optional too."""

    @staticmethod
    def _chunks(stream_options):
        stream = [_r("hi", 1), _r(" there", 2, finish_reason="length")]
        harness = _AssemblyHarness(
            _ctx(), stream, stream=True, stream_options=stream_options
        )
        return harness.raw().decode()

    def test_an_empty_stream_options_still_terminates_the_stream(self):
        raw = self._chunks({})
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))
        self.assertNotIn('"usage"', raw)

    def test_include_usage_false_sends_no_usage_chunk(self):
        raw = self._chunks({"include_usage": False})
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))
        self.assertNotIn('"usage"', raw)

    def test_include_usage_true_sends_the_usage_chunk(self):
        raw = self._chunks({"include_usage": True})
        self.assertIn('"usage"', raw)
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))

    def test_a_non_dict_stream_options_is_rejected(self):
        handler = APIHandler.__new__(APIHandler)
        handler.stream = False
        handler.stream_options = ["include_usage"]
        with self.assertRaises(ValueError) as cm:
            # Only the first two checks are reachable without a full request.
            APIHandler.validate_model_parameters(handler)
        self.assertIn("stream_options", str(cm.exception))


class TestPreGenerationErrorStatus(unittest.TestCase):
    """An unsupported combination is a 400; an unknown model stays 404."""

    @staticmethod
    def _status(error):
        harness = _AssemblyHarness(_ctx(), [])

        def _raise(*args, **kwargs):
            raise error

        harness.response_generator.generate = _raise
        harness.raw()
        return harness.status_codes[-1]

    def test_request_composition_errors_are_400(self):
        self.assertEqual(
            self._status(RequestCompositionError("n>1 needs a batchable model")),
            400,
        )

    def test_other_errors_stay_404(self):
        self.assertEqual(self._status(ValueError("no such model")), 404)


# Captured from mlx_lm/server.py at a38e356^ (pre parallel sampling).
PRE_N_WIRE_BYTES = {
    "completion_stop": (
        '{"id": "req-test", "system_fingerprint": "fp-test", "object": "chat.'
        'completion", "model": "test-model", "created": 0, "choices": [{"inde'
        'x": 0, "finish_reason": "stop", "message": {"role": "assistant", "co'
        'ntent": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": '
        '2, "total_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}}'
        '}'
    ),
    "completion_length_cached": (
        '{"id": "req-test", "system_fingerprint": "fp-test", "object": "chat.'
        'completion", "model": "test-model", "created": 0, "choices": [{"inde'
        'x": 0, "finish_reason": "length", "message": {"role": "assistant", "'
        'content": "hi there"}}], "usage": {"prompt_tokens": 3, "completion_t'
        'okens": 2, "total_tokens": 5, "prompt_tokens_details": {"cached_toke'
        'ns": 2}}}'
    ),
    "completion_logprobs": (
        '{"id": "req-test", "system_fingerprint": "fp-test", "object": "chat.'
        'completion", "model": "test-model", "created": 0, "choices": [{"inde'
        'x": 0, "finish_reason": "stop", "logprobs": {"content": [{"id": 1, "'
        'token": "hi", "logprob": -0.5, "top_logprobs": [{"id": 1, "token": "'
        'hi", "logprob": -0.5}]}, {"id": 2, "token": " there", "logprob": -1.'
        '25, "top_logprobs": [{"id": 2, "token": " there", "logprob": -1.25}]'
        '}]}, "message": {"role": "assistant", "content": "hi"}}], "usage": {'
        '"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "prom'
        'pt_tokens_details": {"cached_tokens": 0}}}'
    ),
    "stream_stop": (
        'data: {"id": "req-test", "system_fingerprint": "fp-test", "object": '
        '"chat.completion.chunk", "model": "test-model", "created": 0, "choic'
        'es": [{"index": 0, "finish_reason": null, "delta": {"role": "assista'
        'nt", "content": "hi"}}]}\n\ndata: {"id": "req-test", "system_fingerpri'
        'nt": "fp-test", "object": "chat.completion.chunk", "model": "test-mo'
        'del", "created": 0, "choices": [{"index": 0, "finish_reason": "stop"'
        ', "delta": {"role": "assistant"}}]}\n\ndata: [DONE]\n\n'
    ),
    "stream_usage": (
        'data: {"id": "req-test", "system_fingerprint": "fp-test", "object": '
        '"chat.completion.chunk", "model": "test-model", "created": 0, "choic'
        'es": [{"index": 0, "finish_reason": null, "delta": {"role": "assista'
        'nt", "content": "hi"}}]}\n\ndata: {"id": "req-test", "system_fingerpri'
        'nt": "fp-test", "object": "chat.completion.chunk", "model": "test-mo'
        'del", "created": 0, "choices": [{"index": 0, "finish_reason": "lengt'
        'h", "delta": {"role": "assistant"}}]}\n\ndata: {"id": "req-test", "sys'
        'tem_fingerprint": "fp-test", "object": "chat.completion", "model": "'
        'test-model", "created": 0, "choices": [], "usage": {"prompt_tokens":'
        ' 3, "completion_tokens": 1, "total_tokens": 4, "prompt_tokens_detail'
        's": {"cached_tokens": 0}}}\n\ndata: [DONE]\n\n'
    ),
}


class TestSingleSampleWireBytes(unittest.TestCase):
    """n=1 serialization is pinned to the bytes it had before n>1 existed.

    The goldens below were captured from ``mlx_lm/server.py`` at ``a38e356^``
    -- the commit before parallel sampling -- with these same scripted
    responses. A change in field order, key set, number formatting or chunk
    framing fails here, so the byte-identity claim is a guard and not a
    one-time hand check.
    """

    def _bytes(self, **kwargs):
        return _AssemblyHarness(**kwargs).raw().decode()

    def test_completion_stop(self):
        stream = [_r("hi", 1), _r(" there", 2, finish_reason="stop")]
        self.assertEqual(
            self._bytes(ctx=_ctx(), raw_stream=stream),
            PRE_N_WIRE_BYTES["completion_stop"],
        )

    def test_completion_length_with_cached_tokens(self):
        stream = [_r("hi", 1), _r(" there", 2, finish_reason="length")]
        self.assertEqual(
            self._bytes(ctx=_ctx(prompt_cache_count=2), raw_stream=stream),
            PRE_N_WIRE_BYTES["completion_length_cached"],
        )

    def test_completion_with_logprobs(self):
        stream = [
            _r(
                "hi",
                1,
                logprob=-0.5,
                top_tokens=({"id": 1, "token": "hi", "logprob": -0.5},),
            ),
            _r(
                " there",
                2,
                finish_reason="stop",
                logprob=-1.25,
                top_tokens=({"id": 2, "token": " there", "logprob": -1.25},),
            ),
        ]
        self.assertEqual(
            self._bytes(
                ctx=_ctx(), raw_stream=stream, logprobs=True, top_logprobs=1
            ),
            PRE_N_WIRE_BYTES["completion_logprobs"],
        )

    def test_stream_stop(self):
        stream = [_r("hi", 1), _r(" there", 2, finish_reason="stop")]
        self.assertEqual(
            self._bytes(ctx=_ctx(), raw_stream=stream, stream=True),
            PRE_N_WIRE_BYTES["stream_stop"],
        )

    def test_stream_with_usage(self):
        stream = [_r("hi", 1, finish_reason="length")]
        self.assertEqual(
            self._bytes(
                ctx=_ctx(),
                raw_stream=stream,
                stream=True,
                stream_options={"include_usage": True},
            ),
            PRE_N_WIRE_BYTES["stream_usage"],
        )


if __name__ == "__main__":
    unittest.main()
