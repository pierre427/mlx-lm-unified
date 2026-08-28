# Copyright © 2026 Apple Inc.

"""Tests for OpenAI ``n>1`` parallel sampling (CPU-only, no model weights).

Covers the four things the feature has to get right:

  * the API surface -- ``n`` is parsed, bounded, and greedy ``n>1`` is refused
    because argmax makes every sample the same continuation;
  * the composition with self-MTP -- there is no batched MTP path, so an n>1
    request that would use MTP is refused (or explicitly demoted to plain);
  * per-sample independence -- one shared prefill, but separate sampling draws,
    separate token histories and separate logits processors;
  * the wire format -- one choice per sample with its own index, usage summing
    the completions, and an unchanged single-choice response at n=1.

Everything runs on tiny synthetic tensors and a fake model.
"""

import io
import json
import types
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import (
    ParallelSampleGenerator,
    PromptProcessingBatch,
    StopSequenceMatcher,
    TextStateMachine,
    prefill_prompt_cache,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    ResponseGenerator,
    _parallel_sampling_route,
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
            self_mtp_rate_gate=True,
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

    def test_mtp_admissible_request_is_refused_by_default(self):
        with self.assertRaises(ValueError) as cm:
            _parallel_sampling_route(self.args(), self.cli, self.model)
        self.assertIn("self-MTP", str(cm.exception))

    def test_plain_mode_demotes_with_a_stated_reason(self):
        self.cli.parallel_sampling_mtp = "plain"
        route, note = _parallel_sampling_route(self.args(), self.cli, self.model)
        self.assertEqual(route, "plain")
        self.assertIn("self-MTP disabled", note)

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


class CachedCoinModel(CoinModel):
    def make_cache(self):
        return [KVCache()]


class MTPCoinModel(CachedCoinModel):
    """A model that carries an MTP head, so self-MTP admission can engage."""

    mtp = object()


class TestServeParallelSamples(unittest.TestCase):
    """The server path end to end: one prefill, n queued sample streams."""

    def _generator(self, *, self_mtp=False, mtp_mode="refuse"):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = types.SimpleNamespace(
            model=MTPCoinModel() if self_mtp else CachedCoinModel(),
            tokenizer=_StubTokenizer(),
            model_key=("stub", None, None),
            cli_args=types.SimpleNamespace(
                self_mtp=self_mtp,
                self_mtp_num_draft=1,
                self_mtp_persistent=True,
                self_mtp_rate_gate=True,
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
        ctx = rqueue.get()
        if isinstance(ctx, Exception):
            raise ctx
        return ctx, self._drain(rqueue)

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

    def test_mtp_server_refuses_by_default(self):
        generator = self._generator(self_mtp=True)
        with self.assertRaises(ValueError):
            self._serve(generator, self._args(n=2, max_tokens=2))

    def test_mtp_server_serves_plain_when_configured(self):
        mx.random.seed(4)
        generator = self._generator(self_mtp=True, mtp_mode="plain")
        _, items = self._serve(generator, self._args(n=2, max_tokens=2))
        responses = [i for i in items if not isinstance(i, tuple)]
        self.assertEqual(sorted({r.index for r in responses}), [0, 1])


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

    def __init__(self, ctx, raw_stream, *, n=1, stream=False):
        self.wfile = io.BytesIO()
        self.stream = stream
        self.created = 0
        self.system_fingerprint = "fp-test"
        self.request_id = "req-test"
        self.object_type = "chat.completion.chunk" if stream else "chat.completion"
        self.requested_model = "test-model"
        self.requested_draft_model = None
        self.adapter = None
        self.stream_options = None
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
                logprobs=False,
                top_logprobs=0,
                seed=None,
                chat_template_kwargs=None,
            )
        )

    def _set_completion_headers(self, status_code=200):
        pass

    def _set_stream_headers(self, status_code=200):
        pass

    def send_header(self, *args, **kwargs):
        pass

    def end_headers(self):
        pass

    def run(self):
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[],
            tools=None,
            role_mapping=None,
        )
        self.handle_completion(request, stop_words=[])
        raw = self.wfile.getvalue().decode()
        if not self.stream:
            return json.loads(raw)
        return [
            json.loads(line[len("data: ") :])
            for line in raw.strip().split("\n\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]


def _ctx():
    return types.SimpleNamespace(
        tool_parser=None,
        text_sm=TextStateMachine({"normal": []}),
        initial_state="normal",
        prompt=[1, 2, 3],
        prompt_cache_count=0,
        stop=lambda: None,
    )


def _r(text, token, finish_reason=None, index=0):
    from mlx_lm.server import Response

    return Response(text, token, 0.0, finish_reason, (), index)


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


if __name__ == "__main__":
    unittest.main()
