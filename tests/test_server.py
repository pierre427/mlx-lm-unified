# Copyright © 2024-2026 Apple Inc.

import http
import io
import json
import threading
import types
import unittest
from unittest.mock import patch

import mlx.core as mx
import requests

from mlx_lm.generate import TextStateMachine
from mlx_lm.models.cache import KVCache, PromptTrie
from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    LRUPromptCache,
    ModelProvider,
    Response,
    ResponseGenerator,
    SamplingArguments,
    _make_sampler,
    _measure_kv_cost,
)
from mlx_lm.tool_parsers.mistral import parse_tool_call as mistral_parse_tool_call
from mlx_lm.utils import load


class TestModelProvider(unittest.TestCase):
    @staticmethod
    def make_provider(model):
        provider = ModelProvider.__new__(ModelProvider)
        provider.cli_args = types.SimpleNamespace(
            trust_remote_code=False,
            use_default_chat_template=False,
        )
        provider.is_distributed = False
        provider.model_key = None
        provider.model = model
        provider.tokenizer = object() if model is not None else None
        provider.draft_model = None
        provider._tokenizer_config = {}
        return provider

    def test_model_swap_clears_cache_before_loading_replacement(self):
        provider = self.make_provider(object())
        replacement_model = object()
        replacement_tokenizer = types.SimpleNamespace(chat_template="template")

        with (
            patch("mlx_lm.server.mx.clear_cache") as clear_cache,
            patch("mlx_lm.server.make_prompt_cache", return_value=[]),
        ):

            def load_replacement(*args, **kwargs):
                clear_cache.assert_called_once_with()
                self.assertIsNone(provider.model)
                self.assertIsNone(provider.tokenizer)
                return replacement_model, replacement_tokenizer

            with patch("mlx_lm.server.load", side_effect=load_replacement):
                provider._load("replacement")

        self.assertIs(provider.model, replacement_model)
        self.assertIs(provider.tokenizer, replacement_tokenizer)

    def test_initial_model_load_does_not_clear_cache(self):
        provider = self.make_provider(None)
        model = object()
        tokenizer = types.SimpleNamespace(chat_template="template")

        with (
            patch("mlx_lm.server.mx.clear_cache") as clear_cache,
            patch("mlx_lm.server.load", return_value=(model, tokenizer)),
            patch("mlx_lm.server.make_prompt_cache", return_value=[]),
        ):
            provider._load("model")

        clear_cache.assert_not_called()

    def test_kv_bits_keeps_mergeable_model_batchable(self):
        """Quantized KV is supported by BatchGenerator and must not make the
        server reject an otherwise mergeable model from continuous batching.
        """
        provider = self.make_provider(None)
        provider.cli_args.kv_bits = 4
        model = object()
        tokenizer = types.SimpleNamespace(chat_template="template")
        mergeable_cache = types.SimpleNamespace(merge=lambda *args: None)

        with (
            patch("mlx_lm.server.load", return_value=(model, tokenizer)),
            patch(
                "mlx_lm.server.make_prompt_cache",
                return_value=[mergeable_cache],
            ),
        ):
            provider._load("model")

        self.assertTrue(provider.is_batchable)


class DummyModelProvider:
    def __init__(self, with_draft=False):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.model_key = (HF_MODEL_PATH, None)
        self.is_batchable = True

        # Add draft model support
        self.draft_model = None
        self.draft_model_key = None
        self.cli_args = type(
            "obj",
            (object,),
            {
                "adapter_path": None,
                "chat_template": None,
                "use_default_chat_template": False,
                "trust_remote_code": False,
                "draft_model": None,
                "state_budget_gb": None,
                "num_draft_tokens": 3,
                "prompt_lookup_ngram": 0,
                "prompt_lookup_tokens": 8,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
                "model": None,
                "decode_concurrency": 32,
                "prompt_concurrency": 8,
                "prefill_step_size": 2048,
                "prompt_batch_window": None,
                "prompt_cache_size": 10,
                "prompt_cache_bytes": 1 << 63,
                "prompt_cache_total_bytes": None,
                "allowed_origins": ["*"],
            },
        )

        if with_draft:
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH
            self.cli_args.draft_model = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer

    def load_default(self):
        return self.load("default_model", None, "default_model")


class MockCache:
    def __init__(self, value, is_trimmable: bool = True):
        self.value = value
        self._is_trimmable = is_trimmable

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value

    def is_trimmable(self):
        return self._is_trimmable

    def trim(self, n):
        assert self._is_trimmable
        return n


class TestTextStateMachine(unittest.TestCase):
    """Test the TextStateMachine buffering and stripping behavior."""

    def test_strips_control_sequences(self):
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "hi <tool_call>body</tool_call> bye")
        state, rest, s = sm.flush(state)
        full = text + rest
        self.assertEqual(full, "hi body bye")

    def test_back_to_back_tool_calls(self):
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        state = sm.make_state()
        state, t1, s = sm.step(state, "<tool_call>call1</tool_call>")
        state, t2, s = sm.step(state, "<tool_call>call2</tool_call>")
        state, rest, s = sm.flush(state)
        full = t1 + t2 + rest
        self.assertEqual(full, "call1call2")

    def test_partial_match_buffered_then_flushed(self):
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        # First enter tool state
        state = sm.make_state()
        state, text, s = sm.step(state, "<tool_call>body</")
        self.assertEqual(s, "tool")
        # 'body' is emitted, '</' is buffered (partial match of '</tool_call>')
        self.assertEqual(text, "body")
        # flush releases the buffered text
        state, rest, s = sm.flush(state)
        self.assertEqual(rest, "</")

    def test_discard_drops_buffer(self):
        sm = TextStateMachine(
            {
                "normal": [("STOP", "normal")],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "hello ST")
        self.assertEqual(text, "hello ")
        # discard drops the buffered 'ST'
        state, s = sm.discard(state)
        self.assertEqual(s, "normal")

    def test_stop_words_stripped(self):
        sm = TextStateMachine(
            {
                "normal": [("STOP", "normal")],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "hello STOP world")
        state, rest, s = sm.flush(state)
        self.assertEqual(text + rest, "hello  world")

    def test_reasoning_to_tool_transition(self):
        # A tool call started inside a reasoning block must enter "tool".
        sm = TextStateMachine(
            {
                "normal": [("<think>", "reasoning"), ("<tool>", "tool")],
                "reasoning": [("</think>", "normal"), ("<tool>", "tool")],
                "tool": [("</tool>", "normal")],
            }
        )
        state = sm.make_state()
        state, _, s = sm.step(state, "<think>hmm")
        self.assertEqual(s, "reasoning")
        state, _, s = sm.step(state, "<tool>")
        self.assertEqual(s, "tool")
        state, _, s = sm.step(state, "</tool>")
        self.assertEqual(s, "normal")

    def test_empty_end_marker_stays_in_tool_on_discard(self):
        # Models with an empty tool_call_end (e.g. Mistral) never leave "tool";
        # discard on stop must preserve the state so the tool call is flushed.
        sm = TextStateMachine(
            {
                "normal": [("[TOOL_CALLS]", "tool")],
                "tool": [],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "[TOOL_CALLS]f[ARGS]{}")
        self.assertEqual(s, "tool")
        self.assertEqual(text, "f[ARGS]{}")
        state, s = sm.discard(state)
        self.assertEqual(s, "tool")


class _AssemblyHarness(APIHandler):
    """APIHandler wired to a scripted (ctx, response) so the response-assembly
    loop in ``handle_completion`` can be exercised without a model or socket."""

    def __init__(self, ctx, raw_stream, tools=None):
        # Deliberately skip BaseHTTPRequestHandler.__init__ (no socket).
        self.wfile = io.BytesIO()
        self.stream = False
        self.created = 0
        self.system_fingerprint = "fp-test"
        self.request_id = "req-test"
        self.object_type = "chat.completion"
        self.requested_model = "mistral-test"
        self.requested_draft_model = None
        self.adapter = None
        self.stream_options = None
        # Mirror production: generate() returns (ctx, raw token stream). The
        # assembly loop drives ctx.text_sm itself to strip control sequences.
        response = iter(raw_stream)
        self.response_generator = types.SimpleNamespace(
            generate=lambda *a, **k: (ctx, response),
            cli_args=types.SimpleNamespace(allowed_origins=["*"]),
        )
        self._request_tools = tools
        # Everything below only feeds GenerationArguments, which the stubbed
        # generate() ignores; values are irrelevant but must exist.
        self.__dict__.update(
            {
                k: v
                for k, v in dict(
                    temperature=0.0,
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
                    logprobs=False,
                    top_logprobs=0,
                    seed=None,
                    chat_template_kwargs=None,
                ).items()
            }
        )

    # No-op the socket-backed header plumbing.
    def _set_completion_headers(self, status_code=200):
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
            tools=self._request_tools,
            role_mapping=None,
        )
        self.handle_completion(request, stop_words=[])
        return json.loads(self.wfile.getvalue().decode())


class TestToolCallAssembly(unittest.TestCase):
    """Regression tests for surfacing tool calls in the assembled response.

    Guards the empty-tool_call_end (Mistral-format) case: the "tool" state is
    only left by a stop transition, so on EOS the assembly loop must
    ``discard`` (preserving current_state="tool") and flush the tool call.
    Post-#1501 architecture: the raw stream carries decoded text (markers
    included); ``ctx.text_sm`` strips them inside ``handle_completion``.
    """

    @staticmethod
    def _r(text, token, finish_reason=None):
        return Response(text, token, 0.0, finish_reason, ())

    def test_mistral_empty_end_token_flushes_tool_call_at_eos(self):
        # Mistral-format: "[TOOL_CALLS]NAME[ARGS]{json}" then EOS, empty
        # tool_call_end -> the "tool" state has no exit transition.
        ctx = types.SimpleNamespace(
            tool_parser=mistral_parse_tool_call,
            text_sm=TextStateMachine(
                {"normal": [("[TOOL_CALLS]", "tool")], "tool": []}
            ),
            initial_state="normal",
            prompt=[1, 2, 3],
            prompt_cache_count=0,
            stop=lambda: None,
        )
        raw_stream = [
            self._r("[TOOL_CALLS]", 9),
            self._r('get_weather[ARGS]{"city": "Paris"}', 42),
            self._r("", 2, finish_reason="stop"),  # EOS: text discarded
        ]

        resp = _AssemblyHarness(ctx, raw_stream).run()

        message = resp["choices"][0]["message"]
        self.assertEqual(resp["choices"][0]["finish_reason"], "tool_calls")
        self.assertNotIn("content", message)  # no stray text
        tool_calls = message["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]), {"city": "Paris"}
        )

    def test_non_empty_end_token_tool_call_still_works(self):
        # Guardrail: models whose tool_call_end is a real marker exit "tool"
        # via a tool->normal transition before EOS; that path must keep working.
        ctx = types.SimpleNamespace(
            tool_parser=mistral_parse_tool_call,
            text_sm=TextStateMachine(
                {
                    "normal": [("[TOOL_CALLS]", "tool")],
                    "tool": [("[/TOOL_CALLS]", "normal")],
                }
            ),
            initial_state="normal",
            prompt=[1, 2, 3],
            prompt_cache_count=0,
            stop=lambda: None,
        )
        raw_stream = [
            self._r("[TOOL_CALLS]", 9),
            self._r('get_weather[ARGS]{"city": "Paris"}', 42),
            self._r("[/TOOL_CALLS]", 10),
            self._r("", 2, finish_reason="stop"),
        ]

        resp = _AssemblyHarness(ctx, raw_stream).run()

        message = resp["choices"][0]["message"]
        self.assertEqual(resp["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(len(message["tool_calls"]), 1)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "seed": 999,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = json.loads(response.text)

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        first_text = response_body["choices"][0]["text"]
        self.assertEqual(
            first_text,
            json.loads(requests.post(url, json=post_data).text)["choices"][0]["text"],
        )

    def test_prompt_lookup_parameters_reject_invalid_values_before_streaming(self):
        url = f"http://localhost:{self.port}/v1/completions"
        invalid = (
            ("prompt_lookup_ngram", "3"),
            ("prompt_lookup_ngram", True),
            ("prompt_lookup_ngram", -1),
            ("prompt_lookup_tokens", "8"),
            ("prompt_lookup_tokens", False),
            ("prompt_lookup_tokens", 0),
            ("prompt_lookup_tokens", -1),
        )
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                response = requests.post(
                    url,
                    json={
                        "model": "default_model",
                        "prompt": "hello",
                        "stream": True,
                        name: value,
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.headers["Content-Type"], "application/json")
                self.assertIn(name, response.json()["error"])

    def test_malformed_logit_bias_returns_json_400(self):
        # {"1": null} used to raise an uncaught TypeError inside parameter
        # normalization, bypassing the documented JSON 400 path entirely.
        url = f"http://localhost:{self.port}/v1/completions"
        invalid = (
            {"1": None},
            {"1": "abc"},
            {"1": [1.0]},
            {"x": 1.0},
            {"1.5": 1.0},
            {"1": float("nan")},
            {"1": float("inf")},
        )
        for bias in invalid:
            with self.subTest(logit_bias=bias):
                response = requests.post(
                    url,
                    data=json.dumps(
                        {
                            "model": "default_model",
                            "prompt": "hello",
                            "max_tokens": 1,
                            "logit_bias": bias,
                        }
                    ),
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.headers["Content-Type"], "application/json")
                self.assertIn("logit_bias", response.json()["error"])

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_content_fragments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_null_tool_content(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_make_state_machine_empty_tool_call_end(self):
        class FakeTokenizer:
            has_thinking = False
            has_tool_calling = True
            tool_call_start = "[TOOL_CALLS]"
            tool_call_end = ""
            tool_call_start_tokens = (100,)
            tool_call_end_tokens = ()
            eos_token_ids = [2]

            def convert_ids_to_tokens(self, t):
                return f"<eos{t}>"

            def encode(self, text, add_special_tokens=False):
                return []

        stop_matcher, text_sm = self.response_generator._make_state_machine(
            ("fake-empty-end", None, None),
            FakeTokenizer(),
            stop_words=[],
        )

        # Verify the text state machine strips tool call markers
        text_state = text_sm.make_state()
        text_state, clean_text, s = text_sm.step(text_state, "hello[TOOL_CALLS]body")
        self.assertEqual(s, "tool")
        # 'hello' is before the match, 'body' flows through (no tool_call_end)
        self.assertEqual(clean_text, "hellobody")

        # Verify EOS stops via the stop matcher
        stop_state = stop_matcher.make_state()
        stop_state, matched = stop_matcher.match(stop_state, stop_matcher._trie, 2)
        self.assertTrue(matched)

    def test_handle_models(self):
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["object"], "list")
        self.assertIsInstance(response_body["data"], list)
        self.assertGreater(len(response_body["data"]), 0)
        model = response_body["data"][0]
        self.assertIn("id", model)
        self.assertEqual(model["object"], "model")
        self.assertIn("created", model)


class TestServerWithDraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(with_draft=True), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_handle_chat_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_streaming_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk_data = json.loads(data[6:])  # Skip the "data: " prefix
                    self.assertIn("choices", chunk_data)
                    self.assertEqual(len(chunk_data["choices"]), 1)
                    self.assertIn("delta", chunk_data["choices"][0])
                    chunk_count += 1

        # Make sure we got some streaming chunks
        self.assertGreater(chunk_count, 0)

    def test_prompt_cache_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        # First request to initialize cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about"},
            ],
        }

        first_response = requests.post(url, json=chat_post_data)
        self.assertEqual(first_response.status_code, 200)

        # Second request with same prefix should use cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about dragons."},
            ],
        }

        second_response = requests.post(url, json=chat_post_data)
        self.assertEqual(second_response.status_code, 200)

        # Both responses should have content
        first_response_body = json.loads(first_response.text)
        second_response_body = json.loads(second_response.text)

        self.assertIn("choices", first_response_body)
        self.assertIn("choices", second_response_body)
        self.assertIn("message", first_response_body["choices"][0])
        self.assertIn("message", second_response_body["choices"][0])
        self.assertIn("content", first_response_body["choices"][0]["message"])
        self.assertIn("content", second_response_body["choices"][0]["message"])

        # Ensure both generated content
        self.assertIsNotNone(first_response_body["choices"][0]["message"]["content"])
        self.assertIsNotNone(second_response_body["choices"][0]["message"]["content"])


class TestKeepalive(unittest.TestCase):
    def test_keepalive_callback(self):
        """Test keepalive callback sends SSE comments and handles errors"""
        from unittest.mock import Mock

        # Mock handler
        mock_wfile = io.BytesIO()
        handler = Mock()
        handler.wfile = mock_wfile

        # Test callback logic (same as in server.py)
        def keepalive_callback(processed_tokens, total_tokens):
            if handler.stream:
                try:
                    handler.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        # Test streaming enabled
        handler.stream = True
        keepalive_callback(1024, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, ": keepalive 1024/4096\n\n")

        # Test streaming disabled
        handler.stream = False
        mock_wfile.seek(0)
        mock_wfile.truncate(0)
        keepalive_callback(2048, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, "")

        # Test error handling
        handler.stream = True
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError("Connection broken")

        # Should not raise exception
        try:
            keepalive_callback(3072, 4096)
        except Exception as e:
            self.fail(f"Callback should handle BrokenPipeError: {e}")


class TestPromptTrie(unittest.TestCase):
    def test_search_finds_shortest_longer_sequence(self):
        trie = PromptTrie()
        model = object()
        trie.add(model, [1, 2, 3, 4], "long")
        trie.add(model, [1, 5, 6], "short")

        result = trie.search(model, [1])

        self.assertEqual(result.longer, [1, 5, 6])
        self.assertEqual(result.common_prefix, 1)

    def test_search_longer_sequence_tie_breaking(self):
        trie = PromptTrie()
        model = object()
        trie.add(model, [1, 2, 3], "first")
        trie.add(model, [1, 4, 5], "second")

        result = trie.search(model, [1])

        self.assertEqual(result.longer, [1, 4, 5])


class TestLRUPromptCache(unittest.TestCase):
    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertTrue(c is None)
        self.assertEqual(t, tokens)

        c = [KVCache()]
        c[0].update_and_fetch(*get_kv(24))
        cache.insert_cache(model, t, c)

        # Fetching a cache that is strictly a prefix doesn't remove it from the
        # lru cache
        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].keys_and_values()
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache), 1)

        # Inserting a trimmable cache with shared prefix removes the prefixes
        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 1)

        # Fetching a cache with a shared prefix doesn't remove it either
        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].keys_and_values()
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)
        self.assertEqual(len(cache), 1)

        # Inserting a diverged cache actually creates another entry
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 2)

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [2])  # exact hit re-processes the last token
        c, t = cache.fetch_nearest_cache(model, [1])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [1])
        c, t = cache.fetch_nearest_cache(model, [1, 3, 4])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [3, 4])
        c, t = cache.fetch_nearest_cache(model, [2, 3, 4])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4])
        c, t = cache.fetch_nearest_cache(model, [2, 4, 5])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4, 5])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])
        cache.insert_cache(model, [3, 4], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [4])

        cache.insert_cache(model, [4, 5], [MockCache("test4")], cache_type="user")
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, None)
        self.assertEqual(t, [2, 3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [4])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [5])

        cache.insert_cache(model, [5, 6], [MockCache("test5")])
        cache.insert_cache(model, [6, 7], [MockCache("test6")])
        c, t = cache.fetch_nearest_cache(model, [5, 6])
        self.assertEqual(c, None)
        self.assertEqual(t, [5, 6])
        c, t = cache.fetch_nearest_cache(model, [6, 7])
        self.assertEqual(c, [MockCache("test6")])
        self.assertEqual(t, [7])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [5])

    def test_insert_trimmable_cache_removes_immediate_prefix(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("ab")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 2)

        cache.insert_cache(model, [1, 2, 3], [MockCache("abc")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 3)

    def test_insert_empty_tokens_does_not_self_destruct(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 4)

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNotNone(c)
        self.assertEqual(t, [])

    def test_fetch_empty_tokens_after_root_eviction(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        cache.insert_cache(model, [1], [MockCache("a")])

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNone(c)
        self.assertEqual(t, [])

    def test_lru_bytes(self):
        cache = LRUPromptCache(max_size=100, max_bytes=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("aaa")])
        cache.insert_cache(model, [3, 4], [MockCache("bbb")])
        cache.insert_cache(model, [4, 5], [MockCache("ccc")])
        cache.insert_cache(model, [6, 7], [MockCache("ddd")])

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.nbytes, 9)

        cache.trim_to(n_bytes=7)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 6)

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, None)
        self.assertEqual(t, [3, 4])


class TestMakeSampler(unittest.TestCase):
    def test_xtc_special_tokens(self):
        class FakeTokenizer:
            eos_token_ids = [0, 1, 9]

            def encode(self, text, add_special_tokens=False):
                return [3]

        sampling = SamplingArguments(
            temperature=0.6,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            xtc_probability=1.0,
            xtc_threshold=0.1,
        )
        args = type("obj", (object,), {"sampling": sampling})
        sampler = _make_sampler(args, FakeTokenizer())
        logits = mx.log(
            mx.array([[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02]])
        )
        token = sampler(logits)
        mx.eval(token)
        self.assertEqual(token.shape, (1,))


class TestLogitBiasValidation(unittest.TestCase):
    """validate_model_parameters must turn every malformed logit_bias into a
    ValueError (the JSON 400 path), never an uncaught TypeError, and must
    reject non-finite values."""

    @staticmethod
    def _handler(**overrides):
        handler = APIHandler.__new__(APIHandler)
        defaults = dict(
            stream=False,
            max_tokens=10,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            num_draft_tokens=3,
            prompt_lookup_ngram=0,
            prompt_lookup_tokens=8,
            repetition_penalty=0.0,
            repetition_context_size=20,
            presence_penalty=0.0,
            presence_context_size=20,
            frequency_penalty=0.0,
            frequency_context_size=20,
            logprobs=False,
            top_logprobs=-1,
            xtc_probability=0.0,
            xtc_threshold=0.0,
            requested_model="default_model",
            adapter=None,
            seed=None,
            logit_bias=None,
        )
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(handler, key, value)
        return handler

    def test_malformed_logit_bias_raises_value_error_not_type_error(self):
        malformed = (
            {"1": None},
            {"1": "abc"},
            {"1": [1.0]},
            {"1": {"v": 1.0}},
            {"1": True},
            {None: 1.0},
            {"x": 1.0},
            {"1.5": 1.0},
        )
        for bias in malformed:
            with self.subTest(logit_bias=bias):
                handler = self._handler(logit_bias=bias)
                with self.assertRaises(ValueError):
                    handler.validate_model_parameters()

    def test_non_finite_logit_bias_raises_value_error(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                handler = self._handler(logit_bias={"1": value})
                with self.assertRaisesRegex(ValueError, "finite"):
                    handler.validate_model_parameters()

    def test_valid_logit_bias_is_normalized(self):
        handler = self._handler(logit_bias={"5": 2, "7": -1.5, 9: 0.25})
        handler.validate_model_parameters()
        self.assertEqual(handler.logit_bias, {5: 2.0, 7: -1.5, 9: 0.25})


class TestSingleModelMode(unittest.TestCase):
    @staticmethod
    def _handler(requested_model):
        handler = TestLogitBiasValidation._handler(requested_model=requested_model)
        handler.response_generator = types.SimpleNamespace(
            cli_args=types.SimpleNamespace(
                single_model=True,
                model="configured-model",
            )
        )
        return handler

    def test_only_default_or_configured_model_is_admitted(self):
        self._handler("default_model").validate_model_parameters()
        self._handler("configured-model").validate_model_parameters()
        with self.assertRaisesRegex(ValueError, "only admits"):
            self._handler("another-model").validate_model_parameters()

    def test_models_endpoint_only_advertises_configured_model(self):
        handler = APIHandler.__new__(APIHandler)
        handler.path = "/v1/models"
        handler.created = 123
        handler.wfile = io.BytesIO()
        handler.response_generator = types.SimpleNamespace(
            cli_args=types.SimpleNamespace(
                single_model=True,
                model="configured-model",
                allowed_origins=["*"],
            )
        )
        handler._set_completion_headers = lambda *_args, **_kwargs: None
        handler.end_headers = lambda: None
        with patch("mlx_lm.server.scan_cache_dir") as scan_cache:
            handler.handle_models_request()
        scan_cache.assert_not_called()
        self.assertEqual(
            json.loads(handler.wfile.getvalue()),
            {
                "object": "list",
                "data": [
                    {
                        "id": "configured-model",
                        "object": "model",
                        "created": 123,
                    }
                ],
            },
        )


class TestKVBudgetProbeFailure(unittest.TestCase):
    def test_probe_failure_reaches_requester_and_thread_survives(self):
        """A probe error must be delivered to the requesting queue, and the
        generation thread must remain usable afterwards."""
        from queue import Queue
        from types import SimpleNamespace
        from unittest.mock import patch

        import mlx_lm.server as server_mod

        provider = DummyModelProvider()
        provider.cli_args.state_budget_gb = 1.0  # budget mode on → probe runs
        provider.cli_args.decode_concurrency = 32
        provider.cli_args.prompt_concurrency = 8
        provider.cli_args.prefill_step_size = 2048

        gen = ResponseGenerator(provider, LRUPromptCache())
        try:

            def request():
                rqueue = Queue()
                args = SimpleNamespace(
                    model=SimpleNamespace(
                        model="default_model", adapter=None, draft=None
                    ),
                    seed=None,
                )
                gen.requests.put((rqueue, {"prompt": "hello"}, args))
                return rqueue

            with patch.object(
                server_mod,
                "_measure_kv_cost",
                side_effect=ValueError("probe refused"),
            ):
                result = request().get(timeout=30)
                self.assertIsInstance(result, ValueError)

                # Thread must still be alive and serving further requests
                self.assertTrue(gen._generation_thread.is_alive())
                second = request().get(timeout=30)
                self.assertIsInstance(second, ValueError)
        finally:
            gen.stop_and_join()


class _FakeLeaf:
    """Synthetic stepped-capacity cache leaf for probe refusal tests."""

    def __init__(self, slope, step=256, fixed=0):
        self._slope = slope
        self._fixed = fixed
        self._tokens = 0
        self.step = step
        self.state = []

    def grow(self, n):
        self._tokens += n

    @property
    def nbytes(self):
        if self._slope == 0:
            return self._fixed
        cap = self._tokens
        if self.step:
            cap = -(-cap // self.step) * self.step
        return self._fixed + int(self._slope * cap)


class _FakeModel:
    """Model stand-in: make_cache returns crafted leaves; forwards grow them."""

    def __init__(self, leaf_specs):
        # (slope, step, fixed) specs — make_cache mints FRESH leaves each
        # call, matching real make_prompt_cache semantics
        self._specs = leaf_specs

    def make_cache(self):
        out = []
        for spec in self._specs:
            if isinstance(spec, tuple):
                leaf = _FakeLeaf(spec[0], step=spec[1], fixed=spec[2])
            else:
                leaf = spec  # pre-built object (opaque/step-less cases)
            out.append(leaf)
        return out

    def __call__(self, toks, cache=None):
        n = toks.shape[-1]

        def grow(cs):
            for c in cs:
                inner = getattr(c, "caches", None)
                if inner:
                    grow(inner)
                elif hasattr(c, "grow"):
                    c.grow(n)

        grow(cache)
        return toks


class TestMeasureKVCostRefusals(unittest.TestCase):
    def test_growing_leaf_without_step_refused(self):
        leaf = _FakeLeaf(slope=100.0)
        leaf.step = None
        with self.assertRaises(ValueError):
            _measure_kv_cost(_FakeModel([leaf]))

    def test_mixed_steps_refused(self):
        with self.assertRaises(ValueError):
            _measure_kv_cost(
                _FakeModel([_FakeLeaf(100.0, step=256), _FakeLeaf(100.0, step=128)])
            )

    def test_opaque_composite_refused(self):
        class _Opaque:
            state = []

        with self.assertRaises(ValueError):
            _measure_kv_cost(_FakeModel([_Opaque()]))

    def test_consistent_stepped_fit_accepted(self):
        """Per-leaf consistency: a well-behaved stepped linear cache yields
        the raw slope, near-zero fixed, and the validated common step; the
        internal 528-token verification passes."""
        fixed, per_tok, step = _measure_kv_cost(
            _FakeModel([(100.0, 256, 0), (50.0, 256, 0)])
        )
        self.assertEqual(step, 256)
        self.assertAlmostEqual(per_tok, 150.0, delta=1.0)
        self.assertLessEqual(fixed, per_tok * 2)

    def test_fixed_only_leaf_adds_no_step_requirement(self):
        """A fixed-size leaf (ArraysCache-like) alongside stepped growth is
        fine; its bytes appear in fixed, not slope."""
        fixed, per_tok, step = _measure_kv_cost(
            _FakeModel([(100.0, 256, 0), (0, None, 5000)])
        )
        self.assertEqual(step, 256)
        self.assertAlmostEqual(per_tok, 100.0, delta=1.0)
        self.assertGreaterEqual(fixed, 5000 * 0.99)


class _SlopeChangeLeaf(_FakeLeaf):
    """Grows at slope until 1280 tokens, then twice as fast — must be
    refused by the independent 2048 consistency check."""

    @property
    def nbytes(self):
        cap = self._tokens
        if self.step:
            cap = -(-cap // self.step) * self.step
        if cap <= 1280:
            return int(self._slope * cap)
        return int(self._slope * 1280 + 2 * self._slope * (cap - 1280))


class _FakeComposite:
    """CacheList-like wrapper exposing .caches."""

    def __init__(self, children):
        self.caches = children
        self.state = []


class _CancellingLeaf(_FakeLeaf):
    """Grows slower after 1280 — pairs with _SlopeChangeLeaf so aggregate
    bytes match the linear fit while each leaf individually deviates."""

    @property
    def nbytes(self):
        cap = self._tokens
        if self.step:
            cap = -(-cap // self.step) * self.step
        if cap <= 1280:
            return int(self._slope * cap)
        return int(self._slope * 1280)  # stops growing entirely


class TestMeasureKVCostConsistency(unittest.TestCase):
    def test_two_leaf_cancellation_refused(self):
        """Per-leaf verification: one leaf doubles and another stalls after
        1280 so AGGREGATE 2048 bytes match the fit exactly — each leaf
        individually deviates and must be refused."""

        class _M(_FakeModel):
            def make_cache(self):
                return [
                    _SlopeChangeLeaf(100.0, step=256),
                    _CancellingLeaf(100.0, step=256),
                ]

        with self.assertRaises(ValueError):
            _measure_kv_cost(_M([]))

    def test_negative_per_leaf_intercept_refused(self):
        """A leaf whose growth is superlinear before the warm point has a
        materially negative fitted intercept and must refuse — even when
        another leaf's positive intercept cancels it in aggregate."""

        class _LateStartLeaf(_FakeLeaf):
            @property
            def nbytes(self):
                cap = self._tokens
                if self.step:
                    cap = -(-cap // self.step) * self.step
                return int(self._slope * max(cap - 128, 0) * 1.5)

        class _M(_FakeModel):
            def make_cache(self):
                return [
                    _LateStartLeaf(100.0, step=256),
                    _FakeLeaf(0, step=None, fixed=50_000),  # cancels aggregate
                ]

        with self.assertRaises(ValueError):
            _measure_kv_cost(_M([]))

    def test_mild_late_growth_refused(self):
        """Codex P1 regression: slope 100 through 1280 then 120 after —
        only 7.5% underprojection at 2048 — must be refused (exact-boundary
        predictions tolerate arithmetic error only, not mild drift)."""

        class _MildLateLeaf(_FakeLeaf):
            @property
            def nbytes(self):
                cap = self._tokens
                if self.step:
                    cap = -(-cap // self.step) * self.step
                if cap <= 1280:
                    return int(self._slope * cap)
                return int(self._slope * 1280 + 1.2 * self._slope * (cap - 1280))

        class _M(_FakeModel):
            def make_cache(self):
                return [_MildLateLeaf(100.0, step=256)]

        with self.assertRaises(ValueError):
            _measure_kv_cost(_M([]))

    def test_rotating_window_inside_third_point_refused(self):
        """Codex P2 regression: a rotating window between 1280 and 2048
        must refuse BEFORE probing (the third point would cross it)."""
        from mlx_lm.models.cache import RotatingKVCache

        class _M(_FakeModel):
            def make_cache(self):
                return [RotatingKVCache(max_size=1536)]

        with self.assertRaises(ValueError):
            _measure_kv_cost(_M([]))

    def test_slope_change_after_fit_range_refused(self):
        class _M(_FakeModel):
            def make_cache(self):
                return [_SlopeChangeLeaf(100.0, step=256)]

        with self.assertRaises(ValueError):
            _measure_kv_cost(_M([]))

    def test_invalid_step_values_refused(self):
        for bad in (True, 2.5, 0, -8):
            leaf = _FakeLeaf(100.0)
            leaf.step = bad
            with self.assertRaises(ValueError):
                _measure_kv_cost(_FakeModel([leaf]))

    def test_nested_composite_recursed(self):
        """Reviewer nested-CacheList positive case: a composite wrapping a
        fixed leaf and a stepped growing child measures correctly through
        recursion."""

        class _M(_FakeModel):
            def make_cache(self):
                return [
                    _FakeComposite(
                        [
                            _FakeLeaf(0, step=None, fixed=4000),
                            _FakeLeaf(100.0, step=256),
                        ]
                    )
                ]

        fixed, per_tok, step = _measure_kv_cost(_M([]))
        self.assertEqual(step, 256)
        self.assertAlmostEqual(per_tok, 100.0, delta=1.0)
        self.assertGreaterEqual(fixed, 4000 * 0.99)


class TestMeasureKVCost(unittest.TestCase):
    def test_measures_linear_cache(self):
        model, _ = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")
        fixed, per_token, step = _measure_kv_cost(model)
        self.assertGreater(per_token, 0)
        self.assertEqual(step, 256)  # validated common KVCache step
        # RAW fit: no slack folded in (rounding is the admission layer's
        # job at cohort level — folding it here would double count)
        self.assertLessEqual(fixed, per_token * 8)
        # Cross-check against the analytic per-token KV size
        cfg = model.args
        expected = (
            cfg.num_hidden_layers
            * 2
            * cfg.num_key_value_heads
            * (cfg.hidden_size // cfg.num_attention_heads)
            * 2  # bf16
        )
        self.assertAlmostEqual(per_token, expected, delta=expected * 0.01)

    def test_safe_projection_covers_non_boundary_allocation(self):
        """528-token regression (codex safety design): step-rounded live
        bytes at a NON-boundary length must not exceed the safe projection.
        This is the exact shape of the W3 smoke defect (projection admitted
        more live cache than the budget)."""
        from mlx_lm.models.cache import make_prompt_cache

        model, _ = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")
        fixed, per_token, step = _measure_kv_cost(model)

        caches = make_prompt_cache(model)
        toks = mx.array([[(i % 100) + 1 for i in range(528)]])
        model(toks, cache=caches)
        mx.eval([c.state for c in caches])
        observed = sum(c.nbytes for c in caches)
        rounded = -(-528 // step) * step
        projected = fixed + per_token * rounded
        self.assertLessEqual(observed, projected)
        # And the UNROUNDED exact projection must be shown insufficient,
        # proving step-rounding is load-bearing
        self.assertGreater(observed, fixed + per_token * 528)


if __name__ == "__main__":
    unittest.main()
