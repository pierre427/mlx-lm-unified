import io
import json
from types import SimpleNamespace

from mlx_lm.cache_planes import PromptHostPlaneCache
from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    ResponseGenerator,
    setup_arg_parser,
)


class FakeTokenizer:
    has_chat_template = True
    has_tool_calling = True
    has_thinking = False
    chat_template = "fake-template-v1"
    init_kwargs = {"_commit_hash": "tokenizer-rev-a"}

    def __init__(self):
        self.apply_calls = 0
        self.encode_calls = 0

    def __len__(self):
        return 4096

    def apply_chat_template(
        self, messages, *, add_generation_prompt, tools, tokenize, **kwargs
    ):
        assert add_generation_prompt is True
        assert tokenize is True
        self.apply_calls += 1
        rendered = json.dumps(
            {"messages": messages, "tools": tools, "kwargs": kwargs},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return [byte + 1 for byte in rendered.encode()]

    def encode(self, text):
        self.encode_calls += 1
        return [byte + 1 for byte in text.encode()]


class FakeLegacyTokenizer(FakeTokenizer):
    has_chat_template = False
    chat_template = None

    def encode(self, text):
        self.encode_calls += 1
        self.last_encoded_text = text
        return [byte + 1 for byte in text.encode()]


class FakeThinkingLegacyTokenizer(FakeLegacyTokenizer):
    has_thinking = True

    @staticmethod
    def rfind_think_start(prompt, start=None):
        return 3

    @staticmethod
    def rfind_think_end(prompt, start=None):
        return -1


class FakeSegmentingTokenizer(FakeTokenizer):
    def apply_chat_template(
        self, messages, *, add_generation_prompt, tools, tokenize, **kwargs
    ):
        assert tokenize is True
        self.apply_calls += 1
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return [byte + 1 for byte in rendered.encode()]


class FakeOrderSensitiveTokenizer(FakeTokenizer):
    def apply_chat_template(
        self, messages, *, add_generation_prompt, tools, tokenize, **kwargs
    ):
        self.apply_calls += 1
        rendered = json.dumps(
            {"messages": messages, "tools": tools, "kwargs": kwargs},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        return [byte + 1 for byte in rendered.encode()]


def _generator(enabled=True):
    cli_args = SimpleNamespace(
        prompt_host_cache=enabled,
        prompt_host_cache_size=8,
        chat_template_args={"preserve_thinking": True},
    )
    generator = ResponseGenerator.__new__(ResponseGenerator)
    generator.model_provider = SimpleNamespace(
        cli_args=cli_args,
        model_key=("model-a", "adapter-a", None),
    )
    generator._prompt_host_cache = PromptHostPlaneCache(8)
    generator._prompt_host_tokenizer = None
    generator._prompt_host_tokenizer_epoch = 0
    return generator


def _request():
    return CompletionRequest(
        request_type="chat",
        prompt="",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ],
        tools=[{"type": "function", "function": {"name": "probe"}}],
        role_mapping=None,
    )


def _args(**chat_template_kwargs):
    return SimpleNamespace(chat_template_kwargs=chat_template_kwargs or None)


def test_prompt_host_serving_cache_miss_then_hit_reuses_exact_tokens_and_segments():
    generator = _generator()
    tokenizer = FakeTokenizer()

    first = generator._tokenize(tokenizer, _request(), _args(enable_thinking=False))
    second = generator._tokenize(tokenizer, _request(), _args(enable_thinking=False))

    assert first == second
    assert first[0] is not second[0]
    assert tokenizer.apply_calls == 1
    assert generator._prompt_host_cache.stats() == {
        "lookups": 2,
        "hits": 1,
        "misses": 1,
        "stores": 1,
        "replacements": 0,
        "evictions": 0,
        "invalidations": 0,
        "bypasses": 0,
        "entries": 1,
        "max_entries": 8,
        "bypass_reasons": {},
    }


def test_prompt_host_serving_cache_keys_template_options_and_invalidates_on_reload():
    generator = _generator()
    tokenizer = FakeTokenizer()

    base = generator._tokenize(tokenizer, _request(), _args(enable_thinking=False))
    changed = generator._tokenize(tokenizer, _request(), _args(enable_thinking=True))
    assert base != changed
    assert tokenizer.apply_calls == 2

    replacement = FakeTokenizer()
    restored = generator._tokenize(
        replacement, _request(), _args(enable_thinking=False)
    )
    assert restored == base
    assert replacement.apply_calls == 1

    stats = generator._prompt_host_cache.stats()
    assert stats["lookups"] == 3
    assert stats["hits"] == 0
    assert stats["misses"] == 3
    assert stats["stores"] == 3
    assert stats["invalidations"] == 2
    assert stats["entries"] == 1


def test_prompt_host_hit_restores_system_and_user_segment_boundaries():
    generator = _generator()
    tokenizer = FakeSegmentingTokenizer()
    request = CompletionRequest(
        request_type="chat",
        prompt="",
        messages=[
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
        ],
        tools=None,
        role_mapping=None,
    )

    first = generator._tokenize(tokenizer, request, _args())
    second = generator._tokenize(tokenizer, request, _args())

    assert first == second
    assert first[2] == ["system", "user"]
    assert first[0] == first[1][0] + first[1][1]
    assert tokenizer.apply_calls == 2
    assert generator._prompt_host_cache.stats()["hits"] == 1


def test_prompt_host_serving_cache_is_default_off_and_counts_bypasses():
    generator = _generator(enabled=False)
    tokenizer = FakeTokenizer()

    generator._tokenize(tokenizer, _request(), _args())
    generator._tokenize(tokenizer, _request(), _args())

    assert tokenizer.apply_calls == 2
    stats = generator._prompt_host_cache.stats()
    assert stats["lookups"] == 0
    assert stats["stores"] == 0
    assert stats["bypasses"] == 2
    assert stats["bypass_reasons"] == {"disabled": 2}


def test_legacy_chat_path_keeps_its_existing_unprocessed_content_semantics():
    generator = _generator()
    tokenizer = FakeLegacyTokenizer()
    request = _request()
    original_content = list(request.messages[0]["content"])

    first = generator._tokenize(tokenizer, request, _args())
    second = generator._tokenize(tokenizer, request, _args())

    assert first == second
    assert request.messages[0]["content"] == original_content
    assert "{'type': 'text', 'text': 'hello'}" in tokenizer.last_encoded_text
    assert tokenizer.encode_calls == 1
    assert generator._prompt_host_cache.stats()["hits"] == 1


def test_text_completion_reuses_the_server_encode_result():
    generator = _generator()
    tokenizer = FakeLegacyTokenizer()
    request = CompletionRequest("text", "plain prompt", [], None, None)

    first = generator._tokenize(tokenizer, request, _args())
    second = generator._tokenize(tokenizer, request, _args())

    assert first == second
    assert first[2] == ["assistant"]
    assert tokenizer.encode_calls == 1
    assert generator._prompt_host_cache.stats()["hits"] == 1


def test_prompt_host_hit_preserves_uncached_initial_state_for_unsegmented_paths():
    requests = (
        CompletionRequest("text", "plain prompt", [], None, None),
        CompletionRequest(
            "chat",
            "",
            [{"role": "user", "content": "legacy prompt"}],
            None,
            None,
        ),
    )

    for request in requests:
        generator = _generator()
        tokenizer = FakeThinkingLegacyTokenizer()
        first = generator._tokenize(tokenizer, request, _args())
        second = generator._tokenize(tokenizer, request, _args())

        # These legacy/plain paths deliberately return normal without running
        # the chat-template state scan.  A cache hit must not reinterpret it.
        assert first[3] == "normal"
        assert second == first


def test_prompt_host_fingerprint_preserves_template_input_mapping_order():
    generator = _generator()
    tokenizer = FakeTokenizer()

    message_a = CompletionRequest(
        "chat", "", [{"role": "user", "content": "hello"}], None, None
    )
    message_b = CompletionRequest(
        "chat", "", [{"content": "hello", "role": "user"}], None, None
    )
    assert (
        generator._prompt_host_metadata(tokenizer, message_a, _args())[
            "input_fingerprint"
        ]
        != generator._prompt_host_metadata(tokenizer, message_b, _args())[
            "input_fingerprint"
        ]
    )

    schema_a = {
        "type": "function",
        "function": {
            "name": "probe",
            "parameters": {
                "type": "object",
                "properties": {"alpha": {"type": "string"}, "beta": {}},
            },
        },
    }
    schema_b = {
        "type": "function",
        "function": {
            "name": "probe",
            "parameters": {
                "type": "object",
                "properties": {"beta": {}, "alpha": {"type": "string"}},
            },
        },
    }
    tool_a = CompletionRequest("chat", "", message_a.messages, [schema_a], None)
    tool_b = CompletionRequest("chat", "", message_a.messages, [schema_b], None)
    assert (
        generator._prompt_host_metadata(tokenizer, tool_a, _args())[
            "input_fingerprint"
        ]
        != generator._prompt_host_metadata(tokenizer, tool_b, _args())[
            "input_fingerprint"
        ]
    )

    kwargs_a = _args(alpha=1, beta=2)
    kwargs_b = _args(beta=2, alpha=1)
    assert (
        generator._prompt_host_metadata(tokenizer, message_a, kwargs_a)[
            "input_fingerprint"
        ]
        != generator._prompt_host_metadata(tokenizer, message_a, kwargs_b)[
            "input_fingerprint"
        ]
    )


def test_prompt_host_cache_does_not_alias_order_sensitive_template_requests():
    generator = _generator()
    tokenizer = FakeOrderSensitiveTokenizer()
    first_request = CompletionRequest(
        "chat", "", [{"role": "user", "content": "hello"}], None, None
    )
    second_request = CompletionRequest(
        "chat", "", [{"content": "hello", "role": "user"}], None, None
    )

    first = generator._tokenize(tokenizer, first_request, _args())
    second = generator._tokenize(tokenizer, second_request, _args())

    assert first != second
    assert tokenizer.apply_calls == 2
    assert generator._prompt_host_cache.stats()["hits"] == 0


def test_prompt_host_cache_cli_is_explicit_opt_in():
    parser = setup_arg_parser()
    defaults = parser.parse_args([])
    enabled = parser.parse_args(
        ["--prompt-host-cache", "--prompt-host-cache-size", "7"]
    )

    assert defaults.prompt_host_cache is False
    assert defaults.prompt_host_cache_size == 64
    assert enabled.prompt_host_cache is True
    assert enabled.prompt_host_cache_size == 7


def test_prompt_host_cache_status_exposes_configuration_and_engagement():
    generator = _generator()
    tokenizer = FakeTokenizer()
    generator._tokenize(tokenizer, _request(), _args())
    generator._tokenize(tokenizer, _request(), _args())

    handler = APIHandler.__new__(APIHandler)
    handler.path = "/v1/status/prompt-host-cache"
    handler.response_generator = generator
    handler.wfile = io.BytesIO()
    handler._set_completion_headers = lambda status=200: None
    handler.end_headers = lambda: None
    handler.do_GET()

    payload = json.loads(handler.wfile.getvalue())
    assert payload["configured"] is True
    assert payload["stats"]["hits"] == 1
    assert payload["stats"]["misses"] == 1
