"""CPU replay of parser/HTTP boundaries; no model, tokenizer download or socket."""

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlx_lm.generate import TextStateMachine
from mlx_lm.server import (
    APIHandler,
    ResponseGenerator,
    _ChoiceAssembler,
    process_message_content,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call
from mlx_lm.tool_protocol import (
    ToolCallFormatter,
    ToolCallValidator,
    unsupported_constraint,
)
from test_parallel_sampling import _AssemblyHarness, _r


def offered(schema):
    return [{"type": "function", "function": {"name": "f", "parameters": schema}}]


def xml(value="hi", name="f"):
    return f"<function={name}><parameter=x>{value}</parameter></function>"


def context():
    return SimpleNamespace(
        tool_parser=parse_tool_call,
        text_sm=TextStateMachine(
            {
                "normal": [("<think>", "reasoning"), ("<tool_call>", "tool")],
                "reasoning": [("</think>", "normal"), ("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        ),
        initial_state="normal",
        prompt=[1, 2, 3],
        prompt_cache_count=0,
        stop=lambda: None,
    )


def assembled(parts, stream=False, finish="stop"):
    events = [_r(part, i) for i, part in enumerate(parts)] + [_r("", 999, finish)]
    result = _AssemblyHarness(context(), events, stream=stream).run()
    choices = [chunk["choices"][0] for chunk in result] if stream else result["choices"]
    calls, content, reasoning = [], "", ""
    for choice in choices:
        msg = choice["delta" if stream else "message"]
        content += msg.get("content") or ""
        reasoning += msg.get("reasoning") or ""
        calls.extend(call["function"] for call in msg.get("tool_calls", []))
    return content, reasoning, calls, choices[-1]["finish_reason"]


@pytest.mark.parametrize("stream", [False, True])
def test_all_two_piece_splits_multiple_calls_and_reasoning(stream):
    text = (
        "<think>plan</think><tool_call>" + xml() + xml("bye", "g") + "</tool_call>tail"
    )
    expected = (
        "tail",
        "plan",
        [
            {"name": "f", "arguments": '{"x": "hi"}'},
            {"name": "g", "arguments": '{"x": "bye"}'},
        ],
        "tool_calls",
    )
    for cut in range(len(text) + 1):
        assert assembled([text[:cut], text[cut:]], stream) == expected, cut
    assert assembled(list(text), stream) == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", ["stop", "length"])
def test_every_eof_cut_never_claims_success_without_a_complete_call(stream, finish):
    text = "<tool_call>" + xml() + "</tool_call>"
    for cut in range(len(text) + 1):
        _, _, calls, reason = assembled([text[:cut]], stream, finish)
        assert bool(calls) == (
            cut >= text.index("</function>") + len("</function>")
        ), cut
        assert reason == ("tool_calls" if calls and finish == "stop" else finish)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", ["stop", "length"])
def test_malformed_block_cannot_be_success(stream, finish):
    malformed = "<tool_call><function=f><parameter=x missing-angle</parameter></function></tool_call>"
    assert assembled([malformed], stream, finish)[2:] == ([], finish)


def test_formatted_once_and_finalize_idempotent():
    parser = Mock(return_value={"name": "f", "arguments": {}})
    a = _ChoiceAssembler(0, context(), ToolCallFormatter(parser, None, True))
    a.feed(_r("<tool_call>body</tool_call>", 1))
    calls = a.take_stream_payload()[1]
    a.feed(_r("", 2, "stop"))
    a.finalize()
    a.finalize()
    assert parser.call_count == 1
    assert len(calls) == 1 and calls[0]["index"] == 0
    assert a.tool_calls == [] and a.finish_reason == "tool_calls"


@pytest.mark.parametrize("stream", [False, True])
def test_interleaved_choices_keep_tool_state_and_terminal_reason(stream):
    events = [
        _r("<tool_call>" + xml("first"), 1, index=0),
        _r("<think>second</think>", 2, index=1),
        _r("</tool_call>", 3, index=0),
        _r("<tool_call>broken", 4, index=1),
        _r("", 5, "stop", index=0),
        _r("", 6, "length", index=1),
    ]
    result = _AssemblyHarness(context(), events, n=2, stream=stream).run()
    choices = (
        [c for chunk in result for c in chunk["choices"]]
        if stream
        else result["choices"]
    )
    terminals = {c["index"]: c["finish_reason"] for c in choices if c["finish_reason"]}
    assert terminals == {0: "tool_calls", 1: "length"}
    calls = [
        (c["index"], tc)
        for c in choices
        for tc in c["delta" if stream else "message"].get("tool_calls", [])
    ]
    assert len(calls) == 1 and calls[0][0] == 0
    assert json.loads(calls[0][1]["function"]["arguments"]) == {"x": "first"}


@pytest.mark.parametrize(
    "schema,value,expected",
    [
        ({"type": "integer"}, "9007199254740993", 9007199254740993),
        ({"type": "integer"}, "9.007199254740993e15", 9007199254740993),
        ({"type": "number"}, "9007199254740993", 9007199254740993),
        ({"type": "string"}, "null", "null"),
        ({"type": ["string", "null"]}, "null", None),
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "null", None),
        ({"type": "boolean"}, "false", False),
        ({"type": "boolean"}, "true", True),
    ],
)
def test_qwen_argument_fidelity(schema, value, expected):
    tools = offered({"type": "object", "properties": {"x": schema}})
    result = ToolCallFormatter(parse_tool_call, tools)([xml(value)])
    assert json.loads(result[0]["function"]["arguments"]) == {"x": expected}


@pytest.mark.parametrize(
    "typ,value",
    [
        ("integer", "1.5"),
        ("integer", "NaN"),
        ("integer", "1e9999999"),
        ("boolean", "not-a-bool"),
        ("boolean", "null"),
        ("number", "Infinity"),
    ],
)
def test_invalid_typed_literals_are_rejected(typ, value):
    tools = offered({"properties": {"x": {"type": typ}}})
    assert ToolCallFormatter(parse_tool_call, tools)([xml(value)]) == []


@pytest.mark.parametrize(
    "arguments", [{}, {"x": "bad"}, {"x": "ok", "extra": 1}, {"x": None}]
)
def test_declared_schema_is_validated_after_parsing(arguments):
    tools = offered(
        {
            "type": "object",
            "required": ["x"],
            "additionalProperties": False,
            "properties": {"x": {"type": "string", "enum": ["ok"]}},
        }
    )
    parser = lambda *_: [
        {"name": "f", "arguments": arguments},
        {"name": "f", "arguments": {"x": "ok"}},
    ]
    result = ToolCallFormatter(parser, tools, True)(["raw"])
    assert len(result) == 1 and result[0]["index"] == 0
    assert json.loads(result[0]["function"]["arguments"]) == {"x": "ok"}


def test_local_schema_ref_and_unresolvable_remote_ref():
    schema = {
        "$defs": {"arg": {"type": "integer"}},
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/arg"}},
    }
    v = ToolCallValidator(offered(schema))
    assert json.loads(v.arguments_json({"name": "f", "arguments": {"x": 3}})) == {
        "x": 3
    }
    schema["properties"]["x"] = {"$ref": "https://example.invalid/never-fetch"}
    with pytest.raises(ValueError, match="Invalid arguments"):
        ToolCallValidator(offered(schema)).arguments_json(
            {"name": "f", "arguments": {"x": 3}}
        )


@pytest.mark.parametrize(
    "call",
    [
        None,
        {},
        {"name": "", "arguments": {}},
        {"name": "g", "arguments": {}},
        {"name": "f", "arguments": []},
        {"name": "f", "arguments": {"x": float("nan")}},
    ],
)
def test_invalid_call_shape_or_unknown_function_is_not_emitted(call):
    assert ToolCallFormatter(lambda *_: call, offered({}))(["raw"]) == []


CONSTRAINTS = [
    {"tool_choice": "required"},
    {"tool_choice": {"type": "function", "function": {"name": "f"}}},
    {"tool_choice": "none"},
    {"tool_choice": "unknown"},
    {"parallel_tool_calls": False},
    {"grammar": {}},
    {"response_format": {"type": "json_schema"}},
    {"tools": [{"function": {"name": "f", "strict": True}}]},
]


@pytest.mark.parametrize("body", CONSTRAINTS)
def test_unified_rejects_unsupported_constraints_before_generation(body):
    class Handler(APIHandler):
        def _set_completion_headers(self, code=200):
            self.status = code

        def end_headers(self):
            pass

    h = object.__new__(Handler)
    raw = json.dumps({"messages": [], **body}).encode()
    h.path = "/v1/chat/completions"
    h.headers = {"Content-Length": str(len(raw))}
    h.rfile, h.wfile = io.BytesIO(raw), io.BytesIO()
    # No generator attribute: reaching generation/parameter lookup is a failure.
    h.do_POST()
    assert (
        h.status == 400 and "not supported" in json.loads(h.wfile.getvalue())["error"]
    )


def test_auto_and_plain_text_remain_supported():
    assert (
        unsupported_constraint(
            {
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "response_format": {"type": "text"},
            }
        )
        is None
    )


@pytest.mark.parametrize(
    "suffix,expected",
    [("", "normal"), ("<think>", "reasoning"), ("<think></think>", "normal")],
)
def test_reasoning_initialization_ignores_history_markers(suffix, expected):
    # Rendered ChatML boundaries, represented by deterministic CPU token IDs.
    symbols = {"<think>": 1, "</think>": 2, "<|im_end|>": 3}

    def encode(s):
        for token, tid in symbols.items():
            s = s.replace(token, chr(tid))
        return list(map(ord, s))

    tok = SimpleNamespace(
        has_thinking=True,
        eos_token_ids={3},
        rfind_think_start=lambda p, start=0: TokenizerWrapper._find(
            p, [1], start=start, reverse=True
        ),
        rfind_think_end=lambda p, start=0: TokenizerWrapper._find(
            p, [2], start=start, reverse=True
        ),
    )
    for history in ("assistant\n<think>old", "user\nHere is literal <think>"):
        prompt = encode(history + "<|im_end|>assistant\n" + suffix)
        assert ResponseGenerator._prompt_initial_state(tok, prompt) == expected


def test_tool_response_roundtrips_into_next_request():
    calls = ToolCallFormatter(parse_tool_call, None)(
        [xml('quote " / slash \\ / 日本語')]
    )
    messages = [
        {
            "role": "assistant",
            "content": None,
            "reasoning": "plan",
            "tool_calls": calls,
        },
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": "done"},
    ]
    echoed = json.loads(json.dumps(messages))
    process_message_content(echoed)
    assert echoed[0]["tool_calls"][0]["function"]["arguments"] == {
        "x": 'quote " / slash \\ / 日本語'
    }
    assert echoed[0]["reasoning_content"] == "plan"
    assert echoed[1]["tool_call_id"] == calls[0]["id"]
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)


def test_bad_qwen_sibling_does_not_discard_completed_calls():
    tools = offered({"properties": {"x": {"type": "boolean"}}})
    result = ToolCallFormatter(parse_tool_call, tools)(
        [xml("true") + xml("not-a-bool") + xml("false")]
    )
    assert [json.loads(c["function"]["arguments"]) for c in result] == [
        {"x": True},
        {"x": False},
    ]


@pytest.mark.parametrize("schema", [{"type": "invented"}, None, {"required": "x"}])
def test_invalid_tool_schema_is_rejected_before_generation(schema):
    with pytest.raises(ValueError, match="Invalid parameters schema"):
        ToolCallValidator(offered(schema))
