"""Regression tests for tool-parser hygiene bugs (H3, L2, L4, M7, M8).

Each test targets one confirmed bug from the parser audit:
  * union / anyOf param resolves to a concrete type (not defaulted to string)
  * an unknown / unresolvable type never raises out of the parser
  * multiple <function>/[TOOL_CALLS] blocks all parse (none dropped)
  * a nested dict-valued arg is preserved (not truncated at first '}')
"""

import unittest

from mlx_lm.tool_parsers import (
    function_gemma,
    glm47,
    laguna,
    mistral,
    qwen3_coder,
)
from mlx_lm.tool_parsers._schema import (
    infer_type_from_json_schema,
    is_string_type,
)


def _tools(name, properties):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": properties},
            },
        }
    ]


class TestSchemaHelper(unittest.TestCase):
    def test_resolves_anyof_union(self):
        self.assertEqual(
            infer_type_from_json_schema(
                {"anyOf": [{"type": "array"}, {"type": "null"}]}
            ),
            "array",
        )

    def test_resolves_list_form_type(self):
        self.assertEqual(
            infer_type_from_json_schema({"type": ["string", "null"]}), "string"
        )

    def test_prefers_non_null_branch_oneof(self):
        self.assertEqual(
            infer_type_from_json_schema(
                {"oneOf": [{"type": "null"}, {"type": "integer"}]}
            ),
            "integer",
        )

    def test_unresolvable_returns_none(self):
        self.assertIsNone(infer_type_from_json_schema({}))
        self.assertIsNone(infer_type_from_json_schema({"type": "null"}))
        self.assertIsNone(infer_type_from_json_schema("not-a-dict"))

    def test_is_string_type_handles_union(self):
        self.assertTrue(is_string_type({"type": ["string", "null"]}))
        self.assertTrue(is_string_type({"anyOf": [{"type": "string"}]}))
        self.assertFalse(is_string_type({"anyOf": [{"type": "array"}]}))


class TestQwen3CoderH3(unittest.TestCase):
    def test_anyof_array_param_typed_not_string(self):
        # H3: {"anyOf":[{"type":"array"},{"type":"null"}]} had no top-level
        # "type" and defaulted to string, returning the array as raw text.
        tools = _tools(
            "search",
            {"tags": {"anyOf": [{"type": "array"}, {"type": "null"}]}},
        )
        call = qwen3_coder.parse_tool_call(
            '<function=search>\n<parameter=tags>\n["a", "b"]\n</parameter>\n</function>',
            tools,
        )
        self.assertEqual(call["arguments"]["tags"], ["a", "b"])

    def test_list_form_string_union_preserved(self):
        # H3: list-form {"type":["string","null"]} matched no branch and hit
        # ast.literal_eval(prose) -> SyntaxError escaping the parser.
        tools = _tools("note", {"body": {"type": ["string", "null"]}})
        call = qwen3_coder.parse_tool_call(
            "<function=note>\n<parameter=body>\nhello world\n</parameter>\n</function>",
            tools,
        )
        self.assertEqual(call["arguments"]["body"], "hello world")

    def test_unknown_type_does_not_raise(self):
        # H3: an unresolved/unknown type must fall back to the raw string,
        # never raise (server only catches ValueError/JSONDecodeError).
        tools = _tools("f", {"x": {"type": ["weirdtype", "null"]}})
        call = qwen3_coder.parse_tool_call(
            "<function=f>\n<parameter=x>\nnot: valid, python\n</parameter>\n</function>",
            tools,
        )
        self.assertEqual(call["arguments"]["x"], "not: valid, python")

    def test_multiple_function_blocks_all_parsed(self):
        # L4: <function=(.*?)</function>$ merged blocks; calls 2..n were lost.
        text = (
            "<function=first>\n<parameter=a>\n1\n</parameter>\n</function>\n"
            "<function=second>\n<parameter=b>\n2\n</parameter>\n</function>"
        )
        tools = _tools("first", {"a": {"type": "integer"}}) + _tools(
            "second", {"b": {"type": "integer"}}
        )
        calls = qwen3_coder.parse_tool_call(text, tools)
        self.assertIsInstance(calls, list)
        self.assertEqual([c["name"] for c in calls], ["first", "second"])
        self.assertEqual(calls[0]["arguments"], {"a": 1})
        self.assertEqual(calls[1]["arguments"], {"b": 2})


class TestGlmLagunaL2(unittest.TestCase):
    def test_glm47_union_string_preserved(self):
        # L2: exact type=="string" check; union string fell through and got
        # wrongly deserialized.
        tools = _tools("q", {"loc": {"type": ["string", "null"]}})
        call = glm47.parse_tool_call(
            "q<arg_key>loc</arg_key><arg_value>12345</arg_value>", tools
        )
        self.assertEqual(call["arguments"]["loc"], "12345")

    def test_laguna_anyof_string_preserved(self):
        tools = _tools("q", {"loc": {"anyOf": [{"type": "string"}, {"type": "null"}]}})
        call = laguna.parse_tool_call(
            "<tool_call>q<arg_key>loc</arg_key><arg_value>67890</arg_value></tool_call>",
            tools,
        )
        self.assertEqual(call["arguments"]["loc"], "67890")


class TestFunctionGemmaM7(unittest.TestCase):
    def test_nested_dict_arg_preserved(self):
        # M7: \{(.*?)\} stopped at the first '}', truncating dict-valued args.
        call = function_gemma.parse_tool_call(
            "call:configure{settings:{enabled:true,limit:5}}"
        )
        self.assertEqual(
            call["arguments"], {"settings": {"enabled": True, "limit": 5}}
        )

    def test_no_colon_value_does_not_raise(self):
        # M7: index(":") could raise; balanced/JSON conversion is robust.
        call = function_gemma.parse_tool_call("call:ping{}")
        self.assertEqual(call, {"name": "ping", "arguments": {}})

    def test_multiple_calls_all_parsed(self):
        calls = function_gemma.parse_tool_call(
            "call:a{x:1}call:b{y:2}"
        )
        self.assertIsInstance(calls, list)
        self.assertEqual(calls, [
            {"name": "a", "arguments": {"x": 1}},
            {"name": "b", "arguments": {"y": 2}},
        ])


class TestMistralM8(unittest.TestCase):
    def test_two_calls_yield_two(self):
        # M8: greedy (\{.*\}) merged calls into invalid JSON -> all dropped.
        text = (
            '[TOOL_CALLS]search[ARGS]{"q": "weather"}'
            '[TOOL_CALLS]read_file[ARGS]{"path": "/tmp/x.txt"}'
        )
        calls = mistral.parse_tool_call(text)
        self.assertIsInstance(calls, list)
        self.assertEqual(calls, [
            {"name": "search", "arguments": {"q": "weather"}},
            {"name": "read_file", "arguments": {"path": "/tmp/x.txt"}},
        ])

    def test_nested_json_and_braces_in_string(self):
        # Balanced-brace + JSON-string handling: nested object and a '}' inside
        # a string value must not split the call.
        text = '[TOOL_CALLS]f[ARGS]{"cfg": {"a": 1}, "s": "has } brace"}'
        call = mistral.parse_tool_call(text)
        self.assertEqual(
            call,
            {"name": "f", "arguments": {"cfg": {"a": 1}, "s": "has } brace"}},
        )


if __name__ == "__main__":
    unittest.main()
