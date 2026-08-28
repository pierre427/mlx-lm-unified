# Copyright © 2026 Apple Inc.

import unittest

from mlx_lm.tool_parsers.pythonic import parse_tool_call


class TestPythonicToolParser(unittest.TestCase):
    def test_single_call_returns_dict(self):
        out = parse_tool_call('[get_weather(city="SF", units=2)]')
        self.assertEqual(
            out, {"name": "get_weather", "arguments": {"city": "SF", "units": 2}}
        )

    def test_multiple_calls_all_drained(self):
        # A single non-greedy search over the block used to merge the args of
        # call 1..n and silently drop calls 2..n.
        out = parse_tool_call('[get_weather(city="SF"), get_time(tz="PST")]')
        self.assertEqual(
            out,
            [
                {"name": "get_weather", "arguments": {"city": "SF"}},
                {"name": "get_time", "arguments": {"tz": "PST"}},
            ],
        )

    def test_paren_inside_quoted_arg(self):
        out = parse_tool_call('[run(cmd="echo )")]')
        self.assertEqual(out, {"name": "run", "arguments": {"cmd": "echo )"}})

    def test_no_function_raises(self):
        with self.assertRaises(ValueError):
            parse_tool_call("no calls here")

    def test_comma_inside_single_quoted_value(self):
        # The value regex accepted only double quotes, so a single-quoted
        # value was cut at its first comma and the call still succeeded --
        # with a truncated file body.
        out = parse_tool_call("[write_file(path='a.txt', content='# Hello, world!')]")
        self.assertEqual(
            out,
            {
                "name": "write_file",
                "arguments": {"path": "a.txt", "content": "# Hello, world!"},
            },
        )

    def test_escaped_double_quotes_with_a_comma(self):
        out = parse_tool_call('[w(content="say \\"hi, there\\" now", mode="a")]')
        self.assertEqual(
            out,
            {
                "name": "w",
                "arguments": {"content": 'say "hi, there" now', "mode": "a"},
            },
        )

    def test_escaped_single_quotes_with_a_comma(self):
        out = parse_tool_call("[w(content='say \\'hi, there\\' now', mode='a')]")
        self.assertEqual(
            out,
            {
                "name": "w",
                "arguments": {"content": "say 'hi, there' now", "mode": "a"},
            },
        )

    def test_empty_string_values_stay_empty_strings(self):
        for text in ('[w(a="", b="x")]', "[w(a='', b='x')]"):
            with self.subTest(text=text):
                out = parse_tool_call(text)
                self.assertEqual(out["arguments"], {"a": "", "b": "x"})
                self.assertIsInstance(out["arguments"]["a"], str)

    def test_multiple_calls_with_commas_in_single_quoted_values(self):
        out = parse_tool_call("[a(x='one, two'), b(y='three, four')]")
        self.assertEqual(
            out,
            [
                {"name": "a", "arguments": {"x": "one, two"}},
                {"name": "b", "arguments": {"y": "three, four"}},
            ],
        )

    def test_double_quoted_path_is_unchanged(self):
        out = parse_tool_call('[f(a="SF", b=2, c=True, d=None, e="echo )")]')
        self.assertEqual(
            out["arguments"],
            {"a": "SF", "b": 2, "c": True, "d": None, "e": "echo )"},
        )

    def test_comma_inside_a_nested_literal(self):
        # A list or a dict argument holds top-level commas of its own. A
        # value pattern that stops at any comma keeps only its first item.
        out = parse_tool_call('[f(a=[1, 2, 3], d={"k": 1, "j": 2}, b=2)]')
        self.assertEqual(
            out["arguments"],
            {"a": [1, 2, 3], "d": {"k": 1, "j": 2}, "b": 2},
        )

    def test_comma_inside_a_deeply_nested_literal(self):
        out = parse_tool_call('[f(v=[{"a": 1}, {"b": [2, 3]}], z=9)]')
        self.assertEqual(out["arguments"], {"v": [{"a": 1}, {"b": [2, 3]}], "z": 9})

    def test_comma_inside_a_string_inside_a_nested_literal(self):
        out = parse_tool_call("[f(v=['one, two', 'three'], z=9)]")
        self.assertEqual(out["arguments"], {"v": ["one, two", "three"], "z": 9})


if __name__ == "__main__":
    unittest.main()
