# Copyright © 2026 Apple Inc.

import ast
from typing import Any, Dict, List

import regex as re

"""
Tool parser for Pythonic function call formats.

Parses assistant responses containing tool calls in formats like:
<|tool_call_start|>[function_name(arg1="value1", arg2=2)]<|tool_call_end|>
"""


# The block is a LIST of calls: [f1(a=1), f2(b="x")]. Match each call
# individually (quote-aware, so ')' inside a string arg doesn't terminate
# the call) — a single non-greedy search over the whole block merges the
# args of call 1..n together and silently drops calls 2..n.
_tool_block_regex = re.compile(r"\[(.*)\]", re.DOTALL)
_single_call_regex = re.compile(
    r"(\w+)\(((?:[^()\"']|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')*)\)",
    re.DOTALL,
)
# One argument per match. A value is quoted -- either quote style, escapes
# allowed -- or bare, and a bare value can hold a list or a dict, so brackets
# match as balanced groups. Without this a comma inside a string or inside a
# nested literal ends the value early and the argument arrives truncated.
_tool_args_regex = re.compile(
    r"""
    (?(DEFINE)
        (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
        (?P<grp>
            \[(?:(?&str)|(?&grp)|[^\[\]{}()"'])*\]
          | \{(?:(?&str)|(?&grp)|[^\[\]{}()"'])*\}
          | \((?:(?&str)|(?&grp)|[^\[\]{}()"'])*\)
        )
    )
    (?P<key>\w+)\s*=\s*
    (?:
        "(?P<dq>(?:[^"\\]|\\.)*)"
      | '(?P<sq>(?:[^'\\]|\\.)*)'
      | (?P<bare>(?:(?&str)|(?&grp)|[^,])+)
    )
    \s*(?:,\s*|$)
    """,
    re.DOTALL | re.VERBOSE,
)


def _parse_value(text: str, quote: str | None):
    """Turn matched value text into a Python object.

    A quoted value is tried as written first, then with its quotes put back
    so escape sequences survive. That order keeps the behavior a quoted
    scalar has today, where "12" reaches the tool as the number 12.
    """
    candidates = [text] if quote is None else [text, f"{quote}{text}{quote}"]
    for candidate in candidates:
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            pass
    # Not a literal, so pass the text through as a string.
    return text


def parse_tool_call(text: str, tools: Any | None = None):
    block = _tool_block_regex.search(text)
    if not block:
        raise ValueError("No function provided.")

    calls = []
    for match in _single_call_regex.finditer(block.group(1)):
        func_name = match.group(1)
        args_str = match.group(2)

        arguments = {}
        if args_str:
            for arg in _tool_args_regex.finditer(args_str):
                # Test the quoted groups for None, not for truth: an empty
                # string is a legitimate value.
                if (quoted := arg.group("dq")) is not None:
                    value = _parse_value(quoted, '"')
                elif (quoted := arg.group("sq")) is not None:
                    value = _parse_value(quoted, "'")
                else:
                    value = _parse_value(arg.group("bare").strip(), None)
                arguments[arg.group("key")] = value
        calls.append(dict(name=func_name, arguments=arguments))

    if not calls:
        raise ValueError("No function provided.")
    return calls[0] if len(calls) == 1 else calls


tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"
