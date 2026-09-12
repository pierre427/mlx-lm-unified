"""CPU-only request capability and completed tool-call validation.

Validation happens after parsing; it does not constrain model generation.
Schemas use their declared draft (2020-12 by default), with local references
only. JSON Schema ``format`` remains an annotation.
"""

import json
import logging
import uuid


def normalize_tool_history(messages):
    """Copy history, decoding OpenAI argument strings for chat templates."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object")
        item = dict(message)
        if item.get("tool_calls") is not None:
            if not isinstance(item["tool_calls"], list):
                raise ValueError("message.tool_calls must be a list")
            calls = []
            for call in item["tool_calls"]:
                if not isinstance(call, dict) or not isinstance(
                    call.get("function"), dict
                ):
                    raise ValueError(
                        "Historical tool calls must contain a function object"
                    )
                function = dict(call["function"])
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError as exc:
                        raise ValueError(
                            "Historical tool arguments must be valid JSON"
                        ) from exc
                if not isinstance(arguments, dict):
                    raise ValueError(
                        "Historical tool arguments must decode to an object"
                    )
                # Also copy decoded mappings so template rendering cannot
                # mutate the client's history through a shared object.
                function["arguments"] = json.loads(
                    json.dumps(arguments, allow_nan=False)
                )
                calls.append({**call, "function": function})
            item["tool_calls"] = calls
        normalized.append(item)
    return normalized


def unsupported_constraint(body):
    """Return the constraint these post-hoc tool servers cannot guarantee."""
    if body.get("grammar") is not None:
        return "grammar"
    response_format = body.get("response_format")
    if response_format is not None:
        if not isinstance(response_format, dict):
            return "response_format"
        if response_format.get("type") not in (None, "text"):
            return "response_format"
        if response_format.get("grammar") is not None:
            return "response_format"
    # Even `none` requires suppressing tool emission, which these parsers do
    # not enforce. Do not silently treat any explicit constraint as `auto`.
    if body.get("tool_choice") not in (None, "auto"):
        return "tool_call"
    if body.get("parallel_tool_calls") is False:
        return "parallel_tool_calls"
    tools = body.get("tools")
    if tools is not None and not isinstance(tools, list):
        return "tools"
    for tool in tools or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            if tool["function"].get("strict") is True:
                return "strict"
    return None


class ToolCallValidator:
    """Validate emitted calls against the offered functions, without coercion."""

    def __init__(self, tools):
        self.validators = {}
        if tools is None:
            return
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")
        if not tools:
            return
        from jsonschema.exceptions import SchemaError
        from jsonschema.validators import validator_for
        from referencing import Registry

        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                raise ValueError("Each tool must contain a function object")
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Tool function names must be nonempty strings")
            if name in self.validators:
                raise ValueError(f"Duplicate tool function name: {name}")
            schema = function.get("parameters", {"type": "object"})
            try:
                cls = validator_for(schema)
                cls.check_schema(schema)
            except (SchemaError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid parameters schema for {name}: {exc}"
                ) from exc
            # An explicit empty registry cannot fetch client-provided URLs.
            self.validators[name] = cls(schema, registry=Registry())

    def arguments_json(self, call):
        """Return validated JSON arguments, or raise ValueError for a bad call."""
        if not isinstance(call, dict):
            raise ValueError("Tool call must be an object")
        name = call.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool call name must be a nonempty string")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("Tool call arguments must be an object")
        if self.validators and name not in self.validators:
            raise ValueError(f"Tool call names an unoffered function: {name}")
        encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        validator = self.validators.get(name)
        if validator is not None:
            from jsonschema.exceptions import ValidationError
            from referencing.exceptions import Unresolvable

            try:
                validator.validate(json.loads(encoded))
            except (ValidationError, Unresolvable) as exc:
                raise ValueError(f"Invalid arguments for {name}: {exc}") from exc
        return encoded


def tool_finish_reason(finish_reason, has_calls):
    """Only a natural stop with a valid call is successful tool completion."""
    if finish_reason in ("stop", "tool_calls"):
        return "tool_calls" if has_calls else "stop"
    return finish_reason


class ToolCallFormatter:
    def __init__(self, tool_parser, tools, streaming=False):
        self._idx = 0
        self._tool_parser = tool_parser
        self._tools = tools
        self._streaming = streaming
        self._validator = ToolCallValidator(tools)

    def _format(self, tc):
        # Copy before mutating -- `tc` is owned by the tool parser and may be
        # reused/inspected by its caller; pop/assign must not touch it.
        arguments = self._validator.arguments_json(tc)
        tc = dict(tc)
        tc_id = tc.pop("id", None) or str(uuid.uuid4())
        tc["arguments"] = arguments
        out = {
            "function": tc,
            "type": "function",
            "id": tc_id,
        }
        if self._streaming:
            out["index"] = self._idx
            self._idx += 1
        return out

    def __call__(self, tool_calls):
        if not tool_calls or self._tool_parser is None:
            return []

        result = []
        for tool_text in tool_calls:
            try:
                parsed = self._tool_parser(tool_text, self._tools)
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(
                    f"Failed to parse tool call ({type(e).__name__}: {e}) — "
                    f"tool text was likely truncated mid-generation."
                )
                continue
            if not isinstance(parsed, list):
                parsed = [parsed]
            for tc in parsed:
                try:
                    result.append(self._format(tc))
                except (KeyError, TypeError, ValueError) as e:
                    # One malformed call (e.g. missing "arguments") must not
                    # discard the valid siblings already parsed from this block.
                    logging.warning(
                        f"Dropping malformed tool call ({type(e).__name__}: {e})"
                    )
                    continue
        return result
