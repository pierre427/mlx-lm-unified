import json

from benchmarks.qwen4_ane_verifier_multisession import (
    choose_turns,
    jain,
    load_codex_turns,
)


def _line(payload):
    return json.dumps({"type": "response_item", "payload": payload})


def _message(role, turn, text):
    return {
        "type": "message",
        "role": role,
        "content": [
            {
                "type": "input_text" if role == "user" else "output_text",
                "text": text,
            }
        ],
        "internal_chat_message_metadata_passthrough": {"turn_id": turn},
    }


def test_private_trace_parser_keeps_only_completed_user_turns(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                _line(_message("user", "a", "real request")),
                _line(_message("assistant", "a", "real answer")),
                _line(_message("user", "b", "<app-context>injected")),
                _line(_message("assistant", "b", "ignored")),
                _line(_message("user", "c", "unfinished")),
            ]
        )
        + "\n"
    )
    assert load_codex_turns(trace) == [
        {"user": "real request", "assistant": "real answer"}
    ]


def test_turn_selection_spans_the_session_and_fairness_is_bounded():
    turns = [{"turn": value} for value in range(10)]
    assert [row["turn"] for row in choose_turns(turns, 4)] == [0, 3, 6, 9]
    assert jain([1.0, 1.0, 1.0]) == 1.0
    assert 0.0 < jain([1.0, 2.0, 3.0]) < 1.0
