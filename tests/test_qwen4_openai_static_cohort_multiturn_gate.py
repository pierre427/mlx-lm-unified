import importlib.util
from pathlib import Path
from types import SimpleNamespace


PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "qwen4_openai_static_cohort_multiturn_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen4_openai_static_cohort_multiturn_gate", PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def args():
    return SimpleNamespace(
        python="/venv/python",
        model="/model",
        host="127.0.0.1",
        port=8297,
        prefill_step_size=512,
        num_draft=2,
    )


def test_server_command_keeps_static_cohort_explicit_and_default_off_elsewhere():
    command = MODULE.server_command(args())
    assert "--self-mtp-segment-aware-live-tip" in command
    index = command.index("--self-mtp-segment-aware-cohort-size")
    assert command[index + 1] == "4"
    assert "--no-self-mtp-rate-gate" in command
    assert "--self-mtp-share-qsa-indices" not in command
    assert command[:3] == ["/venv/python", "-m", "mlx_lm.server"]


def test_server_environment_explicitly_opts_into_segmented_experiment(monkeypatch):
    monkeypatch.delenv("MLX_LM_SEGMENTED_SELF_MTP", raising=False)
    monkeypatch.delenv("MLX_LM_SEGMENTED_SELF_MTP_TIMING", raising=False)
    environment = MODULE.server_environment()
    assert environment["MLX_LM_SEGMENTED_SELF_MTP"] == "1"
    assert environment["MLX_LM_SEGMENTED_SELF_MTP_TIMING"] == "1"
    assert "MLX_LM_SEGMENTED_SELF_MTP" not in __import__("os").environ


def test_response_token_digest_is_over_openai_logprob_ids():
    response = {
        "choices": [
            {"logprobs": {"content": [{"id": 4}, {"id": 9}, {"id": 17}]}}
        ]
    }
    tokens = MODULE.response_tokens(response)
    assert tokens == [4, 9, 17]
    assert MODULE.token_digest(tokens) == MODULE.token_digest([4, 9, 17])
    assert MODULE.token_digest(tokens) != MODULE.token_digest([4, 9, 18])


def test_first_divergence_requires_bidirectional_top_two_near_tie():
    left = {
        "token_trace": [
            {
                "id": 4,
                "logprob": -0.10,
                "top_logprobs": [
                    {"id": 4, "logprob": -0.10},
                    {"id": 9, "logprob": -0.35},
                ],
            }
        ]
    }
    right = {
        "token_trace": [
            {
                "id": 9,
                "logprob": -0.11,
                "top_logprobs": [
                    {"id": 9, "logprob": -0.11},
                    {"id": 4, "logprob": -0.36},
                ],
            }
        ]
    }
    divergence = MODULE.first_divergence(left, right, 0.5)
    assert divergence["certified_near_tie"] is True
    assert abs(divergence["left_gap"] - 0.25) < 1e-12
    assert MODULE.first_divergence(left, right, 0.2)["certified_near_tie"] is False


def test_replay_followup_is_wave_independent():
    assert MODULE.followup_text(3) == (
        "Immediate follow-up 3: verify cache ownership briefly."
    )


def test_counter_delta_ignores_non_integer_status_fields():
    assert MODULE.counter_delta(
        {"engaged": 2, "environment_enabled": True, "mode": "old"},
        {"engaged": 7, "environment_enabled": True, "mode": "new"},
    ) == {"engaged": 5, "environment_enabled": 0}


def test_qualification_fails_closed_on_error_or_missing_checks():
    complete = {key: True for key in MODULE.REQUIRED_CHECKS}
    assert MODULE.qualification_passed({"checks": complete}) is True
    assert (
        MODULE.qualification_passed(
            {"checks": {"isolated_server_reaped": True}}
        )
        is False
    )
    assert MODULE.qualification_passed({"checks": complete, "error": "HTTP 400"}) is False


def test_plan_only_does_not_start_server(capsys):
    assert MODULE.main(["--port", "8299"]) == 0
    output = capsys.readouterr().out
    assert '"execute": false' in output
    assert '"--self-mtp-segment-aware-live-tip"' in output
