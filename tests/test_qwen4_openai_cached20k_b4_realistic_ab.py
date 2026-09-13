import importlib.util
from pathlib import Path
from types import SimpleNamespace


PATH = Path(__file__).parents[1] / "benchmarks" / "qwen4_openai_cached20k_b4_realistic_ab.py"
SPEC = importlib.util.spec_from_file_location("qwen4_cached20k_ab", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def args():
    return SimpleNamespace(
        python="/venv/python", model="/model", host="127.0.0.1", port=8298,
        prefill_step_size=512,
    )


def test_plain_command_has_apc_and_no_speculative_lane():
    command = MODULE.server_command(args(), "plain")
    assert command[command.index("--prompt-cache-size") + 1] == "64"
    assert command[command.index("--decode-concurrency") + 1] == "4"
    assert "--self-mtp" not in command
    assert not any("prompt-lookup" in item for item in command)
    assert "--draft-model" not in command


def test_mtp_command_is_matched_except_explicit_depth_two_lane_flags():
    plain = MODULE.server_command(args(), "plain")
    mtp = MODULE.server_command(args(), "mtp")
    assert mtp[mtp.index("--self-mtp-num-draft") + 1] == "2"
    assert mtp[mtp.index("--self-mtp-max-lanes") + 1] == "4"
    assert mtp[mtp.index("--self-mtp-segment-aware-cohort-size") + 1] == "4"
    assert "--self-mtp-segment-aware-live-tip" in mtp
    for flag in ("--prompt-cache-size", "--decode-concurrency", "--prompt-concurrency"):
        assert plain[plain.index(flag) + 1] == mtp[mtp.index(flag) + 1]


def test_environment_enables_segmented_only_for_mtp(monkeypatch):
    for key in MODULE.MTP_ENV:
        monkeypatch.setenv(key, "stale")
    plain = MODULE.server_environment("plain")
    mtp = MODULE.server_environment("mtp")
    assert all(key not in plain for key in MODULE.MTP_ENV)
    assert all(mtp[key] == "1" for key in MODULE.MTP_ENV)


def test_common_prefix_length_stops_at_first_difference():
    assert MODULE.common_prefix_length([1, 2, 3], [1, 2, 4, 5]) == 2
    assert MODULE.common_prefix_length([1, 2], [1, 2, 3]) == 2


def test_compare_arms_requires_all_twelve_exact():
    def arm(token):
        return {
            "aggregate_completion_tps": 2.0,
            "turns": [
                {"responses": [{"tokens": [token]} for _ in range(4)]}
                for _ in range(3)
            ],
        }
    assert MODULE.compare_arms(arm(7), arm(7))["all_token_exact"] is True
    changed = arm(7)
    changed["turns"][2]["responses"][3]["tokens"] = [8]
    assert MODULE.compare_arms(arm(7), changed)["all_token_exact"] is False


def test_plan_only_exposes_both_arms_and_apc(capsys):
    assert MODULE.main([]) == 0
    output = capsys.readouterr().out
    assert '"plain_command"' in output
    assert '"mtp_command"' in output
    assert '"--prompt-cache-size"' in output
