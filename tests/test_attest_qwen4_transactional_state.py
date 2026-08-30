from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "attest_qwen4_transactional_state.py"
)
_SPEC = importlib.util.spec_from_file_location("qwen4_real_attestation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
attestation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(attestation)


def test_no_warning_pmset_snapshot_is_qualified(monkeypatch) -> None:
    output = """\
Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded
"""

    class Result:
        stdout = output

    monkeypatch.setattr(attestation.subprocess, "run", lambda *args, **kwargs: Result())
    assert attestation._thermal_snapshot()["qualified"] is True


def test_unknown_pmset_snapshot_fails_closed(monkeypatch) -> None:
    class Result:
        stdout = "unrecognized healthy-looking output"

    monkeypatch.setattr(attestation.subprocess, "run", lambda *args, **kwargs: Result())
    assert attestation._thermal_snapshot()["qualified"] is False


def test_numeric_throttle_snapshot_fails_closed(monkeypatch) -> None:
    class Result:
        stdout = "CPU_Speed_Limit = 80\nThermal_Level = 1"

    monkeypatch.setattr(attestation.subprocess, "run", lambda *args, **kwargs: Result())
    assert attestation._thermal_snapshot()["qualified"] is False
