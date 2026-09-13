import importlib.util
import sys
from pathlib import Path


BENCHMARKS = Path(__file__).parents[1] / "benchmarks"
PATH = BENCHMARKS / "qwen4_mtp_static_cohort_immediate_multiturn_gate.py"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "qwen4_mtp_static_cohort_immediate_multiturn_gate", PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def diagnostic(*, near_tie):
    return {"near_tie_candidate": near_tie}


def test_b1_diagnostic_checks_only_aligned_first_turns():
    diagnostics = {
        "0:1": None,
        "1:1": diagnostic(near_tie=True),
        # A follow-up inherits compute-shape-specific cache state and is not
        # an aligned first-divergence oracle against a B1 execution.
        "0:2": diagnostic(near_tie=False),
    }
    assert MODULE.first_turns_within_batch_shape_band(diagnostics)


def test_b1_diagnostic_rejects_first_turn_outside_shape_band():
    diagnostics = {
        "0:1": None,
        "1:1": diagnostic(near_tie=False),
    }
    assert not MODULE.first_turns_within_batch_shape_band(diagnostics)


def test_plan_defaults_to_same_shape_exact_replay(capsys):
    assert MODULE.main([]) == 0
    output = capsys.readouterr().out
    assert '"baseline_mode": "same-shape-exact"' in output


def test_plan_accepts_explicit_b1_diagnostic_mode(capsys):
    assert MODULE.main(["--baseline-mode", "b1-first-turn-diagnostic"]) == 0
    output = capsys.readouterr().out
    assert '"baseline_mode": "b1-first-turn-diagnostic"' in output
