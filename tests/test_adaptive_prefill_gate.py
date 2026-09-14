import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / "benchmarks" / "adaptive_prefill_gate.py"
SPEC = importlib.util.spec_from_file_location("adaptive_prefill_gate", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def run(*, adaptive, itl=100.0, ttft=1000.0, wall=2000.0, scheduler=None):
    scheduler = scheduler or {}
    return {
        "server_metrics_before": {"scheduler": {}},
        "server_metrics_after": {
            "configured": {"adaptive_prefill": adaptive},
            "scheduler": {"prefill_rounds": 1, **scheduler},
        },
        "requests": [
            {
                "action": "complete",
                "tenant_id": "a",
                "started_offset_ms": 0.0,
                "wall_ms": wall,
                "ttft_ms": ttft,
                "itl_ms": [itl] * 20,
                "completion_tokens": 20,
                "finish_reason": "length",
            }
        ],
    }


def test_adaptive_gate_passes_engaged_improvement():
    baseline = run(adaptive=False, itl=100.0)
    candidate = run(
        adaptive=True,
        itl=75.0,
        scheduler={
            "adaptive_prefill_release_rounds": 2,
            "adaptive_prefill_slack_deferred_rounds": 2,
            "adaptive_prefill_chunk_histogram": {"64": 2},
        },
    )

    report = MODULE.compare(baseline, candidate)

    assert report["passed"]
    assert report["candidate"]["engagement"]["adaptive_prefill_chunk_histogram"] == {
        "64": 2
    }


def test_adaptive_gate_rejects_null_mechanism():
    baseline = run(adaptive=False, itl=100.0)
    candidate = run(adaptive=True, itl=75.0)

    report = MODULE.compare(baseline, candidate)

    assert not report["passed"]
    assert not report["gates"]["candidate_engaged"]
