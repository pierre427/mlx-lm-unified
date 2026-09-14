import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / "benchmarks" / "decode_priority_cadence_gate.py"
SPEC = importlib.util.spec_from_file_location("decode_priority_cadence_gate", PATH)
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def run(cadence, *, ttft=100.0, itl=100.0, wall=1000.0, tokens=100):
    before = {
        "prefill_rounds": 10,
        "decode_priority_release_rounds": 2,
        "decode_priority_deferred_rounds": 8,
    }
    after = {
        "prefill_rounds": 12,
        "decode_priority_release_rounds": 4,
        "decode_priority_deferred_rounds": 14,
    }
    return {
        "server_metrics_before": {"scheduler": before},
        "server_metrics_after": {
            "configured": {"decode_priority_cadence": cadence},
            "scheduler": after,
        },
        "requests": [
            {
                "request_id": "a",
                "tenant_id": "tenant-a",
                "action": "complete",
                "ttft_ms": ttft,
                "itl_ms": [itl, itl],
                "started_offset_ms": 0.0,
                "wall_ms": wall,
                "completion_tokens": tokens,
                "quality": {
                    "expected_contains_pass": True,
                    "cross_request_canary_pass": True,
                    "own_canary_pass": True,
                },
                "error": None,
            }
        ],
    }


def test_gate_accepts_engaged_candidate_that_meets_tradeoff():
    report = GATE.compare(
        run(1),
        run(4, ttft=120.0, itl=75.0, wall=1000.0, tokens=100),
    )

    assert report["passed"] is True
    assert report["candidate"]["engagement"]["decode_priority_deferred_rounds"] == 6


def test_gate_rejects_zero_engagement_candidate():
    candidate = run(4, itl=75.0)
    candidate["server_metrics_after"]["scheduler"] = candidate["server_metrics_before"][
        "scheduler"
    ].copy()

    report = GATE.compare(run(1), candidate)

    assert report["gates"]["candidate_engaged"] is False
    assert report["passed"] is False
