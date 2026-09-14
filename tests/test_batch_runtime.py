from queue import Queue
from threading import Condition
from types import SimpleNamespace

import pytest

from mlx_lm.batch_runtime import (
    BatchFaultSpec,
    BatchOverloaded,
    BatchRuntimeMetrics,
    InjectedBatchFault,
)
from mlx_lm.server import APIHandler, GenerationArguments, Response, ResponseGenerator


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_request_lifecycle_reports_latency_fairness_and_receipts():
    clock = Clock()
    metrics = BatchRuntimeMetrics(history_size=8, clock=clock)
    metrics.admitted("a", "tenant-a", 0)
    clock.advance(0.010)
    metrics.dequeued("a", 0)
    metrics.lane_attached("a", 2, "self_mtp")
    clock.advance(0.020)
    metrics.token("a", {"engaged": True})
    clock.advance(0.005)
    metrics.token("a")
    clock.advance(0.005)
    metrics.terminal("a", "completed")

    snapshot = metrics.snapshot(queue_depth=3, memory={"prompt_cache_bytes": 7})
    assert snapshot["latency_ms"]["queue"]["p99"] == pytest.approx(10.0)
    assert snapshot["latency_ms"]["ttft"]["p99"] == pytest.approx(30.0)
    assert snapshot["latency_ms"]["itl"]["p99"] == pytest.approx(5.0)
    assert snapshot["fairness"]["jain_tenant_token_rate"] == 1.0
    assert snapshot["counters"]["mechanism_receipts"] == 1
    assert snapshot["gauges"]["queue_depth"] == 3
    assert snapshot["memory"]["prompt_cache_bytes"] == 7
    assert snapshot["completed_requests"][0]["mechanism"] == "self_mtp"


def test_histories_are_bounded():
    clock = Clock()
    metrics = BatchRuntimeMetrics(history_size=2, clock=clock)
    for index in range(3):
        request_id = str(index)
        metrics.admitted(request_id, "tenant", 0)
        metrics.dequeued(request_id, 0)
        metrics.terminal(request_id, "completed")
    snapshot = metrics.snapshot()
    assert [row["request_id"] for row in snapshot["completed_requests"]] == ["1", "2"]
    assert len(snapshot["events"]) == 2


def test_fault_spec_is_default_off_and_strict():
    with pytest.raises(ValueError, match="disabled"):
        BatchFaultSpec.parse({"kind": "lane_abort"}, enabled=False)
    with pytest.raises(ValueError, match="one of"):
        BatchFaultSpec.parse({"kind": "unknown"}, enabled=True)
    with pytest.raises(ValueError, match="only valid"):
        BatchFaultSpec.parse({"kind": "cache_evict", "after_tokens": 1}, enabled=True)
    assert BatchFaultSpec.parse(
        {"kind": "lane_abort", "after_tokens": 4}, enabled=True
    ) == BatchFaultSpec("lane_abort", 4)


def test_inflight_limit_rejects_without_incrementing():
    generator = ResponseGenerator.__new__(ResponseGenerator)
    generator._admission = Condition()
    generator._paused = False
    generator._inflight = 1
    generator.admission_timeout = 0.0
    generator.requests = Queue()
    generator.batch_metrics = BatchRuntimeMetrics()
    generator.model_provider = SimpleNamespace(
        cli_args=SimpleNamespace(max_inflight_requests=1)
    )

    with pytest.raises(BatchOverloaded, match="limit is 1"):
        generator._admit_request("rejected", "tenant")

    assert generator._inflight == 1
    snapshot = generator.batch_metrics.snapshot()
    assert snapshot["admission_decisions"] == {"max_inflight": 1}
    assert snapshot["counters"]["rejected"] == 1


def test_batching_status_endpoint_exposes_configuration_and_memory():
    handler = APIHandler.__new__(APIHandler)
    handler.path = "/v1/status/batching"
    captured = []
    handler._json_ok = captured.append
    metrics = BatchRuntimeMetrics()

    class PromptCache:
        nbytes = 17

        def __len__(self):
            return 2

    handler.response_generator = SimpleNamespace(
        prompt_cache=PromptCache(),
        batch_metrics=metrics,
        requests=Queue(),
        cli_args=SimpleNamespace(
            decode_concurrency=8,
            prompt_concurrency=4,
            decode_priority_cadence=4,
            max_inflight_requests=12,
            self_mtp_verification_row_cap=24,
            batch_fault_injection=False,
        ),
        _batch_decode_stats={
            "prefill_rounds": 3,
            "decode_priority_release_rounds": 1,
            "decode_priority_deferred_rounds": 7,
        },
    )
    handler.do_GET()
    assert captured[0]["memory"]["prompt_cache_entries"] == 2
    assert captured[0]["memory"]["prompt_cache_bytes"] == 17
    assert "system_available_bytes" in captured[0]["memory"]
    assert "metal_active_bytes" in captured[0]["memory"]
    assert captured[0]["configured"]["verification_row_cap"] == 24
    assert captured[0]["configured"]["decode_priority_cadence"] == 4
    assert captured[0]["scheduler"]["decode_priority_deferred_rounds"] == 7


def test_lane_abort_counts_only_acknowledged_delivery():
    class ImmediateQueue(Queue):
        def put(self, item, *args, **kwargs):
            response_queue, _request, _generation_args = item
            response_queue.put(SimpleNamespace())
            response_queue.put(Response("one", 1, 0.0, None, ()))
            response_queue.put(Response("two", 2, 0.0, None, ()))
            response_queue.put(None)

    generator = ResponseGenerator.__new__(ResponseGenerator)
    generator._admission = Condition()
    generator._paused = False
    generator._inflight = 0
    generator.admission_timeout = 0.0
    generator.requests = ImmediateQueue()
    generator.batch_metrics = BatchRuntimeMetrics()
    generator.model_provider = SimpleNamespace(
        cli_args=SimpleNamespace(max_inflight_requests=0)
    )
    args = GenerationArguments.__new__(GenerationArguments)
    args.request_id = "faulted"
    args.tenant_id = "tenant"
    args.batch_fault = BatchFaultSpec("lane_abort", 1)

    _ctx, stream = generator.generate(SimpleNamespace(), args)
    assert next(stream).text == "one"
    with pytest.raises(InjectedBatchFault, match="after 1 tokens"):
        next(stream)

    snapshot = generator.batch_metrics.snapshot()
    assert snapshot["counters"]["tokens_delivered"] == 1
    assert snapshot["counters"]["terminal_faulted"] == 1
    assert generator._inflight == 0
