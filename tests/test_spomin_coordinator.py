from dataclasses import replace

import pytest

from mlx_lm.adaptive_work_coordinator import (
    ComputeDomain,
    ComputeTopologySnapshot,
    EstimateProvenance,
    OperationCostBook,
    OperationCostEstimate,
    WorkPlacement,
)
from mlx_lm.cache_planes import TranscriptLedgerPlane, TranscriptLedgerSegment
from mlx_lm.heterogeneous_execution import Engine
from mlx_lm.spomin_coordinator import (
    SegmentScoreResult,
    SpominCoordinator,
    SpominCoordinatorConfig,
    SpominMethod,
    SpominMethodKind,
    SpominPressureSignals,
    SpominSafePoint,
    SpominSafePointError,
)
from mlx_lm.spomin_layer import (
    InMemorySpominBackend,
    SpominConfig,
    SpominTargetState,
    StaticSegmentScorer,
)


def state():
    segments = []
    cursor = 0
    for index in range(4):
        tokens = tuple(range(index * 30, (index + 1) * 30))
        segments.append(
            TranscriptLedgerSegment(
                f"turn:{index + 1}", cursor, cursor + len(tokens), tokens
            )
        )
        cursor += len(tokens)
    transcript = TranscriptLedgerPlane("tokenizer", "v1", "ledger-r1", tuple(segments))
    return SpominTargetState(
        "target-r1",
        120,
        transcript,
        tuple(segment.segment_id for segment in segments),
    )


class FakeAsyncScorer:
    def __init__(self, engine, scores):
        self.engine = engine
        self.scores = scores
        self.submissions = []
        self.abandoned = []

    def submit(self, transcript, stamp):
        ticket = object()
        self.submissions.append((ticket, transcript, stamp))
        return ticket

    def resolve(self, ticket):
        _, _, stamp = self.submissions[-1]
        return SegmentScoreResult(stamp, self.scores, {"source": "fake"})

    def abandon(self, ticket):
        self.abandoned.append(ticket)


def method(method_id, kind, engine, scorer=None, adapter=None, reclaim=60):
    return SpominMethod(
        method_id,
        kind,
        "lowest_importance",
        engine.value,
        engine,
        reclaim,
        scorer=scorer,
        async_adapter=adapter,
    )


def estimate(item, *, service, memory=64, confidence=0.9, risk=0.01):
    return OperationCostEstimate(
        item.operation,
        item.domain_id,
        service,
        memory,
        risk,
        confidence,
        EstimateProvenance.CALIBRATED,
    )


def placement(item, service=1.0):
    return WorkPlacement(
        item.work_id,
        "topology-r1",
        item.domain_id,
        item.engine,
        service,
        1.0,
        0.9,
        EstimateProvenance.CALIBRATED,
    )


def topology(*items, ane_headroom=1024):
    operations = {item.domain_id: set() for item in items}
    engines = {}
    for item in items:
        operations[item.domain_id].add(item.operation)
        engines[item.domain_id] = item.engine
    return ComputeTopologySnapshot(
        "topology-r1",
        1,
        tuple(
            ComputeDomain(
                domain_id,
                engines[domain_id],
                frozenset(domain_operations),
                ane_headroom if engines[domain_id] is Engine.ANE else 1024,
                supports_overlap=engines[domain_id] is Engine.ANE,
            )
            for domain_id, domain_operations in sorted(operations.items())
        ),
    )


def coordinator(costs, *, enabled=True):
    return SpominCoordinator(
        SpominConfig(
            capacity_tokens=100,
            pressure_ratio=0.70,
            target_ratio=0.65,
            protect_recent_segments=1,
        ),
        OperationCostBook(costs),
        config=SpominCoordinatorConfig(enabled=enabled),
    )


def test_default_off_never_submits_ane_work():
    adapter = FakeAsyncScorer(Engine.ANE, {})
    ane = method("ane", SpominMethodKind.ANE_SEMANTIC, Engine.ANE, adapter=adapter)
    control = coordinator((estimate(ane, service=1.0),), enabled=False)
    decision, ticket = control.begin(
        state(),
        SpominPressureSignals(0.9, 0.9, 10.0),
        (ane,),
        topology(ane),
        (placement(ane),),
    )
    assert decision.reason == "disabled"
    assert ticket is None
    assert adapter.submissions == []


def test_batch_pressure_selects_hidden_ane_semantic_scoring():
    scores = {f"turn:{index}": float(index) for index in range(1, 5)}
    ane_adapter = FakeAsyncScorer(Engine.ANE, scores)
    ane = method("ane", SpominMethodKind.ANE_SEMANTIC, Engine.ANE, adapter=ane_adapter)
    cpu = method(
        "cpu",
        SpominMethodKind.CPU_SEMANTIC,
        Engine.CPU,
        scorer=StaticSegmentScorer(scores),
    )
    control = coordinator((estimate(ane, service=8.0), estimate(cpu, service=2.0)))
    decision, ticket = control.begin(
        state(),
        SpominPressureSignals(0.1, 0.9, 8.0),
        (cpu, ane),
        topology(cpu, ane),
        (placement(cpu, 2.0), placement(ane, 8.0)),
    )
    assert decision.method is ane
    assert ticket is not None
    assert len(ane_adapter.submissions) == 1


def test_topology_headroom_forces_cpu_fallback():
    scores = {f"turn:{index}": float(index) for index in range(1, 5)}
    ane_adapter = FakeAsyncScorer(Engine.ANE, scores)
    ane = method("ane", SpominMethodKind.ANE_SEMANTIC, Engine.ANE, adapter=ane_adapter)
    cpu = method(
        "cpu",
        SpominMethodKind.CPU_SEMANTIC,
        Engine.CPU,
        scorer=StaticSegmentScorer(scores),
    )
    control = coordinator(
        (
            estimate(ane, service=1.0, memory=512),
            estimate(cpu, service=3.0),
        )
    )
    decision, _ = control.begin(
        state(),
        SpominPressureSignals(0.1, 0.9, 10.0),
        (ane, cpu),
        topology(ane, cpu, ane_headroom=256),
        (placement(ane), placement(cpu, 3.0)),
    )
    assert decision.method is cpu
    assert ane_adapter.submissions == []


def test_stale_topology_placement_cannot_authorize_scoring():
    scores = {f"turn:{index}": float(index) for index in range(1, 5)}
    adapter = FakeAsyncScorer(Engine.ANE, scores)
    ane = method("ane", SpominMethodKind.ANE_SEMANTIC, Engine.ANE, adapter=adapter)
    stale = replace(placement(ane), topology_revision="topology-old")
    control = coordinator((estimate(ane, service=1.0),))
    decision, ticket = control.begin(
        state(),
        SpominPressureSignals(0.1, 0.9, 1.0),
        (ane,),
        topology(ane),
        (stale,),
    )
    assert decision.reason == "no_eligible_method"
    assert ticket is None
    assert adapter.submissions == []


def test_async_ane_scores_plan_but_safe_point_owns_commit():
    scores = {"turn:1": 0.1, "turn:2": 0.2, "turn:3": 10.0, "turn:4": 20.0}
    adapter = FakeAsyncScorer(Engine.ANE, scores)
    ane = method("ane", SpominMethodKind.ANE_SEMANTIC, Engine.ANE, adapter=adapter)
    control = coordinator((estimate(ane, service=3.0),))
    original = state()
    _, ticket = control.begin(
        original,
        SpominPressureSignals(0.1, 0.9, 3.0),
        (ane,),
        topology(ane),
        (placement(ane, 3.0),),
    )
    plan = control.plan(ticket, original, protected_segment_ids=("turn:3",))
    assert plan.selection.segment_ids == ("turn:1", "turn:2")
    with pytest.raises(SpominSafePointError, match="not quiescent"):
        control.commit(
            original,
            plan,
            InMemorySpominBackend(),
            SpominSafePoint(original.revision, False, True),
        )
    updated = control.commit(
        original,
        plan,
        InMemorySpominBackend(),
        SpominSafePoint(original.revision, True, True),
    )
    assert updated.visible_segment_ids == ("turn:3", "turn:4")


def test_revision_change_discards_late_async_score():
    scores = {f"turn:{index}": float(index) for index in range(1, 5)}
    adapter = FakeAsyncScorer(Engine.ANE, scores)
    ane = method("ane", SpominMethodKind.ANE_SEMANTIC, Engine.ANE, adapter=adapter)
    control = coordinator((estimate(ane, service=1.0),))
    original = state()
    _, ticket = control.begin(
        original,
        SpominPressureSignals(0.1, 0.9, 1.0),
        (ane,),
        topology(ane),
        (placement(ane),),
    )
    assert control.plan(ticket, replace(original, revision="target-r2")) is None
    assert adapter.abandoned == [ticket.adapter_ticket]


def test_scheduler_can_abandon_disposable_async_scoring():
    scores = {f"turn:{index}": float(index) for index in range(1, 5)}
    adapter = FakeAsyncScorer(Engine.ANE, scores)
    ane = method("ane", SpominMethodKind.ANE_SEMANTIC, Engine.ANE, adapter=adapter)
    control = coordinator((estimate(ane, service=1.0),))
    _, ticket = control.begin(
        state(),
        SpominPressureSignals(0.1, 0.9, 1.0),
        (ane,),
        topology(ane),
        (placement(ane),),
    )
    assert control.abandon(ticket)
    assert adapter.abandoned == [ticket.adapter_ticket]


def test_memory_pressure_prefers_more_reclaim_over_faster_scoring():
    scores = {f"turn:{index}": float(index) for index in range(1, 5)}
    small = method(
        "small",
        SpominMethodKind.HEAD_HOTSPOT,
        Engine.CPU,
        scorer=StaticSegmentScorer(scores),
        reclaim=30,
    )
    large = method(
        "large",
        SpominMethodKind.CPU_SEMANTIC,
        Engine.CPU,
        scorer=StaticSegmentScorer(scores),
        reclaim=60,
    )
    control = coordinator((estimate(small, service=1.0), estimate(large, service=4.0)))
    decision, _ = control.begin(
        state(),
        SpominPressureSignals(0.9, 0.1, 0.0),
        (small, large),
        topology(small, large),
        (placement(small), placement(large, 4.0)),
    )
    assert decision.method is large
