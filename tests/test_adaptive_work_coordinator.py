import pytest

from mlx_lm.adaptive_work_coordinator import (
    AdaptiveCoordinatorConfig,
    AdaptiveWorkCoordinator,
    AsyncWorkItem,
    ComputeDomain,
    ComputeTopologySnapshot,
    EstimateProvenance,
    OperationCostBook,
    OperationCostEstimate,
    OperationObservation,
    SchedulerBoundary,
    ServingKnobBounds,
    ServingKnobController,
    ServingKnobOption,
    ServingKnobState,
)
from mlx_lm.heterogeneous_execution import (
    Engine,
    EngineCapability,
    OperationGeometry,
    OperationMeasurement,
)


def estimate(operation, domain, service, *, memory=64, confidence=0.8):
    return OperationCostEstimate(
        operation,
        domain,
        service,
        memory,
        0.0,
        confidence,
        EstimateProvenance.CALIBRATED,
    )


def topology(*domains):
    return ComputeTopologySnapshot("topology-r1", 1, tuple(domains))


def test_topology_first_placement_uses_cost_and_headroom():
    operation = "semantic-score"
    costs = OperationCostBook(
        (
            estimate(operation, "cpu", 8.0),
            estimate(operation, "ane", 2.0, memory=512),
        )
    )
    coordinator = AdaptiveWorkCoordinator(
        costs, config=AdaptiveCoordinatorConfig(enabled=True)
    )
    placement = coordinator.allocate(
        topology(
            ComputeDomain("cpu", Engine.CPU, frozenset({operation}), 1024),
            ComputeDomain(
                "ane", Engine.ANE, frozenset({operation}), 256, supports_overlap=True
            ),
        ),
        (AsyncWorkItem("work", operation, "session"),),
    )
    assert [(item.domain_id, item.engine) for item in placement] == [
        ("cpu", Engine.CPU)
    ]


def test_deficit_allocation_is_fair_across_calls():
    operation = "shared-operation"
    costs = OperationCostBook((estimate(operation, "cpu", 1.0),))
    coordinator = AdaptiveWorkCoordinator(
        costs, config=AdaptiveCoordinatorConfig(enabled=True)
    )
    host = topology(ComputeDomain("cpu", Engine.CPU, frozenset({operation}), 1024))
    pending = (
        AsyncWorkItem("a-work", operation, "owner-a"),
        AsyncWorkItem("b-work", operation, "owner-b"),
    )
    first = coordinator.allocate(host, pending, max_placements=1)
    second = coordinator.allocate(host, pending, max_placements=1)
    assert first[0].work_id == "a-work"
    assert second[0].work_id == "b-work"


def test_aging_prevents_an_old_request_from_losing_a_tie():
    operation = "shared-operation"
    coordinator = AdaptiveWorkCoordinator(
        OperationCostBook((estimate(operation, "cpu", 1.0),)),
        config=AdaptiveCoordinatorConfig(enabled=True, aging_credit_per_tick=0.5),
    )
    placement = coordinator.allocate(
        topology(ComputeDomain("cpu", Engine.CPU, frozenset({operation}), 1024)),
        (
            AsyncWorkItem("new", operation, "new-owner"),
            AsyncWorkItem("old", operation, "old-owner", queued_ticks=10),
        ),
        max_placements=1,
    )
    assert placement[0].work_id == "old"


def test_observations_update_cost_and_confidence_without_device_probing():
    book = OperationCostBook(confidence_step=0.25)
    first = book.observe(OperationObservation("op", "ane", 4.0, 100))
    second = book.observe(OperationObservation("op", "ane", 2.0, 200))
    assert first.confidence == 0.25
    assert second.service_ms == 3.0
    assert second.working_set_bytes == 150
    assert second.confidence == 0.5
    assert second.sample_count == 2


def test_existing_heterogeneous_evidence_adapts_into_live_topology_costs():
    geometry = OperationGeometry(100, 80, (25,), (20,), "f32", "row", "op-r1")
    sample = OperationMeasurement(
        "score",
        Engine.ANE,
        geometry,
        "host-r1",
        100,
        200,
        300,
        100,
        100,
        800,
    )
    observation = OperationObservation.from_measurement(sample, domain_id="ane0")
    capability = EngineCapability(
        Engine.ANE, frozenset({"score"}), 1024, supports_overlap=True
    )
    domain = ComputeDomain.from_engine_capability(
        "ane0", capability, memory_headroom_bytes=512
    )
    assert observation.service_ms == 0.8
    assert observation.working_set_bytes == 100
    assert domain.engine is Engine.ANE
    assert domain.supports_overlap


def knob_option(option_id, state, *, memory_delta, latency_delta, throughput):
    return ServingKnobOption(
        option_id,
        state,
        memory_delta,
        latency_delta,
        throughput,
        0.0,
        0.9,
        EstimateProvenance.MEASURED,
    )


def test_knob_controller_changes_batch_concurrency_and_mtp_only_at_safe_boundaries():
    current = ServingKnobState(16, 16, 4)
    reduced = knob_option(
        "reduce-pressure",
        ServingKnobState(8, 8, 2),
        memory_delta=-(4 * 1024**3),
        latency_delta=-1.0,
        throughput=-0.5,
    )
    controller = ServingKnobController(
        ServingKnobBounds(1, 32, 1, 32, 1, 8), enabled=True, cooldown_ticks=2
    )
    unsafe = controller.decide(
        current,
        (reduced,),
        SchedulerBoundary(10, False, True),
        memory_pressure=1.0,
        latency_pressure=0.0,
        batch_pressure=0.0,
    )
    assert unsafe.reason == "hysteresis"
    safe = controller.decide(
        current,
        (reduced,),
        SchedulerBoundary(11, True, True),
        memory_pressure=1.0,
        latency_pressure=0.0,
        batch_pressure=0.0,
    )
    assert safe.option_id == "reduce-pressure"
    assert safe.state == ServingKnobState(8, 8, 2)
    cooldown = controller.decide(
        safe.state,
        (reduced,),
        SchedulerBoundary(12, True, True),
        memory_pressure=1.0,
        latency_pressure=0.0,
        batch_pressure=0.0,
    )
    assert cooldown.reason == "cooldown"


def test_knob_controller_can_expand_throughput_when_batch_pressure_dominates():
    current = ServingKnobState(4, 4, 2)
    expanded = knob_option(
        "expand",
        ServingKnobState(8, 8, 4),
        memory_delta=512 * 1024**2,
        latency_delta=0.1,
        throughput=2.0,
    )
    controller = ServingKnobController(
        ServingKnobBounds(1, 16, 1, 16, 1, 8), enabled=True, cooldown_ticks=0
    )
    decision = controller.decide(
        current,
        (expanded,),
        SchedulerBoundary(1, True, True),
        memory_pressure=0.0,
        latency_pressure=0.0,
        batch_pressure=1.0,
    )
    assert decision.state == expanded.state


def test_invalid_cost_observation_is_rejected():
    with pytest.raises(ValueError, match="service time"):
        OperationObservation("op", "cpu", float("nan"), 0)
