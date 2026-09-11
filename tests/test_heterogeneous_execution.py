from dataclasses import replace

import pytest

from mlx_lm.heterogeneous_execution import (
    Engine,
    EngineCapability,
    HeterogeneousExecutionProfile,
    HostRuntimeFingerprint,
    OperationGeometry,
    OperationMeasurement,
    OperationNode,
    PlanningRefused,
    TransferBoundary,
    TransferMeasurement,
)


def fingerprint():
    return HostRuntimeFingerprint(
        host="lab-mac",
        chip="M5 Ultra",
        os_build="25A1",
        model_id="qwen4",
        model_revision="abc123",
        quantization="4bit",
        runtimes=(("e5rt", "1"), ("mlx", "0.31.2")),
    )


def geometry(size=32, *, suffix="v1"):
    return OperationGeometry(
        input_bytes=size,
        output_bytes=size * 2,
        input_shape=(1, size),
        output_shape=(1, size * 2),
        dtype="bfloat16",
        layout="row-major",
        operation_fingerprint=f"op-{suffix}",
    )


def measurement(fp, operation, engine, total, *, geom=None):
    return OperationMeasurement(
        operation=operation,
        engine=engine,
        geometry=geom or geometry(),
        fingerprint_digest=fp.digest,
        dispatch_us=total * 0.1,
        copy_in_us=total * 0.2,
        compute_us=total * 0.4,
        sync_us=total * 0.2,
        copy_out_us=total * 0.1,
        owner_lifetime_us=total + 1,
    )


def profile(*operations, overlap=False):
    fp = fingerprint()
    capabilities = {
        engine: EngineCapability(
            engine,
            frozenset(operations),
            4096,
            supports_overlap=overlap,
        )
        for engine in (Engine.CPU, Engine.ANE, Engine.GPU)
    }
    return HeterogeneousExecutionProfile(fp, capabilities)


def test_fingerprint_includes_model_and_runtime_identity():
    fp = fingerprint()
    assert fp.digest != replace(fp, model_revision="def456").digest
    assert fp.digest != replace(
        fp, runtimes=(("e5rt", "2"), ("mlx", "0.31.2"))
    ).digest
    with pytest.raises(ValueError, match="sorted"):
        replace(fp, runtimes=(("mlx", "1"), ("e5rt", "1")))


def test_geometry_and_measurements_reject_invalid_numbers():
    with pytest.raises(ValueError, match="input bytes"):
        replace(geometry(), input_bytes=1.5)
    with pytest.raises(ValueError, match="shape dimensions"):
        replace(geometry(), input_shape=(1, 0))
    sample = measurement(fingerprint(), "copy", Engine.ANE, 10)
    assert sample.total_us == pytest.approx(10)
    for value in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError, match="finite"):
            replace(sample, compute_us=value)
    with pytest.raises(ValueError, match="lifetime"):
        replace(sample, owner_lifetime_us=9)


def test_capability_map_and_limits_are_validated():
    fp = fingerprint()
    ane = EngineCapability(Engine.ANE, frozenset({"copy"}), 64)
    with pytest.raises(ValueError, match="does not match"):
        HeterogeneousExecutionProfile(fp, {Engine.CPU: ane})
    capabilities = {Engine.ANE: ane}
    profile_snapshot = HeterogeneousExecutionProfile(fp, capabilities)
    capabilities[Engine.CPU] = EngineCapability(
        Engine.CPU, frozenset({"copy"}), 64
    )
    assert Engine.CPU not in profile_snapshot.capabilities
    for value in (0, -1, 1.5):
        with pytest.raises(ValueError, match="max bytes"):
            EngineCapability(Engine.ANE, frozenset({"copy"}), value)


def test_transfer_measurement_rejects_nonfinite_costs_and_invalid_credit():
    sample = TransferMeasurement(
        Engine.CPU,
        Engine.ANE,
        geometry(suffix="handoff"),
        fingerprint().digest,
        copy_us=5,
        sync_us=2,
        overlap_credit_us=1,
    )
    with pytest.raises(ValueError, match="finite"):
        replace(sample, sync_us=float("nan"))
    with pytest.raises(ValueError, match="exceeds"):
        replace(sample, overlap_credit_us=8)


def test_measurements_never_cross_fingerprints_or_capabilities():
    p = profile("cache_copy")
    wrong = measurement(
        replace(p.fingerprint, model_revision="other"),
        "cache_copy",
        Engine.ANE,
        10,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        p.record(wrong)
    with pytest.raises(ValueError, match="capability"):
        p.record(measurement(p.fingerprint, "route", Engine.ANE, 10))


def test_exact_geometry_prevents_small_sample_from_authorizing_large_work():
    p = profile("copy")
    small = geometry(8, suffix="copy")
    large = geometry(512, suffix="copy")
    p.record(measurement(p.fingerprint, "copy", Engine.CPU, 1, geom=small))
    with pytest.raises(PlanningRefused, match="exact-geometry"):
        p.plan((OperationNode("large", "copy", large, "copy"),))


def test_exact_geometry_preserves_engine_size_crossover():
    p = profile("copy")
    small = geometry(8, suffix="copy")
    large = geometry(512, suffix="copy")
    for geom, cpu_us, ane_us in ((small, 1, 5), (large, 9, 3)):
        p.record(measurement(p.fingerprint, "copy", Engine.CPU, cpu_us, geom=geom))
        p.record(measurement(p.fingerprint, "copy", Engine.ANE, ane_us, geom=geom))
    small_plan = p.plan((OperationNode("small", "copy", small, "copy"),))
    large_plan = p.plan((OperationNode("large", "copy", large, "copy"),))
    assert small_plan.placements[0].engine == Engine.CPU
    assert large_plan.placements[0].engine == Engine.ANE


def test_whole_island_uses_one_engine_instead_of_per_layer_ping_pong():
    p = profile("layer_a", "layer_b")
    for operation, engine, cost in (
        ("layer_a", Engine.CPU, 1),
        ("layer_a", Engine.ANE, 9),
        ("layer_b", Engine.CPU, 9),
        ("layer_b", Engine.ANE, 1),
    ):
        p.record(measurement(p.fingerprint, operation, engine, cost))
    plan = p.plan(
        (
            OperationNode("l0", "layer_a", geometry(), "decode-stack"),
            OperationNode(
                "l1", "layer_b", geometry(), "decode-stack", ("l0",)
            ),
        )
    )
    assert len({placement.engine for placement in plan.placements}) == 1
    assert plan.transfers == ()


def test_deadline_is_checked_against_dependency_critical_path():
    p = profile("prepare", "consume")
    for operation, engine, cost in (
        ("prepare", Engine.CPU, 1),
        ("consume", Engine.CPU, 3),
        ("prepare", Engine.ANE, 1),
        ("consume", Engine.ANE, 1),
    ):
        p.record(measurement(p.fingerprint, operation, engine, cost))
    plan = p.plan(
        (
            OperationNode("p", "prepare", geometry(), "capsule"),
            OperationNode(
                "c",
                "consume",
                geometry(),
                "capsule",
                ("p",),
                deadline_us=3,
            ),
        )
    )
    assert {placement.engine for placement in plan.placements} == {Engine.ANE}
    assert plan.critical_path_us == pytest.approx(2)
    assert plan.placements[1].ready_us == pytest.approx(1)
    assert plan.placements[1].completion_us == pytest.approx(2)


def test_cross_engine_edge_accounts_for_measured_copy_and_sync():
    p = HeterogeneousExecutionProfile(
        fingerprint(),
        {
            Engine.CPU: EngineCapability(
                Engine.CPU, frozenset({"prepare"}), 4096
            ),
            Engine.ANE: EngineCapability(
                Engine.ANE, frozenset({"consume"}), 4096
            ),
        },
    )
    p.record(measurement(p.fingerprint, "prepare", Engine.CPU, 1))
    p.record(measurement(p.fingerprint, "consume", Engine.ANE, 2))
    transfer_geometry = geometry(32, suffix="handoff")
    p.record_transfer(
        TransferMeasurement(
            Engine.CPU,
            Engine.ANE,
            transfer_geometry,
            p.fingerprint.digest,
            copy_us=5,
            sync_us=2,
            overlap_credit_us=5,
        )
    )
    plan = p.plan(
        (
            OperationNode("p", "prepare", geometry(), "host"),
            OperationNode(
                "c",
                "consume",
                geometry(),
                "device",
                ("p",),
                transfer_boundaries=(TransferBoundary("p", transfer_geometry),),
            ),
        )
    )
    assert plan.transfers[0].projected_us == pytest.approx(7)
    assert plan.placements[1].ready_us == pytest.approx(8)
    assert plan.critical_path_us == pytest.approx(10)


def test_overlap_credit_needs_support_on_both_engines():
    p = profile("work", overlap=True)
    transfer_geometry = geometry(32, suffix="handoff")
    sample = TransferMeasurement(
        Engine.CPU,
        Engine.ANE,
        transfer_geometry,
        p.fingerprint.digest,
        copy_us=5,
        sync_us=2,
        overlap_credit_us=5,
    )
    p.record_transfer(sample)
    assert p.projected_transfer_us(
        Engine.CPU, Engine.ANE, transfer_geometry
    ) == pytest.approx(2)


def test_cross_engine_edges_need_edge_specific_evidence():
    p = HeterogeneousExecutionProfile(
        fingerprint(),
        {
            Engine.CPU: EngineCapability(
                Engine.CPU, frozenset({"prepare"}), 4096
            ),
            Engine.ANE: EngineCapability(
                Engine.ANE, frozenset({"consume"}), 4096
            ),
        },
    )
    p.record(measurement(p.fingerprint, "prepare", Engine.CPU, 1))
    p.record(measurement(p.fingerprint, "consume", Engine.ANE, 1))
    with pytest.raises(PlanningRefused, match="transfer evidence"):
        p.plan(
            (
                OperationNode("p", "prepare", geometry(), "host"),
                OperationNode(
                    "c", "consume", geometry(), "device", ("p",)
                ),
            )
        )


def test_transfer_boundary_cannot_enable_per_layer_ping_pong():
    p = HeterogeneousExecutionProfile(
        fingerprint(),
        {
            Engine.CPU: EngineCapability(
                Engine.CPU, frozenset({"layer_a"}), 4096
            ),
            Engine.ANE: EngineCapability(
                Engine.ANE, frozenset({"layer_b"}), 4096
            ),
        },
    )
    p.record(measurement(p.fingerprint, "layer_a", Engine.CPU, 1))
    p.record(measurement(p.fingerprint, "layer_b", Engine.ANE, 1))
    handoff = geometry(32, suffix="handoff")
    p.record_transfer(
        TransferMeasurement(
            Engine.CPU,
            Engine.ANE,
            handoff,
            p.fingerprint.digest,
            copy_us=1,
            sync_us=1,
        )
    )
    with pytest.raises(PlanningRefused, match="transfer evidence"):
        p.plan(
            (
                OperationNode(
                    "l0", "layer_a", geometry(), "zero", model_layer=0
                ),
                OperationNode(
                    "l1",
                    "layer_b",
                    geometry(),
                    "one",
                    ("l0",),
                    transfer_boundaries=(TransferBoundary("l0", handoff),),
                    model_layer=1,
                ),
            )
        )


def test_unmeasured_and_cyclic_graphs_fail_closed():
    p = profile("work")
    with pytest.raises(PlanningRefused, match="empty"):
        p.plan(())
    with pytest.raises(PlanningRefused, match="exact-geometry"):
        p.plan((OperationNode("a", "work", geometry(), "island"),))
    with pytest.raises(PlanningRefused, match="cycle"):
        p.plan(
            (
                OperationNode("a", "work", geometry(), "island", ("b",)),
                OperationNode("b", "work", geometry(), "island", ("a",)),
            )
        )
