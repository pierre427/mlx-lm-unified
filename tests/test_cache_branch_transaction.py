from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass, replace
from threading import Barrier, Event

import pytest

from mlx_lm.cache_branch_transaction import (
    BranchStatus,
    CacheBranchTransactionError,
    PlaneBase,
    PlaneDelta,
    create_cache_delta_lineage,
)
from mlx_lm.cache_planes import CachePlaneKind


PLANES = (
    CachePlaneKind.ATTENTION_KV,
    CachePlaneKind.ATTENTION_RING,
    CachePlaneKind.QSA_SUMMARY,
    CachePlaneKind.GDN_RECURRENT,
    CachePlaneKind.MTP_DRAFT,
)


def _layout(kind):
    return f"{kind.value}-v1"


def _bases(position=6, generation=0):
    return tuple(
        PlaneBase(
            kind,
            object(),
            0,
            position,
            _layout(kind),
            generation,
            logical_bytes=100 + index,
            payload_is_immutable=True,
            ring_capacity=4 if kind == CachePlaneKind.ATTENTION_RING else None,
        )
        for index, kind in enumerate(PLANES)
    )


def _deltas(start, length=1, generation=0, byte_scale=10):
    result = {}
    for index, kind in enumerate(PLANES):
        ring = kind == CachePlaneKind.ATTENTION_RING
        result[kind] = PlaneDelta(
            kind,
            f"{kind.value}@{start + length}",
            start,
            length,
            _layout(kind),
            generation,
            logical_bytes=byte_scale + index,
            payload_is_immutable=True,
            ring_capacity=4 if ring else None,
            physical_start=start % 4 if ring else None,
            physical_stop=(start + length) % 4 if ring else None,
            wrap_epoch=(start + length) // 4 if ring else None,
        )
    return result


def _lineage(position=6, generation=0):
    lineage = create_cache_delta_lineage(
        _bases(position, generation), enabled=True
    )
    assert lineage is not None
    return lineage


def _replacement(base, deltas, generation, position):
    return PlaneBase(
        base.kind,
        object(),
        0,
        position,
        base.layout_id,
        generation,
        logical_bytes=base.logical_bytes
        + sum(delta.logical_bytes for delta in deltas),
        payload_is_immutable=True,
        ring_capacity=base.ring_capacity,
    )


class _Compactor:
    source_payloads_read_only = True
    destinations_are_fresh = True

    def __init__(self, callback):
        self._callback = callback

    def materialize(self, base, deltas):
        return self._callback(base, deltas)


def test_delta_promotion_gate_is_default_off(monkeypatch):
    monkeypatch.delenv("MLX_LM_CACHE_DELTA_PROMOTION", raising=False)
    assert create_cache_delta_lineage(_bases()) is None


def test_payloads_require_explicit_immutable_source_contract():
    with pytest.raises(ValueError, match="declared immutable"):
        PlaneBase(
            CachePlaneKind.ATTENTION_KV,
            object(),
            0,
            1,
            "kv",
            0,
        )
    with pytest.raises(ValueError, match="declared immutable"):
        PlaneDelta(
            CachePlaneKind.ATTENTION_KV,
            object(),
            1,
            1,
            "kv",
            0,
        )


def test_ring_kind_requires_capacity_and_other_planes_forbid_it():
    with pytest.raises(ValueError, match="requires ring_capacity"):
        PlaneBase(
            CachePlaneKind.ATTENTION_RING,
            object(),
            0,
            1,
            "ring",
            0,
            payload_is_immutable=True,
        )
    with pytest.raises(ValueError, match="only valid"):
        PlaneBase(
            CachePlaneKind.ATTENTION_KV,
            object(),
            0,
            1,
            "kv",
            0,
            payload_is_immutable=True,
            ring_capacity=4,
        )


def test_fork_shares_every_immutable_base_without_payload_copy():
    lineage = _lineage()
    left = lineage.fork(owner_id="left")
    right = lineage.fork(owner_id="right")

    assert left.position == right.position == 6
    assert lineage.stats()["forks"] == 2
    assert lineage.stats()["promotion_copied_bytes"] == 0


def test_fork_view_retains_exact_original_base_payload_identities():
    bases = _bases()
    lineage = create_cache_delta_lineage(bases, enabled=True)
    lineage.fork(owner_id="request")
    view = lineage.current_view()
    expected = {base.kind: base.payload for base in bases}
    assert all(base.payload is expected[base.kind] for base in view.bases)


@pytest.mark.parametrize("start,length", [(6, 1), (7, 1), (7, 3), (11, 2)])
def test_aligned_ring_boundaries_cover_wrap_and_offsets(start, length):
    ring = _deltas(start, length)[CachePlaneKind.ATTENTION_RING]
    assert ring.physical_start == start % 4
    assert ring.physical_stop == (start + length) % 4
    assert ring.wrap_epoch == (start + length) // 4


def test_invalid_ring_coordinates_fail_before_branch_mutation():
    with pytest.raises(ValueError, match="physical_stop"):
        PlaneDelta(
            CachePlaneKind.ATTENTION_RING,
            "bad",
            7,
            2,
            _layout(CachePlaneKind.ATTENTION_RING),
            0,
            payload_is_immutable=True,
            ring_capacity=4,
            physical_start=3,
            physical_stop=0,
            wrap_epoch=2,
        )


def test_append_requires_every_plane_at_one_exact_boundary():
    lineage = _lineage()
    branch = lineage.fork(owner_id="request")
    missing = _deltas(6)
    missing.pop(CachePlaneKind.MTP_DRAFT)
    with pytest.raises(CacheBranchTransactionError, match="every lineage plane"):
        branch.append_checkpoint(missing)
    assert branch.delta_depth == 0

    misaligned = _deltas(6)
    delta = misaligned[CachePlaneKind.GDN_RECURRENT]
    misaligned[delta.kind] = PlaneDelta(
        delta.kind,
        delta.payload,
        7,
        1,
        delta.layout_id,
        0,
        logical_bytes=delta.logical_bytes,
        payload_is_immutable=True,
    )
    with pytest.raises(CacheBranchTransactionError, match="contiguous boundary"):
        branch.append_checkpoint(misaligned)
    assert branch.delta_depth == 0


def test_layout_change_is_rejected_atomically():
    lineage = _lineage()
    branch = lineage.fork(owner_id="request")
    changed = _deltas(6)
    delta = changed[CachePlaneKind.QSA_SUMMARY]
    changed[delta.kind] = PlaneDelta(
        delta.kind,
        delta.payload,
        delta.start,
        delta.length,
        "qsa-v2",
        delta.generation,
        logical_bytes=delta.logical_bytes,
        payload_is_immutable=True,
    )
    with pytest.raises(CacheBranchTransactionError, match="layout changed"):
        branch.append_checkpoint(changed)
    assert branch.delta_depth == 0


def test_full_acceptance_is_one_pointer_swap_with_zero_copy():
    lineage = _lineage()
    branch = lineage.fork(owner_id="winner")
    checkpoint = branch.append_checkpoint(_deltas(6, 2))
    receipt = branch.promote(8)

    assert receipt.pointer_swaps == 1
    assert receipt.copied_bytes == 0
    assert receipt.accepted_delta_nodes == 1
    assert receipt.abandoned_delta_nodes == 0
    assert lineage.current_view().tip is checkpoint
    assert lineage.position == 8
    assert lineage.generation == 1
    assert branch.status == BranchStatus.PROMOTED


def test_zero_acceptance_must_use_rejection_without_fake_pointer_swap():
    lineage = _lineage()
    branch = lineage.fork(owner_id="zero")
    branch.append_checkpoint(_deltas(6))
    with pytest.raises(CacheBranchTransactionError, match="use cheap branch rejection"):
        branch.promote(6)
    receipt = branch.reject()
    assert receipt.pointer_swaps == 0
    assert lineage.generation == 0
    assert lineage.stats()["promotion_pointer_swaps"] == 0


def test_partial_acceptance_selects_checkpoint_and_abandons_suffix_without_copy():
    lineage = _lineage()
    branch = lineage.fork(owner_id="winner")
    accepted = branch.append_checkpoint(_deltas(6, byte_scale=10))
    branch.append_checkpoint(_deltas(7, byte_scale=20))
    branch.append_checkpoint(_deltas(8, byte_scale=30))
    receipt = branch.promote(7)

    assert lineage.current_view().tip is accepted
    assert receipt.accepted_delta_nodes == 1
    assert receipt.abandoned_delta_nodes == 2
    assert receipt.abandoned_bytes == sum(range(20, 25)) + sum(range(30, 35))
    assert receipt.copied_bytes == 0


def test_rejection_is_constant_receipt_and_does_not_move_lineage():
    lineage = _lineage()
    branch = lineage.fork(owner_id="loser")
    branch.append_checkpoint(_deltas(6))
    before = lineage.current_view().tip
    receipt = branch.reject()

    assert branch.status == BranchStatus.REJECTED
    assert lineage.current_view().tip is before
    assert receipt.pointer_swaps == 0
    assert receipt.copied_bytes == 0
    assert receipt.abandoned_delta_nodes == 1


def test_rejected_arena_bytes_trigger_off_path_compaction_cleanup():
    lineage = _lineage()
    with lineage.fork(owner_id="loser") as branch:
        branch.append_checkpoint(_deltas(6))
    stats = lineage.stats()
    assert stats["rejections"] == 1
    assert stats["arena_delta_bytes"] == sum(range(10, 15))
    assert lineage.needs_compaction(
        max_delta_depth=100, max_delta_bytes=sum(range(10, 15))
    )

    lineage.compact(
        _Compactor(lambda base, deltas: _replacement(base, deltas, 1, 6))
    )
    assert lineage.stats()["arena_delta_bytes"] == 0
    assert lineage.stats()["arena_nodes"] == 1


def test_losing_branch_can_be_rejected_after_sibling_promotion():
    lineage = _lineage()
    winner = lineage.fork(owner_id="winner")
    loser = lineage.fork(owner_id="loser")
    winner.append_checkpoint(_deltas(6))
    loser.append_checkpoint(_deltas(6))
    winner.promote(7)

    receipt = loser.reject()
    assert receipt.abandoned_delta_nodes == 1
    assert loser.status == BranchStatus.REJECTED


def test_concurrent_close_is_idempotent_and_counts_one_rejection():
    lineage = _lineage()
    branch = lineage.fork(owner_id="close-race")
    branch.append_checkpoint(_deltas(6))
    barrier = Barrier(2)

    def close():
        barrier.wait()
        return branch.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: close(), range(2)))
    assert sum(outcome is not None for outcome in outcomes) == 1
    assert sum(outcome is None for outcome in outcomes) == 1
    assert lineage.stats()["rejections"] == 1


def test_concurrent_promotion_has_exactly_one_generation_cas_winner():
    lineage = _lineage()
    branches = [lineage.fork(owner_id=name) for name in ("left", "right")]
    for branch in branches:
        branch.append_checkpoint(_deltas(6))
    barrier = Barrier(2)

    def promote(branch):
        barrier.wait()
        try:
            return branch.promote(7)
        except CacheBranchTransactionError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(promote, branches))
    assert sum(not isinstance(value, Exception) for value in outcomes) == 1
    assert sum(isinstance(value, Exception) for value in outcomes) == 1
    assert lineage.generation == 1


def test_two_phase_compaction_materializes_outside_lineage_lock():
    lineage = _lineage()
    branch = lineage.fork(owner_id="winner")
    branch.append_checkpoint(_deltas(6))
    branch.append_checkpoint(_deltas(7))
    branch.promote(8)
    entered = Event()
    release = Event()

    def materialize(base, deltas):
        if base.kind == CachePlaneKind.ATTENTION_KV:
            entered.set()
            assert release.wait(timeout=2)
        return _replacement(base, deltas, 2, 8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        compacting = executor.submit(lineage.compact, _Compactor(materialize))
        assert entered.wait(timeout=2)
        forked = executor.submit(lineage.fork, owner_id="during-compaction")
        stale_branch = forked.result(timeout=2)
        release.set()
        receipt = compacting.result(timeout=2)

    assert receipt.delta_nodes_compacted == 2
    assert receipt.source_generation == 1
    assert receipt.successor_generation == 2
    assert lineage.stats()["delta_depth"] == 0
    assert lineage.stats()["arena_nodes"] == 1
    assert stale_branch.status == BranchStatus.REJECTED
    with pytest.raises(CacheBranchTransactionError, match="already rejected"):
        stale_branch.append_checkpoint(_deltas(8, generation=1))
    assert stale_branch.close() is None
    assert lineage.stats()["compaction_retired_empty_branches"] == 1


def test_compaction_plan_is_authentic_single_use_and_forgery_does_not_consume_it():
    lineage = _lineage()
    branch = lineage.fork(owner_id="winner")
    branch.append_checkpoint(_deltas(6))
    branch.promote(7)
    plan = lineage.prepare_compaction()
    replacements = {
        base.kind: _replacement(base, dict(plan.plane_deltas)[base.kind], 2, 7)
        for base in plan.bases
    }

    forged = replace(plan, position=700)
    with pytest.raises(CacheBranchTransactionError, match="not issued"):
        lineage.commit_compaction(forged, replacements)
    assert lineage.stats()["active_compaction_plans"] == 1

    receipt = lineage.commit_compaction(plan, replacements)
    assert receipt.position == 7
    with pytest.raises(CacheBranchTransactionError, match="already consumed"):
        lineage.commit_compaction(plan, replacements)


def test_cancel_compaction_consumes_only_the_exact_issued_plan():
    lineage = _lineage()
    plan = lineage.prepare_compaction()
    assert lineage.cancel_compaction(replace(plan, position=99)) is False
    assert lineage.stats()["active_compaction_plans"] == 1
    assert lineage.cancel_compaction(plan) is True
    assert lineage.cancel_compaction(plan) is False
    assert lineage.stats()["active_compaction_plans"] == 0
    assert lineage.stats()["compaction_cancellations"] == 1


def test_only_one_compaction_plan_can_remain_outstanding():
    lineage = _lineage()
    plan = lineage.prepare_compaction()
    with pytest.raises(CacheBranchTransactionError, match="already outstanding"):
        lineage.prepare_compaction()
    assert lineage.stats()["active_compaction_plans"] == 1
    assert lineage.cancel_compaction(plan)
    replacement = lineage.prepare_compaction()
    assert replacement.plan_id != plan.plan_id
    assert lineage.cancel_compaction(replacement)


def test_append_during_materialization_invalidates_arena_cas_without_cleanup():
    lineage = _lineage()
    accepted = lineage.fork(owner_id="accepted")
    accepted.append_checkpoint(_deltas(6))
    accepted.promote(7)
    entered = Event()
    release = Event()

    def materialize(base, deltas):
        if base.kind == CachePlaneKind.ATTENTION_KV:
            entered.set()
            assert release.wait(timeout=2)
        return _replacement(base, deltas, 2, 7)

    with ThreadPoolExecutor(max_workers=2) as executor:
        compacting = executor.submit(lineage.compact, _Compactor(materialize))
        assert entered.wait(timeout=2)
        live = lineage.fork(owner_id="concurrent-writer")
        live.append_checkpoint(_deltas(7, generation=1))
        release.set()
        with pytest.raises(CacheBranchTransactionError, match="stale"):
            compacting.result(timeout=2)

    assert lineage.generation == 1
    assert lineage.position == 7
    assert live.status == BranchStatus.ACTIVE
    assert lineage.stats()["arena_nodes"] >= 3
    assert lineage.stats()["active_compaction_plans"] == 0
    assert live.reject().abandoned_delta_nodes == 1


def test_prepare_compaction_refuses_existing_active_delta_branch():
    lineage = _lineage()
    branch = lineage.fork(owner_id="writer")
    branch.append_checkpoint(_deltas(6))
    with pytest.raises(CacheBranchTransactionError, match="active branches"):
        lineage.prepare_compaction()
    branch.reject()


def test_stale_compaction_plan_cannot_overwrite_new_promotion():
    lineage = _lineage()
    first = lineage.fork(owner_id="first")
    first.append_checkpoint(_deltas(6))
    first.promote(7)
    plan = lineage.prepare_compaction()
    second = lineage.fork(owner_id="second")
    second.append_checkpoint(_deltas(7, generation=1))
    second.promote(8)
    replacements = {
        base.kind: _replacement(base, dict(plan.plane_deltas)[base.kind], 2, 7)
        for base in plan.bases
    }

    with pytest.raises(CacheBranchTransactionError, match="stale"):
        lineage.commit_compaction(plan, replacements)
    assert lineage.position == 8


def test_partial_compaction_failure_does_not_change_tip_or_generation():
    lineage = _lineage()
    branch = lineage.fork(owner_id="winner")
    branch.append_checkpoint(_deltas(6))
    branch.promote(7)
    before = lineage.current_view()

    def materialize(base, deltas):
        if base.kind == CachePlaneKind.GDN_RECURRENT:
            raise RuntimeError("GDN compaction failed")
        return _replacement(base, deltas, 2, 7)

    with pytest.raises(RuntimeError, match="GDN compaction failed"):
        lineage.compact(_Compactor(materialize))
    after = lineage.current_view()
    assert after.tip is before.tip
    assert after.generation == before.generation
    assert lineage.stats()["compaction_failures"] == 1


def test_unattested_compactor_is_refused_before_it_sees_source_payloads():
    lineage = _lineage()
    called = False

    def unsafe(base, deltas):
        nonlocal called
        called = True
        return base

    with pytest.raises(CacheBranchTransactionError, match="must attest"):
        lineage.compact(unsafe)
    assert called is False
    assert lineage.stats()["compaction_plans"] == 0


def test_compaction_rejects_replacement_payload_aliasing_any_source():
    lineage = _lineage()
    branch = lineage.fork(owner_id="winner")
    branch.append_checkpoint(_deltas(6))
    branch.promote(7)
    before = lineage.current_view()

    def aliases_source(base, deltas):
        return PlaneBase(
            base.kind,
            base.payload,
            0,
            7,
            base.layout_id,
            2,
            payload_is_immutable=True,
            ring_capacity=base.ring_capacity,
        )

    with pytest.raises(CacheBranchTransactionError, match="aliases source"):
        lineage.compact(_Compactor(aliases_source))
    after = lineage.current_view()
    assert after.tip is before.tip
    assert after.generation == before.generation


@dataclass(frozen=True)
class _PayloadWrapper:
    child: object


class _DictPayloadWrapper:
    def __init__(self, child):
        self.child = child


class _PrivateSlotPayloadWrapper:
    __slots__ = ("__child",)

    def __init__(self, child):
        self.__child = child


def test_compaction_rejects_source_alias_hidden_in_dataclass_wrapper():
    bases = _bases()
    lineage = create_cache_delta_lineage(bases, enabled=True)
    assert lineage is not None

    def aliases_nested_source(base, deltas):
        return PlaneBase(
            base.kind,
            _PayloadWrapper(base.payload),
            0,
            6,
            base.layout_id,
            1,
            payload_is_immutable=True,
            ring_capacity=base.ring_capacity,
        )

    with pytest.raises(CacheBranchTransactionError, match="aliases source"):
        lineage.compact(_Compactor(aliases_nested_source))
    assert lineage.generation == 0


@pytest.mark.parametrize(
    "wrapper",
    [
        pytest.param(_DictPayloadWrapper, id="instance-dict"),
        pytest.param(_PrivateSlotPayloadWrapper, id="private-slot"),
        pytest.param(lambda child: deque([child]), id="deque"),
    ],
)
def test_compaction_rejects_source_alias_in_common_python_graphs(wrapper):
    bases = _bases()
    lineage = create_cache_delta_lineage(bases, enabled=True)
    assert lineage is not None

    def aliases_nested_source(base, deltas):
        return PlaneBase(
            base.kind,
            wrapper(base.payload),
            0,
            6,
            base.layout_id,
            1,
            payload_is_immutable=True,
            ring_capacity=base.ring_capacity,
        )

    with pytest.raises(CacheBranchTransactionError, match="aliases source"):
        lineage.compact(_Compactor(aliases_nested_source))
    assert lineage.generation == 0


def test_lineage_dispose_releases_arena_and_retires_active_branches():
    lineage = _lineage()
    branch = lineage.fork(owner_id="active")
    branch.append_checkpoint(_deltas(6))
    lineage.dispose()

    stats = lineage.stats()
    assert stats["closed"] is True
    assert stats["arena_nodes"] == 0
    assert stats["arena_delta_bytes"] == 0
    assert branch.status == BranchStatus.REJECTED
    assert branch.close() is None
    with pytest.raises(CacheBranchTransactionError, match="closed"):
        lineage.fork(owner_id="late")


def test_compaction_threshold_and_post_compaction_append():
    lineage = _lineage()
    branch = lineage.fork(owner_id="winner")
    branch.append_checkpoint(_deltas(6))
    branch.promote(7)
    assert lineage.needs_compaction(max_delta_depth=1, max_delta_bytes=10_000)
    receipt = lineage.compact(
        _Compactor(lambda base, deltas: _replacement(base, deltas, 2, 7))
    )
    assert receipt.position == 7
    assert not lineage.needs_compaction(
        max_delta_depth=2, max_delta_bytes=10_000
    )
    next_branch = lineage.fork(owner_id="next")
    next_branch.append_checkpoint(_deltas(7, generation=2))
    next_branch.promote(8)
    assert lineage.position == 8
