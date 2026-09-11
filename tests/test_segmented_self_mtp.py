from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_lm.hybrid_speculative import (
    DetachedSelfMTPLane,
    HybridStats,
    MTPToken,
    SelfMTPCachePair,
    SelfMTPCycleResult,
    SelfMTPLane,
    attach_segmented_self_mtp_lanes,
    close_segmented_self_mtp_state,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    propose_batched_self_mtp,
)
from mlx_lm.segmented_self_mtp import (
    SegmentedLaneTransaction,
    require_segmented_self_mtp_engagement,
    segmented_self_mtp_enabled,
    segmented_self_mtp_stats,
)


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _FakeCache:
    def __init__(self, offset, nbytes=32):
        self.offset = int(offset)
        self.nbytes = int(nbytes)
        self.speculating = False

    @property
    def state(self):
        return ()

    def is_trimmable(self):
        return self.speculating

    def supports_ragged_trim(self):
        return True

    def start_speculation(self, rollback_window=None):
        self.speculating = True

    def stop_speculation(self):
        self.speculating = False

    def trim(self, count):
        count = int(count)
        self.offset -= count
        return count

    def trim_ragged(self, counts, validate=True):
        counts = list(counts)
        if len(counts) != 1:
            raise ValueError("fake B1 cache requires exactly one row")
        self.offset -= int(counts[0])
        return counts

    def preflight_ragged_trim(self, counts, validate=True):
        counts = list(counts)
        if len(counts) != 1:
            raise ValueError("fake B1 cache requires exactly one row")
        if validate and int(counts[0]) > self.offset:
            raise ValueError("fake B1 cache trim exceeds its offset")

    def prepare(self, lengths, right_padding):
        return None

    def finalize(self):
        return None


class _FakeKVCache(_FakeCache):
    keys = object()


class _FakeQSAKVCache(_FakeKVCache):
    index_keys = object()


class _FakeArraysCache(_FakeCache):
    def __init__(self, offset):
        super().__init__(offset)
        self.cache = [object()]


class _FakeMTPKVCache(_FakeKVCache):
    pass


class _FakeModel:
    def mtp_step(self, hidden, tokens, caches):
        width = int(tokens.shape[1])
        for cache in caches:
            cache.offset += width
        return hidden


class _FakeMatcher:
    def make_state(self):
        return None


class _CloseableCacheList(list):
    def __init__(self, values):
        super().__init__(values)
        self.closed = False

    def close(self):
        self.closed = True


def _detached(uid, position=6):
    lane = SelfMTPLane(
        uid=uid,
        cur=20 + uid,
        seed_h=mx.zeros((1, 1, 2)),
        pending_hs=None,
        pending_ts=[],
        token_prefix=mx.array([1, 2, 3], mx.uint32),
        rng=None,
        ntoks=1,
        max_tokens=8,
        num_draft=2,
        sampling_temp=0.0,
        accept_rule="residual",
        logprob_transform=None,
        logits_processors=[],
        stats=HybridStats(),
        share_qsa_indices=False,
    )
    return DetachedSelfMTPLane(
        lane,
        SelfMTPCachePair(
            target=[
                _FakeKVCache(position),
                _FakeQSAKVCache(position),
                _FakeArraysCache(position),
            ],
            draft=[_FakeMTPKVCache(position - 1)],
        ),
    )


def _fake_row_proposal(accepted_by_uid):
    def propose(_model, row_state):
        lane = row_state.lanes[0]
        accepted = int(accepted_by_uid[lane.uid])
        for cache in row_state.caches.target:
            cache.offset += accepted + 1
        drafts = (31 + lane.uid, 41 + lane.uid)
        bonus = 51 + lane.uid
        logprobs = mx.zeros((3, 64))
        hidden = mx.zeros((1, 3, 2))
        outputs = tuple(
            [MTPToken(drafts[index], logprobs[index], True) for index in range(accepted)]
            + [MTPToken(bonus, logprobs[accepted], False)]
        )
        proposal = SelfMTPCycleResult(
            membership_epoch=row_state.membership_epoch,
            lane_uids=(lane.uid,),
            draft_depths=(2,),
            accepted_lengths=(accepted,),
            target_drops=(2 - accepted,),
            head_drops=(2,),
            outputs=(outputs,),
            _old_curs=(lane.cur,),
            _old_seed_hs=(lane.seed_h,),
            _drafts=(drafts,),
            _vhidden=(hidden,),
            _logprobs=(logprobs,),
            _bonuses=(bonus,),
        )
        row_state.proposal_open = True
        row_state._open_proposal = proposal
        return proposal

    return propose


def test_segmented_self_mtp_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("MLX_LM_SEGMENTED_SELF_MTP", raising=False)
    assert segmented_self_mtp_enabled() is False


def test_serving_knob_honors_env_and_explicit_override(monkeypatch):
    from mlx_lm.generate import _segment_aware_live_tip_enabled

    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    assert _segment_aware_live_tip_enabled({}) is True
    assert _segment_aware_live_tip_enabled({"segment_aware_live_tip": False}) is False


def test_independent_b1_cycle_engages_without_physical_b2_and_promotes():
    segmented_self_mtp_stats(reset=True)
    detached = [_detached(0), _detached(1)]
    original_cache_ids = [
        {id(cache) for cache in item.caches.target + item.caches.draft}
        for item in detached
    ]
    assert original_cache_ids[0].isdisjoint(original_cache_ids[1])
    state = attach_segmented_self_mtp_lanes(object(), None, detached)

    with patch(
        "mlx_lm.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({0: 0, 1: 1}),
    ):
        proposal = propose_batched_self_mtp(object(), state)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[1, 2],
        terminal=[False, False],
    )

    assert [transaction.position for transaction in state.transactions] == [7, 8]
    assert [lane.cur for lane in state.lanes] == [51, 52]
    counters = segmented_self_mtp_stats()
    require_segmented_self_mtp_engagement(counters)
    assert counters["b1_target_forwards"] == 2
    assert counters["b1_draft_forwards"] == 4
    assert counters["physical_b2_formations"] == 0
    assert counters["transaction_promotions"] == 2
    assert counters["accepted_zero"] == 1
    assert counters["accepted_partial"] == 1
    assert counters["proposal_ns"] == counters["commit_ns"] == 0
    close_segmented_self_mtp_state(state)


def test_zero_delivery_rejects_transaction_and_restores_target_position():
    segmented_self_mtp_stats(reset=True)
    state = attach_segmented_self_mtp_lanes(object(), None, [_detached(7)])
    with patch(
        "mlx_lm.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({7: 0}),
    ):
        proposal = propose_batched_self_mtp(object(), state)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[0],
        terminal=[True],
    )
    assert state.transactions[0].position == 6
    assert all(cache.offset == 6 for cache in state.row_caches[0].target)
    counters = segmented_self_mtp_stats()
    assert counters["transaction_rejections"] == 1
    assert counters["transaction_promotions"] == 0
    close_segmented_self_mtp_state(state)


def test_segmented_attach_rejects_shared_mutable_cache_objects():
    left = _detached(0)
    right = _detached(1)
    right.caches = left.caches
    with pytest.raises(ValueError, match="alias"):
        attach_segmented_self_mtp_lanes(object(), None, [left, right])


def test_gate_refuses_silent_fallback_or_nonengagement():
    with pytest.raises(RuntimeError, match="never engaged"):
        require_segmented_self_mtp_engagement(
            {"engaged": 0, "b1_target_forwards": 0, "physical_b2_formations": 0}
        )
    with pytest.raises(RuntimeError, match="physical B2"):
        require_segmented_self_mtp_engagement(
            {"engaged": 1, "b1_target_forwards": 2, "physical_b2_formations": 1}
        )
    with pytest.raises(RuntimeError, match="completed no transaction"):
        require_segmented_self_mtp_engagement(
            {
                "engaged": 1,
                "b1_target_forwards": 2,
                "physical_b2_formations": 0,
                "transaction_branches": 2,
                "committed_cycles": 0,
            }
        )


def test_generation_batch_seam_keeps_two_concrete_b1_rows():
    from mlx_lm.generate import MTPGenerationBatch

    detached = [_detached(0), _detached(1)]
    with patch(
        "mlx_lm.hybrid_speculative.attach_self_mtp_lanes",
        side_effect=AssertionError("physical B2 attach must not run"),
    ):
        batch = MTPGenerationBatch(
            object(),
            detached,
            [None, None],
            [_FakeMatcher(), _FakeMatcher()],
            segmented_live_tip=True,
        )
    assert batch.state.row_caches == [item.caches for item in detached]
    assert not hasattr(batch.state, "caches")
    assert batch.cache_nbytes == 256
    batch.close()


def test_discarded_generation_row_releases_transaction_and_cache_owner():
    from mlx_lm.generate import MTPGenerationBatch

    item = _detached(0)
    target = _CloseableCacheList(item.caches.target)
    item.caches.target = target
    batch = MTPGenerationBatch(
        _FakeModel(),
        [item],
        [None],
        [_FakeMatcher()],
        segmented_live_tip=True,
    )
    transaction = batch.state.transactions[0]
    batch.filter([])
    assert transaction.closed is True
    assert target.closed is True
    batch.close()


def test_detach_flushes_pending_draft_to_caught_up_live_tip():
    state = attach_segmented_self_mtp_lanes(object(), None, [_detached(4)])
    with patch(
        "mlx_lm.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({4: 1}),
    ):
        proposal = propose_batched_self_mtp(object(), state)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[2],
        terminal=[False],
    )
    assert state.lanes[0].pending_hs is not None
    assert state.lanes[0].pending_ts

    state, detached = detach_self_mtp_lanes(_FakeModel(), state, [0])
    assert not state.lanes
    assert detached[0].lane.pending_hs is None
    assert detached[0].lane.pending_ts == []
    target_position = max(cache.offset for cache in detached[0].caches.target)
    draft_position = max(cache.offset for cache in detached[0].caches.draft)
    assert draft_position == target_position - 1
    assert detached[0].segment_transaction.predecessor_lineage_id is not None
    state = attach_segmented_self_mtp_lanes(object(), state, detached)
    assert len(state.lanes) == 1
    close_segmented_self_mtp_state(state)


def test_attach_partial_failure_releases_only_new_transaction():
    first = _detached(0)
    second = _detached(1, position=7)
    reused = SegmentedLaneTransaction(second.caches, second.lane, 7)
    second.segment_transaction = reused
    for cache in second.caches.target:
        cache.offset += 1
    second.caches.draft[0].offset += 1
    with pytest.raises(ValueError, match="state changed|position disagrees"):
        attach_segmented_self_mtp_lanes(object(), None, [first, second])
    assert first.segment_transaction is None
    assert reused.closed is False
    assert all(not cache.speculating for cache in first.caches.target)
    assert all(not cache.speculating for cache in second.caches.target)
    reused.close()


def test_generation_stale_parallel_branch_is_rejected():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    winner = transaction.fork("winner")
    stale = transaction.fork("stale")
    for cache in item.caches.target:
        cache.offset += 1
    item.caches.draft[0].offset += 1
    transaction.publish(
        winner, item.caches, item.lane, 1, proposed=2, accepted=1
    )
    with pytest.raises(Exception, match="generation|stale"):
        transaction.publish(
            stale, item.caches, item.lane, 1, proposed=2, accepted=1
        )
    stale.close()
    transaction.close()


@pytest.mark.parametrize("mutation", ["cur", "qsa", "gdn"])
def test_reattach_rejects_same_position_out_of_lineage_state_mutation(mutation):
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    item.segment_transaction = transaction
    if mutation == "cur":
        item.lane.cur += 1
    elif mutation == "qsa":
        item.caches.target[1].index_keys = object()
    else:
        item.caches.target[2].cache = [object()]
    with pytest.raises(ValueError, match="state changed outside its lineage"):
        attach_segmented_self_mtp_lanes(object(), None, [item])
    transaction.close()


def test_zero_delivery_refuses_changed_state_before_cheap_rejection():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    branch = transaction.fork("zero")
    item.caches.target[2].cache = [object()]
    with pytest.raises(ValueError, match="state changed outside its lineage"):
        transaction.publish(
            branch, item.caches, item.lane, 0, proposed=2, accepted=0
        )
    branch.close()
    transaction.close()


def test_each_position_bearing_target_cache_must_be_aligned():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    item.caches.target[1].offset -= 1
    with pytest.raises(ValueError, match="positions .* disagree"):
        transaction.validate(item.caches, item.lane, 6)
    transaction.close()


def test_reused_transaction_rejects_same_position_cache_swap():
    original = _detached(0)
    transaction = SegmentedLaneTransaction(
        original.caches, original.lane, 6
    )
    replacement = _detached(0)
    replacement.segment_transaction = transaction
    with pytest.raises(ValueError, match="fingerprint changed"):
        attach_segmented_self_mtp_lanes(object(), None, [replacement])
    assert transaction.closed is False
    transaction.close()


def test_qsa_transaction_planes_name_nonoverlapping_components():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    bases = {
        base.kind.value: base for base in transaction.lineage.current_view().bases
    }
    attention_fields = {
        field
        for entry in bases["attention_kv"].payload.fingerprint
        if "qsa" in entry[0].lower()
        for field in entry[2]
    }
    summary_fields = {
        field
        for entry in bases["qsa_summary"].payload.fingerprint
        for field in entry[2]
    }
    assert attention_fields == {"keys", "values", "key_scale", "value_scale"}
    assert summary_fields == {
        "index_keys",
        "_qsa_pooled_keys",
        "_qsa_pooled_ratio",
        "_qsa_summary_identity",
        "_qsa_summary_restored",
        "_qsa_pending_pooled",
        "_mtp_share_topk",
        "_mtp_shared_topk",
    }
    assert attention_fields.isdisjoint(summary_fields)
    transaction.close()


def test_mtp_plane_requires_physical_cache_plus_pending_sidecar_alignment():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    item.lane.pending_ts = [9]
    item.lane.pending_hs = mx.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="logical coordinate|not caught up"):
        transaction.validate(item.caches, item.lane, 6)
    item.lane.pending_hs = None
    with pytest.raises(ValueError, match="geometry disagrees"):
        transaction.validate(item.caches, item.lane, 6)
    transaction.close()


def test_commit_failure_clears_open_row_ownership_and_poisons():
    state = attach_segmented_self_mtp_lanes(object(), None, [_detached(0)])
    with patch(
        "mlx_lm.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({0: 0}),
    ):
        proposal = propose_batched_self_mtp(object(), state)
    with (
        patch.object(state.transactions[0], "publish", side_effect=RuntimeError("CAS")),
        pytest.raises(RuntimeError, match="CAS"),
    ):
        commit_batched_self_mtp(
            state,
            proposal,
            emitted_counts=[1],
            terminal=[False],
        )
    assert state.poisoned is True
    assert state._row_states == []
    assert state._row_proposals == []
    assert state._transaction_branches == []
    close_segmented_self_mtp_state(state)


def test_tiny_qwen4_gdn_qsa_mtp_runs_real_independent_b1_cycle():
    from test_batched_self_mtp_qwen4 import _prepare_lane, _tiny_qwen4_model

    model = _tiny_qwen4_model()
    detached = [
        _prepare_lane(model, 0, [1, 2, 3, 4]),
        _prepare_lane(model, 1, [5, 6, 7, 8, 9]),
    ]
    assert [type(cache).__name__ for cache in detached[0].caches.target] == [
        "Qwen4ArraysCache",
        "QSAKVCache",
    ]
    assert [type(cache).__name__ for cache in detached[0].caches.draft] == [
        "QSAKVCache"
    ]
    state = attach_segmented_self_mtp_lanes(model, None, detached)
    accepted = iter([0, 1])

    def force(logprobs, *_args, **_kwargs):
        count = next(accepted)
        return count, int(mx.argmax(logprobs[count]).item())

    with patch(
        "mlx_lm.hybrid_speculative._batched_residual_verify",
        side_effect=force,
    ):
        proposal = propose_batched_self_mtp(model, state)
    assert proposal.accepted_lengths == (0, 1)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[len(row) for row in proposal.outputs],
        terminal=[False, False],
    )
    assert [transaction.position for transaction in state.transactions] == [5, 7]
    planes = {
        base.kind.value
        for base in state.transactions[0].lineage.current_view().bases
    }
    assert planes == {"attention_kv", "qsa_summary", "gdn_recurrent", "mtp_draft"}

    accepted = iter([2, 0])
    with patch(
        "mlx_lm.hybrid_speculative._batched_residual_verify",
        side_effect=force,
    ):
        proposal = propose_batched_self_mtp(model, state)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[len(row) for row in proposal.outputs],
        terminal=[False, False],
    )
    assert [transaction.position for transaction in state.transactions] == [8, 8]

    state, lanes = detach_self_mtp_lanes(model, state, [0, 1])
    assert not state.lanes
    for item in lanes:
        target_position = max(
            int(getattr(cache, "offset", 0)) for cache in item.caches.target
        )
        draft_position = max(
            int(getattr(cache, "offset", 0)) for cache in item.caches.draft
        )
        assert draft_position == target_position - 1
    state = attach_segmented_self_mtp_lanes(model, state, lanes)
    assert len(state.lanes) == 2
    close_segmented_self_mtp_state(state)

    zero = _prepare_lane(model, 3, [11, 12, 13, 14])
    state = attach_segmented_self_mtp_lanes(model, None, [zero])

    def reject_all(logprobs, *_args, **_kwargs):
        return 0, int(mx.argmax(logprobs[0]).item())

    with patch(
        "mlx_lm.hybrid_speculative._batched_residual_verify",
        side_effect=reject_all,
    ):
        proposal = propose_batched_self_mtp(model, state)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[0],
        terminal=[True],
    )
    assert state.transactions[0].position == 4
    assert state.transactions[0].predecessor_lineage_id is not None
    close_segmented_self_mtp_state(state)
