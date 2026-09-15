from dataclasses import dataclass

import pytest

from mlx_lm.cache_planes import TranscriptLedgerPlane, TranscriptLedgerSegment
from mlx_lm.spomin_layer import (
    HeadHotspotScorer,
    InMemorySpominBackend,
    SpominBackendError,
    SpominBackendCapabilities,
    SpominCapabilityError,
    SpominConfig,
    SpominLayer,
    SpominPlanError,
    SpominRevisionError,
    SpominTargetState,
    StaticSegmentScorer,
)


def ledger():
    return TranscriptLedgerPlane(
        tokenizer_identity="test-tokenizer",
        tokenizer_version="1",
        revision="transcript-r1",
        segments=(
            TranscriptLedgerSegment("turn:1", 0, 3, (1, 2, 3)),
            TranscriptLedgerSegment("turn:2", 3, 7, (4, 5, 6, 7)),
            TranscriptLedgerSegment("turn:3", 7, 9, (8, 9)),
            TranscriptLedgerSegment("turn:4", 9, 12, (10, 11, 12)),
        ),
    )


def state(*, tokens=80, recurrent=False, mtp=False):
    transcript = ledger()
    return SpominTargetState(
        revision="target-r1",
        target_tokens=tokens,
        transcript=transcript,
        visible_segment_ids=tuple(s.segment_id for s in transcript.segments),
        has_recurrent_state=recurrent,
        has_mtp_state=mtp,
    )


def test_layer_does_not_plan_below_pressure():
    layer = SpominLayer(SpominConfig(capacity_tokens=100))
    current = state(tokens=69)
    assert layer.pressure_state(current).value == "below_pressure"
    assert layer.plan(current) is None


def test_oldest_plan_protects_explicit_and_recent_segments():
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.75,
            pressure_ratio=0.79,
        )
    )
    plan = layer.plan(state(), protected_segment_ids=("turn:2",))

    assert plan.selection.segment_ids == ("turn:1",)
    assert set(plan.protected_segment_ids) == {"turn:2", "turn:4"}
    assert plan.projected_target_tokens == 77
    assert plan.shortfall_tokens == 2


def test_incomplete_plan_refuses_to_apply():
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.75,
            pressure_ratio=0.79,
        )
    )
    plan = layer.plan(state(), protected_segment_ids=("turn:2",))
    with pytest.raises(SpominPlanError, match="misses target by 2"):
        layer.apply(state(), plan, InMemorySpominBackend())


def test_lowest_importance_uses_external_scorer_and_can_select_holes():
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.75,
            pressure_ratio=0.79,
            protect_recent_segments=0,
            strategy="lowest_importance",
        ),
        scorer=StaticSegmentScorer(
            {"turn:1": 0.1, "turn:2": 10.0, "turn:3": 0.2, "turn:4": 8.0}
        ),
    )
    plan = layer.plan(state())

    assert plan.selection.segment_ids == ("turn:1", "turn:3")
    assert plan.noncontiguous
    assert plan.ready


def test_head_hotspot_scorer_preserves_one_strong_token():
    scorer = HeadHotspotScorer(
        {
            (0, 0): [0.1] * 12,
            (0, 1): [0.0, 0.0, 0.0, 0.2, 9.0, 0.2, 0.2, 0.3, 0.3, 0, 0, 0],
        }
    )
    scores = scorer.score(ledger())
    assert scores["turn:2"] == 9.0
    assert scores["turn:1"] == 0.1


def test_head_hotspot_scorer_requires_full_alignment():
    with pytest.raises(ValueError, match="11 scores for 12 tokens"):
        HeadHotspotScorer({(0, 0): [0.1] * 11}).score(ledger())


def test_summary_tokens_are_included_in_reclaim_target():
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.76,
            pressure_ratio=0.79,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(state(), replacement_token_ids=(90, 91))
    assert plan.selection.reclaimed_tokens == 7
    assert plan.projected_target_tokens == 75


def test_simulation_updates_target_but_preserves_uncompacted_pld_ledger():
    original = state()
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.76,
            pressure_ratio=0.79,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(original)
    updated = layer.apply(original, plan, InMemorySpominBackend())

    assert updated.target_tokens == 76
    assert updated.visible_segment_ids == ("turn:1", "turn:3", "turn:4")
    assert layer.proposal_transcript(updated) is original.transcript
    assert layer.proposal_transcript(updated).token_ids == tuple(range(1, 13))


def test_revision_change_refuses_backend_call():
    original = state()
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.75,
            pressure_ratio=0.79,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(original)
    changed = SpominTargetState(
        revision="target-r2",
        target_tokens=original.target_tokens,
        transcript=original.transcript,
        visible_segment_ids=original.visible_segment_ids,
    )
    with pytest.raises(SpominRevisionError, match="revision changed"):
        layer.apply(changed, plan, InMemorySpominBackend())


@dataclass
class RecordingBackend:
    capabilities: SpominBackendCapabilities
    called: bool = False

    def apply(self, state, plan):
        self.called = True
        return InMemorySpominBackend().apply(state, plan)


def test_real_backend_must_advertise_all_required_state_repairs():
    original = state(recurrent=True, mtp=True)
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.75,
            pressure_ratio=0.79,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(original)
    backend = RecordingBackend(
        SpominBackendCapabilities(attention_kv_edit=True)
    )

    with pytest.raises(SpominCapabilityError) as exc:
        layer.apply(original, plan, backend)
    assert "recurrent_state_repair" in str(exc.value)
    assert "mtp_state_repair" in str(exc.value)
    assert not backend.called


def test_noncontiguous_plan_requires_explicit_backend_support():
    original = state()
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.74,
            pressure_ratio=0.79,
            protect_recent_segments=0,
            strategy="lowest_importance",
        ),
        scorer=StaticSegmentScorer(
            {"turn:1": 0.1, "turn:2": 10, "turn:3": 0.2, "turn:4": 8}
        ),
    )
    plan = layer.plan(original)
    backend = RecordingBackend(
        SpominBackendCapabilities(attention_kv_edit=True)
    )

    with pytest.raises(SpominCapabilityError, match="noncontiguous_edit"):
        layer.apply(original, plan, backend)
    assert not backend.called


def test_exact_rebuild_backend_covers_model_state_requirements():
    original = state(recurrent=True, mtp=True)
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.75,
            pressure_ratio=0.79,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(original)
    backend = RecordingBackend(SpominBackendCapabilities(exact_rebuild=True))

    updated = layer.apply(original, plan, backend)
    assert updated.revision == "target-r1:spomin"
    assert updated.target_tokens == plan.projected_target_tokens
    assert backend.called


def test_force_can_plan_below_pressure():
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.60,
            pressure_ratio=0.70,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(state(tokens=65), force=True)
    assert plan.trigger.value == "forced"
    assert plan.ready


def test_adaptive_pressure_can_override_the_static_target_limit():
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.60,
            pressure_ratio=0.70,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(state(tokens=80), target_limit_tokens=76, force=True)
    assert plan.target_limit_tokens == 76
    assert plan.projected_target_tokens == 76
    assert plan.ready


def test_second_compaction_only_selects_segments_still_visible():
    original = state()
    partially_compacted = SpominTargetState(
        revision="target-r2",
        target_tokens=80,
        transcript=original.transcript,
        visible_segment_ids=("turn:1", "turn:3", "turn:4"),
    )
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.77,
            pressure_ratio=0.79,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(partially_compacted)
    assert plan.selection.segment_ids == ("turn:1",)
    assert "turn:2" not in plan.selection.segment_ids


def test_oldest_strategy_advances_past_segments_removed_earlier():
    original = state()
    partially_compacted = SpominTargetState(
        revision="target-r2",
        target_tokens=80,
        transcript=original.transcript,
        visible_segment_ids=("turn:3", "turn:4"),
    )
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.78,
            pressure_ratio=0.79,
            protect_recent_segments=0,
        )
    )
    plan = layer.plan(partially_compacted)
    assert plan.selection.segment_ids == ("turn:3",)


def test_backend_must_report_the_planned_state_transition():
    original = state()
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=100,
            target_ratio=0.76,
            pressure_ratio=0.79,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(original)

    @dataclass
    class LyingBackend:
        capabilities = SpominBackendCapabilities(simulation=True)

        def apply(self, state, plan):
            return state

    with pytest.raises(SpominBackendError, match="advance target revision"):
        layer.apply(original, plan, LyingBackend())
