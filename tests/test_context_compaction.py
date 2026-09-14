from types import SimpleNamespace

import pytest

from mlx_lm.cache_planes import TranscriptLedgerPlane, TranscriptLedgerSegment
from mlx_lm.context_compaction import select_transcript_segments
from mlx_lm.server import (
    RequestCompositionError,
    _prompt_lookup_transcript_plane,
    _transcript_scoped_apc_key,
)


def _ledger(strategy="oldest_contiguous"):
    return TranscriptLedgerPlane(
        "test-tokenizer",
        "rev-a",
        "ledger-a",
        (
            TranscriptLedgerSegment("turn:1", 0, 2, (1, 2)),
            TranscriptLedgerSegment("turn:2", 2, 7, (3, 4, 5, 6, 7)),
            TranscriptLedgerSegment("turn:3", 7, 10, (8, 9, 10)),
        ),
        compaction_strategy=strategy,
    )


def test_oldest_contiguous_stops_at_first_protected_segment():
    selection = select_transcript_segments(
        _ledger(),
        6,
        protected_segment_ids={"turn:2"},
    )
    assert selection.segment_ids == ("turn:1",)
    assert selection.reclaimed_tokens == 2


def test_largest_first_is_selectable_and_preserves_protected_segments():
    selection = select_transcript_segments(
        _ledger("largest_first"),
        4,
        protected_segment_ids={"turn:3"},
    )
    assert selection.segment_ids == ("turn:2",)
    assert selection.reclaimed_tokens == 5


def test_compaction_strategy_and_target_fail_closed():
    with pytest.raises(ValueError, match="unknown compaction strategy"):
        select_transcript_segments(_ledger(), 1, strategy="mystery")
    with pytest.raises(ValueError, match="non-negative"):
        select_transcript_segments(_ledger(), -1)


def test_lowest_importance_preserves_hot_segments():
    selection = select_transcript_segments(
        _ledger(),
        4,
        strategy="lowest_importance",
        importance_by_segment={"turn:1": 0.8, "turn:2": 0.1, "turn:3": 0.9},
    )
    assert selection.segment_ids == ("turn:2",)
    assert selection.reclaimed_tokens == 5
    with pytest.raises(ValueError, match="requires segment importance"):
        select_transcript_segments(_ledger(), 1, strategy="lowest_importance")


def test_request_segments_materialize_as_one_apcv2_transcript_plane():
    class Tokenizer:
        name_or_path = "test-tokenizer"
        revision = "rev-a"

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(char) for char in text]

    args = SimpleNamespace(
        prompt_lookup_context=[
            {"id": "turn:1", "content": "ab"},
            {"id": "turn:2", "content": [3, 4, 5]},
        ],
        context_compaction_strategy="largest_first",
    )
    plane = _prompt_lookup_transcript_plane(Tokenizer(), args, max_tokens=5)

    assert plane.token_ids == (97, 98, 3, 4, 5)
    assert [segment.segment_id for segment in plane.segments] == [
        "turn:1",
        "turn:2",
    ]
    assert plane.compaction_strategy == "largest_first"
    with pytest.raises(RequestCompositionError, match="exceeds server limit"):
        _prompt_lookup_transcript_plane(Tokenizer(), args, max_tokens=4)


def test_transcript_fingerprint_scopes_the_apc_key():
    first = _ledger()
    second = TranscriptLedgerPlane(
        "test-tokenizer",
        "rev-a",
        "ledger-b",
        first.segments,
    )

    assert _transcript_scoped_apc_key("model", None) == "model"
    assert _transcript_scoped_apc_key(
        "model", first
    ) != _transcript_scoped_apc_key("model", second)
