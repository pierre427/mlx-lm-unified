"""APCv2 model opt-in, layer segmentation, and atomic restore contracts."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_lm.apc import (
    APCKey,
    AutomaticPrefixCache,
    AutomaticPrefixCacheV2,
    MTPAPCSidecar,
)
from mlx_lm.cache_planes import (
    CachePlaneKind,
    TranscriptLedgerPlane,
    TranscriptLedgerSegment,
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.server import ResponseGenerator


def _state(cache, length, seed=0):
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(
        1, 1, length, 1
    )
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def _recurrent(length):
    cache = ArraysCache(1)
    cache[0] = mx.ones((1, 4), dtype=mx.float32)
    cache.lengths = mx.array([length], dtype=mx.int32)
    cache._host_lengths = (cache.lengths, [length])
    mx.eval(cache.state)
    return cache


def test_apcv2_records_layers_token_segments_and_mtp_plane():
    apc = AutomaticPrefixCacheV2(
        max_size=4, layout_name="test-hybrid-v1"
    )
    target = [
        _recurrent(600),
        _state(KVCache(), 600),
        _state(RotatingKVCache(max_size=128), 600, seed=1000),
    ]
    draft = [_state(KVCache(), 599, seed=2000)]
    sidecar = MTPAPCSidecar(
        (draft, mx.ones((1, 1, 4), dtype=mx.float32)),
        covered_tokens=600,
    )
    apc.store(APCKey("qwen4"), list(range(600)), target, sidecar=sidecar)

    hit = apc.lookup(APCKey("qwen4"), list(range(601)))
    assert hit.hit_kind == "mtp_sidecar"
    assert hit.cached_tokens == 600
    assert hit.segment_manifest["schema"] == "apcv2.layer-segments.v1"
    assert hit.segment_manifest["layers"] == 3
    assert hit.segment_manifest["by_plane"]["gdn_recurrent"]["segments"] == 1
    assert hit.segment_manifest["by_plane"]["attention_kv"]["segments"] == 2
    assert hit.segment_manifest["by_plane"]["attention_ring"]["segments"] == 1
    assert hit.segment_manifest["by_plane"]["mtp_draft"]["segments"] == 2

    stats = apc.apc_stats
    assert stats["version"] == 2
    assert stats["layout_name"] == "test-hybrid-v1"
    assert stats["layer_segments"]["fallback_entries"] == 0
    assert stats["layer_segments"]["segments"] == 6


@pytest.mark.parametrize("prompt_length", [2047, 2048, 2049])
def test_apcv2_hybrid_mtp_boundary_hits_around_prefill_page(prompt_length):
    """P-1 target/P-2 draft coverage must not collapse at a 2048 boundary."""

    covered = prompt_length - 1
    prompt = list(range(prompt_length))
    target = [_recurrent(covered), _state(KVCache(), covered)]
    draft = [_state(KVCache(), covered - 1, seed=prompt_length)]
    sidecar = MTPAPCSidecar(
        (draft, mx.ones((1, 1, 4), dtype=mx.float32)),
        covered_tokens=covered,
    )
    apc = AutomaticPrefixCacheV2(
        max_size=2, layout_name="qwen4-exp-layer-segments-v1"
    )
    apc.store(
        APCKey("qwen4"),
        prompt[:covered],
        target,
        sidecar=sidecar,
    )

    hit = apc.lookup(APCKey("qwen4"), prompt)

    assert hit.hit
    assert hit.hit_kind == "mtp_sidecar"
    assert hit.cached_tokens == covered
    assert hit.remaining_tokens == [prompt[-1]]
    assert hit.sidecar.covered_tokens == covered
    assert hit.sidecar.state[0][0].offset == covered - 1
    assert hit.cache[0].lengths.item() == covered
    assert hit.cache[1].offset == covered
    assert hit.segment_manifest["by_plane"]["gdn_recurrent"]["segments"] == 1
    assert hit.segment_manifest["by_plane"]["mtp_draft"]["segments"] == 2


def test_legacy_apc_cow_does_not_implicitly_adopt_v2_segments():
    legacy = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("legacy")
    legacy.store(key, [1, 2], [_state(KVCache(), 2)])

    hit = legacy.lookup(key, [1, 2, 3])
    assert hit.hit
    assert hit.segment_manifest["segments"] == 0
    assert "version" not in legacy.apc_stats


def test_apcv2_spills_idle_target_and_mtp_sidecar_then_restores_exactly():
    with tempfile.TemporaryDirectory() as directory:
        now = [0.0]
        apc = AutomaticPrefixCacheV2(
            max_size=4,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
            idle_disk_max_bytes=1 << 30,
            now_fn=lambda: now[0],
        )
        target = [_recurrent(3), _state(KVCache(), 3)]
        draft = [_state(KVCache(), 2, seed=100)]
        sidecar = MTPAPCSidecar(
            (draft, mx.ones((1, 1, 4), dtype=mx.float32)),
            covered_tokens=3,
            rng_key=mx.array([7, 11], dtype=mx.uint32),
            rng_draws=5,
        )
        key = APCKey("qwen4")
        apc.store(key, [1, 2, 3], target, sidecar=sidecar)

        now[0] = 179
        assert apc.spill_idle_entries() == 0
        now[0] = 180
        assert apc.spill_idle_entries() == 1
        assert apc.nbytes == 0
        assert apc.apc_stats["idle_disk"]["disk_entries"] == 1
        assert len(list(Path(directory).glob("apc-idle-*.safetensors"))) == 3

        hit = apc.lookup(key, [1, 2, 3, 4])

        assert hit.hit_kind == "mtp_sidecar"
        assert hit.cached_tokens == 3
        assert hit.remaining_tokens == [4]
        assert hit.cache[0].lengths.item() == 3
        assert hit.cache[1].offset == 3
        assert hit.sidecar.state[0][0].offset == 2
        assert hit.sidecar.covered_tokens == 3
        assert hit.sidecar.rng_draws == 5
        assert mx.array_equal(hit.sidecar.rng_key, mx.array([7, 11], dtype=mx.uint32))
        assert apc.apc_stats["idle_disk"]["restores"] == 1
        hit.cache.close()


def test_apcv2_resident_byte_pressure_spills_instead_of_dropping_entry():
    with tempfile.TemporaryDirectory() as directory:
        apc = AutomaticPrefixCacheV2(
            max_size=4,
            max_bytes=1,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
        )
        key = APCKey("qwen4")
        apc.store(key, [1, 2, 3], [_state(KVCache(), 3)])

        assert len(apc) == 1
        assert apc.nbytes == 0
        disk = apc.apc_stats["idle_disk"]
        assert disk["pressure_spills"] == 1
        assert disk["disk_entries"] == 1

        hit = apc.lookup(key, [1, 2, 3, 4])
        assert hit.hit
        assert hit.cached_tokens == 3
        hit.cache.close()


def test_apcv2_idle_spill_waits_for_live_cow_branch_to_close():
    with tempfile.TemporaryDirectory() as directory:
        now = [0.0]
        apc = AutomaticPrefixCacheV2(
            max_size=4,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
            now_fn=lambda: now[0],
        )
        key = APCKey("qwen4")
        apc.store(key, [1, 2, 3], [_state(KVCache(), 3)])
        live = apc.lookup(key, [1, 2, 3, 4]).cache

        now[0] = 180
        assert apc.spill_idle_entries() == 0
        assert apc.nbytes > 0

        live.close()
        now[0] = 181
        assert apc.spill_idle_entries() == 1
        assert apc.nbytes == 0


def test_apcv2_missing_disk_snapshot_fails_closed_as_cache_miss():
    with tempfile.TemporaryDirectory() as directory:
        now = [0.0]
        apc = AutomaticPrefixCacheV2(
            max_size=4,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
            now_fn=lambda: now[0],
        )
        key = APCKey("qwen4")
        tokens = [1, 2, 3]
        apc.store(key, tokens, [_state(KVCache(), 3)])
        now[0] = 180
        assert apc.spill_idle_entries() == 1
        entry = apc._trie.get(key, tokens)
        Path(entry._apc_disk["target"]).unlink()

        miss = apc.lookup(key, tokens + [4])

        assert not miss.hit
        assert len(apc) == 0
        assert apc.apc_stats["idle_disk"]["restore_failures"] == 1


def test_apcv2_disk_budget_evicts_oldest_disk_only_entry():
    with tempfile.TemporaryDirectory() as directory:
        apc = AutomaticPrefixCacheV2(
            max_size=4,
            max_bytes=1,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
            idle_disk_max_bytes=1,
        )

        apc.store(APCKey("qwen4"), [1, 2, 3], [_state(KVCache(), 3)])

        disk = apc.apc_stats["idle_disk"]
        assert len(apc) == 0
        assert disk["disk_entries"] == 0
        assert disk["disk_evictions"] == 1
        assert not list(Path(directory).glob("apc-idle-*.safetensors"))


def test_apcv2_target_segment_invalidation_rejects_atomic_restore():
    apc = AutomaticPrefixCacheV2(max_size=2, layout_name="test-kv-v1")
    key = APCKey("model")
    apc.store(key, list(range(300)), [_state(KVCache(), 300)])
    entry = apc._trie.get(key, list(range(300)))
    owner = entry.prompt_cache.cow_owner
    target_segment = next(
        segment
        for segment in owner.segment_manifest.segments
        if segment.key.kind == CachePlaneKind.ATTENTION_KV
    )

    assert owner.invalidate_segment(target_segment.key, "test-stale")
    miss = apc.lookup(key, list(range(301)))
    assert not miss.hit
    assert miss.miss_reason == "stale_cow_generation"


def test_apcv2_transcript_segments_are_optional_and_independently_invalidatable():
    apc = AutomaticPrefixCacheV2(max_size=2, layout_name="test-kv-v1")
    key = APCKey("model")
    ledger = TranscriptLedgerPlane(
        "test-tokenizer",
        "rev-a",
        "transcript-a",
        (
            TranscriptLedgerSegment("turn:1", 0, 3, (7, 8, 9)),
            TranscriptLedgerSegment("turn:2", 3, 5, (10, 11)),
        ),
    )
    tokens = [1, 2, 3]
    apc.store(
        key,
        tokens,
        [_state(KVCache(), len(tokens))],
        transcript_ledger=ledger,
    )

    hit = apc.lookup(key, tokens + [4])
    assert hit.hit
    assert hit.transcript_ledger is ledger
    assert hit.segment_manifest["by_plane"]["transcript_ledger"]["segments"] == 2

    entry = apc._trie.get(key, tokens)
    owner = entry.prompt_cache.cow_owner
    transcript_segment = next(
        segment
        for segment in owner.segment_manifest.segments
        if segment.key.kind == CachePlaneKind.TRANSCRIPT_LEDGER
    )
    assert owner.invalidate_segment(transcript_segment.key, "ledger-stale")

    target_only_hit = apc.lookup(key, tokens + [4])
    assert target_only_hit.hit
    assert target_only_hit.transcript_ledger is None


def test_server_selects_v2_only_for_declared_model_layout():
    response = ResponseGenerator.__new__(ResponseGenerator)
    response.prompt_cache = AutomaticPrefixCache(max_size=3, max_bytes=123456)
    response._cache_capsule_pool = None

    response._configure_apc_for_model(
        SimpleNamespace(apc_v2_layout="qwen4-exp-layer-segments-v1")
    )
    assert isinstance(response.prompt_cache, AutomaticPrefixCacheV2)
    assert response.prompt_cache.max_size == 3
    assert response.prompt_cache.max_bytes == 123456

    response._configure_apc_for_model(SimpleNamespace())
    assert type(response.prompt_cache) is AutomaticPrefixCache
