"""APCv2 model opt-in, layer segmentation, and atomic restore contracts."""

from types import SimpleNamespace

import mlx.core as mx

from mlx_lm.apc import (
    APCKey,
    AutomaticPrefixCache,
    AutomaticPrefixCacheV2,
    MTPAPCSidecar,
)
from mlx_lm.cache_planes import CachePlaneKind
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


def test_legacy_apc_cow_does_not_implicitly_adopt_v2_segments():
    legacy = AutomaticPrefixCache(cow_branching=True)
    key = APCKey("legacy")
    legacy.store(key, [1, 2], [_state(KVCache(), 2)])

    hit = legacy.lookup(key, [1, 2, 3])
    assert hit.hit
    assert hit.segment_manifest["segments"] == 0
    assert "version" not in legacy.apc_stats


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
