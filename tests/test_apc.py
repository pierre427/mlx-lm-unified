import unittest

import mlx.core as mx

from mlx_lm.apc import (
    APCKey,
    AutomaticPrefixCache,
    MTPAPCSidecar,
    inspect_apc_capabilities,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache


def _state(cache, length, seed=0):
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    return cache


def _mixed_cache(length, window=8):
    return [
        _state(RotatingKVCache(max_size=window), length),
        _state(KVCache(), length, seed=100),
    ]


class TestAutomaticPrefixCache(unittest.TestCase):
    def test_standard_kv_uses_regular_apc_interface(self):
        apc = AutomaticPrefixCache()
        key = APCKey("model", revision="rev-a", tokenizer_fingerprint="tok-a")
        apc.store(key, [1, 2, 3], [_state(KVCache(), 3)])

        hit = apc.lookup(key, [1, 2, 3, 4])

        self.assertTrue(hit.hit)
        self.assertEqual(hit.hit_kind, "prefix")
        self.assertEqual(hit.cached_tokens, 3)
        self.assertEqual(hit.remaining_tokens, [4])

    def test_laguna_north_mixed_rotating_topology_hits_exact_prefix(self):
        apc = AutomaticPrefixCache()
        key = APCKey("laguna", cache_layout_fingerprint="30r-10kv-w512")
        capabilities = apc.store(key, list(range(12)), _mixed_cache(12, window=8))

        hit = apc.lookup(key, list(range(12)) + [99])

        self.assertEqual(capabilities.topology, "mixed_rotating_kv")
        self.assertTrue(capabilities.exact_prefix)
        self.assertTrue(hit.hit)
        self.assertEqual(hit.cached_tokens, 12)
        self.assertEqual(hit.remaining_tokens, [99])

    def test_wrapped_rotating_branch_fails_closed_when_no_prefix_snapshot_exists(self):
        apc = AutomaticPrefixCache()
        key = APCKey("north", cache_layout_fingerprint="36r-13kv-w8")
        apc.store(key, list(range(12)), _mixed_cache(12, window=8))

        result = apc.lookup(key, list(range(6)) + [900, 901])

        self.assertFalse(result.hit)
        self.assertIsNone(result.cache)
        self.assertEqual(result.remaining_tokens, list(range(6)) + [900, 901])
        self.assertEqual(result.miss_reason, "untrimmable_branch")

    def test_muse_semantic_fingerprint_separates_equal_text_tokens(self):
        apc = AutomaticPrefixCache()
        text_tokens = [7, 8, 9]
        image_a = APCKey("muse", semantic_fingerprint="image-sha256-a")
        image_b = APCKey("muse", semantic_fingerprint="image-sha256-b")
        apc.store(image_a, text_tokens, _mixed_cache(3, window=8))

        self.assertTrue(apc.lookup(image_a, text_tokens + [10]).hit)
        self.assertFalse(apc.lookup(image_b, text_tokens + [10]).hit)

    def test_capabilities_report_rotating_and_full_mix(self):
        capabilities = inspect_apc_capabilities(_mixed_cache(4, window=8))

        self.assertEqual(capabilities.topology, "mixed_rotating_kv")
        self.assertTrue(capabilities.exact_prefix)
        self.assertTrue(capabilities.arbitrary_branch)

    def test_legacy_server_methods_remain_compatible_and_are_counted(self):
        apc = AutomaticPrefixCache()
        cache = [_state(KVCache(), 2)]
        apc.insert_cache(("model", None, None), [1, 2], cache)

        restored, remaining = apc.fetch_nearest_cache(
            ("model", None, None), [1, 2, 3]
        )

        self.assertIsNotNone(restored)
        self.assertEqual(remaining, [3])
        self.assertEqual(apc.apc_stats["hits"], 1)
        self.assertEqual(apc.apc_stats["stores"], 1)

    def test_mtp_sidecar_restores_only_at_joint_capture_boundary(self):
        apc = AutomaticPrefixCache()
        key = APCKey("qwen4")
        target = [_state(KVCache(), 2)]
        mtp = [_state(KVCache(), 1)]
        sidecar = MTPAPCSidecar(
            (mtp, mx.zeros((1, 1, 4), mx.float32)), covered_tokens=2
        )
        # The path contains the last yielded token, while both model caches
        # cover the two tokens before it.
        apc.store(key, [1, 2, 3], target, sidecar=sidecar)

        hit = apc.lookup(key, [1, 2, 3, 4])
        self.assertEqual(hit.hit_kind, "mtp_sidecar")
        self.assertEqual(hit.cached_tokens, 2)
        self.assertEqual(hit.remaining_tokens, [3, 4])
        self.assertIsNot(hit.sidecar, sidecar)
        self.assertEqual(hit.sidecar.covered_tokens, 2)

        # A branch before the joint boundary must never pair a trimmed target
        # cache with an untrimmed draft state.
        branch = apc.lookup(key, [1, 9, 10])
        self.assertIsNone(branch.sidecar)


if __name__ == "__main__":
    unittest.main()
