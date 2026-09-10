import types
import unittest

from mlx_lm.apc import APCKey
from mlx_lm.apc_backends import BlockAPCAdapter, SnapshotAPCAdapter


class FakeSnapshotBackend:
    def __init__(self, lookup):
        self._lookup = lookup
        self.lookup_kwargs = None
        self.save_kwargs = None
        self.trim_prompt_cache = object()
        self.total_bytes = 123
        self.restore_status = "restored"
        self.restored_entries = 2

    def __len__(self):
        return 2

    def lookup_result(self, tokens, **kwargs):
        self.lookup_kwargs = kwargs
        return self._lookup

    def save(self, tokens, cache, **kwargs):
        self.save_kwargs = kwargs
        return "/cache/snapshot.safetensors"

    def prepare_cache(self, *args, **kwargs):
        return "delegated"


class FakeBlockManager:
    def __init__(self):
        self.released = []
        self.exact_store = None
        self.stats = types.SimpleNamespace(hits=3)

    def release(self, blocks):
        self.released.extend(blocks)

    def store_exact_cache(self, tokens, cache, **kwargs):
        self.exact_store = (tokens, cache, kwargs)
        return True


class FakeBlockModule:
    def __init__(self, plan=None):
        self.plan = plan
        self.lookup_kwargs = None
        self.commit_kwargs = None

    def apc_lookup_plan(self, manager, tokens, **kwargs):
        self.lookup_kwargs = kwargs
        return self.plan

    def commit_prefix_blocks(self, manager, cache, tokens, **kwargs):
        self.commit_kwargs = kwargs
        return ["new-block"]


class TestSnapshotAPCAdapter(unittest.TestCase):
    def test_hit_preserves_native_snapshot_metadata(self):
        hit = types.SimpleNamespace(
            cache=["kv"],
            tail_ids=[4],
            reused_tokens=3,
            kind="full_prefix",
        )
        native = types.SimpleNamespace(hit=hit, miss_reason=None)
        backend = FakeSnapshotBackend(native)
        adapter = SnapshotAPCAdapter(backend)
        key = APCKey(
            "laguna",
            revision="rev-a",
            adapter="lora-a",
            tokenizer_fingerprint="tok-a",
            cache_layout_fingerprint="30r-10kv",
            semantic_fingerprint="tenant-a",
        )

        result = adapter.lookup(key, [1, 2, 3, 4], cache_key="conversation")

        self.assertTrue(result.hit)
        self.assertEqual(result.cache, ["kv"])
        self.assertEqual(result.remaining_tokens, [4])
        self.assertIs(result.native, native)
        self.assertEqual(backend.lookup_kwargs["cache_key"], "conversation")
        self.assertIn('"revision":"rev-a"', backend.lookup_kwargs["model_id"])
        self.assertEqual(backend.lookup_kwargs["cache_salt"], "tenant-a")

    def test_miss_and_store_keep_backend_contract(self):
        native = types.SimpleNamespace(hit=None, miss_reason="compatibility_mismatch")
        backend = FakeSnapshotBackend(native)
        adapter = SnapshotAPCAdapter(backend)
        key = APCKey("north", cache_layout_fingerprint="36r-13kv")

        miss = adapter.lookup(key, [1, 2])
        stored = adapter.store(key, [1], ["cache"], kind="branch_anchor")

        self.assertFalse(miss.hit)
        self.assertEqual(miss.miss_reason, "compatibility_mismatch")
        self.assertTrue(stored.stored)
        self.assertEqual(stored.native, "/cache/snapshot.safetensors")
        self.assertEqual(backend.save_kwargs["kind"], "branch_anchor")
        self.assertEqual(adapter.prepare_cache(), "delegated")
        self.assertEqual(adapter.apc_stats["restored_entries"], 2)


class TestBlockAPCAdapter(unittest.TestCase):
    def test_block_lookup_preserves_plan_and_release_ownership(self):
        plan = {"matched_blocks": ["b1", "b2"], "prefix_len": 4}
        module = FakeBlockModule(plan)
        manager = FakeBlockManager()
        adapter = BlockAPCAdapter(manager, module, mode="block")
        key = APCKey("muse", semantic_fingerprint=77)

        result = adapter.lookup(key, [1, 2, 3, 4, 5])
        adapter.release(result)
        adapter.release(result)

        self.assertTrue(result.hit)
        self.assertIs(result.native, plan)
        self.assertEqual(result.remaining_tokens, [5])
        self.assertEqual(module.lookup_kwargs["extra_hash"], 77)
        self.assertEqual(manager.released, ["b1", "b2"])

    def test_exact_lookup_and_store(self):
        plan = {
            "matched_blocks": [],
            "warm_cache": ["warm"],
            "prefix_len": 3,
        }
        module = FakeBlockModule(plan)
        manager = FakeBlockManager()
        adapter = BlockAPCAdapter(manager, module, mode="exact")
        key = APCKey("muse", semantic_fingerprint=9)

        hit = adapter.lookup(key, [1, 2, 3, 4])
        stored = adapter.store(key, [1, 2, 3], ["cache"], clone=False, disk=False)

        self.assertEqual(hit.cache, ["warm"])
        self.assertTrue(stored.stored)
        self.assertEqual(
            manager.exact_store,
            ([1, 2, 3], ["cache"], {"extra_hash": 9, "clone": False, "disk": False}),
        )

    def test_non_integer_semantic_identity_fails_closed(self):
        adapter = BlockAPCAdapter(FakeBlockManager(), FakeBlockModule(), mode="block")
        key = APCKey("muse", semantic_fingerprint="image-not-hashed")

        miss = adapter.lookup(key, [1, 2])
        stored = adapter.store(key, [1], ["cache"])

        self.assertFalse(miss.hit)
        self.assertEqual(
            miss.miss_reason, "semantic_fingerprint_must_be_integer"
        )
        self.assertFalse(stored.stored)

    def test_block_store_routes_through_commit_and_releases_in_use(self):
        module = FakeBlockModule()
        adapter = BlockAPCAdapter(FakeBlockManager(), module, mode="block")

        result = adapter.store(
            APCKey("muse", semantic_fingerprint=5),
            [1, 2, 3],
            ["cache"],
            batch_idx=1,
            blocks_in_use=["old"],
        )

        self.assertTrue(result.stored)
        self.assertEqual(module.commit_kwargs["extra_hash"], 5)
        self.assertEqual(module.commit_kwargs["batch_idx"], 1)
        self.assertEqual(module.commit_kwargs["blocks_in_use"], ["old"])


if __name__ == "__main__":
    unittest.main()
