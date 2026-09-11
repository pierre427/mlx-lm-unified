import threading
import time
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from mlx_lm.cache_capsule import (
    CacheCapsuleDisabled,
    CacheCapsuleError,
    CacheCapsuleGeneration,
    CacheCapsuleOwner,
    CacheCapsuleOwnerReleased,
    CacheCapsulePool,
    CacheCapsuleProduct,
    KVCacheCapsulePayload,
    StaleCacheCapsule,
    build_kv_cache_capsule_cpu,
    build_kv_cache_capsule_gpu,
    capture_kv_cache_plane,
    inspect_kv_cache_capsule,
    prepare_prompt_cache_capsules,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache


def _cache(length=5, dtype=mx.bfloat16):
    cache = KVCache()
    keys = mx.arange(length * 6, dtype=mx.float32).reshape(1, 2, length, 3)
    values = keys + 100
    cache.update_and_fetch(keys.astype(dtype), values.astype(dtype))
    mx.eval(cache.keys, cache.values)
    return cache


def _source(clock, length=5, target_batch=2, verify_raw_bits=False):
    return capture_kv_cache_plane(
        _cache(length),
        generation=clock.current,
        source_id="apc:key:layer-0",
        target_batch=target_batch,
        verify_raw_bits=verify_raw_bits,
    )


def _e5rt_product(source, backing_owner=None):
    product = build_kv_cache_capsule_cpu(source)
    payload = product.payload
    return CacheCapsuleProduct(
        KVCacheCapsulePayload(
            payload.keys,
            payload.values,
            payload.offset,
            payload.source_generation,
            payload.source_id,
                payload.layout_fingerprint,
                "e5rt",
                payload.raw_digest,
        ),
        backing_owner=backing_owner,
    )


class _SlowAdapter:
    def __init__(self, release, product):
        self.release_event = release
        self.product = product

    def stage(self, source):
        return self.product

    def build(self, staged):
        self.release_event.wait(timeout=5)
        return staged

    def adopt(self, built, source):
        return built


class _BrokenAdapter:
    def stage(self, source):
        return None

    def build(self, staged):
        raise RuntimeError("adapter failed")

    def adopt(self, built, source):
        return built


class _CountingBacking:
    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1


class _CompletedAdapter:
    def __init__(self, product):
        self.product = product
        self.discard_calls = 0

    def stage(self, source):
        return self.product

    def build(self, staged):
        return staged

    def adopt(self, built, source):
        return built

    def discard(self, value):
        self.discard_calls += 1
        value.backing_owner.release()


class _StageLease:
    def __init__(self):
        self.releases = 0
        self._lock = threading.Lock()

    def release(self):
        with self._lock:
            self.releases += 1


class _StageAbortAdapter:
    def __init__(self, staged, *, clock=None, entered=None, resume=None):
        self.staged = staged
        self.clock = clock
        self.entered = entered
        self.resume = resume
        self.abort_calls = 0
        self.build_calls = 0

    def stage(self, source):
        if self.entered is not None:
            self.entered.set()
        if self.resume is not None:
            self.resume.wait(timeout=2)
        if self.clock is not None:
            self.clock.advance()
        return self.staged

    def build(self, staged):
        self.build_calls += 1
        raise AssertionError("stale or closed staged work must not build")

    def abort(self, staged, reason):
        self.abort_calls += 1
        staged.release()


class TestCacheCapsule(unittest.TestCase):
    def test_capability_is_deliberately_plain_kv_only(self):
        self.assertTrue(inspect_kv_cache_capsule(_cache(), 2).supported)
        self.assertEqual(
            inspect_kv_cache_capsule(RotatingKVCache(max_size=8), 2).reason,
            "plain_kv_only",
        )

    def test_default_off_gate_fails_closed(self):
        clock = CacheCapsuleGeneration()
        with CacheCapsulePool(clock, enabled=False) as pool:
            with self.assertRaises(CacheCapsuleDisabled):
                pool.prepare(_source(clock), primary="cpu")

    def test_cpu_constructor_preserves_bf16_raw_bits(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)

        product = build_kv_cache_capsule_cpu(source)
        mx.eval(product.payload.keys, product.payload.values)

        expected_keys = mx.concatenate([source.keys, source.keys], axis=0)
        expected_values = mx.concatenate([source.values, source.values], axis=0)
        self.assertTrue(
            mx.array_equal(
                product.payload.keys.view(mx.uint16),
                expected_keys.view(mx.uint16),
            ).item()
        )
        self.assertTrue(
            mx.array_equal(
                product.payload.values.view(mx.uint16),
                expected_values.view(mx.uint16),
            ).item()
        )
        self.assertEqual(product.payload.keys.dtype, mx.bfloat16)

    def test_mixed_prompt_cache_reaches_real_attention_consumer(self):
        class MergeOnly:
            def merge(self, caches):
                return ("ordinary", len(caches))

        clock = CacheCapsuleGeneration()
        source_cache = _cache(length=7)
        with CacheCapsulePool(clock, enabled=True) as pool:
            prepared = prepare_prompt_cache_capsules(
                [source_cache, MergeOnly()],
                target_batch=2,
                generation=clock.current,
                pool=pool,
                backend="gpu",
                source_prefix="test-apc-hit",
            )
            self.assertIsNotNone(prepared)
            self.assertEqual(prepared.capsule_planes, 1)
            self.assertEqual(prepared.ordinary_planes, 1)
            restored = prepared.prompt_cache[0]
            self.assertEqual(restored.keys.shape[0], 2)
            self.assertEqual(prepared.prompt_cache[1], ("ordinary", 2))

            query = mx.ones((2, 2, 1, 3), dtype=mx.bfloat16)
            keys, values = restored.keys_and_values()
            output = mx.fast.scaled_dot_product_attention(
                query,
                keys,
                values,
                scale=3 ** -0.5,
            )
            mx.eval(output)
            self.assertEqual(output.shape, (2, 2, 1, 3))
            self.assertTrue(mx.array_equal(output[0], output[1]).item())
            prepared.close()
            self.assertTrue(all(r.owner.released for r in prepared.receipts))

    def test_no_eligible_plane_declines_without_replacing_incumbent_merge(self):
        class MergeOnly:
            def __init__(self):
                self.calls = 0

            def merge(self, caches):
                self.calls += 1
                return len(caches)

        clock = CacheCapsuleGeneration()
        plane = MergeOnly()
        with CacheCapsulePool(clock, enabled=True) as pool:
            self.assertIsNone(
                prepare_prompt_cache_capsules(
                    [plane],
                    target_batch=2,
                    generation=clock.current,
                    pool=pool,
                )
            )
        self.assertEqual(plane.calls, 0)

    def test_restore_failure_releases_current_capsule_owner_and_lease(self):
        class BackingOwner:
            def __init__(self):
                self.releases = 0

            def release(self):
                self.releases += 1

        backing = BackingOwner()
        clock = CacheCapsuleGeneration()

        def build_with_backing(source):
            product = build_kv_cache_capsule_gpu(source)
            return CacheCapsuleProduct(product.payload, backing_owner=backing)

        with (
            CacheCapsulePool(clock, enabled=True) as pool,
            mock.patch(
                "mlx_lm.cache_capsule.build_kv_cache_capsule_gpu",
                side_effect=build_with_backing,
            ),
            self.assertRaisesRegex(RuntimeError, "consumer sync failed"),
        ):
            prepare_prompt_cache_capsules(
                [_cache()],
                target_batch=2,
                generation=clock.current,
                pool=pool,
                backend="gpu",
                synchronize=lambda _payload: (_ for _ in ()).throw(
                    RuntimeError("consumer sync failed")
                ),
            )
        self.assertEqual(backing.releases, 1)

    def test_close_releases_owner_even_when_stream_synchronize_raises(self):
        class BackingOwner:
            def __init__(self):
                self.releases = 0

            def release(self):
                self.releases += 1

        backing = BackingOwner()
        clock = CacheCapsuleGeneration()

        def build_with_backing(source):
            product = build_kv_cache_capsule_gpu(source)
            return CacheCapsuleProduct(product.payload, backing_owner=backing)

        with (
            CacheCapsulePool(clock, enabled=True) as pool,
            mock.patch(
                "mlx_lm.cache_capsule.build_kv_cache_capsule_gpu",
                side_effect=build_with_backing,
            ),
        ):
            prepared = prepare_prompt_cache_capsules(
                [_cache()],
                target_batch=2,
                generation=clock.current,
                pool=pool,
                backend="gpu",
            )
            with (
                mock.patch(
                    "mlx_lm.cache_capsule.mx.synchronize",
                    side_effect=RuntimeError("stream sync failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "stream sync failed"),
            ):
                prepared.close()
        self.assertEqual(backing.releases, 1)
        self.assertTrue(all(receipt.owner.released for receipt in prepared.receipts))

    def test_captured_source_isolated_from_later_live_cache_write(self):
        clock = CacheCapsuleGeneration()
        cache = _cache()
        source = capture_kv_cache_plane(
            cache,
            generation=clock.current,
            source_id="immutable-capture",
            target_batch=2,
        )
        before = mx.array(source.keys)
        mx.eval(before)

        cache.keys[..., 0, :] = mx.full(
            cache.keys[..., 0, :].shape, 99, dtype=mx.bfloat16
        )
        mx.eval(cache.keys, source.keys)

        self.assertTrue(mx.array_equal(source.keys, before).item())
        self.assertFalse(mx.array_equal(cache.keys, before).item())

    def test_first_consumer_synchronizes_once_and_restores_real_cache(self):
        clock = CacheCapsuleGeneration()
        with CacheCapsulePool(clock, enabled=True) as pool:
            receipt = pool.prepare(_source(clock), primary="gpu", fallback=None)
            lease = receipt.owner.lease()
            calls = []

            def synchronize(payload):
                calls.append(payload.source_id)
                mx.eval(payload.keys, payload.values)

            restored = lease.restore_kv_cache(synchronize)
            second = lease.payload_for_consumer(synchronize)

            self.assertEqual(calls, ["apc:key:layer-0"])
            self.assertEqual(restored.offset, 5)
            self.assertEqual(restored.keys.shape[0], 2)
            self.assertIs(restored.keys, second.keys)
            lease.close()
            receipt.owner.release()

    def test_stale_generation_is_rejected_before_build(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)
        clock.advance()
        with CacheCapsulePool(clock, enabled=True) as pool:
            with self.assertRaises(StaleCacheCapsule):
                pool.prepare(source, primary="cpu")
            self.assertEqual(pool.counters["stale"], 1)

    def test_timeout_uses_named_cpu_fallback(self):
        clock = CacheCapsuleGeneration()
        release = threading.Event()
        source = _source(clock)
        pool = CacheCapsulePool(
            clock,
            e5rt_adapter=_SlowAdapter(release, _e5rt_product(source)),
            enabled=True,
        )
        try:
            receipt = pool.prepare(
                source,
                primary="e5rt",
                fallback="cpu",
                primary_timeout_s=0.001,
            )
            self.assertEqual(receipt.backend, "cpu")
            self.assertEqual(receipt.fallback_reason, "e5rt_timeout")
            self.assertEqual(pool.counters["timeouts"], 1)
            self.assertEqual(pool.counters["fallbacks"], 1)
            receipt.owner.release()
        finally:
            release.set()
            pool.close()

    def test_prepared_cache_reports_requested_and_actual_fallback_backends(self):
        clock = CacheCapsuleGeneration()
        release = threading.Event()
        cache = _cache()
        source = capture_kv_cache_plane(
            cache,
            generation=clock.current,
            source_id="apc:plane:0",
            target_batch=2,
        )
        pool = CacheCapsulePool(
            clock,
            e5rt_adapter=_SlowAdapter(release, _e5rt_product(source)),
            enabled=True,
        )
        try:
            prepared = prepare_prompt_cache_capsules(
                [cache],
                target_batch=2,
                generation=clock.current,
                pool=pool,
                backend="e5rt",
                fallback="cpu",
                timeout_s=0,
            )
            self.assertEqual(prepared.requested_backend, "e5rt")
            self.assertEqual(prepared.backend, "cpu")
            self.assertEqual(prepared.actual_backends, ("cpu",))
            self.assertEqual(prepared.fallback_reasons, ("e5rt_timeout",))
            prepared.close()
        finally:
            release.set()
            pool.close()

    def test_adapter_error_uses_named_cpu_fallback(self):
        clock = CacheCapsuleGeneration()
        with CacheCapsulePool(
            clock, e5rt_adapter=_BrokenAdapter(), enabled=True
        ) as pool:
            receipt = pool.prepare(
                _source(clock), primary="e5rt", fallback="cpu"
            )
            self.assertEqual(receipt.backend, "cpu")
            self.assertEqual(receipt.fallback_reason, "e5rt_error:RuntimeError")
            self.assertEqual(pool.counters["errors"], 1)
            receipt.owner.release()

    def test_e5rt_stage_and_adopt_stay_on_consumer_thread(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)
        caller = threading.get_ident()

        class Adapter:
            def __init__(self):
                self.threads = {}
                self.product = _e5rt_product(source)

            def stage(self, value):
                self.threads["stage"] = threading.get_ident()
                return self.product

            def build(self, staged):
                self.threads["build"] = threading.get_ident()
                return staged

            def adopt(self, built, value):
                self.threads["adopt"] = threading.get_ident()
                return built

        adapter = Adapter()
        with CacheCapsulePool(
            clock, e5rt_adapter=adapter, enabled=True
        ) as pool:
            receipt = pool.prepare(source, primary="e5rt", fallback=None)
            receipt.owner.release()

        self.assertEqual(adapter.threads["stage"], caller)
        self.assertNotEqual(adapter.threads["build"], caller)
        self.assertEqual(adapter.threads["adopt"], caller)

    def test_submit_returns_before_external_build_finishes(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)
        release = threading.Event()
        with CacheCapsulePool(
            clock,
            e5rt_adapter=_SlowAdapter(release, _e5rt_product(source)),
            enabled=True,
        ) as pool:
            ticket = pool.submit(source)
            self.assertFalse(ticket.future.done())
            release.set()
            receipt = ticket.await_adopt(timeout_s=1, fallback=None)
            receipt.owner.release()

    def test_generation_rechecked_at_lease_and_first_consumer(self):
        clock = CacheCapsuleGeneration()
        with CacheCapsulePool(clock, enabled=True) as pool:
            first = pool.prepare(_source(clock), primary="cpu", fallback=None)
            clock.advance()
            with self.assertRaises(StaleCacheCapsule):
                first.owner.lease()

            source = _source(clock)
            second = pool.prepare(source, primary="cpu", fallback=None)
            lease = second.owner.lease()
            clock.advance()
            with self.assertRaises(StaleCacheCapsule):
                lease.payload_for_consumer(lambda value: None)
            lease.close()

    def test_wrong_dtype_and_checksum_fail_closed(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock, verify_raw_bits=True)
        good = build_kv_cache_capsule_cpu(source).payload
        wrong_dtype = KVCacheCapsulePayload(
            good.keys.astype(mx.float16), good.values.astype(mx.float16),
            good.offset, good.source_generation, good.source_id,
            good.layout_fingerprint, good.backend, good.raw_digest,
        )
        wrong_digest = KVCacheCapsulePayload(
            good.keys, good.values, good.offset, good.source_generation,
            good.source_id, good.layout_fingerprint, good.backend, "bad",
        )
        with CacheCapsulePool(
            clock, enabled=True, verify_raw_bits=True
        ) as pool:
            with self.assertRaisesRegex(CacheCapsuleError, "dtype"):
                pool._accept_product(
                    CacheCapsuleProduct(wrong_dtype), source, "cpu", None
                )
            with self.assertRaisesRegex(CacheCapsuleError, "checksum"):
                pool._accept_product(
                    CacheCapsuleProduct(wrong_digest), source, "cpu", None
                )

    def test_full_integrity_is_explicit_not_a_hot_path_default(self):
        clock = CacheCapsuleGeneration()
        metadata_source = _source(clock)
        self.assertIsNone(metadata_source.raw_digest)
        with CacheCapsulePool(clock, enabled=True) as pool:
            receipt = pool.prepare(metadata_source, primary="cpu", fallback=None)
            receipt.owner.release()

        with CacheCapsulePool(
            clock, enabled=True, verify_raw_bits=True
        ) as pool:
            with self.assertRaisesRegex(CacheCapsuleError, "verified source"):
                pool.prepare(metadata_source, primary="cpu", fallback=None)

    def test_pool_close_is_nonblocking_for_wedged_external_job(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)
        release = threading.Event()

        class Adapter(_SlowAdapter):
            def __init__(self):
                super().__init__(release, _e5rt_product(source))
                self.cancel_calls = 0
                self.discard_calls = 0

            def cancel(self, staged, reason):
                self.cancel_calls += 1

            def discard(self, value):
                self.discard_calls += 1

        adapter = Adapter()
        pool = CacheCapsulePool(clock, e5rt_adapter=adapter, enabled=True)
        pool.submit(source)
        started = time.monotonic()
        pool.close()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.1)
        deadline = time.monotonic() + 1
        while adapter.cancel_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(adapter.cancel_calls, 1)
        release.set()
        deadline = time.monotonic() + 1
        while adapter.discard_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(adapter.discard_calls, 1)

    def test_completed_then_cancel_disposes_external_result_exactly_once(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)
        backing = _CountingBacking()
        adapter = _CompletedAdapter(_e5rt_product(source, backing))
        pool = CacheCapsulePool(clock, e5rt_adapter=adapter, enabled=True)

        ticket = pool.submit(source)
        ticket.future.result(timeout=1)
        deadline = time.monotonic() + 1
        while pool._active_ticket is not None and time.monotonic() < deadline:
            time.sleep(0.001)

        self.assertTrue(ticket.cancel("completed_then_cancel"))
        self.assertFalse(ticket.cancel("repeat_cancel"))
        pool.close()
        pool.close()
        self.assertEqual(adapter.discard_calls, 1)
        self.assertEqual(backing.releases, 1)

    def test_completed_then_stale_await_disposes_external_result_exactly_once(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)
        backing = _CountingBacking()
        adapter = _CompletedAdapter(_e5rt_product(source, backing))
        pool = CacheCapsulePool(clock, e5rt_adapter=adapter, enabled=True)

        ticket = pool.submit(source)
        ticket.future.result(timeout=1)
        clock.advance()
        with self.assertRaises(StaleCacheCapsule):
            ticket.await_adopt(timeout_s=1, fallback=None)
        pool.close()

        self.assertEqual(adapter.discard_calls, 1)
        self.assertEqual(backing.releases, 1)

    def test_pool_close_reclaims_completed_unadopted_result_exactly_once(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)
        backing = _CountingBacking()
        adapter = _CompletedAdapter(_e5rt_product(source, backing))
        pool = CacheCapsulePool(clock, e5rt_adapter=adapter, enabled=True)

        ticket = pool.submit(source)
        ticket.future.result(timeout=1)
        deadline = time.monotonic() + 1
        while pool._active_ticket is not None and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertIsNone(pool._active_ticket)

        pool.close()
        pool.close()
        self.assertEqual(adapter.discard_calls, 1)
        self.assertEqual(backing.releases, 1)

    def test_generation_change_after_stage_aborts_staged_owner_exactly_once(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)
        staged = _StageLease()
        adapter = _StageAbortAdapter(staged, clock=clock)
        pool = CacheCapsulePool(clock, e5rt_adapter=adapter, enabled=True)

        ticket = pool.submit(source)
        with self.assertRaises(StaleCacheCapsule):
            ticket.future.result(timeout=1)
        pool.close()
        deadline = time.monotonic() + 1
        while staged.releases == 0 and time.monotonic() < deadline:
            time.sleep(0.001)

        self.assertEqual(adapter.build_calls, 0)
        self.assertEqual(adapter.abort_calls, 1)
        self.assertEqual(staged.releases, 1)

    def test_pool_close_during_stage_aborts_returned_owner_exactly_once(self):
        clock = CacheCapsuleGeneration()
        staged = _StageLease()
        entered = threading.Event()
        resume = threading.Event()
        adapter = _StageAbortAdapter(staged, entered=entered, resume=resume)
        pool = CacheCapsulePool(clock, e5rt_adapter=adapter, enabled=True)
        errors = []

        def submit_from_creator_thread():
            try:
                pool.submit(_source(clock))
            except BaseException as error:
                errors.append(error)

        submitter = threading.Thread(target=submit_from_creator_thread)
        submitter.start()
        self.assertTrue(entered.wait(timeout=1))
        pool.close()
        resume.set()
        submitter.join(timeout=1)
        self.assertFalse(submitter.is_alive())

        deadline = time.monotonic() + 1
        while staged.releases == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        pool.close()

        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "closed during stage")
        self.assertEqual(adapter.build_calls, 0)
        self.assertEqual(adapter.abort_calls, 1)
        self.assertEqual(staged.releases, 1)

    def test_owner_release_is_deferred_until_lease_closes(self):
        clock = CacheCapsuleGeneration()
        with CacheCapsulePool(clock, enabled=True) as pool:
            receipt = pool.prepare(_source(clock), primary="cpu", fallback=None)
            lease = receipt.owner.lease()
            receipt.owner.release()

            payload = lease.payload_for_consumer(lambda value: None)
            self.assertEqual(payload.source_generation, 0)
            self.assertFalse(receipt.owner.released)
            with self.assertRaises(CacheCapsuleOwnerReleased):
                receipt.owner.lease()

            lease.close()
            self.assertTrue(receipt.owner.released)
            with self.assertRaises(CacheCapsuleOwnerReleased):
                lease.payload_for_consumer(lambda value: None)

    def test_external_backing_release_is_deferred_with_owner(self):
        class Backing:
            def __init__(self):
                self.releases = 0

            def release(self):
                self.releases += 1

        clock = CacheCapsuleGeneration()
        product = build_kv_cache_capsule_cpu(_source(clock))
        backing = Backing()
        owner = CacheCapsuleOwner(
            CacheCapsuleProduct(product.payload, backing_owner=backing),
            clock,
            clock.current,
            threading.get_ident(),
        )
        lease = owner.lease()

        owner.release()
        self.assertEqual(backing.releases, 0)
        lease.close()
        owner.release()

        self.assertEqual(backing.releases, 1)

    def test_numpy_uint16_transport_is_bit_exact(self):
        bits = np.array(
            [[[[0x0000, 0x3F80, 0xBF80, 0x7F7F, 0x0080]]]],
            dtype=np.uint16,
        )
        fingerprint = (bits.shape, bits.shape, "uint16", 1, 3)
        source = type(_source(CacheCapsuleGeneration()))(
            generation=7,
            source_id="raw-bf16",
            keys=bits,
            values=bits.copy(),
            offset=1,
            target_batch=3,
            layout_fingerprint=fingerprint,
            creator_thread=threading.get_ident(),
            raw_digest="raw-source-only",
        )

        product = build_kv_cache_capsule_cpu(source)

        np.testing.assert_array_equal(
            product.payload.keys, np.repeat(bits, 3, axis=0)
        )

    def test_generation_change_while_primary_runs_discards_handoff(self):
        clock = CacheCapsuleGeneration()
        release = threading.Event()
        entered = threading.Event()

        class Adapter(_SlowAdapter):
            def build(self, staged):
                entered.set()
                return super().build(staged)

        source = _source(clock)
        with CacheCapsulePool(
            clock,
            e5rt_adapter=Adapter(release, _e5rt_product(source)),
            enabled=True,
        ) as pool:
            ticket = pool.submit(source)
            self.assertTrue(entered.wait(timeout=5))
            clock.advance()
            with self.assertRaises(StaleCacheCapsule):
                ticket.await_adopt(timeout_s=1, fallback=None)
            release.set()

    def test_repeated_await_reuses_receipt_without_releasing_backing(self):
        class Backing:
            def __init__(self):
                self.releases = 0

            def release(self):
                self.releases += 1

        clock = CacheCapsuleGeneration()
        source = _source(clock)
        backing = Backing()
        product = _e5rt_product(source, backing)

        class Adapter:
            def stage(self, value):
                return product

            def build(self, staged):
                return staged

            def adopt(self, built, value):
                return built

        with CacheCapsulePool(
            clock, e5rt_adapter=Adapter(), enabled=True
        ) as pool:
            ticket = pool.submit(source)
            first = ticket.await_adopt(timeout_s=1, fallback=None)
            second = ticket.await_adopt(timeout_s=1, fallback=None)
            self.assertIs(first, second)
            self.assertEqual(backing.releases, 0)
            first.owner.release()
            self.assertEqual(backing.releases, 1)

    def test_timeout_keeps_circuit_busy_until_physical_job_finishes(self):
        clock = CacheCapsuleGeneration()
        source = _source(clock)
        release = threading.Event()
        adapter = _SlowAdapter(release, _e5rt_product(source))
        pool = CacheCapsulePool(clock, e5rt_adapter=adapter, enabled=True)
        try:
            ticket = pool.submit(source)
            fallback = ticket.await_adopt(timeout_s=0, fallback="cpu")
            with self.assertRaisesRegex(CacheCapsuleError, "circuit_busy"):
                pool.submit(source)
            fallback.owner.release()
            release.set()
            deadline = time.monotonic() + 1
            while pool._active_ticket is not None and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertIsNone(pool._active_ticket)
        finally:
            release.set()
            pool.close()

    def test_stale_generation_preserves_backing_until_all_leases_close(self):
        class Backing:
            def __init__(self):
                self.releases = 0

            def release(self):
                self.releases += 1

        clock = CacheCapsuleGeneration()
        product = build_kv_cache_capsule_cpu(_source(clock))
        backing = Backing()
        owner = CacheCapsuleOwner(
            CacheCapsuleProduct(product.payload, backing_owner=backing),
            clock,
            clock.current,
            threading.get_ident(),
        )
        first = owner.lease()
        second = owner.lease()
        clock.advance()
        with self.assertRaises(StaleCacheCapsule):
            first.payload_for_consumer(lambda value: None)
        self.assertEqual(backing.releases, 0)
        first.close()
        self.assertEqual(backing.releases, 0)
        second.close()
        self.assertEqual(backing.releases, 1)

    def test_concurrent_lease_close_is_idempotent(self):
        clock = CacheCapsuleGeneration()
        with CacheCapsulePool(clock, enabled=True) as pool:
            receipt = pool.prepare(_source(clock), primary="cpu", fallback=None)
            lease = receipt.owner.lease()
            receipt.owner.release()
            threads = [threading.Thread(target=lease.close) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=1)
            self.assertTrue(receipt.owner.released)


if __name__ == "__main__":
    unittest.main()
