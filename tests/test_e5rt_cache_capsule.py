import threading
import unittest
from unittest import mock

import numpy as np
import mlx.core as mx

from mlx_lm.cache_capsule import CacheCapsuleGeneration, CacheCapsulePool
from mlx_lm.e5rt_cache_capsule import (
    E5RTCacheCapsuleAdapter,
    E5RTCacheCapsuleSpec,
    PINNED_ANEFORGE_REVISION,
)


def test_real_mlx_dtype_names_are_accepted():
    from mlx_lm.e5rt_cache_capsule import _dtype_name

    assert _dtype_name(mx.zeros((1,), dtype=mx.float16)) == "float16"
    assert _dtype_name(mx.zeros((1,), dtype=mx.bfloat16)) == "bfloat16"
from mlx_lm.cache_capsule import KVCachePlaneSource


class _FakeProgram:
    def __init__(self, shape, target_batch):
        self.inputs = {
            "keys": np.empty(shape, dtype=np.float16),
            "values": np.empty(shape, dtype=np.float16),
        }
        output_shape = (target_batch, *shape[1:])
        self.outputs = {
            "keys": np.empty(output_shape, dtype=np.float16),
            "values": np.empty(output_shape, dtype=np.float16),
        }
        self.execute_threads = []
        self.release_calls = 0

    def input_view(self, name):
        return self.inputs[name]

    def output_view(self, name):
        return self.outputs[name]

    def execute(self):
        self.execute_threads.append(threading.get_ident())
        repeats = self.outputs["keys"].shape[0]
        for name in ("keys", "values"):
            np.copyto(self.outputs[name], np.repeat(self.inputs[name], repeats, axis=0))

    def release(self):
        self.release_calls += 1


def _source(shape=(1, 2, 5, 3), target_batch=2):
    keys = np.arange(np.prod(shape), dtype=np.float16).reshape(shape)
    values = keys + np.float16(100)
    fingerprint = (shape, shape, "float16", shape[2], target_batch)
    return KVCachePlaneSource(
        generation=7,
        source_id="apc:test:plane:0",
        keys=keys,
        values=values,
        offset=shape[2],
        target_batch=target_batch,
        layout_fingerprint=fingerprint,
        creator_thread=threading.get_ident(),
        raw_digest=None,
    )


class TestE5RTCacheCapsuleAdapter(unittest.TestCase):
    def _adapter(self, source=None):
        source = source or _source()
        spec = E5RTCacheCapsuleSpec.from_source(source)
        program = _FakeProgram(spec.source_shape, spec.target_batch)
        adapter = E5RTCacheCapsuleAdapter(
            program,
            spec,
            adopt_array=lambda view, dtype: view,
        )
        return adapter, program

    def test_module_pin_is_explicit(self):
        self.assertEqual(
            PINNED_ANEFORGE_REVISION,
            "026de27ea57b1fe42b608821d7b76ee9a9a66494",
        )

    def test_stage_build_adopt_transports_both_planes_exactly(self):
        source = _source(target_batch=3)
        adapter, program = self._adapter(source)
        caller = threading.get_ident()

        staged = adapter.stage(source)
        built_holder = []
        worker = threading.Thread(
            target=lambda: built_holder.append(adapter.build(staged))
        )
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        product = adapter.adopt(built_holder[0], source)

        np.testing.assert_array_equal(
            product.payload.keys, np.repeat(source.keys, 3, axis=0)
        )
        np.testing.assert_array_equal(
            product.payload.values, np.repeat(source.values, 3, axis=0)
        )
        self.assertNotEqual(program.execute_threads, [caller])
        self.assertEqual(adapter.state, "leased")
        self.assertEqual(adapter.counters["executes"], 1)
        self.assertGreater(adapter.timing_ns["stage"], 0)

        product.backing_owner.release()
        self.assertEqual(adapter.state, "idle")
        adapter.close()
        self.assertEqual(program.release_calls, 1)

    def test_stage_preserves_arbitrary_raw_16_bit_patterns(self):
        source = _source(shape=(1, 1, 1, 8))
        source.keys[:] = np.array(
            [0x0000, 0x3F80, 0xBF80, 0x7F7F, 0x0080, 0xFFFF, 0x7FC1, 0x8000],
            dtype=np.uint16,
        ).view(np.float16)
        source.values[:] = source.keys[..., ::-1]
        adapter, program = self._adapter(source)

        staged = adapter.stage(source)

        np.testing.assert_array_equal(
            program.inputs["keys"].view(np.uint16), source.keys.view(np.uint16)
        )
        np.testing.assert_array_equal(
            program.inputs["values"].view(np.uint16), source.values.view(np.uint16)
        )
        adapter.discard(adapter.build(staged))
        adapter.close()

    def test_pool_holds_output_slot_until_owner_and_lease_close(self):
        source = _source()
        adapter, program = self._adapter(source)
        clock = CacheCapsuleGeneration(initial=source.generation)
        with CacheCapsulePool(
            clock, e5rt_adapter=adapter, enabled=True
        ) as pool:
            receipt = pool.prepare(source, primary="e5rt", fallback=None)
            lease = receipt.owner.lease()
            payload = lease.payload_for_consumer(lambda value: None)
            np.testing.assert_array_equal(
                payload.keys, np.repeat(source.keys, 2, axis=0)
            )
            with self.assertRaisesRegex(Exception, "adapter_busy"):
                pool.submit(source)

            receipt.owner.release()
            self.assertEqual(adapter.state, "leased")
            lease.close()
            self.assertEqual(adapter.state, "idle")

        adapter.close()
        self.assertEqual(program.release_calls, 1)

    def test_close_defers_program_release_until_backing_release(self):
        source = _source()
        adapter, program = self._adapter(source)
        staged = adapter.stage(source)
        built = adapter.build(staged)
        product = adapter.adopt(built, source)

        adapter.close()
        self.assertEqual(program.release_calls, 0)
        self.assertEqual(adapter.state, "leased")

        product.backing_owner.release()
        self.assertEqual(program.release_calls, 1)

    def test_discard_reopens_slot_after_cancelled_physical_completion(self):
        source = _source()
        adapter, program = self._adapter(source)
        staged = adapter.stage(source)
        adapter.cancel(staged, "deadline")
        built = adapter.build(staged)
        adapter.discard(built)

        self.assertEqual(adapter.state, "idle")
        self.assertEqual(adapter.counters["cancels"], 1)
        self.assertEqual(adapter.counters["discards"], 1)
        adapter.close()
        self.assertEqual(program.release_calls, 1)

    def test_abort_releases_unbuilt_staged_slot_and_closed_program_once(self):
        source = _source()
        adapter, program = self._adapter(source)
        staged = adapter.stage(source)

        adapter.close()
        self.assertEqual(program.release_calls, 0)
        adapter.abort(staged, "pool_closed_during_stage")

        self.assertEqual(adapter.state, "idle")
        self.assertEqual(adapter.counters["cancels"], 1)
        self.assertEqual(program.release_calls, 1)
        adapter.abort(staged, "repeat")
        adapter.close()
        self.assertEqual(adapter.counters["cancels"], 1)
        self.assertEqual(program.release_calls, 1)

    def test_shape_and_dtype_mismatch_fail_before_execute(self):
        source = _source()
        adapter, program = self._adapter(source)
        wrong = _source(shape=(1, 2, 6, 3))
        with self.assertRaisesRegex(Exception, "shape_mismatch"):
            adapter.stage(wrong)
        self.assertEqual(program.execute_threads, [])
        adapter.close()

    def test_stage_copy_failure_racing_close_releases_program_exactly_once(self):
        source = _source()
        adapter, program = self._adapter(source)
        second_copy_entered = threading.Event()
        fail_copy = threading.Event()
        calls = 0
        errors = []

        def controlled_copy(destination, value):
            nonlocal calls
            calls += 1
            if calls == 2:
                second_copy_entered.set()
                fail_copy.wait(timeout=2)
                raise RuntimeError("injected stage-copy failure")
            np.copyto(destination.view(np.uint16), value.view(np.uint16))

        def run_stage():
            try:
                adapter.stage(source)
            except BaseException as error:
                errors.append(error)

        with mock.patch(
            "mlx_lm.e5rt_cache_capsule._copy_raw_bits",
            side_effect=controlled_copy,
        ):
            worker = threading.Thread(target=run_stage)
            worker.start()
            self.assertTrue(second_copy_entered.wait(timeout=1))
            adapter.close()
            self.assertEqual(program.release_calls, 0)
            fail_copy.set()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "stage-copy failure")
        self.assertEqual(adapter.state, "idle")
        self.assertEqual(adapter.counters["errors"], 1)
        self.assertEqual(program.release_calls, 1)
        adapter.close()
        self.assertEqual(program.release_calls, 1)


if __name__ == "__main__":
    unittest.main()
