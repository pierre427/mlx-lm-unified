"""Real-fork lifecycle regressions with no MLX import or device work.

Execute the production class AST against NumPy array adapters. Descriptor,
mutex, executor, ownership and lookup control flow are the real source.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
import errno
import gc
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "mlx_lm/models/qwen4_ple_nvme.py"
NAMES = {
    "_before_fork", "_after_fork_parent", "_after_fork_child",
    "bf16_bits_to_f32", "f32_to_bf16_bits", "dequant_rows_numpy",
    "FileBackedShardedEmbedding",
}
tree = ast.parse(SOURCE.read_text())
nodes = [n for n in tree.body if getattr(n, "name", None) in NAMES]
# Register the actual module hooks once; none capture a table instance.
namespace = {
    "os": os, "threading": threading, "time": time, "np": np,
    "OrderedDict": OrderedDict, "ThreadPoolExecutor": ThreadPoolExecutor,
    "wait": wait, "nn": SimpleNamespace(Module=object),
    "mx": SimpleNamespace(array=np.array, bfloat16=np.uint16),
    "_ht": SimpleNamespace(STUB_PLE=False, ENABLED=False),
    "_lv": SimpleNamespace(bump=lambda *a: None),
    "PREFILL_ID_THRESHOLD": 512, "DECODE_WORKERS": 1,
    "PREFILL_WORKERS": 1, "PREFETCH_WORKERS": 1,
    "_FORK_RESOURCE_LOCK": threading.RLock(),
}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
if hasattr(os, "register_at_fork") and "_before_fork" in namespace:
    os.register_at_fork(before=namespace["_before_fork"],
                        after_in_parent=namespace["_after_fork_parent"],
                        after_in_child=namespace["_after_fork_child"])
Table = namespace["FileBackedShardedEmbedding"]


@unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
class ForkContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "ple_rows.bin"
        self.rows = np.zeros((8, 20), dtype=np.uint8)
        self.rows[:, :16] = 0x21
        self.rows[:, 16:18] = np.array([0x3F80], dtype=np.uint16).view(np.uint8)
        self.path.write_bytes(self.rows.tobytes())
        with patch.dict(os.environ, {"MLX_QWEN4_PLE_NVME_LRU_MB": "0.001",
                                     "MLX_QWEN4_PLE_NVME_DEQUANT": "numpy",
                                     "MLX_QWEN4_PLE_NVME_DECODE_WORKERS": "1",
                                     "MLX_QWEN4_PLE_NVME_PREFILL_WORKERS": "1"}):
            self.table = Table(str(self.path), 8, 32, 2)
        self.addCleanup(self.table.close)
        self.ids = np.array([0, 1], dtype=np.int64)
        self.expected = self.table.lookup_numpy(self.ids)

    def child(self, fn, held_lock=None):
        acquired, release = threading.Event(), threading.Event()
        thread = None
        if held_lock is not None:
            def hold():
                with held_lock:
                    acquired.set()
                    release.wait(10)
            thread = threading.Thread(target=hold)
            thread.start()
            self.assertTrue(acquired.wait(2))
        read_fd, write_fd = os.pipe()
        try:
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                signal.signal(signal.SIGALRM, lambda *a: os._exit(124))
                signal.alarm(4)
                try:
                    fn()
                except BaseException as exc:
                    os.write(write_fd, repr(exc).encode()[:2048])
                    os._exit(1)
                os._exit(0)
            os.close(write_fd)
            write_fd = None
            _, status = os.waitpid(pid, 0)
            detail = os.read(read_fd, 2048).decode()
            self.assertEqual(os.waitstatus_to_exitcode(status), 0, detail or "child timed out")
        finally:
            release.set()
            if thread is not None:
                thread.join(2)
                self.assertFalse(thread.is_alive())
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def assert_closed_fd(self, fd):
        try:
            os.fstat(fd)
        except OSError as exc:
            self.assertEqual(exc.errno, errno.EBADF)
        else:
            self.fail("descriptor still open")

    def test_lookup_reopens_before_every_inherited_mutex_and_staged_hit(self):
        table = self.table
        parent_fd, parent_pid = table._fd, table._owner_pid
        # Poison staged rows: a complete hit must also reset before use.
        table._dq_cache = {int(i): np.zeros(32, np.uint16) for i in self.ids}
        for name in ("_lifecycle_lock", "_lru_lock", "_dq_lock"):
            with self.subTest(lock=name):
                def lookup():
                    np.testing.assert_array_equal(table.lookup_numpy(self.ids), self.expected)
                    self.assertEqual(table._owner_pid, os.getpid())
                    self.assert_closed_fd(parent_fd)
                    child_fd = table._fd
                    table.close()
                    table.close()
                    self.assert_closed_fd(child_fd)
                    self.assertFalse(table._dq_cache)
                self.child(lookup, getattr(table, name))
        self.assertEqual(table._owner_pid, parent_pid)
        os.fstat(parent_fd)
        table._dq_cache.clear()
        np.testing.assert_array_equal(table.lookup_numpy(self.ids), self.expected)

    def test_direct_submission_recovers_before_lifecycle_lock(self):
        def submit():
            futures = self.table._submit(False, [lambda fd: os.pread(fd, 20, 0)], True)
            self.assertEqual(futures[0].result(timeout=1), self.rows[0].tobytes())
            self.table.close()
        self.child(submit, self.table._lifecycle_lock)

    def test_child_close_never_joins_inherited_pools_and_is_idempotent(self):
        table = self.table
        fd, pid = table._fd, os.getpid()
        original = [table._pool.shutdown, table._prefetch_pool.shutdown]
        def guard(index):
            def shutdown(*args, **kwargs):
                if os.getpid() != pid:
                    raise AssertionError("joined inherited executor")
                return original[index](*args, **kwargs)
            return shutdown
        with patch.object(table._pool, "shutdown", guard(0)), patch.object(table._prefetch_pool, "shutdown", guard(1)):
            for name in ("_lifecycle_lock", "_lru_lock", "_dq_lock"):
                with self.subTest(lock=name):
                    def close():
                        table.close()
                        table.close()
                        self.assert_closed_fd(fd)
                        self.assertFalse(table._lru)
                        with self.assertRaisesRegex(RuntimeError, "closed"):
                            table.lookup_numpy(self.ids)
                    self.child(close, getattr(table, name))
        os.fstat(fd)

    def test_concurrent_child_first_use_reopens_once(self):
        table = self.table
        def run():
            count = []
            original = table._open_resources
            def opened():
                count.append(1)
                time.sleep(0.02)
                original()
            table._open_resources = opened
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(table.lookup_numpy, self.ids) for _ in range(2)]
                for future in futures:
                    np.testing.assert_array_equal(future.result(timeout=2), self.expected)
            self.assertEqual(len(count), 1)
            table.close()
        self.child(run, table._lifecycle_lock)

    def test_closed_table_stays_closed_after_fork(self):
        self.table.close()
        def run():
            with patch.object(self.table, "_open_resources", side_effect=AssertionError("reopened closed table")):
                self.table.close()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    self.table.lookup_numpy(self.ids)
        self.child(run)

    def test_concurrent_child_close_closes_descriptor_once(self):
        table = self.table
        fd = table._fd
        def run():
            calls = []
            original = os.close
            def close(value):
                if value == fd:
                    calls.append(value)
                    time.sleep(0.02)
                return original(value)
            with patch.object(os, "close", close), ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(table.close) for _ in range(2)]
                for future in futures:
                    future.result(timeout=2)
            self.assertEqual(calls, [fd])
        self.child(run, table._lifecycle_lock)

    def test_fork_waits_for_close_descriptor_publication(self):
        table, fd = self.table, self.table._fd
        closed = threading.Event()
        original = os.close
        def close(value):
            result = original(value)
            if value == fd:
                closed.set()
                time.sleep(0.05)
            return result
        with patch.object(os, "close", close):
            thread = threading.Thread(target=table.close)
            thread.start()
            self.assertTrue(closed.wait(2))
            def run():
                self.assertIsNone(table._fd)
                other = os.open(self.path, os.O_RDONLY)
                try:
                    table.close()
                    os.fstat(other)
                finally:
                    original(other)
            self.child(run)
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_close_reenters_resource_guard_during_finalization(self):
        # Cyclic GC may finalize another table during resource allocation.
        with namespace["_FORK_RESOURCE_LOCK"]:
            self.table.close()

    def test_registration_does_not_retain_instances(self):
        table = Table(str(self.path), 8, 32, 2)
        ref = weakref.ref(table)
        table.close()
        del table
        gc.collect()
        self.assertIsNone(ref())


if __name__ == "__main__":
    unittest.main()
