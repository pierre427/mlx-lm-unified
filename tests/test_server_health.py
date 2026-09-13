"""Server-lifecycle tests for /health.

/health used to answer 200 unconditionally. On 2026-08-27 a wiring bug broke
generation while the endpoint kept reporting ok, so the process looked healthy
to its supervisor while every completion came back empty. These tests run a
real HTTP server and a real generation thread; no model is loaded.

The check reads thread liveness, not the serving path. It catches a generation
thread that exits; it does not catch generation that stays up and fails every
request. Only a check that runs a real completion sees that.
"""

import http.server
import json
import sys
import threading
import time
import types
import unittest
from pathlib import Path

import requests

from mlx_lm.server import (
    APIHandler,
    LRUPromptCache,
    ResponseGenerator,
    _run_http_server,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_soft_reload import make_cli_args  # noqa: E402


class StubModelProvider:
    """A provider that loads nothing. The generation thread idles on it."""

    def __init__(self, cli_args=None, load_error=None):
        self.cli_args = cli_args if cli_args is not None else make_cli_args()
        self.model = None
        self.model_key = None
        self.draft_model = None
        self.is_batchable = True
        self._load_error = load_error

    def load_default(self):
        if self._load_error is not None:
            raise self._load_error
        return None

    def load(self, model, adapter=None, draft_model=None):
        return None, None


class BlockingCrashProvider(StubModelProvider):
    """Hold startup until an HTTP waiter has registered, then fail."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def load_default(self):
        self.started.set()
        self.release.wait(timeout=10)
        raise RuntimeError("simulated Metal OOM")


class HealthServer:
    """A live APIHandler on a loopback port, wired to one generator."""

    def __init__(self, generator):
        self.generator = generator
        self.httpd = http.server.ThreadingHTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(generator, *args, **kwargs),
        )
        self.port = self.httpd.server_port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def get_health(self):
        response = requests.get(f"http://localhost:{self.port}/health", timeout=10)
        return response.status_code, json.loads(response.text)

    def get_json(self, path):
        response = requests.get(f"http://localhost:{self.port}{path}", timeout=10)
        return response.status_code, json.loads(response.text)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()


class TestHealthLiveness(unittest.TestCase):
    def setUp(self):
        self.generators = []
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.close()
        for generator in self.generators:
            generator._stop = True
            if generator._generation_thread.is_alive():
                generator._generation_thread.join(timeout=10)

    def start(self, **provider_args):
        generator = ResponseGenerator(
            StubModelProvider(**provider_args), LRUPromptCache()
        )
        self.generators.append(generator)
        server = HealthServer(generator)
        self.servers.append(server)
        return generator, server

    def test_idle_prompt_cache_hook_delegates_to_cache_tier(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        calls = []
        generator.prompt_cache = types.SimpleNamespace(
            spill_idle_entries=lambda: calls.append("spill") or 2
        )

        self.assertEqual(generator._spill_idle_prompt_cache(), 2)
        self.assertEqual(calls, ["spill"])

    def test_running_generation_is_200_ok(self):
        _generator, server = self.start()
        status, body = server.get_health()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    def test_decode_lane_and_cache_capsule_status_routes_both_respond(self):
        _generator, server = self.start()
        status, body = server.get_json("/v1/status/decode-lanes")
        self.assertEqual(status, 200)
        self.assertIn("requests", body)

        status, body = server.get_json("/v1/status/cache-capsules")
        self.assertEqual(status, 200)
        self.assertIn("enabled", body)
        self.assertIn("requests", body)

    def test_dead_generation_thread_is_503(self):
        # Stop the thread, then clear the stop flag: an exit nobody asked
        # for, which is what a crashed generator leaves behind.
        generator, server = self.start()
        generator.stop_and_join()
        generator._stop = False

        status, body = server.get_health()
        self.assertEqual(status, 503)
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["state"], "dead")
        self.assertFalse(generator.is_healthy)

    def test_crashed_generation_thread_is_503_and_names_the_error(self):
        generator, server = self.start(load_error=RuntimeError("no Stream(gpu, 0)"))
        deadline = time.monotonic() + 10
        while generator._generation_thread.is_alive():
            self.assertLess(time.monotonic(), deadline, "thread never exited")
            time.sleep(0.02)

        status, body = server.get_health()
        self.assertEqual(status, 503)
        self.assertIn("no Stream(gpu, 0)", body["reason"])

    def test_crash_wakes_registered_request_and_releases_inflight_slot(self):
        provider = BlockingCrashProvider()
        generator = ResponseGenerator(provider, LRUPromptCache())
        self.generators.append(generator)
        self.assertTrue(provider.started.wait(timeout=10))
        result = {}

        def request():
            try:
                generator.generate(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(request_id="waiting", tenant_id="default"),
                )
            except BaseException as error:
                result["error"] = error

        waiter = threading.Thread(target=request)
        waiter.start()
        deadline = time.monotonic() + 10
        while not generator._response_queues:
            self.assertLess(time.monotonic(), deadline, "request never registered")
            time.sleep(0.01)

        provider.release.set()
        waiter.join(timeout=10)

        self.assertFalse(waiter.is_alive(), "request stayed blocked after worker crash")
        self.assertIn("simulated Metal OOM", str(result["error"]))
        self.assertEqual(generator.inflight, 0)
        self.assertFalse(generator._response_queues)

    def test_new_request_is_rejected_after_generation_worker_crashes(self):
        generator, _server = self.start(load_error=RuntimeError("dead worker"))
        deadline = time.monotonic() + 10
        while generator._generation_thread.is_alive():
            self.assertLess(time.monotonic(), deadline, "thread never exited")
            time.sleep(0.02)

        with self.assertRaisesRegex(RuntimeError, "dead worker"):
            generator._admit_request("late", "default")
        self.assertEqual(generator.inflight, 0)

    def test_production_http_server_stops_when_generation_worker_is_dead(self):
        generator = ResponseGenerator(
            StubModelProvider(load_error=RuntimeError("fatal worker")),
            LRUPromptCache(),
        )
        self.generators.append(generator)
        generator._generation_thread.join(timeout=10)
        server = threading.Thread(
            target=_run_http_server,
            args=("127.0.0.1", 0, generator),
        )
        server.start()
        server.join(timeout=10)

        self.assertFalse(server.is_alive(), "HTTP shell survived its dead worker")

    def test_shutdown_is_not_reported_as_unhealthy(self):
        generator, server = self.start()
        generator.stop_and_join()

        status, body = server.get_health()
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["state"], "stopped")
        self.assertTrue(generator.is_healthy)

    def test_soft_reload_never_reports_unhealthy(self):
        generator, server = self.start()
        # Hold one request in flight so the reload parks at the drain wait
        # with the admission gate shut. That is the window a supervisor
        # would see, and it must not read as a fault.
        generator._admit_request("reload-holder", "default")

        done = threading.Event()
        errors = []

        def reload():
            try:
                generator.soft_reload({"self_mtp_num_draft": 4}, drain_timeout=30)
            except Exception as e:  # pragma: no cover - failure detail only
                errors.append(e)
            finally:
                done.set()

        thread = threading.Thread(target=reload)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            observed = []
            while not generator._paused:
                self.assertLess(time.monotonic(), deadline, "gate never closed")
                time.sleep(0.01)
            # Poll across the whole drained window.
            while generator._paused and time.monotonic() < deadline:
                observed.append(server.get_health())
                time.sleep(0.02)
            self.assertTrue(observed, "no health check ran while draining")
            for status, body in observed:
                self.assertEqual(status, 200)
                self.assertEqual(body["status"], "ok")
            states = [body.get("state") for _status, body in observed]
            self.assertIn("reloading", states)
        finally:
            generator._retire_request()
            self.assertTrue(done.wait(30))
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(generator.cli_args.self_mtp_num_draft, 4)
        status, body = server.get_health()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
