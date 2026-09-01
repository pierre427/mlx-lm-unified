# Copyright © 2026 Apple Inc.

"""Tests for in-place serving reconfiguration (soft reload).

Soft reload keeps the loaded weights resident, so it must not be used for
promotion-grade A/B measurement: the A/B harness runs one process per arm on
purpose, and that isolates the allocator, the compilation cache, module
constants and KV between arms. These tests cover the serving contract only.
"""

import io
import json
import sys
import threading
import time
import types
import unittest
from queue import Queue
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.apc import AutomaticPrefixCache, MTPAPCSidecar
from mlx_lm.models.cache import KVCache
from mlx_lm.server import (
    EFFECTIVE_CONFIG_PATH,
    SOFT_RELOAD_KEYS,
    SOFT_RELOAD_PATH,
    SOFT_RELOAD_RESTART_KEYS,
    APIHandler,
    MutableKey,
    ResponseGenerator,
    SoftReloadBusy,
    SoftReloadError,
    SoftReloadRestartRequired,
    apply_soft_reload,
    plan_soft_reload,
    read_effective_config,
)


def make_cli_args(**overrides):
    values = dict(
        model="/models/qwen3.8-flash-next",
        adapter_path=None,
        draft_model=None,
        trust_remote_code=False,
        single_model=False,
        allowed_origins=["*"],
        soft_reload_key=None,
        kv_bits=None,
        decode_concurrency=8,
        self_mtp=False,
        self_mtp_num_draft=1,
        self_mtp_adaptive_depth_ceiling=None,
        self_mtp_persistent=True,
        self_mtp_rate_gate=True,
        self_mtp_transformed_verifier=False,
        self_mtp_share_qsa_indices=False,
        self_mtp_share_qsa_indices_min_prompt_tokens=0,
        self_mtp_window_size=0,
        self_mtp_window_sink_size=4,
        self_mtp_window_min_prompt_tokens=0,
        self_mtp_apc_retain_min_prompt_tokens=64,
        num_draft_tokens=3,
        prompt_lookup_ngram=0,
        prompt_lookup_tokens=8,
        prompt_lookup_adaptive=True,
        prompt_lookup_rate_gate=True,
        prompt_lookup_warmup=48,
        prompt_lookup_gate=0.12,
        prompt_lookup_rate_gate_probe=32,
        prompt_lookup_rate_gate_margin=0.0,
        temp=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=512,
        log_level="INFO",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_generator(cli_args=None, prompt_cache=None):
    """A ResponseGenerator with no generation thread and no model."""
    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.model_provider = types.SimpleNamespace(
        cli_args=cli_args if cli_args is not None else make_cli_args(),
        model=object(),
        model_key=("model", None, None),
    )
    gen.prompt_cache = prompt_cache if prompt_cache is not None else stored_apc()
    gen.requests = Queue()
    gen._admission = threading.Condition()
    gen._paused = False
    gen._inflight = 0
    gen._reload_lock = threading.Lock()
    gen.admission_timeout = 5.0
    return gen


def tiny_cache(tokens=4):
    cache = [KVCache()]
    cache[0].update_and_fetch(
        mx.zeros((1, 1, tokens, 2)), mx.zeros((1, 1, tokens, 2))
    )
    return cache


def stored_apc(n_entries=3, with_sidecar=True):
    apc = AutomaticPrefixCache(max_size=16)
    for i in range(n_entries):
        sidecar = None
        if with_sidecar and i == 0:
            sidecar = MTPAPCSidecar(
                state=(tiny_cache(2), mx.zeros((1, 1, 4))), covered_tokens=2
            )
        apc.store(
            ("model", None, None),
            [i + 1, i + 2, i + 3],
            tiny_cache(),
            sidecar=sidecar,
        )
    return apc


class TestAPCClear(unittest.TestCase):
    def test_clear_empties_entries_sidecars_and_bytes(self):
        apc = stored_apc()
        self.assertEqual(len(apc), 3)
        self.assertGreater(apc.nbytes, 0)

        report = apc.clear()

        self.assertEqual(report["entries"], 3)
        self.assertEqual(report["sidecars"], 1)
        self.assertGreater(report["bytes"], 0)
        self.assertEqual(len(apc), 0)
        self.assertEqual(apc.nbytes, 0)
        self.assertEqual(
            {t["n_sequences"] for t in apc.stats_by_type().values()}, {0}
        )

    def test_cleared_prefix_is_dropped_not_reused(self):
        apc = stored_apc(n_entries=1, with_sidecar=False)
        self.assertTrue(apc.lookup(("model", None, None), [1, 2, 3]).hit)

        apc.clear()

        result = apc.lookup(("model", None, None), [1, 2, 3])
        self.assertFalse(result.hit)
        self.assertIsNone(result.cache)
        self.assertEqual(result.remaining_tokens, [1, 2, 3])

    def test_stats_restart_but_lifetime_totals_are_kept(self):
        apc = stored_apc(n_entries=1, with_sidecar=False)
        apc.lookup(("model", None, None), [1, 2, 3])

        before = apc.apc_stats
        self.assertEqual(before["stores"], 1)
        self.assertEqual(before["hits"], 1)
        self.assertEqual(before["clears"], 0)

        apc.clear()

        after = apc.apc_stats
        self.assertEqual(after["hits"], 0)
        self.assertEqual(after["stores"], 0)
        self.assertEqual(after["clears"], 1)
        self.assertEqual(after["lifetime"]["hits"], 1)
        self.assertEqual(after["lifetime"]["stores"], 1)

    def test_clear_releases_buffers_by_default(self):
        apc = stored_apc(n_entries=1, with_sidecar=False)
        with patch("mlx_lm.apc.mx.clear_cache") as clear_cache:
            apc.clear()
        clear_cache.assert_called_once_with()

        apc = stored_apc(n_entries=1, with_sidecar=False)
        with patch("mlx_lm.apc.mx.clear_cache") as clear_cache:
            apc.clear(release_memory=False)
        clear_cache.assert_not_called()


class TestWhitelist(unittest.TestCase):
    def test_unknown_key_is_refused(self):
        cli_args = make_cli_args()
        with self.assertRaisesRegex(SoftReloadError, "not a soft-reloadable key"):
            plan_soft_reload(cli_args, {"not_a_real_key": 1})

    def test_existing_attribute_outside_the_registry_is_still_refused(self):
        """The route is not a remote setattr: being an attribute is not enough."""
        cli_args = make_cli_args()
        self.assertTrue(hasattr(cli_args, "log_level"))
        with self.assertRaisesRegex(SoftReloadError, "not a soft-reloadable key"):
            plan_soft_reload(cli_args, {"log_level": "DEBUG"})
        self.assertEqual(cli_args.log_level, "INFO")

    def test_dunder_attribute_is_refused(self):
        with self.assertRaises(SoftReloadError):
            plan_soft_reload(make_cli_args(), {"__class__": "x"})

    def test_type_and_range_are_validated(self):
        cli_args = make_cli_args()
        for config, pattern in [
            ({"self_mtp": "yes"}, "expected a boolean"),
            ({"self_mtp_num_draft": 1.5}, "expected an integer"),
            ({"self_mtp_num_draft": 0}, "outside"),
            ({"self_mtp_num_draft": 10_000}, "outside"),
            ({"prompt_lookup_gate": 2.0}, "outside"),
            ({"prompt_lookup_gate": "0.5"}, "expected a number"),
            ({"top_k": True}, "expected an integer"),
        ]:
            with self.subTest(config=config):
                with self.assertRaisesRegex(SoftReloadError, pattern):
                    plan_soft_reload(cli_args, config)

    def test_config_must_be_an_object(self):
        with self.assertRaisesRegex(SoftReloadError, "must be a JSON object"):
            plan_soft_reload(make_cli_args(), [("self_mtp", True)])

    def test_valid_plan_reports_old_and_new(self):
        cli_args = make_cli_args()
        plan = plan_soft_reload(cli_args, {"self_mtp_num_draft": 4})
        # Planning alone must not write anything.
        self.assertEqual(cli_args.self_mtp_num_draft, 1)
        changes = apply_soft_reload(cli_args, plan)
        self.assertEqual(changes, {"self_mtp_num_draft": {"old": 1, "new": 4}})
        self.assertEqual(cli_args.self_mtp_num_draft, 4)

    def test_module_lever_targets_the_registered_module_only(self):
        fake = types.ModuleType("fake_model_module")
        fake._LEVER = False
        registry = dict(SOFT_RELOAD_KEYS)
        registry["fake_lever"] = MutableKey(
            "module", "_LEVER", SOFT_RELOAD_KEYS["self_mtp"].validate,
            "fake_model_module",
        )
        cli_args = make_cli_args()
        with patch.dict(sys.modules, {"fake_model_module": fake}):
            with patch.dict(SOFT_RELOAD_KEYS, registry, clear=True):
                self.assertIsNone(
                    read_effective_config(cli_args).get("fake_lever_missing")
                )
                self.assertFalse(read_effective_config(cli_args)["fake_lever"])
                plan = plan_soft_reload(cli_args, {"fake_lever": True})
                changes = apply_soft_reload(cli_args, plan)
        self.assertEqual(changes, {"fake_lever": {"old": False, "new": True}})
        self.assertTrue(fake._LEVER)
        self.assertFalse(hasattr(cli_args, "_LEVER"))

    def test_qwen4_nax_lever_accepts_auto_and_boolean_modes(self):
        from mlx_lm.models import qwen4_exp

        cli_args = make_cli_args()
        original = qwen4_exp._QSA_NAX_KERNEL
        try:
            qwen4_exp._QSA_NAX_KERNEL = False
            plan = plan_soft_reload(cli_args, {"qwen4_qsa_nax_kernel": "auto"})
            changes = apply_soft_reload(cli_args, plan)
            self.assertEqual(
                changes,
                {"qwen4_qsa_nax_kernel": {"old": False, "new": None}},
            )
            self.assertIsNone(qwen4_exp._QSA_NAX_KERNEL)

            plan = plan_soft_reload(cli_args, {"qwen4_qsa_nax_kernel": True})
            apply_soft_reload(cli_args, plan)
            self.assertTrue(qwen4_exp._QSA_NAX_KERNEL)
            with self.assertRaisesRegex(SoftReloadError, "expected a boolean"):
                plan_soft_reload(cli_args, {"qwen4_qsa_nax_kernel": "sometimes"})
        finally:
            qwen4_exp._QSA_NAX_KERNEL = original

    def test_qwen4_stage1_lever_is_independent(self):
        from mlx_lm.models import qwen4_exp

        cli_args = make_cli_args()
        original = qwen4_exp._QSA_STAGE1_KERNEL
        try:
            qwen4_exp._QSA_STAGE1_KERNEL = False
            plan = plan_soft_reload(cli_args, {"qwen4_qsa_stage1_kernel": True})
            changes = apply_soft_reload(cli_args, plan)
            self.assertEqual(
                changes,
                {"qwen4_qsa_stage1_kernel": {"old": False, "new": True}},
            )
            self.assertTrue(qwen4_exp._QSA_STAGE1_KERNEL)

            plan = plan_soft_reload(
                cli_args, {"qwen4_qsa_stage1_kernel": "auto"}
            )
            changes = apply_soft_reload(cli_args, plan)
            self.assertEqual(
                changes,
                {"qwen4_qsa_stage1_kernel": {"old": True, "new": None}},
            )
            self.assertIsNone(qwen4_exp._QSA_STAGE1_KERNEL)

            plan = plan_soft_reload(cli_args, {"qwen4_qsa_stage1_kernel": False})
            changes = apply_soft_reload(cli_args, plan)
            self.assertEqual(
                changes,
                {"qwen4_qsa_stage1_kernel": {"old": None, "new": False}},
            )
            self.assertFalse(qwen4_exp._QSA_STAGE1_KERNEL)
        finally:
            qwen4_exp._QSA_STAGE1_KERNEL = original


class TestHardTierRefusal(unittest.TestCase):
    def test_model_path_is_refused_with_a_restart_message(self):
        cli_args = make_cli_args()
        with self.assertRaisesRegex(SoftReloadRestartRequired, "restart the server"):
            plan_soft_reload(cli_args, {"model": "/models/other"})
        self.assertEqual(cli_args.model, "/models/qwen3.8-flash-next")

    def test_every_restart_key_is_refused_and_never_silently_accepted(self):
        for key in SOFT_RELOAD_RESTART_KEYS:
            with self.subTest(key=key):
                self.assertNotIn(key, SOFT_RELOAD_KEYS)
                with self.assertRaises(SoftReloadRestartRequired):
                    plan_soft_reload(make_cli_args(), {key: 1})

    def test_structural_moe_lever_is_restart_tier(self):
        """_MOE_SHARED_IN_GATHER is read in __init__, so the built model
        already carries the layout; flipping it in place would be a no-op."""
        with self.assertRaisesRegex(SoftReloadRestartRequired, "layout"):
            plan_soft_reload(
                make_cli_args(), {"qwen4_moe_shared_in_gather": True}
            )

    def test_a_batch_with_one_bad_key_applies_nothing(self):
        cli_args = make_cli_args()
        with self.assertRaises(SoftReloadError):
            plan_soft_reload(
                cli_args, {"self_mtp_num_draft": 4, "model": "/models/other"}
            )
        self.assertEqual(cli_args.self_mtp_num_draft, 1)
        self.assertEqual(cli_args.model, "/models/qwen3.8-flash-next")


class TestDrainOrdering(unittest.TestCase):
    def test_mutation_waits_for_in_flight_generation(self):
        gen = make_generator()
        gen._admit_request()  # stand in for a request that is generating

        observed = []
        done = threading.Event()

        def reload():
            observed.append(("reload_start", gen.cli_args.self_mtp_num_draft))
            gen.soft_reload({"self_mtp_num_draft": 5}, drain_timeout=10)
            done.set()

        thread = threading.Thread(target=reload)
        thread.start()
        # Give the reload thread time to reach the drain wait; the value must
        # still be the old one while the request is in flight.
        time.sleep(0.2)
        observed.append(("while_inflight", gen.cli_args.self_mtp_num_draft))
        self.assertEqual(len(gen.prompt_cache), 3)
        self.assertFalse(done.is_set())

        gen._retire_request()
        self.assertTrue(done.wait(10))
        thread.join()

        self.assertEqual(observed[1], ("while_inflight", 1))
        self.assertEqual(gen.cli_args.self_mtp_num_draft, 5)
        self.assertEqual(len(gen.prompt_cache), 0)

    def test_drain_timeout_leaves_config_and_cache_untouched(self):
        gen = make_generator()
        gen._admit_request()
        try:
            with self.assertRaisesRegex(SoftReloadBusy, "did not drain"):
                gen.soft_reload({"self_mtp_num_draft": 5}, drain_timeout=0.05)
        finally:
            gen._retire_request()
        self.assertEqual(gen.cli_args.self_mtp_num_draft, 1)
        self.assertEqual(len(gen.prompt_cache), 3)
        # The gate must reopen even after a failed reload.
        self.assertFalse(gen._paused)
        gen._admit_request()
        gen._retire_request()

    def test_new_requests_wait_at_the_closed_gate(self):
        gen = make_generator()
        gen._paused = True
        gen.admission_timeout = 0.05
        with self.assertRaises(SoftReloadBusy):
            gen._admit_request()

        admitted = threading.Event()

        def waiter():
            gen._admit_request(timeout=10)
            admitted.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.1)
        self.assertFalse(admitted.is_set())
        with gen._admission:
            gen._paused = False
            gen._admission.notify_all()
        self.assertTrue(admitted.wait(10))
        thread.join()
        gen._retire_request()

    def test_generate_releases_its_slot_when_the_stream_finishes(self):
        gen = make_generator()

        def responder():
            rqueue, _request, _args = gen.requests.get(timeout=10)
            rqueue.put(types.SimpleNamespace(prompt=[1]))
            rqueue.put(None)

        thread = threading.Thread(target=responder)
        thread.start()
        _ctx, stream = gen.generate(object(), types.SimpleNamespace())
        self.assertEqual(gen.inflight, 1)
        self.assertEqual(list(stream), [])
        self.assertEqual(gen.inflight, 0)
        thread.join()

    def test_generate_releases_its_slot_when_admission_fails_upstream(self):
        gen = make_generator()

        def responder():
            rqueue, _request, _args = gen.requests.get(timeout=10)
            rqueue.put(ValueError("no such model"))

        thread = threading.Thread(target=responder)
        thread.start()
        with self.assertRaises(ValueError):
            gen.generate(object(), types.SimpleNamespace())
        self.assertEqual(gen.inflight, 0)
        thread.join()

    def test_concurrent_reload_is_refused_rather_than_interleaved(self):
        gen = make_generator()
        gen._reload_lock.acquire()
        try:
            with self.assertRaisesRegex(SoftReloadBusy, "already in progress"):
                gen.soft_reload({"self_mtp_num_draft": 5})
        finally:
            gen._reload_lock.release()
        self.assertEqual(gen.cli_args.self_mtp_num_draft, 1)


class TestSoftReloadReport(unittest.TestCase):
    def test_report_names_each_change_and_the_dropped_entries(self):
        gen = make_generator()
        report = gen.soft_reload(
            {"self_mtp_num_draft": 5, "prompt_lookup_ngram": 0}
        )
        self.assertEqual(
            report["changed"], {"self_mtp_num_draft": {"old": 1, "new": 5}}
        )
        self.assertEqual(report["unchanged"], ["prompt_lookup_ngram"])
        self.assertEqual(report["prompt_cache"]["entries_dropped"], 3)
        self.assertEqual(report["prompt_cache"]["sidecars_dropped"], 1)
        self.assertGreater(report["prompt_cache"]["bytes_freed"], 0)
        self.assertEqual(report["drain"]["inflight_at_start"], 0)
        self.assertEqual(report["effective_config"]["self_mtp_num_draft"], 5)

    def test_plain_lru_prompt_cache_is_emptied_too(self):
        from mlx_lm.models.cache import LRUPromptCache

        cache = LRUPromptCache(max_size=8)
        cache.insert_cache(("model", None, None), [1, 2, 3], tiny_cache())
        gen = make_generator(prompt_cache=cache)
        report = gen.soft_reload({"self_mtp_num_draft": 2})
        self.assertEqual(report["prompt_cache"]["entries_dropped"], 1)
        self.assertEqual(len(cache), 0)


class StubHandler:
    """APIHandler wired to byte buffers, with the status code captured."""

    @staticmethod
    def build(generator, path, body=None, headers=None):
        handler = APIHandler.__new__(APIHandler)
        handler.path = path
        handler.created = 0
        handler.response_generator = generator
        handler.headers = dict(headers or {})
        payload = b"" if body is None else json.dumps(body).encode()
        handler.headers.setdefault("Content-Length", str(len(payload)))
        handler.rfile = io.BytesIO(payload)
        handler.wfile = io.BytesIO()
        handler.status = []
        handler._set_completion_headers = lambda code=200: handler.status.append(code)
        handler.end_headers = lambda: None
        return handler

    @staticmethod
    def result(handler):
        raw = handler.wfile.getvalue()
        body = json.loads(raw.decode()) if raw else None
        return handler.status[-1] if handler.status else None, body


class TestAdminRoutes(unittest.TestCase):
    def post(self, generator, body, headers=None):
        handler = StubHandler.build(generator, SOFT_RELOAD_PATH, body, headers)
        handler.do_POST()
        return StubHandler.result(handler)

    def get_config(self, generator, headers=None):
        handler = StubHandler.build(generator, EFFECTIVE_CONFIG_PATH, None, headers)
        handler.do_GET()
        return StubHandler.result(handler)

    def test_routes_are_disabled_without_a_configured_key(self):
        gen = make_generator(make_cli_args(soft_reload_key=None))
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("MLX_LM_SOFT_RELOAD_KEY", None)
            status, body = self.post(gen, {"config": {"self_mtp_num_draft": 4}})
            self.assertEqual(status, 404)
            self.assertIn("disabled", body["error"])
            self.assertEqual(self.get_config(gen)[0], 404)
        self.assertEqual(gen.cli_args.self_mtp_num_draft, 1)

    def test_wrong_key_is_rejected(self):
        gen = make_generator(make_cli_args(soft_reload_key="secret"))
        status, body = self.post(
            gen,
            {"config": {"self_mtp_num_draft": 4}},
            headers={"Authorization": "Bearer wrong"},
        )
        self.assertEqual(status, 401)
        self.assertEqual(gen.cli_args.self_mtp_num_draft, 1)

        status, _ = self.post(gen, {"config": {"self_mtp_num_draft": 4}})
        self.assertEqual(status, 401)
        self.assertEqual(gen.cli_args.self_mtp_num_draft, 1)

    def test_correct_key_applies_and_reports(self):
        gen = make_generator(make_cli_args(soft_reload_key="secret"))
        status, body = self.post(
            gen,
            {"config": {"self_mtp_num_draft": 4}},
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["changed"]["self_mtp_num_draft"]["new"], 4)
        self.assertEqual(body["prompt_cache"]["entries_dropped"], 3)
        self.assertEqual(gen.cli_args.self_mtp_num_draft, 4)

    def test_environment_key_enables_the_route(self):
        import os

        gen = make_generator(make_cli_args(soft_reload_key=None))
        with patch.dict(os.environ, {"MLX_LM_SOFT_RELOAD_KEY": "envkey"}):
            status, _ = self.post(
                gen,
                {"config": {"self_mtp_num_draft": 4}},
                headers={"Authorization": "Bearer envkey"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(gen.cli_args.self_mtp_num_draft, 4)

    def test_unknown_key_is_a_400_and_hard_tier_is_a_409(self):
        gen = make_generator(make_cli_args(soft_reload_key="secret"))
        auth = {"Authorization": "Bearer secret"}
        status, body = self.post(gen, {"config": {"nope": 1}}, headers=auth)
        self.assertEqual(status, 400)
        self.assertIn("not a soft-reloadable key", body["error"])

        status, body = self.post(gen, {"config": {"model": "/x"}}, headers=auth)
        self.assertEqual(status, 409)
        self.assertIn("restart the server", body["error"])
        self.assertEqual(gen.cli_args.model, "/models/qwen3.8-flash-next")
        # A refused request must not have dropped the cache either.
        self.assertEqual(len(gen.prompt_cache), 3)

    def test_bad_body_is_a_400(self):
        gen = make_generator(make_cli_args(soft_reload_key="secret"))
        auth = {"Authorization": "Bearer secret"}
        handler = StubHandler.build(gen, SOFT_RELOAD_PATH, None, auth)
        handler.rfile = io.BytesIO(b"{not json")
        handler.headers["Content-Length"] = "9"
        handler.do_POST()
        self.assertEqual(StubHandler.result(handler)[0], 400)

        status, _ = self.post(
            gen, {"config": {}, "drain_timeout": -1}, headers=auth
        )
        self.assertEqual(status, 400)

    def test_drain_timeout_surfaces_as_503(self):
        gen = make_generator(make_cli_args(soft_reload_key="secret"))
        gen._admit_request()
        try:
            status, body = self.post(
                gen,
                {"config": {"self_mtp_num_draft": 4}, "drain_timeout": 0.05},
                headers={"Authorization": "Bearer secret"},
            )
        finally:
            gen._retire_request()
        self.assertEqual(status, 503)
        self.assertIn("did not drain", body["error"])

    def test_effective_config_readback(self):
        gen = make_generator(make_cli_args(soft_reload_key="secret"))
        auth = {"Authorization": "Bearer secret"}
        self.post(gen, {"config": {"self_mtp_num_draft": 6}}, headers=auth)
        status, body = self.get_config(gen, headers=auth)
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "effective_config")
        self.assertEqual(body["effective_config"]["self_mtp_num_draft"], 6)
        self.assertIn("model", body["restart_required"])
        self.assertEqual(body["prompt_cache"]["entries"], 0)
        self.assertEqual(body["inflight"], 0)


class TestAdminRoutesOverHTTP(unittest.TestCase):
    """Same contract over a real socket, with real header parsing."""

    @classmethod
    def setUpClass(cls):
        import http.server

        cls.generator = make_generator(make_cli_args(soft_reload_key="s3cret"))
        cls.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(cls.generator, *args, **kwargs),
        )
        cls.base = f"http://localhost:{cls.httpd.server_port}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join()

    def test_round_trip(self):
        import requests

        auth = {"Authorization": "Bearer s3cret"}
        self.assertEqual(
            requests.post(
                self.base + SOFT_RELOAD_PATH, json={"config": {"self_mtp": True}}
            ).status_code,
            401,
        )
        response = requests.post(
            self.base + SOFT_RELOAD_PATH,
            json={"config": {"self_mtp": True}},
            headers=auth,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["changed"], {"self_mtp": {"old": False, "new": True}}
        )
        self.assertEqual(response.json()["prompt_cache"]["entries_dropped"], 3)

        self.assertEqual(
            requests.post(
                self.base + SOFT_RELOAD_PATH,
                json={"config": {"model": "/x"}},
                headers=auth,
            ).status_code,
            409,
        )
        readback = requests.get(self.base + EFFECTIVE_CONFIG_PATH, headers=auth)
        self.assertEqual(readback.status_code, 200)
        self.assertTrue(readback.json()["effective_config"]["self_mtp"])
        # The completion routes must be unaffected by the new admin paths.
        self.assertEqual(requests.get(self.base + "/health").status_code, 200)
        nax_status = requests.get(
            self.base + "/v1/status/qwen4-qsa-nax"
        )
        self.assertEqual(nax_status.status_code, 200)
        self.assertIn(nax_status.json()["mode"], {"auto", "on", "off"})
        self.assertEqual(nax_status.json()["auto_min_physical_kv"], 16_384)
        self.assertEqual(requests.get(self.base + "/v1/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
