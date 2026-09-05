# Copyright © 2026 Apple Inc.

"""CPU-only source contracts for compiled replay's fail-closed boundary.

This module deliberately does not import ``mlx`` or ``mlx_lm``.  It is safe to
run while Metal is unavailable or a host is awaiting reboot.
"""

import ast
import contextlib
import copy
import importlib.util
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]


def _source(path):
    return (ROOT / path).read_text()


def _definitions(relative, names, namespace):
    nodes = [
        n
        for n in ast.walk(ast.parse(_source(relative)))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    for node in nodes:
        node.decorator_list = []
        node.returns = None
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            argument.annotation = None
    exec(compile(ast.Module(nodes, []), relative, "exec"), namespace)


class TestCompiledServerAdmission(unittest.TestCase):
    def setUp(self):
        self.env = {
            "compiled_decode_enabled": lambda: True,
            "compiled_decode_context_policy": lambda *_: (None, object()),
            "compiled_decode_numerics_accepted": lambda _: True,
            "compiled_decode_serving_reason": lambda *_: None,
        }
        _definitions(
            "mlx_lm/server.py",
            {
                "_compiled_request_selected",
                "_is_batchable",
                "_generate",
                "_serve_request",
            },
            self.env,
        )
        self.cls = type(
            "Scheduler",
            (),
            {
                name: self.env[name]
                for name in (
                    "_compiled_request_selected",
                    "_is_batchable",
                    "_generate",
                    "_serve_request",
                )
            },
        )
        self.scheduler = self.cls()
        self.scheduler.cli_args = types.SimpleNamespace(self_mtp=False)
        self.scheduler.model_provider = types.SimpleNamespace(
            is_batchable=True,
            draft_model=None,
            model=object(),
        )
        self.args = types.SimpleNamespace(
            n=1,
            seed=None,
            prompt_lookup_ngram=0,
            max_tokens=64,
            model=types.SimpleNamespace(model="test", adapter=None, draft=None),
        )

    def test_ordinary_n1_reaches_real_scheduler_single_route(self):
        # Execute the real scheduler entry point, not generate_step directly.
        selected = []
        s = self.scheduler
        s._stop = False
        s._is_distributed = False
        s.model_provider.load_default = lambda: None
        s.model_provider.load = lambda *_: (s.model_provider.model, object())
        s._next_request = lambda *_: (object(), {"prompt": "hi"}, self.args)
        s._tokenize = mock.Mock(return_value=([1], [[1]], [], None))

        def serve(request, *, tokenized=None):
            self.assertEqual(tokenized[0], [1])
            selected.append(request)
            s._stop = True

        s._serve_single = serve
        self.env.update(
            mx=types.SimpleNamespace(
                default_stream=lambda _: None, default_device=lambda: None
            ),
            BATCH_IDLE_BACKOFF_SECONDS=0,
        )
        s._generate()
        self.assertEqual(len(selected), 1)
        s._tokenize.assert_called_once()

    def test_selected_requests_are_serial_and_unrelated_modes_keep_batching(self):
        self.assertFalse(self.scheduler._is_batchable(self.args, 1))
        self.args.seed = 7
        self.assertFalse(self.scheduler._is_batchable(self.args, 1))
        self.args.seed = None
        self.env["compiled_decode_enabled"] = lambda: False
        self.assertTrue(self.scheduler._is_batchable(self.args))
        self.env["compiled_decode_enabled"] = lambda: True
        self.env["compiled_decode_serving_reason"] = lambda *_: "unqualified"
        self.assertTrue(self.scheduler._is_batchable(self.args))
        self.env["compiled_decode_serving_reason"] = lambda *_: None
        self.scheduler.cli_args.self_mtp = True
        self.assertTrue(self.scheduler._is_batchable(self.args))
        self.scheduler.cli_args.self_mtp = False
        self.scheduler.cli_args.kv_bits = 8
        self.assertTrue(self.scheduler._is_batchable(self.args))

    def test_parallel_and_prompt_lookup_are_not_compiled(self):
        self.args.n = 3
        self.assertFalse(self.scheduler._compiled_request_selected(self.args))

    def test_full_context_budget_and_unknown_prompt_keep_batching(self):
        self.env["compiled_decode_context_policy"] = lambda context, steps: (
            ("over limit", None) if context + steps > 4096 else (None, object())
        )
        self.assertTrue(self.scheduler._is_batchable(self.args))
        self.assertTrue(self.scheduler._is_batchable(self.args, 4096))
        self.assertFalse(self.scheduler._is_batchable(self.args, 4032))

    def test_overlimit_real_scheduler_keeps_batch_route_and_reuses_tokens(self):
        class ReachedBatch(BaseException):
            pass

        self.env["compiled_decode_context_policy"] = lambda context, steps: (
            ("over limit", None) if context + steps > 4096 else (None, object())
        )
        s = self.scheduler
        s._stop = False
        s._is_distributed = False
        s.model_provider.load_default = lambda: None
        s.model_provider.load = lambda *_: (s.model_provider.model, object())
        s._next_request = lambda *_: (object(), {"prompt": "long"}, self.args)
        s._tokenize = mock.Mock(return_value=([1] * 4096, [], [], None))
        s._serve_single = mock.Mock(side_effect=AssertionError("must remain batchable"))

        def reached_batch(*_, **__):
            raise ReachedBatch()

        self.env.update(
            mx=types.SimpleNamespace(
                default_stream=lambda _: None, default_device=lambda: None
            ),
            BATCH_IDLE_BACKOFF_SECONDS=0,
            _batched_self_mtp_config=reached_batch,
        )
        with self.assertRaises(ReachedBatch):
            s._generate()
        s._serve_single.assert_not_called()
        s._tokenize.assert_called_once()
        self.args.n = 1
        self.args.prompt_lookup_ngram = 3
        self.assertFalse(self.scheduler._compiled_request_selected(self.args))


class TestCompiledStreamLifecycle(unittest.TestCase):
    def _loop(self):
        events = []

        class Output:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Step:
            def __init__(self):
                self.pending = []
                self.completed = []
                self.failed = []
                self._poisoned = False
                self.cache = [types.SimpleNamespace(state=object())]

            @property
            def _pending_receipts(self):
                return [("key", output) for output in self.pending]

            def materialize_and_confirm(self, output, *_, **__):
                if output is not self.pending[0]:
                    raise AssertionError("not the oldest exact submitted output")
                self.completed.append(self.pending.pop(0))
                events.append("confirmed")
                return 1

            def receipt(self):
                return {
                    "pending": len(self.pending),
                    "completed": len(self.completed),
                    "failed": len(self.failed),
                    "poisoned": self._poisoned,
                }

        ns = {}
        _definitions("mlx_lm/compiled_decode.py", {"drain_pending"}, ns)
        Step.drain_pending = ns["drain_pending"]
        step = Step()
        status = {}

        def submit(y):
            out = Output(y.value + 1)
            step.pending.append(out)
            events.append("submitted")
            return out, object(), out

        def failure(error, _phase):
            step._poisoned = True
            step.failed.extend(step.pending)
            step.pending.clear()
            return error

        env = {
            "mx": types.SimpleNamespace(
                async_eval=lambda *_: None,
                eval=lambda *_: None,
                clear_cache=lambda: None,
            ),
            "y": Output(0),
            "logprobs": object(),
            "completion_output": None,
            "max_tokens": 4,
            "compiled_step": step,
            "_step": submit,
            "_compiled_failure": failure,
            "_compiled_decode_status": status,
            "prompt_cache": step.cache,
            "CACHE_STATE_EVAL_INTERVAL": 256,
            "prompt_progress_callback": lambda *_: None,
            "total_prompt_tokens": 1,
        }
        tree = ast.parse(_source("mlx_lm/generate.py"))
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "generate_step"
        )
        start = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign) and ast.unparse(n) == "n = 0"
        )
        loop = ast.parse("def run():\n    pass\n").body[0]
        env["_initial_values"] = (env["y"], env["logprobs"], env["completion_output"])
        loop.body = (
            ast.parse("y, logprobs, completion_output = _initial_values").body
            + fn.body[start:]
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module([loop], [])), "decode-loop", "exec"
            ),
            env,
        )
        return env["run"](), step, status, events, env

    def test_exhaustion_and_length_close_drain_exact_lookahead(self):
        for exhaust in (True, False):
            gen, step, status, events, _ = self._loop()
            if exhaust:
                self.assertEqual(len(list(gen)), 4)
            else:
                self.assertEqual([next(gen)[0] for _ in range(4)], list(range(4)))
                status["stop_reason"] = "length"
                gen.close()
            self.assertEqual(len(step.completed), 4)
            self.assertEqual(events.count("submitted"), 4)
            self.assertEqual(status["receipt"]["pending"], 0)
            self.assertEqual(status["terminal_reason"], "length")

    def test_cancel_eos_and_throw_have_terminal_receipts(self):
        for reason in ("cancelled", "eos", "error"):
            gen, step, status, events, _ = self._loop()
            next(gen)
            status["stop_reason"] = reason
            if reason == "error":
                with self.assertRaisesRegex(ValueError, "consumer"):
                    gen.throw(ValueError("consumer"))
                self.assertTrue(status["receipt"]["poisoned"])
                self.assertEqual(len(step.failed), 1)
            else:
                gen.close()
                self.assertEqual(len(step.completed), 1)
            self.assertEqual(events.count("submitted"), 1)
            self.assertEqual(status["receipt"]["pending"], 0)
            self.assertEqual(status["terminal_reason"], reason)

    def test_wrapper_close_reaches_inner_generator_without_gc(self):
        gen, _, status, _, _ = self._loop()
        ns = {"contextlib": contextlib}
        _definitions("mlx_lm/generate.py", {"_non_speculative_tokens"}, ns)
        wrapped = ns["_non_speculative_tokens"](gen)
        self.assertEqual(next(wrapped)[2], False)
        wrapped.close()
        self.assertEqual(status["receipt"]["pending"], 0)

    def test_drain_error_still_publishes_terminal_receipt(self):
        gen, step, status, _, _ = self._loop()
        next(gen)

        def fail(**_):
            step._poisoned = True
            step.failed.extend(step.pending)
            step.pending.clear()
            raise RuntimeError("device failure")

        step.drain_pending = fail
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            gen.close()
        self.assertEqual(status["terminal_reason"], "error")
        self.assertEqual(status["receipt"]["failed"], 1)

    def test_real_stream_wrapper_finishes_before_terminal_response(self):
        for exit_kind in ("length", "eos", "cancelled", "error"):
            inner, _, status, events, _ = self._loop()

            class Array(list):
                @property
                def size(self):
                    return len(self)

            class Tokenizer:
                eos_token_ids = {0} if exit_kind == "eos" else set()

                def __init__(self):
                    self.detokenizer = types.SimpleNamespace(
                        last_segment="text",
                        finalize=lambda: None,
                        add_token=self.add_token,
                    )

                def add_token(self, _):
                    if exit_kind == "error":
                        raise ValueError("detokenizer failure")

            env = {
                "contextlib": contextlib,
                "time": time,
                "TokenizerWrapper": Tokenizer,
                "mx": types.SimpleNamespace(array=Array, get_peak_memory=lambda: 0),
                "generate_step": lambda *_, **__: inner,
                "generation_stream": None,
                "wired_limit": lambda *_: contextlib.nullcontext(),
                "GenerationResponse": types.SimpleNamespace,
            }
            _definitions(
                "mlx_lm/generate.py",
                {"stream_generate", "_non_speculative_tokens"},
                env,
            )
            stream = env["stream_generate"](
                object(),
                Tokenizer(),
                Array([1]),
                max_tokens=4,
                _compiled_decode_status=status,
            )
            if exit_kind == "error":
                with self.assertRaisesRegex(ValueError, "detokenizer"):
                    next(stream)
            elif exit_kind == "cancelled":
                next(stream)
                status["stop_reason"] = "cancelled"
                stream.close()
            else:
                for response in stream:
                    if response.finish_reason is not None:
                        self.assertEqual(status["receipt"]["pending"], 0)
            self.assertEqual(status["receipt"]["pending"], 0)
            self.assertEqual(status["terminal_reason"], exit_kind)
            self.assertEqual(
                events.count("submitted"), 4 if exit_kind == "length" else 1
            )


class TestCheckpointQualification(unittest.TestCase):
    def setUp(self):
        name = "_cpu_compiled_qualification"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "mlx_lm/compiled_qualification.py"
        )
        self.q = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.q
        spec.loader.exec_module(self.q)
        self.addCleanup(sys.modules.pop, name)
        self.model = types.SimpleNamespace()
        self.parameter = types.SimpleNamespace(shape=(2, 3), dtype="bfloat16")
        self.parameters = [("weight", self.parameter)]
        self.config = {
            "hidden_size": 2,
            "num_hidden_layers": 4,
            "model_type": "qwen3_5_moe",
        }
        self.runtime = {"mlx_version": "test", "sources": {"model": "abc"}}
        self.policy = types.SimpleNamespace(
            name="short",
            max_context=4096,
            buckets=(4096,),
            numerical_acceptance="class3-padded-sdpa-v1",
        )

    def _approve_fixture(self, files):
        self.q.SERVING_QUALIFICATIONS["fixture-only"] = {
            "schema": 1,
            "config_sha256": self.q._digest(self.config),
            "weights_sha256": {Path(p).name: self.q._file_digest(p) for p in files},
            "parameter_layout": [("weight", [2, 3], "bfloat16")],
            "runtime": self.runtime,
            "environment": self.q.execution_environment(),
            "evidence": "inert test fixture, not a model qualification",
            "profiles": {
                "short": {
                    "max_context": 4096,
                    "buckets": [4096],
                    "numerical_acceptance": "class3-padded-sdpa-v1",
                }
            },
        }

    def _bind(self, files):
        self.q.bind_serving_qualification(
            self.model,
            self.config,
            files,
            runtime=self.runtime,
            parameters=self.parameters,
        )

    def _reason(self):
        return self.q.serving_qualification_reason(
            self.model, parameters=self.parameters, policy=self.policy
        )

    def test_empty_catalogue_never_approves_family_or_opt_in(self):
        self.assertEqual(self.q.SERVING_QUALIFICATIONS, {})
        self.model.supports_compiled_decode_replay = "qwen3_5_moe_m1_v1"
        with mock.patch.dict(self.q.os.environ, {"MLX_LM_COMPILED_DECODE": "1"}):
            self._bind(["must-not-read-unqualified-weights"])
            self.assertIn("no reviewed checkpoint", self._reason())

    def test_loader_binding_matches_config_weights_dtype_and_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            path.write_bytes(b"fixture weights")
            self._approve_fixture([path])
            self._bind([path])
            self.assertIsNone(self._reason())
            path.write_bytes(b"different weights")
            self._bind([path])
            self.assertIsNotNone(self._reason())
            self._approve_fixture([path])
            self.parameter.dtype = "float16"
            self._bind([path])
            self.assertIsNotNone(self._reason())
            self.parameter.dtype = "bfloat16"
            self.runtime = {"mlx_version": "other build"}
            self._bind([path])
            self.assertIsNotNone(self._reason())

    def test_environment_parameter_replacement_and_profile_invalidate_binding(self):
        self._approve_fixture([])
        self._bind([])
        self.assertIsNone(self._reason())
        with mock.patch.dict(self.q.os.environ, {"MLX_NEW_FUSION": "1"}):
            self.assertIn("environment", self._reason())
        self.policy.buckets = (16384,)
        self.assertIn("profile", self._reason())
        self.policy.buckets = (4096,)
        self.parameters = [("weight", object())]
        self.assertIn("parameters", self._reason())

    def _manifest_fixture(self, temporary):
        path = Path(temporary) / "model.safetensors"
        path.write_bytes(b"fixture")
        identity = self.q.qualification_identity(
            self.config,
            [path],
            runtime=self.runtime,
            parameters=self.parameters,
        )
        identity = json.loads(json.dumps(identity))
        point = {
            "context_policy": "short",
            "policy_buckets": [4096],
            "seed_context": 3968,
            "measurement_end_context": 4096,
            "steps": 128,
            "digest_kv": list(range(128)),
            "digest_ring": list(range(128)),
            "digest_compiled": list(range(128)),
            "submitted": 128,
            "completed": 128,
            "pending": 0,
            "failed": 0,
            "poisoned": False,
            "single_trace": True,
            "traces": [1],
            "capacities_observed": [4096],
            "ring_capacities_observed": [4096],
            "logits_compiled_vs_ring": {"bitidentical": True, "maxdelta": 0},
            "logits_ring_vs_kv": {"maxdelta": 0.01},
            "logits_compiled_vs_kv": {"maxdelta": 0.01},
        }
        numerical = {
            "qualification_identity": identity,
            "class3_maxdelta_bound": 0.02,
            "numerical_operating_points": {"ctx4096_M1_short": point},
            "numerical_growth_boundaries": {},
        }
        numerical_path = Path(temporary) / "numerical.json"
        numerical_path.write_text(json.dumps(numerical))
        case = {
            "stock_tokens": [1, 2],
            "candidate_tokens": [1, 2],
            "compiled_used": True,
            "pending": 0,
            "failed": 0,
            "poisoned": False,
            "stock_ttft_ms": 1,
            "candidate_ttft_ms": 2,
            "stock_total_ms": 3,
            "candidate_total_ms": 4,
        }
        cases = {
            name: dict(case)
            for name in ("cold", "apc_hit", "eos", "length", "cancel", "concurrent")
        }
        cases["apc_hit"]["compiled_used"] = False
        cases["concurrent"].update(requests=2, serialized=True)
        serving = {
            "schema": "compiled-serving-e2e-v1",
            "qualification_identity": identity,
            "route": "ordinary-unseeded-n1",
            "cases": cases,
        }
        serving_path = Path(temporary) / "serving.json"
        serving_path.write_text(json.dumps(serving))
        manifest = {
            "schema": 1,
            "qualification_id": "operator-fixture",
            "approval": "approved",
            "approved_by": "test-only",
            **identity,
            "class3_maxdelta_bound": 0.02,
            "profiles": {
                "short": {
                    "max_context": 4096,
                    "buckets": [4096],
                    "numerical_acceptance": "class3-padded-sdpa-v1",
                }
            },
            "evidence": {
                "path": str(numerical_path),
                "sha256": self.q._file_digest(numerical_path),
            },
            "serving_evidence": {
                "path": str(serving_path),
                "sha256": self.q._file_digest(serving_path),
            },
        }
        manifest_path = Path(temporary) / "approval.json"
        manifest_path.write_text(json.dumps(manifest))
        return path, manifest_path, manifest, numerical_path, numerical

    def test_operator_manifest_binds_and_path_move_provenance_are_not_numerics(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights, manifest_path, manifest, _, _ = self._manifest_fixture(temporary)
            with mock.patch.dict(
                self.q.os.environ,
                {
                    self.q._MANIFEST_ENV: str(manifest_path),
                    "MLX_LM_COMPILED_DECODE": "1",
                    "MLXUAG_RUN_PROVENANCE": "serving",
                    "MLX_LM_COMPILED_DECODE_ACCEPTANCE": "class3-padded-sdpa-v1",
                },
            ):
                self._bind([weights])
                self.assertIsNone(self._reason())
                moved = Path(temporary) / "moved-approval.json"
                moved.write_text(json.dumps(manifest))
                self.q.os.environ[self.q._MANIFEST_ENV] = str(moved)
                self._bind([weights])
                self.assertIsNone(self._reason())
                with mock.patch.dict(self.q.os.environ, {"MLX_NEW_FUSION": "1"}):
                    self.assertIn("environment", self._reason())
                moved.write_text("{}")
                self.assertIn("manifest changed", self._reason())

    def test_operator_manifest_rejects_missing_approval_and_false_numerics(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights, manifest_path, manifest, numerical_path, numerical = (
                self._manifest_fixture(temporary)
            )
            with mock.patch.dict(
                self.q.os.environ, {self.q._MANIFEST_ENV: str(manifest_path)}
            ):
                manifest["approval"] = "candidate"
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "operator approval"):
                    self._bind([weights])
                manifest["approval"] = "approved"
                numerical["numerical_operating_points"]["ctx4096_M1_short"][
                    "digest_compiled"
                ][-1] = 999
                numerical_path.write_text(json.dumps(numerical))
                manifest["evidence"]["sha256"] = self.q._file_digest(numerical_path)
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "token evidence"):
                    self._bind([weights])

    def test_operator_manifest_requires_serving_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights, manifest_path, manifest, _, _ = self._manifest_fixture(temporary)
            serving_path = Path(manifest["serving_evidence"]["path"])
            serving = json.loads(serving_path.read_text())
            del serving["cases"]["concurrent"]
            serving_path.write_text(json.dumps(serving))
            manifest["serving_evidence"]["sha256"] = self.q._file_digest(serving_path)
            manifest_path.write_text(json.dumps(manifest))
            with mock.patch.dict(
                self.q.os.environ, {self.q._MANIFEST_ENV: str(manifest_path)}
            ):
                with self.assertRaisesRegex(ValueError, "lifecycle"):
                    self._bind([weights])

    def test_malformed_manifest_mappings_leave_actual_loader_eager(self):
        tree = ast.parse(_source("mlx_lm/utils.py"))
        load = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "load_model"
        )
        gate = copy.deepcopy(
            next(
                n
                for n in load.body
                if isinstance(n, ast.If) and ast.unparse(n.test) == "strict"
            )
        )
        # Execute the loader's real admission/exception block without importing MLX.
        gate.body = [n for n in gate.body if not isinstance(n, ast.ImportFrom)]
        with tempfile.TemporaryDirectory() as temporary:
            weights, manifest_path, original, numerical_path, numerical_original = (
                self._manifest_fixture(temporary)
            )
            self.model.parameters = lambda: self.parameters
            for malformed in (
                "profile",
                "reference",
                "serving_reference",
                "comparison",
            ):
                manifest = copy.deepcopy(original)
                numerical = copy.deepcopy(numerical_original)
                if malformed == "profile":
                    manifest["profiles"]["short"] = []
                elif malformed == "reference":
                    manifest["evidence"] = []
                elif malformed == "serving_reference":
                    manifest["serving_evidence"] = []
                else:
                    numerical["numerical_operating_points"]["ctx4096_M1_short"][
                        "logits_compiled_vs_ring"
                    ] = []
                    numerical_path.write_text(json.dumps(numerical))
                    manifest["evidence"]["sha256"] = self.q._file_digest(numerical_path)
                manifest_path.write_text(json.dumps(manifest))
                env = {
                    "strict": True,
                    "SERVING_QUALIFICATIONS": {},
                    "os": self.q.os,
                    "mx": object(),
                    "runtime_identity": lambda _: self.runtime,
                    "bind_serving_qualification": self.q.bind_serving_qualification,
                    "tree_flatten": lambda pairs: pairs,
                    "model": self.model,
                    "config": self.config,
                    "weight_files": [weights],
                }
                with mock.patch.dict(
                    self.q.os.environ, {self.q._MANIFEST_ENV: str(manifest_path)}
                ):
                    exec(
                        compile(ast.Module([gate], []), "loader-admission", "exec"), env
                    )
                self.assertIsNone(self.model._compiled_decode_serving_binding)
                self.assertIn(
                    "must be an object", self.model._compiled_decode_qualification_error
                )

    def test_candidate_identity_is_json_roundtrip_stable(self):
        identity = self.q.qualification_identity(
            self.config,
            [],
            runtime=self.runtime,
            parameters=self.parameters,
        )
        self.assertEqual(identity, json.loads(json.dumps(identity)))


class TestCompiledDecodeSourceContracts(unittest.TestCase):
    def test_sources_parse_without_importing_mlx(self):
        for relative in (
            "mlx_lm/compiled_decode.py",
            "mlx_lm/compiled_qualification.py",
            "mlx_lm/generate.py",
            "mlx_lm/utils.py",
            "mlx_lm/models/cache.py",
            "mlx_lm/models/qwen3_5.py",
            "mlx_lm/models/qwen3_next.py",
            "mlx_lm/server.py",
        ):
            ast.parse(_source(relative), filename=relative)

    def test_qualification_and_acceptance_are_exact(self):
        compiled = _source("mlx_lm/compiled_decode.py")
        qwen_next = _source("mlx_lm/models/qwen3_next.py")
        self.assertIn('_QUALIFIED_MODEL_TYPES = ("qwen3_5_moe",)', compiled)
        self.assertIn('"qwen3_5_moe_m1_v1"', compiled)
        self.assertIn('"class3-padded-sdpa-v1"', compiled)
        self.assertIn("supports_compiled_decode_replay = False", qwen_next)

    def test_failure_and_completion_receipt_contracts_are_present(self):
        compiled = _source("mlx_lm/compiled_decode.py")
        self.assertIn("class CompiledDecodePoisoned", compiled)
        self.assertNotIn("def confirm_completed", compiled)
        self.assertNotIn("def confirm_all_completed", compiled)
        self.assertIn("def materialize_and_confirm", compiled)
        self.assertIn("output is not self._pending_receipts[0][1]", compiled)
        self.assertIn("def receipt", compiled)
        self.assertIn("mx.eval(out)", compiled)
        self.assertIn("self._restore_failed_call", compiled)
        self.assertIn("unconfirmed calls", compiled)

    def test_generate_keeps_caller_cache_private_and_skips_zero_tokens(self):
        generate = _source("mlx_lm/generate.py")
        self.assertIn("if caller_supplied_prompt_cache:", generate)
        self.assertIn("compiled replay requires a private cache", generate)
        self.assertIn("if compiled_decode and max_tokens != 0:", generate)
        self.assertIn("compiled_step.materialize_and_confirm(", generate)

    def test_server_private_cache_contract_is_explicit_and_not_persisted(self):
        generate = _source("mlx_lm/generate.py")
        server = _source("mlx_lm/server.py")
        self.assertIn("_prompt_cache_is_request_private", generate)
        self.assertIn("_compiled_decode_status", generate)
        self.assertIn("cache_is_request_private = cache is None", server)
        self.assertIn(
            "_prompt_cache_is_request_private=cache_is_request_private", server
        )
        tree = ast.parse(server)
        guard = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "compiled_decode_status['used']"
        )
        body_calls = {
            ast.unparse(node.func)
            for statement in guard.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        }
        else_calls = {
            ast.unparse(node.func)
            for statement in guard.orelse
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        }
        self.assertNotIn("_store_single_request_prompt_cache", body_calls)
        self.assertIn("_store_single_request_prompt_cache", else_calls)

    def test_model_swap_collects_python_cycles_before_mlx_cache(self):
        tree = ast.parse(_source("mlx_lm/server.py"))
        load_fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_load"
        )
        calls = {
            (
                f"{node.func.value.id}.{node.func.attr}"
                if isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                else ""
            ): node.lineno
            for node in ast.walk(load_fn)
            if isinstance(node, ast.Call)
        }
        self.assertLess(calls["gc.collect"], calls["mx.clear_cache"])


if __name__ == "__main__":
    unittest.main()
