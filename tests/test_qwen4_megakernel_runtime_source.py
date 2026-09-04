"""GPU-free source contracts for the megakernel runtime safety boundary.

These tests intentionally do not import ``mlx_lm`` or ``mlx``. They protect
the fail-closed wiring that can be reviewed before the post-reboot Metal gate.
"""

import ast
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "mlx_lm" / "models"


def _tree(name: str) -> ast.Module:
    path = MODELS / name
    return ast.parse(path.read_text(), filename=str(path))


def _literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing module assignment {name}")


def _contract():
    path = MODELS / "qwen4_megakernel_contract.py"
    spec = importlib.util.spec_from_file_location("mega_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pure_contract_refuses_wrong_model_and_host_inputs():
    from types import SimpleNamespace

    contract = _contract()
    args = SimpleNamespace(**contract.MODEL_CONTRACT)
    args.num_hidden_layers = 48
    args.layer_types = [
        "linear_attention" if (i + 1) % 4 else "full_attention"
        for i in range(48)
    ]
    args.partial_rotary_factor = 0.25
    args.rope_scaling = {"type": "default"}
    args.rope_theta = 10_000_000.0
    args.ple_layer_ids = [2]
    contract.validate_model_contract(args)
    model = SimpleNamespace(language_model=SimpleNamespace(args=args))
    contract.validate_model_binding(model, args)
    other = SimpleNamespace(**vars(args))
    other.rope_theta = 1.0
    with pytest.raises(ValueError, match="model/args mismatch"):
        contract.validate_model_binding(
            SimpleNamespace(language_model=SimpleNamespace(args=other)), args)
    args.hidden_size = 2559
    with pytest.raises(ValueError, match="hidden_size"):
        contract.validate_model_contract(args)

    contract.validate_launch_shapes(
        (3, 2560), (3, 2560), width=3, has_ple=True)
    with pytest.raises(ValueError, match="required"):
        contract.validate_launch_shapes(
            (3, 2560), None, width=3, has_ple=True)
    with pytest.raises(ValueError, match="shape"):
        contract.validate_launch_shapes(
            (3, 2559), (3, 2560), width=3, has_ple=True)


def test_pure_position_contract_is_monotonic_and_capacity_bounded():
    contract = _contract()
    contract.validate_position(125, 125, 3, 128)
    with pytest.raises(ValueError, match="does not match"):
        contract.validate_position(10, 11, 1, 128)
    with pytest.raises(ValueError, match="exceeds ledger capacity"):
        contract.validate_position(126, 126, 3, 128)


def test_transactional_ple_state_stays_within_the_binding_budget():
    path = MODELS / "qwen4_megakernel_body.py"
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    inputs = _literal_assignment(tree, "IN_NAMES")
    outputs = _literal_assignment(tree, "OUT_NAMES")

    assert len(inputs) + len(outputs) == 30
    assert "pconv" in inputs
    assert "pconv_out" in outputs
    assert "pconv_out[(size_t)r * HCH + c] = pconv[(size_t)r * HCH + c]" in source
    assert "device BFT* st = pconv_out;" in source


def test_device_status_is_consumed_before_a_launch_becomes_pending():
    source = (MODELS / "qwen4_megakernel_runtime.py").read_text()
    start = source.index("    def _consume_launch(")
    end = source.index("\n    # ----------------------------------------------------------- the token", start)
    consume = source[start:end]

    sync = consume.index("mx.eval(status)")
    abort = consume.index("if abort_count:")
    phase = consume.index("if device_phase != expected_phase:")
    pending = consume.index("self._pending = outs")
    assert sync < abort < phase < pending
    assert consume.count("raise MegakernelDeviceAbort") >= 3


def test_runtime_pack_plan_can_omit_lm_head():
    source = (MODELS / "qwen4_megakernel_pack.py").read_text()
    start = source.index("def decode_path_keys(")
    end = source.index("\n\ndef ", start + 1)
    plan = source[start:end]

    assert "include_lm_head: bool = True" in plan
    assert 'if include_lm_head:\n        add("language_model.lm_head", "lm_head")' in plan


def test_all_runtime_safety_modules_parse_without_importing_mlx():
    for name in (
        "qwen4_megakernel.py",
        "qwen4_megakernel_body.py",
        "qwen4_megakernel_contract.py",
        "qwen4_megakernel_config.py",
        "qwen4_megakernel_pack.py",
        "qwen4_megakernel_runtime.py",
        "qwen4_verify_tree.py",
    ):
        _tree(name)


def test_runtime_owns_transactions_and_marks_failed_rebinds():
    source = (MODELS / "qwen4_megakernel_runtime.py").read_text()
    assert "self._state_lock = threading.RLock()" in source
    assert "self._in_flight_owner = owner" in source
    assert "self._pending_owner = self._in_flight_owner" in source
    assert "pending megakernel transaction belongs to another thread" in source
    assert "have begun; the model is mutated and MUST NOT be used as" in source
    assert '"a stock fallback"' in source
    assert "MegakernelConstructionError" in source


def test_preflight_uses_exact_pack_and_individual_buffer_sizes():
    runtime = (MODELS / "qwen4_megakernel_runtime.py").read_text()
    pack = (MODELS / "qwen4_megakernel_pack.py").read_text()
    config = (MODELS / "qwen4_megakernel_config.py").read_text()
    assert "self.pack_estimate = MP.estimate_pack(" in runtime
    assert 'self.pack_estimate["packed_bytes"]' in runtime
    assert 'self.pack_estimate["largest_group_bytes"]' in runtime
    assert "def estimate_pack(" in pack
    assert "individual_buffer_bytes" in config
    assert "tested_geometry != (int(threads), int(groups))" in config
    assert "actual_threadgroup_bytes" in config
    assert "rounded_ledger_capacity(max_context, IDX_COMPRESS)" in runtime
    assert "MAX_BLOCKS" not in runtime
    assert '"score_tiles": score_layout["bytes"]' in runtime
    assert "score_blocks=self.pooled_stride" in runtime
    assert runtime.count("total=self.total") >= 2
    assert "self.position, position, width, self.max_context" in runtime


def test_dynamic_tiled_scores_remove_the_fixed_context_plane():
    contract = _contract()
    layout = contract.score_tile_layout(65_536, width=3)
    assert layout == {
        "tile_blocks": 4096,
        "tiles": 16,
        "block_count": 65_536,
        "stride": 65_536,
        "width": 3,
        "elements": 196_608,
        "bytes": 786_432,
    }
    ragged = contract.score_tile_layout(16_385, width=1)
    assert ragged["tiles"] == 5
    assert ragged["stride"] == 20_480
    assert ragged["stride"] >= ragged["block_count"]
    with pytest.raises(OverflowError, match="uint32"):
        contract.score_tile_layout(contract.METAL_UINT32_MAX, width=1)
    with pytest.raises(OverflowError, match="host shape"):
        contract.score_tile_layout(1 << 30, width=3)
    with pytest.raises(ValueError, match="power of two"):
        contract.score_tile_layout(100, tile_blocks=3000)


def test_score_source_uses_disjoint_dynamic_planes_and_global_order():
    kernel = (MODELS / "qwen4_megakernel.py").read_text()
    body = (MODELS / "qwen4_megakernel_body.py").read_text()
    schedule = (MODELS / "qwen4_megakernel_schedule.py").read_text()
    assert '("IDX_SCORE",' not in kernel
    assert "for (uint base = 0u; base < n; base += SCTILE)" in kernel
    assert "const uint chunk = (n + nt - 1u) / nt;" in kernel
    assert "score_tiles + (size_t)mq * score_stride" in body
    assert "const uint score_stride = actl[24];" in body
    assert "select_top_blocks(score_row, mc[4], a0," in body
    assert '"score_tiles"' in body
    assert 'op=OP_INDEX_SCORE' in schedule and 'dst=0, arg0=slot' in schedule


def test_score_output_keeps_binding_shape_and_dtype_lists_aligned():
    tree = _tree("qwen4_megakernel_body.py")
    outputs = _literal_assignment(tree, "OUT_NAMES")
    assert outputs == [
        "scratch", "score_tiles", "out", "cs_out", "rec_out",
        "pconv_out", "apm", "apo", "logits", "status",
    ]
    launches = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "kernel"
        ):
            keywords = {kw.arg: kw.value for kw in node.keywords}
            if "output_shapes" in keywords:
                launches.append(keywords)
    assert len(launches) == 1
    assert len(launches[0]["output_shapes"].elts) == len(outputs)
    assert len(launches[0]["output_dtypes"].elts) == len(outputs)


def test_ledger_rounding_refuses_signed_metal_overflow():
    contract = _contract()
    assert contract.rounded_ledger_capacity(262_144, 4) == 262_144
    assert contract.rounded_ledger_capacity(262_143, 4) == 262_144
    with pytest.raises(OverflowError, match="int32"):
        contract.rounded_ledger_capacity(contract.METAL_INT32_MAX, 4)


def test_wide_wrapper_requires_explicit_prepare_without_a_compile_claim():
    runtime = (MODELS / "qwen4_megakernel_runtime.py").read_text()
    body = (MODELS / "qwen4_megakernel_body.py").read_text()
    assert "wide wrapper is unprepared; call prepare_width_wrapper(width)" in runtime
    assert "def prepare_width_wrapper(self, width: int) -> dict" in body
    assert '"launched": False' in body
    assert '"wide_wrapper_prepared": self._wide is not None' in body
    assert '"pipeline_compilation": "deferred_to_first_evaluation"' in body
    assert '"wide_build_seconds"' not in body


def _source_function(filename, name, namespace, *, class_name=None):
    tree = _tree(filename)
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef)
                     and n.name == class_name).body
    function = next(n for n in nodes if isinstance(n, ast.FunctionDef)
                    and n.name == name)
    future = ast.ImportFrom(module="__future__", level=0,
                            names=[ast.alias(name="annotations")])
    module = ast.fix_missing_locations(ast.Module(body=[future, function],
                                                  type_ignores=[]))
    exec(compile(module, str(MODELS / filename), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("mode", ["skip", "auto", "off", "force"])
@pytest.mark.parametrize("entry", [None, {"primitives": {"ok": True}}])
def test_actual_cold_config_receipt_reports_mode_without_running_tuning(mode, entry):
    import builtins

    calls = []
    tune = SimpleNamespace(
        read_entry=lambda signature: (entry, None),
        tune_mode=lambda: mode,
    )

    def resolve(**kwargs):
        calls.append(kwargs)
        return {"cache": {"geometry_ignored": "not adopted"}}

    def import_stub(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "__future__" and level == 0:
            return builtins.__import__(name, globals, locals, fromlist, level)
        assert name == "" and level == 1
        assert fromlist == ("qwen4_megakernel_tune",)
        return SimpleNamespace(qwen4_megakernel_tune=tune)

    probe = SimpleNamespace(signature="test-device", as_dict=lambda: {"name": "test"})
    ns = {"__builtins__": {**vars(builtins), "__import__": import_stub},
          "MD": SimpleNamespace(probe_device=lambda: probe),
          "resolve": resolve, "ConfigError": ValueError, "_LAST": {}}
    receipt = _source_function("qwen4_megakernel_config.py", "config_receipt", ns)
    cache = receipt(autotune=False)["cache"]
    assert cache == {
        "geometry_ignored": "not adopted", "mode": mode,
        "consulted": True, "hit": entry is not None, "calibrated": False,
        "autotune_allowed": False, "primitive_tests_run": False,
        "error": None, "cold_safe": True,
    }
    assert calls == [{"probe": probe, "cache_entry": entry, "tune": False}]


def test_actual_wrapper_preparation_never_claims_compilation_or_launch():
    body_name = "qwen4_megakernel_body.py"
    ns = {}
    prepared = _source_function(body_name, "is_width_wrapper_prepared", ns,
                                class_name="DualWidthMegakernelBody")
    prepare = _source_function(body_name, "prepare_width_wrapper", ns,
                               class_name="DualWidthMegakernelBody")
    receipt = _source_function(body_name, "receipt", ns,
                               class_name="DualWidthMegakernelBody")
    body = SimpleNamespace(_wide=None, wide_width=3, narrow_width=1,
                           wide_wrapper_prepare_seconds=None,
                           calls_narrow=0, calls_wide=0, threads=512,
                           groups=40, spin_cap=400000)
    body.receipt = lambda: receipt(body)
    body.is_width_wrapper_prepared = lambda width: prepared(body, width)

    def ensure():
        body._wide = object()
        body.wide_wrapper_prepare_seconds = 0.001

    body._ensure_wide = ensure
    before = prepare(body, 1)
    assert before["wide_wrapper_prepared"] is False
    assert before["wrapper_built_now"] is False
    after = prepare(body, 3)
    assert after["wrapper_built_now"] is True
    assert after["wrapper_prepared"] is True
    assert after["launched"] is False
    assert after["pipeline_compilation"] == "deferred_to_first_evaluation"
    assert prepare(body, 3)["wrapper_built_now"] is False
    assert prepared(body, 0) is False
    assert prepared(body, 4) is False
    with pytest.raises(ValueError, match="outside"):
        prepare(body, 4)


@pytest.mark.parametrize("bad_shape", [False, True])
def test_actual_completion_checks_score_shape_before_recording_allocation(bad_shape):
    layout = _contract().score_tile_layout(16385, 3)
    receipts = []
    ns = {"mx": SimpleNamespace(eval=lambda _: None),
          "OUT": {"status": 0, "score_tiles": 1, "logits": 2},
          "score_tile_layout": _contract().score_tile_layout,
          "IDX_COMPRESS": 4, "threading": threading,
          "record_megakernel_receipt": lambda **kw: receipts.append(kw),
          "MegakernelDeviceAbort": RuntimeError}
    consume = _source_function("qwen4_megakernel_runtime.py", "_consume_launch", ns,
                               class_name="MegakernelDecoder")
    decoder = SimpleNamespace(body=SimpleNamespace(phase=19), _poisoned_reason=None,
                              _state_lock=threading.RLock(), _in_flight=True,
                              _in_flight_owner=threading.get_ident(),
                              schedule=[], op_counts={}, device_barriers=0,
                              include_lm_head=True, _pending=None,
                              _geometry_fields=lambda _: {})
    outs = [SimpleNamespace(tolist=lambda: [0, 19]),
            SimpleNamespace(shape=(layout["elements"] - int(bad_shape),)), object()]
    if bad_shape:
        with pytest.raises(RuntimeError, match="score output shape"):
            consume(decoder, outs, position=65537, width=3, record=True)
        assert decoder._pending is None
        assert receipts[-1]["engaged"] is False
    else:
        assert consume(decoder, outs, position=65537, width=3, record=True) is outs[2]
        assert receipts[-1]["score_layout"] == layout
        assert receipts[-1]["position"] == 65537
        assert receipts[-1]["context"] == 65540


def test_runtime_and_status_do_not_implicitly_autotune_the_gpu():
    kernel = (MODELS / "qwen4_megakernel.py").read_text()
    config = (MODELS / "qwen4_megakernel_config.py").read_text()
    assert "def _portable_config(*, autotune: bool = False)" in kernel
    assert "return MC.config_receipt(autotune=autotune)" in kernel
    assert "def config_receipt(*, refresh: bool = False," in config
    assert '"cold_safe": True' in config
    assert '"spin_cap": int(values.get("spin_cap", _SPIN_CAP))' in kernel


def test_tree_clone_keeps_qsa_side_ledgers_and_refuses_armed_state():
    source = (MODELS / "qwen4_verify_tree.py").read_text()
    assert "cannot clone an armed QSA MTP cache" in source
    assert "c.index_keys = mx.contiguous" in source
    assert "c._qsa_pooled_keys = mx.contiguous" in source
    assert 'identity["complete_blocks"] = 0' in source
