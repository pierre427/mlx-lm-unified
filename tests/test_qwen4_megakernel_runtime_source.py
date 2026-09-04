"""GPU-free source contracts for the megakernel runtime safety boundary.

These tests intentionally do not import ``mlx_lm`` or ``mlx``. They protect
the fail-closed wiring that can be reviewed before the post-reboot Metal gate.
"""

import ast
from pathlib import Path


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


def test_transactional_ple_state_stays_within_the_binding_budget():
    path = MODELS / "qwen4_megakernel_body.py"
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    inputs = _literal_assignment(tree, "IN_NAMES")
    outputs = _literal_assignment(tree, "OUT_NAMES")

    assert len(inputs) + len(outputs) == 29
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
        "qwen4_megakernel_config.py",
        "qwen4_megakernel_pack.py",
        "qwen4_megakernel_runtime.py",
        "qwen4_verify_tree.py",
    ):
        _tree(name)
