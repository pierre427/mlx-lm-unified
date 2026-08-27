import ast
import gc
import importlib.util
import time
import weakref
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "mlx_lm_int8_prefill_under_test",
    Path(__file__).parents[1] / "mlx_lm" / "int8_prefill.py",
)
int8_prefill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(int8_prefill)


def _clear_overlay_caches():
    int8_prefill._int8_weights.clear()
    int8_prefill._ws_cache.clear()
    int8_prefill._act_cache.clear()


@pytest.fixture(autouse=True)
def restore_quantized_linear():
    int8_prefill.remove()
    _clear_overlay_caches()
    yield
    int8_prefill.remove()
    _clear_overlay_caches()


def _small_w4_linear():
    linear = nn.QuantizedLinear(
        1024, 1024, bias=False, group_size=64, bits=4
    )
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    return linear


def _expected_ws(linear):
    s = linear["scales"].astype(mx.float32)
    b = linear["biases"].astype(mx.float32)
    bound = mx.maximum(mx.abs(b), mx.abs(15.0 * s + b))
    return mx.maximum(bound.max(axis=1), 1e-8) / 127.0


def _fake_m5(monkeypatch):
    monkeypatch.setattr(
        int8_prefill.mx,
        "device_info",
        lambda: {"device_name": "Apple M5 Max"},
    )


@pytest.mark.skipif(
    "M5" not in str(mx.device_info().get("device_name", "")),
    reason="the int8 prefill overlay requires M5 tensor operations",
)
def test_int8_prefill_is_bounded_reversible_and_close(monkeypatch):
    monkeypatch.setattr(int8_prefill, "SCOPE", "all")
    mx.random.seed(25)
    linear = _small_w4_linear()
    x = mx.random.normal((1, 512, 1024)).astype(mx.bfloat16)

    reference = linear(x)
    mx.eval(reference)
    assert int8_prefill.apply() is True
    assert int8_prefill.apply() is False

    actual = linear(x)
    mx.eval(actual)
    rel_l2 = mx.linalg.norm(
        actual.astype(mx.float32) - reference.astype(mx.float32)
    ) / mx.maximum(mx.linalg.norm(reference.astype(mx.float32)), 1e-8)
    assert float(rel_l2) < 0.03
    # CACHE=none frees each int8 copy after its GEMM; CACHE=ttl keeps it.
    expected_copies = 1 if int8_prefill.CACHE == "ttl" else 0
    assert len(int8_prefill._int8_weights) == expected_copies
    assert len(int8_prefill._act_cache) <= 1

    assert int8_prefill.remove() is True
    assert int8_prefill.remove() is False
    restored = linear(x)
    mx.eval(restored)
    assert bool(mx.array_equal(reference, restored).item())


@pytest.mark.skipif(
    "M5" not in str(mx.device_info().get("device_name", "")),
    reason="the int8 prefill overlay requires M5 tensor operations",
)
def test_int8_prefill_preserves_float32_semantics(monkeypatch):
    monkeypatch.setattr(int8_prefill, "SCOPE", "all")
    linear = _small_w4_linear()
    x = mx.random.normal((1, 512, 1024)).astype(mx.float32)
    reference = linear(x)
    mx.eval(reference)

    int8_prefill.apply()
    actual = linear(x)
    mx.eval(actual)

    assert actual.dtype == reference.dtype == mx.float32
    assert bool(mx.array_equal(reference, actual).item())


def test_ws_cache_serves_and_validates_live_entries():
    mx.random.seed(11)
    linear = _small_w4_linear()
    ws = int8_prefill._ws_for(linear)
    # Cache hit: same module returns the same array object.
    assert int8_prefill._ws_for(linear) is ws
    assert bool(mx.allclose(ws, _expected_ws(linear)).item())


def test_ws_cache_ignores_stale_entry_after_id_reuse():
    mx.random.seed(0)
    first = _small_w4_linear()
    stale_ws = int8_prefill._ws_for(first)
    dead_ref = weakref.ref(first)
    del first
    gc.collect()
    assert dead_ref() is None

    mx.random.seed(1)
    second = _small_w4_linear()
    # Simulate CPython handing the new module the freed module's id(): plant
    # the first model's cache entry under the second module's id. With the
    # old id()-only keying this entry would be served as-is.
    _clear_overlay_caches()
    int8_prefill._ws_cache[id(second)] = (dead_ref, stale_ws)

    ws = int8_prefill._ws_for(second)
    assert ws is not stale_ws
    assert bool(mx.allclose(ws, _expected_ws(second)).item())
    assert not bool(mx.allclose(ws, stale_ws).item())
    # The stale entry was replaced and the fresh one is served afterwards.
    assert int8_prefill._ws_for(second) is ws


def test_reaper_evicts_ws_cache(monkeypatch):
    mx.random.seed(2)
    live = _small_w4_linear()
    dead = _small_w4_linear()
    live_ws = int8_prefill._ws_for(live)
    int8_prefill._ws_for(dead)
    dead_key = id(dead)
    del dead
    gc.collect()

    # Recently used: only entries whose module has been freed are pruned.
    int8_prefill._touch()
    int8_prefill._reap_once()
    assert dead_key not in int8_prefill._ws_cache
    assert int8_prefill._ws_for(live) is live_ws

    # Idle past TTL_S: the whole scale cache is evicted.
    monkeypatch.setattr(
        int8_prefill,
        "_last_use",
        time.monotonic() - int8_prefill.TTL_S - 1.0,
    )
    int8_prefill._reap_once()
    assert int8_prefill._ws_cache == {}


def test_release_at_unload_boundary_flushes_scales_once(monkeypatch):
    _fake_m5(monkeypatch)
    assert int8_prefill.apply() is True

    mx.random.seed(3)
    first_model = _small_w4_linear()
    stale_ws = int8_prefill._ws_for(first_model)
    del first_model
    gc.collect()

    remove_calls = []
    real_remove = int8_prefill.remove

    def counting_remove():
        result = real_remove()
        remove_calls.append(result)
        return result

    monkeypatch.setattr(int8_prefill, "remove", counting_remove)

    # The server's model-swap unload boundary: remove() runs exactly once,
    # the first model's scales are flushed, and the patch is reinstalled so
    # the next model keeps int8 prefill.
    assert int8_prefill.release() is True
    assert remove_calls == [True]
    assert int8_prefill._ws_cache == {}
    assert int8_prefill._int8_weights == {}
    assert int8_prefill._original_quantized_linear_call is not None
    assert int8_prefill._reaper_started is True

    # The second model computes its own scales; nothing of the first model's
    # cached state is visible after the swap.
    mx.random.seed(4)
    second_model = _small_w4_linear()
    ws = int8_prefill._ws_for(second_model)
    assert ws is not stale_ws
    assert bool(mx.allclose(ws, _expected_ws(second_model)).item())
    assert not bool(mx.allclose(ws, stale_ws).item())

    # Process shutdown: remove() runs once more, the patch stays off and the
    # reaper thread is stopped.
    assert int8_prefill.release(reapply=False) is True
    assert remove_calls == [True, True]
    assert int8_prefill._original_quantized_linear_call is None
    assert int8_prefill._reaper_started is False
    assert int8_prefill._reaper_stop is None

    # With the overlay gone, further unload boundaries are no-ops.
    assert int8_prefill.release() is False
    assert remove_calls == [True, True, False]


def test_server_unload_paths_release_int8_overlay():
    server_source = (
        Path(__file__).parents[1] / "mlx_lm" / "server.py"
    ).read_text()
    tree = ast.parse(server_source)
    funcs = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def calls(func_name, callee):
        for node in ast.walk(funcs[func_name]):
            if isinstance(node, ast.Call):
                target = node.func
                if isinstance(target, ast.Name) and target.id == callee:
                    return True
                if isinstance(target, ast.Attribute) and target.attr == callee:
                    return True
        return False

    # The server helpers delegate to int8_prefill.release()/apply().
    assert calls("_release_int8_prefill_overlay", "release")
    assert calls("_maybe_apply_int8_prefill", "apply")
    # Install site: main() honors --int8-prefill / MLX_LM_INT8_PREFILL.
    assert calls("main", "_maybe_apply_int8_prefill")
    # Every unload path reaches the release helper:
    #   - the model swap in ModelProvider._load drops the old model
    assert calls("_load", "_release_int8_prefill_overlay")
    #   - server shutdown releases without reinstalling the patch
    assert calls("_run_http_server", "_release_int8_prefill_overlay")
