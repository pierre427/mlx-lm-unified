# Copyright © 2025 Apple Inc.

"""Contracts for the default-off fused Qwen4 GDN speculative-verify path."""

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx_lm.models import qwen4_exp, qwen4_fused_gdn, qwen4_fused_gdn_verify


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeCache:
    """Stand-in for ``ArraysCache`` with the replay-closure rollback contract."""

    def __init__(self, conv_state=None, recurrent_state=None, *, speculating=True):
        self.cache = [conv_state, recurrent_state]
        self.lengths = None
        self.speculating = speculating
        self.advanced = 0
        self.spans = ()
        self.records = []
        # One ordered log of every mutation: record -> set slots -> advance.
        self.events = []

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value
        self.events.append(("set", index))

    def advance(self, amount):
        self.advanced += amount
        self.events.append(("advance", amount))

    def rollback_spans(self, length, mask=None):
        return self.spans

    def record_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None):
        self.records.append((num_tokens, fn, list(snapshot)))
        self.events.append(("record", num_tokens))


class RecordFailsCache(FakeCache):
    def record_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None):
        raise RuntimeError("Qwen4 PLE/GDN rollback span mismatch: 2 != 3")


class NoRollbackCache(FakeCache):
    rollback_spans = None


def production_values(steps=3, dtype=mx.bfloat16):
    return dict(
        qkv=FakeArray((1, steps, 10240), dtype),
        z=FakeArray((1, steps, 6144), dtype),
        b=FakeArray((1, steps, 48), dtype),
        a=FakeArray((1, steps, 48), dtype),
        conv_state=FakeArray((1, 3, 10240), dtype),
        recurrent_state=FakeArray((1, 48, 128, 128), mx.float32),
        conv_weight=FakeArray((10240, 4, 1), dtype),
        A_log=FakeArray((48,), mx.float32),
        dt_bias=FakeArray((48,), dtype),
        norm_weight=FakeArray((128,), dtype),
    )


GEOMETRY = dict(
    training=False,
    sharded=False,
    num_key_heads=16,
    num_value_heads=48,
    key_head_dim=128,
    value_head_dim=128,
    conv_kernel=4,
    gate_activation="sigmoid",
)


def admission(steps=3, *, mask=None, spans=(), speculating=True, **overrides):
    values = production_values(steps)
    geometry = dict(GEOMETRY)
    for key, value in overrides.items():
        (geometry if key in geometry else values)[key] = value
    return qwen4_fused_gdn_verify.admit_qwen4_fused_gdn_verify(
        **values,
        mask=mask,
        spans=spans,
        speculating=speculating,
        **geometry,
    )


def tiny_args():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1.0e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )


class Identity:
    def __call__(self, value):
        return value


def test_production_verify_widths_are_admitted():
    for steps in range(2, qwen4_fused_gdn_verify.MAX_VERIFY_STEPS + 1):
        result = admission(steps)
        assert result.accepted, (steps, result.reason)
    # Adaptive prompt lookup proposes spans up to 16 wide; those are the
    # widths that were declining before the bound was raised.
    # The admitted bound is the gated default; the kernel is proven wider.
    assert qwen4_fused_gdn_verify.MAX_VERIFY_STEPS == 8
    assert qwen4_fused_gdn_verify.MAX_VERIFY_WIDTH_PROVEN == 17
    # 17 is the width adaptive PLD actually presents (bonus + max_span), and
    # it dispatches correctly; it is simply not admitted by default.
    with patch.object(qwen4_fused_gdn_verify, "MAX_VERIFY_STEPS", 17):
        for steps in (9, 15, 16, 17):
            assert admission(steps).accepted


def test_verify_kernel_source_is_pinned_across_the_width_bound():
    """``S`` is a template constant, so widening the bound must not touch it.

    Every admitted width compiles from this one source; pinning its hash is
    what makes "the arithmetic for S <= 8 is unchanged" a checked claim
    rather than a reading of the diff.
    """
    digest = hashlib.sha256(qwen4_fused_gdn_verify._SOURCE.encode()).hexdigest()
    assert digest == (
        "2d5d84dc1869b7d74115605f2391e6a8e0767916db4f2214df19321641c90842"
    )


@pytest.mark.parametrize("steps", [2, 8, 9, 15, 16, 17])
def test_wide_dispatch_shapes_scale_only_the_token_axis(steps):
    """Geometry is width independent; only the snapshot extents move."""
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return [
            FakeArray(shape, dtype)
            for shape, dtype in zip(kwargs["output_shapes"], kwargs["output_dtypes"])
        ]

    values = production_values(steps=steps)
    with patch.object(qwen4_fused_gdn_verify, "_kernel", return_value=fake_kernel):
        outputs = qwen4_fused_gdn_verify.qwen4_fused_gdn_verify(
            values["qkv"],
            values["z"],
            values["b"],
            values["a"],
            values["conv_state"],
            values["conv_weight"],
            values["A_log"],
            values["dt_bias"],
            values["recurrent_state"],
            values["norm_weight"],
            1.0e-6,
            threadgroup_y=16,
        )
    assert calls[0]["grid"] == (32, 16, 48)
    assert calls[0]["threadgroup"] == (32, 16, 1)
    assert ("S", steps) in calls[0]["template"]
    assert [item.shape for item in outputs] == [
        (1, steps, 6144),
        (1, 3, 10240),
        (1, 48, 128, 128),
        (1, steps - 1, 48, 128, 128),
        (1, steps - 1, 3, 10240),
    ]


def test_single_token_batch_mask_ragged_and_plain_forwards_fall_back():
    assert admission(1).reason == "verify width 1 below 2"
    wide = qwen4_fused_gdn_verify.MAX_VERIFY_STEPS + 1
    assert admission(wide).reason == (
        f"verify width {wide} above {qwen4_fused_gdn_verify.MAX_VERIFY_STEPS}"
    )
    assert "qkv shape" in admission(qkv=FakeArray((2, 3, 10240), mx.bfloat16)).reason
    assert admission(mask=object()).reason == "masked verify"
    assert admission(spans=None).reason == "rollback geometry not describable"
    assert admission(spans=[2]).reason == "padded rollback geometry"
    assert admission(spans=[3, 3]).reason == "padded rollback geometry"
    # A fully valid one-lane slab under a ragged engine: lengths stamped, mask
    # derived from them (all ones), still exact for the mask-free kernel.
    assert admission(spans=[3]).accepted
    assert admission(spans=[3], mask=object()).accepted
    assert admission(speculating=False).reason == "not a speculative verify"
    assert admission(training=True).reason == "training"
    assert admission(sharded=True).reason == "distributed sharding"
    assert "unsupported geometry" in admission(num_key_heads=24).reason


def test_dtype_checks_are_strict():
    assert "A_log" in admission(A_log=FakeArray((48,), mx.float16)).reason
    assert admission(A_log=FakeArray((48,), mx.bfloat16)).accepted
    assert "z shape" in admission(z=FakeArray((1, 2, 6144), mx.bfloat16)).reason
    assert (
        "recurrent_state must be float32"
        in admission(recurrent_state=FakeArray((1, 48, 128, 128), mx.bfloat16)).reason
    )
    assert admission(qkv=FakeArray((1, 3, 10240), mx.float16)).reason == (
        "unsupported activation dtype mlx.core.float16"
    )


def test_kernel_dispatch_emits_restore_points_for_every_earlier_position():
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return [
            FakeArray(shape, dtype)
            for shape, dtype in zip(kwargs["output_shapes"], kwargs["output_dtypes"])
        ]

    values = production_values(steps=3)
    with patch.object(qwen4_fused_gdn_verify, "_kernel", return_value=fake_kernel):
        outputs = qwen4_fused_gdn_verify.qwen4_fused_gdn_verify(
            values["qkv"],
            values["z"],
            values["b"],
            values["a"],
            values["conv_state"],
            values["conv_weight"],
            values["A_log"],
            values["dt_bias"],
            values["recurrent_state"],
            values["norm_weight"],
            1.0e-6,
            threadgroup_y=16,
        )
    assert calls[0]["grid"] == (32, 16, 48)
    assert calls[0]["threadgroup"] == (32, 16, 1)
    assert ("S", 3) in calls[0]["template"]
    assert [item.shape for item in outputs] == [
        (1, 3, 6144),
        (1, 3, 10240),
        (1, 48, 128, 128),
        (1, 2, 48, 128, 128),
        (1, 2, 3, 10240),
    ]
    assert outputs[3].dtype == mx.float32
    assert outputs[4].dtype == mx.bfloat16


def test_probe_ladder_caches_per_width():
    with (
        patch.dict(qwen4_fused_gdn_verify._PROBED_STEPS, {}, clear=True),
        patch.object(
            qwen4_fused_gdn_verify, "fused_gdn_runtime_supported", return_value=True
        ),
        patch.object(
            qwen4_fused_gdn_verify, "probe_qwen4_fused_gdn_decode", return_value=16
        ),
        patch.object(
            qwen4_fused_gdn_verify,
            "qwen4_fused_gdn_verify",
            side_effect=[
                RuntimeError("threadgroup resources"),
                (object(),) * 5,
                (object(),) * 5,
            ],
        ) as execute,
        patch.object(qwen4_fused_gdn_verify.mx, "eval"),
    ):
        assert qwen4_fused_gdn_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 3) == 8
        assert qwen4_fused_gdn_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 3) == 8
        assert qwen4_fused_gdn_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 4) == 16
        assert (
            qwen4_fused_gdn_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, 1) is None
        )
    assert [c.kwargs["threadgroup_y"] for c in execute.call_args_list] == [16, 8, 16]


def test_resident_verify_switch_is_independent_of_decode():
    with (
        patch.object(qwen4_exp, "_FUSED_GDN_DECODE", False),
        patch.object(qwen4_exp, "_FUSED_GDN_VERIFY", False),
    ):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
    assert qwen4_exp.qwen4_fused_gdn_verify_mode_counts(layer) == {
        "stock": 1,
        "fused": 0,
    }
    assert qwen4_exp.set_qwen4_fused_gdn_verify_mode(layer, "fused") == 1
    assert layer.fused_gdn_verify_mode == "fused"
    assert layer.fused_gdn_decode_mode == "stock"
    with pytest.raises(ValueError, match="unknown fused GDN verify mode"):
        qwen4_exp.set_qwen4_fused_gdn_verify_mode(layer, "other")
    stats = qwen4_exp.qwen4_fused_gdn_stats(layer)
    assert stats["verify_calls"] == 0 and stats["verify_fallbacks"] == 0
    assert stats["verify_last_fallbacks"] == {}


def test_decode_hook_routes_speculating_multi_token_forwards_to_verify():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    sentinel = object()
    values = production_values(steps=3)
    speculating = FakeCache(values["conv_state"], values["recurrent_state"])
    with patch.object(layer, "_try_fused_verify", return_value=sentinel) as verify:
        assert (
            layer._try_fused_decode(
                values["qkv"], values["z"], values["b"], values["a"], None, speculating
            )
            is sentinel
        )
    verify.assert_called_once()
    # A speculating single-token forward keeps the decode admission (refused).
    layer.set_fused_gdn_decode_mode("fused")
    single = production_values(steps=1)
    with patch.object(layer, "_try_fused_verify") as verify:
        assert (
            layer._try_fused_decode(
                single["qkv"], single["z"], single["b"], single["a"], None, speculating
            )
            is None
        )
    verify.assert_not_called()
    assert layer.fused_gdn_decode_last_fallback == "speculative rollback"


def test_stock_mode_and_unfit_caches_do_not_probe_metal():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    values = production_values(steps=3)
    args = (values["qkv"], values["z"], values["b"], values["a"], None)
    with patch.object(qwen4_exp, "fused_gdn_runtime_supported") as runtime:
        layer.set_fused_gdn_verify_mode("stock")
        assert layer._try_fused_verify(*args, FakeCache()) is None
        assert layer.fused_gdn_verify_fallbacks == 0

        layer.set_fused_gdn_verify_mode("fused")
        assert layer._try_fused_verify(*args, FakeCache()) is None
        assert layer.fused_gdn_verify_last_fallback == "uninitialized cache"

        cache = NoRollbackCache(values["conv_state"], values["recurrent_state"])
        assert layer._try_fused_verify(*args, cache) is None
        assert layer.fused_gdn_verify_last_fallback == "cache lacks rollback records"

        cache = FakeCache(values["conv_state"], values["recurrent_state"])
        cache.spans = None  # undescribable padding geometry
        assert layer._try_fused_verify(*args, cache) is None
        assert (
            layer.fused_gdn_verify_last_fallback == "rollback geometry not describable"
        )

        cache = FakeCache(values["conv_state"], values["recurrent_state"])
        cache.spans = [2]  # a right-padded lane
        assert layer._try_fused_verify(*args, cache) is None
        assert layer.fused_gdn_verify_last_fallback == "padded rollback geometry"

        cache = FakeCache(values["conv_state"], values["recurrent_state"])
        cache.spans = [3, 3]  # more than one lane
        assert layer._try_fused_verify(*args, cache) is None
        assert layer.fused_gdn_verify_last_fallback == "padded rollback geometry"

        cache = FakeCache(
            values["conv_state"], values["recurrent_state"], speculating=False
        )
        assert layer._try_fused_verify(*args, cache) is None
        assert layer.fused_gdn_verify_last_fallback == "not a speculative verify"
    runtime.assert_not_called()
    assert layer.fused_gdn_verify_fallbacks == 6
    assert layer.fused_gdn_verify_calls == 0


def _admitted_patches(outputs):
    accepted = qwen4_fused_gdn.FusedGdnAdmission(True, "eligible")
    return (
        patch.object(qwen4_exp, "admit_qwen4_fused_gdn_verify", return_value=accepted),
        patch.object(qwen4_exp, "fused_gdn_runtime_supported", return_value=True),
        patch.object(qwen4_exp, "probe_qwen4_fused_gdn_verify", return_value=8),
        patch.object(qwen4_exp, "qwen4_fused_gdn_verify", **outputs),
    )


def test_admitted_verify_records_snapshot_closure_before_live_slots():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_verify_mode("fused")
    layer.out_proj = Identity()
    values = production_values(steps=3)
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    fused_output = FakeArray((1, 3, 6144), mx.bfloat16)
    next_conv, next_state = object(), object()
    state_snapshots = mx.arange(2 * 2 * 2 * 2, dtype=mx.float32).reshape(1, 2, 2, 2, 2)
    conv_snapshots = mx.arange(2 * 3 * 4, dtype=mx.float32).reshape(1, 2, 3, 4)
    patches = _admitted_patches(
        dict(
            return_value=(
                fused_output,
                next_conv,
                next_state,
                state_snapshots,
                conv_snapshots,
            )
        )
    )
    with patches[0], patches[1], patches[2] as probe, patches[3] as execute:
        result = layer._try_fused_verify(
            values["qkv"], values["z"], values["b"], values["a"], None, cache
        )
    assert result is fused_output
    assert probe.call_args.args == (mx.bfloat16, 3)
    assert execute.call_args.kwargs["threadgroup_y"] == 8
    assert cache[0] is next_conv and cache[1] is next_state
    assert cache.advanced == 3
    assert layer.fused_gdn_verify_calls == 1
    # The record is made before either live slot changes, with the pre-forward
    # entries as the m == 0 snapshot and the kernel's restore points for m > 0.
    assert cache.events == [("record", 3), ("set", 0), ("set", 1), ("advance", 3)]
    num_tokens, fn, snapshot = cache.records[0]
    assert num_tokens == 3
    assert snapshot == [values["conv_state"], values["recurrent_state"]]
    for m in (1, 2):
        conv_m, state_m = fn(m)
        assert mx.array_equal(conv_m, conv_snapshots[:, m - 1]).item()
        assert mx.array_equal(state_m, state_snapshots[:, m - 1]).item()


def test_record_failure_and_dispatch_failure_leave_cache_untouched():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_verify_mode("fused")
    values = production_values(steps=3)
    outputs = (
        FakeArray((1, 3, 6144), mx.bfloat16),
        object(),
        object(),
        mx.zeros((1, 2, 2, 2, 2)),
        mx.zeros((1, 2, 3, 4)),
    )
    # A snapshot-contract failure propagates (stock raises the same) and the
    # live slots and cursor are untouched.
    cache = RecordFailsCache(values["conv_state"], values["recurrent_state"])
    patches = _admitted_patches(dict(return_value=outputs))
    with patches[0], patches[1], patches[2], patches[3]:
        with pytest.raises(RuntimeError, match="span mismatch"):
            layer._try_fused_verify(
                values["qkv"], values["z"], values["b"], values["a"], None, cache
            )
    assert cache[0] is values["conv_state"] and cache[1] is values["recurrent_state"]
    assert cache.advanced == 0
    assert layer.fused_gdn_verify_calls == 0
    # A dispatch failure falls back to stock with the cache untouched.
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    patches = _admitted_patches(dict(side_effect=RuntimeError("dispatch rejected")))
    with patches[0], patches[1], patches[2], patches[3]:
        assert (
            layer._try_fused_verify(
                values["qkv"], values["z"], values["b"], values["a"], None, cache
            )
            is None
        )
    assert cache.records == [] and cache.events == []
    assert layer.fused_gdn_verify_last_fallback == (
        "Metal kernel dispatch failed: RuntimeError"
    )


def test_ragged_engine_one_lane_geometry_is_admitted_and_recorded():
    """A ragged self-MTP engine stamps ``lengths`` on every verify slab, so at
    one fully valid lane ``rollback_spans`` is ``[steps]`` (not ``()``); the
    hook must still dispatch and record exactly as in the unpadded case."""
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_verify_mode("fused")
    layer.out_proj = Identity()
    values = production_values(steps=3)
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    cache.spans = [3]
    cache.lengths = mx.array([3])
    fused_output = FakeArray((1, 3, 6144), mx.bfloat16)
    next_conv, next_state = object(), object()
    state_snapshots = mx.arange(2 * 2 * 2 * 2, dtype=mx.float32).reshape(1, 2, 2, 2, 2)
    conv_snapshots = mx.arange(2 * 3 * 4, dtype=mx.float32).reshape(1, 2, 3, 4)
    patches = _admitted_patches(
        dict(
            return_value=(
                fused_output,
                next_conv,
                next_state,
                state_snapshots,
                conv_snapshots,
            )
        )
    )
    with patches[0], patches[1], patches[2], patches[3]:
        result = layer._try_fused_verify(
            values["qkv"], values["z"], values["b"], values["a"], None, cache
        )
    assert result is fused_output
    assert layer.fused_gdn_verify_calls == 1
    assert layer.fused_gdn_verify_fallbacks == 0
    assert cache.events == [("record", 3), ("set", 0), ("set", 1), ("advance", 3)]
