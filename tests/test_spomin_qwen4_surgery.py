from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_lm.models.qwen4_exp import (
    QSAKVCache,
    TextModel,
    TextModelArgs,
    _apply_rope_positions,
)
from mlx_lm.spomin_layer import (
    SpominCapabilityError,
    SpominConfig,
    SpominLayer,
    SpominTargetState,
)
from mlx_lm.spomin_qwen4_surgery import Qwen4SpominSurgeryBackend

from test_spomin_layer import ledger


def fixture():
    args = SimpleNamespace(
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        rope_scaling={"rope_type": "default"},
    )
    layers = [SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=False)]
    model = SimpleNamespace(layers=layers, args=args)
    linear = SimpleNamespace(state="uncompacted-recurrent-summary")
    cache = QSAKVCache()
    raw = mx.arange(1 * 2 * 12 * 8, dtype=mx.float32).reshape(1, 2, 12, 8) / 100
    positions = mx.arange(12, dtype=mx.float32)[None, None, :]
    cache.keys = _apply_rope_positions(raw, positions, dims=4, base=10_000.0)
    cache.values = raw + 100
    cache.index_keys = mx.arange(1 * 12 * 3, dtype=mx.float32).reshape(1, 12, 3)
    cache.offset = 12
    transcript = ledger()
    state = SpominTargetState(
        revision="target-r1",
        target_tokens=12,
        transcript=transcript,
        visible_segment_ids=tuple(s.segment_id for s in transcript.segments),
        has_recurrent_state=True,
    )
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=16,
            pressure_ratio=0.70,
            target_ratio=0.50,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(state)
    return model, linear, cache, raw, state, layer, plan


def test_surgery_gathers_and_rephases_qsa_rows_but_preserves_recurrent_state():
    model, linear, cache, raw, state, layer, plan = fixture()
    old_linear_state = linear.state
    backend = Qwen4SpominSurgeryBackend(model, [linear, cache])

    updated = layer.apply(state, plan, backend)

    # largest_first removes turn:2 (old positions 3..6).
    keep = [0, 1, 2, 7, 8, 9, 10, 11]
    expected_raw = mx.take(raw, mx.array(keep), axis=2)
    expected_keys = _apply_rope_positions(
        expected_raw,
        mx.arange(8, dtype=mx.float32)[None, None, :],
        dims=4,
        base=10_000.0,
    )
    np.testing.assert_allclose(np.asarray(cache.keys), np.asarray(expected_keys), atol=2e-5)
    np.testing.assert_allclose(
        np.asarray(cache.values), np.asarray(mx.take(raw + 100, mx.array(keep), axis=2))
    )
    assert cache.offset == 8
    assert cache.index_keys.shape == (1, 8, 3)
    assert linear.state == old_linear_state
    assert updated.target_tokens == 8
    assert updated.visible_segment_ids == ("turn:1", "turn:3", "turn:4")


def test_preflight_refuses_armed_mtp_without_mutating_cache():
    model, linear, cache, _, state, layer, plan = fixture()
    original_keys = cache.keys
    cache._mtp_share_topk = True
    backend = Qwen4SpominSurgeryBackend(model, [linear, cache])

    with pytest.raises(SpominCapabilityError, match="armed MTP"):
        layer.apply(state, plan, backend)
    assert cache.keys is original_keys
    assert cache.offset == 12


def test_preflight_refuses_visible_transcript_mismatch():
    model, linear, cache, _, state, layer, plan = fixture()
    mismatched = SimpleNamespace(**{**state.__dict__, "target_tokens": 13})
    backend = Qwen4SpominSurgeryBackend(model, [linear, cache])

    with pytest.raises(SpominCapabilityError, match="exactly match"):
        backend.apply(mismatched, plan)


def test_preflight_refuses_nondefault_rope_scaling():
    model, linear, cache, _, state, layer, plan = fixture()
    model.args.rope_scaling = {"rope_type": "dynamic", "factor": 2.0}
    backend = Qwen4SpominSurgeryBackend(model, [linear, cache])

    with pytest.raises(SpominCapabilityError, match="dynamic"):
        layer.apply(state, plan, backend)


def test_single_attention_layer_continuation_matches_compacted_prompt_rebuild():
    args = TextModelArgs(
        hidden_size=16,
        # At layer zero, cached K/V are projected from token-local embeddings,
        # so a compacted rebuild is a valid exact oracle.  At deeper layers,
        # retained K/V intentionally keep hidden-state influence from removed
        # tokens and live surgery is not rebuild-equivalent.
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        mtp_num_hidden_layers=0,
        rope_parameters={
            "type": "default",
            "rope_theta": 10_000,
            "partial_rotary_factor": 0.5,
        },
    )
    model = TextModel(args)
    full_tokens = mx.array([list(range(1, 13))], dtype=mx.int32)
    compacted_tokens = mx.array([[1, 2, 3, 8, 9, 10, 11, 12]], dtype=mx.int32)
    next_token = mx.array([[13]], dtype=mx.int32)

    surgical_cache = model.make_cache()
    mx.eval(model(full_tokens, surgical_cache))
    transcript = ledger()
    state = SpominTargetState(
        revision="target-r1",
        target_tokens=12,
        transcript=transcript,
        visible_segment_ids=tuple(s.segment_id for s in transcript.segments),
    )
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=16,
            pressure_ratio=0.70,
            target_ratio=0.50,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    plan = layer.plan(state)
    layer.apply(
        state,
        plan,
        Qwen4SpominSurgeryBackend(model, surgical_cache),
    )
    surgical = model(next_token, surgical_cache)

    rebuilt_cache = model.make_cache()
    mx.eval(model(compacted_tokens, rebuilt_cache))
    rebuilt = model(next_token, rebuilt_cache)
    mx.eval(surgical, rebuilt)

    np.testing.assert_allclose(
        np.asarray(surgical), np.asarray(rebuilt), rtol=2e-5, atol=2e-5
    )
