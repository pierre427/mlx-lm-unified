import numpy as np
import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.spomin_live_surgery import (
    SpominLiveSurgeryManager,
    live_surgery_enabled,
)

from test_spomin_qwen4_surgery import fixture


class TinyCache:
    def __init__(self):
        self.offset = 0
        self.state = mx.array([0])

    def size(self):
        return self.offset


class TinyModel:
    def __call__(self, tokens, cache):
        cache[0].offset += int(tokens.shape[1])
        cache[0].state = mx.array([cache[0].offset])
        return mx.zeros((1, tokens.shape[1], 8))


def prepare(manager, *, has_mtp_state=False):
    model, _, cache, _, state, _, _ = fixture()
    transaction = manager.prepare(
        request_id="req-1",
        prompt_token_ids=state.transcript.token_ids,
        transcript=state.transcript,
        capacity_tokens=16,
        strategy="largest_first",
        has_mtp_state=has_mtp_state,
        has_recurrent_state=True,
        cache_is_request_private=True,
    )
    return model, cache, transaction


def test_live_surgery_is_wired_on_by_default_and_has_an_explicit_opt_out(monkeypatch):
    monkeypatch.delenv("MLX_LM_SPOMIN_LIVE_SURGERY", raising=False)
    assert live_surgery_enabled()
    monkeypatch.setenv("MLX_LM_SPOMIN_LIVE_SURGERY", "off")
    assert not live_surgery_enabled()


def test_generate_step_calls_hook_only_after_full_prompt_is_cached_and_drained():
    cache = [TinyCache()]
    seen = []

    list(
        generate_step(
            mx.array([1, 2, 3, 4]),
            TinyModel(),
            prompt_cache=cache,
            max_tokens=1,
            compiled_decode=False,
            _post_prefill_hook=lambda active: seen.append(active[0].offset),
        )
    )

    assert seen == [4]


def test_manager_applies_at_current_epoch_and_returns_compacted_prompt_key():
    manager = SpominLiveSurgeryManager(enabled=True)
    model, cache, transaction = prepare(manager)

    receipt = transaction.apply(
        model,
        [fixture()[1], cache],
        request_quiescent=True,
        device_work_drained=True,
    )

    assert receipt["status"] == "applied"
    assert receipt["reason"] == "committed"
    assert transaction.retained_token_ids == (1, 2, 3, 8, 9, 10, 11, 12)
    assert cache.offset == 8
    assert manager.snapshot()["counts"]["applied"] == 1


def test_stale_epoch_refuses_without_mutating_cache():
    manager = SpominLiveSurgeryManager(enabled=True)
    model, cache, transaction = prepare(manager)
    original = np.asarray(cache.keys).copy()
    transaction.close()

    receipt = transaction.apply(
        model,
        [fixture()[1], cache],
        request_quiescent=True,
        device_work_drained=True,
    )

    assert receipt["reason"] == "stale_epoch"
    np.testing.assert_array_equal(np.asarray(cache.keys), original)
    assert cache.offset == 12


def test_unsafe_barrier_refuses_without_consuming_or_mutating_transaction():
    manager = SpominLiveSurgeryManager(enabled=True)
    model, cache, transaction = prepare(manager)
    original = np.asarray(cache.keys).copy()

    receipt = transaction.apply(
        model,
        [fixture()[1], cache],
        request_quiescent=True,
        device_work_drained=False,
    )

    assert receipt["reason"] == "device_work_not_drained"
    assert manager.snapshot()["active_epochs"] == 1
    np.testing.assert_array_equal(np.asarray(cache.keys), original)
    assert cache.offset == 12


def test_mtp_state_and_transcript_mismatch_decline_before_opening_epoch():
    manager = SpominLiveSurgeryManager(enabled=True)
    _, _, transaction = prepare(manager, has_mtp_state=True)
    assert transaction is None
    model, _, cache, _, state, _, _ = fixture()
    transaction = manager.prepare(
        request_id="req-2",
        prompt_token_ids=(99,),
        transcript=state.transcript,
        capacity_tokens=16,
        strategy="largest_first",
        has_mtp_state=False,
        has_recurrent_state=True,
        cache_is_request_private=True,
    )
    assert transaction is None
    snapshot = manager.snapshot()
    assert snapshot["active_epochs"] == 0
    assert snapshot["counts"]["reason:mtp_state_active"] == 1
    assert snapshot["counts"]["reason:transcript_prompt_mismatch"] == 1
