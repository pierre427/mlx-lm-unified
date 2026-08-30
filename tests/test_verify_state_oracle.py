import copy
import os
import struct
from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from unittest.mock import patch

import numpy as np
import pytest


os.environ.setdefault("MLX_ENABLE_TF32", "0")
os.environ.pop("MLX_GDN_CORE", None)

import mlx.core as mx

from mlx_lm.hybrid_speculative import (
    _finalize_self_mtp_cache_group,
    _prepare_self_mtp_cache_group,
    BatchedSelfMTPState,
    DetachedSelfMTPLane,
    SelfMTPCachePair,
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from mlx_lm.models import gated_delta as gated_delta_module
from mlx_lm.models import qwen4_exp as qwen4_exp_module
from mlx_lm.models.cache import ArraysCache, _RollbackRecord
from mlx_lm.models.qwen4_exp import (
    BatchQSAKVCache,
    Model,
    ModelArgs,
    QSAKVCache,
    Qwen4ArraysCache,
    TextModelArgs,
)
from mlx_lm.sample_utils import LaneRNG


M = 4
PROMPTS = ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11])
RAGGED_ACCEPTS = tuple(product(range(M + 1), repeat=2))
DETAIL_ACCEPTS = (1, 3)


@dataclass(frozen=True)
class OracleAtom:
    kind: str
    dtype: str = ""
    shape: Tuple[int, ...] = ()
    payload: Any = None

    def summary(self) -> str:
        if self.kind == "array":
            return (
                f"array(dtype={self.dtype}, shape={self.shape}, "
                f"bytes={len(self.payload)})"
            )
        return f"{self.kind}({self.payload!r})"


Capture = Dict[str, OracleAtom]


def _array_atom(value: mx.array) -> OracleAtom:
    uint_name = {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}[
        value.dtype.size
    ]
    bits = mx.view(value, getattr(mx, uint_name))
    mx.eval(bits)
    payload = np.array(bits, copy=True).tobytes()
    return OracleAtom("array", str(value.dtype), tuple(value.shape), payload)


def _host_atom(value: Any) -> OracleAtom:
    if value is None:
        return OracleAtom("none")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        payload = struct.pack("!d", value)
    elif isinstance(value, (bool, int, str, bytes)):
        payload = value
    else:
        payload = repr(value)
    return OracleAtom("host", type(value).__name__, payload=payload)


def _put(capture: Capture, path: str, value: Any) -> None:
    if isinstance(value, mx.array):
        capture[path] = _array_atom(value)
    else:
        capture[path] = _host_atom(value)


def _capture_sequence(capture: Capture, path: str, values: Any) -> None:
    if values is None:
        _put(capture, path, None)
        return
    values = list(values)
    _put(capture, f"{path}.count", len(values))
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        if isinstance(value, mx.array) or value is None:
            _put(capture, item_path, value)
        elif isinstance(value, (list, tuple, deque)):
            _capture_sequence(capture, item_path, value)
        else:
            _put(capture, item_path, value)


def _capture_host_mirror(
    capture: Capture, path: str, mirror: Any, source: Any
) -> None:
    _put(capture, f"{path}.present", mirror is not None)
    if mirror is None:
        return
    _put(capture, f"{path}.matches_source", mirror[0] is source)
    _put(capture, f"{path}.source", mirror[0])
    _capture_sequence(capture, f"{path}.values", mirror[1])


def _materialize_tree(value: Any) -> None:
    arrays: List[mx.array] = []

    def visit(item: Any) -> None:
        if isinstance(item, mx.array):
            arrays.append(item)
        elif isinstance(item, (list, tuple, deque)):
            for child in item:
                visit(child)

    visit(value)
    if arrays:
        mx.eval(*arrays)


def _record_replays(record: _RollbackRecord) -> Dict[int, List[Any]]:
    replays = {
        m: list(record.fn(m)) for m in range(record.num_tokens + 1)
    }
    _materialize_tree(list(replays.values()))
    return replays


def _ragged_replay(
    replays: Mapping[int, Sequence[Any]], lengths: Sequence[int]
) -> List[Any]:
    selected = [replays[int(length)] for length in lengths]
    result: List[Any] = []
    for slot in range(len(selected[0])):
        values = [row[slot] for row in selected]
        if all(value is None for value in values):
            result.append(None)
        elif any(value is None for value in values):
            raise AssertionError("rollback replay mixes None and array rows")
        else:
            result.append(
                mx.concatenate(
                    [value[row : row + 1] for row, value in enumerate(values)]
                )
            )
    _materialize_tree(result)
    return result


def _ragged_vectors(num_tokens: int, batch_size: int) -> Tuple[Tuple[int, ...], ...]:
    if batch_size == 1:
        return ((0,), (num_tokens,))
    vectors = [
        (0, num_tokens),
        (num_tokens, 0),
        (min(1, num_tokens), max(0, num_tokens - 1)),
    ]
    return tuple(tuple(vector[:batch_size]) for vector in vectors)


def _capture_rollback_record(
    capture: Capture,
    path: str,
    record: _RollbackRecord,
    batch_size: int,
) -> None:
    _put(capture, f"{path}.num_tokens", record.num_tokens)
    _put(capture, f"{path}.replayable", record[0])
    _put(capture, f"{path}.span", record.span)
    _put(capture, f"{path}.depths.present", record.depths is not None)
    if record.depths is not None:
        _capture_sequence(capture, f"{path}.depths", record.depths)
    _capture_sequence(capture, f"{path}.snapshot", record.snapshot)

    replays = _record_replays(record)
    for m, replay in replays.items():
        _capture_sequence(capture, f"{path}.fn[{m}]", replay)

    _put(capture, f"{path}.per_row_fn.present", record.per_row_fn is not None)
    for lengths in _ragged_vectors(record.num_tokens, batch_size):
        label = ",".join(map(str, lengths))
        if record.per_row_fn is None:
            replay = _ragged_replay(replays, lengths)
            mode = "fallback"
        else:
            replay = list(record.per_row_fn(list(lengths)))
            _materialize_tree(replay)
            mode = "per_row_fn"
        _put(capture, f"{path}.ragged[{label}].mode", mode)
        _capture_sequence(capture, f"{path}.ragged[{label}].result", replay)


def _capture_checkpoints(capture: Capture, path: str, cache: ArraysCache) -> None:
    checkpoints = cache._checkpoints
    _put(capture, f"{path}.lane_count", len(checkpoints))
    for lane, entries in enumerate(checkpoints):
        lane_path = f"{path}.lane[{lane}]"
        _put(capture, f"{lane_path}.count", len(entries))
        for index, (position, snapshot) in enumerate(entries):
            entry_path = f"{lane_path}.entry[{index}]"
            _put(capture, f"{entry_path}.position", position)
            _capture_sequence(capture, f"{entry_path}.snapshot", snapshot)


def _capture_staged_ple(
    capture: Capture, path: str, cache: Qwen4ArraysCache
) -> None:
    staged = cache._ple_rollback
    _put(capture, f"{path}.present", staged is not None)
    if staged is None:
        return
    num_tokens, fn, snapshot, per_row_fn = staged
    _put(capture, f"{path}.num_tokens", num_tokens)
    _capture_sequence(capture, f"{path}.snapshot", snapshot)
    for m in range(num_tokens + 1):
        replay = list(fn(m))
        _materialize_tree(replay)
        _capture_sequence(capture, f"{path}.fn[{m}]", replay)
    _put(capture, f"{path}.per_row_fn.present", per_row_fn is not None)
    if per_row_fn is not None:
        for lengths in _ragged_vectors(num_tokens, cache.batch_size):
            label = ",".join(map(str, lengths))
            replay = list(per_row_fn(list(lengths)))
            _materialize_tree(replay)
            _capture_sequence(
                capture,
                f"{path}.per_row_fn[{label}]",
                replay,
            )


def capture_cache_list(caches: Sequence[Any], prefix: str = "cache") -> Capture:
    capture: Capture = {}
    _put(capture, f"{prefix}.layer_count", len(caches))
    for layer, cache in enumerate(caches):
        path = f"{prefix}.layer[{layer}]"
        _put(capture, f"{path}.type", type(cache).__name__)

        if isinstance(cache, ArraysCache):
            _capture_sequence(capture, f"{path}.cache", cache.cache)
            for name in ("left_padding", "lengths"):
                _put(capture, f"{path}.{name}", getattr(cache, name))
            _capture_host_mirror(
                capture,
                f"{path}._host_lengths",
                cache._host_lengths,
                cache.lengths,
            )
            _capture_host_mirror(
                capture,
                f"{path}._host_left_padding",
                cache._host_left_padding,
                cache.left_padding,
            )
            for name in (
                "speculating",
                "_rollback_window",
                "_rollback_invalid_reason",
            ):
                _put(capture, f"{path}.{name}", getattr(cache, name))
            _capture_checkpoints(capture, f"{path}._checkpoints", cache)
            _put(capture, f"{path}._rollbacks.count", len(cache._rollbacks))
            for index, record in enumerate(cache._rollbacks):
                _capture_rollback_record(
                    capture,
                    f"{path}._rollbacks[{index}]",
                    record,
                    cache.batch_size,
                )
            if isinstance(cache, Qwen4ArraysCache):
                _capture_staged_ple(capture, f"{path}._ple_rollback", cache)

        if isinstance(cache, (QSAKVCache, BatchQSAKVCache)):
            for name in (
                "keys",
                "values",
                "offset",
                "index_keys",
                "_qsa_pooled_keys",
                "_qsa_pooled_ratio",
                "_mtp_share_topk",
                "_mtp_shared_topk",
            ):
                _put(capture, f"{path}.{name}", getattr(cache, name))
            for name in ("left_padding", "_idx", "_right_padding"):
                if hasattr(cache, name):
                    _put(capture, f"{path}.{name}", getattr(cache, name))
            if hasattr(cache, "_max_left_pad"):
                mirror = cache._max_left_pad
                _put(capture, f"{path}._max_left_pad.present", mirror is not None)
                if mirror is not None:
                    _put(
                        capture,
                        f"{path}._max_left_pad.matches_source",
                        mirror[0] is cache.left_padding,
                    )
                    _put(capture, f"{path}._max_left_pad.source", mirror[0])
                    _put(capture, f"{path}._max_left_pad.value", mirror[1])
    return capture


def capture_batched_state(
    batch: BatchedSelfMTPState, prefix: str = "batch"
) -> Capture:
    """Capture every persistent cache, lane, RNG, and transaction surface."""

    capture: Capture = {}
    _put(capture, f"{prefix}.membership_epoch", batch.membership_epoch)
    _put(capture, f"{prefix}.proposal_open", batch.proposal_open)
    proposal = batch._open_proposal
    _put(capture, f"{prefix}._open_proposal.present", proposal is not None)
    if proposal is not None:
        for name in (
            "membership_epoch",
            "lane_uids",
            "draft_depths",
            "accepted_lengths",
            "target_drops",
            "head_drops",
        ):
            value = getattr(proposal, name)
            if isinstance(value, (list, tuple)):
                _capture_sequence(capture, f"{prefix}._open_proposal.{name}", value)
            else:
                _put(capture, f"{prefix}._open_proposal.{name}", value)
        _put(capture, f"{prefix}._open_proposal.outputs.count", len(proposal.outputs))
        for row, outputs in enumerate(proposal.outputs):
            _put(
                capture,
                f"{prefix}._open_proposal.outputs[{row}].count",
                len(outputs),
            )
            for index, token in enumerate(outputs):
                token_path = f"{prefix}._open_proposal.outputs[{row}][{index}]"
                _put(capture, f"{token_path}.token", token.token)
                _put(capture, f"{token_path}.logprobs", token.logprobs)
                _put(capture, f"{token_path}.from_draft", token.from_draft)
        for name in (
            "_old_curs",
            "_old_seed_hs",
            "_drafts",
            "_vhidden",
            "_logprobs",
            "_bonuses",
        ):
            _capture_sequence(
                capture,
                f"{prefix}._open_proposal.{name}",
                getattr(proposal, name),
            )

    _put(capture, f"{prefix}.lane_count", len(batch.lanes))
    _capture_sequence(
        capture, f"{prefix}.lane_uids", [lane.uid for lane in batch.lanes]
    )
    rng_aliases: Dict[int, int] = {}
    for row, lane in enumerate(batch.lanes):
        path = f"{prefix}.lane[{row}]"
        for name in (
            "uid",
            "cur",
            "ntoks",
            "max_tokens",
            "num_draft",
            "sampling_temp",
            "accept_rule",
            "share_qsa_indices",
        ):
            _put(capture, f"{path}.{name}", getattr(lane, name))
        for name in ("seed_h", "pending_hs", "token_prefix"):
            _put(capture, f"{path}.{name}", getattr(lane, name))
        _capture_sequence(capture, f"{path}.pending_ts", lane.pending_ts)
        for name, value in vars(lane.stats).items():
            _put(capture, f"{path}.stats.{name}", value)

        rng = lane.rng
        _put(capture, f"{path}.rng.present", rng is not None)
        if rng is not None:
            # Object addresses differ across deep clones.  Record the alias class
            # instead: 0 means the first distinct RNG, a repeated value proves two
            # lanes accidentally share one mutable stream object.
            rng_alias = rng_aliases.setdefault(id(rng), len(rng_aliases))
            _put(capture, f"{path}.rng.alias", rng_alias)
            _put(capture, f"{path}.rng.type", type(rng).__name__)
            _put(capture, f"{path}.rng.key", rng.key)
            _put(capture, f"{path}.rng.draws", rng.draws)

    _merge_capture(
        capture,
        capture_cache_list(batch.caches.target, f"{prefix}.target"),
    )
    _merge_capture(
        capture,
        capture_cache_list(batch.caches.draft, f"{prefix}.draft"),
    )
    return capture


def _prefixed(capture: Mapping[str, OracleAtom], prefix: str) -> Capture:
    return {f"{prefix}.{path}": atom for path, atom in capture.items()}


def _merge_capture(target: Capture, values: Mapping[str, OracleAtom]) -> None:
    overlap = set(target).intersection(values)
    if overlap:
        raise AssertionError(f"duplicate oracle paths: {sorted(overlap)[:3]}")
    target.update(values)


def assert_oracle_equal(
    expected: Mapping[str, OracleAtom], actual: Mapping[str, OracleAtom]
) -> None:
    all_paths = sorted(set(expected).union(actual))
    for path in all_paths:
        if path not in expected:
            raise AssertionError(f"{path}: unexpected surface in actual capture")
        if path not in actual:
            raise AssertionError(f"{path}: surface missing from actual capture")
        if expected[path] != actual[path]:
            raise AssertionError(
                f"{path}: expected {expected[path].summary()}, "
                f"got {actual[path].summary()}"
            )


def _clone_replay_fn(fn: Any, num_tokens: int):
    values = {m: copy.deepcopy(list(fn(m))) for m in range(num_tokens + 1)}
    _materialize_tree(list(values.values()))

    def replay(m: int):
        return copy.deepcopy(values[int(m)])

    return replay


def _clone_per_row_fn(fn: Any, num_tokens: int, batch_size: int):
    if fn is None:
        return None
    values = {
        lengths: copy.deepcopy(list(fn(list(lengths))))
        for lengths in product(range(num_tokens + 1), repeat=batch_size)
    }
    _materialize_tree(list(values.values()))

    def replay(lengths: Sequence[int]):
        return copy.deepcopy(values[tuple(map(int, lengths))])

    return replay


def clone_cache_list(caches: Sequence[Any]) -> List[Any]:
    clones: List[Any] = []
    for source in caches:
        if not isinstance(source, ArraysCache):
            clones.append(copy.deepcopy(source))
            continue
        cloned = type(source).__new__(type(source))
        memo: Dict[int, Any] = {}
        for name, value in vars(source).items():
            if name not in {"_rollbacks", "_ple_rollback"}:
                setattr(cloned, name, copy.deepcopy(value, memo))
        records = deque()
        for record in source._rollbacks:
            records.append(
                _RollbackRecord(
                    record.num_tokens,
                    _clone_replay_fn(record.fn, record.num_tokens),
                    copy.deepcopy(record.snapshot),
                    _clone_per_row_fn(
                        record.per_row_fn, record.num_tokens, source.batch_size
                    ),
                    None if record.depths is None else list(record.depths),
                )
            )
        cloned._rollbacks = records
        if isinstance(source, Qwen4ArraysCache) and source._ple_rollback is not None:
            num_tokens, fn, snapshot, per_row_fn = source._ple_rollback
            cloned._ple_rollback = (
                num_tokens,
                _clone_replay_fn(fn, num_tokens),
                copy.deepcopy(snapshot),
                _clone_per_row_fn(per_row_fn, num_tokens, source.batch_size),
            )
        clones.append(cloned)
    _materialize_tree([cache.state for cache in clones])
    return clones


def _clone_batch(batch: BatchedSelfMTPState) -> BatchedSelfMTPState:
    lanes = copy.deepcopy(batch.lanes)
    return BatchedSelfMTPState(
        lanes=lanes,
        caches=SelfMTPCachePair(
            target=clone_cache_list(batch.caches.target),
            draft=clone_cache_list(batch.caches.draft),
        ),
        membership_epoch=batch.membership_epoch,
        proposal_open=batch.proposal_open,
        _open_proposal=copy.deepcopy(batch._open_proposal),
    )


def _tiny_args() -> TextModelArgs:
    return TextModelArgs(
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[2],
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
        mtp_num_hidden_layers=1,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
        },
    )


def _prepare_lane(
    model: Model, uid: int, prompt: Sequence[int]
) -> DetachedSelfMTPLane:
    detached, _ = prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=uid,
        max_tokens=32,
        prompt_cache=None,
        mtp_state=None,
        lane_rng=LaneRNG(700 + uid),
        num_draft=M,
        sampling_temp=0.8,
        sampling_top_p=1.0,
        sampling_top_k=8,
        sampling_min_p=0.0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=4,
        share_qsa_indices=True,
    )
    return detached


@dataclass
class OracleFixture:
    model: Model
    base: BatchedSelfMTPState
    clone_proof_original: Capture
    clone_proof_left: Capture
    clone_proof_right: Capture
    standalone_proof: Capture
    standalone_clone: Capture


@pytest.fixture(scope="module")
def oracle_fixture() -> Iterable[OracleFixture]:
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    assert "MLX_GDN_CORE" not in os.environ
    assert not gated_delta_module._ENABLE_GDN_CORE
    assert mx.metal.is_available()

    previous_device = mx.default_device()
    previous_pooled = qwen4_exp_module._QSA_POOLED_KEY_CACHE
    mx.set_default_device(mx.gpu)
    qwen4_exp_module._QSA_POOLED_KEY_CACHE = True
    try:
        mx.random.seed(3407)
        args = _tiny_args()
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=args.__dict__))
        model.eval()
        mx.eval(model.parameters())

        detached = [
            _prepare_lane(model, uid, prompt)
            for uid, prompt in enumerate(PROMPTS)
        ]
        for item, prompt in zip(detached, PROMPTS):
            for cache in item.caches.target:
                if isinstance(cache, ArraysCache):
                    cache.state_checkpoint([len(prompt)], force=True)

        standalone = capture_cache_list(
            detached[0].caches.target, "standalone.target"
        )
        standalone_clone = capture_cache_list(
            clone_cache_list(detached[0].caches.target),
            "standalone.target",
        )

        base = attach_self_mtp_lanes(model, None, detached)
        original = capture_batched_state(base, "base")
        left = _clone_batch(base)
        right = _clone_batch(base)
        left_capture = capture_batched_state(left, "base")
        right_capture = capture_batched_state(right, "base")
        yield OracleFixture(
            model=model,
            base=base,
            clone_proof_original=original,
            clone_proof_left=left_capture,
            clone_proof_right=right_capture,
            standalone_proof=standalone,
            standalone_clone=standalone_clone,
        )
    finally:
        qwen4_exp_module._QSA_POOLED_KEY_CACHE = previous_pooled
        mx.set_default_device(previous_device)


def _capture_target(caches: Sequence[Any], path: str) -> Capture:
    return _prefixed(capture_cache_list(caches, "target"), path)


def _run_scenario(
    model: Model,
    starting: BatchedSelfMTPState,
    accepted: Tuple[int, int],
) -> Capture:
    batch = _clone_batch(starting)
    result: Capture = {}
    detailed = accepted == DETAIL_ACCEPTS
    backbone_calls = 0
    mtp_calls = 0
    logits: List[mx.array] = []

    original_backbone = model.mtp_backbone
    original_mtp_step = model.mtp_step
    original_logits = model.logits

    def checked_backbone(inputs, cache=None):
        nonlocal backbone_calls
        call = backbone_calls
        backbone_calls += 1
        if detailed and cache is batch.caches.target:
            _merge_capture(
                result,
                _capture_target(cache, f"backbone[{call}].prepared"),
            )
        output = original_backbone(inputs, cache=cache)
        _materialize_tree(output)
        if detailed and cache is batch.caches.target:
            _merge_capture(
                result,
                _capture_target(cache, f"backbone[{call}].post_forward"),
            )
        return output

    def checked_mtp_step(hidden, tokens, cache):
        nonlocal mtp_calls
        if detailed and mtp_calls == 1:
            _merge_capture(
                result,
                _prefixed(
                    capture_cache_list(cache, "draft"),
                    "draft_cycle.shared_topk_live",
                ),
            )
        output = original_mtp_step(hidden, tokens, cache)
        mtp_calls += 1
        return output

    def checked_logits(hidden):
        output = original_logits(hidden)
        mx.eval(output)
        logits.append(mx.array(output))
        return output

    model.mtp_backbone = checked_backbone
    model.mtp_step = checked_mtp_step
    model.logits = checked_logits
    try:
        pending = iter(accepted)

        def force_accept(logprobs, *_args, **_kwargs):
            count = next(pending)
            bonus = int(mx.argmax(logprobs[count]).item())
            return count, bonus

        with patch(
            "mlx_lm.hybrid_speculative._batched_residual_verify",
            side_effect=force_accept,
        ):
            proposal = propose_batched_self_mtp(model, batch)

        assert proposal.draft_depths == (M, M)
        assert proposal.accepted_lengths == accepted
        assert proposal.target_drops == tuple(M - value for value in accepted)
        assert len(logits) == M + 1
        _put(result, "verify.accepted[0]", accepted[0])
        _put(result, "verify.accepted[1]", accepted[1])
        _put(result, "verify.target_drops[0]", proposal.target_drops[0])
        _put(result, "verify.target_drops[1]", proposal.target_drops[1])
        _put(result, "verify.logits", logits[-1])
        _merge_capture(
            result,
            _prefixed(capture_batched_state(batch), "post_proposal"),
        )
        if detailed:
            live = capture_batched_state(batch, "post_proposal_clone")
            cloned = capture_batched_state(
                _clone_batch(batch), "post_proposal_clone"
            )
            assert_oracle_equal(live, cloned)
            _put(result, "post_proposal.deep_clone_with_rollbacks", True)

        commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[len(row) for row in proposal.outputs],
            terminal=[False, False],
        )
        _merge_capture(
            result,
            _prefixed(capture_batched_state(batch), "post_commit"),
        )

        continuation_ids = mx.array(
            [[lane.cur] for lane in batch.lanes], dtype=mx.uint32
        )
        _prepare_self_mtp_cache_group(
            batch.caches.target, lengths=[1, 1], right_padding=[0, 0]
        )
        try:
            continuation_hidden, _ = model.mtp_backbone(
                continuation_ids, cache=batch.caches.target
            )
            continuation_logits = model.logits(continuation_hidden)
            mx.eval(continuation_logits)
        finally:
            _finalize_self_mtp_cache_group(batch.caches.target)
        assert len(logits) == M + 2
        _put(result, "continuation.logits", logits[-1])
        _merge_capture(
            result,
            _prefixed(capture_batched_state(batch), "post_continuation"),
        )
        _put(result, "route.mtp_step_calls", mtp_calls)
        _put(result, "route.backbone_calls", backbone_calls)
    finally:
        model.mtp_backbone = original_backbone
        model.mtp_step = original_mtp_step
        model.logits = original_logits
    return result


def _run_stream(model: Model, base: BatchedSelfMTPState) -> Capture:
    stream: Capture = {}
    for accepted in RAGGED_ACCEPTS:
        label = f"accept[{accepted[0]},{accepted[1]}]"
        _merge_capture(stream, _prefixed(_run_scenario(model, base, accepted), label))
    return stream


@dataclass
class OracleRun:
    left: Capture
    right: Capture
    kernel_calls: int
    ops_calls: int
    core_calls: int


@pytest.fixture(scope="module")
def oracle_run(oracle_fixture: OracleFixture) -> OracleRun:
    local_kernel = gated_delta_module.gated_delta_kernel
    ops = gated_delta_module.gated_delta_ops
    core = gated_delta_module._core_gated_delta_update
    left_base = _clone_batch(oracle_fixture.base)
    right_base = _clone_batch(oracle_fixture.base)
    with (
        patch.object(
            gated_delta_module,
            "gated_delta_kernel",
            wraps=local_kernel,
        ) as kernel_spy,
        patch.object(gated_delta_module, "gated_delta_ops", wraps=ops) as ops_spy,
        patch.object(
            gated_delta_module,
            "_core_gated_delta_update",
            wraps=core,
        ) as core_spy,
    ):
        left = _run_stream(oracle_fixture.model, left_base)
        right = _run_stream(oracle_fixture.model, right_base)
    return OracleRun(
        left=left,
        right=right,
        kernel_calls=kernel_spy.call_count,
        ops_calls=ops_spy.call_count,
        core_calls=core_spy.call_count,
    )


def test_fixture_geometry_and_deep_clone(oracle_fixture: OracleFixture) -> None:
    model = oracle_fixture.model
    assert mx.default_device() == mx.gpu
    assert not model.training
    assert model.language_model.args.num_hidden_layers == 3
    assert model.language_model.args.ple_layer_ids == [2]
    assert model.language_model.args.layer_types == [
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert [type(cache) for cache in oracle_fixture.base.caches.target] == [
        ArraysCache,
        Qwen4ArraysCache,
        BatchQSAKVCache,
    ]
    assert any(
        atom.payload == "QSAKVCache"
        for path, atom in oracle_fixture.standalone_proof.items()
        if path.endswith(".type")
    )
    assert_oracle_equal(
        oracle_fixture.standalone_proof,
        oracle_fixture.standalone_clone,
    )
    assert_oracle_equal(
        oracle_fixture.clone_proof_original,
        oracle_fixture.clone_proof_left,
    )
    assert_oracle_equal(
        oracle_fixture.clone_proof_original,
        oracle_fixture.clone_proof_right,
    )


def test_production_local_metal_gdn_kernel_ran(oracle_run: OracleRun) -> None:
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    assert "MLX_GDN_CORE" not in os.environ
    assert mx.default_device() == mx.gpu
    assert mx.metal.is_available()
    assert not gated_delta_module._ENABLE_GDN_CORE
    assert oracle_run.kernel_calls > 0
    assert oracle_run.ops_calls == 0
    assert oracle_run.core_calls == 0


def test_eager_clone_stream_is_raw_bit_identical(oracle_run: OracleRun) -> None:
    assert len(RAGGED_ACCEPTS) == (M + 1) ** 2
    assert_oracle_equal(oracle_run.left, oracle_run.right)


REQUIRED_SURFACE_FRAGMENTS = (
    ".layer[0].cache[0]",
    ".layer[0].cache[1]",
    ".layer[1].cache[0]",
    ".layer[1].cache[1]",
    ".layer[1].cache[2]",
    ".layer[1].cache[3]",
    ".layer[2].keys",
    ".layer[2].values",
    ".layer[2].offset",
    ".layer[2].index_keys",
    ".layer[2]._qsa_pooled_keys",
    ".layer[2]._qsa_pooled_ratio",
    "shared_topk_live.draft.layer[0]._mtp_shared_topk",
    "._mtp_share_topk",
    ".left_padding",
    ".lengths",
    "._host_lengths.present",
    "._host_left_padding.present",
    "._rollback_window",
    "._rollback_invalid_reason",
    "._checkpoints.lane[0].entry[0].position",
    "._checkpoints.lane[0].entry[0].snapshot[0]",
    "._ple_rollback.present",
    "._rollbacks[0].num_tokens",
    "._rollbacks[0].depths[0]",
    "._rollbacks[0].snapshot[0]",
    "._rollbacks[0].fn[0][0]",
    "._rollbacks[0].ragged[0,5].result[0]",
    ".verify.logits",
    ".continuation.logits",
    ".batch.membership_epoch",
    ".batch.proposal_open",
    ".batch._open_proposal.present",
    ".batch._open_proposal.outputs[0][0].logprobs",
    ".batch._open_proposal._old_seed_hs[0]",
    ".batch.lane[0].uid",
    ".batch.lane_uids[0]",
    ".batch.lane[0].cur",
    ".batch.lane[0].seed_h",
    ".batch.lane[0].pending_hs",
    ".batch.lane[0].pending_ts.count",
    ".batch.lane[0].token_prefix",
    ".batch.lane[0].rng.alias",
    ".batch.lane[0].rng.key",
    ".batch.lane[0].rng.draws",
    ".batch.lane[0].ntoks",
    ".batch.lane[0].stats.draft_cycles",
    ".batch.target.layer[0].cache[0]",
    ".batch.draft.layer[0].keys",
)


def _flip_atom(atom: OracleAtom) -> OracleAtom:
    if atom.kind == "array":
        if not atom.payload:
            return OracleAtom("array", atom.dtype, atom.shape + (1,), b"\x01")
        payload = bytearray(atom.payload)
        payload[0] ^= 1
        return OracleAtom("array", atom.dtype, atom.shape, bytes(payload))
    if atom.kind == "none":
        return OracleAtom("array", "uint8", (1,), b"\x01")
    if atom.dtype == "bool":
        return OracleAtom("host", atom.dtype, payload=not atom.payload)
    if atom.dtype == "int":
        return OracleAtom("host", atom.dtype, payload=int(atom.payload) ^ 1)
    if atom.dtype == "float":
        payload = bytearray(atom.payload)
        payload[-1] ^= 1
        return OracleAtom("host", atom.dtype, payload=bytes(payload))
    if isinstance(atom.payload, bytes):
        payload = bytearray(atom.payload or b"\x00")
        payload[0] ^= 1
        return OracleAtom("host", atom.dtype, payload=bytes(payload))
    return OracleAtom("host", atom.dtype, payload=f"{atom.payload}#planted")


def test_planted_one_bit_diff_is_detected_on_every_surface(
    oracle_fixture: OracleFixture,
    oracle_run: OracleRun,
) -> None:
    representative_prefix = f"accept[{DETAIL_ACCEPTS[0]},{DETAIL_ACCEPTS[1]}]."
    representative = {
        path: atom
        for path, atom in oracle_run.left.items()
        if path.startswith(representative_prefix)
    }
    planted_baseline: Capture = {}
    _merge_capture(planted_baseline, oracle_fixture.standalone_proof)
    _merge_capture(planted_baseline, representative)

    missing = [
        fragment
        for fragment in REQUIRED_SURFACE_FRAGMENTS
        if not any(fragment in path for path in planted_baseline)
    ]
    assert not missing, f"oracle omitted required surfaces: {missing}"

    for path, atom in planted_baseline.items():
        planted = dict(planted_baseline)
        planted[path] = _flip_atom(atom)
        with pytest.raises(AssertionError) as error:
            assert_oracle_equal(planted_baseline, planted)
        assert str(error.value).startswith(f"{path}:")


def _distinct_like(value: Any) -> mx.array:
    """A same-shape/dtype array whose bits differ from any realistic state."""
    if value is None:
        return mx.array([1], dtype=mx.uint8)
    filled = mx.full(value.shape, 7, dtype=value.dtype)
    mx.eval(filled)
    return filled


def test_real_cache_mutation_propagates_to_capture(
    oracle_fixture: OracleFixture,
) -> None:
    """Airtight completeness: perturbing a REAL cache leaf (not the captured
    dict) on one arm must make the oracle report a diff localized to that
    surface. This proves capture() reads each surface from the correct live
    object — the property the dict-mutation planted-diff cannot establish (a
    capture that read the wrong/duplicate object would pass self-consistency,
    the presence check, and the comparator test, yet miss a real state change).
    """
    caches = oracle_fixture.base.caches.target
    baseline = capture_cache_list(clone_cache_list(caches), "target")

    def _mut_gdn_matrix(cl: List[Any]) -> None:  # layer 0 plain GDN, cache[1]
        cl[0].cache[1] = _distinct_like(cl[0].cache[1])

    def _mut_gdn_conv(cl: List[Any]) -> None:  # layer 1 PLE+GDN, cache[0]
        cl[1].cache[0] = _distinct_like(cl[1].cache[0])

    def _mut_ple_conv(cl: List[Any]) -> None:  # layer 1 PLE, cache[2]
        cl[1].cache[2] = _distinct_like(cl[1].cache[2])

    def _mut_ple_history(cl: List[Any]) -> None:  # layer 1 PLE, cache[3]
        cl[1].cache[3] = _distinct_like(cl[1].cache[3])

    def _mut_qsa_keys(cl: List[Any]) -> None:  # layer 2 QSA K
        cl[2].keys = _distinct_like(cl[2].keys)

    def _mut_qsa_ledger(cl: List[Any]) -> None:  # layer 2 QSA raw-key ledger
        cl[2].index_keys = _distinct_like(cl[2].index_keys)

    def _mut_offset(cl: List[Any]) -> None:  # layer 2 host offset (int or batched array)
        cl[2].offset = cl[2].offset + 1
        if isinstance(cl[2].offset, mx.array):
            mx.eval(cl[2].offset)

    mutations = {
        "target.layer[0].cache[1]": _mut_gdn_matrix,
        "target.layer[1].cache[0]": _mut_gdn_conv,
        "target.layer[1].cache[2]": _mut_ple_conv,
        "target.layer[1].cache[3]": _mut_ple_history,
        "target.layer[2].keys": _mut_qsa_keys,
        "target.layer[2].index_keys": _mut_qsa_ledger,
        "target.layer[2].offset": _mut_offset,
    }

    for expected_path, mutate in mutations.items():
        clone = clone_cache_list(caches)
        mutate(clone)
        mutated = capture_cache_list(clone, "target")
        with pytest.raises(AssertionError) as error:
            assert_oracle_equal(baseline, mutated)
        # The reported diff must localize to the surface we actually perturbed,
        # not some incidental leaf — proves the capture path/object mapping.
        assert str(error.value).startswith(expected_path), (
            f"perturbed {expected_path} but oracle reported {error.value!r}"
        )


def test_real_batched_state_mutation_propagates_to_capture(
    oracle_fixture: OracleFixture,
) -> None:
    """Prove the expanded walker reads live draft, lane, RNG, and batch state."""

    baseline_batch = _clone_batch(oracle_fixture.base)
    baseline = capture_batched_state(baseline_batch)

    def check(path: str, mutate) -> None:
        batch = _clone_batch(baseline_batch)
        mutate(batch)
        actual = capture_batched_state(batch)
        assert baseline[path] != actual[path], f"mutation did not reach {path}"
        with pytest.raises(AssertionError):
            assert_oracle_equal(baseline, actual)

    check(
        "batch.membership_epoch",
        lambda batch: setattr(batch, "membership_epoch", batch.membership_epoch + 1),
    )
    check(
        "batch.proposal_open",
        lambda batch: setattr(batch, "proposal_open", True),
    )
    check("batch.lane[0].uid", lambda batch: setattr(batch.lanes[0], "uid", 99))
    check(
        "batch.lane_uids[0]",
        lambda batch: batch.lanes.reverse(),
    )
    check(
        "batch.lane[0].cur",
        lambda batch: setattr(batch.lanes[0], "cur", batch.lanes[0].cur + 1),
    )
    check(
        "batch.lane[0].seed_h",
        lambda batch: setattr(
            batch.lanes[0], "seed_h", _distinct_like(batch.lanes[0].seed_h)
        ),
    )
    check(
        "batch.lane[0].pending_hs",
        lambda batch: setattr(
            batch.lanes[0], "pending_hs", mx.ones_like(batch.lanes[0].seed_h)
        ),
    )
    check(
        "batch.lane[0].pending_ts.count",
        lambda batch: batch.lanes[0].pending_ts.append(17),
    )
    check(
        "batch.lane[0].token_prefix",
        lambda batch: setattr(
            batch.lanes[0],
            "token_prefix",
            _distinct_like(batch.lanes[0].token_prefix),
        ),
    )
    check(
        "batch.lane[0].rng.key",
        lambda batch: setattr(
            batch.lanes[0].rng, "_key", _distinct_like(batch.lanes[0].rng.key)
        ),
    )
    check(
        "batch.lane[0].rng.draws",
        lambda batch: setattr(batch.lanes[0].rng, "_draws", batch.lanes[0].rng.draws + 1),
    )
    check(
        "batch.lane[1].rng.alias",
        lambda batch: setattr(batch.lanes[1], "rng", batch.lanes[0].rng),
    )
    check(
        "batch.lane[0].ntoks",
        lambda batch: setattr(batch.lanes[0], "ntoks", batch.lanes[0].ntoks + 1),
    )
    check(
        "batch.lane[0].stats.draft_cycles",
        lambda batch: setattr(
            batch.lanes[0].stats,
            "draft_cycles",
            batch.lanes[0].stats.draft_cycles + 1,
        ),
    )
    check(
        "batch.draft.layer[0].keys",
        lambda batch: setattr(
            batch.caches.draft[0],
            "keys",
            _distinct_like(batch.caches.draft[0].keys),
        ),
    )
