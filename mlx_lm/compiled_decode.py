# Copyright © 2026 Apple Inc.

"""Trace the decode step once, replay it every token.

The stock decode loop rebuilds the token's MLX graph in Python on every
step and hides most of that cost under depth-1 ``async_eval``. What
pipelining cannot hide is still ~0.2 ms per four layers at width 1 --
about 2.7 ms of a 28 ms token on a 48-layer model, the same order as the
whole dispatch floor, for no kernel work at all. Tracing the step once
with ``mx.compile`` and replaying it removes that. Measured on a
production-shape 4-layer Flash-Next stack: 2.257 -> 2.035 ms per step at
M=1 (1.109x), 2.721 -> 2.599 at M=3. See
``wiki/docs/research/qwen4-compiled-replay-microbench-2026-09-04.md``.

Replay has one hard precondition: **every array the step touches keeps
its shape**. ``KVCache`` fails it twice over -- it grows the slab by
concatenating 256-token blocks and hands attention a Python-sliced view
whose length is the token count -- so this module drives
``cache.RingKVCache`` instead (fixed slab, ``mx.array`` offset, mask
built in the graph). ``ArraysCache`` (the GDN conv + recurrent state) is
already fixed-shape and needs no replacement, only explicit threading.

State is threaded **explicitly**, not captured: the traced function takes
the cache arrays as arguments and returns their successors, so nothing a
replay depends on is baked in as a constant. The wrapper writes the
returned arrays back into the cache objects and advances their host-side
mirrors. Sampling stays outside the traced function -- the compiled step
returns logits, exactly as production reads them today.

Not covered, deliberately:

* ``qwen4_exp.QSAKVCache`` (Flash-Next indexed attention). Its
  ``index_keys`` grow by concatenation on every token, its pooled
  block-summary ledger grows by concatenation each time a block closes,
  and the indexer branches on Python ints (``count == n_blocks``,
  ``count > closed``) to decide what to recompute.
  ``model_is_compilable`` rejects it by name.
* Speculation. ``ArraysCache.record_rollback`` stashes a Python closure
  over the step's own intermediates; under tracing that closure would
  capture tracer arrays. ``CompiledDecodeStep`` refuses to run while any
  cache has ``speculating`` set, so the self-MTP verify width keeps the
  eager path until the rollback record is expressible as array state.
* Batched decode with per-row ``lengths``/``left_padding``: those are
  host vectors read with ``tolist()``.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import mlx.core as mx
from mlx.utils import tree_flatten

from .compiled_qualification import serving_qualification_reason
from .models.cache import ArraysCache, KVCache, RingKVCache, _ring_buckets
from .models.precise_ops import precise_span

__all__ = [
    "CompiledDecodeStep",
    "CompiledDecodePoisoned",
    "CompiledDecodePolicy",
    "compiled_decode_step",
    "compiled_decode_enabled",
    "compiled_decode_context_policy",
    "compiled_decode_numerics_accepted",
    "model_is_compilable",
    "compiled_decode_serving_reason",
    "to_shape_stable_cache",
]


_QUALIFIED_MODEL_TYPES = ("qwen3_5_moe",)
_QUALIFIED_MODEL_CLASSES = (
    ("mlx_lm.models.qwen3_5", "Model"),
    ("mlx_lm.models.qwen3_5", "TextModel"),
    ("mlx_lm.models.qwen3_5_moe", "Model"),
)
_QUALIFICATION_TOKEN = "qwen3_5_moe_m1_v1"
_PADDED_SDPA_ACCEPTANCE = "class3-padded-sdpa-v1"
_CLASS1_BUCKET_ACCEPTANCE = "class1-bucketed-v1"
_SHORT_CONTEXT_LIMIT = 4096
_EXTENDED_CONTEXT_LIMIT = 16384
_LONG_CONTEXT_LIMIT = 262144
# Context limit per profile. ``long`` (2026-09-05, operator request) extends
# replay to the model's full window on its own ladder; the three original
# profiles and their qualified ladders are unchanged.
_PROFILE_LIMITS = {
    "short": _SHORT_CONTEXT_LIMIT,
    "memory": _EXTENDED_CONTEXT_LIMIT,
    "latency": _EXTENDED_CONTEXT_LIMIT,
    "long": _LONG_CONTEXT_LIMIT,
}
_CONTEXT_POLICIES = tuple(_PROFILE_LIMITS)
_LONG_LADDER_EXTENSION = (131072, 262144)
_MAX_VARIANTS_LIMIT = 64


@dataclass(frozen=True)
class CompiledDecodePolicy:
    """Resolved context limit and KV bucket choice for one request."""

    name: str
    max_context: int
    buckets: tuple
    numerical_acceptance: Optional[str] = None


class CompiledDecodePoisoned(RuntimeError):
    """The compiled step failed after state entered its transaction boundary.

    A poisoned step must be discarded.  In particular, callers must never
    retry its last token eagerly: a deferred device failure may have consumed
    only part of the submitted graph.
    """


def _numerical_acceptance() -> Optional[str]:
    value = os.environ.get("MLX_LM_COMPILED_DECODE_ACCEPTANCE", "").strip()
    return value or None


def compiled_decode_enabled() -> bool:
    """Default ON for width-1 decode (operator decision 2026-09-05, after the
    class-1 bucket ladder made every qualified operating point bit-identical to
    stock); ``MLX_LM_COMPILED_DECODE=0`` opts out. Enablement is still gated per
    request by the context policy, the numerical acceptance, and the reviewed
    serving qualification bound at model load."""
    value = os.environ.get("MLX_LM_COMPILED_DECODE", "1").strip().lower()
    return value not in ("0", "false", "no", "off", "")


def _max_variants() -> int:
    raw = os.environ.get("MLX_LM_COMPILED_DECODE_MAX_VARIANTS", "16")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "MLX_LM_COMPILED_DECODE_MAX_VARIANTS must be an integer"
        ) from exc
    return _validate_max_variants(value)


def _validate_max_variants(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_VARIANTS_LIMIT
    ):
        raise ValueError(
            f"max variants must be an integer between 1 and {_MAX_VARIANTS_LIMIT}"
        )
    return value


def compiled_decode_context_policy(
    context_tokens: int,
    max_tokens: int,
    policy: Optional[str] = None,
):
    """Return ``(decline_reason, policy)`` for a replay request.

    ``short`` is the qualified default and stops at 4K. ``memory`` and
    ``latency`` are explicit 16K profiles: the first keeps the 16384 bucket;
    the second skips it and accepts the 32768-bucket memory cost. ``long``
    admits the full 256K window on the memory ladder extended geometrically
    (131072, 262144) so a long context costs O(log n) traces rather than one
    per 65536 columns. No profile admits an unbounded completion.
    """
    policy = (
        os.environ.get("MLX_LM_COMPILED_DECODE_CONTEXT_POLICY", "short")
        if policy is None
        else policy
    ).strip().lower()
    if policy not in _CONTEXT_POLICIES:
        return (
            "MLX_LM_COMPILED_DECODE_CONTEXT_POLICY must be one of "
            + ", ".join(_CONTEXT_POLICIES),
            None,
        )
    if context_tokens < 0:
        return "negative context length", None
    if max_tokens < 0:
        return "an unbounded completion", None

    projected = context_tokens + max_tokens
    limit = _PROFILE_LIMITS[policy]
    if projected > limit:
        return (
            f"projected context {projected} exceeds the {policy} profile "
            f"limit of {limit}",
            None,
        )

    try:
        buckets = _resolved_policy_buckets(policy)
    except (TypeError, ValueError) as exc:
        return str(exc), None
    acceptance = _numerical_acceptance()
    if acceptance not in (None, _PADDED_SDPA_ACCEPTANCE):
        return (
            "MLX_LM_COMPILED_DECODE_ACCEPTANCE must be "
            f"{_PADDED_SDPA_ACCEPTANCE!r}",
            None,
        )
    if acceptance is None and _class1_bucket_ladder(buckets):
        # No reorder exists on this ladder, so there is nothing to accept; the
        # policy carries the class-1 marker that the serving manifest binds.
        acceptance = _CLASS1_BUCKET_ACCEPTANCE
    return None, CompiledDecodePolicy(
        policy,
        limit,
        buckets,
        numerical_acceptance=acceptance,
    )


def _resolved_policy_buckets(policy_name: str) -> tuple:
    buckets = _ring_buckets()
    if policy_name == "latency":
        # The 16384 SDPA operating point is slow on the qualified 35B model.
        # Make the 2x memory choice local to this request instead of changing
        # RingKVCache's global, memory-efficient default.
        buckets = tuple(sorted((set(buckets) - {16384}) | {32768}))
    elif policy_name == "long":
        # Extend the memory ladder to the full window without touching the
        # global default the three qualified profiles bind.
        buckets = tuple(sorted(set(buckets) | set(_LONG_LADDER_EXTENSION)))
    return buckets


def _validate_context_policy(policy: CompiledDecodePolicy) -> CompiledDecodePolicy:
    if not isinstance(policy, CompiledDecodePolicy):
        raise TypeError("context_policy must be a CompiledDecodePolicy")
    if policy.name not in _CONTEXT_POLICIES:
        raise ValueError(f"unknown compiled decode context policy {policy.name!r}")
    expected_limit = _PROFILE_LIMITS[policy.name]
    if policy.max_context != expected_limit:
        raise ValueError(
            f"{policy.name} policy limit must be {expected_limit}, "
            f"not {policy.max_context}"
        )
    buckets = policy.buckets
    if (
        not isinstance(buckets, tuple)
        or not buckets
        or any(
            isinstance(b, bool) or not isinstance(b, int) or b <= 0
            for b in buckets
        )
        or tuple(sorted(set(buckets))) != buckets
    ):
        raise ValueError(
            "context policy buckets must be sorted unique positive integers"
        )
    expected_buckets = _resolved_policy_buckets(policy.name)
    if buckets != expected_buckets:
        raise ValueError(
            f"context policy buckets do not match the resolved {policy.name} profile"
        )
    if policy.numerical_acceptance not in (
        None, _PADDED_SDPA_ACCEPTANCE, _CLASS1_BUCKET_ACCEPTANCE
    ):
        raise ValueError(
            "unknown compiled decode numerical acceptance token "
            f"{policy.numerical_acceptance!r}"
        )
    return policy


def _class1_bucket_ladder(buckets) -> bool:
    """The ladder measured bit-identical to stock at every operating point and
    growth boundary (2026-09-04): it carries both the 1023 bucket (sub-1,024 live
    lengths stay on SDPA's single-pass kernel, as stock does) and the 1024 bucket
    (a cache of exactly 1,024 keys is not padded into a 2,048 slab)."""
    return 1023 in buckets and 1024 in buckets


def compiled_decode_numerics_accepted(policy: CompiledDecodePolicy) -> bool:
    """Whether this policy's numerical class is accepted: either the bucket
    ladder is the class-1 ladder, or the padded-SDPA reorder (class 3) was
    accepted explicitly through ``MLX_LM_COMPILED_DECODE_ACCEPTANCE``."""
    return _validate_context_policy(policy).numerical_acceptance in (
        _CLASS1_BUCKET_ACCEPTANCE,
        _PADDED_SDPA_ACCEPTANCE,
    )


# --------------------------------------------------------------------------
# cache conversion
# --------------------------------------------------------------------------


def to_shape_stable_cache(
    cache: List[Any], buckets=None, *, in_place: bool = True
) -> List[Any]:
    """Return ``cache`` with every ``KVCache`` swapped for a ``RingKVCache``.

    By default the caller's list object is mutated so any other holder of it
    (a server session, a prompt-cache entry) sees the swap. ``in_place=False``
    returns a candidate list without committing it, for transactional setup.
    ``ArraysCache`` entries pass through -- they are already fixed-shape.
    Raises for any other cache type, because silently leaving a growing
    cache in the list would make the compiled step wrong rather than slow.
    """
    for i, c in enumerate(cache):
        if isinstance(c, RingKVCache):
            if buckets is not None and tuple(c.buckets) != tuple(buckets):
                raise TypeError(
                    f"RingKVCache at cache[{i}] uses a different bucket policy"
                )
            continue
        if type(c) is KVCache or isinstance(c, ArraysCache):
            continue
        raise TypeError(
            f"{type(c).__name__} at cache[{i}] is not shape-stable; "
            "compiled decode supports KVCache/RingKVCache and ArraysCache"
        )

    # Build every replacement before changing the caller's list. Allocation
    # or conversion failure must leave an eager prompt cache untouched.
    converted = [
        RingKVCache.from_kv_cache(c, buckets=buckets) if type(c) is KVCache else c
        for c in cache
    ]
    if in_place:
        cache[:] = converted
        return cache
    return converted


def _model_type(model) -> str:
    value = getattr(model, "model_type", None)
    if value is None:
        value = getattr(getattr(model, "args", None), "model_type", None)
    return value if isinstance(value, str) else ""


def _qualified_model_reason(model) -> Optional[str]:
    """Research topology eligibility, not a checkpoint serving approval."""
    model_class = (type(model).__module__, type(model).__name__)
    model_type = _model_type(model)
    if model_class not in _QUALIFIED_MODEL_CLASSES:
        return f"model class {model_class[0]}.{model_class[1]} has not been qualified"
    if model_type not in _QUALIFIED_MODEL_TYPES:
        return f"model type {model_type!r} has not been qualified"
    if getattr(model, "supports_compiled_decode_replay", None) != _QUALIFICATION_TOKEN:
        return "model does not carry the exact compiled replay qualification token"

    text_model = getattr(model, "language_model", model)
    args = getattr(text_model, "args", None)
    if (
        not isinstance(getattr(args, "num_experts", None), int)
        or args.num_experts <= 0
    ):
        return "only the qwen3_5 MoE topology has been qualified"
    pipeline = getattr(text_model, "model", None)
    if getattr(pipeline, "pipeline_size", 1) != 1:
        return "pipeline-parallel execution has not been qualified"

    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for _, module in named_modules():
            if getattr(module, "sharding_group", None) is not None:
                return "tensor-parallel execution has not been qualified"
    return None


def _cache_geometry_reason(cache: Sequence[Any]) -> Optional[str]:
    has_kv = False
    kv_positions = []
    for i, c in enumerate(cache):
        if type(c) is RingKVCache or type(c) is KVCache:
            has_kv = True
            if c.keys is None or c.values is None:
                return f"cache[{i}] has not been filled by a forward yet"
            if c.keys.ndim < 3 or c.values.ndim < 3:
                return f"cache[{i}] has invalid KV rank"
            if c.keys.shape[0] != 1 or c.values.shape[0] != 1:
                return f"cache[{i}] is not batch 1"
            if c.keys.shape[0:3] != c.values.shape[0:3]:
                return f"cache[{i}] has mismatched key/value geometry"
            if type(c) is RingKVCache:
                if c.offset.ndim != 0:
                    return f"cache[{i}] has a non-scalar ring offset"
                if c.capacity != c.keys.shape[2]:
                    return f"cache[{i}] has a stale ring capacity"
                if not 0 <= c.size() <= c.capacity:
                    return f"cache[{i}] has an out-of-range ring offset"
                kv_positions.append(c.size())
            elif not 0 <= c.offset <= c.keys.shape[2]:
                return f"cache[{i}] has an out-of-range KV offset"
            else:
                kv_positions.append(c.offset)
            continue
        if type(c) is ArraysCache:
            if getattr(c, "lengths", None) is not None:
                return f"cache[{i}] carries per-row lengths (batched decode)"
            if getattr(c, "left_padding", None) is not None:
                return f"cache[{i}] carries left padding (batched decode)"
            if any(a is None for a in c.cache):
                return f"cache[{i}] has not been filled by a forward yet"
            if any(a.ndim < 1 or a.shape[0] != 1 for a in c.cache):
                return f"cache[{i}] is not batch 1"
            continue
        return f"cache[{i}] is a {type(c).__name__}, which is not shape-stable"
    if not has_kv:
        return "model has no full-attention KV cache"
    if len(set(kv_positions)) != 1:
        return "full-attention KV cache positions are not synchronized"
    return None


def model_is_compilable(model, cache: Sequence[Any]) -> Optional[str]:
    """Research eligibility only; serving also needs loader-bound evidence."""
    why = _qualified_model_reason(model)
    return why if why is not None else _cache_geometry_reason(cache)


def compiled_decode_serving_reason(model, policy=None) -> Optional[str]:
    why = _qualified_model_reason(model)
    if why is not None:
        return why
    return serving_qualification_reason(
        model, parameters=tree_flatten(model.parameters()), policy=policy
    )


# --------------------------------------------------------------------------
# state threading
# --------------------------------------------------------------------------


class _RingSlot:
    """State plan for one ``RingKVCache``: keys, values, offset."""

    n_arrays = 3

    def __init__(self, cache):
        self.cache = cache

    def collect(self):
        return [self.cache.keys, self.cache.values, self.cache.offset]

    def install(self, arrays):
        self.cache.keys, self.cache.values, self.cache.offset = arrays

    def signature(self):
        # Only the capacity moves: dtype and the head geometry are fixed by
        # the model, and the write is always at an array index.
        return self.cache.capacity

    def host_state(self):
        return self.cache._host_offset

    def restore(self, snapshot, width):
        # Tracing executes the Python body, so the first call to a variant
        # advances the host mirror inside ``update_and_fetch`` as well as
        # here. Assign from the pre-call snapshot instead of incrementing,
        # so a traced step and a replayed step advance identically.
        self.cache._host_offset = snapshot + width

    def restore_failure(self, arrays, snapshot):
        self.install(arrays)
        self.cache._host_offset = snapshot

    def reserve(self, width):
        return self.cache.reserve(width)


class _ArraysSlot:
    """State plan for one ``ArraysCache`` (GDN conv + recurrent state).

    Its ``cache`` list is already a fixed-length list of fixed-shape arrays;
    the only reason it needs a plan at all is that the entries start as
    ``None`` before the first forward, so a compiled variant may only be
    built once every slot holds an array.
    """

    def __init__(self, cache):
        self.cache = cache
        self.n_arrays = len(cache.cache)

    def collect(self):
        return list(self.cache.cache)

    def install(self, arrays):
        self.cache.cache = list(arrays)

    def signature(self):
        # Fixed for the life of the cache once a forward has filled it.
        return None

    def host_state(self):
        return None

    def restore(self, snapshot, width):
        # ``ArraysCache.advance`` only moves the per-row ``lengths`` /
        # ``left_padding`` vectors, which ``model_is_compilable`` has already
        # rejected -- there is no scalar position to carry here.
        assert self.cache.lengths is None and self.cache.left_padding is None

    def restore_failure(self, arrays, snapshot):
        del snapshot
        self.install(arrays)

    def reserve(self, width):
        return False


def _plan(cache) -> List[Any]:
    plan = []
    for c in cache:
        if type(c) is RingKVCache:
            plan.append(_RingSlot(c))
        elif type(c) is ArraysCache:
            plan.append(_ArraysSlot(c))
        else:
            raise TypeError(f"compiled decode cannot thread a {type(c).__name__}")
    return plan


# --------------------------------------------------------------------------
# the compiled step
# --------------------------------------------------------------------------


class CompiledDecodeStep:
    """One compiled variant per (width, capacity signature).

    ``step(x)`` takes ``[B, width]`` input tokens and returns the logits,
    replaying a traced graph instead of rebuilding one. The caches are
    advanced exactly as the eager path advances them.

    ``trace_counts`` is cumulative, including evicted variants.  A submitted
    call is not a successful replay receipt until its exact returned logits
    object is passed to :meth:`materialize_and_confirm`.  This distinction
    matters because MLX reports device failures when a lazy result is
    evaluated, not necessarily when this wrapper returns it.
    """

    def __init__(
        self,
        model,
        cache,
        *,
        max_variants: Optional[int] = None,
        context_policy: Optional[CompiledDecodePolicy] = None,
    ):
        self.max_variants = (
            _max_variants()
            if max_variants is None
            else _validate_max_variants(max_variants)
        )
        why = model_is_compilable(model, cache)
        if why is not None:
            raise TypeError(f"compiled decode is not available: {why}")
        self.model = model
        self.cache = cache
        self.plan = _plan(cache)
        self._ring_slots = [s for s in self.plan if isinstance(s, _RingSlot)]
        if not self._ring_slots:
            raise TypeError(
                "compiled decode requires at least one full-attention KV cache"
            )
        if context_policy is None:
            context_tokens = max(s.cache.size() for s in self._ring_slots)
            why, context_policy = compiled_decode_context_policy(context_tokens, 0)
            if why is not None:
                raise ValueError(f"compiled decode is not available: {why}")
        self.context_policy = _validate_context_policy(context_policy)
        for slot in self._ring_slots:
            if tuple(slot.cache.buckets) != self.context_policy.buckets:
                raise TypeError(
                    "compiled decode RingKVCache buckets do not match the "
                    f"{self.context_policy.name} context policy"
                )
        self._capacities = ()
        self._variants = {}
        self.trace_counts = {}
        self.submission_counts = {}
        self.replay_counts = {}
        self.failure_counts = {}
        self._pending_receipts = []
        self._poisoned = False
        self._poison_reason = None

    # -- keying ---------------------------------------------------------

    def _signature(self, x):
        """Variant key. Cheap on purpose -- it runs on every replay, and a
        key that walked all 48 caches' array shapes would give back part of
        what the replay saves. Only the input shape/dtype and the KV
        capacities can change after the first forward."""
        return (x.shape, x.dtype) + self._capacities

    def _guard(self):
        if self._poisoned:
            raise CompiledDecodePoisoned(
                "compiled decode step is poisoned and must be discarded: "
                f"{self._poison_reason}"
            )
        for c in self.cache:
            if getattr(c, "speculating", False):
                raise RuntimeError(
                    "compiled decode cannot run while a cache is speculating: "
                    "ArraysCache.record_rollback stashes a Python closure over "
                    "the step's intermediates, which tracing would capture"
                )

    # -- build ----------------------------------------------------------

    def _build(self, key):
        plan = self.plan
        model = self.model
        counts = self.trace_counts
        counts.setdefault(key, 0)

        splits = []
        start = 0
        for slot in plan:
            splits.append((start, start + slot.n_arrays))
            start += slot.n_arrays

        def fn(x, *state):
            # Runs once per variant: MLX executes the Python body only while
            # tracing. Everything below therefore has to be graph ops --
            # no .item(), no mx.eval, no branching on a traced value.
            counts[key] += 1
            for slot, (lo, hi) in zip(plan, splits):
                slot.install(state[lo:hi])
            # This Python body runs only while mx.compile traces a variant.
            # Replays call the compiled graph directly and never enter the
            # precision-routing context.
            with precise_span():
                logits = model(x, cache=self.cache)
            out = []
            for slot in plan:
                out.extend(slot.collect())
            return (logits, *out)

        return mx.compile(fn), splits

    def _evict_all(self):
        """Drop live graphs while preserving cumulative audit receipts."""
        self._variants.clear()

    def _restore_failed_call(self, state, host, splits):
        for slot, (lo, hi), snapshot in zip(self.plan, splits, host):
            slot.restore_failure(state[lo:hi], snapshot)

    def poison(self, error, *, phase="device execution"):
        """Permanently disable this decoder after an uncommitted failure.

        Deferred device failures cannot be retried safely because Python cannot
        know which kernels completed.  Pending calls become failed receipts and
        every future call raises :class:`CompiledDecodePoisoned`.
        """
        if not self._poisoned:
            self._poisoned = True
            self._poison_reason = f"{phase}: {error!r}"
            for key, _ in self._pending_receipts:
                self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
            self._pending_receipts.clear()
            self._evict_all()
        return CompiledDecodePoisoned(
            "compiled decode failed; discard this step and its cache without "
            f"retrying the token ({self._poison_reason})"
        )

    def materialize_and_confirm(
        self, output, *dependent_values, phase="materialization"
    ):
        """Materialize one exact submitted output and issue its success receipt."""
        self._guard()
        if not self._pending_receipts or output is not self._pending_receipts[0][1]:
            raise RuntimeError(
                "completion output does not match the oldest compiled submission"
            )
        try:
            mx.eval(output, *dependent_values)
        except Exception as error:
            raise self.poison(error, phase=phase) from error
        key, _ = self._pending_receipts.pop(0)
        self.replay_counts[key] = self.replay_counts.get(key, 0) + 1
        return 1

    def drain_pending(self, *, phase="stream finalization"):
        """Resolve only submitted outputs, without running another model step."""
        if self._poisoned:
            return 0  # poison() already classified pending calls as failed.
        count = 0
        while self._pending_receipts:
            output = self._pending_receipts[0][1]
            count += self.materialize_and_confirm(
                output, [c.state for c in self.cache], phase=phase
            )
        return count

    # -- call -----------------------------------------------------------

    def __call__(self, x: mx.array) -> mx.array:
        self._guard()
        if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] != 1:
            raise ValueError(
                "compiled decode is qualified only for batch 1, width 1"
            )
        width = x.shape[1]
        position = max(s.cache.size() for s in self._ring_slots)
        if position + width > self.context_policy.max_context:
            raise RuntimeError(
                f"compiled decode reached the {self.context_policy.name} "
                f"profile limit of {self.context_policy.max_context} tokens"
            )
        # Grow every slab *before* keying the variant: the mask is built
        # from ``capacity`` at the top of the forward, so a growth inside
        # the traced step would leave the mask narrower than the keys.
        grew = False
        for slot in self._ring_slots:
            grew = slot.reserve(width) or grew
        capacities = tuple(s.signature() for s in self._ring_slots)
        if grew or capacities != self._capacities:
            self._capacities = capacities

        key = self._signature(x)
        entry = self._variants.get(key)
        is_new_variant = entry is None
        if entry is None:
            if len(self._variants) >= self.max_variants:
                # Bounded: every live variant pins a compiled graph. Drop the
                # whole table rather than guess which one is cold.
                self._evict_all()
            entry = self._build(key)
            self._variants[key] = entry
        compiled, splits = entry

        state = []
        host = []
        for slot in self.plan:
            state.extend(slot.collect())
            host.append(slot.host_state())
        try:
            out = compiled(x, *state)
            # The first call is both trace and first launch.  Materialize it
            # before committing state so a trace/first-launch failure can be
            # rolled back exactly.  Replays remain asynchronous.
            if is_new_variant:
                mx.eval(out)
        except Exception as error:
            self._restore_failed_call(state, host, splits)
            self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
            phase = "trace/first launch" if is_new_variant else "replay submission"
            raise self.poison(error, phase=phase) from error
        logits, new_state = out[0], out[1:]
        for slot, (lo, hi), snap in zip(self.plan, splits, host):
            slot.install(new_state[lo:hi])
            slot.restore(snap, width)
        self.submission_counts[key] = self.submission_counts.get(key, 0) + 1
        self._pending_receipts.append((key, logits))
        return logits

    # -- mechanism proof ------------------------------------------------

    def assert_single_trace(self):
        """Every variant traced exactly once. Raises with the offender."""
        if self._poisoned:
            raise AssertionError(f"compiled decode is poisoned: {self._poison_reason}")
        if not self.trace_counts:
            raise AssertionError("compiled decode did not trace any variant")
        if self._pending_receipts:
            raise AssertionError(
                f"compiled decode has {len(self._pending_receipts)} "
                "unconfirmed calls"
            )
        bad = {k: v for k, v in self.trace_counts.items() if v != 1}
        if bad:
            raise AssertionError(
                f"compiled decode retraced: {len(bad)} of "
                f"{len(self.trace_counts)} variants have trace count != 1"
            )
        return True

    def receipt(self):
        """Return cumulative, completion-backed replay evidence."""
        return {
            "model_type": _model_type(self.model),
            "qualification": _QUALIFICATION_TOKEN,
            "context_policy": self.context_policy.name,
            "max_context": self.context_policy.max_context,
            "buckets": self.context_policy.buckets,
            "numerical_acceptance": self.context_policy.numerical_acceptance,
            "live_variants": len(self._variants),
            "trace_counts": {repr(k): v for k, v in self.trace_counts.items()},
            "submission_counts": {
                repr(k): v for k, v in self.submission_counts.items()
            },
            "completed_counts": {repr(k): v for k, v in self.replay_counts.items()},
            "failure_counts": {repr(k): v for k, v in self.failure_counts.items()},
            "pending": len(self._pending_receipts),
            "poisoned": self._poisoned,
            "poison_reason": self._poison_reason,
        }

    @property
    def n_variants(self):
        return len(self._variants)


def compiled_decode_step(model, cache, width=None, capacity=None):
    """Build a ``CompiledDecodeStep`` for ``model``/``cache``.

    ``capacity`` optionally pre-reserves the KV slabs so the first step does
    not immediately grow (and retrace) them. There is deliberately no dummy
    warm-up step: a warm-up would advance the GDN recurrence, and unlike the
    KV slabs that state cannot be rewound. The trace is paid on the first
    real step, once per (width, capacity) variant.
    """
    if width not in (None, 1):
        raise ValueError("compiled decode is qualified only for width 1")
    step = CompiledDecodeStep(model, cache)
    if capacity is not None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
        ):
            raise ValueError("capacity must be a positive integer")
        if capacity > step.context_policy.max_context:
            raise ValueError(
                f"capacity {capacity} exceeds the "
                f"{step.context_policy.name} profile limit of "
                f"{step.context_policy.max_context}"
            )
        for c in cache:
            if isinstance(c, RingKVCache):
                c.reserve(max(0, capacity - c.size()))
    return step
