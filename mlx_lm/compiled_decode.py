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
  ``index_keys`` grow by concatenation, its block-summary ledger is
  appended to per block, and its MTP cycle state branches on Python
  ints. ``model_is_compilable`` rejects it; use ``compiled_segments``
  to compile the linear (GDN) runs and leave the QSA layers eager.
* Speculation. ``ArraysCache.record_rollback`` stashes a Python closure
  over the step's own intermediates; under tracing that closure would
  capture tracer arrays. ``CompiledDecodeStep`` refuses to run while any
  cache has ``speculating`` set, so the self-MTP verify width keeps the
  eager path until the rollback record is expressible as array state.
* Batched decode with per-row ``lengths``/``left_padding``: those are
  host vectors read with ``tolist()``.
"""

import os
from typing import Any, List, Optional, Sequence

import mlx.core as mx

from .models.cache import ArraysCache, KVCache, RingKVCache

__all__ = [
    "CompiledDecodeStep",
    "compiled_decode_step",
    "compiled_decode_enabled",
    "model_is_compilable",
    "to_shape_stable_cache",
]


def compiled_decode_enabled() -> bool:
    """Default-off opt-in via ``MLX_LM_COMPILED_DECODE=1``."""
    return os.environ.get("MLX_LM_COMPILED_DECODE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _max_variants() -> int:
    return int(os.environ.get("MLX_LM_COMPILED_DECODE_MAX_VARIANTS", "16"))


# --------------------------------------------------------------------------
# cache conversion
# --------------------------------------------------------------------------


def to_shape_stable_cache(cache: List[Any], buckets=None) -> List[Any]:
    """Return ``cache`` with every ``KVCache`` swapped for a ``RingKVCache``.

    In place for the list: the caller's list object is mutated so any other
    holder of it (a server session, a prompt-cache entry) sees the swap.
    ``ArraysCache`` entries pass through -- they are already fixed-shape.
    Raises for any other cache type, because silently leaving a growing
    cache in the list would make the compiled step wrong rather than slow.
    """
    for i, c in enumerate(cache):
        if isinstance(c, RingKVCache):
            continue
        if type(c) is KVCache:
            cache[i] = RingKVCache.from_kv_cache(c, buckets=buckets)
        elif isinstance(c, ArraysCache):
            continue
        else:
            raise TypeError(
                f"{type(c).__name__} at cache[{i}] is not shape-stable; "
                "compiled decode supports KVCache/RingKVCache and ArraysCache"
            )
    return cache


def model_is_compilable(cache: Sequence[Any]) -> Optional[str]:
    """``None`` if this cache list can drive a compiled step, else why not."""
    for i, c in enumerate(cache):
        if isinstance(c, RingKVCache) or type(c) is KVCache:
            continue
        if isinstance(c, ArraysCache):
            if getattr(c, "lengths", None) is not None:
                return f"cache[{i}] carries per-row lengths (batched decode)"
            if getattr(c, "left_padding", None) is not None:
                return f"cache[{i}] carries left padding (batched decode)"
            continue
        return f"cache[{i}] is a {type(c).__name__}, which is not shape-stable"
    return None


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
        c = self.cache
        return ("ring", c.keys.shape, str(c.keys.dtype), c.values.shape, c.capacity)

    def host_state(self):
        return self.cache._host_offset

    def restore(self, snapshot, width):
        # Tracing executes the Python body, so the first call to a variant
        # advances the host mirror inside ``update_and_fetch`` as well as
        # here. Assign from the pre-call snapshot instead of incrementing,
        # so a traced step and a replayed step advance identically.
        self.cache._host_offset = snapshot + width

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
        return ("arrays",) + tuple(
            (a.shape, str(a.dtype)) for a in self.cache.cache
        )

    def host_state(self):
        return None

    def restore(self, snapshot, width):
        # ``ArraysCache.advance`` only moves the per-row ``lengths`` /
        # ``left_padding`` vectors, which ``model_is_compilable`` has already
        # rejected -- there is no scalar position to carry here.
        assert self.cache.lengths is None and self.cache.left_padding is None

    def reserve(self, width):
        return False


def _plan(cache) -> List[Any]:
    plan = []
    for c in cache:
        if isinstance(c, RingKVCache):
            plan.append(_RingSlot(c))
        elif isinstance(c, ArraysCache):
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

    ``trace_counts`` maps a variant key to how many times the Python body
    was executed for it. It must be exactly 1 per variant no matter how
    many steps ran: a second entry means MLX retraced, which would give
    back all of the saving and then some. Tests assert on it, and so does
    ``assert_single_trace``.
    """

    def __init__(self, model, cache, *, max_variants: Optional[int] = None):
        why = model_is_compilable(cache)
        if why is not None:
            raise TypeError(f"compiled decode is not available: {why}")
        self.model = model
        self.cache = cache
        self.plan = _plan(cache)
        self._variants = {}
        self.trace_counts = {}
        self.replay_counts = {}
        self.max_variants = _max_variants() if max_variants is None else max_variants

    # -- keying ---------------------------------------------------------

    def _signature(self, x):
        return (x.shape, str(x.dtype)) + tuple(s.signature() for s in self.plan)

    def _guard(self):
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
            logits = model(x, cache=self.cache)
            out = []
            for slot in plan:
                out.extend(slot.collect())
            return (logits, *out)

        return mx.compile(fn), splits

    # -- call -----------------------------------------------------------

    def __call__(self, x: mx.array) -> mx.array:
        self._guard()
        width = x.shape[1]
        # Grow every slab *before* keying the variant: the mask is built
        # from ``capacity`` at the top of the forward, so a growth inside
        # the traced step would leave the mask narrower than the keys.
        for slot in self.plan:
            slot.reserve(width)

        key = self._signature(x)
        entry = self._variants.get(key)
        if entry is None:
            if len(self._variants) >= self.max_variants:
                # Bounded: every live variant pins a compiled graph. Drop the
                # whole table rather than guess which one is cold.
                self._variants.clear()
            entry = self._build(key)
            self._variants[key] = entry
        compiled, splits = entry

        state = []
        host = []
        for slot in self.plan:
            state.extend(slot.collect())
            host.append(slot.host_state())
        out = compiled(x, *state)
        logits, new_state = out[0], out[1:]
        for slot, (lo, hi), snap in zip(self.plan, splits, host):
            slot.install(new_state[lo:hi])
            slot.restore(snap, width)
        self.replay_counts[key] = self.replay_counts.get(key, 0) + 1
        return logits

    # -- mechanism proof ------------------------------------------------

    def assert_single_trace(self):
        """Every variant traced exactly once. Raises with the offender."""
        bad = {k: v for k, v in self.trace_counts.items() if v != 1}
        if bad:
            raise AssertionError(
                f"compiled decode retraced: {len(bad)} of "
                f"{len(self.trace_counts)} variants have trace count != 1"
            )
        return True

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
    step = CompiledDecodeStep(model, cache)
    del width
    if capacity is not None:
        for c in cache:
            if isinstance(c, RingKVCache):
                c.reserve(max(0, capacity - c.size()))
    return step
