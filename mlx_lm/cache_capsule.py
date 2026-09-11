"""Experimental, generation-safe materialization of one APC KV-cache plane.

This module deliberately does not alter APC lookup or model semantics.  It is
the product-shaped boundary needed to test whether a second execution engine
can prepare a physical B>1 cache plane before its first authoritative
consumer.  The feature is default-off and currently supports only a plain
``KVCache`` plane; recurrent, rotating, quantized and QSA-specific state must
earn their own exact contracts before integration.

An externally supplied e5rt adapter implements ``stage(source)`` and
``adopt(built, source)`` on the caller thread, plus ``build(staged)`` on the
worker. Nothing here imports or requires the private e5rt/ANE runtime.
"""

from __future__ import annotations

import os
import hashlib
import threading
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from .models.cache import BatchKVCache, KVCache


class CacheCapsuleError(RuntimeError):
    """Base class for fail-closed cache-capsule errors."""


class CacheCapsuleDisabled(CacheCapsuleError):
    pass


class CacheCapsuleUnsupported(CacheCapsuleError):
    pass


class StaleCacheCapsule(CacheCapsuleError):
    pass


class CacheCapsuleDeadline(CacheCapsuleError):
    pass


class CacheCapsuleOwnerReleased(CacheCapsuleError):
    pass


def cache_capsules_enabled() -> bool:
    """Return the process gate.  Importing the module never enables it."""

    return os.environ.get("MLX_LM_CACHE_CAPSULE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class CacheCapsuleCapability:
    supported: bool
    reason: Optional[str]
    source_batch: Optional[int] = None
    target_batch: Optional[int] = None
    dtype: Optional[str] = None


@dataclass(frozen=True)
class KVCachePlaneSource:
    """Immutable descriptor for one generation-stamped APC cache plane.

    MLX arrays are captured through copy-on-write descriptors. ``generation``
    makes late work disposable. An asynchronous adapter must not read these
    MLX arrays on its worker thread: request-thread ingress into adapter-owned
    input buffers is part of the adapter boundary.
    """

    generation: int
    source_id: str
    keys: Any
    values: Any
    offset: int
    target_batch: int
    layout_fingerprint: Tuple[Any, ...]
    creator_thread: int
    raw_digest: Optional[str]


@dataclass(frozen=True)
class KVCacheCapsulePayload:
    keys: Any
    values: Any
    offset: int
    source_generation: int
    source_id: str
    layout_fingerprint: Tuple[Any, ...]
    backend: str
    raw_digest: Optional[str]


@dataclass(frozen=True)
class CacheCapsuleProduct:
    """Constructor output plus anything that owns its backing buffers."""

    payload: KVCacheCapsulePayload
    backing_owner: Any = None


@dataclass(frozen=True)
class CacheCapsuleReceipt:
    owner: "CacheCapsuleOwner"
    backend: str
    fallback_reason: Optional[str]


class PreparedPromptCacheCapsules:
    """Own a mixed capsule/ordinary batched prompt cache until stream drain."""

    def __init__(
        self,
        prompt_cache: List[Any],
        receipts: Sequence[CacheCapsuleReceipt],
        leases: Sequence["CacheCapsuleLease"],
        *,
        requested_backend: str,
        ordinary_planes: int,
    ):
        self.prompt_cache = prompt_cache
        self.receipts = tuple(receipts)
        self.leases = tuple(leases)
        self.requested_backend = str(requested_backend)
        self.actual_backends = tuple(receipt.backend for receipt in receipts)
        self.fallback_reasons = tuple(
            receipt.fallback_reason for receipt in receipts
        )
        self.ordinary_planes = int(ordinary_planes)
        self._closed = False

    @property
    def backend(self) -> str:
        """Actual backend label, or ``mixed`` when planes took different paths."""

        backends = set(self.actual_backends)
        return next(iter(backends)) if len(backends) == 1 else "mixed"

    @property
    def capsule_planes(self) -> int:
        return len(self.receipts)

    def close(self, *, synchronize: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        sync_error = None
        try:
            if synchronize:
                mx.synchronize()
        except BaseException as error:
            sync_error = error
        finally:
            for lease in reversed(self.leases):
                lease.close()
            for receipt in reversed(self.receipts):
                receipt.owner.release()
        if sync_error is not None:
            raise sync_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class CacheCapsuleGeneration:
    """Small generation authority shared by APC invalidation and workers."""

    def __init__(self, initial: int = 0):
        self._value = int(initial)
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        with self._lock:
            return self._value

    def advance(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


def inspect_kv_cache_capsule(
    cache: Any, target_batch: int = 2
) -> CacheCapsuleCapability:
    """Fail-closed capability probe for the first deliberately narrow cut."""

    if type(cache) is not KVCache:
        return CacheCapsuleCapability(False, "plain_kv_only")
    if cache.keys is None or cache.values is None:
        return CacheCapsuleCapability(False, "empty_cache")
    if len(cache.keys.shape) != 4 or len(cache.values.shape) != 4:
        return CacheCapsuleCapability(False, "expected_rank_4")
    if cache.keys.shape[0] != 1 or cache.values.shape[0] != 1:
        return CacheCapsuleCapability(False, "source_batch_must_be_one")
    if cache.keys.shape[:3] != cache.values.shape[:3]:
        return CacheCapsuleCapability(False, "key_value_geometry_mismatch")
    if int(target_batch) < 2:
        return CacheCapsuleCapability(False, "target_batch_must_exceed_one")
    if cache.offset < 0 or cache.offset > cache.keys.shape[2]:
        return CacheCapsuleCapability(False, "invalid_offset")
    if cache.keys.dtype != cache.values.dtype:
        return CacheCapsuleCapability(False, "key_value_dtype_mismatch")
    if cache.keys.dtype not in (mx.bfloat16, mx.float16):
        return CacheCapsuleCapability(False, "bf16_or_fp16_only")
    return CacheCapsuleCapability(
        True,
        None,
        source_batch=1,
        target_batch=int(target_batch),
        dtype=str(cache.keys.dtype),
    )


def prepare_prompt_cache_capsules(
    prompt_cache: Sequence[Any],
    *,
    target_batch: int,
    generation: int,
    pool: "CacheCapsulePool",
    backend: str = "gpu",
    fallback: Optional[str] = "gpu",
    timeout_s: Optional[float] = None,
    source_prefix: str = "apc",
    synchronize: Optional[Callable[[KVCacheCapsulePayload], None]] = None,
) -> Optional[PreparedPromptCacheCapsules]:
    """Build the first mixed batched cache directly from an APC-restored cache.

    Plain BF16/FP16 ``KVCache`` planes use the capsule ownership path. Every
    other cache class retains its existing exact ``merge`` implementation.
    Returning ``None`` means no plane was eligible and lets the caller use its
    incumbent whole-cache merge without changing behavior.
    """

    if int(target_batch) < 2:
        raise ValueError("cache-capsule batch must contain at least two rows")
    if synchronize is None:
        synchronize = lambda payload: mx.eval(payload.keys, payload.values)
    capabilities = [
        inspect_kv_cache_capsule(cache, target_batch) for cache in prompt_cache
    ]
    if not any(capability.supported for capability in capabilities):
        return None
    batched = []
    receipts = []
    leases = []
    ordinary = 0
    try:
        for index, (cache, capability) in enumerate(
            zip(prompt_cache, capabilities)
        ):
            if capability.supported:
                source = capture_kv_cache_plane(
                    cache,
                    generation=int(generation),
                    source_id=f"{source_prefix}:plane:{index}",
                    target_batch=int(target_batch),
                    verify_raw_bits=pool.verify_raw_bits,
                )
                receipt = pool.prepare(
                    source,
                    primary=str(backend),
                    fallback=fallback,
                    primary_timeout_s=timeout_s,
                )
                receipts.append(receipt)
                lease = receipt.owner.lease()
                leases.append(lease)
                restored = lease.restore_batch_kv_cache(synchronize)
                batched.append(restored)
                continue
            merge = getattr(cache, "merge", None)
            if not callable(merge):
                raise CacheCapsuleUnsupported(
                    f"unsupported_nonmergeable_plane:{type(cache).__name__}"
                )
            batched.append(merge([cache] * int(target_batch)))
            ordinary += 1
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        for receipt in reversed(receipts):
            receipt.owner.release()
        raise
    return PreparedPromptCacheCapsules(
        batched,
        receipts,
        leases,
        requested_backend=str(backend),
        ordinary_planes=ordinary,
    )


def capture_kv_cache_plane(
    cache: KVCache,
    *,
    generation: int,
    source_id: str,
    target_batch: int = 2,
    verify_raw_bits: bool = False,
) -> KVCachePlaneSource:
    """Capture a real KVCache plane without changing or cloning APC state."""

    capability = inspect_kv_cache_capsule(cache, target_batch)
    if not capability.supported:
        raise CacheCapsuleUnsupported(capability.reason or "unsupported")
    fingerprint = (
        tuple(cache.keys.shape),
        tuple(cache.values.shape),
        str(cache.keys.dtype),
        int(cache.offset),
        int(target_batch),
    )
    # MLX assignments are copy-on-write. Holding stop-gradient descriptors
    # here pins the captured values even if the live request later rebinds or
    # updates its cache arrays, without paying a physical clone at capture.
    frozen_keys = mx.stop_gradient(cache.keys)
    frozen_values = mx.stop_gradient(cache.values)
    return KVCachePlaneSource(
        generation=int(generation),
        source_id=str(source_id),
        keys=frozen_keys,
        values=frozen_values,
        offset=int(cache.offset),
        target_batch=int(target_batch),
        layout_fingerprint=fingerprint,
        creator_thread=threading.get_ident(),
        # Full hashing materializes MLX arrays on the host.  Keep it as an
        # explicit qualification/debug gate, not an unconditional hot-path
        # synchronization that would erase the overlap being measured.
        raw_digest=(
            _raw_digest(frozen_keys, frozen_values) if verify_raw_bits else None
        ),
    )


def _raw_array(value):
    if isinstance(value, mx.array):
        raw = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
        return np.asarray(raw)
    return np.asarray(value)


def _raw_digest(keys, values, repeats=1):
    digest = hashlib.sha256()
    for value in (keys, values):
        raw = _raw_array(value)
        if repeats != 1:
            raw = np.repeat(raw, repeats, axis=0)
        digest.update(str(raw.dtype).encode())
        digest.update(str(tuple(raw.shape)).encode())
        digest.update(raw.tobytes(order="C"))
    return digest.hexdigest()


def _repeat_numpy_raw(value: Any, repeats: int):
    """Repeat batch rows on CPU without numerically converting BF16 bits."""

    if isinstance(value, mx.array):
        dtype = value.dtype
        if dtype == mx.bfloat16:
            raw = np.asarray(value.view(mx.uint16))
            output = mx.array(np.repeat(raw, repeats, axis=0)).view(mx.bfloat16)
        else:
            output = mx.array(np.repeat(np.asarray(value), repeats, axis=0))
            if output.dtype != dtype:
                output = output.astype(dtype)
        return output
    return np.repeat(np.asarray(value), repeats, axis=0)


def build_kv_cache_capsule_cpu(source: KVCachePlaneSource) -> CacheCapsuleProduct:
    keys = _repeat_numpy_raw(source.keys, source.target_batch)
    values = _repeat_numpy_raw(source.values, source.target_batch)
    return CacheCapsuleProduct(
        KVCacheCapsulePayload(
            keys,
            values,
            source.offset,
            source.generation,
            source.source_id,
            source.layout_fingerprint,
            "cpu",
            _raw_digest(keys, values) if source.raw_digest is not None else None,
        )
    )


def build_kv_cache_capsule_gpu(source: KVCachePlaneSource) -> CacheCapsuleProduct:
    if not isinstance(source.keys, mx.array) or not isinstance(source.values, mx.array):
        raise CacheCapsuleUnsupported("gpu_backend_requires_mlx_arrays")
    keys = mx.concatenate([source.keys] * source.target_batch, axis=0)
    values = mx.concatenate([source.values] * source.target_batch, axis=0)
    return CacheCapsuleProduct(
        KVCacheCapsulePayload(
            keys,
            values,
            source.offset,
            source.generation,
            source.source_id,
            source.layout_fingerprint,
            "gpu",
            (
                _raw_digest(source.keys, source.values, source.target_batch)
                if source.raw_digest is not None
                else None
            ),
        )
    )


class CacheCapsuleOwner:
    """Own a completed capsule and defer backing release through all leases."""

    def __init__(
        self,
        product: CacheCapsuleProduct,
        generation: CacheCapsuleGeneration,
        expected_generation: int,
        creator_thread: int,
    ):
        self._product = product
        self._generation = generation
        self._expected_generation = expected_generation
        self._creator_thread = creator_thread
        self._leases = 0
        self._release_requested = False
        self._lock = threading.Lock()

    def lease(self) -> "CacheCapsuleLease":
        stale_product = None
        stale = False
        with self._lock:
            if threading.get_ident() != self._creator_thread:
                raise CacheCapsuleError(
                    "capsule lease must be created on source thread"
                )
            if self._generation.current != self._expected_generation:
                stale = True
                self._release_requested = True
                if self._leases == 0:
                    stale_product = self._product
                    self._product = None
            elif self._product is None or self._release_requested:
                raise CacheCapsuleOwnerReleased("cache capsule owner was released")
            else:
                self._leases += 1
        if stale:
            _release_product(stale_product)
            raise StaleCacheCapsule("capsule became stale before lease")
        return CacheCapsuleLease(self)

    def release(self):
        """Drop ownership now, or after the final consumer lease closes."""

        product = None
        with self._lock:
            self._release_requested = True
            if self._leases == 0:
                product = self._product
                self._product = None
        _release_product(product)

    @property
    def released(self) -> bool:
        with self._lock:
            return self._product is None

    def _payload(self) -> KVCacheCapsulePayload:
        stale_product = None
        with self._lock:
            if threading.get_ident() != self._creator_thread:
                raise CacheCapsuleError("capsule must be consumed on source thread")
            if self._generation.current != self._expected_generation:
                self._release_requested = True
                if self._leases == 0:
                    stale_product = self._product
                    self._product = None
            elif self._product is None:
                raise CacheCapsuleOwnerReleased("cache capsule backing was released")
            else:
                return self._product.payload
        _release_product(stale_product)
        raise StaleCacheCapsule("capsule became stale before consumption")

    def _close_lease(self):
        product = None
        with self._lock:
            if self._leases <= 0:
                raise RuntimeError("cache capsule lease underflow")
            self._leases -= 1
            if self._release_requested and self._leases == 0:
                product = self._product
                self._product = None
        _release_product(product)


class CacheCapsuleLease:
    """Pin backing memory and enforce synchronization at first consumption."""

    def __init__(self, owner: CacheCapsuleOwner):
        self._owner = owner
        self._closed = False
        self._synchronized = False
        self._lock = threading.Lock()

    def payload_for_consumer(
        self, synchronize: Callable[[KVCacheCapsulePayload], None]
    ) -> KVCacheCapsulePayload:
        with self._lock:
            if self._closed:
                raise CacheCapsuleOwnerReleased("cache capsule lease is closed")
            if not callable(synchronize):
                raise TypeError(
                    "the first consumer must provide a synchronization callback"
                )
            payload = self._owner._payload()
            if not self._synchronized:
                synchronize(payload)
                self._synchronized = True
            return payload

    def restore_kv_cache(
        self,
        synchronize: Callable[[KVCacheCapsulePayload], None],
    ) -> KVCache:
        """Build the first authoritative KVCache consumer after synchronization."""

        payload = self.payload_for_consumer(synchronize)
        return KVCache.from_state(
            (payload.keys, payload.values),
            (str(payload.offset),),
        )

    def restore_batch_kv_cache(
        self,
        synchronize: Callable[[KVCacheCapsulePayload], None],
    ) -> BatchKVCache:
        """Restore the capsule with the cache API required by batch serving."""

        payload = self.payload_for_consumer(synchronize)
        batch = int(payload.keys.shape[0])
        cache = BatchKVCache([0] * batch)
        cache.keys = payload.keys
        cache.values = payload.values
        cache.offset = mx.array([int(payload.offset)] * batch)
        cache.left_padding = mx.zeros((batch,), dtype=mx.int32)
        cache._idx = int(payload.offset)
        return cache

    def close(self):
        close_owner = False
        with self._lock:
            if not self._closed:
                self._closed = True
                close_owner = True
        if close_owner:
            self._owner._close_lease()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _release_product(product: Optional[CacheCapsuleProduct]):
    """Release an external backing owner exactly when our ownership ends."""

    if product is None or product.backing_owner is None:
        return
    release = getattr(product.backing_owner, "release", None)
    if callable(release):
        release()


class CacheCapsuleTicket:
    """A submitted external build that may overlap unrelated GPU work."""

    def __init__(self, pool, source, staged, future):
        self._pool = pool
        self.source = source
        self.staged = staged
        self.future = future
        self._lock = threading.Lock()
        self._state = "pending"
        self._receipt = None
        self._built_disposed = False
        self._stage_aborted = False

    def await_adopt(self, *, timeout_s=None, fallback="gpu"):
        return self._pool._await_ticket(self, timeout_s, fallback)

    def cancel(self, reason="cancelled"):
        with self._lock:
            if self._state != "pending":
                return False
            self._state = "cancelled"

        # A Future's callback may already have observed the old pending state
        # before this transition.  In that completed-then-cancel race there is
        # no later callback to reclaim the external result, so cancellation
        # must also attempt the idempotent disposal itself.
        completed = self.future.done()
        if completed:
            self._dispose_if_completed()
        if (
            not completed
            or self.future.cancelled()
            or self.future.exception() is not None
        ):
            self._abort_stage(reason)

        self._pool._ticket_terminal(self)

        return True

    def _dispose_if_completed(self):
        if (
            not self.future.done()
            or self.future.cancelled()
            or self.future.exception() is not None
            or not self._claim_built_disposal()
        ):
            return False
        self._pool._discard_e5rt(self.future.result())
        return True

    def _abort_stage(self, reason):
        with self._lock:
            if self._stage_aborted:
                return False
            self._stage_aborted = True
        threading.Thread(
            target=self._pool._abort_e5rt_stage,
            args=(self.staged, reason),
            name="cache-capsule-stage-abort",
            daemon=True,
        ).start()
        return True

    def _claim_built_disposal(self):
        with self._lock:
            if self._built_disposed:
                return False
            self._built_disposed = True
            return True

    def _repeat_result(self):
        with self._lock:
            if self._state in ("adopted", "fallback"):
                return self._receipt
            if self._state != "pending":
                raise CacheCapsuleError("capsule ticket is already terminal")
            return None

    def _begin_adoption(self):
        with self._lock:
            if self._state in ("adopted", "fallback"):
                return self._receipt
            if self._state != "pending":
                raise CacheCapsuleError("capsule ticket is already terminal")
            self._state = "adopting"
            return None

    def _finish(self, receipt, state="adopted"):
        with self._lock:
            self._receipt = receipt
            self._state = state
        self._pool._ticket_terminal(self)


class CacheCapsulePool:
    """One-worker preparation lane with deadline/error fallback.

    Only the external e5rt backend runs off-thread. MLX streams are bound to
    their creating thread, so the CPU and GPU constructors run on the caller
    thread; moving either to this worker would produce arrays the request
    thread cannot safely evaluate. A fallback also runs on the caller thread.
    Late e5rt results never become visible: the generation authority is
    checked both before construction and at handoff. An e5rt adapter must
    return externally owned views that are valid on the consumer thread.
    """

    _COUNTER_KEYS = (
        "requests",
        "primary_successes",
        "fallbacks",
        "timeouts",
        "errors",
        "stale",
    )
    _STAGING = object()

    def __init__(
        self,
        generation: CacheCapsuleGeneration,
        *,
        e5rt_adapter: Any = None,
        enabled: Optional[bool] = None,
        verify_raw_bits: bool = False,
    ):
        self.generation = generation
        self.e5rt_adapter = e5rt_adapter
        self.enabled = cache_capsules_enabled() if enabled is None else bool(enabled)
        self.verify_raw_bits = bool(verify_raw_bits)
        self._counters = {key: 0 for key in self._COUNTER_KEYS}
        self._counter_lock = threading.Lock()
        self._ticket_lock = threading.Lock()
        self._active_ticket = None
        # Completed products remain owned by their tickets until adopted or
        # cancelled.  Track them separately from the physical worker slot so a
        # later pool close can still reclaim a result whose callback has run.
        self._tickets = set()
        self._closed = False

    @property
    def counters(self) -> Dict[str, int]:
        with self._counter_lock:
            return dict(self._counters)

    def _count(self, key: str):
        with self._counter_lock:
            self._counters[key] += 1

    def _check_generation(self, source: KVCachePlaneSource):
        if source.generation != self.generation.current:
            self._count("stale")
            raise StaleCacheCapsule(
                f"source generation {source.generation} != current generation "
                f"{self.generation.current}"
            )

    def _build(self, backend: str, source: KVCachePlaneSource) -> CacheCapsuleProduct:
        self._check_generation(source)
        if backend == "cpu":
            return build_kv_cache_capsule_cpu(source)
        if backend == "gpu":
            return build_kv_cache_capsule_gpu(source)
        if backend == "e5rt":
            raise CacheCapsuleUnsupported("e5rt_requires_staged_worker_path")
        raise CacheCapsuleUnsupported(f"unknown_backend:{backend}")

    def _build_e5rt(self, source, staged):
        self._check_generation(source)
        return self.e5rt_adapter.build(staged)

    def _abort_e5rt_stage(self, staged, reason):
        """Relinquish an adapter's staged slot without assuming build ran."""

        abort = getattr(self.e5rt_adapter, "abort", None)
        if not callable(abort):
            abort = getattr(self.e5rt_adapter, "cancel", None)
        if callable(abort):
            abort(staged, reason)

    def submit(self, source: KVCachePlaneSource) -> CacheCapsuleTicket:
        """Stage and submit e5rt work, returning before it completes."""

        if not self.enabled:
            raise CacheCapsuleDisabled(
                "cache capsules are experimental; set MLX_LM_CACHE_CAPSULE=1"
            )
        if threading.get_ident() != source.creator_thread:
            raise CacheCapsuleError("capsule submission must use source thread")
        self._check_generation(source)
        if self.e5rt_adapter is None:
            raise CacheCapsuleUnsupported("e5rt_adapter_unavailable")
        stage = getattr(self.e5rt_adapter, "stage", None)
        if not callable(stage):
            raise CacheCapsuleUnsupported("e5rt_adapter_requires_stage")
        with self._ticket_lock:
            if self._closed:
                raise CacheCapsuleError("cache capsule pool is closed")
            if self._active_ticket is not None:
                raise CacheCapsuleUnsupported("e5rt_circuit_busy")
            self._active_ticket = self._STAGING
        try:
            staged = stage(source)
        except BaseException:
            with self._ticket_lock:
                if self._active_ticket is self._STAGING:
                    self._active_ticket = None
            raise
        with self._ticket_lock:
            if self._closed:
                if self._active_ticket is self._STAGING:
                    self._active_ticket = None
                threading.Thread(
                    target=self._abort_e5rt_stage,
                    args=(staged, "pool_closed_during_stage"),
                    name="cache-capsule-stage-abort",
                    daemon=True,
                ).start()
                raise CacheCapsuleError("cache capsule pool closed during stage")
            future = Future()
            ticket = CacheCapsuleTicket(self, source, staged, future)
            self._active_ticket = ticket
            self._tickets.add(ticket)

        def physical_done(completed):
            with ticket._lock:
                cancelled = ticket._state in ("cancelled", "fallback")
            if cancelled:
                ticket._dispose_if_completed()
            self._ticket_done(ticket)

        future.add_done_callback(physical_done)

        def run():
            try:
                future.set_result(self._build_e5rt(source, staged))
            except BaseException as error:
                if isinstance(error, StaleCacheCapsule):
                    ticket._abort_stage("stale_before_build")
                future.set_exception(error)

        threading.Thread(
            target=run, name="cache-capsule-e5rt", daemon=True
        ).start()
        self._count("requests")
        return ticket

    def _ticket_done(self, ticket):
        with self._ticket_lock:
            if self._active_ticket is ticket:
                self._active_ticket = None

    def _ticket_terminal(self, ticket):
        with self._ticket_lock:
            self._tickets.discard(ticket)

    def _await_ticket(self, ticket, timeout_s, fallback):
        if threading.get_ident() != ticket.source.creator_thread:
            ticket.cancel("wrong_adopt_thread")
            raise CacheCapsuleError("capsule adoption must use source thread")
        repeated = ticket._repeat_result()
        if repeated is not None:
            return repeated
        try:
            self._check_generation(ticket.source)
        except StaleCacheCapsule:
            ticket.cancel("stale_before_await")
            raise
        try:
            built = ticket.future.result(timeout=timeout_s)
        except TimeoutError as error:
            self._count("timeouts")
            ticket.cancel("deadline")
            if fallback is None:
                raise CacheCapsuleDeadline("e5rt_timeout") from error
            self._count("fallbacks")
            receipt = self._accept_product(
                self._build(fallback, ticket.source), ticket.source, fallback,
                "e5rt_timeout"
            )
            ticket._finish(receipt, "fallback")
            return receipt
        except Exception as error:
            self._count("errors")
            ticket.cancel("build_error")
            if fallback is None:
                raise CacheCapsuleError("e5rt_error") from error
            self._count("fallbacks")
            receipt = self._accept_product(
                self._build(fallback, ticket.source), ticket.source, fallback,
                f"e5rt_error:{type(error).__name__}"
            )
            ticket._finish(receipt, "fallback")
            return receipt
        repeated = ticket._begin_adoption()
        if repeated is not None:
            return repeated
        try:
            self._check_generation(ticket.source)
        except StaleCacheCapsule:
            if ticket._claim_built_disposal():
                self._discard_e5rt(built)
            ticket._finish(None, "cancelled")
            raise
        adopt = getattr(self.e5rt_adapter, "adopt", None)
        if not callable(adopt):
            if ticket._claim_built_disposal():
                self._discard_e5rt(built)
            ticket._finish(None, "cancelled")
            raise CacheCapsuleUnsupported("e5rt_adapter_requires_adopt")
        try:
            product = adopt(built, ticket.source)
        except Exception:
            if ticket._claim_built_disposal():
                self._discard_e5rt(built)
            ticket._finish(None, "cancelled")
            raise
        try:
            receipt = self._accept_product(product, ticket.source, "e5rt", None)
        except Exception:
            ticket._finish(None, "cancelled")
            raise
        ticket._finish(receipt, "adopted")
        self._count("primary_successes")
        return receipt

    def _discard_e5rt(self, value):
        discard = getattr(self.e5rt_adapter, "discard", None)
        if callable(discard):
            discard(value)
        elif isinstance(value, CacheCapsuleProduct):
            _release_product(value)
        else:
            release = getattr(value, "release", None)
            if callable(release):
                release()

    @staticmethod
    def _validate_product(
        product: CacheCapsuleProduct,
        source: KVCachePlaneSource,
        backend: str,
    ):
        payload = product.payload
        if payload.backend != backend:
            raise CacheCapsuleError(
                f"backend label mismatch: {payload.backend!r} != {backend!r}"
            )
        if payload.source_id != source.source_id or payload.offset != source.offset:
            raise CacheCapsuleError("backend changed cache identity or offset")
        for name, output, original in (
            ("keys", payload.keys, source.keys),
            ("values", payload.values, source.values),
        ):
            if not hasattr(output, "shape") or not hasattr(original, "shape"):
                raise CacheCapsuleError(f"{name} output has no shape")
            if tuple(output.shape[1:]) != tuple(original.shape[1:]):
                raise CacheCapsuleError(f"{name} output geometry mismatch")
            if int(output.shape[0]) != source.target_batch:
                raise CacheCapsuleError(f"{name} output batch mismatch")
            if getattr(output, "dtype", None) != getattr(original, "dtype", None):
                raise CacheCapsuleError(f"{name} output dtype mismatch")

    def prepare(
        self,
        source: KVCachePlaneSource,
        *,
        primary: str,
        fallback: Optional[str] = "gpu",
        primary_timeout_s: Optional[float] = None,
    ) -> CacheCapsuleReceipt:
        if primary == "e5rt":
            return self.submit(source).await_adopt(
                timeout_s=primary_timeout_s, fallback=fallback
            )
        if not self.enabled:
            raise CacheCapsuleDisabled("cache capsules are experimental")
        self._count("requests")
        product = self._build(primary, source)
        receipt = self._accept_product(product, source, primary, None)
        self._count("primary_successes")
        return receipt

    def _accept_product(self, product, source, backend, fallback_reason):
        if not isinstance(product, CacheCapsuleProduct):
            raise CacheCapsuleError("backend did not return CacheCapsuleProduct")
        try:
            self._check_generation(source)
        except StaleCacheCapsule:
            _release_product(product)
            raise
        try:
            self._validate_product(product, source, backend)
        except Exception:
            _release_product(product)
            raise
        if product.payload.source_generation != source.generation:
            _release_product(product)
            raise StaleCacheCapsule("backend returned a capsule for another generation")
        if product.payload.layout_fingerprint != source.layout_fingerprint:
            _release_product(product)
            raise CacheCapsuleError("backend changed the cache layout fingerprint")
        if self.verify_raw_bits:
            if source.raw_digest is None:
                _release_product(product)
                raise CacheCapsuleError(
                    "raw-bit verification requires a verified source capture"
                )
            if _raw_digest(source.keys, source.values) != source.raw_digest:
                _release_product(product)
                raise CacheCapsuleError("captured source failed raw-bit checksum")
            expected_digest = _raw_digest(
                source.keys, source.values, source.target_batch
            )
            actual_digest = _raw_digest(product.payload.keys, product.payload.values)
            if (
                product.payload.raw_digest != actual_digest
                or actual_digest != expected_digest
            ):
                _release_product(product)
                raise CacheCapsuleError("backend failed raw-bit checksum")
        return CacheCapsuleReceipt(
            CacheCapsuleOwner(
                product, self.generation, source.generation, source.creator_thread
            ),
            backend,
            fallback_reason,
        )

    def close(self):
        with self._ticket_lock:
            self._closed = True
            tickets = tuple(self._tickets)
        for ticket in tickets:
            ticket.cancel("pool_close")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


__all__ = [
    "CacheCapsuleCapability",
    "CacheCapsuleDeadline",
    "CacheCapsuleDisabled",
    "CacheCapsuleError",
    "CacheCapsuleGeneration",
    "CacheCapsuleLease",
    "CacheCapsuleOwner",
    "CacheCapsuleOwnerReleased",
    "CacheCapsulePool",
    "CacheCapsuleProduct",
    "CacheCapsuleReceipt",
    "CacheCapsuleTicket",
    "CacheCapsuleUnsupported",
    "KVCacheCapsulePayload",
    "KVCachePlaneSource",
    "PreparedPromptCacheCapsules",
    "StaleCacheCapsule",
    "build_kv_cache_capsule_cpu",
    "build_kv_cache_capsule_gpu",
    "cache_capsules_enabled",
    "capture_kv_cache_plane",
    "inspect_kv_cache_capsule",
    "prepare_prompt_cache_capsules",
]
