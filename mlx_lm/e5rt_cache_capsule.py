"""Pinned ANEForge adapter for the experimental cache-capsule pool.

This module is standalone and default inert. Importing it does not import
ANEForge, compile a program, or execute an accelerator. ``compile()`` builds one
fixed-shape two-output e5rt program. The adapter then implements the
``stage/build/adopt`` boundary consumed by :mod:`mlx_lm.cache_capsule`.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .cache_capsule import (
    CacheCapsuleError,
    CacheCapsuleProduct,
    CacheCapsuleUnsupported,
    KVCacheCapsulePayload,
    KVCachePlaneSource,
)


PINNED_ANEFORGE_REVISION = "026de27ea57b1fe42b608821d7b76ee9a9a66494"


@dataclass(frozen=True)
class ANEForgeRevisionReceipt:
    root: str
    revision: str
    tracked_clean: bool
    module_file: str


@dataclass(frozen=True)
class E5RTCacheCapsuleSpec:
    source_shape: Tuple[int, int, int, int]
    target_batch: int
    source_dtype: str

    def __post_init__(self):
        shape = tuple(int(value) for value in self.source_shape)
        if len(shape) != 4 or shape[0] != 1 or any(value <= 0 for value in shape):
            raise ValueError("e5rt cache source must have positive rank-4 B1 shape")
        if int(self.target_batch) < 2:
            raise ValueError("e5rt cache target batch must exceed one")
        if self.source_dtype not in ("float16", "bfloat16"):
            raise ValueError("e5rt cache transport supports float16 or bfloat16")
        object.__setattr__(self, "source_shape", shape)
        object.__setattr__(self, "target_batch", int(self.target_batch))

    @property
    def output_shape(self) -> Tuple[int, int, int, int]:
        return (self.target_batch, *self.source_shape[1:])

    @property
    def input_bytes(self) -> int:
        return int(np.prod(self.source_shape, dtype=np.int64)) * 2 * 2

    @property
    def output_bytes(self) -> int:
        return int(np.prod(self.output_shape, dtype=np.int64)) * 2 * 2

    @classmethod
    def from_source(cls, source: KVCachePlaneSource) -> "E5RTCacheCapsuleSpec":
        if tuple(source.keys.shape) != tuple(source.values.shape):
            raise CacheCapsuleUnsupported("e5rt_key_value_shape_mismatch")
        return cls(
            tuple(int(value) for value in source.keys.shape),
            int(source.target_batch),
            _dtype_name(source.keys),
        )


@dataclass(frozen=True)
class E5RTStagedCapsule:
    nonce: int
    source_generation: int
    source_id: str
    offset: int
    layout_fingerprint: Tuple[Any, ...]
    stage_ns: int
    input_bytes: int


@dataclass(frozen=True)
class E5RTBuiltCapsule:
    staged: E5RTStagedCapsule
    key_output: np.ndarray
    value_output: np.ndarray
    execute_ns: int


def inspect_aneforge_revision(
    module: Any,
    *,
    expected_revision: str = PINNED_ANEFORGE_REVISION,
) -> ANEForgeRevisionReceipt:
    """Verify that an imported ANEForge module comes from the pinned checkout."""

    module_file = Path(module.__file__).resolve()
    root = module_file.parent.parent
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CacheCapsuleUnsupported("aneforge_revision_unavailable") from error
    if revision != expected_revision:
        raise CacheCapsuleUnsupported(
            f"aneforge_revision_mismatch:{revision or 'unknown'}"
        )
    if status:
        raise CacheCapsuleUnsupported("aneforge_tracked_checkout_dirty")
    return ANEForgeRevisionReceipt(str(root), revision, True, str(module_file))


class _ANEForgeCapsuleProgram:
    def __init__(self, model, key_input, value_input, key_output, value_output):
        self._model = model
        self._program = model.prog
        self._input_names = {
            id(tensor): name for tensor, name in model.input_ports
        }
        self._output_names = {
            id(tensor): name for tensor, name in model.output_ports
        }
        self._key_input = key_input
        self._value_input = value_input
        self._key_output = key_output
        self._value_output = value_output

    def input_view(self, name: str) -> np.ndarray:
        tensor = self._key_input if name == "keys" else self._value_input
        return self._program.input_view(self._input_names[id(tensor)])

    def output_view(self, name: str) -> np.ndarray:
        tensor = self._key_output if name == "keys" else self._value_output
        return self._program.output_view(self._output_names[id(tensor)])

    def execute(self) -> None:
        self._program.execute()

    def release(self) -> None:
        self._model.release()


def compile_aneforge_capsule_program(
    spec: E5RTCacheCapsuleSpec,
    *,
    build_dir: Optional[Path] = None,
    aneforge_module: Any = None,
    expected_revision: str = PINNED_ANEFORGE_REVISION,
):
    """Compile a fixed-shape K/V B1-to-BN transport program without executing it."""

    aneforge = aneforge_module or importlib.import_module("aneforge")
    revision = inspect_aneforge_revision(
        aneforge, expected_revision=expected_revision
    )
    compiler = importlib.import_module("aneforge._compile")
    key_input = aneforge.input(spec.source_shape)
    value_input = aneforge.input(spec.source_shape)
    key_output = aneforge.concat([key_input] * spec.target_batch, axis=0)
    value_output = aneforge.concat([value_input] * spec.target_batch, axis=0)
    model = compiler.compile_multi(
        [key_output, value_output],
        build_dir=build_dir,
        int8=False,
    )
    return (
        _ANEForgeCapsuleProgram(
            model, key_input, value_input, key_output, value_output
        ),
        revision,
    )


class _CapsuleBacking:
    def __init__(self, adapter: "E5RTCacheCapsuleAdapter", nonce: int):
        self._adapter = adapter
        self._nonce = nonce
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._adapter._release_adopted(self._nonce)


class E5RTCacheCapsuleAdapter:
    """One-program e5rt adapter with explicit output-buffer lease ownership."""

    _COUNTER_KEYS = (
        "stages",
        "executes",
        "adoptions",
        "discards",
        "cancels",
        "busy_refusals",
        "errors",
        "backing_releases",
    )

    def __init__(
        self,
        program: Any,
        spec: E5RTCacheCapsuleSpec,
        *,
        revision_receipt: Optional[ANEForgeRevisionReceipt] = None,
        adopt_array: Optional[Callable[[np.ndarray, str], Any]] = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ):
        self.program = program
        self.spec = spec
        self.revision_receipt = revision_receipt
        self._adopt_array = adopt_array or _adopt_mlx_array
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._state = "idle"
        self._nonce = 0
        self._cancel_reason = None
        self._close_requested = False
        self._released = False
        self._counters = {key: 0 for key in self._COUNTER_KEYS}
        self._timing_ns = {"stage": 0, "execute": 0, "adopt": 0}
        self._key_input = program.input_view("keys")
        self._value_input = program.input_view("values")
        self._key_output = program.output_view("keys")
        self._value_output = program.output_view("values")
        for name, view, shape in (
            ("key input", self._key_input, spec.source_shape),
            ("value input", self._value_input, spec.source_shape),
            ("key output", self._key_output, spec.output_shape),
            ("value output", self._value_output, spec.output_shape),
        ):
            if tuple(view.shape) != tuple(shape) or view.dtype != np.float16:
                raise CacheCapsuleUnsupported(f"invalid_e5rt_{name.replace(' ', '_')}")

    @classmethod
    def compile(
        cls,
        spec: E5RTCacheCapsuleSpec,
        *,
        build_dir: Optional[Path] = None,
        expected_revision: str = PINNED_ANEFORGE_REVISION,
    ) -> "E5RTCacheCapsuleAdapter":
        program, receipt = compile_aneforge_capsule_program(
            spec,
            build_dir=build_dir,
            expected_revision=expected_revision,
        )
        try:
            return cls(program, spec, revision_receipt=receipt)
        except BaseException:
            program.release()
            raise

    @property
    def counters(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    @property
    def timing_ns(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._timing_ns)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _count(self, key: str, elapsed_ns: int = 0) -> None:
        self._counters[key] += 1
        timing_key = {
            "stages": "stage",
            "executes": "execute",
            "adoptions": "adopt",
        }.get(key)
        if timing_key is not None:
            self._timing_ns[timing_key] += int(elapsed_ns)

    def _validate_source(self, source: KVCachePlaneSource) -> None:
        if tuple(source.keys.shape) != self.spec.source_shape:
            raise CacheCapsuleUnsupported("e5rt_key_shape_mismatch")
        if tuple(source.values.shape) != self.spec.source_shape:
            raise CacheCapsuleUnsupported("e5rt_value_shape_mismatch")
        if int(source.target_batch) != self.spec.target_batch:
            raise CacheCapsuleUnsupported("e5rt_target_batch_mismatch")
        if _dtype_name(source.keys) != self.spec.source_dtype:
            raise CacheCapsuleUnsupported("e5rt_key_dtype_mismatch")
        if _dtype_name(source.values) != self.spec.source_dtype:
            raise CacheCapsuleUnsupported("e5rt_value_dtype_mismatch")

    def stage(self, source: KVCachePlaneSource) -> E5RTStagedCapsule:
        self._validate_source(source)
        with self._lock:
            if self._released or self._close_requested:
                raise CacheCapsuleError("e5rt capsule adapter is closed")
            if self._state != "idle":
                self._count("busy_refusals")
                raise CacheCapsuleUnsupported("e5rt_adapter_busy")
            self._state = "staging"
            self._nonce += 1
            nonce = self._nonce
            self._cancel_reason = None
        started = self._clock_ns()
        try:
            _copy_raw_bits(self._key_input, source.keys)
            _copy_raw_bits(self._value_input, source.values)
            elapsed = self._clock_ns() - started
            staged = E5RTStagedCapsule(
                nonce,
                int(source.generation),
                str(source.source_id),
                int(source.offset),
                tuple(source.layout_fingerprint),
                int(elapsed),
                self.spec.input_bytes,
            )
            with self._lock:
                if self._state != "staging" or self._nonce != nonce:
                    raise CacheCapsuleError("e5rt capsule stage lost ownership")
                self._state = "staged"
                self._count("stages", elapsed)
            return staged
        except BaseException:
            with self._lock:
                if self._nonce == nonce:
                    self._state = "idle"
                    self._count("errors")
                    self._release_if_requested_locked()
            raise

    def build(self, staged: E5RTStagedCapsule) -> E5RTBuiltCapsule:
        with self._lock:
            if self._state != "staged" or staged.nonce != self._nonce:
                raise CacheCapsuleError("e5rt capsule build has no staged slot")
            self._state = "running"
        started = self._clock_ns()
        try:
            self.program.execute()
            elapsed = self._clock_ns() - started
            built = E5RTBuiltCapsule(
                staged,
                self._key_output,
                self._value_output,
                int(elapsed),
            )
            with self._lock:
                if self._state != "running" or staged.nonce != self._nonce:
                    raise CacheCapsuleError("e5rt capsule build lost ownership")
                self._state = "built"
                self._count("executes", elapsed)
            return built
        except BaseException:
            with self._lock:
                if staged.nonce == self._nonce:
                    self._state = "idle"
                    self._count("errors")
                    self._release_if_requested_locked()
            raise

    def adopt(
        self,
        built: E5RTBuiltCapsule,
        source: KVCachePlaneSource,
    ) -> CacheCapsuleProduct:
        self._validate_source(source)
        staged = built.staged
        if (
            staged.source_generation != source.generation
            or staged.source_id != source.source_id
            or staged.offset != source.offset
            or staged.layout_fingerprint != tuple(source.layout_fingerprint)
        ):
            raise CacheCapsuleError("e5rt capsule identity changed before adoption")
        with self._lock:
            if self._state != "built" or staged.nonce != self._nonce:
                raise CacheCapsuleError("e5rt capsule adoption has no built slot")
            if self._cancel_reason is not None:
                raise CacheCapsuleError(
                    f"e5rt capsule was cancelled:{self._cancel_reason}"
                )
            self._state = "adopting"
        started = self._clock_ns()
        try:
            keys = self._adopt_array(built.key_output, self.spec.source_dtype)
            values = self._adopt_array(built.value_output, self.spec.source_dtype)
            elapsed = self._clock_ns() - started
            payload = KVCacheCapsulePayload(
                keys,
                values,
                source.offset,
                source.generation,
                source.source_id,
                source.layout_fingerprint,
                "e5rt",
                _raw_digest(keys, values) if source.raw_digest is not None else None,
            )
            with self._lock:
                if self._state != "adopting" or staged.nonce != self._nonce:
                    raise CacheCapsuleError("e5rt capsule adoption lost ownership")
                self._state = "leased"
                self._count("adoptions", elapsed)
            return CacheCapsuleProduct(
                payload,
                backing_owner=_CapsuleBacking(self, staged.nonce),
            )
        except BaseException:
            with self._lock:
                if staged.nonce == self._nonce:
                    self._state = "built"
                    self._count("errors")
            raise

    def cancel(self, staged: E5RTStagedCapsule, reason: str) -> None:
        with self._lock:
            if staged.nonce != self._nonce or self._state == "idle":
                return
            self._cancel_reason = str(reason)
            self._count("cancels")

    def abort(self, staged: E5RTStagedCapsule, reason: str) -> None:
        """Cancel a staged slot, releasing it immediately if build has not begun."""

        with self._lock:
            if staged.nonce != self._nonce or self._state == "idle":
                return
            self._cancel_reason = str(reason)
            self._count("cancels")
            if self._state == "staged":
                self._state = "idle"
                self._cancel_reason = None
                self._release_if_requested_locked()

    def discard(self, built: E5RTBuiltCapsule) -> None:
        with self._lock:
            if built.staged.nonce != self._nonce:
                return
            if self._state == "leased":
                raise CacheCapsuleError("cannot discard a leased e5rt capsule")
            self._state = "idle"
            self._cancel_reason = None
            self._count("discards")
            self._release_if_requested_locked()

    def _release_adopted(self, nonce: int) -> None:
        with self._lock:
            if nonce != self._nonce:
                return
            if self._state != "leased":
                raise CacheCapsuleError("e5rt capsule backing is not leased")
            self._state = "idle"
            self._cancel_reason = None
            self._count("backing_releases")
            self._release_if_requested_locked()

    def _release_if_requested_locked(self) -> None:
        if self._close_requested and self._state == "idle" and not self._released:
            self.program.release()
            self._released = True

    def close(self) -> None:
        with self._lock:
            self._close_requested = True
            self._release_if_requested_locked()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _dtype_name(value: Any) -> str:
    name = str(getattr(value, "dtype", ""))
    normalized = name.rsplit(".", 1)[-1]
    if normalized in ("float16", "bfloat16"):
        return normalized
    raise CacheCapsuleUnsupported(f"unsupported_e5rt_source_dtype:{name or 'unknown'}")


def _raw_u16(value: Any) -> np.ndarray:
    dtype = _dtype_name(value)
    if isinstance(value, np.ndarray):
        raw = value.view(np.uint16)
    else:
        import mlx.core as mx

        raw = value.view(mx.uint16) if dtype == "bfloat16" else value
        raw = np.asarray(raw)
        if raw.dtype != np.uint16:
            raw = raw.view(np.uint16)
    return np.ascontiguousarray(raw)


def _copy_raw_bits(destination: np.ndarray, source: Any) -> None:
    raw = _raw_u16(source)
    if tuple(destination.shape) != tuple(raw.shape):
        raise CacheCapsuleUnsupported("e5rt_input_geometry_mismatch")
    np.copyto(destination.view(np.uint16), raw)


def _adopt_mlx_array(view: np.ndarray, dtype: str):
    import mlx.core as mx

    adopted = mx.from_dlpack(view, copy=False)
    if dtype == "bfloat16":
        adopted = adopted.view(mx.uint16).view(mx.bfloat16)
    elif adopted.dtype != mx.float16:
        raise CacheCapsuleError("e5rt FP16 output adoption changed dtype")
    return adopted


def _raw_digest(keys: Any, values: Any) -> str:
    digest = hashlib.sha256()
    for value in (keys, values):
        raw = _raw_u16(value)
        digest.update(str(raw.dtype).encode())
        digest.update(str(tuple(raw.shape)).encode())
        digest.update(raw.tobytes(order="C"))
    return digest.hexdigest()
