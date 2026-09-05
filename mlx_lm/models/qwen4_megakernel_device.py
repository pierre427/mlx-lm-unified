"""What the machine actually is, read from the machine.

The megakernel's shipped numbers -- ``T=512``, ``G=40``, the per-phase row
counts, the 16 KiB threadgroup budget, "one kernel, not two" -- are one
measured point on one chip: an M5 Max, 40 GPU cores, ~600 GB/s, Metal 4.  None
of them was ever read from the device.  On a 10-core M5 the grid is four times
the core count; on an Ultra the grid spans two dies and a device-scope barrier
crosses the interposer; on a 24 GiB part the model does not fit at all and the
first evidence of that today would be an allocation failure inside a launch.

This module is the part that reads.  It answers, for the current device:

* what MLX reports -- ``architecture``, ``memory_size``,
  ``max_recommended_working_set_size``, ``max_buffer_length``,
  ``resource_limit``;
* the GPU **core count**, which MLX does not report, from the IORegistry
  (``gpu-core-count``) and, failing that, ``system_profiler``;
* the Metal **device limits** MLX does not expose -- max threads per
  threadgroup, max threadgroup memory -- read straight off ``MTLDevice``
  through the Objective-C runtime;
* the Metal **family** the device supports, which is how "does a device-scope
  barrier exist here" is asked without launching anything.

**Nothing here guesses.**  A value that cannot be read is ``None`` and is named
in ``DeviceProbe.unknown``; it is the caller's business whether an unknown
limit is a refusal, a skipped check, or a fallback.  A probe that silently
substituted a plausible number would be worse than one that failed: the whole
point of the exercise is that the shipped constants are already plausible
numbers substituted for measurements.

``DeviceProbe.signature`` is the cache key for everything downstream --
architecture, core count, memory class and the MLX build.  Two machines with
the same signature may share a tuning; a machine whose signature is not in the
cache has never been calibrated here and must be, primitive tests included.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------- constants
# MTLGPUFamily values, from Metal/MTLDevice.h.  Only the ones that tell us
# something the megakernel cares about are listed: Metal3 is where the
# device-scope threadgroup barrier the grid barrier is built on appears, and
# the Apple family fixes the SIMD width at 32, which every phase body assumes.
_GPU_FAMILIES = (
    ("apple7", 1007),
    ("apple8", 1008),
    ("apple9", 1009),
    ("apple10", 1010),
    ("metal3", 5001),
    ("metal4", 5002),
)
# The environment override for a core count that cannot be read.  A machine
# whose IORegistry is not readable (a VM, a sandbox) can still be tuned if the
# operator states the count; stating it is a decision, not a guess.
ENV_GPU_CORES = "MLX_QWEN4_MEGAKERNEL_GPU_CORES"
# Every phase body maps its rows over 32-lane simdgroups.
SIMD_WIDTH = 32


@dataclass(frozen=True)
class DeviceProbe:
    """One reading of the device.  ``None`` means "could not be read"."""

    architecture: Optional[str] = None
    device_name: Optional[str] = None
    memory_size: Optional[int] = None
    max_recommended_working_set_size: Optional[int] = None
    max_buffer_length: Optional[int] = None
    resource_limit: Optional[int] = None
    gpu_cores: Optional[int] = None
    max_threads_per_threadgroup: Optional[int] = None
    max_threadgroup_memory: Optional[int] = None
    unified_memory: Optional[bool] = None
    metal_families: tuple[str, ...] = ()
    metal_version: Optional[int] = None
    mlx_version: Optional[str] = None
    # Where each field came from, so a receipt can say "cores: ioregistry" and
    # a bug report can say which reader lied.
    sources: dict[str, str] = field(default_factory=dict)
    # Fields that could not be read at all, by name.
    unknown: tuple[str, ...] = ()

    # ------------------------------------------------------------ derived
    @property
    def simd_width(self) -> int:
        return SIMD_WIDTH

    @property
    def memory_gib(self) -> Optional[int]:
        if self.memory_size is None:
            return None
        return int(round(self.memory_size / (1 << 30)))

    @property
    def apple_gpu(self) -> bool:
        """An Apple GPU, which is the only thing the phase bodies are written
        for: 32-lane simdgroups, unified memory, no discrete transfer."""
        return str(self.architecture or "").startswith("applegpu")

    @property
    def multi_die(self) -> Optional[bool]:
        """Ultra-class parts fuse two dies.  A grid-wide device-scope barrier
        then crosses the interposer, which is the one geometry assumption in
        the build that no measurement on this machine can stand in for.

        ``None`` when the device name could not be read -- the answer is not
        "no".
        """
        if self.device_name is None:
            return None
        return "ultra" in self.device_name.lower()

    @property
    def signature(self) -> str:
        """Stable cache key: architecture, cores, memory class, MLX build.

        Deliberately coarse in one direction and exact in another.  Coarse:
        two M5 Maxes with different display counts or thermal states share a
        key, because the tuning does.  Exact on the MLX build: a wheel change
        can move a kernel's codegen, and a tuning measured against another
        compiler is not evidence about this one.
        """
        arch = self.architecture or "unknown"
        cores = "unk" if self.gpu_cores is None else str(self.gpu_cores)
        mem = "unk" if self.memory_gib is None else f"{self.memory_gib}g"
        mlx = self.mlx_version or "unknown"
        return f"{arch}-{cores}c-{mem}-mlx{mlx}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "device_name": self.device_name,
            "memory_size": self.memory_size,
            "memory_gib": self.memory_gib,
            "max_recommended_working_set_size":
                self.max_recommended_working_set_size,
            "max_buffer_length": self.max_buffer_length,
            "resource_limit": self.resource_limit,
            "gpu_cores": self.gpu_cores,
            "max_threads_per_threadgroup": self.max_threads_per_threadgroup,
            "max_threadgroup_memory": self.max_threadgroup_memory,
            "unified_memory": self.unified_memory,
            "metal_families": list(self.metal_families),
            "metal_version": self.metal_version,
            "mlx_version": self.mlx_version,
            "apple_gpu": self.apple_gpu,
            "multi_die": self.multi_die,
            "signature": self.signature,
            "sources": dict(self.sources),
            "unknown": list(self.unknown),
        }


# ------------------------------------------------------------- MLX readings
def _mlx_device_info() -> dict[str, Any]:
    try:
        import mlx.core as mx

        return dict(mx.device_info())
    except Exception:  # pragma: no cover - no Metal device
        return {}


def _mlx_version() -> Optional[str]:
    try:
        import mlx.core as mx

        return str(mx.__version__)
    except Exception:  # pragma: no cover - MLX not importable
        return None


# ----------------------------------------------------------- Metal readings
class _MTLSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_ulong),
                ("height", ctypes.c_ulong),
                ("depth", ctypes.c_ulong)]


def _metal_device_limits() -> dict[str, Any]:
    """``MTLDevice`` limits MLX does not surface, through the ObjC runtime.

    MLX's ``device_info`` stops at buffer and working-set sizes; the two
    numbers that decide whether a *geometry* is even legal -- the maximum
    threads per threadgroup and the threadgroup memory arena -- are only on
    ``MTLDevice``.  Reading them costs one ``MTLCreateSystemDefaultDevice``
    and no GPU work.

    Every failure here is caught: this must never be the reason a decode dies.
    """
    out: dict[str, Any] = {}
    try:
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        metal = ctypes.CDLL(
            "/System/Library/Frameworks/Metal.framework/Metal")
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        device = metal.MTLCreateSystemDefaultDevice()
        if not device:
            return {}

        libc = ctypes.CDLL(None)

        def send(restype, obj, selector, argtypes=(), *args):
            fn = libc.objc_msgSend
            fn.restype = restype
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
            return fn(ctypes.c_void_p(obj),
                      ctypes.c_void_p(objc.sel_registerName(selector)), *args)

        size = send(_MTLSize, device, b"maxThreadsPerThreadgroup")
        out["max_threads_per_threadgroup"] = int(size.width)
        out["max_threadgroup_memory"] = int(
            send(ctypes.c_ulong, device, b"maxThreadgroupMemoryLength"))
        out["unified_memory"] = bool(
            send(ctypes.c_bool, device, b"hasUnifiedMemory"))
        name_obj = send(ctypes.c_void_p, device, b"name")
        if name_obj:
            raw = send(ctypes.c_char_p, name_obj, b"UTF8String")
            if raw:
                out["device_name"] = raw.decode("utf-8", "replace")
        families = []
        for label, value in _GPU_FAMILIES:
            if send(ctypes.c_bool, device, b"supportsFamily:",
                    (ctypes.c_long,), ctypes.c_long(value)):
                families.append(label)
        out["metal_families"] = tuple(families)
    except Exception:  # pragma: no cover - no Metal framework
        return out
    return out


def _metal_version_from_families(families) -> Optional[int]:
    """Metal *language* version implied by the supported families.

    The spike's probe kernel reported ``__METAL_VERSION__ 400`` on this
    machine; the family query gets the same answer without a launch.  The
    calibration re-reads it from an actual kernel and stores that -- this is
    the pre-launch estimate, and it is labelled as such by its source.
    """
    if "metal4" in families:
        return 400
    if "metal3" in families:
        return 320
    return None


# ----------------------------------------------------------- GPU core count
_IOREG_CORES = re.compile(rb'"gpu-core-count"\s*=\s*(\d+)')
_SYSPROFILE_CORES = re.compile(r"Total Number of Cores:\s*(\d+)")


def _cores_from_ioregistry() -> Optional[int]:
    try:
        raw = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
            capture_output=True, timeout=10).stdout
    except Exception:  # pragma: no cover - no ioreg
        return None
    match = _IOREG_CORES.search(raw or b"")
    return int(match.group(1)) if match else None


def _cores_from_system_profiler() -> Optional[int]:
    try:
        raw = subprocess.run(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType"],
            capture_output=True, timeout=30).stdout.decode("utf-8", "replace")
    except Exception:  # pragma: no cover - no system_profiler
        return None
    match = _SYSPROFILE_CORES.search(raw)
    return int(match.group(1)) if match else None


def _cores_from_env() -> Optional[int]:
    raw = os.environ.get(ENV_GPU_CORES)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def gpu_core_count() -> tuple[Optional[int], str]:
    """Cores and where they came from.

    Order is cheapest-and-most-authoritative first: an explicit environment
    override, then the IORegistry (a few milliseconds), then
    ``system_profiler`` (seconds, and only asked once per process).  If all
    three fail the answer is ``None`` -- never a default -- and the caller
    falls back to the shipped geometry with that fact on the receipt.
    """
    value = _cores_from_env()
    if value is not None:
        return value, "env"
    value = _cores_from_ioregistry()
    if value is not None:
        return value, "ioregistry"
    value = _cores_from_system_profiler()
    if value is not None:
        return value, "system_profiler"
    return None, "unreadable"


# --------------------------------------------------------------- the probe
_CACHED: Optional[DeviceProbe] = None


def probe_device(*, refresh: bool = False) -> DeviceProbe:
    """Read the device once per process (``refresh=True`` to read again)."""
    global _CACHED
    if _CACHED is not None and not refresh:
        return _CACHED

    info = _mlx_device_info()
    limits = _metal_device_limits()
    cores, cores_source = gpu_core_count()

    sources: dict[str, str] = {}
    unknown: list[str] = []

    def take(name: str, value: Any, source: str) -> Any:
        if value is None:
            unknown.append(name)
            sources[name] = "unreadable"
            return None
        sources[name] = source
        return value

    def from_info(name: str, cast=int) -> Any:
        raw = info.get(name)
        return take(name, None if raw is None else cast(raw), "mlx")

    families = tuple(limits.get("metal_families", ()))
    probe = DeviceProbe(
        architecture=from_info("architecture", str),
        # MLX and Metal both name the device; they agree, and MLX is the one
        # that is present whenever the megakernel can run at all.
        device_name=take("device_name",
                         info.get("device_name") or limits.get("device_name"),
                         "mlx" if info.get("device_name") else "metal"),
        memory_size=from_info("memory_size"),
        max_recommended_working_set_size=from_info(
            "max_recommended_working_set_size"),
        max_buffer_length=from_info("max_buffer_length"),
        resource_limit=from_info("resource_limit"),
        gpu_cores=take("gpu_cores", cores, cores_source),
        max_threads_per_threadgroup=take(
            "max_threads_per_threadgroup",
            limits.get("max_threads_per_threadgroup"), "metal"),
        max_threadgroup_memory=take(
            "max_threadgroup_memory",
            limits.get("max_threadgroup_memory"), "metal"),
        unified_memory=take("unified_memory",
                            limits.get("unified_memory"), "metal"),
        metal_families=families,
        metal_version=take("metal_version",
                           _metal_version_from_families(families),
                           "metal_family"),
        mlx_version=take("mlx_version", _mlx_version(), "mlx"),
        sources=sources,
        unknown=tuple(unknown),
    )
    _CACHED = probe
    return probe


def device_signature(*, refresh: bool = False) -> str:
    return probe_device(refresh=refresh).signature
