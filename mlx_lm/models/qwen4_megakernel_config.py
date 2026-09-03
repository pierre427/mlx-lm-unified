"""One place that decides the megakernel's settings, and says where each came from.

Before this module there were three kinds of number in the build and no way to
tell them apart from the outside: values an operator set (``THREADS`` from the
environment), values a measurement produced (``T=512``, ``G=40``, the per-phase
row counts), and values that were simply true of an M5 Max (40 threadgroups is
40 GPU cores).  All three arrived as module constants.  A run could not state
which of its settings were chosen for it and which were inherited from a
different machine.

``resolve()`` answers that.  Every setting comes back as a value *and* a
source, in a fixed precedence:

1. **env** -- an explicit environment variable.  An operator overrides
   everything, including a calibration, because that is what an override is
   for.
2. **cache** -- a measurement for THIS device signature, from the autotune
   cache (``qwen4_megakernel_tune``).
3. **probe** -- derived from what the device reports.  ``G`` is the GPU core
   count scaled to the measured 512-threads-per-core residency budget, so the
   geometry follows the chip instead of the chip we tuned on.
4. **shipped** -- the phase-B M5 Max measurement, used when the probe cannot
   read what the derivation needs.

``portability_refusal()`` is the other half.  The build's caps were tuned, not
checked: 512 threads is under this device's 1024 limit and the 16 KiB
threadgroup budget is under its 32 KiB arena, and neither fact was ever
asserted.  On a part with a smaller arena, a smaller threadgroup, or less
memory than the model needs, the first evidence today is a compile failure, an
allocation failure, or a wrong answer.  Each of those is now a named refusal
before the GPU is touched.

**Environment variables** (all optional; the megakernel is default OFF):

``MLX_QWEN4_MEGAKERNEL``
    ``1``/``on`` to enable the path at all.  Read by ``qwen4_megakernel``.
``MLX_QWEN4_MEGAKERNEL_THREADS``
    Threads per threadgroup.  Explicit; beats the cache and the probe.
``MLX_QWEN4_MEGAKERNEL_GROUPS``
    Threadgroups in the persistent grid.  Explicit; beats the cache and probe.
``MLX_QWEN4_MEGAKERNEL_SPIN_CAP``
    Grid-barrier spin cap before a threadgroup declares the grid non-resident.
``MLX_QWEN4_MEGAKERNEL_ROWS``
    Per-phase rows per simdgroup as ``name=rows,name=rows`` (for example
    ``moe_down=2,gdn_in_proj=2``).  Names are the keys of ``PHASE_ROWS``.
    **Resolved and reported, not yet consumed**: ``build_body_kernel`` still
    takes its row counts from its own defaults, so this setting reaches the
    receipt and not the kernel.  Wiring it is one keyword at that call site
    and is deliberately left to the phase-E rewrite of the body, which owns
    that file today.
``MLX_QWEN4_MEGAKERNEL_THREADGROUP_BYTES``
    Threadgroup arena budget in bytes.  The 16 KiB default is a residency
    choice (two threadgroups per core), not the hardware limit.
``MLX_QWEN4_MEGAKERNEL_GPU_CORES``
    Core count for a machine whose IORegistry cannot be read.
``MLX_QWEN4_MEGAKERNEL_TUNE``
    ``auto`` (default) calibrates once per signature and yields the timing
    sweep to whoever holds the GPU lease; ``off`` skips the sweep but still
    runs the primitive tests; ``skip`` touches nothing at all; ``force``
    re-runs both and ignores the lease.
``MLX_QWEN4_MEGAKERNEL_TUNE_CACHE``
    Path of the tuning cache JSON.  Default
    ``~/.cache/mlxuag/megakernel-tune.json``.
``MLX_QWEN4_MEGAKERNEL_TUNE_BUDGET_S``
    Wall-clock budget for a calibration sweep.  Default 120 s.
``MLX_QWEN4_MEGAKERNEL_TUNE_BUSY_PATH``
    A lease path that DEFERS the timing sweep while it exists.  Default is
    the lab GPU lease directory; empty disables the check.
``MLX_QWEN4_MEGAKERNEL_REQUIRE_PRIMITIVES``
    ``1`` (default) refuses the kernel on a signature whose grid-barrier and
    device-scope fence tests have not passed here.  ``0`` is an escape hatch
    for bringing up a new part, and says so on the receipt.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from . import qwen4_megakernel_device as MD

# ---------------------------------------------------------- shipped defaults
# The phase-B measurement on an M5 Max (40 cores, 128 GiB, Metal 4).  These
# are the LAST resort, used when the probe cannot read what a derivation
# needs; ``tests/test_qwen4_megakernel_portability.py`` asserts they still
# equal the constants ``qwen4_megakernel`` ships.
SHIPPED_THREADS = 512
SHIPPED_GROUPS = 40
SHIPPED_SPIN_CAP = 400_000
SHIPPED_THREADGROUP_BYTES = 16 * 1024
SHIPPED_ROWS = {
    "gdn_in_proj": 2,
    "gdn_out_proj": 4,
    "moe_router": 4,
    "moe_gate_up": 2,
    "moe_down": 2,
    "generic_qmv": 4,
}
# The residency budget the geometry sweep landed on: every route to 512
# threads per core beat 256 and beat 1024-2048 (spec Sec. 2).  It is what
# turns a core count into a threadgroup count, and it is the ONE tuned number
# the probe-derived default carries forward from this chip -- the calibration
# re-measures it per signature and overwrites it from the cache.
THREADS_PER_CORE_BUDGET = 512

ENV_THREADS = "MLX_QWEN4_MEGAKERNEL_THREADS"
ENV_GROUPS = "MLX_QWEN4_MEGAKERNEL_GROUPS"
ENV_SPIN_CAP = "MLX_QWEN4_MEGAKERNEL_SPIN_CAP"
ENV_ROWS = "MLX_QWEN4_MEGAKERNEL_ROWS"
ENV_THREADGROUP_BYTES = "MLX_QWEN4_MEGAKERNEL_THREADGROUP_BYTES"
ENV_REQUIRE_PRIMITIVES = "MLX_QWEN4_MEGAKERNEL_REQUIRE_PRIMITIVES"


class ConfigError(ValueError):
    """A setting an operator stated that cannot be honoured."""


def _env_int(name: str, *, minimum: int = 1) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer; got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}; got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no"}:
        return False
    raise ConfigError(f"{name} must be 0/off or 1/on; got {raw!r}")


def _env_rows() -> Optional[dict[str, int]]:
    raw = os.environ.get(ENV_ROWS)
    if raw is None or not raw.strip():
        return None
    rows: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in SHIPPED_ROWS:
            raise ConfigError(
                f"{ENV_ROWS}: unknown phase {name!r}; "
                f"known phases are {sorted(SHIPPED_ROWS)}")
        try:
            rows[name] = int(value)
        except ValueError as exc:
            raise ConfigError(
                f"{ENV_ROWS}: {name} rows must be an integer; "
                f"got {value!r}") from exc
        if rows[name] < 1:
            raise ConfigError(f"{ENV_ROWS}: {name} rows must be at least 1")
    return rows or None


# ------------------------------------------------------- probe-derived rules
def derive_geometry(probe: MD.DeviceProbe) -> tuple[Optional[int],
                                                    Optional[int]]:
    """Threads and threadgroups implied by the device, or ``None``.

    The rule is the one the geometry round actually found: put
    ``THREADS_PER_CORE_BUDGET`` threads on every core, and prefer to get there
    with the WIDEST legal threadgroup rather than with more threadgroups --
    T=512/G=40 beat T=256/G=80 at the same 512 threads per core, because the
    hyper-connection phase wants the wider threadgroup and the barrier is
    charged for G.

    On this machine that reproduces 512 x 40 exactly.  On a 10-core part it
    gives 512 x 10; on a device whose threadgroup limit is 512 it gives
    512 x cores; on one limited to 256 it gives 256 x 2*cores.
    """
    cores = probe.gpu_cores
    limit = probe.max_threads_per_threadgroup
    if cores is None or cores < 1:
        return None, None
    threads = SHIPPED_THREADS if limit is None else min(SHIPPED_THREADS,
                                                        int(limit))
    threads -= threads % MD.SIMD_WIDTH
    if threads < MD.SIMD_WIDTH:
        return None, None
    groups = max(1, (cores * THREADS_PER_CORE_BUDGET) // threads)
    return threads, groups


def derive_threadgroup_bytes(probe: MD.DeviceProbe) -> Optional[int]:
    """Threadgroup arena budget implied by the device.

    16 KiB on this part is half of a 32 KiB arena -- the choice that keeps two
    threadgroups per core resident, and staging that crossed it (25.6 KiB)
    measured 4-6% SLOWER.  Expressed as "half the arena", not as 16 KiB, so a
    device with a different arena keeps the residency property rather than the
    number.
    """
    arena = probe.max_threadgroup_memory
    if arena is None or arena < 1024:
        return None
    return int(arena) // 2


# -------------------------------------------------------------- resolution
_KEYS = ("threads", "groups", "spin_cap", "threadgroup_bytes", "rows")


def resolve(
    *,
    probe: Optional[MD.DeviceProbe] = None,
    cache_entry: Optional[dict[str, Any]] = None,
    tune: bool = True,
) -> dict[str, Any]:
    """Final settings, each with the source it came from.

    ``tune=False`` resolves without consulting (or running) the calibration,
    which is what the tests and the calibration itself need -- the sweep
    cannot ask the cache for the answer it is about to measure.
    """
    probe = probe or MD.probe_device()
    cache_state: dict[str, Any] = {"consulted": bool(tune)}
    if tune and cache_entry is None:
        from . import qwen4_megakernel_tune as MT

        cache_entry, cache_state = MT.ensure_tuned(probe=probe)

    d_threads, d_groups = derive_geometry(probe)
    d_tg_bytes = derive_threadgroup_bytes(probe)
    cached = cache_entry or {}

    def pick(key: str, env_value, cache_value, probe_value, shipped_value):
        if env_value is not None:
            return env_value, "env"
        if cache_value is not None:
            return cache_value, "cache"
        if probe_value is not None:
            return probe_value, "probe"
        return shipped_value, "shipped"

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    for key, env_value, cache_value, probe_value, shipped_value in (
        ("threads", _env_int(ENV_THREADS, minimum=MD.SIMD_WIDTH),
         cached.get("threads"), d_threads, SHIPPED_THREADS),
        ("groups", _env_int(ENV_GROUPS), cached.get("groups"), d_groups,
         SHIPPED_GROUPS),
        ("spin_cap", _env_int(ENV_SPIN_CAP), cached.get("spin_cap"), None,
         SHIPPED_SPIN_CAP),
        ("threadgroup_bytes", _env_int(ENV_THREADGROUP_BYTES),
         cached.get("threadgroup_bytes"), d_tg_bytes,
         SHIPPED_THREADGROUP_BYTES),
    ):
        values[key], sources[key] = pick(
            key, env_value, cache_value, probe_value, shipped_value)

    # Rows resolve per phase, so a partial override -- one phase from the
    # environment, the rest from a calibration -- keeps both sources visible
    # instead of collapsing to whichever dict won.
    env_rows = _env_rows() or {}
    cache_rows = dict(cached.get("rows") or {})
    rows: dict[str, int] = {}
    row_sources: dict[str, str] = {}
    for name, shipped_value in SHIPPED_ROWS.items():
        if name in env_rows:
            rows[name], row_sources[name] = env_rows[name], "env"
        elif name in cache_rows:
            rows[name], row_sources[name] = int(cache_rows[name]), "cache"
        else:
            rows[name], row_sources[name] = shipped_value, "shipped"
    values["rows"] = rows
    sources["rows"] = ("env" if set(env_rows) == set(SHIPPED_ROWS)
                       else "mixed" if env_rows or cache_rows
                       else "shipped")

    return {
        "signature": probe.signature,
        "values": values,
        "sources": sources,
        "row_sources": row_sources,
        "cache": cache_state,
        "primitives": (cache_entry or {}).get("primitives"),
        "derived": {"threads": d_threads, "groups": d_groups,
                    "threadgroup_bytes": d_tg_bytes},
    }


# -------------------------------------------------------------- guardrails
def _pack_bytes(pack: Any) -> tuple[Optional[int], Optional[int]]:
    """Total packed bytes and the largest single group, or ``(None, None)``.

    Both matter and they fail differently: the total is a working-set
    question, the largest group is a ``maxBufferLength`` question, and a
    machine can pass one and fail the other.
    """
    buffers = getattr(pack, "buffers", None)
    if not buffers:
        return None, None
    try:
        sizes = [int(buf.size) * 4 for buf in buffers]
    except Exception:  # pragma: no cover - a stand-in pack in a unit test
        return None, None
    return sum(sizes), max(sizes)


def portability_refusal(
    *,
    threads: int,
    groups: int,
    width: int = 1,
    pack: Any = None,
    probe: Optional[MD.DeviceProbe] = None,
    scratch_bytes: int = 0,
    threadgroup_bytes: Optional[int] = None,
    primitives: Optional[dict[str, Any]] = None,
    require_primitives: Optional[bool] = None,
) -> Optional[str]:
    """The named reason this device cannot run this geometry, or ``None``.

    A refusal here is the difference between "this machine cannot hold the
    model" and a mid-launch allocation failure, and between "this threadgroup
    is wider than the device allows" and a compile error inside a decode.

    Unknown limits do NOT become refusals.  A probe that could not read the
    threadgroup arena leaves that check unmade and lets the primitive tests --
    which launch a real kernel at the real geometry -- be the evidence, which
    is the same rule the grid barrier already lives by: verified behaviour
    over documented guarantee.
    """
    probe = probe or MD.probe_device()

    if probe.architecture is not None and not probe.apple_gpu:
        return f"device architecture {probe.architecture}"

    limit = probe.max_threads_per_threadgroup
    if limit is not None and threads > int(limit):
        return f"threads {threads} over device max {int(limit)}"

    arena = probe.max_threadgroup_memory
    want_tg = (SHIPPED_THREADGROUP_BYTES if threadgroup_bytes is None
               else int(threadgroup_bytes))
    if arena is not None and want_tg > int(arena):
        return (f"threadgroup arena {want_tg} B over device limit "
                f"{int(arena)} B")

    total, largest = _pack_bytes(pack)
    if total is not None:
        working_set = probe.max_recommended_working_set_size
        needed = total + max(int(scratch_bytes), 0)
        if working_set is not None and needed > int(working_set):
            return (f"working set {needed / (1 << 30):.1f} GiB over device "
                    f"budget {int(working_set) / (1 << 30):.1f} GiB")
        max_buffer = probe.max_buffer_length
        if (largest is not None and max_buffer is not None
                and largest > int(max_buffer)):
            return (f"weight group {largest / (1 << 30):.1f} GiB over max "
                    f"buffer length {int(max_buffer) / (1 << 30):.1f} GiB")

    if require_primitives is None:
        require_primitives = _env_bool(ENV_REQUIRE_PRIMITIVES, True)
    if require_primitives:
        if not primitives:
            return f"primitives unvalidated for {probe.signature}"
        if not primitives.get("ok"):
            failed = primitives.get("failed") or "unknown"
            return f"primitive test failed: {failed}"
        # A barrier validated at a narrower grid says nothing about a wider
        # one: the spike's residency ceiling is a THREAD budget, and a grid
        # that exceeds it aborts rather than completing.
        tested = primitives.get("geometry") or {}
        if (int(tested.get("threads", 0)) * int(tested.get("groups", 0))
                < threads * groups):
            return (f"primitives validated at "
                    f"{tested.get('threads')}x{tested.get('groups')}, "
                    f"below {threads}x{groups}")
    return None


# ----------------------------------------------------------------- receipt
_LAST: Optional[dict[str, Any]] = None


def config_receipt(*, refresh: bool = False) -> dict[str, Any]:
    """Resolved settings, their sources, the probe and the cache state.

    This is what makes "the geometry came from a calibration on THIS machine"
    a checkable claim rather than an assumption a reader has to make.
    """
    global _LAST
    if _LAST is None or refresh:
        probe = MD.probe_device()
        try:
            resolved = resolve(probe=probe)
        except ConfigError as exc:
            resolved = {"error": str(exc), "signature": probe.signature}
        _LAST = {"device": probe.as_dict(), **resolved}
    return _LAST


def invalidate() -> None:
    """Forget the resolved settings (tests, and a re-tune)."""
    global _LAST
    _LAST = None
