"""Calibrate the megakernel on a machine it has never run on, then remember.

Two different things happen here, and they answer two different questions.

**The primitive tests answer "is the kernel legal here at all".**  The grid
barrier is not a documented Metal guarantee -- it is a spin over device-scope
atomics that works because threadgroups happen to be co-resident, and the
feasibility spike measured its ceiling as a THREAD budget (~2,048 per core,
i.e. the barrier completes at G=40/80 at T=512 and aborts cleanly above it).
The device-scope fence is the same kind of fact: 0 stale words in 1.024e9 word
reads WITH the fence, ~99.98% stale without.  Both were measured on one chip.
Neither transfers.  So a signature that has not passed them here does not run
the kernel -- ``qwen4_megakernel_config.portability_refusal`` turns a missing
or failed primitive result into a named refusal.

**The sweep answers "which geometry is fastest here".**  It is deliberately
small: a bandwidth-bound 4-bit matvec chain separated by grid barriers, which
is the dominant per-token cost class, run at a bounded set of (threads,
threadgroups) and then at a bounded set of rows-per-simdgroup at the winner.
It is a starting point for an unknown part, not a replacement for the tuning
round that produced the shipped numbers -- and the receipt says ``cache`` so
nobody mistakes one for the other.

Both results go into a JSON cache keyed by the device signature.  A later load
reads the cache and launches nothing.  The cache is advisory in exactly one
direction: it can choose a geometry, and it can REFUSE one, but it can never
make a kernel run that the primitive tests did not clear.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import mlx.core as mx

from . import qwen4_megakernel_device as MD

# --------------------------------------------------------------- cache file
ENV_CACHE = "MLX_QWEN4_MEGAKERNEL_TUNE_CACHE"
ENV_MODE = "MLX_QWEN4_MEGAKERNEL_TUNE"
ENV_BUDGET = "MLX_QWEN4_MEGAKERNEL_TUNE_BUDGET_S"
CACHE_VERSION = 1
DEFAULT_CACHE = "~/.cache/mlxuag/megakernel-tune.json"
DEFAULT_BUDGET_S = 120.0
MODES = ("auto", "off", "skip", "force")


def cache_path() -> str:
    return os.path.expanduser(os.environ.get(ENV_CACHE) or DEFAULT_CACHE)


def tune_mode() -> str:
    mode = (os.environ.get(ENV_MODE) or "auto").strip().lower()
    if mode in {"1", "true", "on", "yes"}:
        mode = "auto"
    if mode in {"0", "false", "no"}:
        mode = "off"
    if mode not in MODES:
        raise ValueError(
            f"{ENV_MODE} must be one of {MODES}; got {mode!r}")
    return mode


def tune_budget_s() -> float:
    raw = os.environ.get(ENV_BUDGET)
    if raw is None or not raw.strip():
        return DEFAULT_BUDGET_S
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{ENV_BUDGET} must be positive; got {value}")
    return value


def load_cache(path: Optional[str] = None) -> tuple[dict[str, Any],
                                                    Optional[str]]:
    """The cache and the reason it could not be used, if any.

    A corrupt cache is not an error an operator should have to clear by hand:
    it is moved aside once, reported, and re-tuned over.  Losing a tuning is
    cheap; refusing to start because a JSON file is truncated is not.
    """
    path = path or cache_path()
    if not os.path.exists(path):
        return {"version": CACHE_VERSION, "entries": {}}, None
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(
                data.get("entries"), dict):
            raise ValueError("cache is not a {version, entries} object")
        if int(data.get("version", 0)) != CACHE_VERSION:
            raise ValueError(
                f"cache version {data.get('version')} != {CACHE_VERSION}")
    except Exception as exc:
        try:
            os.replace(path, path + ".corrupt")
        except OSError:  # pragma: no cover - unwritable directory
            pass
        return {"version": CACHE_VERSION, "entries": {}}, str(exc)
    return data, None


def read_entry(signature: str, path: Optional[str] = None
               ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    data, error = load_cache(path)
    entry = data["entries"].get(signature)
    return (entry if isinstance(entry, dict) else None), error


def write_entry(signature: str, entry: dict[str, Any],
                path: Optional[str] = None) -> str:
    """Merge one entry into the cache and replace the file atomically.

    Re-read before write: two processes bringing up the same new machine must
    not lose each other's entries, and a partially written cache is exactly
    the corruption the reader above has to clean up.
    """
    path = path or cache_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data, _ = load_cache(path)
    data["version"] = CACHE_VERSION
    data["entries"][signature] = entry
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------- shared MSL bits
# The grid barrier, verbatim in shape from the spike and the tuning harness:
# a device-scope threadgroup barrier, an arrival counter, a generation word,
# and a spin cap that sets an ABORT flag rather than hanging when the grid is
# not co-resident.  That abort is the whole reason this can be tested at all.
_GBAR = r"""
#include <metal_atomic>

inline bool gbar(device atomic_uint* ctr, device atomic_uint* ab,
                 uint ntg, uint tid, thread uint& phase, uint cap) {
  device atomic_uint* gen = ctr + 2;
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  if (tid == 0u) {
    if (atomic_load_explicit(ab, memory_order_relaxed) == 0u) {
      uint want = phase + 1u;
      uint prev = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
      if (prev + 1u == want * ntg) {
        atomic_store_explicit(gen, want, memory_order_relaxed);
      } else {
        uint spins = 0u;
        while (atomic_load_explicit(gen, memory_order_relaxed) < want) {
          if (atomic_load_explicit(ab, memory_order_relaxed) != 0u) break;
          if (++spins >= cap) {
            atomic_store_explicit(ab, 1u, memory_order_relaxed);
            break;
          }
        }
      }
    }
  }
  phase += 1u;
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  return atomic_load_explicit(ab, memory_order_relaxed) == 0u;
}
"""

# One kernel for both primitive tests, because they are the same experiment
# read two ways.  Each round: one threadgroup PUBLISHES a pattern, the grid
# barriers, every threadgroup READS it back and counts words that are not what
# was published.  Completing all the rounds is U1 (the barrier is usable at
# this geometry); reading zero stale words is U2 (the device-scope fence
# publishes across threadgroups).  A grid that is not co-resident sets the
# abort flag and stops -- it does not hang.
_PROBE_SRC = r"""
  uint tid = thread_position_in_threadgroup.x;
  const uint nt = NT;
  uint tg  = threadgroup_position_in_grid.x;
  uint ntg = threadgroups_per_grid.x;

  device atomic_uint* ctr =
      reinterpret_cast<device atomic_uint*>(const_cast<device uint*>(ctrl));
  device atomic_uint* ab = ctr + 1;
  device uint* pubw = const_cast<device uint*>(pub);

  const uint ROUNDS = params[0];
  const uint PUBW   = params[1];
  const uint CAP    = params[2];

  uint phase = 0u;
  uint stale_local = 0u;
  uint done = 0u;
  bool live = true;
  for (uint r = 0u; r < ROUNDS && live; ++r) {
    uint owner = r % ntg;
    uint stamp = (r + 1u) * 2654435761u;
    if (tg == owner) {
      for (uint i = tid; i < PUBW; i += nt) pubw[i] = stamp + i;
    }
    live = gbar(ctr, ab, ntg, tid, phase, CAP);
    if (!live) break;
    for (uint i = tid; i < PUBW; i += nt) {
      if (pub[i] != stamp + i) stale_local += 1u;
    }
    live = gbar(ctr, ab, ntg, tid, phase, CAP);
    done = r + 1u;
  }

  threadgroup uint red[32];
  uint sg = tid / 32u, lane = tid % 32u;
  uint s = simd_sum(stale_local);
  if (lane == 0u) red[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint total = 0u;
    for (uint i = 0u; i < nt / 32u; ++i) total += red[i];
    stale[tg] = total;
    phases[tg] = done;
  }
  if (tg == 0u && tid == 0u) {
    info[0] = (uint)__METAL_VERSION__;
    info[1] = atomic_load_explicit(ab, memory_order_relaxed);
    info[2] = nt;
    info[3] = ntg;
  }
"""

# The sweep workload: a persistent chain of 4-bit matvec phases separated by
# grid barriers.  Nothing here is Qwen-specific and nothing is loaded from
# disk -- the shapes are chosen so one phase streams 8 MiB, which is the cost
# class the real phases live in, and the activation is staged in threadgroup
# memory exactly as the real bodies stage theirs.
_SWEEP_SRC = r"""
  uint tid = thread_position_in_threadgroup.x;
  const uint nt = NT;
  uint tg  = threadgroup_position_in_grid.x;
  uint ntg = threadgroups_per_grid.x;
  uint sg  = tid / 32u;
  uint lane = tid % 32u;
  uint nsg = nt / 32u;

  device atomic_uint* ctr =
      reinterpret_cast<device atomic_uint*>(const_cast<device uint*>(ctrl));
  device atomic_uint* ab = ctr + 1;
  const uint CAP = params[0];
  const uint NPHASE = params[1];

  threadgroup float xs[KDIM];
  uint phase = 0u;
  for (uint p = 0u; p < NPHASE; ++p) {
    for (uint i = tid; i < KDIM; i += nt) xs[i] = float(xin[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint blocks = KDIM / 8u;
    for (uint base = (tg * nsg + sg) * ROWS; base < RDIM;
         base += ntg * nsg * ROWS) {
      float acc[ROWS];
      for (uint r = 0u; r < ROWS; ++r) acc[r] = 0.0f;
      for (uint k = lane; k < blocks; k += 32u) {
        float4 a0 = float4(xs[k * 8u + 0u], xs[k * 8u + 1u],
                           xs[k * 8u + 2u], xs[k * 8u + 3u]);
        float4 a1 = float4(xs[k * 8u + 4u], xs[k * 8u + 5u],
                           xs[k * 8u + 6u], xs[k * 8u + 7u]);
        for (uint r = 0u; r < ROWS; ++r) {
          uint row = base + r;
          if (row < RDIM) {
            uint w = qw[row * blocks + k];
            acc[r] += float(w & 0xFu) * a0.x
                    + float((w >> 4u) & 0xFu) * a0.y
                    + float((w >> 8u) & 0xFu) * a0.z
                    + float((w >> 12u) & 0xFu) * a0.w
                    + float((w >> 16u) & 0xFu) * a1.x
                    + float((w >> 20u) & 0xFu) * a1.y
                    + float((w >> 24u) & 0xFu) * a1.z
                    + float((w >> 28u) & 0xFu) * a1.w;
          }
        }
      }
      for (uint r = 0u; r < ROWS; ++r) {
        float total = simd_sum(acc[r]);
        uint row = base + r;
        if (lane == 0u && row < RDIM) out[row] = total;
      }
    }
    if (!gbar(ctr, ab, ntg, tid, phase, CAP)) break;
  }
  if (tg == 0u && tid == 0u) status[0] =
      atomic_load_explicit(ab, memory_order_relaxed);
"""

_PROBE_CACHE: dict[tuple, Any] = {}
_SWEEP_CACHE: dict[tuple, Any] = {}


def _probe_kernel(threads: int):
    if threads not in _PROBE_CACHE:
        _PROBE_CACHE[threads] = mx.fast.metal_kernel(
            name=f"qwen4_mk_primitives_t{threads}",
            input_names=["ctrl", "pub", "params"],
            output_names=["phases", "stale", "info"],
            source=_PROBE_SRC, header=_GBAR + f"\n#define NT {threads}u\n",
        )
    return _PROBE_CACHE[threads]


def _sweep_kernel(threads: int, rows: int, rdim: int, kdim: int):
    key = (threads, rows, rdim, kdim)
    if key not in _SWEEP_CACHE:
        header = _GBAR + (f"\n#define ROWS {rows}u\n#define NT {threads}u\n"
                          f"#define RDIM {rdim}u\n#define KDIM {kdim}u\n")
        _SWEEP_CACHE[key] = mx.fast.metal_kernel(
            name=f"qwen4_mk_sweep_t{threads}_r{rows}",
            input_names=["xin", "qw", "ctrl", "params"],
            output_names=["out", "status"],
            source=_SWEEP_SRC, header=header,
        )
    return _SWEEP_CACHE[key]


# ---------------------------------------------------------- primitive tests
def run_primitive_tests(*, threads: int, groups: int, rounds: int = 32,
                        pub_words: int = 4096,
                        spin_cap: int = 400_000) -> dict[str, Any]:
    """U1 (grid-barrier completion) and U2 (device-scope publication).

    Returns a result dict rather than raising: "this geometry aborts here" is
    an ANSWER, and it is the answer the guardrail needs.
    """
    started = time.perf_counter()
    kernel = _probe_kernel(threads)
    ctrl = mx.zeros((8,), dtype=mx.uint32)
    pub = mx.zeros((pub_words,), dtype=mx.uint32)
    params = mx.array([rounds, pub_words, spin_cap], dtype=mx.uint32)
    phases, stale, info = kernel(
        inputs=[ctrl, pub, params],
        output_shapes=[(groups,), (groups,), (4,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
        grid=(threads * groups, 1, 1),
        threadgroup=(threads, 1, 1),
    )
    mx.eval(phases, stale, info)
    phases_done = [int(v) for v in phases.tolist()]
    stale_words = int(sum(int(v) for v in stale.tolist()))
    info = [int(v) for v in info.tolist()]

    completed = min(phases_done) if phases_done else 0
    words_read = rounds * pub_words * groups
    failed = None
    if info[1] != 0:
        failed = "grid barrier aborted: the grid is not co-resident"
    elif completed != rounds:
        failed = (f"grid barrier completed {completed} of {rounds} rounds")
    elif stale_words:
        failed = (f"device-scope fence let {stale_words} of {words_read} "
                  f"words read stale")
    return {
        "ok": failed is None,
        "failed": failed,
        "geometry": {"threads": int(threads), "groups": int(groups)},
        "rounds": int(rounds),
        "rounds_completed": int(completed),
        "stale_words": stale_words,
        "words_read": int(words_read),
        "aborted": bool(info[1]),
        "metal_version": int(info[0]),
        "seconds": round(time.perf_counter() - started, 3),
    }


# ------------------------------------------------------------------- sweep
# Bounded on purpose.  Threads: the three widths the geometry round actually
# separated.  Per-core thread budgets: the residency band around the measured
# optimum.  Rows: the band where register spill starts (8 rows measured 217
# GB/s against 340 at 2).  9 geometries + 3 row settings = 12 measured points,
# 6 kernel compiles.
SWEEP_THREADS = (128, 256, 512)
SWEEP_THREADS_PER_CORE = (256, 512, 1024)
SWEEP_ROWS = (1, 2, 4)
SWEEP_RDIM = 8192
SWEEP_KDIM = 2048
SWEEP_PHASES = 16


def _time_config(*, threads: int, groups: int, rows: int, xin, qw,
                 reps: int = 7, warmup: int = 2,
                 spin_cap: int = 400_000) -> Optional[float]:
    """Best-of-``reps`` milliseconds for one call, or ``None`` if it aborted.

    Best-of, not mean: a persistent grid shares the machine with whatever else
    is resident, and the minimum is the only statistic that is about the
    geometry rather than about the neighbours.
    """
    kernel = _sweep_kernel(threads, rows, SWEEP_RDIM, SWEEP_KDIM)
    params = mx.array([spin_cap, SWEEP_PHASES], dtype=mx.uint32)
    best = None
    for i in range(warmup + reps):
        ctrl = mx.zeros((8,), dtype=mx.uint32)
        start = time.perf_counter()
        out, status = kernel(
            inputs=[xin, qw, ctrl, params],
            output_shapes=[(SWEEP_RDIM,), (1,)],
            output_dtypes=[mx.float32, mx.uint32],
            grid=(threads * groups, 1, 1),
            threadgroup=(threads, 1, 1),
        )
        mx.eval(out, status)
        elapsed = (time.perf_counter() - start) * 1e3
        if int(status.item()) != 0:
            return None
        if i >= warmup and (best is None or elapsed < best):
            best = elapsed
    return best


def run_sweep(*, probe: MD.DeviceProbe, budget_s: float = DEFAULT_BUDGET_S,
              spin_cap: int = 400_000) -> dict[str, Any]:
    """Measure a bounded grid of geometries, then rows at the winner."""
    started = time.perf_counter()
    cores = probe.gpu_cores or 1
    tg_limit = probe.max_threads_per_threadgroup or 1024

    xin = mx.random.uniform(shape=(SWEEP_KDIM,), dtype=mx.float32)
    qw = mx.random.randint(
        0, 2 ** 31 - 1, shape=(SWEEP_RDIM * SWEEP_KDIM // 8,)).astype(
            mx.uint32)
    mx.eval(xin, qw)

    results: list[dict[str, Any]] = []
    truncated = False

    def over_budget() -> bool:
        return (time.perf_counter() - started) > budget_s

    candidates = []
    for threads in SWEEP_THREADS:
        if threads > tg_limit:
            continue
        for per_core in SWEEP_THREADS_PER_CORE:
            groups = max(1, (cores * per_core) // threads)
            if (threads, groups) not in [(c[0], c[1]) for c in candidates]:
                candidates.append((threads, groups, per_core))

    best = None
    for threads, groups, per_core in candidates:
        if over_budget():
            truncated = True
            break
        ms = _time_config(threads=threads, groups=groups, rows=2, xin=xin,
                          qw=qw, spin_cap=spin_cap)
        record = {"threads": threads, "groups": groups,
                  "threads_per_core": per_core, "rows": 2,
                  "ms": None if ms is None else round(ms, 4),
                  "aborted": ms is None}
        results.append(record)
        if ms is not None and (best is None or ms < best["ms"]):
            best = {"threads": threads, "groups": groups, "rows": 2, "ms": ms}

    if best is None:
        return {
            "ok": False,
            "reason": "every geometry aborted the grid barrier",
            "configs": results, "truncated": truncated,
            "seconds": round(time.perf_counter() - started, 3),
        }

    for rows in SWEEP_ROWS:
        if rows == 2 or over_budget():
            truncated = truncated or over_budget()
            continue
        ms = _time_config(threads=best["threads"], groups=best["groups"],
                          rows=rows, xin=xin, qw=qw, spin_cap=spin_cap)
        results.append({"threads": best["threads"], "groups": best["groups"],
                        "threads_per_core": None, "rows": rows,
                        "ms": None if ms is None else round(ms, 4),
                        "aborted": ms is None})
        if ms is not None and ms < best["ms"]:
            best = {"threads": best["threads"], "groups": best["groups"],
                    "rows": rows, "ms": ms}

    best["ms"] = round(best["ms"], 4)
    return {
        "ok": True,
        "winner": best,
        "configs": results,
        "truncated": truncated,
        "workload": {
            "kind": "4-bit matvec chain with grid barriers",
            "rows": SWEEP_RDIM, "cols": SWEEP_KDIM,
            "phases": SWEEP_PHASES,
            "bytes_per_phase": SWEEP_RDIM * SWEEP_KDIM // 2,
        },
        "seconds": round(time.perf_counter() - started, 3),
    }


# -------------------------------------------------------------- calibration
def calibrate(*, probe: Optional[MD.DeviceProbe] = None, sweep: bool = True,
              budget_s: Optional[float] = None,
              path: Optional[str] = None,
              write: bool = True) -> dict[str, Any]:
    """Primitive tests, then (optionally) the sweep; write the cache entry.

    The entry is written after the primitive tests and again after the sweep,
    so an interrupted calibration still leaves the machine's PERMISSION to run
    behind it -- the expensive half is the sweep, and its absence only costs a
    geometry, never correctness.
    """
    probe = probe or MD.probe_device()
    budget_s = tune_budget_s() if budget_s is None else budget_s
    threads, groups = _default_geometry(probe)

    entry: dict[str, Any] = {
        "signature": probe.signature,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_name": probe.device_name,
        "gpu_cores": probe.gpu_cores,
        "mlx_version": probe.mlx_version,
        "primitives": run_primitive_tests(threads=threads, groups=groups),
    }
    if write:
        write_entry(probe.signature, entry, path)
    if not entry["primitives"]["ok"] or not sweep:
        return entry

    result = run_sweep(probe=probe, budget_s=budget_s)
    entry["sweep"] = result
    if result.get("ok"):
        winner = result["winner"]
        entry["threads"] = int(winner["threads"])
        entry["groups"] = int(winner["groups"])
        entry["rows"] = {"generic_qmv": int(winner["rows"])}
        # A geometry is only adopted if the barrier is proven AT that
        # geometry: the residency ceiling is a thread budget, and the sweep's
        # winner may be wider than the geometry the primitives cleared.
        if (winner["threads"], winner["groups"]) != (threads, groups):
            entry["primitives"] = run_primitive_tests(
                threads=int(winner["threads"]), groups=int(winner["groups"]))
            if not entry["primitives"]["ok"]:
                entry.pop("threads", None)
                entry.pop("groups", None)
                entry.pop("rows", None)
                entry["sweep"]["adopted"] = False
    if write:
        write_entry(probe.signature, entry, path)
    return entry


def _default_geometry(probe: MD.DeviceProbe) -> tuple[int, int]:
    """The geometry the primitive tests run at before anything is measured."""
    from . import qwen4_megakernel_config as MC

    threads, groups = MC.derive_geometry(probe)
    if threads is None or groups is None:
        return MC.SHIPPED_THREADS, MC.SHIPPED_GROUPS
    return threads, groups


_ENSURED: dict[str, tuple[Optional[dict[str, Any]], dict[str, Any]]] = {}


def ensure_tuned(*, probe: Optional[MD.DeviceProbe] = None,
                 path: Optional[str] = None
                 ) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """The cache entry for this device, calibrating once if there is none.

    Runs at most once per signature per process.  Every failure -- no Metal
    device, an unwritable cache, a kernel that will not compile -- comes back
    as state, not an exception: a calibration that cannot run must leave the
    caller refusing the kernel cleanly, not crashing the model load.
    """
    probe = probe or MD.probe_device()
    signature = probe.signature
    if signature in _ENSURED:
        return _ENSURED[signature]

    mode = tune_mode()
    entry, error = read_entry(signature, path)
    state: dict[str, Any] = {
        "path": path or cache_path(), "mode": mode, "hit": entry is not None,
        "error": error, "calibrated": False, "signature": signature,
    }
    if mode == "skip":
        state["skipped"] = "MLX_QWEN4_MEGAKERNEL_TUNE=skip"
        _ENSURED[signature] = (entry, state)
        return entry, state

    need_primitives = entry is None or not (entry.get("primitives") or {}).get(
        "ok")
    # A sweep that RAN and produced no winner is an answer, not a gap: the
    # entry keeps its `sweep` record and later loads stop re-measuring it.
    need_sweep = mode == "force" or (
        mode == "auto" and (entry is None or ("threads" not in entry
                                              and "sweep" not in entry)))
    if mode == "force" or need_primitives or need_sweep:
        try:
            entry = calibrate(probe=probe, sweep=(mode != "off") and (
                need_sweep or mode == "force"), path=path)
            state["calibrated"] = True
        except Exception as exc:  # pragma: no cover - no Metal device
            state["error"] = f"{type(exc).__name__}: {exc}"
    _ENSURED[signature] = (entry, state)
    return entry, state


def forget() -> None:
    """Drop the per-process memo (tests, and a forced re-tune)."""
    _ENSURED.clear()


def main(argv=None) -> int:  # pragma: no cover - operator entry point
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate the megakernel.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--budget", type=float, default=None)
    args = parser.parse_args(argv)

    probe = MD.probe_device()
    if args.force:
        entry = calibrate(probe=probe, sweep=not args.no_sweep,
                          budget_s=args.budget, path=args.cache)
    else:
        entry, state = ensure_tuned(probe=probe, path=args.cache)
        print(json.dumps({"state": state}, indent=2))
    print(json.dumps({"device": probe.as_dict(), "entry": entry}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
