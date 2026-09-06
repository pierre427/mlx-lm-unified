# Copyright © 2026 Apple Inc.
"""Opt-in host-side stall attribution for the FN self-MTP decode round.

UNCOMMITTED LAB INSTRUMENTATION (mlx-uag, 2026-09-06). Default OFF and
behavior-neutral when off: the flags below are captured once at import (the
same convention as the qwen4 env levers), so a live lane that imports this
module pays only one boolean test per guarded span and nothing else.

WHY. A clean-GPU probe (results/fn-verify-launchbound-probe-20260906.py) showed
FN fused-GDN forwards are GPU-bound (host-exposed 4.1% width-1, ~0% width-3
verify). So the self-MTP round's ~29% host-side idle lives in the orchestration
BETWEEN forwards, not in the forwards. This module attributes that idle across
the four host sources the round runs on the generation thread:

  * ``ple``       -- the per-layer PLE NVMe pread/dequant (lookup_numpy), the
                     serial CPU work embedded in every trunk forward's graph
                     construction. Off the ideal critical path (prefetchable).
  * ``receipts``  -- indexed-QSA / PLE-route receipt bookkeeping recorded per
                     step. Off-path (pure host dict/counter work; the device
                     attest sync is deferred to status readback).
  * ``verify_wait`` -- the verify-boundary ``mx.eval`` that drains the (GPU-
                     bound) verify forward. This is GPU compute wall, NOT a
                     recoverable host stall; split out so ``accept`` is the
                     pure host share.
  * ``accept``    -- the self-MTP acceptance HOST work AFTER the verify sync:
                     the .tolist(), the longest-accepted-prefix scan, and the
                     commit/rollback cache trims. Serial (on the critical path
                     between the verify forward and the next draft); the
                     fuse-on-device candidate.
  * ``mtp_draft`` -- the host control of the k-step autoregressive MTP draft
                     loop (python loop + per-step graph build + async dispatch).
                     Serial.

TIMED vs EXPOSED. ``ple`` and ``receipts`` are off-critical-path: their timed
host cost can be overlapped (async prefetch / deferral), so their EXPOSED cost
is measured separately by the harness as the round-wall (decode tok/s) delta
when the work is stubbed out (MLXUAG_STUB_PLE / MLXUAG_STUB_RECEIPTS) while the
GPU graph shape is held fixed -- since the forwards are known GPU-bound, a wall
delta from stubbing a host hook attributes to that hook's exposed host cost.
``accept`` and ``mtp_draft`` are serial, so exposed == timed (no stub).

The per-bucket spans are all accumulated BEFORE the round's first yield, so the
round total never includes downstream consumer (detokenizer/server) time.
"""

from __future__ import annotations

import os
import threading
import time

# Captured once at import (env set before mlx_lm import, like the qwen4 levers).
ENABLED = os.getenv("MLXUAG_HOST_TIMING") == "1"
STUB_PLE = os.getenv("MLXUAG_STUB_PLE") == "1"
STUB_RECEIPTS = os.getenv("MLXUAG_STUB_RECEIPTS") == "1"

BUCKETS = ("ple", "receipts", "verify_wait", "accept", "mtp_draft")
_MAX_ROUNDS = 1 << 20

_LOCAL = threading.local()


def _state():
    state = getattr(_LOCAL, "state", None)
    if state is None:
        state = {
            "rounds": 0,
            "totals": {b: 0.0 for b in BUCKETS},  # summed ms across rounds
            "current": {b: 0.0 for b in BUCKETS},  # ms in the round in flight
        }
        _LOCAL.state = state
    return state


def start_round() -> None:
    """Open a fresh per-round accumulator (call once per spec cycle)."""
    if not ENABLED:
        return
    cur = _state()["current"]
    for b in BUCKETS:
        cur[b] = 0.0


def end_round() -> None:
    """Flush the round in flight into the aggregate (call before the yields)."""
    if not ENABLED:
        return
    state = _state()
    if state["rounds"] >= _MAX_ROUNDS:
        return
    for b in BUCKETS:
        state["totals"][b] += state["current"][b]
    state["rounds"] += 1


def tic() -> float:
    """Span start. Cheap; callers still guard with ``if ENABLED``."""
    return time.perf_counter()


def toc(bucket: str, t0: float) -> None:
    """Accumulate one span's elapsed ms into the round in flight."""
    if not ENABLED:
        return
    _state()["current"][bucket] += (time.perf_counter() - t0) * 1000.0


def report() -> dict:
    """Per-round mean ms per bucket for this thread's decode."""
    state = _state()
    rounds = max(state["rounds"], 0)
    denom = rounds if rounds else 1
    means = {b: state["totals"][b] / denom for b in BUCKETS}
    return {
        "enabled": ENABLED,
        "stub_ple": STUB_PLE,
        "stub_receipts": STUB_RECEIPTS,
        "rounds": rounds,
        "ms_per_round": means,
        "ms_per_round_total": sum(means.values()),
        "totals_ms": dict(state["totals"]),
    }


def reset() -> None:
    """Clear this thread's accumulator (harness warmup boundary)."""
    _LOCAL.state = None
