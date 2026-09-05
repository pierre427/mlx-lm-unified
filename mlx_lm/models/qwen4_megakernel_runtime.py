"""The megakernel as a SELECTABLE decode path: admission, ledgers, receipts.

Phases A and B built the pieces -- a weight pack with an offset table, a phase
schedule, and one persistent dispatch that walks it -- and proved each of them
against the stock arithmetic.  What none of them had was a caller.
``admit_megakernel_decode`` existed and failed closed, and nothing called it.

This module is that caller.  ``MegakernelDecoder`` owns one packed model, one
token schedule, and the ledgers a decode token reads and writes, and turns a
single token id into logits with ONE dispatch.  It is default OFF
(``MLX_QWEN4_MEGAKERNEL``). Structural and device-limit checks run before the
body and ledgers are allocated, and the packed-size check runs before those
large persistent allocations. Every launch synchronizes its device status
before it can be published as engaged.

**What stays on the host, and why.**  Two things, both because they depend on
the input TOKEN and not on any activation: ``embed_tokens`` and the PLE n-gram
gather.  The first is a lookup; the second is a file-backed lookup into a
29.8 GiB table that is deliberately never packed.  Both are done before the
launch and arrive as kernel inputs.  Everything else -- every projection, every
norm, the GDN core, attention including its ledger appends, the MoE, the
hyper-connection mixers, ``lm_head`` -- is inside the dispatch.

**The ledgers are bindings the kernel writes in place.**  A decode token
appends one KV column, one raw index key and (one time in four) one pooled
block, and it reads all three back later in the SAME launch, so returning them
as outputs would put the host in the middle of one dispatch.  They are written
through their bindings, the same ``const_cast`` the grid barrier already makes
on ``ctrl``.  ``seed_from_caches`` fills them from a stock prefill, which is
what makes a mixed run -- stock prefill, megakernel decode -- possible.
"""

from __future__ import annotations

import os
import re
import resource
import subprocess
import sys
import threading
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from . import qwen4_megakernel_pack as MP
from . import qwen4_megakernel_schedule as MS
from .qwen4_megakernel_body import (
    MAX_GROUPS,
    DualWidthMegakernelBody,
    MegakernelBody,
    OUT_NAMES,
    compute_tg_layout,
)
from .qwen4_megakernel_contract import (
    METAL_INT32_MAX,
    rounded_ledger_capacity,
    score_tile_layout,
    validate_launch_shapes,
    validate_model_binding,
    validate_model_contract,
    validate_position,
)

# Phase G: both a plain-decode-only build (one MegakernelBody, MAX_QUERY_WIDTH
# planes reserved but never touched at width 1) and a dual-width build (a
# narrow pipeline built eagerly plus a wide one built lazily on first verify
# call) stay reachable from the same MegakernelDecoder.  Default OFF, same
# posture as MLX_QWEN4_MEGAKERNEL itself -- opting a served profile in is a
# separate decision from having the code exist.
DUAL_WIDTH = bool(int(os.environ.get("MLX_QWEN4_MEGAKERNEL_DUAL_WIDTH") or 0))
from .qwen4_megakernel import (
    ACTL,
    ACTL_HEADER,
    ACTL_IDS,
    ACTL_M,
    ACTL_M0,
    ACTL_M_STRIDE,
    actl_words,
    BLOCK_TOPK,
    CONV_DIM,
    CONV_KERNEL,
    GDN_KEY_DIM,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HC_COUNT,
    HC_HIDDEN,
    HEAD_DIM,
    HIDDEN,
    IDX_COMPRESS,
    IDX_HEAD_DIM,
    MAX_QUERY_WIDTH,
    N_Q_HEADS,
    N_KV_HEADS,
    OP_NAMES,
    PLE_STATE_LEN,
    SDPA_BLOCKS,
    SCRATCH,
    scratch_floats,
    VOCAB,
    MegakernelAdmission,
    _SPIN_CAP,
    _THREADGROUPS,
    _THREADS,
    _portable_config,
    _portability_refusal,
    admit_megakernel_decode,
    record_megakernel_receipt,
)

OUT = {name: index for index, name in enumerate(OUT_NAMES)}


class MegakernelDeviceAbort(RuntimeError):
    """A launch did not complete its device-wide barrier protocol."""


class MegakernelConstructionError(RuntimeError):
    """Construction failed after a progressive source rebind may have begun."""

    model_mutated = True
    stock_fallback_safe = False


def packing_memory_snapshot() -> dict[str, int]:
    """Read allocator use and host pressure without submitting device work."""
    if sys.platform != "darwin":
        raise RuntimeError("packing host-memory guard requires macOS")
    swap = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        check=True, capture_output=True, text=True, timeout=5,
    ).stdout
    match = re.search(r"\bused\s*=\s*([0-9.]+)([KMGTP]?)", swap)
    if match is None:
        raise RuntimeError("cannot read swap usage before packing")
    scale = 1024 ** ("KMGTP".index(match[2]) + 1) if match[2] else 1
    rss = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True, capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    if not rss.isdigit():
        raise RuntimeError("cannot read process RSS before packing")
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "host_rss_bytes": int(rss) * 1024,
        # Keep the historical peak as evidence, not current admission usage.
        "host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "swap_used_bytes": int(float(match[1]) * scale),
    }


def packing_memory_refusal(snapshot: dict[str, int], *, baseline_swap: int,
                           limit: int, group_bytes: int,
                           reserved_bytes: int) -> Optional[str]:
    if limit <= 0:
        return "device has no positive packing memory budget"
    if snapshot["swap_used_bytes"] - baseline_swap > 512 * (1 << 20):
        return "host swap grew by more than 512 MiB during packing"
    resident = max(snapshot["active_bytes"] + snapshot["cache_bytes"],
                   snapshot["host_rss_bytes"])
    required = resident + 2 * group_bytes + reserved_bytes
    if required > limit:
        return (f"packing peak bound {required} bytes exceeds device budget "
                f"{limit} bytes (resident={resident}, group={group_bytes}, "
                f"reserved={reserved_bytes})")
    return None


def ledger_allocation_bytes(*, total: int, attention_layers: int,
                            gdn_layers: int) -> dict[str, int]:
    """Exact persistent ledger allocation sizes, without creating arrays."""
    n_attn = max(int(attention_layers), 1)
    n_gdn = max(int(gdn_layers), 1)
    total = int(total)
    pooled_stride = total // IDX_COMPRESS
    sizes = {
        "kv": 2 * n_attn * N_KV_HEADS * total * HEAD_DIM * 2,
        "index": n_attn * (total + pooled_stride) * IDX_HEAD_DIM * 2,
        "gdn_conv": n_gdn * (CONV_KERNEL - 1) * CONV_DIM * 2,
        "gdn_recurrent": (
            n_gdn * GDN_VALUE_HEADS * GDN_VALUE_DIM * GDN_KEY_DIM * 4
        ),
        "ple_conv": PLE_STATE_LEN * HC_HIDDEN * 2,
    }
    sizes["total"] = sum(sizes.values())
    return sizes


def launch_allocation_bytes(*, width: int, gdn_layers: int,
                            vocab: int, score_blocks: int = 0) -> dict[str, int]:
    """Conservative non-scratch allocations created by one launch.

    The portability wrapper accounts for scratch separately.  This covers the
    input/control arrays constructed after admission plus every other kernel
    output while the old transactional state is still live.
    """
    width = max(int(width), 1)
    gdn_layers = max(int(gdn_layers), 1)
    vocab = max(int(vocab), 1)
    score_layout = score_tile_layout(int(score_blocks), width)
    sizes = {
        "input": width * (HC_COUNT * HIDDEN + HIDDEN + HC_HIDDEN) * 2,
        "actl": actl_words(width) * 4,
        "score_tiles": score_layout["bytes"],
        "out": width * HIDDEN * 2,
        "gdn_conv_out": gdn_layers * (CONV_KERNEL - 1) * CONV_DIM * 2,
        "gdn_recurrent_out": (
            gdn_layers * GDN_VALUE_HEADS * GDN_VALUE_DIM * GDN_KEY_DIM * 4
        ),
        "ple_conv_out": PLE_STATE_LEN * HC_HIDDEN * 2,
        "attention_partial_max": width * 2 * N_Q_HEADS * SDPA_BLOCKS * 4,
        "attention_partial_out": (
            width * N_Q_HEADS * SDPA_BLOCKS * HEAD_DIM * 2
        ),
        "result": width * vocab * 2,
        "status": 4 * 4,
    }
    sizes["total"] = sum(sizes.values())
    return sizes


def restored_scale_bias_bytes(pack) -> int:
    """Upper bound for contiguous source scale/bias copies after rebind."""
    return sum(int(entry.n_sb) * 4 for entry in pack.entries.values())


def build_control_block(position: int, width: int, *, total: int,
                        pooled_stride: int, n_attn_layers: int,
                        rope_theta: float) -> np.ndarray:
    """The per-slab attention control block, as plain numpy.

    A shared header, then ONE per-query block for each of the `width`
    queries, then room for each query's own selected-block list.

    **What advances mid-slab.**  Verify query `m` sits at position `p + m`, so
    its logical length, its closed-block count, its RoPE position, its
    physical KV slot and its TAIL BLOCK are all its own.  The tail is
    `(p + m) // block_size`, which crosses a block boundary for one slab in
    `block_size` -- at p = 3 and width 3 the three queries' tails are blocks
    0, 1, 1, and the first query is the one that CLOSES block 0.  Sharing any
    of these across the slab reproduces the 2026-09-03 defect at M queries
    instead of one: positions that are in no scored block and are simply not
    attended.

    Pure and model-free on purpose -- the trap it guards is arithmetic on
    positions, so a test should be able to reach it without a checkpoint.
    """
    position = int(position)
    width = int(width)
    total = int(total)
    if position < 0:
        raise ValueError(f"position must be non-negative; got {position}")
    if width < 1:
        raise ValueError(f"width must be positive; got {width}")
    if width > MAX_QUERY_WIDTH:
        raise ValueError(
            f"width {width} exceeds control capacity {MAX_QUERY_WIDTH}"
        )
    if total < 1:
        raise ValueError(f"total must be positive; got {total}")
    if total > METAL_INT32_MAX:
        raise OverflowError(
            f"ledger capacity {total} exceeds Metal int32 addressing"
        )
    if total % IDX_COMPRESS:
        raise ValueError(
            f"ledger capacity {total} is not divisible by {IDX_COMPRESS}"
        )
    expected_pooled_stride = total // IDX_COMPRESS
    if int(pooled_stride) != expected_pooled_stride:
        raise ValueError(
            f"pooled_stride {pooled_stride} does not match ledger geometry "
            f"{expected_pooled_stride}"
        )
    if position + width > total:
        raise ValueError(
            f"control span [{position}, {position + width}) exceeds "
            f"ledger capacity {total}"
        )
    actl = np.zeros(actl_words(width), np.uint32)
    actl[ACTL["total"]] = total
    actl[ACTL["block_size"]] = IDX_COMPRESS
    actl[ACTL["ids_from_scratch"]] = 1
    actl[ACTL["scale_bits"]] = np.float32(HEAD_DIM ** -0.5).view(np.uint32)
    actl[ACTL["rope_theta_bits"]] = np.float32(rope_theta).view(np.uint32)
    actl[ACTL["pooled_stride"]] = pooled_stride
    actl[ACTL["n_attn_layers"]] = n_attn_layers
    actl[ACTL["m_width"]] = width
    actl[ACTL["score_stride"]] = score_tile_layout(
        (position + width) // IDX_COMPRESS, width
    )["stride"]
    for m in range(width):
        pos = position + m
        length = pos + 1
        n_blocks = length // IDX_COMPRESS
        budget = min(BLOCK_TOPK, n_blocks)
        # THE TAIL BLOCK.  `n_blocks` counts only CLOSED blocks, so when the
        # length is not a multiple of the compression ratio the 1..3 newest
        # positions -- the query's own included -- are in no scored block.
        # Stock's sparse mask is `selected_tokens | tail` with
        # `tail = (complete <= t <= q_pos)`, and the shipped indexed kernel
        # appends that tail block at slot `n_sel` and counts it in `count`.
        # This path did neither until 2026-09-03, so three tokens in four
        # could not attend their own most recent context; the teacher-forced
        # perplexity gate is what found it.
        tail_block = pos // IDX_COMPRESS
        has_tail = int(length % IDX_COMPRESS != 0)
        count = budget + has_tail
        b = ACTL_M0 + m * ACTL_M_STRIDE
        actl[b + ACTL_M["u_width"]] = count
        actl[b + ACTL_M["count"]] = count
        actl[b + ACTL_M["n_sel"]] = budget
        actl[b + ACTL_M["tail_block"]] = tail_block
        actl[b + ACTL_M["q_pos"]] = pos
        actl[b + ACTL_M["logical_len"]] = length
        actl[b + ACTL_M["n_blocks"]] = n_blocks
        # Every block the grid names is causally valid at this query: a block
        # ends at 4b+3 and the deepest query is at `pos`, and
        # n_blocks = length // 4 already excludes the open tail block.
        actl[b + ACTL_M["n_valid"]] = n_blocks
        actl[b + ACTL_M["rope_pos"]] = pos
        actl[b + ACTL_M["kv_slot"]] = pos
        actl[b + ACTL_M["pool_new_block"]] = int(length % IDX_COMPRESS == 0)
        actl[b + ACTL_M["complete"]] = n_blocks * IDX_COMPRESS
    # Query 0's block also lands in the shared header slots, so a phase body
    # that has not been widened yet reads query 0 and a width-1 control block
    # is byte-for-byte the one the M=1 kernel shipped with.
    for name, off in ACTL_M.items():
        if name in ACTL:
            actl[ACTL[name]] = actl[ACTL_M0 + off]
    return actl


class MegakernelDecoder:
    """One packed model, one token schedule, and its ledgers."""

    def __init__(
        self,
        model,
        args,
        *,
        max_context: int,
        layers: Optional[list[int]] = None,
        threads: Optional[int] = None,
        groups: Optional[int] = None,
        rebind: bool = True,
        validate: bool = False,
        include_experts: bool = True,
        include_mtp: bool = False,
        include_lm_head: bool = True,
        max_group_bytes: int = 8 << 30,
        contiguous_source_sb: bool = True,
    ):
        if isinstance(max_context, bool) or not isinstance(max_context, int):
            raise TypeError("max_context must be an integer")
        if max_context < 1:
            raise ValueError(f"max_context must be positive; got {max_context}")
        validate_model_contract(args)
        validate_model_binding(model, args)
        if max_context > int(args.max_position_embeddings):
            raise ValueError(
                f"max_context {max_context} exceeds model position capacity "
                f"{args.max_position_embeddings}")
        self.max_context = max_context
        self.include_lm_head = bool(include_lm_head)
        self.args, self.model = args, model
        self.layer_types = list(args.layer_types)
        self.ple_layer_ids = [int(v) for v in getattr(args, "ple_layer_ids", ())]
        if len(self.ple_layer_ids) > 1:
            raise ValueError("megakernel supports at most one PLE layer")
        self.layers = (list(range(len(self.layer_types))) if layers is None
                       else [int(v) for v in layers])
        if len(self.layers) != len(set(self.layers)):
            raise ValueError("layers must not contain duplicates")
        invalid_layers = [
            index for index in self.layers
            if index < 0 or index >= len(self.layer_types)
        ]
        if invalid_layers:
            raise ValueError(f"layer index out of range: {invalid_layers[0]}")
        self.rope_theta = float(args.rope_theta)
        # The KV ledger is allocated once at its widest, so a decode run never
        # reallocates mid-flight.  The block grid divides it exactly.
        self.total = rounded_ledger_capacity(max_context, IDX_COMPRESS)
        self.pooled_stride = self.total // IDX_COMPRESS
        self.score_layout = score_tile_layout(
            self.pooled_stride, MAX_QUERY_WIDTH
        )

        self.attn_layers = [i for i in self.layers
                            if self.layer_types[i] != "linear_attention"]
        self.gdn_layers = [i for i in self.layers
                           if self.layer_types[i] == "linear_attention"]
        self.attn_slot = {i: r for r, i in enumerate(self.attn_layers)}
        self.ple_layers = [int(v) - 1 for v in self.ple_layer_ids
                           if int(v) - 1 in self.layers]

        # Finish model/shape/pack planning before progressive rebind can
        # mutate the live source model.
        plan = MP.decode_path_keys(
            num_layers=len(self.layer_types), layer_types=self.layer_types,
            ple_layer_ids=self.ple_layer_ids, layers=self.layers,
            include_mtp=include_mtp, include_experts=include_experts,
            include_lm_head=self.include_lm_head)
        self.source = MP.ModuleSource(model)
        self.pack_estimate = MP.estimate_pack(
            self.source, plan, max_group_bytes=max_group_bytes)
        if self.pack_estimate["groups"] > MAX_GROUPS:
            raise RuntimeError(
                f"pack requires {self.pack_estimate['groups']} groups, over "
                f"the kernel binding limit {MAX_GROUPS}"
            )

        self.portability = _portable_config()
        if self.portability.get("error"):
            raise RuntimeError(
                f"megakernel configuration failed: {self.portability['error']}"
            )
        values = self.portability.get("values", {})
        self.threads = int(values.get("threads", _THREADS) if threads is None
                           else threads)
        self.groups = int(values.get("groups", _THREADGROUPS) if groups is None
                          else groups)
        self.spin_cap = int(values.get("spin_cap", _SPIN_CAP))
        if self.threads <= 0 or self.threads > 512 or self.threads % 32:
            raise ValueError(
                f"threads={self.threads} must be a multiple of 32, <= 512"
            )
        if self.groups < 1:
            raise ValueError(f"groups must be positive; got {self.groups}")
        if GDN_VALUE_DIM % (self.threads // 32):
            raise ValueError(
                f"threads={self.threads} does not divide the GDN value dim"
            )
        self.ledger_bytes = ledger_allocation_bytes(
            total=self.total, attention_layers=len(self.attn_layers),
            gdn_layers=len(self.gdn_layers))
        self.max_launch_bytes = launch_allocation_bytes(
            width=MAX_QUERY_WIDTH, gdn_layers=len(self.gdn_layers),
            vocab=VOCAB if self.include_lm_head else 1,
            score_blocks=self.pooled_stride)
        # One output group plus up to one group of fusion/reshape staging.
        # Rebind release is checked between groups; it is not assumed.
        self.pack_transient_budget = int(
            2 * self.pack_estimate["largest_group_bytes"] if rebind
            else (self.pack_estimate["packed_bytes"]
                  + self.pack_estimate["largest_group_bytes"]))
        planned_restore_bytes = int(
            self.pack_estimate["scale_bias_bytes"]
            if rebind and contiguous_source_sb else 0)
        individual_buffers = {
            **{f"ledger.{k}": v for k, v in self.ledger_bytes.items()
               if k != "total"},
            **{f"launch.{k}": v for k, v in self.max_launch_bytes.items()
               if k != "total"},
            "planned_weight_group": self.pack_estimate["largest_group_bytes"],
        }
        preflight = _portability_refusal(
            self.portability, threads=self.threads, groups=self.groups,
            width=MAX_QUERY_WIDTH,
            extra_bytes=(self.ledger_bytes["total"]
                         + self.max_launch_bytes["total"]
                         + self.pack_transient_budget
                         + planned_restore_bytes),
            resident_bytes=int(mx.get_active_memory()),
            individual_buffer_bytes=individual_buffers)
        if preflight is not None:
            raise RuntimeError(f"megakernel preflight declined: {preflight}")

        self.pack_memory_checks = []
        self._pack_memory_limit = int(self.portability.get("device", {}).get(
            "max_recommended_working_set_size") or 0)
        self._pack_reserved_bytes = (
            self.ledger_bytes["total"] + self.max_launch_bytes["total"]
            + scratch_floats(MAX_QUERY_WIDTH) * 4 + planned_restore_bytes)
        baseline = packing_memory_snapshot()
        self._pack_baseline_swap = baseline["swap_used_bytes"]
        self._guard_pack_memory(stage="before_pack", group_index=-1,
                                group_bytes=self.pack_estimate["largest_group_bytes"])

        # 8 GiB is a HARD cap, not a preference: a group is one flat uint32
        # buffer, MLX shape dimensions are int32, and 2^31 words is 8 GiB.
        # Raising it to shrink the buffer count therefore is not available, so
        # the kernel carries ten weight bindings instead -- 8 expert groups
        # plus `main` and `lm_head`.
        self.pack = None
        self._source_may_be_mutated = False
        try:
            self._source_may_be_mutated = bool(rebind)
            self.pack = MP.build_pack(
                self.source, plan, validate=validate, rebind=rebind,
                max_group_bytes=max_group_bytes,
                memory_guard=self._guard_pack_memory)
            self.restore_source_bytes = (
                restored_scale_bias_bytes(self.pack)
                if rebind and contiguous_source_sb else 0)
            self._guard_pack_memory(stage="before_restore", group_index=-1,
                                    group_bytes=0)
            if rebind and contiguous_source_sb:
                self.restored_sb = self.restore_source_contiguity()
            self._pack_reserved_bytes -= planned_restore_bytes
            self._guard_pack_memory(stage="after_restore", group_index=-1,
                                    group_bytes=0)
            self.schedule = MS.build_token_schedule(
                self.pack, layer_types=self.layer_types, layers=self.layers,
                ple_layer_ids=self.ple_layer_ids,
                include_lm_head=self.include_lm_head)
            self.op_counts = _op_histogram(self.schedule)
            body_cls = DualWidthMegakernelBody if DUAL_WIDTH else MegakernelBody
            self.body = body_cls(
                self.pack, self.schedule,
                gdn_layers=max(len(self.gdn_layers), 1),
                vocab=VOCAB if self.include_lm_head else 1,
                threads=self.threads, groups=self.groups,
                spin_cap=self.spin_cap)
            self.dual_width = DUAL_WIDTH
            self._guard_pack_memory(stage="before_allocate", group_index=-1,
                                    group_bytes=0)
            self._allocate()
            self._pack_reserved_bytes -= self.ledger_bytes["total"]
            self._guard_pack_memory(stage="after_allocate", group_index=-1,
                                    group_bytes=0)
        except Exception as exc:
            if rebind:
                error = MegakernelConstructionError(
                    "megakernel construction failed after source rebind may "
                    "have begun; the model is mutated and MUST NOT be used as "
                    "a stock fallback"
                )
                error.pack_memory_checks = list(self.pack_memory_checks)
                raise error from exc
            raise
        self.position = 0
        self._pending = None
        self._pending_width = 0
        self._pending_position: Optional[int] = None
        self._pending_owner: Optional[int] = None
        self._poisoned_reason: Optional[str] = None
        self._state_lock = threading.RLock()
        self._in_flight = False
        self._in_flight_owner: Optional[int] = None

    def _guard_pack_memory(self, *, stage: str, group_index: int,
                           group_bytes: int) -> None:
        snapshot = packing_memory_snapshot()
        receipt = {"stage": stage, "group_index": group_index,
                   "group_bytes": group_bytes,
                   "reserved_bytes": self._pack_reserved_bytes, **snapshot}
        reason = packing_memory_refusal(
            snapshot, baseline_swap=self._pack_baseline_swap,
            limit=self._pack_memory_limit, group_bytes=group_bytes,
            reserved_bytes=self._pack_reserved_bytes)
        receipt["refusal"] = reason
        self.pack_memory_checks.append(receipt)
        if reason:
            raise RuntimeError(f"megakernel packing declined at {stage}: {reason}")

    def restore_source_contiguity(self) -> int:
        """Give the STOCK path back CONTIGUOUS scales and biases.

        The adopted scale/bias layout interleaves the two per row, so a rebind
        hands the source module two STRIDED views.  The megakernel reads the
        packed buffer directly and does not care, but a stock forward reading a
        strided ``scales`` pays a gather on every projection of every call --
        which would make an interleaved A/B measure the rebind rather than the
        kernel.  Copying just the scale/bias region back to contiguous costs
        one eighth of the 4-bit payload it describes (two bfloat16 per group of
        64 values), and the values are unchanged, so both arms still read the
        same numbers.

        Returns the number of entries restored.
        """
        restored = 0
        restored_arrays = []
        for key, entry in self.pack.entries.items():
            if entry.n_sb == 0 or not self.source.has(key):
                continue
            module = self.source._resolve(key)
            for part in ("scales", "biases"):
                value = getattr(module, part, None)
                if isinstance(value, mx.array):
                    contiguous = mx.contiguous(value)
                    setattr(module, part, contiguous)
                    restored_arrays.append(contiguous)
            restored += 1
        mx.eval(restored_arrays)
        return restored

    # ------------------------------------------------------------- ledgers
    def _allocate(self) -> None:
        n_attn = max(len(self.attn_layers), 1)
        n_gdn = max(len(self.gdn_layers), 1)
        # ONE binding each, not two.  `kv` is [keys | values] and `idxl` is
        # [raw index keys | pooled block keys]; the kernel splits them by a
        # constant offset it computes from control words it already has.  That
        # bought two of the three binding slots phase E needed, at the cost of
        # exactly nothing -- these were already allocated as one shape apiece.
        self.kv = mx.zeros(
            (2, n_attn * N_KV_HEADS, self.total, HEAD_DIM), mx.bfloat16)
        self.idxl = mx.zeros(
            (n_attn * (self.total + self.pooled_stride), IDX_HEAD_DIM),
            mx.bfloat16)
        self._raw_rows = n_attn * self.total
        self.cs = mx.zeros((n_gdn, CONV_KERNEL - 1, CONV_DIM), mx.bfloat16)
        self.rec = mx.zeros(
            (n_gdn, GDN_VALUE_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM), mx.float32)
        self.pconv = mx.zeros((PLE_STATE_LEN, HC_HIDDEN), mx.bfloat16)
        mx.eval(self.kv, self.idxl, self.cs, self.rec, self.pconv)

    # Views onto the merged ledgers, so seeding and the tests keep their
    # names.  These are slices of one buffer, not copies.
    @property
    def kbuf(self):
        return self.kv[0]

    @property
    def vbuf(self):
        return self.kv[1]

    @property
    def rawk(self):
        return self.idxl[: self._raw_rows]

    @property
    def pooled(self):
        return self.idxl[self._raw_rows:]

    def seed_from_caches(self, caches, *, indexer_of=None,
                         position: Optional[int] = None) -> dict[str, Any]:
        """Fill the ledgers from a stock prefill.

        ``caches`` is the list ``TextModel.make_cache`` returns, after a stock
        forward.  Everything copied here is a value the stock path produced, so
        a megakernel decode continuing from it is continuing the same sequence
        -- which is what makes the interleaved gate a comparison of the DECODE
        step rather than of two different histories.
        """
        with self._state_lock:
            if self._in_flight or self._pending is not None:
                raise RuntimeError("cannot seed during a megakernel transaction")
            if self._poisoned_reason is not None:
                raise MegakernelDeviceAbort(
                    "cannot seed a poisoned megakernel decoder")
            required = max(self.layers, default=-1) + 1
            if len(caches) < required:
                raise ValueError(
                    f"cache list has {len(caches)} entries; need {required}")

            # Validate and derive every payload before the first ledger write.
            attn_payloads = []
            seeded_length: Optional[int] = None
            pooled_count = 0
            for slot, index in enumerate(self.attn_layers):
                cache = caches[index]
                length = int(cache.offset)
                if length < 0 or length > self.total:
                    raise ValueError(
                        f"attention layer {index} offset {length} outside "
                        f"0..{self.total}")
                if seeded_length is not None and length != seeded_length:
                    raise ValueError(
                        f"attention cache lengths disagree: {seeded_length} "
                        f"and {length} at layer {index}")
                seeded_length = length
                keys, values, raw = cache.keys, cache.values, cache.index_keys
                for name, value in (("keys", keys), ("values", values)):
                    shape = () if value is None else tuple(value.shape)
                    if (len(shape) != 4 or shape[0] != 1
                            or shape[1] != N_KV_HEADS
                            or shape[2] < length or shape[3] != HEAD_DIM):
                        raise ValueError(
                            f"attention layer {index} {name} shape {shape} "
                            f"cannot seed [1,{N_KV_HEADS},{length},{HEAD_DIM}]")
                rshape = () if raw is None else tuple(raw.shape)
                if (len(rshape) != 3 or rshape[0] != 1
                        or rshape[1] < length or rshape[2] != IDX_HEAD_DIM):
                    raise ValueError(
                        f"attention layer {index} index_keys shape {rshape} "
                        f"cannot seed [1,{length},{IDX_HEAD_DIM}]")
                n_blocks = length // IDX_COMPRESS
                pooled = getattr(cache, "_qsa_pooled_keys", None)
                ratio = getattr(cache, "_qsa_pooled_ratio", None)
                if n_blocks and pooled is not None:
                    pshape = tuple(pooled.shape)
                    if (ratio != IDX_COMPRESS or len(pshape) != 3
                            or pshape[0] != 1 or pshape[1] < n_blocks
                            or pshape[2] != IDX_HEAD_DIM):
                        raise ValueError(
                            f"attention layer {index} cached pooled ledger "
                            "has incompatible ratio or shape")
                    pooled = pooled[0, :n_blocks]
                elif n_blocks:
                    if indexer_of is None:
                        raise ValueError(
                            f"attention layer {index} needs {n_blocks} pooled "
                            "blocks but has no cached summary or indexer")
                    starts = mx.arange(n_blocks) * IDX_COMPRESS
                    pooled = indexer_of(index)._pool_blocks(
                        raw[:, :n_blocks * IDX_COMPRESS], starts)[0]
                    if tuple(pooled.shape) != (n_blocks, IDX_HEAD_DIM):
                        raise ValueError(
                            f"attention layer {index} pooler returned "
                            f"{tuple(pooled.shape)}")
                attn_payloads.append(
                    (slot, length, keys, values, raw, pooled, n_blocks))
                pooled_count += n_blocks

            target_position = seeded_length if position is None else int(position)
            if target_position is None:
                raise ValueError(
                    "position is required when seeding without attention caches")
            if seeded_length is not None and target_position != seeded_length:
                raise ValueError(
                    f"seed position {target_position} != attention offset "
                    f"{seeded_length}")
            if target_position < 0 or target_position > self.max_context:
                raise ValueError(f"seed position {target_position} outside ledger")

            gdn_payloads = []
            for slot, index in enumerate(self.gdn_layers):
                cache = caches[index]
                cs, rec = cache[0], cache[1]
                if target_position and (cs is None or rec is None):
                    raise ValueError(
                        f"GDN layer {index} has no state at position "
                        f"{target_position}")
                if cs is not None and tuple(cs.shape) != (
                        1, CONV_KERNEL - 1, CONV_DIM):
                    raise ValueError(
                        f"GDN layer {index} conv state shape {tuple(cs.shape)}")
                if rec is not None and tuple(rec.shape) != (
                        1, GDN_VALUE_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM):
                    raise ValueError(
                        f"GDN layer {index} recurrent shape {tuple(rec.shape)}")
                gdn_payloads.append((slot, cs, rec))
            ple_payloads = []
            for index in self.ple_layers:
                state = caches[index][2]
                if target_position and state is None:
                    raise ValueError(
                        f"PLE layer {index} has no state at position "
                        f"{target_position}")
                if state is not None and tuple(state.shape) != (
                        1, PLE_STATE_LEN, HC_HIDDEN):
                    raise ValueError(
                        f"PLE layer {index} state shape {tuple(state.shape)}")
                ple_payloads.append(state)

            try:
                for slot, length, keys, values, raw, pooled, n_blocks in attn_payloads:
                    base = slot * N_KV_HEADS
                    self.kv[0, base:base + N_KV_HEADS, :length] = keys[0, :, :length]
                    self.kv[1, base:base + N_KV_HEADS, :length] = values[0, :, :length]
                    rbase = slot * self.total
                    self.idxl[rbase:rbase + length] = raw[0, :length]
                    if n_blocks:
                        pbase = self._raw_rows + slot * self.pooled_stride
                        self.idxl[pbase:pbase + n_blocks] = pooled.astype(mx.bfloat16)
                for slot, cs, rec in gdn_payloads:
                    if cs is not None:
                        self.cs[slot] = cs[0].astype(mx.bfloat16)
                    if rec is not None:
                        self.rec[slot] = rec[0].astype(mx.float32)
                for state in ple_payloads:
                    if state is not None:
                        self.pconv[:] = state[0].astype(mx.bfloat16)
                mx.eval(self.kv, self.idxl, self.cs, self.rec, self.pconv)
            except Exception as exc:
                self._poisoned_reason = (
                    f"partial cache seed: {type(exc).__name__}")
                raise MegakernelDeviceAbort(self._poisoned_reason) from exc
            self.position = target_position
            return {
                "attention": len(attn_payloads),
                "gdn": len(gdn_payloads),
                "ple": len(ple_payloads),
                "pooled_blocks": pooled_count,
                "position": target_position,
            }

    # ------------------------------------------------------------ admission
    def admit(self, *, width: int = 1, batch: int = 1, dtype=mx.bfloat16,
              speculating: bool = False, training: bool = False,
              sharded: bool = False, mask=None) -> MegakernelAdmission:
        decision = admit_megakernel_decode(
            width=width, batch=batch, pack=self.pack, schedule=self.schedule,
            speculating=speculating, training=training, sharded=sharded,
            mask=mask, dtype=dtype, threads=self.body.threads,
            groups=self.body.groups,
            layer_types=[self.layer_types[i] for i in self.layers],
        )
        if not decision.accepted:
            return decision
        launch_bytes = launch_allocation_bytes(
            width=width, gdn_layers=len(self.gdn_layers),
            vocab=VOCAB if self.include_lm_head else 1,
            score_blocks=(self.position + int(width)) // IDX_COMPRESS)
        compiled_width = (
            self.body.narrow_width
            if self.dual_width and width <= self.body.narrow_width
            else self.body.wide_width if self.dual_width
            else self.body.max_query_width
        )
        refusal = _portability_refusal(
            self.portability, threads=self.body.threads,
            groups=self.body.groups, width=width, pack=self.pack,
            extra_bytes=launch_bytes["total"],
            resident_bytes=int(mx.get_active_memory()),
            compiled_width=compiled_width,
            individual_buffer_bytes={
                f"launch.{name}": size for name, size in launch_bytes.items()
                if name != "total"
            })
        if refusal is not None:
            return MegakernelAdmission(False, refusal)
        return decision

    def _prepare_launch(self, position: int, width: int) -> None:
        owner = threading.get_ident()
        with self._state_lock:
            if self._poisoned_reason is not None:
                raise MegakernelDeviceAbort(
                    "megakernel decoder is poisoned after "
                    f"{self._poisoned_reason}; construct a new decoder")
            if self._in_flight:
                raise RuntimeError("megakernel launch is already in flight")
            if self._pending is not None:
                raise RuntimeError(
                    "megakernel launch already pending; commit or rollback first")
            validate_position(
                self.position, position, width, self.max_context)
            if (self.dual_width and width > 1
                    and not self.body.is_width_wrapper_prepared(width)):
                raise RuntimeError(
                    "wide wrapper is unprepared; call prepare_width_wrapper(width); "
                    "the first evaluated launch still compiles the Metal pipeline")
            self._in_flight = True
            self._in_flight_owner = owner

    def _release_launch_claim(self) -> None:
        owner = threading.get_ident()
        with self._state_lock:
            if self._in_flight and self._in_flight_owner == owner:
                self._in_flight = False
                self._in_flight_owner = None

    def prepare_width_wrapper(self, width: int) -> dict[str, Any]:
        """Prepare a Python wrapper, leaving Metal compilation to first launch."""
        width = int(width)
        with self._state_lock:
            if self._in_flight or self._pending is not None:
                raise RuntimeError("cannot prepare a wrapper during a transaction")
            if self._poisoned_reason is not None:
                raise MegakernelDeviceAbort("cannot prepare a poisoned decoder")
            if not self.dual_width:
                if width < 1 or width > self.body.max_query_width:
                    raise ValueError(
                        f"query width {width} outside 1..{self.body.max_query_width}")
                return {"requested_width": width, "wrapper_prepared": True,
                        "wrapper_built_now": False, "launched": False,
                        "pipeline_compilation": "deferred_to_first_evaluation",
                        "variant": "single"}
            return self.body.prepare_width_wrapper(width)

    def prepare_width(self, width: int) -> dict[str, Any]:
        """Compatibility alias for wrapper preparation, not Metal compilation."""
        return self.prepare_width_wrapper(width)

    def _geometry_fields(self, width: int = 1) -> dict[str, int]:
        compiled_width = (
            self.body.narrow_width
            if self.dual_width and width <= self.body.narrow_width
            else self.body.wide_width if self.dual_width
            else self.body.max_query_width
        )
        return {
            "threads": int(self.body.threads),
            "groups": int(self.body.groups),
            "spin_cap": int(self.body.spin_cap),
            "compiled_width": int(compiled_width),
            "threadgroup_bytes": int(compute_tg_layout(compiled_width)[2]),
        }

    def _invoke_body(self, *args, position: int, width: int, **kwargs):
        try:
            return self.body(*args, **kwargs)
        except Exception as exc:
            reason = f"launch failed: {type(exc).__name__}"
            self._poisoned_reason = reason
            record_megakernel_receipt(
                engaged=False, reason=reason, aborted=True, width=width,
                position=position, context=position + width,
                launched="unknown", error=str(exc),
                **self._geometry_fields(width))
            raise MegakernelDeviceAbort(reason) from exc

    def _consume_launch(self, outs, *, position: int, width: int,
                        record: bool):
        """Synchronize the status word and publish only a completed launch."""
        status = outs[OUT["status"]]
        try:
            mx.eval(status)
            device_status = status.tolist()
            abort_count = int(device_status[0])
            device_phase = int(device_status[1])
        except Exception as exc:
            reason = f"unreadable device status: {type(exc).__name__}"
            self._poisoned_reason = reason
            record_megakernel_receipt(
                engaged=False, reason=reason, aborted=True, width=width,
                position=position, context=position + width, launched=True,
                **self._geometry_fields(width))
            raise MegakernelDeviceAbort(reason) from exc
        if abort_count:
            reason = f"device barrier abort ({abort_count})"
            self._poisoned_reason = reason
            record_megakernel_receipt(
                engaged=False, reason=reason, aborted=True, width=width,
                position=position, context=position + width, launched=True,
                device_phase=device_phase, abort_count=abort_count,
                variant=getattr(self.body, "last_variant", "single"),
                **self._geometry_fields(width))
            raise MegakernelDeviceAbort(reason)
        expected_phase = int(self.body.phase)
        if device_phase != expected_phase:
            reason = (
                f"device phase mismatch ({device_phase} != {expected_phase})"
            )
            self._poisoned_reason = reason
            record_megakernel_receipt(
                engaged=False, reason=reason, aborted=True, width=width,
                position=position, context=position + width, launched=True,
                device_phase=device_phase, expected_phase=expected_phase,
                variant=getattr(self.body, "last_variant", "single"),
                **self._geometry_fields(width))
            raise MegakernelDeviceAbort(reason)

        score_layout = score_tile_layout((position + width) // IDX_COMPRESS, width)
        if tuple(outs[OUT["score_tiles"]].shape) != (score_layout["elements"],):
            reason = "score output shape does not match launch geometry"
            self._poisoned_reason = reason
            record_megakernel_receipt(
                engaged=False, reason=reason, aborted=True, width=width,
                position=position, context=position + width, launched=True,
                **self._geometry_fields(width))
            raise MegakernelDeviceAbort(reason)

        with self._state_lock:
            if (not self._in_flight
                    or self._in_flight_owner != threading.get_ident()):
                self._poisoned_reason = "lost launch ownership"
                raise MegakernelDeviceAbort(self._poisoned_reason)
            self._pending = outs
            self._pending_width = width
            self._pending_position = position
            self._pending_owner = self._in_flight_owner
            self._in_flight = False
            self._in_flight_owner = None
        if record:
            record_megakernel_receipt(
                engaged=True, reason="engaged", phases=len(self.schedule),
                op_counts=self.op_counts, position=position, width=width,
                context=position + width, launched=True,
                score_layout=score_layout,
                device_phase=device_phase,
                device_barriers=self.device_barriers,
                variant=getattr(self.body, "last_variant", "single"),
                **self._geometry_fields(width))
        result_name = "logits" if self.include_lm_head else "out"
        return outs[OUT[result_name]]

    # ----------------------------------------------------------- the token
    def control(self, position: int, width: int = 1) -> mx.array:
        """The per-slab attention control block for this decoder's ledgers."""
        return mx.array(build_control_block(
            position, width, total=self.total,
            pooled_stride=self.pooled_stride,
            n_attn_layers=len(self.attn_layers),
            rope_theta=self.rope_theta))

    def step_slab(self, embeddings: mx.array, *, ple_embeddings=None,
                  position: Optional[int] = None, record: bool = True):
        """A width-M VERIFY SLAB: M embeddings in, M result rows out, ONE
        dispatch.

        This is the half of a k=2 self-MTP round the megakernel used to
        refuse.  `embeddings` is (M, HIDDEN) -- the draft token and its k
        proposals, already embedded on the host -- and the queries land at
        positions `position, position + 1, ... position + M - 1`. Results are
        logits when ``include_lm_head`` is true and final hidden states when it
        is false.

        **What the caller owns.**  The recurrent and conv state after the slab
        is the state after ALL M queries, because writing a restore point per
        query costs +227 MB of write traffic a token.  `cs_in`/`rec_in` are a
        separate buffer from the outputs, so the pre-slab state survives the
        launch untouched: on a partial accept the caller re-launches a
        narrower slab from it rather than restoring.  Commit the outputs only
        when every query was accepted.
        """
        shape = tuple(int(v) for v in embeddings.shape)
        width = shape[0] if len(shape) == 2 else -1
        validate_launch_shapes(
            shape, None if ple_embeddings is None else ple_embeddings.shape,
            width=width, has_ple=bool(self.ple_layers), hidden=HIDDEN)
        if embeddings.dtype != mx.bfloat16:
            raise ValueError(f"embedding dtype must be bfloat16; got {embeddings.dtype}")
        if ple_embeddings is not None and ple_embeddings.dtype != mx.bfloat16:
            raise ValueError(
                f"PLE embedding dtype must be bfloat16; got {ple_embeddings.dtype}")
        position = self.position if position is None else int(position)
        self._prepare_launch(position, width)
        try:
            decision = self.admit(width=width)
            if not decision.accepted:
                if record:
                    record_megakernel_receipt(
                        engaged=False, reason=decision.reason, width=width,
                        **self._geometry_fields(width))
                raise RuntimeError(f"megakernel declined: {decision.reason}")
            parts = []
            for m in range(width):
                streams = mx.tile(embeddings[m].reshape(-1), (HC_COUNT,))
                row = [streams, mx.zeros((HIDDEN,), mx.bfloat16)]
                if ple_embeddings is not None:
                    row.append(ple_embeddings[m].reshape(-1))
                parts.append(mx.concatenate(row))
            xin = mx.stack(parts)
            outs = self._invoke_body(
                xin, self.cs, self.rec, reps=1, kv=self.kv, idxl=self.idxl,
                pconv=self.pconv, actl=self.control(position, width),
                total=self.total, mwidth=width,
                score_blocks=(position + width) // IDX_COMPRESS,
                position=position, width=width)
            return self._consume_launch(
                outs, position=position, width=width, record=record)
        except Exception:
            self._release_launch_claim()
            raise

    def step(self, embedding: mx.array, *, ple_embedding=None,
             position: Optional[int] = None, record: bool = True):
        """One decode token: embedding in, one result row out, ONE dispatch.

        ``embedding`` is ``embed_tokens(token)`` -- the host lookup -- and
        ``ple_embedding`` the n-gram gather's row for the PLE layer.  Both are
        token lookups, not activations, which is exactly why they are the two
        things left on the host. The result is logits when ``include_lm_head``
        is true and the final hidden state otherwise.
        """
        validate_launch_shapes(
            embedding.shape,
            None if ple_embedding is None else ple_embedding.shape,
            width=1, has_ple=bool(self.ple_layers), hidden=HIDDEN)
        if embedding.dtype != mx.bfloat16:
            raise ValueError(f"embedding dtype must be bfloat16; got {embedding.dtype}")
        if ple_embedding is not None and ple_embedding.dtype != mx.bfloat16:
            raise ValueError(
                f"PLE embedding dtype must be bfloat16; got {ple_embedding.dtype}")
        position = self.position if position is None else int(position)
        self._prepare_launch(position, 1)
        try:
            decision = self.admit()
            if not decision.accepted:
                if record:
                    record_megakernel_receipt(
                        engaged=False, reason=decision.reason,
                        **self._geometry_fields())
                raise RuntimeError(f"megakernel declined: {decision.reason}")
            streams = mx.tile(embedding.reshape(-1), (HC_COUNT,))
            parts = [streams, mx.zeros((HIDDEN,), mx.bfloat16)]
            if ple_embedding is not None:
                parts.append(ple_embedding.reshape(-1))
            xin = mx.concatenate(parts)[None, :]
            outs = self._invoke_body(
                xin, self.cs, self.rec, reps=1, kv=self.kv, idxl=self.idxl,
                pconv=self.pconv, actl=self.control(position), position=position,
                total=self.total,
                score_blocks=(position + 1) // IDX_COMPRESS, width=1)
            return self._consume_launch(
                outs, position=position, width=1, record=record)
        except Exception:
            self._release_launch_claim()
            raise

    def commit(self) -> None:
        """Roll the GDN and conv states the last ``step`` produced.

        The recurrent and conv states are OUTPUTS, not in-place ledgers: the
        GDN core reads the previous state and writes the next one, so aliasing
        them would be a read-write hazard on the same address inside one phase.
        The attention ledgers are the opposite case and are written in place.
        """
        with self._state_lock:
            if self._pending is None or self._pending_position is None:
                raise RuntimeError("no megakernel launch is pending")
            if self._pending_owner != threading.get_ident():
                raise RuntimeError("pending megakernel transaction belongs to another thread")
            if self._pending_position != self.position:
                self._poisoned_reason = "pending position diverged from decoder cursor"
                raise MegakernelDeviceAbort(self._poisoned_reason)
            outs = self._pending
            self.cs = outs[OUT["cs_out"]]
            self.rec = outs[OUT["rec_out"]]
            self.pconv = outs[OUT["pconv_out"]]
            self.position = self._pending_position + self._pending_width
            self._pending = None
            self._pending_width = 0
            self._pending_position = None
            self._pending_owner = None

    def rollback(self) -> None:
        """Discard a slab whose queries were not all accepted.

        There is nothing to undo in the recurrent, GDN conv, or PLE conv state:
        each input is separate from its output, so the pre-slab state is still
        what this decoder points at until `commit` swaps them.  Declining to
        commit IS the rollback, and a partial accept re-launches a narrower
        slab from the same position.
        That is the contract the unified fused GDN verify kernel settled, and
        the reason no restore point is written: one per query would cost
        +227 MB of write traffic per token for a rollback that mostly does
        not happen.

        The KV and index ledgers are written IN PLACE, so a rejected slab
        leaves stale columns past the accepted position.  They are harmless:
        every read is bounded by the query's own `logical_len` and `n_valid`,
        both of which come from the position, and the next launch overwrites
        the same physical slots.
        """
        with self._state_lock:
            if self._pending is None:
                raise RuntimeError("no megakernel launch is pending")
            if self._pending_owner != threading.get_ident():
                raise RuntimeError("pending megakernel transaction belongs to another thread")
            self._pending = None
            self._pending_width = 0
            self._pending_position = None
            self._pending_owner = None

    @property
    def device_barriers(self) -> int:
        return self.schedule.device_barriers

    def status_fields(self) -> dict[str, Any]:
        return {
            "layers": len(self.layers),
            "attention_layers": len(self.attn_layers),
            "gdn_layers": len(self.gdn_layers),
            "ple_layers": len(self.ple_layers),
            "phases": len(self.schedule),
            "device_barriers": self.device_barriers,
            "op_counts": dict(self.op_counts),
            "threads": self.body.threads,
            "threadgroups": self.body.groups,
            "spin_cap": self.body.spin_cap,
            "kv_columns": self.total,
            "max_context": self.max_context,
            "position": self.position,
            "score_layout": dict(self.score_layout),
            "ledger_bytes": dict(self.ledger_bytes),
            "max_launch_bytes": dict(self.max_launch_bytes),
            "restore_source_bytes": self.restore_source_bytes,
            "pack_transient_budget": self.pack_transient_budget,
            "include_lm_head": self.include_lm_head,
            "result_kind": "logits" if self.include_lm_head else "hidden",
            "pending": self._pending is not None,
            "pending_position": self._pending_position,
            "pending_width": self._pending_width,
            "pending_owner": self._pending_owner,
            "in_flight": self._in_flight,
            "in_flight_owner": self._in_flight_owner,
            "source_may_be_mutated": self._source_may_be_mutated,
            "pack_estimate": dict(self.pack_estimate),
            "pack_memory_checks": list(self.pack_memory_checks),
            "poisoned_reason": self._poisoned_reason,
            "pack": self.pack.summary(),
            "dual_width": self.dual_width,
            "dual_width_receipt": (self.body.receipt() if self.dual_width
                                   else None),
        }


def _op_histogram(schedule) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in schedule.steps:
        name = OP_NAMES.get(step.op, f"OP_{step.op}")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))
