"""The megakernel as a SELECTABLE decode path: admission, ledgers, receipts.

Phases A and B built the pieces -- a weight pack with an offset table, a phase
schedule, and one persistent dispatch that walks it -- and proved each of them
against the stock arithmetic.  What none of them had was a caller.
``admit_megakernel_decode`` existed and failed closed, and nothing called it.

This module is that caller.  ``MegakernelDecoder`` owns one packed model, one
token schedule, and the ledgers a decode token reads and writes, and turns a
single token id into logits with ONE dispatch.  It is default OFF
(``MLX_QWEN4_MEGAKERNEL``), it admits or declines by a named reason before it
touches the GPU, and every launch leaves a receipt carrying the per-opcode
phase histogram, so "every phase engaged" is checkable from the receipt rather
than by reading the schedule.

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

import math
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from . import qwen4_megakernel_pack as MP
from . import qwen4_megakernel_schedule as MS
from .qwen4_megakernel_body import MegakernelBody, OUT_NAMES
from .qwen4_megakernel import (
    ACTL,
    ACTL_HEADER,
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
    N_KV_HEADS,
    OP_NAMES,
    PLE_STATE_LEN,
    SCRATCH,
    VOCAB,
    MegakernelAdmission,
    admit_megakernel_decode,
    record_megakernel_receipt,
)

OUT = {name: index for index, name in enumerate(OUT_NAMES)}


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
        max_group_bytes: int = 8 << 30,
        contiguous_source_sb: bool = True,
    ):
        self.args = args
        self.model = model
        self.layer_types = list(args.layer_types)
        self.ple_layer_ids = [int(v) for v in getattr(args, "ple_layer_ids", ())]
        self.layers = (list(range(len(self.layer_types))) if layers is None
                       else list(layers))
        self.rope_theta = float(args.rope_theta)
        # The KV ledger is allocated once at its widest, so a decode run never
        # reallocates mid-flight.  The block grid divides it exactly.
        self.total = int(math.ceil(max_context / IDX_COMPRESS) * IDX_COMPRESS)
        self.pooled_stride = self.total // IDX_COMPRESS

        self.attn_layers = [i for i in self.layers
                            if self.layer_types[i] != "linear_attention"]
        self.gdn_layers = [i for i in self.layers
                           if self.layer_types[i] == "linear_attention"]
        self.attn_slot = {i: r for r, i in enumerate(self.attn_layers)}
        self.ple_layers = [int(v) - 1 for v in self.ple_layer_ids
                           if int(v) - 1 in self.layers]

        plan = MP.decode_path_keys(
            num_layers=len(self.layer_types),
            layer_types=self.layer_types,
            ple_layer_ids=self.ple_layer_ids,
            layers=self.layers,
            include_mtp=include_mtp,
            include_experts=include_experts,
        )
        # 8 GiB is a HARD cap, not a preference: a group is one flat uint32
        # buffer, MLX shape dimensions are int32, and 2^31 words is 8 GiB.
        # Raising it to shrink the buffer count therefore is not available, so
        # the kernel carries ten weight bindings instead -- 8 expert groups
        # plus `main` and `lm_head`.
        self.source = MP.ModuleSource(model)
        self.pack = MP.build_pack(
            self.source, plan, validate=validate, rebind=rebind,
            max_group_bytes=max_group_bytes)
        if rebind and contiguous_source_sb:
            self.restored_sb = self.restore_source_contiguity()
        self.schedule = MS.build_token_schedule(
            self.pack, layer_types=self.layer_types, layers=self.layers,
            ple_layer_ids=self.ple_layer_ids, include_lm_head=True)
        self.op_counts = _op_histogram(self.schedule)
        self.body = MegakernelBody(
            self.pack, self.schedule, gdn_layers=max(len(self.gdn_layers), 1),
            vocab=VOCAB, threads=threads, groups=groups)
        self._allocate()
        self.position = 0

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
        for key, entry in self.pack.entries.items():
            if entry.n_sb == 0 or not self.source.has(key):
                continue
            module = self.source._resolve(key)
            for part in ("scales", "biases"):
                value = getattr(module, part, None)
                if isinstance(value, mx.array):
                    setattr(module, part, mx.contiguous(value))
            restored += 1
        return restored

    # ------------------------------------------------------------- ledgers
    def _allocate(self) -> None:
        n_attn = max(len(self.attn_layers), 1)
        n_gdn = max(len(self.gdn_layers), 1)
        self.kbuf = mx.zeros(
            (n_attn * N_KV_HEADS, self.total, HEAD_DIM), mx.bfloat16)
        self.vbuf = mx.zeros(
            (n_attn * N_KV_HEADS, self.total, HEAD_DIM), mx.bfloat16)
        self.rawk = mx.zeros(
            (n_attn * self.total, IDX_HEAD_DIM), mx.bfloat16)
        self.pooled = mx.zeros(
            (n_attn * self.pooled_stride, IDX_HEAD_DIM), mx.bfloat16)
        self.cs = mx.zeros((n_gdn, CONV_KERNEL - 1, CONV_DIM), mx.bfloat16)
        self.rec = mx.zeros(
            (n_gdn, GDN_VALUE_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM), mx.float32)
        self.pconv = mx.zeros((PLE_STATE_LEN, HC_HIDDEN), mx.bfloat16)
        mx.eval(self.kbuf, self.vbuf, self.rawk, self.pooled,
                self.cs, self.rec, self.pconv)

    def seed_from_caches(self, caches, *, indexer_of=None) -> dict[str, Any]:
        """Fill the ledgers from a stock prefill.

        ``caches`` is the list ``TextModel.make_cache`` returns, after a stock
        forward.  Everything copied here is a value the stock path produced, so
        a megakernel decode continuing from it is continuing the same sequence
        -- which is what makes the interleaved gate a comparison of the DECODE
        step rather than of two different histories.
        """
        report = {"attention": 0, "gdn": 0, "ple": 0, "pooled_blocks": 0}
        for slot, index in enumerate(self.attn_layers):
            cache = caches[index]
            keys, values = cache.keys, cache.values
            length = int(cache.offset)
            assert length <= self.total, (
                f"prefill {length} over the ledger's {self.total} columns")
            base = slot * N_KV_HEADS
            self.kbuf[base: base + N_KV_HEADS, :length] = keys[0, :, :length]
            self.vbuf[base: base + N_KV_HEADS, :length] = values[0, :, :length]
            raw = cache.index_keys
            rbase = slot * self.total
            self.rawk[rbase: rbase + length] = raw[0, :length]
            n_blocks = length // IDX_COMPRESS
            if n_blocks and indexer_of is not None:
                starts = mx.arange(n_blocks) * IDX_COMPRESS
                pooled = indexer_of(index)._pool_blocks(
                    raw[:, : n_blocks * IDX_COMPRESS], starts)[0]
                pbase = slot * self.pooled_stride
                self.pooled[pbase: pbase + n_blocks] = pooled.astype(
                    mx.bfloat16)
                report["pooled_blocks"] += n_blocks
            report["attention"] += 1
            self.position = length
        for slot, index in enumerate(self.gdn_layers):
            cache = caches[index]
            if cache[0] is not None:
                self.cs[slot] = cache[0][0].astype(mx.bfloat16)
            if cache[1] is not None:
                self.rec[slot] = cache[1][0].astype(mx.float32)
            report["gdn"] += 1
        for index in self.ple_layers:
            state = caches[index][2]
            if state is not None:
                self.pconv[:] = state[0].astype(mx.bfloat16)
            report["ple"] += 1
        mx.eval(self.kbuf, self.vbuf, self.rawk, self.pooled,
                self.cs, self.rec, self.pconv)
        return report

    # ------------------------------------------------------------ admission
    def admit(self, *, width: int = 1, batch: int = 1, dtype=mx.bfloat16,
              speculating: bool = False, training: bool = False,
              sharded: bool = False, mask=None) -> MegakernelAdmission:
        return admit_megakernel_decode(
            width=width, batch=batch, pack=self.pack, schedule=self.schedule,
            speculating=speculating, training=training, sharded=sharded,
            mask=mask, dtype=dtype, threads=self.body.threads,
            groups=self.body.groups,
            layer_types=[self.layer_types[i] for i in self.layers],
        )

    # ----------------------------------------------------------- the token
    def control(self, position: int) -> mx.array:
        """The per-token attention control block.

        Everything in it except the per-layer ledger bases is shared by all
        twelve attention layers, so the layer index lives in the schedule and
        this stays one array per token.
        """
        length = position + 1
        n_blocks = length // IDX_COMPRESS
        budget = min(BLOCK_TOPK, n_blocks)
        actl = np.zeros(ACTL_HEADER + BLOCK_TOPK, np.uint32)
        actl[ACTL["total"]] = self.total
        actl[ACTL["u_width"]] = budget
        actl[ACTL["count"]] = budget
        actl[ACTL["n_sel"]] = budget
        actl[ACTL["q_pos"]] = position
        actl[ACTL["block_size"]] = IDX_COMPRESS
        actl[ACTL["ids_from_scratch"]] = 1
        actl[ACTL["scale_bits"]] = np.float32(
            HEAD_DIM ** -0.5).view(np.uint32)
        actl[ACTL["logical_len"]] = length
        actl[ACTL["n_blocks"]] = n_blocks
        # Every block the grid names is causally valid at this query: a block
        # ends at 4b+3 and the deepest query is at `position` = length - 1,
        # and n_blocks = length // 4 already excludes the open tail block.
        actl[ACTL["n_valid"]] = n_blocks
        actl[ACTL["rope_pos"]] = position
        actl[ACTL["rope_theta_bits"]] = np.float32(
            self.rope_theta).view(np.uint32)
        actl[ACTL["kv_slot"]] = position
        actl[ACTL["pooled_stride"]] = self.pooled_stride
        actl[ACTL["pool_new_block"]] = int(length % IDX_COMPRESS == 0)
        actl[ACTL["n_attn_layers"]] = len(self.attn_layers)
        return mx.array(actl)

    def step(self, embedding: mx.array, *, ple_embedding=None,
             position: Optional[int] = None, record: bool = True):
        """One decode token: embedding in, logits out, ONE dispatch.

        ``embedding`` is ``embed_tokens(token)`` -- the host lookup -- and
        ``ple_embedding`` the n-gram gather's row for the PLE layer.  Both are
        token lookups, not activations, which is exactly why they are the two
        things left on the host.
        """
        decision = self.admit()
        if not decision.accepted:
            if record:
                record_megakernel_receipt(engaged=False,
                                          reason=decision.reason)
            raise RuntimeError(f"megakernel declined: {decision.reason}")
        position = self.position if position is None else int(position)
        # The residual streams are the embedding tiled over the hyper count,
        # which is what Qwen4ExpTextModel.__call__ does before layer 0.
        streams = mx.tile(embedding.reshape(-1), (HC_COUNT,))
        parts = [streams.astype(mx.bfloat16), mx.zeros((HIDDEN,), mx.bfloat16)]
        if ple_embedding is not None:
            parts.append(ple_embedding.reshape(-1).astype(mx.bfloat16))
        xin = mx.concatenate(parts)[None, :]
        outs = self.body(
            xin, self.cs, self.rec, reps=1, kbuf=self.kbuf, vbuf=self.vbuf,
            pooled=self.pooled, rawk=self.rawk, pconv=self.pconv,
            actl=self.control(position))
        logits = outs[OUT["logits"]]
        self._pending = outs
        if record:
            record_megakernel_receipt(
                engaged=True, reason="engaged", phases=len(self.schedule),
                op_counts=self.op_counts, position=position,
                context=position + 1, device_barriers=self.device_barriers)
        return logits

    def commit(self) -> None:
        """Roll the GDN and conv states the last ``step`` produced.

        The recurrent and conv states are OUTPUTS, not in-place ledgers: the
        GDN core reads the previous state and writes the next one, so aliasing
        them would be a read-write hazard on the same address inside one phase.
        The attention ledgers are the opposite case and are written in place.
        """
        outs = self._pending
        self.cs = outs[OUT["cs_out"]]
        self.rec = outs[OUT["rec_out"]]
        self.position += 1

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
            "kv_columns": self.total,
            "pack": self.pack.summary(),
        }


def _op_histogram(schedule) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in schedule.steps:
        name = OP_NAMES.get(step.op, f"OP_{step.op}")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))
