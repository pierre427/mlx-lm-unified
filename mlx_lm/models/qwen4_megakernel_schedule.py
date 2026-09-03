"""The per-token phase schedule, and the CPU mirror that defines each phase.

Two things live here, deliberately together.

``build_token_schedule`` emits the ordered list of phases one decode token
runs -- the thing the kernel's dispatcher walks -- from the weight pack's
offset table.  ``MirrorExecutor`` executes that same list with MLX ops.  The
mirror is not a fallback and is never shipped: it is the *definition* of what
each opcode means, in a form a test can compare against the stock module
before any Metal exists, and against the kernel once it does.

**Which stock op each phase matches.**  The tolerance class of the whole
kernel is decided at these boundaries, so each one is named:

* ``OP_GROUP_RMSNORM`` -- ``GroupRMSNorm``: ONE fp32 upcast, a per-group
  ``rsqrt(mean(x*x) + eps)``, the per-group weight applied in fp32, and the
  result cast back to the activation dtype.  The stock module may take
  ``mx.fast.rms_norm`` at decode widths (an accepted class-3 reorder); the
  kernel matches the *eager* fp32 form, which is the arithmetic both agree on.
* ``OP_HC_MIX`` -- ``GatedResidual``: ``silu(down / hc_count)``, then the up
  projection, then ``sigmoid``, then ``mean(weights * streams, axis=-2)``.
  The stock chain runs all of this in **bfloat16** (``_run_glue`` refuses any
  span whose operands are not bf16, and the compiled span was measured to
  match eager over all 65,280 finite bf16 values), so the kernel's fp32 form
  is class 2 -- closer to fp32 than stock -- not bit-identical.  The
  ``sigmoid`` is deliberately the PRECISE one: the stock module keeps it eager
  because the fused Metal variant disagrees at x = -6.85, so the kernel uses
  ``metal::precise::exp``, not the fast intrinsic.
* ``OP_INJECT`` -- ``_apply_inject``: ``residual + branch * inject`` broadcast
  over the H streams, bf16 in stock, fp32 here.
* ``OP_QMV`` -- ``mx.quantized_matmul(transpose=True)``.  Dequant-on-the-fly
  with an fp32 accumulator; stock accumulates in the activation dtype.
* ``OP_GDN_CORE`` -- the conv step, SiLU, L2 normalisation, gated delta rule
  and gated RMS norm of ``GatedDeltaNet``, ported from ``qwen4_fused_gdn``.
* ``OP_MOE_TOPK`` -- ``mx.softmax(precise=True)`` over the fp32 router logits
  then top-k; the spike measured the kernel's top-10 agreeing with the fp32
  router more often than the stock bf16 path does (7/8 vs 6/8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx

from .qwen4_megakernel import (
    BAR_DEVICE,
    BAR_NONE,
    BAR_THREADGROUP,
    BLOCK_TOPK,
    CONV_DIM,
    CONV_KERNEL,
    FF,
    GDN_KEY_DIM,
    GDN_KEY_HEADS,
    GDN_RATIO,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HC_COUNT,
    HC_HIDDEN,
    HC_LOWRANK,
    HIDDEN,
    KEY_DIM,
    NUM_EXPERTS,
    OP_ADD,
    OP_ATTN,
    OP_COPY,
    OP_GDN_CORE,
    OP_GROUP_RMSNORM,
    OP_HC_MIX,
    OP_INDEX_TOPB,
    OP_INJECT,
    OP_MOE_E1,
    OP_MOE_E2,
    OP_MOE_TOPK,
    OP_NOP,
    OP_QMV,
    OP_RMSNORM,
    OP_SILU_MUL,
    RMS_EPS,
    SCRATCH,
    SCRATCH_FLOATS,
    TOPK,
    VALUE_DIM,
    Schedule,
    Step,
)

# Post-activations an OP_QMV may apply to its own output, in ``arg1``.
ACT_NONE = 0
ACT_SILU = 1
ACT_SIGMOID = 2
ACT_SILU_SCALED = 3     # silu(x / hc_count) -- the hyper gate
ACT_INJECT_GATE = 4     # 2 * sigmoid(x / hc_count) -- the block inject

# ``OP_QMV.arg2``: where the result goes.
DST_SCRATCH = 0
DST_OUT = 1
# Every threadgroup computes the WHOLE output into its own threadgroup memory.
# Costs G times the weight reads, so it is only for tiny projections, and buys
# a device barrier -- the same trade the spike made for its top-k phase, which
# profiled at ~0.00 ms.  ``block_inject_weight`` is 4 x 10,240 at 4 bits
# (20 KiB) and ``shared_expert_gate`` is 1 x 2,560 (1.3 KiB); replicated over
# 80 threadgroups that is 1.6 MiB and 100 KiB against a 5.2 us barrier.
DST_REPLICATED = 2


@dataclass
class LayerPlan:
    """Where a layer's weights live, by pack key."""

    index: int
    is_linear: bool
    prefix: str

    def key(self, suffix: str) -> str:
        return f"{self.prefix}.{suffix}"


def _entry_id(pack, key: str) -> int:
    return pack.entries[key].index


def _hyper_block(
    schedule: Schedule, pack, plan: LayerPlan, which: str, resid: int
) -> None:
    """The five phases of one ``GatedResidual``.

    ``resid`` names which residual slab holds the H streams on entry.  The
    norm reads it and the inject writes the OTHER slab: the spike's U2 result
    is that a reused scratch address is stale without a device fence and this
    kernel reuses every address 48 times, so a phase that both reads and
    writes the streams ping-pongs rather than trusting an in-place update.
    """
    hyper = plan.key(which)
    schedule.add(Step(
        op=OP_GROUP_RMSNORM,
        entry=_entry_id(pack, f"{hyper}.hc_norm.weight"),
        src=resid, dst=SCRATCH["NORMED"],
        arg0=HC_HIDDEN, arg1=HIDDEN,      # dim, group size
        barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{hyper}.input_mix_weight_down"),
        src=SCRATCH["NORMED"], dst=SCRATCH["HC_LR"], arg1=ACT_SILU_SCALED,
        barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{hyper}.input_mix_weight_up"),
        src=SCRATCH["HC_LR"], dst=SCRATCH["HC_W"], arg1=ACT_SIGMOID,
        barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_HC_MIX, src=SCRATCH["HC_W"], dst=SCRATCH["MIXED"],
        arg0=HC_COUNT, arg1=HIDDEN, barrier=BAR_DEVICE,
    ))
    if pack.entries.get(f"{hyper}.block_inject_weight") is not None:
        schedule.add(Step(
            op=OP_QMV, entry=_entry_id(pack, f"{hyper}.block_inject_weight"),
            src=SCRATCH["NORMED"], dst=SCRATCH["INJECT"],
            arg1=ACT_INJECT_GATE, arg2=DST_REPLICATED,
            barrier=BAR_THREADGROUP,
        ))


def _gdn_branch(schedule: Schedule, pack, plan: LayerPlan) -> None:
    attn = plan.key("linear_attn")
    for name, dst, width in (
        ("in_proj_qkv", SCRATCH["GDN_QKV"], CONV_DIM),
        ("in_proj_z", SCRATCH["GDN_Z"], VALUE_DIM),
        ("in_proj_b", SCRATCH["GDN_BA"], GDN_VALUE_HEADS),
        ("in_proj_a", SCRATCH["GDN_BA"] + GDN_VALUE_HEADS, GDN_VALUE_HEADS),
    ):
        schedule.add(Step(
            op=OP_QMV, entry=_entry_id(pack, f"{attn}.{name}"),
            src=SCRATCH["MIXED"], dst=dst, arg0=width,
            # the four input projections are independent; only the last needs
            # to publish before the core reads them
            barrier=BAR_NONE if name != "in_proj_a" else BAR_DEVICE,
        ))
    schedule.add(Step(
        op=OP_GDN_CORE, entry=_entry_id(pack, f"{attn}.conv1d.weight"),
        src=SCRATCH["GDN_QKV"], dst=SCRATCH["GDN_Y"], barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{attn}.out_proj"),
        src=SCRATCH["GDN_Y"], dst=SCRATCH["BRANCH"], arg0=HIDDEN,
        barrier=BAR_DEVICE,
    ))


def _attention_branch(schedule: Schedule, pack, plan: LayerPlan) -> None:
    attn = plan.key("self_attn")
    for name, dst in (
        ("q_proj", SCRATCH["ATT_QG"]),
        ("k_proj", SCRATCH["ATT_K"]),
        ("v_proj", SCRATCH["ATT_V"]),
    ):
        schedule.add(Step(
            op=OP_QMV, entry=_entry_id(pack, f"{attn}.{name}"),
            src=SCRATCH["MIXED"], dst=dst,
            barrier=BAR_NONE if name != "v_proj" else BAR_DEVICE,
        ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{attn}.indexer.index_qk_proj"),
        src=SCRATCH["MIXED"], dst=SCRATCH["IDX_QK"], barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_INDEX_TOPB, src=SCRATCH["IDX_SCORE"], dst=SCRATCH["IDX_SEL"],
        arg0=BLOCK_TOPK,
        # the selector is one threadgroup's work end to end, so the boundary
        # inside it is a threadgroup barrier; only its result crosses the grid
        barrier=BAR_DEVICE,
    ))
    schedule.add(Step(op=OP_ATTN, src=SCRATCH["ATT_QG"], dst=SCRATCH["ATT_O"],
                      barrier=BAR_DEVICE))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{attn}.o_proj"),
        src=SCRATCH["ATT_O"], dst=SCRATCH["BRANCH"], arg0=HIDDEN,
        barrier=BAR_DEVICE,
    ))


def _moe_branch(schedule: Schedule, pack, plan: LayerPlan) -> None:
    mlp = plan.key("mlp")
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.gate"),
        src=SCRATCH["MIXED"], dst=SCRATCH["MOE_LOGITS"], barrier=BAR_DEVICE,
    ))
    # Recomputed per threadgroup, so it costs no grid barrier -- the spike's
    # phase 5, which profiled at ~0.00 ms.
    schedule.add(Step(
        op=OP_MOE_TOPK, src=SCRATCH["MOE_LOGITS"], dst=SCRATCH["MOE_TOPI"],
        arg0=TOPK, arg1=NUM_EXPERTS, barrier=BAR_THREADGROUP,
    ))
    # One fused [E, 2*FF, HID] table: gate rows then up rows, which is the
    # layout ``transform_moe_weights`` already leaves resident, so this phase
    # streams one table instead of two and needs no concatenation at pack time.
    schedule.add(Step(
        op=OP_MOE_E1, entry=_entry_id(pack, f"{mlp}.switch_mlp.gate_up_proj"),
        src=SCRATCH["MIXED"], dst=SCRATCH["MOE_ACT"], arg1=FF,
        barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_MOE_E2, entry=_entry_id(pack, f"{mlp}.switch_mlp.down_proj"),
        src=SCRATCH["MOE_ACT"], dst=SCRATCH["BRANCH"], arg0=HIDDEN,
        barrier=BAR_DEVICE,
    ))
    # shared expert, added into the same branch slot
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.shared_expert.gate_proj"),
        src=SCRATCH["MIXED"], dst=SCRATCH["SHARED_ACT"], arg1=ACT_SILU,
        barrier=BAR_NONE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.shared_expert.up_proj"),
        src=SCRATCH["MIXED"], dst=SCRATCH["SHARED_UP"], barrier=BAR_NONE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.shared_expert_gate"),
        src=SCRATCH["MIXED"], dst=SCRATCH["SHARED_GATE"], arg0=1,
        arg1=ACT_SIGMOID, arg2=DST_REPLICATED, barrier=BAR_THREADGROUP,
    ))
    schedule.add(Step(
        op=OP_SILU_MUL, src=SCRATCH["SHARED_ACT"], dst=SCRATCH["SHARED_ACT"],
        arg0=SCRATCH["SHARED_UP"], arg1=FF, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.shared_expert.down_proj"),
        src=SCRATCH["SHARED_ACT"], dst=SCRATCH["SHARED_OUT"],
        arg0=HIDDEN, arg1=ACT_NONE, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_ADD, src=SCRATCH["SHARED_OUT"], dst=SCRATCH["BRANCH"],
        arg0=HIDDEN, arg1=SCRATCH["SHARED_GATE"], barrier=BAR_DEVICE,
    ))


def build_layer_schedule(
    schedule: Schedule, pack, plan: LayerPlan, resid: int, other: int
) -> int:
    """Append one decoder layer.  Returns the slab the output landed in."""
    _hyper_block(schedule, pack, plan, "attn_hyper_connection", resid)
    if plan.is_linear:
        _gdn_branch(schedule, pack, plan)
    else:
        _attention_branch(schedule, pack, plan)
    schedule.add(Step(
        op=OP_INJECT, src=resid, dst=other, arg0=SCRATCH["BRANCH"],
        arg1=SCRATCH["INJECT"], arg2=HC_COUNT, barrier=BAR_DEVICE,
    ))
    resid, other = other, resid
    _hyper_block(schedule, pack, plan, "mlp_hyper_connection", resid)
    _moe_branch(schedule, pack, plan)
    schedule.add(Step(
        op=OP_INJECT, src=resid, dst=other, arg0=SCRATCH["BRANCH"],
        arg1=SCRATCH["INJECT"], arg2=HC_COUNT, barrier=BAR_DEVICE,
    ))
    return other


def build_token_schedule(
    pack,
    *,
    layer_types: list[str],
    layers: Optional[list[int]] = None,
    prefix: str = "language_model.model.layers",
    include_lm_head: bool = True,
    mixer: str = "language_model.model.hyper_connection_mixer",
) -> Schedule:
    """The whole per-token phase sequence, hyper-connections included."""
    schedule = Schedule()
    resid, other = SCRATCH["RESID_A"], SCRATCH["RESID_B"]
    indices = range(len(layer_types)) if layers is None else layers
    for index in indices:
        plan = LayerPlan(
            index=index,
            is_linear=layer_types[index] == "linear_attention",
            prefix=f"{prefix}.{index}",
        )
        landed = build_layer_schedule(schedule, pack, plan, resid, other)
        resid, other = landed, (resid if landed == other else other)
    # final mixer: same GatedResidual, without the inject combine
    tail = LayerPlan(index=-1, is_linear=True, prefix=mixer.rsplit(".", 1)[0])
    _hyper_block(
        schedule, pack,
        LayerPlan(index=-1, is_linear=True, prefix=mixer.rsplit(".", 1)[0]),
        mixer.rsplit(".", 1)[1], resid,
    )
    if include_lm_head:
        schedule.add(Step(
            op=OP_QMV, entry=_entry_id(pack, "language_model.lm_head"),
            src=SCRATCH["MIXED"], dst=0, arg0=0, arg2=DST_OUT,
            barrier=BAR_NONE,
        ))
    return schedule


# ------------------------------------------------------------------- mirror
class MirrorExecutor:
    """Runs a schedule with MLX ops.  Defines what each opcode means.

    Everything is fp32, which is what the kernel holds.  The stock chain is
    bf16 across the hyper-connection glue and the projections, so the mirror is
    expected to be CLOSER to an fp32 reference than stock, not equal to it --
    class 2 in ``wiki/docs/lessons/exactness-tolerance-classes.md``.
    """

    def __init__(self, pack, *, state: Optional[dict] = None):
        self.pack = pack
        self.scratch = mx.zeros((SCRATCH_FLOATS,), mx.float32)
        self.out: Optional[mx.array] = None
        self.state = state or {}
        self.trace: list[str] = []
        self._weights: dict[str, dict[str, mx.array]] = {}
        self._by_index = {
            entry.index: key for key, entry in pack.entries.items()
        }

    # -- scratch helpers.  MLX arrays are immutable, so a write is a scatter.
    def read(self, offset: int, width: int) -> mx.array:
        return self.scratch[offset: offset + width]

    def write(self, offset: int, values: mx.array) -> None:
        # MLX arrays are immutable, so a scratch write rebuilds the slab.  The
        # mirror is a definition, not a fast path; the kernel writes in place.
        values = values.reshape(-1).astype(mx.float32)
        end = offset + values.size
        if end > SCRATCH_FLOATS:
            raise ValueError(
                f"scratch write of {values.size} at {offset} overruns "
                f"{SCRATCH_FLOATS}"
            )
        self.scratch = mx.concatenate(
            [self.scratch[:offset], values, self.scratch[end:]]
        )

    def weights(self, entry_index: int) -> dict[str, mx.array]:
        key = self._by_index[entry_index]
        if key not in self._weights:
            self._weights[key] = self.pack.views(key)
        return self._weights[key]

    def entry(self, entry_index: int):
        return self.pack.entries[self._by_index[entry_index]]

    # -- ops
    def run(self, schedule: Schedule) -> None:
        for step in schedule.steps:
            handler = getattr(self, f"_op_{step.op}", None)
            if handler is None:
                raise NotImplementedError(
                    f"mirror has no body for opcode {step.op}"
                )
            self.trace.append(f"{step.op}:{step.entry}")
            handler(step)

    def _op_0(self, step: Step) -> None:  # OP_NOP
        return None

    def _op_1(self, step: Step) -> None:  # OP_QMV
        entry = self.entry(step.entry)
        parts = self.weights(step.entry)
        x = self.read(step.src, entry.cols)
        out = mx.quantized_matmul(
            x, parts["weight"], parts["scales"], parts["biases"],
            transpose=True, group_size=entry.group_size, bits=entry.bits,
        )
        out = _activate(out, step.arg1)
        if step.arg2 == DST_OUT:
            self.out = out
        else:
            # DST_REPLICATED differs from DST_SCRATCH only in the kernel, where
            # the value lands in every threadgroup's own memory rather than in
            # device scratch.  The mirror has one address space, so the two are
            # the same write and the same arithmetic.
            self.write(step.dst, out)

    def _op_3(self, step: Step) -> None:  # OP_RMSNORM
        parts = self.weights(step.entry)
        width = step.arg0
        x = self.read(step.src, width)
        scale = mx.rsqrt(mx.mean(x * x) + RMS_EPS)
        self.write(step.dst, x * scale * parts["weight"].astype(mx.float32))

    def _op_4(self, step: Step) -> None:  # OP_GROUP_RMSNORM
        parts = self.weights(step.entry)
        dim, group = step.arg0, step.arg1
        x = self.read(step.src, dim).reshape(dim // group, group)
        scale = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + RMS_EPS)
        out = (x * scale).reshape(dim) * parts["weight"].astype(mx.float32)
        self.write(step.dst, out)

    def _op_5(self, step: Step) -> None:  # OP_HC_MIX
        count, width = step.arg0, step.arg1
        weights = self.read(step.src, count * width).reshape(count, width)
        streams = self.read(SCRATCH["NORMED"], count * width).reshape(count, width)
        self.write(step.dst, mx.mean(weights * streams, axis=0))

    def _op_6(self, step: Step) -> None:  # OP_INJECT
        count = step.arg2
        residual = self.read(step.src, count * HIDDEN).reshape(count, HIDDEN)
        branch = self.read(step.arg0, HIDDEN)
        inject = self.read(step.arg1, count)
        self.write(step.dst, (residual + branch[None, :] * inject[:, None]))

    def _op_13(self, step: Step) -> None:  # OP_COPY
        self.write(step.dst, self.read(step.src, step.arg0))

    def _op_14(self, step: Step) -> None:  # OP_ADD
        width = step.arg0
        gate = self.read(step.arg1, 1)
        self.write(
            step.dst, self.read(step.dst, width) + gate * self.read(step.src, width)
        )

    def _op_15(self, step: Step) -> None:  # OP_SILU_MUL
        width = step.arg1
        left = self.read(step.src, width)
        right = self.read(step.arg0, width)
        self.write(step.dst, left * right)

    def _op_8(self, step: Step) -> None:  # OP_MOE_TOPK
        k, experts = step.arg0, step.arg1
        logits = self.read(step.src, experts)
        probs = mx.softmax(logits, axis=-1, precise=True)
        order = mx.argpartition(probs, kth=experts - k)[-k:]
        weights = mx.take(probs, order)
        weights = weights / weights.sum()
        self.write(step.dst, order.astype(mx.float32))
        self.write(SCRATCH["MOE_TOPW"], weights)

    def _op_9(self, step: Step) -> None:  # OP_MOE_E1
        table = self.weights(step.entry)
        entry = self.entry(step.entry)
        width = step.arg1
        chosen = self.read(SCRATCH["MOE_TOPI"], TOPK).astype(mx.uint32)
        x = self.read(step.src, entry.cols)[None, None, :]
        fused = mx.gather_qmm(
            x, table["weight"], table["scales"], table["biases"],
            rhs_indices=chosen, transpose=True,
            group_size=entry.group_size, bits=entry.bits,
        ).reshape(TOPK, 2 * width)
        gate, up = fused[:, :width], fused[:, width:]
        self.write(step.dst, (gate * mx.sigmoid(gate) * up).reshape(-1))

    def _op_10(self, step: Step) -> None:  # OP_MOE_E2
        down = self.weights(step.entry)
        entry = self.entry(step.entry)
        chosen = self.read(SCRATCH["MOE_TOPI"], TOPK).astype(mx.uint32)
        # ``gather_qmm`` reads x as [...batch, M, K] and broadcasts
        # ``rhs_indices`` against the BATCH dims, so the expert axis has to be
        # a batch axis and M stays 1.  A [TOPK, FF] input would pair every
        # expert with every row and return TOPK^2 of them.
        act = self.read(step.src, TOPK * FF).reshape(TOPK, 1, FF)
        out = mx.gather_qmm(
            act, down["weight"], down["scales"], down["biases"],
            rhs_indices=chosen, transpose=True,
            group_size=entry.group_size, bits=entry.bits,
        ).reshape(TOPK, HIDDEN)
        weights = self.read(SCRATCH["MOE_TOPW"], TOPK)
        self.write(step.dst, (out * weights[:, None]).sum(0))


def _activate(x: mx.array, kind: int) -> mx.array:
    if kind == ACT_NONE:
        return x
    if kind == ACT_SILU:
        return x * mx.sigmoid(x)
    if kind == ACT_SIGMOID:
        return mx.sigmoid(x)
    if kind == ACT_SILU_SCALED:
        scaled = x / HC_COUNT
        return scaled * mx.sigmoid(scaled)
    if kind == ACT_INJECT_GATE:
        return 2 * mx.sigmoid(x / HC_COUNT)
    raise ValueError(f"unknown activation {kind}")
