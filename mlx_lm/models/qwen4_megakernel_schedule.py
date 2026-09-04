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
    HEAD_DIM,
    HIDDEN,
    IDX_COMPRESS,
    IDX_HEADS,
    IDX_HEAD_DIM,
    KEY_DIM,
    NUM_EXPERTS,
    N_KV_HEADS,
    N_Q_HEADS,
    OP_ADD,
    OP_ADD_BCAST,
    OP_ATTN,
    OP_ATTN_COMBINE,
    OP_COPY,
    OP_GATE_MUL,
    OP_GDN_CORE,
    OP_GROUP_RMSNORM,
    OP_HC_DOWN,
    OP_HC_MIX,
    OP_HC_UP,
    OP_INDEX_SCORE,
    OP_INDEX_TOPB,
    OP_INJECT,
    OP_KV_APPEND,
    OP_MOE_E1,
    OP_MOE_E2,
    OP_MOE_TOPK,
    OP_NOP,
    OP_PLE_CONV,
    OP_PLE_GATE,
    OP_POOL_BLOCK,
    OP_QK_NORM_ROPE,
    OP_QMV,
    OP_RMSNORM,
    OP_SILU_MUL,
    PHASE_ROWS,
    PLE_CONV_KERNEL,
    PLE_NGRAM,
    PLE_STATE_LEN,
    RMS_EPS,
    ROTARY_DIM,
    SCRATCH,
    SCRATCH_FLOATS,
    Schedule,
    Step,
    TOPK,
    VALUE_DIM,
)

# ``OP_QMV.arg0``: rows per simdgroup.  The spec tuned a different value for
# almost every projection and a data-driven dispatcher cannot specialise per
# call site, so the tuned number travels with the phase.  1, 2 and 4 are the
# values the kernel offers; 8 is deliberately absent because the spec measured
# it spilling (217 GB/s against 340 at 2 rows on the GDN input projection).
R_GENERIC = PHASE_ROWS["generic_qmv"]

# ``entry`` / entry-carrying argument words use this for "no such tensor".
NO_ENTRY = 0xFFFFFFFF

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
    schedule: Schedule, pack, plan: LayerPlan, which: str, resid: int,
    *, fused: bool = True,
) -> None:
    """One ``GatedResidual``, as the phases the kernel actually runs.

    ``resid`` names which residual slab holds the H streams on entry.  The
    norm reads it and the inject writes the OTHER slab: the spike's U2 result
    is that a reused scratch address is stale without a device fence and this
    kernel reuses every address 48 times, so a phase that both reads and
    writes the streams ping-pongs rather than trusting an in-place update.

    **Fused is the shipped form and it is what was measured.**  Phase A's
    prototype -- the only timing anyone has for this block -- is THREE phases:
    the group norm publishes the normed vector, then one phase does the
    10,240 -> 320 mix-down *and* the 4-row inject gate from that same vector
    (both K-split across the simdgroups, worth 1.37x here), then one phase
    does the up-mix d-major with the mean folded in.  The five-phase spelling
    below is the same arithmetic in separate steps; it is kept because it is
    what the mirror was first written against, and because a barrier-cost
    experiment wants both spellings.
    """
    hyper = plan.key(which)
    inject_key = f"{hyper}.block_inject_weight"
    has_inject = pack.entries.get(inject_key) is not None
    schedule.add(Step(
        op=OP_GROUP_RMSNORM,
        entry=_entry_id(pack, f"{hyper}.hc_norm.weight"),
        src=resid, dst=SCRATCH["NORMED"],
        arg0=HC_HIDDEN, arg1=HIDDEN,      # dim, group size
        barrier=BAR_DEVICE,
    ))
    if fused:
        schedule.add(Step(
            op=OP_HC_DOWN,
            entry=_entry_id(pack, f"{hyper}.input_mix_weight_down"),
            src=SCRATCH["NORMED"], dst=SCRATCH["HC_LR"],
            arg0=_entry_id(pack, inject_key) if has_inject else NO_ENTRY,
            arg1=SCRATCH["INJECT"],
            barrier=BAR_DEVICE,
        ))
        schedule.add(Step(
            op=OP_HC_UP,
            entry=_entry_id(pack, f"{hyper}.input_mix_weight_up"),
            src=SCRATCH["HC_LR"], dst=SCRATCH["MIXED"],
            arg0=SCRATCH["NORMED"], arg1=HC_COUNT, arg2=HIDDEN,
            barrier=BAR_DEVICE,
        ))
        return
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{hyper}.input_mix_weight_down"),
        src=SCRATCH["NORMED"], dst=SCRATCH["HC_LR"],
        arg0=R_GENERIC, arg1=ACT_SILU_SCALED, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{hyper}.input_mix_weight_up"),
        src=SCRATCH["HC_LR"], dst=SCRATCH["HC_W"],
        arg0=R_GENERIC, arg1=ACT_SIGMOID, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_HC_MIX, src=SCRATCH["HC_W"], dst=SCRATCH["MIXED"],
        arg0=HC_COUNT, arg1=HIDDEN, barrier=BAR_DEVICE,
    ))
    if has_inject:
        schedule.add(Step(
            op=OP_QMV, entry=_entry_id(pack, inject_key),
            src=SCRATCH["NORMED"], dst=SCRATCH["INJECT"],
            arg0=1, arg1=ACT_INJECT_GATE, arg2=DST_REPLICATED,
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
            src=SCRATCH["MIXED"], dst=dst,
            arg0=PHASE_ROWS["gdn_in_proj"],
            # the four input projections are independent; only the last needs
            # to publish before the core reads them
            barrier=BAR_NONE if name != "in_proj_a" else BAR_DEVICE,
        ))
    # The core reads four packed tensors, so the three that do not fit the
    # ``entry`` field travel in the argument words: A_log, dt_bias and the
    # gated RMS norm gain.  All three are dense bf16 in the pack.
    schedule.add(Step(
        op=OP_GDN_CORE, entry=_entry_id(pack, f"{attn}.conv1d.weight"),
        src=SCRATCH["GDN_QKV"], dst=SCRATCH["GDN_Y"],
        arg0=_entry_id(pack, f"{attn}.A_log"),
        arg1=_entry_id(pack, f"{attn}.dt_bias"),
        arg2=_entry_id(pack, f"{attn}.norm.weight"),
        barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{attn}.out_proj"),
        src=SCRATCH["GDN_Y"], dst=SCRATCH["BRANCH"],
        arg0=PHASE_ROWS["gdn_out_proj"], barrier=BAR_DEVICE,
    ))


def _attention_branch(schedule: Schedule, pack, plan: LayerPlan,
                      slot: int = 0) -> None:
    """One full-attention branch, complete: projections, q/k norms and RoPE,
    the KV and raw-index-key appends, the block the token closes, the indexer
    score and selection, split-K attention, the output gate and ``o_proj``.

    ``slot`` is this layer's index AMONG THE ATTENTION LAYERS.  It is the only
    thing about attention that varies with depth -- the twelve KV, raw-key and
    pooled ledgers share one binding each -- so it travels in the schedule and
    the per-token control block stays a single array.
    """
    attn = plan.key("self_attn")
    for name, dst in (
        ("q_proj", SCRATCH["ATT_QG"]),
        ("k_proj", SCRATCH["ATT_K"]),
        ("v_proj", SCRATCH["ATT_V"]),
    ):
        schedule.add(Step(
            op=OP_QMV, entry=_entry_id(pack, f"{attn}.{name}"),
            src=SCRATCH["MIXED"], dst=dst, arg0=R_GENERIC,
            barrier=BAR_NONE if name != "v_proj" else BAR_DEVICE,
        ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{attn}.indexer.index_qk_proj"),
        src=SCRATCH["MIXED"], dst=SCRATCH["IDX_QK"], arg0=R_GENERIC,
        barrier=BAR_DEVICE,
    ))
    # q/k norms and RoPE.  q_proj emits [q | gate] per head, so its q is
    # strided by 2 * head_dim and lands in a packed slab of its own; k and the
    # indexer's q are already packed and are normed in place.
    schedule.add(Step(
        op=OP_QK_NORM_ROPE, entry=_entry_id(pack, f"{attn}.q_norm.weight"),
        src=SCRATCH["ATT_QG"], dst=SCRATCH["ATT_Q"],
        arg0=N_Q_HEADS, arg1=HEAD_DIM, arg2=2 * HEAD_DIM, barrier=BAR_NONE,
    ))
    schedule.add(Step(
        op=OP_QK_NORM_ROPE, entry=_entry_id(pack, f"{attn}.k_norm.weight"),
        src=SCRATCH["ATT_K"], dst=SCRATCH["ATT_K"],
        arg0=N_KV_HEADS, arg1=HEAD_DIM, arg2=HEAD_DIM, barrier=BAR_NONE,
    ))
    schedule.add(Step(
        op=OP_QK_NORM_ROPE,
        entry=_entry_id(pack, f"{attn}.indexer.q_layernorm.weight"),
        src=SCRATCH["IDX_QK"], dst=SCRATCH["IDX_QK"],
        arg0=IDX_HEADS, arg1=IDX_HEAD_DIM, arg2=IDX_HEAD_DIM,
        barrier=BAR_DEVICE,
    ))
    # The KV and raw-index-key appends, then the one block this token can
    # close.  Both write their bindings in place; the pooling MUST be here
    # rather than on the host, because ``n_blocks = total // compress_ratio``
    # counts the current token, so every fourth token scores a block whose raw
    # keys were written two phases earlier in this same launch.
    schedule.add(Step(
        op=OP_KV_APPEND, src=SCRATCH["ATT_K"], dst=0,
        arg0=SCRATCH["ATT_V"],
        arg1=SCRATCH["IDX_QK"] + IDX_HEADS * IDX_HEAD_DIM,
        arg2=slot, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_POOL_BLOCK,
        entry=_entry_id(pack, f"{attn}.indexer.k_layernorm.weight"),
        src=0, dst=0, arg2=slot, barrier=BAR_DEVICE,
    ))
    # The indexer's score, then the selection.  The pooled-key ledger is a
    # host binding: one block closes every ``compress_ratio`` tokens, so
    # pooling it is incremental host state in the PLE class.  The block count
    # and the causal validity edge are context-dependent and arrive per token
    # in the control block, not in the schedule.
    schedule.add(Step(
        op=OP_INDEX_SCORE,
        entry=_entry_id(pack, f"{attn}.indexer.q_layernorm.weight"),
        # Scores live in the launch-sized score_tiles output, not in the
        # fixed per-query activation scratch. ``dst`` is intentionally unused.
        src=SCRATCH["IDX_QK"], dst=0, arg0=slot,
        barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        # ``src`` is intentionally unused; OP_INDEX_TOPB reads the same
        # query's dynamic score plane using actl.score_stride.
        op=OP_INDEX_TOPB, src=0, dst=SCRATCH["IDX_SEL"],
        arg0=BLOCK_TOPK,
        # every threadgroup runs the whole selection for itself, so its own
        # internal boundaries are threadgroup barriers; the result still has
        # to be visible to the attention phase, which is a device barrier
        barrier=BAR_DEVICE,
    ))
    schedule.add(Step(op=OP_ATTN, src=SCRATCH["ATT_Q"], dst=SCRATCH["ATT_O"],
                      arg0=slot, barrier=BAR_DEVICE))
    schedule.add(Step(op=OP_ATTN_COMBINE, src=SCRATCH["ATT_Q"],
                      dst=SCRATCH["ATT_O"], barrier=BAR_DEVICE))
    # ``out * mx.sigmoid(gate)``: the gate is q_proj's second per-head half.
    schedule.add(Step(
        op=OP_GATE_MUL, src=SCRATCH["ATT_QG"], dst=SCRATCH["ATT_O"],
        arg0=N_Q_HEADS, arg1=HEAD_DIM, arg2=2 * HEAD_DIM, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{attn}.o_proj"),
        src=SCRATCH["ATT_O"], dst=SCRATCH["BRANCH"], arg0=R_GENERIC,
        barrier=BAR_DEVICE,
    ))


def _moe_branch(schedule: Schedule, pack, plan: LayerPlan) -> None:
    mlp = plan.key("mlp")
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.gate"),
        src=SCRATCH["MIXED"], dst=SCRATCH["MOE_LOGITS"],
        arg0=PHASE_ROWS["moe_router"], barrier=BAR_DEVICE,
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
        src=SCRATCH["MOE_ACT"], dst=SCRATCH["BRANCH"],
        arg0=PHASE_ROWS["moe_down"], barrier=BAR_DEVICE,
    ))
    # shared expert, added into the same branch slot
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.shared_expert.gate_proj"),
        src=SCRATCH["MIXED"], dst=SCRATCH["SHARED_ACT"],
        arg0=R_GENERIC, arg1=ACT_SILU, barrier=BAR_NONE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.shared_expert.up_proj"),
        src=SCRATCH["MIXED"], dst=SCRATCH["SHARED_UP"],
        arg0=R_GENERIC, barrier=BAR_NONE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.shared_expert_gate"),
        src=SCRATCH["MIXED"], dst=SCRATCH["SHARED_GATE"], arg0=1,
        arg1=ACT_SIGMOID, arg2=DST_SCRATCH, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_SILU_MUL, src=SCRATCH["SHARED_ACT"], dst=SCRATCH["SHARED_ACT"],
        arg0=SCRATCH["SHARED_UP"], arg1=FF, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{mlp}.shared_expert.down_proj"),
        src=SCRATCH["SHARED_ACT"], dst=SCRATCH["SHARED_OUT"],
        arg0=R_GENERIC, arg1=ACT_NONE, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_ADD, src=SCRATCH["SHARED_OUT"], dst=SCRATCH["BRANCH"],
        arg0=HIDDEN, arg1=SCRATCH["SHARED_GATE"], barrier=BAR_DEVICE,
    ))


def _ple_chain(schedule: Schedule, pack, plan: LayerPlan, resid: int) -> None:
    """The PLE chain's DEVICE half: ``x = x + ple(x, input_ids)``.

    Only the n-gram gather in front of this stays hoisted -- it is a file-backed
    lookup that depends on the input token alone, the same class as
    ``embed_tokens`` -- and its rows arrive in ``SCRATCH["PLE_EMB"]`` as a host
    input.  Everything from ``key_proj`` through the residual add is here.

    Which stock op each boundary matches: ``key_proj``/``value_proj`` are
    ``quantized_matmul``; ``norm_key``/``norm_query``/``norm_conv`` are
    ``GroupRMSNorm`` in its eager fp32 form; ``OP_PLE_GATE`` is
    ``PLELayer._chain_gate`` -- the key/query dot scaled by
    ``1/sqrt(hidden)``, ``sign(g) * sqrt(max(|g|, 1e-6))`` -- followed by the
    STANDALONE ``mx.sigmoid`` and the ``sigmoid * value`` product; and
    ``OP_PLE_CONV`` is ``nn.silu(conv1d(...))`` plus ``gated + conv``.

    The sigmoid is the reason the gate is a phase of its own.  The engine's
    compiled PLE chain is exact only because ``mx.sigmoid`` is kept eager
    OUTSIDE the traced span -- inside one, the fused rewrite disagrees with the
    standalone primitive.  The kernel therefore matches the standalone op, with
    ``metal::precise::exp``, not the fast intrinsic.
    """
    ple = plan.key("ple")
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{ple}.key_proj"),
        src=SCRATCH["PLE_EMB"], dst=SCRATCH["HC_W"], arg0=R_GENERIC,
        barrier=BAR_NONE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{ple}.value_proj"),
        src=SCRATCH["PLE_EMB"], dst=SCRATCH["SHARED_OUT"], arg0=R_GENERIC,
        barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_GROUP_RMSNORM, entry=_entry_id(pack, f"{ple}.norm_key.weight"),
        src=SCRATCH["HC_W"], dst=SCRATCH["NORMED"],
        arg0=HC_HIDDEN, arg1=HIDDEN, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_GROUP_RMSNORM, entry=_entry_id(pack, f"{ple}.norm_query.weight"),
        src=resid, dst=SCRATCH["HC_W"],
        arg0=HC_HIDDEN, arg1=HIDDEN, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_PLE_GATE, src=SCRATCH["NORMED"], dst=SCRATCH["NORMED"],
        arg0=SCRATCH["HC_W"], arg1=SCRATCH["SHARED_OUT"], barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_GROUP_RMSNORM, entry=_entry_id(pack, f"{ple}.norm_conv.weight"),
        src=SCRATCH["NORMED"], dst=SCRATCH["HC_W"],
        arg0=HC_HIDDEN, arg1=HIDDEN, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_PLE_CONV, entry=_entry_id(pack, f"{ple}.conv1d.weight"),
        src=SCRATCH["HC_W"], dst=resid, arg0=SCRATCH["NORMED"],
        barrier=BAR_DEVICE,
    ))


def build_layer_schedule(
    schedule: Schedule, pack, plan: LayerPlan, resid: int, other: int,
    *, fused_hyper: bool = True, attn_slot: int = 0, has_ple: bool = False,
) -> int:
    """Append one decoder layer.  Returns the slab the output landed in."""
    if has_ple:
        _ple_chain(schedule, pack, plan, resid)
    _hyper_block(schedule, pack, plan, "attn_hyper_connection", resid,
                 fused=fused_hyper)
    if plan.is_linear:
        _gdn_branch(schedule, pack, plan)
    else:
        _attention_branch(schedule, pack, plan, attn_slot)
    schedule.add(Step(
        op=OP_INJECT, src=resid, dst=other, arg0=SCRATCH["BRANCH"],
        arg1=SCRATCH["INJECT"], arg2=HC_COUNT, barrier=BAR_DEVICE,
    ))
    resid, other = other, resid
    _hyper_block(schedule, pack, plan, "mlp_hyper_connection", resid,
                 fused=fused_hyper)
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
    fused_hyper: bool = True,
    ple_layer_ids: Optional[list[int]] = None,
    mixer: str = "language_model.model.hyper_connection_mixer",
) -> Schedule:
    """The whole per-token phase sequence, hyper-connections included."""
    schedule = Schedule()
    resid, other = SCRATCH["RESID_A"], SCRATCH["RESID_B"]
    indices = list(range(len(layer_types))) if layers is None else list(layers)
    # A layer's attention slot is its rank among the FULL-ATTENTION layers of
    # the whole model, not of the subset a probe selects -- the ledgers are
    # allocated per model layer, so a 3-layer probe on layers 0/1/3 must still
    # address slot 0 for layer 3 only if it allocated one ledger.  Ranking
    # within the emitted set keeps probe and full token consistent.
    attn_rank = {
        index: rank
        for rank, index in enumerate(
            i for i in indices if layer_types[i] != "linear_attention"
        )
    }
    ple = set(int(i) - 1 for i in (ple_layer_ids or ()))
    for index in indices:
        plan = LayerPlan(
            index=index,
            is_linear=layer_types[index] == "linear_attention",
            prefix=f"{prefix}.{index}",
        )
        landed = build_layer_schedule(schedule, pack, plan, resid, other,
                                      fused_hyper=fused_hyper,
                                      attn_slot=attn_rank.get(index, 0),
                                      has_ple=index in ple)
        resid, other = landed, (resid if landed == other else other)
    # final mixer: same GatedResidual, without the inject combine
    tail = LayerPlan(index=-1, is_linear=True, prefix=mixer.rsplit(".", 1)[0])
    _hyper_block(
        schedule, pack,
        LayerPlan(index=-1, is_linear=True, prefix=mixer.rsplit(".", 1)[0]),
        mixer.rsplit(".", 1)[1], resid, fused=fused_hyper,
    )
    if include_lm_head:
        schedule.add(Step(
            op=OP_QMV, entry=_entry_id(pack, "language_model.lm_head"),
            src=SCRATCH["MIXED"], dst=0, arg0=R_GENERIC, arg2=DST_OUT,
            barrier=BAR_NONE,
        ))
    return schedule


def build_mtp_schedule(
    pack,
    *,
    prefix: str = "mtp",
    include_lm_head: bool = True,
    fused_hyper: bool = True,
) -> Schedule:
    """The MTP head: fuse, one full-attention layer, its mixer, the lm head.

    ``Qwen4ExpMTP.fuse`` is four phases, not one op: a plain RMSNorm over the
    embedding (a GroupRMSNorm whose group IS the whole vector), its projection,
    a GroupRMSNorm over the four hyper streams, and ``fc_hidden`` applied to
    each stream separately -- one ``OP_QMV`` per stream, because a matvec op
    reads one source vector.  The embedding is then broadcast into all four.

    The embedding arrives in ``SCRATCH["BRANCH"]`` and the trunk's scheme-A
    hyper hiddens in ``SCRATCH["RESID_A"]``; both are host inputs, in the same
    class as the PLE gather.
    """
    schedule = Schedule()
    # embedding: RMSNorm over the whole vector, then fc_embedding
    schedule.add(Step(
        op=OP_GROUP_RMSNORM,
        entry=_entry_id(pack, f"{prefix}.pre_fc_norm_embedding.weight"),
        src=SCRATCH["BRANCH"], dst=SCRATCH["MIXED"],
        arg0=HIDDEN, arg1=HIDDEN, barrier=BAR_DEVICE,
    ))
    schedule.add(Step(
        op=OP_QMV, entry=_entry_id(pack, f"{prefix}.fc_embedding"),
        src=SCRATCH["MIXED"], dst=SCRATCH["SHARED_OUT"],
        arg0=R_GENERIC, barrier=BAR_DEVICE,
    ))
    # the four hyper streams: one GroupRMSNorm, then fc_hidden per stream
    schedule.add(Step(
        op=OP_GROUP_RMSNORM,
        entry=_entry_id(pack, f"{prefix}.pre_fc_norm_hidden.weight"),
        src=SCRATCH["RESID_A"], dst=SCRATCH["NORMED"],
        arg0=HC_HIDDEN, arg1=HIDDEN, barrier=BAR_DEVICE,
    ))
    for stream in range(HC_COUNT):
        schedule.add(Step(
            op=OP_QMV, entry=_entry_id(pack, f"{prefix}.fc_hidden"),
            src=SCRATCH["NORMED"] + stream * HIDDEN,
            dst=SCRATCH["RESID_B"] + stream * HIDDEN,
            arg0=R_GENERIC,
            barrier=BAR_NONE if stream < HC_COUNT - 1 else BAR_DEVICE,
        ))
    schedule.add(Step(
        op=OP_ADD_BCAST, src=SCRATCH["SHARED_OUT"], dst=SCRATCH["RESID_B"],
        arg0=HIDDEN, arg1=HC_COUNT, barrier=BAR_DEVICE,
    ))
    plan = LayerPlan(index=0, is_linear=False, prefix=f"{prefix}.layers.0")
    landed = build_layer_schedule(
        schedule, pack, plan, SCRATCH["RESID_B"], SCRATCH["RESID_A"],
        fused_hyper=fused_hyper,
    )
    _hyper_block(schedule, pack,
                 LayerPlan(index=-1, is_linear=True, prefix=prefix),
                 "hyper_connection_mixer", landed, fused=fused_hyper)
    if include_lm_head:
        schedule.add(Step(
            op=OP_QMV, entry=_entry_id(pack, "language_model.lm_head"),
            src=SCRATCH["MIXED"], dst=0, arg0=R_GENERIC, arg2=DST_OUT,
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
        self.gdn_slot = 0
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
        self.gdn_slot = 0
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

    def _op_7(self, step: Step) -> None:  # OP_GDN_CORE
        """The conv step, SiLU, L2 normalisation, gated delta rule, gated norm.

        ``self.state`` carries ``conv`` [L, K-1, CONV_DIM] and ``rec``
        [L, HV, DV, DK]; the slot is the running count of cores executed, the
        same rule the kernel uses, so neither side needs a field for it.
        """
        conv_w = self.weights(step.entry)["weight"].astype(mx.float32)
        conv_w = conv_w.reshape(CONV_DIM, CONV_KERNEL)
        a_log = self.weights(step.arg0)["weight"].astype(mx.float32)
        dt_bias = self.weights(step.arg1)["weight"].astype(mx.float32)
        gain = self.weights(step.arg2)["weight"].astype(mx.float32)

        slot = self.gdn_slot
        self.gdn_slot += 1
        conv_state = self.state["conv"][slot].astype(mx.float32)
        rec = self.state["rec"][slot].astype(mx.float32)

        qkv = self.read(step.src, CONV_DIM)
        z = self.read(SCRATCH["GDN_Z"], VALUE_DIM)
        bb = self.read(SCRATCH["GDN_BA"], GDN_VALUE_HEADS)
        aa = self.read(SCRATCH["GDN_BA"] + GDN_VALUE_HEADS, GDN_VALUE_HEADS)

        hist = mx.concatenate([conv_state, qkv[None, :]], axis=0)
        conv_out = (hist * conv_w.T).sum(axis=0)
        conv_out = conv_out * mx.sigmoid(conv_out)
        self.state["conv_out"] = self.state.get("conv_out", {})
        self.state["conv_out"][slot] = hist[1:]

        key_dim = GDN_KEY_HEADS * GDN_KEY_DIM
        qh = conv_out[:key_dim].reshape(GDN_KEY_HEADS, GDN_KEY_DIM)
        kh = conv_out[key_dim: 2 * key_dim].reshape(GDN_KEY_HEADS, GDN_KEY_DIM)
        vh = conv_out[2 * key_dim:].reshape(GDN_VALUE_HEADS, GDN_VALUE_DIM)
        qh = qh * mx.rsqrt(mx.sum(qh * qh, axis=-1, keepdims=True) + 1.0e-6)
        kh = kh * mx.rsqrt(mx.sum(kh * kh, axis=-1, keepdims=True) + 1.0e-6)
        qh = qh * (GDN_KEY_DIM ** -0.5)

        decay = mx.exp(
            -mx.exp(a_log) * mx.logaddexp(aa + dt_bias, mx.zeros_like(aa))
        )
        beta = mx.sigmoid(bb)
        qf = mx.repeat(qh, GDN_RATIO, axis=0)
        kf = mx.repeat(kh, GDN_RATIO, axis=0)

        state = rec * decay[:, None, None]
        kv = (state * kf[:, None, :]).sum(-1)
        delta = (vh - kv) * beta[:, None]
        state = state + delta[:, :, None] * kf[:, None, :]
        y = (state * qf[:, None, :]).sum(-1)
        self.state["rec_out"] = self.state.get("rec_out", {})
        self.state["rec_out"][slot] = state

        scale = mx.rsqrt((y * y).mean(axis=-1, keepdims=True) + RMS_EPS)
        y = gain * (y * scale)
        y = y * mx.sigmoid(z.reshape(GDN_VALUE_HEADS, GDN_VALUE_DIM))
        self.write(step.dst, y.reshape(VALUE_DIM))

    # ---- the attention layer's non-matvec steps, and the PLE device half
    def _rope(self, x: mx.array, position, dims: int) -> mx.array:
        """``_apply_rope_positions``: the eager, non-traditional partial form.

        The indexer runs exactly this; ``mx.fast.rope`` runs the same formula
        for attention q/k with its own reduction of the angle, which is where
        the attention layer's class-2 gap against stock comes from.
        """
        base = float(self.state.get("rope_theta", 1.0e7))
        half = dims // 2
        freqs = mx.exp(-math.log(base) * mx.arange(0, dims, 2) / dims)
        angles = mx.array(float(position), mx.float32) * freqs
        cos, sin = mx.cos(angles), mx.sin(angles)
        left, right = x[..., :half], x[..., half:dims]
        rot = mx.concatenate(
            [left * cos - right * sin, right * cos + left * sin], axis=-1
        )
        return mx.concatenate([_bf(rot), x[..., dims:]], axis=-1)

    def _op_21(self, step: Step) -> None:  # OP_QK_NORM_ROPE
        heads, hd, stride = step.arg0, step.arg1, step.arg2 or step.arg1
        gain = self.weights(step.entry)["weight"].astype(mx.float32)
        rows = mx.stack([
            self.read(step.src + h * stride, hd) for h in range(heads)
        ])
        scale = mx.rsqrt(mx.mean(rows * rows, axis=-1, keepdims=True) + RMS_EPS)
        normed = _bf(rows * scale * gain)
        roped = self._rope(normed, self.state.get("rope_pos", 0), ROTARY_DIM)
        self.write(step.dst, roped.reshape(-1))

    def _op_22(self, step: Step) -> None:  # OP_KV_APPEND
        """The mirror keeps the appends as plain host state; the KERNEL writes
        its bindings in place.  Both are the same values in the same slots."""
        slot, li = self.state.get("kv_slot", 0), step.arg2
        for name, base, width in (
            ("k", step.src, N_KV_HEADS * HEAD_DIM),
            ("v", step.arg0, N_KV_HEADS * HEAD_DIM),
            ("raw", step.arg1, IDX_HEAD_DIM),
        ):
            self.state.setdefault(name, {})[(li, slot)] = _bf(
                self.read(base, width)
            )

    def _op_26(self, step: Step) -> None:  # OP_POOL_BLOCK
        if not self.state.get("pool_new_block"):
            return
        li, block = step.arg2, self.state["n_blocks"] - 1
        size = self.state.get("block_size", IDX_COMPRESS)
        start = block * size
        gain = self.weights(step.entry)["weight"].astype(mx.float32)
        rows = mx.stack([
            self.state["raw"][(li, start + t)] for t in range(size)
        ])
        pooled = _bf(mx.mean(rows, axis=0))
        scale = mx.rsqrt(mx.mean(pooled * pooled, keepdims=True) + RMS_EPS)
        pooled = self._rope(_bf(pooled * scale * gain), start, ROTARY_DIM)
        self.state.setdefault("pooled", {})[(li, block)] = pooled

    def _op_23(self, step: Step) -> None:  # OP_GATE_MUL
        heads, hd, stride = step.arg0, step.arg1, step.arg2
        gate = mx.stack([
            self.read(step.src + h * stride + hd, hd) for h in range(heads)
        ]).reshape(-1)
        self.write(step.dst, self.read(step.dst, heads * hd) * mx.sigmoid(gate))

    def _op_24(self, step: Step) -> None:  # OP_PLE_GATE
        key = self.read(step.src, HC_HIDDEN).reshape(HC_COUNT, HIDDEN)
        query = self.read(step.arg0, HC_HIDDEN).reshape(HC_COUNT, HIDDEN)
        value = self.read(step.arg1, HIDDEN)
        gate = mx.sum(key * query, axis=-1, keepdims=True) / math.sqrt(HIDDEN)
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        self.write(step.dst, (mx.sigmoid(gate) * value[None, :]).reshape(-1))

    def _op_25(self, step: Step) -> None:  # OP_PLE_CONV
        weight = self.weights(step.entry)["weight"].astype(mx.float32)
        weight = weight.reshape(HC_HIDDEN, PLE_CONV_KERNEL)
        state = self.state["ple_conv"].astype(mx.float32)
        new = self.read(step.src, HC_HIDDEN)
        window = mx.concatenate([state, new[None, :]], axis=0)
        taps = mx.stack([
            window[k * PLE_NGRAM] for k in range(PLE_CONV_KERNEL)
        ])
        conv = mx.sum(taps * weight.T, axis=0)
        conv = conv * mx.sigmoid(conv)
        self.state["ple_conv_out"] = window[1:]
        self.write(step.dst,
                   self.read(step.dst, HC_HIDDEN)
                   + self.read(step.arg0, HC_HIDDEN) + conv)

    def _op_16(self, step: Step) -> None:  # OP_HC_DOWN
        entry = self.entry(step.entry)
        parts = self.weights(step.entry)
        x = self.read(step.src, entry.cols)
        down = mx.quantized_matmul(
            x, parts["weight"], parts["scales"], parts["biases"],
            transpose=True, group_size=entry.group_size, bits=entry.bits,
        )
        self.write(step.dst, _activate(down, ACT_SILU_SCALED))
        if step.arg0 == NO_ENTRY:
            return
        gate_entry = self.entry(step.arg0)
        gate = self.weights(step.arg0)
        inject = mx.quantized_matmul(
            x, gate["weight"], gate["scales"], gate["biases"], transpose=True,
            group_size=gate_entry.group_size, bits=gate_entry.bits,
        )
        self.write(step.arg1, _activate(inject, ACT_INJECT_GATE))

    def _op_17(self, step: Step) -> None:  # OP_HC_UP
        entry = self.entry(step.entry)
        parts = self.weights(step.entry)
        count, width = step.arg1, step.arg2
        up = mx.quantized_matmul(
            self.read(step.src, entry.cols), parts["weight"], parts["scales"],
            parts["biases"], transpose=True, group_size=entry.group_size,
            bits=entry.bits,
        )
        weights = mx.sigmoid(up).reshape(count, width)
        streams = self.read(step.arg0, count * width).reshape(count, width)
        self.write(step.dst, mx.mean(weights * streams, axis=0))

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


def _bf(x: mx.array) -> mx.array:
    """Round at a boundary stock rounds at.  ``mx.fast.rms_norm`` and
    ``mx.fast.rope`` both return the activation dtype, so a phase that keeps
    fp32 through them would not be comparable at the next boundary -- and the
    attention phase's bit-identity with the shipped indexed kernel requires
    exactly ``float(bf16 q)``."""
    return x.astype(mx.bfloat16).astype(mx.float32)


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
