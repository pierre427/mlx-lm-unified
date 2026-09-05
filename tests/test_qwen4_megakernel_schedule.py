"""Structure of the per-token schedule, with no checkpoint and no GPU.

What a real token costs in PHASES is the number the megakernel design turns
on -- the spike priced 150 empty phases at 0.38-0.55 ms against ~13.5 ms of
dispatch floor -- so it is asserted here rather than discovered on the GPU.
"""

import unittest

from mlx_lm.models import qwen4_megakernel as mk
from mlx_lm.models.qwen4_megakernel_pack import decode_path_keys
from mlx_lm.models.qwen4_megakernel_schedule import (
    LayerPlan,
    DST_OUT,
    build_layer_schedule,
    build_mtp_schedule,
    build_token_schedule,
)

LAYER_TYPES = [
    "linear_attention" if (i + 1) % 4 else "full_attention" for i in range(48)
]


class _StubEntry:
    def __init__(self, index):
        self.index = index


class _StubPack:
    """Only the index map, which is all the schedule builder reads."""

    def __init__(self, keys):
        self.entries = {key: _StubEntry(i) for i, key in enumerate(keys)}


def _pack():
    plan = decode_path_keys(
        num_layers=48, layer_types=LAYER_TYPES, ple_layer_ids=[2],
        include_mtp=True,
    )
    return _StubPack([key for key, _ in plan])


class TestSchedule(unittest.TestCase):
    def test_layer_phase_counts(self):
        pack = _pack()
        for index, expected_branch in ((0, "gdn"), (3, "attention")):
            schedule = mk.Schedule()
            plan = LayerPlan(
                index=index,
                is_linear=LAYER_TYPES[index] == "linear_attention",
                prefix=f"language_model.model.layers.{index}",
            )
            landed = build_layer_schedule(
                schedule, pack, plan,
                mk.SCRATCH["RESID_A"], mk.SCRATCH["RESID_B"],
            )
            # two hyper blocks of 3, two injects, one branch, one MoE block
            ops = [step.op for step in schedule.steps]
            self.assertEqual(ops.count(mk.OP_GROUP_RMSNORM), 2, expected_branch)
            self.assertEqual(ops.count(mk.OP_HC_DOWN), 2, expected_branch)
            self.assertEqual(ops.count(mk.OP_HC_UP), 2, expected_branch)
            self.assertEqual(ops.count(mk.OP_HC_MIX), 0, expected_branch)
            self.assertEqual(ops.count(mk.OP_INJECT), 2, expected_branch)
            self.assertEqual(ops.count(mk.OP_MOE_TOPK), 1, expected_branch)
            self.assertEqual(ops.count(mk.OP_MOE_E1), 1, expected_branch)
            self.assertEqual(ops.count(mk.OP_MOE_E2), 1, expected_branch)
            if plan.is_linear:
                self.assertEqual(ops.count(mk.OP_GDN_CORE), 1)
                self.assertEqual(ops.count(mk.OP_ATTN), 0)
            else:
                self.assertEqual(ops.count(mk.OP_ATTN), 1)
                self.assertEqual(ops.count(mk.OP_INDEX_TOPB), 1)
            # a layer ends in the slab it did not start in: two injects, so
            # back where it started
            self.assertEqual(landed, mk.SCRATCH["RESID_A"])

    def test_the_unfused_hyper_spelling_is_the_same_arithmetic(self):
        # Five steps instead of three: the norm, the two projections, the mix
        # and the inject gate.  Kept selectable so a barrier-cost experiment
        # can price the fusion, and because the mirror was written against it.
        pack = _pack()
        counts = {}
        for fused in (True, False):
            schedule = mk.Schedule()
            plan = LayerPlan(index=0, is_linear=True,
                             prefix="language_model.model.layers.0")
            build_layer_schedule(schedule, pack, plan, mk.SCRATCH["RESID_A"],
                                 mk.SCRATCH["RESID_B"], fused_hyper=fused)
            ops = [step.op for step in schedule.steps]
            counts[fused] = (len(schedule), ops.count(mk.OP_HC_MIX))
        self.assertEqual(counts[True][1], 0)
        self.assertEqual(counts[False][1], 2)
        # two mixers x two extra steps
        self.assertEqual(counts[False][0] - counts[True][0], 4)

    def test_residual_slab_ping_pongs(self):
        """No phase may read and write the same residual address.

        The spike's U2 result is that a REUSED scratch address is what goes
        stale, and this kernel reuses every address 48 times.
        """
        pack = _pack()
        schedule = mk.Schedule()
        plan = LayerPlan(index=0, is_linear=True,
                         prefix="language_model.model.layers.0")
        build_layer_schedule(schedule, pack, plan,
                             mk.SCRATCH["RESID_A"], mk.SCRATCH["RESID_B"])
        for step in schedule.steps:
            if step.op == mk.OP_INJECT:
                self.assertNotEqual(step.src, step.dst)
                self.assertIn(step.src, (mk.SCRATCH["RESID_A"],
                                         mk.SCRATCH["RESID_B"]))
                self.assertIn(step.dst, (mk.SCRATCH["RESID_A"],
                                         mk.SCRATCH["RESID_B"]))

    def test_token_schedule_phase_budget(self):
        pack = _pack()
        schedule = build_token_schedule(pack, layer_types=LAYER_TYPES)
        ops = [step.op for step in schedule.steps]
        self.assertEqual(ops.count(mk.OP_GDN_CORE), 36)
        self.assertEqual(ops.count(mk.OP_ATTN), 12)
        self.assertEqual(ops.count(mk.OP_MOE_E2), 48)
        self.assertEqual(ops.count(mk.OP_INJECT), 96)
        # MEASURED 2026-09-03: 1,361 phases, 940 device barriers.  At the
        # spike's per-barrier cost that is 2.3 ms (2.4 us, synthetic kernel)
        # to 4.9 ms (5.2 us, the register-heavy GDN kernel), against ~13.5 ms
        # of per-dispatch floor and 24.5 ms of GPU busy in a token today.
        #
        # The spike's "150 phases = 0.4 ms" is NOT the comparison: it covered
        # one GDN block and one MoE block with no hyper-connection glue, and a
        # real token runs two GatedResidual blocks per layer -- the part the
        # bandwidth page priced at 7.8 ms/token of dispatches, and the part a
        # megakernel should swallow most profitably.  Cutting the count
        # further is phase FUSION, which is the tuning spec's call; this gate
        # only stops it growing.
        self.assertLessEqual(schedule.device_barriers, 1000)
        self.assertGreater(len(schedule), 500)

    def test_barrier_map_is_data(self):
        """Adjacent independent projections may skip the grid barrier."""
        pack = _pack()
        schedule = mk.Schedule()
        plan = LayerPlan(index=0, is_linear=True,
                         prefix="language_model.model.layers.0")
        build_layer_schedule(schedule, pack, plan,
                             mk.SCRATCH["RESID_A"], mk.SCRATCH["RESID_B"])
        kinds = {step.barrier for step in schedule.steps}
        self.assertIn(mk.BAR_NONE, kinds)
        self.assertIn(mk.BAR_DEVICE, kinds)
        self.assertLess(schedule.device_barriers, len(schedule))


if __name__ == "__main__":
    unittest.main()


class TestPhaseBSchedules(unittest.TestCase):
    """The shapes phase B's kernel body actually walks."""

    def test_attention_branch_emits_the_indexer_and_both_attention_passes(self):
        pack = _pack()
        schedule = mk.Schedule()
        plan = LayerPlan(index=3, is_linear=False,
                         prefix="language_model.model.layers.3")
        build_layer_schedule(schedule, pack, plan, mk.SCRATCH["RESID_A"],
                             mk.SCRATCH["RESID_B"])
        ops = [step.op for step in schedule.steps]
        for op in (mk.OP_INDEX_SCORE, mk.OP_INDEX_TOPB, mk.OP_ATTN,
                   mk.OP_ATTN_COMBINE):
            self.assertEqual(ops.count(op), 1, mk.OP_NAMES[op])
        # the score has to be published before the selection reads it, and the
        # selection before the attend, or the phases are simply out of order
        self.assertLess(ops.index(mk.OP_INDEX_SCORE), ops.index(mk.OP_INDEX_TOPB))
        self.assertLess(ops.index(mk.OP_INDEX_TOPB), ops.index(mk.OP_ATTN))
        self.assertLess(ops.index(mk.OP_ATTN), ops.index(mk.OP_ATTN_COMBINE))

    def test_mtp_head_fuses_every_stream_and_ends_in_the_lm_head(self):
        pack = _pack()
        schedule = build_mtp_schedule(pack)
        ops = [step.op for step in schedule.steps]
        # fc_hidden is one OP_QMV per hyper stream: a matvec op reads ONE
        # source vector, so four streams are four steps, not one
        fc = [s for s in schedule.steps
              if s.op == mk.OP_QMV
              and s.entry == pack.entries["mtp.fc_hidden"].index]
        self.assertEqual(len(fc), mk.HC_COUNT)
        self.assertEqual(
            sorted(s.src for s in fc),
            [mk.SCRATCH["NORMED"] + i * mk.HIDDEN for i in range(mk.HC_COUNT)],
        )
        self.assertEqual(ops.count(mk.OP_ADD_BCAST), 1)
        self.assertEqual(schedule.steps[-1].arg2, DST_OUT)
        self.assertEqual(schedule.steps[-1].entry,
                         pack.entries["language_model.lm_head"].index)


class TestThreadgroupBudget(unittest.TestCase):
    def test_arena_and_the_selector_both_fit(self):
        from mlx_lm.models import qwen4_megakernel_body as body
        self.assertLessEqual(body.THREADGROUP_BYTES, mk.THREADGROUP_BYTES_CAP)
        # the top-block selector aliases its histogram and per-thread counters
        # over the TGX staging block, which is not live while it runs
        for threads in (128, 256, 512):
            self.assertLessEqual(256 + 2 * threads + 2, body.TG["TLOG"])
