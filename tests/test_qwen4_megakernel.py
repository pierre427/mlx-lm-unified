"""Contract tests for the megakernel skeleton: admission, schedule, top-k.

CPU only.  The Metal kernel itself is exercised by
``tests/test_qwen4_megakernel_metal.py``; what is proven here is the part that
has to be right BEFORE a kernel runs -- that admission fails closed, that the
schedule the kernel walks round-trips, and that the in-kernel block selection
algorithm agrees with the stock ``argpartition`` it replaces.
"""

import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models import qwen4_megakernel as mk


class TestAdmission(unittest.TestCase):
    def setUp(self):
        self._saved = mk._MEGAKERNEL_ENABLED
        mk.set_qwen4_megakernel(True)
        self.kwargs = dict(
            width=1, batch=1, pack=object(), schedule=mk.Schedule(),
            speculating=False, training=False, sharded=False,
        )
        self.kwargs["schedule"].add(mk.Step(op=mk.OP_NOP))

    def tearDown(self):
        mk.set_qwen4_megakernel(self._saved)

    def test_default_is_off(self):
        mk.set_qwen4_megakernel(False)
        self.assertFalse(
            mk.admit_megakernel_decode(**self.kwargs).accepted
        )
        self.assertEqual(
            mk.admit_megakernel_decode(**self.kwargs).reason, "disabled"
        )

    def test_refuses_everything_it_does_not_serve(self):
        for field, value, reason in (
            ("width", 4, "query width 4"),
            ("batch", 2, "batch 2"),
            ("training", True, "training"),
            ("sharded", True, "distributed sharding"),
            ("speculating", True, "speculative rollback"),
            ("mask", mx.zeros((1, 1)), "masked decode"),
            ("pack", None, "weights not packed"),
            ("schedule", None, "empty schedule"),
        ):
            kwargs = dict(self.kwargs)
            kwargs[field] = value
            decision = mk.admit_megakernel_decode(**kwargs)
            self.assertFalse(decision.accepted, field)
            self.assertEqual(decision.reason, reason, field)

    def test_empty_schedule_is_refused(self):
        kwargs = dict(self.kwargs)
        kwargs["schedule"] = mk.Schedule()
        self.assertEqual(
            mk.admit_megakernel_decode(**kwargs).reason, "empty schedule"
        )

    def test_accepts_the_one_shape_it_serves(self):
        decision = mk.admit_megakernel_decode(**self.kwargs)
        if not mk._device_supported():
            self.skipTest("no Apple GPU")
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "engaged")


class TestSchedule(unittest.TestCase):
    def test_round_trip(self):
        schedule = mk.Schedule()
        schedule.add(mk.Step(op=mk.OP_QMV, entry=7, src=1, dst=2, arg0=3))
        schedule.add(mk.Step(op=mk.OP_GDN_CORE, barrier=mk.BAR_THREADGROUP))
        schedule.add(mk.Step(op=mk.OP_MOE_TOPK, barrier=mk.BAR_NONE))
        flat = schedule.to_array().tolist()
        self.assertEqual(len(flat), len(schedule) * mk.STEP_STRIDE)
        for index, step in enumerate(schedule.steps):
            base = index * mk.STEP_STRIDE
            self.assertEqual(flat[base: base + mk.STEP_STRIDE], step.row())
        self.assertEqual(schedule.device_barriers, 1)

    def test_scratch_blocks_do_not_overlap(self):
        offsets = sorted(mk.SCRATCH.items(), key=lambda item: item[1])
        sizes = dict(mk._SCRATCH_BLOCKS)
        for (name, offset), (next_name, next_offset) in zip(offsets, offsets[1:]):
            self.assertLessEqual(offset + sizes[name], next_offset, name)
        last, offset = offsets[-1]
        self.assertEqual(offset + sizes[last], mk.SCRATCH_FLOATS)

    def test_opcodes_are_distinct(self):
        values = [
            value for name, value in vars(mk).items()
            if name.startswith("OP_") and isinstance(value, int)
        ]
        self.assertEqual(len(values), len(set(values)))


class TestTopBlockSelection(unittest.TestCase):
    """The in-kernel block selection, against the stock ``argpartition``."""

    def _stock(self, scores, k):
        arr = mx.array(np.asarray(scores, dtype=np.float32))
        n = arr.size
        k = min(k, n)
        return set(
            mx.argpartition(arr, kth=n - k)[-k:].tolist()
        )

    def test_sort_key_is_monotone(self):
        values = np.array(
            [-np.inf, -1e30, -1.0, -1e-30, -0.0, 0.0, 1e-30, 1.0, 1e30, np.inf],
            dtype=np.float32,
        )
        keys = mk.float_sort_key(values)
        # -0.0 and 0.0 are distinct bit patterns and must not reorder anything
        self.assertTrue(np.all(np.diff(keys.astype(np.int64)) >= 0), keys)
        # -inf is the lowest key any non-NaN value takes, so an invalid block
        # sorts below every real score
        self.assertEqual(int(keys[0]), 0x007FFFFF)
        self.assertEqual(int(keys[0]), int(mk.float_sort_key([-np.inf])[0]))

    def test_matches_argpartition_without_ties(self):
        rng = np.random.default_rng(0)
        for n, k in ((100, 7), (4096, 512), (16384, 512), (513, 512)):
            scores = rng.normal(size=n).astype(np.float32)
            got = set(mk.select_top_blocks_mirror(scores, k).tolist())
            self.assertEqual(len(got), min(k, n), (n, k))
            self.assertEqual(got, self._stock(scores, k), (n, k))

    def test_k_at_or_above_n_selects_everything(self):
        scores = np.arange(64, dtype=np.float32)
        got = mk.select_top_blocks_mirror(scores, 64)
        self.assertEqual(got.tolist(), list(range(64)))
        got = mk.select_top_blocks_mirror(scores, 500)
        self.assertEqual(got.tolist(), list(range(64)))

    def test_ties_select_the_right_scores_and_break_by_index(self):
        # the real shape: a relu-sum score is exactly 0.0 for a block no head
        # likes, and an invalid block is exactly -inf
        scores = np.zeros(2000, dtype=np.float32)
        scores[:100] = np.linspace(1.0, 5.0, 100)
        scores[1500:] = -np.inf
        got = mk.select_top_blocks_mirror(scores, 512)
        self.assertEqual(len(got), 512)
        self.assertEqual(len(set(got.tolist())), 512)
        picked = np.sort(scores[got])[::-1]
        want = np.sort(scores)[::-1][:512]
        np.testing.assert_array_equal(picked, want)
        # ties resolved by lowest index: the 100 real scores, then blocks
        # 100..511 of the zero run
        self.assertEqual(got.tolist(), list(range(512)))

    def test_all_equal_scores(self):
        scores = np.full(4096, 0.25, dtype=np.float32)
        got = mk.select_top_blocks_mirror(scores, 512)
        self.assertEqual(got.tolist(), list(range(512)))

    def test_all_invalid_blocks(self):
        scores = np.full(1024, -np.inf, dtype=np.float32)
        got = mk.select_top_blocks_mirror(scores, 512)
        self.assertEqual(len(got), 512)
        self.assertEqual(got.tolist(), list(range(512)))

    def test_negative_scores_only(self):
        rng = np.random.default_rng(3)
        scores = -np.abs(rng.normal(size=4096)).astype(np.float32)
        got = set(mk.select_top_blocks_mirror(scores, 512).tolist())
        self.assertEqual(got, self._stock(scores, 512))

    def test_threaded_mirror_matches_the_algorithm(self):
        rng = np.random.default_rng(11)
        cases = [
            rng.normal(size=4096).astype(np.float32),
            np.zeros(2000, dtype=np.float32),
            np.full(1024, -np.inf, dtype=np.float32),
        ]
        mixed = np.zeros(3000, dtype=np.float32)
        mixed[:200] = rng.normal(size=200)
        mixed[2500:] = -np.inf
        cases.append(mixed.astype(np.float32))
        for scores in cases:
            want = mk.select_top_blocks_mirror(scores, 512)
            for nt in (32, 64, 256, 1024):
                got = mk.select_top_blocks_threaded_mirror(scores, 512, nt=nt)
                self.assertEqual(len(set(got.tolist())), len(got), nt)
                self.assertNotIn(0xFFFFFFFF, got.tolist(), nt)
                np.testing.assert_array_equal(got, want, err_msg=str(nt))

    def test_threaded_mirror_handles_a_ragged_chunk(self):
        rng = np.random.default_rng(12)
        # n not divisible by nt, and n only just above k
        scores = rng.normal(size=577).astype(np.float32)
        want = mk.select_top_blocks_mirror(scores, 512)
        got = mk.select_top_blocks_threaded_mirror(scores, 512, nt=256)
        np.testing.assert_array_equal(got, want)

    def test_selection_fits_the_scratch_slot(self):
        self.assertGreaterEqual(mk.MAX_BLOCKS, 16384)
        self.assertEqual(dict(mk._SCRATCH_BLOCKS)["IDX_SEL"], mk.BLOCK_TOPK)
        self.assertEqual(mk.BLOCK_TOPK, 512)


class TestStatus(unittest.TestCase):
    def test_receipts_accumulate_and_reset(self):
        mk.qwen4_megakernel_status(reset=True)
        mk.record_megakernel_receipt(engaged=True, reason="engaged", phases=150)
        mk.record_megakernel_receipt(engaged=False, reason="batch 2")
        mk.record_megakernel_receipt(engaged=True, reason="engaged", phases=150,
                                     aborted=True)
        report = mk.qwen4_megakernel_status(reset=True)
        self.assertEqual(report["counts"], {"engaged": 2, "batch 2": 1})
        self.assertEqual(report["launches"], 2)
        self.assertEqual(report["phases"], 300)
        self.assertEqual(report["aborts"], 1)
        self.assertEqual(report["scratch_floats"], mk.SCRATCH_FLOATS)
        self.assertEqual(mk.qwen4_megakernel_status()["counts"], {})


if __name__ == "__main__":
    unittest.main()
