"""Re-arm / re-entry regression tests for the reasoning-termination guards.

Covers three confirmed bugs, driven entirely by synthetic token/marker
sequences (no model, no GPU):

  M2a  RunawayGuard: a natural </think> close must reset the per-channel
       counters so a later re-opened <think> starts fresh and does not trip on
       stale counts.
  M2b  make_reasoning_budget: a second <think> channel must be re-armed and
       budgeted, not left permanently unguarded after the first </think>.
  M3   Both guards: a tool-call-start marker closes the guarded channel for
       counting purposes WITHOUT forcing </think> into the tool body.

RunawayGuard lives at the repo root (imported as ``runaway_guard``); make it
importable regardless of where pytest is invoked from.
"""
import os
import sys
import unittest

import mlx.core as mx

_HERE = os.path.abspath(os.path.dirname(__file__))
_RUNAWAY_ROOT = next(
    (
        parent
        for parent in (
            os.path.abspath(os.path.join(_HERE, *([os.pardir] * depth)))
            for depth in range(1, 5)
        )
        if os.path.isfile(os.path.join(parent, "runaway_guard.py"))
    ),
    None,
)
if _RUNAWAY_ROOT is None:
    raise RuntimeError("could not locate the superproject runaway_guard.py")
if _RUNAWAY_ROOT not in sys.path:
    sys.path.insert(0, _RUNAWAY_ROOT)

from mlx_lm.sample_utils import make_reasoning_budget  # noqa: E402
from runaway_guard import (  # noqa: E402
    THINK_CLOSE,
    THINK_OPEN,
    TOOL_CALL_START,
    RunawayGuard,
)


class _FakeTok:
    """Minimal tokenizer stub: RunawayGuard only ever calls ``decode``. Return
    empty text so the line-repetition detector never fires; these tests isolate
    the budget / channel-tracking logic."""

    def decode(self, ids):
        return ""


class TestRunawayGuardReArm(unittest.TestCase):
    # High check_every keeps the cycle/line detectors dormant so trips are
    # driven purely by the token budget — the counter we care about here.
    def _guard(self, think_budget):
        return RunawayGuard(
            _FakeTok(), think_budget=think_budget, check_every=10 ** 6
        )

    def test_close_then_reopen_starts_fresh_no_early_trip(self):
        # M2a: 8 tokens, natural close, re-open, 8 more. Budget is 10, so a
        # correctly-reset second channel (8 < 10) must NOT trip. With the bug,
        # 8 + 8 = 16 >= 10 would trip on stale counts.
        guard = self._guard(think_budget=10)
        fillers_a = list(range(100, 108))  # 8 non-marker ids
        for t in fillers_a:
            self.assertFalse(guard.observe([t], True))

        self.assertFalse(guard.observe([THINK_CLOSE], True))  # natural close
        self.assertFalse(guard.observe([THINK_OPEN], True))   # re-open

        fillers_b = list(range(200, 208))  # 8 more in the fresh channel
        for t in fillers_b:
            self.assertFalse(
                guard.observe([t], True),
                msg="second channel tripped on stale counts (M2a)",
            )
        self.assertIsNone(guard.tripped)

    def test_second_channel_is_still_guarded(self):
        # M2a/M2b analog: after close+reopen the guard must still enforce the
        # budget on the second channel (it is not permanently disabled).
        guard = self._guard(think_budget=10)
        self.assertFalse(guard.observe([THINK_CLOSE], True))
        self.assertFalse(guard.observe([THINK_OPEN], True))

        tripped = False
        for i, t in enumerate(range(300, 300 + 10)):
            tripped = guard.observe([t], True)
        self.assertTrue(tripped, msg="second channel not budgeted (M2)")
        self.assertEqual(guard.tripped, "budget")

    def test_tool_call_from_think_does_not_force_close(self):
        # M3: enter a tool call from think; the tool body must not be counted
        # and the guard must never signal a force-close (which the caller turns
        # into an injected </think> mid-tool).
        guard = self._guard(think_budget=5)
        for t in range(400, 403):  # 3 think tokens, below budget
            self.assertFalse(guard.observe([t], True))

        # Tool-call-start closes guarding even though the caller still reports
        # in_think=True (no </think> was emitted).
        self.assertFalse(guard.observe([TOOL_CALL_START], True))

        # A long tool body (well past the budget of 5) must never force-close.
        for t in range(500, 520):
            self.assertFalse(
                guard.observe([t], True),
                msg="guard force-closed mid-tool-body (M3)",
            )
        self.assertIsNone(guard.tripped)


class TestReasoningBudgetReArm(unittest.TestCase):
    """make_reasoning_budget re-arm / tool-exit, driven like the existing
    speculative-rewind test: feed the growing committed sequence and check
    whether the processor forces the close token."""

    OPEN = 50
    CLOSE = 99
    VOCAB = 128

    def _forced(self, processor, tokens):
        out = processor(mx.array(tokens, mx.uint32), mx.zeros((1, self.VOCAB)))
        return bool(
            out[0, self.CLOSE].item() == 0.0
            and out[0, 0].item() == float("-inf")
        )

    def test_second_think_channel_is_budgeted(self):
        # M2b: with an explicit think_open, the first channel closes under
        # budget, then a fresh OPEN must re-arm so the second channel is
        # budgeted and forces the close once it overruns.
        proc = make_reasoning_budget(
            think_close=self.CLOSE,
            max_think_tokens=4,
            think_open=self.OPEN,
            check_every=10 ** 6,
        )
        # First channel: OPEN, a, b, c (3 ids < 4), then CLOSE — no force.
        seq = [self.OPEN, 1, 2, 3, self.CLOSE]
        for i in range(1, len(seq) + 1):
            self.assertFalse(self._forced(proc, seq[:i]))

        # Second channel: OPEN then d, e, f, g. The 4th id overruns the budget
        # and must force the close — proving the channel was re-armed (M2b).
        seq2 = seq + [self.OPEN, 4, 5, 6]
        self.assertFalse(self._forced(proc, seq2))         # 3 ids: still fine
        self.assertTrue(
            self._forced(proc, seq2 + [7]),                # 4th id: overrun
            msg="second think channel was left unguarded (M2b)",
        )

    def test_tool_call_exit_stops_counting_and_does_not_force(self):
        # M3: a tool-call-start id (passed via reasoning_exit_ids) leaves the
        # channel WITHOUT forcing the close; the tool body that follows is not
        # counted, so no </think> is injected mid-tool.
        exit_id = 70
        proc = make_reasoning_budget(
            think_close=self.CLOSE,
            max_think_tokens=4,
            reasoning_exit_ids={exit_id},  # think_open=None: starts in-think
            check_every=10 ** 6,
        )
        # 3 think tokens, then exit via the tool-call marker, then a long body.
        seq = [1, 2, 3, exit_id] + list(range(10, 30))
        self.assertFalse(
            self._forced(proc, seq),
            msg="tool body counted; close forced mid-tool (M3)",
        )

    def test_control_budget_still_forces_without_exit(self):
        # Control for the M3 test: with no exit marker, the same-length run of
        # in-think tokens DOES force — so the no-force above is due to the
        # tool-exit, not an over-generous budget.
        proc = make_reasoning_budget(
            think_close=self.CLOSE,
            max_think_tokens=4,
            check_every=10 ** 6,  # think_open=None: starts in-think
        )
        self.assertTrue(self._forced(proc, [1, 2, 3, 4, 5]))


if __name__ == "__main__":
    unittest.main()
