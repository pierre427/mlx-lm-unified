"""Invariant tests for batched generation (CPU-only, no model).

These cover four foundational bugs in continuous batching:

  * H1 -- empty logits-processor lanes must always be an (iterable) ``[]`` and
    never ``None``, so the ``GenerationBatch._step`` consumer that iterates
    ``for processor in self.logits_processors[e]`` is always safe even for a
    mixed batch like ``[[], [proc]]``.
  * H2 -- ``GenerationBatch.filter`` must keep ``samplers`` /
    ``logits_processors`` index-aligned with ``uids`` even when every lane is
    falsy, otherwise a later ``extend`` binds lanes to the wrong sampler.
  * L3 -- per-lane empty processor lists must be independent objects, never a
    single shared list.
  * M1 -- a decoded token that merges body bytes with a marker
    (``"}</tool_call>"``) must route the body to tool text, not content.

Everything uses tiny synthetic tensors, fake samplers/processors and a fake
model -- no weights are loaded and the GPU is never touched.
"""

import importlib
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import (
    GenerationBatch,
    PromptProcessingBatch,
    StopSequenceMatcher,
    TextStateMachine,
)
from mlx_lm.server import _segment_by_state

generate_module = importlib.import_module("mlx_lm.generate")


class FakeModel(nn.Module):
    """A model that returns all-zero logits (argmax -> token 0)."""

    def __init__(self, vocab: int = 8):
        super().__init__()
        self.vocab = vocab

    def __call__(self, inputs, cache=None):
        # inputs: (batch, seq) -> logits: (batch, seq, vocab)
        return mx.zeros((inputs.shape[0], inputs.shape[1], self.vocab))


class CountingStateCache:
    """Minimal cache whose state accessor records maintenance evaluation."""

    def __init__(self):
        self.state_reads = 0

    @property
    def state(self):
        self.state_reads += 1
        return mx.array(self.state_reads)


def const_sampler(token: int):
    """A sampler that always returns ``token`` (shape (1,)) so we can detect
    which lane it was applied to."""

    def _sampler(logprobs):
        return mx.array([token], dtype=mx.uint32)

    return _sampler


def _argmax_fallback(logprobs):
    return mx.argmax(logprobs, axis=-1)


def _make_gen_batch(model, uids, samplers, logits_processors, fallback):
    n = len(uids)
    inputs = mx.zeros((n,), dtype=mx.uint32)
    tokens = [[1] for _ in range(n)]
    stop_matchers = [StopSequenceMatcher() for _ in range(n)]
    max_tokens = [100] * n
    return GenerationBatch(
        model,
        list(uids),
        inputs,
        [],  # no caches; FakeModel ignores them
        tokens,
        list(samplers),
        fallback,
        list(logits_processors),
        stop_matchers,
        max_tokens,
    )


class TestBatchProcessorInvariant(unittest.TestCase):
    def setUp(self):
        # Restore in tearDown: the default device is process-global, so leaking
        # cpu here makes every later test in the session see a device_info()
        # without max_recommended_working_set_size.
        self._prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.model = FakeModel()

    def tearDown(self):
        mx.set_default_device(self._prev_device)

    def test_plain_decode_periodically_materializes_cache_state(self):
        cache = CountingStateCache()
        generation = generate_module.generate_step(
            mx.array([1]),
            self.model,
            max_tokens=3,
            prompt_cache=[cache],
            compiled_decode=False,
        )

        with patch.object(generate_module, "CACHE_STATE_EVAL_INTERVAL", 2):
            list(generation)

        # The existing maintenance branch runs at decode indices 0 and 2.
        self.assertEqual(cache.state_reads, 2)

    def test_batch_decode_folds_cache_state_into_periodic_async_eval(self):
        cache = CountingStateCache()
        inputs = mx.zeros((1,), dtype=mx.uint32)
        batch = GenerationBatch(
            self.model,
            [0],
            inputs,
            [cache],
            [[1]],
            [None],
            _argmax_fallback,
            [[]],
            [StopSequenceMatcher()],
            [100],
        )

        with patch.object(generate_module, "CACHE_STATE_EVAL_INTERVAL", 2):
            batch._step()

        self.assertEqual(cache.state_reads, 1)

    # ----- H1: mixed no-processor + processor lanes step without TypeError ---
    def test_mixed_processor_lanes_step_without_typeerror(self):
        ran = {"count": 0}

        def proc(tokens, logits):
            ran["count"] += 1
            return logits

        # Lane 0 has no processors ([]), lane 1 has one. The consumer iterates
        # each lane, so a None lane here would raise TypeError.
        batch = _make_gen_batch(
            self.model,
            uids=[0, 1],
            samplers=[None, None],
            logits_processors=[[], [proc]],
            fallback=_argmax_fallback,
        )
        # __init__ already ran one _step; run another to be sure.
        tokens, _ = batch._step()
        self.assertEqual(len(tokens), 2)
        # The processor on lane 1 actually ran (twice: __init__ + explicit step).
        self.assertGreaterEqual(ran["count"], 1)

    def test_extend_normalizes_empty_lanes_to_list_not_none(self):
        # A batch that came through PromptProcessingBatch.extend from an
        # empty-processor batch must have [] lanes, never None -- otherwise the
        # GenerationBatch consumer raises TypeError.
        def proc(tokens, logits):
            return logits

        a = PromptProcessingBatch.empty(self.model, _argmax_fallback)
        a.uids = [0]
        a.tokens = [[1]]
        a.samplers = [None]
        a.logits_processors = [[proc]]
        a.max_tokens = [100]
        a.stop_matchers = [StopSequenceMatcher()]
        a.prompt_cache = []

        b = PromptProcessingBatch.empty(self.model, _argmax_fallback)
        b.uids = [1]
        b.tokens = [[1]]
        b.samplers = []  # no per-lane samplers provided
        b.logits_processors = []  # no per-lane processors provided
        b.max_tokens = [100]
        b.stop_matchers = [StopSequenceMatcher()]
        b.prompt_cache = []

        a.extend(b)

        self.assertEqual(len(a.logits_processors), len(a.uids))
        self.assertEqual(a.logits_processors, [[proc], []])
        self.assertNotIn(None, a.logits_processors)
        for lane in a.logits_processors:
            self.assertIsInstance(lane, list)

    # ----- H2: after a lane finishes, each lane keeps ITS OWN sampler --------
    def test_filter_keeps_samplers_index_aligned(self):
        fallback = _argmax_fallback  # zeros -> token 0
        sampler_b = const_sampler(5)

        # Start with two all-None sampler lanes (the falsy case that the old
        # `if any(...)` guard failed to filter).
        batch = _make_gen_batch(
            self.model,
            uids=[0, 1],
            samplers=[None, None],
            logits_processors=[[], []],
            fallback=fallback,
        )
        # Lane 0 finishes -> filter down to lane 1 only.
        batch.filter([1])
        self.assertEqual(len(batch.samplers), len(batch.uids))
        self.assertEqual(len(batch.samplers), 1)

        # A new lane with its OWN (distinct) sampler is added.
        new = _make_gen_batch(
            self.model,
            uids=[2],
            samplers=[sampler_b],
            logits_processors=[[]],
            fallback=fallback,
        )
        batch.extend(new)

        # samplers stay index-aligned with uids: [None (fallback), sampler_b].
        self.assertEqual(len(batch.samplers), len(batch.uids))
        self.assertEqual(batch.uids, [1, 2])
        self.assertIsNone(batch.samplers[0])
        self.assertIs(batch.samplers[1], sampler_b)

        # Step and confirm each lane used its own sampler: lane uid=1 falls back
        # to argmax (token 0), lane uid=2 uses sampler_b (token 5). Under the
        # H2 bug the misaligned lists would give token 0 for BOTH lanes.
        batch._step()
        self.assertEqual(batch._next_tokens.tolist(), [0, 5])

    # ----- L3: per-lane empty processor lists are independent objects --------
    def test_empty_processor_lanes_are_independent_objects(self):
        # PromptProcessingBatch.filter's empty branch must build a fresh [] per
        # lane; a shared [[]] * n would alias every lane.
        a = PromptProcessingBatch.empty(self.model, _argmax_fallback)
        a.uids = [0, 1, 2]
        a.tokens = [[1], [1], [1]]
        a.samplers = [None, None, None]
        a.logits_processors = [[], [], []]
        a.max_tokens = [100, 100, 100]
        a.stop_matchers = [StopSequenceMatcher() for _ in range(3)]
        a.prompt_cache = []

        a.filter([0, 1])
        self.assertEqual(len(a.logits_processors), 2)
        self.assertIsNot(a.logits_processors[0], a.logits_processors[1])

        # Mutating one lane must not leak into the other.
        a.logits_processors[0].append("x")
        self.assertEqual(a.logits_processors[1], [])

        # Same independence via GenerationBatch.filter empty->[] normalization
        # path through extend.
        b = _make_gen_batch(
            self.model,
            uids=[10, 11],
            samplers=[None, None],
            logits_processors=[[], []],
            fallback=_argmax_fallback,
        )
        b.filter([0, 1])
        self.assertIsNot(b.logits_processors[0], b.logits_processors[1])

    # ----- M1: a marker merged with body bytes routes body to tool text ------
    def _tool_state_machine(self):
        return TextStateMachine(
            transitions={
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )

    def test_merged_marker_documents_whole_chunk_bug(self):
        # Baseline: feeding the whole merged chunk to step() attributes the body
        # "}" to the FINAL state ("normal") -- this is exactly the M1 leak.
        sm = self._tool_state_machine()
        state = sm.make_state("tool")
        _, clean_text, final_state = TextStateMachine.step(state, "}</tool_call>")
        self.assertEqual(clean_text, "}")
        self.assertEqual(final_state, "normal")  # "}" would leak to content

    def test_merged_marker_routes_body_to_tool_text(self):
        sm = self._tool_state_machine()
        state = sm.make_state("tool")
        _, segments = _segment_by_state(state, "}</tool_call>")

        # The body byte is attributed to "tool"; no body byte is labelled
        # "normal"; a transition marker to "normal" is present.
        self.assertIn(("}", "tool"), segments)
        self.assertNotIn(("}", "normal"), segments)
        self.assertTrue(any(s == "normal" for _, s in segments))

        # Drive the server's per-segment routing and assert the split.
        tool_text = ""
        content = ""
        prev_state = "tool"
        tool_calls = []
        for seg_text, seg_state in segments:
            if seg_state == "tool":
                tool_text += seg_text
            elif seg_state == "normal":
                if prev_state == "tool":
                    tool_calls.append(tool_text)
                    tool_text = ""
                content += seg_text
            prev_state = seg_state

        self.assertEqual(tool_calls, ["}"])  # full body kept in the tool call
        self.assertEqual(content, "")  # nothing leaked to content

    def test_body_before_and_after_marker_split_correctly(self):
        # "{}" body in tool, then exit, then "hi" in normal, all in one chunk.
        sm = self._tool_state_machine()
        state = sm.make_state("tool")
        _, segments = _segment_by_state(state, "{}</tool_call>hi")

        tool_text = ""
        content = ""
        prev_state = "tool"
        tool_calls = []
        for seg_text, seg_state in segments:
            if seg_state == "tool":
                tool_text += seg_text
            elif seg_state == "normal":
                if prev_state == "tool":
                    tool_calls.append(tool_text)
                    tool_text = ""
                content += seg_text
            prev_state = seg_state

        self.assertEqual(tool_calls, ["{}"])
        self.assertEqual(content, "hi")


if __name__ == "__main__":
    unittest.main()
