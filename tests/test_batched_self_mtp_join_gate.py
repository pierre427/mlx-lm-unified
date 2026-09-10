"""Mid-flight JOIN contracts for the batched persistent self-MTP engine.

The forced-acceptance cache gate (test_batched_self_mtp_qwen4.py) only ever
attaches lanes into a FRESH batch (``attach(None, lanes)``).  These tests pin
the other membership path: ``attach_self_mtp_lanes(model, active_batch,
[joining])`` merging a lane into an already-decoded batch.

Two contracts:

* ``test_join_preserves_lane_caches_exactly`` -- immediately after the join,
  the joining row extracted from the batch equals the lane's standalone
  caches on every VALID position.  The comparison slices each single-lane
  cache to its own offset first: ``QSAKVCache.state`` returns the raw
  step-allocated buffer (256-wide after a short prefill) while ``extract``
  returns the exact valid region, so a raw ``state``-vs-``state`` comparison
  reports a shape mismatch on every QSA slot even when the join is perfect.

* ``test_joined_lane_matches_solo_generation`` -- the joined lane's full
  greedy trace equals the single-lane ``self_mtp_generate_step`` oracle,
  with only documented near-tie flips admitted (same contract as the digest
  test's in-from-start lanes).

The CPU class runs the tiny random-weight qwen4_exp model; the production
class reruns both bodies against the artifact named by
``MLX_BATCHED_MTP_GATE_MODEL`` on real hardware.
"""

import os
import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.hybrid_speculative import (
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
    require_only_near_tie_greedy_flips,
    self_mtp_generate_step,
)
from mlx_lm.models.qwen4_exp import QSAKVCache
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.utils import load
from test_batched_self_mtp_qwen4 import _tiny_qwen4_model, _tree_arrays

_ROWS = (
    (10, (1, 2, 3, 4, 5)),
    (20, (6, 7, 8, 9, 10, 11)),
    (30, (12, 13, 14, 15, 16, 17, 18)),
    (40, (19, 20, 21, 22, 23)),
)
_JOIN_UID = 50
_JOIN_PROMPT = (24, 25, 26, 27, 28, 29)


def _valid_arrays(cache):
    """A cache's per-position content, sliced to its own valid region.

    ``QSAKVCache`` step-allocates its K/V buffers, so ``state`` is wider than
    the sequence; a batch ``extract`` is exact-width.  Slicing both sides to
    ``offset`` makes the two representations comparable value for value.
    """
    if isinstance(cache, QSAKVCache):
        arrays = [
            cache.keys[..., : cache.offset, :],
            cache.values[..., : cache.offset, :],
        ]
        if cache.index_keys is not None:
            arrays.append(cache.index_keys[:, : cache.offset])
        return arrays
    return list(_tree_arrays(cache.state))


class TestTinyQwen4Join(unittest.TestCase):
    tolerance = dict(rtol=0.0, atol=0.0)

    @classmethod
    def setUpClass(cls):
        cls._previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        mx.random.seed(41)
        cls.model = _tiny_qwen4_model()

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._previous_device)

    def _lane(self, uid, prompt, maximum=8):
        return prepare_self_mtp_lane(
            mx.array(prompt, mx.uint32),
            self.model,
            uid=uid,
            max_tokens=maximum,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=LaneRNG(100 + uid),
            num_draft=2,
            sampling_temp=0.0,
            sampling_top_p=1.0,
            sampling_top_k=0,
            sampling_min_p=0.0,
            accept_rule="residual",
            logits_processors=[],
            prefill_step_size=4,
            share_qsa_indices=False,
        )

    def _active_batch(self, traces=None):
        """One committed cycle over four lanes, then detach the last row."""
        prepared = [(uid, *self._lane(uid, list(p))) for uid, p in _ROWS]
        ntoks = {}
        for uid, _lane, first in prepared:
            if traces is not None:
                traces[uid] = [(int(first.token), first.logprobs)]
            ntoks[uid] = 1
        batch = attach_self_mtp_lanes(
            self.model, None, [lane for _uid, lane, _first in prepared]
        )
        proposal = propose_batched_self_mtp(self.model, batch)
        for lane, outputs in zip(batch.lanes, proposal.outputs):
            if traces is not None:
                traces[lane.uid].extend(
                    (int(o.token), o.logprobs) for o in outputs
                )
            ntoks[lane.uid] += len(outputs)
        commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[len(row) for row in proposal.outputs],
            terminal=[False] * len(batch.lanes),
        )
        batch, _ = detach_self_mtp_lanes(self.model, batch, [3])
        return batch, ntoks

    def assert_group_equal(self, standalone, extracted, label):
        self.assertEqual(len(standalone), len(extracted))
        for slot, (want, got) in enumerate(zip(standalone, extracted)):
            want_arrays = _valid_arrays(want)
            got_arrays = _valid_arrays(got)
            self.assertEqual(
                len(want_arrays),
                len(got_arrays),
                f"{label} slot {slot} ({type(want).__name__}) array count",
            )
            for index, (left, right) in enumerate(zip(want_arrays, got_arrays)):
                self.assertEqual(
                    left.shape,
                    right.shape,
                    f"{label} slot {slot} ({type(want).__name__}) array "
                    f"{index} valid-region shape",
                )
                np.testing.assert_allclose(
                    np.asarray(left.astype(mx.float32)),
                    np.asarray(right.astype(mx.float32)),
                    err_msg=(
                        f"{label} slot {slot} ({type(want).__name__}) array "
                        f"{index} content changed across the join"
                    ),
                    **self.tolerance,
                )

    def test_join_preserves_lane_caches_exactly(self):
        batch, _ntoks = self._active_batch()
        joining, _first = self._lane(_JOIN_UID, list(_JOIN_PROMPT))
        # Snapshot BEFORE the attach consumes the detached lane's caches.
        standalone_target = [
            (cache, list(_valid_arrays(cache)))
            for cache in joining.caches.target
        ]
        standalone_draft = [
            (cache, list(_valid_arrays(cache))) for cache in joining.caches.draft
        ]
        mx.eval(
            [arrays for _c, arrays in standalone_target],
            [arrays for _c, arrays in standalone_draft],
        )

        batch = attach_self_mtp_lanes(self.model, batch, [joining])
        join_row = len(batch.lanes) - 1
        self.assertEqual(batch.lanes[join_row].uid, _JOIN_UID)

        for group, standalone, label in (
            (batch.caches.target, standalone_target, "target"),
            (batch.caches.draft, standalone_draft, "draft"),
        ):
            for slot, (cache, want_arrays) in enumerate(standalone):
                extracted = group[slot].extract(join_row)
                got_arrays = _valid_arrays(extracted)
                self.assertEqual(len(want_arrays), len(got_arrays))
                for index, (left, right) in enumerate(
                    zip(want_arrays, got_arrays)
                ):
                    self.assertEqual(
                        left.shape,
                        right.shape,
                        f"{label} slot {slot} ({type(cache).__name__}) "
                        f"array {index} valid-region shape",
                    )
                    np.testing.assert_allclose(
                        np.asarray(left.astype(mx.float32)),
                        np.asarray(right.astype(mx.float32)),
                        err_msg=(
                            f"{label} slot {slot} ({type(cache).__name__}) "
                            f"array {index} content changed across the join"
                        ),
                        **self.tolerance,
                    )

        # Geometry invariants for the joined row on every batch KV slot.
        for group, label in (
            (batch.caches.target, "target"),
            (batch.caches.draft, "draft"),
        ):
            for slot, cache in enumerate(group):
                offset = getattr(cache, "offset", None)
                if not isinstance(offset, mx.array):
                    continue
                row_offset = int(offset[join_row].item())
                row_pad = int(cache.left_padding[join_row].item())
                self.assertEqual(
                    row_pad + row_offset,
                    cache._idx,
                    f"{label} slot {slot}: joined row is not right-justified "
                    "at the shared cursor",
                )
                ledger = getattr(cache, "index_keys", None)
                if ledger is not None:
                    self.assertEqual(
                        ledger.shape[1],
                        cache._idx,
                        f"{label} slot {slot}: QSA ledger width diverged "
                        "from the cursor across the join",
                    )

    def test_joined_lane_matches_solo_generation(self):
        traces = {}
        batch, ntoks = self._active_batch(traces)
        joining, first = self._lane(_JOIN_UID, list(_JOIN_PROMPT))
        traces[_JOIN_UID] = [(int(first.token), first.logprobs)]
        ntoks[_JOIN_UID] = 1
        batch = attach_self_mtp_lanes(self.model, batch, [joining])

        while batch.lanes:
            proposal = propose_batched_self_mtp(self.model, batch)
            mx.eval(
                [o.logprobs for row in proposal.outputs for o in row]
            )
            terminal = [
                ntoks[lane.uid] + len(outputs) >= lane.max_tokens
                for lane, outputs in zip(batch.lanes, proposal.outputs)
            ]
            for lane, outputs in zip(batch.lanes, proposal.outputs):
                traces[lane.uid].extend(
                    (int(o.token), o.logprobs) for o in outputs
                )
                ntoks[lane.uid] += len(outputs)
            commit_batched_self_mtp(
                batch,
                proposal,
                emitted_counts=[len(row) for row in proposal.outputs],
                terminal=terminal,
            )
            leaving = [i for i, value in enumerate(terminal) if value]
            if leaving:
                batch, _ = detach_self_mtp_lanes(self.model, batch, leaving)

        prompts = dict(_ROWS)
        prompts[_JOIN_UID] = _JOIN_PROMPT
        for uid in (10, 20, 30, _JOIN_UID):
            reference = []
            for token, logprobs, _from_draft in self_mtp_generate_step(
                mx.array(list(prompts[uid]), mx.uint32),
                self.model,
                num_draft=2,
                max_tokens=8,
                prefill_step_size=4,
                sampling_temp=0.0,
                sampling_top_p=1.0,
                sampling_top_k=0,
                sampling_min_p=0.0,
                accept_rule="residual",
                persistent_mtp=True,
                lane_rng=LaneRNG(100 + uid),
            ):
                reference.append((int(token), mx.reshape(logprobs, (-1,))))
            candidate = traces[uid]
            self.assertEqual(len(reference), len(candidate), f"uid {uid}")
            reference_tokens = [token for token, _ in reference]
            candidate_tokens = [token for token, _ in candidate]
            if reference_tokens == candidate_tokens:
                continue
            first_flip = next(
                index
                for index, (want, got) in enumerate(
                    zip(reference_tokens, candidate_tokens)
                )
                if want != got
            )
            diagnostic = require_only_near_tie_greedy_flips(
                mx.stack(
                    [
                        mx.reshape(logprobs, (-1,))
                        for _, logprobs in reference[: first_flip + 1]
                    ]
                ),
                mx.stack(
                    [
                        mx.reshape(logprobs, (-1,))
                        for _, logprobs in candidate[: first_flip + 1]
                    ]
                ),
            )
            self.assertEqual(
                diagnostic.near_tie_flip_positions, (first_flip,), f"uid {uid}"
            )
            self.assertFalse(
                diagnostic.non_near_tie_flip_positions,
                f"uid {uid}: joined-lane divergence is outside the near-tie "
                "band",
            )


@unittest.skipUnless(
    os.environ.get("MLX_BATCHED_MTP_GATE_MODEL"),
    "set MLX_BATCHED_MTP_GATE_MODEL to the production Qwen3.8 artifact",
)
class TestProductionQwen38Join(TestTinyQwen4Join):
    """Rerun both join contracts against the production Qwen4 artifact.

    The cache-equality body stays exact (production join must be a pure data
    movement); the trace body already admits only documented near-tie flips.
    """

    @classmethod
    def setUpClass(cls):
        cls._previous_device = mx.default_device()
        cls.model, _ = load(os.environ["MLX_BATCHED_MTP_GATE_MODEL"])
        if getattr(cls.model, "mtp", None) is None:
            raise AssertionError("production join gate model has no MTP head")

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._previous_device)


if __name__ == "__main__":
    unittest.main()
