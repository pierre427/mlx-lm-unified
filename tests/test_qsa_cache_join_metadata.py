"""QSA cache joins: lane-aware metadata and offset/length separation.

The omlx `BatchQSAKVCache` join fixes (jundot/omlx#3369) closed two runtime
defects that are structural, not model-specific:

* metadata for a join picked by *first non-None* rather than by a lane-aware
  rule, so the outcome depended on which operand was ``self``; and
* ``merge`` reading a row's length off the **KV offset** while using it to
  slice the **indexer ledger**, which silently mis-slices whenever the two
  diverge (a trim, a rollback, a restored prompt cache).

Our tree carries no indexer position ledger (`index_position_ids`) and no
mRoPE lanes on these caches, so the first class can only appear as an
order-dependent *shape template* selection; the ragged text lanes below stand
in for the mixed geometry omlx hits with image rows.  The second class applies
directly: ``extend`` slices ``index_keys`` with ``_idx`` and ``merge`` slices
it with ``size()``, both KV quantities.

CPU only, tiny synthetic tensors, no model load.
"""

import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.models import qwen4_exp
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen4_exp import (
    BatchQSAKVCache,
    QSAKVCache,
    Qwen4ArraysCache,
)

H, DK, DV, DI = 2, 4, 4, 3


def _lane(length, seed, *, ledger=None):
    """Singleton QSA cache holding ``length`` tokens of distinct content."""
    cache = QSAKVCache()
    shape = (1, H, length, DK)
    base = float(seed)
    cache.keys = base + mx.arange(H * length * DK, dtype=mx.float32).reshape(shape)
    cache.values = -cache.keys
    cache.offset = length
    width = length if ledger is None else ledger
    cache.index_keys = base + mx.arange(
        width * DI, dtype=mx.float32
    ).reshape(1, width, DI)
    return cache


def _batch(lengths, seed, *, ledger=None):
    """Ragged batch QSA cache: rows left-padded to a common width."""
    width = max(lengths)
    padding = [width - length for length in lengths]
    cache = BatchQSAKVCache(padding)
    rows = len(lengths)
    base = float(seed)
    cache.keys = base + mx.arange(
        rows * H * width * DK, dtype=mx.float32
    ).reshape(rows, H, width, DK)
    cache.values = -cache.keys
    cache.offset = mx.array(lengths)
    cache.left_padding = mx.array(padding)
    cache._idx = width
    ledger_width = width if ledger is None else ledger
    cache.index_keys = base + mx.arange(
        rows * ledger_width * DI, dtype=mx.float32
    ).reshape(rows, ledger_width, DI)
    return cache


def _selection(index_keys, query, k):
    """Stand-in for the indexer's top-k over the raw key ledger."""
    scores = (index_keys * query).sum(axis=-1)
    order = mx.argsort(-scores, axis=-1)[..., :k]
    return sorted(int(v) for v in order[0].tolist())


class TestSingletonMergeJoins(unittest.TestCase):
    """QSAKVCache.merge -> BatchQSAKVCache.merge (the ragged-lane join)."""

    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def _assert_rows_match(self, batch, lanes):
        width = batch.index_keys.shape[1]
        for row, lane in enumerate(lanes):
            left = width - lane.offset
            self.assertTrue(
                mx.array_equal(
                    batch.index_keys[row, left:], lane.index_keys[0]
                ).item(),
                f"row {row} ledger does not match its unjoined lane",
            )
            self.assertTrue(
                mx.array_equal(
                    batch.keys[row, :, left:], lane.keys[0]
                ).item(),
                f"row {row} KV does not match its unjoined lane",
            )

    def test_merge_row_content_is_order_independent(self):
        """Ragged lanes join by their own length in either operand order."""
        lanes = [_lane(5, 10), _lane(3, 100), _lane(8, 1000)]
        forward = QSAKVCache.merge(lanes)
        backward = QSAKVCache.merge(list(reversed(lanes)))
        self._assert_rows_match(forward, lanes)
        self._assert_rows_match(backward, list(reversed(lanes)))
        self.assertEqual(forward.index_keys.shape, (3, 8, DI))
        self.assertEqual(backward.index_keys.shape, (3, 8, DI))

    def test_merge_with_an_unpopulated_lane_is_order_independent(self):
        """A lane that never ran a QSA layer must not decide the template.

        The legitimate shape of "no ledger" is a lane with no KV either: a
        fresh admission joining warm lanes. It contributes a zero row of the
        join width, in either operand order.
        """
        empty = QSAKVCache()
        populated = _lane(6, 7)
        first = QSAKVCache.merge([empty, populated])
        second = QSAKVCache.merge([populated, empty])
        self.assertEqual(first.index_keys.shape, second.index_keys.shape)
        self.assertTrue(
            mx.array_equal(first.index_keys[0], second.index_keys[1]).item()
        )
        self.assertTrue(
            mx.array_equal(first.index_keys[1], second.index_keys[0]).item()
        )
        self.assertEqual(first.index_keys.shape, (2, 6, DI))

    def test_merge_refuses_kv_with_no_ledger_at_all(self):
        """An absent ledger over live KV is the shortest possible ledger.

        Zero-filling it would hand the joined row raw keys the lane never
        wrote, and the width the join stamps then satisfies the next
        forward's desync check -- silent wrong state where the same lane
        expressed as a zero-width ledger array is refused.
        """
        unledgered = QSAKVCache()
        unledgered.keys = mx.zeros((1, H, 4, DK), dtype=mx.float32)
        unledgered.values = mx.zeros((1, H, 4, DK), dtype=mx.float32)
        unledgered.offset = 4
        with self.assertRaises(RuntimeError) as ctx:
            QSAKVCache.merge([unledgered, _lane(6, 7)])
        self.assertIn("ledger", str(ctx.exception).lower())

    def test_merge_indexer_selection_matches_the_unjoined_lane(self):
        """Top-k over the joined row selects the same logical tokens."""
        lanes = [_lane(4, 3), _lane(7, 50)]
        batch = QSAKVCache.merge(lanes)
        query = mx.array([[0.5, -1.25, 2.0]], dtype=mx.float32)
        width = batch.index_keys.shape[1]
        for row, lane in enumerate(lanes):
            left = width - lane.offset
            alone = _selection(lane.index_keys, query, 3)
            joined = _selection(
                batch.index_keys[row : row + 1, left:], query, 3
            )
            self.assertEqual(alone, joined, f"row {row} selection diverged")

    def test_merge_separates_kv_offset_from_indexer_length(self):
        """A lane whose ledger is shorter than its KV must not join silently.

        This is omlx #3369's item 4 on our shapes: ``merge`` slices
        ``index_keys`` with ``cache.size()``, a KV quantity.  A divergent lane
        (a restored prompt cache, a draft cycle) must either be refused or be
        joined by its *ledger* length -- never clamped into a row that claims
        KV columns the ledger does not describe.
        """
        short = _lane(8, 11, ledger=6)
        whole = _lane(8, 900)
        with self.assertRaises(Exception) as ctx:
            QSAKVCache.merge([short, whole])
        message = str(ctx.exception)
        self.assertIn(
            "ledger",
            message.lower(),
            f"join failed without naming the QSA ledger: {message!r}",
        )

    def test_merge_preserves_summary_identity_and_coverage(self):
        """R5 APC summaries: identity survives a join, coverage is the min."""
        identity = {
            "format_version": 1,
            "model_config_hash": "abc",
            "block_size": 2,
            "compress_ratio": 2,
            "producer_version": "test",
            "layer_id": "0",
            "complete_blocks": 0,
        }
        lanes = [_lane(6, 5), _lane(4, 60)]
        for lane in lanes:
            blocks = lane.offset // 2
            lane._qsa_pooled_keys = mx.ones((1, blocks, DI), dtype=mx.float32)
            lane._qsa_pooled_ratio = 2
            lane._qsa_summary_identity = dict(identity)
        with mock.patch.object(qwen4_exp, "_QSA_APC_SUMMARIES", True):
            batch = QSAKVCache.merge(lanes)
        self.assertIsNotNone(
            batch._qsa_summary_identity, "merge dropped the summary identity"
        )
        self.assertEqual(batch._qsa_summary_identity["layer_id"], "0")
        self.assertEqual(batch._qsa_summary_identity["complete_blocks"], 2)
        self.assertEqual(batch._qsa_pooled_keys.shape, (2, 2, DI))

    def test_merge_never_claims_coverage_it_did_not_join(self):
        """A join that closes no shared block carries no coverage claim.

        ``release_qsa_cycle`` keeps the identity at coverage 0 while ``merge``
        drops it outright.  The asymmetry is inert -- the next forward reads a
        missing identity and a missing pooled tensor the same way -- so pin the
        property that is load-bearing: a merged batch never carries a coverage
        number the joined rows do not support.
        """
        identity = {
            "format_version": 1,
            "model_config_hash": "abc",
            "block_size": 4,
            "compress_ratio": 4,
            "producer_version": "test",
            "layer_id": "0",
            "complete_blocks": 0,
        }
        lanes = [_lane(8, 5), _lane(2, 60)]
        for lane in lanes:
            blocks = lane.offset // 4
            lane._qsa_pooled_keys = mx.ones((1, blocks, DI), dtype=mx.float32)
            lane._qsa_pooled_ratio = 4
            lane._qsa_summary_identity = dict(identity)
        with mock.patch.object(qwen4_exp, "_QSA_APC_SUMMARIES", True):
            batch = QSAKVCache.merge(lanes)
        self.assertIsNone(batch._qsa_pooled_keys)
        self.assertEqual(
            0,
            (batch._qsa_summary_identity or {}).get("complete_blocks", 0),
            "merge kept a coverage claim with no pooled keys behind it",
        )


class TestBatchExtendJoins(unittest.TestCase):
    """BatchQSAKVCache.extend (the continuous-batching admission join)."""

    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def test_extend_is_order_independent_for_ragged_lanes(self):
        """Joining widths 6 and 4 gives the same rows in either order."""
        a_rows, b_rows = [6, 4], [3]
        forward = _batch(a_rows, 1)
        forward.extend(_batch(b_rows, 200))
        backward = _batch(b_rows, 200)
        backward.extend(_batch(a_rows, 1))
        self.assertEqual(forward.index_keys.shape[1], 6)
        self.assertEqual(backward.index_keys.shape[1], 6)
        self.assertEqual(forward.index_keys.shape[0], 3)
        self.assertEqual(backward.index_keys.shape[0], 3)
        # The lane that joined second in one order joined first in the other;
        # its row content must be identical either way.
        self.assertTrue(
            mx.array_equal(forward.index_keys[2], backward.index_keys[0]).item()
        )
        self.assertTrue(
            mx.array_equal(forward.index_keys[0], backward.index_keys[1]).item()
        )

    def test_extend_refuses_a_ledger_shorter_than_the_cursor(self):
        """``_idx`` is a KV quantity; the ledger width is the QSA one.

        omlx #3369 item 4 on the extend side.  A cache whose ledger runs short
        of its physical cursor (an armed shared-top-k draft cycle) must be
        refused by name, not concatenated into a row that is silently narrow.
        """
        short = _batch([5, 5], 1, ledger=3)
        with self.assertRaises(Exception) as ctx:
            short.extend(_batch([5], 400))
        message = str(ctx.exception)
        self.assertIn(
            "ledger",
            message.lower(),
            f"extend failed without naming the QSA ledger: {message!r}",
        )

    def test_extend_refuses_a_short_ledger_on_the_other_operand(self):
        """Order-independence of the refusal above."""
        whole = _batch([5], 400)
        with self.assertRaises(Exception) as ctx:
            whole.extend(_batch([5, 5], 1, ledger=3))
        self.assertIn("ledger", str(ctx.exception).lower())


class TestArraysCacheJoins(unittest.TestCase):
    """The GDN/PLE recurrent-state caches that join alongside QSA."""

    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def _state(self, rows, fill, *, slots=2, second=None):
        cache = Qwen4ArraysCache(slots)
        cache[0] = mx.full((rows, 2, 3), fill, dtype=mx.float32)
        cache[1] = second if second is not None else mx.full(
            (rows, 4), fill, dtype=mx.float32
        )
        return cache

    def test_merge_places_each_lane_in_its_own_row(self):
        lanes = [self._state(1, 1.0), self._state(1, 2.0), self._state(1, 3.0)]
        merged = Qwen4ArraysCache.merge(lanes)
        for row, lane in enumerate(lanes):
            self.assertTrue(
                mx.array_equal(merged[0][row : row + 1], lane[0]).item()
            )
            self.assertTrue(
                mx.array_equal(merged[1][row : row + 1], lane[1]).item()
            )

    def test_merge_slot_template_is_not_order_dependent(self):
        """A lane that has not reached a slot must not set its geometry.

        ``ArraysCache.merge`` picks the slot template by first non-None -- the
        selection rule omlx #3369 replaced.  Here it is only a shape/dtype
        template, so pin that an absent lane yields the same join in either
        order rather than an order-dependent shape.
        """
        blank = Qwen4ArraysCache(2)
        blank[0] = None
        blank[1] = None
        filled = self._state(1, 5.0)
        first = Qwen4ArraysCache.merge([blank, filled])
        second = Qwen4ArraysCache.merge([filled, blank])
        self.assertEqual(first[0].shape, second[0].shape)
        self.assertEqual(first[0].dtype, second[0].dtype)
        self.assertTrue(mx.array_equal(first[0][1:], second[0][:1]).item())

    def test_merge_refuses_mismatched_slot_geometry(self):
        """Two lanes whose state slot disagrees must fail, not broadcast."""
        wide = self._state(1, 1.0, second=mx.ones((1, 4), dtype=mx.float32))
        narrow = self._state(1, 2.0, second=mx.ones((1, 1), dtype=mx.float32))
        with self.assertRaises(Exception):
            merged = Qwen4ArraysCache.merge([wide, narrow])
            mx.eval(merged[1])

    def test_extract_marks_rollback_records_unusable_on_both_twins(self):
        """``Qwen4ArraysCache.extract`` overrides the base; keep the contract.

        The base ``extract`` stamps ``_rollback_invalid_reason`` because
        rollback closures capture whole-batch tensors.  A subclass that
        re-implements the method must keep that stamp, or a later trim failure
        on the extracted cache loses the reason it cannot replay.
        """
        base = ArraysCache(2)
        base[0] = mx.ones((2, 3), dtype=mx.float32)
        base[1] = mx.ones((2, 3), dtype=mx.float32)
        derived = self._state(2, 1.0)
        self.assertIsNotNone(base.extract(0)._rollback_invalid_reason)
        self.assertIsNotNone(
            derived.extract(0)._rollback_invalid_reason,
            "Qwen4ArraysCache.extract dropped the base class's rollback stamp",
        )


if __name__ == "__main__":
    unittest.main()
