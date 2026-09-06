# Copyright © 2026 Apple Inc.

"""Publish-back for the Flash-Next megakernel lane (CPU only).

``MegakernelDecoder.commit_to_caches`` is the reverse of ``seed_from_caches``:
it writes the kernel's live ledgers (QSA KV columns, raw index keys, pooled
summaries, GDN conv / recurrent state, PLE conv + n-gram state) back into the
request's stock cache objects so the server can publish the stock form to the
prefix cache.  These tests prove the copy is lossless -- a seed/commit round
trip leaves the stock caches byte-equal to a fresh prefill of the same tokens
-- and that only a clean close publishes.  No Metal launch: the ledgers are
filled directly, standing in for the kernel's decode.
"""

import os
import threading
import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models import qwen4_megakernel_runtime as runtime
from mlx_lm.models.qwen4_megakernel import (
    CONV_DIM,
    CONV_KERNEL,
    GDN_KEY_DIM,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HC_HIDDEN,
    HEAD_DIM,
    IDX_COMPRESS,
    IDX_HEAD_DIM,
    N_KV_HEADS,
    PLE_STATE_LEN,
)
from mlx_lm.models.qwen4_exp import (
    ArraysCache,
    QSAIndexer,
    QSAKVCache,
    Qwen4ArraysCache,
    TextModelArgs,
    _qsa_summary_with_coverage,
)

# Layer plan for the fixture: two full-attention layers, two linear-attention
# (GDN) layers, and the second GDN layer also carries the PLE state.
LAYER_TYPES = [
    "full_attention", "linear_attention", "linear_attention", "full_attention",
]
ATTN_LAYERS = [0, 3]
GDN_LAYERS = [1, 2]
PLE_LAYERS = [2]
TOTAL = 64  # ledger columns; must divide by IDX_COMPRESS


def _args():
    return TextModelArgs(
        hidden_size=16,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        vocab_size=64,
        max_position_embeddings=TOTAL,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=IDX_HEAD_DIM,
        indexer_budget=IDX_HEAD_DIM,
        indexer_compress_ratio=IDX_COMPRESS,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
        },
    )


def _shell(indexers, *, position):
    """A MegakernelDecoder with only the fields commit / seed touch."""
    dec = runtime.MegakernelDecoder.__new__(runtime.MegakernelDecoder)
    dec.max_context = TOTAL
    dec.total = TOTAL
    dec.pooled_stride = TOTAL // IDX_COMPRESS
    dec.layer_types = list(LAYER_TYPES)
    dec.layers = list(range(len(LAYER_TYPES)))
    dec.attn_layers = list(ATTN_LAYERS)
    dec.gdn_layers = list(GDN_LAYERS)
    dec.ple_layers = list(PLE_LAYERS)
    dec.attn_slot = {i: r for r, i in enumerate(ATTN_LAYERS)}
    n_attn = len(ATTN_LAYERS)
    n_gdn = len(GDN_LAYERS)
    dec.kv = mx.zeros((2, n_attn * N_KV_HEADS, TOTAL, HEAD_DIM), mx.bfloat16)
    dec.idxl = mx.zeros(
        (n_attn * (TOTAL + dec.pooled_stride), IDX_HEAD_DIM), mx.bfloat16)
    dec._raw_rows = n_attn * TOTAL
    dec.cs = mx.zeros((n_gdn, CONV_KERNEL - 1, CONV_DIM), mx.bfloat16)
    dec.rec = mx.zeros(
        (n_gdn, GDN_VALUE_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM), mx.float32)
    dec.pconv = mx.zeros((PLE_STATE_LEN, HC_HIDDEN), mx.bfloat16)
    dec.position = position
    dec._state_lock = threading.RLock()
    dec._in_flight = False
    dec._in_flight_owner = None
    dec._pending = None
    dec._pending_width = 0
    dec._pending_position = None
    dec._pending_owner = None
    dec._poisoned_reason = None
    dec._indexers = indexers
    return dec


def _indexer_of(indexers):
    return lambda index: indexers[index]


def _fresh_caches(indexers):
    caches = []
    for i, kind in enumerate(LAYER_TYPES):
        if kind == "linear_attention":
            caches.append(Qwen4ArraysCache(size=4) if i in PLE_LAYERS
                          else ArraysCache(size=2))
        else:
            caches.append(QSAKVCache(indexers[i].summary_identity))
    return caches


def _random_source(length, *, seed=0):
    """One random history of ``length`` columns, sliceable to any prefix."""
    rng = np.random.default_rng(seed)
    src = {"attn": {}, "gdn": {}, "ple": {}}
    for i in ATTN_LAYERS:
        src["attn"][i] = {
            "keys": mx.array(rng.standard_normal(
                (1, N_KV_HEADS, length, HEAD_DIM)).astype("float32")).astype(mx.bfloat16),
            "values": mx.array(rng.standard_normal(
                (1, N_KV_HEADS, length, HEAD_DIM)).astype("float32")).astype(mx.bfloat16),
            "raw": mx.array(rng.standard_normal(
                (1, length, IDX_HEAD_DIM)).astype("float32")).astype(mx.bfloat16),
        }
    for i in GDN_LAYERS:
        src["gdn"][i] = {
            "conv": mx.array(rng.standard_normal(
                (1, CONV_KERNEL - 1, CONV_DIM)).astype("float32")).astype(mx.bfloat16),
            "rec": mx.array(rng.standard_normal(
                (1, GDN_VALUE_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM)).astype("float32")),
        }
    for i in PLE_LAYERS:
        src["ple"][i] = {
            "conv": mx.array(rng.standard_normal(
                (1, PLE_STATE_LEN, HC_HIDDEN)).astype("float32")).astype(mx.bfloat16),
            "ngram": mx.array(rng.integers(0, 63, size=(1, 2)), dtype=mx.int64),
        }
    mx.eval(*[a for group in src.values() for row in group.values()
              for a in row.values()])
    return src


def _caches_from_source(indexers, src, length):
    """Stock caches filled to ``length`` from ``src`` as a prefill would leave."""
    caches = _fresh_caches(indexers)
    for i in ATTN_LAYERS:
        cache = caches[i]
        raw = src["attn"][i]["raw"][:, :length]
        cache.update_index_keys(raw)
        cache.update_and_fetch(
            src["attn"][i]["keys"][:, :, :length],
            src["attn"][i]["values"][:, :, :length])
        n_blocks = length // IDX_COMPRESS
        if n_blocks:
            starts = mx.arange(n_blocks) * IDX_COMPRESS
            pooled = indexers[i]._pool_blocks(
                raw[:, :n_blocks * IDX_COMPRESS], starts)
            cache._qsa_pooled_keys = mx.contiguous(pooled)
            cache._qsa_pooled_ratio = IDX_COMPRESS
            cache._qsa_summary_identity = _qsa_summary_with_coverage(
                indexers[i].summary_identity, n_blocks)
            cache._qsa_summary_restored = False
    for i in GDN_LAYERS:
        caches[i][0] = src["gdn"][i]["conv"]
        caches[i][1] = src["gdn"][i]["rec"]
    for i in PLE_LAYERS:
        caches[i][2] = src["ple"][i]["conv"]
        caches[i][3] = src["ple"][i]["ngram"]
    mx.eval(*[c for cache in caches for c in
              (getattr(cache, "keys", None), getattr(cache, "values", None),
               getattr(cache, "index_keys", None))
              if c is not None])
    return caches


def _prefill_caches(indexers, length, *, seed=0):
    return _caches_from_source(indexers, _random_source(length, seed=seed), length)


def _assert_caches_equal(test, a, b):
    for i in ATTN_LAYERS:
        ka, va = a[i].keys_and_values()
        kb, vb = b[i].keys_and_values()
        test.assertTrue(mx.array_equal(ka, kb), f"keys layer {i}")
        test.assertTrue(mx.array_equal(va, vb), f"values layer {i}")
        test.assertEqual(int(a[i].offset), int(b[i].offset))
        test.assertTrue(
            mx.array_equal(a[i].index_keys, b[i].index_keys),
            f"index_keys layer {i}")
        pa, pb = a[i]._qsa_pooled_keys, b[i]._qsa_pooled_keys
        if pa is None or pb is None:
            test.assertIs(pa, pb, f"pooled presence layer {i}")
        else:
            test.assertTrue(mx.array_equal(pa, pb), f"pooled layer {i}")
            test.assertEqual(a[i]._qsa_pooled_ratio, b[i]._qsa_pooled_ratio)
            test.assertEqual(
                a[i]._qsa_summary_identity, b[i]._qsa_summary_identity,
                f"summary identity layer {i}")
    for i in GDN_LAYERS:
        test.assertTrue(mx.array_equal(a[i][0], b[i][0]), f"gdn conv layer {i}")
        test.assertTrue(mx.array_equal(a[i][1], b[i][1]), f"gdn rec layer {i}")
    for i in PLE_LAYERS:
        test.assertTrue(mx.array_equal(a[i][2], b[i][2]), f"ple conv layer {i}")
        test.assertTrue(mx.array_equal(a[i][3], b[i][3]), f"ple ngram layer {i}")


class CommitToCachesRoundTrip(unittest.TestCase):
    def setUp(self):
        args = _args()
        self.indexers = {i: QSAIndexer(args, i) for i in ATTN_LAYERS}
        mx.eval(*[p for idx in self.indexers.values()
                  for p in idx.parameters().values() if isinstance(p, mx.array)])

    def test_seed_then_commit_reproduces_a_fresh_prefill(self):
        # A fresh prefill of N tokens (ground truth), seeded into the ledgers
        # and committed into an empty cache list, must reproduce it byte-for-
        # byte across QSA KV / index keys / pooled summary and GDN + PLE state.
        length = 40
        ref = _prefill_caches(self.indexers, length, seed=1)
        dec = _shell(self.indexers, position=length)
        info = dec.seed_from_caches(ref, indexer_of=_indexer_of(self.indexers))
        self.assertEqual(info["position"], length)
        ple_embed = Qwen4ArraysCache(size=4)
        ple_embed[3] = ref[PLE_LAYERS[0]][3]
        out = _fresh_caches(self.indexers)
        receipt = dec.commit_to_caches(
            out, indexer_of=_indexer_of(self.indexers),
            ple_embedding_cache=ple_embed)
        self.assertEqual(receipt["position"], length)
        self.assertEqual(receipt["attention"], len(ATTN_LAYERS))
        self.assertEqual(receipt["pooled_blocks"], (length // IDX_COMPRESS) * len(ATTN_LAYERS))
        _assert_caches_equal(self, out, ref)

    def test_delta_publish_matches_the_full_prefill(self):
        # The production path: seed the shorter prefill, inject the emitted
        # span's faithful columns into the ledgers, then publish the delta into
        # the already-seeded cache. The result must equal a fresh prefill of
        # the whole span.
        short, full = 20, 40
        src = _random_source(full, seed=2)
        seed_caches = _caches_from_source(self.indexers, src, short)
        ref_full = _caches_from_source(self.indexers, src, full)
        dec = _shell(self.indexers, position=short)
        dec.seed_from_caches(seed_caches, indexer_of=_indexer_of(self.indexers))
        # Inject the [short:full] columns from the faithful full prefill.
        for slot, i in enumerate(ATTN_LAYERS):
            base = slot * N_KV_HEADS
            kf, vf = ref_full[i].keys_and_values()
            dec.kv[0, base:base + N_KV_HEADS, short:full] = kf[0, :, short:full]
            dec.kv[1, base:base + N_KV_HEADS, short:full] = vf[0, :, short:full]
            rbase = slot * TOTAL
            dec.idxl[rbase + short:rbase + full] = ref_full[i].index_keys[0, short:full]
        for slot, i in enumerate(GDN_LAYERS):
            dec.cs[slot] = ref_full[i][0][0].astype(mx.bfloat16)
            dec.rec[slot] = ref_full[i][1][0].astype(mx.float32)
        dec.pconv[:] = ref_full[PLE_LAYERS[0]][2][0].astype(mx.bfloat16)
        dec.position = full
        mx.eval(dec.kv, dec.idxl, dec.cs, dec.rec, dec.pconv)
        ple_embed = Qwen4ArraysCache(size=4)
        ple_embed[3] = ref_full[PLE_LAYERS[0]][3]
        dec.commit_to_caches(
            seed_caches, indexer_of=_indexer_of(self.indexers),
            ple_embedding_cache=ple_embed)
        _assert_caches_equal(self, seed_caches, ref_full)

    def test_published_summary_identity_and_coverage_match(self):
        length = 40
        ref = _prefill_caches(self.indexers, length, seed=3)
        dec = _shell(self.indexers, position=length)
        dec.seed_from_caches(ref, indexer_of=_indexer_of(self.indexers))
        out = _fresh_caches(self.indexers)
        dec.commit_to_caches(out, indexer_of=_indexer_of(self.indexers))
        n_blocks = length // IDX_COMPRESS
        for i in ATTN_LAYERS:
            identity = out[i]._qsa_summary_identity
            expected = _qsa_summary_with_coverage(
                self.indexers[i].summary_identity, n_blocks)
            self.assertEqual(identity, expected)
            self.assertEqual(identity["complete_blocks"], n_blocks)
            self.assertEqual(out[i]._qsa_pooled_keys.shape, (1, n_blocks, IDX_HEAD_DIM))

    def test_commit_refuses_a_pending_launch(self):
        dec = _shell(self.indexers, position=8)
        dec._pending = object()
        dec._pending_position = 8
        with self.assertRaises(RuntimeError):
            dec.commit_to_caches(
                _fresh_caches(self.indexers),
                indexer_of=_indexer_of(self.indexers))

    def test_commit_refuses_a_poisoned_decoder(self):
        dec = _shell(self.indexers, position=8)
        dec._poisoned_reason = "device abort"
        with self.assertRaises(runtime.MegakernelDeviceAbort):
            dec.commit_to_caches(
                _fresh_caches(self.indexers),
                indexer_of=_indexer_of(self.indexers))

    def test_commit_refuses_an_in_flight_launch(self):
        dec = _shell(self.indexers, position=8)
        dec._in_flight = True
        with self.assertRaises(RuntimeError):
            dec.commit_to_caches(
                _fresh_caches(self.indexers),
                indexer_of=_indexer_of(self.indexers))


class LaneCloseGates(unittest.TestCase):
    """The lane publishes only at a clean close, and only when enabled."""

    class _StubDecoder:
        def __init__(self):
            self.position = 12
            self._pending = None
            self._poisoned_reason = None
            self.ple_layers = []
            self.calls = 0

        def commit_to_caches(self, caches, *, indexer_of=None,
                             ple_embedding_cache=None):
            self.calls += 1
            return {"published_ok": True, "position": self.position,
                    "attention": 2, "gdn": 2, "ple": 0, "pooled_blocks": 6}

    def _lane(self, *, publish, monkeypatch_env):
        from mlx_lm import megakernel_lane as ML

        os.environ["MLX_QWEN4_MEGAKERNEL_PUBLISH"] = "1" if publish else "0"
        lane = ML.MegakernelLane.__new__(ML.MegakernelLane)
        lane.decoder = self._StubDecoder()
        lane.status = {}
        lane.prompt_cache = []
        lane.indexer_of = lambda i: None
        lane.ple_cache = None
        lane.publish = ML.megakernel_publish_enabled()
        lane.tokens = 12
        lane.closed = False
        lane._publish_receipt = {"published": False, "reason": "close not reached"}
        return lane, ML

    def setUp(self):
        self._prev = os.environ.get("MLX_QWEN4_MEGAKERNEL_PUBLISH")
        # The lane acquires the module lock in __init__; here we bypass it, so
        # take/release the lock around close() to mirror the real path.
        from mlx_lm import megakernel_lane as ML
        ML._LANE_LOCK.acquire(blocking=False)

    def tearDown(self):
        from mlx_lm import megakernel_lane as ML
        if ML._LANE_LOCK.locked():
            try:
                ML._LANE_LOCK.release()
            except RuntimeError:
                pass
        if self._prev is None:
            os.environ.pop("MLX_QWEN4_MEGAKERNEL_PUBLISH", None)
        else:
            os.environ["MLX_QWEN4_MEGAKERNEL_PUBLISH"] = self._prev

    def test_clean_close_publishes_when_enabled(self):
        lane, ML = self._lane(publish=True, monkeypatch_env=True)
        lane.close(error=False)
        self.assertEqual(lane.decoder.calls, 1)
        self.assertTrue(lane.status["publish_back"]["published"])

    def test_error_close_does_not_publish(self):
        lane, ML = self._lane(publish=True, monkeypatch_env=True)
        ML._LANE_LOCK.acquire(blocking=False)
        lane.close(error=True)
        self.assertEqual(lane.decoder.calls, 0)
        self.assertFalse(lane.status["publish_back"]["published"])
        self.assertIn("clean", lane.status["publish_back"]["reason"])

    def test_disabled_close_does_not_publish(self):
        lane, ML = self._lane(publish=False, monkeypatch_env=True)
        lane.close(error=False)
        self.assertEqual(lane.decoder.calls, 0)
        self.assertFalse(lane.status["publish_back"]["published"])
        self.assertIn("disabled", lane.status["publish_back"]["reason"])

    def test_poisoned_close_does_not_publish(self):
        lane, ML = self._lane(publish=True, monkeypatch_env=True)
        lane.decoder._poisoned_reason = "abort"
        lane.close(error=False)
        self.assertEqual(lane.decoder.calls, 0)
        self.assertFalse(lane.status["publish_back"]["published"])


class ServerPublishableGate(unittest.TestCase):
    def test_gate_reads_the_receipt(self):
        from mlx_lm.server import _megakernel_cache_publishable

        self.assertIsNone(
            _megakernel_cache_publishable({"publish_back": {"published": True}}))
        self.assertEqual(
            _megakernel_cache_publishable(
                {"publish_back": {"published": False, "reason": "off"}}),
            "off")
        self.assertEqual(
            _megakernel_cache_publishable({}),
            "the lane recorded no publish-back")


if __name__ == "__main__":
    unittest.main()
