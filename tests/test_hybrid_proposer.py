"""CPU-only tests for the hybrid retrieval-proposer prototype.

Pure token-id logic: no model is loaded and mlx is never imported. The proto and
its dependency (``prompt_lookup``) are loaded directly from the ``mlx_lm``
directory by path, so the heavyweight ``mlx_lm`` package ``__init__`` (which
imports mlx/transformers) is bypassed entirely.
"""
import importlib.util
import os
import sys
import unittest

_MLX_LM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlx_lm"
)


def _load(module_name: str):
    if _MLX_LM_DIR not in sys.path:
        sys.path.insert(0, _MLX_LM_DIR)  # lets the proto's `import prompt_lookup` work
    path = os.path.join(_MLX_LM_DIR, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_proto = _load("hybrid_proposer")
_pl = _load("prompt_lookup")

DatastoreProposer = _proto.DatastoreProposer
HybridProposer = _proto.HybridProposer
HybridProposerStats = _proto.HybridProposerStats
make_hybrid_proposer = _proto.make_hybrid_proposer
NgramProposer = _pl.NgramProposer
SuffixAutomatonProposer = _pl.SuffixAutomatonProposer
plan_proposal_around_verify_cliff = _pl.plan_proposal_around_verify_cliff
snap_proposal_around_verify_cliff = _pl.snap_proposal_around_verify_cliff


class TestDatastoreProposer(unittest.TestCase):
    def test_cliff_aware_span_policy_is_opt_in_safe(self):
        proposal = list(range(32))
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:8]), proposal[:7]
        )
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:14]), proposal[:7]
        )
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:15]), proposal[:15]
        )
        self.assertEqual(
            snap_proposal_around_verify_cliff(proposal[:7], pending_rows=2),
            proposal[:6],
        )
        self.assertEqual(plan_proposal_around_verify_cliff(8, 20), 15)
        self.assertEqual(plan_proposal_around_verify_cliff(10, 12), 7)
        self.assertEqual(plan_proposal_around_verify_cliff(16, 20), 16)
        self.assertEqual(plan_proposal_around_verify_cliff(7, 20), 7)

    def test_returns_datastore_continuation_for_seeded_prefix(self):
        # Datastore holds the pattern "10 20 30 40 50". A running sequence whose
        # tail is "20 30" should retrieve the datastore continuation "40 50 ...".
        ds = DatastoreProposer([[10, 20, 30, 40, 50, 60]], min_match=2, window=16)
        seq = [7, 7, 20, 30]  # tail "20 30" occurs in the datastore, not in seq
        self.assertEqual(ds.propose(seq, max_span=3, prompt_len=len(seq)), [40, 50, 60])

    def test_no_match_returns_empty(self):
        ds = DatastoreProposer([[10, 20, 30, 40]], min_match=2, window=16)
        # Tail "98 99" never appears in the datastore.
        self.assertEqual(ds.propose([1, 98, 99], max_span=4, prompt_len=3), [])

    def test_min_match_gate(self):
        ds = DatastoreProposer([[10, 20, 30, 40]], min_match=3, window=16)
        # Only a length-2 suffix ("20 30") matches -> below the min_match=3 gate.
        self.assertEqual(ds.propose([5, 20, 30], max_span=4, prompt_len=3), [])
        # A length-3 suffix ("10 20 30") clears the gate.
        self.assertEqual(ds.propose([10, 20, 30], max_span=4, prompt_len=3), [40])

    def test_does_not_cross_document_boundary(self):
        # "30 40" ends doc A; its continuation must NOT leak into doc B ("40 99").
        ds = DatastoreProposer([[10, 20, 30, 40], [40, 99, 98]], min_match=2, window=16)
        self.assertEqual(ds.propose([1, 30, 40], max_span=5, prompt_len=3), [])

    def test_flat_sequence_datastore(self):
        # A bare token list is accepted as a single document.
        ds = DatastoreProposer([10, 20, 30, 40, 50], min_match=2, window=16)
        self.assertEqual(ds.propose([9, 30, 40], max_span=2, prompt_len=3), [50])

    def test_adaptive_span_caps_at_match_length(self):
        ds = DatastoreProposer(
            [[10, 20, 30, 40, 50, 60]], min_match=2, window=16, adaptive_span=True
        )
        # Match "20 30" has length 2 -> at most 2 draft tokens even with max_span=5.
        self.assertEqual(ds.propose([1, 20, 30], max_span=5, prompt_len=3), [40, 50])
        # Longer match "10 20 30" (length 3) -> up to 3 draft tokens.
        self.assertEqual(ds.propose([10, 20, 30], max_span=5, prompt_len=3), [40, 50, 60])

    def test_observe_is_stateless_noop(self):
        ds = DatastoreProposer([[10, 20, 30, 40]], min_match=2, window=16)
        for t in (1, 2, 3):
            ds.observe(t)  # must not raise or alter results
        self.assertEqual(ds.propose([9, 20, 30], max_span=2, prompt_len=3), [40])

    def test_exact_document_dedup_and_footprint_stats(self):
        ds = DatastoreProposer(
            [[10, 20, 30, 40], [10, 20, 30, 40], [20, 30, 50]],
            min_match=2,
        )
        self.assertEqual(ds.stats.input_documents, 3)
        self.assertEqual(ds.stats.documents, 2)
        self.assertEqual(ds.stats.duplicate_documents, 1)
        self.assertEqual(ds.stats.tokens, 7)
        self.assertGreater(ds.stats.automaton_states, 1)
        self.assertGreater(ds.stats.automaton_transitions, 0)
        self.assertGreater(ds.footprint_bytes(), 0)
        # The frozen suffix automaton sequence is the single token store.
        self.assertFalse(hasattr(ds, "_store"))

    def test_dedup_can_be_disabled_for_experimental_weighting(self):
        ds = DatastoreProposer(
            [[10, 20, 30], [10, 20, 30]], deduplicate=False, min_match=2
        )
        self.assertEqual(ds.stats.documents, 2)
        self.assertEqual(ds.stats.duplicate_documents, 0)
        self.assertEqual(ds.stats.tokens, 6)

    def test_adaptive_span_scale_and_floor(self):
        ds = DatastoreProposer(
            [[10, 20, 30, 40, 50, 60]],
            min_match=2,
            adaptive_span=True,
            span_scale=0.5,
            min_span=2,
        )
        # A three-token match scales to ceil(1.5), then the two-token floor.
        self.assertEqual(ds.propose([10, 20, 30], max_span=5, prompt_len=3), [40, 50])

    def test_constructor_rejects_unsafe_tokens_and_budgets(self):
        for docs in ([[-1, 2]], [[-2, 2]], [[True, 2]], [["1", 2]]):
            with self.assertRaises((TypeError, ValueError)):
                DatastoreProposer(docs)
        with self.assertRaises(ValueError):
            DatastoreProposer([[1, 2, 3]], max_tokens=2)
        with self.assertRaises(ValueError):
            DatastoreProposer([[1]], span_scale=0)
        with self.assertRaises(ValueError):
            DatastoreProposer([[1]], sep=0)

    def test_generator_input_is_consumed_once(self):
        seen = []

        def documents():
            for doc in ([10, 20, 30], [40, 50, 60]):
                seen.append(doc[0])
                yield doc

        ds = DatastoreProposer(documents(), min_match=2)
        self.assertEqual(seen, [10, 40])
        self.assertEqual(ds.stats.documents, 2)

    def test_from_texts_build_api(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.last_add_special_tokens = add_special_tokens
                return [ord(char) for char in text]

        tokenizer = Tokenizer()
        ds = DatastoreProposer.from_texts(
            ["abc", "bcd"], tokenizer, min_match=2, window=8
        )
        self.assertFalse(tokenizer.last_add_special_tokens)
        self.assertEqual(
            ds.propose([0, ord("a"), ord("b")], max_span=2, prompt_len=3),
            [ord("c")],
        )


class TestHybridProposer(unittest.TestCase):
    def _seed(self, proposer, tokens):
        """Mimic generate.py: feed every committed token via observe()."""
        for t in tokens:
            proposer.observe(t)

    def test_primary_wins_when_it_has_a_match(self):
        # In-context repetition "1 2 3 4 ... 1 2 3" -> the primary (default
        # min_match=3) retrieves "4" from the current sequence; the datastore
        # (disjoint tokens) must not be consulted.
        datastore = [[91, 92, 93, 94]]
        hp = make_hybrid_proposer(datastore, primary="suffix_automaton")
        seq = [1, 2, 3, 4, 7, 1, 2, 3]
        self._seed(hp, seq)
        prop = hp.propose(seq, max_span=3, prompt_len=len(seq))
        self.assertTrue(prop, "primary should have found the in-context continuation")
        self.assertEqual(prop[0], 4)

    def test_falls_back_to_datastore_when_primary_misses(self):
        # No in-context repetition, but the tail "20 30" lives in the datastore.
        datastore = [[10, 20, 30, 40, 50]]
        hp = make_hybrid_proposer(datastore, primary="suffix_automaton", min_match=2)
        seq = [5, 6, 7, 8, 20, 30]
        self._seed(hp, seq)
        # Sanity: the in-context primary alone finds nothing here.
        primary_only = SuffixAutomatonProposer()
        for t in seq:
            primary_only.observe(t)
        self.assertEqual(primary_only.propose(seq, 3, len(seq)), [])
        # Hybrid recovers the proposal from the datastore fallback.
        self.assertEqual(hp.propose(seq, max_span=3, prompt_len=len(seq)), [40, 50])

    def test_empty_when_neither_matches(self):
        datastore = [[91, 92, 93]]
        hp = make_hybrid_proposer(datastore, primary="ngram", min_match=2)
        seq = [1, 2, 3, 4, 5]
        self._seed(hp, seq)
        self.assertEqual(hp.propose(seq, max_span=3, prompt_len=len(seq)), [])

    def test_observe_forwarded_to_both(self):
        datastore = [[10, 20, 30, 40]]
        ds = DatastoreProposer(datastore, min_match=2, window=16)
        primary = SuffixAutomatonProposer()
        hp = HybridProposer(primary, ds)
        before = len(primary.sam)
        hp.observe(42)
        self.assertEqual(len(primary.sam), before + 1)  # primary saw the token

    def test_source_telemetry_counts_primary_datastore_and_miss(self):
        datastore = [[10, 20, 30, 40, 50]]
        hp = make_hybrid_proposer(datastore, min_match=2)

        primary_seq = [1, 2, 3, 4, 1, 2, 3]
        self._seed(hp, primary_seq)
        self.assertEqual(hp.propose(primary_seq, 2, len(primary_seq)), [4, 1])

        datastore_seq = [91, 92, 20, 30]
        # Use a fresh proposer so the previous sequence cannot create a primary hit.
        hp2 = make_hybrid_proposer(datastore, min_match=2)
        self._seed(hp2, datastore_seq)
        self.assertEqual(hp2.propose(datastore_seq, 2, len(datastore_seq)), [40, 50])
        self.assertEqual(hp2.propose([97, 98, 99], 2, 3), [])
        self.assertEqual(
            hp2.stats.as_dict(),
            {
                "calls": 2,
                "primary_hits": 0,
                "datastore_hits": 1,
                "misses": 1,
                "primary_proposed": 0,
                "datastore_proposed": 2,
                "datastore_suppressed": 0,
            },
        )

    def test_primary_kwargs_match_baseline_configuration(self):
        hp = make_hybrid_proposer(
            [[10, 20, 30]],
            primary="suffix_automaton",
            primary_kwargs={"min_match": 5, "max_lookback": 9},
        )
        self.assertEqual(hp.primary.min_match, 5)
        self.assertEqual(hp.primary.max_lookback, 9)

    def test_primary_hit_cooldown_suppresses_datastore_interruptions(self):
        datastore = DatastoreProposer([[20, 30, 40, 50]], min_match=2)
        primary = SuffixAutomatonProposer()
        hp = HybridProposer(primary, datastore, datastore_cooldown=2)
        hot_seq = [1, 2, 3, 4, 1, 2, 3]
        self._seed(hp, hot_seq)
        self.assertTrue(hp.propose(hot_seq, 2, len(hot_seq)))

        # The datastore would match both calls, but two post-primary misses are
        # intentionally decoded plainly so a hot in-context stream can recover.
        fallback_seq = [91, 20, 30]
        self._seed(hp, fallback_seq)
        self.assertEqual(hp.datastore.propose(fallback_seq, 2, 3), [40, 50])
        self.assertEqual(hp.propose(fallback_seq, 2, 3), [])
        self.assertEqual(hp.propose(fallback_seq, 2, 3), [])
        self.assertEqual(hp.propose(fallback_seq, 2, 3), [40, 50])
        self.assertEqual(hp.stats.datastore_suppressed, 2)

        with self.assertRaises(ValueError):
            HybridProposer(primary, datastore, datastore_cooldown=-1)

    def test_primary_only_warmup_delays_datastore_fallback(self):
        class MissPrimary:
            def observe(self, token):
                pass

            def propose(self, seq, max_span, prompt_len):
                return []

        datastore = DatastoreProposer([[20, 30, 40, 50]], min_match=2)
        hp = HybridProposer(
            MissPrimary(), datastore, datastore_warmup_tokens=2
        )
        prompt_len = 3
        self.assertEqual(hp.propose([91, 20, 30], 2, prompt_len), [])
        self.assertEqual(hp.propose([91, 20, 30, 20], 2, prompt_len), [])
        self.assertEqual(hp.propose([91, 20, 30, 20, 30], 2, prompt_len), [40, 50])
        self.assertEqual(hp.stats.datastore_suppressed, 2)

        with self.assertRaises(ValueError):
            HybridProposer(
                MissPrimary(), datastore, datastore_warmup_tokens=-1
            )


if __name__ == "__main__":
    unittest.main()
