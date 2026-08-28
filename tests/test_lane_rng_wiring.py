# Copyright © 2026 Apple Inc.

"""The per-lane random key as the product actually uses it (CPU-only).

``tests/test_lane_rng.py`` proves the primitive and the engine. This file
covers the wiring the engine is useless without:

  * ``stream_generate`` forwards the lane key into ``self_mtp_generate_step``;
  * the server builds one lane per request and hands it over;
  * the lane's position round-trips through the APC sidecar, so a resumed
    request continues its own stream instead of falling back to the global one.
"""

import types
import unittest
from queue import Queue

import mlx.core as mx

from mlx_lm.apc import AutomaticPrefixCache, MTPAPCSidecar
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import LaneRNG
from mlx_lm.server import (
    CompletionRequest,
    ResponseGenerator,
    _make_lane_rng,
    _self_mtp_config,
)


def _key_list(key):
    return [int(v) for v in key.tolist()]


class TestLaneRNGFactory(unittest.TestCase):
    """``_make_lane_rng``: seeded, resumed, or forked -- never shared."""

    @classmethod
    def setUpClass(cls):
        cls._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._device)

    @staticmethod
    def _root():
        return LaneRNG(1234)

    def test_explicit_seed_reproduces_the_lane(self):
        args = types.SimpleNamespace(seed=7)
        a = _make_lane_rng(args, self._root())
        b = _make_lane_rng(args, self._root())
        self.assertEqual(_key_list(a.key), _key_list(b.key))

    def test_two_requests_do_not_share_a_key(self):
        root = self._root()
        args = types.SimpleNamespace(seed=None)
        a = _make_lane_rng(args, root)
        b = _make_lane_rng(args, root)
        self.assertNotEqual(_key_list(a.key), _key_list(b.key))

    def test_unseeded_lanes_are_reproducible_for_a_given_root(self):
        args = types.SimpleNamespace(seed=None)
        first = [_make_lane_rng(args, r) for r in (self._root(),)][0]
        second = [_make_lane_rng(args, r) for r in (self._root(),)][0]
        self.assertEqual(_key_list(first.key), _key_list(second.key))

    def test_a_sidecar_resumes_the_carried_stream(self):
        lane = LaneRNG(99)
        lane.next_key()
        lane.next_key()
        sidecar = MTPAPCSidecar(
            state=None, covered_tokens=4, rng_key=lane.key, rng_draws=lane.draws
        )
        resumed = _make_lane_rng(types.SimpleNamespace(seed=None), self._root(), sidecar)
        self.assertEqual(_key_list(resumed.key), _key_list(lane.key))
        self.assertEqual(resumed.draws, 2)
        # And it continues rather than repeating: the next draw of the resumed
        # lane is the next draw of the original lane.
        self.assertEqual(_key_list(resumed.next_key()), _key_list(lane.next_key()))

    def test_a_sidecar_without_a_key_falls_back_to_a_fresh_lane(self):
        sidecar = MTPAPCSidecar(state=None, covered_tokens=4)
        lane = _make_lane_rng(types.SimpleNamespace(seed=None), self._root(), sidecar)
        self.assertIsInstance(lane, LaneRNG)

    def test_config_carries_the_lane(self):
        lane = LaneRNG(5)
        config = _self_mtp_config(
            _mtp_args(),
            _mtp_cli(),
            types.SimpleNamespace(mtp=object()),
            lane_rng=lane,
        )
        self.assertIs(config["lane_rng"], lane)


@unittest.skip(
    "Lane-key wiring is disabled in generate.py (c1bcb51) after it took the "
    "service down: the server builds the LaneRNG on the request thread while "
    "generation runs on a worker thread, and mlx-lm binds generation_stream to "
    "its creating thread, so the first draw raised 'There is no Stream(gpu, 0) "
    "in current thread'. These assertions are correct for the intended "
    "behaviour -- un-skip them together with the wiring, once the lane is built "
    "on the generation thread (see Rapid-MLX's _run_on_step_thread discipline, "
    "wiki lessons/single-thread-tests-cannot-see-cross-thread-state.md)."
)
class TestStreamGenerateForwardsTheLane(unittest.TestCase):
    """``stream_generate`` is the only door into the MTP engine from serving."""

    @classmethod
    def setUpClass(cls):
        cls._device = mx.default_device()
        mx.set_default_device(mx.cpu)
        from mlx_lm.models.qwen3_5 import TextModel

        from test_qwen3_5_mtp import tiny_args

        mx.random.seed(0)
        cls.model = TextModel(tiny_args())
        mx.eval(cls.model.parameters())
        cls.prompt = [int(t) for t in mx.random.randint(0, 64, (12,)).tolist()]

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._device)

    def _text(self, *, lane_rng, global_seed):
        from mlx_lm.generate import stream_generate

        mx.random.seed(global_seed)
        self_mtp = {
            "num_draft": 2,
            "sampling_temp": 0.8,
            "top_p": 0.9,
            "top_k": 8,
            "persistent": True,
            "state_out": {},
        }
        if lane_rng is not None:
            self_mtp["lane_rng"] = lane_rng
        return [
            int(r.token)
            for r in stream_generate(
                self.model,
                _tiny_tokenizer(),
                mx.array(self.prompt),
                max_tokens=10,
                self_mtp=self_mtp,
            )
        ]

    def test_the_lane_key_survives_a_different_global_seed(self):
        a = self._text(lane_rng=LaneRNG(31337), global_seed=1)
        b = self._text(lane_rng=LaneRNG(31337), global_seed=987_654)
        self.assertEqual(a, b)

    def test_without_a_lane_the_global_seed_still_decides(self):
        # The control: the tokens above are equal because the lane key is
        # forwarded, not because this tiny model is deterministic.
        a = self._text(lane_rng=None, global_seed=1)
        b = self._text(lane_rng=None, global_seed=987_654)
        self.assertNotEqual(a, b)

    def test_two_lanes_differ(self):
        a = self._text(lane_rng=LaneRNG(1), global_seed=5)
        b = self._text(lane_rng=LaneRNG(2), global_seed=5)
        self.assertNotEqual(a, b)


class _HFStub:
    """The little of a HuggingFace tokenizer that ``TokenizerWrapper`` reads."""

    eos_token_id = 0
    bos_token = None
    chat_template = None

    def get_vocab(self):
        return {}

    def convert_ids_to_tokens(self, ids):
        return [f"<{i}>" for i in ids]

    def decode(self, tokens, **kwargs):
        return "".join(f"<{int(t)}>" for t in tokens)

    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 64 for c in text]


def _tiny_tokenizer():
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    # eos ids that the tiny random model will not emit, so every run decodes
    # the full ``max_tokens``.
    return TokenizerWrapper(_HFStub(), eos_token_ids={-1})


def _mtp_cli(**overrides):
    cli = types.SimpleNamespace(
        self_mtp=True,
        self_mtp_num_draft=1,
        self_mtp_persistent=True,
        self_mtp_rate_gate=True,
        self_mtp_share_qsa_indices=False,
        self_mtp_share_qsa_indices_min_prompt_tokens=16384,
        self_mtp_window_size=0,
        self_mtp_window_sink_size=4,
        self_mtp_window_min_prompt_tokens=32768,
        self_mtp_apc_retain_min_prompt_tokens=0,
        kv_bits=None,
        kv_key_bits=None,
        kv_value_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
        prefill_step_size=8,
        chat_template_args={},
        prompt_cache_bytes=None,
    )
    for key, value in overrides.items():
        setattr(cli, key, value)
    return cli


def _mtp_args(**overrides):
    args = types.SimpleNamespace(
        n=1,
        max_tokens=4,
        model=types.SimpleNamespace(draft="default_model", model=None, adapter=None),
        sampling=types.SimpleNamespace(
            temperature=0.8,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            xtc_probability=0.0,
            xtc_threshold=0.1,
        ),
        logits=types.SimpleNamespace(
            logit_bias=None,
            repetition_penalty=0.0,
            repetition_context_size=20,
            presence_penalty=0.0,
            presence_context_size=20,
            frequency_penalty=0.0,
            frequency_context_size=20,
        ),
        stop_words=[],
        top_logprobs=0,
        logprobs=False,
        seed=None,
        num_draft_tokens=0,
        prompt_lookup_ngram=0,
        chat_template_kwargs=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class _Detokenizer:
    last_segment = ""

    def add_token(self, token):
        self.last_segment = f"<{token}>"

    def finalize(self):
        pass


class _Tokenizer:
    has_thinking = False
    has_tool_calling = False
    has_chat_template = False
    tool_parser = None
    eos_token_ids = [7]

    def __init__(self, tokens):
        self._tokens = list(tokens)

    def encode(self, text, add_special_tokens=True):
        return list(self._tokens)

    @property
    def detokenizer(self):
        return _Detokenizer()


class _MTPModel:
    """A model that only has to look MTP-capable and own a KV cache."""

    mtp = object()

    def make_cache(self):
        return [KVCache()]


class TestServeSingleWiresTheLane(unittest.TestCase):
    """``_serve_single`` builds a lane, hands it over, and stores its position."""

    @classmethod
    def setUpClass(cls):
        cls._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._device)

    def _generator(self, tokens):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = types.SimpleNamespace(
            model=_MTPModel(),
            tokenizer=_Tokenizer(tokens),
            draft_model=None,
            is_batchable=True,
            model_key=("stub", None, None),
            cli_args=_mtp_cli(),
        )
        generator.prompt_cache = AutomaticPrefixCache()
        generator._state_machine_cache = {}
        generator._lane_rng_root = LaneRNG(2026)
        generator._is_distributed = False
        return generator

    def _serve(self, generator, args, captured, *, covered_key=None, draws=0):
        """Run ``_serve_single`` with a scripted ``stream_generate``."""
        import mlx_lm.server as server_module

        def fake_stream_generate(*, model, prompt, prompt_cache, self_mtp, **kwargs):
            captured.append(self_mtp)

            def _gen():
                # Fill the target cache so the captured sidecar is at an exact
                # boundary, then report the lane's position like the engine.
                kv = mx.ones((1, 1, len(prompt), 4), dtype=mx.float32)
                prompt_cache[0].update_and_fetch(kv, kv)
                lane = self_mtp.get("lane_rng") if self_mtp else None
                if self_mtp is not None:
                    self_mtp["state_out"].update(
                        state=("mtp-cache", "hidden"),
                        covered_tokens=prompt_cache[0].offset,
                        reusable=True,
                        rng_key=covered_key if covered_key is not None else (
                            lane.key if lane is not None else None
                        ),
                        rng_draws=draws,
                    )
                yield types.SimpleNamespace(
                    text="hi",
                    token=3,
                    logprobs=mx.zeros((8,), dtype=mx.float32),
                    finish_reason="stop",
                )

            return _gen()

        original = server_module.stream_generate
        server_module.stream_generate = fake_stream_generate
        try:
            rqueue = Queue()
            request = CompletionRequest("text", "hello", [], None, None)
            generator._serve_single((rqueue, request, args))
        finally:
            server_module.stream_generate = original
        items = []
        while True:
            item = rqueue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            items.append(item)
        return items

    def test_the_request_decodes_with_a_lane_key(self):
        generator = self._generator(range(1, 9))
        captured = []
        self._serve(generator, _mtp_args(), captured)
        self.assertEqual(len(captured), 1)
        self.assertIsNotNone(captured[0])
        self.assertIsInstance(captured[0].get("lane_rng"), LaneRNG)

    def test_an_explicit_seed_reaches_the_lane(self):
        generator = self._generator(range(1, 9))
        captured = []
        self._serve(generator, _mtp_args(seed=4242), captured)
        self.assertEqual(
            _key_list(captured[0]["lane_rng"].key), _key_list(LaneRNG(4242).key)
        )

    @staticmethod
    def _stored_sidecar(generator, tokens):
        """The sidecar an extending prompt would restore, via the APC lookup."""
        return generator.prompt_cache.lookup(
            generator.model_provider.model_key, list(tokens)
        ).sidecar

    def test_the_stored_sidecar_carries_the_lane_position(self):
        generator = self._generator(range(1, 9))
        captured = []
        self._serve(generator, _mtp_args(), captured)
        lane = captured[0]["lane_rng"]
        sidecar = self._stored_sidecar(generator, range(1, 13))
        self.assertIsNotNone(sidecar)
        self.assertEqual(_key_list(sidecar.rng_key), _key_list(lane.key))

    def test_a_resumed_request_continues_its_own_stream(self):
        # Turn 1 stores a sidecar at the 8-token boundary.
        generator = self._generator(range(1, 9))
        first = []
        self._serve(generator, _mtp_args(), first, draws=3)
        stored = self._stored_sidecar(generator, range(1, 13))
        self.assertIsNotNone(stored)

        # Turn 2 extends the prompt, so the lookup returns that sidecar.
        generator.model_provider.tokenizer = _Tokenizer(range(1, 13))
        second = []
        self._serve(generator, _mtp_args(), second)
        resumed = second[0]["lane_rng"]
        self.assertEqual(_key_list(resumed.key), _key_list(stored.rng_key))
        self.assertEqual(resumed.draws, 3)


if __name__ == "__main__":
    unittest.main()
