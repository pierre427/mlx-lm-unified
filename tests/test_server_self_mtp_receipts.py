"""Request-owned self-MTP evidence on completion and streaming responses."""

import types
import unittest
from unittest.mock import patch

from mlx_lm.server import (
    Response,
    _completed_self_mtp_receipt,
)
from test_parallel_sampling import _AssemblyHarness, _ctx


def _receipt(proposed=4):
    return {
        "route": "segmented_b1_self_mtp",
        "accept_rule": "residual",
        "sampling_temperature": 0.7,
        "stats": {"draft_proposed": proposed, "draft_cycles": 2},
    }


def _terminal(index=0, receipt=None):
    return Response("answer", 1, 0.0, "length", (), index, receipt)


class TestSelfMTPResponseReceipts(unittest.TestCase):
    def test_terminal_evidence_is_completed_before_queue_publication(self):
        lane_receipt = _receipt()
        source = types.SimpleNamespace(
            finish_reason="length", mtp_receipt=lane_receipt
        )
        with patch("mlx_lm.server.time.time", return_value=123):
            receipt = _completed_self_mtp_receipt(source, _ctx(2))
        self.assertEqual(receipt["ts"], 123)
        self.assertEqual(receipt["prompt_tokens"], 3)
        self.assertEqual(receipt["cached_prompt_tokens"], 2)
        self.assertTrue(receipt["completed"])
        self.assertNotIn("completed", lane_receipt)
        source.finish_reason = None
        self.assertIsNone(_completed_self_mtp_receipt(source, _ctx()))
        source.finish_reason = "stop"
        source.mtp_receipt = None
        self.assertIsNone(_completed_self_mtp_receipt(source, _ctx()))

    def test_single_chat_and_text_receipt_belongs_to_http_request(self):
        for object_type in ("chat.completion", "text_completion"):
            with self.subTest(object_type=object_type):
                receipt = _completed_self_mtp_receipt(
                    _terminal(receipt=_receipt()), _ctx()
                )
                receipt["request_id"] = "stale-id"
                harness = _AssemblyHarness(_ctx(), [_terminal(receipt=receipt)])
                harness.object_type = object_type
                result = harness.run()
                own = result["self_mtp_receipt"]
                self.assertEqual(own["request_id"], result["id"])
                self.assertEqual(own["choice_index"], 0)
                self.assertEqual(own, result["choices"][0]["self_mtp_receipt"])
                self.assertTrue(own["completed"])
                self.assertEqual(receipt["request_id"], "stale-id")

    def test_unrelated_global_receipts_cannot_fill_an_absent_receipt(self):
        with patch("mlx_lm.server.SELF_MTP_RECEIPTS", [_receipt(999)]):
            result = _AssemblyHarness(_ctx(), [_terminal()]).run()
        self.assertNotIn("self_mtp_receipt", result)
        self.assertNotIn("self_mtp_receipt", result["choices"][0])

    def test_interleaved_choices_keep_receipts_and_fallback_separate(self):
        stream = [
            Response("first", 1, 0.0, None, (), index=0),
            _terminal(2, _receipt(20)),
            _terminal(1),
            _terminal(0, _receipt(4)),
        ]
        result = _AssemblyHarness(_ctx(), stream, n=3).run()
        self.assertNotIn("self_mtp_receipt", result)
        choices = result["choices"]
        for i, proposed in ((0, 4), (2, 20)):
            receipt = choices[i]["self_mtp_receipt"]
            self.assertEqual(receipt["request_id"], result["id"])
            self.assertEqual(receipt["choice_index"], i)
            self.assertEqual(receipt["stats"]["draft_proposed"], proposed)
        self.assertNotIn("self_mtp_receipt", choices[1])

    def test_stream_receipt_is_on_terminal_chunk_only(self):
        stream = [
            Response("first", 1, 0.0, None, ()),
            _terminal(receipt=_receipt()),
        ]
        result = _AssemblyHarness(
            _ctx(), stream, stream=True, stream_options={"include_usage": True}
        ).run()
        terminal = [
            r for r in result
            if r["choices"] and r["choices"][0]["finish_reason"]
        ]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["self_mtp_receipt"]["request_id"], "req-test")
        for chunk in result:
            if chunk is not terminal[0]:
                self.assertNotIn("self_mtp_receipt", chunk)
                for choice in chunk["choices"]:
                    self.assertNotIn("self_mtp_receipt", choice)

    def test_parallel_terminal_stream_receipts_are_per_choice(self):
        result = _AssemblyHarness(
            _ctx(), [_terminal(1, _receipt(20)), _terminal(0, _receipt(4))],
            n=2, stream=True,
        ).run()
        terminals = [r for r in result if r["choices"][0]["finish_reason"]]
        self.assertEqual([r["choices"][0]["index"] for r in terminals], [1, 0])
        for chunk in terminals:
            self.assertNotIn("self_mtp_receipt", chunk)
            choice = chunk["choices"][0]
            self.assertEqual(
                choice["self_mtp_receipt"]["choice_index"], choice["index"]
            )


if __name__ == "__main__":
    unittest.main()
