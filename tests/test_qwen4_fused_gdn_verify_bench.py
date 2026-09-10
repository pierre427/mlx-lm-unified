"""Pure-helper tests for benchmarks/qwen4_fused_gdn_verify.py (no Metal)."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_BENCH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "qwen4_fused_gdn_verify.py"
)
_spec = importlib.util.spec_from_file_location("qwen4_fused_gdn_verify_bench", _BENCH)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)


class TestScheduling(unittest.TestCase):
    def test_scripted_drafts_cover_every_acceptance_count(self):
        oracle = [10, 20, 30]
        seen = set()
        for round_index in range(6):
            drafts, correct = bench.scripted_drafts(oracle, 2, round_index, vocab=100)
            self.assertEqual(len(drafts), 2)
            self.assertEqual(drafts[:correct], oracle[:correct])
            if correct < 2:
                self.assertNotEqual(drafts[correct], oracle[correct])
            self.assertEqual(bench.greedy_accept_count(oracle, drafts), correct)
            seen.add(correct)
        self.assertEqual(seen, {0, 1, 2})

    def test_scripted_drafts_wrap_vocab(self):
        drafts, correct = bench.scripted_drafts([99, 5], 2, round_index=2, vocab=100)
        self.assertEqual((drafts, correct), ([0, 5], 0))

    def test_greedy_accept_stops_at_first_miss(self):
        self.assertEqual(bench.greedy_accept_count([1, 2, 3], [1, 9]), 1)
        self.assertEqual(bench.greedy_accept_count([1, 2, 3], [1, 2]), 2)
        self.assertEqual(bench.greedy_accept_count([1, 2, 3], [7, 2]), 0)

    def test_expected_verify_calls(self):
        self.assertEqual(bench.expected_verify_calls("stock", 36, 10), (0, 0))
        self.assertEqual(bench.expected_verify_calls("fused", 36, 10), (360, 0))
        with self.assertRaises(ValueError):
            bench.expected_verify_calls("other", 36, 10)


class TestHostGuards(unittest.TestCase):
    def _args(self, **overrides):
        base = {"min_free_percent": 20, "max_swap_growth_mib": 2048.0}
        base.update(overrides)
        return SimpleNamespace(**base)

    def _receipt(self, **overrides):
        base = {
            "scan_ok": True,
            "free_percent": 80.0,
            "swap_used_mib": 100.0,
            "port_8282_listeners": 0,
            "model_processes": [],
            "large_processes": [],
        }
        base.update(overrides)
        return base

    def test_parsers(self):
        self.assertEqual(
            bench.parse_free_percent("System-wide memory free percentage: 63%"), 63.0
        )
        self.assertIsNone(bench.parse_free_percent("garbage"))
        self.assertEqual(
            bench.parse_swap_used_mib(
                "vm.swapusage: total = 0.00M  used = 12.50M  free = 0.00M"
            ),
            12.5,
        )
        self.assertIsNone(bench.parse_swap_used_mib(""))

    def test_violations(self):
        args = self._args()
        base = self._receipt()
        self.assertIsNone(bench.host_violation(base, base, args, "x"))
        self.assertIn(
            "unreadable",
            bench.host_violation(self._receipt(scan_ok=False), base, args, "x"),
        )
        self.assertIn(
            "tenant",
            bench.host_violation(self._receipt(port_8282_listeners=1), base, args, "x"),
        )
        self.assertIn(
            "tenant",
            bench.host_violation(
                self._receipt(large_processes=["1 9000000 python big"]), base, args, "x"
            ),
        )
        self.assertIn(
            "free memory",
            bench.host_violation(self._receipt(free_percent=5.0), base, args, "x"),
        )
        self.assertIn(
            "swap grew",
            bench.host_violation(self._receipt(swap_used_mib=5000.0), base, args, "x"),
        )
        self.assertIn(
            "unreadable",
            bench.host_violation(self._receipt(free_percent=None), base, args, "x"),
        )

    def test_model_process_filter(self):
        own = os.getpid()
        self.assertTrue(bench.is_model_process("1 100 python -m mlx_lm.server", own))
        self.assertTrue(bench.is_model_process("2 100 /opt/rapid-mlx serve", own))
        self.assertTrue(bench.is_model_process("3 100 python3 bench_decode.py", own))
        self.assertFalse(bench.is_model_process("4 100 python -m http.server", own))
        self.assertFalse(
            bench.is_model_process("5 100 codex exec 'mlx_lm prompt'", own)
        )
        self.assertFalse(
            bench.is_model_process(f"{own} 100 python -m mlx_lm.server", own)
        )
        self.assertFalse(bench.is_model_process("garbage", own))

    def test_completion_status(self):
        full = {"layer": {}, "rounds": {}, "e2e": {}}
        self.assertEqual(
            bench.completion_status(full, True),
            {
                "phases_executed": ["layer", "rounds", "e2e"],
                "complete": True,
                "partial_passed": True,
                "passed": True,
            },
        )
        partial = bench.completion_status({"layer": {}}, True)
        self.assertFalse(partial["complete"])
        self.assertTrue(partial["partial_passed"])
        self.assertFalse(partial["passed"])
        self.assertFalse(bench.completion_status(full, False)["passed"])


if __name__ == "__main__":
    unittest.main()
