"""Contract tests for the portability layer: probe, guardrails, cache, order.

CPU only.  Nothing here launches a kernel -- the primitive tests that DO are
the device's own gate and live behind ``MLX_QWEN4_MEGAKERNEL_TUNE``.  What is
proven here is everything that decides whether a launch is even attempted: a
probe that reports what it could not read instead of inventing it, a refusal
for each way a device can be wrong for this kernel, a cache that survives
being corrupt, and the precedence an operator's override depends on.
"""

import json
import os
import tempfile
import unittest

from mlx_lm.models import qwen4_megakernel as mk
from mlx_lm.models import qwen4_megakernel_config as MC
from mlx_lm.models import qwen4_megakernel_device as MD
from mlx_lm.models import qwen4_megakernel_tune as MT

M5_MAX = dict(
    architecture="applegpu_g17s",
    device_name="Apple M5 Max",
    memory_size=137438953472,
    max_recommended_working_set_size=120259084288,
    max_buffer_length=86586540032,
    resource_limit=499000,
    gpu_cores=40,
    max_threads_per_threadgroup=1024,
    max_threadgroup_memory=32768,
    unified_memory=True,
    metal_families=("apple9", "metal3", "metal4"),
    metal_version=400,
    mlx_version="0.32.2",
)


def probe_from(**overrides) -> MD.DeviceProbe:
    fields = dict(M5_MAX)
    fields.update(overrides)
    unknown = tuple(name for name, value in fields.items() if value is None)
    return MD.DeviceProbe(**fields, unknown=unknown,
                          sources={n: "stub" for n in fields})


class _EnvMixin:
    """Environment and module memos are process state; put them back."""

    ENV = (
        "MLX_QWEN4_MEGAKERNEL_THREADS", "MLX_QWEN4_MEGAKERNEL_GROUPS",
        "MLX_QWEN4_MEGAKERNEL_SPIN_CAP", "MLX_QWEN4_MEGAKERNEL_ROWS",
        "MLX_QWEN4_MEGAKERNEL_THREADGROUP_BYTES",
        "MLX_QWEN4_MEGAKERNEL_REQUIRE_PRIMITIVES",
        "MLX_QWEN4_MEGAKERNEL_TUNE", "MLX_QWEN4_MEGAKERNEL_TUNE_CACHE",
        "MLX_QWEN4_MEGAKERNEL_TUNE_BUDGET_S",
        "MLX_QWEN4_MEGAKERNEL_GPU_CORES",
        "MLX_QWEN4_MEGAKERNEL_TUNE_BUSY_PATH",
        "MLX_QWEN4_MEGAKERNEL_TUNE_ADOPT",
    )

    def setUp(self):
        self._env = {name: os.environ.get(name) for name in self.ENV}
        for name in self.ENV:
            os.environ.pop(name, None)
        MC.invalidate()
        MT.forget()

    def tearDown(self):
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        MC.invalidate()
        MT.forget()


class TestProbeParsing(_EnvMixin, unittest.TestCase):
    def _stub(self, info, limits, cores=(40, "ioregistry"), version="0.32.2"):
        self.addCleanup(setattr, MD, "_mlx_device_info", MD._mlx_device_info)
        self.addCleanup(setattr, MD, "_metal_device_limits",
                        MD._metal_device_limits)
        self.addCleanup(setattr, MD, "gpu_core_count", MD.gpu_core_count)
        self.addCleanup(setattr, MD, "_mlx_version", MD._mlx_version)
        self.addCleanup(setattr, MD, "_CACHED", MD._CACHED)
        MD._mlx_device_info = lambda: dict(info)
        MD._metal_device_limits = lambda: dict(limits)
        MD.gpu_core_count = lambda: cores
        MD._mlx_version = lambda: version
        return MD.probe_device(refresh=True)

    def test_reads_every_field_it_is_given(self):
        probe = self._stub(
            {"architecture": "applegpu_g17s", "device_name": "Apple M5 Max",
             "memory_size": 137438953472,
             "max_recommended_working_set_size": 120259084288,
             "max_buffer_length": 86586540032, "resource_limit": 499000},
            {"max_threads_per_threadgroup": 1024,
             "max_threadgroup_memory": 32768, "unified_memory": True,
             "metal_families": ("apple9", "metal3", "metal4")})
        self.assertEqual(probe.gpu_cores, 40)
        self.assertEqual(probe.memory_gib, 128)
        self.assertEqual(probe.metal_version, 400)
        self.assertTrue(probe.apple_gpu)
        self.assertFalse(probe.multi_die)
        self.assertEqual(probe.unknown, ())
        self.assertEqual(probe.sources["gpu_cores"], "ioregistry")
        self.assertEqual(
            probe.signature, "applegpu_g17s-40c-128g-mlx0.32.2")

    def test_missing_fields_are_unknown_not_guessed(self):
        probe = self._stub({"architecture": "applegpu_g99p"}, {},
                           cores=(None, "unreadable"))
        for name in ("memory_size", "max_buffer_length", "gpu_cores",
                     "max_threads_per_threadgroup", "max_threadgroup_memory",
                     "metal_version"):
            self.assertIsNone(getattr(probe, name), name)
            self.assertIn(name, probe.unknown, name)
            self.assertEqual(probe.sources[name], "unreadable", name)
        self.assertEqual(probe.signature, "applegpu_g99p-unkc-unk-mlx0.32.2")
        # An unreadable device name is not evidence of a single die.
        self.assertIsNone(probe.multi_die)

    def test_no_metal_device_at_all(self):
        probe = self._stub({}, {}, cores=(None, "unreadable"), version=None)
        self.assertFalse(probe.apple_gpu)
        self.assertEqual(probe.signature, "unknown-unkc-unk-mlxunknown")

    def test_metal_version_follows_the_family(self):
        self.assertEqual(
            MD._metal_version_from_families(("apple8", "metal3")), 320)
        self.assertIsNone(MD._metal_version_from_families(("apple7",)))

    def test_ultra_class_is_named_as_multi_die(self):
        self.assertTrue(probe_from(device_name="Apple M5 Ultra").multi_die)


class TestDerivedDefaults(_EnvMixin, unittest.TestCase):
    def test_this_machine_reproduces_the_shipped_geometry(self):
        """The shipped M5 Max numbers are what the RULE produces here.

        This is the claim the whole layer rests on: 512 x 40 was not a
        coincidence of one tuning round, it is 512 threads per core on a
        40-core part, and a probe that reads 40 cores must land back on it
        with no cache present.
        """
        threads, groups = MC.derive_geometry(probe_from())
        self.assertEqual((threads, groups),
                         (MC.SHIPPED_THREADS, MC.SHIPPED_GROUPS))
        self.assertEqual(MC.derive_threadgroup_bytes(probe_from()),
                         MC.SHIPPED_THREADGROUP_BYTES)
        resolved = MC.resolve(probe=probe_from(), tune=False)
        self.assertEqual(resolved["values"]["threads"], 512)
        self.assertEqual(resolved["values"]["groups"], 40)
        self.assertEqual(resolved["sources"]["threads"], "probe")
        self.assertEqual(resolved["sources"]["groups"], "probe")

    def test_the_shipped_constants_still_match_the_kernel(self):
        self.assertEqual(MC.SHIPPED_THREADS, mk._THREADS)
        self.assertEqual(MC.SHIPPED_GROUPS, mk._THREADGROUPS)
        self.assertEqual(MC.SHIPPED_SPIN_CAP, mk._SPIN_CAP)
        self.assertEqual(MC.SHIPPED_ROWS, dict(mk.PHASE_ROWS))
        self.assertEqual(MC.SHIPPED_THREADGROUP_BYTES,
                         mk.THREADGROUP_BYTES_CAP)

    def test_smaller_and_narrower_parts_scale_the_rule(self):
        # A 10-core part: same threads per core, a quarter of the grid.
        self.assertEqual(MC.derive_geometry(probe_from(gpu_cores=10)),
                         (512, 10))
        # A part that caps threadgroups at 256 gets there with more of them.
        self.assertEqual(
            MC.derive_geometry(probe_from(max_threads_per_threadgroup=256)),
            (256, 80))
        # A 16 KiB arena halves the threadgroup budget rather than keeping
        # the number that happened to be half of 32 KiB.
        self.assertEqual(
            MC.derive_threadgroup_bytes(probe_from(
                max_threadgroup_memory=16384)), 8192)

    def test_an_unreadable_core_count_falls_back_to_shipped(self):
        probe = probe_from(gpu_cores=None)
        self.assertEqual(MC.derive_geometry(probe), (None, None))
        resolved = MC.resolve(probe=probe, tune=False)
        self.assertEqual(resolved["values"]["threads"], MC.SHIPPED_THREADS)
        self.assertEqual(resolved["sources"]["threads"], "shipped")


class TestPrecedence(_EnvMixin, unittest.TestCase):
    CACHE = {"threads": 256, "groups": 96, "rows": {"moe_down": 4}}

    def test_env_beats_cache_beats_probe_beats_shipped(self):
        probe = probe_from()
        resolved = MC.resolve(probe=probe, cache_entry=self.CACHE, tune=False)
        self.assertEqual(resolved["values"]["threads"], 256)
        self.assertEqual(resolved["sources"]["threads"], "cache")

        os.environ["MLX_QWEN4_MEGAKERNEL_THREADS"] = "128"
        resolved = MC.resolve(probe=probe, cache_entry=self.CACHE, tune=False)
        self.assertEqual(resolved["values"]["threads"], 128)
        self.assertEqual(resolved["sources"]["threads"], "env")

        # No cache: the probe rule, then the shipped constant when the probe
        # cannot support the rule.
        os.environ.pop("MLX_QWEN4_MEGAKERNEL_THREADS")
        self.assertEqual(
            MC.resolve(probe=probe, tune=False)["sources"]["groups"], "probe")
        self.assertEqual(
            MC.resolve(probe=probe_from(gpu_cores=None),
                       tune=False)["sources"]["groups"], "shipped")
        # spin_cap has no probe rule at all, so it falls straight through.
        self.assertEqual(
            MC.resolve(probe=probe, tune=False)["sources"]["spin_cap"],
            "shipped")

    def test_rows_resolve_per_phase(self):
        os.environ["MLX_QWEN4_MEGAKERNEL_ROWS"] = "gdn_in_proj=8"
        resolved = MC.resolve(probe=probe_from(), cache_entry=self.CACHE,
                              tune=False)
        self.assertEqual(resolved["values"]["rows"]["gdn_in_proj"], 8)
        self.assertEqual(resolved["row_sources"]["gdn_in_proj"], "env")
        self.assertEqual(resolved["values"]["rows"]["moe_down"], 4)
        self.assertEqual(resolved["row_sources"]["moe_down"], "cache")
        self.assertEqual(resolved["values"]["rows"]["moe_router"],
                         MC.SHIPPED_ROWS["moe_router"])
        self.assertEqual(resolved["row_sources"]["moe_router"], "shipped")

    def test_unadopted_legacy_sweep_geometry_is_ignored(self):
        cached = {
            "threads": 512,
            "groups": 80,
            "rows": {"generic_qmv": 8},
            "sweep": {"ok": True, "winner": {"threads": 512, "groups": 80}},
        }
        resolved = MC.resolve(
            probe=probe_from(), cache_entry=cached, tune=False
        )
        self.assertEqual(resolved["values"]["groups"], 40)
        self.assertEqual(resolved["sources"]["groups"], "probe")
        self.assertIn("geometry_ignored", resolved["cache"])

    def test_a_setting_that_cannot_be_honoured_is_an_error_not_a_default(self):
        for name, value in (("MLX_QWEN4_MEGAKERNEL_THREADS", "not-a-number"),
                            ("MLX_QWEN4_MEGAKERNEL_GROUPS", "0"),
                            ("MLX_QWEN4_MEGAKERNEL_ROWS", "no_such_phase=2"),
                            ("MLX_QWEN4_MEGAKERNEL_ROWS", "moe_down=x")):
            os.environ[name] = value
            with self.assertRaises(MC.ConfigError):
                MC.resolve(probe=probe_from(), tune=False)
            os.environ.pop(name)


class TestGuardrails(_EnvMixin, unittest.TestCase):
    OK = {"ok": True, "geometry": {"threads": 512, "groups": 40}}

    class _Buffer:
        def __init__(self, words):
            self.size = words

    class _Pack:
        def __init__(self, *word_counts):
            self.buffers = [TestGuardrails._Buffer(n) for n in word_counts]

    def refusal(self, **kw):
        kw.setdefault("threads", 512)
        kw.setdefault("groups", 40)
        kw.setdefault("probe", probe_from())
        kw.setdefault("primitives", self.OK)
        return MC.portability_refusal(**kw)

    def test_a_geometry_the_device_allows_is_admitted(self):
        self.assertIsNone(self.refusal())

    def test_threads_over_the_device_limit(self):
        reason = self.refusal(
            threads=1024, probe=probe_from(max_threads_per_threadgroup=512),
            primitives={"ok": True,
                        "geometry": {"threads": 1024, "groups": 40}})
        self.assertEqual(reason, "threads 1024 over device max 512")

    def test_threadgroup_arena_over_the_device_limit(self):
        reason = self.refusal(threadgroup_bytes=32768,
                              probe=probe_from(max_threadgroup_memory=16384))
        self.assertEqual(reason,
                         "threadgroup arena 32768 B over device limit 16384 B")

    def test_actual_threadgroup_arena_must_fit_the_residency_budget(self):
        reason = self.refusal(
            threadgroup_bytes=8192, actual_threadgroup_bytes=12288)
        self.assertEqual(
            reason,
            "actual threadgroup arena 12288 B over configured residency "
            "budget 8192 B",
        )

    def test_a_model_the_machine_cannot_hold(self):
        # 24 GiB of packed weights on a part with a 16 GiB working set.
        pack = self._Pack(6 * (1 << 30))          # 6 G words = 24 GiB
        reason = self.refusal(
            pack=pack,
            probe=probe_from(memory_size=24 * (1 << 30),
                             max_recommended_working_set_size=16 * (1 << 30)))
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("working set 24.0 GiB over device"),
                        reason)

    def test_persistent_ledgers_are_part_of_the_working_set(self):
        reason = self.refusal(
            pack=self._Pack(3 * (1 << 30) // 4),
            extra_bytes=6 * (1 << 30),
            scratch_bytes=1 * (1 << 30),
            probe=probe_from(
                max_recommended_working_set_size=8 * (1 << 30)
            ),
        )
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("working set 10.0 GiB"), reason)

    def test_live_resident_memory_replaces_the_smaller_pack_base(self):
        reason = self.refusal(
            pack=self._Pack(1 << 30),
            resident_bytes=7 * (1 << 30),
            extra_bytes=2 * (1 << 30),
            probe=probe_from(
                max_recommended_working_set_size=8 * (1 << 30)
            ),
        )
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("working set 9.0 GiB"), reason)

    def test_a_weight_group_over_the_max_buffer_length(self):
        pack = self._Pack(1 << 30)                 # 4 GiB in one buffer
        reason = self.refusal(
            pack=pack, probe=probe_from(max_buffer_length=2 * (1 << 30)))
        self.assertIsNotNone(reason)
        self.assertIn("over max buffer length", reason)

    def test_every_non_weight_buffer_obeys_max_buffer_length(self):
        reason = self.refusal(
            individual_buffer_bytes={"ledger.kv": 3 * (1 << 30)},
            probe=probe_from(max_buffer_length=2 * (1 << 30)))
        self.assertIn("ledger.kv buffer 3.0 GiB over max buffer length", reason)

    def test_a_device_that_is_not_an_apple_gpu(self):
        self.assertEqual(self.refusal(probe=probe_from(architecture="nvgpu")),
                         "device architecture nvgpu")

    def test_primitives_are_required_by_default(self):
        self.assertEqual(
            self.refusal(primitives=None),
            f"primitives unvalidated for {probe_from().signature}")
        self.assertEqual(
            self.refusal(primitives={"ok": False, "failed": "barrier hung"}),
            "primitive test failed: barrier hung")

    def test_primitives_validated_at_a_narrower_grid_do_not_carry(self):
        """The residency ceiling is a THREAD budget, so a wider grid is a new
        question, not a covered case."""
        reason = self.refusal(
            threads=512, groups=160,
            primitives={"ok": True,
                        "geometry": {"threads": 512, "groups": 40}})
        self.assertEqual(
            reason, "primitives validated at 512x40, not requested 512x160")

    def test_equal_thread_budget_with_different_geometry_does_not_carry(self):
        reason = self.refusal(
            threads=256, groups=80,
            primitives={"ok": True,
                        "geometry": {"threads": 512, "groups": 40}})
        self.assertEqual(
            reason, "primitives validated at 512x40, not requested 256x80")

    def test_the_primitive_gate_has_an_escape_hatch(self):
        os.environ["MLX_QWEN4_MEGAKERNEL_REQUIRE_PRIMITIVES"] = "0"
        self.assertIsNone(self.refusal(primitives=None))

    def test_an_unknown_limit_is_not_a_refusal(self):
        """A limit that could not be read leaves the check unmade and lets the
        primitive tests be the evidence -- it must not invent a bound."""
        probe = probe_from(max_threads_per_threadgroup=None,
                           max_threadgroup_memory=None,
                           max_recommended_working_set_size=None,
                           max_buffer_length=None)
        self.assertIsNone(self.refusal(probe=probe, threads=1024,
                                       threadgroup_bytes=1 << 20,
                                       pack=self._Pack(1 << 30),
                                       primitives={"ok": True,
                                                   "geometry": {
                                                       "threads": 1024,
                                                       "groups": 40}}))


class TestCache(_EnvMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "sub", "tune.json")
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE_CACHE"] = self.path

    def test_round_trip(self):
        entry = {"signature": "sig-a", "threads": 512, "groups": 40,
                 "primitives": {"ok": True}}
        MT.write_entry("sig-a", entry)
        self.assertEqual(MT.cache_path(), self.path)
        read, error = MT.read_entry("sig-a")
        self.assertIsNone(error)
        self.assertEqual(read, entry)

    def test_a_second_entry_does_not_evict_the_first(self):
        MT.write_entry("sig-a", {"threads": 512})
        MT.write_entry("sig-b", {"threads": 256})
        self.assertEqual(MT.read_entry("sig-a")[0], {"threads": 512})
        self.assertEqual(MT.read_entry("sig-b")[0], {"threads": 256})

    def test_a_missing_cache_is_not_an_error(self):
        entry, error = MT.read_entry("sig-a")
        self.assertIsNone(entry)
        self.assertIsNone(error)

    def test_a_corrupt_cache_is_moved_aside_and_reported(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as handle:
            handle.write('{"version": 1, "entries": {"sig-a": ')
        entry, error = MT.read_entry("sig-a")
        self.assertIsNone(entry)
        self.assertIsNotNone(error)
        self.assertTrue(os.path.exists(self.path + ".corrupt"))
        # And the next write starts a clean cache rather than failing.
        MT.write_entry("sig-a", {"threads": 512})
        self.assertEqual(MT.read_entry("sig-a")[0], {"threads": 512})

    def test_a_cache_from_another_schema_version_is_discarded(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump({"version": 99, "entries": {"sig-a": {"threads": 8}}},
                      handle)
        entry, error = MT.read_entry("sig-a")
        self.assertIsNone(entry)
        self.assertIn("version", error)

    # ------------------------------------------------------------ retuning
    def _stub_calibrate(self, entry):
        calls = []

        def fake(*, probe=None, sweep=True, budget_s=None, path=None,
                 write=True, respect_lease=True):
            calls.append({"signature": probe.signature, "sweep": sweep,
                          "respect_lease": respect_lease})
            stored = dict(entry, signature=probe.signature)
            MT.write_entry(probe.signature, stored, path)
            return stored

        self.addCleanup(setattr, MT, "calibrate", MT.calibrate)
        MT.calibrate = fake
        return calls

    def test_a_hit_does_not_calibrate(self):
        probe = probe_from()
        MT.write_entry(probe.signature,
                       {"threads": 256, "groups": 96,
                        "primitives": {"ok": True}})
        calls = self._stub_calibrate({})
        entry, state = MT.ensure_tuned(probe=probe)
        self.assertEqual(calls, [])
        self.assertTrue(state["hit"])
        self.assertFalse(state["calibrated"])
        self.assertEqual(entry["threads"], 256)

    def test_a_different_signature_re_tunes(self):
        MT.write_entry(probe_from().signature,
                       {"threads": 256, "groups": 96,
                        "primitives": {"ok": True}})
        calls = self._stub_calibrate({"threads": 512, "groups": 10,
                                      "primitives": {"ok": True}})
        other = probe_from(gpu_cores=10, device_name="Apple M5")
        entry, state = MT.ensure_tuned(probe=other)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["signature"], other.signature)
        self.assertFalse(state["hit"])
        self.assertTrue(state["calibrated"])
        self.assertEqual(entry["groups"], 10)
        # ...and the first machine's tuning is still there.
        self.assertEqual(MT.read_entry(probe_from().signature)[0]["threads"],
                         256)

    def test_an_entry_whose_primitives_failed_is_re_run(self):
        probe = probe_from()
        MT.write_entry(probe.signature,
                       {"threads": 512, "primitives": {"ok": False}})
        calls = self._stub_calibrate({"threads": 512,
                                      "primitives": {"ok": True}})
        MT.ensure_tuned(probe=probe)
        self.assertEqual(len(calls), 1)

    def test_tune_off_runs_the_primitives_but_not_the_sweep(self):
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE"] = "off"
        calls = self._stub_calibrate({"primitives": {"ok": True}})
        MT.ensure_tuned(probe=probe_from())
        self.assertEqual([c["sweep"] for c in calls], [False])

    def test_tune_skip_touches_nothing(self):
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE"] = "skip"
        calls = self._stub_calibrate({"primitives": {"ok": True}})
        entry, state = MT.ensure_tuned(probe=probe_from())
        self.assertEqual(calls, [])
        self.assertIsNone(entry)
        self.assertIn("skipped", state)

    def test_calibration_failure_is_state_not_an_exception(self):
        def boom(**kwargs):
            raise RuntimeError("no Metal device")

        self.addCleanup(setattr, MT, "calibrate", MT.calibrate)
        MT.calibrate = boom
        entry, state = MT.ensure_tuned(probe=probe_from())
        self.assertIsNone(entry)
        self.assertIn("no Metal device", state["error"])

    def test_the_sweep_yields_to_whoever_holds_the_gpu_lease(self):
        """A timing sweep taken under load is not a measurement of geometry.

        Proven the hard way on 2026-09-03: the same machine answered 256x160
        while a perplexity gate was running and 512x80 under the lock.
        """
        lease = os.path.join(self.dir.name, "gpu.lock")
        os.makedirs(lease)
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE_BUSY_PATH"] = lease
        self.assertEqual(MT.gpu_is_busy(), lease)

        probe = probe_from()
        entry = MT.calibrate(probe=probe, sweep=True)
        self.assertTrue(entry["primitives"]["ok"])
        self.assertFalse(entry["sweep"]["ok"])
        self.assertIn("deferred", entry["sweep"])
        self.assertNotIn("threads", entry)
        # A deferred sweep is NOT an answer: the next load tries again.
        MT.forget()
        calls = self._stub_calibrate({"primitives": {"ok": True}})
        MT.ensure_tuned(probe=probe)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["respect_lease"])

        # ...and the resolution falls back to the probe rule meanwhile.
        MT.forget()
        MC.invalidate()
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE"] = "skip"
        resolved = MC.resolve(probe=probe)
        self.assertEqual(resolved["values"]["threads"], 512)
        self.assertEqual(resolved["sources"]["threads"], "probe")

    def test_force_ignores_the_lease(self):
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE"] = "force"
        calls = self._stub_calibrate({"threads": 512,
                                      "primitives": {"ok": True}})
        MT.ensure_tuned(probe=probe_from())
        self.assertEqual([c["respect_lease"] for c in calls], [False])

    def test_the_sweep_is_recorded_not_adopted_where_the_rule_answers(self):
        """The proxy is weaker evidence than the rule it would override.

        Measured twice on this M5 Max: the sweep chose T=512/G=80 while the
        real per-token mix measures that geometry 13% slower than the shipped
        T=512/G=40 the rule reproduces.
        """
        probe = probe_from()
        adopt, why = MT._adopt_decision(probe)
        self.assertFalse(adopt)
        self.assertIn("recorded, not adopted", why)

        # ...but where the rule has nothing, a measurement beats a constant.
        adopt, why = MT._adopt_decision(probe_from(gpu_cores=None))
        self.assertTrue(adopt)

        # ...and an operator can override the whole judgement.
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE_ADOPT"] = "1"
        self.assertTrue(MT._adopt_decision(probe)[0])

    def test_an_unknown_tune_mode_is_refused(self):
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE"] = "sideways"
        with self.assertRaises(ValueError):
            MT.tune_mode()


class TestAdmissionIsWired(_EnvMixin, unittest.TestCase):
    """The guardrails are only worth anything if admission asks them."""

    def setUp(self):
        super().setUp()
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE"] = "skip"
        # An empty cache of our own: this machine's real cache may hold a
        # passing primitive result, and "uncalibrated" is the case under test.
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        os.environ["MLX_QWEN4_MEGAKERNEL_TUNE_CACHE"] = os.path.join(
            self.dir.name, "tune.json")
        self._enabled = mk._MEGAKERNEL_ENABLED
        mk.set_qwen4_megakernel(True)
        self.addCleanup(mk.set_qwen4_megakernel, self._enabled)
        schedule = mk.Schedule()
        schedule.add(mk.Step(op=mk.OP_NOP))
        self.kwargs = dict(width=1, batch=1, pack=object(), schedule=schedule,
                           speculating=False, training=False, sharded=False)

    def test_an_uncalibrated_signature_is_refused_by_name(self):
        MC.invalidate()
        decision = mk.admit_megakernel_decode(**self.kwargs)
        self.assertFalse(decision.accepted)
        self.assertTrue(decision.reason.startswith("primitives unvalidated"),
                        decision.reason)

    def test_a_bad_setting_declines_instead_of_raising(self):
        os.environ["MLX_QWEN4_MEGAKERNEL_THREADS"] = "seven"
        MC.invalidate()
        decision = mk.admit_megakernel_decode(**self.kwargs)
        self.assertFalse(decision.accepted)
        self.assertTrue(decision.reason.startswith("config "), decision.reason)

    def test_the_status_receipt_states_every_source(self):
        os.environ["MLX_QWEN4_MEGAKERNEL_REQUIRE_PRIMITIVES"] = "0"
        os.environ["MLX_QWEN4_MEGAKERNEL_GROUPS"] = "24"
        MC.invalidate()
        decision = mk.admit_megakernel_decode(**self.kwargs)
        self.assertTrue(decision.accepted, decision.reason)
        report = mk.qwen4_megakernel_status()["portability"]
        self.assertEqual(report["sources"]["groups"], "env")
        self.assertEqual(report["values"]["groups"], 24)
        self.assertIn("signature", report)
        self.assertIn("device", report)
        self.assertIn("cache", report)


if __name__ == "__main__":
    unittest.main()
