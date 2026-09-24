"""Small but real-file checks for bounded session orchestration and resume."""

import hashlib
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from terasort.session_manifest import load_session
from terasort.session_calibration import calibrate_day, select_calibration_windows
from terasort.session_models import LocalModels, match_residual_cpu
from terasort.session_signal import (Core, QC_ARTIFACT, QC_DROPOUT,
                                     assess_quality, iter_cores, preprocess)
from terasort.session_sort import run_session
from terasort.session_sort import preprocess as original_preprocess


def pulse():
    x = np.arange(61) - 30
    return -np.exp(-(x / 2.7) ** 2).astype(np.float32)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_source(self, *, gap=False):
        rng = np.random.default_rng(14)
        raw = rng.normal(0, 3, (6000, 2)).astype(np.int16)
        raw[600-30:600+31, 0] += np.rint(400 * pulse()).astype(np.int16)
        raw[2600-30:2600+31, 0] += np.rint(450 * pulse()).astype(np.int16)
        raw[4800-30:4800+31, 0] += np.rint(425 * pulse()).astype(np.int16)
        one = self.root / "first.dat"
        two = self.root / "second.dat"
        raw[:4000].tofile(one)
        raw[4000:].tofile(two)
        manifest = {
            "schema_version": 1, "session_id": "synthetic",
            "probes": [{
                "probe_id": "probeA", "sample_rate_hz": 20_000,
                "gain_uv_per_count": .2,
                "geometry": {"x_um": [0, 100], "y_um": [0, 0],
                             "shank": [0, 1]},
                "segments": [
                    {"path": str(one), "start_sample": 0,
                     "n_samples": 4000, "day_id": "day1"},
                    {"path": str(two), "start_sample": 5000 if gap else 4000,
                     "n_samples": 2000, "day_id": "day1"},
                ],
                "gaps": ([{"start_sample": 4000, "stop_sample": 5000,
                            "reason": "acquisition stopped"}] if gap else []),
            }],
        }
        path = self.root / "session.json"
        path.write_text(json.dumps(manifest))
        return path, raw

    def make_seed(self, manifest):
        session = load_session(manifest)
        probe = session.probes[0]
        core = next(iter_cores(probe, core_seconds=.2))
        voltage = preprocess(core, probe)
        template = np.zeros((1, 61, 2), np.float32)
        template[0] = voltage[600-30:600+31]
        seed = self.root / "seeds.npy"
        np.save(seed, template)
        data = json.loads(manifest.read_text())
        data["probes"][0]["seed_templates"] = str(seed)
        data["probes"][0]["seed_preprocessing_id"] = "terasort-session-v1"
        manifest.write_text(json.dumps(data))
        return seed

    def test_explicit_gap_and_bounded_reader(self):
        manifest, _ = self.make_source(gap=True)
        probe = load_session(manifest).probes[0]
        cores = list(iter_cores(probe, core_seconds=.1, halo_samples=100))
        self.assertEqual([(c.core_start, c.core_stop) for c in cores],
                         [(0, 2000), (2000, 4000), (5000, 7000)])
        self.assertTrue(all(c.raw.nbytes <= (2000 + 200) * 2 * 2 for c in cores))
        self.assertEqual(cores[-1].data_start, 5000)
        data = json.loads(manifest.read_text())
        data["probes"][0]["gaps"] = []
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "gap"):
            load_session(manifest)

    def test_quality_flags_zero_contact(self):
        manifest, raw = self.make_source()
        raw[:, 1] = 0
        raw[:4000].tofile(self.root / "first.dat")
        probe = load_session(manifest).probes[0]
        core = next(iter_cores(probe, core_seconds=.1))
        quality = assess_quality(core, preprocess(core, probe), probe)
        self.assertTrue(quality.flags[1] & QC_DROPOUT)
        self.assertFalse(quality.usable_channels[1])

    def test_grouped_reads_preserve_cores_halos_gaps_and_ownership(self):
        from terasort.session_signal import _read_range, prefetch_cores
        manifest, raw = self.make_source(gap=True)
        probe = load_session(manifest).probes[0]
        expanded = np.tile(raw, (100, 1))
        expanded[:300000].tofile(self.root / "first.dat")
        expanded[300000:].tofile(self.root / "second.dat")
        probe = replace(probe, segments=(
            replace(probe.segments[0], n_samples=300000),
            replace(probe.segments[1], start_sample=300173, n_samples=300000)))
        kwargs = dict(core_seconds=.1, halo_samples=100,
                      start_sample=1234, stop_sample=590321)
        original = list(iter_cores(probe, **kwargs))
        with patch("terasort.session_signal._read_range", wraps=_read_range) as reader:
            grouped = list(prefetch_cores(iter_cores(probe, read_buffer_mb=1, **kwargs), depth=4))
        self.assertEqual(len(original), len(grouped))
        self.assertLess(reader.call_count, len(original)//10)
        self.assertTrue(all(call.args[3] <= 1024**2 for call in reader.call_args_list))
        self.assertLess(sum(c.source_bytes_read for c in grouped),
                        sum(c.source_bytes_read for c in original))
        for left, right in zip(original, grouped):
            self.assertEqual((left.core_start, left.core_stop, left.data_start),
                             (right.core_start, right.core_stop, right.data_start))
            np.testing.assert_array_equal(left.raw, right.raw)
            self.assertTrue(right.raw.flags.owndata)
            self.assertFalse(right.raw.flags.writeable)

    def test_buffer_size_validation(self):
        manifest, _ = self.make_source()
        probe = load_session(manifest).probes[0]
        for value in (-1, 1.5, True, 1025):
            with self.assertRaises(ValueError):
                next(iter_cores(probe, read_buffer_mb=value))
        with self.assertRaisesRegex(ValueError, "fit one core"):
            next(iter_cores(probe, core_seconds=20, read_buffer_mb=1))
    def test_synchronous_artifact_is_visible_without_disabling_all_contacts(self):
        manifest, _ = self.make_source()
        data = json.loads(manifest.read_text())
        data["probes"][0]["geometry"] = {
            "x_um": list(range(8)), "y_um": [0] * 8,
            "shank": list(range(8))}
        manifest.write_text(json.dumps(data))
        # Quality testing uses an in-memory core, not the two-channel files.
        probe = load_session(manifest, verify_sources=False).probes[0]
        rng = np.random.default_rng(4)
        raw = rng.normal(0, 5, (1000, 8)).astype(np.int16)
        voltage = raw.astype(np.float32)
        voltage[500] += 200
        core = Core("probeA", "day1", self.root / "unused", 0, 1000,
                    0, raw, raw.nbytes)
        quality = assess_quality(core, voltage, probe)
        self.assertTrue(quality.interval_bad)
        self.assertTrue(np.all(quality.flags & QC_ARTIFACT))
        self.assertTrue(np.all(quality.usable_channels))

    def test_residual_match_recovers_two_colliding_templates(self):
        n = 300
        signal = np.zeros((n, 2), np.float32)
        left = np.zeros((61, 2), np.float32)
        right = np.zeros_like(left)
        left[:, 0] = 15 * pulse()
        right[:, 0] = 12 * pulse()
        signal[100-30:100+31] += left
        signal[103-30:103+31, 1] += right[:, 0]
        models = LocalModels(np.stack([left, right]),
                             np.array([[0, -1], [1, -1]], np.int32),
                             np.array([0, 1], np.int32),
                             np.zeros(2, np.int64))
        matches, residual = match_residual_cpu(
            signal, [(100, 0, 15.), (103, 1, 12.)], models,
            score_floor=.6, core_start=0, core_stop=n)
        self.assertEqual({match.unit for match in matches}, {0, 1})
        self.assertLess(np.square(residual).sum(), 1e-2)

    def test_session_writes_complete_shards_and_resume_is_idempotent(self):
        manifest, _ = self.make_source()
        self.make_seed(manifest)
        probe = load_session(manifest).probes[0]
        source = self.root / "first.dat"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        output = self.root / "out"
        settings = dict(backend="cpu", core_seconds=.1, shard_seconds=.2,
                        floor_snr=4., cache_fraction=.05)
        report = run_session(manifest, output, **settings)
        self.assertEqual(report["shards"], 2)
        shards = sorted((output / "probeA").glob("*.h5"))
        self.assertEqual(len(shards), 2)
        sizes = [path.stat().st_size for path in shards]
        all_times = []
        for path in shards:
            with h5py.File(path) as handle:
                self.assertTrue(handle.attrs["complete"])
                self.assertEqual(handle.attrs["candidate_format"], "SCB0.3")
                self.assertEqual(handle.attrs["day_id"], "day1")
                self.assertIn("telemetry_json", handle.attrs)
                self.assertEqual(len(handle["qc_saturated_fraction"]),
                                 len(handle["qc"]))
                candidates = handle["candidates"][:]
                self.assertTrue(np.all(candidates["sample_index"] >=
                                       handle.attrs["start_sample"]))
                self.assertTrue(np.all(candidates["sample_index"] <
                                       handle.attrs["stop_sample"]))
                cache = handle["waveform_candidate_row"][:]
                self.assertTrue(np.all(cache < len(candidates)))
                all_times.extend(handle["spikes"]["sample_index"][:].tolist())
        self.assertEqual(len(all_times), len(set(all_times)))
        self.assertTrue(any(abs(t - 600) <= 2 for t in all_times))
        self.assertTrue(any(abs(t - 2600) <= 2 for t in all_times))
        self.assertTrue(any(abs(t - 4800) <= 2 for t in all_times))
        resumed = run_session(manifest, output, resume=True, **settings)
        self.assertEqual(resumed["shards"], 0)
        self.assertEqual(sizes, [path.stat().st_size for path in shards])
        self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_interrupted_shard_reprocesses_without_duplicate_events(self):
        manifest, _ = self.make_source()
        self.make_seed(manifest)
        settings = dict(backend="cpu", core_seconds=.1, shard_seconds=.2,
                        floor_snr=4., cache_fraction=.05)
        interrupted = self.root / "interrupted"
        calls = 0
        def fail_on_third_core(core, probe):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated worker crash")
            return original_preprocess(core, probe)
        with patch("terasort.session_sort.preprocess",
                   side_effect=fail_on_third_core):
            with self.assertRaisesRegex(RuntimeError, "simulated worker crash"):
                run_session(manifest, interrupted, **settings)
        self.assertEqual(len(list((interrupted / "probeA").glob("*.h5"))), 1)
        self.assertEqual(len(list((interrupted / "probeA").glob("*.partial"))), 1)
        resumed = run_session(manifest, interrupted, resume=True, **settings)
        self.assertEqual(resumed["shards"], 1)
        fresh = self.root / "fresh"
        run_session(manifest, fresh, **settings)
        for a, b in zip(sorted((interrupted / "probeA").glob("*.h5")),
                        sorted((fresh / "probeA").glob("*.h5"))):
            with h5py.File(a) as left, h5py.File(b) as right:
                np.testing.assert_array_equal(left["candidates"][:],
                                              right["candidates"][:])
                np.testing.assert_array_equal(left["spikes"][:],
                                              right["spikes"][:])

    def test_cross_day_links_remain_unresolved(self):
        manifest, _ = self.make_source(gap=True)
        self.make_seed(manifest)
        data = json.loads(manifest.read_text())
        data["probes"][0]["segments"][1]["day_id"] = "day2"
        manifest.write_text(json.dumps(data))
        out = self.root / "days"
        run_session(manifest, out, backend="cpu", core_seconds=.1,
                    shard_seconds=.2, floor_snr=4.)
        graph = json.loads((out / "cross_day_links.json").read_text())
        self.assertTrue(graph["edges"])
        self.assertTrue(all(edge["status"] == "unresolved" and
                            edge["confidence"] is None
                            for edge in graph["edges"]))
        self.assertTrue(all(edge["left_local_id"] != edge["right_local_id"]
                            for edge in graph["edges"]))

    def test_bounded_calibration_uses_clean_windows(self):
        manifest, raw = self.make_source()
        for center in range(100, 5900, 100):
            raw[center-30:center+31, 0] += np.rint(
                250 * pulse()).astype(np.int16)
        raw[:4000].tofile(self.root / "first.dat")
        raw[4000:].tofile(self.root / "second.dat")
        probe = load_session(manifest).probes[0]
        windows = select_calibration_windows(probe, "day1")
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows, [(0, 4000), (4000, 6000)])
        models, details = calibrate_day(probe, "day1")
        self.assertGreaterEqual(len(models.waveforms), 1)
        self.assertLessEqual(details["sampled_waveforms"], 32_768)
        self.assertEqual(details["method"],
                         "bounded_high_snr_local_kmeans_v1")


if __name__ == "__main__":
    unittest.main()
