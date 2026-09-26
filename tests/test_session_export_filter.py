"""Tests for the reversible, output-only spike amplitude view."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import h5py
import numpy as np

from terasort.session_models import Match
from terasort.session_quality import load_session_spikes
from terasort.session_store import ShardWriter


class SessionExportFilterTests(unittest.TestCase):
    def _write_shard(self, root, threshold):
        probe = SimpleNamespace(
            probe_id="probeA", n_channels=2,
            geometry=np.asarray([[0., 0.], [0., 20.]], np.float32),
            shank=np.zeros(2, np.int32))
        path = Path(root) / "probeA" / "0000000000000000-0000000000000100.h5"
        writer = ShardWriter(
            path, probe=probe, day_id="dayA", start_sample=0, stop_sample=100,
            config_sha256="test", export_amplitude_min=threshold)
        core = SimpleNamespace(data_start=0, core_start=0, core_stop=100,
                               source_bytes_read=400)
        quality = SimpleNamespace(
            flags=np.zeros(2, np.uint8), noise_uv=np.ones(2, np.float32),
            interval_bad=False, saturated_fraction=np.zeros(2, np.float32),
            flatline_fraction=np.zeros(2, np.float32),
            zero_fraction=np.zeros(2, np.float32))
        matches = [
            Match(40, 0, 0, .9, .2, .5, .5, 0, candidate_sample=40),
            Match(50, 1, 1, .9, .2, .8, .8, 0, candidate_sample=50),
        ]
        writer.append_core(
            core=core, voltage_uv=np.zeros((100, 2), np.float32), quality=quality,
            threshold_events=[(40, 0, 6.), (50, 1, 6.)], matches=matches,
            channel_map=np.asarray([[0, 1], [1, 0]], np.int32),
            cache_fraction=0.)
        models = SimpleNamespace(
            waveforms=np.zeros((2, 61, 2), np.float32),
            channels=np.asarray([[0, 1], [1, 0]], np.int32),
            anchors=np.asarray([0, 1], np.int32),
            assigned=np.asarray([1, 1], np.int64), version=1)
        writer.finish(models)
        return path

    def test_filter_is_aligned_and_keeps_full_candidates_and_spikes(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_shard(root, .6)
            with h5py.File(path, "r") as handle:
                spikes = handle["spikes"][:]
                self.assertEqual(len(spikes), 2)
                self.assertEqual(len(handle["candidates"]), 2)
                np.testing.assert_array_equal(
                    handle["spike_export_mask"][:], [0, 1])
                metadata = handle.attrs["spike_export_filter_json"]
                self.assertIn("output_view_only", metadata)
            all_times, _, _ = load_session_spikes(root, "probeA")
            kept_times, kept_labels, _ = load_session_spikes(
                root, "probeA", export_filter=True)
            np.testing.assert_array_equal(all_times, [40, 50])
            np.testing.assert_array_equal(kept_times, [50])
            np.testing.assert_array_equal(kept_labels, ["probeA/dayA/1"])

    def test_filter_loader_preserves_legacy_shards_without_mask(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_shard(root, None)
            with h5py.File(path, "r") as handle:
                self.assertNotIn("spike_export_mask", handle)
            all_times, all_labels, _ = load_session_spikes(root, "probeA")
            filtered_times, filtered_labels, _ = load_session_spikes(
                root, "probeA", export_filter=True)
            np.testing.assert_array_equal(filtered_times, all_times)
            np.testing.assert_array_equal(filtered_labels, all_labels)


if __name__ == "__main__":
    unittest.main()
