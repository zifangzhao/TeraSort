"""Ground-truth evaluation must use only completed source-clock spans."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from terasort.session_quality import evaluate_case, evaluate_suite, score_sorting
from terasort.session_store import SPIKE_DTYPE


class SessionQualityTests(unittest.TestCase):
    def test_unit_recovery_and_collision_diagnostics(self):
        gt_t = np.array([100, 102, 200, 300, 302, 400], np.int64)
        gt_l = np.array([0, 1, 0, 0, 1, 1], np.int64)
        report = score_sorting(gt_t, gt_l, gt_t, gt_l,
                               tolerance_samples=1,
                               collision_tolerance_samples=3)
        self.assertEqual(report["recovered_units_iou_0p8"], 2)
        self.assertEqual(report["collision_recovered_spikes"],
                         report["collision_gt_spikes"])
        self.assertEqual(report["split_gt_units"], 0)
        self.assertEqual(report["merged_predicted_units"], 0)
        self.assertEqual(report["spike_precision"], 1.)

    def test_gap_excluded_from_both_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "session" / "probeA"
            output.mkdir(parents=True)
            for first, stop, spike in [(0, 1000, 100), (2000, 3000, 2100)]:
                path = output / f"{first:016d}-{stop:016d}.h5"
                with h5py.File(path, "w") as handle:
                    handle.attrs["complete"] = True
                    handle.attrs["start_sample"] = first
                    handle.attrs["stop_sample"] = stop
                    handle.attrs["day_id"] = "day1"
                    row = np.zeros(1, SPIKE_DTYPE)
                    row["sample_index"] = spike
                    row["unit_id"] = 0
                    handle.create_dataset("spikes", data=row)
            truth = root / "ground_truth.npz"
            np.savez(truth, times=np.array([100, 1500, 2100]),
                     labels=np.array([0, 0, 0]), sampling_frequency=20_000.)
            ks = root / "kilosort"
            ks.mkdir()
            np.save(ks / "spike_times.npy", np.array([100, 1500, 2100]))
            np.save(ks / "spike_clusters.npy", np.array([0, 0, 0]))
            case = {"name": "gap", "kind": "drift",
                    "ground_truth": str(truth),
                    "session_output": str(root / "session"),
                    "probe_id": "probeA", "kilosort_dir": str(ks)}
            result = evaluate_case(case)
            self.assertEqual(result["session"]["spike_recall"], 1.)
            self.assertEqual(result["kilosort4"]["spike_recall"], 1.)
            self.assertEqual(result["session"]["gt_units"], 1)
            suite = root / "suite.json"
            suite.write_text(json.dumps({"cases": [case]}))
            self.assertFalse(evaluate_suite(suite)["quality_gate_passed"])


if __name__ == "__main__":
    unittest.main()
