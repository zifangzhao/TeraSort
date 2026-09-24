"""SCB sampling retains immutable row coordinates across HDF5 chunk sizes."""

import json
from pathlib import Path

import h5py
import numpy as np

from terasort.candidates.sampling import infer_depth_axis, sample_bank


def test_scb_sampling_is_reproducible_and_source_is_unchanged(tmp_path):
    bank = tmp_path / "candidates.scb.h5"
    event_dtype = np.dtype([("sample_index", "<i8"), ("channel_index", "<u4"),
                            ("amplitude_uv", "<f4"), ("snr", "<f4")])
    events = np.zeros(120, dtype=event_dtype)
    events["sample_index"] = np.arange(120)
    events["channel_index"] = np.arange(120) % 4
    events["snr"] = 5 + np.arange(120) % 17
    positions = np.array([[0, 0, 0], [20, 0, 0],
                          [0, 40, 0], [20, 40, 0]], dtype=np.float32)
    with h5py.File(bank, "w") as handle:
        handle.attrs["complete"] = True
        handle.attrs["manifest_json"] = json.dumps(
            {"format": "SCB", "version": "0.1", "sample_rate_hz": 20000})
        handle.create_dataset("events", data=events)
        handle.create_dataset("channel_positions_um", data=positions)
        blocks = np.array([(0, 120, 0, 120)], dtype=[("start_sample", "<i8"),
                                                 ("stop_sample", "<i8"),
                                                 ("row_start", "<i8"),
                                                 ("row_stop", "<i8")])
        handle.create_dataset("blocks", data=blocks)
    before = bank.stat().st_size
    a = tmp_path / "sample_a"
    b = tmp_path / "sample_b"
    report = sample_bank(bank, a, budget=30, seed=7, time_bins=4,
                         depth_bin_um=40, minimum_per_stratum=1, chunk_size=7)
    sample_bank(bank, b, budget=30, seed=7, time_bins=4,
                depth_bin_um=40, minimum_per_stratum=1, chunk_size=41)
    selected = np.load(a / "stratified_row_indices.npy")
    assert len(selected) == 30 and np.array_equal(selected,
                                                  np.load(b / "stratified_row_indices.npy"))
    assert report["source_start_sample"] == 0
    assert report["depth_axis"] == 1
    assert bank.stat().st_size == before


def test_depth_axis_can_be_third_scb_coordinate():
    positions = np.array([[0, -20, z] for z in range(0, 200, 20)], dtype=float)
    assert infer_depth_axis(positions) == 2
