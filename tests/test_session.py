"""Multi-file source mapping, including exact boundary ownership."""

import numpy as np
import pytest

from terasort.session import locate_spikes, prepare_session, write_session_manifest


def test_session_boundaries_and_mapping(tmp_path):
    first, second = tmp_path / "first.bin", tmp_path / "second.bin"
    np.arange(12, dtype=np.int16).tofile(first)  # 6 two-channel samples
    np.arange(8, dtype=np.int16).tofile(second)   # 4 two-channel samples
    manifest = prepare_session([first, second], {"n_chan_bin": 2, "fs": 20000})
    assert [(s["virtual_start_sample"], s["virtual_stop_sample"]) for s in manifest["sources"]] == [(0, 6), (6, 10)]
    assert manifest["sources"][0]["gap_after_samples"] is None
    indices, local = locate_spikes(np.array([0, 5, 6, 9]), manifest)
    np.testing.assert_array_equal(indices, [0, 0, 1, 1])
    np.testing.assert_array_equal(local, [0, 5, 0, 3])
    with pytest.raises(ValueError, match="outside"):
        locate_spikes(np.array([10]), manifest)
    target = write_session_manifest(tmp_path / "output", manifest)
    assert target.is_file()


def test_session_rejects_duplicate_or_bad_frames(tmp_path):
    first, second = tmp_path / "first.bin", tmp_path / "second.bin"
    first.write_bytes(b"\0" * 8)
    second.write_bytes(b"\0" * 3)
    with pytest.raises(ValueError, match="same source"):
        prepare_session([first, first], {"n_chan_bin": 2, "fs": 20000})
    with pytest.raises(ValueError, match="incomplete"):
        prepare_session([first, second], {"n_chan_bin": 2, "fs": 20000})
