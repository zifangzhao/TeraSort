"""LFP filtering, chunk alignment, atomic output, and checkpoint recovery."""

import json

import numpy as np
import pytest
from scipy.signal import freqz, resample_poly

import terasort.lfp as lfp


@pytest.mark.parametrize("sample_rate", [30000, 32000])
@pytest.mark.parametrize("passband_hz", [300.0, 500.0])
@pytest.mark.parametrize("workers", [1, 3])
def test_chunked_lfp_matches_whole_recording(sample_rate, passband_hz, workers, tmp_path):
    raw = np.random.default_rng(11).integers(-12000, 12000, (10013, 3), dtype=np.int16)
    source, output = tmp_path / "raw.i16", tmp_path / "lfp.i16"
    raw.tofile(source)
    result = lfp.export_lfp(source, output, sample_rate_hz=sample_rate,
                            n_channels=3, scale_uv_per_count=0.05,
                            passband_hz=passband_hz, chunk_seconds=0.012,
                            workers=workers)
    up, down, kernel, _ = lfp._filter(sample_rate, 1250, passband_hz)
    reference = resample_poly(raw.astype(np.float32), up, down,
                              axis=0, window=kernel, padtype="constant")
    expected = np.clip(np.rint(reference), -32768, 32767).astype("<i2")
    actual = np.fromfile(output, dtype="<i2").reshape(-1, 3)
    np.testing.assert_array_equal(actual, expected)
    assert result["output_samples"] == len(expected)
    assert result["scale_uv_per_count"] == 0.05
    assert result["filter_workers"] == workers
    assert json.loads((tmp_path / "lfp.i16.json").read_text())["complete"] is True
    assert not (tmp_path / "lfp.i16.partial").exists()


def test_resume_discards_uncheckpointed_tail(tmp_path, monkeypatch):
    raw = np.random.default_rng(12).integers(-1000, 1000, (14001, 2), dtype=np.int16)
    source, output = tmp_path / "raw.i16", tmp_path / "lfp.i16"
    raw.tofile(source)
    original = lfp.resample_poly
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(lfp, "resample_poly", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        lfp.export_lfp(source, output, sample_rate_hz=32000,
                       n_channels=2, chunk_seconds=0.01)
    assert not output.exists()
    checkpoint = json.loads((tmp_path / "lfp.i16.progress.json").read_text())
    assert checkpoint["completed_input_samples"] > 0
    with (tmp_path / "lfp.i16.partial").open("ab") as stream:
        stream.write(b"incomplete tail")
    monkeypatch.setattr(lfp, "resample_poly", original)
    lfp.export_lfp(source, output, sample_rate_hz=32000,
                   n_channels=2, chunk_seconds=0.01, resume=True)
    up, down, kernel, _ = lfp._filter(32000, 1250, 500.0)
    reference = resample_poly(raw.astype(np.float32), up, down, axis=0,
                              window=kernel, padtype="constant")
    np.testing.assert_array_equal(np.fromfile(output, dtype="<i2").reshape(-1, 2),
                                  np.clip(np.rint(reference), -32768, 32767).astype("<i2"))
    assert not (tmp_path / "lfp.i16.progress.json").exists()


@pytest.mark.parametrize("passband_hz", [300.0, 500.0])
def test_filter_passband_and_nyquist_attenuation(passband_hz):
    _, _, kernel, _ = lfp._filter(32000, 1250, passband_hz)
    _, response = freqz(kernel, worN=np.array([passband_hz, 625]), fs=160000)
    loss = 20 * np.log10(np.abs(response))
    assert loss[0] > -0.2
    assert loss[1] < -55


def test_lfp_cli(tmp_path, capsys):
    from terasort.cli import main
    raw = np.zeros((2000, 2), dtype="<i2")
    source, output = tmp_path / "raw.i16", tmp_path / "lfp.i16"
    raw.tofile(source)
    assert main(["lfp", "--filename", str(source), "--output", str(output),
                 "--sample-rate", "32000", "--n-channels", "2"]) == 0
    assert output.stat().st_size == 2 * 2 * 79
    printed = json.loads(capsys.readouterr().out)
    assert printed["output_sample_rate_hz"] == 1250
    assert printed["passband_hz"] == 500.0
