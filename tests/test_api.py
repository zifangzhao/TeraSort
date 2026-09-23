"""Public compatibility behavior; no recording or CUDA required."""

import inspect

import kilosort
import pytest

from terasort import DEFAULT_SETTINGS, run_kilosort
from terasort.api import _select_backend, _use_int16_reader, available_backends
from terasort.cli import main


def test_kilosort_signature_and_pass_through(monkeypatch, tmp_path):
    upstream = inspect.signature(kilosort.run_kilosort)
    ours = inspect.signature(run_kilosort)
    assert tuple(ours.parameters)[:len(upstream.parameters)] == tuple(upstream.parameters)
    assert DEFAULT_SETTINGS is kilosort.DEFAULT_SETTINGS
    expected = tuple(range(9))
    calls = []
    monkeypatch.setattr(kilosort, "run_kilosort", lambda *args, **kwargs: calls.append((args, kwargs)) or expected)
    filename = tmp_path / "recording.bin"
    probe = {"chanMap": [0]}
    actual = run_kilosort({"n_chan_bin": 1}, probe=probe, filename=filename,
                          results_dir=tmp_path / "out", backend="standard")
    assert actual is expected
    assert calls[0][0] == ({"n_chan_bin": 1},)
    assert calls[0][1]["probe"] is probe
    assert calls[0][1]["filename"] == filename
    assert calls[0][1]["results_dir"] == tmp_path / "out"


def test_backend_and_reader_guard():
    assert _select_backend("auto", "cpu", {}) == "standard"
    assert _select_backend("auto", None, {"nt": 41}) == "standard"
    assert "standard" in available_backends()
    with pytest.raises(ValueError, match="Unknown"):
        _select_backend("missing", None, {})
    assert _use_int16_reader("x.bin", None, None, "cublas", True)
    assert not _use_int16_reader(["x.bin"], None, None, "cublas", True)
    assert not _use_int16_reader("x.bin", None, "float32", "cublas", True)
    assert not _use_int16_reader("x.bin", object(), "int16", "cublas", True)


def test_drift_switch_uses_kilosort_setting_without_mutating_caller(monkeypatch):
    calls = []
    monkeypatch.setattr(kilosort, "run_kilosort", lambda settings, **kwargs: calls.append(settings) or ())
    settings = {"n_chan_bin": 1, "nblocks": 3}
    run_kilosort(settings, backend="standard", skip_drift_correction=True)
    assert calls[-1] == {"n_chan_bin": 1, "nblocks": 0}
    assert settings["nblocks"] == 3
    run_kilosort(settings, backend="standard")
    assert calls[-1] is settings
    assert calls[-1]["nblocks"] == 3


def test_cli_backends(capsys):
    assert main(["backends"]) == 0
    assert "standard" in capsys.readouterr().out


def test_cli_drift_switch(monkeypatch, tmp_path):
    import terasort.cli as cli
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"n_chan_bin": 1}')
    calls = []
    monkeypatch.setattr(cli, "run_kilosort", lambda *args, **kwargs: calls.append(kwargs))
    assert cli.main(["sort", "--settings", str(settings_file),
                     "--filename", str(tmp_path / "recording.bin"),
                     "--results-dir", str(tmp_path / "out"),
                     "--skip-drift-correction"]) == 0
    assert calls[-1]["skip_drift_correction"] is True
