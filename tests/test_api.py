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


def test_cli_backends(capsys):
    assert main(["backends"]) == 0
    assert "standard" in capsys.readouterr().out
