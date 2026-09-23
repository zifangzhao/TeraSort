"""Source-preserving local staging for repeated remote reads."""

import json
import hashlib

import pytest

from terasort.staging import stage_inputs


def test_stages_ordered_files_without_overwriting_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    files = [source / "a.bin", source / "b.bin"]
    files[0].write_bytes(b"first" * 100)
    files[1].write_bytes(b"second" * 70)
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
    scratch = tmp_path / "scratch"
    updates = []
    staged, manifest = stage_inputs(files, scratch, progress=updates.append)
    assert updates[0] == 0 and updates[-1] == 100
    assert [p.read_bytes() for p in staged] == [item[0] for item in before]
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in files] == before
    assert [p.name for p in staged] == ["0000_a.bin", "0001_b.bin"]
    assert [r["source"] for r in manifest["sources"]] == [str(p) for p in files]
    assert [r["sha256"] for r in manifest["sources"]] == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    assert json.loads((scratch / "staging_manifest.json").read_text())["total_bytes"] == 920
    with pytest.raises(FileExistsError, match="must be new"):
        stage_inputs(files, scratch)


def test_staging_rejects_writes_inside_source_folder(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    recording = source / "recording.bin"
    recording.write_bytes(b"data")
    with pytest.raises(ValueError, match="outside the source"):
        stage_inputs([recording], source / "scratch")
    assert not (source / "scratch").exists()
