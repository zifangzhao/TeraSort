"""End-to-end checks for causal SCB consolidation and local template assignment."""

import h5py
import numpy as np
import pytest

from terasort.candidates.bank import BankWriter, CandidateBank, events_from_indices
from terasort.candidates.waveforms import WaveformSpec, extract_waveforms
from terasort.streaming import (AssignmentWriter, ConsolidatedEvent,
                                LocalTemplateBank, _process_epoch,
                                iter_consolidated, sort_candidate_bank)


def make_fixture(tmp_path, *, same_templates=False):
    ns, nc, nt = 700, 4, 61
    source = np.zeros((ns, nc), np.float32)
    x = np.arange(nt) - nt // 2
    pulse = -np.exp(-(x / 2.5) ** 2).astype(np.float32)
    templates = np.zeros((2, nt, nc), np.float32)
    templates[0, :, 0] = pulse * 9
    templates[0, :, 1] = pulse * 5
    templates[1, :, 2] = pulse * 8
    templates[1, :, 3] = pulse * 4
    if same_templates:
        templates[1] = templates[0]
    centers = [(100, 0), (190, 1), (280, 0), (370, 1), (460, 0), (550, 1)]
    indices = []
    for t, unit in centers:
        source[t-30:t+31] += templates[unit]
        for ch in ([0, 1] if unit == 0 or same_templates else [2, 3]):
            indices.append(t * nc + ch)
    if same_templates:
        source[:] = 0
        for t, _ in centers:
            source[t-30:t+31] += templates[0]
    events = events_from_indices(indices, source, np.ones(nc))
    channel_map = np.array([[0, 1, -1, -1], [1, 0, -1, -1],
                            [2, 3, -1, -1], [3, 2, -1, -1]], np.int32)
    spec = WaveformSpec(channel_map, 30, 30, .1, "int16")
    values, starts, stops = extract_waveforms(events, source, 0, spec, 0, ns)
    path = tmp_path / "synthetic.scb.h5"
    with BankWriter(path, {"floor_snr": 3., "sample_rate_hz": 20_000},
                    np.ones(nc), waveform_spec=spec) as writer:
        writer.append(events[:6], 0, 300, values[:6], starts[:6], stops[:6])
        writer.append(events[6:], 300, ns, values[6:], starts[6:], stops[6:])
    return path, templates


def test_consolidation_crosses_hdf5_query_batches_and_preserves_rows(tmp_path):
    path, _ = make_fixture(tmp_path)
    bank = CandidateBank(path)
    grouped = list(iter_consolidated(bank, floor_snr=3., batch_size=1))
    assert len(grouped) == 6
    assert [len(e.candidate_rows) for e in grouped] == [2] * 6
    assert sorted(r for e in grouped for r in e.candidate_rows) == list(range(12))
    assert [e.anchor_channel for e in grouped] == [0, 2, 0, 2, 0, 2]


def test_streaming_assignment_updates_clean_templates_and_keeps_source(tmp_path):
    path, templates = make_fixture(tmp_path)
    before = path.stat().st_size
    output = tmp_path / "assignments.h5"
    report = sort_candidate_bank(path, output, templates=templates,
                                 max_epoch_events=2, bin_seconds=1)
    assert report["candidate_rows"] == 12
    assert report["consolidated_events"] == 6
    assert report["assigned_events"] == 6
    assert report["template_updates"] == 6
    assert path.stat().st_size == before
    with h5py.File(output) as f:
        assert f.attrs["complete"]
        events = f["events"][:]
        np.testing.assert_array_equal(events["sample_index"], [100, 190, 280, 370, 460, 550])
        np.testing.assert_array_equal(events["unit_id"], [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(f["candidate_rows"][:], np.arange(12))
        assert (f["final_templates/version"][:] > 0).all()
        np.testing.assert_array_equal(f["final_templates/assigned_count"][:], [3, 3])
    with pytest.raises(FileExistsError):
        sort_candidate_bank(path, output, templates=templates)


def test_ambiguous_template_is_unknown_and_never_updates(tmp_path):
    path, templates = make_fixture(tmp_path, same_templates=True)
    output = tmp_path / "unknown.h5"
    report = sort_candidate_bank(path, output, templates=templates)
    assert report["consolidated_events"] == 6
    assert report["assigned_events"] == 0
    assert report["template_updates"] == 0
    with h5py.File(output) as f:
        assert (f["events"]["unit_id"][:] == -1).all()


def test_slot_overflow_fails_before_output_creation(tmp_path):
    path, templates = make_fixture(tmp_path)
    crowded = np.repeat(templates[:1], 33, axis=0)
    output = tmp_path / "overflow.h5"
    with pytest.raises(OverflowError, match="exceeds 32"):
        sort_candidate_bank(path, output, templates=crowded)
    assert not output.exists()


def test_no_seed_preserves_every_event_as_unknown(tmp_path):
    path, _ = make_fixture(tmp_path)
    report = sort_candidate_bank(path, tmp_path / "no_seed.h5")
    assert report["consolidated_events"] == 6
    assert report["assigned_events"] == 0


def test_temporal_neighbor_across_epoch_boundary_cannot_update_template(tmp_path):
    _, templates = make_fixture(tmp_path)
    channel_map = np.array([[0, 1, -1, -1], [1, 0, -1, -1],
                            [2, 3, -1, -1], [3, 2, -1, -1]], np.int32)
    bank = LocalTemplateBank.from_dense_templates(
        templates, channel_map, pre_samples=30, sample_rate_hz=20_000)
    waveform = np.zeros((61, 4), np.float32)
    waveform[:, :2] = templates[0, :, :2]
    e1 = ConsolidatedEvent(100, 100, 0, -9., 9., channel_map[0], waveform, (0,))
    e2 = ConsolidatedEvent(110, 110, 0, -9., 9., channel_map[0], waveform, (1,))
    writer = AssignmentWriter(tmp_path / "neighbor.h5", source="fixture", config={})
    stop, assigned1, updated1 = _process_epoch([e1], bank, writer, 0, next_event=e2)
    stop, assigned2, updated2 = _process_epoch([e2], bank, writer, stop,
                                               previous_event=e1)
    writer.finish(bank)
    assert stop == 111
    assert assigned1 + assigned2 == 2
    assert updated1 + updated2 == 0


def test_cuda_matches_cpu_assignments_and_template_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy_cache"))
    pytest.importorskip("cupy")
    path, templates = make_fixture(tmp_path)
    cpu = tmp_path / "cpu.h5"
    gpu = tmp_path / "gpu.h5"
    a = sort_candidate_bank(path, cpu, templates=templates, max_epoch_events=2)
    b = sort_candidate_bank(path, gpu, templates=templates,
                            max_epoch_events=2, backend="cuda")
    assert a["assigned_events"] == b["assigned_events"]
    assert a["template_updates"] == b["template_updates"]
    with h5py.File(cpu) as x, h5py.File(gpu) as y:
        for field in ("sample_index", "unit_id", "template_version",
                      "candidate_start", "candidate_stop"):
            np.testing.assert_array_equal(x["events"][field][:], y["events"][field][:])
        np.testing.assert_allclose(x["events"]["score"][:],
                                   y["events"]["score"][:], atol=2e-5)
        np.testing.assert_allclose(x["final_templates/waveform"][:],
                                   y["final_templates/waveform"][:], atol=2e-5)
