"""Causal rolling template update, dormancy, and checkpoint behavior."""

import numpy as np
import pytest

from terasort.matching.rolling import RollingTemplate


def bank():
    return RollingTemplate(np.zeros((3, 2), np.float32), sample_rate_hz=1,
                           coordinate_frame="probe1:aligned:v1",
                           window_seconds=30, bin_seconds=5,
                           min_confidence=.9)


def commit_one(state, bin_id, value, *, confidence=1):
    snap = state.snapshot()
    return state.commit(bin_id * 5, (bin_id + 1) * 5,
                        mean_waveform=np.full((3, 2), value, np.float32),
                        spike_count=10, confidence=confidence,
                        coordinate_frame=snap.coordinate_frame,
                        expected_version=snap.version)


def test_rolling_window_updates_only_after_commit_and_expires_old_bin():
    state = bank()
    before = state.snapshot()
    assert not before.active and np.all(before.waveform == 0)
    for bin_id in range(6):
        assert commit_one(state, bin_id, bin_id + 1)
    assert np.allclose(state.snapshot().waveform, 3.5)
    assert np.all(before.waveform == 0)  # Frozen assignment snapshot.
    assert commit_one(state, 6, 7)
    assert np.allclose(state.snapshot().waveform, 4.5)
    assert state.snapshot().recent_spike_count == 60


def test_rejected_updates_do_not_contaminate_or_decay_dormant_unit():
    state = bank()
    assert commit_one(state, 0, 2)
    assert not commit_one(state, 1, 200, confidence=.2)
    assert np.allclose(state.snapshot().waveform, 2)
    snap = state.snapshot()
    state.commit(10, 40, coordinate_frame=snap.coordinate_frame,
                 expected_version=snap.version)
    dormant = state.snapshot()
    assert not dormant.active and dormant.recent_spike_count == 0
    assert np.allclose(dormant.waveform, 2)
    assert np.all(dormant.anchor == 0)


def test_stale_frame_and_cross_bin_updates_fail_without_state_change():
    state = bank()
    snap = state.snapshot()
    with pytest.raises(ValueError):
        state.commit(0, 5, mean_waveform=np.ones((3, 2)), spike_count=10,
                     confidence=1, coordinate_frame="wrong", expected_version=0)
    with pytest.raises(ValueError):
        state.commit(0, 6, mean_waveform=np.ones((3, 2)), spike_count=10,
                     confidence=1, coordinate_frame=snap.coordinate_frame,
                     expected_version=0)
    assert state.snapshot().version == 0
    assert commit_one(state, 0, 1)
    with pytest.raises(ValueError):
        state.commit(5, 10, mean_waveform=np.ones((3, 2)), spike_count=10,
                     confidence=1, coordinate_frame=snap.coordinate_frame,
                     expected_version=snap.version)


def test_checkpoint_restores_template_and_future_updates():
    state = bank()
    commit_one(state, 0, 1)
    commit_one(state, 1, 3)
    restored = RollingTemplate.from_state_dict(state.state_dict())
    assert np.array_equal(restored.snapshot().waveform, state.snapshot().waveform)
    assert restored.snapshot().version == state.snapshot().version
    commit_one(restored, 2, 5)
    commit_one(state, 2, 5)
    assert np.array_equal(restored.snapshot().waveform, state.snapshot().waveform)


def test_large_source_sample_coordinate_survives_checkpoint():
    start = 50_000_000_000
    state = RollingTemplate(np.ones((3, 2), np.float32), sample_rate_hz=1,
                            coordinate_frame="probe1:aligned:v1", start_sample=start,
                            window_seconds=30, bin_seconds=5)
    state.commit(start, start + 5, mean_waveform=np.full((3, 2), 2),
                 spike_count=20, confidence=1,
                 coordinate_frame="probe1:aligned:v1", expected_version=0)
    restored = RollingTemplate.from_state_dict(state.state_dict())
    assert restored.snapshot().through_sample == start + 5
    assert np.all(restored.snapshot().waveform == 2)
