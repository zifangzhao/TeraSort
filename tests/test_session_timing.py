from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from terasort.session_timing import AdaptiveShiftRadius


def _match(unit, candidate, shift):
    return SimpleNamespace(unit=unit, candidate_sample=candidate,
                           source_sample=candidate + shift)


def test_adaptive_radius_uses_boundary_support_and_restores_exactly(tmp_path):
    state = AdaptiveShiftRadius(unit_count=2, sample_rate_hz=100., base_radius=2)
    np.testing.assert_array_equal(state.for_core(), [3, 3])
    matches = ([_match(0, i * 10, 2) for i in range(10)] +
               [_match(1, i * 10, 3) for i in range(20)])
    assert state.observe(matches, state.pilot_samples)
    np.testing.assert_array_equal(state.radii, [2, 3])
    assert state.fitted

    state.extend_units(3)
    np.testing.assert_array_equal(state.radii, [2, 3, 2])
    path = tmp_path / "timing.h5"
    with h5py.File(path, "w") as handle:
        state.save(handle.create_group("adaptive_shift"))

    resumed = AdaptiveShiftRadius(unit_count=3, sample_rate_hz=100., base_radius=2)
    with h5py.File(path, "r") as handle:
        resumed.restore(handle["adaptive_shift"], 3)
    np.testing.assert_array_equal(resumed.for_core(), [2, 3, 2])
    np.testing.assert_array_equal(resumed.observations, state.observations)
    np.testing.assert_array_equal(resumed.boundary_fits, state.boundary_fits)


def test_bad_intervals_advance_pilot_without_training_templates():
    state = AdaptiveShiftRadius(unit_count=1, sample_rate_hz=100., base_radius=1)
    completed = state.observe([_match(0, 5, 2)], state.pilot_samples,
                              interval_good=False)
    assert completed
    np.testing.assert_array_equal(state.observations, [0])
    np.testing.assert_array_equal(state.radii, [1])


@pytest.mark.parametrize("base", [-1, 8, 2.0])
def test_adaptive_radius_rejects_invalid_base(base):
    with pytest.raises(ValueError):
        AdaptiveShiftRadius(1, 30000., base)
