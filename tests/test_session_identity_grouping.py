import numpy as np
import pytest

from terasort.session_gpu import _best_template_per_identity
from terasort.session_models import collapse_identity_scores, normalize_template_identity_map


def test_gpu_ranking_collapses_alternate_waveforms_by_identity():
    event_ids = np.array([0, 0, 0, 1, 1], np.int32)
    template_ids = np.array([0, 1, 2, 0, 2], np.int32)
    gains = np.array([10., 9.8, 9.9, 0., 3.], np.float32)
    event_rank, gain_rank, positions = _best_template_per_identity(
        event_ids, template_ids, gains, 3, np.array([41, 41, 83], np.int32))
    np.testing.assert_array_equal(event_rank, [0, 0, 1])
    np.testing.assert_array_equal(positions, [0, 2, 4])
    np.testing.assert_array_equal(gain_rank, gains[positions])


def test_cpu_ranking_uses_best_template_per_identity_as_runner_up():
    scored = [
        (10., .90, 0, 1., 0),
        (9.8, .99, 1, 1., 0),
        (9.9, .85, 2, 1., 0),
    ]
    collapsed = collapse_identity_scores(scored, np.array([4, 4, 9]))
    assert collapsed == [scored[0], scored[2]]
    best, runner = collapsed
    assert 1. - runner[0] / best[0] > .009


def test_identity_map_adds_distinct_ids_for_later_promoted_templates():
    actual = normalize_template_identity_map(np.array([8, 8, 13]), 5)
    np.testing.assert_array_equal(actual, [8, 8, 13, 14, 15])



def test_cuda_margin_ignores_alternate_waveforms_with_same_identity():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device required")
    from terasort.session_gpu import CudaResidualMatcher
    from terasort.session_models import LocalModels

    x = np.arange(61) - 30
    pulse = (-np.exp(-(x / 2.7) ** 2)).astype(np.float32)
    waveforms = np.zeros((3, 61, 1), np.float32)
    waveforms[:, :, 0] = 20 * pulse
    waveforms[2, 0, 0] = 10
    models = LocalModels(
        waveforms, np.array([[0], [0], [0]], np.int32),
        np.array([0, 0, 0], np.int32), np.zeros(3, np.int64))
    signal = np.zeros((200, 1), np.float32)
    signal[70:131, 0] = waveforms[0, :, 0]

    matcher = CudaResidualMatcher(
        models, template_identity_map=np.array([9, 9, 11], np.int32))
    choice = matcher._score(cp.asarray(signal), [(100, 0, 20.)],
                             score_floor=.8)[0]
    assert choice is not None
    assert choice[2] == 0
    assert choice[6] > .03
