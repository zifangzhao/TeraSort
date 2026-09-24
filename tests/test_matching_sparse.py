"""Bounded spatial candidate generation and selective waveform scoring."""

import numpy as np
import pytest

from terasort.matching import iter_spatial_pairs, score_template_pairs, template_centers


def test_spatial_pairs_are_probe_safe_batched_and_complete():
    left = np.array([[0., 0.], [0., 50.], [0., 100.]])
    right = np.array([[0., 20.], [0., 60.], [0., 101.]])
    batches = list(iter_spatial_pairs(left, right, 45, left_probe=[1, 1, 2],
                                      right_probe=[1, 2, 2], pair_batch=2))
    assert all(len(batch) <= 2 for batch in batches)
    assert np.array_equal(np.concatenate(batches), [[0, 0], [1, 0], [2, 1], [2, 2]])
    with pytest.raises(OverflowError):
        list(iter_spatial_pairs(left, right, 200, max_neighbors_per_unit=1))


def test_template_centers_and_scores_do_not_depend_on_batch_size():
    rng = np.random.default_rng(23)
    left = rng.normal(size=(5, 11, 4)).astype(np.float32)
    right = rng.normal(size=(6, 11, 4)).astype(np.float32)
    positions = np.array([[0., 0.], [20., 0.], [0., 20.], [20., 20.]])
    assert np.allclose(template_centers(left, positions, batch_size=1),
                       template_centers(left, positions, batch_size=5))
    pairs = np.array([(i, j) for i in range(len(left)) for j in range(len(right))])
    assert np.array_equal(score_template_pairs(left, right, pairs, batch_size=3),
                          score_template_pairs(left, right, pairs, batch_size=30))
