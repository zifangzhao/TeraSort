"""Balanced event sampling must be exact, reproducible, and chunk-size invariant."""

import numpy as np

from terasort.candidates.sampling_core import (allocate_quotas, select_stratified,
                                               select_uniform, stratum_ids)


def test_quotas_exact_and_capped():
    counts = np.array([0, 3, 100, 9, 1])
    quotas = allocate_quotas(counts, 20, 2)
    assert quotas.sum() == 20
    assert np.all(quotas <= counts)
    assert quotas[1] >= 2 and quotas[2] >= 2 and quotas[3] >= 2


def test_reservoir_is_independent_of_scan_chunk_size():
    n = 123
    times = np.arange(n, dtype=np.int64)
    positions = np.stack([np.where(np.arange(n) % 3, 200., 400.),
                          np.where(np.arange(n) % 5, -100., -300.)], axis=1)
    amplitudes = 10. + np.arange(n) % 17
    options = dict(source_samples=n, shank_centers=np.array([200., 400.]),
                   depth_min=-400., depth_max=0., budget=37, seed=19,
                   time_bins=4, depth_bin_um=100., amp_edges=np.array([16., 22.]),
                   min_per_nonempty=1)
    a, wa, counts, quotas = select_stratified(times, positions, amplitudes,
                                               chunk_size=7, **options)
    b, wb, _, _ = select_stratified(times, positions, amplitudes,
                                   chunk_size=41, **options)
    assert len(a) == 37 and np.array_equal(a, b)
    assert np.array_equal(wa, wb)
    assert np.array_equal(select_uniform(n, 37, seed=19, chunk_size=7),
                          select_uniform(n, 37, seed=19, chunk_size=41))
    ids = stratum_ids(times[a], positions[a], amplitudes[a],
                      source_samples=n, time_bins=4, shank_centers=np.array([200., 400.]),
                      depth_min=-400., depth_bin_um=100., depth_bins=4,
                      amp_edges=np.array([16., 22.]))
    assert np.array_equal(np.bincount(ids, minlength=len(quotas)), quotas)
    assert counts.sum() == n
