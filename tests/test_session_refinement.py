import numpy as np
import pytest

from terasort.session_models import LocalModels
from terasort.session_refinement import StratifiedEvidence, refine_templates


def bank():
    wave = np.zeros((1,61,1), np.float32)
    wave[0,30,0] = 10
    return LocalModels(wave, [[0]], [0], [0])


def evidence(wave, windows):
    e = StratifiedEvidence(1, windows)
    for core in windows:
        for t in range(8):
            e.observe(0, core+t, core, wave)
    return e


def test_refinement_improves_independent_waveform_and_preserves_ids():
    m = bank()
    target = m.waveforms[0].copy()
    target[29] = 1
    train, valid = evidence(target, [100,200]), evidence(target, [300])
    audit = refine_templates(m, train, valid)
    assert audit[0]['promoted'] and m.version == 1
    assert m.waveforms[0,29,0] == .25
    assert m.anchors.tolist() == [0]


def test_refinement_rejects_training_contamination_and_partition_leak():
    m = bank()
    target = m.waveforms[0].copy()
    target[29] = 1
    train = evidence(target, [100,200])
    audit = refine_templates(m, train, evidence(m.waveforms[0], [300]))
    assert not audit[0]['promoted'] and m.version == 0
    with pytest.raises(ValueError, match='disjoint'):
        refine_templates(m, train, train)


def test_stratified_cap_is_order_independent_and_deduplicated():
    a = StratifiedEvidence(2, [0,100], budget=12)
    b = StratifiedEvidence(2, [0,100], budget=12)
    for e, order in [(a, range(20)), (b, reversed(range(20)))]:
        for t in order:
            for c in (0,100):
                for unit in range(2):
                    e.observe(unit, c+t, c, np.array([t]))
                    e.observe(unit, c+t, c, np.array([t]))
    assert sum(map(len,a.rows.values())) == 12
    np.testing.assert_array_equal(a.samples(0), b.samples(0))
    with pytest.raises(ValueError, match='strata'):
        a.observe(0,300,300,np.array([0]))


def test_refinement_does_not_learn_from_insufficient_support():
    m = bank()
    audit = refine_templates(m, evidence(m.waveforms[0], [100]),
                             evidence(m.waveforms[0], [300]))
    assert not audit[0]['promoted'] and m.version == 0


def test_refinement_rejects_a_peak_shift_on_seed_reload():
    m = bank()
    m.waveforms[0,29,0] = 9.9
    target = m.waveforms[0].copy()
    target[29] = 10.5
    audit = refine_templates(m, evidence(target, [100,200]), evidence(target, [300]))
    assert audit[0]['reason'] == 'alignment_changed'
    assert m.version == 0 and m.waveforms[0,29,0] == np.float32(9.9)
