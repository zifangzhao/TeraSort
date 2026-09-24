import numpy as np

from terasort.session_models import LocalModels
from terasort.session_interference import InterferenceScheduler


def models():
    w = np.zeros((3,61,1), np.float32)
    w[0,30,0] = 1
    w[1,30:32,0] = [.006,1]
    w[2,30,0] = .006
    w[2,32,0] = 1
    return LocalModels(w, [[0],[0],[0]], [0,0,0], [0,0,0])


def test_cumulative_interference_is_transactional():
    s = InterferenceScheduler(models(), [1.])
    assert s.admit(100,1,1.) == (0,None)
    assert s.admit(100,2,1.) == (1,None)
    charges = [r['charge'] for r in s.fits]
    assert s.admit(100,0,1.)[1] is not None
    assert [r['charge'] for r in s.fits] == charges


def test_amplitude_ratio_and_cache_budget_cannot_relax_rules():
    s = InterferenceScheduler(models(), [1.])
    s.admit(100,1,3.)
    assert s.admit(100,0,.3)[1] is not None
    s = InterferenceScheduler(models(), [1.], cache_limit=0)
    s.admit(100,1,1.)
    assert s.admit(100,2,1.)[1] is not None


def test_temporally_disjoint_fits_need_no_coupling_entry():
    s = InterferenceScheduler(models(), [1.])
    s.admit(100,0,1.)
    assert s.admit(161,0,1.) == (0,None)
    assert not s.cache


def test_low_confidence_fit_does_not_relax_geometric_exclusion():
    s = InterferenceScheduler(models(), [1.])
    s.admit(100,1,1., score=.7)
    # A previously admitted weak fit has no veto over a stronger, weakly
    # coupled candidate; both still obey the cumulative projection budget.
    assert s.admit(100,2,1.) == (1,None)
    s.reset()
    s.admit(100,1,1.)
    assert s.admit(100,2,1., margin=.05)[1] is not None


def test_subtraction_launch_groups_are_bounded():
    w = np.eye(61, dtype=np.float32)[:4, :, None]
    m = LocalModels(w, np.zeros((4,1), np.int32), np.zeros(4,np.int32), np.zeros(4,np.int64))
    s = InterferenceScheduler(m, [1.], max_colors=2)
    assert s.admit(100,0,1.) == (0,None)
    assert s.admit(100,1,1.) == (1,None)
    assert s.admit(100,2,1.)[1] is not None
    assert len(s.fits) == 2
