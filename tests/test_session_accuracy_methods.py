import numpy as np
import pytest

from terasort.session_models import LocalModels,Match
from terasort.session_local_refit import local_refit
from terasort.session_refinement import StratifiedEvidence
from terasort.session_pca_split import split_templates


def test_local_refit_corrects_wrong_shape_without_adding_events():
    w=np.zeros((2,61,1),np.float32);w[:,30,0]=10
    w[0,26,0]=5;w[1,34,0]=5
    model=LocalModels(w,[[0],[0]],[0,0],[0,0])
    signal=np.zeros((400,1),np.float32);signal[170:231]=w[1]
    old=Match(200,0,0,.7,.69,.8,80,0,200,.05)
    actual=local_refit(signal,np.ones(1),model,[old],core_start=0,core_stop=400)
    assert len(actual)==1 and actual[0].unit==1 and actual[0].source_sample==200
    assert local_refit(signal,np.ones(1),model,[old],core_start=0,core_stop=400,rounds=0)==[old]


def test_local_refit_preserves_refractory_and_core_edge():
    w=np.zeros((1,61,1),np.float32);w[0,30,0]=10
    model=LocalModels(w,[[0]],[0],[0])
    signal=np.zeros((400,1),np.float32)
    old=Match(20,0,0,.7,-1,1,100,0,20,.05)
    assert local_refit(signal,np.ones(1),model,[old],core_start=0,core_stop=400)==[old]


def test_local_refit_uses_neighbor_subtraction():
    w=np.zeros((3,61,1),np.float32);w[:2,30,0]=10
    w[0,26,0]=5;w[1,34,0]=5;w[2,30,0]=8
    model=LocalModels(w,[[0],[0],[0]],[0,0,0],[0,0,0])
    signal=np.zeros((400,1),np.float32);signal[170:231]+=w[1];signal[166:227]+=w[2]
    old=Match(200,0,0,.7,.69,.8,80,0,200,.05)
    neighbor=Match(196,0,2,1.,-1,1.,64,0,196,1.)
    actual=local_refit(signal,np.ones(1),model,[old,neighbor],core_start=0,core_stop=400)
    assert actual[0]==neighbor and actual[1].unit==1 and len(actual)==2


def test_local_refit_can_correct_amplitude_with_same_identity():
    w=np.zeros((1,61,1),np.float32);w[0,30,0]=10
    model=LocalModels(w,[[0]],[0],[0])
    signal=np.zeros((400,1),np.float32);signal[200,0]=10
    old=Match(200,0,0,.7,-1,.5,75,0,200,.05)
    new=local_refit(signal,np.ones(1),model,[old],core_start=0,core_stop=400)
    assert new[0].unit==0 and new[0].source_sample==200 and new[0].amplitude==1


def test_local_refit_rejects_a_refractory_collision_with_existing_unit():
    w=np.zeros((2,61,1),np.float32);w[:,30,0]=10;w[0,26,0]=5;w[1,34,0]=5
    model=LocalModels(w,[[0],[0]],[0,0],[0,0])
    signal=np.zeros((400,1),np.float32);signal[170:231]+=w[1];signal[175:236]+=w[1]
    old=Match(200,0,0,.7,-1,.8,80,0,200,.05)
    neighbor=Match(205,0,1,1,-1,1,125,0,205,1.)
    new=local_refit(signal,np.ones(1),model,[old,neighbor],core_start=0,core_stop=400)
    assert new==[old,neighbor]


def test_pca_split_requires_repeatable_shape_groups():
    w=np.zeros((1,61,1),np.float32);w[0,30,0]=10
    model=LocalModels(w,[[0]],[0],[0])
    train=StratifiedEvidence(1,[100,200,300,400],per_window=16)
    valid=StratifiedEvidence(1,[600,700],per_window=16)
    for e in (train,valid):
        for c in e.windows:
            for t in range(16):
                wave=w[0].copy();wave[28,0]=2 if t%2 else -2
                e.observe(0,c+t,c,wave)
    new,audit=split_templates(model,train,valid)
    assert len(new.waveforms)==2 and audit[0]['split']
    unchanged,audit=split_templates(model,train,valid,max_new=0)
    assert len(unchanged.waveforms)==1 and audit[0]['reason']=='model_budget'
    with pytest.raises(ValueError,match='Disjoint'):
        split_templates(model,train,train)


def test_pca_does_not_split_identical_waveforms():
    w=np.zeros((1,61,1),np.float32);w[0,30,0]=10
    model=LocalModels(w,[[0]],[0],[0])
    train=StratifiedEvidence(1,[100,200,300,400],per_window=16)
    valid=StratifiedEvidence(1,[600,700],per_window=16)
    for e in (train,valid):
        for c in e.windows:
            for t in range(16):e.observe(0,c+t,c,w[0])
    new,audit=split_templates(model,train,valid)
    assert len(new.waveforms)==1 and not audit[0]['split']
