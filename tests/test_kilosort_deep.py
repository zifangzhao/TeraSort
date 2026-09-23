"""Exact score reductions and disjoint in-place learned-template updates."""
import numpy as np
import pytest


@pytest.fixture(scope='module')
def backend():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from terasort.kilosort_deep import DeepCuda
    return torch,DeepCuda()


@pytest.mark.parametrize('parallel',[False,True])
@pytest.mark.parametrize('strided',[False,True])
def test_learned_reduction_nan_ties_edges_and_strides(backend,parallel,strided):
    torch,native=backend
    native.parallel=parallel
    rng=np.random.default_rng(39)
    B=torch.as_tensor(rng.normal(size=(19,178)).astype('float32'),device='cuda')
    norm=torch.as_tensor(rng.uniform(.2,3,size=38).astype('float32'),device='cuda')[::2]
    if strided: B=B[:,::2]
    B[:,10]=0
    B[0,20]=float('nan')
    B[5,20]=float('nan')
    B[:,21]=0; B[1,21]=float('inf'); B[2,21]=float('inf')
    C=torch.relu(B)**2/norm[:,None]
    C[:,:5]=0; C[:,-5:]=0
    expected=C.max(0)
    actual=native.learned_max(B,norm,5)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)


@pytest.mark.parametrize('strided',[False,True])
def test_disjoint_correlation_subtraction_exact(backend,strided):
    torch,native=backend
    rng=np.random.default_rng(9)
    B=torch.as_tensor(rng.normal(size=(7,152)).astype('float32'),device='cuda')
    ctc=torch.as_tensor(rng.normal(size=(7,9,14)).astype('float32'),device='cuda')
    if strided: B=B[:,::2]
    ctc=ctc[:,:,::2]
    times=torch.tensor([4,0,17,0,32,0,60,0],device='cuda')[::2]
    units=torch.tensor([0,0,3,0,6,0,8,0],device='cuda')[::2]
    amps=torch.tensor([.3,0,-.5,0,2,0,.8,0],device='cuda')[::2]
    expected=B.clone()
    expected[:,times[:,None]+torch.arange(-3,4,device='cuda')]-=amps[:,None]*ctc[:,units,:]
    native.subtract_correlations(B,ctc,times,units,amps,3)
    torch.testing.assert_close(B,expected,rtol=0,atol=0)


def test_subtraction_overlap_fallback_and_bounds(backend):
    torch,native=backend
    B=torch.ones((2,23),device='cuda')
    ctc=torch.ones((2,3,7),device='cuda')
    times=torch.tensor([4,5],device='cuda')
    units=torch.tensor([0,1],device='cuda')
    amps=torch.ones(2,device='cuda')
    expected=B.clone()
    expected[:,times[:,None]+torch.arange(-3,4,device='cuda')]-=amps[:,None]*ctc[:,units,:]
    count=native.fallback_count
    native.subtract_correlations(B,ctc,times,units,amps,3)
    assert native.fallback_count==count+1
    torch.testing.assert_close(B,expected,rtol=0,atol=0)
    with pytest.raises(ValueError,match='beyond'):
        native.subtract_correlations(B,ctc,torch.tensor([0,10],device='cuda'),units,amps,3)
    with pytest.raises(ValueError,match='Unit index'):
        native.subtract_correlations(B,ctc,torch.tensor([4,15],device='cuda'),torch.tensor([0,3],device='cuda'),amps,3)


def test_waveform_cache_invalidates_after_template_mutation(backend):
    torch,native=backend
    U=torch.arange(4*6*8,device='cuda',dtype=torch.float32).reshape(4,6,8)/100
    W=torch.arange(6*13,device='cuda',dtype=torch.float32).reshape(13,6).T/100
    assert not W.is_contiguous()
    before=native.waveform_builds
    a=native.waveforms(U,W)
    assert native.waveforms(U,W) is a
    # Kilosort makes a new contiguous copy each batch; the cache key must be
    # the persistent original PCA tensor rather than any of those copies.
    for _ in range(3):
        per_batch=W.contiguous()
        assert per_batch is not W
        assert native.waveforms(U,W) is a
    assert native.waveform_builds==before+1
    U[0,0,0]+=1
    b=native.waveforms(U,W)
    assert b is not a
    torch.testing.assert_close(b,torch.einsum('upc,pt->cut',U,W).contiguous(),rtol=0,atol=0)
    W[1,2]-=1
    c=native.waveforms(U,W)
    assert c is not b
    torch.testing.assert_close(c,torch.einsum('upc,pt->cut',U,W).contiguous(),rtol=0,atol=0)


def test_deep_context_restores_both_functions(backend):
    from kilosort import spikedetect,template_matching
    from terasort.kilosort_deep import deep_kilosort
    a,b=spikedetect.template_match,template_matching.run_matching
    with pytest.raises(RuntimeError,match='interrupt'):
        with deep_kilosort():
            assert spikedetect.template_match is not a
            assert template_matching.run_matching is not b
            raise RuntimeError('interrupt')
    assert spikedetect.template_match is a
    assert template_matching.run_matching is b
