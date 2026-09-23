"""Boundary, tie, stride, stream and mutation cases for fused GPU reductions."""
import numpy as np
import pytest


@pytest.fixture(scope='module')
def backend():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from terasort.kilosort_native import NativeReductions
    return torch,NativeReductions()


@pytest.mark.parametrize('strided',[False,True])
@pytest.mark.parametrize('mode',['basic','shared','warp'])
def test_neighbor_max_is_exact_and_checks_mutations(backend,strided,mode):
    torch,native = backend
    native.neighbor_mode=mode
    values = torch.as_tensor(np.random.default_rng(4).normal(size=(17,74)).astype('float32'),device='cuda')
    if strided:
        values = values[:,::2]
    values[2,2]=float('nan')
    values[3,4]=float('-inf')
    neighbors = torch.as_tensor(np.random.default_rng(8).integers(0,17,(11,17)),device='cuda')
    neighbors[0]=2
    expected = values[neighbors].max(0).values
    torch.testing.assert_close(native.neighbor_max(values,neighbors),expected,rtol=0,atol=0,equal_nan=True)
    neighbors[0,0] = 17
    with pytest.raises(ValueError,match='out of bounds'):
        native.neighbor_max(values,neighbors)


def test_large_neighbor_set_uses_generic_fallback(backend):
    torch,native=backend
    native.neighbor_mode='shared'
    values=torch.arange(13*37,device='cuda',dtype=torch.float32).reshape(13,37)
    neighbors=torch.arange(300*13,device='cuda').reshape(300,13)%13
    torch.testing.assert_close(native.neighbor_max(values,neighbors),values[neighbors].max(0).values,rtol=0,atol=0)


@pytest.mark.parametrize('block',[64,128,256])
@pytest.mark.parametrize('strided',[False,True])
def test_fused_scores_with_float64_reference(backend,block,strided):
    torch,native=backend
    native.score_block_size=block
    rng=np.random.default_rng(42)
    B=torch.as_tensor(rng.normal(size=(19,6,158)).astype('float32'),device='cuda')
    channels=torch.as_tensor(rng.integers(0,19,(10,11)),device='cuda')
    weights=torch.as_tensor(rng.normal(size=(5,10,11)).astype('float32'),device='cuda')
    if strided:
        B=B[:,:,::2]
        weights=weights[:,:,::2]
        channels=channels[:,::2]
    # Deliberately use a partial time tile and nonzero offset.
    start,stop=3,67
    maxima,indices,raw=native.template_scores(B,channels,weights,start,stop,diagnostic=True)
    reference=torch.einsum('ijk,jklm->iklm',weights.double(),B.double()[channels,:,start:stop])
    reference=reference.transpose(1,2).reshape(-1,channels.shape[1],stop-start)
    torch.testing.assert_close(raw.double(),reference,rtol=2e-6,atol=2e-6)
    magnitude,choice=reference.abs().max(0)
    signed=((choice+1)*reference.gather(0,choice[None]).squeeze(0).sign()).long()
    torch.testing.assert_close(maxima.double(),magnitude,rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(indices,signed,rtol=0,atol=0)


def test_fused_ties_zeros_and_geometry_fallback(backend):
    torch,native=backend
    B=torch.zeros((10,6,19),device='cuda')
    B[:,0,0]=-2
    B[:,1,0]=2
    channels=torch.arange(10,device='cuda')[:,None].expand(-1,3)
    weights=torch.ones((5,10,3),device='cuda')
    maxima,indices=native.template_scores(B,channels,weights,0,19)
    assert torch.all(indices[:,0]==-1)
    assert torch.all(indices[:,1:]==0)
    assert torch.all(maxima[:,0]==20)
    # Unsupported but valid geometry uses the exact existing arithmetic.
    A=torch.einsum('ijk,jklm->iklm',weights[:3,:7],B[channels[:7]])
    A=A.transpose(1,2).reshape(-1,3,19)
    expected=native.abs_max_signed(A)
    actual=native.template_scores(B,channels[:7],weights[:3,:7],0,19)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('strided',[False,True])
def test_signed_max_ties_zeros_nan_and_infinity(backend,strided):
    torch,native = backend
    array = np.random.default_rng(3).normal(size=(7,5,74)).astype('float32')
    array[:,0,0] = 0
    array[0,1,0],array[1,1,0] = -100,100
    array[0,2,0],array[1,2,0] = np.nan,np.nan
    array[3,3,0] = -np.inf
    values = torch.as_tensor(array,device='cuda')
    if strided:
        values = values[:,:,::2]
    maxima,indices = torch.max(values.abs(),dim=0)
    chosen = values.gather(0,indices[None]).squeeze(0)
    signed = ((indices+1)*chosen.sign()).to(torch.int64)
    actual_max,actual_indices = native.abs_max_signed(values)
    torch.testing.assert_close(actual_max,maxima,rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(actual_indices,signed,rtol=0,atol=0)


def test_nondefault_stream_and_adapter_restoration(backend):
    torch,native = backend
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = torch.ones((3,5,31),device='cuda')
        values[2] = -2
        maxima,indices = native.abs_max_signed(values)
        output = native.neighbor_max(maxima,torch.arange(5,device='cuda')[None])
        assert torch.all(output==2)
        assert torch.all(indices==-3)
    from kilosort import spikedetect
    from terasort.kilosort_native import native_kilosort_reductions
    original = spikedetect.template_match
    with pytest.raises(RuntimeError,match='test interruption'):
        with native_kilosort_reductions():
            assert spikedetect.template_match is not original
            raise RuntimeError('test interruption')
    assert spikedetect.template_match is original
