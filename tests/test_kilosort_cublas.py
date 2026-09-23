"""Numerical, stride, cache and stream checks for native projection/filtering."""
import pytest


@pytest.fixture(scope='module')
def backend():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from terasort.kilosort_cublas import CublasProjection
    native = CublasProjection()
    yield torch, native
    native.close()


@pytest.mark.parametrize('shape', [(5, 7, 13), (1, 1, 1), (23, 67, 79)])
def test_native_gemm_with_float64_reference(backend, shape):
    torch, native = backend
    m, k, n = shape
    g = torch.Generator(device='cuda').manual_seed(234)
    W = torch.randn((m, k), device='cuda', generator=g)
    X = torch.randn((k, n), device='cuda', generator=g)
    actual = native.gemm(W, X)
    expected = W.double() @ X.double()
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-5)
    with pytest.raises(ValueError, match='contiguous'):
        native.gemm(W.double(), X.double())


@pytest.mark.parametrize('layout', ['channel', 'pca'])
@pytest.mark.parametrize('length', [1, 79, 517])
def test_temporal_padding_tail_and_projection(backend, layout, length):
    torch, native = backend
    g = torch.Generator(device='cuda').manual_seed(234)
    X = torch.randn((3, length), device='cuda', generator=g)
    W = torch.randn((6, 61), device='cuda', generator=g)
    U = torch.randn((7, 6, 3), device='cuda', generator=g)
    actual = native.convolve(X, W, layout)
    expected = torch.nn.functional.conv1d(X.double()[:, None], W.double()[:, None], padding=30)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-5)
    projected = native.project(U, actual, layout)
    reference = torch.einsum('upc,cpt->ut', U.double(), actual.double())
    torch.testing.assert_close(projected.double(), reference, rtol=2e-5, atol=2e-5)


def test_cache_mutation_and_nondefault_stream(backend):
    torch, native = backend
    U = torch.ones((7, 6, 3), device='cuda')
    packed = native.pack_weights(U)
    assert native.pack_weights(U) is packed
    U.add_(1)
    assert native.pack_weights(U) is not packed
    torch.testing.assert_close(native.pack_weights(U), torch.full((7, 18), 2., device='cuda'))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        X = torch.ones((3, 79), device='cuda')
        W = torch.zeros((6, 61), device='cuda')
        W[:, 30] = 1
        U = torch.ones((7, 6, 3), device='cuda')
        B = native.convolve(X, W, 'pca')
        actual = native.project(U, B, 'pca')
        torch.testing.assert_close(actual, torch.full((7, 79), 18., device='cuda'), rtol=0, atol=0)


def test_invalid_convolution_inputs(backend):
    torch, native = backend
    X, W = torch.ones((3, 79), device='cuda'), torch.ones((6, 61), device='cuda')
    for x, w in ((X[:, ::2], W), (X, W[:, :-2]), (X.double(), W)):
        with pytest.raises(ValueError, match='contiguous'):
            native.convolve(x, w)


def test_scoped_full_adapter_restores_on_exception(backend):
    from kilosort import spikedetect, template_matching
    from terasort.kilosort_cublas import cublas_kilosort
    originals = spikedetect.template_match, template_matching.run_matching
    with pytest.raises(RuntimeError, match='test interruption'):
        with cublas_kilosort() as details:
            assert spikedetect.template_match is not originals[0]
            assert template_matching.run_matching is not originals[1]
            assert details['runtime_stats']()['cublas_version'] > 120000
            raise RuntimeError('test interruption')
    assert (spikedetect.template_match, template_matching.run_matching) == originals
