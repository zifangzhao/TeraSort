"""Tiling must retain suppression across seams, ties and global borders."""
import pytest


@pytest.fixture(scope='module')
def backend():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from terasort.kilosort_native import NativeReductions
    return torch, NativeReductions()


def inputs(torch, empty=False):
    # The retained reference always iterates 40 times; choose a length whose
    # final reference tile is nonempty (the production batch is 60,122).
    X = torch.zeros((12, 197), device='cuda')
    # Ties, signs, adjacent tile-edge competitors, border-excluded competitors,
    # and an event in the final partial tile; centers have separate channels.
    for c in range(7):
        for t, value in [(7, 20), (10, -5), (15, 8), (16, -8), (31, 6),
                         (33, -9), (79, 12), (81, -10), (186, 4), (189, 30)]:
            X[c, t] = 0 if empty else value + c * 0.125
    W = torch.zeros((3, 9), device='cuda')
    W[:, 4] = 1  # Identical templates exercise first-index ties.
    ops = dict(nt=9, settings=dict(nt0min=3, n_templates=3), wTEMP=W, Th_universal=2)
    channels = torch.arange(7, device='cuda')[None].expand(10, -1)
    weights = torch.zeros((5, 10, 7), device='cuda')
    weights[:, 0] = 1  # Identical spatial widths exercise ties too.
    neighbors = torch.arange(7, device='cuda')[None]
    return X, ops, channels, neighbors, weights


@pytest.mark.parametrize('tile', [1, 2, 16, 17, 64, 256])
@pytest.mark.parametrize('empty', [False, True])
def test_exact_seams_edges_ties_tail_and_empty(backend, tile, empty):
    torch, native = backend
    from terasort.kilosort_native import make_native_template_match
    from terasort.kilosort_tiled import make_tiled_template_match
    reference, _ = make_native_template_match(native, 'fused')
    tiled = make_tiled_template_match(native, tile)
    args = inputs(torch, empty)
    expected, actual = reference(*args), tiled(*args)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if not empty:
        times = actual[0][:, 1]
        # Border t=7 must not suppress t=10; t=33 must suppress t=31.
        assert 10 in times and 31 not in times and 33 in times
        assert 15 in times and 16 in times and 186 in times
        assert 7 not in times and 189 not in times


def test_stream_and_random_neighbor_suppression(backend):
    torch, native = backend
    from terasort.kilosort_native import make_native_template_match
    from terasort.kilosort_tiled import make_tiled_template_match
    reference, _ = make_native_template_match(native, 'fused')
    tiled = make_tiled_template_match(native, 17)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        X, ops, channels, _, weights = inputs(torch)
        generator = torch.Generator(device='cuda').manual_seed(71)
        X.copy_(torch.randn(X.shape, device='cuda', generator=generator))
        weights.copy_(torch.randn(weights.shape, device='cuda', generator=generator))
        neighbors = torch.randint(7, (4, 7), device='cuda', generator=generator)
        args = X, ops, channels, neighbors, weights
        for a, b in zip(tiled(*args), reference(*args)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_unsupported_geometry_preserves_fallback(backend):
    torch, native = backend
    from terasort.kilosort_native import make_native_template_match
    from terasort.kilosort_tiled import make_tiled_template_match
    reference, _ = make_native_template_match(native, 'fused')
    tiled = make_tiled_template_match(native, 17)
    X, ops, channels, neighbors, weights = inputs(torch)
    args = X, ops, channels[:7], neighbors, weights[:3, :7]
    for a, b in zip(tiled(*args), reference(*args)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('deep', [False, True])
def test_patch_restores_on_exception(backend, deep):
    from kilosort import spikedetect, template_matching
    from terasort.kilosort_tiled import tiled_kilosort
    originals = spikedetect.template_match, template_matching.run_matching
    with pytest.raises(RuntimeError, match='test interruption'):
        with tiled_kilosort(17, deep=deep) as details:
            assert spikedetect.template_match is not originals[0]
            assert (template_matching.run_matching is not originals[1]) == deep
            assert details['universal_tile_samples'] == 17
            raise RuntimeError('test interruption')
    assert (spikedetect.template_match, template_matching.run_matching) == originals


def test_invalid_tile(backend):
    _, native = backend
    from terasort.kilosort_tiled import make_tiled_template_match
    with pytest.raises(ValueError, match='positive'):
        make_tiled_template_match(native, 0)
