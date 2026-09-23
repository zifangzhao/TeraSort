import numpy as np
import pytest


@pytest.fixture(scope='module')
def backend():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from terasort.kilosort_int16 import Int16Reader
    reader = Int16Reader()
    yield torch, reader
    reader.close()


@pytest.mark.parametrize('channels,n_samples,crop,downsampling', [
    (3, 421, False, 1), (37, 421, False, 1), (32, 42, False, 1),
    (33, 421, True, 1), (37, 809, False, 2), (1, 421, False, 1)])
def test_reader_matches_decoded_reference_at_all_edges(backend, channels, n_samples, crop, downsampling):
    torch, backend = backend
    from kilosort.io import BinaryRWFile
    rng = np.random.default_rng(16)
    counts = rng.integers(-32768, 32768, (n_samples, channels), dtype=np.int16)
    counts[0] = -32768
    counts[-1] = 32767
    scale = np.linspace(.01, .5, channels, dtype=np.float32)
    offset = np.linspace(-1, 2, channels, dtype=np.float32)
    decoded = counts.astype(np.float32)*scale+offset
    kwargs = dict(filename='test.bin', n_chan_bin=channels, fs=1000, NT=128, nt=7,
                  device=torch.device('cuda'), batch_downsampling=downsampling)
    if crop:
        kwargs.update(tmin=.017, tmax=.387)
    raw = BinaryRWFile(**kwargs, file_object=counts, scale=scale[:, None], shift=offset[:, None])
    reference = BinaryRWFile(**kwargs, file_object=decoded)
    before = backend.buffers_created
    for batch in range(raw.n_batches):
        actual, ai = backend.read(raw, batch, return_inds=True)
        expected, ei = reference.padded_batch_to_torch(batch, return_inds=True)
        assert ai == ei
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert backend.buffers_created == before+1


def test_buffer_reuse_across_cuda_streams(backend):
    torch, backend = backend
    from kilosort.io import BinaryRWFile
    counts = np.arange(421*37, dtype=np.int16).reshape(421, 37)
    reader = BinaryRWFile('test.bin', 37, NT=128, nt=7, file_object=counts, device=torch.device('cuda'))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for batch in range(reader.n_batches):
        with torch.cuda.stream(streams[batch % 2]):
            results.append(backend.read(reader, batch))
    for stream in streams:
        stream.synchronize()
    for batch, actual in enumerate(results):
        torch.testing.assert_close(actual, reader.padded_batch_to_torch(batch), rtol=0, atol=0)


def test_context_scope_restoration_and_guards(backend, tmp_path):
    torch, _ = backend
    from kilosort.io import BinaryRWFile
    from terasort.kilosort_int16 import native_int16_reader
    path = tmp_path/'counts.bin'
    counts = np.ones((420, 3), dtype=np.int16)
    raw = BinaryRWFile(path, 3, NT=128, nt=7, file_object=counts, device=torch.device('cuda'))
    original = BinaryRWFile.padded_batch_to_torch
    with pytest.raises(RuntimeError, match='interrupt'):
        with native_int16_reader(path) as reader:
            assert raw.padded_batch_to_torch(0).dtype == torch.float32
            assert reader.calls == 1
            with pytest.raises(IndexError):
                raw.padded_batch_to_torch(-1)
            raw.scale = float('nan')
            with pytest.raises(ValueError, match='Calibration'):
                raw.padded_batch_to_torch(0)
            raise RuntimeError('interrupt')
    assert BinaryRWFile.padded_batch_to_torch is original
