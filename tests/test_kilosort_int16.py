import numpy as np
import pytest


def test_grouped_reader_opens_only_overlapping_files(tmp_path, monkeypatch):
    from kilosort.io import BinaryRWFile
    from terasort.kilosort_int16 import source_slice
    files = [tmp_path / f"part{i}.bin" for i in range(5)]
    for i, path in enumerate(files):
        np.full((10, 2), i, dtype=np.int16).tofile(path)
    reader = BinaryRWFile(files, 2, device="cpu")
    group = reader.file
    opened = []
    original = group.get_file

    def tracked(index):
        opened.append(index)
        return original(index)

    monkeypatch.setattr(group, "get_file", tracked)
    np.testing.assert_array_equal(source_slice(reader, 28, 32),
                                  np.array([[2, 2], [2, 2], [3, 3], [3, 3]], dtype=np.int16))
    assert opened == [2, 3]


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


def test_multifile_reader_matches_reference_across_boundary(backend, tmp_path):
    torch, _ = backend
    from kilosort.io import BinaryRWFile
    from terasort.kilosort_int16 import native_int16_reader
    counts = np.random.default_rng(37).integers(-1000, 1000, (421, 3), dtype=np.int16)
    files = [tmp_path / "part1.bin", tmp_path / "part2.bin"]
    counts[:195].tofile(files[0])
    counts[195:].tofile(files[1])
    kwargs = dict(n_chan_bin=3, NT=128, nt=7, device=torch.device('cuda'))
    grouped = BinaryRWFile(files, **kwargs)
    reference = BinaryRWFile(files[0], file_object=counts, **kwargs)
    with native_int16_reader(files) as reader:
        for batch in range(grouped.n_batches):
            actual, ai = grouped.padded_batch_to_torch(batch, return_inds=True)
            expected, ei = reference.padded_batch_to_torch(batch, return_inds=True)
            assert ai == ei
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert reader.calls == grouped.n_batches
