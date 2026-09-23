import json
import numpy as np
import pytest

from terasort.candidates.detectors import numpy_detect, neighbor_table
from terasort.candidates.bank import BankWriter, CandidateBank, events_from_indices


@pytest.fixture(scope='module')
def gpu():
    cp = pytest.importorskip('cupy')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from terasort.candidates.detectors import CudaDetector
    return cp, torch, CudaDetector()


@pytest.mark.parametrize('radius', [1, 5, 12])
@pytest.mark.parametrize('spatial', [False, True])
def test_gpu_matches_independent_reference_with_ties(gpu, radius, spatial):
    cp, torch, detector = gpu
    from terasort.candidates.detectors import torch_detect
    q = np.abs(np.random.default_rng(15).normal(size=(233, 7))).astype('float32')
    q[60:64, 2] = 5  # plateau, earliest temporal tie
    q[130, 0:2] = 7  # equal spatial peaks retained
    q[160, 3] = 3  # strict threshold excludes equality
    q[2, 0] = 5
    ns = neighbor_table(np.c_[np.zeros(7), np.arange(7)*20], 30) if spatial else None
    expected = numpy_detect(q, radius, 3., ns, 1, len(q)-1)
    actual = np.sort(detector.detect(cp.asarray(q), radius, 3., ns, 1, len(q)-1).get())
    other = torch_detect(torch.as_tensor(q, device='cuda'), radius, 3., ns, 1, len(q)-1).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(other, expected)


def test_empty_dense_and_overflow(gpu):
    cp, _, detector = gpu
    assert len(detector.detect(cp.zeros((50, 3), cp.float32))) == 0
    q = cp.zeros((100, 4), cp.float32)
    q[::3] = 6
    with pytest.raises(OverflowError):
        detector.detect(q, capacity=2)
    np.testing.assert_array_equal(np.sort(detector.detect(q, capacity=200).get()), numpy_detect(q.get()))


@pytest.mark.parametrize('spatial', [False, True])
def test_chunk_boundaries_equal_whole_recording(gpu, spatial):
    cp, _, detector = gpu
    q = np.abs(np.random.default_rng(9).normal(size=(900, 6))).astype('float32')
    q[199, 1] = 5
    q[200, 2] = 6
    q[400, 3] = 6
    r = 5 if spatial else 1
    ns = neighbor_table(np.c_[np.zeros(6), np.arange(6)*20], 30) if spatial else None
    expected = numpy_detect(q, r, 3., ns, 1, len(q)-1)
    pieces = []
    for start in range(0, len(q), 200):
        stop = min(start+200, len(q))
        lo, hi = max(0, start-3*r), min(len(q), stop+3*r)
        idx = detector.detect(cp.asarray(q[lo:hi]), r, 3., ns,
                              max(start, 1)-lo, min(stop, len(q)-1)-lo).get()
        pieces.append(idx + lo*q.shape[1])
    np.testing.assert_array_equal(np.sort(np.concatenate(pieces)), expected)


def test_bank_rethreshold_equals_detection_and_preserves_int64(tmp_path):
    rng = np.random.default_rng(10)
    voltage = (rng.normal(size=(700, 5))*3).astype(np.float32)
    noise = np.array([1, 2, 3, 4, 5], np.float32)
    scores = np.abs(voltage)/noise
    offset = 2**33
    path = tmp_path/'example.scb.h5'
    with BankWriter(path, {'floor_snr': 2., 'sample_rate_hz': 30000}, noise) as bank:
        idx = numpy_detect(scores, floor=2.)
        events = events_from_indices(idx, voltage, noise, offset)
        bank.append(events[events['sample_index'] < offset+350], offset, offset+350)
        bank.append(events[events['sample_index'] >= offset+350], offset+350, offset+700)
    bank = CandidateBank(path)
    for threshold in [2, 2.5, 3, 4, 6, 20]:
        expected = events_from_indices(numpy_detect(scores, floor=threshold), voltage, noise, offset)
        np.testing.assert_array_equal(bank.query(threshold), expected)
    chosen = bank.query(3, channels=[0, 2], start_sample=offset+100, stop_sample=offset+400, polarity='neg')
    assert np.all(np.isin(chosen['channel_index'], [0, 2]))
    assert np.all(chosen['amplitude_uv'] < 0)
    assert np.all((chosen['sample_index'] >= offset+100) & (chosen['sample_index'] < offset+400))
    with pytest.raises(ValueError):
        bank.query(1.5)
    with pytest.raises(FileExistsError):
        BankWriter(path, {'floor_snr': 2}, noise)


def test_failed_write_never_publishes_complete_bank(tmp_path):
    path = tmp_path/'broken.scb.h5'
    with pytest.raises(ValueError):
        with BankWriter(path, {'floor_snr': 2}, np.ones(2)) as bank:
            bank.append([], 0, 100)
            bank.append([], 99, 200)
    assert not path.exists()
    with pytest.raises(ValueError):
        CandidateBank(path.with_suffix('.h5.partial'))


def test_bank_keeps_neighboring_events_for_later_spatial_decisions(tmp_path):
    voltage = np.zeros((100, 2), np.float32)
    voltage[50, 0], voltage[51, 1] = -6, -4
    noise = np.ones(2, np.float32)
    events = events_from_indices(numpy_detect(abs(voltage)), voltage, noise)
    assert len(events) == 2
    path = tmp_path/'neighbors.scb.h5'
    with BankWriter(path, {'floor_snr': 3}, noise) as bank:
        bank.append(events, 0, 100)
    assert len(CandidateBank(path).query(3)) == 2
    assert len(CandidateBank(path).query(5)) == 1


def test_reader_rejects_corrupt_index(tmp_path):
    import h5py
    path = tmp_path/'corrupt.scb.h5'
    with BankWriter(path, {'floor_snr': 3}, np.ones(2)) as bank:
        bank.append([], 0, 100)
    with h5py.File(path, 'r+') as f:
        row = f['blocks'][0]
        row['row_stop'] = 20
        f['blocks'][0] = row
    with pytest.raises(ValueError, match='index'):
        CandidateBank(path)


def test_empty_bank_query_and_invalid_time_or_channel(tmp_path):
    path = tmp_path/'empty.scb.h5'
    with BankWriter(path, {'floor_snr': 3}, np.ones(2)) as bank:
        bank.append([], 0, 100)
    reader = CandidateBank(path)
    assert len(reader.query(3)) == 0
    with pytest.raises(ValueError):
        reader.query(3, start_sample=100, stop_sample=0)
    with pytest.raises(ValueError):
        reader.query(3, channels=[2])


def test_writer_rejects_duplicate_or_miscalibrated_events(tmp_path):
    x = np.zeros((100, 2), np.float32)
    x[30, 1] = -5
    events = events_from_indices(numpy_detect(abs(x)), x, np.ones(2))
    with BankWriter(tmp_path/'checked.scb.h5', {'floor_snr':3}, np.ones(2)) as bank:
        with pytest.raises(ValueError, match='unique'):
            bank.append(np.concatenate([events, events]), 0, 100)
        bad = events.copy()
        bad['amplitude_uv'] = -6
        with pytest.raises(ValueError, match='calibration'):
            bank.append(bad, 0, 100)
        bank.append(events, 0, 100)
