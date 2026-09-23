import h5py
import numpy as np
import pytest

from terasort.candidates.bank import BankWriter, CandidateBank, EVENT_DTYPE
from terasort.candidates.cache import cache_waveforms
from terasort.candidates.waveforms import (
    WaveformSpec, extract_waveforms, geometry_channel_map, grouped_channel_map)


def fixture_bank(tmp_path, nc=5, nt=30):
    offset = 2**33
    # Unique values reveal wrong channel order, shifted windows, and sign loss.
    voltage = (np.arange(nt)[:, None]*.371 + np.arange(nc)[None, :]*3.17+4).astype(np.float32)
    voltage[:, 1::2] *= -1
    t = np.array([0, 1, 14, 15, 16, nt-1])
    c = np.array([0, 1, 0, 1, nc-1, nc-1])
    events = np.empty(len(t), EVENT_DTYPE)
    events['sample_index'], events['channel_index'] = t+offset, c
    events['amplitude_uv'], events['snr'] = voltage[t, c], np.abs(voltage[t, c])
    path = tmp_path/'ledger.scb.h5'
    with BankWriter(path, {'floor_snr': 3, 'sample_rate_hz': 30000}, np.ones(nc)) as writer:
        writer.append(events[:3], offset, offset+15)
        writer.append(events[3:], offset+15, offset+nt)
    return path, voltage, events, offset


@pytest.mark.parametrize('dtype,scale', [('int16', .05), ('float32', 1.)])
def test_cached_waveforms_match_direct_source_at_edges_and_chunk_boundaries(tmp_path, dtype, scale):
    path, voltage, events, origin = fixture_bank(tmp_path)
    spec = WaveformSpec(grouped_channel_map([0, 0, 0, 1, 1]), 2, 3, scale, dtype)
    output = tmp_path/'cached.scb.h5'
    cached = cache_waveforms(path, output, spec,
                            [(origin, origin+15, origin, voltage[:18]),
                             (origin+15, origin+30, origin+13, voltage[13:])],
                            {'preprocessing': 'fixture voltage'})
    assert CandidateBank(path).waveform_spec is None
    np.testing.assert_array_equal(cached.query(3), events)
    rows = []
    for batch in cached.iter_waveforms(3, batch_size=2):
        assert len(batch.events) <= 2
        rows.extend(batch.row_indices)
        for j, event in enumerate(batch.events):
            t, c = int(event['sample_index']-origin), int(event['channel_index'])
            for k, ch in enumerate(batch.channel_indices[j]):
                for s, source_t in enumerate(range(t-2, t+4)):
                    actual = batch.waveforms_uv[j, s, k]
                    if ch < 0 or source_t < 0 or source_t >= len(voltage):
                        assert np.isnan(actual)
                    else:
                        assert abs(actual-voltage[source_t, ch]) <= (.025002 if dtype == 'int16' else 0)
            assert batch.channel_indices[j, 0] == c
            assert abs(batch.waveforms_uv[j, spec.pre_samples, 0]-event['amplitude_uv']) <= .025002
    np.testing.assert_array_equal(rows, np.arange(len(events)))
    selected = list(cached.iter_waveforms(5, channels=[1], polarity='neg', start_sample=origin+2))
    assert sum(len(b.events) for b in selected) == 1
    assert selected[0].row_indices[0] == 3
    with pytest.raises(FileExistsError):
        cache_waveforms(path, output, spec, [], {})


def test_geometry_respects_probe_groups_radius_and_cap():
    pos = [[0, 0], [0, 20], [20, 0], [0, 0], [0, 20], [0, 80]]
    channels = geometry_channel_map(pos, [0, 0, 0, 1, 1, 1], 25, 3)
    np.testing.assert_array_equal(channels, [[0, 1, 2], [1, 0, -1], [2, 0, -1],
                                           [3, 4, -1], [4, 3, -1], [5, -1, -1]])
    assert geometry_channel_map(pos, [0, 0, 0, 1, 1, 1], 100, 1).shape == (6, 1)
    with pytest.raises(ValueError):
        geometry_channel_map([[0, np.nan]], [0], 25, 3)


@pytest.mark.parametrize('nc', [384, 768])
def test_large_probe_neighborhood_extracts_only_local_synchronous_channels(nc):
    groups = np.arange(nc)//192
    within = np.arange(nc) % 192
    positions = np.c_[(within % 2)*20, (within//2)*20]
    channels = geometry_channel_map(positions, groups, 55, 12)
    spec = WaveformSpec(channels, 3, 5, 1, 'float32')
    voltage = (np.arange(20)[:, None]*1000 + np.arange(nc)[None, :]).astype(np.float32)
    events = np.empty(4, EVENT_DTYPE)
    events['sample_index'] = [4, 5, 6, 7]
    events['channel_index'] = [0, 191, 192, nc-1]
    values, starts, stops = extract_waveforms(events, voltage, 0, spec, 0, len(voltage))
    assert values.shape == (4, 9, 12)
    for i, (t, c) in enumerate(zip(events['sample_index'], events['channel_index'])):
        for slot, neighbor in enumerate(channels[c]):
            if neighbor < 0:
                np.testing.assert_array_equal(values[i, :, slot], 0)
            else:
                assert groups[neighbor] == groups[c]
                np.testing.assert_array_equal(values[i, :, slot], voltage[int(t)-3:int(t)+6, neighbor])


def test_source_block_omission_cannot_publish_a_short_cache(tmp_path):
    path, voltage, _, origin = fixture_bank(tmp_path)
    spec = WaveformSpec(grouped_channel_map([0]*5), 2, 3)
    output = tmp_path/'incomplete.scb.h5'
    with pytest.raises(ValueError, match='every bank block'):
        cache_waveforms(path, output, spec, [(origin, origin+15, origin, voltage)], {})
    assert not output.exists()


def test_missing_halo_and_misalignment_never_publish(tmp_path):
    path, voltage, events, origin = fixture_bank(tmp_path)
    spec = WaveformSpec(grouped_channel_map([0]*5), 2, 3)
    with pytest.raises(ValueError, match='halo'):
        cache_waveforms(path, tmp_path/'short.scb.h5', spec,
                        [(origin, origin+15, origin, voltage[:15])], {})
    assert not (tmp_path/'short.scb.h5').exists()
    with pytest.raises(ValueError, match='Incomplete'):
        CandidateBank(tmp_path/'short.scb.h5.partial')
    wrong = voltage.copy()
    wrong[0, 0] += 1
    with pytest.raises(ValueError, match='anchor voltage'):
        cache_waveforms(path, tmp_path/'wrong.scb.h5', spec,
                        [(origin, origin+15, origin, wrong)], {})
    assert not (tmp_path/'wrong.scb.h5').exists()


def test_int16_overflow_and_invalid_padding_rejected_before_append(tmp_path):
    spec = WaveformSpec(np.array([[0, -1], [1, -1]]), 1, 1, .01)
    events = np.array([(5, 0, 5., 5.)], EVENT_DTYPE)
    values = np.zeros((1, 3, 2), np.float32)
    values[0, 1, 0] = 5
    with BankWriter(tmp_path/'valid.scb.h5', {'floor_snr': 3}, np.ones(2), waveform_spec=spec) as writer:
        bad = values.copy()
        bad[0, 0, 0] = 9999
        with pytest.raises(OverflowError):
            writer.append(events, 0, 10, bad, [0], [3])
        bad = values.copy()
        bad[0, 0, 1] = 1
        with pytest.raises(ValueError, match='store zero'):
            writer.append(events, 0, 10, bad, [0], [3])
        writer.append(events, 0, 10, values, [0], [3])
    assert CandidateBank(tmp_path/'valid.scb.h5').n_events == 1


@pytest.mark.parametrize('badmap', [np.array([[1, 0], [1, 0]]), np.array([[0, 0], [1, 0]]),
                                    np.array([[0, -1, 1], [1, 0, -1]]), np.array([[0, 2], [1, 0]])])
def test_invalid_channel_maps_rejected(badmap):
    with pytest.raises(ValueError):
        WaveformSpec(badmap, 1, 2)


def test_reader_rejects_corrupt_waveform_row_count(tmp_path):
    path, voltage, _, origin = fixture_bank(tmp_path)
    spec = WaveformSpec(grouped_channel_map([0]*5), 2, 3)
    output = tmp_path/'cached.scb.h5'
    cache_waveforms(path, output, spec, [(origin, origin+15, origin, voltage),
                                       (origin+15, origin+30, origin, voltage)], {})
    with h5py.File(output, 'r+') as f:
        f['waveforms/data'].resize((5, spec.n_samples, spec.width))
    with pytest.raises(ValueError, match='row alignment'):
        CandidateBank(output)


def test_empty_cache_and_legacy_error(tmp_path):
    spec = WaveformSpec(np.array([[0]]), 1, 1)
    p = tmp_path/'empty.scb.h5'
    with BankWriter(p, {'floor_snr': 3}, np.ones(1), waveform_spec=spec) as writer:
        writer.append([], 0, 100, np.empty((0, 3, 1)), np.array([], np.int32), np.array([], np.int32))
    assert list(CandidateBank(p).iter_waveforms(3)) == []
    with pytest.raises(ValueError):
        list(CandidateBank(p).iter_waveforms(3, batch_size=0))
    path, _, _, _ = fixture_bank(tmp_path)
    with pytest.raises(ValueError, match='no waveform'):
        list(CandidateBank(path).iter_waveforms(3))
