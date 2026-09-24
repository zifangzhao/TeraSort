import h5py
import numpy as np
import pytest

from terasort.session_models import LocalModels
from terasort.session_novelty import NoveltyBank, similarity
from terasort.session_adaptation import TemplateEvidence


def fixtures():
    pulse = -np.exp(-((np.arange(61)-30)/2.5)**2).astype(np.float32)
    w = np.zeros((61, 2), np.float32)
    w[:, 0] = pulse*20
    model = LocalModels(w[None], [[0, 1]], [0], [0])
    return model, w, np.array([2, 3], np.int32)


def populate(bank, waveform, contacts):
    for i in range(36):
        bank.observe(i*100, (i//12)*1200, contacts, waveform)


def test_shadow_then_enroll_stable_new_id():
    model, waveform, contacts = fixtures()
    bank = NoveltyBank(20000)
    populate(bank, waveform, contacts)
    assert bank.evaluate(model, 3600)[0]['reason'] == 'eligible_shadow'
    assert len(model.waveforms) == 1
    audit = bank.evaluate(model, 3600, enroll=True)
    assert audit[0]['enrolled_unit'] == 1
    assert model.by_channel[2] == [1]
    assert model.version == 1 and not bank.proposals
    np.testing.assert_array_equal(model.waveforms[0], waveform)


def test_existing_unit_rejected_even_with_swapped_contact_slots():
    model, waveform, contacts = fixtures()
    bank = NoveltyBank(20000)
    populate(bank, waveform[:, ::-1], np.array([1, 0], np.int32))
    audit = bank.evaluate(model, 3600, enroll=True)
    assert audit[0]['existing_similarity'] > .99
    assert audit[0]['enrolled_unit'] is None


def test_checkpoint_replay_has_identical_enrollment(tmp_path):
    model, waveform, contacts = fixtures()
    bank = NoveltyBank(20000)
    populate(bank, waveform, contacts)
    path = tmp_path / 'state.h5'
    with h5py.File(path, 'w') as handle:
        bank.save(handle, [])
    restored = NoveltyBank(20000)
    with h5py.File(path) as handle:
        restored.restore(handle)
    other, _, _ = fixtures()
    assert bank.evaluate(model, 3600, enroll=True) == restored.evaluate(other, 3600, enroll=True)
    np.testing.assert_array_equal(model.waveforms, other.waveforms)


def test_proposal_bounds_and_expiry():
    model, waveform, contacts = fixtures()
    bank = NoveltyBank(20000, max_proposals=2)
    for c in range(10):
        populate(bank, waveform, np.array([2*c, 2*c+1], np.int32))
    assert len(bank.proposals) == 2
    bank.expire(20000*1801+3600)
    assert not bank.proposals


def test_model_growth_keeps_evidence_budget():
    model, waveform, contacts = fixtures()
    evidence = TemplateEvidence(model, 20000)
    bank = NoveltyBank(20000)
    populate(bank, waveform, contacts)
    bank.evaluate(model, 3600, enroll=True)
    evidence.extend_models(model)
    assert evidence.last_promotion.tolist() == [-1, -1]
    assert evidence.capacity * len(model.waveforms) <= 32768


def test_single_core_cannot_enroll():
    model, waveform, contacts = fixtures()
    bank = NoveltyBank(20000)
    for i in range(36):
        bank.observe(i*100, 0, contacts, waveform)
    assert bank.evaluate(model, 3600, enroll=True)[0]['reason'] == 'insufficient_temporal_support'


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_session_enrollment_and_resume_preserve_new_identity(tmp_path, backend):
    import json
    from unittest.mock import patch
    from terasort.session_manifest import load_session
    from terasort.session_signal import iter_cores, preprocess
    from terasort.session_sort import run_session
    if backend == 'cuda':
        cp = pytest.importorskip('cupy')
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CUDA device required')

    rate = 20000
    raw = np.random.default_rng(11).normal(0, 2, (rate*6, 8)).astype(np.int16)
    pulse = np.rint(-400*np.exp(-((np.arange(61)-30)/2.5)**2)).astype(np.int16)
    for t in range(500, len(raw)-100, 1000):
        raw[t-30:t+31, 6] += pulse
    raw[470:531, 0] += pulse
    source = tmp_path / 'source.bin'
    raw.tofile(source)
    manifest = tmp_path / 'manifest.json'
    definition = dict(schema_version=1, session_id='enrollment', probes=[dict(
        probe_id='p', sample_rate_hz=rate, gain_uv_per_count=.2,
        geometry=dict(x_um=[0]*8, y_um=list(range(0, 800, 100)), shank=[0]*8),
        segments=[dict(path=str(source), start_sample=0, n_samples=len(raw), day_id='d')], gaps=[])])
    manifest.write_text(json.dumps(definition))
    probe = load_session(manifest).probes[0]
    core = next(iter_cores(probe))
    seed = np.zeros((1, 61, 8), np.float32)
    seed[0, :, 0] = preprocess(core, probe)[470:531, 0]
    np.save(tmp_path / 'seed.npy', seed)
    definition['probes'][0].update(seed_templates=str(tmp_path / 'seed.npy'),
                                  seed_preprocessing_id='terasort-session-v1')
    manifest.write_text(json.dumps(definition))
    settings = dict(backend=backend, core_seconds=1., shard_seconds=3., novelty='enroll')
    fresh, resumed = tmp_path / 'fresh', tmp_path / 'resumed'
    report = run_session(manifest, fresh, **settings)
    assert report['enrolled_units'] >= 1
    calls = 0
    def crash(core, probe):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('after enrollment')
        return preprocess(core, probe)
    with patch('terasort.session_sort.preprocess', side_effect=crash):
        with pytest.raises(RuntimeError, match='after enrollment'):
            run_session(manifest, resumed, **settings)
    run_session(manifest, resumed, resume=True, **settings)
    paths = sorted((fresh / 'p').glob('*.h5'))
    with h5py.File(paths[-1]) as handle:
        assert np.count_nonzero(handle['spikes']['unit_id'][:] == 1) > 40
    for path in paths:
        with h5py.File(path) as a, h5py.File(resumed / 'p' / path.name) as b:
            for key in ('spikes', 'model_after/waveforms', 'adaptation/last_promotion'):
                np.testing.assert_array_equal(a[key][:], b[key][:])
