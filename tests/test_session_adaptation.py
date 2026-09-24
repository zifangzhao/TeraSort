import h5py
import numpy as np

from terasort.session_adaptation import TemplateEvidence
from terasort.session_models import LocalModels


def model():
    w = np.zeros((1, 61, 1), np.float32)
    w[0, 27:34, 0] = [-1, -3, -8, -10, -8, -3, -1]
    return LocalModels(w, [[0]], [0], [0])


def populate(evidence, waveform, start=0):
    for i in range(24):
        evidence.observe(0, start+i, start+(i//8)*8, waveform)


def test_promotes_consistent_change_and_requires_fresh_evidence():
    models = model()
    evidence = TemplateEvidence(models, 1)
    populate(evidence, models.waveforms[0]*1.2)
    audit = evidence.promote(models, 24)
    assert audit[0]['promoted']
    assert models.version == 1
    assert not evidence.promote(models, 24)[0]['promoted']


def test_later_evidence_rejects_contaminated_training():
    models = model()
    original = models.waveforms.copy()
    evidence = TemplateEvidence(models, 1)
    for i in range(24):
        evidence.observe(0, i, (i//8)*8,
                         models.waveforms[0]*(1.5 if i < 16 else 1.))
    assert evidence.promote(models, 24)[0]['reason'] == 'validation_rejected'
    np.testing.assert_array_equal(models.waveforms, original)


def test_budget_expiration_and_checkpoint_determinism(tmp_path):
    models = model()
    evidence = TemplateEvidence(models, 1, budget=20, horizon_seconds=30)
    populate(evidence, models.waveforms[0]*1.2)
    assert len(evidence.rows[0]) == 20
    path = tmp_path / 'checkpoint.h5'
    with h5py.File(path, 'w') as handle:
        evidence.save(handle, [])
    restored = TemplateEvidence(models, 1, budget=20, horizon_seconds=30)
    with h5py.File(path) as handle:
        restored.restore(handle)
    other = model()
    assert evidence.promote(models, 24) == restored.promote(other, 24)
    np.testing.assert_array_equal(models.waveforms, other.waveforms)
    restored.expire(55)
    assert not restored.rows


def test_one_core_cannot_validate_itself():
    models = model()
    evidence = TemplateEvidence(models, 1)
    for i in range(24):
        evidence.observe(0, i, 0, models.waveforms[0]*1.2)
    assert not evidence.promote(models, 24)[0]['promoted']
