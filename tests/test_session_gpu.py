"""Execute the session CUDA kernels against independent CPU calculations."""

import numpy as np
import pytest
import json
import h5py
from unittest.mock import patch

from terasort.session_models import LocalModels, _score_one


@pytest.fixture
def gpu():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device required")
    from terasort.session_gpu import CudaResidualMatcher
    return cp, CudaResidualMatcher


def model_fixture():
    x = np.arange(61) - 30
    pulse = -np.exp(-(x / 2.7) ** 2).astype(np.float32)
    templates = np.zeros((2, 61, 2), np.float32)
    templates[0, :, 0] = 15 * pulse
    templates[0, :, 1] = 4 * pulse
    templates[1, :, 0] = 12 * pulse
    templates[1, :, 1] = 3 * pulse
    return LocalModels(templates, np.array([[0, 1], [1, 0]], np.int32),
                       np.array([0, 1], np.int32), np.zeros(2, np.int64))


def test_cuda_scores_match_cpu_for_partial_warp_batch(gpu):
    cp, matcher_type = gpu
    models = model_fixture()
    voltage = np.random.default_rng(47).normal(0, .01, (400, 2)).astype(np.float32)
    for center in (80, 180, 280):
        voltage[center-30:center+31] += models.waveforms[0]
    matcher = matcher_type(models)
    # Six pairs exercise the final block's inactive warps.
    choices = matcher._score(cp.asarray(voltage),
                            [(t, 0, 15.) for t in (80, 180, 280)])
    for t, choice in zip((80, 180, 280), choices):
        score, amplitude, gain = _score_one(voltage, t, models.waveforms[0],
                                            models.channels[0])
        assert choice[2] == 0
        np.testing.assert_allclose([choice[0], choice[3], choice[1]],
                                   [score, amplitude, gain], rtol=2e-5, atol=2e-5)


def test_cuda_residual_recovers_overlapping_contact_support(gpu):
    cp, matcher_type = gpu
    models = model_fixture()
    voltage = np.zeros((400, 2), np.float32)
    voltage[70:131] += models.waveforms[0]
    voltage[85:146, ::-1] += models.waveforms[1]
    candidates, matches = matcher_type(models).match(
        voltage, np.ones(2, np.float32), 4.5,
        core_start=0, core_stop=400, score_floor=.65)
    assert {(m.source_sample, m.unit) for m in matches} == {(100, 0), (115, 1)}
    assert any(m.pass_index > 0 for m in matches)
    assert len(candidates) >= 2


@pytest.mark.parametrize('overlap_policy', ['strict','interference'])
@pytest.mark.parametrize('rescue_floor_snr,rescue_passes,shift_radius,detector_mode',
                         [(None,1,2,'raw'),(3.5,1,2,'raw'),(3.5,3,4,'raw'),(3.5,1,2,'smooth3')])
@pytest.mark.parametrize('refit_rounds', [0,1])
def test_cuda_checkpoint_resume_preserves_exact_events(gpu, tmp_path, overlap_policy, rescue_floor_snr, rescue_passes, shift_radius, detector_mode, refit_rounds):
    from terasort.session_manifest import load_session
    from terasort.session_signal import iter_cores, preprocess
    from terasort.session_sort import run_session

    raw = np.random.default_rng(18).normal(0, 3, (3000, 1)).astype(np.int16)
    pulse = -np.exp(-((np.arange(61)-30)/2.7)**2)
    for center in (200, 1200, 2200):
        raw[center-30:center+31, 0] += np.rint(400*pulse).astype(np.int16)
    source = tmp_path / "raw.bin"
    raw.tofile(source)
    manifest = tmp_path / "manifest.json"
    definition = {"schema_version": 1, "session_id": "cuda-resume", "probes": [{
        "probe_id": "probeA", "sample_rate_hz": 20000., "gain_uv_per_count": .2,
        "geometry": {"x_um": [0], "y_um": [0], "shank": [0]},
        "segments": [{"path": str(source), "start_sample": 0,
                      "n_samples": 3000, "day_id": "day1"}], "gaps": []}]}
    manifest.write_text(json.dumps(definition))
    probe = load_session(manifest).probes[0]
    core = next(iter_cores(probe, core_seconds=.025))
    seed = tmp_path / "seed.npy"
    np.save(seed, preprocess(core, probe)[170:231][None])
    definition["probes"][0].update(seed_templates=str(seed),
                                   seed_preprocessing_id="terasort-session-v1")
    manifest.write_text(json.dumps(definition))
    settings = dict(backend="cuda", core_seconds=.025, shard_seconds=.05,
                    overlap_policy=overlap_policy, rescue_floor_snr=rescue_floor_snr,
                    rescue_passes=rescue_passes, shift_radius=shift_radius)
    settings['refit_rounds'] = refit_rounds
    settings['detector_mode'] = detector_mode
    interrupted, fresh = tmp_path / "interrupted", tmp_path / "fresh"
    calls = 0
    def crash(core, probe):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected interruption")
        return preprocess(core, probe)
    with patch("terasort.session_sort.preprocess", side_effect=crash):
        with pytest.raises(RuntimeError, match="injected interruption"):
            run_session(manifest, interrupted, **settings)
    run_session(manifest, interrupted, resume=True, **settings)
    run_session(manifest, fresh, **settings)
    total_spikes = 0
    for left in sorted((fresh / "probeA").glob("*.h5")):
        with h5py.File(left) as a, h5py.File(interrupted / "probeA" / left.name) as b:
            for key in ("candidates", "spikes", "model_after/waveforms",
                        "adaptation/last_promotion"):
                np.testing.assert_array_equal(a[key][:], b[key][:])
            total_spikes += len(a["spikes"])
    assert total_spikes == 3


def test_full_waveform_rejects_center_only_false_fit(gpu):
    cp, matcher_type = gpu
    waveform = np.ones((1, 61, 1), np.float32) * 10
    models = LocalModels(waveform, [[0]], [0], [0])
    signal = np.zeros((180, 1), np.float32)
    signal[60:121] = -10
    signal[82:99] = 10
    # Its central cosine is exactly one, but subtraction would increase
    # full-window error. Both implementations must reject that proposal.
    score, amplitude, gain = _score_one(signal, 90, waveform[0], np.array([0]))
    assert amplitude < 0 and gain == 0
    assert matcher_type(models)._score(cp.asarray(signal), [(90, 0, 10.)]) == [None]


def test_shift_fit_preserves_detection_provenance(gpu):
    cp, matcher_type = gpu
    models = model_fixture()
    voltage = np.zeros((250, 2), np.float32)
    voltage[70:131] = models.waveforms[0]
    choice = matcher_type(models)._score(cp.asarray(voltage), [(102, 0, 15.)])[0]
    assert choice[2] == 0 and choice[5] == -2
    np.testing.assert_allclose(choice[3], 1., atol=1e-5)


def test_seed_canonicalization_preserves_physical_contacts():
    waveform = np.zeros((1, 61, 3), np.float32)
    waveform[0, 38, 1] = -20
    waveform[0, 36, 0] = -5
    models = LocalModels(waveform, [[4, 7, -1]], [4], [0])
    assert models.canonicalize() == 1
    assert models.anchors.tolist() == [7]
    assert models.channels.tolist() == [[7, 4, -1]]
    assert models.waveforms[0, 30, 0] == -20
    assert models.waveforms[0, 28, 1] == -5


def test_ineligible_best_cosine_does_not_hide_valid_template(gpu):
    cp, matcher_type = gpu
    pulse = -np.exp(-((np.arange(61)-30)/2.7)**2).astype(np.float32)
    waveforms = np.zeros((2, 61, 1), np.float32)
    waveforms[0, :, 0] = pulse  # Perfect cosine, but required amplitude is 10.
    waveforms[1, :, 0] = 10*pulse
    waveforms[1, 33, 0] += .1
    models = LocalModels(waveforms, [[0], [0]], [0, 0], [0, 0])
    voltage = np.zeros((250, 1), np.float32)
    voltage[70:131, 0] = 10*pulse
    choice = matcher_type(models)._score(cp.asarray(voltage), [(100, 0, 10.)])[0]
    assert choice[2] == 1
    assert .99 < choice[3] < 1.01


def test_fused_detection_matches_reference_with_ties_masks_and_edges(gpu):
    from terasort.candidates.detectors import numpy_detect
    cp, matcher_type = gpu
    signal = np.random.default_rng(72).integers(-10, 11, (107, 7)).astype(np.float32)
    signal[0, 0] = 30
    signal[-1, 1] = -30
    signal[30:34, 2] = 30
    noise = np.array([1, 2, 3, np.inf, .7, 4, 1.1], np.float32)
    q = np.abs(signal)/noise
    expected = numpy_detect(q, floor=4.5)
    matcher = matcher_type(model_fixture())
    events = matcher.detect(cp.asarray(signal), noise, 4.5)
    actual = np.array([int(t)*7+int(c) for t,c,s in events])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose([s for t,c,s in events], q.ravel()[expected], rtol=1e-6)
    with pytest.raises(OverflowError):
        matcher.detect(cp.asarray(signal), noise, 4.5, max_candidates=1)


def test_extra_passes_recover_a_four_spike_overlap_chain(gpu):
    cp, matcher_type = gpu
    pulse = (-20*np.exp(-((np.arange(61)-30)/2.)**2)).astype(np.float32)
    models = LocalModels(pulse[None, :, None], [[0]], [0], [0])
    signal = np.zeros((300, 1), np.float32, order='F')
    expected = {100, 112, 124, 136}
    for t in expected:
        signal[t-30:t+31, 0] += pulse
    matcher = matcher_type(models)
    _, first = matcher.match(signal, np.ones(1, np.float32), 4.5,
                             core_start=0, core_stop=300, max_passes=3)
    _, extra = matcher.match(signal, np.ones(1, np.float32), 4.5,
                             core_start=0, core_stop=300, max_passes=6)
    assert len(first) == 3
    assert {m.source_sample for m in extra} == expected


def test_diagnostic_trace_does_not_change_assignments(gpu):
    cp, matcher_type = gpu
    models = model_fixture()
    signal = np.zeros((400, 2), np.float32)
    signal[70:131] += models.waveforms[0]
    signal[85:146, ::-1] += models.waveforms[1]
    matcher = matcher_type(models)
    kwargs = dict(core_start=0, core_stop=400)
    expected_events, expected_matches = matcher.match(signal, np.ones(2, np.float32), 4.5, **kwargs)
    records = []
    def trace(stage, **data):
        records.append((stage, data['pass_index']))
        if stage == 'scores':
            assert len(data['events']) == len(data['choices'])
    events, matches = matcher.match(signal, np.ones(2, np.float32), 4.5, trace=trace, **kwargs)
    assert events == expected_events and matches == expected_matches
    assert any(stage == 'accepted' for stage, _ in records)
    assert any(stage == 'overlap_deferred' for stage, _ in records)


def test_weakly_interacting_fits_share_pass_and_repeat_exactly(gpu):
    cp, matcher_type = gpu
    pulse = -np.exp(-((np.arange(61)-30)/2.)**2).astype(np.float32)
    w = np.zeros((2,61,2), np.float32)
    w[:, :, 0] = 20*pulse
    w[:, :, 1] = .1*pulse
    models = LocalModels(w, [[0,1],[1,0]], [0,1], [0,0])
    signal = np.zeros((400,2), np.float32)
    signal[70:131] += w[0]
    signal[82:143, ::-1] += w[1]
    matcher = matcher_type(models)
    kwargs = dict(core_start=0, core_stop=400)
    _, strict = matcher.match(signal, np.ones(2, np.float32), 4.5, max_passes=1, **kwargs)
    assert len(strict) == 1
    _, actual = matcher.match(signal, np.ones(2, np.float32), 4.5,
                              overlap_policy='interference', **kwargs)
    assert {(m.source_sample,m.unit,m.pass_index) for m in actual} == {(100,0,0),(112,1,0)}
    for _ in range(3):
        _, repeat = matcher.match(signal, np.ones(2, np.float32), 4.5,
                                  overlap_policy='interference', **kwargs)
        assert repeat == actual


@pytest.mark.parametrize('with_strong', [False, True])
def test_rescue_recovers_subthreshold_waveform_without_changing_primary(gpu, with_strong):
    cp, matcher_type = gpu
    pulse = -np.exp(-((np.arange(61)-30)/2.)**2).astype(np.float32)
    models = LocalModels((12*pulse)[None,:,None], [[0]], [0], [0])
    signal = np.zeros((400,1), np.float32)
    if with_strong:
        signal[70:131,0] += 12*pulse
    signal[200:261,0] += 4*pulse
    matcher = matcher_type(models)
    kwargs = dict(core_start=0, core_stop=400)
    _, before = matcher.match(signal, np.ones(1,np.float32), 4.5, **kwargs)
    _, after = matcher.match(signal, np.ones(1,np.float32), 4.5,
                             rescue_floor_snr=3.5, **kwargs)
    assert [m for m in after if m.pass_index < 3] == before
    assert [(m.source_sample,m.pass_index) for m in after if m.pass_index == 3] == [(230,3)]
    _, repeat = matcher.match(signal, np.ones(1,np.float32), 4.5,
                              rescue_floor_snr=3.5, **kwargs)
    assert after == repeat


def test_rescue_retains_primary_residual_gain_requirement(gpu):
    cp, matcher_type = gpu
    w = np.zeros((1,61,1), np.float32)
    w[0,30,0] = -10
    matcher = matcher_type(LocalModels(w, [[0]], [0], [0]))
    signal = np.zeros((200,1), np.float32)
    signal[100,0] = -4  # perfect shape but gain=16 < 4.5**2
    events, matches = matcher.match(signal, np.ones(1,np.float32), 4.5,
        rescue_floor_snr=3.5, core_start=0, core_stop=200)
    assert events and not matches
    with pytest.raises(ValueError, match='Rescue threshold'):
        matcher.match(signal, np.ones(1,np.float32), 4.5,
                      rescue_floor_snr=5, core_start=0, core_stop=200)


def test_more_rescue_passes_recover_a_weak_overlap_chain(gpu):
    cp, matcher_type = gpu
    pulse = -np.exp(-((np.arange(61)-30)/2.)**2).astype(np.float32)
    models = LocalModels((12*pulse)[None,:,None], [[0]], [0], [0])
    signal = np.zeros((300,1), np.float32)
    for t in (100,112,124):
        signal[t-30:t+31,0] += 4*pulse
    matcher = matcher_type(models)
    kwargs = dict(core_start=0, core_stop=300, rescue_floor_snr=3.5)
    _, one = matcher.match(signal, np.ones(1,np.float32), 4.5, **kwargs)
    _, three = matcher.match(signal, np.ones(1,np.float32), 4.5, rescue_passes=3, **kwargs)
    assert len(one) == 1
    assert {m.source_sample for m in three} == {100,112,124}
    with pytest.raises(ValueError, match='Timing radius'):
        matcher.match(signal, np.ones(1,np.float32), 4.5, shift_radius=9, **kwargs)


def test_wider_timing_search_can_recover_an_offset_proposal(gpu):
    cp, matcher_type = gpu
    pulse = -10*np.exp(-((np.arange(61)-30)/.7)**2).astype(np.float32)
    model = LocalModels(pulse[None,:,None], [[0]], [0], [0])
    signal = np.zeros((200,1), np.float32)
    signal[70:131,0] = pulse
    matcher = matcher_type(model)
    assert matcher._score(cp.asarray(signal), [(104,0,10.)], shift_radius=2)[0] is None
    choice = matcher._score(cp.asarray(signal), [(104,0,10.)], shift_radius=4)[0]
    assert choice[2] == 0 and choice[5] == -4 and choice[0] > .99


def test_smoothed_detector_matches_cpu_filter_and_mad(gpu):
    from terasort.candidates.detectors import numpy_detect
    cp,matcher_type=gpu
    signal=np.random.default_rng(71).normal(0,1,(401,2)).astype(np.float32)
    signal[200,0]=-10
    filtered=signal.copy()
    filtered[1:-1]=.25*(signal[:-2]+2*signal[1:-1]+signal[2:])
    noise=np.median(np.abs(filtered-np.median(filtered,axis=0)),axis=0)/.67448975
    noise[1]=np.inf
    expected=numpy_detect(np.abs(filtered)/noise,floor=4.5)
    matcher=matcher_type(model_fixture())
    events=matcher.detect(cp.asarray(signal),[1,np.inf],4.5,mode='smooth3')
    np.testing.assert_array_equal([int(t)*2+int(c) for t,c,s in events],expected)
    np.testing.assert_allclose([s for t,c,s in events],(np.abs(filtered)/noise).ravel()[expected],rtol=1e-6)


@pytest.mark.parametrize("nt,nc", [(1, 1), (2, 33), (7, 31), (9, 33), (401, 37)])
def test_smooth3_shared_detector_matches_cpu_for_tile_edges(gpu, nt, nc):
    from terasort.candidates.detectors import numpy_detect

    cp, matcher_type = gpu
    signal = np.random.default_rng(nt * 100 + nc).normal(
        0, 1, (nt, nc)).astype(np.float32)
    signal[0, 0] = -8
    signal[nt // 2, 0] = -11
    signal[-1, 0] = -7
    raw_noise = np.ones(nc, np.float32)
    if nc > 1:
        raw_noise[-1] = np.inf

    filtered = signal.copy()
    if nt > 2:
        filtered[1:-1] = .25 * (
            signal[:-2] + 2 * signal[1:-1] + signal[2:])
    step = max(1, nt // 4000)
    sample = filtered[::step]
    estimate = np.median(
        np.abs(sample - np.median(sample, axis=0)), axis=0) / .67448975
    noise = np.where(np.isfinite(raw_noise), np.maximum(estimate, .01), np.inf)
    expected = numpy_detect(np.abs(filtered) / noise, floor=2.5)

    events = matcher_type(model_fixture()).detect(
        cp.asarray(signal), raw_noise, 2.5, mode="smooth3")
    indices = np.asarray([int(t) * nc + int(c) for t, c, _ in events])
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(
        [score for _, _, score in events],
        (np.abs(filtered) / noise).ravel()[expected],
        rtol=1e-6, atol=1e-6)


def test_smooth3_match_reuses_core_noise_for_residual_passes(gpu, monkeypatch):
    cp, matcher_type = gpu
    models = model_fixture()
    signal = np.zeros((2000, 2), np.float32)
    signal[200:261] += models.waveforms[0]
    signal[1000:1061] += models.waveforms[0]
    matcher = matcher_type(models)
    estimate_calls = []
    detect_noise = []
    estimate = matcher._estimate_smooth3_noise
    detect = matcher.detect

    def tracked_estimate(residual, noise_uv):
        estimate_calls.append(1)
        return estimate(residual, noise_uv)

    def tracked_detect(residual, noise_uv, floor_snr, **kwargs):
        if kwargs.get("mode") == "smooth3":
            detect_noise.append(kwargs.get("smooth3_noise_uv"))
        return detect(residual, noise_uv, floor_snr, **kwargs)

    monkeypatch.setattr(matcher, "_estimate_smooth3_noise", tracked_estimate)
    monkeypatch.setattr(matcher, "detect", tracked_detect)
    candidates, matches = matcher.match(
        signal, np.ones(2, np.float32), 4.5,
        core_start=0, core_stop=len(signal), detector_mode="smooth3",
        max_passes=3, score_floor=.65)
    assert len(estimate_calls) == 1
    assert len(detect_noise) >= 2
    assert all(noise is detect_noise[0] for noise in detect_noise)
    assert len(candidates) > 0 and len(matches) >= 2
