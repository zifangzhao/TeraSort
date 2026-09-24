"""Trace missed GT spikes through our unchanged detector and CUDA matcher.

Ground truth and evaluation-derived unit links select diagnostic events only.
They never alter detection, templates or assignment. No Kilosort comparison
results or new Kilosort run are used; the existing seed bank is retained.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cupy as cp
import h5py
import numpy as np

from terasort.session_manifest import load_session
from terasort.session_models import LocalModels, _score_one
from terasort.session_signal import iter_cores, preprocess, assess_quality
from terasort.session_gpu import CudaResidualMatcher


def missed_times(gt, predicted, tolerance):
    """Same chronological one-to-one event matching as the evaluator."""
    i = j = 0
    missed = []
    while i < len(gt):
        if j == len(predicted):
            missed.extend(map(int, gt[i:])); break
        delta = int(gt[i])-int(predicted[j])
        if abs(delta) <= tolerance:
            i += 1; j += 1
        elif delta < 0:
            missed.append(int(gt[i])); i += 1
        else:
            j += 1
    return missed


class FailureTrace:
    def __init__(self, targets, models, core, signal, noise, tolerance):
        self.targets, self.models, self.core = targets, models, core
        self.noise, self.tolerance = noise, tolerance
        self.records = {}
        for target in targets:
            unit, sample = target['unit'], target['sample']
            center = sample-core.data_start
            contacts = models.channels[unit]
            valid = contacts >= 0
            q = np.abs(signal[max(0,center-tolerance):center+tolerance+1,
                              contacts[valid]]) / noise[contacts[valid]]
            target.update(max_local_snr=float(np.max(q)), candidates=[],
                          peak_channel=int(contacts[np.argmax(np.max(q, axis=0))]))

    def __call__(self, stage, **data):
        if stage == 'scores':
            events, choices = data['events'], data['choices']
            times = np.array([e[0] for e in events], np.int64)
            # One bounded diagnostic transfer per residual pass; never retained
            # after the callback. Normal production calls do not allocate this.
            voltage = cp.asnumpy(data['residual'])
            for target in self.targets:
                unit = target['unit']
                center = target['sample']-self.core.data_start
                contacts = set(map(int, self.models.channels[unit])) - {-1}
                lo, hi = np.searchsorted(times, [center-self.tolerance-2,
                                                center+self.tolerance+3])
                for i in range(lo, hi):
                    t, c, snr = events[i]
                    if c not in contacts:
                        continue
                    options = []
                    raw = []
                    for shift in range(-2,3):
                        if abs(int(t)+shift-center) > self.tolerance:
                            continue
                        score, amp, gain = _score_one(voltage, int(t)+shift,
                            self.models.waveforms[unit], self.models.channels[unit],
                            noise_uv=self.noise)
                        raw.append((score,amp,gain,shift))
                        if score >= .65 and .3 <= amp <= 3:
                            options.append((gain,score,amp,shift))
                    if not raw:
                        continue
                    best_raw = max(raw, key=lambda row:row[0])
                    choice = choices[i]
                    record = dict(pass_index=data['pass_index'],
                        candidate_sample=self.core.data_start+int(t), channel=int(c),
                        candidate_snr=float(snr), expected_center_score=best_raw[0],
                        expected_amplitude=best_raw[1], expected_gain=best_raw[2],
                        expected_eligible=bool(options), winner=None)
                    if choice is None:
                        record['decision'] = 'no_eligible_template'
                    else:
                        score,gain,winner,amp,runner,shift,margin = choice
                        record.update(winner=winner, winner_score=score, winner_gain=gain,
                                      winner_amplitude=amp, margin=margin)
                        record['decision'] = ('gain_rejected' if gain < 4.5**2 else
                            'ambiguity_rejected' if margin < .03 else 'proposal')
                    if options:
                        gain,score,amp,shift = max(options)
                        record.update(expected_gain=gain, expected_center_score=score,
                                      expected_amplitude=amp, expected_shift=shift)
                    target['candidates'].append(record)
                    self.records.setdefault((data['pass_index'],int(t),int(c)), []).append(record)
        else:
            for record in self.records.get((data['pass_index'],data['candidate'],data['channel']), []):
                record['decision'] = stage
                record['fitted_sample'] = self.core.data_start+data['fitted_time']
                record['inside_output_core'] = self.core.core_start <= record['fitted_sample'] < self.core.core_stop
                if 'blocker_unit' in data:
                    record.update(blocker_unit=data['blocker_unit'],
                                  blocker_sample=self.core.data_start+data['blocker_time'])
                if 'blocker_contact' in data:
                    contact = data['blocker_contact']
                    fractions = []
                    for unit in (data['unit'], data['blocker_unit']):
                        w = self.models.waveforms[unit]
                        slot = np.flatnonzero(self.models.channels[unit] == contact)[0]
                        fractions.append(float(np.sum(w[:,slot]**2)/max(np.sum(w**2),1e-12)))
                    record.update(blocker_contact=contact, conflict_contact_energy_fractions=fractions)

    def finish(self):
        for target in self.targets:
            rows = target['candidates']
            expected = [r for r in rows if r['winner'] == target['unit']]
            accepted = [r for r in expected if r['decision'] == 'accepted' and
                        abs(r['fitted_sample']-target['sample']) <= self.tolerance]
            if accepted:
                reason = ('accepted_nearby_despite_one_to_one_miss' if any(r.get('inside_output_core',True) for r in accepted)
                          else 'fitted_peak_outside_evaluated_core')
            elif any(r['decision'] == 'overlap_deferred' for r in expected):
                reason = 'overlap_deferred_without_recovery'
            elif any(r['decision'] == 'refractory' for r in expected):
                reason = 'refractory_suppressed'
            elif any(r['decision'] == 'ambiguity_rejected' for r in expected):
                reason = 'expected_winner_but_ambiguous'
            elif any(r['decision'] == 'gain_rejected' for r in expected):
                reason = 'expected_winner_low_gain'
            elif any(r['expected_eligible'] for r in rows):
                reason = 'expected_eligible_other_template_wins'
            elif rows:
                reason = ('expected_template_amplitude_rejected'
                          if any(r['expected_center_score'] >= .65 for r in rows)
                          else 'expected_template_shape_rejected')
            else:
                reason = ('no_candidate_below_threshold_or_masked' if target['max_local_snr'] <= 4.5
                          else 'no_candidate_in_mapped_patch_time_window')
            target['classification'] = reason
            target['trace_semantics'] = 'furthest_observed_path_across_local_candidates_and_passes'
        return self.targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--focus-sample', type=int, help='Trace only the selected miss at this source sample')
    args = parser.parse_args()
    source = args.run_root
    session = load_session(source / 'manifest.json')
    probe = session.probes[0]
    run = json.loads((source / 'session/run.json').read_text())['config']
    if (len(session.probes) != 1 or not run['freeze_templates'] or run['residual_passes'] != 3
            or run['novelty'] != 'off' or run['score_floor'] != .65 or run['min_margin'] != .03
            or run['floor_snr'] != 4.5):
        parser.error('Trace currently supports the frozen default three-pass single-probe diagnostic')
    if run.get('half_width', 8) != 8:
        parser.error('Trace expected-template diagnostic currently requires half_width=8')
    if run.get('refit_rounds', 0):
        parser.error('This trace does not replay the experimental local-refit post-pass')
    if run.get('detector_mode','raw')!='raw':
        parser.error('This trace currently assumes raw-noise candidate SNR')
    quality = json.loads((source / 'quality.json').read_text())
    tolerance = quality['tolerance_samples']
    first, stop = run['start_sample'], run['stop_sample']
    if stop is None or stop-first > probe.sample_rate_hz*120:
        parser.error('Select a diagnostic run spanning at most 120 seconds')
    with np.load(args.ground_truth, allow_pickle=False) as gt:
        times, labels = gt['times'], gt['labels']
        if float(gt['sampling_frequency']) != probe.sample_rate_hz:
            raise ValueError('Ground truth sample rate differs')
        keep = (times >= first) & (times < stop)
        times, labels = times[keep], labels[keep]
    shards = sorted((source / 'session' / probe.probe_id).glob('*.h5'))
    spike_rows = []
    for path in shards:
        with h5py.File(path) as handle:
            spike_rows.append(handle['spikes'][:])
    spikes = np.concatenate(spike_rows)
    models = LocalModels.load(source / 'session' / probe.probe_id / 'calibration/day1.npz', probe)
    selected, missed_total = [], 0
    for link in quality['session']['unit_matches']:
        unit = int(link['predicted_unit'].split('/')[-1])
        gtunit = link['ground_truth_unit']
        actual = np.sort(times[labels.astype(str) == gtunit])
        predicted = np.sort(spikes['sample_index'][spikes['unit_id'] == unit])
        missed = missed_times(actual, predicted, tolerance)
        missed_total += len(missed)
        for i in np.linspace(0, len(missed)-1, min(4,len(missed)), dtype=int):
            selected.append(dict(ground_truth_unit=gtunit, unit=unit, sample=missed[i],
                                 mapping_iou=link['iou'], unit_missed_spikes=len(missed)))
    if args.focus_sample is not None:
        selected = [r for r in selected if r['sample'] == args.focus_sample]
        if not selected:
            parser.error('Focus sample is not among the deterministic selected misses')
    args.output_root.mkdir(parents=True, exist_ok=False)
    matcher = CudaResidualMatcher(models)
    results = []
    checked_spikes = 0
    for path in shards:
        with h5py.File(path) as handle:
            a,b = int(handle.attrs['start_sample']), int(handle.attrs['stop_sample'])
        for core in iter_cores(probe, core_seconds=run['core_seconds'],
                halo_samples=max(31,round(probe.sample_rate_hz*run['halo_ms']/1000)),
                start_sample=a, stop_sample=b):
            targets = [dict(row) for row in selected if core.core_start <= row['sample'] < core.core_stop]
            if not targets:
                continue
            voltage = preprocess(core, probe)
            qc = assess_quality(core, voltage, probe)
            signal = voltage.copy()
            signal[:, ~qc.usable_channels] = 0
            noise = np.where(qc.usable_channels, qc.noise_uv, np.inf).astype(np.float32)
            tracer = FailureTrace(targets, models, core, signal, noise, tolerance)
            events, matches = matcher.match(signal, noise, 4.5,
                core_start=core.core_start-core.data_start, core_stop=core.core_stop-core.data_start,
                refractory_samples=max(2,round(probe.sample_rate_hz*.0005)), trace=tracer,
                overlap_policy=run.get('overlap_policy','strict'),
                rescue_floor_snr=run.get('rescue_floor_snr'),
                rescue_passes=run.get('rescue_passes', 1),
                shift_radius=run.get('shift_radius', 2))
            expected = spikes[(spikes['sample_index'] >= core.core_start) &
                              (spikes['sample_index'] < core.core_stop)]
            actual = [(core.data_start+m.source_sample,m.channel,m.unit,m.pass_index) for m in matches]
            reference = [(int(r['sample_index']),int(r['channel_index']),int(r['unit_id']),int(r['pass_index']))
                         for r in expected]
            assert actual == reference, 'Tracing changed assignments or configuration differs'
            checked_spikes += len(actual)
            results.extend(tracer.finish())
            print(json.dumps(dict(core_stop=core.core_stop, traced=len(results))), flush=True)
    credible = [r for r in results if r['mapping_iou'] >= .5]
    report = dict(diagnostic_only=True, ground_truth_used_only_for_error_selection=True,
        additional_kilosort_outputs_used=False, existing_seed_bank_retained=True,
        selected_misses=len(results), total_missed_spikes=missed_total,
        checked_replayed_spikes=checked_spikes, exact_replay=True,
        all_mapping_counts=dict(Counter(r['classification'] for r in results)),
        mapping_iou_at_least_half_counts=dict(Counter(r['classification'] for r in credible)),
        credible_mapping_selected_misses=len(credible),
        sampling='up to four evenly spaced missed spikes per evaluation-linked unit; not prevalence weighted',
        caveats=['Ground-truth unit links come from evaluation and may be wrong, particularly at low IoU.',
                 'Local candidate proximity does not prove it was generated by that neuron.',
                 'Categories summarize the furthest observed path, not independent causal interventions.'])
    with (args.output_root / 'events.jsonl').open('x') as handle:
        for row in results:
            handle.write(json.dumps(row)+'\n')
    (args.output_root / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
