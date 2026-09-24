"""Bounded two-round CUDA learning experiment; evaluation is a separate command.

Explicit nonoverlapping two-second windows are preprocessed once to local
scratch. Each round reassigns them with the current bank. No GT is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import psutil

from terasort.session_manifest import load_session
from terasort.session_models import LocalModels
from terasort.session_refinement import StratifiedEvidence, refine_templates
from terasort.session_signal import iter_cores, preprocess, assess_quality
from terasort.session_sort import _collect_updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--seed-templates', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--training-seconds', type=float, nargs='+', required=True)
    parser.add_argument('--validation-seconds', type=float, nargs='+', required=True)
    parser.add_argument('--rounds', type=int, default=2, choices=(1, 2, 3))
    parser.add_argument('--scratch-gb', type=float, default=1.25)
    args = parser.parse_args()
    session = load_session(args.manifest)
    if len(session.probes) != 1:
        parser.error('Pilot requires one probe per invocation')
    probe = session.probes[0]
    rate = probe.sample_rate_hz
    groups = {name: [round(t*rate) for t in values] for name, values in
              [('training', args.training_seconds), ('validation', args.validation_seconds)]}
    starts = sorted(groups['training'] + groups['validation'])
    length, halo = round(2*rate), round(.1*rate)
    if (not starts or len(starts) > 32 or min(starts) < 0
            or any(b-a < length+2*halo for a, b in zip(starts, starts[1:]))):
        parser.error('Use at most 32 nonoverlapping windows, including preprocessing halos')
    for start in starts:
        if not any(s.start_sample <= start and start+length <= s.stop_sample
                   for s in probe.segments):
            parser.error('Each complete window must lie inside one source segment')
        if any(g.start_sample < start+length and g.stop_sample > start for g in probe.gaps):
            parser.error('Learning windows must not intersect declared gaps')
    bound = len(starts)*(length+2*halo)*probe.n_channels*4 + len(starts)*65536
    if not np.isfinite(args.scratch_gb) or bound > args.scratch_gb*1024**3:
        parser.error('Declared windows exceed scratch budget')
    args.output_root.mkdir(parents=True, exist_ok=False)
    cache = args.output_root/'cache'
    cache.mkdir()
    started = time.perf_counter()
    import cupy as cp
    from terasort.session_gpu import CudaResidualMatcher
    cp.get_default_memory_pool().set_limit(size=12*1024**3)
    models = LocalModels.load(args.seed_templates, probe)
    models.canonicalize()
    report = dict(algorithm='fixed_identity_refinement_v1', groups=groups,
                  seed=str(args.seed_templates.resolve()), raw_bytes_read=0,
                  seed_sha256=hashlib.sha256(args.seed_templates.read_bytes()).hexdigest(),
                  source=session.as_source_record(),
                  settings=dict(floor_snr=4.5, score_floor=.65, min_margin=.03,
                                residual_passes=3, overlap_policy='strict', blend=.25,
                                evidence_budget_per_partition=16384, per_window=8),
                  preprocessing='terasort-session-v1', rounds=[],
                  ground_truth_used=False, scratch_bound_bytes=bound)

    def save_bank(index):
        with (args.output_root/f'round_{index}.npz').open('xb') as f:
            np.savez_compressed(f, waveforms=models.waveforms, channels=models.channels,
                                anchors=models.anchors, assigned=models.assigned,
                                version=models.version)

    save_bank(0)
    for start in starts:
        core = next(iter_cores(probe, start_sample=start, stop_sample=start+length,
                               core_seconds=2., halo_samples=halo))
        if core.core_stop != start+length:
            raise ValueError('Incomplete learning core')
        signal = preprocess(core, probe)
        quality = assess_quality(core, signal, probe)
        signal[:, ~quality.usable_channels] = 0
        noise = np.where(quality.usable_channels, quality.noise_uv, np.inf).astype(np.float32)
        np.savez(cache/f'{start}.npz', signal=signal, noise=noise,
                 usable=quality.usable_channels, bad=quality.interval_bad,
                 data_start=core.data_start)
        report['raw_bytes_read'] += core.source_bytes_read
        del core, signal
    report['cache_bytes'] = sum(p.stat().st_size for p in cache.iterdir())
    for index in range(1, args.rounds+1):
        begin = time.perf_counter()
        evidence = {name: StratifiedEvidence(len(models.waveforms), windows,
                    budget=16384) for name, windows in groups.items()}
        matcher = CudaResidualMatcher(models)
        for name, windows in groups.items():
            for start in windows:
                with np.load(cache/f'{start}.npz') as data:
                    quality = SimpleNamespace(interval_bad=bool(data['bad']),
                                              usable_channels=data['usable'])
                    if quality.interval_bad:
                        continue
                    signal, noise = data['signal'], data['noise']
                    data_start = int(data['data_start'])
                    _, matches = matcher.match(signal, noise, 4.5,
                        core_start=start-data_start, core_stop=start+length-data_start,
                        score_floor=.65, min_margin=.03, max_passes=3,
                        refractory_samples=max(2, round(rate*.0005)), overlap_policy='strict')
                    _collect_updates(models, matches, signal, quality, evidence[name],
                                     data_start=data_start, core_start=start)
                    del signal, matches
                print(json.dumps(dict(stage='window_complete', round=index,
                                      partition=name, start_sample=start)), flush=True)
        audit = refine_templates(models, evidence['training'], evidence['validation'])
        (args.output_root/f'round_{index}_audit.json').write_text(json.dumps(audit, indent=2))
        save_bank(index)
        summary = dict(round=index, promoted=sum(r['promoted'] for r in audit),
                       wall_seconds=time.perf_counter()-begin,
                       evidence_rows={n: sum(len(v) for v in e.rows.values())
                                      for n, e in evidence.items()})
        report['rounds'].append(summary)
        print(json.dumps(summary), flush=True)
    report.update(wall_seconds=time.perf_counter()-started,
                  process_memory=psutil.Process().memory_info()._asdict(),
                  gpu_pool_bytes=cp.get_default_memory_pool().total_bytes())
    (args.output_root/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
