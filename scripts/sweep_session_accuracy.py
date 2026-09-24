"""Sequential bounded accuracy screen, followed by a separate-interval check.

Ground truth ranks experimental configurations only; it never enters template
transforms or fitting. This finite screen is not an exhaustive algorithm search.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from terasort.session_models import LocalModels


def transformed_bank(source, target, *, rank=None, fraction=None):
    with np.load(source) as data:
        model = LocalModels(data['waveforms'], data['channels'], data['anchors'],
                            data['assigned'], int(data['version']))
    for unit in range(len(model.waveforms)):
        if rank is not None:
            u, s, vt = np.linalg.svd(model.waveforms[unit], full_matrices=False)
            model.waveforms[unit] = (u[:, :rank]*s[:rank]) @ vt[:rank]
        if fraction is not None:
            energy = np.sum(model.waveforms[unit]**2, axis=0, dtype=np.float64)
            energy[model.channels[unit] < 0] = 0
            order = np.argsort(-energy, kind='stable')
            count = np.searchsorted(np.cumsum(energy[order]), fraction*sum(energy))+1
            keep = np.zeros(len(energy), bool)
            keep[order[:count]] = True
            keep[0] = True
            model.channels[unit, ~keep] = -1
            model.waveforms[unit, :, ~keep] = 0
    model.canonicalize()
    with target.open('xb') as handle:
        np.savez_compressed(handle, waveforms=model.waveforms, channels=model.channels,
                            anchors=model.anchors, assigned=model.assigned, version=model.version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--kilosort-dir', type=Path, required=True)
    parser.add_argument('--seed-templates', type=Path, required=True)
    parser.add_argument('--refined-templates', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    banks = {'original': args.seed_templates, 'refined': args.refined_templates}
    for name, spec in [('rank1', dict(rank=1)), ('rank2', dict(rank=2)),
                       ('mask90', dict(fraction=.90)), ('mask97', dict(fraction=.97)),
                       ('rank2_mask97', dict(rank=2, fraction=.97))]:
        banks[name] = args.output_root/f'{name}.npz'
        transformed_bank(args.seed_templates, banks[name], **spec)
    configs = [dict(name='reference', bank='original', options={})]
    for flag, values in [('--score-floor', [.55,.6,.7,.75]),
                         ('--min-margin', [0,.01,.07,.15]),
                         ('--half-width', [4,12,20,30]),
                         ('--rescue-floor-snr', [3.,4.])]:
        for value in values:
            configs.append(dict(name=flag[2:].replace('-','_')+'_'+str(value),
                                bank='original', options={flag:value}))
    configs += [dict(name='no_rescue', bank='original', options={'--rescue-floor-snr':None}),
                dict(name='interference', bank='original', options={'--overlap-policy':'interference'})]
    configs += [dict(name=name, bank=name, options={}) for name in banks if name != 'original']
    configs += [dict(name=name+'_score07', bank=name, options={'--score-floor':.7})
                for name in ('rank2','mask97')]
    configs += [dict(name='short_score07', bank='original', options={'--half-width':4,'--score-floor':.7}),
                dict(name='wide_score06', bank='original', options={'--half-width':12,'--score-floor':.6})]
    plan = dict(configurations=configs, development=[60,90], validation=[420,450],
                seed_sha256=hashlib.sha256(args.seed_templates.read_bytes()).hexdigest(),
                selection='Within 0.2 percentage points of reference precision; no recall loss; rank recovered units then F1',
                profile=False, note='No evaluation labels used to transform seed banks')
    (args.output_root/'plan.json').write_text(json.dumps(plan, indent=2))
    reports = []

    def run(config, phase, start):
        output = args.output_root/f'{phase}_{config["name"]}'
        options = {'--rescue-floor-snr':3.5, **config['options']}
        command = [sys.executable, '-u', str(Path(__file__).with_name('benchmark_session_recording.py')),
                   '--dataset-root', str(args.dataset_root), '--kilosort-dir', str(args.kilosort_dir),
                   '--output-root', str(output), '--start-seconds', str(start), '--seconds','30',
                   '--shard-seconds','10', '--freeze-templates', '--seed-templates',str(banks[config['bank']])]
        for key, value in options.items():
            if value is not None:
                command += [key,str(value)]
        started = time.perf_counter()
        with (args.output_root/f'{phase}_{config["name"]}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row = dict(name=config['name'], phase=phase, exit_code=result.returncode,
                   command=command, end_to_end_seconds=time.perf_counter()-started)
        if not result.returncode:
            quality = json.loads((output/'quality.json').read_text())['session']
            quality.pop('unit_matches')
            row.update(quality)
            p, r = quality['spike_precision'], quality['spike_recall']
            row['f1'] = 2*p*r/max(p+r,1e-12)
            row['resources'] = json.loads((output/'run_report.json').read_text())
        reports.append(row)
        with (args.output_root/'results.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('resources','command')}), flush=True)
        return row

    for config in configs:
        run(config,'screen',60)
    baseline = reports[0]
    if baseline['exit_code']:
        raise RuntimeError('Reference failed; no valid comparison')
    eligible = [r for r in reports if not r['exit_code'] and
                r['spike_precision'] >= baseline['spike_precision']-.002 and
                r['spike_recall'] >= baseline['spike_recall']]
    eligible.sort(key=lambda r:(r['recovered_units_iou_0p8'],r['f1']), reverse=True)
    selected = [r['name'] for r in eligible if r['name'] != 'reference'][:2]
    (args.output_root/'selection.json').write_text(json.dumps(dict(selected=selected,
                    eligible=[r['name'] for r in eligible]), indent=2))
    for name in ['reference']+selected:
        run(next(c for c in configs if c['name']==name), 'validation',420)
    (args.output_root/'complete.json').write_text(json.dumps(dict(runs=len(reports),
        failed=sum(bool(r['exit_code']) for r in reports), selected=selected),indent=2))


if __name__ == '__main__':
    main()
