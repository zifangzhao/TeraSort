"""Predeclared follow-up families after the finite settings screen completes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--kilosort-dir',type=Path,required=True)
    p.add_argument('--seed-templates',type=Path,required=True)
    p.add_argument('--pca-templates',type=Path,required=True)
    args=p.parse_args()
    if not (args.screen_root/'complete.json').exists():raise ValueError('Finish the screen first')
    args.output_root.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(line) for line in (args.screen_root/'results.jsonl').read_text().splitlines()]
    screening=[r for r in rows if r['phase']=='screen' and not r['exit_code']]
    best=max(screening,key=lambda r:(r['f1'],r['recovered_units_iou_0p8']))
    configs=json.loads((args.screen_root/'plan.json').read_text())['configurations']
    selected=next(c for c in configs if c['name']==best['name'])
    # The screen command records the exact seed path, including transformed banks.
    seed=Path(best['command'][best['command'].index('--seed-templates')+1])
    plan=[dict(name='best_f1_'+best['name'],seed=str(seed),options=selected['options'],starts=[420])]
    for rounds in (1,2):
        plan.append(dict(name='local_refit_'+str(rounds),seed=str(args.seed_templates),
                         options={'--refit-rounds':rounds},starts=[60]))
    for score in (.8,.85):
        plan.append(dict(name='score_'+str(score),seed=str(args.seed_templates),
                         options={'--score-floor':score},starts=[60]))
    plan.append(dict(name='smooth3',seed=str(args.seed_templates),options={'--detector-mode':'smooth3'},starts=[60]))
    with np.load(args.pca_templates) as data,np.load(args.seed_templates) as original:
        changed=(data['waveforms'].shape!=original['waveforms'].shape or
                 not np.array_equal(data['waveforms'],original['waveforms']))
    if changed:
        plan.append(dict(name='pca_split',seed=str(args.pca_templates),options={},starts=[60]))
    (args.output_root/'plan.json').write_text(json.dumps(dict(configurations=plan,pca_changed=changed,
        selection='Best F1 on development only; then validate the best new family only if F1 improves over reference',
        validation_interval=[420,450]),indent=2))
    outputs=[]

    def run(config,start):
        name=config['name']+'_'+str(start)
        output=args.output_root/name
        options={'--rescue-floor-snr':3.5,**config['options']}
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('benchmark_session_recording.py')),
             '--dataset-root',str(args.dataset_root),'--kilosort-dir',str(args.kilosort_dir),
             '--output-root',str(output),'--seed-templates',config['seed'],
             '--start-seconds',str(start),'--seconds','30','--shard-seconds','10','--freeze-templates']
        for k,v in options.items():
            if v is not None:cmd.extend([k,str(v)])
        begin=time.perf_counter()
        with (args.output_root/(name+'.log')).open('x') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        row=dict(name=config['name'],start=start,exit_code=proc.returncode,
                 end_to_end_seconds=time.perf_counter()-begin,command=cmd)
        if not proc.returncode:
            q=json.loads((output/'quality.json').read_text())['session'];q.pop('unit_matches')
            row.update(q);pr,re=q['spike_precision'],q['spike_recall'];row['f1']=2*pr*re/max(pr+re,1e-12)
        outputs.append(row)
        with (args.output_root/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
    for config in plan:
        for start in config['starts']:run(config,start)
    new=[r for r in outputs if r['start']==60 and not r['exit_code']]
    reference=next(r for r in screening if r['name']=='reference')
    if new:
        winner=max(new,key=lambda r:(r['f1'],r['recovered_units_iou_0p8']))
        if winner['f1']>reference['f1']:
            run(next(c for c in plan if c['name']==winner['name']),420)
    (args.output_root/'complete.json').write_text(json.dumps(dict(runs=len(outputs)),indent=2))


if __name__=='__main__':main()
