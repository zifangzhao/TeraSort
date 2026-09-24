"""Two detector/acceptance combinations, with a fresh-interval matched control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('dataset-root','kilosort-dir','seed-templates','pca-templates','screen-root','output-root'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=False)
    (a.output_root/'plan.json').write_text(json.dumps(dict(
        development=dict(seconds=[60,90],smooth3_score_floors=[.7,.75]),pca_validation=[420,450],
        fresh_validation=[480,510],selection='Best development F1; validate only if it beats raw detection at the same score floor',
        fresh_controls=['raw score .65','raw at selected score','smooth3 at selected score']),indent=2))
    outputs=[]
    def run(name,start,score,mode,seed):
        out=a.output_root/name
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('benchmark_session_recording.py')),
             '--dataset-root',str(a.dataset_root),'--kilosort-dir',str(a.kilosort_dir),
             '--output-root',str(out),'--seed-templates',str(seed),'--start-seconds',str(start),
             '--seconds','30','--shard-seconds','10','--freeze-templates','--rescue-floor-snr','3.5',
             '--score-floor',str(score),'--detector-mode',mode]
        with (a.output_root/(name+'.log')).open('x') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        row=dict(name=name,start=start,score_floor=score,mode=mode,command=cmd,exit_code=proc.returncode)
        if not proc.returncode:
            q=json.loads((out/'quality.json').read_text())['session'];q.pop('unit_matches');row.update(q)
            pr,re=q['spike_precision'],q['spike_recall'];row['f1']=2*pr*re/max(pr+re,1e-12)
        outputs.append(row)
        with (a.output_root/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
        return row
    candidates=[run('smooth_score_'+str(score),60,score,'smooth3',a.seed_templates) for score in (.7,.75)]
    run('pca_validation',420,.65,'raw',a.pca_templates)
    ok=[r for r in candidates if not r['exit_code']]
    if ok:
        best=max(ok,key=lambda r:r['f1']);score=best['score_floor']
        screen=[json.loads(s) for s in (a.screen_root/'results.jsonl').read_text().splitlines()]
        control=next(r for r in screen if r['phase']=='screen' and r['name']=='score_floor_'+str(score))
        selected=best['f1']>control['f1']
        (a.output_root/'selection.json').write_text(json.dumps(dict(score=score,advance=selected),indent=2))
        if selected:
            run('fresh_reference',480,.65,'raw',a.seed_templates)
            run('fresh_matched_score',480,score,'raw',a.seed_templates)
            run('fresh_smooth',480,score,'smooth3',a.seed_templates)
    (a.output_root/'complete.json').write_text(json.dumps(dict(runs=len(outputs)),indent=2))


if __name__=='__main__':main()
