"""Learn split proposals from the retained bounded refinement cache, without GT."""
import argparse
from bisect import bisect_left,bisect_right
import json
from pathlib import Path
import time

import numpy as np
import psutil

from terasort.session_manifest import load_session
from terasort.session_models import LocalModels
from terasort.session_refinement import StratifiedEvidence
from terasort.session_pca_split import split_templates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--seed-templates',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    started=time.perf_counter()
    probe=load_session(args.manifest).probes[0]
    report=json.loads((args.cache_root/'report.json').read_text())
    groups=report['groups']
    windows_all=groups['training']+groups['validation']
    if (not groups['training'] or not groups['validation'] or len(windows_all)>32
            or len(set(windows_all))!=len(windows_all)):
        raise ValueError('Use nonempty, disjoint partitions totaling at most 32 cached windows')
    args.output_root.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    from terasort.session_gpu import CudaResidualMatcher
    cp.get_default_memory_pool().set_limit(size=12*1024**3)
    model=LocalModels.load(args.seed_templates,probe)
    model.canonicalize()
    matcher=CudaResidualMatcher(model)
    evidence={name:StratifiedEvidence(len(model.waveforms),windows,budget=16384,per_window=16)
              for name,windows in groups.items()}
    contact_sets=[set(row[row>=0]) for row in model.channels]
    for name,windows in groups.items():
        for start in windows:
            with np.load(args.cache_root/'cache'/f'{start}.npz') as data:
                if bool(data['bad']):continue
                signal,noise,usable=data['signal'],data['noise'],data['usable']
                data_start=int(data['data_start']);stop=start+round(2*probe.sample_rate_hz)
                _,matches=matcher.match(signal,noise,4.5,core_start=start-data_start,
                    core_stop=stop-data_start,refractory_samples=round(.0005*probe.sample_rate_hz),rescue_floor_snr=3.5)
                times=[m.source_sample for m in matches]
                for i,m in enumerate(matches):
                    if m.score < .75 or m.relative_margin < .03 or not .3<=m.amplitude<=3:continue
                    t=m.source_sample
                    if any(j!=i and contact_sets[m.unit]&contact_sets[matches[j].unit]
                           for j in range(bisect_left(times,t-60),bisect_right(times,t+60))):continue
                    ch=model.channels[m.unit];valid=ch>=0
                    if not np.all(usable[ch[valid]]):continue
                    w=np.zeros_like(model.waveforms[m.unit]);w[:,valid]=signal[t-30:t+31,ch[valid]]/m.amplitude
                    evidence[name].observe(m.unit,t+data_start,start,w)
            print(json.dumps(dict(stage='evidence',partition=name,start=start)),flush=True)
    split,audit=split_templates(model,evidence['training'],evidence['validation'])
    with (args.output_root/'seeds.npz').open('xb') as f:
        np.savez_compressed(f,waveforms=split.waveforms,channels=split.channels,anchors=split.anchors,
                            assigned=split.assigned,version=split.version)
    (args.output_root/'audit.json').write_text(json.dumps(audit,indent=2))
    summary=dict(splits=sum(r['split'] for r in audit),models=len(split.waveforms),groups=groups,
                 ground_truth_used=False,cache_root=str(args.cache_root),seed=str(args.seed_templates),
                 wall_seconds=time.perf_counter()-started,
                 process_memory=psutil.Process().memory_info()._asdict(),
                 gpu_pool_bytes=cp.get_default_memory_pool().total_bytes(),
                 evidence_rows={name:sum(len(v) for v in e.rows.values()) for name,e in evidence.items()})
    (args.output_root/'report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
