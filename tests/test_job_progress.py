import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from terasort.job_progress import JobProgress, apply_progress_snapshot
from terasort.kilosort_progress import SOURCES, instrument, kilosort_progress


def test_three_phases_counters_and_all_remaining_work(tmp_path):
    now=[0.]
    p=JobProgress(tmp_path,clock=lambda:now[0],interval=0)
    p.stage('preprocess')
    list(p.track(range(10),'Whitening batches',0.,.98))
    now[0]=10.
    p.publish(force=True)
    snapshot=p.snapshot()
    assert len(snapshot['phases'])==3
    assert snapshot['task_progress']['completed']==10
    assert snapshot['phases'][0]['percent']<100
    assert snapshot['phases'][1]['percent']==0
    assert snapshot['eta_seconds']>0
    assert all(x['eta_seconds'] is not None for x in snapshot['phases'])
    p.stage('cluster2')
    assert [x['percent'] for x in p.snapshot()['phases'][:2]]==[100,100]
    p.counter(2,10,'Clustering regions')
    before=p.snapshot()['percent']
    p.counter(0,10,'Another operation')
    assert p.snapshot()['percent']>=before
    assert json.loads((tmp_path/'terasort_progress.json').read_text())['pid']==os.getpid()


def test_iterator_counts_completed_work_and_break_is_not_completion():
    p=JobProgress(interval=0)
    p.stage('preprocess')
    for i in p.track(range(10),'work'):
        if i==3: break
    assert p.task['completed']==3
    with pytest.raises(RuntimeError):
        for i in p.track(range(10),'failed work'):
            if i==2: raise RuntimeError('test')
    assert p.task['completed']==2


def test_skip_drift_staging_lfp_and_terminal_state(tmp_path):
    p=JobProgress(tmp_path,skip_drift=True,staging_seconds=10,lfp=True)
    assert 'drift' not in p.fractions
    p.stage('lfp')
    assert p.snapshot()['eta_seconds'] is None
    job={'pid':os.getpid(),'status':'running'}
    apply_progress_snapshot(job,tmp_path/'terasort_progress.json')
    assert job['stage']=='Waiting for LFP export'
    job['status']='completed'
    apply_progress_snapshot(job,tmp_path/'terasort_progress.json')
    assert all(x['percent']==100 for x in job['phases'])
    other={'pid':os.getpid()+1,'status':'running'}
    apply_progress_snapshot(other,tmp_path/'terasort_progress.json')
    assert 'phases' not in other


def test_bad_snapshot_and_io_errors_do_not_abort(tmp_path):
    path=tmp_path/'terasort_progress.json'
    for text in ('{', 'x'*65537, '{}'):
        path.write_text(text)
        apply_progress_snapshot({'pid':os.getpid(),'status':'running'},path)
    p=JobProgress(tmp_path/'missing')
    p.stage('preprocess')


def test_all_pinned_hooks_compile_and_restore_on_exception(tmp_path):
    originals={(m,n):getattr(importlib.import_module('kilosort.'+m),n) for m,n in SOURCES}
    with pytest.raises(RuntimeError):
        with kilosort_progress(tmp_path):
            for (m,n),original in originals.items():
                assert getattr(importlib.import_module('kilosort.'+m),n) is not original
            raise RuntimeError('interrupted')
    for (m,n),original in originals.items():
        assert getattr(importlib.import_module('kilosort.'+m),n) is original


def test_whitening_outputs_identical_and_counts_sampled_batches():
    module=importlib.import_module('kilosort.preprocessing')
    p=JobProgress(interval=0)
    p.stage('preprocess')
    generator=torch.Generator().manual_seed(2)
    batches=[torch.randn(2,18,generator=generator) for _ in range(6)]
    class Reader:
        chan_map=[0,1]
        device=torch.device('cpu')
        nt=1
        n_batches=6
        def padded_batch_to_torch(self,j): return batches[j].clone()
    reader=Reader()
    args=(reader,np.asarray([0.,0.]),np.asarray([0.,20.]))
    original=module.get_whitening_matrix(*args,nskip=2,nrange=2)
    tracked=instrument(module.get_whitening_matrix,'preprocessing',p)(*args,nskip=2,nrange=2)
    torch.testing.assert_close(original,tracked,rtol=0,atol=0)
    assert p.task['completed']==3 and p.task['total']==3
