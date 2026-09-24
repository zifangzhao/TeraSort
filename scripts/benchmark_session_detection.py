"""Paired detector timings on one real source core; no ground truth required."""
import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from terasort.candidates.detectors import CudaDetector
from terasort.session_gpu import CudaResidualMatcher
from terasort.session_models import LocalModels
from terasort.session_manifest import load_session
from terasort.session_signal import iter_cores, preprocess, assess_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-seconds', type=float, default=60.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    probe = load_session(args.manifest).probes[0]
    first = round(args.start_seconds*probe.sample_rate_hz)
    core = next(iter_cores(probe, start_sample=first,
                          stop_sample=first+round(2*probe.sample_rate_hz),
                          halo_samples=round(.1*probe.sample_rate_hz)))
    voltage = preprocess(core, probe)
    quality = assess_quality(core, voltage, probe)
    voltage[:, ~quality.usable_channels] = 0
    noise = np.where(quality.usable_channels, quality.noise_uv, np.inf).astype(np.float32)
    device = cp.asarray(voltage, dtype=cp.float32, order='C')
    old = CudaDetector()
    model = LocalModels(np.ones((1, 61, 1), np.float32), [[0]], [0], [0])
    new = CudaResidualMatcher(model)

    def old_detect():
        q = cp.abs(device)/cp.asarray(noise)[None, :]
        flat = cp.sort(old.detect(q, floor=4.5, capacity=500000))
        indices = cp.asnumpy(flat)
        t, c = np.divmod(indices, device.shape[1])
        snr = cp.asnumpy(q.ravel()[flat])
        return list(zip(t.astype(np.int32), c.astype(np.int32), snr.astype(np.float32)))

    def new_detect():
        return new.detect(device, noise, 4.5)

    # Warm compilation and allocations, then alternate order to reduce bias.
    expected, actual = old_detect(), new_detect()
    np.testing.assert_array_equal(np.array(expected), np.array(actual))
    timings = {'old': [], 'fused': []}
    for repeat in range(20):
        order = [('old', old_detect), ('fused', new_detect)]
        if repeat % 2:
            order.reverse()
        for name, call in order:
            cp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            call()
            cp.cuda.get_current_stream().synchronize()
            timings[name].append(time.perf_counter()-start)
    report = dict(shape=list(voltage.shape), events=len(actual), repeats=20,
                  seconds=timings, median_seconds={k:float(np.median(v)) for k,v in timings.items()},
                  exact_event_equality=True, includes_host_event_list=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
