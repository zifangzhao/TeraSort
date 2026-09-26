import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from terasort.session_gpu import CudaResidualMatcher
from terasort.session_models import LocalModels
from terasort.session_manifest import load_session
from terasort.session_signal import assess_quality, iter_cores, preprocess


def main():
    parser = argparse.ArgumentParser(
        description="Paired benchmark of legacy and shared-memory smooth3 detection.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=60.)
    parser.add_argument("--floor-snr", type=float, default=4.5)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.output.exists():
        raise FileExistsError(args.output)

    probe = load_session(args.manifest).probes[0]
    first = round(args.start_seconds * probe.sample_rate_hz)
    core = next(iter_cores(
        probe, start_sample=first,
        stop_sample=first + round(2 * probe.sample_rate_hz),
        halo_samples=round(.1 * probe.sample_rate_hz)))
    voltage = preprocess(core, probe)
    quality = assess_quality(core, voltage, probe)
    voltage[:, ~quality.usable_channels] = 0
    raw_noise = np.where(
        quality.usable_channels, quality.noise_uv, np.inf).astype(np.float32)
    device = cp.asarray(voltage, dtype=cp.float32, order="C")
    dummy = LocalModels(np.ones((1, 61, 1), np.float32), [[0]], [0], [0])
    matcher = CudaResidualMatcher(dummy)

    def legacy_smooth3():
        # Original implementation: allocate the full smoothed core, estimate
        # noise from sampled rows, then run the generic fused peak detector.
        filtered = cp.empty_like(device)
        filtered[0] = device[0]
        filtered[-1] = device[-1]
        filtered[1:-1] = .25 * (
            device[:-2] + 2 * device[1:-1] + device[2:])
        sample = cp.asnumpy(filtered[::max(1, len(filtered) // 4000)])
        center = np.median(sample, axis=0)
        estimate = np.median(np.abs(sample - center), axis=0) / .67448975
        noise = np.where(
            np.isfinite(raw_noise), np.maximum(estimate, .01), np.inf
        ).astype(np.float32)
        return matcher.detect(
            filtered, noise, args.floor_snr, mode="raw")

    def shared_smooth3():
        return matcher.detect(
            device, raw_noise, args.floor_snr, mode="smooth3")

    expected = legacy_smooth3()
    actual = shared_smooth3()
    old = np.asarray(expected, dtype=np.float64).reshape((-1, 3))
    new = np.asarray(actual, dtype=np.float64).reshape((-1, 3))
    np.testing.assert_array_equal(old, new)

    timings = {"legacy_full_temporary": [], "shared_memory_fused": []}
    calls = [
        ("legacy_full_temporary", legacy_smooth3),
        ("shared_memory_fused", shared_smooth3),
    ]
    for repeat in range(args.repeats):
        order = calls if repeat % 2 == 0 else list(reversed(calls))
        for name, call in order:
            cp.cuda.get_current_stream().synchronize()
            started = time.perf_counter()
            call()
            cp.cuda.get_current_stream().synchronize()
            timings[name].append(time.perf_counter() - started)

    medians = {name: float(np.median(values))
               for name, values in timings.items()}
    report = {
        "shape": list(voltage.shape),
        "start_sample_with_halo": int(core.data_start),
        "floor_snr": args.floor_snr,
        "candidate_count": len(actual),
        "repeats_per_method": args.repeats,
        "median_seconds": medians,
        "speedup": medians["legacy_full_temporary"] /
                   medians["shared_memory_fused"],
        "exact_candidate_and_snr_equality": True,
        "max_absolute_snr_difference": 0.0,
        "includes_noise_estimation_sort_and_host_event_list": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
