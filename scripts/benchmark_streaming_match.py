"""Reproducible CPU/CUDA timing of the bounded local matcher, excluding I/O.

Synthetic waveforms exercise 32 templates on one anchor and 16 contacts. This
is an engineering throughput test, not a spike-sorting accuracy benchmark.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from terasort.streaming import ConsolidatedEvent, LocalTemplateBank


def make_case(n_events, seed=17):
    rng = np.random.default_rng(seed)
    nt, nc, n_templates = 61, 16, 32
    channel_map = np.vstack([np.r_[c, np.arange(nc)[np.arange(nc) != c]]
                             for c in range(nc)]).astype(np.int32)
    pulse = -np.exp(-((np.arange(nt) - nt // 2) / 2.5)**2).astype(np.float32)
    amplitudes = rng.uniform(-4., 4., size=(n_templates, nc)).astype(np.float32)
    amplitudes[:, 0] = 9.
    templates = pulse[None, :, None] * amplitudes[:, None, :]
    bank = LocalTemplateBank.from_dense_templates(
        templates, channel_map, pre_samples=30, sample_rate_hz=20_000)
    events = []
    for i in range(n_events):
        unit = i % n_templates
        waveform = templates[unit] + rng.normal(0, .1, (nt, nc)).astype(np.float32)
        events.append(ConsolidatedEvent(i * 100, i * 100, 0,
                      float(waveform[30, 0]), 9., channel_map[0],
                      waveform, (i,)))
    return bank, events


def benchmark(n_events, repeats):
    os.environ.setdefault("CUPY_CACHE_DIR", str(Path(tempfile.gettempdir()) /
                                                "terasort-cupy-cache"))
    bank, events = make_case(n_events)
    snapshots = bank.snapshots()
    from terasort.streaming_cuda import CudaLocalMatcher
    matcher = CudaLocalMatcher(bank)
    # Compile and initialize outside measured runs.
    gpu_results = matcher.match(events, snapshots)
    gpu_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        gpu_results = matcher.match(events, snapshots)
        gpu_times.append(time.perf_counter() - start)
    start = time.perf_counter()
    cpu_results = [bank.assign(e, snapshots) for e in events]
    cpu_seconds = time.perf_counter() - start
    gpu_ids = np.asarray([r[0] for r in gpu_results], np.int64)
    cpu_ids = np.asarray([r[0] for r in cpu_results], np.int64)
    if not np.array_equal(gpu_ids, cpu_ids):
        raise AssertionError("CPU/CUDA unit assignment differs")
    gpu_scores = np.asarray([r[1] for r in gpu_results])
    cpu_scores = np.asarray([r[1] for r in cpu_results])
    max_score_difference = float(np.max(np.abs(gpu_scores - cpu_scores)))
    if max_score_difference > 2e-5:
        raise AssertionError("CPU/CUDA matching score differs")
    return {"events": n_events, "templates_per_anchor": 32,
            "contacts_per_event": 16, "candidate_pairs": n_events * 32,
            "cpu_matching_seconds": cpu_seconds,
            "gpu_matching_seconds_median": float(np.median(gpu_times)),
            "speedup": cpu_seconds / float(np.median(gpu_times)),
            "max_score_difference": max_score_difference,
            "scope": "matching only; synthetic data; warm CUDA; no SCB I/O or detection"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.events < 1 or args.repeats < 1:
        raise ValueError("Positive events and repeats required")
    print(json.dumps(benchmark(args.events, args.repeats), indent=2))


if __name__ == "__main__":
    main()
