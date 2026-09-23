"""Optional end-to-end run on the prepared 384-channel public sample."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from terasort import run_kilosort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--backend", choices=("auto", "standard", "deep_tiled", "cublas"), default="auto")
    args = parser.parse_args()
    metadata = json.loads((args.sample_root / "data/recording_int16.json").read_text())
    positions = np.load(args.sample_root / "data/channel_positions_um.npy")
    probe = dict(chanMap=np.arange(len(positions), dtype=np.int32),
                 xc=positions[:, 0].astype(np.float32),
                 yc=positions[:, 1].astype(np.float32),
                 kcoords=np.zeros(len(positions), dtype=np.float32), n_chan=len(positions))
    settings = dict(n_chan_bin=metadata["shape"][1], fs=metadata["sampling_frequency"],
                    tmin=0., tmax=args.seconds, scale=metadata["conversion_to_uv"],
                    shift=metadata.get("offset_uv", 0.))
    started = time.perf_counter()
    result = run_kilosort(settings, probe=probe, filename=metadata["binary"],
                          results_dir=args.results_dir, data_dtype="int16",
                          device=torch.device("cuda"), backend=args.backend,
                          torch_thread_lim=8)
    spike_times = np.load(args.results_dir / "spike_times.npy")
    spike_clusters = np.load(args.results_dir / "spike_clusters.npy")
    assert len(result) == 9 and len(spike_times) == len(spike_clusters)
    print(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,
                          spikes=len(spike_times), units=len(np.unique(spike_clusters)),
                          backend=args.backend, results_dir=str(args.results_dir))))


if __name__ == "__main__":
    main()
