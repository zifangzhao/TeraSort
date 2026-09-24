"""Small repeated-source memory smoke test; not a real long-recording claim.

Run with an installed TeraSort environment. Each duration reuses the same
read-only INT16 file without making 10x/100x source copies. Published shards
and memory samples are real; cache behavior of a remote 100 TB source is not.
"""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from terasort.session_manifest import load_session
from terasort.session_signal import iter_cores, preprocess
from terasort.session_sort import run_session


def benchmark(root, *, backend="cpu"):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(400)
    n_samples = 6000
    raw = rng.normal(0, 3, (n_samples, 2)).astype(np.int16)
    x = np.arange(61) - 30
    shape = np.exp(-(x / 2.7) ** 2)
    for center in (600, 2600, 4800):
        raw[center-30:center+31, 0] -= np.rint(400 * shape).astype(np.int16)
    source = root / "source.bin"
    raw.tofile(source)
    base_probe = {
        "probe_id": "probeA", "sample_rate_hz": 20_000,
        "gain_uv_per_count": .2,
        "geometry": {"x_um": [0, 100], "y_um": [0, 0],
                     "shank": [0, 1]},
        "seed_templates": "seed.npy",
        "seed_preprocessing_id": "terasort-session-v1",
    }
    manifest_path = root / "seed-manifest.json"
    seed_probe = {**base_probe,
                  "segments": [{"path": "source.bin", "start_sample": 0,
                                "n_samples": n_samples, "day_id": "day1"}],
                  "gaps": []}
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "session_id": "seed",
        "probes": [{key: value for key, value in seed_probe.items()
                    if key not in ("seed_templates", "seed_preprocessing_id")}]}))
    probe = load_session(manifest_path).probes[0]
    voltage = preprocess(next(iter_cores(probe, core_seconds=.3)), probe)
    seed = np.zeros((1, 61, 2), np.float32)
    seed[0] = voltage[570:631]
    np.save(root / "seed.npy", seed)
    reports = {}
    for repeat in (1, 10, 100):
        manifest = root / f"manifest-{repeat}.json"
        probe_data = {**base_probe,
                      "segments": [{"path": "source.bin",
                                    "start_sample": i * n_samples,
                                    "n_samples": n_samples,
                                    "day_id": "day1"}
                                   for i in range(repeat)],
                      "gaps": []}
        manifest.write_text(json.dumps({"schema_version": 1,
                                        "session_id": f"repeat-{repeat}",
                                        "probes": [probe_data]}))
        report = run_session(manifest, root / f"out-{repeat}",
                             backend=backend, core_seconds=.3,
                             shard_seconds=300., floor_snr=4.)
        reports[str(repeat)] = report
    # RSS includes the interpreter and libraries. Allow 128 MiB allocator
    # headroom while rejecting growth proportional to the 100x duration.
    baseline = reports["1"]["peak_process_rss_bytes"]
    long_peak = reports["100"]["peak_process_rss_bytes"]
    passed = long_peak <= baseline + 128 * 1024**2
    return {"bounded_rss_smoke_passed": passed,
            "rss_growth_bytes": long_peak-baseline,
            "runs": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-root", type=Path,
                        help="New directory to retain fixture and results")
    args = parser.parse_args()
    if args.output_root:
        if args.output_root.exists():
            parser.error("--output-root must be a new directory")
        result = benchmark(args.output_root, backend=args.backend)
        (args.output_root / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark(Path(directory), backend=args.backend)
    print(json.dumps(result, indent=2))
    return 0 if result["bounded_rss_smoke_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
