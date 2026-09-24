"""Run a GPU session pilot on the retained public recording and score it.

Sources and historical Kilosort outputs are read-only. Results require a new
directory. The historical baseline is a quality comparison, not a paired
timing measurement.
"""

import argparse
import cProfile
import json
from pathlib import Path
import time

import numpy as np
import psutil

from terasort.session_quality import evaluate_case
from terasort.session_sort import run_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--kilosort-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--start-seconds", type=float, default=0.)
    parser.add_argument("--shard-seconds", type=float, default=300.)
    parser.add_argument("--residual-passes", type=int, default=3)
    parser.add_argument("--floor-snr", type=float, default=4.5,
                        help="Primary candidate threshold; also sets the residual-gain floor")
    parser.add_argument('--rescue-floor-snr', type=float)
    parser.add_argument('--rescue-passes', type=int, default=1)
    parser.add_argument('--shift-radius', type=int, default=2)
    parser.add_argument('--score-floor', type=float, default=.65)
    parser.add_argument('--min-margin', type=float, default=.03)
    parser.add_argument('--half-width', type=int, default=8)
    parser.add_argument('--refit-rounds', type=int, default=0)
    parser.add_argument('--detector-mode', choices=('raw','smooth3'), default='raw')
    parser.add_argument('--overlap-policy', choices=('strict','interference'), default='strict')
    parser.add_argument("--freeze-templates", action="store_true")
    parser.add_argument("--novelty", choices=("off", "shadow", "enroll"), default="off")
    parser.add_argument("--seed-templates", type=Path,
                        help="Reuse a session-preprocessed calibration bank for fitting ablations")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    meta = json.loads((args.dataset_root / "recording_int16.json").read_text())
    positions = np.load(args.dataset_root / "channel_positions_um.npy")
    ks_positions = np.load(args.kilosort_dir / "channel_positions.npy")
    np.testing.assert_allclose(positions, ks_positions, atol=1e-4)
    shanks = np.load(args.kilosort_dir / "channel_shanks.npy").astype(int)
    rate = float(meta["sampling_frequency"])
    first = round(args.start_seconds * rate)
    stop = first + round(args.seconds * rate)
    if not 0 <= first < stop <= meta["shape"][0]:
        parser.error("--seconds must be within the source recording")
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = args.output_root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "session_id": "public250-gpu-pilot",
        "probes": [{
            "probe_id": "probeA", "sample_rate_hz": rate,
            "gain_uv_per_count": meta["conversion_to_uv"],
            "geometry": {"x_um": positions[:, 0].tolist(),
                         "y_um": positions[:, 1].tolist(),
                         "shank": shanks.ravel().tolist()},
            "segments": [{"path": meta["binary"], "start_sample": 0,
                          "n_samples": meta["shape"][0], "day_id": "day1"}],
            "gaps": [],
        }],
    }, indent=2))
    if args.seed_templates:
        definition = json.loads(manifest.read_text())
        definition["probes"][0].update(
            seed_templates=str(args.seed_templates.resolve(strict=True)),
            seed_preprocessing_id="terasort-session-v1")
        manifest.write_text(json.dumps(definition, indent=2))
    started = time.perf_counter()
    def progress(event):
        event = {**event, "elapsed_seconds": time.perf_counter()-started,
                 "rss_bytes": psutil.Process().memory_info().rss}
        print(json.dumps(event), flush=True)
        with (args.output_root / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    try:
        report = run_session(manifest, args.output_root / "session",
                             backend="cuda", start_sample=first, stop_sample=stop,
                             shard_seconds=min(args.shard_seconds, args.seconds),
                             freeze_templates=args.freeze_templates,
                             novelty=args.novelty,
                             residual_passes=args.residual_passes,
                             floor_snr=args.floor_snr,
                             rescue_floor_snr=args.rescue_floor_snr,
                             rescue_passes=args.rescue_passes, shift_radius=args.shift_radius,
                             score_floor=args.score_floor, min_margin=args.min_margin,
                             half_width=args.half_width,
                             refit_rounds=args.refit_rounds,
                             detector_mode=args.detector_mode,
                             overlap_policy=args.overlap_policy,
                             progress=progress)
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(args.output_root / "profile.pstats"))
    report["profile_enabled"] = args.profile
    report["process_memory"] = psutil.Process().memory_info()._asdict()
    (args.output_root / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"stage": "scoring", **report}), flush=True)
    case = {"name": "public250-pilot", "kind": "base",
            "ground_truth": str(args.dataset_root / "ground_truth.npz"),
            "session_output": str(args.output_root / "session"),
            "probe_id": "probeA", "kilosort_dir": str(args.kilosort_dir),
            "start_sample": first, "stop_sample": stop}
    result = evaluate_case(case)
    (args.output_root / "quality.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"stage": "quality_complete", **{
        route: {key: value for key, value in result[route].items()
                if key != "unit_matches"} for route in ("session", "kilosort4")}}),
        flush=True)


if __name__ == "__main__":
    main()
