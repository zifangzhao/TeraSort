"""Command-line entry point for Kilosort-compatible sorting."""

import argparse
import json
from pathlib import Path

import numpy as np

from .api import available_backends, run_kilosort


def main(argv=None):
    parser = argparse.ArgumentParser(prog="terasort", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backends", help="List available sorting backends")
    sort = sub.add_parser("sort", help="Sort one binary recording with Kilosort-compatible output")
    sort.add_argument("--settings", type=Path, required=True,
                      help="JSON dictionary of Kilosort settings, including n_chan_bin")
    sort.add_argument("--filename", type=Path, required=True, help="Raw binary file")
    probe = sort.add_mutually_exclusive_group()
    probe.add_argument("--probe-name", help="Bundled Kilosort probe name")
    probe.add_argument("--probe-json", type=Path, help="Probe dictionary saved as JSON")
    sort.add_argument("--results-dir", type=Path, required=True)
    sort.add_argument("--data-dtype", default="int16")
    sort.add_argument("--backend", choices=("auto", "standard", "deep_tiled", "cublas"),
                      default="auto")
    sort.add_argument("--no-fast-int16", action="store_true")
    sort.add_argument("--skip-drift-correction", action="store_true",
                      help="Use Kilosort nblocks=0; skips motion estimation and its detection pass")
    sort.add_argument("--invert-sign", action="store_true")
    sort.add_argument("--no-car", action="store_true")
    sort.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "backends":
        print("\n".join(available_backends()))
        return 0
    settings = json.loads(args.settings.read_text())
    if not isinstance(settings, dict) or not isinstance(settings.get("n_chan_bin"), int):
        parser.error("--settings must be a JSON object with integer n_chan_bin")
    probe_dict = json.loads(args.probe_json.read_text()) if args.probe_json else None
    if probe_dict is not None:
        if not isinstance(probe_dict, dict):
            parser.error("--probe-json must contain a Kilosort probe dictionary")
        for key, dtype in (("chanMap", np.int32), ("xc", np.float32),
                           ("yc", np.float32), ("kcoords", np.float32)):
            if key in probe_dict:
                probe_dict[key] = np.asarray(probe_dict[key], dtype=dtype)
    run_kilosort(settings, probe=probe_dict, probe_name=args.probe_name,
                 filename=args.filename, results_dir=args.results_dir,
                 data_dtype=args.data_dtype, backend=args.backend,
                 fast_int16=not args.no_fast_int16,
                 skip_drift_correction=args.skip_drift_correction,
                 invert_sign=args.invert_sign, do_CAR=not args.no_car,
                 verbose_console=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
