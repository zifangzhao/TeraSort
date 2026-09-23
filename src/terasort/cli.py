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
    web = sub.add_parser("web", help="Run the local browser dashboard and job queue")
    web.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost)")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--state-dir", type=Path, help="Persistent job state directory")
    web.add_argument("--token", help="Bearer token required when binding beyond localhost")
    web.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    lfp = sub.add_parser("lfp", help="Stream anti-aliased INT16 voltage to an LFP binary")
    lfp.add_argument("--filename", type=Path, required=True, help="Time-major interleaved INT16 input")
    lfp.add_argument("--output", type=Path, required=True, help="Time-major interleaved INT16 LFP output")
    lfp.add_argument("--sample-rate", type=int, required=True, help="Input sample rate in Hz")
    lfp.add_argument("--n-channels", type=int, required=True)
    lfp.add_argument("--output-rate", type=int, default=1250)
    lfp.add_argument("--passband-hz", type=float, default=500.0)
    lfp.add_argument("--scale-uv-per-count", type=float)
    lfp.add_argument("--chunk-seconds", type=float, default=5.0)
    lfp.add_argument("--workers", type=int, default=8,
                     help="CPU channel-filter workers (default 8)")
    lfp.add_argument("--resume", action="store_true", help="Continue a matching .partial export")
    sort = sub.add_parser("sort", help="Sort one or more ordered binary files into one Kilosort session")
    sort.add_argument("--settings", type=Path, required=True,
                      help="JSON dictionary of Kilosort settings, including n_chan_bin")
    sort.add_argument("--filename", type=Path, required=True, action="append",
                      help="Raw binary file; repeat in acquisition order for one shared session")
    probe = sort.add_mutually_exclusive_group()
    probe.add_argument("--probe-name", help="Bundled Kilosort probe name")
    probe.add_argument("--probe-json", type=Path, help="Probe dictionary saved as JSON")
    sort.add_argument("--results-dir", type=Path, required=True)
    sort.add_argument("--data-dtype", default="int16")
    sort.add_argument("--backend", choices=("auto", "standard", "deep_tiled", "cublas"),
                      default="auto")
    sort.add_argument("--no-fast-int16", action="store_true")
    sort.add_argument("--stage-dir", type=Path,
                      help="New local scratch directory for one-time source copies (retained after sorting)")
    sort.add_argument("--skip-drift-correction", action="store_true",
                      help="Use Kilosort nblocks=0; skips motion estimation and its detection pass")
    sort.add_argument("--invert-sign", action="store_true")
    sort.add_argument("--no-car", action="store_true")
    sort.add_argument("--verbose", action="store_true")
    sort.add_argument("--lfp-output", type=Path,
                      help="Export 1250 Hz LFP during final clustering, then wait for completion")
    sort.add_argument("--lfp-passband-hz", type=float, default=500.0)
    sort.add_argument("--lfp-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "backends":
        print("\n".join(available_backends()))
        return 0
    if args.command == "web":
        from .web import serve
        import os
        serve(host=args.host, port=args.port, state_dir=args.state_dir,
              token=args.token or os.environ.get("TERASORT_WEB_TOKEN"),
              open_browser=not args.no_browser)
        return 0
    if args.command == "lfp":
        from .lfp import export_lfp
        result = export_lfp(args.filename, args.output,
            sample_rate_hz=args.sample_rate, n_channels=args.n_channels,
            output_rate_hz=args.output_rate, passband_hz=args.passband_hz,
            scale_uv_per_count=args.scale_uv_per_count,
            chunk_seconds=args.chunk_seconds, workers=args.workers,
            resume=args.resume)
        print(json.dumps(result, indent=2))
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
    filename = args.filename[0] if len(args.filename) == 1 else args.filename
    run_kilosort(settings, probe=probe_dict, probe_name=args.probe_name,
                 filename=filename, results_dir=args.results_dir,
                 data_dtype=args.data_dtype, backend=args.backend,
                 fast_int16=not args.no_fast_int16,
                 skip_drift_correction=args.skip_drift_correction,
                 stage_dir=args.stage_dir,
                 lfp_output=args.lfp_output,
                 lfp_passband_hz=args.lfp_passband_hz,
                 lfp_workers=args.lfp_workers,
                 invert_sign=args.invert_sign, do_CAR=not args.no_car,
                 verbose_console=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
