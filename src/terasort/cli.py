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
    session = sub.add_parser(
        "session-sort", help="Experimental bounded multi-file session sorter")
    session.add_argument("--manifest", type=Path, required=True,
                         help="Versioned JSON source clock, geometry and gains")
    session.add_argument("--output-root", type=Path, required=True,
                         help="New run directory; use --resume to continue")
    session.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    session.add_argument("--resume", action="store_true")
    session.add_argument("--freeze-templates", action="store_true",
                         help="Disable evidence collection and template adaptation")
    session.add_argument("--novelty", choices=("off", "shadow", "enroll"), default="off",
                         help="Experimental bounded unknown-event proposals; default off")
    session.add_argument("--core-seconds", type=float, default=2.)
    session.add_argument("--shard-seconds", type=float, default=300.)
    session.add_argument("--residual-passes", type=int, default=3,
                         help="Bounded overlap-recovery passes (1–12; default 3)")
    session.add_argument('--overlap-policy', choices=('strict','interference'), default='strict')
    session.add_argument("--halo-ms", type=float, default=100.)
    session.add_argument("--read-buffer-mb", type=int, default=0,
                         help="Sequential read buffer in MiB, 0 disables grouping (max 1024)")
    session.add_argument("--prefetch-depth", type=int, default=2,
                         help="Bounded queue of raw cores (1–64; default 2)")
    session.add_argument("--floor-snr", type=float, default=4.5)
    session.add_argument('--rescue-floor-snr', type=float,
                         help='Optional lower-threshold final CUDA pass with stricter fitting')
    session.add_argument('--rescue-passes', type=int, default=1)
    session.add_argument('--shift-radius', type=int, default=2,
                         help='Candidate timing search radius in samples (0–8)')
    session.add_argument("--score-floor", type=float, default=.65)
    session.add_argument('--half-width', type=int, default=8,
                         help='Scoring half width in samples (3–30)')
    session.add_argument('--refit-rounds', type=int, default=0,
                         help='Experimental bounded neighbor-subtracted CPU refits (0–2)')
    session.add_argument('--detector-mode', choices=('raw','smooth3'), default='raw')
    session.add_argument("--min-margin", type=float, default=.03)
    session.add_argument("--cache-fraction", type=float, default=.05)
    session.add_argument("--max-candidates", type=int, default=500_000)
    session.add_argument("--vram-limit-gb", type=float, default=12.)
    session.add_argument("--start-sample", type=int, default=0)
    session.add_argument("--stop-sample", type=int)
    quality = sub.add_parser(
        "session-quality", help="Compare completed session shards with ground truth and Kilosort4")
    quality.add_argument("--suite", type=Path, required=True,
                         help="JSON suite with ground truth, session output and Kilosort4 output")
    quality.add_argument("--report", type=Path,
                         help="Optional new JSON report file")
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
    if args.command == "session-sort":
        from .session_sort import run_session
        result = run_session(
            args.manifest, args.output_root, backend=args.backend,
            resume=args.resume, core_seconds=args.core_seconds,
            shard_seconds=args.shard_seconds, halo_ms=args.halo_ms,
            read_buffer_mb=args.read_buffer_mb, prefetch_depth=args.prefetch_depth,
            floor_snr=args.floor_snr, score_floor=args.score_floor,
            half_width=args.half_width,
            refit_rounds=args.refit_rounds,
            detector_mode=args.detector_mode,
            rescue_floor_snr=args.rescue_floor_snr,
            rescue_passes=args.rescue_passes, shift_radius=args.shift_radius,
            min_margin=args.min_margin, cache_fraction=args.cache_fraction,
            max_candidates=args.max_candidates,
            start_sample=args.start_sample, stop_sample=args.stop_sample,
            vram_limit_gb=args.vram_limit_gb,
            freeze_templates=args.freeze_templates, novelty=args.novelty,
            residual_passes=args.residual_passes, overlap_policy=args.overlap_policy)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "session-quality":
        from .session_quality import evaluate_suite
        result = evaluate_suite(args.suite)
        report = json.dumps(result, indent=2) + "\n"
        if args.report is not None:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(report)
        print(report, end="")
        return 0 if result["quality_gate_passed"] else 2
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
