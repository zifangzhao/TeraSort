"""Learn bounded Kilosort4 seed identities, then rebuild waveforms in session units.

Only a fixed preview is copied. Kilosort never receives the complete recording.
Ground truth and previous whole-recording sorter outputs are not inputs.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from terasort.api import run_kilosort
from terasort.candidates.waveforms import geometry_channel_map
from terasort.session_calibration import select_calibration_windows
from terasort.session_manifest import load_session
from terasort.session_models import LocalModels
from terasort.session_signal import iter_cores, preprocess, assess_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--day-id", required=True)
    parser.add_argument("--budget-seconds", type=float, default=60.)
    args = parser.parse_args()
    if not 6 <= args.budget_seconds <= 180:
        parser.error("Calibration budget must be 6–180 seconds")
    session = load_session(args.manifest)
    probe = next((p for p in session.probes if p.probe_id == args.probe_id), None)
    if probe is None:
        parser.error(f"Unknown probe: {args.probe_id}")
    # Three fixed windows regardless of recording duration.
    windows = select_calibration_windows(
        probe, args.day_id, window_seconds=args.budget_seconds/3,
        spacing_seconds=1e30)
    raw_bytes = sum(b-a for a,b in windows) * probe.n_channels * 2
    if raw_bytes > 5 * 1024**3:
        parser.error("Preview exceeds the 5 GiB scratch cap; lower --budget-seconds")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths, mapping, offset = [], [], 0
    for index, (first, stop) in enumerate(windows):
        path = args.output_root / f"preview-{index}.bin"
        with path.open("xb") as handle:
            for core in iter_cores(probe, start_sample=first, stop_sample=stop,
                                   halo_samples=0):
                core.raw.tofile(handle)
        paths.append(path)
        mapping.append((offset, first, stop))
        offset += stop-first
    print(json.dumps({"stage":"preview_written", "raw_bytes":raw_bytes,
                      "windows":windows}), flush=True)
    ksprobe = dict(chanMap=np.arange(probe.n_channels, dtype=np.int32),
                   xc=probe.x_um.astype(np.float32), yc=probe.y_um.astype(np.float32),
                   kcoords=probe.shank.astype(np.float32), n_chan=probe.n_channels)
    run_kilosort(dict(n_chan_bin=probe.n_channels, fs=probe.sample_rate_hz,
                      scale=probe.gain_uv_per_count, nblocks=0),
                  filename=paths, probe=ksprobe, results_dir=args.output_root / "kilosort",
                  backend="deep_tiled", data_dtype="int16", verbose_console=True)
    ksroot = args.output_root / "kilosort"
    times = np.load(ksroot / "spike_times.npy").ravel()
    labels = np.load(ksroot / "spike_clusters.npy").ravel()
    ks_templates = np.load(ksroot / "templates.npy")
    channel_map = np.load(ksroot / "channel_map.npy").ravel()
    inverse_whitening = np.load(ksroot / "whitening_mat_inv.npy")
    physical = ks_templates @ inverse_whitening
    anchors = channel_map[np.argmax(np.max(np.abs(physical), axis=1), axis=1)]
    units, counts = np.unique(labels, return_counts=True)
    units = units[counts >= 20]
    if not len(units) or len(units) > 4096:
        raise RuntimeError("Calibration unit count outside bounded seed budget")
    cap = min(128, 32768 // len(units))
    selected = {}
    for unit in units:
        spikes = np.sort(times[labels == unit])
        selected[int(unit)] = spikes[np.linspace(0, len(spikes)-1,
                                                min(cap, len(spikes)), dtype=int)]
    local_map = geometry_channel_map(probe.geometry, probe.shank, 75., 16)
    samples = {int(unit): [] for unit in units}
    for preview_start, first, stop in mapping:
        for core in iter_cores(probe, start_sample=first, stop_sample=stop,
                               halo_samples=max(31, round(.1*probe.sample_rate_hz))):
            voltage = preprocess(core, probe)
            quality = assess_quality(core, voltage, probe)
            if quality.interval_bad:
                continue
            lo = preview_start + core.core_start-first
            hi = preview_start + core.core_stop-first
            for unit in units:
                anchor = anchors[unit]
                channels = local_map[anchor]
                valid = channels >= 0
                if not np.all(quality.usable_channels[channels[valid]]):
                    continue
                for spike in selected[int(unit)]:
                    if not lo <= spike < hi or not preview_start+31 <= spike < preview_start+stop-first-31:
                        continue
                    t = int(spike-preview_start+first-core.data_start)
                    waveform = np.zeros((61, local_map.shape[1]), np.float32)
                    waveform[:, valid] = voltage[t-30:t+31, channels[valid]]
                    samples[int(unit)].append(waveform)
    waveforms, contacts, retained, supports = [], [], [], []
    for unit in units:
        if len(samples[int(unit)]) < 10:
            continue
        waveform = np.median(np.stack(samples[int(unit)]), axis=0)
        # Bounded rank-three denoising of each local waveform.
        u, s, vt = np.linalg.svd(waveform, full_matrices=False)
        waveform = (u[:, :3] * s[:3]) @ vt[:3]
        waveforms.append(waveform)
        contacts.append(local_map[anchors[unit]])
        retained.append(int(unit))
        supports.append(len(samples[int(unit)]))
    if not retained:
        raise RuntimeError("No calibration units retained after waveform/QC checks")
    models = LocalModels(np.array(waveforms), np.array(contacts),
                         anchors[retained], np.zeros(len(retained), np.int64))
    aligned = models.canonicalize()
    with (args.output_root / "seeds.npz").open("xb") as handle:
        np.savez_compressed(handle, waveforms=models.waveforms, channels=models.channels,
                            anchors=models.anchors, assigned=models.assigned,
                            version=models.version)
    report = dict(method="bounded_kilosort4_identities_session_waveforms_v1",
                  preprocessing="terasort-session-v1", windows=windows,
                  raw_preview_bytes=raw_bytes, calibration_unit_ids=retained,
                  waveform_support=supports, seed_units=len(retained),
                  canonicalized_units=aligned, wall_seconds=time.perf_counter()-started)
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"stage":"seeds_complete", "units":len(retained),
                      "wall_seconds":report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
