"""Diagnostic only: reconstruct K4 identities in the session voltage frame.

Uses fixed source windows and 48 waveforms/unit for both full-recording and
preview K4 identities. Full-recording identities contain evaluation/future
information: results are not an independent quality gate or production route.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from terasort.candidates.waveforms import geometry_channel_map
from terasort.session_calibration import select_calibration_windows
from terasort.session_manifest import load_session
from terasort.session_models import LocalModels
from terasort.session_signal import iter_cores, preprocess, assess_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--kilosort-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--preview-report', type=Path,
                        help='Required only for K4 timestamps in concatenated preview coordinates')
    args = parser.parse_args()
    session = load_session(args.manifest)
    if len(session.probes) != 1 or len(session.probes[0].segments) != 1:
        parser.error('This diagnostic requires one probe and one source segment')
    p = session.probes[0]
    if p.segments[0].start_sample != 0:
        parser.error('Diagnostic expects K4 source timestamps starting at zero')
    windows = select_calibration_windows(p, p.segments[0].day_id,
                                         window_seconds=20., spacing_seconds=1e30)
    root = args.kilosort_dir
    positions = np.load(root / 'channel_positions.npy')
    mapping = np.load(root / 'channel_map.npy').ravel().astype(int)
    np.testing.assert_allclose(positions, p.geometry[mapping], atol=1e-4)
    times = np.load(root / 'spike_times.npy', mmap_mode='r').ravel()
    labels = np.load(root / 'spike_clusters.npy', mmap_mode='r').ravel()
    if len(times) != len(labels):
        raise ValueError('Expected sorted K4 times aligned with labels')
    for start in range(0, len(times), 1_000_000):
        if np.any(np.diff(times[max(0,start-1):start+1_000_000]) < 0):
            raise ValueError('Expected sorted K4 times aligned with labels')
    if args.preview_report:
        preview = json.loads(args.preview_report.read_text())
        if preview['windows'] != [list(w) for w in windows]:
            raise ValueError('Preview windows differ from diagnostic source windows')
        start, source_ranges = 0, []
        for a, b in windows:
            source_ranges.append((start, a, b))
            start += b-a
    else:
        source_ranges = [(a, a, b) for a, b in windows]
    # Bound the event table independently of the full K4 output length.
    event_count = sum(np.searchsorted(times, clock+b-a)-np.searchsorted(times, clock)
                      for clock,a,b in source_ranges)
    if event_count > 2_000_000:
        raise ValueError('Calibration event budget exceeded')
    selected_times, selected_labels = [], []
    for clock, a, b in source_ranges:
        lo, hi = np.searchsorted(times, [clock+31, clock+b-a-31])
        selected_times.append(np.asarray(times[lo:hi], np.int64)-clock+a)
        selected_labels.append(np.asarray(labels[lo:hi], np.int64))
    st, sl = np.concatenate(selected_times), np.concatenate(selected_labels)
    units, counts = np.unique(sl, return_counts=True)
    units = units[counts >= 20]
    if not len(units) or len(units)*48 > 32768:
        raise ValueError('Fixed 48-waveform/unit budget exceeds 32768 or no supported units')
    templates = np.load(root / 'templates.npy', mmap_mode='r')
    inverse = np.load(root / 'whitening_mat_inv.npy')
    if units.min() < 0 or units.max() >= len(templates):
        raise ValueError('K4 cluster IDs do not index templates')
    anchors = {}
    selected = {}
    # Unwhiten one template at a time: no full unit x time x channel tensor copy.
    for u in units:
        physical = templates[u] @ inverse
        anchors[int(u)] = int(mapping[np.argmax(np.max(np.abs(physical), axis=0))])
        spikes = st[sl == u]
        selected[int(u)] = spikes[np.linspace(0, len(spikes)-1, min(48,len(spikes)), dtype=int)]
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    local = geometry_channel_map(p.geometry, p.shank, 75., 16)
    samples = {int(u): [] for u in units}
    read_bytes = 0
    for a,b in windows:
        for core in iter_cores(p, start_sample=a, stop_sample=b,
                               halo_samples=max(31, round(.1*p.sample_rate_hz))):
            voltage = preprocess(core, p)
            quality = assess_quality(core, voltage, p)
            read_bytes += core.source_bytes_read
            if quality.interval_bad:
                continue
            for u in units:
                channels = local[anchors[int(u)]]
                valid = channels >= 0
                if not np.all(quality.usable_channels[channels[valid]]):
                    continue
                for spike in selected[int(u)]:
                    if core.core_start <= spike < core.core_stop:
                        t = int(spike-core.data_start)
                        w = np.zeros((61, local.shape[1]), np.float32)
                        w[:, valid] = voltage[t-30:t+31, channels[valid]]
                        samples[int(u)].append(w)
        print(json.dumps({'window_complete': [a,b], 'source_bytes_read':read_bytes}), flush=True)
    waves, contacts, retained, support = [], [], [], []
    for u in units:
        snippets = samples[int(u)]
        if len(snippets) < 10:
            continue
        w = np.median(np.stack(snippets), axis=0)
        left, singular, right = np.linalg.svd(w, full_matrices=False)
        waves.append((left[:, :3]*singular[:3]) @ right[:3])
        contacts.append(local[anchors[int(u)]])
        retained.append(int(u))
        support.append(len(snippets))
    if not retained:
        raise ValueError('No usable waveforms')
    models = LocalModels(np.array(waves), np.array(contacts),
                         [anchors[u] for u in retained], np.zeros(len(retained), np.int64))
    aligned = models.canonicalize()
    with (args.output_root / 'seeds.npz').open('xb') as handle:
        np.savez_compressed(handle, waveforms=models.waveforms, channels=models.channels,
                            anchors=models.anchors, assigned=models.assigned, version=models.version)
    report = dict(diagnostic_only=True, preprocessing='terasort-session-v1',
        identities_from=str(root.resolve()), uses_full_recording_identities=not bool(args.preview_report),
        windows=windows, waveform_cap_per_unit=48, seed_units=len(retained),
        calibration_unit_ids=retained, waveform_support=support, canonicalized_units=aligned,
        source_bytes_read=read_bytes, reconstruction_seconds=time.perf_counter()-started)
    (args.output_root / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key:report[key] for key in (
        'diagnostic_only', 'uses_full_recording_identities', 'seed_units',
        'source_bytes_read', 'reconstruction_seconds')}), flush=True)


if __name__ == '__main__':
    main()
