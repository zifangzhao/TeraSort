# FF2 real-data pilot, 2026-09-24

Completed three CUDA runs on D82 Linear A, 128 channels at 20 kHz. Each sorts source seconds 60–90 directly from the read-only SMB amplifier.dat. Results are in `F:\sortingDevelopment\ff2_real_pilot_20260924_01`; no original files or earlier outputs were replaced.

## Calibration and provenance

A fixed 60-second preview comprises 20 seconds near the beginning, middle, and end of the 1761.17-second recording (307.2 MB scratch). Kilosort 4.1.7 via the existing deep_tiled route supplies seed identities; waveforms are rebuilt with session preprocessing. Calibration retained 72 templates and took 40.82 seconds including preview staging and rebuilding. These are templates, not independently validated neurons. All dense runs freeze the same bank and use rescue SNR 3.5. Calibration does not include the test interval.

Coordinates came from the retained FF2 SCB channel_positions_um dataset, acquisition-order shank groups from amplifier.xml, and gain 0.195 uV/count from Intan/retained preprocessing metadata. The SCB's third coordinate is z, not a shank ID. Independent verification against the physical probe drawing is still pending. The old local staging copy was absent; current sources are read directly from the original network folder.

## Dense-pass results

| Setting | Active template IDs | Assigned spikes | ISIs <1.5 ms | Wall seconds | Peak process RSS | Peak total device VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Raw, score 0.65 | 72 | 26,251 | 3.75% | 6.58 | 495 MB | 1.24 GB |
| Raw, score 0.75 | 71 | 18,453 | 3.36% | 6.05 | 496 MB | 1.24 GB |
| Smooth3, score 0.75 | 71 | 19,601 | 3.51% | 6.64 | 500 MB | 1.31 GB |

ISIs are consecutive sorted within-unit intervals divided by all within-unit intervals, pooled across units; this is not an estimated contamination fraction. All sizes are decimal. Dense times exclude one-time calibration. Runs were sequential, not a randomized cold/warm benchmark. Device VRAM includes other device allocations; process RSS is sampled telemetry, not a guaranteed instantaneous peak. No same-interval independent Kilosort comparison was performed.

Each run read 168.96 MB including halos for 153.6 MB of raw coverage. Assignment shard size was 7.23–7.99 MB (4.71–5.20% of covered raw bytes); this excludes calibration scratch and ancillary files. Waveform payload was 6.14 MB per run. No duplicate sample/unit pairs or out-of-range spike coordinates were found. This short pilot does not establish long-duration scaling or restart equivalence.

## Quality findings

- QC marked 70–72, 84–86, and 88–90 seconds as possible artifact intervals: 6 of 30 seconds. Those intervals remain visible and contribute 5,966/4,417/4,707 assignments for raw 0.65/raw 0.75/smooth3 0.75 respectively.
- No contact received saturation, flatline, dropout, or noisy-channel flags. Median interval/channel noise was 7.97 uV.
- At matched score 0.75, smoothing adds 1,148 assignments (+6.2%) with no increase in active IDs, while short-ISI fraction rises from 3.36% to 3.51%. It is not yet evidence of better accuracy.
- The stricter acceptance threshold explains most of the spike-count reduction relative to raw 0.65. Ground truth is unavailable; the synthetic-data improvement has not been established on FF2.

Next quality work should inspect artifact-associated waveforms and per-unit refractory distributions, validate probe geometry, and compare waveform stability and independent Kilosort agreement on longer, separated intervals. Do not promote either route to the production baseline on this evidence.

`scripts/pilot_ff2_real.py` creates a new output root and runs calibration plus all three conditions. `scripts/summarize_ff2_pilot.py` writes diagnostics.json with per-unit counts and short-ISI counts. The initial pilot driver passed a Session object to the path-taking API; this was fixed before dense sorting and completed calibration was reused. No sorter algorithm or defaults changed.
