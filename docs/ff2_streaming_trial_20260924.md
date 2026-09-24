# TeraSort streaming trial on FF2 D82 Linear A

Date: 2026-09-24. This is an experimental first pass, not a validated spike sort.
The source was the checksum-verified local staging copy of the read-only
FF2 D82 Linear A Intan recording (128 channels, 20 kHz, 29 min 21.2 s).
The first 30 s supplied calibration waveforms; samples 600,000–1,199,999
(30–60 s) were held out for assignment. Prior Kilosort4 output supplied
labels for calibration and a pseudo-reference for evaluation. It is not
ground truth.

## Method

Per-shank median reference, a zero-phase third-order 300–6000 Hz Butterworth
filter, and a fixed first-5-s MAD noise estimate preceded GPU absolute-peak
detection at SNR 4.5. SCB 0.2 cached 61-sample, up-to-16-channel INT16
waveforms at 0.5 microvolt per count. Twenty-four Kilosort "good" units
had at least 20 isolated calibration waveforms in the first 30 s and became
the initial local template bank. The held-out interval was processed using
the direct CUDA matcher with default score and update gates. No raw data
or prior sort was intentionally modified or overwritten.

## Measured result

| Measure | Result |
| --- | ---: |
| Candidate bank, first 60 s | 444,106 rows; 387,849,833 bytes; 36.01 s build wall time |
| Candidate rows, held-out 30 s | 230,525 |
| Consolidated events, held-out 30 s | 146,767 |
| Assigned events | 80 (0.0545% of consolidated events) |
| Rolling-template waveform updates | 0 |
| CUDA held-out sort wall time | 16.21 s |
| Assignment shard size | 2,706,627 bytes |
| Kilosort spikes from the 24 seeded units, held-out 30 s | 9,873 |
| Assignments agreeing with the seeded Kilosort unit within 5 samples | 36/80 (45.0%); 36/9,873 (0.365%) |

Channel-constrained detection is materially better than a timing-only check:
7,222/9,873 seeded-unit Kilosort spikes (73.2%) had a consolidated event on
that unit's template anchor channel within two samples. A 1,000-sample
circular time shift gave 206/9,873 (2.1%). Allowing any contact in that
unit's local template patch gave 7,902/9,873 (80.0%) versus 633/9,873
(6.4%) for the shifted control. These are agreement measures against
Kilosort, not ground-truth detection sensitivity.

On the shared first held-out second, CPU and CUDA each produced 12,733
events from 21,165 candidate rows. Every event and unit ID agreed;
the maximum absolute score difference was 2.38e-7. The CPU reference
wall time for that one second was 2.73 s. The 30-s GPU time and 1-s CPU
time are different intervals, so they are not a fair speedup ratio.
Reading, decoding, and grouping the held-out SCB without matching took
6.05 s, about 37% of the complete CUDA run's wall time.

The 61-sample cosine score penalizes spikes with nearby activity. In one
diagnostic on seeded Kilosort cluster 66, 40 same-anchor labeled events
and 188 local control events were compared. A 17-sample central temporal
window raised median positive cosine from 0.431 to 0.705 and the
positive-versus-control AUC from 0.910 to 0.954. This was a one-unit
read-only diagnostic; no threshold or algorithm was changed from these
numbers. The current 0.88 assignment gate yielded 80 assignments and
no update waveforms. The next implementation test should use a masked
central waveform score and calibrate its thresholds across many units
with held-out labels and collision controls.

The earlier Kilosort4.1.7 trial reported 26.61 s of Kilosort processing
for its first 60-s smoke test and 806.72 s for the full 29-min recording.
Those runs use a different output and preprocessing pipeline. This
experimental streaming pass has not demonstrated a speed or sorting-quality
advantage over Kilosort4. GPU memory was not sampled in this trial.

## Artifacts and reproducibility limit

New outputs are in F:\sortingDevelopment\terasort_ff2_streaming_20260924:
the SCB candidate bank, 24 calibration templates, seed-to-Kilosort cluster
map, complete 30-s CUDA assignment shard, one-second CPU reference shard,
and build_summary.json. The original network recording was only read
through the local staging copy.

During later validation, the local C: staging copy, prior Kilosort comparison
directory, and Python 3.11 runtime disappeared from the machine. The trial
outputs on F: remained complete. No delete or overwrite command was run
by this trial. Repeating the experiment will require restoring the local
source/reference and Python environment; the source path recorded in
build_summary.json no longer resolves.
