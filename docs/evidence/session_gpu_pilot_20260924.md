# First session-sort GPU pilot, 2026-09-24

Environment repair succeeded. The complete regression suite passed 159 tests,
including CUDA proposal scores against the CPU reference, iterative recovery
of spikes with overlapping contact support, and exact candidate/spike/template
equivalence after an interrupted CUDA shard is resumed.

The public-data pilot completed but the experimental sorter has a substantial
quality deficit. It must remain behind the Kilosort4 production route.

## Protocol and retained results

The source was the existing 600-second, 384-channel, 250-unit recording at
32 kHz, encoded as INT16 with 0.050000000745 µV/count. Probe coordinates were
checked against the retained Kilosort output, and its shank array was reused.
Only samples 0–960,000 (30 seconds) were densely sorted. Calibration used
three 60-second windows near the start, middle and end of the full recording.
Ground-truth labels were used only for evaluation, not learning.

Command:

```powershell
.venv\Scripts\python.exe -u scripts\benchmark_session_recording.py `
  --dataset-root F:\sortingDevelopment\data `
  --kilosort-dir F:\sortingDevelopment\cublas_sorting_v1\jobs\deep_tiled_control\attempt_001\kilosort `
  --output-root F:\sortingDevelopment\session_gpu_pilot_20260924_01 `
  --seconds 30 --profile
```

Results are retained beneath
`F:/sortingDevelopment/session_gpu_pilot_20260924_01`:
`manifest.json`, `run_report.json`, `quality.json`, `progress.jsonl`,
`profile.pstats`, and immutable session shards. Existing inputs and Kilosort
results were not overwritten.

The comparison uses a historical Kilosort 4.1.7 `deep_tiled` result trained
on the complete 600 seconds, cropped to the same 30-second evaluation span.
It is not a fresh paired speed benchmark or the full three-dataset gate.

## Quality

All output units are included in the precision/recall calculation; the
Kilosort result is not filtered to its “good” clusters. Spike matching uses
the custom session evaluator with 0.4 ms tolerance and one-to-one unit
matching. Independent SpikeInterface validation remains outstanding.

| Metric, first 30 seconds | Session sorter | Retained Kilosort4 |
| --- | ---: | ---: |
| Ground-truth units recovered at IoU ≥ 0.8 | 67 / 250 | 229 / 250 |
| Spike precision | 50.72% | 69.87% |
| Spike recall | 66.21% | 98.24% |
| Output units with spikes | 783 | 517 |
| Ground-truth units flagged as split | 110 | 83 |
| Output units flagged as merged | 273 | 72 |

The local learner created 821 templates from 24,303 cached calibration
waveforms. This excess of provisional templates and the split/merge results
make template learning and assignment quality the next priority. They do not
establish a specific causal defect yet. No rare/late subgroups were represented
under the evaluator's definitions in this short span. The current collision
counter measures cross-unit temporal coincidence across the probe; spatially
local collision validation is still required.

## Time, memory, storage

This run enabled cProfile and overlapped a short GPU smoke test while the
pilot was in CPU calibration. Use these numbers to identify bottlenecks,
not as fair production speed figures.

- Total profiled sorter wall time: 396.17 seconds.
- Calibration: 343.90 seconds; CPU `numpy_detect`: 192.37 seconds.
- Dense sorting plus finalization: approximately 52.3 seconds for 30 seconds.
- CUDA matcher orchestration: 23.93 seconds total; host-side score-choice
  construction alone accounted for 15.24 seconds in the profile.
- Windows process peak working set: 1,650,429,952 bytes (1.54 GiB). The
  less frequent per-core RSS samples reported a lower peak of 1.00 GB.
- Peak CuPy reserved pool: 737,281,536 bytes (703 MiB).
- Sampled total-device VRAM used: 1,959,329,792 bytes (includes other apps).
- Dense source bytes covered: 737,280,000; successful reads including halos:
  808,550,400. These counters currently exclude calibration reads.
- Assignment shard size: 37,709,119 bytes, 5.11% of dense raw bytes.
- Waveform payload budget charged: 36,840,960 bytes, within 5% of dense raw
  bytes. The shard is compressed; calibration/progress files are additional.

The GPU 1×/10×/100× repeated-source smoke test is retained at
`F:/sortingDevelopment/session_gpu_scale_20260924_01`. Across those tiny
fixtures, the CuPy pool remained 193,024 bytes; sampled total VRAM remained
1,212,743,680 bytes. RSS increased from 336,461,824 to 338,665,472 bytes.
This does not establish realistic long-recording or network scale.

During testing, calibration reservoir entries were changed to own their
waveform arrays. Previously, keeping a snippet view retained its whole source
batch and could inflate memory across calibration windows. Whole-recording
motion correction, novelty retraining, real long runs, and the full quality
gate remain outstanding.
