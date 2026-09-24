# Template fitting refinement — 2026-09-24

## Problems and changes

The original 821-template calibration bank included 175 off-center peaks and
322 templates whose strongest contact differed from their anchor. Many
templates were highly correlated. Fitting the central 17 samples while
subtracting all 61 could increase residual error. Selection also let an
amplitude-ineligible winner hide a valid alternative.

The CPU reference and direct CUDA matcher now fit full-waveform amplitudes,
weight contacts by inverse noise variance, search +/-2 samples, and rank
eligible templates by residual reduction. Relative residual-gain margins
preserve ambiguous events. Seed canonicalization preserves unit IDs and
physical contact indices. A 0.5 ms exclusion protects against repeat
assignment of one unit; detection coordinates remain available when fitted
times shift. Segmented NumPy reductions replace per-pair Python sorting.
No whole-recording tensor or dense all-template matrix was introduced.

Fitting alone was insufficient. The optional `learn_session_seeds.py` script
uses pinned Kilosort4 identities from a bounded raw preview, then rebuilds
median, rank-three local waveforms in the session preprocessing frame.
It does not use ground truth or the whole-recording Kilosort output to learn.

## Ground-truth pilot

Input: existing 600-second, 384-channel, 32 kHz, 250-unit INT16 dataset at
`F:\sortingDevelopment\data`. RTX 5060 Ti 16 GB; project Python 3.11.16,
CuPy 14.2.0, Torch 2.10.0+cu128, Kilosort 4.1.7.
Results below evaluate only 0–30 seconds, not the entire dataset.

| Route | Units at IoU >= 0.8 | Spike precision | Spike recall |
|---|---:|---:|---:|
| Original session pilot | 67/250 | 50.72% | 66.21% |
| Refined fitting, original bank | 73/250 | 52.69% | 63.83% |
| Refined fitting, bounded K4 seeds | 157/250 | 75.43% | 86.00% |
| Historical full-recording K4 reference | 229/250 | 69.87% | 98.24% |

Seed learning used 60 seconds total: 0–20, approximately 300–320, and
580–600 seconds. It produced 319 seeds in 96.64 seconds including preview
copy and session-frame reconstruction; K4 reported 44.06 seconds internally.
The retained preview occupies 1,474,560,000 bytes. The 0–30 second diagnostic
partly overlaps calibration and must not be treated as held-out validation.

On held-out 60–90 seconds, the refined route recovered **151/250** units,
with **75.22% precision and 86.03% recall**. The historical K4 reference
recovered **227/250**, with 69.92% precision and 98.31% recall. Thresholds
and seeds were unchanged for this held-out run. Precision uses all output
units, including noise clusters, not K4 good-only units. This custom
evaluator still needs an independent SpikeInterface comparison. Temporal
coincidence counts are not spatially validated collision benchmarks.

## Timing and memory

Fitting-only rerun with the original bank took 34.10 seconds versus roughly
52.3 seconds for the original dense stage. Both are profiled pilot timings;
original calibration time is excluded. The new-bank first interval took
33.86 seconds, but briefly overlapped the test suite and is not a clean
speed measurement. The separate held-out run took 31.01 seconds including
shard writing, excluding seed learning and quality evaluation.

Held-out Windows peak working set was 1,640,685,568 bytes (1.53 GiB),
sampled total device VRAM peaked at 1,648,951,296 bytes (1.54 GiB), and the
sampled CuPy pool peaked at 432,539,136 bytes. Sampled device telemetry can
miss brief peaks. It read 811,008,000 source bytes including halos to cover
737,280,000 raw bytes. The shard was 36,662,193 bytes (4.97% of raw).
These are short-run measurements, not proof of flat long-session memory,
network throughput, or superiority in speed over Kilosort4.

## Reproducible artifacts

Validation: `scripts/test.ps1` completed with **163 passed**. Regressions
cover misleading central-only fits, shift recovery, physical channel
preservation, amplitude-ineligible winners, CPU/CUDA score agreement,
overlap recovery, and exact events/templates after interrupted resume.

All output directories are new; source recording and prior results remain
unchanged under `F:\sortingDevelopment`:

- `session_fit_v2_20260924_01`: fitting-only ablation, original seeds.
- `session_seeds_ks60_20260924_01`: calibration preview, K4 output, seeds and report.
- `session_fit_ks60_20260924_01`: first interval with learned seeds.
- `session_fit_ks60_heldout_20260924_01`: held-out interval.

Each evaluation run contains `manifest.json`, `quality.json`,
`run_report.json`, `profile.pstats`, progress and immutable session shards.
Use `scripts/benchmark_session_recording.py` with `--seed-templates`,
`--seconds 30`, and `--start-seconds 60` to reproduce the held-out procedure
into a new output root.

The quality gate still fails. Next work should address missed units,
template contamination and collision fitting, followed by the full
600-second evaluation and independent drift/hybrid datasets. The K4
production route remains the baseline; no multiday or 100 TB quality claim
is established by these pilots.
