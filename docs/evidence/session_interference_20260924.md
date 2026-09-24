# Interference-aware overlap scheduling, 2026-09-24

## Finding and correction

The strict CUDA scheduler deferred every pair sharing any contact within a
61-sample waveform window. The prior event-level trace showed a correctly
detected and confidently fitted spike blocked by templates with weak shifted
waveform coupling. This is a scheduling failure, not a missing detection.

The optional `--overlap-policy interference` accepts new overlapping fits
only with score >= 0.8 and margin >= 0.1, and bounds cumulative projected
amplitude interference to 1% for both the new fit and every affected prior
fit. Noise-weighted full-waveform shifted dots use physical channel overlap.
Previously admitted low-confidence fits do not veto stronger candidates
when this budget passes. Refractory and ordinary fitting gates remain.

A per-core 32,768-entry cache and 32-color subtraction cap bound additional
resources. Cache/color exhaustion defers fits. Separate CUDA launches for
geometrically overlapping fits preserve deterministic subtraction; within
each launch writes are disjoint. Scheduling runs on CPU; dense scoring,
detection and subtraction remain CUDA. This change does not add a new
Kilosort learning stage or use ground truth in sorting.

## Fixed-bank comparison

Same 600-second, 250-unit recording; same 319-template frozen seed bank
`session_seeds_ks60_20260924_01/seeds.npz`; three residual passes, two-second
cores and ten-second shards. The existing seed bank retains its previous
Kilosort-derived provenance. Ground truth is used only for evaluation.
The 60–90 second interval was used for development; 180–210 seconds was
evaluated after choosing the final rule, without further quality tuning.

| Interval / policy | Units IoU >= 0.8 | True spikes | Precision | Recall | Splits / merges | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| 60–90 s strict | 151 | 44,485 | 75.2173% | 86.0278% | 60 / 66 | 31.502 |
| 60–90 s interference | 152 | 44,542 | 75.2551% | 86.1381% | 60 / 68 | 34.073 |
| 180–210 s strict | 154 | 43,871 | 75.1744% | 85.9373% | 59 / 71 | 31.395 |
| 180–210 s interference | 153 | 43,911 | 75.2029% | 86.0157% | 59 / 69 | 32.718 |

The exact traced GT spike at sample 2,013,593 is now emitted at that sample
with local unit 97, pass 1, score 0.81875366, amplitude 0.8899467 and residual
gain 673.99146. In validation, GT unit 122 moves from IoU 0.8022599 to
0.7977528, accounting for the one-unit recovery loss. The result is mixed;
an aggregate true-spike gain does not establish improved unit recovery.

Measured total VRAM was unchanged within each comparison: 1,330,184,192
bytes in development and 1,328,087,040 in validation. GPU pool peaks were
116,387,840 and 116,100,608 bytes respectively. Final experimental process
peak working sets were about 1.605 GB. Each run read 811,008,000 source
bytes for 737,280,000 evaluated raw bytes; derived output was about 4.99%
and 4.90% respectively. This is not a long-duration memory validation.

The new policy took 8.2% and 4.2% longer in these single profiled runs.
These are not repeated cold/warm performance estimates. It is an optional
quality experiment, not a speed improvement. Kilosort4 recovered 227 and
231 units respectively on these same evaluated intervals; the replacement
quality gate remains unmet. Keep strict scheduling as default.

## Validation and artifacts

All 185 tests pass with `scripts/test.ps1`, including cumulative budget
accounting, amplitude ratios, saturation fallback, subtraction group limits,
CUDA repeatability, and exact checkpoint/resume for both policies.
The final code adds a scheduler-version field to the run configuration;
benchmark artifacts below predate that metadata-only addition. Start a new
directory rather than resuming these exploratory outputs.

Artifacts under `F:/sortingDevelopment`:

- `session_fused3_20260924_01`: development strict control.
- `session_interference_candidate_guard_20260924_01`: final development.
- `session_strict_validation_20260924_01`: separate strict control.
- `session_interference_validation_20260924_01`: final separate validation.

Each contains `quality.json`, `run_report.json`, `profile.pstats` and immutable
session shards. Earlier `session_interference_20260924_01` had no confidence
guard and reduced precision; `session_interference_guarded_20260924_01`
required confidence of prior fits too and still missed the focused spike.
Neither earlier variant is the implemented final rule.

The next major quality target remains template competition and representation,
not simply relaxing thresholds. This correction alone does not explain or
close the large missing-unit gap, and no 100 TB capability is established.
