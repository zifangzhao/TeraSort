# Bounded accuracy search, 2026-09-24

## Outcome

Completed **44 sorting evaluations: 35 development configurations and nine
validation runs**, with no execution failures. All **217 regression tests
pass**. The strongest tested combination is smooth3 candidate detection with
a 0.75 fitting score floor and the existing 3.5-SNR rescue pass. It improves
precision and unit recovery with a small recall tradeoff, and remains opt-in.

On fresh seconds 480–510, reference versus this combination changes recovered
units from **161 to 173**, precision **75.470% to 81.471%**, recall **87.486%
to 87.156%**, and F1 **81.035% to 84.218%**. Compared with raw detection at
the same 0.75 score threshold, smoothing adds seven recovered units and
799 true-positive spikes, improving F1 from 83.894% to 84.218%. This isolates
a detection benefit beyond the stricter acceptance threshold alone.

Measured sorting time rises from **31.099 to 35.732 seconds** (about 15%),
total VRAM from **1,328,087,040 to 1,758,003,200 bytes**, and GPU pool peak
from 116,083,200 to 540,663,296 bytes. Process peak working set remains about
1.60 GB. Candidate rows rise from 667,571 to 809,167; compressed derived
output is 40,174,808 versus 39,594,667 bytes because cache selection and
compression also vary. Both read 811,008,000 source bytes. These are single
unprofiled runs, not repeated cold/warm estimates.

PCA accepted one split (parent 3, child 319), with 44 training and 22 internal
validation examples, 10.6% held-out waveform-error reduction, and child
similarity 0.9002. The bank expanded to 320 templates. Its independent
420–450 second evaluation improves recovered units 160 to 162, precision
75.337% to 75.458%, and recall 87.274% to 87.397%. This is a small positive
pilot, not evidence that arbitrary PCA splitting is safe. Learning retained
7,820 training and 5,272 validation snippets, took 4.434 seconds, and peaked
at a 717,840,384-byte working set using the existing cache.

Local refitting changed 13 identities, 71 timings and 74 amplitudes after
one round; two rounds changed 13 identities, 75 timings and 78 amplitudes.
Both added only five true-positive spikes and no recovered units in development.
Aggressive rank-one denoising and 90%-energy masking substantially reduced
quality. Extra compute and reduced representation size were not generally
beneficial. All rejected alternatives remain in the tables below.

The historical Kilosort4 reference still recovers **229 units** at 480–510
seconds. This search has not met the replacement quality gate, covered
additional independent drift/collision datasets, or established long-file
scale. No production defaults were changed.

To reproduce the promising experimental configuration with a suitable manifest:

```powershell
terasort session-sort --manifest C:\data\session.json --output-root F:\sorts\smooth-quality-1 --freeze-templates --rescue-floor-snr 3.5 --detector-mode smooth3 --score-floor 0.75
```

Full machine-readable and rendered results are in
`F:/sortingDevelopment/session_accuracy_summary_20260924_02/results.json`,
`summary.md`, and `precision_recall.png`. Other retained roots are
`session_accuracy_extensions_20260924_01`, `session_accuracy_combinations_20260924_01`,
and `session_pca_splits_20260924_01`, all under `F:/sortingDevelopment`.

![Precision–recall comparison](F:/sortingDevelopment/session_accuracy_summary_20260924_02/precision_recall.png)

This experiment screens practical alternatives rather than claiming to try
every possible spike-sorting algorithm. All recording and historical Kilosort
inputs are read-only; each trial writes to a new directory. Production defaults
remain unchanged. The dataset is the existing 600-second, 250-unit, 384-channel
recording at 32 kHz. The original 319-template bank retains its earlier
Kilosort-derived provenance; no new Kilosort learning is performed.

## Design

The initial screen declares 27 configurations before running: fit score floors,
ambiguity margins, center scoring windows, rescue floors, overlap scheduling,
the refined bank, rank-one/rank-two SVD template denoising, 90%/97% energy
spatial masks, and several combinations. Template transforms do not use GT.
All dense evaluation passes use every core in the evaluated interval.

Development is seconds 60–90. The initial selection rule keeps configurations
within 0.2 percentage points of reference precision with no recall loss,
ranking by recovered units then F1. Two selected alternatives and the reference
are evaluated on seconds 420–450. This interval is excluded from the screen
and from the earlier refinement/PCA learning windows. It belongs to the same
recording, so it is not an independent dataset or publication quality gate.

The follow-up script separately selects the best development F1 point, without
the no-recall-loss restriction, to expose precision/recall tradeoffs. It also
tests score floors 0.8/0.85 and one/two local-refit rounds, then validates the
best new family by development F1 if it improves over reference. Selection
code uses development results only; validation outcomes are not fed back
into model or threshold selection.

After inspecting single-factor development results, a final declared test
combines smooth3 detection with score floors 0.70 and 0.75. The best
development F1 combination advances only if it beats raw detection at the
same score floor. It is then compared with the reference and a matched-score
raw detector on fresh seconds 480–510, excluded from both development and
PCA/refinement learning windows. The PCA bank also receives its own 420–450
second check after a small across-metric development improvement. These
follow-ups are retained separately; they are not retroactively included in
the initial screen's selection rule.

The follow-up also tests a [1,2,1]/4 temporal candidate detector. It estimates
filtered noise per residual pass from approximately 4,000 sampled rows,
while leaving raw-noise waveform fitting and residual-gain acceptance intact.
This changes the candidate SNR frame, explicitly recorded in configuration,
and uses extra bounded GPU temporaries. It is not fused or presumed faster.

All reference settings use three primary passes, one rescue at SNR 3.5,
primary SNR 4.5, score floor 0.65, margin 0.03, half-width eight, timing
radius two, strict scheduling, frozen templates and no novelty enrollment.
The refined-bank and SVD/mask arms change the bank; other parameters are
explicitly recorded in each command and run configuration.

## Additional algorithm experiments

The local-refit post-pass revisits at most 64 uncertain fits per core per
round, for up to two rounds. It subtracts accepted neighbors from a local
patch, evaluates competing local templates and amplitudes, and requires
at least 5% improvement over the old fit's actual reconstruction gain,
score >= 0.85 and margin >= 0.1. Refractory and amplitude constraints remain.
It preserves event count and can change unit, timing or amplitude. This is
coordinate refinement with current neighbors, not a global joint optimizer.
Core-edge fits are skipped because halo assignments are not in the output list.

PCA splitting uses actual spike features, not per-template SVD denoising.
It reuses the bounded preprocessed cache at training starts 40,100,220,340,
460,540 seconds and validation starts 44,224,464 seconds, two seconds each.
It retains at most 16,384 snippets per partition, stratified by unit/window.
Moderately confident, amplitude-normalized, isolated accepted waveforms enter
the sample. Three PCA components fit on training data propose two clusters;
both must recur across at least two training and validation windows, improve
held-out waveform error by 10%, meet separation and non-duplicate criteria,
and preserve peak alignment. At most 16 child templates may be appended.
No GT labels enter learning. A zero-split outcome is valid; clusters are not
forced merely to increase unit count.

## Reproducibility and limits

The screen root is `F:/sortingDevelopment/session_accuracy_sweep_20260924_01`.
It contains `plan.json`, transformed seed banks, per-run logs, full session
shards, quality/run reports, `results.jsonl`, selection and completion markers.
The PCA and extended-family outputs use separate new directories recorded
with the results below. All timings are single unprofiled runs, with sorting
time distinct from subprocess end-to-end time including evaluation. They
are not repeated cold/warm speed estimates. Memory reports describe sorting,
not the offline quality evaluator.

Optional width/refit controls were added with their default behavior disabled
or unchanged. Metadata fields and the descriptive scoring-window string were
added during the experiment; they do not change default fitting decisions.
The reference reproduced the earlier rescue result exactly at the metric level.
Updated configurations prevent resuming older run directories silently.

The tests include valid and rejected PCA splits, unsupported partitions,
neighbor-subtracted identity correction, amplitude correction, edge handling,
and deterministic interrupted/resumed runs with local refitting. Broader
drift/collision datasets, rare/late units and long-file memory gates still
require separate validation. A finite search on one recording does not
establish a global optimum or 100 TB capability.

## Complete comparison tables

## 60–90 seconds

| Configuration | Recovered units | Precision % | Recall % | F1 % | Splits / merges |
|---|---:|---:|---:|---:|---:|
| reference | 161 | 75.323 | 87.550 | 80.977 | 60 / 67 |
| score_floor_0.55 | 147 | 69.984 | 88.058 | 77.988 | 69 / 70 |
| score_floor_0.6 | 156 | 72.245 | 87.766 | 79.253 | 65 / 72 |
| score_floor_0.7 | 166 | 78.804 | 87.014 | 82.706 | 51 / 61 |
| score_floor_0.75 | 168 | 82.214 | 85.709 | 83.925 | 43 / 60 |
| min_margin_0 | 165 | 75.596 | 87.828 | 81.255 | 60 / 63 |
| min_margin_0.01 | 162 | 75.448 | 87.751 | 81.136 | 60 / 61 |
| min_margin_0.07 | 154 | 74.952 | 86.639 | 80.373 | 63 / 72 |
| min_margin_0.15 | 149 | 74.447 | 85.289 | 79.500 | 61 / 74 |
| half_width_4 | 156 | 70.321 | 89.406 | 78.724 | 71 / 71 |
| half_width_12 | 165 | 79.654 | 86.339 | 82.862 | 46 / 62 |
| half_width_20 | 160 | 83.833 | 83.993 | 83.913 | 40 / 60 |
| half_width_30 | 157 | 86.227 | 80.828 | 83.440 | 34 / 54 |
| rescue_floor_snr_3.0 | 162 | 75.327 | 87.652 | 81.024 | 60 / 67 |
| rescue_floor_snr_4.0 | 156 | 75.291 | 87.200 | 80.809 | 60 / 67 |
| no_rescue | 151 | 75.217 | 86.028 | 80.260 | 60 / 66 |
| interference | 160 | 75.383 | 87.635 | 81.049 | 60 / 69 |
| refined | 161 | 75.305 | 87.538 | 80.962 | 63 / 67 |
| rank1 | 127 | 68.931 | 84.349 | 75.865 | 69 / 74 |
| rank2 | 158 | 75.525 | 87.267 | 80.973 | 61 / 68 |
| mask90 | 124 | 67.113 | 88.917 | 76.491 | 83 / 90 |
| mask97 | 156 | 73.530 | 88.188 | 80.195 | 65 / 66 |
| rank2_mask97 | 153 | 73.744 | 88.180 | 80.318 | 63 / 74 |
| rank2_score07 | 164 | 78.759 | 86.618 | 82.501 | 54 / 59 |
| mask97_score07 | 162 | 76.965 | 87.749 | 82.004 | 58 / 62 |
| short_score07 | 159 | 72.817 | 89.395 | 80.259 | 68 / 67 |
| wide_score06 | 161 | 76.131 | 86.921 | 81.169 | 57 / 64 |
| local_refit_1 | 161 | 75.331 | 87.559 | 80.986 | 60 / 67 |
| local_refit_2 | 161 | 75.331 | 87.559 | 80.986 | 60 / 67 |
| score_0.8 | 170 | 85.307 | 82.783 | 84.026 | 36 / 48 |
| score_0.85 | 157 | 88.301 | 75.550 | 81.429 | 28 / 48 |
| smooth3 | 165 | 73.481 | 89.594 | 80.741 | 67 / 70 |
| pca_split | 163 | 75.450 | 87.683 | 81.108 | 59 / 67 |
| smooth_score_0.7 | 169 | 77.527 | 88.958 | 82.850 | 55 / 63 |
| smooth_score_0.75 | 172 | 81.404 | 87.380 | 84.286 | 46 / 60 |

## 420–450 seconds

| Configuration | Recovered units | Precision % | Recall % | F1 % | Splits / merges |
|---|---:|---:|---:|---:|---:|
| reference | 160 | 75.337 | 87.274 | 80.867 | 62 / 71 |
| min_margin_0 | 164 | 75.576 | 87.491 | 81.098 | 62 / 67 |
| mask97_score07 | 166 | 77.036 | 87.450 | 81.913 | 56 / 67 |
| best_f1_score_floor_0.75 | 172 | 81.939 | 85.014 | 83.448 | 43 / 64 |
| score_0.8 | 170 | 85.308 | 82.245 | 83.748 | 36 / 49 |
| pca_validation | 162 | 75.458 | 87.397 | 80.990 | 61 / 71 |

## 480–510 seconds

| Configuration | Recovered units | Precision % | Recall % | F1 % | Splits / merges |
|---|---:|---:|---:|---:|---:|
| fresh_reference | 161 | 75.470 | 87.486 | 81.035 | 64 / 67 |
| fresh_matched_score | 166 | 82.268 | 85.585 | 83.894 | 44 / 66 |
| fresh_smooth | 173 | 81.471 | 87.156 | 84.218 | 47 / 66 |
