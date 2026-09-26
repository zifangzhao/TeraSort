# Adaptive template timing search, 2026-09-26

## Result

An optional pilot-based per-unit timing radius improved held-out quality on the
public 250-unit ground-truth recording without measurable end-to-end slowdown.
On the 480–510 s interval, it recovered 174/250 units at IoU >= 0.8 and matched
44,089 spikes. Fixed radius 2 recovered 170 units and matched 43,903 spikes;
fixed radius 3 recovered 172 and matched 43,977. Adaptive precision/recall/F1
were 81.974% / 86.706% / 84.274%, compared with 81.901% / 86.340% / 84.062%
for radius 2. This is a promising single-dataset experiment, not a replacement
quality result. Kilosort4 remains the production baseline.

## Method

The opt-in `--adaptive-shift-radius` mode runs a 30-second pilot with the
configured radius plus one sample. For each template it counts accepted fits
and fits at the expanded search boundary. Units with at least 20 pilot fits
and at least 5% boundary fits retain the wider radius; other units return to
the base radius. The CUDA scorer reads a per-unit radius, preserving one fused
scoring path. Pilot progress and the radius vector are checkpointed in every
completed shard so resume restores the same policy. The mode currently
requires CUDA, `shift_radius < 8`, and `refit_rounds=0`; it remains opt-in.

The experiment sorted samples 420–510 s in three 30-second shards using
smooth3 detection, frozen templates, strict overlap handling, three residual
passes, base radius 2, and otherwise matched scoring settings. The final
480–510 s output was evaluated against the same ground truth with a 0.4 ms
spike tolerance. The fixed-radius-2 control covered the same 90-second span;
the fixed-radius-3 comparison covered the same final 30-second interval.

| Setting, 480–510 s | Units recovered | TP spikes | Precision | Recall | F1 | Splits / merges |
|---|---:|---:|---:|---:|---:|---:|
| Fixed radius 2 | 170 | 43,903 | 81.901% | 86.340% | 84.062% | 44 / 66 |
| Fixed radius 3 | 172 | 43,977 | 81.799% | 86.485% | 84.077% | 43 / 65 |
| Adaptive, base 2 | 174 | 44,089 | 81.974% | 86.706% | 84.274% | 43 / 64 |

The adaptive pilot selected radius 3 for 199 of 319 templates. It processed
the full 90-second interval in 82.90 s, compared with 82.95 s for the fixed-2
run. Per-shard sort times were 27.46, 27.80, and 27.47 s. Reported peak total
device VRAM was 1.37 GB; peak process RSS was 0.97 GB. The small timing
difference is within run-to-run noise and is not evidence of a speedup.

Across the entire 420–510 s evaluation span, adaptive recovered 177 units and
132,699 spikes at 81.912% precision, 86.557% recall, and 84.170% F1; fixed
radius 2 recovered 173 units and 132,086 spikes at 81.797% precision, 86.157%
recall, and 83.920% F1. The first 30-second pilot interval itself was slightly
worse than fixed radius 2 (170 versus 174 recovered units; F1 83.781% versus
83.843%). The subsequent two intervals improved, which is consistent with a
useful but delayed calibration effect. Since this is a single session, the
aggregate result still needs independent replication.

## Validation and limits

All 255 repository tests passed, including per-unit CUDA radius scoring,
validation of invalid radius vectors, checkpoint restoration, and exact
interrupted-versus-uninterrupted event equality with adaptive mode enabled.

This is one recording and one held-out interval. The pilot learns boundary
behavior from earlier activity in that same recording, so the result is an
online policy evaluation, not an independent-dataset test. It improves over
the TeraSort fixed-radius controls but does not close the broader Kilosort4
unit-recovery gap or satisfy the multi-dataset replacement gate. Do not enable
it by default until it is replicated across drift, collision, and quiet-unit
benchmarks, with time-matched Kilosort4 comparisons.

## Later-interval replication

The same matched fixed-radius-2 and adaptive configurations were run on the
later 510–600 s interval from the same 600-second recording. Adaptive selected
radius 3 for 200/319 templates after its 30-second pilot.

| Setting, 510–600 s | Units recovered | TP spikes | Precision | Recall | F1 | Splits / merges | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed radius 2 | 174 | 133,013 | 81.820% | 86.423% | 84.059% | 46 / 58 | 84.30 |
| Adaptive, base 2 | 175 | 133,338 | 81.758% | 86.634% | 84.125% | 44 / 58 | 83.41 |

The later interval adds one recovered unit and 325 true positives; F1 increases
by 0.066 percentage points while measured runtime falls by 1.1%. Peak total
VRAM is 1.41 GB for both runs. The direction matches the 420–510 s result,
though the later-window quality gain is small and each result is a single run.
This is temporal replication within one recording, not independent-dataset
validation. The retained Kilosort4 comparison files cover only samples
16–1,919,960, so they do not provide a Kilosort4 result for this interval.

Two follow-up settings did not improve the speed/quality tradeoff. Adding the
existing PCA split bank to adaptive radius 2 increased 420–510 s recovery from
177 to 179 units and F1 from 84.170% to 84.310%, but runtime rose from 82.90 s
to 83.48 s. Reducing the base radius to 1 with that bank recovered 177 units
at 84.268% F1 and took 84.37 s. Both remain unselected. A short-window-first
CUDA fitting-kernel experiment preserved aggregate quality on 420–510 s but
did not reduce end-to-end time (83.56 s versus 82.90 s) and changed some
floating-point fit fields, so it was discarded.

Raising the pilot boundary-fraction criterion from 5% to 10% expanded 135
templates instead of 199 on 420–510 s. It recovered the same 177 units but
had slightly lower F1 (84.145% versus 84.170%) and took 99.99 s versus 82.90 s;
the final shard alone took 42.37 s. This single unprofiled run does not isolate
the cause of the slowdown, but it provides no reason to select the 10%
setting. Keep 5% as the default. The threshold is exposed as an experimental
CLI parameter for controlled future tuning.

A diagnostic per-contact timing-offset fit raised the score above 0.75 for
83 of 231 sampled, credible shape-rejection misses. It also showed high
scores on many sampled targets whose misses were ultimately caused by a
different template winning or overlap deferral. Because this is a selected
miss sample and the alternate fit has not been integrated with residual
subtraction or evaluated against all negative candidates, it is not yet safe
to add as an assignment rule.

Outputs are under
`C:\Frank\Code\PainProject\outputs\smooth3_adaptive_shift_420_510_20260926_01`.
The later-window control and adaptive outputs are
`C:\Frank\Code\PainProject\outputs\smooth3_fixed2_510_600_20260926_01` and
`C:\Frank\Code\PainProject\outputs\smooth3_adaptive_510_600_20260926_01`.
PCA-bank experiments are in `smooth3_pca_adaptive_420_510_20260926_01` and
`smooth3_pca_adaptive_r1_420_510_20260926_01` under the same `outputs` folder.
The rejected 10% criterion run is in
`C:\Frank\Code\PainProject\outputs\smooth3_adaptive10_420_510_20260926_01`.
