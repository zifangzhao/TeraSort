# Guarded low-threshold detection rescue, 2026-09-24

## Outcome

Appending one guarded low-threshold CUDA residual pass improves end-to-end
spike recall and recovered-unit counts on two 30-second intervals, with a
small precision increase. It remains optional: these intervals come from
one recording and do not establish the multi-dataset replacement gate.

| Interval / method | Units IoU >= 0.8 | True-positive spikes | Precision | Recall | Splits / merges |
|---|---:|---:|---:|---:|---:|
| 60–90 s baseline | 151 | 44,485 | 75.2173% | 86.0278% | 60 / 66 |
| 60–90 s rescue 3.5 | 161 | 45,272 | 75.3228% | 87.5498% | 60 / 67 |
| 180–210 s baseline | 154 | 43,871 | 75.1744% | 85.9373% | 59 / 71 |
| 180–210 s extra-pass control 4.5 | 155 | 43,905 | 75.1502% | 86.0039% | 59 / 71 |
| 180–210 s rescue 3.5 | 160 | 44,667 | 75.3035% | 87.4966% | 59 / 69 |

The matched extra-pass control uses the same acceptance gates as rescue,
but retains the 4.5 candidate threshold. Rescue 3.5 yields five more
recovered units and 762 more true-positive spikes than this control.
Thus the improvement is not explained solely by one more residual pass.

All 59,142 and 58,359 original assignments are preserved exactly in the
two frozen-bank comparisons: source time, contact, unit, score, runner-up,
amplitude, residual gain and pass index. Rescue adds 962 and 957 assignments.
Of those, 910 in each interval reference candidate records whose first
observed SNR is below 4.5. This is descriptive evidence, not standalone
detector precision: candidates can recur across residual passes and nearby
candidate peaks need not originate from a particular GT neuron.

Collision spike matches improve from 39,049 to 39,733 in development and
38,330 to 39,005 in validation. Kilosort4's retained same-input reference
recovers 227 and 231 units respectively. The large quality gap remains.

## Why a separate rescue stage

The earlier failure trace found true spikes below the primary 4.5 SNR
threshold. Initial global-floor trials on 60–90 seconds gave:

| Primary floor | Recovered units | Precision | Recall |
|---|---:|---:|---:|
| 4.0 | 162 | 72.4158% | 89.6732% |
| 3.5 | 159 | 69.3653% | 90.9476% |

These global trials also changed residual-gain acceptance because existing
code uses `floor_snr ** 2` for that gate. They are exploratory coupled
changes, not isolated detector ablations. Their precision regression rules
them out as a straightforward default change.

The new `--rescue-floor-snr` control appends a single pass after primary
fitting. It changes the final candidate threshold independently while
retaining the primary gain floor, adds score >= 0.85 and margin >= 0.1,
and preserves amplitude/refractory checks. Larger user-specified score or
margin floors remain in effect. It also executes after primary detection
returns no events or primary fitting accepts no events. A value of 4.5
is accepted to support an extra-pass control; the default is disabled.

Detection, scoring and subtraction reuse the existing direct CUDA kernels.
There is no PCA, new Kilosort learning, or GT-dependent selection in this
change. No candidate truncation is introduced: existing per-core candidate
and pair limits raise explicit errors if exceeded. All candidates retain
metadata, with selective waveform storage as before. With frozen models,
primary assignments are preserved. Adaptation/enrollment may change later
cores and is not covered by that preservation claim.

## Protocol and resources

Use the existing 600-second, 250-unit, 384-channel, 32 kHz recording, with
the same frozen 319-template seed bank in all runs. The bank retains its
earlier Kilosort-derived provenance. Primary settings remain three residual
passes, SNR 4.5, center score 0.65, relative margin 0.03, strict overlap
scheduling, two-second cores, ten-second shards and novelty off.
The final rescue rule was chosen before evaluation on 180–210 seconds;
this interval has been used in earlier project experiments, so it is not
an untouched dataset. No rule tuning followed the validation results.

| Metric | Development baseline → rescue | Validation baseline → rescue |
|---|---:|---:|
| Profiled sorting wall seconds | 31.502 → 35.939 | 32.013 → 34.813 |
| Candidate metadata rows | 467,912 → 677,427 | 462,947 → 671,912 |
| Derived output bytes | 36,750,565 → 41,107,813 | 36,159,172 → 40,630,312 |
| Measured total VRAM bytes | 1,330,184,192 → same | 1,328,087,040 → same |
| GPU pool peak bytes | 116,387,840 → same | 116,100,608 → same |

Rescue increases wall time by roughly 9–14% in these single profiled runs;
they are not repeated cold/warm timing estimates. The extra-pass control
took 30.633 seconds, illustrating timing variability. Each run reads the
same 811,008,000 raw bytes. Rescue process peak working sets are about
1.605 and 1.606 GB. Derived/source ratios rise to 5.576% and 5.511%; the
5% waveform budget is not a cap on total output. Candidate metadata grows
roughly 45%, which matters for long recordings despite unchanged VRAM.
No long-duration or 100 TB claim follows from this experiment.

## Artifacts and validation

Under `F:/sortingDevelopment`:

- `session_detector_floor4.0_20260924_01`, `session_detector_floor3.5_20260924_01`: initial coupled global-floor trials.
- `session_detector_rescue60_20260924_01`: development rescue.
- `session_detector_rescue180_20260924_01`: separate-interval rescue.
- `session_detector_rescue_control180_20260924_01`: same-gates, original-threshold fourth-pass control.
- `session_fused3_20260924_01`: development baseline.
- `session_refinement_eval_r0_20260924_01`: recent validation baseline using the original bank.

Run reports, quality reports, profiles and immutable assignment shards are
retained. The source recording and seed bank were not modified.
All **195 tests pass** with `scripts/test.ps1`. New regression coverage checks subthreshold waveform recovery, recovery
after an empty primary stage, exact primary-assignment preservation,
repeatability, retention of the primary residual-gain gate, invalid thresholds,
and interrupted/resumed CUDA runs with both overlap policies. Run configuration
includes rescue settings and an algorithm version; older run directories
must not be resumed across a configuration change.

Use `terasort session-sort ... --rescue-floor-snr 3.5` to opt in.
Further work should evaluate drift and collision datasets, and then profile
ways to reject weak noise proposals cheaply without losing this recall gain.
