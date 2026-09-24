# Extra computation versus sorting accuracy, 2026-09-24

## Conclusion

More residual passes and a wider timing search do not improve overall
quality in this pilot. Keep the previous three-primary/one-rescue, radius-two
configuration as the preferred rescue option. The new controls remain
available for experiments, not as a validated accuracy preset.

## Controlled settings

Same existing 600-second, 250-unit recording and frozen 319-template seed
bank as the rescue experiment. No learning, new Kilosort execution, ground
truth in fitting, or threshold tuning. All settings use primary SNR 4.5,
rescue SNR 3.5, strict overlap scheduling and unchanged fit gates. Later
primary passes retain the existing score floor >= 0.85; rescue requires
score >= 0.85, margin >= 0.1 and the primary residual-gain floor.

- Reference: three primary passes, one rescue pass, timing radius two samples.
- Extra rescue: three primary, three rescue, radius two.
- Wider timing: three primary, one rescue, radius four.
- Combined: six primary, three rescue, radius four.

At 32 kHz, radius two is +/-62.5 microseconds; radius four is +/-125
microseconds. More passes stop early when no additional fits are accepted.

## Development interval, 60–90 seconds

| Setting | Recovered units IoU >= 0.8 | True-positive spikes | Precision | Recall | Splits / merges | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 161 | 45,272 | 75.3228% | 87.5498% | 60 / 67 | 35.939 |
| Extra rescue | 161 | 45,275 | 75.3202% | 87.5556% | 60 / 67 | 36.055 |
| Wider timing | 161 | 45,357 | 75.0310% | 87.7142% | 59 / 62 | 32.418 |
| Combined | 161 | 45,365 | 75.0343% | 87.7296% | 59 / 62 | 37.155 |

Two additional rescue passes add six assignments, only three of which add
to the true-positive count, and recover no extra units. The wider timing
search accounts for most of the combined change: more true spikes but lower
precision. Split/merge diagnostics improve in that arm, so the tradeoff is
not uniformly negative, but unit recovery does not increase.

## Separate interval, 180–210 seconds

The combined setting was fixed before this evaluation, with no subsequent
tuning. This interval was used in prior experiments; it is not a fresh dataset.

| Setting | Recovered units | True-positive spikes | Precision | Recall | F1 | Splits / merges | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference | 160 | 44,667 | 75.3035% | 87.4966% | 80.9434% | 59 / 69 | 34.813 |
| Combined | 160 | 44,673 | 74.8204% | 87.5083% | 80.6685% | 60 / 66 | 36.195 |

The combined run adds 391 assignments but only six true-positive spikes
to the aggregate result. Precision falls 0.483 percentage points and F1
falls 0.275 points. This does not justify replacing the reference setting.
The retained Kilosort4 comparison still recovers 231 units on this interval.

## Resources, implementation and validation

Every run reads 811,008,000 bytes for 737,280,000 covered raw bytes. Combined
derived output is 41,119,877 bytes in development and 40,659,470 in validation,
about 5.58% and 5.51% of raw bytes. Measured total VRAM remains unchanged
at 1,330,184,192 and 1,328,087,040 bytes respectively; GPU pool peaks remain
116,387,840 and 116,100,608 bytes. Combined process peak working sets are
about 1.616 and 1.617 GB. All timings are single profiled runs. The wider-only
run being faster than the older reference illustrates variability; do not
infer a speedup or stable marginal cost from this table.

Added bounded `--rescue-passes` (1–4, default 1) and `--shift-radius`
(0–8 samples, default 2). CUDA rescue uses the same bounded detector and
pair buffers, residual-gain check, amplitude limits and refractory rules.
Timing search also propagates to the CPU matcher. Settings and a new rescue
algorithm version are recorded in the resume configuration. Defaults remain
unchanged; older configuration versions require a new output directory.

All **199 tests pass** with `scripts/test.ps1`. Tests cover recovery of a synthetic weak overlap chain with multiple rescue
passes, recovery of an offset proposal with a wider timing search, invalid
bounds, and exact interrupted/resumed outputs with radius four and three
rescue passes under both overlap policies.

Artifacts under `F:/sortingDevelopment`:

- `session_accuracy_compute60_20260924_01`
- `session_accuracy_rescue3_60_20260924_01`
- `session_accuracy_shift4_60_20260924_01`
- `session_accuracy_compute180_20260924_01`

References: `session_detector_rescue60_20260924_01` and
`session_detector_rescue180_20260924_01`. Each run retains its quality report,
run report, profile and assignment shards. Sources were not modified.

The next hypothesis is to spend additional computation on model separation
and joint overlap fitting, rather than more iterations of the same greedy
matcher. Local PCA features, independently validated splits and local joint
amplitude fitting remain proposed work; none is implemented or validated by
this experiment. These measurements do not establish an accuracy ceiling.
