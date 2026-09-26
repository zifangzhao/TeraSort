# Shared-memory smooth3 detection, 2026-09-25

## Change

The opt-in smooth3 candidate detector computes the three-sample temporal
filter, local temporal maximum, threshold, and warp-aggregated candidate packing
in one CUDA kernel. Each block stages an 8-sample time tile with a two-sample
halo on either side in shared memory (1.5 KiB per block). It avoids a full
time-by-channel filtered-core allocation. Robust channel-noise estimation
samples at most about 4,000 rows per core.

In the session matcher, smooth3 noise is now estimated once from the original
core and reused for every residual-subtraction pass. This avoids repeating the
sampling/host-MAD work and keeps the candidate SNR frame stable while the
residual changes. Standalone `detect(..., mode="smooth3")` still estimates
noise per call. Run metadata is versioned as `smooth3_core_MAD_v2` so an
in-progress session cannot silently resume with the previous per-pass scoring
semantics. Raw detection is unchanged; no defaults or score thresholds changed.

## Correctness and tests

A 70,400 x 384 sample core from the public 600-second, 250-unit ground-truth
recording produced 35,508 candidates. In 12 alternating detector-only
measurements, the fused path exactly matched the previous filter-then-detect
path in candidate coordinates and SNR values.

The complete repository test suite passed: 241 tests. The CUDA matcher tests
cover detector tile edges, partial channel/time tiles, masked channels,
overlap matching, residual passes, and resume integrity.

## Held-out quality and speed

On the held-out 480–510 second interval, two runs of the frozen-core-noise
smooth3 configuration produced identical quality metrics: 170/250 units at
IoU >= 0.8, 43,903 true-positive spikes, 81.901% precision, and 86.340%
recall. End-to-end times were 27.691 s and 27.570 s (27.631 s median). Peak
CuPy pool use was 146,336,256 bytes; sampled total VRAM use was 1,372,127,232
bytes; peak process RSS was about 969 MB.

The same-window raw-detector run took 28.420 s and recovered 166 units, with
43,519 true positives, 82.268% precision, and 85.585% recall. Frozen-noise
smooth3 reduced end-to-end time by 2.8%, added four recovered units and 384
true positives, and improved F1 from 83.894% to 84.062%. Precision declined by
0.37 percentage points while recall improved by 0.75 points.

On this interval, the Kilosort4 reference recovered 227 units with 50,057 true
positives, 71.051% precision, and 98.442% recall. TeraSort still recovered 57
fewer units and has substantially lower recall; this detector optimization
does not close the quality gap. The comparison is an offline held-out
evaluation, not a claim of a replacement-quality sort.

For context, the prior per-pass-noise smooth3 run recovered 173 units and
44,318 true positives at 81.471% precision and 87.156% recall, with a 29.772 s
same-machine run time. The frozen-core policy is about 7.2% faster on this
single paired comparison and trades three recovered units and 415 true
positives for 0.43 points more precision. Its F1 is 84.062% versus 84.218%.
This is a configurable tradeoff, not an across-dataset conclusion.

The detector-only paired benchmark, including noise estimation, event sorting,
and host event construction, measured:

| Path | Median detector time |
|---|---:|
| Previous full-core temporary | 39.16 ms |
| Shared-memory fused | 37.28 ms |

That is 1.05x detector throughput with exact candidate/SNR equality for the
standalone call. This detector benchmark is reproducible with
`scripts/benchmark_smooth3_detector.py`.

The held-out outputs are retained under
`C:\\Frank\\Code\\PainProject\\outputs\\smooth3_core_noise_quality_20260925_01`
and
`C:\\Frank\\Code\\PainProject\\outputs\\smooth3_core_noise_repeat_20260925_01`.
The raw comparison is under
`C:\\Frank\\Code\\PainProject\\outputs\\smooth3_pair_raw_20260925_01`.
The prior per-pass-noise comparison is under
`C:\\Frank\\Code\\PainProject\\outputs\\smooth3_pair_smooth_20260925_01`.

These results use one dataset and one held-out interval. The broader quality
gate and multiday/generalization checks remain open.

## Exact miss replay and targeted experiments

The diagnostic replay in `scripts/trace_session_failures.py` was extended to
use the run's configured threshold, score floor, overlap policy, and smooth3
core-MAD candidate frame. On the held-out run it reproduced all 53,605
assignments exactly. It selected 686 missed ground-truth events for tracing
(up to four evenly spaced misses per linked unit, not a prevalence-weighted
sample); 526 had a predicted-unit mapping with IoU >= 0.5.

For those 526 more-credible sampled misses, the furthest observed failure was
template-shape rejection for 231, overlap deferral for 128, a different
template winning for 135, ambiguity rejection for 28, amplitude rejection for
2, and no above-threshold candidate on the mapped contacts for 2. This points
to residual fitting and template competition as the larger opportunity than
lowering the detector threshold. The mapping still comes from the evaluation,
and the selected misses do not estimate whole-recording error prevalence.

Two controlled alternatives were run on the same 30-second interval with the
same detector and thresholds. Allowing the bounded template updater promoted
108 templates but recovered 170 units and 43,899 true positives, essentially
unchanged from frozen templates (170 units, 43,903 true positives). It took
28.646 s versus the two-run frozen median of 27.631 s. Enabling the existing
interference scheduler recovered 170 units and 43,922 true positives in
28.180 s. Its F1 gain over frozen templates was only 0.016 percentage points
in this single run. Neither alternative is a meaningful quality advance yet.

These experiments do not justify loosening quality controls globally. The
next quality work should improve drift-aware template representation and
collision/competitor decisions while retaining the detector's bounded memory
and exact restart behavior.
