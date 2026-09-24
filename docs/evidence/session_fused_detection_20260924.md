# Fused detection and bounded overlap recovery — 2026-09-24

## Implemented

The direct CUDA session detector now reads residual voltage and per-channel
noise, applies the existing absolute-SNR threshold and temporal peak rule,
and packs candidates with warp-aggregated atomics. It avoids full-sized
absolute-value/SNR intermediates. Indices use 64-bit arithmetic; the output
buffer is capped at the per-core candidate limit. Overflow is an error,
never silent truncation. Infinite channel noise masks unreliable contacts.
The NumPy detector and existing generic candidate detector remain references.

Added `--residual-passes` (1–12, default 3). Additional passes use a central
cosine floor of max(configured floor, 0.85) after the third pass. This is an
optional experimental overlap-recovery tradeoff, not a new default. CPU
and CUDA implement the same late-pass score requirement.

## Detection equivalence and resource results

Using frozen seeds on 60–90 seconds of the existing public recording, all
candidate and spike rows in three 10-second shards were exactly equal to
the retained pre-fusion control. Recovery remained 151/250 units, precision
75.2173%, recall 86.0278%. Raw source files were not modified.

The paired detector benchmark used a real 70,400 x 384 source core with
halos, 32,791 detected events, warm allocations/compilation and 20 alternating
old/new repetitions. It includes host event-list construction. Median time:

- Previous detector path: 4.724 ms.
- Fused detector path: 3.485 ms, about 26% lower time (1.36x throughput).

This is detection-only timing, not whole-sorter speedup. On the full pilot,
sampled CuPy pool use decreased from 432,539,136 to 116,387,840 bytes, about
73%. Sampled total device memory decreased from 1,648,951,296 to
1,330,184,192 bytes. These are sampled peaks and depend on allocation history.

The single profiled end-to-end pass took 31.50 seconds versus the earlier
29.88-second control, so no whole-pipeline speedup is established. The
profile attributes 18.62 seconds to CPU preprocessing, 5.95 seconds to GPU
matching including host selection, and 3.58 seconds to QC (inclusive timings,
not disjoint GPU kernel durations). Preprocessing dominates this workload.

## Overlap-recovery experiments

Same frozen seeds, input interval and geometry; only the residual pass policy
changed. No ground truth was used inside detection or fitting. This interval
has been reused for development, so these are exploratory measurements,
not independent final validation.

| Policy | Recovered units | Matched spikes | Precision | Recall | Profiled wall time |
|---|---:|---:|---:|---:|---:|
| 3 passes | 151/250 | 44,485 | 75.2173% | 86.0278% | 31.50 s |
| 6 passes, original score floor throughout | 152/250 | 44,587 | 74.6801% | 86.2251% | 32.89 s |
| 6 passes, stricter late score floor | 151/250 | 44,536 | 75.2031% | 86.1265% | 32.14 s |

Unrestricted extra passes added too many unmatched assignments. The guarded
version added 51 matched spikes and 28 unmatched assignments: a modest
recall gain with a slight precision loss. Default remains three passes.
The original unrestricted six-pass experiment is retained as evidence but
the final six-pass implementation uses the stricter late score floor.

## Why a repeated strong unknown stayed unassigned

At source sample 2,069,352 on channel 94, a bounded raw-core inspection with
the original seed bank found eligible templates 312 and 313. Their residual
gains were approximately 15,345.93 and 15,308.49, central cosines 0.9929 and
0.9925, amplitudes 1.0345 and 1.0264. The relative gain margin is only 0.244%,
below the 3% assignment requirement. This example is detected but ambiguous;
lowering the detection threshold would not resolve its identity. It does
not justify automatically merging the templates. Cross-time competition
and contamination evidence are needed before identity correction.

## Validation and artifacts

**177 tests passed.** Added exact detector comparisons for ties, boundaries,
masked channels and partial warps; overflow checks; and a four-spike overlap
chain that needs more than three passes. Existing CPU/CUDA enrollment,
interruption/resume, fitting and bounded-state tests also pass.

New artifacts under `F:\sortingDevelopment`:

- `session_fused3_20260924_01`
- `session_fused6_20260924_01` (unrestricted experiment)
- `session_fused6_guarded_20260924_01`
- `session_fused_detection_micro_20260924_01.json`

The benchmark script is `scripts/benchmark_session_detection.py`; recording
pilots use `scripts/benchmark_session_recording.py --freeze-templates` and
`--residual-passes`. New configuration fields require a fresh run directory
for older session outputs. Further quality claims require new drift/collision
datasets and full-session validation; the Kilosort4 baseline remains ahead
in recovery and recall.
