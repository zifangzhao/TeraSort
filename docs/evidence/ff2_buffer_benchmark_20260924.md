# FF2 bounded read buffers: measured effect

Larger buffers reduce read requests and repeated halo bytes, but did not improve end-to-end speed on this recording in the measured cache conditions. The default remains the original reader. The new options are available for higher-latency or source-limited workloads.

## Implementation

`terasort session-sort --read-buffer-mb 64 --prefetch-depth 2` groups adjacent cores into a sequential read of at most 64 MiB. Each queued core owns a small, read-only copy, preventing old large buffers from staying alive through queued views. Grouping never crosses a file boundary or explicit gap, and preserves the original core/halo samples. Shared halos are read once per group. The buffer is configurable from 0 (disabled) to 1024 MiB; queue depth is 1–64 cores. Defaults remain 0 and 2. The configured buffer must fit a full core plus both halos.

Nondefault IO options are recorded in the immutable run configuration. Default checkpoint digests remain compatible with earlier runs. Dense sorting uses the optional buffers; calibration is unchanged. Raw sources remain read-only and no whole-file staging is introduced.

## Paired end-to-end trials

All trials sort the same source seconds 0–120 of FF2 D82 Linear A, with the same frozen 72-template bank, raw detector, score 0.75 and rescue SNR 3.5. Eight fresh worker processes run in the order baseline, 64 MiB, 256 MiB, deeper queue, then reverse order. Results below are medians of two trials, except RAM which is the larger observed sampled peak. These are sequential warm/cache-affected trials, not cold-network bandwidth measurements.

| Read buffer | Queued cores | Median wall time | Peak process RSS | Source bytes read |
|---|---:|---:|---:|---:|
| Per-core reader | 2 | 24.78 s | 510 MB | 675.33 MB |
| 64 MiB | 2 | 24.89 s | 572 MB | 624.13 MB |
| 256 MiB | 2 | 25.98 s | 774 MB | 616.96 MB |
| 64 MiB | 8 | 26.10 s | 637 MB | 624.13 MB |

Each trial covers 614.4 MB of raw input and assigns exactly 74,491 spikes from 1,071,998 candidate rows. All 21 scientific datasets are equal across all eight runs; only the QC source-byte accounting and run/telemetry attributes are excluded from comparison. This includes exact spike, candidate, waveform, QC flag and model arrays. CUDA pool/total device VRAM remain 51.25 MB/1.265 GB across settings.

64 MiB reduces source bytes by 7.6%, but changes median wall time by +0.4%; that is no demonstrated speedup. 256 MiB uses more RAM and is about 4.8% slower. A deeper queue is about 5.3% slower. With only two trials per setting, small timing differences should not be treated as statistically established effects.

Separate read-only 60-second trials reduce read requests from 30 to 5 with 64 MiB, or 2 with 256 MiB. The first unbuffered read took 3.56 seconds; the repeat took 0.105 seconds, while grouped reads took 0.154–0.168 seconds. This strong cache effect prevents attributing these times to physical network bandwidth. Grouped reads also pay for small-core copies. No caches were forcibly flushed, and NIC wire bytes were not measured.

## Profile: what currently limits throughput

A separate profiled 30-second run took about 7.76 seconds including profiler overhead. Python cumulative timings were:

| Stage | Cumulative time | Approximate share |
|---|---:|---:|
| Preprocessing: reference, filtering, array operations | 3.245 s | 42% |
| GPU matcher and its host orchestration | 1.972 s | 25% |
| Candidate/waveform output | 1.215 s | 16% |
| Quality assessment | 0.779 s | 10% |
| Prefetch generator, including waiting | 0.108 s | 1.4% |

These are stage timings, not a GPU-kernel trace or uninstrumented benchmark. Reader work runs on a separate thread; consumer prefetch time is evidence of little exposed source waiting in this run. Preprocessing plus QC accounts for roughly half the time. Faster preprocessing is therefore a better next optimization target for this case than increasing the read queue. It must preserve the calibration/dense preprocessing frame and be checked against existing scientific outputs.

## Validation and artifacts

The test suite passes: 219 tests in 17.14 seconds. New tests compare grouped and original raw cores through multiple read-buffer boundaries, odd start/stop positions and explicit file gaps, verify buffer request limits and array ownership, and exercise invalid sizes. The full 29-minute validation and abrupt-kill replay used the original reader and matched exactly; see [full recording results](ff2_full_validation_20260924.md). Grouped reads have been tested on 120-second real intervals, not a full multiday run or a killed grouped-read run.

Artifacts: `F:\sortingDevelopment\ff2_buffer_benchmark_20260924_01` contains all eight outputs, `read_only.json`, `integrity.json`, and `profile.pstats`/`profile.txt`. Reproduce the paired trials with `scripts/benchmark_ff2_buffers.py` and a new output directory. Sources and prior output folders were preserved.
