# Optional parallel LFP: 600-second sample

The original `terasort lfp` command performed a separate pass after sorting.
The optional `terasort sort --lfp-output PATH` mode now starts the CPU LFP
worker only after Kilosort 4.1.7 completes drift correction, universal spike
detection, first clustering, and learned spike detection. It overlaps LFP with
final clustering, merge, postprocessing, and plotting. The LFP worker uses a
separate file handle, eight channel-filter threads, and lower Windows process
priority. This avoids LFP competition during spike detection by scheduling;
it does **not** eliminate LFP compute, output writes, or a second logical raw
read.

Input: 600 seconds, 384 channels, 32 kHz interleaved INT16
(`F:/sortingDevelopment/data/recording_int16_uv.bin`, 14,745,600,000 bytes).
Output: 1,250 Hz interleaved INT16 LFP, 0–500 Hz passband, 625 Hz stopband
start, 576,000,000 bytes, zero clipped values. The deferred output SHA-256 is
`DFB5978D0344F361ACFED82B3FC9082E5220B5BC70CBA29D8B52470BAE5C2FEB`,
identical to the earlier standalone 500 Hz export.

| Full 600-second run | Sort only | LFP started immediately | LFP after detection |
|---|---:|---:|---:|
| Process elapsed, seconds | 229.69 | 229.99 | 227.07 |
| Kilosort runtime, seconds | 200.38 | 208.66 | 204.54 |
| Drift correction, seconds | 38.31 | 43.53 | 38.05 |
| Universal spike detection, seconds | 38.2 | 38.4 | 38.07 |
| Learned spike detection, seconds | 37.9 | 37.5 | 36.75 |
| Final clustering, seconds | 32.8 | 33.9 | 36.4 |

Immediate launch delayed drift correction by about 5.2 seconds. Deferred
launch began only at final clustering and finished about nine seconds before
sorting completed. The three detection stages showed no slowdown relative to
the fresh sort-only control in this run. Kilosort's overall runtime was 4.16
seconds longer with deferred LFP, chiefly in final clustering; process elapsed
was 2.62 seconds shorter. These are single runs on a warm local file, so the
small process-time difference is run variation, not a measured speedup or a
guarantee of zero cost. Different disk, CPU load, channel count, passband, or
sort duration can leave LFP running past sorting. The command waits for both
outputs and retains a resumable partial LFP if interrupted.

The standalone exporter and the deferred mode were validated by 110 automated
tests, including exact whole-array/chunked resampling and a deferred-start
integration test. The two full 500 Hz exports also have identical SHA-256.
