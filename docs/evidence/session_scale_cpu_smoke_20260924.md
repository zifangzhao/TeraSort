# Session-sort repeated-source CPU smoke, 2026-09-24

Command: `python scripts/benchmark_session_scale.py --backend cpu`, using
Python 3.10 with `PYTHONPATH=src` because the repository's Python 3.11
environment was unavailable. This is an uninstalled reference-path test;
the package declares Python 3.11 or later.

The fixture is one 6,000-sample, two-channel INT16 file reused 1, 10, and
100 times through distinct source-clock segments. The short segment duration
produced one HDF5 shard per repetition, so fixed HDF5 metadata dominates
storage. It is intentionally a memory/restart infrastructure smoke test,
not an end-to-end long-recording or remote-I/O benchmark.

| Repeats | Shards | Raw bytes covered | Candidates | Spikes | Peak process RSS | Wall time | Derived/raw bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 24,000 | 24 | 3 | 111,460,352 | 0.022 s | 5.65 |
| 10 | 10 | 240,000 | 240 | 30 | 113,209,344 | 0.155 s | 5.65 |
| 100 | 100 | 2,400,000 | 2,400 | 300 | 114,221,056 | 1.461 s | 5.65 |

Peak RSS rose 2,760,704 bytes from 1× to 100×. All 100 shards were
published. The temporary fixture and shards were removed when the script
finished. GPU VRAM and utilization were not tested. The next scale gate must
use realistic five-minute shards, network sources, artifacts, killed workers,
and real long recordings; this result does not establish a 100 TB capability.

A separate read-only diagnostic on the first two seconds of the existing
600-second, 384-channel, 250-unit public recording used the same source reader,
preprocessing, QC, and 4.5-SNR candidate rule. It read 50,688,000 bytes
including a 100 ms right halo, found 29,913 candidates, marked zero contacts
or the interval bad, and measured 11.96 µV median channel noise. No templates
were learned or assignments made in this two-second diagnostic.
