# C++/cuBLAS and custom CUDA filter: full sample comparison

The selected native path uses a shared-memory six-by-61 temporal filter,
writes PCA-major output directly, and calls FP32 cuBLAS SGEMM through a compiled
C++ library. It retains tiled universal detection, the deep CUDA matcher and
calibrated INT16 I/O. The control is our previous optimized Kilosort 4.1.7
tiled implementation, not unmodified stock Kilosort.

| Complete 600-second, 384-channel sort | Tiled control | C++/cuBLAS + CUDA |
|---|---:|---:|
| Worker seconds | 214.379 | 208.186 |
| Process seconds | 219.093 | 213.230 |
| Peak sampled Torch allocation GiB | 1.544 | 1.545 |
| Peak sampled Torch reservation GiB | 2.141 | 2.318 |
| Peak sampled host RSS GiB | 3.325 | 3.319 |
| Recovered GT neurons, IoU >= 0.8 | 229/250 | 230/250 |
| Sorted clusters | 517 | 519 |
| Sorted spikes | 1,441,232 | 1,440,042 |
| Global spike recall | 98.2507% | 98.1924% |
| Global spike precision | 69.9877% | 70.0039% |

Observed worker speedup: 1.030x, or
2.89% less worker time.
Peak sampled Torch allocation was essentially unchanged.
Fresh native INT16 full sort followed by fresh tiled INT16 control. Each input fully pre-read and SHA256 verified before timing. One warmed run per route, no concurrent GPU benchmark. Not a randomized repeated full-run comparison or cold-disk benchmark. Torch allocator peaks exclude native library workspace and other non-Torch VRAM.
Input pre-reading/checksumming took 10.50 s for
native and 10.12 s for control, excluded from
sorting. Worker time includes sorting/export/plots; process time also includes
imports and setup. Stage sums omit some preparation/export overhead.

| Stage seconds | Tiled control | Native |
|---|---:|---:|
| preproc | 0.825 | 0.829 |
| drift | 40.117 | 38.586 |
| st0 | 40.227 | 38.415 |
| clu0 | 29.358 | 28.077 |
| st | 36.168 | 34.593 |
| clu | 31.039 | 31.679 |
| merge | 3.414 | 3.394 |
| postproc | 5.404 | 5.213 |

## Where the improvement comes from

| Four-batch warmed measurement | Previous ms | Native ms |
|---|---:|---:|
| Temporal convolution | 11.888 | 6.294 |
| Complete learned matching | 66.575 | 58.183 |
| Complete universal detection | 63.973 | 57.654 |

At the representative learned-matching shape, incremental live Torch
allocation fell from 1,232,505,344 to 678,420,992 bytes. The native filter
avoids a 554 MB signal permutation copy by writing directly into PCA-major
layout; it also reduces temporal convolution time. Filtering, projection,
scores and residuals remain FP32. Raw storage/transfer remains INT16.

Direct cuBLAS alone was not faster than an equivalent Torch call: channel-major
projection was 10.685 ms with C++/cuBLAS versus
10.581 ms with Torch matrix multiply. With the
PCA copy included, cuBLAS took 13.555 ms versus
13.524 ms with Torch. The exact installed
einsum took 13.496 ms. This separates library dispatch
from layout and memory-traffic improvements.

PyTorch already uses NVIDIA BLAS libraries, as documented in its
[2.10 backend reference](https://docs.pytorch.org/docs/2.10/backends.html#torch.backends.cuda.preferred_blas_library).
The C++ bridge currently calls SGEMM, not autotuned cuBLASLt. The CUDA kernels
compile with NVRTC; Torch still owns allocations and performs the remaining
preprocessing, learning/clustering, orchestration and export.

## Accuracy and validation

The selected path preserved all four learned outputs for 20,000
events and all four universal outputs for 11,332 events across
four fixed-model batches, including amplitudes, scores and residuals. Alternative
channel-major projection changed rounding and was not selected. Forty-six
targeted tests passed, including FP64 references, boundary/tail padding,
invalid inputs, layout, streams, cache invalidation, patch restoration and
existing deep/tiled/INT16 behavior.

Ground truth is used only after sorting, with all clusters, no curation and
0.4 ms tolerance. Between the full runs, 3 GT units crossed below 0.8
IoU and 4 crossed above it. Full outputs can vary with training and
clustering; exact fixed-model comparisons do not establish biological
equivalence. Final array equality: `{"spike_times.npy": false, "spike_clusters.npy": false, "amplitudes.npy": false, "templates.npy": false}`.
All 60 sorter-output/metric checksums were verified. Sources, native DLL,
build metadata, configurations, telemetry and per-unit scores are retained.

This remains a 600-second validation. Global feature collection/clustering and
multiday unit tracking still require bounded training and streamed output for
100 TB operation. Smaller GPU intermediates alone do not establish that scale.

## Reproduce

```powershell
.venv-spike-candidates/Scripts/python.exe scripts/build_cublas_bridge.py
.venv-spike-candidates/Scripts/python.exe scripts/benchmark_kilosort_cublas.py --root F:/sortingDevelopment
.venv-spike-candidates/Scripts/python.exe scripts/run_native_sorting_trial.py --root F:/sortingDevelopment --mode cublas --storage int16 --trial cublas_sorting_v1 --job-name cublas_int16 --warm-input
.venv-spike-candidates/Scripts/python.exe scripts/run_native_sorting_trial.py --root F:/sortingDevelopment --mode deep_tiled --storage int16 --trial cublas_sorting_v1 --job-name deep_tiled_control --warm-input
.venv-spike-candidates/Scripts/python.exe scripts/report_cublas_sorting.py --root F:/sortingDevelopment
```
