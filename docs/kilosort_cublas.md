# Native C++ / cuBLAS and temporal filtering

This extends the tiled INT16 Kilosort 4.1.7 route with a small compiled C++
library calling cuBLAS directly and a specialized CUDA C++ temporal filter.
It is an opt-in GPU hot-path change, not a rewrite of the entire sorter in C++.
Torch still owns tensors and runs preprocessing, learning, clustering, export
and the remaining orchestration. Results are retained under
`F:/sortingDevelopment/cublas_sorting_v1`.

The complete 600-second INT16 trial took **208.186 s**, compared with
**214.379 s** for a fresh tiled control: **2.9% less worker time**. Peak sampled
Torch allocation was essentially unchanged at about 1.55 GiB. The native run
recovered 230/250 GT neurons at IoU >= 0.8, versus 229/250 for control; three
units crossed below and four above the criterion. This single warmed pair
does not establish an accuracy improvement or a robust full-sort speedup.
All 60 sorter-output/metric checksums passed. See the full `report.md` in the
evidence directory for resource, stage and per-unit measurements. Native cuBLAS
private workspace is excluded from Torch's allocated-memory metric.

## Measured opportunity and implementation

The retained profile identified temporal convolution and learned projection as
remaining costs. The exact installed projection expression is
`torch.einsum('ijk, kjl -> il', U, B)`, with U[unit,PCA,channel] and
B[channel,PCA,time]. The existing path materializes a reordered full signal
tensor before matrix multiplication. A 384 x 6 x 60,122 FP32 tensor is about
554 MB, and duplicating it raises both memory traffic and peak allocation.

`kilosort_cublas.cu` computes the six 61-tap temporal correlations using a
256-sample shared-memory tile, a 60-sample halo, shared filter coefficients and
coalesced output stores. FP32 FMA is explicit; there is no new signal
quantization or fast-math option. It can write channel-major data for universal
detection or PCA-major data for learned projection.

Writing PCA-major data directly removes the later full-tensor copy while
retaining the installed projection's summation order. The C++ bridge then calls
FP32 `cublasSgemm` on that buffer. It owns a separate cuBLAS handle, sets the
caller's current CUDA stream and leaves Torch's handle/precision flags alone.
The bridge uses default cuBLAS math, with no explicit TF32/half opt-in. Selected
ops guard shape/dtype/device and use 64-bit addressing within kernels.

The current specialization requires six 61-tap temporal filters. The native
bridge is built for Windows x64 with installed MSVC 2019; CUDA kernels compile
through NVRTC. It loads the absolute cuBLAS DLL shipped with Torch, so no old
system CUDA library is silently substituted. Measured cuBLAS version was 120804
with Torch 2.10.0+cu128 on an RTX 5060 Ti. Build logs, compiler location, source
hash and binary hash are retained.

## Four-batch microbenchmark

The benchmark uses saved final templates projected onto the saved PCA basis,
four real batches (0, 47, 160, 319), seven warmed rotating trials per batch,
and identical processed voltage. It includes output allocation but excludes
file I/O, preprocessing, compilation and one-time model packing.

| Operation | Prior route ms | Native route ms |
|---|---:|---:|
| Temporal convolution | 11.888 | 6.294 |
| Full learned matching | 66.575 | 58.183 |
| Tiled universal detection | 63.973 | 57.654 |

The complete learned-matching stage improved by 12.6% in elapsed time;
universal detection improved by 9.9%. Incremental live Torch allocation during
learned matching fell from 1,232,505,344 to 678,420,992 bytes, about 45%.
These are stage measurements, not whole-sort speed or total process VRAM.
Native cuBLAS private workspace is not counted by the Torch allocator.

All four learned outputs (20,000 events across these batches) and all four
universal outputs (11,332 events) matched the previous optimized route bitwise
for the selected PCA-major/native-filter path. This fixed-model result is not
a claim that every full Kilosort run produces identical final labels.

## cuBLAS alone does not explain the gain

PyTorch already uses cuBLAS/cuBLASLt for CUDA BLAS operations; see the
[PyTorch 2.10 backend documentation](https://docs.pytorch.org/docs/2.10/backends.html#torch.backends.cuda.preferred_blas_library).
On matching layouts our direct C++ cuBLAS call was not faster than Torch's
matrix multiply:

| Projection variant | Median ms |
|---|---:|
| Exact installed einsum, including reorder | 13.496 |
| Torch matrix multiply, including PCA reorder | 13.524 |
| C++ cuBLAS, including PCA reorder | 13.555 |
| Torch matrix multiply, channel-major without signal copy | 10.581 |
| C++ cuBLAS, channel-major without signal copy | 10.685 |

The channel-major no-copy variants change summation order and produced small
amplitude/residual differences in learned matching. The integrated option uses
the native filter's PCA-major output, which avoids copying without that order
change and matched all tested outputs exactly.

An initial diagnostic used a relabeled equivalent einsum expression, which
changed its flattening order and copy cost. The accepted benchmark was rerun
with the exact installed labels. This is why we compare layouts and math
explicitly rather than attribute every improvement to replacing Python.
NVIDIA's [cuBLAS documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html)
describes matrix layouts, stream/handle behavior and default math modes.

## Integration and verification

Use `--mode cublas --storage int16` with the complete-sample harness. The
adapter combines native universal filtering, 2,048-sample score tiles, native
PCA-major learned filtering/projection and the earlier deep CUDA kernels.
Installed Kilosort functions are source guarded and restored on exit, including
exceptions; the native handle is released. Completed jobs include source/DLL
hashes, build information and runtime statistics. `--mode deep_tiled` remains
the comparison route and `--mode deep` remains the CLI default.

Forty-six targeted tests passed for native GEMM against FP64 reference,
convolution boundaries/tails, both data layouts, shape/dtype rejection, CUDA
streams, cache invalidation, context restoration, earlier deep matching,
tiled detection and INT16 I/O. Full-sort speed, memory, per-unit ground truth
and checksum verification are reported separately in the evidence directory.

```powershell
.venv-spike-candidates/Scripts/python.exe scripts/build_cublas_bridge.py
.venv-spike-candidates/Scripts/python.exe scripts/benchmark_kilosort_cublas.py --root F:/sortingDevelopment
.venv-spike-candidates/Scripts/python.exe scripts/run_native_sorting_trial.py --root F:/sortingDevelopment --mode cublas --storage int16 --trial cublas_sorting_v1 --job-name cublas_int16 --warm-input
.venv-spike-candidates/Scripts/python.exe scripts/run_native_sorting_trial.py --root F:/sortingDevelopment --mode deep_tiled --storage int16 --trial cublas_sorting_v1 --job-name deep_tiled_control --warm-input
```

This does not remove the global feature/clustering growth described in
[the bounded-memory plan](bounded_sorting.md). Full C++ orchestration, reusable
workspaces, asynchronous I/O, real-FFT preprocessing and cuBLASLt algorithm
selection remain possible experiments; none is claimed as a measured gain here.
