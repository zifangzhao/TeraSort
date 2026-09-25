# Kilosort post-clustering acceleration check, 2026-09-25

The latest completed Kilosort 4.1.7 run extracted 2,476,960 spikes and finished
in 470.85 seconds. Kilosort reported 42.64 seconds for final clustering,
1.10 seconds for merging, and 6.37 seconds for Phy export. Final clustering
visited 32 active spatial regions; the largest held 167,680 spikes. This makes
graph-neighbor search during final clustering the useful CPU/GPU optimization
target. Phy export was a much smaller part of this run.

The TeraSort adapter now replaces Kilosort's `neigh_mat` only when both the
installed version and the pinned function source match Kilosort 4.1.7. It
computes L2 neighbors on CUDA in bounded batches, recomputes numerically
ambiguous rows through FAISS on the CPU, and checks a deterministic spread of
additional rows against FAISS. If that check differs, it reruns the whole
spatial region through Kilosort's original CPU implementation. CUDA allocation
failure also takes the original CPU route. The sparse neighbor graph creation
and downstream Kilosort clustering stay unchanged. A separate guarded change
vectorizes Kilosort's per-template CPU feature gather.

For a focused check, 150,000 rows of 60-dimensional features were read from the
existing Phy export (`pc_features.npy`) and clustered with Kilosort's settings
of `cluster_downsampling=20` and `cluster_neighbors=10`. The CPU FAISS
neighbor search plus graph construction took 1.190 seconds; the guarded CUDA
path took 0.575 seconds on the NVIDIA RTX 5060 Ti, a 2.07x speedup for that
substage. The resulting sparse neighbor edges were exactly equal. Some
individual neighbor rows had a different order, which does not change the
Kilosort graph.

The same feature slice was also used to compare CPU FAISS thread counts,
including sparse graph construction. Medians across two runs were 1.157
seconds at 4 threads, 1.030 seconds at 8, 1.132 seconds at 16, and 1.188
seconds at 28 (the machine default). TeraSort now caps FAISS at eight threads
per sorting worker and restores the prior setting afterward. This also helps
the CPU-only route and avoids launching 28 FAISS workers for each concurrent
session.

This is a focused substage measurement, not a complete rerun or a ground-truth
quality comparison. The full final-clustering stage includes iterative cluster
assignment and splitting, so its end-to-end gain will be smaller. The CPU
feature-gather microbenchmark was exactly equivalent and 1.9x faster for a
synthetic region with 40,000 spikes and 240 selected templates; the measured
recording's final-clustering regions had fewer templates, so that gather change
is expected to contribute less on this specific run.
