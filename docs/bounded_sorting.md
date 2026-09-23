# Bounded-memory route toward 100 TB spike sorting

The objective is fixed working memory as recording duration grows, faster
processing per input byte, and resumable output. Integer arithmetic alone does
not provide those properties. Keep raw recordings and cached waveforms in
calibrated INT16 where suitable; choose compute precision by measured throughput,
overflow bounds and detection/sorting accuracy.

## Implemented first step

`kilosort_tiled.py` is an opt-in adapter for the validated Kilosort 4.1.7 fused
universal detector. It keeps full-batch temporal convolution unchanged, then
computes spatial scores and neighborhood suppression in time tiles. Each tile
includes a temporal suppression halo; global border masking precedes suppression.
Outputs are restored to the original center/time ordering. No candidate cap or
new quantization is introduced.

The specialized 5-width, 10-neighbor geometry uses existing shared-memory CUDA
kernels. Other geometries retain the previous detector, without this workspace
reduction. This is not a bound on Kilosort's global clustering or on total GPU
process memory. The adapter remains source/version guarded and scoped; the
installed Kilosort package is unchanged.

The benchmark compares identical saved models and preprocessed samples from
six positions in the 600-second, 384-channel public sample. It measures seven
warmed trials per batch in rotating order, excluding I/O and training. All four
outputs must match the prior fused implementation exactly before accepting the
measurements. Tests also cover seam competitors, tied peaks, signs, global
borders, empty output, partial tiles, another CUDA stream and patch restoration.

On the RTX 5060 Ti, the completed six-batch benchmark measured:

| Detector workspace | Median ms | Incremental peak MiB |
|---|---:|---:|
| Prior fused, full score arrays | 69.584 | 3,025.2 |
| Tile 1,024 | 71.862 | 572.1 |
| Tile 2,048 (integrated option) | 66.016 | 613.9 |
| Tile 4,096 | 66.187 | 697.7 |
| Tile 8,192 | 65.529 | 865.2 |

The 2,048-sample option reduced incremental detector allocation by 79.7% and
elapsed time by 5.1%. All 16,983 candidates and all four output tensors were
bitwise identical to the prior fused detector for each tile size. These savings
are within the detector, not a claim about full-sort peak memory or runtime.
The very smallest tile incurs extra launches/synchronizations; smaller is not
automatically faster. Twenty-eight targeted tiled-detector, INT16-reader and
window-ownership tests passed.

Reproduce in the prepared environment:

```powershell
.venv-spike-candidates/Scripts/python.exe scripts/benchmark_kilosort_tiled.py --root F:/sortingDevelopment
.venv-spike-candidates/Scripts/python.exe scripts/run_native_sorting_trial.py --root F:/sortingDevelopment --mode deep_tiled --storage int16 --trial bounded_sorting_v1 --job-name deep_tiled_int16 --warm-input
```

`F:/sortingDevelopment/bounded_sorting_v1/` contains measurements, source hashes
and snapshots. `--mode deep` remains the previous default; `deep_tiled` opts into
2,048-sample score tiles plus the existing learned-template CUDA implementation.
Full-sort results and the distinction between detector and whole-sort memory
are recorded in `report.md` there. The complete 600-second sample comparison
with calibrated INT16 input measured:

| Full-sort metric | Fresh deep control | Deep with 2,048-sample tiles |
|---|---:|---:|
| Worker seconds | 218.610 | 213.393 |
| Sampled peak Torch allocation GiB | 3.104 | 1.545 |
| Well-recovered GT neurons, IoU >= 0.8 | 228/250 | 228/250 |

That is 50.2% lower sampled peak allocation and 2.4% less worker time. The tiled
run preceded the control; both inputs were fully pre-read/checksummed, and no
GPU benchmark ran concurrently. One warmed run per option is not a repeated
randomized speed study or a cold-storage result. Allocator peaks exclude CUDA
context/library and other non-Torch memory. Both runs recovered the same count
of GT units, but two fell below and two rose above 0.8 IoU; final arrays differ.
Exact detector equivalence with fixed models is narrower than full-sort
equivalence when training/clustering vary. Sixty output/metric checksums passed.

## Remaining growth in the existing sorter

Inspection of the installed Kilosort 4.1.7 implementation shows that
`spikedetect.run` collects all detected-spike features in arrays that double as
needed. `template_matching.extract` similarly grows the final spike/feature
arrays and sorts/reorders them globally. `clustering_qr` retains feature/query
and assignment data proportional to the event count. Its capped neighbor
reference subset does not bound all those arrays.

For scale, ten channels x six FP32 PCA values cost 240 bytes per event before
timestamps, graph data, temporary copies and other metadata. Ten billion events
would require 2.4 TB for that feature tensor alone. Storing raw voltage in INT16
does not reduce this tensor; its lifetime and training scope need to change.

CUDA tiling addresses temporary working arrays, but none of those changes makes
an entire multiday recording fit into a fixed global clustering problem. The
previous naive independent-window pilot also increased runtime and fragmented
units. Repeatedly running a complete sorter on every small window is not the
preferred architecture.

## Proposed architecture, not yet implemented

| Component | Bound/resource policy | Reuse |
|---|---|---|
| Reader | Fixed pool of pinned INT16 chunks with filter/NMS halos; backpressure on every queue | Native INT16 reader and calibrated metadata |
| Calibration/training | Capped reservoirs stratified by probe, location, time, amplitude and drift epoch | Existing preprocessing and Kilosort model learning on bounded input |
| GPU detection/matching | Fixed input duration; score tiles; local active template limit; overflow triggers spatial/epoch partitioning | Existing CUDA kernels and Kilosort matching semantics |
| Candidate ledger | Immutable probe/time shards, time index and 64-bit source sample coordinates | SCB metadata, waveform geometry and quantization machinery |
| Features/waveforms | Explicit bytes-per-input-byte budget; selected snippets or shared waveform tiles, plus optional compact features | INT16 local waveform storage; a new schema for partial cache coverage |
| Assignment/output | Stream spikes/features to shards instead of retaining full-recording arrays | Local template assignment and existing quality metrics |
| Unit continuity | Bounded overlap/summary comparisons between adjacent drift epochs; explicit uncertain links | Evaluate existing matching tools on local summaries |
| Scheduling | Workers own disjoint probes/shards; local checkpoints and a small indexed manifest | Existing atomic completion/checksum approach |

Training samples must cover the full recording: a model learned only from the
first minutes will miss late-appearing or transient units. A sparse calibration
pass can precede dense assignment, with drift/novelty-triggered model updates.
Each new model gets a version and an owned time range. Preserve preprocessing,
geometry, units and scale for every epoch. Flexible probes may require geometry
as well as drift updates; do not assume one stationary template bank for days.

Subsampling is for model learning and optional waveform caches, not for silently
discarding detections in the dense assignment pass. Track unassigned/novel events
and cache coverage so difficult periods remain inspectable. The guarantee of a
candidate ledger is limited to its stored detection floor and preprocessing;
lower floors or changed filtering still require the raw source.

Artifact bursts need an explicit policy: split or drain output chunks under
backpressure, record artifact intervals, and never silently overflow a fixed
candidate buffer. Test these cases separately from ordinary neural event rates.

Use 64-bit sample timestamps, byte offsets and shard/event identifiers. INT16
is for signal samples, not recording positions. Namespace local unit IDs by
probe/epoch and preserve uncertain continuity links rather than force merges.
Bound the number of loaded shard indices too; the current one-file SCB reader
loads its block index, which is a reason to keep shards finite.

## Waveform storage can exceed the input

A 61-sample, 10-channel INT16 snippet is 1,220 bytes. Current SCB v0.2 adds a
20-byte event row and 8 bytes of validity bounds: at least 1,248 bytes/event
before chunk/index overhead and compression. At 20,000 candidates/s this is
24.96 MB/s, already slightly above a 384-channel, 32 kHz INT16 stream's
24.576 MB/s. A permissive threshold and repeated detections of one event on
neighboring channels can amplify this further. Compression ratios must be
measured, not assumed.

Retain metadata for all detector-defined events, with a declared policy for
payload caching. Options to benchmark are budgeted waveform sampling, compact
features, or shared time/channel tiles referenced by multiple candidates to
avoid overlapping snippets. Partial waveform coverage needs a new manifest and
reader contract; do not silently change v0.2's one-waveform-per-event guarantee.
Use local physical channel neighborhoods, with explicit probe/shank boundaries.

## Capacity arithmetic, not a throughput promise

Here TB and GB are decimal. A single sequential read of 100 TB takes at least:

| Sustained storage throughput | Time per read pass |
|---:|---:|
| 1 GB/s | 27.8 hours |
| 5 GB/s | 5.56 hours |
| 10 GB/s | 2.78 hours |

These are assumed effective bandwidths, not measurements of this computer.
Writes, extra passes, decode and sorting add work. A 384-channel probe sampled at
32 kHz in INT16 produces 24.576 MB/s or 2.123 TB/day; 100 TB represents about
47.1 probe-days. Our earlier 14.746 GB sample sorted in 215.15 s, roughly
68.5 MB/s including sorting/export. Linear projection would be about 16.9 days
on one GPU, but global clustering growth and multiday changes make that only a
capacity illustration. Short-sample speed is not evidence of 100 TB throughput.

## Acceptance gates for the next stages

1. Demonstrate a fixed-model streaming assignment path with 1x, 10x and 100x
   virtual durations and bounded host/GPU allocations. Repeated data tests
   engineering scaling, not biological multiday accuracy.
2. Replace growing spike/feature arrays and global reorders with atomic output
   shards. Kill/restart a worker and verify no missing/duplicated core events,
   identical timestamps and bounded recovery work.
3. Test capped, stratified training and novelty updates on independent public
   ground-truth recordings, including drift. Measure rare/transient-unit recall,
   splits/merges, boundary errors and full sorting agreement, not speed alone.
4. Measure cold and warm end-to-end throughput separately, including reads,
   writes, transfers, training and tracking. Record peak RAM/VRAM, bytes written
   per input byte and total passes. Test multi-probe concurrency under actual
   storage contention before projecting multi-GPU performance.
5. Validate continuity across real multiday/flexible-probe recordings. Treat
   publication claims about duration, scale and accuracy as separate evidence
   requirements. The current 600-second sample cannot establish these claims.

The most defensible research contribution is a bounded, reusable detection and
candidate-cache/streaming layer around established model learning, with measured
scaling and accuracy. A new integer sorter is not required to establish it.
