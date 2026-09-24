# Experimental streaming template assignment

`terasort.streaming` is a bounded first assignment pass over an existing SCB
0.2 waveform bank. It is separate from the validated Kilosort-compatible full
sort. The original SCB file is opened read-only; output is a new atomic HDF5
shard and an existing output is never overwritten.

The current path is:

1. Read threshold candidates and their sparse, synchronous local waveforms in
   bounded SCB batches. SCB's saved detection floor and preprocessing remain
   authoritative; a lower floor requires reading the raw recording again.
2. Group nearby same-polarity detections that share local contacts. The output
   retains every contributing SCB row ID. Grouping can fuse two overlapping
   neurons, so these are *putative events*, not guaranteed spikes.
3. Compare each event only with templates anchored on its local contacts. The
   default cap is 32 templates per anchor contact, with an explicit overflow
   error. Global unit IDs are independent of the anchor contact.
4. Assign a unit only if the best waveform cosine exceeds 0.88 and leads the
   runner-up by at least 0.03. Otherwise `unit_id=-1`. Only isolated events
   with cosine at least 0.96, margin at least 0.08, plausible fitted amplitude,
   and complete matching contact support update the rolling template.
5. Freeze templates while assigning an epoch; append results durably, then
   commit high-confidence waveform summaries to the 30-minute rolling state.

Seed templates must be shaped `unit × waveform_sample × recording_channel`,
unwhitened, in the same channel order and preprocessing frame as the SCB
waveforms. A Kilosort `templates.npy` is **not automatically compatible** with
an independently filtered SCB bank; verify scale, referencing, sample alignment,
and channel order before using it. Omitting `--templates` is supported for
consolidation/format testing: every event remains unknown and no units are
learned.

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m terasort.streaming `
  --bank F:\data\segment.scb.h5 `
  --templates F:\data\calibration\templates.npy `
  --output F:\data\segment.assignments.h5 `
  --backend cuda --slots-per-anchor 32
```

`--start-sample` and `--stop-sample` select a half-open source-clock interval
for a short trial. `--backend cpu` is the independent reference. `--floor-snr`
can raise, but cannot lower, the stored SCB floor.

The output `events` table contains the earliest contributing detection sample,
the strongest anchor's original sample/channel, signed voltage and SNR,
assigned global unit ID or `-1`, best/runner-up scores, and the frozen template
version used. `candidate_start:candidate_stop` indexes the flat
`candidate_rows` dataset of original SCB row IDs. `final_templates` stores
the last local waveform, contact map, anchor, version and assigned-event count.
The `complete` attribute is set only after successful finalization. Interrupted
`.partial` output is not yet resumable; a new output path is required for a retry.

## GPU execution and measured scope

The CUDA backend uses a direct NVRTC kernel, not Torch matching operations. A
warp scores one event/template pair across local contacts and five temporal
shifts. Only spatially eligible pairs are launched; there is no all-template
pair matrix. Templates stay on the GPU across epochs, and changed versions are
refreshed selectively. A bounded two-epoch reader queue overlaps HDF5 decoding
and cross-channel grouping with matching. The event batch cap is 2,048 by
default; queues and candidate counts have explicit bounds.

On the RTX 5060 Ti, a warmed synthetic *matching-only* benchmark with 2,048
events, 32 templates per anchor and 16 contacts produced 65,536 eligible pairs:

| Matcher | Time |
|---|---:|
| CPU reference | 5.876 s |
| Direct CUDA, median of five repeats | 0.0341 s |

The assigned IDs agreed and maximum cosine-score difference was 3.0e-7. This is
about 172× for this matching stage, not for spike detection or a full sort.
Reproduce with `python scripts/benchmark_streaming_match.py --events 2048`.

A separate 9-second, 32-channel Intan SCB excerpt had 47,764 threshold rows and
44,560 grouped events. With 49 noisy exemplar seeds drawn from the preceding
second, the complete CPU pass took 20.48 s and CUDA pass 2.32 s. Event times,
source row links and unit IDs were identical; maximum score difference was
3.6e-7. **Neither pass assigned a unit at the conservative score threshold.**
These seeds were a performance fixture, not identified neurons; this result
does not validate sorting accuracy. Output and trial seeds are under
`C:/Frank/Code/PainProject/outputs/streaming_sort_trial_20260924/`.

CuPy's default profile cache stalled NVRTC on this Windows machine. The matcher
uses a local temporary cache unless `CUPY_CACHE_DIR` is already set. In an
embedding process that imported CuPy earlier, set `CUPY_CACHE_DIR` to a writable
local directory before importing CuPy.

## Scientific and scale limits

- Threshold candidates do not include every biological spike. Local grouping
  can fuse collisions, and the current matcher does not perform residual
  subtraction or recover subthreshold multi-contact spikes.
- The bank is seeded from a calibration sorter; it does not discover late or
  novel neurons yet. A strict 32-slot overflow fails visibly rather than
  dropping units. Anchors are currently fixed even if a unit moves between
  contacts; global identity migration needs a separate policy.
- SCB 0.2 caches one waveform per threshold row, which can exceed raw-data
  volume at permissive thresholds. Shared tiles or selective waveform payloads
  need a new format version; this first pass does not change SCB semantics.
- Output is bounded per input shard but not restartable mid-shard. Multiday
  identity links, probe deformation, artifact handling, and data-derived
  template discovery remain separate validation work.

The next scientific gate is an SCB candidate cache and aligned calibration
templates from the **same** public ground-truth recording. Compare recall,
precision, unit splits/merges, overlap recovery, drift continuity, bytes per
hour, and end-to-end throughput against Kilosort4. Do not infer those results
from the synthetic matching benchmark.
