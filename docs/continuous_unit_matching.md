# Bounded unit matching primitives

TeraSort can now generate spatially eligible unit pairs and score their template cosine without creating an all-units × all-units matrix. This is a matching **primitive**, not a UnitMatch replacement or a validated cross-day identity pipeline. The [UnitMatch paper](https://www.nature.com/articles/s41592-024-02440-1) and [UnitMatchPy](https://github.com/EnnyvanBeest/UnitMatch/tree/main/UnitMatchPy) remain the scientific baseline. UnitMatch sorts each recording independently, summarizes two waveform halves per unit, corrects drift, and estimates match probabilities from several waveform properties. TeraSort's current sparse scorer computes only a five-temporal-shift cosine.

```python
import numpy as np
from terasort.matching import template_centers, iter_spatial_pairs, score_template_pairs

left = np.load("left/templates.npy", mmap_mode="r")
right = np.load("right/templates.npy", mmap_mode="r")
positions = np.load("left/channel_positions.npy")
left_centers = template_centers(left, positions)
right_centers = template_centers(right, positions)
left_probe_ids = np.zeros(len(left), np.int64)   # Single physical probe example
right_probe_ids = np.zeros(len(right), np.int64) # Use real probe IDs for multiple probes

for pairs in iter_spatial_pairs(left_centers, right_centers, 200,
                                left_probe=left_probe_ids,
                                right_probe=right_probe_ids,
                                pair_batch=4096,
                                max_neighbors_per_unit=256):
    scores = score_template_pairs(left, right, pairs, batch_size=64)
    # Persist candidate edges and scores; calibrate decisions separately.
```

`iter_spatial_pairs` compares only within a physical probe and sorts pair IDs deterministically. The explicit neighbor cap raises `OverflowError` so a dense area can be split or reviewed instead of silently losing pairs. The 200 µm radius is a conservative pilot setting, not a universal threshold. Estimate and apply probe drift or deformation first, and include uncertainty in the search radius. Two sorted chunks from the same continuous source can use halo spike overlap as stronger evidence. Across a file gap or day, require stronger waveform and trajectory evidence and allow unmatched or ambiguous outcomes. Do not link different probes.

On an existing adjacent-window MEArec pilot, a 200 µm spatial gate retained all 558 previously linked overlap pairs and reduced potential pairs from 373,625 to 38,936 (89.6%). A 50 µm gate reduced them to 9,613 (97.4%) but excluded four overlap links. For one 433×427-unit pair, scoring the 19,233 sparse candidates within 200 µm took 3.12 seconds on CPU and differed from the saved dense CUDA cosine by at most 8.35×10⁻⁷. This retrospective check does not measure total sorter speed, UnitMatch's six metrics, match accuracy, or flexible-probe drift.

## Causal rolling template state

`RollingTemplate` is an experimental per-unit state for online assignment. With defaults it keeps six five-minute bins, an immutable anchor, and the last accepted template. Each bin stores a float64 waveform sum and spike count; the rolling mean caps each bin's effective count so one high-rate period cannot dominate. A 61-sample × 16-contact template needs about 55 kB per active unit for these arrays. This memory is independent of recording duration, provided dormant states are checkpointed outside the active set.

```python
from terasort.matching import RollingTemplate

state = RollingTemplate(initial_waveform, sample_rate_hz=20_000,
                        coordinate_frame="probe1:motion_aligned:v1")
frozen = state.snapshot()
# Assign the next source core using frozen.waveform, then durably write outputs.
state.commit(start_sample, stop_sample, mean_waveform=accepted_mean,
             spike_count=accepted_count, confidence=assignment_confidence,
             coordinate_frame=frozen.coordinate_frame,
             expected_version=frozen.version)
checkpoint_payload = state.state_dict()  # External writer publishes atomically.
```

Summaries with spikes must fit inside one five-minute bin; split a crossing core upstream. An empty or low-confidence interval advances time without changing the template. When all recent bins expire, the last waveform is frozen and `snapshot().active` becomes false. Stale versions, overlapping intervals, changed channel/motion frames, and malformed waveforms are rejected. Confidence is a caller-supplied policy value, **not a calibrated UnitMatch probability**. Update only from high-confidence, motion-aligned spikes, and compare the rolling waveform with the anchor to detect sudden identity or probe changes. The class itself performs neither spike assignment nor drift correction.

To support arbitrary recording durations, finish each probe/epoch with an immutable compact per-unit summary, keep 64-bit sample coordinates, and store spike shards separately from the global unit-ID mapping. Match new epochs against recent tracks and spatially indexed compact dormant prototypes; persist candidate edges, evidence, model version, and ambiguity in a checkpointed ledger. Processing memory can then depend on one raw/GPU chunk, active unit summaries, and a fixed pair batch instead of total samples or historical pair count. Disk and elapsed time still grow with recording duration.
