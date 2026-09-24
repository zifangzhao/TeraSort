# Accumulating evidence and guarded adaptation

Implemented a separate learning path around the existing direct CUDA
assignment path. Active templates remain fixed during each shard. Clean,
isolated, high-confidence first-pass assignments supply amplitude-normalized
waveforms to bounded per-unit queues. Unknown events remain in the candidate
bank and do not silently become training labels.

The queues retain at most 64 recent events per unit, 32,768 overall, and
expire events older than 30 minutes of source time. This is not a uniform
30-minute moving average. Very high firing rates may leave evidence from
too few independent cores, in which case promotion is deferred.

Each proposed template blends 5% of an earlier-core median into the active
template. Later cores validate the proposal. Promotions require at least
16 events, eight newer than the previous promotion, at least four training
and four validation events, mean held-out error improvement >=0.5%, relative
shape change <=10%, and a bound on upper-tail error increases. These are
provisional engineering gates, not statistical significance tests or
calibrated neuron-identity confidence. Reusing accepted assignments introduces
selection bias; repeated local validation is not an independent scientific
quality benchmark. No ground truth is supplied to adaptation.

Model versions, bounded evidence, promotion positions and per-unit audit
decisions are checkpointed together in immutable shards. This replaces the
old unconditional first-64-waveform mean update. Failed proposals preserve
the active model. Updates affect subsequent shards; old assignments are not
rewritten. `--freeze-templates` disables evidence collection and adaptation
for controls. Automatic future rollback, new-unit enrollment, and deferred
assignment correction are not implemented by this change.

## Held-out controlled test

Used the same 319 K4-derived seeds, raw input, thresholds and 60–90 second
interval as the preceding fitting pilot, outside calibration windows. Both
runs used three 10-second shards, two-second cores, profiling enabled, and
the RTX 5060 Ti. No other GPU tests ran during these timed passes.

| Metric | Frozen | Guarded adaptation |
|---|---:|---:|
| Recovered units, IoU >=0.8 | 151/250 | 151/250 |
| Spike precision | 75.2173% | 75.2058% |
| Spike recall | 86.0278% | 86.0278% |
| Assigned spikes | 59,142 | 59,151 |
| Matched spikes | 44,485 | 44,485 |
| Split / merged diagnostics | 60 / 66 | 60 / 67 |
| Profiled wall time, excluding calibration/evaluation | 29.884 s | 31.644 s |
| Windows peak working set | 1,597,325,312 B | 1,644,900,352 B |
| Sampled total VRAM peak | 1,648,951,296 B | 1,648,951,296 B |
| Derived shard bytes / raw bytes | 4.985% | 14.762% |

There were 25, 37 and 31 per-unit promotions at the three boundaries, not
necessarily distinct units; the last boundary affects only future data.
This pilot shows operation of the evidence/promotion loop, not improved
sorting accuracy. Nine extra unmatched assignments slightly reduced
precision. A longer drift benchmark is needed before claiming benefit.
Single profiled runs are insufficient for a stable overhead estimate.

State snapshots substantially increase output with frequent short shards.
The existing 5% allowance applies to candidate waveform payload only, not
adaptation checkpoints or total derived output. Five-minute default shards
amortize snapshots, but their total storage ratio has not been measured in
this experiment. State is bounded in RAM as recording duration increases;
checkpoint disk use still grows with the number of shards. Compression or
incremental checkpointing should be measured before large-scale deployment.

Artifacts, all new directories under `F:\sortingDevelopment`:

- `session_adaptive_heldout_20260924_01`
- `session_frozen_heldout_20260924_01`

Each contains manifests, progress, quality reports, run reports, profiles
and immutable shards. The first adaptive pilot stored `audit_json` as an
HDF5 attribute; final code stores it as a dataset to avoid attribute-size
limits with many units. Both layouts are accepted by state restoration.

Final regression suite: **167 passed**. Tests include successful promotion,
rejection of training contamination by later evidence, need for fresh
evidence, cache capacity/expiry, checkpoint round-trip equivalence, and
existing interrupted CPU/CUDA resume and fitting tests. Checkpointed audit
storage and reporting counters were finalized after the timed runs; these
do not change template decisions.

Next experiment should evaluate future-window spike accuracy after each
promotion against the frozen control, especially during drift, and train
novel-unit proposals from an independent bounded unknown-event sample.
Enrollment needs evidence across time plus duplicate/competition checks;
reduced waveform error alone must not establish a new neuron identity.
