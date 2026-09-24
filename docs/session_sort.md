# Experimental bounded session sorter

`terasort session-sort` reads ordered INT16 time-major files directly and
publishes immutable per-probe shards. It is an opt-in research path. The
existing `terasort sort` Kilosort4 route remains the production baseline until
the quality suite passes.

Install with the pinned environment in the main README, then create a manifest:

```json
{
  "schema_version": 1,
  "session_id": "example",
  "probes": [{
    "probe_id": "probeA",
    "sample_rate_hz": 20000,
    "gain_uv_per_count": 0.195,
    "geometry": {
      "x_um": [0, 32], "y_um": [0, 20], "shank": [0, 0]
    },
    "bad_channels": [],
    "segments": [
      {"path": "day1.bin", "start_sample": 0,
       "n_samples": 1200000, "day_id": "day1"},
      {"path": "day2.bin", "start_sample": 1300000,
       "n_samples": 1200000, "day_id": "day2"}
    ],
    "gaps": [{"start_sample": 1200000, "stop_sample": 1300000,
              "reason": "acquisition stopped"}]
  }]
}
```

The geometry arrays use acquisition channel order. Each source file must be
exactly `n_samples * n_channels * 2` bytes. Sample positions are signed 64-bit
coordinates within a probe's manifest. Every discontinuity must be declared
as a gap. Multiple probes can each have their own ordered source list. Paths
may be absolute local/UNC paths or relative to the manifest. Source files are
opened read-only; no full local staging copy is made. The manifest and source
sizes/timestamps are pinned in `run.json` so a changed source cannot be
silently resumed.

For larger sequential network reads, opt into `--read-buffer-mb 64` (MiB).
This groups adjacent cores into one bounded read and avoids rereading their
shared halos. Core boundaries, preprocessing and event coordinates stay
unchanged. `--prefetch-depth 2` controls the raw-core queue; values up to 64
are supported. The read buffer accepts 0–1024 MiB and must fit one core plus
both halos. Zero keeps the original per-core reader and is still the default.

Queued cores own small copies, so they cannot retain older large read buffers.
Budget the configured read buffer, roughly `(prefetch_depth + 2)` haloed raw
cores, and the sorter's other working memory. Retries can temporarily retain
an additional partial read. Larger buffers trade RAM and copying for fewer
requests; they help only when source delivery limits processing. These options
apply to dense sorting; calibration retains its separate bounded reader.
Keep the same nondefault buffer/queue settings when resuming a run. Old runs
using the defaults remain compatible with their original checkpoint digest.

```powershell
terasort session-sort --manifest C:\data\session.json --output-root F:\sorts\session-1
terasort session-sort --manifest C:\data\session.json --output-root F:\sorts\session-1 --resume
```

The default is one CUDA GPU with a 12 GiB CuPy pool limit, two-second cores,
100 ms preprocessing halos, and five-minute shards. `--backend cpu` is a
reference path. Both use the same per-shank median reference and 300–6000 Hz
zero-phase Butterworth frame. Calibration samples bounded one-minute windows
at the start, middle and end of each day (and roughly every two recorded
hours on long days), stores at most 64 snippets per anchor and 32,768 total, and
trains provisional local templates. If you supply `seed_templates` (or
`seed_templates_by_day`), each `.npy` must have shape
`unit × 61 samples × recording channels` **after the exact session
preprocessing frame** and the manifest must include
`"seed_preprocessing_id": "terasort-session-v1"`. Kilosort's `templates.npy`
cannot be used directly without aligning its preprocessing, channel map,
waveform scale, and time center.

The refined matcher uses a central 17-sample cosine proposal, then fits the
amplitude against all 61 samples with inverse-noise-variance channel weights.
It searches shifts of up to two samples and ranks eligible proposals by
residual energy reduction. Ambiguity is the relative gain margin between
the best two eligible templates. A 0.5 ms same-unit exclusion reduces
duplicate assignments. Seed peaks and dominant contacts are canonicalized
without merging unit IDs. Candidate coordinates retain the original
detection position; fitted spike coordinates may shift by two samples.
The dense pass still uses direct CUDA kernels, bounded local pair batches,
two-second cores, residual subtraction, and the existing shard checkpoints.
Detection now fuses absolute-value normalization, temporal peak checks and
bounded candidate packing in one CUDA kernel. It avoids the full SNR matrix
while preserving threshold and tie rules. Candidate-buffer overflow remains
an explicit error. `--residual-passes` accepts 1–12 (default 3); optional
passes after the third require central cosine >=0.85, or the configured
score floor if higher. Extra passes cost time and can add false assignments;
they have not passed the replacement quality gate.

An optional research calibration script learns identities with pinned
Kilosort4 on three windows spread across a day, then reconstructs local
waveforms using the exact session preprocessing frame:

```powershell
.venv\Scripts\python.exe scripts\learn_session_seeds.py --manifest C:\data\session.json --output-root F:\sorts\seeds-1 --probe-id probeA --day-id day1 --budget-seconds 60
```

Set that probe's `seed_templates` to the resulting `seeds.npz` and
`seed_preprocessing_id` to `terasort-session-v1`. For multiple days use
`seed_templates_by_day`. This script requires a new output directory and
retains the preview for inspection. Its budget is 6–180 seconds, with a
5 GiB raw preview cap, at most 4,096 retained units and 32,768 extracted
waveforms. Kilosort runs only on that preview; the dense pass uses the CUDA
session matcher. This is optional, not the default automatic calibration.
Its fixed three-window coverage can miss rare or temporally localized units.
See [fitting evidence](evidence/session_template_fitting_20260924.md) for
the measured improvements and remaining quality gap.
The [matched template-bank diagnostic](evidence/session_template_diagnostic_20260924.md)
separately tests bounded-preview versus full-recording K4 identities with
identical waveform reconstruction and our unchanged matcher. Full-recording
identities are diagnostic inputs, not a validated production calibration path.

Template adaptation now keeps recent clean, isolated, high-confidence
waveforms across shards. The cache holds at most 64 per unit and 32,768
overall, with a 30-minute age limit. It is a recent-event cache, not a
uniform 30-minute average; high-rate units can have much shorter coverage.
Poor-quality contacts and intervals cannot contribute evidence.

At shard boundaries a robust median proposes a 5% update. At least 16
samples, eight samples newer than the previous promotion, and two distinct
cores are required. Earlier cores train the proposal; later cores test it.
The mean held-out waveform error must improve by at least 0.5%, with
limits on shape change and the upper tail of error increases. These are
experimental engineering thresholds, not calibrated identity probabilities.
They do not remove selection bias from learning on accepted events.

Each shard checkpoints the rolling evidence, last-promotion positions, and
per-unit decisions under `adaptation`. This adds storage beyond the 5%
candidate waveform-cache allowance; that allowance is not a total-output
cap. Frequent short shards amplify checkpoint overhead. The normal shard
duration remains five minutes. Updates affect future shards only; previous
assignments stay immutable. Restarts restore the evidence as well as the
model. Runs made before this update require a new output root because the
adaptation configuration changed.

Use `--freeze-templates` to disable adaptation for controlled comparisons.
The guarded adaptation path does not automatically roll back promotions
after future deterioration or rewrite prior assignments.

Optional `--novelty shadow` accumulates unknown-event proposals without
changing assignments. `--novelty enroll` allows eligible proposals into
future shards as experimental probe/day-local units. The default is `off`;
enrollment cannot be combined with `--freeze-templates`. This is not yet
validated as a quality improvement.

Novelty collects at most 256 candidate waveforms per core, 128 proposals
overall, four per anchor, and 64 waveforms per proposal. Entries expire
after 30 minutes. Assigned neighborhoods and QC failures are excluded before
deterministic sampling. Each proposal needs at least 24 observations in
three cores, including 12 training and six later validation observations.
Physical-contact waveform similarity checks help reject duplicates of
existing units, including small timing shifts. Sampled refractory violations
are a screening heuristic, not a full unit contamination estimate.

At most eight proposals can be enrolled per shard, subject to 4,096 active
units and 512 units per contact. Existing IDs are never renumbered. Evidence
queues shrink when the bank grows so their total budget stays bounded.
The `novelty` checkpoint records pending proposals and all enrollment
decisions; completed assignments are not retroactively corrected. Novelty
requires seed width compatible with the session's local contact map.
Its checkpoint waveforms add to storage outside the candidate payload cap.

Dense sorting still uses the same CUDA kernels. Proposal collection and
validation currently run synchronously on CPU between bounded core/shard
operations. The prototype cap can fill with weak proposals until expiration;
drift across anchor contacts and overlapping unknown units need further
work. Waveform agreement alone cannot establish neuron identity.

For each probe and source interval, a completed `.h5` contains every
first-pass and residual-pass threshold candidate after duplicate removal,
including unassigned events (`unit_id=-1`), matched spikes, QC flags and
noise per core, selective INT16 waveform payloads, and the model checkpoint.
The candidate format marker is `SCB0.3`; existing SCB0.2 files keep their old
meaning. Cached waveform rows are indexed by `waveform_candidate_row` and
limited to 5% of raw core bytes by default; candidate metadata remains
complete regardless of cache coverage. A `.h5.partial` is discarded and
reprocessed during `--resume`; completed shards are never rewritten. The
run also writes unresolved cross-day link evidence in
`cross_day_links.json`. Shard `telemetry_json` records wall time, bytes read,
process RSS and CUDA memory peaks.

The sorter refuses a core when its candidate count or event/template pair
count exceeds the configured bound; this is an explicit failure, not silent
truncation. Lower `--core-seconds` and resume if an artifact burst exceeds the
bound. The CPU implementation and CUDA implementation are experimental and
have not been shown equivalent on real data. There is no validated motion
correction, novelty-driven retraining, or online mode yet. A checkpoint
currently contains template and assignment state; motion state is absent.

## Ground-truth gate

### Accuracy screening tools

`scripts/sweep_session_accuracy.py` runs a finite sequential screen of fit
score, ambiguity margin, score-window width, rescue floor, overlap policy,
spatial masks, SVD denoising and a previously refined bank. It writes its
plan before executing, preserves each run's logs and outputs, and evaluates
selected settings on a separate source interval. Ground truth ranks
configurations, but does not enter detector or template-transform code.
The pilot uses development seconds 60–90 and validation seconds 420–450
of the retained 600-second dataset. It is not a general cross-validation
framework or an exhaustive search over sorting algorithms.

`--half-width` controls the center scoring window (3–30 samples per side,
default 8). Full 61-sample amplitude/residual fitting remains in effect.
Changing the score window changes acceptance behavior as well as cost.

`--refit-rounds 1` or `2` enables an experimental CPU post-pass on either
backend. Each round revisits at most 64 uncertain assignments per core,
subtracts accepted neighbors from a local patch, then tests local competing
templates and amplitudes. It requires score >= 0.85, margin >= 0.1 and at
least 5% gain improvement over the current fit, alongside ordinary amplitude,
refractory and gain gates. It preserves event count, can change unit/time/
amplitude, skips core edges, and does not globally optimize all overlaps.
Its decisions and template changes are not replayed by the failure-trace
script, which rejects this mode. Default is off.

`scripts/learn_session_pca_splits.py` tests actual spike-feature PCA using
the retained bounded refinement cache. It fits a three-component basis on
training snippets, proposes two clusters, and checks independent temporal
validation snippets before adding at most 16 children. Both clusters need
support across multiple windows, distinct waveform shapes and reduced
held-out error. It can legitimately propose no splits. This is distinct
from per-template low-rank SVD denoising; neither is established as a quality
improvement merely by its inclusion in the experiment.

`scripts/extend_session_accuracy.py` validates the development screen's best
F1 point and tests local refitting, higher score floors and the PCA split
bank if it changed, plus a temporal detector. Scripts require new output directories. See
[accuracy search evidence](evidence/session_accuracy_search_20260924.md)
for tested scope, held-out results and limitations.

`--detector-mode smooth3` is an optional CUDA experiment that proposes
candidates from a [1,2,1]/4 temporal smoothing filter. Noise is re-estimated
per channel and residual pass using the median absolute deviation of about
4,000 uniformly spaced samples. Raw-noise template fitting and the primary
residual-gain gate stay unchanged. This mode allocates additional bounded
GPU temporaries; it is not a zero-cost or custom fused detector. Candidate
SNR now refers to the filtered per-pass noise estimate, recorded in the run
configuration, while QC noise and saved voltage remain in the original
preprocessing frame. The default `raw` mode is unchanged.

`scripts/combine_session_accuracy.py` tests smoothing with stricter fit
acceptance and, if development F1 improves over a matched-score raw detector,
compares both against reference on a fresh interval. It also checks the
PCA split bank independently. These dataset-specific experiments are
separate from the production quality gate.

### Optional low-threshold rescue pass

CUDA sessions can append one guarded residual pass with
`--rescue-floor-snr 3.5`, leaving the primary `--floor-snr 4.5` and three
normal passes unchanged. The rescue pass requires center score >= 0.85,
relative margin >= 0.1, and the primary residual-gain floor (4.5 squared
with default settings). It keeps the ordinary amplitude and refractory
checks. A stronger configured score/margin threshold remains in effect.
It can execute even when primary detection found no candidates.

```powershell
terasort session-sort --manifest C:\data\session.json --output-root F:\sorts\rescue-1 --freeze-templates --rescue-floor-snr 3.5
```

The option defaults to off and currently requires CUDA. It uses the existing
CUDA detector and matcher with the same per-core candidate/pair limits;
overflow remains an explicit error. Added candidates retain metadata and
uncertain candidates remain unassigned. Additional metadata and another
pass cost storage and time even when waveform caching stays capped.

With frozen models, the rescue pass preserves primary assignments and adds
only final-pass assignments. Template adaptation or novelty enrollment can
change subsequent cores, so that preservation is not a guarantee across
an adapting session. The option and algorithm version are included in the
resume configuration; use a new directory for older experimental runs.

`--rescue-floor-snr 4.5` supplies an extra-pass control with the same stricter
fit gates. In contrast, changing `--floor-snr` globally changes both primary
candidate detection and the residual-gain gate; it is not an isolated
detector experiment. See [rescue evidence](evidence/session_detection_rescue_20260924.md)
for measured quality and resource tradeoffs.

For compute/quality experiments, `--rescue-passes` accepts 1–4 (default 1),
and `--shift-radius` accepts 0–8 samples (default 2). Rescue passes only run
when `--rescue-floor-snr` is supplied. Each pass refits the current residual;
processing stops early if no more fits are accepted. The timing radius is
used by both CPU and CUDA matching, while rescue remains CUDA-only.
These bounded settings and the updated algorithm version are recorded in
the run configuration, so older checkpoints require a new output directory.

An experimental higher-compute setting is:

```powershell
terasort session-sort --manifest C:\data\session.json --output-root F:\sorts\compute-test-1 --freeze-templates --residual-passes 6 --rescue-floor-snr 3.5 --rescue-passes 3 --shift-radius 4
```

More search can increase false assignments as well as recall. These settings
are not a validated high-accuracy preset; see the
[compute/accuracy experiment](evidence/session_accuracy_compute_20260924.md).

### Bounded refinement experiment

`scripts/refine_session_templates.py` tests repeated assignment and waveform
refitting with stable unit IDs. Supply a single-probe manifest, a seed bank
in the session preprocessing frame, and explicit two-second window starts:

```powershell
python scripts/refine_session_templates.py --manifest C:\data\session.json --seed-templates F:\sorts\seeds.npz --output-root F:\sorts\refinement-1 --training-seconds 40 100 220 340 460 540 --validation-seconds 44 224 464 --rounds 2
```

The example requires a sufficiently long recording. Window cores and halos
must be disjoint and cores must fit in individual source segments without gaps.
The pilot accepts at most 32 windows and a default 1.25 GiB scratch budget.
It reads and preprocesses windows once, then reuses the local cache. It writes
`round_0.npz`, `round_1.npz`, `round_2.npz` and per-template audit files to a
new directory. The original bank remains available for rollback.

Each round reassigns all learning windows using the current CUDA bank and
collects deterministic per-unit, per-window evidence (at most 32,768 total
waveforms across the two partitions). Evidence requires first-pass score >=
0.9, margin >= 0.1, fitted amplitude in [0.5, 2], usable contacts, and no
nearby accepted fit sharing contacts. Undetected collisions may still exist.
Templates move 25% toward the amplitude-normalized training median only if
independent validation waveforms improve, change stays <= 10%, and alignment
stays fixed. This waveform gate is a surrogate for clustering quality.

Evaluate every bank with frozen templates on a separate interval excluded
from both learning partitions. Ground truth must never enter refinement.
This experiment does not split, merge or enroll units, does not reread the
full session for each learning round, and does not change default sorting.
See [refinement evidence](evidence/session_refinement_20260924.md).

### Experimental overlap scheduling

CUDA runs can opt into `--overlap-policy interference` (default: `strict`).
This permits a confident candidate (score >= 0.8, margin >= 0.1) to share a
residual pass with another fit when cumulative, amplitude-weighted template
projections remain below 1% in both directions. The projection limit is not
an identity-confidence probability. A previously accepted weak fit does not
automatically veto a strong, weakly interacting candidate.

The scheduler caches at most 32,768 shifted template couplings per core.
Cache saturation defers uncached overlapping fits. At most 32 subtraction
groups are allowed per pass; fits sharing waveform samples and contacts
execute in separate groups to preserve deterministic subtraction. Dense
detection and fitting remain CUDA; this scheduling step currently runs on CPU.
The CPU backend does not expose this option because it already fits sequentially.
The run configuration records the scheduler version and prevents resuming
with a different configuration. Start a new output directory for older
experimental scheduler versions.

The initial two 30-second comparisons show small precision/recall gains,
mixed unit recovery, and extra runtime. See
[overlap evidence](evidence/session_interference_20260924.md). Keep the
strict default until broader quality gates pass.

Create a suite JSON with at least the existing 600-second 250-unit ground-truth
recording plus two independent public/hybrid cases covering drift and
collisions. Each case points to a ground-truth `.npz` with `times`, `labels`,
and `sampling_frequency`, this sorter output, and Kilosort4's output directory
containing `spike_times.npy` and `spike_clusters.npy` from the **same raw input
and geometry**:

```json
{"cases": [
  {"name": "600s-250-unit", "kind": "base",
   "ground_truth": "C:/data/ground_truth.npz",
   "session_output": "F:/sorts/session-1", "probe_id": "probeA",
   "kilosort_dir": "F:/sorts/kilosort4"}
]}
```

Run `terasort session-quality --suite suite.json --report quality.json`.
The command exits 0 only when at least three cases include drift and collision
cases and all declared recovery, precision and recall loss limits pass;
otherwise it exits 2. It reports splits, merges, collision recovery, and
rare/late unit recovery as diagnostics. Spike matching uses a 0.4 ms
tolerance and measures only completed source-clock spans. Treat
SpikeInterface comparison as an independent publication cross-check.

The present code and small synthetic tests do not establish the quality gate,
flat 1×/10×/100× resource use, or a 100 TB capability claim. Those require
the planned datasets and real long-run measurements. Do not promote this
experimental path over Kilosort4 based on the synthetic tests alone.

`python scripts/benchmark_session_scale.py --backend cpu` runs a small
1×/10×/100× repeated-source memory smoke test and prints peak RSS, bytes,
and wall time. It deliberately reuses one short file and is not evidence of
remote I/O or a real 100 TB run. Use `--output-root` with a new directory to
retain its inputs and shards for inspection.
