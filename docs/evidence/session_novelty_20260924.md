# Bounded provisional enrollment — 2026-09-24

Added optional novelty modes to `terasort session-sort`:

- `--novelty off` (default): unchanged dense assignment/adaptation path.
- `--novelty shadow`: collect and assess proposals without adding units.
- `--novelty enroll`: permit experimentally qualified proposals into future shards.

Existing IDs remain unchanged. New IDs append to the probe/day-local bank;
template versions invalidate the CUDA cache. Proposals, evidence and decisions
are checkpointed with model state. The existing evidence budget is resized
when units are appended, keeping its total cap intact. Frozen templates and
active enrollment are mutually exclusive.

## Screening and bounds

Only clean unknown neighborhoods with SNR >=6 are considered. Candidates
near assigned waveforms are excluded. Local dominant contacts and temporal
peaks suppress repeated proposals from adjacent detector channels. Metadata
filtering precedes deterministic source-coordinate sampling, so known events
do not consume the 256-waveform observation budget per core.

There are at most 128 pending proposals, four per anchor, 64 waveforms per
proposal, and a 30-minute evidence lifetime. A fixed initial reference groups
similar observations. At least 24 observations across three cores are needed;
earlier cores train a median and later cores supply at least six validation
observations. Held-out similarity, physical-contact duplicate comparison with
existing templates, and a sampled refractory screen determine eligibility.
At most eight units can be added per shard, with 4,096 total units and 512
templates per contact as hard enrollment limits.

These are experimental screening thresholds, not calibrated probabilities
or proof of neuronal identity. Sampled refractory checks miss violations
between uncached spikes. Single-reference clustering, fixed proposal caps,
anchor changes during drift and conservative collision exclusion limit
coverage. A full pending bank rejects new prototypes until entries expire;
all detected candidates still retain metadata in the normal output.

## Tests and recording experiments

**175 tests passed**, including a raw synthetic recording with an omitted
unit. Both CPU and CUDA paths enroll that unit, assign more than 40 later
spikes to its appended ID, and preserve exact assignments/model state after
a crash following enrollment and restart. Other tests cover duplicate
rejection with permuted physical contacts, bounded state, expiration,
checkpoint replay and rejection of evidence from only one core.

Used the same public 600-second/250-unit/384-channel recording, bounded
319-template seed bank, and RTX 5060 Ti as the earlier pilots. Ground truth
is used only for post-run evaluation. Sources and prior results remain intact.

An initial 30-second shadow trial exposed ineffective sampling before
metadata filtering (only seven proposals). After correcting the ordering,
the 60–90 second enrollment-mode test collected 83 proposals; the strongest
had 15 observations, so none enrolled. Assignment metrics were unchanged
from guarded adaptation: 151/250 recovered, 75.2058% precision, 86.0278%
recall. Profiled wall time was 32.92 seconds, excluding calibration and
evaluation. This is a diagnostic timing, not a controlled speed claim.

The longer test covered **60–150 seconds**, three 30-second shards, with
thresholds unchanged. It reached the 128-proposal cap. Of these, 127 lacked
enough repeated evidence; one had 40 observations and passed the waveform
consistency screen but matched an existing template at **0.9885 cosine**.
It was rejected as a duplicate. No units were enrolled.

The duplicate-like proposal's anchor was channel 94; a bounded inspection
identified existing unit 313, amplitude ratio about 0.997 and central cosine
about 0.998 for the median waveform. This suggests a fitting/competition
problem worth investigating; median similarity does not establish why
individual events failed assignment.

For that longer interval, recovery was **154/250**, precision **75.4635%**,
recall **86.0073%**. The historical K4 reference on the same interval recovered
229/250 with 69.9986% all-unit precision and 98.3233% recall. These cannot
be compared as an improvement over the earlier 30-second recovery count,
because the evaluation duration differs. There is no demonstrated quality
gain from novelty enrollment on this recording.

The 90-second run took 100.43 seconds with profiling, excluding calibration
and evaluation. Windows peak working set was 1,676,615,680 bytes; sampled
total VRAM peaked at 1,648,951,296 bytes. Three shards occupied 208,909,547
bytes, 9.445% of covered raw bytes. Evidence checkpoints remain outside the
5% candidate waveform payload allowance; their storage cost is not solved
by this change. No full-session, multiday, drift or 100 TB gate is established.

## Artifacts

All directories are under `F:\sortingDevelopment`:

- `session_novelty_shadow_20260924_01`: initial sampling diagnostic.
- `session_novelty_enroll_20260924_02`: corrected sampling, 30-second trial.
- `session_novelty_enroll90_20260924_01`: 90-second evidence accumulation.

Final code also records proposal source ranges, train/validation counts and
supporting source samples on enrollment. These audit-only fields were added
after the recording runs; final tests cover them. Raw recordings were not
modified. Next priority is investigating duplicate-template competition and
bounded reassessment of rejected existing-unit events, while keeping novel
enrollment optional.
