# Template learning versus fitting diagnostic — 2026-09-24

**Result: both stages limit quality.** Swapping the identity source while
holding waveform reconstruction and the matcher fixed increased recovered
units from 155 to 189, but reduced precision. The full Kilosort4 pipeline
still recovered 227. This diagnostic does not establish an independent
quality improvement or a new production configuration.

## Matched reconstruction

Added `scripts/diagnose_session_templates.py`. It reads existing K4 outputs
and reconstructs waveforms from the original raw data in the exact session
preprocessing frame. It does not feed whitened K4 templates directly into
the matcher. K4 templates are unwhitened one at a time only to identify
physical anchor contacts. Channel geometry is checked against the manifest.

Two banks were reconstructed:

1. Identities from the previous 60-second bounded K4 preview.
2. Identities from the historical K4 sort of the full 600-second recording.

Both use the same source windows: 0–20, approximately 300–320, and 580–600
seconds. Preview-clock K4 times are explicitly mapped back to source time.
Each unit must have at least 20 detections in those windows; at most 48
evenly spaced waveforms are selected, with at least ten clean snippets
required after QC. Both apply the same median, rank-three denoising,
61-sample/16-contact representation and peak/contact canonicalization.
No ground truth is used in reconstruction or assignment.

The preview bank retained 319 identities; the full-recording bank retained
468. All 227 identities recovered by K4 on the evaluation interval were
present in the reconstructed full bank. Both read 1,617,100,800 raw source
bytes including halos, without copying the complete recording. Total
waveform count is capped at 32,768; the full bank used 21,330 snippets.
Per-unit sampling is matched, but total samples and bank cardinality differ.
Different historical K4 learning configurations may also contribute; this
is a bank-source swap, not a pure experiment on calibration duration alone.

## Fixed matching comparison

Evaluation covers source seconds 60–90 on the existing 250-unit public
ground-truth recording. Both runs freeze templates, disable novelty, use
three residual passes, two-second cores, ten-second shards, the same
thresholds and the current fused CUDA detector. Ground truth is used only
by the post-run evaluator.

| Identity source / sorter | Seed units | Recovered at IoU >=0.8 | Precision | Recall |
|---|---:|---:|---:|---:|
| Matched 60-second preview bank + our matcher | 319 | 155/250 | 75.5456% | 86.1477% |
| Matched full-recording bank + our matcher | 468 | 189/250 | 68.6781% | 89.7795% |
| Historical full Kilosort4 pipeline | — | 227/250 | 69.9195% | 98.3079% |

The previous production-style bounded seed pilot recovered 151; it used a
different waveform cap. Use **155**, not 151, as this experiment's matched
control. The full bank gains 37 recovered units and loses three relative
to that control, for a net gain of 34. It adds 1,878 matched spikes, but
also many unmatched assignments: total assigned spikes increase from
58,967 to 67,598. More templates are not an unconditional quality benefit.

With the full bank, 189 ground-truth units have IoU >=0.8, 38 have IoU
0.5–0.8, 20 have IoU 0.2–0.5, and three have IoU below 0.2. Forty-six units
recovered by K4 remain unrecovered by our route, while our route recovers
eight that fall below K4's recovery threshold. Thus the net count gap of
38 is not the exact number of K4-recovered identities still missed.

Precision includes all output clusters, including noise, and uses the
existing custom evaluator. No good-only K4 filter or independent
SpikeInterface cross-check was introduced. Both runs remain below the
replacement gate. The full identity bank incorporates evaluation/future
information from K4's full-recording learning: it must not be reported as
independent generalization, bounded production calibration, or a 100 TB
solution.

## Interpretation and next experiment

Improved identity learning accounts for a substantial recovery increase
with fixed matching. However, every K4-recovered identity is represented
in the full bank and a large recall gap remains. That remaining loss may
come from waveform reconstruction, local channel support, noise weighting,
template competition, thresholding, temporal alignment or collision fitting;
this experiment cannot assign it solely to the CUDA fitting kernel.

Prioritize two separate follow-ups:

- Improve bounded calibration and template selection, measuring precision
  as well as recovery; adding more noise/duplicate templates is costly.
- Keep the full identity bank as a diagnostic control while testing local
  waveform reconstruction/noise weighting and overlap fitting. Inspect
  failed K4-recovered units before changing global thresholds.

Production sorting logic was not changed by this diagnostic. Source data
and historical outputs remain read-only. No automatic bank replacement or
unit merging was performed.

## Artifacts and validation

New directories under `F:\sortingDevelopment`:

- `session_diagnostic_full_seeds_20260924_01`
- `session_diagnostic_preview_seeds_20260924_01`
- `session_diagnostic_full_match_20260924_01`
- `session_diagnostic_preview_match_20260924_01`

Seed reports preserve source windows, K4-to-local identity mappings, support
counts and reconstruction metadata. Matching directories contain manifests,
quality reports, profiles, progress and immutable shards. Both reconstructions
and both GPU runs completed successfully. Additional assertions verified
finite waveforms, anchor/contact consistency, per-unit/global sample caps,
and identical source windows/read coverage. The new diagnostic is limited
to a single probe/source starting at sample zero; it rejects unsupported
clock configurations. It is not a general multiday importer.

Single profiled matching wall times were 36.74 seconds (full bank) and 41.87
seconds (preview bank). These are not a stable performance comparison;
the experiment's conclusion concerns quality. Both sampled total VRAM
peaks were 1,330,184,192 bytes, and Windows peak working sets were about
1.60 GB. The full-bank CuPy pool peak was 116,969,472 bytes.
