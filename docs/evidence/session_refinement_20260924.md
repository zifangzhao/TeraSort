# Repeated bounded template refinement, 2026-09-24

## Result

Two refinement rounds produced a small improvement on a separate 30-second
interval, with diminishing returns and mixed split/merge diagnostics. This
does not establish a generally better clustering algorithm. Default sorting
is unchanged and the original seed bank remains available for rollback.

| Refinement rounds | Recovered units / 250, IoU >= 0.8 | True-positive spikes | Precision | Recall | Split / merged units |
|---|---:|---:|---:|---:|---:|
| 0 | 154 | 43,871 | 75.1744% | 85.9373% | 59 / 71 |
| 1 | 155 | 43,880 | 75.1949% | 85.9549% | 60 / 69 |
| 2 | 155 | 43,883 | 75.1988% | 85.9608% | 61 / 71 |

After two rounds, GT units 62 and 241 cross above IoU 0.8 (0.79333 to
0.80405 and 0.79849 to 0.80353); unit 9 crosses below it (0.80240 to
0.79641). Net gain is one unit, not uniformly improved recovery.
Collision spike matches increase from 38,330 to 38,340 to 38,343 out of
44,667. There are no rare or late GT units in this evaluated interval.
Kilosort4's retained same-input reference recovers 231 units here. The
replacement gate remains unmet.

## Protocol

Dataset: existing 600-second, 250-unit, 384-channel recording at 32 kHz.
Initialize from the same 319-template bank used in previous experiments:
`F:/sortingDevelopment/session_seeds_ks60_20260924_01/seeds.npz`.
Its existing Kilosort-derived seed provenance is retained; this experiment
does not run Kilosort learning or feed ground truth into refinement.

Training uses two-second windows starting at 40, 100, 220, 340, 460 and
540 seconds. Internal validation uses two-second windows starting at 44,
224 and 464 seconds. Each has a 100 ms preprocessing halo. All learning
windows and halos are disjoint. Evaluation covers 180–210 seconds, excluded
from both learning partitions. This interval was used in earlier project
experiments, so it is a learning holdout, not an untouched new dataset.
No refinement settings were tuned after inspecting these results.

Each round reassigns the cached learning windows using the current bank.
Training evidence is sampled deterministically by unit and window, with at
most eight snippets per stratum and a 16,384-waveform cap per partition.
Evidence requires first-pass score >= 0.9, relative margin >= 0.1, fitted
amplitude between 0.5 and 2, usable contacts, and no accepted overlapping
fit sharing contacts within 61 samples. Snippets are amplitude-normalized.
This excludes known overlaps, not undetected collisions.

Proposals move 25% toward the training median. A unit needs at least 12
training examples from two windows and six internal-validation examples.
Promotion requires >= 0.5% mean held-out waveform-error reduction, <= 10%
relative template change, a held-out error-tail guard, and unchanged dominant
contact/peak alignment. No GT criterion enters this gate. It is a waveform
surrogate and does not ensure correct identity or prevent every contamination.

Round one promotes 73 templates; round two promotes 32 (not necessarily
distinct units). Both rounds leave 186 templates unchanged for insufficient
evidence. Training/validation evidence rows are 4,838/2,425 and 4,891/2,437.
The same internal-validation windows are reused across rounds; the separate
evaluation helps reveal overfitting to this surrogate.

All evaluation runs use strict overlap scheduling, unchanged detection and
fitting thresholds, three residual passes, two-second cores, ten-second
shards, frozen templates and no novelty enrollment. The bank has 319 entries
throughout. Predicted active units change from 308 to 309 as existing templates
receive assignments; no new identity is enrolled. The zero-round rerun
reproduces the prior strict control's quality metrics exactly.

## Resource measurements

Learning reads 486,604,800 raw bytes once and writes 973,238,139 bytes of
preprocessed scratch. Both refinement rounds read that cache, avoiding more
raw-source reads. Learning plus cache construction takes 21.69 seconds;
individual rounds take 3.87 and 3.42 seconds without profiling. Learning
process peak working set is 856,559,616 bytes and the GPU pool reports
116,114,944 bytes. These fixed-budget measurements are not a long-run proof.

The three evaluation runs each read 811,008,000 bytes. Profiled sorting
times are 32.01, 31.48 and 30.65 seconds; single runs do not demonstrate
a speedup. Measured total VRAM is identical at 1,328,087,040 bytes, GPU
pool peak 116,100,608 bytes, and process peak working set about 1.60 GB.
Learning overhead is separate from those evaluation times. A production
workflow would refine on bounded samples then perform one dense full pass;
these three full evaluation passes are experimental comparisons.

## Artifacts and checks

New implementation: `src/terasort/session_refinement.py` and
`scripts/refine_session_templates.py`. See the session guide for invocation.
The script requires a new output directory, bounds scratch before source
reads, and writes separate bank versions and per-template decisions.
Seed hashing and full source/settings metadata were added after this pilot;
the change does not affect model decisions or the retained results.

Artifacts under `F:/sortingDevelopment`:

- `session_refinement_20260924_01`: cache, round 0/1/2 seed banks, audits and report.
- `session_refinement_eval_r0_20260924_01`: frozen original-bank evaluation.
- `session_refinement_eval_r1_20260924_01`: one-round evaluation.
- `session_refinement_eval_r2_20260924_01`: two-round evaluation.

All **190 tests pass** with `scripts/test.ps1`. Regression coverage includes evidence caps, deterministic sample selection,
duplicate rejection, disjoint partitions, insufficient support, training
contamination rejection, identity preservation and alignment preservation.

Further identical rounds are unlikely to resolve templates with no reliable
assignments. The next distinct experiment should address local competition,
splits and novel templates with independent evidence; it should not force
assignments or relax thresholds simply to raise the unit count.
