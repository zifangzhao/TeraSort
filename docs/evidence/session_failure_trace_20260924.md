# Event-level failure trace — 2026-09-24

Traced our detector, fitting and scheduling decisions directly, without new
Kilosort execution or use of its comparison scores. The existing bounded
seed bank was retained; it was originally learned by the optional K4 preview
route. Ground truth selects diagnostic errors and evaluation-derived unit
links identify the template to inspect. Neither changes live assignment.

## Scope

Replayed source seconds 60–90 from `session_fused3_20260924_01`, frozen
templates, three residual passes, SNR threshold 4.5 and center-score floor
0.65. All **59,142 replayed spike rows** matched original source times,
channels, unit IDs and residual-pass numbers exactly.

Selected up to four evenly spaced misses per evaluation-linked unit: 646
examples from 7,225 misses. Of these, 470 have a unit-link IoU >=0.5. This
is a unit-stratified diagnostic sample, **not a prevalence estimate**.
Incorrect links can make good assignments look like template failures.
Nearby candidates need not originate from the target neuron. Categories
identify the furthest observed path, not independent causal interventions.

| Observed path among the 470 better-linked examples | Count |
|---|---:|
| Expected template deferred by overlap scheduler, not recovered | 136 |
| Expected template never passes waveform-shape acceptance | 107 |
| No candidate; local raw SNR <=4.5 or contacts masked | 103 |
| Expected template eligible, another template wins | 93 |
| Expected template wins, ambiguity margin rejects it | 26 |
| Amplitude acceptance fails | 3 |
| Refractory suppression | 1 |
| Fitted peak outside evaluated output core | 1 |

## Concrete scheduling failure

GT unit 236 maps to local unit 97 with IoU **0.99475**. At source sample
**2,013,593**, a candidate occurs at **2,013,594**, channel 31:

| Pass (zero-based) | Score | Amplitude | Gain | Margin | Blocker |
|---|---:|---:|---:|---:|---|
| 1 | 0.81875 | 0.88995 | 673.99 | 1.0 | Unit 99, sample 2,013,644, contact 31 |
| 2 | 0.77474 | 0.74983 | 478.46 | 1.0 | Unit 92, sample 2,013,603, contact 125 |

Both fits pass score, amplitude, residual-gain and ambiguity requirements
and align to the true source sample. Scheduling alone rejects these
proposals. Pass 2 is the final default pass.

`session_gpu.py`, around line 201, rejects proposals sharing **any valid
contact within 61 samples**, regardless of waveform energy or temporal
interaction. In the final rejection, the shared contact contains only
**0.297%** of the blocker's energy and 2.58% of the target's energy.
Using the same core's noise estimates, absolute normalized shifted-template
inner products are **0.00582** for units 97/99 at 51-sample separation and
**0.00385** for units 97/92 at 10-sample separation. This supports testing
scheduling based on actual waveform interference. It does not yet prove
an aggregate quality gain from relaxing the rule.

Among the 136 deferred examples, the last observed deferral was pass 0 for
76, pass 1 for 35, and pass 2 for 25. Earlier deferral can be followed by
changed residual shape or competition; not all 136 are cured by more passes.

## Other concrete paths

- **Threshold:** GT unit 246, local unit 86, sample 2,042,377, mapping IoU
  0.98453. Maximum SNR in its mapped patch/time window is **4.4278**; the
  4.5 detector threshold produces no candidate there.
- **Shape:** GT unit 236, local unit 97, sample 1,955,729. Candidate SNR
  **5.740**, expected-template center score **0.4454**, below 0.65.
- **Competition:** GT unit 237, local unit 59, sample 2,731,561, mapping
  IoU 0.99631. Expected template is eligible (score 0.8934, gain 8,730.96),
  but unit 60 wins and is assigned nearby.
- **Ambiguity:** GT unit 93, local unit 255, sample 1,993,737, mapping IoU
  0.98958. Unit 255 wins with score **0.9692**, but margin **0.00167** is
  below 0.03. This does not justify forcing the identity or merging units.
- **Boundary:** GT sample 1,920,000 fits at 1,919,999, just before the
  evaluated range. It supplies halo context but is excluded from output.

## Implementation and validation

Added an optional `trace` callback to `CudaResidualMatcher.match`. It reports
candidate choices and accepted/refractory/overlap decisions, including exact
blocker identities, times and contacts. Normal calls do not retain diagnostic
owner tables or copy residuals to host memory.

`scripts/trace_session_failures.py` inspects expected-template fits on CPU
from one bounded residual core per pass. It rejects unsupported settings
and intervals longer than 120 seconds; currently it expects a single-probe
`day1` diagnostic layout. `--focus-sample` reproduces one selected miss.

**178 tests pass**, including exact candidate/match equivalence with tracing
enabled. Production thresholds, IDs and scheduling decisions were unchanged
in this debugging step. No sorting accuracy improvement is claimed yet.

Artifacts under `F:\sortingDevelopment`:

- `session_failure_trace_20260924_02`: final `summary.json` and `events.jsonl`.
- `session_failure_focus_20260924_01`: focused sample 2,013,593 reproduction.
- `session_failure_trace_20260924_01`: initial trace without blocker details.

Initial artifacts use `kilosort_inputs_used: false` to mean no additional
K4 comparison/execution. Final script metadata explicitly distinguishes
that from retaining the existing seed bank.

Next correction: test interference-aware overlap scheduling on these exact
failures and a separate interval, measuring new false assignments as well
as recovered spikes. Low-SNR multi-contact detection and template competition
should remain separate experiments so their effects can be attributed.
