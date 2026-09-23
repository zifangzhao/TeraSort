# GPU spike candidates and the SCB 0.1/0.2 formats

This is an experimental detector and file-format pilot. It is separate from the
existing spike labels, waveform audit, and paper analyses. Results are in
[`outputs/spike_candidates_v1/report.md`](../outputs/spike_candidates_v1/report.md).
The waveform extension is described in [SCB 0.2 waveforms](spike_candidate_waveforms.md)
and measured in [the waveform report](../outputs/spike_candidates_v1/waveform_report.md).
The real example is a 120-second, 32-channel Intan tetrode excerpt. The 384- and
768-channel cases are synthetic engineering fixtures, not real Neuropixels or
flexible-probe validation. No full spike sorter is implemented or benchmarked.

## What was built

- CUDA C++ kernels (`detect.cu`) compiled by NVRTC through CuPy. They do not use
  Torch detection operators. The permissive detector fuses thresholding, temporal
  maxima, and event packing. Optional spatial exclusion uses two kernels.
- An independent NumPy reference and a simplified Torch implementation with the
  same rules. The installed SpikeInterface Torch detector is another baseline.
- An HDF5 candidate ledger with an explicit threshold floor, fixed noise
  calibration, source identity, sample clock, signed amplitudes, and time index.
- CPU-only querying at higher thresholds, selected channels, polarity and time.
- SCB 0.2 local multichannel waveform caches, with explicit channel maps, sample
  alignment, boundary masks, and int16 or float32 storage. Version 0.1 ledgers
  remain readable and can be extended into a new file without redetection.

The primary comparisons use normalized magnitude scores already computed from
the voltage. Candidate sorting on CPU is included in CUDA timings because atomic
packing does not promise event order. Timing excludes cold compilation. Resident
and host-transfer measurements are separate. Upstream wrapper sample-offset
conventions are normalized in the benchmark adapter; zero guards isolate the
interior comparison. Separate tests cover actual chunk boundaries.

The reference semantics are a strict score > threshold rule, first temporal
maximum on ties, and retained spatial ties. Spatial exclusion, when enabled,
compares detected temporal peaks within twice the temporal exclusion radius.
These are explicit detector rules, not a claim that all algorithms or all
SpikeInterface CPU/Torch versions share the same semantics.

## SCB file schema

Use `<segment>.scb.h5`, one probe and acquisition segment per file. A multiday
collection should use a manifest over time/probe shards. HDF5 is the container;
SCB defines the schema rather than inventing a new binary encoding.

| Entry | Type | Meaning |
|---|---|---|
| `manifest_json` attribute | JSON | Format/version, source identity, detector and preprocessing settings, coordinate conventions |
| `complete` attribute | Boolean | Only true after successful writing |
| `events.sample_index` | int64 | Native original-source sample clock, not float seconds |
| `events.channel_index` | uint32 | Recorded channel, with zero-based indexing |
| `events.amplitude_uv` | float32 | Signed filtered/referenced peak voltage |
| `events.snr` | float32 | Absolute peak voltage / saved channel noise |
| `noise_uv` | float32[channel] | Fixed noise estimate used to detect and query |
| `channel_positions_um` | float32[channel, 3] | Electrode geometry, NaN when unknown; not neuron localization |
| `blocks` | Structured int64 table | Half-open source-sample bounds and event-row bounds |

An event's identity is `(source segment, sample_index, channel_index)`. Rows are
ordered by sample then channel and must be unique. The event table is 20 bytes
per row before compression; compressed HDF5 chunks add overhead. There is no
spatial suppression, deduplication across channels, or unit assignment in the
saved permissive ledger. Several channels can represent the same biological
spike. Both polarities are represented by local extrema of absolute normalized
voltage, with a one-sample temporal radius. This preserves detector-defined
candidates, not every spike in the biological recording.

Event datasets use LZF compression and shuffling. Writes publish a `.partial`
file only on success. An interrupted file remains explicitly incomplete; it is
not automatically resumed in this prototype. Existing completed files are not
overwritten. The writer validates event ordering, bounds, and amplitude/SNR
consistency. `iter_query` processes one indexed block at a time; `query` and NPY
export deliberately materialize the result and are for selections that fit RAM.

## What can be changed without reading voltage

- Raise SNR threshold above the saved floor (including per-query threshold sweeps).
- Select times, channels, and polarity.
- Implement alternative spatial exclusion over retained candidates later; this
  operation is not yet exposed by the bank CLI.

Lowering the threshold below the floor, changing the reference/filter/noise
estimator, changing detection polarity semantics, finding missed overlaps, or
fitting the full continuous recording requires raw voltage. Filtering saved positive or negative
events is not equivalent to rerunning a polarity-specific detector when extrema
of the opposite sign compete. The exact replay guarantee here is for raising the
threshold under the original magnitude-peak detector and fixed preprocessing.

PCA features, estimated neuron positions, artifact labels, and an adaptive
learning reservoir are not yet stored/implemented. SCB 0.2 waveforms can support
later feature extraction, waveform inspection and local clustering. Keep the raw
archive. The ledger and complete waveform cache both grow with time; an eventual
learning bank should have a separate fixed memory budget.

## Real-source preprocessing and provenance

The fixed excerpt starts 60 seconds into the selected source recording and lasts
120 seconds. Original network files are opened read-only. The local excerpt and
Intan header have SHA-256 fingerprints; the entire multi-GB source was not hashed.
The source size, modification timestamp and original sample offset are retained.
No stimulus or accepted-unit labels determine the excerpt.

For the real pilot, int16 values are converted using 0.195 microvolts/count,
referenced to the median within each consecutive group of four contacts, and
filtered with a third-order 300–6000 Hz Butterworth filter applied forwards and
backwards. Two-second processing cores have 50-ms filter halos. Noise is fixed
from the first ten seconds of the cached excerpt using centered MAD/0.67448975.
Replays use the same filtering chunks and halos. Exact replay does not establish
equivalence to a different whole-recording filter implementation. Physical
contact coordinates are unknown and remain NaN.

## Running

Use a dedicated environment; the existing scientific environments are unchanged.
The driver must support the chosen CUDA wheels. The tested machine is Windows
with an RTX 5060 Ti (16 GB). The custom kernel uses the environment's CUDA 12.9
NVRTC; it does not require updating the old system-wide CUDA toolkit or compiling
a PyTorch C++ extension with MSVC.

```powershell
# Recreate with a Python 3.11 environment at .venv-spike-candidates:
./.venv-spike-candidates/Scripts/python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-spike-candidates-lock.txt

./scripts/run_spike_candidates.ps1 -Stage prepare  # needs original network source
./scripts/run_spike_candidates.ps1 -Stage benchmark
./scripts/run_spike_candidates.ps1 -Stage bank
./scripts/run_spike_candidates.ps1 -Stage report
./scripts/run_spike_candidates.ps1 -Stage waveforms  # extend, validate, benchmark waveform queries

$env:PYTHONPATH='src'
./.venv-spike-candidates/Scripts/python.exe -m pytest tests/test_spike_candidates.py tests/test_spike_waveforms.py -q
./.venv-spike-candidates/Scripts/python.exe -m painproject.candidates.cli outputs/spike_candidates_v1/real_floor3.scb.h5
./.venv-spike-candidates/Scripts/python.exe -m painproject.candidates.cli outputs/spike_candidates_v1/real_floor3.scb.h5 --snr 5 --channels 0 1 2 3 --polarity neg
```

After `prepare`, all later steps replay from the retained local excerpt. Bank
queries require only NumPy and h5py; GPU libraries and raw voltage are unnecessary.
Keep the lock file, source hashes, raw excerpt, report, and test results together.

## Remaining evaluation

Measure representative real Neuropixels and flexible-probe recordings, duration
scaling, and performance under artifact bursts. Profile with Nsight, compare a
compiled-Torch baseline, and benchmark double-buffered transfers before judging
the hardware performance ceiling. Validate detector recall with injected known
events and full sorting outcomes before claiming improved neuron recovery.

References: [SpikeInterface detector](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sortingcomponents/peak_detection/by_channel.py),
[CuPy runtime compilation](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawModule.html).
