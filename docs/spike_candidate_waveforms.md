# SCB 0.2 local multichannel waveform cache

The extended example is `outputs/spike_candidates_v1/real_floor3_waveforms.scb.h5`.
Its event table is unchanged from `real_floor3.scb.h5`. Each candidate receives
a signed, referenced, filtered voltage snippet on its local contacts, centered
on the **original candidate sample**. Neighbor channels retain their simultaneous
sample times; they are not independently aligned to their own peaks. No candidate
merging, spatial suppression, unit identity or overlap separation is performed.

See the [measured results](../outputs/spike_candidates_v1/waveform_report.md) and
[base candidate format](spike_candidate_bank.md).

## Schema

| Entry | Type | Meaning |
|---|---|---|
| `manifest_json.waveforms` | JSON | Pre/post sample counts, units, encoding, axes, alignment and boundary policy |
| `manifest_json.waveform_provenance` | JSON | Parent ledger SHA-256, source fingerprint, preprocessing view and neighborhood definition |
| `waveforms/channel_index` | int32[recording channel, local slot] | Source-channel map, anchor first, `-1` trailing padding |
| `waveforms/data` | int16 or float32[event row, time, local slot] | Row-aligned snippets; microvolts = stored value × saved scale |
| `waveforms/valid_start` | int32[event row] | Inclusive first valid time index within the snippet |
| `waveforms/valid_stop` | int32[event row] | Exclusive last valid time index within the snippet |

The real pilot uses the four contacts of the same tetrode, with 20 samples before
and 40 after the event at 20 kHz: 61 samples including the center, spanning 3 ms.
The map is stored once per anchor channel. Physical coordinates remain unknown;
channel IDs are not estimated neuron positions. `geometry_channel_map` supports
known geometry with a radius, a channel cap, and explicit probe/shank groups.
It never chooses contacts from another group. Unequal neighborhoods use `-1`
padding. These rules are unit-tested, but not yet validated on a real Neuropixels
or flexible-probe recording.

The real cache uses **0.1 microvolt/count int16**, with rounding error at most
0.05 microvolt plus float32 decoding roundoff. An initial 0.05 microvolt/count
build correctly rejected a large transient; the observed filtered range was
about -2558.5 to +1302.1 microvolts. Encoding never clips: values outside the
selected int16 range raise an error before the block is appended. For other
recordings choose a sufficient scale or `storage_dtype='float32', scale_uv=1`
to preserve the preprocessed float32 values exactly. Amplitude/SNR metadata stay
float32 and threshold querying is unaffected by waveform quantization.

Source-edge samples and missing channel slots are stored as zero **with explicit
masks**, and decoded to NaN. They are never presented as valid measured zeros.
A processing chunk boundary is not a recording boundary: an insufficient voltage
halo is an error. The pilot reuses each detection block's exact filter halo;
snippets crossing a block boundary use their anchor block's filtered view. They
are not claimed to match a different whole-recording filtering implementation.

Waveform HDF5 chunks are capped at 512 rows and approximately 1 MiB where one
waveform fits. `iter_waveforms` bounds the span of rows read per batch, returns
immutable bank row IDs, and loads only selected waveform row ranges. Threshold
queries continue to read just the small event table. Channel filters select
**anchor candidates**; their cached neighbor channels remain in the output.
Compressed chunk reads can include unselected rows in the same storage chunk.

## Usage

```powershell
./scripts/run_spike_candidates.ps1 -Stage waveforms
$env:PYTHONPATH='src'
./.venv-spike-candidates/Scripts/python.exe -m painproject.candidates.cli outputs/spike_candidates_v1/real_floor3_waveforms.scb.h5 --snr 5 --channels 0 1 2 3 --waveforms
# Optional small-selection export; .npz materializes the selection in RAM:
./.venv-spike-candidates/Scripts/python.exe -m painproject.candidates.cli outputs/spike_candidates_v1/real_floor3_waveforms.scb.h5 --snr 5 --channels 0 1 2 3 --polarity neg --waveforms --output selected_waveforms.npz
```

The NPZ includes events, waveform values in microvolts, source channel IDs, bank
row IDs, validity bounds, sample offsets, sample rate and waveform metadata. NPY
event-only export remains supported. Exports refuse to overwrite existing files.
For large selections, use the bounded iterator rather than materializing NPZ:

```python
from painproject.candidates.bank import CandidateBank

bank = CandidateBank('outputs/spike_candidates_v1/real_floor3_waveforms.scb.h5')
for batch in bank.iter_waveforms(5, channels=[0, 1, 2, 3], batch_size=512):
    # waveforms_uv: selected events x 61 time samples x 4 local contacts
    # channel_indices: selected events x 4, actual recording-channel IDs
    # row_indices: join IDs into the original, unchanged event table
    waveforms = batch.waveforms_uv
    channels = batch.channel_indices
```

To attach waveforms from another source, call `cache_waveforms` with a
`WaveformSpec`, a provenance dictionary, and an iterator yielding
`(core_start, core_stop, voltage_start, voltage_uv)`. All sample coordinates are
absolute source samples and each core must match the original ledger's block
index. The iterator must preserve original preprocessing and supply enough halo.
The center voltage must exactly equal the saved event amplitude before encoding.
The extension publishes a new complete file atomically; the old bank remains
unchanged. Missing data, overflow and row-count mismatches fail explicitly.
Building uses memory proportional to one source block and its local snippets.

## Why multichannel storage is harder

One biological spike can produce nearby candidates on several channels and at
slightly different sample times. Their waveforms share both signal and background
samples. Storing every local snippet therefore duplicates data even when HDF5
compression is enabled. At a permissive threshold the waveform cache can exceed
the raw file; payload scales as `events × samples × local contacts × bytes`.
Use local neighborhoods, never all recording channels by default.

The current implementation preserves these candidates for later decisions. It
does **not** implement shared snippet storage, deduplication or a bounded reservoir.
For multiday work the next step is to compare a stratified, bounded waveform
learning subset against shared preprocessed time tiles when candidate windows
cover most of the recording. Time/probe sharding is still required; do not put a
multiday collection into one unlimited learning array.

Drift changes which contacts best represent a neuron, while the cache's channel
map describes acquisition contacts. Flexible probes can require changing local
geometry or broader neighborhoods. Unit tracking must be inferred downstream;
additional channels, longer windows, different filtering or lower candidate
floors require the raw archive. Overlapping spikes remain superimposed in the
saved waveform and are not resolved by caching alone.
