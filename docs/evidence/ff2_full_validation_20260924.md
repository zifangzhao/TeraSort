# Complete FF2 recording and interrupted replay

The 1761.1712-second FF2 D82 Linear A recording (128 channels, 20 kHz, 9,017,196,544 raw bytes) completed directly from the read-only network source. This validates one 29-minute recording, not a 2 TB or multiday workload.

## Results

| Measurement | Uninterrupted run |
|---|---:|
| Dense sorting wall time | 385.56 s (6 min 26 s) |
| Recording / wall time | 4.57x |
| Peak sampled process RSS | 524.8 MB |
| Peak CUDA pool | 51.25 MB |
| Peak total device VRAM | 1.265 GB |
| Candidate metadata rows | 14,726,039 |
| Assigned spikes | 1,061,782 |
| Immutable shards | 6 |
| Source bytes read, including repeated halos | 9.918 GB |
| Assignment shard bytes | 429.94 MB (4.77% of raw) |
| Cached waveform payload | 404.01 MB |

Sizes use decimal units. Timings exclude the previously completed 60-second seed-calibration preview. This run uses the pilot's 72-template bank, raw detection, score floor 0.75, rescue SNR 3.5, two-second cores and five-minute shards. Templates are frozen; novelty enrollment is off. Kilosort provided calibration identities, not an independent same-interval quality reference.

Median RSS rose from 508.1 MB in the first shard to 522.0 MB in the third, then settled at 518.0 MB in the last. CUDA pool allocation stayed at 51.25 MB. This is encouraging evidence of bounded operation at this duration and event density; it does not establish a terabyte-scale memory bound. Process/device memory was sampled, not continuously traced for every transient allocation. Source-read bytes are application accounting, not measured NIC traffic; OS/SMB caches were not flushed.

## Actual process kill and resume

A second run was killed by its parent process at source second 310, with one completed shard and one incomplete shard. On resume, the first processed core covered seconds 300–302, confirming that the completed 0–300-second checkpoint was reused. The remaining five shards completed in 310.50 seconds. That resumed summary excludes the already completed first shard; the final output includes the entire recording.

All 21 scientific datasets in each of the six shards match the uninterrupted run exactly, including candidate records, spike records, selectively cached waveforms, QC, and model state. Scientific attributes match; timing telemetry was excluded. QC intervals cover the complete source clock contiguously, spike coordinates stay inside their shards, and no duplicate sample/unit pairs were found. No partial files remain. This verifies one abrupt process-kill scenario, not power-loss durability or forced network-disconnect recovery.

## Quality findings

All 72 template IDs remain active in the first five shards; 71 remain active in the final shard. Within-unit consecutive intervals below 1.5 ms account for 2.90% overall. These measures describe assigned activity; they do not prove stable neuron identity, precision, or recall.

QC flags 182 seconds (10.3%) as unreliable intervals. All contacts are masked for 14 seconds total. In one inspected 418–420-second interval, saturation occupied about 0.33% of samples per channel, but the current QC rule disables affected contacts for the entire two-second core. This preserves flagged time in the QC outputs but can discard usable activity around brief artifacts. Finer temporal artifact masks are a next quality improvement. Geometry comes from retained trial coordinates and acquisition XML; independent physical-map verification remains pending.

At this measured raw-byte throughput, 2 TB would extrapolate to approximately 23.8 hours and about 95 GB of assignment shards, before calibration and other outputs. Different channel counts, event density, drift, network behavior, or adaptive templates can change this substantially. No 2 TB run has been completed.

## Artifacts and reproduction

- Output root: `F:\sortingDevelopment\ff2_full_validation_20260924_01`.
- `validation.json`: per-shard scientific and resource diagnostics, exact comparisons.
- `kill_evidence.json`: process-kill point and completed/partial filenames.
- `code_environment.json`: source fingerprints captured before subsequent reader changes; `report_code_environment.json` fingerprints the reporting environment.
- `scripts/validate_ff2_full.py`: create a new root, run the full pass, kill a second process and resume it.
- `scripts/report_ff2_full.py`: bounded scientific-payload comparison and plots.

![Memory, throughput and assigned activity](F:/sortingDevelopment/ff2_full_validation_20260924_01/streaming_validation.png)
