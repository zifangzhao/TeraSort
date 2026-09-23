# Integrated INT16 reader

The optional `--storage int16` route reads calibrated INT16 samples through a
reusable pinned host buffer, transfers INT16 to the GPU, and decodes to FP32.
One CUDA C++ kernel performs gain/offset application, sample/channel transpose
and Kilosort boundary padding. A shared-memory tile supports coalesced input
loads and output stores. FP32 filtering and the existing deep CUDA matcher
then consume the same tensor layout as the original reader.

The adapter checks Kilosort 4.1.7 and the installed reader source hash before
substitution. It applies only to the named recording and restores the original
method on exit, including exceptions. A CUDA event protects each reusable
pinned buffer until its previous transfer finishes; device buffer reuse also
waits for the previous decode across streams. Source sample coordinates and
candidate timestamps remain INT64.

The current implementation has one pinned/device staging pair per live
Kilosort file reader. It does not keep a whole-recording memory map alive,
prefetch disk reads or overlap future batches with current computation.
The default batch uses roughly 46.17 MB for each INT16 staging buffer. Other
sorting state and FP32 output tensors still occupy memory. Whole-sort
clustering and multiday continuity are separate scalability work.

## Public sample conversion

The source is FLOAT32 microvolts. `prepare_int16_sorting_sample.py` creates a
separate INT16 file using a 0.05 uV/count scale, without modifying the original.
It processes bounded chunks, rejects nonfinite values and overflow, verifies
the source hash while reading, and verifies the entire output by read-back
hash plus six independent slice comparisons before committing metadata.
Interrupted uncommitted outputs are preserved and rejected on rerun, not
silently reused. Conversion itself does not yet resume midway through a file.
Completed conversions are checksum-verified and reused.

The complete sample audit found no overflow among 7,372,800,000 values.
Maximum error was 0.025001526 uV, RMS error 0.014433632 uV, and maximum source
absolute voltage 762.614258 uV. The encoded file contains 14,745,600,000 bytes,
versus 29,491,200,000 bytes for the original binary. Native ADC recordings can
retain their original counts and calibration instead of re-quantizing.

## Validation and reproduction

Eight GPU reader tests cover signed ADC limits, uneven channel tiles, tiny
files, edges, cropped recordings, downsampling, per-channel calibration,
cross-stream buffer reuse and context restoration. Six real batches also
matched independently decoded FP32 raw and preprocessed tensors exactly.
That equivalence is against the same quantized values, not the unquantized
recording. A separate complete-sort ground-truth evaluation tests the latter.

```powershell
.venv-spike-candidates/Scripts/python.exe -m pytest tests/test_kilosort_int16.py tests/test_sorting_trial_evaluation.py tests/test_windowed_sorting.py -q
.venv-spike-candidates/Scripts/python.exe scripts/prepare_int16_sorting_sample.py --root F:/sortingDevelopment
.venv-spike-candidates/Scripts/python.exe scripts/benchmark_int16_reader.py --root F:/sortingDevelopment
.venv-spike-candidates/Scripts/python.exe scripts/run_native_sorting_trial.py --root F:/sortingDevelopment --storage int16 --mode deep --trial int16_sorting_v1 --job-name deep_int16
.venv-spike-candidates/Scripts/python.exe scripts/run_native_sorting_trial.py --root F:/sortingDevelopment --storage float32 --mode deep --trial int16_sorting_v1 --job-name deep_fp32_warm_control --warm-input
.venv-spike-candidates/Scripts/python.exe scripts/report_int16_sorting.py --root F:/sortingDevelopment
```

Full sorting jobs retain source identity, calibration/dtype, reader CUDA source
snapshots, telemetry and checksummed outputs. Completed sorting jobs resume
through the existing trial harness. Use a new job/trial name when source
identity changes. The original default FP32 route is still available.

Results are in `F:/sortingDevelopment/int16_sorting_v1/report.md`; raw full-run
results are `full_deep.json` and `full_deep_int16.json`. Conversion/verification
time is reported separately from sorting time. The INT16 input was fully read
back during verification. An initial FP32 control exposed a slower first pass;
an additional explicitly pre-read FP32 control is used in the main comparison.
Both controls are retained. These are warmed-input results without cache
flushing, not a repeated cold-disk performance claim.
