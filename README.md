# TeraSort

TeraSort is an installable, Kilosort 4.1.7-compatible sorting entry point with
source-guarded CUDA optimizations, plus a separate spike candidate bank with
optional cached multichannel waveforms. The sorter calls Kilosort for
preprocessing, learning, clustering and Phy export; it substitutes measured hot
paths during a scoped run. It returns Kilosort's nine-value tuple and writes
the usual Kilosort/Phy output files. The installed Kilosort package is not
modified.

## Install on Windows with an NVIDIA GPU

Use Python 3.11 and run this from the TeraSort directory in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\terasort.exe backends
```

Alternatively, `powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1`
runs these commands and checks CUDA. The included cuBLAS bridge is compiled for
Windows x64 and uses the cuBLAS DLL from the tested PyTorch 2.10 CUDA 12.8
installation. CuPy compiles the `.cu` kernels on first use. A supported NVIDIA
driver is required. Installation downloads Python dependencies; it does not
download a recording.

Other platforms have not been validated in this package. The standard Kilosort
backend can run where its dependencies install; `deep_tiled` additionally
requires CUDA and a matching CuPy wheel. The included Windows DLL does not run
elsewhere.

## Sort

Change one import in an existing Kilosort script:

```python
from terasort import run_kilosort  # instead of from kilosort import run_kilosort

result = run_kilosort(
    settings={"n_chan_bin": 384, "fs": 30000, "scale": 0.05},
    probe=probe,
    filename=r"F:\data\recording.bin",
    results_dir=r"F:\data\sorting_result",
    data_dtype="int16",
)
```

`backend="auto"` selects the cuBLAS path for the tested six-template,
61-sample geometry on Windows/CUDA and uses unmodified Kilosort otherwise.
`backend="standard"` always runs Kilosort unchanged. Explicit options are
`deep_tiled` and `cublas`. For a single named read-only INT16 binary, the fast
reader transfers INT16 to the GPU and converts to calibrated FP32 there;
sorting arithmetic remains FP32. Pass `fast_int16=False` to use Kilosort's
reader. Other Kilosort arguments are forwarded unchanged.

Drift correction is enabled by Kilosort's default settings. For a recording
where you want to skip motion estimation and its extra spike-detection pass,
set `skip_drift_correction=True` in Python or add `--skip-drift-correction` to
`terasort sort`. This uses Kilosort's documented `nblocks=0` path for that run.
The caller's settings dictionary is unchanged. Existing scripts that already
set `settings["nblocks"] = 0` continue to work. Skipping correction can change
sorting quality on drifting recordings, so compare outputs before adopting it.

CLI example (the settings file must include `n_chan_bin`):

```powershell
.\.venv\Scripts\terasort.exe sort --settings settings.json --probe-json probe.json --filename F:\data\recording.bin --results-dir F:\data\sorting_result
```

Probe names must be valid Kilosort bundled names. A JSON probe dictionary can
be passed with `--probe-json`. `terasort sort --help` lists all options. The
`scale` setting is the recording's calibrated microvolts per INT16 count; use
the value for your acquisition, rather than the example value. The included
[`settings`](examples/public_sample_settings.json) and
[`probe`](examples/public_sample_probe.json) files are configured for the
60-second smoke test on `F:/sortingDevelopment/data/recording_int16_uv.bin`:

```powershell
.\.venv\Scripts\terasort.exe sort --settings examples\public_sample_settings.json --probe-json examples\public_sample_probe.json --filename F:\sortingDevelopment\data\recording_int16_uv.bin --results-dir F:\sortingDevelopment\terasort_smoke
```

The `tmax` field in this settings file limits the run to the first 60 seconds.
Remove it to process the whole file. The candidate bank query tool is
`terasort-candidates`; see
[`docs/spike_candidate_bank.md`](docs/spike_candidate_bank.md) and
[`docs/spike_candidate_waveforms.md`](docs/spike_candidate_waveforms.md).

For a full-file sort, add `--lfp-output PATH` to run the 1,250 Hz LFP export
on lower-priority CPU workers during final clustering, after both spike-detection
stages. The command waits for both outputs. LFP always covers the complete
input file, even when sorting uses `tmin` or `tmax`. This overlap can reduce
total elapsed time, but the LFP worker still reads the raw file separately and
may compete for CPU or disk bandwidth.

On the 600-second sample, deferred LFP did not slow either spike-detection stage
in a single comparison; final clustering took longer. See the
[`full-run timing`](docs/evidence/lfp_parallel_600s.md).

## Optional 1,250 Hz LFP export

Use this command for a standalone LFP export; sorting never reads the LFP
file. The `sort --lfp-output` option above generates it during sorting:

```powershell
.\.venv\Scripts\terasort.exe lfp --filename F:\sortingDevelopment\data\recording_int16_uv.bin --output F:\sortingDevelopment\data\recording_lfp_1250_lp500.i16 --sample-rate 32000 --n-channels 384 --scale-uv-per-count 0.05 --passband-hz 500
```

This reads the original interleaved INT16 voltage in bounded chunks, applies
an anti-alias FIR, and writes time-major interleaved INT16 at exactly 1,250 Hz.
The default flat LFP passband is 0–500 Hz, with nominal 60 dB attenuation
beginning at the 625 Hz output Nyquist frequency. Use `--passband-hz 300` for a
shorter, faster filter, or another cutoff below 625 Hz. The 500 Hz filter takes
more CPU time because its transition band is narrower. Channels are filtered
in parallel on eight CPU workers by default; `--workers 1` limits CPU use.
A JSON sidecar records source identity, channel count, sample rates,
filtering, scale and any saturated output values. Output has the same channel
order and microvolts-per-count scale as the input. At 600 seconds
and 384 channels the LFP binary is 576,000,000 bytes, versus 14,745,600,000
bytes for the 32 kHz source. No full source hash is computed because that would
add another complete read. If interrupted, rerun the same command with
`--resume`; completed chunks are retained and the last partial chunk is
discarded. LFP is derived from raw voltage, not Kilosort's high-pass signal.

## Scale and evidence

The current accelerated sorter remains a whole-recording Kilosort run. Tiled
detection limits one large GPU intermediate, but Kilosort's global clustering,
feature collection and output stages are **not yet bounded by recording length**.
Do not treat this version as a validated hundred-terabyte sorter. The candidate
bank is shard-oriented and can be written per segment; it is not yet a general
multiday unit linker. See [`docs/bounded_sorting.md`](docs/bounded_sorting.md).

On one 600 s, 384-channel public synthetic recording, the C++/cuBLAS INT16 run
took 208.186 s against 214.379 s for the fresh `deep_tiled` INT16 control
(2.9% less worker time). The native run recovered 230/250 ground-truth neurons
at IoU ≥ 0.8 versus 229/250 for control. This is a single warmed pair, not a
general speed or accuracy guarantee. The retained
[`report`](docs/evidence/cublas_600s_report.md) and
[`summary`](docs/evidence/cublas_600s_summary.json) document the run. See also
[`docs/kilosort_cublas.md`](docs/kilosort_cublas.md).

## Development and license

Run `python -m pytest -q tests` in the installed environment. The source
guards refuse to patch an unverified Kilosort implementation. TeraSort's
adapters include transformations of Kilosort 4.1.7 GPL-3.0 code at runtime;
this package is distributed under GPL-3.0-only. See `LICENSE` and the
[Kilosort repository](https://github.com/MouseLand/Kilosort) for upstream
source and citation. Benchmark figures are reproduced from the PainProject
development trials; the packaged entry point and install path are tested
separately.
