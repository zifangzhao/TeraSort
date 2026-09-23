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
Remove it to process the whole file. The
candidate bank query tool is `terasort-candidates`; see
[`docs/spike_candidate_bank.md`](docs/spike_candidate_bank.md) and
[`docs/spike_candidate_waveforms.md`](docs/spike_candidate_waveforms.md).

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
