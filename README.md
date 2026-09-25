# TeraSort

TeraSort is an installable, Kilosort 4.1.7-compatible sorting entry point with
source-guarded CUDA optimizations, plus a separate spike candidate bank with
optional cached multichannel waveforms. The sorter calls Kilosort for
preprocessing, learning, clustering and Phy export; it substitutes measured hot
paths during a scoped run. It returns Kilosort's nine-value tuple and writes
the usual Kilosort/Phy output files. The installed Kilosort package is not
modified.

## Install on Windows with an NVIDIA GPU

For double-click setup, run **`install_and_start.bat`** in this repository.
It calls the existing installer, verifies CUDA, then starts the dashboard.
On later visits, use **`start_server.bat`**; it opens an already running
dashboard or starts a new local server. Keep the server console open while
using it. Both batch files locate the repository relative to their own paths.
They require the complete repository; copying only a batch file elsewhere
does not install the source code or Python itself. A custom Python executable
can be supplied as `install_and_start.bat -Python "C:\path\to\python.exe"`
or positionally as `install_and_start.bat "C:\path\to\python.exe"`.

Use 64-bit Python 3.10–3.14, Git, an NVIDIA GPU and compatible driver, and an
internet connection for Python packages. In PowerShell, clone the repository
and run its installer:

```powershell
git clone https://github.com/zifangzhao/TeraSort.git
Set-Location TeraSort
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

If you already cloned TeraSort, run the last command from that directory and
skip cloning. The installer reuses an existing supported `.venv`, or creates
one with a detected system Python 3.10–3.14, falling back to the bundled Python
3.11 runtime. Detection checks the active `python.exe`, Windows install
registry, common Conda folders, and the Python launcher. It installs missing
packages, repairs an incompatible PyTorch build, checks the
dependency set, then compiles a small CUDA detection kernel and lists available
sorting backends. To select a specific supported Python executable, pass it
with `-Python`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1 -Python 'C:\path\to\python.exe'
```

If a dependency is later removed or becomes inconsistent, run
`install_and_start.bat` to repair the environment. `start_server.bat` also
repairs missing or incompatible packages when no dashboard is already running.

Verify an existing installation or inspect the commands without activation:

```powershell
.\.venv\Scripts\python.exe -c "import torch, terasort; print(terasort.__version__, torch.cuda.is_available())"
.\.venv\Scripts\terasort.exe backends
.\.venv\Scripts\terasort.exe sort --help
.\.venv\Scripts\terasort.exe lfp --help
```

The expected CUDA check is `True`; `backends` lists `cublas` on the tested
Windows x64 setup. For a manual installation, create `.venv` with Python 3.10–3.14,
then run the following from the repository root:

```powershell
& 'C:\path\to\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e .
```

The included cuBLAS bridge uses the cuBLAS DLL from the tested PyTorch 2.10
CUDA 12.8 installation. The package installs the NVRTC runtime required by
CuPy; other `.cu` kernels compile on first use.
Installation does not download a recording. Other platforms have not been
validated in this package; the included Windows DLL does not run elsewhere.

## Browser dashboard

The dashboard also accepts an existing Neuroscope `session.xml` or
`amplifier.xml` instead of a settings JSON. Click **Read XML** to preview the
sample rate, channel count, channel groups and excluded contacts. If exactly
one matching binary is found next to the XML, its path fills automatically;
otherwise select inputs explicitly in acquisition order.

Leave the probe fields empty to generate `probe.json` automatically from XML.
This follows the MATLAB wrapper's default **staggered** convention: alternating
−20/+20 µm horizontal offsets, 20 µm vertical steps, and 200 µm between anatomical
groups. XML supplies group/channel order; these dimensions are a layout
assumption, not measurements stored in XML. Skipped contacts keep their original
positions before exclusion, and channel numbers remain zero-based. Supply a
custom probe JSON only for a different physical layout. Generated settings,
probe JSON and layout provenance are saved alongside each queued job. A bundled
probe name alone is not supported in XML mode.

Gain is optional for sorting in ADC units; enter the verified microvolts per
count for scaled output (for example, 0.195 for the tested Intan recording).
XML-configured LFP export requires this gain. Legacy XML voltage-range and
amplification fields are not silently converted into an ADC gain. The optional
LFP route retains its existing 1250 Hz output and 500 Hz passband settings.

This imports metadata for the existing dashboard's Kilosort-compatible sorting
route. It does not run MATLAB, sleep-state scoring or behavioral analysis from
`preprocessSession.m`, modify source XML, or physically concatenate raw files.
The recording picker filters supported raw INT16 `.dat` and `.bin` files, with
Intan export and Neuropixels views. Native Intan `.rhd`/`.rhs` and compressed
`.cbin` need conversion before sorting. You can select multiple recording
segments together; click order is acquisition order and all segments become one
sort session. TeraSort looks for a valid same-name XML or `amplifier.xml` beside
the first selected file and loads it automatically. The output defaults to a
dedicated `<recording>_terasort` folder beside that file, keeping the acquisition
folder contents intact. Change the output path if the source folder is read-only
or you want results elsewhere. LFP export currently supports one source per job.
Apply one XML only to files known to share its acquisition configuration.

Launch the local web service after installation:

```powershell
.\.venv\Scripts\terasort.exe web
```

It opens `http://127.0.0.1:8765/`. Select one or more raw binaries in
acquisition order, settings JSON, probe
JSON (or bundled probe name), and a new results directory. The browser lists
files on the recording machine; it never uploads the raw data. The queue runs
one session at a time by default. Set **Concurrent sessions** in the queue
panel to 1–4; the limit persists with the dashboard state and applies to newly
launched jobs. Running jobs continue if the limit is lowered. Concurrent jobs
share GPU memory, CPU, network and disk bandwidth, so test higher limits on
your hardware before queuing large sessions. The queue and logs persist in
`%USERPROFILE%\.terasort\web` by default, or a directory supplied with
`--state-dir`. Keep the service running to launch queued jobs. An active worker
continues if the browser closes; the dashboard can reconnect after a server
restart.

Failed or cancelled runs can be retried from their detail panel. A retry keeps
the input and sorter settings, then queues a new attempt with fresh result and
staging paths so partial outputs are preserved. The service also accepts
`POST /api/jobs/<id>/retry`.

The run view shows Kilosort's current stage, elapsed time, a rough stage-based
ETA, worker RAM, system CPU/RAM, and device-wide NVIDIA GPU utilization and
memory. Stage progress advances at Kilosort log boundaries, so it may pause
for a long clustering stage. `nvidia-smi` must be available for GPU readings.
The current sorter still has whole-recording Kilosort memory limits described
under “Scale and evidence” below; this dashboard does not make a hundred-TB
sort bounded.

The service listens only on localhost by default. To access it from another
computer, supply a bearer token and bind to a network interface (use a trusted
network or tunnel; HTTP itself is unencrypted):

```powershell
$env:TERASORT_WEB_TOKEN = '<long-random-token>'
.\.venv\Scripts\terasort.exe web --host 0.0.0.0 --no-browser
```

The page prompts for the token, which is held in browser session storage. You
can inspect the service with `GET /api/jobs`, `GET /api/jobs/<id>`, and
`GET /api/browse?path=...`; `POST /api/jobs` accepts the same form fields as
JSON, and `POST /api/jobs/<id>/cancel` stops a queued or running job.

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
`deep_tiled` and `cublas`. For named read-only INT16 binaries, the fast
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

To sort multiple segments together, repeat `--filename` in acquisition order:

```powershell
.\.venv\Scripts\terasort.exe sort --settings settings.json --probe-json probe.json --filename F:\data\segment_0001\neuropixels_ap.dat --filename F:\data\segment_0002\neuropixels_ap.dat --results-dir F:\data\session_sort
```

The Python API likewise accepts `filename=[path1, path2, ...]`. Kilosort
learns one set of units and produces one normal Phy output. Raw files are read
in place; TeraSort does not create a combined binary. All files must have the
same INT16 channel layout and sample rate specified by the settings. The
result's `session_sources.json` records each path, byte count, sample count,
and half-open virtual sample range. `spike_times.npy` uses this virtual sample
axis: for a spike at virtual sample `t`, locate the source whose start ≤ `t`
< stop and subtract its start to get the file-local sample. Python users can
call `terasort.session.locate_spikes(spike_times, manifest)`.

The virtual boundary is only for Kilosort's shared sort. Recording gaps have
unknown duration unless separately measured; no gap samples or timestamps are
invented. Kilosort may process a batch across a file boundary, so inspect or
exclude spikes near those boundaries. The sidecar preserves this limitation
rather than implying a continuous acquisition. Session LFP export is not yet
available from `sort --lfp-output`; export LFP from each source separately.

### Bounded background reads for network recordings

The fast INT16 Kilosort route can download sequential blocks to a local SSD
while the GPU processes the current data. The dashboard enables this for new
runs by default at `F:\TeraSortReadCache`; change or clear that path to choose a
different drive or disable it. For the CLI, add `--read-cache-dir
F:\TeraSortReadCache` to `terasort sort`. The defaults,
`--read-cache-mb 4096 --read-cache-slots 2`, use 4 GiB disk blocks with an
8 GiB total disk budget, independent of recording length. One background
downloader streams each block in 8 MiB pieces; the data stays on disk rather
than occupying Python RAM.
Sparse calibration reads fetch only their requested ranges to avoid downloading
unused gaps. Inputs remain read-only, including across concatenated files.

This option requires the fast CUDA INT16 reader and cannot be combined with
full staging. It applies to newly started jobs; it does not change a running
worker. Cache files are removed on normal close; a killed process can leave its
unique `terasort-read-*` folder behind. This is temporary read-ahead, not a
persistent restart cache. Network errors fail the read rather than silently
returning incomplete data. Actual speed depends on the server, network, local
SSD, and sorting stage; a real-network speed improvement has not yet been measured.
The separate LFP process does not use this cache. The cache needs at least
8 GiB plus 64 MiB free on its selected volume when a job starts.

### Optional local staging for network recordings

For a recording on a slower SMB/NAS share, add `--stage-dir` with a **new local
directory**. TeraSort makes one sequential, read-only-source copy before the
sort, then reuses the local INT16 file for Kilosort's repeated passes:

```powershell
.\.venv\Scripts\terasort.exe sort --settings settings.json --probe-json probe.json --filename '\\server\share\recording\amplifier.dat' --results-dir C:\results\run01 --stage-dir C:\scratch\run01_input
```

The dashboard exposes the same optional field. TeraSort refuses an existing
staging directory, checks free scratch space before copying, writes each file
through a `.partial` name, checks source size/mtime, and verifies a SHA-256
checksum against the local copy. It records the original and staged paths in `input_staging.json` next to
the Phy output and in `staging_manifest.json` inside the scratch directory.
Scratch copies are deliberately retained after sorting; remove them yourself
when the run is verified. For multiple inputs, each is staged in the original
order, and `session_sources.json` still maps spikes to the **original** files.

Staging is optional because it consumes local disk equal to the entire input
and adds an initial copy. It is useful only when repeated remote reads cost
more than that copy. It is not a bounded-storage solution for a hundred-TB
session; the whole-recording Kilosort limits below still apply.

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

## Bounded candidate subsampling (experimental)

`terasort-sample-candidates` scans only SCB event metadata in two bounded
passes, then selects deterministic candidate row IDs across time, shank,
depth and SNR strata. Pass `--probe-json` for a multishank probe and verify the
coordinate axes recorded in the output manifest. A matching-budget uniform
sample is saved for comparison. The returned inverse-inclusion weights correct
for deliberate oversampling of sparse strata when estimating population
statistics. For example:

```powershell
.\.venv\Scripts\terasort-sample-candidates.exe --bank F:\data\candidates.scb.h5 --probe-json probe.json --budget 30000 --seed 17 --time-bins 6 --depth-bin-um 160 --minimum-per-stratum 20 --output F:\data\calibration_sample
```

The command writes `stratified_row_indices.npy`,
`stratified_inverse_inclusion_weight.npy`, `uniform_row_indices.npy`, stratum
counts and quotas, and `manifest.json` into a new directory. It leaves the SCB
bank unchanged. SCB may contain several channel detections of one spike; row
IDs are candidates, not unique neuron events. This selects training examples
only. The streaming assignment pass below consumes waveform banks separately;
sampling must not discard detections in a dense pass.

The experimental [bounded unit-matching API](docs/continuous_unit_matching.md)
generates spatial candidate pairs and scores templates in fixed batches. It
does not yet assign cross-day identities or reproduce UnitMatch probabilities.

The [experimental streaming assignment pass](docs/streaming_sort.md) groups
SCB 0.2 waveform candidates, compares them with at most 32 calibration
templates per anchor contact, and updates clean, high-confidence templates
after each bounded epoch. Run `terasort-stream-sort --bank bank.scb.h5
--templates templates.npy --output new.assignments.h5 --backend cuda`
for direct CUDA matching; `--backend cpu` is its reference. Calibration
templates must share the SCB channel order and preprocessing frame. Unknown
events remain unassigned, and the current pass is not a validated Kilosort
replacement.

## Scale and evidence

The production accelerated sorter remains a whole-recording Kilosort run. Tiled
detection limits one large GPU intermediate, but Kilosort's global clustering,
feature collection and output stages are **not yet bounded by recording length**.
The opt-in `session-sort` path has bounded source cores, calibration samples,
waveform cache, and immutable restartable shards, but its sorting quality and
CUDA resource bounds need the full benchmark gates. Do not treat either route
as a validated hundred-terabyte sorter. See
[`docs/session_sort.md`](docs/session_sort.md) and
[`docs/bounded_sorting.md`](docs/bounded_sorting.md).

On one 600 s, 384-channel public synthetic recording, the C++/cuBLAS INT16 run
took 208.186 s against 214.379 s for the fresh `deep_tiled` INT16 control
(2.9% less worker time). The native run recovered 230/250 ground-truth neurons
at IoU ≥ 0.8 versus 229/250 for control. This is a single warmed pair, not a
general speed or accuracy guarantee. The retained
[`report`](docs/evidence/cublas_600s_report.md) and
[`summary`](docs/evidence/cublas_600s_summary.json) document the run. See also
[`docs/kilosort_cublas.md`](docs/kilosort_cublas.md).

## Development and license

Install test dependencies with `.venv\Scripts\python.exe -m pip install -e ".[test]"`,
then run `powershell -File scripts/test.ps1`. This uses a new project-local
temporary directory, avoiding stale shared pytest directories on Windows.
The source
guards refuse to patch an unverified Kilosort implementation. TeraSort's
adapters include transformations of Kilosort 4.1.7 GPL-3.0 code at runtime;
this package is distributed under GPL-3.0-only. See `LICENSE` and the
[Kilosort repository](https://github.com/MouseLand/Kilosort) for upstream
source and citation. Benchmark figures are reproduced from the PainProject
development trials; the packaged entry point and install path are tested
separately.
