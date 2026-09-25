"""Local-first browser dashboard for queued TeraSort runs."""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import psutil


STAGES = (
    ("Preprocessing", "Computing preprocessing variables", 2),
    ("Drift correction", "Computing drift correction", 19),
    ("Template detection", "Extracting spikes using templates", 18),
    ("First clustering", "First clustering", 15),
    ("Learned detection", "Extracting spikes using cluster waveforms", 18),
    ("Final clustering", "Final clustering", 16),
    ("Merging clusters", "Merging clusters", 5),
    ("Saving results", "Saving to phy and computing refractory periods", 7),
)
TOTAL_WEIGHT = sum(stage[2] for stage in STAGES)
MAX_BODY = 16 * 1024
LOG_TAIL_BYTES = 64 * 1024


def _iso(timestamp: float | None) -> str | None:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp else None


def _read_tail(path: Path, limit: int = LOG_TAIL_BYTES) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def _job_log(job: dict, state_dir: Path) -> str:
    worker_log = _read_tail(state_dir / job["id"] / "job.log")
    # Kilosort writes its stage messages to a separate file when the CLI does
    # not enable verbose console output. Read only its tail, never the raw data.
    kilosort_log = _read_tail(Path(job["request"]["results_dir"]) / "kilosort4.log")
    return kilosort_log + ("\n--- Worker output ---\n" + worker_log if worker_log else "")


def _latest_batch_progress(log: str) -> dict | None:
    """Extract the newest Kilosort tqdm batch counter and its pass ETA."""
    matches = list(re.finditer(r"([\d,]+)\s*/\s*([\d,]+)\s*\[([^\]\r\n]*)\]", log))
    if not matches:
        return None
    match = matches[-1]
    current, total = (int(value.replace(",", "")) for value in match.group(1, 2))
    if total <= 0 or current < 0 or current > total:
        return None
    eta = None
    # tqdm formats either MM:SS or H:MM:SS before the rate field.
    remaining = re.search(r"<\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s*,", match.group(3))
    if remaining:
        hours = int(remaining.group(1) or 0)
        eta = hours * 3600 + int(remaining.group(2)) * 60 + int(remaining.group(3))
    return {"current": current, "total": total, "eta_seconds": eta}


def _progress(log: str, elapsed: float, status: str, staged: bool = False) -> dict:
    if status == "completed":
        return {"stage": "Completed", "percent": 100, "eta_seconds": 0}
    if status == "failed":
        return {"stage": "Failed", "percent": None, "eta_seconds": None}
    if status == "cancelled":
        return {"stage": "Cancelled", "percent": None, "eta_seconds": None}
    if status == "queued":
        return {"stage": "Queued", "percent": 0, "eta_seconds": None}
    batches = _latest_batch_progress(log)
    if batches:
        percent = 100 * batches["current"] / batches["total"]
        return {"stage": "Batch processing", "percent": round(percent, 1),
                "progress_label": "Current pass",
                "progress_detail": f"{batches['current']:,} / {batches['total']:,} batches",
                "eta_label": "Current pass ETA", "eta_seconds": batches["eta_seconds"]}
    last = -1
    for index, (_, marker, _) in enumerate(STAGES):
        if marker.lower() in log.lower():
            last = index
    if last < 0:
        if staged:
            values = re.findall(r"TERASORT_STAGING (\d{1,3})", log)
            if values:
                fraction = min(100, int(values[-1]))
                eta = round(elapsed * (100 - fraction) / fraction) if 0 < fraction < 100 else None
                return {"stage": "Staging input" if fraction < 100 else "Initializing sorter",
                        "percent": round(fraction / 10), "eta_seconds": eta}
        return {"stage": "Starting", "percent": 0, "eta_seconds": None}
    completed_weight = sum(stage[2] for stage in STAGES[:last])
    percent = round((10 if staged else 0) + (90 if staged else 100) * completed_weight / TOTAL_WEIGHT)
    # A stage boundary is the only trustworthy progress signal in Kilosort's log.
    # Delay ETA until detection begins to avoid extrapolating from fast startup.
    eta = None
    if last >= 2 and completed_weight:
        remaining = TOTAL_WEIGHT - completed_weight
        staging_seconds = re.findall(r"TERASORT_STAGING_SECONDS ([0-9.]+)", log)
        compute_elapsed = max(0, elapsed - float(staging_seconds[-1])) if staged and staging_seconds else elapsed
        eta = max(0, round(compute_elapsed * remaining / completed_weight))
    return {"stage": STAGES[last][0], "percent": percent, "eta_seconds": eta}


def _drives() -> list[dict]:
    if os.name == "nt":
        drives = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            path = f"{letter}:\\"
            try:
                if Path(path).is_dir():
                    drives.append({"name": path, "path": path, "kind": "drive"})
            except OSError:
                continue
        return drives
    return [{"name": "/", "path": "/", "kind": "drive"}]


def browse(path: str | None, kinds: str = "all") -> dict:
    if not path:
        return {"path": None, "parent": None, "entries": _drives()}
    directory = Path(path).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Selected path is not a directory")
    entries = []
    truncated = False
    extensions = {"recordings": {".dat", ".bin"},
                  "intan": {".dat"},
                  "neuropixels": {".dat", ".bin"}}.get(kinds)
    for child in directory.iterdir():
        try:
            if child.is_dir():
                entries.append({"name": child.name, "path": str(child), "kind": "directory"})
            elif kinds != "directories" and child.is_file():
                if extensions is not None and child.suffix.lower() not in extensions:
                    continue
                entries.append({"name": child.name, "path": str(child), "kind": "file",
                                "bytes": child.stat().st_size})
        except (OSError, PermissionError):
            continue
        if len(entries) > 2000:
            truncated = True
            break
    entries.sort(key=lambda e: (e["kind"] == "file", e["name"].lower()))
    parent = None if directory.parent == directory else str(directory.parent)
    return {"path": str(directory), "parent": parent, "entries": entries[:2000],
            "truncated": truncated}


def find_dat_files_recursive(path: str) -> dict:
    """List .dat recordings under a selected folder in natural path order."""
    directory = Path(path).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Selected path is not a directory")

    inaccessible = 0

    def on_walk_error(_error: OSError) -> None:
        nonlocal inaccessible
        inaccessible += 1

    files = []
    for current, _, filenames in os.walk(
            directory, topdown=True, onerror=on_walk_error, followlinks=False):
        for filename in filenames:
            if Path(filename).suffix.lower() != ".dat":
                continue
            files.append(str(Path(current) / filename))
    files.sort(key=lambda item: _natural_path_key(str(Path(item).relative_to(directory))))
    return {"path": str(directory), "files": files, "count": len(files),
            "inaccessible_directories": inaccessible}


def _natural_path_key(value: str) -> tuple:
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r"(\d+)", value))


def recording_info(path: str, reserved_outputs: set[str] | None = None) -> dict:
    """Find a valid adjacent Neuroscope XML and a safe output folder suggestion."""
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in {".dat", ".bin"}:
        raise ValueError("Choose a raw INT16 .dat or .bin recording")
    xml_path = None
    candidates = (source.with_suffix(".xml"), source.parent / "amplifier.xml")
    seen = set()
    from .neuroscope import read_xml
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            try:
                read_xml(candidate)
            except (OSError, ValueError):
                continue
            xml_path = str(candidate.resolve())
            break
    reserved = {os.path.normcase(str(Path(p).resolve())) for p in (reserved_outputs or set())}
    base = f"{source.stem[:160]}_terasort"
    index = 1
    while True:
        name = base if index == 1 else f"{base}_{index:02d}"
        output = source.parent / name
        output_key = os.path.normcase(str(output.resolve()))
        occupied = output.exists() and (not output.is_dir() or any(output.iterdir()))
        if output_key not in reserved and not occupied:
            break
        index += 1
    return {"source": str(source), "xml_path": xml_path, "results_dir": str(output.resolve())}


def validate_request(data: dict, existing_outputs: set[str] | None = None) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    request = {}
    names = data.get("filenames")
    if names is None:
        names = [data.get("filename")]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("At least one filename is required")
    paths = []
    for name in names:
        path = Path(name).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Each filename must be a file")
        paths.append(str(path))
    if len({os.path.normcase(p) for p in paths}) != len(paths):
        raise ValueError("The same input file was selected more than once")
    request["filename"] = paths[0]
    request["filenames"] = paths
    if data.get("neuroscope_xml"):
        from .neuroscope import read_xml
        if data.get("settings"):
            raise ValueError("Choose Neuroscope XML or settings JSON, not both")
        metadata = read_xml(data["neuroscope_xml"], data.get("gain_uv_per_count"))
        settings = metadata["settings"]
        request["xml_import"] = metadata
        request["_xml_settings"] = settings
    else:
        value = data.get("settings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("settings or neuroscope_xml is required")
        settings_path = Path(value).expanduser().resolve(strict=True)
        if not settings_path.is_file():
            raise ValueError("settings must be a file")
        request["settings"] = str(settings_path)
        settings = json.loads(Path(request["settings"]).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or type(settings.get("n_chan_bin")) is not int or settings["n_chan_bin"] <= 0:
        raise ValueError("Settings must contain a positive integer n_chan_bin")
    for path in paths:
        source_bytes = Path(path).stat().st_size
        if not source_bytes or source_bytes % (2 * settings["n_chan_bin"]):
            raise ValueError(f"INT16 recording is empty or size is not divisible by n_chan_bin: {path}")
    if len(paths) > 1 and (not isinstance(settings.get("fs"), (int, float)) or settings["fs"] <= 0):
        raise ValueError("Multi-file sessions require positive fs in settings")
    output = data.get("results_dir")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("results_dir is required")
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("Results path must be a directory")
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("Results directory must be empty; choose a new directory")
    if existing_outputs and os.path.normcase(str(output_path)) in {os.path.normcase(p) for p in existing_outputs}:
        raise ValueError("Another job already uses this results directory")
    request["results_dir"] = str(output_path)
    if data.get("stage_dir"):
        if not isinstance(data["stage_dir"], str):
            raise ValueError("stage_dir must be a path")
        stage_path = Path(data["stage_dir"]).expanduser().resolve()
        if stage_path.exists():
            raise ValueError("Staging directory must be new")
        if existing_outputs and os.path.normcase(str(stage_path)) in {os.path.normcase(p) for p in existing_outputs}:
            raise ValueError("Another job already uses this staging directory")
        if stage_path.is_relative_to(output_path) or output_path.is_relative_to(stage_path):
            raise ValueError("Staging and results directories must be separate")
        if any(stage_path.is_relative_to(Path(path).parent) for path in paths):
            raise ValueError("Staging directory must be outside input folders")
        request["stage_dir"] = str(stage_path)
    probe_json = data.get("probe_json")
    probe_name = data.get("probe_name")
    auto_xml_probe = bool(request.get("xml_import")) and not probe_json and not probe_name
    if not auto_xml_probe and bool(probe_json) == bool(probe_name):
        raise ValueError("Select exactly one probe JSON or bundled probe name")
    if probe_json:
        probe_path = Path(probe_json).expanduser().resolve(strict=True)
        if not probe_path.is_file() or not isinstance(json.loads(probe_path.read_text(encoding="utf-8")), dict):
            raise ValueError("probe_json must contain a JSON object")
        request["probe_json"] = str(probe_path)
    elif isinstance(probe_name, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", probe_name):
        request["probe_name"] = probe_name
    elif not auto_xml_probe:
        raise ValueError("Invalid bundled probe name")
    if request.get("xml_import"):
        from .neuroscope import configure_probe, generate_probe
        if auto_xml_probe:
            request["_xml_probe"] = generate_probe(request["xml_import"])
            request["xml_import"]["probe_layout"] = "staggered: MATLAB wrapper convention"
        elif probe_json:
            request["_xml_probe"] = configure_probe(request["xml_import"],
                json.loads(Path(request["probe_json"]).read_text(encoding="utf-8")))
            request["xml_import"]["probe_layout"] = "custom probe JSON"
        else:
            raise ValueError("Leave probe fields empty for automatic XML geometry, or supply a custom probe JSON")
    backend = data.get("backend", "auto")
    if backend not in ("auto", "standard", "deep_tiled", "cublas"):
        raise ValueError("Invalid backend")
    request["backend"] = backend
    if data.get("read_cache_dir"):
        cache = Path(data["read_cache_dir"]).expanduser().resolve()
        if data.get("stage_dir") or backend == "standard" or data.get("no_fast_int16"):
            raise ValueError('Read cache needs fast INT16 mode and cannot combine with full staging')
        from .api import _select_backend
        try:
            cache_backend = _select_backend(backend, None, settings)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        if cache_backend == "standard":
            raise ValueError('Read cache needs the cublas or deep_tiled backend; auto selected standard Kilosort')
        if any(cache.is_relative_to(Path(p).parent) for p in paths):
            raise ValueError('Read cache must be outside source folders')
        if cache.is_relative_to(output_path) or output_path.is_relative_to(cache):
            raise ValueError('Read cache and results must be separate')
        mb,slots = data.get("read_cache_mb",4096),data.get("read_cache_slots",2)
        workers = data.get("read_cache_workers",4)
        if (type(mb) is not int or not 1<=mb<=4096 or type(slots) is not int or not 2<=slots<=16
                or type(workers) is not int or not 1<=workers<=16):
            raise ValueError('Read cache requires 1–4096 MiB blocks, 2–16 slots, and 1–16 download workers')
        request.update(read_cache_dir=str(cache),read_cache_mb=mb,read_cache_slots=slots,
                       read_cache_workers=workers)
    for key in ("skip_drift_correction", "no_fast_int16"):
        request[key] = bool(data.get(key, False))
    if data.get("lfp_output"):
        if request.get("xml_import") and "scale" not in settings:
            raise ValueError("Specify gain in microvolts per count for XML-configured LFP export")
        if len(paths) > 1:
            raise ValueError("Parallel LFP output currently supports one source file per run")
        lfp_path = Path(data["lfp_output"]).expanduser().resolve()
        if lfp_path.exists():
            raise ValueError("LFP output already exists")
        if lfp_path in map(Path, paths):
            raise ValueError("LFP output cannot overwrite the input")
        request["lfp_output"] = str(lfp_path)
    return request


class JobManager:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = set()
        self.max_concurrent_jobs = 1
        self.settings_path = self.state_dir / "dashboard_settings.json"
        self._load_settings()
        self.metrics = {"system_cpu_percent": None, "system_ram_percent": None,
                        "gpu_percent": None, "gpu_memory_mib": None,
                        "gpu_memory_total_mib": None, "job_ram_mib": None}
        self._load()
        self.thread = threading.Thread(target=self._loop, name="terasort-web-scheduler", daemon=True)
        self.thread.start()

    def _load_settings(self) -> None:
        try:
            settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
            value = settings.get("max_concurrent_jobs", 1)
            if type(value) is int and 1 <= value <= 4:
                self.max_concurrent_jobs = value
        except (OSError, ValueError, AttributeError):
            pass

    def settings(self) -> dict:
        with self.lock:
            return {"max_concurrent_jobs": self.max_concurrent_jobs,
                    "active_jobs": len(self.active),
                    "queued_jobs": sum(job["status"] == "queued" for job in self.jobs.values())}

    def update_settings(self, data: dict) -> dict:
        value = data.get("max_concurrent_jobs") if isinstance(data, dict) else None
        if type(value) is not int or not 1 <= value <= 4:
            raise ValueError("Concurrent sessions must be an integer from 1 to 4")
        with self.lock:
            self.max_concurrent_jobs = value
            temporary = self.settings_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"max_concurrent_jobs": value}, indent=2), encoding="utf-8")
            temporary.replace(self.settings_path)
            return self.settings()

    def _job_dir(self, job_id: str) -> Path:
        return self.state_dir / job_id

    def _save(self, job: dict) -> None:
        path = self._job_dir(job["id"]) / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load(self) -> None:
        for path in self.state_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["id"] != path.parent.name:
                    continue
                done = path.parent / "done.json"
                if done.exists():
                    result = json.loads(done.read_text(encoding="utf-8"))
                    job["status"] = "completed" if result["exit_code"] == 0 else "failed"
                    job["finished_at"] = result["finished_at"]
                elif job["status"] == "running":
                    if self._alive(job):
                        self.active.add(job["id"])
                    else:
                        job["status"] = "failed"
                        job["error"] = "Worker stopped without a completion record"
                self.jobs[job["id"]] = job
                self._save(job)
            except (OSError, ValueError, KeyError):
                continue

    @staticmethod
    def _alive(job: dict) -> bool:
        try:
            process = psutil.Process(job["pid"])
            return abs(process.create_time() - job["process_created_at"]) < 2 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.Error, KeyError):
            return False

    def add(self, data: dict) -> dict:
        with self.lock:
            outputs = {os.path.normcase(job["request"]["results_dir"]) for job in self.jobs.values()
                       if job["status"] in ("queued", "running")}
            outputs.update(os.path.normcase(job["request"]["stage_dir"])
                           for job in self.jobs.values()
                           if job["status"] in ("queued", "running") and job["request"].get("stage_dir"))
            request = validate_request(data, outputs)
            job_id = uuid.uuid4().hex[:12]
            directory = self._job_dir(job_id)
            directory.mkdir()
            if "_xml_settings" in request:
                for key, filename in (("settings", "settings.json"), ("probe_json", "probe.json")):
                    source_key = "_xml_settings" if key == "settings" else "_xml_probe"
                    target = directory / filename
                    target.write_text(json.dumps(request.pop(source_key), indent=2), encoding="utf-8")
                    request[key] = str(target)
                (directory / "xml_import.json").write_text(
                    json.dumps(request["xml_import"], indent=2), encoding="utf-8")
            job = {"id": job_id, "status": "queued", "created_at": time.time(),
                   "started_at": None, "finished_at": None, "request": request}
            self.jobs[job_id] = job
            self._save(job)
            return self.view(job_id, include_log=False)

    @staticmethod
    def _retry_path(original: str, reserved: set[str], *, file_path: bool = False) -> str:
        path = Path(original).expanduser().resolve()
        for attempt in range(1, 10000):
            suffix = "_retry" if attempt == 1 else f"_retry_{attempt:02d}"
            if file_path:
                candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
            else:
                candidate = path.with_name(f"{path.name}{suffix}")
            key = os.path.normcase(str(candidate.resolve()))
            if key not in reserved and not candidate.exists():
                return str(candidate)
        raise ValueError("Could not find an unused retry output path")

    def retry(self, job_id: str) -> dict:
        with self.lock:
            original = self.jobs[job_id]
            if original["status"] not in ("failed", "cancelled"):
                raise ValueError("Only failed or cancelled jobs can be retried")
            if self._alive(original):
                raise ValueError("The worker is still stopping; retry after it exits")
            existing = next((job for job in self.jobs.values()
                             if job.get("retry_of") == job_id and
                             job["status"] in ("queued", "running")), None)
            if existing is not None:
                return self.view(existing["id"], include_log=False)

            request = original["request"]
            reserved = set()
            for job in self.jobs.values():
                for key in ("results_dir", "stage_dir", "lfp_output"):
                    value = job["request"].get(key)
                    if value:
                        reserved.add(os.path.normcase(str(Path(value).expanduser().resolve())))

            results_dir = self._retry_path(request["results_dir"], reserved)
            payload = {"filenames": request.get("filenames", [request["filename"]]),
                       "settings": request.get("settings"),
                       "probe_json": request.get("probe_json"),
                       "probe_name": request.get("probe_name"),
                       "results_dir": results_dir,
                       "backend": request.get("backend", "auto"),
                       "skip_drift_correction": request.get("skip_drift_correction", False),
                       "no_fast_int16": request.get("no_fast_int16", False)}
            for key in ("read_cache_dir", "read_cache_mb", "read_cache_slots", "read_cache_workers"):
                if key in request:
                    payload[key] = request[key]
            if request.get("stage_dir"):
                payload["stage_dir"] = self._retry_path(request["stage_dir"], reserved)
            if request.get("lfp_output"):
                old_output = Path(request["lfp_output"]).expanduser().resolve()
                old_results = Path(request["results_dir"]).expanduser().resolve()
                try:
                    relative_output = old_output.relative_to(old_results)
                except ValueError:
                    payload["lfp_output"] = self._retry_path(
                        str(old_output), reserved, file_path=True)
                else:
                    payload["lfp_output"] = str(Path(results_dir) / relative_output)

            retried = self.add(payload)
            job = self.jobs[retried["id"]]
            job["retry_of"] = job_id
            if request.get("xml_import"):
                job["request"]["xml_import"] = request["xml_import"]
                (self._job_dir(job["id"]) / "xml_import.json").write_text(
                    json.dumps(request["xml_import"], indent=2), encoding="utf-8")
            self._save(job)
            return self.view(job["id"], include_log=False)

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] in ("queued", "running"):
                if job["status"] == "running" and self._alive(job):
                    process = psutil.Process(job["pid"])
                    children = process.children(recursive=True)
                    for child in reversed(children):
                        try:
                            child.terminate()
                        except psutil.Error:
                            pass
                    process.terminate()
                job["status"] = "cancelled"
                job["finished_at"] = time.time()
                self.active.discard(job_id)
                self._save(job)
            return self.view(job_id, include_log=False)

    def view(self, job_id: str, include_log: bool = True) -> dict:
        with self.lock:
            job = dict(self.jobs[job_id])
            log = _job_log(job, self.state_dir)
            elapsed = (job.get("finished_at") or time.time()) - (job.get("started_at") or time.time())
            job.update(_progress(log, elapsed, job["status"], bool(job["request"].get("stage_dir"))))
            job["created_at"] = _iso(job["created_at"])
            job["started_at"] = _iso(job.get("started_at"))
            job["finished_at"] = _iso(job.get("finished_at"))
            job["elapsed_seconds"] = round(max(elapsed, 0)) if job.get("started_at") else 0
            if include_log:
                job["log"] = log[-12000:]
                job["metrics"] = dict(self.metrics) if job["status"] == "running" else None
            return job

    def list(self) -> list[dict]:
        with self.lock:
            return [self.view(job["id"], include_log=False)
                    for job in sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True)]

    def _launch(self, job: dict) -> None:
        directory = self._job_dir(job["id"])
        log = (directory / "job.log").open("ab", buffering=0)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        environment = os.environ.copy()
        if not environment.get("CUPY_CACHE_DIR"):
            cache_dir = self.state_dir / "cupy_cache"
            cache_dir.mkdir(exist_ok=True)
            environment["CUPY_CACHE_DIR"] = str(cache_dir)
        try:
            process = subprocess.Popen([sys.executable, "-u", "-m", "terasort.web_worker",
                                        str(directory / "job.json")], stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, creationflags=flags, env=environment)
        finally:
            log.close()
        job["status"] = "running"
        job["started_at"] = time.time()
        job["pid"] = process.pid
        job["process_created_at"] = psutil.Process(process.pid).create_time()
        self.active.add(job["id"])
        self._save(job)

    def _refresh(self) -> None:
        for job_id in tuple(self.active):
            job = self.jobs[job_id]
            done = self._job_dir(job["id"]) / "done.json"
            if done.exists():
                result = json.loads(done.read_text(encoding="utf-8"))
                job["status"] = "completed" if result["exit_code"] == 0 else "failed"
                job["finished_at"] = result["finished_at"]
                self._save(job)
                self.active.discard(job_id)
            elif not self._alive(job):
                job["status"] = "failed"
                job["finished_at"] = time.time()
                job["error"] = "Worker stopped without a completion record"
                self._save(job)
                self.active.discard(job_id)
        pending = sorted((j for j in self.jobs.values() if j["status"] == "queued"),
                         key=lambda j: j["created_at"])
        while pending and len(self.active) < self.max_concurrent_jobs:
            job = pending.pop(0)
            try:
                self._launch(job)
            except (OSError, psutil.Error) as exc:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = time.time()
                self._save(job)

    def _sample(self) -> None:
        metrics = {"system_cpu_percent": psutil.cpu_percent(interval=None),
                   "system_ram_percent": psutil.virtual_memory().percent,
                   "gpu_percent": None, "gpu_memory_mib": None,
                   "gpu_memory_total_mib": None, "job_ram_mib": None}
        with self.lock:
            active = tuple(self.active)
        ram_bytes = 0
        for job_id in active:
            try:
                process = psutil.Process(self.jobs[job_id]["pid"])
                ram_bytes += sum(p.memory_info().rss for p in
                                 [process, *process.children(recursive=True)])
            except psutil.Error:
                continue
        if ram_bytes:
            metrics["job_ram_mib"] = round(ram_bytes / 1048576)
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                    timeout=2, check=True)
            first = result.stdout.strip().splitlines()[0].split(",")
            metrics["gpu_percent"], metrics["gpu_memory_mib"], metrics["gpu_memory_total_mib"] = [int(v.strip()) for v in first]
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        self.metrics = metrics

    def _loop(self) -> None:
        while True:
            with self.lock:
                self._refresh()
            self._sample()
            time.sleep(3)


def create_handler(manager: JobManager, token: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TeraSortWeb/0.3"

        def log_message(self, format, *args):
            # The browser polls every three seconds; avoid flooding the terminal.
            return

        def _json(self, status: int, data: object) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            host = self.headers.get("Host", "").lower()
            hostname = host.split("]", 1)[0] + "]" if host.startswith("[") else host.split(":", 1)[0]
            if not token and hostname not in ("127.0.0.1", "localhost", "[::1]"):
                self._json(403, {"error": "Localhost Host header required"})
                return False
            if token and not secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                self._json(401, {"error": "Bearer token required"})
                return False
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self._json(403, {"error": "Cross-origin request rejected"})
                return False
            return True

        def do_GET(self) -> None:
            split = urlsplit(self.path)
            if split.path == "/":
                html = (Path(__file__).parent / "webui" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers()
                self.wfile.write(html)
                return
            if not self._authorized():
                return
            try:
                if split.path == "/api/health":
                    self._json(200, {"ok": True, "token_required": bool(token)})
                elif split.path == "/api/metrics":
                    self._json(200, dict(manager.metrics))
                elif split.path == "/api/settings":
                    self._json(200, manager.settings())
                elif split.path == "/api/runtime":
                    self._json(200, {"platform": sys.platform,
                                     "read_cache_default_dir": r"F:\TeraSortReadCache" if sys.platform == "win32" else ""})
                elif split.path == "/api/browse":
                    query = parse_qs(split.query)
                    self._json(200, browse(query.get("path", [None])[0], query.get("kind", ["all"])[0]))
                elif split.path == "/api/recordings-recursive":
                    query = parse_qs(split.query)
                    self._json(200, find_dat_files_recursive(query.get("path", [""])[0]))
                elif split.path == "/api/recording-info":
                    query = parse_qs(split.query)
                    with manager.lock:
                        reserved = {job["request"]["results_dir"] for job in manager.jobs.values()}
                    self._json(200, recording_info(query.get("path", [""])[0], reserved))
                elif split.path == "/api/neuroscope":
                    from .neuroscope import read_xml, generate_probe
                    query = parse_qs(split.query)
                    metadata = read_xml(query.get("path", [""])[0], query.get("gain", [None])[0])
                    metadata["generated_probe"] = generate_probe(metadata)
                    self._json(200, metadata)
                elif split.path == "/api/jobs":
                    self._json(200, {"jobs": manager.list()})
                elif re.fullmatch(r"/api/jobs/[0-9a-f]{12}", split.path):
                    self._json(200, manager.view(split.path.rsplit("/", 1)[-1]))
                else:
                    self._json(404, {"error": "Not found"})
            except KeyError:
                self._json(404, {"error": "Job not found"})
            except (OSError, ValueError) as exc:
                self._json(400, {"error": str(exc)})

        def do_POST(self) -> None:
            if not self._authorized():
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                self._json(415, {"error": "Content-Type must be application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY:
                    self._json(413, {"error": "Invalid request size"})
                    return
                data = json.loads(self.rfile.read(length))
                if self.path == "/api/jobs":
                    self._json(201, manager.add(data))
                elif self.path == "/api/settings":
                    self._json(200, manager.update_settings(data))
                elif re.fullmatch(r"/api/jobs/[0-9a-f]{12}/retry", self.path):
                    self._json(201, manager.retry(self.path.split("/")[3]))
                elif re.fullmatch(r"/api/jobs/[0-9a-f]{12}/cancel", self.path):
                    self._json(200, manager.cancel(self.path.split("/")[3]))
                else:
                    self._json(404, {"error": "Not found"})
            except KeyError:
                self._json(404, {"error": "Job not found"})
            except (OSError, ValueError, TypeError) as exc:
                self._json(400, {"error": str(exc)})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, state_dir: Path | None = None,
          token: str | None = None, open_browser: bool = True) -> None:
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise ValueError("A bearer token is required when binding beyond localhost")
    manager = JobManager(state_dir or Path.home() / ".terasort" / "web")
    server = ThreadingHTTPServer((host, port), create_handler(manager, token))
    url = f"http://{host}:{server.server_port}/"
    print(f"TeraSort dashboard: {url}", flush=True)
    print(f"Job state: {manager.state_dir}", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
