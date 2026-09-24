"""Validated, explicit source clock for a multi-file, multi-probe session.

The source files are always opened read-only. Gaps are first-class intervals:
the sorter never pretends that two files are contiguous unless the manifest
says so.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a portable, nonempty identifier")
    return value


@dataclass(frozen=True)
class Segment:
    path: Path
    start_sample: int
    n_samples: int
    day_id: str

    @property
    def stop_sample(self):
        return self.start_sample + self.n_samples


@dataclass(frozen=True)
class Gap:
    start_sample: int
    stop_sample: int
    reason: str


@dataclass(frozen=True)
class Probe:
    probe_id: str
    sample_rate_hz: float
    gain_uv_per_count: float
    x_um: np.ndarray
    y_um: np.ndarray
    shank: np.ndarray
    segments: tuple[Segment, ...]
    gaps: tuple[Gap, ...]
    bad_channels: tuple[int, ...]
    seed_templates: Path | None
    seed_templates_by_day: dict[str, Path]
    seed_preprocessing_id: str | None

    @property
    def n_channels(self):
        return len(self.x_um)

    @property
    def stop_sample(self):
        return self.segments[-1].stop_sample

    @property
    def geometry(self):
        return np.column_stack((self.x_um, self.y_um))

    def seed_for_day(self, day_id):
        return self.seed_templates_by_day.get(day_id, self.seed_templates)


@dataclass(frozen=True)
class Session:
    session_id: str
    probes: tuple[Probe, ...]
    manifest_path: Path

    def as_source_record(self):
        def file_state(path):
            if path is None:
                return None
            stat = path.stat()
            return {"path": str(path), "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns}

        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": hashlib.sha256(
                self.manifest_path.read_bytes()).hexdigest(),
            "probes": [{
                "probe_id": probe.probe_id,
                "sample_rate_hz": probe.sample_rate_hz,
                "gain_uv_per_count": probe.gain_uv_per_count,
                "geometry": probe.geometry.tolist(),
                "shank": probe.shank.tolist(),
                "bad_channels": list(probe.bad_channels),
                "seed_templates": file_state(probe.seed_templates),
                "seed_templates_by_day": {
                    day: file_state(seed) for day, seed in
                    sorted(probe.seed_templates_by_day.items())},
                "segments": [{
                    "path": str(seg.path), "start_sample": seg.start_sample,
                    "n_samples": seg.n_samples, "day_id": seg.day_id,
                    "source_bytes": seg.path.stat().st_size,
                    "source_mtime_ns": seg.path.stat().st_mtime_ns,
                } for seg in probe.segments],
                "gaps": [vars(gap) for gap in probe.gaps],
            } for probe in self.probes],
        }


def load_session(path, *, verify_sources=True):
    """Load a JSON manifest and validate every source boundary and geometry."""
    path = Path(path).expanduser().resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Expected session manifest schema_version=1")
    session_id = _identifier(raw.get("session_id"), "session_id")
    items = raw.get("probes")
    if not isinstance(items, list) or not items:
        raise ValueError("A session needs at least one probe")
    probes = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Probe must be an object")
        probe_id = _identifier(item.get("probe_id"), "probe_id")
        if any(probe.probe_id == probe_id for probe in probes):
            raise ValueError(f"Duplicate probe_id {probe_id}")
        rate = float(item.get("sample_rate_hz", float("nan")))
        gain = float(item.get("gain_uv_per_count", float("nan")))
        if not np.isfinite(rate) or rate <= 0 or not np.isfinite(gain) or gain <= 0:
            raise ValueError("Positive finite sample rate and gain required")
        geometry = item.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError("Probe geometry required")
        x = np.asarray(geometry.get("x_um", []), dtype=np.float64)
        y = np.asarray(geometry.get("y_um", []), dtype=np.float64)
        shank = np.asarray(geometry.get("shank", []))
        if (x.ndim != 1 or not len(x) or y.shape != x.shape
                or shank.shape != x.shape or shank.dtype.kind not in "iu"
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("Finite x/y geometry and integer shank per channel required")
        n_channels = len(x)
        bad = item.get("bad_channels", [])
        if (not isinstance(bad, list) or
                any(isinstance(c, bool) or not isinstance(c, int)
                    or c < 0 or c >= n_channels for c in bad) or
                len(set(bad)) != len(bad)):
            raise ValueError("bad_channels must be unique recording-channel indices")
        source_items = item.get("segments")
        if not isinstance(source_items, list) or not source_items:
            raise ValueError("Probe needs one or more source segments")
        segments = []
        for source in source_items:
            if not isinstance(source, dict):
                raise ValueError("Segment must be an object")
            name = source.get("path")
            if not isinstance(name, str) or not name:
                raise ValueError("Segment path required")
            source_path = Path(name).expanduser()
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            source_path = source_path.resolve()
            first = _integer(source.get("start_sample"), "start_sample")
            count = _integer(source.get("n_samples"), "n_samples", 1)
            day_id = _identifier(source.get("day_id"), "day_id")
            if first > np.iinfo(np.int64).max - count:
                raise ValueError("Source clock exceeds int64")
            if verify_sources:
                expected = count * n_channels * np.dtype("<i2").itemsize
                if not source_path.is_file() or source_path.stat().st_size != expected:
                    raise ValueError(f"Missing or incorrectly sized source: {source_path}")
            segments.append(Segment(source_path, first, count, day_id))
        segments.sort(key=lambda seg: seg.start_sample)
        if segments[0].start_sample != 0:
            raise ValueError("Probe source clock must begin at sample zero")
        gap_items = item.get("gaps", [])
        if not isinstance(gap_items, list):
            raise ValueError("gaps must be an array")
        gaps = []
        for gap in gap_items:
            if not isinstance(gap, dict):
                raise ValueError("Gap must be an object")
            first = _integer(gap.get("start_sample"), "gap start_sample")
            stop = _integer(gap.get("stop_sample"), "gap stop_sample", first + 1)
            reason = gap.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Gap reason required")
            gaps.append(Gap(first, stop, reason))
        gaps.sort(key=lambda gap: gap.start_sample)
        expected_gaps = []
        for left, right in zip(segments, segments[1:]):
            if right.start_sample < left.stop_sample:
                raise ValueError("Overlapping source segments")
            if right.start_sample > left.stop_sample:
                expected_gaps.append((left.stop_sample, right.start_sample))
        if [(gap.start_sample, gap.stop_sample) for gap in gaps] != expected_gaps:
            raise ValueError("Every source-clock gap must be listed exactly once")
        seed = item.get("seed_templates")
        if seed is not None:
            if not isinstance(seed, str) or not seed:
                raise ValueError("seed_templates must be a path")
            seed = Path(seed).expanduser()
            if not seed.is_absolute():
                seed = path.parent / seed
            seed = seed.resolve()
            if verify_sources and not seed.is_file():
                raise ValueError(f"Missing calibration templates: {seed}")
        seeds_by_day = {}
        mapping = item.get("seed_templates_by_day", {})
        if not isinstance(mapping, dict):
            raise ValueError("seed_templates_by_day must be an object")
        days = {segment.day_id for segment in segments}
        for day_id, name in mapping.items():
            _identifier(day_id, "seed day_id")
            if day_id not in days or not isinstance(name, str) or not name:
                raise ValueError("Seed template day must name a source day")
            seed_path = Path(name).expanduser()
            if not seed_path.is_absolute():
                seed_path = path.parent / seed_path
            seed_path = seed_path.resolve()
            if verify_sources and not seed_path.is_file():
                raise ValueError(f"Missing calibration templates: {seed_path}")
            seeds_by_day[day_id] = seed_path
        preprocessing_id = item.get("seed_preprocessing_id")
        if seed is not None or seeds_by_day:
            if preprocessing_id != "terasort-session-v1":
                raise ValueError(
                    "External templates must declare seed_preprocessing_id="
                    "terasort-session-v1 and use the session preprocessing frame")
        probes.append(Probe(probe_id, rate, gain, x, y, shank,
                            tuple(segments), tuple(gaps), tuple(bad), seed,
                            seeds_by_day, preprocessing_id))
    return Session(session_id, tuple(probes), path)
