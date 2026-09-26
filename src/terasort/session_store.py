"""Immutable session shards with a partial-coverage waveform cache and checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .candidates.bank import EVENT_DTYPE
from .candidates.waveforms import WaveformSpec, encode_waveforms, extract_waveforms


CANDIDATE_DTYPE = np.dtype([
    ("sample_index", "<i8"), ("channel_index", "<u4"),
    ("amplitude_uv", "<f4"), ("snr", "<f4"),
    ("qc_flags", "u1"), ("unit_id", "<i8"),
    ("score", "<f4"), ("runner_up", "<f4"),
    ("fitted_amplitude", "<f4"), ("residual_gain", "<f4"),
    ("pass_index", "<i2"),
])
SPIKE_DTYPE = np.dtype([
    ("sample_index", "<i8"), ("channel_index", "<u4"),
    ("unit_id", "<i8"), ("score", "<f4"), ("runner_up", "<f4"),
    ("fitted_amplitude", "<f4"), ("residual_gain", "<f4"),
    ("pass_index", "<i2"), ("candidate_row", "<i8"),
    ("model_version", "<i4"),
])
QC_DTYPE = np.dtype([
    ("start_sample", "<i8"), ("stop_sample", "<i8"),
    ("interval_bad", "u1"), ("source_bytes_read", "<i8"),
])


def _dataset(handle, name, dtype, chunk=65_536):
    return handle.create_dataset(
        name, shape=(0,), maxshape=(None,), dtype=dtype,
        chunks=(chunk,), compression="lzf", shuffle=True)


def _append(dataset, values):
    values = np.asarray(values, dtype=dataset.dtype)
    first = len(dataset)
    dataset.resize((first + len(values),) + dataset.shape[1:])
    if len(values):
        dataset[first:] = values
    return first


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class RunJournal:
    """A run directory is new, or a resume with the exact same source/config."""

    def __init__(self, root, source_record, config, *, resume=False):
        self.root = Path(root).expanduser().resolve()
        self.record = {"schema_version": 1, "source": source_record,
                       "config": config, "config_sha256": config_digest(config)}
        path = self.root / "run.json"
        if resume:
            if not path.is_file() or json.loads(path.read_text()) != self.record:
                raise ValueError("Resume source or configuration differs from run.json")
        else:
            if self.root.exists():
                raise FileExistsError(self.root)
            self.root.mkdir(parents=True)
            partial = path.with_suffix(".json.partial")
            partial.write_text(json.dumps(self.record, indent=2) + "\n",
                               encoding="utf-8")
            os.replace(partial, path)

    def shard_path(self, probe_id, start_sample, stop_sample):
        return self.root / probe_id / f"{start_sample:016d}-{stop_sample:016d}.h5"

    def completed(self, path):
        path = Path(path)
        if not path.is_file():
            return False
        with h5py.File(path, "r") as handle:
            if not handle.attrs.get("complete", False):
                raise ValueError(f"Published shard is incomplete: {path}")
            if handle.attrs.get("config_sha256") != self.record["config_sha256"]:
                raise ValueError(f"Shard configuration differs: {path}")
        return True

    def latest_model(self, probe_id, before_sample):
        folder = self.root / probe_id
        if not folder.is_dir():
            return None
        found = []
        for path in folder.glob("*.h5"):
            try:
                start, stop = map(int, path.stem.split("-"))
            except ValueError:
                raise ValueError(f"Unknown shard filename: {path}")
            if stop <= before_sample:
                found.append((stop, path))
        if not found:
            return None
        stop, path = max(found)
        self.completed(path)
        with h5py.File(path, "r") as handle:
            group = handle["model_after"]
            return stop, (group["waveforms"][:], group["channels"][:],
                          group["anchors"][:], group["assigned"][:],
                          int(group.attrs["version"]))


class ShardWriter:
    """SCB 0.3 candidate metadata, sparse waveform cache, TSA-like spikes."""

    def __init__(self, path, *, probe, day_id, start_sample, stop_sample,
                 config_sha256, waveform_scale_uv=1.):
        self.path = Path(path)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial, "x")
        h = self.handle
        h.attrs["complete"] = False
        h.attrs["format"] = "TeraSortSessionShard"
        h.attrs["version"] = 1
        h.attrs["candidate_format"] = "SCB0.3"
        h.attrs["candidate_coverage"] = "all threshold detections; waveforms selective"
        h.attrs["probe_id"] = probe.probe_id
        h.attrs["day_id"] = day_id
        h.attrs["start_sample"] = int(start_sample)
        h.attrs["stop_sample"] = int(stop_sample)
        h.attrs["config_sha256"] = config_sha256
        h.attrs["waveform_scale_uv"] = float(waveform_scale_uv)
        self.candidates = _dataset(h, "candidates", CANDIDATE_DTYPE)
        self.spikes = _dataset(h, "spikes", SPIKE_DTYPE)
        self.qc = _dataset(h, "qc", QC_DTYPE, chunk=1024)
        nchan = probe.n_channels
        self.noise = h.create_dataset("qc_noise_uv", (0, nchan), maxshape=(None, nchan),
                                      dtype="<f4", chunks=(1, nchan), compression="lzf")
        self.flags = h.create_dataset("qc_channel_flags", (0, nchan),
                                      maxshape=(None, nchan), dtype="u1",
                                      chunks=(1, nchan), compression="lzf")
        self.saturated = h.create_dataset(
            "qc_saturated_fraction", (0, nchan), maxshape=(None, nchan),
            dtype="<f4", chunks=(1, nchan), compression="lzf")
        self.flatline = h.create_dataset(
            "qc_flatline_fraction", (0, nchan), maxshape=(None, nchan),
            dtype="<f4", chunks=(1, nchan), compression="lzf")
        self.zero = h.create_dataset(
            "qc_zero_fraction", (0, nchan), maxshape=(None, nchan),
            dtype="<f4", chunks=(1, nchan), compression="lzf")
        self.position = h.create_dataset("channel_positions_um",
                                         data=probe.geometry.astype(np.float32))
        self.shank = h.create_dataset("channel_shank", data=probe.shank.astype(np.int32))
        self.cache_rows = _dataset(h, "waveform_candidate_row", "<i8")
        self.cache_spec = None
        self.cache_data = None
        self.cached_bytes = 0
        self.waveform_scale_uv = waveform_scale_uv

    def _ensure_waveforms(self, channel_map):
        if self.cache_data is not None:
            return
        self.cache_spec = WaveformSpec(channel_map, 30, 30,
                                       self.waveform_scale_uv, "int16")
        width = self.cache_spec.width
        self.handle.create_dataset("waveform_channel_index",
                                   data=self.cache_spec.channel_index)
        self.cache_data = self.handle.create_dataset(
            "waveform_data", shape=(0, 61, width),
            maxshape=(None, 61, width), dtype="<i2",
            chunks=(max(1, min(512, 1_048_576 // (61 * width * 2))), 61, width),
            compression="lzf", shuffle=True)
        self.cache_start = _dataset(self.handle, "waveform_valid_start", "<i4")
        self.cache_stop = _dataset(self.handle, "waveform_valid_stop", "<i4")

    def append_core(self, *, core, voltage_uv, quality, threshold_events,
                    matches, channel_map, cache_fraction=.05,
                    model_version=0):
        """Append one core. Every threshold event gets metadata, no silent cap."""
        first = len(self.candidates)
        # A residual redetection may rediscover the same peak. Keep its first
        # observation, while retaining peaks that appeared only after fitting.
        unique_events = {}
        for event in threshold_events:
            if core.core_start <= core.data_start + event[0] < core.core_stop:
                unique_events.setdefault((int(event[0]), int(event[1])), event)
        core_events = list(unique_events.values())
        candidates = np.empty(len(core_events), CANDIDATE_DTYPE)
        candidates["unit_id"] = -1
        candidates["score"] = -1.
        candidates["runner_up"] = -1.
        candidates["fitted_amplitude"] = 0.
        candidates["residual_gain"] = 0.
        candidates["pass_index"] = -1
        lookup = {}
        for i, (t, channel, snr) in enumerate(core_events):
            absolute = core.data_start + int(t)
            value = float(voltage_uv[int(t), int(channel)])
            candidates[i]["sample_index"] = absolute
            candidates[i]["channel_index"] = int(channel)
            candidates[i]["amplitude_uv"] = value
            candidates[i]["snr"] = float(snr)
            candidates[i]["qc_flags"] = quality.flags[int(channel)]
            lookup[(absolute, int(channel))] = i
        spikes = np.empty(len(matches), SPIKE_DTYPE)
        for i, match in enumerate(matches):
            absolute = core.data_start + match.source_sample
            candidate_absolute = (core.data_start + match.candidate_sample
                                  if match.candidate_sample is not None else absolute)
            row = lookup.get((candidate_absolute, match.channel), -1)
            spikes[i] = (absolute, match.channel, match.unit, match.score,
                         match.runner_up, match.amplitude, match.residual_gain,
                         match.pass_index, first + row if row >= 0 else -1,
                         model_version)
            if row >= 0:
                candidates[row]["unit_id"] = match.unit
                candidates[row]["score"] = match.score
                candidates[row]["runner_up"] = match.runner_up
                candidates[row]["fitted_amplitude"] = match.amplitude
                candidates[row]["residual_gain"] = match.residual_gain
                candidates[row]["pass_index"] = match.pass_index
        _append(self.candidates, candidates)
        _append(self.spikes, spikes)
        _append(self.qc, [(core.core_start, core.core_stop,
                           int(quality.interval_bad), core.source_bytes_read)])
        n = len(self.qc)
        self.noise.resize((n, len(quality.noise_uv)))
        self.noise[n-1] = quality.noise_uv
        self.flags.resize((n, len(quality.flags)))
        self.flags[n-1] = quality.flags
        for dataset, values in (
                (self.saturated, quality.saturated_fraction),
                (self.flatline, quality.flatline_fraction),
                (self.zero, quality.zero_fraction)):
            dataset.resize((n, len(values)))
            dataset[n-1] = values

        # The waveform payload budget is charged against raw core bytes.
        # Unknown/ambiguous clean events receive half the slots; deterministic
        # time/channel hashing samples the rest without storing all snippets.
        self._ensure_waveforms(channel_map)
        row_bytes = 61 * self.cache_spec.width * 2 + 16
        raw_core_bytes = (core.core_stop-core.core_start) * len(quality.flags) * 2
        budget = min(len(candidates),
                     int(cache_fraction * raw_core_bytes) // row_bytes)
        if budget:
            clean = np.flatnonzero(candidates["qc_flags"] == 0)
            uncertain = clean[(candidates["unit_id"][clean] < 0) &
                              (candidates["snr"][clean] >= 5.)]
            priority = uncertain[np.argsort(-candidates["snr"][uncertain],
                                            kind="stable")[:budget//2]]
            remaining = np.setdiff1d(clean, priority, assume_unique=False)
            keys = (candidates["sample_index"][remaining].astype(np.uint64) *
                    np.uint64(11400714819323198485) ^
                    candidates["channel_index"][remaining].astype(np.uint64))
            keep = min(budget - len(priority), len(remaining))
            sampled = remaining[np.argpartition(keys, keep-1)[:keep]] if keep else np.empty(0, int)
            selected = np.sort(np.r_[priority, sampled])
            if len(selected):
                event_rows = np.empty(len(selected), EVENT_DTYPE)
                for name in EVENT_DTYPE.names:
                    event_rows[name] = candidates[name][selected]
                values, starts, stops = extract_waveforms(
                    event_rows, voltage_uv, core.data_start, self.cache_spec,
                    core.data_start, core.data_start + len(voltage_uv))
                encoded = encode_waveforms(values, self.cache_spec)
                _append(self.cache_rows, first + selected)
                old = len(self.cache_data)
                self.cache_data.resize((old + len(selected), 61, self.cache_spec.width))
                self.cache_data[old:] = encoded
                _append(self.cache_start, starts)
                _append(self.cache_stop, stops)
                self.cached_bytes += len(selected) * row_bytes
        self.handle.flush()

    def finish(self, models, *, telemetry=None, adaptive_shift=None):
        group = self.handle.create_group("model_after")
        group.create_dataset("waveforms", data=models.waveforms,
                             compression="lzf")
        group.create_dataset("channels", data=models.channels)
        group.create_dataset("anchors", data=models.anchors)
        group.create_dataset("assigned", data=models.assigned)
        group.attrs["version"] = models.version
        if adaptive_shift is not None:
            adaptive_shift.save(self.handle.create_group("adaptive_shift"))
        if telemetry is not None:
            self.handle.attrs["telemetry_json"] = json.dumps(
                telemetry, sort_keys=True)
        self.handle.attrs["complete"] = True
        self.handle.flush()
        self.handle.close()
        self.handle = None
        os.replace(self.partial, self.path)

    def abort(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
