"""Offline bounded session sorter. Opt-in until ground-truth quality gates pass."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from bisect import bisect_left

import h5py
import numpy as np
import psutil

from .candidates.detectors import numpy_detect
from .candidates.waveforms import geometry_channel_map
from .session_calibration import calibrate_day
from .session_links import build_cross_day_links
from .session_manifest import load_session
from .session_models import LocalModels, match_residual_cpu
from .session_adaptation import TemplateEvidence
from .session_novelty import NoveltyBank
from .session_signal import (assess_quality, iter_cores, prefetch_cores,
                             preprocess)
from .session_store import RunJournal, ShardWriter


def _detect_cpu(voltage, noise, floor, *, max_candidates):
    q = np.abs(voltage) / np.asarray(noise, np.float32)[None, :]
    indices = numpy_detect(q, floor=floor)
    if len(indices) > max_candidates:
        raise OverflowError("Artifact burst exceeds per-core candidate budget")
    t, c = np.divmod(indices, voltage.shape[1])
    return list(zip(t.astype(np.int32), c.astype(np.int32),
                    q[t, c].astype(np.float32)))


def _day_models(probe, day_id, root, *, resume):
    calibration = root / probe.probe_id / "calibration"
    path = calibration / f"{day_id}.npz"
    metadata = calibration / f"{day_id}.json"
    partial = path.with_suffix(".npz.partial")
    if path.is_file():
        return LocalModels.load(path, probe)
    if partial.exists():
        if not resume:
            raise FileExistsError(partial)
        partial.unlink()
    seed = probe.seed_for_day(day_id)
    if seed is not None:
        models = LocalModels.load(seed, probe)
        aligned = models.canonicalize()
        description = {"method": "aligned_external_seed",
                       "seed_path": str(seed), "seed_units": len(models.waveforms),
                       "canonicalized_units": aligned}
    else:
        models, description = calibrate_day(probe, day_id)
    calibration.mkdir(parents=True, exist_ok=True)
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    with partial.open("xb") as handle:
        np.savez_compressed(handle, waveforms=models.waveforms,
                            channels=models.channels, anchors=models.anchors,
                            assigned=models.assigned, version=models.version)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    metadata.write_text(json.dumps(description, indent=2) + "\n",
                        encoding="utf-8")
    return models


def _model_from_completed(path):
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"Incomplete published shard: {path}")
        group = handle["model_after"]
        return LocalModels(group["waveforms"][:], group["channels"][:],
                           group["anchors"][:], group["assigned"][:],
                           int(group.attrs["version"]))


def _shard_intervals(probe, *, shard_seconds, start_sample, stop_sample):
    step = max(1, round(probe.sample_rate_hz * shard_seconds))
    for segment in probe.segments:
        first = max(segment.start_sample, start_sample)
        last = min(segment.stop_sample, stop_sample)
        for start in range(first, last, step):
            yield segment, start, min(start + step, last)


def _collect_updates(models, matches, voltage, quality, evidence,
                     *, data_start, core_start):
    if quality.interval_bad:
        return
    event_times = [match.source_sample for match in matches]
    contact_sets = [set(map(int, contacts[contacts >= 0]))
                    for contacts in models.channels]
    for i, match in enumerate(matches):
        if (match.pass_index != 0 or match.score < .9 or
                match.relative_margin < .1 or
                not .5 <= match.amplitude <= 2.):
            continue
        t = match.source_sample
        if t < 30 or t + 31 > len(voltage):
            continue
        left = bisect_left(event_times, t - 60)
        right = bisect_left(event_times, t + 61)
        if any(contact_sets[match.unit] & contact_sets[matches[j].unit]
               for j in range(left, right) if j != i):
            continue
        channels = models.channels[match.unit]
        snippet = np.zeros_like(models.waveforms[match.unit])
        valid = channels >= 0
        if not np.all(quality.usable_channels[channels[valid]]):
            continue
        snippet[:, valid] = (voltage[t-30:t+31, channels[valid]] /
                             np.float32(match.amplitude))
        if np.isfinite(snippet).all():
            evidence.observe(match.unit, data_start + t, core_start, snippet)


def run_session(manifest, output_root, *, backend="cuda", resume=False,
                core_seconds=2., shard_seconds=300., halo_ms=100.,
                floor_snr=4.5, score_floor=.65, min_margin=.03,
                cache_fraction=.05, max_candidates=500_000,
                start_sample=0, stop_sample=None, vram_limit_gb=12.,
                progress=None, freeze_templates=False, novelty="off", residual_passes=3,
                overlap_policy='strict', rescue_floor_snr=None, rescue_passes=1,
                shift_radius=2, half_width=8, refit_rounds=0, detector_mode='raw',
                read_buffer_mb=0, prefetch_depth=2):
    """Process one session into immutable, per-probe time shards.

    The existing Kilosort-compatible sorter remains the quality baseline.
    This route is opt-in and deliberately records every unknown detection.
    """
    session = load_session(manifest)
    if (isinstance(read_buffer_mb, bool) or not isinstance(read_buffer_mb, int)
            or not 0 <= read_buffer_mb <= 1024 or isinstance(prefetch_depth, bool)
            or not isinstance(prefetch_depth, int) or not 1 <= prefetch_depth <= 64):
        raise ValueError('Read buffer must be 0–1024 MiB; prefetch depth must be 1–64 cores')
    if detector_mode not in ('raw','smooth3') or (detector_mode!='raw' and backend!='cuda'):
        raise ValueError('smooth3 detection currently requires CUDA')
    if not isinstance(refit_rounds, int) or not 0 <= refit_rounds <= 2:
        raise ValueError('Refit rounds must be 0–2')
    if not isinstance(half_width, int) or not 3 <= half_width <= 30:
        raise ValueError('Scoring half width must be 3–30 samples')
    def emit(stage, **details):
        if progress is not None:
            progress({"stage": stage, **details})
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be cpu or cuda")
    if (not isinstance(shift_radius, int) or not 0 <= shift_radius <= 8
            or not isinstance(rescue_passes, int) or not 1 <= rescue_passes <= 4):
        raise ValueError('Timing radius must be 0–8 and rescue passes 1–4')
    if rescue_floor_snr is not None and (backend != 'cuda' or
            not np.isfinite(rescue_floor_snr) or not 0 < rescue_floor_snr <= floor_snr):
        raise ValueError('CUDA rescue threshold must be positive and at most primary floor_snr')
    if overlap_policy not in ('strict', 'interference') or (backend == 'cpu' and overlap_policy != 'strict'):
        raise ValueError('Interference scheduling requires CUDA; CPU already fits sequentially')
    if not isinstance(residual_passes, int) or not 1 <= residual_passes <= 12:
        raise ValueError('Residual passes must be an integer from 1 to 12')
    if novelty not in ('off', 'shadow', 'enroll') or (novelty == 'enroll' and freeze_templates):
        raise ValueError('Invalid novelty mode or enrollment with frozen templates')
    if (not 0 < core_seconds <= shard_seconds or halo_ms < 0 or
            floor_snr <= 0 or not 0 < score_floor <= 1 or
            not 0 <= min_margin <= 1 or not 0 <= cache_fraction <= 1 or
            max_candidates < 1 or start_sample < 0 or
            (stop_sample is not None and stop_sample <= start_sample)):
        raise ValueError("Invalid bounded session configuration")
    config = {
        "backend": backend, "core_seconds": core_seconds,
        "shard_seconds": shard_seconds, "halo_ms": halo_ms,
        "floor_snr": floor_snr, "score_floor": score_floor,
        "min_margin": min_margin, "cache_fraction": cache_fraction,
        "max_candidates": max_candidates, "start_sample": start_sample,
        "stop_sample": stop_sample, "vram_limit_gb": vram_limit_gb,
        "preprocessing": "per-shank median; zero-phase 300-6000 Hz Butterworth v1",
        "matcher": f"center{2*half_width+1} proposal; noise-weighted full61 fitting v2; experimental",
        "adaptation": "bounded_recent_evidence_temporal_holdout_v1",
        "freeze_templates": freeze_templates,
        "residual_passes": residual_passes,
        "overlap_policy": overlap_policy,
        "rescue_floor_snr": rescue_floor_snr,
        "rescue_passes": rescue_passes,
        "shift_radius": shift_radius,
        "half_width": half_width,
        "detector_mode": detector_mode,
        "candidate_snr_frame": 'raw_noise' if detector_mode=='raw' else 'smooth3_per_pass_MAD_v1',
        "refit_rounds": refit_rounds,
        "refit_algorithm": "bounded_neighbor_subtraction_64_v1",
        "rescue_algorithm": "bounded_final_passes_score085_margin010_primary_gain_v2",
        "overlap_scheduler": ("candidate_confidence_bidirectional_1pct_colors32_v1"
                              if overlap_policy == "interference" else "strict_v1"),
        "late_residual_score_floor": .85,
        "novelty": novelty,
        "novelty_algorithm": "bounded_temporal_proposals_v1",
    }
    # Preserve existing default-run checkpoint digests. Nondefault IO settings
    # are immutable for a run, just like the scientific configuration.
    if read_buffer_mb or prefetch_depth != 2:
        config.update(read_buffer_mb=read_buffer_mb, prefetch_depth=prefetch_depth)
    source_record = session.as_source_record()
    journal = RunJournal(output_root, source_record, config, resume=resume)
    if backend == "cuda":
        import cupy as cp
        if vram_limit_gb <= 0:
            raise ValueError("Positive GPU pool limit required")
        cp.get_default_memory_pool().set_limit(
            size=int(vram_limit_gb * 1024 ** 3))
        from .session_gpu import CudaResidualMatcher
    started = time.perf_counter()
    process = psutil.Process()
    totals = {"shards": 0, "candidate_rows": 0, "spikes": 0,
              "template_promotions": 0, "peak_evidence_rows": 0,
              "enrolled_units": 0,
              "source_bytes_read": 0, "cached_waveform_bytes": 0,
              "raw_bytes_covered": 0,
              "derived_bytes": 0, "peak_process_rss_bytes": 0,
              "peak_gpu_pool_bytes": 0,
              "peak_total_vram_used_bytes": 0}
    for probe in session.probes:
        day_id = None
        models = None
        matcher = None
        neighbor_map = geometry_channel_map(
            probe.geometry, probe.shank, 75., 16)
        halo = max(31, int(round(probe.sample_rate_hz * halo_ms / 1000.)))
        final = probe.stop_sample if stop_sample is None else stop_sample
        for segment, start, stop in _shard_intervals(
                probe, shard_seconds=shard_seconds,
                start_sample=start_sample, stop_sample=final):
            if segment.day_id != day_id:
                day_id = segment.day_id
                emit("calibration_start", probe_id=probe.probe_id, day_id=day_id)
                models = _day_models(probe, day_id, journal.root, resume=resume)
                evidence = TemplateEvidence(models, probe.sample_rate_hz)
                novel = NoveltyBank(probe.sample_rate_hz) if novelty != 'off' else None
                if novel is not None and models.waveforms.shape[2] != neighbor_map.shape[1]:
                    raise ValueError('Novelty requires seeds with the session local-channel width')
                emit("calibration_complete", probe_id=probe.probe_id,
                     day_id=day_id, templates=len(models.waveforms))
                matcher = (CudaResidualMatcher(models) if backend == "cuda"
                           else None)
            output = journal.shard_path(probe.probe_id, start, stop)
            if journal.completed(output):
                if not resume:
                    raise FileExistsError(output)
                models = _model_from_completed(output)
                evidence = TemplateEvidence(models, probe.sample_rate_hz)
                with h5py.File(output, 'r') as handle:
                    evidence.restore(handle)
                    if novel is not None:
                        novel.restore(handle)
                if matcher is not None:
                    matcher.models = models
                continue
            partial = output.with_suffix(".h5.partial")
            if partial.exists():
                if not resume:
                    raise FileExistsError(partial)
                partial.unlink()
            writer = ShardWriter(
                output, probe=probe, day_id=segment.day_id,
                start_sample=start, stop_sample=stop,
                config_sha256=journal.record["config_sha256"])
            shard_started = time.perf_counter()
            shard_bytes = 0
            shard_raw_bytes = 0
            shard_peak_rss = 0
            shard_peak_gpu_pool = 0
            shard_peak_total_vram = 0
            try:
                cores = iter_cores(
                    probe, core_seconds=core_seconds, halo_samples=halo,
                    start_sample=start, stop_sample=stop, read_buffer_mb=read_buffer_mb)
                for core in prefetch_cores(cores, depth=prefetch_depth):
                    voltage = preprocess(core, probe)
                    quality = assess_quality(core, voltage, probe)
                    signal = voltage.copy()
                    signal[:, ~quality.usable_channels] = 0
                    noise = np.where(quality.usable_channels,
                                     quality.noise_uv, np.inf).astype(np.float32)
                    local_start = core.core_start - core.data_start
                    local_stop = core.core_stop - core.data_start
                    if matcher is not None:
                        events, matches = matcher.match(
                            signal, noise, floor_snr, core_start=local_start,
                            core_stop=local_stop, score_floor=score_floor,
                            min_margin=min_margin,
                            max_passes=residual_passes,
                            overlap_policy=overlap_policy,
                            rescue_floor_snr=rescue_floor_snr,
                            rescue_passes=rescue_passes, shift_radius=shift_radius,
                            half_width=half_width,
                            detector_mode=detector_mode,
                            refractory_samples=max(2, round(probe.sample_rate_hz*.0005)),
                            max_candidates=max_candidates)
                    else:
                        events = _detect_cpu(signal, noise, floor_snr,
                                             max_candidates=max_candidates)
                        all_events = []
                        def redetect(residual, noise_uv, floor):
                            return _detect_cpu(residual, noise_uv, floor,
                                               max_candidates=max_candidates)
                        matches, _ = match_residual_cpu(
                            signal, events, models, score_floor=score_floor,
                            min_margin=min_margin, detector=redetect,
                            max_passes=residual_passes,
                            shift_radius=shift_radius,
                            half_width=half_width,
                            noise_uv=noise, floor_snr=floor_snr,
                            core_start=local_start, core_stop=local_stop,
                            refractory_samples=max(2, round(probe.sample_rate_hz*.0005)),
                            all_events=all_events)
                        events = all_events
                    if refit_rounds:
                        from .session_local_refit import local_refit
                        matches = local_refit(signal, noise, models, matches,
                            core_start=local_start, core_stop=local_stop, rounds=refit_rounds,
                            half_width=half_width, shift_radius=shift_radius,
                            score_floor=score_floor, min_margin=min_margin,
                            gain_floor=floor_snr**2,
                            refractory_samples=max(2,round(probe.sample_rate_hz*.0005)))
                    writer.append_core(
                        core=core, voltage_uv=signal, quality=quality,
                        threshold_events=events, matches=matches,
                        channel_map=neighbor_map,
                        cache_fraction=cache_fraction,
                        model_version=models.version)
                    if not freeze_templates:
                        _collect_updates(models, matches, signal, quality,
                                         evidence, data_start=core.data_start,
                                         core_start=core.core_start)
                    evidence.expire(core.core_stop)
                    if novel is not None:
                        novel.collect(models, events, matches, signal, quality, neighbor_map, core)
                    for match in matches:
                        models.assigned[match.unit] += 1
                    totals["source_bytes_read"] += core.source_bytes_read
                    emit("core_complete", probe_id=probe.probe_id,
                         stop_sample=core.core_stop, spikes=len(matches),
                         detections=len(events))
                    shard_bytes += core.source_bytes_read
                    raw_bytes = ((core.core_stop - core.core_start) *
                                 probe.n_channels * 2)
                    totals["raw_bytes_covered"] += raw_bytes
                    shard_raw_bytes += raw_bytes
                    shard_peak_rss = max(shard_peak_rss,
                                         process.memory_info().rss)
                    if backend == "cuda":
                        pool = cp.get_default_memory_pool()
                        shard_peak_gpu_pool = max(
                            shard_peak_gpu_pool, pool.total_bytes())
                        free, total = cp.cuda.runtime.memGetInfo()
                        shard_peak_total_vram = max(
                            shard_peak_total_vram, total-free)
                update_audit = evidence.promote(models, stop)
                if novel is not None:
                    novel_audit = novel.evaluate(models, stop, enroll=novelty == 'enroll')
                    evidence.extend_models(models)
                    novel.save(writer.handle, novel_audit)
                    enrolled = sum(row['enrolled_unit'] is not None for row in novel_audit)
                    totals['enrolled_units'] += enrolled
                    emit('novelty_complete', probe_id=probe.probe_id, stop_sample=stop,
                         proposals=len(novel_audit), enrolled=enrolled)
                evidence.save(writer.handle, update_audit)
                promotions = sum(row['promoted'] for row in update_audit)
                evidence_rows = sum(len(rows) for rows in evidence.rows.values())
                totals['template_promotions'] += promotions
                totals['peak_evidence_rows'] = max(totals['peak_evidence_rows'], evidence_rows)
                emit("adaptation_complete", probe_id=probe.probe_id,
                     stop_sample=stop, proposed=len(update_audit),
                     promoted=promotions, evidence_rows=evidence_rows)
                candidate_count = len(writer.candidates)
                spike_count = len(writer.spikes)
                writer.finish(models, telemetry={
                    "wall_seconds": time.perf_counter() - shard_started,
                    "source_bytes_read": shard_bytes,
                    "raw_bytes_covered": shard_raw_bytes,
                    "peak_process_rss_bytes": shard_peak_rss,
                    "peak_gpu_pool_bytes": shard_peak_gpu_pool,
                    "peak_total_vram_used_bytes": shard_peak_total_vram,
                })
            except BaseException:
                writer.abort()
                raise
            totals["shards"] += 1
            totals["candidate_rows"] += candidate_count
            totals["spikes"] += spike_count
            totals["cached_waveform_bytes"] += writer.cached_bytes
            totals["derived_bytes"] += output.stat().st_size
            totals["peak_process_rss_bytes"] = max(
                totals["peak_process_rss_bytes"], shard_peak_rss)
            totals["peak_gpu_pool_bytes"] = max(
                totals["peak_gpu_pool_bytes"], shard_peak_gpu_pool)
            totals["peak_total_vram_used_bytes"] = max(
                totals["peak_total_vram_used_bytes"], shard_peak_total_vram)
    totals["unresolved_cross_day_edges"] = build_cross_day_links(
        session, journal.root)
    totals["wall_seconds"] = time.perf_counter() - started
    totals["derived_bytes_per_raw_byte"] = (
        totals["derived_bytes"] / totals["raw_bytes_covered"]
        if totals["raw_bytes_covered"] else None)
    totals["output_root"] = str(journal.root)
    totals["experimental_quality_gate_passed"] = False
    return totals
