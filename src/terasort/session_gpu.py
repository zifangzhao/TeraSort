"""Direct-CUDA bounded residual matcher for the experimental session path."""

from __future__ import annotations

from bisect import bisect_left, insort
from pathlib import Path

import numpy as np

from .cuda_environment import configure_cupy_cache
from .session_models import Match, normalize_template_identity_map


def _best_template_per_identity(event_ids, template_ids, gains, n_templates,
                                identity_map):
    positions = np.arange(len(gains), dtype=np.int64)
    if identity_map is None:
        return event_ids, gains, positions
    identities = normalize_template_identity_map(identity_map, n_templates)
    if len(np.unique(identities)) == n_templates:
        return event_ids, gains, positions
    _, template_groups = np.unique(identities, return_inverse=True)
    group_count = int(template_groups.max()) + 1
    keys = (event_ids.astype(np.int64) * group_count
            + template_groups[template_ids])
    _, inverse = np.unique(keys, return_inverse=True)
    group_gain = np.zeros(int(inverse.max()) + 1, np.float32)
    np.maximum.at(group_gain, inverse, gains)
    group_position = np.full(len(group_gain), len(gains), np.int64)
    np.minimum.at(group_position, inverse, np.where(
        (gains > 0) & (gains == group_gain[inverse]), positions, len(gains)))
    selected = group_position[group_position < len(gains)]
    return event_ids[selected], gains[selected], selected


class CudaResidualMatcher:
    """Keep local templates resident; score sparse pairs, subtract independent fits."""

    def __init__(self, models, *, pair_batch=65_536, max_pairs=2_000_000,
                 template_identity_map=None):
        configure_cupy_cache()
        import cupy as cp
        if models.waveforms.shape[2] > 32:
            raise ValueError("CUDA local template width must be <= 32")
        self.cp = cp
        self.models = models
        self.template_identity_map = normalize_template_identity_map(
            template_identity_map, len(models.waveforms))
        self.pair_batch = int(pair_batch)
        self.max_pairs = int(max_pairs)
        if self.pair_batch < 1 or self.max_pairs < self.pair_batch:
            raise ValueError("Invalid CUDA pair budget")
        code = Path(__file__).with_name("session_match.cu").read_text()
        module = cp.RawModule(code=code, options=("--std=c++11", "--fmad=false"))
        self.score_kernel = module.get_function("score_residual_pairs")
        self.subtract_kernel = module.get_function("subtract_fits")
        self.detect_kernel = module.get_function("detect_residual_peaks")
        self.detect_smooth3_kernel = module.get_function("detect_residual_peaks_smooth3_shared")
        self.detect_buffer = None
        self.detect_count = cp.zeros(1, cp.uint64)
        self.template_version = -1
        self.templates = None
        self.channels = None

    def _refresh(self):
        if self.template_version != self.models.version:
            self.templates = self.cp.asarray(self.models.waveforms)
            self.channels = self.cp.asarray(self.models.channels)
            self.template_version = self.models.version

    def _estimate_smooth3_noise(self, residual, noise_uv):
        cp = self.cp
        noise = np.asarray(noise_uv, np.float32)
        if (noise.shape != (residual.shape[1],)
                or np.any(np.isnan(noise) | (noise <= 0))):
            raise ValueError('Positive channel noise required; infinity masks contacts')
        nt = residual.shape[0]
        step = max(1, nt // 4000)
        sample_rows = cp.arange(0, nt, step, dtype=cp.int64)
        sample = residual[sample_rows].copy()
        interior = (sample_rows > 0) & (sample_rows < nt - 1)
        rows = sample_rows[interior]
        if rows.size:
            sample[interior] = .25 * (
                residual[rows - 1] + 2 * residual[rows] + residual[rows + 1])
        sample = cp.asnumpy(sample)
        center = np.median(sample, axis=0)
        estimate = np.median(np.abs(sample - center), axis=0) / .67448975
        return np.where(np.isfinite(noise),
                        np.maximum(estimate, .01), np.inf).astype(np.float32)

    def detect(self, residual, noise_uv, floor_snr, *, max_candidates=500_000,
               mode='raw', smooth3_noise_uv=None):
        cp = self.cp
        if mode not in ('raw','smooth3'):
            raise ValueError('Unknown detector mode')
        if (residual.dtype != cp.float32 or not residual.flags.c_contiguous
                or residual.ndim != 2 or not residual.size or max_candidates < 1
                or not np.isfinite(floor_snr) or floor_snr <= 0):
            raise ValueError('Invalid contiguous FP32 residual or detection budget')
        noise = np.asarray(noise_uv, np.float32)
        if noise.shape != (residual.shape[1],) or np.any(np.isnan(noise) | (noise <= 0)):
            raise ValueError('Positive channel noise required; infinity masks contacts')
        if mode == 'smooth3':
            if smooth3_noise_uv is None:
                noise = self._estimate_smooth3_noise(residual, noise)
            else:
                noise = np.asarray(smooth3_noise_uv, np.float32)
                if (noise.shape != (residual.shape[1],)
                        or np.any(np.isnan(noise) | (noise <= 0))):
                    raise ValueError('Invalid precomputed smooth3 noise')
        elif smooth3_noise_uv is not None:
            raise ValueError('Precomputed smooth3 noise requires smooth3 mode')
        nt, nc = residual.shape
        device_noise = cp.asarray(noise)
        capacity = min(residual.size, max_candidates)
        if self.detect_buffer is None or len(self.detect_buffer) != capacity:
            self.detect_buffer = cp.empty(capacity, cp.int64)
        self.detect_count.fill(0)
        if mode == 'smooth3':
            self.detect_smooth3_kernel(((nc + 31) // 32, (nt + 7) // 8), (32, 8),
                (residual, device_noise, np.int64(nt), np.int32(nc),
                 np.float32(floor_snr), self.detect_count, self.detect_buffer,
                 np.uint64(capacity)))
        else:
            self.detect_kernel(((residual.size+255)//256,), (256,),
                (residual, device_noise, np.int64(nt), np.int32(nc),
                 np.float32(floor_snr), self.detect_count, self.detect_buffer,
                 np.uint64(capacity)))
        count = int(self.detect_count.get()[0])
        if count > capacity:
            raise OverflowError("Artifact burst exceeds per-core candidate budget")
        flat = cp.sort(self.detect_buffer[:count])
        indices = cp.asnumpy(flat)
        t, c = np.divmod(indices, nc)
        if mode == 'smooth3':
            event_t = flat // nc
            event_c = flat % nc
            center_value = residual[event_t, event_c].copy()
            interior = (event_t > 0) & (event_t < nt - 1)
            rows = event_t[interior]
            channels = event_c[interior]
            center_value[interior] = .25 * (
                residual[rows - 1, channels] + 2 * residual[rows, channels]
                + residual[rows + 1, channels])
            snr = cp.asnumpy(cp.abs(center_value) / device_noise[event_c])
        else:
            snr = cp.asnumpy(
                cp.abs(residual.ravel()[flat]) / device_noise[flat % nc])
        return list(zip(t.astype(np.int32), c.astype(np.int32),
                        snr.astype(np.float32)))

    def _score(self, residual, events, *, half_width=8, noise_uv=None,
               shift_radius=2, score_floor=.65, amplitude_min=.3, amplitude_max=3.):
        cp = self.cp
        self._refresh()
        radius_values = np.asarray(shift_radius)
        if radius_values.ndim == 0:
            if (not np.issubdtype(radius_values.dtype, np.integer)
                    or not 0 <= int(radius_values) <= 8):
                raise ValueError('Timing search radius must be an integer from 0 to 8')
            device_radii = cp.full(len(self.models.waveforms), int(radius_values),
                                   dtype=cp.int32)
        else:
            if (radius_values.shape != (len(self.models.waveforms),)
                    or not np.issubdtype(radius_values.dtype, np.integer)
                    or np.any((radius_values < 0) | (radius_values > 8))):
                raise ValueError('Per-unit timing radii must match model count and be from 0 to 8')
            device_radii = cp.asarray(radius_values, dtype=cp.int32)
        pair_event, pair_unit = [], []
        proposed_event = np.zeros(len(events), dtype=bool)
        nt = len(residual)
        for i, (t, channel, _) in enumerate(events):
            if 30 <= t and t + 31 <= nt:
                units = self.models.by_channel.get(int(channel), ())
                if len(units) > 512:
                    raise OverflowError("Spatial model partition exceeds 512 units")
                if len(pair_event) + len(units) > self.max_pairs:
                    raise OverflowError("Too many event/template pairs in one source core")
                if len(units):
                    proposed_event[i] = True
                pair_event.extend([i] * len(units))
                pair_unit.extend(units)
        if len(pair_event) > self.max_pairs:
            raise OverflowError("Too many event/template pairs in one source core")
        self.last_score_diagnostics = {
            'events_with_template_proposals': int(np.count_nonzero(proposed_event)),
            'event_template_pairs': int(len(pair_event)),
        }
        scores = np.full(len(pair_event), -1., np.float32)
        amplitudes = np.zeros(len(pair_event), np.float32)
        gains = np.zeros(len(pair_event), np.float32)
        shifts = np.zeros(len(pair_event), np.int32)
        weights = cp.ones(residual.shape[1], cp.float32) if noise_uv is None else cp.asarray(
            1. / np.maximum(np.asarray(noise_uv, np.float32), .01) ** 2)
        events_t = cp.asarray([event[0] for event in events], dtype=cp.int32)
        width = self.models.waveforms.shape[2]
        for start in range(0, len(pair_event), self.pair_batch):
            stop = min(start + self.pair_batch, len(pair_event))
            pe = cp.asarray(pair_event[start:stop], dtype=cp.int32)
            pu = cp.asarray(pair_unit[start:stop], dtype=cp.int32)
            n = stop - start
            ds = cp.empty(n, cp.float32)
            da = cp.empty(n, cp.float32)
            dg = cp.empty(n, cp.float32)
            dt = cp.empty(n, cp.int32)
            self.score_kernel(((n + 3) // 4,), (128,),
                (residual, events_t, self.templates, self.channels,
                 pe, pu, weights, np.int32(n), np.int32(nt),
                 np.int32(residual.shape[1]), np.int32(width),
                 np.int32(half_width), device_radii,
                 np.float32(score_floor), np.float32(amplitude_min),
                 np.float32(amplitude_max), ds, da, dg, dt))
            scores[start:stop] = cp.asnumpy(ds)
            amplitudes[start:stop] = cp.asnumpy(da)
            gains[start:stop] = cp.asnumpy(dg)
            shifts[start:stop] = cp.asnumpy(dt)
        if not len(pair_event):
            return [None] * len(events)
        # Segmented reductions replace a Python loop and sort for every pair.
        event_ids = np.asarray(pair_event, np.int32)
        units = np.asarray(pair_unit, np.int32)
        rank_events, rank_gains, rank_positions = _best_template_per_identity(
            event_ids, units, gains, len(self.models.waveforms),
            self.template_identity_map)
        best_gain = np.zeros(len(events), np.float32)
        np.maximum.at(best_gain, rank_events, rank_gains)
        best = np.full(len(events), len(gains), np.int64)
        np.minimum.at(best, rank_events, np.where(
            (rank_gains > 0) & (rank_gains == best_gain[rank_events]),
            rank_positions, len(gains)))
        runner_gain = np.zeros(len(events), np.float32)
        np.maximum.at(runner_gain, rank_events,
                      np.where(rank_positions != best[rank_events],
                               rank_gains, 0))
        runner = np.full(len(events), len(gains), np.int64)
        np.minimum.at(runner, rank_events, np.where(
            (rank_positions != best[rank_events]) & (rank_gains > 0) &
            (rank_gains == runner_gain[rank_events]), rank_positions, len(gains)))
        choices = []
        for i in range(len(events)):
            p = best[i]
            if p == len(gains):
                choices.append(None)
                continue
            second = float(scores[runner[i]]) if runner[i] != len(gains) else -1.
            choices.append((float(scores[p]), float(gains[p]), int(units[p]),
                            float(amplitudes[p]), second, int(shifts[p]),
                            float(1 - runner_gain[i] / gains[p])))
        return choices

    def match(self, voltage_uv, noise_uv, floor_snr, *,
              core_start, core_stop, score_floor=.65, min_margin=.03,
              amplitude_min=.3, amplitude_max=3., max_passes=3,
              max_candidates=500_000, shift_radius=2, refractory_samples=2,
              trace=None, overlap_policy='strict', rescue_floor_snr=None, rescue_passes=1,
              half_width=8, detector_mode='raw', overlap_window_samples=61):
        """Return all first-pass candidates and fitted spikes in source-core offsets."""
        cp = self.cp
        if not isinstance(half_width, int) or not 3 <= half_width <= 30:
            raise ValueError('Scoring half width must be 3–30 samples')
        radius_values = np.asarray(shift_radius)
        valid_radius = (
            np.issubdtype(radius_values.dtype, np.integer) and
            ((radius_values.ndim == 0 and 0 <= int(radius_values) <= 8) or
             (radius_values.shape == (len(self.models.waveforms),) and
              np.all((radius_values >= 0) & (radius_values <= 8))))
        )
        if (not valid_radius or not isinstance(rescue_passes, int)
                or not 1 <= rescue_passes <= 4):
            raise ValueError('Timing radius must be 0–8 and rescue passes 1–4')
        if overlap_policy not in ('strict', 'interference'):
            raise ValueError('Unknown overlap policy')
        if (isinstance(overlap_window_samples, (bool, np.bool_))
                or not isinstance(overlap_window_samples, (int, np.integer))
                or not 0 <= int(overlap_window_samples) <= 61):
            raise ValueError('Strict overlap window must be an integer from 0 to 61 samples')
        overlap_window_samples = int(overlap_window_samples)
        if overlap_window_samples != 61 and overlap_policy != 'strict':
            raise ValueError('A shortened overlap window requires strict mode')
        if rescue_floor_snr is not None and (
                not np.isfinite(rescue_floor_snr) or not 0 < rescue_floor_snr <= floor_snr):
            raise ValueError('Rescue threshold must be positive and at most the primary threshold')
        scheduler = None
        if overlap_policy == 'interference':
            from .session_interference import InterferenceScheduler
            scheduler = InterferenceScheduler(self.models, noise_uv)
        residual = cp.asarray(voltage_uv, dtype=cp.float32, order='C')
        smooth3_noise = (self._estimate_smooth3_noise(residual, noise_uv)
                         if detector_mode == 'smooth3' else None)
        original = self.detect(
            residual, noise_uv, floor_snr, max_candidates=max_candidates,
            mode=detector_mode, smooth3_noise_uv=smooth3_noise)
        current = original
        all_events = []
        accepted = []
        occupied = {unit: [] for unit in range(len(self.models.waveforms))}
        total_passes = max_passes + (rescue_passes if rescue_floor_snr is not None else 0)
        strong_done = False
        diagnostics = {
            'candidate_events': 0, 'events_with_template_proposals': 0,
            'event_template_pairs': 0, 'events_with_fit': 0,
            'events_without_eligible_fit': 0, 'gain_rejections': 0,
            'margin_rejections': 0, 'amplitude_rejections': 0,
            'same_unit_refractory_rejections': 0,
            'contact_overlap_rejections': 0,
            'interference_scheduler_rejections': 0,
            'accepted_fits': 0, 'emitted_matches': 0, 'passes': [],
        }
        for pass_index in range(total_passes):
            rescue = pass_index >= max_passes
            if strong_done and not rescue:
                continue
            if pass_index:
                current = self.detect(
                    residual, noise_uv,
                    rescue_floor_snr if rescue else floor_snr,
                    max_candidates=max_candidates, mode=detector_mode,
                    smooth3_noise_uv=smooth3_noise)
            if not current:
                if rescue_floor_snr is not None and not rescue:
                    strong_done = True
                    continue
                break
            all_events.extend(current)
            pass_diagnostics = {
                'pass_index': int(pass_index),
                'rescue': bool(rescue),
                'candidate_events': int(len(current)),
                'events_with_template_proposals': 0, 'event_template_pairs': 0,
                'events_with_fit': 0, 'events_without_eligible_fit': 0,
                'gain_rejections': 0, 'margin_rejections': 0,
                'amplitude_rejections': 0,
                'same_unit_refractory_rejections': 0,
                'contact_overlap_rejections': 0,
                'interference_scheduler_rejections': 0,
                'accepted_fits': 0, 'emitted_matches': 0,
            }
            diagnostics['candidate_events'] += len(current)
            diagnostics['passes'].append(pass_diagnostics)
            choices = self._score(residual, current, noise_uv=noise_uv,
                                  half_width=half_width,
                                  shift_radius=shift_radius,
                                  score_floor=max(score_floor, .85) if pass_index >= 3 or rescue else score_floor,
                                  amplitude_min=amplitude_min, amplitude_max=amplitude_max)
            if trace is not None:
                trace('scores', pass_index=pass_index, residual=residual,
                      events=current, choices=choices)
            proposals = []
            score_diagnostics = self.last_score_diagnostics
            diagnostics['events_with_template_proposals'] += score_diagnostics['events_with_template_proposals']
            diagnostics['event_template_pairs'] += score_diagnostics['event_template_pairs']
            pass_diagnostics.update(score_diagnostics)
            diagnostics['events_with_fit'] += sum(choice is not None for choice in choices)
            diagnostics['events_without_eligible_fit'] += sum(choice is None for choice in choices)
            pass_diagnostics['events_with_fit'] = sum(choice is not None for choice in choices)
            pass_diagnostics['events_without_eligible_fit'] = sum(choice is None for choice in choices)
            required_margin = max(min_margin, .1) if rescue else min_margin
            for i, choice in enumerate(choices):
                if choice is None:
                    continue
                score, gain, unit, amplitude, runner, shift, margin = choice
                # Rescue changes proposals only; retain the primary gain floor.
                if gain < floor_snr ** 2:
                    diagnostics['gain_rejections'] += 1
                    pass_diagnostics['gain_rejections'] += 1
                    continue
                if margin < required_margin:
                    diagnostics['margin_rejections'] += 1
                    pass_diagnostics['margin_rejections'] += 1
                    continue
                if not amplitude_min <= amplitude <= amplitude_max:
                    diagnostics['amplitude_rejections'] += 1
                    pass_diagnostics['amplitude_rejections'] += 1
                    continue
                t, channel, _ = current[i]
                proposals.append((gain, int(t) + shift, int(channel), int(unit),
                                  score, runner, amplitude, int(t), margin))
            proposals.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
            accepted_in_pass = []
            subtraction_colors = []
            if scheduler is not None:
                scheduler.reset()
            occupied_contacts = {}
            # In shortened-window mode this stores time/color pairs for
            # deterministic overlap coloring, reusing the conflict index.
            trace_owners = {} if trace is not None else None
            for gain, t, channel, unit, score, runner, amplitude, original_t, margin in proposals:
                prior = occupied[unit]
                pos = bisect_left(prior, t)
                if ((pos and t - prior[pos-1] <= refractory_samples) or
                        (pos < len(prior) and prior[pos] - t <= refractory_samples)):
                    if trace is not None:
                        trace('refractory', pass_index=pass_index, candidate=original_t,
                              channel=channel, unit=unit, fitted_time=t,
                              blocker_time=prior[pos-1] if pos and t-prior[pos-1] <= refractory_samples else prior[pos],
                              blocker_unit=unit)
                    diagnostics['same_unit_refractory_rejections'] += 1
                    pass_diagnostics['same_unit_refractory_rejections'] += 1
                    continue
                contacts = self.models.channels[unit]
                conflict = False
                color = 0
                if scheduler is not None:
                    color, blocker = scheduler.admit(t, unit, amplitude, score=score, margin=margin)
                    conflict = blocker is not None
                    if conflict:
                        blocker_time = blocker['blocker_time']
                        blocker_unit = blocker['blocker_unit']
                        blocker_contact = blocker['blocker_contact']
                for contact in (contacts[contacts >= 0] if scheduler is None else ()):
                    contact = int(contact)
                    rows = occupied_contacts.get(contact, ())
                    if overlap_window_samples < 61:
                        j = bisect_left(rows, (t, -1))
                        previous_time = rows[j-1][0] if j else None
                        next_time = rows[j][0] if j < len(rows) else None
                    else:
                        j = bisect_left(rows, t)
                        previous_time = rows[j-1] if j else None
                        next_time = rows[j] if j < len(rows) else None
                    previous_conflict = (previous_time is not None and
                                         t - previous_time < overlap_window_samples)
                    next_conflict = (next_time is not None and
                                     next_time - t < overlap_window_samples)
                    if previous_conflict or next_conflict:
                        conflict = True
                        if trace is not None:
                            blocker_time = previous_time if previous_conflict else next_time
                            blocker_unit = trace_owners[(contact, blocker_time)]
                            blocker_contact = contact
                        break
                if conflict:
                    if scheduler is None:
                        diagnostics['contact_overlap_rejections'] += 1
                        pass_diagnostics['contact_overlap_rejections'] += 1
                    else:
                        diagnostics['interference_scheduler_rejections'] += 1
                        pass_diagnostics['interference_scheduler_rejections'] += 1
                    if trace is not None:
                        trace('overlap_deferred', pass_index=pass_index, candidate=original_t,
                              channel=channel, unit=unit, fitted_time=t,
                              blocker_time=blocker_time, blocker_unit=blocker_unit,
                              blocker_contact=blocker_contact)
                    continue
                if scheduler is None and overlap_window_samples < 61:
                    # Color overlapping template writes without a second
                    # per-contact index of all accepted events.
                    used_colors = set()
                    for contact in contacts[contacts >= 0]:
                        rows = occupied_contacts.get(int(contact), ())
                        lo = bisect_left(rows, (t - 60, -1))
                        hi = bisect_left(rows, (t + 61, -1))
                        for position in range(lo, hi):
                            used_colors.add(rows[position][1])
                    for color in range(32):
                        if color not in used_colors:
                            break
                    else:
                        raise OverflowError(
                            'Overlapping subtraction needs more than 32 colors; '
                            'increase the overlap window or shorten the core')
                insort(prior, t)
                for contact in (contacts[contacts >= 0] if scheduler is None else ()):
                    contact = int(contact)
                    rows = occupied_contacts.setdefault(contact, [])
                    if overlap_window_samples < 61:
                        insort(rows, (t, color))
                    else:
                        insort(rows, t)
                    if trace is not None:
                        trace_owners[(contact, t)] = unit
                accepted_in_pass.append((t, unit, amplitude))
                diagnostics['accepted_fits'] += 1
                pass_diagnostics['accepted_fits'] += 1
                subtraction_colors.append(color)
                if trace is not None:
                    trace('accepted', pass_index=pass_index, candidate=original_t,
                          channel=channel, unit=unit, fitted_time=t)
                if core_start <= t < core_stop:
                    accepted.append(Match(t, channel, unit, score, runner,
                                          amplitude, gain, pass_index, original_t, margin))
                    diagnostics['emitted_matches'] += 1
                    pass_diagnostics['emitted_matches'] += 1
            if not accepted_in_pass:
                if rescue_floor_snr is not None and not rescue:
                    strong_done = True
                    continue
                break
            if pass_index + 1 == total_passes:
                break
            # Geometrically disjoint color groups avoid nondeterministic
            # atomic addition order when weakly interacting fits overlap.
            for color in range(max(subtraction_colors)+1):
                batch = [row for row,c in zip(accepted_in_pass, subtraction_colors) if c == color]
                times = cp.asarray([row[0] for row in batch], cp.int32)
                units = cp.asarray([row[1] for row in batch], cp.int32)
                amps = cp.asarray([row[2] for row in batch], cp.float32)
                self.subtract_kernel((len(batch),), (256,),
                    (residual, self.templates, self.channels, times, units, amps,
                     np.int32(len(batch)), np.int32(len(residual)),
                     np.int32(residual.shape[1]),
                     np.int32(self.models.waveforms.shape[2])))
        accepted.sort(key=lambda item: (item.source_sample, item.channel,
                                         item.pass_index))
        self.last_diagnostics = diagnostics
        return all_events, accepted
