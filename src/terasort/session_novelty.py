"""Experimental bounded unknown-event proposals, isolated from dense sorting."""
from bisect import bisect_left
from collections import deque
import json

import numpy as np


def similarity(a, ac, b, bc):
    """Cosine on physical contacts, penalizing energy outside the intersection."""
    av, bv = ac >= 0, bc >= 0
    _, ai, bi = np.intersect1d(ac[av], bc[bv], return_indices=True)
    aa, bb = a[:, av], b[:, bv]
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-9:
        return 0.
    return max(float(np.sum(aa[2:59, ai] * bb[2+s:59+s, bi])) / denom
               for s in range(-2, 3))


class NoveltyBank:
    """At most 128 proposals, 64 waveforms each, 256 observations per core.

    Enrollment is experimental. Waveform consistency does not prove identity.
    Proposals use fixed reference shapes so held-out samples cannot train them.
    """
    def __init__(self, sample_rate, *, max_proposals=128):
        self.rate = float(sample_rate)
        self.max_proposals = max_proposals
        self.next_id = 0
        self.proposals = {}

    def expire(self, now):
        for key in list(self.proposals):
            proposal = self.proposals[key]
            if now - proposal['rows'][-1][0] > self.rate * 1800:
                del self.proposals[key]
            else:
                rows = proposal['rows']
                while rows and rows[0][0] < now - self.rate * 1800:
                    rows.popleft()

    def observe(self, sample, core_start, channels, waveform):
        if not np.isfinite(waveform).all() or np.linalg.norm(waveform) < 1e-6:
            return
        anchor = int(channels[0])
        nearby = [(similarity(waveform, channels, p['reference'], p['channels']), key)
                  for key, p in self.proposals.items() if p['channels'][0] == anchor]
        best = max(nearby, default=(0., -1))
        if best[0] >= .9:
            rows = self.proposals[best[1]]['rows']
            if sample > rows[-1][0] + max(2, round(.0005*self.rate)):
                rows.append((int(sample), int(core_start), waveform.copy()))
        elif len(self.proposals) < self.max_proposals and len(nearby) < 4:
            self.proposals[self.next_id] = dict(reference=waveform.copy(),
                channels=channels.copy(), rows=deque(
                    [(int(sample), int(core_start), waveform.copy())], maxlen=64))
            self.next_id += 1

    def collect(self, models, events, matches, signal, quality, channel_map, core):
        self.expire(core.core_stop)
        if quality.interval_bad:
            return
        # Reject candidate neighborhoods containing any assigned waveform.
        occupied = {}
        for match in matches:
            for c in models.channels[match.unit]:
                if c >= 0:
                    occupied.setdefault(int(c), []).append(match.source_sample)
        candidates = sorted(set((int(t), int(c)) for t, c, snr in events
            if snr >= 6 and core.core_start <= core.data_start+t < core.core_stop))
        # Spend the observation budget on unknowns, not detections already
        # explained by accepted units. Filter metadata before copying snippets.
        eligible = []
        if candidates:
            candidate_array = np.asarray(candidates, np.int64)
            for c in np.unique(candidate_array[:, 1]):
                channels = channel_map[c]
                valid = channels >= 0
                if not np.all(quality.usable_channels[channels[valid]]):
                    continue
                times = candidate_array[candidate_array[:, 1] == c, 0]
                busy = np.array(sorted({t for contact in channels[valid]
                                        for t in occupied.get(int(contact), ())}), np.int64)
                left = np.searchsorted(busy, times-60, side='left')
                right = np.searchsorted(busy, times+60, side='right')
                times = times[left == right]
                if len(times):
                    amplitudes = np.abs(signal[times[:, None], channels[valid]])
                    times = times[np.argmax(amplitudes, axis=1) == 0]
                eligible.extend((int(t), int(c)) for t in times)
        # Deterministic source-coordinate sampling before waveform extraction.
        def key(event):
            t, c = event
            value = ((core.data_start+t)*11400714819323198485 ^ c*7046029254386353131)
            return value & ((1 << 64)-1)
        candidates = sorted(sorted(eligible, key=key)[:256])
        for t, c in candidates:
            channels = channel_map[c]
            valid = channels >= 0
            if t < 30 or t+31 > len(signal) or not np.all(quality.usable_channels[channels[valid]]):
                continue
            conflict = False
            for contact in channels[valid]:
                times = occupied.get(int(contact), ())
                j = bisect_left(times, t-60)
                if j < len(times) and times[j] <= t+60:
                    conflict = True
                    break
            if conflict:
                continue
            waveform = np.zeros((61, len(channels)), np.float32)
            waveform[:, valid] = signal[t-30:t+31, channels[valid]]
            # One physical anchor and time center per event, avoiding repeated
            # enrollment from multiple detector contacts and rebound phases.
            peak, slot = np.unravel_index(np.argmax(np.abs(waveform)), waveform.shape)
            if slot != 0 or peak != 30:
                continue
            self.observe(core.data_start+t, core.core_start, channels, waveform)

    def evaluate(self, models, now, *, enroll=False):
        self.expire(now)
        audit, added = [], 0
        for key, p in sorted(self.proposals.items()):
            rows = list(p['rows'])
            cores = sorted({row[1] for row in rows})
            record = dict(proposal_id=key, anchor=int(p['channels'][0]),
                          support=len(rows), enrolled_unit=None,
                          first_sample=rows[0][0], last_sample=rows[-1][0])
            audit.append(record)
            if len(rows) < 24 or len(cores) < 3:
                record['reason'] = 'insufficient_temporal_support'
                continue
            split = cores[(2*len(cores))//3]
            train = [r[2] for r in rows if r[1] < split]
            valid = [r[2] for r in rows if r[1] >= split]
            if len(train) < 12 or len(valid) < 6:
                record['reason'] = 'insufficient_holdout'
                continue
            template = np.median(np.stack(train), axis=0)
            consistency = float(np.quantile([
                similarity(template, p['channels'], w, p['channels']) for w in valid], .1))
            ids = sorted({unit for c in p['channels'] if c >= 0
                          for unit in models.by_channel.get(int(c), ())})
            duplicate = max((similarity(template, p['channels'], models.waveforms[u],
                                        models.channels[u]) for u in ids), default=0.)
            violations = float(np.mean(np.diff([r[0] for r in rows]) < .001*self.rate))
            record.update(consistency=consistency, existing_similarity=duplicate,
                          training_count=len(train), validation_count=len(valid),
                          sampled_refractory_violation_fraction=violations)
            if consistency < .9 or duplicate >= .85 or violations > .01:
                record['reason'] = 'consistency_duplicate_or_refractory_rejected'
                continue
            record['reason'] = 'eligible_shadow'
            if not enroll:
                continue
            if (added >= 8 or len(models.waveforms) >= 4096 or
                    any(len(models.by_channel.get(int(c), ())) >= 512
                        for c in p['channels'] if c >= 0)):
                record['reason'] = 'active_model_budget'
                continue
            record.update(reason='experimental_enrollment', enrolled_unit=len(models.waveforms))
            record['supporting_samples'] = [row[0] for row in rows]
            models.waveforms = np.concatenate((models.waveforms, template[None]))
            models.channels = np.concatenate((models.channels, p['channels'][None]))
            models.anchors = np.append(models.anchors, p['channels'][0])
            models.assigned = np.append(models.assigned, np.int64(0))
            models.version += 1
            models.__post_init__()
            added += 1
        for row in audit:
            if row['enrolled_unit'] is not None:
                del self.proposals[row['proposal_id']]
        return audit

    def save(self, handle, audit):
        group = handle.create_group('novelty')
        group.attrs.update(version=1, next_id=self.next_id, sample_rate=self.rate,
                           max_proposals=self.max_proposals)
        group.create_dataset('audit_json', data=json.dumps(audit, sort_keys=True))
        for key, p in sorted(self.proposals.items()):
            child = group.create_group(str(key))
            child.create_dataset('reference', data=p['reference'])
            child.create_dataset('channels', data=p['channels'])
            child.create_dataset('coordinates', data=np.array([(r[0], r[1]) for r in p['rows']], np.int64))
            child.create_dataset('waveforms', data=np.stack([r[2] for r in p['rows']]), compression='lzf')

    def restore(self, handle):
        group = handle['novelty']
        if (group.attrs['version'] != 1 or group.attrs['sample_rate'] != self.rate
                or group.attrs['max_proposals'] != self.max_proposals):
            raise ValueError('Novelty checkpoint configuration mismatch')
        self.next_id = int(group.attrs['next_id'])
        self.proposals.clear()
        for key in group:
            if key == 'audit_json':
                continue
            child = group[key]
            self.proposals[int(key)] = dict(reference=child['reference'][:],
                channels=child['channels'][:], rows=deque([
                    (int(t), int(c), w) for (t, c), w in zip(
                        child['coordinates'][:], child['waveforms'][:])], maxlen=64))
