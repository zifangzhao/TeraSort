"""Auditable ground-truth gate for opt-in session sorting.

This evaluator is bounded by the benchmark dataset, not by a 100 TB archive.
Use the standard SpikeInterface comparison as an independent cross-check
before a publication claim.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment


def _one_to_one(left, right, tolerance):
    i = j = matched = 0
    while i < len(left) and j < len(right):
        delta = int(left[i]) - int(right[j])
        if abs(delta) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matched


def _collision_keys(times, labels, tolerance):
    """Events close to a spike from a different ground-truth unit."""
    order = np.argsort(times, kind="stable")
    times = np.asarray(times)[order]
    labels = np.asarray(labels)[order]
    collided = set()
    for i in range(1, len(times)):
        if times[i] - times[i-1] <= tolerance and labels[i] != labels[i-1]:
            collided.add((labels[i-1], int(times[i-1])))
            collided.add((labels[i], int(times[i])))
    return collided


def _group_times(times, labels):
    order = np.argsort(times, kind="stable")
    result = defaultdict(list)
    for time, label in zip(np.asarray(times)[order], np.asarray(labels)[order]):
        result[label].append(int(time))
    return {unit: np.asarray(values, np.int64) for unit, values in result.items()}


def score_sorting(gt_times, gt_labels, predicted_times, predicted_labels, *,
                  tolerance_samples, collision_tolerance_samples=None,
                  late_start_sample=None):
    """One-to-one unit matching with IoU and spike precision/recall."""
    if tolerance_samples < 0:
        raise ValueError("Negative spike match tolerance")
    gt = _group_times(gt_times, gt_labels)
    predicted = _group_times(predicted_times, predicted_labels)
    collision_tolerance = (tolerance_samples if collision_tolerance_samples is None
                           else collision_tolerance_samples)
    collision_keys = _collision_keys(gt_times, gt_labels, collision_tolerance)
    duration_midpoint = (late_start_sample if late_start_sample is not None
                         else ((min(gt_times) + max(gt_times)) / 2
                               if len(gt_times) else 0))
    late_gt = {unit for unit, times in gt.items()
               if times[0] > duration_midpoint}
    rare_gt = {unit for unit, times in gt.items() if len(times) <= 50}
    gt_ids, pred_ids = list(gt), list(predicted)
    if not gt_ids or not pred_ids:
        return {"gt_units": len(gt_ids), "predicted_units": len(pred_ids),
                "recovered_units_iou_0p8": 0, "true_positive_spikes": 0,
                "spike_precision": 0., "spike_recall": 0.,
                "split_gt_units": 0, "merged_predicted_units": 0,
                "collision_gt_spikes": len(collision_keys),
                "collision_recovered_spikes": 0,
                "rare_gt_units": len(rare_gt), "rare_recovered_units": 0,
                "late_gt_units": len(late_gt), "late_recovered_units": 0,
                "unit_matches": []}
    # A time sweep first discovers sparse unit pairs. No all-unit waveform
    # comparison or full recording x unit score tensor is built.
    gt_t = np.asarray(gt_times, np.int64)
    pred_t = np.asarray(predicted_times, np.int64)
    gt_l = np.asarray(gt_labels)
    pred_l = np.asarray(predicted_labels)
    gt_order = np.argsort(gt_t, kind="stable")
    pred_order = np.argsort(pred_t, kind="stable")
    gt_t, gt_l = gt_t[gt_order], gt_l[gt_order]
    pred_t, pred_l = pred_t[pred_order], pred_l[pred_order]
    possible = set()
    left = 0
    for time, label in zip(pred_t, pred_l):
        while left < len(gt_t) and gt_t[left] < time - tolerance_samples:
            left += 1
        k = left
        while k < len(gt_t) and gt_t[k] <= time + tolerance_samples:
            possible.add((gt_l[k], label))
            k += 1
    gi = {unit: index for index, unit in enumerate(gt_ids)}
    pi = {unit: index for index, unit in enumerate(pred_ids)}
    iou = np.zeros((len(gt_ids), len(pred_ids)), np.float32)
    matched = np.zeros_like(iou, dtype=np.int64)
    for gt_id, pred_id in possible:
        count = _one_to_one(gt[gt_id], predicted[pred_id],
                            tolerance_samples)
        union = len(gt[gt_id]) + len(predicted[pred_id]) - count
        iou[gi[gt_id], pi[pred_id]] = count / union if union else 0
        matched[gi[gt_id], pi[pred_id]] = count
    row, col = linear_sum_assignment(iou, maximize=True)
    links = [{
        "ground_truth_unit": str(gt_ids[i]),
        "predicted_unit": str(pred_ids[j]),
        "iou": float(iou[i, j]),
        "matched_spikes": int(matched[i, j]),
    } for i, j in zip(row, col) if iou[i, j] > 0]
    true_positive = sum(link["matched_spikes"] for link in links)
    # These are diagnostics rather than extra acceptance gates. A split has
    # substantial spike overlap with two output units; a merge is reciprocal.
    overlap_gt = matched / np.maximum(1, np.array(
        [len(gt[unit]) for unit in gt_ids], np.int64))[:, None]
    overlap_pred = matched / np.maximum(1, np.array(
        [len(predicted[unit]) for unit in pred_ids], np.int64))[None, :]
    recovered = {gt_ids[i] for i, j in zip(row, col) if iou[i, j] >= .8}
    collision_recovered = 0
    for i, j in zip(row, col):
        if not matched[i, j]:
            continue
        left, right = gt[gt_ids[i]], predicted[pred_ids[j]]
        a = b = 0
        while a < len(left) and b < len(right):
            delta = int(left[a]) - int(right[b])
            if abs(delta) <= tolerance_samples:
                collision_recovered += (gt_ids[i], int(left[a])) in collision_keys
                a += 1
                b += 1
            elif delta < 0:
                a += 1
            else:
                b += 1
    return {
        "gt_units": len(gt_ids), "predicted_units": len(pred_ids),
        "recovered_units_iou_0p8": sum(link["iou"] >= .8 for link in links),
        "true_positive_spikes": true_positive,
        "spike_precision": true_positive / len(pred_t) if len(pred_t) else 0.,
        "spike_recall": true_positive / len(gt_t) if len(gt_t) else 0.,
        "split_gt_units": int(np.sum(np.sum(overlap_gt >= .2, axis=1) >= 2)),
        "merged_predicted_units": int(np.sum(np.sum(overlap_pred >= .2, axis=0) >= 2)),
        "collision_gt_spikes": len(collision_keys),
        "collision_recovered_spikes": collision_recovered,
        "rare_gt_units": len(rare_gt),
        "rare_recovered_units": len(recovered & rare_gt),
        "late_gt_units": len(late_gt),
        "late_recovered_units": len(recovered & late_gt),
        "unit_matches": links,
    }


def load_session_spikes(root, probe_id, *, first=0, stop=None,
                        resolve_template_links=False, export_filter=False):
    folder = Path(root) / probe_id
    times, labels = [], []
    spans = []
    day_link_maps = {}
    for path in sorted(folder.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            if not handle.attrs.get("complete", False):
                raise ValueError(f"Incomplete shard: {path}")
            start = int(handle.attrs["start_sample"])
            end = int(handle.attrs["stop_sample"])
            if end <= first or (stop is not None and start >= stop):
                continue
            spans.append((max(start, first), min(end, stop) if stop is not None else end))
            spikes = handle["spikes"][:]
            if export_filter and "spike_export_mask" in handle:
                export_mask = handle["spike_export_mask"][:]
                if export_mask.shape != (len(spikes),) or np.any(export_mask > 1):
                    raise ValueError(f"Invalid spike export mask in shard: {path}")
                spikes = spikes[export_mask.astype(bool)]
            keep = spikes["sample_index"] >= first
            if stop is not None:
                keep &= spikes["sample_index"] < stop
            spikes = spikes[keep]
            times.append(spikes["sample_index"])
            # Day/probe namespace prevents accidental cross-day identity.
            day_id = str(handle.attrs["day_id"])
            units = spikes["unit_id"]
            if resolve_template_links:
                if "template_linking/template_to_unit" not in handle:
                    raise ValueError(
                        f"Template links requested but absent from shard: {path}"
                    )
                mapping = handle["template_linking/template_to_unit"][:]
                previous = day_link_maps.get(day_id)
                if previous is not None and not np.array_equal(previous, mapping):
                    raise ValueError(f"Inconsistent template map for day {day_id}")
                day_link_maps[day_id] = mapping
                if len(units) and (np.any(units < 0) or np.any(units >= len(mapping))):
                    raise ValueError(f"Shard spike references missing template map entry: {path}")
                units = mapping[units]
            labels.extend(f"{probe_id}/{day_id}/{int(unit)}"
                          for unit in units)
    if not spans:
        raise ValueError("No completed session shards in benchmark interval")
    return (np.concatenate(times) if times else np.empty(0, np.int64),
            np.asarray(labels, dtype=str), spans)


def _inside_spans(times, spans):
    """Mask to the exact disjoint source-clock intervals with completed shards."""
    times = np.asarray(times, np.int64)
    keep = np.zeros(times.shape, dtype=bool)
    for first, stop in spans:
        keep |= (times >= first) & (times < stop)
    return keep


def evaluate_case(case):
    required = {"name", "kind", "ground_truth", "session_output",
                "probe_id", "kilosort_dir"}
    if not required.issubset(case):
        raise ValueError(f"Benchmark case is missing {required - set(case)}")
    first = int(case.get("start_sample", 0))
    stop = case.get("stop_sample")
    stop = int(stop) if stop is not None else None
    with np.load(case["ground_truth"], allow_pickle=False) as data:
        gt_t = np.asarray(data["times"], np.int64)
        gt_l = np.asarray(data["labels"], np.int64)
        rate = float(data["sampling_frequency"])
    if gt_t.shape != gt_l.shape or rate <= 0:
        raise ValueError("Invalid ground-truth bundle")
    tolerance = max(1, round(rate * .0004))
    collision_tolerance = max(1, round(rate * .0006))
    resolve_template_links = bool(case.get("resolve_template_links", False))
    export_filter = bool(case.get("export_filter", False))
    pred_t, pred_l, spans = load_session_spikes(
        case["session_output"], case["probe_id"], first=first, stop=stop,
        resolve_template_links=resolve_template_links,
        export_filter=export_filter)
    keep = _inside_spans(gt_t, spans)
    gt_t, gt_l = gt_t[keep], gt_l[keep]
    ks = Path(case["kilosort_dir"])
    ks_t = np.load(ks / "spike_times.npy", mmap_mode="r").ravel()
    ks_l = np.load(ks / "spike_clusters.npy", mmap_mode="r").ravel()
    keep = _inside_spans(ks_t, spans)
    result = {
        "name": case["name"], "kind": case["kind"],
        "template_links_resolved": resolve_template_links,
        "export_filter_applied": export_filter,
        "tolerance_samples": tolerance,
        "collision_tolerance_samples": collision_tolerance,
        "session": score_sorting(gt_t, gt_l, pred_t, pred_l,
                                 tolerance_samples=tolerance,
                                 collision_tolerance_samples=collision_tolerance,
                                 late_start_sample=(min(a for a, _ in spans) +
                                                    max(b for _, b in spans)) / 2),
        "kilosort4": score_sorting(gt_t, gt_l, ks_t[keep], ks_l[keep],
                                  tolerance_samples=tolerance,
                                  collision_tolerance_samples=collision_tolerance,
                                  late_start_sample=(min(a for a, _ in spans) +
                                                     max(b for _, b in spans)) / 2),
    }
    return result


def evaluate_suite(suite_path):
    suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Quality suite needs cases")
    results = [evaluate_case(case) for case in cases]
    gt_units = sum(item["session"]["gt_units"] for item in results)
    if gt_units == 0 or any(item["session"]["gt_units"] == 0 for item in results):
        raise ValueError("Every quality case needs ground-truth units in completed spans")
    recovery_loss = sum(
        item["kilosort4"]["recovered_units_iou_0p8"] -
        item["session"]["recovered_units_iou_0p8"] for item in results
    ) / gt_units
    def weighted(metric, route):
        # Dataset-level mean keeps the acceptance gate transparent even when
        # datasets differ greatly in duration and spike density.
        return float(np.mean([item[route][metric] for item in results]))
    precision_loss = weighted("spike_precision", "kilosort4") - weighted(
        "spike_precision", "session")
    recall_loss = weighted("spike_recall", "kilosort4") - weighted(
        "spike_recall", "session")
    kinds = {item["kind"] for item in results}
    ready = len(results) >= 3 and {"base", "drift", "collision"}.issubset(kinds)
    subgroup_losses = []
    for item in results:
        session, baseline = item["session"], item["kilosort4"]
        row = {
            "name": item["name"],
            "recovery_fraction_loss": (
                baseline["recovered_units_iou_0p8"] -
                session["recovered_units_iou_0p8"]) / session["gt_units"],
            "precision_loss": baseline["spike_precision"] -
                              session["spike_precision"],
            "recall_loss": baseline["spike_recall"] -
                           session["spike_recall"],
        }
        for name, denominator, numerator in (
                ("collision_recall", "collision_gt_spikes",
                 "collision_recovered_spikes"),
                ("rare_unit_recovery", "rare_gt_units",
                 "rare_recovered_units"),
                ("late_unit_recovery", "late_gt_units",
                 "late_recovered_units")):
            count = session[denominator]
            row[name + "_loss"] = ((baseline[numerator] - session[numerator]) /
                                    count if count else None)
        subgroup_losses.append(row)
    passed = bool(ready and recovery_loss <= .02 and
                  precision_loss <= .02 and recall_loss <= .02 and
                  all(max(value for key, value in row.items()
                          if key.endswith("_loss") and value is not None) <= .05
                      for row in subgroup_losses))
    return {
        "quality_gate_passed": passed,
        "suite_complete": ready,
        "aggregate_recovery_fraction_loss": recovery_loss,
        "mean_precision_loss": precision_loss,
        "mean_recall_loss": recall_loss,
        "subgroup_losses": subgroup_losses,
        "cases": results,
    }
