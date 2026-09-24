"""Descriptive real-data QC; these metrics are not ground-truth accuracy."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np

root = Path(sys.argv[1])
rows = []
for name in ("raw", "smooth3", "raw_strict"):
    result = json.loads((root / (name + "_summary.json")).read_text())
    with h5py.File(next((root / name / "FF2").glob("*.h5"))) as f:
        spikes, qc = f["spikes"][:], f["qc"][:]
        units, counts = np.unique(spikes["unit_id"], return_counts=True)
        violations, pairs = 0, 0
        per_unit = []
        for unit, count in zip(units, counts):
            times = np.sort(spikes["sample_index"][spikes["unit_id"] == unit])
            bad = int(np.sum(np.diff(times) < 30))  # 1.5 ms at 20 kHz
            violations += bad
            pairs += max(0, len(times)-1)
            per_unit.append(dict(unit=int(unit), spikes=int(count), isi_under_1p5ms=bad))
        bad_intervals = qc[qc["interval_bad"] != 0]
        bad_spikes = sum(int(np.sum((spikes["sample_index"] >= q["start_sample"]) &
                                    (spikes["sample_index"] < q["stop_sample"]))) for q in bad_intervals)
        result.update(name=name, active_units=len(units), units_at_least_30_spikes=int(np.sum(counts >= 30)),
                      isi_under_1p5ms_fraction=violations/max(1,pairs),
                      bad_interval_seconds=sum(int(q["stop_sample"]-q["start_sample"])/20000 for q in bad_intervals),
                      spikes_in_bad_intervals=bad_spikes,
                      noise_median_uv=float(np.median(f["qc_noise_uv"][:])),
                      flagged_channels_any=int(np.sum(np.any((f["qc_channel_flags"][:] & 15) != 0, axis=0))),
                      median_score=float(np.median(spikes["score"])), per_unit=per_unit)
        assert np.all((spikes["sample_index"] >= 1200000) & (spikes["sample_index"] < 1800000))
        keys = np.rec.fromarrays([spikes["sample_index"], spikes["unit_id"]])
        result["duplicate_sample_unit_pairs"] = len(keys)-len(np.unique(keys))
    rows.append(result)
(root / "diagnostics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps([{k:v for k,v in row.items() if k != "per_unit"} for row in rows], indent=2))
