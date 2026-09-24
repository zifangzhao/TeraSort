"""Bounded comparison of uninterrupted/restarted real FF2 shards and diagnostics."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import h5py
import numpy as np


def compare(left, right):
    def compare_attrs(a, b):
        keys = set(a.attrs) - {"telemetry_json"}
        assert keys == set(b.attrs) - {"telemetry_json"}
        for key in keys:
            assert np.array_equal(a.attrs[key], b.attrs[key]), key
    compare_attrs(left, right)
    groups = []
    left.visititems(lambda name, item: groups.append(name) if isinstance(item, h5py.Group) else None)
    for name in groups:
        compare_attrs(left[name], right[name])
    names_a, names_b = [], []
    left.visititems(lambda name, item: names_a.append(name) if isinstance(item, h5py.Dataset) else None)
    right.visititems(lambda name, item: names_b.append(name) if isinstance(item, h5py.Dataset) else None)
    assert names_a == names_b
    checked = 0
    for name in names_a:
        a, b = left[name], right[name]
        compare_attrs(a,b)
        assert a.shape == b.shape and a.dtype == b.dtype, name
        # Compare exact scientific payload, including FP bits; omit wall-time attrs.
        if not a.shape:
            av, bv = a[()], b[()]
            if isinstance(av, (bytes, str)):
                assert av == bv, name
            else:
                assert av.tobytes() == bv.tobytes(), name
        else:
            row_bytes = max(1, a.dtype.itemsize * int(np.prod(a.shape[1:])))
            step = max(1, 4*1024**2 // row_bytes)
            for start in range(0, len(a), step):
                assert a[start:start+step].tobytes() == b[start:start+step].tobytes(), (name,start)
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    repo = Path(__file__).resolve().parents[1]
    provenance = dict(packages={name: importlib.metadata.version(name)
                               for name in ("numpy", "scipy", "h5py", "torch", "kilosort", "cupy-cuda12x")},
                      source_sha256={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((repo / "src" / "terasort").rglob("*"))
                                     if p.suffix in (".py", ".cu")})
    (root / "report_code_environment.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    uninterrupted = sorted((root / "uninterrupted" / "FF2").glob("*.h5"))
    restarted = sorted((root / "restarted" / "FF2").glob("*.h5"))
    assert [p.name for p in uninterrupted] == [p.name for p in restarted]
    assert uninterrupted and not list(root.glob("*/*/*.partial"))
    manifest = json.loads((root / "manifest.json").read_text())
    stop = manifest["probes"][0]["segments"][0]["n_samples"]
    cursor, last_time, all_counts, all_short, all_pairs = 0, {}, {}, 0, 0
    blocks, comparisons = [], []
    for path, replay in zip(uninterrupted, restarted):
        with h5py.File(path) as f, h5py.File(replay) as g:
            assert f.attrs["complete"] and g.attrs["complete"]
            assert f.attrs["start_sample"] == cursor
            comparisons.append(dict(shard=path.name, exact_datasets=compare(f,g)))
            cursor = int(f.attrs["stop_sample"])
            spikes, qc = f["spikes"][:], f["qc"][:]
            assert qc[0]["start_sample"] == f.attrs["start_sample"]
            assert qc[-1]["stop_sample"] == cursor
            assert np.all(qc["stop_sample"][:-1] == qc["start_sample"][1:])
            assert np.all((spikes["sample_index"] >= f.attrs["start_sample"]) & (spikes["sample_index"] < cursor))
            keys = np.rec.fromarrays([spikes["sample_index"], spikes["unit_id"]])
            assert len(keys) == len(np.unique(keys))
            units, counts = np.unique(spikes["unit_id"], return_counts=True)
            short, pairs, per_unit = 0, 0, []
            for unit, count in zip(units, counts):
                unit = int(unit)
                selected = spikes["unit_id"] == unit
                times = np.sort(spikes["sample_index"][selected])
                intervals = np.diff(times)
                if unit in last_time:
                    intervals = np.append(intervals, times[0]-last_time[unit])
                last_time[unit] = int(times[-1])
                short += int(np.sum(intervals < 30))
                pairs += len(intervals)
                all_counts[unit] = all_counts.get(unit,0) + int(count)
                per_unit.append(dict(unit=unit, spikes=int(count),
                                     median_score=float(np.median(spikes["score"][selected])),
                                     median_amplitude=float(np.median(spikes["fitted_amplitude"][selected]))))
            all_short += short
            all_pairs += pairs
            bad = qc["interval_bad"] != 0
            flags = f["qc_channel_flags"][:]
            all_masked = np.all((flags & np.uint8(0xFF ^ 16)) != 0, axis=1)
            durations = (qc["stop_sample"]-qc["start_sample"])/20000
            qindex = np.searchsorted(qc["stop_sample"], spikes["sample_index"], side="right")
            block = dict(start_seconds=float(f.attrs["start_sample"])/20000,
                         stop_seconds=cursor/20000, spikes=len(spikes),
                         candidates=len(f["candidates"]), active_units=len(units),
                         isi_under_1p5ms_fraction=short/max(1,pairs),
                         artifact_seconds=int(np.sum((qc["stop_sample"]-qc["start_sample"])[bad]))/20000,
                         spikes_in_artifact_intervals=int(np.sum(bad[qindex])),
                         all_contacts_masked_seconds=float(np.sum(durations[all_masked])),
                         all_contacts_masked_intervals=[[int(q["start_sample"])/20000,int(q["stop_sample"])/20000]
                                                       for q in qc[all_masked]],
                         saturation_flag_channel_intervals=int(np.sum((flags & 1) != 0)),
                         max_saturated_fraction=float(np.max(f["qc_saturated_fraction"][:])),
                         median_noise_uv=float(np.median(f["qc_noise_uv"][:])),
                         derived_bytes=path.stat().st_size, per_unit=per_unit,
                         telemetry=json.loads(f.attrs["telemetry_json"]))
            blocks.append(block)
    assert cursor == stop
    cores = [json.loads(line) for line in (root / "uninterrupted_cores.jsonl").read_text().splitlines()]
    memory = []
    for block in blocks:
        selected = [r for r in cores if block["start_seconds"] < r["stop_sample"]/20000 <= block["stop_seconds"]]
        memory.append(dict(start_seconds=block["start_seconds"],
                           median_rss_bytes=float(np.median([r["rss_bytes"] for r in selected])),
                           peak_rss_bytes=max(r["rss_bytes"] for r in selected),
                           peak_gpu_pool_bytes=max(r["gpu_pool_bytes"] for r in selected),
                           peak_total_vram_bytes=max(r["total_vram_bytes"] for r in selected)))
    original = json.loads((root / "uninterrupted_summary.json").read_text())
    result = dict(duration_seconds=stop/20000, integrity_passed=True,
                  exact_payload_comparisons=comparisons, memory_by_shard=memory,
                  active_units=len(all_counts), isi_under_1p5ms_fraction=all_short/max(1,all_pairs),
                  bad_interval_seconds=sum(b["artifact_seconds"] for b in blocks),
                  original_summary=original, blocks=blocks)
    (root / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    minutes = np.array([r["stop_sample"] for r in cores])/20000/60
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    for key, label in [("rss_bytes", "Process RSS"), ("gpu_pool_bytes", "CUDA pool"),
                       ("total_vram_bytes", "Total device VRAM")]:
        axes[0].plot(minutes, np.array([r[key] for r in cores])/1e6, label=label)
    axes[0].set(ylabel="MB (decimal)", title="FF2: complete real recording, frozen 72-template bank")
    axes[0].legend(loc="upper right")
    samples = np.array([0]+[r["stop_sample"] for r in cores])/20000
    wall = np.array([0]+[r["wall_seconds"] for r in cores])
    ends = np.arange(30, len(samples), 30)
    axes[1].plot(samples[ends]/60, (samples[ends]-samples[ends-30])/(wall[ends]-wall[ends-30]))
    axes[1].set(ylabel="Recording seconds / wall second", xlabel="Recording minute")
    order = sorted(all_counts, key=all_counts.get, reverse=True)
    rates = np.zeros((len(order), len(blocks)))
    for col, block in enumerate(blocks):
        for unit in block["per_unit"]:
            rates[order.index(unit["unit"]),col] = unit["spikes"]/(block["stop_seconds"]-block["start_seconds"])
    display = axes[2].imshow(np.log1p(rates), aspect="auto", interpolation="nearest", cmap="viridis")
    axes[2].set(xticks=range(len(blocks)),
                xticklabels=[f'{b["start_seconds"]/60:.0f}–{b["stop_seconds"]/60:.1f}' for b in blocks],
                xlabel="Recording-minute interval", ylabel="Template rank by total spike count",
                title="Activity continuity only; this does not validate neuron identity")
    fig.colorbar(display, ax=axes[2], label="log(1 + assigned spikes/s)")
    fig.savefig(root / "streaming_validation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k != "blocks"}, indent=2))


if __name__ == "__main__":
    main()
