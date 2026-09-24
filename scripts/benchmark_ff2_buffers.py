"""Warm-read throughput and paired end-to-end FF2 buffer experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--buffer", type=int, default=0)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    manifest = Path(r"F:\sortingDevelopment\ff2_full_validation_20260924_01\manifest.json")
    if args.worker is not None:
        from terasort.session_sort import run_session
        result = run_session(manifest, args.root / f"run-{args.worker}", backend="cuda",
                             stop_sample=2400000, freeze_templates=True,
                             rescue_floor_snr=3.5, score_floor=.75,
                             read_buffer_mb=args.buffer, prefetch_depth=args.depth)
        result.update(buffer_mb=args.buffer, prefetch_depth=args.depth)
        (args.root / f"run-{args.worker}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    args.root.mkdir(parents=True, exist_ok=False)
    from terasort.session_manifest import load_session
    from terasort.session_signal import iter_cores
    probe = load_session(manifest).probes[0]
    reads = []
    for buffer_mb in (0,64,256,256,64,0):
        started, received, count = time.perf_counter(), 0, 0
        for core in iter_cores(probe, stop_sample=1200000, read_buffer_mb=buffer_mb):
            received += core.source_bytes_read
            count += int(core.source_bytes_read != 0)
        row = dict(buffer_mb=buffer_mb, wall_seconds=time.perf_counter()-started,
                   source_bytes_read=received, requests=count,
                   raw_bytes_covered=1200000*128*2)
        reads.append(row)
        print(json.dumps(dict(stage="read_only", **row)), flush=True)
    (args.root / "read_only.json").write_text(json.dumps(reads, indent=2))
    settings = [(0,2),(64,2),(256,2),(64,8),(64,8),(256,2),(64,2),(0,2)]
    for i, (buffer_mb, depth) in enumerate(settings):
        with (args.root / f"run-{i}.log").open("x", encoding="utf-8") as log:
            command = [sys.executable, "-u", __file__, str(args.root), "--worker",str(i),
                       "--buffer",str(buffer_mb),"--depth",str(depth)]
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print((args.root / f"run-{i}.json").read_text(), flush=True)
    import h5py
    # IO telemetry/config differ by design, so compare all scientific arrays
    # individually; the QC record's source-read accounting is excluded.
    paths = [next((args.root / f"run-{i}" / "FF2").glob("*.h5")) for i in range(len(settings))]
    with h5py.File(paths[0]) as base:
        names = []
        base.visititems(lambda name,item: names.append(name) if isinstance(item,h5py.Dataset) else None)
        for path in paths[1:]:
            with h5py.File(path) as other:
                for name in names:
                    a,b=base[name],other[name]
                    assert a.shape==b.shape and a.dtype==b.dtype,(path,name)
                    if name == "qc":
                        for field in ("start_sample","stop_sample","interval_bad"):
                            assert a[:][field].tobytes()==b[:][field].tobytes(),(path,field)
                    elif not a.shape:
                        assert a[()]==b[()],(path,name)
                    else:
                        for start in range(0,len(a),1024):
                            assert a[start:start+1024].tobytes()==b[start:start+1024].tobytes(),(path,name,start)
    (args.root / "integrity.json").write_text(json.dumps(dict(
        exact_scientific_payloads=True, runs=len(paths), datasets_per_run=len(names),
        excluded="QC source_bytes_read and run/telemetry attributes",
        note="Sequential warm-cache trials; read throughput is application-level, not NIC wire traffic"),indent=2))


if __name__ == "__main__":
    main()
