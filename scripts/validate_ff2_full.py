"""Full FF2 stream plus a killed-process replay; never modifies raw sources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(root, name, resume=False, interrupt=False):
    import cupy as cp
    import psutil
    from terasort.session_sort import run_session
    process = psutil.Process()
    started = time.perf_counter()
    tag = name + ("_resume" if resume else "")
    with (root / (tag + "_cores.jsonl")).open("x", encoding="utf-8") as log:
        def progress(row):
            if row["stage"] == "core_complete":
                free, total = cp.cuda.runtime.memGetInfo()
                row.update(wall_seconds=time.perf_counter()-started,
                           rss_bytes=process.memory_info().rss,
                           gpu_pool_bytes=cp.get_default_memory_pool().total_bytes(),
                           total_vram_bytes=total-free)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if row["stop_sample"] % 1200000 == 0:
                    print(json.dumps(row), flush=True)
                if interrupt and row["stop_sample"] >= 6200000:
                    print("READY_FOR_PROCESS_KILL", flush=True)
                    while True:
                        time.sleep(1)
        result = run_session(root / "manifest.json", root / name,
                             backend="cuda", resume=resume, freeze_templates=True,
                             rescue_floor_snr=3.5, score_floor=.75,
                             progress=progress)
    (root / (tag + "_summary.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(stage="finished", name=tag, result=result)), flush=True)


def launch(root, name, resume=False, interrupt=False):
    args = [sys.executable, "-u", __file__, str(root), "--worker", name]
    if resume:
        args.append("--resume")
    if interrupt:
        args.append("--interrupt")
    tag = name + ("_resume" if resume else "")
    with (root / (tag + ".log")).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        killed = False
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            if line.strip() == "READY_FOR_PROCESS_KILL":
                process.kill()
                killed = True
        code = process.wait()
        if interrupt:
            assert killed and code != 0, (killed, code)
            shards = sorted((root / name / "FF2").glob("*.h5"))
            partials = sorted((root / name / "FF2").glob("*.partial"))
            assert len(shards) == 1 and len(partials) == 1
            (root / "kill_evidence.json").write_text(json.dumps(dict(
                exit_code=code, completed=[p.name for p in shards],
                incomplete=[p.name for p in partials], kill_after_sample=6200000), indent=2))
        elif code:
            raise RuntimeError(f"{tag} exited with {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--worker")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--interrupt", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.root, args.worker, args.resume, args.interrupt)
    args.root.mkdir(parents=True, exist_ok=False)
    original = Path(r"F:\sortingDevelopment\ff2_real_pilot_20260924_01\manifest.json")
    manifest = json.loads(original.read_text())
    manifest["session_id"] = "FF2-D82-full-validation"
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.root / "provenance.json").write_text(json.dumps(dict(
        pilot_manifest=str(original), seed_bank="unchanged pilot 72-unit bank",
        geometry="retained SCB coordinates and acquisition XML shanks; physical-map verification pending",
        configuration="raw detector, score .75, rescue SNR 3.5, frozen templates",
        scope="single real recording; no quality or 2-TB capability claim"), indent=2))
    launch(args.root, "uninterrupted")
    launch(args.root, "restarted", interrupt=True)
    launch(args.root, "restarted", resume=True)


if __name__ == "__main__":
    main()
