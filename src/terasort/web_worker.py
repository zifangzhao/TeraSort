"""Run one queued sort independently of the dashboard process."""

import json
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    job_file = Path(sys.argv[1])
    job = json.loads(job_file.read_text(encoding="utf-8"))
    request = job["request"]
    args = ["sort", "--settings", request["settings"],
            "--results-dir", request["results_dir"], "--backend", request["backend"]]
    for filename in request.get("filenames", [request["filename"]]):
        args += ["--filename", filename]
    if request.get("probe_json"):
        args += ["--probe-json", request["probe_json"]]
    elif request.get("probe_name"):
        args += ["--probe-name", request["probe_name"]]
    for key, flag in (("skip_drift_correction", "--skip-drift-correction"),
                      ("no_fast_int16", "--no-fast-int16")):
        if request.get(key):
            args.append(flag)
    if request.get("lfp_output"):
        args += ["--lfp-output", request["lfp_output"]]
    if request.get("stage_dir"):
        args += ["--stage-dir", request["stage_dir"]]
    if request.get("read_cache_dir"):
        args += ["--read-cache-dir",request["read_cache_dir"],
                 "--read-cache-mb",str(request["read_cache_mb"]),
                 "--read-cache-slots",str(request["read_cache_slots"]),
                 "--read-cache-workers",str(request.get("read_cache_workers", 4))]
    code = 1
    try:
        print("TeraSort worker started; initializing Kilosort and CUDA...", flush=True)
        from .cli import main as cli_main
        code = cli_main(args)
    except BaseException:
        traceback.print_exc()
    finally:
        done = job_file.with_name("done.json")
        temporary = done.with_suffix(".tmp")
        temporary.write_text(json.dumps({"exit_code": code, "finished_at": time.time()}),
                             encoding="utf-8")
        temporary.replace(done)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
