"""Web queue contract and stage estimates, without requiring a GPU."""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from terasort.web import JobManager, _job_log, _progress, browse, create_handler, validate_request


def _request(tmp_path):
    raw = tmp_path / "recording.bin"
    raw.write_bytes(b"\0" * 24)
    settings = tmp_path / "settings.json"
    settings.write_text('{"n_chan_bin": 2, "fs": 30000}')
    probe = tmp_path / "probe.json"
    probe.write_text('{"chanMap": [0, 1]}')
    return {"filename": str(raw), "settings": str(settings), "probe_json": str(probe),
            "results_dir": str(tmp_path / "result"), "backend": "auto"}


def test_validation_protects_input_and_result(tmp_path):
    request = _request(tmp_path)
    assert validate_request(request)["filename"] == request["filename"]
    request["results_dir"] = request["filename"]
    with pytest.raises(ValueError, match="directory"):
        validate_request(request)
    request["results_dir"] = str(tmp_path / "result")
    with pytest.raises(ValueError, match="Another job"):
        validate_request(request, {request["results_dir"]})
    request["lfp_output"] = request["filename"]
    with pytest.raises(ValueError, match="exists"):
        validate_request(request)


def test_validation_accepts_ordered_session(tmp_path):
    request = _request(tmp_path)
    second = tmp_path / "second.bin"
    second.write_bytes(b"\0" * 24)
    request.pop("filename")
    request["filenames"] = [str(tmp_path / "recording.bin"), str(second)]
    validated = validate_request(request)
    assert validated["filenames"] == request["filenames"]
    assert validated["filename"] == request["filenames"][0]
    request["filenames"].append(str(second))
    with pytest.raises(ValueError, match="more than once"):
        validate_request(request)
    request["filenames"].pop()
    request["lfp_output"] = str(tmp_path / "lfp.i16")
    with pytest.raises(ValueError, match="one source"):
        validate_request(request)


def test_validation_requires_new_separate_staging_directory(tmp_path):
    request = _request(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "recording.bin").write_bytes((tmp_path / "recording.bin").read_bytes())
    request["filename"] = str(source / "recording.bin")
    request["stage_dir"] = str(tmp_path / "scratch")
    assert validate_request(request)["stage_dir"] == request["stage_dir"]
    with pytest.raises(ValueError, match="staging directory"):
        validate_request(request, {request["stage_dir"]})
    (tmp_path / "scratch").mkdir()
    with pytest.raises(ValueError, match="must be new"):
        validate_request(request)


def test_read_cache_validation(tmp_path):
    request = _request(tmp_path)
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'recording.bin').write_bytes(b'\0' * 24)
    request['filename'] = str(source / 'recording.bin')
    request['read_cache_dir'] = str(tmp_path / 'cache')
    validated = validate_request(request)
    assert validated['read_cache_mb'] == 256
    assert validated['read_cache_slots'] == 3
    request['read_cache_slots'] = 1
    with pytest.raises(ValueError, match='slots'):
        validate_request(request)
    request['read_cache_slots'] = 3
    request['no_fast_int16'] = True
    with pytest.raises(ValueError, match='fast INT16'):
        validate_request(request)
    request['no_fast_int16'] = False
    request['read_cache_dir'] = str(source / 'cache')
    with pytest.raises(ValueError, match='outside source'):
        validate_request(request)


def test_browse_and_stage_eta(tmp_path):
    _request(tmp_path)
    listing = browse(str(tmp_path))
    assert listing["parent"] == str(tmp_path.parent)
    assert {entry["name"] for entry in listing["entries"]} >= {"recording.bin", "settings.json"}
    assert _progress("Computing preprocessing variables", 10, "running")["eta_seconds"] is None
    progress = _progress("Computing drift correction\nExtracting spikes using templates", 50, "running")
    assert progress["stage"] == "Template detection"
    assert progress["percent"] == 21
    assert progress["eta_seconds"] > 0
    staging = _progress("TERASORT_STAGING 50", 40, "running", staged=True)
    assert staging == {"stage": "Staging input", "percent": 5, "eta_seconds": 40}
    assert _progress("TERASORT_STAGING 100\nComputing drift correction", 100,
                     "running", staged=True)["percent"] >= 10
    result = tmp_path / "result"
    result.mkdir()
    (result / "kilosort4.log").write_text("Computing drift correction")
    assert "Computing drift correction" in _job_log({"id": "abc", "request": {"results_dir": str(result)}}, tmp_path)


def test_http_token_and_persistent_queue(tmp_path, monkeypatch):
    # Keep the scheduler from launching a real Kilosort process in this test.
    monkeypatch.setattr(JobManager, "_launch", lambda self, job: None)
    manager = JobManager(tmp_path / "state")
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(manager, "test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as denied:
            urlopen(base + "/api/jobs")
        assert denied.value.code == 401
        payload = json.dumps(_request(tmp_path)).encode()
        req = Request(base + "/api/jobs", payload, method="POST",
                      headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"})
        with urlopen(req) as response:
            job = json.load(response)
        assert job["status"] == "queued"
        assert (tmp_path / "state" / job["id"] / "job.json").is_file()
        with urlopen(Request(base + "/api/jobs", headers={"Authorization": "Bearer test-token"})) as response:
            assert json.load(response)["jobs"][0]["id"] == job["id"]
        with urlopen(Request(base + "/api/metrics", headers={"Authorization": "Bearer test-token"})) as response:
            assert "system_cpu_percent" in json.load(response)
    finally:
        server.shutdown()
        server.server_close()


def test_local_api_rejects_untrusted_host(tmp_path):
    manager = JobManager(tmp_path / "state")
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(manager))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/jobs"
        with pytest.raises(HTTPError) as denied:
            urlopen(Request(url, headers={"Host": "attacker.example"}))
        assert denied.value.code == 403
        with urlopen(url) as response:
            assert "jobs" in json.load(response)
    finally:
        server.shutdown()
        server.server_close()
