"""Reproducible bounded FF2 pilot; sources and prior outputs remain read-only."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from terasort.session_manifest import load_session
from terasort.session_sort import run_session


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=False)
    source = Path(r"\\132.236.112.15\ayadataB2\Data\Fabrication\FF2\Day82toDay100\D82_LinearA_250725_000002")
    bank = Path(r"F:\sortingDevelopment\terasort_ff2_streaming_20260924\ff2_d82_first60s_floor4p5.scb.h5")
    xml = ET.parse(source / "amplifier.xml").getroot()
    groups = xml.findall("./anatomicalDescription/channelGroups/group")
    with h5py.File(bank) as f:
        positions = f["channel_positions_um"][:]
    shanks = np.full(len(positions), -1, dtype=int)
    for index, group in enumerate(groups):
        for ch in group.findall("channel"):
            channel = int(ch.text)
            assert shanks[channel] == -1
            shanks[channel] = index
    assert np.all(shanks >= 0) and len(positions) == 128
    rate = int(xml.findtext("./acquisitionSystem/samplingRate"))
    assert rate == 20000
    raw = source / "amplifier.dat"
    size = raw.stat().st_size
    assert size % (128 * 2) == 0
    probe = dict(probe_id="FF2", sample_rate_hz=rate, gain_uv_per_count=.195,
                 geometry=dict(x_um=positions[:, 0].tolist(), y_um=positions[:, 1].tolist(),
                               shank=shanks.tolist()), gaps=[],
                 segments=[dict(path=str(raw), start_sample=0, n_samples=size//256, day_id="D82")])
    manifest = dict(schema_version=1, session_id="FF2-D82-real-pilot", probes=[probe])
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps(dict(
        geometry_source=str(bank), shank_source=str(source / "amplifier.xml"),
        gain_source="Intan .195 uV/count; retained SCB preprocessing metadata",
        geometry_status="retained prior trial coordinates; independent physical-map verification pending",
        ground_truth=False, source_bytes=size), indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(Path(__file__).with_name("learn_session_seeds.py")),
                    "--manifest", str(path), "--output-root", str(root / "learning"),
                    "--probe-id", "FF2", "--day-id", "D82", "--budget-seconds", "60"], check=True)
    probe.update(seed_templates=str(root / "learning" / "seeds.npz"),
                 seed_preprocessing_id="terasort-session-v1")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sort_existing(root)


def sort_existing(root):
    root = Path(root)
    path = root / "manifest.json"
    session = load_session(path)
    rate = int(session.probes[0].sample_rate_hz)
    for name, mode, floor in [("raw", "raw", .65), ("smooth3", "smooth3", .75),
                              ("raw_strict", "raw", .75)]:
        result = run_session(path, root / name, backend="cuda", start_sample=60*rate,
                             stop_sample=90*rate, freeze_templates=True, rescue_floor_snr=3.5,
                             detector_mode=mode, score_floor=floor,
                             progress=lambda row: print(json.dumps(row), flush=True))
        (root / (name + "_summary.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
