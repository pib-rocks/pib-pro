"""Tests for Thor probe (synthetic sysfs) and setup-thor.sh dry-run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pib_pro.diagnostics.thor_probe import (
    ThorProbeConfig,
    ThorProbePaths,
    evaluate_headroom,
    evaluate_thermal,
    parse_meminfo_kb,
    run_thor_probe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = REPO_ROOT / "setup" / "setup-thor.sh"


def _write_fixture_tree(base: Path, *, temp_mc: int, power_micro_w: int) -> ThorProbePaths:
    thermal = base / "sys" / "class" / "thermal" / "thermal_zone0"
    thermal.mkdir(parents=True)
    (thermal / "temp").write_text(str(temp_mc), encoding="utf-8")
    (thermal / "type").write_text("CPU-therm", encoding="utf-8")

    hwmon = base / "sys" / "class" / "hwmon" / "hwmon1"
    hwmon.mkdir(parents=True)
    (hwmon / "power1_input").write_text(str(power_micro_w), encoding="utf-8")

    meminfo = base / "proc" / "meminfo"
    meminfo.parent.mkdir(parents=True)
    meminfo.write_text(
        "MemTotal:       65536000 kB\nMemFree:        32000000 kB\n",
        encoding="utf-8",
    )

    jetson = base / "etc" / "nv_tegra_release"
    jetson.parent.mkdir(parents=True)
    jetson.write_text("# R36 (release), REVISION: 7.0\n", encoding="utf-8")

    return ThorProbePaths(
        meminfo=meminfo,
        thermal_glob=str(base / "sys" / "class" / "thermal" / "thermal_zone*"),
        hwmon_glob=str(base / "sys" / "class" / "hwmon" / "hwmon*"),
        gpu_clock_glob=str(base / "sys" / "class" / "devfreq" / "missing" / "*"),
        jetson_release=jetson,
    )


def test_parse_meminfo_kb() -> None:
    parsed = parse_meminfo_kb("MemTotal:       1024 kB\n")
    assert parsed["MemTotal"] == 1024


def test_evaluate_helpers() -> None:
    ok, pct = evaluate_headroom(30_000.0, 60_000.0, 30.0)
    assert ok is True
    assert pct == pytest.approx(50.0)
    ok_t, max_c = evaluate_thermal([{"temp_c": 70.0}], 85.0)
    assert ok_t is True
    assert max_c == 70.0


def test_run_thor_probe_healthy_fixture(tmp_path: Path) -> None:
    paths = _write_fixture_tree(tmp_path, temp_mc=50_000, power_micro_w=10_000_000)
    report = run_thor_probe(
        ThorProbeConfig(
            power_rail_limit_mw=60_000.0,
            power_headroom_min_pct=30.0,
            thermal_throttle_c=85.0,
            paths=paths,
        )
    )
    assert report.healthy is True
    assert report.power_headroom_ok is True
    assert report.thermal_ok is True
    data = json.loads(report.to_json())
    assert "meminfo_kb" in data


def test_run_thor_probe_thermal_throttled(tmp_path: Path) -> None:
    paths = _write_fixture_tree(tmp_path, temp_mc=90_000, power_micro_w=5_000_000)
    report = run_thor_probe(
        ThorProbeConfig(paths=paths, thermal_throttle_c=85.0, power_rail_limit_mw=100_000.0)
    )
    assert report.thermal_ok is False
    assert report.healthy is False


def test_run_thor_probe_power_headroom_violation(tmp_path: Path) -> None:
    paths = _write_fixture_tree(tmp_path, temp_mc=40_000, power_micro_w=50_000_000)
    report = run_thor_probe(
        ThorProbeConfig(
            paths=paths,
            power_rail_limit_mw=60_000.0,
            power_headroom_min_pct=30.0,
        )
    )
    assert report.power_headroom_ok is False
    assert report.healthy is False


def test_setup_thor_dry_run() -> None:
    assert SETUP_SCRIPT.is_file()
    proc = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "MARKER: PIB_PRO_SETUP_OK" in out
    assert "DRY-RUN" in out
    assert "setup-thor complete" in out
