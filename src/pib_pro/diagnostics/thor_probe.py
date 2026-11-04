"""Jetson AGX Thor power, thermal, clock, and memory probe."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ThorProbePaths:
    """Injectable sysfs/proc paths for CI on non-Thor hosts."""

    meminfo: Path = Path("/proc/meminfo")
    thermal_glob: str = "/sys/class/thermal/thermal_zone*"
    hwmon_glob: str = "/sys/class/hwmon/hwmon*"
    gpu_clock_glob: str = "/sys/class/devfreq/*gpu*/cur_freq"
    jetson_release: Path = Path("/etc/nv_tegra_release")


@dataclass
class ThorProbeConfig:
    power_rail_limit_mw: float = 60_000.0
    power_headroom_min_pct: float = 30.0
    thermal_throttle_c: float = 85.0
    paths: ThorProbePaths = field(default_factory=ThorProbePaths)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_meminfo_kb(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+(?:\(\w+\))?):\s+(\d+)\s+kB", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def read_thermal_zones(base_glob: str) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for zone_dir in sorted(Path("/").glob(base_glob.lstrip("/"))):
        if not zone_dir.is_dir():
            continue
        try:
            temp_raw = _read_text(zone_dir / "temp").strip()
            type_raw = _read_text(zone_dir / "type").strip()
        except OSError:
            continue
        temp_mc = int(temp_raw)
        zones.append(
            {
                "zone": zone_dir.name,
                "type": type_raw,
                "temp_c": temp_mc / 1000.0,
            }
        )
    return zones


def read_hwmon_power_mw(base_glob: str) -> list[dict[str, Any]]:
    readings: list[dict[str, Any]] = []
    for hwmon_dir in sorted(Path("/").glob(base_glob.lstrip("/"))):
        if not hwmon_dir.is_dir():
            continue
        name = hwmon_dir.name
        for power_file in hwmon_dir.glob("power*_input"):
            try:
                micro_w = int(_read_text(power_file).strip())
            except (OSError, ValueError):
                continue
            readings.append(
                {
                    "hwmon": name,
                    "sensor": power_file.name,
                    "power_mw": micro_w / 1000.0,
                }
            )
    return readings


def read_gpu_clocks_hz(base_glob: str) -> list[dict[str, Any]]:
    clocks: list[dict[str, Any]] = []
    for freq_path in sorted(Path("/").glob(base_glob.lstrip("/"))):
        try:
            hz = int(_read_text(freq_path).strip())
        except (OSError, ValueError):
            continue
        clocks.append({"path": str(freq_path), "cur_freq_hz": hz})
    return clocks


def read_l4t_release(path: Path) -> str | None:
    try:
        return _read_text(path).strip()
    except OSError:
        return None


def evaluate_headroom(total_power_mw: float, limit_mw: float, min_pct: float) -> tuple[bool, float]:
    if limit_mw <= 0:
        raise ValueError("power rail limit must be positive")
    headroom_pct = max(0.0, (limit_mw - total_power_mw) / limit_mw * 100.0)
    ok = headroom_pct >= min_pct
    return ok, headroom_pct


def evaluate_thermal(zones: list[dict[str, Any]], throttle_c: float) -> tuple[bool, float]:
    if not zones:
        return True, 0.0
    max_c = max(float(z["temp_c"]) for z in zones)
    ok = max_c < throttle_c
    return ok, max_c


@dataclass(frozen=True, slots=True)
class ThorProbeReport:
    l4t_release: str | None
    meminfo_kb: dict[str, int]
    thermal_zones: list[dict[str, Any]]
    hwmon_power: list[dict[str, Any]]
    gpu_clocks: list[dict[str, Any]]
    total_power_mw: float
    power_headroom_pct: float
    power_headroom_ok: bool
    thermal_max_c: float
    thermal_ok: bool
    healthy: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def run_thor_probe(config: ThorProbeConfig | None = None) -> ThorProbeReport:
    cfg = config or ThorProbeConfig()
    paths = cfg.paths

    mem_text = _read_text(paths.meminfo) if paths.meminfo.exists() else ""
    meminfo = parse_meminfo_kb(mem_text) if mem_text else {}

    zones = read_thermal_zones(paths.thermal_glob)
    hwmon = read_hwmon_power_mw(paths.hwmon_glob)
    clocks = read_gpu_clocks_hz(paths.gpu_clock_glob)
    l4t = read_l4t_release(paths.jetson_release)

    total_power = sum(float(r["power_mw"]) for r in hwmon)
    power_ok, headroom = evaluate_headroom(
        total_power, cfg.power_rail_limit_mw, cfg.power_headroom_min_pct
    )
    thermal_ok, thermal_max = evaluate_thermal(zones, cfg.thermal_throttle_c)
    healthy = power_ok and thermal_ok

    return ThorProbeReport(
        l4t_release=l4t,
        meminfo_kb=meminfo,
        thermal_zones=zones,
        hwmon_power=hwmon,
        gpu_clocks=clocks,
        total_power_mw=total_power,
        power_headroom_pct=headroom,
        power_headroom_ok=power_ok,
        thermal_max_c=thermal_max,
        thermal_ok=thermal_ok,
        healthy=healthy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jetson AGX Thor hardware probe")
    parser.add_argument(
        "--power-limit-mw",
        type=float,
        default=60_000.0,
        help="Configurable rail power limit in milliwatts",
    )
    parser.add_argument(
        "--headroom-min-pct",
        type=float,
        default=30.0,
        help="Minimum required power headroom percentage",
    )
    parser.add_argument(
        "--thermal-throttle-c",
        type=float,
        default=85.0,
        help="Thermal throttling threshold in Celsius",
    )
    args = parser.parse_args(argv)
    report = run_thor_probe(
        ThorProbeConfig(
            power_rail_limit_mw=args.power_limit_mw,
            power_headroom_min_pct=args.headroom_min_pct,
            thermal_throttle_c=args.thermal_throttle_c,
        )
    )
    print(report.to_json())
    return 0 if report.healthy else 1


if __name__ == "__main__":
    sys.exit(main())
