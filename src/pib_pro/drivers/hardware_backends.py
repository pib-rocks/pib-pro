"""Linux USB/serial auto-discovery for Feetech STS, Seeed ReSpeaker, and Chest OAK-4."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_FEETECH_BY_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"usb-1a86_", re.IGNORECASE),
    re.compile(r"usb-FTDI_", re.IGNORECASE),
    re.compile(r"usb-Silicon_Labs_", re.IGNORECASE),
    re.compile(r"ttyACM", re.IGNORECASE),
    re.compile(r"ttyUSB", re.IGNORECASE),
)

_LEFT_ARM_HEAD_ID_MIN: int = 18
_LEFT_ARM_HEAD_ID_MAX: int = 45

# Permanent assignment cache: keyed by by-id symlink name (stable across re-probes).
_feetech_assignment_cache: dict[str, str] = {}


def _feetech_symlink_matches(name: str) -> bool:
    return any(p.search(name) for p in _FEETECH_BY_ID_PATTERNS)


def _resolve_serial_path(by_id_dir: Path, entry_name: str) -> str | None:
    link = by_id_dir / entry_name
    try:
        if not link.is_symlink():
            return None
        target = link.resolve()
    except OSError:
        return None
    return str(target)


def _classify_feetech_group(responding_servo_ids: Sequence[int]) -> str:
    ids = tuple(int(i) for i in responding_servo_ids)
    if not ids:
        return "generic"
    if any(_LEFT_ARM_HEAD_ID_MIN <= sid <= _LEFT_ARM_HEAD_ID_MAX for sid in ids):
        return "left_arm_and_head"
    if all(1 <= sid <= 17 for sid in ids):
        return "right_arm"
    return "generic"


def _default_feetech_probe(_device_path: str) -> tuple[int, ...]:
    """Non-blocking default: no serial I/O in CI or headless hosts."""
    return ()


def _prune_feetech_cache(active_symlink_names: set[str]) -> None:
    stale = [k for k in _feetech_assignment_cache if k not in active_symlink_names]
    for k in stale:
        del _feetech_assignment_cache[k]


@dataclass(frozen=True, slots=True)
class DiscoveredFeetechPort:
    path: str
    serial_id: str
    responding_servo_ids: tuple[int, ...]
    assigned_group: str


def discover_feetech_serial_ports(
    by_id_dir: Path = Path("/dev/serial/by-id"),
    probe_fn: Callable[[str], Sequence[int]] | None = None,
) -> tuple[DiscoveredFeetechPort, ...]:
    """Scan ``by_id_dir`` for Feetech USB serial adapters and classify servo buses."""
    probe = probe_fn if probe_fn is not None else _default_feetech_probe
    if not by_id_dir.is_dir():
        _prune_feetech_cache(set())
        return ()

    entries = sorted(by_id_dir.iterdir(), key=lambda p: p.name)
    active_names: set[str] = set()
    discovered: list[DiscoveredFeetechPort] = []

    for entry in entries:
        if not _feetech_symlink_matches(entry.name):
            continue
        device_path = _resolve_serial_path(by_id_dir, entry.name)
        if device_path is None:
            continue
        active_names.add(entry.name)
        responding = tuple(int(x) for x in probe(device_path))
        if entry.name in _feetech_assignment_cache:
            assigned = _feetech_assignment_cache[entry.name]
        else:
            assigned = _classify_feetech_group(responding)
            _feetech_assignment_cache[entry.name] = assigned
        discovered.append(
            DiscoveredFeetechPort(
                path=device_path,
                serial_id=entry.name,
                responding_servo_ids=responding,
                assigned_group=assigned,
            ),
        )

    _prune_feetech_cache(active_names)
    return tuple(discovered)


@dataclass(frozen=True, slots=True)
class DiscoveredSeeedMicArray:
    usb_vid: int
    usb_pid: int
    usb_bus_dev: str
    alsa_card_index: int | None
    is_present: bool


_RESPEAKER_VID: int = 0x2886
_RESPEAKER_PID: int = 0x0018


def _read_sysfs_id(usb_device_dir: Path, filename: str) -> str | None:
    path = usb_device_dir / filename
    try:
        return path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None


def _parse_asound_respeaker_card(text: str) -> int | None:
    for line in text.splitlines():
        if "ReSpeaker" not in line and "ArrayUAC10" not in line:
            continue
        m = re.match(r"^\s*(\d+)\s+", line)
        if m:
            return int(m.group(1))
    return None


def discover_seeed_respeaker_usb(
    sysfs_usb_root: Path = Path("/sys/bus/usb/devices"),
    proc_asound_cards: Path = Path("/proc/asound/cards"),
) -> DiscoveredSeeedMicArray:
    """Locate Seeed ReSpeaker Mic Array v2.0 (2886:0018) via sysfs and ALSA."""
    usb_bus_dev = ""
    alsa_index: int | None = None
    found_usb = False

    if sysfs_usb_root.is_dir():
        for dev_dir in sorted(sysfs_usb_root.iterdir()):
            if not dev_dir.is_dir():
                continue
            vendor = _read_sysfs_id(dev_dir, "idVendor")
            product = _read_sysfs_id(dev_dir, "idProduct")
            if vendor != f"{_RESPEAKER_VID:04x}" or product != f"{_RESPEAKER_PID:04x}":
                continue
            found_usb = True
            usb_bus_dev = dev_dir.name
            break

    if proc_asound_cards.is_file():
        try:
            cards_text = proc_asound_cards.read_text(encoding="utf-8", errors="replace")
            alsa_index = _parse_asound_respeaker_card(cards_text)
        except OSError:
            alsa_index = None

    is_present = found_usb or alsa_index is not None
    return DiscoveredSeeedMicArray(
        usb_vid=_RESPEAKER_VID,
        usb_pid=_RESPEAKER_PID,
        usb_bus_dev=usb_bus_dev,
        alsa_card_index=alsa_index,
        is_present=is_present,
    )


@dataclass(frozen=True, slots=True)
class ChestOak4HardwareStatus:
    depthai_available: bool
    device_connected: bool
    mxid: str
    mode: Literal["live_depthai", "simulated"]


def probe_chest_oak4_hardware() -> ChestOak4HardwareStatus:
    """Probe DepthAI for a connected OAK device without requiring root."""
    try:
        import depthai as dai
    except ImportError:
        return ChestOak4HardwareStatus(
            depthai_available=False,
            device_connected=False,
            mxid="",
            mode="simulated",
        )

    try:
        devices = dai.Device.getAllAvailableDevices()
    except Exception:
        return ChestOak4HardwareStatus(
            depthai_available=True,
            device_connected=False,
            mxid="",
            mode="simulated",
        )

    if not devices:
        return ChestOak4HardwareStatus(
            depthai_available=True,
            device_connected=False,
            mxid="",
            mode="simulated",
        )

    first = devices[0]
    mxid = ""
    try:
        mxid = str(getattr(first, "getMxId", lambda: "")() or getattr(first, "mxid", ""))
    except Exception:
        mxid = ""
    return ChestOak4HardwareStatus(
        depthai_available=True,
        device_connected=True,
        mxid=mxid,
        mode="live_depthai",
    )


def reset_feetech_discovery_cache_for_tests() -> None:
    """Clear cached Feetech group assignments (test helper)."""
    _feetech_assignment_cache.clear()
