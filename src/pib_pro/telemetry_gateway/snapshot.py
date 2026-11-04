"""Telemetry snapshot dataclass and export protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    stamp_monotonic_s: float
    cpu_util_pct: float
    gpu_util_pct: float
    bus_hz: float
    layer0_hz: float
    thermal_max_c: float
    power_headroom_pct: float


class TelemetryGatewayProtocol(Protocol):
    def collect(self, stamp_monotonic_s: float) -> TelemetrySnapshot: ...

    def publish_json(self, snapshot: TelemetrySnapshot) -> str: ...
