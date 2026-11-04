"""Non-blocking 10 Hz telemetry bridge for Cerebra UI and pib.Cloud."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pib_pro.telemetry_gateway.snapshot import TelemetrySnapshot
from pib_pro.world_state_bus.schema import MAX_OBJECTS

if TYPE_CHECKING:
    from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer

TELEMETRY_HZ: float = 10.0
TELEMETRY_PERIOD_S: float = 1.0 / TELEMETRY_HZ
BUS_HZ: float = 30.0


@dataclass(frozen=True, slots=True)
class TelemetryPayload:
    stamp_monotonic_s: float
    world_state_summary: dict[str, Any]
    scene_markers_rep103: tuple[dict[str, Any], ...]
    chest_oak4_status: dict[str, Any]
    seeed_mic_status: dict[str, Any]
    active_bt_node_id: int
    surprise_scalar: float
    system_health: dict[str, Any]


@dataclass(slots=True)
class NonBlockingTelemetryGateway:
    """Downsample 30 Hz bus snapshots to 10 Hz JSON without writer lock contention."""

    bus: WorldStateRingBuffer
    cpu_util_pct: float = 12.0
    gpu_util_pct: float = 34.0
    thermal_max_c: float = 58.0
    power_headroom_pct: float = 72.0
    _last_publish_monotonic_s: float = field(default=-1.0, init=False)
    _last_payload: TelemetryPayload | None = field(default=None, init=False)

    def _scene_markers(self, slot: np.ndarray) -> tuple[dict[str, Any], ...]:
        markers: list[dict[str, Any]] = []
        objs = slot["objects"]
        for i in range(MAX_OBJECTS):
            oid = int(objs[i]["object_id"])
            if oid == 0:
                continue
            pos = objs[i]["position_xyz"]
            markers.append(
                {
                    "object_id": oid,
                    "class_id": int(objs[i]["class_id"]),
                    "position_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "confidence": float(objs[i]["confidence"]),
                }
            )
        return tuple(markers)

    def _build_payload(self, stamp_monotonic_s: float, slot: np.ndarray) -> TelemetryPayload:
        summary = {
            "human_present": bool(slot["human_present"]),
            "joint_attention_object_id": int(slot["joint_attention_object_id"]),
            "frame_id": int(slot["frame_id"]),
            "sequence_id": int(slot["sequence_id"]),
            "timestamp_ns": int(slot["timestamp_ns"]),
            "object_count": len(self._scene_markers(slot)),
        }
        mic = {
            "vad_active": bool(slot["vad_active"]),
            "doa_azimuth_rad": float(slot["speaker_doa_azimuth_rad"]),
            "model": "Seeed ReSpeaker Mic Array v2.0 (XVF3000)",
        }
        camera = {
            "frame": "chest_camera_oak4_optical",
            "streaming": True,
            "depth_ok": True,
        }
        health = {
            "cpu_util_pct": self.cpu_util_pct,
            "gpu_util_pct": self.gpu_util_pct,
            "thermal_max_c": self.thermal_max_c,
            "power_headroom_pct": self.power_headroom_pct,
            "bus_hz": BUS_HZ,
            "telemetry_hz": TELEMETRY_HZ,
        }
        return TelemetryPayload(
            stamp_monotonic_s=stamp_monotonic_s,
            world_state_summary=summary,
            scene_markers_rep103=self._scene_markers(slot),
            chest_oak4_status=camera,
            seeed_mic_status=mic,
            active_bt_node_id=int(slot["active_bt_node_id"]),
            surprise_scalar=float(slot["surprise_scalar"]),
            system_health=health,
        )

    def should_publish(self, stamp_monotonic_s: float) -> bool:
        if self._last_publish_monotonic_s < 0:
            return True
        return (stamp_monotonic_s - self._last_publish_monotonic_s) >= TELEMETRY_PERIOD_S - 1e-6

    def collect(self, stamp_monotonic_s: float) -> TelemetrySnapshot:
        latest = self.bus.read_latest()
        bus_hz = BUS_HZ if latest is not None else 0.0
        return TelemetrySnapshot(
            stamp_monotonic_s=stamp_monotonic_s,
            cpu_util_pct=self.cpu_util_pct,
            gpu_util_pct=self.gpu_util_pct,
            bus_hz=bus_hz,
            layer0_hz=250.0,
            thermal_max_c=self.thermal_max_c,
            power_headroom_pct=self.power_headroom_pct,
        )

    def sample_payload(self, stamp_monotonic_s: float) -> TelemetryPayload | None:
        """Read latest slot via seqlock (no writer lock) and emit at most 10 Hz."""
        if not self.should_publish(stamp_monotonic_s):
            return self._last_payload
        latest = self.bus.read_latest()
        if latest is None:
            return None
        payload = self._build_payload(stamp_monotonic_s, latest.slot)
        self._last_publish_monotonic_s = stamp_monotonic_s
        self._last_payload = payload
        return payload

    def publish_json(self, snapshot: TelemetrySnapshot) -> str:
        latest = self.bus.read_latest()
        if latest is None:
            return json.dumps(
                {
                    "stamp_monotonic_s": snapshot.stamp_monotonic_s,
                    "cpu_util_pct": snapshot.cpu_util_pct,
                    "gpu_util_pct": snapshot.gpu_util_pct,
                }
            )
        payload = self._build_payload(snapshot.stamp_monotonic_s, latest.slot)
        return json.dumps(
            {
                "stamp_monotonic_s": payload.stamp_monotonic_s,
                "world_state_summary": payload.world_state_summary,
                "scene_markers_rep103": payload.scene_markers_rep103,
                "chest_oak4_status": payload.chest_oak4_status,
                "seeed_mic_status": payload.seeed_mic_status,
                "active_bt_node_id": payload.active_bt_node_id,
                "surprise_scalar": payload.surprise_scalar,
                "system_health": payload.system_health,
                "legacy_snapshot": {
                    "cpu_util_pct": snapshot.cpu_util_pct,
                    "gpu_util_pct": snapshot.gpu_util_pct,
                    "bus_hz": snapshot.bus_hz,
                    "layer0_hz": snapshot.layer0_hz,
                    "thermal_max_c": snapshot.thermal_max_c,
                    "power_headroom_pct": snapshot.power_headroom_pct,
                },
            },
            sort_keys=True,
        )

    def run_10hz_tick(self) -> TelemetryPayload | None:
        return self.sample_payload(time.monotonic())
