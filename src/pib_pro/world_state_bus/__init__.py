"""World-state bus slot and ring-buffer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pib_pro.common import Pose3D


@dataclass(frozen=True, slots=True)
class WorldStateSlotContract:
    """Metadata for a single typed slot in the unified world-state ring."""

    slot_id: str
    schema_version: int
    frame_id: str
    max_staleness_s: float


class WorldStateBusProtocol(Protocol):
    """30 Hz unified-memory bus: publish/consume typed slots."""

    def register_slot(self, contract: WorldStateSlotContract) -> None: ...

    def write_slot(self, slot_id: str, payload: object, stamp_monotonic_s: float) -> None: ...

    def read_latest(self, slot_id: str) -> tuple[object, float] | None: ...

    def snapshot_pose(self) -> Pose3D: ...
