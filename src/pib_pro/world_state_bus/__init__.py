"""World-state bus slot and ring-buffer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pib_pro.common import Pose3D
from pib_pro.world_state_bus.replay import WorldStatePlayer, WorldStateRecorder
from pib_pro.world_state_bus.ring_buffer import SlotReadResult, WorldStateRingBuffer
from pib_pro.world_state_bus.schema import (
    BUS_HZ,
    BUS_PERIOD_NS,
    DEFAULT_CAPACITY,
    INTERPOLATABLE_SCALAR_FIELDS,
    INTERPOLATABLE_VECTOR_FIELDS,
    LATENT_DIM,
    MAX_JOINTS,
    MAX_OBJECTS,
    OBJECT_DTYPE,
    SLOT_DTYPE,
    SURPRISE_REGION_COUNT,
    WorldStateSlotView,
    as_slot_scalar,
    bytes_from_slot,
    empty_slot_array,
    interpolate_slots,
    ring_buffer_byte_size,
    slot_from_bytes,
)

__all__ = [
    "BUS_HZ",
    "BUS_PERIOD_NS",
    "DEFAULT_CAPACITY",
    "INTERPOLATABLE_SCALAR_FIELDS",
    "INTERPOLATABLE_VECTOR_FIELDS",
    "LATENT_DIM",
    "MAX_JOINTS",
    "MAX_OBJECTS",
    "OBJECT_DTYPE",
    "SLOT_DTYPE",
    "SURPRISE_REGION_COUNT",
    "SlotReadResult",
    "WorldStateBusProtocol",
    "WorldStatePlayer",
    "WorldStateRecorder",
    "WorldStateRingBuffer",
    "WorldStateSlotContract",
    "WorldStateSlotView",
    "as_slot_scalar",
    "bytes_from_slot",
    "empty_slot_array",
    "interpolate_slots",
    "ring_buffer_byte_size",
    "slot_from_bytes",
]


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
