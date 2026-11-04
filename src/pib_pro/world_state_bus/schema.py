"""Structured NumPy schema and typed views for the 30 Hz world-state bus slot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_CAPACITY: int = 900
BUS_HZ: float = 30.0
BUS_PERIOD_NS: int = int(1_000_000_000 // 30)
MAX_OBJECTS: int = 32
MAX_JOINTS: int = 32
LATENT_DIM: int = 256
SURPRISE_REGION_COUNT: int = 8

INTERPOLATABLE_SCALAR_FIELDS: tuple[str, ...] = ("surprise_scalar",)
INTERPOLATABLE_VECTOR_FIELDS: tuple[str, ...] = (
    "objects.position_xyz",
    "joint_positions_rad",
    "z_t",
)

OBJECT_DTYPE = np.dtype(
    [
        ("object_id", np.uint64),
        ("class_id", np.uint32),
        ("position_xyz", np.float32, (3,)),
        ("orientation_xyzw", np.float32, (4,)),
        ("velocity_xyz", np.float32, (3,)),
        ("confidence", np.float32),
        ("last_seen_ns", np.int64),
        ("staleness_sec", np.float32),
        ("joint_attention_prob", np.float32),
    ]
)

SLOT_DTYPE = np.dtype(
    [
        ("sequence_id", np.uint64),
        ("timestamp_ns", np.int64),
        ("frame_id", np.uint64),
        ("objects", OBJECT_DTYPE, (MAX_OBJECTS,)),
        ("human_present", np.bool_),
        ("head_position_xyz", np.float32, (3,)),
        ("gaze_ray_origin_xyz", np.float32, (3,)),
        ("gaze_ray_dir_xyz", np.float32, (3,)),
        ("pointing_ray_origin_xyz", np.float32, (3,)),
        ("pointing_ray_dir_xyz", np.float32, (3,)),
        ("left_hand_xyz", np.float32, (3,)),
        ("left_hand_vel_xyz", np.float32, (3,)),
        ("right_hand_xyz", np.float32, (3,)),
        ("right_hand_vel_xyz", np.float32, (3,)),
        ("speaker_doa_azimuth_rad", np.float32),
        ("vad_active", np.bool_),
        ("joint_positions_rad", np.float32, (MAX_JOINTS,)),
        ("joint_velocities_rad_s", np.float32, (MAX_JOINTS,)),
        ("joint_loads_nm_or_ma", np.float32, (MAX_JOINTS,)),
        ("external_contact_flags", np.uint32, (MAX_JOINTS,)),
        ("z_t", np.float32, (LATENT_DIM,)),
        ("z_hat_self", np.float32, (LATENT_DIM,)),
        ("z_hat_human", np.float32, (LATENT_DIM,)),
        ("z_goal", np.float32, (LATENT_DIM,)),
        ("surprise_scalar", np.float32),
        ("surprise_by_region", np.float32, (SURPRISE_REGION_COUNT,)),
        ("floor_state", np.uint8),
        ("proactivity_tier", np.uint8),
        ("active_bt_node_id", np.uint32),
        ("joint_attention_object_id", np.int32),
    ]
)

CONTROL_DTYPE = np.dtype(
    [
        ("write_cursor", np.uint64),
        ("total_commits", np.uint64),
    ]
)

SLOT_BYTE_SIZE: int = int(SLOT_DTYPE.itemsize)
CONTROL_BYTE_SIZE: int = int(CONTROL_DTYPE.itemsize)


def empty_slot_array(count: int = 1) -> np.ndarray:
    """Allocate zero-initialized slot(s) with the bus structured dtype."""
    return np.zeros(count, dtype=SLOT_DTYPE)


def slot_seqlock_offset(capacity: int) -> int:
    """Byte offset where per-slot seqlock counters begin."""
    return CONTROL_BYTE_SIZE


def slot_data_offset(capacity: int) -> int:
    """Byte offset where slot payload ring begins."""
    return slot_seqlock_offset(capacity) + capacity * np.dtype(np.uint64).itemsize


def ring_buffer_byte_size(capacity: int) -> int:
    """Total bytes for control, seqlocks, and slot ring."""
    return slot_data_offset(capacity) + capacity * SLOT_BYTE_SIZE


@dataclass(slots=True)
class WorldStateSlotView:
    """High-level read/write facade over one structured slot element."""

    _data: np.ndarray

    @classmethod
    def wrap(cls, slot: np.ndarray) -> WorldStateSlotView:
        if slot.dtype != SLOT_DTYPE:
            msg = f"expected dtype {SLOT_DTYPE}, got {slot.dtype}"
            raise TypeError(msg)
        return cls(slot)

    @property
    def raw(self) -> np.ndarray:
        return self._data

    @property
    def sequence_id(self) -> int:
        return int(self._data["sequence_id"])

    @sequence_id.setter
    def sequence_id(self, value: int) -> None:
        self._data["sequence_id"] = value

    @property
    def timestamp_ns(self) -> int:
        return int(self._data["timestamp_ns"])

    @timestamp_ns.setter
    def timestamp_ns(self, value: int) -> None:
        self._data["timestamp_ns"] = value

    @property
    def frame_id(self) -> int:
        return int(self._data["frame_id"])

    @frame_id.setter
    def frame_id(self, value: int) -> None:
        self._data["frame_id"] = value

    @property
    def objects(self) -> np.ndarray:
        return self._data["objects"]

    @property
    def human_present(self) -> bool:
        return bool(self._data["human_present"])

    @human_present.setter
    def human_present(self, value: bool) -> None:
        self._data["human_present"] = value

    @property
    def joint_positions_rad(self) -> np.ndarray:
        return self._data["joint_positions_rad"]

    @property
    def z_t(self) -> np.ndarray:
        return self._data["z_t"]

    @property
    def surprise_scalar(self) -> float:
        return float(self._data["surprise_scalar"])

    @surprise_scalar.setter
    def surprise_scalar(self, value: float) -> None:
        self._data["surprise_scalar"] = np.float32(value)

    def fill_from(self, other: np.ndarray | WorldStateSlotView) -> None:
        """Copy all fields from another slot payload."""
        src = other.raw if isinstance(other, WorldStateSlotView) else as_slot_scalar(other)
        for name in SLOT_DTYPE.names or ():
            self._data[name] = src[name]


def bytes_from_slot(slot: np.ndarray) -> bytes:
    """Return a byte-identical copy of one slot element."""
    return slot.tobytes()


def slot_from_bytes(data: bytes) -> np.ndarray:
    """Deserialize one slot scalar from raw bytes."""
    arr = np.frombuffer(data, dtype=SLOT_DTYPE, count=1)
    res: np.ndarray = np.asarray(arr[0]).copy()
    return res


def as_slot_scalar(slot: np.ndarray) -> np.ndarray:
    """Normalize a structured slot to a 0-d element."""
    if slot.dtype != SLOT_DTYPE:
        msg = f"expected dtype {SLOT_DTYPE}, got {slot.dtype}"
        raise TypeError(msg)
    if slot.shape == ():
        return slot
    if slot.shape == (1,):
        res: np.ndarray = np.asarray(slot[0])
        return res
    msg = f"expected scalar slot, got shape {slot.shape}"
    raise ValueError(msg)


def interpolate_slots(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Linearly interpolate continuous fields between two slot snapshots."""
    if out is None:
        out = empty_slot_array(1)[0]
    else:
        for name in SLOT_DTYPE.names or ():
            out[name] = before[name]
    a = np.float32(alpha)
    one_minus = np.float32(1.0 - alpha)
    out["objects"]["position_xyz"] = (
        before["objects"]["position_xyz"] * one_minus + after["objects"]["position_xyz"] * a
    )
    out["joint_positions_rad"] = (
        before["joint_positions_rad"] * one_minus + after["joint_positions_rad"] * a
    )
    out["z_t"] = before["z_t"] * one_minus + after["z_t"] * a
    out["surprise_scalar"] = np.float32(
        float(before["surprise_scalar"]) * (1.0 - alpha) + float(after["surprise_scalar"]) * alpha
    )
    ts = int(before["timestamp_ns"] + alpha * (after["timestamp_ns"] - before["timestamp_ns"]))
    out["timestamp_ns"] = ts
    out["sequence_id"] = before["sequence_id"]
    out["frame_id"] = before["frame_id"]
    return out


def slot_field_path_get(slot: np.ndarray, path: str) -> Any:
    """Resolve a dotted field path on a structured slot (for tests and tooling)."""
    parts = path.split(".")
    value: Any = slot
    for part in parts:
        value = value[part]
    return value
