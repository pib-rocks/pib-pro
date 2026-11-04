"""Deterministic record and replay of world-state bus traces."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from pib_pro.world_state_bus.ring_buffer import SlotReadResult, WorldStateRingBuffer
from pib_pro.world_state_bus.schema import (
    DEFAULT_CAPACITY,
    SLOT_DTYPE,
    as_slot_scalar,
    bytes_from_slot,
    slot_from_bytes,
)

_WSB_MAGIC = b"WSB1"
_WSB_VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sIIQ")


def _encode_header(capacity: int, slot_count: int) -> bytes:
    return _HEADER_STRUCT.pack(_WSB_MAGIC, _WSB_VERSION, capacity, slot_count)


def _decode_header(data: bytes) -> tuple[int, int]:
    magic, version, capacity, slot_count = _HEADER_STRUCT.unpack_from(data, 0)
    if magic != _WSB_MAGIC:
        msg = "invalid WSB magic"
        raise ValueError(msg)
    if version != _WSB_VERSION:
        msg = f"unsupported WSB version {version}"
        raise ValueError(msg)
    return capacity, slot_count


class WorldStateRecorder:
    """Capture ring-buffer slots into a compact binary trace."""

    def __init__(self) -> None:
        self._slots: list[bytes] = []
        self._capacity: int | None = None

    def record_slot(self, slot: np.ndarray) -> None:
        self._slots.append(bytes_from_slot(slot))

    def record_from_ring(
        self,
        ring: WorldStateRingBuffer,
        *,
        duration_ns: int | None = None,
        max_slots: int | None = None,
    ) -> None:
        self._capacity = ring.capacity
        if duration_ns is not None:
            for result in ring.read_window(duration_ns):
                self.record_slot(result.slot)
            return
        if max_slots is not None:
            indices = ring._occupied_indices_newest_first()  # noqa: SLF001
            for idx in reversed(indices[-max_slots:]):
                snap = ring._read_slot_at_index(idx)  # noqa: SLF001
                if not snap.torn:
                    self.record_slot(snap.slot)
            return
        latest = ring.read_latest()
        if latest is not None:
            self.record_slot(latest.slot)

    def to_bytes(self, *, capacity: int = DEFAULT_CAPACITY) -> bytes:
        cap = self._capacity if self._capacity is not None else capacity
        header = _encode_header(cap, len(self._slots))
        payload = b"".join(self._slots)
        return header + payload

    def save(self, path: str | Path, *, capacity: int = DEFAULT_CAPACITY) -> None:
        path = Path(path)
        data = self.to_bytes(capacity=capacity)
        if path.suffix == ".npz":
            slots_arr = self.slots_array()
            np.savez_compressed(
                path,
                format="wsb",
                capacity=cap if (cap := self._capacity) else capacity,
                slots=slots_arr,
            )
            return
        path.write_bytes(data)

    def slots_array(self) -> np.ndarray:
        if not self._slots:
            return np.zeros(0, dtype=SLOT_DTYPE)
        return np.array([slot_from_bytes(raw) for raw in self._slots])

    @classmethod
    def from_bytes(cls, data: bytes) -> WorldStateRecorder:
        capacity, slot_count = _decode_header(data)
        offset = _HEADER_STRUCT.size
        expected = offset + slot_count * SLOT_DTYPE.itemsize
        if len(data) != expected:
            msg = "WSB payload size mismatch"
            raise ValueError(msg)
        rec = cls()
        rec._capacity = capacity
        for i in range(slot_count):
            start = offset + i * SLOT_DTYPE.itemsize
            end = start + SLOT_DTYPE.itemsize
            rec._slots.append(bytes(data[start:end]))
        return rec

    @classmethod
    def load(cls, path: str | Path) -> WorldStateRecorder:
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as npz:
                slots = npz["slots"]
                rec = cls()
                rec._capacity = int(npz["capacity"])
                for row in slots:
                    rec._slots.append(bytes_from_slot(as_slot_scalar(row)))
                return rec
        return cls.from_bytes(path.read_bytes())


class WorldStatePlayer:
    """Replay a recorded trace into any WorldStateRingBuffer."""

    def __init__(self, recorder: WorldStateRecorder) -> None:
        self._recorder = recorder

    @classmethod
    def from_path(cls, path: str | Path) -> WorldStatePlayer:
        return cls(WorldStateRecorder.load(path))

    def replay(self, ring: WorldStateRingBuffer, *, realtime: bool = False) -> list[SlotReadResult]:
        """Write each recorded slot in order; returns consistent read snapshots."""
        results: list[SlotReadResult] = []
        prev_ts: int | None = None
        for raw in self._recorder._slots:  # noqa: SLF001
            slot = slot_from_bytes(raw)
            if realtime and prev_ts is not None:
                delay_s = (int(slot["timestamp_ns"]) - prev_ts) / 1_000_000_000
                if delay_s > 0:
                    import time

                    time.sleep(delay_s)
            ring.write_slot(slot)
            snap = ring.read_latest()
            if snap is not None:
                results.append(snap)
            prev_ts = int(slot["timestamp_ns"])
        return results

    def replay_bytes(self, ring: WorldStateRingBuffer) -> list[bytes]:
        """Replay and return byte-identical payloads as read from the ring."""
        out: list[bytes] = []
        for raw in self._recorder._slots:  # noqa: SLF001
            ring.write_slot(slot_from_bytes(raw))
            snap = ring.read_latest()
            if snap is not None:
                out.append(bytes_from_slot(snap.slot))
        return out
