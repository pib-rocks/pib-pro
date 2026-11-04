"""Lock-free seqlock ring buffer for the unified 30 Hz world-state bus."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np

from pib_pro.world_state_bus.schema import (
    BUS_PERIOD_NS,
    CONTROL_DTYPE,
    DEFAULT_CAPACITY,
    SLOT_DTYPE,
    WorldStateSlotView,
    empty_slot_array,
    interpolate_slots,
    ring_buffer_byte_size,
    slot_data_offset,
    slot_seqlock_offset,
)

if TYPE_CHECKING:
    from types import TracebackType


_MAX_SEQLOCK_RETRIES = 10_000


@dataclass(frozen=True, slots=True)
class SlotReadResult:
    """Consistent snapshot of one ring slot (may be a private copy)."""

    index: int
    slot: np.ndarray
    torn: bool = False

    @property
    def view(self) -> WorldStateSlotView:
        return WorldStateSlotView.wrap(self.slot)


class WorldStateRingBuffer:
    """900-slot (default) unified-memory ring with per-slot seqlocks."""

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        *,
        shared_memory_name: str | None = None,
        create: bool = True,
    ) -> None:
        if capacity <= 0:
            msg = "capacity must be positive"
            raise ValueError(msg)
        self._capacity = capacity
        self._shm_name = shared_memory_name
        self._owns_shm = False
        self._shm: shared_memory.SharedMemory | None = None
        self._buf: bytearray | memoryview
        nbytes = ring_buffer_byte_size(capacity)
        if shared_memory_name is None:
            self._buf = bytearray(nbytes)
        else:
            if create:
                try:
                    self._shm = shared_memory.SharedMemory(
                        name=shared_memory_name,
                        create=True,
                        size=nbytes,
                    )
                except FileExistsError:
                    self._shm = shared_memory.SharedMemory(
                        name=shared_memory_name,
                        create=False,
                    )
                self._owns_shm = create
            else:
                self._shm = shared_memory.SharedMemory(
                    name=shared_memory_name,
                    create=False,
                )
            assert self._shm.buf is not None
            self._buf = self._shm.buf
        self._control = np.ndarray(1, dtype=CONTROL_DTYPE, buffer=self._buf, offset=0)
        self._seqlocks = np.ndarray(
            capacity,
            dtype=np.uint64,
            buffer=self._buf,
            offset=slot_seqlock_offset(capacity),
        )
        self._slots = np.ndarray(
            capacity,
            dtype=SLOT_DTYPE,
            buffer=self._buf,
            offset=slot_data_offset(capacity),
        )
        if create:
            self._control[...] = 0
            self._seqlocks[...] = 0
            self._slots[...] = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def shared_memory_name(self) -> str | None:
        return self._shm_name

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()

    def unlink(self) -> None:
        if self._shm is not None and self._owns_shm:
            self._shm.unlink()

    def __enter__(self) -> WorldStateRingBuffer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _write_cursor(self) -> int:
        return int(self._control["write_cursor"][0])

    def _total_commits(self) -> int:
        return int(self._control["total_commits"][0])

    def _begin_write(self) -> tuple[int, np.ndarray]:
        total = self._total_commits()
        idx = total % self._capacity
        seq = int(self._seqlocks[idx])
        self._seqlocks[idx] = np.uint64(seq + 1)
        return idx, self._slots[idx]

    def _commit_write(self, idx: int, sequence_id: int | None = None) -> None:
        slot = self._slots[idx]
        if sequence_id is not None:
            slot["sequence_id"] = sequence_id
        elif int(slot["sequence_id"]) == 0:
            slot["sequence_id"] = self._total_commits() + 1
        seq = int(self._seqlocks[idx])
        self._seqlocks[idx] = np.uint64(seq + 1)
        self._control["write_cursor"][0] = idx
        self._control["total_commits"][0] = self._total_commits() + 1

    @contextlib.contextmanager
    def acquire_write_view(self) -> Iterator[WorldStateSlotView]:
        """Zero-copy in-place slot population with seqlock-protected commit."""
        idx, slot = self._begin_write()
        view = WorldStateSlotView.wrap(slot)
        try:
            yield view
        finally:
            self._commit_write(idx)

    def write_slot(self, slot: np.ndarray | WorldStateSlotView) -> int:
        """Copy payload into the next ring slot and commit."""
        with self.acquire_write_view() as view:
            view.fill_from(slot)
        return self._write_cursor()

    def _read_slot_at_index(self, idx: int, *, copy: bool = True) -> SlotReadResult:
        for _ in range(_MAX_SEQLOCK_RETRIES):
            seq1 = int(self._seqlocks[idx])
            if seq1 & 1:
                continue
            slot_view = self._slots[idx]
            data = slot_view.copy() if copy else slot_view
            seq2 = int(self._seqlocks[idx])
            if seq1 == seq2 and not (seq2 & 1):
                return SlotReadResult(index=idx, slot=data, torn=False)
        torn_copy = self._slots[idx].copy()
        return SlotReadResult(index=idx, slot=torn_copy, torn=True)

    def read_latest(self) -> SlotReadResult | None:
        """O(1) read of the most recently committed slot."""
        total = self._total_commits()
        if total == 0:
            return None
        idx = self._write_cursor()
        return self._read_slot_at_index(idx)

    def _occupied_indices_newest_first(self) -> list[int]:
        total = self._total_commits()
        if total == 0:
            return []
        count = min(total, self._capacity)
        cursor = self._write_cursor()
        indices: list[int] = []
        for offset in range(count):
            indices.append((cursor - offset) % self._capacity)
        return indices

    def read_window(self, duration_ns: int) -> list[SlotReadResult]:
        """Return recent slots with timestamp_ns within duration_ns of latest."""
        latest = self.read_latest()
        if latest is None:
            return []
        latest_ts = int(latest.slot["timestamp_ns"])
        cutoff = latest_ts - duration_ns
        results: list[SlotReadResult] = []
        for idx in self._occupied_indices_newest_first():
            snap = self._read_slot_at_index(idx)
            if snap.torn:
                continue
            ts = int(snap.slot["timestamp_ns"])
            if ts < cutoff:
                break
            results.append(snap)
        results.reverse()
        return results

    def read_at_timestamp(
        self,
        timestamp_ns: int,
        *,
        interpolate: bool = True,
    ) -> SlotReadResult | None:
        """Bracketing lookup with optional linear interpolation of continuous fields."""
        indices = self._occupied_indices_newest_first()
        if not indices:
            return None
        snaps: list[SlotReadResult] = []
        for idx in indices:
            snap = self._read_slot_at_index(idx)
            if not snap.torn:
                snaps.append(snap)
        if not snaps:
            return None
        snaps.sort(key=lambda s: int(s.slot["timestamp_ns"]))
        if timestamp_ns <= int(snaps[0].slot["timestamp_ns"]):
            return snaps[0]
        if timestamp_ns >= int(snaps[-1].slot["timestamp_ns"]):
            return snaps[-1]
        before: SlotReadResult | None = None
        after: SlotReadResult | None = None
        for snap in snaps:
            ts = int(snap.slot["timestamp_ns"])
            if ts <= timestamp_ns:
                before = snap
            if ts >= timestamp_ns and after is None:
                after = snap
                break
        if before is None or after is None:
            return snaps[-1]
        if not interpolate or before.index == after.index:
            return before
        t0 = int(before.slot["timestamp_ns"])
        t1 = int(after.slot["timestamp_ns"])
        if t1 <= t0:
            return before
        alpha = (timestamp_ns - t0) / (t1 - t0)
        out = empty_slot_array(1)[0]
        interpolate_slots(before.slot, after.slot, alpha, out=out)
        return SlotReadResult(index=before.index, slot=out, torn=False)

    def write_demo_frame(
        self,
        frame_index: int,
        *,
        timestamp_ns: int | None = None,
    ) -> int:
        """Test helper: populate one slot with deterministic content."""
        ts = timestamp_ns if timestamp_ns is not None else frame_index * BUS_PERIOD_NS
        with self.acquire_write_view() as view:
            view.sequence_id = frame_index + 1
            view.timestamp_ns = ts
            view.frame_id = frame_index
            view.objects[0]["position_xyz"] = np.array(
                [float(frame_index), 0.0, 0.0],
                dtype=np.float32,
            )
            view.joint_positions_rad[0] = np.float32(frame_index * 0.01)
            view.z_t[0] = np.float32(frame_index)
            view.surprise_scalar = float(frame_index * 0.1)
        return self._write_cursor()
