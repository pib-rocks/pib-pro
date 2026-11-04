"""Tests for the 30 Hz unified world-state ring buffer."""

from __future__ import annotations

import contextlib
import statistics
import threading
import time
import uuid
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

from pib_pro.world_state_bus import (
    BUS_PERIOD_NS,
    DEFAULT_CAPACITY,
    LATENT_DIM,
    MAX_JOINTS,
    MAX_OBJECTS,
    SLOT_DTYPE,
    WorldStatePlayer,
    WorldStateRecorder,
    WorldStateRingBuffer,
    empty_slot_array,
    ring_buffer_byte_size,
    slot_from_bytes,
)


def _make_slot(
    sequence_id: int,
    timestamp_ns: int,
    *,
    x: float = 0.0,
    joint0: float = 0.0,
    z0: float = 0.0,
    surprise: float = 0.0,
) -> np.ndarray:
    slot = empty_slot_array(1)[0]
    slot["sequence_id"] = sequence_id
    slot["timestamp_ns"] = timestamp_ns
    slot["frame_id"] = sequence_id
    slot["objects"][0]["position_xyz"] = np.array([x, 0.0, 0.0], dtype=np.float32)
    slot["joint_positions_rad"][0] = np.float32(joint0)
    slot["z_t"][0] = np.float32(z0)
    slot["surprise_scalar"] = np.float32(surprise)
    return slot


def test_schema_constants_and_slot_size() -> None:
    assert DEFAULT_CAPACITY == 900
    assert MAX_OBJECTS == 32
    assert MAX_JOINTS == 32
    assert LATENT_DIM == 256
    assert ring_buffer_byte_size(DEFAULT_CAPACITY) > DEFAULT_CAPACITY * SLOT_DTYPE.itemsize


def test_ring_capacity_and_wrap_around() -> None:
    ring = WorldStateRingBuffer(capacity=900)
    for i in range(1205):
        ring.write_demo_frame(i, timestamp_ns=i * BUS_PERIOD_NS)
    assert ring._total_commits() == 1205  # noqa: SLF001
    latest = ring.read_latest()
    assert latest is not None
    assert int(latest.slot["sequence_id"]) == 1205
    assert int(latest.slot["frame_id"]) == 1204


def test_shared_memory_attach_zero_copy() -> None:
    name = f"pib_wsb_{uuid.uuid4().hex}"
    try:
        creator = WorldStateRingBuffer(
            capacity=64,
            shared_memory_name=name,
            create=True,
        )
        slot = _make_slot(1, 100, x=3.5)
        creator.write_slot(slot)
        attached = WorldStateRingBuffer(
            capacity=64,
            shared_memory_name=name,
            create=False,
        )
        latest = attached.read_latest()
        assert latest is not None
        assert latest.slot["objects"][0]["position_xyz"][0] == pytest.approx(3.5)
        creator.close()
        attached.close()
    finally:
        with contextlib.suppress(FileNotFoundError):
            shared_memory.SharedMemory(name=name).unlink()


def test_read_at_timestamp_interpolation() -> None:
    ring = WorldStateRingBuffer(capacity=32)
    t0 = 1_000_000_000
    t1 = t0 + BUS_PERIOD_NS
    ring.write_slot(_make_slot(1, t0, x=0.0, joint0=0.0, z0=0.0, surprise=0.0))
    ring.write_slot(_make_slot(2, t1, x=10.0, joint0=1.0, z0=100.0, surprise=2.0))
    mid_ts = t0 + BUS_PERIOD_NS // 2
    result = ring.read_at_timestamp(mid_ts, interpolate=True)
    assert result is not None
    assert result.torn is False
    assert result.slot["objects"][0]["position_xyz"][0] == pytest.approx(5.0)
    assert result.slot["joint_positions_rad"][0] == pytest.approx(0.5)
    assert result.slot["z_t"][0] == pytest.approx(50.0)
    assert result.slot["surprise_scalar"] == pytest.approx(1.0)


def test_read_window_returns_recent_history() -> None:
    ring = WorldStateRingBuffer(capacity=16)
    base = 5_000_000_000
    for i in range(10):
        ring.write_slot(_make_slot(i + 1, base + i * BUS_PERIOD_NS, x=float(i)))
    window = ring.read_window(3 * BUS_PERIOD_NS)
    assert len(window) >= 3
    assert int(window[-1].slot["timestamp_ns"]) == base + 9 * BUS_PERIOD_NS


def test_concurrent_multirate_stress_no_torn_reads() -> None:
    ring = WorldStateRingBuffer(capacity=900)
    stop = threading.Event()
    torn_reads = 0
    read_latencies_us: list[float] = []
    write_latencies_us: list[float] = []
    lock = threading.Lock()

    def writer() -> None:
        frame = 0
        while not stop.is_set():
            t0 = time.perf_counter_ns()
            ring.write_demo_frame(frame, timestamp_ns=frame * BUS_PERIOD_NS)
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
            with lock:
                write_latencies_us.append(elapsed_us)
            frame += 1
            time.sleep(1.0 / 30.0)

    def reader(poll_hz: float) -> None:
        nonlocal torn_reads
        interval = 1.0 / poll_hz
        while not stop.is_set():
            t0 = time.perf_counter_ns()
            snap = ring.read_latest()
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
            with lock:
                read_latencies_us.append(elapsed_us)
                if snap is not None and snap.torn:
                    torn_reads += 1
            time.sleep(interval)

    threads = [
        threading.Thread(target=writer, daemon=True),
        threading.Thread(target=lambda: reader(250.0), daemon=True),
        threading.Thread(target=lambda: reader(30.0), daemon=True),
        threading.Thread(target=lambda: reader(10.0), daemon=True),
        threading.Thread(target=lambda: reader(2.0), daemon=True),
    ]
    for th in threads:
        th.start()
    time.sleep(2.5)
    stop.set()
    for th in threads:
        th.join(timeout=5.0)

    assert torn_reads == 0
    all_latencies = read_latencies_us + write_latencies_us
    assert all_latencies
    p99 = statistics.quantiles(all_latencies, n=100)[98]
    assert p99 <= 500.0, f"p99 latency {p99:.1f} us exceeds 500 us"


def test_record_replay_round_trip_byte_identical() -> None:
    ring = WorldStateRingBuffer(capacity=32)
    for i in range(8):
        ring.write_slot(_make_slot(i + 1, i * BUS_PERIOD_NS, x=float(i), z0=float(i * 2)))
    recorder = WorldStateRecorder()
    recorder.record_from_ring(ring, max_slots=8)
    blob = recorder.to_bytes(capacity=32)
    loaded = WorldStateRecorder.from_bytes(blob)
    dest = WorldStateRingBuffer(capacity=32)
    player = WorldStatePlayer(loaded)
    replayed_bytes = player.replay_bytes(dest)
    assert len(replayed_bytes) == 8
    for original, replayed in zip(recorder._slots, replayed_bytes, strict=True):  # noqa: SLF001
        assert original == replayed


def test_npz_and_wsb_save_load(tmp_path: Path) -> None:
    ring = WorldStateRingBuffer(capacity=8)
    ring.write_slot(_make_slot(1, 0, x=1.0))
    rec = WorldStateRecorder()
    rec.record_from_ring(ring)
    wsb_path = tmp_path / "trace.wsb"
    npz_path = tmp_path / "trace.npz"
    rec.save(wsb_path, capacity=8)
    rec.save(npz_path, capacity=8)
    assert WorldStateRecorder.load(wsb_path).to_bytes(capacity=8) == rec.to_bytes(capacity=8)
    npz_loaded = WorldStateRecorder.load(npz_path)
    assert len(npz_loaded._slots) == 1  # noqa: SLF001
    assert slot_from_bytes(npz_loaded._slots[0])["objects"][0]["position_xyz"][0] == pytest.approx(  # noqa: SLF001
        1.0
    )


def test_acquire_write_view_zero_copy_commit() -> None:
    ring = WorldStateRingBuffer(capacity=4)
    with ring.acquire_write_view() as view:
        view.timestamp_ns = 42
        view.objects[0]["position_xyz"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    latest = ring.read_latest()
    assert latest is not None
    assert int(latest.slot["timestamp_ns"]) == 42
    assert latest.slot["objects"][0]["position_xyz"][2] == pytest.approx(3.0)
