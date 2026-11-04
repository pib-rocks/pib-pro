"""Tests for HJEPA world model latency and surprise on the world-state bus."""

from __future__ import annotations

import statistics
import time

import numpy as np

from pib_pro.world_model import HJEPAConfig, HJEPAWorldModelEngine
from pib_pro.world_state_bus import BUS_PERIOD_NS, MAX_JOINTS, WorldStateRingBuffer


def test_encode_predict_self_latency_30hz() -> None:
    engine = HJEPAWorldModelEngine(HJEPAConfig())
    scene = np.zeros(32 * 3, dtype=np.float32)
    joints = np.zeros(MAX_JOINTS, dtype=np.float32)
    vel = np.zeros(MAX_JOINTS, dtype=np.float32)
    samples_ms: list[float] = []
    for _ in range(200):
        t0 = time.perf_counter()
        z = engine.encode(scene, joints, vel)
        _ = engine.predict_self(z, joints)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
    p95 = statistics.quantiles(samples_ms, n=20)[18]
    assert p95 <= 12.0


def test_batch_predict_self_k8_horizon_1s_under_8ms() -> None:
    engine = HJEPAWorldModelEngine(HJEPAConfig(dt_sec=1.0 / 30.0))
    z = engine.encode(np.zeros(96, dtype=np.float32), np.zeros(MAX_JOINTS), np.zeros(MAX_JOINTS))
    k = 8
    horizon_steps = 30
    actions = np.zeros((k, horizon_steps, MAX_JOINTS), dtype=np.float32)
    t0 = time.perf_counter()
    out = engine.predict_self(z, actions)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out.shape == (k, 256)
    assert elapsed_ms <= 8.0


def test_surprise_spike_on_object_displacement() -> None:
    config = HJEPAConfig(surprise_threshold=0.35)
    engine = HJEPAWorldModelEngine(config)
    ring = WorldStateRingBuffer(capacity=64)
    threshold = config.surprise_threshold

    for frame in range(8):
        with ring.acquire_write_view() as view:
            view.timestamp_ns = frame * BUS_PERIOD_NS
            view.frame_id = frame
            view.objects[0]["object_id"] = np.uint64(1)
            view.objects[0]["position_xyz"] = np.array([0.5, 0.0, 0.1], dtype=np.float32)
            view.joint_positions_rad[:] = 0.0
            view.raw["joint_velocities_rad_s"][:] = 0.0
        engine.step_on_bus(ring)

    spike_frame: int | None = None
    for offset in range(6):
        frame = 8 + offset
        with ring.acquire_write_view() as view:
            view.timestamp_ns = frame * BUS_PERIOD_NS
            view.frame_id = frame
            view.objects[0]["object_id"] = np.uint64(1)
            x = 0.5 + (0.04 if offset >= 1 else 0.0)
            view.objects[0]["position_xyz"] = np.array([x, 0.0, 0.1], dtype=np.float32)
        surprise, _ = engine.step_on_bus(ring)
        if surprise > threshold and spike_frame is None:
            spike_frame = offset

    assert spike_frame is not None
    assert spike_frame <= 2


def test_compute_surprise_regions() -> None:
    engine = HJEPAWorldModelEngine()
    z = np.ones(256, dtype=np.float32) * 0.1
    zh = np.zeros(256, dtype=np.float32)
    scalar, regions = engine.compute_surprise(z, zh)
    assert scalar == np.linalg.norm(z - zh)
    assert regions.shape == (8,)
