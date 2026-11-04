"""Hybrid visuomotor controller at 25 Hz."""

from __future__ import annotations

import numpy as np

from pib_pro.layer1_autopilot import (
    GR00TActionProposer,
    HybridVisuomotorController,
)
from pib_pro.layer1_autopilot.visuomotor_hybrid import (
    OBSTACLE_CLEARANCE_M,
    forward_kinematics_ee,
)
from pib_pro.world_model.hjepa_engine import HJEPAWorldModelEngine
from pib_pro.world_state_bus import WorldStateRingBuffer


def _seed_reach_with_obstacle(
    bus: WorldStateRingBuffer,
    *,
    target_xyz: np.ndarray,
    obstacle_xyz: np.ndarray,
) -> None:
    with bus.acquire_write_view() as view:
        view.timestamp_ns = 0
        view.sequence_id = 1
        q = np.zeros(32, dtype=np.float32)
        q[0] = -0.25
        q[1] = 0.0
        q[2] = -0.12
        view.joint_positions_rad[...] = q
        view.raw["joint_velocities_rad_s"][...] = 0.0
        view.raw["joint_loads_nm_or_ma"][...] = 0.0
        view.objects[0]["object_id"] = np.uint64(101)
        view.objects[0]["position_xyz"] = np.asarray(target_xyz, dtype=np.float32)
        view.objects[1]["object_id"] = np.uint64(202)
        view.objects[1]["position_xyz"] = np.asarray(obstacle_xyz, dtype=np.float32)
        view.z_t[...] = 0.0
        view.raw["z_goal"][...] = 0.0


def test_hybrid_step_p95_within_40ms() -> None:
    bus = WorldStateRingBuffer(capacity=16)
    target = np.array([0.58, 0.0, 1.08], dtype=np.float32)
    start = forward_kinematics_ee(np.array([-0.25, 0.0, -0.12], dtype=np.float32))
    mid = (start + target) * 0.5
    _seed_reach_with_obstacle(bus, target_xyz=target, obstacle_xyz=mid + np.array([0.0, 0.0, 0.0]))

    controller = HybridVisuomotorController(hjepa=HJEPAWorldModelEngine())
    timings: list[float] = []
    for _ in range(80):
        result = controller.step(bus)
        timings.append(result.elapsed_s)
    timings.sort()
    p95 = timings[int(0.95 * len(timings)) - 1]
    assert p95 <= 0.040


def test_obstacle_avoidance_arch_selection() -> None:
    bus = WorldStateRingBuffer(capacity=16)
    controller = HybridVisuomotorController(
        hjepa=HJEPAWorldModelEngine(),
        obstacle_weight=200.0,
    )
    successes = 0
    for trial in range(20):
        target = np.array([0.56 + trial * 0.001, 0.0, 1.06], dtype=np.float32)
        start = forward_kinematics_ee(np.array([-0.25, 0.0, -0.12], dtype=np.float32))
        obstacle = (start + target) * 0.5
        obstacle[1] = 0.0
        _seed_reach_with_obstacle(bus, target_xyz=target, obstacle_xyz=obstacle)
        err, min_clear = controller.simulate_reach(bus, max_steps=90)
        if min_clear >= OBSTACLE_CLEARANCE_M and err <= 0.02:
            successes += 1
    assert successes >= 18


def test_mid_reach_target_shift_converges() -> None:
    bus = WorldStateRingBuffer(capacity=32)
    target = np.array([0.55, 0.0, 1.05], dtype=np.float32)
    _seed_reach_with_obstacle(
        bus,
        target_xyz=target,
        obstacle_xyz=np.array([0.0, 0.5, 0.5], dtype=np.float32),
    )
    controller = HybridVisuomotorController(hjepa=HJEPAWorldModelEngine())
    shift = np.array([0.04, 0.0, 0.0], dtype=np.float32)
    err, _ = controller.simulate_reach(
        bus,
        max_steps=100,
        mid_reach_target_shift_m=shift,
        shift_after_step=40,
    )
    assert err <= 0.015


def test_groot_proposes_eight_candidates() -> None:
    proposer = GR00TActionProposer()
    q = np.zeros(32, dtype=np.float32)
    target = np.array([0.5, 0.1, 1.0], dtype=np.float32)
    z = np.zeros(256, dtype=np.float32)
    chunks = proposer.propose(q, target, z)
    assert chunks.shape == (8, 20, 32)
    mid = chunks.shape[1] // 2
    ee0 = forward_kinematics_ee(chunks[0, mid])
    ee1 = forward_kinematics_ee(chunks[1, mid])
    assert float(np.linalg.norm(ee0 - ee1)) > 0.01
