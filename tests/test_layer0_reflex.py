"""Layer 0 somatic reflex: 250 Hz safety, capsules, and compliance."""

from __future__ import annotations

import numpy as np
import pytest

from pib_pro.common import Vector3D
from pib_pro.layer0_reflex import (
    Capsule3D,
    ComplianceEventKind,
    JointLimitsConfig,
    SafeMotorCommand,
    SomaticReflexController,
    capsule_distance_m,
    proximity_velocity_scale,
)
from pib_pro.world_state_bus import WorldStateRingBuffer, empty_slot_array


def _zeros(n: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(n))


def _sts(n: int, val: int) -> tuple[int, ...]:
    return tuple(val for _ in range(n))


def test_joint_limits_sts_floor_invariant() -> None:
    cfg = JointLimitsConfig.default_pib(4)
    assert cfg.clamp_sts_acceleration(1) == 30
    assert cfg.clamp_sts_acceleration(300) == 254
    assert cfg.clamp_sts_acceleration(80) == 80


def test_capsule_distance_parallel_segments() -> None:
    c1 = Capsule3D(Vector3D(0, 0, 0), Vector3D(1, 0, 0), 0.05)
    c2 = Capsule3D(Vector3D(0, 1, 0), Vector3D(1, 1, 0), 0.05)
    d = capsule_distance_m(c1, c2)
    assert d == pytest.approx(0.9)


def test_proximity_velocity_scaling_and_stop() -> None:
    assert proximity_velocity_scale(0.30) == 1.0
    mid = proximity_velocity_scale(0.155)
    assert 0.0 < mid < 1.0
    assert proximity_velocity_scale(0.06) == 0.0
    assert proximity_velocity_scale(0.02) == 0.0


def test_250hz_one_tick_clamping_and_sts_never_below_30() -> None:
    ctrl = SomaticReflexController()
    n = 8
    dt = 0.004
    desired_pos = tuple(99.0 for _ in range(n))
    desired_vel = tuple(99.0 for _ in range(n))
    measured_pos = tuple(-99.0 for _ in range(n))
    loads = tuple(500.0 for _ in range(n))
    cmd = ctrl.step(
        desired_pos,
        desired_vel,
        _sts(n, 1),
        measured_pos,
        _zeros(n),
        loads,
        dt,
    )
    assert isinstance(cmd, SafeMotorCommand)
    for i in range(n):
        assert cmd.position_targets_rad[i] <= ctrl.limits.max_pos_rad[i]
        assert cmd.position_targets_rad[i] >= ctrl.limits.min_pos_rad[i]
        assert abs(cmd.velocity_targets_rad_s[i]) <= ctrl.limits.max_velocity_rad_s + 1e-6
        assert cmd.sts_acceleration[i] >= 30
        assert cmd.sts_acceleration[i] <= 254
    assert cmd.sts_acceleration[0] == 30


def test_human_proximity_scales_velocity_on_bus() -> None:
    ring = WorldStateRingBuffer(capacity=8)
    slot = empty_slot_array(1)[0]
    slot["right_hand_xyz"] = np.array([0.42, -0.2, 0.82], dtype=np.float32)
    ring.write_slot(slot)
    ctrl = SomaticReflexController()
    n = 4
    cmd_far = ctrl.step(
        _zeros(n),
        tuple(3.0 for _ in range(n)),
        _sts(n, 40),
        _zeros(n),
        _zeros(n),
        _zeros(n),
        0.004,
    )
    slot["right_hand_xyz"] = np.array([0.44, -0.2, 0.82], dtype=np.float32)
    ring.write_slot(slot)
    latest2 = ring.read_latest()
    assert latest2 is not None
    cmd_near = ctrl.step(
        _zeros(n),
        tuple(3.0 for _ in range(n)),
        _sts(n, 40),
        _zeros(n),
        _zeros(n),
        _zeros(n),
        0.004,
        latest2.view,
    )
    assert abs(cmd_near.velocity_targets_rad_s[0]) < abs(cmd_far.velocity_targets_rad_s[0])


def test_handover_release_after_40ms_not_on_4ms_spike() -> None:
    ctrl = SomaticReflexController(holding_object=True)
    n = 16
    dt = 0.004
    loads_quiet = tuple(20.0 for _ in range(n))
    loads_tug = list(loads_quiet)
    for idx in (8, 9, 10):
        loads_tug[idx] = 400.0

    for _ in range(1):
        ctrl.step(_zeros(n), _zeros(n), _sts(n, 50), _zeros(n), _zeros(n), tuple(loads_tug), dt)
    assert not ctrl.handover_released

    ctrl2 = SomaticReflexController(holding_object=True)
    for _ in range(10):
        ctrl2.step(_zeros(n), _zeros(n), _sts(n, 50), _zeros(n), _zeros(n), tuple(loads_tug), dt)
    assert ctrl2.handover_released
    events = ctrl2.poll_compliance()
    assert any(e.kind is ComplianceEventKind.HANDOVER_RELEASE for e in events)


def test_compliant_guidance_sets_bus_flags() -> None:
    ring = WorldStateRingBuffer(capacity=4)
    slot = empty_slot_array(1)[0]
    ring.write_slot(slot)
    latest = ring.read_latest()
    assert latest is not None
    view = latest.view

    ctrl = SomaticReflexController()
    n = 8
    dt = 0.004
    loads = [50.0] * n
    for idx in (0, 1, 2):
        loads[idx] = 900.0
    load_tuple = tuple(loads)

    for _ in range(12):
        ctrl.step(
            _zeros(n),
            _zeros(n),
            _sts(n, 60),
            tuple(0.1 * i for i in range(n)),
            _zeros(n),
            load_tuple,
            dt,
            view,
        )
    assert ctrl.compliant_guidance_active
    flags = view.raw["external_contact_flags"]
    assert int(flags[0]) == 1
    assert int(flags[1]) == 1
    events = ctrl.poll_compliance()
    assert any(e.kind is ComplianceEventKind.COMPLIANT_GUIDANCE for e in events)


def test_estop_within_one_tick() -> None:
    ctrl = SomaticReflexController()
    n = 4
    ctrl.request_estop()
    cmd = ctrl.step(
        tuple(1.0 for _ in range(n)),
        tuple(2.0 for _ in range(n)),
        _sts(n, 100),
        _zeros(n),
        tuple(1.0 for _ in range(n)),
        _zeros(n),
        0.004,
    )
    assert cmd.estop_active
    assert all(v == 0.0 for v in cmd.velocity_targets_rad_s)
