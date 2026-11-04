"""250 Hz somatic reflex: limits, proximity capsules, and compliant interaction."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pib_pro.common import Vector3D
from pib_pro.layer0_reflex import (
    ActuatorCommand,
    ComplianceEvent,
    ComplianceEventKind,
    SafetyEnvelopeConfig,
)
from pib_pro.world_state_bus.schema import MAX_JOINTS, WorldStateSlotView

STS_ACC_MIN: int = 30
STS_ACC_MAX: int = 254
DEFAULT_SLOWDOWN_MARGIN_M: float = 0.25
DEFAULT_STOP_MARGIN_M: float = 0.06
HANDOVER_TUG_MIN_S: float = 0.032
HANDOVER_TUG_MAX_S: float = 0.050
GUIDANCE_HOLD_S: float = 0.040
NOISE_SPIKE_MAX_S: float = 0.015

_ARM_HEAD_HALF_RANGE_RAD: float = 1.5708
_FINGER_HALF_RANGE_RAD: float = 3.1416
_FINGER_JOINT_START: int = 20

_GUIDANCE_JOINT_INDICES: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5})
_HANDOVER_JOINT_INDICES: frozenset[int] = frozenset({6, 7, 8, 9, 10, 11, 12, 13, 14, 15})


@dataclass(frozen=True, slots=True)
class JointLimitsConfig:
    """Per-joint reflex limits with STS acceleration floor invariant."""

    min_pos_rad: tuple[float, ...]
    max_pos_rad: tuple[float, ...]
    max_velocity_rad_s: float = 3.5
    max_current_ma: float = 1200.0
    min_sts_acceleration: int = STS_ACC_MIN

    @classmethod
    def default_pib(cls, n_joints: int = MAX_JOINTS) -> JointLimitsConfig:
        lo: list[float] = []
        hi: list[float] = []
        for idx in range(n_joints):
            if idx >= _FINGER_JOINT_START:
                span = _FINGER_HALF_RANGE_RAD
            else:
                span = _ARM_HEAD_HALF_RANGE_RAD
            lo.append(-span)
            hi.append(span)
        return cls(
            min_pos_rad=tuple(lo),
            max_pos_rad=tuple(hi),
        )

    def clamp_sts_acceleration(self, value: int | float) -> int:
        floor = max(STS_ACC_MIN, int(self.min_sts_acceleration))
        return int(min(STS_ACC_MAX, max(floor, int(round(value)))))


@dataclass(frozen=True, slots=True)
class Capsule3D:
    start_xyz: Vector3D
    end_xyz: Vector3D
    radius_m: float

    def __post_init__(self) -> None:
        if self.radius_m < 0.0 or not math.isfinite(self.radius_m):
            msg = "radius_m must be finite and non-negative"
            raise ValueError(msg)


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _norm3(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot3(a, a))


def _segment_segment_distance_m(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    q1: tuple[float, float, float],
    q2: tuple[float, float, float],
) -> float:
    """Minimum distance between two 3D line segments."""
    u = _sub3(p2, p1)
    v = _sub3(q2, q1)
    w0 = _sub3(p1, q1)
    a = _dot3(u, u)
    b = _dot3(u, v)
    c = _dot3(v, v)
    d = _dot3(u, w0)
    e = _dot3(v, w0)
    denom = a * c - b * b
    sc = 0.0
    tc = 0.0
    if denom < 1e-12:
        sc = 0.0
        tc = e / c if c > 1e-12 else 0.0
    else:
        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom
    sc = min(1.0, max(0.0, sc))
    tc = min(1.0, max(0.0, tc))
    dp = _add3(w0, _sub3(_scale3(u, sc), _scale3(v, tc)))
    return _norm3(dp)


def capsule_distance_m(c1: Capsule3D, c2: Capsule3D) -> float:
    """Segment-segment clearance minus combined capsule radii."""
    p1 = (c1.start_xyz.x, c1.start_xyz.y, c1.start_xyz.z)
    p2 = (c1.end_xyz.x, c1.end_xyz.y, c1.end_xyz.z)
    q1 = (c2.start_xyz.x, c2.start_xyz.y, c2.start_xyz.z)
    q2 = (c2.end_xyz.x, c2.end_xyz.y, c2.end_xyz.z)
    raw = _segment_segment_distance_m(p1, p2, q1, q2)
    return raw - (c1.radius_m + c2.radius_m)


def proximity_velocity_scale(
    clearance_m: float,
    *,
    slowdown_margin_m: float = DEFAULT_SLOWDOWN_MARGIN_M,
    stop_margin_m: float = DEFAULT_STOP_MARGIN_M,
) -> float:
    if clearance_m >= slowdown_margin_m:
        return 1.0
    if clearance_m <= stop_margin_m:
        return 0.0
    return (clearance_m - stop_margin_m) / (slowdown_margin_m - stop_margin_m)


def build_robot_forearm_capsule(
    side: str,
    elbow_xyz: Vector3D,
    wrist_xyz: Vector3D,
    *,
    radius_m: float = 0.045,
) -> Capsule3D:
    _ = side
    return Capsule3D(elbow_xyz, wrist_xyz, radius_m)


def build_human_hand_capsule(
    hand_xyz: tuple[float, float, float] | np.ndarray,
    *,
    radius_m: float = 0.05,
    reach_m: float = 0.12,
) -> Capsule3D:
    hx, hy, hz = float(hand_xyz[0]), float(hand_xyz[1]), float(hand_xyz[2])
    start = Vector3D(hx, hy, hz)
    end = Vector3D(hx, hy, hz + reach_m)
    return Capsule3D(start, end, radius_m)


def build_human_torso_capsule(
    head_xyz: tuple[float, float, float] | np.ndarray,
    *,
    radius_m: float = 0.18,
    torso_length_m: float = 0.45,
) -> Capsule3D:
    hx, hy, hz = float(head_xyz[0]), float(head_xyz[1]), float(head_xyz[2])
    start = Vector3D(hx, hy, hz - 0.15)
    end = Vector3D(hx, hy, hz - torso_length_m)
    return Capsule3D(start, end, radius_m)


@dataclass(frozen=True, slots=True)
class SafeMotorCommand:
    position_targets_rad: tuple[float, ...]
    velocity_targets_rad_s: tuple[float, ...]
    sts_acceleration: tuple[int, ...]
    current_limits_ma: tuple[float, ...]
    handover_released: bool
    compliant_guidance_active: bool
    estop_active: bool


@dataclass(slots=True)
class SomaticReflexController:
    """250 Hz safety envelope with compliant hand-over and kinesthetic teaching."""

    limits: JointLimitsConfig = field(default_factory=JointLimitsConfig.default_pib)
    slowdown_margin_m: float = DEFAULT_SLOWDOWN_MARGIN_M
    stop_margin_m: float = DEFAULT_STOP_MARGIN_M
    handover_tug_threshold_ma: float = 180.0
    guidance_threshold_ma: float = 350.0
    holding_object: bool = False
    handover_released: bool = False
    compliant_guidance_active: bool = False
    estop_active: bool = False
    _envelope: SafetyEnvelopeConfig | None = field(default=None, init=False, repr=False)
    _pending_commands: ActuatorCommand | None = field(default=None, init=False, repr=False)
    _compliance_events: deque[ComplianceEvent] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )
    _residual_window: deque[tuple[float, float]] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )
    _tug_active_s: float = field(default=0.0, init=False, repr=False)
    _guidance_active_s: float = field(default=0.0, init=False, repr=False)
    _guided_log: list[tuple[float, tuple[float, ...]]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def configure(self, envelope: SafetyEnvelopeConfig) -> None:
        self._envelope = envelope

    def apply_command(self, cmd: ActuatorCommand, stamp_monotonic_s: float) -> None:
        _ = stamp_monotonic_s
        self._pending_commands = cmd

    def poll_compliance(self) -> tuple[ComplianceEvent, ...]:
        events = tuple(self._compliance_events)
        self._compliance_events.clear()
        return events

    def request_estop(self) -> None:
        self.estop_active = True
        self._compliance_events.append(
            ComplianceEvent(
                ComplianceEventKind.ESTOP,
                "immediate estop",
                Vector3D(0.0, 0.0, 0.0),
            ),
        )

    def clear_estop(self) -> None:
        self.estop_active = False

    def guided_trajectory(self) -> tuple[tuple[float, tuple[float, ...]], ...]:
        return tuple(self._guided_log)

    def _clamp_positions(self, values: tuple[float, ...]) -> tuple[float, ...]:
        out: list[float] = []
        n = min(len(values), len(self.limits.min_pos_rad))
        for i in range(n):
            lo = self.limits.min_pos_rad[i]
            hi = self.limits.max_pos_rad[i]
            out.append(min(hi, max(lo, float(values[i]))))
        return tuple(out)

    def _clamp_velocities(
        self,
        values: tuple[float, ...],
        scale: float,
    ) -> tuple[float, ...]:
        cap = self.limits.max_velocity_rad_s * scale
        if self._envelope is not None:
            cap = min(cap, self._envelope.max_joint_velocity_rad_s * scale)
        return tuple(min(cap, max(-cap, float(v))) for v in values)

    def _clamp_currents(self, measured: tuple[float, ...]) -> tuple[float, ...]:
        cap = self.limits.max_current_ma
        return tuple(min(cap, max(-cap, float(c))) for c in measured)

    def _clamp_sts(self, desired: tuple[float, ...] | tuple[int, ...]) -> tuple[int, ...]:
        return tuple(self.limits.clamp_sts_acceleration(v) for v in desired)

    def _update_compliance(
        self,
        measured_loads_ma: tuple[float, ...],
        measured_positions_rad: tuple[float, ...],
        dt_s: float,
        bus_view: WorldStateSlotView | None,
    ) -> None:
        if len(measured_loads_ma) == 0:
            return
        expected = tuple(0.0 for _ in measured_loads_ma)
        if self._pending_commands is not None:
            expected = tuple(0.0 for _ in self._pending_commands.joint_names)
        residuals = [
            float(measured_loads_ma[i]) - float(expected[i % len(expected)])
            for i in range(len(measured_loads_ma))
        ]
        hand_residual = max(
            (abs(residuals[i]) for i in _HANDOVER_JOINT_INDICES if i < len(residuals)),
            default=0.0,
        )
        guide_residual = max(
            (abs(residuals[i]) for i in _GUIDANCE_JOINT_INDICES if i < len(residuals)),
            default=0.0,
        )
        self._residual_window.append((dt_s, hand_residual))
        total_window = sum(w[0] for w in self._residual_window)
        while total_window > HANDOVER_TUG_MAX_S and self._residual_window:
            dropped = self._residual_window.popleft()[0]
            total_window -= dropped

        if hand_residual >= self.handover_tug_threshold_ma:
            self._tug_active_s += dt_s
        elif self._tug_active_s <= NOISE_SPIKE_MAX_S:
            self._tug_active_s = 0.0
        else:
            self._tug_active_s = max(0.0, self._tug_active_s - dt_s)

        if (
            self.holding_object
            and not self.handover_released
            and self._tug_active_s >= HANDOVER_TUG_MIN_S
            and self._tug_active_s <= HANDOVER_TUG_MAX_S
        ):
            self.handover_released = True
            self._compliance_events.append(
                ComplianceEvent(
                    ComplianceEventKind.HANDOVER_RELEASE,
                    "gentle tug hand-over",
                    Vector3D(0.0, 0.0, 1.0),
                ),
            )

        if guide_residual >= self.guidance_threshold_ma:
            self._guidance_active_s += dt_s
        else:
            self._guidance_active_s = 0.0

        if self._guidance_active_s >= GUIDANCE_HOLD_S:
            if not self.compliant_guidance_active:
                self.compliant_guidance_active = True
                self._compliance_events.append(
                    ComplianceEvent(
                        ComplianceEventKind.COMPLIANT_GUIDANCE,
                        "kinesthetic teaching engaged",
                        Vector3D(1.0, 0.0, 0.0),
                    ),
                )
            self._guided_log.append((self._guidance_active_s, measured_positions_rad))
            if bus_view is not None:
                flags = bus_view.raw["external_contact_flags"]
                for idx in _GUIDANCE_JOINT_INDICES:
                    if idx < len(flags):
                        flags[idx] = np.uint32(1)

    def step(
        self,
        desired_positions_rad: tuple[float, ...],
        desired_velocities_rad_s: tuple[float, ...],
        desired_sts_acc: tuple[int, ...],
        measured_positions_rad: tuple[float, ...],
        measured_velocities_rad_s: tuple[float, ...],
        measured_loads_ma: tuple[float, ...],
        dt_s: float,
        bus: WorldStateSlotView | None = None,
    ) -> SafeMotorCommand:
        """One reflex tick: clamp, proximity scale, compliance, optional bus flags."""
        if self.estop_active:
            n = len(desired_positions_rad)
            zeros_v = tuple(0.0 for _ in range(n))
            zeros_i = tuple(self.limits.clamp_sts_acceleration(STS_ACC_MIN) for _ in range(n))
            caps = tuple(self.limits.max_current_ma for _ in range(n))
            return SafeMotorCommand(
                self._clamp_positions(measured_positions_rad),
                zeros_v,
                zeros_i,
                caps,
                self.handover_released,
                self.compliant_guidance_active,
                True,
            )

        prox_scale = 1.0
        if bus is not None:
            slot = bus.raw
            hand = slot["right_hand_xyz"]
            elbow = Vector3D(0.25, -0.2, 0.85)
            wrist = Vector3D(0.45, -0.2, 0.82)
            if len(measured_positions_rad) > 3:
                wrist = Vector3D(
                    0.35 + float(measured_positions_rad[3]) * 0.1,
                    -0.2,
                    0.82,
                )
            robot_cap = build_robot_forearm_capsule("right", elbow, wrist)
            human_cap = build_human_hand_capsule(hand)
            clearance = capsule_distance_m(robot_cap, human_cap)
            prox_scale = proximity_velocity_scale(
                clearance,
                slowdown_margin_m=self.slowdown_margin_m,
                stop_margin_m=self.stop_margin_m,
            )

        pos_targets = desired_positions_rad
        if self.compliant_guidance_active:
            pos_targets = measured_positions_rad

        clamped_pos = self._clamp_positions(pos_targets)
        clamped_vel = self._clamp_velocities(desired_velocities_rad_s, prox_scale)
        clamped_sts = self._clamp_sts(desired_sts_acc)
        clamped_curr = self._clamp_currents(measured_loads_ma)

        if self.handover_released:
            opened: list[float] = list(clamped_pos)
            for idx in _HANDOVER_JOINT_INDICES:
                if idx < len(opened):
                    opened[idx] = min(opened[idx], 0.35)
            clamped_pos = tuple(opened)

        _ = measured_velocities_rad_s
        self._update_compliance(measured_loads_ma, measured_positions_rad, dt_s, bus)

        return SafeMotorCommand(
            clamped_pos,
            clamped_vel,
            clamped_sts,
            clamped_curr,
            self.handover_released,
            self.compliant_guidance_active,
            False,
        )
