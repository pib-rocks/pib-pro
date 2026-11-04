"""Layer 0 somatic reflex safety and actuation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

from pib_pro.common import Vector3D


@dataclass(frozen=True, slots=True)
class SafetyEnvelopeConfig:
    """Hard limits for reflex loop (250 Hz)."""

    max_joint_velocity_rad_s: float
    max_end_effector_speed_m_s: float
    collision_margin_m: float


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    """Low-level command issued by Layer 0."""

    joint_names: tuple[str, ...]
    position_targets_rad: tuple[float, ...]
    torque_scale: float


class ComplianceEventKind(Enum):
    LIMIT = auto()
    COLLISION = auto()
    ESTOP = auto()
    HANDOVER_RELEASE = auto()
    COMPLIANT_GUIDANCE = auto()


@dataclass(frozen=True, slots=True)
class ComplianceEvent:
    kind: ComplianceEventKind
    detail: str
    reaction_vector: Vector3D


class SomaticReflexProtocol(Protocol):
    """250 Hz reflex controller."""

    def configure(self, envelope: SafetyEnvelopeConfig) -> None: ...

    def apply_command(self, cmd: ActuatorCommand, stamp_monotonic_s: float) -> None: ...

    def poll_compliance(self) -> tuple[ComplianceEvent, ...]: ...


from pib_pro.layer0_reflex.somatic_reflex import (  # noqa: E402
    Capsule3D,
    JointLimitsConfig,
    SafeMotorCommand,
    SomaticReflexController,
    build_human_hand_capsule,
    build_human_torso_capsule,
    build_robot_forearm_capsule,
    capsule_distance_m,
    proximity_velocity_scale,
)

__all__ = [
    "ActuatorCommand",
    "Capsule3D",
    "ComplianceEvent",
    "ComplianceEventKind",
    "JointLimitsConfig",
    "SafeMotorCommand",
    "SafetyEnvelopeConfig",
    "SomaticReflexController",
    "SomaticReflexProtocol",
    "build_human_hand_capsule",
    "build_human_torso_capsule",
    "build_robot_forearm_capsule",
    "capsule_distance_m",
    "proximity_velocity_scale",
]
