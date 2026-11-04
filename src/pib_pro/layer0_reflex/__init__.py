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
