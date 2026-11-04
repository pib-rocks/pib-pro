"""Shared REP-103 types and constants."""

from __future__ import annotations

from typing import Any

from pib_pro.common.rep103 import (
    AABB3D,
    AXIS_X_FORWARD,
    AXIS_Y_LEFT,
    AXIS_Z_UP,
    FRAME_BASE_LINK,
    FRAME_CAMERA_OAK4_OPTICAL,
    FRAME_CHEST_CAMERA_OAK4_OPTICAL,
    FRAME_HAND_LEFT,
    FRAME_HAND_RIGHT,
    FRAME_NECK_LINK,
    STANDARD_FRAMES,
    FloorState,
    JointStateSnapshot,
    Pose3D,
    ProactivityTier,
    Quaternion,
    Ray3D,
    Vector3D,
    ray_aabb_intersection,
    ray_sphere_intersection,
)

_KINEMATICS_EXPORTS: frozenset[str] = frozenset(
    {
        "ArmSide",
        "ArmFKResult",
        "ArmIKResult",
        "DualArmKinematicsConfig",
        "chest_camera_fov_penalty",
        "forward_kinematics_arm_6dof",
        "full_body_capsules",
        "solve_arm_ik_dls",
    },
)

__all__ = [
    "ArmFKResult",
    "ArmIKResult",
    "ArmSide",
    "DualArmKinematicsConfig",
    "AABB3D",
    "AXIS_X_FORWARD",
    "AXIS_Y_LEFT",
    "AXIS_Z_UP",
    "FRAME_BASE_LINK",
    "FRAME_CAMERA_OAK4_OPTICAL",
    "FRAME_CHEST_CAMERA_OAK4_OPTICAL",
    "FRAME_HAND_LEFT",
    "FRAME_HAND_RIGHT",
    "FRAME_NECK_LINK",
    "FloorState",
    "JointStateSnapshot",
    "Pose3D",
    "ProactivityTier",
    "Quaternion",
    "Ray3D",
    "STANDARD_FRAMES",
    "Vector3D",
    "chest_camera_fov_penalty",
    "forward_kinematics_arm_6dof",
    "full_body_capsules",
    "solve_arm_ik_dls",
    "ray_aabb_intersection",
    "ray_sphere_intersection",
]


def __getattr__(name: str) -> Any:
    if name in _KINEMATICS_EXPORTS:
        from pib_pro.common import kinematics as _kinematics

        return getattr(_kinematics, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
