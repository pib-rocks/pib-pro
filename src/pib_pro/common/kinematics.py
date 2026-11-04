"""Dual-arm 6-DoF REP-103 kinematics, DLS IK, and chest OAK-4 FOV clearance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pib_pro.common.rep103 import Vector3D
from pib_pro.layer0_reflex.somatic_reflex import Capsule3D

ArmSide = Literal["left", "right"]

JOINT_LIMIT_LO: float = -1.5708
JOINT_LIMIT_HI: float = 1.5708
CHEST_CAMERA_CONE_HALF_RAD: float = 0.38
_CHEST_TILT_RAD: float = 0.1745


@dataclass(frozen=True, slots=True)
class DualArmKinematicsConfig:
    left_shoulder_xyz: tuple[float, float, float] = (0.0, 0.16, 0.92)
    right_shoulder_xyz: tuple[float, float, float] = (0.0, -0.16, 0.92)
    upper_arm_m: float = 0.22
    forearm_m: float = 0.20
    hand_m: float = 0.10
    joint_limit_lo_rad: float = JOINT_LIMIT_LO
    joint_limit_hi_rad: float = JOINT_LIMIT_HI
    chest_camera_xyz: tuple[float, float, float] = (0.08, 0.0, 0.82)
    chest_camera_axis_xyz: tuple[float, float, float] = (
        math.cos(_CHEST_TILT_RAD),
        0.0,
        -math.sin(_CHEST_TILT_RAD),
    )
    cone_half_angle_rad: float = CHEST_CAMERA_CONE_HALF_RAD


DEFAULT_DUAL_ARM_CONFIG = DualArmKinematicsConfig()


@dataclass(frozen=True, slots=True)
class ArmFKResult:
    shoulder_xyz: NDArray[np.float64]
    elbow_xyz: NDArray[np.float64]
    wrist_xyz: NDArray[np.float64]
    end_effector_xyz: NDArray[np.float64]
    end_effector_rotation: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ArmIKResult:
    q_6dof: NDArray[np.float64]
    converged: bool
    position_error_m: float
    iterations: int


def _R_z(angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _R_y(angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _R_x(angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _shoulder_mount(side: ArmSide, config: DualArmKinematicsConfig) -> NDArray[np.float64]:
    if side == "left":
        return np.array(config.left_shoulder_xyz, dtype=np.float64)
    return np.array(config.right_shoulder_xyz, dtype=np.float64)


def _side_yaw_sign(side: ArmSide) -> float:
    return 1.0 if side == "left" else -1.0


def _clamp_joints(q: NDArray[np.float64], config: DualArmKinematicsConfig) -> NDArray[np.float64]:
    return np.clip(q, config.joint_limit_lo_rad, config.joint_limit_hi_rad)


def forward_kinematics_arm_6dof(
    q_6dof: NDArray[np.float64],
    side: ArmSide,
    config: DualArmKinematicsConfig | None = None,
) -> ArmFKResult:
    """Compute shoulder/elbow/wrist/EE poses in REP-103 ``base_link``."""
    cfg = config or DEFAULT_DUAL_ARM_CONFIG
    q = _clamp_joints(np.asarray(q_6dof, dtype=np.float64).reshape(6), cfg)
    yaw_sign = _side_yaw_sign(side)
    shoulder = _shoulder_mount(side, cfg)

    r = _R_z(yaw_sign * q[0])
    r = r @ _R_y(q[1])
    r = r @ _R_x(q[2])
    upper = np.array([cfg.upper_arm_m, 0.0, 0.0], dtype=np.float64)
    elbow = shoulder + r @ upper

    r = r @ _R_y(q[3])
    forearm = np.array([cfg.forearm_m, 0.0, 0.0], dtype=np.float64)
    wrist = elbow + r @ forearm

    r = r @ _R_x(q[4])
    r = r @ _R_y(q[5])
    hand = np.array([cfg.hand_m, 0.0, 0.0], dtype=np.float64)
    ee = wrist + r @ hand

    return ArmFKResult(
        shoulder_xyz=shoulder,
        elbow_xyz=elbow,
        wrist_xyz=wrist,
        end_effector_xyz=ee,
        end_effector_rotation=r,
    )


def _vec_to_capsule(a: NDArray[np.float64], b: NDArray[np.float64], radius: float) -> Capsule3D:
    return Capsule3D(
        Vector3D(float(a[0]), float(a[1]), float(a[2])),
        Vector3D(float(b[0]), float(b[1]), float(b[2])),
        radius,
    )


def full_body_capsules(
    q_left_6dof: NDArray[np.float64],
    q_right_6dof: NDArray[np.float64],
    config: DualArmKinematicsConfig | None = None,
) -> tuple[Capsule3D, ...]:
    """Torso, head, and both arm segment capsules for reflex clearance."""
    cfg = config or DEFAULT_DUAL_ARM_CONFIG
    left = forward_kinematics_arm_6dof(q_left_6dof, "left", cfg)
    right = forward_kinematics_arm_6dof(q_right_6dof, "right", cfg)
    torso = _vec_to_capsule(
        np.array([0.0, 0.0, 0.35], dtype=np.float64),
        np.array([0.0, 0.0, 0.95], dtype=np.float64),
        0.14,
    )
    head = _vec_to_capsule(
        np.array([0.0, 0.0, 0.95], dtype=np.float64),
        np.array([0.0, 0.0, 1.25], dtype=np.float64),
        0.11,
    )
    return (
        torso,
        head,
        _vec_to_capsule(left.shoulder_xyz, left.elbow_xyz, 0.05),
        _vec_to_capsule(left.elbow_xyz, left.wrist_xyz, 0.045),
        _vec_to_capsule(left.wrist_xyz, left.end_effector_xyz, 0.04),
        _vec_to_capsule(right.shoulder_xyz, right.elbow_xyz, 0.05),
        _vec_to_capsule(right.elbow_xyz, right.wrist_xyz, 0.045),
        _vec_to_capsule(right.wrist_xyz, right.end_effector_xyz, 0.04),
    )


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def chest_camera_fov_penalty(
    elbow_xyz: NDArray[np.float64],
    wrist_xyz: NDArray[np.float64],
    config: DualArmKinematicsConfig | None = None,
) -> float:
    """Depth of elbow/forearm intrusion into the chest camera central viewing cone."""
    cfg = config or DEFAULT_DUAL_ARM_CONFIG
    origin = np.array(cfg.chest_camera_xyz, dtype=np.float64)
    axis = _unit(np.array(cfg.chest_camera_axis_xyz, dtype=np.float64))
    half = cfg.cone_half_angle_rad
    max_pen = 0.0
    for pt in (elbow_xyz, wrist_xyz):
        rel = np.asarray(pt, dtype=np.float64).reshape(3) - origin
        along = float(np.dot(rel, axis))
        if along <= 0.0:
            continue
        perp = rel - along * axis
        perp_dist = float(np.linalg.norm(perp))
        allowed = math.tan(half) * along
        intrusion = perp_dist - allowed
        if intrusion > max_pen:
            max_pen = intrusion
    return max(0.0, max_pen)


def _numerical_jacobian(
    q: NDArray[np.float64],
    side: ArmSide,
    config: DualArmKinematicsConfig,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    j = np.zeros((3, 6), dtype=np.float64)
    base = forward_kinematics_arm_6dof(q, side, config).end_effector_xyz
    for i in range(6):
        dq = q.copy()
        dq[i] += eps
        pert = forward_kinematics_arm_6dof(dq, side, config).end_effector_xyz
        j[:, i] = (pert - base) / eps
    return j


def _dls_step(
    j: NDArray[np.float64],
    err: NDArray[np.float64],
    damping: float,
) -> NDArray[np.float64]:
    jjt = j @ j.T
    damped = jjt + (damping**2) * np.eye(3, dtype=np.float64)
    return j.T @ np.linalg.solve(damped, err)


def _nullspace_elbow_gradient(
    q: NDArray[np.float64],
    side: ArmSide,
    config: DualArmKinematicsConfig,
) -> NDArray[np.float64]:
    fk = forward_kinematics_arm_6dof(q, side, config)
    grad = np.zeros(6, dtype=np.float64)
    outward = 1.0 if side == "left" else -1.0
    grad[1] = outward * (config.chest_camera_xyz[1] - float(fk.elbow_xyz[1]))
    grad[3] = outward * 0.5
    return grad


def _geometric_ik_seeds(
    target: NDArray[np.float64],
    side: ArmSide,
    config: DualArmKinematicsConfig,
) -> list[NDArray[np.float64]]:
    """Planar 3-link guesses in the shoulder yaw frame (elbow up/down variants)."""
    shoulder = _shoulder_mount(side, config)
    dt = target - shoulder
    yaw_sign = _side_yaw_sign(side)
    if float(np.linalg.norm(dt)) < 1e-9:
        return [np.zeros(6, dtype=np.float64)]

    q0 = math.atan2(float(dt[1]), float(dt[0])) / yaw_sign
    q0 = float(np.clip(q0, config.joint_limit_lo_rad, config.joint_limit_hi_rad))
    rz = _R_z(yaw_sign * q0)
    local = rz.T @ dt
    px = float(local[0]) - config.hand_m
    pz = float(local[2])
    reach = math.hypot(px, pz)
    l1, l2 = config.upper_arm_m, config.forearm_m
    max_reach = l1 + l2 - 1e-4
    reach = min(max(reach, 1e-4), max_reach)
    cos_elbow = (reach * reach - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_elbow = float(np.clip(cos_elbow, -1.0, 1.0))
    alpha = math.atan2(pz, px)
    seeds: list[NDArray[np.float64]] = []
    for elbow_sign in (1.0, -1.0):
        q3 = elbow_sign * math.acos(cos_elbow)
        beta = math.acos((l1 * l1 + reach * reach - l2 * l2) / (2.0 * l1 * reach))
        for beta_sign in (1.0, -1.0):
            q1 = alpha - beta_sign * beta
            q = np.array([q0, q1, 0.0, q3, 0.0, 0.0], dtype=np.float64)
            seeds.append(_clamp_joints(q, config))
    return seeds


def _run_ik_dls_from_seed(
    target: NDArray[np.float64],
    side: ArmSide,
    q_seed: NDArray[np.float64],
    cfg: DualArmKinematicsConfig,
    *,
    avoid_chest_camera_fov: bool,
    max_iters: int,
    damping: float,
    tol_m: float,
) -> ArmIKResult:
    q = _clamp_joints(np.asarray(q_seed, dtype=np.float64).reshape(6), cfg)
    converged = False
    iters = 0
    for it in range(max_iters):
        iters = it + 1
        fk = forward_kinematics_arm_6dof(q, side, cfg)
        err = target - fk.end_effector_xyz
        err_norm = float(np.linalg.norm(err))
        if err_norm <= tol_m:
            converged = True
            break
        j = _numerical_jacobian(q, side, cfg)
        dq = _dls_step(j, err, damping)
        if avoid_chest_camera_fov:
            j_pinv = np.linalg.pinv(j)
            n_proj = np.eye(6, dtype=np.float64) - j_pinv @ j
            fov_pen = chest_camera_fov_penalty(fk.elbow_xyz, fk.wrist_xyz, cfg)
            grad = _nullspace_elbow_gradient(q, side, cfg)
            if fov_pen > 0.0:
                grad = grad + 2.0 * fov_pen * grad
            dq = dq + n_proj @ (0.08 * grad)
        q = _clamp_joints(q + dq, cfg)

    fk_final = forward_kinematics_arm_6dof(q, side, cfg)
    final_err = float(np.linalg.norm(target - fk_final.end_effector_xyz))
    return ArmIKResult(
        q_6dof=q,
        converged=converged or final_err <= 0.005,
        position_error_m=final_err,
        iterations=iters,
    )


def solve_arm_ik_dls(
    target_xyz: NDArray[np.float64],
    side: ArmSide,
    q_init: NDArray[np.float64] | None = None,
    *,
    avoid_chest_camera_fov: bool = True,
    max_iters: int = 80,
    damping: float = 0.04,
    tol_m: float = 0.003,
    config: DualArmKinematicsConfig | None = None,
) -> ArmIKResult:
    """Damped least-squares IK with optional chest-camera FOV null-space bias."""
    cfg = config or DEFAULT_DUAL_ARM_CONFIG
    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    seeds: list[NDArray[np.float64]] = []
    if q_init is not None:
        seeds.append(np.asarray(q_init, dtype=np.float64).reshape(6))
    else:
        seeds.extend(_geometric_ik_seeds(target, side, cfg))
        seeds.append(
            np.array(
                [0.35 * _side_yaw_sign(side), -0.45, 0.0, 0.85, 0.0, 0.0],
                dtype=np.float64,
            ),
        )

    best = _run_ik_dls_from_seed(
        target,
        side,
        seeds[0],
        cfg,
        avoid_chest_camera_fov=avoid_chest_camera_fov,
        max_iters=max_iters,
        damping=damping,
        tol_m=tol_m,
    )
    for seed in seeds[1:]:
        trial = _run_ik_dls_from_seed(
            target,
            side,
            seed,
            cfg,
            avoid_chest_camera_fov=avoid_chest_camera_fov,
            max_iters=max_iters,
            damping=damping,
            tol_m=tol_m,
        )
        if trial.position_error_m < best.position_error_m:
            best = trial
    return best
