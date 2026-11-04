"""Human 3D observations, workspace prediction, and joint-attention solver."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from pib_pro.common.rep103 import (
    AABB3D,
    Ray3D,
    Vector3D,
    ray_aabb_intersection,
    ray_sphere_intersection,
)

if TYPE_CHECKING:
    from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer
    from pib_pro.world_state_bus.schema import WorldStateSlotView

GAZE_WEIGHT: float = 1.0
POINTING_WEIGHT: float = 2.5
HIT_BONUS: float = 0.35
SOFTMAX_TEMPERATURE: float = 0.08

SEED_RESPEAKER_USB_VID: Final[int] = 0x2886
SEED_RESPEAKER_USB_PID: Final[int] = 0x0018


@dataclass(frozen=True, slots=True)
class SeeedMicArrayConfig:
    """Seeed Studio ReSpeaker Mic Array v2.0 (XVF3000) capture profile."""

    usb_vid: int = SEED_RESPEAKER_USB_VID
    usb_pid: int = SEED_RESPEAKER_USB_PID
    model_name: str = "Seeed ReSpeaker Mic Array v2.0 (XVF3000)"
    channels: int = 6
    aec_reference_channel: int = 0
    sample_rate_hz: int = 16000
    period_size_frames: int = 256
    mounting_yaw_offset_rad: float = 0.0


def seeed_doa_deg_to_rep103_azimuth_rad(
    doa_deg: float,
    mounting_yaw_offset_rad: float = 0.0,
) -> float:
    """Map Seeed 0..360° DOA to REP-103 azimuth (0 = +X forward, +π/2 = +Y left)."""
    if not math.isfinite(doa_deg):
        raise ValueError("seeed_doa_deg_to_rep103_azimuth_rad: doa_deg must be finite")
    wrapped = doa_deg % 360.0
    az = math.radians(wrapped) + mounting_yaw_offset_rad
    return math.atan2(math.sin(az), math.cos(az))


@dataclass(frozen=True, slots=True)
class HumanObservation3D:
    """Fused human cues in REP-103 ``base_link``."""

    head_position: Vector3D
    gaze_direction: Vector3D
    pointing_origin: Vector3D
    pointing_direction: Vector3D
    pointing_active: bool
    left_hand_position: Vector3D
    right_hand_position: Vector3D
    doa_azimuth_rad: float
    vad_active: bool
    person_id: int = 0


@dataclass(slots=True)
class _HandHistory:
    last_position: Vector3D | None = None
    last_timestamp_ns: int = 0
    velocity: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 0.0))


def _ray_object_score(
    ray: Ray3D,
    center: Vector3D,
    half_extents: Vector3D,
    *,
    weight: float,
) -> float:
    radius = max(half_extents.x, half_extents.y, half_extents.z, 0.02)
    sphere_t = ray_sphere_intersection(ray, center, radius)
    box = AABB3D(
        Vector3D(
            center.x - half_extents.x,
            center.y - half_extents.y,
            center.z - half_extents.z,
        ),
        Vector3D(
            center.x + half_extents.x,
            center.y + half_extents.y,
            center.z + half_extents.z,
        ),
    )
    aabb_t = ray_aabb_intersection(ray, box)
    to_center = Vector3D(
        center.x - ray.origin.x,
        center.y - ray.origin.y,
        center.z - ray.origin.z,
    )
    dist = to_center.norm()
    if dist < 1e-6:
        return weight * 10.0
    to_center_n = to_center.normalized()
    cos_angle = max(-1.0, min(1.0, to_center_n.dot(ray.direction)))
    angle = math.acos(cos_angle)
    oc = to_center
    dir_n = ray.direction
    parallel = abs(oc.dot(dir_n))
    ortho = math.sqrt(max(0.0, oc.dot(oc) - parallel * parallel))
    score = weight * (-angle * 8.0 - ortho * 0.5)
    if sphere_t is not None or aabb_t is not None:
        score += HIT_BONUS * weight
    return score


def _softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    if scores.size == 0:
        return scores
    scaled = scores / SOFTMAX_TEMPERATURE
    scaled -= np.max(scaled)
    exp = np.exp(scaled)
    total = float(np.sum(exp))
    if total < 1e-12:
        return np.ones_like(scores) / scores.size
    return exp / total


@dataclass
class HumanStateAndAttentionSolver:
    """Hand kinematics, workspace entry prediction, and deictic joint attention."""

    _left_history: _HandHistory = field(default_factory=_HandHistory)
    _right_history: _HandHistory = field(default_factory=_HandHistory)

    def compute_hand_velocity(
        self,
        hand_position: Vector3D,
        timestamp_ns: int,
        *,
        which: str,
    ) -> Vector3D:
        hist = self._left_history if which == "left" else self._right_history
        if hist.last_position is None or hist.last_timestamp_ns <= 0:
            hist.last_position = hand_position
            hist.last_timestamp_ns = timestamp_ns
            hist.velocity = Vector3D(0.0, 0.0, 0.0)
            return hist.velocity
        dt = (timestamp_ns - hist.last_timestamp_ns) / 1e9
        if dt <= 1e-9:
            return hist.velocity
        vel = Vector3D(
            (hand_position.x - hist.last_position.x) / dt,
            (hand_position.y - hist.last_position.y) / dt,
            (hand_position.z - hist.last_position.z) / dt,
        )
        hist.velocity = vel
        hist.last_position = hand_position
        hist.last_timestamp_ns = timestamp_ns
        return vel

    def predict_workspace_entry(
        self,
        hand_position: Vector3D,
        hand_velocity: Vector3D,
        workspace_bounds_min: Vector3D,
        workspace_bounds_max: Vector3D,
        *,
        horizon_sec: float = 0.5,
        dt_sample_sec: float = 1.0 / 120.0,
    ) -> tuple[bool, float | None]:
        """Return whether the hand enters the workspace within ``horizon_sec`` and when."""
        if horizon_sec <= 0:
            return False, None
        t = 0.0
        while t <= horizon_sec + 1e-9:
            px = hand_position.x + hand_velocity.x * t
            py = hand_position.y + hand_velocity.y * t
            pz = hand_position.z + hand_velocity.z * t
            if (
                workspace_bounds_min.x <= px <= workspace_bounds_max.x
                and workspace_bounds_min.y <= py <= workspace_bounds_max.y
                and workspace_bounds_min.z <= pz <= workspace_bounds_max.z
            ):
                return True, t
            t += dt_sample_sec
        return False, None

    def solve_joint_attention(
        self,
        observation: HumanObservation3D,
        object_ids: NDArray[np.uint64],
        object_positions: NDArray[np.float32],
        object_half_extents: NDArray[np.float32],
        active_mask: NDArray[np.bool_] | None = None,
    ) -> tuple[NDArray[np.float64], int]:
        """Normalized attention over active 3D objects; returns probs and winning object_id."""
        n_objects = int(object_ids.shape[0])
        if n_objects == 0:
            return np.zeros(0, dtype=np.float64), -1
        n = min(n_objects, int(object_positions.shape[0]))
        gaze_ray = Ray3D(
            origin=observation.head_position,
            direction=observation.gaze_direction.normalized(),
        )
        pointing_ray = Ray3D(
            origin=observation.pointing_origin,
            direction=observation.pointing_direction.normalized(),
        )
        scores = np.zeros(n_objects, dtype=np.float64)
        for i in range(n):
            if active_mask is not None and not bool(active_mask[i]):
                continue
            if int(object_ids[i]) == 0:
                continue
            center = Vector3D(
                float(object_positions[i, 0]),
                float(object_positions[i, 1]),
                float(object_positions[i, 2]),
            )
            half = Vector3D(
                float(object_half_extents[i, 0]),
                float(object_half_extents[i, 1]),
                float(object_half_extents[i, 2]),
            )
            scores[i] += _ray_object_score(gaze_ray, center, half, weight=GAZE_WEIGHT)
            if observation.pointing_active:
                scores[i] += _ray_object_score(
                    pointing_ray,
                    center,
                    half,
                    weight=POINTING_WEIGHT,
                )
        probs = _softmax(scores)
        winner_idx = int(np.argmax(probs))
        winner_id = int(object_ids[winner_idx]) if int(object_ids[winner_idx]) != 0 else -1
        return probs, winner_id

    def select_speaker_person_id(
        self,
        observations: tuple[HumanObservation3D, ...],
        *,
        reference_azimuth_rad: float | None = None,
    ) -> int:
        """Fuse 4-mic DOA with 3D head positions for multi-person VAD selection."""
        if not observations:
            return -1
        ref_az = (
            reference_azimuth_rad
            if reference_azimuth_rad is not None
            else observations[0].doa_azimuth_rad
        )
        best_id = observations[0].person_id
        best_score = float("-inf")
        for obs in observations:
            head_az = math.atan2(obs.head_position.y, obs.head_position.x)
            doa_err = abs(
                math.atan2(math.sin(head_az - ref_az), math.cos(head_az - ref_az)),
            )
            vad_bonus = 1.0 if obs.vad_active else 0.0
            score = vad_bonus * 2.0 - doa_err
            if score > best_score:
                best_score = score
                best_id = obs.person_id
        return best_id

    def update_bus(
        self,
        bus: WorldStateRingBuffer,
        observation: HumanObservation3D,
        timestamp_ns: int,
        *,
        object_half_extents_default: tuple[float, float, float] = (0.05, 0.05, 0.05),
    ) -> tuple[NDArray[np.float64], int]:
        """Write human state and joint attention into the latest committed slot in-place."""
        latest = bus.read_latest()
        if latest is None:
            with bus.acquire_write_view() as view:
                view.timestamp_ns = timestamp_ns
                self._fill_human_fields(view, observation, timestamp_ns)
            return np.zeros(0, dtype=np.float64), -1

        left_vel = self.compute_hand_velocity(
            observation.left_hand_position,
            timestamp_ns,
            which="left",
        )
        right_vel = self.compute_hand_velocity(
            observation.right_hand_position,
            timestamp_ns,
            which="right",
        )

        with bus.acquire_write_view() as view:
            view.fill_from(latest.slot)
            view.timestamp_ns = timestamp_ns
            self._fill_human_fields(view, observation, timestamp_ns, left_vel, right_vel)
            objs = view.objects
            n = objs.shape[0]
            positions = objs["position_xyz"]
            ids = objs["object_id"]
            half = np.tile(np.array(object_half_extents_default, dtype=np.float32), (n, 1))
            active = ids != 0
            probs, winner_id = self.solve_joint_attention(
                observation,
                ids,
                positions,
                half,
                active_mask=active,
            )
            for i in range(min(len(probs), n)):
                objs["joint_attention_prob"][i] = np.float32(probs[i])
            view.raw["joint_attention_object_id"] = np.int32(winner_id)
        return probs, winner_id

    def _fill_human_fields(
        self,
        view: WorldStateSlotView,
        observation: HumanObservation3D,
        timestamp_ns: int,
        left_vel: Vector3D | None = None,
        right_vel: Vector3D | None = None,
    ) -> None:
        raw = view.raw
        raw["human_present"] = True
        raw["head_position_xyz"] = np.array(
            [
                observation.head_position.x,
                observation.head_position.y,
                observation.head_position.z,
            ],
            dtype=np.float32,
        )
        gaze = observation.gaze_direction.normalized()
        raw["gaze_ray_origin_xyz"] = raw["head_position_xyz"]
        raw["gaze_ray_dir_xyz"] = np.array([gaze.x, gaze.y, gaze.z], dtype=np.float32)
        pdir = observation.pointing_direction.normalized()
        raw["pointing_ray_origin_xyz"] = np.array(
            [
                observation.pointing_origin.x,
                observation.pointing_origin.y,
                observation.pointing_origin.z,
            ],
            dtype=np.float32,
        )
        raw["pointing_ray_dir_xyz"] = np.array([pdir.x, pdir.y, pdir.z], dtype=np.float32)
        raw["left_hand_xyz"] = np.array(
            [
                observation.left_hand_position.x,
                observation.left_hand_position.y,
                observation.left_hand_position.z,
            ],
            dtype=np.float32,
        )
        raw["right_hand_xyz"] = np.array(
            [
                observation.right_hand_position.x,
                observation.right_hand_position.y,
                observation.right_hand_position.z,
            ],
            dtype=np.float32,
        )
        lv = left_vel if left_vel is not None else Vector3D(0.0, 0.0, 0.0)
        rv = right_vel if right_vel is not None else Vector3D(0.0, 0.0, 0.0)
        raw["left_hand_vel_xyz"] = np.array([lv.x, lv.y, lv.z], dtype=np.float32)
        raw["right_hand_vel_xyz"] = np.array([rv.x, rv.y, rv.z], dtype=np.float32)
        raw["speaker_doa_azimuth_rad"] = np.float32(observation.doa_azimuth_rad)
        raw["vad_active"] = observation.vad_active
        _ = timestamp_ns
