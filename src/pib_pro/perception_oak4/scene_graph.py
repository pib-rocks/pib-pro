"""OAK-4 metric semantic scene graph with 3D tracking and bus publication."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pib_pro.common.rep103 import Quaternion, Vector3D

if TYPE_CHECKING:
    from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer

DEFAULT_OCCLUSION_TTL_SEC: float = 15.0
DEFAULT_STALENESS_THRESHOLD_SEC: float = 5.0
EMBEDDING_DIM: int = 32
DISTANCE_WEIGHT: float = 1.0
APPEARANCE_WEIGHT: float = 0.35
MAX_MATCH_DISTANCE_M: float = 0.45
APPEARANCE_MATCH_MIN: float = 0.55


@dataclass(frozen=True, slots=True)
class Oak4CameraIntrinsics:
    """Pinhole intrinsics for OAK-4 depth/RGB alignment."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Oak4CameraIntrinsics: width and height must be positive")
        for name, val in (("fx", self.fx), ("fy", self.fy), ("cx", self.cx), ("cy", self.cy)):
            if not math.isfinite(val):
                raise ValueError(f"Oak4CameraIntrinsics: {name} must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Oak4CameraIntrinsics: fx and fy must be positive")


@dataclass(frozen=True, slots=True)
class PoseTransform3D:
    """Rigid transform from OAK-4 optical frame to REP-103 ``base_link``."""

    translation_base_from_optical: Vector3D
    rotation_base_from_optical: NDArray[np.float64]

    def __post_init__(self) -> None:
        rot = np.asarray(self.rotation_base_from_optical, dtype=np.float64)
        if rot.shape != (3, 3):
            raise ValueError("PoseTransform3D: rotation must be 3x3")
        if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-5):
            raise ValueError("PoseTransform3D: rotation must be orthonormal")

    @classmethod
    def oak4_optical_to_rep103_base(cls) -> PoseTransform3D:
        """Standard OAK optical (X right, Y down, Z forward) -> REP-103 base_link (head mount)."""
        rot = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return cls(
            translation_base_from_optical=Vector3D(0.0, 0.0, 0.0),
            rotation_base_from_optical=rot,
        )

    @classmethod
    def oak4_chest_optical_to_rep103_base(
        cls,
        chest_x_m: float = 0.08,
        chest_y_m: float = 0.0,
        chest_z_m: float = 0.82,
        pitch_down_rad: float = 0.1745,
    ) -> PoseTransform3D:
        """Chest-mounted OAK 4 D Pro fixed to ``base_link`` (invariant to head/neck pose)."""
        base_rot = cls.oak4_optical_to_rep103_base().rotation_base_from_optical
        cp = math.cos(pitch_down_rad)
        sp = math.sin(pitch_down_rad)
        pitch_rot = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float64,
        )
        rot = pitch_rot @ base_rot
        return cls(
            translation_base_from_optical=Vector3D(chest_x_m, chest_y_m, chest_z_m),
            rotation_base_from_optical=rot,
        )

    def transform_point_optical_to_base(self, x_opt: float, y_opt: float, z_opt: float) -> Vector3D:
        p_opt = np.array([x_opt, y_opt, z_opt], dtype=np.float64)
        p_base = self.rotation_base_from_optical @ p_opt
        p_base += self.translation_base_from_optical.as_array()
        return Vector3D(float(p_base[0]), float(p_base[1]), float(p_base[2]))


@dataclass(frozen=True, slots=True)
class Detection2DWithDepth:
    """Single OAK-4 detection with metric depth and appearance embedding."""

    class_id: int
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    depth_m: float
    confidence: float
    appearance_embedding: NDArray[np.float32]
    half_extents_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.depth_m <= 0 or not math.isfinite(self.depth_m):
            raise ValueError("Detection2DWithDepth: depth_m must be positive and finite")
        emb = np.asarray(self.appearance_embedding, dtype=np.float32).reshape(-1)
        if emb.size == 0 or not np.all(np.isfinite(emb)):
            raise ValueError("Detection2DWithDepth: appearance_embedding must be finite")

    def bbox_center_uv(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_xyxy
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)

    def deproject_to_rep103(
        self,
        intrinsics: Oak4CameraIntrinsics,
        extrinsic: PoseTransform3D,
    ) -> Vector3D:
        """Deproject bbox center + depth into REP-103 ``base_link`` (<= 1.5 cm target)."""
        u, v = self.bbox_center_uv()
        x_opt = (u - intrinsics.cx) * self.depth_m / intrinsics.fx
        y_opt = (v - intrinsics.cy) * self.depth_m / intrinsics.fy
        z_opt = self.depth_m
        return extrinsic.transform_point_optical_to_base(x_opt, y_opt, z_opt)


@dataclass(frozen=True, slots=True)
class SelfArmMaskConfig:
    """Parameters for masking self-arm false positives in chest OAK-4 detections."""

    intrinsics: Oak4CameraIntrinsics
    extrinsic: PoseTransform3D
    persistent_track_positions: tuple[Vector3D, ...] = ()
    track_match_radius_m: float = 0.12


def filter_self_arm_occlusions(
    detections: tuple[Detection2DWithDepth, ...],
    left_ee_xyz: Vector3D,
    right_ee_xyz: Vector3D,
    config: SelfArmMaskConfig,
    *,
    arm_mask_radius_m: float = 0.065,
) -> tuple[Detection2DWithDepth, ...]:
    """Drop detections that coincide with the robot's own arms in 3D."""
    cam_pos = config.extrinsic.translation_base_from_optical
    kept: list[Detection2DWithDepth] = []
    for det in detections:
        pos = det.deproject_to_rep103(config.intrinsics, config.extrinsic)
        d_left = _euclidean(pos, left_ee_xyz)
        d_right = _euclidean(pos, right_ee_xyz)
        near_arm = min(d_left, d_right) <= arm_mask_radius_m
        if not near_arm:
            kept.append(det)
            continue
        matched_track = False
        for track_pos in config.persistent_track_positions:
            if _euclidean(pos, track_pos) <= config.track_match_radius_m:
                matched_track = True
                break
        arm_ref = left_ee_xyz if d_left <= d_right else right_ee_xyz
        det_range = _euclidean(pos, cam_pos)
        arm_range = _euclidean(arm_ref, cam_pos)
        if matched_track and det_range >= arm_range:
            kept.append(det)
            continue
        # Self-arm false positive (untracked blob on the hand / forearm).
    return tuple(kept)


@dataclass(slots=True)
class _TrackedObject:
    object_id: int
    class_id: int
    label: str
    position: Vector3D
    velocity: Vector3D
    orientation: Quaternion
    half_extents: Vector3D
    confidence: float
    appearance_embedding: NDArray[np.float32]
    last_seen_ns: int
    created_ns: int

    def predicted_position(self, timestamp_ns: int) -> Vector3D:
        dt = (timestamp_ns - self.last_seen_ns) / 1e9
        if dt <= 0:
            return self.position
        return Vector3D(
            self.position.x + self.velocity.x * dt,
            self.position.y + self.velocity.y * dt,
            self.position.z + self.velocity.z * dt,
        )

    def staleness_sec(self, now_ns: int) -> float:
        return max(0.0, (now_ns - self.last_seen_ns) / 1e9)

    def is_stale(self, now_ns: int, threshold_sec: float) -> bool:
        return self.staleness_sec(now_ns) >= threshold_sec


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    va = a.astype(np.float64)
    vb = b.astype(np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _euclidean(a: Vector3D, b: Vector3D) -> float:
    return math.sqrt(
        (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2,
    )


def _match_cost(
    track: _TrackedObject,
    detection: Detection2DWithDepth,
    det_pos: Vector3D,
    predicted: Vector3D,
) -> float:
    dist = _euclidean(predicted, det_pos)
    if dist > MAX_MATCH_DISTANCE_M:
        return float("inf")
    app = _cosine_similarity(track.appearance_embedding, detection.appearance_embedding)
    if app < APPEARANCE_MATCH_MIN:
        return float("inf")
    return DISTANCE_WEIGHT * dist + APPEARANCE_WEIGHT * (1.0 - app)


def _greedy_assignment(cost: NDArray[np.float64]) -> list[tuple[int, int]]:
    """Greedy minimum-cost matching (Hungarian fallback for small bipartite graphs)."""
    n_tracks, n_dets = cost.shape
    pairs: list[tuple[int, int, float]] = []
    for i in range(n_tracks):
        for j in range(n_dets):
            c = float(cost[i, j])
            if math.isfinite(c):
                pairs.append((i, j, c))
    pairs.sort(key=lambda x: x[2])
    used_t: set[int] = set()
    used_d: set[int] = set()
    result: list[tuple[int, int]] = []
    for i, j, _ in pairs:
        if i in used_t or j in used_d:
            continue
        used_t.add(i)
        used_d.add(j)
        result.append((i, j))
    return result


@dataclass
class MetricSemanticSceneGraph:
    """30 Hz 3D scene graph with occlusion-tolerant identity persistence."""

    intrinsics: Oak4CameraIntrinsics
    extrinsic: PoseTransform3D
    occlusion_ttl_sec: float = DEFAULT_OCCLUSION_TTL_SEC
    staleness_threshold_sec: float = DEFAULT_STALENESS_THRESHOLD_SEC
    _tracks: list[_TrackedObject] = field(default_factory=list)
    _next_object_id: int = 1
    _last_timestamp_ns: int = 0

    def active_tracks(self) -> tuple[_TrackedObject, ...]:
        return tuple(self._tracks)

    def _prune_expired(self, now_ns: int) -> None:
        kept: list[_TrackedObject] = []
        for tr in self._tracks:
            if tr.staleness_sec(now_ns) <= self.occlusion_ttl_sec:
                kept.append(tr)
        self._tracks = kept

    def _update_track_kinematics(
        self,
        track: _TrackedObject,
        new_pos: Vector3D,
        timestamp_ns: int,
    ) -> None:
        dt = (timestamp_ns - track.last_seen_ns) / 1e9
        if dt > 1e-9:
            track.velocity = Vector3D(
                (new_pos.x - track.position.x) / dt,
                (new_pos.y - track.position.y) / dt,
                (new_pos.z - track.position.z) / dt,
            )
        track.position = new_pos
        track.last_seen_ns = timestamp_ns

    def update_and_publish(
        self,
        detections: tuple[Detection2DWithDepth, ...],
        timestamp_ns: int,
        frame_id: int,
        bus: WorldStateRingBuffer | None = None,
    ) -> tuple[_TrackedObject, ...]:
        """Associate detections, predict occluded tracks, optionally write the bus slot."""
        if timestamp_ns < self._last_timestamp_ns:
            raise ValueError("MetricSemanticSceneGraph: timestamp_ns must be non-decreasing")
        self._last_timestamp_ns = timestamp_ns
        self._prune_expired(timestamp_ns)

        det_positions: list[Vector3D] = [
            d.deproject_to_rep103(self.intrinsics, self.extrinsic) for d in detections
        ]

        if self._tracks and detections:
            n_t = len(self._tracks)
            n_d = len(detections)
            cost = np.full((n_t, n_d), np.inf, dtype=np.float64)
            for i, tr in enumerate(self._tracks):
                pred = tr.predicted_position(timestamp_ns)
                for j, det in enumerate(detections):
                    cost[i, j] = _match_cost(tr, det, det_positions[j], pred)
            matches = _greedy_assignment(cost)
            matched_tracks = {i for i, _ in matches}
            matched_dets = {j for _, j in matches}
            for i, j in matches:
                tr = self._tracks[i]
                det = detections[j]
                tr.class_id = det.class_id
                tr.label = det.label
                tr.confidence = det.confidence
                tr.appearance_embedding = np.asarray(det.appearance_embedding, dtype=np.float32)
                tr.half_extents = Vector3D(*det.half_extents_xyz)
                self._update_track_kinematics(tr, det_positions[j], timestamp_ns)
            for j, det in enumerate(detections):
                if j not in matched_dets:
                    self._spawn_track(det, det_positions[j], timestamp_ns)
            for i, tr in enumerate(self._tracks):
                if i not in matched_tracks:
                    _ = tr.predicted_position(timestamp_ns)
        elif detections:
            for det, pos in zip(detections, det_positions, strict=True):
                self._spawn_track(det, pos, timestamp_ns)
        else:
            for tr in self._tracks:
                _ = tr.predicted_position(timestamp_ns)

        if bus is not None:
            self._write_bus(bus, timestamp_ns, frame_id)
        return tuple(self._tracks)

    def _spawn_track(
        self,
        det: Detection2DWithDepth,
        position: Vector3D,
        timestamp_ns: int,
    ) -> None:
        emb = np.asarray(det.appearance_embedding, dtype=np.float32)
        track = _TrackedObject(
            object_id=self._next_object_id,
            class_id=det.class_id,
            label=det.label,
            position=position,
            velocity=Vector3D(0.0, 0.0, 0.0),
            orientation=Quaternion.identity(),
            half_extents=Vector3D(*det.half_extents_xyz),
            confidence=det.confidence,
            appearance_embedding=emb.copy(),
            last_seen_ns=timestamp_ns,
            created_ns=timestamp_ns,
        )
        self._next_object_id += 1
        self._tracks.append(track)

    def _write_bus(
        self,
        bus: WorldStateRingBuffer,
        timestamp_ns: int,
        frame_id: int,
    ) -> None:
        from pib_pro.world_state_bus.schema import MAX_OBJECTS

        with bus.acquire_write_view() as view:
            view.timestamp_ns = timestamp_ns
            view.frame_id = frame_id
            objs = view.objects
            objs["object_id"] = 0
            objs["class_id"] = 0
            objs["confidence"] = 0.0
            objs["joint_attention_prob"] = 0.0
            objs["last_seen_ns"] = 0
            objs["staleness_sec"] = 0.0
            for i in range(MAX_OBJECTS):
                objs["position_xyz"][i] = 0.0
                objs["velocity_xyz"][i] = 0.0
                objs["orientation_xyzw"][i] = np.array([0, 0, 0, 1], dtype=np.float32)
            count = min(len(self._tracks), MAX_OBJECTS)
            for i in range(count):
                tr = self._tracks[i]
                stale = tr.staleness_sec(timestamp_ns)
                objs["object_id"][i] = np.uint64(tr.object_id)
                objs["class_id"][i] = np.uint32(tr.class_id)
                objs["position_xyz"][i] = np.array(
                    [tr.position.x, tr.position.y, tr.position.z],
                    dtype=np.float32,
                )
                objs["velocity_xyz"][i] = np.array(
                    [tr.velocity.x, tr.velocity.y, tr.velocity.z],
                    dtype=np.float32,
                )
                objs["orientation_xyzw"][i] = np.array(
                    [
                        tr.orientation.x,
                        tr.orientation.y,
                        tr.orientation.z,
                        tr.orientation.w,
                    ],
                    dtype=np.float32,
                )
                objs["confidence"][i] = np.float32(tr.confidence)
                objs["last_seen_ns"][i] = np.int64(tr.last_seen_ns)
                objs["staleness_sec"][i] = np.float32(stale)
                objs["joint_attention_prob"][i] = np.float32(0.0)
