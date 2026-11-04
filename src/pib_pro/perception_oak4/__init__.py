"""OAK-4 scene perception types and protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pib_pro.common import Pose3D, Vector3D
from pib_pro.perception_oak4.human_state import (
    HumanObservation3D,
    HumanStateAndAttentionSolver,
    SeeedMicArrayConfig,
    seeed_doa_deg_to_rep103_azimuth_rad,
)
from pib_pro.perception_oak4.scene_graph import (
    Detection2DWithDepth,
    MetricSemanticSceneGraph,
    Oak4CameraIntrinsics,
    PoseTransform3D,
    SelfArmMaskConfig,
    filter_self_arm_occlusions,
)

__all__ = [
    "Detection2DWithDepth",
    "HumanObservation3D",
    "HumanState3D",
    "HumanStateAndAttentionSolver",
    "MetricSemanticSceneGraph",
    "Oak4CameraIntrinsics",
    "PoseTransform3D",
    "ScenePerceptionProtocol",
    "SeeedMicArrayConfig",
    "SelfArmMaskConfig",
    "filter_self_arm_occlusions",
    "seeed_doa_deg_to_rep103_azimuth_rad",
    "TrackedObject4D",
]


@dataclass(frozen=True, slots=True)
class TrackedObject4D:
    """Object track with 3D pose and coarse 4D extent (time + spatial)."""

    track_id: int
    label: str
    pose: Pose3D
    extent_m: Vector3D
    last_seen_monotonic_s: float


@dataclass(frozen=True, slots=True)
class HumanState3D:
    """Estimated human skeleton / attention anchor in REP-103."""

    person_id: int
    head_pose: Pose3D
    gaze_direction: Vector3D
    speaking: bool


class ScenePerceptionProtocol(Protocol):
    """OAK-4 depth + tracking pipeline interface."""

    def perceive(self, stamp_monotonic_s: float) -> tuple[TrackedObject4D, ...]: ...

    def estimate_humans(self, stamp_monotonic_s: float) -> tuple[HumanState3D, ...]: ...
