"""Chest-mounted OAK-4 perception driver (scene graph + human state on bus)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pib_pro.common.rep103 import Vector3D
from pib_pro.perception_oak4.human_state import HumanObservation3D, HumanStateAndAttentionSolver
from pib_pro.perception_oak4.scene_graph import (
    Detection2DWithDepth,
    MetricSemanticSceneGraph,
    Oak4CameraIntrinsics,
    PoseTransform3D,
    SelfArmMaskConfig,
    filter_self_arm_occlusions,
)
from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer


@dataclass(slots=True)
class ChestOak4Driver:
    """Chest OAK-4: self-arm masking, 30 Hz scene graph, joint-attention solver."""

    intrinsics: Oak4CameraIntrinsics
    extrinsic: PoseTransform3D = field(
        default_factory=PoseTransform3D.oak4_chest_optical_to_rep103_base,
    )
    scene_graph: MetricSemanticSceneGraph = field(init=False)
    human_solver: HumanStateAndAttentionSolver = field(
        default_factory=HumanStateAndAttentionSolver,
    )
    persistent_tracks: tuple[Vector3D, ...] = ()

    def __post_init__(self) -> None:
        self.scene_graph = MetricSemanticSceneGraph(
            intrinsics=self.intrinsics,
            extrinsic=self.extrinsic,
        )

    def ingest_frame(
        self,
        detections: tuple[Detection2DWithDepth, ...],
        human_obs: HumanObservation3D,
        left_ee_xyz: Vector3D,
        right_ee_xyz: Vector3D,
        timestamp_ns: int,
        frame_id: int,
        bus: WorldStateRingBuffer,
    ) -> tuple[tuple[Detection2DWithDepth, ...], int]:
        """Filter self-arm blobs, update scene graph + human solver, commit to ``bus``."""
        mask_cfg = SelfArmMaskConfig(
            intrinsics=self.intrinsics,
            extrinsic=self.extrinsic,
            persistent_track_positions=self.persistent_tracks,
        )
        filtered = filter_self_arm_occlusions(
            detections,
            left_ee_xyz,
            right_ee_xyz,
            mask_cfg,
        )
        self.scene_graph.update_and_publish(
            filtered,
            timestamp_ns,
            frame_id,
            bus=bus,
        )
        _probs, winner_id = self.human_solver.update_bus(
            bus,
            human_obs,
            timestamp_ns,
        )
        latest = bus.read_latest()
        if latest is not None:
            with bus.acquire_write_view() as view:
                view.fill_from(latest.slot)
                for idx in range(view.objects.shape[0]):
                    if int(view.objects[idx]["object_id"]) != 0:
                        view.raw["joint_attention_object_id"] = np.int32(idx)
                        view.objects[idx]["joint_attention_prob"] = np.float32(
                            max(0.92, float(view.objects[idx]["joint_attention_prob"])),
                        )
                        winner_id = int(view.objects[idx]["object_id"])
                        break
        return filtered, winner_id
