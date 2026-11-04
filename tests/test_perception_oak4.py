"""Tests for OAK-4 scene graph, human attention, and deprojection accuracy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pib_pro.common.rep103 import Vector3D
from pib_pro.perception_oak4 import (
    Detection2DWithDepth,
    HumanObservation3D,
    HumanStateAndAttentionSolver,
    MetricSemanticSceneGraph,
    Oak4CameraIntrinsics,
    PoseTransform3D,
)
from pib_pro.world_state_bus import BUS_PERIOD_NS, WorldStateRingBuffer


def _intrinsics() -> Oak4CameraIntrinsics:
    return Oak4CameraIntrinsics(
        fx=700.0,
        fy=700.0,
        cx=640.0,
        cy=360.0,
        width=1280,
        height=720,
    )


def _base_to_optical(base: Vector3D) -> tuple[float, float, float]:
    rot = PoseTransform3D.oak4_optical_to_rep103_base().rotation_base_from_optical
    p_opt = rot.T @ base.as_array()
    return float(p_opt[0]), float(p_opt[1]), float(p_opt[2])


def _detection_at_base(
    base_pos: Vector3D,
    intrinsics: Oak4CameraIntrinsics,
    *,
    class_id: int = 1,
    embedding: np.ndarray | None = None,
) -> Detection2DWithDepth:
    x_opt, y_opt, z_opt = _base_to_optical(base_pos)
    u = intrinsics.fx * x_opt / z_opt + intrinsics.cx
    v = intrinsics.fy * y_opt / z_opt + intrinsics.cy
    half = 20.0
    emb = embedding if embedding is not None else np.ones(32, dtype=np.float32)
    return Detection2DWithDepth(
        class_id=class_id,
        label="obj",
        bbox_xyxy=(u - half, v - half, u + half, v + half),
        depth_m=z_opt,
        confidence=0.95,
        appearance_embedding=emb,
        half_extents_xyz=(0.04, 0.04, 0.04),
    )


def test_oak4_deprojection_rep103_within_15mm() -> None:
    intrinsics = _intrinsics()
    extrinsic = PoseTransform3D.oak4_optical_to_rep103_base()
    errors: list[float] = []
    for i in range(12):
        truth = Vector3D(0.55 + 0.02 * i, -0.12 + 0.03 * i, 0.08 + 0.01 * i)
        det = _detection_at_base(truth, intrinsics, class_id=i)
        est = det.deproject_to_rep103(intrinsics, extrinsic)
        err = math.sqrt(
            (est.x - truth.x) ** 2 + (est.y - truth.y) ** 2 + (est.z - truth.z) ** 2,
        )
        errors.append(err)
    assert max(errors) <= 0.015


def test_occlusion_identity_persistence_and_staleness() -> None:
    intrinsics = _intrinsics()
    extrinsic = PoseTransform3D.oak4_optical_to_rep103_base()
    graph = MetricSemanticSceneGraph(
        intrinsics=intrinsics,
        extrinsic=extrinsic,
        occlusion_ttl_sec=15.0,
        staleness_threshold_sec=30.0,
    )
    rng = np.random.default_rng(7)
    embeddings = [rng.standard_normal(32).astype(np.float32) for _ in range(10)]
    for i in range(10):
        emb = embeddings[i]
        emb /= np.linalg.norm(emb) + 1e-6
        pos = Vector3D(0.5, -0.4 + i * 0.08, 0.1)
        det = _detection_at_base(pos, intrinsics, class_id=i + 1, embedding=emb)
        graph.update_and_publish((det,), i * BUS_PERIOD_NS, i)

    original_ids = {tr.object_id for tr in graph.active_tracks()}
    assert len(original_ids) == 10
    t0 = 10 * BUS_PERIOD_NS
    for frame in range(1, 301):
        ts = t0 + frame * BUS_PERIOD_NS
        graph.update_and_publish((), ts, 100 + frame)

    redetect_ts = t0 + 301 * BUS_PERIOD_NS
    redetections: list[Detection2DWithDepth] = []
    for i in range(10):
        pos = Vector3D(0.5, -0.4 + i * 0.08, 0.1)
        redetections.append(
            _detection_at_base(pos, intrinsics, class_id=i + 1, embedding=embeddings[i]),
        )
    graph.update_and_publish(tuple(redetections), redetect_ts, 500)
    survived = sum(1 for tr in graph.active_tracks() if tr.object_id in original_ids)
    assert survived >= 9

    stale_ts = redetect_ts + int(30.0 / (BUS_PERIOD_NS / 1e9)) * BUS_PERIOD_NS
    graph.update_and_publish((), stale_ts, 900)
    stale_flags = [tr.is_stale(stale_ts, 30.0) for tr in graph.active_tracks()]
    assert all(stale_flags)


def test_joint_attention_deictic_trials_and_workspace_prediction() -> None:
    solver = HumanStateAndAttentionSolver()
    head = Vector3D(0.0, 0.0, 1.5)
    distance = 0.6
    object_count = 5
    centers_y = [-0.2, -0.1, 0.0, 0.1, 0.2]
    ids = np.arange(1, object_count + 1, dtype=np.uint64)
    positions = np.zeros((32, 3), dtype=np.float32)
    half = np.full((32, 3), 0.05, dtype=np.float32)
    for i, y in enumerate(centers_y):
        positions[i] = np.array([distance, y, 0.05], dtype=np.float32)

    correct = 0
    trials = 20
    for trial in range(trials):
        target_idx = trial % object_count
        target = Vector3D(float(positions[target_idx, 0]), float(positions[target_idx, 1]), 0.05)
        direction = Vector3D(
            target.x - head.x,
            target.y - head.y,
            target.z - head.z,
        ).normalized()
        obs = HumanObservation3D(
            head_position=head,
            gaze_direction=direction,
            pointing_origin=head,
            pointing_direction=direction,
            pointing_active=True,
            left_hand_position=Vector3D(0.0, 0.0, 1.0),
            right_hand_position=Vector3D(0.0, 0.0, 1.0),
            doa_azimuth_rad=math.atan2(direction.y, direction.x),
            vad_active=True,
            person_id=0,
        )
        _, winner_id = solver.solve_joint_attention(obs, ids, positions, half)
        if winner_id == int(ids[target_idx]):
            correct += 1
    assert correct / trials >= 0.9

    hand_pos = Vector3D(0.85, 0.0, 0.25)
    hand_vel = Vector3D(-0.35, 0.0, 0.0)
    ws_min = Vector3D(0.2, -0.3, 0.0)
    ws_max = Vector3D(0.7, 0.3, 0.5)
    will_enter, t_entry = solver.predict_workspace_entry(
        hand_pos,
        hand_vel,
        ws_min,
        ws_max,
        horizon_sec=0.5,
    )
    assert will_enter
    assert t_entry is not None
    assert t_entry >= 0.4


def test_scene_graph_writes_bus_objects() -> None:
    ring = WorldStateRingBuffer(capacity=8)
    intrinsics = _intrinsics()
    graph = MetricSemanticSceneGraph(
        intrinsics=intrinsics,
        extrinsic=PoseTransform3D.oak4_optical_to_rep103_base(),
    )
    det = _detection_at_base(Vector3D(0.5, 0.0, 0.1), intrinsics)
    graph.update_and_publish((det,), BUS_PERIOD_NS, 1, bus=ring)
    latest = ring.read_latest()
    assert latest is not None
    assert int(latest.slot["objects"][0]["object_id"]) == 1
    assert float(latest.slot["objects"][0]["position_xyz"][0]) == pytest.approx(0.5, abs=0.02)
