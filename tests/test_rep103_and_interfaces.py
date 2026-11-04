"""Comprehensive tests for REP-103, geometry helpers, and architecture contracts."""

from __future__ import annotations

import math

import pytest

from pib_pro.common import (
    AABB3D,
    FRAME_BASE_LINK,
    FRAME_CAMERA_OAK4_OPTICAL,
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
from pib_pro.layer0_reflex import (
    ActuatorCommand,
    ComplianceEvent,
    ComplianceEventKind,
    SafetyEnvelopeConfig,
    SomaticReflexProtocol,
)
from pib_pro.layer1_autopilot import (
    ActionChunkProposal,
    HybridVisuomotorProtocol,
    ScoredActionChunk,
    SociomotorAutopilotProtocol,
    SociomotorTarget,
)
from pib_pro.layer2_executive import (
    CommunicativeAct,
    CommunicativeActSchedulerProtocol,
    SurpriseBTProtocol,
    SurpriseEvent,
)
from pib_pro.layer3_cognition import (
    OmniDialogueProtocol,
    OmniDialogueTurn,
    SubgoalPlan,
    System3DeliberationProtocol,
)
from pib_pro.perception_oak4 import HumanState3D, ScenePerceptionProtocol, TrackedObject4D
from pib_pro.telemetry_gateway import TelemetryGatewayProtocol, TelemetrySnapshot
from pib_pro.world_model import HJEPAWorldModelProtocol, LatentStateVector
from pib_pro.world_state_bus import WorldStateBusProtocol, WorldStateSlotContract


def test_standard_frames_and_enums() -> None:
    assert FRAME_BASE_LINK in STANDARD_FRAMES
    assert FRAME_CAMERA_OAK4_OPTICAL in STANDARD_FRAMES
    assert FloorState.STABLE.name == "STABLE"
    assert ProactivityTier.T0_SILENT.value < ProactivityTier.T3_FORBIDDEN.value


def test_vector3d_math_and_non_finite() -> None:
    a = Vector3D(1.0, 0.0, 0.0)
    b = Vector3D(0.0, 1.0, 0.0)
    assert a.dot(b) == 0.0
    assert a.normalized().x == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite"):
        Vector3D(1.0, math.nan, 0.0)


def test_quaternion_unit_validation() -> None:
    Quaternion.identity()
    with pytest.raises(ValueError, match="unit norm"):
        Quaternion(1.0, 1.0, 1.0, 1.0)


def test_pose3d_requires_frame() -> None:
    pose = Pose3D(
        FRAME_BASE_LINK,
        Vector3D(0.0, 0.0, 0.0),
        Quaternion.identity(),
    )
    assert pose.frame_id == FRAME_BASE_LINK
    with pytest.raises(ValueError, match="frame_id"):
        Pose3D("", Vector3D(0, 0, 0), Quaternion.identity())


def test_ray_direction_normalized_on_init() -> None:
    ray = Ray3D(Vector3D(0, 0, 0), Vector3D(3.0, 4.0, 0.0))
    assert ray.direction.x == pytest.approx(0.6)
    assert ray.direction.y == pytest.approx(0.8)


def test_ray_sphere_intersection() -> None:
    ray = Ray3D(Vector3D(0, 0, 0), Vector3D(1, 0, 0))
    center = Vector3D(5.0, 0.0, 0.0)
    t = ray_sphere_intersection(ray, center, 1.0)
    assert t == pytest.approx(4.0)
    miss_center = Vector3D(0.0, 5.0, 0.0)
    assert ray_sphere_intersection(ray, miss_center, 0.1) is None


def test_ray_aabb_intersection() -> None:
    ray = Ray3D(Vector3D(-2, 0.5, 0.5), Vector3D(1, 0, 0))
    box = AABB3D(Vector3D(0, 0, 0), Vector3D(1, 1, 1))
    t = ray_aabb_intersection(ray, box)
    assert t == pytest.approx(2.0)


def test_joint_state_snapshot_validation() -> None:
    snap = JointStateSnapshot(("j1",), (0.1,), (0.0,))
    assert snap.names[0] == "j1"
    with pytest.raises(ValueError, match="length mismatch"):
        JointStateSnapshot(("j1",), (0.1,), (0.0, 0.0))


def test_architecture_protocol_imports() -> None:
    """Ensure all eight packages export core contracts."""

    class _Bus(WorldStateBusProtocol):
        def register_slot(self, contract: WorldStateSlotContract) -> None:
            pass

        def write_slot(self, slot_id: str, payload: object, stamp_monotonic_s: float) -> None:
            pass

        def read_latest(self, slot_id: str) -> tuple[object, float] | None:
            return None

        def snapshot_pose(self) -> Pose3D:
            return Pose3D(FRAME_BASE_LINK, Vector3D(0, 0, 0), Quaternion.identity())

    class _Perception(ScenePerceptionProtocol):
        def perceive(self, stamp_monotonic_s: float) -> tuple[TrackedObject4D, ...]:
            return ()

        def estimate_humans(self, stamp_monotonic_s: float) -> tuple[HumanState3D, ...]:
            return ()

    bus: WorldStateBusProtocol = _Bus()
    perception: ScenePerceptionProtocol = _Perception()
    assert bus.read_latest("x") is None
    assert perception.perceive(0.0) == ()

    # Dataclass contracts constructible
    SafetyEnvelopeConfig(2.0, 0.5, 0.02)
    ActuatorCommand(("j",), (0.0,), 1.0)
    ComplianceEvent(ComplianceEventKind.LIMIT, "ok", Vector3D(0, 0, 1))
    SociomotorTarget(
        Pose3D(FRAME_BASE_LINK, Vector3D(0, 0, 0), Quaternion.identity()),
        Vector3D(0, 0, 0),
        ProactivityTier.T1_MINIMAL,
    )
    chunk = ActionChunkProposal("c1", 0.5, ())
    ScoredActionChunk(chunk, 1.0, "test")
    SurpriseEvent("e1", 0.9, "wm", 1.0)
    CommunicativeAct("a1", ProactivityTier.T2_NORMAL, "hello", 1)
    OmniDialogueTurn("user", "hi", 0.0)
    SubgoalPlan("p1", ("g1",), 5.0)
    TelemetrySnapshot(0.0, 10.0, 20.0, 30.0, 250.0, 40.0, 50.0)
    import numpy as np

    LatentStateVector(4, np.zeros(4, dtype=np.float32))

    # Protocol types referenced
    _: SomaticReflexProtocol
    _: SociomotorAutopilotProtocol
    _: HybridVisuomotorProtocol
    _: SurpriseBTProtocol
    _: CommunicativeActSchedulerProtocol
    _: OmniDialogueProtocol
    _: System3DeliberationProtocol
    _: HJEPAWorldModelProtocol
    _: TelemetryGatewayProtocol
