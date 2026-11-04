"""Layer 1 sociomotor autopilot and hybrid visuomotor contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pib_pro.common import Pose3D, ProactivityTier, Vector3D


@dataclass(frozen=True, slots=True)
class SociomotorTarget:
    """Socially grounded motion intent."""

    target_pose: Pose3D
    approach_offset_m: Vector3D
    tier: ProactivityTier


@dataclass(frozen=True, slots=True)
class ActionChunkProposal:
    """Short horizon motion chunk before scoring."""

    chunk_id: str
    duration_s: float
    waypoints: tuple[Pose3D, ...]


@dataclass(frozen=True, slots=True)
class ScoredActionChunk:
    proposal: ActionChunkProposal
    score: float
    rationale: str


class SociomotorAutopilotProtocol(Protocol):
    def plan(
        self,
        target: SociomotorTarget,
        stamp_monotonic_s: float,
    ) -> tuple[ScoredActionChunk, ...]: ...


class HybridVisuomotorProtocol(Protocol):
    def fuse_visual_servo(
        self,
        chunks: tuple[ScoredActionChunk, ...],
        optical_error: Vector3D,
    ) -> ScoredActionChunk: ...


from pib_pro.layer1_autopilot.procedural_memory import (  # noqa: E402
    LeRobotEpisodeFrame,
    LeRobotV2Episode,
    OfflineSkillConsolidator,
    ProceduralMemoryRecorder,
)
from pib_pro.layer1_autopilot.sociomotor import (  # noqa: E402
    JerkLimitedPCHIPFilter,
    NeckEyeTarget,
    SociomotorAutopilot,
)
from pib_pro.layer1_autopilot.visuomotor_hybrid import (  # noqa: E402
    GR00TActionProposer,
    HybridVisuomotorController,
    HybridVisuomotorStepResult,
)

__all__ = [
    "ActionChunkProposal",
    "GR00TActionProposer",
    "HybridVisuomotorController",
    "HybridVisuomotorProtocol",
    "HybridVisuomotorStepResult",
    "JerkLimitedPCHIPFilter",
    "LeRobotEpisodeFrame",
    "LeRobotV2Episode",
    "NeckEyeTarget",
    "OfflineSkillConsolidator",
    "ProceduralMemoryRecorder",
    "ScoredActionChunk",
    "SociomotorAutopilot",
    "SociomotorAutopilotProtocol",
    "SociomotorTarget",
]
