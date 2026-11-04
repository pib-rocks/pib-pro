"""Layer 2 executive: surprise BT, multimodal scheduling, chest OAK-4 & ReSpeaker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pib_pro.common import ProactivityTier
from pib_pro.layer2_executive.communicative_act import (
    ActOutputState,
    ActPhase,
    BargeInCancelReport,
    MultimodalActScheduler,
    MultimodalCommunicativeAct,
    ScheduledActTimeline,
)
from pib_pro.layer2_executive.surprise_bt import (
    BTSkillStep,
    BTStatus,
    SurpriseGatedBehaviourTree,
    SurprisePreemptionResult,
    run_workbench_task_plan,
)

__all__ = [
    "ActOutputState",
    "ActPhase",
    "BTSkillStep",
    "BTStatus",
    "BargeInCancelReport",
    "CommunicativeAct",
    "CommunicativeActSchedulerProtocol",
    "MultimodalActScheduler",
    "MultimodalCommunicativeAct",
    "ScheduledActTimeline",
    "SurpriseBTProtocol",
    "SurpriseEvent",
    "SurpriseGatedBehaviourTree",
    "SurprisePreemptionResult",
    "run_workbench_task_plan",
]


@dataclass(frozen=True, slots=True)
class SurpriseEvent:
    """World-model prediction error crossing executive threshold."""

    event_id: str
    surprise_score: float
    source: str
    stamp_monotonic_s: float


@dataclass(frozen=True, slots=True)
class CommunicativeAct:
    """Scheduled speech/gesture act."""

    act_id: str
    tier: ProactivityTier
    utterance_key: str
    priority: int


class SurpriseBTProtocol(Protocol):
    """Behavior tree reacting to surprise."""

    def ingest(self, event: SurpriseEvent) -> None: ...

    def tick(self, stamp_monotonic_s: float) -> tuple[str, ...]: ...


class CommunicativeActSchedulerProtocol(Protocol):
    def enqueue(self, act: CommunicativeAct) -> None: ...

    def pop_due(self, stamp_monotonic_s: float) -> CommunicativeAct | None: ...
