"""Shared Layer-3 dataclass contracts and protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OmniDialogueTurn:
    role: str
    text: str
    stamp_monotonic_s: float


@dataclass(frozen=True, slots=True)
class SubgoalPlan:
    plan_id: str
    subgoals: tuple[str, ...]
    horizon_s: float


class OmniDialogueProtocol(Protocol):
    def submit_user_turn(self, turn: OmniDialogueTurn) -> None: ...

    def generate_assistant_turn(self, stamp_monotonic_s: float) -> OmniDialogueTurn | None: ...


class System3DeliberationProtocol(Protocol):
    def deliberate(self, context_summary: str, stamp_monotonic_s: float) -> SubgoalPlan: ...
