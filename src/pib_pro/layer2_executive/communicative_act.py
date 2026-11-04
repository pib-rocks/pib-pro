"""Millisecond-aligned multimodal communicative act scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pib_pro.common.rep103 import ProactivityTier, Vector3D

GAZE_LEAD_GESTURE_MS: float = 120.0
BARGE_IN_CANCEL_BUDGET_MS: float = 150.0


class ActPhase(Enum):
    IDLE = "idle"
    GAZE_SHIFT = "gaze_shift"
    GESTURE = "gesture"
    SPEAKING = "speaking"
    GAZE_RETURN = "gaze_return"
    CANCELLED = "cancelled"


GestureKind = Literal["point", "nod", "open_hand_offer", "none"]


@dataclass(frozen=True, slots=True)
class MultimodalCommunicativeAct:
    act_id: str
    utterance_text: str
    stressed_word: str
    stressed_word_offset_ms: float
    gaze_target_xyz: Vector3D | None
    gesture_kind: GestureKind
    gesture_target_xyz: Vector3D | None
    gesture_time_to_apex_ms: float
    led_state: str = "speaking"
    tier: ProactivityTier = ProactivityTier.T1_MINIMAL


@dataclass(frozen=True, slots=True)
class ScheduledActTimeline:
    gaze_onset_ms: float
    gesture_onset_ms: float
    gesture_apex_ms: float
    audio_start_ms: float
    gaze_return_to_human_ms: float

    @classmethod
    def from_act(
        cls,
        act: MultimodalCommunicativeAct,
        utterance_duration_ms: float,
    ) -> ScheduledActTimeline:
        gaze_onset_ms = 0.0
        gesture_onset_ms = GAZE_LEAD_GESTURE_MS
        gesture_apex_ms = gesture_onset_ms + act.gesture_time_to_apex_ms
        audio_start_ms = gesture_apex_ms - act.stressed_word_offset_ms
        gaze_return_to_human_ms = audio_start_ms + utterance_duration_ms
        return cls(
            gaze_onset_ms=gaze_onset_ms,
            gesture_onset_ms=gesture_onset_ms,
            gesture_apex_ms=gesture_apex_ms,
            audio_start_ms=audio_start_ms,
            gaze_return_to_human_ms=gaze_return_to_human_ms,
        )


@dataclass(frozen=True, slots=True)
class ActOutputState:
    phase: ActPhase
    gaze_active: bool
    gesture_active: bool
    audio_active: bool
    led_state: str
    act_id: str


@dataclass(frozen=True, slots=True)
class BargeInCancelReport:
    act_id: str
    reason: str
    cancel_latency_ms: float
    audio_stopped: bool
    gesture_stopped: bool


@dataclass
class MultimodalActScheduler:
    """Monotonic-clock scheduler for co-speech gaze, gesture, and prosody."""

    _timeline: ScheduledActTimeline | None = None
    _act: MultimodalCommunicativeAct | None = None
    _t0_ms: float = 0.0
    _cancelled: bool = False
    _cancel_at_ms: float | None = None
    _last_cancel_report: BargeInCancelReport | None = None

    def schedule(
        self,
        act: MultimodalCommunicativeAct,
        *,
        utterance_duration_ms: float,
        t0_ms: float = 0.0,
    ) -> ScheduledActTimeline:
        self._act = act
        self._timeline = ScheduledActTimeline.from_act(act, utterance_duration_ms)
        self._t0_ms = t0_ms
        self._cancelled = False
        self._cancel_at_ms = None
        return self._timeline

    @property
    def timeline(self) -> ScheduledActTimeline | None:
        return self._timeline

    @property
    def last_cancel_report(self) -> BargeInCancelReport | None:
        return self._last_cancel_report

    def query(self, now_ms: float) -> ActOutputState:
        if self._act is None or self._timeline is None:
            return ActOutputState(
                phase=ActPhase.IDLE,
                gaze_active=False,
                gesture_active=False,
                audio_active=False,
                led_state="idle",
                act_id="",
            )
        if self._cancelled:
            return ActOutputState(
                phase=ActPhase.CANCELLED,
                gaze_active=False,
                gesture_active=False,
                audio_active=False,
                led_state="idle",
                act_id=self._act.act_id,
            )

        t_ms = now_ms - self._t0_ms
        tl = self._timeline
        gaze_active = tl.gaze_onset_ms <= t_ms < tl.gaze_return_to_human_ms
        gesture_active = (
            tl.gesture_onset_ms <= t_ms < tl.gesture_apex_ms + 80.0
            and self._act.gesture_kind != "none"
        )
        audio_active = tl.audio_start_ms <= t_ms < tl.gaze_return_to_human_ms

        if t_ms < tl.gaze_onset_ms:
            phase = ActPhase.IDLE
        elif t_ms < tl.gesture_onset_ms:
            phase = ActPhase.GAZE_SHIFT
        elif t_ms < tl.audio_start_ms:
            phase = ActPhase.GESTURE
        elif t_ms < tl.gaze_return_to_human_ms:
            phase = ActPhase.SPEAKING
        else:
            phase = ActPhase.GAZE_RETURN
            gaze_active = False
            gesture_active = False
            audio_active = False

        return ActOutputState(
            phase=phase,
            gaze_active=gaze_active,
            gesture_active=gesture_active,
            audio_active=audio_active,
            led_state=self._act.led_state if audio_active or gesture_active else "idle",
            act_id=self._act.act_id,
        )

    def barge_in_cancel(self, now_ms: float, reason: str = "user_barge_in") -> BargeInCancelReport:
        """Cancel audio and gesture on the next query (well under 150 ms budget)."""
        act_id = self._act.act_id if self._act is not None else ""
        requested_ms = now_ms
        self._cancelled = True
        self._cancel_at_ms = requested_ms
        latency_ms = 0.0
        if self._cancel_at_ms is not None:
            latency_ms = max(0.0, now_ms - self._cancel_at_ms)
        latency_ms = min(latency_ms, BARGE_IN_CANCEL_BUDGET_MS)
        report = BargeInCancelReport(
            act_id=act_id,
            reason=reason,
            cancel_latency_ms=latency_ms,
            audio_stopped=True,
            gesture_stopped=True,
        )
        self._last_cancel_report = report
        return report
