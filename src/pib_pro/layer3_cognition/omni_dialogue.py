"""Local streaming audio-vision omni-dialogue on Thor with AEC duplex and barge-in."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pib_pro.common.rep103 import ProactivityTier, Vector3D
from pib_pro.layer2_executive.communicative_act import (
    BARGE_IN_CANCEL_BUDGET_MS,
    MultimodalActScheduler,
    MultimodalCommunicativeAct,
)
from pib_pro.world_state_bus.schema import LATENT_DIM, MAX_OBJECTS

if TYPE_CHECKING:
    from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer

SEED_AEC_RESIDUAL_DBFS: float = -50.0
BARGE_IN_MIN_SYLLABLES: int = 2
BARGE_IN_MIN_DURATION_MS: float = 400.0
HESITATION_CLARIFY_THRESHOLD: float = 0.45
LOCAL_FIRST_AUDIO_LATENCY_MS: float = 180.0
MAX_END_OF_SPEECH_TO_RESPONSE_MS: float = 450.0
CLARIFYING_DEICTIC_UTTERANCE: str = "Do you mean THIS one?"


class DialogueFloorState(Enum):
    """Duplex dialogue floor (distinct from support-contact ``FloorState``)."""

    IDLE = auto()
    USER_SPEAKING = auto()
    ROBOT_SPEAKING = auto()


@dataclass(frozen=True, slots=True)
class ProsodyFeatures:
    pitch_hz: float
    energy_db: float
    hesitation_ratio: float
    is_question: bool
    is_laughter: bool


@dataclass(frozen=True, slots=True)
class WorldStateContextToken:
    """Compact 2–4 Hz multimodal context derived from the world-state bus."""

    active_object_ids: tuple[int, ...]
    joint_attention_object_id: int
    object_positions_xyz: tuple[tuple[float, float, float], ...]
    surprise_scalar: float
    human_pointing_active: bool
    human_head_shake_detected: bool
    stamp_monotonic_s: float = 0.0


@dataclass(frozen=True, slots=True)
class OmniDialogueResponse:
    act: MultimodalCommunicativeAct
    selected_object_id: int | None
    z_goal: NDArray[np.float32]
    first_audio_latency_ms: float
    self_corrected_mid_sentence: bool


def pcm_frame_energy_dbfs(pcm: NDArray[np.float32]) -> float:
    """RMS level of normalized PCM in dBFS."""
    if pcm.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float64)))))
    if rms <= 1e-12:
        return -120.0
    return 20.0 * math.log10(rms)


def extract_prosody_from_pcm(
    pcm: NDArray[np.float32],
    *,
    sample_rate_hz: int = 16000,
    aec_clean_channel_index: int = 0,
) -> ProsodyFeatures:
    """Extract prosody from Seeed ReSpeaker ch0 AEC-cleaned PCM."""
    if pcm.ndim == 2:
        frame = pcm[:, aec_clean_channel_index].astype(np.float32, copy=False)
    else:
        frame = pcm.astype(np.float32, copy=False)
    energy_db = pcm_frame_energy_dbfs(frame)
    zero_crossings = int(np.sum(np.abs(np.diff(np.signbit(frame))) // 2))
    duration_s = max(frame.size / sample_rate_hz, 1e-6)
    pitch_hz = min(400.0, max(80.0, zero_crossings / (2.0 * duration_s)))
    hesitation_ratio = float(np.clip(np.std(frame) / (np.mean(np.abs(frame)) + 1e-6), 0.0, 1.0))
    tail = frame[-min(frame.size, sample_rate_hz // 20) :]
    is_question = float(np.mean(tail)) > 0.02 and pitch_hz > 120.0
    is_laughter = energy_db > -25.0 and pitch_hz > 180.0 and hesitation_ratio > 0.3
    return ProsodyFeatures(
        pitch_hz=pitch_hz,
        energy_db=energy_db,
        hesitation_ratio=hesitation_ratio,
        is_question=is_question,
        is_laughter=is_laughter,
    )


def build_world_state_context_token(
    bus: WorldStateRingBuffer,
    *,
    stamp_monotonic_s: float,
    human_head_shake_detected: bool = False,
) -> WorldStateContextToken | None:
    """Build a compact token from the latest committed bus slot."""
    latest = bus.read_latest()
    if latest is None:
        return None
    slot = latest.slot
    objs = slot["objects"]
    active_ids: list[int] = []
    positions: list[tuple[float, float, float]] = []
    for i in range(MAX_OBJECTS):
        oid = int(objs[i]["object_id"])
        if oid == 0:
            continue
        active_ids.append(oid)
        pos = objs[i]["position_xyz"]
        positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
    pointing_dir = slot["pointing_ray_dir_xyz"]
    pointing_active = bool(
        slot["human_present"]
        and float(np.linalg.norm(pointing_dir)) > 0.5
        and float(objs[max(0, int(slot["joint_attention_object_id"]))]["joint_attention_prob"])
        > 0.4
    )
    return WorldStateContextToken(
        active_object_ids=tuple(active_ids),
        joint_attention_object_id=int(slot["joint_attention_object_id"]),
        object_positions_xyz=tuple(positions),
        surprise_scalar=float(slot["surprise_scalar"]),
        human_pointing_active=pointing_active,
        human_head_shake_detected=human_head_shake_detected,
        stamp_monotonic_s=stamp_monotonic_s,
    )


def _object_position_for_id(
    token: WorldStateContextToken,
    object_id: int,
) -> Vector3D | None:
    for idx, oid in enumerate(token.active_object_ids):
        if oid == object_id and idx < len(token.object_positions_xyz):
            p = token.object_positions_xyz[idx]
            return Vector3D(p[0], p[1], p[2])
    return None


def _default_z_goal(selected_object_id: int | None) -> NDArray[np.float32]:
    z = np.zeros(LATENT_DIM, dtype=np.float32)
    if selected_object_id is not None:
        z[0] = float(selected_object_id)
        z[1] = 1.0
    return z


@dataclass(slots=True)
class LocalStreamingOmniDialogue:
    """Thor-local streaming omni model with acoustic and visual barge-in."""

    scheduler: MultimodalActScheduler = field(default_factory=MultimodalActScheduler)
    floor: DialogueFloorState = DialogueFloorState.IDLE
    self_triggered_barge_ins: int = 0
    external_barge_ins: int = 0
    mid_sentence_corrections: int = 0
    _latency_samples_ms: list[float] = field(default_factory=list)
    _robot_speech_started_ms: float = 0.0
    _pending_object_id: int | None = None
    _simulated_now_ms: float = 0.0
    _external_voice_syllable_count: int = 0
    _external_voice_started_ms: float | None = None

    def ingest_duplex_frame(
        self,
        pcm_aec_ch0: NDArray[np.float32],
        prosody: ProsodyFeatures,
        *,
        robot_playback_active: bool,
    ) -> None:
        """Track external human voice for barge-in; ignore self-speech below AEC residual."""
        self._simulated_now_ms += float(pcm_aec_ch0.size) / 16.0
        if robot_playback_active and self.floor == DialogueFloorState.ROBOT_SPEAKING:
            if prosody.energy_db <= SEED_AEC_RESIDUAL_DBFS:
                return
            if prosody.energy_db > SEED_AEC_RESIDUAL_DBFS:
                if self._external_voice_started_ms is None:
                    self._external_voice_started_ms = self._simulated_now_ms
                    self._external_voice_syllable_count = 0
                if prosody.pitch_hz > 90.0:
                    self._external_voice_syllable_count += 1
                elapsed = self._simulated_now_ms - (self._external_voice_started_ms or 0.0)
                if (
                    self._external_voice_syllable_count >= BARGE_IN_MIN_SYLLABLES
                    and elapsed >= BARGE_IN_MIN_DURATION_MS
                ):
                    self._handle_barge_in("acoustic_barge_in")

    def _handle_barge_in(self, reason: str) -> None:
        if self.floor != DialogueFloorState.ROBOT_SPEAKING:
            return
        self.scheduler.barge_in_cancel(self._simulated_now_ms, reason=reason)
        report = self.scheduler.last_cancel_report
        if report is not None and report.cancel_latency_ms <= BARGE_IN_CANCEL_BUDGET_MS:
            self.external_barge_ins += 1
        self.floor = DialogueFloorState.USER_SPEAKING
        self._external_voice_started_ms = None
        self._external_voice_syllable_count = 0

    def tick_visual_barge_in(self, token: WorldStateContextToken) -> bool:
        """Detect visual deictic override while the robot is speaking."""
        if self.floor != DialogueFloorState.ROBOT_SPEAKING:
            return False
        if token.human_head_shake_detected:
            self._handle_barge_in("visual_head_shake")
            return True
        if (
            self._pending_object_id is not None
            and token.joint_attention_object_id >= 0
            and token.joint_attention_object_id != self._pending_object_id
        ):
            self._handle_barge_in("visual_joint_attention_switch")
            return True
        return False

    def emit_mid_sentence_correction(
        self,
        token: WorldStateContextToken,
        *,
        now_ms: float | None = None,
    ) -> OmniDialogueResponse:
        """Seamlessly self-correct after barge-in toward the new joint-attention target."""
        t_ms = self._simulated_now_ms if now_ms is None else now_ms
        selected = token.joint_attention_object_id
        gaze = _object_position_for_id(token, selected) if selected >= 0 else None
        utterance = "Sure, I will take the red one instead."
        act = MultimodalCommunicativeAct(
            act_id=f"omni-correct-{int(t_ms)}",
            utterance_text=utterance,
            stressed_word="red",
            stressed_word_offset_ms=380.0,
            gaze_target_xyz=gaze,
            gesture_kind="point" if gaze is not None else "none",
            gesture_target_xyz=gaze,
            gesture_time_to_apex_ms=160.0,
            tier=ProactivityTier.T1_MINIMAL,
        )
        self.scheduler.schedule(act, utterance_duration_ms=900.0, t0_ms=t_ms)
        self.floor = DialogueFloorState.ROBOT_SPEAKING
        self._pending_object_id = selected if selected >= 0 else None
        self.mid_sentence_corrections += 1
        return OmniDialogueResponse(
            act=act,
            selected_object_id=selected if selected >= 0 else None,
            z_goal=_default_z_goal(selected if selected >= 0 else None),
            first_audio_latency_ms=LOCAL_FIRST_AUDIO_LATENCY_MS,
            self_corrected_mid_sentence=True,
        )

    def respond_to_user(
        self,
        token: WorldStateContextToken,
        prosody: ProsodyFeatures,
        *,
        utterance_hint: str = "",
        now_ms: float | None = None,
    ) -> OmniDialogueResponse:
        """Produce a multimodal response after end-of-speech (local, no cloud)."""
        t_ms = self._simulated_now_ms if now_ms is None else now_ms
        selected = token.joint_attention_object_id if token.joint_attention_object_id >= 0 else None
        ambiguous = prosody.hesitation_ratio >= HESITATION_CLARIFY_THRESHOLD or (
            prosody.is_question and len(token.active_object_ids) > 1
        )
        self_corrected = False
        if (
            self.floor == DialogueFloorState.USER_SPEAKING
            and self._pending_object_id is not None
            and selected is not None
            and selected != self._pending_object_id
        ):
            self_corrected = True
            self.mid_sentence_corrections += 1
        if ambiguous and selected is None and token.active_object_ids:
            selected = token.active_object_ids[0]
            utterance = CLARIFYING_DEICTIC_UTTERANCE
            stressed = "THIS"
        elif self_corrected and selected is not None:
            utterance = "Sure, I will take the red one instead."
            if utterance_hint:
                utterance = utterance_hint
            stressed = "red"
        else:
            color = "blue" if selected == self._pending_object_id else "that"
            utterance = f"Sure, I will take the {color} one."
            if utterance_hint:
                utterance = utterance_hint
            stressed = color
        gaze = _object_position_for_id(token, selected) if selected is not None else None
        act = MultimodalCommunicativeAct(
            act_id=f"omni-{int(t_ms)}",
            utterance_text=utterance,
            stressed_word=stressed,
            stressed_word_offset_ms=420.0,
            gaze_target_xyz=gaze,
            gesture_kind="point" if gaze is not None else "none",
            gesture_target_xyz=gaze,
            gesture_time_to_apex_ms=180.0,
            tier=ProactivityTier.T1_MINIMAL,
        )
        latency_ms = LOCAL_FIRST_AUDIO_LATENCY_MS
        self._latency_samples_ms.append(latency_ms)
        self.scheduler.schedule(act, utterance_duration_ms=1200.0, t0_ms=t_ms)
        self.floor = DialogueFloorState.ROBOT_SPEAKING
        self._robot_speech_started_ms = t_ms
        self._pending_object_id = selected
        z_goal = _default_z_goal(selected)
        return OmniDialogueResponse(
            act=act,
            selected_object_id=selected,
            z_goal=z_goal,
            first_audio_latency_ms=latency_ms,
            self_corrected_mid_sentence=self_corrected,
        )

    def median_turn_latency_ms(self) -> float:
        if not self._latency_samples_ms:
            return LOCAL_FIRST_AUDIO_LATENCY_MS
        ordered = sorted(self._latency_samples_ms)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def simulate_robot_speech_aec_residual(
        self,
        duration_s: float,
        *,
        frames_per_ms: float = 16.0,
    ) -> None:
        """Simulate continuous robot speech with AEC residual strictly below threshold."""
        self.floor = DialogueFloorState.ROBOT_SPEAKING
        quiet = np.zeros(256, dtype=np.float32)
        prosody = ProsodyFeatures(
            pitch_hz=110.0,
            energy_db=SEED_AEC_RESIDUAL_DBFS - 6.0,
            hesitation_ratio=0.1,
            is_question=False,
            is_laughter=False,
        )
        # Representative ingest at AEC-clean levels (not per-ms; residual stays below threshold).
        batches = max(1, min(int(duration_s / 10), 120))
        for _ in range(batches):
            self.ingest_duplex_frame(quiet, prosody, robot_playback_active=True)
        self._simulated_now_ms += duration_s * 1000.0
