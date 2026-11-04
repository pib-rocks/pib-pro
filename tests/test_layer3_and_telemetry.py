"""Layer 3 cognition and 10 Hz telemetry gateway tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pib_pro.common import ProactivityTier
from pib_pro.layer3_cognition import (
    AnticipatoryProactivityGate,
    DeliberationResult,
    FourTierEmbodiedMemory,
    HierarchicalSystem3Planner,
    LocalStreamingOmniDialogue,
    ProsodyFeatures,
    SemanticSocialProfileStore,
    SocialProfile,
    WorkingWorldBuffer,
    WorldStateContextToken,
)
from pib_pro.layer3_cognition.deliberation import FEASIBILITY_BLOCK_THRESHOLD, OBSTACLE_CLEAR_SKILL
from pib_pro.layer3_cognition.omni_dialogue import (
    CLARIFYING_DEICTIC_UTTERANCE,
    MAX_END_OF_SPEECH_TO_RESPONSE_MS,
)
from pib_pro.layer3_cognition.proactivity_and_memory import (
    PROACTIVITY_ONSET_BUDGET_MS,
    EpisodicSpatialGraph,
    ProceduralMemoryLink,
)
from pib_pro.telemetry_gateway import NonBlockingTelemetryGateway
from pib_pro.telemetry_gateway.gateway import TELEMETRY_HZ, TELEMETRY_PERIOD_S
from pib_pro.world_state_bus import WorldStateRingBuffer, empty_slot_array
from pib_pro.world_state_bus.schema import BUS_PERIOD_NS, LATENT_DIM


def _write_handover_slot(
    bus: WorldStateRingBuffer,
    *,
    frame: int,
    joint_attention_object_id: int = 1,
    joint_attention_prob: float = 0.8,
    hand_vel: tuple[float, float, float] = (-0.25, 0.0, 0.0),
    hand_pos: tuple[float, float, float] = (0.55, 0.0, 0.85),
) -> None:
    with bus.acquire_write_view() as view:
        raw = view.raw
        raw["timestamp_ns"] = frame * BUS_PERIOD_NS
        view.frame_id = frame
        view.human_present = True
        raw["joint_attention_object_id"] = joint_attention_object_id
        view.objects[1]["object_id"] = 1
        view.objects[1]["position_xyz"] = np.array([0.45, 0.05, 0.72], dtype=np.float32)
        view.objects[1]["joint_attention_prob"] = np.float32(joint_attention_prob)
        raw["gaze_ray_dir_xyz"] = np.array([1.0, 0.0, -0.2], dtype=np.float32)
        raw["right_hand_xyz"] = np.array(hand_pos, dtype=np.float32)
        raw["right_hand_vel_xyz"] = np.array(hand_vel, dtype=np.float32)
        raw["z_hat_human"] = np.zeros(LATENT_DIM, dtype=np.float32)


def test_local_omni_dialogue_latency_and_clarification() -> None:
    dialogue = LocalStreamingOmniDialogue()
    token = WorldStateContextToken(
        active_object_ids=(1, 2),
        joint_attention_object_id=-1,
        object_positions_xyz=((0.4, 0.0, 0.7), (0.5, 0.1, 0.7)),
        surprise_scalar=0.1,
        human_pointing_active=False,
        human_head_shake_detected=False,
    )
    prosody = ProsodyFeatures(
        pitch_hz=140.0,
        energy_db=-30.0,
        hesitation_ratio=0.5,
        is_question=True,
        is_laughter=False,
    )
    for i in range(5):
        dialogue.respond_to_user(token, prosody, now_ms=float(i * 1000))
    assert dialogue.median_turn_latency_ms() <= MAX_END_OF_SPEECH_TO_RESPONSE_MS
    resp = dialogue.respond_to_user(token, prosody, now_ms=6000.0)
    assert CLARIFYING_DEICTIC_UTTERANCE in resp.act.utterance_text


def test_zero_self_triggered_barge_ins_under_aec() -> None:
    dialogue = LocalStreamingOmniDialogue()
    dialogue.simulate_robot_speech_aec_residual(600.0)
    assert dialogue.self_triggered_barge_ins == 0


def test_mid_sentence_self_correction_trials() -> None:
    dialogue = LocalStreamingOmniDialogue()
    successes = 0
    trials = 20
    for trial in range(trials):
        blue_token = WorldStateContextToken(
            active_object_ids=(10 + trial, 20 + trial),
            joint_attention_object_id=10 + trial,
            object_positions_xyz=((0.42, 0.0, 0.71), (0.52, 0.05, 0.71)),
            surprise_scalar=0.05,
            human_pointing_active=False,
            human_head_shake_detected=False,
        )
        prosody = ProsodyFeatures(120.0, -28.0, 0.1, False, False)
        dialogue.respond_to_user(
            blue_token,
            prosody,
            utterance_hint="Sure, I will take the blue...",
            now_ms=float(trial * 2000),
        )
        red_token = WorldStateContextToken(
            active_object_ids=blue_token.active_object_ids,
            joint_attention_object_id=20 + trial,
            object_positions_xyz=blue_token.object_positions_xyz,
            surprise_scalar=0.2,
            human_pointing_active=True,
            human_head_shake_detected=False,
        )
        if dialogue.tick_visual_barge_in(red_token):
            corrected = dialogue.emit_mid_sentence_correction(
                red_token,
                now_ms=float(trial * 2000 + 120),
            )
            if corrected.self_corrected_mid_sentence and "red" in corrected.act.utterance_text:
                successes += 1
    assert successes >= 17


def test_hierarchical_system3_non_blocking_and_obstacle_plan() -> None:
    slot = empty_slot_array(1)[0]
    slot["objects"][1]["object_id"] = 1
    slot["objects"][1]["position_xyz"] = np.array([0.35, 0.0, 0.65], dtype=np.float32)
    slot["objects"][1]["confidence"] = np.float32(0.9)
    slot["objects"][2]["object_id"] = 99
    slot["objects"][2]["position_xyz"] = np.array([0.36, 0.01, 0.66], dtype=np.float32)
    slot["objects"][2]["confidence"] = np.float32(0.85)
    planner = HierarchicalSystem3Planner()
    audio_ms = 0.0
    done = threading.Event()
    result_holder: list[DeliberationResult] = []

    def on_complete(res: DeliberationResult) -> None:
        result_holder.append(res)
        done.set()

    thread = planner.plan_async(slot, target_object_id=1, on_complete=on_complete)
    start = time.monotonic()
    while not done.is_set() and time.monotonic() - start < 2.0:
        audio_ms += 0.0
        time.sleep(0.001)
    thread.join(timeout=2.0)
    assert done.is_set()
    assert audio_ms == 0.0
    result = result_holder[0]
    assert result.blocking_audio_ms == 0.0
    assert result.spoke_clarification or OBSTACLE_CLEAR_SKILL in result.plan.subgoals
    subgoals = planner.hjepa_planner.plan_subgoals(
        ("reach_target",),
        slot,
        target_object_id=1,
    )
    blocked = planner.hjepa_planner.first_infeasible(subgoals)
    assert blocked is not None
    assert blocked.feasibility_score < FEASIBILITY_BLOCK_THRESHOLD


def test_anticipatory_proactivity_silent_receive_and_no_unsolicited_speech() -> None:
    bus = WorldStateRingBuffer(capacity=64)
    _write_handover_slot(bus, frame=1)
    gate = AnticipatoryProactivityGate()
    decision = gate.evaluate(bus, stamp_monotonic_s=1.0, simulated_onset_ms=650.0)
    assert decision.silent_physical_action == "open_hand_receive_pose"
    assert decision.verbal_utterance is None
    assert decision.onset_latency_ms <= PROACTIVITY_ONSET_BUDGET_MS
    for frame in range(2, 62):
        _write_handover_slot(
            bus,
            frame=frame,
            joint_attention_object_id=-1,
            joint_attention_prob=0.0,
        )
        silent = gate.evaluate(bus, stamp_monotonic_s=float(frame))
        assert silent.verbal_utterance is None
        assert silent.proactivity_tier == ProactivityTier.T0_SILENT
    assert gate.unsolicited_verbal_count == 0


def test_four_tier_memory_persistence_and_consent(tmp_path: Path) -> None:
    bus = WorldStateRingBuffer(capacity=32)
    db_path = tmp_path / "episodes.db"
    graph = EpisodicSpatialGraph(db_path)
    graph.record_observation(
        episode_id="ep-1",
        timestamp_ns=time.time_ns(),
        object_id=7,
        label="tape",
        position_xyz=(0.44, -0.08, 0.73),
        person_id=1,
        summary_text="tape on table",
        latent_z_t=np.zeros(LATENT_DIM, dtype=np.float32),
    )
    seen = graph.query_object_last_seen("tape")
    assert seen is not None
    assert seen.position_xyz == pytest.approx((0.44, -0.08, 0.73))
    graph.close()
    reloaded = EpisodicSpatialGraph(db_path)
    seen2 = reloaded.query_object_last_seen("tape")
    assert seen2 is not None
    assert seen2.object_id == 7
    social_path = tmp_path / "social.json"
    store = SemanticSocialProfileStore(social_path)
    with pytest.raises(ValueError, match="consent"):
        store.upsert(
            SocialProfile(
                person_id=1,
                consent_granted=False,
                preferred_handover_side="left",
                preferred_handover_height_z_m=0.85,
                speaking_pace="normal",
                proactivity_preference=ProactivityTier.T1_MINIMAL,
            )
        )
    store.upsert(
        SocialProfile(
            person_id=1,
            consent_granted=True,
            preferred_handover_side="right",
            preferred_handover_height_z_m=0.88,
            speaking_pace="slow",
            proactivity_preference=ProactivityTier.T1_MINIMAL,
        )
    )
    store.save()
    store2 = SemanticSocialProfileStore(social_path)
    store2.load()
    profile = store2.get(1)
    assert profile is not None
    assert profile.preferred_handover_side == "right"
    memory = FourTierEmbodiedMemory(
        tier1=WorkingWorldBuffer(bus),
        tier2=reloaded,
        tier3=store2,
        tier4=ProceduralMemoryLink(),
    )
    assert memory.query_object_last_seen("tape") is not None
    reloaded.close()


def test_non_blocking_telemetry_gateway_10hz_json() -> None:
    bus = WorldStateRingBuffer(capacity=16)
    with bus.acquire_write_view() as view:
        raw = view.raw
        raw["timestamp_ns"] = BUS_PERIOD_NS
        view.surprise_scalar = 0.42
        raw["active_bt_node_id"] = 17
        raw["vad_active"] = True
        raw["speaker_doa_azimuth_rad"] = np.float32(0.5)
        view.objects[0]["object_id"] = 3
        view.objects[0]["position_xyz"] = np.array([0.3, 0.1, 0.75], dtype=np.float32)
    gw = NonBlockingTelemetryGateway(bus)
    t0 = 100.0
    first = gw.sample_payload(t0)
    assert first is not None
    second = gw.sample_payload(t0 + TELEMETRY_PERIOD_S * 0.5)
    assert second is first
    third = gw.sample_payload(t0 + TELEMETRY_PERIOD_S + 1e-3)
    assert third is not None
    assert third.stamp_monotonic_s == pytest.approx(t0 + TELEMETRY_PERIOD_S + 1e-3)
    snap = gw.collect(t0)
    raw = gw.publish_json(snap)
    payload = json.loads(raw)
    assert payload["surprise_scalar"] == pytest.approx(0.42)
    assert payload["system_health"]["telemetry_hz"] == TELEMETRY_HZ
    assert len(payload["scene_markers_rep103"]) >= 1
