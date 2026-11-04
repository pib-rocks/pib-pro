"""Layer 1 sociomotor autopilot at 30 Hz."""

from __future__ import annotations

import numpy as np
import pytest

from pib_pro.layer1_autopilot import (
    JerkLimitedPCHIPFilter,
    NeckEyeTarget,
    SociomotorAutopilot,
)
from pib_pro.world_state_bus import WorldStateRingBuffer, empty_slot_array
from pib_pro.world_state_bus.schema import WorldStateSlotView


def _slot_view(**kwargs: object) -> WorldStateSlotView:
    slot = empty_slot_array(1)[0]
    slot["head_position_xyz"] = np.array([0.0, 0.0, 1.35], dtype=np.float32)
    for key, val in kwargs.items():
        slot[key] = val  # type: ignore[index]
    return WorldStateSlotView.wrap(slot)


def test_deictic_first_tick_nonzero_head_velocity() -> None:
    slot = empty_slot_array(1)[0]
    slot["head_position_xyz"] = np.array([0.0, 0.0, 1.35], dtype=np.float32)
    slot["joint_attention_object_id"] = 0
    slot["objects"][0]["position_xyz"] = np.array([0.8, 0.3, 1.1], dtype=np.float32)
    slot["objects"][0]["joint_attention_prob"] = np.float32(0.9)
    view = WorldStateSlotView.wrap(slot)

    pilot = SociomotorAutopilot()
    dt = 1.0 / 30.0
    out = pilot.step_from_view(view, dt_s=dt)
    assert out.mode == "deictic_object"
    assert out.onset_latency_ms <= 33.4
    assert abs(out.yaw_velocity_rad_s) > 0.01


def test_speaker_doa_onset_within_first_tick() -> None:
    slot = empty_slot_array(1)[0]
    slot["head_position_xyz"] = np.array([0.0, 0.0, 1.35], dtype=np.float32)
    slot["joint_attention_object_id"] = -1
    slot["vad_active"] = True
    slot["human_present"] = True
    slot["speaker_doa_azimuth_rad"] = np.float32(0.55)
    view = WorldStateSlotView.wrap(slot)

    pilot = SociomotorAutopilot()
    out = pilot.step_from_view(view, dt_s=1.0 / 30.0)
    assert out.mode == "speaker_face"
    assert out.onset_latency_ms <= 33.4
    assert abs(out.yaw_velocity_rad_s) > 0.0


def test_reversal_without_velocity_spikes() -> None:
    filt = JerkLimitedPCHIPFilter(max_velocity_rad_s=1.2, max_jerk_rad_s3=8.0)
    filt.reset(0.0)
    dt = 1.0 / 30.0
    velocities: list[float] = []
    filt.set_target(0.6)
    for _ in range(20):
        _, v = filt.step(dt)
        velocities.append(v)
    filt.set_target(-0.6)
    for _ in range(40):
        _, v = filt.step(dt)
        velocities.append(abs(v))
    peak = max(velocities)
    assert peak <= filt.max_velocity_rad_s + 1e-3
    assert not any(v > filt.max_velocity_rad_s * 1.05 for v in velocities)


def test_backchannel_nod_smooth_pitch() -> None:
    slot = empty_slot_array(1)[0]
    slot["head_position_xyz"] = np.array([0.5, 0.1, 1.4], dtype=np.float32)
    slot["vad_active"] = True
    slot["human_present"] = True
    view = WorldStateSlotView.wrap(slot)
    pilot = SociomotorAutopilot()
    dt = 1.0 / 30.0
    pitches: list[float] = []
    for _ in range(15):
        out = pilot.step_from_view(view, dt_s=dt)
        pitches.append(out.pitch_rad)
    delta = max(pitches) - min(pitches)
    assert delta > 0.02
    assert delta < 0.5


def test_blend_with_manipulation_priority() -> None:
    pilot = SociomotorAutopilot()
    socio = NeckEyeTarget(0.2, 0.1, 0.2, 0.1, "speaker_face", 10.0, 0.5, 0.1)
    blended = pilot.blend_with_manipulation(socio, (0.8, -0.2), 0.5)
    assert blended.yaw_rad == pytest.approx(0.5)
    assert blended.pitch_rad == pytest.approx(-0.05)
    assert blended.yaw_velocity_rad_s == pytest.approx(0.25)


def test_ring_buffer_integration() -> None:
    ring = WorldStateRingBuffer(capacity=8)
    slot = empty_slot_array(1)[0]
    slot["head_position_xyz"] = np.array([0.0, 0.0, 1.35], dtype=np.float32)
    slot["joint_attention_object_id"] = 1
    slot["objects"][1]["position_xyz"] = np.array([1.0, 0.0, 1.2], dtype=np.float32)
    slot["objects"][1]["joint_attention_prob"] = np.float32(0.7)
    ring.write_slot(slot)
    pilot = SociomotorAutopilot()
    out = pilot.step(ring)
    assert isinstance(out, NeckEyeTarget)
    assert out.mode == "deictic_object"
