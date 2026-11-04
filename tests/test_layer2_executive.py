"""Layer 2 executive: surprise-gated BT and multimodal communicative scheduling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pib_pro.common import FRAME_CHEST_CAMERA_OAK4_OPTICAL, Vector3D
from pib_pro.layer2_executive import (
    BTSkillStep,
    MultimodalActScheduler,
    MultimodalCommunicativeAct,
    ScheduledActTimeline,
    SurpriseGatedBehaviourTree,
    run_workbench_task_plan,
)
from pib_pro.layer2_executive.surprise_bt import (
    INTERRUPT_FRAME_PERIOD_S,
    MAX_PREEMPTION_LATENCY_MS,
)
from pib_pro.perception_oak4 import (
    Detection2DWithDepth,
    Oak4CameraIntrinsics,
    PoseTransform3D,
    SeeedMicArrayConfig,
    SelfArmMaskConfig,
    filter_self_arm_occlusions,
    seeed_doa_deg_to_rep103_azimuth_rad,
)
from pib_pro.world_state_bus import WorldStateRingBuffer
from pib_pro.world_state_bus.schema import BUS_PERIOD_NS


def _intrinsics() -> Oak4CameraIntrinsics:
    return Oak4CameraIntrinsics(fx=700.0, fy=700.0, cx=640.0, cy=360.0, width=1280, height=720)


def _det_at_depth(
    label: str,
    u: float,
    v: float,
    depth_m: float,
    *,
    emb: float = 1.0,
) -> Detection2DWithDepth:
    return Detection2DWithDepth(
        class_id=1,
        label=label,
        bbox_xyxy=(u - 20, v - 20, u + 20, v + 20),
        depth_m=depth_m,
        confidence=0.9,
        appearance_embedding=np.array([emb, 0.2, 0.1], dtype=np.float32),
        half_extents_xyz=(0.03, 0.03, 0.03),
    )


def test_chest_oak4_extrinsic_invariant_to_neck_pose() -> None:
    chest_a = PoseTransform3D.oak4_chest_optical_to_rep103_base()
    chest_b = PoseTransform3D.oak4_chest_optical_to_rep103_base()
    head = PoseTransform3D.oak4_optical_to_rep103_base()
    assert chest_a.translation_base_from_optical.z == pytest.approx(0.82)
    assert chest_a.translation_base_from_optical == chest_b.translation_base_from_optical
    assert head.translation_base_from_optical.z == pytest.approx(0.0)
    assert FRAME_CHEST_CAMERA_OAK4_OPTICAL == "chest_camera_oak4_optical_frame"
    pt = chest_a.transform_point_optical_to_base(0.0, 0.0, 1.0)
    assert pt.x > chest_a.translation_base_from_optical.x


def _det_at_base_xyz(
    label: str,
    position: Vector3D,
    intr: Oak4CameraIntrinsics,
    extrinsic: PoseTransform3D,
    *,
    emb: float = 1.0,
) -> Detection2DWithDepth:
    rel = position - extrinsic.translation_base_from_optical
    rot = extrinsic.rotation_base_from_optical
    p_opt = rot.T @ rel.as_array()
    z_opt = float(p_opt[2])
    x_opt = float(p_opt[0])
    y_opt = float(p_opt[1])
    u = intr.fx * x_opt / z_opt + intr.cx
    v = intr.fy * y_opt / z_opt + intr.cy
    return _det_at_depth(label, u, v, z_opt, emb=emb)


def test_filter_self_arm_occlusions_masks_hand_not_table() -> None:
    intr = _intrinsics()
    extrinsic = PoseTransform3D.oak4_chest_optical_to_rep103_base()
    left_hand = Vector3D(0.35, 0.12, 0.72)
    right_hand = Vector3D(0.35, -0.12, 0.72)
    table_pos = Vector3D(0.40, 0.0, 0.70)
    config = SelfArmMaskConfig(
        intrinsics=intr,
        extrinsic=extrinsic,
        persistent_track_positions=(table_pos,),
    )
    hand_det = _det_at_base_xyz("hand_blob", left_hand, intr, extrinsic, emb=0.5)
    table_det = _det_at_base_xyz("cup", table_pos, intr, extrinsic, emb=0.9)
    filtered = filter_self_arm_occlusions(
        (hand_det, table_det),
        left_hand,
        right_hand,
        config,
    )
    labels = {d.label for d in filtered}
    assert "hand_blob" not in labels
    assert "cup" in labels


def test_seeed_mic_array_config_and_doa_conversion() -> None:
    cfg = SeeedMicArrayConfig()
    assert cfg.usb_vid == 0x2886
    assert cfg.usb_pid == 0x0018
    assert cfg.channels == 6
    assert cfg.sample_rate_hz == 16000
    assert cfg.period_size_frames == 256
    assert cfg.aec_reference_channel == 0
    assert seeed_doa_deg_to_rep103_azimuth_rad(0.0) == pytest.approx(0.0)
    assert seeed_doa_deg_to_rep103_azimuth_rad(90.0) == pytest.approx(math.pi / 2)
    assert seeed_doa_deg_to_rep103_azimuth_rad(180.0) == pytest.approx(math.pi)
    assert seeed_doa_deg_to_rep103_azimuth_rad(-90.0) == pytest.approx(-math.pi / 2)


def _write_bus(
    bus: WorldStateRingBuffer,
    frame: int,
    *,
    surprise: float = 0.0,
    z_t: np.ndarray | None = None,
) -> None:
    with bus.acquire_write_view() as view:
        view.timestamp_ns = frame * BUS_PERIOD_NS
        view.frame_id = frame
        view.surprise_scalar = np.float32(surprise)
        if z_t is not None:
            view.z_t[...] = np.float32(0.0)
            view.z_t[:3] = np.asarray(z_t, dtype=np.float32)[:3]


def test_surprise_preempts_grasp_within_100ms() -> None:
    bus = WorldStateRingBuffer(capacity=64)
    goal = np.array([0.1, 0.0, 0.5], dtype=np.float64)
    steps = [
        BTSkillStep(
            step_id="grasp_cup",
            skill_name="grasp",
            z_goal=goal,
            surprise_threshold=0.35,
            recovery_step_id="reacquire",
            recovery_utterance="The cup moved; let me look again.",
        ),
        BTSkillStep(
            step_id="reacquire",
            skill_name="look",
            z_goal=goal,
            surprise_threshold=0.9,
        ),
    ]
    tree = SurpriseGatedBehaviourTree()
    tree.load_plan(steps)
    _write_bus(bus, 0, surprise=0.0, z_t=goal + 0.2)

    t = 0.0
    preemption = None
    for _ in range(6):
        _write_bus(bus, int(t / INTERRUPT_FRAME_PERIOD_S), surprise=0.85, z_t=goal + 0.2)
        hit = tree.check_surprise_interrupt(bus, t)
        if hit is not None:
            preemption = hit
            break
        t += INTERRUPT_FRAME_PERIOD_S

    assert preemption is not None
    assert preemption.latency_ms <= MAX_PREEMPTION_LATENCY_MS
    assert preemption.recovery_step_id == "reacquire"
    assert preemption.spoken_explanation
    assert tree.active_step_id == "reacquire"


def _workbench_plan(prefix: str) -> list[BTSkillStep]:
    goals = [np.array([0.1 + i * 0.02, 0.0, 0.5], dtype=np.float64) for i in range(5)]
    steps: list[BTSkillStep] = []
    for i, g in enumerate(goals):
        sid = f"{prefix}_step_{i}"
        recovery = f"{prefix}_recover_{i}" if i == 2 else None
        steps.append(
            BTSkillStep(
                step_id=sid,
                skill_name="pick" if i == 2 else "move",
                z_goal=g,
                surprise_threshold=0.35,
                recovery_step_id=recovery,
                recovery_utterance=f"Recovering task {prefix} at step {i}.",
            ),
        )
        if recovery:
            steps.append(
                BTSkillStep(
                    step_id=recovery,
                    skill_name="recover",
                    z_goal=g,
                    surprise_threshold=0.95,
                ),
            )
    return steps


def test_workbench_benchmark_recovers_from_surprise_perturbations() -> None:
    recovered = 0
    for task_idx in range(10):
        bus = WorldStateRingBuffer(capacity=128)
        plan = _workbench_plan(f"bench_{task_idx}")
        tree, _status = run_workbench_task_plan(
            plan,
            bus,
            duration_s=2.5,
            surprise_injections={0.20: 0.88},
            dt_s=INTERRUPT_FRAME_PERIOD_S,
        )
        if (
            tree.last_preemption is not None
            and tree.last_preemption.recovery_step_id
            and tree.last_preemption.spoken_explanation
        ):
            recovered += 1
    assert recovered >= 8


DEICTIC_UTTERANCES: tuple[tuple[str, str], ...] = (
    ("this red cup", "red"),
    ("that blue block", "blue"),
    ("over there", "there"),
    ("this one", "one"),
    ("that tool", "tool"),
    ("this slot", "slot"),
    ("right here", "here"),
    ("that part", "part"),
    ("this edge", "edge"),
    ("those wires", "wires"),
    ("this panel", "panel"),
    ("that corner", "corner"),
    ("this bin", "bin"),
    ("that tray", "tray"),
    ("this gear", "gear"),
    ("that pin", "pin"),
    ("this hole", "hole"),
    ("that mark", "mark"),
    ("this side", "side"),
    ("that face", "face"),
)


def test_multimodal_scheduler_ac_alignment_and_barge_in() -> None:
    scheduler = MultimodalActScheduler()
    target = Vector3D(0.5, 0.1, 0.8)
    for idx, (text, stressed) in enumerate(DEICTIC_UTTERANCES):
        act = MultimodalCommunicativeAct(
            act_id=f"act_{idx}",
            utterance_text=text,
            stressed_word=stressed,
            stressed_word_offset_ms=170.0,
            gaze_target_xyz=target,
            gesture_kind="point",
            gesture_target_xyz=target,
            gesture_time_to_apex_ms=330.0,
        )
        duration_ms = 900.0 + idx * 5.0
        tl = scheduler.schedule(act, utterance_duration_ms=duration_ms, t0_ms=1000.0)
        assert tl.gaze_onset_ms == pytest.approx(0.0)
        lead = tl.gesture_onset_ms - tl.gaze_onset_ms
        assert 80.0 <= lead <= 160.0
        stressed_at = tl.audio_start_ms + act.stressed_word_offset_ms
        assert abs(stressed_at - tl.gesture_apex_ms) <= 60.0
        assert ScheduledActTimeline.from_act(act, duration_ms).audio_start_ms == pytest.approx(
            tl.audio_start_ms,
        )

        mid_ms = 1000.0 + tl.gesture_apex_ms
        state = scheduler.query(mid_ms)
        assert state.gesture_active or state.audio_active

        cancel_ms = 1000.0 + tl.audio_start_ms + 50.0
        report = scheduler.barge_in_cancel(cancel_ms)
        assert report.cancel_latency_ms <= 150.0
        assert report.audio_stopped and report.gesture_stopped
        after = scheduler.query(cancel_ms + 5.0)
        assert not after.audio_active and not after.gesture_active
