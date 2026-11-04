"""Procedural memory recorder and offline skill consolidation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pib_pro.layer1_autopilot import (
    GR00TActionProposer,
    OfflineSkillConsolidator,
    ProceduralMemoryRecorder,
)
from pib_pro.layer1_autopilot.procedural_memory import LeRobotEpisodeFrame, LeRobotV2Episode
from pib_pro.world_model.hjepa_engine import HJEPAWorldModelEngine
from pib_pro.world_state_bus import WorldStateRingBuffer, empty_slot_array
from pib_pro.world_state_bus.schema import BUS_PERIOD_NS, MAX_JOINTS


def _write_slot(bus: WorldStateRingBuffer, frame_index: int, **kwargs: object) -> None:
    with bus.acquire_write_view() as view:
        view.timestamp_ns = frame_index * BUS_PERIOD_NS
        view.sequence_id = frame_index + 1
        view.frame_id = frame_index
        view.surprise_scalar = np.float32(0.0)
        view.joint_positions_rad[...] = np.float32(frame_index * 0.001)
        for key, val in kwargs.items():
            view.raw[key] = val  # type: ignore[index]


def test_scheduled_skill_records_300_frames_without_drops(tmp_path: Path) -> None:
    bus = WorldStateRingBuffer(capacity=400)
    recorder = ProceduralMemoryRecorder(surprise_threshold=0.35)
    recorder.start_scheduled_skill("pick")
    for i in range(300):
        _write_slot(bus, i)
        snap = bus.read_latest()
        assert snap is not None
        recorder.on_bus_slot(snap.slot)
    recorder.stop_manual_recording()
    assert recorder.dropped_frames == 0
    assert len(recorder.episodes) == 1
    assert len(recorder.episodes[0].frames) == 300
    recorder.save_dataset(tmp_path)
    loaded = ProceduralMemoryRecorder.load_dataset(tmp_path)
    assert len(loaded) == 1
    assert len(loaded[0].frames) == 300
    assert (tmp_path / "meta" / "info.json").is_file()
    assert list((tmp_path / "data").glob("episode_*.npz"))


def test_compliant_guidance_and_surprise_triggers(tmp_path: Path) -> None:
    recorder = ProceduralMemoryRecorder(surprise_threshold=0.25)
    slot = empty_slot_array(1)[0]
    slot["timestamp_ns"] = 0
    slot["external_contact_flags"][0] = np.uint32(1)
    recorder.on_bus_slot(slot)
    assert recorder.episodes or recorder.on_bus_slot(slot) is not None

    slot2 = empty_slot_array(1)[0]
    slot2["timestamp_ns"] = BUS_PERIOD_NS
    slot2["surprise_scalar"] = np.float32(0.9)
    recorder.on_bus_slot(slot2)
    recorder.save_dataset(tmp_path)
    loaded = ProceduralMemoryRecorder.load_dataset(tmp_path)
    assert loaded


def test_offline_consolidator_lowers_residual_and_benchmarks() -> None:
    hjepa = HJEPAWorldModelEngine()
    proposer = GR00TActionProposer()
    consolidator = OfflineSkillConsolidator(hjepa=hjepa, proposer=proposer)

    frames: list[LeRobotEpisodeFrame] = []
    z = np.zeros(256, dtype=np.float32)
    rng = np.random.default_rng(11)
    act_delta = np.full(MAX_JOINTS, 0.02, dtype=np.float32)
    for i in range(40):
        q = np.full(MAX_JOINTS, float(i) * 0.01, dtype=np.float32)
        act = q + act_delta
        z_pred = hjepa.predict_self(z, act_delta)
        z_next = (z_pred + rng.normal(0.0, 0.02, size=256)).astype(np.float32)
        frames.append(
            LeRobotEpisodeFrame(
                timestamp_ns=i * BUS_PERIOD_NS,
                joint_positions_rad=q.copy(),
                joint_velocities_rad_s=np.zeros(MAX_JOINTS, dtype=np.float32),
                joint_loads_nm_or_ma=np.zeros(MAX_JOINTS, dtype=np.float32),
                latent_z_t=z.copy(),
                action_positions_rad=act.copy(),
                action_velocities_rad_s=np.zeros(MAX_JOINTS, dtype=np.float32),
                surprise_scalar=0.0,
                compliant_guidance_flag=True,
                trigger_reason="compliant_guidance",
                skill_name="pick",
            ),
        )
        z = z_next
    episode = LeRobotV2Episode(episode_index=0, skill_name="pick", frames=frames)
    before = consolidator.prediction_residual(episode)
    consolidator.consolidate_episodes([episode])
    after = consolidator.prediction_residual(episode)
    assert after < before

    rng = np.random.default_rng(7)
    for skill in ("pick", "place", "hand_over"):
        placements = [
            np.array(
                [0.52 + rng.uniform(-0.03, 0.03), rng.uniform(-0.05, 0.05), 1.04],
                dtype=np.float32,
            )
            for _ in range(20)
        ]
        rate = consolidator.evaluate_skill_placements(
            skill,  # type: ignore[arg-type]
            placements,
            bus_factory=lambda: WorldStateRingBuffer(capacity=32),
        )
        assert rate >= 0.85
