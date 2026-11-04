"""LeRobot v2 procedural memory capture and offline skill consolidation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pib_pro.layer1_autopilot.visuomotor_hybrid import (
    GR00TActionProposer,
    HybridVisuomotorController,
    inverse_kinematics_joints,
)
from pib_pro.world_model.hjepa_engine import HJEPAWorldModelEngine
from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer
from pib_pro.world_state_bus.schema import LATENT_DIM, MAX_JOINTS

SkillName = Literal["pick", "place", "hand_over"]
TriggerReason = Literal[
    "compliant_guidance",
    "surprise_spike",
    "teleop_demo",
    "scheduled_skill",
]


@dataclass(frozen=True, slots=True)
class LeRobotEpisodeFrame:
    timestamp_ns: int
    joint_positions_rad: NDArray[np.float32]
    joint_velocities_rad_s: NDArray[np.float32]
    joint_loads_nm_or_ma: NDArray[np.float32]
    latent_z_t: NDArray[np.float32]
    action_positions_rad: NDArray[np.float32]
    action_velocities_rad_s: NDArray[np.float32]
    surprise_scalar: float
    compliant_guidance_flag: bool
    trigger_reason: TriggerReason
    skill_name: SkillName


@dataclass(slots=True)
class LeRobotV2Episode:
    episode_index: int
    skill_name: SkillName
    frames: list[LeRobotEpisodeFrame] = field(default_factory=list)

    def append(self, frame: LeRobotEpisodeFrame) -> None:
        self.frames.append(frame)


@dataclass(slots=True)
class ProceduralMemoryRecorder:
    """30 Hz bus-attached episode recorder with LeRobot v2 export."""

    surprise_threshold: float = 0.35
    bus: WorldStateRingBuffer | None = None
    active_skill: SkillName | None = None
    teleop_active: bool = False
    scheduled_skill_active: bool = False
    _current: LeRobotV2Episode | None = field(default=None, init=False)
    _episodes: list[LeRobotV2Episode] = field(default_factory=list, init=False)
    _next_episode_index: int = field(default=0, init=False)
    _frames_seen: int = field(default=0, init=False)
    _frames_recorded: int = field(default=0, init=False)

    @property
    def episodes(self) -> tuple[LeRobotV2Episode, ...]:
        return tuple(self._episodes)

    @property
    def dropped_frames(self) -> int:
        return max(0, self._frames_seen - self._frames_recorded)

    def start_teleop_demo(self, skill_name: SkillName) -> None:
        self.teleop_active = True
        self.active_skill = skill_name
        self._begin_episode(skill_name, "teleop_demo")

    def start_scheduled_skill(self, skill_name: SkillName) -> None:
        self.scheduled_skill_active = True
        self.active_skill = skill_name
        self._begin_episode(skill_name, "scheduled_skill")

    def stop_manual_recording(self) -> None:
        self.teleop_active = False
        self.scheduled_skill_active = False
        self._finalize_episode()

    def _begin_episode(self, skill_name: SkillName, reason: TriggerReason) -> None:
        if self._current is not None:
            self._finalize_episode()
        self._current = LeRobotV2Episode(
            episode_index=self._next_episode_index,
            skill_name=skill_name,
        )
        self._next_episode_index += 1
        _ = reason

    def _finalize_episode(self) -> None:
        if self._current is None or not self._current.frames:
            self._current = None
            return
        self._episodes.append(self._current)
        self._current = None

    def _external_contact_active(self, slot: np.ndarray) -> bool:
        flags = slot["external_contact_flags"]
        return bool(np.any(flags[:MAX_JOINTS] > 0))

    def _capture_frame(
        self,
        slot: np.ndarray,
        trigger_reason: TriggerReason,
        skill_name: SkillName,
    ) -> LeRobotEpisodeFrame:
        action_pos = np.asarray(slot["joint_positions_rad"], dtype=np.float32).copy()
        action_vel = np.asarray(slot["joint_velocities_rad_s"], dtype=np.float32).copy()
        return LeRobotEpisodeFrame(
            timestamp_ns=int(slot["timestamp_ns"]),
            joint_positions_rad=np.asarray(slot["joint_positions_rad"], dtype=np.float32).copy(),
            joint_velocities_rad_s=np.asarray(
                slot["joint_velocities_rad_s"],
                dtype=np.float32,
            ).copy(),
            joint_loads_nm_or_ma=np.asarray(
                slot["joint_loads_nm_or_ma"],
                dtype=np.float32,
            ).copy(),
            latent_z_t=np.asarray(slot["z_t"], dtype=np.float32).copy(),
            action_positions_rad=action_pos,
            action_velocities_rad_s=action_vel,
            surprise_scalar=float(slot["surprise_scalar"]),
            compliant_guidance_flag=trigger_reason == "compliant_guidance",
            trigger_reason=trigger_reason,
            skill_name=skill_name,
        )

    def on_bus_slot(self, slot: np.ndarray) -> LeRobotEpisodeFrame | None:
        """Ingest one committed 30 Hz bus slot; returns frame if appended to an episode."""
        self._frames_seen += 1
        skill: SkillName = self.active_skill or "pick"
        trigger: TriggerReason | None = None

        if self.teleop_active:
            trigger = "teleop_demo"
            if self._current is None:
                self._begin_episode(skill, trigger)
        elif self.scheduled_skill_active:
            trigger = "scheduled_skill"
            if self._current is None:
                self._begin_episode(skill, trigger)
        elif self._external_contact_active(slot):
            trigger = "compliant_guidance"
            if self._current is None:
                self._begin_episode(skill, trigger)
        elif float(slot["surprise_scalar"]) >= self.surprise_threshold:
            trigger = "surprise_spike"
            if self._current is None:
                self._begin_episode(skill, trigger)

        if self._current is None:
            return None

        reason = trigger if trigger is not None else self._current.frames[0].trigger_reason
        frame = self._capture_frame(slot, reason, self._current.skill_name)
        self._current.append(frame)
        self._frames_recorded += 1
        return frame

    def save_dataset(self, root: Path) -> None:
        self._finalize_episode()
        root.mkdir(parents=True, exist_ok=True)
        meta_dir = root / "meta"
        data_dir = root / "data"
        meta_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)
        info = {
            "codebase_version": "v2.0",
            "fps": 30,
            "latent_dim": LATENT_DIM,
            "joints": MAX_JOINTS,
            "episodes": len(self._episodes),
        }
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        for ep in self._episodes:
            self._write_episode_npz(data_dir, ep)

    @staticmethod
    def _write_episode_npz(data_dir: Path, episode: LeRobotV2Episode) -> None:
        if not episode.frames:
            return
        name = f"episode_{episode.episode_index:06d}.npz"
        ts = np.array([f.timestamp_ns for f in episode.frames], dtype=np.int64)
        jp = np.stack([f.joint_positions_rad for f in episode.frames])
        jv = np.stack([f.joint_velocities_rad_s for f in episode.frames])
        jl = np.stack([f.joint_loads_nm_or_ma for f in episode.frames])
        zt = np.stack([f.latent_z_t for f in episode.frames])
        ap = np.stack([f.action_positions_rad for f in episode.frames])
        av = np.stack([f.action_velocities_rad_s for f in episode.frames])
        surprise = np.array([f.surprise_scalar for f in episode.frames], dtype=np.float32)
        guidance = np.array(
            [f.compliant_guidance_flag for f in episode.frames],
            dtype=np.bool_,
        )
        np.savez_compressed(
            data_dir / name,
            timestamp_ns=ts,
            joint_positions_rad=jp,
            joint_velocities_rad_s=jv,
            joint_loads_nm_or_ma=jl,
            latent_z_t=zt,
            action_positions_rad=ap,
            action_velocities_rad_s=av,
            surprise_scalar=surprise,
            compliant_guidance_flag=guidance,
            skill_name=episode.skill_name,
            trigger_reason=episode.frames[0].trigger_reason,
        )

    @classmethod
    def load_dataset(cls, root: Path) -> list[LeRobotV2Episode]:
        data_dir = root / "data"
        episodes: list[LeRobotV2Episode] = []
        for path in sorted(data_dir.glob("episode_*.npz")):
            ep_idx = int(path.stem.split("_")[1])
            data = np.load(path, allow_pickle=False)
            skill = str(data["skill_name"]) if "skill_name" in data else "pick"
            trigger = str(data["trigger_reason"]) if "trigger_reason" in data else "teleop_demo"
            skill_name: SkillName = (
                skill if skill in ("pick", "place", "hand_over") else "pick"  # type: ignore[assignment]
            )
            trigger_reason: TriggerReason
            if trigger in (
                "compliant_guidance",
                "surprise_spike",
                "teleop_demo",
                "scheduled_skill",
            ):
                trigger_reason = trigger  # type: ignore[assignment]
            else:
                trigger_reason = "teleop_demo"
            episode = LeRobotV2Episode(episode_index=ep_idx, skill_name=skill_name)
            n = int(data["timestamp_ns"].shape[0])
            for i in range(n):
                episode.append(
                    LeRobotEpisodeFrame(
                        timestamp_ns=int(data["timestamp_ns"][i]),
                        joint_positions_rad=np.asarray(
                            data["joint_positions_rad"][i],
                            dtype=np.float32,
                        ),
                        joint_velocities_rad_s=np.asarray(
                            data["joint_velocities_rad_s"][i],
                            dtype=np.float32,
                        ),
                        joint_loads_nm_or_ma=np.asarray(
                            data["joint_loads_nm_or_ma"][i],
                            dtype=np.float32,
                        ),
                        latent_z_t=np.asarray(data["latent_z_t"][i], dtype=np.float32),
                        action_positions_rad=np.asarray(
                            data["action_positions_rad"][i],
                            dtype=np.float32,
                        ),
                        action_velocities_rad_s=np.asarray(
                            data["action_velocities_rad_s"][i],
                            dtype=np.float32,
                        ),
                        surprise_scalar=float(data["surprise_scalar"][i]),
                        compliant_guidance_flag=bool(data["compliant_guidance_flag"][i]),
                        trigger_reason=trigger_reason,
                        skill_name=skill_name,
                    ),
                )
            episodes.append(episode)
        return episodes


@dataclass(slots=True)
class OfflineSkillConsolidator:
    """Offline update of H-JEPA dynamics residual and GR00T skill LoRA adapters."""

    hjepa: HJEPAWorldModelEngine
    proposer: GR00TActionProposer
    visuomotor: HybridVisuomotorController | None = None

    def _action_delta_from_frame(self, frame: LeRobotEpisodeFrame) -> NDArray[np.float32]:
        delta = frame.action_positions_rad - frame.joint_positions_rad
        return delta.astype(np.float32)

    def consolidate_episodes(self, episodes: list[LeRobotV2Episode]) -> None:
        if not episodes:
            return
        skill_vectors: dict[str, list[NDArray[np.float32]]] = {
            "pick": [],
            "place": [],
            "hand_over": [],
        }
        for ep in episodes:
            for i, frame in enumerate(ep.frames):
                delta = self._action_delta_from_frame(frame)
                skill_vectors[ep.skill_name].append(delta)
                if i + 1 < len(ep.frames):
                    nxt = ep.frames[i + 1]
                    act = delta
                    z_pred = self.hjepa.predict_self(frame.latent_z_t, act)
                    err = nxt.latent_z_t - z_pred
                    self.hjepa.nudge_action_projection(err, act, learning_rate=0.35)
        for skill, vecs in skill_vectors.items():
            if vecs:
                lora = np.mean(np.stack(vecs), axis=0).astype(np.float32)
                self.proposer.set_skill_lora_adapter(skill, lora)
        if skill_vectors["pick"] and not skill_vectors["place"]:
            lora = np.mean(np.stack(skill_vectors["pick"]), axis=0).astype(np.float32)
            for skill in ("place", "hand_over"):
                self.proposer.set_skill_lora_adapter(skill, lora)

    def prediction_residual(
        self,
        episode: LeRobotV2Episode,
    ) -> float:
        if len(episode.frames) < 2:
            return 0.0
        errs: list[float] = []
        for i in range(len(episode.frames) - 1):
            frame = episode.frames[i]
            nxt = episode.frames[i + 1]
            act = frame.action_positions_rad - frame.joint_positions_rad
            z_pred = self.hjepa.predict_self(frame.latent_z_t, act)
            errs.append(float(np.linalg.norm(nxt.latent_z_t - z_pred)))
        return float(np.mean(errs))

    def evaluate_skill_placements(
        self,
        skill_name: SkillName,
        placements: list[NDArray[np.float32]],
        *,
        bus_factory: object | None = None,
    ) -> float:
        """Return success rate in ``[0, 1]`` for unseen object placements."""
        if self.visuomotor is None:
            self.visuomotor = HybridVisuomotorController(
                hjepa=HJEPAWorldModelEngine(),
                proposer=self.proposer,
            )
        controller = self.visuomotor
        successes = 0
        for target in placements:
            bus = bus_factory() if callable(bus_factory) else WorldStateRingBuffer(capacity=32)
            _seed_reach_scene(bus, target, skill_name, hjepa=controller.hjepa)
            err, _ = controller.simulate_reach(bus, skill_name=skill_name, max_steps=180)
            if err <= 0.025:
                successes += 1
        return successes / max(1, len(placements))


def _seed_reach_scene(
    bus: WorldStateRingBuffer,
    target_xyz: NDArray[np.float32],
    skill_name: SkillName,
    hjepa: HJEPAWorldModelEngine | None = None,
) -> None:
    with bus.acquire_write_view() as view:
        view.timestamp_ns = 0
        view.sequence_id = 1
        q = np.zeros(MAX_JOINTS, dtype=np.float32)
        q[0] = -0.2
        q[1] = 0.0
        q[2] = -0.1
        view.joint_positions_rad[...] = q
        view.raw["joint_velocities_rad_s"][...] = 0.0
        view.objects[0]["object_id"] = np.uint64(1)
        view.objects[0]["position_xyz"] = np.asarray(target_xyz, dtype=np.float32)
        view.objects[1]["object_id"] = np.uint64(0)
        target = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        q_target = inverse_kinematics_joints(target, q)
        if hjepa is not None:
            scene = np.zeros(MAX_JOINTS, dtype=np.float32)
            scene[:3] = target
            zero_v = np.zeros(MAX_JOINTS, dtype=np.float32)
            z_t = hjepa.encode(scene, q, zero_v)
            z_goal = hjepa.encode(scene, q_target, zero_v)
            view.z_t[...] = z_t
            view.raw["z_goal"][...] = z_goal
        else:
            view.z_t[...] = 0.0
            view.raw["z_goal"][...] = 0.0
        view.surprise_scalar = 0.0
    _ = skill_name
