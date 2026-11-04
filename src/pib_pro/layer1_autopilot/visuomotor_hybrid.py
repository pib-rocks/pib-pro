"""25 Hz hybrid visuomotor: GR00T chunk proposal + H-JEPA rollout scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pib_pro.layer0_reflex.somatic_reflex import SafeMotorCommand, SomaticReflexController
from pib_pro.world_model.hjepa_engine import HJEPAWorldModelEngine
from pib_pro.world_state_bus.schema import MAX_JOINTS, MAX_OBJECTS

if TYPE_CHECKING:
    from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer

VISUOMOTOR_HZ: float = 25.0
VISUOMOTOR_DT_S: float = 1.0 / VISUOMOTOR_HZ
DEFAULT_CHUNK_COUNT: int = 8
DEFAULT_HORIZON: int = 20
OBSTACLE_CLEARANCE_M: float = 0.08
DEFAULT_OBSTACLE_RADIUS_M: float = 0.045
EE_BASE_XYZ: tuple[float, float, float] = (0.30, 0.0, 0.90)
EE_SCALE_XYZ: tuple[float, float, float] = (0.45, 0.45, 0.45)


def _minimum_jerk_s(t: NDArray[np.float32]) -> NDArray[np.float32]:
    tt = np.clip(t, 0.0, 1.0)
    out: NDArray[np.float32] = (10.0 * tt**3 - 15.0 * tt**4 + 6.0 * tt**5).astype(np.float32)
    return out


def forward_kinematics_ee(joint_positions_rad: NDArray[np.float32]) -> NDArray[np.float32]:
    """Deterministic REP-103 end-effector proxy from the first three arm joints."""
    q = np.asarray(joint_positions_rad, dtype=np.float32).reshape(-1)
    bx, by, bz = EE_BASE_XYZ
    sx, sy, sz = EE_SCALE_XYZ
    return np.array(
        [
            bx + sx * float(q[0]),
            by + sy * float(q[1]),
            bz + sz * float(q[2]),
        ],
        dtype=np.float32,
    )


def inverse_kinematics_joints(
    ee_xyz: NDArray[np.float32],
    q_current: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Map a 3D target to joint increments (remaining joints hold current values)."""
    q = np.asarray(q_current, dtype=np.float32).reshape(-1).copy()
    if q.size < MAX_JOINTS:
        pad = np.zeros(MAX_JOINTS, dtype=np.float32)
        pad[: q.size] = q
        q = pad
    bx, by, bz = EE_BASE_XYZ
    sx, sy, sz = EE_SCALE_XYZ
    target = np.asarray(ee_xyz, dtype=np.float32).reshape(3)
    q[0] = (target[0] - bx) / sx
    q[1] = (target[1] - by) / sy
    q[2] = (target[2] - bz) / sz
    return q.astype(np.float32)


@dataclass(slots=True)
class GR00TActionProposer:
    """GR00T N1.7 action head (NumPy CI backend + optional TensorRT hook)."""

    num_candidates: int = DEFAULT_CHUNK_COUNT
    horizon: int = DEFAULT_HORIZON
    control_dt_s: float = VISUOMOTOR_DT_S
    _tensorrt_loaded: bool = field(default=False, init=False)
    _skill_residuals: dict[str, NDArray[np.float32]] = field(default_factory=dict, init=False)

    def load_tensorrt_nvfp4_engine(self, engine_bytes: bytes | None = None) -> None:
        """Mark TensorRT NVFP4 backend ready (Thor deployment)."""
        _ = engine_bytes
        self._tensorrt_loaded = True

    def set_skill_lora_adapter(self, skill_name: str, delta: NDArray[np.float32]) -> None:
        arr = np.asarray(delta, dtype=np.float32).reshape(-1)
        if arr.size != MAX_JOINTS:
            msg = f"skill LoRA delta must have {MAX_JOINTS} elements"
            raise ValueError(msg)
        self._skill_residuals[skill_name] = arr

    def _arc_offset_for_candidate(self, candidate_index: int) -> NDArray[np.float32]:
        patterns: dict[int, tuple[float, float, float]] = {
            0: (0.0, 0.0, 0.0),
            1: (0.0, 0.14, 0.0),
            2: (0.0, -0.14, 0.0),
            3: (0.0, 0.0, 0.14),
            4: (0.0, 0.20, 0.05),
            5: (0.0, 0.10, 0.10),
            6: (0.0, -0.18, 0.08),
            7: (0.0, 0.16, 0.12),
        }
        ox, oy, oz = patterns.get(candidate_index % 8, (0.0, 0.12, 0.0))
        return np.array([ox, oy, oz], dtype=np.float32)

    def propose(
        self,
        q_t: NDArray[np.float32],
        target_xyz: NDArray[np.float32],
        z_goal: NDArray[np.float32],
        *,
        skill_name: str | None = None,
        num_candidates: int | None = None,
        horizon: int | None = None,
    ) -> NDArray[np.float32]:
        """Return candidate joint position chunks ``(K, H, 32)``."""
        _ = z_goal  # conditions diffusion prior in TensorRT path
        k = num_candidates if num_candidates is not None else self.num_candidates
        h = horizon if horizon is not None else self.horizon
        q0 = np.asarray(q_t, dtype=np.float32).reshape(-1)
        if q0.size < MAX_JOINTS:
            padded = np.zeros(MAX_JOINTS, dtype=np.float32)
            padded[: q0.size] = q0
            q0 = padded
        start_ee = forward_kinematics_ee(q0)
        goal_ee = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        _ = inverse_kinematics_joints(goal_ee, q0)
        residual = np.zeros(MAX_JOINTS, dtype=np.float32)
        if skill_name is not None and skill_name in self._skill_residuals:
            residual = self._skill_residuals[skill_name]
        chunks = np.zeros((k, h, MAX_JOINTS), dtype=np.float32)
        for ci in range(k):
            arc = self._arc_offset_for_candidate(ci)
            speed_scale = 1.0 if ci != 5 else 1.25
            for hi in range(h):
                frac = min(1.0, float(hi + 1) / float(h) * speed_scale)
                base = start_ee + (goal_ee - start_ee) * frac
                bulge = arc * np.sin(np.pi * frac)
                ee = base + bulge
                q_step = inverse_kinematics_joints(ee, q0)
                if skill_name is not None and skill_name in self._skill_residuals:
                    blend = (1.0 - frac) * 0.08
                    q_step[:3] = q_step[:3] + residual[:3] * blend
                chunks[ci, hi, :] = q_step
        return chunks


@dataclass(frozen=True, slots=True)
class HybridVisuomotorStepResult:
    safe_command: SafeMotorCommand
    best_chunk_index: int
    composite_cost: float
    latent_goal_cost: float
    obstacle_collision_cost: float
    smoothness_jerk_cost: float
    min_path_clearance_m: float
    elapsed_s: float


@dataclass(slots=True)
class HybridVisuomotorController:
    """Layer-1b visuomotor loop: propose, score, bus update, Layer-0 gate."""

    hjepa: HJEPAWorldModelEngine
    proposer: GR00TActionProposer = field(default_factory=GR00TActionProposer)
    reflex: SomaticReflexController = field(default_factory=SomaticReflexController)
    obstacle_clearance_m: float = OBSTACLE_CLEARANCE_M
    target_object_index: int = 0
    latent_weight: float = 1.0
    obstacle_weight: float = 120.0
    smoothness_weight: float = 0.05
    _last_min_clearance_m: float = field(default=1.0, init=False)

    def _iter_obstacle_spheres(
        self,
        slot_objects: np.ndarray,
        *,
        skip_object_index: int,
    ) -> list[tuple[NDArray[np.float32], float]]:
        obstacles: list[tuple[NDArray[np.float32], float]] = []
        for idx in range(min(MAX_OBJECTS, slot_objects.shape[0])):
            if idx == skip_object_index:
                continue
            obj = slot_objects[idx]
            if int(obj["object_id"]) == 0:
                continue
            pos = np.asarray(obj["position_xyz"], dtype=np.float32)
            if not np.all(np.isfinite(pos)):
                continue
            obstacles.append((pos, DEFAULT_OBSTACLE_RADIUS_M))
        return obstacles

    def _path_clearance_m(
        self,
        joint_path: NDArray[np.float32],
        obstacles: list[tuple[NDArray[np.float32], float]],
    ) -> float:
        if not obstacles:
            return 1.0
        min_clear = 1.0
        for row in joint_path:
            ee = forward_kinematics_ee(row)
            for center, radius in obstacles:
                dist = float(np.linalg.norm(ee - center)) - radius
                min_clear = min(min_clear, dist)
        self._last_min_clearance_m = min_clear
        return min_clear

    def _obstacle_cost(
        self,
        joint_path: NDArray[np.float32],
        obstacles: list[tuple[NDArray[np.float32], float]],
    ) -> float:
        clearance = self._path_clearance_m(joint_path, obstacles)
        if clearance >= self.obstacle_clearance_m:
            return 0.0
        deficit = self.obstacle_clearance_m - clearance
        return float(deficit * deficit)

    @staticmethod
    def _smoothness_cost(chunk: NDArray[np.float32]) -> float:
        if chunk.shape[0] < 3:
            return 0.0
        vel = np.diff(chunk, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        return float(np.sum(acc * acc) + np.sum(jerk * jerk))

    def score_candidates(
        self,
        z_t: NDArray[np.float32],
        z_goal: NDArray[np.float32],
        chunks: NDArray[np.float32],
        obstacles: list[tuple[NDArray[np.float32], float]],
    ) -> tuple[int, NDArray[np.float32], NDArray[np.float32], float]:
        z_term = self.hjepa.predict_self(z_t, chunks)
        latent_costs = np.linalg.norm(z_term - z_goal.reshape(1, -1), axis=1)
        total = np.zeros(chunks.shape[0], dtype=np.float32)
        min_clearances = np.zeros(chunks.shape[0], dtype=np.float32)
        smooth = np.zeros(chunks.shape[0], dtype=np.float32)
        for i in range(chunks.shape[0]):
            path = chunks[i]
            obs = self._obstacle_cost(path, obstacles)
            sm = self._smoothness_cost(path)
            min_clearances[i] = self._last_min_clearance_m
            smooth[i] = sm
            total[i] = (
                self.latent_weight * latent_costs[i]
                + self.obstacle_weight * obs
                + self.smoothness_weight * sm
            )
        best = int(np.argmin(total))
        return best, total, latent_costs, float(min_clearances[best])

    def step(
        self,
        bus: WorldStateRingBuffer,
        *,
        skill_name: str | None = None,
        stamp_monotonic_s: float | None = None,
    ) -> HybridVisuomotorStepResult:
        """One 25 Hz cycle: read bus, select chunk, write ``z_hat_self``, Layer-0 reflex."""
        t0 = time.perf_counter()
        latest = bus.read_latest()
        if latest is None:
            msg = "HybridVisuomotorController.step requires at least one bus slot"
            raise RuntimeError(msg)
        slot = latest.slot
        q_t = np.asarray(slot["joint_positions_rad"], dtype=np.float32)
        z_t = np.asarray(slot["z_t"], dtype=np.float32)
        z_goal = np.asarray(slot["z_goal"], dtype=np.float32)
        if float(np.linalg.norm(z_goal)) < 1e-6:
            z_goal = z_t.copy()
        target_xyz = np.asarray(
            slot["objects"][self.target_object_index]["position_xyz"],
            dtype=np.float32,
        )
        chunks = self.proposer.propose(
            q_t,
            target_xyz,
            z_goal,
            skill_name=skill_name,
        )
        obstacles = self._iter_obstacle_spheres(
            slot["objects"],
            skip_object_index=self.target_object_index,
        )
        best_idx, totals, latent_costs, min_clear = self.score_candidates(
            z_t,
            z_goal,
            chunks,
            obstacles,
        )
        best_chunk = chunks[best_idx]
        z_term = self.hjepa.predict_self(z_t, chunks)
        z_hat_best = z_term[best_idx]
        with bus.acquire_write_view() as view:
            view.fill_from(slot)
            view.raw["z_hat_self"] = z_hat_best

        next_q = best_chunk[0]
        prev_q = np.asarray(q_t, dtype=np.float32)
        vel = (next_q - prev_q) / VISUOMOTOR_DT_S
        n = MAX_JOINTS
        pos_tuple = tuple(float(next_q[i]) for i in range(n))
        vel_tuple = tuple(float(vel[i]) for i in range(n))
        loads = tuple(float(x) for x in slot["joint_loads_nm_or_ma"])
        meas_q = tuple(float(x) for x in slot["joint_positions_rad"])
        meas_v = tuple(float(x) for x in slot["joint_velocities_rad_s"])
        sts = tuple(120 for _ in range(n))
        safe = self.reflex.step(
            pos_tuple,
            vel_tuple,
            sts,
            meas_q,
            meas_v,
            loads,
            VISUOMOTOR_DT_S,
            bus=latest.view,
        )
        obs_cost = self._obstacle_cost(best_chunk, obstacles)
        sm_cost = self._smoothness_cost(best_chunk)
        elapsed = time.perf_counter() - t0
        _ = stamp_monotonic_s
        return HybridVisuomotorStepResult(
            safe_command=safe,
            best_chunk_index=best_idx,
            composite_cost=float(totals[best_idx]),
            latent_goal_cost=float(latent_costs[best_idx]),
            obstacle_collision_cost=obs_cost,
            smoothness_jerk_cost=sm_cost,
            min_path_clearance_m=min_clear,
            elapsed_s=elapsed,
        )

    def simulate_reach(
        self,
        bus: WorldStateRingBuffer,
        *,
        max_steps: int = 120,
        skill_name: str | None = None,
        mid_reach_target_shift_m: NDArray[np.float32] | None = None,
        shift_after_step: int = 60,
    ) -> tuple[float, float]:
        """Integrate open-loop chunk steps with closed-loop 25 Hz re-planning."""
        final_err = 1.0
        min_clear = 1.0
        for step_idx in range(max_steps):
            if mid_reach_target_shift_m is not None and step_idx == shift_after_step:
                shift = np.asarray(mid_reach_target_shift_m, dtype=np.float32).reshape(3)
                with bus.acquire_write_view() as view:
                    pos = view.objects[self.target_object_index]["position_xyz"]
                    pos[...] = pos + shift
            result = self.step(bus, skill_name=skill_name)
            min_clear = min(min_clear, result.min_path_clearance_m)
            latest = bus.read_latest()
            if latest is None:
                break
            target = np.asarray(
                latest.slot["objects"][self.target_object_index]["position_xyz"],
                dtype=np.float32,
            )
            with bus.acquire_write_view() as view:
                view.fill_from(latest.slot)
                new_q = np.asarray(latest.slot["joint_positions_rad"], dtype=np.float32).copy()
                cmd = result.safe_command.position_targets_rad
                for ji in range(min(len(cmd), MAX_JOINTS)):
                    new_q[ji] = float(cmd[ji])
                view.joint_positions_rad[...] = new_q
            ee = forward_kinematics_ee(new_q)
            final_err = float(np.linalg.norm(ee - target))
            if final_err <= 0.015:
                break
        return final_err, min_clear
