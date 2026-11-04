"""Hierarchical System-3 deliberation with H-JEPA Level-2 subgoal planning."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pib_pro.layer2_executive.surprise_bt import BTSkillStep
from pib_pro.layer3_cognition.plans import SubgoalPlan
from pib_pro.world_state_bus.schema import LATENT_DIM, MAX_OBJECTS

REP103_REACH_RADIUS_M: float = 0.75
FEASIBILITY_BLOCK_THRESHOLD: float = 0.5
OBSTACLE_CLEAR_SKILL: str = "clear_blocking_obstacle"

DEFAULT_TASK_STAGES: tuple[str, ...] = (
    "reach_target",
    "grasp_target",
    "lift_target",
    "move_to_bin",
    "release_target",
)


@dataclass(frozen=True, slots=True)
class Level2Subgoal:
    stage_name: str
    z_goal: NDArray[np.float32]
    feasibility_score: float
    blocking_reason: str = ""


@dataclass(frozen=True, slots=True)
class DeliberationResult:
    plan: SubgoalPlan
    bt_steps: tuple[BTSkillStep, ...]
    blocking_audio_ms: float
    used_cloud_planner: bool
    spoke_clarification: bool


class GeminiRoboticsER2AdapterProtocol(Protocol):
    def plan_remote(
        self,
        task_stages: tuple[str, ...],
        scene_summary: str,
    ) -> tuple[str, ...]: ...


@dataclass(slots=True)
class HJEPAHighLevelSubgoalPlanner:
    """H-JEPA Level-2: symbolic stages to latent subgoals with feasibility checks."""

    reach_radius_m: float = REP103_REACH_RADIUS_M

    def _target_position(
        self,
        slot: np.ndarray,
        target_object_id: int,
    ) -> NDArray[np.float32] | None:
        objs = slot["objects"]
        for i in range(MAX_OBJECTS):
            if int(objs[i]["object_id"]) == target_object_id:
                return np.asarray(objs[i]["position_xyz"], dtype=np.float32)
        return None

    def _feasibility_for_stage(
        self,
        stage: str,
        target_xyz: NDArray[np.float32],
        slot: np.ndarray,
        *,
        target_object_id: int,
    ) -> tuple[float, str]:
        dist = float(np.linalg.norm(target_xyz))
        if dist > self.reach_radius_m:
            return 0.0, f"{stage}: target outside {self.reach_radius_m} m workspace"
        for i in range(MAX_OBJECTS):
            oid = int(slot["objects"][i]["object_id"])
            if oid == 0 or oid == target_object_id:
                continue
            pos = np.asarray(slot["objects"][i]["position_xyz"], dtype=np.float32)
            sep = float(np.linalg.norm(pos - target_xyz))
            if sep < 0.12 and float(slot["objects"][i]["confidence"]) > 0.5:
                return 0.25, f"{stage}: obstacle blocks target (object {oid})"
        return 1.0, ""

    def plan_subgoals(
        self,
        task_stages: tuple[str, ...],
        slot: np.ndarray,
        *,
        target_object_id: int,
    ) -> tuple[Level2Subgoal, ...]:
        target_xyz = self._target_position(slot, target_object_id)
        if target_xyz is None:
            target_xyz = np.array([0.4, 0.0, 0.7], dtype=np.float32)
        subgoals: list[Level2Subgoal] = []
        for idx, stage in enumerate(task_stages):
            score, reason = self._feasibility_for_stage(
                stage,
                target_xyz,
                slot,
                target_object_id=target_object_id,
            )
            z = np.zeros(LATENT_DIM, dtype=np.float32)
            z[0] = float(idx)
            z[1:4] = target_xyz
            z[4] = score
            subgoals.append(
                Level2Subgoal(
                    stage_name=stage,
                    z_goal=z,
                    feasibility_score=score,
                    blocking_reason=reason,
                )
            )
        return tuple(subgoals)

    def first_infeasible(self, subgoals: tuple[Level2Subgoal, ...]) -> Level2Subgoal | None:
        for sg in subgoals:
            if sg.feasibility_score < FEASIBILITY_BLOCK_THRESHOLD:
                return sg
        return None


@dataclass(slots=True)
class HierarchicalSystem3Planner:
    """Non-blocking System-3 planner with cloud adapter and local VLM fallback."""

    hjepa_planner: HJEPAHighLevelSubgoalPlanner = field(
        default_factory=HJEPAHighLevelSubgoalPlanner
    )
    cloud_adapter: GeminiRoboticsER2AdapterProtocol | None = None
    cloud_available: bool = True
    budget_remaining: int = 100
    blocking_audio_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _last_result: DeliberationResult | None = field(default=None, init=False)

    def _local_stage_names(self, task_stages: tuple[str, ...]) -> tuple[str, ...]:
        return task_stages

    def _resolve_stages(
        self,
        task_stages: tuple[str, ...],
        scene_summary: str,
    ) -> tuple[tuple[str, ...], bool]:
        if self.cloud_adapter is not None and self.cloud_available and self.budget_remaining > 0:
            try:
                remote = self.cloud_adapter.plan_remote(task_stages, scene_summary)
                self.budget_remaining -= 1
                return remote, True
            except OSError:
                pass
        return self._local_stage_names(task_stages), False

    def deliberate_sync(
        self,
        slot: np.ndarray,
        *,
        task_stages: tuple[str, ...] = DEFAULT_TASK_STAGES,
        target_object_id: int = 1,
        scene_summary: str = "",
        plan_id: str = "sys3",
    ) -> DeliberationResult:
        stages, used_cloud = self._resolve_stages(task_stages, scene_summary)
        subgoals = self.hjepa_planner.plan_subgoals(stages, slot, target_object_id=target_object_id)
        blocked = self.hjepa_planner.first_infeasible(subgoals)
        bt_steps: list[BTSkillStep] = []
        spoke = False
        stage_names = [sg.stage_name for sg in subgoals]
        if blocked is not None:
            clear_z = np.zeros(LATENT_DIM, dtype=np.float64)
            clear_z[1:4] = blocked.z_goal[1:4]
            bt_steps.append(
                BTSkillStep(
                    step_id=f"{plan_id}-clear",
                    skill_name=OBSTACLE_CLEAR_SKILL,
                    z_goal=clear_z,
                    recovery_utterance=("Something is blocking the target; I'll clear it first."),
                )
            )
            spoke = True
            stage_names = [OBSTACLE_CLEAR_SKILL, *stage_names]
        else:
            for sg in subgoals:
                bt_steps.append(
                    BTSkillStep(
                        step_id=f"{plan_id}-{sg.stage_name}",
                        skill_name=sg.stage_name,
                        z_goal=np.asarray(sg.z_goal, dtype=np.float64),
                    )
                )
        plan = SubgoalPlan(
            plan_id=plan_id,
            subgoals=tuple(stage_names),
            horizon_s=float(len(stage_names)) * 2.5,
        )
        result = DeliberationResult(
            plan=plan,
            bt_steps=tuple(bt_steps),
            blocking_audio_ms=0.0,
            used_cloud_planner=used_cloud,
            spoke_clarification=spoke,
        )
        with self._lock:
            self._last_result = result
        return result

    def plan_async(
        self,
        slot: np.ndarray,
        *,
        task_stages: tuple[str, ...] = DEFAULT_TASK_STAGES,
        target_object_id: int = 1,
        scene_summary: str = "",
        plan_id: str = "sys3",
        on_complete: Callable[[DeliberationResult], None] | None = None,
    ) -> threading.Thread:
        """Fire deliberation on a background thread (never blocks Layer-1 / duplex audio)."""

        def _run() -> None:
            result = self.deliberate_sync(
                slot,
                task_stages=task_stages,
                target_object_id=target_object_id,
                scene_summary=scene_summary,
                plan_id=plan_id,
            )
            if on_complete is not None:
                on_complete(result)

        thread = threading.Thread(target=_run, name="system3-deliberation", daemon=True)
        thread.start()
        return thread

    def plan_task(
        self,
        slot: np.ndarray,
        *,
        task_stages: tuple[str, ...] = DEFAULT_TASK_STAGES,
        target_object_id: int = 1,
        scene_summary: str = "",
        plan_id: str = "sys3",
        on_complete: Callable[[DeliberationResult], None] | None = None,
    ) -> threading.Thread:
        """Alias for ``plan_async``."""
        return self.plan_async(
            slot,
            task_stages=task_stages,
            target_object_id=target_object_id,
            scene_summary=scene_summary,
            plan_id=plan_id,
            on_complete=on_complete,
        )

    @property
    def last_result(self) -> DeliberationResult | None:
        with self._lock:
            return self._last_result
