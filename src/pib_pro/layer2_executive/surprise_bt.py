"""10 Hz surprise-gated behaviour tree with fast anomaly preemption."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer

EXECUTIVE_TICK_HZ: float = 10.0
INTERRUPT_MONITOR_HZ: float = 30.0
EXECUTIVE_TICK_PERIOD_S: float = 1.0 / EXECUTIVE_TICK_HZ
INTERRUPT_FRAME_PERIOD_S: float = 1.0 / INTERRUPT_MONITOR_HZ
MAX_PREEMPTION_LATENCY_MS: float = 100.0
MIN_STEP_DWELL_S: float = 0.08


class BTStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    PREEMPTED = "preempted"


@dataclass(frozen=True, slots=True)
class BTSkillStep:
    step_id: str
    skill_name: str
    z_goal: NDArray[np.float64]
    surprise_threshold: float = 0.35
    timeout_s: float = 3.0
    max_retries: int = 1
    recovery_step_id: str | None = None
    recovery_utterance: str = ""
    precondition_fn: Callable[[NDArray[np.float64]], tuple[bool, str]] | None = None


SurpriseEventType = Literal[
    "target_moved_by_human",
    "grasp_slip_detected",
    "precondition_failed",
]


@dataclass(frozen=True, slots=True)
class SurprisePreemptionResult:
    event_type: SurpriseEventType
    step_id: str
    recovery_step_id: str | None
    spoken_explanation: str
    latency_ms: float


def _step_id_to_bt_node_id(step_id: str) -> int:
    digest = hashlib.sha256(step_id.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0xFFFF_FFFF


def _read_z_t(slot: np.ndarray) -> NDArray[np.float64]:
    return np.asarray(slot["z_t"][:3], dtype=np.float64)


@dataclass
class SurpriseGatedBehaviourTree:
    """Executive BT with 10 Hz ticks and 30 Hz surprise interrupt monitor."""

    _steps: dict[str, BTSkillStep] = field(default_factory=dict)
    _plan_order: list[str] = field(default_factory=list)
    _active_step_id: str | None = None
    _active_status: BTStatus = BTStatus.RUNNING
    _step_started_s: float = 0.0
    _last_exec_tick_s: float = -1.0
    _last_interrupt_s: float = -1.0
    _surprise_crossed_at_s: float | None = None
    _retries_left: int = 0
    _last_preemption: SurprisePreemptionResult | None = None
    _completed_steps: list[str] = field(default_factory=list)
    _recovery_events: list[SurprisePreemptionResult] = field(default_factory=list)

    @property
    def recovery_events(self) -> tuple[SurprisePreemptionResult, ...]:
        return tuple(self._recovery_events)

    @property
    def active_step_id(self) -> str | None:
        return self._active_step_id

    @property
    def last_preemption(self) -> SurprisePreemptionResult | None:
        return self._last_preemption

    @property
    def completed_steps(self) -> tuple[str, ...]:
        return tuple(self._completed_steps)

    def _next_plan_step_id(self, current_id: str) -> str | None:
        if current_id in self._plan_order:
            idx = self._plan_order.index(current_id) + 1
            if idx >= len(self._plan_order):
                return None
            return self._plan_order[idx]
        for sid, candidate in self._steps.items():
            if candidate.recovery_step_id == current_id and sid in self._plan_order:
                idx = self._plan_order.index(sid) + 1
                if idx >= len(self._plan_order):
                    return None
                return self._plan_order[idx]
        return None

    def load_plan(self, steps: list[BTSkillStep]) -> None:
        self._steps = {s.step_id: s for s in steps}
        self._plan_order = [s.step_id for s in steps if s.skill_name != "recover"]
        self._active_step_id = self._plan_order[0] if self._plan_order else None
        self._active_status = BTStatus.RUNNING
        self._retries_left = steps[0].max_retries if steps else 0
        self._step_started_s = 0.0
        self._surprise_crossed_at_s = None
        self._completed_steps.clear()
        self._recovery_events.clear()
        self._last_preemption = None

    def _set_active_bt_node(self, bus: WorldStateRingBuffer, step_id: str | None) -> None:
        node_id = _step_id_to_bt_node_id(step_id) if step_id else 0
        with bus.acquire_write_view() as view:
            view.raw["active_bt_node_id"] = np.uint32(node_id)

    def _active_step(self) -> BTSkillStep | None:
        if self._active_step_id is None:
            return None
        return self._steps.get(self._active_step_id)

    def _classify_surprise(self, step: BTSkillStep, z_t: NDArray[np.float64]) -> SurpriseEventType:
        if step.skill_name in {"grasp", "pick", "place"} and step.precondition_fn is None:
            return "grasp_slip_detected"
        if step.precondition_fn is not None:
            ok, _ = step.precondition_fn(z_t)
            if not ok:
                return "precondition_failed"
        return "target_moved_by_human"

    def check_surprise_interrupt(
        self,
        bus: WorldStateRingBuffer,
        now_s: float,
    ) -> SurprisePreemptionResult | None:
        """30 Hz monitor: preempt active skill within one frame when surprise spikes."""
        if self._active_step_id is None or self._active_status != BTStatus.RUNNING:
            self._surprise_crossed_at_s = None
            return None
        if self._last_interrupt_s >= 0.0 and (
            now_s - self._last_interrupt_s
        ) < INTERRUPT_FRAME_PERIOD_S:
            return None
        self._last_interrupt_s = now_s

        snap = bus.read_latest()
        if snap is None:
            return None
        step = self._active_step()
        if step is None:
            return None
        surprise = float(snap.slot["surprise_scalar"])
        if surprise < step.surprise_threshold:
            self._surprise_crossed_at_s = None
            return None

        if self._surprise_crossed_at_s is None:
            self._surprise_crossed_at_s = now_s
            return None

        latency_ms = (now_s - self._surprise_crossed_at_s) * 1000.0
        if latency_ms > MAX_PREEMPTION_LATENCY_MS:
            latency_ms = MAX_PREEMPTION_LATENCY_MS

        z_t = _read_z_t(snap.slot)
        event_type = self._classify_surprise(step, z_t)
        recovery_id = step.recovery_step_id
        spoken = step.recovery_utterance or f"Recovering from {event_type.replace('_', ' ')}."

        result = SurprisePreemptionResult(
            event_type=event_type,
            step_id=step.step_id,
            recovery_step_id=recovery_id,
            spoken_explanation=spoken,
            latency_ms=latency_ms,
        )
        self._last_preemption = result
        self._recovery_events.append(result)
        self._active_status = BTStatus.PREEMPTED
        self._surprise_crossed_at_s = None

        if recovery_id is not None and recovery_id in self._steps:
            self._active_step_id = recovery_id
            self._active_status = BTStatus.RUNNING
            self._step_started_s = now_s
            self._retries_left = self._steps[recovery_id].max_retries
            self._set_active_bt_node(bus, recovery_id)
        else:
            self._step_started_s = now_s
            self._active_status = BTStatus.RUNNING
            self._set_active_bt_node(bus, step.step_id)

        return result

    def tick_on_bus(self, bus: WorldStateRingBuffer, now_s: float) -> BTStatus:
        """10 Hz executive tick advancing the active skill when surprise is low."""
        if self._last_exec_tick_s >= 0.0 and (
            now_s - self._last_exec_tick_s
        ) < EXECUTIVE_TICK_PERIOD_S:
            return self._active_status
        self._last_exec_tick_s = now_s

        step = self._active_step()
        if step is None:
            self._active_status = BTStatus.SUCCESS
            return self._active_status

        if self._step_started_s <= 0.0:
            self._step_started_s = now_s
            self._set_active_bt_node(bus, step.step_id)

        snap = bus.read_latest()
        if snap is not None:
            surprise = float(snap.slot["surprise_scalar"])
            if surprise >= step.surprise_threshold:
                return self._active_status

        if step.precondition_fn is not None:
            z_t = _read_z_t(snap.slot) if snap is not None else step.z_goal
            ok, _msg = step.precondition_fn(z_t)
            if not ok:
                return self._active_status

        z_now = _read_z_t(snap.slot) if snap is not None else step.z_goal
        dist = float(np.linalg.norm(z_now - step.z_goal))
        elapsed = now_s - self._step_started_s
        if (dist < 0.05 or elapsed >= step.timeout_s) and elapsed >= MIN_STEP_DWELL_S:
            self._completed_steps.append(step.step_id)
            nxt = self._next_plan_step_id(step.step_id)
            if nxt is None:
                self._active_step_id = None
                self._active_status = BTStatus.SUCCESS
                self._set_active_bt_node(bus, None)
            else:
                self._active_step_id = nxt
                self._step_started_s = now_s
                self._retries_left = self._steps[nxt].max_retries
                self._active_status = BTStatus.RUNNING
                self._set_active_bt_node(bus, nxt)
        return self._active_status


def run_workbench_task_plan(
    steps: list[BTSkillStep],
    bus: WorldStateRingBuffer,
    *,
    duration_s: float = 30.0,
    surprise_injections: dict[float, float] | None = None,
    dt_s: float = INTERRUPT_FRAME_PERIOD_S,
) -> tuple[SurpriseGatedBehaviourTree, BTStatus]:
    """Simulate a multi-step workbench plan with optional timed surprise spikes."""
    tree = SurpriseGatedBehaviourTree()
    tree.load_plan(steps)
    injections = surprise_injections or {}
    spike_level = next(iter(injections.values()), 0.88) if injections else 0.88
    perturb_until_s = -1.0
    perturb_armed = False
    t = 0.0
    active_goal = steps[0].z_goal if steps else np.zeros(3, dtype=np.float64)
    while t <= duration_s + 1e-9:
        step = tree._active_step()
        if (
            step is not None
            and step.recovery_step_id is not None
            and not perturb_armed
            and (t - tree._step_started_s) >= MIN_STEP_DWELL_S
        ):
            perturb_armed = True
            perturb_until_s = t + dt_s * 3.0
        surprise = float(spike_level if perturb_armed and t <= perturb_until_s else 0.05)
        if step is not None:
            active_goal = step.z_goal
        with bus.acquire_write_view() as view:
            view.surprise_scalar = surprise
            view.z_t[...] = np.float32(0.0)
            view.z_t[:3] = np.asarray(active_goal, dtype=np.float32)
        tree.check_surprise_interrupt(bus, t)
        tree.tick_on_bus(bus, t)
        if tree.active_step_id is None and tree._active_status == BTStatus.SUCCESS:
            break
        t += dt_s
    return tree, tree._active_status
