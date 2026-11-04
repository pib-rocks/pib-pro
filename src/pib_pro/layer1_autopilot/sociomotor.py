"""30 Hz sociomotor autopilot: deictic gaze, speaker tracking, backchannels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer
from pib_pro.world_state_bus.schema import WorldStateSlotView

SOCIO_HZ: float = 30.0
SOCIO_DT_S: float = 1.0 / SOCIO_HZ
JOINT_ATTENTION_PROB_MIN: float = 0.45
BACKCHANNEL_NOD_AMP_RAD: float = 0.12
CURIOSITY_TILT_AMP_RAD: float = 0.08
SURPRISE_TILT_THRESHOLD: float = 0.55


@dataclass(frozen=True, slots=True)
class NeckEyeTarget:
    yaw_rad: float
    pitch_rad: float
    eye_yaw_rad: float
    eye_pitch_rad: float
    mode: str
    onset_latency_ms: float
    yaw_velocity_rad_s: float = 0.0
    pitch_velocity_rad_s: float = 0.0
    eye_yaw_velocity_rad_s: float = 0.0
    eye_pitch_velocity_rad_s: float = 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


@dataclass(slots=True)
class JerkLimitedPCHIPFilter:
    """Monotone smooth trajectory with bounded jerk/velocity for head yaw & pitch."""

    max_velocity_rad_s: float = 1.8
    max_jerk_rad_s3: float = 12.0
    max_accel_rad_s2: float = 6.0
    _pos: float = field(default=0.0, init=False)
    _vel: float = field(default=0.0, init=False)
    _target: float = field(default=0.0, init=False)

    def reset(self, position_rad: float = 0.0) -> None:
        self._pos = position_rad
        self._vel = 0.0
        self._target = position_rad

    def set_target(self, target_rad: float) -> None:
        self._target = target_rad

    @property
    def position(self) -> float:
        return self._pos

    @property
    def velocity(self) -> float:
        return self._vel

    def step(self, dt_s: float) -> tuple[float, float]:
        if dt_s <= 0.0:
            return self._pos, self._vel
        error = self._target - self._pos
        inv_dt = 1.0 / max(dt_s, 1e-4)
        desired_accel = _clamp(
            error * inv_dt,
            -self.max_accel_rad_s2,
            self.max_accel_rad_s2,
        )
        jerk = _clamp(
            (desired_accel - self._vel * inv_dt) * inv_dt,
            -self.max_jerk_rad_s3,
            self.max_jerk_rad_s3,
        )
        accel = _clamp(
            jerk * dt_s + self._vel * inv_dt,
            -self.max_accel_rad_s2,
            self.max_accel_rad_s2,
        )
        self._vel = _clamp(
            self._vel + accel * dt_s,
            -self.max_velocity_rad_s,
            self.max_velocity_rad_s,
        )
        if (self._target - self._pos) * self._vel < 0.0 and abs(self._vel) < 0.02:
            self._vel = 0.0
        self._pos += self._vel * dt_s
        if abs(self._target - self._pos) < 1e-4 and abs(self._vel) < 1e-3:
            self._pos = self._target
            self._vel = 0.0
        return self._pos, self._vel


def _look_at_object(
    obj_xyz: np.ndarray,
    neck_xyz: np.ndarray,
) -> tuple[float, float]:
    x = float(obj_xyz[0]) - float(neck_xyz[0])
    y = float(obj_xyz[1]) - float(neck_xyz[1])
    z = float(obj_xyz[2]) - float(neck_xyz[2])
    yaw = math.atan2(y, x)
    horiz = math.hypot(x, y)
    pitch = math.atan2(z, horiz)
    return yaw, pitch


@dataclass(slots=True)
class SociomotorAutopilot:
    """Layer-1a neck/eye autopilot driven by the world-state bus at 30 Hz."""

    deictic_dwell_s: float = 0.8
    yaw_filter: JerkLimitedPCHIPFilter = field(default_factory=JerkLimitedPCHIPFilter)
    pitch_filter: JerkLimitedPCHIPFilter = field(default_factory=JerkLimitedPCHIPFilter)
    eye_yaw_filter: JerkLimitedPCHIPFilter = field(default_factory=JerkLimitedPCHIPFilter)
    eye_pitch_filter: JerkLimitedPCHIPFilter = field(default_factory=JerkLimitedPCHIPFilter)
    _mode: str = field(default="idle", init=False)
    _deictic_timer_s: float = field(default=0.0, init=False)
    _deictic_object_id: int = field(default=-1, init=False)
    _nod_phase: float = field(default=0.0, init=False)
    _prev_vad: bool = field(default=False, init=False)
    _last_onset_ms: float = field(default=0.0, init=False)

    def blend_with_manipulation(
        self,
        sociomotor: NeckEyeTarget,
        manipulation_yaw_pitch: tuple[float, float],
        alpha: float,
    ) -> NeckEyeTarget:
        a = _clamp(alpha, 0.0, 1.0)
        myaw, mpitch = manipulation_yaw_pitch
        return NeckEyeTarget(
            yaw_rad=sociomotor.yaw_rad * (1.0 - a) + myaw * a,
            pitch_rad=sociomotor.pitch_rad * (1.0 - a) + mpitch * a,
            eye_yaw_rad=sociomotor.eye_yaw_rad * (1.0 - a) + myaw * a,
            eye_pitch_rad=sociomotor.eye_pitch_rad * (1.0 - a) + mpitch * a,
            mode=sociomotor.mode,
            onset_latency_ms=sociomotor.onset_latency_ms,
            yaw_velocity_rad_s=sociomotor.yaw_velocity_rad_s * (1.0 - a),
            pitch_velocity_rad_s=sociomotor.pitch_velocity_rad_s * (1.0 - a),
            eye_yaw_velocity_rad_s=sociomotor.eye_yaw_velocity_rad_s * (1.0 - a),
            eye_pitch_velocity_rad_s=sociomotor.eye_pitch_velocity_rad_s * (1.0 - a),
        )

    def _apply_backchannel(
        self,
        pitch_rad: float,
        pitch_vel: float,
        *,
        surprise: float,
        vad_active: bool,
        dt_s: float,
    ) -> tuple[float, float, str]:
        mode = self._mode
        if surprise >= SURPRISE_TILT_THRESHOLD:
            mode = "curiosity_tilt"
            pitch_rad += CURIOSITY_TILT_AMP_RAD * math.sin(surprise * math.pi)
        if vad_active:
            if not self._prev_vad:
                self._nod_phase = 0.0
            self._nod_phase += dt_s * 2.0 * math.pi
            if self._nod_phase <= 2.0 * math.pi:
                nod = BACKCHANNEL_NOD_AMP_RAD * math.sin(self._nod_phase)
                pitch_rad += nod
                pitch_vel += BACKCHANNEL_NOD_AMP_RAD * 2.0 * math.pi * math.cos(self._nod_phase)
                if mode not in ("speaker_face", "deictic_object"):
                    mode = "backchannel_nod"
        self._prev_vad = vad_active
        return pitch_rad, pitch_vel, mode

    def step_from_view(
        self,
        view: WorldStateSlotView,
        dt_s: float = SOCIO_DT_S,
    ) -> NeckEyeTarget:
        slot = view.raw
        neck = slot["head_position_xyz"]
        if float(neck[0]) == 0.0 and float(neck[1]) == 0.0 and float(neck[2]) == 0.0:
            neck = np.array([0.0, 0.0, 1.35], dtype=np.float32)

        target_yaw = self.yaw_filter._target  # noqa: SLF001
        target_pitch = self.pitch_filter._target  # noqa: SLF001
        mode = "idle"
        onset_ms = 0.0

        obj_id = int(slot["joint_attention_object_id"])
        deictic_active = False
        if obj_id >= 0:
            objs = slot["objects"]
            if 0 <= obj_id < len(objs):
                prob = float(objs[obj_id]["joint_attention_prob"])
                if prob >= JOINT_ATTENTION_PROB_MIN:
                    deictic_active = True
                    mode = "deictic_object"
                    if obj_id != self._deictic_object_id:
                        self._deictic_timer_s = 0.0
                        self._deictic_object_id = obj_id
                        onset_ms = dt_s * 1000.0
                        self._last_onset_ms = onset_ms
                    obj_xyz = objs[obj_id]["position_xyz"]
                    target_yaw, target_pitch = _look_at_object(obj_xyz, neck)
                    self.yaw_filter.set_target(target_yaw)
                    self.pitch_filter.set_target(target_pitch)
                    self.eye_yaw_filter.set_target(target_yaw)
                    self.eye_pitch_filter.set_target(target_pitch)
                    self._deictic_timer_s += dt_s
                    if self._deictic_timer_s >= self.deictic_dwell_s:
                        self._deictic_object_id = -1

        if not deictic_active and (bool(slot["vad_active"]) or bool(slot["human_present"])):
            mode = "speaker_face"
            head = slot["head_position_xyz"]
            hx, hy, hz = float(head[0]), float(head[1]), float(head[2])
            if math.hypot(hx, hy) > 1e-3:
                target_yaw = math.atan2(hy, hx)
                target_pitch = math.atan2(hz - float(neck[2]), math.hypot(hx, hy))
            else:
                target_yaw = float(slot["speaker_doa_azimuth_rad"])
                target_pitch = 0.0
            if bool(slot["vad_active"]) and onset_ms == 0.0:
                onset_ms = dt_s * 1000.0
                self._last_onset_ms = onset_ms
            self.yaw_filter.set_target(target_yaw)
            self.pitch_filter.set_target(target_pitch)
            self.eye_yaw_filter.set_target(target_yaw)
            self.eye_pitch_filter.set_target(target_pitch)

        yaw, yaw_vel = self.yaw_filter.step(dt_s)
        pitch, pitch_vel = self.pitch_filter.step(dt_s)
        eye_yaw, eye_yaw_vel = self.eye_yaw_filter.step(dt_s)
        eye_pitch, eye_pitch_vel = self.eye_pitch_filter.step(dt_s)

        pitch, pitch_vel, bc_mode = self._apply_backchannel(
            pitch,
            pitch_vel,
            surprise=float(slot["surprise_scalar"]),
            vad_active=bool(slot["vad_active"]),
            dt_s=dt_s,
        )
        if bc_mode == "curiosity_tilt" and mode in ("speaker_face", "idle"):
            mode = bc_mode

        self._mode = mode
        return NeckEyeTarget(
            yaw_rad=yaw,
            pitch_rad=pitch,
            eye_yaw_rad=eye_yaw,
            eye_pitch_rad=eye_pitch,
            mode=mode,
            onset_latency_ms=onset_ms if onset_ms > 0.0 else self._last_onset_ms,
            yaw_velocity_rad_s=yaw_vel,
            pitch_velocity_rad_s=pitch_vel,
            eye_yaw_velocity_rad_s=eye_yaw_vel,
            eye_pitch_velocity_rad_s=eye_pitch_vel,
        )

    def step(
        self,
        bus: WorldStateRingBuffer,
        dt_s: float = SOCIO_DT_S,
    ) -> NeckEyeTarget:
        latest = bus.read_latest()
        if latest is None:
            return NeckEyeTarget(0.0, 0.0, 0.0, 0.0, "idle", 0.0)
        return self.step_from_view(latest.view, dt_s=dt_s)
