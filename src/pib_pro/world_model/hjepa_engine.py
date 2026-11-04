"""Deterministic HJEPA-style latent dynamics engine for the world-state bus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pib_pro.world_state_bus.schema import (
    LATENT_DIM,
    MAX_JOINTS,
    MAX_OBJECTS,
    SURPRISE_REGION_COUNT,
)

if TYPE_CHECKING:
    from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer

SCENE_FEATURE_DIM: int = MAX_OBJECTS * 3 + MAX_JOINTS * 2


@dataclass(frozen=True, slots=True)
class HJEPAConfig:
    latent_dim: int = LATENT_DIM
    surprise_threshold: float = 0.35
    num_regions: int = SURPRISE_REGION_COUNT
    dt_sec: float = 1.0 / 30.0


def _default_encoder_matrix(latent_dim: int) -> NDArray[np.float32]:
    w = np.zeros((latent_dim, SCENE_FEATURE_DIM), dtype=np.float32)
    for i in range(min(3, latent_dim)):
        w[i, i] = 9.0
    for i in range(3, min(latent_dim, SCENE_FEATURE_DIM)):
        w[i, i] = 0.05
    return w


def _default_dynamics_matrix(latent_dim: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(43)
    a = np.eye(latent_dim, dtype=np.float32) + rng.standard_normal((latent_dim, latent_dim)) * 0.01
    return a.astype(np.float32)


class HJEPAWorldModelEngine:
    """Fast NumPy latent encoder / predictor with optional pluggable weight tensors."""

    def __init__(
        self,
        config: HJEPAConfig | None = None,
        *,
        encoder_weights: NDArray[np.float32] | None = None,
        dynamics_weights: NDArray[np.float32] | None = None,
        action_weights: NDArray[np.float32] | None = None,
        human_coupling_weights: NDArray[np.float32] | None = None,
    ) -> None:
        self.config = config or HJEPAConfig()
        dim = self.config.latent_dim
        if dim != LATENT_DIM:
            raise ValueError(f"HJEPAWorldModelEngine: latent_dim must be {LATENT_DIM}")
        self._encoder = (
            encoder_weights if encoder_weights is not None else _default_encoder_matrix(dim)
        )
        self._dynamics = (
            dynamics_weights if dynamics_weights is not None else _default_dynamics_matrix(dim)
        )
        self._action_w = (
            action_weights
            if action_weights is not None
            else np.zeros((dim, MAX_JOINTS), dtype=np.float32)
        )
        self._human_w = (
            human_coupling_weights
            if human_coupling_weights is not None
            else np.zeros((dim, 6), dtype=np.float32)
        )
        self._prev_z_hat_self: NDArray[np.float32] | None = None
        self._rng = np.random.default_rng(0)

    def load_tensorrt_engine_weights(
        self,
        encoder: NDArray[np.float32],
        dynamics: NDArray[np.float32],
        action: NDArray[np.float32] | None = None,
        human: NDArray[np.float32] | None = None,
    ) -> None:
        """Replace weight tensors (e.g. deserialized TensorRT export on Thor)."""
        self._encoder = np.asarray(encoder, dtype=np.float32)
        self._dynamics = np.asarray(dynamics, dtype=np.float32)
        if action is not None:
            self._action_w = np.asarray(action, dtype=np.float32)
        if human is not None:
            self._human_w = np.asarray(human, dtype=np.float32)

    def nudge_action_projection(
        self,
        latent_error: NDArray[np.float32],
        action: NDArray[np.float32],
        *,
        learning_rate: float = 0.05,
    ) -> None:
        """Rank-one residual update for offline consolidation."""
        act = np.zeros(MAX_JOINTS, dtype=np.float32)
        src = np.asarray(action, dtype=np.float32).reshape(-1)
        n = min(src.size, MAX_JOINTS)
        act[:n] = src[:n]
        err = np.asarray(latent_error, dtype=np.float32).reshape(-1)
        self._action_w = (
            self._action_w + learning_rate * np.outer(err, act)
        ).astype(np.float32)

    def _normalize_latent(self, z: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.clip(z, -10.0, 10.0).astype(np.float32)

    def _scene_vector_from_arrays(
        self,
        rgb_features_or_scene: NDArray[np.float32],
        joint_positions_rad: NDArray[np.float32],
        joint_velocities_rad_s: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        scene = np.zeros(SCENE_FEATURE_DIM, dtype=np.float32)
        flat_scene = np.asarray(rgb_features_or_scene, dtype=np.float32).reshape(-1)
        n_scene = min(flat_scene.size, MAX_OBJECTS * 3)
        scene[:n_scene] = flat_scene[:n_scene]
        offset = MAX_OBJECTS * 3
        jp = np.asarray(joint_positions_rad, dtype=np.float32).reshape(-1)
        jv = np.asarray(joint_velocities_rad_s, dtype=np.float32).reshape(-1)
        n_j = min(jp.size, MAX_JOINTS)
        scene[offset : offset + n_j] = jp[:n_j]
        scene[offset + MAX_JOINTS : offset + MAX_JOINTS + n_j] = jv[:n_j]
        return scene

    def encode(
        self,
        rgb_features_or_scene: NDArray[np.float32],
        joint_positions_rad: NDArray[np.float32],
        joint_velocities_rad_s: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        scene = self._scene_vector_from_arrays(
            rgb_features_or_scene,
            joint_positions_rad,
            joint_velocities_rad_s,
        )
        z = self._encoder @ scene
        return self._normalize_latent(z.astype(np.float32))

    def predict_self(
        self,
        z_t: NDArray[np.float32],
        action_chunk: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """One-step or batched terminal latent for ``(K, H, num_joints)`` action chunks."""
        z = np.asarray(z_t, dtype=np.float32).reshape(-1)
        actions = np.asarray(action_chunk, dtype=np.float32)
        if actions.ndim == 1:
            act = actions.reshape(-1)[:MAX_JOINTS]
            if float(np.linalg.norm(act)) < 1e-9:
                return z.copy()
            delta = self._action_w @ act
            z_hat = self._dynamics @ z + delta
            return self._normalize_latent(np.tanh(z_hat).astype(np.float32))
        if actions.ndim == 3:
            k, horizon, n_j = actions.shape
            out = np.zeros((k, self.config.latent_dim), dtype=np.float32)
            for i in range(k):
                zi = z
                for h in range(horizon):
                    act = actions[i, h, : min(n_j, MAX_JOINTS)]
                    pad = np.zeros(MAX_JOINTS, dtype=np.float32)
                    pad[: act.size] = act
                    delta = self._action_w @ pad
                    zi = self._dynamics @ zi + delta
                    zi = np.tanh(zi).astype(np.float32)
                out[i] = self._normalize_latent(zi)
            return out
        msg = "action_chunk must be 1D (num_joints) or 3D (K, H, num_joints)"
        raise ValueError(msg)

    def predict_human(
        self,
        z_t: NDArray[np.float32],
        human_hand_pos_xyz: NDArray[np.float32],
        human_hand_vel_xyz: NDArray[np.float32],
        horizon_sec: float = 1.0,
    ) -> NDArray[np.float32]:
        z = np.asarray(z_t, dtype=np.float32).reshape(-1)
        pos = np.asarray(human_hand_pos_xyz, dtype=np.float32).reshape(3)
        vel = np.asarray(human_hand_vel_xyz, dtype=np.float32).reshape(3)
        human_feat = np.concatenate([pos, vel], dtype=np.float32)
        steps = max(1, int(round(horizon_sec / self.config.dt_sec)))
        zi = z
        coupling = self._human_w @ human_feat
        for _ in range(steps):
            zi = self._dynamics @ zi + coupling * self.config.dt_sec
            zi = np.tanh(zi).astype(np.float32)
        return self._normalize_latent(zi)

    def compute_surprise(
        self,
        z_t: NDArray[np.float32],
        z_hat_expected: NDArray[np.float32],
    ) -> tuple[float, NDArray[np.float32]]:
        z = np.asarray(z_t, dtype=np.float32).reshape(-1)
        zh = np.asarray(z_hat_expected, dtype=np.float32).reshape(-1)
        diff = z - zh
        surprise_scalar = float(np.linalg.norm(diff))
        region_size = self.config.latent_dim // self.config.num_regions
        regions = np.zeros(self.config.num_regions, dtype=np.float32)
        for r in range(self.config.num_regions):
            start = r * region_size
            end = start + region_size if r < self.config.num_regions - 1 else self.config.latent_dim
            regions[r] = np.float32(np.linalg.norm(diff[start:end]))
        return surprise_scalar, regions

    def step_on_bus(
        self,
        bus: WorldStateRingBuffer,
        action_chunk: NDArray[np.float32] | None = None,
    ) -> tuple[float, NDArray[np.float32]]:
        """Encode latest bus state, compute surprise vs previous ``z_hat_self``, write latents."""
        latest = bus.read_latest()
        with bus.acquire_write_view() as view:
            if latest is not None:
                view.fill_from(latest.slot)
            slot = view.raw
            positions = slot["objects"]["position_xyz"].reshape(-1)
            z_t = self.encode(
                positions,
                slot["joint_positions_rad"],
                slot["joint_velocities_rad_s"],
            )
            act = (
                action_chunk
                if action_chunk is not None
                else np.zeros(MAX_JOINTS, dtype=np.float32)
            )
            z_hat_self = self.predict_self(z_t, act)
            z_hat_human = self.predict_human(
                z_t,
                slot["right_hand_xyz"],
                slot["right_hand_vel_xyz"],
            )
            surprise = 0.0
            surprise_regions = np.zeros(self.config.num_regions, dtype=np.float32)
            if self._prev_z_hat_self is not None:
                surprise, surprise_regions = self.compute_surprise(z_t, self._prev_z_hat_self)
            slot["z_t"] = z_t
            slot["z_hat_self"] = z_hat_self
            slot["z_hat_human"] = z_hat_human
            slot["surprise_scalar"] = np.float32(surprise)
            slot["surprise_by_region"] = surprise_regions
            self._prev_z_hat_self = z_hat_self.copy()
        return surprise, surprise_regions
