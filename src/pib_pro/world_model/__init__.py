"""HJEPA world model latent state and protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pib_pro.common import JointStateSnapshot, Vector3D
from pib_pro.world_model.hjepa_engine import HJEPAConfig, HJEPAWorldModelEngine

__all__ = [
    "HJEPAConfig",
    "HJEPAWorldModelEngine",
    "HJEPAWorldModelProtocol",
    "LatentStateVector",
]


@dataclass(frozen=True, slots=True)
class LatentStateVector:
    """Fixed-size latent encoding of scene + proprioception."""

    dim: int
    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.values.shape != (self.dim,):
            raise ValueError("LatentStateVector: values shape must match dim")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("LatentStateVector: values must be finite")


class HJEPAWorldModelProtocol(Protocol):
    """Hierarchical JEPA-style predictive world model."""

    def encode(
        self,
        rgb_embedding: NDArray[np.float32],
        proprio: JointStateSnapshot,
    ) -> LatentStateVector: ...

    def predict_self(
        self,
        state: LatentStateVector,
        action_delta: Vector3D,
        horizon_s: float,
    ) -> LatentStateVector: ...

    def predict_human(
        self,
        state: LatentStateVector,
        horizon_s: float,
    ) -> LatentStateVector: ...

    def compute_surprise(
        self,
        predicted: LatentStateVector,
        observed: LatentStateVector,
    ) -> float: ...
