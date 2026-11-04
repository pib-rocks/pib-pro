"""REP-103 coordinate conventions and shared geometric types for pib.Pro."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

import numpy as np
from numpy.typing import NDArray

# REP-103: X forward, Y left, Z up; angles in radians; right-handed frame.
AXIS_X_FORWARD: Final[str] = "x"
AXIS_Y_LEFT: Final[str] = "y"
AXIS_Z_UP: Final[str] = "z"

FRAME_BASE_LINK: Final[str] = "base_link"
FRAME_CAMERA_OAK4_OPTICAL: Final[str] = "camera_oak4_optical_frame"
FRAME_NECK_LINK: Final[str] = "neck_link"
FRAME_HAND_LEFT: Final[str] = "hand_left_link"
FRAME_HAND_RIGHT: Final[str] = "hand_right_link"

STANDARD_FRAMES: Final[frozenset[str]] = frozenset(
    {
        FRAME_BASE_LINK,
        FRAME_CAMERA_OAK4_OPTICAL,
        FRAME_NECK_LINK,
        FRAME_HAND_LEFT,
        FRAME_HAND_RIGHT,
    }
)

QUAT_NORM_TOLERANCE: Final[float] = 1e-6
FINITE_EPS: Final[float] = 1e-12


def _require_finite(name: str, *values: float) -> None:
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"{name}: expected finite value, got {v!r}")


@dataclass(frozen=True, slots=True)
class Vector3D:
    """Position or displacement in meters (REP-103)."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        _require_finite("Vector3D", self.x, self.y, self.z)

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other: Vector3D) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> Vector3D:
        n = self.norm()
        if n < FINITE_EPS:
            raise ValueError("Vector3D: cannot normalize zero vector")
        return Vector3D(self.x / n, self.y / n, self.z / n)


@dataclass(frozen=True, slots=True)
class Quaternion:
    """Orientation quaternion (x, y, z, w); must be unit length."""

    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        _require_finite("Quaternion", self.x, self.y, self.z, self.w)
        n = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if abs(n - 1.0) > QUAT_NORM_TOLERANCE:
            raise ValueError(f"Quaternion: must be unit norm, got norm={n}")

    @classmethod
    def identity(cls) -> Quaternion:
        return cls(0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Pose3D:
    """Rigid transform: position in parent frame + orientation."""

    frame_id: str
    position: Vector3D
    orientation: Quaternion

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("Pose3D: frame_id must be non-empty")


@dataclass(frozen=True, slots=True)
class Ray3D:
    """Ray for deictic / pointing resolution: origin + unit direction."""

    origin: Vector3D
    direction: Vector3D

    def __post_init__(self) -> None:
        d = self.direction
        n = d.norm()
        if abs(n - 1.0) > QUAT_NORM_TOLERANCE:
            object.__setattr__(self, "direction", d.normalized())


@dataclass(frozen=True, slots=True)
class JointStateSnapshot:
    """Named joint positions (rad) and velocities (rad/s)."""

    names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.names) == len(self.positions_rad) == len(self.velocities_rad_s)):
            raise ValueError("JointStateSnapshot: names, positions, velocities length mismatch")
        for p in self.positions_rad:
            _require_finite("JointStateSnapshot.position", p)
        for v in self.velocities_rad_s:
            _require_finite("JointStateSnapshot.velocity", v)


class FloorState(Enum):
    """High-level floor / support contact classification."""

    UNKNOWN = auto()
    STABLE = auto()
    UNSTABLE = auto()
    LOST = auto()


class ProactivityTier(Enum):
    """Communicative proactivity bounds (T0 silent … T3 forbidden)."""

    T0_SILENT = auto()
    T1_MINIMAL = auto()
    T2_NORMAL = auto()
    T3_FORBIDDEN = auto()


@dataclass(frozen=True, slots=True)
class AABB3D:
    """Axis-aligned bounding box in meters."""

    min_corner: Vector3D
    max_corner: Vector3D

    def __post_init__(self) -> None:
        if (
            self.min_corner.x > self.max_corner.x
            or self.min_corner.y > self.max_corner.y
            or self.min_corner.z > self.max_corner.z
        ):
            raise ValueError("AABB3D: min_corner must be <= max_corner on each axis")


def ray_sphere_intersection(
    ray: Ray3D,
    center: Vector3D,
    radius: float,
) -> float | None:
    """Return smallest non-negative ray parameter t along ray, or None if no hit."""
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("ray_sphere_intersection: radius must be positive and finite")
    oc = Vector3D(
        ray.origin.x - center.x,
        ray.origin.y - center.y,
        ray.origin.z - center.z,
    )
    a = ray.direction.dot(ray.direction)
    b = 2.0 * oc.dot(ray.direction)
    c = oc.dot(oc) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    for t in (t0, t1):
        if t >= 0:
            return t
    return None


def ray_aabb_intersection(ray: Ray3D, box: AABB3D) -> float | None:
    """Slab method: smallest non-negative t entering the AABB, or None."""
    origin = ray.origin.as_array()
    direction = ray.direction.as_array()
    t_min = 0.0
    t_max = float("inf")
    for i in range(3):
        o = origin[i]
        d = direction[i]
        lo = (box.min_corner.x, box.min_corner.y, box.min_corner.z)[i]
        hi = (box.max_corner.x, box.max_corner.y, box.max_corner.z)[i]
        if abs(d) < FINITE_EPS:
            if o < lo or o > hi:
                return None
            continue
        t1 = (lo - o) / d
        t2 = (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    if t_max < 0:
        return None
    return t_min if t_min >= 0 else t_max
