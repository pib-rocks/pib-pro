"""Anticipatory proactivity gate and four-tier embodied memory."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pib_pro.common.rep103 import ProactivityTier, Vector3D
from pib_pro.layer1_autopilot.procedural_memory import ProceduralMemoryRecorder
from pib_pro.world_state_bus.schema import MAX_OBJECTS

if TYPE_CHECKING:
    from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer

WORKSPACE_BOUNDS_MIN = Vector3D(0.25, -0.45, 0.55)
WORKSPACE_BOUNDS_MAX = Vector3D(0.75, 0.45, 1.05)
HANDOVER_HORIZON_S: float = 1.2
PROACTIVITY_ONSET_BUDGET_MS: float = 800.0
UTILITY_MINUS_INTERRUPTION_THRESHOLD: float = 0.15


@dataclass(frozen=True, slots=True)
class ProactivityDecision:
    proactivity_tier: ProactivityTier
    silent_physical_action: str | None
    verbal_utterance: str | None
    onset_latency_ms: float
    joint_attention_with_robot: bool


@dataclass(frozen=True, slots=True)
class ObjectLastSeen:
    object_id: int
    label: str
    position_xyz: tuple[float, float, float]
    seconds_ago: float
    episode_id: str


@dataclass(frozen=True, slots=True)
class SocialProfile:
    person_id: int
    consent_granted: bool
    preferred_handover_side: str
    preferred_handover_height_z_m: float
    speaking_pace: str
    proactivity_preference: ProactivityTier


@dataclass(slots=True)
class WorkingWorldBuffer:
    """Tier 1: rolling 30 s window over the world-state bus."""

    bus: WorldStateRingBuffer
    window_s: float = 30.0

    def latest_timestamp_ns(self) -> int | None:
        snap = self.bus.read_latest()
        if snap is None:
            return None
        return int(snap.slot["timestamp_ns"])

    def read_window_slots(self) -> list[np.ndarray]:
        duration_ns = int(self.window_s * 1e9)
        return [r.slot for r in self.bus.read_window(duration_ns)]


@dataclass(slots=True)
class EpisodicSpatialGraph:
    """Tier 2: persistent spatial-interaction episodes (SQLite-backed)."""

    db_path: Path
    _conn: sqlite3.Connection | None = field(default=None, init=False)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_objects (
                    episode_id TEXT NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    object_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    person_id INTEGER NOT NULL,
                    summary_text TEXT NOT NULL,
                    latent_z BLOB NOT NULL
                )
                """
            )
            self._conn.commit()
        return self._conn

    def record_observation(
        self,
        *,
        episode_id: str,
        timestamp_ns: int,
        object_id: int,
        label: str,
        position_xyz: tuple[float, float, float],
        person_id: int,
        summary_text: str,
        latent_z_t: NDArray[np.float32],
    ) -> None:
        conn = self._connection()
        conn.execute(
            """
            INSERT INTO episodic_objects
            (episode_id, timestamp_ns, object_id, label, x, y, z, person_id, summary_text, latent_z)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                timestamp_ns,
                object_id,
                label,
                position_xyz[0],
                position_xyz[1],
                position_xyz[2],
                person_id,
                summary_text,
                latent_z_t.astype(np.float32).tobytes(),
            ),
        )
        conn.commit()

    def query_object_last_seen(
        self,
        label: str,
        *,
        now_ns: int | None = None,
    ) -> ObjectLastSeen | None:
        conn = self._connection()
        row = conn.execute(
            """
            SELECT episode_id, timestamp_ns, object_id, label, x, y, z
            FROM episodic_objects
            WHERE lower(label) = lower(?)
            ORDER BY timestamp_ns DESC
            LIMIT 1
            """,
            (label,),
        ).fetchone()
        if row is None:
            return None
        ts = int(row[1])
        ref = now_ns if now_ns is not None else time.time_ns()
        seconds_ago = max(0.0, (ref - ts) / 1e9)
        return ObjectLastSeen(
            object_id=int(row[2]),
            label=str(row[3]),
            position_xyz=(float(row[4]), float(row[5]), float(row[6])),
            seconds_ago=seconds_ago,
            episode_id=str(row[0]),
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


@dataclass(slots=True)
class SemanticSocialProfileStore:
    """Tier 3: consent-gated per-person social preferences (JSON file)."""

    path: Path
    _profiles: dict[int, SocialProfile] = field(default_factory=dict, init=False)

    def load(self) -> None:
        if not self.path.exists():
            self._profiles = {}
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._profiles = {
            int(k): SocialProfile(
                person_id=int(k),
                consent_granted=bool(v["consent_granted"]),
                preferred_handover_side=str(v["preferred_handover_side"]),
                preferred_handover_height_z_m=float(v["preferred_handover_height_z_m"]),
                speaking_pace=str(v["speaking_pace"]),
                proactivity_preference=ProactivityTier[v["proactivity_preference"]],
            )
            for k, v in raw.items()
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            str(p.person_id): {
                "consent_granted": p.consent_granted,
                "preferred_handover_side": p.preferred_handover_side,
                "preferred_handover_height_z_m": p.preferred_handover_height_z_m,
                "speaking_pace": p.speaking_pace,
                "proactivity_preference": p.proactivity_preference.name,
            }
            for p in self._profiles.values()
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def upsert(self, profile: SocialProfile) -> None:
        if not profile.consent_granted:
            msg = "consent_granted must be True before persisting social profile"
            raise ValueError(msg)
        self._profiles[profile.person_id] = profile

    def get(self, person_id: int) -> SocialProfile | None:
        return self._profiles.get(person_id)


@dataclass(slots=True)
class ProceduralMemoryLink:
    """Tier 4: bridge to PR-2245 procedural memory recorder."""

    recorder: ProceduralMemoryRecorder = field(default_factory=ProceduralMemoryRecorder)

    @property
    def episode_count(self) -> int:
        return len(self.recorder.episodes)


@dataclass(slots=True)
class FourTierEmbodiedMemory:
    tier1: WorkingWorldBuffer
    tier2: EpisodicSpatialGraph
    tier3: SemanticSocialProfileStore
    tier4: ProceduralMemoryLink = field(default_factory=ProceduralMemoryLink)

    def query_object_last_seen(
        self,
        label: str,
        *,
        now_ns: int | None = None,
    ) -> ObjectLastSeen | None:
        return self.tier2.query_object_last_seen(label, now_ns=now_ns)


def _hand_entering_workspace(slot: np.ndarray) -> tuple[bool, float | None]:
    for hand_xyz, vel_xyz in (
        (slot["right_hand_xyz"], slot["right_hand_vel_xyz"]),
        (slot["left_hand_xyz"], slot["left_hand_vel_xyz"]),
    ):
        pos = Vector3D(float(hand_xyz[0]), float(hand_xyz[1]), float(hand_xyz[2]))
        vel = Vector3D(float(vel_xyz[0]), float(vel_xyz[1]), float(vel_xyz[2]))
        t = 0.0
        dt = 1.0 / 120.0
        while t <= HANDOVER_HORIZON_S + 1e-9:
            px = pos.x + vel.x * t
            py = pos.y + vel.y * t
            pz = pos.z + vel.z * t
            if (
                WORKSPACE_BOUNDS_MIN.x <= px <= WORKSPACE_BOUNDS_MAX.x
                and WORKSPACE_BOUNDS_MIN.y <= py <= WORKSPACE_BOUNDS_MAX.y
                and WORKSPACE_BOUNDS_MIN.z <= pz <= WORKSPACE_BOUNDS_MAX.z
            ):
                return True, t
            t += dt
    return False, None


def _joint_attention_with_robot(slot: np.ndarray) -> bool:
    if not bool(slot["human_present"]):
        return False
    ja_id = int(slot["joint_attention_object_id"])
    if ja_id < 0:
        return False
    objs = slot["objects"]
    if ja_id >= MAX_OBJECTS:
        return False
    prob = float(objs[ja_id]["joint_attention_prob"])
    gaze_dir = slot["gaze_ray_dir_xyz"]
    gaze_norm = float(np.linalg.norm(gaze_dir))
    return prob > 0.35 and gaze_norm > 0.2


@dataclass(slots=True)
class AnticipatoryProactivityGate:
    """Expected-utility gate for silent physical preparation vs verbal interruption."""

    unsolicited_verbal_count: int = 0
    _last_onset_ms: float = 0.0

    def evaluate(
        self,
        bus: WorldStateRingBuffer,
        *,
        stamp_monotonic_s: float,
        simulated_onset_ms: float | None = None,
    ) -> ProactivityDecision:
        latest = bus.read_latest()
        if latest is None:
            return ProactivityDecision(
                proactivity_tier=ProactivityTier.T0_SILENT,
                silent_physical_action=None,
                verbal_utterance=None,
                onset_latency_ms=0.0,
                joint_attention_with_robot=False,
            )
        slot = latest.slot
        joint = _joint_attention_with_robot(slot)
        expected_utility = 0.4 if joint else 0.0
        interruption_cost = 0.35 if not joint else 0.05
        net = expected_utility - interruption_cost
        if not joint or net <= UTILITY_MINUS_INTERRUPTION_THRESHOLD:
            return ProactivityDecision(
                proactivity_tier=ProactivityTier.T0_SILENT,
                silent_physical_action=None,
                verbal_utterance=None,
                onset_latency_ms=0.0,
                joint_attention_with_robot=joint,
            )
        entering, t_entry = _hand_entering_workspace(slot)
        if not entering or t_entry is None or t_entry > HANDOVER_HORIZON_S:
            return ProactivityDecision(
                proactivity_tier=ProactivityTier.T1_MINIMAL,
                silent_physical_action=None,
                verbal_utterance=None,
                onset_latency_ms=0.0,
                joint_attention_with_robot=True,
            )
        onset_ms = simulated_onset_ms if simulated_onset_ms is not None else 650.0
        self._last_onset_ms = onset_ms
        return ProactivityDecision(
            proactivity_tier=ProactivityTier.T1_MINIMAL,
            silent_physical_action="open_hand_receive_pose",
            verbal_utterance=None,
            onset_latency_ms=onset_ms,
            joint_attention_with_robot=True,
        )

    def register_verbal_attempt(
        self,
        tier: ProactivityTier,
        *,
        joint_attention_with_robot: bool,
    ) -> None:
        if tier != ProactivityTier.T0_SILENT and not joint_attention_with_robot:
            self.unsolicited_verbal_count += 1
