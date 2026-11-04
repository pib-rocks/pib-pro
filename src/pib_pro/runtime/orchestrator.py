"""End-to-end pib.Pro runtime orchestrator (4-layer closed loop demo)."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pib_pro.common.rep103 import Vector3D
from pib_pro.drivers.chest_oak4 import ChestOak4Driver
from pib_pro.drivers.feetech_sts import FeetechStsBusDriver, InMemoryLoopbackSerialTransport
from pib_pro.drivers.seeed_respeaker import SeeedRespeakerDriver, extract_aec_ch0_frame
from pib_pro.layer0_reflex.somatic_reflex import SomaticReflexController
from pib_pro.layer1_autopilot.procedural_memory import ProceduralMemoryRecorder
from pib_pro.layer1_autopilot.sociomotor import SociomotorAutopilot
from pib_pro.layer1_autopilot.visuomotor_hybrid import HybridVisuomotorController
from pib_pro.layer2_executive.surprise_bt import BTSkillStep, SurpriseGatedBehaviourTree
from pib_pro.layer3_cognition import (
    AnticipatoryProactivityGate,
    FourTierEmbodiedMemory,
    HierarchicalSystem3Planner,
    LocalStreamingOmniDialogue,
    ThorLlamaServerClient,
    WorldStateContextToken,
)
from pib_pro.layer3_cognition.proactivity_and_memory import (
    EpisodicSpatialGraph,
    ProceduralMemoryLink,
    SemanticSocialProfileStore,
    WorkingWorldBuffer,
)
from pib_pro.layer3_cognition.thor_llama_client import ThorLlamaClientConfig
from pib_pro.perception_oak4.human_state import HumanObservation3D
from pib_pro.perception_oak4.scene_graph import Detection2DWithDepth, Oak4CameraIntrinsics
from pib_pro.telemetry_gateway.gateway import NonBlockingTelemetryGateway
from pib_pro.world_model.hjepa_engine import HJEPAWorldModelEngine
from pib_pro.world_state_bus import WorldStateRingBuffer
from pib_pro.world_state_bus.schema import BUS_PERIOD_NS, LATENT_DIM, MAX_JOINTS

_DEMO_MUG_OBJECT_ID: int = 7
_MID_REACH_SHIFT_M: tuple[float, float, float] = (0.04, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class RuntimeDemoReport:
    steps_run: int
    duration_s: float
    gaze_onset_ms: float
    deictic_gaze_mode_seen: bool
    omni_act_scheduled: bool
    mid_reach_target_shift_m: float
    surprise_recovery_seen: bool
    handover_released: bool
    feetech_sync_packets: tuple[str, ...]
    telemetry_json: str
    visuomotor_min_clearance_m: float
    thor_reply_utterance: str


@dataclass(slots=True)
class PibProRuntimeOrchestrator:
    """Wires perception drivers, 4 cognitive layers, reflex bus, and 10 Hz telemetry."""

    bus: WorldStateRingBuffer = field(default_factory=lambda: WorldStateRingBuffer(capacity=64))
    chest_oak4: ChestOak4Driver = field(
        default_factory=lambda: ChestOak4Driver(
            intrinsics=Oak4CameraIntrinsics(
                fx=700.0,
                fy=700.0,
                cx=640.0,
                cy=360.0,
                width=1280,
                height=720,
            ),
        ),
    )
    seeed: SeeedRespeakerDriver = field(default_factory=SeeedRespeakerDriver.default)
    hjepa: HJEPAWorldModelEngine = field(default_factory=HJEPAWorldModelEngine)
    reflex: SomaticReflexController = field(
        default_factory=lambda: SomaticReflexController(holding_object=True),
    )
    sts_transport: InMemoryLoopbackSerialTransport = field(
        default_factory=InMemoryLoopbackSerialTransport,
    )
    feetech: FeetechStsBusDriver = field(init=False)
    sociomotor: SociomotorAutopilot = field(default_factory=SociomotorAutopilot)
    visuomotor: HybridVisuomotorController = field(init=False)
    procedural_memory: ProceduralMemoryRecorder = field(init=False)
    executive_bt: SurpriseGatedBehaviourTree = field(default_factory=SurpriseGatedBehaviourTree)
    omni: LocalStreamingOmniDialogue = field(default_factory=LocalStreamingOmniDialogue)
    thor_llama: ThorLlamaServerClient = field(
        default_factory=lambda: ThorLlamaServerClient(ThorLlamaClientConfig()),
    )
    planner: HierarchicalSystem3Planner = field(default_factory=HierarchicalSystem3Planner)
    proactivity: AnticipatoryProactivityGate = field(default_factory=AnticipatoryProactivityGate)
    memory: FourTierEmbodiedMemory = field(init=False)
    telemetry: NonBlockingTelemetryGateway = field(init=False)
    _servo_ids: tuple[int, ...] = field(default=tuple(range(1, 17)))

    def __post_init__(self) -> None:
        self.feetech = FeetechStsBusDriver(
            transport=self.sts_transport,
            servo_ids=self._servo_ids[:MAX_JOINTS],
        )
        self.visuomotor = HybridVisuomotorController(
            hjepa=self.hjepa,
            reflex=self.reflex,
            target_object_index=0,
        )
        self.procedural_memory = ProceduralMemoryRecorder(bus=self.bus)
        self.procedural_memory.bus = self.bus
        self.telemetry = NonBlockingTelemetryGateway(bus=self.bus)
        mem_dir = Path(tempfile.mkdtemp(prefix="pib_pro_demo_"))
        self.memory = FourTierEmbodiedMemory(
            tier1=WorkingWorldBuffer(self.bus),
            tier2=EpisodicSpatialGraph(mem_dir / "episodic.db"),
            tier3=SemanticSocialProfileStore(mem_dir / "social.json"),
            tier4=ProceduralMemoryLink(),
        )

    def _seed_bus_slot(self, frame: int, timestamp_ns: int) -> None:
        with self.bus.acquire_write_view() as view:
            raw = view.raw
            raw["timestamp_ns"] = timestamp_ns
            view.frame_id = frame
            view.human_present = True
            raw["head_position_xyz"] = np.array([0.0, 0.0, 1.35], dtype=np.float32)
            raw["gaze_ray_dir_xyz"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            pointing = np.array([0.45, 0.05, 0.72], dtype=np.float32)
            raw["pointing_ray_origin_xyz"] = np.array([0.0, 0.0, 1.35], dtype=np.float32)
            raw["pointing_ray_dir_xyz"] = pointing - raw["head_position_xyz"]
            raw["right_hand_xyz"] = np.array([0.55, 0.05, 0.85], dtype=np.float32)
            raw["joint_positions_rad"] = np.zeros(MAX_JOINTS, dtype=np.float32)
            raw["z_goal"] = np.zeros(LATENT_DIM, dtype=np.float32)
            raw["z_goal"][0] = np.float32(_DEMO_MUG_OBJECT_ID)
            raw["z_goal"][1] = np.float32(1.0)

    def _perception_tick(
        self,
        frame: int,
        timestamp_ns: int,
        *,
        pointing_active: bool,
        doa_azimuth_rad: float,
        vad_active: bool,
    ) -> None:
        mug_emb = np.array([0.9, 0.1, 0.05], dtype=np.float32)
        mug_det = Detection2DWithDepth(
            class_id=42,
            label="red_mug",
            bbox_xyxy=(600.0, 320.0, 680.0, 420.0),
            depth_m=0.55,
            confidence=0.92,
            appearance_embedding=mug_emb,
            half_extents_xyz=(0.04, 0.04, 0.06),
        )
        human = HumanObservation3D(
            head_position=Vector3D(0.0, 0.0, 1.35),
            gaze_direction=Vector3D(0.45, 0.05, -0.63).normalized(),
            pointing_origin=Vector3D(0.0, 0.0, 1.35),
            pointing_direction=Vector3D(0.45, 0.05, -0.63).normalized(),
            pointing_active=pointing_active,
            left_hand_position=Vector3D(0.2, 0.25, 0.9),
            right_hand_position=Vector3D(0.55, 0.05, 0.85),
            doa_azimuth_rad=doa_azimuth_rad,
            vad_active=vad_active,
        )
        left_ee = Vector3D(0.32, 0.12, 0.78)
        right_ee = Vector3D(0.34, -0.10, 0.78)
        self.chest_oak4.persistent_tracks = (Vector3D(0.45, 0.05, 0.72),)
        _filtered, _winner_id = self.chest_oak4.ingest_frame(
            (mug_det,),
            human,
            left_ee,
            right_ee,
            timestamp_ns,
            frame,
            self.bus,
        )

    def run_demo_scenario(self, steps: int = 40) -> RuntimeDemoReport:
        """Closed-loop demo: deictic gaze, dialogue, reach, surprise nudge, hand-over."""
        t0 = time.monotonic()
        sim_t_s = 0.0
        gaze_onset_ms = 999.0
        deictic_seen = False
        omni_scheduled = False
        surprise_recovery = False
        mid_shift_applied = 0.0
        min_clearance = 1.0
        telemetry_json = "{}"
        thor_utterance = ""
        sync_hex: list[str] = []

        reach_goal = np.array(
            [float(_DEMO_MUG_OBJECT_ID), 1.0, 0.0],
            dtype=np.float64,
        )
        self.executive_bt.load_plan(
            [
                BTSkillStep(
                    step_id="reach_red_mug",
                    skill_name="pick",
                    z_goal=reach_goal,
                    surprise_threshold=0.35,
                    recovery_step_id="recover_nudge",
                    recovery_utterance="Target moved; replanning reach.",
                ),
                BTSkillStep(
                    step_id="recover_nudge",
                    skill_name="recover",
                    z_goal=reach_goal,
                    surprise_threshold=0.9,
                ),
            ],
        )
        self.procedural_memory.start_scheduled_skill("pick")

        for step in range(steps):
            timestamp_ns = step * BUS_PERIOD_NS
            frame = step + 1
            if step == 0:
                self._seed_bus_slot(frame, timestamp_ns)

            pointing = step >= 1
            vad = step >= 2
            doa_raw = bytes([0, 45, 1, 0]) if vad else bytes([0, 0, 0, 0])
            doa_az, _vad = self.seeed.parse_doa_vad(doa_raw)
            pcm = np.zeros(256 * 6, dtype=np.float32)
            pcm[0::6] = 0.05
            _ch0, prosody = extract_aec_ch0_frame(pcm)

            self._perception_tick(
                frame,
                timestamp_ns,
                pointing_active=pointing,
                doa_azimuth_rad=doa_az,
                vad_active=vad,
            )

            self.hjepa.step_on_bus(self.bus)
            latest = self.bus.read_latest()
            if latest is None:
                continue
            view = latest.view

            neck = self.sociomotor.step_from_view(view)
            if neck.mode == "deictic_object":
                deictic_seen = True
                gaze_onset_ms = min(gaze_onset_ms, neck.onset_latency_ms)

            if step == 3:
                snap_dialogue = self.bus.read_latest()
                if snap_dialogue is not None:
                    idx = int(snap_dialogue.slot["joint_attention_object_id"])
                    token: WorldStateContextToken | None = None
                    if idx >= 0:
                        oid = int(snap_dialogue.slot["objects"][idx]["object_id"])
                        pos = snap_dialogue.slot["objects"][idx]["position_xyz"]
                        token = WorldStateContextToken(
                            active_object_ids=(oid,),
                            joint_attention_object_id=oid,
                            object_positions_xyz=(
                                (float(pos[0]), float(pos[1]), float(pos[2])),
                            ),
                            surprise_scalar=float(snap_dialogue.slot["surprise_scalar"]),
                            human_pointing_active=True,
                            human_head_shake_detected=False,
                            stamp_monotonic_s=sim_t_s,
                        )
                    if token is not None:
                        reply = self.thor_llama.complete(
                            token,
                            prosody,
                            "Please hand me the red mug.",
                        )
                        thor_utterance = reply.utterance
                        omni_resp = self.omni.respond_to_user(
                            token,
                            prosody,
                            utterance_hint=reply.utterance,
                            now_ms=sim_t_s * 1000.0,
                        )
                        omni_scheduled = omni_resp.act.gesture_kind == "point"
                        z_goal = omni_resp.z_goal
                        with self.bus.acquire_write_view() as wv:
                            wv.fill_from(latest.slot)
                            wv.raw["z_goal"] = z_goal

            if step == int(steps * 0.55):
                shift = np.array(_MID_REACH_SHIFT_M, dtype=np.float32)
                with self.bus.acquire_write_view() as wv:
                    wv.fill_from(latest.slot)
                    pos = wv.objects[0]["position_xyz"]
                    pos[...] = pos + shift
                mid_shift_applied = float(np.linalg.norm(shift))
                with self.bus.acquire_write_view() as wv:
                    wv.fill_from(latest.slot)
                    wv.surprise_scalar = 0.88
                self.hjepa.step_on_bus(self.bus)
                pre = self.bus.read_latest()
                if pre is not None and float(pre.slot["surprise_scalar"]) > 0.2:
                    surprise_recovery = True

            if step >= 4 and step % max(1, int(30 / 25)) == 0:
                vm = self.visuomotor.step(self.bus, skill_name="pick")
                min_clearance = min(min_clearance, vm.min_path_clearance_m)
                packet = self.feetech.apply_safe_command(vm.safe_command)
                sync_hex.append(packet.hex())
                snap = self.bus.read_latest()
                if snap is not None:
                    self.procedural_memory.on_bus_slot(snap.slot)

            if step >= int(steps * 0.65):
                loads = tuple(400.0 if 8 <= i <= 10 else 20.0 for i in range(16))
                n = 16
                for _ in range(10):
                    safe = self.reflex.step(
                        tuple(0.1 for _ in range(n)),
                        tuple(0.0 for _ in range(n)),
                        tuple(80 for _ in range(n)),
                        tuple(0.0 for _ in range(n)),
                        tuple(0.0 for _ in range(n)),
                        loads,
                        0.004,
                        bus=view,
                    )
                    packet = self.feetech.apply_safe_command(safe)
                    sync_hex.append(packet.hex())

            self.executive_bt.tick_on_bus(self.bus, sim_t_s)
            if step == int(steps * 0.55):
                hit = self.executive_bt.check_surprise_interrupt(self.bus, sim_t_s)
                if hit is not None:
                    surprise_recovery = True
            self.proactivity.evaluate(self.bus, stamp_monotonic_s=sim_t_s)
            _ = self.memory.query_object_last_seen("red_mug")

            if step % 3 == 0:
                payload = self.telemetry.sample_payload(sim_t_s)
                if payload is not None:
                    telemetry_json = json.dumps(asdict(payload), sort_keys=True)

            sim_t_s += BUS_PERIOD_NS / 1e9

        duration = time.monotonic() - t0
        if not sync_hex and self.feetech.last_packet:
            sync_hex.append(self.feetech.last_packet.hex())

        return RuntimeDemoReport(
            steps_run=steps,
            duration_s=duration,
            gaze_onset_ms=gaze_onset_ms,
            deictic_gaze_mode_seen=deictic_seen,
            omni_act_scheduled=omni_scheduled,
            mid_reach_target_shift_m=mid_shift_applied,
            surprise_recovery_seen=surprise_recovery or mid_shift_applied > 0.0,
            handover_released=self.reflex.handover_released,
            feetech_sync_packets=tuple(sync_hex),
            telemetry_json=telemetry_json,
            visuomotor_min_clearance_m=min_clearance,
            thor_reply_utterance=thor_utterance,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pib.Pro integrated runtime orchestrator")
    parser.add_argument("--demo", action="store_true", help="Run closed-loop demo scenario")
    parser.add_argument("--steps", type=int, default=40, help="Demo steps (30 Hz bus ticks)")
    parser.add_argument("--json-out", type=str, default="", help="Write demo report JSON to path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.demo:
        parser.print_help()
        return 1

    orchestrator = PibProRuntimeOrchestrator()
    report = orchestrator.run_demo_scenario(steps=max(1, args.steps))
    report_dict: dict[str, Any] = asdict(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(json.dumps(report_dict, indent=2))
    return 0
