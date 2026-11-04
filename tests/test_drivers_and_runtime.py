"""PR-2253 hardware drivers, Thor llama client, and runtime orchestrator tests."""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from pib_pro.common import Vector3D
from pib_pro.drivers import (
    STS_ADDR_ACC,
    STS_INST_SYNC_WRITE,
    ChestOak4Driver,
    FeetechStsBusDriver,
    InMemoryLoopbackSerialTransport,
    build_sync_write_pos_ex_packet,
    extract_aec_ch0_frame,
    parse_present_pos_speed_load_response,
    parse_xvf3000_doa_vad_bytes,
    radians_to_sts_ticks,
    sts_checksum,
    sts_ticks_to_radians,
)
from pib_pro.layer0_reflex.somatic_reflex import SomaticReflexController
from pib_pro.layer3_cognition import (
    ThorLlamaClientConfig,
    ThorLlamaServerClient,
    WorldStateContextToken,
)
from pib_pro.layer3_cognition.omni_dialogue import ProsodyFeatures
from pib_pro.perception_oak4 import (
    Detection2DWithDepth,
    HumanObservation3D,
    Oak4CameraIntrinsics,
    PoseTransform3D,
)
from pib_pro.runtime.orchestrator import PibProRuntimeOrchestrator, main
from pib_pro.world_state_bus import WorldStateRingBuffer


def test_radians_sts_roundtrip_and_center() -> None:
    assert radians_to_sts_ticks(0.0) == 2048
    assert radians_to_sts_ticks(-math.pi) == pytest.approx(0, abs=1)
    ticks = radians_to_sts_ticks(0.25)
    assert 0 <= ticks <= 4095
    assert sts_ticks_to_radians(ticks) == pytest.approx(0.25, abs=0.01)


def test_build_sync_write_pos_ex_packet_header_and_clamp() -> None:
    packet = build_sync_write_pos_ex_packet(
        [1, 2],
        [0.0, 0.1],
        [1000, 1000],
        [1, 300],
    )
    assert packet[0:3] == bytes([0xFF, 0xFF, 0xFE])
    assert packet[4] == STS_INST_SYNC_WRITE
    assert packet[5] == STS_ADDR_ACC
    body = packet[2:-1]
    assert sts_checksum(body) == packet[-1]
    acc_first = packet[8]
    acc_second = packet[16]
    assert acc_first == 30
    assert acc_second == 254


def test_parse_present_pos_speed_load_response() -> None:
    pos_ticks = radians_to_sts_ticks(0.2)
    speed_raw = 0x8000 | 120
    load_raw = 0x0400 | 180
    packet = bytes(
        [
            0xFF,
            0xFF,
            5,
            7,
            0,
            pos_ticks & 0xFF,
            (pos_ticks >> 8) & 0xFF,
            speed_raw & 0xFF,
            (speed_raw >> 8) & 0xFF,
            load_raw & 0xFF,
            (load_raw >> 8) & 0xFF,
            0,
        ],
    )
    body = packet[2:-1]
    packet = packet[:-1] + bytes([sts_checksum(body)])
    sid, pos_rad, vel_rad_s, load_ma = parse_present_pos_speed_load_response(packet)
    assert sid == 5
    assert pos_rad == pytest.approx(sts_ticks_to_radians(pos_ticks), abs=0.01)
    assert vel_rad_s < 0.0
    assert load_ma == pytest.approx(-180.0)


def test_feetech_bus_driver_writes_sync_packet() -> None:
    transport = InMemoryLoopbackSerialTransport()
    driver = FeetechStsBusDriver(transport=transport, servo_ids=(1, 2, 3))
    reflex = SomaticReflexController()
    cmd = reflex.step(
        (0.1, 0.2, 0.0),
        (0.0, 0.0, 0.0),
        (1, 80, 200),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        0.004,
    )
    pkt = driver.apply_safe_command(cmd)
    assert transport.written
    assert pkt[4] == STS_INST_SYNC_WRITE
    assert pkt.hex() == driver.last_packet.hex()


def test_seeed_doa_vad_and_aec_ch0_extraction() -> None:
    az, vad = parse_xvf3000_doa_vad_bytes(bytes([45, 0, 1, 0]))
    assert vad is True
    assert az == pytest.approx(math.pi / 4, abs=0.05)
    pcm = np.zeros((256, 6), dtype=np.float32)
    pcm[:, 0] = 0.08
    ch0, prosody = extract_aec_ch0_frame(pcm)
    assert ch0.shape == (256,)
    assert isinstance(prosody, ProsodyFeatures)


def test_chest_oak4_self_arm_filter_and_bus_write() -> None:
    intr = Oak4CameraIntrinsics(fx=700.0, fy=700.0, cx=640.0, cy=360.0, width=1280, height=720)
    bus = WorldStateRingBuffer(capacity=8)
    driver = ChestOak4Driver(intrinsics=intr)
    extrinsic = PoseTransform3D.oak4_chest_optical_to_rep103_base()
    left_hand = Vector3D(0.35, 0.12, 0.72)
    right_hand = Vector3D(0.35, -0.12, 0.72)
    table_pos = Vector3D(0.40, 0.0, 0.70)
    rel = table_pos - extrinsic.translation_base_from_optical
    p_opt = extrinsic.rotation_base_from_optical.T @ rel.as_array()
    u = intr.fx * float(p_opt[0]) / float(p_opt[2]) + intr.cx
    v = intr.fy * float(p_opt[1]) / float(p_opt[2]) + intr.cy
    mug = Detection2DWithDepth(
        class_id=1,
        label="red_mug",
        bbox_xyxy=(u - 20, v - 20, u + 20, v + 20),
        depth_m=float(p_opt[2]),
        confidence=0.9,
        appearance_embedding=np.array([0.9, 0.1, 0.0], dtype=np.float32),
        half_extents_xyz=(0.04, 0.04, 0.04),
    )
    hand_rel = left_hand - extrinsic.translation_base_from_optical
    hp = extrinsic.rotation_base_from_optical.T @ hand_rel.as_array()
    hu = intr.fx * float(hp[0]) / float(hp[2]) + intr.cx
    hv = intr.fy * float(hp[1]) / float(hp[2]) + intr.cy
    hand_det = Detection2DWithDepth(
        class_id=99,
        label="hand_blob",
        bbox_xyxy=(hu - 15, hv - 15, hu + 15, hv + 15),
        depth_m=float(hp[2]),
        confidence=0.7,
        appearance_embedding=np.array([0.2, 0.2, 0.2], dtype=np.float32),
        half_extents_xyz=(0.03, 0.03, 0.03),
    )
    human = HumanObservation3D(
        head_position=Vector3D(0.0, 0.0, 1.35),
        gaze_direction=Vector3D(1.0, 0.0, 0.0),
        pointing_origin=Vector3D(0.0, 0.0, 1.35),
        pointing_direction=Vector3D(0.45, 0.0, -0.2),
        pointing_active=True,
        left_hand_position=left_hand,
        right_hand_position=right_hand,
        doa_azimuth_rad=0.0,
        vad_active=True,
    )
    driver.persistent_tracks = (table_pos,)
    filtered, winner = driver.ingest_frame(
        (hand_det, mug),
        human,
        left_hand,
        right_hand,
        timestamp_ns=1_000_000,
        frame_id=1,
        bus=bus,
    )
    labels = {d.label for d in filtered}
    assert "hand_blob" not in labels
    assert "red_mug" in labels
    latest = bus.read_latest()
    assert latest is not None
    assert int(latest.slot["objects"][0]["object_id"]) != 0
    assert winner >= 0


class _MockLlamaHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        _ = format, args

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "utterance": "Sure, I will take the red one.",
                                    "stressed_word": "red",
                                    "target_object_id": 1,
                                },
                            ),
                        },
                    },
                ],
            },
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_thor_llama_client_mock_server_and_offline_fallback() -> None:
    server = HTTPServer(("127.0.0.1", 0), _MockLlamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = WorldStateContextToken(
        active_object_ids=(1,),
        joint_attention_object_id=1,
        object_positions_xyz=((0.45, 0.0, 0.72),),
        surprise_scalar=0.1,
        human_pointing_active=True,
        human_head_shake_detected=False,
    )
    prosody = ProsodyFeatures(130.0, -28.0, 0.1, False, False)
    live = ThorLlamaServerClient(
        ThorLlamaClientConfig(base_url=f"http://127.0.0.1:{port}", timeout_s=2.0),
    )
    live_reply = live.complete(token, prosody, "hand me the red mug")
    assert live_reply.used_remote_model is True
    assert "red" in live_reply.utterance.lower()
    server.shutdown()

    offline = ThorLlamaServerClient(
        ThorLlamaClientConfig(base_url="http://127.0.0.1:1", timeout_s=0.05),
    )
    off_reply = offline.complete(token, prosody, "hand me the red mug")
    assert off_reply.used_remote_model is False
    assert off_reply.utterance


def test_runtime_demo_scenario_and_cli(tmp_path: Path) -> None:
    orch = PibProRuntimeOrchestrator()
    report = orch.run_demo_scenario(steps=40)
    assert report.deictic_gaze_mode_seen
    assert report.gaze_onset_ms < 120.0
    assert report.omni_act_scheduled
    assert report.mid_reach_target_shift_m == pytest.approx(0.04, abs=0.001)
    assert report.surprise_recovery_seen
    assert report.handover_released
    assert report.feetech_sync_packets
    assert any(p.startswith("fffffe") for p in report.feetech_sync_packets)
    telemetry = json.loads(report.telemetry_json)
    assert "world_state_summary" in telemetry
    assert "system_health" in telemetry

    out = tmp_path / "demo_report.json"
    code = main(["--demo", "--steps", "30", "--json-out", str(out)])
    assert code == 0
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["steps_run"] == 30
    assert saved["handover_released"] is True
