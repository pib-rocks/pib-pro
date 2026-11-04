"""Wave 5: hardware autodiscovery, kinematics, HUD server, model provisioner."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from pib_pro.common.kinematics import (
    DEFAULT_DUAL_ARM_CONFIG,
    JOINT_LIMIT_HI,
    JOINT_LIMIT_LO,
    chest_camera_fov_penalty,
    forward_kinematics_arm_6dof,
    solve_arm_ik_dls,
)
from pib_pro.diagnostics.model_provisioner import (
    DEFAULT_THOR_MODEL_MANIFEST,
    run_model_provisioner,
    verify_sha256_file,
)
from pib_pro.diagnostics.model_provisioner import (
    main as provision_main,
)
from pib_pro.drivers.hardware_backends import (
    discover_feetech_serial_ports,
    discover_seeed_respeaker_usb,
    reset_feetech_discovery_cache_for_tests,
)
from pib_pro.telemetry_gateway.gateway import NonBlockingTelemetryGateway
from pib_pro.telemetry_gateway.hud_server import TelemetryHudServer, render_hud_html
from pib_pro.world_state_bus import WorldStateRingBuffer


def _workbench_targets(side: str, count: int = 15) -> list[np.ndarray]:
    cfg = DEFAULT_DUAL_ARM_CONFIG
    shoulder = np.array(
        cfg.left_shoulder_xyz if side == "left" else cfg.right_shoulder_xyz,
        dtype=np.float64,
    )
    max_reach = cfg.upper_arm_m + cfg.forearm_m + cfg.hand_m - 0.01
    xs = np.linspace(0.22, 0.45, 5)
    ys = np.linspace(-0.28, 0.28, 3)
    zs = np.linspace(0.62, 0.95, 3)
    picked: list[np.ndarray] = []
    for x in xs:
        for y in ys:
            for z in zs:
                t = np.array([x, y, z], dtype=np.float64)
                if float(np.linalg.norm(t - shoulder)) > max_reach:
                    continue
                picked.append(t)
                if len(picked) >= count:
                    return picked
    msg = f"not enough reachable workbench targets for {side}"
    raise RuntimeError(msg)


def test_discover_feetech_serial_ports_synthetic_by_id(tmp_path: Path) -> None:
    reset_feetech_discovery_cache_for_tests()
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    tty = tmp_path / "ttyUSB0"
    tty.write_text("", encoding="utf-8")
    link_name = "usb-FTDI_USB__Serial_Converter_FTA3ABCD-if00-port0"
    (by_id / link_name).symlink_to(tty)

    calls: list[str] = []

    def probe(path: str) -> tuple[int, ...]:
        calls.append(path)
        return (20, 21, 22)

    first = discover_feetech_serial_ports(by_id, probe_fn=probe)
    assert len(first) == 1
    assert first[0].assigned_group == "left_arm_and_head"
    assert first[0].responding_servo_ids == (20, 21, 22)

    def probe_other(_path: str) -> tuple[int, ...]:
        return (1, 2, 3)

    second = discover_feetech_serial_ports(by_id, probe_fn=probe_other)
    assert second[0].assigned_group == "left_arm_and_head"
    assert second[0].responding_servo_ids == (1, 2, 3)

    (by_id / link_name).unlink()
    third = discover_feetech_serial_ports(by_id, probe_fn=probe)
    assert third == ()


def test_discover_feetech_right_arm_classification(tmp_path: Path) -> None:
    reset_feetech_discovery_cache_for_tests()
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    tty = tmp_path / "ttyACM0"
    tty.write_text("", encoding="utf-8")
    name = "usb-1a86_USB_Serial-if00"
    (by_id / name).symlink_to(tty)
    found = discover_feetech_serial_ports(by_id, probe_fn=lambda _p: (5, 6, 7))
    assert found[0].assigned_group == "right_arm"


def test_discover_seeed_respeaker_usb_synthetic(tmp_path: Path) -> None:
    usb_root = tmp_path / "sys" / "bus" / "usb" / "devices"
    dev = usb_root / "1-2.3"
    dev.mkdir(parents=True)
    (dev / "idVendor").write_text("2886\n", encoding="utf-8")
    (dev / "idProduct").write_text("0018\n", encoding="utf-8")

    cards = tmp_path / "proc" / "asound" / "cards"
    cards.parent.mkdir(parents=True)
    cards.write_text(
        " 0 [ArrayUAC10     ]: USB-Audio - ReSpeaker 4 Mic Array (UAC1.0)\n",
        encoding="utf-8",
    )

    mic = discover_seeed_respeaker_usb(usb_root, cards)
    assert mic.usb_vid == 0x2886
    assert mic.usb_pid == 0x0018
    assert mic.usb_bus_dev == "1-2.3"
    assert mic.alsa_card_index == 0
    assert mic.is_present is True


@pytest.mark.parametrize("side", ["left", "right"])
def test_forward_kinematics_and_ik_workbench(side: str) -> None:
    targets = _workbench_targets(side, 15)
    for target in targets:
        ik = solve_arm_ik_dls(target, side)  # type: ignore[arg-type]
        assert ik.position_error_m <= 0.005
        fk = forward_kinematics_arm_6dof(ik.q_6dof, side)  # type: ignore[arg-type]
        err = float(np.linalg.norm(fk.end_effector_xyz - target))
        assert err <= 0.005
        assert np.all(ik.q_6dof >= JOINT_LIMIT_LO - 1e-9)
        assert np.all(ik.q_6dof <= JOINT_LIMIT_HI + 1e-9)


def test_ik_chest_camera_fov_avoidance() -> None:
    target = np.array([0.34, 0.05, 0.78], dtype=np.float64)
    with_avoid = solve_arm_ik_dls(target, "left", avoid_chest_camera_fov=True)
    without_avoid = solve_arm_ik_dls(target, "left", avoid_chest_camera_fov=False)
    fk_a = forward_kinematics_arm_6dof(with_avoid.q_6dof, "left")
    fk_b = forward_kinematics_arm_6dof(without_avoid.q_6dof, "left")
    pen_a = chest_camera_fov_penalty(fk_a.elbow_xyz, fk_a.wrist_xyz)
    pen_b = chest_camera_fov_penalty(fk_b.elbow_xyz, fk_b.wrist_xyz)
    assert pen_a <= pen_b + 1e-6
    assert with_avoid.position_error_m <= 0.005


def test_render_hud_html_offline() -> None:
    html = render_hud_html()
    assert "<!DOCTYPE html>" in html
    assert re.search(r"https?://", html) is None


def test_telemetry_hud_server_endpoints() -> None:
    bus = WorldStateRingBuffer(capacity=4)
    gateway = NonBlockingTelemetryGateway(bus=bus)
    server = TelemetryHudServer(gateway, host="127.0.0.1", port=0)
    server.start_background()

    try:
        base = f"http://127.0.0.1:{server.port}"
        for path in ("/", "/api/v1/health", "/api/v1/telemetry"):
            with urllib.request.urlopen(f"{base}{path}", timeout=5) as resp:
                assert resp.status == 200
                body = resp.read()
                if path.endswith("health"):
                    data = json.loads(body.decode("utf-8"))
                    assert data == {"ok": True, "service": "pib-pro-hud"}
                elif path.endswith("telemetry"):
                    json.loads(body.decode("utf-8"))
                else:
                    text = body.decode("utf-8")
                    assert re.search(r"https?://", text) is None
    finally:
        server.stop()


def test_run_model_provisioner_dry_run(tmp_path: Path) -> None:
    report = run_model_provisioner(tmp_path, dry_run=True)
    assert report.dry_run is True
    assert len(report.entries) == len(DEFAULT_THOR_MODEL_MANIFEST)
    assert report.entries[0].action == "dry_run_trtexec"


def test_verify_sha256_file(tmp_path: Path) -> None:
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"wave5")
    digest = __import__("hashlib").sha256(b"wave5").hexdigest()
    assert verify_sha256_file(blob, digest[:8]) is True
    assert verify_sha256_file(blob, "deadbeef") is False


def test_provision_models_cli_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    rc = provision_main(["--dry-run", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry_run" in out
