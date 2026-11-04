"""Feetech STS serial bus driver and SyncWritePosEx packet helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pib_pro.layer0_reflex.somatic_reflex import STS_ACC_MAX, STS_ACC_MIN, SafeMotorCommand

STS_BAUDRATE: int = 1_000_000
STS_BROADCAST_ID: int = 0xFE
STS_INST_READ: int = 0x02
STS_INST_SYNC_WRITE: int = 0x83
STS_ADDR_ACC: int = 41
STS_ADDR_PRESENT_POS: int = 56
STS_CENTER_TICKS: int = 2048
STS_TICKS_PER_REV: int = 4096
_STS_DATA_LEN: int = 7
_STS_SPEED_TICKS_PER_RAD_S: float = STS_TICKS_PER_REV / (2.0 * math.pi)


class SerialTransportProtocol(Protocol):
    def write(self, data: bytes) -> int: ...

    def read(self, size: int) -> bytes: ...


def radians_to_sts_ticks(rad: float, invert: bool = False) -> int:
    """Map joint radians to STS position ticks (0 rad -> 2048, clamped 0..4095)."""
    angle = -rad if invert else rad
    ticks = STS_CENTER_TICKS + int(round((angle / (2.0 * math.pi)) * STS_TICKS_PER_REV))
    return max(0, min(STS_TICKS_PER_REV - 1, ticks))


def sts_ticks_to_radians(ticks: int, invert: bool = False) -> float:
    """Inverse of ``radians_to_sts_ticks``."""
    t = int(ticks) & 0xFFF
    rad = (t - STS_CENTER_TICKS) * (2.0 * math.pi) / STS_TICKS_PER_REV
    return -rad if invert else rad


def sts_checksum(packet_body_without_header: bytes) -> int:
    """Feetech/Dynamixel checksum: ``(~sum(body)) & 0xFF``."""
    return (~sum(packet_body_without_header)) & 0xFF


def _clamp_sts_acceleration(acc: int | float) -> int:
    return max(STS_ACC_MIN, min(STS_ACC_MAX, int(acc)))


def build_sync_write_pos_ex_packet(
    servo_ids: Sequence[int],
    positions_rad: Sequence[float],
    speeds_ticks_s: Sequence[int],
    accelerations: Sequence[int],
) -> bytes:
    """Build STS broadcast ``SyncWritePosEx`` frame starting at ``STS_ADDR_ACC``."""
    ids = list(servo_ids)
    n = len(ids)
    if not (len(positions_rad) == len(speeds_ticks_s) == len(accelerations) == n):
        msg = "servo_ids, positions_rad, speeds_ticks_s, accelerations must match length"
        raise ValueError(msg)
    params: list[int] = [STS_ADDR_ACC, _STS_DATA_LEN]
    for sid, pos_rad, spd, acc in zip(
        ids,
        positions_rad,
        speeds_ticks_s,
        accelerations,
        strict=True,
    ):
        acc_b = _clamp_sts_acceleration(acc)
        pos_ticks = radians_to_sts_ticks(pos_rad)
        spd_u = max(0, min(0xFFFF, int(spd)))
        params.extend(
            [
                int(sid) & 0xFF,
                acc_b,
                pos_ticks & 0xFF,
                (pos_ticks >> 8) & 0xFF,
                0,
                0,
                spd_u & 0xFF,
                (spd_u >> 8) & 0xFF,
            ],
        )
    length = 1 + len(params)
    body = bytes([STS_BROADCAST_ID, length, STS_INST_SYNC_WRITE, *params])
    chk = sts_checksum(body)
    return bytes([0xFF, 0xFF, *body, chk])


def _decode_sign_magnitude_u16(raw_lo: int, raw_hi: int, *, sign_bit: int) -> float:
    raw = (raw_hi << 8) | raw_lo
    magnitude = raw & ((1 << sign_bit) - 1)
    if raw & (1 << sign_bit):
        return -float(magnitude)
    return float(magnitude)


def parse_present_pos_speed_load_response(packet: bytes) -> tuple[int, float, float, float]:
    """Parse STS read response for present position, speed, and load."""
    if len(packet) < 11 or packet[0] != 0xFF or packet[1] != 0xFF:
        msg = "invalid STS status packet header"
        raise ValueError(msg)
    servo_id = int(packet[2])
    param_start = 5
    if len(packet) < param_start + 6:
        msg = "STS status packet too short for present pos block"
        raise ValueError(msg)
    pos_lo = packet[param_start]
    pos_hi = packet[param_start + 1]
    spd_lo = packet[param_start + 2]
    spd_hi = packet[param_start + 3]
    load_lo = packet[param_start + 4]
    load_hi = packet[param_start + 5]
    pos_ticks = pos_lo | (pos_hi << 8)
    pos_rad = sts_ticks_to_radians(pos_ticks)
    speed_ticks_s = _decode_sign_magnitude_u16(spd_lo, spd_hi, sign_bit=15)
    vel_rad_s = speed_ticks_s / _STS_SPEED_TICKS_PER_RAD_S
    load_ma = _decode_sign_magnitude_u16(load_lo, load_hi, sign_bit=10)
    return servo_id, pos_rad, vel_rad_s, load_ma


@dataclass(slots=True)
class InMemoryLoopbackSerialTransport:
    """In-process serial loopback for CI (records writes, serves scripted reads)."""

    read_queue: list[bytes] = field(default_factory=list)
    written: list[bytes] = field(default_factory=list)

    def enqueue_read(self, data: bytes) -> None:
        self.read_queue.append(data)

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        return len(data)

    def read(self, size: int) -> bytes:
        if not self.read_queue:
            return b""
        chunk = self.read_queue.pop(0)
        if size >= len(chunk):
            return chunk
        return chunk[:size]


@dataclass(slots=True)
class FeetechStsBusDriver:
    """Maps Layer-0 ``SafeMotorCommand`` to Feetech STS ``SyncWritePosEx`` on a serial bus."""

    transport: SerialTransportProtocol
    servo_ids: tuple[int, ...]
    invert_joint: tuple[bool, ...] = ()
    default_speed_ticks_s: int = 2400
    last_packet: bytes = field(default=b"", init=False)
    last_written_positions_rad: tuple[float, ...] = field(default=tuple(), init=False)

    def __post_init__(self) -> None:
        if not self.servo_ids:
            msg = "FeetechStsBusDriver requires at least one servo id"
            raise ValueError(msg)

    def _invert_for(self, index: int) -> bool:
        if index < len(self.invert_joint):
            return bool(self.invert_joint[index])
        return False

    def apply_safe_command(self, command: SafeMotorCommand) -> bytes:
        """Emit one broadcast sync-write from a reflex-safe motor command."""
        n = len(self.servo_ids)
        positions = list(command.position_targets_rad[:n])
        while len(positions) < n:
            positions.append(0.0)
        accs = list(command.sts_acceleration[:n])
        while len(accs) < n:
            accs.append(STS_ACC_MIN)
        speeds = [self.default_speed_ticks_s] * n
        packet = build_sync_write_pos_ex_packet(
            self.servo_ids,
            positions,
            speeds,
            accs,
        )
        self.transport.write(packet)
        self.last_packet = packet
        self.last_written_positions_rad = tuple(positions)
        return packet

    def read_joint_feedback(self, servo_id: int) -> tuple[int, float, float, float]:
        """Read one servo's present pos/speed/load (expects response bytes on transport)."""
        _ = servo_id
        data = self.transport.read(128)
        if not data:
            msg = "no STS feedback available on transport"
            raise RuntimeError(msg)
        return parse_present_pos_speed_load_response(data)
