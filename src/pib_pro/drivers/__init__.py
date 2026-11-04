"""Native hardware driver adapters for pib.Pro on Jetson AGX Thor."""

from __future__ import annotations

from pib_pro.drivers.chest_oak4 import ChestOak4Driver
from pib_pro.drivers.feetech_sts import (
    STS_ADDR_ACC,
    STS_ADDR_PRESENT_POS,
    STS_BAUDRATE,
    STS_BROADCAST_ID,
    STS_CENTER_TICKS,
    STS_INST_READ,
    STS_INST_SYNC_WRITE,
    STS_TICKS_PER_REV,
    FeetechStsBusDriver,
    InMemoryLoopbackSerialTransport,
    SerialTransportProtocol,
    build_sync_write_pos_ex_packet,
    parse_present_pos_speed_load_response,
    radians_to_sts_ticks,
    sts_checksum,
    sts_ticks_to_radians,
)
from pib_pro.drivers.hardware_backends import (
    ChestOak4HardwareStatus,
    DiscoveredFeetechPort,
    DiscoveredSeeedMicArray,
    discover_feetech_serial_ports,
    discover_seeed_respeaker_usb,
    probe_chest_oak4_hardware,
)
from pib_pro.drivers.seeed_respeaker import (
    SeeedRespeakerDriver,
    extract_aec_ch0_frame,
    parse_xvf3000_doa_vad_bytes,
)

__all__ = [
    "STS_ADDR_ACC",
    "STS_ADDR_PRESENT_POS",
    "STS_BAUDRATE",
    "STS_BROADCAST_ID",
    "STS_CENTER_TICKS",
    "STS_INST_READ",
    "STS_INST_SYNC_WRITE",
    "STS_TICKS_PER_REV",
    "ChestOak4HardwareStatus",
    "ChestOak4Driver",
    "DiscoveredFeetechPort",
    "DiscoveredSeeedMicArray",
    "discover_feetech_serial_ports",
    "discover_seeed_respeaker_usb",
    "probe_chest_oak4_hardware",
    "FeetechStsBusDriver",
    "InMemoryLoopbackSerialTransport",
    "SeeedRespeakerDriver",
    "SerialTransportProtocol",
    "build_sync_write_pos_ex_packet",
    "extract_aec_ch0_frame",
    "parse_present_pos_speed_load_response",
    "parse_xvf3000_doa_vad_bytes",
    "radians_to_sts_ticks",
    "sts_checksum",
    "sts_ticks_to_radians",
]
