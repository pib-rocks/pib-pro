"""Seeed ReSpeaker Mic Array v2.0 (XVF3000) driver adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pib_pro.layer3_cognition.omni_dialogue import ProsodyFeatures, extract_prosody_from_pcm
from pib_pro.perception_oak4.human_state import (
    SeeedMicArrayConfig,
    seeed_doa_deg_to_rep103_azimuth_rad,
)


def parse_xvf3000_doa_vad_bytes(raw_4bytes: bytes) -> tuple[float, bool]:
    """Parse XVF3000 USB vendor DOA/VAD response (4 bytes)."""
    if len(raw_4bytes) != 4:
        msg = "parse_xvf3000_doa_vad_bytes expects exactly 4 bytes"
        raise ValueError(msg)
    doa_deg = float(raw_4bytes[0] | (raw_4bytes[1] << 8))
    vad_active = bool(raw_4bytes[2] & 0x01)
    azimuth = seeed_doa_deg_to_rep103_azimuth_rad(
        doa_deg,
        mounting_yaw_offset_rad=0.0,
    )
    return azimuth, vad_active


@dataclass(slots=True)
class SeeedRespeakerDriver:
    """Wraps Seeed mic array capture profile and AEC channel extraction."""

    config: SeeedMicArrayConfig

    @classmethod
    def default(cls) -> SeeedRespeakerDriver:
        return cls(SeeedMicArrayConfig())

    def parse_doa_vad(self, raw_4bytes: bytes) -> tuple[float, bool]:
        return parse_xvf3000_doa_vad_bytes(raw_4bytes)


def extract_aec_ch0_frame(
    interleaved_6ch_pcm: NDArray[np.float32],
) -> tuple[NDArray[np.float32], ProsodyFeatures]:
    """Slice AEC-clean channel 0 (256 frames) and compute prosody features."""
    pcm = np.asarray(interleaved_6ch_pcm, dtype=np.float32)
    period = 256
    if pcm.ndim == 1:
        if pcm.size % (6 * period) != 0 and pcm.size >= period:
            frame = pcm[:period].astype(np.float32, copy=False)
        else:
            frame = pcm.reshape(-1, 6)[:period, 0]
    elif pcm.ndim == 2:
        if pcm.shape[0] == 6:
            frame = pcm[0, :period]
        elif pcm.shape[1] == 6:
            frame = pcm[:period, 0]
        else:
            msg = "interleaved_6ch_pcm must be (6, T) or (T, 6) or flat interleaved"
            raise ValueError(msg)
    else:
        msg = "interleaved_6ch_pcm must be 1D or 2D"
        raise ValueError(msg)
    frame = np.asarray(frame[:period], dtype=np.float32)
    prosody = extract_prosody_from_pcm(
        frame,
        sample_rate_hz=16000,
        aec_clean_channel_index=0,
    )
    return frame, prosody
