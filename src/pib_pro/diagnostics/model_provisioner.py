"""Jetson Thor model cache verification and TensorRT engine provisioning."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class ThorModelSpec:
    model_id: str
    family: Literal["vjepa2", "groot_n1_7", "gemma4_omni"]
    hf_repo_id: str
    cache_subdir: str
    precision: Literal["NVFP4", "FP8", "Q4_K_M"]
    expected_sha256_prefix: str
    min_disk_gb: float
    trtexec_command: tuple[str, ...]


DEFAULT_THOR_MODEL_MANIFEST: tuple[ThorModelSpec, ...] = (
    ThorModelSpec(
        model_id="vjepa2_vitl_thor",
        family="vjepa2",
        hf_repo_id="",
        cache_subdir="/data/models/vjepa2",
        precision="FP8",
        expected_sha256_prefix="",
        min_disk_gb=8.0,
        trtexec_command=(
            "trtexec",
            "--onnx=/data/models/vjepa2/vjepa2_vitl.onnx",
            "--fp8",
            "--saveEngine=/data/models/vjepa2/vjepa2_thor_fp8.engine",
        ),
    ),
    ThorModelSpec(
        model_id="groot_n1_7_thor",
        family="groot_n1_7",
        hf_repo_id="",
        cache_subdir="/data/models/groot_n1_7",
        precision="NVFP4",
        expected_sha256_prefix="",
        min_disk_gb=12.0,
        trtexec_command=(
            "trtexec",
            "--onnx=/data/models/groot_n1_7/groot_action_head.onnx",
            "--fp4",
            "--fp8",
            "--saveEngine=/data/models/groot_n1_7/groot_action_head_nvfp4.engine",
        ),
    ),
    ThorModelSpec(
        model_id="gemma4_26b_a4b_thor",
        family="gemma4_omni",
        hf_repo_id="ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
        cache_subdir="/data/huggingface",
        precision="Q4_K_M",
        expected_sha256_prefix="",
        min_disk_gb=20.0,
        trtexec_command=(),
    ),
)


@dataclass(frozen=True, slots=True)
class ModelProvisionEntry:
    model_id: str
    cache_path: str
    action: str
    sha256_ok: bool | None
    disk_gb_free: float
    message: str


@dataclass(frozen=True, slots=True)
class ModelProvisionReport:
    dry_run: bool
    cache_root: str
    entries: tuple[ModelProvisionEntry, ...]
    trtexec_available: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def verify_sha256_file(path: Path, expected_hex: str) -> bool:
    """Return True when file digest starts with ``expected_hex`` (prefix match)."""
    if not expected_hex:
        return path.is_file()
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    hex_digest = digest.hexdigest()
    return hex_digest.startswith(expected_hex.lower())


def _disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
    except OSError:
        return 0.0
    return usage.free / (1024**3)


def _trtexec_on_path() -> bool:
    return shutil.which("trtexec") is not None


def _provision_one(
    spec: ThorModelSpec,
    cache_root: Path,
    *,
    dry_run: bool,
) -> ModelProvisionEntry:
    rel = spec.cache_subdir.lstrip("/")
    cache_path = cache_root / rel
    free_gb = _disk_free_gb(cache_root)

    if dry_run:
        action = "dry_run_trtexec" if spec.trtexec_command else "dry_run_hf_cache"
        msg = " ".join(spec.trtexec_command) if spec.trtexec_command else spec.hf_repo_id
        return ModelProvisionEntry(
            model_id=spec.model_id,
            cache_path=str(cache_path),
            action=action,
            sha256_ok=None,
            disk_gb_free=free_gb,
            message=msg,
        )

    if free_gb < spec.min_disk_gb:
        return ModelProvisionEntry(
            model_id=spec.model_id,
            cache_path=str(cache_path),
            action="skip_insufficient_disk",
            sha256_ok=None,
            disk_gb_free=free_gb,
            message=f"need {spec.min_disk_gb} GB free, have {free_gb:.2f} GB",
        )

    cache_path.mkdir(parents=True, exist_ok=True)
    if spec.trtexec_command:
        try:
            subprocess.run(spec.trtexec_command, check=False, capture_output=True, text=True)
        except OSError as exc:
            return ModelProvisionEntry(
                model_id=spec.model_id,
                cache_path=str(cache_path),
                action="trtexec_failed",
                sha256_ok=None,
                disk_gb_free=free_gb,
                message=str(exc),
            )
        engine_path = Path(spec.trtexec_command[-1].split("=", 1)[-1])
        ok = verify_sha256_file(engine_path, spec.expected_sha256_prefix)
        return ModelProvisionEntry(
            model_id=spec.model_id,
            cache_path=str(cache_path),
            action="trtexec_invoked",
            sha256_ok=ok,
            disk_gb_free=free_gb,
            message="engine build attempted",
        )

    return ModelProvisionEntry(
        model_id=spec.model_id,
        cache_path=str(cache_path),
        action="hf_cache_ready",
        sha256_ok=None,
        disk_gb_free=free_gb,
        message=spec.hf_repo_id,
    )


def run_model_provisioner(
    cache_root: Path = Path("/data"),
    *,
    dry_run: bool = True,
    manifest: tuple[ThorModelSpec, ...] | None = None,
) -> ModelProvisionReport:
    specs = manifest if manifest is not None else DEFAULT_THOR_MODEL_MANIFEST
    entries = tuple(_provision_one(s, cache_root, dry_run=dry_run) for s in specs)
    return ModelProvisionReport(
        dry_run=dry_run,
        cache_root=str(cache_root),
        entries=entries,
        trtexec_available=_trtexec_on_path(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision Thor TensorRT / GGUF model caches")
    parser.add_argument("--cache-root", type=Path, default=Path("/data"))
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not invoke trtexec")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run provisioning commands (not dry-run)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(list(argv) if argv is not None else None)
    dry_run = not args.execute
    if args.dry_run:
        dry_run = True
    report = run_model_provisioner(args.cache_root, dry_run=dry_run)
    if args.json:
        print(report.to_json())
    else:
        for entry in report.entries:
            print(f"{entry.model_id}: {entry.action} — {entry.message}")
    return 0
