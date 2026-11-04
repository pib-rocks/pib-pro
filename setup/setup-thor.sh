#!/usr/bin/env bash
# Idempotent Jetson AGX Thor host setup for pib.Pro (JetPack 7 / L4T).
set -euo pipefail

DRY_RUN=0
PIB_PRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIB_DATA_MODELS="/data/models"
PIB_DATA_HF="/data/huggingface"
PIB_SHM_SIZE="8G"
SYSTEMD_UNIT="/etc/systemd/system/pib-pro.service"
DEFAULT_IFACE="${PIB_ETH_IFACE:-eth0}"

usage() {
  echo "Usage: $0 [--dry-run]"
  exit 1
}

log() {
  echo "[setup-thor] $*"
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN: $*"
  else
    log "EXEC: $*"
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      usage
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

log "pib-pro root: ${PIB_PRO_ROOT}"
log "dry-run mode: ${DRY_RUN}"

check_jetpack_l4t() {
  log "Checking JetPack 7 / L4T release marker"
  if [[ -f /etc/nv_tegra_release ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      log "DRY-RUN: would verify JetPack 7 / L4T from /etc/nv_tegra_release"
    else
      grep -q "R36" /etc/nv_tegra_release || grep -q "JetPack" /etc/nv_tegra_release || {
        log "WARN: unexpected L4T release; expected JetPack 7 class (R36+)"
      }
    fi
  else
    if [[ "$DRY_RUN" -eq 1 ]]; then
      log "DRY-RUN: skip L4T file check (non-Jetson CI host)"
    else
      log "ERROR: /etc/nv_tegra_release not found; run on Jetson AGX Thor"
      exit 1
    fi
  fi
}

configure_eee_off() {
  log "Disabling Energy Efficient Ethernet (EEE) on ${DEFAULT_IFACE}"
  if command -v ethtool >/dev/null 2>&1; then
    run_cmd ethtool --set-eee "${DEFAULT_IFACE}" eee off || log "WARN: ethtool EEE off failed (iface may differ)"
  else
    log "DRY-RUN: ethtool not installed; would disable EEE on ${DEFAULT_IFACE}"
  fi
}

configure_ptp_grandmaster() {
  log "Configuring IEEE-1588 PTP Grandmaster (ptp4l + phc2sys)"
  local ptp4l_cfg="${PIB_PRO_ROOT}/setup/ptp4l-grandmaster.conf"
  log "PTP config template: ${ptp4l_cfg}"
  run_cmd mkdir -p /etc/linuxptp
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN: would install ${ptp4l_cfg} -> /etc/linuxptp/ptp4l.conf"
    log "DRY-RUN: would enable ptp4l on ${DEFAULT_IFACE} and phc2sys sync"
  else
    if [[ -f "$ptp4l_cfg" ]]; then
      cp "$ptp4l_cfg" /etc/linuxptp/ptp4l.conf
    fi
    systemctl enable ptp4l || true
    systemctl enable phc2sys || true
  fi
}

configure_limits_and_shm() {
  log "Raising memlock/rtprio limits and /dev/shm for 30 Hz bus + 250 Hz Layer-0"
  local limits_dropin="/etc/security/limits.d/99-pib-pro.conf"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN: would write ${limits_dropin} (memlock unlimited, rtprio 99)"
    log "DRY-RUN: would mount /dev/shm size=${PIB_SHM_SIZE} via fstab or tmpfs"
  else
    cat >"$limits_dropin" <<'EOF'
@pib soft memlock unlimited
@pib hard memlock unlimited
@pib soft rtprio 99
@pib hard rtprio 99
EOF
    if ! grep -q "pib-pro-shm" /etc/fstab 2>/dev/null; then
      echo "tmpfs /dev/shm tmpfs defaults,size=${PIB_SHM_SIZE},mode=1777 0 0 # pib-pro-shm" >>/etc/fstab
    fi
  fi
}

configure_model_cache_dirs() {
  log "Persistent model caches: ${PIB_DATA_MODELS}, ${PIB_DATA_HF}"
  run_cmd mkdir -p "${PIB_DATA_MODELS}" "${PIB_DATA_HF}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    chmod 775 "${PIB_DATA_MODELS}" "${PIB_DATA_HF}" || true
  fi
}

write_systemd_unit() {
  log "Generating systemd unit ${SYSTEMD_UNIT}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN: would write pib-pro.service (WorkingDirectory=${PIB_PRO_ROOT})"
    log "DRY-RUN: systemctl daemon-reload && systemctl enable pib-pro.service"
  else
    cat >"$SYSTEMD_UNIT" <<EOF
[Unit]
Description=pib.Pro cognitive stack (Jetson AGX Thor)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${PIB_PRO_ROOT}
Environment=PIB_MODEL_CACHE=${PIB_DATA_MODELS}
Environment=HF_HOME=${PIB_DATA_HF}
ExecStart=/usr/bin/python3 -m pib_pro
Restart=on-failure
LimitMEMLOCK=infinity
LimitRTPRIO=99

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable pib-pro.service
  fi
}

check_jetpack_l4t
configure_eee_off
configure_ptp_grandmaster
configure_limits_and_shm
configure_model_cache_dirs
write_systemd_unit

log "setup-thor complete (dry-run=${DRY_RUN})"
log "MARKER: PIB_PRO_SETUP_OK"
