# pib-pro

**pib.Pro** is the greenfield cognitive architecture for the [pib](https://pib.rocks) humanoid, targeting **NVIDIA Jetson AGX Thor** (JetPack 7, TensorRT NVFP4 inference container).

This repository defines typed layer contracts, REP-103 shared geometry, world-state bus slots, and Thor host diagnostics—without pulling heavy GPU runtime dependencies into CPU CI.

## Architecture overview

```text
┌─────────────────────────────────────────────────────────────┐
│ Layer 3 — cognition (OmniDialogue, System-3 deliberation)   │
├─────────────────────────────────────────────────────────────┤
│ Layer 2 — executive (surprise BT, communicative scheduling) │
├─────────────────────────────────────────────────────────────┤
│ Layer 1 — sociomotor autopilot + hybrid visuomotor          │
├─────────────────────────────────────────────────────────────┤
│ Layer 0 — somatic reflex (250 Hz safety envelope)           │
├─────────────────────────────────────────────────────────────┤
│ World model (HJEPA) ◄── perception (OAK-4) ◄── bus (30 Hz)  │
└─────────────────────────────────────────────────────────────┘
                              │
                    telemetry_gateway
```

Data flows through a **30 Hz unified-memory world-state bus**; Layer 0 runs a **250 Hz** reflex loop with strict safety envelopes. Perception (`perception_oak4`) feeds a hierarchical JEPA-style **world model**; executive and cognition layers schedule speech and subgoals under **proactivity tiers** (`T0_SILENT` … `T3_FORBIDDEN`).

## Module map

| Package | Role |
|---------|------|
| `pib_pro.common` | REP-103 frames, `Vector3D`, `Pose3D`, `Ray3D`, deictic intersection helpers |
| `pib_pro.world_state_bus` | Typed slot contracts and bus protocol |
| `pib_pro.perception_oak4` | OAK-4 tracks, human state, scene protocol |
| `pib_pro.world_model` | Latent state vector, HJEPA world-model protocol |
| `pib_pro.layer0_reflex` | Safety envelope, actuator commands, compliance |
| `pib_pro.layer1_autopilot` | Sociomotor targets, scored action chunks |
| `pib_pro.layer2_executive` | Surprise events, communicative acts |
| `pib_pro.layer3_cognition` | Dialogue turns, subgoal plans |
| `pib_pro.telemetry_gateway` | Runtime telemetry snapshots |
| `pib_pro.diagnostics` | Thor power/thermal/clock probe CLI |

Standard REP-103 frames include `base_link`, `camera_oak4_optical_frame`, `neck_link`, `hand_left_link`, and `hand_right_link` (X forward, Y left, Z up; radians; right-handed).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the Thor hardware probe (works on any Linux host; reads sysfs when present):

```bash
pib-pro-thor-probe
# or
python -m pib_pro.diagnostics.thor_probe
```

Prepare a Jetson AGX Thor host (idempotent; use `--dry-run` on CI or laptops):

```bash
sudo bash setup/setup-thor.sh
bash setup/setup-thor.sh --dry-run
```

Build the JetPack 7 Thor container:

```bash
docker build -f docker/Dockerfile.thor -t pib-pro:thor .
docker compose -f docker/docker-compose.thor.yml up --build
```

Model weights and Hugging Face caches default to `/data/models` and `/data/huggingface` on device.

## Tests and CI

```bash
ruff check .
mypy src
pytest -v
bash setup/setup-thor.sh --dry-run
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same checks on `push` to `main` and on all pull requests.

## License

See [LICENSE](LICENSE).
