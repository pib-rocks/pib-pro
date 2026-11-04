# JetPack 7 / TensorRT NVFP4 runtime for pib.Pro on Jetson AGX Thor
# Build: docker build -f docker/Dockerfile.thor -t pib-pro:thor .

ARG L4T_BASE=nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3
FROM ${L4T_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIB_MODEL_CACHE=/data/models \
    HF_HOME=/data/huggingface \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ethtool \
    linuxptp \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pib-pro
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip3 install --no-cache-dir -e .

RUN mkdir -p /data/models /data/huggingface
VOLUME ["/data/models", "/data/huggingface"]

# 30 Hz unified-memory bus + 250 Hz Layer-0 expect generous shared memory
ENV SHM_SIZE=8g

CMD ["python3", "-m", "pib_pro.diagnostics.thor_probe"]
