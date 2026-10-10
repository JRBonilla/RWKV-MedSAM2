# Build context is deploy/gamma, not the dataset or checkpoint directory.
# Driver 535 supports CUDA 11.8 by backward compatibility.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONPATH=/workspace:/workspace/ext:/workspace/dripp \
    TORCH_CUDA_ARCH_LIST=6.1 \
    VRWKV_T_MAX=256 \
    MAX_JOBS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    HOME=/tmp \
    XDG_CACHE_HOME=/workspace/artifacts/gamma/cache \
    MPLCONFIGDIR=/workspace/artifacts/gamma/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev build-essential \
        ca-certificates libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv
COPY requirements.txt /opt/gamma-requirements.txt
RUN python -m pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install -r /opt/gamma-requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/gamma-pip-freeze.txt

WORKDIR /workspace
# Source and outputs are bind-mounted. No GPU access or extension build at build time.
CMD ["bash", "-c", "source deploy/gamma/env.sh && cp /opt/gamma-pip-freeze.txt artifacts/gamma/pip-freeze.txt && python deploy/gamma/preflight.py --output artifacts/gamma/environment.json"]
