# Use NVIDIA PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from setup.py
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install -v -e .

# Copy the rest of the code
COPY . .

# Set environment variables
ENV SAM2_BUILD_CUDA=1
ENV SAM2_BUILD_ALLOW_ERRORS=1

# Build CUDA extensions
RUN python setup.py build_ext --inplace
