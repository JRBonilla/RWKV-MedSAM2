# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.10\
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python and create a python alias
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && ln -s /usr/bin/python3 /usr/bin/python

# Verify Python 3.10 installation
RUN python3 --version

# Clone the repository
RUN git clone https://github.com/JRBonilla/RWKV-MedSAM2.git

# Set working directory
WORKDIR /app/RWKV-MedSAM2

# Install PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
COPY setup.py .

COPY README.md .

# Install the package
RUN pip3 install --no-cache-dir -v -e .

# Copy the rest of the code
COPY . .

# Set environment variables
ENV SAM2_BUILD_CUDA=1
ENV SAM2_BUILD_ALLOW_ERRORS=1

# Build CUDA extensions
RUN python setup.py build_ext --inplace

# Copy the entrypoint script
COPY scripts/entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]