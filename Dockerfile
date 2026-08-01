# syntax=docker/dockerfile:1.7

# CUDA 12.8 matches the PyTorch 2.7.1+cu128 environment used for the
# completed preprocessing benchmarks and the upcoming training runs.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.10.0
ARG OPENCV_COMMIT=71d3237a093b60a27601c20e9ee6c3e52154e8b1
ARG OPENCV_CONTRIB_COMMIT=1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87
ARG OPENCV_BUILD_JOBS=8
ARG CUDA_ARCH_BIN=8.0

ENV TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    RWKV_NVIDIA_PYTHON_ROOT=/usr/local/lib/python3.10/dist-packages/nvidia \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_BIN} \
    TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
    XDG_CACHE_HOME=/var/tmp/cache \
    MPLCONFIGDIR=/var/tmp/matplotlib

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Build tools, SSH/Jupyter runtime support, NFS diagnostics, and the native
# libraries required by the CUDA-enabled OpenCV build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        g++-11 \
        gcc-11 \
        git \
        iproute2 \
        libavcodec-dev \
        libavdevice-dev \
        libavformat-dev \
        libavutil-dev \
        libdc1394-dev \
        libglib2.0-0 \
        libjpeg-dev \
        libopenexr-dev \
        libpng-dev \
        libsm6 \
        libswresample-dev \
        libswscale-dev \
        libtiff-dev \
        libv4l-dev \
        libwebp-dev \
        libx264-dev \
        libxext6 \
        libxvidcore-dev \
        lsof \
        nfs-common \
        ninja-build \
        openssh-client \
        openssh-server \
        pkg-config \
        procps \
        python-is-python3 \
        python3-dev \
        python3-pip \
        rsync \
        tmux \
        unzip \
        util-linux \
        wget \
        zip \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade \
        pip==25.2 \
        setuptools==80.9.0 \
        wheel==0.45.1 \
    && python -m pip install \
        numpy==1.24.4 \
    && python -m pip install \
        torch==2.7.1 \
        torchvision==0.22.1 \
        torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && test -f "${RWKV_NVIDIA_PYTHON_ROOT}/cudnn/lib/libcudnn.so.9" \
    && test ! -e "${RWKV_NVIDIA_PYTHON_ROOT}/cudnn/lib/libcudnn.so" \
    && ln -s libcudnn.so.9 "${RWKV_NVIDIA_PYTHON_ROOT}/cudnn/lib/libcudnn.so"

# Build OpenCV rather than installing opencv-python. This preserves CUDA
# support and prevents DRIPP's Python dependency from replacing the custom cv2.
# OpenCV 4.10 predates CUDA 12.8's C++17 default, so CUDA_NVCC_FLAGS sets the
# standard explicitly. cudacodec remains disabled because training uses FFmpeg
# videoio/decord and the NVIDIA Video Codec SDK is not part of this image.
COPY docker/verify_opencv.py /tmp/verify_opencv.py
RUN --mount=type=cache,target=/var/cache/opencv-downloads \
    git clone --branch "${OPENCV_VERSION}" --depth 1 \
        https://github.com/opencv/opencv.git /tmp/opencv \
    && git clone --branch "${OPENCV_VERSION}" --depth 1 \
        https://github.com/opencv/opencv_contrib.git /tmp/opencv_contrib \
    && test "$(git -C /tmp/opencv rev-parse HEAD)" = "${OPENCV_COMMIT}" \
    && test "$(git -C /tmp/opencv_contrib rev-parse HEAD)" = "${OPENCV_CONTRIB_COMMIT}" \
    && cmake -S /tmp/opencv -B /tmp/opencv-build -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_C_COMPILER=/usr/bin/gcc-11 \
        -D CMAKE_CXX_COMPILER=/usr/bin/g++-11 \
        -D CMAKE_CXX_STANDARD=17 \
        -D CMAKE_CUDA_ARCHITECTURES=80 \
        -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D BUILD_SHARED_LIBS=ON \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_JAVA=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_opencv_cudacodec=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_TESTS=OFF \
        -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
        -D CUDA_ARCH_PTX= \
        -D CUDA_FAST_MATH=ON \
        -D CUDA_HOST_COMPILER=/usr/bin/g++-11 \
        -D CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc \
        -D CUDA_NVCC_FLAGS=--std=c++17 \
        -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -D CUDNN_INCLUDE_DIR=${RWKV_NVIDIA_PYTHON_ROOT}/cudnn/include \
        -D CUDNN_LIBRARY=${RWKV_NVIDIA_PYTHON_ROOT}/cudnn/lib/libcudnn.so.9 \
        -D ENABLE_FAST_MATH=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D OPENCV_DOWNLOAD_PATH=/var/cache/opencv-downloads \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
        -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.10/dist-packages/numpy/core/include \
        -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.10/dist-packages \
        -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
        -D WITH_CUBLAS=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=OFF \
        -D WITH_GTK=OFF \
        -D WITH_OPENCL=ON \
        -D WITH_TBB=OFF \
    && cmake --build /tmp/opencv-build --parallel "${OPENCV_BUILD_JOBS}" \
    && cmake --install /tmp/opencv-build \
    && ldconfig \
    && python /tmp/verify_opencv.py \
    && rm -rf /tmp/opencv /tmp/opencv_contrib /tmp/opencv-build /tmp/verify_opencv.py

# Pin the packages used by preprocessing, loader construction, notebooks, and
# training. OpenCV is intentionally absent because it was built above.
RUN python -m pip install \
        cupy-cuda12x==13.5.1 \
        eva-decord==0.6.1 \
        gdown==5.2.0 \
        h5py==3.13.0 \
        hydra-core==1.3.2 \
        imageio==2.37.0 \
        imageio-ffmpeg==0.6.0 \
        iopath==0.1.10 \
        ipykernel==6.30.0 \
        ipywidgets==8.1.7 \
        jupyter==1.1.1 \
        jupyterlab==4.4.5 \
        matplotlib==3.10.3 \
        moviepy==1.0.3 \
        nibabel==5.3.2 \
        ninja==1.11.1.4 \
        notebook==7.4.5 \
        openpyxl==3.1.5 \
        pandas==2.2.3 \
        pillow==11.0.0 \
        psutil==7.0.0 \
        pyarrow==20.0.0 \
        pydicom==2.4.4 \
        scikit-learn==1.7.2 \
        scipy==1.15.3 \
        SimpleITK==2.5.0 \
        tciaclient==0.0.3 \
        tensorboard==2.20.0 \
        tqdm==4.67.1

RUN mkdir -p \
        /run/sshd \
        /opt/torch_extensions \
        /var/tmp/cache \
        /var/tmp/matplotlib \
        /var/tmp/jupyter_config \
        /var/tmp/jupyter_data \
        /var/tmp/jupyter_runtime \
        /var/tmp/ipython

WORKDIR /RWKV-MedSAM2
COPY . /RWKV-MedSAM2

# Install local code without dependency resolution so pip cannot replace the
# CUDA-enabled OpenCV build with the opencv-python wheel. Force the SAM2 CUDA
# extension to compile successfully; do not silently continue without it.
RUN python -m pip install --no-deps --no-build-isolation -e ./dripp \
    && SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 \
        python -m pip install --no-deps --no-build-isolation -e ./ext \
    && SAM2_BUILD_CUDA=0 \
        python -m pip install --no-deps --no-build-isolation -e .

# Importing RWKV-MedSAM2 compiles the WKV CUDA extension into /opt instead of
# /root, which will later be hidden by the persistent /root NFS mount.
RUN python -c 'from sam2 import _C; import rwkv_medsam2; print("SAM2 and WKV CUDA extensions loaded")' \
    && python docker/verify_opencv.py \
    && python docker/verify_python_env.py

COPY docker/startup.sh /usr/local/bin/rwkv-container-start
COPY docker/preflight.py /usr/local/bin/rwkv-preflight
RUN chmod 755 /usr/local/bin/rwkv-container-start /usr/local/bin/rwkv-preflight \
    && bash -n /usr/local/bin/rwkv-container-start \
    && python -c 'compile(open("/usr/local/bin/rwkv-preflight").read(), "/usr/local/bin/rwkv-preflight", "exec")'

EXPOSE 22 8888 6006
STOPSIGNAL SIGTERM

ENTRYPOINT ["/usr/local/bin/rwkv-container-start"]
