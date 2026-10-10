# Source this file from the repository root before using the Gamma environment.
export TORCH_CUDA_ARCH_LIST=6.1
export VRWKV_T_MAX=256
export VRWKV_DISABLE_COMPILE=1
export TORCH_COMPILE_DISABLE=1
# A numeric host UID need not have an /etc/passwd entry in the container.
export USER="${USER:-gamma}"
export LOGNAME="${LOGNAME:-$USER}"
export TORCHINDUCTOR_CACHE_DIR="${PWD}/artifacts/gamma/inductor_cache"
export TRITON_CACHE_DIR="${PWD}/artifacts/gamma/triton_cache"
export MAX_JOBS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TORCH_EXTENSIONS_DIR="${PWD}/artifacts/gamma/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# These control extension compilation and CPU resources. They do NOT override
# BF16 autocast in the legacy evaluation runner or implement a new protocol.
