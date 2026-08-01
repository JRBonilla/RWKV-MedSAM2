#!/usr/bin/env bash
set -Eeuo pipefail

readonly SERVICE_NAME="rwkv-container-start"
readonly JUPYTER_SESSION_NAME="jupyter"
readonly DEFAULT_JUPYTER_ROOT="/RWKV-MedSAM2"
readonly DEFAULT_EXPECTED_GPU_COUNT="2"
readonly DEFAULT_MIN_SHM_GB="12"

log() {
    printf '[%s] %s\n' "${SERVICE_NAME}" "$*"
}

fail() {
    log "ERROR: $*" >&2
    exit 1
}

require_unsigned_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a nonnegative integer; got '${value}'."
}

check_persistent_mount() {
    local path="$1"

    mountpoint -q "${path}" || fail "${path} is not a mounted persistent volume."
    [[ -w "${path}" ]] || fail "${path} is mounted but is not writable."
}

check_gpu_count() {
    local expected="$1"
    local actual

    require_unsigned_integer "RWKV_EXPECTED_GPU_COUNT" "${expected}"
    (( expected > 0 )) || fail "RWKV_EXPECTED_GPU_COUNT must be greater than zero."

    actual="$(python -c 'import torch; print(torch.cuda.device_count())')" \
        || fail "PyTorch could not query the available GPUs."
    require_unsigned_integer "detected GPU count" "${actual}"
    [[ "${actual}" == "${expected}" ]] \
        || fail "Expected ${expected} visible GPUs, but PyTorch detected ${actual}."

    log "GPU check passed: ${actual} GPUs are visible to PyTorch."
}

check_shared_memory() {
    local minimum_gb="$1"
    local available_kib
    local minimum_kib

    require_unsigned_integer "RWKV_MIN_SHM_GB" "${minimum_gb}"
    available_kib="$(df -Pk /dev/shm | awk 'NR == 2 {print $4}')"
    require_unsigned_integer "available /dev/shm size" "${available_kib}"
    minimum_kib=$((minimum_gb * 1024 * 1024))

    (( available_kib >= minimum_kib )) \
        || fail "/dev/shm has less than ${minimum_gb} GiB available. Check Compose shm_size."

    log "Shared-memory check passed: at least ${minimum_gb} GiB is available."
}

prepare_runtime_directories() {
    local path

    for path in \
        /run/sshd \
        "${JUPYTER_CONFIG_DIR:-/var/tmp/jupyter_config}" \
        "${JUPYTER_DATA_DIR:-/var/tmp/jupyter_data}" \
        "${JUPYTER_RUNTIME_DIR:-/var/tmp/jupyter_runtime}" \
        "${IPYTHONDIR:-/var/tmp/ipython}" \
        "${XDG_CACHE_HOME:-/var/tmp/cache}" \
        "${MPLCONFIGDIR:-/var/tmp/matplotlib}"
    do
        mkdir -p "${path}"
    done

    chmod 700 "${JUPYTER_RUNTIME_DIR:-/var/tmp/jupyter_runtime}"
}

prepare_authorized_keys() {
    local authorized_keys="/root/.ssh/authorized_keys"

    mkdir -p /root/.ssh
    chmod 700 /root/.ssh

    if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
        touch "${authorized_keys}"
        if ! grep -Fqx -- "${SSH_PUBLIC_KEY}" "${authorized_keys}"; then
            printf '%s\n' "${SSH_PUBLIC_KEY}" >> "${authorized_keys}"
            log "Added SSH_PUBLIC_KEY to ${authorized_keys}."
        fi
    fi

    if [[ -s "${authorized_keys}" ]]; then
        chmod 600 "${authorized_keys}"
    else
        log "WARNING: ${authorized_keys} is empty; SSH login will be unavailable until a public key is added."
    fi
}

generate_jupyter_token() {
    if [[ -n "${JUPYTER_TOKEN:-}" ]]; then
        printf '%s' "${JUPYTER_TOKEN}"
        return
    fi

    python -c 'import secrets; print(secrets.token_urlsafe(32), end="")'
}

stop_jupyter_session() {
    local attempt

    command -v tmux >/dev/null 2>&1 || return 0
    tmux has-session -t "${JUPYTER_SESSION_NAME}" 2>/dev/null || return 0

    # Give Jupyter a chance to shut down cleanly before terminating the session.
    tmux send-keys -t "${JUPYTER_SESSION_NAME}" C-c 2>/dev/null || true
    for attempt in {1..20}; do
        tmux has-session -t "${JUPYTER_SESSION_NAME}" 2>/dev/null || return 0
        sleep 0.25
    done
    tmux kill-session -t "${JUPYTER_SESSION_NAME}" 2>/dev/null || true
}

stop_services() {
    trap - EXIT INT TERM
    stop_jupyter_session
    if [[ -n "${sshd_pid:-}" ]] && kill -0 "${sshd_pid}" 2>/dev/null; then
        kill -TERM "${sshd_pid}" 2>/dev/null || true
    fi
    wait "${sshd_pid:-}" 2>/dev/null || true
}

handle_stop() {
    log "Stopping SSH and Jupyter."
    stop_services
    exit 0
}

main() {
    local expected_gpu_count="${RWKV_EXPECTED_GPU_COUNT:-${DEFAULT_EXPECTED_GPU_COUNT}}"
    local minimum_shm_gb="${RWKV_MIN_SHM_GB:-${DEFAULT_MIN_SHM_GB}}"
    local jupyter_root="${JUPYTER_ROOT_DIR:-${DEFAULT_JUPYTER_ROOT}}"
    local jupyter_token
    local jupyter_command
    local jupyter_stop_reported=0
    local -a jupyter_args

    if (( $# > 0 )); then
        exec "$@"
    fi

    check_persistent_mount /data
    check_persistent_mount /root
    check_gpu_count "${expected_gpu_count}"
    check_shared_memory "${minimum_shm_gb}"
    prepare_runtime_directories
    prepare_authorized_keys

    command -v tmux >/dev/null 2>&1 || fail "tmux is not installed."
    [[ -d "${jupyter_root}" ]] || fail "Jupyter root directory does not exist: ${jupyter_root}"
    if tmux has-session -t "${JUPYTER_SESSION_NAME}" 2>/dev/null; then
        fail "tmux session '${JUPYTER_SESSION_NAME}' already exists."
    fi

    ssh-keygen -A
    /usr/sbin/sshd -t \
        -o PermitRootLogin=prohibit-password \
        -o PasswordAuthentication=no \
        -o KbdInteractiveAuthentication=no \
        -o PubkeyAuthentication=yes \
        -o AllowTcpForwarding=yes

    jupyter_token="$(generate_jupyter_token)"
    [[ -n "${jupyter_token}" ]] || fail "Could not obtain a Jupyter authentication token."

    trap handle_stop INT TERM
    trap stop_services EXIT

    log "Starting key-only SSH on container port 22."
    /usr/sbin/sshd -D -e \
        -o PermitRootLogin=prohibit-password \
        -o PasswordAuthentication=no \
        -o KbdInteractiveAuthentication=no \
        -o PubkeyAuthentication=yes \
        -o AllowTcpForwarding=yes &
    sshd_pid=$!

    if [[ -n "${JUPYTER_TOKEN:-}" ]]; then
        log "Starting JupyterLab on container port 8888 with the configured token."
    else
        log "Starting JupyterLab on container port 8888. Generated token: ${jupyter_token}"
    fi
    jupyter_args=(
        jupyter lab
        --ip=0.0.0.0
        --port=8888
        --no-browser
        --allow-root
        --ServerApp.port_retries=0
        --ServerApp.root_dir="${jupyter_root}"
        --IdentityProvider.token="${jupyter_token}"
    )
    printf -v jupyter_command '%q ' "${jupyter_args[@]}"
    tmux new-session -d -s "${JUPYTER_SESSION_NAME}" "exec ${jupyter_command}" \
        || fail "Could not start the '${JUPYTER_SESSION_NAME}' tmux session."
    tmux set-option -t "${JUPYTER_SESSION_NAME}" remain-on-exit off \
        || fail "Could not configure the '${JUPYTER_SESSION_NAME}' tmux session."

    log "JupyterLab is running in detached tmux session '${JUPYTER_SESSION_NAME}'."
    log "Attach with: tmux attach -t ${JUPYTER_SESSION_NAME}"
    log "Container services started successfully."

    while true; do
        if ! kill -0 "${sshd_pid}" 2>/dev/null; then
            log "ERROR: SSH stopped unexpectedly." >&2
            stop_services
            return 1
        fi
        if tmux has-session -t "${JUPYTER_SESSION_NAME}" 2>/dev/null; then
            jupyter_stop_reported=0
        elif (( jupyter_stop_reported == 0 )); then
            log "WARNING: Jupyter tmux session '${JUPYTER_SESSION_NAME}' stopped; the container and SSH will remain running." >&2
            jupyter_stop_reported=1
        fi
        sleep 2
    done
}

main "$@"
