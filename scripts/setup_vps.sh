#!/usr/bin/env bash
# BaryGraph VPS setup — GPU pipeline edition.
#
# Run via SSH from your local machine:
#   ssh user@vps 'bash -s' < scripts/setup_vps.sh
#
# Override defaults via env vars:
#   DATA_DISK   block device for the 200 GB data disk (default: /dev/sdb)
#   DATA_MOUNT  mount point                           (default: /mnt/data)
#   REPO_URL    git remote to clone from
#   PROJECT_DIR local path for the project
#   OLLAMA_MODEL embedding model to pull
#
# Idempotent: safe to re-run after partial failure or upgrades.
# Tested on Ubuntu 22.04 LTS with NVIDIA T4 / L4.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/oleksiy-perepelytsya/barygraph-kaikki.git}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/barygraph-kaikki}"
OLLAMA_MODEL="${OLLAMA_MODEL:-nomic-embed-text:v1.5}"
DATA_DISK="${DATA_DISK:-/dev/sdb}"
DATA_MOUNT="${DATA_MOUNT:-/mnt/data}"

# ── colours ──────────────────────────────────────────────────────────────────
_info()  { echo -e "\033[1;34m[setup]\033[0m $*"; }
_ok()    { echo -e "\033[1;32m[  ok ]\033[0m $*"; }
_warn()  { echo -e "\033[1;33m[ warn]\033[0m $*"; }
_die()   { echo -e "\033[1;31m[ err ]\033[0m $*" >&2; exit 1; }

# ── 1. System packages ───────────────────────────────────────────────────────
_info "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    curl git ca-certificates gnupg lsb-release \
    python3.11 python3.11-venv python3-pip \
    tmux htop

# ── 2. Data disk (ext4, mounted, Docker data-root) ───────────────────────────
# All Docker volumes (MongoDB data, Ollama models) live on this disk so it
# can be detached and re-attached to the CPU VPS after the pipeline completes.
if [[ -b "$DATA_DISK" ]]; then
    _info "Data disk: $DATA_DISK → $DATA_MOUNT"

    # Format only if blank (no filesystem signature)
    if ! blkid "$DATA_DISK" &>/dev/null; then
        _info "Formatting $DATA_DISK as ext4"
        sudo mkfs.ext4 -F -L barygraph-data "$DATA_DISK"
    else
        _ok "$DATA_DISK already formatted ($(blkid -s TYPE -o value "$DATA_DISK"))"
    fi

    # Mount
    sudo mkdir -p "$DATA_MOUNT"
    if ! mountpoint -q "$DATA_MOUNT"; then
        sudo mount "$DATA_DISK" "$DATA_MOUNT"
        _ok "Mounted $DATA_DISK at $DATA_MOUNT"
    else
        _ok "$DATA_MOUNT already mounted"
    fi

    # Persist across reboots
    UUID=$(sudo blkid -s UUID -o value "$DATA_DISK")
    if ! grep -q "$UUID" /etc/fstab; then
        echo "UUID=$UUID $DATA_MOUNT ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
        _ok "Added to /etc/fstab"
    fi

    # Point Docker data-root here so all volumes land on this disk.
    # Must happen before Docker pulls any images.
    DOCKER_ROOT="$DATA_MOUNT/docker"
    sudo mkdir -p "$DOCKER_ROOT"
    DAEMON_JSON=/etc/docker/daemon.json
    if ! sudo test -f "$DAEMON_JSON" || ! sudo grep -q "data-root" "$DAEMON_JSON"; then
        echo "{\"data-root\": \"$DOCKER_ROOT\"}" | sudo tee "$DAEMON_JSON"
        _ok "Docker data-root → $DOCKER_ROOT"
    else
        _ok "Docker data-root already configured"
    fi
    sudo chown -R root:root "$DOCKER_ROOT"
else
    _warn "Data disk $DATA_DISK not found — using boot disk (not recommended)."
    _warn "Pass DATA_DISK=/dev/sdX if your disk has a different name."
fi

# ── 3. Docker ────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    _info "Installing Docker"
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    _warn "Added $USER to docker group — you may need to run: newgrp docker"
else
    _ok "Docker $(docker --version | awk '{print $3}' | tr -d ',')"
fi

# (Re)start Docker to pick up any data-root or runtime changes
sudo systemctl restart docker
sleep 2

# ── 4. NVIDIA container toolkit (skip if no GPU detected) ────────────────────
if nvidia-smi &>/dev/null; then
    _info "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if ! dpkg -s nvidia-container-toolkit &>/dev/null; then
        _info "Installing nvidia-container-toolkit"
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ct-keyring.gpg
        curl -fsSL "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ct-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update -qq
        sudo apt-get install -y -qq nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        sleep 2
    fi
    _ok "nvidia-container-toolkit ready"
else
    _warn "No NVIDIA GPU — skipping container toolkit. CPU-only run."
fi

# ── 5. Project clone / update ────────────────────────────────────────────────
if [[ -d "$PROJECT_DIR/.git" ]]; then
    _info "Updating repo at $PROJECT_DIR"
    git -C "$PROJECT_DIR" pull --ff-only
else
    _info "Cloning $REPO_URL → $PROJECT_DIR"
    git clone "$REPO_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# ── 6. Python dependencies ───────────────────────────────────────────────────
_info "Python dependencies"
python3.11 -m pip install -q -e ".[dev,mcp]"
_ok "Python deps installed"

# ── 7. .env ──────────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    _info "Creating .env"
    if nvidia-smi &>/dev/null; then
        cat > .env <<'EOF'
# GPU pipeline settings
EMBED_BATCH_SIZE=256
EMBED_TIMEOUT_SECONDS=120
LOG_LEVEL=INFO
EOF
    else
        cat > .env <<'EOF'
# CPU pipeline settings
EMBED_BATCH_SIZE=16
EMBED_TIMEOUT_SECONDS=600
LOG_LEVEL=INFO
EOF
    fi
    _ok ".env written"
else
    _ok ".env already exists — not overwritten"
fi

# ── 8. Start services ────────────────────────────────────────────────────────
_info "Starting services"
if nvidia-smi &>/dev/null; then
    docker compose --profile gpu up -d
    _ok "MongoDB + Ollama (GPU) started"
else
    docker compose up -d mongodb
    _ok "MongoDB started (CPU-only mode)"
fi

# ── 9. Pull embedding model ──────────────────────────────────────────────────
_info "Pulling Ollama model: $OLLAMA_MODEL"
for i in $(seq 1 12); do
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then break; fi
    _info "Waiting for Ollama… ($((i*5))s)"
    sleep 5
done
curl -sf http://localhost:11434/api/tags &>/dev/null \
    || _die "Ollama did not start within 60s"
ollama pull "$OLLAMA_MODEL"
_ok "Model $OLLAMA_MODEL ready"

# ── 10. kaikki data ──────────────────────────────────────────────────────────
if [[ ! -f data/kaikki-en.jsonl ]] || \
   [[ $(stat -c%s data/kaikki-en.jsonl 2>/dev/null || echo 0) -lt $((100*1024*1024)) ]]; then
    _info "Downloading kaikki-en.jsonl (~2.5 GB, resumable)"
    make fetch-kaikki
else
    _ok "kaikki-en.jsonl already present"
fi

# ── 11. Preflight benchmark ───────────────────────────────────────────────────
_info "Running preflight benchmark"
make preflight

echo ""
_ok "═══════════════════════════════════════════════"
_ok "VPS ready. To start the pipeline:"
_ok "  tmux new -s pipeline"
_ok "  make pipeline"
_ok ""
_ok "After pipeline completes, to move to CPU VPS:"
_ok "  1. docker compose down"
_ok "  2. Detach $DATA_DISK in GCP console"
_ok "  3. Attach to CPU VPS as /dev/sdb"
_ok "  4. Re-run this script (skips formatting, mounts disk)"
_ok "  5. make up   ← MongoDB finds its data immediately"
_ok "═══════════════════════════════════════════════"
