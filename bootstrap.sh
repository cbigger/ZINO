#!/bin/bash
# bootstrap.sh — install system-level dependencies for ZINO.
# Intended for Debian 13 (Trixie) fresh installs or disposable VMs.
# Run as root:  sudo bash bootstrap.sh

set -euo pipefail

# ── Preflight ────────────────────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
    echo "Error: bootstrap.sh must be run as root." >&2
    exit 1
fi

echo "=== ZINO Bootstrap ==="
echo "Target: Debian 13 (Trixie)"
echo ""

# ── System packages ──────────────────────────────────────────────────────────

echo "Updating package index..."
apt-get update -qq

echo "Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    rsync \
    shellcheck

# ── Python version check ────────────────────────────────────────────────────

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "Python ${PYTHON_VERSION} — OK (>= 3.11 required)."
else
    echo "Error: Python ${PYTHON_VERSION} found but >= 3.11 is required." >&2
    exit 1
fi

# ── Verify key binaries ─────────────────────────────────────────────────────

for cmd in python3 pip3 rsync shellcheck; do
    if command -v "$cmd" &>/dev/null; then
        echo "  $cmd — $(command -v "$cmd")"
    else
        echo "  $cmd — NOT FOUND (unexpected)" >&2
    fi
done

echo ""
echo "=== Bootstrap complete ==="
echo "Next: run  sudo bash install.sh"
