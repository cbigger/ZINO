#!/bin/bash
# install.sh — install and configure the ZINO agent platform.
# Run after bootstrap.sh.  Idempotent — safe to re-run for upgrades.
# Run as root:  sudo bash install.sh

set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────────────

INSTALL_DIR="/opt/zino"
CONFIG_DIR="/etc/zino"
DATA_DIR="/var/lib/zino"
RUN_DIR="/run/zino"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_USER="zino"
SERVICE_GROUP="zino"

# ── Preflight ────────────────────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
    echo "Error: install.sh must be run as root." >&2
    exit 1
fi

echo "=== ZINO Installation ==="
echo "Source:  ${SRC_DIR}"
echo "Install: ${INSTALL_DIR}"
echo ""

# ── 1. Service user ─────────────────────────────────────────────────────────

if ! id "${SERVICE_USER}" &>/dev/null; then
    echo "[1/10] Creating service user '${SERVICE_USER}'..."
    useradd --system \
            --no-create-home \
            --home-dir "${INSTALL_DIR}" \
            --shell /usr/sbin/nologin \
            "${SERVICE_USER}"
else
    echo "[1/10] Service user '${SERVICE_USER}' already exists."
fi

# ── 2. Directory structure ───────────────────────────────────────────────────

echo "[2/10] Creating directories..."
mkdir -p "${INSTALL_DIR}"
mkdir -p "${CONFIG_DIR}"
mkdir -p "${DATA_DIR}/mem/channels"
mkdir -p "${RUN_DIR}"

# ── 3. Application files ────────────────────────────────────────────────────

echo "[3/10] Syncing application files to ${INSTALL_DIR}..."
rsync -a "${SRC_DIR}/"*.py          "${INSTALL_DIR}/"
rsync -a "${SRC_DIR}/SYSTEM.md"     "${INSTALL_DIR}/"
rsync -a "${SRC_DIR}/requirements.txt" "${INSTALL_DIR}/"
rsync -a --delete "${SRC_DIR}/tools/"  "${INSTALL_DIR}/tools/"
rsync -a --delete "${SRC_DIR}/skills/" "${INSTALL_DIR}/skills/"

# ── 4. Configuration ────────────────────────────────────────────────────────

if [[ ! -f "${CONFIG_DIR}/ZINO.toml" ]]; then
    echo "[4/10] Installing default configuration..."
    cp "${SRC_DIR}/ZINO.toml" "${CONFIG_DIR}/ZINO.toml"

    # Rewrite paths for installed layout
    sed -i 's|^data_dir *= *"data/mem"|data_dir = "/var/lib/zino/mem"|' \
        "${CONFIG_DIR}/ZINO.toml"
else
    echo "[4/10] Configuration exists at ${CONFIG_DIR}/ZINO.toml — not overwriting."
    echo "       To reset: rm ${CONFIG_DIR}/ZINO.toml  and re-run install.sh"
fi

# ── 5. Environment file (API key) ───────────────────────────────────────────

if [[ ! -f "${CONFIG_DIR}/env" ]]; then
    echo "[5/10] Creating environment file..."
    cat > "${CONFIG_DIR}/env" <<'ENVEOF'
# /etc/zino/env — sourced by all ZINO systemd services.
# Set your LLM API key here.
# LLM_API_KEY=sk-your-key-here
ENVEOF
else
    echo "[5/10] Environment file exists — not overwriting."
fi

# ── 6. Python virtual environment ───────────────────────────────────────────

echo "[6/10] Setting up Python virtual environment..."
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --upgrade pip --quiet
"${INSTALL_DIR}/venv/bin/pip" install -r "${SRC_DIR}/requirements.txt" --quiet
echo "       $(${INSTALL_DIR}/venv/bin/python3 --version)"
echo "       Packages installed:"
"${INSTALL_DIR}/venv/bin/pip" list --format=columns 2>/dev/null | grep -iE 'openai|dotenv' || true

# ── 7. Permissions ───────────────────────────────────────────────────────────

echo "[7/10] Setting ownership and permissions..."

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_DIR}"
chmod 750 "${INSTALL_DIR}"

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${DATA_DIR}"
chmod 750 "${DATA_DIR}"

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${RUN_DIR}"
chmod 750 "${RUN_DIR}"

# Config readable by service, owned by root
chown root:"${SERVICE_GROUP}" "${CONFIG_DIR}"
chmod 750 "${CONFIG_DIR}"
chown root:"${SERVICE_GROUP}" "${CONFIG_DIR}/ZINO.toml"
chmod 640 "${CONFIG_DIR}/ZINO.toml"
chown root:"${SERVICE_GROUP}" "${CONFIG_DIR}/env"
chmod 640 "${CONFIG_DIR}/env"

# ── 8. tmpfiles.d (runtime directory across reboots) ────────────────────────

echo "[8/10] Installing tmpfiles.d configuration..."
cp "${SRC_DIR}/systemd/zino-tmpfiles.conf" /usr/lib/tmpfiles.d/zino.conf
systemd-tmpfiles --create

# ── 9. systemd units ────────────────────────────────────────────────────────

echo "[9/10] Installing systemd units..."
cp "${SRC_DIR}/systemd/zino.target"          /etc/systemd/system/
cp "${SRC_DIR}/systemd/zino-rtr.service"     /etc/systemd/system/
cp "${SRC_DIR}/systemd/zino-sys.service"     /etc/systemd/system/
cp "${SRC_DIR}/systemd/zino-exc.service"     /etc/systemd/system/
cp "${SRC_DIR}/systemd/zino-mem.service"     /etc/systemd/system/
cp "${SRC_DIR}/systemd/zino-ctx.service"     /etc/systemd/system/
cp "${SRC_DIR}/systemd/zino-agr.service"     /etc/systemd/system/
cp "${SRC_DIR}/systemd/zino-daemon.service"  /etc/systemd/system/
systemctl daemon-reload

# ── 10. Enable services ─────────────────────────────────────────────────────

echo "[10/10] Enabling ZINO services..."
systemctl enable --quiet zino.target
for svc in zino-rtr zino-sys zino-exc zino-mem zino-ctx zino-agr zino-daemon; do
    systemctl enable --quiet "${svc}.service"
done

# ── CLI wrapper ──────────────────────────────────────────────────────────────

cat > /usr/local/bin/zino-cli <<'CLIEOF'
#!/bin/bash
exec /opt/zino/venv/bin/python3 /opt/zino/zino-cli.py \
    --config /etc/zino/ZINO.toml "$@"
CLIEOF
chmod 755 /usr/local/bin/zino-cli

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "=== Installation complete ==="
echo ""
echo "Before starting ZINO, set your API key:"
echo "  sudo editor /etc/zino/env"
echo "  # uncomment and fill in LLM_API_KEY=sk-..."
echo ""
echo "Start all services:"
echo "  sudo systemctl start zino.target"
echo ""
echo "Check status:"
echo "  systemctl status 'zino-*'"
echo ""
echo "Quick test:"
echo "  zino-cli \"Hello, what can you do?\""
echo "  zino-cli --stream \"List the files in /tmp\""
