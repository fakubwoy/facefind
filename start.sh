#!/usr/bin/env bash
# start.sh — Launch the FaceFind server
# Usage: bash start.sh [--port 8000]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-8000}

# Parse --port flag
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift ;;
  esac
  shift
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║        FaceFind — Starting Server            ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "✗ python3 not found. Please install Python 3.10+."
  exit 1
fi

# Check / create venv
if [ ! -d "$SCRIPT_DIR/venv" ]; then
  echo "→ Creating virtual environment..."
  python3 -m venv "$SCRIPT_DIR/venv"
fi

source "$SCRIPT_DIR/venv/bin/activate"

# ── Dependency check ──────────────────────────────────────────────────────────
# Uses pip's own resolver to compare installed packages against requirements.txt.
# Reinstalls if any package is missing, wrong version, or requirements.txt has
# changed since the last install.

REQS="$SCRIPT_DIR/requirements.txt"
STAMP="$SCRIPT_DIR/venv/.deps_installed"

needs_install() {
  # 1. No stamp file → never installed cleanly
  [ ! -f "$STAMP" ] && return 0
  # 2. requirements.txt is newer than the stamp → file changed since last install
  [ "$REQS" -nt "$STAMP" ] && return 0
  # 3. Ask pip whether anything is missing or needs upgrading (dry-run, no network noise)
  PENDING=$(pip install --dry-run --quiet -r "$REQS" 2>&1 | grep -c "Would install" || true)
  [ "$PENDING" -gt 0 ] && return 0
  # 4. Detect broken / conflicting packages already installed
  if ! pip check --quiet 2>/dev/null; then return 0; fi
  return 1
}

if needs_install; then
  echo "→ Installing / updating dependencies..."
  pip install --quiet --upgrade pip
  pip install --quiet -r "$REQS"
  # Write stamp with current timestamp so we skip this next run
  date -u > "$STAMP"
  echo "✓ Dependencies installed"
else
  echo "✓ All dependencies up to date"
fi

echo ""
echo "  Admin UI   →  http://localhost:$PORT/"
echo "  API Docs   →  http://localhost:$PORT/docs"
echo ""
echo "  To download LFW dataset first:"
echo "  bash scripts/download_lfw.sh"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

cd "$SCRIPT_DIR/backend"
uvicorn app:app --host 0.0.0.0 --port "$PORT" --reload