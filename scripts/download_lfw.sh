#!/usr/bin/env bash
# scripts/download_lfw.sh
# Downloads and extracts the LFW Funneled dataset into backend/datasets/lfw/
# Usage: bash scripts/download_lfw.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEST="$PROJECT_DIR/backend/datasets/lfw"

echo "╔══════════════════════════════════════════════╗"
echo "║  FaceFind — LFW Dataset Downloader           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

if [ -d "$DEST" ] && [ "$(ls -A "$DEST")" ]; then
  echo "✓ LFW dataset already exists at $DEST"
  echo "  Delete the folder and re-run to re-download."
  exit 0
fi

mkdir -p "$DEST"

LFW_URL="http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
TMP_FILE="/tmp/lfw-funneled.tgz"

echo "→ Downloading LFW Funneled (~200 MB)..."
echo "  Source: $LFW_URL"
echo ""

if command -v curl &>/dev/null; then
  curl -L --progress-bar -o "$TMP_FILE" "$LFW_URL"
elif command -v wget &>/dev/null; then
  wget -q --show-progress -O "$TMP_FILE" "$LFW_URL"
else
  echo "Error: curl or wget required."
  exit 1
fi

echo ""
echo "→ Extracting..."
tar -xzf "$TMP_FILE" -C "$DEST" --strip-components=1
rm "$TMP_FILE"

COUNT=$(find "$DEST" -name "*.jpg" | wc -l)
echo ""
echo "✓ Done! $COUNT images extracted to:"
echo "  $DEST"
echo ""
echo "→ Next: start the server and click 'Index LFW Dataset'"
echo "  OR run: curl -X POST http://localhost:8000/api/datasets/lfw"