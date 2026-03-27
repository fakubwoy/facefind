#!/usr/bin/env python3
"""
migrate_to_b2.py
================
Run this ONCE to copy all existing local Railway volume data into Backblaze B2.

Usage (from inside the Railway shell or locally with the same env vars):

    python scripts/migrate_to_b2.py

Or against a non-default data directory:

    DATA_DIR=/data python scripts/migrate_to_b2.py

What it copies
--------------
  /data/datasets/**          → B2: datasets/{dataset_id}/...
  /data/thumbs/**            → B2: thumbs/{dataset_id}/...
  /data/embeddings/**        → B2: embeddings/{dataset_id}/...
  /data/uploads/**           → B2: uploads/...
  /data/releases/facefind-selfhosted.zip → B2: releases/facefind-selfhosted.zip

Nothing is deleted from local disk — this is a safe copy-only operation.
You can re-run it; existing B2 objects are simply overwritten (idempotent).

Prerequisites
-------------
  pip install boto3
  export B2_KEY_ID=...
  export B2_APPLICATION_KEY=...
  export B2_BUCKET_NAME=...
  export B2_REGION=us-west-004   # or your bucket's region
"""

import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("migrate")

# Allow running from repo root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    import b2_storage as b2
except ImportError:
    # Try same directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import b2_storage as b2


DATA_DIR       = Path(os.environ.get("DATA_DIR", "/data"))
DATASETS_DIR   = DATA_DIR / "datasets"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
UPLOADS_DIR    = DATA_DIR / "uploads"
THUMBS_DIR     = DATA_DIR / "thumbs"
EXECUTABLE_PATH = Path(os.environ.get(
    "EXECUTABLE_PATH",
    str(DATA_DIR / "releases" / "facefind-selfhosted.zip")
))


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def upload_file_with_log(key: str, path: Path, counter: list, errors: list, label: str):
    try:
        size = path.stat().st_size
        b2.upload_file(key, path)
        counter[0] += 1
        log.info(f"  ✓ [{counter[0]}] {label} ({human(size)})")
    except Exception as e:
        errors.append(f"{path}: {e}")
        log.error(f"  ✗ {label}: {e}")


def migrate():
    if not b2.b2_configured():
        log.error(
            "B2 is not configured!\n"
            "Set B2_KEY_ID, B2_APPLICATION_KEY, and B2_BUCKET_NAME env vars."
        )
        sys.exit(1)

    # Warm up connection
    try:
        b2.get_b2_client()
    except Exception as e:
        log.error(f"Cannot connect to B2: {e}")
        sys.exit(1)

    log.info(f"Starting migration from {DATA_DIR} → B2 bucket '{b2.B2_BUCKET_NAME}'")
    t0 = time.time()
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    errors = []

    # ── 1. Dataset images ─────────────────────────────────────────────────────
    img_count = [0]
    if DATASETS_DIR.exists():
        dataset_dirs = [d for d in DATASETS_DIR.iterdir() if d.is_dir()]
        log.info(f"\n[1/5] Dataset images — {len(dataset_dirs)} dataset(s) found")
        for dataset_dir in dataset_dirs:
            dataset_id = dataset_dir.name
            images = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]
            log.info(f"  Dataset {dataset_id}: {len(images)} image(s)")
            for img_path in images:
                rel = str(img_path.relative_to(dataset_dir))
                key = b2.dataset_image_key(dataset_id, rel)
                upload_file_with_log(key, img_path, img_count, errors, rel)
    else:
        log.info(f"\n[1/5] Dataset images — {DATASETS_DIR} not found, skipping")

    # ── 2. Thumbnails ─────────────────────────────────────────────────────────
    thumb_count = [0]
    if THUMBS_DIR.exists():
        thumb_dirs = [d for d in THUMBS_DIR.iterdir() if d.is_dir()]
        log.info(f"\n[2/5] Thumbnails — {len(thumb_dirs)} dataset(s) found")
        for dataset_dir in thumb_dirs:
            dataset_id = dataset_dir.name
            thumbs = [p for p in dataset_dir.rglob("*") if p.is_file()]
            log.info(f"  Thumbs for {dataset_id}: {len(thumbs)}")
            for thumb_path in thumbs:
                rel = str(thumb_path.relative_to(dataset_dir))
                key = b2.thumb_key(dataset_id, rel)
                upload_file_with_log(key, thumb_path, thumb_count, errors, rel)
    else:
        log.info(f"\n[2/5] Thumbnails — {THUMBS_DIR} not found, skipping")

    # ── 3. Embeddings (npy / pkl / faiss) ─────────────────────────────────────
    emb_count = [0]
    if EMBEDDINGS_DIR.exists():
        emb_dirs = [d for d in EMBEDDINGS_DIR.iterdir() if d.is_dir()]
        log.info(f"\n[3/5] Embeddings — {len(emb_dirs)} dataset(s) found")
        for emb_dir in emb_dirs:
            dataset_id = emb_dir.name
            files = [f for f in emb_dir.iterdir() if f.is_file()]
            log.info(f"  Embeddings for {dataset_id}: {[f.name for f in files]}")
            for f in files:
                key = f"embeddings/{dataset_id}/{f.name}"
                upload_file_with_log(key, f, emb_count, errors, f.name)
    else:
        log.info(f"\n[3/5] Embeddings — {EMBEDDINGS_DIR} not found, skipping")

    # ── 4. Selfie uploads ─────────────────────────────────────────────────────
    selfie_count = [0]
    if UPLOADS_DIR.exists():
        selfies = [f for f in UPLOADS_DIR.iterdir() if f.is_file()]
        log.info(f"\n[4/5] Selfie uploads — {len(selfies)} file(s)")
        for f in selfies:
            key = b2.selfie_key(f.name)
            upload_file_with_log(key, f, selfie_count, errors, f.name)
    else:
        log.info(f"\n[4/5] Selfie uploads — {UPLOADS_DIR} not found, skipping")

    # ── 5. Self-hosted executable ─────────────────────────────────────────────
    exe_count = [0]
    log.info(f"\n[5/5] Self-hosted executable — {EXECUTABLE_PATH}")
    if EXECUTABLE_PATH.exists():
        upload_file_with_log(b2.RELEASE_KEY, EXECUTABLE_PATH, exe_count, errors, "facefind-selfhosted.zip")
    else:
        log.info(f"  Not found at {EXECUTABLE_PATH}, skipping")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info(f"""
╔══════════════════════════════════════════════════╗
║            Migration Complete                   ║
╠══════════════════════════════════════════════════╣
║  Dataset images : {img_count[0]:>6}                       ║
║  Thumbnails     : {thumb_count[0]:>6}                       ║
║  Embedding files: {emb_count[0]:>6}                       ║
║  Selfie uploads : {selfie_count[0]:>6}                       ║
║  Executable     : {'yes' if exe_count[0] else 'no ':>6}                       ║
║  Errors         : {len(errors):>6}                       ║
║  Time           : {elapsed:>5.1f}s                      ║
╚══════════════════════════════════════════════════╝
""")

    if errors:
        log.warning(f"\n{len(errors)} error(s):")
        for e in errors:
            log.warning(f"  • {e}")
        sys.exit(1)
    else:
        log.info("All done — no errors. Railway volume data safely mirrored to B2.")


if __name__ == "__main__":
    migrate()