"""
FaceFind – Backend API (Railway Edition)
- Postgres for metadata (datasets, shares, license keys, download tokens)
- Redis for caching dataset status + search results
- Local filesystem volume for images + FAISS indexes
- Gmail API for email OTP verification (works on Railway)
"""

import os, io, re, uuid, time, json, pickle, zipfile, threading, hashlib, secrets, base64
import smtplib, random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
import urllib.request, urllib.parse, logging
from pathlib import Path
from typing import Optional

import sqlite3
import numpy as np
import cv2
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("facefind")

# ── Storage root (Railway Volume mounted at /data, falls back to local) ───────
DATA_DIR       = Path(os.environ.get("DATA_DIR", "/data"))
DATASETS_DIR   = DATA_DIR / "datasets"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
UPLOADS_DIR    = DATA_DIR / "uploads"
THUMBS_DIR     = DATA_DIR / "thumbs"   # persistent thumbnail cache
FRONTEND_DIR   = Path(__file__).parent.parent / "frontend"

# Path to the self-hosted executable ZIP served for download
EXECUTABLE_PATH = Path(os.environ.get("EXECUTABLE_PATH", "/data/releases/facefind-selfhosted.zip"))

for d in [DATASETS_DIR, EMBEDDINGS_DIR, UPLOADS_DIR, THUMBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── SQLite (self-hosted) ─────────────────────────────────────────────────────
_DB_PATH = DATA_DIR / "facefind.db" if "DATA_DIR" in os.environ else Path("facefind.db")

class _RealDictRow(dict):
    """Make sqlite3 rows behave like psycopg2 RealDictRow."""
    pass

class _SQLiteConn:
    """Thin wrapper so existing `with get_db() as conn:` code works unchanged."""
    def __init__(self, path):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = lambda cur, row: _RealDictRow(
            zip([d[0] for d in cur.description], row)
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def cursor(self):
        return _CursorWrapper(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

class _CursorWrapper:
    """Translate %s placeholders -> ? and expose fetchone/fetchall/rowcount."""
    def __init__(self, cur):
        self._cur = cur

    @property
    def rowcount(self):
        return self._cur.rowcount

    @property
    def description(self):
        return self._cur.description

    def execute(self, sql, params=()):
        sql = sql.replace("%s", "?")
        # Translate ON CONFLICT (col) DO UPDATE SET ... EXCLUDED.x -> excluded.x (sqlite syntax)
        self._cur.execute(sql, params)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

_db_lock = threading.Lock()

def get_db():
    global _DB_PATH
    _DB_PATH = DATA_DIR / "facefind.db"
    return _SQLiteConn(_DB_PATH)

def _sqlite_add_column_if_missing(conn, table, column, col_def):
    """SQLite does not support ADD COLUMN IF NOT EXISTS, so we check first."""
    with conn.cursor() as cur:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row["name"] for row in cur.fetchall()]
    if column not in cols:
        with conn.cursor() as cur:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
        conn.commit()

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id              TEXT PRIMARY KEY,
                    email           TEXT UNIQUE NOT NULL,
                    name            TEXT NOT NULL,
                    password_hash   TEXT NOT NULL,
                    email_verified  INTEGER DEFAULT 0,
                    created_at      REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS otp_codes (
                    id          TEXT PRIMARY KEY,
                    email       TEXT NOT NULL,
                    code        TEXT NOT NULL,
                    purpose     TEXT NOT NULL,
                    expires_at  REAL,
                    used        INTEGER DEFAULT 0
                )
            """)
            cur.execute("""
                UPDATE users SET email_verified = 1 WHERE email_verified = 0
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token       TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    created_at  REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL DEFAULT '',
                    name        TEXT NOT NULL,
                    source      TEXT DEFAULT 'zip',
                    folder_id   TEXT,
                    status      TEXT DEFAULT 'queued',
                    total       INTEGER DEFAULT 0,
                    processed   INTEGER DEFAULT 0,
                    face_count  INTEGER DEFAULT 0,
                    error       TEXT,
                    created_at  REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shares (
                    share_id     TEXT PRIMARY KEY,
                    dataset_id   TEXT NOT NULL,
                    dataset_name TEXT,
                    created_at   REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS license_keys (
                    key              TEXT PRIMARY KEY,
                    user_id          TEXT NOT NULL,
                    plan             TEXT NOT NULL,
                    created_at       REAL,
                    expires_at       REAL,
                    revoked          INTEGER DEFAULT 0,
                    activations      INTEGER DEFAULT 0,
                    max_activations  INTEGER DEFAULT 3,
                    last_seen_at     REAL,
                    last_seen_ip     TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS download_tokens (
                    token       TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    created_at  REAL,
                    expires_at  REAL,
                    used        INTEGER DEFAULT 0
                )
            """)
        conn.commit()
    # Add columns that may be missing in older DBs
    with get_db() as conn:
        _sqlite_add_column_if_missing(conn, "users", "email_verified", "INTEGER DEFAULT 0")
        _sqlite_add_column_if_missing(conn, "users", "plan", "TEXT DEFAULT 'free'")
        _sqlite_add_column_if_missing(conn, "datasets", "user_id", "TEXT NOT NULL DEFAULT ''")
    log.info("DB tables ready")

# ── Cache (no-op for self-hosted — Redis not needed locally) ─────────────────
def get_redis():
    return None

def cache_get(key: str):
    return None

def cache_set(key: str, value, ttl: int = 30):
    pass

def cache_delete(key: str):
    pass

# ── DB helpers ────────────────────────────────────────────────────────────────

def db_get_dataset(dataset_id: str) -> Optional[dict]:
    cached = cache_get(f"dataset:{dataset_id}")
    if cached:
        return cached
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM datasets WHERE id = %s", (dataset_id,))
            row = cur.fetchone()
    if row:
        result = dict(row)
        cache_set(f"dataset:{dataset_id}", result, ttl=10)
        return result
    return None

def get_dataset_disk_size(dataset_id: str) -> int:
    """Return total bytes of all files in the dataset directory."""
    dataset_dir = DATASETS_DIR / dataset_id
    if not dataset_dir.exists():
        return 0
    return sum(p.stat().st_size for p in dataset_dir.rglob("*") if p.is_file())

def db_list_datasets(user_id: str) -> dict:
    cached = cache_get(f"datasets:{user_id}")
    if cached:
        return cached
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM datasets WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
            rows = cur.fetchall()
    result = {row["id"]: dict(row) for row in rows}
    cache_set(f"datasets:{user_id}", result, ttl=5)
    return result

def db_upsert_dataset(ds: dict):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO datasets (id, user_id, name, source, folder_id, status, total, processed, face_count, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, source=excluded.source, folder_id=excluded.folder_id,
                    status=excluded.status, total=excluded.total, processed=excluded.processed,
                    face_count=excluded.face_count, error=excluded.error
            """, (
                ds["id"], ds["user_id"], ds["name"], ds.get("source","zip"),
                ds.get("folder_id"), ds.get("status","queued"),
                ds.get("total",0), ds.get("processed",0),
                ds.get("face_count",0), ds.get("error"),
                ds.get("created_at", time.time()),
            ))
        conn.commit()
    cache_delete(f"dataset:{ds['id']}")
    cache_delete(f"datasets:{ds['user_id']}")

def db_update_dataset_fields(dataset_id: str, **fields):
    if not fields:
        return
    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [dataset_id]
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE datasets SET {set_clause} WHERE id=?", values)
        conn.commit()
    cache_delete(f"dataset:{dataset_id}")
    # Also invalidate the user's dataset list cache
    ds = db_get_dataset(dataset_id)
    if ds:
        cache_delete(f"datasets:{ds['user_id']}")

def db_get_share(share_id: str) -> Optional[dict]:
    cached = cache_get(f"share:{share_id}")
    if cached:
        return cached
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM shares WHERE share_id = %s", (share_id,))
            row = cur.fetchone()
    if row:
        result = dict(row)
        cache_set(f"share:{share_id}", result, ttl=300)  # shares rarely change
        return result
    return None

def db_insert_share(share: dict):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO shares (share_id, dataset_id, dataset_name, created_at)
                VALUES (%(share_id)s, %(dataset_id)s, %(dataset_name)s, %(created_at)s)
            """, share)
        conn.commit()
    cache_set(f"share:{share['share_id']}", share, ttl=300)

# ── InsightFace model (lazy) ──────────────────────────────────────────────────
_face_model = None
_model_lock  = threading.Lock()

def get_face_model():
    global _face_model
    if _face_model is None:
        with _model_lock:
            if _face_model is None:
                from insightface.app import FaceAnalysis
                model_name = os.environ.get("INSIGHTFACE_MODEL", "buffalo_sc")
                det_w = int(os.environ.get("DET_SIZE", "320"))
                model = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
                model.prepare(ctx_id=-1, det_size=(det_w, det_w))
                _face_model = model
                log.info(f"Face model loaded: {model_name}, det_size={det_w}")
    return _face_model

# ── Image helpers ─────────────────────────────────────────────────────────────

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image.")
    return img

def encode_to_jpg(img_bgr: np.ndarray, quality: int = 92) -> bytes:
    """Encode an OpenCV image to JPEG bytes at the given quality."""
    _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes()

def cap_image(img_bgr: np.ndarray,
              max_width: int = 3840,
              max_bytes: int = None) -> bytes:
    """Apply resolution cap and optional file-size cap to a single image.

    Steps:
      1. If width > max_width, resize down to max_width (4K).
      2. If max_bytes is set and the encoded size exceeds it, reduce JPEG
         quality iteratively (92 → 85 → 75 → 65 → 55) until it fits.
         Stops at quality 55 to avoid visible artefacts.
      3. Returns the final JPEG bytes.

    Images already within both limits are encoded once at quality=92 (free
    tier) or returned without re-encoding (paid — handled by the caller).
    """
    h, w = img_bgr.shape[:2]

    # Step 1: Resolution cap
    if w > max_width:
        scale   = max_width / w
        img_bgr = cv2.resize(img_bgr, (max_width, int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    # Step 2: File-size cap (free tier only)
    if max_bytes:
        for quality in (92, 85, 75, 65, 55):
            data = encode_to_jpg(img_bgr, quality)
            if len(data) <= max_bytes:
                return data
        # If still over at quality 55, return it anyway — best we can do
        return data
    else:
        return encode_to_jpg(img_bgr, 92)

def extract_embedding(img_bgr: np.ndarray):
    model = get_face_model()
    faces = model.get(img_bgr)
    if not faces:
        return None, []
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return best.normed_embedding.astype("float32"), faces

FREE_TIER_MAX_BYTES = 6 * 1024 * 1024  # 6 MB per image

# ── Per-plan cloud image & dataset limits ─────────────────────────────────────
PLAN_LIMITS = {
    "free":          {"max_images": 100,    "max_datasets": 1},
    "personal_lite": {"max_images": 2000,   "max_datasets": 5},
    "personal_pro":  {"max_images": 10000,  "max_datasets": 15},
    "personal_max":  {"max_images": 30000,  "max_datasets": 30},
    "photo_starter": {"max_images": 100000, "max_datasets": 50},
    "photo_pro":     {"max_images": 500000, "max_datasets": 9999},
}

# ── Per-plan self-hosted limits (enforced by /api/license/validate) ───────────
# None = plan does not include self-hosted access.
SELF_HOSTED_PLAN_LIMITS = {
    "free":          None,
    "personal_lite": {
        "max_images":          2_000,
        "max_datasets":        5,
        "max_activations":     1,
        "offline_grace_hours": 24,
    },
    "personal_pro":  {
        "max_images":          10_000,
        "max_datasets":        15,
        "max_activations":     2,
        "offline_grace_hours": 72,
    },
    "personal_max":  {
        "max_images":          30_000,
        "max_datasets":        30,
        "max_activations":     3,
        "offline_grace_hours": 168,
    },
    "photo_starter": {
        "max_images":          100_000,
        "max_datasets":        50,
        "max_activations":     3,
        "offline_grace_hours": 168,
    },
    "photo_pro":     {
        "max_images":          500_000,
        "max_datasets":        9999,
        "max_activations":     5,
        "offline_grace_hours": 336,
    },
}

def get_plan_limits(user: dict) -> dict:
    plan = user.get("plan", "free") or "free"
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

def count_images_in_dir(directory: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in exts)

def compress_images_in_dir(directory: Path,
                           free_tier: bool = False,
                           max_width: int = 3840) -> tuple:
    """Process all images in a directory.

    free_tier=True  → 4K resolution cap + 6 MB per-file size cap.
                      Images already ≤4K AND ≤6 MB are re-encoded once at
                      quality 92 (to normalise formats); if they're already
                      a small JPEG they can stay untouched.
    free_tier=False → 4K resolution cap only. Images already ≤4K are
                      completely untouched — no re-encoding, no quality loss.

    Returns (total_count, processed_count).
    """
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [p for p in directory.rglob("*") if p.suffix.lower() in exts]
    processed = 0

    for img_path in image_paths:
        try:
            file_size = img_path.stat().st_size
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            needs_resize  = w > max_width
            needs_sizecap = free_tier and file_size > FREE_TIER_MAX_BYTES

            if not needs_resize and not needs_sizecap:
                continue  # nothing to do — leave file completely untouched

            max_bytes = FREE_TIER_MAX_BYTES if free_tier else None
            data      = cap_image(img, max_width=max_width, max_bytes=max_bytes)

            output_path = img_path.with_suffix('.jpg')
            output_path.write_bytes(data)
            if output_path != img_path:
                img_path.unlink()
            processed += 1

        except Exception as e:
            log.warning(f"Failed to process {img_path.name}: {e}")

    label = "4K + 6MB cap" if free_tier else "4K cap"
    log.info(f"{label}: {processed}/{len(image_paths)} images processed in {directory}")
    return len(image_paths), processed

# ── Embedding job ─────────────────────────────────────────────────────────────

def run_embedding_job(dataset_id: str):
    ds = db_get_dataset(dataset_id)
    if not ds:
        return

    dataset_dir = DATASETS_DIR / dataset_id
    emb_dir     = EMBEDDINGS_DIR / dataset_id
    emb_dir.mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in exts]

    log.info(f"[{dataset_id}] Embedding {len(image_paths)} images")
    db_update_dataset_fields(dataset_id, status="processing", total=len(image_paths), processed=0)

    embeddings, metadata = [], []
    model = get_face_model()

    for i, img_path in enumerate(image_paths):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            for face in model.get(img):
                emb  = face.normed_embedding.astype("float32")
                bbox = [int(x) for x in face.bbox.tolist()]
                embeddings.append(emb)
                metadata.append({
                    "image_path": str(img_path.relative_to(dataset_dir)),
                    "label":      img_path.parent.name,
                    "bbox":       bbox,
                })
        except Exception as exc:
            log.warning(f"[{dataset_id}] {img_path.name}: {exc}")

        # Update progress more frequently for better UI feedback
        update_freq = 1 if len(image_paths) < 50 else 5
        if (i + 1) % update_freq == 0 or i == len(image_paths) - 1:
            db_update_dataset_fields(dataset_id, processed=i+1)
            log.info(f"[{dataset_id}] Progress: {i+1}/{len(image_paths)}")

    if embeddings:
        emb_matrix = np.stack(embeddings).astype("float32")
        np.save(str(emb_dir / "embeddings.npy"), emb_matrix)
        with open(emb_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        import faiss
        idx = faiss.IndexFlatIP(emb_matrix.shape[1])
        idx.add(emb_matrix)
        faiss.write_index(idx, str(emb_dir / "face_index.faiss"))

    db_update_dataset_fields(dataset_id, status="ready", face_count=len(embeddings))
    log.info(f"[{dataset_id}] Done — {len(embeddings)} face embeddings")

    # Unload model after batch embedding to free RAM
    if os.environ.get("UNLOAD_MODEL_AFTER_EMBED", "true").lower() == "true":
        global _face_model
        with _model_lock:
            _face_model = None
        import gc
        gc.collect()
        log.info(f"[{dataset_id}] Face model unloaded to free RAM")

# ── Search ────────────────────────────────────────────────────────────────────

import collections
MAX_LOADED_INDEXES = int(os.environ.get("MAX_LOADED_INDEXES", "2"))
_index_cache: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
_index_cache_lock = threading.Lock()

def _get_index_and_meta(dataset_id: str):
    with _index_cache_lock:
        if dataset_id in _index_cache:
            _index_cache.move_to_end(dataset_id)
            return _index_cache[dataset_id]

    emb_dir    = EMBEDDINGS_DIR / dataset_id
    index_path = emb_dir / "face_index.faiss"
    meta_path  = emb_dir / "metadata.pkl"
    if not index_path.exists():
        return None, None

    import faiss
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    with _index_cache_lock:
        _index_cache[dataset_id] = (index, metadata)
        _index_cache.move_to_end(dataset_id)
        while len(_index_cache) > MAX_LOADED_INDEXES:
            _index_cache.popitem(last=False)

    return index, metadata

def search_in_dataset(dataset_id: str, query_emb: np.ndarray, top_k: int = 50):
    emb_key = f"search:{dataset_id}:{query_emb[:16].tobytes().hex()}"
    cached = cache_get(emb_key)
    if cached:
        return cached

    index, metadata = _get_index_and_meta(dataset_id)
    if index is None:
        return []

    q = query_emb.reshape(1, -1)
    scores, indices = index.search(q, min(top_k, index.ntotal))

    results, seen = [], set()
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or float(score) < 0.30:
            continue
        meta = metadata[idx]
        if meta["image_path"] in seen:
            continue
        seen.add(meta["image_path"])
        results.append({
            "score":      round(float(score), 4),
            "image_path": meta["image_path"],
            "label":      meta["label"],
            "bbox":       meta["bbox"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    cache_set(emb_key, results, ttl=120)
    return results

# ── Google Drive download ─────────────────────────────────────────────────────

def extract_gdrive_folder_id(url: str) -> Optional[str]:
    patterns = [
        r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{20,}$", url.strip()):
        return url.strip()
    return None

def _gdrive_download_file(file_id: str, out_path: Path, timeout: int = 60):
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    if b"confirm=" in body and b"<html" in body[:500].lower():
        m = re.search(rb'confirm=([0-9A-Za-z_\-]+)', body)
        if m:
            confirm = m.group(1).decode()
            url2 = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={confirm}"
            req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                body = resp2.read()
    out_path.write_bytes(body)

def _gdrive_list_folder(folder_id: str, api_key: str, dataset_id: str) -> list:
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files, page_token = [], None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name,mimeType)",
            "key": api_key,
            "pageSize": "1000",
        }
        if page_token:
            params["pageToken"] = page_token
        list_url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(list_url, headers={"User-Agent": "FaceFind/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        if "error" in data:
            raise RuntimeError(f"Drive API error: {data['error'].get('message', data['error'])}")
        page_files = [f for f in data.get("files", []) if Path(f["name"]).suffix.lower() in img_exts]
        all_files.extend(page_files)
        log.info(f"[{dataset_id}] Listed {len(page_files)} images this page, {len(all_files)} total")
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return all_files

def download_gdrive_folder(folder_id: str, dest_dir: Path, dataset_id: str):
    db_update_dataset_fields(dataset_id, status="downloading", total=0, processed=0)
    api_key    = os.environ.get("GOOGLE_API_KEY", "").strip()
    downloaded = 0
    last_error = ""

    if api_key:
        try:
            files = _gdrive_list_folder(folder_id, api_key, dataset_id)
            total = len(files)
            log.info(f"[{dataset_id}] Downloading {total} images via Drive API")
            db_update_dataset_fields(dataset_id, total=total, processed=0)
            for f in files:
                out_path = dest_dir / f["name"]
                log.info(f"[{dataset_id}] Downloading ({downloaded+1}/{total}) {f['name']}")
                try:
                    _gdrive_download_file(f["id"], out_path)
                    downloaded += 1
                except Exception as exc:
                    log.warning(f"[{dataset_id}] Skipped {f['name']}: {exc}")
                db_update_dataset_fields(dataset_id, processed=downloaded)
        except Exception as exc:
            last_error = str(exc)
            log.error(f"[{dataset_id}] Drive API error: {exc}")
    else:
        log.warning(f"[{dataset_id}] No GOOGLE_API_KEY — trying gdown")
        try:
            import gdown
            gdown.download_folder(id=folder_id, output=str(dest_dir), quiet=False, use_cookies=False)
            img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            downloaded = sum(1 for p in dest_dir.rglob("*") if p.suffix.lower() in img_exts)
        except Exception as exc:
            last_error = str(exc)
            log.error(f"[{dataset_id}] gdown error: {exc}")

    if downloaded == 0:
        msg = "Could not download any images. " + (
            "Set GOOGLE_API_KEY in Railway environment variables. " if not api_key else ""
        ) + (f"Last error: {last_error}" if last_error else "")
        db_update_dataset_fields(dataset_id, status="error", error=msg)
        return

    db_update_dataset_fields(dataset_id, status="queued")
    run_embedding_job(dataset_id)

# ── Auth helpers ──────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ── Email / OTP helpers ───────────────────────────────────────────────────────
OTP_EXPIRY    = int(os.environ.get("OTP_EXPIRY_SECONDS", "600"))  # 10 minutes

# ── Gmail API Email Sending ───────────────────────────────────────────────────

def send_email(to: str, subject: str, html_body: str):
    """Self-hosted mode: email is not required. OTP is printed to console instead."""
    log.info(f"[SELF-HOSTED] Email to {to} | Subject: {subject} | (email sending disabled)")

def generate_otp(email: str, purpose: str) -> str:
    """Generate a 6-digit OTP, store in DB, return the code."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE otp_codes SET used=1 WHERE email=? AND purpose=? AND used=0",
                (email.lower(), purpose)
            )
        conn.commit()
    code      = str(random.randint(100000, 999999))
    otp_id    = str(uuid.uuid4())
    expires_at = time.time() + OTP_EXPIRY
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO otp_codes (id, email, code, purpose, expires_at, used) VALUES (?,?,?,?,?,0)",
                (otp_id, email.lower(), code, purpose, expires_at)
            )
        conn.commit()
    return code

def verify_otp(email: str, code: str, purpose: str) -> bool:
    """Check OTP; marks it used and returns True if valid, False otherwise."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, expires_at FROM otp_codes WHERE email=? AND code=? AND purpose=? AND used=0",
                (email.lower(), code.strip(), purpose)
            )
            row = cur.fetchone()
        if not row:
            return False
        if time.time() > row["expires_at"]:
            return False
        with conn.cursor() as cur:
            cur.execute("UPDATE otp_codes SET used=1 WHERE id=?", (row["id"],))
        conn.commit()
    return True

def send_otp_email(email: str, code: str, purpose: str):
    """Send a nicely formatted OTP email."""
    if purpose == "register":
        subject = "Verify your FaceFind account"
        action  = "confirm your email address"
    else:
        subject = "Your FaceFind sign-in code"
        action  = "sign in to your account"
    
    html = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:480px;margin:0 auto;background:#f9f7f4;padding:32px 24px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:800;color:#4f46e5;letter-spacing:-1px">FaceFind</span>
      </div>
      <div style="background:#fff;border-radius:16px;padding:36px;box-shadow:0 4px 16px rgba(0,0,0,0.07)">
        <h2 style="margin:0 0 8px;font-size:20px;color:#1c1917">Your verification code</h2>
        <p style="margin:0 0 28px;font-size:14px;color:#78716c">
          Use the code below to {action}. It expires in 10 minutes.
        </p>
        <div style="text-align:center;background:#eef2ff;border-radius:12px;padding:24px 0;letter-spacing:10px;font-size:36px;font-weight:800;color:#4f46e5;margin-bottom:28px">
          {code}
        </div>
        <p style="margin:0;font-size:12px;color:#a8a29e;text-align:center">
          If you didn't request this, you can safely ignore this email.
        </p>
      </div>
    </div>
    """
    
    log.info(f"[SELF-HOSTED] OTP for {email} ({purpose}): {code}")
    print(f"*** Verification code for {email}: {code} ***")
    send_email(email, subject, html)

def db_create_user(email: str, name: str, password: str, email_verified: bool = False) -> dict:
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (id, email, name, password_hash, email_verified, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, email.lower(), name, pw_hash, email_verified, time.time()))
        conn.commit()
    return {"id": user_id, "email": email.lower(), "name": name}

def db_get_user_by_email(email: str) -> Optional[dict]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email.lower(),))
            row = cur.fetchone()
    return dict(row) if row else None

def db_create_session(user_id: str) -> str:
    token = secrets.token_hex(32)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sessions (token, user_id, created_at)
                VALUES (?, ?, ?)
            """, (token, user_id, time.time()))
        conn.commit()
    return token

def db_get_session_user(token: str) -> Optional[dict]:
    if not token:
        return None
    cached = cache_get(f"session:{token}")
    if cached:
        return cached
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT u.* FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = %s
            """, (token,))
            row = cur.fetchone()
    if row:
        user = dict(row)
        cache_set(f"session:{token}", user, ttl=30)
        return user
    return None

def db_delete_session(token: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE token = %s", (token,))
        conn.commit()
    cache_delete(f"session:{token}")

def require_auth(request) -> dict:
    from fastapi import Request
    token = request.cookies.get("ff_token") or request.headers.get("X-Auth-Token", "")
    user = db_get_session_user(token)
    if not user:
        raise HTTPException(401, "Not authenticated.")
    return user

# ── License key helpers ───────────────────────────────────────────────────────

def generate_license_key(user_id: str, plan: str) -> str:
    """
    Issue a new license key for a user based on their plan.
    Revokes any existing active key first (one key per user at a time).
    Returns the new license key string.
    """
    limits = SELF_HOSTED_PLAN_LIMITS.get(plan)
    if not limits:
        raise ValueError(f"Plan '{plan}' does not include self-hosted access.")

    # Format: FF-XXXX-XXXX-XXXX-XXXX
    raw = secrets.token_hex(8).upper()
    key = f"FF-{raw[0:4]}-{raw[4:8]}-{raw[8:12]}-{raw[12:16]}"

    # Revoke any existing active keys for this user
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE license_keys SET revoked=1 WHERE user_id=? AND revoked=0",
                (user_id,)
            )
        conn.commit()

    now = time.time()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO license_keys
                  (key, user_id, plan, created_at, expires_at, revoked, activations, max_activations)
                VALUES (?, ?, ?, ?, ?, 0, 0, ?)
            """, (
                key,
                user_id,
                plan,
                now,
                now + 365 * 24 * 3600,   # 1-year validity
                limits["max_activations"],
            ))
        conn.commit()

    log.info(f"License key issued: {key[:12]}… for user {user_id} on plan {plan}")
    return key


def db_get_license_key(key: str) -> Optional[dict]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM license_keys WHERE key = %s", (key,))
            row = cur.fetchone()
    return dict(row) if row else None


def issue_download_token(user_id: str) -> str:
    """Create a short-lived (15 min) one-use token for downloading the binary."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO download_tokens (token, user_id, created_at, expires_at, used)
                VALUES (?, ?, ?, ?, 0)
            """, (token, user_id, now, now + 15 * 60))
        conn.commit()
    return token


def consume_download_token(token: str) -> Optional[str]:
    """Validate and consume a download token. Returns user_id if valid."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM download_tokens WHERE token=? AND used=0 AND expires_at > ?",
                (token, time.time())
            )
            row = cur.fetchone()
        if not row:
            return None
        with conn.cursor() as cur:
            cur.execute("UPDATE download_tokens SET used=1 WHERE token=?", (token,))
        conn.commit()
    return row["user_id"]

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="FaceFind API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    """
    Prevent browsers and CDNs from caching .html files.
    API responses and static assets (JS/CSS/images) are unaffected.
    """
    response = await call_next(request)
    path = request.url.path
    if path.endswith(".html") or path == "/" or path == "":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"]        = "no-cache"
        response.headers["Expires"]       = "0"
    return response

@app.on_event("startup")
def on_startup():
    init_db()

# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/send-otp")
def send_otp(email: str = Form(...), purpose: str = Form(default="register")):
    """
    Send a 6-digit OTP to the given email.
    purpose = 'register' (new account) | 'login' (existing account passwordless, optional).
    For registration, also validates that email isn't already taken.
    """
    email = email.lower().strip()
    if purpose == "register":
        existing = db_get_user_by_email(email)
        if existing and existing.get("email_verified"):
            raise HTTPException(409, "An account with this email already exists.")
    try:
        code = generate_otp(email, purpose)
        send_otp_email(email, code, purpose)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error(f"OTP email failed: {e}")
        raise HTTPException(502, "Failed to send verification email. Please try again.")
    return {"ok": True, "message": "Verification code sent."}

@app.post("/api/auth/register")
def register(
    response:  Response,
    email:     str = Form(...),
    name:      str = Form(...),
    password:  str = Form(...),
    otp_code:  str = Form(...),
):
    email = email.lower().strip()
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    existing = db_get_user_by_email(email)
    if existing and existing.get("email_verified"):
        raise HTTPException(409, "An account with this email already exists.")
    if not verify_otp(email, otp_code, "register"):
        raise HTTPException(400, "Invalid or expired verification code.")
    # If an unverified stub existed, delete it so we can re-insert cleanly
    if existing and not existing.get("email_verified"):
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE email=%s", (email,))
            conn.commit()
    user  = db_create_user(email, name, password, email_verified=True)
    token = db_create_session(user["id"])
    response.set_cookie("ff_token", token, httponly=True, samesite="lax", max_age=60*60*24*30)
    return {"user": {"id": user["id"], "email": user["email"], "name": user["name"], "plan": user.get("plan") or "free"}}

@app.post("/api/auth/login")
def login(response: Response, email: str = Form(...), password: str = Form(...)):
    user = db_get_user_by_email(email)
    if not user or user["password_hash"] != hash_password(password):
        raise HTTPException(401, "Invalid email or password.")
    if not user.get("email_verified"):
        raise HTTPException(403, "Please verify your email before signing in.")
    token = db_create_session(user["id"])
    response.set_cookie("ff_token", token, httponly=True, samesite="lax", max_age=60*60*24*30)
    return {"user": {"id": user["id"], "email": user["email"], "name": user["name"], "plan": user.get("plan") or "free"}}

@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get("ff_token", "")
    if token:
        db_delete_session(token)
    response.delete_cookie("ff_token")
    return {"ok": True}

@app.get("/api/auth/me")
def me(request: Request):
    user = require_auth(request)
    return {"id": user["id"], "email": user["email"], "name": user["name"], "plan": user.get("plan") or "free"}

# ── Dataset endpoints ─────────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets(request: Request):
    user = require_auth(request)
    datasets = db_list_datasets(user["id"])
    # Append live disk size to each dataset (calculated from filesystem)
    for ds_id, ds in datasets.items():
        ds["size_bytes"] = get_dataset_disk_size(ds_id)
    return datasets

@app.post("/api/datasets/upload-zip")
async def upload_zip(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(default=""),
):
    user = require_auth(request)
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file.")

    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir()

    zip_path = dataset_dir / "upload.zip"
    zip_path.write_bytes(await file.read())
    
    # Extract ZIP
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_dir)
    zip_path.unlink()
    
    # ── Enforce dataset count limit ──────────────────────────────────────────
    limits = get_plan_limits(user)
    existing = db_list_datasets(user["id"])
    if len(existing) >= limits["max_datasets"]:
        import shutil; shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(400, f"Dataset limit reached. Your plan allows {limits['max_datasets']} dataset(s). Delete one or upgrade.")

    # ── Enforce image count limit ─────────────────────────────────────────────
    img_count = count_images_in_dir(dataset_dir)
    if img_count > limits["max_images"]:
        import shutil; shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(400, f"Too many images. Your plan allows up to {limits['max_images']:,} images per dataset. This ZIP contains {img_count:,}.")

    # Register dataset first so status is visible in the UI immediately
    ds = {
        "id": dataset_id, "user_id": user["id"],
        "name": name or file.filename.replace(".zip",""),
        "source": "zip", "folder_id": None,
        "status": "compressing", "total": 0, "processed": 0,
        "face_count": 0, "error": None, "created_at": time.time(),
    }
    db_upsert_dataset(ds)

    # Apply caps before embedding.
    is_free = user.get("plan", "free") == "free"
    total_imgs, capped = compress_images_in_dir(dataset_dir, free_tier=is_free)
    log.info(f"[{dataset_id}] {'Free' if is_free else 'Paid'} tier cap: {capped}/{total_imgs} images processed")

    background_tasks.add_task(run_embedding_job, dataset_id)
    return {"dataset_id": dataset_id, "status": "compressing"}

@app.post("/api/datasets/gdrive")
async def use_gdrive_folder(
    request: Request,
    background_tasks: BackgroundTasks,
    folder_url: str = Form(...),
    name: str = Form(default=""),
):
    user = require_auth(request)
    folder_id = extract_gdrive_folder_id(folder_url)
    if not folder_id:
        raise HTTPException(400, "Could not extract a folder ID from the provided URL.")

    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir()

    ds = {
        "id": dataset_id, "user_id": user["id"],
        "name": name or f"Drive Folder ({folder_id[:8]}…)",
        "source": "gdrive", "folder_id": folder_id,
        "status": "downloading", "total": 0, "processed": 0,
        "face_count": 0, "error": None, "created_at": time.time(),
    }
    db_upsert_dataset(ds)
    background_tasks.add_task(download_gdrive_folder, folder_id, dataset_dir, dataset_id)
    return {"dataset_id": dataset_id, "status": "downloading"}

@app.get("/api/datasets/{dataset_id}/status")
def dataset_status(dataset_id: str, request: Request):
    require_auth(request)
    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    return ds

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, request: Request):
    user = require_auth(request)
    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    if ds["user_id"] != user["id"]:
        raise HTTPException(403, "Not your dataset.")

    import shutil
    dataset_dir = DATASETS_DIR / dataset_id
    emb_dir     = EMBEDDINGS_DIR / dataset_id
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    if emb_dir.exists():
        shutil.rmtree(emb_dir)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))
        conn.commit()
    cache_delete(f"dataset:{dataset_id}")
    cache_delete(f"datasets:{user['id']}")
    log.info(f"Dataset {dataset_id} deleted by user {user['id']}")
    return {"ok": True}

@app.post("/api/datasets/{dataset_id}/add-images")
async def add_images_to_dataset(
    dataset_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    user = require_auth(request)
    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    if ds["user_id"] != user["id"]:
        raise HTTPException(403, "Not your dataset.")
    if ds["status"] != "ready":
        raise HTTPException(400, "Dataset must be in 'ready' state to add images.")
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file.")
    
    dataset_dir = DATASETS_DIR / dataset_id
    
    temp_dir = dataset_dir / f"_temp_{int(time.time())}"
    temp_dir.mkdir()
    
    zip_path = temp_dir / "upload.zip"
    zip_path.write_bytes(await file.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(temp_dir)
    zip_path.unlink()

    # ── Enforce image limit on combined total ─────────────────────────────────
    limits = get_plan_limits(user)
    existing_count = count_images_in_dir(dataset_dir)
    new_count      = count_images_in_dir(temp_dir)
    if existing_count + new_count > limits["max_images"]:
        import shutil; shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(400,
            f"Image limit exceeded. Your plan allows {limits['max_images']:,} images per dataset. "
            f"This dataset already has {existing_count:,} and you're adding {new_count:,}.")

    is_free = user.get("plan", "free") == "free"
    total_imgs, capped = compress_images_in_dir(temp_dir, free_tier=is_free)
    log.info(f"[{dataset_id}] {'Free' if is_free else 'Paid'} tier cap on new images: {capped}/{total_imgs} processed")
    
    import shutil
    for item in temp_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(temp_dir)
            target = dataset_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(target))
    
    shutil.rmtree(temp_dir)
    
    db_update_dataset_fields(dataset_id, status="queued")
    background_tasks.add_task(run_embedding_job, dataset_id)
    
    log.info(f"[{dataset_id}] Images added, re-embedding started")
    return {"ok": True, "dataset_id": dataset_id, "status": "queued"}

# ── Share endpoints ───────────────────────────────────────────────────────────

@app.post("/api/shares")
def create_share(request: Request, dataset_id: str = Form(...)):
    require_auth(request)
    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    if ds["status"] != "ready":
        raise HTTPException(400, "Dataset is not ready yet.")
    share_id = str(uuid.uuid4())[:12]
    share = {
        "share_id":    share_id,
        "dataset_id":  dataset_id,
        "dataset_name": ds["name"],
        "created_at":  time.time(),
    }
    db_insert_share(share)
    return {"share_id": share_id}

@app.get("/api/shares/{share_id}")
def get_share(share_id: str):
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share link not found.")
    return share

# ── Search endpoint ───────────────────────────────────────────────────────────

@app.post("/api/shares/{share_id}/detect-faces")
async def detect_faces_in_selfie(share_id: str, file: UploadFile = File(...)):
    """Detect all faces in an uploaded image and return cropped thumbnails as base64."""
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share link not found.")

    contents = await file.read()
    try:
        img = decode_image(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))

    model = get_face_model()
    faces = model.get(img)
    if not faces:
        return JSONResponse({"face_detected": False, "faces": []})

    faces_sorted = sorted(faces, key=lambda f: f.bbox[0])

    face_crops = []
    h, w = img.shape[:2]
    for i, face in enumerate(faces_sorted):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        x1c = max(0, x1 - pad_x)
        y1c = max(0, y1 - pad_y)
        x2c = min(w, x2 + pad_x)
        y2c = min(h, y2 + pad_y)
        crop = img[y1c:y2c, x1c:x2c]
        thumb = cv2.resize(crop, (128, 128))
        _, buf = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        face_crops.append({
            "index": i,
            "thumbnail": f"data:image/jpeg;base64,{b64}",
            "embedding": faces_sorted[i].normed_embedding.astype("float32").tolist(),
        })

    return {"face_detected": True, "faces": face_crops}


@app.post("/api/shares/{share_id}/search")
async def search_by_selfie(share_id: str, file: UploadFile = File(...), face_indices: str = Form(default=None)):
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share link not found.")
    ds = db_get_dataset(share["dataset_id"])
    if not ds or ds["status"] != "ready":
        raise HTTPException(400, "Dataset not ready.")

    contents = await file.read()
    try:
        img = decode_image(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))

    t0 = time.time()
    model = get_face_model()
    all_faces = model.get(img)
    if not all_faces:
        return JSONResponse({"face_detected": False, "matches": [], "latency_ms": round((time.time()-t0)*1000,1)})

    all_faces_sorted = sorted(all_faces, key=lambda f: f.bbox[0])

    if face_indices:
        try:
            selected = [int(i) for i in face_indices.split(",") if i.strip().isdigit()]
            faces_to_search = [all_faces_sorted[i] for i in selected if i < len(all_faces_sorted)]
        except Exception:
            faces_to_search = all_faces_sorted
    else:
        faces_to_search = [max(all_faces_sorted, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]

    merged: dict = {}
    for face in faces_to_search:
        emb = face.normed_embedding.astype("float32")
        results = search_in_dataset(share["dataset_id"], emb, top_k=100)
        for m in results:
            key = m["image_path"]
            if key not in merged or m["score"] > merged[key]["score"]:
                merged[key] = m

    sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    return {
        "face_detected": True,
        "num_faces":     len(all_faces_sorted),
        "matches":       sorted_results,
        "latency_ms":    round((time.time()-t0)*1000, 1),
        "dataset_id":    share["dataset_id"],
    }

# ── Image serving ─────────────────────────────────────────────────────────────

@app.get("/api/image/{dataset_id}/{image_path:path}")
def serve_image(dataset_id: str, image_path: str):
    full_path = DATASETS_DIR / dataset_id / image_path
    if not full_path.exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(str(full_path))

# ── Thumbnail serving ─────────────────────────────────────────────────────────
THUMB_WIDTH = int(os.environ.get("THUMB_WIDTH", "400"))

@app.get("/api/thumb/{dataset_id}/{image_path:path}")
def serve_thumb(dataset_id: str, image_path: str):
    src_path   = DATASETS_DIR / dataset_id / image_path
    if not src_path.exists():
        raise HTTPException(404, "Image not found.")

    thumb_path = THUMBS_DIR / dataset_id / (image_path + ".thumb.jpg")
    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    if not thumb_path.exists():
        img = cv2.imread(str(src_path))
        if img is None:
            raise HTTPException(422, "Cannot decode image.")
        h, w = img.shape[:2]
        if w > THUMB_WIDTH:
            new_h = int(h * THUMB_WIDTH / w)
            img   = cv2.resize(img, (THUMB_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])

    return FileResponse(
        str(thumb_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800, immutable"},
    )

# ── Download & license endpoints ──────────────────────────────────────────────

@app.get("/api/download/info")
def download_info(request: Request):
    """
    Return download eligibility and existing license key info for the current user.
    Used by download.html to show the correct state (locked / eligible / key details).
    """
    user = require_auth(request)
    plan = user.get("plan", "free") or "free"
    limits = SELF_HOSTED_PLAN_LIMITS.get(plan)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM license_keys WHERE user_id=? AND revoked=0 ORDER BY created_at DESC LIMIT 1",
                (user["id"],)
            )
            key_row = cur.fetchone()

    key_data = dict(key_row) if key_row else None

    return {
        "plan": plan,
        "self_hosted_eligible": limits is not None,
        "limits": limits,
        "license_key": {
            "key":             key_data["key"],
            "expires_at":      key_data["expires_at"],
            "activations":     key_data["activations"],
            "max_activations": key_data["max_activations"],
            "plan":            key_data["plan"],
        } if key_data else None,
        "user": {"name": user["name"], "email": user["email"]},
    }


@app.post("/api/download/generate-key")
def generate_key_endpoint(request: Request):
    """
    Generate (or regenerate) a license key for the authenticated user.
    Only available on paid plans that include self-hosted access.
    Regenerating immediately revokes the old key.
    """
    user = require_auth(request)
    plan = user.get("plan", "free") or "free"

    if not SELF_HOSTED_PLAN_LIMITS.get(plan):
        raise HTTPException(403, "Your current plan does not include self-hosted access. Please upgrade.")

    try:
        key = generate_license_key(user["id"], plan)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {"key": key, "plan": plan}


@app.post("/api/download/request-link")
def request_download_link(request: Request):
    """
    Issue a short-lived signed download URL for the self-hosted executable.
    Validates plan eligibility and requires a valid license key before issuing.
    """
    user = require_auth(request)
    plan = user.get("plan", "free") or "free"

    if not SELF_HOSTED_PLAN_LIMITS.get(plan):
        raise HTTPException(403, "Self-hosted downloads require a paid plan.")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT key FROM license_keys WHERE user_id=? AND revoked=0 AND expires_at > ? LIMIT 1",
                (user["id"], time.time())
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(400, "Generate a license key first before downloading.")

    token = issue_download_token(user["id"])
    log.info(f"Download link issued for user {user['id']} (plan={plan})")
    return {
        "download_url":       f"/api/download/file?token={token}",
        "expires_in_seconds": 900,
        "filename":           "facefind-selfhosted.zip",
    }


@app.get("/api/download/file")
def download_file(token: str, request: Request):
    """
    Serve the self-hosted executable ZIP after validating the one-use download token.
    """
    user_id = consume_download_token(token)
    if not user_id:
        raise HTTPException(403, "Invalid or expired download link. Please request a new one.")

    if not EXECUTABLE_PATH.exists():
        raise HTTPException(503, "The download package is not yet available. Please contact support.")

    log.info(f"Executable downloaded by user {user_id}")
    return FileResponse(
        str(EXECUTABLE_PATH),
        media_type="application/zip",
        filename="facefind-selfhosted.zip",
        headers={"Content-Disposition": 'attachment; filename="facefind-selfhosted.zip"'},
    )


# ── License validation endpoint (called by the self-hosted app) ───────────────

@app.post("/api/license/validate")
async def validate_license(request: Request):
    """
    Called by the self-hosted executable on startup and periodically while running.
    Returns the plan limits so the local app can enforce them.

    Request body (JSON):
        { "key": "FF-XXXX-XXXX-XXXX-XXXX", "machine_id": "<stable machine identifier>" }

    Responses:
        200  { valid: true,  plan, limits, expires_at, offline_grace_hours }
        403  { valid: false, reason }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    key_str    = (body.get("key") or "").strip().upper()
    machine_id = (body.get("machine_id") or "").strip()

    if not key_str or not machine_id:
        raise HTTPException(400, "key and machine_id are required.")

    key_data = db_get_license_key(key_str)

    if not key_data:
        return JSONResponse({"valid": False, "reason": "License key not found."}, status_code=403)

    if key_data["revoked"]:
        return JSONResponse({"valid": False, "reason": "License key has been revoked."}, status_code=403)

    if time.time() > key_data["expires_at"]:
        return JSONResponse({"valid": False, "reason": "License key has expired. Please renew your subscription."}, status_code=403)

    plan   = key_data["plan"]
    limits = SELF_HOSTED_PLAN_LIMITS.get(plan)
    if not limits:
        return JSONResponse({"valid": False, "reason": "This plan does not include self-hosted access."}, status_code=403)

    client_ip = request.client.host if request.client else "unknown"

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE license_keys
                SET activations  = MIN(activations + 1, max_activations),
                    last_seen_at = ?,
                    last_seen_ip = ?
                WHERE key = ?
            """, (time.time(), client_ip, key_str))
        conn.commit()

    log.info(f"License validated: {key_str[:12]}… plan={plan} ip={client_ip}")

    return {
        "valid":                True,
        "plan":                 plan,
        "expires_at":           key_data["expires_at"],
        "offline_grace_hours":  limits["offline_grace_hours"],
        "limits": {
            "max_images":   limits["max_images"],
            "max_datasets": limits["max_datasets"],
        },
    }


@app.post("/api/license/revoke")
def revoke_license(request: Request):
    """
    Revoke the caller's active license key (e.g. when moving to a new machine).
    A new key can be generated via /api/download/generate-key after revoking.
    """
    user = require_auth(request)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE license_keys SET revoked=1 WHERE user_id=? AND revoked=0",
                (user["id"],)
            )
            count = cur.rowcount
        conn.commit()

    if count == 0:
        raise HTTPException(404, "No active license key found.")

    log.info(f"License revoked for user {user['id']}")
    return {"ok": True, "message": "License key revoked. You can generate a new one at any time."}


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.post("/api/admin/set-plan")
def admin_set_plan(request: Request, target_email: str = Form(...), plan: str = Form(...)):
    """
    Admin-only: set a user's plan.
    Call this from your payment provider's webhook after a successful payment.
    Requires ADMIN_SECRET environment variable to be set.
    Pass it as the X-Admin-Secret request header.
    """
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    provided     = request.headers.get("X-Admin-Secret", "")
    if not admin_secret or provided != admin_secret:
        raise HTTPException(403, "Forbidden.")

    valid_plans = list(PLAN_LIMITS.keys())
    if plan not in valid_plans:
        raise HTTPException(400, f"Invalid plan. Choose from: {valid_plans}")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET plan=? WHERE email=?", (plan, target_email.lower()))
            if cur.rowcount == 0:
                raise HTTPException(404, "User not found.")
        conn.commit()

    log.info(f"Plan updated: {target_email} → {plan}")
    return {"ok": True, "email": target_email, "plan": plan}

# ── Debug endpoint to check email config ──────────────────────────────────────

@app.get("/api/debug/email")
def debug_email():
    """Self-hosted: email is disabled, OTPs are logged to console."""
    return {"status": "Email disabled in self-hosted mode. OTPs are printed to the console."}

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    r = get_redis()
    return {
        "status":    "ok",
        "timestamp": time.time(),
        "redis":     "connected" if r else "disabled",
        "db":        "postgres",
        "email":     "disabled (self-hosted mode)"
    }

# ── Frontend static files ─────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))