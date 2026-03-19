"""
FaceFind – Backend API (Railway Edition)
- Postgres for metadata (datasets, shares)
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

import numpy as np
import cv2
import psycopg2
import psycopg2.extras
import redis as redis_lib
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Gmail API imports
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build

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

for d in [DATASETS_DIR, EMBEDDINGS_DIR, UPLOADS_DIR, THUMBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Postgres ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ["DATABASE_URL"]  # set automatically by Railway Postgres plugin

def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id              TEXT PRIMARY KEY,
                    email           TEXT UNIQUE NOT NULL,
                    name            TEXT NOT NULL,
                    password_hash   TEXT NOT NULL,
                    email_verified  BOOLEAN DEFAULT FALSE,
                    created_at      DOUBLE PRECISION
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS otp_codes (
                    id          TEXT PRIMARY KEY,
                    email       TEXT NOT NULL,
                    code        TEXT NOT NULL,
                    purpose     TEXT NOT NULL,
                    expires_at  DOUBLE PRECISION,
                    used        BOOLEAN DEFAULT FALSE
                );
            """)
            # Migration: add email_verified to existing users table
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT FALSE;
            """)
            # Mark ALL existing users as verified so they aren't locked out
            cur.execute("""
                UPDATE users SET email_verified = TRUE WHERE email_verified = FALSE;
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token       TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    created_at  DOUBLE PRECISION
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    source      TEXT DEFAULT 'zip',
                    folder_id   TEXT,
                    status      TEXT DEFAULT 'queued',
                    total       INT DEFAULT 0,
                    processed   INT DEFAULT 0,
                    face_count  INT DEFAULT 0,
                    error       TEXT,
                    created_at  DOUBLE PRECISION
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shares (
                    share_id     TEXT PRIMARY KEY,
                    dataset_id   TEXT NOT NULL,
                    dataset_name TEXT,
                    created_at   DOUBLE PRECISION
                );
            """)
            # Migration: add user_id to datasets table if it doesn't exist yet
            cur.execute("""
                ALTER TABLE datasets ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT '';
            """)
        conn.commit()
    log.info("DB tables ready")

# ── Redis cache ───────────────────────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "")
_redis: Optional[redis_lib.Redis] = None

def get_redis() -> Optional[redis_lib.Redis]:
    global _redis
    if _redis is None and REDIS_URL:
        try:
            _redis = redis_lib.from_url(REDIS_URL, decode_responses=True)
            _redis.ping()
            log.info("Redis connected")
        except Exception as e:
            log.warning(f"Redis unavailable: {e}. Caching disabled.")
            _redis = None
    return _redis

def cache_get(key: str):
    r = get_redis()
    if not r:
        return None
    try:
        val = r.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None

def cache_set(key: str, value, ttl: int = 30):
    r = get_redis()
    if not r:
        return
    try:
        r.setex(key, ttl, json.dumps(value))
    except Exception:
        pass

def cache_delete(key: str):
    r = get_redis()
    if not r:
        return
    try:
        r.delete(key)
    except Exception:
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
                VALUES (%(id)s, %(user_id)s, %(name)s, %(source)s, %(folder_id)s, %(status)s, %(total)s, %(processed)s, %(face_count)s, %(error)s, %(created_at)s)
                ON CONFLICT (id) DO UPDATE SET
                    name=EXCLUDED.name, source=EXCLUDED.source, folder_id=EXCLUDED.folder_id,
                    status=EXCLUDED.status, total=EXCLUDED.total, processed=EXCLUDED.processed,
                    face_count=EXCLUDED.face_count, error=EXCLUDED.error
            """, {
                "id": ds["id"], "user_id": ds["user_id"], "name": ds["name"], "source": ds.get("source","zip"),
                "folder_id": ds.get("folder_id"), "status": ds.get("status","queued"),
                "total": ds.get("total",0), "processed": ds.get("processed",0),
                "face_count": ds.get("face_count",0), "error": ds.get("error"),
                "created_at": ds.get("created_at", time.time()),
            })
        conn.commit()
    cache_delete(f"dataset:{ds['id']}")
    cache_delete(f"datasets:{ds['user_id']}")

def db_update_dataset_fields(dataset_id: str, **fields):
    if not fields:
        return
    set_clause = ", ".join(f"{k}=%s" for k in fields)
    values = list(fields.values()) + [dataset_id]
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE datasets SET {set_clause} WHERE id=%s", values)
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
    """
    Send HTML email via Gmail API (works on Railway - uses HTTPS port 443)
    """
    # Get credentials from environment variables
    client_id = os.environ.get("GMAIL_CLIENT_ID")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET")
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN")
    
    if not all([client_id, client_secret, refresh_token]):
        log.error("Gmail API credentials not configured")
        raise RuntimeError(
            "Email not configured. Set GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN in Railway environment variables."
        )
    
    # Create credentials object with refresh token
    credentials = Credentials(
        token=None,  # Will be refreshed automatically
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/gmail.send"]
    )
    
    # Refresh token automatically if needed
    try:
        if credentials.expired:
            credentials.refresh(GoogleRequest())
    except Exception as e:
        log.error(f"Failed to refresh token: {e}")
        raise RuntimeError(f"Gmail API authentication failed: {str(e)}")
    
    # Create the email message
    message = EmailMessage()
    message.set_content("Please enable HTML to view this message")
    message.add_alternative(html_body, subtype="html")
    message["To"] = to
    message["From"] = "FaceFind <admin.facefind@gmail.com>"
    message["Subject"] = subject
    
    # Encode the message (Gmail API requires base64url format)
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    # Send via Gmail API
    try:
        service = build("gmail", "v1", credentials=credentials)
        send_message = service.users().messages().send(
            userId="me", 
            body={"raw": encoded_message}
        ).execute()
        log.info(f"Email sent to {to}: {subject}")
        return send_message
    except Exception as e:
        log.error(f"Gmail API error: {e}")
        raise RuntimeError(f"Failed to send email: {str(e)}")

def generate_otp(email: str, purpose: str) -> str:
    """Generate a 6-digit OTP, store in DB, return the code."""
    # Invalidate any previous unused OTPs for this email+purpose
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE otp_codes SET used=TRUE WHERE email=%s AND purpose=%s AND used=FALSE",
                (email.lower(), purpose)
            )
        conn.commit()
    code      = str(random.randint(100000, 999999))
    otp_id    = str(uuid.uuid4())
    expires_at = time.time() + OTP_EXPIRY
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO otp_codes (id, email, code, purpose, expires_at, used) VALUES (%s,%s,%s,%s,%s,FALSE)",
                (otp_id, email.lower(), code, purpose, expires_at)
            )
        conn.commit()
    return code

def verify_otp(email: str, code: str, purpose: str) -> bool:
    """Check OTP; marks it used and returns True if valid, False otherwise."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, expires_at FROM otp_codes WHERE email=%s AND code=%s AND purpose=%s AND used=FALSE",
                (email.lower(), code.strip(), purpose)
            )
            row = cur.fetchone()
        if not row:
            return False
        if time.time() > row["expires_at"]:
            return False
        with conn.cursor() as cur:
            cur.execute("UPDATE otp_codes SET used=TRUE WHERE id=%s", (row["id"],))
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
    
    send_email(email, subject, html)

def db_create_user(email: str, name: str, password: str, email_verified: bool = False) -> dict:
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (id, email, name, password_hash, email_verified, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
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
                VALUES (%s, %s, %s)
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
        cache_set(f"session:{token}", user, ttl=300)
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

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="FaceFind API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()
    get_redis()  # warm up connection

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
    return {"user": {"id": user["id"], "email": user["email"], "name": user["name"]}}

@app.post("/api/auth/login")
def login(response: Response, email: str = Form(...), password: str = Form(...)):
    user = db_get_user_by_email(email)
    if not user or user["password_hash"] != hash_password(password):
        raise HTTPException(401, "Invalid email or password.")
    if not user.get("email_verified"):
        raise HTTPException(403, "Please verify your email before signing in.")
    token = db_create_session(user["id"])
    response.set_cookie("ff_token", token, httponly=True, samesite="lax", max_age=60*60*24*30)
    return {"user": {"id": user["id"], "email": user["email"], "name": user["name"]}}

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
    return {"id": user["id"], "email": user["email"], "name": user["name"]}

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
    # Free tier: 4K resolution cap + 6 MB per-file size cap.
    # Paid tiers: 4K resolution cap only, originals preserved otherwise.
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

    # Delete files from volume
    import shutil
    dataset_dir = DATASETS_DIR / dataset_id
    emb_dir     = EMBEDDINGS_DIR / dataset_id
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    if emb_dir.exists():
        shutil.rmtree(emb_dir)

    # Delete from DB (shares referencing this dataset are orphaned but harmless)
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
    
    # Create temp directory for new images
    temp_dir = dataset_dir / f"_temp_{int(time.time())}"
    temp_dir.mkdir()
    
    # Extract new images
    zip_path = temp_dir / "upload.zip"
    zip_path.write_bytes(await file.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(temp_dir)
    zip_path.unlink()
    
    # Apply caps to new images before merging.
    is_free = user.get("plan", "free") == "free"
    total_imgs, capped = compress_images_in_dir(temp_dir, free_tier=is_free)
    log.info(f"[{dataset_id}] {'Free' if is_free else 'Paid'} tier cap on new images: {capped}/{total_imgs} processed")
    
    # Move all files from temp to main dataset directory
    import shutil
    for item in temp_dir.rglob("*"):
        if item.is_file():
            # Maintain directory structure
            rel_path = item.relative_to(temp_dir)
            target = dataset_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(target))
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    # Set status to processing and re-run embedding job
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

    # Sort faces left-to-right for consistent ordering
    faces_sorted = sorted(faces, key=lambda f: f.bbox[0])

    face_crops = []
    h, w = img.shape[:2]
    for i, face in enumerate(faces_sorted):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        # Add 20% padding around face
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        x1c = max(0, x1 - pad_x)
        y1c = max(0, y1 - pad_y)
        x2c = min(w, x2 + pad_x)
        y2c = min(h, y2 + pad_y)
        crop = img[y1c:y2c, x1c:x2c]
        # Resize crop to 128x128 thumbnail
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

    # Sort faces left-to-right for consistent ordering
    all_faces_sorted = sorted(all_faces, key=lambda f: f.bbox[0])

    # Determine which faces to search
    if face_indices:
        try:
            selected = [int(i) for i in face_indices.split(",") if i.strip().isdigit()]
            faces_to_search = [all_faces_sorted[i] for i in selected if i < len(all_faces_sorted)]
        except Exception:
            faces_to_search = all_faces_sorted
    else:
        # Default: use the largest face (original behaviour)
        faces_to_search = [max(all_faces_sorted, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]

    # Search each selected face and merge results (deduplicate by image_path, keep best score)
    merged: dict = {}
    for face in faces_to_search:
        emb = face.normed_embedding.astype("float32")
        results = search_in_dataset(share["dataset_id"], emb, top_k=100)
        for m in results:
            key = m["image_path"]
            if key not in merged or m["score"] > merged[key]["score"]:
                merged[key] = m

    # Sort merged results by score descending
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

    # Thumb stored as .jpg regardless of original format
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
        headers={"Cache-Control": "public, max-age=604800, immutable"},  # 7 days
    )

# ── Debug endpoint to check email config ─────────────────────────────────────
@app.get("/api/debug/email")
def debug_email():
    """Debug endpoint to check email configuration"""
    client_id = os.environ.get("GMAIL_CLIENT_ID", "")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET", "")
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN", "")
    
    return {
        "client_id_present": bool(client_id),
        "client_id_preview": client_id[:10] + "..." if client_id else None,
        "client_secret_present": bool(client_secret),
        "refresh_token_present": bool(refresh_token),
        "refresh_token_preview": refresh_token[:10] + "..." if refresh_token else None,
        "status": "Configured" if all([client_id, client_secret, refresh_token]) else "Missing credentials"
    }

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    r = get_redis()
    return {
        "status":    "ok",
        "timestamp": time.time(),
        "redis":     "connected" if r else "disabled",
        "db":        "postgres",
        "email":     "configured" if all([os.environ.get("GMAIL_CLIENT_ID"), os.environ.get("GMAIL_CLIENT_SECRET"), os.environ.get("GMAIL_REFRESH_TOKEN")]) else "not configured"
    }

# ── Frontend static files ─────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))