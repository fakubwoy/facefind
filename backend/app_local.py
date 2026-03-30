"""
Pixmatch – Local Desktop Backend
- SQLite for all metadata (no Postgres/Redis needed)
- Local filesystem for images + FAISS indexes
- Activation-based auth: license key links to cloud account, auto-login on restart
- Plan limits mirror the paid tier of the activated license key
- No pricing / download / B2 endpoints
"""

import os, io, re, uuid, time, json, pickle, zipfile, threading, hashlib, secrets, base64
import sys, sqlite3
from pathlib import Path
from typing import Optional
import logging
import collections
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
import uvicorn
import requests as http_requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pixmatch-local")

# ── Paths ─────────────────────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    # Running as PyInstaller bundle
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

DATA_DIR       = BASE_DIR / "data"
DATASETS_DIR   = DATA_DIR / "datasets"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
UPLOADS_DIR    = DATA_DIR / "uploads"
THUMBS_DIR     = DATA_DIR / "thumbs"
DB_PATH        = DATA_DIR / "local.db"
# Frontend HTML/JS assets are bundled inside _MEIPASS when frozen,
# NOT next to the exe — so we must look there first.
FRONTEND_DIR   = Path(getattr(sys, "_MEIPASS", BASE_DIR)) / "frontend"

for d in [DATASETS_DIR, EMBEDDINGS_DIR, UPLOADS_DIR, THUMBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cloud API base URL for license validation
CLOUD_API_URL = os.environ.get("CLOUD_API_URL", "https://www.lenstagram.com")

# ── Plan limits (mirrors cloud SELF_HOSTED_PLAN_LIMITS) ──────────────────────
PLAN_LIMITS = {
    "personal_lite": {"max_images": 2_000,   "max_datasets": 5},
    "personal_pro":  {"max_images": 10_000,  "max_datasets": 15},
    "personal_max":  {"max_images": 30_000,  "max_datasets": 30},
    "photo_starter": {"max_images": 100_000, "max_datasets": 50},
    "photo_pro":     {"max_images": 500_000, "max_datasets": 9999},
}

def get_plan_limits(plan: str) -> dict:
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["personal_lite"])

# ── SQLite DB ─────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

_db_lock = threading.Lock()

def init_db():
    with _db_lock:
        conn = get_db()
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS activation (
                id          INTEGER PRIMARY KEY,
                license_key TEXT NOT NULL,
                user_email  TEXT NOT NULL,
                user_name   TEXT NOT NULL,
                plan        TEXT NOT NULL,
                activated_at REAL NOT NULL,
                last_validated_at REAL,
                offline_grace_hours INTEGER DEFAULT 72,
                expires_at  REAL
            );

            CREATE TABLE IF NOT EXISTS datasets (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                source      TEXT DEFAULT 'zip',
                folder_id   TEXT,
                status      TEXT DEFAULT 'queued',
                total       INTEGER DEFAULT 0,
                processed   INTEGER DEFAULT 0,
                face_count  INTEGER DEFAULT 0,
                error       TEXT,
                group_id    TEXT,
                created_at  REAL
            );

            CREATE TABLE IF NOT EXISTS event_groups (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                description   TEXT DEFAULT '',
                watermark_text TEXT DEFAULT '',
                event_type    TEXT DEFAULT 'other',
                event_date    TEXT,
                created_at    REAL
            );

            CREATE TABLE IF NOT EXISTS shares (
                share_id     TEXT PRIMARY KEY,
                dataset_id   TEXT NOT NULL UNIQUE,
                dataset_name TEXT,
                watermark_text TEXT,
                view_count   INTEGER DEFAULT 0,
                download_count INTEGER DEFAULT 0,
                last_viewed_at REAL,
                created_at   REAL
            );
        """)
        conn.commit()
        conn.close()
    log.info("Local SQLite DB ready at %s", DB_PATH)

# ── Activation helpers ────────────────────────────────────────────────────────

def get_activation() -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM activation ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None

def save_activation(license_key: str, user_email: str, user_name: str,
                    plan: str, offline_grace_hours: int, expires_at: float):
    conn = get_db()
    now = time.time()
    conn.execute("DELETE FROM activation")
    conn.execute(
        "INSERT INTO activation (license_key, user_email, user_name, plan, activated_at, last_validated_at, offline_grace_hours, expires_at) VALUES (?,?,?,?,?,?,?,?)",
        (license_key, user_email, user_name, plan, now, now, offline_grace_hours, expires_at)
    )
    conn.commit()
    conn.close()

def update_last_validated():
    conn = get_db()
    conn.execute("UPDATE activation SET last_validated_at=?", (time.time(),))
    conn.commit()
    conn.close()

def is_license_valid_offline() -> bool:
    """Return True if the app has ever been activated (local-first: no online re-check needed)."""
    act = get_activation()
    return act is not None

def validate_license_with_cloud(key: str) -> dict:
    """Call cloud /api/license/validate. Returns parsed response or raises."""
    machine_id = _get_machine_id()
    resp = http_requests.post(
        f"{CLOUD_API_URL}/api/license/validate",
        json={"key": key, "machine_id": machine_id},
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json()
    try:
        detail = resp.json()
    except Exception:
        detail = {"reason": f"HTTP {resp.status_code}"}
    raise ValueError(detail.get("reason", "License validation failed."))

def fetch_userinfo_from_cloud(key: str) -> dict:
    resp = http_requests.post(
        f"{CLOUD_API_URL}/api/license/userinfo",
        json={"key": key},
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json()
    raise ValueError("Could not fetch user info from cloud.")

def _get_machine_id() -> str:
    """Return a stable per-machine ID derived from hardware info."""
    try:
        import platform, hashlib
        raw = platform.node() + platform.machine() + platform.processor()
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
    except Exception:
        return "unknown"

# ── Auth guard ────────────────────────────────────────────────────────────────

def require_activation(request: Request) -> dict:
    """
    For local app: activation replaces cloud sessions.
    Once the app has been activated once, it works forever locally —
    no sign-in, no repeated online checks, no grace-period blocks.
    The only hard gate is "has the user ever activated?"
    """
    act = get_activation()
    if not act:
        raise HTTPException(401, "App not activated. Please enter your license key.")
    return act

# ── DB helpers ────────────────────────────────────────────────────────────────

def db_list_datasets() -> dict:
    conn = get_db()
    rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    conn.close()
    return {row["id"]: dict(row) for row in rows}

def db_get_dataset(dataset_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def db_upsert_dataset(ds: dict):
    conn = get_db()
    conn.execute("""
        INSERT INTO datasets (id, name, source, folder_id, status, total, processed, face_count, error, group_id, created_at)
        VALUES (:id,:name,:source,:folder_id,:status,:total,:processed,:face_count,:error,:group_id,:created_at)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name, source=excluded.source, folder_id=excluded.folder_id,
            status=excluded.status, total=excluded.total, processed=excluded.processed,
            face_count=excluded.face_count, error=excluded.error, group_id=excluded.group_id
    """, {
        "id": ds["id"], "name": ds["name"], "source": ds.get("source","zip"),
        "folder_id": ds.get("folder_id"), "status": ds.get("status","queued"),
        "total": ds.get("total",0), "processed": ds.get("processed",0),
        "face_count": ds.get("face_count",0), "error": ds.get("error"),
        "group_id": ds.get("group_id"), "created_at": ds.get("created_at", time.time()),
    })
    conn.commit()
    conn.close()

def db_update_dataset_fields(dataset_id: str, **fields):
    if not fields:
        return
    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [dataset_id]
    conn = get_db()
    conn.execute(f"UPDATE datasets SET {set_clause} WHERE id=?", values)
    conn.commit()
    conn.close()

def db_get_share(share_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM shares WHERE share_id=?", (share_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def db_list_groups() -> list:
    conn = get_db()
    rows = conn.execute("SELECT * FROM event_groups ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_group(group_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM event_groups WHERE id=?", (group_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

# ── Image helpers ─────────────────────────────────────────────────────────────

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image.")
    return img

def encode_to_jpg(img_bgr: np.ndarray, quality: int = 92) -> bytes:
    _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes()

def count_images_in_dir(directory: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in exts)

def compress_images_in_dir(directory: Path, max_width: int = 3840) -> tuple:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [p for p in directory.rglob("*") if p.suffix.lower() in exts]
    processed = 0
    for img_path in image_paths:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            if w > max_width:
                scale = max_width / w
                img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
                data = encode_to_jpg(img, 92)
                output_path = img_path.with_suffix('.jpg')
                output_path.write_bytes(data)
                if output_path != img_path:
                    img_path.unlink()
                processed += 1
        except Exception as e:
            log.warning(f"Failed to process {img_path.name}: {e}")
    return len(image_paths), processed

# ── Face model (lazy) ─────────────────────────────────────────────────────────
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
                log.info(f"Face model loaded: {model_name}")
    return _face_model

# ── Embedding job ─────────────────────────────────────────────────────────────

def run_embedding_job(dataset_id: str):
    ds = db_get_dataset(dataset_id)
    if not ds:
        return
    dataset_dir = DATASETS_DIR / dataset_id
    emb_dir = EMBEDDINGS_DIR / dataset_id
    emb_dir.mkdir(exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in exts]
    total = len(image_paths)
    log.info(f"[{dataset_id}] Embedding {total} images")
    db_update_dataset_fields(dataset_id, status="processing", total=total, processed=0)
    embeddings, metadata = [], []
    model = get_face_model()
    for i, item in enumerate(image_paths):
        try:
            img = cv2.imread(str(item))
            if img is None:
                continue
            rel_path = str(item.relative_to(dataset_dir))
            label = item.parent.name
            for face in model.get(img):
                emb = face.normed_embedding.astype("float32")
                bbox = [int(x) for x in face.bbox.tolist()]
                embeddings.append(emb)
                metadata.append({"image_path": rel_path, "label": label, "bbox": bbox})
        except Exception as exc:
            log.warning(f"[{dataset_id}] item {i}: {exc}")
        update_freq = 1 if total < 50 else 5
        if (i + 1) % update_freq == 0 or i == total - 1:
            db_update_dataset_fields(dataset_id, processed=i+1)
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
    # Unload model to free RAM
    if os.environ.get("UNLOAD_MODEL_AFTER_EMBED", "true").lower() == "true":
        global _face_model
        with _model_lock:
            _face_model = None
        import gc; gc.collect()

# ── Search index ──────────────────────────────────────────────────────────────
MAX_LOADED_INDEXES = 3
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
    return results

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Pixmatch Local", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith(".html") or path in ("/", ""):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"]        = "no-cache"
        response.headers["Expires"]       = "0"
    return response

@app.on_event("startup")
def on_startup():
    init_db()
    # Background: silently try to refresh last_validated_at with cloud.
    # This is best-effort only — failure never affects local functionality.
    def _bg_validate():
        act = get_activation()
        if not act:
            return
        try:
            result = validate_license_with_cloud(act["license_key"])
            if result.get("valid"):
                update_last_validated()
                log.info("License re-validated with cloud on startup (optional)")
        except Exception:
            pass  # Offline or server unreachable — no problem, app works anyway
    threading.Thread(target=_bg_validate, daemon=True).start()

# ── Activation endpoints ──────────────────────────────────────────────────────

@app.get("/api/activation/status")
def activation_status():
    """Check if the app is activated."""
    act = get_activation()
    if not act:
        return {"activated": False}
    expired = bool(act.get("expires_at") and time.time() > act["expires_at"])
    return {
        "activated":   not expired,
        "expired":     expired,
        "user_email":  act["user_email"],
        "user_name":   act["user_name"],
        "plan":        act["plan"],
        "expires_at":  act.get("expires_at"),
        "limits":      get_plan_limits(act["plan"]),
        "last_validated_at": act.get("last_validated_at"),
    }

@app.post("/api/activation/activate")
async def activate(request: Request):
    """Activate the app with a license key. Calls cloud to validate."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    key = (body.get("license_key") or "").strip().upper()
    if not key:
        raise HTTPException(400, "license_key is required.")

    # 1. Validate with cloud
    try:
        result = validate_license_with_cloud(key)
    except ValueError as e:
        raise HTTPException(403, str(e))
    except Exception as e:
        raise HTTPException(503, f"Could not reach activation server: {e}. Check your internet connection.")

    if not result.get("valid"):
        raise HTTPException(403, result.get("reason", "Invalid license key."))

    # 2. Fetch user info
    try:
        userinfo = fetch_userinfo_from_cloud(key)
    except Exception as e:
        raise HTTPException(503, f"Could not fetch account info: {e}")

    plan = result.get("plan", "personal_lite")
    limits = result.get("limits", {})
    grace_hours = result.get("offline_grace_hours", 72)
    expires_at = result.get("expires_at")

    save_activation(
        license_key=key,
        user_email=userinfo["email"],
        user_name=userinfo["name"],
        plan=plan,
        offline_grace_hours=grace_hours,
        expires_at=expires_at,
    )

    log.info(f"App activated: {userinfo['email']} plan={plan}")
    return {
        "ok":       True,
        "user_name": userinfo["name"],
        "user_email": userinfo["email"],
        "plan":     plan,
        "limits":   get_plan_limits(plan),
    }

@app.post("/api/activation/deactivate")
def deactivate():
    """Remove activation (for re-activation with a different key)."""
    conn = get_db()
    conn.execute("DELETE FROM activation")
    conn.commit()
    conn.close()
    return {"ok": True}

# ── Auth/me endpoint (mirrors cloud for frontend compatibility) ───────────────

@app.get("/api/auth/me")
def me(request: Request):
    act = require_activation(request)
    return {
        "id":    "local",
        "email": act["user_email"],
        "name":  act["user_name"],
        "plan":  act["plan"],
        "plan_interval": "local",
        "credits_paise": 0,
        "is_local": True,
    }

@app.post("/api/auth/logout")
def logout():
    # Local app: logout deactivates the session state but keeps activation
    return {"ok": True}

# ── Background task: compress then embed ─────────────────────────────────────

def compress_and_embed(dataset_id: str, dataset_dir: Path):
    """Run image compression followed by face embedding in the background thread."""
    try:
        compress_images_in_dir(dataset_dir)
    except Exception as exc:
        log.warning(f"[{dataset_id}] compress step failed: {exc}")
    run_embedding_job(dataset_id)

# ── Dataset endpoints ─────────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets(request: Request):
    act = require_activation(request)
    datasets = db_list_datasets()
    for ds_id, ds in datasets.items():
        ddir = DATASETS_DIR / ds_id
        ds["size_bytes"] = sum(p.stat().st_size for p in ddir.rglob("*") if p.is_file()) if ddir.exists() else 0
    return datasets

@app.post("/api/datasets/upload-zip")
async def upload_zip(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(default=""),
    group_id: str = Form(default=""),
):
    act = require_activation(request)
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file.")

    # Accept name / group_id from query params too (frontend sends them that way)
    qp = request.query_params
    if not name:
        name = qp.get("name", "")
    if not group_id:
        group_id = qp.get("group_id", "")

    limits = get_plan_limits(act["plan"])
    existing = db_list_datasets()
    if len(existing) >= limits["max_datasets"]:
        raise HTTPException(400, f"Dataset limit reached. Your plan allows {limits['max_datasets']} dataset(s). Delete one to add more.")

    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir()

    raw_bytes = await file.read()
    zip_path = dataset_dir / "upload.zip"
    zip_path.write_bytes(raw_bytes)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_dir)
    zip_path.unlink()

    img_count = count_images_in_dir(dataset_dir)
    if img_count > limits["max_images"]:
        import shutil; shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(400, f"Too many images. Your plan allows up to {limits['max_images']:,} images. This ZIP contains {img_count:,}.")

    ds = {
        "id": dataset_id, "name": name or file.filename.replace(".zip",""),
        "source": "zip", "folder_id": None, "status": "compressing",
        "total": 0, "processed": 0, "face_count": 0, "error": None,
        "group_id": group_id or None, "created_at": time.time(),
    }
    db_upsert_dataset(ds)
    # Run compress + embed together in the background so the response
    # returns immediately and the event loop is never blocked.
    background_tasks.add_task(compress_and_embed, dataset_id, dataset_dir)
    return {"dataset_id": dataset_id, "status": "compressing"}

@app.get("/api/datasets/{dataset_id}/status")
def dataset_status(dataset_id: str, request: Request):
    require_activation(request)
    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    return ds

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, request: Request):
    require_activation(request)
    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    import shutil
    for d in [DATASETS_DIR / dataset_id, EMBEDDINGS_DIR / dataset_id]:
        if d.exists():
            shutil.rmtree(d)
    with _index_cache_lock:
        _index_cache.pop(dataset_id, None)
    conn = get_db()
    conn.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/api/datasets/{dataset_id}/add-images")
async def add_images_to_dataset(
    dataset_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    act = require_activation(request)
    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    if ds["status"] != "ready":
        raise HTTPException(400, "Dataset must be 'ready' to add images.")
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file.")

    limits = get_plan_limits(act["plan"])
    dataset_dir = DATASETS_DIR / dataset_id
    temp_dir = dataset_dir / f"_temp_{int(time.time())}"
    temp_dir.mkdir()
    raw_bytes = await file.read()
    zip_path = temp_dir / "upload.zip"
    zip_path.write_bytes(raw_bytes)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(temp_dir)
    zip_path.unlink()

    existing_count = count_images_in_dir(dataset_dir)
    new_count = count_images_in_dir(temp_dir)
    if existing_count + new_count > limits["max_images"]:
        import shutil; shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(400, f"Image limit exceeded. Your plan allows {limits['max_images']:,} images per dataset.")

    import shutil
    for item in temp_dir.rglob("*"):
        if item.is_file():
            rel = item.relative_to(temp_dir)
            target = dataset_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(target))
    shutil.rmtree(temp_dir)
    db_update_dataset_fields(dataset_id, status="queued")
    background_tasks.add_task(run_embedding_job, dataset_id)
    return {"ok": True, "dataset_id": dataset_id, "status": "queued"}

# ── Share endpoints ───────────────────────────────────────────────────────────

@app.post("/api/shares")
async def create_share(request: Request):
    require_activation(request)
    try:
        body = await request.json() if request.headers.get("content-type","").startswith("application/json") else dict(await request.form())
    except Exception:
        raise HTTPException(400, "Invalid request body.")
    dataset_id = body.get("dataset_id")
    group_id   = body.get("group_id")
    watermark_text = body.get("watermark_text", "")
    if group_id and not dataset_id:
        conn = get_db()
        row = conn.execute("SELECT id FROM datasets WHERE group_id=? AND status='ready' LIMIT 1", (group_id,)).fetchone()
        conn.close()
        if not row:
            raise HTTPException(400, "No ready datasets in this group yet.")
        dataset_id = row["id"]
    if not dataset_id:
        raise HTTPException(422, "dataset_id or group_id is required.")
    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    conn = get_db()
    existing = conn.execute("SELECT share_id FROM shares WHERE dataset_id=?", (dataset_id,)).fetchone()
    if existing:
        if watermark_text is not None:
            conn.execute("UPDATE shares SET watermark_text=? WHERE share_id=?", (watermark_text, existing["share_id"]))
            conn.commit()
        conn.close()
        return {"share_id": existing["share_id"]}
    share_id = secrets.token_urlsafe(12)
    conn.execute(
        "INSERT INTO shares (share_id, dataset_id, dataset_name, watermark_text, view_count, download_count, created_at) VALUES (?,?,?,?,0,0,?)",
        (share_id, dataset_id, ds["name"], watermark_text or "", time.time())
    )
    conn.commit()
    conn.close()
    return {"share_id": share_id}

@app.delete("/api/shares/{share_id}")
def delete_share(share_id: str, request: Request):
    require_activation(request)
    conn = get_db()
    conn.execute("DELETE FROM shares WHERE share_id=?", (share_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.get("/api/shares/{share_id}")
def get_share(share_id: str):
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share link not found.")
    conn = get_db()
    conn.execute("UPDATE shares SET view_count=view_count+1, last_viewed_at=? WHERE share_id=?", (time.time(), share_id))
    conn.commit()
    conn.close()
    return share

# ── Search endpoints ──────────────────────────────────────────────────────────

@app.post("/api/shares/{share_id}/detect-faces")
async def detect_faces_in_selfie(share_id: str, file: UploadFile = File(...)):
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
        pad_x = int((x2-x1)*0.2); pad_y = int((y2-y1)*0.2)
        x1c = max(0,x1-pad_x); y1c = max(0,y1-pad_y)
        x2c = min(w,x2+pad_x); y2c = min(h,y2+pad_y)
        crop = img[y1c:y2c,x1c:x2c]
        thumb = cv2.resize(crop,(128,128))
        _,buf = cv2.imencode('.jpg',thumb,[int(cv2.IMWRITE_JPEG_QUALITY),85])
        b64 = base64.b64encode(buf.tobytes()).decode()
        face_crops.append({"index":i,"thumbnail":f"data:image/jpeg;base64,{b64}","embedding":faces_sorted[i].normed_embedding.astype("float32").tolist()})
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
        "latency_ms":    round((time.time()-t0)*1000,1),
        "dataset_id":    share["dataset_id"],
    }

# ── Image / Thumb serving ─────────────────────────────────────────────────────

@app.get("/api/image/{dataset_id}/{image_path:path}")
def serve_image(dataset_id: str, image_path: str):
    full_path = DATASETS_DIR / dataset_id / image_path
    if not full_path.exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(str(full_path))

THUMB_WIDTH = int(os.environ.get("THUMB_WIDTH", "400"))

@app.get("/api/thumb/{dataset_id}/{image_path:path}")
def serve_thumb(dataset_id: str, image_path: str):
    src_path = DATASETS_DIR / dataset_id / image_path
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
            img = cv2.resize(img, (THUMB_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return FileResponse(str(thumb_path), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=604800, immutable"})

# ── Event groups ──────────────────────────────────────────────────────────────

@app.get("/api/groups")
def list_groups(request: Request):
    require_activation(request)
    groups = db_list_groups()
    conn = get_db()
    for g in groups:
        row = conn.execute("SELECT COUNT(*) as n FROM datasets WHERE group_id=?", (g["id"],)).fetchone()
        g["dataset_count"] = row["n"] if row else 0
        share_row = conn.execute("""
            SELECT s.share_id, s.view_count, s.download_count
            FROM shares s JOIN datasets d ON s.dataset_id=d.id
            WHERE d.group_id=? LIMIT 1
        """, (g["id"],)).fetchone()
        g["share_id"]      = share_row["share_id"] if share_row else None
        g["view_count"]    = share_row["view_count"] if share_row else 0
        g["download_count"]= share_row["download_count"] if share_row else 0
    conn.close()
    return groups

@app.post("/api/groups")
async def create_group(request: Request):
    require_activation(request)
    body = await request.json()
    group_id = str(uuid.uuid4())[:12]
    conn = get_db()
    conn.execute("""
        INSERT INTO event_groups (id, name, description, watermark_text, event_type, event_date, created_at)
        VALUES (?,?,?,?,?,?,?)
    """, (group_id, body.get("name","Untitled"), body.get("description",""),
          body.get("watermark_text",""), body.get("event_type","other"),
          body.get("event_date"), time.time()))
    conn.commit()
    conn.close()
    return {"id": group_id}

@app.put("/api/groups/{group_id}")
async def update_group(group_id: str, request: Request):
    require_activation(request)
    body = await request.json()
    conn = get_db()
    conn.execute("""
        UPDATE event_groups SET name=?,description=?,watermark_text=?,event_type=?,event_date=?
        WHERE id=?
    """, (body.get("name"), body.get("description",""), body.get("watermark_text",""),
          body.get("event_type","other"), body.get("event_date"), group_id))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.delete("/api/groups/{group_id}")
def delete_group(group_id: str, request: Request):
    require_activation(request)
    conn = get_db()
    conn.execute("DELETE FROM event_groups WHERE id=?", (group_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

# ── QR Code ───────────────────────────────────────────────────────────────────

@app.get("/api/shares/{share_id}/qr")
def get_share_qr(share_id: str):
    import qrcode
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share not found.")
    share_url = f"http://localhost:8765/share.html?id={share_id}"
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(share_url)
    qr.make(fit=True)
    from PIL import Image as PILImage
    img = qr.make_image(fill_color="#1c1917", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    from fastapi.responses import Response as FResponse
    return FResponse(content=buf.getvalue(), media_type="image/png")

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    act = get_activation()
    return {
        "status":    "ok",
        "timestamp": time.time(),
        "mode":      "local",
        "activated": act is not None,
        "plan":      act["plan"] if act else None,
    }

# ── Stub endpoints (gracefully refuse cloud-only features) ────────────────────

@app.get("/api/billing/info")
def billing_info_stub(request: Request):
    act = require_activation(request)
    return {
        "plan":            act["plan"],
        "is_local":        True,
        "limits":          get_plan_limits(act["plan"]),
        "message":         "Billing is managed on the Lenstagram web platform.",
    }

# ── Static frontend ───────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    log.info("Serving frontend from %s", FRONTEND_DIR)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    log.warning("FRONTEND_DIR not found: %s — static files will not be served", FRONTEND_DIR)

# ── Entry point ───────────────────────────────────────────────────────────────

def launch():
    """Called by the launcher script / PyInstaller entry point."""
    import webbrowser, threading
    port = int(os.environ.get("PORT", "8765"))

    def _open_browser():
        time.sleep(1.5)
        act = get_activation()
        page = "admin.html" if act else "activate.html"
        webbrowser.open(f"http://localhost:{port}/{page}")
    
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    launch()