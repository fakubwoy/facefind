"""
FaceFind – Backend API (Railway Edition)
- Postgres for metadata (datasets, shares)
- Redis for caching dataset status + search results
- Local filesystem volume for images + FAISS indexes
  (mount a Railway Volume at /data)
"""

import os, io, re, uuid, time, json, pickle, zipfile, threading
import urllib.request, urllib.parse, logging
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import psycopg2
import psycopg2.extras
import redis as redis_lib
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
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
FRONTEND_DIR   = Path(__file__).parent.parent / "frontend"

for d in [DATASETS_DIR, EMBEDDINGS_DIR, UPLOADS_DIR]:
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
                CREATE TABLE IF NOT EXISTS datasets (
                    id          TEXT PRIMARY KEY,
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

def db_list_datasets() -> dict:
    cached = cache_get("datasets:all")
    if cached:
        return cached
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM datasets ORDER BY created_at DESC")
            rows = cur.fetchall()
    result = {row["id"]: dict(row) for row in rows}
    cache_set("datasets:all", result, ttl=5)
    return result

def db_upsert_dataset(ds: dict):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO datasets (id, name, source, folder_id, status, total, processed, face_count, error, created_at)
                VALUES (%(id)s, %(name)s, %(source)s, %(folder_id)s, %(status)s, %(total)s, %(processed)s, %(face_count)s, %(error)s, %(created_at)s)
                ON CONFLICT (id) DO UPDATE SET
                    name=EXCLUDED.name, source=EXCLUDED.source, folder_id=EXCLUDED.folder_id,
                    status=EXCLUDED.status, total=EXCLUDED.total, processed=EXCLUDED.processed,
                    face_count=EXCLUDED.face_count, error=EXCLUDED.error
            """, {
                "id": ds["id"], "name": ds["name"], "source": ds.get("source","zip"),
                "folder_id": ds.get("folder_id"), "status": ds.get("status","queued"),
                "total": ds.get("total",0), "processed": ds.get("processed",0),
                "face_count": ds.get("face_count",0), "error": ds.get("error"),
                "created_at": ds.get("created_at", time.time()),
            })
        conn.commit()
    # Invalidate caches
    cache_delete(f"dataset:{ds['id']}")
    cache_delete("datasets:all")

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
    cache_delete("datasets:all")

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
                model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                model.prepare(ctx_id=-1, det_size=(640, 640))
                _face_model = model
    return _face_model

# ── Image helpers ─────────────────────────────────────────────────────────────

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image.")
    return img

def extract_embedding(img_bgr: np.ndarray):
    model = get_face_model()
    faces = model.get(img_bgr)
    if not faces:
        return None, []
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return best.normed_embedding.astype("float32"), faces

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

        if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
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

# ── Search ────────────────────────────────────────────────────────────────────

def search_in_dataset(dataset_id: str, query_emb: np.ndarray, top_k: int = 50):
    # Cache key = hex of first 16 bytes of embedding + dataset_id
    emb_key = f"search:{dataset_id}:{query_emb[:16].tobytes().hex()}"
    cached = cache_get(emb_key)
    if cached:
        return cached

    emb_dir    = EMBEDDINGS_DIR / dataset_id
    index_path = emb_dir / "face_index.faiss"
    meta_path  = emb_dir / "metadata.pkl"

    if not index_path.exists():
        return []

    import faiss
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

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
    db_update_dataset_fields(dataset_id, status="downloading")
    api_key    = os.environ.get("GOOGLE_API_KEY", "").strip()
    downloaded = 0
    last_error = ""

    if api_key:
        try:
            files = _gdrive_list_folder(folder_id, api_key, dataset_id)
            log.info(f"[{dataset_id}] Downloading {len(files)} images via Drive API")
            for f in files:
                out_path = dest_dir / f["name"]
                log.info(f"[{dataset_id}] Downloading ({downloaded+1}/{len(files)}) {f['name']}")
                try:
                    _gdrive_download_file(f["id"], out_path)
                    downloaded += 1
                except Exception as exc:
                    log.warning(f"[{dataset_id}] Skipped {f['name']}: {exc}")
                if downloaded % 5 == 0:
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

# ── Dataset endpoints ─────────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets():
    return db_list_datasets()

@app.post("/api/datasets/upload-zip")
async def upload_zip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(default=""),
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file.")

    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir()

    zip_path = dataset_dir / "upload.zip"
    zip_path.write_bytes(await file.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_dir)
    zip_path.unlink()

    ds = {
        "id": dataset_id, "name": name or file.filename.replace(".zip",""),
        "source": "zip", "folder_id": None,
        "status": "queued", "total": 0, "processed": 0,
        "face_count": 0, "error": None, "created_at": time.time(),
    }
    db_upsert_dataset(ds)
    background_tasks.add_task(run_embedding_job, dataset_id)
    return {"dataset_id": dataset_id, "status": "queued"}

@app.post("/api/datasets/gdrive")
async def use_gdrive_folder(
    background_tasks: BackgroundTasks,
    folder_url: str = Form(...),
    name: str = Form(default=""),
):
    folder_id = extract_gdrive_folder_id(folder_url)
    if not folder_id:
        raise HTTPException(400, "Could not extract a folder ID from the provided URL.")

    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir()

    ds = {
        "id": dataset_id, "name": name or f"Drive Folder ({folder_id[:8]}…)",
        "source": "gdrive", "folder_id": folder_id,
        "status": "downloading", "total": 0, "processed": 0,
        "face_count": 0, "error": None, "created_at": time.time(),
    }
    db_upsert_dataset(ds)
    background_tasks.add_task(download_gdrive_folder, folder_id, dataset_dir, dataset_id)
    return {"dataset_id": dataset_id, "status": "downloading"}

@app.post("/api/datasets/lfw")
async def use_lfw_dataset(background_tasks: BackgroundTasks):
    lfw_path = DATASETS_DIR / "lfw"
    if not lfw_path.exists():
        raise HTTPException(404, "LFW dataset not found.")
    dataset_id = "lfw"
    existing = db_get_dataset(dataset_id)
    if existing and existing["status"] == "ready":
        return {"dataset_id": dataset_id, "status": "already_ready"}
    ds = {
        "id": dataset_id, "name": "LFW (Labeled Faces in the Wild)",
        "source": "lfw", "folder_id": None,
        "status": "queued", "total": 0, "processed": 0,
        "face_count": 0, "error": None, "created_at": time.time(),
    }
    db_upsert_dataset(ds)
    background_tasks.add_task(run_embedding_job, dataset_id)
    return {"dataset_id": dataset_id, "status": "queued"}

@app.get("/api/datasets/{dataset_id}/status")
def dataset_status(dataset_id: str):
    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    return ds

# ── Share endpoints ───────────────────────────────────────────────────────────

@app.post("/api/shares")
def create_share(dataset_id: str = Form(...)):
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

@app.post("/api/shares/{share_id}/search")
async def search_by_selfie(share_id: str, file: UploadFile = File(...)):
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
    query_emb, faces = extract_embedding(img)
    if query_emb is None:
        return JSONResponse({"face_detected": False, "matches": [], "latency_ms": round((time.time()-t0)*1000,1)})

    results = search_in_dataset(share["dataset_id"], query_emb, top_k=100)
    return {
        "face_detected": True,
        "num_faces":     len(faces),
        "matches":       results,
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

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    r = get_redis()
    return {
        "status":    "ok",
        "timestamp": time.time(),
        "redis":     "connected" if r else "disabled",
        "db":        "postgres",
    }

# ── Frontend static files ─────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))