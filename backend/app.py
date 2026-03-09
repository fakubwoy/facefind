"""
FaceFind – Backend API
FastAPI server that handles:
  - Dataset upload & management
  - Embedding generation (InsightFace ArcFace)
  - Shareable link generation
  - Face search against embedded dataset
"""

import os
import io
import re
import uuid
import time
import json
import pickle
import shutil
import zipfile
import threading
import urllib.request
import urllib.parse
import logging
from pathlib import Path
from typing import Optional, List

# ── Load .env file if present ─────────────────────────────────────────────────
# Allows setting GOOGLE_API_KEY in a .env file next to app.py or in project root
def _load_dotenv():
    candidates = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
            break  # only load the first .env found

_load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("facefind")

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATASETS_DIR   = BASE_DIR / "datasets"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
UPLOADS_DIR    = BASE_DIR / "uploads"
SHARES_DIR     = BASE_DIR / "shares"
FRONTEND_DIR   = BASE_DIR.parent / "frontend"

for d in [DATASETS_DIR, EMBEDDINGS_DIR, UPLOADS_DIR, SHARES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── State store (in-memory + disk) ───────────────────────────────────────────
DATASETS_META = BASE_DIR / "datasets_meta.json"
SHARES_META   = BASE_DIR / "shares_meta.json"


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2))


# ── InsightFace model (lazy) ──────────────────────────────────────────────────
_face_model = None
_model_lock = threading.Lock()


def get_face_model():
    global _face_model
    if _face_model is None:
        with _model_lock:
            if _face_model is None:
                from insightface.app import FaceAnalysis
                model = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"]
                )
                model.prepare(ctx_id=-1, det_size=(640, 640))
                _face_model = model
    return _face_model


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── Background embedding job ──────────────────────────────────────────────────

def run_embedding_job(dataset_id: str):
    """Iterates all images in a dataset folder, extracts embeddings, saves index."""
    datasets = load_json(DATASETS_META)
    ds = datasets.get(dataset_id)
    if not ds:
        return

    dataset_dir = DATASETS_DIR / dataset_id
    emb_dir     = EMBEDDINGS_DIR / dataset_id
    emb_dir.mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [
        p for p in dataset_dir.rglob("*")
        if p.suffix.lower() in exts
    ]

    log.info(f"[{dataset_id}] Starting embedding of {len(image_paths)} images")

    datasets[dataset_id]["status"]    = "processing"
    datasets[dataset_id]["total"]     = len(image_paths)
    datasets[dataset_id]["processed"] = 0
    save_json(DATASETS_META, datasets)

    embeddings = []
    metadata   = []
    model = get_face_model()

    for i, img_path in enumerate(image_paths):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning(f"[{dataset_id}] Could not read image: {img_path.name}")
                continue
            faces = model.get(img)
            for face in faces:
                emb = face.normed_embedding.astype("float32")
                bbox = [int(x) for x in face.bbox.tolist()]
                embeddings.append(emb)
                metadata.append({
                    "image_path": str(img_path.relative_to(dataset_dir)),
                    "abs_path":   str(img_path),
                    "label":      img_path.parent.name,
                    "bbox":       bbox,
                })
        except Exception as exc:
            log.warning(f"[{dataset_id}] Error on {img_path.name}: {exc}")

        if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
            datasets = load_json(DATASETS_META)
            datasets[dataset_id]["processed"] = i + 1
            save_json(DATASETS_META, datasets)

    if embeddings:
        emb_matrix = np.stack(embeddings).astype("float32")
        np.save(str(emb_dir / "embeddings.npy"), emb_matrix)
        with open(emb_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        import faiss
        dim   = emb_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_matrix)
        faiss.write_index(index, str(emb_dir / "face_index.faiss"))

    datasets = load_json(DATASETS_META)
    datasets[dataset_id]["status"]     = "ready"
    datasets[dataset_id]["face_count"] = len(embeddings)
    save_json(DATASETS_META, datasets)
    log.info(f"[{dataset_id}] Done. {len(embeddings)} face embeddings indexed.")


def search_in_dataset(dataset_id: str, query_emb: np.ndarray, top_k: int = 50):
    emb_dir = EMBEDDINGS_DIR / dataset_id
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

    results = []
    seen_images = set()
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or float(score) < 0.30:
            continue
        meta = metadata[idx]
        img_key = meta["image_path"]
        if img_key in seen_images:
            continue
        seen_images.add(img_key)
        results.append({
            "score":      round(float(score), 4),
            "image_path": meta["image_path"],
            "label":      meta["label"],
            "bbox":       meta["bbox"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="FaceFind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dataset endpoints ─────────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets():
    return load_json(DATASETS_META)


@app.post("/api/datasets/upload-zip")
async def upload_zip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(default=""),
):
    """Upload a ZIP of images as a new dataset."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file.")

    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir()

    zip_path = dataset_dir / "upload.zip"
    contents = await file.read()
    zip_path.write_bytes(contents)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_dir)
    zip_path.unlink()

    ds_name = name or file.filename.replace(".zip", "")
    datasets = load_json(DATASETS_META)
    datasets[dataset_id] = {
        "id":         dataset_id,
        "name":       ds_name,
        "status":     "queued",
        "total":      0,
        "processed":  0,
        "face_count": 0,
        "created_at": time.time(),
    }
    save_json(DATASETS_META, datasets)

    background_tasks.add_task(run_embedding_job, dataset_id)
    return {"dataset_id": dataset_id, "status": "queued"}


def extract_gdrive_folder_id(url: str) -> Optional[str]:
    """Extract folder ID from various Google Drive folder URL formats."""
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
    """
    Download a single public Google Drive file using the uc?export=download endpoint.
    Handles the virus-scan confirmation redirect that Google adds for larger files.
    """
    # Initial request — works for small files directly
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()

    # Google redirects large files through a confirm page (returns HTML, not image)
    if b"confirm=" in body and b"<html" in body[:500].lower():
        # Extract the confirm token from the HTML/redirect
        import re as _re
        m = _re.search(rb'confirm=([0-9A-Za-z_\-]+)', body)
        if m:
            confirm = m.group(1).decode()
            url2 = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={confirm}"
            req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                body = resp2.read()

    out_path.write_bytes(body)


def _gdrive_list_folder(folder_id: str, api_key: str, dataset_id: str) -> list:
    """
    List all image files in a public Google Drive folder using the Drive API v3.
    Handles pagination — no file count limit.
    Returns list of {id, name} dicts for image files only.
    """
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = []
    page_token = None

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name,mimeType)",
            "key": api_key,
            "pageSize": "1000",
        }
        if page_token:
            params["pageToken"] = page_token

        list_url = (
            "https://www.googleapis.com/drive/v3/files?"
            + urllib.parse.urlencode(params)
        )
        log.info(f"[{dataset_id}] Listing Drive folder page (pageToken={page_token!r})")
        req = urllib.request.Request(list_url, headers={"User-Agent": "FaceFind/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())

        if "error" in data:
            raise RuntimeError(f"Drive API error: {data['error'].get('message', data['error'])}")

        page_files = [
            f for f in data.get("files", [])
            if Path(f["name"]).suffix.lower() in img_exts
        ]
        all_files.extend(page_files)
        log.info(f"[{dataset_id}] Page returned {len(page_files)} image(s), total so far: {len(all_files)}")

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return all_files


def _gdrive_api_download(folder_id: str, dest_dir: Path, dataset_id: str, api_key: str) -> int:
    """
    List files via Drive API v3, then download each via the public uc?export=download
    URL (no OAuth needed — works for any publicly shared file).
    Returns number of files downloaded.
    """
    files = _gdrive_list_folder(folder_id, api_key, dataset_id)
    log.info(f"[{dataset_id}] Total images to download: {len(files)}")
    downloaded = 0

    for f in files:
        out_path = dest_dir / f["name"]
        log.info(f"[{dataset_id}] Downloading ({downloaded+1}/{len(files)}) {f['name']}")
        try:
            _gdrive_download_file(f["id"], out_path)
            downloaded += 1
        except Exception as exc:
            log.warning(f"[{dataset_id}] Failed to download {f['name']}: {exc}")

        if downloaded % 5 == 0:
            ds2 = load_json(DATASETS_META)
            ds2[dataset_id]["processed"] = downloaded
            save_json(DATASETS_META, ds2)

    return downloaded


def download_gdrive_folder(folder_id: str, dest_dir: Path, dataset_id: str):
    """
    Downloads all image files from a public Google Drive folder.

    Strategy (in order):
      1. Google Drive API v3 with GOOGLE_API_KEY  ← most reliable
      2. gdown fallback (if no API key set)
    """
    datasets = load_json(DATASETS_META)
    datasets[dataset_id]["status"] = "downloading"
    save_json(DATASETS_META, datasets)

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    downloaded = 0
    last_error = ""

    # ── Strategy 1: Drive API v3 with API key ─────────────────────────────────
    if api_key:
        log.info(f"[{dataset_id}] Using Google Drive API v3 (API key found)")
        try:
            downloaded = _gdrive_api_download(folder_id, dest_dir, dataset_id, api_key)
            log.info(f"[{dataset_id}] Drive API download complete: {downloaded} file(s)")
        except Exception as exc:
            last_error = str(exc)
            log.error(f"[{dataset_id}] Drive API error: {exc}")
    else:
        log.warning(
            f"[{dataset_id}] GOOGLE_API_KEY not set — "
            "falling back to gdown (less reliable for large/public folders)"
        )

    # ── Strategy 2: gdown fallback ────────────────────────────────────────────
    if downloaded == 0:
        try:
            import gdown  # type: ignore
            log.info(f"[{dataset_id}] Trying gdown.download_folder …")
            gdown.download_folder(
                id=folder_id,
                output=str(dest_dir),
                quiet=False,
                use_cookies=False,
            )
            # Count what gdown actually placed there
            img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            downloaded = sum(
                1 for p in dest_dir.rglob("*") if p.suffix.lower() in img_exts
            )
            log.info(f"[{dataset_id}] gdown placed {downloaded} image(s)")
        except ImportError:
            log.warning(f"[{dataset_id}] gdown not installed")
        except Exception as exc:
            last_error = str(exc)
            log.error(f"[{dataset_id}] gdown error: {exc}")

    # ── Give up ───────────────────────────────────────────────────────────────
    if downloaded == 0:
        msg = (
            "Could not download any images from the Google Drive folder. "
        )
        if not api_key:
            msg += (
                "No GOOGLE_API_KEY was found. "
                "Please set it in a .env file next to app.py or export it before starting: "
                "export GOOGLE_API_KEY=AIzaSy... "
                "Get a free key at https://console.cloud.google.com/ → APIs & Services → Credentials. "
                "Enable the 'Google Drive API' for your project first. "
            )
        if last_error:
            msg += f"Last error: {last_error}"

        log.error(f"[{dataset_id}] {msg}")
        datasets = load_json(DATASETS_META)
        datasets[dataset_id]["status"] = "error"
        datasets[dataset_id]["error"]  = msg
        save_json(DATASETS_META, datasets)
        return

    datasets = load_json(DATASETS_META)
    datasets[dataset_id]["status"] = "queued"
    save_json(DATASETS_META, datasets)
    run_embedding_job(dataset_id)


@app.post("/api/datasets/gdrive")
async def use_gdrive_folder(
    background_tasks: BackgroundTasks,
    folder_url: str = Form(...),
    name: str = Form(default=""),
):
    """Link a public Google Drive folder as a dataset."""
    folder_id = extract_gdrive_folder_id(folder_url)
    if not folder_id:
        raise HTTPException(400, "Could not extract a folder ID from the provided URL.")

    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir()

    ds_name = name or f"Drive Folder ({folder_id[:8]}…)"
    datasets = load_json(DATASETS_META)
    datasets[dataset_id] = {
        "id":          dataset_id,
        "name":        ds_name,
        "source":      "gdrive",
        "folder_id":   folder_id,
        "status":      "downloading",
        "total":       0,
        "processed":   0,
        "face_count":  0,
        "created_at":  time.time(),
    }
    save_json(DATASETS_META, datasets)

    background_tasks.add_task(download_gdrive_folder, folder_id, dataset_dir, dataset_id)
    return {"dataset_id": dataset_id, "status": "downloading"}


@app.post("/api/datasets/lfw")
async def use_lfw_dataset(background_tasks: BackgroundTasks):
    """Register the pre-existing LFW dataset (if present in datasets/lfw folder)."""
    lfw_path = DATASETS_DIR / "lfw"
    if not lfw_path.exists():
        raise HTTPException(404, "LFW dataset not found. Run scripts/download_lfw.sh first.")

    dataset_id = "lfw"
    datasets = load_json(DATASETS_META)

    if dataset_id in datasets and datasets[dataset_id]["status"] == "ready":
        return {"dataset_id": dataset_id, "status": "already_ready"}

    datasets[dataset_id] = {
        "id":         dataset_id,
        "name":       "LFW (Labeled Faces in the Wild)",
        "status":     "queued",
        "total":      0,
        "processed":  0,
        "face_count": 0,
        "created_at": time.time(),
    }
    save_json(DATASETS_META, datasets)
    background_tasks.add_task(run_embedding_job, dataset_id)
    return {"dataset_id": dataset_id, "status": "queued"}


@app.get("/api/datasets/{dataset_id}/status")
def dataset_status(dataset_id: str):
    datasets = load_json(DATASETS_META)
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found.")
    return datasets[dataset_id]


# ── Share endpoints ───────────────────────────────────────────────────────────

@app.post("/api/shares")
def create_share(dataset_id: str = Form(...)):
    datasets = load_json(DATASETS_META)
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found.")
    if datasets[dataset_id]["status"] != "ready":
        raise HTTPException(400, "Dataset is not ready yet.")

    share_id = str(uuid.uuid4())[:12]
    shares = load_json(SHARES_META)
    shares[share_id] = {
        "share_id":   share_id,
        "dataset_id": dataset_id,
        "created_at": time.time(),
        "dataset_name": datasets[dataset_id]["name"],
    }
    save_json(SHARES_META, shares)
    return {"share_id": share_id}


@app.get("/api/shares/{share_id}")
def get_share(share_id: str):
    shares = load_json(SHARES_META)
    if share_id not in shares:
        raise HTTPException(404, "Share link not found.")
    return shares[share_id]


# ── Search endpoint ───────────────────────────────────────────────────────────

@app.post("/api/shares/{share_id}/search")
async def search_by_selfie(share_id: str, file: UploadFile = File(...)):
    shares = load_json(SHARES_META)
    if share_id not in shares:
        raise HTTPException(404, "Share link not found.")

    dataset_id = shares[share_id]["dataset_id"]
    datasets   = load_json(DATASETS_META)
    if datasets.get(dataset_id, {}).get("status") != "ready":
        raise HTTPException(400, "Dataset not ready.")

    contents = await file.read()
    try:
        img = decode_image(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))

    t0 = time.time()
    query_emb, faces = extract_embedding(img)

    if query_emb is None:
        return JSONResponse({
            "face_detected": False,
            "matches": [],
            "latency_ms": round((time.time()-t0)*1000, 1),
        })

    results = search_in_dataset(dataset_id, query_emb, top_k=100)
    latency = round((time.time()-t0)*1000, 1)

    return {
        "face_detected": True,
        "num_faces": len(faces),
        "matches": results,
        "latency_ms": latency,
        "dataset_id": dataset_id,
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
    return {"status": "ok", "timestamp": time.time()}


# ── Frontend static files ─────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)