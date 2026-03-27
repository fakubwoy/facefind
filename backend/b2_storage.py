"""
b2_storage.py — Backblaze B2 storage layer for FaceFind
========================================================
Wraps all B2 operations so app.py can import clean helper functions.

Public bucket layout
--------------------
  datasets/{dataset_id}/{relative_image_path}   ← original photos
  thumbs/{dataset_id}/{relative_image_path}      ← cached thumbnails
  embeddings/{dataset_id}/embeddings.npy         ← numpy embedding matrix
  embeddings/{dataset_id}/metadata.pkl           ← face metadata pickle
  embeddings/{dataset_id}/face_index.faiss       ← FAISS index
  uploads/{filename}                             ← temp selfie uploads
  releases/facefind-selfhosted.zip               ← self-hosted executable

Environment variables required
-------------------------------
  B2_KEY_ID         — Backblaze application key ID
  B2_APPLICATION_KEY — Backblaze application key (secret)
  B2_BUCKET_NAME    — Bucket name (must already exist)

Optional
--------
  B2_ENDPOINT_URL   — Override endpoint (useful for testing with MinIO etc.)
                      Default: https://s3.{B2_REGION}.backblazeb2.com
  B2_REGION         — Bucket region, e.g. us-west-004 (default: us-west-004)
"""

import io
import os
import logging
import threading
from pathlib import Path
from typing import Optional, Iterator

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

log = logging.getLogger("facefind.b2")

# ── Config ────────────────────────────────────────────────────────────────────

B2_KEY_ID          = os.environ.get("B2_KEY_ID", "")
B2_APPLICATION_KEY = os.environ.get("B2_APPLICATION_KEY", "")
B2_BUCKET_NAME     = os.environ.get("B2_BUCKET_NAME", "")
B2_REGION          = os.environ.get("B2_REGION", "us-west-004")
B2_ENDPOINT_URL    = os.environ.get(
    "B2_ENDPOINT_URL",
    f"https://s3.{B2_REGION}.backblazeb2.com"
)

# ── S3-compatible client (singleton) ─────────────────────────────────────────

_client = None
_client_lock = threading.Lock()

def get_b2_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not B2_KEY_ID or not B2_APPLICATION_KEY or not B2_BUCKET_NAME:
                    raise RuntimeError(
                        "B2 is not configured. Set B2_KEY_ID, B2_APPLICATION_KEY, "
                        "and B2_BUCKET_NAME environment variables."
                    )
                _client = boto3.client(
                    "s3",
                    endpoint_url=B2_ENDPOINT_URL,
                    aws_access_key_id=B2_KEY_ID,
                    aws_secret_access_key=B2_APPLICATION_KEY,
                    config=Config(
                        signature_version="s3v4",
                        retries={"max_attempts": 3, "mode": "standard"},
                    ),
                )
                log.info(f"B2 client ready → bucket={B2_BUCKET_NAME} endpoint={B2_ENDPOINT_URL}")
    return _client


def b2_configured() -> bool:
    """Return True if all B2 env vars are set."""
    return bool(B2_KEY_ID and B2_APPLICATION_KEY and B2_BUCKET_NAME)


# ── Core upload / download helpers ───────────────────────────────────────────

def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Upload raw bytes to B2 at the given key."""
    get_b2_client().put_object(
        Bucket=B2_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    log.debug(f"B2 upload: {key} ({len(data):,} bytes)")


def upload_file(key: str, local_path: Path, content_type: str = "application/octet-stream") -> None:
    """Upload a local file to B2."""
    get_b2_client().upload_file(
        str(local_path),
        B2_BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    log.debug(f"B2 upload file: {local_path} → {key}")


def download_bytes(key: str) -> Optional[bytes]:
    """Download an object from B2 and return its bytes. Returns None if not found."""
    try:
        resp = get_b2_client().get_object(Bucket=B2_BUCKET_NAME, Key=key)
        return resp["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def download_to_file(key: str, local_path: Path) -> bool:
    """Download a B2 object to a local file. Returns False if key not found."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        get_b2_client().download_file(B2_BUCKET_NAME, key, str(local_path))
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return False
        raise


def object_exists(key: str) -> bool:
    """Return True if the key exists in the bucket."""
    try:
        get_b2_client().head_object(Bucket=B2_BUCKET_NAME, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404", "403"):
            return False
        raise


def delete_prefix(prefix: str) -> int:
    """Delete all objects with the given prefix. Returns number of objects deleted."""
    client = get_b2_client()
    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=B2_BUCKET_NAME, Prefix=prefix):
        objects = page.get("Contents", [])
        if not objects:
            continue
        client.delete_objects(
            Bucket=B2_BUCKET_NAME,
            Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
        )
        deleted += len(objects)
    log.info(f"B2 deleted {deleted} objects with prefix={prefix!r}")
    return deleted


def list_keys(prefix: str) -> list[str]:
    """Return all object keys under a prefix."""
    client = get_b2_client()
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=B2_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def stream_object(key: str, chunk_size: int = 8 * 1024 * 1024) -> Iterator[bytes]:
    """Yield chunks of a B2 object (for streaming large file downloads)."""
    resp = get_b2_client().get_object(Bucket=B2_BUCKET_NAME, Key=key)
    stream = resp["Body"]
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        yield chunk


def get_object_size(key: str) -> Optional[int]:
    """Return the size in bytes of a B2 object, or None if not found."""
    try:
        resp = get_b2_client().head_object(Bucket=B2_BUCKET_NAME, Key=key)
        return resp["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


# ── Public URL helper (works for public buckets) ──────────────────────────────

def public_url(key: str) -> str:
    """Return the public HTTPS URL for a key (bucket must be public)."""
    return f"{B2_ENDPOINT_URL}/{B2_BUCKET_NAME}/{key}"


# ── Domain-specific helpers ───────────────────────────────────────────────────

# -- Dataset images --

def dataset_image_key(dataset_id: str, relative_path: str) -> str:
    return f"datasets/{dataset_id}/{relative_path}"


def upload_dataset_image(dataset_id: str, relative_path: str, data: bytes) -> None:
    key = dataset_image_key(dataset_id, relative_path)
    ext = Path(relative_path).suffix.lower()
    ctype = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/webp"
    upload_bytes(key, data, content_type=ctype)


def download_dataset_image(dataset_id: str, relative_path: str) -> Optional[bytes]:
    return download_bytes(dataset_image_key(dataset_id, relative_path))


def delete_dataset_images(dataset_id: str) -> int:
    return delete_prefix(f"datasets/{dataset_id}/")


# -- Thumbnails --

def thumb_key(dataset_id: str, relative_path: str) -> str:
    return f"thumbs/{dataset_id}/{relative_path}"


def upload_thumb(dataset_id: str, relative_path: str, data: bytes) -> None:
    upload_bytes(thumb_key(dataset_id, relative_path), data, content_type="image/jpeg")


def download_thumb(dataset_id: str, relative_path: str) -> Optional[bytes]:
    return download_bytes(thumb_key(dataset_id, relative_path))


def delete_thumbs(dataset_id: str) -> int:
    return delete_prefix(f"thumbs/{dataset_id}/")


# -- FAISS / embeddings --

def embeddings_prefix(dataset_id: str) -> str:
    return f"embeddings/{dataset_id}/"


def upload_embedding_file(dataset_id: str, filename: str, data: bytes) -> None:
    """Upload one of: embeddings.npy, metadata.pkl, face_index.faiss"""
    upload_bytes(f"embeddings/{dataset_id}/{filename}", data)


def download_embedding_file(dataset_id: str, filename: str) -> Optional[bytes]:
    return download_bytes(f"embeddings/{dataset_id}/{filename}")


def delete_embeddings(dataset_id: str) -> int:
    return delete_prefix(embeddings_prefix(dataset_id))


def embeddings_exist(dataset_id: str) -> bool:
    return object_exists(f"embeddings/{dataset_id}/face_index.faiss")


# -- Selfie uploads (temp) --

def selfie_key(filename: str) -> str:
    return f"uploads/{filename}"


def upload_selfie(filename: str, data: bytes) -> None:
    upload_bytes(selfie_key(filename), data, content_type="image/jpeg")


def download_selfie(filename: str) -> Optional[bytes]:
    return download_bytes(selfie_key(filename))


def delete_selfie(filename: str) -> None:
    try:
        get_b2_client().delete_object(Bucket=B2_BUCKET_NAME, Key=selfie_key(filename))
    except Exception:
        pass


# -- Self-hosted executable --

RELEASE_KEY = "releases/facefind-selfhosted.zip"


def executable_exists() -> bool:
    return object_exists(RELEASE_KEY)


def stream_executable() -> Iterator[bytes]:
    return stream_object(RELEASE_KEY)


def get_executable_size() -> Optional[int]:
    return get_object_size(RELEASE_KEY)


# ── Migration helper ──────────────────────────────────────────────────────────

def migrate_local_to_b2(
    datasets_dir: Path,
    embeddings_dir: Path,
    thumbs_dir: Path,
    uploads_dir: Path,
    executable_path: Path,
    progress_callback=None,
) -> dict:
    """
    One-shot migration: copy everything from local Railway volume to B2.

    Returns a summary dict with counts per category.
    Called by POST /api/admin/migrate-to-b2 (admin-only endpoint).
    """
    summary = {
        "dataset_images": 0,
        "thumbs": 0,
        "embeddings": 0,
        "selfies": 0,
        "executable": False,
        "errors": [],
    }

    def _report(msg):
        log.info(f"[migrate] {msg}")
        if progress_callback:
            progress_callback(msg)

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # 1. Dataset images
    if datasets_dir.exists():
        for dataset_dir in datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_id = dataset_dir.name
            for img_path in dataset_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in image_exts:
                    rel = str(img_path.relative_to(dataset_dir))
                    try:
                        upload_dataset_image(dataset_id, rel, img_path.read_bytes())
                        summary["dataset_images"] += 1
                    except Exception as e:
                        summary["errors"].append(f"dataset img {img_path}: {e}")
            _report(f"Dataset {dataset_id}: {summary['dataset_images']} images so far")

    # 2. Thumbnails
    if thumbs_dir.exists():
        for dataset_dir in thumbs_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_id = dataset_dir.name
            for thumb_path in dataset_dir.rglob("*"):
                if thumb_path.is_file():
                    rel = str(thumb_path.relative_to(dataset_dir))
                    try:
                        upload_thumb(dataset_id, rel, thumb_path.read_bytes())
                        summary["thumbs"] += 1
                    except Exception as e:
                        summary["errors"].append(f"thumb {thumb_path}: {e}")

    # 3. Embeddings (npy, pkl, faiss)
    if embeddings_dir.exists():
        for emb_dir in embeddings_dir.iterdir():
            if not emb_dir.is_dir():
                continue
            dataset_id = emb_dir.name
            for f in emb_dir.iterdir():
                if f.is_file():
                    try:
                        upload_embedding_file(dataset_id, f.name, f.read_bytes())
                        summary["embeddings"] += 1
                    except Exception as e:
                        summary["errors"].append(f"embedding {f}: {e}")
            _report(f"Embeddings {dataset_id} uploaded")

    # 4. Selfie uploads
    if uploads_dir.exists():
        for f in uploads_dir.iterdir():
            if f.is_file():
                try:
                    upload_selfie(f.name, f.read_bytes())
                    summary["selfies"] += 1
                except Exception as e:
                    summary["errors"].append(f"selfie {f}: {e}")

    # 5. Self-hosted executable
    if executable_path.exists():
        try:
            upload_file(RELEASE_KEY, executable_path, content_type="application/zip")
            summary["executable"] = True
            _report("Executable uploaded")
        except Exception as e:
            summary["errors"].append(f"executable: {e}")

    _report(f"Migration complete: {summary}")
    return summary