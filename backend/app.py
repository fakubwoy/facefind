"""
Pixmatch – Backend API (Railway Edition)
- Postgres for metadata (datasets, shares, license keys, download tokens)
- Redis for caching dataset status + search results
- Local filesystem volume for images + FAISS indexes
- Gmail API for email OTP verification (works on Railway)
"""

import os, io, re, uuid, time, json, pickle, zipfile, threading, hashlib, secrets, base64
import smtplib, random, hmac
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

# Path to the self-hosted executable ZIP served for download
EXECUTABLE_PATH = Path(os.environ.get("EXECUTABLE_PATH", "/data/releases/facefind-selfhosted.zip"))

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
                    dataset_id   TEXT NOT NULL UNIQUE,
                    dataset_name TEXT,
                    created_at   DOUBLE PRECISION
                );
            """)
            # Migration: add user_id to datasets table if it doesn't exist yet
            cur.execute("""
                ALTER TABLE datasets ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT '';
            """)
            # Migration: add plan column to users
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS plan TEXT DEFAULT 'free';
            """)
            # Migration: billing cycle tracking + credits
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_cycle_start DOUBLE PRECISION DEFAULT NULL;
            """)
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS credits_paise INT DEFAULT 0;
            """)
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS scheduled_downgrade TEXT DEFAULT NULL;
            """)
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS scheduled_downgrade_at DOUBLE PRECISION DEFAULT NULL;
            """)
            # Migration: track target interval for scheduled interval switches
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS scheduled_downgrade_interval TEXT DEFAULT NULL;
            """)
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS loyalty_discount_used BOOLEAN DEFAULT FALSE;
            """)
            # Migration: billing interval (monthly vs annual)
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_interval TEXT DEFAULT 'monthly';
            """)
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS referred_by TEXT DEFAULT NULL;
            """)
            # Referral credits ledger
            cur.execute("""
                CREATE TABLE IF NOT EXISTS referral_credits (
                    id           TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    from_user_id TEXT NOT NULL,
                    amount_paise INT NOT NULL,
                    reason       TEXT,
                    created_at   DOUBLE PRECISION
                );
            """)
            # Migration: backfill plan_cycle_start for existing paid users who have none.
            # Uses their most recent paid order's created_at as a best estimate.
            cur.execute("""
                UPDATE users u
                SET plan_cycle_start = sub.latest_order
                FROM (
                    SELECT user_id, MAX(created_at) AS latest_order
                    FROM razorpay_orders
                    WHERE status = 'paid'
                    GROUP BY user_id
                ) sub
                WHERE u.id = sub.user_id
                  AND u.plan != 'free'
                  AND u.plan_cycle_start IS NULL;
            """)
            cur.execute("""
                ALTER TABLE razorpay_orders ADD COLUMN IF NOT EXISTS discount_code TEXT DEFAULT NULL;
            """)
            # Rate-limit table for discount code validation attempts
            cur.execute("""
                CREATE TABLE IF NOT EXISTS discount_validate_attempts (
                    id         TEXT PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    attempted_at DOUBLE PRECISION
                );
            """)
            # Migration: add previous_plan + proration fields to razorpay_orders
            cur.execute("""
                ALTER TABLE razorpay_orders ADD COLUMN IF NOT EXISTS previous_plan TEXT DEFAULT NULL;
            """)
            cur.execute("""
                ALTER TABLE razorpay_orders ADD COLUMN IF NOT EXISTS plan_interval TEXT DEFAULT 'monthly';
            """)
            cur.execute("""
                ALTER TABLE razorpay_orders ADD COLUMN IF NOT EXISTS credit_applied_paise INT DEFAULT 0;
            """)
            cur.execute("""
                ALTER TABLE razorpay_orders ADD COLUMN IF NOT EXISTS credit_applied_paise INT DEFAULT 0;
            """)
            # Migration: enforce one share per dataset (deduplicate first, then add constraint)
            cur.execute("""
                DELETE FROM shares s1
                USING shares s2
                WHERE s1.created_at < s2.created_at
                  AND s1.dataset_id = s2.dataset_id;
            """)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'shares_dataset_id_unique'
                    ) THEN
                        ALTER TABLE shares ADD CONSTRAINT shares_dataset_id_unique UNIQUE (dataset_id);
                    END IF;
                END$$;
            """)
            # License keys table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS license_keys (
                    key              TEXT PRIMARY KEY,
                    user_id          TEXT NOT NULL,
                    plan             TEXT NOT NULL,
                    created_at       DOUBLE PRECISION,
                    expires_at       DOUBLE PRECISION,
                    revoked          BOOLEAN DEFAULT FALSE,
                    activations      INT DEFAULT 0,
                    max_activations  INT DEFAULT 3,
                    last_seen_at     DOUBLE PRECISION,
                    last_seen_ip     TEXT
                );
            """)
            # Download tokens table (short-lived one-use links)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS download_tokens (
                    token       TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    created_at  DOUBLE PRECISION,
                    expires_at  DOUBLE PRECISION,
                    used        BOOLEAN DEFAULT FALSE
                );
            """)
            # Razorpay orders table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS razorpay_orders (
                    order_id        TEXT PRIMARY KEY,
                    user_id         TEXT NOT NULL,
                    plan            TEXT NOT NULL,
                    amount_paise    INT NOT NULL,
                    status          TEXT DEFAULT 'created',
                    payment_id      TEXT,
                    created_at      DOUBLE PRECISION
                );
            """)
            # ── NEW: Event Groups table ─────────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS event_groups (
                    id            TEXT PRIMARY KEY,
                    user_id       TEXT NOT NULL,
                    name          TEXT NOT NULL,
                    description   TEXT DEFAULT '',
                    cover_image   TEXT DEFAULT NULL,
                    watermark_text TEXT DEFAULT NULL,
                    created_at    DOUBLE PRECISION
                );
            """)
            # Migration: ensure datasets table has group_id column
            cur.execute("""
                ALTER TABLE datasets ADD COLUMN IF NOT EXISTS group_id TEXT DEFAULT NULL;
            """)
            # Migration: shares table — add watermark, analytics, qr fields
            cur.execute("""
                ALTER TABLE shares ADD COLUMN IF NOT EXISTS watermark_text TEXT DEFAULT NULL;
            """)
            cur.execute("""
                ALTER TABLE shares ADD COLUMN IF NOT EXISTS view_count INT DEFAULT 0;
            """)
            cur.execute("""
                ALTER TABLE shares ADD COLUMN IF NOT EXISTS download_count INT DEFAULT 0;
            """)
            cur.execute("""
                ALTER TABLE shares ADD COLUMN IF NOT EXISTS last_viewed_at DOUBLE PRECISION DEFAULT NULL;
            """)
            # ── NEW: Discount codes table ────────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS discount_codes (
                    code            TEXT PRIMARY KEY,
                    discount_pct    INT NOT NULL,
                    interval        TEXT DEFAULT 'both',
                    max_uses        INT DEFAULT 1,
                    use_count       INT DEFAULT 0,
                    expires_at      DOUBLE PRECISION DEFAULT NULL,
                    created_by      TEXT DEFAULT 'admin',
                    created_at      DOUBLE PRECISION
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS discount_code_uses (
                    id          TEXT PRIMARY KEY,
                    code        TEXT NOT NULL,
                    user_id     TEXT NOT NULL,
                    used_at     DOUBLE PRECISION,
                    order_id    TEXT DEFAULT NULL
                );
            """)
            # Share analytics log (optional detail)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS share_analytics (
                    id          TEXT PRIMARY KEY,
                    share_id    TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    ip_hash     TEXT,
                    created_at  DOUBLE PRECISION
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

# ── Razorpay config ───────────────────────────────────────────────────────────
RAZORPAY_KEY_ID     = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")

# Amount in paise (INR × 100), plan key → monthly amount
PLAN_PRICES_PAISE = {
    "personal_lite":  9900,    # ₹99
    "personal_pro":   19900,   # ₹199
    "personal_max":   34900,   # ₹349
    "photo_starter":  59900,   # ₹599
    "photo_pro":      149900,  # ₹1,499
}

# Annual prices (roughly 2 months free — ~17% discount)
PLAN_PRICES_ANNUAL_PAISE = {
    "personal_lite":  99000,   # ₹990/yr  (saves ₹198)
    "personal_pro":   199000,  # ₹1,990/yr (saves ₹398)
    "personal_max":   349000,  # ₹3,490/yr (saves ₹698)
    "photo_starter":  599000,  # ₹5,990/yr (saves ₹1,198)
    "photo_pro":      1499000, # ₹14,990/yr (saves ₹2,998)
}

def get_plan_price(plan: str, interval: str = "monthly") -> int:
    """Return the price in paise for a given plan and billing interval."""
    if interval == "annual":
        return PLAN_PRICES_ANNUAL_PAISE.get(plan, 0)
    return PLAN_PRICES_PAISE.get(plan, 0)

def get_billing_period_days(interval: str = "monthly") -> int:
    """Return the number of days in a billing period."""
    return 365 if interval == "annual" else 30

def get_plan_limits(user: dict) -> dict:
    plan = user.get("plan", "free") or "free"
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

# ── Plan ordering for upgrade/downgrade direction ─────────────────────────────
PLAN_ORDER = ["free", "personal_lite", "personal_pro", "personal_max", "photo_starter", "photo_pro"]

def plan_rank(plan: str) -> int:
    try:
        return PLAN_ORDER.index(plan)
    except ValueError:
        return 0

def compute_proration_credit(current_plan: str, cycle_start: float, interval: str = "monthly") -> int:
    """
    Calculate unused credit (in paise) remaining in the current billing period.
    Supports both monthly (30-day) and annual (365-day) billing cycles.
    Returns 0 if cycle_start is unknown or plan is free.
    """
    if not cycle_start or current_plan == "free" or current_plan not in PLAN_PRICES_PAISE:
        return 0
    price = get_plan_price(current_plan, interval)
    period_days = get_billing_period_days(interval)
    elapsed_seconds = time.time() - cycle_start
    days_elapsed = min(elapsed_seconds / 86400, period_days)
    days_remaining = max(period_days - days_elapsed, 0)
    credit = int((days_remaining / period_days) * price)
    return credit

def compute_upgrade_charge(current_plan: str, target_plan: str, cycle_start: float, credits_paise: int, current_interval: str = "monthly", target_interval: str = "monthly") -> dict:
    """
    Compute what a user actually pays to upgrade.
    - Prorates unused time on current plan as credit (respects monthly vs annual)
    - Also applies any stored account credits (referrals, goodwill)
    - Never charges less than ₹1 (100 paise) — Razorpay minimum
    Returns dict with: full_price, proration_credit, account_credit, total_credit, charge, is_free_upgrade
    """
    full_price = get_plan_price(target_plan, target_interval)
    proration_credit = compute_proration_credit(current_plan, cycle_start, current_interval)
    total_credit = min(proration_credit + (credits_paise or 0), full_price)
    charge = max(full_price - total_credit, 0)
    return {
        "full_price_paise":       full_price,
        "proration_credit_paise": proration_credit,
        "account_credit_paise":   credits_paise or 0,
        "total_credit_paise":     total_credit,
        "charge_paise":           charge,
        "is_free_upgrade":        charge == 0,
    }

def apply_loyalty_discount(user: dict, target_plan: str, target_interval: str = "monthly") -> int:
    """
    If user has been on any paid plan ≥ 60 days without a downgrade,
    and hasn't used a loyalty discount before, give 20% off first month of next tier.
    Returns discount in paise (0 if not eligible).
    """
    if user.get("loyalty_discount_used"):
        return 0
    cycle_start = user.get("plan_cycle_start")
    if not cycle_start:
        return 0
    days_on_plan = (time.time() - cycle_start) / 86400
    if days_on_plan < 60:
        return 0
    current_plan = user.get("plan", "free")
    if current_plan == "free":
        return 0
    if plan_rank(target_plan) <= plan_rank(current_plan):
        return 0
    full_price = get_plan_price(target_plan, target_interval)
    return int(full_price * 0.20)  # 20% off

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


# ── QR Code generation ────────────────────────────────────────────────────────

def generate_qr_code_png(data: str) -> bytes:
    """Generate a QR code PNG as bytes for the given URL/data."""
    import qrcode
    from PIL import Image as PILImage
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1c1917", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Watermark helper ──────────────────────────────────────────────────────────

def apply_watermark(image_bytes: bytes, watermark_text: str) -> bytes:
    """
    Overlay studio name as a watermark at bottom-left of image.
    Uses Pillow. Raises on failure so the caller can log and handle it.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size

    txt_layer = PILImage.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)

    font_size = max(28, int(h * 0.045))
    font = None

    # Broader list covering Debian, Ubuntu, Alpine (Railway)
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerifBoldItalic.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSerif-BoldItalic.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans-Bold.ttf",
    ]
    for fpath in font_candidates:
        if os.path.exists(fpath):
            try:
                font = ImageFont.truetype(fpath, font_size)
                log.info(f"Watermark font: {fpath} size={font_size}")
                break
            except Exception as fe:
                log.warning(f"Font load failed {fpath}: {fe}")

    if font is None:
        log.warning("No system font found — using Pillow default")
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()

    # Text bounding box — textbbox available since Pillow 8.0
    try:
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(watermark_text, font=font)

    pad_x = int(w * 0.02)
    pad_y = int(h * 0.02)
    x = pad_x
    y = h - th - pad_y * 3

    draw.text((x + 2, y + 2), watermark_text, font=font, fill=(0, 0, 0, 130))
    draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 210))

    out = PILImage.alpha_composite(img, txt_layer).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=92)
    log.info(f"Watermark applied: size={w}x{h}")
    return buf.getvalue()


# ── Google Drive accessibility check ─────────────────────────────────────────

def check_gdrive_folder_accessible(folder_id: str) -> dict:
    """
    Try to list the first page of the folder.
    Returns {"accessible": bool, "reason": str}.
    Prefers the Drive API if GOOGLE_API_KEY is set, else tries a public URL.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if api_key:
        try:
            params = {
                "q": f"'{folder_id}' in parents and trashed=false",
                "fields": "files(id,name)",
                "key": api_key,
                "pageSize": "1",
            }
            url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "Pixmatch/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if "error" in data:
                err_msg = data["error"].get("message", str(data["error"]))
                if "not found" in err_msg.lower() or "forbidden" in err_msg.lower():
                    return {
                        "accessible": False,
                        "reason": "This folder is private or doesn't exist. Please set sharing to 'Anyone with the link' and try again.",
                    }
                return {"accessible": False, "reason": f"Drive API error: {err_msg}"}
            return {"accessible": True, "reason": "ok", "file_count_preview": len(data.get("files", []))}
        except Exception as e:
            return {"accessible": False, "reason": f"Could not reach Google Drive: {str(e)}"}
    else:
        # Fallback: try HTML page
        try:
            url = f"https://drive.google.com/drive/folders/{folder_id}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            if "you need access" in body.lower() or "request access" in body.lower():
                return {
                    "accessible": False,
                    "reason": "This Google Drive folder is private. Please change sharing to 'Anyone with the link — Viewer'.",
                }
            return {"accessible": True, "reason": "ok"}
        except Exception as e:
            return {"accessible": False, "reason": f"Could not verify folder access: {str(e)}"}


# ── Event Group DB helpers ────────────────────────────────────────────────────

def db_create_group(user_id: str, name: str, description: str = "", watermark_text: str = "") -> dict:
    group_id = str(uuid.uuid4())[:12]
    now = time.time()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO event_groups (id, user_id, name, description, watermark_text, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (group_id, user_id, name, description, watermark_text or "", now))
        conn.commit()
    return {"id": group_id, "user_id": user_id, "name": name, "description": description,
            "watermark_text": watermark_text, "created_at": now}


def db_get_group(group_id: str) -> Optional[dict]:
    cached = cache_get(f"group:{group_id}")
    if cached:
        return cached
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM event_groups WHERE id = %s", (group_id,))
            row = cur.fetchone()
    if row:
        result = dict(row)
        cache_set(f"group:{group_id}", result, ttl=60)
        return result
    return None


def db_list_groups(user_id: str) -> list:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM event_groups WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,)
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def db_update_group(group_id: str, **fields):
    if not fields:
        return
    set_clause = ", ".join(f"{k}=%s" for k in fields)
    values = list(fields.values()) + [group_id]
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE event_groups SET {set_clause} WHERE id=%s", values)
        conn.commit()
    cache_delete(f"group:{group_id}")


def db_delete_group(group_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM event_groups WHERE id=%s", (group_id,))
        conn.commit()
    cache_delete(f"group:{group_id}")


# ── Discount code helpers ─────────────────────────────────────────────────────

def validate_discount_code(code: str, user_id: str, interval: str = "monthly") -> dict:
    """
    Check if a discount code is valid for this user.
    Returns {"valid": bool, "discount_pct": int, "reason": str}

    All invalid paths return the same generic message to prevent
    enumeration attacks (can't tell if code exists vs expired vs used).
    """
    _INVALID = {"valid": False, "discount_pct": 0, "reason": "Invalid or expired discount code."}
    code = code.strip().upper()
    if not code:
        return _INVALID
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM discount_codes WHERE code=%s", (code,))
            dc = cur.fetchone()
    if not dc:
        return _INVALID
    dc = dict(dc)

    if dc.get("expires_at") and time.time() > dc["expires_at"]:
        return _INVALID

    if dc["use_count"] >= dc["max_uses"]:
        return _INVALID

    # Check if this specific user has already used it
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as n FROM discount_code_uses WHERE code=%s AND user_id=%s",
                (code, user_id)
            )
            row = cur.fetchone()
    if row and row["n"] > 0:
        return _INVALID

    # Check interval compatibility — tell the user this one so they know
    # to try the other billing toggle, but don't expose anything else
    dc_interval = dc.get("interval", "both")
    if dc_interval != "both" and dc_interval != interval:
        return {"valid": False, "discount_pct": 0,
                "reason": f"This code is only valid for {dc_interval} billing."}

    return {"valid": True, "discount_pct": dc["discount_pct"], "reason": "ok", "code": code}


def consume_discount_code(code: str, user_id: str, order_id: str = None):
    """Mark a discount code as used by this user."""
    code = code.strip().upper()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO discount_code_uses (id, code, user_id, used_at, order_id) VALUES (%s,%s,%s,%s,%s)",
                (str(uuid.uuid4()), code, user_id, time.time(), order_id)
            )
            cur.execute("UPDATE discount_codes SET use_count=use_count+1 WHERE code=%s", (code,))
        conn.commit()


# ── Share analytics helpers ───────────────────────────────────────────────────

def record_share_event(share_id: str, event_type: str, ip: str = ""):
    """Record a view or download event for analytics."""
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16] if ip else ""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO share_analytics (id, share_id, event_type, ip_hash, created_at) VALUES (%s,%s,%s,%s,%s)",
                    (str(uuid.uuid4()), share_id, event_type, ip_hash, time.time())
                )
                if event_type == "view":
                    cur.execute(
                        "UPDATE shares SET view_count=COALESCE(view_count,0)+1, last_viewed_at=%s WHERE share_id=%s",
                        (time.time(), share_id)
                    )
                elif event_type == "download":
                    cur.execute(
                        "UPDATE shares SET download_count=COALESCE(download_count,0)+1 WHERE share_id=%s",
                        (share_id,)
                    )
            conn.commit()
        cache_delete(f"share:{share_id}")
    except Exception as e:
        log.warning(f"Analytics record failed: {e}")


# ── Resilient ZIP upload (chunked read) ──────────────────────────────────────

async def resilient_read_upload(file: "UploadFile", max_bytes: int = 2 * 1024 * 1024 * 1024) -> bytes:
    """
    Read an upload in 1 MB chunks with per-chunk timeout awareness.
    Raises HTTPException(413) if file exceeds max_bytes.
    Raises HTTPException(408) on read timeout.
    """
    from fastapi import UploadFile as FU
    chunks = []
    total = 0
    chunk_size = 1024 * 1024  # 1 MB
    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(413, "File too large. Maximum upload size is 2 GB.")
            chunks.append(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(408, f"Upload interrupted. Please check your connection and try again. ({e})")
    return b"".join(chunks)
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ── Email / OTP helpers ───────────────────────────────────────────────────────
OTP_EXPIRY    = int(os.environ.get("OTP_EXPIRY_SECONDS", "600"))  # 10 minutes

# ── Gmail API Email Sending ───────────────────────────────────────────────────

def send_email(to: str, subject: str, html_body: str):
    """
    Send HTML email via Gmail API (works on Railway - uses HTTPS port 443)
    """
    client_id = os.environ.get("GMAIL_CLIENT_ID")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET")
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN")
    
    if not all([client_id, client_secret, refresh_token]):
        log.error("Gmail API credentials not configured")
        raise RuntimeError(
            "Email not configured. Set GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN in Railway environment variables."
        )
    
    credentials = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/gmail.send"]
    )
    
    try:
        if credentials.expired:
            credentials.refresh(GoogleRequest())
    except Exception as e:
        log.error(f"Failed to refresh token: {e}")
        raise RuntimeError(f"Gmail API authentication failed: {str(e)}")
    
    message = EmailMessage()
    message.set_content("Please enable HTML to view this message")
    message.add_alternative(html_body, subtype="html")
    message["To"] = to
    message["From"] = "FaceFind <admin.facefind@gmail.com>"
    message["Subject"] = subject
    
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
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

def generate_license_key(user_id: str, plan: str, interval: str = "monthly") -> str:
    """
    Issue a new license key for a user based on their plan and billing interval.
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
                "UPDATE license_keys SET revoked=TRUE WHERE user_id=%s AND revoked=FALSE",
                (user_id,)
            )
        conn.commit()

    now = time.time()
    # Key expires at end of the billing period: 365 days for annual, 30 days for monthly
    period_days = get_billing_period_days(interval)
    expires_at = now + period_days * 24 * 3600

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO license_keys
                  (key, user_id, plan, created_at, expires_at, revoked, activations, max_activations)
                VALUES (%s, %s, %s, %s, %s, FALSE, 0, %s)
            """, (
                key,
                user_id,
                plan,
                now,
                expires_at,
                limits["max_activations"],
            ))
        conn.commit()

    log.info(f"License key issued: {key[:12]}… for user {user_id} on plan {plan} ({interval}, expires {period_days}d)")
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
                VALUES (%s, %s, %s, %s, FALSE)
            """, (token, user_id, now, now + 15 * 60))
        conn.commit()
    return token


def consume_download_token(token: str) -> Optional[str]:
    """Validate and consume a download token. Returns user_id if valid."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM download_tokens WHERE token=%s AND used=FALSE AND expires_at > %s",
                (token, time.time())
            )
            row = cur.fetchone()
        if not row:
            return None
        with conn.cursor() as cur:
            cur.execute("UPDATE download_tokens SET used=TRUE WHERE token=%s", (token,))
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
    return {
        "id":    user["id"],
        "email": user["email"],
        "name":  user["name"],
        "plan":  user.get("plan") or "free",
        "plan_interval":                user.get("plan_interval") or "monthly",
        "credits_paise":                user.get("credits_paise") or 0,
        "scheduled_downgrade":          user.get("scheduled_downgrade"),
        "scheduled_downgrade_at":       user.get("scheduled_downgrade_at"),
        "scheduled_downgrade_interval": user.get("scheduled_downgrade_interval"),
        "plan_cycle_start":             user.get("plan_cycle_start"),
    }

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
    group_id: str = Form(default=""),
):
    user = require_auth(request)
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file.")

    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir()

    zip_path = dataset_dir / "upload.zip"
    raw_bytes = await resilient_read_upload(file)
    zip_path.write_bytes(raw_bytes)
    
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
    # Assign to group if provided
    if group_id:
        g = db_get_group(group_id)
        if g and g["user_id"] == user["id"]:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE datasets SET group_id=%s WHERE id=%s", (group_id, dataset_id))
                conn.commit()
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
):
    user = require_auth(request)
    
    # FIX 1: Properly read JSON from the frontend instead of Form data
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    folder_url = body.get("folder_url", "").strip()
    name = body.get("name", "").strip()
    group_id = body.get("group_id", "").strip()

    if not folder_url:
        raise HTTPException(400, "Google Drive folder URL is required.")

    folder_id = extract_gdrive_folder_id(folder_url)
    if not folder_id:
        raise HTTPException(400, "Could not extract a folder ID from the provided URL.")
        
    access = check_gdrive_folder_accessible(folder_id)
    if not access["accessible"]:
        raise HTTPException(400, f"Google Drive folder is not accessible: {access['reason']}")

    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir(exist_ok=True)

    ds = {
        "id": dataset_id, "user_id": user["id"],
        "name": name or f"Drive Folder ({folder_id[:8]}…)",
        "source": "gdrive", "folder_id": folder_id,
        "status": "downloading", "total": 0, "processed": 0,
        "face_count": 0, "error": None, "created_at": time.time(),
    }
    
    # FIX 2: Upsert the dataset BEFORE assigning it to an event group
    db_upsert_dataset(ds)

    if group_id:
        g = db_get_group(group_id)
        if g and g["user_id"] == user["id"]:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE datasets SET group_id=%s WHERE id=%s", (group_id, dataset_id))
                conn.commit()

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
    raw_bytes = await resilient_read_upload(file)
    zip_path.write_bytes(raw_bytes)
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
async def create_share(request: Request):
    user = require_auth(request)
    
    # 1. Safely parse the payload (Handles both JSON and Form Data)
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
        else:
            form = await request.form()
            body = dict(form)
    except Exception:
        raise HTTPException(400, "Invalid request body. Expected JSON.")

    # 2. Extract fields (works regardless of what the frontend sent)
    dataset_id = body.get("dataset_id")
    group_id = body.get("group_id")
    watermark_text = body.get("watermark_text", "")

    # 3. If the frontend tried to share a 'Group', find a ready dataset inside it
    if group_id and not dataset_id:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM datasets WHERE group_id=%s AND status='ready' LIMIT 1", (group_id,))
                row = cur.fetchone()
        if not row:
            raise HTTPException(400, "This group has no ready datasets yet. Wait for processing to finish before sharing.")
        dataset_id = row["id"]

    # 4. Ensure we have a valid ID before proceeding
    if not dataset_id:
        raise HTTPException(422, "dataset_id or group_id is required.")

    ds = db_get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found.")
    if ds["status"] != "ready":
        raise HTTPException(400, "Dataset is not ready yet.")

    # If no explicit watermark supplied, inherit from the event group
    if not watermark_text and ds.get("group_id"):
        group = db_get_group(ds["group_id"])
        if group:
            watermark_text = group.get("watermark_text") or ""

    # 5. Check if share already exists
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT share_id FROM shares WHERE dataset_id = %s LIMIT 1", (dataset_id,))
            existing = cur.fetchone()
            
    if existing:
        # Update watermark on existing share if group watermark changed
        if watermark_text:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE shares SET watermark_text=%s WHERE share_id=%s",
                        (watermark_text[:80], existing["share_id"])
                    )
                conn.commit()
            cache_delete(f"share:{existing['share_id']}")
        return {"share_id": existing["share_id"]}

    # 6. Create new share
    share_id = str(uuid.uuid4())[:12]
    share = {
        "share_id":     share_id,
        "dataset_id":   dataset_id,
        "dataset_name": ds["name"],
        "created_at":   time.time(),
    }
    db_insert_share(share)
    
    # 7. Apply watermark if provided
    if watermark_text:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE shares SET watermark_text=%s WHERE share_id=%s",
                    (watermark_text[:80], share_id)
                )
            conn.commit()
        cache_delete(f"share:{share_id}")
        
    return {"share_id": share_id}


@app.delete("/api/shares/{share_id}")
def delete_share(share_id: str, request: Request):
    """
    Delete a share link — revokes guest access and frees the slot so a
    fresh link can be created for the same dataset if needed.
    Only the dataset owner can delete it.
    """
    user = require_auth(request)
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share link not found.")
    ds = db_get_dataset(share["dataset_id"])
    if not ds or ds["user_id"] != user["id"]:
        raise HTTPException(403, "Not your share link.")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM shares WHERE share_id = %s", (share_id,))
        conn.commit()
    cache_delete(f"share:{share_id}")
    log.info(f"Share {share_id} deleted by user {user['id']}")
    return {"ok": True}


@app.get("/api/shares/{share_id}")
def get_share(share_id: str, request: Request):
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share link not found.")
    # Record analytics view (best-effort)
    client_ip = request.client.host if request.client else ""
    record_share_event(share_id, "view", client_ip)
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
                "SELECT * FROM license_keys WHERE user_id=%s AND revoked=FALSE ORDER BY created_at DESC LIMIT 1",
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
        interval = user.get("plan_interval") or "monthly"
        key = generate_license_key(user["id"], plan, interval)
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
                "SELECT key FROM license_keys WHERE user_id=%s AND revoked=FALSE AND expires_at > %s LIMIT 1",
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
                SET activations  = LEAST(activations + 1, max_activations),
                    last_seen_at = %s,
                    last_seen_ip = %s
                WHERE key = %s
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


@app.post("/api/license/userinfo")
async def license_userinfo(request: Request):
    """
    Called by the self-hosted launcher to fetch the user account linked to a license key.
    Used to auto-login the user locally without them needing to enter credentials.
    Returns just enough info to create a local session: email, name, plan.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    key_str = (body.get("key") or "").strip().upper()
    if not key_str:
        raise HTTPException(400, "key is required.")

    key_data = db_get_license_key(key_str)
    if not key_data or key_data["revoked"]:
        raise HTTPException(403, "Invalid or revoked license key.")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT email, name, plan FROM users WHERE id = %s", (key_data["user_id"],))
            row = cur.fetchone()

    if not row:
        raise HTTPException(404, "User not found.")

    return {"email": row["email"], "name": row["name"], "plan": row["plan"]}


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
                "UPDATE license_keys SET revoked=TRUE WHERE user_id=%s AND revoked=FALSE",
                (user["id"],)
            )
            count = cur.rowcount
        conn.commit()

    if count == 0:
        raise HTTPException(404, "No active license key found.")

    log.info(f"License revoked for user {user['id']}")
    return {"ok": True, "message": "License key revoked. You can generate a new one at any time."}


# ── Payment endpoints (Razorpay) ──────────────────────────────────────────────

@app.post("/api/payments/create-order")
async def create_order(request: Request):
    """
    Create a Razorpay order for the selected plan.
    Applies proration credit for unused days on current plan,
    account credits (referrals/goodwill), and loyalty discount where eligible.
    If total credits cover the full price, upgrades immediately (no payment needed).
    """
    user = require_auth(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    plan = (body.get("plan") or "").strip()
    if plan not in PLAN_PRICES_PAISE:
        raise HTTPException(400, f"Unknown plan: {plan}")

    target_interval = (body.get("interval") or "monthly").strip()
    if target_interval not in ("monthly", "annual"):
        raise HTTPException(400, "interval must be 'monthly' or 'annual'.")

    # ── Discount code (optional) ─────────────────────────────────────────────
    discount_code_str = (body.get("discount_code") or "").strip().upper()
    discount_pct = 0
    if discount_code_str:
        dc_result = validate_discount_code(discount_code_str, user["id"], target_interval)
        if not dc_result["valid"]:
            raise HTTPException(400, dc_result["reason"])
        discount_pct = dc_result["discount_pct"]

    current_plan = user.get("plan", "free") or "free"
    current_interval = user.get("plan_interval") or "monthly"

    # Block if literally nothing is changing
    if plan == current_plan and target_interval == current_interval:
        raise HTTPException(400, "You are already on this plan and billing interval.")

    # An interval switch on the same plan tier is treated as an upgrade flow:
    # monthly→annual  : charge the annual price minus proration credit on monthly
    # annual→monthly  : NOT handled here — goes through schedule-downgrade-interval
    #                   because the user has already paid for the full year
    if plan == current_plan and target_interval == "monthly" and current_interval == "annual":
        raise HTTPException(400, "To switch from annual to monthly, use the downgrade flow — your annual access continues until renewal.")

    is_upgrade = plan_rank(plan) > plan_rank(current_plan) or (
        plan == current_plan and target_interval == "annual" and current_interval == "monthly"
    )

    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(503, "Payment gateway not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Railway environment variables.")

    cycle_start = user.get("plan_cycle_start")
    credits_paise = user.get("credits_paise") or 0

    # ── Loyalty discount (one-time, 20% off for loyal paid users upgrading) ──
    loyalty_discount = 0
    if is_upgrade:
        loyalty_discount = apply_loyalty_discount(user, plan, target_interval)

    # ── Compute what the user actually owes ───────────────────────────────────
    billing = compute_upgrade_charge(
        current_plan, plan, cycle_start, credits_paise + loyalty_discount,
        current_interval=current_interval,
        target_interval=target_interval,
    )
    charge_paise = billing["charge_paise"]
    proration_credit = billing["proration_credit_paise"]
    total_credit_used = billing["total_credit_paise"]

    # ── Apply discount code percentage on top of proration/credits ────────────
    discount_amount_paise = 0
    if discount_pct > 0:
        full_for_discount = get_plan_price(plan, target_interval)
        discount_amount_paise = int(full_for_discount * discount_pct / 100)
        charge_paise = max(charge_paise - discount_amount_paise, 0)
        billing["discount_code_pct"] = discount_pct
        billing["discount_code_amount_paise"] = discount_amount_paise

    # ── Free upgrade: credits cover the full price ────────────────────────────
    if is_upgrade and charge_paise == 0:
        now = time.time()
        new_credits = max((credits_paise + loyalty_discount) - (get_plan_price(plan, target_interval) - proration_credit), 0)
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users SET plan=%s, plan_cycle_start=%s, credits_paise=%s,
                           plan_interval=%s,
                           loyalty_discount_used=CASE WHEN %s>0 THEN TRUE ELSE loyalty_discount_used END,
                           scheduled_downgrade=NULL, scheduled_downgrade_at=NULL
                    WHERE id=%s
                """, (plan, now, new_credits, target_interval, loyalty_discount, user["id"]))
                cur.execute("""
                    INSERT INTO razorpay_orders
                      (order_id, user_id, plan, amount_paise, status, created_at, previous_plan, credit_applied_paise, plan_interval)
                    VALUES (%s, %s, %s, %s, 'paid', %s, %s, %s, %s)
                """, (
                    f"free_upgrade_{user['id'][:8]}_{int(now)}",
                    user["id"], plan, 0, now, current_plan, total_credit_used, target_interval
                ))
            conn.commit()
        token = request.cookies.get("ff_token", "")
        if token:
            cache_delete(f"session:{token}")
        # For free upgrades there is no verify step, so consume the discount code here
        if discount_code_str and discount_pct > 0:
            consume_discount_code(discount_code_str, user["id"], f"free_upgrade_{user['id'][:8]}_{int(now)}")
        log.info(f"Free upgrade via credits: user={user['id']} {current_plan}→{plan} credit={total_credit_used}p discount={discount_code_str or 'none'}")
        return {
            "free_upgrade": True,
            "plan": plan,
            "credit_used_paise": total_credit_used,
            "new_credits_paise": new_credits,
        }

    # ── Paid upgrade/switch: create Razorpay order ────────────────────────────
    receipt = f"ff_{user['id'][:8]}_{plan[:6]}_{int(time.time())}"
    order_payload = json.dumps({
        "amount":   charge_paise,
        "currency": "INR",
        "receipt":  receipt,
        "notes": {
            "user_id":          user["id"],
            "user_email":       user["email"],
            "plan":             plan,
            "plan_interval":    target_interval,
            "previous_plan":    current_plan,
            "proration_credit": proration_credit,
            "loyalty_discount": loyalty_discount,
        },
    }).encode()

    auth_str = base64.b64encode(f"{RAZORPAY_KEY_ID}:{RAZORPAY_KEY_SECRET}".encode()).decode()
    rz_req = urllib.request.Request(
        "https://api.razorpay.com/v1/orders",
        data=order_payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Basic {auth_str}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(rz_req, timeout=15) as resp:
            rz_order = json.loads(resp.read())
    except Exception as e:
        log.error(f"Razorpay create order error: {e}")
        raise HTTPException(502, "Could not create payment order. Please try again.")

    order_id = rz_order["id"]

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO razorpay_orders
                  (order_id, user_id, plan, amount_paise, status, created_at, previous_plan, credit_applied_paise, plan_interval, discount_code)
                VALUES (%s, %s, %s, %s, 'created', %s, %s, %s, %s, %s)
            """, (order_id, user["id"], plan, charge_paise, time.time(), current_plan, total_credit_used, target_interval, discount_code_str or None))
        conn.commit()

    # NOTE: discount code is stored on the order row and consumed only after
    # payment is verified in /api/payments/verify — NOT here.
    log.info(f"Razorpay order created: {order_id} user={user['id']} {current_plan}→{plan} ({target_interval}) charge={charge_paise}p credit={total_credit_used}p loyalty={loyalty_discount}p discount={discount_code_str or 'none'}")

    return {
        "order_id":               order_id,
        "amount":                 charge_paise,
        "currency":               "INR",
        "key_id":                 RAZORPAY_KEY_ID,
        "user_name":              user["name"],
        "user_email":             user["email"],
        "plan":                   plan,
        "plan_interval":          target_interval,
        "billing":                billing,
        "loyalty_discount_paise": loyalty_discount,
        "free_upgrade":           False,
        "discount_code":          discount_code_str,
        "discount_pct":           discount_pct,
    }


@app.post("/api/payments/verify")
async def verify_payment(request: Request):
    """
    Verify the Razorpay payment signature after checkout completes.
    On success: updates the user's plan and marks the order paid.

    Body (JSON):
        razorpay_order_id, razorpay_payment_id, razorpay_signature
    """
    user = require_auth(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    order_id   = body.get("razorpay_order_id", "")
    payment_id = body.get("razorpay_payment_id", "")
    signature  = body.get("razorpay_signature", "")

    if not all([order_id, payment_id, signature]):
        raise HTTPException(400, "Missing payment fields.")

    # ── Verify HMAC-SHA256 signature ─────────────────────────────────────────
    expected = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        log.warning(f"Razorpay signature mismatch for order {order_id}")
        raise HTTPException(400, "Payment verification failed. Signature mismatch.")

    # ── Fetch the order from our DB ──────────────────────────────────────────
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM razorpay_orders WHERE order_id=%s AND user_id=%s",
                (order_id, user["id"])
            )
            order_row = cur.fetchone()

    if not order_row:
        raise HTTPException(404, "Order not found.")

    if order_row["status"] == "paid":
        # Idempotent — already processed
        return {"ok": True, "plan": order_row["plan"], "already_processed": True}

    plan = order_row["plan"]
    previous_plan  = order_row.get("previous_plan") or ""
    credit_applied = order_row.get("credit_applied_paise") or 0
    plan_interval  = order_row.get("plan_interval") or "monthly"

    # ── Update user plan + mark order paid ───────────────────────────────────
    now = time.time()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET
                    plan=%s,
                    plan_interval=%s,
                    plan_cycle_start=%s,
                    scheduled_downgrade=NULL,
                    scheduled_downgrade_at=NULL
                WHERE id=%s
            """, (plan, plan_interval, now, user["id"]))
            # Deduct any account credits used (proration is virtual — no deduction needed)
            if credit_applied > 0:
                cur.execute("""
                    UPDATE users SET
                        credits_paise=GREATEST(credits_paise - %s, 0),
                        loyalty_discount_used=TRUE
                    WHERE id=%s
                """, (credit_applied, user["id"]))
            cur.execute(
                "UPDATE razorpay_orders SET status='paid', payment_id=%s WHERE order_id=%s",
                (payment_id, order_id)
            )
        conn.commit()

    # Invalidate any cached session so /api/auth/me returns the new plan
    token = request.cookies.get("ff_token", "")
    if token:
        cache_delete(f"session:{token}")

    log.info(f"Payment verified: order={order_id} payment={payment_id} user={user['id']} plan={plan}")
    # Consume the discount code that was stored on the order (if any)
    try:
        stored_discount = order_row.get("discount_code")
        if stored_discount:
            consume_discount_code(stored_discount, user["id"], order_id)
            log.info(f"Discount code consumed: {stored_discount} for order {order_id}")
    except Exception as e:
        log.warning(f"Failed to consume discount code for order {order_id}: {e}")

    # ── Fire referral credit on first ever payment ────────────────────────────
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) as n FROM razorpay_orders WHERE user_id=%s AND status='paid' AND order_id != %s",
                    (user["id"], order_id)
                )
                prior = cur.fetchone()
        if prior and prior["n"] == 0:
            # This is their first payment — check for a referrer
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT referred_by FROM users WHERE id=%s", (user["id"],))
                    ref_row = cur.fetchone()
            referrer_id = ref_row["referred_by"] if ref_row else None
            if referrer_id:
                REFERRAL_CREDIT_PAISE = 5000  # ₹50
                with get_db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE users SET credits_paise=credits_paise+%s WHERE id=%s",
                            (REFERRAL_CREDIT_PAISE, referrer_id)
                        )
                        cur.execute("""
                            INSERT INTO referral_credits (id, user_id, from_user_id, amount_paise, reason, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (str(uuid.uuid4()), referrer_id, user["id"], REFERRAL_CREDIT_PAISE, "referral_conversion", time.time()))
                    conn.commit()
                log.info(f"Referral credit: ₹50 credited to {referrer_id} for referring {user['id']}")
    except Exception as e:
        log.warning(f"Referral credit payout failed: {e}")

    # Send confirmation email (non-blocking, best-effort)
    try:
        plan_labels = {
            "personal_lite":  "Personal Lite",
            "personal_pro":   "Personal Pro",
            "personal_max":   "Personal Max",
            "photo_starter":  "Studio Starter",
            "photo_pro":      "Studio Pro",
        }
        plan_label = plan_labels.get(plan, plan)
        amount_inr = order_row["amount_paise"] // 100
        html = f"""
        <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:480px;margin:0 auto;background:#f9f7f4;padding:32px 24px">
          <div style="text-align:center;margin-bottom:24px">
            <span style="font-size:28px;font-weight:800;color:#4f46e5;letter-spacing:-1px">FaceFind</span>
          </div>
          <div style="background:#fff;border-radius:16px;padding:36px;box-shadow:0 4px 16px rgba(0,0,0,0.07)">
            <div style="text-align:center;margin-bottom:20px;">
              <div style="width:56px;height:56px;background:#ecfdf5;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
              </div>
            </div>
            <h2 style="margin:0 0 8px;font-size:20px;color:#1c1917;text-align:center">Payment confirmed!</h2>
            <p style="margin:0 0 24px;font-size:14px;color:#78716c;text-align:center">
              Your <strong style="color:#1c1917">{plan_label}</strong> plan is now active.
            </p>
            <div style="background:#f9f7f4;border-radius:10px;padding:16px 20px;margin-bottom:24px;">
              <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:8px;">
                <span style="color:#78716c">Plan</span><span style="font-weight:700;color:#1c1917">{plan_label}</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:8px;">
                <span style="color:#78716c">Amount paid</span><span style="font-weight:700;color:#1c1917">&#x20B9;{amount_inr}</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:13px;">
                <span style="color:#78716c">Payment ID</span><span style="font-weight:600;color:#4f46e5;font-size:11px;font-family:monospace">{payment_id}</span>
              </div>
            </div>
            <a href="https://facefind-production.up.railway.app/admin.html"
               style="display:block;text-align:center;background:#4f46e5;color:#fff;font-weight:700;font-size:14px;padding:12px 24px;border-radius:10px;text-decoration:none;">
              Go to Dashboard
            </a>
          </div>
        </div>
        """
        send_email(user["email"], f"FaceFind — {plan_label} plan activated ✓", html)
    except Exception as e:
        log.warning(f"Could not send payment confirmation email: {e}")

    return {"ok": True, "plan": plan}


def check_admin_secret(request: Request):
    """
    Guard for admin/cron endpoints. Fail-closed:
    - ADMIN_SECRET env var MUST be set or ALL admin calls are denied.
    - Uses constant-time compare to prevent timing attacks.
    """
    secret = os.environ.get("ADMIN_SECRET", "")
    if not secret:
        log.warning("Admin endpoint called but ADMIN_SECRET is not configured — denying.")
        raise HTTPException(403, "Forbidden.")
    provided = request.headers.get("X-Admin-Secret", "")
    if not hmac.compare_digest(secret.encode(), provided.encode()):
        raise HTTPException(403, "Forbidden.")


# ── Billing management endpoints ──────────────────────────────────────────────

@app.get("/api/billing/info")
def billing_info(request: Request):
    """
    Return everything the frontend needs to render upgrade/downgrade options:
    - Current plan + cycle start + billing interval
    - Proration credit available (for upgrades)
    - Account credits balance
    - Loyalty discount eligibility
    - Scheduled downgrade (if any)
    - Dataset usage vs new-plan limits (for downgrade warnings)
    - Renewal date based on correct billing period (30 or 365 days)
    """
    user = require_auth(request)
    current_plan     = user.get("plan", "free") or "free"
    current_interval = user.get("plan_interval") or "monthly"
    cycle_start      = user.get("plan_cycle_start")
    credits          = user.get("credits_paise") or 0
    loyalty_discount = apply_loyalty_discount(user, "")  # just checking eligibility

    # Compute upgrade costs for every higher plan (show both monthly and annual options)
    upgrade_costs = {}
    for plan, price in PLAN_PRICES_PAISE.items():
        if plan_rank(plan) > plan_rank(current_plan):
            upgrade_costs[plan] = {}
            for target_interval in ("monthly", "annual"):
                loyalty = apply_loyalty_discount(user, plan, target_interval)
                billing = compute_upgrade_charge(
                    current_plan, plan, cycle_start, credits + loyalty,
                    current_interval=current_interval,
                    target_interval=target_interval,
                )
                upgrade_costs[plan][target_interval] = {
                    **billing,
                    "loyalty_discount_paise": loyalty,
                    "loyalty_eligible": loyalty > 0,
                }

    # Dataset usage for downgrade warning
    datasets = db_list_datasets(user["id"])
    dataset_count = len(datasets)

    # Days remaining in current cycle (correct period for monthly vs annual)
    days_remaining = None
    renewal_at = None
    if cycle_start and current_plan != "free":
        period_days = get_billing_period_days(current_interval)
        elapsed = (time.time() - cycle_start) / 86400
        days_remaining = max(round(period_days - elapsed), 0)
        renewal_at = cycle_start + period_days * 86400

    return {
        "plan":                          current_plan,
        "plan_interval":                 current_interval,
        "plan_cycle_start":              cycle_start,
        "renewal_at":                    renewal_at,
        "days_remaining":                days_remaining,
        "credits_paise":                 credits,
        "scheduled_downgrade":           user.get("scheduled_downgrade"),
        "scheduled_downgrade_at":        user.get("scheduled_downgrade_at"),
        "scheduled_downgrade_interval":  user.get("scheduled_downgrade_interval"),
        "loyalty_eligible":              apply_loyalty_discount(user, "photo_pro") > 0,
        "upgrade_costs":                 upgrade_costs,
        "dataset_count":                 dataset_count,
        "plan_limits":                   PLAN_LIMITS,
    }


@app.post("/api/billing/schedule-downgrade")
async def schedule_downgrade(request: Request):
    """
    Schedule a downgrade to a lower plan OR an interval switch (annual→monthly)
    at end of the current billing cycle.
    - Validates the target plan is lower than current, OR same plan with annual→monthly switch
    - Warns if user has datasets/images over new plan's limits
    - Does NOT charge or refund anything — access continues until cycle end
    """
    user = require_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    target          = (body.get("plan") or "").strip()
    target_interval = (body.get("interval") or "monthly").strip()
    valid_targets   = list(PLAN_LIMITS.keys())
    if target not in valid_targets:
        raise HTTPException(400, f"Unknown plan: {target}")
    if target_interval not in ("monthly", "annual"):
        raise HTTPException(400, "interval must be 'monthly' or 'annual'.")

    current_plan     = user.get("plan", "free") or "free"
    current_interval = user.get("plan_interval") or "monthly"

    if current_plan == "free":
        raise HTTPException(400, "You are already on the free plan.")

    # Allow: lower plan tier (any interval), OR same plan annual→monthly switch
    is_tier_downgrade    = plan_rank(target) < plan_rank(current_plan)
    is_interval_switch   = (target == current_plan and
                            current_interval == "annual" and
                            target_interval == "monthly")

    if not is_tier_downgrade and not is_interval_switch:
        if target == current_plan and target_interval == current_interval:
            raise HTTPException(400, "No change — you are already on this plan and interval.")
        if target == current_plan and target_interval == "annual":
            raise HTTPException(400, "To switch to annual billing, use the upgrade flow.")
        raise HTTPException(400, "Use the upgrade flow to move to a higher plan.")

    # Tier downgrades always land on monthly — the user hasn't paid for annual on
    # the lower tier and no charge happens in this flow.
    if is_tier_downgrade:
        target_interval = "monthly"

    # Calculate when the change fires (end of current billing cycle)
    cycle_start      = user.get("plan_cycle_start") or time.time()
    period_days      = get_billing_period_days(current_interval)
    elapsed_seconds  = time.time() - cycle_start
    seconds_remaining = max(period_days * 86400 - elapsed_seconds, 0)
    downgrade_at     = time.time() + seconds_remaining

    # Dataset over-limit warning (only relevant for tier downgrades)
    datasets = db_list_datasets(user["id"])
    new_limits = PLAN_LIMITS[target]
    over_datasets = max(len(datasets) - new_limits["max_datasets"], 0)
    over_images_datasets = []
    for ds in datasets.values():
        if ds.get("total", 0) > new_limits["max_images"]:
            over_images_datasets.append({"name": ds["name"], "images": ds.get("total", 0)})

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET
                    scheduled_downgrade=%s,
                    scheduled_downgrade_at=%s,
                    scheduled_downgrade_interval=%s
                WHERE id=%s
            """, (target, downgrade_at, target_interval, user["id"]))
        conn.commit()
    cache_delete(f"session:{request.cookies.get('ff_token', '')}")

    if is_interval_switch:
        msg = (f"Your billing will switch to monthly on "
               f"{__import__('datetime').datetime.utcfromtimestamp(downgrade_at).strftime('%d %b %Y')}. "
               f"You keep annual access until then.")
    else:
        msg = (f"Your plan will change to {target.replace('_',' ').title()} on "
               f"{__import__('datetime').datetime.utcfromtimestamp(downgrade_at).strftime('%d %b %Y')}. "
               f"You keep full access until then.")

    log.info(f"Downgrade/interval-switch scheduled: user={user['id']} "
             f"{current_plan}({current_interval})→{target}({target_interval}) at={downgrade_at}")
    return {
        "ok":                    True,
        "current_plan":          current_plan,
        "current_interval":      current_interval,
        "target_plan":           target,
        "target_interval":       target_interval,
        "downgrade_at":          downgrade_at,
        "days_remaining":        round(seconds_remaining / 86400, 1),
        "over_datasets":         over_datasets,
        "over_images_datasets":  over_images_datasets,
        "message":               msg,
    }


@app.post("/api/billing/cancel-downgrade")
def cancel_downgrade(request: Request):
    """Cancel a previously scheduled downgrade."""
    user = require_auth(request)
    if not user.get("scheduled_downgrade"):
        raise HTTPException(400, "No downgrade is scheduled.")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET
                    scheduled_downgrade=NULL,
                    scheduled_downgrade_at=NULL,
                    scheduled_downgrade_interval=NULL
                WHERE id=%s
            """, (user["id"],))
        conn.commit()
    cache_delete(f"session:{request.cookies.get('ff_token', '')}")
    log.info(f"Downgrade cancelled for user {user['id']}")
    return {"ok": True, "message": "Scheduled downgrade cancelled. Your plan stays as-is."}


@app.post("/api/billing/cancel-subscription")
async def cancel_subscription(request: Request):
    """
    Cancel a paid subscription.
    - Does NOT immediately drop the user to free
    - Schedules a downgrade to 'free' at end of current billing cycle
    - User keeps full access until then
    - Can be undone with /api/billing/cancel-downgrade before the cycle ends
    """
    user = require_auth(request)
    current_plan = user.get("plan", "free") or "free"

    if current_plan == "free":
        raise HTTPException(400, "You are already on the free plan.")

    if user.get("scheduled_downgrade") == "free":
        raise HTTPException(400, "Your subscription is already scheduled for cancellation.")

    cycle_start = user.get("plan_cycle_start") or time.time()
    current_interval = user.get("plan_interval") or "monthly"
    period_days = get_billing_period_days(current_interval)
    elapsed_seconds = time.time() - cycle_start
    seconds_remaining = max(period_days * 86400 - elapsed_seconds, 0)
    cancels_at = time.time() + seconds_remaining

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET
                    scheduled_downgrade='free',
                    scheduled_downgrade_at=%s,
                    scheduled_downgrade_interval='monthly'
                WHERE id=%s
            """, (cancels_at, user["id"]))
        conn.commit()
    cache_delete(f"session:{request.cookies.get('ff_token', '')}")

    import datetime
    cancels_date = datetime.datetime.utcfromtimestamp(cancels_at).strftime('%d %b %Y')
    days_left = round(seconds_remaining / 86400, 1)

    log.info(f"Subscription cancelled: user={user['id']} plan={current_plan} access_until={cancels_at}")
    return {
        "ok":           True,
        "cancels_at":   cancels_at,
        "days_left":    days_left,
        "message":      f"Your subscription is cancelled. You keep full {current_plan.replace('_',' ').title()} access until {cancels_date}, then move to the free plan. You can undo this any time before then.",
    }



@app.post("/api/billing/apply-downgrade")
async def apply_downgrade(request: Request):
    """
    Internal/cron endpoint: apply any scheduled downgrades that are due.
    Safe to call frequently — only acts on items whose downgrade_at has passed.
    Protected by ADMIN_SECRET if that env var is set.
    """
    check_admin_secret(request)

    now = time.time()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, email, plan, plan_interval,
                       scheduled_downgrade, scheduled_downgrade_at,
                       scheduled_downgrade_interval, plan_cycle_start
                FROM users
                WHERE scheduled_downgrade IS NOT NULL
                  AND scheduled_downgrade_at <= %s
            """, (now,))
            due = cur.fetchall()

    applied = []
    for row in due:
        user_id      = row["id"]
        old_plan     = row["plan"]
        new_plan     = row["scheduled_downgrade"]
        # If no target interval stored (legacy rows), default to monthly for tier
        # downgrades and preserve current interval for same-plan switches.
        new_interval = row.get("scheduled_downgrade_interval") or "monthly"
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE users SET
                            plan=%s,
                            plan_interval=%s,
                            plan_cycle_start=CASE WHEN %s = 'free' THEN NULL ELSE %s END,
                            scheduled_downgrade=NULL,
                            scheduled_downgrade_at=NULL,
                            scheduled_downgrade_interval=NULL
                        WHERE id=%s
                    """, (new_plan, new_interval, new_plan, time.time(), user_id))
                    # Revoke license key if new plan doesn't support self-hosted
                    # or reissue at the correct tier
                    cur.execute(
                        "UPDATE license_keys SET revoked=TRUE WHERE user_id=%s AND revoked=FALSE",
                        (user_id,)
                    )
                conn.commit()

            # Issue new license key at new plan tier if eligible
            if SELF_HOSTED_PLAN_LIMITS.get(new_plan):
                try:
                    generate_license_key(user_id, new_plan, new_interval)
                except Exception as e:
                    log.warning(f"Could not reissue key for {user_id} on {new_plan}: {e}")

            # Notify user by email
            try:
                with get_db() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT email, name FROM users WHERE id=%s", (user_id,))
                        u = cur.fetchone()
                if u:
                    plan_labels = {
                        "free": "Starter (Free)", "personal_lite": "Personal Lite",
                        "personal_pro": "Personal Pro", "personal_max": "Personal Max",
                        "photo_starter": "Studio Starter", "photo_pro": "Studio Pro",
                    }
                    interval_label = "Annual" if new_interval == "annual" else "Monthly"
                    is_just_interval = (new_plan == old_plan)
                    if is_just_interval:
                        change_desc = f"switched to <strong style=\"color:#1c1917\">{interval_label} billing</strong>"
                        subject_suffix = f"billing switched to {interval_label}"
                    else:
                        change_desc = f"changed to <strong style=\"color:#1c1917\">{plan_labels.get(new_plan, new_plan)} ({interval_label})</strong>"
                        subject_suffix = f"plan changed to {plan_labels.get(new_plan, new_plan)}"
                    html = f"""
                    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:480px;margin:0 auto;background:#f9f7f4;padding:32px 24px">
                      <div style="text-align:center;margin-bottom:24px">
                        <span style="font-size:28px;font-weight:800;color:#4f46e5;letter-spacing:-1px">FaceFind</span>
                      </div>
                      <div style="background:#fff;border-radius:16px;padding:36px;box-shadow:0 4px 16px rgba(0,0,0,0.07)">
                        <h2 style="margin:0 0 8px;font-size:20px;color:#1c1917">Billing update applied</h2>
                        <p style="font-size:14px;color:#78716c;margin:0 0 20px">
                          Hi {u['name']}, your plan has been {change_desc} today.
                        </p>
                        <p style="font-size:13px;color:#78716c;margin:0 0 24px">
                          Your datasets are safe. If any collections exceed your new plan's limits,
                          they are flagged but not deleted — you have 30 days to manage them.
                        </p>
                        <a href="https://facefind-production.up.railway.app/admin.html"
                           style="display:block;text-align:center;background:#4f46e5;color:#fff;font-weight:700;font-size:14px;padding:12px 24px;border-radius:10px;text-decoration:none;">
                          Go to Dashboard
                        </a>
                      </div>
                    </div>
                    """
                    send_email(u["email"], f"FaceFind — {subject_suffix}", html)
            except Exception as e:
                log.warning(f"Downgrade notification email failed for {user_id}: {e}")

            applied.append({"user_id": user_id, "from": f"{old_plan}", "to": f"{new_plan}({new_interval})"})
            log.info(f"Downgrade applied: user={user_id} {old_plan}→{new_plan} interval={new_interval}")
        except Exception as e:
            log.error(f"Failed to apply downgrade for {user_id}: {e}")

    return {"ok": True, "applied": applied, "count": len(applied)}


@app.post("/api/billing/referral")
async def apply_referral(request: Request):
    """
    Apply a referral code (referrer's user ID or email).
    Credits ₹50 to the referrer when the new user makes their first paid purchase.
    Stores referred_by on the current user so the credit fires on first payment.
    Can only be set once and only before the user has ever paid.
    """
    user = require_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    ref_code = (body.get("referral_code") or "").strip().lower()
    if not ref_code:
        raise HTTPException(400, "referral_code is required.")

    if user.get("referred_by"):
        raise HTTPException(400, "A referral code has already been applied to your account.")

    # Check user has no prior payments
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as n FROM razorpay_orders WHERE user_id=%s AND status='paid'",
                (user["id"],)
            )
            row = cur.fetchone()
    if row and row["n"] > 0:
        raise HTTPException(400, "Referral codes can only be applied before your first payment.")

    # Look up referrer by email or user ID
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email FROM users WHERE email=%s OR id=%s",
                (ref_code, ref_code)
            )
            referrer = cur.fetchone()

    if not referrer:
        raise HTTPException(404, "Referral code not found.")
    if referrer["id"] == user["id"]:
        raise HTTPException(400, "You cannot refer yourself.")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET referred_by=%s WHERE id=%s", (referrer["id"], user["id"]))
        conn.commit()
    cache_delete(f"session:{request.cookies.get('ff_token', '')}")

    return {"ok": True, "message": "Referral applied! Your referrer will receive ₹50 credit when you subscribe."}


@app.post("/api/admin/add-credits")
async def admin_add_credits(request: Request):
    """
    Admin-only: add goodwill/referral credits to a user's account.
    Used for support escalations, promotions, or manual referral payouts.
    """
    check_admin_secret(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    target_email  = (body.get("email") or "").strip().lower()
    amount_paise  = int(body.get("amount_paise", 0))
    reason        = body.get("reason", "goodwill")

    if not target_email or amount_paise <= 0:
        raise HTTPException(400, "email and amount_paise (>0) are required.")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email=%s", (target_email,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(404, "User not found.")

    user_id = row["id"]
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET credits_paise=credits_paise+%s WHERE id=%s",
                (amount_paise, user_id)
            )
            cur.execute("""
                INSERT INTO referral_credits (id, user_id, from_user_id, amount_paise, reason, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (str(uuid.uuid4()), user_id, "admin", amount_paise, reason, time.time()))
        conn.commit()

    log.info(f"Credits added: {amount_paise}p to {target_email} reason={reason}")
    return {"ok": True, "email": target_email, "credits_added_paise": amount_paise}


@app.post("/api/billing/send-renewal-reminders")
async def send_renewal_reminders(request: Request):
    """
    Cron endpoint: email users whose subscription renews within the next 7 days.
    Safe to run daily. Protected by ADMIN_SECRET.
    Respects plan_interval so annual subscribers get the correct renewal date.
    """
    check_admin_secret(request)

    now = time.time()
    window_start = now
    window_end   = now + 7 * 86400  # 7 days ahead

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, email, name, plan, plan_interval, plan_cycle_start
                FROM users
                WHERE plan != 'free'
                  AND plan_cycle_start IS NOT NULL
                  AND scheduled_downgrade IS NULL
            """)
            rows = cur.fetchall()

    plan_labels = {
        "personal_lite": "Personal Lite",   "personal_pro": "Personal Pro",
        "personal_max":  "Personal Max",    "photo_starter": "Studio Starter",
        "photo_pro":     "Studio Pro",
    }
    sent = []
    for row in rows:
        interval    = row.get("plan_interval") or "monthly"
        period_days = get_billing_period_days(interval)
        renewal_at  = row["plan_cycle_start"] + period_days * 86400

        if window_start <= renewal_at <= window_end:
            days_until = max(round((renewal_at - now) / 86400), 0)
            plan_label = plan_labels.get(row["plan"], row["plan"])
            price_paise = get_plan_price(row["plan"], interval)
            price_inr   = price_paise // 100
            renewal_date = __import__('datetime').datetime.utcfromtimestamp(renewal_at).strftime('%d %b %Y')
            period_label = "year" if interval == "annual" else "month"
            try:
                html = f"""
                <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:480px;margin:0 auto;background:#f9f7f4;padding:32px 24px">
                  <div style="text-align:center;margin-bottom:24px">
                    <span style="font-size:28px;font-weight:800;color:#4f46e5;letter-spacing:-1px">FaceFind</span>
                  </div>
                  <div style="background:#fff;border-radius:16px;padding:36px;box-shadow:0 4px 16px rgba(0,0,0,0.07)">
                    <h2 style="margin:0 0 8px;font-size:20px;color:#1c1917">Your subscription renews in {days_until} day{'s' if days_until != 1 else ''}</h2>
                    <p style="margin:0 0 24px;font-size:14px;color:#78716c;line-height:1.65">
                      Hi {row['name']}, just a heads-up that your <strong style="color:#1c1917">{plan_label}</strong> ({interval}) plan
                      will automatically renew on <strong style="color:#1c1917">{renewal_date}</strong> for
                      <strong style="color:#1c1917">&#x20B9;{price_inr:,}/{period_label}</strong>.
                    </p>
                    <div style="background:#f9f7f4;border-radius:10px;padding:16px 20px;margin-bottom:24px;">
                      <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:8px;">
                        <span style="color:#78716c">Plan</span><span style="font-weight:700;color:#1c1917">{plan_label}</span>
                      </div>
                      <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:8px;">
                        <span style="color:#78716c">Billing</span><span style="font-weight:700;color:#1c1917">{interval.title()}</span>
                      </div>
                      <div style="display:flex;justify-content:space-between;font-size:13px;">
                        <span style="color:#78716c">Renewal date</span><span style="font-weight:700;color:#1c1917">{renewal_date}</span>
                      </div>
                    </div>
                    <p style="font-size:12px;color:#a8a29e;margin:0 0 20px;text-align:center">
                      To cancel before renewal, visit your account settings.
                    </p>
                    <a href="https://facefind-production.up.railway.app/admin.html"
                       style="display:block;text-align:center;background:#4f46e5;color:#fff;font-weight:700;font-size:14px;padding:12px 24px;border-radius:10px;text-decoration:none;">
                      Manage subscription
                    </a>
                  </div>
                </div>
                """
                send_email(row["email"], f"FaceFind — your {plan_label} plan renews on {renewal_date}", html)
                sent.append({"user_id": row["id"], "email": row["email"], "renewal_at": renewal_at, "days_until": days_until})
                log.info(f"Renewal reminder sent: user={row['id']} plan={row['plan']} ({interval}) renews={renewal_date}")
            except Exception as e:
                log.warning(f"Renewal reminder email failed for {row['id']}: {e}")

    return {"ok": True, "sent": sent, "count": len(sent)}


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.post("/api/admin/set-plan")
def admin_set_plan(request: Request, target_email: str = Form(...), plan: str = Form(...)):
    """
    Admin-only: set a user's plan.
    Call this from your payment provider's webhook after a successful payment.
    Protected by ADMIN_SECRET if that env var is set.
    """
    check_admin_secret(request)

    valid_plans = list(PLAN_LIMITS.keys())
    if plan not in valid_plans:
        raise HTTPException(400, f"Invalid plan. Choose from: {valid_plans}")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, plan FROM users WHERE email=%s", (target_email.lower(),))
            target_user = cur.fetchone()
            if not target_user:
                raise HTTPException(404, "User not found.")
            cur.execute("""
                UPDATE users SET plan=%s, plan_interval='monthly', plan_cycle_start=%s,
                       scheduled_downgrade=NULL, scheduled_downgrade_at=NULL
                WHERE email=%s
            """, (plan, time.time() if plan != "free" else None, target_email.lower()))
            # Sync license key — revoke old, issue new if plan supports it
            cur.execute(
                "UPDATE license_keys SET revoked=TRUE WHERE user_id=%s AND revoked=FALSE",
                (target_user["id"],)
            )
        conn.commit()

    if SELF_HOSTED_PLAN_LIMITS.get(plan):
        try:
            generate_license_key(target_user["id"], plan)
        except Exception as e:
            log.warning(f"admin_set_plan: could not issue license key for {target_email}: {e}")

    log.info(f"Plan updated: {target_email} → {plan}")
    return {"ok": True, "email": target_email, "plan": plan}

# ── Debug endpoint to check email config ──────────────────────────────────────

@app.get("/api/debug/email")
def debug_email():
    """Debug endpoint to check email configuration"""
    client_id = os.environ.get("GMAIL_CLIENT_ID", "")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET", "")
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN", "")
    
    return {
        "client_id_present":     bool(client_id),
        "client_id_preview":     client_id[:10] + "..." if client_id else None,
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
        "email":     "configured" if all([
            os.environ.get("GMAIL_CLIENT_ID"),
            os.environ.get("GMAIL_CLIENT_SECRET"),
            os.environ.get("GMAIL_REFRESH_TOKEN"),
        ]) else "not configured"
    }

# ── Frontend static files ─────────────────────────────────────────────────────


# ── Event Group endpoints ─────────────────────────────────────────────────────

@app.get("/api/groups")
def list_groups(request: Request):
    """List all event groups for the authenticated user."""
    user = require_auth(request)
    groups = db_list_groups(user["id"])
    # Attach dataset count and share link per group
    for g in groups:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) as n FROM datasets WHERE user_id=%s AND group_id=%s",
                    (user["id"], g["id"])
                )
                row = cur.fetchone()
                g["dataset_count"] = row["n"] if row else 0
                # Find share for this group (join through datasets)
                cur.execute("""
                    SELECT s.share_id, s.view_count, s.download_count
                    FROM shares s
                    JOIN datasets d ON s.dataset_id = d.id
                    WHERE d.user_id=%s AND d.group_id=%s
                    LIMIT 1
                """, (user["id"], g["id"]))
                share_row = cur.fetchone()
                g["share_id"] = share_row["share_id"] if share_row else None
                g["view_count"] = share_row["view_count"] if share_row else 0
                g["download_count"] = share_row["download_count"] if share_row else 0
    return groups


@app.post("/api/groups")
async def create_group(request: Request):
    """Create a new event group."""
    user = require_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Group name is required.")
    description = (body.get("description") or "").strip()
    watermark_text = (body.get("watermark_text") or "").strip()[:80]
    group = db_create_group(user["id"], name, description, watermark_text)
    log.info(f"Group created: {group['id']} by user {user['id']}")
    return group


@app.get("/api/groups/{group_id}")
def get_group(group_id: str, request: Request):
    user = require_auth(request)
    g = db_get_group(group_id)
    if not g:
        raise HTTPException(404, "Group not found.")
    if g["user_id"] != user["id"]:
        raise HTTPException(403, "Not your group.")
    # Attach datasets
    datasets = db_list_datasets(user["id"])
    g["datasets"] = [d for d in datasets.values() if d.get("group_id") == group_id]
    return g


@app.patch("/api/groups/{group_id}")
async def update_group(group_id: str, request: Request):
    user = require_auth(request)
    g = db_get_group(group_id)
    if not g:
        raise HTTPException(404, "Group not found.")
    if g["user_id"] != user["id"]:
        raise HTTPException(403, "Not your group.")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")
    allowed = {"name", "description", "watermark_text"}
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        raise HTTPException(400, "No valid fields to update.")
    if "name" in updates and not updates["name"].strip():
        raise HTTPException(400, "Group name cannot be empty.")
    if "watermark_text" in updates:
        updates["watermark_text"] = updates["watermark_text"][:80]
    db_update_group(group_id, **updates)
    return {"ok": True, "group_id": group_id}


@app.delete("/api/groups/{group_id}")
def delete_group(group_id: str, request: Request):
    user = require_auth(request)
    g = db_get_group(group_id)
    if not g:
        raise HTTPException(404, "Group not found.")
    if g["user_id"] != user["id"]:
        raise HTTPException(403, "Not your group.")
    # Unlink datasets from group (don't delete them)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE datasets SET group_id=NULL WHERE group_id=%s AND user_id=%s",
                (group_id, user["id"])
            )
        conn.commit()
    db_delete_group(group_id)
    return {"ok": True}


@app.post("/api/groups/{group_id}/assign-dataset")
async def assign_dataset_to_group(group_id: str, request: Request):
    """Assign a dataset to a group."""
    user = require_auth(request)
    g = db_get_group(group_id)
    if not g or g["user_id"] != user["id"]:
        raise HTTPException(403, "Group not found or not yours.")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")
    dataset_id = (body.get("dataset_id") or "").strip()
    ds = db_get_dataset(dataset_id)
    if not ds or ds["user_id"] != user["id"]:
        raise HTTPException(404, "Dataset not found.")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE datasets SET group_id=%s WHERE id=%s", (group_id, dataset_id))
        conn.commit()
    cache_delete(f"dataset:{dataset_id}")
    cache_delete(f"datasets:{user['id']}")
    return {"ok": True}


# ── QR Code endpoint ──────────────────────────────────────────────────────────

@app.get("/api/shares/{share_id}/qrcode")
def get_share_qr(share_id: str, request: Request):
    """
    Return a QR code PNG for the share link.
    The QR encodes the full public share URL so anyone who scans it goes
    directly to the selfie-upload page.
    Requires auth (photographer only) — no QR generation for anonymous callers.
    """
    require_auth(request)
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share link not found.")

    base_url = str(request.base_url).rstrip("/")
    share_url = f"{base_url}/share.html?id={share_id}"

    png_bytes = generate_qr_code_png(share_url)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": f'inline; filename="qr-{share_id}.png"',
        },
    )


# ── Watermarked image download ────────────────────────────────────────────────

@app.get("/api/watermarked/{dataset_id}")
def serve_image_watermarked(dataset_id: str, image_path: str, share_id: str = None):
    """
    Serve an image with the share's watermark text burned in.
    Called from share.html when user downloads a result.
    Requires share_id query param to look up watermark settings.
    """
    full_path = DATASETS_DIR / dataset_id / image_path
    if not full_path.exists():
        raise HTTPException(404, "Image not found.")

    watermark_text = None
    if share_id:
        share = db_get_share(share_id)
        if share:
            # Record download event
            record_share_event(share_id, "download")
            # 1. Prefer watermark set directly on the share
            watermark_text = share.get("watermark_text") or None
            # 2. Fall back to the event group's watermark if share has none
            if not watermark_text:
                ds = db_get_dataset(share["dataset_id"])
                if ds and ds.get("group_id"):
                    group = db_get_group(ds["group_id"])
                    if group:
                        watermark_text = group.get("watermark_text") or None

    image_bytes = full_path.read_bytes()
    if watermark_text:
        try:
            image_bytes = apply_watermark(image_bytes, watermark_text)
        except Exception as e:
            log.error(f"apply_watermark failed for {dataset_id}/{image_path}: {e}", exc_info=True)

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{Path(image_path).name}"',
            "Cache-Control": "no-store",
        },
    )


# ── Share watermark update ────────────────────────────────────────────────────

@app.patch("/api/shares/{share_id}")
async def update_share(share_id: str, request: Request):
    """Update share settings: watermark_text."""
    user = require_auth(request)
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share not found.")
    ds = db_get_dataset(share["dataset_id"])
    if not ds or ds["user_id"] != user["id"]:
        raise HTTPException(403, "Not your share.")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")
    watermark_text = (body.get("watermark_text") or "").strip()[:80]
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE shares SET watermark_text=%s WHERE share_id=%s",
                (watermark_text or None, share_id)
            )
        conn.commit()
    cache_delete(f"share:{share_id}")
    return {"ok": True}


# ── Share analytics endpoint ──────────────────────────────────────────────────

@app.get("/api/shares/{share_id}/analytics")
def get_share_analytics(share_id: str, request: Request):
    """Return view/download counts for a share link. Owner-only."""
    user = require_auth(request)
    share = db_get_share(share_id)
    if not share:
        raise HTTPException(404, "Share not found.")
    ds = db_get_dataset(share["dataset_id"])
    if not ds or ds["user_id"] != user["id"]:
        raise HTTPException(403, "Not your share.")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT view_count, download_count, last_viewed_at FROM shares WHERE share_id=%s",
                (share_id,)
            )
            row = cur.fetchone()
            # Last 7 days breakdown
            cur.execute("""
                SELECT event_type, COUNT(*) as cnt
                FROM share_analytics
                WHERE share_id=%s AND created_at > %s
                GROUP BY event_type
            """, (share_id, time.time() - 7 * 86400))
            recent = {r["event_type"]: r["cnt"] for r in cur.fetchall()}

    return {
        "share_id":      share_id,
        "view_count":    row["view_count"] if row else 0,
        "download_count": row["download_count"] if row else 0,
        "last_viewed_at": row["last_viewed_at"] if row else None,
        "last_7_days": recent,
    }


# ── Google Drive accessibility check endpoint ─────────────────────────────────

@app.post("/api/gdrive/check")
async def check_gdrive_access(request: Request):
    """
    Pre-flight check: verify a Google Drive folder URL is publicly accessible
    before the user submits it for processing.
    Returns {accessible, reason, folder_id}.
    """
    require_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")
    folder_url = (body.get("folder_url") or "").strip()
    if not folder_url:
        raise HTTPException(400, "folder_url is required.")
    folder_id = extract_gdrive_folder_id(folder_url)
    if not folder_id:
        return {
            "accessible": False,
            "reason": "Could not extract a folder ID from the URL. Please paste the full Google Drive folder link.",
            "folder_id": None,
        }
    result = check_gdrive_folder_accessible(folder_id)
    result["folder_id"] = folder_id
    return result


# ── Discount code endpoints ───────────────────────────────────────────────────

@app.post("/api/discount/validate")
async def validate_discount_endpoint(request: Request):
    """Validate a discount code before checkout (no side effects).
    Rate-limited: max 5 attempts per user per 60 seconds.
    """
    user = require_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")
    code = (body.get("code") or "").strip().upper()
    interval = (body.get("interval") or "monthly").strip()
    if not code:
        raise HTTPException(400, "code is required.")

    # ── Rate limit: 5 attempts per user per 60s ──────────────────────────────
    window_start = time.time() - 60
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as n FROM discount_validate_attempts WHERE user_id=%s AND attempted_at > %s",
                (user["id"], window_start)
            )
            row = cur.fetchone()
            if row and row["n"] >= 5:
                raise HTTPException(429, "Too many attempts. Please wait a moment before trying again.")
            # Record this attempt
            cur.execute(
                "INSERT INTO discount_validate_attempts (id, user_id, attempted_at) VALUES (%s,%s,%s)",
                (str(uuid.uuid4()), user["id"], time.time())
            )
        conn.commit()

    result = validate_discount_code(code, user["id"], interval)
    # Never return the raw code in the validate response — frontend already has it
    result.pop("code", None)
    return result


@app.post("/api/admin/discount/seed")
async def admin_seed_discounts(request: Request):
    """
    Admin-only: idempotently seed the 40 launch discount codes.
    Safe to run multiple times — uses INSERT ... ON CONFLICT DO NOTHING.

    Generates:
      - 10 x 50% off monthly  (GP50M_01 … GP50M_10)
      - 10 x 100% off monthly (GP100M_01 … GP100M_10)
      - 10 x 50% off annual   (GP50A_01 … GP50A_10)
      - 10 x 100% off annual  (GP100A_01 … GP100A_10)

    Each code: max 1 use (one redemption per code), valid once per user.
    No expiry set — revoke individually via the DB if needed.
    """
    check_admin_secret(request)
    now = time.time()
    codes_to_create = []

    for i in range(1, 11):
        suffix = f"{i:02d}"
        codes_to_create += [
            (f"GP50M_{suffix}",  50,  "monthly", 1, "seed"),
            (f"GP100M_{suffix}", 100, "monthly", 1, "seed"),
            (f"GP50A_{suffix}",  50,  "annual",  1, "seed"),
            (f"GP100A_{suffix}", 100, "annual",  1, "seed"),
        ]

    created, skipped = 0, 0
    with get_db() as conn:
        with conn.cursor() as cur:
            for code, pct, interval, max_uses, created_by in codes_to_create:
                cur.execute("""
                    INSERT INTO discount_codes
                      (code, discount_pct, interval, max_uses, use_count, expires_at, created_by, created_at)
                    VALUES (%s, %s, %s, %s, 0, NULL, %s, %s)
                    ON CONFLICT (code) DO NOTHING
                """, (code, pct, interval, max_uses, created_by, now))
                if cur.rowcount > 0:
                    created += 1
                else:
                    skipped += 1
        conn.commit()

    log.info(f"Discount seed: {created} created, {skipped} already existed")
    return {
        "ok": True,
        "created": created,
        "skipped": skipped,
        "total": len(codes_to_create),
        "codes": [c[0] for c in codes_to_create],
    }


@app.post("/api/admin/discount/create")
async def admin_create_discount(request: Request):
    """
    Admin-only: create a discount code.
    discount_pct: 50 or 100 (or any 1-100 integer)
    interval: 'monthly' | 'annual' | 'both'
    max_uses: how many total redemptions allowed
    expires_days: optional, days until expiry from now
    """
    check_admin_secret(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    code = (body.get("code") or secrets.token_urlsafe(6).upper()).strip().upper()
    discount_pct = int(body.get("discount_pct", 50))
    interval = (body.get("interval") or "both").strip()
    max_uses = int(body.get("max_uses", 10))
    expires_days = body.get("expires_days")
    created_by = (body.get("created_by") or "admin").strip()

    if discount_pct < 1 or discount_pct > 100:
        raise HTTPException(400, "discount_pct must be 1-100.")
    if interval not in ("monthly", "annual", "both"):
        raise HTTPException(400, "interval must be monthly, annual, or both.")

    expires_at = time.time() + int(expires_days) * 86400 if expires_days else None

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO discount_codes (code, discount_pct, interval, max_uses, use_count, expires_at, created_by, created_at)
                VALUES (%s, %s, %s, %s, 0, %s, %s, %s)
                ON CONFLICT (code) DO UPDATE SET
                    discount_pct=EXCLUDED.discount_pct,
                    interval=EXCLUDED.interval,
                    max_uses=EXCLUDED.max_uses,
                    expires_at=EXCLUDED.expires_at
            """, (code, discount_pct, interval, max_uses, expires_at, created_by, time.time()))
        conn.commit()

    log.info(f"Discount code created: {code} ({discount_pct}% off, {interval}, max_uses={max_uses})")
    return {
        "ok": True,
        "code": code,
        "discount_pct": discount_pct,
        "interval": interval,
        "max_uses": max_uses,
        "expires_at": expires_at,
    }


@app.get("/api/admin/discount/list")
def admin_list_discounts(request: Request):
    """Admin-only: list all discount codes."""
    check_admin_secret(request)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM discount_codes ORDER BY created_at DESC")
            codes = [dict(r) for r in cur.fetchall()]
    return codes


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn