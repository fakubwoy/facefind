# FaceFind

A self-hosted face recognition app for large photo collections.

**Upload a dataset → Generate embeddings → Share a link → Anyone can upload a selfie and find all their photos.**

Built with InsightFace (ArcFace), FAISS, FastAPI, and vanilla HTML/JS. No cloud face APIs. Runs fully on Railway with Backblaze B2 for bulk storage.

---

## Storage Architecture

| Layer | What lives there | Why |
|-------|-----------------|-----|
| **Railway Postgres** | Users, sessions, datasets metadata, shares, orders, license keys | Lightweight, relational, 1 GB free |
| **Backblaze B2** | Dataset photos, thumbnails, FAISS indexes, selfie uploads, self-hosted ZIP | Free egress to Cloudflare, cheap storage — handles tens of GBs |
| **Railway ephemeral disk** | Tiny temp files only (model weights cached by InsightFace) | No persistent volume needed |

With B2 as the bulk store, Railway's 5 GB volume limit is no longer a concern. You can upload hundreds of GBs of photos.

---

## Quick Start

### 1. Create a Backblaze B2 bucket

1. Sign up at [backblaze.com](https://www.backblaze.com/b2/cloud-storage.html)
2. Create a **public** bucket (e.g. `facefind-photos`)
3. Under **App Keys**, create an application key with **Read and Write** access to that bucket
4. Note down: **keyID**, **applicationKey**, **bucket name**, and the **region** (e.g. `us-west-004`)

### 2. Set Railway environment variables

In the Railway dashboard → your service → **Variables**, add:

```
B2_KEY_ID=<your keyID>
B2_APPLICATION_KEY=<your applicationKey>
B2_BUCKET_NAME=facefind-photos
B2_REGION=us-west-004
```

Plus the existing variables (Postgres is auto-set by the Railway plugin):

```
GMAIL_CLIENT_ID=...
GMAIL_CLIENT_SECRET=...
GMAIL_REFRESH_TOKEN=...
ADMIN_SECRET=...
RAZORPAY_KEY_ID=...
RAZORPAY_KEY_SECRET=...
```

### 3. Deploy

Push to Railway. The app boots, connects to Postgres and B2, and is ready.

### 4. Migrate existing data (if upgrading from local volume storage)

If you already have photos/embeddings on the Railway volume, run the one-shot migration script **before** switching to the new code:

```bash
# In Railway's one-off shell (or locally with the same env vars):
python scripts/migrate_to_b2.py
```

This copies everything to B2 without deleting anything local. It's idempotent — safe to run again.

Alternatively, trigger it via the admin API:

```bash
curl -X POST https://your-app.railway.app/api/admin/migrate-to-b2 \
  -H "X-Admin-Secret: $ADMIN_SECRET"
```

---

## Using the App

### Option A — Upload your own dataset (ZIP)

1. Prepare a ZIP file of your photos.
   - Recommended structure: `person_name/photo1.jpg` (one folder per person)
   - Or a flat ZIP of event photos: `photo1.jpg`, `photo2.jpg`, ...
2. Go to the Admin UI → drag & drop your ZIP → click **Upload & Start Embedding**
3. Wait for embedding to complete (progress shown in real time)
4. Click **Generate Share Link**
5. Send the link to anyone — they can upload a selfie and search

### Option B — Link a Google Drive folder

1. Make the folder publicly accessible (anyone with link can view)
2. Admin UI → **Link Google Drive folder** → paste URL
3. The server downloads directly from Drive into B2, then embeds

### Option C — LFW demo dataset

```bash
bash scripts/download_lfw.sh
# Then in the Admin UI, click "Index LFW Dataset"
```

---

## Project Structure

```
facefind/
├── backend/
│   ├── app.py                  ← FastAPI server (all endpoints)
│   ├── b2_storage.py           ← Backblaze B2 helpers (NEW)
│   ├── datasets/               ← Temp extraction dir (cleared after upload)
│   ├── embeddings/             ← Temp dir (cleared after B2 upload)
│   ├── uploads/                ← Temp selfie storage (cleared after search)
│   ├── datasets_meta.json      ← (legacy, superseded by Postgres)
│   └── shares_meta.json        ← (legacy, superseded by Postgres)
│
├── frontend/
│   ├── index.html
│   ├── login.html
│   ├── pricing.html
│   ├── admin.html
│   ├── download.html
│   └── share.html
│
├── scripts/
│   ├── download_lfw.sh
│   └── migrate_to_b2.py        ← One-shot data migration (NEW)
│
├── requirements.txt            ← boto3 added for B2
├── railway.toml
├── robots.txt
├── sitemap.xml
└── README.md
```

---

## Data Flow (with B2)

### Uploading a ZIP dataset

```
User uploads ZIP
  → app.py extracts ZIP to /tmp/{dataset_id}/
  → compresses/caps images in place
  → uploads each image to B2: datasets/{dataset_id}/{path}
  → deletes local temp files
  → runs embedding job:
      reads images from B2
      builds FAISS index in memory
      uploads to B2: embeddings/{dataset_id}/face_index.faiss
                     embeddings/{dataset_id}/embeddings.npy
                     embeddings/{dataset_id}/metadata.pkl
  → status → "ready"
```

### Selfie search

```
User uploads selfie
  → app.py saves selfie to /tmp/
  → extracts face embedding
  → downloads FAISS index from B2 (cached in process memory)
  → returns matching image paths
  → app.py serves images by proxying from B2
```

### Serving images / thumbnails

```
GET /api/image/{dataset_id}/{path}
  → app.py downloads from B2: datasets/{dataset_id}/{path}
  → streams bytes to browser

GET /api/thumb/{dataset_id}/{path}
  → check B2 thumbs/{dataset_id}/{path}
  → if missing: download original, resize, upload thumb to B2, return
  → else: stream cached thumb from B2
```

---

## Self-Hosted Download Flow

Upload your self-hosted ZIP to B2 (replaces the Railway Volume approach):

```bash
# Using the B2 CLI or AWS CLI with B2 endpoint:
aws s3 cp facefind-selfhosted.zip s3://facefind-photos/releases/facefind-selfhosted.zip \
  --endpoint-url https://s3.us-west-004.backblazeb2.com
```

The `/api/download/file` endpoint streams it directly from B2.

---

## API Reference

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check — returns Postgres, Redis, B2, and email config status |

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create account (triggers OTP email) |
| POST | `/api/auth/verify-otp` | Verify email OTP |
| POST | `/api/auth/login` | Sign in with email + password |
| GET | `/api/auth/me` | Get current user info |
| POST | `/api/auth/logout` | Invalidate session cookie |

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/datasets` | List all datasets for current user |
| POST | `/api/datasets/upload-zip` | Upload ZIP dataset → extracted + stored in B2 |
| POST | `/api/datasets/gdrive` | Link a public Google Drive folder |
| GET | `/api/datasets/{id}/status` | Dataset status + progress + ETA |
| DELETE | `/api/datasets/{id}` | Delete dataset from Postgres + all B2 objects |
| GET | `/api/image/{dataset_id}/{path}` | Proxy original image from B2 |
| GET | `/api/thumb/{dataset_id}/{path}` | Proxy thumbnail from B2 (generated on first request) |

### Shares

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/shares` | Generate a share link for a ready dataset |
| GET | `/api/shares/{share_id}` | Get share info |
| POST | `/api/shares/{share_id}/search` | Search by selfie upload |

### Download & License

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/download/info` | Eligibility check + current license key details |
| POST | `/api/download/generate-key` | Issue or regenerate license key |
| POST | `/api/download/request-link` | Get a 15-min signed one-use download URL |
| GET | `/api/download/file?token=…` | Stream `facefind-selfhosted.zip` from B2 |
| POST | `/api/license/revoke` | Revoke active license key |
| POST | `/api/license/validate` | Validate key + machine ID (called by self-hosted app) |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/set-plan` | Set a user's plan (`X-Admin-Secret` required) |
| POST | `/api/admin/migrate-to-b2` | Trigger one-shot migration from local volume to B2 |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string (auto-set by Railway plugin) |
| `B2_KEY_ID` | ✅ | Backblaze application key ID |
| `B2_APPLICATION_KEY` | ✅ | Backblaze application key secret |
| `B2_BUCKET_NAME` | ✅ | B2 bucket name |
| `B2_REGION` | optional | Bucket region (default: `us-west-004`) |
| `REDIS_URL` | optional | Redis connection string — caching disabled if absent |
| `GMAIL_CLIENT_ID` | ✅ | Gmail OAuth2 client ID for sending OTP emails |
| `GMAIL_CLIENT_SECRET` | ✅ | Gmail OAuth2 client secret |
| `GMAIL_REFRESH_TOKEN` | ✅ | Gmail OAuth2 refresh token |
| `ADMIN_SECRET` | ✅ | Secret header value for admin endpoints |
| `GOOGLE_API_KEY` | optional | Google Drive API key for listing Drive folders |
| `INSIGHTFACE_MODEL` | optional | InsightFace model name (default: `buffalo_sc`) |
| `DET_SIZE` | optional | Face detection input size in pixels (default: `320`) |
| `UNLOAD_MODEL_AFTER_EMBED` | optional | Unload face model after embedding (default: `true`) |
| `MAX_LOADED_INDEXES` | optional | Max FAISS indexes kept in memory (default: `2`) |
| `DATA_DIR` | optional | Local scratch dir (default: `/data`) — only temp files go here |

---

## Stack

| Component | Technology |
|-----------|-----------|
| Face Detection | InsightFace RetinaFace |
| Face Embedding | ArcFace w600k_r50 (512-dim) |
| Vector Search | FAISS IndexFlatIP |
| Backend API | FastAPI + uvicorn |
| Database | PostgreSQL (Railway plugin) — metadata only |
| Bulk Storage | Backblaze B2 (S3-compatible) — images, embeddings, ZIPs |
| Cache | Redis (optional, Railway plugin) |
| Email | Gmail API (OAuth2) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Performance

- **Embedding speed**: ~3–5 images/sec on CPU
- **Search latency**: <5ms for 16k faces (FAISS index cached in process memory)
- **B2 download for index load**: ~50ms for a 10 MB FAISS file on first load, then memory-cached
- **Memory**: ~100MB for 16k face vectors

---

## Notes

- All user data stays in your B2 bucket — Anthropic/Railway never sees your photos
- The share link works as long as your Railway service is running
- B2 is free for the first 10 GB of storage and has free egress to Cloudflare CDN
- `download.html` and `admin.html` are blocked in `robots.txt` and carry `noindex` meta tags