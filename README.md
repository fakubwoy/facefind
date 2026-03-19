# FaceFind

A self-hosted face recognition app for large photo collections.

**Upload a dataset → Generate embeddings → Share a link → Anyone can upload a selfie and find all their photos.**

Built with InsightFace (ArcFace), FAISS, FastAPI, and vanilla HTML/JS. No cloud. No external storage. Runs fully on your machine.

---

## Quick Start

### 1. Clone / enter the project

```bash
cd facefind
```

### 2. Start the server

```bash
bash start.sh
```

The first run will create a `venv` and install all dependencies automatically.

Open **http://localhost:8000** in your browser.

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

### Option B — Use LFW demo dataset

```bash
# Download LFW (~200 MB, ~13k images)
bash scripts/download_lfw.sh

# Then in the Admin UI, click "Index LFW Dataset"
# OR via curl:
curl -X POST http://localhost:8000/api/datasets/lfw
```

---

## Project Structure

```
facefind/
├── backend/
│   ├── app.py                  ← FastAPI server (all endpoints)
│   ├── datasets/               ← Uploaded/downloaded photo datasets
│   │   └── lfw/                ← LFW dataset (after download)
│   ├── embeddings/             ← FAISS indexes + metadata per dataset
│   ├── uploads/                ← Temp storage for selfie searches
│   ├── shares/                 ← (reserved)
│   ├── datasets_meta.json      ← Dataset registry
│   └── shares_meta.json        ← Share link registry
│
├── frontend/
│   ├── index.html              ← Landing page (public)
│   ├── login.html              ← Sign-in / sign-up (public)
│   ├── pricing.html            ← Plan comparison page (public)
│   ├── admin.html              ← Dashboard: upload datasets, generate share links (auth required)
│   ├── download.html           ← Self-hosted download & license key management (auth + paid plan required)
│   └── share.html              ← Public: selfie upload & results (no auth required)
│
├── scripts/
│   └── download_lfw.sh         ← Download LFW Funneled dataset
│
├── requirements.txt
├── start.sh                    ← One-command launcher
├── robots.txt                  ← Disallows /admin.html, /download.html, /api/
├── sitemap.xml                 ← Lists /, /pricing.html, /login.html
└── README.md
```

---

## Self-Hosted Download Flow

Paid plan users (Personal Lite and above) can download the self-hosted executable from `download.html`. The full flow is:

```
User logs in → visits /download.html
  → GET /api/download/info          (check eligibility + existing key)
  → POST /api/download/generate-key (issue or regenerate a license key)
  → POST /api/download/request-link (get a 15-min one-use signed download URL)
  → GET /api/download/file?token=…  (stream facefind-selfhosted.zip)
```

Once downloaded and running, the self-hosted app validates its license on startup and periodically thereafter:

```
Self-hosted app boots
  → POST /api/license/validate  { key, machine_id }
    ← 200 { valid, plan, limits, expires_at, offline_grace_hours }
    ← 403 { valid: false, reason }  (revoked / expired / unknown key)
```

### Railway deployment prerequisite

Place the self-hosted ZIP on the Railway Volume before enabling downloads:

```bash
# On your Railway volume (mounted at /data by default):
mkdir -p /data/releases
cp facefind-selfhosted.zip /data/releases/facefind-selfhosted.zip
```

Or set the `EXECUTABLE_PATH` environment variable to point to a different path:

```
EXECUTABLE_PATH=/data/releases/facefind-selfhosted.zip
```

The `/api/download/file` endpoint returns `503` until this file is present.

### License key lifecycle

| Action | Endpoint | Effect |
|--------|----------|--------|
| Generate / regenerate | `POST /api/download/generate-key` | Revokes old key, issues new one |
| Revoke | `POST /api/license/revoke` | Marks key revoked; running instances stop on next check-in |
| Validate (self-hosted app) | `POST /api/license/validate` | Increments activation count, returns plan limits |

### Per-plan self-hosted limits

| Plan | Max images | Max datasets | Simultaneous machines | Offline grace |
|------|-----------|--------------|----------------------|---------------|
| Personal Lite | 2,000 | 5 | 1 | 24 h |
| Personal Pro | 10,000 | 15 | 2 | 72 h |
| Personal Max | 30,000 | 30 | 3 | 168 h |
| Studio Starter | 100,000 | 50 | 3 | 168 h |
| Studio Pro | 500,000 | Unlimited | 5 | 336 h |

Free plan does **not** include self-hosted access.

---

## API Reference

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check — returns Redis, DB, and email config status |

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create account (triggers OTP email) |
| POST | `/api/auth/verify-otp` | Verify email OTP |
| POST | `/api/auth/login` | Sign in with email + password |
| GET | `/api/auth/me` | Get current user info (plan, name, email) |
| POST | `/api/auth/logout` | Invalidate session cookie |

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/datasets` | List all datasets for current user |
| POST | `/api/datasets/upload-zip` | Upload ZIP dataset |
| POST | `/api/datasets/gdrive` | Link a public Google Drive folder |
| GET | `/api/datasets/{id}/status` | Dataset status + progress + ETA |
| DELETE | `/api/datasets/{id}` | Delete dataset, embeddings, and share links |
| GET | `/api/image/{dataset_id}/{path}` | Serve original image file |
| GET | `/api/thumb/{dataset_id}/{path}` | Serve resized thumbnail (cached) |

### Shares

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/shares` | Generate a share link for a ready dataset |
| GET | `/api/shares/{share_id}` | Get share info (dataset name, status) |
| POST | `/api/shares/{share_id}/search` | Search by selfie upload |

### Download & License (authenticated)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/download/info` | Eligibility check + current license key details |
| POST | `/api/download/generate-key` | Issue or regenerate license key (paid plans only) |
| POST | `/api/download/request-link` | Get a 15-min signed one-use download URL |
| GET | `/api/download/file?token=…` | Stream `facefind-selfhosted.zip` (token-gated) |
| POST | `/api/license/revoke` | Revoke the current active license key |

### License Validation (called by self-hosted app)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/license/validate` | Validate key + machine ID; returns plan limits |

### Admin (server-to-server)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/set-plan` | Set a user's plan (requires `X-Admin-Secret` header) |

Interactive docs: **http://localhost:8000/docs**

---

## Stack

| Component | Technology |
|-----------|-----------|
| Face Detection | InsightFace RetinaFace |
| Face Embedding | ArcFace w600k_r50 (512-dim) |
| Vector Search | FAISS IndexFlatIP |
| Backend API | FastAPI + uvicorn |
| Database | PostgreSQL (via Railway plugin) |
| Cache | Redis (optional, via Railway plugin) |
| Email | Gmail API (OAuth2) |
| Storage | Railway Volume (persistent local filesystem) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Uploading Other Datasets Later

Any ZIP file works. Recommended structures:

```
# Structure 1: Labeled (folders = person names)
MyEvent.zip
├── Alice/
│   ├── alice_001.jpg
│   └── alice_002.jpg
└── Bob/
    └── bob_001.jpg

# Structure 2: Flat (all photos in root)
MyEvent.zip
├── IMG_0001.jpg
├── IMG_0002.jpg
└── ...
```

The system detects all faces in all images regardless of folder structure. Folder names become labels in search results.

---

## Performance

- **Embedding speed**: ~3–5 images/sec on CPU (no GPU needed)
- **Search latency**: <5ms for 16k faces
- **LFW full index**: ~78 min on CPU (13k images → 16k face embeddings)
- **Memory**: ~100MB for 16k face vectors

For faster embedding on large datasets, a CUDA GPU reduces time by ~23×.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string (set automatically by Railway Postgres plugin) |
| `REDIS_URL` | optional | Redis connection string — caching disabled if absent |
| `GMAIL_CLIENT_ID` | ✅ | Gmail OAuth2 client ID for sending OTP emails |
| `GMAIL_CLIENT_SECRET` | ✅ | Gmail OAuth2 client secret |
| `GMAIL_REFRESH_TOKEN` | ✅ | Gmail OAuth2 refresh token |
| `ADMIN_SECRET` | ✅ | Secret header value for `/api/admin/set-plan` |
| `EXECUTABLE_PATH` | optional | Path to `facefind-selfhosted.zip` on the volume (default: `/data/releases/facefind-selfhosted.zip`) |
| `DATA_DIR` | optional | Root data directory (default: `/data`) |
| `GOOGLE_API_KEY` | optional | Google Drive API key for listing Drive folders |
| `INSIGHTFACE_MODEL` | optional | InsightFace model name (default: `buffalo_sc`) |
| `DET_SIZE` | optional | Face detection input size in pixels (default: `320`) |
| `UNLOAD_MODEL_AFTER_EMBED` | optional | Unload face model from RAM after embedding to save memory (default: `true`) |
| `MAX_LOADED_INDEXES` | optional | Max FAISS indexes kept in memory at once (default: `2`) |

---

## Notes

- All data stays local — no cloud, no external APIs
- The share link works as long as your server is running
- To share over a network, expose port 8000 (or use ngrok: `ngrok http 8000`)
- `download.html` and `admin.html` are blocked in `robots.txt` and carry `noindex` meta tags — they will not appear in search engine results