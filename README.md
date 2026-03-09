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
│   ├── index.html              ← Admin: upload dataset, generate links
│   └── share.html              ← Public: selfie upload & results
│
├── scripts/
│   └── download_lfw.sh         ← Download LFW Funneled dataset
│
├── requirements.txt
├── start.sh                    ← One-command launcher
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/datasets` | List all datasets |
| POST | `/api/datasets/upload-zip` | Upload ZIP dataset |
| POST | `/api/datasets/lfw` | Register LFW dataset |
| GET | `/api/datasets/{id}/status` | Dataset status + progress |
| POST | `/api/shares` | Generate share link |
| GET | `/api/shares/{share_id}` | Get share info |
| POST | `/api/shares/{share_id}/search` | Search by selfie |
| GET | `/api/image/{dataset_id}/{path}` | Serve image file |

Interactive docs: **http://localhost:8000/docs**

---

## Stack

| Component | Technology |
|-----------|-----------|
| Face Detection | InsightFace RetinaFace |
| Face Embedding | ArcFace w600k_r50 (512-dim) |
| Vector Search | FAISS IndexFlatIP |
| Backend API | FastAPI + uvicorn |
| Storage | Local filesystem |
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

## Notes

- All data stays local — no cloud, no external APIs
- The share link works as long as your server is running
- To share over a network, expose port 8000 (or use ngrok: `ngrok http 8000`)