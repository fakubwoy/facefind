FROM python:3.11-slim

# System deps for OpenCV + InsightFace (g++/gcc needed to compile Cython extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    g++ gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Data directory (Railway Volume will be mounted here)
RUN mkdir -p /data/datasets /data/embeddings /data/uploads

EXPOSE 8000

# Tell glibc to return freed memory to the OS promptly instead of hoarding it.
# This is the single biggest win for Python ML workloads on Railway.
ENV MALLOC_TRIM_THRESHOLD_=65536
# Use the small InsightFace model by default (overridable via Railway env vars)
ENV INSIGHTFACE_MODEL=buffalo_sc
ENV DET_SIZE=320
# Unload the model from RAM after batch embedding finishes
ENV UNLOAD_MODEL_AFTER_EMBED=true

# Single worker — avoids loading the ~300MB model N times in parallel.
# Railway scales horizontally via replicas, not per-process workers.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]