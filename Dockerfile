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

# Temp scratch dirs — bulk data lives in B2
RUN mkdir -p /tmp/facefind/datasets /tmp/facefind/embeddings /tmp/facefind/uploads

EXPOSE 8080

ENV MALLOC_TRIM_THRESHOLD_=65536
ENV INSIGHTFACE_MODEL=buffalo_sc
ENV DET_SIZE=320
ENV UNLOAD_MODEL_AFTER_EMBED=true

# Cloud Run injects PORT env var — defaults to 8080
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]