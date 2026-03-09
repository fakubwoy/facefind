FROM python:3.11-slim

# System deps for OpenCV + InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
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

# Railway injects PORT env varFROM python:3.11-slim

# System deps for OpenCV + InsightFace
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

# Railway injects PORT env var
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]