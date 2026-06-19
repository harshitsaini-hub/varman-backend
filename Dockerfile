FROM python:3.11-slim

# System deps required by OpenCV (cv2) and mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "bcrypt<4.0.0" email-validator aiosqlite

# Create storage directory for SQLite DB and uploaded images
RUN mkdir -p /app/storage

# Copy application code + MediaPipe face detection model
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]