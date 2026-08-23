# Runtime-only image. Training deps (tensorflow/torch) are installed
# separately below only if you're serving a trained model — see the
# comment near the bottom for enabling that.
FROM python:3.11-slim

WORKDIR /app

# OpenCV needs these system libraries even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uncomment this line once you have a trained model to serve in production
# (adds tensorflow/torch — a much larger image, but needed for real inference):
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]