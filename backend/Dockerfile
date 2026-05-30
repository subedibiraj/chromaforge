# ── ChromaForge — Hugging Face Spaces Dockerfile ────────────────────────────
# Deploys the FastAPI colorization backend.
# HF Spaces free tier: 16 GB RAM, 2 vCPU (CPU inference only).
# GPU Space (T4): set hardware to "t4-small" in README.md YAML.

FROM python:3.11-slim

# System deps for scikit-image and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create weights directory (populated at runtime or via HF model hub)
RUN mkdir -p weights

# HF Spaces exposes port 7860
EXPOSE 7860

ENV MODEL_PATH=/app/weights/generator.pth
ENV ALLOWED_ORIGINS=*

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
