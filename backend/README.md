---
title: ChromaForge API
emoji: 🎨
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Conditional GAN grayscale image colorization API
---

# ChromaForge — Colorization API

FastAPI backend for the ChromaForge colorization system.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and model status |
| GET | `/health` | Health check |
| POST | `/colorize` | Upload image → receive colorized JPEG |
| POST | `/colorize/base64` | Upload image → receive JSON with base64 image |

## Usage

```bash
curl -X POST "https://your-space.hf.space/colorize" \
  -F "file=@your_image.jpg" \
  --output colorized.jpg
```

## Model

U-Net Generator + PatchGAN Discriminator trained on COCO 2017 (118k images).
See the main repository for training code and metrics.
