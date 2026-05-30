"""
ChromaForge — Grayscale Image Colorization API
FastAPI backend with async inference, proper error handling, and HF Spaces compatibility.
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from skimage.transform import resize

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("MODEL_PATH", "weights/generator.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# ── Generator Architecture (U-Net with skip connections) ─────────────────────

class ConvBlock(nn.Module):
    """Encoder block: Conv → BatchNorm → LeakyReLU"""
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """Decoder block: ConvTranspose → BatchNorm → ReLU (+ optional Dropout)"""
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net Generator for image colorization.
    Input:  (B, 1, 256, 256)  — L channel, normalized to [-1, 1]
    Output: (B, 2, 256, 256)  — AB channels, normalized to [-1, 1]
    
    Architecture follows Isola et al. (2017) "Image-to-Image Translation
    with Conditional Adversarial Networks" (pix2pix).
    """

    def __init__(self):
        super().__init__()
        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )                                        # 256 → 128
        self.enc2 = ConvBlock(64,  128)          # 128 → 64
        self.enc3 = ConvBlock(128, 256)          # 64  → 32
        self.enc4 = ConvBlock(256, 512)          # 32  → 16
        self.enc5 = ConvBlock(512, 512)          # 16  → 8
        self.enc6 = ConvBlock(512, 512)          # 8   → 4
        self.enc7 = ConvBlock(512, 512)          # 4   → 2
        self.enc8 = ConvBlock(512, 512, use_bn=False)  # 2 → 1 (bottleneck)

        # ── Decoder (with skip connections) ─────────────────────────────────
        self.dec1 = DeconvBlock(512,  512, dropout=True)   # 1  → 2
        self.dec2 = DeconvBlock(1024, 512, dropout=True)   # 2  → 4
        self.dec3 = DeconvBlock(1024, 512, dropout=True)   # 4  → 8
        self.dec4 = DeconvBlock(1024, 512)                 # 8  → 16
        self.dec5 = DeconvBlock(1024, 256)                 # 16 → 32
        self.dec6 = DeconvBlock(512,  128)                 # 32 → 64
        self.dec7 = DeconvBlock(256,  64)                  # 64 → 128
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(128, 2, 4, stride=2, padding=1),
            nn.Tanh(),
        )                                                  # 128 → 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        # Decode with skip connections (U-Net concat)
        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        return self.dec8(torch.cat([d7, e1], dim=1))


# ── Image Processing ─────────────────────────────────────────────────────────

def preprocess(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """
    Convert raw image bytes → model-ready tensor.
    Returns (tensor, original_L_channel) for reconstruction.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_resized = resize(img_np, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    img_lab = rgb2lab(img_resized).astype(np.float32)

    L = img_lab[:, :, 0]                        # [0, 100]
    L_norm = (L / 50.0) - 1.0                   # → [-1, 1]
    tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor, L


def postprocess(pred_ab: torch.Tensor, L: np.ndarray) -> bytes:
    """
    Merge predicted AB channels with original L → RGB image bytes (JPEG).
    """
    ab = pred_ab.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,2)
    ab = ab * 128.0                              # [-1,1] → [-128, 128]

    lab = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab

    rgb = lab2rgb(lab)                           # [0, 1]
    rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)

    out = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ── App Lifecycle ─────────────────────────────────────────────────────────────

generator: Optional[UNetGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights on startup; clean up on shutdown."""
    global generator
    logger.info(f"Loading generator from {MODEL_PATH} on {DEVICE}")
    generator = UNetGenerator().to(DEVICE)

    if MODEL_PATH.exists():
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        generator.load_state_dict(state)
        logger.info("Weights loaded successfully.")
    else:
        logger.warning(
            "No weights file found — running with random weights. "
            "Place a trained generator.pth in weights/ to enable real colorization."
        )

    generator.eval()
    yield
    logger.info("Shutting down ChromaForge API.")


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="ChromaForge API",
    description="Automatic grayscale image colorization using a conditional GAN.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "ChromaForge",
        "version": "2.0.0",
        "device": str(DEVICE),
        "model_loaded": generator is not None and MODEL_PATH.exists(),
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/colorize", tags=["Inference"])
async def colorize(file: UploadFile = File(...)):
    """
    Colorize a grayscale image.

    - **file**: JPEG, PNG, WebP, or BMP image (max 10 MB)
    - Returns: colorized JPEG image as binary response
    """
    # ── Validate ─────────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw) / 1e6:.1f} MB). Max: 10 MB.",
        )

    if generator is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    # ── Inference ─────────────────────────────────────────────────────────────
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Colorizing image ({len(raw) / 1024:.1f} KB)")
    t0 = time.perf_counter()

    try:
        tensor, L = preprocess(raw)
        tensor = tensor.to(DEVICE)

        with torch.inference_mode():
            pred_ab = generator(tensor)

        result_bytes = postprocess(pred_ab, L)

    except Exception as exc:
        logger.exception(f"[{request_id}] Inference failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")

    elapsed = time.perf_counter() - t0
    logger.info(f"[{request_id}] Done in {elapsed:.3f}s")

    return Response(
        content=result_bytes,
        media_type="image/jpeg",
        headers={
            "X-Request-Id": request_id,
            "X-Inference-Time": f"{elapsed:.3f}",
        },
    )


@app.post("/colorize/base64", tags=["Inference"])
async def colorize_base64(file: UploadFile = File(...)):
    """Same as /colorize but returns JSON with base64-encoded image."""
    import base64
    response = await colorize(file)
    encoded = base64.b64encode(response.body).decode()
    return JSONResponse({
        "image": f"data:image/jpeg;base64,{encoded}",
        "inference_time": response.headers.get("X-Inference-Time"),
    })
