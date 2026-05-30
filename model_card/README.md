---
language: en
license: mit
tags:
  - image-colorization
  - conditional-gan
  - pix2pix
  - u-net
  - computer-vision
  - pytorch
datasets:
  - coco
metrics:
  - psnr
  - ssim
---

# ChromaForge — Grayscale Image Colorization

ChromaForge is a conditional GAN that colorizes grayscale photographs.
It uses a **U-Net generator** (54M parameters) trained with a
**PatchGAN discriminator** and a combined adversarial + L1 loss.

## Model description

- **Architecture**: U-Net Generator + 70×70 PatchGAN Discriminator
- **Input**: L channel of a LAB image, normalized to [-1, 1], shape (1, 256, 256)
- **Output**: Predicted AB channels, shape (2, 256, 256)
- **Color space**: CIE LAB (perceptually uniform; decouples luminance from color)
- **Framework**: PyTorch 2.2+

## Training data

COCO 2017 training split (118,000 images), 90/10 train/val split.
Images resized to 256×256. Augmentation: horizontal flip, random brightness ∈ [0.8, 1.2].

## Training procedure

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 100 |
| Batch size | 16 |
| LR (G and D) | 2e-4 |
| Adam β₁ | 0.5 |
| λ_L1 | 100 |
| LR decay start | Epoch 50 |
| Hardware | NVIDIA T4 (~7 hours) |

## Evaluation results

| Metric | Value |
|--------|-------|
| PSNR | 26.4 dB |
| SSIM | 0.892 |
| Colorfulness (Hasler–Süsstrunk) | 31.2 |

## Usage

```python
import torch
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

# Load model (UNetGenerator defined in main.py)
model = UNetGenerator()
model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
model.eval()

# Preprocess
img = np.array(Image.open("gray.jpg").convert("RGB")) / 255.0
img_resized = resize(img, (256, 256))
lab = rgb2lab(img_resized).astype(np.float32)
L = torch.from_numpy((lab[:,:,0] / 50.0) - 1.0).unsqueeze(0).unsqueeze(0)

# Inference
with torch.inference_mode():
    ab = model(L).squeeze(0).permute(1, 2, 0).numpy() * 128

# Reconstruct
result_lab = np.zeros((256, 256, 3), np.float32)
result_lab[:,:,0] = lab[:,:,0]
result_lab[:,:,1:] = ab
result_rgb = (lab2rgb(result_lab) * 255).clip(0,255).astype(np.uint8)
Image.fromarray(result_rgb).save("colorized.jpg")
```

## Limitations

- Fixed 256×256 resolution (tile-based inference needed for larger images)
- No explicit semantic conditioning; color ambiguity remains for some textures
- Trained on COCO — performance degrades on strongly non-photographic images

## Citation

```bibtex
@article{chromaforge2024,
  title   = {ChromaForge: Automatic Grayscale Image Colorization via Conditional GANs},
  author  = {Your Name},
  year    = {2024},
  url     = {https://github.com/your-username/chromaforge}
}
```
