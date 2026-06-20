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

COCO 2017 training split (118,287 images), 90/10 train/val split.
Images resized to 256×256. Augmentation: horizontal flip, random brightness ∈ [0.8, 1.2].

## Training procedure

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 100 |
| Batch size | 8 |
| LR (G and D) | 2e-4 |
| Adam β₁ | 0.5 |
| λ_L1 | 100 |
| LR decay start | Epoch 50 (linear decay to 0) |
| Hardware | NVIDIA RTX 3060 (~18 hours) |

## Evaluation results

Validation set metrics (COCO 2017, 90/10 split):

| Model | PSNR (dB) ↑ | SSIM ↑ | Colorfulness ↑ |
|-------|-------------|--------|----------------|
| CNN Baseline | 23.24 | 0.909 | 13.7 |
| U-Net + L1 | **23.74** | **0.914** | 16.1 |
| **U-Net + cGAN** | 23.67 | 0.891 | **31.8** |

The cGAN model achieves colorfulness scores within the range of natural
photographs (25–40) while maintaining competitive PSNR and SSIM. The
regression baselines produce higher PSNR by predicting desaturated
(near-grey) outputs — a known limitation of L1/MSE losses.

## Usage

```python
import torch
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

# Load model (UNetGenerator defined in models/generator.py)
from models.generator import UNetGenerator

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
- Baseline models trained with less compute (see EXPERIMENTS.md)

## Citation

```bibtex
@techreport{subedi2025chromaforge,
  title   = {ChromaForge: A Comparative Study of Deep Learning Approaches
             for Automatic Grayscale Image Colorization},
  author  = {Subedi, Biraj},
  year    = {2025},
  url     = {https://github.com/birajsubedi/chromaforge}
}
```
