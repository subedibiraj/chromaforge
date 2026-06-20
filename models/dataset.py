"""
ChromaForge — Colorization Dataset
====================================

PyTorch ``Dataset`` that loads RGB images, converts them to CIE-LAB
colour space, and returns normalised ``(L, AB)`` tensor pairs ready
for the generator.

Normalisation convention
------------------------
- **L channel**:  ``L_norm = L / 50 − 1``  → range [-1, 1]
- **AB channels**: ``AB_norm = AB / 128``   → range ≈ [-1, 1]

This matches the Tanh output range of the generator and is consistent
with every existing training script in the project.

Data augmentation
-----------------
When ``augment=True``:

1. Random horizontal flip (p = 0.5)
2. Random brightness jitter (factor ∈ [0.8, 1.2], p = 0.5)

These match the augmentation pipeline used in ``train_local.py`` (the
cGAN training script).  The simpler scripts (``train_cnn_baseline.py``,
``train_unet_l1.py``) only used the horizontal flip; callers that want
that behaviour can set ``brightness_jitter=False``.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset
from torchvision import transforms


class ColorizationDataset(Dataset):
    """Dataset of ``(L, AB)`` tensor pairs for image colorization.

    Args:
        paths:             List of filesystem paths to RGB images.
        size:              Resize target (square).  Default ``256``.
        augment:           Enable data augmentation (flip + brightness).
        brightness_jitter: If ``True`` **and** ``augment`` is ``True``,
                           apply random brightness jitter.  Default
                           ``True`` (matches ``train_local.py``).  Set
                           to ``False`` to replicate the simpler
                           augmentation of ``train_unet_l1.py``.

    Returns per sample:
        ``(L, AB)`` where ``L`` has shape ``(1, H, W)`` and ``AB`` has
        shape ``(2, H, W)``, both ``float32``.
    """

    def __init__(
        self,
        paths: Sequence[str],
        size: int = 256,
        augment: bool = True,
        brightness_jitter: bool = True,
    ) -> None:
        self.paths: List[str] = list(paths)
        self.size = size
        self.augment = augment
        self.brightness_jitter = brightness_jitter

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Robust loading — skip corrupt / truncated files
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.paths))

        # ── Augmentation ────────────────────────────────────────────────
        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.brightness_jitter and random.random() > 0.5:
                img = transforms.functional.adjust_brightness(
                    img, random.uniform(0.8, 1.2)
                )

        # ── Resize & convert to LAB ─────────────────────────────────────
        img = img.resize((self.size, self.size), Image.LANCZOS)
        lab = rgb2lab(np.array(img, dtype=np.float32) / 255.0).astype(np.float32)

        # ── Normalise & tensorise ───────────────────────────────────────
        L  = torch.from_numpy((lab[:, :, 0:1] / 50.0) - 1.0).permute(2, 0, 1)
        AB = torch.from_numpy(lab[:, :, 1:3] / 128.0).permute(2, 0, 1)
        return L, AB
