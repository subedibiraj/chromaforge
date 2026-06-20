"""
ChromaForge — CNN Regression Baseline
======================================

Five-layer fully convolutional network that predicts AB channels from L
using only L1 loss (no skip connections, no adversarial component).

This model establishes the lower-bound baseline on colorfulness and the
reference point for isolating the contribution of the U-Net architecture
and the cGAN objective in the main comparison study.

Architecture
------------
All convolutions use 3×3 kernels with ``padding=1`` (same-padding), so
the spatial resolution is preserved throughout — no downsampling or
upsampling occurs.

::

    Layer 1: Conv(1→64,   3×3) → BN(64)  → ReLU
    Layer 2: Conv(64→128, 3×3) → BN(128) → ReLU
    Layer 3: Conv(128→128,3×3) → BN(128) → ReLU
    Layer 4: Conv(128→64, 3×3) → BN(64)  → ReLU
    Layer 5: Conv(64→2,   3×3) → Tanh

Layer naming
------------
``self.net`` is a flat ``nn.Sequential`` with indices 0–14.  Existing
checkpoints use keys ``net.0.weight``, ``net.1.weight``, etc.

Parameter count
---------------
- **CNNBaseline total: ≈ 0.17 M (168,386) parameters**

Input / Output
--------------
- Input:  ``(B, 1, H, W)`` — L channel in [-1, 1]
- Output: ``(B, 2, H, W)`` — AB channels in [-1, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """Plain 5-layer fully convolutional regressor for L → AB.

    No skip connections, no downsampling / upsampling — operates at full
    resolution throughout via same-padding convolutions, isolating the
    effect of architecture depth and capacity from the skip-connection
    and adversarial contributions tested separately.

    Input:  ``(B, 1, H, W)``  — L channel normalised to [-1, 1]
    Output: ``(B, 2, H, W)``  — AB channels normalised to [-1, 1]

    Total trainable parameters: **≈ 0.17 M**
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,   64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  2,   3, padding=1), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, 1, H, W)`` L-channel tensor in [-1, 1].

        Returns:
            ``(B, 2, H, W)`` predicted AB channels in [-1, 1].
        """
        return self.net(x)
