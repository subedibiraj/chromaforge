"""
ChromaForge — PatchGAN Discriminator & GAN Loss
=================================================

70×70 PatchGAN discriminator for conditional image-to-image translation.

Instead of classifying the entire image as real / fake, PatchGAN classifies
overlapping N×N patches.  This penalises structure at the scale of patches
and is more effective for high-frequency detail (textures, edges).

Reference
---------
Isola et al. (2017) — *"Image-to-Image Translation with Conditional
Adversarial Networks"*

Layer naming
------------
``self.model`` is a flat ``nn.Sequential`` with indices 0–9.  Existing
checkpoints saved via ``D.state_dict()`` use keys like
``model.0.weight``, ``model.3.weight``, etc.  This module preserves that
layout exactly.

Parameter count
---------------
- **PatchDiscriminator total: ≈ 2.77 M parameters**
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator (conditional).

    The discriminator receives the concatenation of the L channel
    (condition) and the AB channels (real **or** generated) as a
    3-channel input, and outputs a grid of per-patch logits.

    Architecture (for 256×256 input)::

        block1  : Conv(3→64,   4×4, s2) → LeakyReLU   → 128×128  (no BN)
        block2  : Conv(64→128, 4×4, s2) → BN → LReLU  → 64×64
        block3  : Conv(128→256,4×4, s2) → BN → LReLU  → 32×32
        ZeroPad : pad (1,0,1,0)                        → 33×33
        Conv    : Conv(256→512,4×4, s1, p1) → BN → LReLU → 32×32
        ZeroPad : pad (1,0,1,0)                        → 33×33
        Conv    : Conv(512→1,  4×4, s1, p1)            → 32×32

    The effective receptive field of each output unit is 70×70 pixels.

    Input:
        - ``L``:  ``(B, 1, 256, 256)`` — grayscale L channel (condition)
        - ``ab``: ``(B, 2, 256, 256)`` — real or generated AB channels

    Output:
        ``(B, 1, 30, 30)`` — per-patch real/fake logits (no sigmoid)

    Total trainable parameters: **≈ 2.77 M**

    Args:
        in_channels: Number of input channels after concatenation.
                     Default is 3 (1 L + 2 AB).
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        def block(in_c: int, out_c: int, norm: bool = True) -> list[nn.Module]:
            """Discriminator conv block factory."""
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=not norm)
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, norm=False),   # 256 → 128
            *block(64,  128),                       # 128 → 64
            *block(128, 256),                       # 64  → 32
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            # No sigmoid — use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            L:  ``(B, 1, H, W)`` — grayscale L channel (condition).
            ab: ``(B, 2, H, W)`` — real or generated AB channels.

        Returns:
            ``(B, 1, 30, 30)`` patch-level logits (for 256×256 input).
        """
        return self.model(torch.cat([L, ab], dim=1))


class GANLoss(nn.Module):
    """Binary cross-entropy GAN loss with label smoothing.

    Wraps ``nn.BCEWithLogitsLoss`` and generates target tensors
    automatically from a boolean ``is_real`` flag.

    Label smoothing (``real_label=0.9`` by default) stabilises
    discriminator training by preventing the discriminator from becoming
    over-confident early in training.

    Args:
        real_label: Target value for **real** examples (default ``0.9``).
        fake_label: Target value for **fake** examples (default ``0.0``).
    """

    def __init__(self, real_label: float = 0.9, fake_label: float = 0.0) -> None:
        super().__init__()
        self.real = real_label
        self.fake = fake_label
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Compute loss.

        Args:
            pred:    Raw logits from the discriminator.
            is_real: If ``True``, targets are ``real_label``; otherwise
                     ``fake_label``.

        Returns:
            Scalar BCE loss.
        """
        target = torch.full_like(pred, self.real if is_real else self.fake)
        return self.bce(pred, target)
