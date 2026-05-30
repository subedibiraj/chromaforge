"""
PatchGAN Discriminator for ChromaForge.

Instead of classifying the entire image as real/fake, PatchGAN classifies
overlapping NxN patches. This penalizes structure at the scale of patches,
which is more effective for high-frequency detail (textures, edges).

Reference: Isola et al. (2017) — "Image-to-Image Translation with
Conditional Adversarial Networks"
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.
    
    Input:  (B, 3, 256, 256) — concatenated [L, AB_real_or_fake]
    Output: (B, 1, 30, 30)   — patch-level real/fake scores
    
    Each output unit has a receptive field of 70×70 pixels in the input,
    so the model reasons about local realism rather than global structure.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def block(in_c: int, out_c: int, normalize: bool = True) -> list:
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64,  normalize=False),  # 256 → 128
            *block(64,  128),                            # 128 → 64
            *block(128, 256),                            # 64  → 32
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            # No sigmoid — use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        """
        Args:
            L:  (B, 1, H, W) — grayscale L channel (condition)
            ab: (B, 2, H, W) — real or generated AB channels
        Returns:
            (B, 1, 30, 30) — patch-level logits
        """
        x = torch.cat([L, ab], dim=1)   # Conditional: discriminator sees L
        return self.model(x)


class GANLoss(nn.Module):
    """
    Wraps BCEWithLogitsLoss for GAN training.
    Supports label smoothing to stabilize training.
    """

    def __init__(self, real_label: float = 0.9, fake_label: float = 0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.BCEWithLogitsLoss()

    def _label_tensor(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        val = self.real_label if is_real else self.fake_label
        return torch.full_like(pred, val)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        labels = self._label_tensor(pred, is_real)
        return self.loss(pred, labels)
