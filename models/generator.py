"""
ChromaForge — U-Net Generator
==============================

Pix2pix-style U-Net generator for L → AB colorization.

Architecture follows Isola et al. (2017) *"Image-to-Image Translation
with Conditional Adversarial Networks"*. Eight encoder stages downsample
256 → 1, eight decoder stages upsample back to 256 with skip connections
concatenated along the channel dimension.

Layer names (``enc1`` … ``enc8``, ``dec1`` … ``dec8``) are kept identical
to every existing checkpoint so that ``model.load_state_dict(...)`` works
without any key remapping.

Parameter count
---------------
- ConvBlock / DeconvBlock helpers: ~internal
- **UNetGenerator total: ≈ 54.4 M parameters**
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Encoder building block: ``Conv2d(4×4, stride=2) → [BN] → LeakyReLU``.

    Halves spatial resolution at each stage.

    Args:
        in_ch:  Number of input channels.
        out_ch: Number of output channels.
        use_bn: If ``True`` (default), inserts ``BatchNorm2d`` between the
                convolution and the activation and sets ``bias=False`` on
                the convolution (BN absorbs the bias).

    Shape:
        - Input:  ``(B, in_ch,  H, W)``
        - Output: ``(B, out_ch, H/2, W/2)``
    """

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True) -> None:
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
    """Decoder building block: ``ConvTranspose2d(4×4, stride=2) → BN → ReLU [→ Dropout]``.

    Doubles spatial resolution at each stage.

    Args:
        in_ch:   Number of input channels (includes skip-connection channels).
        out_ch:  Number of output channels.
        dropout: If ``True``, appends ``Dropout(0.5)`` after ReLU (used in
                 the first three decoder stages to prevent overfitting).

    Shape:
        - Input:  ``(B, in_ch,  H, W)``
        - Output: ``(B, out_ch, 2H, 2W)``
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False) -> None:
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
    """Pix2pix U-Net generator for grayscale → colour (L → AB).

    The network follows the classic 8-layer encoder / 8-layer decoder
    design from Isola et al. (2017).  Skip connections concatenate each
    encoder feature map with the corresponding decoder feature map along
    the channel axis.

    Architecture summary (for 256×256 input)::

        Encoder                          Decoder
        -------                          -------
        enc1: 1  → 64   (256→128)       dec1: 512  → 512  (1→2)     + dropout
        enc2: 64 → 128  (128→64)        dec2: 1024 → 512  (2→4)     + dropout
        enc3: 128→ 256  (64→32)         dec3: 1024 → 512  (4→8)     + dropout
        enc4: 256→ 512  (32→16)         dec4: 1024 → 512  (8→16)
        enc5: 512→ 512  (16→8)          dec5: 1024 → 256  (16→32)
        enc6: 512→ 512  (8→4)           dec6: 512  → 128  (32→64)
        enc7: 512→ 512  (4→2)           dec7: 256  → 64   (64→128)
        enc8: 512→ 512  (2→1, no BN)    dec8: 128  → 2    (128→256) + Tanh

    Input:  ``(B, 1, 256, 256)``  — L channel normalised to [-1, 1]
    Output: ``(B, 2, 256, 256)``  — AB channels normalised to [-1, 1]

    Total trainable parameters: **≈ 54.4 M**
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )  # 256 → 128, no BatchNorm (first layer)
        self.enc2 = ConvBlock(64,  128)           # 128 → 64
        self.enc3 = ConvBlock(128, 256)           # 64  → 32
        self.enc4 = ConvBlock(256, 512)           # 32  → 16
        self.enc5 = ConvBlock(512, 512)           # 16  → 8
        self.enc6 = ConvBlock(512, 512)           # 8   → 4
        self.enc7 = ConvBlock(512, 512)           # 4   → 2
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
        )  # 128 → 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, 1, 256, 256)`` L-channel tensor in [-1, 1].

        Returns:
            ``(B, 2, 256, 256)`` predicted AB channels in [-1, 1].
        """
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decode with skip connections (U-Net concatenation)
        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        return self.dec8(torch.cat([d7, e1], dim=1))
