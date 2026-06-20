"""
ChromaForge — Shared Utilities
================================

Helper functions used across training, evaluation, and inference:

- :func:`weights_init` — Normal initialisation for Conv and BatchNorm layers
- :func:`lab_tensor_to_rgb` — Convert LAB tensors back to uint8 RGB numpy arrays
- :func:`colorfulness` — Hasler & Süsstrunk (2003) colorfulness metric
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from skimage.color import lab2rgb


# ── Weight Initialisation ───────────────────────────────────────────────────


def weights_init(m: nn.Module) -> None:
    """Initialise Conv and BatchNorm layers following the pix2pix convention.

    - **Conv / ConvTranspose** weights ← ``N(0, 0.02)``
    - **BatchNorm** weights ← ``N(1, 0.02)``, biases ← ``0``

    Usage::

        model.apply(weights_init)

    Args:
        m: A single ``nn.Module`` (called via ``model.apply``).
    """
    cls = m.__class__.__name__
    if "Conv" in cls and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cls and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ── LAB → RGB Conversion ────────────────────────────────────────────────────


def lab_tensor_to_rgb(
    L_tensor: torch.Tensor, ab_tensor: torch.Tensor
) -> np.ndarray:
    """Convert a single-sample LAB tensor pair to a uint8 RGB numpy array.

    The tensors are expected to use the project's normalisation convention:

    - **L**:  ``(1, H, W)``, range [-1, 1]  (``L_real = (L_tensor + 1) * 50``)
    - **AB**: ``(2, H, W)``, range ≈[-1, 1] (``AB_real = AB_tensor * 128``)

    Args:
        L_tensor:  ``(1, H, W)`` L channel tensor.
        ab_tensor: ``(2, H, W)`` AB channel tensor.

    Returns:
        ``(H, W, 3)`` uint8 numpy array in RGB order, clipped to [0, 255].
    """
    L_np = ((L_tensor[0].cpu().numpy() + 1) * 50).clip(0, 100)
    ab_np = (ab_tensor.permute(1, 2, 0).cpu().numpy() * 128).clip(-128, 128)

    lab = np.zeros((*L_np.shape, 3), dtype=np.float32)
    lab[:, :, 0] = L_np
    lab[:, :, 1:] = ab_np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (lab2rgb(lab) * 255).clip(0, 255).astype(np.uint8)


# ── Colorfulness Metric ─────────────────────────────────────────────────────


def colorfulness(rgb: np.ndarray) -> float:
    """Compute the Hasler & Süsstrunk (2003) colorfulness metric.

    Higher values indicate more vivid / saturated colours.  Useful for
    comparing GAN output (typically higher) with L1-only output (tends
    towards desaturated averages).

    The metric is defined as::

        C = sqrt(σ_rg² + σ_yb²) + 0.3 * sqrt(μ_rg² + μ_yb²)

    where ``rg = R − G`` and ``yb = 0.5(R + G) − B``.

    Args:
        rgb: ``(H, W, 3)`` image in any numeric dtype (will be cast to
             float internally).

    Returns:
        Scalar colorfulness score.
    """
    R = rgb[:, :, 0].astype(float)
    G = rgb[:, :, 1].astype(float)
    B = rgb[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    return (
        math.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * math.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )
