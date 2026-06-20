"""
ChromaForge — Shared Models Package
=====================================

Canonical, single-source-of-truth definitions for every neural network
architecture, dataset class, and utility function used in the ChromaForge
project (grayscale image colorization with conditional GANs).

Quick start::

    from chromaforge.models import (
        UNetGenerator,
        PatchDiscriminator,
        GANLoss,
        CNNBaseline,
        ColorizationDataset,
        weights_init,
        lab_tensor_to_rgb,
        colorfulness,
    )

All layer names are kept identical to the original per-script definitions
so that existing trained checkpoints load without key remapping.
"""

from models.generator import ConvBlock, DeconvBlock, UNetGenerator
from models.discriminator import PatchDiscriminator, GANLoss
from models.baseline import CNNBaseline
from models.dataset import ColorizationDataset
from models.utils import weights_init, lab_tensor_to_rgb, colorfulness

__all__ = [
    # Generator
    "ConvBlock",
    "DeconvBlock",
    "UNetGenerator",
    # Discriminator
    "PatchDiscriminator",
    "GANLoss",
    # Baseline
    "CNNBaseline",
    # Dataset
    "ColorizationDataset",
    # Utilities
    "weights_init",
    "lab_tensor_to_rgb",
    "colorfulness",
]
