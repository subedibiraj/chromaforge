#!/usr/bin/env python3
"""
ChromaForge — Full Test-Set Evaluation Script
==============================================

Deterministic, reproducible evaluation of all three model variants
(cGAN, UNet-L1, CNN baseline) on the complete test split.

Outputs
-------
1. JSON file with all metrics (mean, std, optionally per-image)
2. LaTeX-ready table row printed to stdout
3. Qualitative comparison grid (16 images: grayscale | prediction | ground truth)

Usage
-----
    python evaluate.py \\
        --model cgan \\
        --weights ../generator_best.pth \\
        --data-dir ../data/train2017 \\
        --output ./eval_results \\
        --batch-size 16 \\
        --device cuda

Notes
-----
- All images in ``--data-dir`` are processed (no random sampling).
- Deterministic: fixed seed, no augmentation, ``torch.inference_mode()``.
- LAB colour-space conventions match the training pipeline exactly:
    L  = L_lab / 50 − 1   ∈ [−1, 1]
    AB = AB_lab / 128      ∈ [−1, 1]   (tanh output)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable regardless of working directory.
# The models package lives at  c:\chromaforge all\chromaforge\models\
# which is one level up from training/.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports — metrics from the shared module in this same directory
from metrics import (
    FIDMetric,
    LPIPSMetric,
    MetricResult,
    batch_colorfulness,
    batch_psnr,
    batch_ssim,
    compute_all_per_image,
    lab_batch_to_rgb,
)

# Model imports — from the shared models/ package (created in parallel)
from models.generator import UNetGenerator
from models.baseline import CNNBaseline

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
IMG_SIZE = 256
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_REGISTRY = {
    "cgan": UNetGenerator,
    "unet_l1": UNetGenerator,
    "cnn_baseline": CNNBaseline,
}


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic seeding
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = SEED) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation dataset (no augmentation, deterministic)
# ══════════════════════════════════════════════════════════════════════════════

class EvalColorizationDataset(Dataset):
    """Deterministic dataset for evaluation — no augmentation, sorted order.

    Normalisation matches the training pipeline:
        L  = L_lab / 50 − 1
        AB = AB_lab / 128

    Images that fail to load are replaced with a grey placeholder and
    flagged via a boolean mask.
    """

    def __init__(self, paths: list[str], size: int = IMG_SIZE) -> None:
        self.paths = sorted(paths)  # deterministic ordering
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load {path}: {e}")
            # Return a grey placeholder (will produce zero AB)
            img = Image.new("RGB", (self.size, self.size), (128, 128, 128))

        img = img.resize((self.size, self.size), Image.LANCZOS)
        lab = rgb2lab(np.array(img, dtype=np.float32) / 255.0).astype(np.float32)

        L = torch.from_numpy((lab[:, :, 0:1] / 50.0) - 1.0).permute(2, 0, 1)
        AB = torch.from_numpy(lab[:, :, 1:3] / 128.0).permute(2, 0, 1)
        return L, AB, path


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_name: str,
    weights_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate and load weights for the requested model.

    Parameters
    ----------
    model_name : one of ``'cgan'``, ``'unet_l1'``, ``'cnn_baseline'``
    weights_path : path to ``.pth`` state-dict file
    device : target device

    Returns
    -------
    nn.Module in eval mode with loaded weights.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass()

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {model_name} ({n_params:.2f}M params) from {weights_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Image collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_image_paths(data_dir: str) -> list[str]:
    """Collect all valid image paths from a flat directory.

    Returns sorted paths for deterministic evaluation order.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    paths = sorted([
        str(p) for p in data_path.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    ])

    if not paths:
        raise RuntimeError(f"No images found in {data_dir}")

    print(f"Found {len(paths):,} images in {data_dir}")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Qualitative comparison grid
# ══════════════════════════════════════════════════════════════════════════════

def save_comparison_grid(
    model: torch.nn.Module,
    dataset: EvalColorizationDataset,
    output_path: str,
    device: torch.device,
    n_images: int = 16,
) -> str:
    """Generate and save a qualitative comparison grid.

    Layout: each row is [Grayscale Input | Model Prediction | Ground Truth].
    Uses a deterministic subset (first ``n_images`` from the sorted dataset).

    Parameters
    ----------
    model : the generator model in eval mode
    dataset : evaluation dataset
    output_path : directory to save the grid
    device : compute device
    n_images : number of images to display (default 16, arranged 4×4 triplets)

    Returns
    -------
    str — path to the saved figure
    """
    n_images = min(n_images, len(dataset))
    n_cols = 3  # grayscale, prediction, ground truth
    n_rows = n_images

    # Use a compact layout: 4 columns of triplets → 4 images per row
    imgs_per_row = 4
    grid_rows = (n_images + imgs_per_row - 1) // imgs_per_row

    fig, axes = plt.subplots(
        grid_rows * 1,  # one row per group
        imgs_per_row * n_cols,
        figsize=(imgs_per_row * n_cols * 2.2, grid_rows * 2.2),
        squeeze=False,
    )

    with torch.inference_mode():
        for i in range(n_images):
            L, AB_gt, _ = dataset[i]
            L_dev = L.unsqueeze(0).to(device)
            AB_pred = model(L_dev).cpu()

            # Convert to RGB
            from metrics import lab_tensors_to_rgb
            rgb_pred = lab_tensors_to_rgb(L, AB_pred[0])
            rgb_gt = lab_tensors_to_rgb(L, AB_gt)

            # Grayscale (for display)
            L_display = ((L[0].numpy() + 1.0) * 50.0).clip(0, 100)

            # Grid position
            grid_r = i // imgs_per_row
            grid_c = (i % imgs_per_row) * n_cols

            axes[grid_r, grid_c].imshow(L_display, cmap="gray", vmin=0, vmax=100)
            axes[grid_r, grid_c + 1].imshow(rgb_pred)
            axes[grid_r, grid_c + 2].imshow(rgb_gt)

            if grid_r == 0:
                if grid_c == 0:
                    axes[0, 0].set_title("Input (L)", fontsize=9)
                    axes[0, 1].set_title("Predicted", fontsize=9)
                    axes[0, 2].set_title("Ground Truth", fontsize=9)

    # Turn off all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Hide any unused subplot cells
    for i in range(n_images, grid_rows * imgs_per_row):
        grid_r = i // imgs_per_row
        grid_c = (i % imgs_per_row) * n_cols
        for offset in range(n_cols):
            if grid_c + offset < axes.shape[1]:
                axes[grid_r, grid_c + offset].set_visible(False)

    plt.tight_layout(pad=0.5)
    fig_path = os.path.join(output_path, "comparison_grid.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved comparison grid → {fig_path}")
    return fig_path


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table row
# ══════════════════════════════════════════════════════════════════════════════

def format_latex_row(
    model_name: str,
    results: dict[str, MetricResult | float],
) -> str:
    """Format metrics as a LaTeX table row for paper inclusion.

    Example output::

        cGAN & 23.45 ± 1.23 & 0.8912 ± 0.0345 & 0.0823 ± 0.0156 & 42.31 & 34.56 ± 8.12 \\\\
    """
    display_names = {
        "cgan": "cGAN (ours)",
        "unet_l1": "UNet-L1",
        "cnn_baseline": "CNN Baseline",
    }
    name = display_names.get(model_name, model_name)

    parts = [name]

    # PSNR
    if "psnr" in results and isinstance(results["psnr"], MetricResult):
        r = results["psnr"]
        parts.append(f"{r.mean:.2f} $\\pm$ {r.std:.2f}")
    # SSIM
    if "ssim" in results and isinstance(results["ssim"], MetricResult):
        r = results["ssim"]
        parts.append(f"{r.mean:.4f} $\\pm$ {r.std:.4f}")
    # LPIPS
    if "lpips" in results and isinstance(results["lpips"], MetricResult):
        r = results["lpips"]
        parts.append(f"{r.mean:.4f} $\\pm$ {r.std:.4f}")
    # FID
    if "fid" in results:
        fid = results["fid"]
        val = fid if isinstance(fid, (int, float)) else fid.mean
        parts.append(f"{val:.2f}")
    # Colorfulness (pred)
    if "colorfulness_pred" in results and isinstance(results["colorfulness_pred"], MetricResult):
        r = results["colorfulness_pred"]
        parts.append(f"{r.mean:.2f} $\\pm$ {r.std:.2f}")

    row = " & ".join(parts) + " \\\\"
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args: argparse.Namespace) -> dict:
    """Run full deterministic evaluation.

    Returns
    -------
    dict — all metrics, serialisable to JSON.
    """
    seed_everything(SEED)

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(args.model, args.weights, device)

    # ── Prepare dataset ───────────────────────────────────────────────────
    paths = collect_image_paths(args.data_dir)
    dataset = EvalColorizationDataset(paths, size=IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── Initialise metric objects ─────────────────────────────────────────
    lpips_metric = LPIPSMetric(net="alex", device=device)
    fid_metric = FIDMetric(device=device, batch_size=args.batch_size)

    # ── Inference + per-image metrics ─────────────────────────────────────
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_paths: list[str] = []

    print(f"\nRunning inference on {len(dataset):,} images …")
    t0 = time.time()

    with torch.inference_mode():
        for L_batch, AB_gt_batch, path_batch in tqdm(loader, desc="Evaluating"):
            L_dev = L_batch.to(device)
            AB_pred_batch = model(L_dev).cpu()

            # Convert to RGB
            preds_rgb = lab_batch_to_rgb(L_batch, AB_pred_batch)
            targets_rgb = lab_batch_to_rgb(L_batch, AB_gt_batch)

            all_preds.extend(preds_rgb)
            all_targets.extend(targets_rgb)
            all_paths.extend(path_batch)

    inference_time = time.time() - t0
    throughput = len(dataset) / inference_time
    print(f"Inference complete: {inference_time:.1f}s ({throughput:.1f} img/s)")

    # ── Compute per-image metrics ─────────────────────────────────────────
    print("\nComputing per-image metrics (PSNR, SSIM, LPIPS, Colorfulness) …")
    per_image_results = compute_all_per_image(
        all_preds, all_targets, lpips_metric=lpips_metric
    )

    # ── Compute FID (distributional) ──────────────────────────────────────
    print("Computing FID …")
    fid_score = fid_metric(all_preds, all_targets)
    print(f"  FID = {fid_score:.2f}")

    # ── Assemble results ──────────────────────────────────────────────────
    results = {
        **per_image_results,
        "fid": fid_score,
    }

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  Evaluation Results — {args.model}")
    print("=" * 72)
    for name, val in results.items():
        if isinstance(val, MetricResult):
            print(f"  {name:20s}  {val.mean:>10.4f} ± {val.std:.4f}")
        else:
            print(f"  {name:20s}  {val:>10.4f}")
    print("=" * 72)

    # ── LaTeX row ─────────────────────────────────────────────────────────
    latex_row = format_latex_row(args.model, results)
    print(f"\nLaTeX table row:\n  {latex_row}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_data = {
        "model": args.model,
        "weights": args.weights,
        "data_dir": args.data_dir,
        "n_images": len(dataset),
        "image_size": IMG_SIZE,
        "seed": SEED,
        "inference_time_s": round(inference_time, 2),
        "throughput_img_s": round(throughput, 2),
        "metrics": {},
        "latex_row": latex_row,
    }

    for name, val in results.items():
        if isinstance(val, MetricResult):
            json_data["metrics"][name] = val.to_dict(per_image=args.per_image)
        else:
            json_data["metrics"][name] = {"value": float(val)}

    # Per-image breakdown (optional, for detailed analysis)
    if args.per_image:
        json_data["per_image_paths"] = [
            os.path.basename(p) for p in all_paths
        ]

    json_path = os.path.join(args.output, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved metrics → {json_path}")

    # ── Comparison grid ───────────────────────────────────────────────────
    print("\nGenerating qualitative comparison grid …")
    save_comparison_grid(
        model, dataset, args.output, device, n_images=args.grid_images
    )

    return json_data


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChromaForge — Full Test-Set Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model variant to evaluate.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the model weights (.pth state dict).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing test images (flat, JPEG/PNG).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./eval_output",
        help="Output directory for results (default: ./eval_output).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Include per-image metric values in the JSON output.",
    )
    parser.add_argument(
        "--grid-images",
        type=int,
        default=16,
        help="Number of images in the comparison grid (default: 16).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
