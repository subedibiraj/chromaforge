#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality figure generation for ChromaForge.

Reads training history from three experiments and produces:
  1. training_curves.pdf   — 2×2 subplot grid (losses, PSNR, SSIM, colorfulness)
  2. colorfulness_comparison.pdf — Box plot of final-5-epoch colorfulness
  3. metric_summary.pdf    — Grouped bar chart of final PSNR / SSIM / colorfulness

Usage:
    python generate_figures.py
    python generate_figures.py --history-dir ../training --output-dir ../docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for CI / headless

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
# Colorblind-friendly palette
COLOR_CGAN = "#2196F3"       # blue
COLOR_UNET_L1 = "#FF9800"   # orange
COLOR_CNN = "#4CAF50"        # green
COLOR_LOSS_D = "#E91E63"     # pink – discriminator loss on dual axis

LABEL_CGAN = "cGAN (ours)"
LABEL_UNET = "U-Net + L1"
LABEL_CNN = "CNN baseline"

FONT_LABEL = 11
FONT_TICK = 9
FONT_LEGEND = 9
FONT_TITLE = 12

DPI = 300


def _apply_style() -> None:
    """Apply a clean academic style to all subsequent plots."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            plt.style.use("ggplot")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_LABEL,
        "axes.labelsize": FONT_LABEL,
        "axes.titlesize": FONT_TITLE,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_history(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    """Save *fig* as both PDF and PNG under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = out_dir / f"{stem}.{ext}"
        fig.savefig(str(dest), format=ext)
        print(f"  [OK] saved {dest}")


# ---------------------------------------------------------------------------
# Figure 1 — Training curves (2×2)
# ---------------------------------------------------------------------------

def plot_training_curves(
    cgan: dict, unet: dict, cnn: dict, out_dir: Path
) -> None:
    """2×2 grid: GAN losses | PSNR | SSIM | Colorfulness."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    epochs_cgan = np.arange(1, len(cgan["loss_G"]) + 1)
    epochs_unet = np.arange(1, len(unet["psnr"]) + 1)
    epochs_cnn = np.arange(1, len(cnn["psnr"]) + 1)

    # ---- (0, 0) Generator & Discriminator loss (dual y-axis) ----
    ax0 = axes[0, 0]
    ln1 = ax0.plot(epochs_cgan, cgan["loss_G"], color=COLOR_CGAN,
                   linewidth=1.4, label="Generator loss ($\\mathcal{L}_G$)")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Generator Loss", color=COLOR_CGAN)
    ax0.tick_params(axis="y", labelcolor=COLOR_CGAN)

    ax0b = ax0.twinx()
    ln2 = ax0b.plot(epochs_cgan, cgan["loss_D"], color=COLOR_LOSS_D,
                    linewidth=1.4, linestyle="--",
                    label="Discriminator loss ($\\mathcal{L}_D$)")
    ax0b.set_ylabel("Discriminator Loss", color=COLOR_LOSS_D)
    ax0b.tick_params(axis="y", labelcolor=COLOR_LOSS_D)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax0.legend(lns, labs, loc="upper right", frameon=False)
    ax0.set_title("(a) GAN Losses")

    # ---- (0, 1) PSNR ----
    ax1 = axes[0, 1]
    ax1.plot(epochs_cgan, cgan["psnr"], color=COLOR_CGAN,
             linewidth=1.4, label=LABEL_CGAN)
    ax1.plot(epochs_unet, unet["psnr"], color=COLOR_UNET_L1,
             linewidth=1.4, label=LABEL_UNET)
    ax1.plot(epochs_cnn, cnn["psnr"], color=COLOR_CNN,
             linewidth=1.4, label=LABEL_CNN)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR (dB)")
    ax1.legend(loc="lower right", frameon=False)
    ax1.set_title("(b) PSNR")

    # ---- (1, 0) SSIM ----
    ax2 = axes[1, 0]
    ax2.plot(epochs_cgan, cgan["ssim"], color=COLOR_CGAN,
             linewidth=1.4, label=LABEL_CGAN)
    ax2.plot(epochs_unet, unet["ssim"], color=COLOR_UNET_L1,
             linewidth=1.4, label=LABEL_UNET)
    ax2.plot(epochs_cnn, cnn["ssim"], color=COLOR_CNN,
             linewidth=1.4, label=LABEL_CNN)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM")
    ax2.legend(loc="lower right", frameon=False)
    ax2.set_title("(c) SSIM")

    # ---- (1, 1) Colorfulness ----
    ax3 = axes[1, 1]
    ax3.plot(epochs_cgan, cgan["colorfulness"], color=COLOR_CGAN,
             linewidth=1.4, label=LABEL_CGAN)
    ax3.plot(epochs_unet, unet["colorfulness"], color=COLOR_UNET_L1,
             linewidth=1.4, label=LABEL_UNET)
    ax3.plot(epochs_cnn, cnn["colorfulness"], color=COLOR_CNN,
             linewidth=1.4, label=LABEL_CNN)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Colorfulness")
    ax3.legend(loc="upper right", frameon=False)
    ax3.set_title("(d) Colorfulness")

    fig.tight_layout(pad=2.0)
    save_figure(fig, "training_curves", out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Colorfulness comparison (box plot, last 5 epochs)
# ---------------------------------------------------------------------------

def plot_colorfulness_comparison(
    cgan: dict, unet: dict, cnn: dict, out_dir: Path
) -> None:
    """Box plot comparing colorfulness over the last 5 epochs."""
    data = [
        cgan["colorfulness"][-5:],
        unet["colorfulness"][-5:],
        cnn["colorfulness"][-5:],
    ]
    labels = [LABEL_CGAN, LABEL_UNET, LABEL_CNN]
    colors = [COLOR_CGAN, COLOR_UNET_L1, COLOR_CNN]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.45,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=6),
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Colorfulness (last 5 epochs)")
    ax.set_title("Colorfulness Distribution — Final Training Epochs")

    # Annotate means
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.annotate(
            f"{mean_val:.1f}",
            xy=(i + 1, mean_val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=FONT_TICK,
            color="black",
        )

    fig.tight_layout()
    save_figure(fig, "colorfulness_comparison", out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Metric summary (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_metric_summary(
    cgan: dict, unet: dict, cnn: dict, out_dir: Path
) -> None:
    """Grouped bar chart of final PSNR, SSIM (×25 for visual scaling), colorfulness."""
    models = [LABEL_CGAN, LABEL_UNET, LABEL_CNN]
    colors = [COLOR_CGAN, COLOR_UNET_L1, COLOR_CNN]

    # Use mean of last 5 epochs for robustness
    def _last5_mean(arr):
        return float(np.mean(arr[-5:]))

    psnr_vals = [_last5_mean(cgan["psnr"]),
                 _last5_mean(unet["psnr"]),
                 _last5_mean(cnn["psnr"])]
    ssim_vals = [_last5_mean(cgan["ssim"]),
                 _last5_mean(unet["ssim"]),
                 _last5_mean(cnn["ssim"])]
    color_vals = [_last5_mean(cgan["colorfulness"]),
                  _last5_mean(unet["colorfulness"]),
                  _last5_mean(cnn["colorfulness"])]

    metrics = ["PSNR (dB)", "SSIM (×25)", "Colorfulness"]
    # Scale SSIM by 25 so the bars are visually comparable
    ssim_scaled = [v * 25 for v in ssim_vals]
    raw_ssim = ssim_vals  # keep for annotations

    x = np.arange(len(metrics))
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [psnr_vals[i], ssim_scaled[i], color_vals[i]]
        bars = ax.bar(x + offsets[i], vals, width, label=model,
                      color=color, alpha=0.85, edgecolor="white",
                      linewidth=0.5)

        # Annotate each bar with its true value
        true_vals = [psnr_vals[i], raw_ssim[i], color_vals[i]]
        fmts = ["{:.1f}", "{:.4f}", "{:.1f}"]
        for bar, tv, fmt in zip(bars, true_vals, fmts):
            ax.annotate(
                fmt.format(tv),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value")
    ax.set_title("Final Metric Comparison (mean of last 5 epochs)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)  # extra headroom for annotations

    fig.tight_layout()
    save_figure(fig, "metric_summary", out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for ChromaForge."
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Root directory containing runs/, runs_unet_l1/, runs_cnn_baseline/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "docs" / "figures"),
        help="Directory where figures will be saved.",
    )
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    output_dir = Path(args.output_dir)

    print(f"History dir : {history_dir}")
    print(f"Output dir  : {output_dir}")
    print()

    # Load histories
    cgan = load_history(history_dir / "runs" / "history.json")
    unet = load_history(history_dir / "runs_unet_l1" / "history.json")
    cnn  = load_history(history_dir / "runs_cnn_baseline" / "history.json")

    print(f"cGAN history : {len(cgan['loss_G'])} epochs")
    print(f"U-Net+L1     : {len(unet['loss'])} epochs")
    print(f"CNN baseline : {len(cnn['loss'])} epochs")
    print()

    _apply_style()

    print("Generating Figure 1 -- training curves ...")
    plot_training_curves(cgan, unet, cnn, output_dir)

    print("Generating Figure 2 -- colorfulness comparison ...")
    plot_colorfulness_comparison(cgan, unet, cnn, output_dir)

    print("Generating Figure 3 -- metric summary ...")
    plot_metric_summary(cgan, unet, cnn, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
