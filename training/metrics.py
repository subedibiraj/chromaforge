"""
ChromaForge — Shared Metrics Module
====================================

Research-grade image quality metrics for evaluating grayscale → color models.
Each metric operates on RGB uint8 images (H, W, 3) in [0, 255] unless
otherwise stated.

Metrics implemented
-------------------
- **PSNR**  (Peak Signal-to-Noise Ratio)           — skimage reference impl
- **SSIM**  (Structural Similarity Index)           — skimage reference impl
- **LPIPS** (Learned Perceptual Image Patch Sim.)   — AlexNet backbone (Zhang et al., 2018)
- **FID**   (Fréchet Inception Distance)            — InceptionV3 pool3 features
- **Colorfulness** (Hasler & Süsstrunk, 2003)       — hand-coded formula

All per-image metrics return ``MetricResult(mean, std, values)`` when given
a batch of image pairs, making it easy to report μ ± σ in a paper.

References
----------
[1] Wang et al., "Image Quality Assessment: From Error Visibility to
    Structural Similarity," IEEE TIP, 2004.
[2] Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
    Perceptual Metric," CVPR, 2018.
[3] Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge
    to a Local Nash Equilibrium," NeurIPS, 2017.
[4] Hasler & Süsstrunk, "Measuring Colourfulness in Natural Images,"
    IS&T/SPIE Electronic Imaging, 2003.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio as _psnr_fn
from skimage.metrics import structural_similarity as _ssim_fn


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricResult:
    """Container for a metric computed over a batch of images.

    Attributes
    ----------
    mean : float
        Mean value across the batch.
    std : float
        Standard deviation across the batch.
    values : list[float]
        Per-image metric values (empty when only a summary is stored).
    """
    mean: float
    std: float
    values: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"{self.mean:.4f} ± {self.std:.4f}  (n={len(self.values)})"

    def to_dict(self, per_image: bool = False) -> dict:
        """Serialise to a JSON-friendly dict."""
        d = {"mean": float(self.mean), "std": float(self.std)}
        if per_image:
            d["per_image"] = [float(v) for v in self.values]
        return d


# ══════════════════════════════════════════════════════════════════════════════
# LAB ↔ RGB helpers  (match the normalisation used in all training scripts)
# ══════════════════════════════════════════════════════════════════════════════

def lab_tensors_to_rgb(
    L: torch.Tensor,
    AB: torch.Tensor,
) -> np.ndarray:
    """Convert a single (C, H, W) LAB tensor pair to an (H, W, 3) uint8 RGB array.

    The normalisation convention matches the ChromaForge training pipeline:
        L  ∈ [-1, 1]  →  L_lab  = (L + 1) × 50   ∈ [0, 100]
        AB ∈ [-1, 1]  →  AB_lab = AB × 128         ∈ [-128, 128]

    Parameters
    ----------
    L : Tensor of shape (1, H, W)
        Normalised lightness channel.
    AB : Tensor of shape (2, H, W)
        Normalised a* and b* chrominance channels.

    Returns
    -------
    rgb : ndarray, uint8, shape (H, W, 3)
    """
    if L.dim() != 3 or L.shape[0] != 1:
        raise ValueError(f"L must be (1, H, W), got {tuple(L.shape)}")
    if AB.dim() != 3 or AB.shape[0] != 2:
        raise ValueError(f"AB must be (2, H, W), got {tuple(AB.shape)}")

    L_np = ((L[0].cpu().numpy() + 1.0) * 50.0).clip(0, 100)
    AB_np = (AB.permute(1, 2, 0).cpu().numpy() * 128.0).clip(-128, 128)

    lab = np.zeros((*L_np.shape, 3), dtype=np.float32)
    lab[:, :, 0] = L_np
    lab[:, :, 1:] = AB_np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb = (lab2rgb(lab) * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def lab_batch_to_rgb(
    L_batch: torch.Tensor,
    AB_batch: torch.Tensor,
) -> List[np.ndarray]:
    """Convert a batch of LAB tensors to a list of RGB uint8 arrays.

    Parameters
    ----------
    L_batch : Tensor (B, 1, H, W)
    AB_batch : Tensor (B, 2, H, W)

    Returns
    -------
    list of ndarray, each (H, W, 3) uint8
    """
    if L_batch.dim() != 4 or AB_batch.dim() != 4:
        raise ValueError("Inputs must be 4-D (B, C, H, W)")
    if L_batch.shape[0] != AB_batch.shape[0]:
        raise ValueError("Batch sizes must match")
    return [
        lab_tensors_to_rgb(L_batch[i], AB_batch[i])
        for i in range(L_batch.shape[0])
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Input validation helper
# ══════════════════════════════════════════════════════════════════════════════

def _validate_rgb_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    name_a: str = "img_a",
    name_b: str = "img_b",
) -> None:
    """Assert that two images are valid uint8 RGB arrays of equal shape."""
    for name, img in [(name_a, img_a), (name_b, img_b)]:
        if not isinstance(img, np.ndarray):
            raise TypeError(f"{name} must be an ndarray, got {type(img)}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"{name} must be (H, W, 3), got {img.shape}")
        if img.dtype != np.uint8:
            raise ValueError(f"{name}.dtype must be uint8, got {img.dtype}")
    if img_a.shape != img_b.shape:
        raise ValueError(
            f"Shape mismatch: {name_a} {img_a.shape} vs {name_b} {img_b.shape}"
        )


def _validate_single_rgb(img: np.ndarray, name: str = "img") -> None:
    """Assert that an image is a valid uint8 RGB array."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be an ndarray, got {type(img)}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must be (H, W, 3), got {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"{name}.dtype must be uint8, got {img.dtype}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PSNR — Peak Signal-to-Noise Ratio
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR between two uint8 RGB images.

    PSNR = 10 · log₁₀(MAX² / MSE)

    where MAX = 255 for uint8 images and MSE is the mean squared error
    computed over all pixels and channels.

    Parameters
    ----------
    pred : ndarray, uint8, (H, W, 3) — predicted (colorised) image.
    target : ndarray, uint8, (H, W, 3) — ground-truth image.

    Returns
    -------
    float — PSNR in dB.  Higher is better.
    """
    _validate_rgb_pair(pred, target, "pred", "target")
    return float(_psnr_fn(target, pred, data_range=255))


def batch_psnr(
    preds: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> MetricResult:
    """Compute PSNR for a batch of image pairs.

    Returns MetricResult with mean, std, and per-image values.
    """
    if len(preds) != len(targets):
        raise ValueError("preds and targets must have the same length")
    vals = [compute_psnr(p, t) for p, t in zip(preds, targets)]
    return MetricResult(
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        values=vals,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. SSIM — Structural Similarity Index
# ══════════════════════════════════════════════════════════════════════════════

def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute SSIM between two uint8 RGB images.

    Uses the Wang et al. (2004) formulation with default 7×7 Gaussian
    window, applied per-channel and averaged (``channel_axis=2``).

    Parameters
    ----------
    pred : ndarray, uint8, (H, W, 3)
    target : ndarray, uint8, (H, W, 3)

    Returns
    -------
    float — SSIM ∈ [-1, 1].  Higher is better.
    """
    _validate_rgb_pair(pred, target, "pred", "target")
    return float(
        _ssim_fn(target, pred, data_range=255, channel_axis=2)
    )


def batch_ssim(
    preds: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> MetricResult:
    """Compute SSIM for a batch of image pairs."""
    if len(preds) != len(targets):
        raise ValueError("preds and targets must have the same length")
    vals = [compute_ssim(p, t) for p, t in zip(preds, targets)]
    return MetricResult(
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        values=vals,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. LPIPS — Learned Perceptual Image Patch Similarity
# ══════════════════════════════════════════════════════════════════════════════

class LPIPSMetric:
    """Wrapper around the ``lpips`` package (Zhang et al., 2018).

    Uses AlexNet backbone by default (fastest, most commonly reported).
    The model is loaded lazily on first call to avoid import-time CUDA
    allocation.

    Parameters
    ----------
    net : str
        Backbone network: ``'alex'`` (default), ``'vgg'``, or ``'squeeze'``.
    device : str | torch.device
        Device to place the perceptual network on.
    """

    def __init__(
        self,
        net: str = "alex",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.net_name = net
        self.device = torch.device(device)
        self._model: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import lpips  # type: ignore[import-untyped]
        self._model = lpips.LPIPS(net=self.net_name, verbose=False).to(self.device)
        self._model.eval()

    @staticmethod
    def _rgb_to_tensor(img: np.ndarray) -> torch.Tensor:
        """Convert uint8 (H, W, 3) RGB → float32 (1, 3, H, W) in [-1, 1]."""
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        return t / 127.5 - 1.0  # [0, 255] → [-1, 1]

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute LPIPS distance between two uint8 RGB images.

        Lower is better (0 = identical perceptually).

        Parameters
        ----------
        pred : ndarray, uint8, (H, W, 3)
        target : ndarray, uint8, (H, W, 3)

        Returns
        -------
        float — LPIPS distance ∈ [0, ~1].
        """
        _validate_rgb_pair(pred, target, "pred", "target")
        self._ensure_loaded()
        with torch.inference_mode():
            t_pred = self._rgb_to_tensor(pred).to(self.device)
            t_target = self._rgb_to_tensor(target).to(self.device)
            return float(self._model(t_pred, t_target).item())

    def batch(
        self,
        preds: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
    ) -> MetricResult:
        """Compute LPIPS for a batch of image pairs.

        Images are processed individually (no batch stacking) to
        avoid OOM on large test sets.
        """
        if len(preds) != len(targets):
            raise ValueError("preds and targets must have the same length")
        vals = [self(p, t) for p, t in zip(preds, targets)]
        return MetricResult(
            mean=float(np.mean(vals)),
            std=float(np.std(vals)),
            values=vals,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. FID — Fréchet Inception Distance
# ══════════════════════════════════════════════════════════════════════════════

class FIDMetric:
    """Compute FID between two sets of images using InceptionV3 pool-3 features.

    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)

    The Inception model is loaded lazily.  Feature extraction is batched
    for efficiency.

    Parameters
    ----------
    device : str | torch.device
        Compute device.
    batch_size : int
        Internal batch size for feature extraction (default 64).

    References
    ----------
    Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge
    to a Local Nash Equilibrium," NeurIPS 2017.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._model: Optional[torch.nn.Module] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # We want pool-3 (2048-D) features — remove final FC layer.
        model.fc = torch.nn.Identity()
        model.eval()
        self._model = model.to(self.device)

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Resize to 299×299, normalise to ImageNet stats.

        Parameters
        ----------
        img : ndarray, uint8, (H, W, 3)

        Returns
        -------
        Tensor (3, 299, 299) float32
        """
        from torchvision import transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((299, 299), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        return transform(img)

    def _extract_features(
        self,
        images: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Extract 2048-D InceptionV3 features for a list of images.

        Returns
        -------
        ndarray, shape (N, 2048)
        """
        self._ensure_loaded()
        feats: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch_imgs = images[start : start + self.batch_size]
            batch_tensor = torch.stack(
                [self._preprocess(img) for img in batch_imgs]
            ).to(self.device)
            with torch.inference_mode():
                f = self._model(batch_tensor)
                if isinstance(f, tuple):  # InceptionOutputs named-tuple
                    f = f[0]
                feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    @staticmethod
    def _frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        """Compute the Fréchet distance between two multivariate Gaussians.

        FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)
        """
        from scipy.linalg import sqrtm

        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
        # Numerical stability: discard imaginary parts from sqrtm
        if np.iscomplexobj(covmean):
            if not np.allclose(np.imag(covmean), 0, atol=1e-3):
                warnings.warn(
                    "Imaginary component in sqrtm result — FID may be unreliable."
                )
            covmean = np.real(covmean)
        return float(
            diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        )

    def __call__(
        self,
        preds: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
    ) -> float:
        """Compute FID between predicted and ground-truth image sets.

        Both ``preds`` and ``targets`` must have ≥ 2 images (needed to
        estimate covariance).

        Parameters
        ----------
        preds : sequence of ndarray, uint8, (H, W, 3)
        targets : sequence of ndarray, uint8, (H, W, 3)

        Returns
        -------
        float — FID score.  Lower is better.
        """
        if len(preds) < 2 or len(targets) < 2:
            raise ValueError("Need ≥ 2 images in each set to compute FID")

        for img in preds:
            _validate_single_rgb(img, "pred")
        for img in targets:
            _validate_single_rgb(img, "target")

        feats_pred = self._extract_features(preds)
        feats_target = self._extract_features(targets)

        mu1, sigma1 = feats_pred.mean(axis=0), np.cov(feats_pred, rowvar=False)
        mu2, sigma2 = feats_target.mean(axis=0), np.cov(feats_target, rowvar=False)

        return self._frechet_distance(mu1, sigma1, mu2, sigma2)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Colorfulness — Hasler & Süsstrunk (2003)
# ══════════════════════════════════════════════════════════════════════════════

def compute_colorfulness(img: np.ndarray) -> float:
    """Hasler & Süsstrunk colorfulness metric.

    Operates in opponent-colour space:

        rg = R − G
        yb = ½(R + G) − B
        C  = √(σ_rg² + σ_yb²) + 0.3 · √(μ_rg² + μ_yb²)

    The metric is *not* a distortion measure (no reference image needed).
    It quantifies perceived colourfulness of a single image.

    Parameters
    ----------
    img : ndarray, uint8, (H, W, 3)

    Returns
    -------
    float — Colorfulness score (unitless).  Higher = more colourful.

    Reference
    ---------
    Hasler & Süsstrunk, "Measuring Colourfulness in Natural Images,"
    Proc. IS&T/SPIE Electronic Imaging 5007, 2003, pp. 87–95.
    """
    _validate_single_rgb(img, "img")
    R = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    B = img[:, :, 2].astype(np.float64)

    rg = R - G
    yb = 0.5 * (R + G) - B

    sigma_rgyb = math.sqrt(float(np.std(rg)) ** 2 + float(np.std(yb)) ** 2)
    mean_rgyb = math.sqrt(float(np.mean(rg)) ** 2 + float(np.mean(yb)) ** 2)

    return sigma_rgyb + 0.3 * mean_rgyb


def batch_colorfulness(
    images: Sequence[np.ndarray],
) -> MetricResult:
    """Compute colorfulness for a batch of images.

    This is a *no-reference* metric — only the generated images are needed.
    """
    vals = [compute_colorfulness(img) for img in images]
    return MetricResult(
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        values=vals,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: compute all per-image metrics at once
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_per_image(
    preds: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    lpips_metric: Optional[LPIPSMetric] = None,
) -> dict[str, MetricResult]:
    """Compute PSNR, SSIM, LPIPS, and Colorfulness for a batch of images.

    FID is *not* included here because it is a distributional metric
    (computed over the whole set, not per-image).  Use ``FIDMetric``
    separately.

    Parameters
    ----------
    preds : sequence of uint8 RGB ndarrays
    targets : sequence of uint8 RGB ndarrays
    lpips_metric : optional pre-initialised LPIPSMetric instance
        If ``None``, LPIPS is skipped (avoids loading the network when
        only traditional metrics are needed).

    Returns
    -------
    dict mapping metric name → MetricResult
    """
    results: dict[str, MetricResult] = {
        "psnr": batch_psnr(preds, targets),
        "ssim": batch_ssim(preds, targets),
        "colorfulness_pred": batch_colorfulness(preds),
        "colorfulness_gt": batch_colorfulness(targets),
    }
    if lpips_metric is not None:
        results["lpips"] = lpips_metric.batch(preds, targets)
    return results
