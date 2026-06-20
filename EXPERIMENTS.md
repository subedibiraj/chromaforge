# Experiment Log

Systematic record of all training runs conducted for the ChromaForge comparative study. Each entry documents the configuration, hardware, runtime, and final validation metrics.

---

## Run 1 — cGAN (U-Net Generator + PatchGAN Discriminator)

| Field | Value |
|-------|-------|
| **Date** | 2025 (local training) |
| **Config** | `configs/cgan.yaml` |
| **Script** | `training/train_local.py` |
| **Hardware** | NVIDIA RTX 3060 (12 GB VRAM) |
| **Dataset** | COCO 2017 train (118,287 images, 90/10 split) |
| **Epochs** | 100 |
| **Batch size** | 8 |
| **Mixed precision** | Yes (torch.amp) |
| **LR** | 2e-4 (constant 50 epochs, then linear decay to 0) |
| **λ_L1** | 100 |
| **Label smoothing** | real=0.9, fake=0.0 |
| **Augmentation** | H-flip (p=0.5), brightness jitter ∈ [0.8, 1.2] |
| **Seed** | 42 |

### Results (validation, best epoch)

| Metric | Best | Final (ep. 100) | Mean (last 10) |
|--------|------|-----------------|-----------------|
| PSNR (dB) | **23.67** (ep. 68) | 23.15 | 23.02 |
| SSIM | **0.905** (ep. 21) | 0.893 | 0.892 |
| Colorfulness | 38.47 (ep. 2) | 30.84 | 31.18 |
| Loss G | 9.07 (ep. 76) | 9.29 | 9.30 |
| Loss D | 0.287 (ep. 100) | 0.287 | 0.289 |

**Checkpoint**: `training/runs/generator_best.pth` (208 MB)

### Observations
- Generator loss stabilizes around epoch 60; discriminator continues slowly improving
- PSNR plateaus around 22.5–23.7 dB with significant epoch-to-epoch variance (~±0.5 dB)
- Colorfulness oscillates between 27–38 but averages around 31–32, well within natural photo range
- Training was stable — no mode collapse observed
- Total training time: ~18 hours on RTX 3060

---

## Run 2 — U-Net + L1 Loss Only (no adversarial component)

| Field | Value |
|-------|-------|
| **Date** | 2025 (local training) |
| **Config** | `configs/unet_l1.yaml` |
| **Script** | `training/train_unet_l1.py` |
| **Hardware** | NVIDIA RTX 3060 (12 GB VRAM) |
| **Dataset** | COCO 2017 train (20,000-image subset, 90/10 split) |
| **Epochs** | 20 |
| **Batch size** | 8 |
| **Mixed precision** | Yes |
| **LR** | 2e-4 (constant) |
| **Augmentation** | H-flip (p=0.5) |
| **Seed** | 42 |

### Results (validation, best epoch)

| Metric | Best | Final (ep. 20) |
|--------|------|----------------|
| PSNR (dB) | **23.74** (ep. 10) | 23.22 |
| SSIM | **0.914** (ep. 13) | 0.913 |
| Colorfulness | 25.74 (ep. 16) | 16.46 |
| Loss | 0.0647 (ep. 20) | 0.0647 |

**Checkpoint**: `training/runs_unet_l1/unet_l1_best.pth` (208 MB)

### Observations
- Loss decreases steadily throughout all 20 epochs — model likely benefits from more training
- PSNR improves from 22.79 to 23.74 in 10 epochs, then plateaus with slight variance
- Colorfulness is dramatically lower than cGAN (mean ~16 vs ~31), confirming L1 desaturation
- **Limitation**: Trained on only 20k images for 20 epochs (vs. cGAN on 118k for 100 epochs)

---

## Run 3 — CNN Regression Baseline

| Field | Value |
|-------|-------|
| **Date** | 2025 (local training) |
| **Config** | `configs/cnn_baseline.yaml` |
| **Script** | `training/train_cnn_baseline.py` |
| **Hardware** | NVIDIA RTX 3060 (12 GB VRAM) |
| **Dataset** | COCO 2017 train (20,000-image subset, 90/10 split) |
| **Epochs** | 20 |
| **Batch size** | 16 |
| **Mixed precision** | Yes |
| **LR** | 2e-4 (constant) |
| **Augmentation** | H-flip (p=0.5) |
| **Seed** | 42 |

### Results (validation, best epoch)

| Metric | Best | Final (ep. 20) |
|--------|------|----------------|
| PSNR (dB) | **23.24** (ep. 20) | 23.24 |
| SSIM | **0.909** (ep. 20) | 0.909 |
| Colorfulness | 17.18 (ep. 7) | 12.81 |
| Loss | 0.0714 (ep. 20) | 0.0714 |

**Checkpoint**: `training/runs_cnn_baseline/cnn_baseline_best.pth` (1.2 MB)

### Observations
- Converges faster than U-Net (smaller model, 0.17M vs 54M params)
- PSNR competitive with U-Net despite no skip connections — highlights that skip connections help more for colorfulness than PSNR
- Lowest colorfulness of all three models (mean ~13.7), confirming the "regression to the mean" failure mode
- SSIM slightly lower than U-Net+L1 (0.909 vs 0.914), confirming structural benefit of skip connections
- **Limitation**: Same subset/epoch asymmetry as U-Net+L1

---

## Summary of Key Findings

| Model | Params | Train Images | Epochs | Best PSNR ↑ | Best SSIM ↑ | Mean CF ↑ |
|-------|--------|-------------|--------|-------------|-------------|-----------|
| CNN Baseline | 0.17M | 20,000 | 20 | 23.24 | 0.909 | 13.7 |
| U-Net + L1 | 54.4M | 20,000 | 20 | 23.74 | 0.914 | 16.1 |
| **cGAN (ours)** | 54.4M + 2.8M | 118,287 | 100 | 23.67 | 0.891 | **31.8** |

### Key takeaways
1. **Architecture matters for structure, not color**: U-Net skip connections improve PSNR by ~0.5 dB over CNN baseline, but only modestly increase colorfulness
2. **Loss function determines colorfulness**: The adversarial loss is the dominant factor in producing vivid, realistic colors (CF: 31.8 vs 16.1 for the same architecture)
3. **PSNR rewards desaturation**: The regression baselines achieve competitive/higher PSNR precisely because predicting grey is the "safe" strategy that minimizes MSE
4. **Training compute asymmetry**: The baselines were trained with ~17× fewer image-epochs than the cGAN. Retraining with matched compute is a planned follow-up

---

## Planned Experiments

- [ ] **Fair comparison**: Retrain CNN baseline and U-Net+L1 on full 118k dataset for 100 epochs
- [ ] **FID evaluation**: Compute FID on COCO 2017 val set (5,000 images) for all three models
- [ ] **LPIPS evaluation**: Compute LPIPS (AlexNet) on the same test set
- [ ] **Ablation on λ_L1**: Test λ ∈ {10, 50, 100, 200} to quantify the PSNR/colorfulness trade-off
- [ ] **Higher resolution**: Test 512×512 training with progressive resizing
