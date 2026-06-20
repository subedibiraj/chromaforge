"""
ChromaForge — CNN Regression Baseline
Five-layer fully convolutional network, L1 loss only.
No skip connections, no adversarial component.

Establishes the lower bound on colorfulness and the reference point for
isolating the contribution of the U-Net architecture and the cGAN objective
in the main comparison study.
"""

import os
import math
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

CFG = {
    'data_dir':      r'C:\chromaforge\data\train2017',
    'output_dir':    r'C:\chromaforge\training\runs_cnn_baseline',
    'img_size':      256,
    'batch_size':    16,
    'num_workers':   4,
    'epochs':        20,
    'lr':            2e-4,
    'subset_size':   20000,   # smaller subset — sufficient for a baseline comparison point
    'val_fraction':  0.1,
    'seed':          42,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ColorizationDataset(Dataset):
    def __init__(self, paths, size=256, augment=True):
        self.paths   = paths
        self.size    = size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            return self.__getitem__((idx + 1) % len(self.paths))

        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize((self.size, self.size), Image.LANCZOS)
        lab = rgb2lab(np.array(img, dtype=np.float32) / 255.0).astype(np.float32)

        L  = torch.from_numpy((lab[:, :, 0:1] / 50.0) - 1.0).permute(2, 0, 1)
        AB = torch.from_numpy(lab[:, :, 1:3] / 128.0).permute(2, 0, 1)
        return L, AB


class CNNBaseline(nn.Module):
    """
    Plain 5-layer fully convolutional regressor.
    No skip connections, no downsampling/upsampling — operates at full
    resolution throughout via dilated-free same-padding convolutions,
    isolating the effect of architecture depth/capacity from the
    skip-connection and adversarial contributions tested separately.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def lab_tensor_to_rgb(L_tensor, ab_tensor):
    L_np  = ((L_tensor[0].cpu().numpy() + 1) * 50).clip(0, 100)
    ab_np = (ab_tensor.permute(1, 2, 0).cpu().numpy() * 128).clip(-128, 128)
    lab   = np.zeros((*L_np.shape, 3), dtype=np.float32)
    lab[:, :, 0]  = L_np
    lab[:, :, 1:] = ab_np
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return (lab2rgb(lab) * 255).clip(0, 255).astype(np.uint8)


def colorfulness(rgb):
    R, G, B = rgb[:,:,0].astype(float), rgb[:,:,1].astype(float), rgb[:,:,2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    return math.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * math.sqrt(np.mean(rg)**2 + np.mean(yb)**2)


def evaluate_batch(L_batch, AB_real, AB_fake):
    psnrs, ssims, cfs = [], [], []
    for i in range(L_batch.shape[0]):
        real = lab_tensor_to_rgb(L_batch[i], AB_real[i])
        fake = lab_tensor_to_rgb(L_batch[i], AB_fake[i])
        psnrs.append(psnr_fn(real, fake, data_range=255))
        ssims.append(ssim_fn(real, fake, channel_axis=2, data_range=255))
        cfs.append(colorfulness(fake))
    return {'psnr': np.mean(psnrs), 'ssim': np.mean(ssims), 'colorfulness': np.mean(cfs)}


def train():
    random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])
    os.makedirs(CFG['output_dir'], exist_ok=True)

    all_paths = sorted([
        os.path.join(CFG['data_dir'], f)
        for f in os.listdir(CFG['data_dir'])
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    random.shuffle(all_paths)
    all_paths = all_paths[:CFG['subset_size']]
    split = int((1 - CFG['val_fraction']) * len(all_paths))

    train_loader = DataLoader(
        ColorizationDataset(all_paths[:split], CFG['img_size'], augment=True),
        batch_size=CFG['batch_size'], shuffle=True,
        num_workers=CFG['num_workers'], pin_memory=True, drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        ColorizationDataset(all_paths[split:], CFG['img_size'], augment=False),
        batch_size=8, shuffle=False, num_workers=2, persistent_workers=True
    )

    print(f'Device : {DEVICE}')
    print(f'Train  : {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}')

    model = CNNBaseline().to(DEVICE)
    print(f'Params : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
    scaler    = GradScaler('cuda')

    history = {'loss': [], 'psnr': [], 'ssim': [], 'colorfulness': []}
    best_psnr = 0.0

    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{CFG["epochs"]}', leave=False)

        for L, AB_real in pbar:
            L, AB_real = L.to(DEVICE), AB_real.to(DEVICE)
            optimizer.zero_grad()
            with autocast('cuda'):
                AB_fake = model(L)
                loss = criterion(AB_fake, AB_real)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        batch_metrics = []
        with torch.no_grad():
            for L, AB_real in val_loader:
                AB_fake = model(L.to(DEVICE)).cpu()
                torch.cuda.empty_cache()
                batch_metrics.append(evaluate_batch(L, AB_real, AB_fake))
                if len(batch_metrics) >= 10:
                    break

        n = len(train_loader)
        avg = {k: float(np.mean([m[k] for m in batch_metrics])) for k in batch_metrics[0]}
        history['loss'].append(epoch_loss / n)
        history['psnr'].append(avg['psnr'])
        history['ssim'].append(avg['ssim'])
        history['colorfulness'].append(avg['colorfulness'])

        print(f"[{epoch:3d}] loss={history['loss'][-1]:.4f}  "
              f"PSNR={avg['psnr']:.2f}  SSIM={avg['ssim']:.4f}  CF={avg['colorfulness']:.2f}")

        if avg['psnr'] > best_psnr:
            best_psnr = avg['psnr']
            torch.save(model.state_dict(), os.path.join(CFG['output_dir'], 'cnn_baseline_best.pth'))

        with open(os.path.join(CFG['output_dir'], 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print(f'\nDone. Best PSNR: {best_psnr:.2f} dB')
    print(f"Final  — PSNR: {history['psnr'][-1]:.2f}  SSIM: {history['ssim'][-1]:.4f}  CF: {history['colorfulness'][-1]:.2f}")


if __name__ == '__main__':
    train()