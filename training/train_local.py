"""
ChromaForge — Local Training Script
RTX 3060 optimized: batch_size=8, mixed precision, gradient checkpointing
"""

import os
import math
import random
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    'data_dir':      r'C:\chromaforge\data\train2017',
    'output_dir':    r'C:\chromaforge\training\runs',
    'img_size':      256,
    'batch_size':    8,
    'num_workers':   4,
    'epochs':        100,
    'lr_g':          2e-4,
    'lr_d':          2e-4,
    'beta1':         0.5,
    'beta2':         0.999,
    'lambda_l1':     100.0,
    'decay_epoch':   50,
    'save_every':    10,
    'sample_every':  5,
    'n_val_samples': 8,
    'seed':          42,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset ───────────────────────────────────────────────────────────────────

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

        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = transforms.functional.adjust_brightness(img, random.uniform(0.8, 1.2))

        img = img.resize((self.size, self.size), Image.LANCZOS)
        lab = rgb2lab(np.array(img, dtype=np.float32) / 255.0).astype(np.float32)

        L  = torch.from_numpy((lab[:, :, 0:1] / 50.0) - 1.0).permute(2, 0, 1)
        AB = torch.from_numpy(lab[:, :, 1:3] / 128.0).permute(2, 0, 1)
        return L, AB


# ── Architecture ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = ConvBlock(64,  128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 512)
        self.enc6 = ConvBlock(512, 512)
        self.enc7 = ConvBlock(512, 512)
        self.enc8 = ConvBlock(512, 512, use_bn=False)

        self.dec1 = DeconvBlock(512,  512, dropout=True)
        self.dec2 = DeconvBlock(1024, 512, dropout=True)
        self.dec3 = DeconvBlock(1024, 512, dropout=True)
        self.dec4 = DeconvBlock(1024, 512)
        self.dec5 = DeconvBlock(1024, 256)
        self.dec6 = DeconvBlock(512,  128)
        self.dec7 = DeconvBlock(256,  64)
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(128, 2, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        return self.dec8(torch.cat([d7, e1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride=2, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3,   64,  norm=False),
            *block(64,  128),
            *block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, L, ab):
        return self.model(torch.cat([L, ab], dim=1))


class GANLoss(nn.Module):
    def __init__(self, real_label=0.9, fake_label=0.0):
        super().__init__()
        self.real = real_label
        self.fake = fake_label
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred, is_real):
        target = torch.full_like(pred, self.real if is_real else self.fake)
        return self.bce(pred, target)


def weights_init(m):
    cls = m.__class__.__name__
    if 'Conv' in cls and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in cls and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ── Metrics ───────────────────────────────────────────────────────────────────

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


# ── Visualization ─────────────────────────────────────────────────────────────

def save_sample_grid(G, val_loader, epoch, out_dir):
    G.eval()
    L, AB_real = next(iter(val_loader))
    L, AB_real = L.to(DEVICE), AB_real.to(DEVICE)
    with torch.no_grad():
        AB_fake = G(L)

    n = min(4, L.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        L_np = ((L[i, 0].cpu().numpy() + 1) * 50).clip(0, 100)
        axes[i, 0].imshow(L_np, cmap='gray');                    axes[i, 0].set_title('Input')
        axes[i, 1].imshow(lab_tensor_to_rgb(L[i], AB_real[i])); axes[i, 1].set_title('Ground truth')
        axes[i, 2].imshow(lab_tensor_to_rgb(L[i], AB_fake[i])); axes[i, 2].set_title('Predicted')
        for ax in axes[i]:
            ax.axis('off')

    plt.suptitle(f'Epoch {epoch}', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'samples', f'epoch_{epoch:04d}.png'), dpi=100, bbox_inches='tight')
    plt.close()
    G.train()


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])

    out_dir = CFG['output_dir']
    for sub in ['checkpoints', 'samples']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # Dataset
    all_paths = sorted([
        os.path.join(CFG['data_dir'], f)
        for f in os.listdir(CFG['data_dir'])
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    random.shuffle(all_paths)
    split = int(0.9 * len(all_paths))

    train_loader = DataLoader(
        ColorizationDataset(all_paths[:split], CFG['img_size'], augment=True),
        batch_size=CFG['batch_size'], shuffle=True,
        num_workers=CFG['num_workers'], pin_memory=True, drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        ColorizationDataset(all_paths[split:], CFG['img_size'], augment=False),
        batch_size=CFG['n_val_samples'], shuffle=False,
        num_workers=2, persistent_workers=True
    )

    print(f'Device : {DEVICE}')
    print(f'Train  : {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}')

    # Models
    G = UNetGenerator().to(DEVICE).apply(weights_init)
    D = PatchDiscriminator().to(DEVICE).apply(weights_init)

    criterion_gan = GANLoss().to(DEVICE)
    criterion_l1  = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=CFG['lr_g'], betas=(CFG['beta1'], CFG['beta2']))
    opt_D = optim.Adam(D.parameters(), lr=CFG['lr_d'], betas=(CFG['beta1'], CFG['beta2']))

    def lr_lambda(epoch):
        if epoch < CFG['decay_epoch']:
            return 1.0
        return max(0.0, 1.0 - (epoch - CFG['decay_epoch']) / (CFG['epochs'] - CFG['decay_epoch']))

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # Mixed precision
    scaler_G = GradScaler()
    scaler_D = GradScaler()

    # Resume from checkpoint if exists
    start_epoch = 1
    best_psnr   = 0.0
    history     = {'loss_G': [], 'loss_D': [], 'psnr': [], 'ssim': [], 'colorfulness': []}

    latest = sorted(Path(out_dir, 'checkpoints').glob('G_ep*.pth'))
    if latest:
        ckpt_path = str(latest[-1])
        start_epoch = int(latest[-1].stem.split('ep')[1]) + 1
        G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        D.load_state_dict(torch.load(ckpt_path.replace('G_ep', 'D_ep'), map_location=DEVICE))
        hist_path = os.path.join(out_dir, 'history.json')
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
            best_psnr = max(history['psnr']) if history['psnr'] else 0.0
        print(f'Resumed from epoch {start_epoch - 1}')

    print(f'Starting from epoch {start_epoch}')

    for epoch in range(start_epoch, CFG['epochs'] + 1):
        G.train(); D.train()
        loss_G_epoch = loss_D_epoch = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{CFG["epochs"]}', leave=False)

        for L, AB_real in pbar:
            L, AB_real = L.to(DEVICE), AB_real.to(DEVICE)

            # Discriminator
            opt_D.zero_grad()
            with autocast():
                AB_fake = G(L).detach()
                loss_D  = 0.5 * (criterion_gan(D(L, AB_real), True) + criterion_gan(D(L, AB_fake), False))
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # Generator
            opt_G.zero_grad()
            with autocast():
                AB_fake = G(L)
                loss_G  = criterion_gan(D(L, AB_fake), True) + criterion_l1(AB_fake, AB_real) * CFG['lambda_l1']
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            loss_G_epoch += loss_G.item()
            loss_D_epoch += loss_D.item()
            pbar.set_postfix({'G': f'{loss_G.item():.3f}', 'D': f'{loss_D.item():.3f}'})

        sched_G.step(); sched_D.step()

        # Validation
        G.eval()
        batch_metrics = []
        with torch.no_grad():
            for L, AB_real in val_loader:
                AB_fake = G(L.to(DEVICE))
                batch_metrics.append(evaluate_batch(L, AB_real, AB_fake.cpu()))
                if len(batch_metrics) >= 10:
                    break

        n = len(train_loader)
        avg = {k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0]}
        history['loss_G'].append(loss_G_epoch / n)
        history['loss_D'].append(loss_D_epoch / n)
        history['psnr'].append(float(avg['psnr']))
        history['ssim'].append(float(avg['ssim']))
        history['colorfulness'].append(float(avg['colorfulness']))

        print(f"[{epoch:3d}] loss_G={history['loss_G'][-1]:.4f}  loss_D={history['loss_D'][-1]:.4f}  "
              f"PSNR={avg['psnr']:.2f}  SSIM={avg['ssim']:.4f}  CF={avg['colorfulness']:.2f}")

        if epoch % CFG['sample_every'] == 0:
            save_sample_grid(G, val_loader, epoch, out_dir)

        if epoch % CFG['save_every'] == 0:
            torch.save(G.state_dict(), os.path.join(out_dir, 'checkpoints', f'G_ep{epoch:04d}.pth'))
            torch.save(D.state_dict(), os.path.join(out_dir, 'checkpoints', f'D_ep{epoch:04d}.pth'))

        if avg['psnr'] > best_psnr:
            best_psnr = avg['psnr']
            torch.save(G.state_dict(), os.path.join(out_dir, 'generator_best.pth'))

        with open(os.path.join(out_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print(f'\nTraining complete. Best PSNR: {best_psnr:.2f} dB')
    print(f'Best weights: {out_dir}\\generator_best.pth')


if __name__ == '__main__':
    train()