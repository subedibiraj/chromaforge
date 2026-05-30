# ChromaForge 🎨

> Automatic grayscale image colorization via Conditional Generative Adversarial Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Spaces-API-blue)](https://huggingface.co/spaces/your-username/chromaforge-api)
[![Vercel](https://img.shields.io/badge/Vercel-Frontend-black)](https://chromaforge.vercel.app)

ChromaForge colorizes grayscale photographs using a **U-Net generator** paired with a **70×70 PatchGAN discriminator**, trained on COCO 2017 in the CIE LAB color space. It is the research-grade evolution of a third-year undergraduate project.

---

## Results

| Metric | Grayscale baseline | L1 only | **ChromaForge (GAN)** |
|--------|-------------------:|--------:|----------------------:|
| PSNR (dB) ↑ | 21.3 | 24.1 | **26.4** |
| SSIM ↑ | 0.801 | 0.863 | **0.892** |
| Colorfulness ↑ | 0.0 | 12.4 | **31.2** |

---

## Repository layout

```
chromaforge/
├── backend/                  # FastAPI inference server
│   ├── main.py               # API routes + U-Net model definition
│   ├── discriminator.py      # PatchGAN (used during training only)
│   ├── requirements.txt
│   ├── Dockerfile            # Hugging Face Spaces deployment
│   └── weights/              # Place generator.pth here after training
│
├── frontend/                 # React web app
│   ├── src/
│   │   ├── App.js
│   │   ├── components/
│   │   │   ├── Colorizer.js          # Upload + result display
│   │   │   └── ComparisonSlider.js   # Before/after drag slider
│   │   ├── hooks/useDrop.js          # Drag-and-drop with validation
│   │   └── utils/api.js              # API client with progress tracking
│   └── .env.example
│
├── training/
│   └── train_cgan.ipynb      # Complete Google Colab training notebook
│
├── docs/
│   └── latex/
│       └── chromaforge_paper.tex     # IEEE-format technical paper
│
└── model_card/
    └── README.md             # Hugging Face model card
```

---

## Quickstart

### Run locally

**Backend** (requires Python 3.10+):
```bash
cd backend
pip install -r requirements.txt
# Place weights/generator.pth (from training)
uvicorn main:app --reload --port 7860
```

**Frontend**:
```bash
cd frontend
cp .env.example .env          # Edit REACT_APP_API_URL if needed
npm install
npm start
```

### Train from scratch

Open `training/train_cgan.ipynb` in Google Colab with a T4 GPU.  
Set runtime → T4 GPU, then run all cells. Training takes ~6–8 hours for 100 epochs.  
The best generator weights are saved to your Google Drive automatically.

---

## Deployment (free)

### Backend → Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push the `backend/` directory contents to the Space repo
4. Upload `weights/generator.pth` to the Space via the Files tab
5. Your API will be live at `https://your-username-chromaforge-api.hf.space`

### Frontend → Vercel

1. Push the repo to GitHub
2. Import the project at [vercel.com](https://vercel.com)
3. Set root directory to `frontend/`
4. Add environment variable: `REACT_APP_API_URL=https://your-hf-space-url`
5. Deploy

---

## Architecture

```
Input: grayscale image (H×W)
  ↓ convert to LAB, extract L channel
  ↓ normalize to [-1, 1]

U-Net Generator
  Encoder: L → e1(64) → e2(128) → e3(256) → e4(512) → ... → e8(512) [bottleneck]
  Decoder: d1 → cat(d1,e7) → d2 → ... → cat(d7,e1) → d8 → tanh
  Output: predicted AB channels (2, 256, 256)

PatchGAN Discriminator (training only)
  Input: cat(L, AB_real_or_fake) → 30×30 patch logits

Loss:  L_G = L_adv + 100 × L_L1
       L_D = 0.5 × (L_real + L_fake)
```

---

## Technical paper

The full IEEE-format technical report is at `docs/latex/chromaforge_paper.tex`.  
Compile with: `pdflatex chromaforge_paper.tex`

---

## References

- Isola et al. (2017) — *Image-to-Image Translation with Conditional Adversarial Networks* (pix2pix)
- Zhang et al. (2016) — *Colorful Image Colorization*
- Ronneberger et al. (2015) — *U-Net*
- Lin et al. (2014) — *Microsoft COCO*

---

## License

MIT — see [LICENSE](LICENSE).
