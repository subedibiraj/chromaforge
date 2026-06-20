import React from 'react';
import { BrowserRouter, Routes, Route, Link, NavLink } from 'react-router-dom';
import Colorizer from './components/Colorizer';

const GITHUB_URL = 'https://github.com/subedibiraj/chromaforge';
const PAPER_URL  = 'https://github.com/subedibiraj/chromaforge/blob/master/docs/chromaforge.pdf';

function Navbar() {
  const link = ({ isActive }) => ({
    color: isActive ? '#6366f1' : '#374151',
    textDecoration: 'none',
    fontWeight: isActive ? 600 : 400,
    fontSize: 15,
  });

  return (
    <nav style={{
      position: 'sticky', top: 0, zIndex: 50,
      background: 'rgba(255,255,255,0.9)',
      backdropFilter: 'blur(8px)',
      borderBottom: '1px solid #e5e7eb',
      padding: '0 2rem',
      display: 'flex', alignItems: 'center', height: 56,
      gap: 32,
    }}>
      <Link to="/" style={{ textDecoration: 'none', fontWeight: 700, fontSize: 18, color: '#111' }}>
        <span style={{ color: '#6366f1' }}>Chroma</span>Forge
      </Link>
      <NavLink to="/"      style={link} end>Colorize</NavLink>
      <NavLink to="/about" style={link}>About</NavLink>
      <NavLink to="/paper" style={link}>Report</NavLink>
      <a href={GITHUB_URL}
         target="_blank" rel="noreferrer"
         style={{ marginLeft: 'auto', color: '#6b7280', fontSize: 14 }}>
        GitHub ↗
      </a>
    </nav>
  );
}

function Hero() {
  return (
    <div style={{ textAlign: 'center', padding: '3rem 1rem 1rem' }}>
      <h1 style={{ fontSize: 'clamp(1.8rem, 5vw, 3rem)', fontWeight: 700, margin: 0, lineHeight: 1.2 }}>
        Grayscale to Color
        <br />
        <span style={{ color: '#6366f1' }}>with Conditional GANs</span>
      </h1>
      <p style={{ color: '#6b7280', maxWidth: 560, margin: '1rem auto 0', fontSize: 16, lineHeight: 1.6 }}>
        A U-Net generator and PatchGAN discriminator trained on COCO 2017,
        compared against a CNN regression baseline and an L1-only U-Net
        to isolate what each component actually contributes.
      </p>
    </div>
  );
}

function About() {
  const card = (title, body) => (
    <div style={{ background: '#f9fafb', borderRadius: 12, padding: '1.5rem', marginBottom: 16 }}>
      <h3 style={{ margin: '0 0 8px', fontSize: 16, fontWeight: 600 }}>{title}</h3>
      <p style={{ margin: 0, color: '#374151', lineHeight: 1.6, fontSize: 15 }}>{body}</p>
    </div>
  );

  return (
    <div style={{ maxWidth: 680, margin: '2rem auto', padding: '0 1rem' }}>
      <h2 style={{ fontWeight: 700, fontSize: 24, marginBottom: 8 }}>About this project</h2>
      <p style={{ color: '#6b7280', fontSize: 15, marginBottom: 24, lineHeight: 1.6 }}>
        ChromaForge began as an undergraduate project and was independently
        rebuilt from scratch as a controlled comparison study: a CNN
        regression baseline, a U-Net trained with L1 loss only, and a
        U-Net + PatchGAN conditional GAN — systematically compared using 
        PSNR, SSIM, and perceptual colorfulness to isolate the contribution 
        of the adversarial objective.
      </p>
      {card('Architecture', 'The generator is a U-Net with 8 encoder and 8 decoder blocks connected by skip connections, taking the L (lightness) channel of a LAB image and predicting the AB (chrominance) channels. The discriminator is a 70×70 PatchGAN that judges 30×30 overlapping patches rather than the full image.')}
      {card('Training', 'Trained on the COCO 2017 dataset for 100 epochs with a combined adversarial + L1 reconstruction loss (λ=100), on a single RTX 3060 laptop GPU. Evaluated on PSNR, SSIM, and the Hasler–Süsstrunk colorfulness metric — full training curves and per-epoch qualitative samples are in the technical report.')}
      {card('Key finding', 'Standard pixel-level metrics (PSNR, SSIM) plateau early and do not track the visual quality improvements clearly visible across training epochs — colorfulness and qualitative inspection are necessary complements, not just final-epoch reporting.')}
      {card('Stack', 'PyTorch · FastAPI · React · Hugging Face Spaces · Vercel · COCO 2017 · scikit-image')}
    </div>
  );
}

function Paper() {
  return (
    <div style={{ maxWidth: 680, margin: '2rem auto', padding: '0 1rem', textAlign: 'center' }}>
      <h2 style={{ fontWeight: 700, fontSize: 24, marginBottom: 16 }}>Technical Report</h2>
      <p style={{ color: '#6b7280', marginBottom: 24, lineHeight: 1.6 }}>
        A comparative study of CNN, U-Net, and conditional GAN formulations
        for grayscale image colorization, including full training dynamics,
        an ablation isolating skip connections from the adversarial
        objective, and qualitative failure analysis.
      </p>
      <a href={PAPER_URL}
         target="_blank" rel="noreferrer"
         style={{
           padding: '12px 28px', background: '#6366f1', color: '#fff',
           borderRadius: 8, textDecoration: 'none', fontWeight: 500, fontSize: 15,
         }}>
        Read the report (PDF) ↗
      </a>
    </div>
  );
}

function Footer() {
  return (
    <footer style={{
      borderTop: '1px solid #e5e7eb', textAlign: 'center',
      padding: '2rem', color: '#9ca3af', fontSize: 13, marginTop: 64,
    }}>
      ChromaForge — independent research project by Biraj Subedi
    </footer>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<><Hero /><Colorizer /></>} />
        <Route path="/about" element={<About />} />
        <Route path="/paper" element={<Paper />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  );
}