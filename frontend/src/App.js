import React from 'react';
import { BrowserRouter, Routes, Route, Link, NavLink } from 'react-router-dom';
import Colorizer from './components/Colorizer';

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
      <NavLink to="/"     style={link} end>Colorize</NavLink>
      <NavLink to="/about" style={link}>About</NavLink>
      <NavLink to="/paper" style={link}>Paper</NavLink>
      <a href="https://github.com/your-username/chromaforge"
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
      <p style={{ color: '#6b7280', maxWidth: 520, margin: '1rem auto 0', fontSize: 16, lineHeight: 1.6 }}>
        Upload any grayscale image. Our U-Net + PatchGAN model predicts vivid,
        perceptually realistic colors in the LAB color space.
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
      <h2 style={{ fontWeight: 700, fontSize: 24, marginBottom: 24 }}>About ChromaForge</h2>
      {card('Architecture', 'ChromaForge uses a U-Net generator with 8 encoder and 8 decoder layers connected by skip connections, paired with a 70×70 PatchGAN discriminator. The generator input is the L (lightness) channel of a LAB image; it predicts the AB (chrominance) channels.')}
      {card('Training', 'Trained on the COCO 2017 dataset (118k images) for 100 epochs using an adversarial loss combined with an L1 reconstruction loss (λ=100). The model is evaluated on PSNR, SSIM, and the Hasler–Süsstrunk colorfulness metric.')}
      {card('Deployment', 'The inference backend is a FastAPI service deployed on Hugging Face Spaces. The frontend is deployed on Vercel. Both are free-tier.')}
      {card('Technology', 'PyTorch · FastAPI · React · Hugging Face Spaces · Vercel · COCO 2017 · scikit-image')}
    </div>
  );
}

function Paper() {
  return (
    <div style={{ maxWidth: 680, margin: '2rem auto', padding: '0 1rem', textAlign: 'center' }}>
      <h2 style={{ fontWeight: 700, fontSize: 24, marginBottom: 16 }}>Technical Report</h2>
      <p style={{ color: '#6b7280', marginBottom: 24 }}>
        The full IEEE-format paper with methodology, experiments, and results
        is available in the repository.
      </p>
      <a href="https://github.com/your-username/chromaforge/blob/main/docs/chromaforge_paper.pdf"
         target="_blank" rel="noreferrer"
         style={{
           padding: '12px 28px', background: '#6366f1', color: '#fff',
           borderRadius: 8, textDecoration: 'none', fontWeight: 500, fontSize: 15,
         }}>
        View paper (PDF) ↗
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
      ChromaForge — Final Year Project in Conditional GAN Image Colorization
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
