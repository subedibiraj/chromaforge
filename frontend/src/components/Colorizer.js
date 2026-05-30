import { useState, useCallback, useRef } from 'react';
import { colorizeImage, ApiError } from '../utils/api';
import { useDrop } from '../hooks/useDrop';
import ComparisonSlider from './ComparisonSlider';

const STATES = { IDLE: 'idle', PROCESSING: 'processing', DONE: 'done', ERROR: 'error' };

export default function Colorizer() {
  const [state, setState]             = useState(STATES.IDLE);
  const [progress, setProgress]       = useState(0);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [resultUrl, setResultUrl]     = useState(null);
  const [errorMsg, setErrorMsg]       = useState('');
  const fileInputRef                  = useRef(null);

  const handleFile = useCallback(async (file) => {
    // Revoke previous object URLs
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    if (resultUrl)   URL.revokeObjectURL(resultUrl);

    setOriginalUrl(URL.createObjectURL(file));
    setResultUrl(null);
    setState(STATES.PROCESSING);
    setProgress(0);
    setErrorMsg('');

    try {
      const url = await colorizeImage(file, setProgress);
      setResultUrl(url);
      setState(STATES.DONE);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Something went wrong. Please try again.';
      setErrorMsg(msg);
      setState(STATES.ERROR);
    }
  }, [originalUrl, resultUrl]);

  const { isDragging, error: dropError, setError: setDropError,
          onDragEnter, onDragLeave, onDragOver, onDrop, onInputChange } = useDrop(handleFile);

  const reset = () => {
    setState(STATES.IDLE);
    setProgress(0);
    setErrorMsg('');
    setDropError(null);
  };

  const download = () => {
    if (!resultUrl) return;
    const a = document.createElement('a');
    a.href = resultUrl;
    a.download = 'chromaforge_colorized.jpg';
    a.click();
  };

  return (
    <div style={{ maxWidth: 720, margin: '0 auto', padding: '2rem 1rem' }}>

      {/* Drop zone */}
      {state === STATES.IDLE && (
        <div
          onDragEnter={onDragEnter}
          onDragLeave={onDragLeave}
          onDragOver={onDragOver}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload image drop zone"
          onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${isDragging ? '#6366f1' : '#d1d5db'}`,
            borderRadius: 16,
            padding: '3rem 2rem',
            textAlign: 'center',
            cursor: 'pointer',
            background: isDragging ? 'rgba(99,102,241,0.05)' : 'transparent',
            transition: 'all 0.2s',
          }}
        >
          <div style={{ fontSize: 48, marginBottom: 12 }}>🎨</div>
          <p style={{ fontWeight: 500, fontSize: 18, marginBottom: 6 }}>
            Drop a grayscale image here
          </p>
          <p style={{ color: '#6b7280', fontSize: 14, marginBottom: 16 }}>
            or click to browse — JPEG, PNG, WebP, BMP (max 10 MB)
          </p>
          <button
            style={{
              padding: '10px 24px',
              background: '#6366f1',
              color: '#fff',
              border: 'none',
              borderRadius: 8,
              fontSize: 15,
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            Choose image
          </button>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/bmp"
        style={{ display: 'none' }}
        onChange={onInputChange}
      />

      {/* Validation error */}
      {dropError && (
        <div style={{ marginTop: 12, padding: '10px 16px', background: '#fef2f2',
                      border: '1px solid #fecaca', borderRadius: 8, color: '#b91c1c', fontSize: 14 }}>
          {dropError}
        </div>
      )}

      {/* Processing */}
      {state === STATES.PROCESSING && (
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <p style={{ fontWeight: 500, marginBottom: 16, fontSize: 16 }}>
            Colorizing with ChromaForge GAN…
          </p>
          <div style={{ background: '#e5e7eb', borderRadius: 99, height: 8, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${progress}%`,
              background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
              borderRadius: 99,
              transition: 'width 0.3s ease',
            }} />
          </div>
          <p style={{ color: '#6b7280', fontSize: 13, marginTop: 8 }}>
            {progress < 50 ? 'Uploading…' : progress < 90 ? 'Running inference…' : 'Finalizing…'}
          </p>
          {originalUrl && (
            <img src={originalUrl} alt="uploaded"
                 style={{ marginTop: 20, maxWidth: '100%', maxHeight: 300,
                           borderRadius: 12, filter: 'grayscale(100%)', objectFit: 'contain' }} />
          )}
        </div>
      )}

      {/* Result */}
      {state === STATES.DONE && resultUrl && originalUrl && (
        <div>
          <p style={{ textAlign: 'center', color: '#6b7280', fontSize: 14, marginBottom: 12 }}>
            Drag the slider to compare original and colorized
          </p>
          <ComparisonSlider originalSrc={originalUrl} colorizedSrc={resultUrl} alt="result" />

          <div style={{ display: 'flex', gap: 12, marginTop: 20, justifyContent: 'center' }}>
            <button
              onClick={download}
              style={{
                padding: '10px 24px', background: '#6366f1', color: '#fff',
                border: 'none', borderRadius: 8, fontSize: 15, fontWeight: 500, cursor: 'pointer',
              }}
            >
              ⬇ Download
            </button>
            <button
              onClick={reset}
              style={{
                padding: '10px 24px', background: 'transparent', color: '#374151',
                border: '1.5px solid #d1d5db', borderRadius: 8, fontSize: 15, cursor: 'pointer',
              }}
            >
              Colorize another
            </button>
          </div>
        </div>
      )}

      {/* Error */}
      {state === STATES.ERROR && (
        <div style={{ padding: '1.5rem', textAlign: 'center' }}>
          <p style={{ fontSize: 16, fontWeight: 500, color: '#b91c1c', marginBottom: 8 }}>
            Colorization failed
          </p>
          <p style={{ color: '#6b7280', fontSize: 14, marginBottom: 20 }}>{errorMsg}</p>
          <button onClick={reset}
            style={{
              padding: '10px 24px', background: '#6366f1', color: '#fff',
              border: 'none', borderRadius: 8, fontSize: 15, cursor: 'pointer',
            }}>
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
