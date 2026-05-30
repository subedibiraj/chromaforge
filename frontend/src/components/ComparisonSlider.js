import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * ComparisonSlider — drag the divider to reveal original vs. colorized.
 * Pure CSS + pointer events, no external library.
 */
export default function ComparisonSlider({ originalSrc, colorizedSrc, alt = 'comparison' }) {
  const [pos, setPos]       = useState(50);   // percent
  const [dragging, setDragging] = useState(false);
  const containerRef        = useRef(null);

  const getPercent = useCallback((clientX) => {
    const rect = containerRef.current.getBoundingClientRect();
    return Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
  }, []);

  const onPointerDown = useCallback((e) => {
    e.preventDefault();
    setDragging(true);
    containerRef.current.setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e) => {
    if (!dragging) return;
    setPos(getPercent(e.clientX));
  }, [dragging, getPercent]);

  const onPointerUp = useCallback(() => setDragging(false), []);

  // Keyboard accessibility
  const onKeyDown = useCallback((e) => {
    if (e.key === 'ArrowLeft')  setPos(p => Math.max(0,   p - 2));
    if (e.key === 'ArrowRight') setPos(p => Math.min(100, p + 2));
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        userSelect: 'none',
        borderRadius: '12px',
        overflow: 'hidden',
        cursor: dragging ? 'grabbing' : 'col-resize',
        boxShadow: '0 4px 24px rgba(0,0,0,0.15)',
      }}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {/* Colorized (full width, underneath) */}
      <img
        src={colorizedSrc}
        alt={`${alt} colorized`}
        style={{ display: 'block', width: '100%', height: 'auto', maxHeight: '500px', objectFit: 'contain' }}
        draggable={false}
      />

      {/* Original (clipped to left of slider) */}
      <div style={{
        position: 'absolute', inset: 0,
        clipPath: `inset(0 ${100 - pos}% 0 0)`,
        transition: dragging ? 'none' : 'clip-path 0.05s',
      }}>
        <img
          src={originalSrc}
          alt={`${alt} original`}
          style={{ display: 'block', width: '100%', height: 'auto', maxHeight: '500px', objectFit: 'contain', filter: 'grayscale(100%)' }}
          draggable={false}
        />
      </div>

      {/* Labels */}
      <span style={{
        position: 'absolute', top: 12, left: 12,
        background: 'rgba(0,0,0,0.55)', color: '#fff',
        fontSize: 12, fontWeight: 500, padding: '3px 8px',
        borderRadius: 4, pointerEvents: 'none',
        opacity: pos > 20 ? 1 : 0, transition: 'opacity 0.2s',
      }}>Original</span>
      <span style={{
        position: 'absolute', top: 12, right: 12,
        background: 'rgba(0,0,0,0.55)', color: '#fff',
        fontSize: 12, fontWeight: 500, padding: '3px 8px',
        borderRadius: 4, pointerEvents: 'none',
        opacity: pos < 80 ? 1 : 0, transition: 'opacity 0.2s',
      }}>Colorized</span>

      {/* Divider line + handle */}
      <div
        role="slider"
        aria-label="Comparison slider"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pos)}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onKeyDown={onKeyDown}
        style={{
          position: 'absolute', top: 0, bottom: 0,
          left: `calc(${pos}% - 1px)`,
          width: 2,
          background: '#fff',
          boxShadow: '0 0 8px rgba(0,0,0,0.5)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'col-resize',
          outline: 'none',
        }}
      >
        {/* Handle circle */}
        <div style={{
          width: 36, height: 36,
          borderRadius: '50%',
          background: '#fff',
          boxShadow: '0 2px 12px rgba(0,0,0,0.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
          gap: 3,
        }}>
          {['◀', '▶'].map((ch, i) => (
            <span key={i} style={{ fontSize: 10, color: '#444', lineHeight: 1 }}>{ch}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
