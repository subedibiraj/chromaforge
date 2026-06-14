import { useState, useRef, useCallback } from 'react';

export default function ComparisonSlider({ originalSrc, colorizedSrc, alt = 'comparison' }) {
  const [pos, setPos]           = useState(50);
  const [dragging, setDragging] = useState(false);
  const containerRef            = useRef(null);

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

  const onKeyDown = useCallback((e) => {
    if (e.key === 'ArrowLeft')  setPos(p => Math.max(0,   p - 2));
    if (e.key === 'ArrowRight') setPos(p => Math.min(100, p + 2));
  }, []);

  const imgStyle = {
    display: 'block',
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    objectPosition: 'center',
  };

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        userSelect: 'none',
        borderRadius: 12,
        overflow: 'hidden',
        cursor: dragging ? 'grabbing' : 'col-resize',
        boxShadow: '0 4px 24px rgba(0,0,0,0.15)',
        background: '#000',
        maxHeight: 520,
        aspectRatio: '4/3',
      }}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {/* Colorized — right side, full width base layer */}
      <img
        src={colorizedSrc}
        alt={`${alt} colorized`}
        style={{ ...imgStyle, position: 'absolute', inset: 0 }}
        draggable={false}
      />

      {/* Original — left side, clipped */}
      <div style={{
        position: 'absolute',
        inset: 0,
        clipPath: `inset(0 ${100 - pos}% 0 0)`,
        transition: dragging ? 'none' : 'clip-path 0.05s',
      }}>
        <img
          src={originalSrc}
          alt={`${alt} original`}
          style={{ ...imgStyle, position: 'absolute', inset: 0 }}
          draggable={false}
        />
      </div>

      {/* Label — Original (left) */}
      <span style={{
        position: 'absolute', top: 12, left: 12,
        background: 'rgba(0,0,0,0.6)', color: '#fff',
        fontSize: 11, fontWeight: 600, padding: '4px 10px',
        borderRadius: 4, pointerEvents: 'none', letterSpacing: '0.03em',
        opacity: pos > 15 ? 1 : 0, transition: 'opacity 0.2s',
      }}>
        Original
      </span>

      {/* Label — Colorized (right) */}
      <span style={{
        position: 'absolute', top: 12, right: 12,
        background: 'rgba(99,102,241,0.85)', color: '#fff',
        fontSize: 11, fontWeight: 600, padding: '4px 10px',
        borderRadius: 4, pointerEvents: 'none', letterSpacing: '0.03em',
        opacity: pos < 85 ? 1 : 0, transition: 'opacity 0.2s',
      }}>
        Colorized
      </span>

      {/* Divider */}
      <div
        role="slider"
        aria-label="Comparison slider"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pos)}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onKeyDown={onKeyDown}
        style={{
          position: 'absolute', top: 0, bottom: 0,
          left: `calc(${pos}% - 1px)`,
          width: 2,
          background: '#fff',
          boxShadow: '0 0 8px rgba(0,0,0,0.6)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'col-resize',
          outline: 'none',
        }}
      >
        <div style={{
          width: 38, height: 38,
          borderRadius: '50%',
          background: '#fff',
          boxShadow: '0 2px 12px rgba(0,0,0,0.35)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
          gap: 4,
        }}>
          <span style={{ fontSize: 10, color: '#444', lineHeight: 1 }}>◀</span>
          <span style={{ fontSize: 10, color: '#444', lineHeight: 1 }}>▶</span>
        </div>
      </div>
    </div>
  );
}