import { useState, useCallback, useRef } from 'react';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
const MAX_SIZE_MB = 10;

/**
 * useDrop — handles drag-and-drop and file input selection with validation.
 * Returns drag state, error, and a validated File object.
 */
export function useDrop(onFile) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError]           = useState(null);
  const dragCounter                  = useRef(0);

  const validate = useCallback((file) => {
    if (!file) return 'No file received.';
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return `Unsupported format: ${file.type || 'unknown'}. Use JPEG, PNG, WebP, or BMP.`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File too large (${(file.size / 1e6).toFixed(1)} MB). Max: ${MAX_SIZE_MB} MB.`;
    }
    return null;
  }, []);

  const handleFile = useCallback((file) => {
    const err = validate(file);
    if (err) { setError(err); return; }
    setError(null);
    onFile(file);
  }, [validate, onFile]);

  const onDragEnter = useCallback((e) => {
    e.preventDefault();
    dragCounter.current += 1;
    if (dragCounter.current === 1) setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) setIsDragging(false);
  }, []);

  const onDragOver = useCallback((e) => { e.preventDefault(); }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    dragCounter.current = 0;
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const onInputChange = useCallback((e) => {
    const file = e.target.files[0];
    handleFile(file);
    e.target.value = ''; // Reset so the same file can be re-selected
  }, [handleFile]);

  return { isDragging, error, setError, onDragEnter, onDragLeave, onDragOver, onDrop, onInputChange };
}
