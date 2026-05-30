/**
 * ChromaForge API client.
 * The base URL is configured via REACT_APP_API_URL environment variable,
 * making this work identically in development and production.
 */

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:7860';

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/**
 * Send a grayscale image to the backend and receive a colorized JPEG.
 * @param {File} file - Image file from input or drop
 * @param {(progress: number) => void} onProgress - Progress callback (0–100)
 * @returns {Promise<string>} Object URL of the colorized image
 */
export async function colorizeImage(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file, file.name);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        onProgress?.(Math.round((e.loaded / e.total) * 50));
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(100);
        const blob = new Blob([xhr.response], { type: 'image/jpeg' });
        resolve(URL.createObjectURL(blob));
      } else {
        let detail = 'Unknown error';
        try {
          const text = new TextDecoder().decode(xhr.response);
          detail = JSON.parse(text)?.detail || detail;
        } catch {}
        reject(new ApiError(detail, xhr.status));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new ApiError('Network error — check your connection.', 0));
    });

    xhr.addEventListener('timeout', () => {
      reject(new ApiError('Request timed out. Try a smaller image.', 408));
    });

    xhr.open('POST', `${BASE_URL}/colorize`);
    xhr.responseType = 'arraybuffer';
    xhr.timeout = 60_000;
    xhr.send(formData);

    let inferenceProgress = 50;
    const inferenceTimer = setInterval(() => {
      inferenceProgress = Math.min(inferenceProgress + 3, 90);
      onProgress?.(inferenceProgress);
    }, 500);

    xhr.addEventListener('loadend', () => clearInterval(inferenceTimer));
  });
}

/**
 * Check if the API is reachable.
 * @returns {Promise<{ok: boolean, model_loaded: boolean}>}
 */
export async function checkHealth() {
  try {
    const resp = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await fetch(`${BASE_URL}/`).then(r => r.json()).catch(() => ({}));
    return { ok: resp.ok, model_loaded: data.model_loaded ?? false };
  } catch {
    return { ok: false, model_loaded: false };
  }
}
