/**
 * Run FEFF in a dedicated browser worker with an isolated in-memory filesystem.
 * input/files accept strings or Uint8Arrays. Results contain a CLI JSON report,
 * exitCode, and output files keyed by relative path. AbortSignal terminates the
 * worker and discards that calculation. No SharedArrayBuffer is required.
 */
export function runFeff({ input, files = {}, signal, onLog,
  wasmUrl = new URL('./refeff.wasm', import.meta.url) } = {}) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new DOMException('Calculation aborted', 'AbortError'));
      return;
    }
    const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
    const cleanup = () => {
      signal?.removeEventListener('abort', abort);
      worker.terminate();
    };
    const abort = () => {
      cleanup();
      reject(signal.reason ?? new DOMException('Calculation aborted', 'AbortError'));
    };
    signal?.addEventListener('abort', abort, { once: true });
    worker.onerror = event => {
      cleanup();
      reject(new Error(event.message));
    };
    worker.onmessage = ({ data }) => {
      if (data.type === 'log') {
        try { onLog?.({ stream: data.stream, text: data.text }); }
        catch (error) { cleanup(); reject(error); }
      } else {
        cleanup();
        if (data.type === 'result') resolve(data.result);
        else reject(new Error(data.message));
      }
    };
    try {
      worker.postMessage({ input, files, wasmUrl: new URL(wasmUrl, import.meta.url).href });
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}
