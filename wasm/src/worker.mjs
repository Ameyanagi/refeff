import { execute } from './runtime.mjs';

self.onmessage = async ({ data }) => {
  try {
    const response = await fetch(data.wasmUrl);
    if (!response.ok) throw new Error(`Failed to fetch WebAssembly: HTTP ${response.status}`);
    // Compile bytes so static hosts need not set application/wasm MIME headers.
    const module = await WebAssembly.compile(await response.arrayBuffer());
    const result = await execute(module, data, log => self.postMessage({ type: 'log', ...log }));
    self.postMessage({ type: 'result', result }, Object.values(result.files).map(bytes => bytes.buffer));
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
