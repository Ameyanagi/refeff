import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../dist', import.meta.url));
createServer(async (request, response) => {
  if (request.url === '/') {
    response.setHeader('Content-Type', 'text/html');
    response.end('<!doctype html><title>ReFEFF WebAssembly tests</title>');
    return;
  }
  const path = resolve(root, '.' + new URL(request.url, 'http://localhost').pathname);
  if (!path.startsWith(root + sep)) {
    response.writeHead(403).end();
    return;
  }
  try {
    const bytes = await readFile(path);
    response.setHeader('Content-Type', path.endsWith('.mjs') ? 'text/javascript' : 'application/wasm');
    response.end(bytes);
  } catch {
    response.writeHead(404).end();
  }
}).listen(4179, '127.0.0.1');
