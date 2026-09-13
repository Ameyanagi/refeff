import { createServer } from 'node:http';
import { readFile, access } from 'node:fs/promises';
import { resolve, sep, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const examples = fileURLToPath(new URL('./examples/', import.meta.url));
const dist = fileURLToPath(new URL('./dist/', import.meta.url));
const sample = new URL('../crates/refeff/tests/data/znse.inp', import.meta.url);
try {
  await access(resolve(dist, 'refeff.wasm'));
} catch {
  console.error('Build the WebAssembly files first: npm run build --prefix wasm');
  process.exit(1);
}

const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.css': 'text/css',
  '.wasm': 'application/wasm', '.inp': 'text/plain' };
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    let path;
    if (pathname === '/sample.inp') {
      path = fileURLToPath(sample);
    } else {
      const isDist = pathname.startsWith('/dist/');
      const root = resolve(isDist ? dist : examples);
      const relative = isDist ? pathname.slice('/dist/'.length) : pathname.slice(1) || 'index.html';
      path = resolve(root, relative);
      if (!path.startsWith(root + sep)) {
        response.writeHead(403).end('Forbidden');
        return;
      }
    }
    const bytes = await readFile(path);
    response.writeHead(200, {
      'Content-Type': types[extname(path)] ?? 'application/octet-stream',
      'Cache-Control': 'no-cache',
      'X-Content-Type-Options': 'nosniff',
    });
    response.end(bytes);
  } catch {
    response.writeHead(404).end('Not found');
  }
});
const port = Number(process.env.PORT ?? 4173);
server.on('error', error => {
  console.error(error.message);
  process.exitCode = 1;
});
server.listen(port, '127.0.0.1', () => {
  console.log(`ReFEFF local example: http://127.0.0.1:${server.address().port}`);
});
