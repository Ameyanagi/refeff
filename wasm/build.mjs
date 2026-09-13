import { spawnSync } from 'node:child_process';
import { copyFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { build } from 'esbuild';

const root = fileURLToPath(new URL('../', import.meta.url));
const result = spawnSync('cargo', [
  'build', '--release', '--locked', '--target', 'wasm32-wasip1',
  '-p', 'refeff-cli', '--bin', 'refeff', '--bin', 'feff',
], { cwd: root, stdio: 'inherit' });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);

const dist = new URL('./dist/', import.meta.url);
await mkdir(dist, { recursive: true });
await build({
  entryPoints: [fileURLToPath(new URL('./src/worker.mjs', import.meta.url))],
  outfile: fileURLToPath(new URL('worker.mjs', dist)),
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: 'es2022',
});
await copyFile(new URL('./src/index.mjs', import.meta.url), new URL('index.mjs', dist));
for (const license of ['LICENSE-MIT', 'LICENSE-APACHE']) {
  await copyFile(new URL(`./node_modules/@bjorn3/browser_wasi_shim/${license}`, import.meta.url),
    new URL(`browser-wasi-shim-${license}`, dist));
}
// Honor Cargo's target directory override when used by embedding projects/CI.
const metadata = spawnSync('cargo', ['metadata', '--no-deps', '--format-version', '1', '--locked'],
  { cwd: root, encoding: 'utf8' });
if (metadata.error) throw metadata.error;
if (metadata.status !== 0) throw new Error(metadata.stderr);
const target = JSON.parse(metadata.stdout).target_directory;
await copyFile(resolve(target, 'wasm32-wasip1/release/refeff.wasm'), new URL('refeff.wasm', dist));
