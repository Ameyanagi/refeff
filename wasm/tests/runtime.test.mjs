import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { execute } from '../src/runtime.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const target = JSON.parse(execFileSync('cargo', ['metadata', '--no-deps', '--format-version', '1', '--locked'],
  { cwd: root, encoding: 'utf8' })).target_directory;
const binary = join(target, 'wasm32-wasip1/release/refeff.wasm');
const module = await WebAssembly.compile(await readFile(binary));
const input = await readFile(new URL('../../crates/refeff/tests/data/znse.inp', import.meta.url));
const minimal = 'TITLE contracts\nCONTROL 0 0 0 0 0 0\nPOTENTIALS\n0 29 Cu\nATOMS\n0 0 0 0 Cu\nEND\n';

test('virtual filesystem resolves nested auxiliary inputs and isolates runs', async () => {
  const first = await execute(module, {
    input: 'INCLUDE nested/銅.inp\n', files: { 'nested/銅.inp': minimal },
  });
  assert.equal(first.exitCode, 0, first.stdout);
  assert.equal(first.report.data.atoms, 1);
  assert.ok(first.files['pot.inp'] instanceof Uint8Array);
  const second = await execute(module, { input: 'INCLUDE nested/銅.inp\n' });
  assert.notEqual(second.exitCode, 0);
  assert.equal(second.report.ok, false);
});

test('invalid workspace paths fail before execution', async () => {
  for (const path of ['../escape', '/absolute', 'a/../b', 'a\\b', 'a//b', 'a\0b']) {
    await assert.rejects(execute(module, { input: minimal, files: { [path]: 'bad' } }), /Invalid workspace path/);
  }
});

function rows(bytes) {
  return new TextDecoder().decode(bytes).split('\n')
    .filter(line => line.trim() && !line.trimStart().startsWith('#'))
    .map(line => line.trim().split(/\s+/).map(Number));
}

function compareSpectrum(actual, expected, name) {
  const a = rows(actual), b = rows(expected);
  assert.equal(a.length, b.length, `${name}: row count`);
  assert.ok(a.length > 100, `${name}: complete spectrum`);
  for (let i = 0; i < a.length; i++) {
    assert.equal(a[i].length, b[i].length);
    for (let j = 0; j < a[i].length; j++) {
      assert.ok(Number.isFinite(a[i][j]) && Number.isFinite(b[i][j]));
      assert.ok(Math.abs(a[i][j] - b[i][j]) <= 5e-8 + Math.abs(b[i][j]) * 5e-5,
        `${name}[${i},${j}]: ${a[i][j]} != ${b[i][j]}`);
    }
  }
}

test('complete ZnSe spectra agree across native, WASI and browser filesystem', { timeout: 120_000 }, async () => {
  const directory = await mkdtemp(join(target, 'wasm-parity-'));
  try {
    await writeFile(join(directory, 'feff.inp'), input);
    const options = { cwd: directory, encoding: 'utf8', timeout: 90_000, stdio: ['ignore', 'pipe', 'pipe'] };
    execFileSync(join(target, 'release/refeff'), [
      '--threads', '1', '--json', 'run', '--input', 'feff.inp', '--output', 'native',
    ], options);
    const stdout = execFileSync(process.execPath, [
      join(root, 'wasm/run-wasi.mjs'), binary,
      '--threads', '8', '--json', 'run', '--input', 'feff.inp', '--output', 'wasi',
    ], options);
    assert.equal(JSON.parse(stdout).data.effective_threads, 1);
    const browser = await execute(module, { input });
    assert.equal(browser.exitCode, 0, browser.stdout);
    assert.equal(browser.report.data.effective_threads, 1);
    assert.ok(browser.report.data.stages.some(stage => stage.name === 'ff2x'));
    assert.equal(Object.keys(browser.files).filter(name => /^feff\d{4}\.dat$/.test(name)).length, 15);
    for (const name of ['chi.dat', 'xmu.dat']) {
      const native = await readFile(join(directory, 'native', name));
      compareSpectrum(await readFile(join(directory, 'wasi', name)), native, `WASI ${name}`);
      compareSpectrum(browser.files[name], native, `browser ${name}`);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
