#!/usr/bin/env node
// A Cargo-compatible WASI runner: node wasm/run-wasi.mjs binary.wasm [args...]
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { WASI } from 'node:wasi';

const [binary, ...args] = process.argv.slice(2);
if (!binary) {
  console.error('Usage: node wasm/run-wasi.mjs binary.wasm [args...]');
  process.exitCode = 2;
} else {
  const scratch = await mkdtemp(join(tmpdir(), 'refeff-wasi-'));
  try {
    const wasi = new WASI({
      version: 'preview1',
      args: [binary, ...args],
      env: { ...process.env, TMPDIR: '/tmp' },
      preopens: { '.': process.cwd(), '/tmp': scratch },
      returnOnExit: true,
    });
    const module = await WebAssembly.compile(await readFile(binary));
    const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
    process.exitCode = wasi.start(instance);
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }
}
