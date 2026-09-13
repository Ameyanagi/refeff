import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('complete EXAFS executes in a worker and returns transferable spectra', async ({ page }) => {
  const input = await readFile(new URL('../../crates/refeff/tests/data/znse.inp', import.meta.url), 'utf8');
  await page.goto('/');
  const result = await page.evaluate(async input => {
    const { runFeff } = await import('/index.mjs');
    let ticks = 0, logs = 0;
    const timer = setInterval(() => ticks++, 10);
    const result = await runFeff({ input, onLog: () => logs++ });
    clearInterval(timer);
    return {
      exitCode: result.exitCode, report: result.report, ticks, logs,
      chi: new TextDecoder().decode(result.files['chi.dat']),
      paths: Object.keys(result.files).filter(name => /^feff\d{4}\.dat$/.test(name)).length,
    };
  }, input);
  expect(result.exitCode).toBe(0);
  expect(result.report.ok).toBe(true);
  expect(result.report.data.effective_threads).toBe(1);
  expect(result.paths).toBe(15);
  expect(result.chi).toContain('chi');
  expect(result.ticks).toBeGreaterThan(0);
  expect(result.logs).toBeGreaterThan(0);
});

test('input errors preserve the structured CLI report', async ({ page }) => {
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const { runFeff } = await import('/index.mjs');
    return runFeff({ input: 'INCLUDE missing.inp\n' });
  });
  expect(result.exitCode).not.toBe(0);
  expect(result.report.ok).toBe(false);
  expect(result.report.error.message).toContain('missing.inp');
});

test('cancellation terminates an active worker and supports pre-aborted signals', async ({ page }) => {
  const input = await readFile(new URL('../../crates/refeff/tests/data/znse.inp', import.meta.url), 'utf8');
  await page.goto('/');
  const names = await page.evaluate(async input => {
    const { runFeff } = await import('/index.mjs');
    const controller = new AbortController();
    const pending = runFeff({ input, signal: controller.signal,
      onLog: () => controller.abort() });
    const active = await pending.then(() => 'unexpected success', error => error.name);
    const preAborted = await runFeff({ input, signal: controller.signal })
      .then(() => 'unexpected success', error => error.name);
    return [active, preAborted];
  }, input);
  expect(names).toEqual(['AbortError', 'AbortError']);
});

test('worker fetch failures reject the caller', async ({ page }) => {
  await page.goto('/');
  const message = await page.evaluate(async () => {
    const { runFeff } = await import('/index.mjs');
    return runFeff({ input: '', wasmUrl: '/missing.wasm' }).catch(error => error.message);
  });
  expect(message).toContain('HTTP 404');
});
