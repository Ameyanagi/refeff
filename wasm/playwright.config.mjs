import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: 'browser.spec.mjs',
  timeout: 60_000,
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:4179', browserName: 'chromium' },
  webServer: {
    command: 'node tests/server.mjs',
    url: 'http://127.0.0.1:4179',
    reuseExistingServer: false,
  },
});
