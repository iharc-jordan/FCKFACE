import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests/browser',
  workers: 1,
  timeout: 30_000,
  use: { baseURL: 'http://127.0.0.1:5173', acceptDownloads: true },
  webServer: {
    command: 'node ./node_modules/vite/bin/vite.js --mode research --host 127.0.0.1 --port 5173 --strictPort',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: !process.env.CI,
  },
})
