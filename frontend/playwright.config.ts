/**
 * Playwright E2E test configuration for Opti-Oignon frontend.
 * S149 — Frontend E2E Tests
 *
 * Usage:
 *   npx playwright test              # run all E2E tests
 *   npx playwright test --ui         # interactive UI mode
 *   npx playwright test --headed     # visible browser
 *   npx playwright show-report       # view HTML report
 */
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, devices } from '@playwright/test';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  /* Test directory — E2E specs live in tests/e2e/ at project root */
  testDir: '../tests/e2e',

  /* Maximum time one test can run */
  timeout: 30_000,

  /* Expect assertions timeout */
  expect: {
    timeout: 5_000,
    /* Visual comparison thresholds */
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.05,
      threshold: 0.2,
    },
  },

  /* Fail the build on CI if test.only is left in source */
  forbidOnly: !!process.env.CI,

  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,

  /* Parallel workers */
  workers: process.env.CI ? 1 : undefined,

  /* Reporters */
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: '../tests/e2e/playwright-report' }],
  ],

  /* Shared settings for all projects */
  use: {
    /* Base URL for navigation */
    baseURL: 'http://localhost:5173',

    /* Collect trace on first retry */
    trace: 'on-first-retry',

    /* Screenshot on failure */
    screenshot: 'only-on-failure',

    /* Video on failure (CI) */
    video: process.env.CI ? 'on-first-retry' : 'off',

    /* Default viewport */
    viewport: { width: 1280, height: 720 },

    /* Timeouts */
    actionTimeout: 10_000,
    navigationTimeout: 15_000,
  },

  /* Browser projects */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    /* Mobile viewport for responsive tests */
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] },
    },
  ],

  /* Dev server auto-start */
  webServer: {
    command: 'npm run dev',
    port: 5173,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
    cwd: __dirname,
  },
});
