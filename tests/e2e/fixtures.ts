/**
 * Shared Playwright fixtures for Opti-Oignon E2E tests.
 * S149 — Provides pre-configured test contexts.
 *
 * Usage:
 *   import { test, expect } from './fixtures';
 *   test('my test', async ({ authedPage }) => { ... });
 */

import { test as base, expect, type Page } from '@playwright/test';
import { setupAllMocks, setupAuthMocks } from './mocks/routes';
import { MOCK_TOKENS, MOCK_USER } from './mocks/data';

/**
 * Extended test fixtures.
 */
type E2EFixtures = {
  /** Page with all API mocks active (multi-user mode, pre-authenticated). */
  authedPage: Page;
  /** Page with all API mocks active (single-user mode). */
  singleUserPage: Page;
};

export const test = base.extend<E2EFixtures>({
  authedPage: async ({ page }, use) => {
    await setupAllMocks(page, false);
    // Navigate to login, perform mock login
    await page.goto('/login');
    await page.fill('[data-testid="username"], input[name="username"], input[type="text"]', 'testuser');
    await page.fill('[data-testid="password"], input[name="password"], input[type="password"]', 'Test1234!');
    await page.click('[data-testid="login-btn"], button[type="submit"]');
    // Wait for redirect to chat
    await page.waitForURL('**/chat**', { timeout: 5000 }).catch(() => {
      // May already be on chat if single-user fallback
    });
    await use(page);
  },

  singleUserPage: async ({ page }, use) => {
    await setupAllMocks(page, true);
    await page.goto('/');
    // Single-user mode should auto-redirect to chat
    await page.waitForURL('**/chat**', { timeout: 5000 }).catch(() => {});
    await use(page);
  },
});

export { expect };
