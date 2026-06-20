/**
 * E2E: Plugins — Enable/disable plugin lifecycle
 * S149 — Frontend E2E Tests
 *
 * Tests the Plugins tab in settings with mocked plugin list.
 */
import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/routes';

test.describe('Plugins flow', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, true);
  });

  test('plugins tab loads with plugin list', async ({ page }) => {
    await page.goto('/settings?tab=plugins');
    await page.waitForLoadState('networkidle');

    const tabBtn = page.locator('button[role="tab"]', { hasText: 'Plugins' });
    await expect(tabBtn).toHaveAttribute('aria-selected', 'true', {
      timeout: 5000,
    });

    // Mocked plugins should appear
    await expect(
      page.locator('text=Calculator').first()
    ).toBeVisible({ timeout: 5000 });
    await expect(
      page.locator('text=Web Search').first()
    ).toBeVisible({ timeout: 5000 });
  });

  test('plugins API is called on tab load', async ({ page }) => {
    const pluginCalls: string[] = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/plugins') && !req.url().includes('allowlist')) {
        pluginCalls.push(req.url());
      }
    });

    await page.goto('/settings?tab=plugins');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    expect(pluginCalls.length).toBeGreaterThan(0);
  });

  test('no JS errors on plugins tab', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/settings?tab=plugins');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    expect(errors).toHaveLength(0);
  });
});
