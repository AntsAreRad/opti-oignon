/**
 * E2E: RAG flow — Upload file → query → results displayed
 * S149 — Frontend E2E Tests
 *
 * Tests the Knowledge tab in settings and RAG API interactions.
 */
import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/routes';

test.describe('RAG flow', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, true);
  });

  test('knowledge tab loads with collections', async ({ page }) => {
    await page.goto('/settings?tab=knowledge');
    await page.waitForLoadState('networkidle');

    // Knowledge tab should be active
    const tabBtn = page.locator('button[role="tab"]', { hasText: 'Knowledge' });
    await expect(tabBtn).toHaveAttribute('aria-selected', 'true', {
      timeout: 5000,
    });

    // Mock collection name should appear
    await expect(
      page.locator('text=Test Collection').first()
    ).toBeVisible({ timeout: 5000 });
  });

  test('RAG collections API is called on tab load', async ({ page }) => {
    const ragCalls: string[] = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/rag/collections')) {
        ragCalls.push(req.url());
      }
    });

    await page.goto('/settings?tab=knowledge');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    expect(ragCalls.length).toBeGreaterThan(0);
  });

  test('no JS errors on knowledge tab', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/settings?tab=knowledge');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    expect(errors).toHaveLength(0);
  });
});
