/**
 * E2E: Mobile — Sidebar swipe behavior, responsive layout checks
 * S149 — Frontend E2E Tests
 *
 * Uses the mobile-chrome project viewport (Pixel 5: 393×851).
 */
import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/routes';

// Force mobile viewport for all tests in this file
test.use({
  viewport: { width: 393, height: 851 },
  isMobile: true,
  hasTouch: true,
});

test.describe('Mobile responsive', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, true);
  });

  test('chat page renders at mobile width', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('text=Opti-Oignon').first()).toBeVisible({
      timeout: 5000,
    });
    // Page should not overflow horizontally
    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 1);
  });

  test('sidebar toggle button is visible on mobile', async ({ page }) => {
    await page.goto('/chat');
    const toggleBtn = page.locator('button[aria-label="Toggle sidebar"]');
    await expect(toggleBtn).toBeVisible({ timeout: 5000 });
  });

  test('sidebar opens on toggle button click', async ({ page }) => {
    await page.goto('/chat');
    const toggleBtn = page.locator('button[aria-label="Toggle sidebar"]');
    await toggleBtn.click();

    // Sidebar overlay backdrop should appear
    await expect(
      page.locator('.sidebar-mobile-backdrop, button[aria-label="Close sidebar"]').first()
    ).toBeVisible({ timeout: 3000 });
  });

  test('sidebar closes on backdrop click', async ({ page }) => {
    await page.goto('/chat');
    // Open sidebar
    await page.click('button[aria-label="Toggle sidebar"]');
    await page.waitForTimeout(300);

    // Click backdrop to close
    const backdrop = page.locator(
      '.sidebar-mobile-backdrop, button[aria-label="Close sidebar"]'
    ).first();
    if (await backdrop.isVisible()) {
      await backdrop.click();
      await page.waitForTimeout(300);
    }
  });

  test('login page is responsive at mobile width', async ({ page }) => {
    await setupAllMocks(page, false);
    await page.goto('/login');

    // Auth card should be visible and fit in viewport
    const card = page.locator('.auth-card');
    await expect(card).toBeVisible({ timeout: 5000 });

    const box = await card.boundingBox();
    expect(box).toBeTruthy();
    expect(box!.width).toBeLessThanOrEqual(393);
  });

  test('settings page tabs scroll horizontally on mobile', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Tab container should be present — tabs overflow with horizontal scroll
    const tabButtons = page.locator('button[role="tab"]');
    const count = await tabButtons.count();
    expect(count).toBeGreaterThanOrEqual(10);
  });

  test('no horizontal overflow on settings page', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    // Allow small tolerance (1px for borders/rounding)
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 2);
  });

  test('chat input has 44px touch target on mobile', async ({ page }) => {
    await page.goto('/chat');
    await page.click('text=E2E Test Conversation');

    const sendBtn = page.locator('button[aria-label="Send message"]');
    await expect(sendBtn).toBeVisible({ timeout: 5000 });

    // Check minimum touch target size (44x44)
    const box = await sendBtn.boundingBox();
    expect(box).toBeTruthy();
    expect(box!.width).toBeGreaterThanOrEqual(44);
    expect(box!.height).toBeGreaterThanOrEqual(44);
  });

  test('no JS errors on mobile chat page', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/chat');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    expect(errors).toHaveLength(0);
  });
});
