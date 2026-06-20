/**
 * E2E: Visual Regression Baseline — Capture reference screenshots
 * S149 — Frontend E2E Tests
 *
 * Run once to generate baselines:
 *   npx playwright test visual.spec.ts --update-snapshots
 *
 * Subsequent runs compare against baselines stored in:
 *   tests/e2e/visual.spec.ts-snapshots/
 *
 * Thresholds configured in playwright.config.ts:
 *   maxDiffPixelRatio: 0.05, threshold: 0.2
 */
import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/routes';

// ── Desktop screenshots (1280×720) ───────────────────────────────────────

test.describe('Visual regression — Desktop', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, true);
  });

  test('chat page — empty state', async ({ page }) => {
    await page.goto('/chat');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('desktop-chat-empty.png', {
      fullPage: true,
    });
  });

  test('chat page — with messages', async ({ page }) => {
    await page.goto('/chat');
    await page.click('text=E2E Test Conversation');
    await expect(
      page.locator('text=Hello, how are you?').first()
    ).toBeVisible({ timeout: 5000 });
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('desktop-chat-messages.png', {
      fullPage: true,
    });
  });

  test('login page', async ({ page }) => {
    await setupAllMocks(page, false);
    await page.goto('/login');
    await expect(page.locator('.auth-card')).toBeVisible({ timeout: 5000 });
    await page.waitForTimeout(300);
    await expect(page).toHaveScreenshot('desktop-login.png', {
      fullPage: true,
    });
  });

  test('register page', async ({ page }) => {
    await setupAllMocks(page, false);
    await page.goto('/register');
    await expect(page.locator('.auth-card')).toBeVisible({ timeout: 5000 });
    await page.waitForTimeout(300);
    await expect(page).toHaveScreenshot('desktop-register.png', {
      fullPage: true,
    });
  });

  test('settings page — Quick tab', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('desktop-settings-quick.png', {
      fullPage: true,
    });
  });

  test('settings page — Models tab', async ({ page }) => {
    await page.goto('/settings?tab=models');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('desktop-settings-models.png', {
      fullPage: true,
    });
  });

  test('settings page — Knowledge tab', async ({ page }) => {
    await page.goto('/settings?tab=knowledge');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('desktop-settings-knowledge.png', {
      fullPage: true,
    });
  });

  test('settings page — Plugins tab', async ({ page }) => {
    await page.goto('/settings?tab=plugins');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('desktop-settings-plugins.png', {
      fullPage: true,
    });
  });

  test('settings page — Security tab', async ({ page }) => {
    await page.goto('/settings?tab=security');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('desktop-settings-security.png', {
      fullPage: true,
    });
  });

  test('settings page — Backup tab', async ({ page }) => {
    await page.goto('/settings?tab=backup');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('desktop-settings-backup.png', {
      fullPage: true,
    });
  });

  test('health page', async ({ page }) => {
    await page.goto('/health');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('desktop-health.png', {
      fullPage: true,
    });
  });
});

// ── Mobile screenshots (393×851 — Pixel 5) ───────────────────────────────

test.describe('Visual regression — Mobile', () => {
  test.use({
    viewport: { width: 393, height: 851 },
    isMobile: true,
    hasTouch: true,
  });

  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, true);
  });

  test('chat page — mobile empty state', async ({ page }) => {
    await page.goto('/chat');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('mobile-chat-empty.png', {
      fullPage: true,
    });
  });

  test('chat page — mobile with sidebar open', async ({ page }) => {
    await page.goto('/chat');
    await page.waitForLoadState('networkidle');
    await page.click('button[aria-label="Toggle sidebar"]');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('mobile-chat-sidebar.png', {
      fullPage: true,
    });
  });

  test('login page — mobile', async ({ page }) => {
    await setupAllMocks(page, false);
    await page.goto('/login');
    await expect(page.locator('.auth-card')).toBeVisible({ timeout: 5000 });
    await page.waitForTimeout(300);
    await expect(page).toHaveScreenshot('mobile-login.png', {
      fullPage: true,
    });
  });

  test('settings page — mobile', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot('mobile-settings.png', {
      fullPage: true,
    });
  });
});
