/**
 * E2E: Settings — Navigate all settings tabs without crash
 * S149 — Frontend E2E Tests
 */
import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/routes';

const SETTINGS_TABS = [
  { id: 'quick', label: 'Quick' },
  { id: 'presets', label: 'Presets' },
  { id: 'models', label: 'Models' },
  { id: 'prompt', label: 'Prompt' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'performance', label: 'Observe' },
  { id: 'fine-tune', label: 'Fine-Tune' },
  { id: 'knowledge', label: 'Knowledge' },
  { id: 'plugins', label: 'Plugins' },
  { id: 'backup', label: 'Backup' },
  { id: 'security', label: 'Security' },
  { id: 'advanced', label: 'Advanced' },
];

test.describe('Settings page', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, true);
  });

  test('settings page loads successfully', async ({ page }) => {
    await page.goto('/settings');
    // Quick tab should be active by default
    const quickTab = page.locator('button[role="tab"]', { hasText: 'Quick' });
    await expect(quickTab).toBeVisible({ timeout: 5000 });
    await expect(quickTab).toHaveAttribute('aria-selected', 'true');
  });

  for (const tab of SETTINGS_TABS) {
    test(`tab "${tab.label}" loads without error`, async ({ page }) => {
      await page.goto('/settings');
      // Wait for page to stabilize
      await page.waitForLoadState('networkidle');

      // Click the tab
      const tabBtn = page.locator('button[role="tab"]', { hasText: tab.label });
      await tabBtn.click();

      // Verify tab is selected
      await expect(tabBtn).toHaveAttribute('aria-selected', 'true', {
        timeout: 3000,
      });

      // Verify no uncaught errors (check that the page didn't crash)
      // The tab content container should still be present
      await page.waitForTimeout(500);
      await expect(page.locator('button[role="tab"]').first()).toBeVisible();
    });
  }

  test('tab URL parameter works', async ({ page }) => {
    await page.goto('/settings?tab=security');
    await page.waitForLoadState('networkidle');
    const secTab = page.locator('button[role="tab"]', { hasText: 'Security' });
    await expect(secTab).toHaveAttribute('aria-selected', 'true', {
      timeout: 5000,
    });
  });

  test('navigate all tabs sequentially without crash', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    for (const tab of SETTINGS_TABS) {
      const tabBtn = page.locator('button[role="tab"]', { hasText: tab.label });
      await tabBtn.click();
      await page.waitForTimeout(300);
    }

    // No JS errors should have occurred
    expect(errors).toHaveLength(0);
  });
});
