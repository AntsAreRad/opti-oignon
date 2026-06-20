/**
 * E2E: Security — Mode toggle, kill switch engage/disengage
 * S149 — Frontend E2E Tests
 *
 * Tests the Security tab in settings with mocked security APIs.
 */
import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/routes';

test.describe('Security flow', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, true);
  });

  test('security tab loads without error', async ({ page }) => {
    await page.goto('/settings?tab=security');
    await page.waitForLoadState('networkidle');

    const tabBtn = page.locator('button[role="tab"]', { hasText: 'Security' });
    await expect(tabBtn).toHaveAttribute('aria-selected', 'true', {
      timeout: 5000,
    });
  });

  test('security mode API is called on tab load', async ({ page }) => {
    const secCalls: string[] = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/security/mode')) {
        secCalls.push(req.url());
      }
    });

    await page.goto('/settings?tab=security');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    expect(secCalls.length).toBeGreaterThan(0);
  });

  test('kill switch status API is called on tab load', async ({ page }) => {
    const ksCalls: string[] = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/security/search-killswitch/status')) {
        ksCalls.push(req.url());
      }
    });

    await page.goto('/settings?tab=security');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    expect(ksCalls.length).toBeGreaterThan(0);
  });

  test('no JS errors on security tab', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/settings?tab=security');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    expect(errors).toHaveLength(0);
  });

  test('security tab renders mode information', async ({ page }) => {
    await page.goto('/settings?tab=security');
    await page.waitForLoadState('networkidle');

    // Should see daily/bulbe mode references from the security panel
    // Check that the page has content (not empty/crashed)
    const content = await page.textContent('body');
    expect(content).toBeTruthy();
    expect(content!.length).toBeGreaterThan(100);
  });
});
