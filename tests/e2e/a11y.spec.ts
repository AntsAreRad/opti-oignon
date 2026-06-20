/**
 * E2E: Accessibility sweep (axe-core) -- S170, Polish/Accessibility/Themes.
 *
 * Spec 12.5 / Goal 1: "Final WCAG AA sweep across all 5 themes x 3 densities
 * (contrast on body, secondary, accent, and semantic colors). Automate with
 * axe-core where possible." This spec runs on the developer machine (it needs
 * a real browser + node_modules; the pytest sandbox cannot execute it).
 *
 * Two layers:
 *   1. A theme x density colour-contrast matrix. The five curated palettes
 *      (Anthracite, Parchment, Slate, Linen, High Contrast) x the three
 *      densities (compact, comfortable, spacious) are applied to <html> and
 *      the principal content routes are scanned with the `color-contrast`
 *      rule only -- this is exactly the "contrast on body/secondary/accent/
 *      semantic" check the spec asks for, and it stays free of false
 *      positives from unrelated rules.
 *   2. A structural WCAG 2.1 A/AA scan of the app shell in the default theme
 *      (landmarks, roles, names, focus order on the redesigned surfaces).
 *
 * The palettes/densities are kept in sync with $lib/stores/preferences:
 *   palette -> html[data-oo-theme] (+ .dark for dark palettes)
 *   density -> html.oo-density-<name>
 *   localStorage keys: oo-palette, oo-density (read on store init)
 */
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { setupAllMocks } from './mocks/routes';

const PALETTES = ['anthracite', 'parchment', 'slate', 'linen', 'high-contrast'] as const;
const DENSITIES = ['compact', 'comfortable', 'spacious'] as const;
const DARK_PALETTES = new Set(['anthracite', 'slate', 'high-contrast']);

// Principal content routes (single-user mode skips login).
const CONTENT_ROUTES = ['/chat', '/projects', '/health', '/benchmark'];

type Palette = (typeof PALETTES)[number];
type Density = (typeof DENSITIES)[number];

/** Seed the persisted preference before the app boots. */
async function seedTheme(page: import('@playwright/test').Page, palette: Palette, density: Density) {
  await page.addInitScript(
    ([p, d]) => {
      try {
        localStorage.setItem('oo-palette', p as string);
        localStorage.setItem('oo-density', d as string);
        // Force full motion off so the theme-transition class never lingers
        // during a scan.
        localStorage.setItem('oo-motion', 'reduced');
      } catch {
        /* storage unavailable -- the post-goto evaluate still applies it */
      }
    },
    [palette, density]
  );
}

/** Belt-and-suspenders: apply the palette/density directly to <html>. */
async function forceTheme(page: import('@playwright/test').Page, palette: Palette, density: Density) {
  await page.evaluate(
    ([p, d, isDark, densities]) => {
      const html = document.documentElement;
      html.setAttribute('data-oo-theme', p as string);
      html.classList.toggle('dark', isDark as boolean);
      (densities as string[]).forEach((x) => html.classList.remove(`oo-density-${x}`));
      html.classList.add(`oo-density-${d}`);
      html.classList.remove('theme-transitioning');
    },
    [palette, density, DARK_PALETTES.has(palette), [...DENSITIES]]
  );
  // Let any token-driven repaint settle.
  await page.waitForTimeout(150);
}

test.describe('Accessibility -- theme x density contrast matrix', () => {
  for (const palette of PALETTES) {
    for (const density of DENSITIES) {
      test(`contrast: ${palette} / ${density}`, async ({ page }) => {
        await seedTheme(page, palette, density);
        await setupAllMocks(page, true);

        for (const route of CONTENT_ROUTES) {
          await page.goto(route);
          await expect(page.locator('text=Opti-Oignon').first()).toBeVisible({ timeout: 5000 });
          await forceTheme(page, palette, density);

          const results = await new AxeBuilder({ page })
            .withRules(['color-contrast'])
            .analyze();

          expect(
            results.violations,
            `color-contrast violations on ${route} in ${palette}/${density}`
          ).toEqual([]);
        }
      });
    }
  }
});

test.describe('Accessibility -- structural WCAG 2.1 A/AA (app shell)', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, true);
  });

  for (const route of CONTENT_ROUTES) {
    test(`wcag2a/2aa structure: ${route}`, async ({ page }) => {
      await page.goto(route);
      await expect(page.locator('text=Opti-Oignon').first()).toBeVisible({ timeout: 5000 });

      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
        // color-contrast is exercised exhaustively by the matrix above; the
        // structural pass focuses on roles, names, landmarks and order.
        .disableRules(['color-contrast'])
        .analyze();

      expect(results.violations, `WCAG structural violations on ${route}`).toEqual([]);
    });
  }

  test('skip link and main landmark are present', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('text=Opti-Oignon').first()).toBeVisible({ timeout: 5000 });
    // Skip-to-content link (spec 8.9 / A6) and the main landmark (A7 anchor).
    await expect(page.locator('a[href="#main-content"]').first()).toHaveCount(1);
    await expect(page.locator('#main-content').first()).toHaveCount(1);
  });
});
