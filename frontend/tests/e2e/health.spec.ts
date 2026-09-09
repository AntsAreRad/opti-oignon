/**
 * System status renders and its data comes from a live backend.
 *
 * The narrowest path that proves the whole desktop chain is wired: the
 * browser loads the served application, the application asks the backend
 * for its health, and the answer arrives through the dev proxy. A render
 * assertion alone would pass against a backend that is down, so the
 * network leg is asserted with it.
 */
import { expect, test } from '@playwright/test';

import { dismissFirstRun } from './support/app';

test('system status renders and its health call succeeds', async ({ page }) => {
	const health = page.waitForResponse(
		(response) => response.url().includes('/api/health') && response.status() === 200
	);

	await page.goto('/health');
	await dismissFirstRun(page);

	// The page currently paints its title twice; this pins that the status
	// view rendered at all, not how many headings carry its name.
	const title = page.getByRole('heading', { name: 'System Status' }).first();
	await expect(title).toBeVisible();
	await health;
});
