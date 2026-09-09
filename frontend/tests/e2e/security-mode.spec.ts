/**
 * The served origin reports a security mode the product recognises.
 *
 * The mode is a policy the whole product reads from, so an origin that
 * cannot name it is a broken install however well the interface paints.
 * The request is issued from the page's own origin, which keeps the proxy
 * leg under test while staying independent of which panel happens to be
 * mounted or which layout preference is stored.
 */
import { expect, test } from '@playwright/test';

import { dismissFirstRun } from './support/app';

const KNOWN_MODES = ['daily', 'bulbe'];

test('the origin reports a recognised security mode', async ({ page }) => {
	await page.goto('/');
	await dismissFirstRun(page);

	const response = await page.request.get('/api/security/mode');
	expect(response.status()).toBe(200);

	const body = await response.json();
	expect(KNOWN_MODES).toContain(body.mode);
});
