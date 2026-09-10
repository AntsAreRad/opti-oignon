/**
 * Shared setup for the browser specs.
 *
 * A fresh install greets the first visit with a configuration dialog, and
 * it is modal: it covers whatever the page was going to show. That is
 * correct product behaviour and it is also the state every spec starts
 * from, so dismissing it belongs here rather than in each spec, where
 * forgetting it turns a real regression into a timeout on a locator that
 * was never reachable.
 *
 * The dialog opens only once the backend has answered what it knows about
 * the install. Waiting a fixed budget for it to appear cannot distinguish
 * "not yet" from "never": when the budget ran out first, this helper
 * returned as though there were nothing to dismiss, and the dialog opened
 * over the page a moment later and swallowed the next click. The failure
 * then landed on an unrelated locator and looked like a product regression.
 *
 * So the application marks the moment it has decided, and this waits for
 * that mark before looking. There is no budget here on purpose: if the
 * decision never comes, that is a real failure and the runner's own timeout
 * should report it, rather than this helper reporting success.
 */
import type { Page } from '@playwright/test';

/** Close the first-run dialog when the install has not been configured yet. */
export async function dismissFirstRun(page: Page): Promise<void> {
	await page
		.locator('html[data-onboarding="resolved"]')
		.waitFor({ state: 'attached' });

	const dialog = page.getByRole('dialog', { name: /welcome to opti-oignon/i });
	if (!(await dialog.isVisible())) {
		return;
	}

	await dialog.getByRole('button', { name: /close dialog/i }).click();
	await dialog.waitFor({ state: 'hidden' });
}
