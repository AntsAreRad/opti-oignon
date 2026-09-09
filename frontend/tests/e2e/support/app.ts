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
 * the install, so it is waited for rather than sampled: an instantaneous
 * check runs before it exists, reports nothing to dismiss, and lets it
 * open on top of the page a moment later.
 */
import type { Page } from '@playwright/test';

/** How long a fresh install may take to raise its first-run dialog. */
const APPEARANCE_BUDGET_MS = 7_000;

/** Close the first-run dialog when the install has not been configured yet. */
export async function dismissFirstRun(page: Page): Promise<void> {
	const dialog = page.getByRole('dialog', { name: /welcome to opti-oignon/i });

	const appeared = await dialog
		.waitFor({ state: 'visible', timeout: APPEARANCE_BUDGET_MS })
		.then(() => true)
		.catch(() => false);

	if (!appeared) {
		return;
	}

	await dialog.getByRole('button', { name: /close dialog/i }).click();
	await dialog.waitFor({ state: 'hidden' });
}
