/**
 * Bad credentials are refused, and the refusal is visible.
 *
 * The negative path is the one worth pinning: a sign-in form that fails
 * silently, or that lets a wrong password through, is the same screen as
 * one that works. This asserts the refusal reaches the user rather than
 * only the console.
 */
import { expect, test } from '@playwright/test';

import { dismissFirstRun } from './support/app';

test('a wrong password is refused with a visible message', async ({ page }) => {
	await page.goto('/login');
	await dismissFirstRun(page);

	await page.getByPlaceholder('Enter your username').fill('no-such-account');
	await page.getByPlaceholder('Enter your password').fill('wrong-password');
	await page.getByRole('button', { name: 'Sign in' }).click();

	const alert = page.getByRole('alert');
	await expect(alert).toBeVisible();
	await expect(alert).toContainText(/invalid|too many/i);
	await expect(page).toHaveURL(/\/login$/);
});
