/**
 * E2E: Auth flow — Login → Register → Logout
 * S149 — Frontend E2E Tests
 */
import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/routes';

test.describe('Auth flow', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, false);
  });

  test('login page renders with form elements', async ({ page }) => {
    await page.goto('/login');
    await expect(page.locator('h1.auth-title')).toHaveText('Opti-Oignon');
    await expect(page.locator('.auth-subtitle')).toHaveText('Sign in to your account');
    await expect(page.locator('#login-username')).toBeVisible();
    await expect(page.locator('#login-password')).toBeVisible();
    await expect(page.locator('.auth-btn-primary')).toHaveText('Sign in');
  });

  test('login with valid credentials redirects to chat', async ({ page }) => {
    await page.goto('/login');
    await page.fill('#login-username', 'testuser');
    await page.fill('#login-password', 'Test1234!');
    await page.click('.auth-btn-primary');
    await page.waitForURL('**/chat**', { timeout: 5000 });
    expect(page.url()).toContain('/chat');
  });

  test('login with invalid credentials shows error', async ({ page }) => {
    await page.goto('/login');
    await page.fill('#login-username', 'wrong');
    await page.fill('#login-password', 'wrong');
    await page.click('.auth-btn-primary');
    await expect(page.locator('.auth-error')).toBeVisible({ timeout: 3000 });
  });

  test('login with empty fields shows validation error', async ({ page }) => {
    await page.goto('/login');
    await page.click('.auth-btn-primary');
    await expect(page.locator('.auth-error')).toHaveText(
      'Please enter both username and password.'
    );
  });

  test('Enter key submits login form', async ({ page }) => {
    await page.goto('/login');
    await page.fill('#login-username', 'testuser');
    await page.fill('#login-password', 'Test1234!');
    await page.press('#login-password', 'Enter');
    await page.waitForURL('**/chat**', { timeout: 5000 });
    expect(page.url()).toContain('/chat');
  });

  test('register link navigates to registration page', async ({ page }) => {
    await page.goto('/login');
    await page.click('.auth-link');
    await expect(page).toHaveURL(/\/register/);
    await expect(page.locator('.auth-subtitle')).toHaveText('Create your account');
  });

  test('register page renders with form elements', async ({ page }) => {
    await page.goto('/register');
    await expect(page.locator('#reg-username')).toBeVisible();
    await expect(page.locator('#reg-email')).toBeVisible();
    await expect(page.locator('#reg-password')).toBeVisible();
    await expect(page.locator('#reg-confirm')).toBeVisible();
    await expect(page.locator('.auth-btn-primary')).toHaveText('Create account');
  });

  test('register with valid data redirects to chat', async ({ page }) => {
    await page.goto('/register');
    await page.fill('#reg-username', 'newuser');
    await page.fill('#reg-email', 'new@test.com');
    await page.fill('#reg-password', 'SecurePass1!');
    await page.fill('#reg-confirm', 'SecurePass1!');
    await page.click('.auth-btn-primary');
    await page.waitForURL('**/chat**', { timeout: 5000 });
    expect(page.url()).toContain('/chat');
  });

  test('register with mismatched passwords shows error', async ({ page }) => {
    await page.goto('/register');
    await page.fill('#reg-username', 'newuser');
    await page.fill('#reg-password', 'SecurePass1!');
    await page.fill('#reg-confirm', 'DifferentPass!');
    await page.click('.auth-btn-primary');
    await expect(page.locator('.auth-error')).toHaveText('Passwords do not match.');
  });

  test('register with short password shows error', async ({ page }) => {
    await page.goto('/register');
    await page.fill('#reg-username', 'newuser');
    await page.fill('#reg-password', 'short');
    await page.fill('#reg-confirm', 'short');
    await page.click('.auth-btn-primary');
    await expect(page.locator('.auth-error')).toHaveText(
      'Password must be at least 8 characters.'
    );
  });

  test('sign in link on register page navigates back', async ({ page }) => {
    await page.goto('/register');
    await page.click('.auth-link');
    await expect(page).toHaveURL(/\/login/);
  });
});
