/**
 * E2E: Chat flow — New conversation → send message → receive response
 * S149 — Frontend E2E Tests
 *
 * Uses single-user mode to skip login.
 */
import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/routes';

test.describe('Chat flow', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page, true);
  });

  test('chat page loads in single-user mode', async ({ page }) => {
    await page.goto('/chat');
    // Should see the app shell with Opti-Oignon title
    await expect(page.locator('text=Opti-Oignon').first()).toBeVisible({ timeout: 5000 });
  });

  test('sidebar shows conversation list', async ({ page }) => {
    await page.goto('/chat');
    // The conversation list should render mocked conversations
    await expect(
      page.locator('text=E2E Test Conversation').first()
    ).toBeVisible({ timeout: 5000 });
  });

  test('new conversation button exists', async ({ page }) => {
    await page.goto('/chat');
    const newBtn = page.locator('text=New conversation').first();
    await expect(newBtn).toBeVisible({ timeout: 5000 });
  });

  test('clicking a conversation loads messages', async ({ page }) => {
    await page.goto('/chat');
    // Click on the mocked conversation
    await page.click('text=E2E Test Conversation');
    // Wait for messages to appear
    await expect(
      page.locator('text=Hello, how are you?').first()
    ).toBeVisible({ timeout: 5000 });
    await expect(
      page.locator('text=Hello! I am doing well.').first()
    ).toBeVisible({ timeout: 5000 });
  });

  test('chat input is visible and functional', async ({ page }) => {
    await page.goto('/chat');
    await page.click('text=E2E Test Conversation');
    const textarea = page.locator('textarea[placeholder="Type a message..."]');
    await expect(textarea).toBeVisible({ timeout: 5000 });
    await textarea.fill('Test message from E2E');
    await expect(textarea).toHaveValue('Test message from E2E');
  });

  test('send button becomes active with text input', async ({ page }) => {
    await page.goto('/chat');
    await page.click('text=E2E Test Conversation');
    const textarea = page.locator('textarea[placeholder="Type a message..."]');
    await textarea.fill('Hello world');
    // Send button should be enabled
    const sendBtn = page.locator('button[aria-label="Send message"]');
    await expect(sendBtn).toBeEnabled({ timeout: 3000 });
  });

  test('send button is disabled when input is empty', async ({ page }) => {
    await page.goto('/chat');
    await page.click('text=E2E Test Conversation');
    const sendBtn = page.locator('button[aria-label="Send message"]');
    await expect(sendBtn).toBeDisabled({ timeout: 3000 });
  });

  test('model selector is visible in header', async ({ page }) => {
    await page.goto('/chat');
    await page.click('text=E2E Test Conversation');
    // ModelSelector should be present (desktop)
    const modelSelector = page.locator('text=llama3.2').first();
    await expect(modelSelector).toBeVisible({ timeout: 5000 });
  });
});
