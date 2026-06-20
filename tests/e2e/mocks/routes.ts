/**
 * Playwright route interception for Opti-Oignon E2E tests.
 * S149 — Mocked backend support via page.route().
 *
 * Usage in tests:
 *   import { setupAllMocks, setupAuthMocks } from './mocks/routes';
 *   test.beforeEach(async ({ page }) => { await setupAllMocks(page); });
 */

import type { Page, Route } from '@playwright/test';
import * as D from './data';

// ── Helpers ───────────────────────────────────────────────────────────────

/** Fulfill a route with JSON data. */
function json(route: Route, data: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(data),
  });
}

// ── Auth mocks ────────────────────────────────────────────────────────────

export async function setupAuthMocks(page: Page, singleUser = false) {
  const status = singleUser ? D.MOCK_AUTH_STATUS_SINGLE : D.MOCK_AUTH_STATUS;

  await page.route('**/api/auth/status', (route) => json(route, status));

  await page.route('**/api/auth/login', async (route) => {
    const body = route.request().postDataJSON();
    if (body?.username === 'testuser' && body?.password === 'Test1234!') {
      return json(route, D.MOCK_TOKENS);
    }
    return json(route, { detail: 'Invalid credentials' }, 401);
  });

  await page.route('**/api/auth/register', async (route) => {
    const body = route.request().postDataJSON();
    if (body?.username && body?.password) {
      return json(route, D.MOCK_TOKENS);
    }
    return json(route, { detail: 'Registration failed' }, 400);
  });

  await page.route('**/api/auth/me', (route) => json(route, D.MOCK_USER));

  await page.route('**/api/auth/refresh', (route) => json(route, D.MOCK_TOKENS));

  await page.route('**/api/auth/logout', (route) =>
    json(route, { logged_out: true })
  );

  await page.route('**/api/auth/settings', async (route) => {
    if (route.request().method() === 'GET') {
      return json(route, D.MOCK_USER_SETTINGS);
    }
    // PUT — merge updates and return
    const body = route.request().postDataJSON?.() ?? {};
    return json(route, { ...D.MOCK_USER_SETTINGS, ...body });
  });
}

// ── Conversation mocks ────────────────────────────────────────────────────

export async function setupConversationMocks(page: Page) {
  await page.route('**/api/conversations', async (route) => {
    if (route.request().method() === 'POST') {
      return json(route, D.MOCK_CONVERSATION);
    }
    return json(route, D.MOCK_CONVERSATIONS_LIST);
  });

  await page.route('**/api/conversations/*/messages', (route) =>
    json(route, D.MOCK_MESSAGES)
  );

  await page.route(/\/api\/conversations\/[^/]+$/, async (route) => {
    const method = route.request().method();
    if (method === 'DELETE') {
      return route.fulfill({ status: 204 });
    }
    if (method === 'PATCH') {
      return json(route, { ...D.MOCK_CONVERSATION, title: 'Renamed' });
    }
    return json(route, {
      ...D.MOCK_CONVERSATION,
      messages: D.MOCK_MESSAGES,
    });
  });
}

// ── Chat streaming mock (WebSocket → SSE-like fallback) ───────────────────

export async function setupChatMocks(page: Page) {
  // The real app uses WebSocket for streaming. For E2E, we mock
  // the non-streaming /api/chat/send endpoint as a fallback that
  // Playwright can intercept. The frontend also has a non-WS path.
  await page.route('**/api/chat/send', (route) =>
    json(route, {
      id: 'msg-e2e-new',
      role: 'assistant',
      content: 'Hello! I can help you.',
      timestamp: new Date().toISOString(),
      model: 'llama3.2:3b',
      done: true,
    })
  );

  // Mock cancel endpoint
  await page.route('**/api/chat/cancel', (route) =>
    json(route, { cancelled: true })
  );
}

// ── Models / Ollama mocks ─────────────────────────────────────────────────

export async function setupModelMocks(page: Page) {
  await page.route('**/api/models', (route) =>
    json(route, D.MOCK_OLLAMA_MODELS)
  );

  await page.route('**/api/models/tags', (route) =>
    json(route, D.MOCK_OLLAMA_MODELS)
  );

  await page.route('**/api/health', (route) => json(route, D.MOCK_HEALTH));

  await page.route('**/api/health/**', (route) => json(route, D.MOCK_HEALTH));
}

// ── RAG mocks ─────────────────────────────────────────────────────────────

export async function setupRAGMocks(page: Page) {
  await page.route('**/api/rag/collections', async (route) => {
    if (route.request().method() === 'POST') {
      return json(route, {
        id: 'col-new',
        name: 'New Collection',
        document_count: 0,
        created_at: new Date().toISOString(),
      });
    }
    return json(route, D.MOCK_RAG_COLLECTIONS);
  });

  await page.route('**/api/rag/ingest', (route) =>
    json(route, D.MOCK_RAG_INGEST)
  );

  await page.route('**/api/rag/query', (route) =>
    json(route, D.MOCK_RAG_QUERY)
  );

  // Batch ingest jobs
  await page.route('**/api/rag/ingest/jobs', (route) =>
    json(route, { jobs: [], total: 0 })
  );
}

// ── Plugin mocks ──────────────────────────────────────────────────────────

export async function setupPluginMocks(page: Page) {
  let plugins = JSON.parse(JSON.stringify(D.MOCK_PLUGINS_LIST.plugins));

  await page.route('**/api/plugins', (route) =>
    json(route, { plugins, total: plugins.length })
  );

  await page.route('**/api/plugins/*/enable', async (route) => {
    const url = route.request().url();
    const id = url.split('/plugins/')[1]?.split('/')[0];
    const p = plugins.find((pl: { id: string }) => pl.id === id);
    if (p) p.state = 'enabled';
    return json(route, { id, state: 'enabled', success: true });
  });

  await page.route('**/api/plugins/*/disable', async (route) => {
    const url = route.request().url();
    const id = url.split('/plugins/')[1]?.split('/')[0];
    const p = plugins.find((pl: { id: string }) => pl.id === id);
    if (p) p.state = 'disabled';
    return json(route, { id, state: 'disabled', success: true });
  });

  // Plugin allowlist
  await page.route('**/api/security/plugin-allowlist/**', (route) =>
    json(route, { allowed: true })
  );
}

// ── Security mocks ────────────────────────────────────────────────────────

export async function setupSecurityMocks(page: Page) {
  let killSwitchState = { ...D.MOCK_KILL_SWITCH };

  await page.route('**/api/security/mode', async (route) => {
    if (route.request().method() === 'GET') {
      return json(route, D.MOCK_SECURITY_MODE);
    }
    // POST: switch mode
    const body = route.request().postDataJSON?.() ?? {};
    return json(route, {
      ...D.MOCK_SECURITY_MODE,
      current_mode: body.mode ?? 'daily',
    });
  });

  await page.route('**/api/security/search-killswitch/status', (route) =>
    json(route, killSwitchState)
  );

  await page.route('**/api/security/search-killswitch/engage', (route) => {
    killSwitchState = {
      ...killSwitchState,
      search_enabled: false,
      killed_at: Date.now() / 1000,
      killed_by: 'e2e-user',
      kill_reason: 'E2E test',
    };
    return json(route, killSwitchState);
  });

  await page.route('**/api/security/search-killswitch/reenable', (route) => {
    killSwitchState = { ...D.MOCK_KILL_SWITCH };
    return json(route, killSwitchState);
  });

  // Audit chain
  await page.route('**/api/security/audit-chain/**', (route) =>
    json(route, { entries: [], total: 0, chain_valid: true })
  );

  // Red team
  await page.route('**/api/security/redteam/**', (route) =>
    json(route, { status: 'idle', results: null })
  );
}

// ── Catch-all for unhandled API routes ────────────────────────────────────

export async function setupFallbackMock(page: Page) {
  await page.route('**/api/**', (route) => {
    // Let through if already handled by a more specific mock
    // Otherwise return an empty success to prevent frontend errors
    return json(route, {});
  });
}

// ── Master setup ──────────────────────────────────────────────────────────

/**
 * Set up all API mocks for a full E2E test.
 * Call in test.beforeEach to intercept all API requests.
 */
export async function setupAllMocks(page: Page, singleUser = false) {
  await setupAuthMocks(page, singleUser);
  await setupConversationMocks(page);
  await setupChatMocks(page);
  await setupModelMocks(page);
  await setupRAGMocks(page);
  await setupPluginMocks(page);
  await setupSecurityMocks(page);
  // Fallback must be last — it matches **/api/**
  await setupFallbackMock(page);
}
