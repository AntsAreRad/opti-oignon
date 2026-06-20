/**
 * Mock data for Opti-Oignon E2E tests.
 * S149 — Shared mock payloads used by route interceptors.
 */

// ── Auth ──────────────────────────────────────────────────────────────────

export const MOCK_USER = {
  user_id: 'e2e-user-001',
  username: 'testuser',
  email: 'test@opti-oignon.local',
  role: 'admin',
  created_at: 1700000000,
  updated_at: 1700000000,
  metadata: {},
};

export const MOCK_TOKENS = {
  access_token: 'e2e-access-token-mock',
  refresh_token: 'e2e-refresh-token-mock',
  token_type: 'bearer',
  expires_in: 3600,
};

export const MOCK_AUTH_STATUS = {
  available: true,
  single_user_mode: false,
  registration_enabled: true,
  user_count: 1,
  cookie_mode: false,
};

export const MOCK_AUTH_STATUS_SINGLE = {
  available: true,
  single_user_mode: true,
  registration_enabled: false,
  user_count: 0,
  cookie_mode: false,
};

export const MOCK_USER_SETTINGS = {
  theme: 'dark',
  language: 'en',
  default_model: 'llama3.2:3b',
  temperature: 0.7,
  system_prompt: '',
};

// ── Conversations ─────────────────────────────────────────────────────────

export const MOCK_CONVERSATION = {
  id: 'conv-e2e-001',
  title: 'E2E Test Conversation',
  created_at: '2025-12-01T10:00:00Z',
  updated_at: '2025-12-01T10:05:00Z',
  message_count: 2,
  model: 'llama3.2:3b',
};

export const MOCK_CONVERSATIONS_LIST = [MOCK_CONVERSATION];

export const MOCK_MESSAGES = [
  {
    id: 'msg-001',
    role: 'user',
    content: 'Hello, how are you?',
    timestamp: '2025-12-01T10:00:00Z',
    model: null,
  },
  {
    id: 'msg-002',
    role: 'assistant',
    content: 'Hello! I am doing well. How can I help you today?',
    timestamp: '2025-12-01T10:00:02Z',
    model: 'llama3.2:3b',
  },
];

// ── Ollama / Models ───────────────────────────────────────────────────────

export const MOCK_OLLAMA_MODELS = {
  models: [
    {
      name: 'llama3.2:3b',
      modified_at: '2025-11-01T00:00:00Z',
      size: 2_000_000_000,
      digest: 'sha256:abc123',
      details: {
        format: 'gguf',
        family: 'llama',
        parameter_size: '3B',
        quantization_level: 'Q4_K_M',
      },
    },
  ],
};

export const MOCK_CHAT_RESPONSE_TOKENS = [
  { token: 'Hello', done: false },
  { token: '!', done: false },
  { token: ' I', done: false },
  { token: ' can', done: false },
  { token: ' help', done: false },
  { token: ' you', done: false },
  { token: '.', done: false },
  { token: '', done: true, model: 'llama3.2:3b', total_duration: 500000000 },
];

// ── RAG ───────────────────────────────────────────────────────────────────

export const MOCK_RAG_COLLECTIONS = {
  collections: [
    {
      id: 'col-e2e-001',
      name: 'Test Collection',
      document_count: 3,
      created_at: '2025-12-01T00:00:00Z',
    },
  ],
  total: 1,
};

export const MOCK_RAG_INGEST = {
  document_id: 'doc-e2e-001',
  collection_id: 'col-e2e-001',
  filename: 'test.pdf',
  status: 'completed',
  chunks: 12,
};

export const MOCK_RAG_QUERY = {
  answer: 'Based on the uploaded document, the key finding is that test results are positive.',
  sources: [
    {
      document_id: 'doc-e2e-001',
      chunk_id: 'chunk-003',
      content: 'Test results indicate positive outcomes across all metrics.',
      score: 0.92,
    },
  ],
  model: 'llama3.2:3b',
};

// ── Plugins ───────────────────────────────────────────────────────────────

export const MOCK_PLUGINS_LIST = {
  plugins: [
    {
      id: 'plugin-calculator',
      name: 'Calculator',
      version: '1.0.0',
      description: 'Basic arithmetic operations',
      state: 'enabled',
      author: 'opti-oignon',
    },
    {
      id: 'plugin-websearch',
      name: 'Web Search',
      version: '2.1.0',
      description: 'Search the web via local proxy',
      state: 'disabled',
      author: 'opti-oignon',
    },
  ],
  total: 2,
};

// ── Security Mode ─────────────────────────────────────────────────────────

export const MOCK_SECURITY_MODE = {
  current_mode: 'daily',
  available_modes: ['daily', 'bulbe'],
  policy: {
    web_search_allowed: true,
    db_encryption_required: false,
    two_fa_required: false,
    plugin_allowlist_required: false,
    sandbox_bwrap_required: false,
    session_timeout: 3600,
    backup_encryption_required: false,
    cookie_samesite: 'Lax',
    tool_call_approval_required: false,
    rate_limit_max_attempts: 10,
    rate_limit_window: 300,
    bearer_auth_allowed: true,
  },
};

export const MOCK_KILL_SWITCH = {
  available: true,
  search_enabled: true,
  killed_at: null,
  killed_by: null,
  kill_reason: null,
  circuit_breaker_tripped: false,
  injection_count: 0,
  reenable_pending: false,
  domain_allowlist: { enabled: false, domain_count: 0, domains: [] },
};

// ── Health ─────────────────────────────────────────────────────────────────

export const MOCK_HEALTH = {
  status: 'healthy',
  version: '3.2.0',
  ollama_connected: true,
  uptime: 86400,
};
