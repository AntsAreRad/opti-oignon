# Module Map

## Project structure

```
opti-oignon/
  opti_oignon/           # Backend (Python / FastAPI)
    api/                 # REST API routes (~519 endpoints)
    agents/              # Agentic workflows
    cli/                 # CLI companion (oo command)
    config/              # YAML configuration files
    data/                # Static data (redteam seeds, allowlists)
    middleware/          # Auth, CSRF, CSP middleware
    plugins/             # Built-in plugins
    rag/                 # RAG subsystem (ingest, query, sanitize)
    redteam/             # Red team engine
    ...                  # ~255 modules total
  frontend/              # Frontend (SvelteKit + Tailwind)
    src/
      lib/               # Shared components and stores
      routes/            # SvelteKit page routes
      styles/            # CSS with --oo-* variables
  tests/                 # Test suite (~9300 tests, 151 files)
  scripts/               # Dev scripts (lint, coverage, e2e, signing)
  docs/                  # Documentation (this site)
  assets/                # Logo and static assets
  data/                  # Shared data files
```


## Backend module groups

### Core inference

| Module | Purpose |
|--------|---------|
| `inference_backend.py` | Ollama and llama.cpp integration |
| `model_manager.py` | Model listing, profiles, capability scores |
| `model_health.py` | Health monitoring, availability tracking |
| `model_warmup.py` | Pre-loading models on startup |
| `model_lifecycle.py` | Model loading/unloading lifecycle |
| `model_profiles.py` | 15-dimension numeric capability profiles |

### Routing and pipelines

| Module | Purpose |
|--------|---------|
| `smart_router.py` | Heuristic model selection |
| `learned_router.py` | ML-based routing from feedback history |
| `adaptive_routing.py` | Runtime routing adjustments |
| `pipeline_manager.py` | Pipeline type selection and orchestration |
| `pipelines.py` | Pipeline implementations (9 types) |
| `cascading.py` | 3-tier cascading inference |
| `speculative.py` | Speculative generation |
| `consensus.py` | Multi-model consensus (vote, merge) |

### Intelligence

| Module | Purpose |
|--------|---------|
| `agentic_executor.py` | Orchestrates multi-step pipelines |
| `dynamic_planning.py` | Automatic pipeline construction |
| `reasoning.py` | Advanced reasoning strategies |
| `self_correction.py` | Iterative refinement loop |
| `context_manager.py` | Token budget and context allocation |
| `context_optimizer.py` | Context compression and optimization |
| `conversation_compressor.py` | History summarization |
| `working_memory.py` | Persistent agent memory |

### Coding agent

| Module | Purpose |
|--------|---------|
| `coding_agent.py` | Autonomous coding loop |
| `chat_coding_agent.py` | Chat-integrated coding mode |
| `code_executor.py` | Sandboxed code execution |
| `coding_history.py` | Code generation history |

### Security

| Module | Purpose |
|--------|---------|
| `auth.py` | Authentication, JWT, session management |
| `auth_2fa.py` | WebAuthn/FIDO2 and TOTP |
| `encryption.py` | AES-256-GCM, key management |
| `db_encryption.py` | SQLCipher wrapper |
| `pqc_signatures.py` | ML-DSA-65 post-quantum signatures |
| `secure_bytes.py` | mlock'd key storage |
| `network_hardening.py` | Network security policies |
| `network_bind_guard.py` | Localhost binding enforcement |
| `luks_detector.py` | Disk encryption detection |
| `pii_sanitizer.py` | PII detection and redaction |
| `admin_audit.py` | Audit chain management |
| `audit_anchor_export.py` | QR and JSON export for audit chain |
| `dep_monitor.py` | Dependency vulnerability scanning |
| `security_scheduler.py` | Automated security scheduling |

### RAG

| Module | Purpose |
|--------|---------|
| `rag/embeddings.py` | Embedding generation, batch processing |
| `rag/ingest.py` | Document chunking and ingestion |
| `rag/parallel_ingest.py` | Parallel ingestion with thread pool |
| `rag/query.py` | Retrieval and context augmentation |
| `rag/sanitizer.py` | Prompt injection filtering |
| `rag/pool_integration.py` | Connection pool for RAG databases |

### Plugins

| Module | Purpose |
|--------|---------|
| `plugin_loader.py` | Plugin discovery and loading |
| `plugin_manifest.py` | Manifest validation |
| `plugin_subprocess.py` | Unix socket IPC with HMAC |
| `async_plugin_subprocess.py` | Pipe-based async IPC |
| `plugin_hooks.py` | Hook dispatch and chaining |
| `plugin_allowlist.py` | Marketplace allowlist |
| `plugin_installer.py` | Install and dependency resolution |

### Red team

| Module | Purpose |
|--------|---------|
| `redteam/generator.py` | Attack generation from seeds |
| `redteam/strategies.py` | Obfuscation strategies |
| `redteam/runner.py` | Campaign orchestration |
| `redteam/scoring.py` | Pass/fail scoring |
| `redteam/reports.py` | Report generation and storage |
| `redteam/targets.py` | Target adapter implementations |


## Frontend structure

The frontend is a SvelteKit application with Tailwind CSS. All
styling uses `--oo-*` CSS variables (no hardcoded hex colors).

Key components (~137 total) include the chat interface, settings
panels, benchmark dashboard, plugin marketplace, RAG management,
red team dashboard, and theme engine.
