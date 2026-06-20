# API Reference -- Endpoints Added in v3.2.0 (S138-S150)

This document covers the API endpoints introduced during the S138-S150
development cycle. For the full OpenAPI specification, see
http://localhost:8000/docs when the backend is running.

All endpoints require authentication via JWT cookie unless noted
otherwise. Admin-only endpoints require `role: admin`.


## Red Team Engine (S147-S148)

Prefix: `/api/security/redteam`

### POST /api/security/redteam/run

Launch a red team audit campaign. Runs asynchronously; poll
`/status` for progress.

**Request body:**

```json
{
  "categories": ["injection", "jailbreak"],
  "strategies": ["base64_encode", "roleplay"],
  "targets": ["rag_sanitizer", "chat"],
  "attacks_per_category": 10
}
```

All fields are optional. Omitting a field uses all enabled values
from `config/redteam.yaml`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| categories | list[str] or null | all enabled | Attack categories to test |
| strategies | list[str] or null | all enabled | Obfuscation strategies to apply |
| targets | list[str] or null | all enabled | Target adapters to evaluate |
| attacks_per_category | int (1-100) or null | from config | Attacks generated per category |

**Response (200):**

```json
{
  "status": "started",
  "estimated_steps": 80
}
```

**Errors:** 409 if a campaign is already running. 503 if the red team
module is not available.

### GET /api/security/redteam/status

Get current campaign progress.

**Response (200):**

```json
{
  "running": true,
  "progress": {
    "total_steps": 80,
    "completed_steps": 35,
    "current_category": "injection",
    "current_target": "rag_sanitizer"
  },
  "has_results": false,
  "error": null
}
```

### GET /api/security/redteam/results

Get the latest campaign results after completion.

**Response (200):**

```json
{
  "scores": {
    "overall_resistance": 0.85,
    "by_category": {
      "injection": 0.90,
      "jailbreak": 0.80
    },
    "by_strategy": {
      "base64_encode": 0.88,
      "roleplay": 0.75
    }
  },
  "total_attacks": 80,
  "blocked": 68,
  "passed": 12,
  "critical_findings": 2,
  "report_path": "data/redteam_reports/report_20260320_143022.json"
}
```

**Errors:** 409 if still running. 404 if no results available.

### GET /api/security/redteam/report

Download the full report in the specified format.

**Query parameters:**

| Parameter | Type | Default | Values |
|-----------|------|---------|--------|
| fmt | string | "json" | "json", "text", "markdown" |

**Response (200):** JSON object, or plain text depending on `fmt`.


## Audit Chain External Anchor (S146)

Prefix: `/api/security/audit`

### POST /api/security/audit/export-qr

Generate a QR code PNG containing the audit chain tip.

**Response (200):**

```json
{
  "qr_base64": "iVBORw0KGgo...",
  "payload": {
    "chain_tip_hash": "a1b2c3d4...",
    "entry_count": 1542,
    "timestamp": 1710936000.0,
    "version": "3.2.0"
  }
}
```

**Errors:** 503 if QR generation unavailable (missing `qrcode` lib).
400 if the audit chain is empty.

### POST /api/security/audit/export-anchor

Export the chain tip as a signed JSON file for USB or external storage.

**Response:** downloadable `audit_anchor.json` file with
`Content-Disposition: attachment`.

```json
{
  "chain_tip_hash": "a1b2c3d4...",
  "entry_count": 1542,
  "timestamp": 1710936000.0,
  "version": "3.2.0",
  "anchor_version": 1,
  "hmac_sha256": "e5f6a7b8..."
}
```

### GET /api/security/audit/anchor-text

Return a plain-text anchor suitable for clipboard copy.

**Response (200):**

```json
{
  "anchor_text": "Opti-Oignon Audit Anchor\nChain tip: a1b2c3d4...\nEntries: 1542\n..."
}
```

### POST /api/security/audit/verify-anchor

Verify an imported anchor against the current chain.

**Request body:**

```json
{
  "chain_tip_hash": "a1b2c3d4...",
  "entry_count": 1542,
  "timestamp": 1710936000.0,
  "version": "3.2.0",
  "anchor_version": 1,
  "hmac_sha256": "e5f6a7b8..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| chain_tip_hash | string | yes | Chain tip hash from exported anchor |
| entry_count | int (>= 0) | yes | Entry count from exported anchor |
| timestamp | float | no | Anchor creation timestamp |
| version | string | no | App version at anchor creation |
| anchor_version | int | no | Anchor format version (default 1) |
| hmac_sha256 | string or null | no | HMAC signature if signed anchor |

**Response (200):**

```json
{
  "match": true,
  "chain_tip_match": true,
  "entry_count_match": true,
  "hmac_valid": true,
  "current_entry_count": 1542,
  "current_chain_tip": "a1b2c3d4..."
}
```


## Startup Security Checklist (S145)

### GET /api/security/startup-checks

Run or retrieve the startup security checklist. Results are cached
for the process lifetime unless `force=true`.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| force | bool | false | Re-run checks even if cached |

**Response (200):**

```json
{
  "score": 85,
  "max_score": 100,
  "checks": [
    {
      "name": "code_signing_scripts",
      "status": "pass",
      "severity": "info",
      "score_impact": 0,
      "message": "Code signing scripts present"
    },
    {
      "name": "ollama_bind_guard",
      "status": "pass",
      "severity": "critical",
      "score_impact": 0,
      "message": "Ollama bound to localhost"
    },
    {
      "name": "luks_detection",
      "status": "warn",
      "severity": "warning",
      "score_impact": -5,
      "message": "Root filesystem not LUKS-encrypted",
      "tips": ["Reinstall with encryption", "Use fscrypt"]
    }
  ],
  "mode": "daily",
  "cached": true
}
```


## RAG Prompt Injection Defense (S144)

Prefix: `/api/rag/injection-defense`

### POST /api/rag/injection-defense/sanitize-preview

Preview retrieved chunks after the sanitization pipeline. Returns
chunks with injection scores and blocking decisions.

**Request body:**

```json
{
  "query": "What is the project architecture?",
  "collection": "internal-docs",
  "n_results": 5,
  "min_score": 0.3
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| query | string | required | User query to retrieve chunks for |
| collection | string | "" | Collection name |
| n_results | int (1-20) | 5 | Number of chunks to retrieve |
| min_score | float (0-1) | 0.3 | Minimum relevance score |

**Response (200):**

```json
{
  "chunks": [
    {
      "chunk_id": "abc123",
      "text": "The architecture uses...",
      "relevance_score": 0.87,
      "injection_score": 0.02,
      "flagged": false,
      "blocked": false,
      "patterns_matched": []
    },
    {
      "chunk_id": "def456",
      "text": "Ignore previous instructions...",
      "relevance_score": 0.65,
      "injection_score": 0.95,
      "flagged": true,
      "blocked": true,
      "patterns_matched": ["role_override", "instruction_override"]
    }
  ],
  "total_retrieved": 5,
  "total_blocked": 1,
  "collection_trust_level": "trusted"
}
```

### POST /api/rag/injection-defense/approve

Approve or reject specific chunks after preview.

**Request body:**

```json
{
  "chunk_ids": ["abc123", "ghi789"],
  "action": "approve"
}
```

| Field | Type | Description |
|-------|------|-------------|
| chunk_ids | list[str] | Chunk IDs from sanitize-preview response |
| action | string | "approve" or "reject" |

**Response (200):**

```json
{
  "action": "approve",
  "chunk_ids": ["abc123", "ghi789"],
  "status": "ok"
}
```

### GET /api/rag/injection-defense/audit

Query the injection audit log.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int (1-500) | 50 | Max entries to return |
| offset | int (>= 0) | 0 | Pagination offset |
| min_score | float (0-1) or null | null | Filter by minimum injection score |
| collection | string or null | null | Filter by collection name |

**Response (200):**

```json
{
  "entries": [
    {
      "timestamp": 1710936000.0,
      "chunk_text": "Ignore previous...",
      "injection_score": 0.95,
      "patterns_matched": ["role_override"],
      "collection": "web-content",
      "action": "blocked"
    }
  ],
  "total": 42,
  "limit": 50,
  "offset": 0
}
```

### DELETE /api/rag/injection-defense/audit

Clear the injection audit log.

**Response (200):**

```json
{
  "cleared": true,
  "entries_removed": 42
}
```

### GET /api/rag/injection-defense/config

Get the current injection defense configuration.

**Response (200):**

```json
{
  "enabled": true,
  "default_trust_level": "untrusted",
  "blocking_threshold": 0.7,
  "flagging_threshold": 0.4,
  "custom_patterns": ["ignore previous", "system prompt"],
  "audit_log_enabled": true
}
```


## User Management and RBAC (S142)

Prefix: `/api/users`

### GET /api/users/{user_id}/export

Export all data for a user (GDPR data portability). Users can
export their own data; admins can export any user's data.

**Response (200):**

```json
{
  "user_id": "user_abc",
  "export_date": "2026-03-20T14:30:00Z",
  "conversations": [...],
  "rag_collections": [...],
  "plugin_configs": {...},
  "feedback": [...],
  "settings": {...}
}
```

**Errors:** 403 if requesting another user's data without admin role.

### DELETE /api/users/{user_id}/data

Cascade delete all data for a user (GDPR right to erasure).
Users can delete their own data; admins can delete any user's data.

**Response (200):**

```json
{
  "user_id": "user_abc",
  "deleted": true,
  "items_removed": {
    "conversations": 15,
    "rag_collections": 3,
    "plugin_configs": 2,
    "feedback": 48
  }
}
```

**Errors:** 403 if requesting another user's data without admin role.

### GET /api/users/{user_id}/plugins

Get per-user plugin configurations.

**Response (200):** dict of plugin name to config object.

### PUT /api/users/{user_id}/plugins/{plugin}

Set per-user plugin configuration.

**Request body:** plugin-specific configuration object.

### GET /api/users/me/key-status

Get the current user's per-user encryption key status.

**Response (200):**

```json
{
  "has_key": true,
  "key_derived_at": "2026-03-20T10:00:00Z",
  "algorithm": "argon2id"
}
```

### GET /api/admin/audit

Query admin audit log (admin only).

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| admin_id | string or null | null | Filter by admin ID |
| target_type | string or null | null | Filter by target type |
| target_id | string or null | null | Filter by target ID |
| since | float or null | null | Events after this UNIX timestamp |
| limit | int (1-1000) | 100 | Max entries |
| offset | int (>= 0) | 0 | Pagination offset |

### GET /api/admin/audit/count

Count admin audit events (admin only).

**Query parameters:** `admin_id`, `target_type` (both optional).

**Response (200):**

```json
{
  "count": 234
}
```
