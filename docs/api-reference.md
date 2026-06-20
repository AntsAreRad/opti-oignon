# API Reference

## Overview

Opti-Oignon exposes a REST API via FastAPI with ~519 endpoints. All
endpoints require JWT cookie authentication unless noted otherwise.
Admin-only endpoints require `role: admin`.

The full interactive API documentation is available at
[http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI) and
[http://localhost:8000/redoc](http://localhost:8000/redoc) (ReDoc) when
the backend is running.


## Endpoint groups

| Prefix | Description | Auth |
|--------|-------------|------|
| `/api/health` | Health check and version info | Public |
| `/api/auth/*` | Login, logout, registration, 2FA | Mixed |
| `/api/chat/*` | Conversations, messages, branches | User |
| `/api/models/*` | Model listing, profiles, health | User |
| `/api/rag/*` | Ingest, query, collections, streaming | User |
| `/api/plugins/*` | Install, configure, marketplace | User |
| `/api/sandbox/*` | Tool execution, approval | User |
| `/api/security/*` | Security settings, red team, audit | Admin |
| `/api/benchmark/*` | Performance benchmark, history | User |
| `/api/config/*` | System configuration | Admin |
| `/api/backup/*` | Export and import | Admin |
| `/api/shortcuts/*` | Keyboard shortcut bindings | User |
| `/api/theme/*` | Theme engine, accent colors | User |


## Common patterns

### Authentication

```bash
# Login
curl -c cookies.txt -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "secret"}'

# Authenticated request
curl -b cookies.txt http://localhost:8000/api/models
```

### Error responses

All error responses follow a consistent format:

```json
{
  "detail": "Human-readable error message"
}
```

HTTP status codes follow standard semantics: 400 for bad requests,
401 for unauthenticated, 403 for unauthorized, 404 for not found,
409 for conflicts, 503 for unavailable features.

### Streaming

WebSocket endpoints are used for real-time chat streaming. The RAG
query stream endpoint (`/api/rag/query/stream`) uses chunked transfer
encoding with UTF-8 safe chunk boundaries.

### Pagination

List endpoints support `offset` and `limit` query parameters for
pagination. Default limit is typically 20.


## Red Team API (S147-S148)

Prefix: `/api/security/redteam`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/run` | Launch a red team audit campaign |
| GET | `/status` | Get current campaign progress |
| GET | `/results` | Get latest campaign results |
| GET | `/reports` | List all reports |
| GET | `/reports/{id}` | Get specific report |
| GET | `/compare` | Compare two reports |
| POST | `/suggestions/{id}/accept` | Accept a suggestion |
| POST | `/suggestions/{id}/reject` | Reject a suggestion |


## Security scheduler API (S158)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/security/scheduler/status` | Scheduler status |
| POST | `/api/security/scheduler/trigger` | Manual trigger |


## Streaming API (S159)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/rag/query/stream` | Chunked RAG query streaming |
| GET | `/api/benchmark/stream` | Streaming benchmark results |


## Full endpoint reference

For complete request/response schemas, see the interactive Swagger UI
at `http://localhost:8000/docs` when the backend is running. The OpenAPI
JSON schema is available at `http://localhost:8000/openapi.json`.
