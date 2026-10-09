# API Reference

## Overview

Opti-Oignon exposes a REST API via FastAPI with 540 endpoints. All
endpoints require JWT cookie authentication unless noted otherwise.
Admin-only endpoints require `role: admin`.

The full interactive API documentation is available at
[http://localhost:8001/docs](http://localhost:8001/docs) (Swagger UI) and
[http://localhost:8001/redoc](http://localhost:8001/redoc) (ReDoc) when
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
| `/api/allium/*` | The componion: its served status, read only | User |
| `/api/pending-writes/*` | Memory and notes writes waiting for the user: list, accept or decline a batch | User |


## Common patterns

### Authentication

```bash
# Login
curl -c cookies.txt -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "secret"}'

# Authenticated request
curl -b cookies.txt http://localhost:8001/api/models
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

#### Chat request: attached files

A chat request on `/api/chat/stream` may carry `documents`, a list of
`{"filename": ..., "content": ...}` objects: the files attached to the
turn, sent beside the typed `message` and never inside it. Each is saved
as a document part of the turn, under a line naming it. `config/chat.yaml`
bounds them: files per turn, bytes of text per file, and characters per
name, which must be printable. A request over a bound is answered with an
`error` frame that names the file and the bound, and nothing runs or is
saved. A turn may carry files and no typed words. The request has no field
that makes words typed: the server says who wrote a turn, and `pasted`
(below) can only mark words as not typed.

#### Chat request: pasted text

A chat request may carry `pasted`, the ranges of `message` the user pasted
or dropped rather than typed: `[[start, end], ...]`, whole numbers in code
points of the message as sent, sorted, apart and non-empty; touching ranges
are merged. Each is saved as a document part of the turn's words, and a
turn with a paste in its words has no typed part: its words endorse no
memory or notes write, no tool call's argument and no decision. Ranges out
of shape, or more of them than `config/chat.yaml` allows
(`pasted: max_ranges`), are answered with an `error` frame naming the field
and the rule, and nothing runs or is saved. A request without `pasted` is
read as before.

#### Chat stream frames

The chat WebSockets (`/api/chat/stream`, and `/api/chat/retry`, which
streams a regenerated reply the same way) send JSON frames of the form
`{"type": ..., "content": ..., "metadata": {...}}`, the metadata only when
there is some. The types are `metadata`, `token`, `thinking`, `status`,
`tool_call`, `tool_call_pending`, `tool_call_resolved`, `reasoning_step`,
`reasoning_done`, `consensus_model_done`, `consensus_done`,
`correction_step`, `correction_done`, `vision_delegation`, `verification`,
`pipeline_step`, `ping`, `error` and `done`; the coding path relays its
agent's own `coding_*` frames. A reply normally ends with `done`; an error
ends it with `error` and no `done`.

`pipeline_step` reports one step of a run the server executes: an execution
pipeline, a reasoning strategy, a consensus or a forced self-correction. Its
`content` is empty and its `metadata` always carries every field below,
`null` where it does not apply.

| Field | Meaning |
|---|---|
| `v` | Schema version, `1` |
| `seq` | Counter per reply, from 1, strictly increasing: it orders the frames and removes duplicates |
| `run` | Opaque id of the run within the reply |
| `kind` | `exec_pipeline`, `reasoning`, `consensus` or `self_correct` |
| `name` | The pipeline's name; `decompose`, `tree_of_thought` or `self_consistency` for reasoning; the kind otherwise |
| `pipeline_id`, `step_type` | The pipeline's id and the step's declared type (pipeline steps only) |
| `parent` | `{run, index}` of the step that started this run, when runs nest |
| `index`, `total` | The step's position from 0, and the run's step count once known (a decomposition knows it only when its plan arrives; after that it never changes) |
| `label` | The step's name, at most 120 characters; a decomposition's sub-question titles come from the model and are data |
| `state` | `pending`, `running`, `done`, `failed`, `skipped`, `cancelled` or `not_run` |
| `progress` | `{done, total, unit}` on `running` only, and only when those units are the step's whole work: `model` for a consensus query, `sample` or `sub_step` for the step that started a nested run |
| `reason` | At most 300 characters: the error of a `failed` step, or why a step was `cancelled` or `not_run` |
| `ran_as` | The agentic pipeline an execution step really ran as, which can differ from its declared type |
| `duration_ms` | Measured by the server on a monotonic clock, for a step that ran |

A run announces the steps it knows as `pending` before running them (a
decomposition announces its sub-questions when its plan arrives). Each step
then goes from `pending` to `running` to `done`, `failed` or `cancelled`, or
straight from `pending` to `skipped` (its condition was false) or `not_run`,
and reaches exactly one final state. `done` means finished with no error the
server could see; `failed` is an error of the step's own work (an exception,
or one of the server's own fixed error messages). A safety mechanism never
ends a step `failed`: Stop, Stop all and a resource refusal end it
`cancelled` or `not_run`, with the reason. On Stop and on an error the
server closes every open step before the stream says so, and the `done`
frame carries `steps`, the last `pipeline_step` of every step, whenever the
reply had one. `pipeline_step` is never dropped by backpressure.

### Pagination

List endpoints support `offset` and `limit` query parameters for
pagination. Default limit is typically 20.


## Red Team API

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


## Security scheduler API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/security/scheduler/status` | Scheduler status |
| POST | `/api/security/scheduler/trigger` | Manual trigger |


## Streaming API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/rag/query/stream` | Chunked RAG query streaming |
| GET | `/api/benchmark/stream` | Streaming benchmark results |


## Full endpoint reference

For complete request/response schemas, see the interactive Swagger UI
at `http://localhost:8001/docs` when the backend is running. The OpenAPI
JSON schema is available at `http://localhost:8001/openapi.json`.
