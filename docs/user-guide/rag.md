# RAG (Retrieval-Augmented Generation)

## Overview

Opti-Oignon's RAG system lets you ingest documents into project-specific
collections and query them during conversations. Documents are chunked,
embedded locally (via Ollama), and stored in ChromaDB vector collections.

All processing stays local -- no data leaves your machine.


## Creating a project

1. Open **Settings > Advanced > RAG**
2. Click **New Project**
3. Name your project and optionally set a default collection

Each project maintains isolated collections. In multi-user mode,
projects are scoped to individual users with RBAC enforcement.


## Ingesting documents

### From the UI

1. Open a project in the RAG settings panel
2. Click **Ingest** and select files (PDF, text, markdown, code files)
3. Progress is shown per-chunk with error isolation

### From the CLI

```bash
oo rag ingest paper.pdf --collection ecology
oo rag ingest data/ --collection ecology   # directory of files
```

### From the API

```
POST /api/rag/ingest
Content-Type: multipart/form-data

file: <uploaded file>
collection: ecology
```

### Parallel ingestion

Large document sets are processed in parallel using a thread pool
(default 4 workers). Each chunk is embedded independently, so a
failure in one chunk does not block the others.

Batch embedding groups chunks for efficient Ollama calls with
auto-flush and progress callbacks.


## Querying

### From the UI

When a RAG project is active, your chat messages are automatically
augmented with relevant document context. The RAG augmenter retrieves
the top-k chunks by similarity and injects them into the prompt.

### From the CLI

```bash
oo rag query "What is BCI?" --collection ecology --n-results 5
```

### From the API

```
POST /api/rag/query
{
  "question": "What is BCI?",
  "collection": "ecology",
  "n_results": 5
}
```

Streaming queries are available at `/api/rag/query/stream` using
chunked transfer encoding with UTF-8 safe boundaries.


## Security

RAG is protected by multiple defense layers:

- **RAGSanitizer** -- strips prompt injection markers from ingested
  content (10 injection patterns detected)
- **RAGAugmenter** -- validates retrieved chunks before injection into
  the prompt
- **SearchResultSanitizer** -- cleans web search results used for
  augmentation
- **PIISanitizer** -- optionally redacts personally identifiable
  information from ingested documents
- **Per-user isolation** -- in multi-user mode, each user's collections
  are isolated at the database level

See the [Security Guide](../security/overview.md) for details on the
prompt injection defense system.


## Connection pooling

RAG databases use a connection pool with health checks and WAL mode
for concurrent read performance. The pool is shared across the
application with a singleton registry pattern. When the pool is
unavailable, a fallback mode opens per-call connections.
