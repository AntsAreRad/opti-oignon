# Data Flow

## Request lifecycle

A typical chat request flows through these stages:

```mermaid
sequenceDiagram
    participant U as User (Browser)
    participant F as Frontend (SvelteKit)
    participant A as API (FastAPI)
    participant M as Auth Middleware
    participant P as Pipeline Manager
    participant R as Smart Router
    participant O as Ollama

    U->>F: Type message, press Ctrl+Enter
    F->>A: POST /api/chat or WebSocket
    A->>M: Verify JWT cookie
    M->>A: Authorized
    A->>P: Select pipeline type
    P->>R: Choose model
    R->>O: Inference request
    O->>R: Streaming response
    R->>P: Pipeline post-processing
    P->>A: Final response
    A->>F: WebSocket stream
    F->>U: Render markdown
```


## Pipeline selection

The pipeline manager analyzes each query to select the appropriate
pipeline:

1. **Query analysis** -- classify the query (code, reasoning, factual,
   creative, etc.)
2. **Pipeline matching** -- select from 9 pipeline types based on query
   type and configuration
3. **Model selection** -- smart router picks the best model considering
   capability profiles, context window, and health
4. **Execution** -- run the pipeline with the selected model
5. **Post-processing** -- plugin hooks, response formatting


## RAG data flow

```mermaid
flowchart LR
    subgraph Ingestion
        D[Document] --> C[Chunker]
        C --> S[RAGSanitizer]
        S --> E[Embeddings]
        E --> V[(ChromaDB)]
    end

    subgraph Query
        Q[User Query] --> QE[Query Embedding]
        QE --> V
        V --> R[Retrieved Chunks]
        R --> A[RAGAugmenter]
        A --> P[Prompt Assembly]
    end
```

During ingestion, documents are chunked, sanitized (injection markers
stripped), embedded via Ollama, and stored in ChromaDB. Parallel
ingestion uses a thread pool for concurrent processing.

During query, the user's question is embedded, similar chunks are
retrieved, sanitized again by the augmenter, and injected into the
prompt before inference.


## Plugin data flow

Plugins run in subprocesses and communicate via IPC:

```mermaid
flowchart LR
    M[Main Process] -->|HMAC socket| S1[Plugin Subprocess 1]
    M -->|stdin/stdout pipes| S2[Plugin Subprocess 2]
    M -->|Hook dispatch| H[Hook Chain]
    H --> S1
    S1 --> H
    H --> S2
    S2 --> H
    H --> M
```

Hook data flows through a chain: each plugin receives the output of
the previous one. Errors are isolated -- a failing plugin does not
break the chain.


## Coding agent data flow

```mermaid
flowchart TD
    U[User Request] --> P[Planner]
    P --> G[Code Generator]
    G --> S[Sandbox]
    S --> T{Tests Pass?}
    T -->|No| F[Auto-Fix]
    F --> G
    T -->|Yes| D[Diff]
    D --> A{Human Approval?}
    A -->|Yes| W[Write to Filesystem]
    A -->|No| X[Discard]
```

The coding agent operates entirely within the sandbox until the apply
phase. Working memory persists context across steps. Cascading
auto-escalates to larger models on repeated failures.


## Data storage

| Data | Storage | Encryption |
|------|---------|------------|
| Conversations | SQLite (per-user) | SQLCipher when available |
| User accounts | SQLite | bcrypt passwords, SQLCipher DB |
| RAG vectors | ChromaDB | Collection-level isolation |
| RAG metadata | SQLite | SQLCipher when available |
| Audit chain | SQLite | SQLCipher + ML-DSA-65 signatures |
| Plugin config | SQLite | Per-user, SQLCipher |
| Benchmarks | SQLite | SQLCipher when available |
| Configuration | YAML files | Filesystem permissions |

All databases use WAL mode for concurrent read performance and are
accessed through module-level singletons with `reset_*()` functions
for test isolation.
