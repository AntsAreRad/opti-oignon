# Ollama Setup

Opti-Oignon uses [Ollama](https://ollama.com) as its primary inference
backend. This page covers installation, model selection, and configuration.


## Installing Ollama

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS

Download from [ollama.com/download](https://ollama.com/download) or use
Homebrew:

```bash
brew install ollama
```

### Verify installation

```bash
ollama --version
ollama serve   # Start the Ollama server
```

The Ollama API should be available at `http://localhost:11434`.


## Pulling models

Opti-Oignon works with any Ollama-compatible model. Pull at least one
model before starting:

```bash
# Small and fast (recommended for Minimal preset)
ollama pull llama3.2:3b

# Medium balanced model
ollama pull llama3.1:8b

# Larger model for Power preset
ollama pull llama3.1:70b

# Code-focused model
ollama pull codellama:13b
```

Check your available models:

```bash
ollama list
```


## Model recommendations

| Preset | Recommended models | Min RAM |
|--------|--------------------|---------|
| Minimal | 1 small model (3B-7B) | 8 GB |
| Balanced | 2 models (7B-13B) | 16 GB |
| Power | 3+ models including 70B | 32+ GB |

Opti-Oignon's smart router automatically selects the best model for each
query based on capability profiles, context window size, and model health.
Having multiple models of different sizes enables cascading inference
(try a small fast model first, escalate to larger if quality is low).


## Configuration

Ollama's default bind address is `127.0.0.1:11434`. If you need to change
it, set the `OLLAMA_HOST` environment variable before starting Ollama:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama serve
```

!!! warning "Security note"
    Opti-Oignon's **Bulbe mode** enforces that Ollama binds to localhost
    only. The network bind guard checks this at startup and blocks
    external-facing Ollama configurations. See
    [Bulbe Mode](../security/bulbe-mode.md).

Opti-Oignon detects your Ollama models automatically via the
`/api/tags` endpoint. No manual model configuration is needed in most
cases.


## llama.cpp backend

Opti-Oignon also supports direct llama.cpp inference as an alternative
backend. This is configured in `config/inference.yaml` and is useful for
custom quantizations or models not available through Ollama.

The backend is selected per-model in the routing configuration. Both
backends can run simultaneously.


## Troubleshooting

**Ollama not starting:** Check if another process is using port 11434.
Kill it with `lsof -i :11434` and retry.

**Model download stuck:** Ollama downloads can be large (several GB).
Check your disk space and internet connection. Resume interrupted
downloads by running the `pull` command again.

**GPU not detected:** Ollama uses GPU acceleration automatically when
available (NVIDIA CUDA, Apple Metal). Check `ollama run <model>` output
for GPU info. For NVIDIA, ensure CUDA drivers are installed.
