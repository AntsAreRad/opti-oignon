# Installation Guide

Complete guide to install and configure Opti-Oignon on your system.

## Prerequisites

### Required

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime environment |
| Ollama | Latest | Local LLM inference |
| RAM | 16GB+ | For 30B+ parameter models |

### Recommended

- **SSD storage** for faster model loading
- **NVIDIA GPU** with CUDA for accelerated inference
- **Linux/macOS** (Windows works but may require adjustments)

## Step 1: Install Ollama

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### macOS
```bash
brew install ollama
```

### Windows
Download from [ollama.ai](https://ollama.ai/download)

### Verify installation
```bash
ollama --version
ollama serve  # Start the server (keep running in background)
```

## Step 2: Download Models

Pull the recommended models for optimal performance:

```bash
# Essential models
ollama pull qwen3-coder:30b      # Best for coding tasks
ollama pull nemotron-3-nano:30b  # Fast general queries
ollama pull deepseek-r1:32b      # Complex reasoning

# Embeddings (required for RAG)
ollama pull mxbai-embed-large

# Optional: Additional models
ollama pull qwen3:32b            # General purpose
ollama pull devstral-small-2     # Code fallback
```

### Verify models
```bash
ollama list
```

## Step 3: Install Opti-Oignon

### Option A: From GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/AntsAreRad/opti-oignon.git
cd opti-oignon

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e .
```

### Option B: Direct pip install

```bash
pip install git+https://github.com/AntsAreRad/opti-oignon.git
```

## Step 4: Verify Installation

```bash
# Check CLI is available
opti-oignon --help

# Or run as module
python -m opti_oignon --help
```

Expected output:
```
🧅 Opti-Oignon - Local LLM Optimization Suite

Commands:
  ui         Launch the Gradio web interface
  benchmark  Run model benchmarks
  rag        RAG system commands
  chat       Quick chat from terminal
```

## Step 5: First Launch

```bash
# Start the web interface
opti-oignon ui

# Or with custom port
opti-oignon ui --port 7861
```

Open your browser at `http://localhost:7860`

## Configuration

### Model Configuration

Edit `opti_oignon/config/models.yaml`:

```yaml
models:
  default: "nemotron-3-nano:30b"
  code: "qwen3-coder:30b"
  reasoning: "deepseek-r1:32b"
  
ollama:
  host: "http://localhost:11434"
  timeout: 300
```

### User Profile

Edit `opti_oignon/config/user_profile.yaml`:

```yaml
user:
  name: "Your Name"
  expertise: "intermediate"  # beginner, intermediate, expert
  preferred_language: "en"   # en, fr, es, etc.
  
preferences:
  verbose_explanations: true
  include_examples: true
  code_comments: true
```

## Troubleshooting

### Ollama connection error

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### Model not found

```bash
# List available models
ollama list

# Pull missing model
ollama pull model-name
```

### Memory issues

For systems with limited RAM:
1. Use smaller models (7B-14B instead of 30B+)
2. Edit `config/models.yaml` to use lighter alternatives
3. Close other applications before running large models

### Permission denied

```bash
# Make sure you own the installation directory
sudo chown -R $USER:$USER ~/.local/lib/python3.*/site-packages/opti_oignon/
```

## Updating

```bash
cd opti-oignon
git pull origin main
pip install -e . --upgrade
```

## Uninstalling

```bash
pip uninstall opti-oignon
rm -rf opti-oignon/  # Remove local files
```

---

[← Back to README](../README.md) | [Benchmark Guide →](BENCHMARK.md)
