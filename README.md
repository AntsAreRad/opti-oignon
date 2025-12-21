# 🧅 Opti-Oignon

> **Local LLM Optimization Suite** - Intelligent routing, RAG, and multi-agent orchestration for Ollama models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-green.svg)](https://ollama.ai/)

---

## Overview

**Opti-Oignon** is a comprehensive optimization framework for local LLMs running on Ollama. It maximizes the performance of your local models through intelligent task routing based on a custom benchmark, RAG (Retrieval-Augmented Generation), and multi-agent orchestration.

### Key Features

- **Intelligent Routing** - Automatically selects the best model for each task type
- **RAG System** - Enrich prompts with context from your personal documents
- **Multi-Agent Pipelines** - Orchestrate multiple models for complex workflows
- **Benchmarking** - Evaluate models and auto-generate routing configuration
- **Dark Mode UI** - Modern Gradio interface with keyboard shortcuts
- **Multilingual** - Interface in English, responses match user's language

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** running locally with models installed
- **16GB+ RAM** recommended for 30B+ models

### Installation

```bash
# Clone the repository
git clone https://github.com/AntsAreRad/opti-oignon.git
cd opti-oignon

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install the package
pip install -e .

```

### First Launch

```bash
# Start the Gradio UI
python -m opti_oignon

# Or directly
opti-oignon ui
```

The interface will open at `http://localhost:7860`

---

## Benchmarking

Run benchmarks to evaluate your models and generate optimal routing configuration:

```bash
# Estimate benchmark time (no execution)
opti-oignon benchmark --estimate

# Run quick benchmark (3 models)
opti-oignon benchmark --quick --confirm

# Full benchmark with all models
opti-oignon benchmark --confirm

# Interactive mode with manual scoring
opti-oignon benchmark --interactive --confirm

```

### Benchmark Results

Results are saved to `routing/benchmarks/`:
- `benchmark_YYYY-MM-DD_HH-MM.json` - Detailed results
- `benchmark_YYYY-MM-DD_HH-MM.md` - Human-readable report
- `benchmark_latest.json` - Latest results for comparison

---

## Configuration

### Model Configuration

Edit `opti_oignon/routing/config.yaml` to customize:

```yaml
# Task routing configuration
task_routing:
  code_r:
    primary: "qwen3-coder:30b"
    fallback: ["devstral-small-2:latest"]
    fast: "qwen2.5-coder:14b"
    temperature: 0.3
    timeout: 120
```

### MVP Models (Recommended)

Based on extensive benchmarking:

| Task | Model | Score | Speed |
|------|-------|-------|-------|
| Code (R/Python) | `qwen3-coder:30b` | 9/10 | ~30s |
| Reasoning | `deepseek-r1:32b` | 8/10 | ~180s |
| Fast responses | `nemotron-3-nano:30b` | 8/10 | ~70s |
| Embeddings | `mxbai-embed-large` | - | - |

---

## Project Structure

```
opti-oignon/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── setup.py
├── requirements.txt
├── .gitignore
│
├── opti_oignon/           # Main package
│   ├── __init__.py
│   ├── __main__.py        # Entry point
│   ├── main.py            # CLI
│   ├── ui.py              # Gradio interface
│   ├── config.py          # Configuration loader
│   ├── analyzer.py        # Task detection
│   ├── router.py          # Model routing
│   ├── executor.py        # Query execution
│   ├── presets.py         # Quick presets
│   ├── history.py         # Conversation history
│   │
│   ├── config/            # Configuration files
│   │   ├── models.yaml
│   │   ├── presets.yaml
│   │   └── user_profile.yaml
│   │
│   ├── routing/           # Intelligent routing
│   │   └── benchmark.py
│   │
│   ├── agents/            # Multi-agent system
│   │   ├── orchestrator.py
│   │   ├── base.py
│   │   └── specialists/
│   │
│   └── rag/               # RAG system
│       ├── indexer.py
│       ├── retriever.py
│       ├── chunkers.py
│       └── augmenter.py
│
├── docs/                  # Documentation
│   ├── INSTALLATION.md
│   ├── BENCHMARK.md
│   ├── ARCHITECTURE.md
│   └── CONFIGURATION.md
│
├── examples/              # Usage examples
│   ├── basic_usage.py
│   ├── rag_example.py
│   └── multi_agent_example.py
│
└── .github/               # GitHub templates
    └── ISSUE_TEMPLATE/
```

---

## Features in Detail

### Intelligent Routing

The router analyzes your query to determine:
1. **Task type** (code, debug, explanation, etc.)
2. **Language** (R, Python, Bash, etc.)
3. **Complexity** (simple, medium, complex)

Then selects the optimal model based on benchmark data.

### RAG System

Index your personal documents for context-aware responses:

```bash
# Index a folder
opti-oignon rag index ./docs --recursive

# Search indexed content
 opti-oignon rag search "Shannon diversity"
```

### Multi-Agent Pipelines

Orchestrate multiple models for complex tasks:

- **Code Review Pipeline**: Planner → Coder → Reviewer
- **Research Pipeline**: Search → Analyze → Synthesize
- **Debug Pipeline**: Analyze → Fix → Explain

---

## UI Screenshots

The Gradio interface provides:
- Dark mode theme
- Real-time task detection
- Model selection override
- RAG toggle and status
- Conversation history
- Export to Markdown

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [Gradio](https://gradio.app/) for the web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage

---

## Contact

**Léon Brouillé** - M2 IMABEE (Ecology)

Project Link: [https://github.com/AntsAreRad/opti-oignon](https://github.com/AntsAreRad/opti-oignon)
