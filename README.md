# 🧅 Opti-Oignon

> **Local LLM Optimization Suite** - Intelligent routing, RAG, and multi-agent orchestration for Ollama models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-green.svg)](https://ollama.ai/)
[![Version](https://img.shields.io/badge/version-1.2.0-brightgreen.svg)](CHANGELOG.md)

---

## Overview

**Opti-Oignon** is a comprehensive optimization framework for local LLMs running on Ollama. It maximizes the performance of your local models through intelligent task routing based on a custom benchmark, RAG (Retrieval-Augmented Generation), and multi-agent orchestration.

### Key Features

- **Intelligent Routing** - Automatically selects the best model for each task type
- **RAG System** - Enrich prompts with context from your personal documents
- **Multi-Agent Pipelines** - Orchestrate multiple models for complex workflows
- **Pipeline Manager** - Create and manage custom pipelines via visual UI
- **Context Manager** - Dynamic context monitoring with model-specific limits
- **Benchmarking** - Evaluate models and auto-generate routing configuration
- **Dark Mode UI** - Modern Gradio interface with keyboard shortcuts
- **Multilingual** - Interface in English, responses match user's language

---

## What's New in v1.2

### Pipeline Manager (v1.2.0+)
Create, modify, and manage custom multi-agent pipelines directly from the UI:
- Visual step editor with drag-and-drop reordering
- LLM-powered prompt generation for pipeline steps
- Import/Export pipelines in YAML format
- Automatic pipeline detection via weighted keywords

### Context Manager (v1.1.0+)
Intelligent context monitoring for optimal model utilization:
- Dynamic context limits fetched from `ollama show`
- Real-time token estimation and usage display
- Smart truncation with preserved document structure
- Visual indicators (🟢🟡🔴) for context status

### Performance Improvements
- Keepalive mechanism prevents Gradio timeouts during model loading
- Threaded execution for responsive UI
- Model override per pipeline step

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

Supported formats: PDF, DOCX, CSV, Excel, Markdown, Python, R

### Multi-Agent Pipelines

Orchestrate multiple models for complex tasks:

- **Code Review Pipeline**: Planner → Coder → Reviewer
- **Research Pipeline**: Search → Analyze → Synthesize
- **Debug Pipeline**: Analyze → Fix → Explain

### Pipeline Manager

Create custom pipelines via the Pipelines tab:

1. **Define steps** with visual editor (up to 10 steps)
2. **Assign agents** (coder, reviewer, planner, explainer)
3. **Generate prompts** using LLM assistance
4. **Set keywords** for automatic detection
5. **Configure weights** for pipeline priority

### Context Manager

Monitor your context usage in real-time:

```
Context: qwen3-coder:30b (262K)
Input: ~15,000 / 253,952 tokens
[████░░░░░░░░░░░░░░░░] 6%
```

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

Edit `routing/config.yaml` to customize:

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
│   ├── context_manager.py # Context monitoring (v1.1+)
│   ├── pipeline_manager.py # Pipeline CRUD (v1.2+)
│   ├── dynamic_pipeline_ui.py # Dynamic planning (v1.2+)
│   │
│   ├── config/            # Configuration files
│   │   ├── models.yaml
│   │   ├── presets.yaml
│   │   └── user_profile.yaml
│   │
│   ├── data/              # User data
│   │   └── pipelines_custom.yaml
│   │
│   ├── routing/           # Intelligent routing
│   │   └── benchmark.py
│   │
│   ├── agents/            # Multi-agent system
│   │   ├── orchestrator.py
│   │   ├── base.py
│   │   ├── dynamic_pipeline.py
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
│   ├── CONFIGURATION.md
│   ├── CONTEXT_MANAGER.md
│   └── PIPELINE_MANAGER.md
│
├── examples/              # Usage examples
│   ├── basic_usage.py
│   ├── rag_example.py
│   └── multi_agent_example.py
│
└── routing/               # Routing configuration
    ├── config.yaml
    └── benchmarks/
```

---

## UI Overview

The Gradio interface provides:

### Chat Tab
- Real-time task detection and model routing
- Context usage indicator
- Preset and pipeline selection
- RAG toggle and status
- Document upload support

### Pipelines Tab
- Visual pipeline editor
- Step-by-step creation (up to 10 steps)
- LLM-powered prompt generation
- Import/Export functionality

### History Tab
- Searchable conversation history
- Export to Markdown
- Multi-agent execution metadata

### Settings Tab
- Model configuration
- RAG settings
- Preset management
- Benchmark controls

---

## API Usage

```python
from opti_oignon import analyzer, router, executor

# Analyze a query
analysis = analyzer.analyze("Write a function to calculate Shannon index in R")
print(f"Task: {analysis.task_type}, Language: {analysis.language}")

# Get optimal model
routing = router.route(analysis)
print(f"Model: {routing.model}")

# Execute query
response = executor.execute("Your prompt here")
print(response)
```

### Pipeline Manager API

```python
from opti_oignon import get_pipeline_manager, Pipeline, PipelineStep

pm = get_pipeline_manager()

# List all pipelines
for p in pm.list_all():
    print(f"{p.emoji} {p.id}: {p.name}")

# Create custom pipeline
new_pipeline = Pipeline(
    id="my_pipeline",
    name="My Custom Pipeline",
    steps=[
        PipelineStep(name="Analyze", agent="reviewer"),
        PipelineStep(name="Implement", agent="coder"),
    ],
    keywords=["custom", "workflow"],
)
pm.create(new_pipeline)
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

### Recent Highlights
- **v1.2.0** - Pipeline Manager, keepalive executor, model override per step
- **v1.1.0** - Context Manager with dynamic limits

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
