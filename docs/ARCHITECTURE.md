# Architecture

Technical overview of Opti-Oignon's modular architecture.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio UI (ui.py)                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  Chat   │ │ Agents  │ │   RAG   │ │ History │ │  Info   │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └─────────┘   │
└───────┼──────────┼──────────┼──────────┼────────────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
┌───────────────────────────────────────────────────────────────┐
│                      Core Pipeline                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Analyzer │ → │  Router  │ → │ Executor │ → │ History  │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
└───────────────────────────────────────────────────────────────┘
        │                │               │
        ▼                ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Presets   │  │  Benchmark  │  │ Orchestrator│
│ (presets.py)│  │(benchmark.py│  │(agents/)    │
└─────────────┘  └─────────────┘  └─────────────┘
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │ Specialists │
                                  │ Planner     │
                                  │ Coder       │
                                  │ Reviewer    │
                                  │ Explainer   │
                                  └─────────────┘
```

## Core Modules

### 1. Analyzer (`analyzer.py`)

Analyzes user queries to determine task characteristics.

```python
from opti_oignon.analyzer import TaskAnalyzer

analyzer = TaskAnalyzer()
result = analyzer.analyze("Write a Python function for sorting")

# Returns:
# {
#     "task_type": "code_python",
#     "language": "python",
#     "complexity": "medium",
#     "keywords": ["function", "sorting"],
#     "suggested_preset": "code"
# }
```

**Detection capabilities:**
- Programming language (Python, R, Bash, SQL, etc.)
- Task type (code, debug, explain, analyze, etc.)
- Complexity level (simple, medium, complex)
- Domain hints (bioinformatics, web, data science)

### 2. Router (`router.py`)

Selects the optimal model based on task analysis and benchmark data.

```python
from opti_oignon.router import ModelRouter

router = ModelRouter()
model = router.route(task_type="code_python", complexity="complex")

# Returns: "qwen3-coder:30b"
```

**Routing logic:**
1. Check task-specific primary model
2. Verify model availability in Ollama
3. Fall back to alternatives if needed
4. Apply speed/quality preferences

### 3. Executor (`executor.py`)

Executes queries against Ollama with prompt enhancement.

```python
from opti_oignon.executor import QueryExecutor

executor = QueryExecutor()
response = executor.execute(
    prompt="Write a sorting function",
    model="qwen3-coder:30b",
    system_prompt="You are an expert Python developer...",
    temperature=0.3
)
```

**Features:**
- Automatic prompt refinement
- System prompt injection
- Streaming support
- Timeout handling
- Error recovery

### 4. Presets (`presets.py`)

Quick configuration profiles for common use cases.

```python
from opti_oignon.presets import PresetManager

presets = PresetManager()
config = presets.get("bioinformatics_r")

# Returns:
# {
#     "model": "qwen3-coder:30b",
#     "system_prompt": "Expert in R and bioinformatics...",
#     "temperature": 0.3,
#     "keywords": ["vegan", "ggplot", "tidyverse"]
# }
```

**Preset detection:**
- Keyword matching with weights
- Auto-suggestion based on query content
- User-defined custom presets

### 5. History (`history.py`)

Conversation persistence and export.

```python
from opti_oignon.history import HistoryManager

history = HistoryManager()
history.add_exchange(prompt, response, metadata)
history.export_markdown("conversation.md")
```

**Storage:**
- Daily JSON files in `data/history/`
- Searchable index
- Markdown export
- Statistics tracking

## RAG System (`rag/`)

Retrieval-Augmented Generation for context-aware responses.

```
┌─────────────────────────────────────────────────┐
│                  RAG Pipeline                    │
│                                                  │
│  Documents → Chunker → Embeddings → ChromaDB    │
│                                                  │
│  Query → Retriever → Augmenter → Enhanced Prompt│
└─────────────────────────────────────────────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| `indexer.py` | Document ingestion and chunking |
| `chunkers.py` | Format-specific text splitting |
| `embeddings.py` | Vector generation via Ollama |
| `retriever.py` | Similarity search in ChromaDB |
| `augmenter.py` | Context injection into prompts |

### Supported Formats

- **Code**: `.py`, `.r`, `.js`, `.ts`, `.sh`
- **Documents**: `.pdf`, `.docx`, `.md`, `.txt`
- **Data**: `.csv`, `.xlsx`, `.json`, `.yaml`

### Chunking Strategies

```python
# Python: Function-aware chunking
def chunk_python(code):
    # Splits at function/class boundaries
    # Preserves docstrings and imports
    
# Markdown: Header-aware chunking
def chunk_markdown(text):
    # Splits at ## headers
    # Keeps context hierarchy
    
# CSV: Row-based chunking
def chunk_csv(data):
    # Groups rows with column headers
    # Handles large datasets
```

## Multi-Agent System (`agents/`)

Orchestrated pipelines for complex tasks.

```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                       │
│                                                      │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐           │
│  │ Planner │ → │  Coder  │ → │Reviewer │           │
│  └─────────┘   └─────────┘   └─────────┘           │
│       │             │             │                 │
│       ▼             ▼             ▼                 │
│   Plan Doc      Code Output   Review Report        │
└─────────────────────────────────────────────────────┘
```

### Available Pipelines

**Code Review Pipeline:**
1. Planner → Analyzes requirements
2. Coder → Implements solution
3. Reviewer → Checks quality, suggests improvements

**Research Pipeline:**
1. Planner → Breaks down question
2. Multiple Coders → Parallel research
3. Explainer → Synthesizes findings

### Agent Configuration

```yaml
# agents/config.yaml
agents:
  planner:
    model: "deepseek-r1:32b"
    temperature: 0.5
    role: "Break down complex tasks into steps"
    
  coder:
    model: "qwen3-coder:30b"
    temperature: 0.3
    role: "Implement code solutions"
    
  reviewer:
    model: "qwen3:32b"
    temperature: 0.4
    role: "Review and improve code quality"
```

## Configuration System

### Hierarchy

```
opti_oignon/config/
├── models.yaml        # Model definitions and defaults
├── presets.yaml       # System presets (read-only)
└── user_profile.yaml  # User preferences

opti_oignon/data/
└── user_presets.yaml  # Custom user presets
```

### Loading Priority

1. User overrides (`user_presets.yaml`)
2. User profile (`user_profile.yaml`)
3. System defaults (`presets.yaml`, `models.yaml`)

## Data Flow

### Standard Query

```
User Input
    │
    ▼
┌─────────────────┐
│    Analyzer     │ ← Detect task type, language, complexity
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preset Match   │ ← Auto-select or user-specified
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Router      │ ← Select optimal model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RAG Check     │ ← Retrieve relevant context (if enabled)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Executor     │ ← Build prompt, call Ollama
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    History      │ ← Save exchange
└────────┬────────┘
         │
         ▼
    Response
```

### Multi-Agent Query

```
User Input (complex task)
    │
    ▼
┌─────────────────┐
│  Orchestrator   │ ← Detect pipeline type
└────────┬────────┘
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Planner │   │  Coder  │   │Reviewer │
    └────┬────┘   └────┬────┘   └────┬────┘
         │              │              │
         ▼              ▼              ▼
    Plan Output → Code Output → Final Review
                                      │
                                      ▼
                              Aggregated Response
```

## Extension Points

### Custom Chunkers

```python
# rag/chunkers.py
def chunk_custom_format(content: str) -> List[str]:
    """Add your custom chunking logic."""
    chunks = []
    # Your implementation
    return chunks

CHUNKERS["custom"] = chunk_custom_format
```

### Custom Agents

```python
# agents/specialists/custom.py
from ..base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="custom",
            model="your-model",
            system_prompt="Your agent's role..."
        )
    
    def process(self, input_data: dict) -> dict:
        # Your processing logic
        return {"result": "..."}
```

### Custom Presets

```yaml
# data/user_presets.yaml
my_preset:
  name: "My Custom Preset"
  model: "qwen3-coder:30b"
  system_prompt: |
    You are an expert in my specific domain...
  temperature: 0.3
  keywords:
    - keyword1: 2.0  # weight
    - keyword2: 1.5
```

---

[← Benchmark](BENCHMARK.md) | [Configuration →](CONFIGURATION.md)
