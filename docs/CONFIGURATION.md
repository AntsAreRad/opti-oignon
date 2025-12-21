# Configuration Guide

Complete reference for all configuration options in Opti-Oignon.

## Configuration Files

| File | Purpose | User Editable |
|------|---------|---------------|
| `config/models.yaml` | Model definitions | Yes |
| `config/presets.yaml` | System presets | ⚠️ Careful |
| `config/user_profile.yaml` | Personal preferences | Yes |
| `data/user_presets.yaml` | Custom presets | Yes |
| `agents/config.yaml` | Multi-agent settings | Yes |
| `rag/config.py` | RAG parameters | ⚠️ Advanced |

## Models Configuration

### `config/models.yaml`

```yaml
# Ollama connection
ollama:
  host: "http://localhost:11434"
  timeout: 300  # seconds

# Default model for unclassified tasks
default_model: "nemotron-3-nano:30b"

# Task-specific routing
task_routing:
  code_python:
    primary: "qwen3-coder:30b"
    fallback: ["devstral-small-2:latest", "qwen2.5-coder:14b"]
    fast: "qwen2.5-coder:14b"
    temperature: 0.3
    timeout: 120
    
  code_r:
    primary: "qwen3-coder:30b"
    fallback: ["qwen3:32b"]
    temperature: 0.3
    timeout: 120
    
  debug:
    primary: "qwen3-coder:30b"
    fallback: ["deepseek-r1:32b"]
    temperature: 0.2
    timeout: 180
    
  explain:
    primary: "nemotron-3-nano:30b"
    fallback: ["qwen3:32b"]
    temperature: 0.5
    timeout: 90
    
  reasoning:
    primary: "deepseek-r1:32b"
    fallback: ["qwen3:32b"]
    temperature: 0.4
    timeout: 300
    
  general:
    primary: "nemotron-3-nano:30b"
    fallback: ["qwen3:32b"]
    temperature: 0.7
    timeout: 60

# Speed preference (affects model selection)
# Options: "quality", "balanced", "speed"
speed_preference: "balanced"

# Embeddings model for RAG
embeddings_model: "mxbai-embed-large"
```

### Parameter Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `primary` | string | First choice model |
| `fallback` | list | Backup models in order |
| `fast` | string | Quick response alternative |
| `temperature` | float | Creativity (0.0-1.0) |
| `timeout` | int | Max response time (seconds) |

## User Profile

### `config/user_profile.yaml`

```yaml
user:
  name: "Your Name"
  expertise_level: "intermediate"  # beginner, intermediate, expert
  preferred_language: "en"         # Response language

preferences:
  # Response style
  verbose_explanations: true
  include_examples: true
  code_comments: true
  markdown_formatting: true
  
  # Behavior
  auto_detect_language: true
  confirm_before_long_tasks: true
  save_history: true
  
  # UI
  theme: "dark"
  font_size: 14

domains:
  # Your areas of expertise (affects prompt enhancement)
  - "bioinformatics"
  - "ecology"
  - "data science"
  - "R programming"

context:
  # Persistent context added to all prompts
  background: |
    I work in ecological research, primarily using R for 
    data analysis with packages like vegan, tidyverse, and ggplot2.
```

## Presets Configuration

### System Presets (`config/presets.yaml`)

```yaml
presets:
  code:
    name: "Code Assistant"
    description: "General programming help"
    system_prompt: |
      You are an expert programmer. Write clean, efficient, 
      well-documented code. Include error handling and tests 
      when appropriate.
    temperature: 0.3
    keywords:
      - "code": 2.0
      - "function": 1.5
      - "script": 1.5
      - "program": 1.5
      
  debug:
    name: "Debug Helper"
    description: "Find and fix bugs"
    system_prompt: |
      You are a debugging expert. Analyze code systematically,
      identify root causes, and provide clear fixes with 
      explanations.
    temperature: 0.2
    keywords:
      - "error": 2.5
      - "bug": 2.5
      - "fix": 2.0
      - "doesn't work": 2.0
      - "traceback": 2.0
      
  explain:
    name: "Explainer"
    description: "Clear explanations"
    system_prompt: |
      Explain concepts clearly and progressively. Use analogies
      and examples. Adapt to the user's expertise level.
    temperature: 0.5
    keywords:
      - "explain": 2.0
      - "what is": 2.0
      - "how does": 2.0
      - "why": 1.5
```

### Custom Presets (`data/user_presets.yaml`)

```yaml
# Your personal presets
bioinformatics_r:
  name: "Bioinformatics (R)"
  description: "R code for ecological analysis"
  model: "qwen3-coder:30b"
  system_prompt: |
    You are an expert in R programming for bioinformatics and ecology.
    You are proficient with: vegan (diversity indices, ordination),
    tidyverse (data manipulation), ggplot2 (visualization),
    phyloseq (microbiome data), and BiocManager packages.
    
    Always provide reproducible code with clear comments.
    Include example data when demonstrating functions.
  temperature: 0.3
  keywords:
    - "diversity": 2.5
    - "vegan": 2.5
    - "ggplot": 2.0
    - "tidyverse": 2.0
    - "ordination": 2.0
    - "NMDS": 2.0
    - "PCA": 1.5
    - "species": 1.5

metabarcoding:
  name: "Metabarcoding Analysis"
  description: "eDNA and amplicon analysis"
  model: "qwen3-coder:30b"
  system_prompt: |
    Expert in metabarcoding and eDNA analysis pipelines.
    Familiar with DADA2, QIIME2, OBITools, and custom R workflows.
    Provide complete, reproducible analysis pipelines.
  temperature: 0.3
  keywords:
    - "ASV": 3.0
    - "OTU": 3.0
    - "amplicon": 2.5
    - "DADA2": 2.5
    - "primer": 2.0
```

## Multi-Agent Configuration

### `agents/config.yaml`

```yaml
# Global agent settings
global:
  max_iterations: 5
  timeout_per_agent: 120
  parallel_execution: false
  verbose_logging: true

# Agent definitions
agents:
  planner:
    model: "deepseek-r1:32b"
    temperature: 0.5
    max_tokens: 2000
    system_prompt: |
      You are a strategic planner. Break down complex tasks into
      clear, actionable steps. Consider dependencies and potential
      challenges. Output structured plans.
      
  coder:
    model: "qwen3-coder:30b"
    temperature: 0.3
    max_tokens: 4000
    system_prompt: |
      You are an expert programmer. Implement solutions based on
      the provided plan. Write clean, efficient, well-tested code.
      Follow best practices for the target language.
      
  reviewer:
    model: "qwen3:32b"
    temperature: 0.4
    max_tokens: 2000
    system_prompt: |
      You are a code reviewer. Analyze code for:
      - Correctness and logic errors
      - Performance issues
      - Security vulnerabilities
      - Code style and readability
      Provide specific, actionable feedback.
      
  explainer:
    model: "nemotron-3-nano:30b"
    temperature: 0.5
    max_tokens: 2000
    system_prompt: |
      You are a technical writer. Create clear explanations of
      code and concepts. Use examples, analogies, and diagrams
      when helpful. Adapt to the audience's level.

# Pipeline definitions
pipelines:
  code_review:
    description: "Complete code development with review"
    steps:
      - agent: "planner"
        action: "analyze_requirements"
      - agent: "coder"
        action: "implement"
      - agent: "reviewer"
        action: "review"
      - agent: "coder"
        action: "refine"
        condition: "if_review_has_issues"
        
  research:
    description: "Multi-source research synthesis"
    steps:
      - agent: "planner"
        action: "decompose_question"
      - agent: "coder"
        action: "search_and_analyze"
        parallel: true
        count: 3
      - agent: "explainer"
        action: "synthesize"
```

## RAG Configuration

### `rag/config.py`

```python
# ChromaDB settings
CHROMA_PERSIST_DIR = "data/rag/chroma_db"
COLLECTION_NAME = "opti_oignon_docs"

# Chunking settings
CHUNK_SIZE = 1000          # characters
CHUNK_OVERLAP = 200        # characters
MIN_CHUNK_SIZE = 100       # minimum viable chunk

# Retrieval settings
TOP_K = 5                  # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7  # minimum relevance score

# Embedding settings
EMBEDDING_MODEL = "mxbai-embed-large"
EMBEDDING_BATCH_SIZE = 32

# File processing
SUPPORTED_EXTENSIONS = [
    ".py", ".r", ".R", ".js", ".ts", ".sh",  # Code
    ".md", ".txt", ".rst",                    # Text
    ".pdf", ".docx",                          # Documents
    ".csv", ".json", ".yaml", ".yml",         # Data
]

MAX_FILE_SIZE_MB = 50

# Cache settings
CACHE_EMBEDDINGS = True
CACHE_DIR = "data/rag/cache"
```

## Environment Variables

Optional environment variables that override config files:

```bash
# Ollama connection
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_TIMEOUT=300

# UI settings
export OPTI_OIGNON_PORT=7860
export OPTI_OIGNON_SHARE=false  # Gradio public link

# Logging
export OPTI_OIGNON_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Data directories
export OPTI_OIGNON_DATA_DIR="~/.opti-oignon/data"
export OPTI_OIGNON_CACHE_DIR="~/.opti-oignon/cache"
```

## Tips

### Optimizing for Speed

```yaml
# config/models.yaml
speed_preference: "speed"

task_routing:
  code_python:
    primary: "qwen2.5-coder:14b"  # Smaller, faster
    timeout: 60
```

### Optimizing for Quality

```yaml
# config/models.yaml
speed_preference: "quality"

task_routing:
  code_python:
    primary: "qwen3-coder:30b"
    timeout: 180  # Allow more time
```

### Domain-Specific Setup

1. Create custom presets in `data/user_presets.yaml`
2. Add domain keywords with high weights
3. Set appropriate system prompts
4. Index relevant documents with RAG

---

[← Architecture](ARCHITECTURE.md) | [Back to README](../README.md)
