# Benchmark Guide

How to evaluate your models and generate optimal routing configuration.

## Overview

The benchmark system tests each model across multiple task categories to determine:
- **Quality**: How well the model performs each task type
- **Speed**: Response time for different complexity levels
- **Consistency**: Reliability across multiple runs

Results are used to automatically configure the intelligent routing system.

## Quick Start

```bash
# Estimate time without running
opti-oignon benchmark --estimate

# Quick benchmark (3 models, ~15 min)
opti-oignon benchmark --quick --confirm

# Full benchmark (all models, ~2 hours)
opti-oignon benchmark --confirm
```

## Benchmark Modes

### Automatic Mode (Default)

Models are scored automatically based on predefined criteria:

```bash
opti-oignon benchmark --confirm
```

### Interactive Mode

You manually score each response (recommended for fine-tuning):

```bash
opti-oignon benchmark --interactive --confirm
```

You'll be prompted to rate responses from 1-10:
```
Model: qwen3-coder:30b
Task: code_python
Prompt: "Write a function to calculate Fibonacci..."

Response:
[Model output displayed here]

Your score (1-10): _
```

### Quick Mode

Tests only the top 3 recommended models:

```bash
opti-oignon benchmark --quick --confirm
```

## Task Categories

| Category | Description | Example Prompt |
|----------|-------------|----------------|
| `code_python` | Python programming | "Write a decorator for caching" |
| `code_r` | R programming | "Create a ggplot visualization" |
| `debug` | Finding and fixing bugs | "Why does this code fail?" |
| `explain` | Technical explanations | "Explain async/await" |
| `reasoning` | Complex problem solving | "Design a database schema" |
| `general` | General questions | "What is machine learning?" |

## Output Files

Results are saved to `routing/benchmarks/`:

```
routing/benchmarks/
├── benchmark_2025-01-15_14-30.json   # Detailed results
├── benchmark_2025-01-15_14-30.md     # Human-readable report
├── benchmark_latest.json              # Latest for comparison
└── routing_config_generated.yaml      # Auto-generated routing
```

### JSON Format

```json
{
  "metadata": {
    "date": "2025-01-15T14:30:00",
    "duration_seconds": 3420,
    "models_tested": 5
  },
  "results": {
    "qwen3-coder:30b": {
      "code_python": {"score": 9.2, "time": 28.5},
      "code_r": {"score": 9.0, "time": 31.2},
      "debug": {"score": 8.5, "time": 25.8}
    }
  },
  "rankings": {
    "code_python": ["qwen3-coder:30b", "devstral-small-2"],
    "reasoning": ["deepseek-r1:32b", "qwen3:32b"]
  }
}
```

### Markdown Report

```markdown
# Benchmark Report - 2025-01-15

## Summary
- Models tested: 5
- Duration: 57 minutes
- Best overall: qwen3-coder:30b

## Rankings by Category

### Code (Python)
| Rank | Model | Score | Speed |
|------|-------|-------|-------|
| 1 | qwen3-coder:30b | 9.2 | 28s |
| 2 | devstral-small-2 | 8.1 | 15s |
```

## Custom Benchmarks

### Add Custom Prompts

Edit `opti_oignon/routing/benchmark_prompts.yaml`:

```yaml
custom_prompts:
  bioinformatics:
    - prompt: "Write R code to calculate Shannon diversity index"
      expected_keywords: ["vegan", "diversity", "H'"]
    - prompt: "Parse a FASTA file and extract sequences"
      expected_keywords: ["BioPython", "SeqIO", "parse"]
```

### Run Custom Category

```bash
opti-oignon benchmark --category bioinformatics --confirm
```

## Interpreting Results

### Score Ranges

| Score | Quality | Recommendation |
|-------|---------|----------------|
| 9-10 | Excellent | Primary model for this task |
| 7-8 | Good | Reliable fallback |
| 5-6 | Acceptable | Use only if faster alternatives needed |
| <5 | Poor | Avoid for this task type |

### Speed vs Quality Trade-offs

```
                    Quality
                       ↑
              ★ deepseek-r1:32b (slow but thorough)
                       │
         ★ qwen3-coder:30b (balanced)
                       │
    ★ nemotron-3-nano:30b (fast, good quality)
                       │
  ★ qwen2.5-coder:14b (very fast, acceptable)
    ───────────────────┼──────────────────→ Speed
```

## Applying Results

### Automatic Routing Update

After benchmarking, apply results:

```bash
opti-oignon routing update --from-benchmark latest
```

### Manual Configuration

Edit `opti_oignon/routing/config.yaml`:

```yaml
task_routing:
  code_python:
    primary: "qwen3-coder:30b"
    fallback: ["devstral-small-2", "qwen2.5-coder:14b"]
    fast: "qwen2.5-coder:14b"
    
  reasoning:
    primary: "deepseek-r1:32b"
    fallback: ["qwen3:32b"]
    timeout: 300  # Allow more time for complex reasoning
```

## Tips for Accurate Benchmarks

1. **Close other applications** to ensure consistent performance
2. **Run multiple times** and average results for reliability
3. **Use interactive mode** for subjective quality assessment
4. **Test with your actual use cases** by adding custom prompts
5. **Consider your hardware** - results vary by GPU/RAM

## Comparing Benchmarks

```bash
# Compare two benchmark runs
opti-oignon benchmark compare \
  routing/benchmarks/benchmark_2025-01-10.json \
  routing/benchmarks/benchmark_2025-01-15.json
```

Output:
```
Model Performance Changes:
┌─────────────────────┬──────────┬──────────┬────────┐
│ Model               │ Before   │ After    │ Change │
├─────────────────────┼──────────┼──────────┼────────┤
│ qwen3-coder:30b     │ 8.5      │ 9.2      │ +0.7   │
│ deepseek-r1:32b     │ 8.0      │ 8.0      │ 0.0    │
└─────────────────────┴──────────┴──────────┴────────┘
```

---

[← Installation](INSTALLATION.md) | [Architecture →](ARCHITECTURE.md)
