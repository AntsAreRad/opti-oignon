# Adaptive Context Management

## Overview

The Context Manager provides intelligent context management for LLM interactions in Opti-Oignon. It handles model limit detection, token estimation, context validation, and smart truncation to ensure optimal performance and prevent errors.

## Features

- **Dynamic Model Limit Detection**: Fetches actual context limits from Ollama for installed models
- **Token Estimation**: Approximates token counts using character-to-token ratios
- **Context Validation**: Checks if prompts fit within model limits before execution
- **Smart Truncation**: Intelligently truncates content while preserving important parts
- **Warning System**: Provides visual feedback when approaching context limits
- **UI Integration**: Real-time context indicator in the Gradio interface

## Verified Model Limits

| Model | Context Window | Max Output | Source |
|-------|---------------|------------|--------|
| qwen3-coder:30b | 262,144 (256K) | 65,536 | HuggingFace, Ollama |
| deepseek-r1:32b | 131,072 (128K) | 32,768 | DeepSeek GitHub |
| gemma3:27b | 131,072 (128K) | 8,192 | Google AI docs |
| nemotron-3-nano:30b | 1,048,576 (1M) | 16,384 | NVIDIA research |
| llama3.3:latest | 131,072 (128K) | 16,384 | Meta official |
| devstral-small-2:latest | 262,144 (256K) | 16,384 | Mistral AI |
| qwen2.5-coder:7b | 131,072 (128K) | 8,192 | QwenLM |
| mistral-small3.2:latest | 131,072 (128K) | 16,384 | Mistral AI |

## Usage

### Basic Context Check

```python
from opti_oignon.context_manager import check_context

check = check_context(
    prompt="Write a function to calculate Shannon index",
    document="species,count\n" + "sp1,100\n" * 1000,
    system_prompt="You are an R expert.",
    model="qwen3-coder:30b"
)

print(f"Total: ~{check.total_tokens:,} tokens")
print(f"Usage: {check.usage_percent:.1f}%")
print(f"Safe: {check.is_safe}")

if check.warning_message:
    print(f"⚠️ {check.warning_message}")
```

### Token Estimation

```python
from opti_oignon.context_manager import estimate_tokens

text = "Your document content here..."
tokens = estimate_tokens(text, model="qwen3-coder:30b")
print(f"Estimated tokens: {tokens:,}")
```

### Smart Truncation

```python
from opti_oignon.context_manager import smart_truncate

large_doc = "..." * 100000
truncated, removed = smart_truncate(
    large_doc,
    max_tokens=50000,
    model="qwen3-coder:30b"
)

print(f"Removed ~{removed:,} tokens")
```

### Get Model Limits

```python
from opti_oignon.context_manager import get_model_limits

limits = get_model_limits("qwen3-coder:30b")
print(f"Context window: {limits.context_window:,}")
print(f"Recommended input: {limits.recommended_input:,}")
print(f"Source: {limits.source}")
```

## UI Integration

The context manager integrates with the Gradio UI to provide real-time feedback:

- **Context Indicator**: Shows current usage as a progress bar
- **Status Emoji**: 🟢 Safe / 🟠 Warning / 🟡 Caution / 🔴 Critical
- **Warnings**: Displayed when approaching or exceeding limits
- **Auto-Update**: Updates when model, question, or document changes

### Context Indicator Display

```
📊 Context Usage • `qwen3-coder:30b`

Input: ~12,450 / 196,608 tokens
[████████░░░░░░░░░░░░] 6%

*Prompt: ~150 • Doc: ~12,200 • System: ~100*
```

## Priority System

The context manager uses a priority system for determining model limits:

1. **Cached Result** (fastest)
2. **Dynamic Ollama Info** (most accurate for installed models)
3. **Config File Limits** (from models.yaml)
4. **Built-in Verified Defaults** (researched values)
5. **Universal Fallback** (8K context)

## API Reference

### Classes

#### `ModelLimits`
Dataclass containing model context limits.

| Field | Type | Description |
|-------|------|-------------|
| context_window | int | Total context capacity in tokens |
| max_output | int | Maximum output tokens |
| recommended_input | int | Recommended input size |
| chars_per_token | float | Character-to-token ratio |
| display_name | str | Human-readable model name |
| source | str | Where limits came from |

#### `ContextCheck`
Dataclass containing context validation results.

| Field | Type | Description |
|-------|------|-------------|
| total_tokens | int | Estimated total tokens |
| prompt_tokens | int | Tokens in prompt |
| document_tokens | int | Tokens in document |
| system_tokens | int | Tokens in system prompt |
| exceeds_limit | bool | True if context exceeds limit |
| exceeds_recommended | bool | True if exceeds recommended |
| usage_percent | float | Percentage of available input used |
| warning_message | str | Warning message if applicable |

### Functions

| Function | Description |
|----------|-------------|
| `check_context(prompt, document, system, model)` | Validate context against model limits |
| `estimate_tokens(text, model)` | Estimate token count for text |
| `get_model_limits(model)` | Get limits for a specific model |
| `smart_truncate(text, max_tokens, model)` | Intelligently truncate text |
| `format_context_indicator(model, question, doc)` | Generate UI display string |

## Testing

```bash
# Run all tests
python -m pytest tests/test_context_manager.py -v

# Run standalone
python tests/test_context_manager.py
```

## Troubleshooting

### "Using default limits for unknown model"

The model is not in the verified defaults and Ollama info is unavailable.

**Solution**: Add the model to `config/models.yaml` or ensure Ollama is running.

### Incorrect token estimates

Token estimation uses character ratios which vary by content type.

**Solution**: Adjust `chars_per_token` in config for specific use cases.

### Context check shows 0 tokens

Empty text or estimation failure.

**Solution**: Verify text is not empty and model name is correct.
