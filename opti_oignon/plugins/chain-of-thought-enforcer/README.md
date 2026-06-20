# Chain-of-Thought Enforcer Plugin

Detects complex questions and injects chain-of-thought reasoning
instructions into the LLM prompt, then formats the response to
clearly separate reasoning from the final answer.

## How It Works

**Pre-inference:** Analyzes the user prompt for complexity signals
(keywords like "why", "compare", "calculate"; math operators;
multi-part questions). If complex, injects a CoT instruction into
the system message.

**Post-inference:** Parses the LLM response for reasoning markers
("Step 1", "Therefore", "Final answer") and formats the output with
a clear visual separation between the thought process and conclusion.

## Hook Points

- `pre_inference` — injects CoT instruction
- `post_inference` — formats reasoning vs answer

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `complexity_keywords` | string | `why,how,compare,...` | Comma-separated trigger keywords |
| `injection_template` | string | `Think step by step...` | Instruction injected into system message |
| `format_style` | string | `separator` | Output format: `separator`, `collapsible`, `plain` |
| `min_question_words` | integer | `5` | Minimum word count to consider a prompt complex |

## Permissions

None required. Only reads and modifies prompt/response data.

## Format Styles

**separator** (default):
```
**Reasoning:**
[step-by-step analysis]

---

**Answer:**
[final conclusion]
```

**collapsible:**
```
<details><summary>Reasoning</summary>
[step-by-step analysis]
</details>

**Answer:**
[final conclusion]
```

**plain:**
```
[reasoning]

[answer]
```
