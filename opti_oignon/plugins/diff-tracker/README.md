# Diff-Tracker Plugin

Detects code blocks in LLM responses that are iterations of previously
seen code in the conversation. Computes and displays a unified diff
(additions/deletions) after the code block.

## How It Works

The plugin maintains a history of all code blocks seen across messages.
When a new code block appears, it is matched against history using two
strategies:

1. **Name matching** -- function/class names are extracted from both
   blocks. If names overlap, blocks are likely related and the similarity
   threshold is lowered.
2. **Structural similarity** -- `difflib.SequenceMatcher` computes a
   line-by-line similarity ratio. Blocks exceeding the threshold are
   considered iterations.

The best-matching previous block is used to generate a unified diff
that is appended after the code block.

## Hook Point

- `post_inference` -- scans the response for code blocks and appends diffs

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `similarity_threshold` | number | `0.4` | Minimum similarity ratio (0.0-1.0) to match blocks |
| `diff_format` | string | `inline` | Diff display format (`inline` or `unified`) |
| `show_stats` | boolean | `true` | Show +N/-N counts in diff header |
| `max_history` | integer | `50` | Maximum code blocks kept in history |

## Permissions

- `conversation_read` -- access to conversation history for code tracking

## Supported Languages

Name extraction works for Python, JavaScript/TypeScript, R, Rust, and
Go. Similarity-based matching works for any language.

## Example

If the first response contains:

```python
def greet(name):
    print(f"Hello {name}")
```

And a later response contains:

```python
def greet(name, greeting="Hello"):
    print(f"{greeting} {name}")
    return True
```

The plugin appends:

```
**Diff:** +2 additions, -1 deletions
```diff
--- previous
+++ current
-def greet(name):
-    print(f"Hello {name}")
+def greet(name, greeting="Hello"):
+    print(f"{greeting} {name}")
+    return True
```
```
