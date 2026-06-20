# Code-Guardian Plugin

Validates code blocks in LLM responses and appends syntax badges
indicating whether the code is correct or contains errors.

## How It Works

After the LLM generates a response, this plugin extracts all fenced
code blocks, identifies the language, and runs the appropriate validator:

- **Python** — `ast.parse()` for syntax validation, plus pitfall detection
  (unused imports, bare except, mutable default arguments)
- **JSON** — `json.loads()` with line/column error reporting
- **R** — Heuristic validation: matching delimiters (parentheses, braces,
  brackets), unclosed string detection, comment handling

A badge is appended after each validated block.

## Hook Point

- `post_inference` — processes the LLM response after generation

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `languages` | string | `python,json,r` | Comma-separated languages to validate |
| `badge_format` | string | `bracket` | Badge style: `bracket`, `emoji`, or `hidden` |
| `min_lines` | integer | `2` | Minimum lines in a code block to trigger validation |

## Permissions

- `conversation_read` — reads the LLM response

## Badge Formats

**bracket** (default):
```
[Python Syntax OK]
[JSON Syntax Error line 3: Expecting ',' delimiter]
[Python Syntax OK] (warnings: Possibly unused import: os)
```

**emoji:**
```
[Python OK]
[Json Error line 3: Expecting ',' delimiter]
```

**hidden:**
Only error badges are shown; valid blocks get no badge.

## Language Aliases

The plugin recognizes common language tag variations:

| Tag | Resolved |
|-----|----------|
| `py`, `python3` | `python` |
| `jsonc` | `json` |
| `rlang` | `r` |

## Python Pitfall Detection

Beyond syntax validation, the Python validator checks for:

- **Unused imports** — imported names not referenced in the code
- **Bare except clauses** — `except:` without specifying an exception type
- **Mutable default arguments** — `def f(x=[])` pattern
