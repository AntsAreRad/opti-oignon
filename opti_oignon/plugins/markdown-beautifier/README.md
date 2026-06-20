# Markdown-Beautifier Plugin

Normalizes and beautifies markdown in LLM responses. Fixes common
formatting issues that LLMs produce: inconsistent header spacing,
broken list indentation, misaligned tables, missing blank lines
around code blocks, and unclosed code fences.

## How It Works

The plugin applies a configurable set of formatting rules to the
LLM response text. Code blocks are protected during content-level
rules to avoid breaking code formatting.

## Rules

| Rule | Effect |
|------|--------|
| `header_spacing` | Ensure blank lines before/after headers, one space after `#` |
| `list_formatting` | Normalize indentation to 2-space multiples; strict mode standardizes markers to `-` |
| `table_alignment` | Pad columns to equal widths for visual alignment |
| `code_block_spacing` | Add missing blank lines before opening and after closing fences |
| `fence_repair` | Close unclosed ` ``` ` fences at end of response |

## Hook Point

- `post_inference` -- processes the LLM response after generation

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rules` | array | all 5 rules | Which rules to apply |
| `strictness` | string | `normal` | `normal` = safe fixes; `strict` = opinionated (e.g. standardize list markers) |

## Permissions

None required. Pure text processing with no external dependencies.

## Example

Input with formatting issues:

```
##Missing space
Some text right under header
- Item 1
   * Sub-item with wrong indent
| Name | Value |
|---|---|
| foo | 123 |
| barbaz | 4 |
```python
code here
```

Output after beautification:

```
## Missing space

Some text right under header

- Item 1
  - Sub-item with wrong indent

| Name   | Value |
| ------ | ----- |
| foo    | 123   |
| barbaz | 4     |

```python
code here
```
```
