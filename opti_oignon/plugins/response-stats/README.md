# Response-Stats Plugin

Appends statistics to LLM responses: estimated token count, reading
time, Flesch-Kincaid readability, word/sentence/paragraph counts,
and a complexity label.

## How It Works

After each LLM response, the plugin analyzes the text (excluding code
blocks) and computes readability metrics. Stats are appended as a
footer or prepended as a header, in compact or detailed format.

## Metrics

| Metric | Description |
|--------|-------------|
| Words | Total word count (including code blocks) |
| Sentences | Sentence count in prose sections |
| Paragraphs | Paragraph count (separated by blank lines) |
| Tokens | Estimated LLM token count (~1.33 per word, ~1.8 for code) |
| Reading time | Estimated at 238 WPM |
| Readability | Flesch-Kincaid grade level + reading ease |
| Complexity | Label: simple (< 6), moderate (6-10), complex (10-14), advanced (14+) |

## Hook Point

- `post_inference` -- analyzes and annotates the LLM response

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled_stats` | array | all metrics | Which stats to include |
| `position` | string | `footer` | `footer` or `header` |
| `style` | string | `compact` | `compact` (single line) or `detailed` (multi-line) |
| `min_words` | integer | `20` | Minimum words to trigger stats |

## Permissions

None required. Pure text processing with no external dependencies.

## Example

Compact footer (default):

```
[LLM response text...]

*142 words | 8 sentences | 3 paragraphs | ~189 tokens | < 1 min | FK 8.2 | moderate*
```
