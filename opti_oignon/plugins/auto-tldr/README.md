# Auto-TLDR Plugin

Automatically generates a concise TL;DR summary for long LLM responses
using extractive summarization.

## How It Works

When a response exceeds the configured word threshold (default 300 words),
the plugin scores every sentence on four criteria and selects the top-scoring
sentences as the summary:

1. **Position (30%)** — first and last sentences score higher (topic
   sentences and conclusions)
2. **Keyword density (40%)** — overlap with the most frequent meaningful
   words in the full text
3. **Length (20%)** — sentences close to the average length score highest;
   very short or very long sentences are penalized
4. **Filler penalty** — sentences containing filler phrases ("it is worth
   noting", "at the end of the day") are penalized

The selected sentences are reordered by their original position and
prepended as a TL;DR block.

## Hook Point

- `post_inference` — processes the LLM response after generation

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `word_threshold` | integer | `300` | Minimum word count to trigger summarization |
| `max_summary_sentences` | integer | `2` | Maximum sentences in the TL;DR |
| `separator` | string | `---` | Separator between TL;DR and full response |

## Permissions

None required. Pure text processing with no external dependencies.

## Example

For a 500-word response about machine learning, the output becomes:

```
**TL;DR:** Neural networks learn patterns through iterative weight
adjustment. The key challenge remains avoiding overfitting on small datasets.

---

[full original response]
```
