# Fact-Checker Plugin

Cross-references factual claims in LLM responses against web search results.

## How It Works

After the LLM generates a response, this plugin scans for factual claims
(dates, numbers, proper nouns, named entities) and verifies them via
DuckDuckGo web search. Each claim is annotated inline:

- **[verified]** — claim matches web search results
- **[unverified]** — no confirmation found (or web search unavailable)
- **[conflict: ...]** — web results contradict the claim

## Hook Point

- `post_inference` — processes the LLM response after generation

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `aggressiveness` | string | `moderate` | `low` (dates/numbers), `moderate` (+ entities), `high` (all claims) |
| `max_checks` | integer | `5` | Maximum claims to verify per response |
| `skip_code_blocks` | boolean | `true` | Skip claims found inside code blocks |
| `trusted_domains` | string | `""` | Comma-separated domains to auto-trust |

## Permissions

- `network_outbound` — required for web search
- `conversation_read` — reads the LLM response

## Graceful Degradation

If DuckDuckGo search is unavailable (missing `duckduckgo-search` package
or network issues), all claims are marked `[unverified]`. The plugin
never blocks or crashes the inference pipeline.

## Examples

Input response:
> The Moon is approximately 384,400 km from Earth. Python was created
> by Guido van Rossum in 1991.

Output (with web search available):
> The Moon is approximately 384,400 km from Earth [verified]. Python
> was created by Guido van Rossum in 1991 [verified].
