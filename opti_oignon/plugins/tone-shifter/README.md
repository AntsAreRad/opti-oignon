# Tone-Shifter Plugin

Transforms LLM response tone via rule-based text processing. No
re-inference needed -- each mode applies regex and string replacement
rules sequentially to shift the response style.

## How It Works

The plugin intercepts the LLM response in the `post_inference` hook and
applies a set of transformation rules based on the active mode. Code
blocks (fenced and inline) are protected from transformation.

## Available Modes

| Mode | Effect |
|------|--------|
| `academic` | Add hedging language ("appears to be", "suggests"), replace absolutes with cautious phrasing |
| `casual` | Add contractions, replace formal connectors with conversational ones |
| `eli5` | Replace technical jargon with simple analogies, simplify transitions |
| `formal` | Expand contractions, replace casual words with professional equivalents |
| `concise` | Strip filler phrases, compress verbose constructions |
| `verbose` | Expand abbreviations, add transitional phrases |
| `none` | Disabled (default) |

## Hook Point

- `post_inference` -- transforms the LLM response after generation

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `active_mode` | string | `none` | Active tone mode |
| `custom_rules` | object | `{}` | Custom regex replacement rules (pattern: replacement) |

## Permissions

None required. Pure text processing with no external dependencies.

## Example

With `active_mode: concise`, the response:

> It is worth noting that, in order to optimize performance, you should
> consider implementing a caching layer due to the fact that repeated
> queries are expensive.

Becomes:

> To optimize performance, consider implementing a caching layer because
> repeated queries are expensive.
