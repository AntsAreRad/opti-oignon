# Session-Summarizer Plugin

Generates a running summary of the conversation session. Every N
messages, a background thread produces an extractive summary without
blocking user interaction.

## How It Works

The plugin tracks all user prompts and assistant responses in a
conversation buffer. At configurable intervals (default: every 5
messages), it spawns a daemon thread that builds an extractive
summary by scoring sentences on position, keyword density, and
length. The summary is updated in-place and can be viewed anytime
via the `/summary` command.

Summary generation never blocks the main inference pipeline.

## Hook Points

- `post_inference` -- tracks messages and triggers background summary
- `tool_call` -- handles `/summary` and `/summary reset` commands

## Commands

| Command | Effect |
|---------|--------|
| `/summary` | Display the current session summary |
| `/summary reset` | Clear summary and reset message counter |

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `interval` | integer | `5` | Generate summary every N messages |
| `model_override` | string | `""` | Force a specific model (empty = extractive) |
| `max_summary_length` | integer | `300` | Maximum word count for summary |
| `use_extractive_fallback` | boolean | `true` | Use extractive summarization |

## Permissions

- `conversation_read` -- access to conversation history
- `model_config_read` -- read available models for future generative mode

## Example

After 10 messages about Python optimization:

```
/summary
```

```
**Session Summary** (10 messages)

The conversation focused on Python performance optimization techniques.
Key topics included profiling with cProfile, memoization via functools,
and avoiding unnecessary list copies in hot loops.
```
