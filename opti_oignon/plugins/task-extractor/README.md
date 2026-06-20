# Task-Extractor Plugin

Automatically extracts action items and TODOs from LLM responses and
maintains a persistent task list that you can manage via slash commands.

## How It Works

**Post-inference:** Scans each LLM response for action patterns using
three extraction strategies:

1. **Pattern matching** — phrases like "you should", "next step",
   "TODO:", "make sure to", "remember to"
2. **Numbered steps** — detects "1. Do X", "2) Do Y" sequences
3. **Imperative detection** — sentences starting with action verbs
   (install, configure, create, deploy, test, etc.)

Extracted tasks are stored in SQLite and a summary is appended to the
response.

**Tool call:** Slash commands for viewing, completing, and clearing tasks.

## Hook Points

- `post_inference` — extracts tasks from responses
- `tool_call` — handles task management commands

## Commands

| Command | Description |
|---------|-------------|
| `/tasks` | List pending tasks |
| `/tasks all` | List all tasks including completed |
| `/tasks done <id>` | Mark a task as complete |
| `/tasks clear` | Remove all tasks |
| `/tasks clear done` | Remove only completed tasks |

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_extract` | boolean | `true` | Automatically extract tasks from every response |
| `patterns` | string | `you should,next step,...` | Comma-separated trigger patterns |
| `max_tasks` | integer | `200` | Maximum number of stored tasks |

## Permissions

- `conversation_read` — reads LLM responses for extraction
- `tool_register` — registers slash commands
- `filesystem_plugin_dir` — stores the SQLite database

## Deduplication

Tasks are deduplicated by normalized text (lowercase, trimmed). Tasks
shorter than 10 characters are ignored. Code blocks are stripped before
extraction to avoid false positives from code comments.
