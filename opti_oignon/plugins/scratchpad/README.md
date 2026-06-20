# Scratchpad Plugin

Persistent note-taking directly from the chat interface. Save, search,
and organize snippets of information with automatic keyword tagging.

## How It Works

Notes are stored in a local SQLite database within the plugin directory.
Use slash commands to manage notes, and view them in the dedicated UI
side panel.

## Hook Points

- `tool_call` — intercepts slash commands
- `ui_panel` — renders the scratchpad side panel

## Commands

| Command | Description |
|---------|-------------|
| `/note <text>` | Save a new note |
| `/notes` | List all notes (most recent first) |
| `/note delete <id>` | Delete a note by its ID |
| `/note search <query>` | Search notes by keyword |
| `/note export` | Export all notes as markdown |

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_note_length` | integer | `2000` | Maximum character length per note |
| `max_notes` | integer | `500` | Maximum number of stored notes |
| `auto_tag` | boolean | `true` | Automatically extract keyword tags |

## Permissions

- `conversation_read` — reads user input for command detection
- `tool_register` — registers slash commands
- `ui_panel_register` — registers the side panel
- `filesystem_plugin_dir` — stores the SQLite database

## Auto-Tagging

When enabled, each note is automatically tagged with its most
significant keywords (up to 5). Tags are extracted by filtering
out common stop words and selecting the longest remaining terms.
Tags are searchable via `/note search`.

## Data Storage

Notes are stored in `scratchpad.db` within the plugin directory.
The database uses WAL journal mode for concurrent read performance.
