# CLI Reference

## Overview

The `oo` command-line tool is a companion for interacting with a
running Opti-Oignon backend from the terminal. Most commands communicate
via the HTTP API and never import heavy backend dependencies. Two run in
the calling process instead: `oo chat`, the interactive session, and
`oo core`, the resident core daemon.

Install: `pip install -e ".[all]"` registers the `oo` console script.


## Global options

| Option | Env var | Description |
|--------|---------|-------------|
| `--api-url URL` | `OO_API_URL` | Override the backend API URL |
| `--no-color` | | Disable colored output |
| `--version` | | Show version and exit |


## Commands

### oo ask

Send a prompt to the backend and display the response.

```bash
oo ask "Summarize this dataset"
oo ask -m llama3 "Explain PCA"
cat data.csv | oo ask --pipe "Analyze this"
oo ask -f prompt.txt
```

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Force a specific model |
| `-f, --file PATH` | Read prompt from file |
| `--pipe` | Read stdin as additional context |

### oo chat

An interactive session, one line per turn, run in this process: the
executor, the conversation store, the onion memory and the skill registry
are the local ones, and inference goes through the inference registry
configured from `backends.yaml` -- the core daemon when `core.yaml`
enables it, never a client of the session's own. It does not need the API
server.

```bash
oo chat
oo chat -m llama3
oo chat --conversation <conversation-id>
```

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Force a specific model instead of routing |
| `--conversation ID` | Continue an existing conversation |

A line that does not start with `/` is a turn. A line that does is a
command, and every command is a user action:

| Command | Description |
|---------|-------------|
| `/open ID` | Find a persisted onion again and continue that conversation |
| `/close` | Evict the whole Flesh through the gate, save, and end the conversation |
| `/pin TEXT` | Pin a statement to the conversation's Core, as the user |
| `/recall KEY` | Show the verbatim span behind a receipt; this marks the receipt resolved |
| `/skill NAME ARGS` | Run `ARGS` as a turn with a published skill as the system suffix |
| `/help` | List the commands |
| `/quit` | End the session |

The answer streams to stdout; every refusal goes to stderr with its
reason. The onion commands are refused while `onion.yaml` has the onion
switched off, and `/open` is refused when no persistence path is set or
the conversation was never saved. `/close` never overrides the gate: a
span it refuses stays verbatim in the Flesh and the refusal names the
probes that failed. `/skill` takes `name` or `category/name`; a draft, an
unknown name, or a name published in two categories is refused, and only
a published skill -- human-approved, and named by the user on purpose --
reaches the system prompt.

### oo models

List all models available in the connected Ollama instance.

```bash
oo models
```

Displays model name, size, quantization, and parameter count in a
formatted table.

### oo status

Show backend health and configuration summary.

```bash
oo status
```

Displays version, uptime, active model, security mode, and feature
availability flags.

### oo backup

Export and import conversation history and configuration.

```bash
oo backup export backup.json
oo backup export                    # auto-named with timestamp
oo backup import backup.json
oo backup import backup.json --strategy merge
```

| Subcommand | Description |
|------------|-------------|
| `export [OUTPUT]` | Export all data to JSON |
| `import INPUT` | Import data from JSON |

Import options:

| Option | Description |
|--------|-------------|
| `--strategy {replace,merge}` | How to handle conflicts (default: replace) |

### oo rag

Manage RAG collections from the terminal.

```bash
oo rag ingest paper.pdf --collection ecology
oo rag query "What is BCI?" --collection ecology
oo rag query "species diversity" --collection ecology --n-results 10
```

| Subcommand | Description |
|------------|-------------|
| `ingest FILE --collection NAME` | Ingest a file into a collection |
| `query QUESTION --collection NAME` | Query a collection |

Query options:

| Option | Description |
|--------|-------------|
| `--n-results N` | Number of results to return (default: 5) |

### oo redteam

Run and manage red team security audits.

```bash
oo redteam run
oo redteam run --quick
oo redteam run --categories injection,jailbreak --targets rag_sanitizer
oo redteam status
oo redteam report
oo redteam report --format json
oo redteam report --id <report-id>
oo redteam compare <id1> <id2>
```

| Subcommand | Description |
|------------|-------------|
| `run` | Launch a red team audit campaign |
| `status` | Show current campaign progress |
| `report` | Display the latest or a specific report |
| `compare ID1 ID2` | Compare two reports side by side |

Run options:

| Option | Description |
|--------|-------------|
| `--quick` | Reduced attack count for faster results |
| `--categories LIST` | Comma-separated attack categories |
| `--targets LIST` | Comma-separated target components |

Report options:

| Option | Description |
|--------|-------------|
| `--format {text,json}` | Output format (default: text) |
| `--id ID` | Specific report ID |
| `--last` | Use the most recent report |

### oo core

Serve the resident core daemon, or ask whether it answers. Both read
`core.yaml`; see the architecture page on the core and the packs.

```bash
oo core serve                      # run the daemon in the foreground
oo core status                     # exit 0 when it answers, 1 otherwise
oo core serve --config path/to/core.yaml
```

| Subcommand | Description |
|------------|-------------|
| `serve` | Run the core daemon on the loopback |
| `status` | Ask the configured daemon for its health |

### oo config

View and modify CLI configuration.

```bash
oo config                          # show current config
oo config set api_url http://remote:8000
oo config set bulbe_mode true
oo config reset                    # reset to defaults
```

| Subcommand | Description |
|------------|-------------|
| (none) | Show current configuration |
| `set KEY VALUE` | Set a configuration value |
| `reset` | Reset all settings to defaults |

Configuration is stored in `~/.config/opti-oignon/cli.yaml`.
