# CLI Reference

## Overview

The `oo` command-line tool is a companion for interacting with a
running Opti-Oignon backend from the terminal. It communicates via the
HTTP API and never imports heavy backend dependencies.

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
