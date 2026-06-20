# GitHub-Connector Plugin

First external service connector for Opti-Oignon. Provides slash
commands for interacting with the GitHub API and auto-enrichment
of GitHub references in LLM responses.

## How It Works

The plugin stores a GitHub Personal Access Token (PAT) in a local
SQLite database within the plugin directory. All API calls go through
a single `_github_api()` helper using `urllib.request` (stdlib) for
consistent error handling and rate limit awareness.

The post-inference hook scans responses for GitHub references
(owner/repo#123, full URLs) and appends metadata footnotes with
issue/PR titles and status.

## Commands

| Command | Description |
|---------|-------------|
| `/gh auth <token>` | Store and validate a GitHub PAT |
| `/gh auth status` | Show auth status (username, scopes) |
| `/gh auth revoke` | Remove stored token |
| `/gh issues [owner/repo]` | List open issues |
| `/gh pr list [owner/repo]` | List open pull requests |
| `/gh repo info <owner/repo>` | Repository details (stars, forks, language) |
| `/gh search <query>` | Search repositories |
| `/gh gist create <desc>` | Create gist from last code block |

## Hook Points

- `tool_call` -- handles all `/gh` commands
- `post_inference` -- auto-detects and enriches GitHub references

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_repo` | string | `""` | Default owner/repo for bare #123 references |
| `max_results` | integer | `10` | Maximum results per query |
| `auto_link` | boolean | `true` | Enable reference enrichment |
| `show_rate_limit` | boolean | `false` | Show rate limit info in responses |

## Permissions

- `network_outbound` -- make HTTPS requests to api.github.com
- `conversation_read` -- access conversation for gist creation
- `tool_register` -- register /gh commands
- `filesystem_plugin_dir` -- store auth token in plugin directory

## Security

- Token is **never echoed** in responses
- Token stored in plugin-directory SQLite (sandboxed per plugin)
- `/gh auth revoke` performs a hard DELETE
- All network errors caught gracefully (no pipeline crashes)
- Rate limit warnings when remaining calls < 10

## Example

```
/gh auth ghp_xxxxxxxxxxxxxxxxxxxx
> Authenticated as octocat.

/gh issues torvalds/linux
> **Open issues in torvalds/linux** (10):
> - #1234: Fix memory leak in driver
> - #1235: Add support for new hardware
> ...

/gh repo info anthropics/anthropic-sdk-python
> **anthropics/anthropic-sdk-python**
> The official Python SDK for Anthropic's API
> - Language: Python
> - Stars: 1234
> ...
```

Auto-enrichment in responses:

```
Check out torvalds/linux#1234 for the fix.

---
**GitHub references:**
- torvalds/linux#1234 (Issue) [open]: Fix memory leak in driver
```
