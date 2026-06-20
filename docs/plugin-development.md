# Plugin Development

## Quick start

The fastest way to scaffold a new plugin:

1. Open **Settings > Plugins > Marketplace**
2. Click **New Plugin**
3. Fill in name, author, description, and select hooks
4. Click **Generate**

Or use the API:

```
POST /api/plugins/marketplace/template
{
  "name": "my-plugin",
  "author": "Your Name",
  "description": "Does something useful.",
  "hooks": ["post_inference"],
  "permissions": []
}
```


## Plugin structure

A minimal plugin directory:

```
my-plugin/
  manifest.yaml       # Required: plugin metadata
  entry_point.py      # Required: Python code with hooks
  README.md           # Recommended: documentation
```


## Manifest reference

```yaml
name: "my-plugin"
version: "1.0.0"
author: "Your Name"
description: "Short description of what this plugin does."
entry_point: "entry_point.py"

hooks:
  - post_inference

permissions: []

dependencies: []

min_opti_version: "1.0.0"

config_schema:
  enabled:
    type: "boolean"
    default: true
    description: "Enable or disable this plugin"
```

### Required fields

| Field | Description |
|-------|-------------|
| `name` | Lowercase alphanumeric with hyphens/underscores, 2-64 chars |
| `version` | Semantic versioning (e.g., `1.0.0`) |
| `author` | Author name |
| `description` | Short description |
| `entry_point` | Relative path to the Python file |

### Optional fields

| Field | Description |
|-------|-------------|
| `hooks` | Hook points this plugin uses |
| `permissions` | Permissions requested at runtime |
| `dependencies` | Other plugin names required |
| `min_opti_version` | Minimum Opti-Oignon version |
| `config_schema` | User-editable settings schema |


## Hook points

Plugins interact with Opti-Oignon through hooks. Each hook receives a
`data` dict and must return a (possibly modified) dict.

### Available hooks

| Hook | Data keys | Purpose |
|------|-----------|---------|
| `pre_prompt` | prompt, conversation_id, model | Modify prompt before sending |
| `post_prompt` | prompt, metadata | Process assembled prompt |
| `pre_inference` | messages, options, model | Modify inference parameters |
| `post_inference` | response, model, duration_ms | Post-process model output |
| `tool_call` | tool_name, arguments, result | Intercept tool invocations |
| `pipeline_step` | input, step_name, pipeline_id | Pipeline processing step |
| `ui_panel` | (none required) | Contribute a UI panel (return `html` key) |

### Example: post-inference hook

```python
def hook_post_inference(data: dict) -> dict:
    """Strip trailing whitespace from model responses."""
    data["response"] = data["response"].strip()
    return data
```

### Hook registration

Two options:

**Function naming convention** -- name functions `hook_<hook_name>`:

```python
def hook_post_inference(data: dict) -> dict:
    return data
```

**HOOKS dictionary** -- export a mapping:

```python
def my_handler(data: dict) -> dict:
    return data

HOOKS = {
    "post_inference": my_handler,
}
```

When multiple plugins register the same hook, they execute in load
order. Each receives the output of the previous one (data chaining).


## Permissions

Plugins must declare needed permissions in the manifest. Undeclared
permissions are blocked at runtime.

| Permission | Description |
|------------|-------------|
| `conversation_read` | Read conversation messages |
| `conversation_write` | Modify conversation messages |
| `model_config_read` | Read model configuration |
| `model_config_write` | Modify model configuration |
| `tool_register` | Register new tools |


## Subprocess isolation

Plugins run in isolated subprocesses, not in the main process. This
means:

- Plugins cannot access the main application's memory
- Crashes in a plugin do not crash the backend
- Network access is blocked when bwrap is available
- Filesystem access is restricted to the plugin's own directory
- Communication uses HMAC-authenticated Unix domain sockets (primary)
  or stdin/stdout pipes (async mode)
- Each call has a configurable timeout with SIGTERM/SIGKILL escalation

!!! warning
    Plugins have no direct access to the host filesystem or network.
    Files must be explicitly passed through the plugin API. This is a
    non-negotiable security requirement.


## Testing your plugin

Create a test file alongside your plugin:

```python
def test_my_hook():
    from entry_point import hook_post_inference
    result = hook_post_inference({"response": "  hello  ", "model": "test", "duration_ms": 100})
    assert result["response"] == "hello"
```

Run tests with:

```bash
pytest my-plugin/
```


## Publishing

To submit a plugin to the marketplace, ensure your plugin passes
validation (`manifest.yaml` schema check, entry point exists, hooks
are valid) and submit it through the marketplace UI or API.

Only plugins on the verified allowlist can be installed by other users.
