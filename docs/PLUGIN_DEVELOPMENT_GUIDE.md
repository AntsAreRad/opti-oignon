# Opti-Oignon Plugin Development Guide

This guide covers everything you need to create, test, and publish plugins
for Opti-Oignon.

## Overview

Plugins extend Opti-Oignon through a hook-based architecture. Each plugin
is a directory containing a manifest file (`manifest.yaml`) and a Python
entry point. Plugins run in a sandboxed environment with restricted imports
and filesystem access.

## Quick Start

The fastest way to create a new plugin is with the template generator:

1. Open **Settings > Plugins > Marketplace**
2. Click **New Plugin**
3. Fill in name, author, description, and select hooks
4. Click **Generate**

This creates a ready-to-edit scaffold in your plugins directory.

Alternatively, use the API:

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

## Plugin Structure

A minimal plugin directory looks like this:

```
my-plugin/
  manifest.yaml       # Required: plugin metadata
  entry_point.py      # Required: Python code with hooks
  README.md           # Recommended: documentation
```

## Manifest Reference

The `manifest.yaml` file describes your plugin:

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

### Manifest Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Lowercase alphanumeric with hyphens/underscores, 2-64 chars |
| `version` | Yes | Semantic versioning (e.g. `1.0.0`, `2.1.0-beta`) |
| `author` | Yes | Author name |
| `description` | Yes | Short description |
| `entry_point` | Yes | Relative path to the Python file (must end in `.py`) |
| `hooks` | No | List of hook points this plugin uses |
| `permissions` | No | List of permissions requested |
| `dependencies` | No | List of other plugin names required |
| `min_opti_version` | No | Minimum Opti-Oignon version (default: `1.0.0`) |
| `config_schema` | No | Configuration schema for user-editable settings |

### Name Rules

Plugin names must match the pattern `^[a-z][a-z0-9_-]{1,63}$`:
- Start with a lowercase letter
- Only lowercase letters, digits, hyphens, underscores
- 2 to 64 characters total

## Hook Points

Plugins interact with Opti-Oignon through hooks. Each hook receives a `data`
dict and must return a (possibly modified) dict.

### Available Hooks

**`pre_prompt`** — Called before the prompt is sent to the model.
```python
def hook_pre_prompt(data: dict) -> dict:
    # data keys: prompt, conversation_id, model
    data["prompt"] = data["prompt"] + "\nExtra context."
    return data
```

**`post_prompt`** — Called after the prompt is assembled but before display.
```python
def hook_post_prompt(data: dict) -> dict:
    # data keys: prompt, metadata
    return data
```

**`pre_inference`** — Called before model inference starts.
```python
def hook_pre_inference(data: dict) -> dict:
    # data keys: messages, options, model
    return data
```

**`post_inference`** — Called after model inference completes.
```python
def hook_post_inference(data: dict) -> dict:
    # data keys: response, model, duration_ms
    data["response"] = data["response"].strip()
    return data
```

**`tool_call`** — Called when a tool is invoked.
```python
def hook_tool_call(data: dict) -> dict:
    # data keys: tool_name, arguments, result
    return data
```

**`pipeline_step`** — Called as a pipeline processing step.
```python
def hook_pipeline_step(data: dict) -> dict:
    # data keys: input, step_name, pipeline_id
    return data
```

**`ui_panel`** — Called to contribute a UI panel.
```python
def hook_ui_panel(data: dict) -> dict:
    # Return data with "html" key for panel content
    data["html"] = "<div>My Panel</div>"
    return data
```

### Hook Registration

There are two ways to register hooks:

**Option A: Function naming convention**

Name your functions `hook_<hook_name>`:

```python
def hook_post_inference(data: dict) -> dict:
    return data
```

**Option B: HOOKS dictionary**

Export a `HOOKS` dict mapping hook names to callables:

```python
def my_custom_handler(data: dict) -> dict:
    return data

HOOKS = {
    "post_inference": my_custom_handler,
}
```

### Hook Priority

When multiple plugins register the same hook, they execute in the order
they were loaded. Each hook receives the output of the previous one
(data chaining). If a hook raises an exception, the error is logged and
the next hook receives the original (pre-error) data.

## Permissions

Plugins must declare the permissions they need. Undeclared permissions
are not available at runtime.

| Permission | Description |
|-----------|-------------|
| `conversation_read` | Read conversation messages and metadata |
| `conversation_write` | Modify conversation messages |
| `model_config_read` | Read model configuration and parameters |
| `model_config_write` | Modify model configuration |
| `tool_register` | Register new tools |
| `pipeline_register` | Register pipeline steps |
| `ui_panel_register` | Register UI panel contributions |
| `filesystem_plugin_dir` | Read/write files within the plugin directory |
| `network_outbound` | Make outbound network requests |

## Sandbox Restrictions

Plugins run in a restricted environment for security:

### Blocked Imports

The following modules cannot be imported by plugins:
`subprocess`, `shutil`, `ctypes`, `multiprocessing`, `signal`,
`resource`, `pty`, `fcntl`, `termios`, `readline`, `code`,
`codeop`, `compileall`, `py_compile`.

### Network Modules

Network-related modules (`socket`, `http`, `urllib`, `requests`,
`httpx`, `aiohttp`, etc.) are blocked unless the plugin declares
the `network_outbound` permission.

### Filesystem Access

During plugin loading, file access is restricted to:
- The plugin's own directory
- Python standard library and site-packages

Plugins cannot read or write files outside their directory without
the `filesystem_plugin_dir` permission, and even then only within
their own plugin directory.

## Plugin Lifecycle

1. **Install**: Plugin directory is copied to the plugins folder and
   registered in the database.
2. **Enable**: Plugin code is loaded and `init()` is called.
3. **Running**: Hooks are called during normal operation.
4. **Disable**: `shutdown()` is called, hooks are unregistered.
5. **Uninstall**: Plugin is removed from the database and optionally
   from disk.

### init() and shutdown()

Your entry point can define `init()` and `shutdown()` functions:

```python
def init():
    """Called when the plugin is enabled."""
    # Load resources, open connections, etc.
    pass

def shutdown():
    """Called when the plugin is disabled."""
    # Clean up resources, close connections, etc.
    pass
```

## Configuration

Plugins can define a configuration schema in the manifest:

```yaml
config_schema:
  api_key:
    type: "string"
    default: ""
    description: "API key for the external service"
  max_retries:
    type: "integer"
    default: 3
    description: "Maximum retry attempts"
  verbose:
    type: "boolean"
    default: false
    description: "Enable verbose logging"
```

Users can edit plugin configuration in **Settings > Plugins > [plugin] > Config**.
At runtime, access configuration via the `PLUGIN_CONFIG` variable or read
it from the host API.

## Example: Word Counter Plugin

A complete example that counts words in model responses:

**manifest.yaml:**
```yaml
name: "word-counter"
version: "1.0.0"
author: "Example Author"
description: "Counts words in model responses and appends the count."
entry_point: "entry_point.py"
hooks:
  - post_inference
permissions:
  - conversation_read
config_schema:
  show_count:
    type: "boolean"
    default: true
    description: "Append word count to responses"
```

**entry_point.py:**
```python
"""word-counter -- Counts words in model responses."""

PLUGIN_CONFIG = {}


def init():
    """Initialize the word-counter plugin."""
    pass


def shutdown():
    """Clean up the word-counter plugin."""
    pass


def hook_post_inference(data: dict) -> dict:
    """Append word count to the model response."""
    show = PLUGIN_CONFIG.get("show_count", True)
    if not show:
        return data

    response = data.get("response", "")
    word_count = len(response.split())
    data["response"] = f"{response}\n\n[Words: {word_count}]"
    return data


HOOKS = {
    "post_inference": hook_post_inference,
}
```

## Publishing to the Marketplace

To share your plugin with other Opti-Oignon users:

1. Host your plugin in a public GitHub repository
2. Include `manifest.yaml`, `entry_point.py`, and `README.md` at the root
3. Add your plugin to the community index (submit a PR to the index
   repository adding your plugin entry to `index.json`)

### Index Entry Format

```json
{
  "name": "word-counter",
  "version": "1.0.0",
  "description": "Counts words in model responses.",
  "author": "Example Author",
  "url": "https://github.com/example/word-counter",
  "tags": ["utility", "text-analysis"],
  "hooks": ["post_inference"],
  "permissions": ["conversation_read"],
  "sha256": "",
  "stars": 0,
  "downloads": 0
}
```

## Installing Plugins

### From the Marketplace UI

1. Go to **Settings > Plugins > Marketplace**
2. Browse or search for plugins
3. Click **Install** on the plugin card

### From a URL

1. Go to **Settings > Plugins > Marketplace**
2. Click **Install from URL**
3. Paste the GitHub repository URL or direct archive link
4. Optionally provide a SHA-256 hash for verification
5. Click **Install**

### From a Local Directory

1. Go to **Settings > Plugins > Installed**
2. Click **Install Plugin**
3. Enter the path to the plugin directory
4. Click **Install**

## Testing Your Plugin

Before publishing, test your plugin locally:

1. Create your plugin directory with manifest and entry point
2. Install it via Settings > Plugins > Install Plugin
3. Enable it and verify hooks work as expected
4. Check the application logs for any sandbox violations or errors
5. Test edge cases (empty responses, missing data keys, etc.)

### Testing Tips

- Always return the `data` dict from hooks, even if unmodified
- Handle missing keys gracefully with `.get()` and defaults
- Keep hook execution fast (under 100ms ideally)
- Log warnings instead of raising exceptions in hooks
- Test with multiple models to ensure compatibility

## Troubleshooting

**Plugin fails to load**: Check that `manifest.yaml` is valid YAML, all
required fields are present, and the entry point file exists.

**Sandbox violation**: Your code is trying to import a blocked module or
access a restricted file path. Check the sandbox restrictions section above.

**Hook not executing**: Verify the hook name in `manifest.yaml` matches
either the function name (`hook_<name>`) or the key in the `HOOKS` dict.

**Configuration not available**: Make sure `config_schema` is defined in
the manifest and the plugin has been enabled (not just installed).
