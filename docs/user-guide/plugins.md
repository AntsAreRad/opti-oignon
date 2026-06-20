# Plugins

## Overview

Opti-Oignon supports a hook-based plugin system. Plugins can modify
prompts, post-process responses, add custom tools, and integrate with
external services -- all while running in sandboxed subprocesses.


## Installing plugins

### From the marketplace

1. Open **Settings > Plugins > Marketplace**
2. Browse available plugins or search by name
3. Click **Install** on a plugin
4. The plugin is downloaded, verified against the allowlist, and loaded

### Manual installation

Place your plugin directory under the configured plugins path
(default: `plugins/` in the project root). Each plugin must contain
a `manifest.yaml` and an entry point file.

### From the API

```
POST /api/plugins/marketplace/install
{
  "name": "my-plugin",
  "version": "1.0.0"
}
```


## Managing plugins

- **Enable/disable** -- toggle plugins without uninstalling them
- **Configure** -- each plugin exposes a config schema defined in its
  manifest; configure via the UI or API
- **Review** -- view plugin ratings and reviews from the community
- **User config** -- per-user plugin preferences stored separately

Plugins are loaded at startup and can be hot-reloaded via the API.


## Plugin isolation

Plugins run in isolated subprocesses to prevent them from affecting
the main application:

- **Unix domain socket IPC** -- communication via HMAC-authenticated
  Unix sockets (primary mode)
- **Pipe-based IPC** -- lightweight alternative using stdin/stdout
  pipes with length-prefixed JSON (async mode, for short-lived calls)
- **bwrap sandbox** -- kernel-level namespace isolation when available
  (network disabled, filesystem restricted)
- **Timeout enforcement** -- configurable per-call timeout with
  SIGTERM then SIGKILL escalation
- **Allowlist** -- only plugins on the verified allowlist can be
  installed from the marketplace

See [Plugin Development](../plugin-development.md) for creating your
own plugins.
