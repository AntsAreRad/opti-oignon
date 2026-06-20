# Keyboard Shortcuts

## Default shortcuts

| Shortcut | Action | Category |
|----------|--------|----------|
| `Ctrl+N` | New conversation | Navigation |
| `Ctrl+Enter` | Send message | Chat |
| `Ctrl+B` | Toggle sidebar | Navigation |
| `Ctrl+K` | Search conversations | Navigation |
| `Ctrl+,` | Open settings | Navigation |
| `Ctrl+Shift+T` | Toggle theme (dark/light) | UI |
| `Ctrl+Shift+E` | Export conversation | Chat |
| `?` | Show keyboard shortcuts | Help |
| `Escape` | Close dialog or panel | UI |


## Customizing shortcuts

You can rebind any shortcut to a different key combination:

1. Open **Settings > Advanced > Shortcuts** (or press `?`)
2. Click on the shortcut you want to change
3. Press the new key combination
4. The system validates the binding and warns about browser conflicts

Custom bindings are stored per-user and persist across sessions.


## Conflict detection

The shortcut registry automatically detects:

- **Internal conflicts** -- two actions bound to the same key combination
- **Browser conflicts** -- bindings that override browser defaults
  (e.g., `Ctrl+W` closes the tab)

Conflicts are shown as warnings when you edit a binding. You can still
override browser shortcuts if you explicitly confirm.


## Resetting to defaults

To reset a single shortcut, click the reset icon next to it. To reset
all shortcuts at once, use the **Reset All** button in the shortcuts
panel.


## API

Custom bindings can also be managed via the API:

```
GET  /api/shortcuts           # Get all current bindings
PUT  /api/shortcuts/custom    # Apply custom bindings
POST /api/shortcuts/reset     # Reset to defaults
```
