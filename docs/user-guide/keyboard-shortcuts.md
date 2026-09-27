# Keyboard Shortcuts

## Default shortcuts

| Shortcut | Action | Category |
|----------|--------|----------|
| `Ctrl+N` | New chat | Navigation |
| `Ctrl+Enter` | Send the message | Chat |
| `Ctrl+B` | Show or hide the sidebar | Navigation |
| `Ctrl+K` | Search chats and commands (the command palette) | Navigation |
| `Ctrl+,` | Open Preferences | Navigation |
| `Ctrl+Shift+T` | Switch between day and night | UI |
| `Ctrl+Shift+E` | Export this chat | Chat |
| `?` | Show keyboard shortcuts | Help |
| `Escape` | Close this dialog | UI |

The names are the ones the command palette and the list of shortcuts (`?`)
show. While the palette or the list of shortcuts is open, a key runs only
what closes it (`Escape`, and `?` for the list): `Ctrl+Enter` typed in the
palette does not send the message under it.


## The command palette

`Ctrl+K`, or **Search** in the sidebar (on a phone, in the navigation
drawer), opens the command palette: one field that reaches every page of
both spaces, every command, every settings group and your chats. Type a few
letters: the best match comes first, and `Enter` opens the first entry that
can run where you are. The arrow keys move through the list (the entry you
move to stays chosen when the list changes under it, as the search of your
chats answers), `Home` and `End` go to its first and last entry, and
`Escape` closes the palette and returns you where you were. The palette says
how many results it lists, once you pause.

- **Go to** lists the pages. With nothing typed, the palette lists them all,
  then your recent chats.
- **Commands** are the actions above, each with the keys that run it for
  you (your own, if you changed them), and **Show notifications**, which
  opens the notification history in Preferences. A command that cannot run
  where you are stays in the list, greyed, with the reason beside it (for
  example, "Open a chat to export it"); opening it says why.
- **Settings** finds a settings group by its name, its other names, its
  description, the page that holds it and the section of the old settings
  page it sat in, in Preferences or the Workshop.
- **Chats** are searched on the server, in their titles and their messages.
  When more chats match than the palette shows, a last entry opens the
  Chats page on the same words, where every match is listed.

The palette keeps **Stop all** at its top, beside its close button, so the
emergency stop is one tap or click away while it is open, above the field,
where a phone's keyboard never covers it. Typing "stop" or "emergency" also
finds a **Stop all** entry: it opens the stop's confirmation, and nothing
stops until you choose.


## Customizing shortcuts

You can rebind any shortcut to a different key combination:

1. Open **Preferences > Keyboard > Keyboard shortcuts** (or press `?`)
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
