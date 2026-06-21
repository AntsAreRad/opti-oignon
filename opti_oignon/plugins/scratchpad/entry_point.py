"""
Scratchpad plugin for Opti-Oignon.

Persistent note-taking via slash commands and a UI panel.
Notes are stored in a local SQLite database within the plugin
directory (requires filesystem_plugin_dir permission).

Commands:
    /note <text>           Save a new note
    /notes                 List all notes
    /note delete <id>      Delete a note by ID
    /note search <query>   Search notes by keyword
    /note export           Export all notes as markdown
"""

import logging
import re
import sqlite3

# S136 audit fix
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    _safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)
import time
from pathlib import Path
from typing import Any

__plugin_name__: str = "scratchpad"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_MAX_NOTE_LENGTH = 2000
_MAX_NOTES = 500
_AUTO_TAG = True

# =========================================================================
# Tag extraction
# =========================================================================

# Stop words to exclude from auto-tagging
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "this",
    "that", "these", "those", "it", "its", "and", "but", "or",
    "not", "no", "if", "then", "else", "when", "where", "how",
    "what", "which", "who", "whom", "why", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "than",
    "too", "very", "just", "about", "above", "also", "here", "there",
})


def extract_tags(text: str, max_tags: int = 5) -> list[str]:
    """Extract keyword tags from note text.

    Selects the longest non-stop-word tokens as tags.

    Parameters
    ----------
    text : str
        Note content.
    max_tags : int
        Maximum number of tags to extract.

    Returns
    -------
    list[str]
        Extracted tag strings, lowercase.
    """
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    candidates = [w for w in words if w not in _STOP_WORDS]
    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for w in candidates:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    # Sort by length descending, take top N
    unique.sort(key=len, reverse=True)
    return unique[:max_tags]


# =========================================================================
# SQLite database layer
# =========================================================================

class ScratchpadDB:
    """SQLite-backed note storage.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    max_notes : int
        Maximum number of notes to store.
    """

    def __init__(self, db_path: str | Path, max_notes: int = _MAX_NOTES) -> None:
        self.db_path = str(db_path)
        self.max_notes = max_notes
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = _safe_connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()

    def add_note(
        self,
        text: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add a new note.

        Returns
        -------
        dict
            The created note record with id, text, tags, created_at.
        """
        conn = self._get_conn()

        # Check note limit
        count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        if count >= self.max_notes:
            return {
                "error": f"Note limit reached ({self.max_notes}). "
                         "Delete some notes first.",
            }

        now = time.time()
        tags_str = ",".join(tags) if tags else ""
        cursor = conn.execute(
            "INSERT INTO notes (text, tags, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (text, tags_str, now, now),
        )
        conn.commit()

        return {
            "id": cursor.lastrowid,
            "text": text,
            "tags": tags or [],
            "created_at": now,
        }

    def list_notes(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all notes, most recent first.

        Parameters
        ----------
        limit : int
            Maximum number of notes to return.

        Returns
        -------
        list[dict]
            List of note records.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, text, tags, created_at FROM notes "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "tags": row["tags"].split(",") if row["tags"] else [],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete_note(self, note_id: int) -> bool:
        """Delete a note by ID.

        Returns True if a note was deleted, False if not found.
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()
        return cursor.rowcount > 0

    def search_notes(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search notes by text content or tags.

        Parameters
        ----------
        query : str
            Search query (matched with LIKE).
        limit : int
            Maximum results to return.

        Returns
        -------
        list[dict]
            Matching note records.
        """
        conn = self._get_conn()
        pattern = f"%{query}%"
        rows = conn.execute(
            "SELECT id, text, tags, created_at FROM notes "
            "WHERE text LIKE ? OR tags LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "tags": row["tags"].split(",") if row["tags"] else [],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def export_markdown(self) -> str:
        """Export all notes as markdown text.

        Returns
        -------
        str
            Markdown-formatted export of all notes.
        """
        notes = self.list_notes(limit=self.max_notes)
        if not notes:
            return "No notes saved."

        lines: list[str] = ["# Scratchpad Export", ""]
        for note in notes:
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(note["created_at"])
            )
            tags_str = ", ".join(note["tags"]) if note["tags"] else "none"
            lines.append(f"## Note #{note['id']} ({timestamp})")
            lines.append(f"**Tags:** {tags_str}")
            lines.append("")
            lines.append(note["text"])
            lines.append("")
        return "\n".join(lines)

    def get_note_count(self) -> int:
        """Return total number of stored notes."""
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =========================================================================
# Module-level DB instance (lazy init)
# =========================================================================

_db: ScratchpadDB | None = None


def _get_db(ctx: Any) -> ScratchpadDB:
    """Get or create the scratchpad database.

    Uses the plugin directory from ctx.metadata if available,
    otherwise falls back to a temp location.
    """
    global _db
    if _db is not None:
        return _db

    plugin_dir = ctx.metadata.get("plugin_dir", "")
    if plugin_dir:
        db_path = Path(plugin_dir) / "scratchpad.db"
    else:
        import tempfile
        db_path = Path(tempfile.gettempdir()) / "opti_scratchpad.db"

    max_notes = ctx.config.get("max_notes", _MAX_NOTES)
    _db = ScratchpadDB(db_path, max_notes=max_notes)
    logger.debug("Scratchpad DB initialized at %s", db_path)
    return _db


# =========================================================================
# Command parsing
# =========================================================================

# /note <text> | /notes | /note delete <id> | /note search <query> | /note export
_CMD_NOTE_ADD = re.compile(r"^/note\s+(?!delete\b|search\b|export\b)(.+)$", re.DOTALL)
_CMD_NOTES_LIST = re.compile(r"^/notes\s*$")
_CMD_NOTE_DELETE = re.compile(r"^/note\s+delete\s+(\d+)\s*$")
_CMD_NOTE_SEARCH = re.compile(r"^/note\s+search\s+(.+)$")
_CMD_NOTE_EXPORT = re.compile(r"^/note\s+export\s*$")


def _format_note_list(notes: list[dict[str, Any]], title: str = "Notes") -> str:
    """Format a list of notes for display."""
    if not notes:
        return f"**{title}:** No notes found."

    lines = [f"**{title}** ({len(notes)} note{'s' if len(notes) != 1 else ''}):", ""]
    for note in notes:
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(note["created_at"])
        )
        tags_str = f" [{', '.join(note['tags'])}]" if note["tags"] else ""
        preview = note["text"][:80]
        if len(note["text"]) > 80:
            preview += "..."
        lines.append(f"- **#{note['id']}** ({timestamp}){tags_str}: {preview}")
    return "\n".join(lines)


# =========================================================================
# Hook implementations
# =========================================================================

def hook_tool_call(ctx: Any) -> dict[str, Any] | None:
    """Tool call hook: handle scratchpad slash commands.

    Detects /note, /notes, /note delete, /note search, /note export
    commands and executes the corresponding action.
    """
    # Get the user input text
    tool_name = ctx.data.get("tool_name", "")  # noqa: F841
    user_input = ctx.data.get("user_input", "") or ctx.data.get("prompt", "")

    if not user_input:
        return None

    user_input = user_input.strip()

    # Check if this is a scratchpad command
    if not user_input.startswith("/note"):
        return None

    config = ctx.config or {}
    max_note_length = config.get("max_note_length", _MAX_NOTE_LENGTH)
    auto_tag = config.get("auto_tag", _AUTO_TAG)

    db = _get_db(ctx)

    # /notes — list all
    if _CMD_NOTES_LIST.match(user_input):
        notes = db.list_notes()
        return {
            "response": _format_note_list(notes, "Scratchpad"),
            "handled": True,
        }

    # /note delete <id>
    m = _CMD_NOTE_DELETE.match(user_input)
    if m:
        note_id = int(m.group(1))
        deleted = db.delete_note(note_id)
        if deleted:
            return {
                "response": f"Note #{note_id} deleted.",
                "handled": True,
            }
        else:
            return {
                "response": f"Note #{note_id} not found.",
                "handled": True,
            }

    # /note search <query>
    m = _CMD_NOTE_SEARCH.match(user_input)
    if m:
        query = m.group(1).strip()
        results = db.search_notes(query)
        return {
            "response": _format_note_list(results, f"Search results for '{query}'"),
            "handled": True,
        }

    # /note export
    if _CMD_NOTE_EXPORT.match(user_input):
        export = db.export_markdown()
        return {
            "response": export,
            "handled": True,
        }

    # /note <text> — add new note
    m = _CMD_NOTE_ADD.match(user_input)
    if m:
        text = m.group(1).strip()
        if len(text) > max_note_length:
            return {
                "response": (
                    f"Note too long ({len(text)} chars, "
                    f"max {max_note_length}). Please shorten it."
                ),
                "handled": True,
            }

        tags = extract_tags(text) if auto_tag else []
        result = db.add_note(text, tags=tags)

        if "error" in result:
            return {"response": result["error"], "handled": True}

        tags_str = f" (tags: {', '.join(tags)})" if tags else ""
        return {
            "response": f"Note #{result['id']} saved{tags_str}.",
            "handled": True,
        }

    return None


def hook_ui_panel(ctx: Any) -> dict[str, Any] | None:
    """UI panel hook: render scratchpad panel content.

    Returns HTML for displaying notes in a side panel.
    Uses CSS variables for styling (no hardcoded hex).
    """
    try:
        db = _get_db(ctx)
    except Exception:
        return {
            "panel_id": "scratchpad",
            "panel_title": "Scratchpad",
            "panel_html": "<p>Scratchpad unavailable.</p>",
        }

    notes = db.list_notes(limit=20)
    count = db.get_note_count()

    if not notes:
        body = "<p>No notes yet. Use <code>/note your text here</code> to save a note.</p>"
    else:
        items: list[str] = []
        for note in notes:
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(note["created_at"])
            )
            preview = note["text"][:100]
            if len(note["text"]) > 100:
                preview += "..."
            tags_html = ""
            if note["tags"]:
                tag_spans = "".join(
                    f'<span style="'
                    f"display:inline-block;"
                    f"padding:1px 6px;"
                    f"margin:0 2px;"
                    f"border-radius:3px;"
                    f"font-size:0.75em;"
                    f"background:var(--oo-surface-2, rgba(176,125,86,0.15));"
                    f"color:var(--oo-text-secondary, inherit);"
                    f'">{tag}</span>'
                    for tag in note["tags"]
                )
                tags_html = f'<div style="margin-top:2px;">{tag_spans}</div>'
            items.append(
                f'<div style="'
                f"padding:8px;"
                f"margin-bottom:6px;"
                f"border-radius:4px;"
                f"background:var(--oo-surface-1, transparent);"
                f"border:1px solid var(--oo-border, rgba(128,128,128,0.2));"
                f'">'
                f'<div style="'
                f"display:flex;"
                f"justify-content:space-between;"
                f"font-size:0.8em;"
                f"color:var(--oo-text-muted, inherit);"
                f'">'
                f"<span>#{note['id']}</span>"
                f"<span>{timestamp}</span>"
                f"</div>"
                f"<div>{preview}</div>"
                f"{tags_html}"
                f"</div>"
            )
        body = "".join(items)
        if count > 20:
            body += (
                f'<p style="font-size:0.85em;color:var(--oo-text-muted,inherit);">'
                f"Showing 20 of {count} notes. Use /notes to see all."
                f"</p>"
            )

    html = (
        f'<div style="font-family:var(--oo-font-family, sans-serif);">'
        f"<h3>Scratchpad ({count} notes)</h3>"
        f"{body}"
        f"</div>"
    )

    return {
        "panel_id": "scratchpad",
        "panel_title": "Scratchpad",
        "panel_html": html,
    }


# =========================================================================
# Hook registry
# =========================================================================

HOOKS = {
    "tool_call": hook_tool_call,
    "ui_panel": hook_ui_panel,
}


def init() -> None:
    """Plugin initialization."""
    pass


def shutdown() -> None:
    """Plugin shutdown: close database connection."""
    global _db
    if _db is not None:
        _db.close()
        _db = None
