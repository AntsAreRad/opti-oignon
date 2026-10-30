#!/usr/bin/env python3
"""
PROJECT STORE -- Project Management with File Storage
==============================================================

SQLite-backed storage for projects, project files, project outputs,
and conversation linking. Provides CRUD operations, file upload
handling, and cascade deletion.

Prepares the data layer for RAG-based context injection.

Author: Leon
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
# Audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# =============================================================================
# CONSTANTS
# =============================================================================

_DATA_DIR = Path(__file__).parent / "data"
_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_DB_PATH = _DATA_DIR / "projects.db"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "projects.yaml"
_DEFAULT_STORAGE_BASE = _DATA_DIR / "projects"

# File type detection by extension
_FILE_TYPE_MAP: dict[str, str] = {}

# Every write of the store, and the indexer's landing of a file's chunks, hold this lock: a deletion never interleaves
# with a write that would bring back what it removes -- a file added, a file's chunks landed, as it goes.
_WRITES = threading.RLock()

# An id as ``Project``, ``ProjectFile`` and ``ProjectOutput`` make it: the sweep removes only what is named so.
_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{3}")

# What the account wipe's answer calls what a deletion could not remove (``store_inventory``).
_STORE_LABEL = "projects store (single-user)"
_FILES_LABEL = "project files and saved outputs"


@contextmanager
def holding():
    """Hold the store's write lock (``_WRITES``)."""
    with _WRITES:
        yield


def _plain(name: str) -> bool:
    """Whether ``name`` names one entry of a folder, and nothing beyond it."""
    return bool(name) and "/" not in name and "\\" not in name and name not in (".", "..") and "\x00" not in name


def _overwrite_and_remove(path: Path, said: list | None = None) -> bool:
    """Overwrite a file's bytes in place, then remove it; whether it was there. A filesystem that writes in place no
    longer holds them; a copy-on-write filesystem, a snapshot or a backup may. A link is refused, never followed.

    A file another link also names -- a backup made of links, say -- is removed without its overwrite: those bytes
    are the other copy's too. One this process may not write is made writable first, being the process's own; one
    that stays unwritable is removed without its overwrite. Each removal without its overwrite is said: in ``said``
    when it is given, in the log otherwise."""
    # Never blocks: a FIFO with no reader refuses the open rather than wait for one.
    nofollow = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    reason = None
    try:
        fd = os.open(path, os.O_WRONLY | nofollow)
    except FileNotFoundError:
        return False
    except PermissionError:
        try:
            fd, reason = _made_writable(path, nofollow)
        except FileNotFoundError:
            return False
    if fd is not None:
        try:
            held = os.fstat(fd)
            if not stat.S_ISREG(held.st_mode):
                reason = "not a regular file"
            elif held.st_nlink > 1:
                reason = "other links hold its bytes"
            else:
                left = held.st_size
                block = b"\0" * 65536
                while left > 0:
                    left -= os.write(fd, block[:min(len(block), left)])
                os.fsync(fd)
        finally:
            os.close(fd)
    os.unlink(path)
    if reason is not None:
        if said is None:
            logger.warning("a file removed without its overwrite (%s)", reason)
        else:
            said.append(reason)
    return True


def _made_writable(path: Path, nofollow: int) -> tuple:
    """(a descriptor that writes ``path``, None) once the file is made readable and writable by its owner -- it is
    this process's own --, or (None, why) when it cannot be; only a regular file no other link names is changed."""
    try:
        held = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as exc:
        return None, type(exc).__name__
    if not stat.S_ISREG(held.st_mode):
        return None, "not a regular file"
    if held.st_nlink > 1:
        return None, "other links hold its bytes"
    try:
        os.chmod(path, stat.S_IMODE(held.st_mode) | stat.S_IRUSR | stat.S_IWUSR)
        return os.open(path, os.O_WRONLY | nofollow), None
    except OSError as exc:
        return None, type(exc).__name__


# What a deletion says of a link it removed: the link goes, never what it leads to.
_LINK = "a link: what it led to is left"


def _forget_entry(path: Path, said: list) -> None:
    """Overwrite and remove one file; remove a link, never followed, a folder with what it holds, or anything else
    that is not a regular file. A link removed, and a file removed without its overwrite, said in ``said``."""
    if path.is_symlink():
        os.unlink(path)
        said.append(_LINK)
    elif path.is_dir():
        _forget_folder(path, said)
    elif not path.is_file():
        os.unlink(path)
    else:
        _overwrite_and_remove(path, said)


def _forget_folder(folder: Path, said: list) -> None:
    """Overwrite and remove every file under ``folder``, then its folders and itself: a link is removed, never
    followed; a link removed, and a file removed without its overwrite, said in ``said``. Each entry is tried; the
    first that fails raises once the others are done. A folder that is not there holds none."""
    if not os.path.lexists(folder):
        return
    if folder.is_symlink():
        os.unlink(folder)
        said.append(_LINK)
        return
    if not folder.is_dir():
        os.unlink(folder)
        return
    first: OSError | None = None
    for top, dirs, files in os.walk(folder, topdown=False, followlinks=False):
        for name in files:
            try:
                _forget_entry(Path(top) / name, said)
            except OSError as exc:
                first = first or exc
        for name in dirs:
            path = Path(top) / name
            try:
                if path.is_symlink():
                    os.unlink(path)
                    said.append(_LINK)
                else:
                    os.rmdir(path)
            except OSError as exc:
                first = first or exc
    try:
        os.rmdir(folder)
    except OSError as exc:
        first = first or exc
    if first is not None:
        raise first


def _forget_cached_answers(data: Path) -> list[str]:
    """The answers the response and semantic caches keep, emptied (``cached_answers``): any may have been drawn from
    what was deleted. A folder ``data`` that holds neither cache's file holds none of them, and nothing is loaded to
    be asked."""
    if not ((data / "response_cache.db").exists() or (data / "semantic_cache.db").exists()):
        return []
    try:
        from opti_oignon.cached_answers import forget_cached_answers
    except ImportError as exc:
        return [f"response and semantic caches (not emptied: {type(exc).__name__})"]
    return forget_cached_answers(data)


def _absent(error: ModuleNotFoundError, module: str) -> bool:
    """Whether an import failed because the module, or a package it would be in, is not in this tree -- then it holds
    nothing; a module that is there and fails to load is broken, and raises."""
    return bool(error.name) and (error.name == module or module.startswith(error.name + "."))


def _unwritten(said: list, what: str) -> list[str]:
    """What the answer says of files removed without their overwrite: one line per reason."""
    return [f"{_FILES_LABEL} ({what}: removed, not overwritten -- {reason})" for reason in dict.fromkeys(said)]


def _no_index(error: ModuleNotFoundError) -> bool:
    """Whether an import of the projects' index failed because this tree has none: then nothing is indexed; an index
    that is there and fails to load raises."""
    return _absent(error, "opti_oignon.project_context")


def _build_file_type_map(categories: dict[str, list[str]]) -> dict[str, str]:
    """Build extension -> category lookup from config categories."""
    result: dict[str, str] = {}
    for category, extensions in categories.items():
        for ext in extensions:
            result[ext.lower()] = category
    return result


# Default categories if config is missing
_DEFAULT_CATEGORIES = {
    "text": [".txt", ".md", ".rst", ".log"],
    "code": [
        ".py", ".r", ".js", ".ts", ".svelte", ".html", ".css",
        ".sh", ".yaml", ".yml", ".json", ".toml", ".xml", ".sql",
    ],
    "data": [".csv", ".tsv", ".xlsx", ".xls"],
    "document": [".pdf", ".docx", ".tex"],
    "image": [".png", ".jpg", ".jpeg", ".gif", ".svg"],
    "archive": [".zip", ".tar.gz", ".tgz"],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Project:
    """A project grouping conversations and files."""

    id: str = ""
    name: str = ""
    description: str = ""
    system_instructions: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        """Generate id and timestamps if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        now = _iso_now()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with settings as dict (not JSON string)."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        """Create from dictionary, handling settings JSON string."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in data.items():
            if k not in valid_keys:
                continue
            # settings may come as JSON string from SQLite
            if k == "settings" and isinstance(v, str):
                try:
                    filtered[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    filtered[k] = {}
            else:
                filtered[k] = v
        return cls(**filtered)


@dataclass
class ProjectFile:
    """A file belonging to a project."""

    id: str = ""
    project_id: str = ""
    filename: str = ""
    file_path: str = ""
    file_type: str = ""
    file_size_bytes: int = 0
    indexed: bool = False
    chunk_count: int = 0
    summary: str = ""
    key_terms: list[str] = field(default_factory=list)
    uploaded_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        """Generate id and timestamps if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        now = _iso_now()
        if not self.uploaded_at:
            self.uploaded_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectFile":
        """Create from dictionary, handling key_terms JSON string."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in data.items():
            if k not in valid_keys:
                continue
            if k == "key_terms" and isinstance(v, str):
                try:
                    filtered[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    filtered[k] = []
            elif k == "indexed" and isinstance(v, int):
                filtered[k] = bool(v)
            else:
                filtered[k] = v
        return cls(**filtered)


@dataclass
class ProjectOutput:
    """An output file produced within a project context."""

    id: str = ""
    project_id: str = ""
    source_conversation_id: str = ""
    filename: str = ""
    file_path: str = ""
    output_type: str = "code"
    description: str = ""
    created_at: str = ""

    def __post_init__(self):
        """Generate id and timestamp if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = _iso_now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectOutput":
        """Create from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# =============================================================================
# HELPERS
# =============================================================================

def _iso_now() -> str:
    """Return current UTC time in ISO 8601 format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def detect_file_type(filename: str, categories: dict[str, str] | None = None) -> str:
    """Detect file type category from filename extension.

    Args:
        filename: The filename to check.
        categories: Extension-to-category mapping. Uses global map if None.

    Returns:
        Category string (text, code, data, document, image, archive, unknown).
    """
    lookup = categories or _FILE_TYPE_MAP
    name_lower = filename.lower()
    # Check compound extensions first (e.g. .tar.gz)
    for ext in sorted(lookup.keys(), key=len, reverse=True):
        if name_lower.endswith(ext):
            return lookup[ext]
    return "unknown"


def extract_file_metadata(filepath: Path) -> dict[str, Any]:
    """Extract basic metadata from a file on disk.

    Args:
        filepath: Path to the file.

    Returns:
        Dict with size_bytes, line_count (for text files), etc.
    """
    meta: dict[str, Any] = {
        "size_bytes": 0,
        "line_count": None,
    }
    try:
        meta["size_bytes"] = filepath.stat().st_size
    except OSError:
        return meta

    # Attempt line count for text-like files
    file_type = detect_file_type(filepath.name)
    if file_type in ("text", "code", "data"):
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                meta["line_count"] = sum(1 for _ in f)
        except Exception:
            pass

    return meta


# =============================================================================
# PROJECT STORE
# =============================================================================

class ProjectStore:
    """SQLite-backed project storage with file management.

    Manages projects, their files, outputs, and conversation links.
    Files are stored on the local filesystem under a configurable
    base directory.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        config_path: Path | None = None,
        storage_base: Path | None = None,
        vectors_db: Path | None = None,
    ):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._storage_base = storage_base or _DEFAULT_STORAGE_BASE
        # The index of the projects' files: the knowledge base's store beside this store's database (``rag/``), the
        # indexer's default for the default store.
        self._vectors_db = (Path(vectors_db) if vectors_db is not None
                            else None if self._db_path == _DEFAULT_DB_PATH
                            else Path(self._db_path).parent / "rag" / "vectors.db")
        self._config: dict[str, Any] = {}

        # Load config
        self._load_config()

        # Build file type map from config
        global _FILE_TYPE_MAP
        categories = self._config.get("file_type_categories", _DEFAULT_CATEGORIES)
        _FILE_TYPE_MAP = _build_file_type_map(categories)

        # Ensure storage directory exists
        self._storage_base.mkdir(parents=True, exist_ok=True)

        # Initialize database
        # The schema is built at the first connection, not here. This store
        # is constructed at module scope, so building it here opened a
        # database on every import of the package -- with encryption not
        # enforced, at a moment where a refusal could not be handled.
        self._schema_ready = False

        logger.info("ProjectStore initialized (db=%s, storage=%s)", self._db_path, self._storage_base)

    def _load_config(self) -> None:
        """Load projects configuration from YAML."""
        try:
            if self._config_path.exists():
                with open(self._config_path) as f:
                    raw = yaml.safe_load(f) or {}
                self._config = raw.get("projects", {})
            else:
                self._config = {}
        except Exception as e:
            logger.warning("Failed to load projects config: %s", e)
            self._config = {}

    # -- Config properties --

    @property
    def enabled(self) -> bool:
        """Whether the projects feature is enabled."""
        return self._config.get("enabled", True)

    @property
    def max_projects(self) -> int:
        """Maximum number of projects allowed."""
        return self._config.get("max_projects", 50)

    @property
    def max_files_per_project(self) -> int:
        """Maximum files per project."""
        return self._config.get("max_files_per_project", 100)

    @property
    def max_file_size_mb(self) -> int:
        """Maximum file size in megabytes."""
        return self._config.get("max_file_size_mb", 50)

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def allowed_extensions(self) -> list[str]:
        """List of allowed file extensions (empty = allow all)."""
        return self._config.get("allowed_extensions", [])

    @property
    def default_settings(self) -> dict[str, Any]:
        """Default settings for new projects."""
        return self._config.get("default_settings", {
            "default_model": "",
            "default_pipeline": "direct",
            "context_budget_tokens": 4096,
            "auto_index": True,
        })

    # -- Database connection --

    def _get_conn(self) -> sqlite3.Connection:
        """Get a SQLite connection with WAL mode."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # What any write frees is overwritten, never left in the file's free pages.
        conn.execute("PRAGMA secure_delete = ON")
        if not self._schema_ready:
            self._schema_ready = True
            self._init_db()
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    system_instructions TEXT DEFAULT '',
                    settings TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_files (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT DEFAULT '',
                    file_size_bytes INTEGER DEFAULT 0,
                    indexed BOOLEAN DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    summary TEXT DEFAULT '',
                    key_terms TEXT DEFAULT '[]',
                    uploaded_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_outputs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    source_conversation_id TEXT DEFAULT '',
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    output_type TEXT DEFAULT 'code',
                    description TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_conversations (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    conversation_id TEXT NOT NULL,
                    linked_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (project_id, conversation_id)
                )
            """)

            # Indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_project_files_project
                ON project_files(project_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_project_outputs_project
                ON project_outputs(project_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_project_conversations_project
                ON project_conversations(project_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_project_conversations_conv
                ON project_conversations(conversation_id)
            """)

            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # PROJECT CRUD
    # =========================================================================

    def create_project(
        self,
        name: str,
        description: str = "",
        system_instructions: str = "",
        settings: dict[str, Any] | None = None,
    ) -> Project:
        """Create a new project.

        Args:
            name: Project name.
            description: Optional description.
            system_instructions: System prompt for project conversations.
            settings: Project-specific settings (merged with defaults).

        Returns:
            The created Project.

        Raises:
            ValueError: If name is empty or project limit reached.
        """
        if not name or not name.strip():
            raise ValueError("Project name cannot be empty")

        # Check project limit
        existing_count = len(self.list_projects())
        if existing_count >= self.max_projects:
            raise ValueError(
                f"Maximum number of projects reached ({self.max_projects})"
            )

        # Merge settings with defaults
        merged_settings = dict(self.default_settings)
        if settings:
            merged_settings.update(settings)

        project = Project(
            name=name.strip(),
            description=description.strip(),
            system_instructions=system_instructions.strip(),
            settings=merged_settings,
        )

        with _WRITES:
            # Create project directory
            project_dir = self._storage_base / project.id
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "files").mkdir(exist_ok=True)
            (project_dir / "outputs").mkdir(exist_ok=True)

            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO projects (id, name, description, system_instructions, settings, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project.id,
                        project.name,
                        project.description,
                        project.system_instructions,
                        json.dumps(project.settings),
                        project.created_at,
                        project.updated_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        logger.info("Created project %s", project.id)
        return project

    def get_project(self, project_id: str) -> Project | None:
        """Get a project by ID.

        Args:
            project_id: The project ID.

        Returns:
            Project if found, None otherwise.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if row is None:
                return None
            return Project.from_dict(dict(row))
        finally:
            conn.close()

    def list_projects(self) -> list[Project]:
        """List all projects ordered by updated_at descending.

        Returns:
            List of Project objects.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC"
            ).fetchall()
            return [Project.from_dict(dict(r)) for r in rows]
        finally:
            conn.close()

    def update_project(
        self,
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        system_instructions: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Project | None:
        """Update an existing project.

        Only non-None fields are updated. Settings are merged (not replaced).

        Args:
            project_id: The project ID.
            name: New name (if provided).
            description: New description (if provided).
            system_instructions: New system instructions (if provided).
            settings: Settings to merge (if provided).

        Returns:
            Updated Project if found, None otherwise.
        """
        project = self.get_project(project_id)
        if project is None:
            return None

        if name is not None:
            if not name.strip():
                raise ValueError("Project name cannot be empty")
            project.name = name.strip()
        if description is not None:
            project.description = description.strip()
        if system_instructions is not None:
            project.system_instructions = system_instructions.strip()
        if settings is not None:
            project.settings.update(settings)

        project.updated_at = _iso_now()

        conn = self._get_conn()
        try:
            conn.execute(
                """
                UPDATE projects
                SET name = ?, description = ?, system_instructions = ?,
                    settings = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    project.name,
                    project.description,
                    project.system_instructions,
                    json.dumps(project.settings),
                    project.updated_at,
                    project.id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info("Updated project %s", project.id)
        return project

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all associated data, keeping none of its words (``forget_project``).

        Args:
            project_id: The project ID.

        Returns:
            True if the project existed and was deleted.
        """
        return self.forget_project(project_id)[0]

    def forget_project(self, project_id: str, *, sweep: bool = True, empty: bool = True) -> tuple[bool, list[str]]:
        """Delete a project, keeping none of its words: its index first (every chunk of its files, their metadata and
        vectors), then every file and output under its folder overwritten and removed, its rows -- files, outputs and
        conversation links by cascade; the conversations stay --, its folder, and both logs emptied.

        Whether it existed, and what could not be removed, named: the deletion stands, and the next deletion's
        sweep (``sweep_orphans``) removes what it left. With ``sweep``, what earlier deletions left goes too; without
        ``empty``, the logs are left to the caller (``empty_logs``), as the wipe does once at its end.
        """
        with _WRITES:
            if self.get_project(project_id) is None:
                return False, []
            left = self._forget_index(project_id)
            # What this deletion could not reach is not swept again now -- the next deletion does --, so the answer
            # names each failure once, and never one a sweep then mended.
            reached, on_disk, said = not left, True, []
            if _plain(project_id):
                try:
                    _forget_folder(self._storage_base / project_id, said)
                except OSError as exc:
                    logger.warning("project %s: its files could not all be removed: %s", project_id,
                                   type(exc).__name__)
                    left.append(f"{_FILES_LABEL} (project {project_id}'s, not all removed: {type(exc).__name__})")
                    on_disk = False
            left += _unwritten(said, f"project {project_id}'s")
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
                conn.commit()
            finally:
                conn.close()
            # The cached answers: any may have been drawn from the project's files or instructions.
            left += _forget_cached_answers(Path(self._db_path).parent)
            if sweep:
                left += self.sweep_orphans(index=reached, disk=on_disk)
            if empty:
                left += self.empty_logs()
        logger.info("Deleted project id=%s", project_id)
        return True, list(dict.fromkeys(left))

    # =========================================================================
    # FILE MANAGEMENT
    # =========================================================================

    def add_file(
        self,
        project_id: str,
        filename: str,
        content: bytes,
    ) -> ProjectFile:
        """Add a file to a project.

        Validates the file, writes it to disk, and registers it in the DB.

        Args:
            project_id: The project ID.
            filename: Original filename.
            content: File content as bytes.

        Returns:
            The created ProjectFile.

        Raises:
            ValueError: If project not found, limits exceeded, or invalid file.
        """
        # Validate project exists
        project = self.get_project(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        # Validate file count limit
        existing_files = self.list_files(project_id)
        if len(existing_files) >= self.max_files_per_project:
            raise ValueError(
                f"Maximum files per project reached ({self.max_files_per_project})"
            )

        # Validate file size
        if len(content) > self.max_file_size_bytes:
            raise ValueError(
                f"File exceeds maximum size ({self.max_file_size_mb} MB)"
            )

        # Validate extension
        if self.allowed_extensions:
            name_lower = filename.lower()
            ext_ok = any(name_lower.endswith(ext) for ext in self.allowed_extensions)
            if not ext_ok:
                raise ValueError(
                    f"File extension not allowed: {filename}"
                )

        # Sanitize filename (remove path separators, null bytes)
        safe_name = os.path.basename(filename).replace("\x00", "")
        if not safe_name:
            safe_name = "unnamed_file"

        # Detect file type
        file_type = detect_file_type(safe_name)

        # Generate unique file ID and write to disk
        file_id = str(uuid.uuid4())[:12]

        # The bytes and their row land under the write lock, the project read again there: a deletion that came
        # since leaves no file on disk that no row names.
        with _WRITES:
            if self.get_project(project_id) is None:
                raise ValueError(f"Project not found: {project_id}")
            project_files_dir = self._storage_base / project_id / "files"
            project_files_dir.mkdir(parents=True, exist_ok=True)

            # Use file_id prefix to avoid name collisions
            stored_name = f"{file_id}_{safe_name}"
            file_path = project_files_dir / stored_name

            try:
                with open(file_path, "wb") as f:
                    f.write(content)
            except OSError as e:
                raise ValueError(f"Failed to write file to disk: {e}")

            # Extract metadata
            meta = extract_file_metadata(file_path)

            pf = ProjectFile(
                id=file_id,
                project_id=project_id,
                filename=safe_name,
                file_path=str(file_path),
                file_type=file_type,
                file_size_bytes=meta["size_bytes"],
            )

            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO project_files
                        (id, project_id, filename, file_path, file_type,
                         file_size_bytes, indexed, chunk_count, summary,
                         key_terms, uploaded_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pf.id, pf.project_id, pf.filename, pf.file_path,
                        pf.file_type, pf.file_size_bytes, int(pf.indexed),
                        pf.chunk_count, pf.summary, json.dumps(pf.key_terms),
                        pf.uploaded_at, pf.updated_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            # Touch project updated_at
            self._touch_project(project_id)

        logger.info(
            "Added file %s to project %s (type=%s, %d bytes)",
            file_id, project_id, file_type, pf.file_size_bytes,
        )
        return pf

    def get_file(self, file_id: str) -> ProjectFile | None:
        """Get a project file by ID.

        Args:
            file_id: The file ID.

        Returns:
            ProjectFile if found, None otherwise.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM project_files WHERE id = ?", (file_id,)
            ).fetchone()
            if row is None:
                return None
            return ProjectFile.from_dict(dict(row))
        finally:
            conn.close()

    def list_files(self, project_id: str) -> list[ProjectFile]:
        """List all files for a project.

        Args:
            project_id: The project ID.

        Returns:
            List of ProjectFile objects ordered by uploaded_at.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM project_files WHERE project_id = ? ORDER BY uploaded_at DESC",
                (project_id,),
            ).fetchall()
            return [ProjectFile.from_dict(dict(r)) for r in rows]
        finally:
            conn.close()

    def remove_file(self, file_id: str) -> bool:
        """Remove a file from its project, keeping none of its words (``forget_file``).

        Args:
            file_id: The file ID.

        Returns:
            True if the file existed and was removed.
        """
        return self.forget_file(file_id)[0]

    def forget_file(self, file_id: str, *, sweep: bool = True) -> tuple[bool, list[str]]:
        """Remove a file from its project, keeping none of its words: its chunks first, so no turn retrieves them
        again, then its bytes overwritten and removed, its row (its summary and key terms), and both logs emptied.
        Whether it existed, and what could not be removed, named; with ``sweep``, what earlier deletions left goes
        too (``sweep_orphans``)."""
        with _WRITES:
            pf = self.get_file(file_id)
            if pf is None:
                return False, []
            left = self._forget_chunks(pf.project_id, {pf.file_path})
            reached = not left
            stored = self._forget_stored(pf.project_id, "files", file_id)
            left += stored
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM project_files WHERE id = ?", (file_id,))
                conn.commit()
            finally:
                conn.close()
            self._touch_project(pf.project_id)
            # The cached answers: any may have been drawn from the file.
            left += _forget_cached_answers(Path(self._db_path).parent)
            if sweep:
                left += self.sweep_orphans(index=reached, disk=not [line for line in stored if "not removed" in line])
            left += self.empty_logs()
        logger.info("Removed file id=%s from project %s", file_id, pf.project_id)
        return True, list(dict.fromkeys(left))

    def read_file_content(self, file_id: str) -> bytes | None:
        """Read file content from disk.

        Args:
            file_id: The file ID.

        Returns:
            File content as bytes, or None if file not found.
        """
        pf = self.get_file(file_id)
        if pf is None:
            return None
        return self._read_stored(pf.project_id, "files", file_id)

    # =========================================================================
    # OUTPUT MANAGEMENT
    # =========================================================================

    def add_output(
        self,
        project_id: str,
        filename: str,
        content: bytes,
        output_type: str = "code",
        description: str = "",
        source_conversation_id: str = "",
    ) -> ProjectOutput:
        """Add an output file to a project.

        Args:
            project_id: The project ID.
            filename: Output filename.
            content: File content as bytes.
            output_type: Type of output (code, report, data, etc.).
            description: Optional description.
            source_conversation_id: Conversation that produced this output.

        Returns:
            The created ProjectOutput.

        Raises:
            ValueError: If project not found or write fails.
        """
        project = self.get_project(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        safe_name = os.path.basename(filename).replace("\x00", "")
        if not safe_name:
            safe_name = "output_file"

        output_id = str(uuid.uuid4())[:12]

        # As for a file: the bytes and their row land under the write lock, the project read again there.
        with _WRITES:
            if self.get_project(project_id) is None:
                raise ValueError(f"Project not found: {project_id}")
            outputs_dir = self._storage_base / project_id / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            stored_name = f"{output_id}_{safe_name}"
            file_path = outputs_dir / stored_name

            try:
                with open(file_path, "wb") as f:
                    f.write(content)
            except OSError as e:
                raise ValueError(f"Failed to write output to disk: {e}")

            po = ProjectOutput(
                id=output_id,
                project_id=project_id,
                source_conversation_id=source_conversation_id,
                filename=safe_name,
                file_path=str(file_path),
                output_type=output_type,
                description=description,
            )

            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO project_outputs
                        (id, project_id, source_conversation_id, filename,
                         file_path, output_type, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        po.id, po.project_id, po.source_conversation_id,
                        po.filename, po.file_path, po.output_type,
                        po.description, po.created_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._touch_project(project_id)
        logger.info("Added output %s to project %s", output_id, project_id)
        return po

    def get_output(self, output_id: str) -> ProjectOutput | None:
        """Get a project output by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM project_outputs WHERE id = ?", (output_id,)
            ).fetchone()
            if row is None:
                return None
            return ProjectOutput.from_dict(dict(row))
        finally:
            conn.close()

    def list_outputs(self, project_id: str) -> list[ProjectOutput]:
        """List all outputs for a project."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM project_outputs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
            return [ProjectOutput.from_dict(dict(r)) for r in rows]
        finally:
            conn.close()

    def read_output_content(self, output_id: str) -> bytes | None:
        """An output's bytes, or None when it is not there."""
        po = self.get_output(output_id)
        if po is None:
            return None
        return self._read_stored(po.project_id, "outputs", output_id)

    def remove_output(self, output_id: str) -> bool:
        """Remove an output file from its project, keeping none of its words (``forget_output``)."""
        return self.forget_output(output_id)[0]

    def forget_output(self, output_id: str) -> tuple[bool, list[str]]:
        """Remove an output from its project: its bytes overwritten and removed, its row (its description), the log
        emptied. Whether it existed, and what could not be removed, named."""
        with _WRITES:
            po = self.get_output(output_id)
            if po is None:
                return False, []
            left = self._forget_stored(po.project_id, "outputs", output_id)
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM project_outputs WHERE id = ?", (output_id,))
                conn.commit()
            finally:
                conn.close()
            self._touch_project(po.project_id)
            left += self.empty_logs(index=False)
        logger.info("Removed output id=%s from project %s", output_id, po.project_id)
        return True, left

    # =========================================================================
    # FORGETTING
    # =========================================================================

    @property
    def present(self) -> bool:
        """Whether the store's database is there: one that is not holds nothing, and reading it would make it."""
        return Path(self._db_path).exists()

    def project_ids(self) -> list[str]:
        """Every project's id; none, and nothing made, when the store is not there."""
        if not self.present:
            return []
        conn = self._get_conn()
        try:
            return [row["id"] for row in conn.execute("SELECT id FROM projects ORDER BY id")]
        finally:
            conn.close()

    def counts(self) -> dict[str, int]:
        """How many projects, files and outputs the store holds; zeros, and nothing made, when it is not there."""
        if not self.present:
            return {"projects": 0, "files": 0, "outputs": 0}
        conn = self._get_conn()
        try:
            return {key: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for key, table in (("projects", "projects"), ("files", "project_files"),
                                       ("outputs", "project_outputs"))}
        finally:
            conn.close()

    def fingerprint(self) -> str:
        """Four hex characters drawn from every project, file and output id: a project, file or output that came
        while another went, the counts unchanged, changes them. Empty when the store holds nothing."""
        if not self.present:
            return ""
        conn = self._get_conn()
        try:
            ids = [f"{kind}:{row[0]}" for kind, table in (("p", "projects"), ("f", "project_files"),
                                                         ("o", "project_outputs"))
                   for row in conn.execute(f"SELECT id FROM {table} ORDER BY id")]
        finally:
            conn.close()
        return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:4] if ids else ""

    def sweep_orphans(self, *, index: bool = True, disk: bool = True) -> list[str]:
        """What earlier deletions left, removed: in the index (unless ``index`` is False), the collections of projects
        that are gone and the chunks of files that are gone; on disk (unless ``disk`` is False), the folders named
        like a project that is gone, and in each project's folder the files and outputs no row names -- each
        overwritten before it goes; what could not be removed, named. A store that is not there holds no project:
        everything of the projects' that is left goes."""
        with _WRITES:
            live: dict[str, set[str]] = {}
            stored: dict[str, dict[str, set[str]]] = {}
            if self.present:
                conn = self._get_conn()
                try:
                    for row in conn.execute("SELECT id FROM projects"):
                        live[row["id"]] = set()
                        stored[row["id"]] = {"files": set(), "outputs": set()}
                    for row in conn.execute("SELECT id, project_id, file_path FROM project_files"):
                        live.setdefault(row["project_id"], set()).add(row["file_path"])
                        stored.setdefault(row["project_id"], {"files": set(), "outputs": set()})["files"].add(
                            row["id"])
                    for row in conn.execute("SELECT id, project_id FROM project_outputs"):
                        stored.setdefault(row["project_id"], {"files": set(), "outputs": set()})["outputs"].add(
                            row["id"])
                finally:
                    conn.close()
            left = self._sweep_index(live) if index else []
            failed: OSError | None = None
            said: list = []
            try:
                children = (sorted(self._storage_base.iterdir()) if disk and self._storage_base.is_dir()
                            else [])
            except OSError as exc:
                children, failed = [], exc
            for child in children:
                if not _ID.fullmatch(child.name):
                    continue
                try:
                    if child.name not in stored:
                        _forget_folder(child, said)
                        continue
                    for kind in ("files", "outputs"):
                        folder = child / kind
                        if folder.is_symlink() or not folder.is_dir():
                            continue
                        for entry in sorted(folder.iterdir()):
                            owner = entry.name.split("_", 1)[0]
                            if _ID.fullmatch(owner) and owner not in stored[child.name][kind]:
                                _forget_entry(entry, said)
                except OSError as exc:
                    failed = failed or exc
            if failed is not None:
                logger.warning("projects: what earlier deletions left on disk could not all be removed: %s",
                               type(failed).__name__)
                left.append(f"{_FILES_LABEL} (what earlier deletions left, not all removed: {type(failed).__name__})")
            left += _unwritten(said, "what earlier deletions left")
        return left

    def _forget_index(self, project_id: str) -> list[str]:
        """The project's index removed (``ProjectIndexer.forget_project_index``); what stays, named."""
        try:
            from opti_oignon.project_context import ProjectIndexer
        except ModuleNotFoundError as e:
            if _no_index(e):
                return []
            raise
        return ProjectIndexer.forget_project_index(project_id, self._vectors_db)

    def _forget_chunks(self, project_id: str, paths: set[str]) -> list[str]:
        """The chunks drawn from ``paths`` removed from the project's index; what stays, named."""
        try:
            from opti_oignon.project_context import ProjectIndexer
        except ModuleNotFoundError as e:
            if _no_index(e):
                return []
            raise
        return ProjectIndexer.forget_file_chunks(project_id, paths, self._vectors_db)

    def _sweep_index(self, live: dict[str, set[str]]) -> list[str]:
        """What earlier deletions left in the index removed (``ProjectIndexer.sweep_index``); what stays, named."""
        try:
            from opti_oignon.project_context import ProjectIndexer
        except ModuleNotFoundError as e:
            if _no_index(e):
                return []
            raise
        return ProjectIndexer.sweep_index(live, self._vectors_db)

    def _forget_stored(self, project_id: str, kind: str, item_id: str) -> list[str]:
        """The bytes of one file or output -- every entry of the project's ``kind`` folder its id prefixes --
        overwritten and removed; what could not be, named. Only the store's own folder is touched."""
        folder = self._storage_base / project_id / kind
        if not (_plain(project_id) and _plain(item_id)) or folder.is_symlink() or not folder.is_dir():
            return []
        failed: OSError | None = None
        said: list = []
        for entry in sorted(folder.iterdir()):
            if entry.name.startswith(f"{item_id}_"):
                try:
                    _forget_entry(entry, said)
                except OSError as exc:
                    failed = failed or exc
        unwritten = _unwritten(said, f"a removed {kind[:-1]} of project {project_id}")
        if failed is None:
            return unwritten
        logger.warning("project %s: a removed %s's bytes could not be removed: %s", project_id, kind[:-1],
                       type(failed).__name__)
        return [f"{_FILES_LABEL} (a removed {kind[:-1]} of project {project_id}, not removed: "
                f"{type(failed).__name__})", *unwritten]

    def _read_stored(self, project_id: str, kind: str, item_id: str) -> bytes | None:
        """The bytes of one file or output, read where the store keeps them -- the entry of the project's ``kind``
        folder its id prefixes, the place its deletion removes --, a link never followed; None when not there."""
        folder = self._storage_base / project_id / kind
        if (not (_plain(project_id) and _plain(item_id)) or folder.parent.is_symlink() or folder.is_symlink()
                or not folder.is_dir()):
            return None
        for entry in sorted(folder.iterdir()):
            if entry.name.startswith(f"{item_id}_"):
                try:
                    fd = os.open(entry, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(fd, "rb") as handle:
                        return handle.read()
                except OSError:
                    return None
        return None

    def empty_logs(self, *, index: bool = True) -> list[str]:
        """Empty the store's write-ahead log, and the index's unless ``index`` is False (a deletion that removed no
        chunk), after a deletion, so the pages it overwrote reach the database's file and no older frame keeps what
        it erased; what could not be, named. Never waits: a read or a
        write in progress keeps both what the deletion erased in the file and its log, until it ends and the next
        deletion empties them."""
        left: list[str] = []
        try:
            from opti_oignon.project_context import empty_index_log
        except ModuleNotFoundError as e:
            if not _no_index(e):
                raise
        else:
            if index:
                left += empty_index_log(self._vectors_db)
        if not self.present:
            return left
        try:
            conn = _safe_connect(self._db_path, check_same_thread=False)
            try:
                conn.execute("PRAGMA busy_timeout = 0")
                busy = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - the cipher driver's errors are not sqlite3's, and this only says
            logger.warning("projects: a deletion made; the write-ahead log could not be emptied (%s)",
                           type(exc).__name__)
            return [*left, f"{_STORE_LABEL} (its log not emptied: {type(exc).__name__}; the next deletion tries again)"]
        if busy:
            logger.warning("projects: a deletion made; a read or a write in progress keeps what it erased in the "
                           "database and its log until it ends")
            left.append(f"{_STORE_LABEL} (a read or a write in progress keeps what the deletion erased in the database "
                        "until it ends; the next deletion empties it)")
        return left

    # =========================================================================
    # CONVERSATION LINKING
    # =========================================================================

    def link_conversation(self, project_id: str, conversation_id: str) -> bool:
        """Link a conversation to a project.

        Args:
            project_id: The project ID.
            conversation_id: The conversation ID.

        Returns:
            True if linked successfully, False if already linked or project missing.
        """
        project = self.get_project(project_id)
        if project is None:
            return False

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO project_conversations (project_id, conversation_id, linked_at)
                VALUES (?, ?, ?)
                """,
                (project_id, conversation_id, _iso_now()),
            )
            conn.commit()
            changed = conn.total_changes > 0
        finally:
            conn.close()

        if changed:
            self._touch_project(project_id)
            logger.info("Linked conversation %s to project %s", conversation_id, project_id)
        return True

    def forget_conversation(self, conversation_id: str) -> int:
        """Unlink a deleted conversation from every project, and clear it as the source of the outputs it gave; the
        links removed. The outputs stay: they are the project's. A store whose file is not there holds nothing, and
        is not created; a failure raises."""
        if not Path(self._db_path).exists():
            return 0
        conn = self._get_conn()
        try:
            gone = conn.execute(
                "DELETE FROM project_conversations WHERE conversation_id = ?", (conversation_id,)).rowcount
            conn.execute("UPDATE project_outputs SET source_conversation_id = '' WHERE source_conversation_id = ?",
                         (conversation_id,))
            conn.commit()
            return gone
        finally:
            conn.close()

    def unlink_conversation(self, project_id: str, conversation_id: str) -> bool:
        """Unlink a conversation from a project.

        Args:
            project_id: The project ID.
            conversation_id: The conversation ID.

        Returns:
            True if the link existed and was removed.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM project_conversations WHERE project_id = ? AND conversation_id = ?",
                (project_id, conversation_id),
            )
            conn.commit()
            removed = cursor.rowcount > 0
        finally:
            conn.close()

        if removed:
            self._touch_project(project_id)
            logger.info("Unlinked conversation %s from project %s", conversation_id, project_id)
        return removed

    def list_conversations(self, project_id: str) -> list[dict[str, str]]:
        """List all conversation links for a project.

        Args:
            project_id: The project ID.

        Returns:
            List of dicts with conversation_id and linked_at.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT conversation_id, linked_at FROM project_conversations WHERE project_id = ? ORDER BY linked_at DESC",
                (project_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_project_for_conversation(self, conversation_id: str) -> str | None:
        """Find which project a conversation belongs to.

        Args:
            conversation_id: The conversation ID.

        Returns:
            Project ID if linked, None otherwise.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT project_id FROM project_conversations WHERE conversation_id = ? LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return row["project_id"]
        finally:
            conn.close()

    # =========================================================================
    # AGGREGATE QUERIES
    # =========================================================================

    def get_project_stats(self, project_id: str) -> dict[str, Any]:
        """Get aggregate statistics for a project.

        Args:
            project_id: The project ID.

        Returns:
            Dict with file_count, output_count, conversation_count, total_size_bytes.
        """
        conn = self._get_conn()
        try:
            file_row = conn.execute(
                "SELECT COUNT(*) as cnt, COALESCE(SUM(file_size_bytes), 0) as total_size FROM project_files WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            output_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM project_outputs WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            conv_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM project_conversations WHERE project_id = ?",
                (project_id,),
            ).fetchone()

            return {
                "file_count": file_row["cnt"],
                "total_size_bytes": file_row["total_size"],
                "output_count": output_row["cnt"],
                "conversation_count": conv_row["cnt"],
            }
        finally:
            conn.close()

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _touch_project(self, project_id: str) -> None:
        """Update the updated_at timestamp of a project."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (_iso_now(), project_id),
            )
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        """Cleanup (no persistent connections to close)."""
        pass


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

project_store = ProjectStore()


def forget_conversation(conversation_id: str) -> int:
    """Forget what the project store holds of a deleted conversation: its links; how many."""
    return project_store.forget_conversation(conversation_id)
