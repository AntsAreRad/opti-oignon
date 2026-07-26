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

import json
import logging
import os
import shutil
import sqlite3
import uuid
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
    ):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._storage_base = storage_base or _DEFAULT_STORAGE_BASE
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
        self._init_db()

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

        logger.info("Created project '%s' (id=%s)", project.name, project.id)
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

        logger.info("Updated project '%s' (id=%s)", project.name, project.id)
        return project

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all associated data.

        Removes the project record, all linked files (from disk and DB),
        all outputs, and all conversation links.

        Args:
            project_id: The project ID.

        Returns:
            True if the project existed and was deleted.
        """
        project = self.get_project(project_id)
        if project is None:
            return False

        # Remove project directory from disk (files + outputs)
        project_dir = self._storage_base / project_id
        if project_dir.exists():
            try:
                shutil.rmtree(project_dir)
            except OSError as e:
                logger.warning("Failed to remove project directory %s: %s", project_dir, e)

        # Cascade delete in DB (foreign keys handle child tables)
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
        finally:
            conn.close()

        logger.info("Deleted project id=%s", project_id)
        return True

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
            "Added file '%s' to project %s (id=%s, type=%s, %d bytes)",
            safe_name, project_id, file_id, file_type, pf.file_size_bytes,
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
        """Remove a file from its project.

        Deletes the file from disk and the database record.

        Args:
            file_id: The file ID.

        Returns:
            True if the file existed and was removed.
        """
        pf = self.get_file(file_id)
        if pf is None:
            return False

        # Remove from disk
        fp = Path(pf.file_path)
        if fp.exists():
            try:
                fp.unlink()
            except OSError as e:
                logger.warning("Failed to remove file %s: %s", fp, e)

        # Remove from DB
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM project_files WHERE id = ?", (file_id,))
            conn.commit()
        finally:
            conn.close()

        # Touch project updated_at
        self._touch_project(pf.project_id)

        logger.info("Removed file id=%s from project %s", file_id, pf.project_id)
        return True

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
        fp = Path(pf.file_path)
        if not fp.exists():
            return None
        try:
            return fp.read_bytes()
        except OSError:
            return None

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
        logger.info("Added output '%s' to project %s", safe_name, project_id)
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

    def remove_output(self, output_id: str) -> bool:
        """Remove an output file from its project."""
        po = self.get_output(output_id)
        if po is None:
            return False

        fp = Path(po.file_path)
        if fp.exists():
            try:
                fp.unlink()
            except OSError as e:
                logger.warning("Failed to remove output %s: %s", fp, e)

        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM project_outputs WHERE id = ?", (output_id,))
            conn.commit()
        finally:
            conn.close()

        self._touch_project(po.project_id)
        logger.info("Removed output id=%s from project %s", output_id, po.project_id)
        return True

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
