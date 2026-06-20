#!/usr/bin/env python3
"""
Plugin manifest and registry for Opti-Oignon (S101).

PluginManifest: YAML-driven plugin descriptor with validation.
PluginRegistry: discover, register, resolve dependencies, SQLite-backed state.
"""

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# Valid hook points that plugins can register
VALID_HOOKS = frozenset({
    "pre_prompt",
    "post_prompt",
    "pre_inference",
    "post_inference",
    "tool_call",
    "pipeline_step",
    "ui_panel",
})

# Permissions that plugins can request
VALID_PERMISSIONS = frozenset({
    "conversation_read",
    "conversation_write",
    "model_config_read",
    "model_config_write",
    "tool_register",
    "pipeline_register",
    "ui_panel_register",
    "filesystem_plugin_dir",
    "network_outbound",
    # S124: New permissions for security hardening
    "filesystem_read",       # Can read files within plugin dir only
    "filesystem_write",      # Can write files within plugin dir only
    "inference_content",     # Can see prompt/response content in hooks
})

# Semantic version pattern (major.minor.patch, optional pre-release)
_VERSION_RE = re.compile(
    r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$"
)

# Valid plugin name: lowercase alphanumeric + hyphens/underscores
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


class PluginManifestError(Exception):
    """Raised when a plugin manifest is invalid."""


@dataclass
class PluginManifest:
    """Describes a plugin's metadata, entry point, hooks, and permissions."""

    name: str
    version: str
    author: str
    description: str
    entry_point: str  # relative path to Python file (e.g. "entry_point.py")
    hooks: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    min_opti_version: str = "1.0.0"
    config_schema: dict[str, Any] = field(default_factory=dict)
    resource_limits: dict[str, int] = field(default_factory=dict)  # S143

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginManifest":
        """Create a PluginManifest from a parsed YAML/dict.

        Raises PluginManifestError on validation failure.
        """
        # Required fields check
        required = ("name", "version", "author", "description", "entry_point")
        missing = [f for f in required if f not in data or not data[f]]
        if missing:
            raise PluginManifestError(
                f"Missing required manifest fields: {', '.join(missing)}"
            )

        manifest = cls(
            name=str(data["name"]).strip(),
            version=str(data["version"]).strip(),
            author=str(data["author"]).strip(),
            description=str(data["description"]).strip(),
            entry_point=str(data["entry_point"]).strip(),
            hooks=list(data.get("hooks") or []),
            dependencies=list(data.get("dependencies") or []),
            permissions=list(data.get("permissions") or []),
            min_opti_version=str(data.get("min_opti_version", "1.0.0")).strip(),
            config_schema=dict(data.get("config_schema") or {}),
            resource_limits=dict(data.get("resource_limits") or {}),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Validate manifest fields. Raises PluginManifestError on failure."""
        # Name format
        if not _NAME_RE.match(self.name):
            raise PluginManifestError(
                f"Invalid plugin name '{self.name}': must be lowercase "
                f"alphanumeric with hyphens/underscores, 2-64 chars, "
                f"starting with a letter."
            )

        # Version format
        if not _VERSION_RE.match(self.version):
            raise PluginManifestError(
                f"Invalid version '{self.version}': must be semantic "
                f"versioning (e.g. 1.0.0, 1.2.3-beta)."
            )

        # Entry point safety: no path traversal
        if ".." in self.entry_point or self.entry_point.startswith("/"):
            raise PluginManifestError(
                f"Invalid entry_point '{self.entry_point}': must be a "
                f"relative path within the plugin directory."
            )
        if not self.entry_point.endswith(".py"):
            raise PluginManifestError(
                f"Invalid entry_point '{self.entry_point}': must be a "
                f".py file."
            )

        # Hook validation
        invalid_hooks = set(self.hooks) - VALID_HOOKS
        if invalid_hooks:
            raise PluginManifestError(
                f"Invalid hooks: {', '.join(sorted(invalid_hooks))}. "
                f"Valid hooks: {', '.join(sorted(VALID_HOOKS))}"
            )

        # Permission validation
        invalid_perms = set(self.permissions) - VALID_PERMISSIONS
        if invalid_perms:
            raise PluginManifestError(
                f"Invalid permissions: {', '.join(sorted(invalid_perms))}. "
                f"Valid permissions: {', '.join(sorted(VALID_PERMISSIONS))}"
            )

        # min_opti_version format
        if self.min_opti_version and not _VERSION_RE.match(self.min_opti_version):
            raise PluginManifestError(
                f"Invalid min_opti_version '{self.min_opti_version}'."
            )

        # S143: resource_limits validation
        _VALID_RLIMIT_KEYS = {
            "cpu_time_seconds", "memory_bytes", "max_file_descriptors",
        }
        invalid_rl = set(self.resource_limits.keys()) - _VALID_RLIMIT_KEYS
        if invalid_rl:
            raise PluginManifestError(
                f"Invalid resource_limits keys: {', '.join(sorted(invalid_rl))}. "
                f"Valid keys: {', '.join(sorted(_VALID_RLIMIT_KEYS))}"
            )
        for key, val in self.resource_limits.items():
            if not isinstance(val, (int, float)) or val < 0:
                raise PluginManifestError(
                    f"resource_limits.{key} must be a non-negative number, "
                    f"got {val!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a dict (for JSON/YAML export)."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "entry_point": self.entry_point,
            "hooks": self.hooks,
            "dependencies": self.dependencies,
            "permissions": self.permissions,
            "min_opti_version": self.min_opti_version,
            "config_schema": self.config_schema,
            "resource_limits": self.resource_limits,
        }


# =========================================================================
# Plugin state constants
# =========================================================================

PLUGIN_STATE_INSTALLED = "installed"
PLUGIN_STATE_ENABLED = "enabled"
PLUGIN_STATE_DISABLED = "disabled"


@dataclass
class PluginRecord:
    """A registered plugin with its manifest and runtime state."""

    manifest: PluginManifest
    state: str  # installed / enabled / disabled
    plugin_dir: str  # absolute path to plugin directory
    installed_at: float = 0.0
    updated_at: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """Discover, register, and manage plugins with SQLite-backed state.

    Parameters
    ----------
    db_path : Path or str
        Path to the SQLite database file for plugin state persistence.
    plugins_dir : Path or str or None
        Base directory where plugins are stored on disk.
    """

    def __init__(
        self,
        db_path: Path | str,
        plugins_dir: Path | str | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._plugins_dir = Path(plugins_dir) if plugins_dir else None
        self._plugins: dict[str, PluginRecord] = {}
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_from_db()

    # -----------------------------------------------------------------
    # SQLite schema & helpers
    # -----------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plugins (
                    name        TEXT PRIMARY KEY,
                    version     TEXT NOT NULL,
                    author      TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    entry_point TEXT NOT NULL DEFAULT 'entry_point.py',
                    hooks       TEXT NOT NULL DEFAULT '[]',
                    dependencies TEXT NOT NULL DEFAULT '[]',
                    permissions TEXT NOT NULL DEFAULT '[]',
                    min_opti_version TEXT NOT NULL DEFAULT '1.0.0',
                    config_schema TEXT NOT NULL DEFAULT '{}',
                    resource_limits TEXT NOT NULL DEFAULT '{}',
                    state       TEXT NOT NULL DEFAULT 'installed',
                    plugin_dir  TEXT NOT NULL DEFAULT '',
                    config      TEXT NOT NULL DEFAULT '{}',
                    installed_at REAL NOT NULL DEFAULT 0,
                    updated_at  REAL NOT NULL DEFAULT 0
                )
            """)
            # PI-12: migrate pre-existing databases created before the
            # resource_limits column was added (S195).
            try:
                conn.execute(
                    "ALTER TABLE plugins ADD COLUMN resource_limits "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plugin_version_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    version     TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    timestamp   REAL NOT NULL,
                    FOREIGN KEY (name) REFERENCES plugins(name)
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_from_db(self) -> None:
        """Load all plugin records from the database into memory."""
        import json

        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM plugins").fetchall()
            for row in rows:
                try:
                    manifest = PluginManifest(
                        name=row["name"],
                        version=row["version"],
                        author=row["author"],
                        description=row["description"],
                        entry_point=row["entry_point"],
                        hooks=json.loads(row["hooks"]),
                        dependencies=json.loads(row["dependencies"]),
                        permissions=json.loads(row["permissions"]),
                        min_opti_version=row["min_opti_version"],
                        config_schema=json.loads(row["config_schema"]),
                        # PI-12: resource_limits round-trips through the DB
                        resource_limits=(
                            json.loads(row["resource_limits"])
                            if "resource_limits" in row.keys()
                            else {}
                        ),
                    )
                    record = PluginRecord(
                        manifest=manifest,
                        state=row["state"],
                        plugin_dir=row["plugin_dir"],
                        installed_at=row["installed_at"],
                        updated_at=row["updated_at"],
                        config=json.loads(row["config"]),
                    )
                    self._plugins[manifest.name] = record
                except Exception as exc:
                    logger.warning(
                        "Failed to load plugin '%s' from DB: %s",
                        row["name"], exc,
                    )
        finally:
            conn.close()

    def _save_plugin(self, record: PluginRecord) -> None:
        """Persist a plugin record to the database."""
        import json

        m = record.manifest
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO plugins
                    (name, version, author, description, entry_point,
                     hooks, dependencies, permissions, min_opti_version,
                     config_schema, resource_limits, state, plugin_dir,
                     config, installed_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    m.name, m.version, m.author, m.description, m.entry_point,
                    json.dumps(m.hooks), json.dumps(m.dependencies),
                    json.dumps(m.permissions), m.min_opti_version,
                    json.dumps(m.config_schema),
                    json.dumps(m.resource_limits),
                    record.state,
                    record.plugin_dir, json.dumps(record.config),
                    record.installed_at, record.updated_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _record_version_event(
        self, name: str, version: str, action: str,
    ) -> None:
        """Append a version history event."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO plugin_version_history (name, version, action, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (name, version, action, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def register(
        self,
        manifest: PluginManifest,
        plugin_dir: str,
        *,
        auto_enable: bool = False,
    ) -> PluginRecord:
        """Register a plugin from its manifest.

        If a plugin with the same name already exists, it is updated
        (version upgrade).

        Returns the PluginRecord.
        """
        now = time.time()

        existing = self._plugins.get(manifest.name)
        if existing:
            # Update existing: preserve installed_at, update version
            existing.manifest = manifest
            existing.plugin_dir = plugin_dir
            existing.updated_at = now
            if auto_enable:
                existing.state = PLUGIN_STATE_ENABLED
            self._save_plugin(existing)
            self._record_version_event(manifest.name, manifest.version, "updated")
            logger.info(
                "Updated plugin '%s' to version %s",
                manifest.name, manifest.version,
            )
            return existing

        state = PLUGIN_STATE_ENABLED if auto_enable else PLUGIN_STATE_INSTALLED
        record = PluginRecord(
            manifest=manifest,
            state=state,
            plugin_dir=plugin_dir,
            installed_at=now,
            updated_at=now,
        )
        self._plugins[manifest.name] = record
        self._save_plugin(record)
        self._record_version_event(manifest.name, manifest.version, "installed")
        logger.info("Registered plugin '%s' v%s", manifest.name, manifest.version)
        return record

    def unregister(self, name: str) -> bool:
        """Remove a plugin from the registry.

        Returns True if the plugin was found and removed.
        """
        record = self._plugins.pop(name, None)
        if record is None:
            return False

        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM plugins WHERE name = ?", (name,))
            conn.commit()
        finally:
            conn.close()

        self._record_version_event(name, record.manifest.version, "uninstalled")
        logger.info("Unregistered plugin '%s'", name)
        return True

    def get(self, name: str) -> Optional[PluginRecord]:
        """Get a plugin record by name, or None."""
        return self._plugins.get(name)

    def list_plugins(
        self,
        *,
        state: Optional[str] = None,
    ) -> list[PluginRecord]:
        """List all registered plugins, optionally filtered by state."""
        records = list(self._plugins.values())
        if state:
            records = [r for r in records if r.state == state]
        return sorted(records, key=lambda r: r.manifest.name)

    def set_state(self, name: str, state: str) -> bool:
        """Change a plugin's state (enabled/disabled/installed).

        Returns True if the plugin was found and updated.
        """
        record = self._plugins.get(name)
        if record is None:
            return False
        if state not in (PLUGIN_STATE_INSTALLED, PLUGIN_STATE_ENABLED, PLUGIN_STATE_DISABLED):
            raise ValueError(f"Invalid state: {state}")
        record.state = state
        record.updated_at = time.time()
        self._save_plugin(record)
        self._record_version_event(name, record.manifest.version, f"state:{state}")
        return True

    def set_config(self, name: str, config: dict[str, Any]) -> bool:
        """Update a plugin's configuration.

        Returns True if the plugin was found and updated.
        """
        record = self._plugins.get(name)
        if record is None:
            return False
        record.config = config
        record.updated_at = time.time()
        self._save_plugin(record)
        return True

    def get_version_history(self, name: str) -> list[dict[str, Any]]:
        """Get version history events for a plugin."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT version, action, timestamp FROM plugin_version_history "
                "WHERE name = ? ORDER BY timestamp DESC",
                (name,),
            ).fetchall()
            return [
                {"version": r["version"], "action": r["action"], "timestamp": r["timestamp"]}
                for r in rows
            ]
        finally:
            conn.close()

    def resolve_dependencies(self, name: str) -> list[str]:
        """Return an ordered list of plugin names required to enable `name`.

        Raises PluginManifestError on missing dependency or circular reference.
        """
        visited: set[str] = set()
        order: list[str] = []

        def _visit(current: str, chain: list[str]) -> None:
            if current in chain:
                cycle = " -> ".join(chain + [current])
                raise PluginManifestError(
                    f"Circular dependency detected: {cycle}"
                )
            if current in visited:
                return
            record = self._plugins.get(current)
            if record is None:
                raise PluginManifestError(
                    f"Missing dependency: '{current}' (required by "
                    f"'{chain[-1] if chain else name}')"
                )
            visited.add(current)
            for dep in record.manifest.dependencies:
                _visit(dep, chain + [current])
            order.append(current)

        _visit(name, [])
        return order

    def discover(self, search_dir: Path | str | None = None) -> list[PluginManifest]:
        """Scan a directory for plugin manifests (manifest.yaml files).

        Returns a list of valid manifests found. Invalid manifests are
        logged as warnings and skipped.
        """
        try:
            import yaml as _yaml
        except ImportError:
            logger.warning("PyYAML not available; cannot discover plugins")
            return []

        base = Path(search_dir) if search_dir else self._plugins_dir
        if base is None or not base.is_dir():
            return []

        manifests: list[PluginManifest] = []
        for manifest_path in sorted(base.glob("*/manifest.yaml")):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    data = _yaml.safe_load(fh)
                if not isinstance(data, dict):
                    logger.warning("Skipping %s: not a valid YAML dict", manifest_path)
                    continue
                manifest = PluginManifest.from_dict(data)
                manifests.append(manifest)
            except Exception as exc:
                logger.warning("Skipping %s: %s", manifest_path, exc)
        return manifests

    @property
    def plugin_count(self) -> int:
        """Total number of registered plugins."""
        return len(self._plugins)

    @property
    def enabled_count(self) -> int:
        """Number of currently enabled plugins."""
        return sum(
            1 for r in self._plugins.values()
            if r.state == PLUGIN_STATE_ENABLED
        )


# =========================================================================
# Module-level singleton
# =========================================================================

PLUGIN_MANIFEST_AVAILABLE = True


def _discover_builtins(
    registry: PluginRegistry,
    builtin_dir: Path | None = None,
) -> int:
    """Auto-discover and register builtin plugins from opti_oignon/plugins/.

    PI-21: when a builtin is already registered but its on-disk manifest
    carries a different version, the registry record is refreshed (state
    and installed_at are preserved by register()).  The ``builtin_dir``
    parameter exists for tests; production callers use the default.

    Returns the number of newly registered builtins.
    """
    if builtin_dir is None:
        builtin_dir = Path(__file__).parent / "plugins"
    if not builtin_dir.is_dir():
        return 0

    manifests = registry.discover(builtin_dir)
    registered = 0
    for manifest in manifests:
        # Skip if already registered at the same version
        existing = registry.get(manifest.name)
        if existing is not None:
            if existing.manifest.version != manifest.version:
                try:
                    registry.register(
                        manifest,
                        str(builtin_dir / manifest.name),
                        auto_enable=False,
                    )
                    logger.info(
                        "Refreshed builtin '%s' to version %s",
                        manifest.name, manifest.version,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to refresh builtin '%s': %s",
                        manifest.name, exc,
                    )
            continue
        try:
            plugin_dir = str(builtin_dir / manifest.name)
            registry.register(manifest, plugin_dir, auto_enable=True)
            registered += 1
        except Exception as exc:
            logger.warning("Failed to auto-register builtin '%s': %s", manifest.name, exc)
    if registered > 0:
        logger.info("Auto-registered %d builtin plugin(s)", registered)
    return registered


try:
    from opti_oignon.config import DATA_DIR as _DATA_DIR

    _db = Path(_DATA_DIR) / "plugins.db"
    _pdir = Path(_DATA_DIR) / "plugins"
    _pdir.mkdir(parents=True, exist_ok=True)
    plugin_registry = PluginRegistry(db_path=_db, plugins_dir=_pdir)

    # Auto-register builtin plugins on first run
    _discover_builtins(plugin_registry)
except Exception as _exc:
    logger.debug("PluginRegistry singleton init deferred: %s", _exc)
    plugin_registry = None  # type: ignore[assignment]
