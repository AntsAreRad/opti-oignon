#!/usr/bin/env python3
"""
Plugin index for Opti-Oignon (S102).

PluginIndex: browse available plugins from a local SQLite index
and a remote GitHub-hosted index.json. Supports search by name,
tag, author, hook type. Refresh with staleness-based caching.
"""

import json
import logging
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


# Default staleness threshold: 1 hour
DEFAULT_CACHE_TTL_SECONDS = 3600

# Default GitHub index URL (placeholder — user configures their own)
DEFAULT_INDEX_URL = ""


@dataclass
class IndexEntry:
    """A plugin entry in the marketplace index."""

    name: str
    version: str
    description: str
    author: str
    url: str = ""
    tags: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    min_opti_version: str = "1.0.0"
    stars: int = 0
    downloads: int = 0
    sha256: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "url": self.url,
            "tags": self.tags,
            "hooks": self.hooks,
            "permissions": self.permissions,
            "min_opti_version": self.min_opti_version,
            "stars": self.stars,
            "downloads": self.downloads,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexEntry":
        """Create an IndexEntry from a dict (e.g. parsed JSON)."""
        return cls(
            name=str(data.get("name", "")).strip(),
            version=str(data.get("version", "0.0.0")).strip(),
            description=str(data.get("description", "")).strip(),
            author=str(data.get("author", "")).strip(),
            url=str(data.get("url", "")).strip(),
            tags=list(data.get("tags") or []),
            hooks=list(data.get("hooks") or []),
            permissions=list(data.get("permissions") or []),
            min_opti_version=str(data.get("min_opti_version", "1.0.0")).strip(),
            stars=int(data.get("stars", 0)),
            downloads=int(data.get("downloads", 0)),
            sha256=str(data.get("sha256", "")).strip(),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


class PluginIndex:
    """Local SQLite-backed plugin index with optional remote sync.

    Parameters
    ----------
    db_path : Path or str
        Path to the SQLite database for the index.
    index_url : str
        URL to a remote index.json (GitHub raw URL or similar).
    cache_ttl : int
        Seconds before the cached remote index is considered stale.
    """

    def __init__(
        self,
        db_path: Path | str,
        index_url: str = DEFAULT_INDEX_URL,
        cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._db_path = Path(db_path)
        self._index_url = index_url
        self._cache_ttl = cache_ttl
        self._last_refresh: float = 0.0
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # PI-13: restore the persisted refresh timestamp so a restart
        # does not mark a fresh cache as stale.
        self._last_refresh = self._load_last_refresh()

    # -----------------------------------------------------------------
    # SQLite setup
    # -----------------------------------------------------------------

    def _load_last_refresh(self) -> float:
        """Read the persisted last_refresh timestamp (0.0 when absent)."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT value FROM index_meta WHERE key = ?",
                ("last_refresh",),
            ).fetchone()
            if row is None:
                return 0.0
            try:
                return float(row["value"])
            except (TypeError, ValueError):
                return 0.0
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plugin_index (
                    name            TEXT PRIMARY KEY,
                    version         TEXT NOT NULL DEFAULT '0.0.0',
                    description     TEXT NOT NULL DEFAULT '',
                    author          TEXT NOT NULL DEFAULT '',
                    url             TEXT NOT NULL DEFAULT '',
                    tags            TEXT NOT NULL DEFAULT '[]',
                    hooks           TEXT NOT NULL DEFAULT '[]',
                    permissions     TEXT NOT NULL DEFAULT '[]',
                    min_opti_version TEXT NOT NULL DEFAULT '1.0.0',
                    stars           INTEGER NOT NULL DEFAULT 0,
                    downloads       INTEGER NOT NULL DEFAULT 0,
                    sha256          TEXT NOT NULL DEFAULT '',
                    created_at      REAL NOT NULL DEFAULT 0,
                    updated_at      REAL NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS index_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.commit()
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # CRUD helpers
    # -----------------------------------------------------------------

    def _row_to_entry(self, row: sqlite3.Row) -> IndexEntry:
        """Convert a database row to an IndexEntry."""
        return IndexEntry(
            name=row["name"],
            version=row["version"],
            description=row["description"],
            author=row["author"],
            url=row["url"],
            tags=json.loads(row["tags"]),
            hooks=json.loads(row["hooks"]),
            permissions=json.loads(row["permissions"]),
            min_opti_version=row["min_opti_version"],
            stars=row["stars"],
            downloads=row["downloads"],
            sha256=row["sha256"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert(self, entry: IndexEntry) -> None:
        """Insert or update a plugin entry in the local index."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO plugin_index
                    (name, version, description, author, url, tags, hooks,
                     permissions, min_opti_version, stars, downloads,
                     sha256, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.name, entry.version, entry.description,
                    entry.author, entry.url,
                    json.dumps(entry.tags), json.dumps(entry.hooks),
                    json.dumps(entry.permissions), entry.min_opti_version,
                    entry.stars, entry.downloads, entry.sha256,
                    entry.created_at, entry.updated_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def remove(self, name: str) -> bool:
        """Remove a plugin entry from the index. Returns True if removed."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM plugin_index WHERE name = ?", (name,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get(self, name: str) -> Optional[IndexEntry]:
        """Get a single plugin entry by name, or None."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM plugin_index WHERE name = ?", (name,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_entry(row)
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Listing & search
    # -----------------------------------------------------------------

    def list_all(
        self,
        *,
        sort_by: str = "name",
        limit: int = 100,
        offset: int = 0,
    ) -> list[IndexEntry]:
        """List all entries in the index.

        Parameters
        ----------
        sort_by : str
            Column to sort by: name, stars, downloads, updated_at.
        limit : int
            Max entries to return.
        offset : int
            Pagination offset.
        """
        allowed_sorts = {"name", "stars", "downloads", "updated_at", "created_at"}
        col = sort_by if sort_by in allowed_sorts else "name"
        order = "DESC" if col in ("stars", "downloads", "updated_at", "created_at") else "ASC"

        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM plugin_index ORDER BY {} {} LIMIT ? OFFSET ?".format(col, order),
                (limit, offset),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]
        finally:
            conn.close()

    def search(
        self,
        *,
        keyword: str = "",
        tag: str = "",
        author: str = "",
        hook: str = "",
        sort_by: str = "stars",
        limit: int = 50,
    ) -> list[IndexEntry]:
        """Search the index by keyword, tag, author, or hook type.

        Parameters
        ----------
        keyword : str
            Search in name and description (LIKE match).
        tag : str
            Filter by tag (JSON array LIKE match).
        author : str
            Filter by author (LIKE match).
        hook : str
            Filter by hook type (JSON array LIKE match).
        sort_by : str
            Sort column: name, stars, downloads, updated_at.
        limit : int
            Max results.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if keyword:
            conditions.append(
                "(LOWER(name) LIKE ? OR LOWER(description) LIKE ?)"
            )
            kw = f"%{keyword.lower()}%"
            params.extend([kw, kw])

        if tag:
            conditions.append("LOWER(tags) LIKE ?")
            params.append(f"%{tag.lower()}%")

        if author:
            conditions.append("LOWER(author) LIKE ?")
            params.append(f"%{author.lower()}%")

        if hook:
            conditions.append("LOWER(hooks) LIKE ?")
            params.append(f"%{hook.lower()}%")

        where = " AND ".join(conditions) if conditions else "1=1"

        allowed_sorts = {"name", "stars", "downloads", "updated_at"}
        col = sort_by if sort_by in allowed_sorts else "stars"
        order = "DESC" if col in ("stars", "downloads", "updated_at") else "ASC"

        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM plugin_index WHERE {} "
                "ORDER BY {} {} LIMIT ?".format(where, col, order),
                (*params, limit),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]
        finally:
            conn.close()

    @property
    def count(self) -> int:
        """Total number of entries in the index."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM plugin_index"
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Remote index sync
    # -----------------------------------------------------------------

    @property
    def index_url(self) -> str:
        """Currently configured remote index URL."""
        return self._index_url

    @index_url.setter
    def index_url(self, url: str) -> None:
        self._index_url = url

    @property
    def is_stale(self) -> bool:
        """Whether the cached index is stale and should be refreshed."""
        if self._last_refresh == 0.0:
            return True
        return (time.time() - self._last_refresh) > self._cache_ttl

    def refresh_from_remote(self, *, force: bool = False) -> int:
        """Fetch the remote index.json and upsert entries.

        Returns the number of entries added/updated.
        Skips if index is not stale unless force=True.
        """
        if not force and not self.is_stale:
            logger.debug("Plugin index not stale, skipping refresh")
            return 0

        if not self._index_url:
            logger.debug("No remote index URL configured")
            return 0

        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                self._index_url,
                headers={"User-Agent": "Opti-Oignon-PluginIndex/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except Exception as exc:
            logger.warning("Failed to fetch remote plugin index: %s", exc)
            return 0

        # Expect data to be {"plugins": [...]} or a bare list
        entries_data = data if isinstance(data, list) else data.get("plugins", [])
        if not isinstance(entries_data, list):
            logger.warning("Remote index has unexpected format")
            return 0

        count = 0
        for item in entries_data:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                entry = IndexEntry.from_dict(item)
                self.upsert(entry)
                count += 1
            except Exception as exc:
                logger.warning(
                    "Skipping invalid index entry '%s': %s",
                    item.get("name", "?"), exc,
                )

        self._last_refresh = time.time()
        # Persist last refresh time
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                ("last_refresh", str(self._last_refresh)),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info("Plugin index refreshed: %d entries updated", count)
        return count

    def load_from_json(self, entries: list[dict[str, Any]]) -> int:
        """Bulk load entries from a list of dicts (e.g. parsed JSON).

        Returns the number of entries added/updated.
        """
        count = 0
        for item in entries:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                entry = IndexEntry.from_dict(item)
                self.upsert(entry)
                count += 1
            except Exception as exc:
                logger.warning(
                    "Skipping invalid entry '%s': %s",
                    item.get("name", "?"), exc,
                )
        return count

    def increment_downloads(self, name: str) -> bool:
        """Increment the download counter for a plugin."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "UPDATE plugin_index SET downloads = downloads + 1 WHERE name = ?",
                (name,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


# =========================================================================
# Module-level singleton
# =========================================================================

PLUGIN_INDEX_AVAILABLE = True

try:
    from opti_oignon.config import DATA_DIR as _DATA_DIR

    _idx_db = Path(_DATA_DIR) / "plugin_index.db"

    # Try to load marketplace config for index_url and cache_ttl
    _idx_url = DEFAULT_INDEX_URL
    _idx_ttl = DEFAULT_CACHE_TTL_SECONDS
    try:
        import yaml as _yaml

        _mp_cfg_path = Path(__file__).parent / "config" / "plugin_marketplace.yaml"
        if _mp_cfg_path.exists():
            with open(_mp_cfg_path, "r", encoding="utf-8") as _fh:
                _mp_cfg = _yaml.safe_load(_fh) or {}
            _idx_url = _mp_cfg.get("index", {}).get("url", DEFAULT_INDEX_URL)
            _idx_ttl = _mp_cfg.get("index", {}).get(
                "cache_ttl_seconds", DEFAULT_CACHE_TTL_SECONDS
            )
    except Exception:
        pass

    plugin_index = PluginIndex(
        db_path=_idx_db,
        index_url=_idx_url,
        cache_ttl=_idx_ttl,
    )
except Exception as _exc:
    logger.debug("PluginIndex singleton init deferred: %s", _exc)
    plugin_index = None  # type: ignore[assignment]
