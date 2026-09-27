#!/usr/bin/env python3
"""
Remote plugin installer for Opti-Oignon.

RemotePluginInstaller: download plugins from URLs (GitHub repos,
zip archives, tar.gz), verify manifest, check integrity hash,
copy to plugins directory, rollback on failure.
"""

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# The marketplace configuration; its ``install`` section is read at each
# installer's construction, key by key.
_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "plugin_marketplace.yaml"

_INSTALL_DEFAULTS: dict[str, Any] = {
    "allow_remote_install": True,
    "max_download_size_mb": 50,
    "require_hash": False,
    "timeout_s": 120,
}


def _install_config() -> dict[str, Any]:
    """The ``install`` settings: each missing or unusable value falls back to its default.

    A value that is present and unusable (a wrong type, a size or a timeout
    that is not positive) is logged with the default that replaces it.
    """
    section: Any = {}
    try:
        import yaml

        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            section = (yaml.safe_load(fh) or {}).get("install") or {}
    except Exception as exc:
        logger.warning("Plugin install configuration cannot be read (%s); the defaults apply.", exc)
    if not isinstance(section, dict):
        logger.warning("Plugin install configuration is not a mapping; the defaults apply.")
        section = {}
    config = {}
    for key, default in _INSTALL_DEFAULTS.items():
        value = section.get(key, default)
        if isinstance(default, bool):
            usable = isinstance(value, bool)
        else:
            usable = isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        if not usable:
            logger.warning("install.%s = %r is not usable; the default %r applies.", key, value, default)
            value = default
        config[key] = value
    return config


class PluginInstallError(Exception):
    """Raised when a remote plugin installation fails."""


class RemotePluginInstaller:
    """Download and install plugins from remote URLs.

    Parameters
    ----------
    plugins_dir : Path or str
        Base directory where plugins are stored on disk.
    registry : optional
        PluginRegistry instance for state management.
    loader : optional
        PluginLoader instance for loading after install.
    index : optional
        PluginIndex instance for download tracking.
    max_download_size_mb : int, optional
        Maximum download size in MB (safety limit); the configuration's
        ``install.max_download_size_mb`` when not given.
    """

    def __init__(
        self,
        plugins_dir: Path | str,
        registry: Any = None,
        loader: Any = None,
        index: Any = None,
        max_download_size_mb: int | None = None,
    ) -> None:
        self._plugins_dir = Path(plugins_dir)
        self._registry = registry
        self._loader = loader
        self._index = index
        self._config = _install_config()
        size = self._config["max_download_size_mb"] if max_download_size_mb is None else max_download_size_mb
        self._max_bytes = int(size * 1024 * 1024)
        self._plugins_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def install_from_url(
        self,
        url: str,
        *,
        expected_sha256: str = "",
        auto_enable: bool = False,
    ) -> dict[str, Any]:
        """Download and install a plugin from a URL.

        Supports .zip, .tar.gz archives, and GitHub repository URLs
        (auto-appended /archive/refs/heads/main.zip).

        Parameters
        ----------
        url : str
            URL to the plugin archive or GitHub repository.
        expected_sha256 : str
            Expected SHA-256 hash of the archive (optional).
        auto_enable : bool
            Automatically enable the plugin after install.

        Returns
        -------
        dict with keys: success, name, version, message, error
        """
        tmp_dir = None
        target_dir = None
        try:
            # Refusals come before anything else: nothing is fetched or written.
            self._admit(expected_sha256)

            # Normalize GitHub repo URLs
            download_url = self._normalize_url(url)

            # Download to temp directory
            tmp_dir = Path(tempfile.mkdtemp(prefix="oo_plugin_"))
            archive_path = self._download(download_url, tmp_dir)

            # Hash verification
            if expected_sha256:
                actual_hash = self._compute_sha256(archive_path)
                if actual_hash != expected_sha256.lower():
                    raise PluginInstallError(
                        f"Hash mismatch: expected {expected_sha256}, "
                        f"got {actual_hash}"
                    )

            # Extract archive
            extract_dir = tmp_dir / "extracted"
            extract_dir.mkdir()
            self._extract(archive_path, extract_dir)

            # Find the plugin root (directory containing manifest.yaml)
            plugin_root = self._find_plugin_root(extract_dir)
            if plugin_root is None:
                raise PluginInstallError(
                    "No manifest.yaml found in the downloaded archive"
                )

            # Validate manifest
            manifest = self._validate_manifest(plugin_root)

            # Copy to plugins directory (rollback target)
            target_dir = self._plugins_dir / manifest["name"]
            if target_dir.exists():
                # Backup existing for rollback
                backup_dir = self._plugins_dir / f".{manifest['name']}.backup"
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.move(str(target_dir), str(backup_dir))
            else:
                backup_dir = None

            try:
                shutil.copytree(plugin_root, target_dir)
            except Exception as exc:
                # Rollback: restore backup
                if backup_dir and backup_dir.exists():
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.move(str(backup_dir), str(target_dir))
                raise PluginInstallError(
                    f"Failed to copy plugin files: {exc}"
                ) from exc

            # Clean up backup on success
            if backup_dir and backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

            # Register in registry
            registered_ok = False
            if self._registry:
                try:
                    from opti_oignon.plugin_manifest import PluginManifest

                    m = PluginManifest.from_dict(manifest)
                    # PI-10/PI-11: register as installed; the enable flow
                    # below flips the state only after a successful load.
                    self._registry.register(
                        m, str(target_dir), auto_enable=False,
                    )
                    registered_ok = True
                except Exception as exc:
                    logger.warning("Registry registration failed: %s", exc)

            # Load if auto_enable and loader available
            loaded = None
            if auto_enable and self._loader:
                try:
                    if registered_ok and hasattr(self._loader, "enable_plugin"):
                        # PI-11: full enable flow (load + initialize +
                        # hook registration + state flip); a bare
                        # load_plugin() left hooks inactive until restart.
                        loaded = self._loader.enable_plugin(manifest["name"])
                    else:
                        loaded = self._loader.load_plugin(target_dir)
                        loaded.initialize()
                except Exception as exc:
                    logger.warning("Auto-enable load failed: %s", exc)
            elif auto_enable and self._registry and registered_ok:
                # No loader in this process: mark enabled so the plugin
                # is picked up by load_all_enabled() at next startup.
                try:
                    self._registry.set_state(manifest["name"], "enabled")
                except Exception as exc:
                    logger.warning("Could not mark plugin enabled: %s", exc)

            # Track download in index
            if self._index:
                try:
                    self._index.increment_downloads(manifest["name"])
                except Exception:
                    pass

            name = manifest["name"]
            version = manifest.get("version", "0.0.0")
            msg = f"Plugin '{name}' v{version} installed from URL"
            if auto_enable:
                msg += " and enabled"

            logger.info(msg)
            return {
                "success": True,
                "name": name,
                "version": version,
                "message": msg,
                "error": None,
            }

        except PluginInstallError as exc:
            logger.warning("Plugin install from URL failed: %s", exc)
            return {
                "success": False,
                "name": "",
                "version": "",
                "message": f"Installation failed: {exc}",
                "error": str(exc),
            }
        except Exception as exc:
            logger.warning("Unexpected error during plugin install: %s", exc)
            return {
                "success": False,
                "name": "",
                "version": "",
                "message": f"Unexpected error: {exc}",
                "error": str(exc),
            }
        finally:
            # Clean up temp directory
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # -----------------------------------------------------------------
    # Admission
    # -----------------------------------------------------------------

    def _admit(self, expected_sha256: str) -> None:
        """Refuse an install before any request: turned off, a digest missing, the web gate.

        Raises PluginInstallError carrying the refusal's text. The web gate
        is asked through the front door; a front door that cannot be imported
        refuses as well.
        """
        if not self._config["allow_remote_install"]:
            raise PluginInstallError(
                "Remote plugin install is turned off: install.allow_remote_install is false."
            )
        if self._config["require_hash"] and not expected_sha256:
            raise PluginInstallError(
                "A sha256 digest is required to install a plugin from a URL: "
                "install.require_hash is true."
            )
        from opti_oignon.egress import EgressRefused, require_web

        try:
            require_web("Plugin install")
        except EgressRefused as exc:
            raise PluginInstallError(str(exc)) from None

    # -----------------------------------------------------------------
    # URL normalization
    # -----------------------------------------------------------------

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for downloading.

        A GitHub repository URL (the host exactly ``github.com``, a path of
        two segments) becomes the archive of its main branch; every other URL
        is fetched as given.
        """
        url = url.strip()
        parts = urlsplit(url)
        if (parts.hostname or "").lower() != "github.com":
            return url
        path = parts.path.rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) != 2 or path.endswith((".zip", ".tar.gz", ".tgz")):
            return url
        return f"{parts.scheme}://{parts.netloc}/{segments[0]}/{segments[1]}/archive/refs/heads/main.zip"

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    def _download(self, url: str, dest_dir: Path) -> Path:
        """Download a file from URL to dest_dir, through the page fetch.

        The page fetch asks the web gate before every request and after every
        read, and reaches a public address only. The file is named after the
        last segment of the URL asked for, never after a redirect's.

        Returns the path to the downloaded file.
        Raises PluginInstallError on failure.
        """
        from opti_oignon.web_search import (
            DestinationRefused,
            PageFetchFailed,
            PageFetchRefused,
            fetch_page,
        )

        filename = urlsplit(url).path.rstrip("/").split("/")[-1]
        if not filename or len(filename) > 200:
            filename = "plugin_archive"
        # Ensure a proper extension
        if not any(filename.endswith(ext) for ext in (".zip", ".tar.gz", ".tgz")):
            filename += ".zip"

        dest_path = dest_dir / filename

        try:
            page = fetch_page(
                url,
                timeout=self._config["timeout_s"],
                max_bytes=self._max_bytes,
                user_agent="Opti-Oignon-PluginInstaller/1.0",
                decode=False,
            )
        except (PageFetchRefused, DestinationRefused, PageFetchFailed) as exc:
            raise PluginInstallError(str(exc)) from None
        if page.status != 200:
            raise PluginInstallError(f"HTTP status {page.status} downloading {url}")
        dest_path.write_bytes(page.body)

        logger.debug("Downloaded %d bytes to %s", len(page.body), dest_path)
        return dest_path

    # -----------------------------------------------------------------
    # Extraction
    # -----------------------------------------------------------------

    def _extract(self, archive_path: Path, dest_dir: Path) -> None:
        """Extract a zip or tar.gz archive to dest_dir.

        Raises PluginInstallError on failure.
        """
        name = archive_path.name.lower()

        try:
            if name.endswith(".zip"):
                import zipfile

                with zipfile.ZipFile(archive_path, "r") as zf:
                    # Security: check for path traversal
                    for member in zf.namelist():
                        resolved = (dest_dir / member).resolve()
                        # PI-08: is_relative_to avoids sibling-prefix
                        # collisions (e.g. /x/extracted vs /x/extracted_evil)
                        if not resolved.is_relative_to(dest_dir.resolve()):
                            raise PluginInstallError(
                                f"Zip path traversal detected: {member}"
                            )
                    zf.extractall(dest_dir)

            elif name.endswith((".tar.gz", ".tgz")):
                import tarfile

                with tarfile.open(archive_path, "r:gz") as tf:
                    # Security: check for path traversal
                    for member in tf.getmembers():
                        resolved = (dest_dir / member.name).resolve()
                        # PI-08: is_relative_to avoids sibling-prefix collisions
                        if not resolved.is_relative_to(dest_dir.resolve()):
                            raise PluginInstallError(
                                f"Tar path traversal detected: {member.name}"
                            )
                    tf.extractall(dest_dir, filter="data")

            else:
                raise PluginInstallError(
                    f"Unsupported archive format: {archive_path.name}"
                )
        except PluginInstallError:
            raise
        except Exception as exc:
            raise PluginInstallError(
                f"Failed to extract {archive_path.name}: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # Plugin root discovery
    # -----------------------------------------------------------------

    def _find_plugin_root(self, extract_dir: Path) -> Path | None:
        """Find the directory containing manifest.yaml in extracted files.

        Handles GitHub's pattern of wrapping in a repo-name-branch/ dir.
        Returns None if no manifest found.
        """
        # Direct manifest.yaml
        if (extract_dir / "manifest.yaml").exists():
            return extract_dir

        # One level deep (GitHub archive pattern)
        for child in extract_dir.iterdir():
            if child.is_dir() and (child / "manifest.yaml").exists():
                return child

        # Two levels deep (less common but possible)
        for child in extract_dir.iterdir():
            if child.is_dir():
                for grandchild in child.iterdir():
                    if grandchild.is_dir() and (grandchild / "manifest.yaml").exists():
                        return grandchild

        return None

    # -----------------------------------------------------------------
    # Manifest validation
    # -----------------------------------------------------------------

    def _validate_manifest(self, plugin_dir: Path) -> dict[str, Any]:
        """Read and validate the manifest.yaml in plugin_dir.

        Returns the parsed manifest dict.
        Raises PluginInstallError on failure.
        """
        manifest_path = plugin_dir / "manifest.yaml"

        try:
            import yaml
        except ImportError:
            raise PluginInstallError("PyYAML required for manifest validation")

        try:
            with open(manifest_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:
            raise PluginInstallError(
                f"Failed to parse manifest.yaml: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise PluginInstallError("manifest.yaml must be a YAML mapping")

        # Validate required fields
        required = ("name", "version", "author", "description", "entry_point")
        missing = [f for f in required if not data.get(f)]
        if missing:
            raise PluginInstallError(
                f"Manifest missing required fields: {', '.join(missing)}"
            )

        # Validate via PluginManifest (reuse existing validation)
        try:
            from opti_oignon.plugin_manifest import PluginManifest

            PluginManifest.from_dict(data)
        except Exception as exc:
            raise PluginInstallError(
                f"Manifest validation failed: {exc}"
            ) from exc

        # Verify entry point file exists
        entry = plugin_dir / data["entry_point"]
        if not entry.exists():
            raise PluginInstallError(
                f"Entry point file not found: {data['entry_point']}"
            )

        return data

    # -----------------------------------------------------------------
    # Hash verification
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute the SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


# =========================================================================
# Module-level singleton
# =========================================================================

PLUGIN_INSTALLER_AVAILABLE = True

try:
    from opti_oignon.config import DATA_DIR as _DATA_DIR

    _inst_plugins_dir = Path(_DATA_DIR) / "plugins"
    _inst_plugins_dir.mkdir(parents=True, exist_ok=True)

    # Import registry and loader if available
    _inst_registry = None
    _inst_loader = None
    _inst_index = None
    try:
        from opti_oignon.plugin_manifest import plugin_registry as _inst_registry
    except Exception:
        pass
    try:
        from opti_oignon.plugin_loader import plugin_loader as _inst_loader
    except Exception:
        pass
    try:
        from opti_oignon.plugin_index import plugin_index as _inst_index
    except Exception:
        pass

    remote_installer = RemotePluginInstaller(
        plugins_dir=_inst_plugins_dir,
        registry=_inst_registry,
        loader=_inst_loader,
        index=_inst_index,
    )
except Exception as _exc:
    logger.debug("RemotePluginInstaller singleton init deferred: %s", _exc)
    remote_installer = None  # type: ignore[assignment]
