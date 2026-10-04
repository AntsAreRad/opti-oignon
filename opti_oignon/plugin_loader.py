#!/usr/bin/env python3
"""
Plugin loader for Opti-Oignon.

PluginLoader: load plugins from directories, run each in its own worker
process (see plugin_subprocess and plugin_isolation; no plugin code runs in
the server), manage lifecycle (install, enable, disable, uninstall).
"""

import logging
import shutil
import types
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""


class LoadedPlugin:
    """A successfully loaded plugin with its module and metadata."""

    def __init__(
        self,
        name: str,
        version: str,
        module: types.ModuleType,
        plugin_dir: Path,
        hooks: dict[str, Any],
    ) -> None:
        self.name = name
        self.version = version
        self.module = module
        self.plugin_dir = plugin_dir
        self.hooks = hooks  # {hook_name: callable}
        self._initialized = False

    def initialize(self) -> None:
        """Call the plugin's init() function if it exists."""
        if self._initialized:
            return
        init_fn = getattr(self.module, "init", None)
        if callable(init_fn):
            try:
                init_fn()
            except Exception as exc:
                logger.warning(
                    "Plugin '%s' init() failed: %s", self.name, exc,
                )
        self._initialized = True

    def shutdown(self) -> None:
        """Call the plugin's shutdown() function if it exists."""
        shutdown_fn = getattr(self.module, "shutdown", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception as exc:
                logger.warning(
                    "Plugin '%s' shutdown() failed: %s", self.name, exc,
                )
        self._initialized = False

    def get_hook(self, hook_name: str) -> Any | None:
        """Get a hook callable by name, or None."""
        return self.hooks.get(hook_name)


class SubprocessPluginAdapter(LoadedPlugin):
    """Adapter that makes a subprocess-based plugin look like a LoadedPlugin.

    Translates hook calls into JSON-RPC over the subprocess IPC channel.
    Provides the same public interface as LoadedPlugin so the rest of the
    system (HookManager, PluginLoader) works unchanged.

    Parameters
    ----------
    name : str
        Plugin name.
    version : str
        Plugin version.
    plugin_dir : Path
        Plugin directory.
    hooks : dict[str, Any]
        Hook name -> RPC-proxied callable mapping.
    subprocess_manager : PluginSubprocessManager
        Reference to the manager that owns the subprocess.
    is_subprocess : bool
        Always True: every loaded plugin runs in its own worker process.
    """

    def __init__(
        self,
        name: str,
        version: str,
        plugin_dir: Path,
        hooks: dict[str, Any],
        subprocess_manager: Any,
    ) -> None:
        # Create a dummy module -- subprocess plugins don't have in-process modules
        dummy_module = types.ModuleType(f"_opti_subprocess_{name}")
        dummy_module.__plugin_name__ = name  # type: ignore[attr-defined]
        dummy_module.__plugin_version__ = version  # type: ignore[attr-defined]
        super().__init__(
            name=name,
            version=version,
            module=dummy_module,
            plugin_dir=plugin_dir,
            hooks=hooks,
        )
        self._subprocess_manager = subprocess_manager
        self.is_subprocess = True
        self._initialized = True  # Init done via RPC during start_plugin

    def initialize(self) -> None:
        """No-op: initialization is done during subprocess startup."""
        pass

    def shutdown(self) -> None:
        """Stop the subprocess."""
        try:
            self._subprocess_manager.stop_plugin(self.name)
        except Exception as exc:
            logger.warning(
                "Subprocess plugin '%s' shutdown failed: %s",
                self.name, exc,
            )
        self._initialized = False

    def get_hook(self, hook_name: str) -> Any | None:
        """Get a hook callable (RPC proxy) by name, or None."""
        return self.hooks.get(hook_name)


def _make_rpc_hook_proxy(
    plugin_name: str,
    hook_name: str,
    subprocess_manager: Any,
) -> Any:
    """Create a callable that proxies hook invocations over RPC.

    The returned function has the same signature as a normal hook callback:
    it accepts a HookContext (or dict) and returns a dict or None.
    """

    def _rpc_proxy(context: Any) -> dict[str, Any] | None:
        """Proxy a hook call to the plugin subprocess via JSON-RPC."""
        # Accept both HookContext objects and plain dicts
        if hasattr(context, "data"):
            data = {
                "hook_name": getattr(context, "hook_name", hook_name),
                "plugin_name": getattr(context, "plugin_name", plugin_name),
                "conversation_id": getattr(context, "conversation_id", None),
                "model": getattr(context, "model", None),
                "data": getattr(context, "data", {}),
                "config": getattr(context, "config", {}),
                "metadata": getattr(context, "metadata", {}),
            }
        elif isinstance(context, dict):
            data = context
        else:
            data = {}

        try:
            result = subprocess_manager.call_hook(
                plugin_name, hook_name, data,
            )
            if isinstance(result, dict):
                return result
            return None
        except Exception as exc:
            logger.warning(
                "RPC hook '%s' call to plugin '%s' failed: %s",
                hook_name, plugin_name, exc,
            )
            raise

    _rpc_proxy.__name__ = f"rpc_proxy_{plugin_name}_{hook_name}"
    _rpc_proxy.__qualname__ = f"rpc_proxy_{plugin_name}_{hook_name}"
    return _rpc_proxy


class PluginLoader:
    """Load and manage plugin lifecycles, each plugin in its own worker.

    Parameters
    ----------
    registry : PluginRegistry
        The plugin registry for state management.
    plugins_base_dir : Path or str or None
        Base directory where plugin directories are stored.
    subprocess_mode : str
        Plugin execution mode. ``"subprocess"`` -- the plugin runs in its own
        worker process, and a load that fails there leaves it unloaded -- is
        the only one: no mode runs plugin code inside the server, and any
        other value is refused when the loader is built.
    subprocess_manager : PluginSubprocessManager or None
        External subprocess manager instance. If None, a default manager
        will be created lazily.
    """

    EXECUTION_MODES = ("subprocess",)

    def __init__(
        self,
        registry: Any = None,
        plugins_base_dir: Path | str | None = None,
        *,
        subprocess_mode: str = "subprocess",
        subprocess_manager: Any = None,
    ) -> None:
        if subprocess_mode not in self.EXECUTION_MODES:
            raise ValueError(
                f"Unknown plugin execution mode {subprocess_mode!r}: "
                f"expected one of {', '.join(self.EXECUTION_MODES)}"
            )
        self._registry = registry
        self._base_dir = Path(plugins_base_dir) if plugins_base_dir else None
        self._loaded: dict[str, LoadedPlugin] = {}
        self._subprocess_mode = subprocess_mode
        self._subprocess_manager = subprocess_manager

    @property
    def loaded_plugins(self) -> dict[str, LoadedPlugin]:
        """Currently loaded plugins by name."""
        return dict(self._loaded)

    def load_plugin(self, plugin_dir: Path | str) -> LoadedPlugin:
        """Load a plugin from a directory containing manifest.yaml + entry point.

        The plugin runs in its own worker process. A load that fails there
        raises PluginLoadError naming the cause and leaves the plugin
        unloaded: nothing executes it inside the server.

        Parameters
        ----------
        plugin_dir : Path or str
            Directory containing the plugin files.

        Returns
        -------
        SubprocessPluginAdapter

        Raises
        ------
        PluginLoadError
            If the plugin cannot be loaded.
        """
        return self._load_plugin_subprocess(plugin_dir)

    def _get_subprocess_manager(self) -> Any:
        """Lazily obtain a PluginSubprocessManager instance."""
        if self._subprocess_manager is not None:
            return self._subprocess_manager

        try:
            from opti_oignon.plugin_subprocess import (
                PluginSubprocessManager,
            )
            self._subprocess_manager = PluginSubprocessManager()
            return self._subprocess_manager
        except ImportError as exc:
            raise PluginLoadError(
                f"plugin_subprocess module not available: {exc}"
            ) from exc

    def _load_plugin_subprocess(
        self,
        plugin_dir: Path | str,
    ) -> SubprocessPluginAdapter:
        """Load a plugin in an isolated subprocess.

        Returns a SubprocessPluginAdapter that presents the same
        interface as LoadedPlugin.
        """
        plugin_path = Path(plugin_dir).resolve()

        if not plugin_path.is_dir():
            raise PluginLoadError(f"Plugin directory not found: {plugin_path}")

        manifest_file = plugin_path / "manifest.yaml"
        if not manifest_file.exists():
            raise PluginLoadError(
                f"No manifest.yaml found in {plugin_path}"
            )

        try:
            import yaml
        except ImportError:
            raise PluginLoadError("PyYAML required for plugin loading")

        try:
            with open(manifest_file, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:
            raise PluginLoadError(
                f"Failed to parse manifest.yaml: {exc}"
            ) from exc

        from opti_oignon.plugin_manifest import PluginManifest, PluginManifestError
        try:
            manifest = PluginManifest.from_dict(data)
        except PluginManifestError as exc:
            raise PluginLoadError(f"Invalid manifest: {exc}") from exc

        # --- Bulbe mode allowlist check ---
        try:
            from opti_oignon.security_mode import is_bulbe
            if is_bulbe():
                from opti_oignon.plugin_allowlist import plugin_allowlist_manager
                result = plugin_allowlist_manager.verify_plugin(
                    plugin_id=manifest.name,
                    plugin_dir=plugin_path,
                    permissions=manifest.permissions,
                )
                if not result.get("allowed"):
                    reason = result.get("reason", "Not in allowlist")
                    logger.critical(
                        "BULBE MODE: Plugin '%s' REJECTED: %s",
                        manifest.name, reason,
                    )
                    raise PluginLoadError(
                        f"Bulbe mode: plugin '{manifest.name}' not allowed. "
                        f"{reason}"
                    )
        except ImportError:
            pass

        # Already loaded? Unload first
        if manifest.name in self._loaded:
            self.unload_plugin(manifest.name)

        # Get resource limits from manifest
        try:
            from opti_oignon.plugin_subprocess import PluginResourceLimits
            rlimits = PluginResourceLimits.from_manifest(data)
        except ImportError:
            rlimits = None

        # Launch subprocess
        mgr = self._get_subprocess_manager()
        try:
            mgr.start_plugin(
                plugin_name=manifest.name,
                plugin_dir=plugin_path,
                entry_point=manifest.entry_point,
                resource_limits=rlimits,
                permissions=tuple(manifest.permissions),
            )
        except Exception as exc:
            raise PluginLoadError(
                f"Failed to start subprocess for plugin "
                f"'{manifest.name}': {exc}"
            ) from exc

        # Build RPC-proxied hooks
        hooks: dict[str, Any] = {}
        for hook_name in manifest.hooks:
            hooks[hook_name] = _make_rpc_hook_proxy(
                manifest.name, hook_name, mgr,
            )

        adapter = SubprocessPluginAdapter(
            name=manifest.name,
            version=manifest.version,
            plugin_dir=plugin_path,
            hooks=hooks,
            subprocess_manager=mgr,
        )
        self._loaded[manifest.name] = adapter
        logger.info(
            "Loaded plugin '%s' v%s via subprocess (%d hooks)",
            manifest.name, manifest.version, len(hooks),
        )
        return adapter

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin, calling its shutdown() and cleaning up.

        Returns True if the plugin was loaded and removed.
        """
        loaded = self._loaded.pop(name, None)
        if loaded is None:
            return False

        self._unregister_hooks(name)
        loaded.shutdown()

        logger.info("Unloaded plugin '%s'", name)
        return True

    def install_plugin(
        self,
        source_dir: Path | str,
        *,
        auto_enable: bool = False,
    ) -> LoadedPlugin | None:
        """Install a plugin from a source directory.

        Copies files to the plugins base directory, registers with the
        registry, and optionally loads and enables it.

        Returns the LoadedPlugin if auto_enable is True and loading
        succeeds, otherwise None.
        """
        source = Path(source_dir).resolve()
        if not source.is_dir():
            raise PluginLoadError(f"Source directory not found: {source}")

        manifest_file = source / "manifest.yaml"
        if not manifest_file.exists():
            raise PluginLoadError(f"No manifest.yaml in {source}")

        try:
            import yaml
        except ImportError:
            raise PluginLoadError("PyYAML required")

        with open(manifest_file, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        from opti_oignon.plugin_manifest import PluginManifest
        manifest = PluginManifest.from_dict(data)

        # Copy to plugins_base_dir if different
        if self._base_dir:
            target = self._base_dir / manifest.name
            if source.resolve() != target.resolve():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
                plugin_dir = target
            else:
                plugin_dir = source
        else:
            plugin_dir = source

        # Register in registry
        if self._registry:
            # PI-10/PI-11: register as installed; the enable flow below
            # flips the state only after a successful load.
            self._registry.register(
                manifest, str(plugin_dir), auto_enable=False,
            )

        if auto_enable:
            # PI-11: route through the full enable flow so the plugin's
            # hooks are registered with the HookManager.  A bare
            # load_plugin() left freshly installed plugins with inactive
            # hooks until the next restart.
            if self._registry:
                return self.enable_plugin(manifest.name)
            loaded = self.load_plugin(plugin_dir)
            loaded.initialize()
            self._register_hooks(loaded)
            return loaded
        return None

    def uninstall_plugin(self, name: str) -> bool:
        """Uninstall a plugin: unload, unregister, optionally remove files.

        PI-15: returns True only when the plugin was actually unloaded
        or unregistered; unknown names report False.
        """
        # Unload if loaded
        unloaded = self.unload_plugin(name)

        # Unregister from registry
        unregistered = False
        if self._registry:
            record = self._registry.get(name)
            plugin_dir = Path(record.plugin_dir) if record else None
            unregistered = self._registry.unregister(name)

            # Remove files if in our managed base dir
            if (
                plugin_dir
                and self._base_dir
                and plugin_dir.is_relative_to(self._base_dir)
                and plugin_dir.is_dir()
            ):
                try:
                    shutil.rmtree(plugin_dir)
                    logger.info("Removed plugin directory: %s", plugin_dir)
                except OSError as exc:
                    logger.warning(
                        "Failed to remove plugin dir %s: %s",
                        plugin_dir, exc,
                    )
        return unloaded or unregistered

    # ------------------------------------------------------------------
    # Hook registration bridge
    # ------------------------------------------------------------------

    def _register_hooks(self, loaded: LoadedPlugin) -> int:
        """Register a loaded plugin's hooks with the global HookManager.

        Returns the number of hooks successfully registered.
        """
        try:
            from opti_oignon.plugin_hooks import hook_manager
        except ImportError:
            logger.debug("plugin_hooks not available, skipping hook registration")
            return 0

        count = 0
        for hook_name, callback in loaded.hooks.items():
            if hook_manager.register(hook_name, loaded.name, callback):
                count += 1
                logger.debug(
                    "Registered hook '%s' for plugin '%s'",
                    hook_name, loaded.name,
                )
        if count:
            logger.info(
                "Plugin '%s': %d hook(s) registered with HookManager",
                loaded.name, count,
            )
        return count

    def _unregister_hooks(self, name: str) -> int:
        """Remove all hooks for a plugin from the global HookManager.

        Returns the number of hooks removed.
        """
        try:
            from opti_oignon.plugin_hooks import hook_manager
        except ImportError:
            return 0
        removed = hook_manager.unregister_plugin(name)
        if removed:
            logger.info(
                "Plugin '%s': %d hook(s) unregistered from HookManager",
                name, removed,
            )
        return removed

    def enable_plugin(self, name: str) -> LoadedPlugin | None:
        """Enable a plugin: load it, register hooks, then mark enabled.

        PI-10: the registry state flips to "enabled" only AFTER a
        successful load, so a load failure cannot leave the registry
        claiming "enabled" with nothing actually loaded.

        Returns the LoadedPlugin or None if the plugin is not registered.
        """
        if self._registry:
            record = self._registry.get(name)
            if record is None:
                logger.warning("Cannot enable unknown plugin '%s'", name)
                return None
            loaded = self.load_plugin(record.plugin_dir)
            loaded.initialize()
            self._register_hooks(loaded)
            self._registry.set_state(name, "enabled")
            return loaded
        return None

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin: unload it, unregister hooks, set state to disabled.

        Returns True if successful.
        """
        self._unregister_hooks(name)
        self.unload_plugin(name)
        if self._registry:
            return self._registry.set_state(name, "disabled")
        return False

    def load_all_enabled(self) -> list[LoadedPlugin]:
        """Load and initialize all plugins that are in 'enabled' state.

        Returns list of successfully loaded plugins.
        """
        if not self._registry:
            return []

        from opti_oignon.plugin_manifest import PLUGIN_STATE_ENABLED

        loaded: list[LoadedPlugin] = []
        for record in self._registry.list_plugins(state=PLUGIN_STATE_ENABLED):
            try:
                plugin = self.load_plugin(record.plugin_dir)
                plugin.initialize()
                self._register_hooks(plugin)
                loaded.append(plugin)
            except Exception as exc:
                logger.warning(
                    "Failed to load enabled plugin '%s': %s",
                    record.manifest.name, exc,
                )
        return loaded

    def shutdown_all(self) -> None:
        """Shutdown and unload all loaded plugins.

        Also stops the subprocess manager watchdog if running.
        """
        for name in list(self._loaded.keys()):
            self.unload_plugin(name)

        # Stop subprocess watchdog if active
        if self._subprocess_manager is not None:
            try:
                self._subprocess_manager.stop_watchdog()
            except Exception:
                pass


# =========================================================================
# Module-level singleton
# =========================================================================

PLUGIN_LOADER_AVAILABLE = True

try:
    from opti_oignon.config import DATA_DIR as _DATA_DIR
    from opti_oignon.plugin_manifest import plugin_registry as _registry

    _plugins_dir = Path(_DATA_DIR) / "plugins"
    _plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_loader = PluginLoader(
        registry=_registry,
        plugins_base_dir=_plugins_dir,
    )
except Exception as _exc:
    logger.debug("PluginLoader singleton init deferred: %s", _exc)
    plugin_loader = None  # type: ignore[assignment]
