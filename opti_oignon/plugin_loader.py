#!/usr/bin/env python3
"""
Plugin loader for Opti-Oignon (S101).

PluginLoader: load plugins from directories, execute them in a restricted
sandbox (no host filesystem access outside plugin dir, no network by
default), manage lifecycle (install, enable, disable, uninstall).
"""

import importlib.util
import io
import logging
import os
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Modules that plugins are NOT allowed to import
_BLOCKED_IMPORTS = frozenset({
    # Original S101/S106 blocks
    "subprocess",
    "shutil",
    "ctypes",
    "multiprocessing",
    "signal",
    "resource",
    "pty",
    "fcntl",
    "termios",
    "readline",
    "code",
    "codeop",
    "compileall",
    "py_compile",
    "importlib",      # importlib.import_module() bypass (S106)
    # S124: Critical additions — system access
    "os",             # os.system(), os.popen(), os.environ, os.listdir
    "sys",            # sys.modules manipulation, sys._getframe
    "pathlib",        # Path.read_text() / .write_text() bypass builtins.open
    "io",             # io.open() bypasses builtins.open patch
    "glob",           # File enumeration
    # S124: Critical additions — code execution via deserialization
    "pickle",         # Arbitrary code execution via __reduce__
    "shelve",         # Uses pickle internally
    "marshal",        # Low-level serialization, code object execution
    # S124: Critical additions — introspection / memory access
    "gc",             # gc.get_objects() exposes all Python objects in memory
    "inspect",        # Read source code of any loaded module
    "dis",            # Disassemble bytecode of any function
    "ast",            # Parse + compile arbitrary code
    # S124: Critical additions — import system manipulation
    "zipimport",      # Load code from zip files
    "pkgutil",        # Package utilities, import scanning
    "runpy",          # Run modules as scripts
    # S124: Miscellaneous dangerous modules
    "webbrowser",     # Open URLs (information disclosure)
    "antigravity",    # Opens browser via webbrowser
    "turtle",         # Can open GUI windows
    "tkinter",        # GUI access
})

# Modules conditionally blocked unless plugin has network_outbound permission
_NETWORK_MODULES = frozenset({
    "socket",
    "http",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "websocket",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "xmlrpc",
})


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""


class PluginSandboxViolation(Exception):
    """Raised when a plugin attempts a forbidden operation."""


class _RestrictedImporter:
    """Meta-path finder that blocks forbidden imports for plugin modules.

    Installed into sys.meta_path during plugin loading and removed after.
    Uses find_spec (Python 3.4+) instead of deprecated find_module.
    """

    def __init__(
        self,
        blocked: frozenset[str],
        network_blocked: frozenset[str],
        has_network_permission: bool = False,
    ) -> None:
        self._blocked = blocked
        self._network_blocked = network_blocked
        self._has_network = has_network_permission
        self._active = True

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        """Python 3 meta-path protocol: raise if blocked, return None to allow."""
        if not self._active:
            return None
        top = fullname.split(".")[0]
        if top in self._blocked:
            raise PluginSandboxViolation(
                f"Plugin attempted to import blocked module: '{fullname}'"
            )
        if not self._has_network and top in self._network_blocked:
            raise PluginSandboxViolation(
                f"Plugin attempted to import blocked module: '{fullname}'"
            )
        return None

    def deactivate(self) -> None:
        """Deactivate this importer so it stops blocking."""
        self._active = False


class _RestrictedPathAccessor:
    """Context manager that patches file access paths to restrict plugins.

    Patches builtins.open, io.open, and pathlib.Path file methods so that
    only files within the plugin directory and standard library / site-packages
    can be accessed.  This is defense-in-depth: the primary defense is
    blocking os/pathlib/io at import time via _RestrictedImporter.

    S124: Extended to cover pathlib.Path.open / read_text / read_bytes /
    write_text / write_bytes / iterdir / glob / rglob and io.open.
    """

    def __init__(self, allowed_dirs: list[Path]) -> None:
        self._allowed = [p.resolve() for p in allowed_dirs]
        self._original_open: Any = None
        self._original_io_open: Any = None
        self._original_path_open: Any = None
        self._original_path_read_text: Any = None
        self._original_path_read_bytes: Any = None
        self._original_path_write_text: Any = None
        self._original_path_write_bytes: Any = None
        self._original_path_iterdir: Any = None
        self._original_path_glob: Any = None
        self._original_path_rglob: Any = None
        # Capture module refs so __exit__ works even with __import__ blocked
        self._builtins_module: Any = None
        self._io_module: Any = None

    def _is_allowed(self, filepath: str | Path) -> bool:
        """Check if a file path is within an allowed directory."""
        try:
            resolved = Path(filepath).resolve()
        except (OSError, ValueError):
            return False
        for allowed in self._allowed:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        # Allow stdlib and site-packages
        for sp in sys.path:
            if sp and resolved.is_relative_to(Path(sp).resolve()):
                return True
        return False

    def _is_dir_allowed(self, dirpath: Path) -> bool:
        """Check if a directory path is within an allowed directory."""
        try:
            resolved = dirpath.resolve()
        except (OSError, ValueError):
            return False
        for allowed in self._allowed:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        for sp in sys.path:
            if sp and resolved.is_relative_to(Path(sp).resolve()):
                return True
        return False

    def __enter__(self) -> "_RestrictedPathAccessor":
        import builtins
        # PI-25: reference the module-level `io` (imported before any
        # import blocking) instead of `import io` here -- io is hidden
        # from sys.modules and the active importer would otherwise raise
        # a PluginSandboxViolation on the loader's own setup.
        _io = io

        self._builtins_module = builtins
        self._io_module = _io

        self._original_open = builtins.open
        allowed_check = self._is_allowed
        dir_allowed_check = self._is_dir_allowed
        original = self._original_open

        # --- builtins.open ---
        def restricted_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, (str, Path)):
                if not allowed_check(str(file)):
                    raise PluginSandboxViolation(
                        f"Plugin attempted to access restricted path: {file}"
                    )
            return original(file, *args, **kwargs)

        builtins.open = restricted_open  # type: ignore[assignment]

        # --- io.open ---
        self._original_io_open = _io.open

        def restricted_io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, (str, Path)):
                if not allowed_check(str(file)):
                    raise PluginSandboxViolation(
                        f"Plugin attempted io.open on restricted path: {file}"
                    )
            return self._original_io_open(file, *args, **kwargs)

        _io.open = restricted_io_open  # type: ignore[assignment]

        # --- pathlib.Path methods ---
        _Path = Path

        self._original_path_open = _Path.open

        def restricted_path_open(self_path: Any, *args: Any, **kwargs: Any) -> Any:
            if not allowed_check(str(self_path)):
                raise PluginSandboxViolation(
                    f"Plugin attempted Path.open on restricted path: {self_path}"
                )
            return self._original_path_open(self_path, *args, **kwargs)

        _Path.open = restricted_path_open  # type: ignore[assignment]

        self._original_path_read_text = _Path.read_text

        def restricted_read_text(self_path: Any, *args: Any, **kwargs: Any) -> str:
            if not allowed_check(str(self_path)):
                raise PluginSandboxViolation(
                    f"Plugin attempted Path.read_text on restricted path: {self_path}"
                )
            return self._original_path_read_text(self_path, *args, **kwargs)

        _Path.read_text = restricted_read_text  # type: ignore[assignment]

        self._original_path_read_bytes = _Path.read_bytes

        def restricted_read_bytes(self_path: Any, *args: Any, **kwargs: Any) -> bytes:
            if not allowed_check(str(self_path)):
                raise PluginSandboxViolation(
                    f"Plugin attempted Path.read_bytes on restricted path: {self_path}"
                )
            return self._original_path_read_bytes(self_path, *args, **kwargs)

        _Path.read_bytes = restricted_read_bytes  # type: ignore[assignment]

        self._original_path_write_text = _Path.write_text

        def restricted_write_text(self_path: Any, *args: Any, **kwargs: Any) -> Any:
            if not allowed_check(str(self_path)):
                raise PluginSandboxViolation(
                    f"Plugin attempted Path.write_text on restricted path: {self_path}"
                )
            return self._original_path_write_text(self_path, *args, **kwargs)

        _Path.write_text = restricted_write_text  # type: ignore[assignment]

        self._original_path_write_bytes = _Path.write_bytes

        def restricted_write_bytes(self_path: Any, *args: Any, **kwargs: Any) -> Any:
            if not allowed_check(str(self_path)):
                raise PluginSandboxViolation(
                    f"Plugin attempted Path.write_bytes on restricted path: {self_path}"
                )
            return self._original_path_write_bytes(self_path, *args, **kwargs)

        _Path.write_bytes = restricted_write_bytes  # type: ignore[assignment]

        self._original_path_iterdir = _Path.iterdir

        def restricted_iterdir(self_path: Any) -> Any:
            if not dir_allowed_check(self_path):
                raise PluginSandboxViolation(
                    f"Plugin attempted Path.iterdir on restricted path: {self_path}"
                )
            return self._original_path_iterdir(self_path)

        _Path.iterdir = restricted_iterdir  # type: ignore[assignment]

        self._original_path_glob = _Path.glob

        def restricted_glob(self_path: Any, pattern: str, *args: Any, **kwargs: Any) -> Any:
            if not dir_allowed_check(self_path):
                raise PluginSandboxViolation(
                    f"Plugin attempted Path.glob on restricted path: {self_path}"
                )
            return self._original_path_glob(self_path, pattern, *args, **kwargs)

        _Path.glob = restricted_glob  # type: ignore[assignment]

        self._original_path_rglob = _Path.rglob

        def restricted_rglob(self_path: Any, pattern: str, *args: Any, **kwargs: Any) -> Any:
            if not dir_allowed_check(self_path):
                raise PluginSandboxViolation(
                    f"Plugin attempted Path.rglob on restricted path: {self_path}"
                )
            return self._original_path_rglob(self_path, pattern, *args, **kwargs)

        _Path.rglob = restricted_rglob  # type: ignore[assignment]

        return self

    def __exit__(self, *exc: Any) -> None:
        builtins = self._builtins_module
        _io = self._io_module

        if builtins is not None and self._original_open is not None:
            builtins.open = self._original_open  # type: ignore[assignment]
        if _io is not None and self._original_io_open is not None:
            _io.open = self._original_io_open  # type: ignore[assignment]

        _Path = Path
        if self._original_path_open is not None:
            _Path.open = self._original_path_open  # type: ignore[assignment]
        if self._original_path_read_text is not None:
            _Path.read_text = self._original_path_read_text  # type: ignore[assignment]
        if self._original_path_read_bytes is not None:
            _Path.read_bytes = self._original_path_read_bytes  # type: ignore[assignment]
        if self._original_path_write_text is not None:
            _Path.write_text = self._original_path_write_text  # type: ignore[assignment]
        if self._original_path_write_bytes is not None:
            _Path.write_bytes = self._original_path_write_bytes  # type: ignore[assignment]
        if self._original_path_iterdir is not None:
            _Path.iterdir = self._original_path_iterdir  # type: ignore[assignment]
        if self._original_path_glob is not None:
            _Path.glob = self._original_path_glob  # type: ignore[assignment]
        if self._original_path_rglob is not None:
            _Path.rglob = self._original_path_rglob  # type: ignore[assignment]


# Builtins that plugins are NOT allowed to use (S124 Phase 2c).
# __import__ is handled separately with a selective wrapper.
# exec/eval/compile are blocked at the builtins level but with a
# caller-check so Python's own import machinery still works.
_BLOCKED_BUILTINS_TOTAL = frozenset({
    "globals",      # Access the full global namespace
    "vars",         # Access object dictionaries
})

# These builtins are blocked with a caller check: only raise if
# called from plugin code (module name starting with _opti_plugin_)
_BLOCKED_BUILTINS_CALLSITE = frozenset({
    "exec",         # Execute arbitrary code strings
    "eval",         # Evaluate arbitrary expressions
    "compile",      # Compile code objects
})


class _RestrictedBuiltins:
    """Context manager that blocks dangerous builtins during plugin loading.

    - globals, vars: blocked entirely (raise PluginSandboxViolation)
    - exec, eval, compile: blocked when called from plugin code
      (detected by checking the caller's module name), but allowed
      when called from Python internals (import system, etc.)
    - __import__: selective wrapper that blocks _BLOCKED_IMPORTS modules
      but allows safe module imports

    Restored on exit.
    """

    def __init__(self) -> None:
        self._originals: dict[str, Any] = {}
        self._builtins_module: Any = None

    def __enter__(self) -> "_RestrictedBuiltins":
        import builtins
        self._builtins_module = builtins

        # Block globals/vars entirely
        for name in _BLOCKED_BUILTINS_TOTAL:
            original = getattr(builtins, name, None)
            if original is not None:
                self._originals[name] = original

                def _make_blocker(blocked_name: str) -> Any:
                    def _blocked(*args: Any, **kwargs: Any) -> Any:
                        raise PluginSandboxViolation(
                            f"Plugin attempted to call blocked builtin: {blocked_name}()"
                        )
                    return _blocked

                setattr(builtins, name, _make_blocker(name))

        # Block exec/eval/compile with caller check
        # PI-25: use the module-level `sys` (sys is hidden during a
        # sandboxed load; re-importing it would trip the importer).
        _sys = sys
        for name in _BLOCKED_BUILTINS_CALLSITE:
            original = getattr(builtins, name, None)
            if original is not None:
                self._originals[name] = original

                def _make_callsite_blocker(
                    blocked_name: str, orig_fn: Any
                ) -> Any:
                    def _blocked(*args: Any, **kwargs: Any) -> Any:
                        # Check if caller is plugin code
                        frame = _sys._getframe(1)
                        caller_module = frame.f_globals.get(
                            "__name__", ""
                        )
                        if caller_module.startswith("_opti_plugin_"):
                            raise PluginSandboxViolation(
                                f"Plugin attempted to call blocked "
                                f"builtin: {blocked_name}()"
                            )
                        return orig_fn(*args, **kwargs)
                    return _blocked

                setattr(
                    builtins, name,
                    _make_callsite_blocker(name, original),
                )

        # Selective __import__ wrapper: blocks _BLOCKED_IMPORTS modules
        # but allows internal Python imports to function normally
        original_import = builtins.__import__
        self._originals["__import__"] = original_import
        blocked_set = _BLOCKED_IMPORTS

        def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
            top = name.split(".")[0]
            if top in blocked_set:
                raise PluginSandboxViolation(
                    f"Plugin attempted to import blocked module via "
                    f"__import__: '{name}'"
                )
            return original_import(name, *args, **kwargs)

        builtins.__import__ = _restricted_import  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        builtins = self._builtins_module
        if builtins is None:
            return
        for name, original in self._originals.items():
            setattr(builtins, name, original)
        self._originals.clear()


class LoadedPlugin:
    """A successfully loaded plugin with its module and metadata."""

    def __init__(
        self,
        name: str,
        version: str,
        module: types.ModuleType,
        plugin_dir: Path,
        hooks: dict[str, Any],
        importer: Optional[_RestrictedImporter] = None,
    ) -> None:
        self.name = name
        self.version = version
        self.module = module
        self.plugin_dir = plugin_dir
        self.hooks = hooks  # {hook_name: callable}
        self._importer = importer
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
        if self._importer:
            self._importer.deactivate()
        self._initialized = False

    def get_hook(self, hook_name: str) -> Optional[Any]:
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
        Always True, used to distinguish from in-process LoadedPlugins.
    """

    def __init__(
        self,
        name: str,
        version: str,
        plugin_dir: Path,
        hooks: dict[str, Any],
        subprocess_manager: Any,
    ) -> None:
        # Create a dummy module — subprocess plugins don't have in-process modules
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

    def get_hook(self, hook_name: str) -> Optional[Any]:
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

    def _rpc_proxy(context: Any) -> Optional[dict[str, Any]]:
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
    """Load and manage plugin lifecycles with sandboxed execution.

    Parameters
    ----------
    registry : PluginRegistry
        The plugin registry for state management.
    plugins_base_dir : Path or str or None
        Base directory where plugin directories are stored.
    subprocess_mode : str
        Plugin execution mode:
        - ``"auto"`` — try subprocess, fall back to in-process (default)
        - ``"subprocess"`` — force subprocess only (no fallback)
        - ``"inprocess"`` — force in-process only (legacy behavior)
    subprocess_manager : PluginSubprocessManager or None
        External subprocess manager instance.  If None and subprocess_mode
        is not ``"inprocess"``, a default manager will be created lazily.
    """

    def __init__(
        self,
        registry: Any = None,
        plugins_base_dir: Path | str | None = None,
        *,
        subprocess_mode: str = "auto",
        subprocess_manager: Any = None,
    ) -> None:
        self._registry = registry
        self._base_dir = Path(plugins_base_dir) if plugins_base_dir else None
        self._loaded: dict[str, LoadedPlugin] = {}
        self._subprocess_mode = subprocess_mode
        self._subprocess_manager = subprocess_manager

    @property
    def loaded_plugins(self) -> dict[str, LoadedPlugin]:
        """Currently loaded plugins by name."""
        return dict(self._loaded)

    def load_plugin(
        self,
        plugin_dir: Path | str,
        *,
        sandbox: bool = True,
    ) -> LoadedPlugin:
        """Load a plugin from a directory containing manifest.yaml + entry point.

        Depending on ``subprocess_mode``, this will attempt to run the
        plugin in an isolated subprocess.  If subprocess loading fails
        and mode is ``"auto"``, falls back to in-process loading with
        a warning.

        Parameters
        ----------
        plugin_dir : Path or str
            Directory containing the plugin files.
        sandbox : bool
            Whether to apply import and filesystem restrictions
            (only relevant for in-process fallback).

        Returns
        -------
        LoadedPlugin or SubprocessPluginAdapter

        Raises
        ------
        PluginLoadError
            If the plugin cannot be loaded.
        """
        mode = self._subprocess_mode

        # In Bulbe mode, force subprocess-only unless explicitly inprocess
        if mode == "auto":
            try:
                from opti_oignon.security_mode import is_bulbe
                if is_bulbe():
                    mode = "subprocess"
                    logger.info(
                        "Bulbe mode active: forcing subprocess isolation"
                    )
            except ImportError:
                pass

        if mode == "subprocess":
            return self._load_plugin_subprocess(plugin_dir)

        if mode == "inprocess":
            return self._load_plugin_inprocess(plugin_dir, sandbox=sandbox)

        # mode == "auto": try subprocess, fall back to inprocess
        try:
            return self._load_plugin_subprocess(plugin_dir)
        except Exception as exc:
            logger.warning(
                "Subprocess loading failed for '%s', falling back to "
                "in-process: %s",
                plugin_dir, exc,
            )
            return self._load_plugin_inprocess(plugin_dir, sandbox=sandbox)

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
            with open(manifest_file, "r", encoding="utf-8") as fh:
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

        # --- S126: Bulbe mode allowlist check ---
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

    def _load_plugin_inprocess(
        self,
        plugin_dir: Path | str,
        *,
        sandbox: bool = True,
    ) -> LoadedPlugin:
        """Load a plugin in-process (legacy behavior).

        Parameters
        ----------
        plugin_dir : Path or str
            Directory containing the plugin files.
        sandbox : bool
            Whether to apply import and filesystem restrictions.

        Returns
        -------
        LoadedPlugin

        Raises
        ------
        PluginLoadError
            If the plugin cannot be loaded.
        """
        plugin_path = Path(plugin_dir).resolve()

        if not plugin_path.is_dir():
            raise PluginLoadError(f"Plugin directory not found: {plugin_path}")

        # Load manifest
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
            with open(manifest_file, "r", encoding="utf-8") as fh:
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

        # --- S126: Bulbe mode allowlist check ---
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
                logger.info(
                    "BULBE MODE: Plugin '%s' allowlist verified",
                    manifest.name,
                )
        except ImportError:
            pass  # security_mode not available, skip check

        # Check entry point exists
        entry_file = plugin_path / manifest.entry_point
        if not entry_file.exists():
            raise PluginLoadError(
                f"Entry point not found: {entry_file}"
            )

        # Already loaded?
        if manifest.name in self._loaded:
            logger.info(
                "Plugin '%s' already loaded, reloading", manifest.name,
            )
            self.unload_plugin(manifest.name)

        # Setup sandbox
        importer = None
        hidden_modules: dict[str, types.ModuleType] = {}
        has_network = "network_outbound" in manifest.permissions
        if sandbox:
            importer = _RestrictedImporter(
                blocked=_BLOCKED_IMPORTS,
                network_blocked=_NETWORK_MODULES,
                has_network_permission=has_network,
            )
            sys.meta_path.insert(0, importer)
            # Temporarily hide blocked modules from sys.modules so
            # the meta-path finder is actually invoked for them
            all_blocked = set(_BLOCKED_IMPORTS)
            if not has_network:
                all_blocked |= set(_NETWORK_MODULES)
            for mod_name in list(sys.modules.keys()):
                top = mod_name.split(".")[0]
                if top in all_blocked:
                    hidden_modules[mod_name] = sys.modules.pop(mod_name)

        # Load the module
        try:
            module_name = f"_opti_plugin_{manifest.name}"
            spec = importlib.util.spec_from_file_location(
                module_name, str(entry_file),
            )
            if spec is None or spec.loader is None:
                raise PluginLoadError(
                    f"Cannot create module spec for {entry_file}"
                )
            module = importlib.util.module_from_spec(spec)

            # Inject plugin metadata into the module namespace
            module.__plugin_name__ = manifest.name  # type: ignore[attr-defined]
            module.__plugin_version__ = manifest.version  # type: ignore[attr-defined]
            module.__plugin_dir__ = str(plugin_path)  # type: ignore[attr-defined]

            sys.modules[module_name] = module

            if sandbox:
                allowed_dirs = [plugin_path]
                with _RestrictedPathAccessor(allowed_dirs), _RestrictedBuiltins():
                    spec.loader.exec_module(module)
            else:
                spec.loader.exec_module(module)

        except PluginSandboxViolation:
            raise
        except PluginLoadError:
            raise
        except Exception as exc:
            raise PluginLoadError(
                f"Failed to load plugin '{manifest.name}': {exc}"
            ) from exc
        finally:
            # Remove the importer from meta_path after loading
            if importer and importer in sys.meta_path:
                sys.meta_path.remove(importer)
            # Restore hidden modules
            sys.modules.update(hidden_modules)

        # Collect hook callables from the module
        hooks: dict[str, Any] = {}
        for hook_name in manifest.hooks:
            # Look for hook_<name> function or a HOOKS dict
            hooks_dict = getattr(module, "HOOKS", None)
            if isinstance(hooks_dict, dict) and hook_name in hooks_dict:
                hooks[hook_name] = hooks_dict[hook_name]
            else:
                fn_name = f"hook_{hook_name}"
                fn = getattr(module, fn_name, None)
                if callable(fn):
                    hooks[hook_name] = fn

        loaded = LoadedPlugin(
            name=manifest.name,
            version=manifest.version,
            module=module,
            plugin_dir=plugin_path,
            hooks=hooks,
            importer=importer,
        )
        self._loaded[manifest.name] = loaded
        logger.info(
            "Loaded plugin '%s' v%s (%d hooks)",
            manifest.name, manifest.version, len(hooks),
        )
        return loaded

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin, calling its shutdown() and cleaning up.

        Handles both in-process and subprocess plugins.
        Returns True if the plugin was loaded and removed.
        """
        loaded = self._loaded.pop(name, None)
        if loaded is None:
            return False

        self._unregister_hooks(name)
        loaded.shutdown()

        # Clean up sys.modules (in-process plugins only)
        if not isinstance(loaded, SubprocessPluginAdapter):
            module_name = f"_opti_plugin_{name}"
            sys.modules.pop(module_name, None)

        logger.info("Unloaded plugin '%s'", name)
        return True

    def install_plugin(
        self,
        source_dir: Path | str,
        *,
        auto_enable: bool = False,
    ) -> Optional[LoadedPlugin]:
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

        with open(manifest_file, "r", encoding="utf-8") as fh:
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
    # Hook registration bridge (S114)
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

    def enable_plugin(self, name: str) -> Optional[LoadedPlugin]:
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
    from opti_oignon.plugin_manifest import plugin_registry as _registry
    from opti_oignon.config import DATA_DIR as _DATA_DIR

    _plugins_dir = Path(_DATA_DIR) / "plugins"
    _plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_loader = PluginLoader(
        registry=_registry,
        plugins_base_dir=_plugins_dir,
    )
except Exception as _exc:
    logger.debug("PluginLoader singleton init deferred: %s", _exc)
    plugin_loader = None  # type: ignore[assignment]
