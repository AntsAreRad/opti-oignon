#!/usr/bin/env python3
"""
Plugin management API routes.

GET    /api/plugins                — List installed plugins
POST   /api/plugins/install       — Install from directory/archive
POST   /api/plugins/{name}/enable — Enable plugin
POST   /api/plugins/{name}/disable — Disable plugin
DELETE /api/plugins/{name}        — Uninstall plugin
GET    /api/plugins/{name}/config — Get plugin config
PUT    /api/plugins/{name}/config — Update plugin config
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Audit fix: require authentication for all endpoints
try:
    from .routes_auth import _get_current_user
    _auth_dep = [Depends(_get_current_user)]
except ImportError:
    _auth_dep = []

router = APIRouter(prefix="/api/plugins", tags=["plugins"], dependencies=_auth_dep)


# =========================================================================
# PYDANTIC SCHEMAS
# =========================================================================

class PluginInfo(BaseModel):
    name: str
    version: str
    author: str
    description: str
    entry_point: str
    hooks: list[str] = []
    dependencies: list[str] = []
    permissions: list[str] = []
    state: str = "installed"
    plugin_dir: str = ""
    installed_at: float = 0.0
    updated_at: float = 0.0


class PluginListResponse(BaseModel):
    plugins: list[PluginInfo]
    total: int
    enabled: int


class InstallRequest(BaseModel):
    source_dir: str = Field(..., description="Path to the plugin directory to install")
    auto_enable: bool = Field(False, description="Automatically enable after install")


class InstallResponse(BaseModel):
    success: bool
    name: str = ""
    version: str = ""
    message: str = ""
    error: str | None = None


class StateChangeResponse(BaseModel):
    success: bool
    name: str
    state: str
    message: str = ""
    error: str | None = None


class UninstallResponse(BaseModel):
    success: bool
    name: str
    message: str = ""
    error: str | None = None


class PluginConfigResponse(BaseModel):
    name: str
    config: dict[str, Any] = {}
    config_schema: dict[str, Any] = {}


class UpdateConfigRequest(BaseModel):
    config: dict[str, Any] = Field(..., description="New plugin configuration")


class UpdateConfigResponse(BaseModel):
    success: bool
    name: str
    config: dict[str, Any] = {}
    message: str = ""
    error: str | None = None


# =========================================================================
# HELPERS
# =========================================================================

def _get_registry():
    """Get the plugin registry singleton, raise 503 if unavailable."""
    try:
        from opti_oignon.api.deps import (
            PLUGIN_REGISTRY_AVAILABLE,
            plugin_registry_instance,
        )
        if not PLUGIN_REGISTRY_AVAILABLE or plugin_registry_instance is None:
            raise HTTPException(
                status_code=503,
                detail="Plugin registry not available",
            )
        return plugin_registry_instance
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Plugin system not available",
        )


def _get_loader():
    """Get the plugin loader singleton, raise 503 if unavailable."""
    try:
        from opti_oignon.api.deps import (
            PLUGIN_LOADER_AVAILABLE,
            plugin_loader_instance,
        )
        if not PLUGIN_LOADER_AVAILABLE or plugin_loader_instance is None:
            raise HTTPException(
                status_code=503,
                detail="Plugin loader not available",
            )
        return plugin_loader_instance
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Plugin system not available",
        )


def _record_to_info(record: Any) -> PluginInfo:
    """Convert a PluginRecord to a PluginInfo response model."""
    m = record.manifest
    return PluginInfo(
        name=m.name,
        version=m.version,
        author=m.author,
        description=m.description,
        entry_point=m.entry_point,
        hooks=m.hooks,
        dependencies=m.dependencies,
        permissions=m.permissions,
        state=record.state,
        plugin_dir=record.plugin_dir,
        installed_at=record.installed_at,
        updated_at=record.updated_at,
    )


# =========================================================================
# ENDPOINTS
# =========================================================================

@router.get("", response_model=PluginListResponse)
def list_plugins(
    state: str | None = Query(None, description="Filter by state: installed, enabled, disabled"),
) -> dict:
    """List all installed plugins with optional state filter."""
    registry = _get_registry()
    records = registry.list_plugins(state=state)
    plugins = [_record_to_info(r) for r in records]
    enabled_count = sum(1 for r in records if r.state == "enabled")
    return PluginListResponse(
        plugins=plugins,
        total=len(plugins),
        enabled=enabled_count,
    )


@router.post("/install", response_model=InstallResponse)
def install_plugin(req: InstallRequest) -> dict:
    """Install a plugin from a directory path."""
    loader = _get_loader()

    try:
        loaded = loader.install_plugin(
            req.source_dir,
            auto_enable=req.auto_enable,
        )
        # Get the manifest info from the registry
        registry = _get_registry()  # noqa: F841

        # Determine the plugin name from the loaded result or by scanning
        if loaded:
            name = loaded.name
            version = loaded.version
        else:
            # Parse manifest to get name
            from pathlib import Path
            manifest_path = Path(req.source_dir) / "manifest.yaml"
            if manifest_path.exists():
                import yaml
                with open(manifest_path, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                name = data.get("name", "unknown")
                version = data.get("version", "0.0.0")
            else:
                name = "unknown"
                version = "0.0.0"

        msg = f"Plugin '{name}' v{version} installed"
        if req.auto_enable:
            msg += " and enabled"

        return InstallResponse(
            success=True,
            name=name,
            version=version,
            message=msg,
        )
    except Exception as exc:
        logger.warning("Plugin install failed: %s", exc)
        return InstallResponse(
            success=False,
            error=str(exc),
            message=f"Installation failed: {exc}",
        )


# Debug endpoint for tracing hook registration and execution
@router.get("/debug")
def plugin_debug() -> dict:
    """Diagnostic endpoint: shows registered hooks and execution stats.

    Useful for debugging why plugin effects are not applied during inference.
    """
    try:
        from opti_oignon.plugin_hooks import hook_manager
    except ImportError:
        return {"available": False, "error": "plugin_hooks module not available"}

    try:
        from opti_oignon.plugin_loader import plugin_loader as _pl
    except ImportError:
        _pl = None

    loaded_plugins = {}
    if _pl is not None:
        for name, lp in _pl.loaded_plugins.items():
            loaded_plugins[name] = {
                "version": lp.version,
                "initialized": lp._initialized,
                "hooks_defined": list(lp.hooks.keys()),
            }

    return {
        "available": True,
        "hook_manager": {
            "registered_hooks": hook_manager.list_hooks(),
            "total_hooks": hook_manager.get_hook_count(),
            "execution_stats": hook_manager.get_stats(),
        },
        "loaded_plugins": loaded_plugins,
    }


@router.post("/{name}/enable", response_model=StateChangeResponse)
def enable_plugin(name: str) -> dict:
    """Enable an installed plugin."""
    registry = _get_registry()
    record = registry.get(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")

    if record.state == "enabled":
        return StateChangeResponse(
            success=True,
            name=name,
            state="enabled",
            message=f"Plugin '{name}' is already enabled",
        )

    loader = _get_loader()
    try:
        loader.enable_plugin(name)
        return StateChangeResponse(
            success=True,
            name=name,
            state="enabled",
            message=f"Plugin '{name}' enabled successfully",
        )
    except Exception as exc:
        logger.warning("Failed to enable plugin '%s': %s", name, exc)
        return StateChangeResponse(
            success=False,
            name=name,
            state=record.state,
            error=str(exc),
            message=f"Failed to enable: {exc}",
        )


@router.post("/{name}/disable", response_model=StateChangeResponse)
def disable_plugin(name: str) -> dict:
    """Disable an enabled plugin."""
    registry = _get_registry()
    record = registry.get(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")

    if record.state == "disabled":
        return StateChangeResponse(
            success=True,
            name=name,
            state="disabled",
            message=f"Plugin '{name}' is already disabled",
        )

    loader = _get_loader()
    try:
        loader.disable_plugin(name)
        return StateChangeResponse(
            success=True,
            name=name,
            state="disabled",
            message=f"Plugin '{name}' disabled successfully",
        )
    except Exception as exc:
        logger.warning("Failed to disable plugin '%s': %s", name, exc)
        return StateChangeResponse(
            success=False,
            name=name,
            state=record.state,
            error=str(exc),
            message=f"Failed to disable: {exc}",
        )


@router.delete("/{name}", response_model=UninstallResponse)
def uninstall_plugin(name: str) -> dict:
    """Uninstall a plugin completely."""
    registry = _get_registry()
    record = registry.get(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")

    loader = _get_loader()
    try:
        loader.uninstall_plugin(name)
        return UninstallResponse(
            success=True,
            name=name,
            message=f"Plugin '{name}' uninstalled successfully",
        )
    except Exception as exc:
        logger.warning("Failed to uninstall plugin '%s': %s", name, exc)
        return UninstallResponse(
            success=False,
            name=name,
            error=str(exc),
            message=f"Failed to uninstall: {exc}",
        )


@router.get("/{name}/config", response_model=PluginConfigResponse)
def get_plugin_config(name: str) -> dict:
    """Get a plugin's current configuration and schema."""
    registry = _get_registry()
    record = registry.get(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")

    return PluginConfigResponse(
        name=name,
        config=record.config,
        config_schema=record.manifest.config_schema,
    )


@router.put("/{name}/config", response_model=UpdateConfigResponse)
def update_plugin_config(name: str, req: UpdateConfigRequest) -> dict:
    """Update a plugin's configuration."""
    registry = _get_registry()
    record = registry.get(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")

    try:
        success = registry.set_config(name, req.config)
        if success:
            return UpdateConfigResponse(
                success=True,
                name=name,
                config=req.config,
                message=f"Plugin '{name}' configuration updated",
            )
        return UpdateConfigResponse(
            success=False,
            name=name,
            error="Failed to update configuration",
            message="Registry returned failure",
        )
    except Exception as exc:
        logger.warning("Failed to update config for '%s': %s", name, exc)
        return UpdateConfigResponse(
            success=False,
            name=name,
            error=str(exc),
            message=f"Failed to update config: {exc}",
        )
