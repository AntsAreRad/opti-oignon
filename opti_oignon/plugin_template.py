#!/usr/bin/env python3
"""
Plugin template generator for Opti-Oignon (S102).

PluginTemplateGenerator: generate a new plugin scaffold with
manifest.yaml, entry_point.py (with hook stubs), and README.md.
"""

import logging
import re
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# TPL-1: fallback name pattern, used only if plugin_manifest cannot be
# imported. Kept identical to plugin_manifest._NAME_RE.
_FALLBACK_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")

# Valid hooks for stub generation
HOOK_STUBS = {
    "pre_prompt": {
        "signature": "def hook_pre_prompt(data: dict) -> dict:",
        "docstring": "Called before the prompt is sent to the model.",
        "body": "    # Modify data['prompt'] before inference\n    return data",
    },
    "post_prompt": {
        "signature": "def hook_post_prompt(data: dict) -> dict:",
        "docstring": "Called after the prompt is assembled but before display.",
        "body": "    # Modify data['prompt'] after assembly\n    return data",
    },
    "pre_inference": {
        "signature": "def hook_pre_inference(data: dict) -> dict:",
        "docstring": "Called before model inference starts.",
        "body": "    # Modify data['messages'] or data['options'] before inference\n    return data",
    },
    "post_inference": {
        "signature": "def hook_post_inference(data: dict) -> dict:",
        "docstring": "Called after model inference completes.",
        "body": "    # Modify data['response'] after inference\n    return data",
    },
    "tool_call": {
        "signature": "def hook_tool_call(data: dict) -> dict:",
        "docstring": "Called when a tool is invoked.",
        "body": "    # Process data['tool_name'] and data['arguments']\n    return data",
    },
    "pipeline_step": {
        "signature": "def hook_pipeline_step(data: dict) -> dict:",
        "docstring": "Called as a pipeline processing step.",
        "body": "    # Transform data as a pipeline step\n    return data",
    },
    "ui_panel": {
        "signature": "def hook_ui_panel(data: dict) -> dict:",
        "docstring": "Called to render a UI panel contribution.",
        "body": '    # Return data with "html" or "component" key\n    return data',
    },
}


class PluginTemplateGenerator:
    """Generate plugin scaffold files.

    Parameters
    ----------
    output_base_dir : Path or str or None
        Base directory where plugin scaffolds are created.
        If None, a temp directory pattern is used.
    """

    def __init__(
        self,
        output_base_dir: Path | str | None = None,
    ) -> None:
        self._output_base = Path(output_base_dir) if output_base_dir else None

    def generate(
        self,
        name: str,
        *,
        author: str = "Your Name",
        description: str = "A custom Opti-Oignon plugin.",
        version: str = "1.0.0",
        hooks: list[str] | None = None,
        permissions: list[str] | None = None,
        output_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Generate a complete plugin scaffold.

        Parameters
        ----------
        name : str
            Plugin name (lowercase, alphanumeric + hyphens/underscores).
        author : str
            Plugin author name.
        description : str
            Short plugin description.
        version : str
            Initial version string.
        hooks : list[str] or None
            Hook points this plugin will use. If None, defaults to ["post_inference"].
        permissions : list[str] or None
            Permissions to request. If None, defaults to empty.
        output_dir : Path or str or None
            Override output directory. If None, uses output_base_dir / name.

        Returns
        -------
        dict with keys: success, path, files, error
        """
        if hooks is None:
            hooks = ["post_inference"]
        if permissions is None:
            permissions = []

        # TPL-1: validate the name before it becomes a directory name and
        # is embedded into the generated files -- an unvalidated name is a
        # path-traversal (output_base / name) and template-injection vector.
        try:
            from opti_oignon.plugin_manifest import _NAME_RE as _name_re
        except Exception:
            _name_re = _FALLBACK_NAME_RE
        if not _name_re.match(name or ""):
            return {
                "success": False,
                "path": "",
                "files": [],
                "error": (
                    f"Invalid plugin name '{name}': must be lowercase "
                    "alphanumeric with hyphens/underscores, 2-64 chars, "
                    "starting with a letter."
                ),
            }

        # Determine output path
        if output_dir:
            target = Path(output_dir)
        elif self._output_base:
            target = self._output_base / name
        else:
            target = Path.cwd() / name

        try:
            target.mkdir(parents=True, exist_ok=True)

            files_created = []

            # 1. manifest.yaml
            manifest_content = self._generate_manifest(
                name=name,
                author=author,
                description=description,
                version=version,
                hooks=hooks,
                permissions=permissions,
            )
            manifest_path = target / "manifest.yaml"
            manifest_path.write_text(manifest_content, encoding="utf-8")
            files_created.append("manifest.yaml")

            # 2. entry_point.py
            entry_content = self._generate_entry_point(
                name=name,
                description=description,
                hooks=hooks,
            )
            entry_path = target / "entry_point.py"
            entry_path.write_text(entry_content, encoding="utf-8")
            files_created.append("entry_point.py")

            # 3. README.md
            readme_content = self._generate_readme(
                name=name,
                author=author,
                description=description,
                version=version,
                hooks=hooks,
                permissions=permissions,
            )
            readme_path = target / "README.md"
            readme_path.write_text(readme_content, encoding="utf-8")
            files_created.append("README.md")

            logger.info(
                "Generated plugin scaffold '%s' at %s (%d files)",
                name, target, len(files_created),
            )
            return {
                "success": True,
                "path": str(target),
                "files": files_created,
                "error": None,
            }

        except Exception as exc:
            logger.warning("Failed to generate plugin scaffold: %s", exc)
            return {
                "success": False,
                "path": str(target),
                "files": [],
                "error": str(exc),
            }

    # -----------------------------------------------------------------
    # File generators
    # -----------------------------------------------------------------

    def _generate_manifest(
        self,
        name: str,
        author: str,
        description: str,
        version: str,
        hooks: list[str],
        permissions: list[str],
    ) -> str:
        """Generate manifest.yaml content."""
        hooks_yaml = "\n".join(f"  - {h}" for h in hooks) if hooks else "  []"
        perms_yaml = "\n".join(f"  - {p}" for p in permissions) if permissions else "  []"

        # Build config_schema example if hooks warrant it
        config_lines = self._generate_config_schema_yaml(hooks)

        return textwrap.dedent(f"""\
            # Plugin manifest for {name}
            # See docs/PLUGIN_DEVELOPMENT_GUIDE.md for reference.

            name: "{name}"
            version: "{version}"
            author: "{author}"
            description: "{description}"
            entry_point: "entry_point.py"

            hooks:
            {hooks_yaml}

            permissions:
            {perms_yaml}

            dependencies: []

            min_opti_version: "1.0.0"

            config_schema:
            {config_lines}
        """)

    def _generate_entry_point(
        self,
        name: str,
        description: str,
        hooks: list[str],
    ) -> str:
        """Generate entry_point.py with hook stubs."""
        # Module header
        lines = [
            '"""',
            f"{name} -- {description}",
            "",
            "Opti-Oignon plugin entry point.",
            '"""',
            "",
            "",
        ]

        # Plugin config (read from manifest at load time)
        lines.extend([
            "# Plugin configuration (populated by the host at load time)",
            "PLUGIN_CONFIG = {}",
            "",
            "",
        ])

        # init / shutdown
        lines.extend([
            "def init():",
            f'    """Initialize the {name} plugin."""',
            "    pass",
            "",
            "",
            "def shutdown():",
            f'    """Clean up the {name} plugin."""',
            "    pass",
            "",
            "",
        ])

        # Hook stubs
        for hook in hooks:
            stub = HOOK_STUBS.get(hook)
            if stub:
                lines.append(stub["signature"])
                lines.append(f'    """{stub["docstring"]}"""')
                lines.append(stub["body"])
                lines.append("")
                lines.append("")

        # HOOKS dict mapping
        if hooks:
            lines.append("# Hook registry -- maps hook names to callables")
            lines.append("HOOKS = {")
            for hook in hooks:
                if hook in HOOK_STUBS:
                    lines.append(f'    "{hook}": hook_{hook},')
            lines.append("}")
            lines.append("")

        return "\n".join(lines)

    def _generate_readme(
        self,
        name: str,
        author: str,
        description: str,
        version: str,
        hooks: list[str],
        permissions: list[str],
    ) -> str:
        """Generate README.md content."""
        hooks_list = "\n".join(f"- `{h}`" for h in hooks) if hooks else "- None"
        perms_list = "\n".join(f"- `{p}`" for p in permissions) if permissions else "- None"

        return textwrap.dedent(f"""\
            # {name}

            {description}

            ## Info

            - **Version:** {version}
            - **Author:** {author}
            - **Min Opti-Oignon version:** 1.0.0

            ## Hooks

            {hooks_list}

            ## Permissions

            {perms_list}

            ## Installation

            Copy this directory into your Opti-Oignon plugins folder, or install
            via the Plugin Marketplace:

            ```
            Settings > Plugins > Marketplace > Install from URL
            ```

            ## Configuration

            Edit the plugin configuration in Settings > Plugins > {name} > Config.

            ## Development

            1. Edit `entry_point.py` to implement your hook logic.
            2. Update `manifest.yaml` if you add new hooks or permissions.
            3. Test your plugin locally before publishing.

            See `docs/PLUGIN_DEVELOPMENT_GUIDE.md` for the full development guide.
        """)

    def _generate_config_schema_yaml(self, hooks: list[str]) -> str:
        """Generate a sample config_schema section for the manifest."""
        lines = [
            '  enabled:',
            '    type: "boolean"',
            '    default: true',
            '    description: "Enable or disable this plugin"',
        ]
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Utility: list available hooks
    # -----------------------------------------------------------------

    @staticmethod
    def available_hooks() -> list[dict[str, str]]:
        """Return a list of available hook points with descriptions."""
        return [
            {"name": name, "description": stub["docstring"]}
            for name, stub in HOOK_STUBS.items()
        ]

    @staticmethod
    def available_permissions() -> list[str]:
        """Return a list of valid permissions a plugin can request.

        TPL-2: derived from plugin_manifest.VALID_PERMISSIONS so this list
        cannot drift from the validator (the S124 additions
        filesystem_read / filesystem_write / inference_content were missing
        from the previous hardcoded list).
        """
        try:
            from opti_oignon.plugin_manifest import VALID_PERMISSIONS
            return sorted(VALID_PERMISSIONS)
        except Exception:
            return sorted([
                "conversation_read",
                "conversation_write",
                "model_config_read",
                "model_config_write",
                "tool_register",
                "pipeline_register",
                "ui_panel_register",
                "filesystem_plugin_dir",
                "network_outbound",
                "filesystem_read",
                "filesystem_write",
                "inference_content",
            ])


# =========================================================================
# Module-level singleton
# =========================================================================

PLUGIN_TEMPLATE_AVAILABLE = True

try:
    from opti_oignon.config import DATA_DIR as _DATA_DIR

    _tmpl_base = Path(_DATA_DIR) / "plugins"
    _tmpl_base.mkdir(parents=True, exist_ok=True)
    plugin_template_generator = PluginTemplateGenerator(output_base_dir=_tmpl_base)
except Exception as _exc:
    logger.debug("PluginTemplateGenerator singleton init deferred: %s", _exc)
    plugin_template_generator = PluginTemplateGenerator()
