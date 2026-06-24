#!/usr/bin/env python3
"""
TOOL REGISTRY - OPTI-OIGNON v1.5.0 (S44)
==========================================

Registre des outils appelables par le LLM.

Definit les outils disponibles (web_search, execute_code, read_file,
write_file, list_files) avec leurs schemas de parametres et handlers.
Each tool checks the availability of its dependencies at
de l'enregistrement.

Author: Leon
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# -- Limites de securite pour les operations fichier --
MAX_FILE_READ_SIZE = 1024 * 1024  # 1 Mo
ALLOWED_FILE_DIR = os.path.expanduser("~/.opti-oignon/workspace")


@dataclass
class ToolParam:
    """Definition of a parameter d'outil."""
    name: str
    type: str  # "string", "int", "float", "bool", "list"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of a outil appelable par le LLM."""
    name: str
    description: str
    parameters: dict[str, ToolParam] = field(default_factory=dict)
    handler: Callable | None = None
    requires: list[str] = field(default_factory=list)
    enabled: bool = True


class ToolRegistry:
    """Registre des outils disponibles.

    Gere l'enregistrement, la resolution et la generation
    de prompts pour les outils appelables par le LLM.
    """

    # Tools that are UNSAFE outside a sandbox. When sandbox mode is
    # active, these are disabled and replaced by their sandboxed
    # equivalents. This prevents the LLM from bypassing the sandbox
    # by calling an unsandboxed tool instead.
    UNSAFE_TOOLS = frozenset({
        "execute_code",
        "read_file",
        "write_file",
        "list_files",
    })

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._sandbox_mode = False
        self._disabled_by_sandbox: set[str] = set()
        # S117: Quick sandbox mode — transparent handler replacement
        self._quick_sandbox_mode = False
        self._original_handlers: dict[str, Callable | None] = {}
        self._quick_sandbox_session = None

    def register(self, tool: ToolDefinition) -> None:
        """Enregistre un outil dans le registre.

        Check the availability of dependencies and deactivate
        l'outil si un module requis est manquant.
        """
        # Check les dependances
        if tool.requires:
            for req in tool.requires:
                if not self._check_requirement(req):
                    logger.warning(
                        f"Outil '{tool.name}' desactive: "
                        f"dependance '{req}' non disponible"
                    )
                    tool.enabled = False
                    break

        self._tools[tool.name] = tool
        status = "actif" if tool.enabled else "inactif"
        logger.debug(f"Outil enregistre: {tool.name} ({status})")

    def get(self, name: str) -> ToolDefinition | None:
        """Retrieve a tool by its name."""
        return self._tools.get(name)

    def list_available(self) -> list[ToolDefinition]:
        """Liste les outils actifs et disponibles."""
        return [t for t in self._tools.values() if t.enabled]

    def list_all(self) -> list[ToolDefinition]:
        """Liste tous les outils enregistres (actifs ou non)."""
        return list(self._tools.values())

    def is_available(self, name: str) -> bool:
        """Check if a tool is registered and active."""
        tool = self._tools.get(name)
        return tool is not None and tool.enabled

    @property
    def sandbox_mode(self) -> bool:
        """Whether sandbox mode is active (unsafe tools disabled)."""
        return self._sandbox_mode

    def set_sandbox_mode(self, enabled: bool) -> list[str]:
        """Enable or disable sandbox mode.

        When enabled: disables all UNSAFE_TOOLS (execute_code, read_file,
        write_file, list_files) to prevent the LLM from bypassing the
        sandbox by calling unsandboxed tools directly.

        When disabled: re-enables any tools that were disabled by
        sandbox mode (restores previous state).

        Args:
            enabled: True to enter sandbox mode, False to exit.

        Returns:
            List of tool names that were disabled (on enable) or
            re-enabled (on disable).
        """
        affected = []

        if enabled and not self._sandbox_mode:
            # Entering sandbox mode: disable unsafe tools
            for name in self.UNSAFE_TOOLS:
                tool = self._tools.get(name)
                if tool is not None and tool.enabled:
                    tool.enabled = False
                    self._disabled_by_sandbox.add(name)
                    affected.append(name)
                    logger.info(
                        "Sandbox mode: disabled unsafe tool '%s'", name
                    )
            self._sandbox_mode = True

        elif not enabled and self._sandbox_mode:
            # Exiting sandbox mode: re-enable previously disabled tools
            for name in list(self._disabled_by_sandbox):
                tool = self._tools.get(name)
                if tool is not None:
                    tool.enabled = True
                    affected.append(name)
                    logger.info(
                        "Sandbox mode off: re-enabled tool '%s'", name
                    )
            self._disabled_by_sandbox.clear()
            self._sandbox_mode = False

        return affected

    # -- S117: Quick sandbox mode (transparent handler replacement) --

    @property
    def quick_sandbox_mode(self) -> bool:
        """Whether quick sandbox mode is active (handlers replaced)."""
        return self._quick_sandbox_mode

    @property
    def quick_sandbox_session(self):
        """The active QuickSandboxSession, or None."""
        return self._quick_sandbox_session

    def set_quick_sandbox_mode(
        self, enabled: bool, session=None,
    ) -> list[str]:
        """Enable or disable quick sandbox mode.

        Unlike sandbox_mode (which DISABLES unsafe tools entirely),
        quick_sandbox_mode REPLACES the handlers of unsafe tools with
        sandboxed equivalents. The LLM sees the same tool names and
        parameters, but execution is redirected to the sandbox.

        Args:
            enabled: True to activate quick sandbox routing.
            session: A QuickSandboxSession instance (required when
                enabling, ignored when disabling).

        Returns:
            List of tool names whose handlers were replaced or restored.
        """
        affected = []

        if enabled and not self._quick_sandbox_mode:
            if session is None:
                raise ValueError(
                    "QuickSandboxSession required when enabling "
                    "quick_sandbox_mode"
                )
            self._quick_sandbox_session = session

            # Replace handlers for each unsafe tool with sandbox wrappers
            handler_map = {
                "execute_code": lambda code, language="python", timeout=30: (
                    session.handle_execute_code(code, language, timeout)
                ),
                "write_file": lambda path, content: (
                    session.handle_write_file(path, content)
                ),
                "read_file": lambda path: (
                    session.handle_read_file(path)
                ),
                "list_files": lambda path=".": (
                    session.handle_list_files(path)
                ),
            }

            for tool_name, new_handler in handler_map.items():
                tool = self._tools.get(tool_name)
                if tool is not None and tool.enabled:
                    self._original_handlers[tool_name] = tool.handler
                    tool.handler = new_handler
                    affected.append(tool_name)
                    logger.info(
                        "Quick sandbox: replaced handler for '%s'",
                        tool_name,
                    )

            self._quick_sandbox_mode = True

        elif not enabled and self._quick_sandbox_mode:
            # Restore original handlers
            for tool_name, orig_handler in self._original_handlers.items():
                tool = self._tools.get(tool_name)
                if tool is not None:
                    tool.handler = orig_handler
                    affected.append(tool_name)
                    logger.info(
                        "Quick sandbox off: restored handler for '%s'",
                        tool_name,
                    )
            self._original_handlers.clear()
            self._quick_sandbox_session = None
            self._quick_sandbox_mode = False

        return affected

    def get_tools_prompt(self) -> str:
        """Generate a prompt describing the available tools for the LLM.

        Formate les descriptions et parametres de each outil actif
        dans un format lisible par le LLM pour la prise de decision.
        """
        available = self.list_available()
        if not available:
            return ""

        lines = [
            "You have access to the following tools. When the user asks "
            "you to create, read, edit, modify, list, or run files or "
            "code, call the appropriate tool to perform the action "
            "directly instead of only printing or describing it. This "
            "applies whatever language the user writes in.\n"
        ]
        for tool in available:
            lines.append(f"## {tool.name}")
            lines.append(f"{tool.description}")
            if tool.parameters:
                lines.append("Parameters:")
                for param in tool.parameters.values():
                    req = "required" if param.required else "optional"
                    default_str = ""
                    if param.default is not None:
                        default_str = f", default={param.default}"
                    lines.append(
                        f"  - {param.name} ({param.type}, {req}{default_str})"
                        f": {param.description}"
                    )
            lines.append("")

        # When a quick sandbox session is active, list the files already in
        # the workspace so the model edits them in place (read_file +
        # write_file) instead of regenerating their content as text.
        if getattr(self, "_quick_sandbox_mode", False):
            _qs_session = getattr(self, "_quick_sandbox_session", None)
            _existing: list[str] = []
            if _qs_session is not None:
                try:
                    _existing = list(_qs_session.files_created)
                except Exception:
                    _existing = []
            if _existing:
                lines.append(
                    "Files already in your sandbox workspace: "
                    + ", ".join(_existing)
                    + "\nTo change an existing file, call read_file to get "
                    "its current content, then call write_file with the "
                    "same path and the updated content. Do not just print "
                    "the new version in your reply.\n"
                )

        lines.append(
            "To use a tool, respond with a JSON object containing:\n"
            '  - "tool_name": the tool to call\n'
            '  - "arguments": a dict of parameter values\n'
            '  - "reasoning": brief explanation of why you chose this tool\n'
            "\n"
            'If no tool is needed, set tool_name to "none".'
        )
        return "\n".join(lines)

    def _check_requirement(self, requirement: str) -> bool:
        """Check if a module or feature is available."""
        # Check les flags de disponibilite connus
        availability_map = {
            "web_search": self._check_web_search,
            "code_executor": self._check_code_executor,
            "filesystem": self._check_filesystem,
            "sandbox": self._check_sandbox,
        }
        checker = availability_map.get(requirement)
        if checker:
            return checker()

        # Tenter un import generique
        try:
            __import__(requirement)
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_web_search() -> bool:
        """Check if the web search engine is available."""
        try:
            from opti_oignon.web_search import web_search_engine
            return web_search_engine is not None
        except ImportError:
            return False

    @staticmethod
    def _check_code_executor() -> bool:
        """Check if the code executor is available."""
        try:
            from opti_oignon.code_executor import code_executor
            return code_executor is not None
        except ImportError:
            return False

    @staticmethod
    def _check_filesystem() -> bool:
        """Check if file operations are available."""
        return True  # Toujours disponible sur un systeme standard

    @staticmethod
    def _check_sandbox() -> bool:
        """Check if the sandbox manager is available."""
        try:
            from opti_oignon.sandbox_manager import (
                SANDBOX_AVAILABLE,
                sandbox_manager,
            )
            return SANDBOX_AVAILABLE and sandbox_manager is not None
        except ImportError:
            return False


# =============================================================================
# HANDLERS DES OUTILS INTEGRES
# =============================================================================

def _handle_web_search(query: str, max_results: int = 5) -> str:
    """Handler pour l'outil web_search."""
    try:
        from opti_oignon.web_search import web_search_engine
        results = web_search_engine.search(query, max_results=max_results)
        if not results:
            return f"No results found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r.title}\n"
                f"   URL: {r.url}\n"
                f"   {r.snippet}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search error: {e}"


def _handle_execute_code(
    code: str, language: str = "python", timeout: int = 30
) -> str:
    """Handler pour l'outil execute_code."""
    try:
        from opti_oignon.code_executor import code_executor
        result = code_executor.execute(code, language=language, timeout=timeout)
        parts = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")
        if result.error_message:
            parts.append(f"ERROR: {result.error_message}")
        if not parts:
            status = "Success" if result.success else "Failed"
            parts.append(f"Execution {status} (return code: {result.return_code})")
        return "\n".join(parts)
    except Exception as e:
        return f"Code execution error: {e}"


def _handle_read_file(path: str) -> str:
    """Handler pour l'outil read_file."""
    try:
        # Securite: resoudre le chemin et check la taille
        resolved = os.path.abspath(path)
        if not os.path.isfile(resolved):
            return f"File not found: {path}"

        size = os.path.getsize(resolved)
        if size > MAX_FILE_READ_SIZE:
            return (
                f"File too large: {size} bytes "
                f"(max {MAX_FILE_READ_SIZE} bytes)"
            )

        with open(resolved, encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Read file error: {e}"


def _handle_write_file(path: str, content: str) -> str:
    """Handler pour l'outil write_file."""
    try:
        # Creer le repertoire de travail si required
        os.makedirs(ALLOWED_FILE_DIR, exist_ok=True)

        # Securite: ecrire uniquement dans le repertoire autorise
        if not os.path.isabs(path):
            resolved = os.path.join(ALLOWED_FILE_DIR, path)
        else:
            resolved = os.path.abspath(path)

        # Creer les sous-repertoires si required
        os.makedirs(os.path.dirname(resolved), exist_ok=True)

        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written: {resolved} ({len(content)} bytes)"
    except Exception as e:
        return f"Write file error: {e}"


def _handle_list_files(path: str = ".") -> str:
    """Handler pour l'outil list_files."""
    try:
        resolved = os.path.abspath(path)
        if not os.path.isdir(resolved):
            return f"Directory not found: {path}"

        entries = sorted(os.listdir(resolved))
        if not entries:
            return f"Empty directory: {path}"

        lines = []
        for entry in entries[:100]:  # Limiter a 100 entrees
            full_path = os.path.join(resolved, entry)
            if os.path.isdir(full_path):
                lines.append(f"  [DIR]  {entry}/")
            else:
                size = os.path.getsize(full_path)
                lines.append(f"  [FILE] {entry} ({size} bytes)")
        return "\n".join(lines)
    except Exception as e:
        return f"List files error: {e}"


# =============================================================================
# ENREGISTREMENT DES OUTILS INTEGRES
# =============================================================================

def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Enregistre les 5 outils integres dans le registre."""

    # 1. web_search
    registry.register(ToolDefinition(
        name="web_search",
        description="Search the web for current information using DuckDuckGo.",
        parameters={
            "query": ToolParam(
                name="query",
                type="string",
                description="Search query",
                required=True,
            ),
            "max_results": ToolParam(
                name="max_results",
                type="int",
                description="Maximum number of results to return",
                required=False,
                default=5,
            ),
        },
        handler=_handle_web_search,
        requires=["web_search"],
    ))

    # 2. execute_code
    registry.register(ToolDefinition(
        name="execute_code",
        description=(
            "Execute Python or R code in a sandboxed environment "
            "and return the output."
        ),
        parameters={
            "code": ToolParam(
                name="code",
                type="string",
                description="Code to execute",
                required=True,
            ),
            "language": ToolParam(
                name="language",
                type="string",
                description="Programming language (python or r)",
                required=False,
                default="python",
            ),
            "timeout": ToolParam(
                name="timeout",
                type="int",
                description="Maximum execution time in seconds",
                required=False,
                default=30,
            ),
        },
        handler=_handle_execute_code,
        requires=["code_executor"],
    ))

    # 3. read_file
    registry.register(ToolDefinition(
        name="read_file",
        description="Read the contents of a file from disk.",
        parameters={
            "path": ToolParam(
                name="path",
                type="string",
                description="Path to the file to read",
                required=True,
            ),
        },
        handler=_handle_read_file,
        requires=["filesystem"],
    ))

    # 4. write_file
    registry.register(ToolDefinition(
        name="write_file",
        description="Write content to a file on disk.",
        parameters={
            "path": ToolParam(
                name="path",
                type="string",
                description="Path to the file to write",
                required=True,
            ),
            "content": ToolParam(
                name="content",
                type="string",
                description="Content to write to the file",
                required=True,
            ),
        },
        handler=_handle_write_file,
        requires=["filesystem"],
    ))

    # 5. list_files
    registry.register(ToolDefinition(
        name="list_files",
        description="List files and directories at a given path.",
        parameters={
            "path": ToolParam(
                name="path",
                type="string",
                description="Directory path to list",
                required=False,
                default=".",
            ),
        },
        handler=_handle_list_files,
        requires=["filesystem"],
    ))


def _register_sandbox_tools(registry: ToolRegistry) -> None:
    """Register sandboxed file tools (S73) if available."""
    try:
        from opti_oignon.file_tools import (
            FILE_TOOLS_AVAILABLE,
            get_all_sandbox_tool_definitions,
        )
        if not FILE_TOOLS_AVAILABLE:
            return
        for tool_def in get_all_sandbox_tool_definitions():
            registry.register(tool_def)
        logger.debug("Sandbox tools registered: 4 tools")
    except ImportError:
        logger.debug("Sandbox tools not available (import failed)")
    except Exception as exc:
        logger.warning("Failed to register sandbox tools: %s", exc)


# =============================================================================
# SINGLETON
# =============================================================================

tool_registry = ToolRegistry()
_register_builtin_tools(tool_registry)
_register_sandbox_tools(tool_registry)
