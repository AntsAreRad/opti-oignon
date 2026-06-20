#!/usr/bin/env python3
"""
Tests pour le registre d'outils (S44).

Couvre:
- Enregistrement d'un outil
- Recuperation par nom
- Liste des outils disponibles
- Verification de disponibilite (enabled/disabled)
- Generation du prompt d'outils pour le LLM
- Outils integres enregistres correctement
- Outil avec dependance manquante marque indisponible
- Handlers des outils integres
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from opti_oignon.tool_registry import (
    MAX_FILE_READ_SIZE,
    ToolDefinition,
    ToolParam,
    ToolRegistry,
    _handle_list_files,
    _handle_read_file,
    _handle_write_file,
    tool_registry,
)

# =============================================================================
# ENREGISTREMENT ET RECUPERATION
# =============================================================================

class TestToolRegistration:
    """Tests d'enregistrement et recuperation d'outils."""

    def test_register_tool(self):
        """Enregistrer un outil simple."""
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            handler=lambda: "ok",
        )
        reg.register(tool)
        assert reg.get("test_tool") is not None
        assert reg.get("test_tool").name == "test_tool"

    def test_get_nonexistent(self):
        """Recuperer un outil inexistant retourne None."""
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_register_with_params(self):
        """Enregistrer un outil avec des parametres."""
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="parameterized",
            description="Has params",
            parameters={
                "query": ToolParam(
                    name="query", type="string",
                    description="Search query", required=True,
                ),
                "count": ToolParam(
                    name="count", type="int",
                    description="Number of results",
                    required=False, default=10,
                ),
            },
            handler=lambda query, count=10: f"{query}:{count}",
        )
        reg.register(tool)
        retrieved = reg.get("parameterized")
        assert len(retrieved.parameters) == 2
        assert retrieved.parameters["query"].required is True
        assert retrieved.parameters["count"].default == 10

    def test_overwrite_tool(self):
        """Re-enregistrer un outil ecrase l'ancien."""
        reg = ToolRegistry()
        tool_v1 = ToolDefinition(name="myutil", description="v1")
        tool_v2 = ToolDefinition(name="myutil", description="v2")
        reg.register(tool_v1)
        reg.register(tool_v2)
        assert reg.get("myutil").description == "v2"


# =============================================================================
# DISPONIBILITE
# =============================================================================

class TestToolAvailability:
    """Tests de verification de disponibilite."""

    def test_list_available_filters_disabled(self):
        """list_available exclut les outils desactives."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="active", description="ok", enabled=True))
        reg.register(ToolDefinition(name="inactive", description="off", enabled=False))
        available = reg.list_available()
        names = [t.name for t in available]
        assert "active" in names
        assert "inactive" not in names

    def test_is_available_true(self):
        """is_available retourne True pour un outil actif."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="tool1", description="ok", enabled=True))
        assert reg.is_available("tool1") is True

    def test_is_available_false_disabled(self):
        """is_available retourne False pour un outil desactive."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="tool1", description="ok", enabled=False))
        assert reg.is_available("tool1") is False

    def test_is_available_false_not_found(self):
        """is_available retourne False pour un outil inexistant."""
        reg = ToolRegistry()
        assert reg.is_available("ghost") is False

    def test_list_all_includes_disabled(self):
        """list_all inclut les outils desactives."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="a", description="", enabled=True))
        reg.register(ToolDefinition(name="b", description="", enabled=False))
        all_tools = reg.list_all()
        assert len(all_tools) == 2

    def test_missing_requirement_disables_tool(self):
        """Un outil avec dependance manquante est desactive."""
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="needs_magic",
            description="Requires unavailable module",
            requires=["nonexistent_module_xyz_123"],
            enabled=True,
        )
        reg.register(tool)
        assert tool.enabled is False
        assert reg.is_available("needs_magic") is False


# =============================================================================
# PROMPT GENERATION
# =============================================================================

class TestToolsPrompt:
    """Tests de generation du prompt d'outils."""

    def test_empty_prompt_no_tools(self):
        """Aucun outil -> prompt vide."""
        reg = ToolRegistry()
        assert reg.get_tools_prompt() == ""

    def test_prompt_contains_tool_name(self):
        """Le prompt contient le nom de l'outil."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="my_tool", description="Does something",
            enabled=True,
        ))
        prompt = reg.get_tools_prompt()
        assert "my_tool" in prompt
        assert "Does something" in prompt

    def test_prompt_contains_params(self):
        """Le prompt contient les descriptions de parametres."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="searcher",
            description="Search tool",
            parameters={
                "query": ToolParam(
                    name="query", type="string",
                    description="The search query", required=True,
                ),
            },
            enabled=True,
        ))
        prompt = reg.get_tools_prompt()
        assert "query" in prompt
        assert "string" in prompt
        assert "required" in prompt

    def test_prompt_excludes_disabled(self):
        """Le prompt n'inclut pas les outils desactives."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="visible", description="ok", enabled=True))
        reg.register(ToolDefinition(name="hidden", description="nope", enabled=False))
        prompt = reg.get_tools_prompt()
        assert "visible" in prompt
        assert "hidden" not in prompt

    def test_prompt_includes_json_instructions(self):
        """Le prompt contient les instructions de format JSON."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="t", description="d", enabled=True))
        prompt = reg.get_tools_prompt()
        assert "tool_name" in prompt
        assert "arguments" in prompt
        assert "none" in prompt


# =============================================================================
# OUTILS INTEGRES (SINGLETON)
# =============================================================================

class TestBuiltinTools:
    """Tests des outils integres enregistres dans le singleton."""

    def test_builtin_tools_registered(self):
        """Les 5 outils integres sont enregistres."""
        names = [t.name for t in tool_registry.list_all()]
        assert "web_search" in names
        assert "execute_code" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "list_files" in names

    def test_file_tools_always_available(self):
        """Les outils fichier sont toujours disponibles (filesystem)."""
        assert tool_registry.is_available("read_file")
        assert tool_registry.is_available("write_file")
        assert tool_registry.is_available("list_files")

    def test_web_search_has_params(self):
        """web_search a les bons parametres."""
        tool = tool_registry.get("web_search")
        assert tool is not None
        assert "query" in tool.parameters
        assert tool.parameters["query"].required is True
        assert "max_results" in tool.parameters
        assert tool.parameters["max_results"].default == 5

    def test_execute_code_has_params(self):
        """execute_code a les bons parametres."""
        tool = tool_registry.get("execute_code")
        assert tool is not None
        assert "code" in tool.parameters
        assert "language" in tool.parameters
        assert "timeout" in tool.parameters


# =============================================================================
# HANDLERS DES OUTILS FICHIER
# =============================================================================

class TestFileHandlers:
    """Tests des handlers de fichiers (unitaires)."""

    def test_read_file_success(self):
        """Lecture d'un fichier existant."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, world!")
            path = f.name
        try:
            result = _handle_read_file(path)
            assert result == "Hello, world!"
        finally:
            os.unlink(path)

    def test_read_file_not_found(self):
        """Lecture d'un fichier inexistant."""
        result = _handle_read_file("/nonexistent/path/file.txt")
        assert "not found" in result.lower() or "error" in result.lower()

    def test_write_file_success(self):
        """Ecriture dans un fichier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            result = _handle_write_file(path, "Test content")
            assert "written" in result.lower() or "File written" in result
            with open(path) as f:
                assert f.read() == "Test content"

    def test_list_files_success(self):
        """Listing d'un repertoire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Creer quelques fichiers
            open(os.path.join(tmpdir, "a.txt"), "w").close()
            open(os.path.join(tmpdir, "b.py"), "w").close()
            os.makedirs(os.path.join(tmpdir, "subdir"))
            result = _handle_list_files(tmpdir)
            assert "a.txt" in result
            assert "b.py" in result
            assert "subdir" in result

    def test_list_files_not_found(self):
        """Listing d'un repertoire inexistant."""
        result = _handle_list_files("/nonexistent/dir")
        assert "not found" in result.lower() or "error" in result.lower()

    def test_read_file_too_large(self):
        """Lecture d'un fichier trop volumineux."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"x" * (MAX_FILE_READ_SIZE + 100))
            path = f.name
        try:
            result = _handle_read_file(path)
            assert "too large" in result.lower()
        finally:
            os.unlink(path)
