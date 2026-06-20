"""
Tests for S97 -- Conversation Branching & Exploration.

Validates:
- Part 1: ConversationBranchManager (fork, rename, delete, messages, stats)
- Part 2: Branch tree structure and comparison
- Part 3: Merge functionality
- Part 4: API routes (endpoints, schemas, error handling)
- Part 5: Frontend (types, API client, BranchExplorer, chat page integration)
- Part 6: Config persistence (branches.yaml)
- Part 7: Integration wiring (deps.py, app.py, version bump)
- Zero regressions

Target: ~47 tests
"""

import importlib.util
import json
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
API_DIR = os.path.join(BACKEND_DIR, "api")
CONFIG_DIR = os.path.join(BACKEND_DIR, "config")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components")
CHAT_DIR = os.path.join(COMPONENTS_DIR, "chat")
ROUTES_DIR = os.path.join(FRONTEND_SRC, "routes")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    """Read file contents as string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Stub dependencies before loading module
# ---------------------------------------------------------------------------

# Stub opti_oignon package and config
_config_mock = MagicMock()
_config_mock.DATA_DIR = Path(tempfile.mkdtemp())
sys.modules.setdefault("opti_oignon", MagicMock())
sys.modules["opti_oignon.config"] = _config_mock
# Context manager mock: estimate_tokens must return int for SQLite
_ctx_mock = MagicMock()
_ctx_mock.estimate_tokens = MagicMock(side_effect=lambda text, model=None: int(len(text) / 4))
sys.modules["opti_oignon.context_manager"] = _ctx_mock

# Load conversation_branches module in isolation
branches_mod = _load_module(
    "opti_oignon.conversation_branches",
    os.path.join(BACKEND_DIR, "conversation_branches.py"),
)
ConversationBranchManager = branches_mod.ConversationBranchManager
Branch = branches_mod.Branch
BranchMessage = branches_mod.BranchMessage
BranchTreeNode = branches_mod.BranchTreeNode
BranchComparison = branches_mod.BranchComparison
_DEFAULT_CONFIG = branches_mod._DEFAULT_CONFIG


def _make_manager(tmp_dir=None):
    """Create a fresh manager with an isolated temp database."""
    td = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp())
    return ConversationBranchManager(
        db_path=td / "test_branches.db",
        config=_DEFAULT_CONFIG,
    )


# ============================================================================
# Part 1: ConversationBranchManager — Fork, Rename, Delete, Messages, Stats
# ============================================================================

class TestBranchFork(unittest.TestCase):
    """Branch fork creation tests."""

    def setUp(self):
        self.mgr = _make_manager()

    def test_fork_creates_branch(self):
        b = self.mgr.fork("conv-1", fork_message_id=5, name="Alt path")
        self.assertIsNotNone(b)
        self.assertEqual(b.name, "Alt path")
        self.assertEqual(b.fork_message_id, 5)
        self.assertEqual(b.conversation_id, "conv-1")
        self.assertIsNotNone(b.branch_id)

    def test_fork_auto_names(self):
        b1 = self.mgr.fork("conv-1", fork_message_id=1)
        b2 = self.mgr.fork("conv-1", fork_message_id=1)
        self.assertEqual(b1.name, "Branch 1")
        self.assertEqual(b2.name, "Branch 2")

    def test_fork_auto_colors(self):
        b1 = self.mgr.fork("conv-1", fork_message_id=1)
        b2 = self.mgr.fork("conv-1", fork_message_id=1)
        # First two colors from palette
        palette = _DEFAULT_CONFIG["branches"]["color_palette"]
        self.assertEqual(b1.color, palette[0])
        self.assertEqual(b2.color, palette[1])

    def test_fork_custom_color(self):
        b = self.mgr.fork("conv-1", fork_message_id=1, color="#AABBCC")
        self.assertEqual(b.color, "#AABBCC")

    def test_fork_parent_branch(self):
        b1 = self.mgr.fork("conv-1", fork_message_id=3)
        b2 = self.mgr.fork("conv-1", fork_message_id=5, parent_branch_id=b1.branch_id)
        self.assertEqual(b2.parent_branch_id, b1.branch_id)

    def test_fork_limit_enforced(self):
        """Branch limit prevents creation beyond max."""
        mgr = ConversationBranchManager(
            db_path=Path(tempfile.mkdtemp()) / "lim.db",
            config={
                **_DEFAULT_CONFIG,
                "branches": {**_DEFAULT_CONFIG["branches"], "max_per_conversation": 2},
            },
        )
        b1 = mgr.fork("conv-1", 1)
        b2 = mgr.fork("conv-1", 1)
        b3 = mgr.fork("conv-1", 1)
        self.assertIsNotNone(b1)
        self.assertIsNotNone(b2)
        self.assertIsNone(b3)

    def test_fork_returns_uuid(self):
        b = self.mgr.fork("conv-1", fork_message_id=1)
        # UUID format check
        self.assertEqual(len(b.branch_id), 36)
        self.assertEqual(b.branch_id.count("-"), 4)


class TestBranchCRUD(unittest.TestCase):
    """Branch list, get, update, delete tests."""

    def setUp(self):
        self.mgr = _make_manager()
        self.b = self.mgr.fork("conv-1", fork_message_id=5, name="B1")

    def test_list_branches(self):
        self.mgr.fork("conv-1", fork_message_id=5, name="B2")
        branches = self.mgr.list_branches("conv-1")
        self.assertEqual(len(branches), 2)
        names = [b.name for b in branches]
        self.assertIn("B1", names)
        self.assertIn("B2", names)

    def test_list_branches_empty(self):
        branches = self.mgr.list_branches("nonexistent")
        self.assertEqual(branches, [])

    def test_get_branch(self):
        fetched = self.mgr.get_branch(self.b.branch_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "B1")

    def test_get_branch_not_found(self):
        self.assertIsNone(self.mgr.get_branch("no-such-id"))

    def test_update_name(self):
        updated = self.mgr.update_branch(self.b.branch_id, name="Renamed")
        self.assertEqual(updated.name, "Renamed")
        # Verify persistence
        fetched = self.mgr.get_branch(self.b.branch_id)
        self.assertEqual(fetched.name, "Renamed")

    def test_update_color(self):
        updated = self.mgr.update_branch(self.b.branch_id, color="#112233")
        self.assertEqual(updated.color, "#112233")

    def test_update_metadata_merge(self):
        self.mgr.update_branch(self.b.branch_id, metadata={"key1": "val1"})
        self.mgr.update_branch(self.b.branch_id, metadata={"key2": "val2"})
        fetched = self.mgr.get_branch(self.b.branch_id)
        self.assertEqual(fetched.metadata.get("key1"), "val1")
        self.assertEqual(fetched.metadata.get("key2"), "val2")

    def test_update_not_found(self):
        self.assertIsNone(self.mgr.update_branch("bad-id", name="X"))

    def test_delete_branch(self):
        ok = self.mgr.delete_branch(self.b.branch_id)
        self.assertTrue(ok)
        self.assertIsNone(self.mgr.get_branch(self.b.branch_id))

    def test_delete_branch_not_found(self):
        self.assertFalse(self.mgr.delete_branch("no-such-id"))

    def test_delete_reparents_children(self):
        child = self.mgr.fork("conv-1", 5, parent_branch_id=self.b.branch_id, name="Child")
        self.mgr.delete_branch(self.b.branch_id)
        # Child should now have parent_branch_id = None (B1's parent)
        fetched = self.mgr.get_branch(child.branch_id)
        self.assertIsNotNone(fetched)
        self.assertIsNone(fetched.parent_branch_id)

    def test_delete_all_branches(self):
        self.mgr.fork("conv-1", 5, name="B2")
        count = self.mgr.delete_all_branches("conv-1")
        self.assertEqual(count, 2)
        self.assertEqual(self.mgr.list_branches("conv-1"), [])


class TestBranchMessages(unittest.TestCase):
    """Branch message add/get/stats tests."""

    def setUp(self):
        self.mgr = _make_manager()
        self.b = self.mgr.fork("conv-1", fork_message_id=3, name="Msg test")

    def test_add_message(self):
        msg = self.mgr.add_branch_message(
            self.b.branch_id, "conv-1", "user", "Hello branch"
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello branch")
        self.assertEqual(msg.branch_id, self.b.branch_id)

    def test_add_message_assistant_with_model(self):
        msg = self.mgr.add_branch_message(
            self.b.branch_id, "conv-1", "assistant", "Response",
            model="qwen3:32b"
        )
        self.assertEqual(msg.model, "qwen3:32b")

    def test_add_message_bad_branch(self):
        msg = self.mgr.add_branch_message(
            "nonexistent", "conv-1", "user", "X"
        )
        self.assertIsNone(msg)

    def test_get_branch_only_messages(self):
        self.mgr.add_branch_message(self.b.branch_id, "conv-1", "user", "Q1")
        self.mgr.add_branch_message(self.b.branch_id, "conv-1", "assistant", "A1")
        msgs = self.mgr.get_branch_only_messages(self.b.branch_id)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].content, "Q1")
        self.assertEqual(msgs[1].content, "A1")

    def test_get_stats(self):
        self.mgr.add_branch_message(
            self.b.branch_id, "conv-1", "user", "Hello",
        )
        self.mgr.add_branch_message(
            self.b.branch_id, "conv-1", "assistant", "Hi!",
            model="llama3:70b"
        )
        stats = self.mgr.get_branch_stats(self.b.branch_id)
        self.assertEqual(stats["message_count"], 2)
        self.assertEqual(stats["last_model"], "llama3:70b")
        self.assertIsNotNone(stats["last_activity"])
        self.assertGreater(stats["total_tokens"], 0)

    def test_branch_message_to_dict(self):
        msg = self.mgr.add_branch_message(
            self.b.branch_id, "conv-1", "user", "Test"
        )
        d = msg.to_dict()
        self.assertIn("id", d)
        self.assertIn("branch_id", d)
        self.assertIn("content", d)

    def test_branch_message_to_ollama_format(self):
        msg = self.mgr.add_branch_message(
            self.b.branch_id, "conv-1", "user", "Test"
        )
        fmt = msg.to_ollama_format()
        self.assertEqual(fmt, {"role": "user", "content": "Test"})

    def test_get_branch_messages_full_with_provided_main(self):
        """Full messages = shared (provided) + branch-specific."""
        main_msgs = [
            {"id": 1, "role": "user", "content": "M1"},
            {"id": 2, "role": "assistant", "content": "M2"},
            {"id": 3, "role": "user", "content": "M3"},
            {"id": 4, "role": "assistant", "content": "M4"},
        ]
        self.mgr.add_branch_message(self.b.branch_id, "conv-1", "user", "BQ1")
        full = self.mgr.get_branch_messages_full(
            "conv-1", self.b.branch_id, main_messages=main_msgs
        )
        # fork at msg 3 -> shared = msgs 1,2,3 + branch msg BQ1
        self.assertEqual(len(full), 4)  # 3 shared + 1 branch
        self.assertEqual(full[0]["content"], "M1")
        self.assertEqual(full[2]["content"], "M3")
        self.assertEqual(full[3]["content"], "BQ1")


# ============================================================================
# Part 2: Branch tree and comparison
# ============================================================================

class TestBranchTree(unittest.TestCase):
    """Branch tree structure tests."""

    def setUp(self):
        self.mgr = _make_manager()

    def test_empty_tree(self):
        tree = self.mgr.get_branch_tree("conv-1")
        self.assertIsNone(tree.branch_id)  # root = main
        self.assertEqual(tree.name, "Main")
        self.assertEqual(tree.children, [])

    def test_tree_with_branches(self):
        self.mgr.fork("conv-1", 3, name="B1")
        self.mgr.fork("conv-1", 5, name="B2")
        tree = self.mgr.get_branch_tree("conv-1")
        self.assertEqual(len(tree.children), 2)
        names = [c.name for c in tree.children]
        self.assertIn("B1", names)
        self.assertIn("B2", names)

    def test_tree_nested(self):
        b1 = self.mgr.fork("conv-1", 3, name="B1")
        self.mgr.fork("conv-1", 5, name="B1-child", parent_branch_id=b1.branch_id)
        tree = self.mgr.get_branch_tree("conv-1")
        self.assertEqual(len(tree.children), 1)
        self.assertEqual(tree.children[0].name, "B1")
        self.assertEqual(len(tree.children[0].children), 1)
        self.assertEqual(tree.children[0].children[0].name, "B1-child")

    def test_tree_node_to_dict(self):
        self.mgr.fork("conv-1", 3, name="B1")
        tree = self.mgr.get_branch_tree("conv-1")
        d = tree.to_dict()
        self.assertIn("branch_id", d)
        self.assertIn("children", d)
        self.assertIsInstance(d["children"], list)
        self.assertEqual(len(d["children"]), 1)


class TestBranchComparison(unittest.TestCase):
    """Branch comparison tests."""

    def setUp(self):
        self.mgr = _make_manager()
        self.b1 = self.mgr.fork("conv-1", 3, name="B1")
        self.b2 = self.mgr.fork("conv-1", 3, name="B2")
        self.mgr.add_branch_message(self.b1.branch_id, "conv-1", "user", "B1 Q")
        self.mgr.add_branch_message(self.b2.branch_id, "conv-1", "user", "B2 Q")

    def test_compare_two_branches(self):
        comp = self.mgr.compare_branches("conv-1", self.b1.branch_id, self.b2.branch_id)
        self.assertIsNotNone(comp)
        self.assertEqual(comp.branch_a_name, "B1")
        self.assertEqual(comp.branch_b_name, "B2")
        self.assertEqual(len(comp.branch_a_messages), 1)
        self.assertEqual(len(comp.branch_b_messages), 1)

    def test_compare_both_none_returns_none(self):
        comp = self.mgr.compare_branches("conv-1", None, None)
        self.assertIsNone(comp)

    def test_compare_to_dict(self):
        comp = self.mgr.compare_branches("conv-1", self.b1.branch_id, self.b2.branch_id)
        d = comp.to_dict()
        self.assertIn("shared_messages", d)
        self.assertIn("branch_a_messages", d)
        self.assertIn("fork_message_id", d)


# ============================================================================
# Part 3: Merge functionality
# ============================================================================

class TestBranchMerge(unittest.TestCase):
    """Branch merge tests."""

    def setUp(self):
        self.mgr = _make_manager()
        self.src = self.mgr.fork("conv-1", 3, name="Source")
        self.tgt = self.mgr.fork("conv-1", 3, name="Target")
        self.mgr.add_branch_message(self.src.branch_id, "conv-1", "user", "Src Q1")
        self.mgr.add_branch_message(self.src.branch_id, "conv-1", "assistant", "Src A1")

    def test_merge_all(self):
        merged = self.mgr.merge_messages(self.src.branch_id, self.tgt.branch_id)
        self.assertEqual(len(merged), 2)
        # Target should now have 2 messages
        tgt_msgs = self.mgr.get_branch_only_messages(self.tgt.branch_id)
        self.assertEqual(len(tgt_msgs), 2)

    def test_merge_tags_metadata(self):
        merged = self.mgr.merge_messages(self.src.branch_id, self.tgt.branch_id)
        self.assertEqual(merged[0].metadata["merged_from"], self.src.branch_id)
        self.assertIn("original_message_id", merged[0].metadata)

    def test_merge_selective(self):
        src_msgs = self.mgr.get_branch_only_messages(self.src.branch_id)
        first_id = src_msgs[0].id
        merged = self.mgr.merge_messages(
            self.src.branch_id, self.tgt.branch_id,
            message_ids=[first_id]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].content, "Src Q1")

    def test_merge_bad_target(self):
        merged = self.mgr.merge_messages(self.src.branch_id, "no-such-id")
        self.assertEqual(merged, [])


# ============================================================================
# Part 4: API routes (file-level checks)
# ============================================================================

class TestAPIRoutes(unittest.TestCase):
    """Verify routes file structure and endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.routes_src = _read(os.path.join(API_DIR, "routes_branches.py"))

    def test_file_exists(self):
        self.assertTrue(os.path.exists(os.path.join(API_DIR, "routes_branches.py")))

    def test_router_prefix(self):
        self.assertIn('prefix="/api/branches"', self.routes_src)

    def test_fork_endpoint(self):
        self.assertIn("/fork", self.routes_src)
        self.assertIn("def fork_conversation", self.routes_src)

    def test_compare_endpoint(self):
        self.assertIn("/compare", self.routes_src)
        self.assertIn("def compare_branches", self.routes_src)

    def test_merge_endpoint(self):
        self.assertIn("/merge", self.routes_src)
        self.assertIn("def merge_branches", self.routes_src)

    def test_tree_endpoint(self):
        self.assertIn("/tree", self.routes_src)
        self.assertIn("def get_branch_tree", self.routes_src)

    def test_messages_endpoint(self):
        self.assertIn("/messages", self.routes_src)
        self.assertIn("def get_branch_messages", self.routes_src)

    def test_delete_endpoint(self):
        self.assertIn("def delete_branch", self.routes_src)

    def test_update_endpoint(self):
        self.assertIn("def update_branch", self.routes_src)

    def test_pydantic_schemas(self):
        for schema in ["ForkRequest", "BranchUpdateRequest", "CompareRequest",
                       "MergeRequest", "BranchResponse", "BranchMessageResponse",
                       "AddBranchMessageRequest"]:
            self.assertIn(f"class {schema}", self.routes_src)

    def test_route_ordering_literals_first(self):
        """Literal routes (/fork, /compare, /merge, /detail) must appear
        before catch-all {id} routes."""
        fork_pos = self.routes_src.index('"/fork"')
        compare_pos = self.routes_src.index('"/compare"')
        merge_pos = self.routes_src.index('"/merge"')
        detail_pos = self.routes_src.index('"/detail/')
        list_pos = self.routes_src.index('"/{conversation_id}"')
        # All literals before catch-all
        self.assertLess(fork_pos, list_pos)
        self.assertLess(compare_pos, list_pos)
        self.assertLess(merge_pos, list_pos)
        self.assertLess(detail_pos, list_pos)

    def test_check_available_helper(self):
        self.assertIn("def _check_available", self.routes_src)
        self.assertIn("BRANCHES_AVAILABLE", self.routes_src)


# ============================================================================
# Part 5: Frontend checks
# ============================================================================

class TestFrontendTypes(unittest.TestCase):
    """Verify TypeScript type definitions."""

    @classmethod
    def setUpClass(cls):
        cls.types_src = _read(os.path.join(FRONTEND_SRC, "lib", "types.ts"))

    def test_branch_interface(self):
        self.assertIn("export interface Branch", self.types_src)
        self.assertIn("branch_id: string", self.types_src)
        self.assertIn("fork_message_id: number", self.types_src)

    def test_branch_stats_interface(self):
        self.assertIn("export interface BranchStats", self.types_src)
        self.assertIn("message_count: number", self.types_src)

    def test_branch_tree_node_interface(self):
        self.assertIn("export interface BranchTreeNode", self.types_src)
        self.assertIn("children: BranchTreeNode[]", self.types_src)

    def test_branch_comparison_interface(self):
        self.assertIn("export interface BranchComparison", self.types_src)
        self.assertIn("branch_a_messages", self.types_src)

    def test_branch_fork_request(self):
        self.assertIn("export interface BranchForkRequest", self.types_src)

    def test_branch_merge_request(self):
        self.assertIn("export interface BranchMergeRequest", self.types_src)


class TestFrontendAPIClient(unittest.TestCase):
    """Verify branches API client."""

    @classmethod
    def setUpClass(cls):
        cls.api_src = _read(os.path.join(API_TS_DIR, "branches.ts"))

    def test_file_exists(self):
        self.assertTrue(os.path.exists(os.path.join(API_TS_DIR, "branches.ts")))

    def test_imports_client(self):
        self.assertIn("import { apiGet, apiPost, apiPut, apiDelete }", self.api_src)

    def test_fork_function(self):
        self.assertIn("export async function forkBranch", self.api_src)

    def test_list_function(self):
        self.assertIn("export async function listBranches", self.api_src)

    def test_compare_function(self):
        self.assertIn("export async function compareBranches", self.api_src)

    def test_merge_function(self):
        self.assertIn("export async function mergeBranches", self.api_src)

    def test_tree_function(self):
        self.assertIn("export async function getBranchTree", self.api_src)

    def test_delete_function(self):
        self.assertIn("export async function deleteBranch", self.api_src)

    def test_update_function(self):
        self.assertIn("export async function updateBranch", self.api_src)

    def test_add_message_function(self):
        self.assertIn("export async function addBranchMessage", self.api_src)

    def test_base_path(self):
        self.assertIn("/api/branches", self.api_src)


class TestBranchExplorerComponent(unittest.TestCase):
    """Verify BranchExplorer.svelte component."""

    @classmethod
    def setUpClass(cls):
        cls.svelte_src = _read(os.path.join(CHAT_DIR, "BranchExplorer.svelte"))

    def test_file_exists(self):
        self.assertTrue(os.path.exists(os.path.join(CHAT_DIR, "BranchExplorer.svelte")))

    def test_exports_conversation_id(self):
        self.assertIn("export let conversationId", self.svelte_src)

    def test_exports_current_message_id(self):
        self.assertIn("export let currentMessageId", self.svelte_src)

    def test_dispatches_switch_branch(self):
        self.assertIn("switchBranch", self.svelte_src)

    def test_dispatches_fork(self):
        self.assertIn("dispatch('fork'", self.svelte_src)

    def test_imports_api(self):
        self.assertIn("from '$lib/api/branches'", self.svelte_src)

    def test_no_hardcoded_hex_in_css(self):
        style_match = re.search(r"<style>(.*?)</style>", self.svelte_src, re.DOTALL)
        self.assertIsNotNone(style_match, "Style block not found")
        css = style_match.group(1)
        hex_matches = re.findall(r"#[0-9a-fA-F]{3,8}", css)
        self.assertEqual(hex_matches, [], f"Hardcoded hex in CSS: {hex_matches}")

    def test_uses_oo_css_variables(self):
        style_match = re.search(r"<style>(.*?)</style>", self.svelte_src, re.DOTALL)
        css = style_match.group(1)
        var_refs = re.findall(r"var\(--oo-", css)
        self.assertGreater(len(var_refs), 10, "Too few --oo-* CSS variable references")

    def test_tree_visualization(self):
        self.assertIn("tree-node", self.svelte_src)
        self.assertIn("tree-connector", self.svelte_src)

    def test_compare_view(self):
        self.assertIn("compare-columns", self.svelte_src)
        self.assertIn("compare-col", self.svelte_src)

    def test_merge_button(self):
        self.assertIn("merge-btn", self.svelte_src)

    def test_no_emojis(self):
        # Quick emoji check - no common emoji ranges
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F900-\U0001F9FF\U00002702-\U000027B0]"
        )
        self.assertIsNone(
            emoji_pattern.search(self.svelte_src),
            "Emoji found in BranchExplorer.svelte"
        )


class TestChatPageIntegration(unittest.TestCase):
    """Verify BranchExplorer is integrated in the chat page."""

    @classmethod
    def setUpClass(cls):
        cls.page_src = _read(
            os.path.join(ROUTES_DIR, "chat", "[id]", "+page.svelte")
        )

    def test_imports_branch_explorer(self):
        self.assertIn("import BranchExplorer", self.page_src)

    def test_selected_message_id_state(self):
        self.assertIn("selectedMessageId", self.page_src)

    def test_branch_explorer_rendered(self):
        self.assertIn("<BranchExplorer", self.page_src)

    def test_message_wrapper_click(self):
        self.assertIn("selected-fork", self.page_src)

    def test_no_hardcoded_hex_in_css(self):
        style_match = re.search(r"<style>(.*?)</style>", self.page_src, re.DOTALL)
        if style_match:
            css = style_match.group(1)
            hex_matches = re.findall(r"#[0-9a-fA-F]{3,8}", css)
            self.assertEqual(hex_matches, [], f"Hardcoded hex: {hex_matches}")


# ============================================================================
# Part 6: Config persistence
# ============================================================================

class TestBranchesConfig(unittest.TestCase):
    """Verify branches.yaml configuration."""

    @classmethod
    def setUpClass(cls):
        config_path = os.path.join(CONFIG_DIR, "branches.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cls.cfg = yaml.safe_load(f)

    def test_file_exists(self):
        self.assertTrue(os.path.exists(os.path.join(CONFIG_DIR, "branches.yaml")))

    def test_branches_section(self):
        self.assertIn("branches", self.cfg)
        self.assertIn("max_per_conversation", self.cfg["branches"])
        self.assertIn("color_palette", self.cfg["branches"])

    def test_color_palette_length(self):
        palette = self.cfg["branches"]["color_palette"]
        self.assertGreaterEqual(len(palette), 4)

    def test_merge_section(self):
        self.assertIn("merge", self.cfg)
        self.assertIn("max_messages_per_merge", self.cfg["merge"])
        self.assertIn("tag_merged_messages", self.cfg["merge"])

    def test_display_section(self):
        self.assertIn("display", self.cfg)
        self.assertIn("sidebar_mode", self.cfg["display"])


# ============================================================================
# Part 7: Integration wiring
# ============================================================================

class TestIntegrationWiring(unittest.TestCase):
    """Verify app.py and deps.py integration."""

    @classmethod
    def setUpClass(cls):
        cls.app_src = _read(os.path.join(API_DIR, "app.py"))
        cls.deps_src = _read(os.path.join(API_DIR, "deps.py"))

    def test_app_imports_branches_router(self):
        self.assertIn("from .routes_branches import router as branches_router", self.app_src)

    def test_app_includes_branches_router(self):
        self.assertIn("app.include_router(branches_router)", self.app_src)

    def test_version_bump(self):
        self.assertIn('"1.10.0"', self.app_src)
        self.assertNotIn('"1.9.8"', self.app_src)

    def test_health_check_branches(self):
        self.assertIn("BRANCHES_AVAILABLE", self.app_src)
        self.assertIn('"branches"', self.app_src)

    def test_deps_imports_branch_manager(self):
        self.assertIn("branch_manager", self.deps_src)
        self.assertIn("BRANCHES_AVAILABLE", self.deps_src)

    def test_deps_fallback_pattern(self):
        self.assertIn("except ImportError:", self.deps_src)
        # Verify BRANCHES_AVAILABLE = False fallback
        self.assertIn("BRANCHES_AVAILABLE = False", self.deps_src)
        self.assertIn("branch_manager = None", self.deps_src)


# ============================================================================
# Part 8: Data class serialization
# ============================================================================

class TestDataClasses(unittest.TestCase):
    """Verify data class to_dict and serialization."""

    def test_branch_to_dict(self):
        b = Branch(
            branch_id="b1", conversation_id="c1", parent_branch_id=None,
            fork_message_id=5, name="Test", color="#AAA",
            created_at="2025-01-01", updated_at="2025-01-01",
        )
        d = b.to_dict()
        self.assertEqual(d["branch_id"], "b1")
        self.assertEqual(d["fork_message_id"], 5)

    def test_branch_tree_node_to_dict_recursive(self):
        child = BranchTreeNode(
            branch_id="c1", name="Child", color="#BBB",
            fork_message_id=3, message_count=2,
            last_model=None, last_activity=None,
        )
        root = BranchTreeNode(
            branch_id=None, name="Main", color="#AAA",
            fork_message_id=None, message_count=10,
            last_model="qwen3:32b", last_activity="2025-01-01",
            children=[child],
        )
        d = root.to_dict()
        self.assertEqual(len(d["children"]), 1)
        self.assertEqual(d["children"][0]["name"], "Child")

    def test_comparison_to_dict(self):
        comp = BranchComparison(
            branch_a_id="a", branch_b_id="b",
            branch_a_name="A", branch_b_name="B",
            shared_messages=[{"id": 1}],
            branch_a_messages=[{"id": 2}],
            branch_b_messages=[{"id": 3}],
            fork_message_id=1,
        )
        d = comp.to_dict()
        self.assertEqual(d["fork_message_id"], 1)
        self.assertEqual(len(d["shared_messages"]), 1)

    def test_get_config(self):
        mgr = _make_manager()
        cfg = mgr.get_config()
        self.assertIn("branches", cfg)
        self.assertIn("merge", cfg)


if __name__ == "__main__":
    unittest.main()
