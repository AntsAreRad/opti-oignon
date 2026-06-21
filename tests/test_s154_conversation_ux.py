"""
tests/test_s154_conversation_ux.py -- S154 conversation UX improvements tests.

Verifies:
- BranchTreeNode dataclass: to_dict, flatten, find_node, total_descendants
- build_branch_tree: flat list to nested tree, depth assignment, sort order
- MessageCollapseConfig: validation, should_collapse, truncated_content, remaining_lines
- FileUploadConfig: validate_file, validate_batch, get_file_icon, format_size
- RAG_ALLOWED_EXTENSIONS completeness
- FILE_TYPE_ICONS coverage
- Frontend file existence: BranchTreeNodeItem, BranchDiff, ScrollToBottom
- BranchExplorer.svelte uses BranchTreeNodeItem import
- ChatMessage.svelte: fork button, collapse toggle, code block copy
- FileUpload.svelte: full-window overlay, progress tracking, batch props
- Version bump check (3.2.3)
"""

import importlib.util
import os
import re
import sys
import types

# -- Isolation stubs (standard pattern) --
for mod_name in [
    "opti_oignon",
    "opti_oignon.db_utils",
    "opti_oignon.config",
    "opti_oignon.auth",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUX_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "conversation_ux.py")
VERSION_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "__version__.py")

# Frontend paths
BRANCH_TREE_NODE_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "chat",
    "BranchTreeNodeItem.svelte",
)
BRANCH_DIFF_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "chat",
    "BranchDiff.svelte",
)
SCROLL_TO_BOTTOM_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "chat",
    "ScrollToBottom.svelte",
)
BRANCH_EXPLORER_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "chat",
    "BranchExplorer.svelte",
)
CHAT_MESSAGE_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "chat",
    "ChatMessage.svelte",
)
FILE_UPLOAD_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "chat",
    "FileUpload.svelte",
)


def _load_module(name: str, path: str):
    """Load a Python module by file path (isolation pattern)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cux():
    """Load conversation_ux module."""
    return _load_module("conversation_ux", CUX_PATH)


# -- Sample data fixtures --

@pytest.fixture
def sample_branches():
    """Flat list of branches for tree building."""
    return [
        {
            "branch_id": "b1",
            "name": "Feature A",
            "parent_branch_id": None,
            "fork_message_id": 5,
            "message_count": 12,
            "last_activity": "2026-03-20T10:00:00Z",
            "color": "#ff0000",
        },
        {
            "branch_id": "b2",
            "name": "Feature B",
            "parent_branch_id": None,
            "fork_message_id": 10,
            "message_count": 8,
            "last_activity": "2026-03-20T11:00:00Z",
            "color": "#00ff00",
        },
        {
            "branch_id": "b3",
            "name": "Sub-branch",
            "parent_branch_id": "b1",
            "fork_message_id": 7,
            "message_count": 3,
            "last_activity": "2026-03-20T12:00:00Z",
            "color": "#0000ff",
        },
        {
            "branch_id": "b4",
            "name": "Deep child",
            "parent_branch_id": "b3",
            "fork_message_id": 2,
            "message_count": 1,
            "last_activity": "2026-03-20T13:00:00Z",
            "color": "#ffff00",
        },
    ]


# ---- BranchTreeNode dataclass ----

class TestBranchTreeNode:
    """Tests for BranchTreeNode dataclass."""

    def test_basic_creation(self, cux):
        node = cux.BranchTreeNode(
            branch_id="test",
            name="Test",
            color="#aaa",
            message_count=5,
            last_activity="2026-01-01",
            fork_message_id=3,
            parent_branch_id=None,
            depth=0,
        )
        assert node.branch_id == "test"
        assert node.name == "Test"
        assert node.depth == 0
        assert node.collapsed is False
        assert node.children == []

    def test_to_dict(self, cux):
        node = cux.BranchTreeNode(
            branch_id="x", name="X", color="#fff", message_count=1,
            last_activity="", fork_message_id=None, parent_branch_id=None, depth=0,
        )
        d = node.to_dict()
        assert d["branch_id"] == "x"
        assert d["name"] == "X"
        assert isinstance(d["children"], list)
        assert d["collapsed"] is False

    def test_to_dict_with_children(self, cux):
        child = cux.BranchTreeNode(
            branch_id="c1", name="C1", color="#aaa", message_count=0,
            last_activity="", fork_message_id=1, parent_branch_id="p", depth=1,
        )
        parent = cux.BranchTreeNode(
            branch_id="p", name="P", color="#bbb", message_count=0,
            last_activity="", fork_message_id=None, parent_branch_id=None, depth=0,
            children=[child],
        )
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["branch_id"] == "c1"
        assert d["children"][0]["depth"] == 1

    def test_flatten_single(self, cux):
        node = cux.BranchTreeNode(
            branch_id="a", name="A", color="#000", message_count=0,
            last_activity="", fork_message_id=None, parent_branch_id=None, depth=0,
        )
        flat = node.flatten()
        assert len(flat) == 1
        assert flat[0].branch_id == "a"

    def test_flatten_with_children(self, cux):
        grandchild = cux.BranchTreeNode(
            branch_id="gc", name="GC", color="#000", message_count=0,
            last_activity="", fork_message_id=1, parent_branch_id="c", depth=2,
        )
        child = cux.BranchTreeNode(
            branch_id="c", name="C", color="#000", message_count=0,
            last_activity="", fork_message_id=1, parent_branch_id="r", depth=1,
            children=[grandchild],
        )
        root = cux.BranchTreeNode(
            branch_id="r", name="R", color="#000", message_count=0,
            last_activity="", fork_message_id=None, parent_branch_id=None, depth=0,
            children=[child],
        )
        flat = root.flatten()
        assert len(flat) == 3
        assert [n.branch_id for n in flat] == ["r", "c", "gc"]

    def test_find_node_root(self, cux):
        node = cux.BranchTreeNode(
            branch_id=None, name="Main", color="#000", message_count=0,
            last_activity="", fork_message_id=None, parent_branch_id=None, depth=0,
        )
        assert node.find_node(None) is node

    def test_find_node_child(self, cux):
        child = cux.BranchTreeNode(
            branch_id="c1", name="C1", color="#000", message_count=0,
            last_activity="", fork_message_id=1, parent_branch_id=None, depth=1,
        )
        root = cux.BranchTreeNode(
            branch_id=None, name="Main", color="#000", message_count=0,
            last_activity="", fork_message_id=None, parent_branch_id=None, depth=0,
            children=[child],
        )
        found = root.find_node("c1")
        assert found is child

    def test_find_node_missing(self, cux):
        node = cux.BranchTreeNode(
            branch_id=None, name="Main", color="#000", message_count=0,
            last_activity="", fork_message_id=None, parent_branch_id=None, depth=0,
        )
        assert node.find_node("nonexistent") is None

    def test_total_descendants_empty(self, cux):
        node = cux.BranchTreeNode(
            branch_id="a", name="A", color="#000", message_count=0,
            last_activity="", fork_message_id=None, parent_branch_id=None, depth=0,
        )
        assert node.total_descendants == 0

    def test_total_descendants_nested(self, cux):
        gc = cux.BranchTreeNode(
            branch_id="gc", name="GC", color="#000", message_count=0,
            last_activity="", fork_message_id=1, parent_branch_id="c", depth=2,
        )
        c = cux.BranchTreeNode(
            branch_id="c", name="C", color="#000", message_count=0,
            last_activity="", fork_message_id=1, parent_branch_id="r", depth=1,
            children=[gc],
        )
        root = cux.BranchTreeNode(
            branch_id="r", name="R", color="#000", message_count=0,
            last_activity="", fork_message_id=None, parent_branch_id=None, depth=0,
            children=[c],
        )
        assert root.total_descendants == 2
        assert c.total_descendants == 1


# ---- build_branch_tree ----

class TestBuildBranchTree:
    """Tests for build_branch_tree function."""

    def test_empty_list(self, cux):
        tree = cux.build_branch_tree([])
        assert tree.branch_id is None
        assert tree.name == "Main"
        assert tree.children == []
        assert tree.depth == 0

    def test_custom_conversation_name(self, cux):
        tree = cux.build_branch_tree([], conversation_name="My Chat")
        assert tree.name == "My Chat"

    def test_single_branch(self, cux):
        branches = [{"branch_id": "b1", "name": "B1", "parent_branch_id": None, "fork_message_id": 5}]
        tree = cux.build_branch_tree(branches)
        assert len(tree.children) == 1
        assert tree.children[0].branch_id == "b1"
        assert tree.children[0].depth == 1

    def test_two_branches_sorted(self, cux):
        branches = [
            {"branch_id": "b2", "name": "B2", "parent_branch_id": None, "fork_message_id": 10},
            {"branch_id": "b1", "name": "B1", "parent_branch_id": None, "fork_message_id": 5},
        ]
        tree = cux.build_branch_tree(branches)
        assert len(tree.children) == 2
        # Should be sorted by fork_message_id
        assert tree.children[0].branch_id == "b1"
        assert tree.children[1].branch_id == "b2"

    def test_nested_branches(self, cux, sample_branches):
        tree = cux.build_branch_tree(sample_branches)
        assert tree.depth == 0
        assert len(tree.children) == 2  # b1 and b2 are direct children

        b1 = tree.find_node("b1")
        assert b1 is not None
        assert b1.depth == 1
        assert len(b1.children) == 1  # b3 is child of b1

        b3 = tree.find_node("b3")
        assert b3 is not None
        assert b3.depth == 2
        assert b3.parent_branch_id == "b1"
        assert len(b3.children) == 1  # b4 is child of b3

        b4 = tree.find_node("b4")
        assert b4 is not None
        assert b4.depth == 3

    def test_tree_preserves_metadata(self, cux, sample_branches):
        tree = cux.build_branch_tree(sample_branches)
        b1 = tree.find_node("b1")
        assert b1.message_count == 12
        assert b1.last_activity == "2026-03-20T10:00:00Z"
        assert b1.color == "#ff0000"

    def test_missing_optional_fields(self, cux):
        branches = [{"branch_id": "b1", "parent_branch_id": None, "fork_message_id": 1}]
        tree = cux.build_branch_tree(branches)
        child = tree.children[0]
        assert child.message_count == 0
        assert child.last_activity == ""
        assert child.color == "#888888"

    def test_total_descendants_on_tree(self, cux, sample_branches):
        tree = cux.build_branch_tree(sample_branches)
        assert tree.total_descendants == 4  # b1, b2, b3, b4

    def test_flatten_on_tree(self, cux, sample_branches):
        tree = cux.build_branch_tree(sample_branches)
        flat = tree.flatten()
        assert len(flat) == 5  # root + 4 branches
        ids = [n.branch_id for n in flat]
        assert None in ids  # root
        for bid in ["b1", "b2", "b3", "b4"]:
            assert bid in ids

    def test_to_dict_roundtrip(self, cux, sample_branches):
        tree = cux.build_branch_tree(sample_branches)
        d = tree.to_dict()
        assert d["branch_id"] is None
        assert d["name"] == "Main"
        assert len(d["children"]) == 2
        # Verify nested structure
        b1_dict = next(c for c in d["children"] if c["branch_id"] == "b1")
        assert len(b1_dict["children"]) == 1
        b3_dict = b1_dict["children"][0]
        assert b3_dict["branch_id"] == "b3"
        assert len(b3_dict["children"]) == 1


# ---- MessageCollapseConfig ----

class TestMessageCollapseConfig:
    """Tests for MessageCollapseConfig."""

    def test_defaults(self, cux):
        cfg = cux.MessageCollapseConfig()
        assert cfg.line_threshold == 500
        assert cfg.default_collapsed is True

    def test_validate_ok(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=100)
        assert cfg.validate() == []

    def test_validate_too_low(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=5)
        errors = cfg.validate()
        assert len(errors) == 1
        assert "10" in errors[0]

    def test_validate_too_high(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=20000)
        errors = cfg.validate()
        assert len(errors) == 1
        assert "10000" in errors[0]

    def test_should_collapse_short(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=10)
        content = "line1\nline2\nline3"
        assert cfg.should_collapse(content) is False

    def test_should_collapse_long(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=5)
        content = "\n".join(f"line {i}" for i in range(20))
        assert cfg.should_collapse(content) is True

    def test_should_collapse_exact_threshold(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=10)
        content = "\n".join(f"line {i}" for i in range(10))
        # 10 lines = 10, threshold is 10, not > so should not collapse
        assert cfg.should_collapse(content) is False

    def test_should_collapse_one_over(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=10)
        content = "\n".join(f"line {i}" for i in range(11))
        assert cfg.should_collapse(content) is True

    def test_truncated_content(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=100)
        content = "\n".join(f"line {i}" for i in range(50))
        truncated = cfg.truncated_content(content, visible_lines=5)
        lines = truncated.split("\n")
        assert len(lines) == 5

    def test_truncated_content_default_visible(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=500)
        content = "\n".join(f"line {i}" for i in range(600))
        truncated = cfg.truncated_content(content)
        lines = truncated.split("\n")
        # Default visible_lines = min(20, 500//10) = 20
        assert len(lines) == 20

    def test_remaining_lines(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=100)
        content = "\n".join(f"line {i}" for i in range(50))
        remaining = cfg.remaining_lines(content, visible_lines=5)
        assert remaining == 45

    def test_to_dict(self, cux):
        cfg = cux.MessageCollapseConfig(line_threshold=200, default_collapsed=False)
        d = cfg.to_dict()
        assert d["line_threshold"] == 200
        assert d["default_collapsed"] is False

    def test_from_dict(self, cux):
        cfg = cux.MessageCollapseConfig.from_dict({"line_threshold": 300, "default_collapsed": True})
        assert cfg.line_threshold == 300
        assert cfg.default_collapsed is True

    def test_from_dict_defaults(self, cux):
        cfg = cux.MessageCollapseConfig.from_dict({})
        assert cfg.line_threshold == 500


# ---- FileUploadConfig ----

class TestFileUploadConfig:
    """Tests for FileUploadConfig."""

    def test_defaults(self, cux):
        cfg = cux.FileUploadConfig()
        assert cfg.max_file_size_bytes == 500_000
        assert cfg.max_batch_size == 10
        assert len(cfg.allowed_extensions) > 20

    def test_validate_file_ok(self, cux):
        cfg = cux.FileUploadConfig()
        assert cfg.validate_file("script.py", 1000) is None

    def test_validate_file_no_extension(self, cux):
        cfg = cux.FileUploadConfig()
        result = cfg.validate_file("Makefile", 100)
        assert result is not None
        assert "no extension" in result.lower()

    def test_validate_file_bad_extension(self, cux):
        cfg = cux.FileUploadConfig()
        result = cfg.validate_file("photo.png", 100)
        assert result is not None
        assert "unsupported" in result.lower()

    def test_validate_file_too_large(self, cux):
        cfg = cux.FileUploadConfig(max_file_size_bytes=1000)
        result = cfg.validate_file("data.csv", 2000)
        assert result is not None
        assert "too large" in result.lower()

    def test_validate_file_at_limit(self, cux):
        cfg = cux.FileUploadConfig(max_file_size_bytes=1000)
        assert cfg.validate_file("data.csv", 1000) is None

    def test_validate_file_over_limit(self, cux):
        cfg = cux.FileUploadConfig(max_file_size_bytes=1000)
        result = cfg.validate_file("data.csv", 1001)
        assert result is not None

    def test_validate_batch_ok(self, cux):
        cfg = cux.FileUploadConfig(max_batch_size=5)
        assert cfg.validate_batch(3) is None

    def test_validate_batch_too_many(self, cux):
        cfg = cux.FileUploadConfig(max_batch_size=5)
        result = cfg.validate_batch(10)
        assert result is not None
        assert "too many" in result.lower()

    def test_validate_batch_zero(self, cux):
        cfg = cux.FileUploadConfig()
        result = cfg.validate_batch(0)
        assert result is not None
        assert "no files" in result.lower()

    def test_get_file_icon_python(self, cux):
        cfg = cux.FileUploadConfig()
        assert cfg.get_file_icon("script.py") == "python"

    def test_get_file_icon_unknown(self, cux):
        cfg = cux.FileUploadConfig()
        assert cfg.get_file_icon("data.xyz") == "file"

    def test_get_file_icon_no_ext(self, cux):
        cfg = cux.FileUploadConfig()
        assert cfg.get_file_icon("Makefile") == "file"

    def test_format_size_bytes(self, cux):
        cfg = cux.FileUploadConfig()
        assert cfg.format_size(500) == "500B"

    def test_format_size_kb(self, cux):
        cfg = cux.FileUploadConfig()
        result = cfg.format_size(2048)
        assert "KB" in result

    def test_format_size_mb(self, cux):
        cfg = cux.FileUploadConfig()
        result = cfg.format_size(2 * 1024 * 1024)
        assert "MB" in result

    def test_to_dict(self, cux):
        cfg = cux.FileUploadConfig()
        d = cfg.to_dict()
        assert "max_file_size_bytes" in d
        assert "max_batch_size" in d
        assert "allowed_extensions" in d
        assert isinstance(d["allowed_extensions"], list)
        assert sorted(d["allowed_extensions"]) == d["allowed_extensions"]

    def test_from_dict(self, cux):
        cfg = cux.FileUploadConfig.from_dict({
            "max_file_size_bytes": 1_000_000,
            "max_batch_size": 20,
        })
        assert cfg.max_file_size_bytes == 1_000_000
        assert cfg.max_batch_size == 20

    def test_from_dict_custom_extensions(self, cux):
        cfg = cux.FileUploadConfig.from_dict({
            "allowed_extensions": [".py", ".txt"],
        })
        assert cfg.allowed_extensions == frozenset({".py", ".txt"})


# ---- RAG_ALLOWED_EXTENSIONS ----

class TestRAGAllowedExtensions:
    """Tests for RAG_ALLOWED_EXTENSIONS constant."""

    def test_contains_common_types(self, cux):
        for ext in [".py", ".r", ".R", ".md", ".txt", ".json", ".csv", ".pdf"]:
            assert ext in cux.RAG_ALLOWED_EXTENSIONS, f"{ext} missing"

    def test_is_frozenset(self, cux):
        assert isinstance(cux.RAG_ALLOWED_EXTENSIONS, frozenset)

    def test_has_code_extensions(self, cux):
        for ext in [".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"]:
            assert ext in cux.RAG_ALLOWED_EXTENSIONS, f"{ext} missing"


# ---- FILE_TYPE_ICONS ----

class TestFileTypeIcons:
    """Tests for FILE_TYPE_ICONS mapping."""

    def test_has_python(self, cux):
        assert cux.FILE_TYPE_ICONS[".py"] == "python"

    def test_has_pdf(self, cux):
        assert ".pdf" in cux.FILE_TYPE_ICONS

    def test_has_markdown(self, cux):
        assert ".md" in cux.FILE_TYPE_ICONS

    def test_all_values_are_strings(self, cux):
        for k, v in cux.FILE_TYPE_ICONS.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ---- Frontend file existence ----

class TestFrontendFileExistence:
    """Verify all new S154 frontend files exist."""

    def test_branch_tree_node_item_exists(self):
        assert os.path.isfile(BRANCH_TREE_NODE_PATH), "BranchTreeNodeItem.svelte missing"

    def test_branch_diff_exists(self):
        assert os.path.isfile(BRANCH_DIFF_PATH), "BranchDiff.svelte missing"

    def test_scroll_to_bottom_exists(self):
        assert os.path.isfile(SCROLL_TO_BOTTOM_PATH), "ScrollToBottom.svelte missing"

    def test_branch_explorer_exists(self):
        assert os.path.isfile(BRANCH_EXPLORER_PATH), "BranchExplorer.svelte missing"

    def test_chat_message_exists(self):
        assert os.path.isfile(CHAT_MESSAGE_PATH), "ChatMessage.svelte missing"

    def test_file_upload_exists(self):
        assert os.path.isfile(FILE_UPLOAD_PATH), "FileUpload.svelte missing"


# ---- BranchExplorer integration ----

class TestBranchExplorerIntegration:
    """Verify BranchExplorer.svelte uses BranchTreeNodeItem."""

    @pytest.fixture(scope="class")
    def explorer_content(self):
        with open(BRANCH_EXPLORER_PATH) as f:
            return f.read()

    def test_imports_branch_tree_node_item(self, explorer_content):
        assert "BranchTreeNodeItem" in explorer_content

    def test_import_from_correct_path(self, explorer_content):
        assert "from './BranchTreeNodeItem.svelte'" in explorer_content

    def test_uses_component_tag(self, explorer_content):
        assert "<BranchTreeNodeItem" in explorer_content

    def test_passes_active_branch_id(self, explorer_content):
        assert "activeBranchId" in explorer_content

    def test_has_role_tree(self, explorer_content):
        assert 'role="tree"' in explorer_content

    def test_s154_comment_present(self, explorer_content):
        assert "S154" in explorer_content


# ---- BranchTreeNodeItem content ----

class TestBranchTreeNodeItemContent:
    """Verify BranchTreeNodeItem.svelte content."""

    @pytest.fixture(scope="class")
    def node_content(self):
        with open(BRANCH_TREE_NODE_PATH) as f:
            return f.read()

    def test_has_svelte_self(self, node_content):
        assert "svelte:self" in node_content

    def test_has_collapse_toggle(self, node_content):
        assert "collapse-toggle" in node_content

    def test_has_role_treeitem(self, node_content):
        assert 'role="treeitem"' in node_content

    def test_has_aria_expanded(self, node_content):
        assert "aria-expanded" in node_content

    def test_no_hardcoded_hex_colors(self, node_content):
        # Extract CSS only
        style_match = re.search(r"<style>(.*?)</style>", node_content, re.DOTALL)
        if style_match:
            css = style_match.group(1)
            hex_colors = re.findall(r"(?<!var\()#[0-9a-fA-F]{3,8}\b", css)
            assert len(hex_colors) == 0, f"Hardcoded hex colors in CSS: {hex_colors}"

    def test_dispatches_switch_branch(self, node_content):
        assert "switchBranch" in node_content

    def test_has_message_count_display(self, node_content):
        assert "messageCount" in node_content or "message_count" in node_content


# ---- BranchDiff content ----

class TestBranchDiffContent:
    """Verify BranchDiff.svelte content."""

    @pytest.fixture(scope="class")
    def diff_content(self):
        with open(BRANCH_DIFF_PATH) as f:
            return f.read()

    def test_has_side_by_side_mode(self, diff_content):
        assert "side-by-side" in diff_content

    def test_has_inline_mode(self, diff_content):
        assert "inline" in diff_content

    def test_has_divergence_marker(self, diff_content):
        assert "divergence" in diff_content.lower()

    def test_imports_compare_branches(self, diff_content):
        assert "compareBranches" in diff_content

    def test_has_shared_messages_display(self, diff_content):
        assert "shared_messages" in diff_content

    def test_no_hardcoded_hex_in_css(self, diff_content):
        style_match = re.search(r"<style>(.*?)</style>", diff_content, re.DOTALL)
        if style_match:
            css = style_match.group(1)
            hex_colors = re.findall(r"(?<!var\()#[0-9a-fA-F]{3,8}\b", css)
            assert len(hex_colors) == 0, f"Hardcoded hex colors: {hex_colors}"


# ---- ChatMessage S154 features ----

class TestChatMessageS154:
    """Verify ChatMessage.svelte S154 additions."""

    @pytest.fixture(scope="class")
    def msg_content(self):
        with open(CHAT_MESSAGE_PATH) as f:
            return f.read()

    def test_has_fork_event(self, msg_content):
        assert "fork" in msg_content

    def test_has_fork_button(self, msg_content):
        assert "Fork conversation from this message" in msg_content or "Fork from this message" in msg_content

    def test_has_conversation_id_prop(self, msg_content):
        assert "conversationId" in msg_content

    def test_has_collapse_threshold_prop(self, msg_content):
        assert "collapseThreshold" in msg_content

    def test_has_show_more_toggle(self, msg_content):
        assert "Show more" in msg_content
        assert "Show less" in msg_content

    def test_has_code_block_copy(self, msg_content):
        assert "copyCodeBlock" in msg_content

    def test_has_code_blocks_extraction(self, msg_content):
        assert "extractCodeBlocks" in msg_content

    def test_has_collapse_toggle_btn_class(self, msg_content):
        assert "collapse-toggle-btn" in msg_content

    def test_has_code_copy_btn_class(self, msg_content):
        assert "code-copy-btn" in msg_content

    def test_s154_comment(self, msg_content):
        assert "S154" in msg_content


# ---- FileUpload S154 features ----

class TestFileUploadS154:
    """Verify FileUpload.svelte S154 additions."""

    @pytest.fixture(scope="class")
    def upload_content(self):
        with open(FILE_UPLOAD_PATH) as f:
            return f.read()

    def test_has_full_window_overlay(self, upload_content):
        assert "drop-overlay" in upload_content

    def test_has_position_fixed(self, upload_content):
        assert "position: fixed" in upload_content

    def test_has_window_event_listeners(self, upload_content):
        assert "window.addEventListener" in upload_content

    def test_has_drag_depth_tracking(self, upload_content):
        assert "dragDepth" in upload_content

    def test_has_upload_progress(self, upload_content):
        assert "uploadQueue" in upload_content or "upload-progress" in upload_content

    def test_has_max_file_size_prop(self, upload_content):
        assert "maxFileSize" in upload_content

    def test_has_max_batch_size_prop(self, upload_content):
        assert "maxBatchSize" in upload_content

    def test_has_validate_file_with_size(self, upload_content):
        assert "validateFileWithSize" in upload_content

    def test_has_batch_validation(self, upload_content):
        assert "Too many files" in upload_content

    def test_has_progress_states(self, upload_content):
        for state in ["uploading", "done", "error"]:
            assert state in upload_content

    def test_has_backdrop_blur(self, upload_content):
        assert "backdrop-filter" in upload_content

    def test_s154_comment(self, upload_content):
        assert "S154" in upload_content

    def test_has_on_destroy_cleanup(self, upload_content):
        assert "onDestroy" in upload_content
        assert "removeEventListener" in upload_content


# ---- ScrollToBottom content ----

class TestScrollToBottomContent:
    """Verify ScrollToBottom.svelte content."""

    @pytest.fixture(scope="class")
    def stb_content(self):
        with open(SCROLL_TO_BOTTOM_PATH) as f:
            return f.read()

    def test_has_scroll_container_prop(self, stb_content):
        assert "scrollContainer" in stb_content

    def test_has_new_message_count_prop(self, stb_content):
        assert "newMessageCount" in stb_content

    def test_has_threshold_prop(self, stb_content):
        assert "threshold" in stb_content

    def test_has_smooth_scroll(self, stb_content):
        assert "smooth" in stb_content

    def test_has_scroll_badge(self, stb_content):
        assert "scroll-badge" in stb_content

    def test_has_aria_label(self, stb_content):
        assert "aria-label" in stb_content

    def test_has_debounced_scroll_check(self, stb_content):
        assert "setTimeout" in stb_content

    def test_has_on_destroy_cleanup(self, stb_content):
        assert "onDestroy" in stb_content

    def test_no_hardcoded_hex_in_css(self, stb_content):
        style_match = re.search(r"<style>(.*?)</style>", stb_content, re.DOTALL)
        if style_match:
            css = style_match.group(1)
            # Allow fallback hex in var() patterns
            hex_outside_var = re.findall(r"(?<!,\s)(?<!var\()#[0-9a-fA-F]{3,8}\b", css)
            # Filter out those inside var() fallbacks
            lines = css.split("\n")
            bad = []
            for line in lines:
                stripped = line.strip()
                if re.search(r"#[0-9a-fA-F]{3,8}", stripped):
                    if "var(--" not in stripped:
                        bad.append(stripped)
            assert len(bad) == 0, f"Hardcoded hex colors outside var(): {bad}"


# ---- Version bump ----

class TestVersionBump:
    """Verify version is bumped to 3.2.3."""

    def test_version_is_3_2_3(self):
        with open(VERSION_PATH) as f:
            content = f.read()
        assert '"3.2.3"' in content or "'3.2.3'" in content, \
            f"Version not bumped to 3.2.3. Content: {content.strip()}"
