#!/usr/bin/env python3
"""
Conversation UX utilities.

Provides:
- Branch tree builder: converts flat branch list into nested tree structure
  with collapsible subtrees, enriched node metadata (message count, last activity).
- Message collapse configuration: configurable line threshold for long messages.
- File upload configuration: configurable max file size with validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# -- Branch tree builder --


@dataclass
class BranchTreeNode:
    """A node in the branch tree hierarchy."""

    branch_id: str | None
    name: str
    color: str
    message_count: int
    last_activity: str
    fork_message_id: int | None
    parent_branch_id: str | None
    depth: int
    collapsed: bool = False
    children: list[BranchTreeNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API response."""
        return {
            "branch_id": self.branch_id,
            "name": self.name,
            "color": self.color,
            "message_count": self.message_count,
            "last_activity": self.last_activity,
            "fork_message_id": self.fork_message_id,
            "parent_branch_id": self.parent_branch_id,
            "depth": self.depth,
            "collapsed": self.collapsed,
            "children": [c.to_dict() for c in self.children],
        }

    def flatten(self) -> list[BranchTreeNode]:
        """Return a flat list of all nodes in depth-first order."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result

    def find_node(self, branch_id: str | None) -> BranchTreeNode | None:
        """Find a node by branch_id (None matches the root/main node)."""
        if self.branch_id == branch_id:
            return self
        for child in self.children:
            found = child.find_node(branch_id)
            if found is not None:
                return found
        return None

    @property
    def total_descendants(self) -> int:
        """Count all descendants recursively."""
        count = 0
        for child in self.children:
            count += 1 + child.total_descendants
        return count


def build_branch_tree(
    branches: list[dict[str, Any]],
    conversation_name: str = "Main",
) -> BranchTreeNode:
    """
    Build a nested tree from a flat list of branch dictionaries.

    Each branch dict must have at minimum:
      - branch_id: str
      - name: str
      - parent_branch_id: str | None (None = child of main)
      - fork_message_id: int | None
      - message_count: int (optional, defaults to 0)
      - last_activity: str (optional, defaults to empty)
      - color: str (optional, defaults to '#888888')

    Returns a BranchTreeNode representing the root (main conversation).
    """
    root = BranchTreeNode(
        branch_id=None,
        name=conversation_name,
        color="#4a9e8e",
        message_count=0,
        last_activity="",
        fork_message_id=None,
        parent_branch_id=None,
        depth=0,
    )

    # Index nodes by branch_id
    node_map: dict[str | None, BranchTreeNode] = {None: root}

    # Create all nodes first
    for b in branches:
        bid = b["branch_id"]
        node = BranchTreeNode(
            branch_id=bid,
            name=b.get("name", bid[:8]),
            color=b.get("color", "#888888"),
            message_count=b.get("message_count", 0),
            last_activity=b.get("last_activity", ""),
            fork_message_id=b.get("fork_message_id"),
            parent_branch_id=b.get("parent_branch_id"),
            depth=0,
        )
        node_map[bid] = node

    # Link children to parents
    for b in branches:
        bid = b["branch_id"]
        parent_id = b.get("parent_branch_id")
        node = node_map[bid]
        parent = node_map.get(parent_id, root)
        parent.children.append(node)

    # Compute depths
    _assign_depths(root, 0)

    # Sort children by fork_message_id (chronological order)
    _sort_children(root)

    return root


def _assign_depths(node: BranchTreeNode, depth: int) -> None:
    """Recursively assign depth to each node."""
    node.depth = depth
    for child in node.children:
        _assign_depths(child, depth + 1)


def _sort_children(node: BranchTreeNode) -> None:
    """Recursively sort children by fork_message_id, then name."""
    node.children.sort(key=lambda n: (n.fork_message_id or 0, n.name))
    for child in node.children:
        _sort_children(child)


# -- Message collapse configuration --


@dataclass
class MessageCollapseConfig:
    """Configuration for collapsible long messages."""

    line_threshold: int = 500
    min_threshold: int = 10
    max_threshold: int = 10000
    default_collapsed: bool = True

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty if valid)."""
        errors: list[str] = []
        if not isinstance(self.line_threshold, int):
            errors.append("line_threshold must be an integer")
        elif self.line_threshold < self.min_threshold:
            errors.append(
                f"line_threshold must be >= {self.min_threshold}"
            )
        elif self.line_threshold > self.max_threshold:
            errors.append(
                f"line_threshold must be <= {self.max_threshold}"
            )
        return errors

    def should_collapse(self, content: str) -> bool:
        """Check if content exceeds the line threshold."""
        line_count = content.count("\n") + 1
        return line_count > self.line_threshold

    def truncated_content(self, content: str, visible_lines: int | None = None) -> str:
        """Return the first N lines of content for collapsed display."""
        if visible_lines is None:
            visible_lines = min(20, self.line_threshold // 10)
        lines = content.split("\n")
        if len(lines) <= visible_lines:
            return content
        return "\n".join(lines[:visible_lines])

    def remaining_lines(self, content: str, visible_lines: int | None = None) -> int:
        """Return how many lines are hidden after truncation."""
        if visible_lines is None:
            visible_lines = min(20, self.line_threshold // 10)
        total = content.count("\n") + 1
        return max(0, total - visible_lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "line_threshold": self.line_threshold,
            "default_collapsed": self.default_collapsed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageCollapseConfig:
        """Create from dictionary (e.g. user preferences)."""
        return cls(
            line_threshold=data.get("line_threshold", 500),
            default_collapsed=data.get("default_collapsed", True),
        )


# -- File upload configuration --


# Allowed extensions for RAG ingestion
RAG_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".py", ".r", ".R", ".sh", ".json", ".yaml", ".yml",
    ".csv", ".tsv", ".xml", ".html", ".css", ".js", ".ts", ".jsx", ".tsx",
    ".c", ".cpp", ".h", ".java", ".go", ".rs", ".lua", ".rb", ".pl",
    ".toml", ".ini", ".cfg", ".conf", ".log", ".tex", ".bib", ".nf",
    ".pdf",
})

# File type icons mapping (extension -> icon name for frontend)
FILE_TYPE_ICONS: dict[str, str] = {
    ".py": "python",
    ".r": "r-lang",
    ".R": "r-lang",
    ".js": "javascript",
    ".ts": "typescript",
    ".json": "json",
    ".md": "markdown",
    ".csv": "spreadsheet",
    ".tsv": "spreadsheet",
    ".pdf": "pdf",
    ".html": "html",
    ".css": "css",
    ".sh": "terminal",
    ".yaml": "config",
    ".yml": "config",
    ".toml": "config",
    ".xml": "xml",
    ".tex": "latex",
}


@dataclass
class FileUploadConfig:
    """Configuration for file uploads."""

    max_file_size_bytes: int = 500_000  # 500 KB default
    max_batch_size: int = 10
    allowed_extensions: frozenset[str] = field(default_factory=lambda: RAG_ALLOWED_EXTENSIONS)

    def validate_file(self, filename: str, size_bytes: int) -> str | None:
        """
        Validate a file for upload.
        Returns None if valid, error message string if invalid.
        """
        dot_idx = filename.rfind(".")
        if dot_idx == -1:
            return "File has no extension"

        ext = filename[dot_idx:]
        if ext not in self.allowed_extensions and ext.lower() not in {
            e.lower() for e in self.allowed_extensions
        }:
            return f"Unsupported extension: {ext}"

        if size_bytes > self.max_file_size_bytes:
            size_kb = size_bytes / 1024
            max_kb = self.max_file_size_bytes / 1024
            return f"File too large: {size_kb:.0f}KB (max {max_kb:.0f}KB)"

        return None

    def validate_batch(self, file_count: int) -> str | None:
        """Validate batch size. Returns None if valid, error string if not."""
        if file_count > self.max_batch_size:
            return f"Too many files: {file_count} (max {self.max_batch_size})"
        if file_count < 1:
            return "No files provided"
        return None

    def get_file_icon(self, filename: str) -> str:
        """Get the icon name for a file based on its extension."""
        dot_idx = filename.rfind(".")
        if dot_idx == -1:
            return "file"
        ext = filename[dot_idx:]
        return FILE_TYPE_ICONS.get(ext, "file")

    def format_size(self, size_bytes: int) -> str:
        """Format file size for display."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        return f"{size_bytes / (1024 * 1024):.1f}MB"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_file_size_bytes": self.max_file_size_bytes,
            "max_batch_size": self.max_batch_size,
            "allowed_extensions": sorted(self.allowed_extensions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileUploadConfig:
        """Create from dictionary."""
        exts = data.get("allowed_extensions")
        return cls(
            max_file_size_bytes=data.get("max_file_size_bytes", 500_000),
            max_batch_size=data.get("max_batch_size", 10),
            allowed_extensions=frozenset(exts) if exts else RAG_ALLOWED_EXTENSIONS,
        )
