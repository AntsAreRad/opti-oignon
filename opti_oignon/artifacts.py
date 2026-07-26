#!/usr/bin/env python3
"""
ARTIFACTS -- OPTI-OIGNON 1.4.0 (F4)

Detect, store, and manage artifacts from LLM responses.

An artifact is a complete, self-contained file produced by the LLM
within a fenced code block (HTML pages, SVG graphics, Python/R scripts,
CSV data, markdown documents, etc.).

Architecture:
    - Artifact: dataclass for a detected artifact
    - ArtifactDetector: extracts artifacts from LLM response text
    - ArtifactManager: stores/retrieves artifacts per conversation

Author: Leon
"""

import logging
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Conversation manager (optional dependency)
try:
    from .conversation import conversation_manager
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False
    conversation_manager = None


# -- Artifact types and their detection patterns --

ARTIFACT_TYPES = {
    "html": {
        "extensions": [".html", ".htm"],
        "mime": "text/html",
        "display": "rendered",
    },
    "svg": {
        "extensions": [".svg"],
        "mime": "image/svg+xml",
        "display": "rendered",
    },
    "python": {
        "extensions": [".py"],
        "mime": "text/x-python",
        "display": "code",
    },
    "r": {
        "extensions": [".R", ".r"],
        "mime": "text/x-r",
        "display": "code",
    },
    "javascript": {
        "extensions": [".js"],
        "mime": "text/javascript",
        "display": "code",
    },
    "css": {
        "extensions": [".css"],
        "mime": "text/css",
        "display": "code",
    },
    "csv": {
        "extensions": [".csv"],
        "mime": "text/csv",
        "display": "table",
    },
    "json": {
        "extensions": [".json"],
        "mime": "application/json",
        "display": "code",
    },
    "yaml": {
        "extensions": [".yaml", ".yml"],
        "mime": "text/yaml",
        "display": "code",
    },
    "markdown": {
        "extensions": [".md"],
        "mime": "text/markdown",
        "display": "rendered",
    },
    "bash": {
        "extensions": [".sh"],
        "mime": "text/x-bash",
        "display": "code",
    },
    "sql": {
        "extensions": [".sql"],
        "mime": "text/x-sql",
        "display": "code",
    },
    "text": {
        "extensions": [".txt"],
        "mime": "text/plain",
        "display": "code",
    },
}

# Minimum line count for a code block to be considered an artifact
# (short snippets like `print("hello")` are not artifacts)
MIN_ARTIFACT_LINES = 5

# Language tag to artifact type mapping
_LANG_TO_TYPE = {
    "html": "html",
    "htm": "html",
    "svg": "svg",
    "python": "python",
    "python3": "python",
    "py": "python",
    "r": "r",
    "rlang": "r",
    "javascript": "javascript",
    "js": "javascript",
    "css": "css",
    "csv": "csv",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "markdown": "markdown",
    "md": "markdown",
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
    "sql": "sql",
    "text": "text",
    "txt": "text",
}

# Patterns that indicate a code block is a complete artifact (not a snippet)
_ARTIFACT_INDICATORS = {
    "html": [
        re.compile(r"<!DOCTYPE\s+html", re.IGNORECASE),
        re.compile(r"<html[\s>]", re.IGNORECASE),
    ],
    "svg": [
        re.compile(r"<svg[\s>]", re.IGNORECASE),
    ],
    "python": [
        re.compile(r"^(#!/usr/bin/env\s+python|#!/usr/bin/python)", re.MULTILINE),
        re.compile(r"^if\s+__name__\s*==\s*['\"]__main__['\"]", re.MULTILINE),
        re.compile(r"^(import|from)\s+\w+.*\n.*(def|class)\s+\w+", re.DOTALL),
    ],
    "r": [
        re.compile(r"^#!/usr/bin/env\s+Rscript", re.MULTILINE),
        re.compile(r"^library\(", re.MULTILINE),
    ],
    "csv": [
        # Any CSV-tagged block is likely a data artifact
        re.compile(r"^[\w\"'].+[,;|\t]", re.MULTILINE),
    ],
    "json": [
        re.compile(r"^\s*[\[{]", re.MULTILINE),
    ],
    "markdown": [
        re.compile(r"^#\s+", re.MULTILINE),
    ],
}

# Code block regex (same as in code_executor)
_CODE_BLOCK_RE = re.compile(
    r"```(\w*)\s*\n(.*?)```",
    re.DOTALL,
)


@dataclass
class Artifact:
    """A detected artifact from an LLM response."""
    id: str
    artifact_type: str
    title: str
    content: str
    language: str
    created_at: str
    conversation_id: str = ""
    display_mode: str = "code"
    line_count: int = 0
    # Versioning (Session 15 -- A2)
    version: int = 1
    parent_id: str = ""  # ID of the first version (empty = this is v1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Artifact':
        """Deserialize from dictionary."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })

    @property
    def file_extension(self) -> str:
        """Get the primary file extension for this artifact type."""
        type_info = ARTIFACT_TYPES.get(self.artifact_type, {})
        extensions = type_info.get("extensions", [])
        return extensions[0] if extensions else ".txt"

    @property
    def filename(self) -> str:
        """Generate a filename from the title."""
        safe_title = re.sub(r"[^\w\s-]", "", self.title).strip()
        safe_title = re.sub(r"\s+", "_", safe_title)
        if not safe_title:
            safe_title = f"artifact_{self.id[:8]}"
        return f"{safe_title}{self.file_extension}"


class ArtifactDetector:
    """Detect artifacts in LLM response text."""

    def __init__(self, min_lines: int = MIN_ARTIFACT_LINES):
        self.min_lines = min_lines

    def detect(
        self,
        response_text: str,
        conversation_id: str = "",
    ) -> list[Artifact]:
        """Extract artifacts from a response.

        Args:
            response_text: full LLM response text
            conversation_id: associated conversation ID

        Returns:
            List of detected Artifact objects
        """
        artifacts = []

        for match in _CODE_BLOCK_RE.finditer(response_text):
            lang_tag = match.group(1).strip().lower()
            content = match.group(2)

            if not content.strip():
                continue

            line_count = content.count("\n") + 1
            if line_count < self.min_lines:
                continue

            artifact_type = _LANG_TO_TYPE.get(lang_tag)
            if artifact_type is None:
                # Try content-based detection
                artifact_type = self._detect_type_from_content(content)

            if artifact_type is None:
                continue

            # Check if content looks like a complete artifact (not a snippet)
            if not self._is_complete_artifact(artifact_type, content, line_count):
                continue

            title = self._extract_title(artifact_type, content, lang_tag)
            type_info = ARTIFACT_TYPES.get(artifact_type, {})

            artifact = Artifact(
                id=str(uuid.uuid4())[:8],
                artifact_type=artifact_type,
                title=title,
                content=content,
                language=lang_tag or artifact_type,
                created_at=datetime.now().isoformat(),
                conversation_id=conversation_id,
                display_mode=type_info.get("display", "code"),
                line_count=line_count,
            )
            artifacts.append(artifact)

        return artifacts

    def _detect_type_from_content(self, content: str) -> str | None:
        """Infer artifact type from content when no language tag is given."""
        for art_type, patterns in _ARTIFACT_INDICATORS.items():
            for pattern in patterns:
                if pattern.search(content):
                    return art_type
        return None

    def _is_complete_artifact(
        self,
        artifact_type: str,
        content: str,
        line_count: int,
    ) -> bool:
        """Determine if a code block is a complete artifact vs a snippet.

        Heuristic: either matches a structural indicator, or is long enough.
        """
        # HTML/SVG are artifacts if they have the root tag
        indicators = _ARTIFACT_INDICATORS.get(artifact_type, [])
        for pattern in indicators:
            if pattern.search(content):
                return True

        # CSV and JSON are always considered artifacts if they pass min_lines
        if artifact_type in ("csv", "json", "yaml"):
            return True

        # For code types, require either structural patterns or significant length
        if artifact_type in ("python", "r", "bash", "javascript"):
            # Longer code blocks (20+ lines) are likely complete scripts
            if line_count >= 20:
                return True
            # Check for structural indicators
            for pattern in indicators:
                if pattern.search(content):
                    return True
            return False

        # Markdown: has headers -> artifact
        if artifact_type == "markdown":
            if re.search(r"^#\s+", content, re.MULTILINE):
                return True
            return line_count >= 10

        return line_count >= self.min_lines

    def _extract_title(
        self, artifact_type: str, content: str, lang_tag: str,
    ) -> str:
        """Extract a meaningful title from the artifact content."""
        # HTML: look for <title> tag
        if artifact_type == "html":
            m = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # SVG: look for <title> or first <text> element
        if artifact_type == "svg":
            m = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # Markdown: first heading
        if artifact_type == "markdown":
            m = re.search(r"^#+\s+(.+)", content, re.MULTILINE)
            if m:
                return m.group(1).strip()

        # Python: module docstring or first class/function
        if artifact_type == "python":
            m = re.search(r'^"""(.*?)"""', content, re.DOTALL)
            if m:
                first_line = m.group(1).strip().split("\n")[0]
                if first_line:
                    return first_line[:60]
            m = re.search(r"^(?:class|def)\s+(\w+)", content, re.MULTILINE)
            if m:
                return m.group(1)

        # R: first comment block
        if artifact_type == "r":
            m = re.search(r"^#\s*(.+)", content, re.MULTILINE)
            if m:
                return m.group(1).strip()[:60]

        # CSV: use first header fields
        if artifact_type == "csv":
            first_line = content.strip().split("\n")[0]
            return first_line[:50] + ("..." if len(first_line) > 50 else "")

        # Fallback: type-based generic title
        type_names = {
            "html": "HTML Page",
            "svg": "SVG Graphic",
            "python": "Python Script",
            "r": "R Script",
            "javascript": "JavaScript",
            "css": "Stylesheet",
            "csv": "Data (CSV)",
            "json": "JSON Data",
            "yaml": "YAML Config",
            "markdown": "Document",
            "bash": "Shell Script",
            "sql": "SQL Query",
        }
        return type_names.get(artifact_type, f"{lang_tag or artifact_type} file")


class ArtifactManager:
    """Store and retrieve artifacts per conversation."""

    def __init__(self):
        self._detector = ArtifactDetector()
        # In-memory cache: conv_id -> list of Artifact
        self._cache: dict[str, list[Artifact]] = {}

    @property
    def detector(self) -> ArtifactDetector:
        return self._detector

    # -- Versioning helpers (Session 15 -- A2) --

    @staticmethod
    def _title_similarity(a: str, b: str) -> float:
        """Compute simple word-overlap similarity between two titles.

        Returns a float in [0, 1]. 1.0 = identical word sets.
        """
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    def _find_version_parent(
        self,
        conversation_id: str,
        new_artifact: 'Artifact',
        similarity_threshold: float = 0.5,
    ) -> Optional['Artifact']:
        """Find the latest version of an existing artifact that matches.

        Matches by: same artifact_type + title similarity >= threshold.
        Returns the latest version (highest version number) of the chain,
        or None if no match.
        """
        existing = self._cache.get(conversation_id, [])
        if not existing:
            return None

        best_match = None
        best_sim = similarity_threshold

        for a in existing:
            if a.artifact_type != new_artifact.artifact_type:
                continue
            sim = self._title_similarity(a.title, new_artifact.title)
            if sim >= best_sim:
                # Prefer the latest version in the chain
                if best_match is None or a.version > best_match.version:
                    best_match = a
                    best_sim = sim

        return best_match

    def detect_and_store(
        self,
        response_text: str,
        conversation_id: str,
    ) -> list[Artifact]:
        """Detect artifacts in a response and store them.

        Automatically links new artifacts as versions of existing ones
        when type and title match (Session 15 -- A2).

        Args:
            response_text: LLM response text
            conversation_id: conversation UUID

        Returns:
            List of newly detected artifacts
        """
        artifacts = self._detector.detect(response_text, conversation_id)
        if not artifacts:
            return []

        # Add to in-memory cache with version linking
        if conversation_id not in self._cache:
            self._cache[conversation_id] = []

        for artifact in artifacts:
            parent = self._find_version_parent(conversation_id, artifact)
            if parent is not None:
                # Link as a new version. Number from the
                # chain maximum, not the matched member -- similarity
                # ordering can match an older member of the chain and
                # would otherwise produce duplicate version numbers.
                root_id = parent.parent_id or parent.id
                chain_max = max(
                    (
                        a.version
                        for a in self._cache[conversation_id]
                        if a.id == root_id or a.parent_id == root_id
                    ),
                    default=parent.version,
                )
                artifact.parent_id = root_id
                artifact.version = chain_max + 1
            self._cache[conversation_id].append(artifact)

        # Persist to conversation metadata
        self._save_to_metadata(conversation_id)

        return artifacts

    def get_artifacts(self, conversation_id: str) -> list[Artifact]:
        """Get all artifacts for a conversation.

        Loads from metadata if not in cache.
        """
        if conversation_id in self._cache:
            return self._cache[conversation_id]

        # Try loading from conversation metadata
        loaded = self._load_from_metadata(conversation_id)
        self._cache[conversation_id] = loaded
        return loaded

    def get_artifact_by_id(
        self, conversation_id: str, artifact_id: str,
    ) -> Artifact | None:
        """Get a specific artifact by ID."""
        for a in self.get_artifacts(conversation_id):
            if a.id == artifact_id:
                return a
        return None

    # -- Version queries (Session 15 -- A2) --

    def get_version_history(
        self, conversation_id: str, artifact_id: str,
    ) -> list[Artifact]:
        """Get all versions of an artifact chain, sorted by version number.

        Works whether artifact_id is the root or any version in the chain.
        """
        target = self.get_artifact_by_id(conversation_id, artifact_id)
        if target is None:
            return []

        # Determiner l'ID racine de la chaine
        root_id = target.parent_id or target.id

        # Collecter toutes les versions de la chaine
        chain = []
        for a in self.get_artifacts(conversation_id):
            if a.id == root_id or a.parent_id == root_id:
                chain.append(a)

        chain.sort(key=lambda a: a.version)
        return chain

    def get_latest_version(
        self, conversation_id: str, artifact_id: str,
    ) -> Artifact | None:
        """Get the latest version in an artifact's version chain."""
        chain = self.get_version_history(conversation_id, artifact_id)
        return chain[-1] if chain else None

    def delete_artifact(
        self, conversation_id: str, artifact_id: str,
    ) -> bool:
        """Delete a specific artifact."""
        artifacts = self.get_artifacts(conversation_id)
        original_len = len(artifacts)
        self._cache[conversation_id] = [
            a for a in artifacts if a.id != artifact_id
        ]
        if len(self._cache[conversation_id]) < original_len:
            self._save_to_metadata(conversation_id)
            return True
        return False

    def clear_artifacts(self, conversation_id: str) -> int:
        """Remove all artifacts for a conversation. Returns count removed."""
        count = len(self.get_artifacts(conversation_id))
        self._cache[conversation_id] = []
        self._save_to_metadata(conversation_id)
        return count

    def get_conversation_ids(self) -> list[str]:
        """Get all conversation IDs that have cached artifacts."""
        return list(self._cache.keys())

    def _save_to_metadata(self, conversation_id: str):
        """Persist artifacts to conversation metadata."""
        if not CONVERSATION_AVAILABLE or conversation_manager is None:
            return
        if not conversation_id:
            return

        artifacts = self._cache.get(conversation_id, [])
        artifacts_data = [a.to_dict() for a in artifacts]

        try:
            conversation_manager.update_conversation_metadata(
                conversation_id,
                metadata={"artifacts": artifacts_data},
            )
        except Exception as e:
            logger.debug(f"Could not save artifacts to metadata: {e}")

    def _load_from_metadata(self, conversation_id: str) -> list[Artifact]:
        """Load artifacts from conversation metadata."""
        if not CONVERSATION_AVAILABLE or conversation_manager is None:
            return []
        if not conversation_id:
            return []

        try:
            conv = conversation_manager.get_conversation(conversation_id)
            if conv is None:
                return []
            artifacts_data = conv.metadata.get("artifacts", [])
            return [Artifact.from_dict(d) for d in artifacts_data]
        except Exception as e:
            logger.debug(f"Could not load artifacts from metadata: {e}")
            return []

    def export_artifacts(
        self, conversation_id: str,
    ) -> list[dict[str, str]]:
        """Export artifacts as a list of {filename, content} dicts."""
        return [
            {"filename": a.filename, "content": a.content}
            for a in self.get_artifacts(conversation_id)
        ]

    def export_single_to_file(
        self,
        conversation_id: str,
        artifact_id: str,
        output_dir: str | None = None,
    ) -> str | None:
        """Save a single artifact to disk.

        Args:
            conversation_id: conversation UUID
            artifact_id: artifact ID
            output_dir: target directory (default: tempdir)

        Returns:
            Path to the saved file, or None if not found.
        """
        artifact = self.get_artifact_by_id(conversation_id, artifact_id)
        if artifact is None:
            return None

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="opti_artifact_")
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, os.path.basename(artifact.filename))
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(artifact.content)
        return filepath

    def export_all_to_dir(
        self,
        conversation_id: str,
        output_dir: str | None = None,
    ) -> list[str]:
        """Save all artifacts for a conversation to a directory.

        Returns:
            List of saved file paths.
        """
        artifacts = self.get_artifacts(conversation_id)
        if not artifacts:
            return []

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="opti_artifacts_")
        os.makedirs(output_dir, exist_ok=True)

        paths = []
        seen_names = set()
        for a in artifacts:
            # Sanitize filename to prevent path traversal
            fname = os.path.basename(a.filename)
            # Handle duplicate filenames
            if fname in seen_names:
                base, ext = os.path.splitext(fname)
                fname = f"{base}_{a.id}{ext}"
            seen_names.add(fname)

            filepath = os.path.join(output_dir, fname)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(a.content)
            paths.append(filepath)
        return paths

    def export_as_zip(
        self,
        conversation_id: str,
        zip_path: str | None = None,
    ) -> str | None:
        """Export all artifacts as a ZIP archive.

        Args:
            conversation_id: conversation UUID
            zip_path: output zip path (default: auto-generated in tempdir)

        Returns:
            Path to the zip file, or None if no artifacts.
        """
        import zipfile

        artifacts = self.get_artifacts(conversation_id)
        if not artifacts:
            return None

        if zip_path is None:
            tmpdir = tempfile.mkdtemp(prefix="opti_artifact_zip_")
            zip_path = os.path.join(
                tmpdir,
                f"artifacts_{conversation_id[:8]}.zip",
            )

        seen_names = set()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for a in artifacts:
                fname = a.filename
                if fname in seen_names:
                    base, ext = os.path.splitext(fname)
                    fname = f"{base}_{a.id}{ext}"
                seen_names.add(fname)
                zf.writestr(fname, a.content)

        return zip_path


# Module-level singleton
artifact_manager = ArtifactManager()
