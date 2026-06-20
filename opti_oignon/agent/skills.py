#!/usr/bin/env python3
"""The on-disk SKILL.md registry (S177, Theme 3 / Odysseus Core).

The evolving-skills half of the agent (ODYSSEUS_SPEC.md Section 6). Skills are
plain SKILL.md files on disk, each with YAML-style frontmatter and a structured
body (When to Use / Procedure / Pitfalls / Verification), so a procedure the
agent learns once can be consulted before later domain work. Two layers live
here:

- ``SkillRegistry`` -- the on-disk store. Published skills live at
  ``<root>/<category>/<name>/SKILL.md``; a usage counter lives in a
  ``_usage.json`` sidecar next to the skill, so an unchanged SKILL.md is never
  rewritten just to bump a count; old versions are archived under
  ``<root>/<category>/<name>/.versions/`` for audit; and drafts (proposals
  awaiting human approval) are kept apart under ``<root>/.drafts/`` so a draft
  never shadows a live skill. The registry is read / list / search / view /
  view_ref plus the raw CRUD the gated tool drives. It is local and
  per-instance: there is no cross-instance marketplace and nothing here reaches
  the network.

- The approval-gated ``manage_skills`` tool and the teacher-draft publish path
  (added on top of this registry in S177's later phases) gate every write
  behind explicit human approval.

Filesystem hygiene: every path segment for a category or a name is sanitised to
a strict ``[a-z0-9_-]`` slug and the resolved path is verified to stay under the
root, so a traversal payload (``..`` / ``/`` / an absolute path) can never
escape the registry. No f-string SQL is involved -- this is a file store.

Veilid sync (SYN-01, S201, Bloc 0 lot 3): the registry's write seams publish
to the change feed AFTER the domain commit -- for this file store the commit
is the completed file write -- through ``_sync_publish_skill``. Every
published-tree write (``_write`` with ``draft=False``) journals the new full
state; a published ``delete`` journals a tombstone. Drafts are device-local
and journal nothing (only the human-approved published tree syncs; syncing a
draft would propagate unapproved executable surface to peers, against the
``RecordKind.SKILL`` posture in ``veilid/producers.py``). The ``_usage.json``
sidecar is device-local telemetry and journals nothing; usage numbers never
ride a payload. Best-effort and mode-free: a journalling failure never breaks
the write, and only the wire is Daily-gated downstream at the engine/guard.

Importlib-isolatable: the default root is resolved from ``config.DATA_DIR``
lazily and guarded (falling back to a per-user data directory), and the audit
hook imports ``signed_audit_log`` lazily, so this module loads and is exercised
with a temporary registry root and without the backend. The module-level
registry has a ``reset_skill_registry()`` and an injectable root for tests.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# Module conventions (Theme 3).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Skill status: a draft is a proposal awaiting human approval; published is live.
STATUS_DRAFT = "draft"
STATUS_PUBLISHED = "published"

# Provenance of a skill.
SOURCE_MANUAL = "manual"
SOURCE_AGENT = "agent"
SOURCE_TEACHER = "teacher-escalation"

# The structured body sections, in order.
SECTION_WHEN = "When to Use"
SECTION_PROCEDURE = "Procedure"
SECTION_PITFALLS = "Pitfalls"
SECTION_VERIFICATION = "Verification"
BODY_SECTIONS: tuple[str, ...] = (
    SECTION_WHEN,
    SECTION_PROCEDURE,
    SECTION_PITFALLS,
    SECTION_VERIFICATION,
)

# On-disk names.
SKILL_FILENAME = "SKILL.md"
USAGE_FILENAME = "_usage.json"
VERSIONS_DIR = ".versions"
DRAFTS_DIR = ".drafts"

# Directory names never treated as a category when scanning the registry.
_RESERVED_DIRS = frozenset({VERSIONS_DIR, DRAFTS_DIR, "__pycache__"})

# Filesystem hygiene: only these characters survive a path segment.
_SEGMENT_RE = re.compile(r"[^a-z0-9_-]+")
# A word tokeniser for relevance scoring.
_WORD_RE = re.compile(r"[a-z0-9]+")

# Frontmatter fields written, in a stable order.
_META_FIELDS = ("name", "category", "status", "version", "source", "created_at", "updated_at")


# Filesystem hygiene


def _safe_segment(value: Any, fallback: str) -> str:
    """Sanitise a category or name into a strict slug, rejecting traversal.

    Lower-cases, collapses any disallowed run to a single dash, and strips
    leading / trailing dashes. A ``..`` or ``/`` payload cannot survive because
    only ``[a-z0-9_-]`` is kept; an empty result falls back to ``fallback``.
    """
    cleaned = _SEGMENT_RE.sub("-", str(value or "").strip().lower()).strip("-")
    return cleaned or fallback


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Frontmatter parse / serialise (dependency-free; no PyYAML requirement)


def _serialise_frontmatter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for key in _META_FIELDS:
        if key in meta and meta[key] is not None:
            lines.append(f"{key}: {meta[key]}")
    lines.append("---")
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split SKILL.md text into a frontmatter mapping and the body.

    Accepts a leading ``---`` ... ``---`` block of ``key: value`` lines. Text
    with no frontmatter yields an empty mapping and the whole text as the body.
    Tolerant: a malformed block is treated as body.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    body_start = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            body_start = i + 1
            break
        if ":" in lines[i]:
            key, _, value = lines[i].partition(":")
            meta[key.strip()] = value.strip()
    if body_start is None:
        return {}, text
    body = "\n".join(lines[body_start:]).lstrip("\n")
    return meta, body


def _extract_section(body: str, header: str) -> str:
    """Return the text of one structured section, or an empty string.

    A section header is a line that, with leading ``#`` and surrounding
    whitespace and a trailing colon removed, equals the section name
    (case-insensitively). The section runs until the next header line or EOF.
    """
    pattern = re.compile(
        r"^#{0,6}\s*" + re.escape(header) + r"\s*:?\s*$", re.IGNORECASE | re.MULTILINE
    )
    m = pattern.search(body)
    if not m:
        return ""
    start = m.end()
    nxt = re.compile(r"^#{1,6}\s+\S", re.MULTILINE).search(body, start)
    end = nxt.start() if nxt else len(body)
    return body[start:end].strip()


# Skill record


@dataclass
class Skill:
    """One skill: its frontmatter metadata plus the structured body."""

    name: str
    category: str
    status: str = STATUS_PUBLISHED
    version: int = 1
    source: str = SOURCE_MANUAL
    body: str = ""
    created_at: str = ""
    updated_at: str = ""

    def meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "version": self.version,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_markdown(self) -> str:
        """The on-disk representation: frontmatter then body."""
        return _serialise_frontmatter(self.meta()) + "\n\n" + self.body.strip() + "\n"

    def reference(self) -> str:
        """A compact reference: identity plus the When to Use trigger only.

        This is what the agent consults to decide whether a skill applies,
        without pulling the full procedure into the prompt.
        """
        when = _extract_section(self.body, SECTION_WHEN)
        head = f"{self.name} ({self.category}) v{self.version} [{self.status}]"
        return head if not when else f"{head}\nWhen to Use: {when}"

    def to_dict(self) -> dict[str, Any]:
        """Metadata plus a short summary, for the index / the UI (no full body)."""
        d = self.meta()
        d["summary"] = _extract_section(self.body, SECTION_WHEN)[:280]
        return d


@dataclass
class ScoredSkill:
    """A skill paired with its relevance score from a search."""

    skill: Skill
    score: float

    def to_dict(self) -> dict[str, Any]:
        d = self.skill.to_dict()
        d["score"] = round(self.score, 4)
        return d


@dataclass
class SkillUsage:
    """The usage sidecar for one skill."""

    uses: int = 0
    last_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"uses": self.uses, "last_used": self.last_used}


# Default root resolution (lazy / guarded)


def _default_root() -> Path:
    """The default registry root, ``config.DATA_DIR / 'skills'``, guarded.

    Falls back to a per-user data directory when the backend config cannot be
    imported, so the registry never fails to resolve a root.
    """
    try:
        from opti_oignon.config import DATA_DIR

        return Path(DATA_DIR) / "skills"
    except Exception:  # pragma: no cover - constrained environments only
        return Path.home() / ".opti-oignon" / "data" / "skills"


# Audit hook (lazy / guarded): skill mutations join the hash-chain audit log


def _audit(action: str, **details: Any) -> None:
    """Record a skill mutation in the hash-chain audit log, best-effort.

    Defense in depth behind the approval gate (which already chain-logs the
    decision). Lazy and guarded so this never raises and stays isolatable.
    """
    try:
        from opti_oignon.signed_audit_log import chain_log

        chain_log(
            event_type="skill_mutation",
            source="agent.skills",
            action=action,
            severity="INFO",
            **details,
        )
    except Exception:  # pragma: no cover - audit is best-effort
        logger.debug("skill audit log unavailable", exc_info=True)


# Veilid sync publish hook (SYN-01, S201, Bloc 0 lot 3)

# The lot-3 adaptation of the lot-1/2 lock-order convention: SkillRegistry is
# a file store and holds no lock of its own, so this module-level RLock is
# what serialises mint + append per process (same-key clocks stay strictly
# monotonic). Lock order is hook lock -> feed lock; the feed never calls back
# into domain code, so the order is acyclic. The registry's own file writes
# remain unserialised (pre-existing behaviour, out of this lot's scope; every
# production write is human-gated, so same-key write concurrency is
# practically nil) -- under a hypothetical same-key race the journal's latest
# state could trail the disk briefly and converges on the next write
# (at-least-once).
_SYNC_LOCK = threading.RLock()


def _sync_owner_id() -> str:
    """The owning user for a sync payload (the ``effective_user_id`` pattern).

    Skills are a per-instance registry with no per-user scoping, so this
    resolves to the single-user default. Scoping rides in the PAYLOAD, never
    the key: the ``category/name`` slug join stays the stable per-kind key on
    every device (the lot-1 rule, carried by lot 2 the same way).
    """
    try:
        from opti_oignon.user_isolation import effective_user_id

        return effective_user_id(None)
    except Exception:  # pragma: no cover - isolation module is optional here
        return "local"


def _skill_sync_key(category: str, name: str) -> str:
    """The stable per-kind key for a skill: the ``category/name`` slug join.

    This is the path identity. ``_safe_segment`` keeps only ``[a-z0-9_-]``,
    so the ``/`` separator can never appear inside a segment, and the
    fallbacks guarantee non-empty segments -- the join is therefore injective
    on slug pairs (no two distinct category/name pairs share a key). Raw
    inputs that sanitise to the same pair are the SAME skill on disk, so the
    key follows storage identity. Idempotent on already-sanitised slugs.
    """
    return f"{_safe_segment(category, 'general')}/{_safe_segment(name, 'untitled-skill')}"


def _skill_payload(skill: Skill) -> dict[str, Any]:
    """Full-state payload for a published skill (state-based LWW).

    ``user_id`` is hoisted to the top level (the lot-1 scoping rule); the
    nested skill carries the frontmatter metadata plus ``markdown`` -- the
    EXACT text ``_write`` puts on disk (``to_markdown``), so a receiver can
    rebuild the file byte-faithfully by writing the field verbatim, without
    re-serialising frontmatter. Excluded as device-local: the ``_usage.json``
    counters, the ``.versions/`` archives, absolute paths, and the derived
    ``summary`` of ``to_dict``.
    """
    nested = skill.meta()
    nested["markdown"] = skill.to_markdown()
    return {"user_id": _sync_owner_id(), "skill": nested}


def _sync_publish_skill(
    skill_id: str,
    payload_fn: Callable[[], dict[str, Any] | None] | None = None,
    *,
    deleted: bool = False,
    updated_at: str = "",
) -> None:
    """Journal a skill change for Veilid sync, best-effort (SYN-01).

    Called by the registry's write seams AFTER the domain commit -- for this
    file store the commit is the completed file write (``_write``) or unlink
    (``delete``). ``payload_fn`` is a zero-arg callable building the
    full-state payload; it runs INSIDE this hook's protection, and only after
    the availability probe passes, so when sync is absent the write pays
    nothing (no payload build, no journal append). The contract
    (ROADMAP_SYNC_CYCLE, Bloc 0, the lot-1/2 precedents):

    - A payload or journalling failure must never break the write: any error
      is logged and swallowed (at-least-once on the next write).
    - No-op when the optional veilid framework is absent
      (``guard.veilid_available`` is the cheap probe).
    - Mode-free: producing and journalling are local-disk operations
      permitted in ANY mode (the documented ``producers.py`` posture); only
      the wire is Daily-gated, downstream at the engine/guard.
    - Drafts never reach this hook (the ``draft`` guard sits at the call
      sites): a draft is device-local until the human-approved ``publish``.
    - Applying a RECEIVED skill is the sensitive action, gated at the engine
      (``sync_engine.SENSITIVE_KINDS`` / ``_gate_records``); producing one
      locally is not.

    Clock discipline: next = the highest clock journalled for the key, plus
    one (an unseen key yields 0, so the first clock is 1). ``_SYNC_LOCK``
    serialises mint + append per process (see its note: the lot-3 adaptation
    for a lockless file store).
    """
    try:
        with _SYNC_LOCK:
            from opti_oignon.veilid.guard import veilid_available

            if not veilid_available():
                return
            payload: dict[str, Any] | None = None
            if not deleted:
                payload = payload_fn() if payload_fn is not None else None
                if payload is None:
                    # The state could not be built. Publishing an empty
                    # non-tombstone payload would wipe the skill on peers
                    # under LWW -- skip instead.
                    logger.debug(
                        "sync publish skipped for skill %s: no state available",
                        skill_id,
                    )
                    return
            from opti_oignon.veilid.records import RecordKind
            from opti_oignon.veilid.sync_engine import get_sync_engine

            engine = get_sync_engine()
            clock = engine.current_clock(RecordKind.SKILL, skill_id) + 1
            engine.publish_skill(
                skill_id,
                payload,
                clock=clock,
                deleted=deleted,
                updated_at=updated_at,
            )
    except Exception:
        logger.warning(
            "veilid sync publish failed for skill %s (write unaffected)",
            skill_id,
            exc_info=True,
        )


# The registry


class SkillRegistry:
    """The on-disk SKILL.md store.

    The root is injectable (tests pass a temporary directory). Published skills
    live at ``<root>/<category>/<name>/SKILL.md``; drafts under
    ``<root>/.drafts/<category>/<name>/SKILL.md``; old published versions under
    ``<root>/<category>/<name>/.versions/v<N>.md``; the usage counter in a
    ``_usage.json`` sidecar next to the published skill.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_root()
        self._drafts_root = self.root / DRAFTS_DIR

    # Path helpers (all traversal-safe)

    def _skill_dir(self, category: str, name: str, *, draft: bool) -> Path:
        cat = _safe_segment(category, "general")
        nm = _safe_segment(name, "untitled-skill")
        base = self._drafts_root if draft else self.root
        return (base / cat / nm).resolve()

    def _within_root(self, path: Path) -> bool:
        try:
            root = self.root.resolve()
            path.resolve().relative_to(root)
            return True
        except Exception:
            return False

    def _skill_path(self, category: str, name: str, *, draft: bool) -> Path | None:
        d = self._skill_dir(category, name, draft=draft)
        if not self._within_root(d):
            return None
        return d / SKILL_FILENAME

    # Read side

    def _read_path(self, path: Path) -> Skill | None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None
        meta, body = _parse_frontmatter(text)
        category = path.parent.parent.name
        name = path.parent.name
        try:
            version = int(meta.get("version", "1"))
        except Exception:
            version = 1
        return Skill(
            name=meta.get("name", name),
            category=meta.get("category", category),
            status=meta.get("status", STATUS_PUBLISHED),
            version=version,
            source=meta.get("source", SOURCE_MANUAL),
            body=body,
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
        )

    def get(self, name: str, category: str, *, draft: bool = False) -> Skill | None:
        """Read one skill (published by default, or a draft), or None."""
        path = self._skill_path(category, name, draft=draft)
        if path is None or not path.is_file():
            return None
        return self._read_path(path)

    def exists(self, name: str, category: str, *, draft: bool = False) -> bool:
        path = self._skill_path(category, name, draft=draft)
        return path is not None and path.is_file()

    def _iter_skill_files(self, base: Path) -> Iterator[Path]:
        """Yield every ``<base>/<category>/<name>/SKILL.md`` under a base dir."""
        if not base.is_dir():
            return
        for cat_dir in sorted(base.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name in _RESERVED_DIRS:
                continue
            if cat_dir.name.startswith(".") or cat_dir.name.startswith("_"):
                continue
            for name_dir in sorted(cat_dir.iterdir()):
                if not name_dir.is_dir() or name_dir.name in _RESERVED_DIRS:
                    continue
                skill_file = name_dir / SKILL_FILENAME
                if skill_file.is_file():
                    yield skill_file

    def list(
        self, *, include_drafts: bool = False, category: str | None = None
    ) -> list[Skill]:
        """List skills, published by default, optionally filtered by category.

        With ``include_drafts`` the draft proposals are appended after the
        published skills.
        """
        skills: list[Skill] = []
        for path in self._iter_skill_files(self.root):
            skill = self._read_path(path)
            if skill is not None:
                skills.append(skill)
        if include_drafts:
            for path in self._iter_skill_files(self._drafts_root):
                skill = self._read_path(path)
                if skill is not None:
                    skill.status = STATUS_DRAFT
                    skills.append(skill)
        if category:
            cat = _safe_segment(category, "")
            skills = [s for s in skills if _safe_segment(s.category, "") == cat]
        return skills

    def search(
        self, query: str, *, limit: int = 5, include_drafts: bool = False
    ) -> list[ScoredSkill]:
        """Rank skills by keyword relevance to a query.

        Tokens shared with a skill's name or category weigh more than tokens in
        the body. Skills with no overlap are dropped. Ties break by version then
        name for a stable order.
        """
        terms = set(_WORD_RE.findall((query or "").lower()))
        if not terms:
            return []
        scored: list[ScoredSkill] = []
        for skill in self.list(include_drafts=include_drafts):
            name_tokens = set(_WORD_RE.findall(skill.name.lower()))
            cat_tokens = set(_WORD_RE.findall(skill.category.lower()))
            body_tokens = set(_WORD_RE.findall(skill.body.lower()))
            score = (
                3.0 * len(terms & name_tokens)
                + 2.0 * len(terms & cat_tokens)
                + 1.0 * len(terms & body_tokens)
            )
            if score > 0:
                scored.append(ScoredSkill(skill=skill, score=score))
        scored.sort(key=lambda s: (-s.score, -s.skill.version, s.skill.name))
        return scored[: max(0, int(limit))]

    def relevant(self, query: str, *, limit: int = 3) -> list[Skill]:
        """The top published skills most relevant to a query (for planning)."""
        return [s.skill for s in self.search(query, limit=limit, include_drafts=False)]

    def view(self, name: str, category: str, *, draft: bool = False) -> str:
        """The full SKILL.md text of a skill, or an empty string when absent."""
        skill = self.get(name, category, draft=draft)
        return skill.to_markdown() if skill is not None else ""

    def view_ref(self, name: str, category: str, *, draft: bool = False) -> str:
        """A compact reference for a skill, or an empty string when absent."""
        skill = self.get(name, category, draft=draft)
        return skill.reference() if skill is not None else ""

    def index(self) -> dict[str, list[dict[str, Any]]]:
        """An index of published skills plus draft proposals (metadata only)."""
        published = [s.to_dict() for s in self.list(include_drafts=False)]
        drafts = [
            s.to_dict()
            for s in self.list(include_drafts=True)
            if s.status == STATUS_DRAFT
        ]
        return {"published": published, "drafts": drafts}

    # Write side (raw; the human-approval gate wraps these in the tool)

    def _archive(self, category: str, name: str, skill: Skill) -> None:
        """Archive a published skill's current text under .versions for audit."""
        d = self._skill_dir(category, name, draft=False)
        versions = d / VERSIONS_DIR
        try:
            versions.mkdir(parents=True, exist_ok=True)
            (versions / f"v{skill.version}.md").write_text(
                skill.to_markdown(), encoding="utf-8"
            )
        except Exception:  # pragma: no cover - archival is best-effort
            logger.debug("skill version archive failed", exc_info=True)

    def _write(self, skill: Skill, *, draft: bool) -> Skill:
        path = self._skill_path(skill.category, skill.name, draft=draft)
        if path is None:
            raise ValueError("refusing to write skill outside the registry root")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(skill.to_markdown(), encoding="utf-8")
        if not draft:
            # SYN-01 (S201): the completed file write IS the domain commit
            # for this file store; publish the new full state after it. The
            # payload closes over the skill already in hand (zero extra
            # reads) and is built inside the hook's guard. Draft writes are
            # device-local and journal nothing.
            _sync_publish_skill(
                _skill_sync_key(skill.category, skill.name),
                lambda: _skill_payload(skill),
                updated_at=skill.updated_at,
            )
        return skill

    def add(
        self,
        name: str,
        category: str,
        body: str,
        *,
        source: str = SOURCE_AGENT,
        status: str = STATUS_DRAFT,
    ) -> Skill:
        """Create a skill record (a draft proposal by default, or published).

        A new published skill starts at version 1; re-adding over an existing
        published skill archives the old version and bumps the version. A draft
        is written to the drafts area and carries the version it would become.
        """
        cat = _safe_segment(category, "general")
        nm = _safe_segment(name, "untitled-skill")
        now = _now()
        existing_pub = self.get(nm, cat, draft=False)
        if status == STATUS_DRAFT:
            version = (existing_pub.version + 1) if existing_pub else 1
            skill = Skill(
                name=nm,
                category=cat,
                status=STATUS_DRAFT,
                version=version,
                source=source,
                body=body,
                created_at=now,
                updated_at=now,
            )
            self._write(skill, draft=True)
            _audit("draft_add", name=nm, category=cat, version=version, source=source)
            return skill
        # Published.
        if existing_pub is not None:
            self._archive(cat, nm, existing_pub)
            version = existing_pub.version + 1
            created = existing_pub.created_at or now
        else:
            version = 1
            created = now
        skill = Skill(
            name=nm,
            category=cat,
            status=STATUS_PUBLISHED,
            version=version,
            source=source,
            body=body,
            created_at=created,
            updated_at=now,
        )
        self._write(skill, draft=False)
        _audit("publish", name=nm, category=cat, version=version, source=source)
        return skill

    def update(
        self,
        name: str,
        category: str,
        *,
        body: str | None = None,
        source: str | None = None,
    ) -> Skill | None:
        """Edit a published skill: archive the old version, write a new one.

        Returns None when no published skill exists for the name / category.
        """
        existing = self.get(name, category, draft=False)
        if existing is None:
            return None
        cat = _safe_segment(category, "general")
        nm = _safe_segment(name, "untitled-skill")
        self._archive(cat, nm, existing)
        new = Skill(
            name=nm,
            category=cat,
            status=STATUS_PUBLISHED,
            version=existing.version + 1,
            source=source or existing.source,
            body=existing.body if body is None else body,
            created_at=existing.created_at or _now(),
            updated_at=_now(),
        )
        self._write(new, draft=False)
        _audit("edit", name=nm, category=cat, version=new.version)
        return new

    def patch(
        self, name: str, category: str, old_str: str, new_str: str = ""
    ) -> Skill | None:
        """Find-and-replace a unique string in a published skill's body.

        Mirrors the str_replace contract: the search string must occur exactly
        once. Returns None when the skill is missing or the match is not unique.
        """
        existing = self.get(name, category, draft=False)
        if existing is None:
            return None
        if not old_str or existing.body.count(old_str) != 1:
            return None
        return self.update(name, category, body=existing.body.replace(old_str, new_str, 1))

    def publish(self, name: str, category: str) -> Skill | None:
        """Promote a draft to published, archiving any existing published skill.

        The draft file is removed once it is published. Returns None when no
        draft exists for the name / category.
        """
        draft = self.get(name, category, draft=True)
        if draft is None:
            return None
        cat = _safe_segment(category, "general")
        nm = _safe_segment(name, "untitled-skill")
        existing_pub = self.get(nm, cat, draft=False)
        if existing_pub is not None:
            self._archive(cat, nm, existing_pub)
            version = existing_pub.version + 1
            created = existing_pub.created_at or _now()
        else:
            version = max(1, draft.version)
            created = draft.created_at or _now()
        published = Skill(
            name=nm,
            category=cat,
            status=STATUS_PUBLISHED,
            version=version,
            source=draft.source,
            body=draft.body,
            created_at=created,
            updated_at=_now(),
        )
        self._write(published, draft=False)
        self.delete(nm, cat, draft=True)
        _audit("publish", name=nm, category=cat, version=version, source=draft.source)
        return published

    def delete(self, name: str, category: str, *, draft: bool = False) -> bool:
        """Remove a skill. A published skill's final version is archived first;
        its ``.versions`` history is retained for audit. Returns False when the
        target does not exist."""
        existing = self.get(name, category, draft=draft)
        if existing is None:
            return False
        cat = _safe_segment(category, "general")
        nm = _safe_segment(name, "untitled-skill")
        if not draft:
            self._archive(cat, nm, existing)
        path = self._skill_path(category, name, draft=draft)
        if path is None:
            return False
        try:
            path.unlink(missing_ok=True)
            usage = path.parent / USAGE_FILENAME
            usage.unlink(missing_ok=True)
        except Exception:  # pragma: no cover - defensive
            return False
        _audit("delete", name=nm, category=cat, draft=draft)
        if not draft:
            # SYN-01 (S201): a published deletion is the converged tombstone.
            # A draft deletion -- including publish()'s internal cleanup of
            # the promoted draft -- is device-local and journals nothing.
            _sync_publish_skill(
                _skill_sync_key(cat, nm), deleted=True, updated_at=_now()
            )
        return True

    # Usage sidecar (written separately; never rewrites SKILL.md)

    def _usage_path(self, name: str, category: str) -> Path | None:
        d = self._skill_dir(category, name, draft=False)
        if not self._within_root(d):
            return None
        return d / USAGE_FILENAME

    def get_usage(self, name: str, category: str) -> SkillUsage:
        path = self._usage_path(name, category)
        if path is None or not path.is_file():
            return SkillUsage()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SkillUsage(
                uses=int(data.get("uses", 0)), last_used=str(data.get("last_used", ""))
            )
        except Exception:
            return SkillUsage()

    def increment_usage(self, name: str, category: str) -> SkillUsage:
        """Bump a skill's use counter in the sidecar only.

        The SKILL.md file is never touched, so consuming a skill does not
        rewrite it. A no-op when the published skill is absent.
        """
        if not self.exists(name, category, draft=False):
            return SkillUsage()
        path = self._usage_path(name, category)
        if path is None:
            return SkillUsage()
        usage = self.get_usage(name, category)
        usage.uses += 1
        usage.last_used = _now()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(usage.to_dict(), sort_keys=True), encoding="utf-8")
        except Exception:  # pragma: no cover - defensive
            logger.debug("skill usage write failed", exc_info=True)
        return usage


# Coercion helpers (defensive: the tool handler must never raise)


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# Verification sandbox-testing (the S73/S74 seam; refuse when bwrap is absent)

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _sandbox_ready(session: Any) -> bool:
    """Whether an injected sandbox session is backed by an available bwrap.

    Mirrors ``dispatch.sandbox_ready`` without importing ``dispatch``, so this
    module stays decoupled and isolatable. A missing session, a missing
    manager, or an unavailable bwrap all return False.
    """
    if session is None:
        return False
    mgr = getattr(session, "sandbox_manager", None)
    if mgr is None:
        return False
    return bool(getattr(mgr, "bwrap_available", False))


def _verification_commands(body: str) -> list[str]:
    """Fenced command blocks in the Verification section (the executable steps)."""
    section = _extract_section(body, SECTION_VERIFICATION)
    if not section:
        return []
    return [m.group(1).strip() for m in _FENCE_RE.finditer(section) if m.group(1).strip()]


@dataclass
class VerificationResult:
    """The outcome of sandbox-testing a skill's verification steps."""

    ok: bool
    tested: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "tested": self.tested, "detail": self.detail}


def sandbox_test_verification(body: str, sandbox: Any) -> VerificationResult:
    """Sandbox-test a skill body's verification steps, where present.

    A body with no fenced verification commands needs no sandbox and passes
    (``tested`` False). When commands are present they run ONLY in the
    disposable bwrap sandbox via the injected session; if bwrap is unavailable
    the operation is refused (``ok`` False), never run on the host. Never
    raises -- a failing step becomes a result, not an exception.
    """
    cmds = _verification_commands(body)
    if not cmds:
        return VerificationResult(ok=True, tested=False, detail="no executable verification steps")
    if not _sandbox_ready(sandbox) or not bool(getattr(sandbox, "active", False)):
        return VerificationResult(
            ok=False,
            tested=False,
            detail=(
                "the disposable bwrap sandbox is unavailable; refusing to act on a "
                "skill that carries verification steps"
            ),
        )
    outputs: list[str] = []
    for cmd in cmds:
        try:
            out = sandbox.bash(cmd)
        except Exception as exc:
            return VerificationResult(ok=False, tested=True, detail=f"verification step failed: {exc}")
        outputs.append(_as_str(out))
    return VerificationResult(ok=True, tested=True, detail="; ".join(outputs)[:2000])


# The approval-gated manage_skills tool

_READ_ACTIONS = frozenset({"list", "index", "view", "view_ref", "search"})
_WRITE_ACTIONS = frozenset({"add", "edit", "patch", "publish", "delete"})
_ALL_ACTIONS = _READ_ACTIONS | _WRITE_ACTIONS


def _format_index(idx: dict[str, list[dict[str, Any]]]) -> str:
    pub = idx.get("published", [])
    drafts = idx.get("drafts", [])
    lines = [f"Published skills ({len(pub)}):"]
    for s in pub:
        lines.append(f"- {s['name']} ({s['category']}) v{s['version']}")
    lines.append(f"Drafts awaiting approval ({len(drafts)}):")
    for s in drafts:
        src = s.get("source", "")
        tag = f" [{src}]" if src else ""
        lines.append(f"- {s['name']} ({s['category']}) v{s['version']}{tag}")
    return "\n".join(lines)


def _format_search(results: list[ScoredSkill]) -> str:
    if not results:
        return "No matching skills."
    lines = ["Matching skills:"]
    for r in results:
        lines.append(
            f"- {r.skill.name} ({r.skill.category}) v{r.skill.version} [score {round(r.score, 2)}]"
        )
    return "\n".join(lines)


@dataclass
class SkillPublishResult:
    """The outcome of the teacher-draft publish path."""

    published: bool
    reason: str
    skill: Skill | None = None
    verification: VerificationResult | None = None
    detail: str = ""

    def observation(self) -> str:
        if self.published and self.skill is not None:
            return f"Teacher skill '{self.skill.name}' ({self.skill.category}) published as v{self.skill.version}."
        return f"Teacher skill draft was not published ({self.reason}): {self.detail}".rstrip(": ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "reason": self.reason,
            "skill": self.skill.to_dict() if self.skill is not None else None,
            "detail": self.detail,
        }


def make_manage_skills_handler(
    *,
    registry: SkillRegistry | None = None,
    approval_fn: Any = None,
    sandbox: Any = None,
    conversation_id: str = "",
    manager: Any = None,
) -> Any:
    """Build the ``manage_skills`` handler.

    Read actions (list / view / view_ref / search) run freely; every write
    (add / edit / patch / publish / delete) passes the fail-secure human gate,
    and any body that carries verification steps is sandbox-tested through the
    S73/S74 seam (refused when bwrap is unavailable). The handler returns an
    observation string and never raises. ``registry``, ``approval_fn``,
    ``sandbox``, ``conversation_id`` and ``manager`` are injectable; with
    ``approval_fn`` None the default manager-backed ``allowlists.request_approval``
    is used (lazy / guarded).
    """

    def _gate(action: str, gate_args: dict[str, Any]) -> bool:
        label = f"manage_skills:{action}"
        if approval_fn is not None:
            try:
                return bool(approval_fn(conversation_id, label, dict(gate_args)))
            except Exception:
                return False
        try:
            from opti_oignon.agent import allowlists

            return allowlists.request_approval(
                conversation_id, label, dict(gate_args), manager=manager
            )
        except Exception:  # pragma: no cover - fail-secure
            return False

    def handler(arguments: dict[str, Any]) -> str:
        try:
            reg = registry if registry is not None else get_skill_registry()
            args = arguments or {}
            action = _as_str(args.get("action")).strip().lower()
            name = _as_str(args.get("name")).strip()
            category = _as_str(args.get("category")).strip() or "general"

            if action in {"list", "index"}:
                return _format_index(reg.index())
            if action == "view":
                out = reg.view(name, category, draft=_as_bool(args.get("draft")))
                return out or f"No skill '{name}' ({category})."
            if action == "view_ref":
                out = reg.view_ref(name, category, draft=_as_bool(args.get("draft")))
                return out or f"No skill '{name}' ({category})."
            if action == "search":
                return _format_search(
                    reg.search(
                        _as_str(args.get("query")),
                        limit=_as_int(args.get("limit"), 5),
                        include_drafts=_as_bool(args.get("draft")),
                    )
                )

            if action not in _WRITE_ACTIONS:
                return "manage_skills 'action' must be one of: " + ", ".join(sorted(_ALL_ACTIONS)) + "."
            if not name:
                return f"manage_skills '{action}' requires a 'name'."
            gate_args = {"name": name, "category": category, "action": action}

            if action == "add":
                body = _as_str(args.get("body"))
                if not body.strip():
                    return "manage_skills 'add' requires a non-empty 'body'."
                vres = sandbox_test_verification(body, sandbox)
                if not vres.ok:
                    return f"Skill '{name}' not drafted: {vres.detail}."
                if not _gate("add", gate_args):
                    return f"Skill draft '{name}' was not approved."
                skill = reg.add(name, category, body, source=SOURCE_AGENT, status=STATUS_DRAFT)
                note = " (verification sandbox-tested)" if vres.tested else ""
                return (
                    f"Draft skill '{skill.name}' ({skill.category}) created as v{skill.version}; "
                    f"awaiting approval before publication{note}."
                )

            if action == "edit":
                body = _as_str(args.get("body"))
                if not body.strip():
                    return "manage_skills 'edit' requires a non-empty 'body'."
                if reg.get(name, category) is None:
                    return f"No published skill '{name}' ({category}) to edit."
                vres = sandbox_test_verification(body, sandbox)
                if not vres.ok:
                    return f"Skill '{name}' not edited: {vres.detail}."
                if not _gate("edit", gate_args):
                    return f"Skill edit '{name}' was not approved."
                skill = reg.update(name, category, body=body)
                return f"Skill '{skill.name}' ({skill.category}) edited to v{skill.version}."

            if action == "patch":
                old_str = _as_str(args.get("old_str"))
                new_str = _as_str(args.get("new_str"))
                existing = reg.get(name, category)
                if existing is None:
                    return f"No published skill '{name}' ({category}) to patch."
                if not old_str or existing.body.count(old_str) != 1:
                    return f"The patch target must occur exactly once in '{name}'."
                vres = sandbox_test_verification(existing.body.replace(old_str, new_str, 1), sandbox)
                if not vres.ok:
                    return f"Skill '{name}' not patched: {vres.detail}."
                if not _gate("patch", gate_args):
                    return f"Skill patch '{name}' was not approved."
                skill = reg.patch(name, category, old_str, new_str)
                return f"Skill '{skill.name}' ({skill.category}) patched to v{skill.version}."

            if action == "publish":
                draft = reg.get(name, category, draft=True)
                if draft is None:
                    return f"No draft '{name}' ({category}) to publish."
                if not _gate("publish", gate_args):
                    return f"Skill publication '{name}' was not approved."
                vres = sandbox_test_verification(draft.body, sandbox)
                if not vres.ok:
                    return f"Skill '{name}' not published: {vres.detail}."
                skill = reg.publish(name, category)
                note = " (verification sandbox-tested)" if vres.tested else ""
                return f"Skill '{skill.name}' ({skill.category}) published as v{skill.version}{note}."

            # delete
            draft_flag = _as_bool(args.get("draft"))
            if not _gate("delete", gate_args):
                return f"Skill deletion '{name}' was not approved."
            ok = reg.delete(name, category, draft=draft_flag)
            kind = "Draft" if draft_flag else "Skill"
            return (
                f"{kind} '{name}' ({category}) deleted."
                if ok
                else f"No {kind.lower()} '{name}' ({category}) to delete."
            )
        except Exception as exc:  # pragma: no cover - the handler never raises
            return f"manage_skills failed: {exc}"

    return handler


def publish_teacher_draft(
    draft: Any,
    *,
    registry: SkillRegistry | None = None,
    approval_fn: Any = None,
    sandbox: Any = None,
    conversation_id: str = "",
    manager: Any = None,
) -> SkillPublishResult:
    """Publish a teacher-produced SKILL.md draft, gated and sandbox-tested.

    Consumes a ``teacher.TeacherSkillDraft`` through ``teacher.request_skill_approval``
    (the fail-secure publication gate); on approval, the draft's verification
    steps are sandbox-tested where present (refused when bwrap is unavailable),
    then the draft is published. A teacher draft is guidance, not authority: it
    is never published without the human gate. Never raises.
    """
    try:
        reg = registry if registry is not None else get_skill_registry()
        from opti_oignon.agent import teacher

        approved = teacher.request_skill_approval(
            draft, approval_fn=approval_fn, conversation_id=conversation_id, manager=manager
        )
        if not approved:
            return SkillPublishResult(published=False, reason="not_approved")
        body = _as_str(getattr(draft, "content", "") or getattr(draft, "body", ""))
        vres = sandbox_test_verification(body, sandbox)
        if not vres.ok:
            return SkillPublishResult(
                published=False, reason="verification_failed", verification=vres, detail=vres.detail
            )
        name = _as_str(getattr(draft, "name", "")) or "untitled-skill"
        category = _as_str(getattr(draft, "category", "")) or "general"
        source = _as_str(getattr(draft, "source", "")) or SOURCE_TEACHER
        skill = reg.add(name, category, body, source=source, status=STATUS_PUBLISHED)
        return SkillPublishResult(
            published=True, reason="published", skill=skill, verification=vres
        )
    except Exception as exc:  # pragma: no cover - never raises into the loop
        return SkillPublishResult(published=False, reason="error", detail=str(exc))


# Consumption: feed the most relevant skills into the prompt as untrusted data


@dataclass
class SkillConsultation:
    """The skills retrieved for a piece of domain work, wrapped for the prompt."""

    skills: list[Skill] = field(default_factory=list)
    block: str = ""

    def references(self) -> list[str]:
        return [s.reference() for s in self.skills]

    def message(self) -> dict[str, str] | None:
        """The consultation as an untrusted user message (reusing the block)."""
        if not self.skills or not self.block:
            return None
        try:
            from opti_oignon.agent import untrusted_context

            return {"role": untrusted_context.ROLE, "content": self.block}
        except Exception:  # pragma: no cover - defensive
            return {"role": "user", "content": self.block}

    def to_dict(self) -> dict[str, Any]:
        return {"skills": [s.to_dict() for s in self.skills], "block": self.block}


def consult_skills(
    query: str,
    *,
    registry: SkillRegistry | None = None,
    limit: int = 3,
    record_usage: bool = True,
    full: bool = False,
) -> SkillConsultation:
    """Retrieve the skills most relevant to a query, wrapped as untrusted data.

    This is the consumption seam: before domain work the agent's planner can
    surface procedures it may already have. Every skill text that re-enters the
    prompt is wrapped through ``untrusted_context`` (a skill is reference, never
    an instruction). Each consulted skill's use counter is bumped in its
    ``_usage.json`` sidecar -- the SKILL.md itself is never rewritten. With
    ``full`` the whole body is included; otherwise the compact reference
    (identity plus the When to Use trigger). Never raises.
    """
    try:
        reg = registry if registry is not None else get_skill_registry()
        skills = reg.relevant(query, limit=limit)
        if not skills:
            return SkillConsultation()
        parts: list[str] = []
        for s in skills:
            if record_usage:
                reg.increment_usage(s.name, s.category)
            parts.append(s.to_markdown() if full else s.reference())
        from opti_oignon.agent import untrusted_context

        block = untrusted_context.wrap(
            "\n\n".join(parts), source=untrusted_context.SOURCE_SKILL
        )
        return SkillConsultation(skills=skills, block=block)
    except Exception:  # pragma: no cover - consumption never raises into the loop
        return SkillConsultation()


# Module-level registry (lazily constructed; injectable; reset for tests)

_REGISTRY: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    """The process-level skill registry (lazily constructed at the default root)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SkillRegistry()
    return _REGISTRY


def set_skill_registry(registry: SkillRegistry) -> None:
    """Install a registry (e.g. one rooted at a configured directory)."""
    global _REGISTRY
    _REGISTRY = registry


def reset_skill_registry() -> None:
    """Drop the registry singleton so tests do not leak state across runs."""
    global _REGISTRY
    _REGISTRY = None
