#!/usr/bin/env python3
"""The on-disk SKILL.md registry.

The evolving-skills half of the agent. Skills are
plain SKILL.md files on disk, each with YAML-style frontmatter and a structured
body (When to Use / Procedure / Pitfalls / Verification), so a procedure the
agent learns once can be consulted before later domain work. Two layers live
here:

- ``SkillRegistry`` -- the on-disk store. Published skills live at
  ``<root>/<category>/<name>/SKILL.md``; a usage counter lives in a
  ``_usage.json`` sidecar next to the skill, so an unchanged SKILL.md is never
  rewritten just to bump a count; old versions are archived under
  ``<root>/<category>/<name>/.versions/`` while the skill lives, and its
  deletion takes them with it (``forget_skill``); and drafts (proposals
  awaiting human approval) are kept apart under ``<root>/.drafts/`` so a draft
  never shadows a live skill. The registry is read / list / search / view /
  view_ref plus the raw CRUD the gated tool drives. It is local and
  per-instance: there is no cross-instance marketplace and nothing here reaches
  the network.

- The ``manage_skills`` tool and the teacher's draft entry, added on top of
  this registry in later phases. Neither writes: every skill they write is a
  proposal in the review queue (``opti_oignon.pending_writes``), and only the
  person publishes, by naming the digest of the text they were shown.

Filesystem hygiene: every path segment for a category or a name is sanitised to
a strict ``[a-z0-9_-]`` slug and the resolved path is verified to stay under the
root, so a traversal payload (``..`` / ``/`` / an absolute path) can never
escape the registry. No f-string SQL is involved -- this is a file store.

Veilid sync: the registry's write seams publish
to the change feed AFTER the domain commit -- for this file store the commit
is the completed file write -- through ``_sync_publish_skill``. Every
published-tree write (``_write`` with ``draft=False``) journals the new full
state; a published ``delete`` erases the skill's versions first, then journals a tombstone that names the
skill, so a paired device forgets it too. Drafts are device-local
and journal nothing (only the human-approved published tree syncs; syncing a
draft would propagate unapproved executable surface to peers, against the
``RecordKind.SKILL`` posture in ``veilid/producers.py``). The ``_usage.json``
sidecar is device-local telemetry and journals nothing; usage numbers never
ride a payload. Best-effort and mode-free: a journalling failure never breaks
the write, and only the wire is Daily-gated downstream at the engine/guard.

Sync origin: the engine's human gate lets a received skill through on its
provenance -- peer, device, the category/name it lands under -- and never
shows its text. So a skill applied from sync gets an ``_origin.json`` mark beside
it, device-local and never journalled, holding the SHA-256 of the bytes
adopted on this device, none at first. Once a skill carries the mark, only
adopted bytes count (:meth:`SkillRegistry.sync_state`): bytes are adopted by
naming their digest after being shown (:meth:`SkillRegistry.adopt_synced`),
or by a local write that supplies the whole text (a new skill, a published
draft); an edit of unadopted bytes stays unadopted, and a mark that cannot
be read adopts nothing.

Canonical text: a body is written with its ends stripped and read back with
each line break Python splits on as a newline; :func:`canonical_text` is that
text, its own canonical text, and :func:`text_digest` the SHA-256 of its
UTF-8 bytes. A proposal carries both. A write named by a digest
(:meth:`SkillRegistry.write_accepted`, :meth:`SkillRegistry.publish_draft`,
:meth:`SkillRegistry.delete_named`) holds the registry's lock from its check
to its write, hashes the body read back from the very bytes it is about to
write, and refuses, writing nothing, when they are not the text the digest
names or when the text it changes is no longer the one it was shown against.

Admission: a skill's text reaches a model's prompt two ways -- ``oo chat``'s
``/skill`` puts its body in the system prompt, and the agent's consultation
hands the relevant ones to the model as untrusted data -- and both admit the
same bytes (:meth:`SkillRegistry.admits`), judged on the one read of them
that is used: bytes written here by hand (the ``manual`` source, never
received by sync), and bytes a person named by their digest on this device --
adopted after sync, or written from a proposal or a draft the person accepted
by the digest of its text, or adopted later. The digests of the files so
written or adopted are kept in an ``_approved.json`` mark beside the skill,
device-local and never journalled. The agent's or the teacher's text that a
person approved on its name alone, before proposals, is admitted once
adopted the same way.

Importlib-isolatable: the default root is resolved from ``config.DATA_DIR``
lazily and guarded (falling back to a per-user data directory), and the audit
hook imports ``signed_audit_log`` lazily, so this module loads and is exercised
with a temporary registry root and without the backend. The module-level
registry has a ``reset_skill_registry()`` and an injectable root for tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import threading
from contextlib import contextmanager
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
ORIGIN_FILENAME = "_origin.json"

# What the bytes of a published skill are to this device (see sync_state).
SYNC_LOCAL = "local"
SYNC_ADOPTED = "adopted"
SYNC_UNADOPTED = "unadopted"
# The shortest digest prefix that names bytes for adoption.
ADOPT_DIGEST_MIN = 12
DRAFTS_DIR = ".drafts"
APPROVED_FILENAME = "_approved.json"
LOCK_FILENAME = ".lock"

# The sources a proposal writes: the agent's and the teacher's. Never the
# hand-written ``manual``, which is admitted to a prompt without a digest.
_PROPOSAL_SOURCES = frozenset({SOURCE_AGENT, SOURCE_TEACHER})
_HEX = frozenset("0123456789abcdef")

# Directory names never treated as a category when scanning the registry.
_RESERVED_DIRS = frozenset({VERSIONS_DIR, DRAFTS_DIR, "__pycache__"})
# What the registry writes beside a skill's file, which a deletion before this build could leave.
_RESIDUE = frozenset({VERSIONS_DIR, USAGE_FILENAME, APPROVED_FILENAME, ORIGIN_FILENAME})


def _residue(entry: Path) -> bool:
    """Whether an entry of a skill's folder is one the registry writes there: its history, its counter, its marks,
    or a write's temporary file (``_write_atomic``); never a link."""
    if entry.is_symlink():
        return False
    return entry.name in _RESIDUE or (entry.name.startswith(f".{SKILL_FILENAME}.") and entry.name.endswith(".tmp"))

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
    The header is read within its line (blanks, never a line break, around the
    name and the colon), from each line's start once: linear on a long body.
    A colon alone past the header's line, with blanks alone before it and
    after it to its line's end, still closes the header, as it did before.
    """
    pattern = re.compile(
        r"^#{0,6}[^\S\n]*" + re.escape(header) + r"[^\S\n]*(?P<colon>:[^\S\n]*)?$", re.IGNORECASE | re.MULTILINE
    )
    m = pattern.search(body)
    if not m:
        return ""
    start = m.end()
    if m.group("colon") is None:
        rest = body[start:].lstrip()
        if rest.startswith(":") and not rest[1:].partition("\n")[0].strip():
            start = len(body) - len(rest) + 1
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
    # Where this skill was read from, and the SHA-256 of the bytes read: the
    # one read admission judges. Empty for a skill built in memory.
    path: str = field(default="", compare=False, repr=False)
    file_digest: str = field(default="", compare=False, repr=False)
    raw: str = field(default="", compare=False, repr=False)

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

    def canonical(self) -> str:
        """The body as a person is shown it and as the registry writes it: its canonical text."""
        return canonical_text(self.body)

    def digest(self) -> str:
        """The SHA-256 of the canonical body: the digest a person names to act on this skill's text."""
        return text_digest(self.canonical())

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


# Canonical text, digests, refusals


def canonical_text(body: Any) -> str:
    """``body`` as the registry reads it back once written: the text a proposal shows and its digest names.

    The registry writes a body with its ends stripped and reads each line
    break Python splits on -- a carriage return, a form feed, a line or a
    paragraph separator -- back as a newline. This is that text, worked out by
    serialising and parsing in memory exactly as a write and a read do, so it
    is its own canonical text.
    """
    return _parse_frontmatter(Skill(name="x", category="x", body=_as_str(body)).to_markdown())[1]


def text_digest(text: Any) -> str:
    """The SHA-256 of ``text``'s UTF-8 bytes: the digest a person names to accept the text they were shown."""
    return hashlib.sha256(_as_str(text).encode("utf-8")).hexdigest()


class SkillNotForgotten(RuntimeError):
    """A deletion refused while the skill's text stays: nothing that lets it be listed was removed, or what was
    removed is said; it is listed, and deleting it again finishes it."""


class SkillRefused(LookupError):
    """A write named by a digest that the registry refuses; nothing was written.

    ``reason`` says why: ``missing``, nothing there to change; ``changed``,
    the text there is no longer the one the digest names; ``corrupt``, the
    write carries what no proposal can -- a category or a name that is not its
    own slug, a source other than the agent's or the teacher's, or a text that
    is not canonical or not the one its digest names.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` at ``path`` whole or not at all: a temporary file beside it, renamed over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# What a deletion's answer, and the account wipe's, call what it could not remove (``store_inventory``).
_SKILLS_LABEL = "skills (written by the user or accepted from the agent)"
_OFFERS_LABEL = "skills a paired device offered (waiting for this device's approval)"
# What a deletion says of a link it removed: the link goes, never what it leads to.
_LINK = "a link: what it led to is left"


def _overwrite_and_remove(path: Path, said: list | None = None) -> bool:
    """Overwrite a file's bytes in place, then remove it; whether it was there. A filesystem that writes in place no
    longer holds them; a copy-on-write filesystem, a snapshot or a backup may. A link is refused, never followed.

    A file another link also names -- a backup made of links, say -- is removed without its overwrite: those bytes
    are the other copy's too. One this process may not write is made writable first, being the process's own; one
    that stays unwritable is removed without its overwrite. Each removal without its overwrite is said: in ``said``
    when it is given, in the log otherwise."""
    # Never blocks: a FIFO with no reader refuses the open rather than wait for one.
    nofollow = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    reason = None
    try:
        fd = os.open(path, os.O_WRONLY | nofollow)
    except FileNotFoundError:
        return False
    except PermissionError:
        try:
            fd, reason = _made_writable(path, nofollow)
        except FileNotFoundError:
            return False
    if fd is not None:
        try:
            held = os.fstat(fd)
            if not stat.S_ISREG(held.st_mode):
                reason = "not a regular file"
            elif held.st_nlink > 1:
                reason = "other links hold its bytes"
            else:
                left = held.st_size
                block = b"\0" * 65536
                while left > 0:
                    left -= os.write(fd, block[:min(len(block), left)])
                os.fsync(fd)
        finally:
            os.close(fd)
    os.unlink(path)
    if reason is not None:
        if said is None:
            logger.warning("a file removed without its overwrite (%s)", reason)
        else:
            said.append(reason)
    return True


def _made_writable(path: Path, nofollow: int) -> tuple:
    """(a descriptor that writes ``path``, None) once the file is made readable and writable by its owner -- it is
    this process's own --, or (None, why) when it cannot be; only a regular file no other link names is changed."""
    try:
        held = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as exc:
        return None, type(exc).__name__
    if not stat.S_ISREG(held.st_mode):
        return None, "not a regular file"
    if held.st_nlink > 1:
        return None, "other links hold its bytes"
    try:
        os.chmod(path, stat.S_IMODE(held.st_mode) | stat.S_IRUSR | stat.S_IWUSR)
        return os.open(path, os.O_WRONLY | nofollow), None
    except OSError as exc:
        return None, type(exc).__name__


def _forget_entry(path: Path, said: list) -> None:
    """Overwrite and remove one file; remove a link, never followed, a folder with what it holds, or anything else
    that is not a regular file. A link removed, and a file removed without its overwrite, said in ``said``."""
    if path.is_symlink():
        os.unlink(path)
        said.append(_LINK)
    elif path.is_dir():
        _forget_folder(path, said)
    elif not path.is_file():
        os.unlink(path)
    else:
        _overwrite_and_remove(path, said)


def _forget_folder(folder: Path, said: list) -> None:
    """Overwrite and remove every file under ``folder``, then its folders and itself: a link is removed, never
    followed; a link removed, and a file removed without its overwrite, said in ``said``. Each entry is tried; the
    first that fails raises once the others are done. A folder that is not there holds none."""
    if not os.path.lexists(folder):
        return
    if folder.is_symlink():
        os.unlink(folder)
        said.append(_LINK)
        return
    if not folder.is_dir():
        os.unlink(folder)
        return
    first: OSError | None = None
    for top, dirs, files in os.walk(folder, topdown=False, followlinks=False):
        for name in files:
            try:
                _forget_entry(Path(top) / name, said)
            except OSError as exc:
                first = first or exc
        for name in dirs:
            path = Path(top) / name
            try:
                if path.is_symlink():
                    os.unlink(path)
                    said.append(_LINK)
                else:
                    os.rmdir(path)
            except OSError as exc:
                first = first or exc
    try:
        os.rmdir(folder)
    except OSError as exc:
        first = first or exc
    if first is not None:
        raise first


def _absent(error: ImportError, module: str) -> bool:
    """Whether an import failed because the module, or a package it would be in, is not in this tree -- then it holds
    nothing; a module that is there and fails to load is broken, and raises."""
    return (isinstance(error, ModuleNotFoundError) and bool(error.name)
            and (error.name == module or module.startswith(error.name + ".")))


def _sync_absent(error: ImportError) -> bool:
    """Whether an import of the sync journal, the ledger or their records failed because they are not in this tree."""
    return any(_absent(error, f"opti_oignon.veilid.{name}") for name in ("change_feed", "deferred_ledger", "records"))


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


# Veilid sync publish hook

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


def _tombstone_payload(category: str, name: str) -> dict[str, Any]:
    """A deletion's payload: the skill it deletes, named as its key names it, and nothing of its text -- a paired
    device lands it by that name (one built before this, by the key alone)."""
    return {"user_id": _sync_owner_id(), "skill": {"category": category, "name": name}}


def _keyed(record_id: Any) -> tuple[str, str] | None:
    """The category and the name a skill's key names, when it is one: two segments the sanitiser keeps as they are,
    joined as ``_skill_sync_key`` joins them; None otherwise."""
    if not isinstance(record_id, str):
        return None
    category, sep, name = record_id.partition("/")
    if not sep or _safe_segment(category, "") != category or _safe_segment(name, "") != name:
        return None
    return (category, name) if _skill_sync_key(category, name) == record_id else None


# The stores a deletion asks beside the registry's folder, by the names their modules spell: each is asked only when
# its file is there, in the folder the registry's own lies in -- a registry rooted elsewhere has none of them.
_JOURNAL_FILE = "veilid_change_feed.db"
_LEDGER_FILE = "veilid_deferred.db"
_QUEUE_FILE = "pending_writes.db"


def _forget_in_journal(key: str, data: Path) -> list[str]:
    """Erase every version of a skill in the sync journal, overwritten, before its deletion is done
    (``change_feed.forget_record``): its words leave this device even when its tombstone cannot be journalled after,
    and the highest clock erased stays the key's floor, which the tombstone minted after outclocks. A journal whose
    file is not in ``data`` holds none, and nothing is loaded to be asked. A journal that is there and cannot be
    written -- or that this process cannot load, its module broken or absent -- raises: the deletion refuses, the
    skill listed, since nothing else would ever erase what it holds."""
    if not (data / _JOURNAL_FILE).exists():
        return []
    from opti_oignon.veilid.change_feed import forget_record
    from opti_oignon.veilid.records import RecordKind

    with _SYNC_LOCK:
        forget_record(RecordKind.SKILL, key)
    return []


def _journal_state(key: str, data: Path) -> str | None:
    """What the sync journal holds of a skill (``change_feed.record_states``: ``tombstone``, ``live``, ``owed``,
    ``none``); ``absent`` when its file is not in ``data``, or in a tree without it; None when it cannot be read."""
    if not (data / _JOURNAL_FILE).exists():
        return "absent"
    try:
        from opti_oignon.veilid.change_feed import record_states
        from opti_oignon.veilid.records import RecordKind
    except ImportError as exc:
        return "absent" if _sync_absent(exc) else None
    try:
        return record_states(RecordKind.SKILL, [key]).get(key)
    except Exception:  # noqa: BLE001 - not read is said by the caller
        logger.warning("the sync journal could not be read for a deleted skill", exc_info=True)
        return None


def _owed_keys(data: Path) -> list[str]:
    """The skills whose deletion here erased their versions and whose tombstone the sync journal never took
    (``change_feed.owed``); none when its file is not in ``data``, or in a tree without it. A journal that cannot be
    read raises."""
    if not (data / _JOURNAL_FILE).exists():
        return []
    try:
        from opti_oignon.veilid.change_feed import owed
        from opti_oignon.veilid.records import RecordKind
    except ImportError as exc:
        if _sync_absent(exc):
            return []
        raise
    return owed(RecordKind.SKILL)


def _forget_offers(key: str, data: Path, below: int, beaten_by: Any = None) -> list[str]:
    """Forget what the deferred ledger holds of a deleted skill -- a paired device's offer its deletion outclocks,
    strictly below ``below``, or at that clock beaten by its tombstone ``beaten_by``; what stays, named. A ledger
    whose file is not in ``data`` holds none, and nothing is loaded to be asked."""
    if not (data / _LEDGER_FILE).exists():
        return []
    try:
        from opti_oignon.veilid.deferred_ledger import forget_offers
        from opti_oignon.veilid.records import RecordKind
    except ImportError as exc:
        return [] if _sync_absent(exc) else [f"{_OFFERS_LABEL} (not read: {type(exc).__name__})"]
    try:
        forget_offers(RecordKind.SKILL.value, key, below=below, beaten_by=beaten_by)
    except Exception as exc:  # noqa: BLE001 - what stays is said
        logger.warning("the deferred ledger could not forget a deleted skill's offers: %s", type(exc).__name__)
        return [f"{_OFFERS_LABEL} (not removed: {type(exc).__name__}; the wipe removes them)"]
    return []


def _tombstone_of(key: str) -> Any:
    """The tombstone the sync journal holds for a skill now, or None: none, or one that cannot be read. Asked only of
    a journal that is there."""
    try:
        from opti_oignon.veilid.change_feed import get_change_feed
        from opti_oignon.veilid.records import RecordKind

        held = get_change_feed().latest(RecordKind.SKILL, key)
    except Exception:  # noqa: BLE001 - an unread tombstone beats nothing: no offer at its clock is forgotten on it
        logger.debug("the sync journal's tombstone could not be read for a deleted skill", exc_info=True)
        return None
    return held if held is not None and held.deleted else None


def _clock_now(key: str) -> int:
    """The highest clock the sync journal holds for a skill, its floor included; 0 when there is none to read. Asked
    only of a journal that is there."""
    try:
        from opti_oignon.veilid.change_feed import get_change_feed
        from opti_oignon.veilid.records import RecordKind

        return int(get_change_feed().current_clock(RecordKind.SKILL, key))
    except Exception:  # noqa: BLE001 - an unread clock outclocks nothing: no offer is forgotten on it
        logger.debug("the sync journal's clock could not be read for a deleted skill", exc_info=True)
        return 0


def _proposal_key(arguments: Any) -> str | None:
    """The key of the skill a proposal names, or None for arguments that name none."""
    if not isinstance(arguments, dict):
        return None
    category, name = arguments.get("category"), arguments.get("name")
    if not isinstance(category, str) or not isinstance(name, str) or not category or not name:
        return None
    return _skill_sync_key(category, name)


def _forget_proposals(category: str, name: str, *, data: Path, present: Any = None) -> list[str]:
    """Forget the review queue's decided proposals of a deleted skill -- declined, or accepted and written; one still
    pending is listed for a decision --, or with ``present`` (the keys of the skills here) the edits and deletions of
    every skill not here, decided since its deletion -- a proposal of a new skill, which was never one or went with
    its own deletion, stays --; what stays, named. A queue whose file is not in ``data`` holds none, and nothing is
    loaded to be asked."""
    if not (data / _QUEUE_FILE).exists():
        return []
    key = _skill_sync_key(category, name)

    def names(action: str, status: str, arguments: dict) -> bool:
        named = _proposal_key(arguments)
        if present is None:
            return named == key
        return named is not None and named not in present and action != "add"

    try:
        from opti_oignon.pending_writes import forget_decided
    except ImportError as exc:
        if _absent(exc, "opti_oignon.pending_writes"):
            return []
        return [f"pending writes (the skill's decided proposals not read: {type(exc).__name__})"]
    try:
        forget_decided("skills", names)
    except Exception as exc:  # noqa: BLE001 - what stays is said
        logger.warning("the review queue could not forget a deleted skill's proposals: %s", type(exc).__name__)
        return [f"pending writes (the skill's decided proposals not removed: {type(exc).__name__}; the wipe removes "
                "them)"]
    return []


def _sync_publish_skill(
    skill_id: str,
    payload_fn: Callable[[], dict[str, Any] | None] | None = None,
    *,
    deleted: bool = False,
    updated_at: str = "",
) -> str:
    """Journal a skill change for Veilid sync, best-effort (SYN-01); ``journalled``, ``absent`` (no framework),
    ``skipped`` (no state to publish) or ``failed``.

    Called by the registry's write seams AFTER the domain commit -- for this
    file store the commit is the completed file write (``_write``) or unlink
    (``delete``). ``payload_fn`` is a zero-arg callable building the
    full-state payload; it runs INSIDE this hook's protection, and only after
    the availability probe passes, so when sync is absent the write pays
    nothing (no payload build, no journal append). The contract
    (ROADMAP_SYNC_CYCLE, the lot-1/2 precedents):

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
    - A tombstone's payload, when ``payload_fn`` gives one, names the skill it
      deletes and nothing of its text (``_tombstone_payload``): a paired device
      lands it by that name.

    Clock discipline: next = the highest clock journalled for the key, plus
    one (an unseen key yields 0, so the first clock is 1). ``_SYNC_LOCK``
    serialises mint + append per process (see its note: the lot-3 adaptation
    for a lockless file store).
    """
    try:
        with _SYNC_LOCK:
            from opti_oignon.veilid.guard import veilid_available

            if not veilid_available():
                return "absent"
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
                    return "skipped"
            elif payload_fn is not None:
                payload = payload_fn()
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
            return "journalled"
    except Exception:
        logger.warning(
            "veilid sync publish failed for skill %s (write unaffected)",
            skill_id,
            exc_info=True,
        )
        return "failed"


# The registry


# The registry's write lock: a check and the write it permits never interleave
# with another such write. In this process by this lock, across processes by
# an advisory file lock where the platform has one; the depth keeps a thread
# that already holds it from locking the file twice.
_WRITE_LOCK = threading.RLock()
_LOCK_DEPTH = threading.local()


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
        """The skill at ``path``, read once: its text, where it was read, and the digest of the bytes read.

        Its name, category and status are where it lies -- the folders of its
        path, and the drafts area or the published tree -- never what its text
        claims: a file cannot pose as another skill, nor a published one as a
        draft.
        """
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except Exception:
            return None
        meta, body = _parse_frontmatter(text)
        category = path.parent.parent.name
        name = path.parent.name
        try:
            draft = self._drafts_root.resolve() in path.resolve().parents
        except Exception:
            return None
        try:
            version = int(meta.get("version", "1"))
        except Exception:
            version = 1
        return Skill(
            name=name,
            category=category,
            status=STATUS_DRAFT if draft else STATUS_PUBLISHED,
            version=version,
            source=meta.get("source", SOURCE_MANUAL),
            body=body,
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
            path=str(path),
            file_digest=hashlib.sha256(raw).hexdigest(),
            raw=text,
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

    def _every_skill_file(self, base: Path) -> Iterator[Path]:
        """Every ``<base>/<category>/<name>/SKILL.md``, a category the listing hides (one the sanitiser keeps,
        beginning with ``_``) included: what a deletion, the wipe and the export must reach; never a reserved
        folder nor a link."""
        if not base.is_dir():
            return
        for cat_dir in sorted(base.iterdir()):
            if (not cat_dir.is_dir() or cat_dir.is_symlink() or cat_dir.name in _RESERVED_DIRS
                    or cat_dir.name.startswith(".")):
                continue
            for name_dir in sorted(cat_dir.iterdir()):
                if not name_dir.is_dir() or name_dir.is_symlink() or name_dir.name in _RESERVED_DIRS:
                    continue
                skill_file = name_dir / SKILL_FILENAME
                if skill_file.is_file():
                    yield skill_file

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
        self, query: str, *, limit: int = 5, include_drafts: bool = False, admitted_only: bool = False
    ) -> list[ScoredSkill]:
        """Rank skills by keyword relevance to a query.

        Tokens shared with a skill's name or category weigh more than tokens in
        the body. Skills with no overlap are dropped. Ties break by version then
        name for a stable order. With ``admitted_only``, a skill whose bytes may
        not enter a prompt (:meth:`admits`) is left out before the limit counts.
        """
        terms = set(_WORD_RE.findall((query or "").lower()))
        if not terms:
            return []
        scored: list[ScoredSkill] = []
        for skill in self.list(include_drafts=include_drafts):
            if admitted_only and not self.admits(skill):
                continue
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
        """The top published skills most relevant to a query (for planning), among those admitted to a prompt."""
        return [s.skill for s in self.search(query, limit=limit, include_drafts=False, admitted_only=True)]

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

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the registry's write lock: in this process always, across processes where file locks exist."""
        with _WRITE_LOCK:
            depth = getattr(_LOCK_DEPTH, "value", 0)
            handle = None
            if depth == 0:
                try:
                    import fcntl
                except ImportError:  # pragma: no cover - no advisory file locks here
                    fcntl = None
                if fcntl is not None:
                    self.root.mkdir(parents=True, exist_ok=True)
                    handle = open(self.root / LOCK_FILENAME, "a+b")
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    except Exception:
                        handle.close()
                        raise
            _LOCK_DEPTH.value = depth + 1
            try:
                yield
            finally:
                _LOCK_DEPTH.value = depth
                if handle is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        handle.close()

    def _write(self, skill: Skill, *, draft: bool, adopt: bool = True, data: bytes | None = None) -> Skill:
        path = self._skill_path(skill.category, skill.name, draft=draft)
        if path is None:
            raise ValueError("refusing to write skill outside the registry root")
        # Under the write lock: a deletion's sweep never takes the folder a write is making for its file.
        with self._locked():
            _write_atomic(path, data if data is not None else skill.to_markdown().encode("utf-8"))
            if not draft:
                self._note_local_write(path, adopt=adopt)
                # The completed file write IS the domain commit
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
        # An edit of bytes never adopted here does not adopt them.
        adopt = self.sync_state(nm, cat) != SYNC_UNADOPTED
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
        self._write(new, draft=False, adopt=adopt)
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
        """Remove a skill, forgotten (:meth:`forget_skill`); False when the target does not exist, or when its
        deletion refused, said in the log."""
        try:
            return self.forget_skill(name, category, draft=draft)
        except SkillNotForgotten as refused:
            logger.warning("a skill deletion refused: %s", refused)
            return False

    def forget_skill(self, name: str, category: str, *, draft: bool = False, sweep: bool = True,
                     left: list | None = None) -> bool:
        """Forget a skill, or a draft: whether its file was there. What it keeps elsewhere goes with it, so no word
        of it stays on this device, and its deletion reaches the paired devices.

        Under the registry's write lock: a published skill's versions in the sync journal erased first, the floor
        they leave kept (``_forget_in_journal``; one that cannot be erased is named, and its tombstone erases them
        after); then every entry of its folder -- its history, its counter, its
        marks, a write's leftover -- overwritten and removed, its file last, the folder after, and its category's when
        it is left empty; for a published skill, its decided proposals in the review queue, its tombstone, which
        names it (``_tombstone_payload``), and a paired device's offer that tombstone outclocks; then, with ``sweep``,
        what deletions before this one left (:meth:`sweep_orphans`). An entry that cannot be removed while the file
        stays refuses (:class:`SkillNotForgotten`): the skill is listed, and deleting it again finishes it. What else stays is named in ``left``: the peers not told, a queue or a ledger
        that could not be reached, a file removed without its overwrite. A draft's deletion is device-local: no
        journal, no tombstone."""
        said: list = [] if left is None else left
        cat = _safe_segment(category, "general")
        nm = _safe_segment(name, "untitled-skill")
        path = self._skill_path(cat, nm, draft=draft)
        if path is None or not self.root.is_dir():
            return False
        key = _skill_sync_key(cat, nm)
        with self._locked():
            there = path.is_file() or path.is_symlink()
            # A folder made outside the registry under a name it does not keep (``Ops``): a skill the registry never
            # wrote, journalled or proposed under its key, forgotten as a draft is, here only.
            local = None if there else self._unkept(category, name, draft=draft)
            if local is not None:
                path, there, draft = local, True, True
                cat, nm = category, name
            if there or os.path.lexists(path.parent):
                if there and not draft:
                    try:
                        said += _forget_in_journal(key, self.root.parent)
                    except Exception as exc:  # noqa: BLE001 - refused, nothing removed
                        raise SkillNotForgotten(
                            f"'{cat}/{nm}' was not deleted: the sync journal could not erase its versions "
                            f"({type(exc).__name__}); it stays listed, delete it again") from exc
                self._forget_folder_of(path, said, listed=there)
            if there:
                _audit("delete", name=nm, category=cat, draft=draft)
                if not draft:
                    said += _forget_proposals(cat, nm, data=self.root.parent)
                    said += self._tell_peers(cat, nm, key)
            if sweep:
                said += self.sweep_orphans()
        said[:] = list(dict.fromkeys(said))
        return there

    def _unkept(self, category: str, name: str, *, draft: bool) -> Path | None:
        """The file of a skill folder made outside the registry under names it does not keep -- not its own slugs --,
        within its root, when it is there; None otherwise."""
        if _safe_segment(category, "") == category and _safe_segment(name, "") == name:
            return None
        if not all(part and "/" not in part and "\\" not in part and part not in (".", "..") and "\x00" not in part
                   for part in (category, name)):
            return None
        base = self._drafts_root if draft else self.root
        path = base / category / name / SKILL_FILENAME
        if not self._within_root(path.parent) or not (path.is_file() and not path.parent.is_symlink()):
            return None
        return path

    def _forget_folder_of(self, path: Path, said: list, *, listed: bool) -> None:
        """Overwrite and remove what a skill's folder holds, its file ``path`` last, then the folder, and its
        category's folder when nothing else is left in it. While the file is there (``listed``), an entry that cannot
        be removed refuses before the file goes: the skill stays listed."""
        folder = path.parent
        first: OSError | None = None
        try:
            entries = sorted(folder.iterdir())
        except FileNotFoundError:
            entries = []
        for entry in entries:
            if entry.name == SKILL_FILENAME:
                continue
            try:
                _forget_entry(entry, said)
            except OSError as exc:
                first = first or exc
        label = f"{folder.parent.name}/{folder.name}"
        if first is not None and listed:
            raise SkillNotForgotten(f"'{label}' was not deleted: its history or its marks could not all be removed "
                                    f"({type(first).__name__}); it stays listed, delete it again")
        if listed:
            try:
                _forget_entry(path, said)
            except OSError as exc:
                raise SkillNotForgotten(f"'{label}' was not deleted: its file could not be removed "
                                        f"({type(exc).__name__}); it stays listed, delete it again") from exc
        try:
            os.rmdir(folder)
        except FileNotFoundError:
            pass
        except OSError as exc:
            said.append(f"{_SKILLS_LABEL} (the folder of '{label}' left: {type(first or exc).__name__}; the next "
                        "deletion sweeps it)")
            return
        try:
            os.rmdir(folder.parent)
        except OSError:
            pass  # another skill of the category is there

    def _tell_peers(self, category: str, name: str, key: str) -> list[str]:
        """Publish a deleted skill's tombstone, which names it; when the journal, which exists, does not hold it
        after -- the framework absent, or the publication failed --, the peers were not told: said, and owed, so
        the next deletion or the wipe publishes it (:meth:`sweep_orphans`). A paired device's offer the tombstone
        outclocks is forgotten either way."""
        _sync_publish_skill(key, lambda: _tombstone_payload(category, name), deleted=True, updated_at=_now())
        data = self.root.parent
        state = _journal_state(key, data)
        said: list[str] = []
        if state is None:
            said.append(f"paired devices (not checked whether told of the skill '{category}/{name}''s deletion: the "
                        "sync journal could not be read)")
        elif state not in ("tombstone", "absent", "none"):
            said.append(f"paired devices (not told of the skill '{category}/{name}''s deletion: the sync journal did "
                        "not take its tombstone; the next deletion or wipe tells them)")
        # The journal's clock is now the tombstone's, or the floor its versions left: an offer below it is dead, and one
        # at the tombstone's clock that the tombstone beats. A journal that is not there is not made to be asked.
        if state in ("tombstone", "owed", "live"):
            clock = _clock_now(key)
            if clock:
                if state == "tombstone":
                    said += _forget_offers(key, data, clock, _tombstone_of(key))
                else:
                    said += _forget_offers(key, data, clock + 1)
        return said

    def _orphan_folders(self) -> list[Path]:
        """The folders of the published tree and of the drafts that hold no skill file and nothing but what the
        registry itself writes beside one (``_residue``) -- what deletions before this build left: a history, a
        counter, a mark, a dead write's temporary file. A folder holding anything else is no skill's residue, and is
        left. Asked under the registry's lock, which every write holds while it makes its folder and its temporary
        file: none is being written."""
        found = []
        for base in (self.root, self._drafts_root):
            if not base.is_dir():
                continue
            for cat_dir in sorted(base.iterdir()):
                if (not cat_dir.is_dir() or cat_dir.is_symlink() or cat_dir.name in _RESERVED_DIRS
                        or cat_dir.name.startswith(".")):
                    continue
                for name_dir in sorted(cat_dir.iterdir()):
                    if not name_dir.is_dir() or name_dir.is_symlink() or name_dir.name in _RESERVED_DIRS:
                        continue
                    if os.path.lexists(name_dir / SKILL_FILENAME):
                        continue
                    if any(not _residue(entry) for entry in name_dir.iterdir()):
                        continue
                    found.append(name_dir)
        return found

    def sweep_orphans(self) -> list[str]:
        """Finish what deletions before this one left: every folder that holds no skill file, overwritten and
        removed with what it holds (``_orphan_folders``), and every tombstone the sync journal owes
        (``change_feed.owed``: a deletion here erased the skill's versions and the journal never took its tombstone)
        published, when no skill of that key is here; what stays, named. A registry whose folder is not there holds
        none, and is not made to be asked."""
        if not self.root.is_dir():
            return []
        said: list[str] = []
        with self._locked():
            try:
                orphans = self._orphan_folders()
            except OSError as exc:
                orphans = []
                said.append(f"{_SKILLS_LABEL} (what deletions before this left not read: {type(exc).__name__})")
            for folder in orphans:
                try:
                    _forget_folder(folder, said)
                except OSError as exc:
                    said.append(f"{_SKILLS_LABEL} (the folder of '{folder.parent.name}/{folder.name}' left: "
                                f"{type(exc).__name__}; the next deletion sweeps it)")
                    continue
                try:
                    os.rmdir(folder.parent)
                except OSError:
                    pass  # another skill of the category is there
            try:
                owed = _owed_keys(self.root.parent)
            except Exception as exc:  # noqa: BLE001 - what stays is said
                owed = []
                said.append(f"paired devices (the tombstones the sync journal owes not read: {type(exc).__name__})")
            for key in owed:
                named = _keyed(key)
                if named is None:
                    continue
                path = self._skill_path(named[0], named[1], draft=False)
                if path is None or os.path.lexists(path):
                    continue
                said += self._tell_peers(named[0], named[1], key)
            # The decided proposals of a skill no longer here: one decided after the skill's deletion.
            try:
                present = {_skill_sync_key(path.parent.parent.name, path.parent.name)
                           for path in self._every_skill_file(self.root)}
            except OSError as exc:
                said.append(f"pending writes (the decided proposals of deleted skills not read: "
                            f"{type(exc).__name__})")
            else:
                said += _forget_proposals("", "", data=self.root.parent, present=present)
        return list(dict.fromkeys(said))

    def skill_keys(self) -> list[tuple[str, str, bool]]:
        """Every skill and draft here, as ``(category, name, draft)``, by its file: one whose text cannot be read is
        listed all the same."""
        return [(path.parent.parent.name, path.parent.name, base is self._drafts_root)
                for base in (self.root, self._drafts_root) for path in self._every_skill_file(base)]

    def counts(self) -> dict[str, int]:
        """What a wipe of the skills removes, counted: the skills and drafts, and the versions their histories
        keep."""
        published = list(self._every_skill_file(self.root))
        drafts = list(self._every_skill_file(self._drafts_root))
        versions = 0
        for path in published:
            history = path.parent / VERSIONS_DIR
            if history.is_dir() and not history.is_symlink():
                versions += sum(1 for entry in history.iterdir() if entry.is_file())
        return {"skills": len(published) + len(drafts), "published": len(published), "drafts": len(drafts),
                "versions": versions}

    def fingerprint(self) -> str:
        """Four hexadecimal characters drawn from every skill and draft counted -- where it lies and the digest of
        its bytes --, so one that came while another went, the count unchanged, changes them; empty when there is
        none."""
        digest = hashlib.sha256()
        seen = False
        for base in (self.root, self._drafts_root):
            for path in self._every_skill_file(base):
                seen = True
                try:
                    held = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    held = "unread"
                digest.update(f"{base is self._drafts_root}/{path.parent.parent.name}/{path.parent.name}/{held}\n"
                              .encode())
        return digest.hexdigest()[:4] if seen else ""

    def export_all(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Every skill and draft, its place, status and text, and for a skill the versions its history keeps; and
        what could not be read, named, never taken for none."""
        exported: list[dict[str, Any]] = []
        missing: list[str] = []
        for category, name, draft in self.skill_keys():
            path = self._skill_path(category, name, draft=draft)
            try:
                text = path.read_bytes().decode("utf-8") if path is not None else ""
            except (OSError, UnicodeDecodeError) as exc:
                missing.append(f"skills (the skill '{category}/{name}' not exported: {type(exc).__name__})")
                continue
            meta, _body = _parse_frontmatter(text)
            item: dict[str, Any] = {"category": category, "name": name,
                                    "status": STATUS_DRAFT if draft else STATUS_PUBLISHED,
                                    "version": meta.get("version", ""), "source": meta.get("source", ""),
                                    "created_at": meta.get("created_at", ""), "updated_at": meta.get("updated_at", ""),
                                    "markdown": text, "versions": []}
            history = path.parent / VERSIONS_DIR if path is not None else None
            if not draft and history is not None and history.is_dir() and not history.is_symlink():
                for entry in sorted(history.iterdir()):
                    if not entry.is_file() or entry.is_symlink():
                        continue
                    try:
                        item["versions"].append({"file": entry.name,
                                                 "markdown": entry.read_bytes().decode("utf-8")})
                    except (OSError, UnicodeDecodeError) as exc:
                        missing.append(f"skills (a version of '{category}/{name}' not exported: "
                                       f"{type(exc).__name__})")
            exported.append(item)
        return exported, missing

    def apply_synced_skill(
        self,
        record_id: str,
        payload: dict[str, Any],
        *,
        deleted: bool = False,
        updated_at: str = "",
    ) -> bool:
        """Materialise a synced SKILL (SYN-01 apply, receive half).

        The RECEIVING half of a sync round for ``SKILL`` -- a GATED, sensitive
        kind. By the time a record reaches this method the engine's human gate
        has ALREADY approved its adoption (the round defers an unapproved skill
        to the per-record ledger and never lands it), so this is purely the
        materialisation of an approved skill onto this device's file store.

        Deliberately HOOK-FREE -- it never calls ``_sync_publish_skill`` -- so
        applying a received skill cannot re-publish it and start an
        apply -> write -> publish echo. The skill is rebuilt BYTE-FAITHFULLY:
        the ``markdown`` field is the exact ``to_markdown`` text the producer
        wrote, so it is written verbatim, with no frontmatter re-serialisation.

        Security: the write is routed through :meth:`_skill_path`, which
        sanitises ``category``/``name`` to ``[a-z0-9_-]`` and refuses any path
        that escapes the registry root -- a hostile pair cannot write outside
        the skills directory (it either lands as a sanitised subdirectory of
        the root or is refused). The nested identity must agree with the record
        key (``_skill_sync_key``) or the apply is refused (integrity). Only
        ``SKILL.md`` and its ``_origin.json`` mark are touched, so the
        device-local ``_usage.json`` and the ``.versions/`` audit are
        preserved across an update. The mark is written first: a skill that
        cannot be marked as received is not written, and the bytes land
        unadopted until they are shown and adopted here.

        A ``deleted`` record forgets the skill as a deletion here does
        (:meth:`forget_skill`'s folder: its file, its history, its counter, its
        marks, overwritten and removed) with its decided proposals; the round
        journals the tombstone after, which erases its versions. A tombstone
        names the skill; one from a build before this names nothing, and lands
        by its key alone. A skill that is not here lands its tombstone all the
        same, so it is served on. Fail-secure: a malformed payload, a key
        mismatch, an out-of-root path, or any write error returns False and
        never raises into the round -- a tombstone whose forgetting stopped
        halfway is not journalled, and lands again at the next round.
        """
        try:
            if not isinstance(payload, dict):
                return False
            skill = payload.get("skill")
            if not isinstance(skill, dict):
                named = _keyed(record_id) if deleted and not skill else None
                if named is None:
                    return False
                skill = {"category": named[0], "name": named[1]}
            category = skill.get("category")
            name = skill.get("name")
            if (
                not isinstance(category, str)
                or not isinstance(name, str)
                or not category
                or not name
            ):
                return False
            # Integrity: the nested identity must hash to the record key.
            if _skill_sync_key(category, name) != record_id:
                return False
            path = self._skill_path(category, name, draft=False)
            if path is None:  # would escape the registry root -- refuse
                return False
            with self._locked():
                # Bytes a peer lands, or a skill it deletes, were named by no
                # one here: an approval of the bytes they replace is retired,
                # so those bytes, sent back later, are shown and named again.
                if deleted:
                    said: list = []
                    there = path.is_file() or path.is_symlink()
                    if there or os.path.lexists(path.parent):
                        self._forget_folder_of(path, said, listed=there)
                    if there:
                        cat, nm = _safe_segment(category, "general"), _safe_segment(name, "untitled-skill")
                        said += _forget_proposals(cat, nm, data=self.root.parent)
                    for what in said:
                        logger.warning("a paired device's deletion of skill %s: %s", record_id, what)
                    return True
                markdown = skill.get("markdown")
                if not isinstance(markdown, str) or not markdown:
                    return False
                (path.parent / APPROVED_FILENAME).unlink(missing_ok=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                origin = path.parent / ORIGIN_FILENAME
                if not origin.exists():
                    self._write_origin(origin, [])
                path.write_bytes(markdown.encode("utf-8"))
                return True
        except Exception:
            logger.debug("skill apply failed for %s", record_id, exc_info=True)
            return False

    # Sync origin (device-local; the bytes adopted here for a received skill)

    def _origin_path(self, name: str, category: str) -> Path | None:
        d = self._skill_dir(category, name, draft=False)
        if not self._within_root(d):
            return None
        return d / ORIGIN_FILENAME

    @staticmethod
    def _read_origin(origin: Path) -> list[str] | None:
        """The digests a mark records as adopted, or None when it cannot be read."""
        try:
            data = json.loads(origin.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict) or data.get("origin") != "sync":
            return None
        adopted = data.get("adopted")
        if not isinstance(adopted, list) or not all(isinstance(d, str) for d in adopted):
            return None
        return list(adopted)

    @staticmethod
    def _write_origin(origin: Path, adopted: list[str]) -> None:
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_text(
            json.dumps({"origin": "sync", "adopted": sorted(set(adopted))}, sort_keys=True),
            encoding="utf-8",
        )

    def _note_local_write(self, path: Path, *, adopt: bool) -> None:
        """Adopt what a local write left, for a skill that once arrived by sync."""
        origin = path.parent / ORIGIN_FILENAME
        if not adopt or not origin.exists():
            return
        try:
            adopted = self._read_origin(origin) or []
            self._write_origin(origin, adopted + [hashlib.sha256(path.read_bytes()).hexdigest()])
        except Exception:  # the bytes stay unadopted: closed, not open
            logger.debug("skill origin update failed", exc_info=True)

    def raw_text(self, name: str, category: str) -> str:
        """The published SKILL.md as it is on disk, or an empty string."""
        path = self._skill_path(category, name, draft=False)
        if path is None or not path.is_file():
            return ""
        try:
            return path.read_bytes().decode("utf-8")
        except Exception:
            return ""

    def current_digest(self, name: str, category: str) -> str | None:
        """The SHA-256 of the published SKILL.md bytes on disk now, or None."""
        path = self._skill_path(category, name, draft=False)
        if path is None or not path.is_file():
            return None
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return None

    def sync_state(self, name: str, category: str) -> str:
        """What the published skill's bytes are to this device.

        ``SYNC_LOCAL`` when the skill never arrived by sync here. Once it
        has, only bytes adopted here count: ``SYNC_ADOPTED`` when the bytes
        on disk are, ``SYNC_UNADOPTED`` otherwise -- a mark that cannot be
        read adopts nothing, and a path this registry cannot place is
        answered unadopted, never local.
        """
        origin = self._origin_path(name, category)
        if origin is None:
            return SYNC_UNADOPTED
        if not origin.exists():
            return SYNC_LOCAL
        adopted = self._read_origin(origin)
        digest = self.current_digest(name, category)
        if adopted is None or digest is None or digest not in adopted:
            return SYNC_UNADOPTED
        return SYNC_ADOPTED

    def adopt_synced(self, name: str, category: str, digest_prefix: str) -> str | None:
        """Adopt the bytes on disk of a skill received by sync, named by their digest.

        ``digest_prefix`` is at least ``ADOPT_DIGEST_MIN`` hex characters of
        the SHA-256 of the bytes on disk now; the full digest is recorded
        and returned. None when there is nothing to adopt (a local skill,
        bytes already adopted, no skill) or when the prefix does not name
        the bytes on disk now.
        """
        prefix = str(digest_prefix or "").strip().lower()
        if len(prefix) < ADOPT_DIGEST_MIN or self.sync_state(name, category) != SYNC_UNADOPTED:
            return None
        digest = self.current_digest(name, category)
        origin = self._origin_path(name, category)
        if digest is None or origin is None or not digest.startswith(prefix):
            return None
        try:
            self._write_origin(origin, (self._read_origin(origin) or []) + [digest])
        except Exception:
            logger.debug("skill adoption write failed", exc_info=True)
            return None
        _audit("adopt_synced", name=name, category=category, digest=digest)
        return digest

    # Admission and the writes named by a digest (device-local marks; never journalled)

    @staticmethod
    def _read_approved(mark: Path) -> list[str]:
        """The digests an ``_approved.json`` mark records, none when it is absent or cannot be read."""
        try:
            data = json.loads(mark.read_text(encoding="utf-8"))
        except Exception:
            return []
        approved = data.get("approved") if isinstance(data, dict) else None
        if not isinstance(approved, list) or not all(isinstance(d, str) for d in approved):
            return []
        return list(approved)

    def _approve(self, folder: Path, digest: str, also: tuple[str, ...] = ()) -> None:
        """Record ``digest``, of bytes a person named on this device, as the one the skill in ``folder`` may enter a
        prompt with -- with ``also``, the digest of bytes still standing while a write that replaces them is under
        way. An earlier approval named bytes this one replaces, and is retired with them, so bytes put back later are
        shown and named again."""
        approved = sorted({digest, *also})
        _write_atomic(folder / APPROVED_FILENAME, json.dumps({"approved": approved}, sort_keys=True).encode("utf-8"))

    def _folder(self, skill: Skill | None) -> Path | None:
        """The published folder ``skill`` was read from, or None for a draft, a skill built in memory, or a path
        outside the root."""
        if skill is None or not skill.path or not skill.file_digest:
            return None
        try:
            path = Path(skill.path).resolve()
            drafts = self._drafts_root.resolve()
        except Exception:
            return None
        if not self._within_root(path) or path.name != SKILL_FILENAME or drafts in path.parents:
            return None
        return path.parent

    def admission(self, skill: Skill | None) -> str | None:
        """Why the bytes ``skill`` was read from may enter a model's prompt here, judged on that one read, or None.

        ``approved``: a person named their digest on this device -- accepting
        a proposal or a draft by the digest of its text, or adopting them.
        ``adopted``: received by sync and adopted here, never rewritten here
        since. ``manual``: written here by hand, never received and never
        rewritten through the registry. Bytes the registry rewrote before
        proposals -- a folder holding a ``.versions`` history, where the
        agent's edit kept the source it replaced and was adopted as the bytes
        it rewrote -- count only once named by their digest. Anything else is
        None: a draft, bytes received and never adopted, the agent's or the
        teacher's text never named by its digest, bytes changed since, a mark
        that cannot be read. The marks are found where the bytes were read,
        never by the name their text claims.
        """
        folder = self._folder(skill)
        if folder is None:
            return None
        if skill.file_digest in self._read_approved(folder / APPROVED_FILENAME):
            return "approved"
        if (folder / VERSIONS_DIR).exists():
            return None
        origin = folder / ORIGIN_FILENAME
        if origin.exists():
            adopted = self._read_origin(origin)
            return "adopted" if adopted is not None and skill.file_digest in adopted else None
        return "manual" if skill.source == SOURCE_MANUAL else None

    def admits(self, skill: Skill | None) -> bool:
        """Whether the bytes ``skill`` was read from may enter a model's prompt here (:meth:`admission`)."""
        return self.admission(skill) is not None

    def received(self, skill: Skill | None) -> bool:
        """Whether the bytes ``skill`` was read from arrived by sync: a mark of origin lies where they were read."""
        if skill is None or not skill.path:
            return False
        return (Path(skill.path).parent / ORIGIN_FILENAME).exists()

    def rewritten(self, skill: Skill | None) -> bool:
        """Whether the registry rewrote the skill ``skill`` was read from: a ``.versions`` history lies beside it."""
        if skill is None or not skill.path:
            return False
        return (Path(skill.path).parent / VERSIONS_DIR).exists()

    def prompt_state(self, name: str, category: str) -> str:
        """What the published skill's bytes are to a prompt here, on one read of them.

        ``SYNC_LOCAL`` for bytes written here by hand; ``SYNC_ADOPTED`` for
        bytes a person named by their digest on this device;
        ``SYNC_UNADOPTED`` for anything else, a missing skill included.
        """
        admission = self.admission(self.get(name, category))
        if admission is None:
            return SYNC_UNADOPTED
        return SYNC_LOCAL if admission == "manual" else SYNC_ADOPTED

    def _publish_named(self, nm: str, cat: str, text: str, source: str, current: Skill | None) -> Skill:
        """Publish ``text`` over ``current``, once the caller has checked both under the lock; record its digest.

        The body read back from the very bytes written is hashed again, and
        refused when it is not the text: what is written is what was named.
        """
        stamp = _now()
        skill = Skill(
            name=nm,
            category=cat,
            status=STATUS_PUBLISHED,
            version=(current.version + 1) if current is not None else 1,
            source=source,
            body=text,
            created_at=(current.created_at if current is not None and current.created_at else stamp),
            updated_at=stamp,
        )
        data = skill.to_markdown().encode("utf-8")
        if _parse_frontmatter(data.decode("utf-8"))[1] != text:
            raise SkillRefused("corrupt", f"the bytes about to be written for '{cat}/{nm}' do not hold the text named")
        path = self._skill_path(cat, nm, draft=False)
        if path is None:
            raise SkillRefused("corrupt", f"'{cat}/{nm}' lies outside the registry")
        # The digest first, beside the one of the bytes it replaces while they
        # stand, if they were admitted: a write said saved is admitted, an
        # approval that cannot be recorded writes nothing, and a write that
        # fails leaves the skill as it was, still admitted. The write landed,
        # the bytes it replaced are retired.
        digest = hashlib.sha256(data).hexdigest()
        standing = (current.file_digest,) if current is not None and self.admits(current) else ()
        self._approve(path.parent, digest, also=standing)
        if current is not None:
            self._archive(cat, nm, current)
        self._write(skill, draft=False, data=data)
        self._approve(path.parent, digest)
        return skill

    def write_accepted(
        self, category: str, name: str, text: str, *, sha256: str, base_sha256: str | None, source: str
    ) -> Skill:
        """Publish ``text``, accepted by its digest, over exactly the published text ``base_sha256`` names.

        Refused (:class:`SkillRefused`), nothing written: ``corrupt`` when the
        category or the name is not its own slug, the source is not the
        agent's or the teacher's, or the text is not canonical or not the one
        ``sha256`` names; ``changed`` when the published text is no longer the
        one ``base_sha256`` names (None: no published skill). The check and the
        write hold the registry's lock, the bytes are hashed again as they are
        written, and the digest of the file written is recorded as approved
        here, so the skill may enter a prompt.
        """
        cat, nm = _as_str(category), _as_str(name)
        if not cat or not nm or _safe_segment(cat, "") != cat or _safe_segment(nm, "") != nm:
            raise SkillRefused("corrupt", f"'{cat}/{nm}' is not the slug a proposal writes")
        if source not in _PROPOSAL_SOURCES:
            raise SkillRefused("corrupt", f"a proposal does not write the source {source!r}")
        if not isinstance(text, str) or canonical_text(text) != text or text_digest(text) != sha256:
            raise SkillRefused("corrupt", f"the text for '{cat}/{nm}' is not the canonical text its digest names")
        with self._locked():
            current = self.get(nm, cat)
            if (current.digest() if current is not None else None) != base_sha256:
                raise SkillRefused("changed", f"the published skill '{cat}/{nm}' changed since it was proposed")
            skill = self._publish_named(nm, cat, text, source, current)
        _audit("publish_accepted", name=nm, category=cat, version=skill.version, source=source, digest=sha256)
        return skill

    def publish_draft(self, name: str, category: str, sha256: str) -> Skill:
        """Publish the draft whose text ``sha256`` names, as that text, and remove the draft.

        The draft is read once under the registry's lock. Refused
        (:class:`SkillRefused`), nothing written, when there is no draft
        (``missing``) or its text is not the one the digest names
        (``changed``): a draft rewritten after it was shown is not published.
        The published skill takes the draft's source and the next version,
        archiving the one it replaces; the digest of the file written is
        recorded as approved here.
        """
        cat, nm = _safe_segment(category, "general"), _safe_segment(name, "untitled-skill")
        with self._locked():
            draft = self.get(nm, cat, draft=True)
            if draft is None:
                raise SkillRefused("missing", f"no draft '{cat}/{nm}' to publish")
            if draft.digest() != _as_str(sha256):
                raise SkillRefused("changed", f"the draft '{cat}/{nm}' is not the text the digest names")
            skill = self._publish_named(nm, cat, draft.canonical(), draft.source, self.get(nm, cat))
            self.delete(nm, cat, draft=True)
        _audit("publish", name=nm, category=cat, version=skill.version, source=draft.source, digest=sha256)
        return skill

    def delete_named(self, name: str, category: str, *, draft: bool, sha256: str, left: list | None = None) -> bool:
        """Delete the draft, or the published skill, whose text ``sha256`` names, and nothing else, forgotten
        (:meth:`forget_skill`, what stays named in ``left``).

        Refused (:class:`SkillRefused`), nothing deleted, when there is no
        such skill (``missing``) or its text is no longer the one the digest
        names (``changed``); refused (:class:`SkillNotForgotten`) when its
        forgetting cannot remove its text, the skill listed.
        """
        with self._locked():
            current = self.get(name, category, draft=draft)
            if current is None:
                kind = "draft" if draft else "published skill"
                raise SkillRefused("missing", f"no {kind} '{category}/{name}' to delete")
            if current.digest() != _as_str(sha256):
                raise SkillRefused("changed", f"'{category}/{name}' is not the text the digest names")
            return self.forget_skill(name, category, draft=draft, left=left)

    def adopt(self, name: str, category: str, digest_prefix: str) -> str | None:
        """Adopt the bytes on disk of a published skill not admitted here, named by their digest.

        Whatever kept them out -- received by sync and never adopted, the
        agent's or the teacher's text, bytes the registry rewrote before
        proposals -- their digest is recorded as approved here, and a received
        skill's mark of origin records it as adopted too. ``digest_prefix`` is
        at least ``ADOPT_DIGEST_MIN`` hex characters of the SHA-256 of the
        bytes on disk now, read once under the registry's lock; the full
        digest is returned. None when there is nothing to adopt or the prefix
        does not name those bytes.
        """
        prefix = _as_str(digest_prefix).strip().lower()
        if len(prefix) < ADOPT_DIGEST_MIN or not set(prefix) <= _HEX:
            return None
        with self._locked():
            skill = self.get(name, category)
            folder = self._folder(skill)
            if folder is None or self.admits(skill) or not skill.file_digest.startswith(prefix):
                return None
            origin = folder / ORIGIN_FILENAME
            if origin.exists():
                adopted = self._read_origin(origin) or []
                if skill.file_digest not in adopted:
                    self._write_origin(origin, adopted + [skill.file_digest])
            self._approve(folder, skill.file_digest)
        _audit("adopt", name=name, category=category, digest=skill.file_digest)
        return skill.file_digest

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


# Verification sandbox-testing (the seam; refuse when bwrap is absent)

# Linear on a fence left open: one with no line break after it, or no closing fence after its line, takes the rest
# at once -- no fence after it can close either -- and only a block that reached its closing sets ``real``.
_FENCE_RE = re.compile(r"```(?:[^\n]*\n(?:(.*?)```(?P<real>)|.*)|.*)", re.DOTALL)


def _sandbox_ready(session: Any) -> bool:
    """Whether an injected sandbox session runs its commands under bwrap.

    Mirrors ``dispatch.sandbox_ready`` without importing ``dispatch``, so this
    module stays decoupled and isolatable. A missing session, a missing
    manager, or a manager that does not run bwrap (absent, or installed while
    the resolved backend is tempdir) all return False.
    """
    if session is None:
        return False
    mgr = getattr(session, "sandbox_manager", None)
    if mgr is None:
        return False
    in_use = getattr(mgr, "bwrap_in_use", None)
    if in_use is not None:
        return bool(in_use)
    return bool(getattr(mgr, "bwrap_available", False))


def _verification_commands(body: str) -> list[str]:
    """Fenced command blocks in the Verification section (the executable steps)."""
    section = _extract_section(body, SECTION_VERIFICATION)
    if not section:
        return []
    blocks = (m.group(1).strip() for m in _FENCE_RE.finditer(section) if m.group("real") is not None)
    return [block for block in blocks if block]


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


# The manage_skills tool: reads run freely, every write is a proposal

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
    """The outcome of the teacher's draft entry: proposed to the user, or why not."""

    published: bool
    reason: str
    skill: Skill | None = None
    verification: VerificationResult | None = None
    detail: str = ""
    proposal_id: str = ""
    sha256: str = ""

    @property
    def proposed(self) -> bool:
        return self.reason == "proposed" and bool(self.proposal_id)

    def observation(self) -> str:
        if self.proposed:
            return (f"Teacher skill draft proposed to the user for review ({self.proposal_id}, sha256 "
                    f"{self.sha256[:16]}): not published unless the user accepts it.")
        return f"Teacher skill draft was not proposed ({self.reason}): {self.detail}".rstrip(": ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "proposed": self.proposed,
            "reason": self.reason,
            "skill": self.skill.to_dict() if self.skill is not None else None,
            "detail": self.detail,
            "proposal_id": self.proposal_id,
            "sha256": self.sha256,
        }


class _GateSlot:
    """The gate a handler's writes pass: the run's when one is bound, else the handler's own, built on first use.

    The handler's own gate endorses nothing -- a skill is never written on
    the strength of typed words, it is always proposed -- and keeps its bound
    on proposals over the handler's life.
    """

    def __init__(self, gate: Any = None, conversation_id: str = "") -> None:
        self._gate = gate
        self._conversation_id = conversation_id
        self._lock = threading.Lock()

    def get(self) -> Any:
        with self._lock:
            if self._gate is None:
                from opti_oignon import pending_writes

                self._gate = pending_writes.WriteGate(conversation_id=self._conversation_id)
            return self._gate


def _propose(slot: _GateSlot, action: str, proposal: dict[str, Any], lead: str) -> str:
    """Hand one skill proposal to the gate of ``slot``; what the model is told. A gate that cannot be had writes
    nothing."""
    try:
        gate = slot.get()
    except Exception:
        logger.warning("review queue unavailable; skill '%s' not proposed", action, exc_info=True)
        return f"{lead}: not proposed, the review queue is unavailable, so nothing was written or proposed."
    return f"{lead}. {gate.write('skills', action, proposal)}"


def make_manage_skills_handler(
    *,
    registry: SkillRegistry | None = None,
    approval_fn: Any = None,
    sandbox: Any = None,
    conversation_id: str = "",
    manager: Any = None,
    gate: Any = None,
) -> Any:
    """Build the ``manage_skills`` handler.

    Reads ask no one: list and index show names; search, view and view_ref
    hand the model only skills whose bytes may enter a prompt here, as the
    consultation does, and never a draft's text. Every write is a proposal
    in the review queue, never a write: through ``gate``,
    the pending-write gate a run binds to its turn, or, with none bound, a
    gate of the handler's own that endorses nothing. A proposal holds the
    slug it would write, the canonical text, its SHA-256 and the digest of
    the published text it would change (none for a new skill); a patch is
    proposed as the text it would result in, a delete names its target and
    the digest of the text it would delete. The person reads the whole text
    and accepts it by that digest, later. A body that carries verification
    steps is sandbox-tested first, on that exact text (refused when bwrap is
    unavailable). ``publish`` publishes nothing: publishing is the person's.
    No one is asked during the run: ``approval_fn`` and ``manager`` are
    accepted from callers built for a gate that asked, and never called. The
    handler returns an observation string and never raises.
    """
    slot = _GateSlot(gate, conversation_id)

    def handler(arguments: dict[str, Any]) -> str:
        try:
            reg = registry if registry is not None else get_skill_registry()
            args = arguments or {}
            action = _as_str(args.get("action")).strip().lower()
            name = _as_str(args.get("name")).strip()
            category = _as_str(args.get("category")).strip() or "general"

            if action in {"list", "index"}:
                return _format_index(reg.index())
            # The agent's own reading is a consultation: only text that may
            # enter a prompt here is handed to it (:meth:`SkillRegistry.admits`).
            if action in ("view", "view_ref"):
                if _as_bool(args.get("draft")):
                    return f"Draft '{name}' ({category}) is not read here: a draft waits for the user."
                skill = reg.get(name, category)
                if skill is None:
                    return f"No skill '{name}' ({category})."
                if not reg.admits(skill):
                    return (f"Skill '{skill.name}' ({skill.category}) is not adopted on this device: its text is not "
                            "consulted until the user adopts it.")
                return skill.to_markdown() if action == "view" else skill.reference()
            if action == "search":
                return _format_search(
                    reg.search(_as_str(args.get("query")), limit=_as_int(args.get("limit"), 5), admitted_only=True)
                )

            if action not in _WRITE_ACTIONS:
                return "manage_skills 'action' must be one of: " + ", ".join(sorted(_ALL_ACTIONS)) + "."
            if action == "publish":
                return ("Not published: publishing a skill is the user's own act. A skill you add or change waits "
                        "for the user, who reads its whole text and accepts it by the digest of that text.")
            if not name:
                return f"manage_skills '{action}' requires a 'name'."
            cat, nm = _safe_segment(category, "general"), _safe_segment(name, "untitled-skill")

            if action == "delete":
                draft_flag = _as_bool(args.get("draft"))
                target = reg.get(nm, cat, draft=draft_flag)
                if target is None:
                    kind = "draft" if draft_flag else "skill"
                    return f"No {kind} '{name}' ({category}) to delete."
                proposal = {"category": cat, "name": nm, "draft": draft_flag, "base_sha256": target.digest()}
                kind = "draft" if draft_flag else "skill"
                lead = f"Deletion of the {kind} '{nm}' ({cat}), its text sha256 {proposal['base_sha256'][:16]}"
                return _propose(slot, "delete", proposal, lead)

            existing = reg.get(nm, cat)
            if action == "patch":
                old_str = _as_str(args.get("old_str"))
                new_str = _as_str(args.get("new_str"))
                if existing is None:
                    return f"No published skill '{name}' ({category}) to patch."
                if not old_str or existing.body.count(old_str) != 1:
                    return f"The patch target must occur exactly once in '{name}'."
                body = existing.body.replace(old_str, new_str, 1)
            else:
                body = _as_str(args.get("body"))
                if not body.strip():
                    return f"manage_skills '{action}' requires a non-empty 'body'."
                if action == "edit" and existing is None:
                    return f"No published skill '{name}' ({category}) to edit."
            text = canonical_text(body)
            if not text.strip():
                return f"manage_skills '{action}' would leave '{name}' empty; nothing was proposed."
            vres = sandbox_test_verification(text, sandbox)
            if not vres.ok:
                verb = {"add": "drafted", "edit": "edited", "patch": "patched"}[action]
                return f"Skill '{name}' not {verb}: {vres.detail}."
            digest = text_digest(text)
            proposal = {
                "category": cat,
                "name": nm,
                "text": text,
                "sha256": digest,
                "base_sha256": existing.digest() if existing is not None else None,
                "source": SOURCE_AGENT,
                "tested": bool(vres.tested),
            }
            note = ", verification sandbox-tested" if vres.tested else ""
            what = "Draft skill" if action == "add" else "Change of skill"
            return _propose(slot, "add" if existing is None else "edit", proposal,
                            f"{what} '{nm}' ({cat}), sha256 {digest[:16]}{note}")
        except Exception as exc:  # pragma: no cover - the handler never raises
            return f"manage_skills failed: {exc}"

    return handler


def publish_teacher_draft(
    draft: Any,
    *,
    registry: SkillRegistry | None = None,
    sandbox: Any = None,
    conversation_id: str = "",
    pending: Any = None,
    user_id: str | None = None,
) -> SkillPublishResult:
    """Submit a teacher-produced SKILL.md draft for publication: it is proposed to the user, never published here.

    The draft's text is brought to its canonical form and its verification
    steps are sandbox-tested where present (refused when bwrap is
    unavailable); it is then proposed in the review queue (``pending``, or the
    process's queue) under its slug, with the teacher's source, against the
    digest of the published text it would replace. The person reads the whole
    text and publishes it by accepting it with its digest: a teacher's draft is
    guidance, never authority, and no one is asked as the run ends. Never
    raises.
    """
    try:
        reg = registry if registry is not None else get_skill_registry()
        name = _safe_segment(_as_str(getattr(draft, "name", "")), "untitled-skill")
        category = _safe_segment(_as_str(getattr(draft, "category", "")), "general")
        text = canonical_text(_as_str(getattr(draft, "content", "") or getattr(draft, "body", "")))
        if not text.strip():
            return SkillPublishResult(published=False, reason="empty", detail="the draft holds no text")
        existing = reg.get(name, category)
        vres = sandbox_test_verification(text, sandbox)
        if not vres.ok:
            return SkillPublishResult(
                published=False, reason="verification_failed", verification=vres, detail=vres.detail
            )
        digest = text_digest(text)
        proposal = {
            "category": category,
            "name": name,
            "text": text,
            "sha256": digest,
            "base_sha256": existing.digest() if existing is not None else None,
            "source": SOURCE_TEACHER,
            "tested": bool(vres.tested),
        }
        provenance = {"source": "teacher", "turn": "none", "typed": {}, "untyped": ["text"],
                      "target": "name" if existing is not None else None, "read": []}
        if pending is None:
            from opti_oignon import pending_writes

            pending = pending_writes.get_pending_store()
        record, _new = pending.propose("skills", "add" if existing is None else "edit", proposal, provenance,
                                       conversation_id=conversation_id, user_id=user_id)
        return SkillPublishResult(published=False, reason="proposed", verification=vres, proposal_id=record.id,
                                  sha256=digest)
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
    surface procedures it may already have. Only skills whose bytes may enter
    a prompt are consulted -- the ones ``/skill`` would run
    (:meth:`SkillRegistry.admits`) -- and the limit counts those. Every skill
    text that re-enters the prompt is wrapped through ``untrusted_context`` (a
    skill is reference, never an instruction). Each consulted skill's use counter is bumped in its
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
