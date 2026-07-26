"""Conservative memory curation for Opti-Oignon.

A periodic, deliberately conservative pass that consolidates near-duplicate
facts and retires clearly obsolete ones, governed by three rules:

1. Fingerprint gating. The pass is short-circuited by a ``memory_tidy_state.json``
   sidecar that stores, per user, a hash of the active-fact set. When the
   fingerprint is unchanged the pass is a no-op, so an unchanged state is never
   re-audited. After a pass the fingerprint is recomputed over the post-pass set
   and saved, which makes the pass idempotent.

2. Conservative removal. Consolidation only merges high-confidence near-duplicates
   (Jaccard at or above ``CONSOLIDATE_JACCARD``, well above the add-time 0.6), and
   an optional LLM curator may propose retirements only when it returns a
   confidence at or above ``HIGH_CONFIDENCE``; when in doubt, keep. Removal
   prefers soft delete over hard delete.

3. Coordinated mutations. Every mutation routes through the coordinated
   ``MemoryStore`` (soft_delete / touch), so the cross-layer consistency and the
   per-user isolation apply uniformly. Curation never writes the canonical store
   or the vector layer directly.

The LLM client is a guarded import (``ollama``) with an injectable chat callable,
and the sidecar path is injectable, so the module loads and tests in isolation
via ``spec_from_file_location`` without ollama or a configured DATA_DIR.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Module sentinels (project convention).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Guarded model client (the same ollama client the inference path uses).
try:
    import ollama

    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None  # type: ignore[assignment]
    OLLAMA_AVAILABLE = False

# Jaccard helper, sourced from the dedup module so the threshold semantics match.
# Guarded with a local fallback for pure isolation (the runtime tests preload
# dedup, exercising the real import).
try:
    from .dedup import jaccard_similarity
except Exception:  # pragma: no cover - pure-isolation fallback
    _TOK = re.compile(r"[a-z0-9]+")

    def jaccard_similarity(a: str, b: str) -> float:  # type: ignore[misc]
        ta = set(_TOK.findall((a or "").lower()))
        tb = set(_TOK.findall((b or "").lower()))
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


STATE_FILENAME = "memory_tidy_state.json"

# Consolidation requires a high-confidence near-duplicate, deliberately stricter
# than the add-time dedup threshold (0.6), since curation removes rather than
# declines to insert.
CONSOLIDATE_JACCARD = 0.85
# An LLM-proposed retirement is acted on only at or above this confidence.
HIGH_CONFIDENCE = 0.8

CURATOR_FALLBACK_MODELS = [
    "qwen2.5:7b",
    "llama3.1:8b",
    "qwen2.5:3b",
    "llama3.2:3b",
    "mistral:7b",
]
CURATION_TEMPERATURE = 0.1
MAX_CURATION_TOKENS = 300

CURATION_SYSTEM_PROMPT = """You audit a user's saved memory facts and flag only those that are clearly safe to remove.

Rules:
1. Flag a fact ONLY if it is clearly obsolete, contradicted by a newer fact, or an exact redundancy.
2. When in doubt, KEEP. Prefer keeping a fact over removing it.
3. Never flag a fact merely for being old or short.
4. Output ONLY a JSON object: {"retire": [{"id": "<id>", "confidence": <0..1>}]}. Empty list if nothing is clearly removable.

Each flagged fact must carry a confidence in [0, 1]; only high-confidence flags will be acted on.
"""

_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class ConsolidationPair:
    """A near-duplicate pair: keep_id is retained, retire_id is removed."""

    keep_id: str
    retire_id: str
    score: float


@dataclass
class CurationReport:
    """The outcome of a curation pass.

    skipped is True when the fingerprint was unchanged (a no-op). Otherwise the
    counts describe what the pass did.
    """

    skipped: bool = True
    fingerprint: str = ""
    considered: int = 0
    consolidated: int = 0
    retired: int = 0
    retired_ids: list[str] = field(default_factory=list)


def compute_fingerprint(facts: list[Any]) -> str:
    """A stable hash of the active-fact set (id, text, category), order-free."""
    payload = json.dumps(
        sorted([[f.id, f.text, f.category] for f in facts]),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reply_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(response.get("content") or "")
    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "")
    return ""


def parse_curation_response(raw: str) -> list[tuple[str, float]]:
    """Parse a curator reply into (id, confidence) pairs. Returns [] on problems.

    Accepts ``{"retire": [...]}`` or a bare list. Each entry is a dict
    ``{"id": ..., "confidence": ...}`` (confidence defaults to 1.0 when absent)
    or a bare id string (treated as confidence 1.0).
    """
    if not raw:
        return []
    cleaned = _THINK_RE.sub("", raw)
    cleaned = _FENCE_RE.sub("", cleaned)

    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")
    data: Any = None
    # Prefer the object form when it appears first; otherwise try the array.
    try:
        if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
            end = cleaned.rfind("}")
            if end > obj_start:
                data = json.loads(cleaned[obj_start : end + 1])
        else:
            end = cleaned.rfind("]")
            if arr_start != -1 and end > arr_start:
                data = json.loads(cleaned[arr_start : end + 1])
    except json.JSONDecodeError as exc:
        logger.debug("curation: malformed JSON (%s)", exc)
        return []

    if isinstance(data, dict):
        data = data.get("retire", [])
    if not isinstance(data, list):
        return []

    out: list[tuple[str, float]] = []
    for item in data:
        if isinstance(item, str):
            out.append((item, 1.0))
        elif isinstance(item, dict) and "id" in item:
            try:
                conf = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                conf = 1.0
            out.append((str(item["id"]), conf))
    return out


class MemoryCurator:
    """Conservative curation over the coordinated store, gated by a fingerprint."""

    def __init__(
        self,
        store: Any | None = None,
        *,
        chat_fn: Callable[..., Any] | None = None,
        model: str | None = None,
        state_path: Path | str | None = None,
        consolidate_jaccard: float = CONSOLIDATE_JACCARD,
        high_confidence: float = HIGH_CONFIDENCE,
        fallback_models: list[str] | None = None,
    ) -> None:
        self._store = store
        self._chat_fn = chat_fn
        self._model = model
        self._state_path_override = Path(state_path) if state_path is not None else None
        self._consolidate_jaccard = consolidate_jaccard
        self._high_confidence = high_confidence
        self._fallback_models = (
            list(fallback_models) if fallback_models is not None else list(CURATOR_FALLBACK_MODELS)
        )
        self._lock = threading.RLock()

    # Store and client resolution (lazy so the module imports in isolation).

    def _get_store(self) -> Any:
        if self._store is None:
            from .dedup import get_memory_store

            self._store = get_memory_store()
        return self._store

    def _get_chat_fn(self) -> Callable[..., Any] | None:
        if self._chat_fn is not None:
            return self._chat_fn
        if OLLAMA_AVAILABLE and ollama is not None:
            return ollama.chat
        return None

    def _resolve_model(self) -> str | None:
        if self._model:
            return self._model
        return self._fallback_models[0] if self._fallback_models else None

    # Sidecar state under DATA_DIR (overridable for tests).

    def _state_path(self) -> Path:
        if self._state_path_override is not None:
            return self._state_path_override
        try:
            from ..config import DATA_DIR

            base = Path(DATA_DIR)
        except Exception:
            base = Path("data")
        return base / STATE_FILENAME

    def _load_state(self) -> dict[str, str]:
        path = self._state_path()
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # pragma: no cover - corrupt sidecar
            logger.debug("curation: unreadable state (%s)", exc)
            return {}

    def _save_state(self, state: dict[str, str]) -> None:
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False)

    def reset_state(self, user_id: str | None = None) -> None:
        """Clear the sidecar (all users) or just one user's fingerprint."""
        if user_id is None:
            path = self._state_path()
            if path.exists():
                path.unlink()
            return
        store = self._get_store()
        uid = store.resolve_user(user_id)
        state = self._load_state()
        if uid in state:
            del state[uid]
            self._save_state(state)

    # Fingerprint

    def compute_fingerprint(self, user_id: str | None = None) -> str:
        store = self._get_store()
        uid = store.resolve_user(user_id)
        facts = store.list(active_only=True, user_id=uid)
        return compute_fingerprint(facts)

    def needs_pass(self, user_id: str | None = None) -> bool:
        store = self._get_store()
        uid = store.resolve_user(user_id)
        current = self.compute_fingerprint(uid)
        return self._load_state().get(uid) != current

    # Consolidation (deterministic, high-confidence near-duplicates)

    def find_consolidations(self, facts: list[Any]) -> list[ConsolidationPair]:
        """Greedy near-duplicate clustering; the strongest fact represents each.

        Strength order: higher use_count first, then older created_at, then id.
        Each fact near (Jaccard >= threshold) an already-kept representative is
        flagged for retirement, paired with that representative.
        """
        ordered = sorted(
            facts,
            key=lambda r: (-int(getattr(r, "use_count", 0)), getattr(r, "created_at", ""), r.id),
        )
        kept: list[Any] = []
        pairs: list[ConsolidationPair] = []
        for fact in ordered:
            match = None
            best = 0.0
            for rep in kept:
                score = jaccard_similarity(fact.text, rep.text)
                if score >= self._consolidate_jaccard and score > best:
                    match, best = rep, score
            if match is not None:
                pairs.append(ConsolidationPair(match.id, fact.id, best))
            else:
                kept.append(fact)
        return pairs

    # LLM-proposed retirements (optional, high-confidence gate, never raises)

    def _llm_retirements(self, facts: list[Any], *, user_id: str | None = None) -> set[str]:
        out: set[str] = set()
        try:
            chat_fn = self._get_chat_fn()
            if chat_fn is None or not facts:
                return out
            model = self._resolve_model()
            if not model:
                return out
            listing = "\n".join(f"{f.id}: [{f.category}] {f.text}" for f in facts)
            response = chat_fn(
                model=model,
                messages=[
                    {"role": "system", "content": CURATION_SYSTEM_PROMPT},
                    {"role": "user", "content": "Audit these facts:\n\n" + listing},
                ],
                options={"temperature": CURATION_TEMPERATURE, "num_predict": MAX_CURATION_TOKENS},
            )
            proposals = parse_curation_response(_reply_text(response))
            valid_ids = {f.id for f in facts}
            for fid, confidence in proposals:
                if fid in valid_ids and confidence >= self._high_confidence:
                    out.add(fid)
        except Exception as exc:
            logger.warning("curation LLM pass failed, swallowed: %s", exc)
        return out

    # The pass

    def curate(
        self,
        user_id: str | None = None,
        *,
        force: bool = False,
        use_llm: bool = True,
        hard_delete: bool = False,
    ) -> CurationReport:
        """Run one conservative pass; a no-op when the fingerprint is unchanged."""
        report = CurationReport()
        try:
            store = self._get_store()
            uid = store.resolve_user(user_id)
            facts = store.list(active_only=True, user_id=uid)
            report.considered = len(facts)
            current_fp = compute_fingerprint(facts)
            state = self._load_state()

            if not force and state.get(uid) == current_fp:
                report.skipped = True
                report.fingerprint = current_fp
                return report

            report.skipped = False
            retire_ids: set[str] = set()

            pairs = self.find_consolidations(facts)
            for pair in pairs:
                retire_ids.add(pair.retire_id)
                try:
                    store.touch(pair.keep_id, user_id=uid)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("curation touch failed (%s)", exc)
            report.consolidated = len(pairs)

            if use_llm:
                retire_ids |= self._llm_retirements(facts, user_id=uid)

            applied = 0
            for fid in retire_ids:
                try:
                    if hard_delete:
                        ok = store.hard_delete(fid, user_id=uid)
                    else:
                        ok = store.soft_delete(fid, user_id=uid)
                    if ok:
                        applied += 1
                except Exception as exc:
                    logger.warning("curation removal failed, skipped: %s", exc)
            report.retired = applied
            report.retired_ids = sorted(retire_ids)

            # Recompute over the post-pass active set so a re-run is a no-op.
            post = store.list(active_only=True, user_id=uid)
            new_fp = compute_fingerprint(post)
            state[uid] = new_fp
            self._save_state(state)
            report.fingerprint = new_fp
            return report
        except Exception as exc:
            logger.warning("curation failed, swallowed: %s", exc)
            return report


# Module-level singleton with a reset for test isolation. The store is resolved
# lazily inside the curator so this module stays importable in isolation.
_curator: MemoryCurator | None = None


def get_curator() -> MemoryCurator:
    global _curator
    if _curator is None:
        _curator = MemoryCurator()
    return _curator


def reset_curator() -> None:
    global _curator
    _curator = None


def curate(
    user_id: str | None = None,
    *,
    force: bool = False,
    use_llm: bool = True,
    hard_delete: bool = False,
) -> CurationReport:
    """Module-level convenience over the singleton curator."""
    return get_curator().curate(
        user_id, force=force, use_llm=use_llm, hard_delete=hard_delete
    )
