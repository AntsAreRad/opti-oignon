"""Background fact extraction for Opti-Oignon personal memory.

After a finished conversation a background task extracts at most a couple of
durable facts and writes each through the coordinated ``MemoryStore`` so the
double deduplication (Jaccard 0.6 then cosine 0.92) decides insert-versus-merge.
Two invariants govern this module:

1. Conservative budget. At most ``MAX_FACTS`` facts per conversation, each under
   ``MAX_WORDS`` words, categorised into the six canonical categories. A regex
   fallback catches a few obvious facts (a stated name, location, preference, or
   intent) when the model is unavailable or returns nothing.

2. Never raise. A missing model, a malformed reply, a timeout, or a failing
   store write is logged and swallowed; the conversation path is never broken.

The model client is a guarded import (``ollama``, the same client the rest of
the inference path uses) and a chat callable is injectable, so the runtime tests
drive extraction deterministically without ollama installed. Extraction never
writes the canonical store or the vector layer directly -- only through
``MemoryStore.add`` -- so dedup and the cross-layer consistency apply uniformly.
The module imports no backend module that pulls fastapi or chromadb, so it loads
and tests in isolation via ``spec_from_file_location``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Module sentinels (project convention).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Guarded model client. The same ollama client the rest of the inference path
# uses; absent in the sandbox, where a chat_fn is injected instead.
try:
    import ollama

    OLLAMA_AVAILABLE = True
except Exception:  # ImportError in the sandbox
    ollama = None  # type: ignore[assignment]
    OLLAMA_AVAILABLE = False

# The six canonical categories, sourced from the canonical store so the two
# never drift. Guarded with a local fallback for pure isolation (the runtime
# tests preload canonical_store, exercising the real import).
try:
    from .canonical_store import CATEGORIES, DEFAULT_CATEGORY
except Exception:  # pragma: no cover - pure-isolation fallback
    CATEGORIES = frozenset(
        {"identity", "preference", "fact", "contact", "project", "goal"}
    )
    DEFAULT_CATEGORY = "fact"

# Conservative budget.
MAX_FACTS = 2
MAX_WORDS = 15
MIN_FACT_CHARS = 4

# Model selection and call parameters (mirrors context_summary / legacy memory).
FALLBACK_MODELS = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "qwen2.5:7b",
    "llama3.1:8b",
    "mistral:7b",
]
EXTRACTION_TEMPERATURE = 0.2
MAX_EXTRACTION_TOKENS = 400
EXTRACTION_TIMEOUT = 20.0
MAX_INPUT_MESSAGES = 30
_MODEL_CACHE_TTL = 300.0

EXTRACTION_SYSTEM_PROMPT = """You extract durable facts about the user from a conversation.

Rules:
1. Extract only facts worth remembering for future conversations.
2. Output AT MOST 2 facts. Fewer is better; output an empty array if nothing is durable.
3. Each fact is a short standalone sentence, fewer than 15 words, in the third person ("The user ...").
4. Ignore ephemeral questions, pleasantries, and one-off requests.
5. Categorise each fact as one of: identity, preference, fact, contact, project, goal.
6. Output ONLY a JSON array, no preamble and no explanation. If nothing is durable, output [].

Categories:
- identity: name, age, location, occupation, who the user is.
- preference: likes, dislikes, style, language, workflow choices.
- contact: an email, phone number, or handle the user shares.
- project: a project, repository, or research topic the user works on.
- goal: an objective or intent the user states.
- fact: any other durable fact that fits no other category.

Output format (strict JSON):
[
  {"fact": "The user works in bioinformatics and ecology", "category": "project"},
  {"fact": "The user prefers French replies", "category": "preference"}
]
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass
class ExtractedFact:
    """A single extracted fact and its provenance.

    origin is "llm" when the model produced it and "regex" when the fallback
    did, recorded so curation and the audit log can tell them apart.
    """

    text: str
    category: str
    origin: str = "llm"


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _word_count(text: str) -> int:
    return len(text.split())


def _valid_fact_text(text: str) -> bool:
    if len(text) < MIN_FACT_CHARS:
        return False
    n = _word_count(text)
    return 0 < n < MAX_WORDS


def _coerce_category(category: str) -> str:
    cat = (category or "").strip().lower()
    return cat if cat in CATEGORIES else DEFAULT_CATEGORY


def _dedupe_and_cap(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Drop case-insensitive duplicates within the batch and cap at MAX_FACTS."""
    seen: set[str] = set()
    out: list[ExtractedFact] = []
    for fact in facts:
        key = fact.text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(fact)
        if len(out) >= MAX_FACTS:
            break
    return out


def parse_extraction_response(raw: str) -> list[ExtractedFact]:
    """Parse the model reply into validated facts. Returns [] on any problem."""
    if not raw:
        return []
    cleaned = _THINK_RE.sub("", raw)
    cleaned = _FENCE_RE.sub("", cleaned)

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.debug("extraction: no JSON array in reply: %s", raw[:80])
        return []
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.debug("extraction: malformed JSON (%s)", exc)
        return []

    if isinstance(data, dict):
        data = data.get("facts", [])
    if not isinstance(data, list):
        return []

    facts: list[ExtractedFact] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = _normalize_text(str(item.get("fact", "")))
        if not _valid_fact_text(text):
            continue
        facts.append(ExtractedFact(text, _coerce_category(str(item.get("category", ""))), "llm"))
    return _dedupe_and_cap(facts)


# Regex fallback. Conservative patterns over the user's own turns only; each
# capture is rebuilt as a short third-person fact and then validated. Names and
# places require a leading capital so an adjective or nationality is not mistaken
# for a name.

# The lead-in matches both sentence-initial and mid-sentence casing ([Mm], [Ii]);
# the capture stays strictly capitalised (no IGNORECASE) so a lowercase
# adjective or nationality is not mistaken for a name or place.
_NAME_RE = re.compile(r"\b[Mm]y name is ([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?)")
_CALLME_RE = re.compile(r"\b[Cc]all me ([A-Z][a-zA-Z'\-]+)")
_LIVE_RE = re.compile(r"\b[Ii] live in ([A-Z][a-zA-Z'\- ]{1,30}?)(?:[.,!?;\n]|$)")
_FROM_RE = re.compile(r"\b[Ii](?:'m| am) (?:from|based in) ([A-Z][a-zA-Z'\- ]{1,30}?)(?:[.,!?;\n]|$)")
_PREFER_RE = re.compile(r"\bI (?:really )?prefer ([^.,;:!?\n]{2,40})", re.IGNORECASE)
_LIKE_RE = re.compile(r"\bI (?:really )?(?:like|love|enjoy) ([^.,;:!?\n]{2,40})", re.IGNORECASE)
_GOAL_RE = re.compile(r"\bmy goal is to ([^.,;:!?\n]{2,50})", re.IGNORECASE)
_WANT_RE = re.compile(r"\bI want to ([^.,;:!?\n]{2,50})", re.IGNORECASE)


def _clean_capture(value: str) -> str:
    return _normalize_text(value).rstrip(" .,:;!?")


def regex_fallback(messages: list[dict[str, Any]]) -> list[ExtractedFact]:
    """Extract a few obvious facts from the user's turns without a model."""
    candidates: list[ExtractedFact] = []
    for msg in messages:
        if (msg.get("role") or "").lower() != "user":
            continue
        text = str(msg.get("content") or "")

        for match in _NAME_RE.finditer(text):
            candidates.append(ExtractedFact(f"The user's name is {_clean_capture(match.group(1))}", "identity", "regex"))
        for match in _CALLME_RE.finditer(text):
            candidates.append(ExtractedFact(f"The user's name is {_clean_capture(match.group(1))}", "identity", "regex"))
        for match in _LIVE_RE.finditer(text):
            candidates.append(ExtractedFact(f"The user lives in {_clean_capture(match.group(1))}", "identity", "regex"))
        for match in _FROM_RE.finditer(text):
            candidates.append(ExtractedFact(f"The user is from {_clean_capture(match.group(1))}", "identity", "regex"))
        for match in _PREFER_RE.finditer(text):
            candidates.append(ExtractedFact(f"The user prefers {_clean_capture(match.group(1))}", "preference", "regex"))
        for match in _LIKE_RE.finditer(text):
            candidates.append(ExtractedFact(f"The user likes {_clean_capture(match.group(1))}", "preference", "regex"))
        for match in _GOAL_RE.finditer(text):
            candidates.append(ExtractedFact(f"The user's goal is to {_clean_capture(match.group(1))}", "goal", "regex"))
        for match in _WANT_RE.finditer(text):
            candidates.append(ExtractedFact(f"The user wants to {_clean_capture(match.group(1))}", "goal", "regex"))

    valid = [f for f in candidates if _valid_fact_text(f.text)]
    return _dedupe_and_cap(valid)


def format_conversation(messages: list[dict[str, Any]]) -> str:
    """Render the most recent turns as a plain transcript for the prompt."""
    recent = messages[-MAX_INPUT_MESSAGES:]
    lines: list[str] = []
    for msg in recent:
        role = (msg.get("role") or "user").strip().lower()
        content = _normalize_text(str(msg.get("content") or ""))
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


class FactExtractor:
    """Background extractor that writes through the coordinated MemoryStore."""

    def __init__(
        self,
        store: Any | None = None,
        *,
        chat_fn: Callable[..., Any] | None = None,
        model: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        self._store = store
        self._chat_fn = chat_fn
        self._model = model
        self._fallback_models = list(fallback_models) if fallback_models is not None else list(FALLBACK_MODELS)
        self._cached_model: str | None = None
        self._model_checked_at = 0.0

    # Store resolution (lazy so the module imports in isolation).

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
        now = time.time()
        if self._cached_model and (now - self._model_checked_at) < _MODEL_CACHE_TTL:
            return self._cached_model
        if not OLLAMA_AVAILABLE or ollama is None:
            return self._fallback_models[0] if self._fallback_models else None
        try:
            listed = ollama.list()
            names: set[str] = set()
            models = getattr(listed, "models", None)
            if models is not None:
                for m in models:
                    names.add(getattr(m, "model", None) or str(m))
            elif isinstance(listed, dict):
                for m in listed.get("models", []):
                    names.add(m.get("model") or m.get("name") or "")
            for candidate in self._fallback_models:
                if candidate in names:
                    self._cached_model = candidate
                    self._model_checked_at = now
                    return candidate
                prefix = candidate.split(":")[0] + ":"
                for avail in names:
                    if avail.startswith(prefix):
                        self._cached_model = avail
                        self._model_checked_at = now
                        return avail
            if names:
                first = sorted(names)[0]
                self._cached_model = first
                self._model_checked_at = now
                return first
        except Exception as exc:  # pragma: no cover - runtime listing failures
            logger.debug("extraction: model listing failed (%s)", exc)
        return None

    # Extraction (LLM path, then fallback). Both never raise.

    def extract(self, messages: list[dict[str, Any]], *, model: str | None = None) -> list[ExtractedFact]:
        """Run the model extraction. Returns [] on any failure (never raises)."""
        try:
            if not messages or len(messages) < 2:
                return []
            chat_fn = self._get_chat_fn()
            if chat_fn is None:
                return []
            transcript = format_conversation(messages)
            if not transcript.strip():
                return []
            chosen = model or self._resolve_model()
            if not chosen:
                return []
            started = time.time()
            response = chat_fn(
                model=chosen,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": "Extract durable facts from this conversation:\n\n" + transcript},
                ],
                options={"temperature": EXTRACTION_TEMPERATURE, "num_predict": MAX_EXTRACTION_TOKENS},
            )
            elapsed = time.time() - started
            if elapsed > EXTRACTION_TIMEOUT:
                logger.warning("extraction slow: %.1fs", elapsed)
            raw = _reply_text(response)
            return parse_extraction_response(raw)
        except Exception as exc:
            logger.warning("extraction failed, swallowed: %s", exc)
            return []

    def extract_with_fallback(self, messages: list[dict[str, Any]], *, model: str | None = None) -> list[ExtractedFact]:
        """Model extraction first; the regex fallback only when it yields nothing."""
        facts = self.extract(messages, model=model)
        if facts:
            return facts
        try:
            return regex_fallback(messages)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("regex fallback failed, swallowed: %s", exc)
            return []

    # Extraction + storage. Each candidate goes through MemoryStore.add so dedup
    # decides insert-versus-merge. Never raises; one failing add does not abort.

    def extract_and_store(
        self,
        messages: list[dict[str, Any]],
        *,
        source: str = "",
        user_id: str | None = None,
        model: str | None = None,
    ) -> list[tuple[Any, Any]]:
        """Extract and persist; return the (record, decision) pairs that succeeded."""
        results: list[tuple[Any, Any]] = []
        try:
            facts = self.extract_with_fallback(messages, model=model)
            if not facts:
                return results
            store = self._get_store()
            provenance = source or "extraction"
            for fact in facts:
                try:
                    record, decision = store.add(
                        fact.text, fact.category, source=provenance, user_id=user_id
                    )
                    results.append((record, decision))
                except Exception as exc:
                    logger.warning("extraction store.add failed, skipped: %s", exc)
        except Exception as exc:
            logger.warning("extract_and_store failed, swallowed: %s", exc)
        return results

    async def aextract_and_store(
        self,
        messages: list[dict[str, Any]],
        *,
        source: str = "",
        user_id: str | None = None,
        model: str | None = None,
    ) -> list[tuple[Any, Any]]:
        """Awaitable wrapper so the conversation path can create_task this."""
        return self.extract_and_store(messages, source=source, user_id=user_id, model=model)


def _reply_text(response: Any) -> str:
    """Pull the assistant text out of an ollama-style chat response."""
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


# Module-level singleton with a reset for test isolation. The store is resolved
# lazily inside the extractor so this module stays importable in isolation.
_extractor: FactExtractor | None = None


def get_extractor() -> FactExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FactExtractor()
    return _extractor


def reset_extractor() -> None:
    global _extractor
    _extractor = None


def extract_and_store(
    messages: list[dict[str, Any]],
    *,
    source: str = "",
    user_id: str | None = None,
    model: str | None = None,
) -> list[tuple[Any, Any]]:
    """Module-level convenience over the singleton extractor (never raises)."""
    return get_extractor().extract_and_store(
        messages, source=source, user_id=user_id, model=model
    )


def schedule_extraction(
    messages: list[dict[str, Any]],
    *,
    source: str = "",
    user_id: str | None = None,
    model: str | None = None,
) -> Any | None:
    """Fire-and-forget background extraction via asyncio.create_task.

    Returns the task when a running event loop exists; otherwise logs and
    returns None. Never raises, so the conversation path is never broken.
    """
    try:
        import asyncio

        loop = asyncio.get_running_loop()
    except Exception:
        logger.debug("schedule_extraction: no running loop; skipped")
        return None
    try:
        return loop.create_task(
            get_extractor().aextract_and_store(
                messages, source=source, user_id=user_id, model=model
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("schedule_extraction failed, swallowed: %s", exc)
        return None
