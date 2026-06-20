#!/usr/bin/env python3
"""Tests for S174 -- background fact extraction.

The conservative budget, the JSON parsing, and the regex fallback are exercised
directly. The extract-and-store path runs against a real coordinated MemoryStore
(tmp SQLite canonical store, the vector layer with an injected cosine
collection, and a deterministic embedder) so the double dedup decides
insert-versus-merge. A chat callable is injected so no ollama is required, and
the modules are loaded in isolation via spec_from_file_location.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
MEM = OO / "memory"

sys.path.insert(0, str(ROOT / "tests"))
from _memory_fakes import FakeChromaCollection, FakeEmbedder  # noqa: E402


def _ensure_pkg():
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.memory" not in sys.modules:
        mpkg = types.ModuleType("opti_oignon.memory")
        mpkg.__path__ = [str(MEM)]
        sys.modules["opti_oignon.memory"] = mpkg


def _ensure_real(name: str):
    full = f"opti_oignon.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(OO / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_mem(name: str):
    full = f"opti_oignon.memory.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(MEM / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg()
_ensure_real("db_encryption")
_ensure_real("user_isolation")
canon = _ensure_mem("canonical_store")
vec = _ensure_mem("vector_store")
ded = _ensure_mem("dedup")
ext = _ensure_mem("extraction")


# A chat callable shaped like ollama.chat, returning a fixed reply.
def make_chat_fn(reply: str, *, calls: list | None = None):
    def chat_fn(model=None, messages=None, options=None):
        if calls is not None:
            calls.append({"model": model, "messages": messages, "options": options})
        return {"message": {"content": reply}}

    return chat_fn


def raising_chat_fn(model=None, messages=None, options=None):
    raise RuntimeError("model unavailable")


def _json(facts: list[dict]) -> str:
    import json

    return json.dumps(facts)


KUB_LINUX = "The user likes Kubuntu Linux"
KUB = "The user likes Kubuntu"
BIO = "The user works in bioinformatics and ecology"
PARIS_FACT = "The user lives in Paris"

MAPPING = {
    KUB_LINUX: [0.0, 0.0, 1.0, 0.0],
    KUB: [0.0, 0.0, 0.97, 0.03],
    BIO: [1.0, 0.0, 0.0, 0.0],
    PARIS_FACT: [0.0, 1.0, 0.0, 0.0],
}

CONV = [
    {"role": "user", "content": "I work on bioinformatics."},
    {"role": "assistant", "content": "Tell me more."},
    {"role": "user", "content": "Mostly ecology data."},
]


def _build_store(tmp_path, *, single_user_mode=True, name="ext.db"):
    canon_store = canon.CanonicalMemoryStore(
        tmp_path / name, single_user_mode=single_user_mode
    )
    embedder = FakeEmbedder(mapping=MAPPING, dim=4)
    vstore = vec.MemoryVectorStore(
        collection=FakeChromaCollection(name=vec.COLLECTION_NAME), embedder=embedder
    )
    store = ded.MemoryStore(canon_store, vstore)
    return store, canon_store, vstore


@pytest.fixture(autouse=True)
def _reset_singletons():
    ext.reset_extractor()
    yield
    ext.reset_extractor()


@pytest.fixture
def store(tmp_path):
    return _build_store(tmp_path)[0]


# Module sentinels


class TestModuleSentinels:
    def test_feature_available(self):
        assert ext.FEATURE_AVAILABLE is True

    def test_checkpoint_sentinel(self):
        assert ext.checkpoint_before_apply is True

    def test_singleton_helpers(self):
        assert hasattr(ext, "get_extractor")
        assert hasattr(ext, "reset_extractor")

    def test_budget_constants(self):
        assert ext.MAX_FACTS == 2
        assert ext.MAX_WORDS == 15

    def test_six_categories(self):
        assert ext.CATEGORIES == frozenset(
            {"identity", "preference", "fact", "contact", "project", "goal"}
        )


# Parsing


class TestParse:
    def test_parses_valid_array(self):
        facts = ext.parse_extraction_response(
            _json([{"fact": BIO, "category": "project"}])
        )
        assert len(facts) == 1
        assert facts[0].text == BIO
        assert facts[0].category == "project"
        assert facts[0].origin == "llm"

    def test_strips_preamble_and_fences(self):
        raw = "Here you go:\n```json\n" + _json([{"fact": BIO, "category": "fact"}]) + "\n```"
        facts = ext.parse_extraction_response(raw)
        assert len(facts) == 1

    def test_strips_think_tags(self):
        raw = "<think>reasoning</think>" + _json([{"fact": BIO, "category": "fact"}])
        assert len(ext.parse_extraction_response(raw)) == 1

    def test_malformed_json_returns_empty(self):
        assert ext.parse_extraction_response("[ not json") == []

    def test_no_array_returns_empty(self):
        assert ext.parse_extraction_response("no brackets here") == []

    def test_empty_returns_empty(self):
        assert ext.parse_extraction_response("") == []

    def test_unknown_category_coerced_to_fact(self):
        facts = ext.parse_extraction_response(
            _json([{"fact": BIO, "category": "nonsense"}])
        )
        assert facts[0].category == "fact"

    def test_caps_at_two_facts(self):
        facts = ext.parse_extraction_response(
            _json(
                [
                    {"fact": "The user likes apples", "category": "preference"},
                    {"fact": "The user likes pears", "category": "preference"},
                    {"fact": "The user likes plums", "category": "preference"},
                ]
            )
        )
        assert len(facts) == ext.MAX_FACTS

    def test_rejects_too_long_fact(self):
        long_fact = "The user " + " ".join(["word"] * 20)
        assert ext.parse_extraction_response(_json([{"fact": long_fact, "category": "fact"}])) == []

    def test_accepts_fourteen_word_fact(self):
        fact14 = " ".join(["w"] * 14)
        facts = ext.parse_extraction_response(_json([{"fact": fact14, "category": "fact"}]))
        assert len(facts) == 1

    def test_rejects_fifteen_word_fact(self):
        fact15 = " ".join(["w"] * 15)
        assert ext.parse_extraction_response(_json([{"fact": fact15, "category": "fact"}])) == []

    def test_dedupes_within_batch(self):
        facts = ext.parse_extraction_response(
            _json(
                [
                    {"fact": BIO, "category": "fact"},
                    {"fact": BIO.lower(), "category": "fact"},
                ]
            )
        )
        assert len(facts) == 1

    def test_dict_with_facts_key(self):
        import json

        raw = json.dumps({"facts": [{"fact": BIO, "category": "fact"}]})
        assert len(ext.parse_extraction_response(raw)) == 1

    def test_all_stored_facts_under_word_limit(self):
        facts = ext.parse_extraction_response(
            _json([{"fact": BIO, "category": "project"}, {"fact": KUB, "category": "preference"}])
        )
        assert all(len(f.text.split()) < ext.MAX_WORDS for f in facts)


# Regex fallback


class TestRegexFallback:
    def test_extracts_name(self):
        facts = ext.regex_fallback([{"role": "user", "content": "Hi, my name is Leon."}])
        assert any(f.category == "identity" and "Leon" in f.text for f in facts)

    def test_extracts_location(self):
        facts = ext.regex_fallback([{"role": "user", "content": "I live in Montpellier."}])
        assert any("Montpellier" in f.text for f in facts)

    def test_extracts_preference(self):
        facts = ext.regex_fallback([{"role": "user", "content": "I prefer dark mode."}])
        assert any(f.category == "preference" for f in facts)

    def test_extracts_goal(self):
        facts = ext.regex_fallback([{"role": "user", "content": "My goal is to finish the thesis."}])
        assert any(f.category == "goal" for f in facts)

    def test_ignores_assistant_turns(self):
        facts = ext.regex_fallback([{"role": "assistant", "content": "My name is Claude."}])
        assert facts == []

    def test_caps_at_two(self):
        content = "My name is Leon. I live in Paris. I prefer tea. My goal is to win."
        facts = ext.regex_fallback([{"role": "user", "content": content}])
        assert len(facts) <= ext.MAX_FACTS

    def test_no_facts_for_plain_text(self):
        facts = ext.regex_fallback([{"role": "user", "content": "What time is it?"}])
        assert facts == []

    def test_does_not_capture_lowercase_nationality_as_name(self):
        facts = ext.regex_fallback([{"role": "user", "content": "I am french and tired."}])
        assert not any(f.category == "identity" and "french" in f.text.lower() and "name" in f.text.lower() for f in facts)


# Never raises


class TestNeverRaises:
    def test_extract_with_raising_chat_fn(self):
        extractor = ext.FactExtractor(chat_fn=raising_chat_fn, model="m")
        assert extractor.extract(CONV) == []

    def test_extract_without_chat_fn(self, monkeypatch):
        monkeypatch.setattr(ext, "OLLAMA_AVAILABLE", False)
        monkeypatch.setattr(ext, "ollama", None)
        extractor = ext.FactExtractor(chat_fn=None, model="m")
        # No injected chat_fn and no ollama -> empty, no raise.
        assert extractor.extract(CONV) == []

    def test_extract_short_conversation(self):
        extractor = ext.FactExtractor(chat_fn=make_chat_fn(_json([{"fact": BIO, "category": "fact"}])), model="m")
        assert extractor.extract([{"role": "user", "content": "hi"}]) == []

    def test_store_add_failure_is_swallowed(self, store):
        class Boom:
            def add(self, *a, **k):
                raise RuntimeError("store down")

        extractor = ext.FactExtractor(Boom(), chat_fn=make_chat_fn(_json([{"fact": BIO, "category": "fact"}])), model="m")
        assert extractor.extract_and_store(CONV) == []

    def test_schedule_without_loop_returns_none(self):
        assert ext.schedule_extraction(CONV) is None


# Fallback chaining


class TestFallbackChaining:
    def test_llm_empty_falls_back_to_regex(self):
        convo = [
            {"role": "user", "content": "My name is Leon."},
            {"role": "assistant", "content": "Nice to meet you, Leon."},
        ]
        extractor = ext.FactExtractor(chat_fn=make_chat_fn("[]"), model="m")
        facts = extractor.extract_with_fallback(convo)
        assert facts and facts[0].origin == "regex"

    def test_llm_nonempty_skips_fallback(self):
        convo = [
            {"role": "user", "content": "My name is Leon."},
            {"role": "assistant", "content": "Nice to meet you, Leon."},
        ]
        extractor = ext.FactExtractor(chat_fn=make_chat_fn(_json([{"fact": BIO, "category": "project"}])), model="m")
        facts = extractor.extract_with_fallback(convo)
        assert facts and facts[0].origin == "llm"
        assert all("name" not in f.text.lower() for f in facts)


# Dedup integration through MemoryStore


class TestStoreIntegration:
    def test_insert_when_new(self, store):
        extractor = ext.FactExtractor(store, chat_fn=make_chat_fn(_json([{"fact": BIO, "category": "project"}])), model="m")
        results = extractor.extract_and_store(CONV, source="conv1")
        assert len(results) == 1
        _, decision = results[0]
        assert decision.action == "insert"
        assert store.count() == 1

    def test_merge_when_duplicate(self, store):
        store.add(KUB_LINUX, "preference")
        assert store.count() == 1
        extractor = ext.FactExtractor(store, chat_fn=make_chat_fn(_json([{"fact": KUB, "category": "preference"}])), model="m")
        results = extractor.extract_and_store(CONV, source="conv1")
        assert len(results) == 1
        _, decision = results[0]
        assert decision.action == "merge"
        # No new row: the near-duplicate merged into the existing fact.
        assert store.count() == 1

    def test_source_recorded(self, store):
        extractor = ext.FactExtractor(store, chat_fn=make_chat_fn(_json([{"fact": PARIS_FACT, "category": "identity"}])), model="m")
        extractor.extract_and_store(CONV, source="conv-xyz")
        rows = store.list()
        assert any(r.source == "conv-xyz" for r in rows)

    def test_at_most_two_facts_stored(self, store):
        reply = _json(
            [
                {"fact": "The user likes apples", "category": "preference"},
                {"fact": "The user likes pears", "category": "preference"},
                {"fact": "The user likes plums", "category": "preference"},
            ]
        )
        extractor = ext.FactExtractor(store, chat_fn=make_chat_fn(reply), model="m")
        results = extractor.extract_and_store(CONV)
        assert len(results) <= ext.MAX_FACTS

    def test_async_wrapper_stores(self, store):
        extractor = ext.FactExtractor(store, chat_fn=make_chat_fn(_json([{"fact": BIO, "category": "project"}])), model="m")
        results = asyncio.run(extractor.aextract_and_store(CONV, source="async"))
        assert len(results) == 1
        assert store.count() == 1

    def test_per_user_scoping(self, tmp_path):
        store = _build_store(tmp_path, single_user_mode=False)[0]
        extractor = ext.FactExtractor(store, chat_fn=make_chat_fn(_json([{"fact": BIO, "category": "project"}])), model="m")
        extractor.extract_and_store(CONV, user_id="alice")
        assert store.count(user_id="alice") == 1
        assert store.count(user_id="bob") == 0


# Model resolution


class TestModelResolution:
    def test_explicit_model_used(self):
        calls: list = []
        extractor = ext.FactExtractor(chat_fn=make_chat_fn("[]", calls=calls), model="my-model")
        extractor.extract(CONV)
        assert calls and calls[0]["model"] == "my-model"

    def test_resolve_model_without_ollama_uses_fallback_head(self, monkeypatch):
        # Force the no-ollama condition explicitly so the test is independent of
        # any sibling test that stubs `ollama` in sys.modules (a known
        # order-dependent suite defect): the resolver then returns the first
        # fallback model.
        monkeypatch.setattr(ext, "OLLAMA_AVAILABLE", False)
        monkeypatch.setattr(ext, "ollama", None)
        extractor = ext.FactExtractor(chat_fn=make_chat_fn("[]"))
        assert extractor._resolve_model() == ext.FALLBACK_MODELS[0]
