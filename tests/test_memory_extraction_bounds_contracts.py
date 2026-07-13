#!/usr/bin/env python3
"""Extraction contracts: a bounded, silent, store-mediated capture path.

Background extraction distils at most a couple of short facts from a
conversation and persists each one through the coordinated store, so the
double deduplication and the cross-layer consistency apply to everything it
writes. It must stay conservative and must never break the conversation
path. This suite pins those bounds:

  * EX1 -- the parsed batch is capped at the fact budget, rejects
    over-length and under-length texts, and collapses case duplicates;
  * EX2 -- categories outside the canon coerce to the default and a
    malformed or array-less reply reads as empty;
  * EX3 -- the pattern fallback scans only the user's own turns, anchors
    names and places on a leading capital, and fires only when the model
    path yields nothing;
  * EX4 -- a failing model call is swallowed (an empty result, never an
    exception) and a non-empty model result suppresses the fallback;
  * EX5 -- persistence flows only through the coordinated store's add, with
    the declared provenance, and one failing add does not abort the rest;
  * EX6 -- the transcript is bounded to the most recent turns and skips
    empty contents.

Loads the extraction module in isolation; the model client import is
blocked so resolution stays deterministic. Local-only. Runs under pytest or
the __main__ runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_MEMORY = _REPO / "opti_oignon" / "memory"


def _load():
    """Load the extraction module under a stand-in package.

    Every ``opti_oignon.*`` entry plus the model client entry is snapshotted
    and evicted first so a previously imported real module cannot leak into
    the isolation window, then restored afterwards.
    """
    keys = ["ollama"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # imports of the client fail deterministically

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    memory = types.ModuleType("opti_oignon.memory")
    memory.__path__ = []
    sys.modules["opti_oignon"] = root
    sys.modules["opti_oignon.memory"] = memory

    full = "opti_oignon.memory.extraction"
    spec = importlib.util.spec_from_file_location(full, _MEMORY / "extraction.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    memory.extraction = mod
    spec.loader.exec_module(mod)

    def restore():
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        for k, v in saved.items():
            sys.modules[k] = v

    return mod, restore


class _Store:
    """Recorder coordinated store; failures are injectable per call index."""

    def __init__(self, fail_first=False):
        self.add_calls = []
        self.fail_first = fail_first

    def add(self, text, category="fact", *, source="", user_id=None):
        self.add_calls.append((text, category, source))
        if self.fail_first and len(self.add_calls) == 1:
            raise RuntimeError("injected store failure")
        record = types.SimpleNamespace(id=f"r{len(self.add_calls)}", text=text)
        decision = types.SimpleNamespace(action="insert")
        return record, decision


def _chat_returning(payload):
    def chat(**_kw):
        return {"message": {"content": payload}}

    return chat


def test_ex1_the_parsed_batch_is_capped_and_length_bounded():
    mod, restore = _load()
    try:
        long_fact = "the user " + "really " * 13 + "walks"
        raw = (
            "["
            f'{{"fact": "{long_fact}", "category": "fact"}},'
            ' {"fact": "ab", "category": "fact"},'
            ' {"fact": "The user rows on the river", "category": "fact"},'
            ' {"fact": "THE USER ROWS ON THE RIVER", "category": "fact"},'
            ' {"fact": "The user keeps bees", "category": "fact"},'
            ' {"fact": "The user paints landscapes", "category": "fact"}'
            "]"
        )
        facts = mod.parse_extraction_response(raw)
        assert [f.text for f in facts] == [
            "The user rows on the river",
            "The user keeps bees",
        ]
        assert len(facts) == mod.MAX_FACTS
    finally:
        restore()


def test_ex2_categories_coerce_and_malformed_replies_read_empty():
    mod, restore = _load()
    try:
        got = mod.parse_extraction_response(
            '[{"fact": "The user swims weekly", "category": "banana"}]'
        )
        assert [f.category for f in got] == ["fact"]
        spaced = mod.parse_extraction_response(
            '[{"fact": "The user swims weekly", "category": " Preference "}]'
        )
        assert [f.category for f in spaced] == ["preference"]
        assert mod.parse_extraction_response("no array here") == []
        assert mod.parse_extraction_response('[{"fact": "x", broken') == []
        wrapped = mod.parse_extraction_response(
            '{"facts": [{"fact": "The user rows daily", "category": "fact"}]}'
        )
        assert [f.text for f in wrapped] == ["The user rows daily"]
    finally:
        restore()


def test_ex3_the_fallback_is_user_only_capital_anchored_and_last_resort():
    mod, restore = _load()
    try:
        messages = [
            {"role": "assistant", "content": "My name is Trap and I live in Berlin."},
            {"role": "user", "content": "my name is bob and I am from paris"},
            {"role": "user", "content": "My name is Alice."},
        ]
        facts = mod.regex_fallback(messages)
        assert [f.text for f in facts] == ["The user's name is Alice"]
        assert all(f.origin == "regex" for f in facts)
        # The model path is unavailable here, so the full pipeline lands on
        # the fallback and returns the same conservative capture.
        extractor = mod.FactExtractor(_Store())
        assert extractor.extract(messages) == []
        combined = extractor.extract_with_fallback(messages)
        assert [f.text for f in combined] == ["The user's name is Alice"]
    finally:
        restore()


def test_ex4_model_failures_are_swallowed_and_model_output_wins():
    mod, restore = _load()
    try:
        def bad_chat(**_kw):
            raise RuntimeError("model down")

        messages = [
            {"role": "user", "content": "My name is Bob."},
            {"role": "assistant", "content": "Noted."},
        ]
        broken = mod.FactExtractor(_Store(), chat_fn=bad_chat, model="stub")
        assert broken.extract(messages) == []

        good = mod.FactExtractor(
            _Store(),
            chat_fn=_chat_returning(
                '[{"fact": "The user tends a rooftop garden", "category": "fact"}]'
            ),
            model="stub",
        )
        combined = good.extract_with_fallback(messages)
        assert [f.text for f in combined] == ["The user tends a rooftop garden"]
        assert all("Bob" not in f.text for f in combined)
    finally:
        restore()


def test_ex5_writes_flow_only_through_the_store_and_survive_one_failure():
    mod, restore = _load()
    try:
        payload = (
            '[{"fact": "The user keeps bees", "category": "fact"},'
            ' {"fact": "The user rows daily", "category": "goal"}]'
        )
        store = _Store()
        extractor = mod.FactExtractor(
            store, chat_fn=_chat_returning(payload), model="stub"
        )
        results = extractor.extract_and_store(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            source="capture-test",
        )
        assert store.add_calls == [
            ("The user keeps bees", "fact", "capture-test"),
            ("The user rows daily", "goal", "capture-test"),
        ]
        assert len(results) == 2

        flaky = _Store(fail_first=True)
        extractor2 = mod.FactExtractor(
            flaky, chat_fn=_chat_returning(payload), model="stub"
        )
        results2 = extractor2.extract_and_store(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        )
        assert len(flaky.add_calls) == 2, "one failing add must not abort the rest"
        assert len(results2) == 1
    finally:
        restore()


def test_ex6_the_transcript_is_bounded_and_skips_empty_turns():
    mod, restore = _load()
    try:
        messages = [
            {"role": "user", "content": f"marker{i}"} for i in range(40)
        ]
        messages.append({"role": "user", "content": "   "})
        transcript = mod.format_conversation(messages)
        assert "marker39" in transcript
        assert "marker11" in transcript
        assert "marker9" not in transcript
        assert transcript.count("\n") + 1 == mod.MAX_INPUT_MESSAGES - 1
        both = mod.format_conversation(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        )
        assert both == "User: question\nAssistant: answer"
    finally:
        restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
