#!/usr/bin/env python3
"""What the pre-cache warmer promises about its counting and its restraint.

The warmer walks a configured list of common queries and fills the
semantic cache ahead of demand. Its whole value is in the bookkeeping:
an operator reads the run result to know what the warm actually did, so
the figures have to mean what they say. These contracts pin that
arithmetic and the warmer's restraint from both sides.

Counted honestly. A query that generates and stores lands in ``cached``;
one already present lands in ``skipped`` and its generator is never
invoked; one whose generator raises lands in ``failed`` with its error
recorded, and the run continues past it instead of aborting the batch.
``total`` is the size of the configured list, and for the non-empty
queries the three counters partition exactly what was walked. One
recorded oddity is pinned as it stands rather than judged here: a
query definition with an empty text is walked past without landing in
any counter, so ``total`` includes it while the counters do not -- a
change to that arithmetic must surface as a red here, not slip through.

Restrained by design. With warming disabled the run returns immediately
with the total and zeroed counters and consults nothing. With no
injected generator, no inference client and no model, a query fails
gracefully with a recorded reason instead of raising. Both response
shapes of the inference client -- the mapping form and the object form
-- yield their content, so a client library upgrade cannot silently turn
every warm into a failure.

Loaded through the shared isolation window with the cache stood in by a
counting seam and the inference client scripted or proven absent per
contract. Configuration comes from a per-test file, so the shipped
defaults never leak into an assertion. No real cache database, no model
and no network is ever reached.
"""

import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.pre_cache"


class _CacheSeam:
    """A stand-in semantic cache that records every write."""

    def __init__(self, preloaded=()):
        self.known = set(preloaded)
        self.puts = []
        self.gets = 0

    def get(self, query, model=""):
        self.gets += 1
        return object() if query in self.known else None

    def put(self, **kwargs):
        self.puts.append(kwargs)
        return "stored"


def _write_config(queries, **overrides):
    """Write a throwaway warm configuration and return its path."""
    import yaml

    cfg_dir = Path(tempfile.mkdtemp(prefix="pre_cache_cfg_"))
    payload = {"enabled": True, "default_model": "", **overrides,
               "queries": queries}
    path = cfg_dir / "warm.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _load(*, ollama=None):
    """Load the real module; ``ollama`` seeds or (None) removes the client."""
    had = "ollama" in sys.modules
    prev = sys.modules.get("ollama")
    if ollama is None:
        sys.modules["ollama"] = None
    else:
        sys.modules["ollama"] = ollama

    loaded, win_restore = isolate(targets={_TARGET: source("pre_cache.py")})

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return loaded[_TARGET], restore


def _q(text, model="", task_type="general"):
    return {"query": text, "task_type": task_type, "model": model}


# ---------------------------------------------------------------------------
# p1 -- an uncached query is generated once and stored, and counted as cached
# ---------------------------------------------------------------------------

def test_p1_uncached_queries_are_generated_and_counted_as_cached():
    module, restore = _load()
    try:
        seam = _CacheSeam()
        cfg = _write_config([_q("Hello"), _q("Explain this code", task_type="code")])
        warmer = module.PreCache(config_path=cfg, cache=seam)

        calls = []

        def generate(query, model, task_type):
            calls.append((query, model, task_type))
            return f"answer to {query}"

        result = warmer.warm_common_queries(generate_fn=generate)

        assert result.total == 2
        assert result.cached == 2
        assert result.skipped == 0 and result.failed == 0
        assert len(calls) == 2
        assert len(seam.puts) == 2
        stored = seam.puts[0]
        assert stored["query"] == "Hello"
        assert stored["response"] == "answer to Hello"
        assert stored["metadata"]["source"] == "pre_cache"
    finally:
        restore()


# ---------------------------------------------------------------------------
# p2 -- an already-cached query is skipped and its generator never invoked
# ---------------------------------------------------------------------------

def test_p2_already_cached_queries_are_skipped_without_generation():
    module, restore = _load()
    try:
        seam = _CacheSeam(preloaded={"Hello"})
        cfg = _write_config([_q("Hello"), _q("What can you do?")])
        warmer = module.PreCache(config_path=cfg, cache=seam)

        calls = []

        def generate(query, model, task_type):
            calls.append(query)
            return "fresh answer"

        result = warmer.warm_common_queries(generate_fn=generate)

        assert result.skipped == 1
        assert result.cached == 1
        assert calls == ["What can you do?"], (
            "a query already in the cache must not be regenerated"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# p3 -- a raising generator is one failure with its reason, not an abort
# ---------------------------------------------------------------------------

def test_p3_a_raising_generator_is_counted_failed_and_the_run_continues():
    module, restore = _load()
    try:
        seam = _CacheSeam()
        cfg = _write_config([_q("broken one"), _q("healthy two")])
        warmer = module.PreCache(config_path=cfg, cache=seam)

        def generate(query, model, task_type):
            if query == "broken one":
                raise RuntimeError("generator exploded")
            return "fine"

        result = warmer.warm_common_queries(generate_fn=generate)

        assert result.failed == 1
        assert result.cached == 1, "the run must continue past a failed query"
        assert any("generator exploded" in err for err in result.errors)
    finally:
        restore()


# ---------------------------------------------------------------------------
# p4 -- the counters partition the non-empty queries and total is the list size
# ---------------------------------------------------------------------------

def test_p4_counters_partition_the_walked_queries():
    module, restore = _load()
    try:
        seam = _CacheSeam(preloaded={"kept"})
        cfg = _write_config([_q("kept"), _q("made"), _q("boom")])
        warmer = module.PreCache(config_path=cfg, cache=seam)

        def generate(query, model, task_type):
            if query == "boom":
                raise ValueError("no")
            return "made it"

        result = warmer.warm_common_queries(generate_fn=generate)

        assert result.total == 3
        assert (result.cached, result.skipped, result.failed) == (1, 1, 1)
        assert result.cached + result.skipped + result.failed == result.total
    finally:
        restore()


# ---------------------------------------------------------------------------
# p5 -- the empty-query oddity is pinned exactly as it stands today
# ---------------------------------------------------------------------------

def test_p5_an_empty_query_is_in_the_total_but_in_no_counter():
    module, restore = _load()
    try:
        seam = _CacheSeam()
        cfg = _write_config([_q(""), _q("real question")])
        warmer = module.PreCache(config_path=cfg, cache=seam)

        result = warmer.warm_common_queries(
            generate_fn=lambda q, m, t: "answer"
        )

        # Pinned as recorded: the empty definition inflates the total but is
        # walked past before any counter, so the partition excludes it.
        assert result.total == 2
        assert result.cached == 1
        assert result.skipped == 0 and result.failed == 0
        assert result.cached + result.skipped + result.failed == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# p6 -- both client response shapes yield their content
# ---------------------------------------------------------------------------

def test_p6_mapping_and_object_client_responses_both_yield_content():
    responses = {
        "dict question": {"message": {"content": "FROM THE MAPPING"}},
        "object question": types.SimpleNamespace(
            message=types.SimpleNamespace(content="FROM THE OBJECT")
        ),
    }

    ollama_stub = types.ModuleType("ollama")

    def chat(model, messages, options):
        return responses[messages[0]["content"]]

    ollama_stub.chat = chat

    module, restore = _load(ollama=ollama_stub)
    try:
        seam = _CacheSeam()
        cfg = _write_config(
            [_q("dict question", model="m1"), _q("object question", model="m1")]
        )
        warmer = module.PreCache(config_path=cfg, cache=seam)

        result = warmer.warm_common_queries()

        assert result.cached == 2 and result.failed == 0
        bodies = {p["query"]: p["response"] for p in seam.puts}
        assert bodies["dict question"] == "FROM THE MAPPING"
        assert bodies["object question"] == "FROM THE OBJECT"
    finally:
        restore()


# ---------------------------------------------------------------------------
# p7 -- no generator anywhere is a recorded graceful failure
# ---------------------------------------------------------------------------

def test_p7_no_available_generator_fails_gracefully_with_a_reason():
    module, restore = _load(ollama=None)
    try:
        seam = _CacheSeam()
        cfg = _write_config([_q("orphan question", model="some-model")])
        warmer = module.PreCache(config_path=cfg, cache=seam)

        result = warmer.warm_common_queries()

        assert result.failed == 1 and result.cached == 0
        assert any("No generator available" in err for err in result.errors)
        assert seam.puts == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# p8 -- disabled warming returns immediately and consults nothing
# ---------------------------------------------------------------------------

def test_p8_disabled_warming_returns_totals_and_touches_nothing():
    module, restore = _load()
    try:
        seam = _CacheSeam()
        cfg = _write_config([_q("never walked")], enabled=False)
        warmer = module.PreCache(config_path=cfg, cache=seam)

        called = []
        result = warmer.warm_common_queries(
            generate_fn=lambda q, m, t: called.append(q) or "x"
        )

        assert result.total == 1
        assert (result.cached, result.skipped, result.failed) == (0, 0, 0)
        assert called == [] and seam.gets == 0 and seam.puts == []
        assert result.duration_ms >= 0.0
        assert warmer.last_result is result
    finally:
        restore()
