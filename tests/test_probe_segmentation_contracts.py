#!/usr/bin/env python3
"""Contracts for how the probe generator reads a text: lists, fenced code, inline code.

A probe is a question a summary must still answer. Drawn from raw text, the
generator asked the wrong questions: a list's numbering became a number to
keep, an identifier inside code an entity, a fenced block a heap of sentences
no summary should repeat, and a list read as one sentence merged the words
and negations of every item. The generator now reads a text as blocks first:
prose, list items without their markers, fenced code. A fenced block is an
artifact: its key is a digest of its body, the librarian's model reads the
marker in its place, the marker answers for it, and the user recalls the
block by its key.

  * SM1 -- a list item is its own unit and its marker, bullet or number, is
    never a probe.
  * SM2 -- a fenced block yields one code probe whose answer is its marker,
    and nothing inside the fence is probed.
  * SM3 -- inline code yields no date, number or entity, and a flag inside it
    does not set a decision's polarity; the decision keeps its words.
  * SM4 -- a code probe is answered by its marker or by the same block
    verbatim, and by nothing else.
  * SM5 -- the librarian hands its model the marker in place of each fenced
    block: the model never reads the code, and is told to copy markers.
  * SM6 -- recall by key returns the block from the Cellar and changes
    nothing; a malformed, unknown or ambiguous key is refused by name.
  * SM7 -- a summary written as a list is scored item by item: a negation in
    one item does not invert the decision of another.
  * SM8 -- a native twin that does not declare the generator's version is
    never asked to draw or score; one that declares it is asked, and answers.
  * SM9 -- masking replaces each fence whole and nothing else; an unclosed
    fence runs to the end of the text; a fence closes only on a fence of its
    own character at least as long; a backtick fence whose info string holds
    a backtick is no fence.
  * SM10 -- a blank line ends a prose unit; a single line break joins, as in
    Markdown.
  * SM11 -- what the generator draws on a fixed corpus is pinned with its
    version: changing what it draws without raising the version turns this
    red.
  * SM12 -- a fence is read within its piece by every reader: an unclosed
    fence in a typed question ends at its piece, the summariser is handed
    the marker the probe asks for, the second face holds its key, and the
    block is recalled by it.

Local-only (the public distribution ships no tests).
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_MEMORY = ("probes", "core_store", "receipts", "composer", "peels", "librarian")

_BULLET = chr(0x2022)
_LIST = (
    "Plan for the launch:\n"
    "1. Ship the release on 2026-10-09\n"
    "2) We will not use Docker for the demo\n"
    "- Alice reviews the notes\n"
    "* Bob keeps 3 spare laptops\n"
    f"{_BULLET} Carol books the room"
)
_BODY = "MAX_RETRIES = 42\nClient.connect('2026-01-01')"
_FENCED = f"Here is the fix we agreed on.\n```python\n{_BODY}\n```\nIt runs on Linux."
_INLINE = "Set `MAX_RETRIES = 5` in `Config` and Alice keeps 7 nodes."
_INLINE_FLAG = "We will use `--no-cache` for the builds."
_PARAGRAPHS = "Launch plan\n\nWe will use Docker."

# The digest of what the generator draws on the corpus of SM11, by version. A
# new version adds its line; a line is never rewritten.
_PINS = {
    2: "497955ed0d18d5f7fb7be6293c3b685c9296e7db1f18cf6b82ee2df783eeef87",
    3: "497955ed0d18d5f7fb7be6293c3b685c9296e7db1f18cf6b82ee2df783eeef87",
    4: "b1afe9c878ebf69079b40171d0f20ea507e2d9d185a9ec9b374fa51766593f43",
    5: "b1afe9c878ebf69079b40171d0f20ea507e2d9d185a9ec9b374fa51766593f43",
    6: "b1afe9c878ebf69079b40171d0f20ea507e2d9d185a9ec9b374fa51766593f43",
    7: "b1afe9c878ebf69079b40171d0f20ea507e2d9d185a9ec9b374fa51766593f43",
}


def _key(body):
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def _probes():
    loaded, restore = isolate(
        targets={"opti_oignon.memory.probes": source("memory", "probes.py")},
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.probes"], restore


def _memory():
    return isolate(
        targets={f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _MEMORY},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils"),
        packages=("opti_oignon.memory",),
    )


def _span(*texts):
    return [{"turn_id": f"t{i}", "role": "user", "origin": "typed", "text": t} for i, t in enumerate(texts, 1)]


def _answers(drawn, kind):
    return [p.answer for p in drawn if p.kind == kind]


def _faithful(turns):
    return " ".join(str(t.get("text", "")) for t in turns)


# ---------------------------------------------------------------------------
# SM1 -- a list item is its own unit, and its marker is never a probe
# ---------------------------------------------------------------------------
def test_sm1_a_list_item_is_its_own_unit_and_its_marker_is_never_a_probe():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span(_LIST))
        numbers = _answers(drawn, "number")
        assert "1" not in numbers and "2" not in numbers, f"a list's numbering is not a number to keep: {numbers}"
        assert "3" in numbers and _answers(drawn, "date") == ["2026-10-09"], "the items' own figures are drawn"
        decisions = [p for p in drawn if p.kind == "decision"]
        assert [p.answer for p in decisions] == ["We will not use Docker for the demo"], "the item alone, without its marker"
        assert decisions[0].key == frozenset({"will", "use", "docker", "demo"}) and decisions[0].negations == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM2 -- a fenced block yields one code probe, and nothing inside is probed
# ---------------------------------------------------------------------------
def test_sm2_a_fenced_block_yields_one_code_probe_and_nothing_inside_it_is_probed():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span(_FENCED))
        assert _answers(drawn, "code") == [f"[code:{_key(_BODY)}]"], "one probe, answered by the block's marker"
        others = [p.answer for p in drawn if p.kind != "code"]
        for inside in ("42", "2026-01-01", "Client", "MAX_RETRIES", "connect"):
            assert not any(inside in answer for answer in others), f"{inside!r} lives in the code: {others}"
        assert "Linux" in _answers(drawn, "entity"), "the prose after the fence is still read"
        code = [p for p in drawn if p.kind == "code"][0]
        assert (code.turn_id, code.origin, code.role) == ("t1", "typed", "user"), "a code probe names its turn and origin"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM3 -- inline code is never probed, and sets no polarity
# ---------------------------------------------------------------------------
def test_sm3_inline_code_yields_no_date_number_or_entity_and_sets_no_polarity():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span(_INLINE))
        numbers, entities = _answers(drawn, "number"), _answers(drawn, "entity")
        assert "5" not in numbers and "7" in numbers, f"numbers read outside the code only: {numbers}"
        assert "Config" not in entities and "Alice" in entities, f"entities read outside the code only: {entities}"
        decisions = [p for p in mod.generate_probes(_span(_INLINE_FLAG)) if p.kind == "decision"]
        assert len(decisions) == 1 and decisions[0].negations == 0, "a flag in code is not a negation of the decision"
        assert {"cache", "builds"} <= decisions[0].key, "the decision keeps the words of what it decided"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM4 -- a code probe is answered by its marker or by the same block
# ---------------------------------------------------------------------------
def test_sm4_a_code_probe_is_answered_by_its_marker_or_the_same_block_and_by_nothing_else():
    mod, restore = _probes()
    try:
        probe = [p for p in mod.generate_probes(_span(_FENCED)) if p.kind == "code"][0]
        marker = f"[code:{_key(_BODY)}]"
        assert mod.answers(probe, f"The fix is {marker}, on Linux.")
        assert mod.answers(probe, _FENCED), "the verbatim answers for itself"
        assert mod.answers(probe, f"Again:\n~~~\n{_BODY}\n~~~"), "the same block under another fence"
        other = marker[:-2] + ("1" if marker[-2] == "0" else "0") + "]"
        for wrong in ("", f"code:{_key(_BODY)}", other, _FENCED.replace("42", "43"), f"The fix:\n```\n{_BODY} \n```"):
            assert not mod.answers(probe, wrong), f"{wrong!r} does not answer for the block"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM5 -- the librarian's model reads the marker, never the code
# ---------------------------------------------------------------------------
class _Backend:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(content="A summary.")


def test_sm5_the_librarian_hands_its_model_the_marker_in_place_of_each_fenced_block():
    loaded, restore = _memory()
    try:
        lib = loaded["opti_oignon.memory.librarian"]
        backend = _Backend()
        cfg = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=4,
                                  temperature=0.1, num_predict=128)
        summarize = lib.registry_summarizer(cfg, resolve=lambda model: backend)
        summarize(_span(_FENCED, "No fence here: `x = 1` stays inline."))
        (call,) = backend.calls
        system, user = call["messages"]
        lines = [json.loads(line) for line in user["content"].split("\n")]
        assert [line["text"] for line in lines] == [
            f"Here is the fix we agreed on.\n[code:{_key(_BODY)}]\nIt runs on Linux.",
            "No fence here: `x = 1` stays inline.",
        ], "each fence, fences included, becomes its marker; nothing else moves"
        assert "MAX_RETRIES" not in json.dumps(call["messages"]), "the model never reads the code"
        assert "[code:" in system["content"], "the model is told what a marker is"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM6 -- recall by key reads the Cellar and changes nothing
# ---------------------------------------------------------------------------
_CODE_A = "def ship(release):\n    return release.tag"
_CODE_B = "SELECT name FROM nodes WHERE alive = 1"
_TURNS = (
    f"Alice wrote the deploy helper on 2026-03-01.\n```python\n{_CODE_A}\n```\nIt stays in the tools folder.",
    "Noted: the helper stays in the tools folder.",
    f"Here is the query for the 12 nodes.\n```sql\n{_CODE_B}\n```\nCarol runs it every morning.",
    "Noted: Carol runs the query every morning.",
)


def _evicted(loaded, cid="c1"):
    lib = loaded["opti_oignon.memory.librarian"]
    peels = loaded["opti_oignon.memory.peels"]
    cfg = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1,
                              temperature=0.1, num_predict=64)
    gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    state = lib.state_for(cid, config=cfg)
    roles = ("user", "assistant")
    origins = ("typed", "assistant")
    state.mirror([{"role": roles[i % 2], "origin": origins[i % 2], "content": t} for i, t in enumerate(_TURNS)])
    while state.flesh.turns():
        step = peels.evict_gated(flesh=state.flesh, cellar=state.cellar, ledger=state.ledger,
                                 tree=state.tree, gate=gate, summarize=_faithful)
        assert step.evicted, step.reason
    return lib, state, cfg


def test_sm6_recall_by_key_reads_the_block_from_the_cellar_and_refuses_by_name():
    loaded, restore = _memory()
    try:
        lib, state, cfg = _evicted(loaded)
        held = [(k, state.cellar.get(k)) for k in state.cellar.keys()]
        receipts = [(r.key, r.resolved) for r in state.ledger.all()]
        got = lib.recall_code("c1", f"code:{_key(_CODE_A)}", config=cfg)
        assert got == {"key": f"code:{_key(_CODE_A)}", "language": "python", "code": _CODE_A}
        assert lib.recall_code("c1", f"code:{_key(_CODE_B)}", config=cfg)["code"] == _CODE_B
        assert [(k, state.cellar.get(k)) for k in state.cellar.keys()] == held, "the Cellar is unchanged"
        assert [(r.key, r.resolved) for r in state.ledger.all()] == receipts, "no receipt moves"
        for key, named in (
            ("code:XYZ", "is not a code key"),
            (_key(_CODE_A), "is not a code key"),
            (f"code:{_key(_CODE_A).upper()}", "is not a code key"),
            ("code:" + "f" * 12, "no code block behind code:" + "f" * 12),
        ):
            with pytest.raises(KeyError) as refused:
                lib.recall_code("c1", key, config=cfg)
            assert named in str(refused.value), f"{key!r} is refused by name, got {refused.value}"
        with pytest.raises(KeyError, match="no onion state"):
            lib.recall_code("nobody", f"code:{_key(_CODE_A)}", config=cfg)
        probes = loaded["opti_oignon.memory.probes"]
        probes.code_key = lambda body: "0" * 12
        with pytest.raises(KeyError, match="2 different blocks"):
            lib.recall_code("c1", "code:" + "0" * 12, config=cfg)
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM7 -- a summary written as a list is scored item by item
# ---------------------------------------------------------------------------
def test_sm7_a_summary_written_as_a_list_is_scored_item_by_item():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span("We will use Docker for the demo."))
        assert [p.kind for p in drawn] == ["entity", "decision"]
        listed = "- We will use Docker for the demo\n- We will not ship on Friday"
        assert mod.score(drawn, listed).failed == 0, "each item is a sentence of its own"
        inverted = "- We will not use Docker for the demo\n- We ship on Friday"
        assert [p.kind for p in mod.score(drawn, inverted).failures] == ["decision"], "and an inverted item still fails"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM8 -- a twin of another generator version is never asked
# ---------------------------------------------------------------------------
class _Twin:
    """A native core that draws nothing and fails nothing, and records whether it was asked."""

    def __init__(self, version):
        if version is not None:
            self.probe_generator_version = version
        self.asked = []

    def probe_generate(self, *args):
        self.asked.append("generate")
        return []

    def probe_score(self, *args):
        self.asked.append("score")
        return []


def test_sm8_a_twin_that_does_not_declare_the_generator_version_is_never_asked():
    mod, restore = _probes()
    try:
        assert type(mod.GENERATOR_VERSION) is int and mod.GENERATOR_VERSION >= 2
        span = _span(_FENCED, _LIST)
        mod._native = lambda: None
        reference = mod.generate_probes(span)
        verdict = mod.score(reference, "Nothing kept.")
        assert reference and verdict.failed > 0, "the reference draws and fails something here"
        for declared in (None, 1, mod.GENERATOR_VERSION - 1, str(mod.GENERATOR_VERSION), float(mod.GENERATOR_VERSION)):
            twin = _Twin(declared)
            mod._native = lambda: twin
            assert mod.generate_probes(span) == reference, f"a twin declaring {declared!r} does not draw"
            assert mod.score(reference, "Nothing kept.") == verdict, f"a twin declaring {declared!r} does not score"
            assert twin.asked == [], f"a twin declaring {declared!r} is never asked"
        twin = _Twin(mod.GENERATOR_VERSION)
        mod._native = lambda: twin
        assert mod.generate_probes(span) == [] and mod.score(reference, "Nothing kept.").failed == 0
        assert twin.asked == ["generate", "score"], "a twin of the current version is asked, and its answer stands"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM9 -- masking replaces each fence whole, and nothing else
# ---------------------------------------------------------------------------
def test_sm9_masking_replaces_each_fence_whole_and_nothing_else():
    mod, restore = _probes()
    try:
        nl = chr(10)
        for plain in ("", "No fence here.", "Inline ```js``` stays.", "```js```", "    ```\n    indented\n    ```",
                      "Use `a` and ``b`c``."):
            assert mod.mask_code(plain) == plain, f"{plain!r} holds no fence"
        assert mod.mask_code(_FENCED) == f"Here is the fix we agreed on.\n[code:{_key(_BODY)}]\nIt runs on Linux."
        assert mod.mask_code("Intro\n```\nline one\nline two") == f"Intro\n[code:{_key('line one' + nl + 'line two')}]", (
            "an unclosed fence runs to the end of the text"
        )
        assert mod.mask_code("~~~~\nx\n~~~~~\nafter") == f"[code:{_key('x')}]\nafter", "a longer closing fence closes"
        assert mod.mask_code("````\na\n```\nb\n````") == f"[code:{_key('a' + nl + '```' + nl + 'b')}]", "a shorter one does not"
        assert mod.mask_code("~~~\na\n```\nb\n~~~") == f"[code:{_key('a' + nl + '```' + nl + 'b')}]", "nor one of the other character"
        blocks = mod.segment(_FENCED)
        assert [(b.kind, b.info) for b in blocks] == [("prose", ""), ("code", "python"), ("prose", "")]
        assert _FENCED[blocks[1].start:blocks[1].end] == f"```python\n{_BODY}\n```", "the block's bounds hold its fences"
        assert [b.text for b in blocks] == ["Here is the fix we agreed on.", _BODY, "It runs on Linux."]
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM10 -- a blank line ends a prose unit
# ---------------------------------------------------------------------------
def test_sm10_a_blank_line_ends_a_prose_unit():
    mod, restore = _probes()
    try:
        decisions = [p for p in mod.generate_probes(_span(_PARAGRAPHS)) if p.kind == "decision"]
        assert [(p.answer, p.key) for p in decisions] == [("We will use Docker.", frozenset({"will", "use", "docker"}))]
        joined = [p for p in mod.generate_probes(_span("Launch plan\nWe will use Docker.")) if p.kind == "decision"]
        assert [p.answer for p in joined] == ["Launch plan\nWe will use Docker."], "a single line break joins"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM11 -- what the generator draws is pinned with its version
# ---------------------------------------------------------------------------
def _canonical(drawn):
    rows = [[p.kind, p.answer, p.turn_id, sorted(p.key), p.negations, p.origin, p.role] for p in drawn]
    return json.dumps(rows, ensure_ascii=True, separators=(",", ":"))


def test_sm11_what_the_generator_draws_is_pinned_with_its_version():
    mod, restore = _probes()
    try:
        mod._native = lambda: None
        corpus = (_LIST, _FENCED, _INLINE, _INLINE_FLAG, _PARAGRAPHS, "Alice moved to Berlin on 2024-03-15.",
                  "The team agreed not to use Docker.")
        drawn = mod.generate_probes(_span(*corpus))
        assert {p.kind for p in drawn} == {"code", "date", "number", "entity", "decision"}, "the corpus draws every kind"
        digest = hashlib.sha256(_canonical(drawn).encode("ascii")).hexdigest()
        assert mod.GENERATOR_VERSION == max(_PINS), "the version is the last one pinned"
        assert _PINS[mod.GENERATOR_VERSION] == digest, (
            "the generator draws something else than its version pinned: raise GENERATOR_VERSION and pin the new digest"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# SM12 -- a fence is read within its piece by every reader
# ---------------------------------------------------------------------------
_UNCLOSED = "Run this first:\n\n```\nprint('setup')"
_PIECE_GAP = "\n\n---\nDocument provided:\n"
_PIECE_DOC = "The report is due on 2026-11-01 with 3 owners."


class _Recording:
    def __init__(self):
        self.calls = []

    def generate(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
        self.calls.append(messages)
        return types.SimpleNamespace(content="summary")


def test_sm12_a_fence_is_read_within_its_piece_by_every_reader():
    loaded, restore = _memory()
    try:
        probes = loaded["opti_oignon.memory.probes"]
        lib = loaded["opti_oignon.memory.librarian"]
        peels = loaded["opti_oignon.memory.peels"]
        probes._native = lambda: None
        content = _UNCLOSED + _PIECE_GAP + _PIECE_DOC
        segments = [[0, len(_UNCLOSED), "typed"], [len(_UNCLOSED) + len(_PIECE_GAP), len(content), "document"]]
        turn = {"turn_id": "t1", "role": "user", "origin": "typed", "text": content, "segments": segments}
        marker = probes.code_marker("print('setup')")
        drawn = probes.generate_probes([turn])
        assert [p.answer for p in drawn if p.kind == "code"] == [marker], "control: the probe reads the fence in its piece"
        assert "2026-11-01" in [p.answer for p in drawn if p.kind == "date"], "control: the document is read as itself"
        backend = _Recording()
        cfg = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1,
                                  temperature=0.1, num_predict=64)
        lib.registry_summarizer(cfg, resolve=lambda model: backend)([turn])
        quoted = [m for m in backend.calls[0] if m["role"] == "user"][0]["content"]
        assert marker in quoted and "2026-11-01" in quoted, "the summariser is handed the probe's marker"
        assert marker[6:-1] in probes.holdings([turn]).keys, "the second face holds the probe's key"
        state = lib.state_for("c12", config=cfg)
        state.mirror([{"role": "user", "origin": "typed", "segments": segments, "content": content}])
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=1)
        told = lambda turns: "Run this first: " + marker + " " + _PIECE_DOC  # noqa: E731
        step = peels.evict_gated(flesh=state.flesh, cellar=state.cellar, ledger=state.ledger, tree=state.tree,
                                 gate=gate, summarize=told)
        assert step.evicted, step.reason
        assert lib.recall_code("c12", "code:" + marker[6:-1], config=cfg)["code"] == "print('setup')"
    finally:
        restore()
