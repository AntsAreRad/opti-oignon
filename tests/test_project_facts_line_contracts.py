#!/usr/bin/env python3
"""Contracts for the line under an answer that says where its project's context came from.

The server keeps with each answer of a project turn the facts of its project's context (``project_facts``, block
268: the source of the passages kept, their number, the form of what the turn said; never a name), carries them in
the ``done`` frame and gives them back with the conversation's messages. The chat said them only in passing, in
status lines its loader replaces one by one. Now the answer says them again under itself, in the loader's own words,
after a reload too.

  * PU1 -- EACH FACT IS SAID IN THE TABLE'S WORDS: ``projectFactsLines`` (``lib/chat/loaderWords.ts``) says the
    passages with their source (one passage as one), a text cut to fit, withheld or left out, then what the turn
    said (every closed form), each in the very words the loader gave the turn's own status line; an answer that kept
    no passage says only why, or what the turn said; files counted, never named; a form the table does not know says
    nothing of itself; anything that is not the facts' closed shape (a name of the object's own prototype among them,
    a count that is not whole, a number out of bounds) says nothing of it, and never throws; every word passes both
    of the garden's nets, printable ASCII.
  * PU2 -- THE FORMS ARE THE SERVER'S: the forms the line knows and the numbers each keeps are those the server
    keeps (``opti_oignon/project_facts.py``), and so are the sources.
  * PU3 -- THE REPLY CARRIES THEM: the reply a ``done`` frame describes (``responseOf``) carries the facts it brought
    (nothing when it brought none, or not an object), and the end of a reply keeps them on the answer, after a
    retry's reload too -- but not a reply that was stopped (review 269-1): the server saves no stopped reply, and a
    retry's reload would lay its facts under the answer before it.
  * PU4 -- THE LINE IS UNDER THE ANSWER: ``ChatMessage.svelte`` shows ``ProjectContextLine`` under every finished
    answer, from its ``project_context``, which the message type carries; the line, rendered, says the table's words
    and nothing else (nothing at all for no facts), in the run summary's own size and colour, with no inline
    style.
  * PU5 -- THE TEXTS: the CHANGELOG says it.

Local-only (the public distribution ships no tests). The TypeScript runs under Node (``tests/_frontend.py``,
``run_ts``), the nets through the loader contracts' own window; the server's module is read from its file.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_chat_loader_contracts as _ld  # noqa: E402
from _frontend import run_ts, ssr  # noqa: E402
from _isolation import REPO  # noqa: E402

BUDGET_S = {
    "test_pu1_each_fact_is_said_in_the_tables_words": 10.0,
    "test_pu2_the_forms_are_the_servers": 2.0,
    "test_pu3_the_reply_carries_them": 8.0,
    "test_pu4_the_line_is_under_the_answer": 2.0,
    "test_pu5_the_texts": 2.0,
}

_LIB = REPO / "frontend" / "src" / "lib"
_DRIVER = r"""
const W = await import(process.env.OO_WORDS);
const S = await import(process.env.OO_STREAM);
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT || 'null');
let out = null;
if (clause === 'facts') {
  out = input.map((facts) => W.projectFactsLines(facts).map((line) => W.said(line)));
} else if (clause === 'statuses') {
  out = input.map((status) => { const line = W.statusLine(status); return line ? W.said(line) : null; });
} else if (clause === 'reply') {
  out = input.map((meta) => S.responseOf({ type: 'done', content: 'An answer.', metadata: meta }, 'c1').project_context ?? null);
}
console.log('RESULT ' + JSON.stringify(out));
console.log('PASS ' + clause);
"""
_MODULES = {"OO_WORDS": _ld._MODULES["OO_WORDS"], "OO_STREAM": _ld._MODULES["OO_STREAM"]}


def _node(clause, data):
    out = run_ts(_MODULES, _DRIVER, clause, env={"OO_INPUT": json.dumps(data)})
    results = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, f"the driver printed no single RESULT line:\n{out}"
    return json.loads(results[0][len("RESULT "):])


def _facts_module():
    spec = importlib.util.spec_from_file_location("project_facts_under_contract",
                                                  REPO / "opti_oignon" / "project_facts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PF = _facts_module()
_LEFT = ("the project holds 2 file(s) its Mycelium store does not (scan.pdf, minutes-2024.docx): those are left to "
         "the knowledge base")


def test_pu1_each_fact_is_said_in_the_tables_words():
    unread = "the project's Mycelium store cannot be read (DatabaseError): no project passage"
    lexical = "the search took longer than 2.5 s: the lexical stages answered"
    cases = [
        (PF.facts("mycelium", 3),
         ["[OK] Project context injected: 3 chunks, ~40 tokens, from the project's Mycelium store"]),
        (PF.facts("rag", 1), ["[OK] Project context injected: 1 chunks, ~9 tokens, from the knowledge base"]),
        (PF.facts("mycelium+rag", 4, _LEFT),
         ["[OK] Project context injected: 4 chunks, ~50 tokens, from the project's Mycelium store and the knowledge "
          "base", f"[!] Project context: {_LEFT}"]),
        (PF.facts("cut", 4), ["[OK] Project context injected: 4 chunks, ~50 tokens, cut to its share of the window"]),
        (PF.facts("withheld", 0), ["[!] Project context withheld: the untrusted-data wrapper is unavailable"]),
        (PF.facts("left_out", 0), ["[!] Project context left out: its share of the window holds no chunk"]),
        (PF.facts("none", 0, unread), [f"[!] Project context: {unread}"]),
        (PF.facts("mycelium", 1, lexical),
         ["[OK] Project context injected: 1 chunks, ~9 tokens, from the project's Mycelium store",
          f"[!] Project context: {lexical}"]),
        (PF.facts("none", 0), []),
    ]
    said = _node("facts", [facts for facts, _statuses in cases])
    lines = _node("statuses", [s for _facts, statuses in cases for s in statuses])
    at = 0
    for (facts, statuses), got in zip(cases, said):
        want = lines[at:at + len(statuses)]
        at += len(statuses)
        assert all(want), f"control: the loader says each status of the turn: {statuses} -> {want}"
        assert got == want, f"{facts}: the answer says what the turn's lines said: {got} != {want}"
    assert said[0] == ["Project context: 3 passages from its Mycelium store."], said[0]
    assert said[1] == ["Project context: 1 passage from the knowledge base."], said[1]
    unknown = _node("facts", [PF.facts("mycelium", 2, "a new form that names notes.md")])[0]
    assert unknown == ["Project context: 2 passages from its Mycelium store."], (
        f"a form the table does not know says nothing of itself: {unknown}")
    failed = "the project's Mycelium search failed (TimeoutError)"
    serves = "the project holds files its Mycelium store does not: the knowledge base serves it"
    both = "the search took longer than 10 s: the lexical stages answered; " + _LEFT
    more = [(PF.facts("none", 0, failed), [f"[!] Project context: {failed}"]),
            (PF.facts("rag", 2, serves), ["[OK] Project context injected: 2 chunks, ~9 tokens, from the knowledge base",
                                          f"[!] Project context: {serves}"]),
            (PF.facts("mycelium", 1, both), ["[OK] Project context injected: 1 chunks, ~9 tokens, from the project's "
                                             "Mycelium store", f"[!] Project context: {both}"]),
            (PF.facts("withheld", 0, _LEFT), ["[!] Project context withheld: the untrusted-data wrapper is unavailable",
                                              f"[!] Project context: {_LEFT}"])]
    got = _node("facts", [facts for facts, _s in more])
    want = _node("statuses", [s for _f, statuses in more for s in statuses])
    assert all(want) and got == [want[0:1], want[1:3], want[3:5], want[5:7]], (got, want)
    said += got
    crafted = _node("facts", [
        {"source": "__proto__", "passages": 1}, {"source": "constructor", "passages": 1},
        {"source": "toString", "passages": 1},
        {"source": "rag", "passages": 1e21}, {"source": "rag", "passages": 2, "said": {"form": "lexical", "s": -1}},
        {"source": "rag", "passages": 2, "said": {"form": "left", "files": 2.5}},
        {"source": "rag", "passages": 2, "said": {"form": "lexical", "s": 1e21}},
        {"source": "rag", "passages": 2, "said": {"form": "__proto__"}},
        {"source": "rag", "passages": 2, "said": {"form": "constructor"}},
        {"source": "rag", "passages": 2, "said": {"form": "toString"}}])
    only = ["Project context: 2 passages from the knowledge base."]
    assert crafted == [[], [], [], []] + [only] * 6, f"what is not the facts' shape says nothing of it: {crafted}"
    zero = _node("facts", [PF.facts("cut", 0), PF.facts("mycelium", 0)])
    assert zero == [[], []], f"an answer that kept no passage says nothing of passages: {zero}"
    wrong = _node("facts", [{"source": "the store at /home/x", "passages": 2, "said": None},
                            {"source": "rag", "passages": -1}, {"source": "rag", "passages": 1.5},
                            {"source": "rag", "passages": "2"}, {"source": "rag"}, "scan.pdf", None, [1]])
    assert wrong == [[]] * 8, f"anything that is not the facts' closed shape says nothing: {wrong}"
    smuggled = _node("facts", [{"source": "none", "passages": 0, "said": {"form": "left", "files": "scan.pdf"}},
                               {"source": "none", "passages": 0, "said": {"form": "left", "files": 2,
                                                                         "names": "scan.pdf"}}])
    every = [w for group in said + smuggled + [unknown] for w in group]
    assert not [w for w in every if "scan.pdf" in w or "minutes-2024" in w or "notes.md" in w], every
    ethics, restore = _ld._nets()
    try:
        assert ethics.class_hits("Thinking"), "control: the class net refuses a governed word"
        assert not [w for w in every if ethics.class_hits(w) or ethics.banned_hits(w)], every
    finally:
        restore()
    assert not [w for w in every if re.search(r"[^ -~]", w)], every


def test_pu2_the_forms_are_the_servers():
    written = (_LIB / "chat" / "loaderWords.ts").read_text(encoding="utf-8")
    forms = written.split("const PROJECT_FORMS", 1)[1].split("};", 1)[0]
    theirs = dict(re.findall(r"(\w+): \[([^\]]*)\]", forms))
    ours = {form: names for _p, form, names in PF.SAID}
    assert set(theirs) == set(ours), (theirs, ours)
    for form, names in ours.items():
        assert re.findall(r"'(\w+)'", theirs[form]) == list(names), (form, theirs[form], names)
    sources = written.split("const PROJECT_SOURCE_KEYS", 1)[1].split("};", 1)[0]
    keyed = set(re.findall(r"^\s*'?([\w+]+)'?:", sources, re.M))
    assert keyed | {"withheld", "left_out", "none"} == set(PF.SOURCES), (keyed, PF.SOURCES)


def test_pu3_the_reply_carries_them():
    facts = PF.facts("mycelium", 3, _LEFT)
    got = _node("reply", [{"project_context": facts}, {}, {"project_context": "scan.pdf"}])
    assert got == [facts, None, None], got
    store = (_LIB / "stores" / "chat.ts").read_text(encoding="utf-8")
    end = store.split("function endOf(", 1)[1].split("\n}\n", 1)[0]
    assert "if (response.project_context && !end.stopped) end.project_context = response.project_context;" in end, (
        f"a stopped reply, which the server does not save, keeps no facts: {end}")
    assert end.index("end.stopped = true") < end.index("end.project_context"), "control: stopped is known first"
    assert "{ ...msg, ...end }" in store, "control: a retry's reload keeps the end of the reply on its answer"


def test_pu4_the_line_is_under_the_answer():
    message = (_LIB / "components" / "chat" / "ChatMessage.svelte").read_text(encoding="utf-8")
    assert "import ProjectContextLine from './ProjectContextLine.svelte';" in message
    footer = message.split("{#if !isUser && !isStreaming}\n\t\t\t<RunSummary", 1)
    assert len(footer) == 2, "control: the run's summary sits under every finished answer"
    assert footer[1].split("{/if}", 1)[0].count("<ProjectContextLine facts={message.project_context} />") == 1
    line = (_LIB / "components" / "chat" / "ProjectContextLine.svelte").read_text(encoding="utf-8")
    assert "import { projectFactsLines, said } from '$lib/chat/loaderWords';" in line
    assert "style=" not in line and "{@html" not in line, "no inline style, no raw markup"
    scoped = line.split("<style>", 1)[1] if "<style>" in line else ""
    assert "font-size: var(--oo-text-sm);" in scoped and "color: var(--oo-fg-muted);" in scoped, (
        "the run summary's own size and colour")
    path = "frontend/src/lib/components/chat/ProjectContextLine.svelte"
    facts = PF.facts("mycelium+rag", 4, _LEFT)
    html = ssr().render(path, {"facts": facts}).html
    text = re.sub(r"<[^>]+>", "", re.sub(r"<!--.*?-->", "", html)).strip()
    words = _node("facts", [facts])[0]
    assert text == " ".join(words), f"the rendered line says the table's words and nothing else: {text!r}"
    assert "scan.pdf" not in html and "{" not in text, html
    assert re.sub(r"<!--.*?-->", "", ssr().render(path, {"facts": None}).html).strip() == "", "no facts, no line"
    types = (_LIB / "types.ts").read_text(encoding="utf-8")
    item = types.split("export interface MessageItem {", 1)[1].split("\n}\n", 1)[0]
    assert "project_context?: ProjectContextFacts | null;" in item, item


def test_pu5_the_texts():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "PU1-PU5" in e), "").split())
    assert "Under each answer" in entry and "never named" in entry and "reload" in entry, entry[:300]
