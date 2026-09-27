#!/usr/bin/env python3
"""The note action that claimed the web is gone, and none claims a source.

A note action labelled "Fact-check with web" asked the model to verify a
selection "against current web sources and cite them". It was one
``generate`` call with no search behind it: an invitation to invent
citations, under a label that promised a web check. It is removed outright,
not renamed: a checker worth the name has to search and cite for real, and
that is a later piece of work with its own proof. Its sibling, the local
fact-check, was sourceless too, so its label says "(no sources)".

The Daily-only gate stays as a mechanism: no action reaches the web today,
so the set of web actions is empty, and a later sourced verifier joins it
and inherits the gate. The old id is refused as unknown, by the runner and
by the route, and the model is never called for it.

  * NA1 -- one contract, eleven clauses, each asserted in order:
      c1  the set of web actions is empty;
      c2  the backend's action set is exactly the five local actions, with
          no constant and no instruction left for a removed one;
      c3  no instruction asks for a source in any word of a class (citing,
          a source, a reference, a link or URL, the web, a search, looking
          up, checking against), each word proven findable; the local
          fact-check says it works from the model's own knowledge, and
          carries the two sentences that forbid browsing and citing;
      c4  the gate is kept: an action planted in the web set is refused
          outside Daily without a model call, and reaches the model in Daily;
      c5  the runner refuses a removed id as unknown in both modes, before
          any model call, and so does the message builder;
      c6  the HTTP route does the same, in Daily;
      c7  the frontend's actions are the backend's five, read in any quote
          style, every entry and every union member accounted for;
      c8  the frontend's labels claim no source and no web, in any word of
          the class, "(no sources)" allowed only as the fact-check's suffix;
      c9  the removed names are gone from the package (modules,
          configuration and documents, its data directory never listed),
          the frontend sources, the documentation and the phone app, and
          the note-action surfaces name no critical review;
      c10 the route keeps its half of the gate: the handler's mode is the
          live-mode seam's, and a web action posted outside Daily is refused
          without a model call;
      c11 the live-mode seam reads the security mode, and an empty, missing,
          unreadable or unreachable mode is Bulbe.

The exact action set of c2 is expected to be superseded, by name, by the
block that adds a sourced action.

Loaded through the shared isolation window; the model is a scripted callable.
"""

import inspect
import os
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_na1_the_web_note_action_is_gone_and_no_action_claims_a_source": 2.0,
}

_WRAPPER = "opti_oignon.agent.untrusted_context"
_ACTIONS = "opti_oignon.agent.note_actions"
_SCHEMAS = "opti_oignon.api.schemas"
_ROUTE = "opti_oignon.api.routes_note_actions"
_MODE = "opti_oignon.security_mode"
_TS = REPO / "frontend" / "src" / "lib" / "api" / "noteActions.ts"

# The note-action surfaces: the runner, the route, the client and the panel.
_SURFACES = (
    REPO / "opti_oignon" / "agent" / "note_actions.py",
    REPO / "opti_oignon" / "api" / "routes_note_actions.py",
    _TS,
    REPO / "frontend" / "src" / "lib" / "components" / "panels" / "NoteActionPanel.svelte",
)

# What an instruction must never ask of a model that has no source.
_SOURCED_PHRASES = ("web search", "web sources", "cite them", "its source", "current web")

# The class of words an instruction or a label uses to ask for, or to promise,
# a source: citing, a source, a reference, a link, the web, a search, looking
# something up, checking against something. Matched as words, in any case.
_SOURCE_WORDS = re.compile(
    r"\b(?:"
    r"cit(?:e|es|ed|ing|ations?)"
    r"|sourc(?:e|es|ed|ing)"
    r"|referenc(?:e|es|ed|ing)"
    r"|urls?"
    r"|(?:hyper)?link(?:s|ed|ing)?"
    r"|bibliograph\w*"
    r"|footnotes?"
    r"|web\w*"
    r"|online"
    r"|internet"
    r"|googl\w*"
    r"|wikipedia"
    r"|(?:re)?search\w*"
    r"|brows(?:e|es|ed|ing|er)"
    r"|look\w*(?:\s+\w+)?\s+up"
    r"|lookups?"
    r"|(?:verif\w*|check\w*|cross-check\w*)\s+(?:it\s+|them\s+|this\s+|each\s+)?(?:against|online|with)"
    r"|according\s+to"
    r")\b",
    re.IGNORECASE,
)

# One planted phrase per branch of the class above: each must be found.
_SOURCE_WITNESSES = (
    "cite", "citations", "sources", "sourced", "a reference", "the URL",
    "a link", "hyperlinks", "bibliography", "a footnote", "the web", "a website",
    "online", "the internet", "google it", "Wikipedia", "search", "research",
    "browse", "look it up", "a lookup", "verify against", "check online",
    "cross-check with", "according to",
)

# The two sentences by which the local fact-check says it has no source. They
# are the only place the class above may appear in an instruction: each is a
# prohibition, and each is required, word for word, in the fact-check.
_NO_WEB = "Do not browse the web."
_NO_CITATION = (
    "Do not cite sources, references, links or URLs, and do not claim to have "
    "looked anything up."
)

# The one place the class may appear in a label: this exact suffix.
_NO_SOURCES_SUFFIX = " (no sources)"

# The instruction the removed action carried, kept as the witness that the
# helper below can find what it looks for.
_OLD_INSTRUCTION = (
    "You are fact-checking the user's note with web search. Verify the "
    "claims in the untrusted-data block below against current web sources "
    "and cite them. List each notable claim with a verdict and its source."
)

# The three ways a review found a sourced instruction could slip past a list
# of the old sentence's phrases.
_SLIPPED_INSTRUCTIONS = (
    "List each notable claim with a verdict and cite a source for each.",
    "List each notable claim with a verdict and a reference URL.",
    "Add structure, supporting points with citations, and next steps.",
)

# The two ids a removed action has had: the original, and the rename that was
# proposed for it and withdrawn.
_REMOVED = ("fact_check_web", "critical_review")

# Every spelling of a removed action the tree used, anywhere in the sources.
_REMOVED_NAMES = (
    "fact-check with web", "fact-check-with-web", "fact_check_web",
    "critical_review", "critical-review",
)

# The withdrawn rename's words, charged on the note-action surfaces only:
# elsewhere they are ordinary English.
_REVIEW_WORDS = ("critical review",)

_EXPECTED = {"fact_check", "develop", "summarize", "rewrite", "make_checklist"}

_SELECTION = "The moon is made of cheese."


def _load_actions():
    loaded, restore = isolate(
        targets={
            _WRAPPER: source("agent", "untrusted_context.py"),
            _ACTIONS: source("agent", "note_actions.py"),
        },
        packages=("opti_oignon.agent",),
    )
    return loaded[_ACTIONS], restore


def _load_route(security_mode=None):
    """The route module, over the real runner and the real schemas.

    The auth dependency and the registry's client are declared unreachable.
    The live security mode is unreachable too unless ``security_mode`` is a
    stand-in for it: the route then reads its mode from that stand-in when it
    is asked for the live mode, and from the arguments the contract passes
    otherwise.
    """
    blocked = ["opti_oignon.api.routes_auth", "opti_oignon.registry_clients"]
    seeded = {}
    if security_mode is None:
        blocked.append(_MODE)
    else:
        seeded[_MODE] = security_mode
    loaded, restore = isolate(
        targets={
            _WRAPPER: source("agent", "untrusted_context.py"),
            _ACTIONS: source("agent", "note_actions.py"),
            _SCHEMAS: source("api", "schemas.py"),
            _ROUTE: source("api", "routes_note_actions.py"),
        },
        blocked=tuple(blocked),
        seeded=seeded,
        packages=("opti_oignon.agent", "opti_oignon.api"),
    )
    return loaded[_ROUTE], loaded[_SCHEMAS], loaded[_ACTIONS], restore


def _asks_for_sources(text):
    low = text.lower()
    return [p for p in _SOURCED_PHRASES if p in low]


def _source_words(text):
    """Every word of the source class in ``text``."""
    return [m.group(0) for m in _SOURCE_WORDS.finditer(text)]


def _instruction_source_words(instruction):
    """The source words of an instruction, outside its two prohibitions."""
    return _source_words(instruction.replace(_NO_WEB, " ").replace(_NO_CITATION, " "))


def _names_removed(text):
    low = text.lower()
    return [p for p in _REMOVED_NAMES if p in low]


def _names_review(text):
    low = text.lower()
    return [p for p in _REVIEW_WORDS if p in low]


def _package_files(pkg_dir):
    """Every module under ``pkg_dir``, never listing the package's data directory.

    The data directory holds no module, and a walk into it is a path the
    test session's firewall has to keep off the maintainer's data.
    """
    return _source_files(pkg_dir, (".py",))


# The trees c9 reads, and the kinds of file it reads in each. The package's
# data directory is never listed; no directory named ``data`` is.
_SCANNED = (
    (REPO / "opti_oignon", (".py", ".yaml", ".yml", ".json", ".toml", ".md")),
    (REPO / "frontend" / "src", (".ts", ".js", ".svelte", ".json")),
    (REPO / "docs", (".md",)),
    (REPO / "android", (".kt", ".kts", ".md")),
)
_SKIPPED_DIRS = {"data", "__pycache__", "node_modules", "build", ".gradle"}


def _source_files(root, suffixes):
    """Every file under ``root`` with one of ``suffixes``, never in a data directory."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIPPED_DIRS)
        for name in sorted(filenames):
            if name.endswith(suffixes):
                yield Path(dirpath) / name


class _Scripted:
    def __init__(self):
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return "A reply."


# One entry of NOTE_ACTIONS, in any of the three quote styles TypeScript takes,
# with or without a trailing comma. An entry this cannot read is still counted
# by its ``kind`` key, so it is refused rather than skipped.
_TS_ENTRY = re.compile(
    r"\{\s*kind\s*:\s*(['\"`])([a-z_]+)\1\s*,"
    r"\s*label\s*:\s*(['\"`])((?:(?!\3).)*)\3\s*,"
    r"\s*requiresWeb\s*:\s*(true|false)\s*,?\s*\}"
)
_TS_MEMBER = re.compile(r"(['\"`])([a-z_]+)\1")


def _ts_block(text):
    return text.split("export const NOTE_ACTIONS", 1)[1].split("];", 1)[0]


def _ts_entries(text):
    return [(m.group(2), m.group(4), m.group(5)) for m in _TS_ENTRY.finditer(_ts_block(text))]


def _ts_kind_keys(text):
    return re.findall(r"\bkind\s*:", _ts_block(text))


def _ts_union_members(text):
    block = text.split("export type NoteActionKind =", 1)[1].split(";", 1)[0]
    return [part.strip() for part in block.split("|") if part.strip()]


def _ts_union(text):
    return {m.group(2) for m in map(_TS_MEMBER.fullmatch, _ts_union_members(text)) if m}


def _c1_no_web_action(mod):
    assert mod.WEB_ACTIONS == frozenset(), sorted(mod.WEB_ACTIONS)


def _c2_the_five_local_actions(mod):
    assert set(mod.ALL_ACTIONS) == _EXPECTED, sorted(mod.ALL_ACTIONS)
    assert set(mod.LOCAL_ACTIONS) == _EXPECTED, sorted(mod.LOCAL_ACTIONS)
    assert not set(_REMOVED) & set(mod.ALL_ACTIONS), sorted(mod.ALL_ACTIONS)
    constants = {name: getattr(mod, name) for name in dir(mod) if name.startswith("ACTION_")}
    assert len(constants) == 5, sorted(constants)
    assert not [name for name, value in constants.items() if value in _REMOVED], constants
    assert set(mod._ACTION_INSTRUCTIONS) == _EXPECTED, sorted(mod._ACTION_INSTRUCTIONS)


def _c3_no_instruction_claims_a_source(mod):
    assert _asks_for_sources(_OLD_INSTRUCTION), "witness: the helper finds a sourced instruction"
    for action, instruction in mod._ACTION_INSTRUCTIONS.items():
        assert not _asks_for_sources(instruction), (action, _asks_for_sources(instruction))
    local = mod._ACTION_INSTRUCTIONS["fact_check"]
    assert "only your own knowledge" in local and "Do not browse the web" in local, local
    # The class of source words, proven able to find each of its branches, the
    # old instruction and every phrasing a review showed slipping past the list.
    missed = [w for w in _SOURCE_WITNESSES if not _source_words(w)]
    assert missed == [], missed
    assert _instruction_source_words(_OLD_INSTRUCTION), "witness: the class finds the old instruction"
    slipped = [s for s in _SLIPPED_INSTRUCTIONS if not _instruction_source_words(s)]
    assert slipped == [], slipped
    assert _source_words(_NO_WEB) and _source_words(_NO_CITATION), "witness: the prohibitions are set aside, not unseen"
    # No instruction asks for a source outside the two prohibitions, and the
    # local fact-check carries both, word for word.
    asking = {a: _instruction_source_words(i) for a, i in mod._ACTION_INSTRUCTIONS.items()}
    assert not [a for a, words in asking.items() if words], asking
    assert _NO_WEB in local, local
    assert _NO_CITATION in local, local


def _c4_the_gate_is_kept(mod):
    """A later sourced verifier joins the web set and inherits the gate."""
    probe = "probe_web_action"
    saved = (mod.WEB_ACTIONS, mod.ALL_ACTIONS, dict(mod._ACTION_INSTRUCTIONS))
    mod.WEB_ACTIONS = frozenset({probe})
    mod.ALL_ACTIONS = mod.LOCAL_ACTIONS | mod.WEB_ACTIONS
    mod._ACTION_INSTRUCTIONS[probe] = "Probe instruction."
    try:
        assert mod.requires_web(probe) is True
        for mode in ("bulbe", "", "unknown"):
            client = _Scripted()
            run = mod.make_note_action_runner(client, mode_provider=lambda m=mode: m)
            refused = run(probe, _SELECTION)
            assert refused.ok is False and refused.refused is True, (mode, refused)
            assert client.calls == [], (mode, "the model is never called for a refused web action")
        client = _Scripted()
        run = mod.make_note_action_runner(client, mode_provider=lambda: "daily")
        admitted = run(probe, _SELECTION)
        assert admitted.ok is True, admitted
        assert len(client.calls) == 1 and client.calls[0][0]["content"] == "Probe instruction."
    finally:
        mod.WEB_ACTIONS, mod.ALL_ACTIONS = saved[0], saved[1]
        mod._ACTION_INSTRUCTIONS.clear()
        mod._ACTION_INSTRUCTIONS.update(saved[2])


def _c5_the_runner_refuses_a_removed_id(mod):
    spellings = [*_REMOVED, *(" " + old.upper() + " " for old in _REMOVED)]
    for mode in ("bulbe", "daily"):
        client = _Scripted()
        run = mod.make_note_action_runner(client, mode_provider=lambda m=mode: m)
        for old in spellings:
            result = run(old, _SELECTION)
            assert result.ok is False and result.refused is False, (mode, old, result)
            assert result.reason.startswith("Unknown action"), (mode, old, result)
        assert client.calls == [], (mode, "the model is never called for an unknown action")
        control = run("fact_check", _SELECTION)
        assert control.ok is True and len(client.calls) == 1, (mode, control)
        assert client.calls[0][0] == {"role": "system", "content": mod._ACTION_INSTRUCTIONS["fact_check"]}
    for old in _REMOVED:
        with pytest.raises(ValueError):
            mod.build_messages(old, _SELECTION)


def _c6_the_route_refuses_a_removed_id(route, schemas):
    assert route.FEATURE_AVAILABLE is True, "control: the route found the runner"
    client = _Scripted()
    built = []

    def builder(model):
        built.append(model)
        return client

    def post(action):
        request = schemas.NoteActionRequest(action=action, selection=_SELECTION, model="m")
        return route.run_note_action(request, build_client=builder, mode="daily", current_user={})

    for old in _REMOVED:
        result = post(old)
        assert result.ok is False and result.refused is False, (old, result)
        assert result.reason.startswith("Unknown action"), (old, result)
    assert client.calls == [], "the route never reaches the model for a removed id"
    control = post("fact_check")
    assert control.ok is True and len(client.calls) == 1, control
    assert built == ["m"] * (len(_REMOVED) + 1), built


def _c7_the_frontend_offers_the_backend_actions(text):
    entries = _ts_entries(text)
    assert len(entries) == 5, entries
    assert {kind for kind, _label, _web in entries} == _EXPECTED, entries
    assert _ts_union(text) == _EXPECTED, _ts_union(text)
    # Every entry and every member is read, whatever its quotes: the parser is
    # proven on the other two styles, and an entry it cannot read is counted.
    planted = (
        'export const NOTE_ACTIONS = [\n'
        '\t{ kind: "a_b", label: "A", requiresWeb: true },\n'
        '\t{ kind: `c_d`, label: `C`, requiresWeb: false, },\n'
        '\t{ label: \'E\', kind: \'e_f\', requiresWeb: false }\n];'
    )
    assert _ts_entries(planted) == [("a_b", "A", "true"), ("c_d", "C", "false")], _ts_entries(planted)
    assert len(_ts_kind_keys(planted)) == 3, "witness: an entry the parser cannot read is still counted"
    union = 'export type NoteActionKind =\n\t| "a_b"\n\t| `c_d`\n\t| string;'
    assert _ts_union_members(union) == ['"a_b"', "`c_d`", "string"], _ts_union_members(union)
    assert _ts_union(union) == {"a_b", "c_d"}, _ts_union(union)
    assert len(_ts_kind_keys(text)) == len(entries), (_ts_kind_keys(text), entries)
    assert "..." not in _ts_block(text), "no entry is spread in from elsewhere"
    members = _ts_union_members(text)
    assert len(members) == 5 and all(_TS_MEMBER.fullmatch(m) for m in members), members


def _c8_the_frontend_labels_claim_no_source(text):
    entries = _ts_entries(text)
    labels = {kind: label for kind, label, _web in entries}
    assert labels.get("fact_check") == "Fact-check (no sources)", labels
    assert not [label for label in labels.values() if re.search("web", label, re.I)], labels
    assert entries and all(web == "false" for _kind, _label, web in entries), entries
    # No label names a source, a search or the web, in any word of the class;
    # "(no sources)" is allowed only as the exact suffix.
    assert _source_words("Develop with sources") and _source_words("Summarize and verify online"), "witness"
    bare = {k: (v[: -len(_NO_SOURCES_SUFFIX)] if v.endswith(_NO_SOURCES_SUFFIX) else v) for k, v in labels.items()}
    assert not {k: _source_words(v) for k, v in bare.items() if _source_words(v)}, bare
    assert not [v for v in bare.values() if "no sources" in v.lower()], bare


def _c9_the_removed_names_are_gone():
    assert _names_removed("A button labelled Fact-check with web."), "witness: a planted name is found"
    assert _names_removed("kind: 'critical_review'"), "witness: a planted id is found"
    assert _names_review("one of six actions -- critical review, develop"), "witness: a planted word is found"
    scanned = 0
    found = []
    for path in _package_files(REPO / "opti_oignon"):
        scanned += 1
        if _names_removed(path.read_text(encoding="utf-8", errors="ignore")):
            found.append(path.relative_to(REPO).as_posix())
    for pattern in ("*.ts", "*.svelte"):
        for path in sorted((REPO / "frontend" / "src").rglob(pattern)):
            scanned += 1
            if _names_removed(path.read_text(encoding="utf-8", errors="ignore")):
                found.append(path.relative_to(REPO).as_posix())
    assert scanned > 300, scanned
    assert found == [], found
    surfaces = {path.relative_to(REPO).as_posix(): path.read_text(encoding="utf-8") for path in _SURFACES}
    assert all(surfaces.values()), sorted(surfaces)
    assert not [name for name, text in surfaces.items() if _names_review(text)], sorted(surfaces)
    # The wider sweep: the package's configuration and documents, the
    # frontend's scripts and data, the user and API documentation, the phone
    # app. Each tree is proven read, and each kind of file in it that exists.
    per_suffix = {}
    in_data = []
    for root, suffixes in _SCANNED:
        for path in _source_files(root, suffixes):
            if "data" in path.relative_to(REPO).parts:
                in_data.append(path.relative_to(REPO).as_posix())
            key = (root.relative_to(REPO).as_posix(), path.suffix)
            per_suffix[key] = per_suffix.get(key, 0) + 1
            if _names_removed(path.read_text(encoding="utf-8", errors="ignore")):
                found.append(path.relative_to(REPO).as_posix())
    assert found == [], found
    assert per_suffix.get(("opti_oignon", ".py"), 0) > 300, per_suffix
    assert per_suffix.get(("opti_oignon", ".yaml"), 0) > 20, per_suffix
    assert per_suffix.get(("opti_oignon", ".json"), 0) > 5, per_suffix
    assert per_suffix.get(("frontend/src", ".ts"), 0) > 50, per_suffix
    assert per_suffix.get(("frontend/src", ".svelte"), 0) > 50, per_suffix
    assert per_suffix.get(("docs", ".md"), 0) > 20, per_suffix
    assert per_suffix.get(("android", ".kt"), 0) > 0, per_suffix
    assert in_data == [], in_data


def _plant_web_action(mod, probe):
    """Put ``probe`` in the web set of ``mod``; returns what puts it back."""
    saved = (mod.WEB_ACTIONS, mod.ALL_ACTIONS, dict(mod._ACTION_INSTRUCTIONS))
    mod.WEB_ACTIONS = frozenset({probe})
    mod.ALL_ACTIONS = mod.LOCAL_ACTIONS | mod.WEB_ACTIONS
    mod._ACTION_INSTRUCTIONS[probe] = "Probe instruction."

    def unplant():
        mod.WEB_ACTIONS, mod.ALL_ACTIONS = saved[0], saved[1]
        mod._ACTION_INSTRUCTIONS.clear()
        mod._ACTION_INSTRUCTIONS.update(saved[2])

    return unplant


def _c10_the_route_keeps_the_gate(route, schemas, actions):
    """The route's half of the gate a later sourced verifier inherits.

    The handler hands the runner the mode its dependency resolved, and that
    dependency is the live-mode seam: a web action posted outside Daily is
    refused by the route without a model call, and reaches the model in Daily.
    """
    runner = route.make_note_action_runner
    assert runner.__globals__ is vars(actions), "witness: the route runs the loaded runner"
    mode_param = inspect.signature(route.run_note_action).parameters["mode"]
    assert getattr(mode_param.default, "dependency", None) is route._mode_dep, mode_param
    probe = "probe_web_action"
    unplant = _plant_web_action(actions, probe)
    try:
        for mode in ("bulbe", "", "unknown"):
            client = _Scripted()
            request = schemas.NoteActionRequest(action=probe, selection=_SELECTION, model="m")
            refused = route.run_note_action(request, build_client=lambda _m, c=client: c, mode=mode, current_user={})
            assert refused.ok is False and refused.refused is True, (mode, refused)
            assert client.calls == [], (mode, "the route never reaches the model for a refused web action")
        client = _Scripted()
        request = schemas.NoteActionRequest(action=probe, selection=_SELECTION, model="m")
        admitted = route.run_note_action(request, build_client=lambda _m: client, mode="daily", current_user={})
        assert admitted.ok is True and admitted.refused is False, admitted
        assert len(client.calls) == 1 and client.calls[0][0]["content"] == "Probe instruction.", client.calls
    finally:
        unplant()
    assert actions.WEB_ACTIONS == frozenset(), "the probe is taken back out"


def _c11_the_route_reads_the_live_mode_fail_secure(route, live):
    """The live-mode seam reads security_mode, and anything else is Bulbe."""

    def raising():
        raise RuntimeError("mode store unreadable")

    cases = (
        (lambda: "daily", "daily"),
        (lambda: "  DAILY ", "daily"),
        (lambda: "bulbe", "bulbe"),
        (lambda: "", "bulbe"),
        (lambda: None, "bulbe"),
        (raising, "bulbe"),
    )
    for reader, expected in cases:
        live.get_current_mode = reader
        assert route._live_mode() == expected, (expected, route._live_mode())
        assert route._mode_dep() == expected, (expected, route._mode_dep())
    del live.get_current_mode
    assert route._live_mode() == "bulbe", "a mode module without its reader is Bulbe"
    assert route._mode_dep() == "bulbe", "a mode module without its reader is Bulbe"


def test_na1_the_web_note_action_is_gone_and_no_action_claims_a_source():
    mod, restore = _load_actions()
    try:
        _c1_no_web_action(mod)
        _c2_the_five_local_actions(mod)
        _c3_no_instruction_claims_a_source(mod)
        _c4_the_gate_is_kept(mod)
        _c5_the_runner_refuses_a_removed_id(mod)
    finally:
        restore()

    route, schemas, actions, restore = _load_route()
    try:
        _c6_the_route_refuses_a_removed_id(route, schemas)
        _c10_the_route_keeps_the_gate(route, schemas, actions)
    finally:
        restore()

    live = types.ModuleType(_MODE)
    route, schemas, actions, restore = _load_route(security_mode=live)
    try:
        _c11_the_route_reads_the_live_mode_fail_secure(route, live)
    finally:
        restore()

    text = _TS.read_text(encoding="utf-8")
    _c7_the_frontend_offers_the_backend_actions(text)
    _c8_the_frontend_labels_claim_no_source(text)
    _c9_the_removed_names_are_gone()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
