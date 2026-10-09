#!/usr/bin/env python3
"""Contracts for the interface of an approval that shows what it approves.

The review of pending writes and the skills panel are where a person reads
what the agent and its teacher proposed, and decides. These contracts pin
what they show and what they send:

  * Contract APU1 -- THE PURE MODULE SHOWS WITHOUT HIDING: a value is shown
    as the server rendered it, else with every character but printable ASCII
    and a line break written as its escape; a skill proposal's view gives
    where it writes, its whole text, what it replaces or deletes, its digest;
    a decision's digests are the shown ones of the proposals it applies to; a
    target changed since, or a text no longer its digest, leaves the list,
    and a digest that no longer names what was shown stays.
  * Contract APU2 -- THE LIST SHOWS A SKILL WHOLE AND NOTHING HIDDEN: a skill
    proposal is rendered with its high risk, where it writes, its whole text
    past the drawer's bound, the text it replaces and its digest; a memory or
    notes value is rendered as shown, or escaped when the server sent no
    rendering; no character a screen hides reaches the page.
  * Contract APU3 -- ONE REQUEST OF IDS AND DIGESTS, TO ROUTES THE SERVER
    SERVES: the review client accepts with the ids and the digests shown, in
    one request per decision, under the fields the server reads; the skills
    client views by status, publishes and adopts by a digest in the body,
    deletes by status and digest in the query, on routes the agent router
    serves with those parameters required. Supersedes PWU3.
  * Contract APU4 -- THE PANEL KEYS EACH ROW BY ITS STATUS AND NAMES EACH
    TARGET: rows are keyed by status, category and name; a row's view, delete,
    publication and adoption send that row's status and the digest of what
    was shown; each delete button names its target; the agent's skill
    proposals are reviewed in the panel through the shared review.

The pure module runs under Node (``_frontend.run_ts``); the list and the
panel are server-rendered (``_frontend.ssr``); the routers are loaded through
the shared isolation window. Local-only. Runs under pytest or the __main__
runner.
"""

import inspect
import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import read, run_ts, ssr  # noqa: E402
from _isolation import isolate, source  # noqa: E402
from test_ui_primitives_contracts import _dom  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file.
BUDGET_S = {
    "test_apu1_the_pure_module_shows_every_value_without_hiding_a_character": 10.0,
    "test_apu2_the_list_shows_a_skill_whole_and_no_character_a_screen_hides": 10.0,
    "test_apu3_the_clients_send_ids_and_digests_to_routes_the_server_serves": 6.0,
    "test_apu4_the_panel_keys_each_row_by_its_status_and_names_each_target": 10.0,
}

_PURE = "frontend/src/lib/pendingWrites.ts"
_CLIENT = "frontend/src/lib/api/pendingWrites.ts"
_SKILLS_CLIENT = "frontend/src/lib/api/skills.ts"
_LIST = "frontend/src/lib/components/panels/PendingWritesList.svelte"
_REVIEW = "frontend/src/lib/components/panels/PendingWritesReview.svelte"
_PANEL = "frontend/src/lib/components/panels/SkillsPanel.svelte"
_APPROVAL = "opti_oignon.tool_call_approval"
_BS = "\\"


def _hidden_left(text):
    from test_provenance_gate_contracts import _hidden

    return _hidden(text)


def _sly():
    """Text a screen draws otherwise than its bytes: a zero-width space, a direction override, Unicode tags, a no-break
    space, a variation selector and an escape it holds as plain text."""
    return ("Say hello" + chr(0x200B) + chr(0x202E) + "".join(chr(0xE0000 + ord(c)) for c in "rm -rf ~")
            + chr(0xA0) + chr(0xFE0F) + _BS + "u200b" + " done")


def _drawer():
    loaded, restore = isolate(targets={_APPROVAL: source("tool_call_approval.py")}, packages=())
    return loaded[_APPROVAL]._visible, restore


def _item(pid, store, action, arguments, *, shown=None, target=None, digest="", source_name="agent", untyped=()):
    item = {"id": pid, "store": store, "action": action, "arguments": arguments,
            "provenance": {"source": source_name, "turn": "none", "typed": {}, "untyped": list(untyped),
                           "target": "name" if store == "skills" and action != "add" else None, "read": []},
            "conversation_id": "conv-1", "run_id": "run-1", "created_at": "2026-10-09T10:00:00+00:00",
            "target": target}
    if shown is not None:
        item["shown"] = shown
    if digest:
        item["digest"] = digest
    if store == "skills":
        item["risk"] = "high"
    return item


# ---------------------------------------------------------------------------
# Contract APU1 -- the pure module shows without hiding
# ---------------------------------------------------------------------------
_DRIVER = r"""
const mod = await import(process.env.OO_PENDING);
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT);
const result = {
  escaped: mod.escapeAll(input.sly),
  plain: mod.escapeAll(input.plain),
  shownWins: mod.shownValue('as the server drew it', input.sly),
  fallback: mod.shownValue(undefined, input.sly),
  views: input.items.map((item) => mod.skillView(item)),
  labels: input.items.map((item) => mod.describeWrite(item)),
  sources: input.items.map((item) => mod.sourceLabel(item)),
  digests: mod.decisionDigests(input.items, ['p3', 'p1', 'gone']),
  left: mod.remaining(input.items, input.results).map((item) => item.id),
};
console.log('RESULT ' + JSON.stringify(result));
console.log('PASS ' + clause);
"""


def test_apu1_the_pure_module_shows_every_value_without_hiding_a_character():
    visible, restore = _drawer()
    try:
        drawn = visible(_sly())
    finally:
        restore()
    items = [
        _item("p1", "skills", "edit", {"category": "general", "name": "greet", "text": _sly(), "sha256": "a" * 64,
                                       "base_sha256": "b" * 64, "source": "agent", "tested": True},
              shown={"arguments": {"text": drawn}, "target": {"text": "Say hi."}},
              target={"text": "Say hi.", "sha256": "b" * 64}, digest="a" * 64),
        _item("p2", "skills", "delete", {"category": "general", "name": "greet", "draft": True,
                                         "base_sha256": "c" * 64},
              shown={"arguments": {}, "target": {"text": "Old draft."}},
              target={"text": "Old draft.", "sha256": "c" * 64}, digest="c" * 64, source_name="teacher"),
        _item("p3", "memory", "add", {"text": "I like tea.", "category": "preference"}, digest="d" * 64),
        _item("p4", "notes", "make", {"title": "Groceries"}),
        _item("p5", "skills", "add", {"category": "ops", "name": "deploy", "text": "Ship it.", "sha256": "e" * 64,
                                      "base_sha256": None, "source": "agent", "tested": False}, digest="e" * 64),
        _item("p6", "skills", "edit", {"category": "ops", "name": "moved", "text": "New.", "sha256": "f" * 64,
                                       "base_sha256": "b" * 64, "source": "agent", "tested": False},
              target={"text": "Changed by another hand.", "sha256": "9" * 64}, digest="f" * 64),
    ]
    results = [{"id": "p1", "applied": False, "reason": "target changed"},
               {"id": "p2", "applied": False, "reason": "digest mismatch"},
               {"id": "p3", "applied": False, "reason": "the digest does not name the text shown"}]
    out = run_ts({"OO_PENDING": _PURE}, _DRIVER, "showing",
                 env={"OO_INPUT": json.dumps({"sly": _sly(), "plain": "Plain words.\nTwo lines.", "items": items,
                                              "results": results})})
    lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, out
    got = json.loads(lines[0][len("RESULT "):])
    escaped = got["escaped"]
    assert all(c == "\n" or " " <= c <= "~" for c in escaped), "the fallback leaves a character a screen may hide"
    assert _BS + _BS + "u200b" in escaped and "rm -rf" not in escaped, escaped
    assert got["plain"] == "Plain words.\nTwo lines.", "control: printable words and line breaks stay as they are"
    assert got["shownWins"] == "as the server drew it" and got["fallback"] == escaped, got
    edit, delete, memory, notes, add, moved = got["views"]
    assert edit == {"where": "general/greet", "draft": False, "text": drawn, "replaces": "Say hi.",
                    "digest": "a" * 64, "tested": True, "changed": False}, edit
    assert delete == {"where": "general/greet", "draft": True, "text": "", "replaces": "Old draft.",
                      "digest": "c" * 64, "tested": False, "changed": False}, delete
    assert memory is None and notes is None, (memory, notes)
    assert add["text"] == "Ship it." and add["replaces"] == "" and add["digest"] == "e" * 64, add
    assert add["changed"] is False and moved["changed"] is True, "a target changed since the proposal is said"
    assert got["labels"][:2] == ["Change a skill", "Delete a skill"] and got["labels"][4] == "Add a skill", got
    assert got["sources"][1] == "Proposed by the teacher model" and got["sources"][0] == "Proposed by the agent", got
    assert got["digests"] == {"p1": "a" * 64, "p3": "d" * 64}, got["digests"]
    assert got["left"] == ["p3", "p4", "p5", "p6"], "a changed target or text leaves, an unnamed digest stays"


# ---------------------------------------------------------------------------
# Contract APU2 -- the list shows a skill whole and nothing hidden
# ---------------------------------------------------------------------------
def _render(props):
    html = ssr().render(_LIST, props).html
    return html, _dom(html)


def _normal(text):
    return " ".join(text.split())


def test_apu2_the_list_shows_a_skill_whole_and_no_character_a_screen_hides():
    visible, restore = _drawer()
    long_text = "## Procedure\n" + _sly() + "\n" + "x" * 3000
    current = "## Procedure\nSay hi" + chr(0x200B) + "."
    try:
        drawn, drawn_current, drawn_fact = visible(long_text), visible(current), visible("I live in Paris" + chr(0x2060))
        bound = 2000
    finally:
        restore()
    items = [
        _item("s1", "skills", "edit", {"category": "general", "name": "greet", "text": long_text, "sha256": "a" * 64,
                                       "base_sha256": "b" * 64, "source": "agent", "tested": False},
              shown={"arguments": {"text": drawn}, "target": {"text": drawn_current}},
              target={"text": current, "sha256": "b" * 64}, digest="a" * 64, untyped=["text"]),
        _item("m1", "memory", "add", {"text": "I live in Paris" + chr(0x2060), "category": "fact"},
              shown={"arguments": {"text": drawn_fact, "category": "fact"}, "target": None}, digest="d" * 64,
              untyped=["text"]),
        _item("n1", "notes", "make", {"title": "Groceries" + chr(0x202E)}, untyped=["title"]),
    ]
    html, root = _render({"items": items, "selected": [], "busy": False})
    rows = {li.get("data-pending-id"): li for li in root.iter("li") if li.get("data-pending-id")}
    assert sorted(rows) == ["m1", "n1", "s1"], sorted(rows)
    skill = rows["s1"]
    said = _normal(skill.text())
    pres = {pre.get("data-pending-text"): pre for pre in skill.iter("pre")}
    assert sorted(pres) == ["current", "proposed"], sorted(pres)
    assert _normal(pres["proposed"].text()) == _normal(drawn) and len(drawn) > bound, "the whole text, never cut"
    assert _normal(pres["current"].text()) == _normal(drawn_current), "what it replaces, as shown"
    assert "Change a skill" in said and "High risk" in said and "general/greet" in said, said
    assert "Digest " + "a" * 64 in said, said
    assert _normal(drawn_fact) in _normal(rows["m1"].text()), "a memory value is rendered as the server drew it"
    assert "Groceries" + _BS + "u202e" in _normal(rows["n1"].text()), "a value with no rendering is escaped"
    assert _hidden_left(html) == [], "a character a screen hides reached the page"


# ---------------------------------------------------------------------------
# Contract APU3 -- one request of ids and digests, to routes the server serves
# ---------------------------------------------------------------------------
_CALL = re.compile(r"\b(apiGet|apiPost|apiDelete)\s*<[^>]*>\s*\(\s*([`'])([^`']+)\2\s*(?:,\s*([^)]*?))?\s*\)")


def _calls(text):
    return {(fn, path): (body or "").strip() for fn, _q, path, body in _CALL.findall(text)}


def test_apu3_the_clients_send_ids_and_digests_to_routes_the_server_serves():
    review_calls = _calls(read(_CLIENT))
    skills_text = read(_SKILLS_CLIENT)
    loaded, restore = isolate(
        targets={"opti_oignon.memory.probes": source("memory", "probes.py"),
                 "opti_oignon.pending_writes": source("pending_writes.py"),
                 "opti_oignon.api.routes_pending_writes": source("api", "routes_pending_writes.py"),
                 "opti_oignon.agent.skills": source("agent", "skills.py"),
                 "opti_oignon.api.routes_agent": source("api", "routes_agent.py")},
        packages=("opti_oignon.memory", "opti_oignon.api", "opti_oignon.agent"))
    try:
        review = loaded["opti_oignon.api.routes_pending_writes"]
        agent = loaded["opti_oignon.api.routes_agent"]
        served = {(method, route.path) for route in review.pending_writes_router.routes for method in route.methods}
        decision_fields = sorted(review.PendingWriteDecision.model_fields)
        agent_routes = {(method, route.path): route for route in agent.router.routes
                        for method in (getattr(route, "methods", None) or ())}
        delete_query = {p.name: p.required for p in agent_routes[("DELETE", "/api/agent/skills/{category}/{name}")]
                        .dependant.query_params}
        view_query = {p.name: p.required for p in agent_routes[("GET", "/api/agent/skills/{category}/{name}")]
                      .dependant.query_params}
        bodies = {path: inspect.signature(agent_routes[("POST", path)].endpoint).parameters["request"].annotation
                  for path in ("/api/agent/skills/{category}/{name}/publish", "/api/agent/skills/{category}/{name}/adopt")}
        body_fields = {path: sorted(getattr(agent, str(getattr(model, "__name__", model))).model_fields)
                       for path, model in bodies.items()}
    finally:
        restore()
    sent = {("GET" if fn == "apiGet" else "POST", path) for fn, path in review_calls}
    assert sent == {("GET", "/api/pending-writes"), ("POST", "/api/pending-writes/accept"),
                    ("POST", "/api/pending-writes/decline")} and sent <= served, (sent, served)
    assert decision_fields == ["digests", "ids"], decision_fields
    assert review_calls[("apiPost", "/api/pending-writes/accept")] == "{ ids, digests }", review_calls
    assert review_calls[("apiPost", "/api/pending-writes/decline")] == "{ ids }", review_calls
    container = read(_REVIEW)
    assert re.search(r"const ids = decisionIds\(items, selected\);", container), "the decision is the selection"
    assert re.search(r"await acceptPendingWrites\(ids, decisionDigests\(items, ids\)\)", container), (
        "an acceptance carries the digests of what was shown, in the one request")
    assert re.search(r"await declinePendingWrites\(ids\)", container), "one request per decision"
    assert not re.search(r"for\s*\(|\.forEach\(|\.map\([^)]*PendingWrites", container), "never id by id"
    assert "apiGet<Skill>(ref(category, name), { status })" in skills_text, "a view names its status"
    assert "apiPost<Skill>(`${ref(category, name)}/publish`, { sha256 })" in skills_text, "a publication its digest"
    assert "apiPost<Skill>(`${ref(category, name)}/adopt`, { sha256 })" in skills_text, "an adoption its digest"
    assert re.search(r"status=\$\{encodeURIComponent\(status\)\}&sha256=\$\{encodeURIComponent\(sha256\)\}",
                     skills_text) and "apiDelete<{ deleted: boolean }>(`${ref(category, name)}?${query}`)" in (
        skills_text), "a delete names its target and its digest"
    assert delete_query == {"status": True, "sha256": True}, delete_query
    assert view_query == {"status": False}, view_query
    assert body_fields == {path: ["sha256"] for path in bodies}, body_fields
    assert ("POST", "/api/agent/skills/{category}/{name}/adopt") in agent_routes, sorted(agent_routes)


# ---------------------------------------------------------------------------
# Contract APU4 -- the panel keys each row by its status and names each target
# ---------------------------------------------------------------------------
def test_apu4_the_panel_keys_each_row_by_its_status_and_names_each_target():
    panel = read(_PANEL)
    skills_text = read(_SKILLS_CLIENT)
    assert re.search(r"\{#each filtered as skill \(skill\.key\)\}", panel), "rows are keyed by their status"
    assert "getSkill(skill.category, skill.name, skill.status)" in panel, "a row's view is that row's item"
    assert "viewed = { ...viewed, [skill.key]: full }" in panel, "a view is kept under the row's own key"
    assert "deleteSkill(skill.category, skill.name, skill.status, shown.sha256)" in panel, (
        "a delete names the row's status and the digest of the text shown")
    assert panel.index("{deleteLabel(skill)}") > panel.index("{#if selectedKey === skill.key}") and (
        "handleDelete(skill, shown)" in panel), "a delete is offered only beside the text it deletes"
    assert "publishSkill(skill.category, skill.name, shown.sha256)" in panel, "a publication the digest shown"
    assert "adoptSkill(skill.category, skill.name, shown.file_sha256 ?? '')" in panel, "an adoption the bytes shown"
    assert "{deleteLabel(skill)}" in panel, "each delete button names its target"
    assert re.search(r"<PendingWritesReview store=\"skills\" on:decided=\{load\} />", panel), (
        "the agent's skill proposals are reviewed in the panel")
    assert "return skill.status === 'draft' ? 'Delete this draft' : `Delete published v${skill.version}`;" in (
        skills_text), "the words of a delete name a draft or a published version"
    assert panel.count("getSkill(") == 1, "no view of a skill without its status"
    html = ssr().render(_PANEL, {}).html
    assert "Filter skills" in html and "<button" in html, "control: the panel renders"


def _run_all():
    cases = (
        ("APU1 the pure module shows without hiding",
         test_apu1_the_pure_module_shows_every_value_without_hiding_a_character),
        ("APU2 the list shows a skill whole", test_apu2_the_list_shows_a_skill_whole_and_no_character_a_screen_hides),
        ("APU3 ids and digests to routes served",
         test_apu3_the_clients_send_ids_and_digests_to_routes_the_server_serves),
        ("APU4 the panel keys rows by status", test_apu4_the_panel_keys_each_row_by_its_status_and_names_each_target),
    )
    failed = 0
    for name, case in cases:
        try:
            case()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
