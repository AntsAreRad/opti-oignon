"""Contracts of the citation gate: the citations of an agent run's final answer, checked mechanically against the
lines the run's Mycelium search showed it -- their provenance as the verdict (shown, unseen, gone, modified, empty),
the words of the citing sentence the cited lines do not hold as an alert apart, never a "verified" -- and the
paragraph shown after the answer.

  * CI1 -- THE CITATIONS READ: ``[path:first-last]`` and ``[path:line]`` are read, several in one sentence; a URL, a
    dict, a time, lines out of order or a line zero are not citations; a text past its bound is read to the bound;
    two thousand shown paths, short or sharing their heads, read in bounded time (review 252-1).
  * CI2 -- UNSEEN: a file the run was never shown, lines past those it was shown, a path the tool never gave (one
    climbing out of the root) are unseen, never read.
  * CI3 -- GONE, MODIFIED, EMPTY: a cited file removed since it was shown is gone; a cited line rewritten is modified,
    the last of a range as the first; a file shortened under a cited range is modified; cited lines with no word are
    empty.
  * CI4 -- THE WORDS OF THE SENTENCE: an identifier or a number of two digits or more said in the sentence that
    carries a citation and in none of its lines is named in its alert; the words of the sentence before, of another
    citation's part of the sentence, a file's name and a path, are not looked for; one found on any cited line counts.
  * CI5 -- WHAT WAS SHOWN IS WHAT IS KEPT: the lines a cut passage left out read unseen, and those of a passage only
    named past the answer's bound; a long line cut in the answer is kept whole as it was read; the lines a run keeps
    are bounded, and a run past the bound is said.
  * CI6 -- THE VERDICT AFTER THE ANSWER: a run whose search showed passages has its ``done`` event's final text kept
    whole, the paragraph after it (where each citation points, then the alerts; never "verified") and the verdicts
    given; an answer that cites none of them is said to cite none; a run that never searched, or that has no
    workspace, is left as it was, whatever its answer cites.
  * CI7 -- THE GATE NEVER BREAKS THE RUN'S END: a gate that fails leaves the ``done`` event as the loop gave it; the
    verdict paragraph is bounded; the card tells the model where its citations are checked to point, never that they
    are checked word by word.
  * CI8 -- THE WHOLE CLAIM, WORD BY WORD (review 1): the words after a citation, of a sentence it opens, of the
    sentence or code block a lone citation follows, are compared; a claim with nothing to compare raises no alert;
    words are whole tokens (409 is not 4096), acronym-led names and numbers with a unit are read; a call's
    parentheses, a qualified name, line numbers named in prose and digit separators raise no alert on a right
    answer; a shown path with a space or brackets is read; the words path is bounded in time.
  * CI9 -- WHAT THE ALERT READS (review 2): a number inside a cited range (not one of its ends), a decimal under ten,
    a port after a host, a dotted name whose qualifier the lines do not hold, are alerts; an abbreviation, thousands
    written 4,096 or 4 096, a path between backquotes, a domain, are no words.
  * CI10 -- THE LINES THE MODEL WAS GIVEN (reviews 252-1 and 252-2): a line counts as shown once the agent loop handed
    the model the text that holds it whole, as its message holds it: through the real loop, a line its round's budget
    cut from a fourth search's answer reads unseen while a line of the first answer, and one the fourth kept before
    its cut, read shown; what the hook is told is found in the message the model is handed; a corpus line holding the
    untrusted wrapper's marker, which the wrapper defangs with what follows it, leaves the lines after it unseen; an
    answer the handler gave and the loop never delivered shows nothing; an answer delivered cut shows the lines wholly
    before the cut, a line cut short of its end not.
The handler's answers in CI2-CI9 are delivered whole, as the loop gives an answer under its caps.
"""
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_mycelium_agent_contracts as my  # noqa: E402
from test_mycelium_agent_contracts import _cloned, _index  # noqa: E402
from test_mycelium_place_contracts import _folder  # noqa: E402

# The Mycelium agent tool's fixtures: no engine reached, a test data folder and a quick opener.
no_engine = my.no_engine
owned = my.owned

BUDGET_S = {
    "test_ci1_the_citations_read": 3.0,
    "test_ci2_unseen": 4.0,
    "test_ci3_gone_modified_empty": 4.0,
    "test_ci4_the_words_of_the_sentence": 4.0,
    "test_ci5_what_was_shown_is_what_is_kept": 4.0,
    "test_ci6_the_verdict_after_the_answer": 5.0,
    "test_ci7_the_gate_never_breaks_the_runs_end": 4.0,
    "test_ci8_the_whole_claim_word_by_word": 6.0,
    "test_ci9_what_the_alert_reads": 4.0,
    "test_ci10_the_lines_the_model_was_given": 10.0,
}

BUDGET = '''"""The budget of a turn."""


def turn_budget(tokens):
    """The tokens a turn may spend."""
    limit_tokens = 4096
    return min(tokens, limit_tokens)
'''


def _shown_run(owned, tmp_path, files=None, query="turn_budget", k=1, corpus="oo"):
    """A folder indexed, a run's handler that searched it once: ``(handler, answer, root)``."""
    root = _folder(tmp_path / "src", files or {"budget.py": BUDGET})
    _index(owned, corpus, root)
    handle = owned.tool.make_handler(_cloned(tmp_path, "s1", root))
    answer = handle({"query": query, "k": k})
    _delivered(handle, answer)
    return handle, answer, root


def _delivered(handle, answer, given=None):
    """The answer given to the model, whole or as ``given``: what the agent loop tells the handler once its caps
    are applied (a handler with no such hook is left as it is)."""
    getattr(handle, "delivered", lambda answer, given: None)(answer, answer if given is None else given)


def _verdicts(text, handle):
    from opti_oignon.mycelium import gate
    from opti_oignon.mycelium.index import file_lines

    return [(c["citation"], c["verdict"], c.get("missing")) for c in gate.check(text, handle.shown,
                                                                                    file_lines)["checked"]]


def test_ci1_the_citations_read():
    """CI1 -- the citations of a text, in order: brackets with a path and lines; none in a URL, a dict, a time, a
    citation whose lines run backwards or start at zero; a text a million characters long is read to its bound, and
    fifty citations at most are checked, the others counted, in bounded time. A false friend read as a citation, or
    an unbounded read -> RED."""
    from opti_oignon.mycelium import gate

    text = ("See [a.py:3-5] and [b/c.md:7]. Not https://x.org/a:3-5, not {'a':1}, not 10:30-11, not [a.py:5-3], "
            "not [a.py:0], not (a.py:3-5).")
    assert [c[2:] for c in gate.citations(text)] == [("a.py", 3, 5), ("b/c.md", 7, 7)]
    many = "x [a.py:1-2] " * 100_000
    started = time.perf_counter()
    found = gate.citations(many)
    assert abs(len(found) - gate.MAX_TEXT // len("x [a.py:1-2] ")) <= 1, len(found)
    result = gate.check(many, gate.Shown(), lambda root, path: None)
    assert len(result["checked"]) == gate.MAX_CHECKED and result["more"] == len(found) - gate.MAX_CHECKED
    assert time.perf_counter() - started < 2.0

    # Two thousand shown paths, short or sharing their first characters, and an answer of brackets.
    short = [f"{n:x}.py" for n in range(2000)]
    started = time.perf_counter()
    assert gate.citations("[" * 200_000, short) == []
    assert time.perf_counter() - started < 1.0, time.perf_counter() - started
    heads = [f"src/pkg/{n:04d}/m.py" for n in range(2000)]
    started = time.perf_counter()
    assert gate.citations("[src/pkg/" * 20_000, heads) == []
    assert gate.citations("See [src/pkg/0007/m.py:3-4].", heads) == [(4, 27, "src/pkg/0007/m.py", 3, 4)]
    assert time.perf_counter() - started < 1.0, time.perf_counter() - started


def test_ci2_unseen(owned, tmp_path):
    """CI2 -- the run was shown budget.py's lines 1-8: a file it was never shown, lines past 8, a path climbing out
    of the root to the same file, and a file the store holds that no answer gave, are unseen; the file is never read
    for them. An unseen citation read and judged -> RED."""
    handle, answer, root = _shown_run(owned, tmp_path, {"budget.py": BUDGET, "other.py": "x_unused = 1\n"})
    assert "[budget.py:1-8]" in answer, answer
    reads = []
    from opti_oignon.mycelium import gate
    from opti_oignon.mycelium.index import file_lines

    def counted(root, path):
        reads.append(path)
        return file_lines(root, path)

    text = ("Nothing [other.py:1]. Past [budget.py:6-9]. Out [../src/budget.py:1-2]. Absent [none.py:1-2]. "
            "Seen `turn_budget` [budget.py:4].")
    checked = gate.check(text, handle.shown, counted)["checked"]
    assert [c["verdict"] for c in checked] == ["unseen", "unseen", "unseen", "unseen", "shown"], checked
    assert reads == ["budget.py"], reads


def test_ci3_gone_modified_empty(owned, tmp_path):
    """CI3 -- after the run was shown budget.py: its blank lines cited are empty; a line rewritten since is
    modified (a line not rewritten still shown); the file removed, its citation is gone. A changed line still
    shown, or a removed file read as shown -> RED."""
    handle, answer, root = _shown_run(owned, tmp_path)
    assert "[budget.py:1-8]" in answer, answer
    assert _verdicts("Blank [budget.py:2-3].", handle) == [("budget.py:2-3", "empty", None)]
    (root / "budget.py").write_text(BUDGET.replace("4096", "2048"), encoding="utf-8")
    assert _verdicts("It is `limit_tokens` [budget.py:6]. And `turn_budget` [budget.py:4].", handle) == [
        ("budget.py:6", "modified", None), ("budget.py:4", "shown", None)]
    (root / "budget.py").write_text(BUDGET.replace("return min", "return max"), encoding="utf-8")
    assert _verdicts("`turn_budget` [budget.py:4-7].", handle) == [("budget.py:4-7", "modified", None)]
    (root / "budget.py").write_text("\n".join(BUDGET.split("\n")[:5]), encoding="utf-8")
    assert _verdicts("`turn_budget` [budget.py:4-7].", handle) == [("budget.py:4-7", "modified", None)]
    (root / "budget.py").unlink()
    assert _verdicts("`turn_budget` [budget.py:4].", handle) == [("budget.py:4", "gone", None)]


def test_ci4_the_words_of_the_sentence(owned, tmp_path):
    """CI4 -- the identifiers and numbers said with a citation are looked for in its lines: a number or a name the
    lines do not hold is named missing; one on another cited line counts; the words of the sentence before, of
    another citation's part of the same sentence, a file's name and a path are not looked for. A wrong word passed,
    or a word of another sentence required -> RED."""
    handle, answer, root = _shown_run(owned, tmp_path)
    assert "[budget.py:1-8]" in answer, answer
    cases = [
        ("The limit is `limit_tokens`, 4096 tokens [budget.py:6].", [("budget.py:6", "shown", None)]),
        ("`turn_budget` caps at 8192 [budget.py:4-7].", [("budget.py:4-7", "shown", ["8192"])]),
        ("It calls spend_tokens and SpendTokens [budget.py:4-7].",
         [("budget.py:4-7", "shown", ["spend_tokens", "SpendTokens"])]),
        ("The cap is 9999 or spend_tokens. Then `turn_budget` [budget.py:4].", [("budget.py:4", "shown", None)]),
        ("`turn_budget` [budget.py:4] and `limit_tokens` [budget.py:6].",
         [("budget.py:4", "shown", None), ("budget.py:6", "shown", None)]),
        ("In tool.py and opti_oignon/x_y.py, `turn_budget` [budget.py:4].", [("budget.py:4", "shown", None)]),
        ("`turn_budget` returns min of limit_tokens [budget.py:4-7].", [("budget.py:4-7", "shown", None)]),
        ("A turn holds 4096 tokens [budget.py:4].", [("budget.py:4", "shown", ["4096"])]),
    ]
    for text, want in cases:
        assert _verdicts(text, handle) == want, text


def test_ci5_what_was_shown_is_what_is_kept(owned, tmp_path, monkeypatch):
    """CI5 -- a passage the answer's bound cut: its lines left out read unseen, those given shown; a line longer
    than the answer shows is kept as it was read: a word past the cut is found. The lines a run keeps are bounded:
    past the bound, the citations of the lines not kept read unseen and the verdict says the run was shown more than
    the gate keeps. Lines left out read as seen, or an unbounded keep -> RED."""
    from opti_oignon.mycelium import gate

    long_line = "WIDE = '" + "w " * 300 + "far_word'"
    body = "\n".join([f"value_{n} = {n}" for n in range(1, 60)] + [long_line]) + "\n"
    monkeypatch.setattr(owned.tool, "MAX_BYTES", 3600)
    handle, answer, root = _shown_run(owned, tmp_path, {"long.py": body}, query="value_1", k=1)
    assert "left out: the answer's bound" in answer, answer[:600]
    given_last = int(answer.split("[long.py:1-")[1].split("]")[0])
    assert given_last < 59, answer[:600]
    assert _verdicts("`value_1` [long.py:1].", handle) == [("long.py:1", "shown", None)]
    assert _verdicts("`value_59` [long.py:59].", handle) == [("long.py:59", "unseen", None)]

    # Passages only named past the answer's bound: none of their lines was given.
    many = {f"part_{n}.py": "\n".join(f"value_{n}_{i} = {i}" for i in range(1, 40)) + "\n" for n in range(4)}
    handle, answer, root = _shown_run(owned, tmp_path / "named", many, query="value_0_1 value_1_1 value_2_1 value_3_1",
                                      k=4, corpus="named")
    assert "Passages left out by the answer's bound: " in answer, answer[-600:]
    named = answer.rsplit("Passages left out by the answer's bound: ", 1)[1].split(", ")[0]
    path, lines = named.rsplit(":", 1)
    first = int(lines.split("-")[0])
    assert f"[{path}:" not in answer, (named, answer[:400])
    assert _verdicts(f"It is set [{path}:{first}].", handle) == [(f"{path}:{first}", "unseen", None)]

    monkeypatch.setattr(owned.tool, "MAX_BYTES", 14000)
    handle, answer, root = _shown_run(owned, tmp_path / "two", {"wide.py": "\n".join(
        [f"x_{n} = {n}" for n in range(1, 10)] + [long_line]) + "\n"}, query="x_1", k=1, corpus="wide")
    assert "more characters of this line" in answer, answer[:300]
    assert _verdicts("The `far_word` [wide.py:10].", handle) == [("wide.py:10", "shown", None)]

    monkeypatch.setattr(gate, "MAX_KEPT_LINES", 3)
    small = gate.Shown()
    small.keep("/r", "a.py", 1, ["one_a", "two_b", "three_c", "four_d", "five_e"])
    assert small.count == 3 and small.full
    lines = {"a.py": ["one_a", "two_b", "three_c", "four_d", "five_e"]}
    result = gate.check("`one_a` [a.py:1]. `four_d` [a.py:4].", small, lambda root, path: lines.get(path))
    assert [c["verdict"] for c in result["checked"]] == ["shown", "unseen"] and result["full"]
    assert "more lines than the gate keeps" in gate.said(result, True)


class _Recorder:
    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(json.loads(payload))

    def done(self):
        return [p for p in self.payloads if p["kind"] == "done"]


def _run(owned, tmp_path, monkeypatch, final_text, search=True, corpus="oo", sandbox=True):
    """A run started for a workspace whose folder is indexed (none when ``sandbox`` is False); its loop (a stand-in)
    searches once when ``search``, its answer given whole to the model, then ends with ``final_text``: the ``done``
    events the subscribers were sent, and the run's manager."""
    from opti_oignon.api import routes_agent

    root = _folder(tmp_path / "src", {"budget.py": BUDGET})
    _index(owned, corpus, root)
    sandbox = _cloned(tmp_path, "s1", root) if sandbox else None

    def loop(**kwargs):
        if search:
            handle = kwargs["tool_handlers"]["mycelium_search"]
            _delivered(handle, handle({"query": "turn_budget", "k": 1}))
        kwargs["on_event"](routes_agent.agent_loop.AgentEvent(kind="done", round=1, data={"final_text": final_text}))
        return types.SimpleNamespace(stop_reason="done", rounds=1, final_text=final_text)

    monkeypatch.setattr(routes_agent.agent_loop, "run", loop)
    manager = routes_agent.AgentRunManager()
    recorder = _Recorder()
    manager.subscribe(recorder)
    started = manager.start("find the budget", model_client=object(), mode="daily", sandbox=sandbox,
                            include_memory=False, consult=False)
    assert started.get("started") is True, started
    manager.join(timeout=10.0)
    return recorder.done(), manager


def test_ci6_the_verdict_after_the_answer(owned, tmp_path, monkeypatch):
    """CI6 -- a run that searched and answers with citations: the ``done`` final text is the model's text whole, then
    the paragraph (both shown, the second's missing word in the alerts, no "verified"), and the verdicts given; an
    answer citing nothing after passages were shown is said to cite none; a run that never searched sends its text as it was. A
    verdict in place of the text, a run's answer unchecked, or a verdict on a run that saw nothing -> RED."""
    text = "The limit is `limit_tokens` [budget.py:6]. It caps at 8192 [budget.py:4-7]."
    [done], manager = _run(owned, tmp_path, monkeypatch, text)
    final = done["data"]["final_text"]
    assert final.startswith(text + "\n\n"), final
    assert "2 of 2 point to lines the search gave, unchanged" in final and "- budget.py:4-7: 8192" in final, final
    assert "verified" not in final.split(text, 1)[1], final
    checked = done["data"]["citations"]["checked"]
    assert [c["verdict"] for c in checked] == ["shown", "shown"] and checked[1]["missing"] == ["8192"], checked
    assert manager._shown is None, "the run's shown lines outlived it"

    [done], _ = _run(owned, tmp_path / "b", monkeypatch, "The budget is fine.", corpus="b")
    assert done["data"]["final_text"].startswith("The budget is fine.\n\nThis answer gives no citation"), done

    [done], _ = _run(owned, tmp_path / "c", monkeypatch, "The budget is fine.", search=False, corpus="c")
    assert done["data"] == {"final_text": "The budget is fine."}, done

    # A run that never searched, or has no workspace, whose answer cites lines it read another way: left as it was.
    cited = "`turn_budget` caps the tokens [budget.py:4-7]."
    [done], _ = _run(owned, tmp_path / "d", monkeypatch, cited, search=False, corpus="d")
    assert done["data"] == {"final_text": cited}, done
    [done], _ = _run(owned, tmp_path / "e", monkeypatch, cited, search=False, corpus="e", sandbox=False)
    assert done["data"] == {"final_text": cited}, done


def test_ci7_the_gate_never_breaks_the_runs_end(owned, tmp_path, monkeypatch):
    """CI7 -- a gate whose check fails leaves the ``done`` event as the loop gave it; the verdict paragraph is
    bounded, however many citations fail; the card says the final answer's citations are checked against the lines
    shown. A run's end broken by the gate, an unbounded verdict, or a model never told -> RED."""
    from opti_oignon.mycelium import gate

    def broken(*args, **kwargs):
        raise RuntimeError("the gate failed")

    real = gate.check
    gate.check = broken
    try:
        [done], _ = _run(owned, tmp_path, monkeypatch, "The limit [budget.py:6].")
    finally:
        gate.check = real
    assert done["data"] == {"final_text": "The limit [budget.py:6]."}, done

    result = {"checked": [{"citation": "p" * 280 + f"{n}.py:1", "verdict": "unseen"} for n in range(50)],
              "more": 9000, "full": False}
    paragraph = gate.said(result)
    assert len(paragraph) <= gate.MAX_SAID + 10 and paragraph.startswith("Citations checked"), len(paragraph)

    root = _folder(tmp_path / "card", {"budget.py": BUDGET})
    _index(owned, "card", root)
    card = owned.tool.card(_cloned(tmp_path, "s9", root))
    assert "[path:first-last]" in card and "checked against the lines this search gave you" in card, card
    assert "word by word" not in card, card


def test_ci8_the_whole_claim_word_by_word(owned, tmp_path):
    """CI8 (review 1) -- the run was shown budget.py: a claim whose wrong number or name stands after its citation,
    in a sentence the citation opens, in the sentence or the code block a lone citation follows, is named in its
    alert; a claim with no identifier or number raises none; wrong words that are parts of right ones, an
    acronym-led name, a number glued to its unit or to a slash, are named; a call written with its parentheses, a
    qualified name, the line numbers a sentence names, a number with its digit separators dropped, are found; a shown
    path with a space and brackets is read as a citation; fifty citations of a long line in a long answer are checked
    in bounded time. A wrong word never alerted, or a right answer alerted -> RED."""
    from opti_oignon.mycelium import gate

    handle, answer, root = _shown_run(owned, tmp_path)
    assert "[budget.py:1-8]" in answer, answer
    wrong = [
        "`turn_budget` caps at 8192 tokens. [budget.py:4-7]",
        "`turn_budget` caps at 8192 tokens\n[budget.py:4-7]",
        "According to [budget.py:4-7], `turn_budget` caps at 8192 tokens.",
        "[budget.py:4-7] `turn_budget` caps at 8192 tokens.",
        "`turn_budget` caps at 8192, i.e. the cap [budget.py:4-7].",
        "```python\nlimit_tokens = 8192\n```\n[budget.py:6]",
        "The limit is 409 tokens [budget.py:6].",
        "The limit is 8192MB [budget.py:6].",
        "The limit is 8192/turn [budget.py:6].",
        "It reads `limit_token` [budget.py:6].",
        "`turn_budget` raises HTTPException [budget.py:4-7].",
    ]
    for text in wrong:
        [(_c, verdict, missing)] = _verdicts(text, handle)
        assert verdict == "shown" and missing, (text, verdict, missing)
    for text in ("[budget.py:4-7] (fresh)\ndef turn_budget(tokens):\n", "See it here [budget.py:6]."):
        assert _verdicts(text, handle)[0][1:] == ("shown", None), text
    right = [
        "`turn_budget()` caps the tokens [budget.py:4-7].",
        "`budget.turn_budget` caps the tokens [budget.py:4-7].",
        "Lines 4-7 define `turn_budget` [budget.py:4-7].",
        "In budget.py:4 `turn_budget` is defined [budget.py:4].",
        "The cap is `limit_tokens`, 4_096 or 4096 tokens [budget.py:6].",
        "`turn_budget` caps at 4096 tokens.\n[budget.py:4-7]",
    ]
    for text in right:
        assert [(v, m) for _c, v, m in _verdicts(text, handle)] == [("shown", None)], (text, _verdicts(text, handle))

    many = "\n".join(f"value_{n} = {n * 7}" for n in range(1, 51)) + "\n"
    handle, answer, root = _shown_run(owned, tmp_path / "many", {"many.py": many}, query="value_45", k=1,
                                      corpus="many")
    assert "[many.py:" in answer and "value_45 = 315" in answer, answer[:400]
    for text in ("Lines 44-46 hold `value_45` [many.py:45].", "In many.py:44 `value_45` is set [many.py:45].",
                 "`value_45` is set on line 45 to 315 [many.py:45]."):
        assert [(v, m) for _c, v, m in _verdicts(text, handle)] == [("shown", None)], (text, _verdicts(text, handle))
    assert _verdicts("`value_45` is 45 [many.py:44-45].", handle) == [("many.py:44-45", "shown", None)]
    assert _verdicts("`value_45` is 45 [many.py:40-50].", handle) == [("many.py:40-50", "shown", ["45"])]
    assert _verdicts("`value_45` is 46 [many.py:45].", handle) == [("many.py:45", "shown", ["46"])]

    spaced = gate.Shown()
    spaced.keep("/r", "docs/User Guide.md", 1, ["The guide_word is here."])
    spaced.keep("/r", "src/routes/[slug]/+page.svelte", 1, ["<script>let page_slug;</script>"])
    lines = {"docs/User Guide.md": ["The guide_word is here."],
             "src/routes/[slug]/+page.svelte": ["<script>let page_slug;</script>"]}
    result = gate.check("It says guide_word [docs/User Guide.md:1]. It binds page_slug [src/routes/[slug]/+page.svelte:1].",
                        spaced, lambda root, path: lines.get(path))
    assert [c["verdict"] for c in result["checked"]] == ["shown", "shown"], result

    wide = gate.Shown()
    minified = "var " + ";".join(f"v{n}=1" for n in range(230_000))      # about 2 MB, the index's bound
    wide.keep("/r", "min.js", 1, [minified])
    answer_text = ("The value v7 and other_word stand there [min.js:1]. " * 4000)[:gate.MAX_TEXT]
    started = time.perf_counter()
    result = gate.check(answer_text, wide, lambda root, path: [minified])
    assert len(result["checked"]) == gate.MAX_CHECKED and time.perf_counter() - started < 2.0
    assert result["checked"][0]["verdict"] == "shown" and result["checked"][0]["missing"] == ["other_word"]


CFG = """\"\"\"Settings of a small service.\"\"\"
import threading


class Cfg:
    def __init__(self):
        self.lock = threading.Lock()
        self.temperature = 0.2
        self.timeout_s = 30
        self.ollama_url = "http://localhost:11434"
        self.window = 4_096
""" + "".join(f"# padding line {n}\n" for n in range(1, 16))


def test_ci9_what_the_alert_reads(owned, tmp_path):
    """CI9 (review 2) -- the run was shown settings.py's lines 1-20: a number inside the cited range that is not one of its
    ends, a decimal under ten, a port after a host, a qualifier the lines do not hold, are alerts; an abbreviation,
    thousands written 4,096 or 4 096, a path between backquotes, a domain, raise none. A wrong word exempted, or a
    right answer alerted -> RED."""
    handle, answer, root = _shown_run(owned, tmp_path, {"settings.py": CFG}, query="timeout_s", k=1, corpus="cfg")
    cite = answer.split("[settings.py:", 1)[1].split("]", 1)[0]
    assert cite == "1-20", answer[:300]
    alerts = [
        ("The `timeout_s` is 15 seconds [settings.py:1-20].", ["15"]),
        ("The `temperature` is 0.7 [settings.py:1-20].", ["0.7"]),
        ("It reaches the engine on localhost:11435 [settings.py:1-20].", ["11435"]),
        ("It is guarded by an `asyncio.Lock` [settings.py:1-20].", ["asyncio.Lock"]),
    ]
    for text, missing in alerts:
        assert _verdicts(text, handle) == [("settings.py:1-20", "shown", missing)], (text, _verdicts(text, handle))
    quiet = [
        "The `timeout_s` is 30, e.g. for a slow engine [settings.py:1-20].",
        "The window holds 4,096 or 4 096 tokens [settings.py:1-20].",
        "In `opti_oignon/settings.py`, `timeout_s` is 30 [settings.py:1-20].",
        "The model comes from ollama.com, `timeout_s` 30 [settings.py:1-20].",
        "The `threading.Lock` guards it [settings.py:1-20].",
        "The `settings.Cfg.timeout_s` value is 30 [settings.py:1-20].",
    ]
    for text in quiet:
        assert _verdicts(text, handle) == [("settings.py:1-20", "shown", None)], (text, _verdicts(text, handle))


def test_ci10_the_lines_the_model_was_given(owned, tmp_path):
    """CI10 (review 252-1) -- through the real agent loop, four searches in one round: the round's budget cuts the
    fourth answer, and a line it cut reads unseen while a line of the first answer, given whole, reads shown; an
    answer the handler gave that never reached the model shows nothing; an answer given cut shows only the lines
    wholly before the cut. A line counted as shown that the model was never given -> RED."""
    import re

    from opti_oignon.agent import loop as agent_loop
    from opti_oignon.mycelium import gate
    from opti_oignon.mycelium.index import file_lines

    pad = "p" * 250
    files = {f"m{f}.py": "".join(f"value_{f}_{n:03d} = '{pad}'\n" for n in range(1, 200)) for f in range(4)}
    root = _folder(tmp_path / "src", files)
    _index(owned, "oo", root)
    handle = owned.tool.make_handler(_cloned(tmp_path, "s1", root))
    answers = []

    def recorded(arguments):
        out = handle(arguments)
        answers.append(out)
        return out

    told = []

    def delivered(answer, given):
        told.append(given)
        handle.delivered(answer, given)

    if hasattr(handle, "delivered"):
        recorded.delivered = delivered              # the hook travels with the handler the loop is given
    state = {"rounds": 0}

    def line_of(answer, transcript, inside):
        """(path, number, text) of a passage line of ``answer`` that is (or is not) in ``transcript``."""
        for match in re.finditer(r"^\[([^\]]+):(\d+)-(\d+)\]", answer, re.M):
            path, first, last = match.group(1), int(match.group(2)), int(match.group(3))
            now = file_lines(str(root.resolve()), path)
            for n in range(last, first - 1, -1):
                if now[n - 1] in answer and (now[n - 1] in transcript) is inside:
                    return path, n, now[n - 1]
        return None

    def model(messages, tools=None):
        state["rounds"] += 1
        if state["rounds"] == 1:
            calls = [{"function": {"name": "mycelium_search", "arguments": {"query": f"value_{f}_100", "k": 10}}}
                     for f in range(4)]
            return {"message": {"content": "", "tool_calls": calls}}
        transcript = "\n".join(str(m.get("content", "")) for m in messages)
        assert "[truncated:" in transcript, "the round's budget cut no answer"
        cut = line_of(answers[3], transcript, False)
        given = line_of(answers[0], transcript, True)
        kept = line_of(answers[3], transcript, True)
        assert cut is not None and given is not None and kept is not None, (cut, given, kept)
        state["transcript"] = transcript
        state["final"] = " ".join(f"The value `{line[2].split(' ', 1)[0]}` is set there [{line[0]}:{line[1]}]."
                                  for line in (cut, given, kept))
        return {"message": {"content": state["final"], "tool_calls": None}}

    agent_loop.run("find the values", model_client=model, mode="daily", tool_handlers={"mycelium_search": recorded},
                   include_memory=False, max_rounds=4)
    checked = gate.check(state["final"], handle.shown, file_lines)["checked"]
    assert [c["verdict"] for c in checked] == ["unseen", "shown", "shown"], checked
    assert len(told) == 4 and all(given and given in state["transcript"] for given in told), [len(g) for g in told]

    # A corpus line holding the wrapper's marker with no closing bracket: the wrapper defangs it with what follows.
    marked = _folder(tmp_path / "marked", {"hygiene.py": '"""Markers a reply must not echo."""\nPREFIXES = (\n'
                                           '    "[tool:",\n    "<untrusted_data",\n    "</untrusted_data",\n)\n'
                                           'HIDDEN_VALUE = 7341\nAFTER_ALL = 1\n'})
    _index(owned, "marked", marked)
    seen = owned.tool.make_handler(_cloned(tmp_path, "s4", marked))
    state.update(rounds=0)

    def asked(messages, tools=None):
        state["rounds"] += 1
        if state["rounds"] == 1:
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "mycelium_search", "arguments": {"query": "PREFIXES HIDDEN_VALUE", "k": 1}}}]}}
        state["transcript"] = "\n".join(str(m.get("content", "")) for m in messages)
        return {"message": {"content": "done", "tool_calls": None}}

    agent_loop.run("find the markers", model_client=asked, mode="daily", tool_handlers={"mycelium_search": seen},
                   include_memory=False, max_rounds=3)
    assert "HIDDEN_VALUE" not in state["transcript"], state["transcript"][-400:]
    assert _verdicts("`PREFIXES` [hygiene.py:2]. `HIDDEN_VALUE` [hygiene.py:7].", seen) == [
        ("hygiene.py:2", "shown", None), ("hygiene.py:7", "unseen", None)]

    # An answer that never reached the model; then one given cut inside line 6.
    lone = owned.tool.make_handler(_cloned(tmp_path, "s2", root))
    lone({"query": "value_0_100", "k": 1})
    assert not lone.shown.any()
    handle, answer, _root = _shown_run(owned, tmp_path / "cut", query="turn_budget", k=1, corpus="cut")
    assert "[budget.py:1-8]" in answer, answer
    second = owned.tool.make_handler(_cloned(tmp_path / "cut", "s3", _root))
    answer = second({"query": "turn_budget", "k": 1})
    at = answer.index("limit_tokens = 4096") + 5
    _delivered(second, answer, answer[:at] + "\n[truncated: 400 bytes, 12 lines]")
    assert _verdicts("`turn_budget` [budget.py:4]. `limit_tokens` [budget.py:6]. It returns [budget.py:7].",
                     second) == [("budget.py:4", "shown", None), ("budget.py:6", "unseen", None),
                                 ("budget.py:7", "unseen", None)]

    # A cut one character before a line's end in a later passage, then at its end: unseen, then shown.
    two = {"alpha.py": "\n".join(f"alpha_{n} = {n}" for n in range(1, 9)) + "\n",
           "beta.py": "\n".join(f"beta_{n} = {n}" for n in range(1, 9)) + "\n"}
    root = _folder(tmp_path / "two", two)
    _index(owned, "two", root)
    for cut, verdict in ((-1, "unseen"), (0, "shown")):
        third = owned.tool.make_handler(_cloned(tmp_path, f"t{cut + 1}", root))
        answer = third({"query": "alpha_3 beta_3", "k": 2})
        assert answer.index("[beta.py:") > answer.index("[alpha.py:") or answer.index("[alpha.py:") > answer.index(
            "[beta.py:"), answer
        later = "beta.py" if answer.index("[beta.py:") > answer.index("[alpha.py:") else "alpha.py"
        name = later[:-3]
        end = answer.index(f"{name}_3 = 3") + len(f"{name}_3 = 3")
        _delivered(third, answer, answer[:end + cut] + "\n[truncated: 9 bytes, 1 lines]")
        assert _verdicts(f"`{name}_2` [{later}:2]. `{name}_3` [{later}:3].", third) == [
            (f"{later}:2", "shown", None), (f"{later}:3", verdict, None)], (cut, answer[:end + 2])
