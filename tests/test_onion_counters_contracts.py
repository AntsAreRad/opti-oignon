"""Contracts for what the onion's queue counted, kept across processes and shown to the user.

The queue counts what it does -- bursts, evictions by rung, refusals by
motive, the memory block's events, proposals, the mirror, the residence --
and the native core counts what it served. Counted in a process alone, a
week of an onion that never evicted reads as zero, and nobody can tell it
from an onion never turned on. So the counts go to a file of aggregates:
names and numbers, never a word of a conversation nor its id, written whole
or not at all, merged under a lock with what other processes wrote, and
never written over when it does not read as counts. The status route and
the terminal show them:

  * AG1 -- at the end of a burst the counts reach the file, and a later
    process adds to them.
  * AG2 -- a write cut between its temporary file and its rename leaves
    the file as it was, readable, and no temporary file behind.
  * AG3 -- two writers at once lose no count.
  * AG4 -- no word of a conversation and no conversation id reaches the
    file, whatever the queue did; the scan finds a word planted in a copy.
  * AG5 -- a file that does not read as counts is refused by name and never
    written over; the counts stay in the process and the refusal is
    counted.
  * AG6 -- the native core's share of the probes joins the counts.
  * AG7 -- the status route answers the switch, the counts by event and
    motive, the day they began and whether they are kept, with the onion
    off as well, and nothing of a conversation.
  * AG8 -- the terminal's ``/status`` shows the same counts.
  * AG9 -- with no path the counts stay in the process: no file is
    written.
  * AG10 -- the counters' path comes from ``onion.yaml`` alone: a file
    that omits it is refused by name.
  * AG11 -- at the end of a close the counts reach the file.
  * AG12 -- two writes of one process at once add its counts once.
  * AG13 -- a decision on a proposal reaches the file at once.
  * AG14 -- what the chat path counts reaches the file within
    ``counters.flush_every_s`` of the last write.
  * AG15 -- the terminal session writes what it counted when it quits.
  * AG16 -- a counters file whose first day is not a day is refused by
    name, and the status route answers the refusal, not an error.
  * AG17 -- a count to add below zero is refused by name, and the file is
    left as it was.
  * AG18 -- ``oo chat`` writes what the session counted however it ends:
    at ``/quit`` and at the end of its input alike.
  * AG19 -- the totals the status reads during a write are the counts once,
    never twice.
  * AG20 -- a listing of the proposals that opens deferred ones writes the
    counts.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source; the registry and the governor are
blocked, so a summariser is always injected.
"""

import json
import os
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
_ONION_YAML = REPO / "opti_oignon" / "config" / "onion.yaml"


def _deps():
    module = types.ModuleType("opti_oignon.api.deps")
    module.MEMORY_AVAILABLE = False
    module.memory_manager = None
    return module


def _open(*, routes=False, session=False, cli=False):
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _MODULES}
    seeded = {}
    if routes:
        targets["opti_oignon.api.schemas"] = source("api", "schemas.py")
        targets["opti_oignon.api.routes_memory"] = source("api", "routes_memory.py")
        seeded["opti_oignon.api.deps"] = _deps()
    if session or cli:
        targets["opti_oignon.cli.session"] = source("cli", "session.py")
    if cli:
        targets.update({f"opti_oignon.cli.{m}": source("cli", f"{m}.py") for m in ("config", "client", "output", "main")})
    loaded, restore = isolate(
        targets=targets,
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils", "opti_oignon.resource_governor"),
        seeded=seeded,
        packages=("opti_oignon.memory", "opti_oignon.api", "opti_oignon.cli"),
    )
    lib = loaded["opti_oignon.memory.librarian"]
    lib.reset_librarian()
    return lib, loaded, restore


def _config(lib, **over):
    fields = dict(enabled=True, model="fake:1b", keep_alive="5m", min_new_turns=4, temperature=0.0, num_predict=64)
    fields.update(over)
    return lib.LibrarianConfig(**fields)


def _line(i, word="service"):
    return f"Turn {i}: Alice reviewed {word} {i} on 2026-03-{i % 28 + 1:02d} and {word} {i} lives on cluster {i}."


def _messages(n, first=1, word="service"):
    return [{"role": "user" if i % 2 else "assistant", "content": _line(i, word)} for i in range(first, first + n)]


def _faithful(turns):
    return " ".join(t["text"] for t in turns)


def _gate(loaded, span_turns=2):
    return loaded["opti_oignon.memory.peels"].Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=span_turns)


def _budget(loaded, flesh=1):
    composer = loaded["opti_oignon.memory.composer"]
    return composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=flesh, turn=60)


def _burst(lib, loaded, cid, config, n=6, first=1, word="service"):
    state = lib.state_for(cid, config)
    state.mirror(_messages(n, first, word))
    return lib._curation_burst(cid, config=config, summarize=_faithful, gate=_gate(loaded), budget=_budget(loaded))


def _kept(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["counts"]


# ---------------------------------------------------------------------------
# AG1-AG3 -- kept across processes, whole, merged
# ---------------------------------------------------------------------------
def test_ag1_at_the_end_of_a_burst_the_counts_reach_the_file_and_a_later_process_adds_to_them(tmp_path):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        assert _burst(lib, loaded, "c1", config) == 3, "control: three spans evicted"
        first = _kept(path)
        assert first["burst"]["ran"] == 1 and first["eviction"]["accepted"] == 3
        assert {e: first[e] for e in lib.counters()} == lib.counters(), "the file holds what the process counted"
        lib.reset_librarian()
        _burst(lib, loaded, "c2", config, n=4)
        second = _kept(path)
        assert second["burst"]["ran"] == 2 and second["eviction"]["accepted"] == 5, "a later process adds to them"
    finally:
        restore()


def test_ag2_a_write_cut_before_its_rename_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        _burst(lib, loaded, "c1", config)
        before = path.read_bytes()
        lib._counted("burst", "ran")

        def cut(*_args, **_kw):
            raise OSError("the power went out")

        monkeypatch.setattr(os, "replace", cut)
        assert lib.flush_counters(config) is False
        monkeypatch.undo()
        assert path.read_bytes() == before and json.loads(before)["counts"]["burst"]["ran"] == 1
        assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == [], "no temporary file behind"
        assert lib.flush_counters(config) is True and _kept(path)["burst"]["ran"] == 2, "the next write keeps the count"
    finally:
        restore()


def test_ag3_two_writers_at_once_lose_no_count(tmp_path):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        read, both = lib._read_counts, threading.Barrier(2, timeout=0.5)

        def slow(p):
            payload = read(p)
            try:
                both.wait()
            except threading.BrokenBarrierError:
                pass
            return payload

        lib._read_counts = slow
        writers = [threading.Thread(target=lib._merge_into, args=(path, {"burst": {"ran": 1}})) for _ in range(2)]
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join(5)
        assert _kept(path) == {"burst": {"ran": 2}}
    finally:
        restore()


# ---------------------------------------------------------------------------
# AG4-AG6 -- names and numbers; a foreign file; the native share
# ---------------------------------------------------------------------------
def _leaks(text, needles):
    return [needle for needle in needles if needle in text]


def test_ag4_no_word_of_a_conversation_and_no_id_reaches_the_file(tmp_path):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        cid, word = "conv-Quokka-5521", "Wolframite"
        _burst(lib, loaded, cid, config, n=8, word=word)
        lib.memory_block(cid, word, budget=_budget(loaded, flesh=5000))
        lib.close_onion(cid, config=config, summarize=lambda turns: f"{word} {_faithful(turns)}", gate=_gate(loaded))
        text = path.read_text(encoding="utf-8")
        assert _kept(path)["eviction"], "control: the queue's work was counted"
        assert _leaks(text, [cid, word, "Alice", "cluster", "2026-03"]) == []
        assert _leaks(text + word, [word]) == [word], "witness: the scan finds a word planted in a copy"
    finally:
        restore()


@pytest.mark.parametrize("written", ["not json at all", json.dumps({"format": 2, "counts": {}}),
                                     json.dumps({"format": 1, "counts": {"burst": {"ran": "many"}}})],
                         ids=["text", "format", "value"])
def test_ag5_a_file_that_does_not_read_as_counts_is_refused_by_name_and_never_written_over(tmp_path, written):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        path.write_text(written, encoding="utf-8")
        config = _config(lib, counters_path=str(path))
        lib._counted("burst", "ran")
        assert lib.flush_counters(config) is False
        assert path.read_text(encoding="utf-8") == written, "never written over"
        totals = lib.counter_totals(config)
        assert totals["refused"] and "counts.json" in totals["refused"] and totals["persisted"] is False
        assert lib.counters()["burst"]["ran"] == 1 and lib.counters()["counters"]["refused"] == 1
    finally:
        restore()


def test_ag6_the_native_core_s_share_joins_the_counts(tmp_path):
    lib, loaded, restore = _open()
    try:
        probes = loaded["opti_oignon.memory.probes"]
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        probes._native_counted("draw", "native")
        probes._native_counted("score", "native")
        assert lib.flush_counters(config) is True
        native = _kept(path)["native"]
        assert native.get("draw:native", 0) >= 1 and native.get("score:native", 0) >= 1
        assert lib.counter_totals(config)["counts"]["native"] == native
    finally:
        restore()


# ---------------------------------------------------------------------------
# AG7-AG8 -- the status route and the terminal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enabled", [True, False], ids=["on", "off"])
def test_ag7_the_status_route_answers_the_counts_and_nothing_of_a_conversation(tmp_path, enabled):
    lib, loaded, restore = _open(routes=True)
    try:
        routes = loaded["opti_oignon.api.routes_memory"]
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        _burst(lib, loaded, "conv-Quokka-5521", config, word="Wolframite")
        lib._counted("block", "folded")
        lib.load_config = lambda path=None: config
        lib.onion_enabled = lambda path=None: enabled
        answer = routes.onion_status()
        assert answer["enabled"] is enabled and answer["persisted"] is True and answer["refused"] is None
        assert answer["counts"]["burst"]["ran"] == 1 and answer["counts"]["block"]["folded"] == 1
        assert answer["since"], "the day the counts began"
        assert _leaks(json.dumps(answer), ["conv-Quokka-5521", "Wolframite", "Alice"]) == []
    finally:
        restore()


def test_ag8_the_terminal_s_status_shows_the_same_counts(tmp_path):
    lib, loaded, restore = _open(session=True)
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        _burst(lib, loaded, "c1", config)
        lib.load_config = lambda path=None: config
        session = loaded["opti_oignon.cli.session"].ChatSession(librarian=lib)
        events = list(session.handle("/status"))
        said = "\n".join(e.text for e in events)
        assert [e.kind for e in events] == ["info"], said
        assert "burst: ran=1" in said and "eviction: accepted=3" in said
        assert "kept across sessions" in said
    finally:
        restore()


# ---------------------------------------------------------------------------
# AG9-AG11 -- no path; the path from onion.yaml; the close
# ---------------------------------------------------------------------------
def test_ag9_with_no_path_the_counts_stay_in_the_process(tmp_path):
    lib, loaded, restore = _open()
    try:
        config = _config(lib, counters_path="")
        _burst(lib, loaded, "c1", config)
        assert lib.flush_counters(config) is False
        totals = lib.counter_totals(config)
        assert totals["persisted"] is False
        assert {event: motives for event, motives in totals["counts"].items() if event != "native"} == lib.counters()
        assert list(tmp_path.iterdir()) == []
    finally:
        restore()


def test_ag10_the_counters_path_comes_from_onion_yaml_alone(tmp_path):
    lib, loaded, restore = _open()
    try:
        shipped = _ONION_YAML.read_text(encoding="utf-8")
        assert "\ncounters:\n  path:" in shipped, "control: the shipped file states it"
        assert lib.load_config().counters_path, "kept by default, under the data directory"
        omitted = tmp_path / "omitted.yaml"
        omitted.write_text(shipped.replace("\ncounters:\n  path:", "\nuncounted:\n  path:"), encoding="utf-8")
        with pytest.raises(lib.LibrarianError, match="counters"):
            lib.load_config(omitted)
    finally:
        restore()


def test_ag11_at_the_end_of_a_close_the_counts_reach_the_file(tmp_path):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        state = lib.state_for("c1", config)
        state.mirror(_messages(6))
        closing = lib.close_onion("c1", config=config, summarize=_faithful, gate=_gate(loaded))
        assert closing.evicted == 3, "control"
        assert _kept(path)["eviction"]["accepted"] == 3
    finally:
        restore()


# ---------------------------------------------------------------------------
# AG12-AG17 -- what the independent review of the counters found
# ---------------------------------------------------------------------------
def test_ag12_two_writes_of_one_process_at_once_add_its_counts_once(tmp_path):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        lib._counted("burst", "ran", 5)
        real, both = lib._merge_into, threading.Barrier(2, timeout=0.5)

        def merge(p, delta):
            # Two bursts ending at once: each would reach the file before the other wrote.
            try:
                both.wait()
            except threading.BrokenBarrierError:
                pass
            return real(p, delta)

        lib._merge_into = merge
        writers = [threading.Thread(target=lib.flush_counters, args=(config,)) for _ in range(2)]
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join(5)
        assert _kept(path)["burst"]["ran"] == 5
    finally:
        restore()


_DECIDED = "We keep Docker on the build server."
_LOSSY = "Alice moved the build to Berlin on 2026-03-04. The Berlin build runs 12 jobs a day. Bob checks the logs every morning."


def _offered(lib, loaded, cid, config):
    """One typed decision held by the queue and offered to the Core, in conversation ``cid``."""
    from dataclasses import replace

    peels = loaded["opti_oignon.memory.peels"]
    state = lib.state_for(cid, config)
    state.mirror([
        {"role": "user", "origin": "typed", "segments": [],
         "content": "Alice moved the build to Berlin on 2026-03-04. " + _DECIDED},
        {"role": "assistant", "origin": "assistant", "segments": [],
         "content": "Noted: the Berlin build runs 12 jobs a day. Bob checks the logs every morning."},
    ])
    ladder = replace(peels.load_ladder(), rho=0.1)
    outcome = lib.curate(state, lambda turns: _LOSSY, gate=replace(peels.load_gate(), span_turns=2),
                         budget=_budget(loaded), ladder=ladder)
    assert outcome.rung == "held", "control"
    return lib.proposals(cid, config=config, ladder=ladder)[0]["id"]


def test_ag13_a_decision_on_a_proposal_reaches_the_file_at_once(tmp_path):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        declined = _offered(lib, loaded, "c1", config)
        accepted = _offered(lib, loaded, "c2", config)
        assert not path.exists(), "control: nothing written by the steps alone"
        lib.decline_proposal("c1", declined, actor="user", config=config)
        assert _kept(path)["proposal"]["declined"] == 1
        composer = loaded["opti_oignon.memory.composer"]
        roomy = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=140, turn=60)
        lib.accept_proposal("c2", accepted, actor="user", config=config, budget=roomy)
        assert _kept(path)["proposal"]["accepted"] == 1
    finally:
        restore()


def test_ag14_what_the_chat_path_counts_reaches_the_file_within_its_interval(tmp_path):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path), counters_flush_s=60.0, min_new_turns=1000)
        clock = [1000.0]
        lib._monotonic = lambda: clock[0]
        lib.maybe_curate("c1", _messages(2), config=config, runner=lambda cid: None)
        assert not path.exists(), "control: the interval has not run out"
        clock[0] += 61.0
        lib.maybe_curate("c1", _messages(4), config=config, runner=lambda cid: None)
        assert _kept(path)["mirror"]["appended"] == 4
    finally:
        restore()


def test_ag15_the_terminal_session_writes_what_it_counted_when_it_quits(tmp_path):
    lib, loaded, restore = _open(session=True)
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        lib.load_config = lambda path=None: config
        lib._counted("block", "folded", 2)
        session = loaded["opti_oignon.cli.session"].ChatSession(librarian=lib)
        assert not path.exists(), "control"
        list(session.handle("/quit"))
        assert _kept(path)["block"]["folded"] == 2
    finally:
        restore()


def test_ag16_a_file_whose_first_day_is_not_a_day_is_refused_by_name_and_the_route_says_so(tmp_path):
    lib, loaded, restore = _open(routes=True)
    try:
        routes = loaded["opti_oignon.api.routes_memory"]
        path = tmp_path / "counts.json"
        path.write_text(json.dumps({"format": 1, "since": 5, "counts": {"burst": {"ran": 1}}}), encoding="utf-8")
        with pytest.raises(lib.CountersRefused, match="counts.json"):
            lib._read_counts(path)
        config = _config(lib, counters_path=str(path))
        lib.load_config = lambda path=None: config
        lib.onion_enabled = lambda path=None: True
        answer = routes.onion_status()
        assert answer["refused"] and "counts.json" in answer["refused"] and answer["persisted"] is False
    finally:
        restore()


def test_ag17_a_count_to_add_below_zero_is_refused_by_name_and_the_file_left_as_it_was(tmp_path):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        lib._merge_into(path, {"burst": {"ran": 3}})
        before = path.read_bytes()
        with pytest.raises(lib.CountersRefused, match="burst"):
            lib._merge_into(path, {"burst": {"ran": -5}})
        assert path.read_bytes() == before
    finally:
        restore()


# ---------------------------------------------------------------------------
# AG18-AG20 -- what the narrow review of the counters found
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stdin", ["/quit\n", ""], ids=["quit", "end-of-input"])
def test_ag18_oo_chat_writes_what_the_session_counted_however_it_ends(tmp_path, monkeypatch, stdin):
    from click.testing import CliRunner

    # The CLI reads its configuration from the test's own directory, never from the user's.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    lib, loaded, restore = _open(cli=True)
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        lib.load_config = lambda path=None: config
        lib._counted("block", "folded", 2)
        session = loaded["opti_oignon.cli.session"]

        def factory(model, conversation_id):
            return session.ChatSession(librarian=lib, conversation_id=conversation_id)

        result = CliRunner(mix_stderr=False).invoke(loaded["opti_oignon.cli.main"].cli, ["--no-color", "chat"],
                                                    input=stdin, obj={"chat_session": factory})
        assert result.exit_code == 0, result.output
        assert _kept(path)["block"]["folded"] == 2
    finally:
        restore()


def test_ag19_the_totals_read_during_a_write_are_the_counts_once(tmp_path):
    lib, loaded, restore = _open()
    try:
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        lib._counted("burst", "ran", 5)
        real, renamed, resume = lib._merge_into, threading.Event(), threading.Event()

        def merge(p, delta):
            result = real(p, delta)
            # The file holds the new counts; the process has not recorded the write yet.
            renamed.set()
            resume.wait(5)
            return result

        lib._merge_into = merge
        writer = threading.Thread(target=lib.flush_counters, args=(config,))
        writer.start()
        assert renamed.wait(5), "control: the write reached its rename"
        seen = []
        reader = threading.Thread(target=lambda: seen.append(lib.counter_totals(config)["counts"]["burst"]["ran"]))
        reader.start()
        reader.join(0.3)
        resume.set()
        writer.join(5)
        reader.join(5)
        assert seen == [5]
    finally:
        restore()


def test_ag20_a_listing_that_opens_deferred_proposals_writes_the_counts(tmp_path):
    from dataclasses import replace

    lib, loaded, restore = _open()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        path = tmp_path / "counts.json"
        config = _config(lib, counters_path=str(path))
        day = ["2026-10-07"]
        lib._today = lambda: day[0]
        gate = replace(peels.load_gate(), span_turns=2)
        ladder = replace(peels.load_ladder(), rho=0.1, proposals_per_day=1)
        state = lib.state_for("c1", config)
        answer = "Noted: the Berlin build runs 12 jobs a day. Bob checks the logs every morning."
        state.mirror([
            {"role": "user", "origin": "typed", "segments": [],
             "content": "Alice moved the build to Berlin on 2026-03-04. " + _DECIDED},
            {"role": "assistant", "origin": "assistant", "segments": [], "content": answer},
            {"role": "user", "origin": "typed", "segments": [],
             "content": "Alice moved the build to Berlin on 2026-03-04. We drop Redis for the session cache."},
            {"role": "assistant", "origin": "assistant", "segments": [], "content": answer},
        ])
        for _ in range(2):
            assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=_budget(loaded), ladder=ladder).rung == "held"
        assert [q.status for q in state.proposals] == ["open", "deferred"], "control: the day's room taken"
        assert not path.exists()
        day[0] = "2026-10-08"
        lib.proposals("c1", config=config, ladder=ladder)
        assert _kept(path)["proposal"]["made"] == 2
    finally:
        restore()
