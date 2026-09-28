#!/usr/bin/env python3
"""Contracts for the chat loader: the stream reader, the progress reducer, its words and its drawing.

While a reply is being written, one status line stands where the reply will
be, and a run the server executes (an execution pipeline, a reasoning
strategy, a consensus, a self-correction) shows one plant per step. What the
line says and how far a plant has grown come only from what the server sent:
no clock grows a plant, no step text is parsed, and every word comes from one
closed table that the garden's ethics nets have read.

The modules and the names the contracts read:

  * ``lib/chat/stream.ts`` -- ``openStream(options, callbacks, deps)``: one
    chat WebSocket read to its end. Every frame goes to ``onFrame`` first,
    then to its own callback; ``done`` and ``error`` end the stream. The
    socket, the timers and the random source are handed in (``deps``), so it
    runs under Node over a fake socket. ``stop(request)`` sends the cancel
    first and keeps reading until ``done`` or ``error``, or until
    ``STOP_WAIT_MS`` has passed; a stream that ends without either calls
    ``onLost``.
  * ``lib/chat/progress.ts`` -- the reducer, pure and clock-free (every
    function takes the time as an argument): ``start(now)``,
    ``observe(state, frame, now, mapStatus)``, ``stop``, ``lose``,
    ``reconnecting``, ``tick``; the views ``lineOf``, ``silent``,
    ``moving``, ``summary`` and ``endLook``; the announcer ``announcer``,
    ``announce`` and ``due``. A line is a key of the word table and its
    values; ``state.said`` holds what the last transition announces.
  * ``lib/chat/loaderWords.ts`` -- the word table: ``WORDS`` (every
    template), ``RAN_AS``, ``RUN_NAMES``, ``UNITS``; ``words(line)``,
    ``said(line)``, ``statusLine(message)``, ``ranAsWords(code)``,
    ``stepWords(step)``, ``summaryText(summary)``.
  * ``lib/chat/pacing.ts`` -- the stage pacing, pure and clock-free:
    ``schedule(arrivals, start, end)`` turns the looks a plant was given,
    and when, into the looks it shows, and when; ``shownAt`` and ``nextAt``
    read a schedule at a time handed in.
  * ``lib/pixel/plantFrames.ts`` and ``lib/pixel/onionFrames.ts`` -- the
    pixel data: ``PLANT_FRAMES``, ``PLANT_INKS``, ``plantStrip(state,
    stage)``, ``ROW_GROUNDS``, ``fitRow`` and ``rowCapacity``;
    ``ONION_FRAMES``, ``ONION_INKS``, ``ONION_GROUNDS``, ``ONION_STRIP``. A
    strip's frame 0 is its still.
  * ``lib/components/chat/StreamingStatus.svelte`` -- renders the reducer's
    state (``loader``) at a time handed in (``now``): the waiting line and
    its onion, or a run's card with its plants (``StepRow``) and its step
    lines (``StepList``), and the one status region. ``ChatMessage`` mounts
    ``RunSummary``, the footer line read from the steps' record ``done``
    carried.
  * ``lib/components/pixel/PixelStrip.svelte`` -- draws a strip, one path
    per colour, and plays it by one CSS class per motion.

They run under Node (fixture streams, an injected clock, a fake socket), as
text, and through ``allium/ethics.py`` loaded by ``isolate``. They prove a
form and a behaviour over fixtures, never a browser.

Local-only (the public distribution ships no tests).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_design_tokens_contracts as _ds  # noqa: E402
import test_markdown_render_contracts as _mk  # noqa: E402
import test_navigation_contracts as _nav  # noqa: E402
from _frontend import REPO, files, read, run_ts, ssr  # noqa: E402
from _isolation import isolate  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file.
BUDGET_S = {
    "test_ld1_the_reducer_is_total_and_grows_only_on_a_true_fraction": 2.0,
    "test_ld2_growth_reads_only_progress_and_child_counts[token]": 2.0,
    "test_ld2_growth_reads_only_progress_and_child_counts[child]": 2.0,
    "test_ld2_growth_reads_only_progress_and_child_counts[clock]": 1.0,
    "test_ld5_thirty_seconds_without_a_frame_stops_the_motion_and_says_so": 2.0,
    "test_ld7_the_status_region_announces_phases_and_step_transitions_coalesced": 2.0,
    "test_ld8_every_loader_word_passes_the_nets_and_says_no_code_and_no_drawing[nets]": 2.0,
    "test_ld8_every_loader_word_passes_the_nets_and_says_no_code_and_no_drawing[ascii]": 2.0,
    "test_ld8_every_loader_word_passes_the_nets_and_says_no_code_and_no_drawing[ran_as]": 2.0,
    "test_ld8_every_loader_word_passes_the_nets_and_says_no_code_and_no_drawing[drawing]": 2.0,
    "test_ld8_every_loader_word_passes_the_nets_and_says_no_code_and_no_drawing[stopped]": 2.0,
    "test_ld8_every_loader_word_passes_the_nets_and_says_no_code_and_no_drawing[status]": 2.0,
    "test_ld13_no_module_under_lib_reads_a_step_status": 1.0,
    "test_ld17_a_closed_stream_leaves_its_open_steps_unknown[unknown]": 2.0,
    "test_ld17_a_closed_stream_leaves_its_open_steps_unknown[announce]": 2.0,
    "test_ld17_a_closed_stream_leaves_its_open_steps_unknown[summary]": 2.0,
    "test_ld20_a_stop_reads_the_servers_closing_frames": 2.0,
    "test_ld9_every_loader_colour_is_an_app_token_that_reads_on_its_ground[tokens]": 2.0,
    "test_ld9_every_loader_colour_is_an_app_token_that_reads_on_its_ground[marks]": 2.0,
    "test_ld9_every_loader_colour_is_an_app_token_that_reads_on_its_ground[row]": 2.0,
    "test_ld9_every_loader_colour_is_an_app_token_that_reads_on_its_ground[straw]": 2.0,
    "test_ld10_the_frame_tables_keep_their_grid_their_silhouette_and_a_calm_pace[grid]": 2.0,
    "test_ld10_the_frame_tables_keep_their_grid_their_silhouette_and_a_calm_pace[channel]": 2.0,
    "test_ld10_the_frame_tables_keep_their_grid_their_silhouette_and_a_calm_pace[silhouette]": 2.0,
    "test_ld10_the_frame_tables_keep_their_grid_their_silhouette_and_a_calm_pace[flicker]": 2.0,
    "test_ld11_no_script_timer_moves_the_loader_and_reduced_motion_draws_frame_zero[timers]": 1.0,
    "test_ld11_no_script_timer_moves_the_loader_and_reduced_motion_draws_frame_zero[reduced]": 6.0,
    "test_ld16_the_step_row_fits_its_column_never_below_two_x": 2.0,
    "test_ld21_each_stage_holds_400_ms_a_crowded_queue_jumps_and_a_brief_run_shows_its_end": 2.0,
    "test_ld22_every_loader_colour_is_an_app_token_that_reads_on_the_rows_ground[tokens]": 2.0,
    "test_ld22_every_loader_colour_is_an_app_token_that_reads_on_the_rows_ground[marks]": 2.0,
    "test_ld22_every_loader_colour_is_an_app_token_that_reads_on_the_rows_ground[row]": 2.0,
    "test_ld3_every_end_has_its_own_form_and_its_own_words[failed]": 6.0,
    "test_ld3_every_end_has_its_own_form_and_its_own_words[ends]": 6.0,
    "test_ld3_every_end_has_its_own_form_and_its_own_words[words]": 6.0,
    "test_ld4_one_thing_moves_at_a_time": 6.0,
    "test_ld6_live_seconds_show_from_five_seconds_and_never_reach_the_status_region": 6.0,
    # Run alone it starts the renderer and compiles ChatMessage's whole tree first.
    "test_ld12_after_done_one_folded_footer_line_counts_the_steps_record": 8.0,
}

_SRC = "frontend/src"
_LIB = f"{_SRC}/lib"
_PROGRESS = f"{_LIB}/chat/progress.ts"
_WORDS = f"{_LIB}/chat/loaderWords.ts"
_STREAM = f"{_LIB}/chat/stream.ts"
_STORE = f"{_LIB}/stores/chat.ts"

_MODULES = {"OO_PROGRESS": _PROGRESS, "OO_WORDS": _WORDS, "OO_STREAM": _STREAM}

_ETHICS = "opti_oignon.allium.ethics"

# The plant's growth, lowest first; `soil` (skipped) stands outside it.
_GROWTH = ("seed", "crook", "flag", "leaf", "bulb")


# ---------------------------------------------------------------------------
# The Node driver: it replays fixture streams through the modules and prints
# what they return.
# ---------------------------------------------------------------------------
_DRIVER = r"""
const load = async (name) => (process.env[name] ? await import(process.env[name]) : null);
const P = await load('OO_PROGRESS');
const W = await load('OO_WORDS');
const S = await load('OO_STREAM');
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT || 'null');

const plain = (value) => JSON.parse(JSON.stringify(value === undefined ? null : value));
const text = (line) => (line ? W.words(line) : null);

const snap = (state, now) => {
    const summary = P.summary(state);
    return plain({
        steps: state.runs.flatMap((run) => run.steps.map((step) => ({
            run: run.id, index: step.index, state: step.state, stage: step.stage,
            look: P.endLook(step), words: W.stepWords(step),
        }))),
        rows: state.runs.filter((run) => run.parent === null).map((run) => run.id),
        line: text(P.lineOf(state, now)),
        said: state.said.map((line) => W.said(line)),
        silent: P.silent(state, now),
        moving: P.moving(state, now),
        end: state.end,
        summary: summary ? W.summaryText(summary) : null,
    });
};

const apply = (state, event) => {
    const [kind, t, arg] = event;
    if (kind === 'start') return P.start(t);
    if (kind === 'frame') return P.observe(state, arg, t, W.statusLine);
    if (kind === 'stop') return P.stop(state, t);
    if (kind === 'lost') return P.lose(state, t);
    if (kind === 'reconnecting') return P.reconnecting(state, arg, 3, t);
    if (kind === 'tick') return P.tick(state, t);
    throw new Error('unknown event ' + kind);
};

const replay = (events) => {
    let state = null;
    return events.map((event) => {
        state = apply(state, event);
        return snap(state, event[1]);
    });
};

const fakeSocket = (log) => ({
    readyState: 1,
    onopen: null, onmessage: null, onerror: null, onclose: null,
    send(data) { log.push(['send', JSON.parse(data)]); },
    close() { this.readyState = 3; log.push(['close']); },
});

const noop = () => {};

const run = {
    replay: () => replay(input),
    // The reducer's own state after each named stream: what the components render.
    states: () => {
        const out = {};
        for (const [name, events] of Object.entries(input)) {
            let state = null;
            for (const event of events) state = apply(state, event);
            out[name] = state;
        }
        return plain(out);
    },
    announce: () => {
        let state = null;
        const said = [];
        for (const event of input.events) {
            state = apply(state, event);
            for (const line of state.said) said.push([event[1], W.said(line)]);
        }
        let a = P.announcer();
        const spoken = [];
        let pushed = 0;
        for (let t = 0; t <= input.until; t += input.step) {
            while (pushed < said.length && said[pushed][0] <= t) {
                a = P.announce(a, said[pushed][1], said[pushed][0]);
                pushed += 1;
            }
            const out = P.due(a, t);
            a = out.announcer;
            if (out.text !== null) spoken.push([t, out.text]);
        }
        return { said, spoken, coalesce: P.COALESCE_MS, gap: P.ANNOUNCE_GAP_MS };
    },
    words: () => {
        const rendered = {};
        for (const [key, template] of Object.entries(W.WORDS)) rendered[key] = W.fill(template, input.sample);
        const tables = { WORDS: W.WORDS, RAN_AS: W.RAN_AS, RUN_NAMES: W.RUN_NAMES, UNITS: W.UNITS };
        const units = {};
        for (const [key, template] of Object.entries(W.UNITS)) units[key] = W.fill(template, input.sample);
        return plain({
            tables,
            rendered,
            units,
            ranAs: Object.fromEntries(input.ranAs.map((code) => [code, W.ranAsWords(code)])),
            statuses: input.statuses.map((message) => text(W.statusLine(message))),
            replays: Object.fromEntries(Object.entries(input.replays).map(([name, events]) => [name, replay(events)])),
        });
    },
    stop: async () => {
        const results = {};
        for (const [name, scenario] of Object.entries(input.scenarios)) {
            const log = [];
            const timers = [];
            const said = [];
            let socket = null;
            let now = 0;
            let state = P.start(0);
            const keep = (next) => { state = next; for (const line of state.said) said.push(W.said(line)); };
            const deps = {
                open: () => { socket = fakeSocket(log); return socket; },
                later: (fn, ms) => { timers.push({ fn, ms, live: true }); log.push(['later', ms]); return timers.length - 1; },
                cancelLater: (id) => { if (timers[id]) timers[id].live = false; },
                random: () => 0,
            };
            const callbacks = {
                onFrame: (frame) => {
                    log.push(['frame', frame.type, frame.metadata?.state ?? null]);
                    keep(P.observe(state, frame, now, W.statusLine));
                },
                onLost: () => { log.push(['lost']); keep(P.lose(state, now)); },
                onDone: (response) => log.push(['done', response.cancelled ?? null, (response.steps ?? []).length]),
                onError: (message) => log.push(['error', message]),
                onToken: noop, onThinking: noop, onVerification: noop, onToolCall: noop,
                onReasoningStep: noop, onReasoningDone: noop, onVisionDelegation: noop,
                onStatus: noop, onCodingEvent: noop, onMetadata: noop, onReconnecting: noop,
            };
            const connection = S.openStream(
                { payload: { conversation_id: 'c1' }, conversationId: 'c1', sendError: 'Failed to send chat request' },
                callbacks, deps,
            );
            socket.onopen();
            const deliver = (frames) => {
                for (const [t, frame] of frames) { now = t; socket.onmessage({ data: JSON.stringify(frame) }); }
            };
            deliver(scenario.before);
            let ending = null;
            if (scenario.stop !== null) {
                now = scenario.stop;
                keep(P.stop(state, now));
                ending = connection.stop(async () => { log.push(['cancel-request']); });
            }
            deliver(scenario.after);
            if (scenario.close !== null) socket.onclose({ code: scenario.close });
            if (scenario.expire) {
                now = scenario.expire;
                for (const timer of timers) if (timer.live) { timer.live = false; timer.fn(); }
            }
            const end = ending === null ? null : await ending;
            results[name] = { end, log, said, snapshot: snap(state, now) };
        }
        return plain({ results, wait: S.STOP_WAIT_MS });
    },
};

const result = await run[clause]();
console.log('RESULT ' + JSON.stringify(result));
console.log('PASS ' + clause);
"""


def _node(clause, data=None):
    """Runs one driver clause over the three modules and returns its result."""
    out = run_ts(_MODULES, _DRIVER, clause, env={"OO_INPUT": json.dumps(data)})
    results = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, f"the driver printed no single RESULT line:\n{out}"
    return json.loads(results[0][len("RESULT "):])


# ---------------------------------------------------------------------------
# Fixture frames, shaped as the server sends them (docs/api-reference.md)
# ---------------------------------------------------------------------------
_LABELS = ("Gather", "Analyse", "Write")


def _step(seq, index, state, *, run="r1", kind="exec_pipeline", name="Research Assistant",
          total=3, label=None, parent=None, progress=None, reason=None, ran_as=None, duration_ms=None):
    """One ``pipeline_step`` frame carrying every field of the schema."""
    if label is None:
        label = _LABELS[index] if run == "r1" and index < len(_LABELS) else f"Step {index}"
    pipeline = kind == "exec_pipeline"
    return {"type": "pipeline_step", "content": "", "metadata": {
        "v": 1, "seq": seq, "run": run, "kind": kind, "name": name,
        "pipeline_id": "research" if pipeline else None, "parent": parent,
        "index": index, "total": total, "label": label,
        "step_type": "llm" if pipeline else None, "state": state,
        "progress": progress, "reason": reason, "ran_as": ran_as, "duration_ms": duration_ms,
    }}


def _progress(done, total, unit="sub_step"):
    return {"done": done, "total": total, "unit": unit}


def _frame(kind, content="", **metadata):
    frame = {"type": kind, "content": content}
    if metadata:
        frame["metadata"] = metadata
    return frame


_SENT = _frame("metadata", model="llama3", conversation_id="c1")
_PING = {"type": "ping", "timestamp": 1.0}


def _states(snapshot):
    """``{(run, index): (state, stage)}`` of one snapshot."""
    return {(s["run"], s["index"]): (s["state"], s["stage"]) for s in snapshot["steps"]}


def _stage(snapshot, run, index):
    return _states(snapshot)[(run, index)][1]


# ---------------------------------------------------------------------------
# LD1: the reducer is total over the event table
# ---------------------------------------------------------------------------
_LD1_EVENTS = [
    ["start", 0],
    ["frame", 10, _SENT],
    ["frame", 20, _step(1, 0, "pending")],
    ["frame", 30, _step(2, 1, "pending")],
    ["frame", 40, _step(3, 2, "pending")],
    ["frame", 50, _frame("nonsense", "x", seq=4, run="r1", index=0, state="done")],
    ["frame", 60, _step(4, 0, "running")],
    ["frame", 70, _step(5, 0, "running", progress=_progress(0, 4))],
    ["frame", 80, _step(6, 0, "running", progress=_progress(1, 4))],
    ["frame", 90, _step(6, 0, "running", progress=_progress(3, 4))],
    ["frame", 100, _step(8, 0, "running", progress=_progress(4, 4))],
    ["frame", 110, _step(7, 0, "running", progress=_progress(2, 4))],
    ["frame", 120, _step(9, 0, "running", progress=_progress(1, 4))],
    ["frame", 130, _step(10, 0, "done", duration_ms=1200)],
    ["frame", 140, _step(11, 1, "skipped")],
    ["frame", 150, _step(12, 2, "running")],
    ["frame", 160, _step(13, 2, "exploded")],
    ["frame", 170, _step(14, 2, "running", progress=_progress(2, 4))],
    ["frame", 180, _step(15, 0, "pending", run="r2", kind="consensus", name="consensus", total=2,
                         label="Query the models")],
    ["frame", 190, _step(16, 1, "pending", run="r2", kind="consensus", name="consensus", total=2,
                         label="Compare")],
    ["frame", 200, _step(17, 0, "running", run="r2", kind="consensus", name="consensus", total=2,
                         label="Query the models", progress=_progress(1, 3, "model"))],
    ["frame", 210, _step(18, 0, "cancelled", run="r2", kind="consensus", name="consensus", total=2,
                         label="Query the models", reason="Stop all engaged")],
    ["frame", 220, _step(19, 1, "not_run", run="r2", kind="consensus", name="consensus", total=2,
                         label="Compare", reason="Stop all engaged")],
]

# (event position, (run, index), (state, stage)) read after that event.
_LD1_EXPECTED = [
    (2, ("r1", 0), ("pending", "seed")),
    (4, ("r1", 2), ("pending", "seed")),
    (6, ("r1", 0), ("running", "crook")),
    (7, ("r1", 0), ("running", "crook")),
    (8, ("r1", 0), ("running", "flag")),
    (9, ("r1", 0), ("running", "flag")),
    (10, ("r1", 0), ("running", "leaf")),
    (11, ("r1", 0), ("running", "leaf")),
    (12, ("r1", 0), ("running", "leaf")),
    (13, ("r1", 0), ("done", "bulb")),
    (14, ("r1", 1), ("skipped", "soil")),
    (15, ("r1", 2), ("running", "crook")),
    (16, ("r1", 2), ("running", "crook")),
    (17, ("r1", 2), ("running", "leaf")),
    (18, ("r2", 0), ("pending", "seed")),
    (20, ("r2", 0), ("running", "flag")),
    (21, ("r2", 0), ("cancelled", "flag")),
    (22, ("r2", 1), ("not_run", "seed")),
]


def test_ld1_the_reducer_is_total_and_grows_only_on_a_true_fraction():
    snaps = _node("replay", _LD1_EVENTS)
    assert len(snaps) == len(_LD1_EVENTS), "one snapshot per event"
    for at, key, want in _LD1_EXPECTED:
        got = _states(snaps[at]).get(key)
        assert got == tuple(want), (
            f"after event {at} ({_LD1_EVENTS[at][2].get('type') if len(_LD1_EVENTS[at]) > 2 else ''}) "
            f"step {key} is {got}, not {want}"
        )
    assert snaps[5]["steps"] == snaps[4]["steps"], "an unknown frame type changes nothing"
    assert snaps[16]["steps"] == snaps[15]["steps"], "an unknown step state changes nothing"
    assert snaps[9]["steps"] == snaps[8]["steps"], "a frame whose seq was seen is a duplicate"
    for snap in snaps:
        for step in snap["steps"]:
            assert (step["stage"] == "bulb") == (step["state"] == "done"), (
                f"the bulb is drawn on done and only on done: {step}"
            )
    # Stages never decrease while a step runs.
    reached = {}
    for snap in snaps:
        for step in snap["steps"]:
            key = (step["run"], step["index"])
            if step["state"] in ("running", "cancelled", "failed") and step["stage"] in _GROWTH:
                rank = _GROWTH.index(step["stage"])
                assert rank >= reached.get(key, 0), f"step {key} fell back to {step['stage']}"
                reached[key] = rank
    assert snaps[-1]["rows"] == ["r1", "r2"], f"each top-level run is one row: {snaps[-1]['rows']}"


# ---------------------------------------------------------------------------
# LD2: growth reads only progress and a child run's counts
# ---------------------------------------------------------------------------
_LD2_TOKEN = [
    ["start", 0],
    ["frame", 10, _SENT],
    ["frame", 20, _step(1, 0, "pending", total=2)],
    ["frame", 30, _step(2, 1, "pending", total=2)],
    ["frame", 40, _step(3, 0, "running", total=2)],
    ["frame", 50, _frame("thinking", "Let me see")],
    ["frame", 60, _frame("thinking", " more")],
    ["frame", 70, _frame("token", "The")],
    ["frame", 80, _frame("token", " answer")],
    ["tick", 60000],
]

_CHILD = {"run": "c1", "kind": "reasoning", "name": "decompose"}


def _child(seq, index, state, total):
    return _step(seq, index, state, run="c1", kind="reasoning", name="decompose", total=total,
                 label=f"Question {index + 1}", parent={"run": "p1", "index": 0})


_LD2_CHILD = [
    ["start", 0],
    ["frame", 10, _step(1, 0, "pending", run="p1", total=2, label="Research")],
    ["frame", 20, _step(2, 1, "pending", run="p1", total=2, label="Write")],
    ["frame", 30, _step(3, 0, "running", run="p1", total=2, label="Research")],
    ["frame", 40, _child(4, 0, "pending", None)],
    ["frame", 50, _child(5, 0, "running", None)],
    ["frame", 60, _child(6, 0, "done", None)],
    ["frame", 70, _child(7, 1, "pending", 4)],
    ["frame", 80, _child(8, 2, "pending", 4)],
    ["frame", 90, _child(9, 3, "pending", 4)],
    ["frame", 100, _child(10, 1, "running", 4)],
    ["frame", 110, _child(11, 1, "done", 4)],
]

_LD2_CHILD_STAGES = ["crook", "crook", "crook", "crook", "flag", "flag", "flag", "flag", "leaf"]

_CLOCK = re.compile(
    r"\bDate\b|\bperformance\b|\bsetTimeout\b|\bsetInterval\b|\brequestAnimationFrame\b"
    r"|\bqueueMicrotask\b|\bnow\s*\(\s*\)"
)


@pytest.mark.parametrize("half", ("token", "child", "clock"))
def test_ld2_growth_reads_only_progress_and_child_counts(half):
    if half == "clock":
        for sample in ("const t = Date.now();", "performance.now()", "setTimeout(grow, 10000)",
                       "setInterval(tick, 1000)", "requestAnimationFrame(draw)", "const at = now();"):
            assert _CLOCK.search(sample), f"the probe reads a clock in {sample!r}"
        assert not _CLOCK.search("export function observe(state, frame, now) { return now - state.at; }"), (
            "the probe reads a time handed in as an argument as no clock"
        )
        code = _nav._script(_PROGRESS)
        found = sorted({m.group(0) for m in _CLOCK.finditer(code)})
        assert not found, f"the reducer calls no clock: {found}"
        return

    if half == "token":
        snaps = _node("replay", _LD2_TOKEN)
        stages = [_stage(snap, "r1", 0) for snap in snaps[4:]]
        assert stages == ["crook"] * len(stages), (
            f"without a fraction the running plant stays at the crook, whatever the frames or the time: {stages}"
        )
        assert snaps[4]["line"] == "Sent to llama3", f"the line before the first word: {snaps[4]['line']!r}"
        assert snaps[5]["line"] == "Reasoning", f"a first thinking frame changes the words: {snaps[5]['line']!r}"
        assert snaps[7]["line"] is None, f"a first token retires the line: {snaps[7]['line']!r}"
        return

    snaps = _node("replay", _LD2_CHILD)
    stages = [_stage(snap, "p1", 0) for snap in snaps[3:]]
    assert stages == _LD2_CHILD_STAGES, (
        f"a parent step grows on its child run's finished count over a known total, and only then: {stages}"
    )
    for snap in snaps[1:]:
        assert snap["rows"] == ["p1"], f"a nested run never makes a second row: {snap['rows']}"
    assert ("c1", 1) in _states(snaps[-1]), "the child run's steps are kept, under their parent"


# ---------------------------------------------------------------------------
# LD5: silence
# ---------------------------------------------------------------------------
_LD5_EVENTS = [
    ["start", 0],
    ["frame", 100, _SENT],
    ["frame", 10000, _PING],
    ["frame", 20000, _PING],
    ["frame", 30000, _PING],
    ["frame", 40000, _PING],
    ["tick", 45000],
    ["tick", 69999],
    ["tick", 70000],
    ["tick", 75000],
    ["frame", 80000, _PING],
    ["tick", 80500],
]


def test_ld5_thirty_seconds_without_a_frame_stops_the_motion_and_says_so():
    snaps = _node("replay", _LD5_EVENTS)
    for at in (6, 7):
        assert not snaps[at]["silent"] and snaps[at]["moving"], (
            f"pings keep the loader alive: at {_LD5_EVENTS[at][1]} ms it reads {snaps[at]}"
        )
        assert snaps[at]["line"] == "Sent to llama3", f"a ping changes nothing visible: {snaps[at]['line']!r}"
    quiet = snaps[8]
    assert quiet["silent"] and not quiet["moving"], f"30 s after the last ping the motion stops: {quiet}"
    assert quiet["line"] == "No word from the server for 30 s", f"and the line says so: {quiet['line']!r}"
    assert quiet["said"] == ["No word from the server for 30 s."], f"once, in the status region: {quiet['said']}"
    assert snaps[9]["silent"] and snaps[9]["said"] == [], f"the silence is announced once: {snaps[9]['said']}"
    back = snaps[10]
    assert not back["silent"] and back["moving"], f"the next frame, a ping included, resumes the motion: {back}"
    assert back["line"] == "Sent to llama3", f"and the line comes back: {back['line']!r}"
    assert not snaps[11]["silent"], "silence is measured from the last frame"


# ---------------------------------------------------------------------------
# LD7: the status region
# ---------------------------------------------------------------------------
_LD7_EVENTS = [
    ["start", 0],
    ["frame", 100, _SENT],
    ["frame", 200, _frame("status", message="[>] Web search for: onion storage...")],
    ["frame", 300, _step(1, 0, "pending")],
    ["frame", 310, _step(2, 1, "pending")],
    ["frame", 320, _step(3, 2, "pending")],
    ["frame", 330, _step(4, 0, "running")],
    ["frame", 400, _step(5, 0, "running", progress=_progress(1, 3))],
    ["frame", 500, _step(6, 0, "running", progress=_progress(2, 3))],
    ["frame", 600, _step(7, 0, "running", progress=_progress(3, 3))],
    ["frame", 700, _step(8, 0, "done", duration_ms=600)],
    ["frame", 2000, _step(9, 1, "running")],
    ["frame", 5000, _PING],
    ["frame", 6000, _step(10, 1, "failed", reason="HTTP 500 from the model server", duration_ms=4000)],
    ["frame", 6050, _step(11, 2, "running")],
    ["frame", 6100, _frame("token", "Onions")],
    ["frame", 9000, _step(12, 2, "done", duration_ms=2950)],
    ["frame", 9500, _frame("done", "Onions keep", conversation_id="c1", duration_ms=9400, steps=[
        _step(8, 0, "done", duration_ms=600)["metadata"],
        _step(10, 1, "failed", reason="HTTP 500 from the model server", duration_ms=4000)["metadata"],
        _step(12, 2, "done", duration_ms=2950)["metadata"],
    ])],
]

_LD7_SAID = [
    "Sent to llama3.",
    "Research Assistant started: 3 steps.",
    "Step 1 of 3 started: Gather.",
    "Step 2 of 3 started: Analyse.",
    "Step 2 of 3 failed: HTTP 500 from the model server.",
    "Step 3 of 3 started: Write.",
    "Reply complete: 3 steps, 1 failed.",
]


def test_ld7_the_status_region_announces_phases_and_step_transitions_coalesced():
    out = _node("announce", {"events": _LD7_EVENTS, "until": 12000, "step": 50})
    said = [text for _, text in out["said"]]
    assert said == _LD7_SAID, (
        "only phases and step transitions are announced, never a sub-step, a status or a ping:\n"
        + "\n".join(said)
    )
    for text in said:
        assert not re.search(r"sub-steps|models finished|samples done", text), f"no sub-progress: {text!r}"
        assert not re.search(r", \d+ s\b", text), f"never the seconds: {text!r}"
        assert not re.search(r"componion|bulbe|beetle|scarab", text, re.I), f"never the being's name: {text!r}"
    coalesce, gap = out["coalesce"], out["gap"]
    assert (coalesce, gap) == (400, 1000), f"a burst ends after 400 ms of calm, one announcement a second: {out}"
    spoken = out["spoken"]
    assert spoken, "the region speaks"
    for (t1, _), (t2, _) in zip(spoken, spoken[1:]):
        assert t2 - t1 >= gap, f"at most one announcement per second: {spoken}"
    for t, text in spoken:
        before = [(at, s) for at, s in out["said"] if at <= t]
        assert before and before[-1][1] == text, f"only the last of a burst is spoken: {t} {text!r}"
        assert t - before[-1][0] >= coalesce, f"after 400 ms of calm: {t} {text!r}"
    words = [text for _, text in spoken]
    assert "Sent to llama3." not in words, f"a burst speaks only its last: {words}"
    assert words[-1] == _LD7_SAID[-1], f"the last announcement is spoken: {words}"


# ---------------------------------------------------------------------------
# LD8: the words
# ---------------------------------------------------------------------------
_SAMPLE = {
    "model": "llama3", "n": 3, "max": 3, "k": 2, "i": 2, "s": 14, "tool": "web_fetch",
    "reason": "Stop all engaged", "message": "HTTP 500 from the model server", "label": "Analyse",
    "run": "Research Assistant", "counts": "3 steps, 1 failed", "done": 1, "total": 3,
}

_RAN_AS = ("direct", "tools", "code_verify", "think", "web_search", "think_tools", "reasoning",
           "consensus", "self_correct", "cascading", "speculative")

_MAPPED = [
    ("[>] Web search for: onion storage...", "Searching the web"),
    ("[OK] 5 search results injected", "Web search: 5 results"),
    ("[!] Web search returned no results", "Web search: no results"),
    ("[!] Web search failed: timeout", "Web search failed"),
    ("[!] Web search skipped (kill switch engaged)", "Web search skipped: the search kill switch is on"),
    ("[>] Project context: L2 trigger (confidence=0.81)", "Reading project context"),
    ("[OK] Project context injected: 7 chunks, ~900 tokens", "Project context: 7 passages"),
    ("[>] Archive retrieval: 2 relevant message(s) injected from history", "Fitting the history into the context"),
    ("[>] Optimizer: 12 messages, ~3,000 tokens (preset=x, trimmed=0t, ...)", "Fitting the history into the context"),
    ("[>] Multi-turn: 8 previous messages, ~2,000 tokens", "Fitting the history into the context"),
    ("[CACHE] Hit for llama3 (key=abcd1234...)", "Found in the cache"),
    ("[SEMANTIC] Hit for llama3 (key=abcd1234...)", "Found in the cache"),
    ("[!] Ollama offline -- request queued", "Queued: Ollama is offline"),
    ("Analyzing image with llava...", "Reading the image with llava"),
]

# Statuses the table does not map: the runner's own step line (kept for the
# terminal), one that precedes the governor's admission, the error line, and
# one the server may add later. Each changes nothing visible.
_UNMAPPED = [
    "Step 2/3: Analyse",
    "[>] Generating with llama3 (temp=0.7)...",
    "Generating initial response...",
    "[ERR] Error: boom",
    "[>] A status the table has never seen",
]

_DRAWING = re.compile(r"\b(dried|dry|withered|wilted|dead|pressed|herbarium)\b", re.I)

# The only keys whose words may say "stopped": a safety mechanism ended the work.
_STOP_KEYS = {"stopped", "stopped_because", "state_cancelled", "state_cancelled_because",
              "count_cancelled", "summary_stopped", "summary_stopped_open"}


def _failed_run():
    """A run whose second step failed, to the reply's end."""
    return [
        ["start", 0],
        ["frame", 10, _SENT],
        ["frame", 20, _step(1, 0, "pending")],
        ["frame", 30, _step(2, 1, "pending")],
        ["frame", 40, _step(3, 2, "pending")],
        ["frame", 50, _step(4, 0, "running")],
        ["frame", 60, _step(5, 0, "done", duration_ms=10)],
        ["frame", 70, _step(6, 1, "running")],
        ["frame", 80, _step(7, 1, "failed", reason="HTTP 500 from the model server", duration_ms=10)],
        ["frame", 90, _step(8, 2, "running")],
        ["frame", 100, _step(9, 2, "done", duration_ms=10)],
        ["frame", 110, _frame("done", "text", conversation_id="c1", duration_ms=58000, steps=[
            _step(5, 0, "done", duration_ms=10)["metadata"],
            _step(7, 1, "failed", reason="HTTP 500 from the model server", duration_ms=10)["metadata"],
            _step(9, 2, "done", duration_ms=10)["metadata"],
        ])],
    ]


def _unmapped_run():
    """A mapped status, then every unmapped one: the line must not move."""
    events = [["start", 0], ["frame", 10, _SENT],
              ["frame", 20, _frame("status", message="[>] Web search for: onions...")]]
    for at, message in enumerate(_UNMAPPED):
        events.append(["frame", 30 + at, _frame("status", message=message)])
    return events


def _words():
    return _node("words", {
        "sample": _SAMPLE,
        "ranAs": list(_RAN_AS) + ["some_new_code"],
        "statuses": [m for m, _ in _MAPPED] + _UNMAPPED,
        "replays": {"failed": _failed_run(), "unmapped": _unmapped_run()},
    })


_STRING = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"|`(?:[^`\\]|\\.)*`")


def _literals():
    """Every string literal of the word table's source, comments removed."""
    return [m.group(0)[1:-1] for m in _STRING.finditer(_nav._script(_WORDS))]


def _nets():
    loaded, restore = isolate(targets={_ETHICS: REPO / "opti_oignon" / "allium" / "ethics.py"},
                              packages=("opti_oignon.allium",))
    return loaded[_ETHICS], restore


def _every_word(out):
    """Every template, rendered word and literal the loader can show."""
    tables = out["tables"]
    found = []
    for table in tables.values():
        found.extend(table.values())
    found.extend(out["rendered"].values())
    found.extend(out["units"].values())
    found.extend(w for w in out["ranAs"].values() if w)
    found.extend(w for w in out["statuses"] if w)
    for snaps in out["replays"].values():
        for snap in snaps:
            found.extend(w for w in (snap["line"], snap["summary"]) if w)
            found.extend(snap["said"])
            found.extend(step["words"] for step in snap["steps"])
    return found


@pytest.mark.parametrize("half", ("nets", "ascii", "ran_as", "drawing", "stopped", "status"))
def test_ld8_every_loader_word_passes_the_nets_and_says_no_code_and_no_drawing(half):
    out = _words()
    every = _every_word(out)
    literals = _literals()
    assert len(out["tables"]["WORDS"]) >= 40 and literals, "the table is read, and it is not empty"

    if half == "nets":
        ethics, restore = _nets()
        try:
            assert ethics.class_hits("Waiting for llama3") and ethics.class_hits("Thinking"), (
                "the class net refuses the governed words"
            )
            refused = {w: (ethics.class_hits(w), ethics.banned_hits(w))
                       for w in set(every) | set(literals)
                       if ethics.class_hits(w) or ethics.banned_hits(w)}
        finally:
            restore()
        assert not refused, f"every loader word passes both nets: {refused}"
        return

    if half == "ascii":
        assert re.search(r"[^ -~]", "Research Assistant \u00b7 3 steps"), "the probe sees a middle dot"
        bad = sorted({w for w in set(every) | set(literals) if re.search(r"[^ -~]", w)})
        assert not bad, f"every loader word is printable ASCII, a comma its separator: {bad}"
        return

    if half == "ran_as":
        ran_as = out["ranAs"]
        assert set(out["tables"]["RAN_AS"]) == set(_RAN_AS), (
            f"the table has the eleven ran_as rows: {sorted(out['tables']['RAN_AS'])}"
        )
        for code in _RAN_AS:
            words = ran_as[code]
            assert words and words.startswith("Ran as "), f"{code} is said in words: {words!r}"
            assert words != code and not ("_" in code and code in words), (
                f"the code itself is never shown: {code} -> {words!r}"
            )
        assert ran_as["some_new_code"] is None, "a code the table does not know is not shown"
        for w in every:
            assert not any(re.search(rf"\b{re.escape(c)}\b", w) for c in _RAN_AS if "_" in c), (
                f"no word carries a raw code: {w!r}"
            )
        return

    if half == "drawing":
        assert _DRAWING.search("Its plant is pressed") and _DRAWING.search("Dried at step 2"), "the probe reads"
        bad = sorted({w for w in set(every) | set(literals) if _DRAWING.search(w)})
        assert not bad, f"no word describes the drawing: {bad}"
        return

    if half == "stopped":
        words = out["tables"]["WORDS"]
        stop_keys = {key for key, template in words.items() if re.search(r"stop", template, re.I)}
        assert stop_keys and stop_keys <= _STOP_KEYS, (
            f"only the words of a safety stop say stopped: {sorted(stop_keys - _STOP_KEYS)}"
        )
        failed = out["replays"]["failed"]
        last = failed[-1]
        failed_words = [step["words"] for snap in failed for step in snap["steps"] if step["state"] == "failed"]
        said = [s for snap in failed for s in snap["said"]]
        assert failed_words and "Failed" in failed_words, f"a failed step says Failed: {failed_words}"
        assert last["summary"] == "Research Assistant, 3 steps, 1 failed, 58 s", (
            f"the summary of a run with a failure: {last['summary']!r}"
        )
        for w in failed_words + said + [last["summary"]]:
            assert not re.search(r"stop", w, re.I), f"a failure is never said stopped: {w!r}"
        return

    statuses = out["statuses"]
    mapped = statuses[:len(_MAPPED)]
    assert mapped == [want for _, want in _MAPPED], (
        "each status the server sends is said in the table's words:\n"
        + "\n".join(f"{m!r} -> {g!r}" for (m, _), g in zip(_MAPPED, mapped))
    )
    unmapped = statuses[len(_MAPPED):]
    assert unmapped == [None] * len(_UNMAPPED), f"a status the table does not map has no words: {unmapped}"
    snaps = out["replays"]["unmapped"]
    lines = [snap["line"] for snap in snaps[2:]]
    assert lines == ["Searching the web"] * len(lines), f"an unmapped status changes nothing visible: {lines}"


# ---------------------------------------------------------------------------
# LD13: no step status is read
# ---------------------------------------------------------------------------
_STEP_STATUS = re.compile(
    r"Step\s*(?:\\s[+*]?\s*)?(?:\d+|\\d[+*]?|\(\\d[+*]?\)|\(\?<\w+>\\d[+*]?\)|[ikN]|\$\{[^}]*\})\s*\\?/"
    r"|Step\s+\S+\s+done"
    r"|\\?\[ERR\\?\]\s*Step"
    r"|startsWith\(\s*['\"`]Step\b"
)

_STEP_FIXTURES = (
    "const m = /^Step (\\d+)\\/(\\d+): (.*)$/.exec(message);",
    "const m = /Step \\d+\\/\\d+/.test(status);",
    "if (message.startsWith('Step ')) { parse(message); }",
    "if (text.includes('[ERR] Step')) failed = true;",
    "const done = /Step \\d+ done/.test(status);",
    "const re = new RegExp(`Step ${i}/${n}`);",
)


def test_ld13_no_module_under_lib_reads_a_step_status():
    for sample in _STEP_FIXTURES:
        assert _STEP_STATUS.search(sample), f"the probe reads a step status parser in {sample!r}"
    assert not _STEP_STATUS.search("// Step 1: Approve"), "a numbered comment is not a step status"
    listed = files((".ts", ".js", ".svelte"), within=_LIB)
    found = {}
    for path in listed:
        hits = [m.group(0) for m in _STEP_STATUS.finditer(_nav._code(path))]
        if hits:
            found[path] = hits
    assert not found, f"no module reads the runner's step lines; the loader reads pipeline_step: {found}"


# ---------------------------------------------------------------------------
# LD17: a stream that closes with steps open
# ---------------------------------------------------------------------------
_LD17_EVENTS = [
    ["start", 0],
    ["frame", 10, _SENT],
    ["frame", 20, _step(1, 0, "pending")],
    ["frame", 30, _step(2, 1, "pending")],
    ["frame", 40, _step(3, 2, "pending")],
    ["frame", 50, _step(4, 0, "running")],
    ["frame", 800, _step(5, 0, "done", duration_ms=750)],
    ["frame", 900, _step(6, 1, "running", progress=_progress(1, 3))],
    ["lost", 5000],
    ["tick", 5100],
    ["lost", 5200],
]


@pytest.mark.parametrize("half", ("unknown", "announce", "summary"))
def test_ld17_a_closed_stream_leaves_its_open_steps_unknown(half):
    snaps = _node("replay", _LD17_EVENTS)
    closed = snaps[8]
    if half == "unknown":
        states = _states(closed)
        assert states[("r1", 0)] == ("done", "bulb"), f"a finished step keeps its end: {states}"
        assert states[("r1", 1)] == ("unknown", "flag"), f"a running step is unknown at its stage: {states}"
        assert states[("r1", 2)] == ("unknown", "seed"), f"a step to come is unknown as a seed: {states}"
        for step in closed["steps"][1:]:
            assert step["look"]["pattern"] == "dotted" and step["look"]["ink"] == "rule", (
                f"an unknown end is its last drawing, dotted, in rule: {step}"
            )
            assert step["look"]["mark"] is None, f"an unknown end carries no mark: {step}"
        assert closed["end"] == "lost" and not closed["moving"], f"nothing moves: {closed}"
        return
    if half == "announce":
        assert closed["said"] == [
            "Connection closed before the reply ended. Outcome unknown for 2 of 3 steps."
        ], f"one announcement for the run: {closed['said']}"
        for step in closed["steps"][1:]:
            assert step["words"] == "Outcome unknown: the connection closed before this step ended", (
                f"each open step says its outcome is unknown: {step}"
            )
        assert snaps[9]["said"] == [] and snaps[10]["said"] == [], "and never again"
        return
    assert closed["summary"] == "Research Assistant, connection closed at step 2 of 3", (
        f"the summary says where the stream closed: {closed['summary']!r}"
    )
    assert not re.search(r"\d+ s\b", closed["summary"]), "with no duration: the server never sent one"


# ---------------------------------------------------------------------------
# LD20: a Stop reads the server's closing frames
# ---------------------------------------------------------------------------
_BEFORE = [
    [10, _SENT],
    [20, _step(1, 0, "pending")],
    [30, _step(2, 1, "pending")],
    [40, _step(3, 2, "pending")],
    [50, _step(4, 0, "running")],
    [3050, _step(5, 0, "done", duration_ms=3000)],
    [3100, _step(6, 1, "running")],
]

_CLOSING = [
    [9100, _step(7, 1, "cancelled", duration_ms=6000)],
    [9110, _step(8, 2, "not_run")],
    [9120, _frame("done", "Partial answer", conversation_id="c1", duration_ms=14000, cancelled=True, steps=[
        _step(5, 0, "done", duration_ms=3000)["metadata"],
        _step(7, 1, "cancelled", duration_ms=6000)["metadata"],
        _step(8, 2, "not_run")["metadata"],
    ])],
]

_SCENARIOS = {
    "answered": {"before": _BEFORE, "stop": 9000, "after": _CLOSING, "close": None, "expire": None},
    "unanswered": {"before": _BEFORE, "stop": 9000, "after": [], "close": None, "expire": 30000},
    "dropped": {"before": _BEFORE, "stop": None, "after": [], "close": 1006, "expire": None},
}


def _at(log, *entry):
    """The position of the first log entry that starts with ``entry``."""
    for i, item in enumerate(log):
        if tuple(item[:len(entry)]) == entry:
            return i
    return None


def _body(script, name):
    match = re.search(rf"\bfunction\s+{re.escape(name)}\s*\(", script)
    assert match, f"{name} is defined"
    start = script.index("{", script.index(")", match.end()))
    depth, at = 0, start
    while True:
        depth += {"{": 1, "}": -1}.get(script[at], 0)
        at += 1
        if depth == 0:
            return script[start:at]


def test_ld20_a_stop_reads_the_servers_closing_frames():
    out = _node("stop", {"scenarios": _SCENARIOS})
    wait = out["wait"]
    assert isinstance(wait, int) and 0 < wait <= 60000, f"the wait after a Stop is bounded: {wait}"
    results = out["results"]

    answered = results["answered"]
    log = answered["log"]
    cancel, close = _at(log, "cancel-request"), _at(log, "close")
    assert cancel is not None and close is not None and cancel < close, (
        f"the cancel is sent before the socket closes: {log}"
    )
    assert _at(log, "later", wait) is not None, f"the wait is held by its constant: {log}"
    after = [item for item in log[cancel:] if item[0] in ("frame", "done")]
    assert after[:3] == [["frame", "pipeline_step", "cancelled"], ["frame", "pipeline_step", "not_run"],
                         ["frame", "done", None]] and ["done", True, 3] in after, (
        f"the server's closing frames and its done are read after the cancel: {log}"
    )
    assert answered["end"] == "done" and _at(log, "lost") is None and _at(log, "error") is None, (
        f"a stop the server answered ends with its done: {answered['end']} {log}"
    )
    snap = answered["snapshot"]
    assert [(s["state"], s["stage"]) for s in snap["steps"]] == [
        ("done", "bulb"), ("cancelled", "crook"), ("not_run", "seed")
    ], f"the steps end as the server closed them: {snap['steps']}"
    assert snap["end"] == "stopped", f"the run reads stopped: {snap['end']}"
    assert answered["said"].count("Stopped.") == 1, f"Stopped is announced once: {answered['said']}"
    assert snap["summary"] == "Research Assistant, stopped at step 2 of 3, 14 s", (
        f"the summary says where it stopped: {snap['summary']!r}"
    )

    unanswered = results["unanswered"]
    log = unanswered["log"]
    assert unanswered["end"] == "unanswered", f"with no answer the wait ends the stop: {unanswered['end']}"
    cancel, close = _at(log, "cancel-request"), _at(log, "close")
    assert cancel is not None and close is not None and cancel < close, f"cancel, then close: {log}"
    assert _at(log, "lost") is not None and _at(log, "error") is None, (
        f"a stop that got no answer is a lost stream, not an error: {log}"
    )
    states = [s["state"] for s in unanswered["snapshot"]["steps"]]
    assert states == ["done", "unknown", "unknown"], f"only a stream closed without done is unknown: {states}"

    dropped = results["dropped"]
    states = [s["state"] for s in dropped["snapshot"]["steps"]]
    assert states == ["done", "unknown", "unknown"] and _at(dropped["log"], "lost") is not None, (
        f"a dropped connection leaves its open steps unknown: {states} {dropped['log']}"
    )
    assert _at(dropped["log"], "error") is not None, f"and says it lost the connection: {dropped['log']}"

    body = _body(_nav._script(_STORE), "cancelCurrentGeneration")
    assert re.search(r"\.stop\s*\(", body) and "cancelGeneration" in body, (
        "the store stops the stream through stop(), handing it the cancel request"
    )
    assert not re.search(r"\.cancel\s*\(", body), "the store never closes the socket before the cancel"


# ===========================================================================
# The drawing: the pixel tables, the strip that plays them, the pacing
# ===========================================================================
_PIXEL = f"{_LIB}/pixel"
_PLANT = f"{_PIXEL}/plantFrames.ts"
_ONION = f"{_PIXEL}/onionFrames.ts"
_PACING = f"{_LIB}/chat/pacing.ts"
_STRIP = f"{_LIB}/components/pixel/PixelStrip.svelte"

# Every look a step's plant can take: (state, stage).
_LOOKS = [
    ["pending", "seed"], ["not_run", "seed"], ["skipped", "soil"],
    ["running", "crook"], ["running", "flag"], ["running", "leaf"], ["done", "bulb"],
    ["failed", "crook"], ["failed", "flag"], ["failed", "leaf"],
    ["cancelled", "seed"], ["cancelled", "crook"], ["cancelled", "flag"], ["cancelled", "leaf"],
    ["unknown", "seed"], ["unknown", "crook"], ["unknown", "flag"], ["unknown", "leaf"],
]

_DRAW_DRIVER = r"""
const load = async (name) => (process.env[name] ? await import(process.env[name]) : null);
const P = await load('OO_PLANT');
const O = await load('OO_ONION');
const W = await load('OO_WORDS');
const G = await load('OO_PACING');
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT || 'null');
const plain = (value) => JSON.parse(JSON.stringify(value === undefined ? null : value));

const run = {
    tables: () => plain({
        plant: {
            w: P.PLANT_W, h: P.PLANT_H, soil: P.SOIL_ROW, pitch: P.PLANT_PITCH,
            frames: P.PLANT_FRAMES, inks: P.PLANT_INKS,
            rowGrounds: P.ROW_GROUNDS, sheetGround: P.SHEET_GROUND,
            looks: input.looks.map(([state, stage]) => ({ state, stage, strip: P.plantStrip(state, stage) })),
        },
        onion: {
            w: O.ONION_W, h: O.ONION_H, frames: O.ONION_FRAMES, inks: O.ONION_INKS,
            grounds: O.ONION_GROUNDS, strip: O.ONION_STRIP,
        },
    }),
    fit: () => plain(input.map(([steps, current, column, dpr]) => {
        const fit = P.fitRow(steps, current, column, dpr);
        const said = (key, n) => (n > 0 ? W.words({ key, values: { n } }) : null);
        return {
            ...fit,
            capacity: [P.rowCapacity(column, 3, dpr), P.rowCapacity(column, 2, dpr)],
            earlierWords: said('row_earlier', fit.earlier),
            laterWords: said('row_later', fit.later),
        };
    })),
    pace: () => {
        // The fake clock: every clock the runtime offers refuses to be read
        // while the schedule is computed; the times are handed in.
        const clocks = [
            [Date, 'now'], [globalThis.performance, 'now'],
            [globalThis, 'setTimeout'], [globalThis, 'setInterval'], [globalThis, 'queueMicrotask'],
        ];
        const kept = clocks.map(([owner, name]) => owner[name]);
        const read = [];
        clocks.forEach(([owner, name]) => {
            owner[name] = () => { read.push(name); throw new Error('a clock was read: ' + name); };
        });
        try {
            const out = {};
            for (const [name, s] of Object.entries(input)) {
                const plan = G.schedule(s.arrivals, s.start, s.end);
                out[name] = {
                    plan,
                    shown: s.probes.map((t) => G.shownAt(plan, t)),
                    next: s.probes.map((t) => G.nextAt(plan, t)),
                };
            }
            return plain({ out, read, min: G.STAGE_MS, waiting: G.MAX_WAITING });
        } finally {
            clocks.forEach(([owner, name], i) => { owner[name] = kept[i]; });
        }
    },
};

const result = await run[clause]();
console.log('RESULT ' + JSON.stringify(result));
console.log('PASS ' + clause);
"""


def _draw(clause, data=None, **modules):
    """Runs one clause of the drawing driver over the modules it names."""
    out = run_ts(modules, _DRAW_DRIVER, clause, env={"OO_INPUT": json.dumps(data)})
    results = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, f"the driver printed no single RESULT line:\n{out}"
    return json.loads(results[0][len("RESULT "):])


def _tables():
    return _draw("tables", {"looks": _LOOKS}, OO_PLANT=_PLANT, OO_ONION=_ONION)


def _resolvers():
    """``{palette id: Resolver}`` over the derivation layer and the three palette files."""
    from _colour import Resolver

    tokens = _ds._derivation()
    return {pid: Resolver(_ds._with_palette(tokens, pid), roles, scheme)
            for pid, (roles, scheme) in _ds._palettes().items()}


def _colour_of(resolver, pid, token):
    try:
        return resolver.colour(token)
    except (KeyError, ValueError) as exc:
        raise AssertionError(f"{pid}: {token} cannot be evaluated ({exc})") from None


def _ratio(ink, ground):
    from _colour import contrast, over

    return contrast(over(ink, ground), ground)


def _lab(colour):
    """CIELAB (D65, 2 degrees) of an opaque sRGB colour ``(r, g, b, ...)`` on 0-255."""
    def linear(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    r, g, b = (linear(c) for c in colour[:3])
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def _delta_e(a, b):
    """The CIE 1976 distance of two colours."""
    return math.dist(_lab(a), _lab(b))


def _hue(colour):
    """The CIELAB hue angle, in degrees."""
    _, a, b = _lab(colour)
    return math.degrees(math.atan2(b, a)) % 360


# ---------------------------------------------------------------------------
# LD9: every loader colour is an app token, and each reads on its ground
# ---------------------------------------------------------------------------
# The running line's ground, in the list under the row (the row never stands on it).
_LINE_GROUND = "--oo-bg-tint-1"
_ROLE_READ = re.compile(r"--oo-role-")
_COLOUR_CALL = re.compile(r"(?<![\w-])(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(", re.I)
_FORCED = re.compile(r"@media\s*\(\s*forced-colors\s*:\s*active\s*\)")


def _literals_in(path, text=None):
    """The colour literals a drawing file spells, outside what forced colours
    keep (the system colours), and every role it reads."""
    text = read(path) if text is None else text
    code = _nav._code(path, text)
    found = [literal for _, literal in _ds._hexes(code)]
    found += [m.group(0) for m in _COLOUR_CALL.finditer(code)]
    for _, body, context in _ds._rules(_ds._css_of(path, text)):
        if any(_FORCED.fullmatch(c) for c in context):
            continue
        for _, value in _ds._declarations(body):
            found += _ds._colour_literals(value)
    found += _ROLE_READ.findall(text)
    return found


@pytest.mark.parametrize("half", ("tokens", "marks", "row", "straw"))
def test_ld9_every_loader_colour_is_an_app_token_that_reads_on_its_ground(half):
    tables = _tables()
    plant, onion = tables["plant"], tables["onion"]
    resolvers = _resolvers()
    assert sorted(resolvers) == ["day", "high-contrast", "night"], f"three palettes: {sorted(resolvers)}"

    if half == "tokens":
        sample = (
            '<svg><path fill="#4B5E43"/></svg><style>.a { fill: rgb(1, 2, 3); color: var(--oo-role-bg); }\n'
            "@media (forced-colors: active) { .a { fill: CanvasText; } }</style>"
        )
        caught = _literals_in("frontend/src/x.svelte", sample)
        assert "#4B5E43" in caught and "rgb(" in caught and "--oo-role-" in caught, (
            f"the census reads a hex, a colour function and a role: {caught}"
        )
        assert "CanvasText" not in caught, f"and passes the system colours forced colours keep: {caught}"
        listed = files((".ts",), within=_PIXEL) + files((".svelte",), within=f"{_LIB}/components/pixel")
        assert {_PLANT, _ONION, _STRIP} <= set(listed), f"the drawing's files are read: {listed}"
        spelled = {path: found for path in listed for found in [_literals_in(path)] if found}
        assert not spelled, f"the drawing spells no colour and reads no role: {spelled}"

        declared = set(_ds._derivation()) | {n for given in _ds._palette_overrides().values() for n in given}
        assert "--oo-bg-base" in declared and "--oo-role-bg" not in declared, "the census reads the app tokens"
        named = list(plant["inks"].values()) + list(onion["inks"].values())
        named += plant["rowGrounds"] + [plant["sheetGround"]] + onion["grounds"]
        undeclared = sorted({n for n in named if n not in declared or n.startswith("--oo-role-")})
        assert not undeclared, f"every ink and ground of the tables is an app token: {undeclared}"
        for label, grids, inks in (
            ("plant", list(plant["frames"].values())
             + [grid for look in plant["looks"] for grid in look["strip"]["frames"]], plant["inks"]),
            ("onion", onion["frames"] + onion["strip"]["frames"], onion["inks"]),
        ):
            drawn = {c for grid in grids for line in grid for c in line} - {"."}
            assert drawn and not drawn - set(inks), (
                f"every character the {label} draws has its ink: {sorted(drawn - set(inks))}"
            )
        return

    def colour(pid, token):
        return _colour_of(resolvers[pid], pid, token)

    if half == "marks":
        marks = {plant["inks"]["g"], plant["inks"]["s"], onion["inks"]["o"]}
        assert marks == {"--oo-acc-mark", "--oo-acc-mark-2"}, (
            f"what grows, the seed, the bulb and the onion are drawn in the brand marks: {sorted(marks)}"
        )
        grounds = set(plant["rowGrounds"]) | {plant["sheetGround"], onion["inks"]["x"], _LINE_GROUND}
        grounds |= set(onion["grounds"])
        assert _ratio(colour("day", "--oo-status-ok"), colour("day", _LINE_GROUND)) < 3.0, (
            "the check sees a pair under 3:1 (the finished leaves on tint_1 by day)"
        )
        low = [
            f"{pid}: {mark} on {ground}: {ratio:.2f}"
            for pid in resolvers for mark in sorted(marks) for ground in sorted(grounds)
            for ratio in [_ratio(colour(pid, mark), colour(pid, ground))] if ratio < 3.0
        ]
        assert not low, "a mark under 3:1 on a ground the loader uses:\n  " + "\n  ".join(low)
        return

    if half == "row":
        inks = {plant["inks"][c] for c in "kwty"}
        assert len(inks) == 4, f"the finished leaves, the seeds, the tag and the kept plant: {sorted(inks)}"
        assert plant["rowGrounds"] and plant["sheetGround"], "the row's grounds and the sheet's are named"
        assert _LINE_GROUND not in plant["rowGrounds"], "the row never stands on tint_1"
        pairs = [(ink, ground) for ink in sorted(inks) for ground in plant["rowGrounds"]]
        pairs.append((plant["inks"]["y"], plant["sheetGround"]))
        low = [
            f"{pid}: {ink} on {ground}: {ratio:.2f}"
            for pid in resolvers for ink, ground in pairs
            for ratio in [_ratio(colour(pid, ink), colour(pid, ground))] if ratio < 3.0
        ]
        assert not low, "a drawing under 3:1 on the row or on the sheet:\n  " + "\n  ".join(low)
        return

    from _colour import parse

    dried = plant["inks"]["y"]
    others = {plant["inks"][c] for c in "gmwkt"}
    assert len(others) == 5 and dried not in others, (
        f"the straw stands apart from what grows, the stopped plant, the seeds, the finished leaves "
        f"and the tag: {sorted(others)}"
    )
    assert not 88 <= _hue(parse("#8A7240")) <= 100, "the probe sees the brown the straw replaced"
    assert _delta_e(parse("#908153"), parse("#908153")) == 0, "a colour stands at no distance from itself"
    for pid in resolvers:
        straw = colour(pid, dried)
        assert 88 <= _hue(straw) <= 100, f"{pid}: the straw is yellow, not brown: hue {_hue(straw):.1f}"
        near = {other: round(_delta_e(straw, colour(pid, other)), 1) for other in sorted(others)}
        assert min(near.values()) >= 15, f"{pid}: the straw stands 15 apart from each ink of the row: {near}"


# ---------------------------------------------------------------------------
# LD10: the frame tables
# ---------------------------------------------------------------------------
# Above three steps a second, at most a tenth of the cells change per step.
_FAST_MS = 1000 / 3
_SHARE = 0.10
_MOTION = re.compile(r"^([\w-]+)\s+(\d+)ms\s+steps\((\d+)\)(?:\s+(\d+)ms)?(\s+infinite)?$")
_TRANSLATE = re.compile(r"^translateX\((-?\d+)(?:px)?\)$")


def _changed(a, b):
    """How many cells of one grid change role between two frames."""
    return sum(1 for ra, rb in zip(a, b) for ca, cb in zip(ra, rb) if ca != cb)


def _leans(a, b):
    """Whether ``b`` is ``a`` with every row above one line moved one column
    one way, nothing lost at the edge, and every other row unchanged."""
    if a == b or len(a) != len(b):
        return False
    for split in range(1, len(a)):
        if a[split:] != b[split:]:
            continue
        for moved, edge in ((["." + row[:-1] for row in a[:split]], -1), ([row[1:] + "." for row in a[:split]], 0)):
            if moved == list(b[:split]) and all(row[edge] == "." for row in a[:split]):
                return True
    return False


def _motions(css):
    """``{motion: [play]}`` of the strip's motion classes (``.oo-pixel-<motion>``),
    each play read from its ``animation`` and its keyframes."""
    keyframes = {}
    for selector, body, context in _ds._rules(css):
        if context and context[-1].startswith("@keyframes"):
            value = dict(_ds._declarations(body)).get("transform", "")
            moved = _TRANSLATE.match(value)
            keyframes.setdefault(context[-1].split()[1], {})[selector.strip()] = (
                int(moved.group(1)) if moved else None
            )
    motions = {}
    for selector, body, context in _ds._rules(css):
        name = re.fullmatch(r"\.oo-pixel-(\w+)", selector.strip())
        animation = dict(_ds._declarations(body)).get("animation")
        if context or not name or not animation:
            continue
        plays = []
        for part in animation.split(","):
            read_ = _MOTION.match(part.strip())
            assert read_, f"a motion the census cannot read: {part!r}"
            frames = keyframes.get(read_.group(1), {})
            plays.append({
                "from": frames.get("from"), "to": frames.get("to"), "ms": int(read_.group(2)),
                "steps": int(read_.group(3)), "delay": int(read_.group(4) or 0), "forever": bool(read_.group(5)),
            })
        motions[name.group(1)] = plays
    return motions


def _played(strip, plays, width):
    """The frames a strip shows as its CSS plays it, in order, with how long
    each holds, and whether the last loops back to the first; the plays are
    held to the strip's table first."""
    lead = [p for p in plays if not p["forever"]]
    loop = [p for p in plays if p["forever"]]
    shown = []
    for p in lead:
        assert p["from"] is not None and p["to"] is not None, f"a lead with its keyframes: {p}"
        start, stop = -p["from"] // width, -p["to"] // width
        assert (start, stop - start, p["ms"] // p["steps"], p["delay"]) == (
            strip["loop"], strip["lead"], strip["leadMs"], 0
        ) and p["from"] % width == 0 and p["to"] % width == 0, f"the lead plays the table's lead: {p} {strip}"
        shown += [(start + k, strip["leadMs"]) for k in range(strip["lead"])]
    if strip["lead"]:
        assert lead, f"a strip with a lead plays it: {strip['motion']}"
    if strip["loop"] > 1:
        assert len(loop) == 1, f"a looping strip plays one loop: {strip['motion']} {plays}"
        p = loop[0]
        assert (p["from"], -p["to"] // width, p["steps"], p["ms"] // p["steps"], p["delay"]) == (
            0, strip["loop"], strip["loop"], strip["stepMs"], strip["lead"] * strip["leadMs"]
        ) and p["to"] % width == 0, f"the loop plays the table's loop, after the lead: {p} {strip}"
        shown += [(k, strip["stepMs"]) for k in range(strip["loop"])]
        return shown, True
    assert not loop, f"a strip that rests does not loop: {strip['motion']}"
    return shown + [(0, math.inf)], False


@pytest.mark.parametrize("half", ("grid", "channel", "silhouette", "flicker"))
def test_ld10_the_frame_tables_keep_their_grid_their_silhouette_and_a_calm_pace(half):
    tables = _tables()
    plant, onion = tables["plant"], tables["onion"]
    looks = {(look["state"], look["stage"]): look["strip"] for look in plant["looks"]}
    strips = list(looks.values()) + [onion["strip"]]

    if half == "grid":
        assert (plant["w"], plant["h"], plant["soil"], plant["pitch"]) == (11, 16, 11, 14), (
            f"a plant is 11 x 16, its soil line row 11, its pitch 14: {plant['w'], plant['h'], plant['soil']}"
        )
        assert (onion["w"], onion["h"]) == (11, 12), f"the onion is 11 x 12: {onion['w'], onion['h']}"
        grids = [(name, grid) for name, grid in plant["frames"].items()]
        grids += [(f"{state} {stage}", grid) for (state, stage), strip in looks.items() for grid in strip["frames"]]
        wrong = [name for name, grid in grids if len(grid) != 16 or any(len(row) != 11 for row in grid)]
        wrong += [f"onion {i}" for i, grid in enumerate(onion["frames"] + onion["strip"]["frames"])
                  if len(grid) != 12 or any(len(row) != 11 for row in grid)]
        assert not wrong, f"every frame keeps its table's grid: {wrong}"
        unsoiled = [name for name, grid in grids if "." in grid[11]]
        stray = [name for name, grid in grids if any("-" in row for y, row in enumerate(grid) if y != 11)]
        assert not unsoiled and not stray, (
            f"one baseline: the soil line fills row 11 of every plant frame and no other: {unsoiled} {stray}"
        )
        for strip in strips:
            assert strip["loop"] >= 1 and strip["lead"] >= 0 and len(strip["frames"]) == strip["loop"] + strip["lead"], (
                f"a strip holds its loop and its lead, frame 0 first: {strip['motion']}"
            )
        return

    if half == "channel":
        a, b = plant["frames"]["crook_a"], plant["frames"]["crook_b"]
        assert _leans(a, b) and not _leans(a, a) and not _leans(a, plant["frames"]["flag_a"]), (
            "the probe reads a lean, and no lean in a still or another drawing"
        )
        for stage in ("crook", "flag", "leaf"):
            frames = looks[("running", stage)]["frames"]
            assert looks[("running", stage)]["loop"] == 2 and _leans(frames[0], frames[1]), (
                f"the {stage} sways one part by one pixel, and nothing else moves:\n"
                + "\n".join(f"{x}  {y}" for x, y in zip(frames[0], frames[1]))
            )
        turn = onion["strip"]["frames"]
        for i, frame in enumerate(turn):
            nxt = turn[(i + 1) % len(turn)]
            moved = {(ca, cb) for ra, rb in zip(frame, nxt) for ca, cb in zip(ra, rb) if ca != cb}
            assert moved <= {("o", "x"), ("x", "o")}, f"each step of the turn moves the meridians only: {moved}"
        return

    if half == "silhouette":
        def outline(grid):
            return {(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c != "."}

        frames = onion["frames"] + onion["strip"]["frames"]
        assert len({tuple(f) for f in frames}) >= 4, "the meridians take four positions"
        moved = [list(row) for row in frames[0]]
        moved[5][0] = "."
        assert outline(moved) != outline(frames[0]), "the probe sees a rim moved by one pixel"
        base = outline(frames[0])
        changed = [i for i, f in enumerate(frames) if outline(f) != base]
        assert not changed, f"the onion's silhouette never changes: frames {changed}"
        return

    cells = 11 * 16
    assert _changed(plant["frames"]["crook_a"], plant["frames"]["bulb"]) > _SHARE * cells, (
        "the counter sees a step that changes more than a tenth of the cells"
    )
    motions = _motions(_ds._css_of(_STRIP, read(_STRIP)))
    for strip in strips:
        width = len(strip["frames"][0][0])
        area = width * len(strip["frames"][0])
        if strip["loop"] == 1 and strip["lead"] == 0:
            assert strip["motion"] == "still", f"a strip of one frame is still: {strip['motion']}"
            continue
        assert strip["motion"] in motions, f"the strip plays {strip['motion']} by its class: {sorted(motions)}"
        shown, loops = _played(strip, motions[strip["motion"]], width)
        steps = list(zip(shown, shown[1:])) + ([(shown[-1], shown[len(shown) - strip["loop"]])] if loops else [])
        for (index, hold), (after, _) in steps:
            if hold < _FAST_MS:
                count = _changed(strip["frames"][index], strip["frames"][after])
                assert count <= _SHARE * area, (
                    f"{strip['motion']}: frame {index} holds {hold} ms and its step changes {count} of {area} cells"
                )


# ---------------------------------------------------------------------------
# LD11: no script timer moves the loader; reduced motion draws frame 0
# ---------------------------------------------------------------------------
_LOADER_FILES = (
    _PROGRESS, _WORDS, _STREAM, _PACING, _PLANT, _ONION, _STRIP,
    f"{_LIB}/components/chat/StreamingStatus.svelte", f"{_LIB}/components/chat/StepRow.svelte",
    f"{_LIB}/components/chat/StepList.svelte", f"{_LIB}/components/chat/RunSummary.svelte",
)
_FRAME_TIMER = re.compile(r"\brequestAnimationFrame\b|\bsetInterval\b|<animate")
_RUN = re.compile(r"M(\d+) (\d+)h(\d+)v1h-(\d+)z")
_REDUCED = re.compile(r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)")


def _drawn(html):
    """``{token: {(x, y)}}`` of the cells a rendered strip draws, read from its paths."""
    cells = {}
    for tag in re.findall(r"<path\b[^>]*>", html):
        d = re.search(r'\sd="([^"]*)"', tag)
        fill = re.search(r'\sfill="var\((--oo-[\w-]+)\)"', tag)
        assert d and fill, f"a path the census cannot read: {tag}"
        runs = _RUN.findall(d.group(1))
        assert "".join(f"M{x} {y}h{w}v1h-{back}z" for x, y, w, back in runs) == d.group(1), (
            f"a path drawn in whole runs of pixels: {d.group(1)[:80]}"
        )
        for x, y, w, back in runs:
            assert w == back, f"a run closes on itself: {w} {back}"
            cells.setdefault(fill.group(1), set()).update((int(x) + dx, int(y)) for dx in range(int(w)))
    return cells


def _cells(frames, inks):
    """``{token: {(x, y)}}`` of frames laid side by side, frame 0 first."""
    cells = {}
    for i, grid in enumerate(frames):
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c != ".":
                    cells.setdefault(inks[c], set()).add((i * len(row) + x, y))
    return cells


@pytest.mark.parametrize("half", ("timers", "reduced"))
def test_ld11_no_script_timer_moves_the_loader_and_reduced_motion_draws_frame_zero(half):
    if half == "timers":
        for sample in ("requestAnimationFrame(step)", "const id = setInterval(tick, 360);",
                       '<animate attributeName="x" />', "<animateTransform type=\"rotate\" />"):
            assert _FRAME_TIMER.search(sample), f"the probe reads a frame timer in {sample!r}"
        assert not _FRAME_TIMER.search("clock = setTimeout(second, 1000);"), (
            "the seconds' one self-rescheduling timeout is not a frame timer"
        )
        listed = [path for path in _LOADER_FILES if (REPO / path).is_file()]
        assert {_PLANT, _ONION, _STRIP, _PACING} <= set(listed), f"the drawing's files are read: {listed}"
        found = {path: hits for path in listed for hits in [sorted(set(_FRAME_TIMER.findall(_nav._code(path))))]
                 if hits}
        assert not found, f"frames move by CSS steps, never by a script timer or SMIL: {found}"
        return

    onion = _tables()["onion"]
    strip, inks = onion["strip"], onion["inks"]
    first, every = _cells(strip["frames"][:1], inks), _cells(strip["frames"], inks)
    assert first != every, "the strip has more than its frame 0 to draw"
    server = ssr()
    moving = server.render(_STRIP, {"strip": strip, "inks": inks, "scale": 2, "reduced": False})
    still = server.render(_STRIP, {"strip": strip, "inks": inks, "scale": 2, "reduced": True})
    unread = server.render(_STRIP, {"strip": strip, "inks": inks, "scale": 2})
    assert _drawn(moving.html) == every, "with motion allowed the strip draws its frames side by side"
    assert re.search(r'class="[^"]*\boo-pixel-frames\b[^"]*\boo-pixel-turn\b', moving.html), (
        "and plays them on its frames"
    )
    assert _drawn(still.html) == first, "under reduced motion only frame 0 is drawn"
    assert _drawn(unread.html) == first, "where the motion preference cannot be read, frame 0 alone"
    assert "oo-pixel-turn" not in still.html + unread.html, "and nothing plays"
    rules = _ds._rules(_ds._css_of(_STRIP, read(_STRIP)))
    stops = [(selector, context) for selector, body, context in rules
             if dict(_ds._declarations(body)).get("animation") == "none" and "oo-pixel-frames" in selector]
    assert any("oo-reduce-motion" in selector for selector, _ in stops), (
        f"the choice to reduce motion stops the strip where it plays: {stops}"
    )
    assert any(any(_REDUCED.fullmatch(c) for c in context) and "oo-motion-full" in selector
               for selector, context in stops), f"and so does the system's, unless full motion was chosen: {stops}"


# ---------------------------------------------------------------------------
# LD16: the step row fits its column
# ---------------------------------------------------------------------------
# (steps, current step, column in CSS px, device pixel ratio) -> (scale, first, count, earlier, later)
_FITS = [
    ([14, 0, 600, 1], (3, 0, 14, 0, 0)),
    ([15, 0, 600, 1], (2, 0, 15, 0, 0)),
    ([21, 20, 600, 1], (2, 0, 21, 0, 0)),
    ([22, 0, 600, 1], (2, 0, 19, 0, 3)),
    ([22, 10, 600, 1], (2, 1, 19, 1, 2)),
    ([22, 21, 600, 1], (2, 3, 19, 3, 0)),
    ([8, 3, 361, 1], (3, 0, 8, 0, 0)),
    ([9, 3, 361, 1], (2, 0, 9, 0, 0)),
    ([12, 11, 361, 1], (2, 0, 12, 0, 0)),
    ([13, 6, 361, 1], (2, 2, 10, 2, 1)),
    ([1000, 500, 361, 1], (2, 496, 10, 496, 494)),
    ([30, 15, 80, 1], (2, 15, 1, 15, 14)),
    ([13, 0, 600, 1.5], (2, 0, 13, 0, 0)),
    ([13, 0, 600, 1], (3, 0, 13, 0, 0)),
]


def test_ld16_the_step_row_fits_its_column_never_below_two_x():
    out = _draw("fit", [case for case, _ in _FITS], OO_PLANT=_PLANT, OO_WORDS=_WORDS)
    assert out[0]["capacity"] == [14, 21] and out[6]["capacity"] == [8, 12], (
        f"a 600 px column holds 14 plants at 3x and 21 at 2x, a 361 px one 8 and 12: "
        f"{out[0]['capacity']} {out[6]['capacity']}"
    )
    for (case, want), fit in zip(_FITS, out):
        steps, current = case[0], case[1]
        got = (fit["scale"], fit["first"], fit["count"], fit["earlier"], fit["later"])
        assert got == want, f"{case}: 3x if all fit, else 2x, else a window around the current step: {got}"
        assert fit["scale"] >= 2, f"{case}: never below 2x"
        assert fit["earlier"] + fit["count"] + fit["later"] == steps and fit["first"] <= current < fit["first"] + fit["count"], (
            f"{case}: the window holds the current step and every plant is counted: {got}"
        )
        said = (fit["earlierWords"], fit["laterWords"])
        assert said == (f"{fit['earlier']} earlier" if fit["earlier"] else None,
                        f"{fit['later']} later" if fit["later"] else None), (
            f"{case}: the plants left out are counted in words at both ends: {said}"
        )


# ---------------------------------------------------------------------------
# LD21: the stage pacing
# ---------------------------------------------------------------------------
def _arrive(at, look, end=False):
    return {"at": at, "look": look, "end": end}


_PACE = {
    "minimum": {"start": 0, "end": 2100, "probes": [0, 399, 400, 799, 800, 1200, 1999, 2000, 2400, 9000], "arrivals": [
        _arrive(0, "pending:seed"), _arrive(500, "running:crook"), _arrive(600, "running:flag"),
        _arrive(2000, "running:leaf"), _arrive(2100, "done:bulb", True),
    ]},
    "crowded": {"start": 0, "end": None, "probes": [400, 800, 5000], "arrivals": [
        _arrive(0, "pending:seed"), _arrive(450, "running:crook"), _arrive(460, "running:flag"),
        _arrive(470, "running:leaf"),
    ]},
    "two": {"start": 0, "end": None, "probes": [400, 800, 1200], "arrivals": [
        _arrive(0, "pending:seed"), _arrive(450, "running:crook"), _arrive(460, "running:flag"),
    ]},
    "brief": {"start": 0, "end": 300, "probes": [0, 299, 300, 5000], "arrivals": [
        _arrive(0, "pending:seed"), _arrive(100, "running:crook"), _arrive(300, "done:bulb", True),
    ]},
}


def test_ld21_each_stage_holds_400_ms_a_crowded_queue_jumps_and_a_brief_run_shows_its_end():
    code = _nav._script(_PACING)
    found = sorted({m.group(0) for m in _CLOCK.finditer(code)})
    assert not found, f"the pacing reads no clock: {found}"
    out = _draw("pace", _PACE, OO_PACING=_PACING)
    assert out["read"] == [], f"no clock is read while a schedule is computed: {out['read']}"
    assert (out["min"], out["waiting"]) == (400, 2), f"400 ms a stage, two stages may wait: {out}"
    plans = {name: [(s["at"], s["look"]) for s in result["plan"]] for name, result in out["out"].items()}

    assert plans["minimum"] == [(400, "pending:seed"), (800, "running:crook"), (1200, "running:flag"),
                                (2000, "running:leaf"), (2400, "done:bulb")], (
        f"each stage is shown at least 400 ms, in order: {plans['minimum']}"
    )
    gaps = [b - a for (a, _), (b, _) in zip(plans["minimum"], plans["minimum"][1:])]
    assert min(gaps) >= 400, f"no stage is shown under 400 ms: {gaps}"
    minimum = out["out"]["minimum"]
    assert minimum["shown"] == [None, None, "pending:seed", "pending:seed", "running:crook", "running:flag",
                                "running:flag", "running:leaf", "done:bulb", "done:bulb"], (
        f"read at a time handed in: {minimum['shown']}"
    )
    assert minimum["next"][0] == 400 and minimum["next"][-1] is None, f"and the next change: {minimum['next']}"

    assert plans["two"] == [(400, "pending:seed"), (800, "running:crook"), (1200, "running:flag")], (
        f"two stages waiting are each shown: {plans['two']}"
    )
    assert plans["crowded"] == [(400, "pending:seed"), (800, "running:leaf")], (
        f"with more than two stages waiting the plant jumps to the last: {plans['crowded']}"
    )
    assert plans["brief"] == [(300, "done:bulb")] and out["out"]["brief"]["shown"] == [None, None, "done:bulb", "done:bulb"], (
        f"a run shorter than 400 ms shows only its final state: {plans['brief']}"
    )



# ---------------------------------------------------------------------------
# LD22: every loader colour is an app token, and each reads on the row's ground
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", ("tokens", "marks", "row"))
def test_ld22_every_loader_colour_is_an_app_token_that_reads_on_the_rows_ground(half):
    tables = _tables()
    plant, onion = tables["plant"], tables["onion"]
    resolvers = _resolvers()
    assert sorted(resolvers) == ["day", "high-contrast", "night"], f"three palettes: {sorted(resolvers)}"

    if half == "tokens":
        sample = (
            '<svg><path fill="#4B5E43"/></svg><style>.a { fill: rgb(1, 2, 3); color: var(--oo-role-bg); }\n'
            "@media (forced-colors: active) { .a { fill: CanvasText; } }</style>"
        )
        caught = _literals_in("frontend/src/x.svelte", sample)
        assert "#4B5E43" in caught and "rgb(" in caught and "--oo-role-" in caught, (
            f"the census reads a hex, a colour function and a role: {caught}"
        )
        assert "CanvasText" not in caught, f"and passes the system colours forced colours keep: {caught}"
        listed = files((".ts",), within=_PIXEL) + files((".svelte",), within=f"{_LIB}/components/pixel")
        assert {_PLANT, _ONION, _STRIP} <= set(listed), f"the drawing's files are read: {listed}"
        spelled = {path: found for path in listed for found in [_literals_in(path)] if found}
        assert not spelled, f"the drawing spells no colour and reads no role: {spelled}"

        declared = set(_ds._derivation()) | {n for given in _ds._palette_overrides().values() for n in given}
        assert "--oo-bg-base" in declared and "--oo-role-bg" not in declared, "the census reads the app tokens"
        named = list(plant["inks"].values()) + list(onion["inks"].values())
        named += plant["rowGrounds"] + onion["grounds"]
        undeclared = sorted({n for n in named if n not in declared or n.startswith("--oo-role-")})
        assert not undeclared, f"every ink and ground of the tables is an app token: {undeclared}"
        for label, grids, inks in (
            ("plant", list(plant["frames"].values())
             + [grid for look in plant["looks"] for grid in look["strip"]["frames"]], plant["inks"]),
            ("onion", onion["frames"] + onion["strip"]["frames"], onion["inks"]),
        ):
            drawn = {c for grid in grids for line in grid for c in line} - {"."}
            assert drawn and not drawn - set(inks), (
                f"every character the {label} draws has its ink: {sorted(drawn - set(inks))}"
            )
        return

    def colour(pid, token):
        return _colour_of(resolvers[pid], pid, token)

    if half == "marks":
        marks = {plant["inks"]["g"], plant["inks"]["s"], onion["inks"]["o"]}
        assert marks == {"--oo-acc-mark", "--oo-acc-mark-2"}, (
            f"what grows, the seed, the bulb and the onion are drawn in the brand marks: {sorted(marks)}"
        )
        grounds = set(plant["rowGrounds"]) | {onion["inks"]["x"], _LINE_GROUND}
        grounds |= set(onion["grounds"])
        assert _ratio(colour("day", "--oo-status-ok"), colour("day", _LINE_GROUND)) < 3.0, (
            "the check sees a pair under 3:1 (the finished leaves on tint_1 by day)"
        )
        low = [
            f"{pid}: {mark} on {ground}: {ratio:.2f}"
            for pid in resolvers for mark in sorted(marks) for ground in sorted(grounds)
            for ratio in [_ratio(colour(pid, mark), colour(pid, ground))] if ratio < 3.0
        ]
        assert not low, "a mark under 3:1 on a ground the loader uses:\n  " + "\n  ".join(low)
        return

    inks = {plant["inks"][c] for c in "kwty"}
    assert len(inks) == 4, f"the finished leaves, the seeds, the tag and the kept plant: {sorted(inks)}"
    assert plant["rowGrounds"], "the row's grounds are named"
    assert "sheetGround" not in plant, (
        "no ground is named for a mount behind the failed step's plant: it stands on the row's own ground"
    )
    assert _LINE_GROUND not in plant["rowGrounds"], "the row never stands on tint_1"
    pairs = [(ink, ground) for ink in sorted(inks) for ground in plant["rowGrounds"]]
    low = [
        f"{pid}: {ink} on {ground}: {ratio:.2f}"
        for pid in resolvers for ink, ground in pairs
        for ratio in [_ratio(colour(pid, ink), colour(pid, ground))] if ratio < 3.0
    ]
    assert not low, "a drawing under 3:1 on the row:\n  " + "\n  ".join(low)


# ===========================================================================
# The components: the waiting line, the run's card, the summary line
# ===========================================================================
# StreamingStatus renders the reducer's state (``loader``) at a time handed in
# (``now``), the motion preference given (``reduced``); the run's card draws
# one plant per step (each a PixelStrip, viewBox 11 x 16) and lists the steps
# (``role="list"``); the waiting line draws the onion (viewBox 11 x 12). The
# live seconds carry the class ``oo-live-seconds``. ChatMessage mounts the
# summary line after done, a disclosure (``aria-expanded``).
_CHAT = f"{_LIB}/components/chat"
_STATUS = f"{_CHAT}/StreamingStatus.svelte"
_MESSAGE = f"{_CHAT}/ChatMessage.svelte"

_FIVE = ("Information Gathering", "Structured Analysis", "Draft", "Fact Check", "Final Review")
_FOUR = ("Gather", "Check", "Write", "Review")


def _five(seq, index, state, **fields):
    return _step(seq, index, state, total=5, label=_FIVE[index], **fields)


def _four(seq, index, state, **fields):
    return _step(seq, index, state, total=4, label=_FOUR[index], **fields)


_WAITING = [["start", 0], ["frame", 100, _SENT]]

# Five steps: two done, the third running and reasoning.
_RUNNING = [
    ["start", 0],
    ["frame", 100, _SENT],
    *[["frame", 200 + 10 * i, _five(i + 1, i, "pending")] for i in range(5)],
    ["frame", 1000, _five(6, 0, "running")],
    ["frame", 13000, _five(7, 0, "done", duration_ms=12000)],
    ["frame", 13100, _five(8, 1, "running")],
    ["frame", 31000, _five(9, 1, "done", duration_ms=18000)],
    ["frame", 31100, _five(10, 2, "running")],
    ["frame", 31200, _frame("thinking", "a")],
]

# The same run: the third step grows to the flag and fails, the fourth runs,
# and a later done for the third changes nothing.
_FAILING = _RUNNING + [
    ["frame", 35000, _five(11, 2, "running", progress=_progress(1, 3))],
    ["frame", 42000, _five(12, 2, "failed", reason="timed out", duration_ms=10900)],
    ["frame", 42100, _five(13, 3, "running")],
    ["frame", 42200, _five(14, 2, "done", duration_ms=11000)],
]

# Four steps: one done, one skipped, one stopped at the first leaf, one never run.
_ENDING = [
    ["start", 0],
    ["frame", 100, _SENT],
    *[["frame", 200 + 10 * i, _four(i + 1, i, "pending")] for i in range(4)],
    ["frame", 1000, _four(5, 0, "running")],
    ["frame", 2000, _four(6, 0, "done", duration_ms=1000)],
    ["frame", 2100, _four(7, 1, "skipped")],
    ["frame", 2200, _four(8, 2, "running", progress=_progress(2, 3))],
    ["stop", 3000],
    ["frame", 3100, _four(9, 2, "cancelled")],
    ["frame", 3200, _four(10, 3, "not_run")],
]

# Three steps, the connection lost while the first runs.
_LOSING = [
    ["start", 0],
    ["frame", 100, _SENT],
    *[["frame", 200 + 10 * i, _step(i + 1, i, "pending")] for i in range(3)],
    ["frame", 1000, _step(4, 0, "running")],
    ["lost", 2000],
]

# A step ends and the next starts at the same instant.
_HANDOVER = [
    ["start", 0],
    ["frame", 100, _SENT],
    *[["frame", 200 + 10 * i, _step(i + 1, i, "pending")] for i in range(3)],
    ["frame", 1000, _step(4, 0, "running")],
    ["frame", 5000, _step(5, 0, "done", duration_ms=4000)],
    ["frame", 5000, _step(6, 1, "running")],
]

_SVG = re.compile(r"<svg\b([^>]*)>(.*?)</svg>", re.S)
_ANIMATED = re.compile(r'class="[^"]*\boo-pixel-(?:turn|sway|root|bloom)\b')
_PLANT_BOX = "0 0 11 16"
_ONION_BOX = "0 0 11 12"


def _loaders(**streams):
    return _node("states", streams)


def _status(state, now, reduced=False):
    return ssr().render(_STATUS, {"loader": state, "now": now, "reduced": reduced}).html


def _strips(html):
    """Every pixel strip a render draws, in order: its viewBox and its cells by token."""
    found = []
    for attrs, body in _SVG.findall(html):
        box = re.search(r'\sviewBox="([^"]*)"', attrs)
        assert box, f"a strip without its viewBox: {attrs}"
        found.append((box.group(1), _drawn(body)))
    return found


def _plants(html):
    return [cells for box, cells in _strips(html) if box == _PLANT_BOX]


def _in_view(cells, width=11):
    """The cells of a strip's window: its frame 0, what shows while it holds still."""
    kept = {token: {(x, y) for x, y in found if x < width} for token, found in cells.items()}
    return {token: found for token, found in kept.items() if found}


def _onions(html):
    return [cells for box, cells in _strips(html) if box == _ONION_BOX]


def _classes(element):
    return (element.get("class") or "").split()


def _lines(html):
    """The text of each line of the step list, spaces folded."""
    root = _mk._dom(html)
    lists = [e for e in root.iter() if e.get("role") == "list"]
    assert len(lists) == 1, f"the card lists its steps once: {len(lists)}"
    return [" ".join(li.text().split()) for li in lists[0].iter("li")]


def _look(tables, name):
    return _cells([tables["plant"]["frames"][name]], tables["plant"]["inks"])


# ---------------------------------------------------------------------------
# LD3: every end has its own form and its own words
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", ("failed", "ends", "words"))
def test_ld3_every_end_has_its_own_form_and_its_own_words(half):
    tables = _tables()
    plant = tables["plant"]
    frames, inks = plant["frames"], plant["inks"]

    if half == "failed":
        state = _loaders(failing=_FAILING)["failing"]
        draft = state["runs"][0]["steps"][2]
        assert (draft["state"], draft["stage"]) == ("failed", "flag"), (
            f"a failed step stays failed against a later done, at the stage it reached: {draft['state']}, {draft['stage']}"
        )
        kept = _look(tables, "pressed_flag")
        straw, tin = inks["y"], inks["t"]
        stood = {(x, y) for y, row in enumerate(frames["flag_a"]) for x, c in enumerate(row)
                 if y <= plant["soil"] and c in "gksrw"}
        assert stood and kept.get(straw) == stood, (
            "the failed step's plant is its stage's silhouette, whole, never an outline, in straw"
        )
        assert kept.get(tin) and set(kept) - {straw, tin} <= {inks["r"], inks["s"], inks["-"]}, (
            f"with the tin tag, and no mount behind it and no strap across it: {sorted(kept)}"
        )
        for reduced in (False, True):
            plants = _plants(_status(state, 60000, reduced))
            assert len(plants) == 5, f"one plant per step: {len(plants)}"
            assert plants[2] == kept, (
                f"reduced motion {reduced}: the failed step's plant is that one still drawing and nothing else"
            )
        cut = _plants(_status(state, 42000))
        assert cut[2] == kept, "the change to it is a cut: no frame stands between the running stage and it"
        return

    if half == "ends":
        states = _loaders(ending=_ENDING, losing=_LOSING)
        plants = [_in_view(cells) for cells in _plants(_status(states["ending"], 60000))]
        want = [_look(tables, name) for name in ("bulb", "bare", "stopped_leaf", "seed")]
        assert plants == want, (
            "done is the bulb, skipped bare soil, a stopped step its stage all in text_muted, "
            f"a step never run a grey seed: {[sorted(p) for p in plants]}"
        )
        assert set(plants[2]) == {inks["m"], inks["-"]} and inks["t"] not in plants[2], (
            "the stopped plant carries no tag"
        )
        assert plants[3].get(inks["w"]), "the seed never run is grey"
        lost = [_in_view(cells) for cells in _plants(_status(states["losing"], 60000))]
        want = [_look(tables, name) for name in ("unknown_crook", "unknown_seed", "unknown_seed")]
        assert lost == want, "a step whose outcome is unknown is its last drawing, dotted"
        return

    states = _loaders(ending=_ENDING, failing=_FAILING)
    lines = _lines(_status(states["ending"], 60000))
    assert len(lines) == 4, f"one line per step: {lines}"
    assert "Skipped: its condition was not met" in lines[1], f"skipped says so: {lines[1]!r}"
    assert re.search(r"\bStopped\b", lines[2]) and "Failed" not in lines[2], f"stopped says Stopped: {lines[2]!r}"
    assert "Not run" in lines[3] and not re.search(r"stop", lines[3], re.I), f"never run says so: {lines[3]!r}"
    failed = _lines(_status(states["failing"], 60000))
    assert "Failed" in failed[2] and "timed out" in failed[2], f"failed says so, with the server's reason: {failed[2]!r}"
    assert not re.search(r"stop", failed[2], re.I), f"a failure is never said stopped: {failed[2]!r}"


# ---------------------------------------------------------------------------
# LD4: one thing moves at a time
# ---------------------------------------------------------------------------
def test_ld4_one_thing_moves_at_a_time():
    assert len(_ANIMATED.findall('<g class="oo-pixel-frames oo-pixel-sway"></g><g class="oo-pixel-frames">')) == 1, (
        "the probe counts a strip that plays, and not one that holds still"
    )
    tables = _tables()
    states = _loaders(waiting=_WAITING, running=_RUNNING, failing=_FAILING, handover=_HANDOVER, ending=_ENDING)

    waiting = _status(states["waiting"], 7100)
    assert len(_onions(waiting)) == 1 and len(_ANIMATED.findall(waiting)) == 1, (
        "before any run, the waiting line's onion turns, and it alone"
    )
    counts = {}
    for name, times in (("running", (1000, 13000, 13100, 31000, 42100, 70000)),
                        ("failing", (42000, 42100, 42500, 60000)),
                        ("handover", (5000, 5100, 5239, 5240, 5300, 9000)),
                        ("ending", (3000, 3200, 9000))):
        for now in times:
            for reduced in (False, True):
                html = _status(states[name], now, reduced)
                assert not _onions(html), f"{name} at {now}: with a run open the onion is not mounted"
                counts[(name, now, reduced)] = len(_ANIMATED.findall(html))
    crowded = {key: n for key, n in counts.items() if n > 1}
    assert not crowded, f"at most one thing moves, for every state and time: {crowded}"
    assert counts[("running", 42100, False)] == 1, "the running plant sways"

    flowering = _status(states["handover"], 5100)
    assert "oo-pixel-bloom" in flowering and _plants(flowering)[1] == _look(tables, "seed"), (
        "while a finished plant flowers, the next step's plant waits as a seed"
    )
    after = _status(states["handover"], 5300)
    assert "oo-pixel-bloom" not in after and re.search(r"oo-pixel-(?:root|sway)", after), (
        "once the flowering is over, the next plant takes its root"
    )


# ---------------------------------------------------------------------------
# LD6: live seconds from five seconds, never in the status region
# ---------------------------------------------------------------------------
def _seconds(html):
    """``[(text, hidden)]`` of the live seconds a render shows."""
    root = _mk._dom(html)
    found = []
    for element in root.iter():
        if "oo-live-seconds" in _classes(element):
            hidden, at = False, element
            while at is not None:
                hidden = hidden or at.get("aria-hidden") == "true"
                at = at.parent
            found.append((element.text(), hidden))
    return found


def _region(html):
    root = _mk._dom(html)
    regions = [e for e in root.iter() if e.get("role") == "status"]
    assert len(regions) == 1, f"one status region: {len(regions)}"
    return regions[0]


def test_ld6_live_seconds_show_from_five_seconds_and_never_reach_the_status_region():
    states = _loaders(waiting=_WAITING, running=_RUNNING)
    assert _seconds('<p><span class="oo-live-seconds" aria-hidden="true">, 5 s</span></p>') == [(", 5 s", True)], (
        "the probe reads the live seconds and their hiding"
    )
    before = _status(states["waiting"], 5099)
    at = _status(states["waiting"], 5100)
    assert _seconds(before) == [], f"under 5 s on the line, no seconds: {_seconds(before)}"
    assert _seconds(at) == [(", 5 s", True)], f"from 5 s they show, hidden from a screen reader: {_seconds(at)}"
    region = _region(at)
    assert not re.search(r"\d+\s*s\b", region.text()) and not any(
        "oo-live-seconds" in _classes(e) for e in region.iter()
    ), f"the status region never holds them: {region.text()!r}"

    def running_line(now):
        root = _mk._dom(_status(states["running"], now))
        lists = [e for e in root.iter() if e.get("role") == "list"]
        line = list(lists[0].iter("li"))[2]
        return _seconds("".join(_rebuilt(line)))

    assert running_line(36099) == [], "the running step shows no seconds before its fifth"
    assert running_line(36100) == [(", 5 s", True)], f"and shows them from 5 s: {running_line(36100)}"

    script = _nav._script(_STATUS)
    assert len(re.findall(r"\bsetTimeout\s*\(", script)) == 1 and re.search(r"\bclearTimeout\s*\(", script), (
        "the seconds come from one timeout, stopped when it is no longer needed"
    )
    markup = _nav._markup(_STATUS)
    spoken = re.findall(r"""role\s*=\s*["']status["'][^>]*>\s*\{\s*(\w+)\s*\}\s*<""", markup)
    assert len(spoken) == 1, f"the region shows one value: {spoken}"
    assigned = re.findall(rf"\b{spoken[0]}\s*=(?!=)\s*([^;\n]+)", script)
    assert assigned and all(rhs.strip() in ("''", '""') or rhs.strip().endswith(".text") for rhs in assigned) and (
        re.search(r"\bdue\s*\(", script)
    ), f"and that value is only what the announcer lets through: {assigned}"


def _rebuilt(element):
    """An element of the parsed tree written back as HTML, for the probes that read text."""
    attrs = "".join(f' {k}="{v}"' for k, v in element.attrs if v is not None)
    yield f"<{element.tag}{attrs}>"
    for child in element.children:
        if isinstance(child, str):
            yield child
        else:
            yield from _rebuilt(child)
    yield f"</{element.tag}>"


# ---------------------------------------------------------------------------
# LD12: after done, one folded footer line, counted from the steps' record
# ---------------------------------------------------------------------------
def _record(seq, index, state, **fields):
    return _step(seq, index, state, **fields)["metadata"]


_WITH_FAILURE = [_record(5, 0, "done", duration_ms=10), _record(7, 1, "failed", reason="HTTP 500 from the model server",
                                                                 duration_ms=10), _record(9, 2, "done", duration_ms=10)]
_ALL_DONE = [_record(5, 0, "done", duration_ms=10), _record(7, 1, "done", duration_ms=10),
             _record(9, 2, "done", duration_ms=10)]
_STOPPED = [_record(5, 0, "done", duration_ms=10), _record(7, 1, "cancelled"), _record(8, 2, "not_run")]


def _summaries(message, **props):
    """``[(text, expanded, controls, controlled)]`` of the lines a reply shows as disclosures."""
    html = ssr().render(_MESSAGE, {"message": {"role": "assistant", "id": 7, "model": "llama3", **message}, **props}).html
    root = _mk._dom(html)
    ids = {e.get("id") for e in root.iter() if e.get("id")}
    return [(" ".join(e.text().split()), e.get("aria-expanded"), e.get("aria-controls"), e.get("aria-controls") in ids)
            for e in root.iter() if e.get("aria-expanded") is not None]


def test_ld12_after_done_one_folded_footer_line_counts_the_steps_record():
    failure = "[ERR] " + "Step" + " 3 failed: boom"
    lines = _summaries({"content": "The reply.", "steps": _WITH_FAILURE, "duration_ms": 58000})
    assert len(lines) == 1, f"after done, one line: {lines}"
    text, expanded, controls, controlled = lines[0]
    assert text == "Research Assistant, 3 steps, 1 failed, 58 s", f"its counts are the steps' record: {text!r}"
    assert expanded == "false" and controls and controlled, (
        f"folded, a disclosure of the region it opens: {lines[0]}"
    )
    lines = _summaries({"content": "The reply.\n" + failure, "steps": _ALL_DONE, "duration_ms": 30000})
    assert [line[0] for line in lines] == ["Research Assistant, 3 steps, 30 s"], (
        f"the reply's text is never read for a count: {lines}"
    )
    stopped = _summaries({"content": "The reply.", "steps": _STOPPED, "duration_ms": 14000, "stopped": True})
    assert [line[0] for line in stopped] == ["Research Assistant, stopped at step 2 of 3, 14 s"], (
        f"after a Stop, where it stopped: {stopped}"
    )
    assert _summaries({"content": "The reply.\n" + failure, "duration_ms": 58000}) == [], (
        "without the steps' record there is no line"
    )
    streaming = _summaries({"content": "", "steps": _WITH_FAILURE, "duration_ms": 58000},
                           isStreaming=True, streamContent="The rep")
    assert streaming == [], f"and none while the reply streams: {streaming}"
