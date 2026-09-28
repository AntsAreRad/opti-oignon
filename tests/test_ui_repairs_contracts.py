#!/usr/bin/env python3
"""Contracts for the interface repairs that outlive its redesign.

Each repair here fixes a defect of today's interface that no later rebuilding
removes by itself: an event nobody hears, request options the server drops,
server switches that show what they asked for rather than what the server
holds, a destructive action without a question, the browser's native
dialogs, pictographs, undeclared tokens, doubled landmarks, a dead shortcut,
a link to a vendor's organisation.

The pure modules the repairs rest on run under Node's type stripping through
``tests/_frontend.run_ts``; the driver calls them and prints what they
return, and every property is asserted here in Python:

  * ``frontend/src/lib/chat/requestFields.ts`` -- the fields a chat request
    carries, each a field of the server's ``ChatRequest``, and the two
    builders: the options from the composer's selections, the request from
    a conversation, a message and those options;
  * ``frontend/src/lib/switches/serverSwitch.ts`` -- a switch whose state is
    the one the server confirmed, never the one it was asked for;
  * ``frontend/src/lib/motion.ts`` -- the scroll behaviour the motion
    preference allows;
  * ``frontend/src/lib/settings/catalog.ts`` -- every settings section and
    group, and the map of the settings page's old tab ids.

``frontend/src/lib/ds/ConfirmDialog.svelte`` asks before an action that
cannot be undone, on the design system's Modal (a native modal dialog,
whose focus is restored to its opener on close).

A contract over a pure module carries a wiring half: the files that use the
module import it, call it, and hold no competing logic. The static halves
read sources as text. The server-rendering halves render the dialog and the
switch components through ``tests/_frontend.ssr()``: what the template emits
when compiled for the server, not what a browser does with it.

  * UX1 -- every ``var(--oo-*)`` read in ``frontend/src`` and the Tailwind
    configuration names a custom property declared somewhere in them.
  * UX2 -- every event a component dispatches is heard at every mount of
    that component: each mount binds ``on:<event>`` to a handler, or
    forwards it (``on:<event>`` with no value) from a component every one of
    whose mounts hears it. Mounts are matched per component, through the
    import that names it, never by the event's name across the tree; a
    forward out of a route, which nothing mounts, reaches nobody. A mount
    that cannot raise the event (a reply still being written, a read-only
    view) is named in an exemption with its reason and its count, which may
    only fall: an exemption above the deaf mounts it covers is stale. The
    design system's primitives offer their events to every consumer and are
    not judged.
  * UX3 -- every mount of ``ChatMessage`` passes ``conversationId``; the
    feedback widget returns before it submits when either of its ids is
    empty, and without both it shows no thumb and a thumb chooses nothing.
  * UX4 -- a chat request carries only fields of the server's
    ``ChatRequest``: the field list of ``requestFields.ts`` is a subset of
    the schema's (read from ``opti_oignon/api/schemas.py`` through ``ast``),
    the options builder builds only those fields, and the request builder
    drops every other key; the chat stores build through those two builders
    and write no request key of their own, by name, by index, by a merge or
    by a spread; the frontend's ``ChatRequest`` type names exactly the fields
    the builder fills. What a message sends when nothing is chosen is held
    apart, in its own half, so the composer's later defaults can supersede
    it alone.
  * UX5 -- a server switch shows the server's confirmed state: a success
    adopts the state the server answers, and names it when it is not the one
    asked for; a refusal (4xx) keeps the state and names the refusal with
    the server's own reason; a server error (5xx), which some routes answer
    after the change has landed, and a failure to reach the server re-read
    the state; a second toggle while one is pending is refused; the state
    never flips before the server has answered; while it is unknown, its
    mirrors forget it and ``pressed()`` announces no state. Every component
    that toggles one of the five server switches does so through
    ``createServerSwitch``: its write returns the state the server answers,
    never the one asked for; it writes a mirror store only where the switch
    adopts or forgets a state, and the state through no other call; it
    calls the API layer rather than spelling an endpoint; it never flips a
    state itself; every control that toggles a switch announces
    ``pressed($switch.value)`` and draws from the switch's value; the value
    is never read for its truthiness, so an unknown state is never drawn as
    off; and it shows each switch's error. Rendered for the server before
    any read, every switch control announces no state and cannot be
    pressed, and the settings panels say the state is being read.
  * UX6 -- no raw ``fetch(`` under ``lib/components/chat/``.
  * UX7 -- a conversation wipe is called only from the ``onConfirm`` of a
    ``ConfirmDialog`` in the same file, never from the markup; that handler
    is reached from the markup only as the dialog's ``onConfirm`` and from
    no other function, and reads no store when it runs (its target is the
    one kept when the question was asked); its failure is assigned to an
    error the file renders as text or hands to a ``ConfirmDialog`` or an
    ``InlineError``. Every other destructive action the frontend offers
    (deleting a document or a collection, unregistering a variant, disabling
    remote access, revoking a certificate) is held the same way, its failure
    rendered or toasted. The dialog is built on the ds Modal, shows the
    error in an alert, holds its buttons and the Modal's close button while
    the action runs, and confirms on its form's submit.
  * UX8 -- no ``confirm(`` or ``prompt(`` dialog of the browser.
  * UX9 -- no pictographic emoji (U+1F000-1FAFF, and the variation selector
    U+FE0F), raw or spelled as a reference or an escape, or built from
    number literals by ``String.fromCodePoint`` (every argument) or
    ``String.fromCharCode`` (a surrogate pair joined). Symbol glyphs are the
    symbol-glyph ratchet's.
  * UX10 -- one skip link in the whole frontend, and every page route
    renders exactly one ``id="main-content"``, counted over the page, its
    layouts and every component they mount; a page that only forwards
    elsewhere on mount renders nothing and is not judged.
  * UX11 -- no literal smooth scrolling outside ``lib/motion.ts``; every
    scroll option takes its behaviour from ``scrollBehavior()``, which
    answers ``auto`` whenever the motion preference or the system reduces
    motion, and when it cannot tell.
  * UX12 -- no navigation by assignment to ``location`` (the reload is the
    reload ratchet's).
  * UX13 -- every rendered settings group, the lazy panels and the groups
    the section introductions render inline, is in ``catalog.ts``, under the
    section that renders it; and no other file declares a list of the
    settings sections or the map of the old tab ids.
  * UX15 -- no URL in ``frontend/src`` names a vendor's organisation.
  * UX16 -- every DOM selector a global shortcut handler queries matches an
    element that a mounted component or a route renders.
  * UX17 -- the settings hub (``SettingsHub.svelte``), which renders the
    catalog's groups on the page that holds each, loads exactly the
    catalog's panels, and its section introductions render exactly the
    catalog's inline groups, each under the section whose introduction
    renders it; no other file declares a list of the settings sections or
    the map of the old tab ids, and no other file loads the panels.

The censuses each carry a positive fixture, a sample they must count, so a
probe that goes blind turns red instead of reading a false zero. Words the
censuses look for are assembled from fragments, so this file does not carry
them.

Local-only (the public distribution ships no tests). The node halves need
Node >= 22.6 and the server-rendering half ``frontend/node_modules``; without
them the helpers raise, and so do the contracts.
"""

import ast
import html
import json
import posixpath
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _frontend import REPO, files, read, run_ts, ssr  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file. The server-rendering half may start the session's
# server when it runs first.
BUDGET_S = {
    "test_ux1_every_custom_property_read_is_declared": 1.0,
    "test_ux2_every_dispatched_event_is_heard_by_a_mount_of_its_component": 1.0,
    "test_ux3_every_message_knows_its_conversation_and_feedback_never_sends_an_empty_id": 1.0,
    "test_ux4_a_chat_request_carries_only_chat_request_fields[node]": 2.0,
    "test_ux4_a_chat_request_carries_only_chat_request_fields[wiring]": 1.0,
    "test_ux4_a_chat_request_carries_only_chat_request_fields[unchosen]": 2.0,
    "test_ux5_a_server_switch_shows_the_servers_confirmed_state[node]": 2.0,
    "test_ux5_a_server_switch_shows_the_servers_confirmed_state[wiring]": 1.0,
    "test_ux5_a_server_switch_shows_the_servers_confirmed_state[ssr]": 6.0,
    "test_ux6_no_raw_fetch_under_the_chat_components": 1.0,
    "test_ux7_a_wipe_runs_only_once_confirmed_and_its_failure_is_shown[wipe]": 1.0,
    "test_ux7_a_wipe_runs_only_once_confirmed_and_its_failure_is_shown[dialog]": 1.0,
    "test_ux7_a_wipe_runs_only_once_confirmed_and_its_failure_is_shown[ssr]": 6.0,
    "test_ux7_a_wipe_runs_only_once_confirmed_and_its_failure_is_shown[destructive]": 1.0,
    "test_ux8_no_native_confirm_or_prompt_dialog": 1.0,
    "test_ux9_no_pictographic_emoji_in_any_spelling": 1.0,
    "test_ux10_one_skip_link_and_one_main_landmark_per_page": 1.0,
    "test_ux11_scrolling_is_smooth_only_where_motion_is_allowed[node]": 2.0,
    "test_ux11_scrolling_is_smooth_only_where_motion_is_allowed[wiring]": 1.0,
    "test_ux12_no_navigation_by_assignment_to_location": 1.0,
    "test_ux13_every_settings_group_is_in_the_catalog_and_nowhere_else[node]": 2.0,
    "test_ux13_every_settings_group_is_in_the_catalog_and_nowhere_else[wiring]": 1.0,
    "test_ux15_no_url_names_a_vendor_organisation": 1.0,
    "test_ux16_every_selector_a_global_shortcut_queries_is_rendered": 1.0,
    "test_ux17_every_settings_group_is_in_the_catalog_and_the_hub_renders_it[node]": 2.0,
    "test_ux17_every_settings_group_is_in_the_catalog_and_the_hub_renders_it[wiring]": 1.0,
}

_SRC = "frontend/src"
_REQUEST_FIELDS = f"{_SRC}/lib/chat/requestFields.ts"
_SERVER_SWITCH = f"{_SRC}/lib/switches/serverSwitch.ts"
_MOTION = f"{_SRC}/lib/motion.ts"
_CATALOG = f"{_SRC}/lib/settings/catalog.ts"
_CONFIRM_DIALOG = f"{_SRC}/lib/ds/ConfirmDialog.svelte"
_DS_INDEX = f"{_SRC}/lib/ds/index.ts"

_CHAT_OPTIONS = f"{_SRC}/lib/stores/chatOptions.ts"
_CHAT_STORE = f"{_SRC}/lib/stores/chat.ts"
_PREFERENCES = f"{_SRC}/lib/stores/preferences.ts"
_SETTINGS_PAGE = f"{_SRC}/routes/settings/+page.svelte"
_SECTION_LIST = f"{_SRC}/lib/components/sidebar/SectionContextList.svelte"
_ROOT_LAYOUT = f"{_SRC}/routes/+layout.svelte"
_SHORTCUTS = f"{_SRC}/lib/components/ui/KeyboardShortcuts.svelte"
_FEEDBACK = f"{_SRC}/lib/components/chat/FeedbackWidget.svelte"
_CHAT_MESSAGE = f"{_SRC}/lib/components/chat/ChatMessage.svelte"
_SCHEMAS = "opti_oignon/api/schemas.py"

_MODULES = {
    "OO_REQUEST_FIELDS": _REQUEST_FIELDS,
    "OO_SERVER_SWITCH": _SERVER_SWITCH,
    "OO_MOTION": _MOTION,
    "OO_CATALOG": _CATALOG,
}

# Every kind of source file the censuses read.
_EVERY = (".svelte", ".ts", ".js", ".mjs", ".cjs", ".css", ".scss", ".html")
_SCRIPTS = (".svelte", ".ts", ".js", ".mjs", ".cjs")

_BACKSLASH = chr(92)


# ---------------------------------------------------------------------------
# The Node driver: it calls the pure modules and prints what they return.
# ---------------------------------------------------------------------------
_DRIVER = r"""
const load = async (name) => (process.env[name] ? await import(process.env[name]) : null);
const fields = await load('OO_REQUEST_FIELDS');
const switches = await load('OO_SERVER_SWITCH');
const motion = await load('OO_MOTION');
const catalog = await load('OO_CATALOG');
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT || 'null');

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

// The API client's error shape: its own sentence in detail, and the server's
// reason for an HTTP error apart, in serverDetail.
function apiError(status, detail, network) {
    const error = new Error(network ? 'Connection failed' : 'API error ' + status);
    error.status = status;
    error.detail = network ? detail : 'Request failed (' + status + ') for cache/semcache.';
    error.serverDetail = network ? '' : detail;
    error.isNetworkError = network;
    return error;
}

// A switch over scripted server answers: each read and each write takes the
// next answer of its queue; an answer is a state, an HTTP refusal, a
// failure to reach the server, a plain thrown error, or a hold (a promise
// the scenario settles itself).
function scripted(reads, writes) {
    const log = { reads: 0, writes: [], adopted: [], held: [], forgot: 0 };
    const answer = (queue) => {
        const next = queue.length ? queue.shift() : { value: false };
        if ('value' in next) return Promise.resolve(next.value);
        if ('http' in next) return Promise.reject(apiError(next.http[0], next.http[1], false));
        if ('network' in next) return Promise.reject(apiError(0, next.network, true));
        if ('thrown' in next) return Promise.reject(new TypeError(next.thrown));
        if ('hold' in next) return new Promise((resolve, reject) => log.held.push({ resolve, reject }));
        throw new Error('unknown scripted answer: ' + JSON.stringify(next));
    };
    const control = switches.createServerSwitch({
        label: 'Semantic cache',
        read: () => { log.reads += 1; return answer(reads); },
        write: (next) => { log.writes.push(next); return answer(writes); },
        adopt: (value) => { log.adopted.push(value); },
        forget: () => { log.forgot += 1; },
    });
    return { control, log };
}

const snapshot = (state) => ({ value: state.value, pending: state.pending, error: state.error });

const scenarios = {
    // A success adopts what the server answers, even when it is not what
    // was asked; subscribers see every state, and stop once unsubscribed.
    adopts: async () => {
        const { control, log } = scripted([{ value: false }], [{ value: true }, { value: true }]);
        const seen = [];
        const stop = control.subscribe((state) => seen.push(snapshot(state)));
        const loaded = snapshot(await control.load());
        const first = await control.toggle();
        const afterFirst = snapshot(control.current());
        const second = await control.toggle();
        const afterSecond = snapshot(control.current());
        const seenBeforeStop = seen.length;
        stop();
        await control.toggle();
        return { loaded, first, afterFirst, second, afterSecond, seen: seen.slice(0, seenBeforeStop),
            seenAfterStop: seen.length - seenBeforeStop, log };
    },
    // A refusal (4xx) keeps the state and names the refusal; no re-read.
    refused: async () => {
        const out = {};
        for (const [name, refusal] of [['forbidden', [403, 'CSRF token missing']], ['bare', [409, '']]]) {
            const { control, log } = scripted([{ value: true }], [{ http: refusal }]);
            await control.load();
            const outcome = await control.toggle();
            out[name] = { outcome, state: snapshot(control.current()), log };
        }
        return out;
    },
    // A failure to reach the server re-reads the state: the write may or
    // may not have landed. A thrown error that is not an HTTP refusal, and
    // an answer that carries no state, are unknowns too.
    unreached: async () => {
        const out = {};
        const cases = {
            landed: [[{ value: false }, { value: true }], [{ network: 'Unable to reach the backend' }]],
            lost: [[{ value: false }, { network: 'still down' }], [{ network: 'Unable to reach the backend' }]],
            thrown: [[{ value: false }, { value: false }], [{ thrown: 'Failed to fetch' }]],
            stateless: [[{ value: false }, { value: true }], [{ value: undefined }]],
        };
        for (const [name, [reads, writes]] of Object.entries(cases)) {
            const { control, log } = scripted(reads, writes);
            await control.load();
            const outcome = await control.toggle();
            out[name] = { outcome, state: snapshot(control.current()), log };
        }
        return out;
    },
    // A second toggle while one is pending is refused and sends nothing;
    // the state does not flip before the server answers.
    pending: async () => {
        const { control, log } = scripted([{ value: false }], [{ hold: true }, { value: false }]);
        await control.load();
        const first = control.toggle();
        await settle();
        const during = snapshot(control.current());
        const second = await control.toggle();
        const loadDuring = snapshot(await control.load());
        const writesDuring = log.writes.length;
        const readsDuring = log.reads;
        log.held[0].resolve(true);
        const firstOutcome = await first;
        const after = snapshot(control.current());
        const third = await control.toggle();
        return { during, second, loadDuring, writesDuring, readsDuring, firstOutcome, after, third, log };
    },
    // A server error (5xx) is not a refusal: the change may have landed
    // before the answer failed, so the state is read again.
    failed: async () => {
        const out = {};
        const cases = {
            landed: [[{ value: false }, { value: true }], [{ http: [500, 'boom'] }]],
            before: [[{ value: false }, { value: false }], [{ http: [503, 'Semantic cache module not available'] }]],
        };
        for (const [name, [reads, writes]] of Object.entries(cases)) {
            const { control, log } = scripted(reads, writes);
            await control.load();
            const outcome = await control.toggle();
            out[name] = { outcome, state: snapshot(control.current()), log };
        }
        return out;
    },
    // A state never read cannot be toggled; a failed first read is named.
    unknown: async () => {
        const { control, log } = scripted([{ network: 'down' }], []);
        const before = await control.toggle();
        const loaded = snapshot(await control.load());
        const after = await control.toggle();
        return { before, loaded, after, log };
    },
};

const clauses = {
    fields: () => [...fields.REQUEST_FIELDS],
    options: () => input.map((selections) => fields.chatOptions(selections)),
    request: () => input.map(([conversation, message, options]) =>
        fields.chatRequest(conversation, message, options)),
    switch: () => scenarios[input](),
    pressed: () => ({
        on: switches.pressed(true),
        off: switches.pressed(false),
        unknownAbsent: switches.pressed(null) === undefined,
    }),
    motion: () => ({
        behaviours: input.map((environment) => motion.scrollBehavior(environment)),
        withoutDocument: motion.scrollBehavior(),
        read: motion.readMotionEnvironment(),
        classes: [motion.MOTION_REDUCED_CLASS, motion.MOTION_FULL_CLASS],
    }),
    catalog: () => ({
        sections: catalog.SETTINGS_SECTIONS,
        inline: catalog.INLINE_GROUPS,
        legacy: catalog.LEGACY_TAB_TO_SECTION,
        resolved: input.map((raw) => catalog.resolveSection(raw)),
    }),
};

if (!(clause in clauses)) {
    console.log('FAIL unknown clause: ' + clause);
    process.exit(1);
}
console.log('RESULT ' + JSON.stringify(await clauses[clause]()));
console.log('PASS ' + clause);
"""


def _node(clause, modules, data=None):
    """Runs one driver clause over the named modules and returns its result."""
    out = run_ts(
        {var: _MODULES[var] for var in modules}, _DRIVER, clause,
        env={"OO_INPUT": json.dumps(data)},
    )
    results = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, f"the driver printed no single RESULT line:\n{out}"
    return json.loads(results[0][len("RESULT "):])


# ---------------------------------------------------------------------------
# Reading components: markup, scripts, imports and mounts
# ---------------------------------------------------------------------------
_SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>(.*?)</script>", re.S)
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>.*?</style>", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _blank(match):
    """The matched text with every character but newlines made a space, so
    offsets and line numbers still hold."""
    return re.sub(r"[^\n]", " ", match.group(0))


def _markup(text):
    """A component's markup: its scripts, styles and comments blanked."""
    for pattern in (_SCRIPT_BLOCK, _STYLE_BLOCK, _HTML_COMMENT):
        text = pattern.sub(_blank, text)
    return text


def _script(text, path=""):
    """A component's script (a module is all script)."""
    if not path.endswith(".svelte"):
        return text
    return "\n".join(match.group(1) for match in _SCRIPT_BLOCK.finditer(text))


def _skip_string(text, i):
    """The offset past the script string that opens at ``i``."""
    quote, i, end = text[i], i + 1, len(text)
    while i < end:
        char = text[i]
        if char == _BACKSLASH:
            i += 2
            continue
        if quote == "`" and text.startswith("${", i):
            i = _skip_expression(text, i + 1)
            continue
        if char == quote:
            return i + 1
        i += 1
    return end


def _skip_expression(text, i):
    """The offset past the ``}`` that closes the ``{`` at ``i``; strings
    inside are skipped, so a brace in a string does not count."""
    depth, end = 0, len(text)
    while i < end:
        char = text[i]
        if char in "'\"`":
            i = _skip_string(text, i)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return end


def _skip_value(text, i):
    """The offset past the quoted attribute value that opens at ``i``;
    expressions inside it are skipped."""
    quote, i, end = text[i], i + 1, len(text)
    while i < end:
        if text[i] == "{":
            i = _skip_expression(text, i)
            continue
        if text[i] == quote:
            return i + 1
        i += 1
    return end


def _tag_end(text, start):
    """The offset just past the ``>`` that closes the tag opened at
    ``start``: quoted values and ``{...}`` expressions are skipped, so a
    ``>`` in an arrow function or a string does not end the tag."""
    i, end = start + 1, len(text)
    while i < end:
        char = text[i]
        if char == "{":
            i = _skip_expression(text, i)
            continue
        if char in "'\"":
            i = _skip_value(text, i)
            continue
        if char == ">":
            return i + 1
        i += 1
    return end


def _tags(markup, pattern):
    """``(name, attributes, line)`` of every tag whose name matches
    ``pattern`` in a component's markup."""
    for match in re.finditer(r"<(" + pattern + r")(?=[\s/>])", markup):
        stop = _tag_end(markup, match.start())
        body = markup[match.end():stop - 1] if markup[stop - 1:stop] == ">" else markup[match.end():stop]
        yield match.group(1), body, markup.count("\n", 0, match.start()) + 1


_NAME = re.compile(r"""[^\s=/>"'{}]+""")


def _attributes(body):
    """``[(name, value)]`` of a tag's attributes. A quoted value keeps its
    text (quotes stripped, references decoded); a value holding an
    expression is ``None`` (not known until run time); a bare attribute is
    ``True``. A shorthand ``{name}`` is the attribute ``name`` with an
    unknown value; a spread is ``("...", None)``."""
    out, i, end = [], 0, len(body)
    while i < end:
        char = body[i]
        if char.isspace() or char == "/":
            i += 1
            continue
        if char == "{":
            stop = _skip_expression(body, i)
            inner = body[i + 1:stop - 1].strip()
            out.append(("...", None) if inner.startswith("...") else (inner, None))
            i = stop
            continue
        match = _NAME.match(body, i)
        if not match:
            i += 1
            continue
        name, i = match.group(0), match.end()
        look = i
        while look < end and body[look].isspace():
            look += 1
        if look >= end or body[look] != "=":
            out.append((name, True))
            continue
        look += 1
        while look < end and body[look].isspace():
            look += 1
        if look < end and body[look] in "'\"":
            stop = _skip_value(body, look)
            inner = body[look + 1:stop - 1]
            out.append((name, None if "{" in inner else html.unescape(inner)))
        elif look < end and body[look] == "{":
            stop = _skip_expression(body, look)
            out.append((name, None))
        else:
            bare = re.match(r"[^\s>]*", body[look:]).group(0)
            stop = look + len(bare)
            out.append((name, bare))
        i = stop
    return out


def _resolve(importer, spec):
    """The repository path an import names, or None when it is a package."""
    if spec == "$lib" or spec.startswith("$lib/"):
        path = f"{_SRC}/lib" + spec[len("$lib"):]
    elif spec.startswith("."):
        path = posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))
    else:
        return None
    return path


class Tree:
    """The frontend's components: their text, what each imports under which
    name, and where each is mounted. Built over any ``{path: text}``, so its
    rules are proven on synthetic trees too."""

    def __init__(self, sources):
        self.sources = dict(sources)
        self.barrels = {}
        for path, text in self.sources.items():
            if posixpath.basename(path) == "index.ts":
                folder = posixpath.dirname(path)
                for match in re.finditer(
                    r"export\s*\{\s*default\s+as\s+(\w+)\s*\}\s*from\s*['\"]([^'\"]+\.svelte)['\"]",
                    text,
                ):
                    self.barrels[(folder, match.group(1))] = posixpath.normpath(
                        posixpath.join(folder, match.group(2))
                    )
        self.names = {path: self._imports(path) for path in self.sources if path.endswith(".svelte")}
        self.mounts = {}
        self.children = {}
        for path in self.names:
            markup = _markup(self.sources[path])
            children = self.children.setdefault(path, set())
            for name, body, line in _tags(markup, r"[A-Z][\w]*"):
                target = self.names[path].get(name)
                if target:
                    self.mounts.setdefault(target, []).append((path, body, line))
                    children.add(target)

    def _imports(self, path):
        script = _script(self.sources[path], path)
        names = {}
        for match in re.finditer(r"import\s+(\w+)\s+from\s+['\"]([^'\"]+\.svelte)['\"]", script):
            target = _resolve(path, match.group(2))
            if target:
                names[match.group(1)] = target
        for match in re.finditer(r"import\s*\{([^}]*)\}\s*from\s*['\"]([^'\"]+)['\"]", script):
            folder = _resolve(path, match.group(2))
            if not folder:
                continue
            folder = re.sub(r"(?:/index)?(?:\.ts)?$", "", folder)
            for part in match.group(1).split(","):
                bits = re.split(r"\s+as\s+", part.strip())
                if bits[0] and (folder, bits[0]) in self.barrels:
                    names[bits[-1]] = self.barrels[(folder, bits[0])]
        return names

    def mounted(self, path):
        """The components ``path`` mounts."""
        return self.children.get(path, set())

    def closure(self, path):
        """``path`` and every component it mounts, transitively."""
        seen, todo = set(), [path]
        while todo:
            current = todo.pop()
            if current in seen or current not in self.sources:
                continue
            seen.add(current)
            todo.extend(self.mounted(current))
        return seen


_REAL = {}


def _real_tree():
    """The tree as it stands, built once per session."""
    if "tree" not in _REAL:
        _REAL["tree"] = Tree({path: read(path) for path in files(_EVERY)})
    return _REAL["tree"]


# ---------------------------------------------------------------------------
# Server rendering: the dialog's output, parsed as HTML
# ---------------------------------------------------------------------------
_VOID = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
})


class _Element:
    def __init__(self, tag, attrs, parent):
        self.tag = tag
        self.attrs = dict(attrs)
        self.parent = parent
        self.children = []

    def text(self):
        return "".join(
            child if isinstance(child, str) else child.text() for child in self.children
        )

    def iter(self, tag=None):
        if tag is None or self.tag == tag:
            yield self
        for child in self.children:
            if isinstance(child, _Element):
                yield from child.iter(tag)


class _Dom(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Element("#root", [], None)
        self._open = self.root

    def handle_starttag(self, tag, attrs):
        element = _Element(tag, attrs, self._open)
        self._open.children.append(element)
        if tag not in _VOID:
            self._open = element

    def handle_startendtag(self, tag, attrs):
        self._open.children.append(_Element(tag, attrs, self._open))

    def handle_endtag(self, tag):
        node = self._open
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root:
            self._open = node.parent

    def handle_data(self, data):
        self._open.children.append(data)


def _dom(text):
    parser = _Dom()
    parser.feed(text)
    parser.close()
    return parser.root


def _source(path, what):
    """A file the contract reads; its absence fails by name."""
    target = REPO / path
    assert target.is_file(), f"{path} is absent: {what}"
    return read(path)


def _imports(text, name, module):
    """True when ``text`` imports ``name`` (named or default) from ``module``."""
    named = (
        r"import\s*(?:type\s*)?\{[^}]*\b" + re.escape(name) + r"\b[^}]*\}\s*from\s*['\"]"
        + re.escape(module) + r"['\"]"
    )
    default = r"import\s+" + re.escape(name) + r"\s+from\s*['\"]" + re.escape(module) + r"['\"]"
    return re.search(named, text) is not None or re.search(default, text) is not None


def _call_spans(text, callee):
    """``(start, end)`` of every call ``callee(...)`` in ``text``, the span
    covering the parentheses and what they hold."""
    spans = []
    for match in re.finditer(r"(?<![\w.$])" + re.escape(callee) + r"\s*\(", text):
        depth, i = 0, match.end() - 1
        quote = None
        while i < len(text):
            char = text[i]
            if quote:
                if char == _BACKSLASH:
                    i += 2
                    continue
                if char == quote:
                    quote = None
            elif char in ("'", '"', "`"):
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        spans.append((match.start(), i + 1))
    return spans


def _function_body(text, name):
    """The body of the function ``name`` (declared with ``function``, or an
    arrow assigned to a ``const``/``let``), braces included, or None."""
    match = re.search(
        r"(?:function\s+" + re.escape(name) + r"\s*\([^)]*\)[^{]*"
        r"|(?:const|let)\s+" + re.escape(name) + r"\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::[^=]*)?=>\s*)\{",
        text,
    )
    if not match:
        return None
    depth, i = 0, match.end() - 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[match.end() - 1:i + 1]
        i += 1
    return None


def _count_in(sources, count_fn):
    """``{path: count}`` of the non-zero counts."""
    counts = {}
    for path, text in sources.items():
        found = count_fn(path, text)
        if found:
            counts[path] = found
    return counts


# ---------------------------------------------------------------------------
# UX1 -- every custom property read is declared
# ---------------------------------------------------------------------------
_TOKEN_READ = re.compile(r"var\(\s*(--oo-[A-Za-z0-9_-]+)")
_TOKEN_DECLARED = re.compile(r"(--oo-[A-Za-z0-9_-]+)['\"`]?\s*:")
_TOKEN_SET = re.compile(r"setProperty\(\s*['\"`](--oo-[A-Za-z0-9_-]+)")


def _undeclared(sources):
    """``{token: [paths]}`` of every ``--oo-*`` read and declared nowhere."""
    declared, read_at = set(), {}
    for path, text in sources.items():
        declared.update(_TOKEN_DECLARED.findall(text))
        declared.update(_TOKEN_SET.findall(text))
        for token in _TOKEN_READ.findall(text):
            read_at.setdefault(token, set()).add(path)
    return {token: sorted(paths) for token, paths in sorted(read_at.items()) if token not in declared}


def test_ux1_every_custom_property_read_is_declared():
    sample = {
        "frontend/src/a.svelte": (
            "<div style=\"color: var(--oo-fixture-missing)\" "
            "class=\"bg-[var(--oo-fixture-declared)]\"></div>"
        ),
        "frontend/src/b.css": ":root { --oo-fixture-declared: #fff; }",
        "frontend/src/c.ts": "el.style.setProperty('--oo-fixture-set', '1'); const x = 'var(--oo-fixture-set)';",
        "frontend/src/d.ts": "const style = { '--oo-fixture-keyed': 2 }; const y = 'var( --oo-fixture-keyed)';",
    }
    assert _undeclared(sample) == {"--oo-fixture-missing": ["frontend/src/a.svelte"]}, (
        "the census reads a token read nowhere declared, and only that one: "
        f"{_undeclared(sample)}"
    )
    sources = {path: read(path) for path in files(_EVERY)}
    sources["frontend/tailwind.config.js"] = read("frontend/tailwind.config.js")
    reads = sum(len(_TOKEN_READ.findall(text)) for text in sources.values())
    assert reads > 100, f"the census reads the tree's tokens: {reads}"
    missing = _undeclared(sources)
    assert not missing, f"custom properties read and declared nowhere: {missing}"


# ---------------------------------------------------------------------------
# UX2 -- every dispatched event is heard
# ---------------------------------------------------------------------------
def _dispatched(text):
    """The events a component dispatches through ``createEventDispatcher``."""
    names = re.findall(r"(?:const|let)\s+(\w+)\s*=\s*createEventDispatcher\b", text)
    events = set()
    for name in names:
        events.update(re.findall(r"(?<![\w.$])" + re.escape(name) + r"\(\s*['\"](\w+)['\"]", text))
    return events


def _listens(body, event):
    """How a mount's attributes treat ``event``: 'bound', 'forwarded' or None."""
    for name, value in _attributes(body):
        if re.fullmatch(r"on:" + re.escape(event) + r"(?:\|[\w|]+)?", name):
            return "forwarded" if value is True else "bound"
    return None


def _deaf_mounts(tree, component, event, seen=()):
    """``[(mounting file, line)]`` of the mounts of ``component`` that do not
    hear ``event``. A mount hears it when it binds ``on:<event>``, or
    forwards it from a component that is mounted and every one of whose
    mounts hears it."""
    deaf = []
    for path, body, line in tree.mounts.get(component, []):
        how = _listens(body, event)
        if how == "bound":
            continue
        if (
            how == "forwarded" and path not in seen and path != component
            and tree.mounts.get(path) and not _deaf_mounts(tree, path, event, (*seen, component))
        ):
            continue
        deaf.append((path, line))
    return deaf


_PRIMITIVES = f"{_SRC}/lib/ds/"

# Mounts that cannot raise an event their component dispatches, named with
# the reason and counted per mounting file. An exemption may only fall; one
# above the deaf mounts it covers is stale.
_DEAF_EXEMPT = {
    (f"{_SRC}/lib/components/chat/ChatMessage.svelte", "retry"): {
        # The reply still being written: Retry shows only on a finished one.
        f"{_SRC}/routes/(app)/(use)/chat/[id]/+page.svelte": 1,
    },
    (f"{_SRC}/lib/components/panels/PipelineEditor.svelte", "change"): {
        # The read-only view of a pipeline, whose steps cannot change.
        f"{_SRC}/lib/components/panels/ExecPipelinePanel.svelte": 1,
    },
}


def _dead_events(tree, exempt=None):
    """``[(component, event, [mount lines])]`` of every event that a mount of
    its component does not hear, beyond the named exemptions, or that no
    mount can hear because nothing mounts the component. The design
    system's primitives are not judged: their events are an interface
    offered to every consumer, and a mount may bind the value instead of
    listening."""
    exempt = exempt or {}
    dead = []
    for path in sorted(tree.names):
        if path.startswith(_PRIMITIVES):
            continue
        for event in sorted(_dispatched(tree.sources[path])):
            allowed = exempt.get((path, event), {})
            by_file = {}
            for where, line in _deaf_mounts(tree, path, event):
                by_file.setdefault(where, []).append(line)
            charged = [
                f"{where}:{line}" for where, lines in sorted(by_file.items())
                if len(lines) > allowed.get(where, 0) for line in lines
            ]
            if charged or not tree.mounts.get(path):
                dead.append((path, event, charged))
    return dead


def _stale_exemptions(tree, exempt):
    """Every exemption above the deaf mounts it covers: the count it allows
    in a mounting file is more than that file's deaf mounts."""
    stale = []
    for (component, event), allowed in sorted(exempt.items()):
        deaf = _deaf_mounts(tree, component, event)
        for where, count in sorted(allowed.items()):
            found = sum(1 for path, _ in deaf if path == where)
            if count > found:
                stale.append(f"{component} '{event}' at {where}: exempted {count}, deaf {found}")
    return stale


_UX2_SAMPLE = {
    "frontend/src/lib/Emitter.svelte": (
        "<script>import { createEventDispatcher } from 'svelte';\n"
        "const emit = createEventDispatcher();\n</script>\n"
        "<button on:click={() => emit('save')}>a</button>"
        "<button on:click={() => emit('drop', 1)}>b</button>"
        "<button on:click={() => emit('relay')}>c</button>"
    ),
    "frontend/src/lib/Other.svelte": (
        "<script>import { createEventDispatcher } from 'svelte';\n"
        "const dispatch = createEventDispatcher();\n</script>\n"
        "<button on:click={() => dispatch('drop')}>x</button>"
    ),
    "frontend/src/lib/Relay.svelte": (
        "<script>import Emitter from './Emitter.svelte';</script>\n"
        "<Emitter on:relay on:save={(e) => { if (e.detail > 1) keep(e); }} />"
    ),
    "frontend/src/lib/Holder.svelte": (
        "<script>import Relay from './Relay.svelte';\nimport Other from './Other.svelte';</script>\n"
        "<Relay on:relay={handle} />\n<Other on:drop={handle} />"
    ),
    "frontend/src/routes/+page.svelte": (
        "<script>import Emitter from '$lib/Emitter.svelte';\nimport Holder from '$lib/Holder.svelte';\n"
        "import { Picker } from '$lib/ds';</script>\n"
        "<Emitter on:drop on:save={keep} on:relay={keep} />\n<Holder />\n<Picker bind:value />"
    ),
    "frontend/src/lib/ds/index.ts": "export { default as Picker } from './Picker.svelte';\n",
    "frontend/src/lib/ds/Picker.svelte": (
        "<script>import { createEventDispatcher } from 'svelte';\n"
        "const dispatch = createEventDispatcher();\nexport let value;</script>\n"
        "<button on:click={() => { value = 1; dispatch('change', 1); }}>p</button>"
    ),
}


def test_ux2_every_dispatched_event_is_heard_by_a_mount_of_its_component():
    sample = Tree(_UX2_SAMPLE)
    dead = {(path.rsplit("/", 1)[-1], event) for path, event, _ in _dead_events(sample)}
    assert dead == {("Emitter.svelte", "drop")}, (
        "an event every mount binds is heard; a forward reaches a mount that "
        "binds it; a forward out of a route reaches nobody; another "
        "component's listener for the same name does not count; and a "
        f"primitive's offered event is not judged: {sorted(dead)}"
    )
    assert sample.mounts.get("frontend/src/lib/ds/Picker.svelte"), "a primitive is found through its barrel"

    deafened = Tree({
        **_UX2_SAMPLE,
        "frontend/src/lib/Second.svelte": "<script>import Other from './Other.svelte';</script>\n<Other />",
    })
    other = ("frontend/src/lib/Other.svelte", "drop")
    charged = {(path, event): mounts for path, event, mounts in _dead_events(deafened)}
    assert charged.get(other) == ["frontend/src/lib/Second.svelte:2"], (
        f"a second mount that does not hear the event is found beside one that does: {charged}"
    )
    covered = {other: {"frontend/src/lib/Second.svelte": 1}}
    assert other not in {(path, event) for path, event, _ in _dead_events(deafened, covered)}, (
        "a named exemption covers the deaf mounts it counts"
    )
    assert not _stale_exemptions(deafened, covered)
    assert _stale_exemptions(deafened, {other: {"frontend/src/lib/Second.svelte": 2}}), (
        "an exemption above the deaf mounts it covers is stale"
    )

    tree = _real_tree()
    dispatchers = [path for path in tree.names if _dispatched(tree.sources[path])]
    assert len(dispatchers) >= 10, f"the census finds the tree's dispatchers: {dispatchers}"
    assert sum(len(tree.mounts.get(path, [])) for path in dispatchers) >= len(dispatchers), (
        "the census finds the mounts of the tree's dispatchers"
    )
    dead = _dead_events(tree, _DEAF_EXEMPT)
    assert not dead, "events dispatched and heard by no mount:\n  " + "\n  ".join(
        f"{path} '{event}' (mounts: {', '.join(mounts) or 'none'})" for path, event, mounts in dead
    )
    stale = _stale_exemptions(tree, _DEAF_EXEMPT)
    assert not stale, "deaf-mount exemptions above what they cover (lower them):\n  " + "\n  ".join(stale)


# ---------------------------------------------------------------------------
# UX3 -- messages know their conversation; feedback never sends an empty id
# ---------------------------------------------------------------------------
def test_ux3_every_message_knows_its_conversation_and_feedback_never_sends_an_empty_id():
    tree = _real_tree()
    mounts = tree.mounts.get(_CHAT_MESSAGE, [])
    assert mounts, "ChatMessage is mounted somewhere"
    missing = [
        f"{path}:{line}" for path, body, line in mounts
        if "conversationId" not in {name for name, _ in _attributes(body)}
    ]
    assert not missing, f"ChatMessage mounts without conversationId: {missing}"

    feedback = tree.mounts.get(_FEEDBACK, [])
    assert feedback, "the feedback widget is mounted somewhere"
    for path, body, line in feedback:
        names = {name for name, _ in _attributes(body)}
        assert {"conversationId", "messageId"} <= names, (
            f"{path}:{line} passes the widget both ids: {sorted(names)}"
        )

    widget = _script(read(_FEEDBACK), _FEEDBACK)
    calls = _call_spans(widget, "submitFeedback")
    assert len(calls) == 1, f"the widget submits in one place: {len(calls)}"
    owner = _enclosing_function(widget, calls[0][0])
    body = _function_body(widget, owner) if owner else None
    assert body, "the widget submits from a named function"
    before = body[:body.index("submitFeedback")]
    guard = re.search(
        r"if\s*\(\s*!\s*(conversationId|messageId)\s*\|\|\s*!\s*(conversationId|messageId)\s*\)\s*"
        r"\{?\s*return\b",
        before,
    )
    assert guard is not None and {guard.group(1), guard.group(2)} == {"conversationId", "messageId"}, (
        "the widget returns before it submits when either id is empty"
    )

    thumb = _function_body(widget, "handleThumb") or ""
    chooses = re.search(r"\bfeedbackState\s*=(?!=)", thumb)
    early = re.search(
        r"if\s*\(\s*!\s*(conversationId|messageId)\s*\|\|\s*!\s*(conversationId|messageId)\s*\)\s*"
        r"\{?\s*return\b",
        thumb,
    )
    assert chooses and early and {early.group(1), early.group(2)} == {"conversationId", "messageId"} and (
        early.start() < chooses.start()
    ), "a thumb chooses nothing when either id is empty"
    markup = _markup(read(_FEEDBACK))
    first_thumb = markup.find("handleThumb(")
    opened = [m for m in re.finditer(r"\{#if\s+([^}]*)\}", markup) if m.start() < first_thumb]
    assert first_thumb > 0 and opened and {"conversationId", "messageId"} <= set(
        re.findall(r"\w+", opened[-1].group(1))
    ), "the thumbs are shown only when both ids are there"


# ---------------------------------------------------------------------------
# UX4 -- a chat request carries only ChatRequest fields
# ---------------------------------------------------------------------------
def _chat_request_fields():
    """The fields of ``ChatRequest``, read from the schema module by ``ast``."""
    tree = ast.parse((REPO / _SCHEMAS).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ChatRequest":
            return {
                item.target.id for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    raise AssertionError(f"no ChatRequest class in {_SCHEMAS}")


# Keys the chat stores sent at one time and the server never read.
_RETIRED_OPTIONS = ("no_cache", "cascading", "speculative", "prompt_" + "enhance", "humanize")

_NO_SELECTION = {
    "model": None, "preset": None, "temperature": None, "usePresets": True,
    "think": False, "webSearch": False, "quickSandbox": False, "chatCoding": False,
    "execPipeline": None,
}
_EVERY_SELECTION = {
    "model": "qwen3:8b", "preset": "code", "temperature": 0.3, "usePresets": False,
    "think": True, "webSearch": True, "quickSandbox": True, "chatCoding": True,
    "execPipeline": "review",
}


_COMPUTED_WRITE = re.compile(
    r"(?<![\w$.])\(?\s*(?:opts|request)\b(?:\s+as\s+[^)]*)?\)?\s*\[[^\]]*\]\s*=(?!=)"
    r"|Object\.assign\(\s*\(?\s*(?:opts|request)\b"
    r"|(?<![\w$.])(?:opts|request)\s*=\s*\{\s*\.\.\."
)


def _computed_writes(text, path):
    """The places a chat store writes a request key the census cannot name:
    by index, through ``Object.assign``, or by a spread into a new object."""
    return [match.group(0) for match in _COMPUTED_WRITE.finditer(_script(text, path))]


def _store_request_keys(text, path):
    """The request keys a chat store writes itself: ``opts.<key> =``,
    ``request.<key> =``, and the keys of a request object literal."""
    script = _script(text, path)
    keys = set(re.findall(r"\b(?:opts|request)\.(\w+)\s*=(?!=)", script))
    for match in re.finditer(r"\b(?:const|let)\s+request\b[^=]*=\s*\{", script):
        depth, i = 0, match.end() - 1
        while i < len(script):
            if script[i] == "{":
                depth += 1
            elif script[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = script[match.end():i]
        keys.update(re.findall(r"(?:^|[,{]\s*|\n\s*)(\w+)\s*:", body))
    return keys


@pytest.mark.parametrize("half", ("node", "wiring", "unchosen"))
def test_ux4_a_chat_request_carries_only_chat_request_fields(half):
    schema = _chat_request_fields()
    assert {"conversation_id", "message", "think", "web_search", "quick_sandbox"} <= schema, (
        f"the schema reader finds ChatRequest's fields: {sorted(schema)}"
    )
    for retired in _RETIRED_OPTIONS:
        assert retired not in schema, f"{retired} is not a ChatRequest field"

    if half == "wiring":
        sample = (
            "const opts = {};\nopts.model = m;\nopts.humanize = true;\n"
            "const request: Record<string, unknown> = {\n\tconversation_id: id,\n\tno_cache: x,\n};\n"
            "request.images = images;\n"
        )
        assert _store_request_keys(sample, "sample.ts") == {
            "model", "humanize", "conversation_id", "no_cache", "images",
        }, "the census reads each way a store writes a request key"

        options = _source(_CHAT_OPTIONS, "the chat options store")
        store = _source(_CHAT_STORE, "the chat store")
        written = {
            path: sorted(_store_request_keys(read(path), path) - schema)
            for path in (_CHAT_OPTIONS, _CHAT_STORE)
        }
        assert not any(written.values()), (
            f"the chat stores write request keys ChatRequest does not have: {written}"
        )
        assert _imports(options, "chatOptions", "$lib/chat/requestFields") and re.search(
            r"(?<![\w.$])chatOptions\s*\(", options
        ), "the options store builds the options through requestFields.chatOptions"
        assert _imports(store, "chatRequest", "$lib/chat/requestFields") and re.search(
            r"(?<![\w.$])chatRequest\s*\(", store
        ), "the chat store builds the request through requestFields.chatRequest"
        assert not re.search(r"\bopts\.\w+\s*=(?!=)", _script(options, _CHAT_OPTIONS)), (
            "the options store writes no option itself"
        )
        assert not re.search(r"\bconversation_id\s*:", _script(store, _CHAT_STORE)), (
            "the chat store builds no request literal of its own"
        )

        computed = (
            "(request as Record<string, unknown>)['humanize'] = true;\nopts[key] = value;\n"
            "Object.assign(request, extra);\nrequest = { ...request, cache: true };\n"
        )
        assert len(_computed_writes(computed, "sample.ts")) == 4, (
            "the census reads a key written by index, merged in, or spread into a new request"
        )
        assert _computed_writes("const x = request['model']; if (request[k] === 1) go();", "sample.ts") == [], (
            "a read by index is not a write"
        )
        for path in (_CHAT_OPTIONS, _CHAT_STORE):
            assert not _computed_writes(read(path), path), (
                f"{path} writes no request key by index, merge or spread: {_computed_writes(read(path), path)}"
            )

        declared_type = re.search(r"export\s+interface\s+ChatRequest\s*\{(.*?)\n\}", read(f"{_SRC}/lib/types.ts"), re.S)
        assert declared_type, "types.ts declares the ChatRequest the web chat sends"
        typed = set(re.findall(r"^\s*(\w+)\??\s*:", declared_type.group(1), re.M))
        listed_fields = re.search(r"REQUEST_FIELDS\s*=\s*\[(.*?)\]", _source(_REQUEST_FIELDS, "the request fields"), re.S)
        carried = set(re.findall(r"['\"](\w+)['\"]", listed_fields.group(1))) if listed_fields else set()
        assert typed and typed <= schema, (
            f"the frontend's ChatRequest type names only ChatRequest fields: {sorted(typed - schema)}"
        )
        assert typed == carried, (
            f"the frontend's ChatRequest type names exactly the fields the builder fills: {sorted(typed ^ carried)}"
        )
        return

    if half == "unchosen":
        # What a message sends when nothing is chosen, held apart so the
        # composer's later defaults can supersede this half alone.
        options = _node("options", ("OO_REQUEST_FIELDS",), [
            _NO_SELECTION,
            _EVERY_SELECTION,
            {**_NO_SELECTION, "temperature": 0},
        ])
        assert options[0] == {}, f"no selection sends no option: {options[0]}"
        assert options[2] == {"temperature": 0}, f"a temperature of 0 is a choice: {options[2]}"
        return

    listed = _node("fields", ("OO_REQUEST_FIELDS",))
    assert len(listed) == len(set(listed)), f"each field is listed once: {listed}"
    assert set(listed) <= schema, (
        f"every field the request may carry is a ChatRequest field: {sorted(set(listed) - schema)}"
    )
    assert {"conversation_id", "message", "images"} <= set(listed)

    options = _node("options", ("OO_REQUEST_FIELDS",), [
        _NO_SELECTION,
        _EVERY_SELECTION,
        {**_NO_SELECTION, "temperature": 0},
        {**_EVERY_SELECTION, "humanize": True, "cache": True, "no_cache": True,
         "cascading": True, "promptEnhance": True},
    ])
    assert options[1] == {
        "model": "qwen3:8b", "preset": "code", "temperature": 0.3, "use_presets": False,
        "think": True, "web_search": True, "quick_sandbox": True, "chat_coding": True,
        "exec_pipeline": "review",
    }, f"every selection sends its field: {options[1]}"
    assert options[3] == options[1], (
        f"a selection that is no request field builds nothing: {options[3]}"
    )
    for built in options:
        assert set(built) <= set(listed) - {"conversation_id", "message", "images"}, (
            f"the options builder builds only request fields, never the request's own: {built}"
        )

    requests = _node("request", ("OO_REQUEST_FIELDS",), [
        ["c-1", "hello", {}],
        ["c-1", "hello", {
            "model": "m", "images": ["aW1n"], "humanize": True, "no_cache": True,
            "cascading": True, "prompt_" + "enhance": True, "speculative": True,
            "usePresets": False, "message": "not this", "conversation_id": "not this",
        }],
        ["c-2", "again", {"images": [], "preset": None, "model": "m", "use_presets": False}],
    ])
    assert requests[0] == {"conversation_id": "c-1", "message": "hello"}
    assert requests[1] == {
        "conversation_id": "c-1", "message": "hello", "model": "m", "images": ["aW1n"],
    }, f"the request drops every key that is not a request field: {requests[1]}"
    assert requests[2] == {
        "conversation_id": "c-2", "message": "again", "model": "m", "use_presets": False,
    }, f"an empty image list and an unset option are not sent: {requests[2]}"
    for built in requests:
        assert set(built) <= schema, f"a request carries only ChatRequest fields: {built}"


# ---------------------------------------------------------------------------
# UX5 -- a server switch shows the server's confirmed state
# ---------------------------------------------------------------------------
_SWITCH_ENDPOINTS = (
    "/api/cache/semcache/toggle", "/api/cascading/config", "/api/humanizer/config",
    "/api/sandbox/quick/toggle", "/api/chat/coding/toggle",
)
_SWITCH_TOGGLES = ("toggleSemCache", "setQuickSandbox", "setChatCoding")
_SWITCH_CONFIGS = ("updateCascadingConfig", "updateHumanizerConfig", "updateSemCacheConfig")
_SWITCH_STORES = (
    "cacheEnabled", "cascadingEnabled", "humanizeEnabled", "quickSandboxEnabled",
    "chatCodingEnabled",
)


def _enabled_only(text, callee):
    """The calls of ``callee`` whose argument is an object holding only
    ``enabled``: a switch, not a form's save."""
    return [
        (start, end) for start, end in _call_spans(text, callee)
        if re.fullmatch(r"\(\s*\{\s*enabled\s*(?::[^,{}]*)?,?\s*\}\s*\)", text[text.index("(", start):end])
    ]


def _switch_writes(text):
    """``(start, end)`` of every place ``text`` writes a server switch."""
    spans = []
    for callee in _SWITCH_TOGGLES:
        spans.extend(_call_spans(text, callee))
    for callee in _SWITCH_CONFIGS:
        spans.extend(_enabled_only(text, callee))
    for endpoint in _SWITCH_ENDPOINTS:
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(endpoint) + r"(?![\w/-])", text))
    return sorted(spans)


_FLIP = re.compile(
    r"(?<![\w$.])([\w$.]+)\s*=\s*!\s*\1(?![\w$])"
    r"|\.update\(\s*\(?\s*(\w+)\s*\)?\s*=>\s*!\s*\2\b"
)


def _flips(script):
    """Places a component flips a server switch's state itself: ``x = !x``,
    or a store updated to its negation, inside a server switch or in a
    function that writes a switch or toggles one. A flip elsewhere (a
    disclosure, a pill the server never sees) is not a switch's."""
    switches = re.findall(r"(?:const|let)\s+(\w+)\s*=\s*createServerSwitch\s*\(", script)
    inside = _call_spans(script, "createServerSwitch")
    found = []
    for match in _FLIP.finditer(script):
        if any(low <= match.start() < high for low, high in inside):
            found.append(match.group(0))
            continue
        owner = _enclosing_function(script, match.start())
        body = (_function_body(script, owner) or "") if owner else ""
        if _switch_writes(body) or any(
            re.search(r"(?<![\w$.])" + re.escape(name) + r"\.toggle\(", body) for name in switches
        ):
            found.append(match.group(0))
    return found


def _object_properties(text, brace):
    """``{name: (start, end)}`` of the values of the top-level properties of
    the object literal whose ``{`` is at ``brace``; a method shorthand's
    value is its whole definition."""
    segments, depth, start, i, end = [], 0, brace + 1, brace + 1, len(text)
    while i < end:
        char = text[i]
        if char in "'\"`":
            i = _skip_string(text, i)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                segments.append((start, i))
                break
            depth -= 1
        elif char == "," and depth == 0:
            segments.append((start, i))
            start = i + 1
        i += 1
    props = {}
    for low, high in segments:
        key = re.match(r"\s*(\w+)\s*:", text[low:high])
        method = re.match(r"\s*(?:async\s+)?(\w+)\s*\(", text[low:high])
        if key:
            props[key.group(1)] = (low + key.end(), high)
        elif method:
            props[method.group(1)] = (low + method.start(1), high)
    return props


def _switches(script):
    """``[(name, (start, end), {property: (start, end)})]`` of every server
    switch the script creates: its name, its ``createServerSwitch`` call, and
    the properties of the options it is given."""
    out = []
    for start, end in _call_spans(script, "createServerSwitch"):
        named = re.search(r"(?:const|let)\s+(\w+)\s*=\s*$", script[:start])
        brace = script.find("{", start, end)
        props = _object_properties(script, brace) if brace >= 0 else {}
        out.append((named.group(1) if named else None, (start, end), props))
    return out


def _expression(body, name):
    """The raw text of the attribute ``name`` of a tag: the expression of
    ``name={...}``, or the text of a quoted value; None when absent."""
    match = re.search(r"(?<![\w:-])" + re.escape(name) + r"(?:\|[\w|]+)?\s*=\s*", body)
    if not match:
        return None
    i = match.end()
    if i < len(body) and body[i] == "{":
        return body[i + 1:_skip_expression(body, i) - 1]
    if i < len(body) and body[i] in "'\"":
        return body[i + 1:_skip_value(body, i) - 1]
    return re.match(r"[^\s>]*", body[i:]).group(0)


def _toggled(script, handler, names):
    """The switches among ``names`` that a click handler toggles: in its own
    text, or in the body of the function it names."""
    text = handler.strip()
    if re.fullmatch(r"\w+", text):
        text = _function_body(script, text) or ""
    return {name for name in names if re.search(r"(?<![\w$.])" + re.escape(name) + r"\.toggle\b", text)}


def _handles_null(script, callee):
    """True when the function ``callee`` of the script tests its argument
    against null (or is the module's ``pressed``)."""
    if callee == "pressed":
        return True
    body = _function_body(script, callee) or ""
    return re.search(r"(?:===|!==|==|!=)\s*null\b|\bnull\s*(?:===|!==|==|!=)", body) is not None


def _blank_comments(script):
    """The script with its comments blanked (a line comment only where a
    ``//`` starts a line or follows a space, so a URL in a string stays)."""
    script = re.sub(r"/\*.*?\*/", _blank, script, flags=re.S)
    return re.sub(r"(?:(?<=^)|(?<=\s))//[^\n]*", _blank, script, flags=re.M)


def _truthy_reads(code, script, names):
    """Reads of a switch's ``$name.value`` that test its truthiness: not
    compared with true, false or null, and not handed to a function that
    handles null. An unknown (null) state read that way is drawn as off."""
    found = []
    for match in re.finditer(r"\$(\w+)\.value\b", code):
        if match.group(1) not in names:
            continue
        if re.match(r"\s*(?:===|!==|==|!=)\s*(?:true|false|null|undefined)\b", code[match.end():]):
            continue
        if re.search(r"\b(?:true|false|null)\s*(?:===|!==|==|!=)\s*$", code[:match.start()]):
            continue
        depth, i = 0, match.start() - 1
        while i >= 0 and not (depth == 0 and code[i] in "{};"):
            if code[i] == ")":
                depth += 1
            elif code[i] == "(":
                if depth == 0:
                    break
                depth -= 1
            i -= 1
        callee = re.search(r"([\w$]+)\s*$", code[:i]) if i >= 0 and code[i] == "(" else None
        if callee and _handles_null(script, callee.group(1)):
            continue
        found.append(match.group(0))
    return found


def _shows_error(markup, name):
    """True when the markup shows the error of the switch ``name``: as text
    (``{$name.error}`` outside every tag), or handed as the ``message`` or
    ``error`` of an InlineError or a ConfirmDialog. A read in a condition
    or another attribute does not show it."""
    text = markup
    for match in re.finditer(r"<[A-Za-z][\w:.-]*(?=[\s/>])", markup):
        stop = _tag_end(markup, match.start())
        text = text[:match.start()] + re.sub(r"[^\n]", " ", text[match.start():stop]) + text[stop:]
    if re.search(r"\{\s*\$" + re.escape(name) + r"\.error\s*\}", text):
        return True
    return any(
        re.search(r"(?<![\w:-])(?:message|error)\s*=\s*\{\s*\$" + re.escape(name) + r"\.error\b", body)
        for _, body, _ in _tags(markup, r"InlineError|ConfirmDialog")
    )


def _switch_components(sources):
    return sorted(
        path for path, text in sources.items()
        if path.endswith(".svelte") and _switch_writes(text)
    )


def _switch_findings(path, text):
    """What keeps a switch component from showing the confirmed state."""
    script = _script(text, path)
    findings = []
    if not _imports(script, "createServerSwitch", "$lib/switches/serverSwitch"):
        findings.append("does not import createServerSwitch")
    inside = _call_spans(script, "createServerSwitch")
    if not inside:
        findings.append("creates no server switch")
    for endpoint in _SWITCH_ENDPOINTS:
        if re.search(re.escape(endpoint) + r"(?![\w/-])", text):
            findings.append(f"spells {endpoint} rather than calling the API layer")
    for start, end in _switch_writes(script):
        if not any(low <= start and end <= high for low, high in inside):
            findings.append(f"writes a switch outside createServerSwitch: {script[start:end][:60]!r}")
    for flip in _flips(script):
        findings.append(f"flips a state itself: {flip!r}")
    switches = _switches(script)
    names = {name for name, _, _ in switches if name}
    kept = [props[key] for _, _, props in switches for key in ("adopt", "forget") if key in props]
    for store in _SWITCH_STORES:
        for match in re.finditer(r"(?<![\w$])" + store + r"\.(?:set|update)\(|\$" + store + r"\s*=(?!=)", script):
            if not any(low <= match.start() < high for low, high in kept):
                findings.append(f"assigns {store} outside the switch's adoption")
    markup = _markup(text)
    if not re.search(r"\$\w+\.error\b", markup):
        findings.append("shows no error a switch names")
    for name, _, props in switches:
        if name and not _shows_error(markup, name):
            findings.append(f"shows no error of the {name} switch")
        low, high = props.get("write", (0, 0))
        write = script[low:high]
        parameter = re.match(r"\s*(?:async\s*)?(?:\(\s*(\w*)[^)]*\)|(\w+))\s*(?::[^=]*)?=>", write)
        asked = parameter and (parameter.group(1) or parameter.group(2))
        echoes = asked and re.search(
            r"(?:\breturn\s+|=>\s*\(?\s*)" + re.escape(asked) + r"\b\s*\)?\s*(?:[;}\n]|$)", write
        )
        answered = re.search(r"\(\s*await\s+[\w$.]+\([^()]*\)\s*\)\.enabled\b", write) or any(
            re.search(r"\breturn\s+" + re.escape(var) + r"\.enabled\b", write)
            for var in re.findall(r"(\w+)\s*=\s*await\s+[\w$.]+\(", write)
        )
        if echoes or not answered:
            findings.append(f"the {name} switch's write does not return the state the server answers")
    for callee in _SWITCH_CONFIGS:
        for start, end in _call_spans(script, callee):
            argument = script[script.index("(", start) + 1:end - 1]
            if re.search(r"(?:^|[{,])\s*enabled\s*[:,}]", argument) and not any(
                low <= start and end <= high for low, high in inside
            ):
                findings.append(f"writes enabled through {callee} beside its switch")
    code = markup + "\n" + _blank_comments(script)
    for read_ in _truthy_reads(code, script, names):
        findings.append(f"reads {read_} for its truthiness: an unknown state would be drawn as off")
    for _, body, line in _tags(markup, r"[A-Za-z][\w:.-]*"):
        handler = _expression(body, "on:click")
        toggles = _toggled(script, handler, names) if handler else set()
        if not toggles:
            continue
        pressed_ = (_expression(body, "aria-pressed") or "").strip()
        if not any(pressed_ == f"pressed(${name}.value)" for name in toggles):
            findings.append(
                f"a control (line {line}) that toggles {'/'.join(sorted(toggles))} does not announce "
                f"pressed(${'/'.join(sorted(toggles))}.value): {pressed_!r}"
            )
        style = _expression(body, "style")
        if style is not None and not any(f"${name}.value" in style for name in toggles):
            findings.append(
                f"a control (line {line}) that toggles {'/'.join(sorted(toggles))} draws from something "
                "other than the switch's value"
            )
    return findings


@pytest.mark.parametrize("half", ("node", "wiring", "ssr"))
def test_ux5_a_server_switch_shows_the_servers_confirmed_state(half):
    if half == "wiring":
        flipping = (
            "<script>\n\tconst cache = createServerSwitch({ label: 'Semantic cache', read,\n"
            "\t\twrite: async () => (await toggleSemCache()).enabled });\n"
            "\tasync function toggleCache() {\n"
            "\t\ttry { const r = await fetch('/api/cache/semcache/toggle'); }\n"
            "\t\tcatch { cacheEnabled.update((v) => !v); }\n\t}\n"
            "\tfunction optimistic() { shown = !shown; cache.toggle(); }\n"
            "\tfunction disclose() { open = !open; }\n"
            "\tfunction think() { thinking.update((v) => !v); }\n</script>\n"
            "<button aria-pressed={$cacheEnabled}>Cache</button>"
        )
        findings = _switch_findings("frontend/src/Fixture.svelte", flipping)
        assert any("spells /api/cache/semcache/toggle" in f for f in findings)
        flips = [f for f in findings if "flips a state itself" in f]
        assert len(flips) == 2 and not any("open" in f or "thinking" in f for f in flips), (
            "a flip next to a switch write or toggle is found, a disclosure or a "
            f"client-only pill is not: {flips}"
        )
        assert any("assigns cacheEnabled" in f for f in findings)
        confirmed = (
            "<script>\n\timport { createServerSwitch, pressed } from '$lib/switches/serverSwitch';\n"
            "\tconst cache = createServerSwitch({ label: 'Semantic cache',\n"
            "\t\tread: async () => (await getSemCacheStatus()).enabled,\n"
            "\t\twrite: async () => (await toggleSemCache()).enabled,\n"
            "\t\tadopt: (value) => cacheEnabled.set(value) });\n</script>\n"
            "<button aria-pressed={pressed($cache.value)} on:click={cache.toggle}>Cache</button>"
            "{#if $cache.error}<p>{$cache.error}</p>{/if}"
        )
        assert _switch_findings("frontend/src/Fixture.svelte", confirmed) == []
        form = "<script>updateHumanizerConfig({ enabled: on, mode });</script>"
        assert _switch_writes(form) == [], "a form's save is not a switch"

        def found(fixture):
            return _switch_findings("frontend/src/Fixture.svelte", fixture)

        answer = "write: async () => (await toggleSemCache()).enabled,"
        early = confirmed.replace(
            answer, "write: async (next) => { cacheEnabled.set(next); return (await toggleSemCache()).enabled; },"
        )
        assert any("assigns cacheEnabled" in f for f in found(early)), (
            "a mirror written in the switch's own write, before the answer, is found"
        )
        echoed = confirmed.replace(answer, "write: async (next) => { await toggleSemCache(); return next; },")
        assert any("does not return the state the server answers" in f for f in found(echoed)), (
            "a write that returns the state it asked for is found"
        )
        fixed = confirmed.replace("aria-pressed={pressed($cache.value)}", "aria-pressed={true}")
        assert any("does not announce" in f for f in found(fixed)), "a control that announces a fixed state is found"
        mirrored = confirmed.replace("<button aria-pressed", "<button style={look(shown)} aria-pressed")
        assert any("draws from something other" in f for f in found(mirrored)), (
            "a control drawn from a mirror of its own is found"
        )
        truthy = confirmed.replace("{#if $cache.error}", "<i class={$cache.value ? 'on' : 'off'}></i>{#if $cache.error}")
        assert any("for its truthiness" in f for f in found(truthy)), "a value read for its truthiness is found"
        handled = confirmed.replace(
            "</script>", "\tfunction look(value) { return value === null ? 'dim' : value ? 'on' : 'off'; }\n</script>"
        ).replace("{#if $cache.error}", "<i class={look($cache.value)}></i>{#if $cache.error}")
        assert found(handled) == [], "a value handed to a function that handles null is not a truthiness read"
        two = confirmed.replace("</script>", (
            "\tconst humanizer = createServerSwitch({ label: 'Output humanizer',\n"
            "\t\tread: async () => (await getHumanizerConfig()).enabled,\n"
            "\t\twrite: async (next) => (await updateHumanizerConfig({ enabled: next })).enabled });\n</script>"
        ))
        assert any("no error of the humanizer switch" in f for f in found(two)), (
            "each switch's error is shown, not one for all"
        )
        tested = confirmed.replace("<p>{$cache.error}</p>", "<p>failed</p>")
        assert any("no error of the cache switch" in f for f in found(tested)), (
            "an error read in a condition and never shown is not shown"
        )
        beside = confirmed.replace(
            "</script>", "\tasync function save() { await updateSemCacheConfig({ enabled: shown, ttl_seconds: 60 }); }\n</script>"
        )
        assert any("beside its switch" in f for f in found(beside)), "a form that saves the state beside its switch is found"

        sources = {path: read(path) for path in files(".svelte")}
        components = _switch_components(sources)
        assert components, "the census finds the components that toggle a server switch"
        assert {
            f"{_SRC}/lib/components/chat/ChatControlBar.svelte",
            f"{_SRC}/lib/components/panels/CacheStatsPanel.svelte",
            f"{_SRC}/lib/components/panels/HumanizerPanel.svelte",
        } <= set(components), f"the census finds each known switch component: {components}"
        findings = {path: _switch_findings(path, sources[path]) for path in components}
        findings = {path: found for path, found in findings.items() if found}
        assert not findings, "switch components that do not show the confirmed state:\n  " + "\n  ".join(
            f"{path}: {'; '.join(found)}" for path, found in findings.items()
        )
        return

    if half == "ssr":
        # Before any read (the server renders without running onMount),
        # every switch control announces no state and cannot be pressed.
        controls = {
            f"{_SRC}/lib/components/chat/ChatControlBar.svelte": (
                "Toggle semantic cache", "Toggle cascading inference", "Toggle humanizer",
                "Toggle quick sandbox", "Toggle chat coding agent",
            ),
            f"{_SRC}/lib/components/panels/CacheStatsPanel.svelte": ("Toggle cache",),
            f"{_SRC}/lib/components/panels/HumanizerPanel.svelte": ("Toggle humanizer",),
        }
        for path, labels in controls.items():
            shown = _dom(ssr().render(path, {}).html)
            buttons = {button.attrs.get("aria-label"): button for button in shown.iter("button")}
            for label in labels:
                control = buttons.get(label)
                assert control is not None, f"{path} renders the control {label!r}: {sorted(map(str, buttons))}"
                assert "aria-pressed" not in control.attrs, (
                    f"{path}: {label!r} announces a state the server has not confirmed: "
                    f"aria-pressed={control.attrs.get('aria-pressed')!r}"
                )
                assert "disabled" in control.attrs, f"{path}: {label!r} can be pressed before its state is known"
            if "/panels/" in path:
                assert "Reading..." in shown.text(), f"{path} says the switch's state is being read"
        return

    pressed_ = _node("pressed", ("OO_SERVER_SWITCH",))
    assert pressed_ == {"on": True, "off": False, "unknownAbsent": True}, (
        f"a control announces the confirmed state, and none while it is unknown: {pressed_}"
    )

    adopts = _node("switch", ("OO_SERVER_SWITCH",), "adopts")
    assert adopts["loaded"] == {"value": False, "pending": False, "error": None}
    assert adopts["first"] == "adopted"
    assert adopts["afterFirst"] == {"value": True, "pending": False, "error": None}
    assert adopts["second"] == "adopted" and adopts["log"]["writes"][:2] == [True, False]
    assert adopts["afterSecond"]["value"] is True, (
        "a success adopts the state the server answers, not the one asked for"
    )
    assert adopts["log"]["adopted"][:3] == [False, True, True], adopts["log"]["adopted"]
    assert adopts["seen"][0] == {"value": None, "pending": False, "error": None}
    assert {"value": False, "pending": True, "error": None} in adopts["seen"], (
        "subscribers see the pending state, with the old value"
    )
    assert adopts["seenAfterStop"] == 0, "an unsubscribed listener hears nothing more"
    mismatch = adopts["afterSecond"]["error"] or ""
    assert "Semantic cache" in mismatch and "on" in mismatch and "off" in mismatch, (
        f"an answer other than the state asked for is named: {mismatch!r}"
    )
    assert adopts["log"]["forgot"] == 0, "a confirmed state is never forgotten"

    refused = _node("switch", ("OO_SERVER_SWITCH",), "refused")
    for name, status in (("forbidden", "403"), ("bare", "409")):
        case = refused[name]
        assert case["outcome"] == "kept", case
        assert case["state"]["value"] is True and case["state"]["pending"] is False, (
            f"an HTTP refusal keeps the state: {case['state']}"
        )
        error = case["state"]["error"] or ""
        assert "Semantic cache" in error and status in error, (
            f"the refusal is named, with the switch and the status: {error!r}"
        )
        assert case["log"]["reads"] == 1, "a refusal is an answer: nothing is re-read"
    assert "CSRF token missing" in refused["forbidden"]["state"]["error"]

    unreached = _node("switch", ("OO_SERVER_SWITCH",), "unreached")
    landed = unreached["landed"]
    assert landed["outcome"] == "reread" and landed["log"]["reads"] == 2, landed
    assert landed["state"]["value"] is True, (
        f"a failure to reach the server re-reads the state: {landed['state']}"
    )
    assert landed["state"]["error"], "the failure is still named"
    lost = unreached["lost"]
    assert lost["outcome"] == "reread" and lost["state"]["value"] is None, (
        f"when the re-read fails too, the state is unknown: {lost['state']}"
    )
    assert "Semantic cache" in (lost["state"]["error"] or "")
    assert lost["log"]["forgot"] == 1 and landed["log"]["forgot"] == 0, (
        "the mirrors forget a state that became unknown, and only then"
    )
    for name in ("thrown", "stateless"):
        case = unreached[name]
        assert case["outcome"] == "reread" and case["log"]["reads"] == 2, (
            f"{name}: an unknown outcome is re-read: {case}"
        )
    assert "may not have reached the server" in (unreached["thrown"]["state"]["error"] or ""), (
        "an error that is no HTTP answer is named as a change that may not have reached the server, "
        f"never as the server's answer: {unreached['thrown']['state']['error']!r}"
    )

    pending = _node("switch", ("OO_SERVER_SWITCH",), "pending")
    assert pending["during"] == {"value": False, "pending": True, "error": None}, (
        f"the state does not flip before the server answers: {pending['during']}"
    )
    assert pending["second"] == "refused", "a second toggle while one is pending is refused"
    assert pending["writesDuring"] == 1, "the refused toggle sends nothing"
    assert pending["loadDuring"] == pending["during"] and pending["readsDuring"] == 1, (
        "a read is not started while a toggle is pending"
    )
    assert pending["firstOutcome"] == "adopted"
    assert pending["after"] == {"value": True, "pending": False, "error": None}
    assert pending["third"] == "adopted" and pending["log"]["writes"] == [True, False], (
        "once settled, the switch toggles again"
    )

    failed = _node("switch", ("OO_SERVER_SWITCH",), "failed")
    for name, value, status in (("landed", True, "500"), ("before", False, "503")):
        case = failed[name]
        assert case["outcome"] == "reread" and case["log"]["reads"] == 2, (
            f"{name}: a server error is not a refusal, and the state is read again: {case}"
        )
        assert case["state"]["value"] is value, f"{name}: the state is the one read again: {case['state']}"
        error = case["state"]["error"] or ""
        assert "Semantic cache" in error and status in error, f"{name}: the server error is named: {error!r}"
    assert "boom" in (failed["landed"]["state"]["error"] or ""), "the server's own reason is named"

    unknown = _node("switch", ("OO_SERVER_SWITCH",), "unknown")
    assert unknown["before"] == "refused" and unknown["log"]["writes"] == [], (
        "a state never read cannot be toggled"
    )
    assert unknown["loaded"]["value"] is None and "Semantic cache" in (unknown["loaded"]["error"] or "")
    assert unknown["after"] == "refused"
    assert unknown["log"]["forgot"] == 1, "a state that could not be read is forgotten by its mirrors"


# ---------------------------------------------------------------------------
# UX6 -- no raw fetch under the chat components
# ---------------------------------------------------------------------------
_RAW_FETCH = re.compile(r"(?<![\w.$])(?:(?:window|globalThis|self)\??\.)?fetch\(")


def test_ux6_no_raw_fetch_under_the_chat_components():
    sample = "await fetch('/a'); window.fetch(b); globalThis.fetch(c); api.fetch(d); prefetch(e);"
    assert len(_RAW_FETCH.findall(sample)) == 3, "the census reads the global fetch in its spellings"
    listed = files(_SCRIPTS, within=f"{_SRC}/lib/components/chat")
    counts = _count_in({path: read(path) for path in listed}, lambda p, t: len(_RAW_FETCH.findall(t)))
    assert not counts, f"raw fetch under the chat components: {counts}"


# ---------------------------------------------------------------------------
# UX7 -- a wipe runs only once confirmed, and its failure is shown
# ---------------------------------------------------------------------------
_WIPES = ("wipeConversation", "wipeAllConversations")
_WIPE_ENDPOINT = "/api/security/conversation" + "-wipe"
# The other actions the frontend offers that cannot be undone.
_DESTRUCTIVE = ("deleteDocument", "deleteCollection", "unregisterVariant", "disableRemoteAccess", "revokeClientCert")


def _enclosing_function(script, offset):
    """The name of the innermost named function around ``offset``."""
    best = None
    for match in re.finditer(
        r"(?:function\s+(\w+)\s*\(|(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::[^=]*)?=>\s*\{)",
        script,
    ):
        name = match.group(1) or match.group(2)
        body = _function_body(script, name)
        if body is None:
            continue
        start = script.index(body, match.start())
        if start <= offset < start + len(body):
            best = name
    return best


def _rendered_errors(markup):
    """The names the markup renders: as text (``{name}`` outside every
    tag), or handed as the ``error`` or ``message`` of a ConfirmDialog or an
    InlineError, which show it in an alert. A name in any other attribute
    (``open={name}``, ``disabled={name}``) is not rendered."""
    text = markup
    for match in re.finditer(r"<[A-Za-z][\w:.-]*(?=[\s/>])", markup):
        stop = _tag_end(markup, match.start())
        text = text[:match.start()] + re.sub(r"[^\n]", " ", text[match.start():stop]) + text[stop:]
    names = set(re.findall(r"\{\s*(\w+)\s*\}", text))
    for _, body, _ in _tags(markup, r"ConfirmDialog|InlineError"):
        names.update(re.findall(r"(?<![\w:-])(?:error|message)\s*=\s*\{\s*(\w+)\b", body))
    return names


def _catch_bodies(body):
    """The text of every ``catch`` block in a function body, braces matched."""
    out = []
    for match in re.finditer(r"\bcatch\b[^{]*\{", body):
        depth, i = 0, match.end() - 1
        while i < len(body):
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append(body[match.end():i])
    return out


def _wipe_findings(path, text, callees=_WIPES, toast=False):
    """What keeps ``callees`` from running only once confirmed, with their
    failure shown (rendered, or with ``toast`` also toasted)."""
    script = _script(text, path)
    markup = _markup(text)
    findings = []
    dialogs = [body for name, body, _ in _tags(markup, r"ConfirmDialog")]
    if not dialogs:
        findings.append("mounts no ConfirmDialog")
    confirmed = set()
    for body in dialogs:
        match = re.search(r"onConfirm\s*=\s*\{\s*(\w+)\s*\}", body)
        if match:
            confirmed.add(match.group(1))
    rendered_errors = _rendered_errors(markup)
    acting = set()
    for callee in callees:
        for _ in _call_spans(markup, callee):
            findings.append(f"{callee} is called from the markup, not from a ConfirmDialog's onConfirm")
        for start, _ in _call_spans(script, callee):
            if re.search(r"\bfunction\s*$", script[:start]):
                continue
            owner = _enclosing_function(script, start)
            if owner not in confirmed:
                findings.append(f"{callee} runs in {owner or 'no named function'}, not a ConfirmDialog's onConfirm")
                continue
            acting.add(owner)
            body = _function_body(script, owner) or ""
            assigned = set()
            for catch in _catch_bodies(body):
                assigned.update(re.findall(r"\b(\w+)\s*=(?!=)", catch))
            toasted = toast and re.search(r"(?<![\w$.])toastError\s*\(", body)
            if not assigned & rendered_errors and not toasted:
                findings.append(f"{owner} assigns its failure to nothing the file renders")
    for owner in sorted(acting):
        reached = len(re.findall(r"(?<![\w$.])" + re.escape(owner) + r"(?![\w$])", markup))
        confirming = len(re.findall(r"onConfirm\s*=\s*\{\s*" + re.escape(owner) + r"\s*\}", markup))
        if reached != confirming:
            findings.append(f"{owner} is reached from the markup other than as a ConfirmDialog's onConfirm")
        for match in re.finditer(r"(?<![\w$.])" + re.escape(owner) + r"\s*\(", script):
            if re.search(r"\bfunction\s*$", script[:match.start()]):
                continue
            findings.append(f"{owner} is called from {_enclosing_function(script, match.start()) or 'the script'}")
        stores = sorted(set(re.findall(r"\$(\w+)", _function_body(script, owner) or "")))
        if stores:
            findings.append(
                f"{owner} reads {', '.join('$' + s for s in stores)} when it runs, not what was asked about"
            )
    if _WIPE_ENDPOINT in text:
        findings.append("spells the wipe endpoint rather than calling the API layer")
    return findings


def _wipe_files(sources, callees=_WIPES):
    """The components that call one of ``callees``, in their script or their
    markup (or spell the wipe endpoint)."""
    return sorted(
        path for path, text in sources.items()
        if path.endswith(".svelte")
        and (_WIPE_ENDPOINT in text or any(_call_spans(_HTML_COMMENT.sub(_blank, text), c) for c in callees))
    )


_CONFIRM_PROPS = {
    "open": True, "title": "Wipe this conversation?",
    "message": "Its messages are zeroed in memory.", "confirmLabel": "Wipe",
    "cancelLabel": "Keep it", "danger": True,
}


@pytest.mark.parametrize("half", ("wipe", "dialog", "ssr", "destructive"))
def test_ux7_a_wipe_runs_only_once_confirmed_and_its_failure_is_shown(half):
    if half == "wipe":
        bare = (
            "<script>\n\tasync function handleWipe() {\n\t\ttry { await wipeConversation(id); }\n"
            "\t\tcatch { }\n\t}\n</script>\n<button on:click={handleWipe}>Wipe</button>"
        )
        assert _wipe_findings("frontend/src/Fixture.svelte", bare), "an unconfirmed wipe is found"
        guarded = (
            "<script>\n\tlet wipeError = null;\n\tasync function runWipe() {\n"
            "\t\ttry { await wipeConversation(id); open = false; }\n"
            "\t\tcatch (e) { wipeError = e.message; }\n\t}\n</script>\n"
            "<ConfirmDialog {open} title=\"Wipe?\" onConfirm={runWipe} error={wipeError} onCancel={close} />"
        )
        assert _wipe_findings("frontend/src/Fixture.svelte", guarded) == []
        silent = guarded.replace("wipeError = e.message;", "console.warn(e);")
        assert any("assigns its failure" in f for f in _wipe_findings("frontend/src/Fixture.svelte", silent))

        def found(fixture):
            return _wipe_findings("frontend/src/Fixture.svelte", fixture)

        closing = guarded.replace("wipeError = e.message;", "open = false;")
        assert any("assigns its failure" in f for f in found(closing)), (
            "a failure that only closes the dialog is swallowed: a name in another attribute is not rendered"
        )
        clicked = guarded + "\n<button on:click={runWipe}>Wipe</button>"
        assert any("reached from the markup" in f for f in found(clicked)), (
            "the confirmed handler bound to a button runs without the question"
        )
        called = guarded.replace("async function runWipe() {", "function ask() { runWipe(); }\n\tasync function runWipe() {")
        assert any("runWipe is called from ask" in f for f in found(called)), (
            "the confirmed handler called from another function runs without the question"
        )
        inline = "<script>\n\tlet id = 'c';\n</script>\n<button on:click={() => wipeConversation(id)}>Wipe</button>"
        assert _wipe_files({"frontend/src/Fixture.svelte": inline}) == ["frontend/src/Fixture.svelte"], (
            "a component that wipes from its markup alone is judged"
        )
        assert any("called from the markup" in f for f in found(inline)), "a wipe called from the markup is found"
        late = guarded.replace("await wipeConversation(id);", "await wipeConversation($activeConversationId);")
        assert any("reads $activeConversationId" in f for f in found(late)), (
            "a wipe that reads its target when confirmed, not when asked, is found"
        )

        sources = {path: read(path) for path in files(".svelte")}
        wiping = _wipe_files(sources)
        assert wiping, "the census finds the components that wipe"
        assert {
            f"{_SRC}/lib/components/chat/ChatControlBar.svelte",
            f"{_SRC}/lib/components/settings/HardeningPanel.svelte",
        } <= set(wiping), f"the census finds each known component that wipes: {wiping}"
        findings = {path: _wipe_findings(path, sources[path]) for path in wiping}
        findings = {path: found for path, found in findings.items() if found}
        assert not findings, "wipes not held behind a confirmation with a shown failure:\n  " + "\n  ".join(
            f"{path}: {'; '.join(found)}" for path, found in findings.items()
        )
        return

    if half == "dialog":
        text = _source(_CONFIRM_DIALOG, "no confirmation dialog exists yet")
        script = _script(text, _CONFIRM_DIALOG)
        markup = _markup(text)
        assert _imports(script, "Modal", "./Modal.svelte"), "the dialog is built on the ds Modal"
        assert [name for name, _, _ in _tags(markup, r"Modal")] == ["Modal"], "one Modal"
        assert _imports(script, "InlineError", "./InlineError.svelte") and re.search(
            r"<InlineError\b[^>]*\bmessage\s*=\s*\{\s*error\s*\}", markup
        ), "the dialog shows its error through the ds InlineError, an alert"
        for prop in ("open", "title", "message", "confirmLabel", "cancelLabel", "danger", "busy",
                     "error", "onConfirm", "onCancel"):
            assert re.search(r"export\s+let\s+" + prop + r"\b", script), f"the dialog takes {prop}"
        accept = _function_body(script, "accept") or ""
        assert re.search(r"\bonConfirm\s*\(", accept), "the confirming button runs onConfirm"
        dismiss = _function_body(script, "dismiss") or ""
        assert re.search(r"\bonCancel\s*\(", dismiss), "the other button, Escape and the backdrop cancel"
        assert re.search(r"onClose\s*=\s*\{\s*dismiss\s*\}", markup), "the Modal closes through dismiss"
        buttons = [body for _, body, _ in _tags(markup, r"Button")]
        assert any(re.search(r"on:click\s*=\s*\{\s*accept\s*\}", b) for b in buttons) and any(
            re.search(r"on:click\s*=\s*\{\s*dismiss\s*\}", b) for b in buttons
        ), "the dialog's two ds Buttons accept and dismiss"
        assert re.search(r"closeOnEsc\s*=\s*\{\s*!\s*busy\s*\}", markup) and re.search(
            r"closeOnBackdrop\s*=\s*\{\s*!\s*busy\s*\}", markup
        ), "while the action runs, Escape and the backdrop do not close the dialog"
        assert re.search(r"export\s*\{\s*default\s+as\s+ConfirmDialog\s*\}\s*from\s*['\"]\./ConfirmDialog\.svelte['\"]",
                         read(_DS_INDEX)), "the design system exports the dialog"
        assert re.search(r"closable\s*=\s*\{\s*!\s*busy\s*\}", markup), (
            "while the action runs, the Modal's close button waits too"
        )
        modal = _markup(read(f"{_SRC}/lib/ds/Modal.svelte"))
        close = [body for _, body, _ in _tags(modal, r"button") if "oo-modal-close" in body]
        assert close and re.search(r"disabled\s*=\s*\{\s*!\s*closable\s*\}", close[0]), (
            "the Modal disables its close button when it is not closable"
        )
        assert re.search(r"<form\b[^>]*\bon:submit\|preventDefault\s*=\s*\{\s*accept\s*\}", markup), (
            "the dialog's body is a form that confirms on submit, so a single field confirms with Enter"
        )
        return

    if half == "destructive":
        toasted = (
            "<script>\n\tlet pending = null;\n\tasync function runDelete() {\n"
            "\t\ttry { await deleteDocument(pending); } catch { toastError('Failed'); }\n\t}\n</script>\n"
            "<ConfirmDialog open={pending !== null} title=\"Delete?\" onConfirm={runDelete} onCancel={close} />"
        )
        assert _wipe_findings("frontend/src/Fixture.svelte", toasted, _DESTRUCTIVE, toast=True) == [], (
            "a failure toasted is shown"
        )
        direct = "<script>\n\tasync function drop(doc) { await deleteCollection(doc); }\n</script>\n"
        direct += "<button on:click={() => drop(doc)}>Delete</button>"
        assert any("not a ConfirmDialog's onConfirm" in f for f in _wipe_findings(
            "frontend/src/Fixture.svelte", direct, _DESTRUCTIVE, toast=True
        )), "a destructive action run from a button is found"

        sources = {path: read(path) for path in files(".svelte")}
        acting = _wipe_files(sources, _DESTRUCTIVE)
        assert {
            f"{_SRC}/lib/components/rag/DocumentManager.svelte",
            f"{_SRC}/lib/components/settings/KnowledgeBasePanel.svelte",
            f"{_SRC}/lib/components/settings/FineTunePanel.svelte",
            f"{_SRC}/lib/components/settings/RemoteAccessPanel.svelte",
        } <= set(acting), f"the census finds each known component with a destructive action: {acting}"
        findings = {path: _wipe_findings(path, sources[path], _DESTRUCTIVE, toast=True) for path in acting}
        findings = {path: found for path, found in findings.items() if found}
        assert not findings, "destructive actions not held behind a confirmation with a shown failure:\n  " + (
            "\n  ".join(f"{path}: {'; '.join(found)}" for path, found in findings.items())
        )
        return

    _source(_CONFIRM_DIALOG, "no confirmation dialog exists yet")
    shown = _dom(ssr().render(_CONFIRM_DIALOG, {**_CONFIRM_PROPS, "error": "The server refused (403)."}).html)
    dialogs = list(shown.iter("dialog"))
    assert len(dialogs) == 1, "the dialog is one native dialog element"
    assert "Wipe this conversation?" in "".join(h.text() for h in shown.iter("h2"))
    assert "Its messages are zeroed in memory." in dialogs[0].text()
    labels = [button.text().strip() for button in shown.iter("button")]
    assert "Wipe" in labels and "Keep it" in labels, f"both choices are buttons: {labels}"
    alerts = [e for e in shown.iter() if e.attrs.get("role") == "alert"]
    assert len(alerts) == 1 and "The server refused (403)." in alerts[0].text(), (
        "the failure is shown in an alert inside the dialog"
    )
    wipe = [b for b in shown.iter("button") if b.text().strip() == "Wipe"][0]
    assert wipe.attrs.get("data-variant") == "danger", "a destructive action reads as danger"

    quiet = _dom(ssr().render(_CONFIRM_DIALOG, {**_CONFIRM_PROPS, "error": None}).html)
    assert not [e for e in quiet.iter() if e.attrs.get("role") == "alert"], "no failure, no alert"

    busy = _dom(ssr().render(_CONFIRM_DIALOG, {**_CONFIRM_PROPS, "busy": True}).html)
    buttons = {b.text().strip(): b for b in busy.iter("button")}
    assert "disabled" in buttons["Keep it"].attrs, "while the action runs, cancelling waits"
    closers = [b for b in busy.iter("button") if b.attrs.get("aria-label") == "Close dialog"]
    assert len(closers) == 1 and "disabled" in closers[0].attrs, "while the action runs, the close button waits"
    assert buttons["Wipe"].attrs.get("aria-busy") == "true" and "disabled" in buttons["Wipe"].attrs, (
        "while the action runs, the confirming button says so and cannot be pressed again"
    )


# ---------------------------------------------------------------------------
# UX8 -- no native confirm or prompt
# ---------------------------------------------------------------------------
_NATIVE_DIALOG = re.compile(
    r"(?<![\w.$])(?:(?:window|globalThis|self)\??\.)?(?:con" + "firm|pro" + r"mpt)\("
)


def test_ux8_no_native_confirm_or_prompt_dialog():
    sample = (
        "if (!con" "firm('Delete?')) return; const n = window.pro" "mpt('Name');"
        " globalThis.con" "firm(x); this.con" "firm(y); onCon" "firm(z); dialog.pro" "mpt(w);"
    )
    assert len(_NATIVE_DIALOG.findall(sample)) == 3, (
        "the census reads the global dialogs in their spellings, and no method or other name"
    )
    counts = _count_in({path: read(path) for path in files(_SCRIPTS)},
                       lambda p, t: len(_NATIVE_DIALOG.findall(t)))
    assert not counts, f"native confirm or prompt dialogs: {counts}"


# ---------------------------------------------------------------------------
# UX9 -- no pictographic emoji in any spelling
# ---------------------------------------------------------------------------
_PICTOGRAPHS = ((0x1F000, 0x1FAFF), (0xFE0F, 0xFE0F))
_RAW_PICTOGRAPH = re.compile(
    "[" + "".join(f"{chr(low)}-{chr(high)}" for low, high in _PICTOGRAPHS) + "]"
)
_REFERENCE = re.compile(r"&#(?:[xX]([0-9a-fA-F]+)|([0-9]+));")
_ESCAPE = re.compile(re.escape(_BACKSLASH) + r"u(?:\{([0-9a-fA-F]+)\}|([0-9a-fA-F]{4}))")
_CSS_ESCAPE = re.compile(re.escape(_BACKSLASH) + r"([0-9a-fA-F]{1,6})")
_FROM_NUMBERS = re.compile(r"\bfrom(CodePoint|CharCode)\(([^()]*)\)")
_NUMBER = re.compile(r"\s*(0[xX][0-9a-fA-F]+|\d+)\s*")


def _code_points(kind, arguments):
    """The code points ``String.fromCodePoint`` or ``fromCharCode`` builds
    from its number literals (an argument that is no literal is skipped;
    a surrogate pair built by ``fromCharCode`` is joined)."""
    numbers = []
    for argument in arguments.split(","):
        match = _NUMBER.fullmatch(argument)
        if match:
            literal = match.group(1)
            numbers.append(int(literal, 16) if literal[:2] in ("0x", "0X") else int(literal))
    if kind == "CodePoint":
        return numbers
    points, i = [], 0
    while i < len(numbers):
        high = numbers[i]
        if 0xD800 <= high <= 0xDBFF and i + 1 < len(numbers) and 0xDC00 <= numbers[i + 1] <= 0xDFFF:
            points.append(0x10000 + ((high - 0xD800) << 10) + (numbers[i + 1] - 0xDC00))
            i += 2
            continue
        points.append(high)
        i += 1
    return points


def _pictographic(point):
    return any(low <= point <= high for low, high in _PICTOGRAPHS)


def _pictographs(path, text):
    """Pictographic code points, raw, as an HTML reference, a script escape
    (a braced code point, a four-digit one, or a surrogate pair), a CSS
    escape, or built from number literals by ``String.fromCodePoint`` or
    ``String.fromCharCode``. No named HTML reference decodes to a pictograph,
    so none is read."""
    found = len(_RAW_PICTOGRAPH.findall(text))
    for match in _REFERENCE.finditer(text):
        hexadecimal, decimal = match.groups()
        if _pictographic(int(hexadecimal, 16) if hexadecimal else int(decimal)):
            found += 1
    escapes = list(_ESCAPE.finditer(text))
    skip = set()
    for index, match in enumerate(escapes):
        if index in skip:
            continue
        braced, plain = match.groups()
        point = int(braced or plain, 16)
        if plain and 0xD800 <= point <= 0xDBFF and index + 1 < len(escapes):
            following = escapes[index + 1]
            low = following.group(2)
            if following.start() == match.end() and low and 0xDC00 <= int(low, 16) <= 0xDFFF:
                point = 0x10000 + ((point - 0xD800) << 10) + (int(low, 16) - 0xDC00)
                skip.add(index + 1)
        if _pictographic(point):
            found += 1
    for match in _CSS_ESCAPE.finditer(text):
        if match.start() > 0 and text[match.start() - 1] == _BACKSLASH:
            continue
        if text[match.start() + 1:match.start() + 2] in ("u", "U"):
            continue
        if _pictographic(int(match.group(1), 16)):
            found += 1
    for match in _FROM_NUMBERS.finditer(text):
        found += sum(1 for point in _code_points(match.group(1), match.group(2)) if _pictographic(point))
    return found


def test_ux9_no_pictographic_emoji_in_any_spelling():
    eye = chr(0x1F441)
    sample = (
        # Raw twice (the eye and the variation selector), two references, a
        # braced escape, a surrogate pair, a code point, a CSS escape: eight.
        f"<span>{eye}{chr(0xFE0F)}</span><span>&#128065;</span><span>&#x1F4C1;</span>\n"
        f"<script>const a = '{_BACKSLASH}u{{1F517}}'; const b = '{_BACKSLASH}uD83D{_BACKSLASH}uDD17';"
        " const c = String.fromCodePoint(0x1F517);</script>\n"
        f"<style>i::before {{ content: '{_BACKSLASH}1F517'; }}</style>\n"
    )
    assert _pictographs("frontend/src/fixture.svelte", sample) == 8, (
        f"each spelling is counted once: {_pictographs('frontend/src/fixture.svelte', sample)}"
    )
    outside = (
        f"<span>{chr(0x2713)}{chr(0x2192)}</span><span>&#10003;</span>"
        f"<script>const d = '{_BACKSLASH}u2014'; const e = '{_BACKSLASH}uD83D';</script>"
    )
    assert _pictographs("frontend/src/fixture.svelte", outside) == 0, (
        "a symbol glyph, an arrow or a lone surrogate is not a pictograph"
    )
    built = (
        "<script>const f = String.fromCodePoint(0x61, 0x1F517); "
        "const g = String.fromCharCode(0xD83D, 0xDD17); const h = String.fromCharCode(55357, 56599); "
        "const i = String.fromCharCode(0xD83D); const j = String.fromCodePoint(n);</script>"
    )
    assert _pictographs("frontend/src/fixture.svelte", built) == 3, (
        "every argument of fromCodePoint is read, a surrogate pair built by fromCharCode is joined, "
        f"and a lone surrogate or a variable is not a pictograph: {_pictographs('frontend/src/fixture.svelte', built)}"
    )
    counts = _count_in({path: read(path) for path in files(_EVERY)}, _pictographs)
    assert not counts, f"pictographic emoji: {counts}"


# ---------------------------------------------------------------------------
# UX10 -- one skip link, one main landmark per page
# ---------------------------------------------------------------------------
_SKIP_LINK = re.compile(r"""href\s*=\s*(["'])#main-content\1""")
_MAIN_ID = re.compile(r"""\bid\s*=\s*(["'])main-content\1""")


def _forwards_only(text):
    """A page that only forwards elsewhere on mount: its script calls
    ``goto(`` inside ``onMount``, and its markup mounts no component and
    shows no text."""
    script = _script(text, "+page.svelte")
    on_mount = _call_spans(script, "onMount")
    if not any("goto(" in script[low:high] for low, high in on_mount):
        return False
    markup = _markup(text)
    if re.search(r"<[A-Z]", markup):
        return False
    return not re.sub(r"<[^>]*>", "", markup).strip()


def _layouts(tree, page):
    """The layouts a page renders in, innermost first."""
    chain = []
    folder = posixpath.dirname(page)
    while True:
        layout = f"{folder}/+layout.svelte"
        if layout in tree.sources:
            chain.append(layout)
        if folder == f"{_SRC}/routes" or "/" not in folder:
            break
        folder = posixpath.dirname(folder)
    return chain


def _main_landmarks(tree):
    """``{page: (count, [files that carry one])}`` for every page route."""
    out = {}
    for page in sorted(path for path in tree.sources if path.endswith("/+page.svelte")):
        if _forwards_only(tree.sources[page]):
            continue
        rendered = set()
        for part in (page, *_layouts(tree, page)):
            rendered |= tree.closure(part)
        carriers = sorted(path for path in rendered if _MAIN_ID.search(tree.sources[path]))
        out[page] = (sum(len(_MAIN_ID.findall(tree.sources[path])) for path in carriers), carriers)
    return out


def test_ux10_one_skip_link_and_one_main_landmark_per_page():
    sample = Tree({
        "frontend/src/lib/Shell.svelte": "<main id=\"main-content\"><slot /></main>",
        "frontend/src/routes/+layout.svelte": "<a href=\"#main-content\">Skip</a><slot />",
        "frontend/src/routes/a/+layout.svelte": "<script>import Shell from '$lib/Shell.svelte';</script><Shell><slot /></Shell>",
        "frontend/src/routes/a/+page.svelte": "<p>one</p>",
        "frontend/src/routes/a/b/+page.svelte": "<div id='main-content'>two</div>",
        "frontend/src/routes/c/+page.svelte": "<p>none</p>",
        "frontend/src/routes/d/+page.svelte": (
            "<script>import { onMount } from 'svelte';\nonMount(() => { goto('/a'); });</script>\n<div></div>"
        ),
    })
    landmarks = {page.rsplit("/routes/", 1)[1]: count for page, (count, _) in _main_landmarks(sample).items()}
    assert landmarks == {"a/+page.svelte": 1, "a/b/+page.svelte": 2, "c/+page.svelte": 0}, (
        f"the census counts over the page, its layouts and what they mount, and "
        f"a forwarding page is not judged: {landmarks}"
    )

    sources = {path: read(path) for path in files(_EVERY)}
    skips = _count_in(sources, lambda p, t: len(_SKIP_LINK.findall(t)))
    assert sum(skips.values()) == 1, f"one skip link in the whole frontend: {skips}"

    tree = Tree(sources)
    pages = _main_landmarks(tree)
    assert len(pages) >= 8, f"the census judges the page routes: {sorted(pages)}"
    wrong = {page: found for page, found in pages.items() if found[0] != 1}
    assert not wrong, "pages without exactly one main-content landmark:\n  " + "\n  ".join(
        f"{page}: {count} ({', '.join(carriers) or 'none'})" for page, (count, carriers) in wrong.items()
    )


# ---------------------------------------------------------------------------
# UX11 -- smooth scrolling only where motion is allowed
# ---------------------------------------------------------------------------
_SMOOTH = re.compile(
    r"""behavior\s*:\s*(['"`])smooth\1|scroll-behavior\s*:\s*smooth\b"""
)
_BEHAVIOUR_OPTION = re.compile(r"(?<![\w-])behavior\s*:\s*([^,}\n]+)")


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_ux11_scrolling_is_smooth_only_where_motion_is_allowed(half):
    if half == "wiring":
        sample = (
            "el.scrollIntoView({ behavior: 'smooth' }); window.scrollTo({ top: 0, behavior: \"smooth\" });\n"
            "html { scroll-behavior: smooth; }\n"
        )
        assert len(_SMOOTH.findall(sample)) == 3, "the census reads smooth scrolling in its spellings"
        assert not _SMOOTH.findall("scroll-behavior: auto; behavior: 'auto'; smoothing: 1;")

        listed = files(_EVERY, exclude=(_MOTION,))
        sources = {path: read(path) for path in listed}
        smooth = _count_in(sources, lambda p, t: len(_SMOOTH.findall(t)))
        assert not smooth, f"literal smooth scrolling outside lib/motion.ts: {smooth}"

        options = {}
        for path, text in sources.items():
            if path.endswith(".css"):
                continue
            for match in _BEHAVIOUR_OPTION.finditer(text):
                options.setdefault(path, []).append(match.group(1).strip())
        assert options, "the census finds the scroll options that set a behaviour"
        for path, values in options.items():
            assert all(value == "scrollBehavior()" for value in values), (
                f"{path} takes every scroll behaviour from scrollBehavior(): {values}"
            )
            assert _imports(sources[path], "scrollBehavior", "$lib/motion"), (
                f"{path} imports scrollBehavior from $lib/motion"
            )

        reduced, full = _node("motion", ("OO_MOTION",), [])["classes"]
        preferences = read(_PREFERENCES)
        for name in (reduced, full):
            assert re.search(r"classList\.toggle\(\s*['\"]" + re.escape(name) + r"['\"]", preferences), (
                f"the motion preference sets the class motion.ts reads: {name}"
            )
        return

    table = [
        {"reducedByChoice": False, "fullByChoice": False, "reducedBySystem": False},
        {"reducedByChoice": False, "fullByChoice": False, "reducedBySystem": True},
        {"reducedByChoice": True, "fullByChoice": False, "reducedBySystem": False},
        {"reducedByChoice": False, "fullByChoice": True, "reducedBySystem": True},
        None,
    ]
    out = _node("motion", ("OO_MOTION",), table)
    assert out["behaviours"] == ["smooth", "auto", "auto", "smooth", "auto"], (
        "smooth only when neither the preference nor the system reduces motion, "
        f"the preference winning, and auto when it cannot tell: {out['behaviours']}"
    )
    assert out["read"] is None and out["withoutDocument"] == "auto", (
        "without a document, the environment is unknown and scrolling is not smooth"
    )


# ---------------------------------------------------------------------------
# UX12 -- no navigation by assignment to location
# ---------------------------------------------------------------------------
_LOCATION_ASSIGNED = re.compile(
    r"(?<![\w.$])(?:(?:window|document|globalThis|self)\.)?location"
    r"(?:\.(?:href|pathname|search))?\s*=(?!=)"
)


def test_ux12_no_navigation_by_assignment_to_location():
    sample = (
        "window.location.href = '/a'; location = '/b'; document.location.pathname = '/c';"
        " self.location.search = '?d';"
    )
    assert len(_LOCATION_ASSIGNED.findall(sample)) == 4, "the census reads each assignment"
    kept = (
        "if (location.href === x) go(); location.reload(); const url = location.href;"
        " state.location = y; location.hash = '#z';"
    )
    assert not _LOCATION_ASSIGNED.findall(kept), "a read, a reload or another object's field is not one"
    counts = _count_in({path: read(path) for path in files(_SCRIPTS)},
                       lambda p, t: len(_LOCATION_ASSIGNED.findall(t)))
    assert not counts, f"navigation by assignment to location: {counts}"


# ---------------------------------------------------------------------------
# UX13 -- every settings group in the catalog, and nowhere else
# ---------------------------------------------------------------------------
# The settings sections and the old tab ids, as the census recognises them;
# the node half holds the catalog equal to them, so the census cannot drift.
_SECTION_IDS = (
    "appearance", "account", "conversation", "models", "knowledge", "plugins",
    "performance", "network", "data",
)
_LEGACY_TABS = {
    "quick": "conversation", "presets": "conversation", "prompt": "conversation",
    "models": "models", "analytics": "performance", "performance": "performance",
    "fine-tune": "data", "knowledge": "knowledge", "plugins": "plugins", "backup": "data",
    "security": "account", "advanced": "performance",
}


def _settings_lists(path, text):
    """Declarations of a section list (more than half the section ids as
    ``id:`` keys) or of the old tab map (more than half its pairs)."""
    ids = {m.group(1) for m in re.finditer(r"\bid\s*:\s*['\"`]([\w-]+)['\"`]", text)} & set(_SECTION_IDS)
    pairs = {
        m.group(1) for m in re.finditer(r"['\"]?([\w-]+)['\"]?\s*:\s*['\"]([\w-]+)['\"]", text)
        if _LEGACY_TABS.get(m.group(1)) == m.group(2)
    }
    return int(len(ids) * 2 > len(_SECTION_IDS)) + int(len(pairs) * 2 > len(_LEGACY_TABS))


def _loader_keys(page):
    return re.findall(r"(\w+)\s*:\s*\(\)\s*=>\s*import\(", _script(page, _SETTINGS_PAGE))


def _inline_groups(tree, page):
    """``{group id: intro}`` of the groups the section introductions render
    inline, read from the components the settings page mounts for each."""
    markup = _markup(page)
    out = {}
    for match in re.finditer(r"intro\s*===\s*['\"](\w+)['\"]\s*\}\s*<([A-Z]\w*)", markup):
        component = tree.names[_SETTINGS_PAGE].get(match.group(2))
        assert component, f"the introduction {match.group(1)} names a component the page imports"
        for _, body, _ in _tags(_markup(tree.sources[component]), r"SettingsGroup"):
            ids = [value for name, value in _attributes(body) if name == "id"]
            assert ids and isinstance(ids[0], str), f"{component}: a SettingsGroup has a literal id"
            out[ids[0]] = match.group(1)
    return out


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_ux13_every_settings_group_is_in_the_catalog_and_nowhere_else(half):
    if half == "wiring":
        sample = (
            "const s = [{ id: 'appearance' }, { id: 'account' }, { id: 'conversation' },\n"
            "{ id: 'models' }, { id: 'knowledge' }];\n"
            "const m = { quick: 'conversation', presets: 'conversation', 'fine-tune': 'data',\n"
            "backup: 'data', security: 'account', advanced: 'performance', models: 'models' };\n"
        )
        assert _settings_lists("sample.ts", sample) == 2, "the census reads both kinds of list"
        assert _settings_lists("sample.ts", "tabs = [{ id: 'models' }, { id: 'network' }, { id: 'plugins' }]") == 0

        sources = {path: read(path) for path in files(_SCRIPTS, exclude=(_CATALOG,))}
        lists = _count_in(sources, _settings_lists)
        assert not lists, f"settings lists declared outside the catalog: {lists}"
        for path in (_SETTINGS_PAGE, _SECTION_LIST):
            assert re.search(r"from\s*['\"]\$lib/settings/catalog['\"]", sources[path]), (
                f"{path} reads the catalog"
            )
        return

    page = read(_SETTINGS_PAGE)
    tree = _real_tree()
    loaders = _loader_keys(page)
    inline = _inline_groups(tree, page)
    assert len(loaders) >= 40 and len(inline) >= 9, (
        f"the census reads the page's panels and inline groups: {len(loaders)}, {len(inline)}"
    )
    catalog = _node("catalog", ("OO_CATALOG",), ["models", "security", "nowhere", None, ""])
    sections = catalog["sections"]
    assert [section["id"] for section in sections] == list(_SECTION_IDS), (
        "the catalog's sections are the ones the census recognises"
    )
    assert catalog["legacy"] == _LEGACY_TABS, "the catalog's old tab map is the one the census recognises"
    assert catalog["resolved"] == ["models", "account", "appearance", "appearance", "appearance"], (
        f"a section id, an old tab id, and anything else: {catalog['resolved']}"
    )

    groups = [group for section in sections for group in section["groups"]]
    ids = [group["id"] for group in groups] + [group["id"] for group in catalog["inline"]]
    assert len(ids) == len(set(ids)), "every group id is unique"
    for group in groups + catalog["inline"]:
        assert group["title"] and group["description"], f"{group['id']} has a title and a description"

    panels = [group["panel"] for group in groups]
    assert sorted(panels) == sorted(loaders) and len(panels) == len(set(panels)), (
        "every lazy panel the page can load is one catalog group, and every group has a panel: "
        f"missing {sorted(set(loaders) - set(panels))}, unloadable {sorted(set(panels) - set(loaders))}"
    )

    intro_of = {section["id"]: section.get("intro") for section in sections}
    listed = {group["id"]: group["sectionId"] for group in catalog["inline"]}
    assert set(listed) == set(inline), (
        "every group an introduction renders is in the catalog, and none it does not: "
        f"missing {sorted(set(inline) - set(listed))}, phantom {sorted(set(listed) - set(inline))}"
    )
    for group_id, intro in inline.items():
        assert intro_of.get(listed[group_id]) == intro, (
            f"{group_id} is listed under the section whose introduction renders it"
        )


# ---------------------------------------------------------------------------
# UX15 -- no URL names a vendor's organisation
# ---------------------------------------------------------------------------
_VENDORS = ("anth" + "ropics", "anth" + "ropic", "open" + "ai")
_URL = re.compile(r"(?:\bhttps?:)?//(?=[^\s/])[^\s'\"`<>)]+", re.I)


def _vendor_urls(path, text):
    found = 0
    for match in _URL.finditer(text):
        parts = re.split(r"[/.?#:@=&-]+", match.group(0).lower())
        if any(vendor in parts for vendor in _VENDORS):
            found += 1
    return found


def test_ux15_no_url_names_a_vendor_organisation():
    host = "git" + "hub.com"
    sample = (
        f'<a href="https://{host}/{_VENDORS[0]}/project/issues">x</a>\n'
        f"const docs = 'https://docs.{_VENDORS[1]}.com/x'; const api = '//api.{_VENDORS[2]}.com';\n"
    )
    assert _vendor_urls("sample.svelte", sample) == 3, "the census reads a vendor in a host or a path"
    kept = f"placeholder=\"https://{host}/user/plugin-repo\" const a = 'https://example.org/{_VENDORS[0]}ish';"
    assert _vendor_urls("sample.svelte", kept) == 0, "another organisation, or a longer word, is not one"
    counts = _count_in({path: read(path) for path in files(_EVERY)}, _vendor_urls)
    assert not counts, f"URLs naming a vendor's organisation: {counts}"


# ---------------------------------------------------------------------------
# UX16 -- every selector a global shortcut queries is rendered
# ---------------------------------------------------------------------------
_QUERY = re.compile(
    r"\b(querySelector(?:All)?|getElementById|closest|matches)\(\s*(?:(['\"])(.*?)\2|(`[^`]*`)|([^)'\"`][^)]*))\s*\)"
)
_SIMPLE = re.compile(r"^([a-z][\w-]*)?((?:#[\w-]+|\.[\w-]+|\[[^\]]+\])*)$")


def _shortcut_files(tree):
    """The global shortcut handler, the layout that mounts it with its
    callbacks, and every file that listens for an event either dispatches."""
    handlers = [path for path in tree.sources if path.endswith("/KeyboardShortcuts.svelte")]
    mounting = sorted({where for handler in handlers for where, _, _ in tree.mounts.get(handler, [])})
    scope = set(handlers) | set(mounting)
    events = set()
    for path in scope:
        events.update(re.findall(r"new\s+CustomEvent\(\s*['\"]([\w-]+)['\"]", tree.sources[path]))
    for path, text in tree.sources.items():
        for event in events:
            if re.search(r"addEventListener\(\s*['\"]" + re.escape(event) + r"['\"]", text):
                scope.add(path)
    return sorted(scope)


def _selectors(text):
    """``[(kind, selector or None)]`` of every DOM query in ``text``; None
    when the selector is not a literal the census can read."""
    out = []
    for match in _QUERY.finditer(text):
        kind, literal = match.group(1), match.group(3)
        if literal is None or "${" in (match.group(4) or ""):
            out.append((kind, None))
        elif kind == "getElementById":
            out.append((kind, "#" + literal))
        else:
            out.append((kind, literal))
    return out


def _compound(selector):
    """A simple compound selector as ``(tag, [(attribute, value or True)])``,
    or None when it is not one (a list, a descendant, a pseudo-class)."""
    match = _SIMPLE.match(selector.strip())
    if not match or not selector.strip():
        return None
    tag, rest = match.group(1), match.group(2)
    needs = []
    for part in re.findall(r"#[\w-]+|\.[\w-]+|\[[^\]]+\]", rest):
        if part[0] == "#":
            needs.append(("id", part[1:]))
        elif part[0] == ".":
            needs.append(("class", part[1:]))
        else:
            inner = re.fullmatch(r"\[\s*([\w:-]+)\s*(?:=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\]\s]+)))?\s*\]", part)
            if not inner:
                return None
            value = next((v for v in inner.groups()[1:] if v is not None), True)
            needs.append((inner.group(1), value))
    return tag, needs


def _rendered_by(tree, selector, candidates):
    """The files among ``candidates`` whose markup renders an element the
    compound selector matches."""
    parsed = _compound(selector)
    if parsed is None:
        return None
    tag, needs = parsed
    found = []
    for path in candidates:
        text = tree.sources[path]
        markup = _markup(text) if path.endswith(".svelte") else text
        for name, body, _ in _tags(markup, r"[a-z][\w-]*"):
            if tag and name != tag:
                continue
            attributes = dict(_attributes(body))
            ok = True
            for attribute, value in needs:
                have = attributes.get(attribute)
                if attribute == "class" and isinstance(have, str):
                    ok = ok and value in have.split()
                elif value is True:
                    ok = ok and attribute in attributes
                else:
                    ok = ok and have == value
            if ok:
                found.append(path)
                break
    return found


def _rendering(tree):
    """The files that render in the app: routes, layouts, the page shell,
    and every component mounted from them."""
    roots = [path for path in tree.sources if "/routes/" in path and path.endswith(".svelte")]
    rendered = set()
    for root in roots:
        rendered |= tree.closure(root)
    rendered |= {path for path in tree.sources if path.endswith("app.html")}
    return sorted(rendered)


def test_ux16_every_selector_a_global_shortcut_queries_is_rendered():
    sample = Tree({
        "frontend/src/routes/+layout.svelte": (
            "<script>import KeyboardShortcuts from '$lib/KeyboardShortcuts.svelte';\n"
            "import Side from '$lib/Side.svelte';\n"
            "function focus() { document.querySelector('input[placeholder=\"Search...\"]'); "
            "document.querySelector('[data-oo-search]'); document.getElementById('announce'); }\n"
            "</script>\n<KeyboardShortcuts onFocus={focus} /><Side /><slot />"
        ),
        "frontend/src/lib/KeyboardShortcuts.svelte": "<script>export let onFocus;</script>",
        "frontend/src/lib/Side.svelte": (
            "<input placeholder=\"Search... (Ctrl+K)\" data-oo-search /><div id=\"announce\"></div>"
        ),
        "frontend/src/lib/Unmounted.svelte": "<input placeholder=\"Search...\" />",
    })
    scope = _shortcut_files(sample)
    assert scope == ["frontend/src/lib/KeyboardShortcuts.svelte", "frontend/src/routes/+layout.svelte"]
    rendering = _rendering(sample)
    matched = {
        selector: bool(_rendered_by(sample, selector, rendering))
        for path in scope for _, selector in _selectors(sample.sources[path])
    }
    assert matched == {
        'input[placeholder="Search..."]': False, "[data-oo-search]": True, "#announce": True,
    }, f"a selector matches only an element a mounted component renders: {matched}"

    tree = _real_tree()
    scope = _shortcut_files(tree)
    assert _ROOT_LAYOUT in scope and _SHORTCUTS in scope, f"the census finds the shortcut handlers: {scope}"
    rendering = _rendering(tree)
    queried = [(path, kind, selector) for path in scope for kind, selector in _selectors(tree.sources[path])]
    assert queried, "the census finds the selectors the shortcut handlers query"
    unread = [(path, kind) for path, kind, selector in queried if selector is None]
    assert not unread, f"selectors the census cannot read: {unread}"
    unmatched = []
    for path, kind, selector in queried:
        found = _rendered_by(tree, selector, rendering)
        if not found:
            unmatched.append(f"{path}: {kind}({selector!r})" + (" is not a simple selector" if found is None else ""))
    assert not unmatched, "selectors a shortcut queries that nothing rendered matches:\n  " + "\n  ".join(unmatched)


# ---------------------------------------------------------------------------
# UX17 -- every settings group in the catalog, rendered by the settings hub
# ---------------------------------------------------------------------------
_HUB = f"{_SRC}/lib/components/settings/SettingsHub.svelte"
_LAZY = re.compile(r"(\w+)\s*:\s*\(\)\s*=>\s*import\(")


def _hub_inline_groups(tree):
    """``{group id: intro}`` of the groups the hub's section introductions
    render inline, read from the components the hub mounts for each."""
    out = {}
    for match in re.finditer(r"intro\s*===\s*['\"](\w+)['\"]\s*\}\s*<([A-Z]\w*)", _markup(tree.sources[_HUB])):
        component = tree.names[_HUB].get(match.group(2))
        assert component, f"the introduction {match.group(1)} names a component the hub imports"
        for _, body, _ in _tags(_markup(tree.sources[component]), r"SettingsGroup"):
            ids = [value for name, value in _attributes(body) if name == "id"]
            assert ids and isinstance(ids[0], str), f"{component}: a SettingsGroup has a literal id"
            out[ids[0]] = match.group(1)
    return out


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_ux17_every_settings_group_is_in_the_catalog_and_the_hub_renders_it(half):
    if half == "wiring":
        sample = (
            "const s = [{ id: 'appearance' }, { id: 'account' }, { id: 'conversation' },\n"
            "{ id: 'models' }, { id: 'knowledge' }];\n"
            "const m = { quick: 'conversation', presets: 'conversation', 'fine-tune': 'data',\n"
            "backup: 'data', security: 'account', advanced: 'performance', models: 'models' };\n"
        )
        assert _settings_lists("sample.ts", sample) == 2, "the census reads both kinds of list"
        assert _settings_lists("sample.ts", "tabs = [{ id: 'models' }, { id: 'network' }, { id: 'plugins' }]") == 0
        assert len(_LAZY.findall("const l = { A: () => import('./A.svelte'), B: () => import('./B.svelte') };")) == 2

        sources = {path: read(path) for path in files(_SCRIPTS, exclude=(_CATALOG,))}
        lists = _count_in(sources, _settings_lists)
        assert not lists, f"settings lists declared outside the catalog: {lists}"
        assert re.search(r"from\s*['\"]\$lib/settings/catalog['\"]", sources[_HUB]), "the hub reads the catalog"
        loaders = {
            path: count for path, text in sources.items()
            if (count := len(_LAZY.findall(_script(text, path)))) >= 10
        }
        assert list(loaders) == [_HUB], f"the hub alone loads the settings panels: {loaders}"
        return

    tree = _real_tree()
    loaders = _LAZY.findall(_script(tree.sources[_HUB], _HUB))
    inline = _hub_inline_groups(tree)
    assert len(loaders) >= 40 and len(inline) >= 9, (
        f"the census reads the hub's panels and inline groups: {len(loaders)}, {len(inline)}"
    )
    catalog = _node("catalog", ("OO_CATALOG",), [])
    sections = catalog["sections"]
    assert [section["id"] for section in sections] == list(_SECTION_IDS), (
        "the catalog's sections are the ones the census recognises"
    )
    assert catalog["legacy"] == _LEGACY_TABS, "the catalog's old tab map is the one the census recognises"

    groups = [group for section in sections for group in section["groups"]]
    ids = [group["id"] for group in groups] + [group["id"] for group in catalog["inline"]]
    assert len(ids) == len(set(ids)), "every group id is unique"
    for group in groups + catalog["inline"]:
        assert group["title"] and group["description"], f"{group['id']} has a title and a description"

    panels = [group["panel"] for group in groups]
    assert sorted(panels) == sorted(loaders) and len(panels) == len(set(panels)), (
        "every lazy panel the hub can load is one catalog group, and every group has a panel: "
        f"missing {sorted(set(loaders) - set(panels))}, unloadable {sorted(set(panels) - set(loaders))}"
    )

    intro_of = {section["id"]: section.get("intro") for section in sections}
    listed = {group["id"]: group["sectionId"] for group in catalog["inline"]}
    assert set(listed) == set(inline), (
        "every group an introduction renders is in the catalog, and none it does not: "
        f"missing {sorted(set(inline) - set(listed))}, phantom {sorted(set(listed) - set(inline))}"
    )
    for group_id, intro in inline.items():
        assert intro_of.get(listed[group_id]) == intro, (
            f"{group_id} is listed under the section whose introduction renders it"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
