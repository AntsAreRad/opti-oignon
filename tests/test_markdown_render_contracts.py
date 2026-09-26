#!/usr/bin/env python3
"""Contracts for the reply renderer: marked's lexer into a closed node tree.

An assistant reply is markdown. marked's lexer (never its parser or its
renderer) turns the source into tokens; ``frontend/src/lib/markdown/tree.ts``
maps every token to a node of a closed set, and Svelte components render
those nodes through text interpolation only. The pure layer is five modules
under ``frontend/src/lib/markdown``:

  * ``tree.ts`` -- the lexer (gfm, breaks) and the token-to-node mapping;
  * ``entities.ts`` -- named and numeric references, through a fixed table;
  * ``collapse.ts`` -- a long finished reply collapsed by whole blocks;
  * ``frame.ts`` -- at most one lex per animation frame while streaming;
  * ``highlight.ts`` -- a two-role highlighter (keywords and defined names).

They import ``marked`` (``tree.ts`` only) or nothing: what they need from one
another is injected, so each runs under Node's type stripping through
``tests/_frontend.run_ts`` with no bundler. The node halves below run them
over real input and assert, in Python, on what they return; the driver only
calls the module and prints the result.

A contract over a pure module carries a wiring half: the component that uses
the module imports it, calls it, and holds no competing logic. The wiring
halves read the components under
``frontend/src/lib/components/chat/markdown`` as text.

The components are ``Markdown.svelte`` (lexing through ``frame.ts``, the
tree, collapse), ``MarkdownNode.svelte`` (one node, recursively),
``CodeBlock.svelte`` (a code block, its label and its Copy) and
``MarkdownTable.svelte``; ``ChatMessage.svelte`` renders an assistant reply
through ``Markdown`` and a user message as plain text. The ssr halves render
them through ``tests/_frontend.ssr()`` and read what the template emits
when compiled for the server, parsed as HTML. The app is client-rendered
only: an ssr half proves the template's output, not what a browser does
with it.

  * MK18 -- ``run_ts`` passes a driver only on its own ``PASS`` line and a
    zero exit; it refuses a driver that exits 0 without that line, one that
    prints it and exits non-zero, one that passes another clause, and a
    module under contract that is absent.
  * MK17 -- ``ssr()`` compiles components for the server and renders them:
    a planted component that uses ``$lib`` and the three ``$app`` stubs
    renders its exact, escaped text; a planted component that renders the
    wrong text is caught; a component that throws is refused, never read as
    empty; the server runs on a copy under ``$TMPDIR`` and writes nothing in
    the tree.
  * MK2 -- ``marked`` is imported by ``tree.ts`` alone, and only as
    ``Lexer``; the census that reads every spelling of an import counts each
    one in its sample.
  * MK3 -- raw HTML, block and inline, becomes text with its exact
    characters, and the text inside a raw HTML element keeps its own; a
    block of it is a raw paragraph, shown with its line breaks and its
    indentation.
  * MK4 -- only an absolute ``http:``, ``https:`` or ``mailto:`` destination
    becomes a link; every other one is its source text.
  * MK5 -- an image becomes the text "Image: alt (url)"; no node can load it.
  * MK6 -- a code block is open only while streaming, as the last block, with
    its fence unterminated; its copy text is its code; its label is the first
    word of the info string when that word is at most 20 characters of
    ``[A-Za-z0-9+#.-]``, and the figure is named by that label alone.
  * MK7 -- characters round-trip with no double decoding; the fixed table of
    named references and every numeric reference decode in text, an unknown
    name stays literal; code spans and code blocks keep references literal.
  * MK8 -- a heading of depth d is a heading of level min(6, d + 2).
  * MK10 -- the node kinds form a closed set, and a token of an unknown type
    renders its raw source as text.
  * MK11 -- a single newline inside a paragraph is a line break.
  * MK12 -- collapse keeps whole top-level blocks within a budget, and a
    hidden block is not in the output.
  * MK13 -- a hundred updates inside one frame cause one lex, and a reply
    lexes through the frame's push while it streams and at once only
    otherwise.
  * MK14 -- highlighter spans concatenate to the input; the only roles are
    plain, keyword and name, and a name is only one a definition introduces;
    an unknown language is one plain span; strings, comments and members
    (a word after a dot) stay plain; the code block renders those spans,
    each in its role's class.
  * MK1 -- no raw HTML sink anywhere in ``frontend/src`` (``{@html}``, with
    or without space after the brace, ``innerHTML``, ``outerHTML``,
    ``insertAdjacentHTML``, ``Function(`` and ``eval(``, bare or on an
    object); the other calls that write markup (``document.write``,
    ``createContextualFragment``, ``srcdoc``, the unsafe HTML setters) stand
    only where they stand today, the recovery codes' print window; and the
    lint keeps ``plugin:svelte/recommended``, whose
    ``svelte/no-at-html-tags`` is an error, with no disable naming it.
  * MK9 -- a table is a focusable, labelled region; its header cells are
    column headers, and each cell carries its column's alignment.
  * MK15 -- thirty hostile replies, finished or streaming, emit no element
    that loads or runs (script, img, iframe, object), no event handler
    attribute, and no URL attribute with a script or data scheme.
  * MK16 -- a code block's Copy writes the block's code and announces
    "Copied" through a status region once the write resolves, and "Copy
    failed" when it is refused; the message keeps its own Copy of the whole
    raw reply, and copies no code block itself.
  * MK19 -- ChatMessage renders an assistant reply through Markdown, once
    (no raw markup of it is shown beside), and every other message, a user's
    or one of any other role, as plain text.
  * MK20 -- the renderer is total: a reply marked cannot lex, one that nests
    past the tree's depth cap, or a missing source renders, and every
    character of the source is shown (as nested elements or as text).
  * MK21 -- lexing a reply costs a bounded time: a reply whose estimated
    lexing work is over the limit is not lexed but shown as plain text; the
    repository's own markdown is lexed; what the estimate lets through
    lexes fast; a reply whose lex was slow once is not lexed again, for the
    rest of its stream or when it is shown again.
  * MK22 -- building the tree is linear in a run of newlines: a raw block or
    a streaming fence of a hundred thousand newlines builds at once.
  * MK23 -- while a reply streams, its caret is a 2 px bar drawn after the
    reply's last character, inside its last block, and blinks through the
    class that reduced motion stills; there is none otherwise.
  * MK24 -- plain text collapses by whole lines: a long user message, or a
    reply shown as plain text, shows its first lines and says how many it
    hides; a reply's first block is never cut, so a reply that is one long
    block is shown whole.
  * MK25 -- the renderer's type never computes under 12 px: every font size
    in its components is a token, an absolute size of at least 12 px, or a
    relative size floored at 12 px.
  * MK26 -- a task item's box sits beside its text, in a row, whether the
    list is tight or loose; a thematic break is drawn as a small mark of
    tone, never an empty gap.

Local-only (the public distribution ships no tests). Needs Node >= 22.6 and
``frontend/node_modules`` (``marked``, and vite for ``ssr()``); without them
the helpers raise, and so do the contracts.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import REPO, files, read, run_ts, ssr  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file. MK17 starts the session's server-rendering process.
BUDGET_S = {
    "test_mk18_run_ts_passes_a_driver_only_on_its_own_pass_line_and_a_zero_exit": 2.0,
    "test_mk17_ssr_renders_a_component_and_refuses_what_it_cannot_render": 6.0,
    "test_mk2_marked_is_imported_by_the_tree_alone_and_only_as_its_lexer": 1.0,
    "test_mk3_raw_html_becomes_text_with_its_exact_characters[node]": 2.0,
    "test_mk3_raw_html_becomes_text_with_its_exact_characters[wiring]": 1.0,
    "test_mk4_only_absolute_http_https_and_mailto_destinations_become_links[node]": 2.0,
    "test_mk4_only_absolute_http_https_and_mailto_destinations_become_links[wiring]": 1.0,
    "test_mk5_an_image_becomes_text_and_nothing_can_load_it[node]": 2.0,
    "test_mk5_an_image_becomes_text_and_nothing_can_load_it[wiring]": 1.0,
    "test_mk6_a_code_block_is_open_only_while_its_last_fence_streams[node]": 2.0,
    "test_mk6_a_code_block_is_open_only_while_its_last_fence_streams[wiring]": 1.0,
    "test_mk7_references_decode_once_in_text_and_never_in_code[node]": 2.0,
    "test_mk7_references_decode_once_in_text_and_never_in_code[wiring]": 1.0,
    "test_mk8_a_heading_of_depth_d_has_level_min_6_d_plus_2[node]": 2.0,
    "test_mk8_a_heading_of_depth_d_has_level_min_6_d_plus_2[wiring]": 1.0,
    "test_mk10_the_node_kinds_are_a_closed_set_and_an_unknown_token_is_its_raw_text[node]": 2.0,
    "test_mk10_the_node_kinds_are_a_closed_set_and_an_unknown_token_is_its_raw_text[wiring]": 2.0,
    "test_mk11_a_single_newline_in_a_paragraph_is_a_line_break[node]": 2.0,
    "test_mk11_a_single_newline_in_a_paragraph_is_a_line_break[wiring]": 1.0,
    "test_mk12_collapse_keeps_whole_top_level_blocks_and_drops_the_hidden_ones[node]": 2.0,
    "test_mk12_collapse_keeps_whole_top_level_blocks_and_drops_the_hidden_ones[wiring]": 1.0,
    "test_mk13_a_hundred_updates_inside_one_frame_cause_one_lex[node]": 2.0,
    "test_mk13_a_hundred_updates_inside_one_frame_cause_one_lex[wiring]": 1.0,
    "test_mk14_highlighting_keeps_every_character_in_two_roles_and_plain[node]": 2.0,
    "test_mk14_highlighting_keeps_every_character_in_two_roles_and_plain[wiring]": 1.0,
    "test_mk3_raw_html_becomes_text_with_its_exact_characters[ssr]": 1.0,
    "test_mk5_an_image_becomes_text_and_nothing_can_load_it[ssr]": 1.0,
    "test_mk6_a_code_block_is_open_only_while_its_last_fence_streams[ssr]": 1.0,
    "test_mk7_references_decode_once_in_text_and_never_in_code[ssr]": 1.0,
    "test_mk8_a_heading_of_depth_d_has_level_min_6_d_plus_2[ssr]": 1.0,
    "test_mk12_collapse_keeps_whole_top_level_blocks_and_drops_the_hidden_ones[ssr]": 1.0,
    "test_mk1_no_raw_html_sink_anywhere_and_the_lint_keeps_its_rule": 1.0,
    "test_mk9_a_table_is_a_labelled_focusable_region_with_column_headers_and_alignment": 1.0,
    "test_mk15_a_hostile_corpus_emits_nothing_that_loads_or_runs": 1.0,
    "test_mk16_copy_writes_the_block_code_and_announces_copied_through_a_status_region": 1.0,
    "test_mk19_chat_message_renders_a_reply_through_markdown_and_a_user_message_as_plain_text": 1.0,
    "test_mk14_highlighting_keeps_every_character_in_two_roles_and_plain[ssr]": 1.0,
    "test_mk20_the_renderer_is_total_and_shows_every_character[node]": 2.0,
    "test_mk20_the_renderer_is_total_and_shows_every_character[wiring]": 1.0,
    "test_mk20_the_renderer_is_total_and_shows_every_character[ssr]": 1.0,
    "test_mk21_lexing_a_reply_costs_a_bounded_time[node]": 2.0,
    "test_mk21_lexing_a_reply_costs_a_bounded_time[boundary]": 2.0,
    "test_mk21_lexing_a_reply_costs_a_bounded_time[guard]": 2.0,
    "test_mk21_lexing_a_reply_costs_a_bounded_time[wiring]": 1.0,
    "test_mk21_lexing_a_reply_costs_a_bounded_time[ssr]": 1.0,
    "test_mk22_building_the_tree_is_linear_in_a_run_of_newlines": 2.0,
    "test_mk23_the_streaming_caret_is_drawn_after_the_last_character[ssr]": 1.0,
    "test_mk23_the_streaming_caret_is_drawn_after_the_last_character[wiring]": 1.0,
    "test_mk24_plain_text_collapses_by_whole_lines_and_a_first_block_is_never_cut[node]": 2.0,
    "test_mk24_plain_text_collapses_by_whole_lines_and_a_first_block_is_never_cut[wiring]": 1.0,
    "test_mk24_plain_text_collapses_by_whole_lines_and_a_first_block_is_never_cut[ssr]": 1.0,
    "test_mk25_the_renderers_type_never_computes_under_twelve_pixels": 1.0,
    "test_mk26_a_task_box_sits_beside_its_text_and_a_rule_is_drawn": 1.0,
}

_LIB = "frontend/src/lib/markdown"
_TREE = f"{_LIB}/tree.ts"
_ENTITIES = f"{_LIB}/entities.ts"
_COLLAPSE = f"{_LIB}/collapse.ts"
_FRAME = f"{_LIB}/frame.ts"
_HIGHLIGHT = f"{_LIB}/highlight.ts"

_COMPONENTS = "frontend/src/lib/components/chat/markdown"
_MARKDOWN = f"{_COMPONENTS}/Markdown.svelte"
_NODE = f"{_COMPONENTS}/MarkdownNode.svelte"
_CODE_BLOCK = f"{_COMPONENTS}/CodeBlock.svelte"
_TABLE = f"{_COMPONENTS}/MarkdownTable.svelte"
_PLAIN = f"{_COMPONENTS}/PlainText.svelte"
_CARET = f"{_COMPONENTS}/Caret.svelte"
_RENDERERS = (_MARKDOWN, _NODE, _CODE_BLOCK, _TABLE, _PLAIN, _CARET)
_CHAT_MESSAGE = "frontend/src/lib/components/chat/ChatMessage.svelte"

_MODULES = {
    "OO_TREE": _TREE,
    "OO_ENTITIES": _ENTITIES,
    "OO_COLLAPSE": _COLLAPSE,
    "OO_FRAME": _FRAME,
    "OO_HIGHLIGHT": _HIGHLIGHT,
}

# The driver calls the modules and prints what they return; every property
# is asserted in Python. Only the frame clock is faked: it is the seam
# frame.ts schedules through.
_DRIVER = r"""
import fs from 'node:fs';

const load = async (name) => (process.env[name] ? await import(process.env[name]) : null);
const tree = await load('OO_TREE');
const entities = await load('OO_ENTITIES');
const collapse = await load('OO_COLLAPSE');
const frame = await load('OO_FRAME');
const highlight = await load('OO_HIGHLIGHT');
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT || 'null');

const build = (source, streaming) =>
    tree.toTree(tree.lexMarkdown(source), { streaming, decode: entities.decodeEntities });

function fakeFrames() {
    const pending = new Map();
    let next = 0;
    return {
        seam: {
            request: (callback) => { next += 1; pending.set(next, callback); return next; },
            cancel: (handle) => { pending.delete(handle); },
        },
        scheduled: () => pending.size,
        run: () => {
            const due = [...pending.values()];
            pending.clear();
            due.forEach((callback) => callback());
            return due.length;
        },
    };
}

function frameRun() {
    const clock = fakeFrames();
    const lexed = [];
    const delivered = [];
    const lexer = frame.createFrameLexer(
        (source) => { lexed.push(source); return 'tokens of ' + source; },
        (result) => { delivered.push(result); },
        clock.seam,
    );
    const report = {};
    for (let i = 1; i <= 100; i += 1) lexer.push('update ' + i);
    report.beforeFrame = { lexed: lexed.length, scheduled: clock.scheduled() };
    report.firstRun = clock.run();
    report.firstFrame = { lexed: [...lexed], delivered: [...delivered] };
    for (let i = 101; i <= 150; i += 1) lexer.push('update ' + i);
    report.secondRun = clock.run();
    report.secondFrame = { lexed: [...lexed], delivered: [...delivered] };
    lexer.push('update 150');
    report.unchanged = { scheduled: clock.scheduled() };
    lexer.push('pending');
    report.now = lexer.now('final');
    report.afterNow = { ran: clock.run(), lexed: [...lexed], delivered: [...delivered] };
    lexer.push('dropped');
    lexer.cancel();
    report.afterCancel = { ran: clock.run(), lexed: [...lexed] };
    const plain = [];
    const unscheduled = frame.createFrameLexer((source) => { plain.push(source); return source; }, () => {});
    unscheduled.push('first');
    unscheduled.push('second');
    report.withoutFrames = plain;
    return report;
}

// The slow-lex guard over a clock that only moves when a lex runs: 'slow'
// and the sources named in slowCost take 150 ms to lex, the rest 5 ms.
function guardRun() {
    let clock = 0;
    const lexed = [];
    const slowCost = new Set(['slow']);
    const lex = (source) => {
        lexed.push(source);
        clock += slowCost.has(source) ? 150 : 5;
        return 'tokens of ' + source;
    };
    const memory = new Set();
    const options = { now: () => clock, slowMs: 100, memory };
    const report = {};
    const stream = frame.guardSlowLex(lex, options);
    report.fast = stream('fast');
    report.slow = stream('slow');
    report.grown = stream('slow and more');
    report.lexedInStream = [...lexed];
    report.memoryAfterStream = [...memory];
    report.remembered = frame.guardSlowLex(lex, options)('slow and more');
    report.fresh = frame.guardSlowLex(lex, options)('fresh');
    report.lexed = [...lexed];
    for (let i = 0; i < 20; i += 1) {
        slowCost.add('s' + i);
        frame.guardSlowLex(lex, options)('s' + i);
    }
    report.memory = [...memory];
    report.remembers = frame.SLOW_REMEMBERED;
    const estimated = frame.guardSlowLex(
        (source) => { clock += 1; return source === 'estimated' ? null : 'tokens of ' + source; },
        { now: () => clock, slowMs: 100, memory: new Set() },
    );
    report.estimated = estimated('estimated');
    report.afterEstimated = estimated('next');
    report.defaults = frame.guardSlowLex((source) => 'tokens of ' + source)('x');
    return report;
}

// A source built from a spec: a motif repeated n times between a prefix and
// a suffix, a run nested n deep, or n reference definitions used n times.
function generate(spec, n) {
    if (spec.kind === 'nest') return spec.open.repeat(n) + spec.middle + spec.close.repeat(n);
    if (spec.kind === 'refs') {
        const defs = Array.from({ length: n }, (_, i) => '[r' + i + ']: http://x.test/' + i);
        const uses = Array.from({ length: n }, (_, i) => '[r' + i + ']');
        return defs.join('\n') + '\n\n' + uses.join(' ');
    }
    return (spec.prefix || '') + spec.motif.repeat(n) + (spec.suffix || '');
}

const timed = (fn) => {
    const start = performance.now();
    const value = fn();
    return { value, ms: performance.now() - start };
};

function maxDepth(nodes, depth = 1) {
    let deepest = nodes.length > 0 ? depth : depth - 1;
    for (const node of nodes) {
        const children = [
            ...(node.children || []),
            ...(node.header || []).flat(),
            ...(node.rows || []).flat(2),
        ];
        deepest = Math.max(deepest, maxDepth(children, depth + 1));
    }
    return deepest;
}

function nestedTokens(depth) {
    let token = { type: 'paragraph', raw: 'x', tokens: [{ type: 'text', raw: 'x', text: 'x' }] };
    for (let level = 1; level <= depth; level += 1) {
        token = { type: 'blockquote', raw: '>'.repeat(level) + ' x', tokens: [token] };
    }
    return [token];
}

// An input item: a source as it is, a file to read, or a spec to build.
function sourceOf(item) {
    if (item === null || typeof item === 'string') return item;
    if (item.file) return fs.readFileSync(item.file, 'utf8');
    if (item.spec) return generate(item.spec, item.n);
    return item.source;
}

const clauses = {
    guarded: () => input.map((item) => {
        const source = sourceOf(item);
        const run = timed(() => tree.lexMarkdown(source));
        return { plain: run.value === null, ms: run.ms };
    }),
    limits: () => ({ work: tree.LEX_WORK_LIMIT, depth: tree.MAX_DEPTH }),
    boundary: () => input.map((spec) => {
        const accepted = (n) => tree.lexWork(generate(spec, n)) <= tree.LEX_WORK_LIMIT;
        let low = 1;
        let high = 2;
        while (accepted(high) && high < spec.cap) { low = high; high *= 2; }
        if (accepted(high)) return { n: high, capped: true };
        while (high - low > 1) {
            const middle = Math.floor((low + high) / 2);
            if (accepted(middle)) low = middle; else high = middle;
        }
        const run = timed(() => tree.lexMarkdown(generate(spec, low)));
        return {
            n: low,
            length: generate(spec, low).length,
            ms: run.ms,
            lexed: Array.isArray(run.value),
            above: tree.lexMarkdown(generate(spec, low + 1)) === null,
        };
    }),
    guard: guardRun,
    depths: () => input.map((source) => {
        const tokens = tree.lexMarkdown(source);
        if (tokens === null) return { plain: true };
        const nodes = tree.toTree(tokens, { streaming: false, decode: entities.decodeEntities });
        return { plain: false, depth: maxDepth(nodes), nodes };
    }),
    capped: () => input.map((depth) => {
        const nodes = tree.toTree(nestedTokens(depth), { streaming: false, decode: entities.decodeEntities });
        return { depth: maxDepth(nodes), nodes };
    }),
    thrown: () => input.map((item) => {
        const source = sourceOf(item);
        try {
            return { value: tree.lexMarkdown(source) === null ? 'plain' : 'tokens' };
        } catch (error) {
            return { value: 'threw', error: String(error) };
        }
    }),
    treeTime: () => input.map(([item, streaming]) => {
        const tokens = tree.lexMarkdown(sourceOf(item));
        if (tokens === null) return { plain: true };
        const run = timed(() => tree.toTree(tokens, { streaming, decode: entities.decodeEntities }));
        return { plain: false, ms: run.ms, blocks: run.value.length };
    }),
    lines: () => input.map(([text, options]) => collapse.collapseLines(text, options)),
    trees: () => input.map(([source, streaming]) => build(source, streaming)),
    injected: () => input.map((tokens) =>
        tree.toTree(tokens, { streaming: false, decode: entities.decodeEntities })),
    kinds: () => [...tree.NODE_KINDS],
    decode: () => input.map((text) => entities.decodeEntities(text)),
    collapse: () => input.map(([source, options]) => {
        const blocks = build(source, false);
        const out = collapse.collapseBlocks(blocks, options);
        return {
            blocks,
            weights: blocks.map((block) => collapse.blockWeight(block)),
            indices: out.shown.map((block) => blocks.indexOf(block)),
            hidden: out.hidden,
            collapsible: out.collapsible,
        };
    }),
    frame: frameRun,
    highlight: () => input.map(([code, lang]) => highlight.highlight(code, lang)),
    roles: () => [...highlight.HIGHLIGHT_ROLES],
};

if (!(clause in clauses)) {
    console.log('FAIL unknown clause: ' + clause);
    process.exit(1);
}
console.log('RESULT ' + JSON.stringify(clauses[clause]()));
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


def _trees(cases):
    """``[(source, streaming)]`` -> the tree of each, lexed and decoded."""
    return _node("trees", ("OO_TREE", "OO_ENTITIES"), [list(case) for case in cases])


def _tree(source, streaming=False):
    return _trees([(source, streaming)])[0]


def _walk(nodes):
    """Every node of a tree, depth first, table cells included."""
    for node in nodes:
        yield node
        yield from _walk(node.get("children", []))
        for cell in node.get("header", []):
            yield from _walk(cell)
        for row in node.get("rows", []):
            for cell in row:
                yield from _walk(cell)


def _text(nodes):
    """The text a run of nodes shows: text and code spans, in order."""
    return "".join(
        node["text"] for node in _walk(nodes) if node["kind"] in ("text", "code_inline")
    )


def _source(path):
    """A renderer component's source. It is absent until the components are
    written, and its absence fails the wiring half by name."""
    target = REPO / path
    assert target.is_file(), (
        f"{path} is absent: no component renders the tree yet, so the wiring "
        f"half cannot hold"
    )
    return read(path)


def _imports(text, name, module):
    """True when a component imports ``name`` from ``module``."""
    pattern = (
        r"import\s*\{[^}]*\b" + re.escape(name) + r"\b[^}]*\}\s*from\s*['\"]"
        + re.escape(module) + r"['\"]"
    )
    return re.search(pattern, text) is not None


# ---------------------------------------------------------------------------
# Server rendering: what the components emit, parsed as HTML
# ---------------------------------------------------------------------------
_VOID = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
})


class _Element:
    """One element of rendered HTML: its tag, its attributes in order, and
    its children (elements and text, references decoded)."""

    def __init__(self, tag, attrs, parent):
        self.tag = tag
        self.attrs = list(attrs)
        self.parent = parent
        self.children = []

    def get(self, name):
        for key, value in self.attrs:
            if key == name:
                return value
        return None

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


def _dom(html_text):
    """The rendered HTML as a tree of ``_Element``."""
    parser = _Dom()
    parser.feed(html_text)
    parser.close()
    return parser.root


def _render(source, streaming=False, **props):
    """What ``Markdown.svelte`` emits for a reply, compiled for the server."""
    return ssr().render(_MARKDOWN, {"source": source, "streaming": streaming, **props}).html


def _paragraphs_shown(root):
    return [element.text() for element in root.iter("p")]


# ---------------------------------------------------------------------------
# MK18 -- run_ts, the Node harness every node half rests on
# ---------------------------------------------------------------------------
def _probe_driver(body):
    return "const clause = process.argv[2];\n" + body + "\n"


def test_mk18_run_ts_passes_a_driver_only_on_its_own_pass_line_and_a_zero_exit():
    passing = _probe_driver("console.log('output kept');\nconsole.log('PASS ' + clause);")
    out = run_ts({}, passing, "probe")
    assert "output kept" in out.splitlines(), "run_ts returns the driver's standard output"

    silent = _probe_driver("console.log('no verdict');")
    with pytest.raises(AssertionError, match="clause probe failed"):
        run_ts({}, silent, "probe")

    failing = _probe_driver("console.log('PASS ' + clause);\nprocess.exit(3);")
    with pytest.raises(AssertionError, match="rc=3"):
        run_ts({}, failing, "probe")

    other = _probe_driver("console.log('PASS ' + clause + '-other');")
    with pytest.raises(AssertionError, match="clause probe failed"):
        run_ts({}, other, "probe")

    with pytest.raises(AssertionError, match="module under contract is absent"):
        run_ts({"OO_ABSENT": f"{_LIB}/absent_module.ts"}, passing, "probe")


# ---------------------------------------------------------------------------
# MK17 -- ssr(), the server rendering every SSR half rests on
# ---------------------------------------------------------------------------
_FIXTURE_DIR = "frontend/src/lib/ssr_fixture"
_GREETING = f"{_FIXTURE_DIR}/greeting.ts"
_GREETING_TEXT = (
    "export function greet(name: string): string {\n"
    "\treturn 'Hello, ' + name;\n"
    "}\n"
)
_PROBE = f"{_FIXTURE_DIR}/Probe.svelte"
_PROBE_TEXT = (
    '<script lang="ts">\n'
    "\timport { page } from '$app/stores';\n"
    "\timport { goto } from '$app/navigation';\n"
    "\timport { browser } from '$app/environment';\n"
    "\timport { greet } from '$lib/ssr_fixture/greeting';\n"
    "\texport let name: string;\n"
    "\tconst where: string = $page.url.pathname;\n"
    "</script>\n"
    "\n"
    '<p class="probe">{greet(name)} at {where}, '
    "{browser ? 'browser' : 'server'}, {typeof goto}</p>\n"
)
_WRONG = f"{_FIXTURE_DIR}/Wrong.svelte"
_WRONG_TEXT = (
    '<script lang="ts">\n'
    "\texport let name: string;\n"
    "</script>\n"
    "\n"
    '<p class="probe">Goodbye, {name}</p>\n'
)
_THROWS = f"{_FIXTURE_DIR}/Throws.svelte"
_THROWS_TEXT = (
    '<script lang="ts">\n'
    "\texport let name: string;\n"
    "\tfunction fail(): string {\n"
    "\t\tthrow new Error('planted failure for ' + name);\n"
    "\t}\n"
    "</script>\n"
    "\n"
    "<p>{fail()}</p>\n"
)
_HOSTILE_NAME = '<script>alert(1)</script> & "q"'
# Svelte escapes & and < in text; the rest of the name is shown as written.
_PROBE_HTML = (
    '<p class="probe">Hello, &lt;script>alert(1)&lt;/script> &amp; "q" at /, '
    "server, function</p>"
)


def _frontend_status():
    """The frontend's status, ignored files included, read without letting
    git refresh the index."""
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored",
         "--", "frontend"],
        cwd=REPO, capture_output=True, text=True, check=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout


def test_mk17_ssr_renders_a_component_and_refuses_what_it_cannot_render():
    before = _frontend_status()
    server = ssr()
    for path, text in (
        (_GREETING, _GREETING_TEXT), (_PROBE, _PROBE_TEXT),
        (_WRONG, _WRONG_TEXT), (_THROWS, _THROWS_TEXT),
    ):
        server.plant(path, text)

    # Renders: the planted probe, through $lib and the three $app stubs, with
    # its text escaped.
    probe = server.render(_PROBE, {"name": _HOSTILE_NAME})
    assert probe.html == _PROBE_HTML, (
        f"the probe rendered {probe.html!r}, not {_PROBE_HTML!r}"
    )

    # Discriminates: a component that renders the wrong text is caught.
    wrong = server.render(_WRONG, {"name": _HOSTILE_NAME})
    assert wrong.html != _PROBE_HTML, "the wrong text passed for the probe's"
    assert "Goodbye, &lt;script>" in wrong.html, (
        f"the wrong component rendered its own text: {wrong.html!r}"
    )

    # Refuses: a component that throws is an error, never an empty render.
    with pytest.raises(AssertionError, match="planted failure for"):
        server.render(_THROWS, {"name": "the probe"})

    # A component of the tree renders too, its prop escaped.
    empty = server.render(
        "frontend/src/lib/ds/EmptyState.svelte", {"title": _HOSTILE_NAME},
    )
    assert "&lt;script>alert(1)&lt;/script>" in empty.html, (
        f"a component of the tree renders its prop escaped: {empty.html!r}"
    )
    assert "<script" not in empty.html

    # Writes nothing in the tree: the server's root and cache live in a copy
    # under the temporary directory, and the frontend's status is unchanged.
    temporary = Path(tempfile.gettempdir()).resolve()
    assert server.root.resolve().is_relative_to(temporary), (
        f"the server runs on {server.root}, not on a copy under {temporary}"
    )
    assert not server.root.resolve().is_relative_to(REPO.resolve())
    assert _frontend_status() == before, "server rendering wrote in the frontend"


# ---------------------------------------------------------------------------
# MK2 -- marked stays behind the tree
# ---------------------------------------------------------------------------
_MARKED_REFERENCE = re.compile(
    r"(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|\brequire\s*\(\s*)(['\"])marked(?:/[^'\"]*)?\1"
)
_MARKED_SAMPLE = (
    "import { Lexer } from 'marked';\n"
    "import {\n\tmarked\n} from \"marked\";\n"
    "const lazy = await import('marked');\n"
    "const old = require(\"marked/lib/marked.cjs\");\n"
    "import 'marked';\n"
)
_MARKED_IMPORT = re.compile(r"\bimport\s+([^;]*?)\s*from\s*(['\"])marked\2", re.DOTALL)


def test_mk2_marked_is_imported_by_the_tree_alone_and_only_as_its_lexer():
    assert len(_MARKED_REFERENCE.findall(_MARKED_SAMPLE)) == 5, (
        "the census reads every spelling of an import of marked (static, "
        "multi-line, dynamic, require, bare), each once"
    )

    listed = files((".svelte", ".ts", ".js", ".mjs", ".cjs"))
    references = {}
    for path in listed:
        found = _MARKED_REFERENCE.findall(read(path))
        if found:
            references[path] = len(found)
    assert references == {_TREE: 1}, (
        f"marked is imported by {_TREE} alone, once; found {references}"
    )

    imports = _MARKED_IMPORT.findall(read(_TREE))
    assert [clause for clause, _quote in imports] == ["{ Lexer }"], (
        f"{_TREE} imports marked's Lexer and nothing else: {imports}"
    )


# ---------------------------------------------------------------------------
# MK3 -- raw HTML is text
# ---------------------------------------------------------------------------
_HALVES = ("node", "wiring")
_WITH_SSR = _HALVES + ("ssr",)


@pytest.mark.parametrize("half", _WITH_SSR)
def test_mk3_raw_html_becomes_text_with_its_exact_characters(half):
    if half == "ssr":
        cases = (
            '<div onclick="steal()">hi</div>',
            "a <b>bold</b> <em>x</em> c",
            "x <script>y &amp; z</script> w",
            "<details open><summary>s</summary>body</details>",
        )
        for source in cases:
            html_text = _render(source)
            root = _dom(html_text)
            tags = {element.tag for element in root.iter()}
            assert tags <= {"#root", "div", "p"} and "onclick" not in {
                name for element in root.iter() for name, _value in element.attrs
            }, f"{source!r} emits no element of its own: {html_text}"
            assert _paragraphs_shown(root) == [source], (
                f"{source!r} is shown with its exact characters: {html_text}"
            )
        # A block keeps its lines and its indentation on screen: its
        # paragraph is a raw one, which the style sets pre-wrapped.
        lined = "<div>\n  <p>hi</p>\n</div>"
        root = _dom(_render(lined + "\n\nafter"))
        shown = [(p.text(), (p.get("class") or "").split()) for p in root.iter("p")]
        assert shown[0][0] == lined and "oo-md-raw" in shown[0][1], (
            f"a block of HTML is a raw paragraph with its exact lines: {shown}"
        )
        assert "oo-md-raw" not in shown[1][1], f"a paragraph of markdown is not raw: {shown}"
        return
    if half == "wiring":
        for path in _RENDERERS:
            text = _source(path)
            assert not re.search(r"['\"]html['\"]", text), f"{path} names an html kind"
        assert "{node.text}" in _source(_NODE), (
            "MarkdownNode shows a text node by interpolation"
        )
        assert re.search(r"\{#if\s+node\.kind\s*===\s*'text'\s*\}\{node\.text\}", _source(_NODE)), (
            "the text branch of MarkdownNode interpolates the text, and nothing else"
        )
        node = _source(_NODE)
        assert re.search(r"<p\b[^>]*\bclass:oo-md-raw=\{node\.raw\}", node), (
            "MarkdownNode marks a raw paragraph with the raw class"
        )
        style = re.search(r"<style[^>]*>(.*?)</style>", node, re.DOTALL)
        rule = style and re.search(r"\.oo-md-raw\s*\{([^}]*)\}", style.group(1))
        assert rule and re.search(r"white-space\s*:\s*pre-wrap\s*;", rule.group(1)), (
            "a raw paragraph keeps its line breaks and indentation (pre-wrap)"
        )
        return

    block, inline, raw_run, prose = _trees([
        ('<div onclick="steal()">hi</div>\n\n<!-- hidden -->\n\ntext after', False),
        ("a <b>bold</b> <em>x</em> c", False),
        ("<pre>\n  a &amp; b\n</pre>\n\nx <script>y &amp; z</script> w", False),
        ("a line\nand the next", False),
    ])
    # A block of HTML is a raw paragraph of text: its characters, less the
    # blank lines that end it.
    assert block[:2] == [
        {"kind": "p", "raw": True,
         "children": [{"kind": "text", "text": '<div onclick="steal()">hi</div>'}]},
        {"kind": "p", "raw": True, "children": [{"kind": "text", "text": "<!-- hidden -->"}]},
    ], f"block HTML is text: {block}"
    assert raw_run[0].get("raw") is True, f"a block of HTML is a raw paragraph: {raw_run}"
    assert all("raw" not in node for node in prose + block[2:] + inline + raw_run[1:]), (
        f"a paragraph of markdown is not raw: {prose + block[2:] + inline + raw_run[1:]}"
    )
    # Inline HTML is text beside text: the paragraph shows its source.
    assert [node["kind"] for node in _walk(inline)] == ["p"] + ["text"] * 9, (
        f"inline HTML is text: {inline}"
    )
    assert _text(inline) == "a <b>bold</b> <em>x</em> c"
    # The text inside a raw element keeps its references literal.
    assert raw_run[0]["children"] == [{"kind": "text", "text": "<pre>\n  a &amp; b\n</pre>"}]
    assert _text(raw_run[1:]) == "x <script>y &amp; z</script> w", (
        f"the run of a raw element is shown as written: {raw_run[1:]}"
    )


# ---------------------------------------------------------------------------
# MK4 -- links are filtered
# ---------------------------------------------------------------------------
_ACCEPTED = (
    ("[t](https://ok.test/a?b=1)", "https://ok.test/a?b=1"),
    ("[t](http://ok.test)", "http://ok.test/"),
    ("[t](mailto:someone@ok.test)", "mailto:someone@ok.test"),
    ("[t](HTTPS://OK.TEST/Path)", "https://ok.test/Path"),
    ("<https://auto.test/x>", "https://auto.test/x"),
    ("see www.auto.test here", "http://www.auto.test/"),
    ("write to someone@auto.test", "mailto:someone@auto.test"),
)
_REFUSED = (
    "[t](javascript:alert(1))",
    "[t](JaVaScRiPt:alert(1))",
    "[t](&#106;avascript:alert(1))",
    "[t](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
    "[t](vbscript:msgbox(1))",
    "[t](file:///etc/passwd)",
    "[t](/relative/path)",
    "[t](#fragment)",
    "[t](//protocol-relative.test/x)",
    "[t](ftp://files.test/x)",
    "[t](blob:https://ok.test/id)",
    "[t](relative.html)",
)


@pytest.mark.parametrize("half", _HALVES)
def test_mk4_only_absolute_http_https_and_mailto_destinations_become_links(half):
    if half == "wiring":
        anchors = {path: re.findall(r"<a\b[^>]*>", _source(path)) for path in _RENDERERS}
        assert {path: len(found) for path, found in anchors.items() if found} == {_NODE: 1}, (
            f"one anchor, in MarkdownNode: {anchors}"
        )
        anchor = anchors[_NODE][0]
        for attribute in (
            "href={node.href}", 'target="_blank"', 'rel="noopener noreferrer"',
            'referrerpolicy="no-referrer"',
        ):
            assert attribute in anchor, f"the anchor carries {attribute}: {anchor}"
        return

    trees = _trees([(source, False) for source, _href in _ACCEPTED])
    for (source, href), tree in zip(_ACCEPTED, trees):
        links = [node for node in _walk(tree) if node["kind"] == "link"]
        assert [link["href"] for link in links] == [href], (
            f"{source!r} is one link to {href}: {tree}"
        )

    trees = _trees([(source, False) for source in _REFUSED])
    for source, tree in zip(_REFUSED, trees):
        assert not [node for node in _walk(tree) if node["kind"] == "link"], (
            f"{source!r} is no link: {tree}"
        )
        assert tree == [{"kind": "p", "children": [{"kind": "text", "text": source}]}], (
            f"{source!r} is shown as its source text: {tree}"
        )


# ---------------------------------------------------------------------------
# MK5 -- images never load
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", _WITH_SSR)
def test_mk5_an_image_becomes_text_and_nothing_can_load_it(half):
    if half == "ssr":
        html_text = _render(
            '![alt text](http://x.test/p.png "t")\n\n'
            "a ![](https://x.test/pixel.gif) b\n\n"
            "| ![x](http://x.test/cell.png) |\n|---|\n| c |"
        )
        root = _dom(html_text)
        assert not list(root.iter("img")), f"no image element is emitted: {html_text}"
        loaders = [
            (element.tag, name) for element in root.iter() for name, _value in element.attrs
            if name in ("src", "srcset", "poster", "background")
        ]
        assert loaders == [], f"no attribute loads anything: {loaders}"
        assert _paragraphs_shown(root)[:2] == [
            "Image: alt text (http://x.test/p.png)",
            "a Image: (https://x.test/pixel.gif) b",
        ], f"an image is shown as its text: {html_text}"
        assert "Image: x (http://x.test/cell.png)" in root.text()
        return
    if half == "wiring":
        for path in _RENDERERS:
            text = _source(path)
            assert "<img" not in text and "src=" not in text, f"{path} can load an image"
        return

    trees = _trees([
        ('![alt text](http://x.test/p.png "t")', False),
        ("![a &amp; b](http://x.test/a&amp;b.png)", False),
        ("![](https://x.test/pixel.gif)", False),
        ("before ![x](javascript:alert(1)) after", False),
    ])
    assert [_text(tree) for tree in trees] == [
        "Image: alt text (http://x.test/p.png)",
        "Image: a & b (http://x.test/a&b.png)",
        "Image: (https://x.test/pixel.gif)",
        "before Image: x (javascript:alert(1)) after",
    ], f"an image is text: {trees}"
    for tree in trees:
        for node in _walk(tree):
            assert node["kind"] == "text" or node["kind"] == "p", f"no image node: {node}"
            assert "src" not in node and "href" not in node, f"nothing to load: {node}"


# ---------------------------------------------------------------------------
# MK6 -- code blocks: closed, copy text, label
# ---------------------------------------------------------------------------
def _code_blocks(tree):
    return [node for node in _walk(tree) if node["kind"] == "code_block"]


@pytest.mark.parametrize("half", _WITH_SSR)
def test_mk6_a_code_block_is_open_only_while_its_last_fence_streams(half):
    if half == "ssr":
        unterminated = "para\n\n```python\ndef f():\n  pass"
        figures = {}
        for key, source, streaming in (
            ("open", unterminated, True),
            ("finished", unterminated, False),
            ("terminated", "```js\nx\n```", True),
            ("indented", "para\n\n    indented", True),
        ):
            found = list(_dom(_render(source, streaming)).iter("figure"))
            assert len(found) == 1, f"{key}: one figure per code block: {found}"
            figures[key] = found[0]
        buttons = {key: [b.text() for b in figure.iter("button")] for key, figure in figures.items()}
        assert buttons == {
            "open": [], "finished": ["Copy code"], "terminated": ["Copy code"],
            "indented": ["Copy code"],
        }, f"Copy is absent only while the last fence streams: {buttons}"
        captions = {
            key: [c.text() for c in figure.iter("figcaption")] for key, figure in figures.items()
        }
        assert captions["open"][0].startswith("python"), captions
        assert captions["terminated"][0].startswith("js"), captions
        for key, figure in figures.items():
            code = [pre.text() for pre in figure.iter("pre")]
            assert code == [{"terminated": "x", "indented": "indented"}.get(key, "def f():\n  pass")], (
                f"{key}: the code is shown as written: {code}"
            )
            assert [pre.get("tabindex") for pre in figure.iter("pre")] == ["0"], (
                f"{key}: the code scrolls and takes the focus"
            )
        # The figure is named by its label alone, not by the caption's
        # status and button.
        root = _dom(_render("```js\na\n```\n\n```\nb\n```"))
        names = []
        for figure in root.iter("figure"):
            ident = figure.get("aria-labelledby")
            labels = [e for e in root.iter() if ident and e.get("id") == ident]
            names.append((ident, [label.text() for label in labels]))
        assert [name for _ident, name in names] == [["js"], ["Code"]], (
            f"each figure is named by its own label, and only by it: {names}"
        )
        assert names[0][0] != names[1][0], f"each figure has its own label: {names}"
        return
    if half == "wiring":
        text = _source(_CODE_BLOCK)
        assert re.search(r"\{#if\s+node\.closed\s*\}", text), "Copy shows only when closed"
        assert "node.lang" in text and "node.code" in text
        assert "```" not in text and "~~~" not in text and ".split(" not in text, (
            "CodeBlock reads no fence of its own"
        )
        return

    unterminated = "para\n\n```python\ndef f():\n  pass"
    cases = [
        (unterminated, True),                       # streaming, last, open
        (unterminated, False),                      # streaming over: closed
        ("```js\nx\n```", True),                    # terminated
        ("```js\nx\n```\n\ntext", True),            # not last
        ("para\n\n    indented code", True),        # indented: always closed
        ("````\n```\ninner", True),                 # a shorter fence closes nothing
        ("~~~\ncode", True),
        ("~~~\ncode\n~~~", True),
        ("~~~\ncode\n```", True),                   # the other fence closes nothing
        ("- item\n\n  ```py\n  code", True),        # the last fence of the last item
        ("- item\n\n  ```py\n  code\n  ```\n- next", True),
        ("```py", True),                            # the fence alone
        ("- item\n\n  ```py\n  code\n- next", True),  # unterminated, not last
        ("````\n```", True),                          # ends on a shorter fence
    ]
    closed = [[block["closed"] for block in _code_blocks(tree)] for tree in _trees(cases)]
    assert closed == [
        [False], [True], [True], [True], [True], [False],
        [False], [True], [False], [False], [True], [False], [True], [False],
    ], f"closed unless streaming, last, and unterminated: {closed}"

    copied = [
        [block["code"] for block in _code_blocks(tree)]
        for tree in _trees([
            (unterminated, False),
            ("```js\nconst a = '<b>' && \"q\";\n```", False),
            ("```\na &amp; b\n```", False),
            ("para\n\n    indented code\n    more", False),
        ])
    ]
    assert copied == [
        ["def f():\n  pass"],
        ["const a = '<b>' && \"q\";"],
        ["a &amp; b"],
        ["indented code\nmore"],
    ], f"the copy text is the code, without fence or info string: {copied}"

    infos = [
        ("js extra words", "js"), ("c++", "c++"), ("C#", "C#"),
        ("objective-c", "objective-c"), ("py3.12", "py3.12"), ("", ""),
        ("<script>", ""), ("x" * 20, "x" * 20), ("x" * 21, ""),
        ("py{.class}", ""), ("rust,ignore", ""),
    ]
    labels = [
        _code_blocks(tree)[0]["lang"]
        for tree in _trees([(f"```{info}\ncode\n```", False) for info, _label in infos])
    ]
    assert labels == [label for _info, label in infos], (
        f"the label is the info string's first word, when it is at most 20 "
        f"characters of [A-Za-z0-9+#.-]: {labels}"
    )
    assert _code_blocks(_tree("    indented"))[0]["lang"] == ""


# ---------------------------------------------------------------------------
# MK7 -- references decode once, in text only
# ---------------------------------------------------------------------------
_TABLE_ROWS = (
    ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'"),
    ("&nbsp;", chr(0xA0)), ("&mdash;", chr(0x2014)), ("&ndash;", chr(0x2013)),
    ("&hellip;", chr(0x2026)), ("&copy;", chr(0xA9)),
    ("&#65;", "A"), ("&#x41;", "A"), ("&#X6a;", "j"), ("&#x1D400;", chr(0x1D400)),
    ("&#1114111;", chr(0x10FFFF)), ("&#0;", chr(0xFFFD)), ("&#xD800;", chr(0xFFFD)),
    ("&#x110000;", chr(0xFFFD)),
)
_LITERAL = (
    "&foo;", "&apos;", "&LT;", "&constructor;", "&toString;", "&hasOwnProperty;",
    "&#;", "&#x;", "&#12345678;", "&amp", "& amp;",
)


@pytest.mark.parametrize("half", _WITH_SSR)
def test_mk7_references_decode_once_in_text_and_never_in_code(half):
    if half == "ssr":
        # The emitted HTML escapes what the text holds, once: parsed back,
        # its text is the decoded text, character for character.
        cases = (
            ("a < b & c > d \" e ' f", "a < b & c > d \" e ' f"),
            ("&amp;lt; is not &lt;", "&lt; is not <"),
            ("&lt;script&gt;x&lt;/script&gt; &#65;&#x42; &foo;", "<script>x</script> AB &foo;"),
            ("x `a &lt; b` y", "x a &lt; b y"),
        )
        for source, shown in cases:
            html_text = _render(source)
            assert _paragraphs_shown(_dom(html_text)) == [shown], (
                f"{source!r} shows {shown!r}: {html_text}"
            )
            assert "<script" not in html_text
        code = [pre.text() for pre in _dom(_render("```\na &amp; b < c\n```")).iter("pre")]
        assert code == ["a &amp; b < c"], f"a code block keeps its references: {code}"
        return
    if half == "wiring":
        text = _source(_MARKDOWN)
        assert _imports(text, "decodeEntities", "$lib/markdown/entities"), (
            "Markdown imports the decoder from entities.ts"
        )
        assert re.search(r"\bdecode\s*:\s*decodeEntities\b", text), (
            "Markdown hands the decoder to the tree"
        )
        return

    # Round trip, and one decoding only.
    plain, once = _trees([("a < b & c > d \" e ' f", False), ("&amp;lt; is not &lt;", False)])
    assert plain == [{"kind": "p", "children": [{"kind": "text", "text": "a < b & c > d \" e ' f"}]}]
    assert _text(once) == "&lt; is not <", f"a reference decodes once: {once}"
    assert _node("decode", ("OO_ENTITIES",), ["&amp;amp;", "&amp;#60;"]) == ["&amp;", "&#60;"]

    # The table, every numeric reference, and unknown names left as written.
    rows = [(entity, value) for entity, value in _TABLE_ROWS] + [(name, name) for name in _LITERAL]
    decoded = _node("decode", ("OO_ENTITIES",), [entity for entity, _value in rows])
    assert decoded == [value for _entity, value in rows], (
        f"decoded {list(zip([e for e, _v in rows], decoded))}"
    )
    shown = [_text(tree) for tree in _trees([(f"x {entity} y", False) for entity, _v in rows])]
    assert shown == [f"x {value} y" for _entity, value in rows], "text decodes in a paragraph"
    heading, link, cell, item = _trees([
        ("## a &amp; b", False),
        ("[a &amp; b](https://ok.test/?a=1&amp;b=2 \"t &amp; u\")", False),
        ("| a &amp; b |\n|---|\n| c &lt; d |", False),
        ("- a &mdash; b", False),
    ])
    assert _text(heading) == "a & b"
    assert link[0]["children"] == [{
        "kind": "link", "href": "https://ok.test/?a=1&b=2", "title": "t & u",
        "children": [{"kind": "text", "text": "a & b"}],
    }], f"a link's text, destination and title decode: {link}"
    assert _text(cell) == "a & bc < d"
    assert _text(item) == "a " + chr(0x2014) + " b"

    # Code spans, code blocks and escapes keep references literal.
    span, block, escaped = _trees([
        ("x `a &lt; b` y", False), ("```\na &amp; b\n```", False), ("\\&lt; x", False),
    ])
    assert [node for node in _walk(span) if node["kind"] == "code_inline"] == [
        {"kind": "code_inline", "text": "a &lt; b"},
    ], f"a code span keeps its references: {span}"
    assert _code_blocks(block)[0]["code"] == "a &amp; b"
    assert _text(escaped) == "&lt; x", f"an escaped ampersand starts no reference: {escaped}"


# ---------------------------------------------------------------------------
# MK8 -- heading levels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", _WITH_SSR)
def test_mk8_a_heading_of_depth_d_has_level_min_6_d_plus_2(half):
    if half == "ssr":
        root = _dom(_render(
            "# one\n\n## two\n\n### three\n\n#### four\n\n##### five\n\n###### six\n\n"
            "Setext one\n==========\n\n> # quoted"
        ))
        headings = [
            (element.tag, element.text()) for element in root.iter()
            if re.fullmatch(r"h[1-6]", element.tag)
        ]
        assert headings == [
            ("h3", "one"), ("h4", "two"), ("h5", "three"), ("h6", "four"),
            ("h6", "five"), ("h6", "six"), ("h3", "Setext one"), ("h3", "quoted"),
        ], f"the emitted heading of depth d is h(min(6, d + 2)): {headings}"
        return
    if half == "wiring":
        assert "node.level" in _source(_NODE), "MarkdownNode renders the node's level"
        for path in _RENDERERS:
            assert "depth" not in _source(path), f"{path} computes a level of its own"
        return

    tree = _tree(
        "# one\n\n## two\n\n### three\n\n#### four\n\n##### five\n\n###### six\n\n"
        "Setext one\n==========\n\nSetext two\n----------"
    )
    assert [(node["kind"], node.get("level")) for node in tree] == [
        ("heading", 3), ("heading", 4), ("heading", 5), ("heading", 6),
        ("heading", 6), ("heading", 6), ("heading", 3), ("heading", 4),
    ], f"level min(6, d + 2): {tree}"
    assert _text(tree[:1]) == "one"


# ---------------------------------------------------------------------------
# MK10 -- a closed set of node kinds
# ---------------------------------------------------------------------------
_EVERY_KIND = (
    "# Heading\n\n"
    "A paragraph with **strong**, *em*, ~~del~~, `code`, a [link](https://ok.test) "
    "and a break\nhere.\n\n"
    "> quote\n\n"
    "1. one\n2. two\n\n"
    "- [x] done\n- [ ] open\n\n"
    "| a | b |\n|:-:|--:|\n| 1 | 2 |\n\n"
    "---\n\n"
    "```py\nx = 1\n```\n"
)


@pytest.mark.parametrize("half", _HALVES)
def test_mk10_the_node_kinds_are_a_closed_set_and_an_unknown_token_is_its_raw_text(half):
    kinds = _node("kinds", ("OO_TREE",))
    if half == "wiring":
        handled = set(re.findall(r"\bkind\s*===\s*['\"]([a-z_]+)['\"]", _source(_NODE)))
        assert handled == set(kinds), (
            f"MarkdownNode handles each kind of the closed set and no other: "
            f"handled {sorted(handled)}, set {sorted(kinds)}"
        )
        return

    produced = {node["kind"] for node in _walk(_tree(_EVERY_KIND))}
    assert produced == set(kinds) and len(kinds) == len(set(kinds)), (
        f"the corpus produces exactly the declared kinds: {sorted(produced)} "
        f"against {kinds}"
    )

    unknown_block, unknown_inline, bare, partial = _node(
        "injected", ("OO_TREE", "OO_ENTITIES"), [
            [{"type": "future_block", "raw": "<x> & y &amp;"}],
            [{"type": "paragraph", "raw": "a@@b", "tokens": [
                {"type": "text", "raw": "a", "text": "a"},
                {"type": "future_inline", "raw": "@@ raw &amp;"},
                {"type": "text", "raw": "b", "text": "b"},
            ]}],
            [{"type": "future_block"}],
            [{"type": "paragraph"}, {"type": "heading", "depth": "deep"}],
        ],
    )
    assert unknown_block == [
        {"kind": "p", "raw": True, "children": [{"kind": "text", "text": "<x> & y &amp;"}]},
    ], f"an unknown block token is its raw source, as a raw paragraph: {unknown_block}"
    assert unknown_inline == [{"kind": "p", "children": [
        {"kind": "text", "text": "a"}, {"kind": "text", "text": "@@ raw &amp;"},
        {"kind": "text", "text": "b"},
    ]}], f"an unknown inline token is its raw text: {unknown_inline}"
    assert bare == [], "an unknown token with no source shows nothing"
    assert partial == [
        {"kind": "p", "children": []}, {"kind": "heading", "level": 3, "children": []},
    ], f"a token missing its fields still maps into the set: {partial}"


# ---------------------------------------------------------------------------
# MK11 -- breaks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", _HALVES)
def test_mk11_a_single_newline_in_a_paragraph_is_a_line_break(half):
    if half == "wiring":
        assert _imports(_source(_MARKDOWN), "lexMarkdown", "$lib/markdown/tree"), (
            "Markdown lexes with the tree's lexer"
        )
        assert re.search(r"<br\b", _source(_NODE)), "MarkdownNode renders a break"
        for path in _RENDERERS:
            assert "new Lexer" not in _source(path), f"{path} builds a lexer of its own"
        return

    soft, hard, quoted = _trees([
        ("line one\nline two", False), ("a  \nb", False), ("> one\n> two", False),
    ])
    assert soft == [{"kind": "p", "children": [
        {"kind": "text", "text": "line one"}, {"kind": "br"},
        {"kind": "text", "text": "line two"},
    ]}], f"a single newline is a break: {soft}"
    assert [node["kind"] for node in _walk(hard)] == ["p", "text", "br", "text"]
    assert [node["kind"] for node in _walk(quoted)] == ["blockquote", "p", "text", "br", "text"]


# ---------------------------------------------------------------------------
# MK12 -- collapse by whole blocks
# ---------------------------------------------------------------------------
def _paragraphs(first, last):
    return [f"Paragraph {i}" for i in range(first, last + 1)]


_SIX_LINES = "```py\n" + "\n".join(f"line_{i} = {i}" for i in range(6)) + "\n```"
_THIRTY_LINES = "```py\n" + "\n".join(f"row_{i} = {i}" for i in range(30)) + "\n```"


@pytest.mark.parametrize("half", _WITH_SSR)
def test_mk12_collapse_keeps_whole_top_level_blocks_and_drops_the_hidden_ones(half):
    if half == "ssr":
        source = "\n\n".join(_paragraphs(1, 30))
        props = {"collapseAbove": 20, "collapseKeep": 8}
        root = _dom(_render(source, **props))
        assert _paragraphs_shown(root) == _paragraphs(1, 8), (
            f"a finished long reply shows its first whole blocks, and no hidden "
            f"block is emitted: {_paragraphs_shown(root)}"
        )
        toggles = [
            b for b in root.iter("button") if b.get("aria-expanded") is not None
        ]
        assert [(b.text(), b.get("aria-expanded")) for b in toggles] == [
            ("Show the rest (22 more blocks)", "false"),
        ], f"one toggle names what it hides and says it is collapsed: {toggles}"
        controlled = [e for e in root.iter() if e.get("id") == toggles[0].get("aria-controls")]
        assert len(controlled) == 1 and _paragraphs_shown(controlled[0]) == _paragraphs(1, 8), (
            "the toggle controls the element that holds the blocks"
        )
        streaming = _dom(_render(source, True, **props))
        assert _paragraphs_shown(streaming) == _paragraphs(1, 30), (
            "a reply still arriving never collapses"
        )
        assert not list(streaming.iter("button")), "nothing to toggle while streaming"
        short = _dom(_render("\n\n".join(_paragraphs(1, 5)), **props))
        assert _paragraphs_shown(short) == _paragraphs(1, 5) and not list(short.iter("button"))
        return
    if half == "wiring":
        text = _source(_MARKDOWN)
        assert _imports(text, "collapseBlocks", "$lib/markdown/collapse")
        assert re.search(r"\bcollapseBlocks\s*\(", text), "Markdown collapses through collapse.ts"
        assert ".split(" not in text and ".slice(" not in text, "Markdown cuts nothing itself"
        chat = _source(_CHAT_MESSAGE)
        assert re.search(r"<Markdown\b", chat), "the reply renders through Markdown"
        assert not re.search(r"\.split\(\s*['\"]\\n['\"]\s*\)", chat), (
            "ChatMessage no longer cuts a reply by lines"
        )
        return

    middle = "\n\n".join(_paragraphs(1, 3) + [_SIX_LINES] + _paragraphs(5, 30))
    head = "\n\n".join([_THIRTY_LINES] + _paragraphs(2, 10))
    budget = {"threshold": 20, "keep": 8, "expanded": False}
    kept, whole_first = _node("collapse", ("OO_TREE", "OO_ENTITIES", "OO_COLLAPSE"), [
        [middle, budget], [head, budget],
    ])
    for case in (kept, whole_first):
        count, weights = len(case["indices"]), case["weights"]
        assert all(isinstance(w, int) and w >= 1 for w in weights), f"weights: {weights}"
        assert case["indices"] == list(range(count)) and count >= 1, (
            f"the shown blocks are the first ones, each whole: {case['indices']}"
        )
        assert count == 1 or sum(weights[:count]) <= budget["keep"]
        assert sum(weights[:count + 1]) > budget["keep"], "a block that fits is not hidden"
        assert case["hidden"] == len(case["blocks"]) - count and case["hidden"] > 0
        assert case["collapsible"] is True
    assert len(kept["indices"]) == 3, f"three paragraphs fit, the code block does not: {kept}"
    assert whole_first["indices"] == [0], "a first block over the budget is kept whole"
    assert whole_first["blocks"][0]["code"].count("\n") == 29

    expanded, short, never, single, under = _node(
        "collapse", ("OO_TREE", "OO_ENTITIES", "OO_COLLAPSE"), [
            [middle, dict(budget, expanded=True)],
            ["\n\n".join(_paragraphs(1, 5)), budget],
            [middle, dict(budget, threshold=0)],
            [_THIRTY_LINES, budget],
            ["\n\n".join(_paragraphs(1, 12)), budget],
        ],
    )
    assert expanded["indices"] == list(range(len(expanded["blocks"])))
    assert expanded["hidden"] == 0 and expanded["collapsible"] is True, (
        "an expanded reply shows every block and can collapse again"
    )
    for case in (short, never, single):
        assert case["indices"] == list(range(len(case["blocks"])))
        assert case["hidden"] == 0 and case["collapsible"] is False, (
            f"a short reply, a zero threshold, or a single block never collapse: {case}"
        )
    assert budget["keep"] < sum(under["weights"]) <= budget["threshold"], (
        f"the fixture weighs more than the kept budget and no more than the threshold: "
        f"{under['weights']}"
    )
    assert under["indices"] == list(range(12)) and under["collapsible"] is False, (
        f"a reply at most as long as the threshold is shown whole, though it holds more "
        f"than the kept budget: {under}"
    )


# ---------------------------------------------------------------------------
# MK13 -- one lex per frame
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", _HALVES)
def test_mk13_a_hundred_updates_inside_one_frame_cause_one_lex(half):
    if half == "wiring":
        text = _source(_MARKDOWN)
        assert _imports(text, "createFrameLexer", "$lib/markdown/frame")
        assert re.search(r"\bcreateFrameLexer\s*\(", text)
        assert not re.search(r"\blexMarkdown\s*\(", text), (
            "Markdown never lexes itself: it hands the lexer to frame.ts"
        )
        branch = re.search(
            r"\$:\s*if\s*\(\s*streaming\s*\)\s*lexer\.push\(\s*(\w+)\s*\)\s*;?\s*"
            r"else\s+lexer\.now\(\s*\1\s*\)",
            text,
        )
        assert branch is not None, (
            "while the reply streams its source is pushed to the frame, and only "
            "otherwise is it lexed at once"
        )
        assert len(re.findall(r"\.push\(", text)) == 1 and len(re.findall(r"\.now\(", text)) == 1, (
            "the frame lexer is pushed and lexed at once in that statement only"
        )
        return

    report = _node("frame", ("OO_FRAME",))
    assert report["beforeFrame"] == {"lexed": 0, "scheduled": 1}, (
        f"a hundred updates schedule one frame and lex nothing yet: {report['beforeFrame']}"
    )
    assert report["firstRun"] == 1
    assert report["firstFrame"] == {
        "lexed": ["update 100"], "delivered": ["tokens of update 100"],
    }, f"the frame lexes once, the last source: {report['firstFrame']}"
    assert report["secondFrame"]["lexed"] == ["update 100", "update 150"]
    assert report["secondRun"] == 1

    assert report["unchanged"] == {"scheduled": 0}, "an unchanged source schedules nothing"
    assert report["now"] == "tokens of final"
    assert report["afterNow"] == {
        "ran": 0, "lexed": ["update 100", "update 150", "final"],
        "delivered": ["tokens of update 100", "tokens of update 150", "tokens of final"],
    }, f"now() lexes at once and drops the pending frame: {report['afterNow']}"
    assert report["afterCancel"]["ran"] == 0
    assert "dropped" not in report["afterCancel"]["lexed"], "cancel() drops the pending frame"

    assert report["withoutFrames"] == ["first", "second"], (
        "where no frames exist (server rendering), each update lexes at once"
    )


# ---------------------------------------------------------------------------
# MK14 -- the two-role highlighter
# ---------------------------------------------------------------------------
_SAMPLES = (
    (
        "python",
        "def parse_reply(text):\n"
        "    # return is only a word in this comment\n"
        "    label = \"class inside a string\"\n"
        "    return label if text else None\n\n"
        "class Tree:\n    pass\n",
        {"def", "return", "if", "else", "None", "class", "pass"},
        {"parse_reply", "Tree"},
        ("# return is only a word in this comment", '"class inside a string"'),
    ),
    (
        "ts",
        "// function inside a comment\n"
        "export function renderNode(node) {\n"
        "  const limit = 3;\n"
        "  if (node.delete) return node.type as string;\n"
        "  return node.kind === 'if' ? limit : null;\n"
        "}\n"
        "interface Shape { kind: string }\n",
        {"export", "function", "const", "return", "null", "interface"},
        {"renderNode", "limit", "Shape"},
        ("// function inside a comment", "'if'"),
    ),
    (
        "rust",
        "// fn in a comment\n"
        "pub fn main() {\n"
        "    let text = \"struct inside a string\";\n"
        "    if text.is_empty() { return; }\n"
        "}\n"
        "struct Node { depth: u8 }\n",
        {"pub", "fn", "let", "if", "return", "struct"},
        {"main", "Node"},
        ("// fn in a comment", '"struct inside a string"'),
    ),
    (
        "bash",
        "# for in a comment\n"
        "deploy() {\n"
        "  if [ -n \"$1\" ]; then echo \"done in a string\"; fi\n"
        "}\n"
        "function clean {\n  return 0\n}\n",
        {"if", "then", "fi", "function", "return"},
        {"deploy", "clean"},
        ("# for in a comment", '"done in a string"'),
    ),
    (
        "SQL",
        "-- select in a comment\n"
        "CREATE TABLE replies (id INTEGER PRIMARY KEY, body TEXT);\n"
        "ALTER TABLE notes ADD COLUMN body TEXT;\n"
        "SELECT r.table FROM replies r WHERE body = 'from inside a string';\n",
        {"CREATE", "TABLE", "PRIMARY", "KEY", "ALTER", "SELECT", "FROM", "WHERE"},
        {"replies"},
        ("-- select in a comment", "'from inside a string'"),
    ),
)
_UNKNOWN = (("if x then y", "brainfudge"), ("return 1", ""), ("fn x", "constructor"))
_UNTERMINATED = (
    ("x = \"open string\ny = 1", "py"), ("/* open comment\nconst a = 1", "js"),
    ("let s = \"open", "rs"), ("echo 'open", "sh"), ("SELECT 'open", "sql"),
)


@pytest.mark.parametrize("half", _WITH_SSR)
def test_mk14_highlighting_keeps_every_character_in_two_roles_and_plain(half):
    if half == "ssr":
        # The code block renders the highlighter's spans, each in its role's
        # class, and they add up to the code.
        code = "def parse_reply(text):\n    return text  # return in a comment"
        root = _dom(_render(f"```python\n{code}\n```\n\n```brainfudge\nif x then y\n```"))
        known, unknown = list(root.iter("pre"))
        roles = {}
        for span in known.iter("span"):
            for name in (span.get("class") or "").split():
                if name.startswith("oo-md-syn-"):
                    roles.setdefault(name[len("oo-md-syn-"):], []).append(span.text())
        assert roles.get("keyword") == ["def", "return"], f"keywords in their class: {roles}"
        assert roles.get("name") == ["parse_reply"], f"the defined name in its class: {roles}"
        assert known.text() == code, "the spans add up to the code, character for character"
        plain = [
            name for span in unknown.iter("span")
            for name in (span.get("class") or "").split() if name.startswith("oo-md-syn-")
        ]
        assert plain == ["oo-md-syn-plain"] and unknown.text() == "if x then y", (
            f"an unknown language is one plain span: {plain}"
        )
        return
    if half == "wiring":
        text = _source(_CODE_BLOCK)
        assert _imports(text, "highlight", "$lib/markdown/highlight")
        assert re.search(r"\bhighlight\s*\(\s*node\.code\s*,\s*node\.lang\s*\)", text), (
            "CodeBlock highlights the node's code in its language"
        )
        return

    inputs = [(code, lang) for lang, code, *_rest in _SAMPLES] + list(_UNKNOWN) + list(_UNTERMINATED)
    results = _node("highlight", ("OO_HIGHLIGHT",), inputs)
    for (code, lang), spans in zip(inputs, results):
        assert "".join(span["text"] for span in spans) == code, (
            f"the spans of {lang!r} concatenate to its code: {spans}"
        )
        assert all(span["text"] for span in spans), "no span is empty"

    roles = _node("roles", ("OO_HIGHLIGHT",))
    assert roles == ["plain", "keyword", "name"]
    for (lang, code, keywords, names, _plain), spans in zip(_SAMPLES, results):
        assert {span["role"] for span in spans} <= set(roles)
        found = {role: {s["text"] for s in spans if s["role"] == role} for role in roles}
        assert keywords <= found["keyword"], f"{lang} keywords: {found['keyword']}"
        assert names == found["name"], f"{lang} names: {found['name']}"

    for (code, lang), spans in zip(_UNKNOWN, results[len(_SAMPLES):]):
        assert spans == [{"role": "plain", "text": code}], (
            f"an unknown language ({lang!r}) is one plain span: {spans}"
        )

    for (lang, code, _keywords, _names, plain), spans in zip(_SAMPLES, results):
        regions = [(code.index(part), code.index(part) + len(part)) for part in plain]
        offset = 0
        for span in spans:
            start, end = offset, offset + len(span["text"])
            offset = end
            if span["role"] != "plain":
                inside = [r for r in regions if start < r[1] and r[0] < end]
                assert not inside, f"{lang}: {span} lies in a string or a comment"
                assert start == 0 or code[start - 1] != ".", (
                    f"{lang}: {span} is a member after a dot, not a keyword or a name"
                )


# ---------------------------------------------------------------------------
# MK1 -- no raw HTML sink, and the lint keeps its rule against one
# ---------------------------------------------------------------------------
_SINKS = re.compile(
    r"\{\s*@html\b|\binnerHTML\b|\bouterHTML\b|\binsertAdjacentHTML\b"
    r"|(?<![\w$])Function\s*\(|(?<![\w$])eval\s*\("
)
_SINK_SAMPLE = (
    "{@html reply}\nnode.innerHTML = x;\nnode.outerHTML = x;\n"
    "node.insertAdjacentHTML('beforeend', x);\nconst f = new Function('return 1');\n"
    "const g = Function('return 2');\nwindow.eval(code);\neval (code);\n"
    "{ @html spaced}\n{\n\t@html broken}\nglobalThis.Function('return 3')();\n"
    "window.Function(code);\n"
)
_NOT_SINKS = (
    "retrieval(x); myFunction(y); node.textContent = z; evaluate(w);\n"
    "type F = Function; const html = 'innerHTMLish'; {@htmlish} isFunction(v);\n"
)
# The other calls that write markup into a document. They are not a reply's
# path; each stands only where it stands today.
_WRITERS = re.compile(
    r"\bdocument\s*\.\s*write(?:ln)?\s*\(|\bcreateContextualFragment\s*\(|\bsrcdoc\b"
    r"|\bsetHTMLUnsafe\s*\(|\bparseHTMLUnsafe\s*\("
)
_WRITER_SAMPLE = (
    "win.document.write(x);\ndocument .writeln(x);\nrange.createContextualFragment(x);\n"
    "<iframe srcdoc={x}></iframe>\nnode.setHTMLUnsafe(x);\nDocument.parseHTMLUnsafe(x);\n"
)
_WRITERS_TODAY = {
    "frontend/src/lib/components/settings/RecoveryCodesPanel.svelte": ["document.write("],
}
_ESLINTRC = "frontend/.eslintrc.cjs"
_RECOMMENDED = "frontend/node_modules/eslint-plugin-svelte/lib/configs/recommended.js"


def test_mk1_no_raw_html_sink_anywhere_and_the_lint_keeps_its_rule():
    assert len(_SINKS.findall(_SINK_SAMPLE)) == 12, (
        "the census reads each sink in its sample: {@html} with or without "
        "space after the brace, innerHTML, outerHTML, insertAdjacentHTML, new "
        "Function(, and Function( and eval( bare or on an object"
    )
    assert _SINKS.findall(_NOT_SINKS) == [], "a lookalike is not a sink"
    assert len(_WRITERS.findall(_WRITER_SAMPLE)) == 6, (
        "the census reads each other markup writer in its sample"
    )

    listed = files((".svelte", ".ts", ".js", ".mjs", ".cjs", ".html"))
    assert set(_RENDERERS) <= set(listed) and _TREE in listed, (
        "the census reads the renderer's components and modules"
    )
    found = {path: _SINKS.findall(read(path)) for path in listed}
    found = {path: sinks for path, sinks in found.items() if sinks}
    assert found == {}, f"no raw HTML sink in frontend/src: {found}"
    writers = {path: _WRITERS.findall(read(path)) for path in listed}
    writers = {path: calls for path, calls in writers.items() if calls}
    assert writers == _WRITERS_TODAY, (
        f"the other markup writers stand only where they stand today: {writers}"
    )

    config = read(_ESLINTRC)
    extends = re.search(r"\bextends\s*:\s*\[([^\]]*)\]", config)
    assert extends is not None and re.search(
        r"['\"]plugin:svelte/recommended['\"]", extends.group(1)
    ), f"{_ESLINTRC} extends plugin:svelte/recommended"
    assert "no-at-html-tags" not in config, (
        f"{_ESLINTRC} names no-at-html-tags: the recommended config already "
        f"holds it as an error, so naming it can only turn it down"
    )
    disabled = [
        path for path in listed
        if re.search(r"eslint-disable[^\n]*no-at-html-tags", read(path))
    ]
    assert disabled == [], f"no source disables the rule: {disabled}"
    assert re.search(
        r"['\"]svelte/no-at-html-tags['\"]\s*:\s*['\"]error['\"]", read(_RECOMMENDED)
    ), "the installed plugin's recommended config holds svelte/no-at-html-tags as an error"


# ---------------------------------------------------------------------------
# MK9 -- tables
# ---------------------------------------------------------------------------
_TABLE_SOURCE = (
    "| Left | Centre | Right | Plain |\n"
    "|:-----|:------:|------:|-------|\n"
    "| a | b | c | d |\n"
    "| e | **f** | g | h |"
)


def test_mk9_a_table_is_a_labelled_focusable_region_with_column_headers_and_alignment():
    root = _dom(_render(_TABLE_SOURCE))
    regions = [element for element in root.iter() if element.get("role") == "region"]
    assert len(regions) == 1, f"one region per table: {[r.tag for r in regions]}"
    region = regions[0]
    assert region.get("tabindex") == "0", "the region takes the focus, so it scrolls by keyboard"
    assert (region.get("aria-label") or "").strip(), "the region is labelled"
    assert len(list(region.iter("table"))) == 1, "the region holds the table"

    headers = [(th.text(), th.get("scope"), th.get("data-align")) for th in region.iter("th")]
    assert headers == [
        ("Left", "col", "left"), ("Centre", "col", "center"),
        ("Right", "col", "right"), ("Plain", "col", None),
    ], f"each header cell is a column header carrying its alignment: {headers}"
    rows = [
        [(td.text(), td.get("data-align")) for td in tr.iter("td")]
        for tr in region.iter("tr") if list(tr.iter("td"))
    ]
    assert rows == [
        [("a", "left"), ("b", "center"), ("c", "right"), ("d", None)],
        [("e", "left"), ("f", "center"), ("g", "right"), ("h", None)],
    ], f"each cell carries its column's alignment: {rows}"
    assert [s.text() for s in region.iter("strong")] == ["f"], "a cell renders its inline nodes"

    style = re.search(r"<style[^>]*>(.*?)</style>", _source(_TABLE), re.DOTALL)
    assert style is not None, "MarkdownTable styles its own cells"
    for align in ("left", "center", "right"):
        rule = re.search(
            r"\[data-align=['\"]?" + align + r"['\"]?\][^{]*\{([^}]*)\}", style.group(1)
        )
        assert rule is not None and re.search(r"text-align\s*:\s*" + align, rule.group(1)), (
            f"a cell marked {align} is aligned {align}"
        )


# ---------------------------------------------------------------------------
# MK15 -- a hostile corpus
# ---------------------------------------------------------------------------
_TAB, _NUL, _LINE_SEPARATOR = chr(9), chr(0), chr(0x2028)
_HOSTILE = (
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    '<iframe src="javascript:alert(1)"></iframe>',
    '<object data="data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=="></object>',
    "<svg onload=alert(1)><circle r=1 /></svg>",
    '<a href="javascript:alert(1)">click</a>',
    '<a href="https://ok.test" onclick="alert(1)">click</a>',
    "[click](javascript:alert(1))",
    "[click](JaVaScRiPt:alert(1))",
    "[click](&#106;avascript:alert(1))",
    "[click](java" + _TAB + "script:alert(1))",
    "[click](" + _LINE_SEPARATOR + "javascript:alert(1))",
    "[click](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
    "[click](vbscript:msgbox(1))",
    "<javascript:alert(1)>",
    "[click][ref]\n\n[ref]: javascript:alert(1)",
    "![x](javascript:alert(1))",
    "![x](data:image/svg+xml;base64,PHN2ZyBvbmxvYWQ9YWxlcnQoMSk+)",
    '[x](https://ok.test "a\\" onmouseover=\\"alert(1)")',
    '[x](https://ok.test/"onmouseover="alert(1))',
    'https://ok.test/?q="><script>alert(1)</script>',
    "`<script>alert(1)</script>`",
    "```html\n<script>alert(1)</script>\n```",
    '```" onmouseover="alert(1)\nx\n```',
    "| <img src=x onerror=alert(1)> | b |\n|---|---|\n| <script>x</script> | c |",
    "# <iframe src=//evil.test></iframe>\n\n> <object data=x></object>\n\n"
    "- [x] <script>alert(1)</script>",
    '<style>*{display:none}</style><form action="javascript:alert(1)">'
    '<button formaction="javascript:alert(1)">go</button></form>',
    "<math><mtext><img src=x onerror=alert(1)></mtext></math>"
    "<!-- <script>alert(1)</script> -->",
    "&lt;script&gt;alert(1)&lt;/script&gt; &#60;img src=x onerror=alert(1)&#62;",
    '<base href="javascript:alert(1)//"><meta http-equiv="refresh" '
    'content="0;url=javascript:alert(1)">' + _NUL,
)
# The elements the renderer emits: anything else in its output came from
# the reply.
_EMITTED = frozenset({
    "#root", "div", "p", "h3", "h4", "h5", "h6", "figure", "figcaption", "pre", "code",
    "span", "button", "blockquote", "ol", "ul", "li", "table", "thead", "tbody", "tr",
    "th", "td", "hr", "br", "strong", "em", "del", "a", "svg", "path",
})
_LOADING = ("<script", "<img", "<iframe", "<object")
_URL_ATTRIBUTES = frozenset({
    "href", "src", "srcset", "action", "formaction", "xlink:href", "poster", "data",
    "background", "cite", "ping",
})
_HANDLER = re.compile(r"on[a-z]+")
_SCHEME = re.compile(r"([a-z][a-z0-9+.-]*):")


def _hostile_findings(html_text):
    """What in emitted HTML could load, run or leave: a loading tag, an
    element the renderer does not emit, an event handler attribute, a URL
    attribute whose scheme is a script or data one (read as a browser
    reads it, white space and control characters dropped), and a link that
    is not http, https or mailto."""
    findings = []
    lowered = html_text.lower()
    findings += [f"a {tag} tag" for tag in _LOADING if tag in lowered]
    for element in _dom(html_text).iter():
        if element.tag not in _EMITTED:
            findings.append(f"an element the renderer does not emit: <{element.tag}>")
        for name, value in element.attrs:
            if _HANDLER.fullmatch(name):
                findings.append(f"an event handler {name}= on <{element.tag}>")
            if name not in _URL_ATTRIBUTES:
                continue
            bare = "".join(char for char in (value or "") if char > " ").lower()
            scheme = _SCHEME.match(bare)
            if scheme and scheme.group(1) in ("javascript", "data", "vbscript"):
                findings.append(f"a {scheme.group(1)}: URL in {name}= on <{element.tag}>")
            if element.tag == "a" and name == "href" and not (
                scheme and scheme.group(1) in ("http", "https", "mailto")
            ):
                findings.append(f"a link to {value!r}")
    return findings


def test_mk15_a_hostile_corpus_emits_nothing_that_loads_or_runs():
    sample = (
        '<a href="JaVa' + _TAB + 'script:alert(1)" onclick="x">t</a>'
        '<img src="data:,x"><script>1</script><iframe></iframe><object></object>'
        '<a href="/relative">r</a><form></form>'
    )
    found = _hostile_findings(sample)
    assert {
        "a <script tag", "a <img tag", "a <iframe tag", "a <object tag",
        "an event handler onclick= on <a>", "a javascript: URL in href= on <a>",
        "a data: URL in src= on <img>", "a link to '/relative'",
        "an element the renderer does not emit: <form>",
    } <= set(found), f"the checker finds each kind in its sample: {found}"

    assert len(_HOSTILE) == 30, len(_HOSTILE)
    findings = {}
    for index, payload in enumerate(_HOSTILE):
        for streaming in (False, True):
            html_text = _render(payload, streaming)
            assert html_text.strip(), f"payload {index} renders something"
            found = _hostile_findings(html_text)
            if found:
                findings[(index, streaming)] = (payload, found)
    assert findings == {}, f"hostile payloads emitted: {findings}"


# ---------------------------------------------------------------------------
# MK16 -- a code block's Copy
# ---------------------------------------------------------------------------
def test_mk16_copy_writes_the_block_code_and_announces_copied_through_a_status_region():
    text = _source(_CODE_BLOCK)
    writes = re.findall(r"clipboard\.writeText\(([^)]*)\)", text)
    assert writes == ["node.code"], f"Copy writes the block's code and nothing else: {writes}"
    status = re.search(r"role=\"status\"[^>]*>\s*\{(\w+)\}\s*</", text)
    assert status is not None, "a status region shows the copy's outcome"
    name = status.group(1)
    copy = re.search(r"writeText\(node\.code\)(.*?)\n\t}", text, re.DOTALL)
    assert copy is not None and re.search(
        r"\b" + name + r"\s*=\s*['\"]Copied['\"]", copy.group(1)
    ), f"once the write resolves, the status region says Copied ({name})"
    # Copied is said in the write's own branch, after it resolved, and only
    # there; a refused write says so.
    attempt = re.search(
        r"\btry\s*\{(?P<body>[^{}]*?writeText\(node\.code\)[^{}]*)\}\s*catch\s*(?:\([^)]*\))?\s*\{"
        r"(?P<refused>[^{}]*)\}",
        text,
    )
    assert attempt is not None, "the write is attempted in a try with its own catch"
    said = r"\b" + name + r"\s*=\s*['\"]{}['\"]"
    body, refused = attempt.group("body"), attempt.group("refused")
    assert re.search(r"await\s+navigator\.clipboard\.writeText\(node\.code\)\s*;\s*" + name
                     + r"\s*=\s*['\"]Copied['\"]", body), (
        "Copied is said once the awaited write resolved, in its branch"
    )
    assert not re.search(said.format("Copied"), refused), "a refused write never says Copied"
    assert re.search(said.format("Copy failed"), refused), "a refused write says Copy failed"
    assert len(re.findall(said.format("Copied"), text)) == 1, "Copied is said in one place only"
    assert re.search(r"<TextButton\b[^>]*on:click=\{copy\}", text), "Copy is a primitive button"

    chat = _source(_CHAT_MESSAGE)
    assert re.findall(r"clipboard\.writeText\(([^)]*)\)", chat) == ["displayContent"], (
        "the message keeps its Copy of the whole raw reply, and copies no code block itself"
    )
    assert "extractCodeBlocks" not in chat and "copyCodeBlock" not in chat


# ---------------------------------------------------------------------------
# MK19 -- a reply through Markdown, a user message as plain text
# ---------------------------------------------------------------------------
_MIXED = "# Title\n\nSome **bold** and <b>raw</b>\n\n```py\nx = 1\n```"


def test_mk19_chat_message_renders_a_reply_through_markdown_and_a_user_message_as_plain_text():
    server = ssr()
    reply = _dom(server.render(_CHAT_MESSAGE, {
        "message": {"role": "assistant", "content": _MIXED, "id": 7, "model": "reply-model"},
    }).html)
    tags = [element.tag for element in reply.iter()]
    assert "h3" in tags and "strong" in tags and "figure" in tags, (
        f"a reply's markdown is rendered: {sorted(set(tags))}"
    )
    assert "b" not in tags, "raw HTML in a reply stays text"
    screen_reader = [(h.text(), h.get("class") or "") for h in reply.iter("h2")]
    assert len(screen_reader) == 1 and "sr-only" in screen_reader[0][1].split(), (
        f"the reply carries one screen-reader heading above its own: {screen_reader}"
    )
    assert tags.index("h2") < tags.index("h3")
    shown = reply.text()
    left = [markup for markup in ("# Title", "**bold**", "```") if markup in shown]
    assert left == [], f"the reply is rendered once, with none of its raw markup beside: {left}"

    user = _dom(server.render(_CHAT_MESSAGE, {
        "message": {"role": "user", "content": _MIXED, "id": 8},
    }).html)
    user_tags = {element.tag for element in user.iter()}
    assert not user_tags & {"h2", "h3", "strong", "figure", "pre", "code", "b"}, (
        f"a user message is not rendered as markdown: {sorted(user_tags)}"
    )
    assert _MIXED in user.text(), "a user message shows its text as written"

    # Any role but the assistant's is shown as written, with no reply heading.
    for role in ("system", "tool"):
        other = _dom(server.render(_CHAT_MESSAGE, {
            "message": {"role": role, "content": _MIXED, "id": 9},
        }).html)
        other_tags = {element.tag for element in other.iter()}
        assert not other_tags & {"h2", "h3", "strong", "figure", "pre", "code", "b"}, (
            f"a {role} message is not rendered as a reply: {sorted(other_tags)}"
        )
        assert _MIXED in other.text(), f"a {role} message shows its text as written"


# ---------------------------------------------------------------------------
# MK20 -- the renderer is total
# ---------------------------------------------------------------------------
# Nestings about 2 KB long, each past the tree's depth cap, with the marker
# each level spends and the element each level renders; and one deep enough
# that marked's own recursion gives up on it.
_DEEP = (
    (">" * 2000 + " x", ">", "blockquote"),
    ("- " * 1000 + "x", "-", "ul"),
    ("1. " * 683 + "x", "1.", "ol"),
    ("> " + "- " * 1000 + "x", "-", "ul"),
)
_BEYOND_MARKED = (">" * 3000 + " x", "- " * 3000 + "x")


def _region(root):
    """The rendered reply: the element carrying the renderer's root class."""
    found = [e for e in root.iter() if "oo-md" in (e.get("class") or "").split()]
    assert found, "the reply's region is rendered"
    return found[0]


def _kinds(nodes, kind):
    return sum(1 for node in _walk(nodes) if node["kind"] == kind)


@pytest.mark.parametrize("half", _WITH_SSR)
def test_mk20_the_renderer_is_total_and_shows_every_character(half):
    if half == "ssr":
        server = ssr()
        cases = _DEEP + tuple((source, source[0], "blockquote") for source in _BEYOND_MARKED[:1])
        for source, marker, tag in cases:
            html_text = server.render(_CHAT_MESSAGE, {
                "message": {"role": "assistant", "content": source, "id": 1},
            }).html
            region = _region(_dom(html_text))
            shown, nested = region.text(), len(list(region.iter(tag)))
            assert shown.count(marker) + nested == source.count(marker) and "x" in shown, (
                f"a reply nesting {source.count(marker)} levels renders and shows each "
                f"marker, as an element or as text: {nested} elements, "
                f"{shown.count(marker)} in text"
            )
        for role in ("assistant", "user"):
            html_text = server.render(_CHAT_MESSAGE, {
                "message": {"role": role, "content": None, "id": 2},
            }).html
            assert "undefined" not in html_text and "null" not in _dom(html_text).text(), (
                f"a {role} message with no content renders as no text"
            )
        return
    if half == "wiring":
        text = _source(_MARKDOWN)
        assert re.search(r"\btry\s*\{[^{}]*\btoTree\s*\(", text) and re.search(
            r"\bcatch\b", text
        ), "Markdown builds the tree inside a try: a failure falls back, never throws"
        assert re.search(r"\{#if\s+blocks\s*===\s*null\s*\}\s*<div\b[^>]*>\s*<PlainText\b", text), (
            "with no tree (no tokens, or a tree that failed), the reply is plain text"
        )
        assert re.search(r"typeof\s+source\s*===\s*['\"]string['\"]", text), (
            "Markdown reads a missing source as an empty one"
        )
        return

    limits = _node("limits", ("OO_TREE",))
    depth = limits["depth"]
    assert isinstance(depth, int) and 8 <= depth <= 64, f"a depth cap: {limits}"

    # marked's own recursion gives up past some depth: the lexer says the
    # reply is plain text, it never throws.
    thrown = _node("thrown", ("OO_TREE",), list(_BEYOND_MARKED) + ["a\n\n> b"])
    assert [case["value"] for case in thrown] == ["plain", "plain", "tokens"], (
        f"a source marked cannot lex is plain text, never an exception: {thrown}"
    )
    assert _node("thrown", ("OO_TREE",), [None])[0]["value"] == "tokens", (
        "a missing source lexes as an empty one"
    )

    # Nestings the lexer takes are cut at the cap: what lies deeper is its
    # source, as text, so every marker is shown.
    results = _node(
        "depths", ("OO_TREE", "OO_ENTITIES"),
        [source for source, _m, _t in _DEEP] + ["*" * 300 + "a" + "*" * 300],
    )
    kinds = {"blockquote": "blockquote", "ul": "ul", "ol": "ol"}
    for (source, marker, tag), result in zip(_DEEP, results):
        if result["plain"]:
            continue
        nodes = result["nodes"]
        assert result["depth"] <= depth + 2, f"the tree stops at the cap: {result['depth']}"
        assert _text(nodes).count(marker) + _kinds(nodes, kinds[tag]) == source.count(marker), (
            f"each of the {source.count(marker)} markers is a level or text"
        )
        assert _text(nodes).endswith("x")
    emphasis = results[-1]
    assert emphasis["plain"] or (
        emphasis["depth"] <= depth + 2 and "a" in _text(emphasis["nodes"])
    ), f"nested emphasis stops at the cap: {emphasis.get('depth')}"
    assert any(not result["plain"] for result in results[:len(_DEEP)]), (
        "the lexer takes at least one nesting, so the cap is what holds it"
    )

    # The cap on tokens injected a hundred levels deep: 'depth' levels, then
    # the rest as its source.
    capped = _node("capped", ("OO_TREE", "OO_ENTITIES"), [100])[0]
    assert capped["depth"] <= depth + 2, f"injected nesting stops at the cap: {capped['depth']}"
    assert _kinds(capped["nodes"], "blockquote") == depth
    assert _text(capped["nodes"]) == ">" * (100 - depth) + " x", (
        f"the levels past the cap are shown as their source: {_text(capped['nodes'])!r}"
    )


# ---------------------------------------------------------------------------
# MK21 -- lexing a reply costs a bounded time
# ---------------------------------------------------------------------------
def _repeat(motif, n, prefix="", suffix=""):
    return {"spec": {"kind": "repeat", "motif": motif, "prefix": prefix, "suffix": suffix}, "n": n}


# Replies marked lexes in super-linear time, each at a size it takes more
# than a second to lex on this machine (links: cubic in a run without white
# space; emphasis and definitions: quadratic).
_PATHOLOGICAL = (
    _repeat("[](", 2000),
    _repeat("[a](", 2000),
    _repeat("f(a[i](b", 800),
    _repeat("[1](", 1000),
    _repeat("a[b](c", 1000),
    _repeat("](a[", 2000),
    _repeat("![](", 1000),
    _repeat(" word", 4000, prefix="[](" * 300),
    _repeat("*a ", 8000),
    _repeat("_a ", 8000),
    _repeat("**a ", 8000),
    {"spec": {"kind": "nest", "open": "*", "middle": "a", "close": "*"}, "n": 8000},
    {"spec": {"kind": "refs"}, "n": 4000},
)
# Replies a person writes: they are lexed, never shown as text.
_REALISTIC = (
    _repeat("- **Name**: a description with `code` and a [link](https://ok.test/a_(b)).\n", 600),
    _repeat("2026-09-26 12:00:01 [INFO] worker(3) matched src/*.ts and _private_name in 12 ms\n", 400),
    _repeat("| a | **b** | [c](https://ok.test) |\n", 400, prefix="| x | y | z |\n|---|---|---|\n"),
    _repeat("Some *emphasis*, some __strong__, a ~~strike~~ and a footnote [1].\n\n", 800),
    _repeat("See [the docs][ref] and [the guide][ref].\n", 300, suffix="\n[ref]: https://ok.test\n"),
)
# The families at the edge of the limit: the largest size the estimate
# accepts must lex fast.
_BOUNDARY = (
    {"kind": "repeat", "motif": "[](", "cap": 65536},
    {"kind": "repeat", "motif": "](a[", "cap": 65536},
    {"kind": "repeat", "motif": "![](", "cap": 65536},
    {"kind": "repeat", "motif": "f(a[i](b", "cap": 65536},
    {"kind": "repeat", "motif": "[a]( (x ", "cap": 65536},
    {"kind": "repeat", "motif": "*a ", "cap": 65536},
    {"kind": "repeat", "motif": "_a ", "cap": 65536},
    {"kind": "repeat", "motif": "~~a ", "cap": 65536},
    {"kind": "nest", "open": "*", "middle": "a", "close": "*", "cap": 65536},
    {"kind": "refs", "cap": 65536},
)
_TRACKED_MARKDOWN = "git ls-files -z -- *.md"


def _tracked_markdown():
    listed = subprocess.run(
        _TRACKED_MARKDOWN.split(), cwd=REPO, capture_output=True, check=True,
    ).stdout.decode("utf-8").split("\0")
    return [str(REPO / path) for path in listed if path]


@pytest.mark.parametrize("half", ("node", "boundary", "guard", "wiring", "ssr"))
def test_mk21_lexing_a_reply_costs_a_bounded_time(half):
    if half == "wiring":
        text = _source(_MARKDOWN)
        assert _imports(text, "guardSlowLex", "$lib/markdown/frame")
        assert re.search(r"\bcreateFrameLexer\s*\(\s*guardSlowLex\s*\(\s*lexMarkdown\s*\)", text), (
            "Markdown lexes through the slow-lex guard, around the tree's bounded lexer"
        )
        tree = read(_TREE)
        assert re.search(
            r"export function lexMarkdown\([^)]*\)[^{]*\{(?:(?!\n\}).)*?lexWork\(",
            tree, re.DOTALL,
        ), "the tree's lexer estimates the work before it lexes"
        return
    if half == "guard":
        report = _node("guard", ("OO_FRAME",))
        assert report["fast"] == "tokens of fast", "a fast lex passes its tokens through"
        assert report["slow"] is None, "a lex slower than the limit gives no tokens: text instead"
        assert report["grown"] is None and report["lexedInStream"] == ["fast", "slow"], (
            f"once slow, the rest of its stream is not lexed: {report['lexedInStream']}"
        )
        assert report["memoryAfterStream"] == ["slow and more"], (
            f"the stream's latest source is remembered, once: {report['memoryAfterStream']}"
        )
        assert report["remembered"] is None, "a remembered source is not lexed when shown again"
        assert report["fresh"] == "tokens of fresh"
        assert report["lexed"] == ["fast", "slow", "fresh"], report["lexed"]
        remembers = report["remembers"]
        assert remembers == 16 and report["memory"] == [f"s{i}" for i in range(20 - remembers, 20)], (
            f"the memory keeps the last {remembers} slow replies: {report['memory']}"
        )
        assert report["estimated"] is None and report["afterEstimated"] == "tokens of next", (
            "the lexer's own refusal passes through and does not stick"
        )
        assert report["defaults"] == "tokens of x", "by default the guard reads the page's clock"
        return
    if half == "boundary":
        results = _node("boundary", ("OO_TREE",), list(_BOUNDARY))
        for spec, result in zip(_BOUNDARY, results):
            assert not result.get("capped"), f"{spec}: the estimate reaches the limit: {result}"
            assert result["lexed"] and result["above"], (
                f"{spec}: the largest size under the limit is lexed and the next is not: {result}"
            )
            assert result["ms"] < 600, (
                f"{spec}: what the estimate lets through lexes fast: {result['ms']:.0f} ms "
                f"for {result['length']} characters"
            )
        return
    if half == "ssr":
        source = "[](" * 2000
        start = time.monotonic()
        root = _dom(_render(source))
        took = time.monotonic() - start
        plain = [p for p in root.iter("p") if "oo-md-plain" in (p.get("class") or "").split()]
        assert len(plain) == 1 and plain[0].text() == source, (
            "a reply over the limit is shown as plain text, every character"
        )
        assert not list(root.iter("a")) and took < 2.0, f"it renders at once: {took:.2f}s"
        return

    limits = _node("limits", ("OO_TREE",))
    assert isinstance(limits["work"], (int, float)) and limits["work"] > 0
    results = _node("guarded", ("OO_TREE",), list(_PATHOLOGICAL))
    for case, result in zip(_PATHOLOGICAL, results):
        assert result["plain"] and result["ms"] < 1000, (
            f"{case}: a reply marked would take long to lex is not lexed: {result}"
        )
    realistic = _node(
        "guarded", ("OO_TREE",),
        list(_REALISTIC) + [{"file": path} for path in _tracked_markdown()],
    )
    assert len(realistic) > len(_REALISTIC) + 10, "the repository's markdown is read"
    refused = [
        (case, result) for case, result in zip(
            list(_REALISTIC) + _tracked_markdown(), realistic,
        ) if result["plain"]
    ]
    assert refused == [], f"what a person writes is lexed, never shown as text: {refused}"


# ---------------------------------------------------------------------------
# MK22 -- the tree is built in linear time
# ---------------------------------------------------------------------------
_NEWLINES = 100000


def test_mk22_building_the_tree_is_linear_in_a_run_of_newlines():
    cases = [
        [_repeat("\n", _NEWLINES, prefix="<pre>\n", suffix="x</pre>"), False],
        [_repeat("\n", _NEWLINES, prefix="<!--\n", suffix="x -->"), False],
        [_repeat("\n", _NEWLINES, prefix="para\n\n```py\n", suffix="x"), True],
        [_repeat("\n", _NEWLINES, prefix="para\n\n```py\n", suffix="x\n```"), True],
    ]
    results = _node("treeTime", ("OO_TREE", "OO_ENTITIES"), cases)
    for (case, streaming), result in zip(cases, results):
        assert not result["plain"] and result["blocks"] >= 1, f"{case}: it is lexed: {result}"
        assert result["ms"] < 250, (
            f"{case['spec']['prefix']!r} with {_NEWLINES} newlines, streaming={streaming}: "
            f"the tree builds in linear time: {result['ms']:.0f} ms"
        )


# ---------------------------------------------------------------------------
# MK23 -- the streaming caret
# ---------------------------------------------------------------------------
def _carets(root):
    """The streaming caret: the element that blinks (no thinking is shown in
    these cases, whose own caret blinks too)."""
    return [
        e for e in root.iter("span") if "animate-cursor-blink" in (e.get("class") or "").split()
    ]


def _streaming(server, source, streaming=True, role="assistant"):
    return _dom(server.render(_CHAT_MESSAGE, {
        "message": {"role": role, "content": "" if streaming else source, "id": 3},
        "isStreaming": streaming, "streamContent": source if streaming else "",
    }).html)


@pytest.mark.parametrize("half", ("ssr", "wiring"))
def test_mk23_the_streaming_caret_is_drawn_after_the_last_character(half):
    if half == "wiring":
        caret = _source(_CARET)
        style = re.search(r"<style[^>]*>(.*?)</style>", caret, re.DOTALL)
        rule = style and re.search(r"\.oo-md-caret\s*\{([^}]*)\}", style.group(1))
        assert rule and re.search(r"(?<![\w-])width\s*:\s*2px\s*;", rule.group(1)), (
            "the caret is a 2 px bar"
        )
        assert re.search(r'class="[^"]*\boo-md-caret\b[^"]*\banimate-cursor-blink\b', caret), (
            "the caret blinks through the class the reduced-motion rule stills"
        )
        assert 'aria-hidden="true"' in caret, "the caret is not read aloud"
        app = read("frontend/src/app.css")
        motion = re.search(
            r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{(.*?)\n\}", app, re.DOTALL,
        )
        assert motion and re.search(r"animation-iteration-count:\s*1\b", motion.group(1)) and re.search(
            r"animation-duration:\s*0\.01ms", motion.group(1)
        ), "under reduced motion every animation runs once, at once: the caret stands still"
        chat = _source(_CHAT_MESSAGE)
        assert re.search(r"<Markdown\b[^>]*\bcaret=\{isStreaming\}", chat), (
            "ChatMessage hands the caret to the reply while it streams"
        )
        assert len(re.findall(r"animate-cursor-blink", chat)) == 1, (
            "ChatMessage draws no caret of its own after the reply (the thinking one stays)"
        )
        return

    server = ssr()
    for source, holder, ending in (
        ("Hello **world** and more", "p", "and more"),
        ("- one\n- two", "li", "two"),
        ("> quoted\n> text", "p", "text"),
        ("para\n\n```py\nx = 1", "pre", "x = 1"),
        ("[](" * 2000, "p", "[]("),
    ):
        root = _streaming(server, source)
        found = _carets(root)
        assert len(found) == 1, f"{source[:20]!r}: one caret while streaming: {len(found)}"
        caret = found[0]
        ancestor = caret.parent
        while ancestor is not None and ancestor.tag != holder:
            ancestor = ancestor.parent
        assert ancestor is not None and ancestor.text().endswith(ending), (
            f"{source[:20]!r}: the caret is inside the last {holder}, after {ending!r}"
        )
        assert caret.parent.children[-1] is caret, (
            f"{source[:20]!r}: nothing follows the caret in its element"
        )
        region = _region(root)
        blocks = [child for child in region.children if isinstance(child, _Element)]
        inside = caret
        while inside is not None and inside.parent is not region:
            inside = inside.parent
        assert inside is not None and inside is blocks[-1], (
            f"{source[:20]!r}: the caret is inside the reply's last block"
        )
        assert caret.get("aria-hidden") == "true", "the caret is not read aloud"
    assert len(_carets(_streaming(server, ""))) == 1, "an empty stream shows the caret alone"
    assert _carets(_streaming(server, "Hello", streaming=False)) == [], "no caret once it is done"


# ---------------------------------------------------------------------------
# MK24 -- plain text collapses by whole lines
# ---------------------------------------------------------------------------
def _numbered(count):
    return "\n".join(f"line {i}" for i in range(count))


@pytest.mark.parametrize("half", _WITH_SSR)
def test_mk24_plain_text_collapses_by_whole_lines_and_a_first_block_is_never_cut(half):
    if half == "wiring":
        plain = _source(_PLAIN)
        assert _imports(plain, "collapseLines", "$lib/markdown/collapse")
        assert re.search(r"\bcollapseLines\s*\(", plain)
        assert ".split(" not in plain and ".slice(" not in plain, "PlainText cuts nothing itself"
        chat = _source(_CHAT_MESSAGE)
        assert re.search(r"\$:\s*isReply\s*=\s*message\.role\s*===\s*['\"]assistant['\"]", chat), (
            "a reply is a message of the assistant's role"
        )
        assert re.search(r"\{#if\s+isReply\s*\}\s*<Markdown\b[^>]*>\s*\{:else\}\s*<PlainText\b", chat), (
            "ChatMessage renders a reply through Markdown and any other message as plain text"
        )
        assert re.search(r"<PlainText\b", _source(_MARKDOWN)), (
            "a reply that is not lexed is plain text too"
        )
        return
    if half == "ssr":
        server = ssr()
        text = _numbered(600)
        for role in ("user", "system"):
            root = _dom(server.render(_CHAT_MESSAGE, {
                "message": {"role": role, "content": text, "id": 5},
            }).html)
            shown = root.text()
            assert "line 19" in shown and "line 20" not in shown, (
                f"a long {role} message shows its first 20 lines, and no more"
            )
            plain = [p for p in root.iter("p") if "oo-md-plain" in (p.get("class") or "").split()]
            assert len(plain) == 1 and plain[0].text() == _numbered(20), (
                f"a long {role} message shows its first 20 lines"
            )
            toggles = [b for b in root.iter("button") if b.get("aria-expanded") is not None]
            assert [(b.text(), b.get("aria-expanded")) for b in toggles] == [
                ("Show the rest (580 more lines)", "false"),
            ], f"one toggle names the lines it hides: {[b.text() for b in toggles]}"
            held = [e for e in root.iter() if e.get("id") == toggles[0].get("aria-controls")]
            assert len(held) == 1 and plain[0] in list(held[0].iter("p")), (
                "the toggle controls the element that holds the text"
            )
        # A reply that is one long block is shown whole: collapse keeps whole
        # blocks, and never cuts the first.
        reply = _dom(server.render(_CHAT_MESSAGE, {
            "message": {"role": "assistant", "content": text, "id": 6},
        }).html)
        region = _region(reply)
        assert "line 599" in region.text() and "line 0" in region.text(), (
            "a reply that is one block of 600 lines is shown whole"
        )
        assert not [b for b in reply.iter("button") if b.get("aria-expanded") is not None], (
            "a reply that is one block has nothing to show the rest of"
        )
        # A reply shown as text collapses by lines too.
        slow = "\n".join(["[](" * 20] * 600)
        root = _dom(server.render(_CHAT_MESSAGE, {
            "message": {"role": "assistant", "content": slow, "id": 7},
        }).html)
        toggles = [b.text() for b in root.iter("button") if b.get("aria-expanded") is not None]
        assert toggles == ["Show the rest (580 more lines)"], (
            f"a long reply shown as text collapses by lines: {toggles}"
        )
        return

    lines = _numbered(600)
    budget = {"threshold": 500, "keep": 20, "expanded": False}
    long, at_line, never, expanded, small = _node("lines", ("OO_COLLAPSE",), [
        [lines, budget],
        [_numbered(500), budget],
        [lines, dict(budget, threshold=0)],
        [lines, dict(budget, expanded=True)],
        ["a\nb\nc", {"threshold": 2, "keep": 1, "expanded": False}],
    ])
    assert long == {"shown": _numbered(20), "hidden": 580, "collapsible": True}, (
        f"a text over the threshold shows its first lines and counts the rest: {long['hidden']}"
    )
    assert at_line == {"shown": _numbered(500), "hidden": 0, "collapsible": False}, (
        "a text at the threshold is shown whole"
    )
    assert never == {"shown": lines, "hidden": 0, "collapsible": False}, "a zero threshold never collapses"
    assert expanded == {"shown": lines, "hidden": 0, "collapsible": True}, (
        "an expanded text is shown whole and can collapse again"
    )
    assert small == {"shown": "a", "hidden": 2, "collapsible": True}


# ---------------------------------------------------------------------------
# MK25 -- the renderer's type never computes under 12 px
# ---------------------------------------------------------------------------
_FONT_SIZE_VALUE = re.compile(r"(?<![\w-])font-size\s*:\s*([^;{}]+)")
_TYPE_SAMPLE = (
    "a { font-size: 0.9em; } b { font-size: 11px; } c { font-size: calc(1em - 3px); } "
    "d { font-size: max(12px, 0.9em); } e { font-size: var(--oo-text-sm); } "
    "f { font-size: 1.25em; } g { font-size: 14px; } h { font-size: 0.7rem; }"
)
_TYPED = (*_RENDERERS, "frontend/src/lib/ds/TextButton.svelte")


def _under_twelve(value):
    """True unless the size is shown to compute at 12 px or above: a text
    token, an absolute size of at least 12 px (0.75rem), a relative size of
    at least the parent's, or a size floored at 12 px."""
    value = value.strip()
    if re.fullmatch(r"var\(--oo-text-[\w-]+\)", value):
        return False
    if re.fullmatch(r"max\(\s*12px\s*,[^()]+\)", value):
        return False
    size = re.fullmatch(r"(\d*\.?\d+)(px|rem|em)", value)
    if size is None:
        return True
    number, unit = float(size.group(1)), size.group(2)
    return number < {"px": 12.0, "rem": 0.75, "em": 1.0}[unit]


def test_mk25_the_renderers_type_never_computes_under_twelve_pixels():
    sample = [value for value in _FONT_SIZE_VALUE.findall(_TYPE_SAMPLE) if _under_twelve(value)]
    assert [value.strip() for value in sample] == ["0.9em", "11px", "calc(1em - 3px)", "0.7rem"], (
        f"the census finds each size that can compute under 12 px: {sample}"
    )
    found = {}
    for path in _TYPED:
        text = _source(path) if path in _RENDERERS else read(path)
        sizes = _FONT_SIZE_VALUE.findall(text)
        low = [value.strip() for value in sizes if _under_twelve(value)]
        if low:
            found[path] = low
        assert not re.search(r"(?<![\w-])font\s*:(?!\s*inherit\b)", text), (
            f"{path} sets no size through the font shorthand"
        )
    assert found == {}, f"each font size is a token, at least 12 px, or floored at 12 px: {found}"


# ---------------------------------------------------------------------------
# MK26 -- task boxes and rules
# ---------------------------------------------------------------------------
def _rule(style, selector):
    found = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", style)
    assert found is not None, f"MarkdownNode styles {selector}"
    return found.group(1)


def test_mk26_a_task_box_sits_beside_its_text_and_a_rule_is_drawn():
    for source in ("- [x] done\n- [ ] open", "- [x] done\n\n- [ ] open"):
        root = _dom(_render(source))
        items = [li for li in root.iter("li") if "oo-md-task" in (li.get("class") or "").split()]
        assert len(items) == 2, f"{source!r}: two task items: {len(items)}"
        for item in items:
            kids = [child for child in item.children if isinstance(child, _Element)]
            assert [kid.tag for kid in kids] == ["span", "svg", "div"], (
                f"{source!r}: the reader's word, the box, then one body: {[k.tag for k in kids]}"
            )
            assert "oo-md-task-body" in (kids[2].get("class") or "").split()
            assert kids[2].text().strip() in ("done", "open"), "the item's text is in its body"
    style = re.search(r"<style[^>]*>(.*?)</style>", _source(_NODE), re.DOTALL).group(1)
    task = _rule(style, ".oo-md-task")
    assert re.search(r"display\s*:\s*flex\s*;", task), "a task item lays its box and body in a row"
    assert re.search(r"flex\s*:\s*1\s+1\s+auto\s*;", _rule(style, ".oo-md-task-body")), (
        "the body takes the rest of the row"
    )

    root = _dom(_render("above\n\n---\n\nbelow"))
    assert [hr.get("class") for hr in root.iter("hr")] and all(
        "oo-md-rule" in (hr.get("class") or "").split() for hr in root.iter("hr")
    ), "a thematic break renders as the rule"
    rule = _rule(style, ".oo-md-rule")
    assert re.search(r"background-color\s*:\s*var\(--oo-[\w-]+\)\s*;", rule), (
        "the rule is drawn in a tone of the palette"
    )
    sizes = dict(re.findall(r"(?<![\w-])(width|height)\s*:\s*(\d+)px\s*;", rule))
    assert set(sizes) == {"width", "height"} and all(int(v) > 0 for v in sizes.values()), (
        f"the rule is a mark with a size, not an empty gap: {sizes}"
    )
