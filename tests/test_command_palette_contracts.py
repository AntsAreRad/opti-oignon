#!/usr/bin/env python3
"""Contracts for the command palette: one palette, opened by Ctrl+K through its store.

The palette reaches every place and every action of the interface from the
keyboard: the destinations of both spaces, the commands (every default
shortcut is one), every settings group that has a page, and the
conversations, which the server searches. It is one dialog, the ds Modal,
mounted once by the layout of both spaces, and opened through its store:
nothing on the way to it reads the page's markup.

The modules and the names the contracts read:

  * ``lib/palette/rank.ts`` -- ``rankPalette(items, query, limit)``: the
    groups to list for the words, each ``{id, label, items}``. An option's
    label meets the words exactly, at its start (prefix), at the start of a
    later word (word prefix), or by its letters in order (subsequence), in
    that order of rank; its keywords (``keywords``) count by the first three
    only, after the label's and before letters in order; an option its
    source found for these words (``found``: the server's search, which
    reads messages too) is kept after every other match; case and accents
    are ignored; options that tie keep the order
    they were given in; each group holds at most ``limit`` options
    (``GROUP_LIMIT`` by default); the groups follow the rank of their best
    option, ties in ``GROUP_ORDER``, each labelled from ``GROUP_LABELS``.
    With no words it lists every destination, then the recent chats, and
    nothing else.
  * ``lib/palette/commands.ts`` -- the registry: ``COMMANDS``, each with an
    ``id``, a ``label``, its ``keywords``, its default ``binding`` and a
    ``when`` that says, for a context (``space``, the conversation on
    screen as ``chatId``, whether a reply is being written as
    ``streaming``, and whether the palette or the shortcuts' help is open),
    why it cannot run there; ``reasonFor(command, context)``;
    ``commandOptions(commands, context)``, every command as an option, the
    ones that cannot run disabled with their ``reason``;
    ``defaultShortcuts(commands)``, the bindings the shortcut handler
    starts from; ``bindingLabel(binding)``.
  * ``lib/palette/sources.ts`` -- the options of each kind:
    ``destinationOptions(visible)``, ``settingOptions(index, groups)``
    (the settings search's index, with each group's synonyms) and
    ``chatOptions(chats, chatsHref, found)``.
  * ``lib/palette/conversationSource.ts`` --
    ``createConversationSource(fetcher, onChange, {delay, schedule})``: a
    source whose ``search(words)`` asks the server (``q`` and
    ``SOURCE_LIMIT``) once the reader pauses, aborts the request before it,
    drops any answer that is not the latest, and whose ``close()`` aborts
    the request in flight and drops what is still to come. Its state is
    ``{query, hits, asking, error}``.
  * ``lib/palette/run.ts`` -- ``HANDLERS``, what each command does, and
    ``runCommand(id)``, which refuses a command its context refuses before
    any handler runs; the shortcut handler and the palette both run
    commands through it. ``contextAt`` reads where the reader is, whether
    the page is in a space (``shell``) and who asks (``from``: a key, or
    the palette).
  * ``lib/palette/listing.ts`` -- ``activeFor(flat, pick)``,
    ``stepActive(flat, active, move)``, ``optionKey`` and
    ``optionDomId``: the active option held by its identity;
    ``statusFor(state)`` and ``createStatusLine(speak, {schedule,
    immediate})``: the status line and its pace.
  * ``lib/stores/palette.ts`` -- ``palette`` (open, and the words it opens
    with), ``openPalette(words)`` and ``closePalette()``.
  * ``lib/components/palette/CommandPalette.svelte`` -- the dialog.

  * NV11 -- Ctrl+K opens the palette through its store: the command bound
    to it is the palette's, its handler opens the store, the palette shows
    what the store says and is mounted once, by the layout of both spaces;
    nothing in the files a shortcut runs through (the handler, the layout
    that mounts it, the palette's modules and runner, its store, and
    whatever listens for an event they send) reads the page's markup. The
    theme's own lookup of its colour meta tag in the page's head
    (``lib/theme/apply.ts``, which the design token contracts pin) is
    outside this census.
  * NV12 -- the palette is a combobox that controls a listbox, the active
    option named by ``aria-activedescendant``, the options in labelled
    groups, hosted in the ds Modal; and the dialog holds Stop all in every
    state, so the stop is one click away while the palette covers the page.
  * NV13 -- the ranking (the order of rank, what is kept, ties, the limit,
    the order of the groups, the empty palette), and the palette ranks with
    it and with nothing of its own.
  * NV14 -- the sources cover the visible, ready destinations (the
    componion dropped while hidden), the fifty groups of the catalog that
    are not retired (an embedded one through its host) and every command;
    the conversation source sends ``q``, drops stale answers and aborts on
    close; and the palette builds its options from them, its request
    carrying the abort signal down to ``fetch``.
  * NV15 -- each of the nine default shortcuts is a command of the registry
    with its binding, and in each space it has a handler or a reason; the
    shortcut handler starts from the registry, keeps the server's own
    bindings, and runs every action through the runner.
  * NV16 -- a command that cannot run here (Export with no chat, Stop this
    reply with none being written, Send outside a chat) is listed disabled
    with its reason, never hidden, and never run.
  * NV17 -- the sidebar's Search entry, expanded and on the rail, opens the
    palette, and a phone reaches it through the drawer, whose content is
    the sidebar. The phone's own More sheet comes with the phone's
    layout, whose contracts pin that it opens the palette.
  * NV26 -- a key pressed while the palette or the list of shortcuts is
    open runs only what closes them (Ctrl+Enter in the palette's field no
    longer sends the draft under it); the palette judges its commands from
    inside and runs the one chosen in that context once closed; outside
    both spaces only the theme and closing a dialog run, and the palette,
    unmounted, shuts its store.
  * NV27 -- the active option is held by its identity: the reader's pick
    stays active however the list is ranked again, otherwise the first
    option that can run; ids in the page follow the identity; opening a
    disabled option says why.
  * NV28 -- the status line says how many results are listed once the
    list settles, that nothing matches, or that a slow search is still
    running, paced so a burst of keys is read once.
  * NV29 -- Stop all sits in the dialog's head, in no block, before the
    field at every width, a 44 px target on a phone; on a touch screen the
    close button is 44 px, the field's type does not make a phone zoom,
    and a disabled command's reason wraps whole.
  * NV30 -- the keys the palette's commands and the sidebar's hint show
    are the reader's own, from one store the shortcut handler writes.
  * NV31 -- the Stop all entry is found by the words of an emergency and
    opens the stop control's confirmation; the palette never stops
    anything itself.
  * NV32 -- the notification history is a command: it goes to
    Preferences and opens the history through its store.
  * NV33 -- the palette finds a settings group by the words the settings
    search reads (description, page, old section), and when the server
    found more chats than the group shows, a last entry opens the chats
    index on the same words.
  * NV34 -- the palette's source follows the palette through a pure
    follower; the shortcut handler sends no event and picks out no command
    of its own; the palette hides no disabled command; every text of the
    palette and of the list of shortcuts is drawn in a text ink.

Every census carries a standing positive fixture, a sample it must find, so
a probe gone blind turns red instead of reading a clean zero. The Node
halves run the pure modules through ``tests/_frontend.run_ts``; each is
paired with a wiring half that reads the files using them. The rendered
halves prove what the templates emit when compiled for the server (the app
is client-rendered); what a browser does (focus returning to the opener,
the arrows, Home and End, a screen reader reading the active option, the
latency of a search on a real store) is owed to the machine.

Local-only (the public distribution ships no tests). Needs Node >= 22.6;
without it the helpers raise, and so do the contracts.
"""

import json
import re
import sys
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_navigation_contracts as _nav  # noqa: E402
from _frontend import REPO, files, read, run_ts, ssr  # noqa: E402
from test_ui_primitives_contracts import _dom  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file. The first rendered contract starts the session's
# server when this suite runs alone.
BUDGET_S = {
    "test_nv11_ctrl_k_opens_the_palette_through_its_store[query]": 1.0,
    "test_nv11_ctrl_k_opens_the_palette_through_its_store[store]": 2.0,
    "test_nv12_the_palette_is_a_combobox_over_a_grouped_listbox_in_the_modal[aria]": 6.0,
    "test_nv12_the_palette_is_a_combobox_over_a_grouped_listbox_in_the_modal[stop]": 1.0,
    "test_nv13_the_palette_ranks_exact_then_prefix_then_word_then_letters[tiers]": 2.0,
    "test_nv13_the_palette_ranks_exact_then_prefix_then_word_then_letters[stable]": 2.0,
    "test_nv13_the_palette_ranks_exact_then_prefix_then_word_then_letters[limit]": 2.0,
    "test_nv13_the_palette_ranks_exact_then_prefix_then_word_then_letters[groups]": 2.0,
    "test_nv13_the_palette_ranks_exact_then_prefix_then_word_then_letters[empty]": 2.0,
    "test_nv13_the_palette_ranks_exact_then_prefix_then_word_then_letters[wiring]": 1.0,
    "test_nv14_the_sources_cover_every_place_command_group_and_chat[sources]": 2.0,
    "test_nv14_the_sources_cover_every_place_command_group_and_chat[conversations]": 2.0,
    "test_nv14_the_sources_cover_every_place_command_group_and_chat[wiring]": 1.0,
    "test_nv15_every_default_shortcut_is_a_command_with_a_handler_or_a_reason[registry]": 2.0,
    "test_nv15_every_default_shortcut_is_a_command_with_a_handler_or_a_reason[wiring]": 2.0,
    "test_nv16_a_command_that_cannot_run_here_is_listed_disabled_with_its_reason[node]": 2.0,
    "test_nv16_a_command_that_cannot_run_here_is_listed_disabled_with_its_reason[wiring]": 2.0,
    "test_nv17_the_sidebar_search_entry_opens_the_palette[handler]": 1.0,
    "test_nv17_the_sidebar_search_entry_opens_the_palette[forms]": 1.0,
    "test_nv26_a_key_runs_nothing_behind_an_open_dialog_and_nothing_outside_the_shell[node]": 2.0,
    "test_nv26_a_key_runs_nothing_behind_an_open_dialog_and_nothing_outside_the_shell[wiring]": 1.0,
    "test_nv27_the_active_option_is_the_readers_pick_while_listed_else_the_first_that_can_run[node]": 2.0,
    "test_nv27_the_active_option_is_the_readers_pick_while_listed_else_the_first_that_can_run[wiring]": 1.0,
    "test_nv27_the_active_option_is_the_readers_pick_while_listed_else_the_first_that_can_run[rendered]": 1.0,
    "test_nv28_the_palette_says_how_many_results_it_lists[node]": 2.0,
    "test_nv28_the_palette_says_how_many_results_it_lists[rendered]": 1.0,
    "test_nv29_stop_all_heads_the_palette_at_every_width_and_every_moment[head]": 1.0,
    "test_nv29_stop_all_heads_the_palette_at_every_width_and_every_moment[desktop]": 1.0,
    "test_nv29_stop_all_heads_the_palette_at_every_width_and_every_moment[phone]": 1.0,
    "test_nv29_stop_all_heads_the_palette_at_every_width_and_every_moment[touch]": 1.0,
    "test_nv30_the_keys_the_palette_and_the_sidebar_show_are_the_readers_own[node]": 2.0,
    "test_nv30_the_keys_the_palette_and_the_sidebar_show_are_the_readers_own[wiring]": 1.0,
    "test_nv31_the_palette_offers_stop_all_and_never_stops_by_itself[node]": 2.0,
    "test_nv31_the_palette_offers_stop_all_and_never_stops_by_itself[wiring]": 1.0,
    "test_nv31_the_palette_offers_stop_all_and_never_stops_by_itself[rendered]": 1.0,
    "test_nv32_the_notification_history_is_a_command_of_the_palette[registry]": 2.0,
    "test_nv32_the_notification_history_is_a_command_of_the_palette[wiring]": 1.0,
    "test_nv33_the_palette_finds_what_the_settings_search_and_the_chats_index_find[settings]": 2.0,
    "test_nv33_the_palette_finds_what_the_settings_search_and_the_chats_index_find[chats]": 2.0,
    "test_nv33_the_palette_finds_what_the_settings_search_and_the_chats_index_find[wiring]": 1.0,
    "test_nv34_the_palette_follows_its_source_the_keys_run_the_runner_and_nothing_hides[follow]": 2.0,
    "test_nv34_the_palette_follows_its_source_the_keys_run_the_runner_and_nothing_hides[follows]": 1.0,
    "test_nv34_the_palette_follows_its_source_the_keys_run_the_runner_and_nothing_hides[handler]": 1.0,
    "test_nv34_the_palette_follows_its_source_the_keys_run_the_runner_and_nothing_hides[hidden]": 1.0,
    "test_nv34_the_palette_follows_its_source_the_keys_run_the_runner_and_nothing_hides[inks]": 1.0,
}

_SRC = "frontend/src"
_ROUTES = f"{_SRC}/routes"
_PALETTE_DIR = f"{_SRC}/lib/palette"
_RANK = f"{_PALETTE_DIR}/rank.ts"
_COMMANDS = f"{_PALETTE_DIR}/commands.ts"
_SOURCES = f"{_PALETTE_DIR}/sources.ts"
_CONVERSATION_SOURCE = f"{_PALETTE_DIR}/conversationSource.ts"
_RUN = f"{_PALETTE_DIR}/run.ts"
_LISTING = f"{_PALETTE_DIR}/listing.ts"
_SHORTCUT_KEYS = f"{_SRC}/lib/stores/shortcutKeys.ts"
_NOTIFICATION_STORE = f"{_SRC}/lib/stores/notificationCenter.ts"
_NOTIFICATION_CENTER = f"{_SRC}/lib/components/ui/NotificationCenter.svelte"
_PALETTE_STORE = f"{_SRC}/lib/stores/palette.ts"
_PALETTE = f"{_SRC}/lib/components/palette/CommandPalette.svelte"
_SHORTCUTS = f"{_SRC}/lib/components/ui/KeyboardShortcuts.svelte"
_ROOT_LAYOUT = f"{_ROUTES}/+layout.svelte"
_APP_LAYOUT = f"{_ROUTES}/(app)/+layout.svelte"
_SIDEBAR = f"{_SRC}/lib/components/layout/Sidebar.svelte"
_STOP_ALL = f"{_SRC}/lib/components/layout/StopAllButton.svelte"
_CONVERSATIONS_API = f"{_SRC}/lib/api/conversations.ts"
_CLIENT = f"{_SRC}/lib/api/client.ts"
_DESTINATIONS = f"{_SRC}/lib/nav/destinations.ts"
_CATALOG = f"{_SRC}/lib/settings/catalog.ts"
_SETTINGS_SEARCH = f"{_SRC}/lib/settings/search.ts"
_MODAL = f"{_SRC}/lib/ds/Modal.svelte"
_DS_INDEX = f"{_SRC}/lib/ds/index.ts"
_FIXTURES = f"{_SRC}/lib/ssr_fixture/palette"

_SCRIPTS = (".svelte", ".ts", ".js")

_code = _nav._code
_script = _nav._script
_markup = _nav._markup
_imports = _nav._imports
_imports_name = _nav._imports_name
_calls = _nav._calls
_call_argument = _nav._call_argument
_filter_arguments = _nav._filter_arguments
_WORD_TEST = _nav._WORD_TEST

_MODULES = {
    "OO_RANK": _RANK,
    "OO_COMMANDS": _COMMANDS,
    "OO_SOURCES": _SOURCES,
    "OO_CONVERSATION_SOURCE": _CONVERSATION_SOURCE,
    "OO_DESTINATIONS": _DESTINATIONS,
    "OO_CATALOG": _CATALOG,
    "OO_SETTINGS_SEARCH": _SETTINGS_SEARCH,
    "OO_LISTING": _LISTING,
}


# ---------------------------------------------------------------------------
# The Node driver: it calls the pure modules and prints what they return.
# ---------------------------------------------------------------------------
_DRIVER = r"""
const load = async (name) => (process.env[name] ? await import(process.env[name]) : null);
const rank = await load('OO_RANK');
const commands = await load('OO_COMMANDS');
const sources = await load('OO_SOURCES');
const conv = await load('OO_CONVERSATION_SOURCE');
const nav = await load('OO_DESTINATIONS');
const catalog = await load('OO_CATALOG');
const search = await load('OO_SETTINGS_SEARCH');
const listing = await load('OO_LISTING');
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT || 'null');

const ids = (list) => list.map((item) => item.id);
const shape = (groups) => groups.map((g) => ({ id: g.id, label: g.label, items: ids(g.items) }));
const plain = (value) => JSON.parse(JSON.stringify(value === undefined ? null : value));

const run = {
    rank: () => ({
        limit: rank.GROUP_LIMIT,
        order: rank.GROUP_ORDER,
        labels: rank.GROUP_LABELS,
        cases: input.map(([items, query, limit]) =>
            shape(limit === null ? rank.rankPalette(items, query) : rank.rankPalette(items, query, limit)),
        ),
    }),
    sources: () => {
        const groups = [...catalog.SETTINGS_SECTIONS.flatMap((s) => s.groups), ...catalog.INLINE_GROUPS];
        const index = search.settingsIndex(
            groups, nav.DESTINATIONS, catalog.PREFERENCES_SECTIONS, catalog.SETTINGS_SECTIONS,
        );
        const probe = input.probe;
        const probeIndex = search.settingsIndex(probe.groups, probe.destinations, probe.preferences);
        const visible = (list, componion) => nav.visibleDestinations(list, { componion });
        return plain({
            groups: groups.map((g) => ({ id: g.id, retired: Boolean(g.retired), synonyms: g.synonyms ?? [] })),
            index: index.map((h) => ({ id: h.id, title: h.title, href: h.href, where: h.where })),
            settings: sources.settingOptions(index, groups),
            probeIndex: probeIndex.map((h) => ({ id: h.id, href: h.href })),
            probeSettings: sources.settingOptions(probeIndex, probe.groups),
            ready: nav.DESTINATIONS.filter((d) => d.ready).map((d) => ({ id: d.id, label: d.label, href: d.href })),
            chatsHref: (nav.DESTINATIONS.find((d) => d.id === 'chats') || {}).href ?? null,
            destinations: {
                on: sources.destinationOptions(visible(nav.DESTINATIONS, true)),
                off: sources.destinationOptions(visible(nav.DESTINATIONS, false)),
                probeOn: sources.destinationOptions(visible(probe.table, true)),
                probeOff: sources.destinationOptions(visible(probe.table, false)),
            },
            commands: ids(commands.COMMANDS),
            commandOptions: input.contexts.map((context) => commands.commandOptions(commands.COMMANDS, context)),
            chats: sources.chatOptions(input.chats, input.chatsHref, true),
            chatsLoose: sources.chatOptions(input.chats, input.chatsHref),
        });
    },
    conversations: async () => {
        const calls = [];
        const fetcher = (query, signal) => new Promise((resolve, reject) => {
            calls.push({ query, signal, resolve, reject });
        });
        const tasks = [];
        const schedule = (task, ms) => {
            const entry = { task, ms, cancelled: false };
            tasks.push(entry);
            return () => { entry.cancelled = true; };
        };
        const flush = () => { for (const entry of tasks.splice(0)) if (!entry.cancelled) entry.task(); };
        const settle = () => new Promise((done) => setTimeout(done, 0));
        const states = [];
        const source = conv.createConversationSource(fetcher, (state) => states.push(plain(state)), { schedule });
        const last = () => (states.length ? states[states.length - 1] : null);
        const seen = (id) => states.some((s) => (s.hits || []).some((hit) => hit && hit.id === id));
        const sent = () => calls.map((c) => plain(c.query));
        const out = { limit: conv.SOURCE_LIMIT, delay: conv.SOURCE_DELAY_MS };

        source.search('  onion ');
        source.search('onions  ');
        out.burst = { beforeFlush: calls.length, delays: tasks.map((t) => t.ms), asking: last() && last().asking };
        flush();
        out.burst.sent = sent();
        out.burst.signal = calls.length > 0 && calls[0].signal instanceof AbortSignal;

        source.search('onions soup');
        flush();
        out.stale = { sent: sent(), firstAborted: calls[0].signal.aborted, secondAborted: calls[1] ? calls[1].signal.aborted : null };
        calls[0].resolve([{ id: 'stale-first' }]);
        if (calls[1]) calls[1].resolve([{ id: 'soup-1' }, { id: 'soup-2' }]);
        await settle();
        out.stale.last = last();
        out.stale.seenStale = seen('stale-first');

        source.search('x1');
        flush();
        source.search('x2');
        flush();
        const older = calls[calls.length - 2];
        const newer = calls[calls.length - 1];
        newer.resolve([{ id: 'newer' }]);
        await settle();
        older.resolve([{ id: 'older' }]);
        await settle();
        out.order = { last: last(), seenOlder: seen('older') };

        let count = calls.length;
        source.search('   ');
        flush();
        out.empty = { calls: calls.length - count, last: last() };

        source.search('closing');
        flush();
        const closing = calls[calls.length - 1];
        source.close();
        out.close = { aborted: closing.signal.aborted, afterClose: last() };
        closing.resolve([{ id: 'late' }]);
        await settle();
        out.close.seenLate = seen('late');
        out.close.last = last();
        count = calls.length;
        source.search('never sent');
        source.close();
        flush();
        out.close.unsent = calls.length - count;

        source.search('broken');
        flush();
        calls[calls.length - 1].reject(new Error('server down'));
        await settle();
        out.error = { last: last() };
        const mark = states.length;
        source.search('first try');
        flush();
        const aborted = calls[calls.length - 1];
        source.search('second try');
        flush();
        aborted.reject(new DOMException('The operation was aborted.', 'AbortError'));
        calls[calls.length - 1].resolve([{ id: 'second' }]);
        await settle();
        out.error.abortedIsNoError = states.slice(mark).every((s) => s.error === null);
        out.error.after = last();
        return out;
    },
    registry: () => {
        const always = {};
        for (const command of commands.COMMANDS) {
            always[command.id] = {};
            for (const [space, contexts] of Object.entries(input.spaces)) {
                always[command.id][space] = contexts.every((context) => commands.reasonFor(command, context) !== null);
            }
        }
        return plain({
            commands: commands.COMMANDS.map((c) => ({ id: c.id, label: c.label, binding: c.binding ?? null })),
            shortcuts: commands.defaultShortcuts(commands.COMMANDS),
            always,
            labels: Object.fromEntries(
                commands.COMMANDS.filter((c) => c.binding).map((c) => [c.id, commands.bindingLabel(c.binding)]),
            ),
        });
    },
    when: () => plain(input.map((context) => commands.commandOptions(commands.COMMANDS, context))),
    keys: () => {
        const judge = (context) => Object.fromEntries(
            commands.COMMANDS.map((command) => [command.id, commands.reasonFor(command, context)]),
        );
        const chosen = typeof commands.chosenBindings === 'function';
        const bindings = chosen ? commands.chosenBindings(commands.COMMANDS, input.chosen) : null;
        return plain({
            ids: ids(commands.COMMANDS),
            judged: input.contexts.map(judge),
            bindings,
            defaults: chosen ? commands.chosenBindings(commands.COMMANDS) : null,
            options: commands.commandOptions(commands.COMMANDS, input.context, bindings ?? undefined)
                .map((o) => ({ id: o.id, detail: o.detail })),
            plainOptions: commands.commandOptions(commands.COMMANDS, input.context)
                .map((o) => ({ id: o.id, detail: o.detail })),
            shortcuts: commands.defaultShortcuts(commands.COMMANDS, bindings ?? undefined),
            labels: input.labels.map((binding) => commands.bindingLabel(binding)),
        });
    },
    listing: () => {
        const clock = () => {
            let now = 0;
            const tasks = [];
            return {
                schedule: (task, ms) => {
                    const entry = { task, due: now + ms, cancelled: false };
                    tasks.push(entry);
                    return () => { entry.cancelled = true; };
                },
                advance: (ms) => {
                    const end = now + ms;
                    for (;;) {
                        const due = tasks.filter((e) => !e.cancelled && e.due <= end).sort((a, b) => a.due - b.due);
                        if (!due.length) break;
                        due[0].cancelled = true;
                        now = due[0].due;
                        due[0].task();
                    }
                    now = end;
                },
            };
        };
        const paced = (steps, immediate) => {
            const time = clock();
            const lines = [];
            const line = listing.createStatusLine((text) => lines.push(text), { schedule: time.schedule, immediate });
            const seen = [];
            for (const [kind, value] of steps) {
                if (kind === 'update') line.update(value);
                else if (kind === 'advance') time.advance(value);
                else if (kind === 'close') line.close();
                seen.push([...lines]);
            }
            return seen;
        };
        return plain({
            limits: { slow: listing.SLOW_MS, settle: listing.SETTLE_MS },
            keys: input.flat.map((option) => listing.optionKey(option)),
            active: input.picks.map(([list, pick]) => listing.activeFor(list, pick)),
            steps: input.steps.map(([list, active, move]) => listing.stepActive(list, active, move)),
            ids: input.idCases.map((option) => listing.optionDomId('p', option)),
            status: input.status.map((state) => listing.statusFor(state)),
            paced: input.paced.map(([steps, immediate]) => paced(steps, immediate)),
        });
    },
    finds: () => {
        const groups = [...catalog.SETTINGS_SECTIONS.flatMap((s) => s.groups), ...catalog.INLINE_GROUPS];
        const index = search.settingsIndex(
            groups, nav.DESTINATIONS, catalog.PREFERENCES_SECTIONS, catalog.SETTINGS_SECTIONS,
        );
        const settings = sources.settingOptions(index, groups);
        const listed = (items, query, limit) => rank.rankPalette(items, query, limit).map(
            (g) => ({ id: g.id, items: ids(g.items) }),
        );
        const has = (name) => typeof sources[name] === 'function';
        const stop = has('stopOption') ? sources.stopOption(null) : null;
        const chats = input.chats.map((c) => ({ id: c.id, group: 'chats', label: c.title, found: true }));
        const more = has('moreChatsOption')
            ? sources.moreChatsOption('soup', '/chat', chats.length, rank.GROUP_LIMIT) : null;
        return plain({
            index: index.map((h) => ({
                id: h.id, description: h.description ?? null, where: h.where, former: h.former ?? null,
            })),
            settings: settings.map((o) => ({ id: o.id, keywords: o.keywords })),
            hub: Object.fromEntries(input.queries.map((q) => [q, ids(search.searchSettings(index, q))])),
            palette: Object.fromEntries(input.queries.map(
                (q) => [q, rank.rankPalette(settings, q, 1000).flatMap((g) => ids(g.items))],
            )),
            stop,
            refused: has('stopOption') ? sources.stopOption('Everything is stopped already') : null,
            blank: has('stopOption') ? sources.stopOption('   ') : null,
            stopRank: stop ? Object.fromEntries(input.stopWords.map((w) => [w, listed([...input.commands, stop], w)]))
                : null,
            more: has('moreChatsOption')
                ? input.more.map(([words, href, found, shown]) => sources.moreChatsOption(words, href, found, shown))
                : null,
            moreRank: more ? listed([...chats, more], 'soup') : null,
            moreEmpty: more ? listed([...chats, more], '') : null,
            moreAlone: more ? listed([more], 'soup') : null,
        });
    },
    follow: () => {
        const log = [];
        const follow = conv.followPalette((words) => log.push(['ask', words]), () => log.push(['close']));
        const after = input.map(([open, words]) => { follow(open, words); return log.length; });
        return plain({ log, after });
    },
};

const result = await run[clause]();
console.log('RESULT ' + JSON.stringify(result));
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
# Reading sources: bodies, object keys, mounts
# ---------------------------------------------------------------------------
def _balanced(text, start, opening, closing):
    """The index just past the bracket that closes the one before ``start``."""
    depth, at = 1, start
    while at < len(text) and depth:
        depth += {opening: 1, closing: -1}.get(text[at], 0)
        at += 1
    return at


def _function(script, name):
    """``(parameters, body)`` of ``function name<...>(...) {...}`` in a
    script (generics and a return type allowed), or ``('', '')``."""
    match = re.search(rf"\bfunction\s+{re.escape(name)}\s*(?:<[^>]*>)?\s*\(", script)
    if not match:
        return "", ""
    end = _balanced(script, match.end(), "(", ")")
    parameters = script[match.end():end - 1]
    brace = script.find("{", end)
    if brace < 0:
        return parameters, ""
    close = _balanced(script, brace + 1, "{", "}")
    return parameters, script[brace + 1:close - 1]


def _skip_string(text, at):
    """The index past the string literal opening at ``at``."""
    quote, at = text[at], at + 1
    while at < len(text) and text[at] != quote:
        at += 2 if text[at] == "\\" else 1
    return at + 1


def _object_entries(script, name):
    """``{key: value text}`` of the object literal assigned to ``name``, at
    its first level (``key: value`` and ``key() {...}`` alike), or None when
    no such literal is assigned."""
    match = re.search(rf"\b{re.escape(name)}\b[^=;\n]*=\s*\{{", script)
    if not match:
        return None
    parts, depth, at, start = [], 0, match.end(), match.end()
    while at < len(script):
        char = script[at]
        if char in "'\"`":
            at = _skip_string(script, at)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                parts.append(script[start:at])
                break
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(script[start:at])
            start = at + 1
        at += 1
    entries = {}
    for part in parts:
        key = re.match(r"\s*(?:async\s+)?(['\"]?)([\w$]+)\1\s*(?::|\()", part)
        if key:
            entries[key.group(2)] = part[key.end():]
    return entries


def _mounts(path, tag, text=None):
    return len(re.findall(rf"<{re.escape(tag)}\b", _markup(path, text)))


def _mounted_by(tag):
    return {path: count for path in files(".svelte") if (count := _mounts(path, tag))}


_ATTRIBUTES = r"""((?:[^>"'{]|"[^"]*"|'[^']*'|\{[^}]*\})*?)"""


def _tags(markup, tag):
    """The attribute text of every ``<tag ...>`` in the markup, an arrow
    function's ``>`` inside a brace read as part of its attribute."""
    return [match.group(1) for match in re.finditer(rf"<{re.escape(tag)}\b{_ATTRIBUTES}/?>", markup, re.S)]


# A read of the page's markup: a DOM query.
_DOM_QUERY = re.compile(
    r"\b(querySelector(?:All)?|getElementById|getElementsBy\w+|closest|matches)\s*\("
)


def _dom_queries(text):
    return [match.group(1) for match in _DOM_QUERY.finditer(text)]


# ---------------------------------------------------------------------------
# Rendering: wrappers planted in the renderer's copy open the palette first
# ---------------------------------------------------------------------------
_PLANTED = set()


def _plant(name, text):
    path = f"{_FIXTURES}/{name}.svelte"
    if path not in _PLANTED:
        ssr().plant(path, text)
        _PLANTED.add(path)
    return path


def _palette_open(name, words):
    """The palette rendered for the server, opened with ``words``. The stop
    is set running first: the renderer's stores are shared by the session,
    and a suite that rendered the stop engaged leaves it so."""
    path = _plant(name, (
        "<script>\n"
        "\timport { openPalette } from '$lib/stores/palette';\n"
        "\timport { estop } from '$lib/stores/estop';\n"
        "\timport CommandPalette from '$lib/components/palette/CommandPalette.svelte';\n"
        "\testop.update((s) => ({ ...s, stopped: false }));\n"
        f"\topenPalette({json.dumps(words)});\n"
        "</script>\n"
        "<CommandPalette />\n"
    ))
    return _dom(ssr().render(path).html)


def _by_id(root):
    return {element.get("id"): element for element in root.iter() if element.get("id")}


def _named(element, ids):
    """An element's accessible name: its label, the text of what labels it,
    or (for a field) the text of its label element."""
    label = element.get("aria-label")
    if label and label.strip():
        return " ".join(label.split())
    labelled = element.get("aria-labelledby")
    if labelled:
        words = " ".join(ids[key].text() for key in labelled.split() if key in ids)
        if words.strip():
            return " ".join(words.split())
    return ""


def _inside(element, ancestor):
    node = element.parent
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


def _stops(root):
    return [
        button for button in root.iter("button")
        if re.search(r"\bstop all\b", " ".join((button.get("aria-label") or button.text()).split()), re.I)
    ]


# ---------------------------------------------------------------------------
# The contexts a command is judged in
# ---------------------------------------------------------------------------
def _context(space="use", chat=None, streaming=False, palette=True, help=False):
    return {"space": space, "chatId": chat, "streaming": streaming, "palette": palette, "help": help}


def _space_contexts():
    """Every context of each space the registry is judged in: in Use, with
    no chat, a chat at rest and a chat being answered; in the Workshop, with
    no chat (no conversation is ever on screen there); each with the palette
    and the help open or shut."""
    use, workshop = [], []
    for palette in (False, True):
        for help_open in (False, True):
            use.append(_context("use", None, False, palette, help_open))
            use.append(_context("use", "c1", False, palette, help_open))
            use.append(_context("use", "c1", True, palette, help_open))
            workshop.append(_context("workshop", None, False, palette, help_open))
    return {"use": use, "workshop": workshop}


# ---------------------------------------------------------------------------
# NV11 -- Ctrl+K opens the palette through its store
# ---------------------------------------------------------------------------
_DISPATCH = re.compile(r"new\s+CustomEvent\(\s*['\"]([\w-]+)['\"]")


def _shortcut_scope(sources):
    """The files a shortcut runs through: the shortcut handler, the layout
    that mounts it, every file of the palette's modules and its store, and
    every file that listens for an event one of these sends."""
    scope = {
        path for path in sources
        if path.endswith("/KeyboardShortcuts.svelte") or path == _PALETTE_STORE
        or path.startswith(_PALETTE_DIR + "/")
    }
    scope |= {
        path for path, text in sources.items()
        if path.endswith(".svelte") and re.search(r"<KeyboardShortcuts\b", _markup(path, text))
    }
    events = set()
    for path in scope:
        events.update(_DISPATCH.findall(_script(path, sources[path])))
    for path, text in sources.items():
        for event in events:
            if re.search(r"addEventListener\(\s*['\"]" + re.escape(event) + r"['\"]", _script(path, text)):
                scope.add(path)
    return sorted(scope)


@pytest.mark.parametrize("half", ("query", "store"))
def test_nv11_ctrl_k_opens_the_palette_through_its_store(half):
    if half == "query":
        assert _dom_queries(
            "function f() { document.querySelector('[data-x] input'); el.closest('form');"
            " document.getElementById('a'); x.matchesShortcut(e); }"
        ) == ["querySelector", "closest", "getElementById"], "the census reads the DOM queries"
        sample = {
            "frontend/src/routes/+layout.svelte": "<script>import K from '$lib/K.svelte';</script><KeyboardShortcuts />",
            "frontend/src/lib/components/ui/KeyboardShortcuts.svelte": (
                "<script>window.dispatchEvent(new CustomEvent('opti-go'));</script>"
            ),
            "frontend/src/lib/Listener.svelte": "<script>window.addEventListener('opti-go', f);</script>",
            "frontend/src/lib/Other.svelte": "<script>document.querySelector('x');</script>",
        }
        assert _shortcut_scope(sample) == [
            "frontend/src/lib/Listener.svelte", "frontend/src/lib/components/ui/KeyboardShortcuts.svelte",
            "frontend/src/routes/+layout.svelte",
        ], "the census follows the handler, the layout that mounts it, and the listeners of its events"

        sources = {path: read(path) for path in files(_SCRIPTS)}
        scope = _shortcut_scope(sources)
        assert _SHORTCUTS in scope and _ROOT_LAYOUT in scope, f"the census finds the shortcut handler: {scope}"
        queries = {path: found for path in scope if (found := _dom_queries(_code(path, sources[path])))}
        assert not queries, f"what a shortcut runs reads the page's markup: {queries}"
        return

    registry = _node("registry", ("OO_COMMANDS",), {"spaces": _space_contexts()})
    ctrl_k = [
        c["id"] for c in registry["commands"]
        if c["binding"] and c["binding"]["key"].lower() == "k" and c["binding"].get("ctrl")
        and not c["binding"].get("shift") and not c["binding"].get("alt")
    ]
    assert ctrl_k == ["search_conversations"], f"Ctrl+K is the palette's command, and only it: {ctrl_k}"

    store = _script(_PALETTE_STORE)
    for name in ("palette", "openPalette", "closePalette"):
        assert re.search(rf"\bexport\s+(?:const|function|let)\s+{name}\b", store), (
            f"the palette's store exports {name}"
        )
    handlers = _object_entries(_script(_RUN), "HANDLERS")
    assert handlers is not None, "the runner holds its handlers in HANDLERS"
    assert _calls(handlers.get("search_conversations", ""), "openPalette"), (
        "the palette's command opens the palette through its store"
    )
    assert _imports_name(_RUN, "openPalette", _PALETTE_STORE), "the runner takes openPalette from the store"

    mounted = _mounted_by("CommandPalette")
    assert mounted == {_APP_LAYOUT: 1}, f"the palette is mounted once, by the layout of both spaces: {mounted}"
    markup = _markup(_PALETTE)
    modals = _tags(markup, "Modal")
    assert len(modals) == 1 and re.search(r"\bopen\s*=\s*\{\s*\$palette\.open\s*\}", modals[0]), (
        f"the palette's one dialog is open when the store says so: {modals}"
    )
    closes = _calls(_code(_PALETTE), "closePalette") or re.search(r"=\s*\{\s*closePalette\s*\}", markup)
    assert _imports_name(_PALETTE, "closePalette", _PALETTE_STORE) and closes, (
        "the palette closes through its store"
    )
    tags = _tags(_markup(_ROOT_LAYOUT), "KeyboardShortcuts")
    assert len(tags) == 1 and not re.search(r"\bon[A-Z]\w*\s*=", tags[0]), (
        f"the shortcut handler is handed no callbacks: it runs the registry's commands: {tags}"
    )


# ---------------------------------------------------------------------------
# NV12 -- a combobox over a grouped listbox, in the ds Modal, holding Stop all
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", ("aria", "stop"))
def test_nv12_the_palette_is_a_combobox_over_a_grouped_listbox_in_the_modal(half):
    assert _imports_name(_PALETTE, "Modal", _DS_INDEX) or _MODAL in _imports(_PALETTE), (
        "the palette is hosted in the ds Modal"
    )
    if half == "stop":
        assert _STOP_ALL in _imports(_PALETTE) and _mounts(_PALETTE, "StopAllButton") == 1, (
            "the palette mounts the one stop control"
        )
        for name, words in (("PaletteStopEmpty", ""), ("PaletteStopNone", "zzqqxxnothing"),
                            ("PaletteStopQuery", "export")):
            root = _palette_open(name, words)
            dialogs = list(root.iter("dialog"))
            assert len(dialogs) == 1, f"{words!r}: one dialog: {len(dialogs)}"
            stops = [stop for stop in _stops(root) if _inside(stop, dialogs[0])]
            assert stops, f"{words!r}: the palette holds Stop all while it covers the page"
        return

    root = _palette_open("PaletteAriaEmpty", "")
    ids = _by_id(root)
    dialogs = list(root.iter("dialog"))
    assert len(dialogs) == 1, f"one dialog: {len(dialogs)}"
    dialog = dialogs[0]
    assert "oo-modal" in dialog.classes() and dialog.get("data-variant") == "center", (
        f"the ds Modal, in its center variant: {dialog.attrs}"
    )
    assert _named(dialog, ids), "the dialog is named by its title"

    combos = [e for e in root.iter() if e.get("role") == "combobox"]
    assert len(combos) == 1 and combos[0].tag == "input", f"one combobox, the field: {[e.tag for e in combos]}"
    combo = combos[0]
    labels = [e for e in root.iter("label") if e.get("for") and e.get("for") == combo.get("id")]
    assert _named(combo, ids) or any(label.text().strip() for label in labels), "the field is named"
    assert combo.get("aria-expanded") == "true", f"the list is shown: {combo.get('aria-expanded')}"
    assert combo.get("aria-autocomplete") == "list", f"the list follows the words: {combo.get('aria-autocomplete')}"
    listbox = ids.get(combo.get("aria-controls") or "")
    assert listbox is not None and listbox.get("role") == "listbox", (
        f"the field controls the listbox: {combo.get('aria-controls')}"
    )
    assert _named(listbox, ids), "the listbox is named"

    groups = [e for e in listbox.iter() if e.get("role") == "group"]
    assert groups, "the options sit in groups"
    names = [_named(group, ids) for group in groups]
    assert all(names) and len(set(names)) == len(names), f"each group is named, by its own label: {names}"
    for group in groups:
        for key in (group.get("aria-labelledby") or "").split():
            assert key in ids and _inside(ids[key], group) and ids[key].visible_text().strip(), (
                f"a group's label is visible, inside it: {key}"
            )
    options = [e for e in listbox.iter() if e.get("role") == "option"]
    assert options, "the empty palette lists options"
    outside = [o.get("id") for o in options if not any(_inside(o, g) for g in groups)]
    assert not outside, f"every option sits in a group: {outside}"
    option_ids = [o.get("id") for o in options]
    assert all(option_ids) and len(set(option_ids)) == len(option_ids), f"every option has its own id: {option_ids}"
    assert all(o.get("aria-selected") in ("true", "false") for o in options), "every option says whether it is selected"
    selected = [o.get("id") for o in options if o.get("aria-selected") == "true"]
    assert selected == [option_ids[0]], f"the first option is the active one, and only it: {selected}"
    assert combo.get("aria-activedescendant") == option_ids[0], (
        f"the field names the active option: {combo.get('aria-activedescendant')}"
    )
    nested = [o.get("id") for o in options if list(o.iter("button")) or list(o.iter("input"))
              or [a for a in o.iter("a") if a.get("href")]]
    assert not nested, f"an option holds no control of its own: {nested}"
    texts = " ".join(o.text() for o in options)
    for label in ("Chats", "Preferences", "Models and inference"):
        assert label in texts, f"the empty palette lists the destinations: {label}"

    none = _palette_open("PaletteAriaNone", "zzqqxxnothing")
    none_ids = _by_id(none)
    field = [e for e in none.iter() if e.get("role") == "combobox"]
    assert len(field) == 1, "the field stays when nothing matches"
    pointed = field[0].get("aria-activedescendant")
    assert not pointed or pointed in none_ids, f"the field names no option that is not there: {pointed}"
    assert not [e for e in none.iter() if e.get("role") == "option"], "nothing matches, nothing is listed"
    assert not [e for e in none.iter() if e.get("role") == "group"], "and no empty group is drawn"


# ---------------------------------------------------------------------------
# NV13 -- the ranking
# ---------------------------------------------------------------------------
def _item(ident, label, group="commands", **extra):
    return {"id": ident, "group": group, "label": label, **extra}


_E_ACUTE = chr(0xE9)
_E_ACUTE_CAPITAL = chr(0xC9)

_TIER_ITEMS = [
    _item("sub", "Change theme"),
    _item("none", "Notes"),
    _item("word", "New chat"),
    _item("found", "Onion soup", found=True),
    _item("prefix", "Chats index"),
    _item("kw", "Talk", keywords=["chat"]),
    _item("exact", "CHAT"),
    _item("lost", "Soup"),
    _item("loose", "Theme", keywords=["cheap tea"]),
    _item("kwword", "Palette", keywords=["open the chat"]),
]
_FOLD_ITEMS = [
    _item("accent", f"R{_E_ACUTE}glages du chat"),
    _item("plain", "reglages"),
]
_STABLE_ITEMS = [
    _item("g3", "Go three"),
    _item("gsub", "Big orange"),
    _item("g1", "Go one"),
    _item("gexact", "go"),
    _item("g2", "Go two"),
]
_LIMIT_ITEMS = [
    *(_item(f"a{n}", f"Alpha {n}") for n in range(1, 10)),
    _item("sa", "Alpha setting", group="settings"),
    _item("sb", "Alpha second setting", group="settings"),
]
_GROUP_ITEMS = [
    _item("d-explore", "Explore reports", group="destinations"),
    _item("c-export", "Export this chat"),
    _item("s-export", "Export", group="settings"),
    _item("h-export", "Trip notes", group="chats", found=True),
    _item("h-other", "Other trip", group="chats"),
]
_TIE_ITEMS = [
    _item("c-backup", "Backup now"),
    _item("s-backup", "Backup", group="settings"),
    _item("d-backup", "Backup", group="destinations"),
]
_EMPTY_ITEMS = [
    *(_item(f"d{n}", f"Place {n}", group="destinations") for n in range(1, 9)),
    _item("c1", "New chat"),
    _item("c2", "Export this chat"),
    _item("s1", "Theme", group="settings"),
    *(_item(f"h{n}", f"Chat {n}", group="chats", found=True) for n in range(1, 10)),
]


def _rank_cases():
    return [
        [_TIER_ITEMS, "chat", 50],
        [_FOLD_ITEMS, f"  R{_E_ACUTE_CAPITAL}GLAGES ", 50],
        [_STABLE_ITEMS, "go", 50],
        [list(reversed(_STABLE_ITEMS)), "go", 50],
        [_LIMIT_ITEMS, "alpha", 4],
        [_LIMIT_ITEMS, "alpha", None],
        [_GROUP_ITEMS, "export", 50],
        [_TIE_ITEMS, "backup", 50],
        [_EMPTY_ITEMS, "", None],
        [_EMPTY_ITEMS, "   ", 3],
        [_STABLE_ITEMS, "zzqq", 50],
    ]


@pytest.mark.parametrize("half", ("tiers", "stable", "limit", "groups", "empty", "wiring"))
def test_nv13_the_palette_ranks_exact_then_prefix_then_word_then_letters(half):
    if half == "wiring":
        code, script = _code(_PALETTE), _script(_PALETTE)
        assert _imports_name(_PALETTE, "rankPalette", _RANK) and _calls(code, "rankPalette"), (
            "the palette ranks its options with rankPalette"
        )
        assert re.search(r"\.sort\s*\(", "list.sort((a, b) => a.rank - b.rank);"), "the census reads a sort"
        assert not re.search(r"\.sort\s*\(", script), "the palette sorts nothing itself"
        filters = [found for found in _filter_arguments(script) if _WORD_TEST.search(found)]
        assert not filters, f"the palette matches no words itself: {filters}"
        assert not _calls(code, "searchSettings"), (
            "the palette ranks the settings index itself, never through the settings page's filter"
        )
        return

    result = _node("rank", ("OO_RANK",), _rank_cases())
    cases = result["cases"]
    limit = result["limit"]
    assert isinstance(limit, int) and 1 <= limit <= 12, f"a group shows a few options: {limit}"
    order, labels = result["order"], result["labels"]
    assert order == ["destinations", "commands", "settings", "chats"], f"the groups' order of ties: {order}"
    assert set(labels) == set(order) and all(isinstance(v, str) and v.strip() for v in labels.values()), (
        f"every group has a label: {labels}"
    )
    for groups in cases:
        for group in groups:
            assert group["label"] == labels[group["id"]] and group["items"], (
                f"a listed group carries its label and at least one option: {group}"
            )

    def one(groups):
        assert len(groups) == 1, f"one group: {groups}"
        return groups[0]["items"]

    if half == "tiers":
        assert one(cases[0]) == ["exact", "prefix", "word", "kw", "kwword", "sub", "found"], (
            f"the label exact, at its start, at a word's start; then a keyword the same ways; then the label's "
            f"letters in order; then what the source found: {cases[0]}"
        )
        assert one(cases[1]) == ["plain", "accent"], (
            f"case, accents and spaces around the words are ignored: {cases[1]}"
        )
        assert cases[10] == [], f"nothing matches, nothing is listed: {cases[10]}"
        return
    if half == "stable":
        assert one(cases[2]) == ["gexact", "g3", "g1", "g2", "gsub"], (
            f"options that tie keep the order they were given in: {cases[2]}"
        )
        assert one(cases[3]) == ["gexact", "g2", "g1", "g3", "gsub"], (
            f"and follow it when it changes: {cases[3]}"
        )
        return
    if half == "limit":
        by_group = {g["id"]: g["items"] for g in cases[4]}
        assert by_group == {"commands": ["a1", "a2", "a3", "a4"], "settings": ["sa", "sb"]}, (
            f"each group holds at most the limit, its best first; the limit is per group: {by_group}"
        )
        by_group = {g["id"]: g["items"] for g in cases[5]}
        assert by_group["commands"] == [f"a{n}" for n in range(1, 10)][:limit], (
            f"the default limit is GROUP_LIMIT: {by_group}"
        )
        return
    if half == "groups":
        assert [g["id"] for g in cases[6]] == ["settings", "commands", "destinations", "chats"], (
            f"the groups follow their best option, so the first option is the best match: {cases[6]}"
        )
        assert {g["id"]: g["items"] for g in cases[6]}["chats"] == ["h-export"], (
            f"a chat the source did not find for these words and whose title does not match is not listed: {cases[6]}"
        )
        assert [g["id"] for g in cases[7]] == ["destinations", "settings", "commands"], (
            f"groups whose best options tie follow the groups' order: {cases[7]}"
        )
        return
    # empty
    destinations = [f"d{n}" for n in range(1, 9)]
    assert cases[8] == [
        {"id": "destinations", "label": labels["destinations"], "items": destinations},
        {"id": "chats", "label": labels["chats"], "items": [f"h{n}" for n in range(1, 10)][:limit]},
    ], f"with no words: every destination, then the recent chats, and nothing else: {cases[8]}"
    assert cases[9] == [
        {"id": "destinations", "label": labels["destinations"], "items": destinations},
        {"id": "chats", "label": labels["chats"], "items": ["h1", "h2", "h3"]},
    ], f"spaces alone are no words; the limit holds the recent chats: {cases[9]}"


# ---------------------------------------------------------------------------
# NV14 -- the sources
# ---------------------------------------------------------------------------
_PROBE_CATALOG = {
    "groups": [
        {"id": "host", "title": "Host", "description": "", "space": "workshop", "section": "models",
         "synonyms": ["hosting"]},
        {"id": "inner", "title": "Inner", "description": "", "embeddedIn": "host"},
        {"id": "gone", "title": "Gone", "description": "", "retired": "No reader"},
        {"id": "pref", "title": "Pref", "description": "", "space": "use", "section": "appearance"},
    ],
    "destinations": [
        {"id": "preferences", "label": "Preferences", "href": "/preferences"},
        {"id": "models", "label": "Models", "href": "/workshop/models"},
    ],
    "preferences": [{"id": "appearance", "label": "Appearance"}],
    "table": _nav._probe_table(),
}
_CHATS = [{"id": "c-1", "title": "Onion soup"}, {"id": "c 2", "title": ""}, {"id": "c3", "title": None}]


def _sources():
    return _node(
        "sources",
        ("OO_SOURCES", "OO_COMMANDS", "OO_DESTINATIONS", "OO_CATALOG", "OO_SETTINGS_SEARCH"),
        {"probe": _PROBE_CATALOG, "chats": _CHATS, "chatsHref": "/chat",
         "contexts": [_context(), _context(chat="c1"), _context(chat="c1", streaming=True),
                      _context("workshop")]},
    )


@pytest.mark.parametrize("half", ("sources", "conversations", "wiring"))
def test_nv14_the_sources_cover_every_place_command_group_and_chat(half):
    if half == "sources":
        result = _sources()
        live = [g for g in result["groups"] if not g["retired"]]
        assert len(result["groups"]) == 51 and len(live) == 50, (
            f"the catalog holds 51 groups, 50 not retired: {len(result['groups'])}, {len(live)}"
        )
        index = {hit["id"]: hit for hit in result["index"]}
        settings = result["settings"]
        assert sorted(option["id"] for option in settings) == sorted(g["id"] for g in live), (
            "the palette lists every group that is not retired, once"
        )
        assert [option["id"] for option in settings] == [hit["id"] for hit in result["index"]], (
            "in the settings index's order"
        )
        synonyms = {g["id"]: g["synonyms"] for g in result["groups"]}
        for option in settings:
            hit = index[option["id"]]
            assert option["group"] == "settings" and option["label"] == hit["title"], option
            assert option["href"] == hit["href"] and option["detail"] == hit["where"], (
                f"a group links to the page holding it, and says where that is: {option}"
            )
            assert set(synonyms[option["id"]]) <= set(option.get("keywords") or []), (
                f"a group is found by its synonyms: {option}"
            )
        probe = {option["id"]: option["href"] for option in result["probeSettings"]}
        assert probe == {"host": "/workshop/models?g=host", "inner": "/workshop/models?g=host",
                         "pref": "/preferences?g=pref"}, (
            f"an embedded group opens its host's page, a retired one is not listed: {probe}"
        )

        ready = result["ready"]
        for switch in ("on", "off"):
            shown = result["destinations"][switch]
            assert [(d["id"], d["label"], d["href"]) for d in shown] == [
                (d["id"], d["label"], d["href"]) for d in ready
            ], f"the ready destinations, from the table, in its order ({switch}): {shown}"
            assert all(d["group"] == "destinations" for d in shown), shown
        assert [d["id"] for d in result["destinations"]["probeOn"]] == ["home", "chats", "componion", "status"]
        assert [d["id"] for d in result["destinations"]["probeOff"]] == ["home", "chats", "status"], (
            "the componion is dropped while its switch is off"
        )

        for options in result["commandOptions"]:
            assert [option["id"] for option in options] == result["commands"], (
                f"every command is listed in every context: {[o['id'] for o in options]}"
            )
            assert all(o["group"] == "commands" and o["command"] == o["id"] for o in options), options
        assert len(result["commands"]) >= 9, f"the registry holds at least the nine shortcuts: {result['commands']}"

        assert result["chatsHref"] == "/chat", f"the chats destination: {result['chatsHref']}"
        chats = result["chats"]
        assert [(c["id"], c["href"], c["group"]) for c in chats] == [
            ("c-1", "/chat/c-1", "chats"), ("c 2", "/chat/c%202", "chats"), ("c3", "/chat/c3", "chats"),
        ], f"a chat links under the chats destination: {chats}"
        assert chats[0]["label"] == "Onion soup" and chats[1]["label"].strip() and chats[2]["label"].strip(), (
            f"a chat is named by its title, or says it has none: {[c['label'] for c in chats]}"
        )
        assert all(c["found"] is True for c in chats) and not any(c.get("found") for c in result["chatsLoose"]), (
            "a chat the server found for the words says so, and only then"
        )
        return

    if half == "conversations":
        result = _node("conversations", ("OO_CONVERSATION_SOURCE",))
        limit, delay = result["limit"], result["delay"]
        assert isinstance(limit, int) and 1 <= limit <= 200, f"a search asks for a bounded number: {limit}"
        assert isinstance(delay, (int, float)) and 0 < delay <= 500, f"the words go once the reader pauses: {delay}"
        burst = result["burst"]
        assert burst["beforeFlush"] == 0 and burst["delays"] and all(ms == delay for ms in burst["delays"]), (
            f"nothing is sent before the pause: {burst}"
        )
        assert burst["asking"] == "onions", f"the source says which words it is asking: {burst['asking']}"
        assert burst["sent"] == [{"q": "onions", "limit": limit}] and burst["signal"], (
            f"a burst of keys asks once, the last words trimmed, in q, with a signal: {burst['sent']}"
        )
        stale = result["stale"]
        assert stale["sent"][-1] == {"q": "onions soup", "limit": limit} and len(stale["sent"]) == 2, stale["sent"]
        assert stale["firstAborted"] is True and stale["secondAborted"] is False, (
            f"a new search aborts the request before it: {stale}"
        )
        assert stale["last"] == {"query": "onions soup", "hits": [{"id": "soup-1"}, {"id": "soup-2"}],
                                 "asking": None, "error": None}, f"the latest answer is shown: {stale['last']}"
        assert stale["seenStale"] is False, "an answer to words no longer asked is dropped"
        order = result["order"]
        assert order["last"]["hits"] == [{"id": "newer"}] and order["seenOlder"] is False, (
            f"an older answer arriving last is dropped too: {order}"
        )
        empty = result["empty"]
        assert empty["calls"] == 0 and empty["last"] == {"query": "", "hits": [], "asking": None, "error": None}, (
            f"no words, no request, and nothing shown: {empty}"
        )
        close = result["close"]
        assert close["aborted"] is True, "closing aborts the request in flight"
        assert close["seenLate"] is False and close["last"] == close["afterClose"], (
            f"and drops its answer: {close}"
        )
        assert close["unsent"] == 0, "closing before the pause is over sends nothing"
        error = result["error"]
        assert error["last"]["query"] == "broken" and error["last"]["hits"] == [] and error["last"]["asking"] is None
        assert "server down" in (error["last"]["error"] or ""), f"a failed search says why: {error['last']}"
        assert error["abortedIsNoError"] is True and error["after"]["hits"] == [{"id": "second"}], (
            f"a request aborted by a newer one is no failure: {error}"
        )
        return

    code = _code(_PALETTE)
    for name, module in (
        ("destinationOptions", _SOURCES), ("settingOptions", _SOURCES), ("chatOptions", _SOURCES),
        ("commandOptions", _COMMANDS), ("visibleDestinations", _DESTINATIONS),
        ("settingsIndex", _SETTINGS_SEARCH), ("createConversationSource", _CONVERSATION_SOURCE),
    ):
        assert _imports_name(_PALETTE, name, module) and _calls(code, name), (
            f"the palette builds its options through {name}"
        )
    assert _imports_name(_PALETTE, "COMMANDS", _COMMANDS), "the palette lists the registry's commands"
    assert _nav._destination_list(_PALETTE, read(_PALETTE)) == 0, "the palette holds no list of its own"
    assert "signal" in _call_argument(code, "listConversations"), (
        "the palette's request carries the source's abort signal"
    )
    held = re.search(r"\b(?:const|let)\s+([\w$]+)\s*(?::[^=]*)?=\s*createConversationSource\s*(?:<[^>]*>)?\s*\(", code)
    assert held, "the palette holds its conversation source"
    name = re.escape(held.group(1))
    assert re.search(rf"\b{name}\s*\.\s*search\s*\(", code) and re.search(rf"\b{name}\s*\.\s*close\s*\(\s*\)", code), (
        "the palette hands the source its words, and closes it"
    )
    api = _script(_CONVERSATIONS_API)
    parameters, body = _function(api, "listConversations")
    assert "signal" in parameters and "signal" in _call_argument(body, "apiGet"), (
        "listConversations takes the signal and hands it to apiGet"
    )
    parameters, body = _function(_script(_CLIENT), "apiGet")
    assert "signal" in parameters and "signal" in _call_argument(body, "fetch"), (
        "and apiGet hands it to fetch, so closing the palette stops the request"
    )


# ---------------------------------------------------------------------------
# NV15 -- every default shortcut is a command
# ---------------------------------------------------------------------------
# The nine default shortcuts the handler declared before the registry, and
# their bindings: (key, ctrl, shift, alt).
_DEFAULT_SHORTCUTS = {
    "new_chat": ("n", True, False, False),
    "send_message": ("Enter", True, False, False),
    "toggle_sidebar": ("b", True, False, False),
    "search_conversations": ("k", True, False, False),
    "open_settings": (",", True, False, False),
    "toggle_theme": ("t", True, True, False),
    "export_conversation": ("e", True, True, False),
    "show_shortcuts": ("?", False, False, False),
    "close_dialog": ("Escape", False, False, False),
}
_BINDING_LABELS = {
    "export_conversation": "Ctrl + Shift + E",
    "search_conversations": "Ctrl + K",
    "send_message": "Ctrl + Enter",
    "open_settings": "Ctrl + ,",
    "close_dialog": "Esc",
    "show_shortcuts": "?",
}
_SHORTCUT_LIST = re.compile(r"\{[^{}]*\bkey\s*:[^{}]*\baction\s*:[^{}]*\}", re.S)


def _binding(value):
    return (value["key"], bool(value.get("ctrl")), bool(value.get("shift")), bool(value.get("alt")))


@pytest.mark.parametrize("half", ("registry", "wiring"))
def test_nv15_every_default_shortcut_is_a_command_with_a_handler_or_a_reason(half):
    if half == "registry":
        registry = _node("registry", ("OO_COMMANDS",), {"spaces": _space_contexts()})
        commands = {c["id"]: c for c in registry["commands"]}
        assert len(commands) == len(registry["commands"]), "every command has its own id"
        assert all(isinstance(c["label"], str) and c["label"].strip() for c in commands.values()), (
            "every command has a label"
        )
        missing = sorted(set(_DEFAULT_SHORTCUTS) - set(commands))
        assert not missing, f"each default shortcut is a command of the registry: {missing}"
        wrong = {
            action: _binding(commands[action]["binding"]) if commands[action]["binding"] else None
            for action, binding in _DEFAULT_SHORTCUTS.items()
            if not commands[action]["binding"] or _binding(commands[action]["binding"]) != binding
        }
        assert not wrong, f"each keeps its default binding: {wrong}"
        shortcuts = registry["shortcuts"]
        bound = [c["id"] for c in registry["commands"] if c["binding"]]
        assert [s["action"] for s in shortcuts] == bound, (
            f"the handler starts from every bound command, once, in the registry's order: {shortcuts}"
        )
        for shortcut in shortcuts:
            command = commands[shortcut["action"]]
            assert shortcut["description"] == command["label"], shortcut
            assert _binding(shortcut) == _binding(command["binding"]), shortcut
            assert all(isinstance(shortcut.get(k), bool) for k in ("ctrl", "shift", "alt")), shortcut
        labels = {action: registry["labels"].get(action) for action in _BINDING_LABELS}
        assert labels == _BINDING_LABELS, f"a binding reads as its keys: {labels}"
        return

    script, markup = _script(_SHORTCUTS), _markup(_SHORTCUTS)
    assert _SHORTCUT_LIST.search("const s = [{ key: 'n', ctrl: true, description: 'New', action: 'new_chat' }];"), (
        "the census reads a list of shortcuts"
    )
    assert not _SHORTCUT_LIST.search(script), "the shortcut handler declares no shortcut of its own"
    assert _imports_name(_SHORTCUTS, "defaultShortcuts", _COMMANDS) and _calls(script, "defaultShortcuts"), (
        "the shortcut handler starts from the registry"
    )
    assert _imports_name(_SHORTCUTS, "runCommand", _RUN) and _calls(script, "runCommand"), (
        "and runs every action through the runner"
    )
    assert not re.search(r"\bactionHandlers\b", script) and not re.search(r"\bexport\s+let\s+on[A-Z]", script), (
        "and holds no handler of its own"
    )
    assert _calls(script, "getKeyboardShortcuts") and "opti-shortcuts-updated" in script, (
        "the server's own bindings still apply"
    )
    assert re.search(r"\{#each\b", markup), "the help lists the shortcuts it runs"
    sample = _object_entries(
        "const HANDLERS: X = { a: () => go('x,y'), async b() { c({ d: 1 }); }, 'c': f };", "HANDLERS",
    )
    assert sample is not None and sorted(sample) == ["a", "b", "c"] and "go('x,y')" in sample["a"], (
        f"the census reads the handlers' keys and what each runs: {sample}"
    )
    handlers = _object_entries(_script(_RUN), "HANDLERS")
    assert handlers is not None, "the runner holds its handlers in HANDLERS"
    registry = _node("registry", ("OO_COMMANDS",), {"spaces": _space_contexts()})
    commands = {c["id"]: c for c in registry["commands"]}
    always = registry["always"]
    dead = {
        command: always[command] for command in commands
        if command not in handlers and not all(always[command].values())
    }
    assert not dead, (
        f"a command runs somewhere and has no handler (in each space it needs a handler or a reason): {dead}"
    )
    stray = sorted(set(handlers) - set(commands))
    assert not stray, f"a handler for no command of the registry: {stray}"
    parameters, body = _function(_script(_RUN), "runCommand")
    asked, looked = body.find("reasonFor"), body.find("HANDLERS")
    assert _calls(body, "reasonFor") and (looked < 0 or asked < looked), (
        "the runner asks for the command's reason before it looks for a handler"
    )


# ---------------------------------------------------------------------------
# NV16 -- a command that cannot run here is disabled with its reason
# ---------------------------------------------------------------------------
_WHEN = {
    "use, no chat": _context(),
    "use, a chat at rest": _context(chat="c1"),
    "use, a reply being written": _context(chat="c1", streaming=True),
    "workshop": _context("workshop"),
}
# Whether each command can run in each context: True runs, False is refused.
_EXPECTED = {
    "export_conversation": {"use, no chat": False, "use, a chat at rest": True,
                            "use, a reply being written": True, "workshop": False},
    "stop_reply": {"use, no chat": False, "use, a chat at rest": False,
                   "use, a reply being written": True, "workshop": False},
    "send_message": {"use, no chat": False, "use, a chat at rest": True,
                     "use, a reply being written": False, "workshop": False},
    "new_chat": {"use, no chat": True, "use, a chat at rest": True,
                 "use, a reply being written": True, "workshop": True},
}


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_nv16_a_command_that_cannot_run_here_is_listed_disabled_with_its_reason(half):
    names = list(_WHEN)
    result = dict(zip(names, _node("when", ("OO_COMMANDS",), [_WHEN[name] for name in names])))
    if half == "node":
        registry = _node("registry", ("OO_COMMANDS",), {"spaces": _space_contexts()})
        every = [c["id"] for c in registry["commands"]]
        for name, options in result.items():
            assert [o["id"] for o in options] == every, f"{name}: every command is listed, none hidden"
            for option in options:
                if option["disabled"]:
                    assert isinstance(option["reason"], str) and option["reason"].strip(), (
                        f"{name}: a disabled command says why: {option}"
                    )
                    assert option["reason"] != option["label"], option
                else:
                    assert option["disabled"] is False and option["reason"] is None, (
                        f"{name}: a command that runs carries no reason: {option}"
                    )
        wrong = {}
        for command, states in _EXPECTED.items():
            for name, runs in states.items():
                option = next((o for o in result[name] if o["id"] == command), None)
                if option is None or option["disabled"] == runs:
                    wrong[f"{command} in {name}"] = None if option is None else option["reason"]
        assert not wrong, f"what runs where: {wrong}"
        return

    reasons = {o["id"]: o for o in result["use, no chat"]}
    for name, words, command in (("PaletteWhyExport", "export", "export_conversation"),
                                 ("PaletteWhySend", "send", "send_message"),
                                 ("PaletteWhyStop", "stop this", "stop_reply")):
        root = _palette_open(name, words)
        want = reasons[command]
        options = [o for o in root.iter() if o.get("role") == "option" and want["label"] in " ".join(o.text().split())]
        assert len(options) == 1, f"{words!r}: the command is listed, not hidden: {len(options)}"
        option = options[0]
        assert option.get("aria-disabled") == "true", f"{words!r}: it is listed disabled: {option.attrs}"
        assert want["reason"] in " ".join(option.visible_text().split()), (
            f"{words!r}: with its reason in words: {option.visible_text()!r}"
        )
    parameters, body = _function(_script(_RUN), "runCommand")
    assert _calls(body, "reasonFor"), "the runner refuses a command its context refuses"
    assert not re.search(r"\.filter\s*\([^)]*\bdisabled\b", _script(_PALETTE)), (
        "the palette hides no disabled command"
    )


# ---------------------------------------------------------------------------
# NV17 -- the sidebar's Search entry opens the palette
# ---------------------------------------------------------------------------
_CONTROL = re.compile(
    r"<(Button|IconButton|TextButton|Input|button|input)\b((?:[^>\"'{]|\"[^\"]*\"|'[^']*'|\{[^}]*\})*?)(/>|>(.*?)</\1>)",
    re.S,
)


def _search_controls(path, text=None):
    """``[(tag, attributes)]`` of the controls in a component's markup whose
    name (their label, or their text) starts with Search."""
    found = []
    for match in _CONTROL.finditer(_markup(path, text)):
        attributes, inner = match.group(2), match.group(4) or ""
        label = re.search(r"""\b(?:label|aria-label|ariaLabel)\s*=\s*(?:"([^"]*)"|'([^']*)'|\{\s*['"`]([^'"`]*)['"`]\s*\})""",
                          attributes)
        name = next((g for g in label.groups() if g is not None), "") if label else ""
        name = name or re.sub(r"<[^>]*>|\{[^}]*\}", " ", inner)
        if re.match(r"\s*search\b", name, re.I):
            found.append((match.group(1), attributes))
    return found


def _opens_palette(attributes, script):
    click = re.search(r"\bon:click\s*=\s*\{([^}]*)\}", attributes)
    if not click:
        return False
    handler = click.group(1).strip()
    if _calls(handler, "openPalette") or handler == "openPalette":
        return True
    named = re.fullmatch(r"[\w$]+", handler)
    return bool(named and _calls(_nav._function_body(script, handler), "openPalette"))


@pytest.mark.parametrize("half", ("handler", "forms"))
def test_nv17_the_sidebar_search_entry_opens_the_palette(half):
    if half == "handler":
        sample = (
            "<script>function go() { openPalette(); }</script>\n"
            "<IconButton icon=\"search\" label=\"Search chats\" on:click={go} />\n"
            "<Button on:click={() => openPalette()}>Search</Button>\n"
            "<Button on:click={other}>Search nothing</Button>\n"
            "<IconButton label=\"Settings\" on:click={go} />\n"
        )
        controls = _search_controls(f"{_SRC}/lib/sample/Side.svelte", sample)
        assert [tag for tag, _ in controls] == ["IconButton", "Button", "Button"], (
            f"the census reads the search controls by their label and their text: {controls}"
        )
        opens = [_opens_palette(attributes, _script(f"{_SRC}/lib/sample/Side.svelte", sample))
                 for _, attributes in controls]
        assert opens == [True, True, False], f"and what their click runs: {opens}"

        assert _imports_name(_SIDEBAR, "openPalette", _PALETTE_STORE), "the sidebar opens the palette through its store"
        controls = _search_controls(_SIDEBAR)
        assert len(controls) >= 2, f"the sidebar has a Search entry, expanded and on the rail: {controls}"
        script = _script(_SIDEBAR)
        dead = [attributes.strip() for _, attributes in controls if not _opens_palette(attributes, script)]
        assert not dead, f"a Search entry that does not open the palette: {dead}"
        assert not _dom_queries(script), f"the sidebar reads no markup to search: {_dom_queries(script)}"
        assert "data-oo-search" not in _code(_SIDEBAR), "no field is looked up by a data attribute"
        return

    for name, props in (("SidebarWide", ""), ("SidebarRail", "collapsed={true}"), ("SidebarPhone", "phone={true}")):
        path = _plant(name, (
            "<script>\n"
            "\timport Sidebar from '$lib/components/layout/Sidebar.svelte';\n"
            "</script>\n"
            f"<Sidebar {props} />\n"
        ))
        root = _dom(ssr().render(path).html)
        entries = [
            b for b in root.iter("button")
            if re.match(r"\s*search\b", " ".join((b.get("aria-label") or b.text()).split()), re.I)
        ]
        assert len(entries) == 1, f"{name}: one Search entry: {len(entries)}"
        assert entries[0].get("aria-haspopup") == "dialog", (
            f"{name}: the entry says it opens a dialog: {entries[0].attrs}"
        )


# ---------------------------------------------------------------------------
# Reading the palette's own markup and script more closely
# ---------------------------------------------------------------------------
_BLOCK = re.compile(r"\{([#/])(if|each|await|key)\b")


def _open_blocks(markup, at):
    """The template blocks (``{#if}``, ``{#each}``, ``{#await}``, ``{#key}``)
    still open at index ``at`` of the markup, innermost last."""
    stack = []
    for match in _BLOCK.finditer(markup, 0, at):
        if match.group(1) == "#":
            stack.append(match.group(2))
        elif stack and stack[-1] == match.group(2):
            stack.pop()
    return stack


def _fragment(markup, slot):
    """The text of ``<svelte:fragment slot="<slot>">...</svelte:fragment>``,
    with its start in the markup, or ``('', -1)``."""
    match = re.search(rf"""<svelte:fragment\s+slot\s*=\s*["']{re.escape(slot)}["']\s*>""", markup)
    if not match:
        return "", -1
    end = markup.find("</svelte:fragment>", match.end())
    return (markup[match.end():end], match.end()) if end >= 0 else ("", -1)


def _reactive(script, name):
    """The expression of ``$: name = ...;`` in a script, or ''."""
    match = re.search(rf"\$:\s*{re.escape(name)}\s*=\s*", script)
    if not match:
        return ""
    depth, end = 0, match.end()
    while end < len(script):
        char = script[end]
        if char in "'\"`":
            end = _skip_string(script, end)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ";" and depth == 0:
            break
        end += 1
    return script[match.end():end]


def _if_body(body, condition):
    """The body of the first ``if (<condition>) {...}`` in a function body
    (``condition`` a pattern), braces balanced, with where it starts; or
    ``('', -1)``."""
    match = re.search(rf"\bif\s*\(\s*{condition}\s*\)\s*\{{", body)
    if not match:
        return "", -1
    end = _balanced(body, match.end(), "{", "}")
    return body[match.end():end - 1], match.start()


def _arrow_body(script, call):
    """The body of the arrow function handed to the first ``call(`` (``onDestroy``
    and the like), or ''."""
    return _call_argument(script, call)


def _ancestor(element, tag):
    node = element.parent
    while node is not None:
        if node.tag == tag:
            return node
        node = node.parent
    return None


def _order(root, element):
    """An element's place in document order."""
    for at, node in enumerate(root.iter()):
        if node is element:
            return at
    return -1


def _palette_at(name, words, phone):
    """The palette rendered for the server at a width (``phone``), opened
    with ``words``, the stop running; the width is set back to the desktop's
    once rendered, since the renderer's stores are shared by the session."""
    path = _plant(name, (
        "<script>\n"
        "\timport { openPalette } from '$lib/stores/palette';\n"
        "\timport { estop } from '$lib/stores/estop';\n"
        "\timport { isPhone } from '$lib/stores/ui';\n"
        "\timport CommandPalette from '$lib/components/palette/CommandPalette.svelte';\n"
        "\testop.update((s) => ({ ...s, stopped: false }));\n"
        f"\tisPhone.set({'true' if phone else 'false'});\n"
        f"\topenPalette({json.dumps(words)});\n"
        "</script>\n"
        "<CommandPalette />\n"
    ))
    try:
        return _dom(ssr().render(path).html)
    finally:
        if phone:
            reset = _plant("PaletteDesktopAgain", (
                "<script>\n\timport { isPhone } from '$lib/stores/ui';\n\tisPhone.set(false);\n</script>\n"
            ))
            ssr().render(reset)


def _options(root):
    return [e for e in root.iter() if e.get("role") == "option"]


def _status(root):
    notes = [e for e in root.iter() if e.get("role") == "status" and "oo-palette-note" in e.classes()]
    assert len(notes) == 1, f"the palette has one status line: {len(notes)}"
    return " ".join(notes[0].text().split())


# ---------------------------------------------------------------------------
# NV26 -- a key runs nothing behind an open dialog, and nothing outside the shell
# ---------------------------------------------------------------------------
def _asked(base, **extra):
    context = dict(base)
    context.update(extra)
    return context


_KEY_CONTEXTS = [
    # A key, the palette open over a chat at rest.
    _asked(_context(chat="c1", palette=True), **{"from": "keys", "shell": True}),
    # A key, the list of shortcuts open.
    _asked(_context(chat="c1", palette=False, help=True), **{"from": "keys", "shell": True}),
    # The palette judging its own commands, open over a chat at rest.
    _asked(_context(chat="c1", palette=True), **{"from": "palette", "shell": True}),
    # A key, no dialog open, in a chat at rest.
    _asked(_context(chat="c1", palette=False), **{"from": "keys", "shell": True}),
    # A key on a page outside both spaces.
    _asked(_context(palette=False), **{"from": "keys", "shell": False}),
    # A key outside both spaces, over the list of shortcuts.
    _asked(_context(palette=False, help=True), **{"from": "keys", "shell": False}),
]


def _keys(chosen=None, context=None, labels=()):
    return _node("keys", ("OO_COMMANDS",), {
        "contexts": _KEY_CONTEXTS, "chosen": chosen or {}, "context": context or _context(palette=False),
        "labels": list(labels),
    })


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_nv26_a_key_runs_nothing_behind_an_open_dialog_and_nothing_outside_the_shell(half):
    if half == "node":
        result = _keys()
        ids = result["ids"]
        palette_key, help_key, from_palette, plain, outside, outside_help = result["judged"]
        assert "close_dialog" in ids and "send_message" in ids and "toggle_theme" in ids, ids
        refused = {c: r for c, r in palette_key.items() if c != "close_dialog"}
        assert refused and all(isinstance(r, str) and "palette" in r.lower() for r in refused.values()), (
            f"a key pressed in the open palette runs nothing behind it (Ctrl+Enter no longer sends the "
            f"draft under it): {palette_key}"
        )
        assert palette_key["close_dialog"] is None, "and closing it still runs"
        refused = {c: r for c, r in help_key.items() if c not in ("close_dialog", "show_shortcuts")}
        assert refused and all(isinstance(r, str) and "shortcuts" in r.lower() for r in refused.values()), (
            f"a key pressed over the open list of shortcuts runs nothing behind it: {help_key}"
        )
        assert help_key["close_dialog"] is None and help_key["show_shortcuts"] is None, (
            f"and Esc or ? still close it: {help_key}"
        )
        for command in ("send_message", "new_chat", "export_conversation", "open_settings", "toggle_theme"):
            assert from_palette[command] is None, (
                f"the palette judges its commands from inside: {command} runs from it: {from_palette[command]}"
            )
            assert plain[command] is None, f"with no dialog open, a key runs {command}: {plain[command]}"
        away = {c: r for c, r in outside.items() if c not in ("toggle_theme", "close_dialog")}
        words = set(away.values())
        assert away and len(words) == 1 and all(isinstance(w, str) and w.strip() for w in words), (
            f"outside both spaces the shell's commands are refused, with one reason: {outside}"
        )
        reason = words.pop()
        assert "palette" not in reason.lower() and "shortcuts" not in reason.lower(), reason
        assert outside["toggle_theme"] is None, f"the theme still switches there: {outside['toggle_theme']}"
        assert outside["close_dialog"] != reason and outside_help["close_dialog"] is None, (
            f"and a dialog still closes: {outside['close_dialog']!r}, {outside_help['close_dialog']!r}"
        )
        assert outside_help["show_shortcuts"] == reason, (
            f"the list of shortcuts opens only in the shell: {outside_help['show_shortcuts']!r}"
        )
        return

    run = _script(_RUN)
    parameters, body = _function(run, "contextAt")
    space = re.search(r"\bconst\s+([\w$]+)\s*=\s*spaceOf\s*\(\s*where\.pathname\s*\)", body)
    assert space and re.search(rf"\bshell\s*:\s*{re.escape(space.group(1))}\s*!==\s*null", body), (
        "the context says whether the path is in a space, by the one rule of the spaces"
    )
    assert re.search(r"\bfrom\s*:\s*where\.from\s*===\s*['\"]palette['\"]\s*\?\s*['\"]palette['\"]\s*:\s*['\"]keys['\"]",
                     body), "the context says a key asks unless the palette says it is asking"
    parameters, body = _function(run, "currentContext")
    assert _calls(body, "contextAt") and not re.search(r"\bfrom\s*:", body), (
        "the runner's own context is a key's"
    )
    script, markup = _script(_PALETTE), _markup(_PALETTE)
    drawn = _reactive(script, "context")
    assert _calls(drawn, "contextAt") and re.search(r"\bfrom\s*:\s*['\"]palette['\"]", drawn), (
        f"the palette judges its commands as the palette: {drawn!r}"
    )
    assert re.search(r"commandOptions\s*\(\s*COMMANDS\s*,\s*context\b", script), (
        "and draws them in that context"
    )
    parameters, body = _function(script, "choose")
    kept = re.search(r"\bconst\s+([\w$]+)\s*=\s*context\s*;", body)
    closed = body.find("closePalette")
    assert kept and 0 <= kept.start() < closed and re.search(
        rf"\brunCommand\s*\(\s*option\.command\s*,\s*{re.escape(kept.group(1))}\s*\)", body
    ), "a command chosen in the palette runs in the context it was drawn in, once the palette has closed"
    assert "closePalette" in _arrow_body(script, "onDestroy"), (
        "unmounted, the palette shuts its store, so it never opens by itself on the next page"
    )


# ---------------------------------------------------------------------------
# NV27 -- the active option is the reader's pick while listed, else the first that can run
# ---------------------------------------------------------------------------
def _opt(ident, group="commands", disabled=False):
    return {"id": ident, "group": group, "disabled": disabled}


_ACTIVE_LIST = [_opt("export", disabled=True), _opt("backup", "settings"), _opt("snap", "settings")]
_REANSWERED = [_opt("c1", "chats"), _opt("export", disabled=True), _opt("backup", "settings"),
               _opt("snap", "settings")]
_ALL_DISABLED = [_opt("a", disabled=True), _opt("b", disabled=True)]


def _listing(data):
    return _node("listing", ("OO_LISTING",), data)


def _listing_input(status=(), paced=()):
    return {
        "flat": _ACTIVE_LIST,
        "picks": [
            [_ACTIVE_LIST, None], [_ACTIVE_LIST, "settings:snap"], [_REANSWERED, "settings:snap"],
            [_REANSWERED, None], [_ACTIVE_LIST, "chats:gone"], [_ALL_DISABLED, None], [[], None],
        ],
        "steps": [
            [_ACTIVE_LIST, "settings:backup", "next"], [_ACTIVE_LIST, "settings:snap", "next"],
            [_ACTIVE_LIST, "commands:export", "previous"], [_ACTIVE_LIST, "settings:snap", "first"],
            [_ACTIVE_LIST, "commands:export", "last"], [_ACTIVE_LIST, None, "next"],
            [_ACTIVE_LIST, None, "previous"], [[], None, "next"],
        ],
        "idCases": [_opt("c 2", "chats"), _opt("c-2", "chats"), _opt("c_2", "chats"), _opt("c:2", "chats"),
                    _opt("export_conversation")],
        "status": list(status),
        "paced": list(paced),
    }


@pytest.mark.parametrize("half", ("node", "wiring", "rendered"))
def test_nv27_the_active_option_is_the_readers_pick_while_listed_else_the_first_that_can_run(half):
    if half == "node":
        result = _listing(_listing_input())
        assert result["keys"] == ["commands:export", "settings:backup", "settings:snap"], result["keys"]
        active = result["active"]
        assert active[0] == "settings:backup", (
            f"with no pick, the first option that can run is active, not a disabled one: {active[0]}"
        )
        assert active[1] == "settings:snap" and active[2] == "settings:snap", (
            f"the reader's pick stays active, however the list is ranked again: {active[1:3]}"
        )
        assert active[3] == "chats:c1", f"with no pick, a new best match that can run is active: {active[3]}"
        assert active[4] == "settings:backup", f"a pick no longer listed gives way to the first that can run: {active[4]}"
        assert active[5] == "commands:a" and active[6] is None, (
            f"with none that can run, the first option; an empty list, none: {active[5:]}"
        )
        assert result["steps"] == [
            "settings:snap", "commands:export", "settings:snap", "commands:export", "settings:snap",
            "commands:export", "settings:snap", None,
        ], f"the arrows move and wrap, Home and End go to the ends, a disabled option on the way: {result['steps']}"
        ids = result["ids"]
        assert len(set(ids)) == len(ids), f"two options never share an id: {ids}"
        assert all(re.fullmatch(r"p-option-[A-Za-z0-9_-]+", i) for i in ids), f"each id is a valid id: {ids}"
        assert "export" in ids[4], f"an id is built from the option's identity: {ids[4]}"
        return

    script, markup = _script(_PALETTE), _markup(_PALETTE)
    if half == "wiring":
        for name in ("activeFor", "stepActive", "optionKey", "optionDomId"):
            assert _imports_name(_PALETTE, name, _LISTING) and _calls(script, name), (
                f"the palette holds its active option through {name}"
            )
        assert re.search(r"\bactiveFor\s*\(\s*flat\s*,\s*pick\s*\)", _reactive(script, "activeKey")), (
            "the active option is the reader's pick while listed, from the listing module"
        )
        assert not re.search(r"\bactive(?:Index)?\s*[+-]\s*1\b|\bactiveIndex\b", script), (
            "the palette counts no places of its own"
        )
        each = re.search(r"\{#each\s+group\.items\s+as\s+\{([^}]*)\}\s*\(\s*([\w$]+)\s*\)\s*\}", markup)
        assert each and re.search(rf"\b{re.escape(each.group(2))}\s*:\s*optionKey\s*\(", script), (
            "the options are keyed by their identity"
        )
        ident = re.search(r"<div\b(?=[^>]*\brole\s*=\s*[\"']option[\"'])[^>]*?\bid\s*=\s*\{\s*([\w$]+)\s*\}", markup, re.S)
        assert ident and re.search(rf"\b{re.escape(ident.group(1))}\s*:\s*optionDomId\s*\(", script), (
            "an option's id in the page is built from its identity"
        )
        assert re.search(r"if\s*\(\s*query\s*!==\s*ranked\s*\)\s*\{[^}]*\bpick\s*=\s*null", script), (
            "new words drop the pick"
        )
        parameters, body = _function(script, "choose")
        refusal, at = _if_body(body, r"option\.disabled")
        assert at >= 0 and re.search(r"\bnotice\s*=", refusal) and "return" in refusal and (
            at < body.find("closePalette")
        ), "a disabled option says why in the status line and runs nothing"
        return

    root = _palette_at("PaletteActiveExport", "export", False)
    options = _options(root)
    assert len(options) >= 2, f"'export' lists several options: {[o.text() for o in options]}"
    assert options[0].get("aria-disabled") == "true" and "Export this chat" in options[0].text(), (
        f"away from a chat, the best match is Export, disabled: {options[0].text()!r}"
    )
    selected = [o for o in options if o.get("aria-selected") == "true"]
    runnable = [o for o in options if o.get("aria-disabled") != "true"]
    assert len(selected) == 1 and runnable and selected[0] is runnable[0], (
        f"the first option that can run is the active one: {[o.text() for o in selected]}"
    )
    combos = [e for e in root.iter() if e.get("role") == "combobox"]
    assert len(combos) == 1 and combos[0].get("aria-activedescendant") == selected[0].get("id"), (
        f"and the field names it: {combos[0].get('aria-activedescendant') if combos else None}"
    )


# ---------------------------------------------------------------------------
# NV28 -- the palette says how many results it lists
# ---------------------------------------------------------------------------
def _state(query="", count=0, asking=False, error=None, notice=None, slow=False):
    return {"query": query, "count": count, "asking": asking, "slow": slow, "error": error, "notice": notice}


def _update(**state):
    state.pop("slow", None)
    return ["update", _state(**state)]


@pytest.mark.parametrize("half", ("node", "rendered"))
def test_nv28_the_palette_says_how_many_results_it_lists(half):
    if half == "node":
        status = [
            _state("exp", 3), _state("exp", 1), _state("zz", 0), _state("", 14),
            _state("soup", 2, asking=True), _state("soup", 2, asking=True, slow=True),
            _state("soup", 2, error="server down"), _state("soup", 4, notice="Export this chat: Open a chat"),
        ]
        paced = [
            [[_update(query="e", count=10), ["advance", 100], _update(query="ex", count=5), ["advance", 100],
              _update(query="exp", count=3), ["advance", 10_000]], False],
            [[_update(query="exp", count=3), ["advance", 10_000], _update(query="", count=14)], False],
            [[_update(query="exp", count=3, notice="Export this chat: Open a chat")], False],
            [[_update(query="soup", count=2, asking=True), ["advance", 10_000],
              _update(query="soup", count=4), ["advance", 10_000]], False],
            [[_update(query="soup", count=2, asking=True), ["advance", 200],
              _update(query="soup", count=4), ["advance", 10_000]], False],
            [[_update(query="exp", count=3), ["close"], ["advance", 10_000]], False],
            [[_update(query="exp", count=3)], True],
        ]
        result = _listing(_listing_input(status, paced))
        assert result["status"] == [
            "3 results", "1 result", 'Nothing matches "zz".', "", "", "2 results, still searching the chats",
            "2 results. The chats could not be searched: server down", "Export this chat: Open a chat",
        ], f"the status line: {result['status']}"
        slow, settle = result["limits"]["slow"], result["limits"]["settle"]
        assert 0 < settle < slow <= 2000, f"a line waits a moment; a slow search waits longer: {settle}, {slow}"
        burst, cleared, notice, slow_search, quick, closed, server = result["paced"]
        assert burst[4] == [] and burst[5] == ["3 results"], (
            f"a burst of keys is read once it settles, its last line only: {burst}"
        )
        assert cleared[2][-1] == "", f"cleared words silence the line at once: {cleared}"
        assert notice[0] == ["Export this chat: Open a chat"], f"a notice is read at once: {notice}"
        assert slow_search[0] == [""] and "2 results, still searching the chats" in slow_search[1] and (
            slow_search[3][-1] == "4 results"
        ), f"a slow search says so, then the count: {slow_search}"
        assert not any("still searching" in line for line in quick[-1]) and quick[-1][-1] == "4 results", (
            f"a quick answer is read as its count alone: {quick}"
        )
        assert closed[-1] == [], f"closed, the line says nothing more: {closed}"
        assert server[0] == ["3 results"], f"rendered on the server, a line is read at once: {server}"
        return

    counted = _palette_at("PaletteCountExport", "export", False)
    count = len(_options(counted))
    assert count >= 2 and _status(counted) == f"{count} results", (
        f"the status line says how many results are listed: {_status(counted)!r} for {count}"
    )
    none = _palette_at("PaletteCountNone", "zzqqxxnothing", False)
    assert not _options(none) and _status(none) == 'Nothing matches "zzqqxxnothing".', _status(none)
    script, markup = _script(_PALETTE), _markup(_PALETTE)
    assert _imports_name(_PALETTE, "createStatusLine", _LISTING) and _calls(script, "createStatusLine"), (
        "the palette paces its status line with createStatusLine"
    )
    assert re.search(r"\.update\s*\(\s*\{[^}]*\bcount\s*:\s*flat\.length\b", script), (
        "and hands it the number of options listed"
    )
    assert _status(_palette_at("PaletteCountEmpty", "", False)) == "", "with no words it says nothing"


# ---------------------------------------------------------------------------
# NV29 -- Stop all heads the palette, at every width and every moment
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", ("head", "desktop", "phone", "touch"))
def test_nv29_stop_all_heads_the_palette_at_every_width_and_every_moment(half):
    if half == "head":
        sample = "{#if $isPhone}<StopAllButton />{/if}\n{#each rows as row}{/each}<StopAllButton />"
        assert _open_blocks(sample, sample.find("<StopAllButton")) == ["if"] and not _open_blocks(
            sample, sample.rfind("<StopAllButton")
        ), "the census reads the blocks around a mount"
        markup = _markup(_PALETTE)
        mounts = [m.start() for m in re.finditer(r"<StopAllButton\b", markup)]
        assert len(mounts) == 1, f"the palette mounts the stop once: {len(mounts)}"
        actions, start = _fragment(markup, "actions")
        assert start >= 0 and start <= mounts[0] < start + len(actions), (
            "the stop sits in the dialog's head, in its actions slot"
        )
        assert not _open_blocks(markup, mounts[0]), (
            f"and in no block, so no width and no moment hides it: {_open_blocks(markup, mounts[0])}"
        )
        modal = _markup(_MODAL)
        header = re.search(r"<header\b[^>]*>(.*?)</header>", modal, re.S)
        assert header, "the ds Modal has a head"
        slot, close = header.group(1).find('name="actions"'), header.group(1).find("oo-modal-close")
        assert 0 <= slot < close, "its head holds an actions slot, before the close button"
        tag = _tags(markup, "StopAllButton")[0]
        assert re.search(r"""\bplacement\s*=\s*["']dialog-head["']""", tag), tag
        opens = re.search(r"['\"]dialog-head['\"]\s*:\s*['\"]([\w-]+)['\"]", _script(_STOP_ALL))
        assert opens and opens.group(1).startswith("bottom"), (
            "its confirmation opens down from the head, over the dialog"
        )
        return

    if half == "touch":
        from test_shell_contracts import _rules, _style

        modal = _rules(_style(_MODAL))
        coarse = [d for s, d in modal if "pointer: coarse" in s and ".oo-modal-close" in s]
        assert coarse and coarse[0].get("width") == "44px" and coarse[0].get("height") == "44px", (
            f"on a touch screen the dialog's close button is a 44 px target: {coarse}"
        )
        rules = _rules(_style(_PALETTE))
        phone = {s: d for s, d in rules if "data-phone='true'" in s}
        field = [d for s, d in phone.items() if "oo-field-control" in s]
        assert field and re.search(r"max\(\s*16px", field[0].get("font-size", "")), (
            f"on a phone the field's type is 16 px at least, so the phone does not zoom into it: {field}"
        )
        reason = [d for s, d in phone.items() if s.endswith(".oo-palette-reason")]
        assert reason and reason[0].get("white-space") == "normal" and reason[0].get("flex-basis") == "100%", (
            f"on a phone a disabled command's reason wraps under its name, whole: {reason}"
        )
        row = [d for s, d in phone.items() if s.endswith(".oo-palette-option")]
        assert row and row[0].get("min-height") == "44px" and row[0].get("flex-wrap") == "wrap", (
            f"and every option is a 44 px target that can take a second line: {row}"
        )
        return

    phone = half == "phone"
    for at, words in enumerate(("", "zzqqxxnothing", "export")):
        root = _palette_at(f"PaletteHead{half.title()}{at}", words, phone)
        dialogs = list(root.iter("dialog"))
        assert len(dialogs) == 1, f"{half} {words!r}: one dialog"
        stops = [stop for stop in _stops(root) if _inside(stop, dialogs[0])]
        assert len(stops) == 1, f"{half} {words!r}: the palette holds Stop all: {len(stops)}"
        combos = [e for e in root.iter() if e.get("role") == "combobox"]
        assert len(combos) == 1 and 0 <= _order(root, stops[0]) < _order(root, combos[0]), (
            f"{half} {words!r}: Stop all comes before the field, at the top, where a phone's keyboard never "
            f"covers it"
        )
        assert _ancestor(stops[0], "header") is not None, f"{half} {words!r}: in the dialog's head"
        if phone:
            assert stops[0].get("data-size") == "lg", (
                f"{words!r}: on a phone, Stop all is a 44 px target: {stops[0].get('data-size')}"
            )


# ---------------------------------------------------------------------------
# NV30 -- the keys shown are the reader's own
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", ("node", "wiring"))
def test_nv30_the_keys_the_palette_and_the_sidebar_show_are_the_readers_own(half):
    if half == "node":
        chosen = {
            "search_conversations": {"key": "p", "ctrl": True, "shift": True},
            "stop_reply": {"key": "x", "ctrl": True},
            "new_chat": {"alt": True},
        }
        result = _keys(chosen, _context(palette=False),
                       labels=[{"key": "enter", "ctrl": True}, {"key": "escape"}, {"key": "k", "ctrl": True}])
        bindings, defaults = result["bindings"], result["defaults"]
        assert bindings is not None and defaults is not None, "the registry reads the reader's keys: chosenBindings"
        assert defaults["search_conversations"] == {"key": "k", "ctrl": True, "shift": False, "alt": False}, (
            f"with no choice, the registry's keys: {defaults['search_conversations']}"
        )
        assert bindings["search_conversations"] == {"key": "p", "ctrl": True, "shift": True, "alt": False}, (
            f"the reader's keys stand over the registry's: {bindings['search_conversations']}"
        )
        assert bindings["new_chat"] == {"key": "n", "ctrl": True, "shift": False, "alt": True}, (
            f"field by field: {bindings['new_chat']}"
        )
        assert "stop_reply" not in bindings, "a choice for a command the registry does not bind is ignored"
        details = {o["id"]: o["detail"] for o in result["options"]}
        plain = {o["id"]: o["detail"] for o in result["plainOptions"]}
        assert details["search_conversations"] == "Ctrl + Shift + P" and plain["search_conversations"] == "Ctrl + K", (
            f"the palette shows the reader's keys: {details['search_conversations']!r}"
        )
        shortcuts = {s["action"]: s for s in result["shortcuts"]}
        assert (shortcuts["search_conversations"]["key"], shortcuts["search_conversations"]["shift"]) == ("p", True), (
            "and the handler runs by them"
        )
        assert result["labels"] == ["Ctrl + Enter", "Esc", "Ctrl + K"], (
            f"a key the server keeps in lower case reads as its name: {result['labels']}"
        )
        return

    assert re.search(r"commandOptions\s*\(\s*COMMANDS\s*,\s*context\s*,\s*\$shortcutKeys\s*\)", _script(_PALETTE)), (
        "the palette shows each command with the reader's keys"
    )
    sidebar = _script(_SIDEBAR)
    assert _imports_name(_SIDEBAR, "shortcutKeys", _SHORTCUT_KEYS) and re.search(
        r"\$shortcutKeys\.search_conversations\b", sidebar
    ), "the sidebar's hint shows the reader's keys for the palette"
    assert not _imports_name(_SIDEBAR, "COMMANDS", _COMMANDS), "and not the registry's own"
    handler = _script(_SHORTCUTS)
    kept = re.search(r"\$:\s*([\w$]+)\s*=\s*chosenBindings\s*\(\s*COMMANDS\s*,\s*chosen\s*\)", handler)
    assert kept and _imports_name(_SHORTCUTS, "chosenBindings", _COMMANDS), (
        "the shortcut handler reads the reader's keys through the registry"
    )
    name = re.escape(kept.group(1))
    assert re.search(rf"\bshortcutKeys\.set\s*\(\s*{name}\s*\)", handler) and re.search(
        rf"\bdefaultShortcuts\s*\(\s*COMMANDS\s*,\s*{name}\s*\)", handler
    ), "runs by them, and writes them to their store"
    assert _imports_name(_SHORTCUTS, "shortcutKeys", _SHORTCUT_KEYS), "the store of the keys"
    store = _script(_SHORTCUT_KEYS)
    assert re.search(r"\bexport\s+const\s+shortcutKeys\s*=\s*writable\b[^;]*chosenBindings\s*\(\s*COMMANDS\s*\)",
                     store), "the store starts from the registry's keys"


# ---------------------------------------------------------------------------
# NV31 -- the palette offers Stop all, and never stops by itself
# ---------------------------------------------------------------------------
_STOP_WORDS = ("stop", "stop all", "emergency", "halt")


def _finds():
    return _node(
        "finds", ("OO_SOURCES", "OO_RANK", "OO_DESTINATIONS", "OO_CATALOG", "OO_SETTINGS_SEARCH"),
        {"queries": list(_SETTING_QUERIES), "stopWords": list(_STOP_WORDS),
         "commands": [_item("stop_reply", "Stop this reply", keywords=["cancel"]), _item("new_chat", "New chat")],
         "more": [["soup", "/chat", 7, 6], ["soup", "/chat", 6, 6], ["  ", "/chat", 9, 6],
                  [" a b&c ", "/chat/", 20, 6]],
         "chats": [{"id": f"c{n}", "title": f"Soup {n}"} for n in range(1, 9)]},
    )


@pytest.mark.parametrize("half", ("node", "wiring", "rendered"))
def test_nv31_the_palette_offers_stop_all_and_never_stops_by_itself(half):
    if half == "node":
        result = _finds()
        stop = result["stop"]
        assert stop is not None, "the sources offer a Stop all entry: stopOption"
        assert stop["group"] == "commands" and stop["label"] == "Stop all" and stop["stop"] is True, stop
        assert not stop.get("command") and not stop.get("href") and stop["disabled"] is False, (
            f"the entry runs no command and goes nowhere: it asks the stop control: {stop}"
        )
        refused = result["refused"]
        assert refused["disabled"] is True and refused["reason"] == "Everything is stopped already", (
            f"refused by the stop control, it is listed disabled with the control's words: {refused}"
        )
        assert result["blank"]["disabled"] is False, "a blank refusal refuses nothing"
        for words, groups in result["stopRank"].items():
            listed = [i for g in groups for i in g["items"]]
            assert "stop_all" in listed, f"{words!r} finds Stop all: {groups}"
        first = result["stopRank"]["stop all"][0]["items"][0]
        assert first == "stop_all", f"typed whole, Stop all comes first: {result['stopRank']['stop all']}"
        return

    script = _script(_PALETTE)
    if half == "wiring":
        bound = re.search(r"<StopAllButton\b[^>]*\bbind:refusal\s*=\s*\{\s*([\w$]+)\s*\}", _markup(_PALETTE))
        assert bound and re.search(rf"\bstopOption\s*\(\s*{re.escape(bound.group(1))}\s*\)", _reactive(script, "stop")), (
            "the palette lists Stop all with the refusal its stop control reads, the one reader of the stop"
        )
        assert _imports_name(_PALETTE, "stopOption", _SOURCES), "from the sources"
        parameters, body = _function(script, "choose")
        asked, at = _if_body(body, r"option\.stop")
        assert at >= 0 and re.search(r"\.askToConfirm\s*\(\s*\)", asked) and "return" in asked and (
            at < body.find("closePalette")
        ), "choosing Stop all opens the stop control's confirmation, and the palette stays open"
        assert not re.search(r"\bengage(?:Stop)?\s*\(", _code(_PALETTE)), "the palette never stops anything itself"
        control = _script(_STOP_ALL)
        match = re.search(r"\bexport\s+async\s+function\s+askToConfirm\s*\(", control)
        assert match, "the stop control can be asked to confirm"
        parameters, ask = _function(control, "askToConfirm")
        assert re.search(r"\bconfirming\s*=\s*true\b", ask) and not re.search(r"\bengage(?:Stop)?\s*\(", ask), (
            "asked, it opens its confirmation and stops nothing"
        )
        assert re.search(r"\$:\s*refusal\s*=", control) and re.search(r"\bexport\s+let\s+refusal\b", control), (
            "and says why it cannot be asked, for its parent"
        )
        return

    root = _palette_at("PaletteStopEntry", "stop all", False)
    options = _options(root)
    assert options and "Stop all" in options[0].text() and options[0].get("aria-disabled") != "true", (
        f"typing Stop all lists the entry first, ready: {[o.text() for o in options]}"
    )
    assert options[0].get("aria-selected") == "true", "and it is the active option"
    emergency = [o for o in _options(_palette_at("PaletteStopEmergency", "emergency", False)) if "Stop all" in o.text()]
    assert len(emergency) == 1, "the words of an emergency find it too"


# ---------------------------------------------------------------------------
# NV32 -- the notification history is a command
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", ("registry", "wiring"))
def test_nv32_the_notification_history_is_a_command_of_the_palette(half):
    if half == "registry":
        registry = _node("registry", ("OO_COMMANDS",), {"spaces": _space_contexts()})
        commands = {c["id"]: c for c in registry["commands"]}
        assert "show_notifications" in commands and "notification" in commands["show_notifications"]["label"].lower(), (
            f"the notification history is a command: {sorted(commands)}"
        )
        result = _keys()
        plain = result["judged"][3]
        assert plain["show_notifications"] is None, "it runs wherever the shell is"
        return

    handlers = _object_entries(_script(_RUN), "HANDLERS")
    entry = handlers.get("show_notifications", "") if handlers else ""
    helper = re.search(r"([\w$]+)\s*\(", entry)
    body = _function(_script(_RUN), helper.group(1))[1] if helper else ""
    assert body and _calls(body, "openPreferences") and _calls(body, "openNotifications") and (
        body.find("openPreferences") < body.find("openNotifications")
    ), f"the command goes to Preferences, then opens the history: {entry!r}"
    assert _imports_name(_RUN, "openNotifications", _NOTIFICATION_STORE), "through its store"
    store = _script(_NOTIFICATION_STORE)
    for name in ("notificationCenter", "openNotifications", "closeNotifications", "toggleNotifications"):
        assert re.search(rf"\bexport\s+(?:const|function)\s+{name}\b", store), f"the store exports {name}"
    center, panel = _script(_NOTIFICATION_CENTER), _markup(_NOTIFICATION_CENTER)
    assert _imports_name(_NOTIFICATION_CENTER, "notificationCenter", _NOTIFICATION_STORE) and re.search(
        r"\{#if\s+\$notificationCenter\s*\}", panel
    ), "the history's panel is open while its store says so"
    assert re.search(r"on:click\s*=\s*\{\s*toggleNotifications\s*\}", panel), "and its bell goes through the store"
    assert not re.search(r"\blet\s+expanded\b", center), "it keeps no state of its own"
    assert "closeNotifications" in _arrow_body(center, "onDestroy"), "leaving the page shuts it"


# ---------------------------------------------------------------------------
# NV33 -- the palette finds what the settings search and the chats index find
# ---------------------------------------------------------------------------
_SETTING_QUERIES = ("plugins & extensions", "account", "chunk")


@pytest.mark.parametrize("half", ("settings", "chats", "wiring"))
def test_nv33_the_palette_finds_what_the_settings_search_and_the_chats_index_find(half):
    if half == "wiring":
        script = _script(_PALETTE)
        more = _reactive(script, "more")
        assert re.search(r"^\s*answered\s*\?\s*moreChatsOption\s*\(\s*query\s*,\s*chatsHref\s*,\s*found\.hits\.length\s*,"
                         r"\s*GROUP_LIMIT\s*\)", more), (
            f"once the server has answered these words, the palette offers the chats index on them: {more!r}"
        )
        assert _imports_name(_PALETTE, "moreChatsOption", _SOURCES) and _imports_name(_PALETTE, "GROUP_LIMIT", _RANK), (
            "from the sources, measured against the group's own limit"
        )
        assert re.search(r"\bmore\b", _call_argument(script, "rankPalette")), "and ranks it with the rest"
        parameters, body = _function(_script(_SETTINGS_SEARCH), "settingsIndex")
        assert re.search(r"\bformer\s*:\s*former\?\.label\b", body), (
            "the settings index names the old section of each group"
        )
        return

    result = _finds()
    if half == "settings":
        index = {hit["id"]: hit for hit in result["index"]}
        for option in result["settings"]:
            hit = index[option["id"]]
            words = set(option["keywords"])
            for field in ("description", "where", "former"):
                if hit[field]:
                    assert hit[field] in words, (
                        f"a group is found by its {field}, as the settings search finds it: {option['id']}"
                    )
        assert any(hit["former"] for hit in result["index"]), "the index names the old sections"
        for query in _SETTING_QUERIES:
            hub, palette = result["hub"][query], result["palette"][query]
            assert hub, f"the settings search finds something for {query!r}"
            missing = sorted(set(hub) - set(palette))
            assert not missing, f"what the settings search finds for {query!r}, the palette finds too: {missing}"
        return

    more = result["more"]
    assert more is not None, "the sources offer the chats index on the words: moreChatsOption"
    assert more[0] and more[0]["href"] == "/chat?q=soup" and more[0]["group"] == "chats" and (
        more[0]["trailing"] is True and "soup" in more[0]["label"]
    ), f"when the server found more chats than the group shows, a last entry opens the index on the words: {more[0]}"
    assert more[1] is None and more[2] is None, f"not when it found no more, nor without words: {more[1:3]}"
    assert more[3] and more[3]["href"] == "/chat?q=a%20b%26c", f"the words travel whole and escaped: {more[3]}"
    ranked = result["moreRank"]
    assert ranked == [{"id": "chats", "items": ["c1", "c2", "c3", "c4", "c5", "c6", "more_chats"]}], (
        f"it closes the chats group, past the group's limit: {ranked}"
    )
    assert result["moreEmpty"] == [{"id": "chats", "items": ["c1", "c2", "c3", "c4", "c5", "c6"]}], (
        f"and is never listed without words: {result['moreEmpty']}"
    )
    assert result["moreAlone"] == [], f"nor alone, in a group that lists nothing else: {result['moreAlone']}"


# ---------------------------------------------------------------------------
# NV34 -- the palette follows its source, the keys run only the runner, nothing hides
# ---------------------------------------------------------------------------
_FOLLOW = [[True, "  soup "], [True, "soup"], [True, "soups"], [False, "soups"], [False, ""],
           [True, ""], [True, "x"], [True, " x "]]
_INK = re.compile(r"var\(\s*--oo-([\w-]+)\s*\)")


def _inks(css):
    """The colour of every rule's text, as ``(selector, token)``: the token a
    ``color`` reads, or the value when it reads none."""
    from test_shell_contracts import _rules

    found = []
    for selector, declarations in _rules(css):
        value = declarations.get("color")
        if value is None:
            continue
        match = _INK.fullmatch(value)
        found.append((selector, match.group(1) if match else value))
    return found


_TEXT_INKS = re.compile(r"fg-(?:primary|secondary|muted|stop)$")
_KEYWORDS = {"if", "for", "while", "switch", "return", "catch", "typeof", "function"}


def _key_calls(body):
    """The functions and methods a function body calls, keywords aside."""
    called = set(re.findall(r"(?<![\w$.])([\w$]+)\s*\(", body)) | set(re.findall(r"\.([\w$]+)\s*\(", body))
    return called - _KEYWORDS


@pytest.mark.parametrize("half", ("follow", "follows", "handler", "hidden", "inks"))
def test_nv34_the_palette_follows_its_source_the_keys_run_the_runner_and_nothing_hides(half):
    if half == "follow":
        result = _node("follow", ("OO_CONVERSATION_SOURCE",), _FOLLOW)
        assert result["log"] == [["ask", "soup"], ["ask", "soups"], ["close"], ["ask", ""], ["ask", "x"]], (
            f"open, each change of the words is asked once, trimmed; closing closes once; opened again, "
            f"it asks again: {result['log']}"
        )
        assert result["after"] == [1, 1, 2, 3, 3, 4, 5, 5], result["after"]
        return

    script, markup = _script(_PALETTE), _markup(_PALETTE)
    if half == "follows":
        built = re.search(r"\b(?:const|let)\s+([\w$]+)\s*=\s*followPalette\s*\(", script)
        argument = _call_argument(script, "followPalette")
        assert built and _imports_name(_PALETTE, "followPalette", _CONVERSATION_SOURCE) and re.search(
            r"^\s*\(\s*([\w$]+)\s*\)\s*=>\s*chatSource\.search\s*\(\s*\1\s*\)\s*,\s*\(\s*\)\s*=>\s*chatSource\.close\s*\(\s*\)\s*$",
            argument,
        ), f"the palette's source is followed by the follower, which asks it and closes it: {argument!r}"
        assert re.search(rf"\$:\s*if\s*\(\s*browser\s*\)\s*{re.escape(built.group(1))}\s*\(\s*\$palette\.open\s*,\s*words\s*\)",
                         script), "handed whether the palette is open and the words in its field"
        assert not re.search(r"\bchatSource\s*\.\s*(?:search|close)\s*\(", script.replace(argument, "")), (
            "and asks the source nothing else"
        )
        return

    if half == "handler":
        sample = (
            "function onKeydown(e) { for (const s of all) { if (s.action === 'send_message') { "
            "window.dispatchEvent(new CustomEvent('opti-send')); } runCommand(s.action); } }"
        )
        events = re.compile(r"\bdispatchEvent\s*\(|\bnew\s+CustomEvent\s*\(")
        chosen = re.compile(r"\baction\s*[!=]==|[!=]==\s*['\"][a-z]+_[a-z_]+['\"]")
        assert len(events.findall(sample)) == 2 and len(chosen.findall(sample)) == 1, (
            "the census reads an event of its own and a command picked out by name"
        )
        assert _key_calls(_function(sample, "onKeydown")[1]) == {"CustomEvent", "dispatchEvent", "runCommand"}, (
            f"and what a key calls: {_key_calls(_function(sample, 'onKeydown')[1])}"
        )
        handler = _script(_SHORTCUTS)
        assert not events.findall(handler), (
            f"the shortcut handler sends no event of its own: {events.findall(handler)}"
        )
        parameters, body = _function(handler, "onKeydown")
        assert body and not chosen.findall(body), (
            f"a key picks out no command by name to act on it itself: {chosen.findall(body)}"
        )
        assert _key_calls(body) <= {"matchesShortcut", "runCommand", "preventDefault"}, (
            f"a key does what the runner does, and nothing more: {sorted(_key_calls(body))}"
        )
        return

    if half == "hidden":
        sample = "const a = rows.filter((o) => !o.disabled); const b = rows.filter((row) => row.disabled !== true);"
        hides = [arg for arg in _filter_arguments(sample) if re.search(r"\bdisabled\b", arg)]
        assert len(hides) == 2, f"the census reads a filter on disabled, in either spelling: {hides}"
        found = [arg for arg in _filter_arguments(script) if re.search(r"\bdisabled\b", arg)]
        assert not found, f"the palette hides no disabled command: {found}"
        return

    sample = ".a { color: var(--oo-fg-muted); } .b { color: var(--oo-edge); } .c { background: red; }"
    assert _inks(sample) == [(".a", "fg-muted"), (".b", "edge")], f"the census reads the inks: {_inks(sample)}"
    from test_shell_contracts import _style

    for path in (_PALETTE, _SHORTCUTS):
        inks = _inks(_style(path))
        assert inks, f"{path}: the census reads its inks"
        wrong = [(selector, ink) for selector, ink in inks if not _TEXT_INKS.fullmatch(ink)]
        assert not wrong, f"{path}: every text is drawn in a text ink, whose contrast the palettes pin: {wrong}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
