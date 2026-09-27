#!/usr/bin/env python3
"""Contracts for the Workshop's pages: quiet groups, one title each, one place per panel.

A Workshop settings page lists its groups as quiet rows, each titled once, by
the design system's ``PanelHeader``. One group is open at a time, and only
the open group mounts its panel, except a group whose panel was used on
this visit (a field typed in, a choice clicked, a file dropped), which
stays mounted and hidden until the page is left, so its draft is not
thrown away when another group opens, nor when the page's search shows its
results. The address names the open group (``?g=``), so the palette, the
settings search and a shared link open it. Observability
is listed once: its group hosts the telemetry, its history, the profiler
and the performance dashboard as tabs drawn from the catalog, which marks
them embedded in it. Every panel is mounted by one place, and nothing that
no page reaches stays in the source. The Security page opens on the grade's
checks, the session mode and the recent security events. The first-run
dialog refreshes what a preset changes instead of reloading the page, and
it holds Stop all in its head, as every dialog of a Workshop page does. The
benchmarks page is one row of tabs over one engine.

The names the contracts read:

  * ``lib/ds/PanelHeader.svelte`` -- ``title`` (required: a blank one
    refuses to render, "PanelHeader needs a title"), ``level`` (2, 3 or 4,
    3 by default), ``description``, ``headingId``, ``expanded`` (undefined
    for a plain heading; a boolean makes the title a disclosure button
    inside the heading) and ``controls``; the event ``toggle``; the slot
    ``actions``.
  * ``lib/settings/disclosure.ts`` -- ``GROUPS_CONTEXT`` (the context key,
    the string ``oo-settings-groups``), ``openOnArrival(g, pageGroupIds,
    hostOf)``, ``toggled(open, id)``, ``addressFor(url, open)``,
    ``mounted(id, open, edited)``, ``withEdited(edited, id)``,
    ``DRAFT_EVENTS`` and ``startsDraft(type, key)``. The hub puts ``{
    level, collapsible, open, edited, toggle, markEdited }`` in that
    context; ``open`` and ``edited`` are stores.
  * ``lib/ds/firstFocus.ts`` -- ``firstFocus(candidates, marked,
    inActions)``, where the ds ``Modal`` opens; ``Tabs.svelte`` exports
    ``focusSelected()``.
  * ``lib/settings/catalog.ts`` -- ``embeddedGroups(hostId)``.
  * ``lib/observability/state.ts`` -- ``telemetryState(stats)``,
    ``profilerState(summary)`` and ``historyState(stats)``: a state in
    words, ``Unavailable`` when nothing was read; ``tabUnavailable(feature,
    featureMap)``. ``ObservabilityPanel.svelte`` takes ``active`` and
    ``featureMap``, which it sets itself when mounted.
  * ``lib/api/security.ts`` -- ``getSecurityStatus()`` and
    ``getSecurityEvents(limit)``; ``SecurityChecks.svelte`` takes
    ``status``; ``SecurityGrade.svelte`` takes ``status``, ``failure``,
    ``events``, ``eventsFailure`` and ``expanded``, which it loads itself
    when mounted.
  * ``lib/stores/modelRoles.ts`` -- ``roles``, ``installedModels``,
    ``rolesRead`` (``{ state, reason }``), ``saveErrors`` (a failure per
    role), ``loadRoles()`` and ``saveRole(role, assignment)``, which says
    true or false and never throws. A role's card carries ``data-role``.
  * ``lib/stores/configRefresh.ts`` -- ``configEpoch`` and
    ``refreshAfterConfigChange()``. The first-run dialog takes ``visible``,
    ``step``, ``presets``, ``detection`` and ``selectedPresetId``, which
    it sets itself when mounted.
  * ``lib/benchmark/tab.ts`` -- ``BENCHMARK_TABS``, ``benchmarkTab(url)``
    and ``tabAddress(url, tab)``; ``BenchmarkPage.svelte`` takes ``tab``
    and dispatches ``change``.

  * PF23 -- ``PanelHeader`` draws its title as a heading of the level given;
    given ``expanded``, the title is a button inside that heading, with
    ``aria-expanded`` and ``aria-controls``, named by the title alone, and
    the description stands outside the heading as the button's description;
    without it the heading holds no button; a blank title refuses to render;
    the button's hit area is stretched over the whole row, and the actions
    sit above it.
  * PF24 -- one title per group: ``SettingsGroup`` draws its title only
    through ``PanelHeader``; a Workshop page draws no section heading under
    its h1; the group's level comes from the hub (2 in the Workshop, 3 in
    Preferences); no Workshop group repeats its page's name; no panel the
    hub mounts, nor anything it mounts, writes a heading at or above its
    group's level; and none of the titles the panels used to repeat is a
    heading of theirs any more.
  * PF25 -- the folding rules are one pure module: the group ``g`` names
    opens (its host for an embedded group), else the page's only group, else
    none; a toggle closes the open group or opens another; the address sets
    or drops ``g`` and keeps the rest; a group mounts while open or edited;
    a group joins the edited ones without the set being changed; a field
    typed in, a click, a key on a control or a drop may start a draft, a
    key that only moves through the page does not. The hub and
    ``SettingsGroup`` call it and hold no rule of their own.
  * PF26 -- a closed, unedited group renders no panel; groups fold on a
    Workshop page only; the context's ``open`` and ``edited`` are stores
    the hub creates; ``open`` follows ``?g=`` on an arrival alone, and an
    address the hub asked for itself neither re-runs the arrival nor
    scrolls; a toggle replaces the address and keeps focus and scroll; the
    body a button controls is always there, hidden while closed, the group's
    frame is no named region, and a Workshop page's section is named by its
    h1; an arrival focuses the opened group's button without scrolling for
    it; the group watches its panel in the capture phase for what may start
    a draft and marks itself edited, and the page's search hides the groups
    rather than dropping them.
  * PF27 -- the four dashboards are embedded in the Observability group, in
    catalog order; the hub loads exactly the panels of the placed groups
    (the successor of UX17's node half, deselected by name in
    ``pyproject.toml``); every group's panel is imported by one file.
  * PF28 -- the Observability host: one row of ds tabs, Overview then the
    embedded groups; every feature key the catalog names is served by the
    health map and is the gate its panel's routes check; each tab gated by
    its own group's key, and only by it; the overview says each state in
    words, on the sunken ground; the host draws its tabs and states from
    the catalog and the state module; the profiler's pick opens the history
    filtered, and clearing the filter draws it afresh; every client helper
    an API module calls is imported; a control of the host that changes the
    tab hands focus to the selected tab.
  * PF29 -- ``ConfirmDialog`` forwards an ``actions`` slot to its head;
    every dialog of the Workshop's pages holds Stop all there; the dialog's
    first focus never falls on the head's actions, and a marked element
    still wins.
  * PF30 -- every module and stylesheet under ``lib`` is reached from a
    route, but the named exceptions (a ledger that only shrinks); the
    orphans are gone; the benchmark sections' stylesheet is imported by the
    page that renders them.
  * PF31 -- the security grade is read through one client, and the Security
    page opens on it: each check with its points and the word Passed or Not
    passed, a named failure with a retry that keeps it until the answer and
    focuses the grade once read, the session mode and the recent security
    events.
  * PF32 -- model assignment is mounted once, says when its roles could not
    be read (a retry reads quietly, keeping the failure until the answer),
    says "none found" only after a read that found none, shows a save's
    failure under its role, and keeps the editor open when a save fails.
  * PF33 -- the first run refreshes instead of reloading: what a preset
    changes is read again, whichever way the dialog closes once a preset is
    applied, and the chat's control bar re-reads its switches when the
    epoch moves after it was built; the dialog keeps its name and its close
    button, holds Stop all in its head, cannot be closed while a preset is
    being applied, hands focus to what each step shows, and its presets are
    one radio group.
  * PF34 -- the benchmarks page is one row of ds tabs over the evaluation
    engine's seven sections; the tab lives in the address; a History run
    link comes back to History; nothing reaches the older engine; System
    status measures component latency; the ds tab row scrolls within
    itself, shows the selected tab when it first shows and keeps it in
    view, and keeps room for the focus ring on every side.

Every census carries a standing positive fixture, so a probe gone blind
turns red instead of reading a false zero. The server-rendering halves
render components through ``tests/_frontend.ssr()``, some through small
wrappers planted in the renderer's copy; they prove what the templates emit
when compiled for the server, not what a browser does. Keyboard, focus
movement, scrolling, the requests a page fires and the colours are owed to
the machine.

Local-only (the public distribution ships no tests). Needs Node >= 22.6 and
``frontend/node_modules``; without them the helpers raise, and so do the
contracts.
"""

import ast
import html
import json
import posixpath
import re
import sys
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_navigation_contracts as _nav  # noqa: E402
import test_ui_repairs_contracts as _repairs  # noqa: E402
from _frontend import REPO, check_ledger, files, read, run_ts, ssr  # noqa: E402
from test_shell_contracts import _inside, _name, _stops  # noqa: E402
from test_ui_primitives_contracts import _dom  # noqa: E402
from test_ui_surface_contracts import _declarations, _rules  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file. The first server-rendering contract starts the
# session's server when this suite runs alone.
BUDGET_S = {
    "test_pf23_a_panel_header_titles_once_and_folds_by_its_whole_row[level]": 6.0,
    "test_pf23_a_panel_header_titles_once_and_folds_by_its_whole_row[disclosure]": 2.0,
    "test_pf23_a_panel_header_titles_once_and_folds_by_its_whole_row[plain]": 2.0,
    "test_pf23_a_panel_header_titles_once_and_folds_by_its_whole_row[blank]": 2.0,
    "test_pf23_a_panel_header_titles_once_and_folds_by_its_whole_row[row]": 1.0,
    "test_pf24_every_group_is_titled_once_at_its_own_level[own]": 1.0,
    "test_pf24_every_group_is_titled_once_at_its_own_level[page]": 2.0,
    "test_pf24_every_group_is_titled_once_at_its_own_level[level]": 2.0,
    "test_pf24_every_group_is_titled_once_at_its_own_level[same]": 2.0,
    "test_pf24_every_group_is_titled_once_at_its_own_level[levels]": 2.0,
    "test_pf24_every_group_is_titled_once_at_its_own_level[listed]": 1.0,
    "test_pf25_the_folding_rules_are_one_pure_module[named]": 2.0,
    "test_pf25_the_folding_rules_are_one_pure_module[embedded]": 2.0,
    "test_pf25_the_folding_rules_are_one_pure_module[default]": 2.0,
    "test_pf25_the_folding_rules_are_one_pure_module[toggle]": 2.0,
    "test_pf25_the_folding_rules_are_one_pure_module[address]": 2.0,
    "test_pf25_the_folding_rules_are_one_pure_module[kept]": 2.0,
    "test_pf25_the_folding_rules_are_one_pure_module[edited]": 2.0,
    "test_pf25_the_folding_rules_are_one_pure_module[draft]": 2.0,
    "test_pf25_the_folding_rules_are_one_pure_module[wiring]": 1.0,
    "test_pf26_one_group_open_at_a_time_and_only_it_or_an_edited_one_mounted[lazy]": 2.0,
    "test_pf26_one_group_open_at_a_time_and_only_it_or_an_edited_one_mounted[scope]": 2.0,
    "test_pf26_one_group_open_at_a_time_and_only_it_or_an_edited_one_mounted[store]": 1.0,
    "test_pf26_one_group_open_at_a_time_and_only_it_or_an_edited_one_mounted[arrival]": 1.0,
    "test_pf26_one_group_open_at_a_time_and_only_it_or_an_edited_one_mounted[address]": 1.0,
    "test_pf26_one_group_open_at_a_time_and_only_it_or_an_edited_one_mounted[present]": 2.0,
    "test_pf26_one_group_open_at_a_time_and_only_it_or_an_edited_one_mounted[focus]": 1.0,
    "test_pf26_one_group_open_at_a_time_and_only_it_or_an_edited_one_mounted[edit]": 1.0,
    "test_pf27_every_panel_is_mounted_by_one_place[catalog]": 2.0,
    "test_pf27_every_panel_is_mounted_by_one_place[hub]": 2.0,
    "test_pf27_every_panel_is_mounted_by_one_place[mounts]": 1.0,
    "test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog[tabs]": 3.0,
    "test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog[keys]": 2.0,
    "test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog[gates]": 3.0,
    "test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog[overview]": 2.0,
    "test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog[ground]": 1.0,
    "test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog[wiring]": 1.0,
    "test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog[link]": 1.0,
    "test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog[save]": 1.0,
    "test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog[focus]": 1.0,
    "test_pf29_every_workshop_dialog_holds_stop_all_and_never_opens_on_it[slot]": 2.0,
    "test_pf29_every_workshop_dialog_holds_stop_all_and_never_opens_on_it[census]": 1.0,
    "test_pf29_every_workshop_dialog_holds_stop_all_and_never_opens_on_it[focus]": 2.0,
    "test_pf30_nothing_unreachable_stays_in_the_source[graph]": 1.0,
    "test_pf30_nothing_unreachable_stays_in_the_source[gone]": 1.0,
    "test_pf30_nothing_unreachable_stays_in_the_source[stylesheet]": 1.0,
    "test_pf31_the_security_page_explains_its_grade[api]": 1.0,
    "test_pf31_the_security_page_explains_its_grade[page]": 1.0,
    "test_pf31_the_security_page_explains_its_grade[checks]": 2.0,
    "test_pf31_the_security_page_explains_its_grade[failure]": 2.0,
    "test_pf31_the_security_page_explains_its_grade[sessions]": 2.0,
    "test_pf32_model_assignment_is_mounted_once_and_says_when_it_fails[importers]": 1.0,
    "test_pf32_model_assignment_is_mounted_once_and_says_when_it_fails[read]": 2.0,
    "test_pf32_model_assignment_is_mounted_once_and_says_when_it_fails[empty]": 2.0,
    "test_pf32_model_assignment_is_mounted_once_and_says_when_it_fails[save]": 2.0,
    "test_pf32_model_assignment_is_mounted_once_and_says_when_it_fails[kept]": 1.0,
    "test_pf33_the_first_run_refreshes_what_a_preset_changes_instead_of_reloading[reload]": 1.0,
    "test_pf33_the_first_run_refreshes_what_a_preset_changes_instead_of_reloading[refresh]": 1.0,
    "test_pf33_the_first_run_refreshes_what_a_preset_changes_instead_of_reloading[dialog]": 2.0,
    "test_pf33_the_first_run_refreshes_what_a_preset_changes_instead_of_reloading[choice]": 2.0,
    "test_pf33_the_first_run_refreshes_what_a_preset_changes_instead_of_reloading[skip]": 2.0,
    "test_pf33_the_first_run_refreshes_what_a_preset_changes_instead_of_reloading[focus]": 1.0,
    "test_pf34_the_benchmarks_page_is_one_row_of_tabs_over_one_engine[tabs]": 3.0,
    "test_pf34_the_benchmarks_page_is_one_row_of_tabs_over_one_engine[address]": 2.0,
    "test_pf34_the_benchmarks_page_is_one_row_of_tabs_over_one_engine[wiring]": 1.0,
    "test_pf34_the_benchmarks_page_is_one_row_of_tabs_over_one_engine[link]": 1.0,
    "test_pf34_the_benchmarks_page_is_one_row_of_tabs_over_one_engine[engine]": 1.0,
    "test_pf34_the_benchmarks_page_is_one_row_of_tabs_over_one_engine[word]": 1.0,
    "test_pf34_the_benchmarks_page_is_one_row_of_tabs_over_one_engine[row]": 1.0,
    "test_pf34_the_benchmarks_page_is_one_row_of_tabs_over_one_engine[ring]": 1.0,
}

_SRC = "frontend/src"
_LIB = f"{_SRC}/lib"
_ROUTES = f"{_SRC}/routes"
_DS = f"{_LIB}/ds"
_PANEL_HEADER = f"{_DS}/PanelHeader.svelte"
_DS_INDEX = f"{_DS}/index.ts"
_TABS = f"{_DS}/Tabs.svelte"
_MODAL = f"{_DS}/Modal.svelte"
_FIRST_FOCUS = f"{_DS}/firstFocus.ts"
_CONFIRM = f"{_DS}/ConfirmDialog.svelte"
_GALLERY = f"{_ROUTES}/dev/components/+page.svelte"
_SETTINGS = f"{_LIB}/components/settings"
_PANELS = f"{_LIB}/components/panels"
_GROUP = f"{_SETTINGS}/SettingsGroup.svelte"
_HUB = f"{_SETTINGS}/SettingsHub.svelte"
_CATALOG = f"{_LIB}/settings/catalog.ts"
_DISCLOSURE = f"{_LIB}/settings/disclosure.ts"
_DESTINATIONS = f"{_LIB}/nav/destinations.ts"
_HOST = f"{_PANELS}/ObservabilityPanel.svelte"
_STATE = f"{_LIB}/observability/state.ts"
_API = f"{_LIB}/api"
_TELEMETRY_API = f"{_API}/telemetry.ts"
_SECURITY_API = f"{_API}/security.ts"
_SECURITY_CHECKS = f"{_SETTINGS}/SecurityChecks.svelte"
_SECURITY_GRADE = f"{_SETTINGS}/SecurityGrade.svelte"
_SECURITY_BADGE = f"{_LIB}/components/sidebar/SecurityBadge.svelte"
_AUTH_STORE = f"{_LIB}/stores/auth.ts"
_MODEL_ASSIGNMENT = f"{_PANELS}/ModelAssignment.svelte"
_ROLES = f"{_LIB}/stores/modelRoles.ts"
_OVERLAY = f"{_LIB}/components/ui/OnboardingOverlay.svelte"
_REFRESH = f"{_LIB}/stores/configRefresh.ts"
_CONVERSATION_DEFAULTS = f"{_SETTINGS}/sections/ConversationDefaults.svelte"
_CONTROL_BAR = f"{_LIB}/components/chat/ChatControlBar.svelte"
_STOP_ALL = f"{_LIB}/components/layout/StopAllButton.svelte"
_BENCHMARK_PAGE = f"{_PANELS}/BenchmarkPage.svelte"
_BENCHMARK_ROUTE = f"{_ROUTES}/(app)/(workshop)/workshop/benchmarks/+page.svelte"
_BENCHMARK_TAB = f"{_LIB}/benchmark/tab.ts"
_BENCHMARK_CSS = f"{_PANELS}/benchmark/benchmark.css"
_HISTORY_SECTION = f"{_PANELS}/benchmark/BenchmarkHistorySection.svelte"
_HEALTH_DASHBOARD = f"{_LIB}/components/health/HealthDashboard.svelte"
_APP_PY = "opti_oignon/api/app.py"
_DEPS_PY = "opti_oignon/api/deps.py"
_FIXTURES = f"{_LIB}/ssr_fixture/workshop"

_SCRIPTS = (".svelte", ".ts", ".js")
_GRAPH_KINDS = (".svelte", ".ts", ".js", ".css")

_script = _nav._script
_markup = _nav._markup
_code = _nav._code
_imports = _nav._imports
_imports_name = _nav._imports_name
_calls = _nav._calls
_function_body = _nav._function_body
_tags = _repairs._tags
_attributes = _repairs._attributes
_tag_end = _repairs._tag_end


# ---------------------------------------------------------------------------
# The Node driver: it calls the pure modules and prints what they return.
# ---------------------------------------------------------------------------
_MODULES = {
    "OO_DISCLOSURE": _DISCLOSURE,
    "OO_CATALOG": _CATALOG,
    "OO_DESTINATIONS": _DESTINATIONS,
    "OO_STATE": _STATE,
    "OO_TAB": _BENCHMARK_TAB,
    "OO_FIRST_FOCUS": _FIRST_FOCUS,
}

_DRIVER = r"""
const load = async (name) => (process.env[name] ? await import(process.env[name]) : null);
const disclosure = await load('OO_DISCLOSURE');
const catalog = await load('OO_CATALOG');
const nav = await load('OO_DESTINATIONS');
const state = await load('OO_STATE');
const tab = await load('OO_TAB');
const focus = await load('OO_FIRST_FOCUS');
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT || 'null');

const run = {
    arrival: () => input.map(([g, ids, hostOf]) => disclosure.openOnArrival(g, ids, hostOf)),
    toggled: () => input.map(([open, id]) => disclosure.toggled(open, id)),
    address: () => input.map(([url, open]) => {
        const given = new URL(url);
        const before = given.href;
        const out = disclosure.addressFor(given, open);
        return { out, untouched: given.href === before };
    }),
    mounted: () => input.map(([id, open, edited]) => disclosure.mounted(id, open, new Set(edited))),
    edited: () => input.map(([edited, id]) => {
        const given = new Set(edited);
        const out = disclosure.withEdited(given, id);
        return { out: [...out].sort(), same: out === given, untouched: given.size === edited.length };
    }),
    draft: () => ({
        events: disclosure.DRAFT_EVENTS,
        verdicts: input.map(([type, key]) => disclosure.startsDraft(type, key ?? undefined)),
    }),
    context: () => disclosure.GROUPS_CONTEXT,
    catalog: () => ({
        sections: catalog.SETTINGS_SECTIONS,
        inline: catalog.INLINE_GROUPS,
        destinations: nav ? nav.DESTINATIONS : null,
        embedded: typeof catalog.embeddedGroups === 'function'
            ? Object.fromEntries(input.map((host) => [host, catalog.embeddedGroups(host).map((g) => g.id)]))
            : null,
    }),
    state: () => ({
        telemetry: input.telemetry.map((value) => state.telemetryState(value)),
        profiler: input.profiler.map((value) => state.profilerState(value)),
        history: input.history.map((value) => state.historyState(value)),
    }),
    gate: () => input.map(([feature, map]) => state.tabUnavailable(feature ?? undefined, map)),
    focus: () => input.map(([candidates, marked]) => {
        const chosen = focus.firstFocus(candidates, marked, (candidate) => candidate.actions === true);
        return chosen ? chosen.id : null;
    }),
    tab: () => ({
        tabs: tab.BENCHMARK_TABS,
        read: input.read.map((url) => tab.benchmarkTab(new URL(url))),
        write: input.write.map(([url, id]) => {
            const given = new URL(url);
            const before = given.href;
            return { out: tab.tabAddress(given, id), untouched: given.href === before };
        }),
    }),
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


def _catalog(hosts=("observability", "cache")):
    return _node("catalog", ("OO_CATALOG", "OO_DESTINATIONS"), list(hosts))


def _all_groups(catalog):
    return [group for section in catalog["sections"] for group in section["groups"]]


# ---------------------------------------------------------------------------
# Rendering: wrappers planted in the renderer's copy
# ---------------------------------------------------------------------------
_PLANTED = set()


def _plant(name, text):
    path = f"{_FIXTURES}/{name}.svelte"
    if path not in _PLANTED:
        ssr().plant(path, text)
        _PLANTED.add(path)
    return path


def _render(path, props=None):
    return _dom(ssr().render(path, props or {}).html)


def _text(element):
    return " ".join(element.text().split())


def _visible(element):
    return " ".join(element.visible_text().split())


def _headings(root):
    return [element for element in root.iter() if re.fullmatch(r"h[1-6]", element.tag or "")]


def _by_id(root, ident):
    found = [element for element in root.iter() if element.get("id") == ident]
    return found[0] if found else None


def _style_rules(path):
    """``[(selector, {property: value}, context)]`` of a component's style
    block or a stylesheet, every rule, at-rules' included."""
    text = read(path)
    css = text if path.endswith(".css") else "\n".join(
        re.findall(r"<style\b[^>]*>(.*?)</style>", text, re.S)
    )
    return [(selector, dict(_declarations(body)), context) for selector, body, context in _rules(css)]


def _rule(rules, predicate):
    """The declarations of every rule whose selector the predicate accepts, merged."""
    merged = {}
    for selector, declared, _ in rules:
        if any(predicate(" ".join(single.split())) for single in selector.split(",")):
            merged.update(declared)
    return merged


# ---------------------------------------------------------------------------
# The import graph: static, dynamic, re-exported, type-only and side-effect
# imports, and a stylesheet's @import, each resolved to a repository path.
# ---------------------------------------------------------------------------
_SPECIFIER = re.compile(
    r"""(?:\bimport\s+(?:type\s+)?(?:[\w$*{}\s,]+?\s+from\s+)?|\bimport\s*\(\s*|\bexport\s+(?:type\s+)?[\w$*{}\s,]+?\s+from\s+)(['"])([^'"]+)\1"""
)
_CSS_IMPORT = re.compile(r"""@import\s+(?:url\(\s*)?(['"])([^'"]+)\1""")
_COMMENTS = re.compile(r"/\*.*?\*/|(?<![:\w'\"`])//[^\n]*|<!--.*?-->", re.S)


def _specifiers(path, text):
    """Every module or stylesheet a file names, as written."""
    if path.endswith(".css"):
        return [match.group(2) for match in _CSS_IMPORT.finditer(re.sub(r"/\*.*?\*/", " ", text, flags=re.S))]
    body = _COMMENTS.sub(" ", text)
    found = [match.group(2) for match in _SPECIFIER.finditer(body)]
    if path.endswith(".svelte"):
        for style in re.findall(r"<style\b[^>]*>(.*?)</style>", text, re.S):
            found += [match.group(2) for match in _CSS_IMPORT.finditer(style)]
    return found


def _resolve_in(importer, spec, known):
    """The path a specifier names among ``known``, or None for a package."""
    if spec.startswith("$lib/"):
        base = posixpath.join(_LIB, spec[len("$lib/"):])
    elif spec.startswith("."):
        base = posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))
    else:
        return None
    for candidate in (base, f"{base}.ts", f"{base}.js", f"{base}.svelte", f"{base}/index.ts"):
        if candidate in known:
            return candidate
    return base


def _graph(sources):
    """``{path: set of paths it imports}`` over ``{path: text}``."""
    known = set(sources)
    return {
        path: {
            resolved for spec in _specifiers(path, text)
            if (resolved := _resolve_in(path, spec, known)) is not None
        }
        for path, text in sources.items()
    }


def _reached(graph, roots):
    seen, todo = set(), list(roots)
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        todo.extend(graph.get(current, ()))
    return seen


_TREE = {}


def _source_graph():
    """The graph of every listed source file under ``frontend/src``, built once."""
    if "graph" not in _TREE:
        sources = {path: read(path) for path in files(_GRAPH_KINDS + (".html",))}
        _TREE["sources"] = sources
        _TREE["graph"] = _graph(sources)
    return _TREE["graph"]


def _importers(target):
    """The files that import ``target``."""
    return sorted(path for path, deps in _source_graph().items() if target in deps)


def _roots(paths):
    return [path for path in paths if path.startswith((f"{_ROUTES}/", f"{_SRC}/params/"))]


# ---------------------------------------------------------------------------
# Headings in markup: what a component writes, dialogs apart
# ---------------------------------------------------------------------------
_DIALOG_OPEN = re.compile(r"<(Modal|ConfirmDialog|dialog)(?=[\s/>])")


def _without_dialogs(markup):
    """The markup with every dialog element and its content removed: a
    dialog is its own context, under its own title."""
    out = markup
    while True:
        match = _DIALOG_OPEN.search(out)
        if not match:
            return out
        stop = _tag_end(out, match.start())
        if out[stop - 2:stop] == "/>":
            out = out[:match.start()] + out[stop:]
            continue
        name = match.group(1)
        pattern = re.compile(rf"<(/?){name}(?=[\s/>])")
        depth, cursor, end = 1, stop, len(out)
        while depth:
            inner = pattern.search(out, cursor)
            if not inner:
                break
            inner_stop = _tag_end(out, inner.start())
            if inner.group(1):
                depth -= 1
            elif out[inner_stop - 2:inner_stop] != "/>":
                depth += 1
            cursor = end = inner_stop
        out = out[:match.start()] + out[end:]


def _heading_levels(markup):
    """The level of every heading a component's markup writes, dialogs apart:
    ``<hN``, ``<svelte:element this="hN">``, and ``PanelHeader`` (its literal
    level, 3 when none is given, 2 when it is not a literal)."""
    markup = _without_dialogs(markup)
    levels = [int(match.group(1)) for match in re.finditer(r"<h([1-6])(?=[\s/>])", markup)]
    for _, body, _ in _tags(markup, r"svelte:element"):
        match = re.search(r"""\bthis\s*=\s*(?:["']h([1-6])["']|\{\s*["'`]h([1-6])["'`]\s*\})""", body)
        if match:
            levels.append(int(match.group(1) or match.group(2)))
    for _, body, _ in _tags(markup, r"PanelHeader"):
        match = re.search(r"""\blevel\s*=\s*(?:\{\s*([1-6])\s*\}|["']([1-6])["'])""", body)
        if match:
            levels.append(int(match.group(1) or match.group(2)))
        elif re.search(r"\blevel\b", body):
            levels.append(2)
        else:
            levels.append(3)
    return levels


_DEFAULT_IMPORT = re.compile(r"""\bimport\s+([A-Z]\w*)\s+from\s*['"]([^'"]+\.svelte)['"]""")


def _outside_dialogs(path, markup=None):
    """The components a file imports, but those it mounts only inside a
    dialog, which is its own context: a component named in no tag is kept
    (it may be drawn through ``svelte:component``)."""
    markup = _markup(path) if markup is None else markup
    outside = _without_dialogs(markup)
    dropped = set()
    for name, spec in _DEFAULT_IMPORT.findall(_script(path)):
        tag = re.compile(rf"<{name}(?=[\s/>])")
        if tag.search(markup) and not tag.search(outside):
            dropped.add(_nav._resolve(path, spec))
    return [target for target in _imports(path) if target not in dropped]


def _mounted_files(path):
    """A component and every component it mounts outside a dialog,
    transitively: the design system's primitives and the group's own frame
    left out."""
    seen, todo = set(), [path]
    while todo:
        current = todo.pop()
        if current in seen or current.startswith(f"{_DS}/") or current == _GROUP:
            continue
        if not current.endswith(".svelte") or not (REPO / current).is_file():
            continue
        seen.add(current)
        todo.extend(_outside_dialogs(current))
    return seen


_HUB_LOADER = re.compile(r"""(\w+)\s*:\s*\(\)\s*=>\s*import\(\s*['"]([^'"]+)['"]\s*\)""")


def _hub_loaders():
    """``{panel name: path}`` of the hub's lazy loaders."""
    return {
        name: _nav._resolve(_HUB, spec)
        for name, spec in _HUB_LOADER.findall(_script(_HUB))
    }


_INTROS = {
    "appearance": f"{_SETTINGS}/sections/AppearanceSection.svelte",
    "conversation": _CONVERSATION_DEFAULTS,
    "account": f"{_SETTINGS}/sections/AccountAuthMode.svelte",
}


def _levels_of_groups(catalog):
    """``[(component, group level)]``: the panel of every placed group and the
    introduction of every group rendered inline, with the level of the group
    it draws under (2 on a Workshop page, 3 in Preferences). An introduction
    that renders groups on both is held to Preferences."""
    loaders = _hub_loaders()
    out = []
    for group in _all_groups(catalog):
        if group.get("space") in ("use", "workshop") and group.get("panel") in loaders:
            out.append((loaders[group["panel"]], 2 if group["space"] == "workshop" else 3))
    intro_of = {section["id"]: section.get("intro") for section in catalog["sections"]}
    levels = {}
    for group in catalog["inline"]:
        intro = intro_of.get(group["sectionId"])
        if intro and group.get("space") in ("use", "workshop"):
            level = 2 if group["space"] == "workshop" else 3
            levels[intro] = max(levels.get(intro, 0), level)
    for intro, level in levels.items():
        out.append((_INTROS[intro], level))
    return out


# The titles the panels used to draw above their own content, each the same
# words as the title of the group that holds them: {panel: title}.
_REPEATED_TITLES = {
    f"{_PANELS}/AnalyticsDashboard.svelte": "Analytics & Feedback",
    f"{_SETTINGS}/AppPasswordsPanel.svelte": "App Passwords",
    f"{_SETTINGS}/BackupRestorePanel.svelte": "Backup & Restore",
    f"{_PANELS}/CacheStatsPanel.svelte": "Semantic Cache",
    f"{_PANELS}/CascadingPanel.svelte": "Cascading Inference",
    f"{_PANELS}/CompressionSettings.svelte": "Conversation Compressor",
    f"{_PANELS}/GovernorPanel.svelte": "Resource governor",
    f"{_PANELS}/LearnedRouterPanel.svelte": "Learned Router",
    f"{_PANELS}/MemoriesPanel.svelte": "Memories",
    f"{_PANELS}/ModelAssignment.svelte": "Model Assignment",
    f"{_SETTINGS}/ModelHealthWidget.svelte": "Model Health",
    f"{_PANELS}/PerformanceDashboard.svelte": "Performance Dashboard",
    f"{_SETTINGS}/PluginAllowlistPanel.svelte": "Plugin Allowlist",
    f"{_SETTINGS}/PluginMarketplace.svelte": "Marketplace",
    f"{_SETTINGS}/PluginsPanel.svelte": "Plugins",
    f"{_PANELS}/ProfilerDashboard.svelte": "Inference Profiler",
    f"{_PANELS}/PromptConfigPanel.svelte": "Prompt Intelligence",
    f"{_PANELS}/ProxySettingsPanel.svelte": "Web Search & Privacy",
    f"{_SETTINGS}/RecoveryCodesPanel.svelte": "Recovery Codes",
    f"{_SETTINGS}/RemoteAccessPanel.svelte": "Remote Access",
    f"{_SETTINGS}/SearchKillSwitchPanel.svelte": "Web Search Kill Switch",
    f"{_SETTINGS}/SecurityModePanel.svelte": "Security Mode",
    f"{_PANELS}/SkillsPanel.svelte": "Skills",
    f"{_PANELS}/SyncPanel.svelte": "Sync (Veilid)",
    f"{_SETTINGS}/TOTPSetup.svelte": "Authenticator App (TOTP)",
    f"{_PANELS}/TelemetryDashboard.svelte": "Inference Telemetry",
    f"{_PANELS}/TelemetryHistoryPanel.svelte": "Telemetry History",
    f"{_SETTINGS}/VisionModelSelector.svelte": "Vision Model",
    f"{_SETTINGS}/WebAuthnSetup.svelte": "Security Keys (WebAuthn/FIDO2)",
}


def _heading_texts(markup):
    """The text of every ``<hN>`` element in a markup, tags and expressions
    dropped, references decoded, spaces collapsed."""
    out = []
    for match in re.finditer(r"<h([1-6])(?=[\s>])[^>]*>(.*?)</h\1\s*>", markup, re.S):
        inner = re.sub(r"<[^>]*>", " ", match.group(2))
        out.append(" ".join(html.unescape(inner).split()))
    return out


def _titles(markup):
    """Every title a markup writes: its ``<hN>`` elements, its
    ``<svelte:element this="hN">`` elements and its ``PanelHeader`` titles
    given as literals."""
    out = _heading_texts(markup)
    for match in re.finditer(
        r"""<svelte:element\s+this\s*=\s*(?:["']h[1-6]["']|\{\s*["'`]h[1-6]["'`]\s*\})[^>]*>(.*?)</svelte:element\s*>""",
        markup, re.S,
    ):
        out.append(" ".join(html.unescape(re.sub(r"<[^>]*>", " ", match.group(1))).split()))
    for _, body, _ in _tags(markup, r"PanelHeader"):
        match = re.search(r"""\btitle\s*=\s*(?:"([^"]*)"|'([^']*)'|\{\s*["'`]([^"'`]*)["'`]\s*\})""", body)
        if match:
            out.append(" ".join(html.unescape(match.group(1) or match.group(2) or match.group(3) or "").split()))
    return out


def _repeats(markup, title):
    wanted = title.lower()
    return [text for text in _titles(markup) if text.lower().startswith(wanted)]


# ===========================================================================
# PF23 -- PanelHeader titles once, and folds by its whole row
# ===========================================================================
@pytest.mark.parametrize("half", ("level", "disclosure", "plain", "blank", "row"))
def test_pf23_a_panel_header_titles_once_and_folds_by_its_whole_row(half):
    if half == "row":
        rules = _style_rules(_PANEL_HEADER)
        row = _rule(rules, lambda s: s == ".oo-panel-header")
        assert row.get("position") == "relative", f"the header is the positioned row: {row}"
        assert row.get("min-height") == "44px", f"the row is a 44 px target: {row}"
        hit = _rule(rules, lambda s: s.endswith("::after") and "toggle" in s)
        assert hit.get("position") == "absolute" and hit.get("inset") == "0" and "content" in hit, (
            f"the button's hit area is stretched over the whole row: {hit}"
        )
        assert row.get("border-radius") and hit.get("border-radius") == row.get("border-radius"), (
            f"and rounded as the row is, never past a rounded card's corner: {hit} against {row}"
        )
        actions = _rule(rules, lambda s: s == ".oo-panel-header-actions")
        assert actions.get("position") == "relative" and actions.get("z-index") == "1", (
            f"the actions sit above the stretched hit area: {actions}"
        )
        markup = _markup(_PANEL_HEADER)
        button = re.search(r"<button\b.*?</button>", markup, re.S)
        assert button and "<slot" not in button.group(0), "the actions stand outside the button"
        assert re.search(r"""class\s*=\s*["']oo-panel-header-actions["'][^>]*>\s*<slot\s+name\s*=\s*["']actions["']""", markup), (
            "the actions slot sits in its own layer"
        )
        return

    if half == "blank":
        for props in ({}, {"title": ""}, {"title": "   "}):
            with pytest.raises(AssertionError, match="PanelHeader needs a title"):
                ssr().render(_PANEL_HEADER, props)
        assert re.search(r"export\s+let\s+title\s*:\s*string\s*;", _script(_PANEL_HEADER)), (
            "the title is a required prop, with no default"
        )
        return

    if half == "level":
        for given, tag in ((2, "h2"), (3, "h3"), (4, "h4"), (None, "h3")):
            props = {"title": "Cache"} if given is None else {"title": "Cache", "level": given}
            root = _render(_PANEL_HEADER, props)
            headings = _headings(root)
            assert [h.tag for h in headings] == [tag], (
                f"level {given}: the title is one heading of that level: {[h.tag for h in headings]}"
            )
            assert _text(headings[0]) == "Cache", f"the heading reads the title alone: {_text(headings[0])!r}"
        return

    if half == "plain":
        root = _render(_PANEL_HEADER, {"title": "Cache", "description": "What the cache holds."})
        heading = _headings(root)[0]
        assert not list(heading.iter("button")), "a plain heading holds no button"
        assert _text(heading) == "Cache", f"the heading reads the title alone: {_text(heading)!r}"
        assert "What the cache holds." in _text(root), "the description is shown"
        assert "What the cache holds." not in _text(heading), "outside the heading"
        return

    for expanded in (False, True):
        root = _render(_PANEL_HEADER, {
            "title": "Cache", "description": "What the cache holds.", "level": 2,
            "expanded": expanded, "controls": "oo-probe-body",
        })
        headings = _headings(root)
        assert [h.tag for h in headings] == ["h2"], f"one heading of the given level: {headings}"
        buttons = list(headings[0].iter("button"))
        assert len(buttons) == 1, "the title is a button inside the heading"
        button = buttons[0]
        assert button.get("aria-expanded") == ("true" if expanded else "false"), button.attrs
        assert button.get("aria-controls") == "oo-probe-body", button.attrs
        assert button.get("type") == "button", "it never submits a form"
        assert _name(button) == "Cache", f"the button is named by the title alone: {_name(button)!r}"
        described = button.get("aria-describedby")
        description = _by_id(root, described) if described else None
        assert description is not None and _text(description) == "What the cache holds.", (
            f"the description is the button's description: {described!r}"
        )
        assert not _inside(description, headings[0]), "and it stands outside the heading"


# ===========================================================================
# PF24 -- every group is titled once, at its own level
# ===========================================================================
def _hub_page(space, section=None):
    """The settings hub rendered on the server for one page."""
    name = f"Hub{space.title()}{(section or '').title()}"
    path = _plant(name, (
        "<script>\n"
        "\timport SettingsHub from '$lib/components/settings/SettingsHub.svelte';\n"
        "</script>\n"
        f"<SettingsHub space=\"{space}\"" + (f" section=\"{section}\"" if section else "") + " />\n"
    ))
    return _render(path)


@pytest.mark.parametrize("half", ("own", "page", "level", "same", "levels", "listed"))
def test_pf24_every_group_is_titled_once_at_its_own_level(half):
    if half == "own":
        markup = _markup(_GROUP)
        assert re.search(r"<PanelHeader\b", markup) and _PANEL_HEADER in _imports(_GROUP), (
            "the group draws its title through PanelHeader"
        )
        assert not re.search(r"<h[1-6](?=[\s/>])|<svelte:element\b", markup), (
            "and through nothing else: it writes no heading of its own"
        )
        assert not re.search(r"\bceremony\b", _code(_GROUP)), "the unused ceremony variant and its badge are gone"
        return

    if half == "page":
        catalog = _catalog()
        pages = sorted({
            group["section"] for group in _all_groups(catalog) + catalog["inline"]
            if group.get("space") == "workshop" and group.get("section")
        })
        assert len(pages) >= 7, f"the census walks every Workshop settings page: {pages}"
        for section in pages:
            root = _hub_page("workshop", section)
            level_one = [h for h in _headings(root) if h.tag == "h1"]
            assert len(level_one) == 1, f"{section}: the page has one h1: {level_one}"
            label = _text(level_one[0])
            second = [h for h in _headings(root) if h.tag == "h2"]
            assert second, f"{section}: the census reads the page's headings"
            repeated = [_text(h) for h in second if _text(h).lower() == label.lower()]
            assert not repeated, f"{section}: a section heading repeats the page's h1 {label!r}: {repeated}"
            head = _PAGE_HEADS.get(section)
            heads = [e for e in root.iter() if set(e.classes()) & set(_PAGE_HEADS.values())]
            assert all(head in e.classes() for e in heads), (
                f"{section}: a page's head shows on its own page alone: {[e.classes() for e in heads]}"
            )
            stray = [
                _text(h) for h in _headings(root) if h.tag != "h1"
                and not any((node.get("id") or "").startswith("oo-set-") for node in _ancestors(h))
                and not any(head and head in node.classes() for node in _ancestors(h))
            ]
            assert not stray, (
                f"{section}: every heading under the h1 is a group's title, or its page head's: {stray}"
            )
        return

    if half == "level":
        workshop = _hub_page("workshop", "models")
        titled = _group_titles(workshop)
        assert titled and {level for level, _ in titled} == {2}, (
            f"a Workshop page draws its groups' titles at level 2: {titled}"
        )
        preferences = _hub_page("use")
        titled = _group_titles(preferences)
        assert titled and {level for level, _ in titled} == {3}, (
            f"Preferences draws its groups' titles at level 3, under its sections: {titled}"
        )
        alone = _render(_GROUP, {"id": "probe", "title": "Probe"})
        assert [h.tag for h in _headings(alone)] == ["h3"], "a group with no hub around it draws at level 3"
        return

    if half == "same":
        catalog = _catalog()
        labels = {
            d["href"].rsplit("/", 1)[-1]: d["label"]
            for d in catalog["destinations"] if d["href"].startswith("/workshop/")
        }
        assert len(labels) >= 7, f"the census reads the Workshop's pages: {labels}"
        assert _bare("Observability (Observe)") == "observability", "parentheses aside"
        same = [
            (group["id"], group["title"], labels[group["section"]])
            for group in _all_groups(catalog) + catalog["inline"]
            if group.get("space") == "workshop" and group.get("section") in labels
            and _bare(group["title"]) == _bare(labels[group["section"]])
        ]
        assert not same, f"a group named after the page that holds it: {same}"
        return

    if half == "levels":
        sample = (
            "<div><h2>Title</h2><PanelHeader level={2} title=\"x\" />"
            "<PanelHeader title=\"y\" /><Modal title=\"m\"><h2>Inside</h2></Modal>"
            "<ConfirmDialog title=\"c\" /></div>"
        )
        assert sorted(_heading_levels(sample)) == [2, 2, 3], "the census reads headings, dialogs apart"
        assert len([level for level in _heading_levels(sample) if level <= 2]) == 2
        dialog_only = "<script>import Inner from './Inner.svelte';</script><Modal title=\"m\"><Inner /></Modal>"
        assert _without_dialogs(_markup("x.svelte", dialog_only)).strip() == "", "a dialog's content is its own"
        catalog = _catalog()
        held = _levels_of_groups(catalog)
        assert len(held) >= 40, f"the census reads every placed panel and introduction: {len(held)}"
        findings = {}
        for panel, level in held:
            for path in sorted(_mounted_files(panel)):
                high = [found for found in _heading_levels(_markup(path)) if found <= level]
                if high:
                    findings.setdefault(f"{path} (under a level {level} group, from {panel})", high)
        assert not findings, (
            "headings at or above their group's level:\n  "
            + "\n  ".join(f"{where}: h{levels}" for where, levels in sorted(findings.items()))
        )
        return

    assert _repeats("<h2 class=\"t\">\n\tBackup &amp; Restore\n</h2>", "Backup & Restore"), (
        "the census reads a title across lines, references decoded"
    )
    assert _repeats("<PanelHeader title=\"Plugins\" level={3} />", "Plugins") and _repeats(
        "<svelte:element this=\"h3\" class=\"t\">Plugins</svelte:element>", "Plugins"
    ) and _repeats("<PanelHeader level={4} title={'Plugins'} />", "Plugins"), (
        "the census reads a title drawn by PanelHeader or by an element of a heading's tag"
    )
    assert len(_REPEATED_TITLES) == 29
    still = {
        path: found for path, title in _REPEATED_TITLES.items()
        if (found := _repeats(_markup(path), title))
    }
    assert not still, f"a panel still titles itself as its group does: {still}"


# The heads a Workshop page shows above its groups, by page, each by the
# class of its frame: the network's reachability and the security grade.
# Their headings title the page's head, not a group.
_PAGE_HEADS = {"network": "oo-reach", "security": "oo-sec-grade"}


def _ancestors(element):
    node = element.parent
    while node is not None:
        yield node
        node = node.parent


def _group_titles(root):
    """``[(level, title)]`` of the group titles a page renders: the heading
    in each ``oo-set-`` group's frame."""
    out = []
    for element in root.iter():
        ident = element.get("id") or ""
        if ident.startswith("oo-set-") and not ident.endswith(("-body", "-title")):
            heading = next(iter(_headings(element)), None)
            if heading is not None:
                out.append((int(heading.tag[1]), _text(heading)))
    return out


def _bare(title):
    return " ".join(re.sub(r"\([^)]*\)", " ", title).split()).lower()


# ===========================================================================
# PF25 -- the folding rules are one pure module
# ===========================================================================
_PAGE = ["cache", "observability", "analytics"]
_HOST_OF = {"telemetry": "observability", "profiler": "observability"}


@pytest.mark.parametrize(
    "half", ("named", "embedded", "default", "toggle", "address", "kept", "edited", "draft", "wiring")
)
def test_pf25_the_folding_rules_are_one_pure_module(half):
    if half == "named":
        got = _node("arrival", ("OO_DISCLOSURE",), [
            ["analytics", _PAGE, _HOST_OF], ["cache", _PAGE, _HOST_OF], ["cache", ["cache"], {}],
        ])
        assert got == ["analytics", "cache", "cache"], f"the group g names opens: {got}"
        return
    if half == "embedded":
        got = _node("arrival", ("OO_DISCLOSURE",), [
            ["telemetry", _PAGE, _HOST_OF], ["profiler", _PAGE, _HOST_OF], ["profiler", ["cache", "x"], _HOST_OF],
        ])
        assert got == ["observability", "observability", None], (
            f"an embedded group opens its host, when its host is on the page: {got}"
        )
        return
    if half == "default":
        got = _node("arrival", ("OO_DISCLOSURE",), [
            [None, ["only"], {}], [None, _PAGE, _HOST_OF], ["", _PAGE, _HOST_OF],
            ["nowhere", _PAGE, _HOST_OF], ["nowhere", ["only"], {}], [None, [], {}],
        ])
        assert got == ["only", None, None, None, "only", None], (
            f"with no group named, the page's only group opens, and none of several: {got}"
        )
        return
    if half == "toggle":
        got = _node("toggled", ("OO_DISCLOSURE",), [["a", "a"], ["a", "b"], [None, "a"]])
        assert got == [None, "b", "a"], f"a toggle closes the open group or opens another: {got}"
        return
    if half == "address":
        got = _node("address", ("OO_DISCLOSURE",), [
            ["http://localhost/workshop/models?q=cache&g=a&z=1", "b"],
            ["http://localhost/workshop/models?q=cache&g=a", None],
            ["http://localhost/workshop/models", "routing"],
            ["http://localhost/workshop/models?g=a", None],
        ])
        outs = [entry["out"] for entry in got]
        assert outs == [
            "/workshop/models?q=cache&g=b&z=1",
            "/workshop/models?q=cache",
            "/workshop/models?g=routing",
            "/workshop/models",
        ], f"the address sets or drops g and keeps every other key: {outs}"
        assert all(entry["untouched"] for entry in got), "the URL it is given is never changed"
        return
    if half == "kept":
        got = _node("mounted", ("OO_DISCLOSURE",), [
            ["a", "a", []], ["a", "b", ["a"]], ["a", "b", []], ["a", None, []], ["a", None, ["b"]],
        ])
        assert got == [True, True, False, False, False], (
            f"a group mounts while it is open, or once edited, and never otherwise: {got}"
        )
        return
    if half == "edited":
        got = _node("edited", ("OO_DISCLOSURE",), [[[], "a"], [["a"], "b"], [["a", "b"], "a"]])
        assert [entry["out"] for entry in got] == [["a"], ["a", "b"], ["a", "b"]], (
            f"a group joins the edited ones, once: {got}"
        )
        assert got[2]["same"] and not got[0]["same"], "a group already edited leaves the set as it was"
        assert all(entry["untouched"] for entry in got), "the set it is given is never changed"
        return
    if half == "draft":
        cases = [
            ["input", None], ["change", None], ["click", None], ["drop", None],
            ["keydown", "a"], ["keydown", " "], ["keydown", "Enter"], ["keydown", "ArrowDown"],
            ["keydown", "Tab"], ["keydown", "Escape"], ["keydown", "Shift"], ["keydown", "Control"],
            ["keydown", "Alt"], ["keydown", "Meta"], ["focusin", None], ["scroll", None], ["pointermove", None],
        ]
        got = _node("draft", ("OO_DISCLOSURE",), cases)
        assert sorted(got["events"]) == sorted(["input", "change", "click", "keydown", "drop"]), (
            f"a group listens for a field typed in or changed, a click, a key and a drop: {got['events']}"
        )
        assert got["verdicts"] == [True] * 8 + [False] * 9, (
            "each of those may start a draft; a key that only moves through the page or holds a "
            f"modifier does not, nor does anything else: {list(zip(cases, got['verdicts']))}"
        )
        return

    hub, group = _script(_HUB), _script(_GROUP)
    for name in ("openOnArrival", "toggled", "addressFor"):
        assert _imports_name(_HUB, name, _DISCLOSURE) and _calls(hub, name), f"the hub decides through {name}"
    assert _imports_name(_GROUP, "mounted", _DISCLOSURE) and _calls(group, "mounted"), (
        "the group decides what it mounts through mounted"
    )
    sample = "url.searchParams.set('g', id); u.searchParams.delete(\"g\");"
    writes = re.compile(r"""\bsearchParams\.(?:set|delete)\(\s*['"]g['"]""")
    assert len(writes.findall(sample)) == 2, "the census reads a hand-made address"
    assert not writes.search(hub), "the hub writes no g of its own"
    assert not re.search(r"\{#if[^}]*\$open\s*===", _markup(_GROUP)), (
        "the group holds no mount rule of its own"
    )


# ===========================================================================
# PF26 -- one group open at a time, and only it (or an edited one) mounted
# ===========================================================================
def _probe_group(name, open_id, edited, with_context=True):
    """A group with a probe for its panel, under a context the wrapper sets."""
    context = (
        "\timport { setContext } from 'svelte';\n"
        "\timport { writable } from 'svelte/store';\n"
        f"\tsetContext('oo-settings-groups', {{ level: 2, collapsible: true,"
        f" open: writable({json.dumps(open_id)}), edited: writable(new Set({json.dumps(edited)})),"
        " toggle: () => {}, markEdited: () => {} });\n"
    ) if with_context else ""
    path = _plant(name, (
        "<script>\n"
        + context
        + "\timport SettingsGroup from '$lib/components/settings/SettingsGroup.svelte';\n"
        "</script>\n"
        "<SettingsGroup id=\"probe\" title=\"Probe group\" description=\"A probe.\">"
        "<span class=\"oo-probe-panel\">probe</span></SettingsGroup>\n"
    ))
    return _render(path)


def _probes(root):
    return [e for e in root.iter("span") if "oo-probe-panel" in e.classes()]


@pytest.mark.parametrize("half", ("lazy", "scope", "store", "arrival", "address", "present", "focus", "edit"))
def test_pf26_one_group_open_at_a_time_and_only_it_or_an_edited_one_mounted(half):
    if half == "lazy":
        closed = _probe_group("GroupClosed", None, [])
        assert not _probes(closed), "a closed group that was never edited renders no panel"
        other = _probe_group("GroupOtherOpen", "else", [])
        assert not _probes(other), "nor while another group is open"
        opened = _probe_group("GroupOpen", "probe", [])
        assert len(_probes(opened)) == 1, "the open group renders its panel"
        edited = _probe_group("GroupEdited", "else", ["probe"])
        assert len(_probes(edited)) == 1, "an edited group keeps its panel mounted, closed"
        return

    if half == "scope":
        assert re.search(r"collapsible\s*:\s*space\s*===\s*['\"]workshop['\"]", _script(_HUB)), (
            "the hub folds the groups of a Workshop page, and nowhere else"
        )
        workshop = _hub_page("workshop", "models")
        disclosures = [
            b for b in workshop.iter("button")
            if b.get("aria-expanded") is not None and any(
                (n.get("id") or "").startswith("oo-set-") for n in _ancestors(b)
            )
        ]
        assert len(disclosures) >= 10, f"a Workshop page's groups fold: {len(disclosures)} disclosures"
        preferences = _hub_page("use")
        folded = [
            b for b in preferences.iter("button")
            if b.get("aria-expanded") is not None and any(
                (n.get("id") or "").startswith("oo-set-") for n in _ancestors(b)
            )
        ]
        assert not folded, f"Preferences' groups do not fold: {len(folded)}"
        bodies = [e for e in preferences.iter() if (e.get("id") or "").endswith("-body")
                  and (e.get("id") or "").startswith("oo-set-")]
        assert bodies and not [b for b in bodies if b.has("hidden")], "they render open"
        return

    if half == "store":
        hub = _script(_HUB)
        assert re.search(r"\bconst\s+open\s*=\s*writable\b", hub) and re.search(
            r"\bconst\s+edited\s*=\s*writable\b", hub
        ), "the hub creates the two stores"
        context = _nav._call_argument(hub, "setContext")
        assert "GROUPS_CONTEXT" in context and all(
            re.search(rf"\b{name}\b", context) for name in ("level", "collapsible", "open", "edited", "toggle", "markEdited")
        ), f"and puts them in the context: {context!r}"
        assert re.search(r"\$open\b", _code(_GROUP)), "the group reads the open store"
        return

    if half == "arrival":
        hub = _script(_HUB)
        assert len(re.findall(r"\bafterNavigate\s*\(", hub)) == 1, "one afterNavigate callback"
        callback = _nav._call_argument(hub, "afterNavigate")
        assert "words =" in callback and "searchParams.get('q')" in callback, "it still reads the words"
        own = re.search(
            r"\bconst\s+(\w+)\s*=\s*wrote\s*!==\s*null\s*&&[^;]*`\$\{to\.url\.pathname\}\$\{to\.url\.search\}`\s*===\s*wrote\s*;",
            callback,
        )
        cleared = re.search(r"\bwrote\s*=\s*null\s*;", callback)
        skip = re.search(rf"\bif\s*\(\s*{own.group(1)}\s*\)\s*return\b", callback) if own else None
        arrive = re.search(r"(?<![\w$.])(?:arrive|openOnArrival)\s*\(", callback)
        assert own and cleared and skip and arrive and own.start() < cleared.start() < skip.start() < arrive.start(), (
            "only the address the hub asked for itself is no arrival, and the mark is spent on the "
            f"next navigation, whichever it is: {callback!r}"
        )
        for name in ("toggle", "syncQuery"):
            body = _function_body(hub, name)
            mark = re.search(r"\bwrote\s*=\s*(\w+)\s*;", body)
            navigate = re.search(rf"\bgoto\(\s*{mark.group(1)}\b", body) if mark else None
            assert mark and navigate and mark.start() < navigate.start(), (
                f"{name} records the very address it writes, before it writes it: {body!r}"
            )
        starts = re.findall(r"block\s*:\s*['\"]start['\"]", hub)
        reveal = _function_body(hub, "revealGroup")
        assert len(starts) == 1 and re.search(r"block\s*:\s*['\"]start['\"]", reveal), (
            "only an arrival scrolls a group to the start"
        )
        return

    if half == "address":
        body = _function_body(_script(_HUB), "toggle")
        call = _nav._call_argument(body, "goto")
        first = re.match(r"\s*(\w+)\s*(\(|,)", call)
        computed = bool(first) and (
            first.group(1) == "addressFor"
            or re.search(rf"\b{first.group(1)}\s*=\s*addressFor\(", body) is not None
        )
        assert computed, f"a toggle writes the address the module computes: {call!r}"
        for option in ("replaceState", "keepFocus", "noScroll"):
            assert re.search(rf"\b{option}\s*:\s*true\b", call), f"with {option}: {call!r}"
        return

    if half == "present":
        for name, open_id in (("GroupShut", None), ("GroupShown", "probe")):
            root = _probe_group(name + "Present", open_id, [])
            buttons = [b for b in root.iter("button") if b.get("aria-controls")]
            assert len(buttons) == 1, f"the group's title is one disclosure button: {len(buttons)}"
            body = _by_id(root, buttons[0].get("aria-controls"))
            assert body is not None, "the region the button controls is always rendered"
            assert body.has("hidden") == (open_id is None), (
                f"hidden while closed, shown while open: {body.attrs}"
            )
            frames = [e for e in root.iter("section") if (e.get("id") or "") == "oo-set-probe"]
            assert len(frames) == 1 and frames[0].get("aria-labelledby") is None and frames[0].get("aria-label") is None, (
                f"the group's frame is no named region: a page of groups is not a page of landmarks: {frames}"
            )
        hub = _hub_page("workshop", "models")
        level_one = [h for h in _headings(hub) if h.tag == "h1"]
        sections = [e for e in hub.iter("section") if e.get("aria-labelledby") and "oo-hub-section" in e.classes()]
        assert level_one and level_one[0].get("id"), "the page's h1 has an id"
        assert sections and all(s.get("aria-labelledby") == level_one[0].get("id") for s in sections), (
            "a Workshop page's section is named by its h1"
        )
        return

    if half == "edit":
        group, markup = _script(_GROUP), _markup(_GROUP)
        for name in ("DRAFT_EVENTS", "startsDraft"):
            assert _imports_name(_GROUP, name, _DISCLOSURE), f"the group asks the module which events count: {name}"
        body = re.search(r"<div\b[^>]*\bid=\{bodyId\}[^>]*>", markup)
        action = re.search(r"\buse:(\w+)", body.group(0)) if body else None
        assert action, f"the panel's body is watched by an action: {body and body.group(0)!r}"
        watch = _function_body(group, action.group(1))
        listen = re.search(r"for\s*\(\s*const\s+(\w+)\s+of\s+DRAFT_EVENTS\s*\)\s*\w+\.addEventListener\(\s*\1\s*,\s*(\w+)\s*,\s*true\s*\)", watch)
        assert listen, f"for every event the module names, in the capture phase: {watch!r}"
        handler = re.search(rf"\bconst\s+{listen.group(2)}\s*=\s*\([^)]*\)\s*=>\s*\{{(.*?)\}};", watch, re.S)
        assert handler and _calls(handler.group(1), "startsDraft") and re.search(r"\bonEdit\(\)", handler.group(1)), (
            f"an event that may start a draft marks the group edited: {watch!r}"
        )
        assert re.search(r"removeEventListener\(\s*\w+\s*,\s*\w+\s*,\s*true\s*\)", watch), "and it stops listening"
        assert re.search(r"context\?\.markEdited\(\s*id\s*\)", _function_body(group, "onEdit")), (
            "through the hub's context"
        )
        assert not re.search(r"\bon:(?:input|change)\b", markup), "no other listener decides what counts"
        hub = _script(_HUB)
        assert _imports_name(_HUB, "withEdited", _DISCLOSURE) and _calls(_function_body(hub, "markEdited"), "withEdited"), (
            "the hub adds the group to the edited ones through the module"
        )
        pages = re.search(
            r"\{/if\}\s*(?:<!--.*?-->\s*)?<div\s+class=\"oo-hub-pages\"\s+hidden=\{!!query\}>\s*\{#each\s+pageSections\b",
            _markup(_HUB), re.S,
        )
        assert pages, "the page's search hides the groups while it shows its results, and drops none"
        return

    reveal = _function_body(_script(_HUB), "revealGroup")
    assert re.search(r"\.focus\(\s*\{\s*preventScroll\s*:\s*true\s*\}\s*\)", reveal), (
        f"an arrival focuses the opened group's button without a scroll of its own: {reveal!r}"
    )
    assert re.search(r"aria-expanded|aria-controls", reveal), "the button it focuses is the group's disclosure"
    callback = _nav._call_argument(_script(_HUB), "afterNavigate")
    arrive = _function_body(_script(_HUB), "arrive")
    assert _calls(arrive, "revealGroup") and _calls(callback, "arrive"), "each arrival reveals its group"


# ===========================================================================
# PF27 -- every panel is mounted by one place
# ===========================================================================
_EMBEDDED = ["telemetry", "telemetry-history", "profiler", "performance-dashboard"]


def _panel_files():
    """``{panel name: path}``: the one component under ``lib`` named so."""
    out = {}
    for path in files(".svelte", within=_LIB):
        out.setdefault(PurePosixPath(path).stem, []).append(path)
    return out


@pytest.mark.parametrize("half", ("catalog", "hub", "mounts"))
def test_pf27_every_panel_is_mounted_by_one_place(half):
    if half == "catalog":
        catalog = _catalog()
        by_id = {group["id"]: group for group in _all_groups(catalog)}
        wrong = [
            gid for gid in _EMBEDDED
            if by_id[gid].get("embeddedIn") != "observability" or by_id[gid].get("space") or by_id[gid].get("section")
        ]
        assert not wrong, f"embedded in the Observability group, with no page of their own: {wrong}"
        host = by_id["observability"]
        assert (host.get("space"), host.get("section")) == ("workshop", "observability"), host
        assert catalog["embedded"] == {"observability": _EMBEDDED, "cache": []}, (
            f"embeddedGroups returns a host's groups in catalog order: {catalog['embedded']}"
        )
        return

    if half == "hub":
        tree = _repairs._real_tree()
        loaders = _repairs._LAZY.findall(_repairs._script(tree.sources[_HUB], _HUB))
        inline = _repairs._hub_inline_groups(tree)
        catalog = _repairs._node("catalog", ("OO_CATALOG",), [])
        sections = catalog["sections"]
        groups = [group for section in sections for group in section["groups"]]
        embedded = [group for group in groups if group.get("embeddedIn")]
        host_imports = []
        for group in embedded:
            host = next(g for g in groups if g["id"] == group["embeddedIn"])
            host_path = _hub_loaders().get(host["panel"])
            if host_path and any(PurePosixPath(p).stem == group["panel"] for p in _imports(host_path)):
                host_imports.append(group["panel"])
        assert len(loaders) + len(host_imports) >= 40 and len(inline) >= 9, (
            f"the census reads the hub's panels, the hosts' and the inline groups: "
            f"{len(loaders)} + {len(host_imports)}, {len(inline)}"
        )
        assert [section["id"] for section in sections] == list(_repairs._SECTION_IDS), (
            "the catalog's sections are the ones the census recognises"
        )
        assert catalog["legacy"] == _repairs._LEGACY_TABS, "the catalog's old tab map is the one the census recognises"
        ids = [group["id"] for group in groups] + [group["id"] for group in catalog["inline"]]
        assert len(ids) == len(set(ids)), "every group id is unique"
        for group in groups + catalog["inline"]:
            assert group["title"] and group["description"], f"{group['id']} has a title and a description"
        placed = [group["panel"] for group in groups if not group.get("embeddedIn")]
        panels = [group["panel"] for group in groups]
        assert sorted(placed) == sorted(loaders) and len(panels) == len(set(panels)), (
            "every lazy panel the hub can load is one placed group's, and every placed group has a panel: "
            f"missing {sorted(set(loaders) - set(placed))}, unloadable {sorted(set(placed) - set(loaders))}"
        )
        assert sorted(host_imports) == sorted(group["panel"] for group in embedded), (
            f"every embedded group's panel is imported by its host's panel: {host_imports}"
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
        return

    catalog = _catalog()
    groups = _all_groups(catalog)
    by_id = {group["id"]: group for group in groups}
    named = _panel_files()
    problems = []
    for group in groups:
        if group.get("retired"):
            continue
        candidates = named.get(group["panel"], [])
        if len(candidates) != 1:
            problems.append(f"{group['id']}: {group['panel']} names {len(candidates)} components")
            continue
        panel = candidates[0]
        if group.get("embeddedIn"):
            host = by_id[group["embeddedIn"]]
            expected = named.get(host["panel"], [None])[0]
        else:
            expected = _HUB
        importers = _importers(panel)
        if importers != [expected]:
            problems.append(f"{group['id']}: {panel} is imported by {importers}, not by {expected} alone")
    assert len([g for g in groups if not g.get("retired")]) >= 40, "the census reads every group"
    assert not problems, "panels mounted by more than one place, or not by theirs:\n  " + "\n  ".join(problems)


# ===========================================================================
# PF28 -- Observability hosts its dashboards as tabs drawn from the catalog
# ===========================================================================
# Each group that keeps a feature key: the key, the route file its panel
# calls, and what that route checks before it answers (a flag it reads, or a
# module it imports). The health map must serve the key from that same flag.
_GATES = {
    "knowledge-base": ("rag_store", "opti_oignon/api/routes_rag.py", "RAG_STORE_AVAILABLE", None),
    "rag-dashboard": ("rag_dashboard", "opti_oignon/api/routes_rag_dashboard.py",
                      "RAG_DASHBOARD_AVAILABLE", "opti_oignon.rag_dashboard"),
    "installed-plugins": ("plugin_registry", "opti_oignon/api/routes_plugins.py", "PLUGIN_REGISTRY_AVAILABLE", None),
    "plugin-marketplace": ("plugin_index", "opti_oignon/api/routes_plugin_marketplace.py",
                           "PLUGIN_INDEX_AVAILABLE", None),
    "telemetry": ("telemetry", "opti_oignon/api/routes_telemetry.py", "TELEMETRY_AVAILABLE", None),
    "profiler": ("inference_profiler", "opti_oignon/api/routes_profiler.py", "INFERENCE_PROFILER_AVAILABLE", None),
    "performance-dashboard": ("performance_monitor", "opti_oignon/api/routes_performance.py",
                              "PERFORMANCE_MONITOR_AVAILABLE", None),
    "analytics": ("analytics", "opti_oignon/api/routes_feedback.py", "ANALYTICS_AVAILABLE", None),
}
# The groups whose routes check a flag the health map does not serve: a key
# borrowed from elsewhere would hide a working panel, so they carry none.
_UNKEYED = ("telemetry-history", "plugin-allowlist")


def _health_map():
    """``{key: flag name}`` of the health map ``app.py`` serves, read by ``ast``."""
    tree = ast.parse(read(_APP_PY))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "health_check":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Dict):
                    keys = [k.value for k in inner.keys if isinstance(k, ast.Constant)]
                    if "modules" in keys:
                        modules = inner.values[keys.index("modules")]
                        return {
                            key.value: (value.id if isinstance(value, ast.Name) else None)
                            for key, value in zip(modules.keys, modules.values)
                            if isinstance(key, ast.Constant)
                        }
    return {}


def _route_checks(path, flag, module):
    tree = ast.parse(read(path))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    return (module in modules) if module else (flag in names)


_CLIENT_HELPERS = ("apiGet", "apiPost", "apiPut", "apiPatch", "apiDelete", "apiUpload", "fetchApi", "wsUrl")


def _unimported_helpers(path, text):
    """How many client helpers an API module calls without importing them."""
    imported = set()
    for match in re.finditer(r"\bimport\s+(?!type\b)\{([^}]*)\}\s*from\s*['\"](?:\./client|\$lib/api/client)['\"]", text):
        imported |= {part.strip().split(" as ")[-1].strip() for part in match.group(1).split(",")}
    body = re.sub(r"\bimport\b[^;]*;", " ", _COMMENTS.sub(" ", text))
    return sum(
        1 for name in _CLIENT_HELPERS
        if re.search(rf"(?<![\w$.]){name}\s*[<(]", body) and name not in imported
        and not re.search(rf"\bfunction\s+{name}\b", body)
    )


@pytest.mark.parametrize("half", ("tabs", "keys", "gates", "overview", "ground", "wiring", "link", "save", "focus"))
def test_pf28_observability_hosts_its_dashboards_as_tabs_from_the_catalog(half):
    if half == "tabs":
        assert _imports_name(_HOST, "Tabs", _DS_INDEX) or f"{_DS}/Tabs.svelte" in _imports(_HOST), (
            "the host draws its tabs with the ds Tabs"
        )
        catalog = _catalog()
        by_id = {group["id"]: group for group in _all_groups(catalog)}
        root = _render(_HOST)
        lists = [e for e in root.iter() if e.get("role") == "tablist"]
        assert len(lists) == 1, f"one row of tabs: {len(lists)}"
        labels = [_text(tab) for tab in lists[0].iter("button") if tab.get("role") == "tab"]
        assert labels == ["Overview"] + [by_id[gid]["title"] for gid in _EMBEDDED], (
            f"Overview, then the embedded groups by their catalog titles, in order: {labels}"
        )
        panels = re.findall(r"\b(\w+Dashboard|\w+Panel)\b", _script(_HOST))
        assert all(by_id[gid]["panel"] in panels for gid in _EMBEDDED), "every embedded panel is in its map"
        return

    if half == "keys":
        served = _health_map()
        assert len(served) >= 40 and served.get("telemetry") == "TELEMETRY_AVAILABLE", (
            f"the census reads the health map: {len(served)} keys"
        )
        catalog = _catalog()
        keyed = {group["id"]: group["feature"] for group in _all_groups(catalog) if group.get("feature")}
        unserved = {gid: key for gid, key in keyed.items() if key not in served}
        assert not unserved, f"feature keys the health map does not serve, so their gate never closes: {unserved}"
        assert keyed == {gid: entry[0] for gid, entry in _GATES.items()}, (
            f"the keys the catalog names are the ones each panel's routes check: {keyed}"
        )
        for gid in _UNKEYED:
            assert gid not in keyed, f"{gid} checks a flag the health map does not serve: it carries no key"
        for gid, (key, route, flag, module) in _GATES.items():
            assert served.get(key) == flag, f"{gid}: the health map serves {key} from {served.get(key)}, not {flag}"
            assert _route_checks(route, flag, module), f"{gid}: {route} does not check {module or flag}"
        deps = read(_DEPS_PY)
        assert re.search(r"RAG_DASHBOARD_AVAILABLE\s*=\s*_module_exists\(\s*['\"]opti_oignon\.rag_dashboard['\"]", deps), (
            "the dashboard's served flag is the import of the module its route imports"
        )
        return

    if half == "gates":
        script = _script(_HOST)
        assert _imports_name(_HOST, "getFeatureMap", f"{_API}/featureCheck.ts") and _calls(script, "getFeatureMap"), (
            "the host reads the feature map"
        )
        assert re.search(r"<FeatureUnavailable\b", _markup(_HOST)), "a tab whose key is off says so"
        assert re.search(r"\.feature\b", script), "each tab is gated by its group's key"
        got = _node("gate", ("OO_STATE",), [
            ["telemetry", {"telemetry": False}], ["telemetry", {"telemetry": True}], ["telemetry", {}],
            [None, {"telemetry": False}], ["telemetry", {"inference_profiler": False}],
        ])
        assert got == [True, False, False, False, False], (
            f"a tab shuts when its own key reads false, and on nothing else: {got}"
        )
        assert _imports_name(_HOST, "tabUnavailable", _STATE) and re.search(
            r"tabUnavailable\(\s*group\.feature\s*,\s*featureMap\s*\)", script
        ), "the host asks the module, with the shown tab's own key"
        off = {"inference_profiler": False, "performance_monitor": False}
        for active, expected in (("overview", []), ("profiler", ["Profiler"]),
                                 ("performance-dashboard", ["Performance dashboard"]), ("telemetry", [])):
            shown = _visible(_render(_HOST, {"active": active, "featureMap": off}))
            said = [title for title in ("Overview", "Telemetry", "Profiler", "Performance dashboard")
                    if f"{title} is not available" in shown]
            assert said == expected, f"on {active}, the tabs said shut are {said}, not {expected}: {shown[:300]!r}"
        return

    if half == "overview":
        markup = _markup(_HOST)
        dot = re.compile(r"""class\s*=\s*["'][^"']*\b(?:status-dot|oo-dot)\b""")
        assert len(dot.findall('<span class="w-2 status-dot"></span><i class="oo-dot on"></i>')) == 2, (
            "the census reads a coloured mark"
        )
        dots = dot.findall(markup)
        assert not dots and "class:green" not in markup, f"no state shown by a coloured mark alone: {dots}"
        got = _node("state", ("OO_STATE",), {
            "telemetry": [{"enabled": True}, {"enabled": False}, None],
            "profiler": [{"total_profiled_requests": 12}, {"total_profiled_requests": 1},
                         {"total_profiled_requests": 0}, None],
            "history": [{"available": True, "total_stored": 340}, {"available": True, "total_stored": 1},
                        {"available": False, "total_stored": 0}, None],
        })
        assert got == {
            "telemetry": ["Collecting", "Off", "Unavailable"],
            "profiler": ["12 requests profiled", "1 request profiled", "Nothing profiled yet", "Unavailable"],
            "history": ["340 events stored", "1 event stored", "History off", "Unavailable"],
        }, f"each state is said in words: {got}"
        return

    if half == "ground":
        assert not re.search(r"<Card\b", _markup(_HOST)), "the overview is never a card on the group's tone"
        rules = _style_rules(_HOST)
        blocks = [
            (selector, declared) for selector, declared, _ in rules
            if declared.get("background-color") == "var(--oo-bg-subtle)"
        ]
        assert blocks, "the overview blocks sit on the sunken ground"
        grid = _rule(rules, lambda s: s == ".oo-obs-overview")
        assert re.search(r"minmax\(\s*min\(\s*[\d.]+rem\s*,\s*100%\s*\)", grid.get("grid-template-columns", "")), (
            f"a block is never wider than the group that holds it, whatever the text size: {grid}"
        )
        for selector, declared in blocks:
            assert declared.get("border") == "1px solid var(--oo-edge)", f"{selector} carries the edge: {declared}"
            assert declared.get("border-radius") == "var(--oo-radius-md)", f"{selector} is rounded: {declared}"
        assert not re.search(r"--oo-bd-|--oo-bg-surface\b", "\n".join(str(d) for _, d, _ in rules)), (
            "no border token, and not the group's own tone"
        )
        return

    if half == "wiring":
        script = _script(_HOST)
        assert _imports_name(_HOST, "embeddedGroups", _CATALOG) and _calls(script, "embeddedGroups"), (
            "the host's tabs are the catalog's embedded groups"
        )
        for name in ("telemetryState", "profilerState", "historyState"):
            assert _imports_name(_HOST, name, _STATE) and _calls(_code(_HOST), name), f"the overview says {name}"
        named = re.compile(r"""['"`](Telemetry history|Profiler|Performance dashboard|Telemetry)['"`]""")
        assert named.findall("{ id: 'x', label: 'Profiler' }, \"Telemetry\"") == ["Profiler", "Telemetry"], (
            "the census reads a tab named by hand"
        )
        literals = named.findall(script)
        assert not literals, f"the host names no tab of its own: {literals}"
        assert not re.search(r"\b(?:enabled|available)\s*\?", _markup(_HOST)), "and decides no state of its own"
        return

    if half == "link":
        script = _script(_HOST)
        handler = re.search(r"on:selectModel\s*=\s*\{\s*(\w+)\s*\}", _markup(_HOST))
        assert handler, "the host listens to the profiler's pick"
        body = _function_body(script, handler.group(1))
        assert re.search(r"\b\w+\s*=\s*(?:['\"](?:telemetry-)?history['\"]|\w*[Hh]istory\w*)", body), (
            f"a pick opens the history: {body!r}"
        )
        assert re.search(r"initialModelFilter\s*=\s*\{", _markup(_HOST)), "filtered by the picked model"
        assert "Clear filter" in _markup(_HOST), "and the filter can be cleared"
        cleared = re.search(r"on:click=\{\s*(\w+)\s*\}\s*>\s*Clear filter", _markup(_HOST))
        emptied = re.search(r"\b(\w+)\s*=\s*['\"]{2}\s*;", _function_body(script, cleared.group(1))) if cleared else None
        keyed = re.search(
            rf"\{{#key\s+{emptied.group(1)}\s*\}}\s*<TelemetryHistoryPanel\s+initialModelFilter=\{{\s*{emptied.group(1)}\s*\}}\s*/>\s*\{{/key\}}",
            _markup(_HOST),
        ) if emptied else None
        assert keyed, (
            "clearing the filter draws the history afresh, unfiltered: the history reads its filter once, "
            "when it is built, so it is built again whenever the picked model changes"
        )
        return

    if half == "focus":
        tabs = _script(_TABS)
        assert re.search(r"\bexport\s+(?:async\s+)?function\s+focusSelected\s*\(", tabs), "the ds Tabs can focus its selected tab"
        focus = _function_body(tabs, "focusSelected")
        assert re.search(r"tabEls\[[^\]]*value[^\]]*\]\??\.focus\(\)", focus), f"the selected one: {focus!r}"
        bound = re.search(r"<Tabs\b[^>]*\bbind:this=\{\s*(\w+)\s*\}", _markup(_HOST))
        assert bound, "the host holds its row of tabs"
        script = _script(_HOST)
        for name in ("openTab", "pickModel", "clearLinkedModel"):
            body = _function_body(script, name)
            assert re.search(rf"\b{bound.group(1)}\??\.focusSelected\(\)", body), (
                f"{name} changes what is shown and removes the control pressed: it hands focus to the selected tab: {body!r}"
            )
        return

    sample = "import { apiGet } from './client';\nexport const f = () => apiPut('/x', {});\n"
    assert _unimported_helpers("frontend/src/lib/api/sample.ts", sample) == 1, "the census reads a missing import"
    assert _unimported_helpers("frontend/src/lib/api/sample.ts", sample.replace("{ apiGet }", "{ apiGet, apiPut }")) == 0
    census = {
        path: count for path in files(".ts", within=_API)
        if path != f"{_API}/client.ts" and (count := _unimported_helpers(path, read(path)))
    }
    assert not census, f"API modules that call a client helper they never import: {census}"


# ===========================================================================
# PF29 -- every Workshop dialog holds Stop all, and never opens on it
# ===========================================================================
_WORKSHOP_DIALOGS = {
    f"{_PANELS}/benchmark/BenchmarkRunDrawer.svelte": 1,
    f"{_SETTINGS}/FineTunePanel.svelte": 1,
    f"{_SETTINGS}/HardeningPanel.svelte": 1,
    f"{_SETTINGS}/KnowledgeBasePanel.svelte": 1,
    f"{_SETTINGS}/PluginMarketplace.svelte": 1,
    f"{_SETTINGS}/RemoteAccessPanel.svelte": 2,
    f"{_PANELS}/SyncPanel.svelte": 1,
    f"{_LIB}/components/rag/DocumentManager.svelte": 2,
}


def _dialogs_with_stops(markup):
    """``[bool]``: for each ConfirmDialog and Modal element, whether it
    holds Stop all, placed for the head, in its actions fragment."""
    out = []
    for match in re.finditer(r"<(ConfirmDialog|Modal)(?=[\s/>])", markup):
        stop = _tag_end(markup, match.start())
        if markup[stop - 2:stop] == "/>":
            out.append(False)
            continue
        name = match.group(1)
        pattern = re.compile(rf"<(/?){name}(?=[\s/>])")
        depth, cursor, end = 1, stop, len(markup)
        while depth:
            inner = pattern.search(markup, cursor)
            if not inner:
                break
            inner_stop = _tag_end(markup, inner.start())
            if inner.group(1):
                depth -= 1
            elif markup[inner_stop - 2:inner_stop] != "/>":
                depth += 1
            cursor = end = inner_stop
        content = markup[stop:end]
        fragment = re.search(
            r"""<svelte:fragment\s+slot\s*=\s*["']actions["']\s*>(.*?)</svelte:fragment>""", content, re.S
        )
        out.append(bool(fragment and re.search(
            r"""<StopAllButton\b(?=[^>]*\bplacement\s*=\s*["']dialog-head["'])""", fragment.group(1)
        )))
    return out


@pytest.mark.parametrize("half", ("slot", "census", "focus"))
def test_pf29_every_workshop_dialog_holds_stop_all_and_never_opens_on_it(half):
    if half == "slot":
        path = _plant("ConfirmWithActions", (
            "<script>\n"
            "\timport ConfirmDialog from '$lib/ds/ConfirmDialog.svelte';\n"
            "</script>\n"
            "<ConfirmDialog open title=\"Delete it?\" onConfirm={() => {}} onCancel={() => {}}>"
            "<svelte:fragment slot=\"actions\"><span class=\"oo-probe-action\">probe</span></svelte:fragment>"
            "</ConfirmDialog>\n"
        ))
        root = _render(path)
        probes = [e for e in root.iter("span") if "oo-probe-action" in e.classes()]
        headers = list(root.iter("header"))
        assert len(probes) == 1 and headers and _inside(probes[0], headers[0]), (
            "ConfirmDialog renders its actions slot in the dialog's head"
        )
        return

    if half == "census":
        sample = (
            "<ConfirmDialog open title=\"x\" />\n"
            "<Modal title=\"y\"><svelte:fragment slot=\"actions\">"
            "<StopAllButton placement=\"dialog-head\" /></svelte:fragment>body</Modal>\n"
        )
        assert _dialogs_with_stops(sample) == [False, True], "the census reads each dialog apart"
        findings, total = {}, 0
        for path, expected in _WORKSHOP_DIALOGS.items():
            held = _dialogs_with_stops(_markup(path))
            total += len(held)
            if len(held) < expected or not all(held):
                findings[path] = held
        assert total >= 10, f"the census reads the Workshop's ten dialogs: {total}"
        assert not findings, f"dialogs without Stop all in their head: {findings}"
        return

    markup, script = _markup(_MODAL), _script(_MODAL)
    assert re.search(
        r"""<span\s+class\s*=\s*["']oo-modal-actions["']\s*>\s*<slot\s+name\s*=\s*["']actions["']""", markup
    ), "the head's actions are held in their own container"
    body = _function_body(script, "openDialog")
    marked = body.find("[data-autofocus]")
    skipped = re.search(r"closest\(\s*['\"]\.oo-modal-actions['\"]\s*\)", body)
    assert marked >= 0 and skipped and marked < skipped.start(), (
        f"a marked element wins; else the first focusable outside the head's actions: {body!r}"
    )
    stop, close, field = {"id": "stop", "actions": True}, {"id": "close"}, {"id": "field"}
    note = {"id": "note", "actions": True}
    got = _node("focus", ("OO_FIRST_FOCUS",), [
        [[stop, close, field], None],
        [[stop, note, field], None],
        [[stop, close, field], field],
        [[stop, close], stop],
        [[stop, note], None],
        [[], None],
    ])
    assert got == ["close", "field", "field", "stop", None, None], (
        "the first focusable outside the head's actions, whatever stands before it in the head; a marked "
        f"element wins wherever it sits; none when only the head's actions can take focus: {got}"
    )
    call = _nav._call_argument(body, "firstFocus")
    assert _imports_name(_MODAL, "firstFocus", _FIRST_FOCUS) and re.search(
        r"querySelector(?:<\w+>)?\(\s*['\"]\[autofocus\],\s*\[data-autofocus\]['\"]\s*\)", call
    ) and re.search(r"\(\s*(\w+)\s*\)\s*=>\s*\1\.closest\(\s*['\"]\.oo-modal-actions['\"]\s*\)\s*!==\s*null", call), (
        f"the dialog asks the module, the marked element and the head's actions named: {call!r}"
    )


# ===========================================================================
# PF30 -- nothing unreachable stays in the source
# ===========================================================================
# The modules under lib that no route reaches, each with the reason it stays.
UNREACHABLE = {
    "frontend/src/lib/api/modelLifecycle.ts": 1,
}
UNREACHABLE_REASONS = {
    "frontend/src/lib/api/modelLifecycle.ts": (
        "the typed client for pulling, deleting and updating models, kept until a page manages them"
    ),
}

_MINI_TREE = {
    f"{_ROUTES}/probe/+page.svelte": (
        "<script>\n\timport A from '$lib/probe/A.svelte';\n\timport type { T } from '$lib/probe/types';\n"
        "\timport '$lib/probe/sheet.css';\n\tconst later = () => import('$lib/probe/Later.svelte');\n</script>\n<A />\n"
    ),
    f"{_LIB}/probe/A.svelte": "<script>\n\timport { b } from '$lib/probe/barrel';\n</script>\n",
    f"{_LIB}/probe/barrel.ts": "export { b } from './b';\n",
    f"{_LIB}/probe/b.ts": "export const b = 1;\n",
    f"{_LIB}/probe/types.ts": "export type T = string;\n",
    f"{_LIB}/probe/sheet.css": ".x { color: red; }\n",
    f"{_LIB}/probe/Later.svelte": "<p>later</p>\n",
    f"{_LIB}/probe/Orphan.svelte": "<p>no one imports me</p>\n",
}


def _reached_from_routes():
    graph = _source_graph()
    return _reached(graph, _roots(graph))


@pytest.mark.parametrize("half", ("graph", "gone", "stylesheet"))
def test_pf30_nothing_unreachable_stays_in_the_source(half):
    if half == "graph":
        mini = _graph(_MINI_TREE)
        reached = _reached(mini, _roots(mini))
        unreached = sorted(set(_MINI_TREE) - reached)
        assert unreached == [f"{_LIB}/probe/Orphan.svelte"], (
            f"the builder follows a chain, a re-export, a type-only import, a stylesheet and a dynamic "
            f"import, and finds the orphan: {unreached}"
        )
        assert set(UNREACHABLE) == set(UNREACHABLE_REASONS) and all(UNREACHABLE_REASONS.values()), (
            "every exception carries its reason"
        )
        reached = _reached_from_routes()
        census = check_ledger(
            "UNREACHABLE", UNREACHABLE,
            lambda path, text: 0 if path in reached or path.endswith(".d.ts") else 1,
            (f"{_LIB}/probe/Orphan.svelte", "<p>no one imports me</p>"),
            test_file=__file__, suffixes=_GRAPH_KINDS, within=_LIB,
        )
        assert census.fixture == 1 and census.counts == UNREACHABLE, census.counts
        return

    if half == "gone":
        gone = (
            f"{_PANELS}/ClaimVerifier.svelte", f"{_API}/claimVerification.ts",
            f"{_SETTINGS}/SecurityPanel.svelte", f"{_LIB}/actions/focusTrap.ts",
        )
        present = [path for path in gone if (REPO / path).exists()]
        assert not present, f"orphans still in the source: {present}"
        return

    assert (REPO / _BENCHMARK_CSS).is_file(), "the sections' stylesheet is there"
    assert _BENCHMARK_CSS in _source_graph().get(_BENCHMARK_PAGE, set()), (
        "the page that renders the benchmark sections imports their stylesheet"
    )
    graph = _source_graph()
    assert _BENCHMARK_CSS in _reached(graph, [_BENCHMARK_ROUTE]), "and the benchmarks page reaches it"


# ===========================================================================
# PF31 -- the security page explains its grade
# ===========================================================================
_SECURITY_ROUTE = re.compile(r"/api/security/(?:status|audit)(?![\w-])")
_STATUS = {
    "grade": "B", "score": 70, "max_score": 100,
    "checks": [
        {"name": "Encryption at rest", "points": 20, "max_points": 20, "passed": True, "detail": "The stores are encrypted"},
        {"name": "Two-factor sign-in", "points": 0, "max_points": 10, "passed": False, "detail": "No second factor"},
    ],
}
_EVENTS = [
    {"source": "sandbox", "event_type": "sandbox", "action": "command_blocked", "severity": "critical",
     "timestamp": 1790000000, "details": {}},
    {"source": "auth", "event_type": "auth", "action": "login_failed", "severity": "warning",
     "timestamp": 1790000100, "details": {}},
]


@pytest.mark.parametrize("half", ("api", "page", "checks", "failure", "sessions"))
def test_pf31_the_security_page_explains_its_grade(half):
    if half == "api":
        assert _SECURITY_ROUTE.search("fetch('/api/security/status')") and not _SECURITY_ROUTE.search(
            "apiGet('/api/security/audit-chain/verify')"
        ), "the census reads the two routes and not the audit chain"
        readers = sorted(path for path in files(_SCRIPTS) if _SECURITY_ROUTE.search(_code(path)))
        assert readers == [_SECURITY_API], f"one client reads the grade and the events: {readers}"
        assert _imports_name(_SECURITY_BADGE, "getSecurityStatus", _SECURITY_API), "the badge reads through it"
        for name in ("getSecurityStatus", "getSecurityEvents"):
            assert _imports_name(_SECURITY_GRADE, name, _SECURITY_API) and _calls(_script(_SECURITY_GRADE), name), (
                f"the grade reads through {name}"
            )
        return

    if half == "page":
        security = re.compile(
            r"\{:else\s+if\s+space\s*===\s*'workshop'\s*&&\s*section\s*===\s*'security'\s*\}\s*<SecurityGrade\s*/>"
        )
        network = re.compile(
            r"\{#if\s+space\s*===\s*'workshop'\s*&&\s*section\s*===\s*'network'\s*\}\s*<NetworkReachability\s*/>\s*"
        )
        markup = _markup(_HUB)
        head = network.search(markup)
        assert head and security.match(markup, head.end()), (
            "the Security page, and it alone, opens on the grade, after the network page's head"
        )
        assert len(re.findall(r"<SecurityGrade\b", markup)) == 1 and _SECURITY_GRADE in _imports(_HUB)
        return

    if half == "checks":
        root = _render(_SECURITY_CHECKS, {"status": _STATUS})
        shown = _visible(root)
        for words in ("B", "70 of 100", "1 of 2 checks passed", "Encryption at rest", "20 of 20",
                      "The stores are encrypted", "Two-factor sign-in", "0 of 10", "No second factor"):
            assert words in shown, f"the checks say {words!r}: {shown!r}"
        assert len(re.findall(r"\bNot passed\b", shown)) == 1 and len(re.findall(r"(?<!Not )\bPassed\b", shown)) == 1, (
            f"each check says Passed or Not passed in words: {shown!r}"
        )
        return

    if half == "failure":
        root = _render(_SECURITY_GRADE, {"failure": "Server unreachable"})
        shown = _visible(root)
        assert "Could not read the security grade: Server unreachable" in shown, f"the failure is named: {shown!r}"
        assert [b for b in root.iter("button") if re.search(r"\bretry\b", _name(b), re.I)], "with a retry"
        script, markup = _script(_SECURITY_GRADE), _markup(_SECURITY_GRADE)
        load = _function_body(script, "loadStatus")
        answer = load.find("await ")
        clears = [m.start() for m in re.finditer(r"\bfailure\s*=\s*null\b", load)]
        assert answer >= 0 and clears and min(clears) > answer, (
            f"the failure and its retry stay on screen until the answer replaces them: {load!r}"
        )
        failing = re.search(r"\{#if\s+failure\s*\}(.*?)\{:else", markup, re.S)
        button = re.search(r"<Button\b([^>]*)>\s*Retry\s*</Button>", failing.group(1)) if failing else None
        assert button and not re.search(r"\b(?:loading|disabled)\s*=", button.group(1)), (
            f"the retry stays enabled while it reads: a disabled button drops its focus: {button and button.group(0)!r}"
        )
        handler = re.search(r"on:click=\{\s*(\w+)\s*\}", button.group(1))
        retry = _function_body(script, handler.group(1)) if handler else ""
        read = re.search(r"\bawait\s+loadStatus\(\)", retry)
        focus = re.search(r"\b(\w+)\??\.focus\(\)", retry)
        assert read and focus and read.start() < focus.start() and re.search(
            rf"\btabindex=\"-1\"[^>]*\bbind:this=\{{\s*{focus.group(1)}\s*\}}|\bbind:this=\{{\s*{focus.group(1)}\s*\}}[^>]*\btabindex=\"-1\"",
            markup,
        ), f"once the grade is read, focus goes to it: {retry!r}"
        return

    path = _plant("GradeDetail", (
        "<script>\n"
        "\timport { authStatus } from '$lib/stores/auth';\n"
        "\timport SecurityGrade from '$lib/components/settings/SecurityGrade.svelte';\n"
        "\tauthStatus.set({ single_user_mode: true, cookie_mode: true });\n"
        "\texport let status;\n\texport let events;\n"
        "</script>\n"
        "<SecurityGrade {status} {events} expanded />\n"
    ))
    root = _render(path, {"status": _STATUS, "events": _EVENTS})
    shown = _visible(root)
    assert "Sessions use httpOnly cookies" in shown, f"the session mode, from the auth store: {shown!r}"
    for words in ("command_blocked", "sandbox", "critical", "login_failed", "warning"):
        assert words in shown, f"the recent events say {words!r}: {shown!r}"
    assert "Encryption at rest" in shown, "and the checks"
    assert "Grade B" in _text(root), f"the letter is named as the grade: {_text(root)!r}"
    assert len(re.findall(r"\b70 of 100\b", _visible(root))) == 1, "and the score is said once on the page"
    assert re.search(r"\$authStatus\b", _code(_SECURITY_GRADE)), "read from the auth store"
    assert re.search(r"getSecurityEvents\(\s*10\s*\)", _script(_SECURITY_GRADE)), "the ten latest events"


# ===========================================================================
# PF32 -- model assignment is mounted once and says when it fails
# ===========================================================================
def _roles_page(name, read_state, roles, errors):
    path = _plant(name, (
        "<script>\n"
        "\timport { rolesRead, roles, installedModels, saveErrors } from '$lib/stores/modelRoles';\n"
        "\timport ModelAssignment from '$lib/components/panels/ModelAssignment.svelte';\n"
        f"\trolesRead.set({json.dumps(read_state)});\n"
        f"\troles.set({json.dumps(roles)});\n"
        "\tinstalledModels.set(['qwen', 'llama']);\n"
        f"\tsaveErrors.set({json.dumps(errors)});\n"
        "</script>\n"
        "<ModelAssignment />\n"
    ))
    return _render(path)


_ROLE_ROWS = [
    {"role": "coding", "primary": "qwen", "fast": "", "quality": "llama"},
    {"role": "chat", "primary": "llama", "fast": "", "quality": ""},
]


@pytest.mark.parametrize("half", ("importers", "read", "empty", "save", "kept"))
def test_pf32_model_assignment_is_mounted_once_and_says_when_it_fails(half):
    if half == "importers":
        importers = _importers(_MODEL_ASSIGNMENT)
        assert importers == [_HUB], f"only the hub mounts model assignment: {importers}"
        return

    if half == "read":
        root = _roles_page("RolesFailed", {"state": "error", "reason": "Server unreachable"}, [], {})
        shown = _visible(root)
        assert "Could not read the roles: Server unreachable" in shown, f"a failed read says so: {shown!r}"
        assert "No role assignments found." not in shown, "never as an empty list"
        assert [b for b in root.iter("button") if re.search(r"\bretry\b", _name(b), re.I)], "with a retry"
        failing = re.search(r"\{:else\s+if\s+\$rolesRead\.state\s*===\s*'error'\s*\}(.*?)\{:else", _markup(_MODEL_ASSIGNMENT), re.S)
        button = re.search(r"<Button\b([^>]*)>\s*Retry\s*</Button>", failing.group(1)) if failing else None
        handler = re.search(r"on:click=\{\s*(\w+)\s*\}", button.group(1)) if button else None
        retry = _function_body(_script(_MODEL_ASSIGNMENT), handler.group(1)) if handler else ""
        quiet = re.search(r"\bawait\s+loadRoles\(\s*true\s*\)", retry)
        focus = re.search(r"\b\w+\??\.focus\(\)", retry)
        assert quiet and focus and quiet.start() < focus.start() and not re.search(r"\bloading\s*=", button.group(1)), (
            "a retry reads quietly, so the failure and its retry stay until the answer, and the roles "
            f"read at last take focus: {retry!r}"
        )
        return

    if half == "empty":
        loading = _visible(_roles_page("RolesLoading", {"state": "loading"}, [], {}))
        assert "Loading roles" in loading and "No role assignments found." not in loading, loading
        idle = _visible(_roles_page("RolesIdle", {"state": "idle"}, [], {}))
        assert "Loading roles" in idle and "No role assignments found." not in idle, idle
        empty = _visible(_roles_page("RolesEmpty", {"state": "ok"}, [], {}))
        assert "No role assignments found." in empty, f"a read that found none says so: {empty!r}"
        return

    if half == "save":
        root = _roles_page("RolesSaveFailed", {"state": "ok"}, _ROLE_ROWS, {"coding": "Refused by the server"})
        cards = {card.get("data-role"): card for card in root.iter() if card.get("data-role")}
        assert set(cards) == {"coding", "chat"}, f"each role has its card: {sorted(cards)}"
        alerts = [e for e in cards["coding"].iter() if e.get("role") == "alert"]
        assert alerts and "Refused by the server" in _text(alerts[0]), "the failure shows under its role"
        assert not [e for e in cards["chat"].iter() if e.get("role") == "alert"], "and under no other"
        return

    handle = _function_body(_script(_MODEL_ASSIGNMENT), "handleSave")
    closing = re.search(r"\bif\s*\(\s*(\w+)\s*\)\s*\{?\s*editingRole\s*=\s*null", handle)
    saved = re.search(r"\b(?:const|let)\s+(\w+)\s*=\s*await\s+saveRole\(", handle)
    assert closing and saved and closing.group(1) == saved.group(1), (
        f"the editor closes only when the save succeeded: {handle!r}"
    )
    assert len(re.findall(r"editingRole\s*=\s*null", handle)) == 1, "and on no other path"
    body = _function_body(_script(_ROLES), "saveRole")
    assert re.search(r"\btry\s*\{", body) and re.search(r"\bcatch\b", body), "saveRole catches its failure"
    assert re.search(r"return\s+true\b", body) and re.search(r"return\s+false\b", body), "and says how it went"
    assert "saveErrors" in body and "throw" not in body, "it records the failure under its role, never throws"


# ===========================================================================
# PF33 -- the first run refreshes what a preset changes instead of reloading
# ===========================================================================
_PRESETS = [
    {"id": "minimal", "name": "Minimal", "description": "The least.", "icon": "leaf", "recommended_ram_gb": 8},
    {"id": "balanced", "name": "Balanced", "description": "The middle.", "icon": "scale", "recommended_ram_gb": 16},
    {"id": "power", "name": "Power", "description": "The most.", "icon": "zap", "recommended_ram_gb": 32},
]
_DETECTION = {"models": [], "recommended_preset": "balanced", "reason": "Sixteen gigabytes found."}


@pytest.mark.parametrize("half", ("reload", "refresh", "dialog", "choice", "skip", "focus"))
def test_pf33_the_first_run_refreshes_what_a_preset_changes_instead_of_reloading(half):
    reload = re.compile(r"\blocation\s*\.\s*reload\s*\(")
    if half == "reload":
        assert reload.search("window.location.reload();"), "the census reads a reload"
        assert not reload.search(_code(_OVERLAY)), "the first-run dialog reloads nothing"
        return

    if half == "refresh":
        overlay = _script(_OVERLAY)
        assert _imports_name(_OVERLAY, "refreshAfterConfigChange", _REFRESH) and _calls(overlay, "refreshAfterConfigChange"), (
            "Get started refreshes what the preset changed"
        )
        defaults = _script(_CONVERSATION_DEFAULTS)
        for name in ("handleApplySystemPreset", "handleReload"):
            assert _calls(_function_body(defaults, name), "refreshAfterConfigChange"), (
                f"so does {name} on the Models page"
            )
        refresh = _script(_REFRESH)
        body = _function_body(refresh, "refreshAfterConfigChange")
        for name in ("loadOptions", "invalidateFeatureCache", "getFeatureMap", "refreshBackendStatus"):
            assert re.search(rf"\b{name}\s*\(", body), f"the refresh calls {name}: {body!r}"
        assert re.search(r"configEpoch|epoch\.update|epoch\.set", body), "and then moves the epoch"
        bar = _script(_CONTROL_BAR)
        assert _imports_name(_CONTROL_BAR, "configEpoch", _REFRESH) and re.search(
            r"\$:\s*[^;\n]*\$configEpoch", bar
        ), "the chat's control bar reads its switches again when the epoch moves"
        seen = re.search(r"\blet\s+(\w+)\s*=\s*\$configEpoch\s*;", bar)
        moved = re.search(
            rf"\$:\s*if\s*\(\s*\$configEpoch\s*!==\s*{seen.group(1)}\s*\)\s*\{{\s*{seen.group(1)}\s*=\s*\$configEpoch\s*;",
            bar,
        ) if seen else None
        assert moved, (
            "only a move after the bar was built reads them again: the epoch it was built at is no "
            "change, since the bar reads its switches once when mounted"
        )
        return

    if half == "skip":
        overlay, markup = _script(_OVERLAY), _markup(_OVERLAY)
        skip = _function_body(overlay, "handleSkip")
        assert re.search(r"\bif\s*\(\s*applyResult\s*\)\s*refreshAfterConfigChange\(\)", skip), (
            f"Escape and the close button, once a preset was applied, read again what it changed: {skip!r}"
        )
        modal = next((body for _, body, _ in _tags(markup, r"Modal")), "")
        assert re.search(r"\bonClose=\{\s*handleSkip\s*\}", modal), "they close through Skip"
        for name in ("closable", "closeOnEsc"):
            assert re.search(rf"\b{name}=\{{\s*step\s*!==\s*'applying'\s*\}}", modal), (
                f"the dialog cannot be closed while a preset is being applied: {name}"
            )
        for step, shut in (("applying", True), ("ready", False), ("done", False)):
            root = _render(_OVERLAY, {"visible": True, "step": step, "presets": _PRESETS,
                                      "detection": _DETECTION, "selectedPresetId": "balanced"})
            closes = [b for b in root.iter("button") if b.get("aria-label") == "Close dialog"]
            assert len(closes) == 1 and closes[0].has("disabled") == shut, (
                f"{step}: the close button {'waits' if shut else 'closes'}: {closes and closes[0].attrs}"
            )
        return

    if half == "focus":
        overlay, markup = _script(_OVERLAY), _markup(_OVERLAY)
        focus = _function_body(overlay, "focusStep")
        for step, target in (("done", "actions"), ("error", ".oo-inline-error button"), ("applying", ".ob-step-note")):
            assert re.search(rf"'{step}'", focus) and target in focus, (
                f"{step}: focus goes to what the step shows ({target}): {focus!r}"
            )
        assert re.search(r"<span\s+class=\"ob-actions\"\s+bind:this=\{actions\}>", markup), "the footer's actions are found"
        notes = re.findall(r"<p\s+class=\"[^\"]*\bob-step-note\b[^\"]*\"\s+tabindex=\"-1\"", markup)
        assert len(notes) == 2, f"the notes of the two steps under way can take focus: {notes}"
        apply = _function_body(overlay, "handleApply")
        calls = [m.start() for m in re.finditer(r"\bfocusStep\(\)", apply)]
        under_way = apply.find("step = 'applying'")
        answer = apply.find("await applySystemPreset(")
        assert len(calls) == 2 and under_way < calls[0] < answer < calls[1], (
            f"applying hands focus to its note, and its outcome to its own control: {apply!r}"
        )
        load = _function_body(overlay, "loadData")
        assert re.search(r"step\s*=\s*'error'\s*;\s*focusStep\(\)", load), "a failed scan hands focus to its retry"
        assert _calls(_function_body(overlay, "retryLoad"), "focusStep") and re.search(
            r"<InlineError\b[^>]*\bonRetry=\{\s*retryLoad\s*\}", markup
        ), "and a retry to the scan's note"
        return

    root = _render(_OVERLAY, {"visible": True, "step": "ready", "presets": _PRESETS,
                              "detection": _DETECTION, "selectedPresetId": "balanced"})
    dialogs = list(root.iter("dialog"))
    assert len(dialogs) == 1, f"one dialog: {len(dialogs)}"
    if half == "dialog":
        header = next(iter(dialogs[0].iter("header")), None)
        assert header is not None and "Welcome to Opti-Oignon" in _text(header), "the dialog keeps its name"
        closes = [b for b in header.iter("button") if b.get("aria-label") == "Close dialog"]
        stops = [s for s in _stops(root) if _inside(s, header)]
        assert closes and len(stops) == 1, (
            f"its head holds its close button and Stop all: {len(closes)} close, {len(stops)} stops"
        )
        return

    groups = [e for e in root.iter() if e.get("role") == "radiogroup"]
    assert len(groups) == 1 and groups[0].get("aria-label") == "System preset", (
        f"the presets are one radio group named System preset: {[g.attrs for g in groups]}"
    )
    radios = [e for e in groups[0].iter() if e.get("role") == "radio"]
    assert [r.get("aria-checked") for r in radios] == ["false", "true", "false"], (
        f"each preset says whether it is chosen: {[r.attrs for r in radios]}"
    )
    chosen = radios[1]
    assert list(chosen.iter("svg")) and not list(radios[0].iter("svg")), "the chosen preset draws a check"
    assert chosen.has("data-autofocus") and not radios[0].has("data-autofocus"), "the dialog opens on it"
    assert "Recommended" in _visible(chosen) and "Recommended" not in _visible(radios[0]), (
        "the recommended preset says so in a word"
    )


# ===========================================================================
# PF34 -- the benchmarks page is one row of tabs over one engine
# ===========================================================================
_BENCHMARK_LABELS = ["Run", "Leaderboard", "Head-to-head", "Trends", "Compare", "History", "Profiles"]
_BENCHMARK_IDS = ["run", "leaderboard", "h2h", "trends", "compare", "history", "profiles"]


@pytest.mark.parametrize("half", ("tabs", "address", "wiring", "link", "engine", "word", "row", "ring"))
def test_pf34_the_benchmarks_page_is_one_row_of_tabs_over_one_engine(half):
    if half == "tabs":
        assert _imports_name(_BENCHMARK_PAGE, "Tabs", _DS_INDEX) or _TABS in _imports(_BENCHMARK_PAGE), (
            "the page draws its tabs with the ds Tabs"
        )
        for current in ("run", "history"):
            root = _render(_BENCHMARK_PAGE, {"tab": current})
            lists = [e for e in root.iter() if e.get("role") == "tablist"]
            assert len(lists) == 1, f"one row of tabs, none nested: {len(lists)}"
            tabs = [b for b in lists[0].iter("button") if b.get("role") == "tab"]
            labels = [_text(tab) for tab in tabs]
            assert labels == _BENCHMARK_LABELS, f"the seven sections, in words alone: {labels}"
            selected = [_text(tab) for tab in tabs if tab.get("aria-selected") == "true"]
            assert selected == [_BENCHMARK_LABELS[_BENCHMARK_IDS.index(current)]], (
                f"the tab it is given is selected: {selected}"
            )
            assert [h.tag for h in _headings(root)][:1] == ["h1"], "the page is titled once, at level one"
        return

    if half == "address":
        got = _node("tab", ("OO_TAB",), {
            "read": [
                "http://localhost/workshop/benchmarks?tab=history",
                "http://localhost/workshop/benchmarks?tab=h2h&run=abc",
                "http://localhost/workshop/benchmarks",
                "http://localhost/workshop/benchmarks?tab=models",
                "http://localhost/workshop/benchmarks?tab=",
            ],
            "write": [
                ["http://localhost/workshop/benchmarks?run=abc&z=1", "trends"],
                ["http://localhost/workshop/benchmarks?tab=history&run=abc", "run"],
            ],
        })
        assert got["tabs"] == _BENCHMARK_IDS, f"the seven tabs: {got['tabs']}"
        assert got["read"] == ["history", "h2h", "run", "run", "run"], (
            f"the address names the tab; anything else is Run: {got['read']}"
        )
        outs = [entry["out"] for entry in got["write"]]
        assert outs == [
            "/workshop/benchmarks?run=abc&z=1&tab=trends",
            "/workshop/benchmarks?tab=run&run=abc",
        ], f"a tab is written into the address, run and the rest kept: {outs}"
        assert all(entry["untouched"] for entry in got["write"]), "the URL it is given is never changed"
        return

    if half == "wiring":
        script, markup = _script(_BENCHMARK_ROUTE), _markup(_BENCHMARK_ROUTE)
        for name in ("benchmarkTab", "tabAddress"):
            assert _imports_name(_BENCHMARK_ROUTE, name, _BENCHMARK_TAB) and _calls(_code(_BENCHMARK_ROUTE), name), (
                f"the page reads and writes its tab through {name}"
            )
        call = _nav._call_argument(script, "goto")
        assert "tabAddress(" in script and re.search(r"replaceState\s*:\s*true", script), (
            "a tab replaces the address entry"
        )
        rule = re.compile(r"""searchParams\.(?:set|get)\(\s*['"]tab['"]""")
        assert len(rule.findall("u.searchParams.get('tab'); u.searchParams.set(\"tab\", x);")) == 2, (
            "the census reads a tab rule written by hand"
        )
        assert not rule.search(_code(_BENCHMARK_ROUTE)), (
            f"the page holds no tab rule of its own: {call!r}"
        )
        assert re.search(r"searchParams\.get\(\s*['\"]run['\"]\s*\)", script) and re.search(
            r"<BenchmarkRunDrawer\b", markup
        ), "?run= still opens the run's drawer"
        return

    if half == "link":
        runs = _function_body(_script(_HISTORY_SECTION), "runHref")
        assert re.search(r"tab=history", runs) and re.search(r"run=", runs), (
            f"a History run opens over History and closes onto it: {runs!r}"
        )
        return

    if half == "engine":
        older = (
            f"{_PANELS}/BenchmarkRunner.svelte", f"{_PANELS}/BenchmarkHistory.svelte",
            f"{_PANELS}/BenchmarkV2Panel.svelte", f"{_LIB}/stores/benchmark.ts",
        )
        importers = {path: found for path in older if (found := _importers(path))}
        assert not importers, f"the older engine's interface is still imported: {importers}"
        routes = re.compile(r"/api/benchmark/(?:llm|runs|suites)\b|\bconnectBenchmarkProgress\b|new\s+WebSocket\s*\(\s*wsUrl\(\s*['\"]/api/benchmark")
        assert len(routes.findall("apiGet('/api/benchmark/runs'); connectBenchmarkProgress();")) == 2
        naming = sorted(path for path in files(_SCRIPTS) if routes.search(_code(path)))
        assert not naming, f"files that still reach the older engine: {naming}"
        return

    if half == "word":
        markup = _markup(_HEALTH_DASHBOARD)
        texts = _heading_texts(markup)
        assert "Component latency" in texts and "Benchmarks" not in texts, (
            f"System status measures component latency, and says so: {texts}"
        )
        assert re.search(r">\s*Measure\s*<", markup), "its button reads Measure"
        return

    rules = _style_rules(_TABS)
    listed = _rule(rules, lambda s: s in (".oo-tablist", ".oo-tabs[data-orientation='horizontal'] .oo-tablist"))
    if half == "ring":
        room = listed.get("--oo-tablist-room", "")
        assert re.fullmatch(r"calc\(\s*var\(--oo-focus-ring-width\)\s*\+\s*var\(--oo-focus-ring-offset\)\s*\)", room), (
            f"the row's room is the focus ring's width and offset: {listed}"
        )
        sides = re.findall(r"(?:max\([^()]*(?:\([^()]*\)[^()]*)*\)|\S+)", listed.get("padding", ""))
        top, inline, bottom = (sides + ["", "", ""])[:3] if len(sides) == 3 else ("", "", "")
        assert all("var(--oo-tablist-room)" in side for side in (top, inline, bottom)), (
            f"a scrolling box clips at its padding: the row keeps that room on every side: {sides}"
        )
        assert "padding-block" not in listed and "padding-inline" not in listed, f"and nothing takes it back: {listed}"
        return
    assert listed.get("overflow-x") == "auto", f"the tab row scrolls within itself: {listed}"
    script = _script(_TABS)
    keep = re.search(r"\$:\s*(\w+)\(\s*value\b", script)
    body = _function_body(script, keep.group(1)) if keep else ""
    assert keep and "scrollIntoView(" in body and re.search(r"inline\s*:\s*['\"]nearest['\"]", body), (
        f"and keeps the selected tab in view whenever it changes: {body!r}"
    )
    first = re.search(r"if\s*\(\s*!shown\s*\)\s*\{(.*?)\breturn\s*;", body, re.S)
    reveal = re.search(r"\b(\w+)\(\s*index\s*\)", first.group(1)) if first else None
    row = _function_body(script, reveal.group(1)) if reveal else ""
    assert re.search(r"\bscrollLeft\s*[-+]?=", row) and "scrollIntoView" not in row and "scrollIntoView" not in first.group(1), (
        "when the row first shows, it is scrolled alone to the selected tab, so a link to a far tab "
        f"lands on it and the page never moves: {body!r}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly"]))
