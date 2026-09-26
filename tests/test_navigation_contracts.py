#!/usr/bin/env python3
"""Contracts for the navigation: one destination table, the two spaces, the old URLs.

The interface has two spaces. Use holds the pages a person works in (home,
chats, notes, projects, the componion, preferences); Workshop holds the
operator pages under ``/workshop``. Every link to a destination is read from
one table, ``frontend/src/lib/nav/destinations.ts``; which entry is current
is decided by ``lib/nav/active.ts``; switching space by ``lib/nav/space.ts``;
and every URL the interface used to serve is answered by
``lib/nav/legacy.ts``, which is handed the settings catalog rather than
importing it, so it runs alone under Node and is called by a redirect in
each old page's ``load``.

The modules and the names the contracts read:

  * ``destinations.ts`` -- ``DESTINATIONS``, the table: each entry has an
    ``id``, a ``label``, an ``href``, a ``space`` (``use`` or
    ``workshop``), an ``icon``, ``ready`` (false while the destination has
    no page: it is never rendered) and ``settings`` (true for a page that
    renders groups of the settings catalog). ``visibleDestinations(list,
    {componion})`` is the entries to render: the ready ones, the componion's
    only while its switch is on. ``spaceHome(space, list)`` is the first
    ready entry of a space. ``isWorkshopSection(param)`` says whether a
    Workshop settings page is named by the last segment of its URL; the
    route parameter matcher ``src/params/workshopSection.ts`` is that
    function and holds no list of its own.
  * ``active.ts`` -- ``activeState(pathname, href, hrefs)`` is ``active``
    for the page itself, ``section`` for a page under it that no other
    destination in ``hrefs`` claims more closely, ``none`` otherwise; home
    is only ever active on itself. ``destinationFor(pathname, list)`` is the
    entry that is active or holds the section, or null.
  * ``space.ts`` -- ``spaceOf(pathname)`` (``workshop``, ``use``, or null
    for a page outside both: sign-in, registration, the component gallery),
    ``rememberRoute(last, route)`` (a new record, the route kept under its
    own space), and ``switchTarget(space, last, homes)`` (the space's last
    route, else its home).
  * ``legacy.ts`` -- ``legacyTarget(url, catalog)``: the page an old URL now
    means, its query kept, or null for a URL that is not an old one.
  * ``lib/settings/catalog.ts`` -- each group now says where it lives:
    ``space`` and ``section`` (a Preferences section, or a Workshop settings
    page), or ``retired`` with the reason, or ``embeddedIn`` the group whose
    panel renders it. ``PREFERENCES_SECTIONS`` lists the Preferences
    sections.
  * ``lib/settings/search.ts`` -- ``settingsIndex(groups, destinations,
    preferencesSections, formerSections)`` is every group of the catalog a
    search can find, each with the page that holds it (``href``, with its
    ``g``) and where that is (``where``); ``searchSettings(index, words)``
    is the groups whose title, description, synonyms, page and old section
    hold every word.
  * ``lib/chat/chatsIndex.ts`` -- ``indexParams(query, page)`` is the
    listing request of the chats index (a search sends ``q`` and the search
    limit, never an offset; the plain listing pages with ``limit`` and
    ``offset``), ``limitReached(query, count)`` says when a search filled
    its limit, ``SEARCH_LIMIT`` and ``PAGE_SIZE`` are the two sizes.

  * NV1 -- the sidebar and the route announcer (and the command palette,
    once there is one) import the table, and no other file declares a list
    of destination links.
  * NV2 -- every ready destination has a page, a destination that is not
    ready has none, and every page is a destination, a page inside one,
    sign-in, registration or the component gallery; the Workshop settings
    pages are served through the parameter matcher, which is the table's;
    nothing renders the table but through the visible list.
  * NV3 -- ``activeState`` marks the exact page active and a page under a
    destination as its section (a conversation makes the chats entry the
    section and its own recent row active), home never by prefix; the
    sidebar decides with it and with nothing of its own.
  * NV4 -- every old URL resolves to the page that holds what it held, its
    query kept; a group given by ``g`` is found wherever the catalog now
    places it (through its host when it is embedded), an unknown or retired
    one falls back to its old section; every old page's ``load`` redirects
    through it.
  * NV5 -- the old pages are ``+page.ts`` redirects in ``load``, with a
    permanent status (the root's to the chats index is temporary, until
    home exists); no page redirects from ``onMount``.
  * NV6 -- the route announcer names the visible destination a page belongs
    to, from the table, and holds no map of its own; the componion is
    dropped when its switch is off.
  * NV7 -- the catalog places each of its groups exactly once, as the
    placement table below says.
  * NV8 -- no link in the interface names an old URL; only ``legacy.ts``
    knows them.
  * NV9 -- in the chats index, rename and delete are visible buttons or a
    visible menu, never revealed by the pointer alone.
  * NV10 -- the chats index searches on the server (``q``, a limit of 200,
    and it says when the limit was reached) and pages the plain listing
    with ``offset`` (the search takes none on the server).
  * NV18 -- no tracked document names a "Settings >" path of the app; the
    repository host's own settings (branch protection) are excluded by
    name.
  * NV19 -- the settings search reads every group of both spaces, by its
    title, description and synonyms, every word of the search found, never
    a retired group; each result links to the page holding it (through its
    host when it is embedded); the settings hub searches with it and
    filters nothing itself.
  * NV20 -- a search of a page's name (Preferences and its sections, the
    Workshop and its pages) or of a section of the old settings page finds
    every group that sits there; the hub hands the search those sections.
  * NV21 -- the hub's search keeps its keys and its address: Enter opens
    the first result, Escape clears, the words go to ``?q=`` once the reader
    pauses, replacing the entry, and after every navigation the words are
    read back from the address.
  * NV22 -- every query a page answers, something in the interface
    produces (a link or a control), unless the page opens the same view by
    a control of its own; the old addresses' readers are apart.
  * NV23 -- every page of both spaces has a heading of level one, drawn by
    the page, by what it mounts, or by a layout above it.
  * NV24 -- no copy of the interface, no text the application writes for a
    developer and no document sends the reader to the old settings page
    ("go to Settings", "the Settings page", "Settings >").

Every census carries a standing positive fixture, a sample it must find,
so a probe gone blind turns red instead of reading a clean zero. The node
halves run the pure modules through ``tests/_frontend.run_ts``; each is
paired with a wiring half that reads the files using them. What a browser
does (the redirects firing, the shell not remounted) is owed to the
machine.

Local-only (the public distribution ships no tests). Needs Node >= 22.6;
without it the helpers raise, and so do the contracts.
"""

import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qsl, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import REPO, files, read, run_ts  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file.
BUDGET_S = {
    "test_nv1_one_destination_table_and_no_other_list": 1.0,
    "test_nv2_ready_destinations_have_pages_and_every_page_is_accounted_for": 2.0,
    "test_nv3_active_marks_the_exact_page_and_its_section[node]": 2.0,
    "test_nv3_active_marks_the_exact_page_and_its_section[wiring]": 1.0,
    "test_nv4_every_old_url_resolves_to_the_page_that_holds_it[map]": 2.0,
    "test_nv4_every_old_url_resolves_to_the_page_that_holds_it[placed]": 2.0,
    "test_nv4_every_old_url_resolves_to_the_page_that_holds_it[wiring]": 1.0,
    "test_nv5_old_pages_redirect_in_load": 1.0,
    "test_nv6_the_announcer_names_the_visible_destination_from_the_table[node]": 2.0,
    "test_nv6_the_announcer_names_the_visible_destination_from_the_table[wiring]": 1.0,
    "test_nv7_every_settings_group_is_placed_exactly_once": 2.0,
    "test_nv8_no_internal_link_names_an_old_url": 1.0,
    "test_nv9_the_chats_index_shows_rename_and_delete": 1.0,
    "test_nv10_the_chats_index_searches_on_the_server_and_pages_the_listing[search]": 2.0,
    "test_nv10_the_chats_index_searches_on_the_server_and_pages_the_listing[paging]": 2.0,
    "test_nv18_no_doc_names_a_settings_path": 1.0,
    "test_nv19_the_settings_search_reads_both_spaces_and_links_to_the_page[node]": 2.0,
    "test_nv19_the_settings_search_reads_both_spaces_and_links_to_the_page[wiring]": 1.0,
    "test_nv20_the_settings_search_finds_a_group_by_its_page_and_its_former_section[node]": 2.0,
    "test_nv20_the_settings_search_finds_a_group_by_its_page_and_its_former_section[wiring]": 1.0,
    "test_nv21_the_settings_search_keeps_its_keys_and_its_address[keys]": 1.0,
    "test_nv21_the_settings_search_keeps_its_keys_and_its_address[address]": 1.0,
    "test_nv22_every_query_a_page_answers_is_produced_by_the_interface": 1.0,
    "test_nv23_every_page_of_both_spaces_has_a_heading_of_level_one": 1.0,
    "test_nv24_no_copy_and_no_document_sends_the_reader_to_the_old_settings_page": 2.0,
}

_SRC = "frontend/src"
_ROUTES = f"{_SRC}/routes"
_NAV = f"{_SRC}/lib/nav"
_DESTINATIONS = f"{_NAV}/destinations.ts"
_ACTIVE = f"{_NAV}/active.ts"
_SPACE = f"{_NAV}/space.ts"
_LEGACY = f"{_NAV}/legacy.ts"
_MATCHER = f"{_SRC}/params/workshopSection.ts"
_CATALOG = f"{_SRC}/lib/settings/catalog.ts"
_CHATS_INDEX = f"{_SRC}/lib/chat/chatsIndex.ts"
_SETTINGS_SEARCH = f"{_SRC}/lib/settings/search.ts"
_HUB = f"{_SRC}/lib/components/settings/SettingsHub.svelte"
_CONVERSATIONS_API = f"{_SRC}/lib/api/conversations.ts"
_SIDEBAR = f"{_SRC}/lib/components/layout/Sidebar.svelte"
_ROOT_LAYOUT = f"{_ROUTES}/+layout.svelte"
_ROOT_PAGE = f"{_ROUTES}/+page.ts"
# The command palette, once it exists, reads the table too.
_PALETTE_FILES = (f"{_SRC}/lib/components/palette/", f"{_SRC}/lib/palette/")

_MODULES = {
    "OO_DESTINATIONS": _DESTINATIONS,
    "OO_ACTIVE": _ACTIVE,
    "OO_SPACE": _SPACE,
    "OO_LEGACY": _LEGACY,
    "OO_CATALOG": _CATALOG,
    "OO_CHATS_INDEX": _CHATS_INDEX,
    "OO_SETTINGS_SEARCH": _SETTINGS_SEARCH,
}

_SCRIPTS = (".svelte", ".ts", ".js")

# The pages the interface used to serve, each now a redirect in its load.
_OLD_PAGES = (
    "settings", "health", "benchmark", "verify", "claims", "verify-answer", "verify-citations",
)
# The pages outside both spaces: sign-in, registration, the component gallery.
_OUTSIDE_PAGES = ("/login", "/register", "/dev/components")


# ---------------------------------------------------------------------------
# The Node driver: it calls the pure modules and prints what they return.
# ---------------------------------------------------------------------------
_DRIVER = r"""
const load = async (name) => (process.env[name] ? await import(process.env[name]) : null);
const nav = await load('OO_DESTINATIONS');
const active = await load('OO_ACTIVE');
const space = await load('OO_SPACE');
const legacy = await load('OO_LEGACY');
const catalog = await load('OO_CATALOG');
const chats = await load('OO_CHATS_INDEX');
const search = await load('OO_SETTINGS_SEARCH');
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT || 'null');

const readyHrefs = (list) => list.filter((d) => d.ready).map((d) => d.href);
const idOf = (d) => (d ? d.id : null);

const run = {
    table: () => ({
        destinations: nav.DESTINATIONS,
        sections: input.map((param) => nav.isWorkshopSection(param)),
        homes: {
            use: nav.spaceHome('use', nav.DESTINATIONS),
            workshop: nav.spaceHome('workshop', nav.DESTINATIONS),
        },
    }),
    visible: () => {
        const tables = { real: nav.DESTINATIONS, ...input.tables };
        const out = {};
        for (const [name, list] of Object.entries(tables)) {
            out[name] = {};
            for (const componion of [true, false]) {
                const shown = nav.visibleDestinations(list, { componion });
                out[name][componion ? 'on' : 'off'] = {
                    ids: shown.map(idOf),
                    named: input.paths.map((path) => idOf(active.destinationFor(path, shown))),
                };
            }
        }
        return out;
    },
    active: () => {
        const hrefs = readyHrefs(nav.DESTINATIONS);
        return {
            hrefs,
            states: input.map(([pathname, href]) => active.activeState(pathname, href, hrefs)),
        };
    },
    space: () => {
        const homes = {
            use: nav.spaceHome('use', nav.DESTINATIONS),
            workshop: nav.spaceHome('workshop', nav.DESTINATIONS),
        };
        const spaces = input.paths.map((path) => space.spaceOf(path));
        const out = { homes, spaces, steps: [] };
        let last = {};
        for (const step of input.steps) {
            if (step[0] === 'remember') {
                const before = JSON.stringify(last);
                const next = space.rememberRoute(last, step[1]);
                out.steps.push({ remember: step[1], kept: JSON.stringify(last) === before, last: next });
                last = next;
            } else if (step[0] === 'switch') {
                out.steps.push({ switch: step[1], to: space.switchTarget(step[1], last, homes) });
            } else if (step[0] === 'stored') {
                out.steps.push({
                    stored: step[1],
                    to: space.switchTarget(step[1], step[2], homes),
                });
            }
        }
        return out;
    },
    legacy: () => {
        const source = input.catalog === null ? catalog : input.catalog;
        return input.urls.map((url) => legacy.legacyTarget(new URL(url, 'http://localhost'), source));
    },
    catalog: () => ({
        sections: catalog.SETTINGS_SECTIONS,
        inline: catalog.INLINE_GROUPS,
        preferences: catalog.PREFERENCES_SECTIONS,
        destinations: nav.DESTINATIONS,
        workshop: input.map((param) => nav.isWorkshopSection(param)),
    }),
    chats: () => ({
        search: chats.SEARCH_LIMIT,
        page: chats.PAGE_SIZE,
        params: input.params.map(([query, page]) => chats.indexParams(query, page)),
        reached: input.reached.map(([query, count]) => chats.limitReached(query, count)),
    }),
    pages: () => {
        const groups = [...catalog.SETTINGS_SECTIONS.flatMap((s) => s.groups), ...catalog.INLINE_GROUPS];
        const index = search.settingsIndex(
            groups, nav.DESTINATIONS, catalog.PREFERENCES_SECTIONS, catalog.SETTINGS_SECTIONS,
        );
        return {
            ids: index.map((hit) => hit.id),
            found: input.queries.map((words) => search.searchSettings(index, words).map((hit) => hit.id)),
        };
    },
    settings: () => {
        const source = input.catalog === null ? catalog : input.catalog;
        const groups = [...source.SETTINGS_SECTIONS.flatMap((s) => s.groups), ...source.INLINE_GROUPS];
        const index = search.settingsIndex(groups, nav.DESTINATIONS, catalog.PREFERENCES_SECTIONS);
        return {
            index: index.map((hit) => ({ id: hit.id, href: hit.href, where: hit.where })),
            found: input.queries.map((words) => search.searchSettings(index, words).map((hit) => hit.id)),
        };
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
# Reading sources: markup, scripts, imports
# ---------------------------------------------------------------------------
_SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>(.*?)</script>", re.S)
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>.*?</style>", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_JS_COMMENT = re.compile(r"/\*.*?\*/|(?<![:\w'\"`])//[^\n]*", re.S)


def _script(path, text=None):
    """A component's script blocks, or a module's text, comments removed."""
    text = read(path) if text is None else text
    body = "\n".join(_SCRIPT_BLOCK.findall(text)) if path.endswith(".svelte") else text
    return _JS_COMMENT.sub(" ", body)


def _markup(path, text=None):
    """A component's markup: no script, no style, no comment."""
    text = read(path) if text is None else text
    return _HTML_COMMENT.sub(" ", _STYLE_BLOCK.sub(" ", _SCRIPT_BLOCK.sub(" ", text)))


def _code(path, text=None):
    """Everything a file runs or renders: script and markup, comments out."""
    text = read(path) if text is None else text
    if not path.endswith(".svelte"):
        return _JS_COMMENT.sub(" ", text)
    return _script(path, text) + "\n" + _markup(path, text)


_IMPORT = re.compile(
    r"""(?:\bimport\s+(?:type\s+)?(?:[\w$*{}\s,]+?\s+from\s+)?|\bimport\s*\(\s*|\bexport\s+[\w$*{}\s,]+?\s+from\s+)(['"])([^'"]+)\1"""
)


def _resolve(importer, spec):
    """The repository path an import names, or None when it is a package."""
    if spec.startswith("$lib/"):
        base = PurePosixPath(_SRC, "lib", spec[len("$lib/"):])
    elif spec.startswith("."):
        base = PurePosixPath(importer).parent / spec
        parts = []
        for part in base.parts:
            if part == "..":
                parts.pop()
            elif part != ".":
                parts.append(part)
        base = PurePosixPath(*parts)
    else:
        return None
    for candidate in (str(base), f"{base}.ts", f"{base}.js", f"{base}/index.ts"):
        if (REPO / candidate).is_file():
            return candidate
    return str(base)


def _imports(path, text=None):
    """The repository paths a file imports (packages left out)."""
    found = []
    for match in _IMPORT.finditer(_script(path, text)):
        resolved = _resolve(path, match.group(2))
        if resolved is not None:
            found.append(resolved)
    return found


def _imports_name(path, name, module, text=None):
    """Whether ``path`` imports ``name`` (a value, not a type) from
    ``module`` (a repository path)."""
    script = _script(path, text)
    for match in re.finditer(
        r"\bimport\s+(?!type\b)\{([^}]*)\}\s*from\s*(['\"])([^'\"]+)\2", script,
    ):
        if _resolve(path, match.group(3)) != module:
            continue
        names = [part.strip().split(" as ")[0].strip() for part in match.group(1).split(",")]
        if name in names:
            return True
    return False


def _calls(text, name):
    return re.search(rf"(?<![\w$.]){re.escape(name)}\s*\(", text) is not None


# ---------------------------------------------------------------------------
# The route census: every page file and the URL it serves
# ---------------------------------------------------------------------------
def _route_url(path):
    """The URL a route file serves: its directory under the routes, groups
    dropped, parameters kept as written."""
    parts = PurePosixPath(path).relative_to(_ROUTES).parts[:-1]
    kept = [part for part in parts if not (part.startswith("(") and part.endswith(")"))]
    return "/" + "/".join(kept)


def _route_groups(path):
    return [
        part for part in PurePosixPath(path).relative_to(_ROUTES).parts[:-1]
        if part.startswith("(") and part.endswith(")")
    ]


def _page_files():
    """``{path: url}`` of every page file under the routes: the page
    components and the page modules (``+page.ts``) alike."""
    return {
        path: _route_url(path)
        for path in files((".svelte", ".ts", ".js"), within=_ROUTES)
        if PurePosixPath(path).name in ("+page.svelte", "+page.ts", "+page.js")
    }


def _page_components():
    """``{path: url}`` of the pages that render: a ``+page.svelte``."""
    return {path: url for path, url in _page_files().items() if path.endswith("+page.svelte")}


def _redirect_only():
    """``{path: url}`` of the page modules with no page component beside
    them: a route that only redirects."""
    rendered = {str(PurePosixPath(path).parent) for path in _page_components()}
    return {
        path: url for path, url in _page_files().items()
        if not path.endswith("+page.svelte") and str(PurePosixPath(path).parent) not in rendered
    }


def _page_for(url):
    found = [path for path, served in _page_components().items() if served == url]
    assert len(found) == 1, f"one page component serves {url}: {found}"
    return found[0]


_PARAM = re.compile(r"^\[(\w+)(?:=(\w+))?\]$")


def _serves(route_url, href, accepted):
    """Whether a page route URL serves ``href``: equal, or a parameter
    segment whose matcher accepts the segment (``accepted`` maps a matcher
    name to the set it accepts; a bare parameter accepts anything)."""
    route = route_url.strip("/").split("/") if route_url != "/" else []
    target = href.strip("/").split("/") if href != "/" else []
    if len(route) != len(target):
        return False
    for part, want in zip(route, target):
        match = _PARAM.match(part)
        if match is None:
            if part != want:
                return False
        elif match.group(2) is not None and want not in accepted.get(match.group(2), ()):
            return False
    return True


# ---------------------------------------------------------------------------
# NV1 -- one destination table
# ---------------------------------------------------------------------------
# A destination-like path: the pages the interface serves or served.
_DESTINATION_PATH = (
    r"/(?:chat|projects|notes|verify|settings|benchmark|health|workshop|preferences|garden"
    r"|claims|verify-answer|verify-citations|login|register)(?:/[\w-]*)*"
)
_LIST_ENTRY = (
    re.compile(r"""\bhref\s*:\s*(['"`])(%s)(?=[?#'"`])""" % _DESTINATION_PATH),
    re.compile(r"""(['"`])(%s)\1\s*:""" % _DESTINATION_PATH),
    re.compile(r"""\bstartsWith\(\s*(['"`])(%s)\1""" % _DESTINATION_PATH),
)


def _destination_list(path, text):
    """How many distinct destination paths a file declares as list entries:
    an ``href:`` value, an object key, or a ``startsWith`` argument. Three
    or more is a list of destinations."""
    code = _code(path, text)
    found = set()
    for pattern in _LIST_ENTRY:
        found |= {match.group(2) for match in pattern.finditer(code)}
    return len(found)


_LIST_SAMPLE = (
    f"{_SRC}/lib/sample/Links.svelte",
    "<script lang=\"ts\">\n"
    "\tconst links = [\n"
    "\t\t{ href: '/chat', label: 'Chat' },\n"
    "\t\t{ href: '/projects', label: 'Projects' },\n"
    "\t];\n"
    "\tconst names = { '/settings': 'Settings' };\n"
    "\t$: here = path.startsWith('/health');\n"
    "</script>\n",
)


def test_nv1_one_destination_table_and_no_other_list():
    assert _destination_list(*_LIST_SAMPLE) == 4, (
        "the census reads href values, object keys and startsWith arguments"
    )
    assert _destination_list(f"{_SRC}/lib/sample/One.ts", "go('/chat'); x = { href: '/notes' };") == 1

    lists = {
        path: count
        for path in files(_SCRIPTS, exclude=(_DESTINATIONS, _LEGACY))
        if (count := _destination_list(path, read(path))) >= 3
    }
    assert not lists, (
        f"lists of destinations declared outside the table (distinct paths each): {lists}"
    )

    readers = [_SIDEBAR, _ROOT_LAYOUT]
    for entry in _PALETTE_FILES:
        if (REPO / entry).is_dir():
            readers += files(_SCRIPTS, within=entry.rstrip("/"))
    missing = [path for path in readers if _DESTINATIONS not in _imports(path)]
    assert not missing, f"these read their destinations from the table: {missing}"


# ---------------------------------------------------------------------------
# NV2 -- ready destinations have pages, and every page is accounted for
# ---------------------------------------------------------------------------
_PROBE_SECTIONS = ("verify", "benchmarks", "nowhere", "", "Models")


def _table():
    return _node("table", ("OO_DESTINATIONS",), [])


def _workshop_slugs(destinations):
    return [
        d["href"][len("/workshop/"):]
        for d in destinations
        if d["space"] == "workshop" and d.get("settings") and d["href"].startswith("/workshop/")
    ]


_EACH_TABLE = re.compile(r"\{#each\s+DESTINATIONS\b")


def test_nv2_ready_destinations_have_pages_and_every_page_is_accounted_for():
    table = _table()["destinations"]
    assert len(table) >= 10, f"the table lists the destinations of both spaces: {len(table)}"
    ids = [d["id"] for d in table]
    assert len(ids) == len(set(ids)), f"every destination id is unique: {ids}"
    for d in table:
        assert d["space"] in ("use", "workshop"), f"{d['id']} belongs to one of the two spaces"
        assert d["label"] and d["icon"] and isinstance(d["ready"], bool), (
            f"{d['id']} has a label, an icon and says whether it is ready"
        )
        assert d["href"].startswith("/") and not d["href"].startswith("//"), d
        assert (d["space"] == "workshop") == (
            d["href"] == "/workshop" or d["href"].startswith("/workshop/")
        ), f"{d['id']}: a Workshop destination lives under /workshop, a Use one does not"

    slugs = _workshop_slugs(table)
    assert slugs, "the table names the Workshop settings pages"
    probe = _node("table", ("OO_DESTINATIONS",), [*slugs, *_PROBE_SECTIONS])
    accepted = dict(zip([*slugs, *_PROBE_SECTIONS], probe["sections"]))
    assert all(accepted[slug] for slug in slugs), (
        f"the matcher accepts every Workshop settings page: {accepted}"
    )
    assert not any(accepted[other] for other in _PROBE_SECTIONS), (
        f"and nothing else: {accepted}"
    )

    matcher = _script(_MATCHER)
    assert _imports_name(_MATCHER, "isWorkshopSection", _DESTINATIONS), (
        "the parameter matcher is the table's isWorkshopSection"
    )
    assert re.search(r"\bexport\s+(?:const|function)\s+match\b", matcher), (
        "the matcher exports match, as the router reads it"
    )
    assert not re.search(r"['\"`]\w[\w-]*['\"`]\s*,\s*['\"`]\w", matcher), (
        "the matcher holds no list of section names of its own"
    )

    matchers = {"workshopSection": {slug for slug in slugs if accepted[slug]}}
    pages = _page_components()
    redirects = _redirect_only()
    problems = []
    for d in table:
        serving = [path for path, url in pages.items() if _serves(url, d["href"], matchers)]
        if d["ready"] and not serving:
            problems.append(f"{d['id']} is ready and no page serves {d['href']}")
        if not d["ready"] and serving:
            problems.append(f"{d['id']} is not ready and a page serves {d['href']}: {serving}")

    ready = [d["href"] for d in table if d["ready"]]
    for path, url in pages.items():
        if url in ready or url in _OUTSIDE_PAGES:
            continue
        params = [part for part in url.split("/") if _PARAM.match(part)]
        if params:
            match = _PARAM.match(params[0])
            if match.group(2):
                served = {f"/workshop/{slug}" for slug in matchers.get(match.group(2), ())}
                if url.startswith("/workshop/") and served and served <= set(ready):
                    continue
            else:
                head = url[: url.index("/" + params[0])]
                if head in ready:
                    continue
        problems.append(f"{path} serves {url}, which is no destination and no page inside one")
    for path, url in redirects.items():
        if url in ready:
            problems.append(f"{path} redirects {url}, a ready destination's own page")
    assert not problems, "\n".join(problems)

    assert _EACH_TABLE.search(_markup("sample.svelte", "<ul>{#each DESTINATIONS as d}<li>{d.label}</li>{/each}</ul>")), (
        "the census reads the table rendered whole"
    )
    renderers = [
        path for path in files(".svelte")
        if _EACH_TABLE.search(_markup(path))
    ]
    assert not renderers, (
        f"these render the table without the visible list (a destination not ready, or the "
        f"componion switched off, would show): {renderers}"
    )
    assert _imports_name(_SIDEBAR, "visibleDestinations", _DESTINATIONS), (
        "the sidebar renders the visible destinations"
    )


# ---------------------------------------------------------------------------
# NV3 -- the current page and its section
# ---------------------------------------------------------------------------
_ACTIVE_CASES = (
    (("/chat", "/chat"), "active"),
    (("/chat/", "/chat"), "active"),
    (("/chat/abc", "/chat"), "section"),
    (("/chat/abc", "/chat/abc"), "active"),
    (("/chat/abc", "/chat/xyz"), "none"),
    (("/chat", "/chat/abc"), "none"),
    (("/chatter", "/chat"), "none"),
    (("/notes", "/chat"), "none"),
    (("/projects/p1", "/projects"), "section"),
    (("/projects", "/projects"), "active"),
    (("/", "/"), "active"),
    (("/chat", "/"), "none"),
    (("/nowhere", "/"), "none"),
    (("/workshop", "/workshop"), "active"),
    (("/workshop/models", "/workshop"), "none"),
    (("/workshop/verify", "/workshop"), "none"),
    (("/workshop/models", "/workshop/models"), "active"),
    (("/workshop/models", "/workshop/security"), "none"),
    (("/preferences", "/preferences"), "active"),
)


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_nv3_active_marks_the_exact_page_and_its_section(half):
    if half == "wiring":
        sidebar = _code(_SIDEBAR)
        assert _imports_name(_SIDEBAR, "activeState", _ACTIVE) and _calls(sidebar, "activeState"), (
            "the sidebar decides the current entry with activeState"
        )
        assert "startsWith(" not in sidebar, (
            "the sidebar holds no prefix test of its own"
        )
        currents = re.findall(r"aria-current\s*=\s*\{([^}]*)\}", _markup(_SIDEBAR))
        assert currents and all("active" in expression for expression in currents), (
            f"the sidebar marks the current page from the state activeState returns: {currents}"
        )
        return

    result = _node("active", ("OO_DESTINATIONS", "OO_ACTIVE"), [list(case) for case, _ in _ACTIVE_CASES])
    got = dict(zip([case for case, _ in _ACTIVE_CASES], result["states"]))
    wrong = {
        f"{path} for {href}": f"{got[(path, href)]} (expected {want})"
        for (path, href), want in _ACTIVE_CASES if got[(path, href)] != want
    }
    assert not wrong, f"activeState(pathname, href, the ready hrefs): {wrong}"
    assert "/workshop/models" in result["hrefs"] and "/workshop" in result["hrefs"], (
        f"the cases run against the table's own ready hrefs: {result['hrefs']}"
    )


# ---------------------------------------------------------------------------
# NV4 -- every old URL resolves
# ---------------------------------------------------------------------------
# The old URLs and the pages that now hold what they held.
_LEGACY_MAP = (
    ("/health", "/workshop"),
    ("/benchmark", "/workshop/benchmarks"),
    ("/benchmark?run=abc", "/workshop/benchmarks?run=abc"),
    ("/verify", "/workshop/verify"),
    ("/verify?mode=cited", "/workshop/verify?mode=cited"),
    ("/verify?mode=pairs", "/workshop/verify?mode=pairs"),
    ("/claims", "/workshop/verify?mode=pairs"),
    ("/verify-answer", "/workshop/verify?mode=pairs"),
    ("/verify-citations", "/workshop/verify?mode=cited"),
    ("/settings", "/preferences"),
    ("/settings/", "/preferences"),
    ("/settings?q=cache", "/preferences?q=cache"),
    ("/settings?section=models&q=vram", "/workshop/models?q=vram"),
    ("/settings?section=appearance", "/preferences"),
    ("/settings?section=account", "/workshop/security"),
    ("/settings?tab=security", "/workshop/security"),
    ("/settings?section=conversation", "/preferences?g=task-presets"),
    ("/settings?tab=presets", "/preferences?g=task-presets"),
    ("/settings?tab=quick", "/workshop/models?g=conversation-system-preset"),
    ("/settings?tab=prompt", "/workshop/models?g=prompt-config"),
    ("/settings?section=models", "/workshop/models"),
    ("/settings?tab=models", "/workshop/models"),
    ("/settings?section=knowledge", "/workshop/knowledge"),
    ("/settings?tab=knowledge", "/workshop/knowledge"),
    ("/settings?section=plugins", "/workshop/extensions"),
    ("/settings?tab=plugins", "/workshop/extensions"),
    ("/settings?section=performance", "/workshop/observability"),
    ("/settings?tab=performance", "/workshop/observability"),
    ("/settings?tab=advanced", "/workshop/observability"),
    ("/settings?tab=analytics", "/workshop/observability?g=analytics"),
    ("/settings?section=network", "/workshop/network"),
    ("/settings?section=data", "/workshop/backup"),
    ("/settings?tab=backup", "/workshop/backup?g=backup-restore"),
    ("/settings?tab=fine-tune", "/workshop/backup?g=fine-tune"),
    ("/settings?section=nowhere", "/preferences"),
    ("/settings?tab=nowhere", "/preferences"),
    ("/settings?section=network&g=device-sync", "/workshop/network?g=device-sync"),
    ("/settings?tab=security&g=totp", "/preferences?g=totp"),
    ("/settings?section=models&g=skills", "/workshop/extensions?g=skills"),
    ("/settings?g=memories", "/preferences?g=memories"),
    ("/settings?section=conversation&g=conversation-defaults", "/preferences?g=task-presets"),
    ("/settings?section=models&g=nowhere", "/workshop/models"),
    ("/settings?g=%2F%2Fevil.example", "/preferences"),
)
# Not an old URL: legacyTarget answers null.
_NOT_OLD = ("/chat", "/chat/abc", "/workshop", "/preferences", "/settingsx", "/")


def _parts(url):
    """A same-origin URL as (path, sorted query pairs)."""
    split = urlsplit(url)
    return split.path, sorted(parse_qsl(split.query, keep_blank_values=True))


def _legacy(urls, catalog=None):
    return _node(
        "legacy", ("OO_LEGACY", "OO_CATALOG"), {"urls": list(urls), "catalog": catalog},
    )


def _catalog():
    return _node("catalog", ("OO_CATALOG", "OO_DESTINATIONS"), [])


def _groups(catalog):
    """Every group of the catalog: the panels' and the introductions' own,
    each with the old section it was listed under."""
    out = []
    for section in catalog["sections"]:
        for group in section["groups"]:
            out.append({**group, "oldSection": section["id"]})
    for group in catalog["inline"]:
        out.append({**group, "oldSection": group["sectionId"]})
    return out


# A catalog of the same shape with one embedded group, one retired, one of
# each space: the rules for a group the real catalog does not embed yet.
_PROBE_CATALOG = {
    "SETTINGS_SECTIONS": [
        {"id": "models", "label": "Models", "icon": "cpu", "description": "d", "groups": [
            {"id": "probe-host", "title": "t", "description": "d", "panel": "PanelA",
             "space": "workshop", "section": "observability"},
            {"id": "probe-child", "title": "t", "description": "d", "panel": "PanelB",
             "embeddedIn": "probe-host"},
            {"id": "probe-gone", "title": "t", "description": "d", "panel": "PanelC",
             "retired": "no reader"},
        ]},
        {"id": "appearance", "label": "Appearance", "icon": "palette", "description": "d", "groups": [
            {"id": "probe-mine", "title": "t", "description": "d", "panel": "PanelD",
             "space": "use", "section": "chats"},
        ]},
    ],
    "INLINE_GROUPS": [
        {"sectionId": "appearance", "id": "probe-inline", "title": "t", "description": "d",
         "synonyms": [], "space": "workshop", "section": "backup"},
    ],
    "LEGACY_TAB_TO_SECTION": {"security": "account", "advanced": "performance"},
}
_PROBE_EXPECTED = (
    ("/settings?g=probe-host", "/workshop/observability?g=probe-host"),
    ("/settings?section=models&g=probe-child", "/workshop/observability?g=probe-host"),
    ("/settings?g=probe-child", "/workshop/observability?g=probe-host"),
    ("/settings?section=models&g=probe-gone", "/workshop/models"),
    ("/settings?g=probe-gone", "/preferences"),
    ("/settings?section=appearance&g=probe-mine", "/preferences?g=probe-mine"),
    ("/settings?g=probe-inline&q=x", "/workshop/backup?g=probe-inline&q=x"),
)

# The pages each old section's own URL leads to (the fallback of a group
# that no longer has a page of its own).
_OLD_SECTION_PAGE = {
    "appearance": "/preferences",
    "account": "/workshop/security",
    "conversation": "/preferences?g=task-presets",
    "models": "/workshop/models",
    "knowledge": "/workshop/knowledge",
    "plugins": "/workshop/extensions",
    "performance": "/workshop/observability",
    "network": "/workshop/network",
    "data": "/workshop/backup",
}


def _placed_page(group, by_id):
    """The page and the ``g`` a group is found at, or None when it has no
    page (retired)."""
    if group.get("embeddedIn"):
        host = by_id[group["embeddedIn"]]
        return _placed_page(host, by_id)[0], host["id"]
    if group.get("retired"):
        return None
    if group.get("space") == "use":
        return "/preferences", group["id"]
    return f"/workshop/{group['section']}", group["id"]


def _value_import_lines(script):
    """The import lines of a module that bring values, not types."""
    imports = re.findall(r"^\s*import\b[^\n]*", script, re.M)
    return [line for line in imports if not re.match(r"\s*import\s+type\b", line)]


@pytest.mark.parametrize("half", ("map", "placed", "wiring"))
def test_nv4_every_old_url_resolves_to_the_page_that_holds_it(half):
    if half == "wiring":
        problems = []
        for name in _OLD_PAGES:
            page = f"{_ROUTES}/{name}/+page.ts"
            if not (REPO / page).is_file():
                problems.append(f"{page} is absent")
                continue
            code = _code(page)
            if not (_imports_name(page, "legacyTarget", _LEGACY) and re.search(r"legacyTarget\(\s*url\b", code)):
                problems.append(f"{page} does not ask legacyTarget where its URL now goes")
            if _CATALOG not in _imports(page):
                problems.append(f"{page} does not hand legacyTarget the catalog")
            if re.search(r"""['"`]/workshop""", code):
                problems.append(f"{page} names a new URL of its own")
        assert _value_import_lines("import { a } from './x';\nimport type { B } from './y';\n") == [
            "import { a } from './x';"
        ], "the census reads a value import, and not a type import"
        values = _value_import_lines(_script(_LEGACY))
        if values:
            problems.append(f"legacy.ts imports values (the catalog is handed to it): {values}")
        assert not problems, "\n".join(problems)
        return

    if half == "map":
        urls = [old for old, _ in _LEGACY_MAP] + list(_NOT_OLD)
        got = dict(zip(urls, _legacy(urls)))
        wrong = {
            old: f"{got[old]} (expected {new})"
            for old, new in _LEGACY_MAP
            if got[old] is None or _parts(got[old]) != _parts(new)
        }
        assert not wrong, f"old URLs that do not land where their content went: {wrong}"
        for old, _ in _LEGACY_MAP:
            assert got[old].startswith("/") and not got[old].startswith("//"), (
                f"{old} goes to a path of this origin: {got[old]}"
            )
        answered = {url: got[url] for url in _NOT_OLD if got[url] is not None}
        assert not answered, f"a URL that is not an old one is left alone: {answered}"
        return

    catalog = _catalog()
    groups = _groups(catalog)
    assert len(groups) >= 51, f"the census reads every group of the catalog: {len(groups)}"
    by_id = {group["id"]: group for group in groups}
    cases = []
    for group in groups:
        placed = _placed_page(group, by_id)
        fallback = _OLD_SECTION_PAGE[group["oldSection"]]
        if placed is None:
            cases.append((f"/settings?section={group['oldSection']}&g={group['id']}", fallback))
            cases.append((f"/settings?g={group['id']}", "/preferences"))
        else:
            page, g = placed
            cases.append((f"/settings?section={group['oldSection']}&g={group['id']}", f"{page}?g={g}"))
            cases.append((f"/settings?g={group['id']}", f"{page}?g={g}"))
    got = _legacy([old for old, _ in cases])
    wrong = {
        old: f"{found} (expected {new})"
        for (old, new), found in zip(cases, got)
        if found is None or _parts(found) != _parts(new)
    }
    assert not wrong, f"groups an old link names that do not land on the page holding them: {wrong}"

    probed = _legacy([old for old, _ in _PROBE_EXPECTED], _PROBE_CATALOG)
    wrong = {
        old: f"{found} (expected {new})"
        for (old, new), found in zip(_PROBE_EXPECTED, probed)
        if found is None or _parts(found) != _parts(new)
    }
    assert not wrong, (
        f"an embedded group goes to its host, a retired or unknown one to its old section: {wrong}"
    )


# ---------------------------------------------------------------------------
# NV5 -- old pages redirect in load
# ---------------------------------------------------------------------------
_ONMOUNT_REDIRECT = re.compile(r"onMount\s*\([^)]*?\)?\s*=>\s*\{?[^}]*?\bgoto\s*\(", re.S)


def _mount_redirects(path, text):
    """A page that sends its reader elsewhere from onMount."""
    return len(_ONMOUNT_REDIRECT.findall(_script(path, text)))


_MOUNT_SAMPLE = (
    f"{_ROUTES}/sample/+page.svelte",
    "<script lang=\"ts\">\n"
    "\timport { onMount } from 'svelte';\n"
    "\timport { goto } from '$app/navigation';\n"
    "\tonMount(() => {\n"
    "\t\tgoto('/somewhere', { replaceState: true });\n"
    "\t});\n"
    "</script>\n",
)


def test_nv5_old_pages_redirect_in_load():
    assert _mount_redirects(*_MOUNT_SAMPLE) == 1, "the census reads a redirect from onMount"

    problems = []
    for name in _OLD_PAGES:
        directory = REPO / _ROUTES / name
        present = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
        if present != ["+page.ts"]:
            problems.append(f"routes/{name} holds {present}, not its redirect alone")
            continue
        code = _code(f"{_ROUTES}/{name}/+page.ts")
        if not re.search(r"\bredirect\(\s*308\s*,", code):
            problems.append(f"routes/{name}/+page.ts does not redirect permanently (308) in load")
        if not re.search(r"\bfunction\s+load\b|\bconst\s+load\b|\bexport\s+(?:const|function)\s+load\b", code):
            problems.append(f"routes/{name}/+page.ts has no load")

    root = sorted(p.name for p in (REPO / _ROUTES).iterdir() if p.name.startswith("+page"))
    if root != ["+page.ts"]:
        problems.append(f"the root holds {root}, not its redirect alone")
    else:
        code = _code(_ROOT_PAGE)
        if not re.search(r"""\bredirect\(\s*30[237]\s*,\s*['"`]/chat['"`]""", code):
            problems.append(
                "the root does not send its reader to the chats index with a temporary status "
                "(home comes later, and a permanent redirect would outlive it)"
            )
    mounted = {
        path: count for path in _page_components()
        if (count := _mount_redirects(path, read(path)))
    }
    if mounted:
        problems.append(f"pages that redirect from onMount: {mounted}")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# NV6 -- the announcer names the visible destination from the table
# ---------------------------------------------------------------------------
def _probe_table():
    """A table of the same shape where the componion is ready, so its
    switch can be seen to hide it."""
    return [
        {"id": "home", "label": "Home", "href": "/", "space": "use", "icon": "home", "ready": True},
        {"id": "chats", "label": "Chats", "href": "/chat", "space": "use", "icon": "chat", "ready": True},
        {"id": "componion", "label": "Componion", "href": "/garden", "space": "use",
         "icon": "sprout", "ready": True},
        {"id": "later", "label": "Later", "href": "/later", "space": "use", "icon": "note",
         "ready": False},
        {"id": "status", "label": "System status", "href": "/workshop", "space": "workshop",
         "icon": "info", "ready": True},
    ]


_ANNOUNCED = ("/chat", "/chat/abc", "/garden", "/later", "/workshop/models", "/nowhere", "/")


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_nv6_the_announcer_names_the_visible_destination_from_the_table(half):
    if half == "wiring":
        holders = [
            path for path in files(_SCRIPTS)
            if re.search(r"""\bid\s*=\s*['"]oo-route-announcer['"]""", _markup(path) if path.endswith(".svelte") else "")
        ]
        assert holders == [_ROOT_LAYOUT], (
            f"one announcer, in the root layout: {holders}"
        )
        writers = [
            path for path in files(_SCRIPTS)
            if "oo-route-announcer" in _script(path) and path != _ROOT_LAYOUT
        ]
        assert not writers, f"only the root layout writes the announcer: {writers}"
        code = _code(_ROOT_LAYOUT)
        for name, module in (("visibleDestinations", _DESTINATIONS), ("destinationFor", _ACTIVE)):
            assert _imports_name(_ROOT_LAYOUT, name, module) and _calls(code, name), (
                f"the announcer names a page through {name}"
            )
        assert _destination_list(_ROOT_LAYOUT, read(_ROOT_LAYOUT)) == 0, (
            "the root layout holds no map of routes of its own"
        )
        return

    result = _node(
        "visible", ("OO_DESTINATIONS", "OO_ACTIVE"),
        {"tables": {"probe": _probe_table()}, "paths": list(_ANNOUNCED)},
    )
    real = _table()["destinations"]
    shown = [d for d in real if d["ready"]]
    for switch in ("on", "off"):
        ids = result["real"][switch]["ids"]
        assert ids == [d["id"] for d in shown], (
            f"the visible destinations are the ready ones, in the table's order ({switch}): {ids}"
        )
    probe = result["probe"]
    assert probe["on"]["ids"] == ["home", "chats", "componion", "status"], probe["on"]["ids"]
    assert probe["off"]["ids"] == ["home", "chats", "status"], (
        f"the componion is dropped while its switch is off: {probe['off']['ids']}"
    )
    named_on = dict(zip(_ANNOUNCED, probe["on"]["named"]))
    named_off = dict(zip(_ANNOUNCED, probe["off"]["named"]))
    assert named_on == {
        "/chat": "chats", "/chat/abc": "chats", "/garden": "componion", "/later": None,
        "/workshop/models": "status", "/nowhere": None, "/": "home",
    }, f"a page is announced by the destination it belongs to: {named_on}"
    assert named_off["/garden"] is None and named_off["/chat/abc"] == "chats", (
        f"a hidden destination is never announced: {named_off}"
    )
    real_named = dict(zip(_ANNOUNCED, result["real"]["on"]["named"]))
    assert real_named["/chat/abc"] == "chats" and real_named["/nowhere"] is None, real_named


# ---------------------------------------------------------------------------
# NV7 -- every settings group is placed exactly once
# ---------------------------------------------------------------------------
# Where each group of the catalog lives: a Preferences section, a Workshop
# settings page, or retired.
_PLACEMENT = {
    **{g: ("use", "appearance") for g in (
        "appearance-theme", "appearance-density", "appearance-typography", "appearance-motion")},
    "appearance-advanced": ("use", "keyboard"),
    "conversation-system-preset": ("workshop", "models"),
    "conversation-defaults": ("retired", None),
    "conversation-config-maintenance": ("workshop", "backup"),
    "account-auth-mode": ("workshop", "security"),
    "security-mode": ("workshop", "security"),
    **{g: ("use", "account") for g in ("totp", "webauthn", "recovery-codes", "app-passwords")},
    **{g: ("workshop", "security") for g in ("hardening", "key-ceremony", "audit-chain")},
    "task-presets": ("use", "chats"),
    "memories": ("use", "memory"),
    **{g: ("workshop", "models") for g in (
        "prompt-config", "compression", "context-optimizer", "output-humanizer",
        "model-health", "model-profiles", "model-assignment", "routing", "cascading",
        "speculative", "vision", "resource-governor", "performance-tuner")},
    **{g: ("workshop", "knowledge") for g in ("knowledge-base", "rag-dashboard")},
    **{g: ("workshop", "extensions") for g in (
        "installed-plugins", "plugin-marketplace", "plugin-allowlist", "skills")},
    **{g: ("workshop", "observability") for g in (
        "cache", "observability", "analytics",
        "telemetry", "telemetry-history", "profiler", "performance-dashboard")},
    **{g: ("workshop", "network") for g in (
        "proxy", "remote-access", "search-kill-switch", "device-sync")},
    **{g: ("workshop", "backup") for g in ("backup-restore", "fine-tune")},
}
# These four may instead be embedded in the observability group, whose
# panel renders them.
_EMBEDDABLE = {"telemetry", "telemetry-history", "profiler", "performance-dashboard"}
_PREFERENCES_SECTIONS = ("appearance", "keyboard", "account", "chats", "memory")
_KINDS = ("space", "retired", "embeddedIn")


def test_nv7_every_settings_group_is_placed_exactly_once():
    catalog = _catalog()
    groups = _groups(catalog)
    ids = [group["id"] for group in groups]
    assert len(ids) == len(set(ids)) == 51, f"the catalog holds its 51 groups once each: {len(ids)}"
    assert set(ids) == set(_PLACEMENT), (
        f"the placement table and the catalog name the same groups: "
        f"unplaced {sorted(set(ids) - set(_PLACEMENT))}, unknown {sorted(set(_PLACEMENT) - set(ids))}"
    )

    preferences = catalog.get("preferences") or []
    assert [section["id"] for section in preferences] == list(_PREFERENCES_SECTIONS), (
        f"the Preferences sections: {preferences}"
    )
    assert all(section.get("label") for section in preferences), "each Preferences section has a label"
    slugs = _workshop_slugs(catalog["destinations"])

    by_id = {group["id"]: group for group in groups}
    problems = []
    counts = {"use": 0, "workshop": 0, "retired": 0, "embedded": 0}
    for group in groups:
        kinds = [kind for kind in _KINDS if group.get(kind)]
        if len(kinds) != 1:
            problems.append(f"{group['id']} is placed {len(kinds)} times: {kinds or 'nowhere'}")
            continue
        want = _PLACEMENT[group["id"]]
        kind = kinds[0]
        if kind == "space":
            place = (group["space"], group.get("section"))
            counts[group["space"]] = counts.get(group["space"], 0) + 1
            if group["space"] == "use" and place[1] not in _PREFERENCES_SECTIONS:
                problems.append(f"{group['id']} is in {place[1]}, no Preferences section")
            if group["space"] == "workshop" and place[1] not in slugs:
                problems.append(f"{group['id']} is in {place[1]}, no Workshop settings page")
            if place != want:
                problems.append(f"{group['id']} is in {place}, not {want}")
        elif kind == "retired":
            counts["retired"] += 1
            if want[0] != "retired" or not str(group["retired"]).strip():
                problems.append(f"{group['id']} is retired ({group['retired']!r}), not {want}")
        else:
            counts["embedded"] += 1
            host = by_id.get(group["embeddedIn"])
            if group["id"] not in _EMBEDDABLE or group["embeddedIn"] != "observability":
                problems.append(f"{group['id']} is embedded in {group['embeddedIn']}, not {want}")
            elif host is None or not host.get("space") or host.get("embeddedIn") or host.get("retired"):
                problems.append(f"{group['id']} is embedded in {group['embeddedIn']}, which has no page")
    assert not problems, "\n".join(problems)
    assert counts["use"] == 11 and counts["retired"] == 1, counts
    assert counts["workshop"] + counts["embedded"] == 39, counts


# ---------------------------------------------------------------------------
# NV8 -- no internal link names an old URL
# ---------------------------------------------------------------------------
_OLD_URL = re.compile(
    r"""(['"`])/(?:settings|health|benchmark|verify|claims|verify-answer|verify-citations)(?=[/?#'"`]|\$\{)"""
)


def _old_links(path, text):
    return len(_OLD_URL.findall(_code(path, text)))


def test_nv8_no_internal_link_names_an_old_url():
    assert _old_links(f"{_SRC}/lib/sample/Badge.svelte", '<a href="/settings?tab=security">A</a>') == 1
    assert _old_links(f"{_SRC}/lib/sample/Go.ts", "goto(`/verify?mode=${m}`); get('/api/health');") == 1
    assert _old_links(f"{_SRC}/lib/sample/Go.ts", "goto('/workshop/verify'); x = '/healthy';") == 0

    found = {
        path: count
        for path in files(_SCRIPTS, exclude=(_LEGACY,))
        if (count := _old_links(path, read(path)))
    }
    assert not found, f"links that name an old URL (legacy.ts alone knows them): {found}"


# ---------------------------------------------------------------------------
# NV9 -- rename and delete are visible in the chats index
# ---------------------------------------------------------------------------
_HOVER_ONLY = re.compile(
    r"(?<![\w-])(?:group-hover|group-focus-within|peer-hover):|(?<![\w-])hover:(?:flex|block|inline-flex|inline-block|visible|opacity-100)\b"
)
_ACTION = r"(?:>\s*|\blabel\s*[=:]\s*\{?\s*['\"`])(%s)\b"


def _reachable(path):
    """The components a page renders, followed through their imports."""
    seen, queue = [], [path]
    while queue:
        current = queue.pop()
        if current in seen or not (REPO / current).is_file():
            continue
        seen.append(current)
        queue += [p for p in _imports(current) if p.endswith(".svelte")]
    return seen


def test_nv9_the_chats_index_shows_rename_and_delete():
    sample = '<div class="hidden group-hover:flex"><button>Rename</button></div>'
    assert len(_HOVER_ONLY.findall(sample)) == 1 and re.search(_ACTION % "Rename", sample)

    index = _page_for("/chat")
    reachable = _reachable(index)
    hidden = {path: len(_HOVER_ONLY.findall(_markup(path) + _script(path))) for path in reachable}
    hidden = {path: count for path, count in hidden.items() if count}
    assert not hidden, f"controls the chats index reveals by the pointer alone: {hidden}"
    text = "\n".join(_code(path) for path in reachable)
    missing = [word for word in ("Rename", "Delete") if not re.search(_ACTION % word, text)]
    assert not missing, (
        f"the chats index ({index} and the {len(reachable) - 1} components it renders) shows no "
        f"visible control for: {missing}"
    )


# ---------------------------------------------------------------------------
# NV10 -- the chats index searches on the server and pages the listing
# ---------------------------------------------------------------------------
_SEARCH_CASES = (("onions", 0), ("  onions  ", 3), ("a b", 1))
_PAGE_CASES = (("", 0), ("", 1), ("", 4), ("   ", 2))
_REACHED_CASES = ((("onions", 200), True), (("onions", 250), True), (("onions", 199), False),
                  (("", 200), False), (("", 0), False))


# A test of words inside a filter's callback: the index matching the words
# itself instead of asking the server.
_WORD_TEST = re.compile(r"\.(?:includes|indexOf|startsWith|endsWith|search|match|test)\s*\(")


def _filter_arguments(code):
    """The argument of every ``.filter(`` call, read to its balanced closing
    parenthesis, so a callback whose parameter sits in parentheses is read
    whole."""
    found = []
    for match in re.finditer(r"\.filter\s*\(", code):
        depth, end = 1, match.end()
        while end < len(code) and depth:
            depth += {"(": 1, ")": -1}.get(code[end], 0)
            end += 1
        found.append(code[match.end():end - 1])
    return found


def _chats():
    return _node(
        "chats", ("OO_CHATS_INDEX",),
        {"params": [list(case) for case in _SEARCH_CASES + _PAGE_CASES],
         "reached": [list(case) for case, _ in _REACHED_CASES]},
    )


@pytest.mark.parametrize("half", ("search", "paging"))
def test_nv10_the_chats_index_searches_on_the_server_and_pages_the_listing(half):
    result = _chats()
    params = dict(zip(_SEARCH_CASES + _PAGE_CASES, result["params"]))
    index = _page_for("/chat")
    page = _code(index)
    if half == "search":
        assert result["search"] == 200, f"a search asks for 200 conversations: {result['search']}"
        for query, number in _SEARCH_CASES:
            assert params[(query, number)] == {"q": query.strip(), "limit": 200}, (
                f"a search sends its words and the limit, and no offset: {params[(query, number)]}"
            )
        reached = dict(zip([case for case, _ in _REACHED_CASES], result["reached"]))
        assert reached == {case: want for case, want in _REACHED_CASES}, (
            f"the index says when a search filled its limit, and only a search: {reached}"
        )
        assert _imports_name(index, "indexParams", _CHATS_INDEX) and _calls(page, "indexParams"), (
            "the index builds its request with indexParams"
        )
        assert re.search(r"\{#if\b[^}]*\blimitReached\s*\(", _markup(index)), (
            "the index says, in its page, when the search limit was reached"
        )
        assert not re.search(r"\.filter\s*\([^)]*\.(?:includes|indexOf|startsWith)\s*\(", page, re.S), (
            "the index does not filter the conversations by the words itself"
        )
        assert re.search(
            r"\.filter\s*\([^)]*\.(?:includes|indexOf|startsWith)\s*\(",
            "chats = list.filter(c => c.title.includes(words));", re.S,
        ), "the first census reads a word test in a filter whose callback takes a bare parameter"
        sample = "chats = list.filter((chat) => chat.title.toLowerCase().includes(words));"
        assert [found for found in _filter_arguments(sample) if _WORD_TEST.search(found)], (
            "the census reads a word test in a filter whose callback takes its parameter in parentheses"
        )
        filters = [found for found in _filter_arguments(page) if _WORD_TEST.search(found)]
        assert not filters, f"the index does not filter the conversations by the words itself: {filters}"
        return

    size = result["page"]
    assert isinstance(size, int) and 1 <= size <= 500, f"a page size the server accepts: {size}"
    for query, number in _PAGE_CASES:
        assert params[(query, number)] == {"limit": size, "offset": number * size}, (
            f"the plain listing pages with limit and offset: {params[(query, number)]}"
        )
    api = _script(_CONVERSATIONS_API)
    signature = re.search(r"export\s+async\s+function\s+listConversations\s*\(([^)]*)\)", api, re.S)
    assert signature and re.search(r"\boffset\s*\?\s*:", signature.group(1)), (
        "listConversations takes an offset"
    )
    assert re.search(r"queryParams\.offset\s*=", api), "and sends it"
    assert _imports_name(index, "listConversations", _CONVERSATIONS_API) and _calls(page, "listConversations"), (
        "the index lists the conversations itself, page by page"
    )


# ---------------------------------------------------------------------------
# NV18 -- no document names a "Settings >" path of the app
# ---------------------------------------------------------------------------
_SETTINGS_PATH = re.compile(r"\bSettings[ \t]*>")
# The repository host's own settings, not the app's.
_HOST_SETTINGS = ("docs/BRANCH_PROTECTION.md",)


def test_nv18_no_doc_names_a_settings_path():
    sample = "1. Open **Settings > Plugins > Marketplace**\n"
    assert len(_SETTINGS_PATH.findall(sample)) == 1, "the census reads a settings path"
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard",
         "--", "*.md"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8").split("\0")
    docs = [path for path in listed if path and path not in _HOST_SETTINGS and (REPO / path).is_file()]
    assert len(docs) >= 20, f"the census reads the tracked documents: {len(docs)}"
    found = {
        path: count for path in docs
        if (count := len(_SETTINGS_PATH.findall(read(path))))
    }
    assert not found, f"documents that name a Settings path of the app (Preferences or Workshop now): {found}"



# ---------------------------------------------------------------------------
# NV19 -- the settings search reads both spaces and links to the page
# ---------------------------------------------------------------------------
# A search, the groups it must find, and the groups it must not.
_SETTINGS_QUERIES = (
    ("mfa", {"totp", "webauthn"}, set()),
    ("ONBOARDING", {"conversation-config-maintenance"}, set()),
    ("ceremony", {"key-ceremony"}, set()),
    ("authenticator mfa", {"totp"}, {"webauthn"}),
    ("temperature", set(), {"conversation-defaults"}),
    ("probe-nothing-matches-this", set(), None),
    ("   ", set(), None),
)


def _settings_search(catalog=None):
    return _node(
        "settings", ("OO_SETTINGS_SEARCH", "OO_CATALOG", "OO_DESTINATIONS"),
        {"catalog": catalog, "queries": [query for query, _, _ in _SETTINGS_QUERIES]},
    )


def _index_expected(groups, destinations, preferences):
    """What the index holds: every group that has a page, linked to it."""
    by_id = {group["id"]: group for group in groups}
    labels = {d["href"]: d["label"] for d in destinations}
    sections = {s["id"]: s["label"] for s in preferences}
    out = {}
    for group in groups:
        placed = _placed_page(group, by_id)
        if placed is None:
            continue
        page, g = placed
        where = (f"Preferences, {sections[by_id[g]['section']]}" if page == "/preferences"
                 else f"Workshop, {labels[page]}")
        out[group["id"]] = {"href": f"{page}?g={g}", "where": where}
    return out


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_nv19_the_settings_search_reads_both_spaces_and_links_to_the_page(half):
    if half == "wiring":
        hub = _code(_HUB)
        for name in ("settingsIndex", "searchSettings"):
            assert _imports_name(_HUB, name, _SETTINGS_SEARCH) and _calls(hub, name), (
                f"the settings hub searches with {name}"
            )
        filters = [found for found in _filter_arguments(hub) if _WORD_TEST.search(found)]
        assert not filters, (
            f"the settings hub filters the groups itself, by their words or their space: {filters}"
        )
        return

    catalog = _catalog()
    result = _settings_search()
    groups = _groups(catalog)
    ids = [hit["id"] for hit in result["index"]]
    assert len(ids) == len(set(ids)), f"each group is in the search once: {ids}"
    expected = _index_expected(groups, catalog["destinations"], catalog["preferences"])
    assert len(expected) >= 40 and {"use", "workshop"} <= {
        "use" if place["href"].startswith("/preferences") else "workshop" for place in expected.values()
    }, "the census reads groups of both spaces"
    got = {hit["id"]: {"href": hit["href"], "where": hit["where"]} for hit in result["index"]}
    wrong = {gid: f"{got.get(gid)} (expected {place})" for gid, place in expected.items() if got.get(gid) != place}
    extra = sorted(set(got) - set(expected))
    assert not wrong and not extra, (
        f"every group with a page is found, linked to the page that holds it: wrong {wrong}; "
        f"found with no page {extra}"
    )
    found = dict(zip([query for query, _, _ in _SETTINGS_QUERIES], result["found"]))
    for query, must, must_not in _SETTINGS_QUERIES:
        hits = set(found[query])
        if must_not is None:
            assert hits == must, f"{query!r} finds {sorted(hits)} (expected {sorted(must)})"
            continue
        assert must <= hits and not (must_not & hits), (
            f"{query!r} finds {sorted(hits)}: it must find {sorted(must)} and not {sorted(must_not)}"
        )
    spaces = {got[gid]["href"].split("?")[0] == "/preferences" for gid in found["mfa"] + found["ONBOARDING"]}
    assert spaces == {True, False}, "a search from either page finds the groups of both spaces"

    probe = _settings_search(_PROBE_CATALOG)
    probe_groups = _groups({"sections": _PROBE_CATALOG["SETTINGS_SECTIONS"], "inline": _PROBE_CATALOG["INLINE_GROUPS"]})
    probe_expected = _index_expected(probe_groups, catalog["destinations"], catalog["preferences"])
    probe_got = {hit["id"]: {"href": hit["href"], "where": hit["where"]} for hit in probe["index"]}
    assert probe_got == probe_expected, (
        f"an embedded group links to its host's page, a retired one is not found: {probe_got}"
    )


# ---------------------------------------------------------------------------
# NV20 -- the settings search finds a group by its page and its former section
# ---------------------------------------------------------------------------
def _page_queries(catalog):
    """``{words: group ids}``: the name of every page that holds groups
    (Preferences and its sections, the Workshop and its pages) and of every
    section of the old settings page, each with the groups a search of it
    must find."""
    groups = _groups(catalog)
    by_id = {group["id"]: group for group in groups}
    labels = {d["href"]: d["label"] for d in catalog["destinations"]}
    sections = {s["id"]: s["label"] for s in catalog["preferences"]}
    former = {s["id"]: s["label"] for s in catalog["sections"]}
    must = {}
    for group in groups:
        placed = _placed_page(group, by_id)
        if placed is None:
            continue
        page, host = placed
        names = (
            ["Preferences", sections[by_id[host]["section"]]] if page == "/preferences"
            else ["Workshop", labels[page]]
        )
        names.append(former[group["oldSection"]])
        for name in names:
            must.setdefault(name, set()).add(group["id"])
    return must


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_nv20_the_settings_search_finds_a_group_by_its_page_and_its_former_section(half):
    if half == "wiring":
        hub = _code(_HUB)
        calls = re.findall(r"\bsettingsIndex\s*\(([^()]*)\)", hub)
        assert calls and all(re.search(r"\bSETTINGS_SECTIONS\b", args) for args in calls), (
            f"the settings hub hands the search the sections of the old page: {calls}"
        )
        assert _imports_name(_HUB, "SETTINGS_SECTIONS", _CATALOG), "from the catalog"
        return

    catalog = _catalog()
    must = _page_queries(catalog)
    assert {"Preferences", "Workshop", "Appearance", "Security", "Plugins & Extensions"} <= set(must) and (
        "appearance-theme" in must["Appearance"] and "totp" in must["Account"]
    ), f"the census names the pages and the old sections: {sorted(must)}"
    queries = sorted(must)
    result = _node(
        "pages", ("OO_SETTINGS_SEARCH", "OO_CATALOG", "OO_DESTINATIONS"), {"queries": queries},
    )
    found = dict(zip(queries, result["found"]))
    missing = {query: sorted(ids - set(found[query])) for query, ids in must.items() if ids - set(found[query])}
    assert not missing, (
        f"a search of a page's name, or of an old section's, misses groups that sit there: {missing}"
    )


# ---------------------------------------------------------------------------
# NV21 -- the hub's search keeps its keys and its address
# ---------------------------------------------------------------------------
def _function_body(script, name):
    """The body of ``function name(...) {...}`` in a script, or ''."""
    match = re.search(rf"\bfunction\s+{re.escape(name)}\s*\([^)]*\)\s*(?::[^{{]*)?\{{", script)
    if not match:
        return ""
    depth, end = 1, match.end()
    while end < len(script) and depth:
        depth += {"{": 1, "}": -1}.get(script[end], 0)
        end += 1
    return script[match.end():end - 1]


def _call_argument(script, name):
    """The argument of the first ``name(`` call, read to its balanced
    closing parenthesis, or ''."""
    match = re.search(rf"(?<![\w$.]){re.escape(name)}\s*\(", script)
    if not match:
        return ""
    depth, end = 1, match.end()
    while end < len(script) and depth:
        depth += {"(": 1, ")": -1}.get(script[end], 0)
        end += 1
    return script[match.end():end - 1]


@pytest.mark.parametrize("half", ("keys", "address"))
def test_nv21_the_settings_search_keeps_its_keys_and_its_address(half):
    script, markup = _script(_HUB), _markup(_HUB)
    if half == "keys":
        sample = "function onKey(e) { if (e.key === 'Enter') goto(results[0].href); }"
        assert "results[0]" in _function_body(sample, "onKey"), "the census reads a handler's body"
        handler = re.search(r"<Input\b[^>]*\bon:keydown\s*=\s*\{\s*(\w+)\s*\}", markup)
        body = _function_body(script, handler.group(1)) if handler else ""
        assert re.search(r"['\"]Enter['\"]", body) and re.search(r"\bgoto\s*\(\s*results\[0\]", body), (
            f"Enter in the search field opens the first result: {body!r}"
        )
        assert re.search(r"['\"]Escape['\"]", body) and _calls(body, "clearSearch"), (
            f"Escape clears the search: {body!r}"
        )
        return

    sample = "afterNavigate(({ to }) => { words = to.url.searchParams.get('q') ?? ''; });"
    assert "words =" in _call_argument(sample, "afterNavigate"), "the census reads a callback whole"
    writers = [
        name for name in re.findall(r"\bfunction\s+(\w+)\s*\(", script)
        if re.search(r"searchParams\.set\(\s*['\"]q['\"]", _function_body(script, name))
        and re.search(r"\breplaceState\s*:\s*true\b", _function_body(script, name))
    ]
    assert writers, "the words go to the address as ?q=, replacing the entry rather than adding one"
    assert re.search(r"\bsetTimeout\s*\(\s*" + "(?:" + "|".join(writers) + r")\b", script), (
        f"once the reader pauses: {writers}"
    )
    after = _call_argument(script, "afterNavigate")
    assert re.search(r"\bwords\s*=", after) and re.search(r"searchParams\.get\(\s*['\"]q['\"]", after), (
        f"after every navigation the words are read from the address, so another page's search "
        f"never stays on screen: {after!r}"
    )


# ---------------------------------------------------------------------------
# NV22 -- every query a page answers, something in the interface produces
# ---------------------------------------------------------------------------
_QUERY_READ = re.compile(r"""\bsearchParams\.get\(\s*['"](\w+)['"]\s*\)""")
# A query a page answers with a view it also opens by a control of its own:
# a shortcut for a link from outside, not the only way in. Each reason is
# checked below.
_SHORTCUTS = {
    "new": (f"{_ROUTES}/(app)/(use)/projects/+page.svelte",
            "the projects list opens the same dialog from its own New project button"),
}
# The files that read the old addresses, which the interface no longer
# produces: that is their purpose.
_OLD_READERS = (_LEGACY, _ROOT_PAGE, *(f"{_ROUTES}/{name}/+page.ts" for name in _OLD_PAGES))


def _produces(code, key):
    return re.search(rf"[?&]{key}=", code) or re.search(rf"""\bsearchParams\.set\(\s*['"]{key}['"]""", code)


def test_nv22_every_query_a_page_answers_is_produced_by_the_interface():
    assert _QUERY_READ.findall("a = url.searchParams.get('run'); b = u.searchParams.get(\"q\");") == ["run", "q"]
    assert _produces("href={`/workshop/benchmarks?run=${id}`}", "run") and not _produces("?running=1", "run")
    codes = {path: _code(path) for path in files(_SCRIPTS)}
    read_by = {}
    for path, code in codes.items():
        if path in _OLD_READERS:
            continue
        for key in _QUERY_READ.findall(code):
            read_by.setdefault(key, set()).add(path)
    assert {"q", "g", "run"} <= set(read_by), f"the census reads the queries pages answer: {sorted(read_by)}"
    orphans = {}
    for key, readers in sorted(read_by.items()):
        if any(_produces(code, key) for code in codes.values()):
            continue
        shortcut = _SHORTCUTS.get(key)
        if shortcut and readers == {shortcut[0]}:
            continue
        orphans[key] = sorted(readers)
    assert not orphans, (
        f"queries a page answers that nothing in the interface produces (a view reachable by a typed "
        f"address alone): {orphans}"
    )
    assert re.search(r"\bNew project\b", _markup(f"{_SRC}/lib/components/panels/ProjectList.svelte")), (
        f"a shortcut's reason holds: {_SHORTCUTS['new'][1]}"
    )


# ---------------------------------------------------------------------------
# NV23 -- every page of both spaces has a heading of level one
# ---------------------------------------------------------------------------
def _layouts_above(page):
    """The layouts between the layout of both spaces and a page, the
    shell's own left out: they frame the page."""
    if not PurePosixPath(page).is_relative_to(_ROUTES):
        return []
    parts = PurePosixPath(page).relative_to(_ROUTES).parts[:-1]
    found = []
    for depth in range(2, len(parts) + 1):
        layout = PurePosixPath(_ROUTES, *parts[:depth], "+layout.svelte")
        if (REPO / str(layout)).is_file():
            found.append(str(layout))
    return found


def _level_one(page, text=None):
    """How many ``<h1`` the page, what it mounts and the layouts above it draw."""
    sources = [_markup(page, text)]
    for imported in _imports(page, text):
        if imported.endswith(".svelte") and (REPO / imported).is_file():
            sources.append(_markup(imported))
    sources += [_markup(layout) for layout in _layouts_above(page)]
    return sum(len(re.findall(r"<h1\b", source)) for source in sources)


def test_nv23_every_page_of_both_spaces_has_a_heading_of_level_one():
    assert _level_one("sample.svelte", "<main><h1 class=\"t\">Notes</h1></main>") == 1
    assert _level_one("sample.svelte", "<main><h2>Notes</h2></main>") == 0
    pages = [path for path in _page_components() if _route_groups(path)[:1] == ["(app)"]]
    assert len(pages) >= 8, f"the census reads the pages of both spaces: {pages}"
    without = [path for path in pages if not _level_one(path)]
    assert not without, f"pages with no heading of level one: {without}"


# ---------------------------------------------------------------------------
# NV24 -- no copy and no document sends the reader to the old settings page
# ---------------------------------------------------------------------------
_OLD_SETTINGS = re.compile(
    r"\b(?:[Gg]o to|[Ii]n|[Ff]rom|[Uu]nder|[Vv]ia|[Oo]pen)[ \t]+(?:the[ \t]+)?Settings\b"
    r"|\bSettings[ \t]+(?:page|tab|screen)\b|\bSettings[ \t]*(?:>|->)"
)
# Text the application writes for a developer to read: the README of a new
# plugin.
_GENERATED = ("opti_oignon/plugin_template.py",)


def test_nv24_no_copy_and_no_document_sends_the_reader_to_the_old_settings_page():
    sample = "Go to Settings, then the Pipelines tab. It is in Settings. The Settings page. Settings > Plugins."
    assert len(_OLD_SETTINGS.findall(sample)) == 4, "the census reads each way of naming the old page"
    assert not _OLD_SETTINGS.findall("import ShortcutSettings from './x'; the Workshop settings pages")
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard",
         "--", "*.md"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8").split("\0")
    docs = [path for path in listed if path and path not in _HOST_SETTINGS and (REPO / path).is_file()]
    texts = {path: read(path) for path in docs}
    texts.update({path: _code(path) for path in files(_SCRIPTS)})
    texts.update({path: read(path) for path in _GENERATED})
    assert len(docs) >= 20 and len(texts) >= 300, f"the census reads the documents and the copy: {len(texts)}"
    found = {path: len(hits) for path, text in texts.items() if (hits := _OLD_SETTINGS.findall(text))}
    assert not found, (
        f"copy and documents that send the reader to the old settings page (Preferences or the "
        f"Workshop now): {found}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly"]))
