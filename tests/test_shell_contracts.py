#!/usr/bin/env python3
"""Contracts for the shell: one frame, two spaces, and a stop always one tap away.

One ``AppShell`` is mounted once, by the layout every page of both spaces
sits under (``routes/(app)/+layout.svelte``), and takes its page through its
default slot alone. Pages sit in two route groups, ``(use)`` and
``(workshop)``, the Workshop's URLs under ``/workshop``. On a desktop the
sidebar is never unmounted: collapsed, it is a 72 px rail that keeps the
stop. On a phone, a header outside the drawer holds the stop and the
drawer's opener. The status card at the foot of the sidebar names the
backend and its state, holds ``Stop all``, the security grade, and the mode
word in Bulbe alone. The emergency stop has one poller and one control.

The names the contracts read, beside those of the navigation contracts
(``tests/test_navigation_contracts.py``):

  * ``lib/stores/estop.ts`` -- ``estop``, a writable store whose value
    carries ``available`` (true, false, or null while unknown) and
    ``stopped``; it is the one caller of the emergency stop's status, and
    its one poller.
  * ``lib/stores/backendStatus.ts`` -- ``backendStatus``, a writable store
    whose value carries ``reachable`` (null before the first answer, false
    when the API itself failed) and ``backend`` (the active backend's
    ``display_name`` and ``healthy``, or null); it reads ``/api/backends``
    once a minute while the page is visible.
  * ``lib/stores/exportDialog.ts`` -- the one export dialog's store, opened
    by ``openExportDialog(id, title)``.
  * ``lib/stores/ui.ts`` -- ``sidebarOpen`` (on a desktop, expanded or the
    rail; on a phone, the drawer open or shut) and ``isPhone`` (the shell
    under 768 px).
  * ``lib/components/layout/`` -- ``StopAllButton`` (the only stop control),
    ``StatusCard``, ``SpaceSwitch``, ``WorkshopBand``, ``PhoneHeader``; the
    sidebar's root says ``data-collapsed`` true or false.

  * SH1 -- exactly one layout mounts the shell.
  * SH2 -- the shell exposes its default slot alone, and nothing fills a
    named slot of it.
  * SH3 -- every page sits in the Use group, the Workshop group, or on the
    list of pages outside both (sign-in, registration, the component
    gallery, the redirects of the old pages and of the root, and the
    catch-all of SH23); Workshop URLs start with ``/workshop`` and Use URLs
    do not.
  * SH4 -- the estop store is the only caller of the stop's status and the
    only poller of it; ``StopAllButton`` is the only stop control, and it
    always renders: available, unavailable (disabled, with the reason as
    visible text), unknown (enabled), and stopped (the pill with Resume).
  * SH5 -- the status card reads ``/api/backends`` through its store, once a
    minute while the page is visible, and tells "Server unreachable" from
    "<backend> unavailable".
  * SH6 -- the approvals drawer is mounted once, by the shell's layout.
  * SH7 -- the status card mounts the security badge (which leads to the
    Workshop's security page); the mode word appears in Bulbe alone,
    leading there too.
  * SH8 -- switching space goes to that space's last route, else its home;
    the space switch decides with ``space.ts`` and nothing of its own.
  * SH9 -- the Workshop layout marks its space and draws it compact, with
    its band; the Use layout marks nothing and follows the reader's density.
  * SH10 -- export is one dialog at the shell's level, opened through its
    store from anywhere, never through a window event.
  * SH11 -- the device sync panel keeps "Show text" and the skills panel
    keeps the sync state, each loaded by its group.
  * SH12 -- on a desktop the sidebar is never unmounted: collapsed, it is
    the 72 px rail, and the rail holds ``Stop all``.
  * SH13 -- under 768 px the shell renders its phone header outside the
    drawer, and the header holds ``Stop all`` and the drawer's opener.
  * SH14 -- the stop's confirmation is fixed to the viewport and placed from
    its button at run time, so no clipping box (the 72 px rail, the card, a
    dialog's panel) cuts it; the approvals drawer, a modal dialog that makes
    the rest of the page inert, holds ``Stop all`` itself.
  * SH15 -- nothing but a dialog covers the phone header: it sits above
    every layer that is not modal, a side panel on a phone stands over the
    page's frame (never fixed to the viewport over the header), and the chat
    frame gives that panel a named close control.
  * SH16 -- pending approvals are shown in every state of the shell: the
    status card, the collapsed rail and the phone header each carry the
    control that opens the drawer, and none shows while nothing waits.
  * SH17 -- on a touch screen the safety controls are 44 px targets: the
    stop's trigger, its confirmation's actions and Resume, the approvals
    control, and the links of the drawer's card and brand.
  * SH18 -- the shell keeps clear of the safe areas at every width: the
    sidebar and the sheet read the insets of their edges, not the phone's
    branch alone.
  * SH19 -- the phone drawer is modal while it is open: the header and the
    page behind it are inert, the drawer is a labelled modal dialog with a
    named close control, focus moves into it and back to its opener.
  * SH20 -- the stop speaks once: only the copy of the control the reader
    acted on announces a change or an error, and the stopped pill is no
    live region of its own.
  * SH21 -- the stop's reads: a status that cannot be read makes the stop
    unknown again (enabled), never the last answer; one read at a time,
    whoever asks.
  * SH22 -- the stop's label reaches 4.5:1 on every ground its styles draw
    it on (at rest, under the pointer, the stopped pill), in every palette.
  * SH23 -- an address no page serves is answered inside the shell: a
    catch-all page under the layout of both spaces throws a 404, which the
    shell's error page draws, with the sidebar and ``Stop all``.
  * SH24 -- the network page shows the server's reachability (the
    inference server online, its latency and queue, its last error), read
    when the page is shown and on request, never on a timer.

The server-rendering halves render components through
``tests/_frontend.ssr()``, with small wrappers planted in the renderer's
copy that set the stores first; they prove what the templates emit when
compiled for the server, not what a browser does. The shell kept across a
space switch, the stop in one tap on a phone on every route, the keyboard
and the colours are owed to the machine.

Local-only (the public distribution ships no tests). Needs Node >= 22.6 and
``frontend/node_modules``; without them the helpers raise, and so do the
contracts.
"""

import re
import sys
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_navigation_contracts as _nav  # noqa: E402
from _frontend import REPO, files, read, ssr  # noqa: E402
from test_ui_primitives_contracts import _dom  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file. The first server-rendering contract starts the
# session's server when this suite runs alone; the first render of the
# whole shell compiles its components.
BUDGET_S = {
    "test_sh1_exactly_one_layout_mounts_the_shell": 1.0,
    "test_sh2_the_shell_exposes_only_its_default_slot": 1.0,
    "test_sh3_every_page_sits_in_its_space_or_on_the_outside_list": 1.0,
    "test_sh4_one_estop_poller_and_one_stop_control_that_always_renders[poller]": 1.0,
    "test_sh4_one_estop_poller_and_one_stop_control_that_always_renders[only]": 1.0,
    "test_sh4_one_estop_poller_and_one_stop_control_that_always_renders[states]": 6.0,
    "test_sh5_the_status_card_reads_the_backends_each_minute_and_tells_two_failures_apart[poll]": 1.0,
    "test_sh5_the_status_card_reads_the_backends_each_minute_and_tells_two_failures_apart[states]": 3.0,
    "test_sh6_the_approvals_drawer_is_mounted_once_by_the_shell": 1.0,
    "test_sh7_the_status_card_mounts_the_badge_and_names_the_mode_in_bulbe_alone[badge]": 1.0,
    "test_sh7_the_status_card_mounts_the_badge_and_names_the_mode_in_bulbe_alone[mode]": 1.0,
    "test_sh8_switching_space_goes_to_its_last_route_else_its_home[node]": 2.0,
    "test_sh8_switching_space_goes_to_its_last_route_else_its_home[wiring]": 1.0,
    "test_sh9_the_workshop_layout_marks_its_space_and_the_use_layout_does_not": 1.0,
    "test_sh10_export_is_one_shell_level_dialog_opened_through_its_store": 1.0,
    "test_sh11_sync_and_skills_panels_keep_show_text_and_the_sync_state": 1.0,
    "test_sh12_on_a_desktop_the_sidebar_stays_and_its_rail_keeps_stop[width]": 1.0,
    "test_sh12_on_a_desktop_the_sidebar_stays_and_its_rail_keeps_stop[mounted]": 3.0,
    "test_sh13_on_a_phone_the_shell_header_holds_stop_outside_the_drawer[static]": 1.0,
    "test_sh13_on_a_phone_the_shell_header_holds_stop_outside_the_drawer[mounted]": 1.0,
    "test_sh14_the_stop_confirmation_is_never_clipped_and_the_approvals_drawer_holds_the_stop[clip]": 1.0,
    "test_sh14_the_stop_confirmation_is_never_clipped_and_the_approvals_drawer_holds_the_stop[dialog]": 3.0,
    "test_sh15_nothing_but_a_dialog_covers_the_phone_header[layer]": 1.0,
    "test_sh15_nothing_but_a_dialog_covers_the_phone_header[close]": 1.0,
    "test_sh16_pending_approvals_show_in_every_state_of_the_shell": 3.0,
    "test_sh17_the_safety_controls_are_forty_four_pixel_targets_on_a_touch_screen[stop]": 3.0,
    "test_sh17_the_safety_controls_are_forty_four_pixel_targets_on_a_touch_screen[pill]": 3.0,
    "test_sh17_the_safety_controls_are_forty_four_pixel_targets_on_a_touch_screen[links]": 1.0,
    "test_sh18_the_shell_keeps_clear_of_the_safe_areas_at_every_width": 1.0,
    "test_sh19_the_phone_drawer_is_modal_while_open[inert]": 3.0,
    "test_sh19_the_phone_drawer_is_modal_while_open[focus]": 1.0,
    "test_sh20_the_stop_speaks_once": 3.0,
    "test_sh21_the_stop_reads_one_at_a_time_and_an_unread_status_is_unknown[node]": 2.0,
    "test_sh21_the_stop_reads_one_at_a_time_and_an_unread_status_is_unknown[wiring]": 1.0,
    "test_sh22_the_stop_label_reaches_four_and_a_half_to_one_on_every_ground": 2.0,
    "test_sh23_an_unknown_address_is_answered_inside_the_shell": 1.0,
    "test_sh24_the_network_page_shows_the_server_reachability": 1.0,
}

_SRC = "frontend/src"
_ROUTES = f"{_SRC}/routes"
_LAYOUT = f"{_SRC}/lib/components/layout"
_APP_SHELL = f"{_LAYOUT}/AppShell.svelte"
_SIDEBAR = f"{_LAYOUT}/Sidebar.svelte"
_STATUS_CARD = f"{_LAYOUT}/StatusCard.svelte"
_STOP_ALL = f"{_LAYOUT}/StopAllButton.svelte"
_SPACE_SWITCH = f"{_LAYOUT}/SpaceSwitch.svelte"
_BAND = f"{_LAYOUT}/WorkshopBand.svelte"
_PHONE_HEADER = f"{_LAYOUT}/PhoneHeader.svelte"
_APP_LAYOUT = f"{_ROUTES}/(app)/+layout.svelte"
_USE_LAYOUT = f"{_ROUTES}/(app)/(use)/+layout.svelte"
_WORKSHOP_LAYOUT = f"{_ROUTES}/(app)/(workshop)/+layout.svelte"
_ROOT_LAYOUT = f"{_ROUTES}/+layout.svelte"
_ESTOP_API = f"{_SRC}/lib/api/estop.ts"
_ESTOP_STORE = f"{_SRC}/lib/stores/estop.ts"
_BACKEND_STORE = f"{_SRC}/lib/stores/backendStatus.ts"
_EXPORT_STORE = f"{_SRC}/lib/stores/exportDialog.ts"
_SECURITY_BADGE = f"{_SRC}/lib/components/sidebar/SecurityBadge.svelte"
_SPACE = f"{_SRC}/lib/nav/space.ts"
_SYNC_PANEL = f"{_SRC}/lib/components/panels/SyncPanel.svelte"
_SKILLS_PANEL = f"{_SRC}/lib/components/panels/SkillsPanel.svelte"
_CATALOG = f"{_SRC}/lib/settings/catalog.ts"
_APPROVALS_DRAWER = f"{_SRC}/lib/components/chat/ToolCallApprovalDrawer.svelte"
_APPROVALS_PILL = f"{_LAYOUT}/ApprovalsPill.svelte"
_SIDE_PANEL = f"{_SRC}/lib/ds/SidePanel.svelte"
_CHAT_LAYOUT = f"{_ROUTES}/(app)/(use)/chat/+layout.svelte"
_ESTOP_STATE = f"{_SRC}/lib/stores/estopState.ts"
_APP_ERROR = f"{_ROUTES}/(app)/+error.svelte"
_HUB = f"{_SRC}/lib/components/settings/SettingsHub.svelte"
_REACHABILITY = f"{_SRC}/lib/components/settings/NetworkReachability.svelte"
_NETWORK_API = f"{_SRC}/lib/api/network.ts"

_SCRIPTS = (".svelte", ".ts", ".js")
_FIXTURES = f"{_SRC}/lib/ssr_fixture/shell"

_code = _nav._code
_script = _nav._script
_markup = _nav._markup
_imports = _nav._imports
_imports_name = _nav._imports_name
_calls = _nav._calls


def _mounts(path, tag, text=None):
    """How many times a component's markup mounts ``<tag``."""
    return len(re.findall(rf"<{re.escape(tag)}\b", _markup(path, text)))


def _value_imports(path, module, text=None):
    """How many value imports (not type imports) of ``module`` a file holds."""
    return sum(
        1 for match in re.finditer(r"\bimport\s+(type\s+)?[^;]*?from\s*(['\"])([^'\"]+)\2", _script(path, text))
        if _nav._resolve(path, match.group(3)) == module and not match.group(1)
    )


def _mounted_by(tag):
    """``{path: count}`` of the files that mount ``<tag``."""
    return {
        path: count for path in files(".svelte")
        if (count := _mounts(path, tag))
    }


_STYLE = re.compile(r"<style\b[^>]*>(.*?)</style>", re.S)


def _style(path):
    return re.sub(r"/\*.*?\*/", " ", "\n".join(_STYLE.findall(read(path))), flags=re.S)


# ---------------------------------------------------------------------------
# Rendering: wrappers planted in the renderer's copy set the stores first
# ---------------------------------------------------------------------------
_PLANTED = set()


def _plant(name, text):
    path = f"{_FIXTURES}/{name}.svelte"
    if path not in _PLANTED:
        ssr().plant(path, text)
        _PLANTED.add(path)
    return path


def _wrapper(name, component, sets, body=None, attrs=""):
    """A wrapper that sets stores (``sets``: lines of script) and mounts a
    component (``component``: its repository path), with ``attrs`` written
    on its tag."""
    tag = PurePosixPath(component).stem
    lib = "$lib/" + str(PurePosixPath(component).relative_to(f"{_SRC}/lib"))
    stores = {}
    for line in sets:
        match = re.match(r"(\w+)\.", line)
        stores.setdefault(_STORE_OF[match.group(1)], []).append(match.group(1))
    imports = [
        f"\timport {{ {', '.join(sorted(set(names)))} }} from '{module}';"
        for module, names in sorted(stores.items())
    ]
    opening = f"{tag} {attrs}".strip()
    inner = f"<{opening}>{body}</{tag}>" if body else f"<{opening} />"
    return _plant(name, (
        "<script>\n"
        + "\n".join(imports) + "\n"
        + f"\timport {tag} from '{lib}';\n"
        + "\n".join(f"\t{line}" for line in sets) + "\n"
        + "</script>\n"
        + inner + "\n"
    ))


_STORE_OF = {
    "estop": "$lib/stores/estop",
    "backendStatus": "$lib/stores/backendStatus",
    "securityModeStatus": "$lib/stores/securityMode",
    "sidebarOpen": "$lib/stores/ui",
    "isPhone": "$lib/stores/ui",
    "pendingApprovals": "$lib/stores/approvals",
}


def _estop(available, stopped=False):
    value = "null" if available is None else ("true" if available else "false")
    return f"estop.update((s) => ({{ ...s, available: {value}, stopped: {'true' if stopped else 'false'} }}));"


def _backend(reachable, name=None, healthy=None):
    backend = "null" if name is None else (
        f"{{ name: 'ollama', display_name: '{name}', healthy: {'true' if healthy else 'false'}, "
        f"active: true, model_count: 3 }}"
    )
    reach = "null" if reachable is None else ("true" if reachable else "false")
    return f"backendStatus.update((s) => ({{ ...s, reachable: {reach}, backend: {backend} }}));"


def _mode(mode):
    return f"securityModeStatus.update((s) => ({{ ...s, mode: '{mode}', available: true }}));"


def _render(path):
    return _dom(ssr().render(path).html)


def _name(element):
    """An element's accessible name as the contracts read it: its label, or
    its text."""
    label = element.get("aria-label")
    if label is not None:
        return " ".join(label.split())
    return " ".join(element.text().split())


def _stops(root):
    return [b for b in root.iter("button") if re.search(r"\bstop all\b", _name(b), re.I)]


def _inside(element, ancestor):
    node = element.parent
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


def _visible(root):
    return " ".join(root.visible_text().split())


# ---------------------------------------------------------------------------
# SH1 -- exactly one layout mounts the shell
# ---------------------------------------------------------------------------
def test_sh1_exactly_one_layout_mounts_the_shell():
    sample = "<script>import AppShell from '$lib/components/layout/AppShell.svelte';</script>\n<AppShell onSelect={x}><slot /></AppShell>\n"
    assert _mounts("sample.svelte", "AppShell", sample) == 1, "the census reads a mount"
    mounted = _mounted_by("AppShell")
    assert mounted == {_APP_LAYOUT: 1}, (
        f"the shell is mounted once, by the layout both spaces sit under: {mounted}"
    )


# ---------------------------------------------------------------------------
# SH2 -- the shell exposes only its default slot
# ---------------------------------------------------------------------------
_SLOT_FILL = re.compile(r"""\bslot\s*=\s*(?:["']([\w-]+)["']|([\w-]+))""")


def _slot_names(markup):
    """The named slots a markup fills, quoted or not."""
    return [quoted or bare for quoted, bare in _SLOT_FILL.findall(markup)]


def test_sh2_the_shell_exposes_only_its_default_slot():
    slots = re.findall(r"<slot\b([^>]*)>", _markup(_APP_SHELL))
    named = [attributes for attributes in slots if re.search(r"\bname\s*=", attributes)]
    assert len(slots) == 1 and not named, (
        f"the shell has one slot, its default: {len(slots)} slots, named {named}"
    )
    filled = {
        path: re.findall(r"""\bslot\s*=\s*["']([\w-]+)["']""", _markup(path))
        for path in _mounted_by("AppShell")
    }
    filled = {path: names for path, names in filled.items() if names}
    assert not filled, f"layouts that fill a named slot of the shell: {filled}"
    spelled = '<p slot="header">a</p><p slot=\'panel\'>b</p><p slot=subheader>c</p>'
    assert _slot_names(spelled) == ["header", "panel", "subheader"], (
        "the census reads a named slot however its value is written"
    )
    unquoted = {path: _slot_names(_markup(path)) for path in _mounted_by("AppShell")}
    unquoted = {path: names for path, names in unquoted.items() if names}
    assert not unquoted, f"layouts that fill a named slot of the shell: {unquoted}"


# ---------------------------------------------------------------------------
# SH3 -- every page sits in its space, or on the outside list
# ---------------------------------------------------------------------------
_OUTSIDE = (
    f"{_ROUTES}/+page.ts",
    f"{_ROUTES}/login/+page.svelte",
    f"{_ROUTES}/register/+page.svelte",
    f"{_ROUTES}/dev/components/+page.svelte",
    *(f"{_ROUTES}/{name}/+page.ts" for name in _nav._OLD_PAGES),
    # The catch-all answers an address no page serves, in neither space
    # (SH23 holds it: one, under the shell's layout, throwing a 404).
    f"{_ROUTES}/(app)/[...missing]/+page.ts",
)


def _space_findings(pages):
    """What is wrong with where each page file sits: ``pages`` maps a page
    file to the URL it serves."""
    findings = []
    spaces = {"use": 0, "workshop": 0}
    for path, url in sorted(pages.items()):
        groups = _nav._route_groups(path)
        workshop_url = url == "/workshop" or url.startswith("/workshop/")
        if groups == ["(app)", "(use)"]:
            spaces["use"] += 1
            if workshop_url:
                findings.append(f"{path} is a Use page and serves {url}, a Workshop URL")
        elif groups == ["(app)", "(workshop)"]:
            spaces["workshop"] += 1
            if not workshop_url:
                findings.append(f"{path} is a Workshop page and serves {url}, outside /workshop")
        elif path not in _OUTSIDE:
            findings.append(f"{path} ({url}) sits in neither space and is not on the outside list")
    for space, count in spaces.items():
        if not count:
            findings.append(f"no page sits in the {space} group")
    return findings


def test_sh3_every_page_sits_in_its_space_or_on_the_outside_list():
    fixture = {
        f"{_ROUTES}/(app)/(use)/workshop/benchmarks/+page.svelte": "/workshop/benchmarks",
        f"{_ROUTES}/(app)/(workshop)/notes/+page.svelte": "/notes",
        f"{_ROUTES}/stray/+page.svelte": "/stray",
        f"{_ROUTES}/login/+page.svelte": "/login",
    }
    assert len(_space_findings(fixture)) == 3, (
        f"the census finds a Workshop URL in Use, a Use URL in the Workshop and a stray page: "
        f"{_space_findings(fixture)}"
    )
    pages = _nav._page_files()
    assert len(pages) >= 10, f"the census reads the pages: {len(pages)}"
    findings = _space_findings(pages)
    assert not findings, "\n".join(findings)


# ---------------------------------------------------------------------------
# SH4 -- one estop poller, one stop control, and it always renders
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", ("poller", "only", "states"))
def test_sh4_one_estop_poller_and_one_stop_control_that_always_renders(half):
    if half == "poller":
        assert _calls("x = await getEmergencyStopStatus();", "getEmergencyStopStatus")
        callers = [
            path for path in files(_SCRIPTS, exclude=(_ESTOP_API,))
            if _calls(_code(path), "getEmergencyStopStatus")
        ]
        assert callers == [_ESTOP_STORE], (
            f"the estop store is the one caller of the stop's status: {callers}"
        )
        near = [
            path for path in files(_SCRIPTS)
            if _ESTOP_API in _imports(path) or _ESTOP_STORE in _imports(path) or path == _ESTOP_STORE
        ]
        # A poll is a timer: an interval, or a timeout the store sets again
        # after each answer (one read at a time). A clear is not one.
        timers = re.compile(r"(?<![\w$.])set(?:Interval|Timeout)\s*\(")
        assert len(timers.findall("a = setInterval(f, 1); b = setTimeout(g, 2); clearTimeout(b);")) == 2, (
            "the census reads an interval and a timeout, and not a clear"
        )
        polling = {path: len(timers.findall(_code(path))) for path in near}
        polling = {path: count for path, count in polling.items() if count}
        assert polling == {_ESTOP_STORE: 1}, (
            f"one timer polls the stop, the store's: {polling}"
        )
        return

    if half == "only":
        actors = [
            path for path in files(_SCRIPTS, exclude=(_ESTOP_API,))
            if _calls(_code(path), "engageEmergencyStop") or _calls(_code(path), "resumeFromEmergencyStop")
        ]
        assert actors == [_ESTOP_STORE], f"the estop store alone engages and resumes: {actors}"
        reaching = (
            "<script>import { getEmergencyStopStatus } from '$lib/api/estop';\n"
            "import type { EmergencyActionResult } from '$lib/api/estop';</script>"
        )
        assert _value_imports(f"{_LAYOUT}/Sample.svelte", _ESTOP_API, reaching) == 1, (
            "the census reads a value import, and not a type import"
        )
        values = [
            path for path in files(_SCRIPTS, exclude=(_ESTOP_API, _ESTOP_STORE))
            if _value_imports(path, _ESTOP_API)
        ]
        assert not values, f"components that reach the stop's API themselves: {values}"
        assert _ESTOP_STORE in _imports(f"{_LAYOUT}/Sample.svelte", "<script>import { estop } from '$lib/stores/estop';</script>"), (
            "the census reads an import of the store"
        )
        readers = [
            path for path in files(".svelte") if _ESTOP_STORE in _imports(path)
        ]
        assert readers == [_STOP_ALL], (
            f"StopAllButton is the one component that holds the stop: {readers}"
        )
        others = _mounted_by("EmergencyStopControl")
        assert not others, f"another stop control is mounted: {others}"
        return

    cases = {
        "available": _render(_wrapper("StopAvailable", _STOP_ALL, [_estop(True)])),
        "unknown": _render(_wrapper("StopUnknown", _STOP_ALL, [_estop(None)])),
        "unavailable": _render(_wrapper("StopUnavailable", _STOP_ALL, [_estop(False)])),
    }
    for state, root in cases.items():
        stops = _stops(root)
        assert len(stops) == 1, f"{state}: one Stop all button renders: {len(stops)}"
        assert "Stop all" in " ".join(stops[0].visible_text().split()), (
            f"{state}: its label is visible: {stops[0].visible_text()!r}"
        )
    for state in ("available", "unknown"):
        stop = _stops(cases[state])[0]
        assert not stop.has("disabled") and stop.get("aria-haspopup") == "dialog", (
            f"{state}: Stop all is enabled and opens its dialog: {stop.attrs}"
        )
    stop = _stops(cases["unavailable"])[0]
    assert stop.has("disabled"), "unavailable: Stop all renders, disabled"
    shown = _visible(cases["unavailable"])
    assert re.search(r"\b(?:unavailable|not available)\b", shown, re.I), (
        f"unavailable: the reason is visible text: {shown!r}"
    )
    assert not re.search(r"\b(?:unavailable|not available)\b", _visible(cases["available"]), re.I)

    stopped = _render(_wrapper("StopStopped", _STOP_ALL, [_estop(True, stopped=True)]))
    assert "Stopped" in _visible(stopped), f"stopped: the pill says so: {_visible(stopped)!r}"
    resume = [b for b in stopped.iter("button") if "Resume" in _name(b)]
    assert len(resume) == 1 and not resume[0].has("disabled"), (
        "stopped: the pill holds Resume"
    )


# ---------------------------------------------------------------------------
# SH5 -- the status card reads the backends each minute, tells failures apart
# ---------------------------------------------------------------------------
_REQUESTS = re.compile(r"\bfetch\s*\(|\bapiGet\s*\(|['\"`]/api/")
_MINUTE = re.compile(r"(?<![\d_.])(?:60_?000|60\s*\*\s*1_?000)(?![\d_.])")


@pytest.mark.parametrize("half", ("poll", "states"))
def test_sh5_the_status_card_reads_the_backends_each_minute_and_tells_two_failures_apart(half):
    if half == "poll":
        assert _MINUTE.search("const EVERY = 60_000;") and not _MINUTE.search("15_000; 160000")
        card = _code(_STATUS_CARD)
        assert _BACKEND_STORE in _imports(_STATUS_CARD), "the card reads the backend's state from its store"
        assert not re.search(r"\bfetch\s*\(|\bapiGet\s*\(|['\"`]/api/", card), (
            "the card requests nothing itself"
        )
        assert all(_REQUESTS.search(sample) for sample in (
            "r = await fetch(url);", "r = await apiGet(x);", "const u = '/api/backends';",
        )), "the census reads a fetch, an API call and an API path"
        assert not _REQUESTS.search(card), "the card requests nothing itself"
        store = _code(_BACKEND_STORE)
        assert re.search(r"""['"`]/api/backends['"`?]""", store), "the store reads /api/backends"
        assert re.search(r"\bset(?:Interval|Timeout)\s*\(", store) and _MINUTE.search(store), (
            "the store reads it once a minute"
        )
        assert "visibilitychange" in store and re.search(r"\bdocument\.(?:hidden|visibilityState)\b", store), (
            "only while the page is visible"
        )
        return

    unreachable = _render(_wrapper("CardUnreachable", _STATUS_CARD, [
        _estop(True), _mode("daily"), _backend(False),
    ]))
    unavailable = _render(_wrapper("CardUnavailable", _STATUS_CARD, [
        _estop(True), _mode("daily"), _backend(True, "Ollama", False),
    ]))
    healthy = _render(_wrapper("CardHealthy", _STATUS_CARD, [
        _estop(True), _mode("daily"), _backend(True, "Ollama", True),
    ]))
    down, missing, well = _visible(unreachable), _visible(unavailable), _visible(healthy)
    assert "Server unreachable" in down and "Ollama unavailable" not in down, (
        f"the API itself failed: {down!r}"
    )
    assert "Ollama unavailable" in missing and "Server unreachable" not in missing, (
        f"the server answers and its backend is down: {missing!r}"
    )
    assert "Ollama" in well and not re.search(r"unavailable|unreachable", well, re.I), (
        f"the backend is up: {well!r}"
    )
    for root in (unreachable, unavailable, healthy):
        assert len(_stops(root)) == 1, "the card holds Stop all in every state"


# ---------------------------------------------------------------------------
# SH6 -- the approvals drawer is mounted once, by the shell
# ---------------------------------------------------------------------------
def test_sh6_the_approvals_drawer_is_mounted_once_by_the_shell():
    assert _mounts("sample.svelte", "ToolCallApprovalDrawer", "<ToolCallApprovalDrawer />") == 1
    mounted = _mounted_by("ToolCallApprovalDrawer")
    assert mounted == {_APP_LAYOUT: 1}, (
        f"the approvals drawer is mounted once, by the layout of both spaces: {mounted}"
    )


# ---------------------------------------------------------------------------
# SH7 -- the badge in the card; the mode word in Bulbe alone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", ("badge", "mode"))
def test_sh7_the_status_card_mounts_the_badge_and_names_the_mode_in_bulbe_alone(half):
    if half == "badge":
        assert _SECURITY_BADGE in _imports(_STATUS_CARD) and _mounts(_STATUS_CARD, "SecurityBadge") == 1, (
            "the status card mounts the security badge"
        )
        links = re.findall(r"""\bhref\s*=\s*["']([^"']+)["']""", _markup(_SECURITY_BADGE))
        assert links == ["/workshop/security"], (
            f"the badge leads to the Workshop's security page: {links}"
        )
        return

    bulbe = _render(_wrapper("CardBulbe", _STATUS_CARD, [
        _estop(True), _mode("bulbe"), _backend(True, "Ollama", True),
    ]))
    daily = _render(_wrapper("CardDaily", _STATUS_CARD, [
        _estop(True), _mode("daily"), _backend(True, "Ollama", True),
    ]))
    words = [
        a for a in bulbe.iter("a")
        if a.get("href") == "/workshop/security" and "Bulbe" in a.visible_text()
    ]
    assert len(words) == 1, (
        f"in Bulbe the card names the mode, leading to the security page: {_visible(bulbe)!r}"
    )
    shown = _visible(daily)
    assert "Bulbe" not in shown and "Daily" not in shown, (
        f"in Daily the card names no mode: {shown!r}"
    )


# ---------------------------------------------------------------------------
# SH8 -- switching space goes to its last route, else its home
# ---------------------------------------------------------------------------
_SPACE_PATHS = (
    ("/workshop", "workshop"), ("/workshop/models", "workshop"), ("/workshopx", "use"),
    ("/chat", "use"), ("/chat/abc", "use"), ("/", "use"), ("/preferences", "use"),
    ("/login", None), ("/register", None), ("/dev/components", None),
)
_SPACE_STEPS = (
    (("switch", "workshop"), "/workshop"),
    (("switch", "use"), "/chat"),
    (("remember", "/workshop/models?g=routing"), None),
    (("remember", "/chat/abc"), None),
    (("switch", "workshop"), "/workshop/models?g=routing"),
    (("switch", "use"), "/chat/abc"),
    (("remember", "/login"), None),
    (("switch", "use"), "/chat/abc"),
    (("remember", "/workshop/security"), None),
    (("switch", "workshop"), "/workshop/security"),
    (("switch", "use"), "/chat/abc"),
    (("stored", "workshop", {"workshop": "/workshop/models"}), "/workshop/models"),
    (("stored", "use", {"use": "/notes"}), "/notes"),
    (("stored", "workshop", {"workshop": "/chat"}), "/workshop"),
    (("stored", "use", {"use": "/workshop/models"}), "/chat"),
    (("stored", "use", {"use": "//evil.example/x"}), "/chat"),
    (("stored", "use", {"use": "/" + chr(92) + "evil.example"}), "/chat"),
    (("stored", "use", {"use": "https://evil.example/"}), "/chat"),
    (("stored", "use", {"use": "javascript:alert(1)"}), "/chat"),
)


_OWN_SPACE = re.compile(
    r"""startsWith\(\s*['"`]/workshop|===\s*['"`]/workshop['"`]|/\^\\/workshop"""
    r"""|\]\s*===\s*['"`]workshop['"`]|indexOf\(\s*['"`]/workshop['"`]\s*\)\s*===\s*0"""
)


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_sh8_switching_space_goes_to_its_last_route_else_its_home(half):
    if half == "wiring":
        switch = _code(_SPACE_SWITCH)
        assert _imports_name(_SPACE_SWITCH, "switchTarget", _SPACE) and _calls(switch, "switchTarget"), (
            "the space switch asks switchTarget where to go"
        )
        assert _calls(switch, "spaceHome"), "with the homes the table gives"
        remembered = [
            path for path in (_SPACE_SWITCH, _APP_SHELL, _APP_LAYOUT)
            if (REPO / path).is_file() and _calls(_code(path), "rememberRoute")
        ]
        assert remembered, "the shell remembers each route through rememberRoute"
        own = [
            path for path in files(_SCRIPTS, exclude=(_SPACE, _nav._ACTIVE, _nav._DESTINATIONS))
            if re.search(
                r"""startsWith\(\s*['"`]/workshop|===\s*['"`]/workshop['"`]|/\^\\/workshop""",
                _code(path),
            )
        ]
        assert not own, f"files that decide a space themselves: {own}"
        spelled = (
            "a = p.startsWith('/workshop'); b = p === '/workshop'; c = /^\\/workshop/.test(p);\n"
            "d = p.split('/')[1] === 'workshop'; e = segments[0] === 'workshop'; f = p.indexOf('/workshop') === 0;"
        )
        assert len(_OWN_SPACE.findall(spelled)) == 6, (
            f"the census reads each way of deciding a space: {_OWN_SPACE.findall(spelled)}"
        )
        deciding = [
            path for path in files(_SCRIPTS, exclude=(_SPACE, _nav._ACTIVE, _nav._DESTINATIONS))
            if _OWN_SPACE.search(_code(path))
        ]
        assert not deciding, f"files that decide a space themselves: {deciding}"
        return

    result = _nav._node(
        "space", ("OO_DESTINATIONS", "OO_SPACE"),
        {"paths": [path for path, _ in _SPACE_PATHS], "steps": [step for step, _ in _SPACE_STEPS]},
    )
    assert result["homes"] == {"use": "/chat", "workshop": "/workshop"}, (
        f"each space's home is its first ready destination: {result['homes']}"
    )
    spaces = dict(zip([path for path, _ in _SPACE_PATHS], result["spaces"]))
    assert spaces == dict(_SPACE_PATHS), f"spaceOf: {spaces}"
    problems = []
    previous = {}
    for (step, want), got in zip(_SPACE_STEPS, result["steps"]):
        if step[0] == "remember":
            if not got["kept"]:
                problems.append(f"remembering {step[1]} changed the record it was given")
            if step[1] == "/login" and got["last"] != previous:
                problems.append(f"a page outside both spaces was remembered: {got['last']}")
            previous = got["last"]
        elif got["to"] != want:
            problems.append(f"{step}: {got['to']} (expected {want})")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# SH9 -- the Workshop layout marks its space; the Use layout does not
# ---------------------------------------------------------------------------
def test_sh9_the_workshop_layout_marks_its_space_and_the_use_layout_does_not():
    tags = re.findall(r"<[A-Za-z][^<>]*>", _markup(_WORKSHOP_LAYOUT))
    marked = [
        tag for tag in tags
        if re.search(r"""\bdata-oo-space\s*=\s*["']workshop["']""", tag)
        and re.search(r"""\bclass\s*=\s*["'][^"']*\boo-density-compact\b""", tag)
    ]
    assert len(marked) == 1, (
        f"the Workshop layout marks its space and draws it compact on one element: {tags}"
    )
    assert _mounts(_WORKSHOP_LAYOUT, "WorkshopBand") == 1 and (REPO / _BAND).is_file(), (
        "the Workshop layout draws its band"
    )
    for path in (_USE_LAYOUT, _APP_LAYOUT):
        markup = _markup(path)
        assert "data-oo-space" not in markup and "oo-density-" not in markup, (
            f"{path} marks no space and follows the reader's density"
        )


# ---------------------------------------------------------------------------
# SH10 -- export is one shell-level dialog, opened through its store
# ---------------------------------------------------------------------------
def test_sh10_export_is_one_shell_level_dialog_opened_through_its_store():
    mounted = _mounted_by("ExportDialog")
    assert mounted == {_APP_LAYOUT: 1}, (
        f"the export dialog is mounted once, by the layout of both spaces: {mounted}"
    )
    assert _EXPORT_STORE in _imports(_APP_LAYOUT), "the shell's dialog follows its store"
    events = [path for path in files(_SCRIPTS) if "opti-export-conversation" in _code(path)]
    assert not events, f"export still travels as a window event: {events}"
    assert _imports_name(_ROOT_LAYOUT, "openExportDialog", _EXPORT_STORE) and _calls(
        _code(_ROOT_LAYOUT), "openExportDialog"
    ), "the export shortcut opens the dialog through its store, on every page"


# ---------------------------------------------------------------------------
# SH11 -- the sync and skills panels keep "Show text" and the sync state
# ---------------------------------------------------------------------------
def _loaders(panel):
    """``{path: imported file}`` of the files that load ``panel`` lazily."""
    found = {}
    for path in files(_SCRIPTS):
        for match in re.finditer(
            rf"\b{panel}\s*:\s*\(\)\s*=>\s*import\(\s*(['\"])([^'\"]+)\1", _script(path),
        ):
            found[path] = _nav._resolve(path, match.group(2))
    return found


def test_sh11_sync_and_skills_panels_keep_show_text_and_the_sync_state():
    catalog = read(_CATALOG)
    for group, panel, component in (
        ("device-sync", "SyncPanel", _SYNC_PANEL), ("skills", "SkillsPanel", _SKILLS_PANEL),
    ):
        entry = re.search(rf"\{{[^{{}}]*\bid\s*:\s*'{group}'[^{{}}]*\}}", catalog)
        assert entry and re.search(rf"\bpanel\s*:\s*'{panel}'", entry.group(0)), (
            f"the {group} group renders {panel}"
        )
        loaders = _loaders(panel)
        assert list(loaders.values()) == [component], (
            f"one page loads {panel}, from {component}: {loaders}"
        )
    assert "Show text" in _markup(_SYNC_PANEL), "the sync panel still shows a deferred skill's text"
    assert re.search(r"\.sync_state\b", _code(_SKILLS_PANEL)), "the skills panel still shows the sync state"


# ---------------------------------------------------------------------------
# SH12 -- on a desktop the sidebar stays; collapsed, its rail keeps Stop all
# ---------------------------------------------------------------------------
def _shell(name, phone, open_):
    return _render(_wrapper(name, _APP_SHELL, [
        f"isPhone.set({'true' if phone else 'false'});",
        f"sidebarOpen.set({'true' if open_ else 'false'});",
        _estop(True), _mode("daily"), _backend(True, "Ollama", True),
    ], body='<p id="oo-page-probe">page</p>'))


@pytest.mark.parametrize("half", ("width", "mounted"))
def test_sh12_on_a_desktop_the_sidebar_stays_and_its_rail_keeps_stop(half):
    if half == "width":
        style = _style(_SIDEBAR)
        assert re.search(
            r"""\[data-collapsed\s*=\s*(['"]?)true\1\][^{}]*\{[^}]*\bwidth\s*:\s*72px""", style,
        ), "the collapsed sidebar is 72 px wide"
        return

    for open_ in (False, True):
        root = _shell(f"ShellDesktop{'Open' if open_ else 'Rail'}", False, open_)
        sidebars = [e for e in root.iter() if e.has("data-collapsed")]
        assert len(sidebars) == 1, (
            f"desktop, sidebar {'open' if open_ else 'collapsed'}: the sidebar is mounted once: "
            f"{len(sidebars)}"
        )
        sidebar = sidebars[0]
        assert sidebar.get("data-collapsed") == ("false" if open_ else "true"), sidebar.attrs
        stops = [stop for stop in _stops(root) if _inside(stop, sidebar)]
        assert len(stops) == 1, (
            f"desktop, sidebar {'open' if open_ else 'collapsed'}: the sidebar holds Stop all: "
            f"{[_name(b) for b in root.iter('button')]}"
        )
        probes = [e for e in root.iter("p") if e.get("id") == "oo-page-probe"]
        assert len(probes) == 1, "the page renders once, in the shell's slot"


# ---------------------------------------------------------------------------
# SH13 -- on a phone the shell header holds Stop all outside the drawer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", ("static", "mounted"))
def test_sh13_on_a_phone_the_shell_header_holds_stop_outside_the_drawer(half):
    if half == "static":
        assert _PHONE_HEADER in _imports(_APP_SHELL) and _mounts(_APP_SHELL, "PhoneHeader") == 1, (
            "the shell mounts its phone header"
        )
        assert _STOP_ALL in _imports(_PHONE_HEADER) and _mounts(_PHONE_HEADER, "StopAllButton") == 1, (
            "the phone header mounts Stop all"
        )
        assert re.search(r"<header\b", _markup(_PHONE_HEADER)), "the phone header is a header"
        return

    for open_ in (False, True):
        root = _shell(f"ShellPhone{'Open' if open_ else 'Shut'}", True, open_)
        openers = [
            b for b in root.iter("button") if b.has("aria-controls") and b.has("aria-expanded")
        ]
        assert len(openers) == 1 and _name(openers[0]), (
            f"phone: one named opener controls the drawer: {[b.attrs for b in openers]}"
        )
        opener = openers[0]
        assert opener.get("aria-expanded") == ("true" if open_ else "false"), opener.attrs
        drawers = [e for e in root.iter() if e.get("id") == opener.get("aria-controls")]
        if open_:
            assert len(drawers) == 1 and any(e.has("data-collapsed") for e in drawers[0].iter()), (
                "phone, drawer open: the drawer holds the sidebar"
            )
        headers = [h for h in root.iter("header") if any(_inside(s, h) for s in _stops(root))]
        outside = [
            stop for stop in _stops(root)
            if not any(_inside(stop, drawer) for drawer in drawers)
            and any(_inside(stop, header) for header in headers)
        ]
        assert len(outside) == 1, (
            f"phone, drawer {'open' if open_ else 'shut'}: the header holds Stop all, outside "
            f"the drawer: {len(outside)}"
        )
        assert any(_inside(opener, header) for header in headers), (
            "the opener sits in the header with Stop all"
        )


# ---------------------------------------------------------------------------
# Reading a component's style as rules
# ---------------------------------------------------------------------------
def _rules(css):
    """``[(selector, {property: value})]`` of every rule of a stylesheet; an
    at-rule's rules come with the at-rule in front of their selector."""
    css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    out = []

    def walk(text, prefix):
        at = 0
        while True:
            brace = text.find("{", at)
            if brace < 0:
                return
            head = text[at:brace].strip()
            depth, end = 1, brace + 1
            while end < len(text) and depth:
                depth += {"{": 1, "}": -1}.get(text[end], 0)
                end += 1
            body = text[brace + 1:end - 1]
            if head.startswith("@"):
                walk(body, f"{prefix}{head} ")
            else:
                declarations = {}
                for part in body.split(";"):
                    if ":" in part:
                        key, value = part.split(":", 1)
                        declarations[key.strip().lower()] = " ".join(value.split())
                out.append((f"{prefix}{head}", declarations))
            at = end

    walk(css, "")
    return out


# ---------------------------------------------------------------------------
# SH14 -- the stop's confirmation is never clipped; the approvals drawer
# holds the stop
# ---------------------------------------------------------------------------
def _confirm_positions(css):
    return [
        declarations["position"] for selector, declarations in _rules(css)
        if ".oo-stop-confirm" in selector and "position" in declarations
    ]


@pytest.mark.parametrize("half", ("clip", "dialog"))
def test_sh14_the_stop_confirmation_is_never_clipped_and_the_approvals_drawer_holds_the_stop(half):
    if half == "clip":
        sample = (
            ".oo-stop { position: relative; }\n"
            ".oo-stop[data-placement='rail'] .oo-stop-confirm { position: absolute; left: calc(100% + 8px); }\n"
        )
        assert _confirm_positions(sample) == ["absolute"], "the census reads the confirmation's position"
        css = _style(_STOP_ALL)
        positions = _confirm_positions(css)
        assert positions and set(positions) == {"fixed"}, (
            f"the confirmation is fixed to the viewport, so no clipping box of the rail, the card or a "
            f"dialog's panel cuts it: {positions}"
        )
        anchored = [
            f"{selector} {prop}: {value}" for selector, declarations in _rules(css)
            if ".oo-stop-confirm" in selector
            for prop, value in declarations.items()
            if prop in ("top", "right", "bottom", "left", "inset") and "%" in value
        ]
        assert not anchored, f"no rule anchors the confirmation to its box: {anchored}"
        script = _script(_STOP_ALL)
        assert re.search(
            r"\bimport\s*\{[^}]*\bcomputePosition\b[^}]*\}\s*from\s*['\"]@floating-ui/dom['\"]", script,
        ) and re.search(
            r"\bcomputePosition\s*\(\s*[\w$.]+\s*,\s*[\w$.]+\s*,\s*\{[^{}]*\bstrategy\s*:\s*['\"]fixed['\"]", script,
        ), "the confirmation is placed at run time from its button, as fixed"
        assert _calls(script, "autoUpdate"), "and follows the button while it is open"
        return

    root = _render(_wrapper("ApprovalsDrawerStop", _APPROVALS_DRAWER, [_estop(True)]))
    dialogs = list(root.iter("dialog"))
    assert len(dialogs) == 1, f"the approvals drawer is one dialog: {len(dialogs)}"
    inside = [stop for stop in _stops(root) if _inside(stop, dialogs[0])]
    assert len(inside) == 1, (
        "the approvals drawer holds Stop all: a modal dialog makes the rest of the page inert, "
        f"and the stop outside it cannot be reached: {[_name(b) for b in dialogs[0].iter('button')]}"
    )


# ---------------------------------------------------------------------------
# SH15 -- nothing but a dialog covers the phone header
# ---------------------------------------------------------------------------
_ABOVE_OVERLAYS = re.compile(r"calc\(\s*var\(\s*--oo-z-overlay\s*\)\s*\+\s*[1-9]\d*\s*\)")
_PANEL_CLOSE = re.compile(
    r"<IconButton\b(?=[^>]*\blabel\s*=\s*[\"']Close panel[\"'])(?=[^>]*\bon:click\s*=\s*\{\s*closePanel\s*\})[^>]*>"
)


def _panel_markup(text):
    """The markup between the chat frame's ``<SidePanel`` and its end."""
    start = text.find("<SidePanel")
    end = text.find("</SidePanel>", start)
    return text[start:end] if start >= 0 and end > start else ""


@pytest.mark.parametrize("half", ("layer", "close"))
def test_sh15_nothing_but_a_dialog_covers_the_phone_header(half):
    if half == "close":
        sample = (
            "<SidePanel label=\"x\" overlay={$isPhone}>\n"
            "\t{#if $isPhone}<IconButton icon=\"x\" size=\"lg\" label=\"Close panel\" on:click={closePanel} />{/if}\n"
            "</SidePanel>\n"
        )
        assert len(_PANEL_CLOSE.findall(_panel_markup(sample))) == 1, "the census reads a named close control"
        inside = _panel_markup(_markup(_CHAT_LAYOUT))
        assert inside, "the chat frame hosts its panels in the side panel primitive"
        closes = _PANEL_CLOSE.findall(inside)
        assert len(closes) == 1 and re.search(r"\{#if\s+\$isPhone\s*\}", inside), (
            "on a phone the panel, standing over the page, holds a named control that closes it: "
            f"{closes}"
        )
        return

    assert _ABOVE_OVERLAYS.fullmatch("calc(var(--oo-z-overlay) + 10)") and not _ABOVE_OVERLAYS.fullmatch(
        "var(--oo-z-elevated)"
    ), "the census reads a layer above the overlays"
    layers = [
        declarations.get("z-index", "") for selector, declarations in _rules(_style(_PHONE_HEADER))
        if selector.strip() == ".oo-phone-header"
    ]
    assert layers and _ABOVE_OVERLAYS.fullmatch(layers[0]), (
        f"the phone header sits above every layer that is not modal: {layers}"
    )
    panel = [
        declarations["position"] for selector, declarations in _rules(_style(_SIDE_PANEL))
        if "data-overlay" in selector and "position" in declarations
    ]
    assert panel == ["absolute"], (
        f"a side panel on a phone stands over the page's frame, never fixed to the viewport: {panel}"
    )
    frame = [
        declarations.get("position") for selector, declarations in _rules(_style(_CHAT_LAYOUT))
        if selector.strip() == ".oo-chat-frame"
    ]
    assert frame == ["relative"], f"the chat frame is the frame its panel stands over: {frame}"


# ---------------------------------------------------------------------------
# SH16 -- pending approvals show in every state of the shell
# ---------------------------------------------------------------------------
def _approvals(root):
    return [
        b for b in root.iter("button")
        if b.get("aria-haspopup") == "dialog" and re.search(r"\bpending approvals?\b", _name(b), re.I)
    ]


_SHELL_STATES = (("desktop, expanded", False, True), ("desktop, rail", False, False), ("phone", True, False))


def test_sh16_pending_approvals_show_in_every_state_of_the_shell():
    for pending in (2, 0):
        for state, phone, open_ in _SHELL_STATES:
            name = f"ShellApprovals{pending}{state.replace(',', '').replace(' ', '')}"
            root = _render(_wrapper(name, _APP_SHELL, [
                f"isPhone.set({'true' if phone else 'false'});",
                f"sidebarOpen.set({'true' if open_ else 'false'});",
                _estop(True), _mode("daily"), _backend(True, "Ollama", True),
                f"pendingApprovals.set({pending});",
            ], body='<p id="oo-page-probe">page</p>'))
            shown = _approvals(root)
            if not pending:
                assert not shown, f"{state}: nothing waits, no approvals control shows: {len(shown)}"
                continue
            assert len(shown) == 1, (
                f"{state}: one control opens the approvals drawer while {pending} wait: "
                f"{[_name(b) for b in root.iter('button')]}"
            )
            assert str(pending) in _visible(shown[0]), (
                f"{state}: the count is visible: {_visible(shown[0])!r}"
            )


# ---------------------------------------------------------------------------
# SH17 -- the safety controls are 44 px targets on a touch screen
# ---------------------------------------------------------------------------
def _literal_sizes(path, text=None):
    """The sizes the ``<Button`` tags of a component write as literals."""
    return [
        match.group(2) for tag in re.findall(r"<Button\b[^>]*>", _markup(path, text))
        for match in [re.search(r"""\bsize\s*=\s*(["'])(\w+)\1""", tag)] if match
    ]


def _sizes(root):
    return [b.get("data-size") for b in root.iter("button")]


@pytest.mark.parametrize("half", ("stop", "pill", "links"))
def test_sh17_the_safety_controls_are_forty_four_pixel_targets_on_a_touch_screen(half):
    if half == "stop":
        sample = '<Button size="sm">a</Button><Button size={touch ? "lg" : "sm"}>b</Button>'
        assert _literal_sizes("sample.svelte", sample) == ["sm"], "the census reads a literal size"
        literal = _literal_sizes(_STOP_ALL)
        assert not literal, (
            f"every button of the stop, its confirmation's actions and Resume included, takes its "
            f"size from the touch rule: {literal}"
        )
        for state, sets in (("available", [_estop(True)]), ("stopped", [_estop(True, stopped=True)])):
            root = _render(_wrapper(f"StopHeader{state.title()}", _STOP_ALL, sets, attrs='placement="header"'))
            sizes = _sizes(root)
            assert sizes and set(sizes) == {"lg"}, f"phone header, {state}: every button is 44 px: {sizes}"
        return

    if half == "pill":
        pending = "pendingApprovals.set(1);"
        for name, component, attrs in (
            ("PillPhoneHeader", _PHONE_HEADER, 'title="Chats" open={false} drawer="oo-drawer"'),
            ("PillLargeCard", _STATUS_CARD, "large"),
        ):
            root = _render(_wrapper(name, component, [
                _estop(True), _mode("daily"), _backend(True, "Ollama", True), pending,
            ], attrs=attrs))
            shown = _approvals(root)
            assert len(shown) == 1 and shown[0].get("data-size") == "lg", (
                f"{component}: the approvals control is a 44 px target: "
                f"{[(b.get('data-size'), _name(b)) for b in shown]}"
            )
        return

    brand = [
        declarations.get("min-height") for selector, declarations in _rules(_style(_SIDEBAR))
        if "data-phone" in selector and ".oo-brand" in selector
    ]
    assert "44px" in brand, f"the drawer's brand link is a 44 px target: {brand}"
    card = {
        subject: [
            declarations.get("min-height") for selector, declarations in _rules(_style(_STATUS_CARD))
            if "data-large" in selector and subject in selector
        ]
        for subject in (".oo-card-mode", ".oo-sec-badge")
    }
    assert all("44px" in heights for heights in card.values()), (
        f"on a touch screen the card's links are 44 px targets: {card}"
    )


# ---------------------------------------------------------------------------
# SH18 -- the shell keeps clear of the safe areas at every width
# ---------------------------------------------------------------------------
def _insets(css, subject):
    """The safe-area insets the rules of ``subject`` read outside the phone's branch."""
    found = set()
    for selector, declarations in _rules(css):
        if subject in selector and "data-phone='true'" not in selector.replace('"', "'"):
            for value in declarations.values():
                found |= set(re.findall(r"env\(\s*safe-area-inset-(top|right|bottom|left)\b", value))
    return found


def test_sh18_the_shell_keeps_clear_of_the_safe_areas_at_every_width():
    sample = (
        ".oo-shell-side { padding: env(safe-area-inset-top, 0px) 0 0 env(safe-area-inset-left, 0px); }\n"
        ".oo-shell[data-phone='true'] .oo-shell-side { padding-bottom: env(safe-area-inset-bottom, 0px); }\n"
    )
    assert _insets(sample, ".oo-shell-side") == {"top", "left"}, "the census reads the insets, the phone's apart"
    css = _style(_APP_SHELL)
    side, sheet = _insets(css, ".oo-shell-side"), _insets(css, ".oo-sheet")
    assert {"top", "bottom", "left"} <= side, (
        f"the sidebar keeps clear of the top, bottom and left insets at every width: {sorted(side)}"
    )
    assert {"top", "right", "bottom"} <= sheet, (
        f"the sheet keeps clear of the top, right and bottom insets at every width: {sorted(sheet)}"
    )


# ---------------------------------------------------------------------------
# SH19 -- the phone drawer is modal while open
# ---------------------------------------------------------------------------
_function_body = _nav._function_body


@pytest.mark.parametrize("half", ("inert", "focus"))
def test_sh19_the_phone_drawer_is_modal_while_open(half):
    if half == "focus":
        sample = "function openDrawer() { tick().then(() => closer.focus()); }\nfunction x() {}"
        assert ".focus()" in _function_body(sample, "openDrawer"), "the census reads a function's body"
        script = _script(_APP_SHELL)
        into = _function_body(script, "focusIntoDrawer")
        back = _function_body(script, "focusBackToOpener")
        assert re.search(r"\.focus\(\s*\)", into), f"focus moves into the drawer when it opens: {into!r}"
        assert re.search(r"\.focus\(\s*\)", back) and "aria-controls" in back, (
            f"and back to the opener that controls it when it shuts: {back!r}"
        )
        assert _calls(script, "focusIntoDrawer") and _calls(script, "focusBackToOpener"), (
            "both are called"
        )
        return

    for open_ in (True, False):
        root = _shell(f"ShellPhoneModal{'Open' if open_ else 'Shut'}", True, open_)
        inert = [e for e in root.iter() if e.has("inert")]
        if not open_:
            assert not inert, f"phone, drawer shut: nothing is inert: {[e.attrs for e in inert]}"
            continue
        opener = [b for b in root.iter("button") if b.has("aria-controls") and b.has("aria-expanded")]
        assert len(opener) == 1 and any(_inside(opener[0], e) or opener[0] is e for e in inert), (
            "phone, drawer open: the header behind the drawer is inert"
        )
        sheets = [e for e in root.iter() if "oo-sheet" in e.classes()]
        assert len(sheets) == 1 and any(_inside(sheets[0], e) or sheets[0] is e for e in inert), (
            "phone, drawer open: the page behind the drawer is inert"
        )
        panels = [
            e for e in root.iter()
            if e.get("role") == "dialog" and e.get("aria-modal") == "true" and (e.get("aria-label") or "").strip()
        ]
        assert len(panels) == 1 and not any(_inside(panels[0], e) or panels[0] is e for e in inert), (
            f"phone, drawer open: the drawer is a labelled modal dialog, not inert: {len(panels)}"
        )
        closers = [b for b in panels[0].iter("button") if _name(b) == "Close navigation"]
        assert len(closers) == 1, "the drawer holds a named control that closes it"
        assert [s for s in _stops(root) if _inside(s, panels[0])], "and the drawer holds Stop all"


# ---------------------------------------------------------------------------
# SH20 -- the stop speaks once
# ---------------------------------------------------------------------------
def test_sh20_the_stop_speaks_once():
    said = (
        "estop.update((s) => ({ ...s, available: true, stopped: false, "
        "announce: 'Emergency stop engaged', error: 'The emergency stop request failed' }));"
    )
    root = _render(_wrapper("StopOnce", _STOP_ALL, [said]))
    twice = _render(_plant("StopTwiceBoth", (
        "<script>\n"
        "\timport { estop } from '$lib/stores/estop';\n"
        "\timport StopAllButton from '$lib/components/layout/StopAllButton.svelte';\n"
        f"\t{said}\n"
        "</script>\n"
        "<StopAllButton placement=\"header\" />\n<StopAllButton placement=\"card\" />\n"
    )))
    for label, tree in (("one copy", root), ("two copies", twice)):
        regions = [e for e in tree.iter() if e.get("aria-live")]
        assert regions, f"{label}: the control keeps its polite live region"
        spoken = [" ".join(e.text().split()) for e in regions if e.text().strip()]
        assert not spoken, (
            f"{label}: a copy the reader did not act on announces nothing: {spoken}"
        )
        alerts = [e for e in tree.iter() if e.get("role") == "alert"]
        assert not alerts, f"{label}: nor shows the error as an alert: {len(alerts)}"
    live = [e for e in twice.iter() if e.get("aria-live")]
    assert len(live) == 2 and "$estop.announce" in _markup(_STOP_ALL), (
        "each copy holds its region, which reads the store's announcement"
    )
    stopped = _render(_wrapper("StopStoppedQuiet", _STOP_ALL, [_estop(True, stopped=True)]))
    statuses = [e for e in stopped.iter() if e.get("role") == "status" and list(e.iter("button"))]
    assert not statuses, "the stopped pill is no live region around its Resume"
    names = [_name(b) for b in [*_stops(root), *stopped.iter("button")]]
    assert any(n.startswith("Stop all (emergency") for n in names) and any(
        n.startswith("Resume from the emergency") for n in names
    ), f"what a screen reader hears reads as words, a space before the hidden tail: {names}"


# ---------------------------------------------------------------------------
# SH21 -- the stop reads one at a time, and an unread status is unknown
# ---------------------------------------------------------------------------
_ESTOP_DRIVER = r"""
const state = await import(process.env.OO_ESTOP_STATE);
const clause = process.argv[2];
const base = { available: false, stopped: true, busy: false, error: 'kept', announce: 'kept' };
const out = {};
out.unread = state.afterRead(base, null);
out.read = state.afterRead({ ...base, available: null }, { available: true, stopped: false });
out.base = base;
let calls = 0;
const releases = [];
const slow = state.oneAtATime(() => { calls += 1; return new Promise((r) => { releases.push(r); }); });
const first = slow();
const second = slow();
out.same = first === second;
out.callsWhileOpen = calls;
releases.splice(0).forEach((release) => release('done'));
await first;
await second;
const third = slow();
out.callsAfter = calls;
releases.splice(0).forEach((release) => release('done'));
await third;
let failing = 0;
const broken = state.oneAtATime(() => { failing += 1; return Promise.reject(new Error('no')); });
await broken().catch(() => {});
await broken().catch(() => {});
out.retried = failing;
console.log('RESULT ' + JSON.stringify(out));
console.log('PASS ' + clause);
"""


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_sh21_the_stop_reads_one_at_a_time_and_an_unread_status_is_unknown(half):
    if half == "wiring":
        store = _script(_ESTOP_STORE)
        for name in ("afterRead", "oneAtATime"):
            assert _imports_name(_ESTOP_STORE, name, _ESTOP_STATE) and _calls(store, name), (
                f"the estop store reads through {name}"
            )
        assert re.search(r"\bafterRead\s*\(\s*[\w$.]+\s*,\s*null\s*\)", store), (
            "a status that cannot be read goes through afterRead as unknown"
        )
        assert re.search(r"\brefreshEstop\s*=\s*oneAtATime\s*\(", store), (
            "every read of the status is one read at a time"
        )
        return

    from _frontend import run_ts

    out = run_ts({"OO_ESTOP_STATE": _ESTOP_STATE}, _ESTOP_DRIVER, "estop")
    result = __import__("json").loads(
        next(line for line in out.splitlines() if line.startswith("RESULT "))[len("RESULT "):]
    )
    unread = result["unread"]
    assert unread["available"] is None, (
        f"a status that cannot be read makes the stop unknown, and enabled: {unread}"
    )
    assert {k: unread[k] for k in ("stopped", "busy", "error", "announce")} == {
        k: result["base"][k] for k in ("stopped", "busy", "error", "announce")
    }, f"and keeps the rest of what was known: {unread}"
    assert (result["read"]["available"], result["read"]["stopped"]) == (True, False), (
        f"a status read is taken as the server says it: {result['read']}"
    )
    assert result["same"] and result["callsWhileOpen"] == 1, (
        f"a second ask while a read is open joins it: {result}"
    )
    assert result["callsAfter"] == 2 and result["retried"] == 2, (
        f"once a read has answered or failed, the next ask reads again: {result}"
    )


# ---------------------------------------------------------------------------
# SH22 -- the stop's label reaches 4.5:1 on every ground it is drawn on
# ---------------------------------------------------------------------------
def _stop_grounds(css):
    """``[(selector, ground, ink)]``: every rule of the stop's style that
    paints a ground, with the ink its text has there (its own, or that of
    the rule it refines, the same selector with no state)."""
    rules = _rules(css)
    inks = {selector.strip(): declarations.get("color") for selector, declarations in rules}
    out = []
    for selector, declarations in rules:
        ground = declarations.get("background-color") or declarations.get("background")
        if not ground:
            continue
        plain = re.sub(r":(?:hover|focus-visible|active|not\([^)]*\))", "", selector).strip()
        ink = declarations.get("color") or inks.get(plain)
        if ink:
            out.append((selector.strip(), ground, ink))
    return out


def _below(grounds):
    import test_design_tokens_contracts as _ds
    from _colour import contrast, over

    tokens, palettes = _ds._derivation(), _ds._palettes()
    low = []
    for pid, (roles, scheme) in palettes.items():
        resolver = _ds._colour_resolver(tokens, roles, scheme, pid)
        for selector, ground, ink in grounds:
            painted = resolver.colour(ground)
            worst = min(
                contrast(over(resolver.colour(ink), solid), solid)
                for solid in (
                    over(painted, resolver.colour(under)) if painted[3] < 1 else painted
                    for under in ("var(--oo-bg-surface)", "var(--oo-bg-base)")
                )
            )
            if worst < 4.5:
                low.append(f"{pid}: {selector}: {ink} on {ground}: {worst:.2f}")
    return low


def test_sh22_the_stop_label_reaches_four_and_a_half_to_one_on_every_ground():
    sample = (
        ".a :global(.oo-btn) { background-color: var(--oo-bg-subtle); color: var(--oo-fg-stop); }\n"
        ".a :global(.oo-btn:hover:not(:disabled)) { background-color: "
        "color-mix(in srgb, var(--oo-fg-stop) 8%, var(--oo-bg-subtle)); }\n"
    )
    caught = _below(_stop_grounds(sample))
    assert len(_stop_grounds(sample)) == 2 and caught and all(":hover" in f for f in caught), (
        f"the census pairs a state's ground with the ink it keeps, and a low one is caught: {caught}"
    )
    grounds = _stop_grounds(_style(_STOP_ALL))
    selectors = " ".join(selector for selector, _, _ in grounds)
    assert ":hover" in selectors and ".oo-stop-pill" in selectors, (
        f"the census reads the stop at rest, under the pointer and stopped: {grounds}"
    )
    low = _below(grounds)
    assert not low, "the stop's label under 4.5:1:\n  " + "\n  ".join(low)


# ---------------------------------------------------------------------------
# SH23 -- an unknown address is answered inside the shell
# ---------------------------------------------------------------------------
_REST = re.compile(r"^\[\.\.\.\w+\]$")


def _catch_alls(pages):
    """The page modules of a rest route sitting straight under the layout of
    both spaces."""
    return [
        path for path in pages
        if _nav._route_groups(path) == ["(app)"]
        and len(PurePosixPath(path).relative_to(_ROUTES).parts) == 3
        and _REST.match(PurePosixPath(path).parts[-2])
        and PurePosixPath(path).name in ("+page.ts", "+page.js")
    ]


def test_sh23_an_unknown_address_is_answered_inside_the_shell():
    sample = [f"{_ROUTES}/(app)/[...missing]/+page.ts", f"{_ROUTES}/(app)/(use)/[...x]/+page.ts",
              f"{_ROUTES}/[...y]/+page.ts"]
    assert _catch_alls(sample) == sample[:1], "the census reads a catch-all under the shell's layout alone"
    found = _catch_alls(_nav._page_files())
    assert len(found) == 1, (
        f"one catch-all page answers every address no page serves, inside the shell: {found}"
    )
    beside = PurePosixPath(found[0]).parent / "+page.svelte"
    assert not (REPO / str(beside)).is_file(), "it renders nothing of its own"
    code = _code(found[0])
    assert re.search(r"\bimport\s*\{[^}]*\berror\b[^}]*\}\s*from\s*['\"]@sveltejs/kit['\"]", code) and re.search(
        r"(?<![\w$.])error\(\s*404\b", code,
    ), f"it throws a 404, which the shell's error page draws: {code!r}"
    assert (REPO / _APP_ERROR).is_file() and re.search(r"\bstatus\s*===\s*404\b", _code(_APP_ERROR)), (
        "the shell's error page tells a missing page from a failed one"
    )
    others = [path for path in _nav._page_files() if _REST.match(PurePosixPath(path).parts[-2])]
    assert others == found, f"no other catch-all shadows it: {others}"


# ---------------------------------------------------------------------------
# SH24 -- the network page shows the server's reachability
# ---------------------------------------------------------------------------
def test_sh24_the_network_page_shows_the_server_reachability():
    sample = "{#if space === 'workshop' && section === 'network'}\n\t<NetworkReachability />\n{/if}"
    network_only = re.compile(
        r"\{#if\s+space\s*===\s*'workshop'\s*&&\s*section\s*===\s*'network'\s*\}\s*<NetworkReachability\s*/>"
    )
    assert network_only.search(sample), "the census reads the network page's own block"
    assert (REPO / _REACHABILITY).is_file() and _REACHABILITY in _imports(_HUB) and network_only.search(
        _markup(_HUB)
    ), "the network page, and it alone, shows the reachability"
    code = _code(_REACHABILITY)
    assert _imports_name(_REACHABILITY, "getNetworkStatus", _NETWORK_API) and _calls(code, "getNetworkStatus"), (
        "it reads the server's reachability through the API layer"
    )
    assert not re.search(r"(?<![\w$.])set(?:Interval|Timeout)\s*\(", code), (
        "when the page is shown and on request, never on a timer"
    )
    shown = _markup(_REACHABILITY)
    for word in ("Refresh", "latency", "queue"):
        assert re.search(rf"\b{word}\b", shown, re.I), f"it shows {word!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly"]))
