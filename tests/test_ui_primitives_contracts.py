#!/usr/bin/env python3
"""Contracts for the design-system primitives and the inline icon set.

Every control the rebuilt surfaces draw comes from ``frontend/src/lib/ds``:
a button that forwards its ARIA state and takes a pill or round shape, an
icon button that cannot render without an accessible name, a toggle chip
whose pressed state is a check and not a colour, a menu whose keys are a pure
module, a native checkbox with a visible label, and a side panel that is a
region beside the page, never a dialog over it. Icons are drawn once, as
compact path data in ``ds/icons.ts``, and ``ds/Icon.svelte`` draws that set
first, at a stroke of 1.5, falling back to the older icon package only for a
name the set has not drawn yet.

The props each primitive takes are named here, because the contracts read
them:

  * ``Button`` -- ``pressed``, ``expanded``, ``haspopup`` and ``controls``
    become ``aria-pressed``, ``aria-expanded``, ``aria-haspopup`` and
    ``aria-controls`` (each omitted when not given); ``shape`` is ``rect``
    (the default), ``pill`` or ``round``, carried as ``data-shape``;
    ``iconOnly`` draws an icon alone, and without a non-blank ``ariaLabel``
    refuses to render, with an error naming the accessible name.
  * ``IconButton`` -- ``icon`` and ``label`` (the accessible name, as
    ``aria-label``); ``size`` is ``md`` (a 36 px box, the desktop size) or
    ``lg`` (44 px, the phone target). A missing or blank ``label`` refuses
    to render, with an error naming the accessible name.
  * ``ToggleChip`` -- ``label``, ``pressed`` and ``note`` (a visible word
    beside the label, such as "auto").
  * ``Menu`` -- ``label`` (a blank one refuses to render) and ``items``;
    the keys of its list go through ``menuKey(key, active, count,
    disabled)`` in ``menuKeys.ts``, which returns ``{active, close,
    handled}``, and the keys that open it from its trigger through
    ``triggerKey(key, count, disabled)``, which returns the item to open on
    (-1 for none) or null for a key that does not open it.
  * ``Checkbox`` -- ``label``, ``checked``, ``indeterminate`` and
    ``description``.
  * ``SidePanel`` -- ``label`` and ``width``; its resize handle's keys go
    through ``resizeKey(key, width, bounds)`` in ``sidePanelSize.ts``, a
    drag through ``dragWidth(startWidth, startX, x, bounds)``, a release of
    the pointer through ``releaseWidth(startWidth, startX, x, bounds)`` (a
    press that did not drag steps the width to the next of three stops),
    and a width from anywhere else through ``clampWidth(width, bounds)``.

The server-rendering halves render components through
``tests/_frontend.ssr()`` and read what the template emits when compiled for
the server. The app is client-rendered only, so they prove the template's
output, not what a browser does with it: the menu's keyboard, the panel's
resize and the icons' drawing are owed to the machine. The node halves run
the pure modules under Node's type stripping through
``tests/_frontend.run_ts``; each is paired with a wiring half that reads the
component using it.

  * PR1 -- ``Button`` forwards ``aria-pressed``, ``aria-expanded``,
    ``aria-haspopup`` and ``aria-controls``, and omits each one not given;
    it renders the pill and the round shapes, both fully rounded, the round
    one square.
  * PR2 -- ``IconButton`` names itself by its label and refuses to render
    without one; its box is 36 px at the desktop size and 44 px at the phone
    size.
  * PR3 -- ``ToggleChip`` carries ``aria-pressed``, draws the check only
    when pressed, and shows its note as visible text.
  * PR4 -- ``menuKey`` moves by the arrows (wrapping, skipping disabled
    items), Home and End, closes on Escape and on Tab, and leaves every
    other key alone; ``triggerKey`` opens the menu from its trigger on the
    down and up arrows alone; ``Menu`` renders a ``menu`` of ``menuitem``
    entries and routes its keys through the two functions alone.
  * PR5 -- ``Checkbox`` is one native checkbox with a visible, associated
    label, whose name is the label alone: a description stands outside the
    label and is the input's description. Its mixed state is the input's
    own ``indeterminate``.
  * PR6 -- ``SidePanel`` is a labelled complementary region, never a dialog
    (no modal dialog, no ``inert``, not hosted in ``Modal``); its resize
    handle is a focusable vertical separator that says its width in pixels
    and the region it sizes, whose keys go through ``resizeKey``, whose
    drag goes through ``dragWidth`` and whose release goes through
    ``releaseWidth``: a single press, with no drag, steps the width, so a
    pointer resizes it without dragging. The app shell hosts its right
    panel in it and holds no resize logic of its own.
  * PR7 -- ``Icon`` draws at a stroke of 1.5 by default; every icon name a
    held file writes resolves in the inline set; the inline set is drawn
    first and the fallback only for a name it lacks; the set is well formed:
    every name kebab-case, the design's thirty drawings present, every
    path compact (no number ending in a point, which would let a command
    letter follow a point) and inside the 24 unit box, and no path the
    public clean guard's detector charges.
  * PR8 -- a pressed control's check reaches 3:1 on the ground it is drawn
    on, in every palette and every button variant, beside a label and at
    the corner of an icon alone, and its line is at least a pixel wide at
    the size it is drawn; a pressed quiet button and a pressed chip keep
    their tint under the pointer. The cascade is read from the CSS the
    server compiled, over the elements it rendered.
  * PR9 -- no icon-only control renders without a name: a button that
    draws an icon alone, and a menu, refuse a missing or blank name as the
    icon button does.
  * PR10 -- the chat frame (the chats' layout) hosts its side panels in
    ``SidePanel``, giving it its width and taking its resizes, and holds no
    resize logic and no panel of its own; the shell, which is every page's
    frame, hosts no side panel and holds no resize logic either.

Local-only (the public distribution ships no tests). Needs Node >= 22.6 and
``frontend/node_modules``; without them the helpers raise, and so do the
contracts.
"""

import functools
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_design_tokens_contracts as _ds  # noqa: E402
from _frontend import held_files, read, run_ts, ssr  # noqa: E402
from _isolation import isolate  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file. The first server-rendering contract may start the
# session's server when this suite runs alone.
BUDGET_S = {
    "test_pr1_a_button_forwards_its_aria_state_and_takes_pill_and_round_shapes[aria]": 6.0,
    "test_pr1_a_button_forwards_its_aria_state_and_takes_pill_and_round_shapes[shape]": 1.0,
    "test_pr2_an_icon_button_needs_its_name_and_sizes_its_box[name]": 1.0,
    "test_pr2_an_icon_button_needs_its_name_and_sizes_its_box[size]": 1.0,
    "test_pr3_a_toggle_chip_says_pressed_with_a_check_and_shows_its_note[pressed]": 2.0,
    "test_pr3_a_toggle_chip_says_pressed_with_a_check_and_shows_its_note[note]": 1.0,
    "test_pr4_a_menu_moves_by_its_keys_through_one_pure_module[node]": 2.0,
    "test_pr4_a_menu_moves_by_its_keys_through_one_pure_module[wiring]": 1.0,
    "test_pr5_a_checkbox_is_a_native_input_with_a_visible_label_and_a_mixed_state[label]": 1.0,
    "test_pr5_a_checkbox_is_a_native_input_with_a_visible_label_and_a_mixed_state[mixed]": 1.0,
    "test_pr6_a_side_panel_is_a_region_beside_the_page_resized_by_keys[landmark]": 1.0,
    "test_pr6_a_side_panel_is_a_region_beside_the_page_resized_by_keys[resize]": 2.0,
    "test_pr6_a_side_panel_is_a_region_beside_the_page_resized_by_keys[host]": 1.0,
    "test_pr7_icons_are_drawn_from_the_inline_set_at_a_stroke_of_one_and_a_half[stroke]": 1.0,
    "test_pr7_icons_are_drawn_from_the_inline_set_at_a_stroke_of_one_and_a_half[held]": 2.0,
    "test_pr7_icons_are_drawn_from_the_inline_set_at_a_stroke_of_one_and_a_half[inline]": 1.0,
    "test_pr7_icons_are_drawn_from_the_inline_set_at_a_stroke_of_one_and_a_half[set]": 2.0,
    "test_pr8_a_pressed_check_is_seen_on_its_ground_and_the_pressed_tint_holds_under_the_pointer[check]": 2.0,
    "test_pr8_a_pressed_check_is_seen_on_its_ground_and_the_pressed_tint_holds_under_the_pointer[hover]": 1.0,
    "test_pr9_no_icon_only_control_renders_without_a_name": 1.0,
    "test_pr10_the_chat_frame_hosts_its_side_panels_in_the_primitive": 1.0,
}

_DS = "frontend/src/lib/ds"
_BUTTON = f"{_DS}/Button.svelte"
_ICON_BUTTON = f"{_DS}/IconButton.svelte"
_TOGGLE_CHIP = f"{_DS}/ToggleChip.svelte"
_MENU = f"{_DS}/Menu.svelte"
_MENU_KEYS = f"{_DS}/menuKeys.ts"
_CHECKBOX = f"{_DS}/Checkbox.svelte"
_SIDE_PANEL = f"{_DS}/SidePanel.svelte"
_SIDE_PANEL_SIZE = f"{_DS}/sidePanelSize.ts"
_ICON = f"{_DS}/Icon.svelte"
_ICONS = f"{_DS}/icons.ts"

_STATE_ATTRIBUTES = ("aria-pressed", "aria-expanded", "aria-haspopup", "aria-controls")

# The design's thirty drawings, under the names the set gives them (the
# older package's name where one exists, so a call site written for it draws
# the new icon).
_DESIGN = (
    "onion", "panel-left", "panel-right", "plus", "search", "home", "chat",
    "note", "folder", "sprout", "sliders", "stop", "chevron-down",
    "chevron-right", "chevron-left", "arrow-up", "arrow-right", "check",
    "bulb", "globe", "box", "code", "copy", "retry", "branch", "chip",
    "pencil", "pin", "stop-fill", "more",
)


# ---------------------------------------------------------------------------
# Reading what a component emits
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

    def has(self, name):
        return any(key == name for key, _ in self.attrs)

    def classes(self):
        return (self.get("class") or "").split()

    def text(self):
        return "".join(
            child if isinstance(child, str) else child.text() for child in self.children
        )

    def visible_text(self):
        """The text a sighted reader sees: nothing under ``aria-hidden`` or a
        visually hidden class."""
        if self.get("aria-hidden") == "true" or {"oo-sr-only", "sr-only"} & set(self.classes()):
            return ""
        return "".join(
            child if isinstance(child, str) else child.visible_text() for child in self.children
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
    parser = _Dom()
    parser.feed(html_text)
    parser.close()
    return parser.root


def _render(path, props):
    """What a component emits compiled for the server, as a tree."""
    return _dom(ssr().render(path, props).html)


def _one(root, tag):
    found = list(root.iter(tag))
    assert len(found) == 1, f"one <{tag}> is rendered: {len(found)}"
    return found[0]


def _paths(element):
    return [path.get("d") for path in element.iter("path")]


# ---------------------------------------------------------------------------
# Reading sources: a component's style rules
# ---------------------------------------------------------------------------
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>(.*?)</style>", re.S)
_SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>(.*?)</script>", re.S)


def _style_rules(path):
    """``{normalised selector: {property: value}}`` of a component's style
    block, top-level rules only, later declarations winning."""
    css = _CSS_COMMENT.sub(" ", "\n".join(_STYLE_BLOCK.findall(read(path))))
    rules = {}
    for selector, body in re.findall(r"([^{}@]+)\{([^{}]*)\}", css):
        declared = {}
        for part in body.split(";"):
            if ":" in part:
                name, value = part.split(":", 1)
                declared[name.strip()] = " ".join(value.split())
        for single in selector.split(","):
            key = " ".join(single.replace('"', "'").split())
            rules.setdefault(key, {}).update(declared)
    return rules


def _script(path):
    return "\n".join(_SCRIPT_BLOCK.findall(read(path)))


def _markup(path):
    text = read(path)
    return _STYLE_BLOCK.sub(" ", _SCRIPT_BLOCK.sub(" ", text))


# ---------------------------------------------------------------------------
# The inline set, read through Node
# ---------------------------------------------------------------------------
_ICONS_DRIVER = r"""
const icons = await import(process.env.OO_ICONS);
const clause = process.argv[2];
const names = JSON.parse(process.env.OO_NAMES || '[]');
console.log('OO_JSON ' + JSON.stringify({
    icons: icons.ICONS,
    keys: names.map((name) => icons.iconKey(name)),
    found: names.map((name) => icons.inlineIcon(name) !== undefined),
}));
console.log('PASS ' + clause);
"""


def _json_line(out):
    for line in out.splitlines():
        if line.startswith("OO_JSON "):
            return json.loads(line[len("OO_JSON "):])
    raise AssertionError(f"the driver printed no result:\n{out}")


@functools.cache
def _inline(names=()):
    """What ``icons.ts`` holds, with ``iconKey`` and ``inlineIcon`` asked
    about ``names``."""
    out = run_ts(
        {"OO_ICONS": _ICONS}, _ICONS_DRIVER, "icons",
        env={"OO_NAMES": json.dumps(list(names))},
    )
    return _json_line(out)


def _shape_d(shape):
    return shape if isinstance(shape, str) else shape.get("d")


# ===========================================================================
# PR1 -- Button forwards its ARIA state and takes pill and round shapes
# ===========================================================================
@pytest.mark.parametrize("half", ("aria", "shape"))
def test_pr1_a_button_forwards_its_aria_state_and_takes_pill_and_round_shapes(half):
    if half == "aria":
        given = _one(_render(_BUTTON, {
            "pressed": True, "expanded": False, "haspopup": "menu", "controls": "oo-menu-1",
        }), "button")
        forwarded = {name: given.get(name) for name in _STATE_ATTRIBUTES}
        assert forwarded == {
            "aria-pressed": "true", "aria-expanded": "false",
            "aria-haspopup": "menu", "aria-controls": "oo-menu-1",
        }, f"each state the button is given is forwarded as written: {forwarded}"
        flipped = _one(_render(_BUTTON, {"pressed": False, "expanded": True}), "button")
        assert (flipped.get("aria-pressed"), flipped.get("aria-expanded")) == ("false", "true"), (
            "false is forwarded as false, not dropped"
        )
        assert not flipped.has("aria-haspopup") and not flipped.has("aria-controls"), (
            "a state not given is omitted"
        )
        plain = _one(_render(_BUTTON, {}), "button")
        leftover = [name for name in _STATE_ATTRIBUTES if plain.has(name)]
        assert not leftover, f"a plain button carries no state attribute: {leftover}"
        return

    shapes = {
        shape: _one(_render(_BUTTON, {"shape": shape}), "button").get("data-shape")
        for shape in ("pill", "round", "rect")
    }
    assert shapes == {"pill": "pill", "round": "round", "rect": "rect"}, shapes
    assert _one(_render(_BUTTON, {}), "button").get("data-shape") == "rect", (
        "the default shape is the rectangle"
    )
    rules = _style_rules(_BUTTON)
    pill = rules.get(".oo-btn[data-shape='pill']", {})
    round_ = rules.get(".oo-btn[data-shape='round']", {})
    assert pill.get("border-radius") == "var(--oo-radius-full)", (
        f"the pill is fully rounded: {pill}"
    )
    assert round_.get("border-radius") == "var(--oo-radius-full)" and (
        round_.get("aspect-ratio", "").replace(" ", "") == "1/1"
    ), f"the round button is a circle: fully rounded and square: {round_}"


# ===========================================================================
# PR2 -- IconButton needs its accessible name and sizes its box
# ===========================================================================
@pytest.mark.parametrize("half", ("name", "size"))
def test_pr2_an_icon_button_needs_its_name_and_sizes_its_box(half):
    if half == "name":
        button = _one(_render(_ICON_BUTTON, {"icon": "x", "label": "Close the panel"}), "button")
        assert button.get("aria-label") == "Close the panel", (
            f"the label is the button's accessible name: {button.attrs}"
        )
        assert button.get("type") == "button", "an icon button never submits a form by default"
        drawn = list(button.iter("svg"))
        assert len(drawn) == 1 and drawn[0].get("aria-hidden") == "true", (
            "the icon is drawn once and hidden from assistive technology"
        )
        for props in ({"icon": "x"}, {"icon": "x", "label": ""}, {"icon": "x", "label": "   "}):
            with pytest.raises(AssertionError, match="accessible name"):
                ssr().render(_ICON_BUTTON, props)
        assert re.search(r"export\s+let\s+label\s*:\s*string\s*;", _script(_ICON_BUTTON)), (
            "the label is a required prop, with no default"
        )
        return

    boxes = {
        size: _one(_render(_ICON_BUTTON, {"icon": "x", "label": "Close", **given}), "button")
        for size, given in (("default", {}), ("md", {"size": "md"}), ("lg", {"size": "lg"}))
    }
    assert {size: button.get("data-size") for size, button in boxes.items()} == {
        "default": "md", "md": "md", "lg": "lg",
    }, "the desktop size is the default"
    assert all("oo-icon-btn" in button.classes() for button in boxes.values()), (
        "the box is drawn by the icon button's own class"
    )
    rules = _style_rules(_ICON_BUTTON)
    for size, pixels in (("md", "36px"), ("lg", "44px")):
        rule = rules.get(f".oo-icon-btn[data-size='{size}']", {})
        assert (rule.get("width"), rule.get("height")) == (pixels, pixels), (
            f"the {size} box is {pixels} square: {rule}"
        )


# ===========================================================================
# PR3 -- ToggleChip says pressed with a check, and shows its note
# ===========================================================================
@pytest.mark.parametrize("half", ("pressed", "note"))
def test_pr3_a_toggle_chip_says_pressed_with_a_check_and_shows_its_note(half):
    if half == "pressed":
        check = [_shape_d(shape) for shape in _inline()["icons"]["check"]]
        on = _one(_render(_TOGGLE_CHIP, {"label": "Web search", "pressed": True}), "button")
        off = _one(_render(_TOGGLE_CHIP, {"label": "Web search", "pressed": False}), "button")
        assert (on.get("aria-pressed"), off.get("aria-pressed")) == ("true", "false"), (
            "the chip says its state as aria-pressed, in both states"
        )
        assert on.get("type") == "button" and off.get("type") == "button"
        assert all(d in _paths(on) for d in check), (
            f"the pressed chip draws the check: {_paths(on)}"
        )
        assert not any(d in _paths(off) for d in check), "the check is drawn only when pressed"
        for chip in (on, off):
            assert "Web search" in chip.visible_text(), "the label is shown in both states"
        return

    noted = _one(_render(_TOGGLE_CHIP, {"label": "Thinking", "pressed": False, "note": "auto"}), "button")
    shown = " ".join(noted.visible_text().split())
    assert "auto" in shown and "Thinking" in shown, (
        f"the note is visible text beside the label: {shown!r}"
    )
    bare = _one(_render(_TOGGLE_CHIP, {"label": "Thinking", "pressed": False}), "button")
    assert " ".join(bare.visible_text().split()) == "Thinking", (
        "without a note, the chip shows its label alone"
    )


# ===========================================================================
# PR4 -- Menu: its keys go through one pure module
# ===========================================================================
_MENU_DRIVER = r"""
const keys = await import(process.env.OO_MENU_KEYS);
const clause = process.argv[2];
const cases = JSON.parse(process.env.OO_INPUT);
const out = cases.map(([key, active, count, disabled]) =>
    disabled === null ? keys.menuKey(key, active, count) : keys.menuKey(key, active, count, disabled));
const triggers = JSON.parse(process.env.OO_TRIGGERS);
const opened = typeof keys.triggerKey !== 'function' ? 'triggerKey is not exported' :
    triggers.map(([key, count, disabled]) =>
        disabled === null ? keys.triggerKey(key, count) : keys.triggerKey(key, count, disabled));
console.log('OO_JSON ' + JSON.stringify({ out, opened }));
console.log('PASS ' + clause);
"""

_OFF = False
_MENU_CASES = (
    # (key, active, count, disabled) -> (active, close, handled)
    (("ArrowDown", 0, 3, None), (1, False, True)),
    (("ArrowDown", 2, 3, None), (0, False, True)),
    (("ArrowUp", 0, 3, None), (2, False, True)),
    (("ArrowUp", 2, 3, None), (1, False, True)),
    (("ArrowDown", -1, 3, None), (0, False, True)),
    (("ArrowUp", -1, 3, None), (2, False, True)),
    (("Home", 2, 3, None), (0, False, True)),
    (("End", 0, 3, None), (2, False, True)),
    (("Escape", 1, 3, None), (1, True, True)),
    (("Tab", 1, 3, None), (1, True, False)),
    (("a", 1, 3, None), (1, False, False)),
    (("Enter", 1, 3, None), (1, False, False)),
    (("ArrowLeft", 1, 3, None), (1, False, False)),
    (("ArrowRight", 1, 3, None), (1, False, False)),
    ((" ", 1, 3, None), (1, False, False)),
    (("ArrowDown", 0, 4, [_OFF, True, _OFF, _OFF]), (2, False, True)),
    (("ArrowUp", 2, 4, [_OFF, True, _OFF, _OFF]), (0, False, True)),
    (("ArrowDown", 3, 4, [_OFF, True, _OFF, _OFF]), (0, False, True)),
    (("ArrowUp", 0, 4, [_OFF, _OFF, _OFF, True]), (2, False, True)),
    (("Home", 2, 3, [True, _OFF, _OFF]), (1, False, True)),
    (("End", 0, 3, [_OFF, _OFF, True]), (1, False, True)),
    (("ArrowDown", -1, 2, [True, True]), (-1, False, True)),
    (("ArrowDown", -1, 0, None), (-1, False, True)),
    (("End", -1, 0, None), (-1, False, True)),
    (("Escape", -1, 0, None), (-1, True, True)),
)
# (key, count, disabled) -> the item the menu opens on (-1: open, no item
# can take focus), or None: the key does not open the menu (Enter and Space
# open it as a click does, natively).
_TRIGGER_CASES = (
    (("ArrowDown", 3, None), 0),
    (("ArrowUp", 3, None), 2),
    (("ArrowDown", 3, [True, _OFF, _OFF]), 1),
    (("ArrowUp", 3, [_OFF, _OFF, True]), 1),
    (("ArrowDown", 2, [True, True]), -1),
    (("ArrowDown", 0, None), -1),
    (("Home", 3, None), None),
    (("End", 3, None), None),
    (("Enter", 3, None), None),
    ((" ", 3, None), None),
    (("Escape", 3, None), None),
    (("Tab", 3, None), None),
    (("a", 3, None), None),
)
_KEY_NAMES = re.compile(r"""["'`](?:ArrowDown|ArrowUp|ArrowLeft|ArrowRight|Home|End|Escape|Tab)["'`]""")


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_pr4_a_menu_moves_by_its_keys_through_one_pure_module(half):
    if half == "node":
        out = run_ts(
            {"OO_MENU_KEYS": _MENU_KEYS}, _MENU_DRIVER, "menu",
            env={
                "OO_INPUT": json.dumps([list(case) for case, _ in _MENU_CASES]),
                "OO_TRIGGERS": json.dumps([list(case) for case, _ in _TRIGGER_CASES]),
            },
        )
        answer = _json_line(out)
        got = answer["out"]
        wrong = []
        for (case, (active, close, handled)), result in zip(_MENU_CASES, got):
            expected = {"active": active, "close": close, "handled": handled}
            if result != expected:
                wrong.append(f"{case}: {result} (expected {expected})")
        assert len(got) == len(_MENU_CASES) and not wrong, (
            "menuKey moves as a menu does:\n  " + "\n  ".join(wrong)
        )
        opened = answer["opened"]
        assert isinstance(opened, list) and len(opened) == len(_TRIGGER_CASES), (
            f"triggerKey answers every key pressed on the trigger: {opened}"
        )
        wrong = [
            f"triggerKey{case}: {result} (expected {expected})"
            for (case, expected), result in zip(_TRIGGER_CASES, opened) if result != expected
        ]
        assert not wrong, (
            "the trigger opens its menu on the down and up arrows alone:\n  " + "\n  ".join(wrong)
        )
        return

    script, markup = _script(_MENU), _markup(_MENU)
    assert re.search(
        r"import\s*\{[^}]*\bmenuKey\b[^}]*\}\s*from\s*['\"]\./menuKeys(?:\.ts|\.js)?['\"]", script,
    ), "the menu imports menuKey from its pure module"
    assert re.search(r"\bmenuKey\s*\(", script), "the menu calls menuKey"
    assert re.search(
        r"import\s*\{[^}]*\btriggerKey\b[^}]*\}\s*from\s*['\"]\./menuKeys(?:\.ts|\.js)?['\"]", script,
    ) and re.search(r"\btriggerKey\s*\(", script), (
        "the menu opens from its trigger through triggerKey, from the same module"
    )
    assert _KEY_NAMES.findall("case 'Escape': break; if (key === \"Home\")") == ["'Escape'", '"Home"'], (
        "the census of key names reads a quoted key name"
    )
    assert re.search(r"""import\s*\{[^}]*\}\s*from\s*['"]@floating-ui/dom['"]""", script), (
        "the menu is placed by floating-ui"
    )
    assert re.search(r"""\brole\s*=\s*["']menu["']""", markup) and re.search(
        r"""\brole\s*=\s*["']menuitem["']""", markup,
    ), "the menu renders role=menu and role=menuitem"
    assert re.search(r"""\b(?:aria-)?haspopup\s*=\s*["']menu["']""", markup) and re.search(
        r"\b(?:aria-)?expanded\s*=\s*\{", markup,
    ), "the menu's trigger says it opens a menu, and whether it is open"
    competing = _KEY_NAMES.findall(read(_MENU))
    assert not competing, f"the menu holds no key logic of its own: {competing}"


# ===========================================================================
# PR5 -- Checkbox: a native input, a visible label, a mixed state
# ===========================================================================
def _associated_label(root, box):
    """The label element that names ``box``: a <label> around it, or one
    whose ``for`` is its id."""
    node = box.parent
    while node is not None:
        if node.tag == "label":
            return node
        node = node.parent
    ident = box.get("id")
    for label in root.iter("label"):
        if ident and label.get("for") == ident:
            return label
    return None


def _states_by_hand(text):
    """The places a source says a checkbox's state by hand."""
    return re.findall(r"aria-checked\b", text)


@pytest.mark.parametrize("half", ("label", "mixed"))
def test_pr5_a_checkbox_is_a_native_input_with_a_visible_label_and_a_mixed_state(half):
    if half == "label":
        for checked in (True, False):
            root = _render(_CHECKBOX, {"label": "Remember this device", "checked": checked})
            boxes = [el for el in root.iter("input") if el.get("type") == "checkbox"]
            assert len(boxes) == 1 and len(list(root.iter("input"))) == 1, (
                "one native checkbox, and no other input"
            )
            box = boxes[0]
            assert box.has("checked") is checked, f"checked={checked} is the input's own state"
            label = _associated_label(root, box)
            assert label is not None, "the checkbox has a label element that names it"
            assert "Remember this device" in label.visible_text(), (
                f"the label is visible text: {label.visible_text()!r}"
            )
        described = _render(_CHECKBOX, {
            "label": "Notify me", "description": "A notice when a long task ends",
        })
        box = _one(described, "input")
        label = _associated_label(described, box)
        assert label is not None and "Notify me" in label.visible_text(), (
            "a checkbox with a description keeps its visible label"
        )
        assert "A notice" not in label.text(), (
            f"the name is the label alone: the description stands outside it: {label.text()!r}"
        )
        pointed = box.get("aria-describedby")
        targets = [el for el in described.iter() if pointed and el.get("id") == pointed]
        assert len(targets) == 1 and " ".join(targets[0].text().split()) == "A notice when a long task ends", (
            f"the input's description is the description's text: {pointed!r}"
        )
        return

    source = read(_CHECKBOX)
    assert re.search(r"export\s+let\s+indeterminate\b", _script(_CHECKBOX)), (
        "the mixed state is a prop"
    )
    tags = re.findall(r"<input\b[^>]*>", _markup(_CHECKBOX), re.S)
    assert len(tags) == 1 and re.search(r"""type\s*=\s*["']checkbox["']""", tags[0]), tags
    assert re.search(
        r"(?:bind:indeterminate(?:\s*=\s*\{\s*indeterminate\s*\})?|(?<![\w:])indeterminate\s*=\s*\{\s*indeterminate\s*\})",
        tags[0],
    ), f"the prop sets the input's own indeterminate state: {tags[0]}"
    assert re.search(
        r"(?<![^\s])(?:bind:indeterminate(?![\w-])|indeterminate\s*=\s*\{\s*indeterminate\s*\})", tags[0],
    ), f"the input's own property, not an attribute whose name ends in it: {tags[0]}"
    assert _states_by_hand('<input type="checkbox" aria-checked="mixed" />'), (
        "the census reads a state said by hand"
    )
    assert not _states_by_hand(source), "a native checkbox says its state itself"
    mixed = _render(_CHECKBOX, {"label": "Select all", "indeterminate": True})
    assert len([el for el in mixed.iter("input") if el.get("type") == "checkbox"]) == 1


# ===========================================================================
# PR6 -- SidePanel: a region beside the page, resized by keys
# ===========================================================================
_SIZE_DRIVER = r"""
const size = await import(process.env.OO_SIZE);
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT);
const keyed = input.keys.map(([key, width, bounds]) => size.resizeKey(key, width, bounds));
const clamped = input.widths.map(([width, bounds]) => size.clampWidth(width === 'NaN' ? NaN : width, bounds));
const dragged = input.drags.map(([start, from, x, bounds]) => size.dragWidth(start, from, x, bounds));
const released = typeof size.releaseWidth !== 'function' ? 'releaseWidth is not exported' :
    input.releases.map(([start, from, x, bounds]) => size.releaseWidth(start, from, x, bounds));
console.log('OO_JSON ' + JSON.stringify({ keyed, clamped, dragged, released }));
console.log('PASS ' + clause);
"""

_BOUNDS = {"min": 280, "max": 640, "step": 16}
_LEFT = {**_BOUNDS, "side": "left"}
_KEY_CASES = (
    # A panel on the right grows toward the left.
    (("ArrowLeft", 400, _BOUNDS), 416),
    (("ArrowRight", 400, _BOUNDS), 384),
    (("ArrowLeft", 632, _BOUNDS), 640),
    (("ArrowRight", 290, _BOUNDS), 280),
    (("ArrowLeft", 640, _BOUNDS), 640),
    (("Home", 500, _BOUNDS), 280),
    (("End", 300, _BOUNDS), 640),
    (("Enter", 400, _BOUNDS), None),
    (("ArrowUp", 400, _BOUNDS), None),
    (("a", 400, _BOUNDS), None),
    # A panel on the left grows toward the right.
    (("ArrowRight", 400, _LEFT), 416),
    (("ArrowLeft", 400, _LEFT), 384),
)
_WIDTH_CASES = (
    (("NaN", _BOUNDS), 280),
    ((1000, _BOUNDS), 640),
    ((100, _BOUNDS), 280),
    ((455.6, _BOUNDS), 456),
    ((400, _BOUNDS), 400),
)
# (startWidth, startX, x, bounds) -> the width. A panel on the right grows as
# its edge moves left; a move of 3 px or less is not yet a drag.
_DRAG_CASES = (
    ((400, 500, 450, _BOUNDS), 450),
    ((400, 500, 560, _BOUNDS), 340),
    ((400, 500, 100, _BOUNDS), 640),
    ((400, 500, 900, _BOUNDS), 280),
    ((400, 500, 503, _BOUNDS), 400),
    ((400, 500, 497, _BOUNDS), 400),
    ((400, 500, 550, _LEFT), 450),
    ((400, 500, 450, _LEFT), 350),
)
# A release: a press that did not drag steps the width to the next of three
# stops (the narrowest, the middle, the widest: 280, 460, 640), past the
# widest back to the narrowest; a drag keeps where it went.
_RELEASE_CASES = (
    ((400, 500, 501, _BOUNDS), 460),
    ((460, 500, 500, _BOUNDS), 640),
    ((640, 500, 498, _BOUNDS), 280),
    ((280, 500, 500, _BOUNDS), 460),
    ((300, 500, 500, _LEFT), 460),
    ((400, 500, 450, _BOUNDS), 450),
    ((400, 500, 550, _LEFT), 450),
)
_DIALOG_PATTERNS = (
    r"""role\s*=\s*["']dialog["']""", r"aria-modal\s*=", r"\.showModal\s*\(",
    r"\.inert\b", r"""["']inert["']""", r"(?<=\s)inert(?=[\s=/>])", r"<dialog\b",
    r"""from\s*['"](?:\./|\$lib/ds/)Modal(?:\.svelte)?['"]""", r"<Modal\b",
)


def _dialogs(source):
    """Every way a source makes itself a dialog or hides the page."""
    return [found for pattern in _DIALOG_PATTERNS for found in re.findall(pattern, source)]


_APP_SHELL = "frontend/src/lib/components/layout/AppShell.svelte"
# What a hand-made resize handle writes: its own pointer or mouse tracking,
# its own cursor, its own separator.
_HAND_RESIZE = re.compile(
    r"\b(?:mousedown|mousemove|mouseup|pointerdown|pointermove|pointerup|setPointerCapture)\b"
    r"|col-resize|role\s*=\s*[\"']separator[\"']"
)


@pytest.mark.parametrize("half", ("landmark", "resize", "host"))
def test_pr6_a_side_panel_is_a_region_beside_the_page_resized_by_keys(half):
    if half == "host":
        sample = (
            "<script>\n\timport SidePanel from '$lib/ds/SidePanel.svelte';\n"
            "\twindow.addEventListener('mousemove', onMouseMove);\n</script>\n"
            '<div role="separator" class="cursor-col-resize" on:mousedown={start}></div>\n'
        )
        assert len(_HAND_RESIZE.findall(sample)) == 4, (
            f"the census reads a hand-made resize: {_HAND_RESIZE.findall(sample)}"
        )
        shell, script = read(_APP_SHELL), _script(_APP_SHELL)
        assert re.search(
            r"""import\s+SidePanel\s+from\s*['"]\$lib/ds/SidePanel(?:\.svelte)?['"]"""
            r"""|import\s*\{[^}]*\bSidePanel\b[^}]*\}\s*from\s*['"]\$lib/ds['"]""", script,
        ), "the app shell imports the side panel primitive"
        hosted = re.findall(r"<SidePanel\b", _markup(_APP_SHELL))
        assert len(hosted) == 1, f"the app shell hosts its right panel in the side panel: {len(hosted)}"
        tag = _markup(_APP_SHELL)
        start = tag.find("<SidePanel")
        opening = tag[start:_tag_end(tag, start)]
        assert re.search(r"\bon:resize\s*=", opening) and re.search(r"(?<![\w-])width\s*=\s*\{", opening), (
            f"the shell gives the panel its width and takes its resizes: {opening}"
        )
        competing = _HAND_RESIZE.findall(shell) + re.findall(r"<aside\b", _markup(_APP_SHELL))
        assert not competing, f"the shell holds no resize logic and no panel of its own: {competing}"
        return

    if half == "landmark":
        source, markup = read(_SIDE_PANEL), _markup(_SIDE_PANEL)
        asides = re.findall(r"<aside\b[^>]*>", markup, re.S)
        assert len(asides) == 1 and re.search(r"""role\s*=\s*["']complementary["']""", asides[0]) and (
            re.search(r"\baria-label(?:ledby)?\s*=", asides[0])
        ), f"the panel is one labelled complementary region: {asides}"
        sample = (
            '<aside role="dialog" aria-modal="true"></aside>\n<dialog></dialog>\n'
            "<script>import Modal from './Modal.svelte'; el.showModal(); main.inert = true;</script>\n"
        )
        assert len(_dialogs(sample)) == 6, f"the census reads each way to be a dialog: {_dialogs(sample)}"
        dialogs = _dialogs(source)
        assert not dialogs, f"the panel is never a dialog and never hides the page: {dialogs}"
        root = _render(_SIDE_PANEL, {"label": "Inspector", "width": 400})
        regions = [el for el in root.iter() if el.get("role") == "complementary"]
        assert len(regions) == 1 and regions[0].tag == "aside" and (
            regions[0].get("aria-label") == "Inspector"
        ), f"it renders as the labelled region: {[r.attrs for r in regions]}"
        assert not [el for el in root.iter() if el.get("role") == "dialog" or el.tag == "dialog"]
        return

    out = run_ts(
        {"OO_SIZE": _SIDE_PANEL_SIZE}, _SIZE_DRIVER, "size",
        env={"OO_INPUT": json.dumps({
            "keys": [list(case) for case, _ in _KEY_CASES],
            "widths": [list(case) for case, _ in _WIDTH_CASES],
            "drags": [list(case) for case, _ in _DRAG_CASES],
            "releases": [list(case) for case, _ in _RELEASE_CASES],
        })},
    )
    got = _json_line(out)
    wrong = [
        f"resizeKey{case}: {result} (expected {expected})"
        for (case, expected), result in zip(_KEY_CASES, got["keyed"]) if result != expected
    ] + [
        f"clampWidth{case}: {result} (expected {expected})"
        for (case, expected), result in zip(_WIDTH_CASES, got["clamped"]) if result != expected
    ]
    assert len(got["keyed"]) == len(_KEY_CASES) and not wrong, (
        "the panel grows, shrinks and clamps as its handle is moved:\n  " + "\n  ".join(wrong)
    )
    wrong = [
        f"dragWidth{case}: {result} (expected {expected})"
        for (case, expected), result in zip(_DRAG_CASES, got["dragged"]) if result != expected
    ]
    assert len(got["dragged"]) == len(_DRAG_CASES) and not wrong, (
        "a drag of the edge widens toward the page and clamps, and a move of 3 px is not "
        "yet a drag:\n  " + "\n  ".join(wrong)
    )
    released = got["released"]
    assert isinstance(released, list) and len(released) == len(_RELEASE_CASES), (
        f"releaseWidth answers every release: {released}"
    )
    wrong = [
        f"releaseWidth{case}: {result} (expected {expected})"
        for (case, expected), result in zip(_RELEASE_CASES, released) if result != expected
    ]
    assert not wrong, (
        "a press with no drag steps the width through its three stops, and a drag keeps "
        "where it went:\n  " + "\n  ".join(wrong)
    )

    script, source = _script(_SIDE_PANEL), read(_SIDE_PANEL)
    assert re.search(
        r"import\s*\{[^}]*\bresizeKey\b[^}]*\}\s*from\s*['\"]\./sidePanelSize(?:\.ts|\.js)?['\"]", script,
    ) and re.search(r"\bresizeKey\s*\(", script), "the panel's handle resizes through resizeKey"
    for name in ("dragWidth", "releaseWidth"):
        assert re.search(
            r"import\s*\{[^}]*\b" + name + r"\b[^}]*\}\s*from\s*['\"]\./sidePanelSize(?:\.ts|\.js)?['\"]", script,
        ) and re.search(r"\b" + name + r"\s*\(", script), f"the panel's pointer goes through {name}"
    competing = _KEY_NAMES.findall(source)
    assert not competing, f"the panel holds no key logic of its own: {competing}"
    root = _render(_SIDE_PANEL, {"label": "Inspector", "width": 400})
    handles = [el for el in root.iter() if el.get("role") == "separator"]
    assert len(handles) == 1, f"one resize handle: {len(handles)}"
    handle = handles[0]
    assert {
        name: handle.get(name) for name in (
            "aria-orientation", "tabindex", "aria-valuenow", "aria-valuemin", "aria-valuemax",
        )
    } == {
        "aria-orientation": "vertical", "tabindex": "0", "aria-valuenow": "400",
        "aria-valuemin": "280", "aria-valuemax": "640",
    }, f"the handle is a focusable vertical separator that says the width: {handle.attrs}"
    assert handle.get("aria-label") or handle.get("aria-labelledby"), "the handle is named"
    assert handle.get("aria-valuetext") == "400 pixels", (
        f"the handle says its width with its unit: {handle.get('aria-valuetext')!r}"
    )
    region = [el for el in root.iter() if el.get("role") == "complementary"]
    controlled = handle.get("aria-controls")
    assert controlled and len(region) == 1 and region[0].get("id") == controlled, (
        f"the handle names the region it sizes: {controlled!r}, {[r.get('id') for r in region]}"
    )


# ===========================================================================
# PR7 -- icons: the inline set, drawn first, at a stroke of 1.5
# ===========================================================================
_ICON_TAG = re.compile(r"<Icon\b")
_ICON_PROPS = ("icon", "iconLeft", "iconRight", "iconOnly")
_STRING = re.compile(r"""'([^'\\\n]*)'|"([^"\\\n]*)"|`([^`\\$\n]*)`""")
_OPERAND = r"""(?:'[^'\n]*'|"[^"\n]*"|[\w$.]+)"""
_COMPARISON = re.compile(_OPERAND + r"\s*(?:===|!==|==|!=)\s*" + _OPERAND)


def _tag_end(text, start):
    """The index just past the tag opening at ``start``: quotes and braces
    are skipped, so a ``>`` inside an expression does not end it."""
    depth, index = 0, start
    while index < len(text):
        char = text[index]
        if char in "\"'`" and depth:
            quote, index = char, index + 1
            while index < len(text) and text[index] != quote:
                index += 2 if text[index] == "\\" else 1
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == ">" and depth == 0:
            return index + 1
        index += 1
    return index


def _attribute_values(tag, names):
    """The raw values of the attributes ``names`` in one tag: a quoted value,
    or the text of a braced expression."""
    values = []
    pattern = re.compile(r"(?<![\w:-])(" + "|".join(map(re.escape, names)) + r")\s*=\s*")
    for match in pattern.finditer(tag):
        index = match.end()
        if index >= len(tag):
            continue
        if tag[index] in "\"'":
            end = tag.find(tag[index], index + 1)
            values.append(tag[index:end + 1])
        elif tag[index] == "{":
            depth, cursor = 0, index
            while cursor < len(tag):
                if tag[cursor] == "{":
                    depth += 1
                elif tag[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                cursor += 1
            values.append(tag[index + 1:cursor])
    return values


def _declared(script, name):
    """The initializer of ``const|let <name> = ...`` in a script, up to its
    statement's end at depth 0, or None."""
    match = re.search(r"(?:const|let|var)\s+" + re.escape(name) + r"\b[^=;]*=", script)
    if not match:
        return None
    depth, index = 0, match.end()
    while index < len(script):
        char = script[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ";" and depth <= 0:
            break
        index += 1
    return script[match.end():index]


def _named_icons(path, text):
    """Every icon name a file writes as a literal: an ``<Icon name>``, an
    icon prop on any tag (``icon``, ``iconLeft``, ``iconRight``,
    ``iconOnly``), an ``icon:`` property of an object, and the literals of
    a script value a name expression reads (a map of names, say). A name
    passed through a prop without a literal of its own is its caller's."""
    names = []
    scripts = "\n".join(_SCRIPT_BLOCK.findall(text)) if path.endswith(".svelte") else text
    markup = _markup_of(path, text)

    def literals(expression):
        # A comparison's operands and an index are not names: in
        # ``kind === 'ok' ? 'a' : 'b'`` the names are the branches, and in
        # ``ICON[kind]`` the map is read, not the key.
        expression = _COMPARISON.sub(" ", expression)
        expression = re.sub(r"\[[^\[\]]*\]", " ", expression)
        found = [a or b or c for a, b, c in _STRING.findall(expression)]
        for ident in set(re.findall(r"(?<![\w$.])([A-Za-z_$][\w$]*)", _STRING.sub(" ", expression))):
            value = _declared(scripts, ident)
            if value is not None:
                found += [a or b or c for a, b, c in _STRING.findall(value)]
        return found

    for match in re.finditer(r"<([A-Za-z][\w.:-]*)\b", markup):
        tag = markup[match.start():_tag_end(markup, match.start())]
        wanted = ("name",) + _ICON_PROPS if match.group(1) == "Icon" else _ICON_PROPS
        for value in _attribute_values(tag, wanted):
            names += literals(value)
    for match in re.finditer(r"(?<![\w$])(?:icon|iconLeft|iconRight|iconOnly)\s*:\s*", scripts + "\n" + markup):
        rest = (scripts + "\n" + markup)[match.end():]
        literal = _STRING.match(rest)
        if literal:
            names.append(next(part for part in literal.groups() if part is not None))
    for match in re.finditer(r"export\s+let\s+(?:icon|iconLeft|iconRight|iconOnly)\s*(?::[^=;]*)?=\s*", scripts):
        literal = _STRING.match(scripts[match.end():])
        if literal:
            names.append(next(part for part in literal.groups() if part is not None))
    return names


def _markup_of(path, text):
    if not path.endswith(".svelte"):
        return ""
    return _STYLE_BLOCK.sub(" ", _SCRIPT_BLOCK.sub(" ", text))


_ICON_SAMPLE = (
    "frontend/src/lib/sample/Named.svelte",
    "<script lang=\"ts\">\n"
    "\timport Icon from '$lib/ds/Icon.svelte';\n"
    "\tconst ICON: Record<string, string> = { ok: 'first-name', no: \"second-name\" };\n"
    "\tconst tabs = [{ id: 'a', label: 'A', icon: 'third-name' }];\n"
    "\texport let kind = 'ok';\n"
    "\texport let icon = 'fourth-name';\n"
    "</script>\n"
    "<Icon name=\"fifth-name\" size=\"sm\" />\n"
    "<Icon name={kind === 'ok' ? 'sixth-name' : 'seventh-name'} />\n"
    "<Icon name={ICON[kind]} />\n"
    "<Button iconLeft=\"eighth-name\" on:click={() => (kind = 'no')}>Go</Button>\n"
    "<Icon name={icon} />\n",
)
_PATH_TOKEN = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?(?:\d+\.?\d*|\.\d+))")
_ARGS = {"m": 2, "l": 2, "h": 1, "v": 1, "c": 6, "s": 4, "q": 4, "t": 2, "a": 7, "z": 0}


def _walk(d):
    """The absolute end points of a path's segments; raises on a malformed
    path (an unknown character, a command short of arguments, an arc flag
    that is not 0 or 1)."""
    rest = _PATH_TOKEN.sub("", d).replace(" ", "")
    if rest:
        raise ValueError(f"characters outside path data: {rest!r}")
    tokens = [(cmd, num) for cmd, num in _PATH_TOKEN.findall(d)]
    points, index, x, y, start = [], 0, 0.0, 0.0, (0.0, 0.0)
    if not tokens or tokens[0][0] not in "Mm":
        raise ValueError("a path starts with a move")
    command = None
    while index < len(tokens):
        if tokens[index][0]:
            command = tokens[index][0]
            index += 1
        elif command is None:
            raise ValueError("a number before any command")
        lower = command.lower()
        count = _ARGS[lower]
        if lower == "z":
            x, y = start
            points.append((x, y))
            command = None
            continue
        args = [tokens[index + k][1] for k in range(count) if index + k < len(tokens)]
        if len(args) < count or any(arg == "" for arg in args):
            raise ValueError(f"{command} is short of arguments")
        index += count
        values = [float(arg) for arg in args]
        relative = command.islower()
        if lower == "a":
            if args[3] not in ("0", "1") or args[4] not in ("0", "1"):
                raise ValueError(f"an arc flag is not 0 or 1: {args}")
            end = values[5:7]
        elif lower == "h":
            end = [values[0], 0.0 if relative else y]
        elif lower == "v":
            end = [0.0 if relative else x, values[0]]
        else:
            end = values[-2:]
        if relative:
            x, y = x + end[0], y + end[1]
        else:
            x, y = end
        points.append((x, y))
        if lower == "m":
            start = (x, y)
            command = "l" if command == "m" else "L"
    return points


_CLEAN_GUARD = "_public_clean_guard_under_icons"
_GUARDS = Path(__file__).resolve().parent.parent / ".github" / "scripts"
_KEBAB = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


def _set_findings(icons, charged):
    """Every way an icon set is malformed: a name not in kebab case, a shape
    that is neither a path nor a filled path, path data that is not compact
    or ends a number in a point (after which a command letter reads as a
    word), a malformed path, a point outside the 24 unit box, and a path
    ``charged`` (the public clean guard's detector) charges."""
    malformed, paths = [], []
    for name, shapes in icons.items():
        if not _KEBAB.match(name):
            malformed.append(f"{name}: not a kebab-case name")
        if not isinstance(shapes, list) or not shapes:
            malformed.append(f"{name}: no shape")
            continue
        for shape in shapes:
            if not isinstance(shape, str) and set(shape) != {"d", "filled"}:
                malformed.append(f"{name}: a shape is a path or a filled path: {shape}")
                continue
            if not isinstance(shape, str) and shape["filled"] is not True:
                malformed.append(f"{name}: a filled shape says so: {shape}")
            d = _shape_d(shape)
            paths.append((name, d))
            if re.search(r"\s[A-Za-z]|[A-Za-z]\s|\s\s|^\s|\s$|,", d):
                malformed.append(f"{name}: path data not compact: {d!r}")
            if re.search(r"\d\.(?!\d)", d):
                malformed.append(f"{name}: a number ends in a point: {d!r}")
            try:
                points = _walk(d)
            except ValueError as exc:
                malformed.append(f"{name}: {exc}: {d!r}")
                continue
            outside = [p for p in points if not (1 <= p[0] <= 23 and 1 <= p[1] <= 23)]
            if outside:
                malformed.append(f"{name}: points outside the 24 unit box: {outside[:3]}")
    for index, kind, _ in charged([d for _, d in paths]):
        name, d = paths[index]
        malformed.append(f"{name}: the clean guard charges it as {kind}: {d!r}")
    return malformed


@pytest.mark.parametrize("half", ("stroke", "held", "inline", "set"))
def test_pr7_icons_are_drawn_from_the_inline_set_at_a_stroke_of_one_and_a_half(half):
    if half == "stroke":
        script = _script(_ICON)
        default = re.search(r"export\s+let\s+strokeWidth\s*(?::\s*number\s*)?=\s*([\d.]+)\s*;", script)
        assert default and float(default.group(1)) == 1.5, (
            f"the icon's stroke is 1.5 unless a call asks otherwise: {default and default.group(0)}"
        )
        markup = _markup(_ICON)
        assert re.search(r"stroke-width\s*=\s*\{\s*strokeWidth\s*\}", markup), (
            "the inline drawing draws at that stroke"
        )
        start = markup.find("<svelte:component")
        fallback = markup[start:_tag_end(markup, start)] if start >= 0 else ""
        assert re.search(
            r"(?<![\w-])\{\s*strokeWidth\s*\}|(?<![\w-])strokeWidth\s*=\s*\{\s*strokeWidth\s*\}", fallback,
        ), f"the fallback draws at that stroke, read in its own tag: {fallback!r}"
        return

    if half == "held":
        path, text = _ICON_SAMPLE
        sample = _named_icons(path, text)
        assert set(sample) == {
            "first-name", "second-name", "third-name", "fourth-name", "fifth-name",
            "sixth-name", "seventh-name", "eighth-name",
        }, f"the census reads each way a file names an icon, and nothing else: {sorted(sample)}"
        written = {}
        for held in held_files():
            for name in _named_icons(held, read(held)):
                written.setdefault(name, set()).add(held)
        assert written, "the held files name icons, and the census reads them"
        names = tuple(sorted(written))
        known = dict(zip(names, _inline(names)["found"]))
        drawn = set(_inline()["icons"])
        missing = {name: sorted(paths) for name, paths in written.items() if name not in drawn or not known[name]}
        assert not missing, (
            f"icon names held files write that the inline set does not draw (the fallback is "
            f"for files not yet held): {missing}"
        )
        return

    icons = _inline()["icons"]
    if half == "inline":
        for name, props in (("check", {"name": "check"}), ("chevron-down", {"name": "ChevronDown"})):
            drawn = _one(_render(_ICON, props), "svg")
            expected = [_shape_d(shape) for shape in icons[name]]
            assert _paths(drawn) == expected, (
                f"{props['name']} is drawn from the inline set's {name}: {_paths(drawn)}"
            )
            # The parser lowercases attribute names: viewBox reads as viewbox.
            attributes = {key: drawn.get(key) for key in (
                "viewbox", "fill", "stroke", "stroke-width", "stroke-linecap",
                "stroke-linejoin", "aria-hidden",
            )}
            assert attributes == {
                "viewbox": "0 0 24 24", "fill": "none", "stroke": "currentColor",
                "stroke-width": "1.5", "stroke-linecap": "round", "stroke-linejoin": "round",
                "aria-hidden": "true",
            }, f"the inline drawing's frame: {attributes}"
            assert not any("lucide" in cls for cls in drawn.classes()), drawn.classes()
        filled = _one(_render(_ICON, {"name": "stop-fill"}), "svg")
        fills = [(p.get("fill"), p.get("stroke")) for p in filled.iter("path")]
        assert ("currentColor", "none") in fills, f"a filled shape is filled with the text colour: {fills}"
        fallback = _one(_render(_ICON, {"name": "fingerprint"}), "svg")
        assert any("lucide" in cls for cls in fallback.classes()) and (
            fallback.get("stroke-width") == "1.5"
        ), f"a name the set lacks falls back, at the same stroke: {fallback.attrs}"
        assert not list(_render(_ICON, {"name": "not-an-icon-anywhere"}).iter("svg")), (
            "an unknown name draws nothing"
        )
        return

    assert set(_DESIGN) <= set(icons), (
        f"the design's thirty drawings are in the set: {sorted(set(_DESIGN) - set(icons))}"
    )
    assert len(_DESIGN) == 30
    loaded, restore = isolate(targets={_CLEAN_GUARD: _GUARDS / "public_clean_guard.py"})
    try:
        charged = loaded[_CLEAN_GUARD].find_violations
        sample = _set_findings({
            "fine": ["m5 12.5 4.5 4.5L19 7.5"],
            "Spaced_Name": ["M5 12 L19 7"],
            # A command letter after a number ending in a point, assembled
            # so this line does not spell what the guard charges.
            "point": ["M5 12." + "s" + "10 3 12 6"],
            "far": ["M0 0h24"],
            "unsaid": [{"d": "M5 5h14v14H5Z", "filled": False}],
        }, charged)
        malformed = _set_findings(icons, charged)
    finally:
        restore()
    said = sorted(finding.split(":")[0] for finding in sample)
    assert said == ["Spaced_Name", "Spaced_Name", "far", "point", "point", "unsaid"] and any(
        "clean guard" in finding for finding in sample
    ), f"the census reads a bad name, spaced data, a number ending in a point, a point the guard charges, a point outside the box and an unsaid fill: {sample}"
    assert not malformed, "the inline set is well formed:\n  " + "\n  ".join(malformed)
    for bad in ("M1 1x2", "M1 1a1 1 0 2 0 2 2", "M1 1c1 1", "L1 1"):
        with pytest.raises(ValueError):
            _walk(bad)
    probe = ("ChevronDown", "chevron_down", "x", "constructor", "toString", "__proto__")
    asked = _inline(probe)
    assert asked["keys"][:3] == ["chevron-down", "chevron-down", "x"], (
        f"a name is looked up in kebab case: {asked['keys']}"
    )
    assert asked["found"] == [True, True, "x" in icons, False, False, False], (
        f"only the set's own names resolve, never an inherited property: {asked['found']}"
    )



# ===========================================================================
# PR8 -- a pressed check is seen on its ground; the pressed tint holds
# ===========================================================================
# The cascade is read from what the server compiled (Svelte's scoping
# classes included, so a rule's weight is the one a browser gives it) over
# the elements it rendered. A rule inside an at-rule is not the resting
# cascade. A pseudo-class the resolver does not know refuses, so a new rule
# never slips past it unread.
_CSS_COMMENT_ANY = re.compile(r"/\*.*?\*/", re.S)
_SIMPLE_SELECTOR = re.compile(
    r"\.(?P<cls>[\w-]+)"
    r"|#(?P<id>[\w-]+)"
    r"|\[\s*(?P<attr>[\w:-]+)\s*(?:(?P<op>[~|^$*]?=)\s*"
    r"(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)'|(?P<bare>[^\]\s]+))\s*)?\]"
    r"|::(?P<pe>[\w-]+)(?:\([^()]*\))?"
    r"|:(?P<pc>[\w-]+)(?P<args>\((?:[^()]|\([^()]*\))*\))?"
    r"|(?P<star>\*)"
    r"|(?P<tag>[A-Za-z][\w-]*)"
)
_AT_REST = frozenset({
    "focus", "focus-visible", "focus-within", "active", "checked", "visited",
    "target", "invalid", "placeholder-shown", "indeterminate",
})


def _top_split(text, separator):
    """``text`` split at ``separator`` outside brackets, parentheses and quotes."""
    parts, depth, start, index = [], 0, 0, 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            end = text.find(char, index + 1)
            index = end + 1 if end > 0 else len(text)
            continue
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _compiled_rules(css):
    """``[(selector list, [(property, value, important)])]`` of the top-level
    rules of compiled CSS, in their order."""
    css = _CSS_COMMENT_ANY.sub(" ", css)
    rules, index = [], 0
    while index < len(css):
        brace = css.find("{", index)
        if brace < 0:
            break
        prelude = css[index:brace].strip().lstrip(";").strip()
        depth, cursor = 1, brace + 1
        while cursor < len(css) and depth:
            char = css[cursor]
            if char in "\"'":
                end = css.find(char, cursor + 1)
                cursor = end + 1 if end > 0 else len(css)
                continue
            depth += char == "{"
            depth -= char == "}"
            cursor += 1
        if not prelude.startswith("@"):
            declared = []
            for part in _top_split(css[brace + 1:cursor - 1], ";"):
                if ":" not in part:
                    continue
                name, value = part.split(":", 1)
                value = " ".join(value.split())
                important = bool(re.search(r"!\s*important$", value))
                declared.append((name.strip().lower(), re.sub(r"\s*!\s*important$", "", value), important))
            rules.append((prelude, declared))
        index = cursor
    return rules


def _compounds(selector):
    """``[(combinator, compound)]`` of one complex selector, left to right;
    the first combinator is None."""
    parts, current, pending, depth = [], "", None, 0
    for char in selector:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if depth == 0 and (char.isspace() or char in ">+~"):
            if current:
                parts.append((pending, current))
                current, pending = "", " "
            if char in ">+~":
                pending = char
            continue
        current += char
    if current:
        parts.append((pending, current))
    return parts


def _simples(compound):
    found = list(_SIMPLE_SELECTOR.finditer(compound))
    if "".join(m.group(0) for m in found) != compound:
        raise AssertionError(f"a selector the resolver does not read: {compound!r}")
    return found


def _compound_matches(element, compound, hovered):
    for simple in _simples(compound):
        kind = simple.lastgroup
        if simple.group("cls"):
            if simple.group("cls") not in element.classes():
                return False
        elif simple.group("id"):
            if element.get("id") != simple.group("id"):
                return False
        elif simple.group("attr"):
            name, op = simple.group("attr").lower(), simple.group("op")
            if not element.has(name):
                return False
            have = element.get(name) or ""
            want = next((v for v in (simple.group("dq"), simple.group("sq"), simple.group("bare")) if v is not None), "")
            if op == "=" and have != want:
                return False
            if op == "~=" and want not in have.split():
                return False
            if op == "^=" and not have.startswith(want):
                return False
            if op == "$=" and not have.endswith(want):
                return False
            if op == "*=" and want not in have:
                return False
            if op == "|=" and not (have == want or have.startswith(want + "-")):
                return False
        elif simple.group("pe"):
            return False
        elif simple.group("pc"):
            name = simple.group("pc")
            inner = _top_split(simple.group("args")[1:-1], ",") if simple.group("args") else []
            if name == "hover":
                if element not in hovered:
                    return False
            elif name == "not":
                if any(_matches(element, part, hovered) for part in inner):
                    return False
            elif name in ("is", "where"):
                if not any(_matches(element, part, hovered) for part in inner):
                    return False
            elif name == "disabled":
                if not element.has("disabled"):
                    return False
            elif name == "enabled":
                if element.has("disabled"):
                    return False
            elif name in _AT_REST:
                return False
            else:
                raise AssertionError(f"a pseudo-class the resolver does not read: :{name}")
        elif simple.group("tag"):
            if element.tag != simple.group("tag").lower():
                return False
        elif kind != "star":
            return False
    return True


def _element_parent(element):
    parent = element.parent
    return None if parent is None or parent.tag == "#root" else parent


def _previous_siblings(element):
    parent = element.parent
    if parent is None:
        return []
    siblings = [child for child in parent.children if isinstance(child, _Element)]
    return list(reversed(siblings[:siblings.index(element)]))


def _matches(element, selector, hovered):
    parts = _compounds(selector)
    if not parts:
        return False

    def matches_at(index, node):
        combinator, compound = parts[index]
        if not _compound_matches(node, compound, hovered):
            return False
        if index == 0:
            return True
        combinator = parts[index][0]
        if combinator == " ":
            ancestor = _element_parent(node)
            while ancestor is not None:
                if matches_at(index - 1, ancestor):
                    return True
                ancestor = _element_parent(ancestor)
            return False
        if combinator == ">":
            parent = _element_parent(node)
            return parent is not None and matches_at(index - 1, parent)
        before = _previous_siblings(node)
        if combinator == "+":
            return bool(before) and matches_at(index - 1, before[0])
        return any(matches_at(index - 1, sibling) for sibling in before)

    return matches_at(len(parts) - 1, element)


def _specificity(selector):
    ids = classes = types = 0
    for _, compound in _compounds(selector):
        for simple in _simples(compound):
            if simple.group("id"):
                ids += 1
            elif simple.group("cls") or simple.group("attr"):
                classes += 1
            elif simple.group("pc"):
                name = simple.group("pc")
                if name in ("not", "is"):
                    inner = _top_split(simple.group("args")[1:-1], ",")
                    best = max((_specificity(part) for part in inner), default=(0, 0, 0))
                    ids, classes, types = ids + best[0], classes + best[1], types + best[2]
                elif name != "where":
                    classes += 1
            elif simple.group("pe") or simple.group("tag"):
                types += 1
    return ids, classes, types


def _winning(element, names, rules, hovered):
    """The value the cascade gives ``element`` for the properties ``names``
    (a longhand and its shorthand compete as one), or None."""
    best = None
    for order, (selectors, declared) in enumerate(rules):
        wanted = [(i, value, important) for i, (name, value, important) in enumerate(declared) if name in names]
        if not wanted:
            continue
        weights = [_specificity(part) for part in _top_split(selectors, ",") if _matches(element, part, hovered)]
        if not weights:
            continue
        for i, value, important in wanted:
            key = (important, max(weights), order, i)
            if best is None or key > best[0]:
                best = (key, value)
    return None if best is None else best[1]


def _ink_of(element, rules, hovered):
    """The colour ``element`` draws its text and its strokes in."""
    node = element
    while node is not None:
        value = _winning(node, ("color",), rules, hovered)
        if value is not None and value.lower() not in ("inherit", "currentcolor", "unset"):
            return value
        node = _element_parent(node)
    return "var(--oo-fg-primary)"


def _background_of(element, rules, hovered):
    value = _winning(element, ("background-color", "background"), rules, hovered)
    if value is None or value in ("transparent", "none", "inherit", "initial", "unset"):
        return None
    if len(_top_split(value, " ")) != 1 and not re.fullmatch(r"(?:var|color-mix)\(.*\)", value):
        raise AssertionError(f"a background the resolver does not read: {value!r}")
    return value


def _layers_under(element, rules, hovered):
    """The grounds drawn under ``element``, innermost first."""
    layers, node = [], element
    while node is not None:
        value = _background_of(node, rules, hovered)
        if value is not None:
            layers.append(value)
        node = _element_parent(node)
    return layers


def _hovering(element):
    """``element`` under the pointer: it and every ancestor match :hover."""
    hovered, node = set(), element
    while node is not None:
        hovered.add(node)
        node = _element_parent(node)
    return hovered


def _rendered(path, props):
    """The tree a component renders, and the rules the server compiled for it."""
    out = ssr().render(path, props)
    return _dom(out.html), _compiled_rules(out.css)


@functools.cache
def _palette_resolvers():
    """``{palette: Resolver}`` over the derivation layer and each palette's roles."""
    from _colour import Resolver

    tokens = _ds._derivation()
    return {
        pid: Resolver(_ds._with_palette(tokens, pid), roles, scheme)
        for pid, (roles, scheme) in _ds._palettes().items()
    }


def _least_ratio(ink, layers):
    """The lowest contrast of ``ink`` on ``layers`` (innermost first) over
    every palette and every page ground the control may stand on."""
    from _colour import contrast, over

    least = None
    for pid, resolver in _palette_resolvers().items():
        for page in _ds._GROUNDS_4:
            ground = resolver.colour(page)
            for layer in reversed(layers):
                ground = over(resolver.colour(layer), ground)
            ratio = contrast(over(resolver.colour(ink), ground), ground)
            if least is None or ratio < least[0]:
                least = (ratio, pid, page)
    return least


_CASCADE_SAMPLE = (
    '<div class="a"><p class="b c" data-x="1"><span class="d">t</span></p></div>',
    ".b{color:blue}.a .b{color:red}.c:not([data-x]){color:green}"
    ".b[data-x='1']:hover{color:black}@media (min-width:1px){.b{color:pink}}"
    ".a{background-color:var(--oo-bg-surface)}.b{background:var(--oo-bg-tint-1)}"
    ".d{color:inherit}.a>.d{color:orange}",
)


def _checks(root):
    return [svg for svg in root.iter("svg") if svg.get("data-oo-icon") == "check"]


def _drawn_px(svg, rules):
    width = _winning(svg, ("width",), rules, set())
    size = float(width[:-2]) if width and width.endswith("px") else float(svg.get("width"))
    return float(svg.get("stroke-width")) * size / 24


@pytest.mark.parametrize("half", ("check", "hover"))
def test_pr8_a_pressed_check_is_seen_on_its_ground_and_the_pressed_tint_holds_under_the_pointer(half):
    sample_root = _dom(_CASCADE_SAMPLE[0])
    sample_rules = _compiled_rules(_CASCADE_SAMPLE[1])
    span = next(sample_root.iter("span"))
    paragraph = span.parent
    assert (
        _ink_of(span, sample_rules, set()), _ink_of(span, sample_rules, _hovering(span)),
        _layers_under(span, sample_rules, set()),
    ) == ("red", "black", ["var(--oo-bg-tint-1)", "var(--oo-bg-surface)"]) and paragraph.tag == "p", (
        "the resolver weighs selectors, skips at-rules, reads :not, :hover and inheritance, and stacks grounds"
    )

    if half == "check":
        low, thin, seen = [], [], 0
        for variant in ("primary", "secondary", "ghost", "danger", "link"):
            for form, extra in (("beside its label", {}), ("at the corner", {"iconOnly": "pin", "ariaLabel": "Pin"})):
                root, rules = _rendered(_BUTTON, {"variant": variant, "pressed": True, **extra})
                checks = _checks(root)
                assert len(checks) == 1, f"a pressed {variant} button {form} draws one check: {len(checks)}"
                seen += 1
                check = checks[0]
                ratio, pid, page = _least_ratio(_ink_of(check, rules, set()), _layers_under(check, rules, set()))
                if ratio < 3.0:
                    low.append(f"{variant} {form}: {ratio:.2f} in {pid} over {page}")
                if _drawn_px(check, rules) < 1.0:
                    thin.append(f"{variant} {form}: {_drawn_px(check, rules):.3f} px")
        root, rules = _rendered(_TOGGLE_CHIP, {"label": "Web search", "pressed": True})
        (check,) = _checks(root)
        ratio, pid, page = _least_ratio(_ink_of(check, rules, set()), _layers_under(check, rules, set()))
        if ratio < 3.0:
            low.append(f"the chip: {ratio:.2f} in {pid} over {page}")
        if _drawn_px(check, rules) < 1.0:
            thin.append(f"the chip: {_drawn_px(check, rules):.3f} px")
        assert seen == 10, seen
        assert not low, "a pressed check under 3:1 on the ground it is drawn on:\n  " + "\n  ".join(low)
        assert not thin, "a pressed check drawn thinner than a pixel:\n  " + "\n  ".join(thin)
        return

    root, rules = _rendered(_BUTTON, {"variant": "ghost"})
    button = _one(root, "button")
    assert (
        _background_of(button, rules, set()), _background_of(button, rules, _hovering(button)),
    ) == (None, "var(--oo-bg-hover)"), "the resolver reads the quiet button's hover wash"
    lost = []
    for path, props, tag in (
        (_BUTTON, {"variant": "ghost", "pressed": True}, "button"),
        (_BUTTON, {"variant": "secondary", "pressed": True}, "button"),
        (_TOGGLE_CHIP, {"label": "Web search", "pressed": True}, "button"),
    ):
        root, rules = _rendered(path, props)
        control = _one(root, tag)
        rest = _background_of(control, rules, set())
        under = _background_of(control, rules, _hovering(control))
        if not (rest and "--oo-bg-tint-1" in rest and under and "--oo-bg-tint-1" in under and under != rest):
            lost.append(f"{path.rsplit('/', 1)[-1]} {props}: at rest {rest}, under the pointer {under}")
    assert not lost, (
        "a pressed control keeps its tint under the pointer, with the hover wash over it:\n  " + "\n  ".join(lost)
    )


# ===========================================================================
# PR9 -- no icon-only control renders without a name
# ===========================================================================
def test_pr9_no_icon_only_control_renders_without_a_name():
    for props in ({"iconOnly": "x"}, {"iconOnly": "x", "ariaLabel": ""}, {"iconOnly": "x", "ariaLabel": "   "}):
        with pytest.raises(AssertionError, match="accessible name"):
            ssr().render(_BUTTON, props)
    named = _one(_render(_BUTTON, {"iconOnly": "x", "ariaLabel": "Close"}), "button")
    assert named.get("aria-label") == "Close", f"a named icon button says its name: {named.attrs}"
    for props in (
        {"label": "", "items": []}, {"label": "   ", "items": []}, {"label": "", "icon": "more", "items": []},
    ):
        with pytest.raises(AssertionError, match="accessible name"):
            ssr().render(_MENU, props)
    trigger = _one(_render(_MENU, {"label": "Actions", "items": []}), "button")
    assert trigger.get("aria-haspopup") == "menu" and "Actions" in trigger.visible_text(), (
        f"a named menu's trigger says its name and what it opens: {trigger.attrs}"
    )


# ---------------------------------------------------------------------------
# PR10 -- the chat frame hosts its side panels in SidePanel
# ---------------------------------------------------------------------------
_CHAT_FRAME = "frontend/src/routes/(app)/(use)/chat/+layout.svelte"


def test_pr10_the_chat_frame_hosts_its_side_panels_in_the_primitive():
    sample = (
        "<script>\n\timport SidePanel from '$lib/ds/SidePanel.svelte';\n"
        "\twindow.addEventListener('mousemove', onMouseMove);\n</script>\n"
        '<div role="separator" class="cursor-col-resize" on:mousedown={start}></div>\n'
    )
    assert len(_HAND_RESIZE.findall(sample)) == 4, (
        f"the census reads a hand-made resize: {_HAND_RESIZE.findall(sample)}"
    )
    frame, script, markup = read(_CHAT_FRAME), _script(_CHAT_FRAME), _markup(_CHAT_FRAME)
    assert re.search(
        r"""import\s+SidePanel\s+from\s*['"]\$lib/ds/SidePanel(?:\.svelte)?['"]"""
        r"""|import\s*\{[^}]*\bSidePanel\b[^}]*\}\s*from\s*['"]\$lib/ds['"]""", script,
    ), "the chat frame imports the side panel primitive"
    hosted = re.findall(r"<SidePanel\b", markup)
    assert len(hosted) == 1, f"the chat frame hosts its side panels in the side panel: {len(hosted)}"
    start = markup.find("<SidePanel")
    opening = markup[start:_tag_end(markup, start)]
    assert re.search(r"\bon:resize\s*=", opening) and re.search(r"(?<![\w-])width\s*=\s*\{", opening), (
        f"the frame gives the panel its width and takes its resizes: {opening}"
    )
    competing = _HAND_RESIZE.findall(frame) + re.findall(r"<aside\b", markup)
    assert not competing, f"the frame holds no resize logic and no panel of its own: {competing}"
    shell = read(_APP_SHELL)
    in_shell = (
        _HAND_RESIZE.findall(shell) + re.findall(r"<aside\b|<SidePanel\b", _markup(_APP_SHELL))
        + re.findall(r"\bSidePanel\b", _script(_APP_SHELL))
    )
    assert not in_shell, f"the shell of every page hosts no side panel and no resize: {in_shell}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
