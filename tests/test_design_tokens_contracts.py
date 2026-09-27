#!/usr/bin/env python3
"""Contracts for the design tokens: three palettes of roles, one derivation layer, one theme path.

Every colour of the interface comes from one of three palette files, day,
night and high contrast, each declaring the same named roles (a ground, a
surface, the text, an accent ink, a rule, ...) and its colour scheme, under
``[data-oo-theme="<id>"]``. One derivation layer, ``styles/theme.css``,
maps every app token (the ``--oo-*`` names components read) to a role, a mix
of roles, or a value that carries no colour; its selector is
``[data-oo-theme]`` because a custom property whose value is ``var(--a)``
resolves where it is declared, so a subtree that names another palette must
derive again. One module, ``lib/theme/apply.ts``, resolves what the browser
stored and what the system asks for into the state the root element carries
(palette, dark class, density, root font size, motion classes, componion
switch), and the inline pre-render in ``app.html`` must reach the same state
before the first paint.

The static contracts read sources as text. The Node contracts run
``apply.ts`` and the pre-render under Node with a fake ``document``,
``localStorage`` and ``matchMedia`` (``tests/_frontend.run_ts``). The colour
contracts evaluate ``color-mix()``, ``light-dark()`` and alpha compositing
with the evaluator in ``tests/_colour.py``, whose own answers are held to
known values; what a browser computes is recorded on the machine, never here.

  * DS1 -- exactly three palette files, each one rule under its own
    ``[data-oo-theme="<id>"]`` declaring exactly the roles and
    ``color-scheme``; no other file declares a role; the values are the
    chosen ones.
  * DS2 -- every pair of the design's pair list reaches its ratio in every
    palette, the pairs named by app token and evaluated through the
    derivation layer; a planted low pair is caught.
  * DS3 -- every app token that carries a colour is declared once, in the
    derivation layer, as an expression of roles; no hex colour stands
    outside the palette files, except the QR code's white ground and the
    pre-render's theme-color map.
  * DS4 -- only ``apply.ts`` and the pre-render write the palette
    attribute, the dark class, the density class, the root font size, the
    motion classes or the componion attribute.
  * DS5 -- the pre-render and ``apply.ts`` reach the same root state over
    the matrix of stored values, system settings, storage failure and text
    sizes; when the pre-render cannot run, the static night palette stands.
  * DS6 -- under "Match system" a change of the system's scheme or contrast
    re-applies; nothing stores ``oo-theme``, and ``oo-palette`` is stored
    only by an explicit choice.
  * DS7 -- each palette declares its colour scheme, and the pre-render's
    theme-color map is each palette's ground.
  * DS9 -- the text scale is carried by the root font size: no
    ``--oo-type-scale``, text and space tokens in rem, no text token under
    0.75rem, and ``apply.ts`` sets the root size of each text level.
  * DS10 -- the nine accent names that are inks never appear in a fill;
    an occurrence the census cannot attribute fails.
  * DS11 -- no named ``white`` or ``black`` colour.
  * DS12 -- the focus ring is drawn in the focus ink and never changes a
    control's shape.
  * DS13 -- the Tailwind configuration holds no hex colour, and no
    ``html:not(.dark)`` override layer remains.
  * DS14 -- the appearance choices are "Match system", day, night and high
    contrast, and every stored key of the retired palettes migrates as
    specified.
  * DS15 -- the card, the modal's panel and the tooltip carry the edge
    border that high contrast and forced colours reveal; a forced-colours
    block exists.
  * DS16 -- the motion neutralisation survives: the forced reduction, the
    system's reduction unless full motion is chosen, and the instant motion
    tokens.
  * DS17 -- the colour evaluator gives known answers: mixing with
    premultiplied alpha, compositing, the contrast ratio; values a browser
    computed on the machine are compared once recorded.
  * DS18 -- status washes are read by the status primitives; elsewhere a
    ratchet holds them.
  * DS19 -- every rule or element filled with the accent fill sets its text
    to the on-accent ink.
  * DS20 -- the motion classes the theme path sets are the ones the scroll
    behaviour reads, and smooth scrolling is chosen in one place only (the
    successor of UX11's wiring half, whose last assertion named the
    preferences store as the writer of those classes).
  * DS21 -- every pair of a ground and a text colour that a component sets
    together, in one rule or on one element (a conditional's branches
    followed, a class list a script holds, the Tailwind names the
    configuration gives), reaches 4.5:1 in every palette; digits glued to
    a token, which paint nothing, are named.
  * DS22 -- a mark that carries no text (a progress bar, a legend dot, the
    streaming caret) is drawn in the mark token, never in the accent fill,
    and the mark and chart inks reach 3:1 on the grounds they sit on; the
    mark token never fills something that holds text.
  * DS23 -- a hand-made switch's knob is the knob token on the switch
    tracks, and the field primitives draw their edge in the field token.
  * DS24 -- at startup the older binary theme is retired where it decides
    nothing (a palette chosen, or equal to the system), and where it still
    pins, the choice it means is held for the visit.
  * DS25 -- the derivation layer maps every app token as the design's table
    says, compared by colour in every palette, and every name of a text
    level is one of the two levels.
  * DS26 -- no colour literal stands outside the palette files but the
    rgb debt UR5 holds: no other colour function, no named colour (a
    fallback included), no Tailwind palette class.
  * DS27 -- the base layer Tailwind generates draws in tokens: the default
    border, the ring and its offset, and a field's placeholder.
  * DS28 -- the colours a browser computes are recorded on the machine; until
    then the comparison is owed, and says so.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner.
"""

import functools
import json
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import REPO, check_ledger, files, read, run_ts  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder from the junit file.
BUDGET_S = {
    "test_ds1_three_palette_files_each_declaring_exactly_the_roles": 1.0,
    "test_ds2_every_pair_of_the_design_passes_in_every_palette": 1.0,
    "test_ds3_every_colour_comes_from_a_role[derivation]": 1.0,
    "test_ds3_every_colour_comes_from_a_role[hex]": 1.0,
    "test_ds4_only_the_theme_path_writes_the_theme_state": 1.0,
    "test_ds5_the_prerender_and_the_theme_path_agree_over_the_matrix": 2.0,
    "test_ds6_the_system_choice_follows_the_system_and_only_a_choice_is_stored[node]": 2.0,
    "test_ds6_the_system_choice_follows_the_system_and_only_a_choice_is_stored[wiring]": 1.0,
    "test_ds7_each_palette_declares_its_scheme_and_the_theme_color_map_is_its_ground": 1.0,
    "test_ds9_the_text_scale_is_carried_by_the_root_size[static]": 1.0,
    "test_ds9_the_text_scale_is_carried_by_the_root_size[node]": 2.0,
    "test_ds10_no_ink_is_used_as_a_fill": 1.0,
    "test_ds11_no_named_white_or_black_colour": 1.0,
    "test_ds12_the_focus_ring_is_the_focus_ink_and_keeps_the_shape": 1.0,
    "test_ds13_tailwind_holds_no_hex_and_no_light_override_layer_remains": 1.0,
    "test_ds14_the_choices_are_four_and_every_legacy_key_migrates[node]": 2.0,
    "test_ds14_the_choices_are_four_and_every_legacy_key_migrates[wiring]": 1.0,
    "test_ds15_tone_only_containers_carry_the_edge_and_forced_colours_are_declared": 1.0,
    "test_ds16_the_motion_neutralisation_survives": 1.0,
    "test_ds17_the_colour_evaluator_gives_known_answers": 1.0,
    "test_ds18_status_washes_outside_the_primitives_never_rise": 1.0,
    "test_ds19_every_accent_fill_carries_the_on_accent_text": 1.0,
    "test_ds20_the_motion_classes_the_theme_path_sets_are_the_ones_scrolling_reads": 2.0,
    "test_ds21_the_pairs_the_components_draw_pass_in_every_palette": 2.0,
    "test_ds22_marks_read_the_mark_token_and_reach_three_to_one": 2.0,
    "test_ds23_a_switch_knob_sits_on_the_switch_tracks_and_a_field_edge_is_seen": 2.0,
    "test_ds24_a_legacy_theme_equal_to_the_system_is_retired_at_startup": 2.0,
    "test_ds25_the_derivation_layer_maps_every_token_as_the_design_says": 1.0,
    "test_ds26_no_colour_literal_but_the_ledgered_rgb_debt_and_no_named_colour": 1.0,
    "test_ds27_the_base_layer_tailwind_generates_draws_in_tokens": 2.0,
    "test_ds28_the_browser_computed_colours_are_recorded_or_owed": 1.0,
}

_SRC = "frontend/src"
_STYLES = f"{_SRC}/styles"
_THEME = f"{_STYLES}/theme.css"
_TOKENS = f"{_STYLES}/tokens.css"
_APP_CSS = f"{_SRC}/app.css"
_APP_HTML = f"{_SRC}/app.html"
_APPLY = f"{_SRC}/lib/theme/apply.ts"
_MOTION = f"{_SRC}/lib/motion.ts"
_PREFERENCES = f"{_SRC}/lib/stores/preferences.ts"
_ROOT_LAYOUT = f"{_SRC}/routes/+layout.svelte"
_TAILWIND = "frontend/tailwind.config.js"
_PRIMITIVES = f"{_SRC}/lib/ds/"
_CARD = f"{_SRC}/lib/ds/Card.svelte"
_MODAL = f"{_SRC}/lib/ds/Modal.svelte"
_TOOLTIP = f"{_SRC}/lib/ds/Tooltip.svelte"
_SWITCH = f"{_SRC}/lib/ds/Switch.svelte"
_FIELDS = (f"{_SRC}/lib/ds/Input.svelte", f"{_SRC}/lib/ds/Select.svelte")
_COMPUTED = "tests/fixtures/computed_colors.json"

# Every kind of source file the censuses read.
_EVERY = (".svelte", ".ts", ".js", ".mjs", ".cjs", ".css", ".scss", ".html")
_SCRIPTS = (".svelte", ".ts", ".js", ".mjs", ".cjs")
_STYLED = (".svelte", ".ts", ".js", ".mjs", ".cjs", ".css", ".scss")

_QUOTES = "\"'`"

# ---------------------------------------------------------------------------
# The palettes: their ids, files, schemes and roles
# ---------------------------------------------------------------------------
_PALETTE_IDS = ("day", "night", "high-contrast")
_PALETTE_FILES = {pid: f"{_STYLES}/theme-{pid}.css" for pid in _PALETTE_IDS}
_SCHEMES = {"day": "light", "night": "dark", "high-contrast": "dark"}

# role: (day, night, high contrast). Day and night are the owner's chosen
# palettes; high contrast is a dark, warm set of its own.
_ROLE_TABLE = {
    "bg": ("#F2EDE4", "#1F1A17", "#0B0A09"),
    "surface": ("#FAF7F1", "#27221F", "#151311"),
    "surface-2": ("#EAE3D7", "#2D2825", "#1D1A17"),
    "sunken": ("#E5DDD0", "#191513", "#000000"),
    "tint-1": ("#D9E1CF", "#473A25", "#1B2616"),
    "tint-2": ("#ECD8D1", "#483336", "#2A1B1D"),
    "tint-3": ("#D4DDE3", "#31382F", "#172128"),
    "text": ("#37312B", "#ECE5DB", "#FFFFFF"),
    "text-muted": ("#645A4F", "#B5AEA7", "#E3DCD2"),
    "primary": ("#94A583", "#A88C76", "#CFE6B4"),
    "on-primary": ("#1F2919", "#27201B", "#0B1206"),
    "primary-ink": ("#4B5E43", "#D9C1B0", "#C2E3A2"),
    "secondary-ink": ("#714C6E", "#C4ADD1", "#EACBF2"),
    "rule": ("#837869", "#7A716B", "#B3A99D"),
    "rule-soft": ("#E2DACD", "#36302D", "#7A7167"),
    "code-bg": ("#EEE8DE", "#312B28", "#17140F"),
    "syn-keyword": ("#405E78", "#9EBCD2", "#A9D5FF"),
    "syn-name": ("#44603D", "#BBCAAF", "#C8EDA6"),
    "stop-ink": ("#93463A", "#E1A096", "#FFB4A6"),
    "status-ok": ("#678A5A", "#95AE8B", "#9EDC86"),
    "shadow-1": ("rgba(86,70,52,.07)", "rgba(12,8,6,.30)", "rgba(0,0,0,.60)"),
    "shadow-2": ("rgba(86,70,52,.12)", "rgba(12,8,6,.44)", "rgba(0,0,0,.80)"),
    "warn-ink": ("#77591D", "#DCBF86", "#F4D38A"),
    "warn-tone": ("#B08A3E", "#C9A566", "#F4D38A"),
    "success-ink": ("#4F5E42", "#C0CAB3", "#B9E6A3"),
    "edge": ("transparent", "transparent", "#7A7167"),
}
_ROLES = tuple(_ROLE_TABLE)
_ROLE_PREFIX = "--oo-role-"


# ---------------------------------------------------------------------------
# Reading CSS: rules, declarations, style blocks
# ---------------------------------------------------------------------------
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>(.*?)</style>", re.S)


def _css_of(path, text):
    """The CSS a source holds: the file itself, or a component's style blocks."""
    if path.endswith((".css", ".scss")):
        return text
    if path.endswith(".svelte"):
        return "\n".join(_STYLE_BLOCK.findall(text))
    return ""


def _skip_string(text, index):
    """The index just past the string opening at ``index``."""
    quote = text[index]
    index += 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == quote:
            return index + 1
        index += 1
    return index


def _rules(css, context=()):
    """``[(selector, body, context)]`` for every rule of ``css``, the rules
    of an at-block read with the at-rule's prelude added to ``context``.
    Comments are removed first; strings are skipped when braces are matched."""
    css = _CSS_COMMENT.sub(" ", css)
    found = []
    index, start = 0, 0
    while index < len(css):
        char = css[index]
        if char in "\"'":
            index = _skip_string(css, index)
            continue
        if char == ";":
            start = index + 1
        elif char == "{":
            prelude = css[start:index].strip()
            depth, cursor = 1, index + 1
            while cursor < len(css) and depth:
                if css[cursor] in "\"'":
                    cursor = _skip_string(css, cursor)
                    continue
                if css[cursor] == "{":
                    depth += 1
                elif css[cursor] == "}":
                    depth -= 1
                cursor += 1
            body = css[index + 1:cursor - 1]
            if prelude.startswith("@"):
                found.extend(_rules(body, context + (prelude,)))
            else:
                found.append((prelude, body, context))
            index = start = cursor
            continue
        elif char == "}":
            start = index + 1
        index += 1
    return found


def _declarations(body):
    """``[(property, value)]`` of a rule body, split at the top-level
    semicolons (never inside parentheses or strings); a nested rule's text
    is not read as a declaration."""
    out = []
    depth, start, index = 0, 0, 0
    while index <= len(body):
        char = body[index] if index < len(body) else ";"
        if char in "\"'" and index < len(body):
            index = _skip_string(body, index)
            continue
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == ";" and depth <= 0:
            part = body[start:index].strip()
            if ":" in part and "{" not in part:
                name, value = part.split(":", 1)
                out.append((name.strip(), value.strip()))
            start = index + 1
        index += 1
    return out


def _sources(kinds=_EVERY, exclude=()):
    return {path: read(path) for path in files(kinds, exclude=exclude)}


def _palette_rules(text):
    """The rules of one palette file."""
    return _rules(text)


_PALETTE_SELECTOR = re.compile(r"""^\[data-oo-theme=(["'])([\w-]+)\1\]$""")


def _palette_declarations(path):
    """``(selector, [(name, value)])`` of a palette file's single rule, or
    raises naming what the file holds instead."""
    rules = _palette_rules(read(path))
    assert len(rules) == 1, f"{path} holds one rule, not {len(rules)}"
    selector, body, context = rules[0]
    assert not context, f"{path}'s rule sits in {context}, not at the top level"
    return selector, _declarations(body)


def _palettes():
    """``{palette id: ({role: value}, scheme)}`` read from the three files."""
    out = {}
    for pid, path in _PALETTE_FILES.items():
        _, declared = _palette_declarations(path)
        roles = {
            name[len(_ROLE_PREFIX):]: value for name, value in declared
            if name.startswith(_ROLE_PREFIX)
        }
        scheme = dict(declared).get("color-scheme")
        out[pid] = (roles, scheme)
    return out


def _derivation_blocks(text):
    """The rules of ``text`` whose selector is exactly ``[data-oo-theme]``."""
    return [
        (selector, body, context) for selector, body, context in _rules(text)
        if re.fullmatch(r"\[data-oo-theme\]", selector.strip())
    ]


def _derivation(text=None):
    """``{token: value}`` of the derivation layer's single block."""
    blocks = _derivation_blocks(read(_THEME) if text is None else text)
    assert len(blocks) == 1, f"{_THEME} holds one [data-oo-theme] rule, not {len(blocks)}"
    return dict(_declarations(blocks[0][1]))


_NAMED_PALETTE = re.compile(r"""\[data-oo-theme=(["'])([\w-]+)\1\]""")


def _palette_rules_of(text):
    """``[(selector, palette ids, body)]`` of the top-level rules of ``text``
    whose selector names palettes and nothing else
    (``[data-oo-theme="day"]``, or a list of them): what the derivation
    layer gives one palette and not another."""
    out = []
    for selector, body, context in _rules(text):
        parts = [part.strip() for part in selector.split(",")]
        ids = [m.group(2) for part in parts for m in [_NAMED_PALETTE.fullmatch(part)] if m]
        if not context and parts and len(ids) == len(parts):
            out.append((selector.strip(), tuple(ids), body))
    return out


def _palette_overrides(text=None):
    """``{palette id: {token: value}}`` of the derivation layer's per-palette rules."""
    out = {pid: {} for pid in _PALETTE_IDS}
    for _, ids, body in _palette_rules_of(read(_THEME) if text is None else text):
        for pid in ids:
            out.setdefault(pid, {}).update(dict(_declarations(body)))
    return out


def _with_palette(tokens, pid):
    """The app tokens as palette ``pid`` sees them: the block's, over what
    the rule naming the palette gives (the two never share a name, DS3)."""
    return {**_palette_overrides().get(pid, {}), **tokens}


# ---------------------------------------------------------------------------
# Colour literals: hex, functions, named colours
# ---------------------------------------------------------------------------
_HEX = re.compile(r"(?<![\w&#])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})(?![\w-])")
_NOT_A_COLOUR_BEFORE = re.compile(r"""(?:href\s*=\s*["']?|url\(\s*["']?|\{)$""")
_COLOUR_FUNCTION = re.compile(r"(?<![\w-])(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(", re.I)
_NAMED_COLOURS = frozenset("""
aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue
blueviolet brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk
crimson cyan darkblue darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki
darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen
darkslateblue darkslategray darkslategrey darkturquoise darkviolet deeppink deepskyblue
dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro ghostwhite
gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory khaki
lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan
lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen
lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen
magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen
mediumslateblue mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream
mistyrose moccasin navajowhite navy oldlace olive olivedrab orange orangered orchid
palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum
powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown
seagreen seashell sienna silver skyblue slateblue slategray slategrey snow springgreen
steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke yellow
yellowgreen canvas canvastext linktext visitedtext activetext buttonface buttontext
buttonborder field fieldtext highlight highlighttext selecteditem selecteditemtext mark
marktext graytext accentcolor accentcolortext
""".split())
_WORD = re.compile(r"(?<![\w-])([A-Za-z]+)(?![\w-])")
_VAR = re.compile(r"var\(\s*(--[\w-]+)")


def _hexes(text):
    """``[(offset, literal)]`` of every hex colour in ``text``: not an HTML
    character reference, and not a fragment in a link, a ``url()`` or a
    Svelte block (``{#each``)."""
    found = []
    for match in _HEX.finditer(text):
        before = text[max(0, match.start() - 12):match.start()]
        if _NOT_A_COLOUR_BEFORE.search(before):
            continue
        found.append((match.start(), match.group(0)))
    return found


def _colour_literals(value):
    """The colour literals a CSS value spells: hex, colour functions other
    than ``color-mix``, ``light-dark`` and ``var``, and named colours (the
    keywords ``transparent`` and ``currentColor`` carry no hue and are not
    literals here)."""
    literals = [literal for _, literal in _hexes(value)]
    literals += [m.group(0) for m in _COLOUR_FUNCTION.finditer(value)]
    stripped = _VAR.sub("var(", value)
    literals += [
        word for word in _WORD.findall(stripped)
        if word.lower() in _NAMED_COLOURS
    ]
    return literals


# ---------------------------------------------------------------------------
# Module-level constants of the frontend's scripts (identifier -> string)
# ---------------------------------------------------------------------------
_STRING_CONST = re.compile(
    r"(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*(?::\s*[^=;\n]+)?=\s*(['\"`])([^'\"`\n$]*)\2"
)


def _string_constants(sources):
    """``{identifier: value}`` of every constant bound to a plain string in
    the frontend's scripts; an identifier bound to two values is dropped."""
    seen = {}
    for path, text in sources.items():
        if not path.endswith(_SCRIPTS):
            continue
        for name, _, value in _STRING_CONST.findall(text):
            seen.setdefault(name, set()).add(value)
    return {name: next(iter(values)) for name, values in seen.items() if len(values) == 1}


def _call_arguments(text, start):
    """The text between the parenthesis opening at ``start`` and its match."""
    depth, index = 0, start
    while index < len(text):
        char = text[index]
        if char in _QUOTES:
            index = _skip_string(text, index)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
        index += 1
    return text[start + 1:]


def _function_body(text, name):
    """The body of the function ``name`` (a ``function`` declaration, or an
    arrow assigned to a ``const``/``let``), braces included, or None."""
    match = re.search(
        r"(?:function\s+" + re.escape(name) + r"\s*(?:<[^>]*>)?\s*\([^)]*\)[^{]*"
        r"|(?:const|let)\s+" + re.escape(name) + r"\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::[^=]*)?=>\s*)\{",
        text,
    )
    if not match:
        return None
    depth, index = 0, match.end() - 1
    while index < len(text):
        char = text[index]
        if char in _QUOTES:
            index = _skip_string(text, index)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[match.end() - 1:index + 1]
        index += 1
    return None


def _count_in(sources, count_fn):
    counts = {}
    for path, text in sources.items():
        found = count_fn(path, text)
        if found:
            counts[path] = found
    return counts


# ---------------------------------------------------------------------------
# The pre-render in app.html
# ---------------------------------------------------------------------------
_INLINE_SCRIPT = re.compile(r"<script\b(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script>", re.S)
_ATTRIBUTE = re.compile(r"""([\w:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?""")
_MAP_ENTRY = re.compile(
    r"""(["']?)(day|night|high-contrast)\1\s*:\s*["'](#[0-9A-Fa-f]{3,8})["']"""
)


def _prerender_script(html=None):
    html = read(_APP_HTML) if html is None else html
    head = html.split("</head>", 1)[0]
    return "\n;\n".join(_INLINE_SCRIPT.findall(head))


def _html_attributes(html=None):
    """The attributes written on ``<html>`` in app.html."""
    html = read(_APP_HTML) if html is None else html
    tag = re.search(r"<html\b([^>]*)>", html)
    assert tag, "app.html has an <html> tag"
    attributes = {}
    for match in _ATTRIBUTE.finditer(tag.group(1)):
        name = match.group(1).lower()
        value = next((g for g in match.groups()[1:] if g is not None), "")
        attributes[name] = value
    return attributes


def _theme_colour_metas(html=None):
    """The ``<meta name="theme-color">`` tags of app.html, as attribute dicts."""
    html = read(_APP_HTML) if html is None else html
    metas = []
    for tag in re.findall(r"<meta\b([^>]*)>", html):
        attributes = {}
        for match in _ATTRIBUTE.finditer(tag):
            value = next((g for g in match.groups()[1:] if g is not None), "")
            attributes[match.group(1).lower()] = value
        if attributes.get("name") == "theme-color":
            metas.append(attributes)
    return metas


def _theme_colour_map(script):
    """``{palette id: [hex, ...]}`` of the pre-render's theme-color map."""
    found = {}
    for _, pid, colour in _MAP_ENTRY.findall(script):
        found.setdefault(pid, []).append(colour)
    return found


# ---------------------------------------------------------------------------
# The Node driver: the pre-render and apply.ts over fake browser objects
# ---------------------------------------------------------------------------
_DRIVER = r"""
import vm from 'node:vm';
import { readFileSync } from 'node:fs';

const clause = process.argv[2];
// The input travels in a file: the matrix is larger than one environment
// string may be.
const input = JSON.parse(process.env.OO_INPUT_FILE ? readFileSync(process.env.OO_INPUT_FILE, 'utf8') : 'null');
const apply = process.env.OO_APPLY ? await import(process.env.OO_APPLY) : null;
const motion = process.env.OO_MOTION ? await import(process.env.OO_MOTION) : null;
const prerender = process.env.OO_PRERENDER || '';
const staticAttributes = JSON.parse(process.env.OO_STATIC || '{}');
const staticMetas = JSON.parse(process.env.OO_METAS || '[]');

// An element: attributes, a class list, an inline style, a dataset.
function element(tag, attributes = {}) {
    const attrs = new Map();
    const classes = new Set();
    const style = new Map();
    const el = { tagName: String(tag).toUpperCase(), ownerDocument: null, children: [] };
    const setClass = (value) => {
        classes.clear();
        String(value).split(/\s+/).filter(Boolean).forEach((c) => classes.add(c));
    };
    el.setAttribute = (name, value) => {
        name = String(name).toLowerCase();
        if (name === 'class') setClass(value);
        else attrs.set(name, String(value));
    };
    el.getAttribute = (name) => {
        name = String(name).toLowerCase();
        if (name === 'class') return classes.size ? [...classes].join(' ') : null;
        return attrs.has(name) ? attrs.get(name) : null;
    };
    el.hasAttribute = (name) => el.getAttribute(name) !== null;
    el.removeAttribute = (name) => {
        name = String(name).toLowerCase();
        if (name === 'class') classes.clear();
        else attrs.delete(name);
    };
    el.toggleAttribute = (name, force) => {
        const on = force === undefined ? !el.hasAttribute(name) : !!force;
        if (on) el.setAttribute(name, ''); else el.removeAttribute(name);
        return on;
    };
    el.classList = {
        add: (...names) => names.forEach((n) => classes.add(String(n))),
        remove: (...names) => names.forEach((n) => classes.delete(String(n))),
        toggle: (name, force) => {
            name = String(name);
            const on = force === undefined ? !classes.has(name) : !!force;
            if (on) classes.add(name); else classes.delete(name);
            return on;
        },
        contains: (name) => classes.has(String(name)),
        replace: (a, b) => {
            if (!classes.has(String(a))) return false;
            classes.delete(String(a));
            classes.add(String(b));
            return true;
        },
        get length() { return classes.size; },
    };
    Object.defineProperty(el, 'className', {
        get: () => [...classes].join(' '),
        set: (value) => setClass(value),
    });
    el.style = {
        setProperty: (name, value) => {
            if (value === null || value === undefined || value === '') style.delete(String(name));
            else style.set(String(name), String(value));
        },
        removeProperty: (name) => { style.delete(String(name)); },
        getPropertyValue: (name) => style.get(String(name)) ?? '',
    };
    Object.defineProperty(el.style, 'fontSize', {
        get: () => style.get('font-size') ?? '',
        set: (value) => el.style.setProperty('font-size', value),
    });
    el.dataset = new Proxy({}, {
        get: (_, key) => el.getAttribute('data-' + String(key).replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())) ?? undefined,
        set: (_, key, value) => {
            el.setAttribute('data-' + String(key).replace(/[A-Z]/g, (c) => '-' + c.toLowerCase()), value);
            return true;
        },
        deleteProperty: (_, key) => {
            el.removeAttribute('data-' + String(key).replace(/[A-Z]/g, (c) => '-' + c.toLowerCase()));
            return true;
        },
    });
    for (const property of ['content', 'name']) {
        Object.defineProperty(el, property, {
            get: () => el.getAttribute(property) ?? '',
            set: (value) => el.setAttribute(property, value),
        });
    }
    el.appendChild = (child) => { el.children.push(child); return child; };
    el._attributes = attrs;
    el._classes = classes;
    el._style = style;
    for (const [name, value] of Object.entries(attributes)) el.setAttribute(name, value);
    return el;
}

// A document whose root is ``root``; its head holds app.html's theme-color
// metas. ``throwing`` makes the root unreachable, as a broken page would.
function documentFor(root, throwing = false) {
    const head = element('head');
    for (const attributes of staticMetas) head.appendChild(element('meta', attributes));
    const matches = (el, selector) => el.tagName === 'META' && /theme-color/.test(selector)
        && el.getAttribute('name') === 'theme-color';
    const doc = {
        head,
        createElement: (tag) => { const el = element(tag); el.ownerDocument = doc; return el; },
        querySelector: (selector) => head.children.find((el) => matches(el, selector)) ?? null,
        querySelectorAll: (selector) => head.children.filter((el) => matches(el, selector)),
        getElementsByTagName: (tag) => head.children.filter((el) => el.tagName === String(tag).toUpperCase()),
    };
    head.querySelector = doc.querySelector;
    Object.defineProperty(doc, 'documentElement', {
        get: () => { if (throwing) throw new Error('the root element cannot be reached'); return root; },
    });
    root.ownerDocument = doc;
    return doc;
}

function themeColour(doc) {
    const metas = doc.head.children.filter((el) => el.tagName === 'META' && el.getAttribute('name') === 'theme-color');
    return metas.length ? metas[metas.length - 1].getAttribute('content') : null;
}

// Storage over ``entries``; every write is logged.
function storageFor(entries, writes) {
    const data = { ...entries };
    return {
        getItem: (key) => (Object.prototype.hasOwnProperty.call(data, key) ? data[key] : null),
        setItem: (key, value) => { writes.push([String(key), String(value)]); data[key] = String(value); },
        removeItem: (key) => { writes.push([String(key), null]); delete data[key]; },
        clear: () => { writes.push(['*', null]); },
        key: (i) => Object.keys(data)[i] ?? null,
        get length() { return Object.keys(data).length; },
        _data: data,
    };
}

function blockedStorage(writes) {
    const blocked = () => { throw new Error('storage is blocked'); };
    return {
        getItem: blocked,
        setItem: (key, value) => { writes.push([String(key), String(value)]); blocked(); },
        removeItem: blocked, clear: blocked, key: blocked,
        get length() { return blocked(); },
    };
}

// The system: a colour scheme and a contrast preference, or no media query
// support at all (``state`` null). ``change`` flips the state and tells every
// listener whose query's answer changed.
function mediaFor(state) {
    if (!state) return undefined;
    const current = { ...state };
    const listeners = [];
    const answer = (query) => {
        const q = String(query).replace(/\s+/g, '').toLowerCase();
        if (q === '(prefers-color-scheme:light)') return current.scheme === 'light';
        if (q === '(prefers-color-scheme:dark)') return current.scheme === 'dark';
        if (q === '(prefers-contrast:more)') return !!current.more;
        if (q === '(prefers-contrast:no-preference)') return !current.more;
        return false;
    };
    const matchMedia = (query) => {
        const list = {
            media: String(query),
            get matches() { return answer(query); },
            onchange: null,
            addEventListener: (type, fn) => { if (type === 'change') listeners.push({ list, fn }); },
            removeEventListener: (type, fn) => {
                const i = listeners.findIndex((l) => l.list === list && l.fn === fn);
                if (i >= 0) listeners.splice(i, 1);
            },
            addListener: (fn) => listeners.push({ list, fn }),
            removeListener: (fn) => {
                const i = listeners.findIndex((l) => l.list === list && l.fn === fn);
                if (i >= 0) listeners.splice(i, 1);
            },
        };
        return list;
    };
    matchMedia.change = (next) => {
        const before = listeners.map((l) => l.list.matches);
        Object.assign(current, next);
        const pending = [];
        listeners.forEach((l, i) => { if (l.list.matches !== before[i]) pending.push(l); });
        for (const l of pending) {
            const event = { matches: l.list.matches, media: l.list.media };
            l.fn(event);
            if (typeof l.list.onchange === 'function') l.list.onchange(event);
        }
    };
    matchMedia.listening = () => listeners.length;
    return matchMedia;
}

function snapshot(root) {
    const attributes = {};
    for (const [name, value] of [...root._attributes.entries()].sort()) attributes[name] = value;
    return {
        attributes,
        classes: [...root._classes].sort(),
        fontSize: root._style.get('font-size') ?? '',
        style: Object.fromEntries([...root._style.entries()].sort()),
    };
}

// The pre-render over one environment.
function runPrerender(entries, media, blocked, throwing = false) {
    const root = element('html', staticAttributes);
    const doc = documentFor(root, throwing);
    const writes = [];
    const sandbox = { document: doc, console: { log() {}, warn() {}, error() {} } };
    const matchMedia = mediaFor(media);
    if (matchMedia) sandbox.matchMedia = matchMedia;
    if (blocked) {
        Object.defineProperty(sandbox, 'localStorage', {
            get: () => { throw new Error('storage is blocked'); },
        });
    } else {
        sandbox.localStorage = storageFor(entries, writes);
    }
    sandbox.window = sandbox;
    sandbox.self = sandbox;
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    let threw = null;
    try {
        vm.runInContext(prerender, sandbox, { timeout: 1000 });
    } catch (error) {
        threw = String(error && error.message || error);
    }
    return { state: snapshot(root), themeColour: themeColour(doc), threw, writes };
}

// apply.ts over the same environment.
function runApply(entries, media, blocked) {
    const root = element('html', staticAttributes);
    documentFor(root);
    const writes = [];
    const stored = blocked ? blockedStorage(writes) : storageFor(entries, writes);
    const env = { matchMedia: mediaFor(media) };
    let threw = null;
    let resolved = null;
    try {
        resolved = apply.resolve(stored, env);
        apply.applyTo(root, resolved);
    } catch (error) {
        threw = String(error && error.message || error);
    }
    const picked = resolved && typeof resolved === 'object'
        ? { choice: resolved.choice ?? null, theme: resolved.theme ?? null } : null;
    return { state: snapshot(root), threw, writes, resolved: picked };
}

function entriesOf(row) {
    const entries = {};
    for (const [key, value] of Object.entries(row.stored || {})) {
        if (value !== null && value !== undefined) entries[key] = value;
    }
    return entries;
}

const clauses = {
    // Every row of the matrix: what the pre-render and apply.ts reach.
    matrix: () => input.rows.map((row) => {
        const entries = entriesOf(row);
        return {
            pre: runPrerender(entries, row.media, row.blocked),
            post: runApply(entries, row.media, row.blocked),
        };
    }),
    // The pre-render when the page's root cannot be reached.
    broken: () => runPrerender({}, { scheme: 'light', more: false }, false, true),
    // What resolve() chooses, row by row.
    resolve: () => input.rows.map((row) => runApply(entriesOf(row), row.media, row.blocked)),
    // The exported choices and their labels.
    choices: () => ({ choices: apply.CHOICES ?? null, labels: apply.CHOICE_LABELS ?? null }),
    // Re-applying one root: a second application leaves the state a fresh
    // root reaches, whatever the first one set.
    reapply: () => input.pairs.map(([first, second]) => {
        const root = element('html', staticAttributes);
        documentFor(root);
        const fresh = element('html', staticAttributes);
        documentFor(fresh);
        const writes = [];
        const run = (target, row) => apply.applyTo(
            target, apply.resolve(storageFor(entriesOf(row), writes), { matchMedia: mediaFor(row.media) }),
        );
        run(root, first);
        run(root, second);
        run(fresh, second);
        return { again: snapshot(root), fresh: snapshot(fresh), writes };
    }),
    // Following the system: flips of the scheme and the contrast, under a
    // stored choice that may change between flips, then after stopping.
    follow: () => input.scenarios.map((scenario) => {
        const root = element('html', staticAttributes);
        documentFor(root);
        const writes = [];
        const stored = storageFor(entriesOf(scenario), writes);
        const matchMedia = mediaFor(scenario.media);
        const env = { matchMedia };
        apply.applyTo(root, apply.resolve(stored, env));
        const seen = [snapshot(root).attributes['data-oo-theme'] ?? null];
        let stop = null;
        let threw = null;
        try {
            stop = apply.followSystem(stored, env, root);
            for (const step of scenario.steps) {
                if (step.store) {
                    for (const [key, value] of Object.entries(step.store)) stored._data[key] = value;
                }
                if (step.stop) {
                    if (typeof stop === 'function') stop();
                    continue;
                }
                if (matchMedia) matchMedia.change(step.system);
                seen.push(snapshot(root).attributes['data-oo-theme'] ?? null);
            }
        } catch (error) {
            threw = String(error && error.message || error);
        }
        return {
            seen, threw, writes, stopIsFunction: typeof stop === 'function',
            listening: matchMedia ? matchMedia.listening() : null,
        };
    }),
    // Startup as the preferences store runs it: the old binary theme retired
    // when it no longer decides anything, the stored state applied, then the
    // system followed through its changes. What storage is left is returned.
    retire: () => input.scenarios.map((scenario) => {
        const root = element('html', staticAttributes);
        documentFor(root);
        const writes = [];
        const stored = scenario.blocked ? blockedStorage(writes) : storageFor(entriesOf(scenario), writes);
        const matchMedia = mediaFor(scenario.media);
        const env = { matchMedia };
        const seen = [];
        let threw = null;
        let pin;
        try {
            if (typeof apply.retireLegacyTheme === 'function') pin = apply.retireLegacyTheme(stored, env);
            apply.applyTo(root, apply.resolve(stored, env));
            seen.push(snapshot(root).attributes['data-oo-theme'] ?? null);
            apply.followSystem(stored, env, root);
            for (const step of scenario.steps) {
                if (matchMedia) matchMedia.change(step.system);
                seen.push(snapshot(root).attributes['data-oo-theme'] ?? null);
            }
        } catch (error) {
            threw = String(error && error.message || error);
        }
        return { seen, threw, writes, left: stored._data ? { ...stored._data } : null, pin: pin ?? null };
    }),
    // The motion classes apply.ts sets, and the ones motion.ts reads.
    motion: () => ({
        reads: [motion.MOTION_REDUCED_CLASS, motion.MOTION_FULL_CLASS],
        sets: Object.fromEntries(input.values.map((value) => {
            const run = runApply(value === null ? {} : { 'oo-motion': value }, { scheme: 'dark', more: false }, false);
            return [String(value), run.state.classes];
        })),
    }),
};

const result = await clauses[clause]();
console.log('RESULT ' + JSON.stringify(result));
console.log('PASS ' + clause);
"""


def _node(clause, data=None, *, with_motion=False):
    """Runs one clause of the driver over apply.ts (and motion.ts) with the
    pre-render of app.html, and returns what it printed."""
    html = read(_APP_HTML)
    modules = {"OO_APPLY": _APPLY}
    if with_motion:
        modules["OO_MOTION"] = _MOTION
    with tempfile.TemporaryDirectory(prefix="oo_ds_input_") as tmp:
        given = Path(tmp) / "input.json"
        given.write_text(json.dumps(data), encoding="utf-8")
        out = run_ts(
            modules, _DRIVER, clause,
            env={
                "OO_INPUT_FILE": str(given),
                "OO_PRERENDER": _prerender_script(html),
                "OO_STATIC": json.dumps(_html_attributes(html)),
                "OO_METAS": json.dumps(_theme_colour_metas(html)),
            },
        )
    results = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, f"the driver printed no single RESULT line:\n{out[-2000:]}"
    return json.loads(results[0][len("RESULT "):])


# ---------------------------------------------------------------------------
# What the theme path must decide: the specification as an oracle
# ---------------------------------------------------------------------------
_CHOICES = ["system", "day", "night", "high-contrast"]
_CHOICE_LABELS = {
    "system": "Match system", "day": "Day", "night": "Night", "high-contrast": "High contrast",
}
_LEGACY = {"anthracite": "night", "slate": "night", "parchment": "day", "linen": "day"}
_STORED_PALETTES = (None, "system", "day", "night", "high-contrast", *_LEGACY, "sepia")
_STORED_THEMES = (None, "light", "dark")
_MEDIA = {
    "light": {"scheme": "light", "more": False},
    "dark": {"scheme": "dark", "more": False},
    "light-more": {"scheme": "light", "more": True},
    "dark-more": {"scheme": "dark", "more": True},
    "none": None,
}
_SIZES = {None: "100%", "small": "92%", "default": "100%", "large": "109%", "x-large": "118%"}
_DENSITIES = (None, "compact", "comfortable", "spacious", "roomy")
_MOTIONS = (None, "system", "reduced", "full", "still")
_COMPONION = (None, "on", "off", "maybe")


def _expected(palette, theme, media, blocked=False):
    """``(choice, palette shown)`` the specification gives for one row."""
    if blocked:
        palette = theme = None
    os_light = media is not None and media["scheme"] == "light"
    if palette in _CHOICES:
        choice = palette
    elif palette in _LEGACY:
        choice = _LEGACY[palette]
    elif theme in ("light", "dark"):
        os_says = "light" if os_light else "dark"
        choice = "system" if theme == os_says else ("day" if theme == "light" else "night")
    else:
        choice = "system"
    if choice != "system":
        return choice, choice
    if media is not None and media["more"]:
        return choice, "high-contrast"
    return choice, "day" if os_light else "night"


def _row(palette, theme, media_name, blocked, size, index):
    stored = {
        "oo-palette": palette, "oo-theme": theme, "oo-type-scale": size,
        "oo-density": _DENSITIES[index % len(_DENSITIES)],
        "oo-motion": _MOTIONS[(index // 3) % len(_MOTIONS)],
        "oo-componion": _COMPONION[(index // 7) % len(_COMPONION)],
    }
    return {"stored": stored, "media": _MEDIA[media_name], "media_name": media_name, "blocked": blocked}


def _matrix():
    rows = []
    for palette in _STORED_PALETTES:
        for theme in _STORED_THEMES:
            for media_name in _MEDIA:
                for blocked in (False, True):
                    for size in _SIZES:
                        rows.append(_row(palette, theme, media_name, blocked, size, len(rows)))
    return rows


# ===========================================================================
# DS1 -- exactly three palette files, each declaring exactly the roles
# ===========================================================================
def _colour_key(value):
    """A value compared by what it means: hex upper-cased, an rgba() as its
    four numbers, a keyword lower-cased."""
    value = value.strip()
    if value.startswith("#"):
        return value.upper()
    match = re.fullmatch(r"rgba?\(\s*([^)]*)\)", value)
    if match:
        parts = [p for p in re.split(r"[\s,/]+", match.group(1)) if p]
        return tuple(round(float(p), 4) for p in parts)
    return value.lower()


def _roles_declared(path, text):
    """The roles a source declares: in a rule, or set by a script."""
    found = re.findall(r"(?<![\w-])(" + re.escape(_ROLE_PREFIX) + r"[\w-]+)\s*:", _css_of(path, text))
    found += re.findall(r"setProperty\(\s*['\"`](" + re.escape(_ROLE_PREFIX) + r"[\w-]+)", text)
    return found


def test_ds1_three_palette_files_each_declaring_exactly_the_roles():
    listed = [path for path in files((".css",), within=_STYLES) if re.search(r"/theme-[^/]+\.css$", path)]
    assert sorted(listed) == sorted(_PALETTE_FILES.values()), (
        f"the palette files are exactly day, night and high contrast: {sorted(listed)}"
    )

    expected = {f"{_ROLE_PREFIX}{role}" for role in _ROLES} | {"color-scheme"}
    assert len(expected) == 27, "twenty-six roles and the colour scheme"
    for pid, path in _PALETTE_FILES.items():
        selector, declared = _palette_declarations(path)
        match = _PALETTE_SELECTOR.fullmatch(selector.strip())
        assert match and match.group(2) == pid, (
            f"{path} declares under [data-oo-theme=\"{pid}\"], so it applies to the "
            f"root and to any subtree: {selector!r}"
        )
        names = [name for name, _ in declared]
        doubled = sorted({name for name in names if names.count(name) > 1})
        assert not doubled, f"{path} declares each property once: {doubled}"
        assert set(names) == expected, (
            f"{path} declares exactly the roles and color-scheme: "
            f"missing {sorted(expected - set(names))}, extra {sorted(set(names) - expected)}"
        )

    assert _roles_declared(
        "frontend/src/x.svelte", "<style>.x { --oo-role-bg: #000; }</style><script>el.style.setProperty('--oo-role-text', v)</script>"
    ) == ["--oo-role-bg", "--oo-role-text"], "the census reads a role declared in a rule and one set by a script"
    elsewhere = {}
    for path, text in _sources(_STYLED).items():
        if path in _PALETTE_FILES.values():
            continue
        found = _roles_declared(path, text)
        if found:
            elsewhere[path] = sorted(set(found))
    assert not elsewhere, f"only the palette files declare a role: {elsewhere}"

    for column, pid in enumerate(_PALETTE_IDS):
        _, declared = _palette_declarations(_PALETTE_FILES[pid])
        values = dict(declared)
        wrong = {
            role: values.get(f"{_ROLE_PREFIX}{role}")
            for role, row in _ROLE_TABLE.items()
            if _colour_key(values.get(f"{_ROLE_PREFIX}{role}", "")) != _colour_key(row[column])
        }
        assert not wrong, f"the {pid} palette holds the chosen values: differs at {wrong}"


# ===========================================================================
# DS2 -- every pair of the design passes in every palette
# ===========================================================================
_GROUNDS_8 = (
    "--oo-bg-base", "--oo-bg-surface", "--oo-bg-overlay", "--oo-bg-subtle",
    "--oo-bg-tint-1", "--oo-bg-tint-2", "--oo-bg-tint-3", "--oo-bg-code",
)
_GROUNDS_7 = _GROUNDS_8[:7]
_GROUNDS_5 = _GROUNDS_8[:5]
_GROUNDS_4 = ("--oo-bg-base", "--oo-bg-surface", "--oo-bg-subtle", "--oo-bg-overlay")
_STATUS = (
    ("--oo-success", "--oo-success-bg"), ("--oo-warning", "--oo-warning-bg"),
    ("--oo-error", "--oo-error-bg"), ("--oo-info", "--oo-info-bg"),
)


def _pairs(pid):
    """The design's pair list for one palette: ``(ink, ground, least ratio)``;
    a ground is a token, or ``("over", wash, ground)`` for a translucent wash
    composited over an opaque ground."""
    pairs = []
    for ink in ("--oo-fg-primary", "--oo-fg-muted"):
        pairs += [(ink, ground, 4.5) for ground in _GROUNDS_8]
    for ink in ("--oo-acc-ink", "--oo-acc-ink-2", "--oo-fg-stop"):
        pairs += [(ink, ground, 4.5) for ground in _GROUNDS_5]
    for ink, wash in _STATUS:
        pairs += [(ink, ground, 4.5) for ground in _GROUNDS_8]
        pairs += [(ink, ("over", wash, ground), 4.5) for ground in ("--oo-bg-surface", "--oo-bg-base")]
    pairs += [
        ("--oo-fg-on-accent", "--oo-acc-fill", 4.5),
        ("--oo-fg-on-accent", "--oo-acc-fill-hover", 4.5),
        ("--oo-fg-on-semantic", "--oo-error", 4.5),
        ("--oo-fg-on-semantic", "--oo-success", 4.5),
        ("--oo-btn-secondary-fg", "--oo-btn-secondary-hover", 4.5),
    ]
    for ink in ("--oo-fg-muted", "--oo-fg-primary"):
        pairs += [(ink, ("over", "--oo-bg-hover", ground), 4.5) for ground in _GROUNDS_7]
    pairs += [("--oo-bd-strong", ground, 3.0) for ground in _GROUNDS_4]
    pairs += [("--oo-focus-ink", ground, 3.0) for ground in _GROUNDS_4]
    pairs += [("--oo-status-ok", "--oo-bg-surface", 3.0), ("--oo-status-ok", "--oo-bg-overlay", 3.0)]
    pairs += [
        ("--oo-toggle-knob", "--oo-switch-on", 3.0),
        ("--oo-toggle-knob", "--oo-switch-off", 3.0),
        ("--oo-switch-on", "--oo-bg-surface", 3.0),
        ("--oo-switch-off", "--oo-bg-surface", 3.0),
        ("--oo-code-keyword", "--oo-bg-code", 4.5),
        ("--oo-code-name", "--oo-bg-code", 4.5),
        ("--oo-pipe-reason", "--oo-bg-surface", 4.5),
    ]
    if pid == "high-contrast":
        pairs += [
            ("--oo-bd-subtle", ground, 3.0)
            for ground in ("--oo-bg-base", "--oo-bg-surface", "--oo-bg-overlay",
                           "--oo-bg-subtle", "--oo-bg-tint-1", "--oo-bg-code")
        ]
    return pairs


def _contrast_failures(tokens, palettes):
    """``(pairs checked, [failure])`` over every palette. A pair whose tokens
    cannot be evaluated is a failure, never skipped."""
    from _colour import Resolver, contrast, over

    checked, failures = 0, []
    for pid, (roles, scheme) in palettes.items():
        resolver = Resolver(_with_palette(tokens, pid), roles, scheme)
        for ink, ground, need in _pairs(pid):
            checked += 1
            label = f"{pid}: {ink} on " + (
                ground if isinstance(ground, str) else f"{ground[1]} over {ground[2]}"
            )
            try:
                if isinstance(ground, str):
                    base = resolver.colour(ground)
                else:
                    base = over(resolver.colour(ground[1]), resolver.colour(ground[2]))
                ratio = contrast(over(resolver.colour(ink), base), base)
            except (KeyError, ValueError) as exc:
                failures.append(f"{label}: cannot be evaluated ({exc})")
                continue
            if ratio < need:
                failures.append(f"{label}: {ratio:.2f} < {need}")
    return checked, failures


def test_ds2_every_pair_of_the_design_passes_in_every_palette():
    tokens = _derivation()
    palettes = _palettes()
    assert sorted(palettes) == sorted(_PALETTE_IDS), sorted(palettes)

    planted = dict(palettes)
    night_roles, night_scheme = palettes["night"]
    planted["night"] = ({**night_roles, "text-muted": "#8C8881"}, night_scheme)
    _, caught = _contrast_failures(tokens, {"night": planted["night"]})
    assert caught and all("--oo-fg-muted" in failure for failure in caught), (
        f"a planted low pair (night muted text at #8C8881) is caught, and only it: {caught}"
    )

    checked, failures = _contrast_failures(tokens, palettes)
    assert checked >= 327, f"the design's pair list holds 327 pairs over the three palettes: {checked}"
    assert not failures, "pairs under their ratio:\n  " + "\n  ".join(failures)


# ===========================================================================
# DS3 -- every colour comes from a role
# ===========================================================================
_QR_EXCEPTION = ("--oo-qr-bg", "#FFFFFF")
_MIX = re.compile(
    r"color-mix\(\s*in\s+srgb\s*,\s*(OPERAND)(?:\s+\d+(?:\.\d+)?%)?\s*,\s*(OPERAND)(?:\s+\d+(?:\.\d+)?%)?\s*\)"
    .replace("OPERAND", r"var\(\s*--[\w-]+\s*\)|transparent")
)


def _derivation_findings(sources):
    """Every way the sources break the derivation rule, as sentences."""
    findings = []
    theme = sources.get(_THEME, "")
    blocks = _derivation_blocks(theme)
    if len(blocks) != 1:
        findings.append(f"{_THEME} holds {len(blocks)} [data-oo-theme] rules, not one")
        block = []
    else:
        block = _declarations(blocks[0][1])
    names = [name for name, _ in block]
    for name in sorted({n for n in names if names.count(n) > 1}):
        findings.append(f"{name} is declared {names.count(name)} times in the derivation layer")
    in_block = dict(block)

    # What a palette's rule gives: once for each palette, never also in the block.
    per_palette = _palette_rules_of(theme)
    per_palette_selectors = {selector for selector, _, _ in per_palette}
    given = {}
    for _, ids, body in per_palette:
        for name, value in _declarations(body):
            for pid in ids:
                given.setdefault(name, []).append((pid, value))
    for name, values in sorted(given.items()):
        if name in in_block:
            findings.append(f"{name} is declared in the block and in a palette's rule")
        pids = sorted(pid for pid, _ in values)
        if pids != sorted(_PALETTE_IDS):
            findings.append(f"{name} is given per palette for {pids}, not once for each of {list(_PALETTE_IDS)}")

    elsewhere = []
    for path, text in sources.items():
        if path in _PALETTE_FILES.values():
            continue
        for selector, body, context in _rules(_css_of(path, text)):
            if path == _THEME and (
                re.fullmatch(r"\[data-oo-theme\]", selector.strip()) or selector.strip() in per_palette_selectors
            ):
                continue
            for name, value in _declarations(body):
                if name.startswith("--oo-") and not name.startswith(_ROLE_PREFIX):
                    elsewhere.append((path, selector, name, value))

    colour_tokens = set(in_block)
    changed = True
    while changed:
        changed = False
        for _, _, name, value in elsewhere:
            if name in colour_tokens:
                continue
            refs = _VAR.findall(value)
            if _colour_literals(value) or "color-mix(" in value or "light-dark(" in value or any(
                ref.startswith(_ROLE_PREFIX) or ref in colour_tokens for ref in refs
            ):
                colour_tokens.add(name)
                changed = True
    for path, selector, name, value in elsewhere:
        if name in colour_tokens:
            findings.append(
                f"{path}: {name} carries a colour and is declared under {selector!r}, "
                f"outside the derivation layer ({value})"
            )

    known_other = {name for _, _, name, _ in elsewhere}
    for name, value in block + [(name, value) for name, values in given.items() for _, value in values]:
        if (name, value.upper()) == (_QR_EXCEPTION[0], _QR_EXCEPTION[1]):
            continue
        literals = _colour_literals(value)
        if literals:
            findings.append(f"{name} spells a colour literal {literals} instead of a role")
        if "light-dark(" in value:
            findings.append(
                f"{name} uses light-dark(), which browsers before 2024 do not compute: "
                f"a value that differs with the palette goes in the rule that names it"
            )
        for ref in _VAR.findall(value):
            if ref.startswith(_ROLE_PREFIX):
                if ref[len(_ROLE_PREFIX):] not in _ROLES:
                    findings.append(f"{name} reads {ref}, which is no role")
            elif ref not in in_block and ref not in given and ref not in known_other:
                findings.append(f"{name} reads {ref}, declared nowhere")
        for mix in re.finditer(r"color-mix\(", value):
            if not _MIX.match(value, mix.start()):
                findings.append(f"{name} mixes other than in srgb between roles, tokens or transparent: {value}")
    return findings


def _hex_findings(sources, map_colours):
    """``{path: [literal]}`` of the hex colours outside the palette files,
    the QR code's ground and the pre-render's theme-color map excepted."""
    found = {}
    for path, text in sources.items():
        if path in _PALETTE_FILES.values():
            continue
        literals = []
        for offset, literal in _hexes(text):
            if path == _THEME:
                line_start = text.rfind("\n", 0, offset) + 1
                line = text[line_start:text.find("\n", offset) if text.find("\n", offset) >= 0 else len(text)]
                if re.fullmatch(r"\s*--oo-qr-bg\s*:\s*#FFFFFF\s*;?\s*", line, re.I):
                    continue
            if path == _APP_HTML and literal in map_colours:
                continue
            literals.append(literal)
        if literals:
            found[path] = literals
    return found


def _stray_page_hexes(html):
    """The hex colours of a page outside the pre-render's theme-color map
    entries and the theme-color meta tags."""
    spans = [m.span() for m in _MAP_ENTRY.finditer(html)]
    spans += [m.span() for m in re.finditer(r"<meta\b[^>]*\btheme-color\b[^>]*>", html)]
    return [literal for offset, literal in _hexes(html) if not any(a <= offset < b for a, b in spans)]


@pytest.mark.parametrize("half", ("derivation", "hex"))
def test_ds3_every_colour_comes_from_a_role(half):
    if half == "derivation":
        sample = {
            _THEME: (
                "[data-oo-theme] { --oo-a: var(--oo-role-text); --oo-b: #123456; --oo-a: var(--oo-role-bg);\n"
                "  --oo-c: color-mix(in oklch, var(--oo-role-text) 4%, transparent); --oo-d: var(--oo-nowhere);\n"
                "  --oo-e: var(--oo-role-ground); --oo-qr-bg: #FFFFFF; --oo-f: 0 1px 2px var(--oo-a); }\n"
                ":root { --oo-motion-x: 1ms; }\n"
            ),
            f"{_SRC}/lib/x.css": ".panel { --oo-g: oklch(0.7 0.06 70); --oo-h: var(--oo-a); --oo-w: 24rem; }",
        }
        caught = _derivation_findings(sample)
        expected_bits = (
            "--oo-a is declared 2 times", "--oo-b spells a colour literal", "--oo-c mixes",
            "--oo-d reads --oo-nowhere", "--oo-e reads --oo-role-ground", "--oo-g carries a colour",
            "--oo-h carries a colour",
        )
        missing = [bit for bit in expected_bits if not any(bit in f for f in caught)]
        assert not missing and len(caught) == len(expected_bits), (
            f"the census names every planted break, and nothing else: missing {missing}, found {caught}"
        )
        # A value that differs with the palette is given by the rule naming
        # it, once for each palette; light-dark() is refused, because an
        # engine that cannot compute it drops every property reading it.
        split = {
            _THEME: (
                "[data-oo-theme] { --oo-a: var(--oo-role-text); --oo-b: var(--oo-role-bg);"
                " --oo-l: light-dark(var(--oo-role-text), var(--oo-role-bg)); --oo-m: var(--oo-h); }\n"
                "[data-oo-theme=\"day\"] { --oo-h: var(--oo-role-surface); --oo-b: var(--oo-role-text); }\n"
                "[data-oo-theme=\"night\"], [data-oo-theme='high-contrast'] { --oo-h: var(--oo-role-text); }\n"
                "[data-oo-theme=\"night\"] { --oo-k: var(--oo-role-text); }\n"
            ),
        }
        caught = _derivation_findings(split)
        wanted = ("--oo-l uses light-dark()", "--oo-b is declared in the block and in a palette's rule",
                  "--oo-b is given per palette for ['day']", "--oo-k is given per palette for ['night']")
        missing = [bit for bit in wanted if not any(bit in f for f in caught)]
        assert not missing and len(caught) == len(wanted) and not any("--oo-h" in f or "--oo-m" in f for f in caught), (
            "a per-palette value given once for each palette stands, a block token read from it resolves, "
            f"and light-dark(), a double or a palette left out is named: missing {missing}, found {caught}"
        )

        sources = _sources((".css", ".scss", ".svelte"))
        tokens = _derivation()
        assert len(tokens) >= 100, f"the derivation layer declares the app's tokens: {len(tokens)}"
        findings = _derivation_findings(sources)
        assert not findings, "the derivation rule is broken:\n  " + "\n  ".join(findings)
        return

    fixture = (
        "a { color: #fff; background: #A0785A; outline-color: #12345678; }\n"
        "<a href=\"#main\">x</a> &#123; {#each items as item} url(#add)\n"
    )
    assert [literal for _, literal in _hexes(fixture)] == ["#fff", "#A0785A", "#12345678"], (
        "the census reads hex colours, and not a fragment, a character reference or a block"
    )
    sources = _sources(_EVERY)
    sources[_TAILWIND] = read(_TAILWIND)
    map_colours = {c for colours in _theme_colour_map(_prerender_script()).values() for c in colours}
    found = _hex_findings(sources, map_colours)
    assert not found, (
        f"hex colours outside the palette files ({sum(len(v) for v in found.values())} "
        f"in {len(found)} files): " + json.dumps({k: len(v) for k, v in sorted(found.items())})
    )
    # In app.html a hex stands in the pre-render's theme-color map or in the
    # static theme-color meta, and nowhere else, even when it is a map colour.
    sample = (
        "<meta name=\"theme-color\" content=\"#1F1A17\" />\n"
        "<script>var colours = { day: '#F2EDE4', night: '#1F1A17' };</script>\n"
        "<body style=\"background: #1F1A17\">\n"
    )
    assert _stray_page_hexes(sample) == ["#1F1A17"], (
        f"the census passes the map and the meta, and reads a map colour anywhere else: {_stray_page_hexes(sample)}"
    )
    assert not _stray_page_hexes(read(_APP_HTML)), (
        f"app.html holds hex colours outside its theme-color map and meta: {_stray_page_hexes(read(_APP_HTML))}"
    )


# ===========================================================================
# DS4 -- only the theme path writes the theme state
# ===========================================================================
_STATE_ATTRIBUTE = re.compile(
    r"""(?:setAttribute|removeAttribute|toggleAttribute)\(\s*['"`]data-oo-(?:theme|componion)['"`]"""
    r"""|\.dataset\.oo(?:Theme|Componion)\s*=(?!=)|delete\s+[\w.$]*\.dataset\.oo(?:Theme|Componion)"""
)
_CLASS_CALL = re.compile(r"classList\s*\.\s*(?:add|remove|toggle|replace)\s*\(")
_STATE_CLASS = re.compile(r"^(?:dark|oo-density-[\w-]*|oo-reduce-motion|oo-motion-full)$")
_ROOT_SIZE = re.compile(
    r"""setProperty\(\s*['"`](?:font-size|--oo-type-scale)['"`]"""
    r"""|\.style\.fontSize\s*=(?!=)|documentElement\.style\.cssText\s*=(?!=)"""
    r"""|documentElement\.className\s*\+?=(?!=)"""
)
_STRING_ARGUMENT = re.compile(r"""(['"`])([^'"`]*)\1""")
_IDENTIFIER = re.compile(r"(?<![\w$.'\"`])([A-Za-z_$][\w$]*)(?![\w$])")


def _theme_writes(text, constants):
    """The writes of theme state in one source: the palette or componion
    attribute, a state class (dark, density, motion) given as a literal, a
    template beginning with one, or a constant bound to one, and the root's
    font size."""
    count = len(_STATE_ATTRIBUTE.findall(text)) + len(_ROOT_SIZE.findall(text))
    for match in _CLASS_CALL.finditer(text):
        arguments = _call_arguments(text, match.end() - 1)
        names = [value.split("${", 1)[0] if quote == "`" else value
                 for quote, value in _STRING_ARGUMENT.findall(arguments)]
        names += [constants[ident] for ident in _IDENTIFIER.findall(_STRING_ARGUMENT.sub("", arguments))
                  if ident in constants]
        if any(_STATE_CLASS.match(name) or name.startswith("oo-density-") for name in names):
            count += 1
    return count


def test_ds4_only_the_theme_path_writes_the_theme_state():
    sample = (
        "html.setAttribute('data-oo-theme', p); root.dataset.ooComponion = 'off';\n"
        "html.classList.toggle('dark', d); el.classList.add(`oo-density-${d}`);\n"
        "html.classList.toggle(REDUCED, m); root.style.setProperty('--oo-type-scale', '1');\n"
        "document.documentElement.style.fontSize = '109%'; html.classList.add('theme-transitioning');\n"
        "el.classList.remove(\"oo-motion-full\"); root.removeAttribute(\"data-oo-theme\");\n"
    )
    assert _theme_writes(sample, {"REDUCED": "oo-reduce-motion"}) == 9, (
        "the census reads each spelling of a theme-state write, and not a transition class"
    )
    assert _theme_writes(
        "document.documentElement.className = 'dark'; document.documentElement.className += ' dark';", {}
    ) == 2, "the census reads the root's class list written whole or appended to"
    sources = _sources(_SCRIPTS + (".html",), exclude=(_APPLY, _APP_HTML))
    constants = _string_constants(_sources(_SCRIPTS))
    assert constants.get("MOTION_REDUCED_CLASS") == "oo-reduce-motion", (
        "the census resolves the constants that name a state class"
    )
    found = _count_in(sources, lambda path, text: _theme_writes(text, constants))
    assert not found, f"theme state written outside the theme path: {found}"


# ===========================================================================
# DS5 -- the pre-render and apply.ts agree over the matrix
# ===========================================================================
def _role_grounds():
    return {pid: roles["bg"] for pid, (roles, _) in _palettes().items()}


def test_ds5_the_prerender_and_the_theme_path_agree_over_the_matrix():
    attributes = _html_attributes()
    assert attributes.get("data-oo-theme") == "night", (
        f"<html> carries the night palette statically, so the tokens exist when the pre-render cannot run: {attributes}"
    )
    script = _prerender_script()
    assert re.search(r"\btry\s*\{", script) and re.search(r"\bcatch\b", script), (
        "the pre-render runs inside try/catch"
    )

    rows = _matrix()
    out = _node("matrix", {"rows": rows})
    assert len(out) == len(rows) and len(rows) >= 1500, f"every row of the matrix ran: {len(out)} of {len(rows)}"
    grounds = _role_grounds()
    disagree, threw, wrote, colours = [], [], [], []
    for row, result in zip(rows, out):
        pre, post = result["pre"], result["post"]
        label = json.dumps({k: v for k, v in row["stored"].items() if v is not None}) + \
            f" media={row['media_name']} blocked={row['blocked']}"
        if pre["threw"] or post["threw"]:
            threw.append(f"{label}: pre-render {pre['threw']!r}, apply {post['threw']!r}")
            continue
        if pre["writes"] or post["writes"]:
            wrote.append(f"{label}: {pre['writes'] or post['writes']}")
        if pre["state"] != post["state"]:
            disagree.append(f"{label}:\n      pre-render {pre['state']}\n      apply.ts   {post['state']}")
        shown = pre["state"]["attributes"].get("data-oo-theme")
        if (pre["themeColour"] or "").upper() != grounds.get(shown, "?").upper():
            colours.append(f"{label}: theme-color {pre['themeColour']!r} for {shown!r}")
    assert not threw, f"{len(threw)} rows threw:\n  " + "\n  ".join(threw[:8])
    assert not wrote, f"resolving and applying never write storage: {wrote[:8]}"
    assert not disagree, f"{len(disagree)} rows disagree:\n  " + "\n  ".join(disagree[:6])
    assert not colours, f"the theme-color is the shown palette's ground: {colours[:6]}"

    anchors = {
        ("none", False): ("night", True), ("dark", False): ("night", True),
        ("light", False): ("day", False), ("light-more", False): ("high-contrast", True),
        ("dark-more", True): ("high-contrast", True), ("light", True): ("day", False),
    }
    for row, result in zip(rows, out):
        stored = row["stored"]
        if stored["oo-palette"] is None and stored["oo-theme"] is None and stored["oo-type-scale"] is None:
            key = (row["media_name"], row["blocked"])
            if key in anchors:
                theme, dark = anchors[key]
                state = result["pre"]["state"]
                assert state["attributes"].get("data-oo-theme") == theme and ("dark" in state["classes"]) == dark, (
                    f"with nothing stored and the system {key}, the {theme} palette shows: {state}"
                )
                assert state["fontSize"] == "100%", f"the default text size is the root at 100%: {state}"
                density = [c for c in state["classes"] if c.startswith("oo-density-")]
                assert len(density) == 1, f"one density class: {state['classes']}"
                if row["blocked"] or stored["oo-density"] in (None, "roomy"):
                    assert density == ["oo-density-comfortable"], f"comfortable by default: {state}"
                if row["blocked"] or stored["oo-componion"] != "off":
                    assert state["attributes"].get("data-oo-componion") == "on", (
                        f"the componion switch is on unless it was turned off: {state}"
                    )

    broken = _node("broken")
    assert broken["threw"] is None, f"a pre-render that fails throws nothing out of the page: {broken['threw']}"
    assert broken["state"]["attributes"].get("data-oo-theme") == "night", (
        f"when the pre-render cannot run, the static night palette stands: {broken['state']}"
    )


# ===========================================================================
# DS6 -- the system choice follows the system; only a choice is stored
# ===========================================================================
_MEDIA_QUERY = re.compile(r"""matchMedia\(\s*['"`]\s*\(\s*prefers-(?:color-scheme|contrast)""")
_STORAGE_INDEX = re.compile(
    r"""\b(?:local|session)Storage\s*\[[^\]]*\]\s*=(?!=)"""
    r"""|\b(?:local|session)Storage\s*\.\s*(?!(?:setItem|getItem|removeItem|clear|key)\b)[A-Za-z_$][\w$]*\s*=(?!=)"""
)


def _stored_keys(text, constants):
    """``[(offset, key)]`` of every ``setItem`` call whose key is a literal
    or a constant bound to a string."""
    found = []
    for match in re.finditer(r"setItem\s*\(", text):
        arguments = _call_arguments(text, match.end() - 1)
        first = arguments.split(",", 1)[0].strip()
        literal = re.fullmatch(r"""(['"`])([^'"`]*)\1""", first)
        if literal:
            found.append((match.start(), literal.group(2)))
        elif first in constants:
            found.append((match.start(), constants[first]))
        else:
            found.append((match.start(), None))
    return found


@pytest.mark.parametrize("half", ("node", "wiring"))
def test_ds6_the_system_choice_follows_the_system_and_only_a_choice_is_stored(half):
    if half == "wiring":
        constants = _string_constants(_sources(_SCRIPTS))
        sample = (
            "const KEY = 'oo-palette';\nfunction setPalette(p) { localStorage.setItem(KEY, p); }\n"
            "function other() { localStorage.setItem('oo-theme', 'dark'); storage.setItem(k, v); }\n"
        )
        keys = [key for _, key in _stored_keys(sample, {"KEY": "oo-palette"})]
        assert keys == ["oo-palette", "oo-theme", None], f"the census reads each stored key: {keys}"

        sources = _sources(_SCRIPTS + (".html",))
        theme_written, palette_outside, unknown = [], [], []
        palette_writes = 0
        for path, text in sources.items():
            body = _function_body(text, "setPalette") if path == _PREFERENCES else None
            body_start = text.find(body) if body else -1
            for offset, key in _stored_keys(text, constants):
                if key == "oo-theme":
                    theme_written.append(path)
                elif key == "oo-palette":
                    if body_start >= 0 and body_start <= offset < body_start + len(body):
                        palette_writes += 1
                    else:
                        palette_outside.append(path)
                elif key is None:
                    unknown.append(path)
        assert not theme_written, f"nothing stores oo-theme: {sorted(set(theme_written))}"
        assert not palette_outside, (
            f"oo-palette is stored only by the explicit choice (setPalette in the preferences store): "
            f"{sorted(set(palette_outside))}"
        )
        assert palette_writes == 1, f"the explicit choice stores oo-palette once: {palette_writes}"
        assert not unknown, f"every stored key is readable by the census: {sorted(set(unknown))}"

        indexed = "localStorage['oo-theme'] = 'dark'; window.localStorage.ooPalette = 'day'; localStorage.getItem('x');"
        assert len(_STORAGE_INDEX.findall(indexed)) == 2, "the census reads storage written by index or by property"
        by_index = _count_in(sources, lambda path, text: len(_STORAGE_INDEX.findall(text)))
        assert not by_index, f"storage is written through setItem, which the census reads, and never by index: {by_index}"
        assert len(_MEDIA_QUERY.findall(
            "window.matchMedia('(prefers-color-scheme: dark)'); matchMedia(\"( prefers-contrast: more)\")"
        )) == 2, "the census reads each query of the system's scheme and contrast"
        readers = _count_in(
            _sources(_SCRIPTS + (".html",), exclude=(_APPLY, _APP_HTML)),
            lambda path, text: len(_MEDIA_QUERY.findall(text)),
        )
        assert not readers, (
            f"the system's scheme and contrast are read by the theme path alone: {readers}"
        )
        callers = [
            path for path, text in sources.items()
            if path != _APPLY and re.search(r"(?<![\w$.])followSystem\s*\(", text)
            and re.search(r"""import\s*\{[^}]*\bfollowSystem\b[^}]*\}\s*from\s*['"][^'"]*theme/apply(?:\.ts)?['"]""", text)
        ]
        assert callers, "the application follows the system through apply.ts's followSystem"
        return

    scenarios = [
        # Nothing stored: the system's scheme and contrast decide, live.
        {"stored": {}, "media": _MEDIA["dark"], "steps": [
            {"system": {"scheme": "light"}}, {"system": {"more": True}},
            {"system": {"more": False}}, {"system": {"scheme": "dark"}},
        ]},
        # An explicit choice is not moved by the system.
        {"stored": {"oo-palette": "night"}, "media": _MEDIA["dark"], "steps": [
            {"system": {"scheme": "light"}}, {"system": {"more": True}},
        ]},
        # A choice made while following is read at the next change.
        {"stored": {}, "media": _MEDIA["light"], "steps": [
            {"system": {"scheme": "dark"}}, {"store": {"oo-palette": "day"}, "system": {"scheme": "light"}},
            {"system": {"scheme": "dark"}},
        ]},
        # After stopping, a change applies nothing.
        {"stored": {}, "media": _MEDIA["dark"], "steps": [
            {"system": {"scheme": "light"}}, {"stop": True}, {"system": {"scheme": "dark"}},
        ]},
        # Without media queries there is nothing to follow, and nothing throws.
        {"stored": {}, "media": None, "steps": []},
    ]
    out = _node("follow", {"scenarios": scenarios})
    assert [r["threw"] for r in out] == [None] * 5, [r["threw"] for r in out]
    assert all(r["stopIsFunction"] for r in out), "followSystem returns a function that stops following"
    assert out[0]["seen"] == ["night", "day", "high-contrast", "day", "night"], (
        f"under Match system each change of the scheme or the contrast re-applies: {out[0]['seen']}"
    )
    assert out[1]["seen"] == ["night", "night", "night"], (
        f"an explicit choice stays whatever the system does: {out[1]['seen']}"
    )
    assert out[2]["seen"] == ["day", "night", "day", "day"], (
        f"a choice stored while following is read at the next change: {out[2]['seen']}"
    )
    assert out[3]["seen"] == ["night", "day", "day"] and out[3]["listening"] == 0, (
        f"once stopped, a change applies nothing and no listener is left: {out[3]}"
    )
    assert all(not r["writes"] for r in out), f"following stores nothing: {[r['writes'] for r in out]}"

    pairs = [
        ({"stored": {"oo-palette": "day", "oo-density": "compact", "oo-motion": "reduced",
                     "oo-type-scale": "large", "oo-componion": "off"}, "media": _MEDIA["light"]},
         {"stored": {}, "media": _MEDIA["dark"]}),
        ({"stored": {}, "media": _MEDIA["dark-more"]}, {"stored": {"oo-motion": "full"}, "media": _MEDIA["light"]}),
    ]
    again = _node("reapply", {"pairs": pairs})
    for first, result in zip(pairs, again):
        assert result["again"] == result["fresh"], (
            f"applying again leaves exactly the state a fresh root reaches, nothing of the first: {result}"
        )
        assert not result["writes"], result["writes"]


# ===========================================================================
# DS7 -- each palette declares its scheme; the theme-color map is its ground
# ===========================================================================
def test_ds7_each_palette_declares_its_scheme_and_the_theme_color_map_is_its_ground():
    for pid, path in _PALETTE_FILES.items():
        assert Path(REPO / path).is_file(), f"{path} is absent"
        _, declared = _palette_declarations(path)
        schemes = [value for name, value in declared if name == "color-scheme"]
        assert schemes == [_SCHEMES[pid]], (
            f"the {pid} palette declares color-scheme: {_SCHEMES[pid]}: {schemes}"
        )

    colour_map = _theme_colour_map(_prerender_script())
    assert sorted(colour_map) == sorted(_PALETTE_IDS) and all(len(v) == 1 for v in colour_map.values()), (
        f"the pre-render holds one theme-color entry per palette: {colour_map}"
    )
    grounds = _role_grounds()
    wrong = {pid: (colour_map[pid][0], grounds[pid]) for pid in _PALETTE_IDS
             if colour_map[pid][0].upper() != grounds[pid].upper()}
    assert not wrong, f"each theme-color entry is its palette's ground: {wrong}"
    metas = _theme_colour_metas()
    static = _html_attributes().get("data-oo-theme")
    assert len(metas) == 1 and static in grounds and metas[0].get("content", "").upper() == grounds[static].upper(), (
        f"the static theme-color is the ground of the static palette, {static}: {metas}"
    )


# ===========================================================================
# DS9 -- the text scale is carried by the root size
# ===========================================================================
_SIZE_TOKEN = re.compile(r"^--oo-text-(?:2xs|xs|sm|base|md|lg|xl|[2-9]xl)$")
_SPACE_TOKEN = re.compile(r"^--oo-space-\d+$")
_REM = re.compile(r"^(\d*\.?\d+)rem$")


def _scale_tokens_declared(path, text):
    """The text and space tokens a source's rules declare."""
    return [name for _, body, _ in _rules(_css_of(path, text)) for name, _ in _declarations(body)
            if _SIZE_TOKEN.match(name) or _SPACE_TOKEN.match(name)]


@pytest.mark.parametrize("half", ("static", "node"))
def test_ds9_the_text_scale_is_carried_by_the_root_size(half):
    if half == "node":
        rows = [{"stored": {"oo-type-scale": size}, "media": _MEDIA["dark"], "blocked": False} for size in _SIZES]
        rows.append({"stored": {"oo-type-scale": "huge"}, "media": _MEDIA["dark"], "blocked": False})
        out = _node("resolve", {"rows": rows})
        sizes = [result["state"]["fontSize"] for result in out]
        assert sizes == [*_SIZES.values(), "100%"], (
            f"apply.ts sets the root size of each text level, 100% when none or an unknown one is stored: {sizes}"
        )
        return

    users = _count_in(_sources(_EVERY), lambda path, text: text.count("--oo-type-scale"))
    assert not users, f"no --oo-type-scale multiplier remains: {users}"

    # The scale is declared by the style sheets alone: a component that
    # declares a text or space token rescales its subtree off the root size.
    assert _scale_tokens_declared("frontend/src/x.svelte", "<style>.c { --oo-text-sm: 10px; --oo-space-2: 3px; }</style>") == [
        "--oo-text-sm", "--oo-space-2"], "the census reads the text and space tokens a component declares"
    in_components = {path: found for path in files((".svelte",)) for found in [_scale_tokens_declared(path, read(path))] if found}
    assert not in_components, f"components declaring text or space tokens: {in_components}"

    declared = []
    for path in files((".css", ".scss")):
        for _, body, _ in _rules(read(path)):
            for name, value in _declarations(body):
                if _SIZE_TOKEN.match(name) or _SPACE_TOKEN.match(name):
                    declared.append((path, name, value))
    assert any(_SIZE_TOKEN.match(n) for _, n, _ in declared) and any(_SPACE_TOKEN.match(n) for _, n, _ in declared), (
        "the census finds the text and space tokens"
    )
    not_rem = [f"{p}: {n}: {v}" for p, n, v in declared
               if not (_REM.match(v) or (_SPACE_TOKEN.match(n) and v == "0"))]
    assert not not_rem, "text and space tokens are in rem:\n  " + "\n  ".join(not_rem)
    small = [f"{p}: {n}: {v}" for p, n, v in declared
             if _SIZE_TOKEN.match(n) and float(_REM.match(v).group(1)) < 0.75]
    assert not small, "no text token is under 0.75rem:\n  " + "\n  ".join(small)


# ===========================================================================
# DS10 -- no ink is used as a fill
# ===========================================================================
_INK_NAMES = ("acc-300", "acc-400", "acc-500", "acc-700", "accent", "accent-primary",
              "accent-secondary", "tobacco", "sage")
_INK_READ = re.compile(
    r"var\(\s*--oo-(?:" + "|".join(re.escape(n) for n in sorted(_INK_NAMES, key=len, reverse=True))
    + r")(?![\w-])"
)
_UTILITY = re.compile(
    r"(?<![\w-])(?:[\w-]+:)*(bg|fill|from|via|to|text|border(?:-[trblxyse])?|ring|ring-offset|outline"
    r"|stroke|decoration|caret|accent|divide|placeholder|shadow)-\[[^\]\s]*\]"
)
_FILL_UTILITIES = {"bg", "fill", "from", "via", "to"}
_FILL_PROPERTIES = {"background", "background-color", "background-image", "fill", "stop-color", "flood-color"}
_INK_PROPERTY = re.compile(
    r"^(?:color|border(?:-[\w-]+)?|outline(?:-[\w-]+)?|stroke|caret-color|accent-color"
    r"|text-decoration(?:-color)?|column-rule(?:-color)?|box-shadow|text-shadow|-webkit-text-fill-color)$"
)
_DECLARATION_START = re.compile(r"^\s*(?:style:)?(--[\w-]+|-?[a-zA-Z][\w-]*)\s*:")
_ATTRIBUTE_BEFORE = re.compile(r"([\w:-]+)\s*[:=]\s*$")


def _context(text, offset):
    """What the value at ``offset`` is written into: ``("utility", name)``,
    ``("property", name)``, ``("attribute", name)`` or ``(None, None)``."""
    for match in _UTILITY.finditer(text, max(0, offset - 200), offset + 200):
        if match.start() <= offset < match.end():
            return "utility", match.group(1)
    start = max(text.rfind(c, 0, offset) for c in ";{}" + _QUOTES)
    segment = text[start + 1:offset]
    declaration = _DECLARATION_START.match(segment)
    if declaration:
        return "property", declaration.group(1).lower()
    if start >= 0 and text[start] in _QUOTES and not segment.strip():
        before = _ATTRIBUTE_BEFORE.search(text[max(0, start - 60):start])
        if before:
            return "attribute", before.group(1).lower().split(":")[-1]
    return None, None


def _ink_uses(text):
    """``(fills, unattributable)`` of the ink names' reads in one source."""
    fills = unattributable = 0
    for match in _INK_READ.finditer(text):
        kind, name = _context(text, match.start())
        if kind == "utility":
            fills += name in _FILL_UTILITIES
        elif kind in ("property", "attribute"):
            if name in _FILL_PROPERTIES:
                fills += 1
            elif name.startswith("--oo-") and name[len("--oo-"):] in _INK_NAMES:
                pass
            elif not _INK_PROPERTY.match(name):
                unattributable += 1
        else:
            unattributable += 1
    return fills, unattributable


def test_ds10_no_ink_is_used_as_a_fill():
    sample = (
        "<div class=\"bg-[var(--oo-acc-500)] text-[var(--oo-acc-400)] hover:bg-[var(--oo-sage)]\"></div>\n"
        "<style>a { background-color: var(--oo-accent); color: var(--oo-tobacco);"
        " border: 1px solid var(--oo-acc-300); }</style>\n"
        "<svg><path fill=\"var(--oo-acc-700)\" stroke=\"var(--oo-acc-700)\"/></svg>\n"
        "const tone = 'var(--oo-sage)'; :root { --oo-x-bg: var(--oo-acc-500); --oo-accent: var(--oo-acc-500); }\n"
    )
    assert _ink_uses(sample) == (4, 2), (
        f"fills in a class, a declaration and an attribute are counted, inks are not, and a read "
        f"the census cannot place is unattributable: {_ink_uses(sample)}"
    )
    sources = _sources(_EVERY, exclude=tuple(_PALETTE_FILES.values()))
    fills, unplaced = {}, {}
    for path, text in sources.items():
        filled, unknown = _ink_uses(text)
        if filled:
            fills[path] = filled
        if unknown:
            unplaced[path] = unknown
    assert not fills and not unplaced, (
        f"inks used as fills: {sum(fills.values())} in {len(fills)} files; reads the census cannot "
        f"attribute: {sum(unplaced.values())} in {len(unplaced)} files\n"
        f"  fills {json.dumps(dict(sorted(fills.items())))}\n  unattributable {json.dumps(dict(sorted(unplaced.items())))}"
    )

    # Every name the derivation layer gives the accent's inks, not only the
    # nine older ones, each read placed through the expressions it travels in.
    tokens = _derivation()
    inks = _accent_inks(tokens)
    assert {f"--oo-{name}" for name in _INK_NAMES} | {"--oo-acc-ink", "--oo-acc-ink-2", "--oo-focus-ink"} <= set(inks), (
        f"the census reads every name of the accent's inks: {inks}"
    )
    assert not set(inks) & set(_SANCTIONED_INK_FILLS), f"the mark fills are not counted as inks: {inks}"
    sample = {
        f"{_SRC}/lib/sample/tone.ts": "export function tone(v: number): string { return v > 1 ? 'var(--oo-acc-ink)' : 'var(--oo-fg-muted)'; }\n",
        f"{_SRC}/lib/sample/Inks.svelte": (
            "<script>\n  import { tone } from './tone';\n  export let colour = 'var(--oo-acc-ink-2)';\n"
            "  const WORDS = { a: 'var(--oo-focus-ink)' };\n  const loose = 'var(--oo-acc-500)';\n</script>\n"
            "<span style=\"color: {on ? 'var(--oo-acc-ink)' : 'var(--oo-fg-muted)'};\">a</span>\n"
            "<div style=\"width: 3px; background-color: {tone(n)};\"></div>\n"
            "<svg style=\"--ring: {colour};\"><circle stroke=\"var(--ring)\" /></svg>\n"
            "<b class=\"{on ? 'border-[var(--oo-accent)]' : ''} {WORDS.a}\">b</b>\n"
        ),
    }
    paints, _ = _paints(sample, _tailwind_colours())
    fills, unplaced = _ink_attribution(sample, paints, inks)
    assert [f.split(" fills")[0] for f in fills] == [f"{_SRC}/lib/sample/tone.ts:1 --oo-acc-ink"] and [
        u.split(" lands")[0] for u in unplaced] == [f"{_SRC}/lib/sample/Inks.svelte:4 --oo-focus-ink",
                                                     f"{_SRC}/lib/sample/Inks.svelte:5 --oo-acc-500"], (
        "a read is placed through a conditional, an imported function, a property's default and the "
        f"custom property it fills, a fill is found, and a read that lands nowhere is unattributable: {fills}, {unplaced}"
    )
    paints, _ = _tree_paints()
    fills, unplaced = _ink_attribution(_sources(_EVERY, exclude=tuple(_PALETTE_FILES.values())), paints, inks)
    assert not fills and not unplaced, (
        f"the accent's inks used as fills: {len(fills)}; reads the census cannot place: {len(unplaced)}\n  "
        + "\n  ".join(fills + unplaced)
    )


# ===========================================================================
# DS11 -- no named white or black colour
# ===========================================================================
_WHITE_BLACK = re.compile(r"(?<![\w-])(?:white|black)(?![\w-])")
_WB_UTILITY = re.compile(
    r"(?<![\w-])(?:[\w-]+:)*(?:text|bg|border(?:-[trblxyse])?|ring|fill|stroke|outline|divide|from|via|to"
    r"|placeholder|decoration|caret|accent|shadow)-(?:white|black)(?![\w-])"
)
_COLOUR_PROPERTY = re.compile(
    r"^(?:color|background(?:-color|-image)?|border(?:-[\w-]+)?|outline(?:-[\w-]+)?|fill|stroke"
    r"|caret-color|accent-color|box-shadow|text-shadow|text-decoration(?:-color)?|column-rule(?:-color)?"
    r"|stop-color|flood-color|lighting-color|-webkit-text-fill-color|-webkit-text-stroke(?:-color)?)$"
)


def _named_colours(text):
    """Uses of ``white`` or ``black`` as a colour: a Tailwind utility, or the
    value of a colour property or attribute."""
    count = len(_WB_UTILITY.findall(text))
    utilities = [(m.start(), m.end()) for m in _WB_UTILITY.finditer(text)]
    for match in _WHITE_BLACK.finditer(text):
        if any(start <= match.start() < end for start, end in utilities):
            continue
        kind, name = _context(text, match.start())
        if kind in ("property", "attribute") and _COLOUR_PROPERTY.match(name):
            count += 1
    return count


def test_ds11_no_named_white_or_black_colour():
    sample = (
        "<div class=\"bg-white text-black/70 hover:border-white\" style=\"color: white\"></div>\n"
        "<style>a { background: black; white-space: nowrap; }</style>\n"
        "<path fill=\"white\" /> const tone = { color: 'black' }; // a black box\n"
    )
    assert _named_colours(sample) == 7, f"each spelling of a named white or black is counted: {_named_colours(sample)}"
    # What the interface draws: style sheets, components and the page. A
    # script module's strings are data (a drawing's pen colour), not paint.
    sources = _sources((".svelte", ".css", ".scss", ".html"), exclude=tuple(_PALETTE_FILES.values()))
    found = _count_in(sources, lambda path, text: _named_colours(text))
    assert not found, (
        f"named white or black colours: {sum(found.values())} in {len(found)} files: "
        + json.dumps(dict(sorted(found.items())))
    )


# ===========================================================================
# DS12 -- the focus ring is the focus ink and keeps the shape
# ===========================================================================
def _rings(path, text):
    """``(reshaped, off the ink)``: the focus rings of a source that set a
    radius, and those whose outline or shadow draws a colour other than the
    focus ink (forced colours aside)."""
    reshaped, off = [], []
    for selector, body, context in _rules(_css_of(path, text)):
        if ":focus-visible" not in selector:
            continue
        declared = _declarations(body)
        if any(name == "border-radius" for name, _ in declared):
            reshaped.append(f"{path}: {selector.strip()}")
        if any(re.search(r"forced-colors", c) for c in context):
            continue
        for name, value in declared:
            if name in ("outline", "outline-color", "box-shadow") and (_VAR.search(value) or _colour_literals(value)):
                if "var(--oo-focus-ink)" not in value:
                    off.append(f"{path}: {selector.strip()}: {name}")
    return reshaped, off


def test_ds12_the_focus_ring_is_the_focus_ink_and_keeps_the_shape():
    global_rules = [
        (selector, body) for selector, body, context in _rules(read(_APP_CSS))
        if not context and re.fullmatch(r"\*?:focus-visible", selector.strip())
    ]
    assert len(global_rules) == 1, f"app.css draws one global focus ring: {len(global_rules)}"
    declared = dict(_declarations(global_rules[0][1]))
    outline = declared.get("outline", "") + " " + declared.get("outline-color", "")
    assert "var(--oo-focus-ink)" in outline, f"the focus ring is drawn in the focus ink: {declared}"

    assert _rings("x.css", ".a:focus-visible { border-radius: 4px; outline: 2px solid var(--oo-focus-ink); }")[0] == [
        "x.css: .a:focus-visible"], "the census reads a focus ring that sets a radius"
    reshaped = []
    for path in files((".css", ".scss", ".svelte")):
        for selector, body, _ in _rules(_css_of(path, read(path))):
            if ":focus-visible" in selector and any(name == "border-radius" for name, _ in _declarations(body)):
                reshaped.append(f"{path}: {selector.strip()}")
    assert not reshaped, f"a focus ring never sets a radius, so no control changes shape: {reshaped}"

    # Every ring, the components' own included, is drawn in the focus ink;
    # forced colours draw it in the system's Highlight.
    sample = (
        ".a:focus-visible { outline: 2px solid var(--oo-acc-500); } .b:focus-visible { outline: 2px solid var(--oo-focus-ink); }\n"
        ".c:focus-visible { outline: none; box-shadow: 0 0 0 2px var(--oo-accent); }\n"
        "@media (forced-colors: active) { :focus-visible { outline-color: Highlight; } }\n"
    )
    assert _rings("x.css", sample)[1] == ["x.css: .a:focus-visible: outline", "x.css: .c:focus-visible: box-shadow"], (
        f"the census reads a ring's colour in an outline or a shadow, and passes the focus ink and forced colours: {_rings('x.css', sample)[1]}"
    )
    off_ink = [found for path in files((".css", ".scss", ".svelte")) for found in _rings(path, read(path))[1]]
    assert not off_ink, f"focus rings drawn off the focus ink: {off_ink}"


# ===========================================================================
# DS13 -- Tailwind holds no hex; no light override layer remains
# ===========================================================================
def _light_overrides(css):
    """The rules of ``css`` that override the palette under ``html:not(.dark)``."""
    return [selector.strip() for selector, _, _ in _rules(css) if "html:not(.dark)" in selector]


def test_ds13_tailwind_holds_no_hex_and_no_light_override_layer_remains():
    config = read(_TAILWIND)
    hexes = [literal for _, literal in _hexes(config)]
    assert not hexes, f"{_TAILWIND} holds no hex colour: {len(hexes)} ({hexes[:6]})"
    assert _light_overrides("html:not(.dark) .x { color: red; } .y { color: red; }") == ["html:not(.dark) .x"], (
        "the census reads a light override rule"
    )
    overrides = _light_overrides(read(_APP_CSS))
    assert not overrides, f"app.css holds no html:not(.dark) override: {len(overrides)} rules"


# ===========================================================================
# DS14 -- four choices, and every legacy key migrates
# ===========================================================================
@pytest.mark.parametrize("half", ("node", "wiring"))
def test_ds14_the_choices_are_four_and_every_legacy_key_migrates(half):
    if half == "wiring":
        preferences = read(_PREFERENCES)
        assert re.search(
            r"""import\s*\{[^}]*\bCHOICES\b[^}]*\}\s*from\s*['"][^'"]*theme/apply(?:\.ts)?['"]""", preferences
        ), "the preferences store offers the choices apply.ts declares"
        retired = re.compile(r"""(['"`])(?:anthracite|parchment|slate|linen)\1""")
        assert len(retired.findall("const p = 'anthracite'; const q = \"linen\"; const r = slate;")) == 2, (
            "the census reads a retired palette's name as a string"
        )
        found = _count_in(
            _sources(_SCRIPTS + (".html",), exclude=(_APPLY, _APP_HTML)),
            lambda path, text: len(retired.findall(text)),
        )
        assert not found, f"no source outside the theme path names a retired palette: {found}"
        return

    names = _node("choices")
    assert names == {"choices": _CHOICES, "labels": _CHOICE_LABELS}, (
        f"the choices are Match system, day, night and high contrast, in that order: {names}"
    )
    rows, expected = [], []
    for palette in _STORED_PALETTES:
        for theme in _STORED_THEMES:
            for media_name, media in _MEDIA.items():
                for blocked in (False, True):
                    rows.append({"stored": {"oo-palette": palette, "oo-theme": theme}, "media": media,
                                 "blocked": blocked})
                    expected.append(_expected(palette, theme, media, blocked))
    out = _node("resolve", {"rows": rows})
    wrong = []
    for row, want, result in zip(rows, expected, out):
        got = result["resolved"]
        shown = result["state"]["attributes"].get("data-oo-theme")
        if result["threw"] or got is None or (got["choice"], got["theme"]) != want or shown != want[1]:
            wrong.append(f"{row['stored']} media={row['media']} blocked={row['blocked']}: "
                         f"want {want}, got {got} showing {shown!r} ({result['threw']})")
    assert len(rows) >= 300 and not wrong, (
        f"{len(wrong)} of {len(rows)} stored states migrate otherwise than specified:\n  " + "\n  ".join(wrong[:10])
    )


# ===========================================================================
# DS15 -- tone-only containers carry the edge; forced colours are declared
# ===========================================================================
def _edge_border(path, class_name):
    """Whether the rule for ``class_name`` in ``path`` draws a border in the edge token."""
    for selector, body, _ in _rules(_css_of(path, read(path))):
        if re.search(r"(?<![\w-])\." + re.escape(class_name) + r"(?![\w-])", selector) and not re.search(
            r":(?:hover|focus|active)|\[", selector
        ):
            for name, value in _declarations(body):
                if name in ("border", "border-color") and "var(--oo-edge)" in value:
                    return True
    return False


def test_ds15_tone_only_containers_carry_the_edge_and_forced_colours_are_declared():
    missing = [
        f"{path} .{name}" for path, name in (
            (_CARD, "oo-card"), (_MODAL, "oo-modal-panel"), (_TOOLTIP, "oo-tip"),
        ) if not _edge_border(path, name)
    ]
    assert not missing, f"tone-only containers draw their border in var(--oo-edge): {missing}"
    forced = [
        body for _, body, context in _rules(read(_APP_CSS))
        if any(re.fullmatch(r"@media\s*\(\s*forced-colors\s*:\s*active\s*\)", c) for c in context)
    ]
    assert forced and any(_declarations(body) for body in forced), (
        "app.css declares what forced colours keep, in an @media (forced-colors: active) block"
    )
    kept = {
        part.strip()
        for selector, body, context in _rules(read(_APP_CSS))
        if any(re.fullmatch(r"@media\s*\(\s*forced-colors\s*:\s*active\s*\)", c) for c in context)
        and dict(_declarations(body)).get("border-color") == "CanvasText"
        for part in selector.split(",")
    }
    assert {".oo-card", ".oo-modal-panel", ".oo-tip"} <= kept, (
        f"forced colours keep the edge of the card, the modal's panel and the tooltip, in CanvasText: {sorted(kept)}"
    )


# ===========================================================================
# DS16 -- the motion neutralisation survives
# ===========================================================================
def _stills_motion(body):
    declared = {name: value for name, value in _declarations(body)}
    return all(
        re.match(r"0\.01ms\s*!important", declared.get(name, ""))
        for name in ("animation-duration", "transition-duration")
    )


def test_ds16_the_motion_neutralisation_survives():
    rules = _rules(read(_APP_CSS))
    forced = [body for selector, body, context in rules
              if not context and "html.oo-reduce-motion *" in selector]
    assert forced and _stills_motion(forced[0]), (
        "the choice to reduce motion stills every animation and transition, whatever the system says"
    )
    system = [body for selector, body, context in rules
              if any(re.fullmatch(r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)", c) for c in context)
              and "html:not(.oo-motion-full) *" in selector]
    assert system and _stills_motion(system[0]), (
        "the system's reduced motion stills them too, unless full motion was chosen"
    )
    tokens = [body for _, body, context in _rules(read(_TOKENS))
              if any(re.fullmatch(r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)", c) for c in context)]
    instant = dict(_declarations(tokens[0])) if tokens else {}
    assert all(instant.get(f"--oo-motion-{speed}") == "var(--oo-motion-instant)"
               for speed in ("fast", "normal", "slow", "slower")), (
        f"under reduced motion every motion token is instant: {instant}"
    )


# ===========================================================================
# DS17 -- the colour evaluator gives known answers
# ===========================================================================
def test_ds17_the_colour_evaluator_gives_known_answers():
    from _colour import Resolver, contrast, over, parse

    def close(colour, expected, tolerance=0.02):
        return all(abs(a - b) <= tolerance for a, b in zip(colour, expected)) and len(colour) == len(expected)

    resolver = Resolver({}, {}, "light")
    cases = {
        # CSS Color 5: mixing with transparent keeps the hue and halves the alpha.
        "color-mix(in srgb, red 50%, transparent)": (255, 0, 0, 0.5),
        "color-mix(in srgb, #ff0000, #0000ff)": (127.5, 0, 127.5, 1),
        "color-mix(in srgb, red 25%, blue)": (63.75, 0, 191.25, 1),
        # Percentages that sum under 100 scale up, and the sum becomes an alpha multiplier.
        "color-mix(in srgb, red 30%, blue 30%)": (127.5, 0, 127.5, 0.6),
        # CSS Color 5's premultiplication example.
        "color-mix(in srgb, rgb(100% 0% 0% / 0.7) 25%, rgb(0% 100% 0% / 0.2))": (
            0.53846 * 255, 0.46154 * 255, 0, 0.325),
        "rgba(12, 8, 6, .30)": (12, 8, 6, 0.3),
        "#1F1A1780": (31, 26, 23, 128 / 255),
    }
    wrong = {expr: resolver.colour(expr) for expr, want in cases.items() if not close(resolver.colour(expr), want)}
    assert not wrong, f"the evaluator's answers differ from the specification's: {wrong}"

    assert close(over(parse("rgba(0, 0, 0, .5)"), parse("#ffffff")), (127.5, 127.5, 127.5, 1)), (
        "half-transparent black over white is mid grey"
    )
    assert close(over(parse("rgb(255 0 0 / .25)"), parse("rgb(0 0 255)")), (63.75, 0, 191.25, 1))
    assert abs(contrast(parse("#000000"), parse("#ffffff")) - 21.0) < 1e-9, "black on white is 21:1"
    assert round(contrast(parse("#767676"), parse("#ffffff")), 2) == 4.54, "#767676 on white is 4.54:1"

    scheme_light = Resolver({"--oo-x": "light-dark(var(--oo-role-a), var(--oo-role-b))"},
                            {"a": "#ff0000", "b": "#0000ff"}, "light")
    scheme_dark = Resolver({"--oo-x": "light-dark(var(--oo-role-a), var(--oo-role-b))"},
                           {"a": "#ff0000", "b": "#0000ff"}, "dark")
    assert close(scheme_light.colour("--oo-x"), (255, 0, 0, 1)) and close(scheme_dark.colour("--oo-x"), (0, 0, 255, 1)), (
        "light-dark() takes its first colour under a light scheme and its second under a dark one"
    )
    aliased = Resolver({"--oo-y": "var(--oo-x)", "--oo-x": "var(--oo-role-a)", "--oo-z": "var(--oo-none, #00ff00)"},
                       {"a": "#123456"}, "dark")
    assert close(aliased.colour("--oo-y"), (0x12, 0x34, 0x56, 1)) and close(aliased.colour("--oo-z"), (0, 255, 0, 1)), (
        "an alias resolves through its chain, and a fallback stands in for an undeclared name"
    )
    with pytest.raises(KeyError):
        aliased.colour("--oo-undeclared")
    loop = Resolver({"--oo-p": "var(--oo-q)", "--oo-q": "var(--oo-p)"}, {}, "dark")
    with pytest.raises(ValueError):
        loop.colour("--oo-p")

    computed = REPO / _COMPUTED
    if computed.is_file():
        # Browser-computed values, recorded on the machine: each is what the
        # evaluator gives for the same token in the same palette.
        recorded = json.loads(computed.read_text(encoding="utf-8"))
        assert recorded, "the recorded browser values are not empty"
        palettes = _palettes()
        tokens = _derivation()
        differ = []
        for entry in recorded:
            roles, scheme = palettes[entry["palette"]]
            ours = Resolver(_with_palette(tokens, entry["palette"]), roles, scheme).colour(entry["token"])
            if not close(ours, parse(entry["computed"]), tolerance=1.01):
                differ.append((entry, ours))
        assert not differ, f"the evaluator differs from the browser: {differ[:6]}"


# ===========================================================================
# DS18 -- status washes outside the primitives never rise
# ===========================================================================
_WASH = re.compile(r"^--oo-(?:success|warning|error|info)-(?:bg|bd)$")


def _wash_aliases(texts):
    """The tokens whose declared value reads a status wash, or an alias of one."""
    declared = []
    for text in texts:
        declared += re.findall(r"(--oo-[\w-]+)\s*:\s*([^;{}]*)", text)
    aliases, changed = set(), True
    while changed:
        changed = False
        for name, value in declared:
            if name in aliases or _WASH.match(name):
                continue
            if any(_WASH.match(ref) or ref in aliases for ref in _VAR.findall(value)):
                aliases.add(name)
                changed = True
    return aliases


def _wash_reads(aliases):
    def count(path, text):
        return sum(1 for ref in _VAR.findall(text) if _WASH.match(ref) or ref in aliases)
    return count


DS18_LEDGER = {
    'frontend/src/lib/components/chat/BranchExplorer.svelte': 1,
    'frontend/src/lib/components/chat/ChatInput.svelte': 3,
    'frontend/src/lib/components/chat/ChatMessage.svelte': 6,
    'frontend/src/lib/components/chat/CodingAgentInline.svelte': 4,
    'frontend/src/lib/components/chat/ContextPanel.svelte': 2,
    'frontend/src/lib/components/chat/CorrectionIndicator.svelte': 2,
    'frontend/src/lib/components/chat/ExportDialog.svelte': 2,
    'frontend/src/lib/components/chat/FeedbackWidget.svelte': 1,
    'frontend/src/lib/components/chat/ProjectContextBadge.svelte': 2,
    'frontend/src/lib/components/chat/RoutingIndicator.svelte': 2,
    'frontend/src/lib/components/chat/ToolCallApprovalDrawer.svelte': 1,
    'frontend/src/lib/components/chat/ToolCallDisplay.svelte': 4,
    'frontend/src/lib/components/health/CacheManager.svelte': 2,
    'frontend/src/lib/components/health/HealthDashboard.svelte': 3,
    'frontend/src/lib/components/panels/AgentPanel.svelte': 4,
    'frontend/src/lib/components/panels/AnalyticsDashboard.svelte': 2,
    'frontend/src/lib/components/panels/CacheStatsPanel.svelte': 2,
    'frontend/src/lib/components/panels/CascadingPanel.svelte': 2,
    'frontend/src/lib/components/panels/CompressionSettings.svelte': 2,
    'frontend/src/lib/components/panels/ExecPipelinePanel.svelte': 7,
    'frontend/src/lib/components/panels/FileManager.svelte': 6,
    'frontend/src/lib/components/panels/HumanizerPanel.svelte': 1,
    'frontend/src/lib/components/panels/LearnedRouterPanel.svelte': 3,
    'frontend/src/lib/components/panels/ModelAssignment.svelte': 1,
    'frontend/src/lib/components/panels/ModelProfilePanel.svelte': 9,
    'frontend/src/lib/components/panels/PanelToggle.svelte': 9,
    'frontend/src/lib/components/panels/PerformanceDashboard.svelte': 2,
    'frontend/src/lib/components/panels/ProfilerDashboard.svelte': 2,
    'frontend/src/lib/components/panels/ProjectDetail.svelte': 2,
    'frontend/src/lib/components/panels/ProjectList.svelte': 3,
    'frontend/src/lib/components/panels/PromptConfigPanel.svelte': 3,
    'frontend/src/lib/components/panels/ProxySettingsPanel.svelte': 6,
    'frontend/src/lib/components/panels/SandboxFileManager.svelte': 10,
    'frontend/src/lib/components/panels/SkillsPanel.svelte': 5,
    'frontend/src/lib/components/panels/SyncPanel.svelte': 2,
    'frontend/src/lib/components/panels/TelemetryDashboard.svelte': 6,
    'frontend/src/lib/components/panels/TelemetryHistoryPanel.svelte': 7,
    'frontend/src/lib/components/panels/benchmark/benchmark.css': 3,
    'frontend/src/lib/components/rag/DocumentManager.svelte': 1,
    'frontend/src/lib/components/rag/IngestProgress.svelte': 1,
    'frontend/src/lib/components/settings/BackupRestorePanel.svelte': 8,
    'frontend/src/lib/components/settings/ContextOptimizerPanel.svelte': 8,
    'frontend/src/lib/components/settings/FineTunePanel.svelte': 2,
    'frontend/src/lib/components/settings/KnowledgeBasePanel.svelte': 1,
    'frontend/src/lib/components/settings/PerformanceTunerPanel.svelte': 6,
    'frontend/src/lib/components/settings/PluginAllowlistPanel.svelte': 1,
    'frontend/src/lib/components/settings/PluginMarketplace.svelte': 2,
    'frontend/src/lib/components/settings/PluginsPanel.svelte': 7,
    'frontend/src/lib/components/settings/PresetManager.svelte': 3,
    'frontend/src/lib/components/settings/RecoveryCodesPanel.svelte': 1,
    'frontend/src/lib/components/settings/SearchKillSwitchPanel.svelte': 4,
    'frontend/src/lib/components/settings/SecurityModePanel.svelte': 2,
    'frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte': 4,
    'frontend/src/lib/components/settings/TOTPSetup.svelte': 2,
    'frontend/src/lib/components/settings/VisionModelSelector.svelte': 2,
    'frontend/src/lib/components/settings/WebAuthnSetup.svelte': 1,
    'frontend/src/lib/components/settings/sections/ConversationDefaults.svelte': 4,
    'frontend/src/lib/components/ui/ErrorBoundary.svelte': 1,
    'frontend/src/routes/login/+page.svelte': 2,
    'frontend/src/routes/register/+page.svelte': 2,
}


def test_ds18_status_washes_outside_the_primitives_never_rise():
    aliases = _wash_aliases(read(path) for path in files(_STYLED))
    assert {"--oo-err-bg", "--oo-danger-bg"} <= aliases, (
        f"the census finds the aliases that carry a wash: {sorted(aliases)}"
    )
    census = check_ledger(
        "DS18_LEDGER", DS18_LEDGER, _wash_reads(aliases),
        "<div style=\"background: var(--oo-error-bg); border-color: var(--oo-info-bd)\"></div>",
        test_file=__file__, suffixes=_STYLED, exclude=(_PRIMITIVES, _THEME),
    )
    assert census.fixture == 2, census.fixture
    assert _wash_reads({"--oo-danger-bg"})(
        "frontend/src/x.svelte", "a { background: var( --oo-danger-bg); color: var(--oo-error); }"
    ) == 1, "a spaced read of an alias is counted; the status ink is not a wash"


# ===========================================================================
# DS19 -- every accent fill carries the on-accent text
# ===========================================================================
_FILL_TOKENS = ("--oo-acc-fill", "--oo-acc-fill-hover")
_ON_ACCENT = "--oo-fg-on-accent"


def _fill_aliases(tokens):
    """The accent fill tokens and every alias of them in the derivation layer,
    and the tokens that read the on-accent ink."""
    fills = set(_FILL_TOKENS) | {n for n, v in tokens.items() if v.strip() == f"var({_ROLE_PREFIX}primary)"}
    inks = {_ON_ACCENT} | {n for n, v in tokens.items() if v.strip() == f"var({_ROLE_PREFIX}on-primary)"}
    changed = True
    while changed:
        changed = False
        for name, value in tokens.items():
            ref = re.fullmatch(r"var\(\s*(--[\w-]+)\s*\)", value.strip())
            if ref and ref.group(1) in fills and name not in fills:
                fills.add(name)
                changed = True
            if ref and ref.group(1) in inks and name not in inks:
                inks.add(name)
                changed = True
    return fills, inks


def _unpaired_fills(sources, fills, inks):
    """``(fills found, [unpaired])``: every rule, inline style or class list
    whose background is an accent fill, and those that set no on-accent text."""
    fill_ref = re.compile(r"var\(\s*(?:" + "|".join(re.escape(f) for f in sorted(fills, key=len, reverse=True)) + r")\s*\)")
    ink_ref = re.compile(r"var\(\s*(?:" + "|".join(re.escape(i) for i in sorted(inks, key=len, reverse=True)) + r")\s*\)")
    found, unpaired = 0, []
    for path, text in sources.items():
        for selector, body, _ in _rules(_css_of(path, text)):
            declared = _declarations(body)
            if any(name in ("background", "background-color") and fill_ref.search(value) for name, value in declared):
                found += 1
                if not any(name == "color" and ink_ref.search(value) for name, value in declared):
                    unpaired.append(f"{path}: {selector.strip()}")
        if not path.endswith(".svelte"):
            continue
        markup = _STYLE_BLOCK.sub("", text)
        for style in re.findall(r"""\sstyle\s*=\s*(["'])(.*?)\1""", markup, re.S):
            declared = _declarations(style[1])
            if any(name in ("background", "background-color") and fill_ref.search(value) for name, value in declared):
                found += 1
                if not any(name == "color" and ink_ref.search(value) for name, value in declared):
                    unpaired.append(f"{path}: style=\"{style[1][:60]}\"")
        for classes in re.findall(r"""\sclass\s*=\s*(["'])(.*?)\1""", markup, re.S):
            value = classes[1]
            if re.search(r"(?<![\w-])(?:[\w-]+:)*bg-\[" + fill_ref.pattern + r"\]", value):
                found += 1
                if not re.search(r"(?<![\w-])(?:[\w-]+:)*text-\[" + ink_ref.pattern + r"\]", value):
                    unpaired.append(f"{path}: class=\"{value[:60]}\"")
    return found, unpaired


def test_ds19_every_accent_fill_carries_the_on_accent_text():
    fills, inks = _fill_aliases({"--oo-btn-primary-bg": "var(--oo-acc-fill)", "--oo-btn-primary-fg": "var(--oo-role-on-primary)"})
    sample = {
        "frontend/src/a.svelte": (
            "<button class=\"bg-[var(--oo-acc-fill)] text-[var(--oo-fg-on-accent)]\"></button>\n"
            "<button class=\"hover:bg-[var(--oo-acc-fill-hover)] text-[var(--oo-fg-primary)]\"></button>\n"
            "<div style=\"background: var(--oo-btn-primary-bg); color: var(--oo-btn-primary-fg)\"></div>\n"
            "<style>.a { background-color: var(--oo-acc-fill); } .b:hover { background: var(--oo-acc-fill-hover);"
            " color: var(--oo-fg-on-accent); }</style>\n"
        ),
    }
    found, unpaired = _unpaired_fills(sample, fills, inks)
    assert (found, len(unpaired)) == (5, 2), (
        f"the census reads fills in class lists, inline styles and rules, through aliases: {found}, {unpaired}"
    )
    theme = read(_THEME)
    tokens = dict(_declarations(_derivation_blocks(theme)[0][1])) if _derivation_blocks(theme) else {}
    fills, inks = _fill_aliases(tokens)
    found, unpaired = _unpaired_fills(_sources((".svelte", ".css", ".scss")), fills, inks)
    assert found, "the census finds the surfaces filled with the accent fill (none today: the rule reads nothing)"
    assert not unpaired, "accent fills without the on-accent text:\n  " + "\n  ".join(unpaired)

    # Every spelling the paints read (an expression in a style, a class list
    # held by a script, a name the Tailwind configuration gives the fill, a
    # style a script writes), and the converse: the on-accent ink is the text
    # on the accent fill alone, and fills nothing.
    fills = _alias_closure(tokens, _FILL_TOKENS + ("--oo-acc-600",)) | fills
    inks = _alias_closure(tokens, (_ON_ACCENT,)) | inks
    sample = {
        f"{_SRC}/lib/sample/Fills.svelte": (
            "<script>\n  const CLS = { a: 'bg-accent-600 text-surface-100' };\n</script>\n"
            "<button style=\"background-color: {on ? 'var(--oo-acc-fill)' : 'transparent'}; color: var(--oo-fg-primary);\">a</button>\n"
            "<button class=\"{CLS.a}\">b</button>\n"
            "<button class=\"bg-accent-600 hover:bg-accent-500 text-[var(--oo-fg-on-accent)]\">c</button>\n"
            "<span style=\"background-color: var(--oo-acc-fill)\"></span>\n"
            "<b style=\"background: color-mix(in srgb, var(--oo-acc-fill) 12%, transparent); color: var(--oo-acc-ink);\">d</b>\n"
            "<i style=\"background-color: var(--oo-error); color: var(--oo-fg-on-accent);\">e</i>\n"
            "<u style=\"background-color: var(--oo-acc-50);\">f</u>\n"
            "<style>.p { background: var(--oo-acc-600); color: var(--oo-acc-50); }</style>\n"
        ),
    }
    paints, _ = _paints(sample, _tailwind_colours())
    found, unpaired = _accent_fill_pairs(paints, fills, inks)
    misplaced = _on_accent_misplaced(paints, fills, inks)

    def lines(found_in):
        return sorted({int(m.group(1)) for f in found_in for m in [re.search(r"Fills\.svelte:(\d+)", f)] if m})

    assert found == 5 and lines(unpaired) == [4, 5] and lines(misplaced) == [9, 10] and len(misplaced) == 2, (
        "the census reads a fill in a conditional style, in a class list a script holds and in a configured name, "
        "passes a mark and a wash, and names the on-accent ink on a status fill and as a fill: "
        f"{found}, {unpaired}, {misplaced}"
    )
    paints, _ = _tree_paints()
    found, unpaired = _accent_fill_pairs(paints, fills, inks)
    assert found >= 60 and not unpaired, (
        f"accent fills the paints read, {found}, each with the on-accent text:\n  " + "\n  ".join(unpaired)
    )
    misplaced = _on_accent_misplaced(paints, fills, inks)
    assert not misplaced, "the on-accent ink off the accent fill:\n  " + "\n  ".join(misplaced)


# ===========================================================================
# DS20 -- the motion classes the theme path sets are the ones scrolling reads
# ===========================================================================
_SMOOTH = re.compile(
    r"""behavior\s*:\s*(['"`])smooth\1|scroll-behavior\s*:\s*smooth\b"""
)
_BEHAVIOUR_OPTION = re.compile(r"(?<![\w-])behavior\s*:\s*([^,}\n]+)")


def _imports(text, name, module):
    named = (
        r"import\s*(?:type\s*)?\{[^}]*\b" + re.escape(name) + r"\b[^}]*\}\s*from\s*['\"]"
        + re.escape(module) + r"['\"]"
    )
    return re.search(named, text) is not None


def test_ds20_the_motion_classes_the_theme_path_sets_are_the_ones_scrolling_reads():
    sample = (
        "el.scrollIntoView({ behavior: 'smooth' }); window.scrollTo({ top: 0, behavior: \"smooth\" });\n"
        "html { scroll-behavior: smooth; }\n"
    )
    assert len(_SMOOTH.findall(sample)) == 3, "the census reads smooth scrolling in its spellings"
    assert not _SMOOTH.findall("scroll-behavior: auto; behavior: 'auto'; smoothing: 1;")

    sources = _sources(_EVERY, exclude=(_MOTION,))
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

    out = _node("motion", {"values": [None, "system", "reduced", "full"]}, with_motion=True)
    reduced, full = out["reads"]
    motion_classes = {value: sorted(c for c in classes if c in (reduced, full)) for value, classes in out["sets"].items()}
    assert motion_classes == {"null": [], "system": [], "reduced": [reduced], "full": [full]}, (
        f"the theme path sets exactly the class motion.ts reads for each motion choice: {motion_classes}"
    )


# ===========================================================================
# Paint: every place the frontend paints, with every alternative it can take
# ===========================================================================
# A CSS rule, an element's style attribute, style directives, class list and
# painting attributes, and a script's style write. An attribute value that
# holds expressions is expanded into every string it can be: each branch of
# a conditional or a fallback, a literal, a template literal, the values of
# an object or an array, and what a name holds (a function's returns, a
# constant, a property's default), followed into the module it is imported
# from. Each character of an expanded string is traced to the source it was
# written in, so a read of a token is placed in the declaration it lands in.
_VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
_EXPANSION_CAP = 64
_EXPANSION_DEPTH = 16
_MARKUP_NOISE = r"<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>|<!--.*?-->"
_STYLE_NOISE = r"<style\b[^>]*>.*?</style>"
_PAINTED = (".svelte", ".css", ".scss", ".ts", ".js")


def _quote_end(text, index):
    """The index just past the string opening at ``index``; a template
    literal's ``${...}`` parts are skipped whole."""
    quote = text[index]
    index += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if quote == "`" and text.startswith("${", index):
            index = _brace_end(text, index + 1)
            continue
        if char == quote:
            return index + 1
        index += 1
    return index


def _brace_end(text, index):
    """The index just past the ``{...}`` opening at ``index``."""
    depth = 0
    while index < len(text):
        char = text[index]
        if char in _QUOTES:
            index = _quote_end(text, index)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return index


def _blanked(text, pattern):
    """``text`` with every match of ``pattern`` turned to spaces, offsets kept."""
    return re.sub(pattern, lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


class _Element:
    """One element of a component's markup: its tag, attributes, parent and
    children, and whether it holds text of its own."""
    __slots__ = ("tag", "start", "attributes", "parent", "children", "text")

    def __init__(self, tag, start, attributes, parent):
        self.tag, self.start, self.attributes, self.parent = tag, start, attributes, parent
        self.children, self.text = [], False

    @property
    def empty(self):
        """No child element and no text: a mark, never a surface for text."""
        return not self.children and not self.text

    def classes(self):
        """The class names written on the element (static, in expressions or as directives)."""
        out = set()
        for name, raw, _ in self.attributes:
            if name == "class" and raw:
                out |= set(re.findall(r"[\w-]+", raw))
            elif name.startswith("class:"):
                out.add(name[6:])
        return out


_ATTRIBUTE_NAME = re.compile(r"[^\s=/>\"'{}]+")


def _value_end(markup, index, end):
    """The index just past the attribute value starting at ``index``."""
    if markup[index] in "\"'":
        close = index + 1
        while close < end and markup[close] != markup[index]:
            close = _brace_end(markup, close) if markup[close] == "{" else close + 1
        return min(close + 1, end)
    if markup[index] == "{":
        return _brace_end(markup, index)
    bare = re.match(r"[^\s>]+", markup[index:end])
    return index + (len(bare.group(0)) if bare else 0)


def _attributes(markup, start, end):
    """``[(name, raw value or None, offset of the value)]`` of a tag."""
    out, index = [], start
    while index < end:
        char = markup[index]
        if char.isspace() or char == "/":
            index += 1
            continue
        if char == "{":
            stop = _brace_end(markup, index)
            out.append(("{}", markup[index:stop], index))
            index = stop
            continue
        match = _ATTRIBUTE_NAME.match(markup, index)
        if not match:
            index += 1
            continue
        name, index = match.group(0), match.end()
        probe = index
        while probe < end and markup[probe].isspace():
            probe += 1
        if probe >= end or markup[probe] != "=":
            out.append((name, None, index))
            continue
        index = probe + 1
        while index < end and markup[index].isspace():
            index += 1
        stop = _value_end(markup, index, end) if index < end else end
        out.append((name, markup[index:stop], index))
        index = stop
    return out


def _elements(markup):
    """Every element of a component's markup, in order, each with its parent,
    its children and whether it holds text (a character, an entity or an
    expression)."""
    found, stack, index = [], [], 0
    while index < len(markup):
        char = markup[index]
        if char == "{":
            stop = _brace_end(markup, index)
            body = markup[index + 1:stop - 1].lstrip()
            if stack and body and body[0] not in "#:/@":
                stack[-1].text = True
            index = stop
            continue
        if char == "<":
            closing = re.match(r"</\s*([\w:.-]+)\s*>", markup[index:index + 80])
            if closing:
                for depth in range(len(stack) - 1, -1, -1):
                    if stack[depth].tag == closing.group(1):
                        del stack[depth:]
                        break
                index += closing.end()
                continue
            opening = re.match(r"<([A-Za-z][\w:.-]*)", markup[index:index + 80])
            if opening:
                cursor = index + opening.end()
                while cursor < len(markup) and markup[cursor] != ">":
                    if markup[cursor] in "\"'{":
                        cursor = _value_end(markup, cursor, len(markup))
                        continue
                    cursor += 1
                tag = opening.group(1)
                parent = stack[-1] if stack else None
                element = _Element(tag, index, _attributes(markup, index + opening.end(), cursor), parent)
                if parent is not None:
                    parent.children.append(element)
                found.append(element)
                if not (markup[cursor - 1] == "/" or tag.lower() in _VOID):
                    stack.append(element)
                index = cursor + 1
                continue
        elif not char.isspace() and stack:
            stack[-1].text = True
        index += 1
    return found


# -- The strings an expression can give ---------------------------------
# A string is a tuple of pieces ``(text, path, offset)``: its characters, each
# traced to the source it was written in (a placeholder piece has no path).
def _joined(pieces):
    return "".join(piece[0] for piece in pieces)


def _origin(pieces, position):
    """``(path, offset)`` of the character at ``position`` of a string."""
    for text, path, offset in pieces:
        if position < len(text):
            return (path, offset + position) if path else None
        position -= len(text)
    return None


def _top_level(expr, targets):
    """``[index]`` of the characters of ``targets`` outside strings, brackets
    and braces; ``?.`` and ``??`` open no conditional."""
    found, depth, index = [], 0, 0
    while index < len(expr):
        char = expr[index]
        if char in _QUOTES:
            index = _quote_end(expr, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char in targets:
            if char == "?" and expr[index + 1:index + 2] in ("?", "."):
                index += 2
                continue
            found.append(index)
        index += 1
    return found


def _module_of(importer, module, known=()):
    """The repository path an import names (a source being read, or a file
    of the tree), or None outside the tree."""
    if module.startswith("$lib/"):
        parts = f"{_SRC}/lib/{module[5:]}".split("/")
    elif module.startswith("."):
        parts = []
        for part in (str(PurePosixPath(importer).parent) + "/" + module).split("/"):
            if part == "..":
                if parts:
                    parts.pop()
            elif part not in ("", "."):
                parts.append(part)
    else:
        return None
    base = "/".join(parts)
    for candidate in (base, base + ".ts", base + ".js", base + "/index.ts"):
        if candidate in known or (REPO / candidate).is_file():
            return candidate
    return None


class _Scope:
    """The sources a census reads, and the scripts in them."""

    def __init__(self, sources):
        self.sources = dict(sources)
        self._scripts = {}

    def script(self, path):
        if path not in self._scripts:
            if path not in self.sources:
                self.sources[path] = read(path)
            text = self.sources[path]
            self._scripts[path] = _blanked(text, _STYLE_NOISE) if path.endswith(".svelte") else text
        return self._scripts[path]

    def imported(self, path, name):
        """The module ``path`` imports ``name`` from, or None."""
        for names, module in re.findall(r"""import\s*\{([^}]*)\}\s*from\s*['"]([^'"]+)['"]""", self.script(path)):
            if name in [n.strip().split(" as ")[-1].strip() for n in names.split(",")]:
                return _module_of(path, module, self.sources)
        return None


def _statement_end(text, start):
    """Where the expression starting at ``start`` ends: a top-level ``;``, a
    bracket it did not open, or a line end that continues nothing."""
    depth, cursor = 0, start
    while cursor < len(text):
        char = text[cursor]
        if char in _QUOTES:
            cursor = _quote_end(text, cursor)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                return cursor
            depth -= 1
        elif char == ";" and depth == 0:
            return cursor
        elif char == "\n" and depth == 0:
            before = text[start:cursor].rstrip()
            after = text[cursor:].lstrip()
            if not before.endswith(("?", ":", "||", "&&", "??", "+", "=", ",", "(")) and not after.startswith(
                ("?", ":", "||", "&&", "??", "+", ".")
            ):
                return cursor
        cursor += 1
    return cursor


def _held(script, name):
    """``[(expression, offset)]``: what ``name`` holds in a script -- each
    expression a function of that name returns (or an arrow's body), or the
    value a constant, a variable, a property default or a reactive statement
    gives it."""
    body = _function_body(script, name)
    if body is not None:
        start = script.find(body)
        return [
            (body[m.end():_statement_end(body, m.end())], start + m.end())
            for m in re.finditer(r"\breturn\b", body)
        ]
    match = re.search(
        r"(?:(?:export\s+)?(?:const|let|var)\s+" + re.escape(name)
        + r"\b\s*(?::\s*[^=;\n]+)?=(?!=)|\$:\s*" + re.escape(name) + r"\s*=(?!=))\s*"
        r"(?:(?:async\s*)?\([^)]*\)\s*(?::[^=]*)?=>\s*(?!\{))?",
        script,
    )
    if not match:
        return []
    return [(script[match.end():_statement_end(script, match.end())], match.end())]


def _strings(expr, path, offset, scope, depth=0):
    """Every string ``expr`` can evaluate to: each branch of a conditional,
    each side of a fallback (``||``, ``??``), a literal, a template literal
    with its parts expanded, the values of an object or an array literal, or
    what a name it reads holds. An expression that gives no string gives
    nothing."""
    lead = len(expr) - len(expr.lstrip())
    expr, offset = expr.strip(), offset + lead
    if not expr or depth > _EXPANSION_DEPTH:
        return []
    marks = _top_level(expr, "?")
    if marks:
        colons = _top_level(expr[marks[0] + 1:], ":")
        if colons:
            first = marks[0] + 1
            split = first + colons[0]
            return (_strings(expr[first:split], path, offset + first, scope, depth + 1)
                    + _strings(expr[split + 1:], path, offset + split + 1, scope, depth + 1))
    for operator in ("||", "??"):
        cuts = [i for i in _top_level(expr, operator[0]) if expr[i:i + 2] == operator]
        if cuts:
            out, last = [], 0
            for cut in cuts + [len(expr)]:
                out += _strings(expr[last:cut], path, offset + last, scope, depth + 1)
                last = cut + 2
            return out
    if expr[0] in "\"'" and _quote_end(expr, 0) == len(expr):
        return [((expr[1:-1], path, offset + 1),)]
    if expr[0] == "`" and _quote_end(expr, 0) == len(expr):
        return _expanded(expr[1:-1], path, offset + 1, scope, depth + 1, template=True)
    if expr[0] == "(" and expr.endswith(")"):
        return _strings(expr[1:-1], path, offset + 1, scope, depth + 1)
    if expr[0] in "{[" and expr[-1] in "}]":
        inner, out, last = expr[1:-1], [], 0
        for cut in _top_level(inner, ",") + [len(inner)]:
            item = inner[last:cut]
            colon = _top_level(item, ":") if expr[0] == "{" else []
            value_at = colon[0] + 1 if colon else 0
            out += _strings(item[value_at:], path, offset + 1 + last + value_at, scope, depth + 1)
            last = cut + 1
        return out
    name = re.match(r"[A-Za-z_$][\w$]*", expr)
    if name and name.group(0) not in ("true", "false", "null", "undefined", "this"):
        return _name_strings(name.group(0), path, scope, depth + 1)
    return []


def _name_strings(name, path, scope, depth):
    """The strings ``name`` holds in ``path``, or in the module it imports it from."""
    held = _held(scope.script(path), name)
    if not held:
        module = scope.imported(path, name)
        return _name_strings(name, module, scope, depth + 1) if module and depth <= _EXPANSION_DEPTH else []
    out = []
    for expr, offset in held:
        out += _strings(expr, path, offset, scope, depth + 1)
    return out


def _expanded(value, path, offset, scope, depth=0, template=False):
    """Every string a value with expressions can be (``{...}`` in markup,
    ``${...}`` in a template literal): each expression replaced by each
    string it can give. Expressions of the same text, or of the same
    condition, move together; an expression that gives no string is a
    placeholder that paints nothing."""
    opener = "${" if template else "{"
    segments, index = [], 0
    while index < len(value):
        at = value.find(opener, index)
        if at < 0:
            segments.append(("text", ((value[index:], path, offset + index),)))
            break
        if at > index:
            segments.append(("text", ((value[index:at], path, offset + index),)))
        brace = at + (1 if template else 0)
        stop = _brace_end(value, brace)
        segments.append(("expr", (value[brace + 1:stop - 1], offset + brace + 1)))
        index = stop
    results = [((), {})]
    for kind, segment in segments:
        if kind == "text":
            results = [(pieces + segment, ranks) for pieces, ranks in results]
            continue
        body, at = segment
        choices = _strings(body, path, at, scope, depth + 1) or [(("\x00", None, 0),)]
        key = body.strip()
        marks = _top_level(key, "?")
        key = key[:marks[0]].strip() if marks else key
        grown = []
        for pieces, ranks in results:
            if key in ranks and ranks[key][1] == len(choices):
                grown.append((pieces + choices[ranks[key][0]], ranks))
                continue
            for rank, choice in enumerate(choices):
                grown.append((pieces + choice, {**ranks, key: (rank, len(choices))}))
        results = grown[:_EXPANSION_CAP]
    return [pieces for pieces, _ in results]


# -- What a painted string declares ---------------------------------------
def _declared(pieces):
    """``[(property, value, start, end)]`` of a style string, split at its
    top-level semicolons, each value's span in the string."""
    text = _joined(pieces)
    out, depth, start = [], 0, 0
    for index in range(len(text) + 1):
        char = text[index] if index < len(text) else ";"
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == ";" and depth <= 0:
            part = text[start:index]
            if ":" in part:
                name, value = part.split(":", 1)
                value_at = start + len(name) + 1 + (len(value) - len(value.lstrip()))
                out.append((name.strip().lower(), value.strip(), value_at, value_at + len(value.strip())))
            start = index + 1
    return out


_TW_KINDS = {
    "bg": "background-color", "text": "color", "border": "border-color", "fill": "fill", "stroke": "stroke",
    "from": "background-image", "via": "background-image", "to": "background-image", "ring": "outline-color",
    "outline": "outline-color", "divide": "border-color", "placeholder": "color",
    "decoration": "text-decoration-color", "caret": "caret-color", "accent": "accent-color", "shadow": "box-shadow",
}
_TW_UTILITY = re.compile(
    r"(?<![\w\[-])((?:[\w-]+:)*)(bg|text|border(?:-[trblxyse])?|fill|stroke|from|via|to|ring|outline|divide"
    r"|placeholder|decoration|caret|accent|shadow)-(\[[^\]\s]+\]|[a-z]+-\d+|white|black)(?:/(\d+))?(?![\w\[-])"
)


def _utilities(pieces, colours):
    """``[(variant, property, colour expression, start, end)]`` of the colour
    utilities of a class string: arbitrary values, and the names the
    Tailwind configuration gives a token (``colours``: name -> token), with
    an opacity modifier mixed in with transparent."""
    text = _joined(pieces)
    out = []
    for match in _TW_UTILITY.finditer(text):
        variant, kind, value, alpha = match.groups()
        root = kind.split("-")[0]
        if value.startswith("["):
            inner = value[1:-1]
            if not re.match(r"(?:var|color-mix|rgba?|hsla?|oklch|#)", inner):
                continue
            expr = inner.replace("_", " ")
        else:
            token = colours.get(f"{root}-{value}")
            if token is None:
                continue
            expr = f"var({token})"
        if alpha:
            expr = f"color-mix(in srgb, {expr} {alpha}%, transparent)"
        out.append((variant, _TW_KINDS[root], expr, match.start(3), match.end(3)))
    return out


_TAILWIND_DRIVER = r"""
const clause = process.argv[2];
const config = (await import(process.env.OO_TAILWIND)).default;
console.log('RESULT ' + JSON.stringify((config.theme && config.theme.extend) || {}));
console.log('PASS ' + clause);
"""
_TW_FAMILIES = {
    "backgroundColor": ("bg",), "textColor": ("text",), "borderColor": ("border",), "ringColor": ("ring",),
    "outlineColor": ("outline",), "divideColor": ("divide",), "placeholderColor": ("placeholder",),
    "gradientColorStops": ("from", "via", "to"),
}


@functools.cache
def _tailwind_colours():
    """``{utility name: token}`` of the colour names the Tailwind
    configuration defines (``bg-accent-600`` -> ``--oo-acc-fill``)."""
    out = run_ts({"OO_TAILWIND": _TAILWIND}, _TAILWIND_DRIVER, "extend")
    extend = json.loads([line for line in out.splitlines() if line.startswith("RESULT ")][0][len("RESULT "):])
    table = {}
    for key, prefixes in _TW_FAMILIES.items():
        for family, steps in (extend.get(key) or {}).items():
            if not isinstance(steps, dict):
                continue
            for step, value in steps.items():
                token = re.search(r"var\((--[\w-]+)\)", str(value))
                if token:
                    for prefix in prefixes:
                        table[f"{prefix}-{family}-{step}"] = token.group(1)
    return table


# -- The paints -----------------------------------------------------------
class _Paint:
    """One painted thing: a CSS rule, an element (its style attribute,
    directives, classes and painting attributes), or a script's style write.
    ``alternatives`` holds what it can paint, each a list of
    ``(variant, property, value, origin)``, where ``origin`` is
    ``(pieces, start, end)``, the span of the string the value came from, or
    None for a rule; ``element`` is the markup element it paints."""
    __slots__ = ("path", "where", "alternatives", "element", "selector")

    def __init__(self, path, where, alternatives, element=None, selector=None):
        self.path, self.where, self.alternatives = path, where, alternatives
        self.element, self.selector = element, selector


def _whole(pieces, prop):
    return [("", prop, _joined(pieces), (pieces, 0, len(_joined(pieces))))]


def _combined(groups):
    """An element's alternatives: one choice from each attribute's."""
    combos = [[]]
    for group in groups:
        if group:
            combos = [combo + choice for combo in combos for choice in group][:_EXPANSION_CAP]
    return combos


_STYLE_WRITE = re.compile(r"\.style\.(?:setProperty\(\s*['\"`]([\w-]+)['\"`]\s*,|([A-Za-z]+)\s*=(?!=))")
_PAINTING_ATTRIBUTES = ("fill", "stroke", "stop-color", "flood-color", "color")


def _paints(sources, colours):
    """``(paints, elements)``: every paint of ``sources`` (``{path: text}``),
    and every component's elements (``{path: [element]}``)."""
    scope = _Scope(sources)
    paints, elements = [], {}
    for path, text in sources.items():
        for selector, body, _ in _rules(_css_of(path, text)):
            declared = [("", name.lower(), value, None) for name, value in _declarations(body)]
            paints.append(_Paint(path, f"{path}: {selector.strip()[:80]}", [declared], selector=selector))
        if path.endswith(_SCRIPTS):
            script = scope.script(path)
            for match in _STYLE_WRITE.finditer(script):
                prop = match.group(1) or re.sub(r"[A-Z]", lambda m: "-" + m.group(0).lower(), match.group(2))
                stop = _statement_end(script, match.end())
                values = _strings(script[match.end():stop], path, match.end(), scope)
                line = text.count("\n", 0, match.start()) + 1
                paints.append(_Paint(path, f"{path}:{line} style write", [_whole(v, prop) for v in values]))
        if not path.endswith(".svelte"):
            continue
        elements[path] = _elements(_blanked(text, _MARKUP_NOISE))
        for element in elements[path]:
            groups = []
            for name, raw, at in element.attributes:
                if raw is None:
                    continue
                value, value_at = (raw[1:-1], at + 1) if raw[0] in "\"'" else (raw, at)
                if name == "style":
                    groups.append([
                        [("", prop, v, (p, start, end)) for prop, v, start, end in _declared(p)]
                        for p in _expanded(value, path, value_at, scope)
                    ])
                elif name.startswith("style:"):
                    prop = name[6:].split("|")[0].lower()
                    groups.append([_whole(p, prop) for p in _expanded(value, path, value_at, scope)])
                elif name == "class":
                    groups.append([
                        [(variant, prop, expr, (p, start, end)) for variant, prop, expr, start, end in _utilities(p, colours)]
                        for p in _expanded(value, path, value_at, scope)
                    ])
                elif name.startswith("class:"):
                    groups.append([[(v, prop, expr, None) for v, prop, expr, _, _ in _utilities(((name[6:], None, 0),), colours)]])
                elif name in _PAINTING_ATTRIBUTES:
                    groups.append([_whole(p, name) for p in _expanded(value, path, value_at, scope)])
            alternatives = _combined(groups)
            if any(alternatives):
                line = text.count("\n", 0, element.start) + 1
                paints.append(_Paint(path, f"{path}:{line} <{element.tag}>", alternatives, element=element))
    return paints, elements


@functools.cache
def _tree_paints():
    """The paints and elements of the tree, read once per session."""
    return _paints(_sources(_PAINTED, exclude=tuple(_PALETTE_FILES.values())), _tailwind_colours())


def _reads(value):
    """The custom properties a value reads."""
    return set(re.findall(r"var\(\s*(--[\w-]+)", value))


def _alias_closure(tokens, seeds):
    """``seeds`` and every derivation token that is exactly ``var()`` of one of them."""
    names, changed = set(seeds), True
    while changed:
        changed = False
        for name, value in tokens.items():
            ref = re.fullmatch(r"var\(\s*(--[\w-]+)\s*\)", value.strip())
            if ref and ref.group(1) in names and name not in names:
                names.add(name)
                changed = True
    return names


def _grounds_of(alternative):
    """``{variant: {"bg": value, "fg": value, "opacity": value}}`` of one alternative."""
    out = {}
    for variant, prop, value, _ in alternative:
        slot = {"background": "bg", "background-color": "bg", "color": "fg", "opacity": "opacity"}.get(prop)
        if slot:
            out.setdefault(variant, {})[slot] = value
    return out


# ===========================================================================
# DS21 -- the pairs the components draw pass in every palette
# ===========================================================================
_NOT_A_PAINT = ("", "transparent", "none", "inherit", "currentcolor", "unset", "initial", "revert", "revert-layer")
_ONE_COLOUR = re.compile(
    r"var\(\s*--[\w-]+\s*(?:,[^()]*(?:\([^()]*\))?[^()]*)?\)|color-mix\(.*\)|#[0-9a-fA-F]{3,8}|rgba?\([^()]*\)", re.S,
)
_GLUED = re.compile(r"var\(\s*--[\w-]+\s*\)[0-9a-fA-F]{2}\b")


def _colour_value(value):
    """The colour a background or text value paints, or None when it paints
    none of its own (none, inherit, a gradient, an image, a placeholder)."""
    value = re.sub(r"\s*!important\s*$", "", value.strip())
    if value.lower() in _NOT_A_PAINT or "\x00" in value or not _ONE_COLOUR.fullmatch(value):
        return None
    return value


def _pair_findings(paints, tokens, palettes):
    """``(pairs checked, [finding])``: every background painted together with
    a text colour, in one rule or on one element (a variant's background
    with its own text colour, or the element's), under 4.5:1 in a palette;
    a translucent background is composited over the surface and over the
    page, and the lower ratio counts. A token with digits glued to it (a hex
    alpha written after ``var()``) is no colour at all, and is named."""
    resolvers = {pid: _colour_resolver(tokens, roles, scheme, pid) for pid, (roles, scheme) in palettes.items()}
    from _colour import contrast, over

    checked, findings = 0, []
    for paint in paints:
        seen = set()
        for alternative in paint.alternatives:
            for variant, prop, value, _ in alternative:
                if _GLUED.search(value):
                    findings.append(f"{paint.where}: {prop}: {value}: digits glued to a token are no colour")
            slots = _grounds_of(alternative)
            base = slots.get("", {})
            for variant, slot in slots.items():
                ground = _colour_value(slot.get("bg", ""))
                ink = _colour_value(slot["fg"] if "fg" in slot else base.get("fg", ""))
                if not ground or not ink or (ground, ink) in seen:
                    continue
                seen.add((ground, ink))
                checked += 1
                for pid, resolver in resolvers.items():
                    try:
                        painted = resolver.colour(ground)
                        worst = min(
                            contrast(over(resolver.colour(ink), solid), solid)
                            for solid in (
                                over(painted, resolver.colour(under)) if painted[3] < 1 else painted
                                for under in ("--oo-bg-surface", "--oo-bg-base")
                            )
                        )
                    except (KeyError, ValueError) as exc:
                        findings.append(f"{paint.where} {variant}: {ink} on {ground} cannot be evaluated in {pid} ({exc})")
                        continue
                    if worst < 4.5:
                        findings.append(f"{paint.where} {variant}: {pid}: {ink} on {ground}: {worst:.2f} < 4.5")
    return checked, sorted(set(findings))


def _colour_resolver(tokens, roles, scheme, pid):
    """The evaluator for palette ``pid``: its roles, and the app tokens as it sees them."""
    from _colour import Resolver

    return Resolver(_with_palette(tokens, pid), roles, scheme)


_PAIRS_SAMPLE = {
    f"{_SRC}/lib/sample/Pairs.svelte": (
        "<script>\n"
        "  function tone(k) { if (k === 'a') return 'var(--oo-error)'; return 'var(--oo-info)'; }\n"
        "  const CHIPS = { a: 'bg-[var(--oo-info)]/20 text-[var(--oo-info)]' };\n"
        "</script>\n"
        "<button style=\"background-color: var(--oo-error); color: var(--oo-fg-on-accent);\">x</button>\n"
        "<span style=\"{on ? 'background-color: var(--oo-warning); color: var(--oo-fg-on-accent);'"
        " : 'background-color: var(--oo-bg-surface); color: var(--oo-fg-primary);'}\">y</span>\n"
        "<span class=\"px-1 {CHIPS[k]}\">z</span>\n"
        "<span style=\"background-color: {tone(k)}20; color: {tone(k)};\">w</span>\n"
        "<span style=\"color: {on ? 'var(--oo-fg-on-accent)' : 'var(--oo-fg-secondary)'};"
        " background-color: {on ? 'var(--oo-acc-fill)' : 'transparent'};\">ok</span>\n"
        "<button class=\"bg-accent-600 hover:bg-accent-500 text-surface-100\">t</button>\n"
        "<style>.v { background: var(--oo-msg-user-bg); color: var(--oo-fg-on-accent); }</style>\n"
    ),
}


def test_ds21_the_pairs_the_components_draw_pass_in_every_palette():
    tokens, palettes = _derivation(), _palettes()
    colours = _tailwind_colours()
    assert colours.get("bg-accent-600") == "--oo-acc-fill" and colours.get("text-surface-100") == "--oo-fg-primary", (
        f"the census reads the colour names the Tailwind configuration defines: {len(colours)}"
    )
    paints, _ = _paints(_PAIRS_SAMPLE, colours)
    _, caught = _pair_findings(paints, tokens, palettes)
    lines = sorted({int(m.group(1)) for f in caught for m in [re.search(r"Pairs\.svelte:(\d+)", f)] if m})
    expected = {5, 6, 7, 8, 10}
    assert set(lines) == expected and any(".v" in f and "night" in f for f in caught), (
        "the census reads a style, a conditional style, a class list from a script, a glued alpha, "
        f"the configured names and a rule, and pairs a condition's branches: lines {lines}, {caught}"
    )
    assert not any("Pairs.svelte:9 " in f for f in caught), (
        f"the branches of one condition pair together, never across: {caught}"
    )

    paints, _ = _tree_paints()
    checked, findings = _pair_findings(paints, tokens, palettes)
    assert checked >= 500, f"the census reads the pairs the components draw: {checked}"
    assert not findings, (
        f"{len(findings)} pairs the components draw fall under 4.5:1:\n  " + "\n  ".join(findings)
    )


# ===========================================================================
# DS22 -- marks: text-less fills read the mark token and reach 3:1
# ===========================================================================
_MARK_TOKENS = ("--oo-acc-mark", "--oo-acc-mark-2")
_CHART_PREFIXES = ("--oo-pipe-", "--oo-cat-", "--oo-radar-")
_MARK_GROUNDS = ("--oo-bg-base", "--oo-bg-surface", "--oo-bg-overlay", "--oo-bg-subtle")
_STATUS_INKS = ("--oo-success", "--oo-warning", "--oo-error", "--oo-info", "--oo-status-ok")
# The grounds a mark ink may sit on, where the design gives it fewer than
# the four: the status dot's tone is held at 3:1 on the surface and the
# second surface (the design's pair list, 3.07 at the least in day), and is
# drawn on those; on the sunken ground it is 2.91 in day.
_MARK_GROUNDS_OF = {"--oo-status-ok": ("--oo-bg-base", "--oo-bg-surface", "--oo-bg-overlay")}


def _subjects(selector):
    """The class sets of the elements a selector styles, one per comma part."""
    out = []
    for part in selector.split(","):
        last = re.split(r"\s+|>|~|\+", part.strip())[-1]
        out.append(set(re.findall(r"\.([\w-]+)", last)))
    return out


def _mark_findings(paints, elements, tokens):
    """Every mark -- an element that holds no text, or a rule whose every
    element holds none -- filled with the accent fill, and every mark token
    filling something that holds text or setting a text colour."""
    fills = _alias_closure(tokens, ("--oo-acc-fill", "--oo-acc-fill-hover"))
    marks = _alias_closure(tokens, _MARK_TOKENS)
    everywhere = [element for listed in elements.values() for element in listed]
    findings, counted = [], 0
    for paint in paints:
        grounds, inks = set(), False
        for alternative in paint.alternatives:
            for _, prop, value, _ in alternative:
                if prop in ("background", "background-color"):
                    grounds |= _reads(value)
                elif prop == "color" and _colour_value(value):
                    inks = True
        if not grounds & (fills | marks):
            continue
        if paint.element is not None:
            empty = paint.element.empty
        else:
            scope = elements.get(paint.path, []) if paint.path.endswith(".svelte") else everywhere
            styled = [e for subject in _subjects(paint.selector or "") if subject
                      for e in scope if subject <= e.classes()]
            if not styled:
                continue
            empty = all(e.empty for e in styled)
        counted += 1
        if empty and grounds & fills:
            findings.append(f"{paint.where}: a mark (it holds no text) filled with the accent fill")
        if grounds & marks and (not empty or inks):
            findings.append(f"{paint.where}: the mark token fills something that holds text or sets a text colour")
    return counted, sorted(set(findings))


_MARKS_SAMPLE = {
    f"{_SRC}/lib/sample/Marks.svelte": (
        "<div class=\"track\"><div class=\"h-full\" style=\"background-color: var(--oo-acc-fill)\"></div></div>\n"
        "<span class=\"bar\"></span><span class=\"bar\" />\n"
        "<div style=\"background: var(--oo-acc-mark)\">text</div>\n"
        "<span class=\"dot bg-accent-600\"></span>\n"
        "<button class=\"bg-[var(--oo-acc-fill)] text-[var(--oo-fg-on-accent)]\">Go</button>\n"
        "<div style=\"background-color: var(--oo-acc-mark)\"></div>\n"
        "<style>.bar { background: var(--oo-acc-fill-hover); }</style>\n"
    ),
}


def test_ds22_marks_read_the_mark_token_and_reach_three_to_one():
    tokens, palettes = _derivation(), _palettes()
    from _colour import contrast, over

    inks = list(_MARK_TOKENS) + sorted(n for n in tokens if n.startswith(_CHART_PREFIXES)) + list(_STATUS_INKS)
    low = []
    for pid, (roles, scheme) in palettes.items():
        resolver = _colour_resolver(tokens, roles, scheme, pid)
        for ink in inks:
            for ground in _MARK_GROUNDS_OF.get(ink, _MARK_GROUNDS):
                try:
                    base = resolver.colour(ground)
                    ratio = contrast(over(resolver.colour(ink), base), base)
                except (KeyError, ValueError) as exc:
                    low.append(f"{pid}: {ink} on {ground} cannot be evaluated ({exc})")
                    continue
                if ratio < 3.0:
                    low.append(f"{pid}: {ink} on {ground}: {ratio:.2f} < 3.0")
    assert not low, "a mark reaches 3:1 on every ground it sits on:\n  " + "\n  ".join(low)

    paints, elements = _paints(_MARKS_SAMPLE, _tailwind_colours())
    _, caught = _mark_findings(paints, elements, tokens)
    lines = sorted({int(m.group(1)) for f in caught for m in [re.search(r"Marks\.svelte:(\d+)", f)] if m})
    assert lines == [1, 3, 4] and any(".bar" in f for f in caught) and len(caught) == 4, (
        "the census finds a mark filled with the fill (inline, by a rule, by a configured name) and "
        f"the mark token on text, and nothing else: {caught}"
    )

    paints, elements = _tree_paints()
    counted, findings = _mark_findings(paints, elements, tokens)
    assert counted >= 20, f"the census reads the tree's accent fills and marks: {counted}"
    assert not findings, f"{len(findings)} marks read otherwise than the mark token:\n  " + "\n  ".join(findings)


# ===========================================================================
# DS23 -- a switch's knob sits on the switch tracks; a field's edge is seen
# ===========================================================================
_TRACKS = ("--oo-switch-on", "--oo-switch-off")
_KNOB = "--oo-toggle-knob"


def _fills_of(paint, *, dimmed=True):
    """The tokens a paint fills with, per alternative; a dimmed alternative
    (one that sets an opacity below 1: an unknown state, inactive) is left
    out unless asked for."""
    out = []
    for alternative in paint.alternatives:
        slots = _grounds_of(alternative).get("", {})
        opacity = slots.get("opacity", "1").strip()
        if not dimmed and re.fullmatch(r"0?\.\d+|0", opacity):
            continue
        out.append(_reads(slots.get("bg", "")))
    return out


def _switch_findings(paints):
    """``(knobs, [finding])``: every knob -- an element filled with the knob
    token, or the one child, holding nothing, of an element whose fill
    changes with its state -- and the ones not filled with the knob token or
    not sitting on the switch tracks."""
    painted = {id(paint.element): paint for paint in paints if paint.element is not None}
    knobs, findings = 0, []
    for paint in paints:
        element = paint.element
        if element is None:
            continue
        own = set().union(*_fills_of(paint)) if paint.alternatives else set()
        parent = element.parent
        track = painted.get(id(parent)) if parent is not None else None
        changing = track is not None and len({frozenset(f) for f in _fills_of(track)}) > 1
        sole = parent is not None and len(parent.children) == 1 and element.empty and not parent.text
        if _KNOB not in own and not (sole and changing and own):
            continue
        knobs += 1
        if own != {_KNOB}:
            findings.append(f"{paint.where}: a knob filled with {sorted(own)}, not {_KNOB}")
        states = [fills for fills in (_fills_of(track, dimmed=False) if track else [])]
        if not states or any(not fills or not fills <= set(_TRACKS) for fills in states):
            findings.append(f"{paint.where}: the knob's track is {[sorted(s) for s in states] or 'unpainted'}")
    return knobs, sorted(set(findings))


_SWITCH_SAMPLE = {
    f"{_SRC}/lib/sample/Switches.svelte": (
        "<button style=\"background-color: {on ? 'var(--oo-success)' : 'var(--oo-bg-overlay)'};\">"
        "<span style=\"background-color: var(--oo-toggle-knob);\"></span></button>\n"
        "<button style=\"background-color: {on ? 'var(--oo-acc-600)' : 'var(--oo-bg-overlay)'};\">"
        "<span style=\"background-color: var(--oo-fg-on-accent);\"></span></button>\n"
        "<button style={track(v)}><span style=\"background-color: var(--oo-toggle-knob);\"></span></button>\n"
        "<button style=\"background-color: {on ? 'var(--oo-switch-on)' : 'var(--oo-switch-off)'};\">"
        "<span class=\"bg-[var(--oo-toggle-knob)]\"></span></button>\n"
        "<script>function track(v) { if (v === null) return 'background-color: var(--oo-bg-overlay); opacity: 0.5;';"
        " return v ? 'background-color: var(--oo-switch-on);' : 'background-color: var(--oo-switch-off);'; }</script>\n"
    ),
}


def test_ds23_a_switch_knob_sits_on_the_switch_tracks_and_a_field_edge_is_seen():
    paints, _ = _paints(_SWITCH_SAMPLE, _tailwind_colours())
    knobs, caught = _switch_findings(paints)
    lines = sorted({int(m.group(1)) for f in caught for m in [re.search(r"Switches\.svelte:(\d+)", f)] if m})
    assert knobs == 4 and lines == [1, 2], (
        "the census finds each knob, by its token or as the one child of a changing track, and "
        f"names the tracks and knobs off the switch tokens; a dimmed unknown state is inactive: {knobs}, {caught}"
    )

    paints, _ = _tree_paints()
    knobs, findings = _switch_findings(paints)
    assert knobs >= 8, f"the census finds the switches drawn by hand: {knobs}"
    assert not findings, "switches off the switch tokens:\n  " + "\n  ".join(findings)

    switch = {selector.strip(): dict(_declarations(body)) for selector, body, _ in _rules(_css_of(_SWITCH, read(_SWITCH)))}
    assert "var(--oo-switch-off)" in switch.get(".oo-switch", {}).get("background-color", "") and (
        "var(--oo-switch-on)" in switch.get(".oo-switch[aria-checked='true']", {}).get("background-color", "")
    ) and switch.get(".oo-switch-knob", {}).get("background-color") == f"var({_KNOB})", (
        "the switch primitive draws its tracks and its knob in the switch tokens"
    )

    # Every edge a field's rule draws (a rule for another state, the reduced
    # motion one, may draw none), and each primitive draws one.
    unseen, drawn = [], set()
    for path in _FIELDS:
        for selector, body, context in _rules(_css_of(path, read(path))):
            if re.fullmatch(r"\.oo-field-control", selector.strip()):
                for name, value in _declarations(body):
                    if name in ("border", "border-color"):
                        if "var(--oo-input-bd)" in value:
                            drawn.add(path)
                        else:
                            unseen.append(f"{path} {context}: {name}: {value}")
    assert not unseen and drawn == set(_FIELDS), (
        f"the field primitives draw their edge in the field token, a boundary that must be seen: {unseen}, drawn by {sorted(drawn)}"
    )


# ===========================================================================
# DS24 -- a legacy binary theme equal to the system is retired at startup
# ===========================================================================
def test_ds24_a_legacy_theme_equal_to_the_system_is_retired_at_startup():
    scenarios = [
        # Stored by the old layout on the last visit, equal to the system:
        # Match system, retired, and the system is followed.
        {"stored": {"oo-theme": "dark"}, "media": _MEDIA["dark"], "steps": [{"system": {"scheme": "light"}}]},
        {"stored": {"oo-theme": "light"}, "media": _MEDIA["light"], "steps": [{"system": {"scheme": "dark"}}]},
        # Unlike the system: the pin it means is kept, and the key with it.
        {"stored": {"oo-theme": "dark"}, "media": _MEDIA["light"], "steps": [{"system": {"scheme": "dark"}}]},
        # A stored palette decides; the old key is read by nothing and retired.
        {"stored": {"oo-palette": "day", "oo-theme": "dark"}, "media": _MEDIA["dark"], "steps": []},
        # Storage that throws: nothing retired, nothing thrown.
        {"stored": {"oo-theme": "dark"}, "media": _MEDIA["dark"], "steps": [], "blocked": True},
    ]
    out = _node("retire", {"scenarios": scenarios})
    assert [r["threw"] for r in out] == [None] * len(scenarios), [r["threw"] for r in out]
    assert out[0]["seen"] == ["night", "day"] and out[1]["seen"] == ["day", "night"], (
        f"an old binary theme equal to the system means Match system, and the system is followed after it: "
        f"{out[0]['seen']}, {out[1]['seen']}"
    )
    assert out[0]["left"] == {} and out[1]["left"] == {}, (
        f"the old key is removed once it said Match system: {out[0]['left']}, {out[1]['left']}"
    )
    assert out[2]["seen"] == ["night", "night"] and out[2]["left"] == {"oo-theme": "dark"}, (
        f"an old binary theme unlike the system is the pin it means, and it stays: {out[2]}"
    )
    assert out[3]["left"] == {"oo-palette": "day"}, f"under a stored palette the old key is retired: {out[3]}"
    assert all(not [w for w in r["writes"] if w[1] is not None] for r in out), (
        f"retiring removes a key and stores nothing: {[r['writes'] for r in out]}"
    )

    preferences = read(_PREFERENCES)
    body = _function_body(preferences, "initPreferences") or ""
    retire = body.find("retireLegacyTheme(")
    follow = body.find("followSystem(")
    assert _imports(preferences, "retireLegacyTheme", "$lib/theme/apply") and 0 <= retire < follow, (
        "the preferences store retires the old key at startup, through apply.ts, before it follows the system"
    )

    # An older theme that still pins is returned as the choice it means, and
    # the store holds it for the visit, where a choice is read before
    # storage: the system's changes then leave the choice shown alone.
    assert [r["pin"] for r in out] == [None, None, "night", None, None], (
        f"the pin an older theme unlike the system means is returned, and nothing else: {[r['pin'] for r in out]}"
    )
    held = re.search(r"(?:const|let)\s+(\w+)\s*=\s*retireLegacyTheme\(", body)
    assert held and re.search(r"chosen\.set\(\s*PALETTE_KEY\s*,\s*" + held.group(1) + r"\s*\)", body), (
        "the preferences store holds an older theme's pin for the visit, in the choices it reads before storage"
    )


# ===========================================================================
# DS25 -- the derivation layer maps every app token as the design's table says
# ===========================================================================
# What each app token is, as an expression of roles. A token is compared by
# the colour it takes in each palette, so an alias through another token
# answers as the role it reaches. The accent fill's hover is one colour per
# scheme: mixed toward the surface under a light one, toward the text under
# a dark one.
def _mix(a, p, b):
    return f"color-mix(in srgb, var(--oo-role-{a}) {p}%, {b if b == 'transparent' else f'var(--oo-role-{b})'})"


_HOVER_BY_SCHEME = {"light": _mix("primary", 80, "surface"), "dark": _mix("primary", 85, "text")}
_DERIVATION_TABLE = {
    # Foundation
    "--oo-bg-base": "bg", "--oo-bg-surface": "surface", "--oo-bg-elevated": "surface",
    "--oo-bg-overlay": "surface-2", "--oo-bg-subtle": "sunken", "--oo-bg-sunken": "sunken",
    "--oo-bg-hover": _mix("text", 4, "transparent"), "--oo-bg-active": _mix("text", 4, "transparent"),
    "--oo-fg-primary": "text", "--oo-fg-secondary": "text-muted", "--oo-fg-tertiary": "text-muted",
    "--oo-fg-muted": "text-muted", "--oo-fg-decorative": "rule-soft", "--oo-fg-on-accent": "on-primary",
    "--oo-acc-50": "on-primary", "--oo-bd-subtle": "rule-soft", "--oo-bd-default": "rule-soft",
    "--oo-bd-strong": "rule",
    **{f"--oo-acc-{step}": "tint-1" for step in (100, 200, 800, 900)},
    **{f"--oo-acc-{step}": "primary-ink" for step in (300, 400, 500, 700)},
    "--oo-acc-600": _HOVER_BY_SCHEME,
    "--oo-success": "success-ink", "--oo-warning": "warn-ink", "--oo-error": "stop-ink", "--oo-info": "syn-keyword",
    "--oo-success-bg": _mix("success-ink", 12, "transparent"), "--oo-success-bd": _mix("success-ink", 40, "transparent"),
    "--oo-warning-bg": _mix("warn-tone", 14, "transparent"), "--oo-warning-bd": _mix("warn-tone", 45, "transparent"),
    "--oo-error-bg": _mix("stop-ink", 12, "transparent"), "--oo-error-bd": _mix("stop-ink", 40, "transparent"),
    "--oo-info-bg": _mix("syn-keyword", 12, "transparent"), "--oo-info-bd": _mix("syn-keyword", 40, "transparent"),
    # New tokens
    "--oo-acc-fill": "primary", "--oo-acc-fill-hover": _HOVER_BY_SCHEME, "--oo-acc-ink": "primary-ink",
    "--oo-acc-ink-2": "secondary-ink", "--oo-switch-on": "primary-ink", "--oo-switch-off": "rule",
    "--oo-toggle-knob": "surface", "--oo-bg-tint-1": "tint-1", "--oo-bg-tint-2": "tint-2", "--oo-bg-tint-3": "tint-3",
    "--oo-bg-code": "code-bg", "--oo-code-keyword": "syn-keyword", "--oo-code-name": "syn-name",
    "--oo-fg-stop": "stop-ink", "--oo-status-ok": "status-ok", "--oo-shadow-c1": "shadow-1",
    "--oo-shadow-c2": "shadow-2", "--oo-edge": "edge", "--oo-focus-ink": "primary-ink", "--oo-pet-ground": "tint-2",
    # The marks: a fill that carries no text reads an ink, which reaches 3:1
    "--oo-acc-mark": "primary-ink", "--oo-acc-mark-2": "secondary-ink",
    # Component layer
    "--oo-fg-faint": "text-muted", "--oo-fg-on-semantic": "surface",
    **{f"--oo-{name}": "primary-ink" for name in ("accent", "accent-primary", "accent-secondary", "tobacco", "sage")},
    "--oo-sidebar-bg": "bg", "--oo-header-bg": "surface", "--oo-panel-bg": "surface", "--oo-input-bg": "surface",
    "--oo-input-bd": "rule", "--oo-msg-user-bg": "sunken", "--oo-msg-bot-bg": "transparent",
    "--oo-btn-primary-bg": "primary", "--oo-btn-secondary-bg": "surface-2", "--oo-btn-ghost-bg": "transparent",
    "--oo-surface-600": "sunken", "--oo-surface-700": "surface-2", "--oo-surface-800": "surface",
    "--oo-qr-bg": "#FFFFFF",
}
# The two text levels: every name for body or secondary text is one of them.
_TEXT_LEVEL = re.compile(r"^--oo-(?:fg|text)-(?:primary|secondary|tertiary|muted|faint)$")


def _expected_expression(spec, scheme):
    if isinstance(spec, dict):
        return spec[scheme]
    if spec.startswith(("color-mix(", "#")) or spec == "transparent":
        return spec
    return f"var(--oo-role-{spec})"


def _derivation_differences(tokens, palettes, table):
    """Every token whose colour differs from the table's in a palette, or
    that the derivation layer does not declare."""
    out = []
    for pid, (roles, scheme) in palettes.items():
        resolver = _colour_resolver(tokens, roles, scheme, pid)
        for token, spec in table.items():
            if token not in _with_palette(tokens, pid):
                out.append(f"{token} is not declared in the derivation layer for {pid}")
                continue
            try:
                ours = resolver.colour(token)
                wanted = resolver.colour(_expected_expression(spec, scheme))
            except (KeyError, ValueError) as exc:
                out.append(f"{pid}: {token} cannot be evaluated ({exc})")
                continue
            if any(abs(a - b) > 0.5 for a, b in zip(ours[:3], wanted[:3])) or abs(ours[3] - wanted[3]) > 0.005:
                out.append(f"{pid}: {token} is {tuple(round(v, 1) for v in ours)}, not {spec}")
    return sorted(set(out))


def test_ds25_the_derivation_layer_maps_every_token_as_the_design_says():
    tokens, palettes = _derivation(), _palettes()
    planted = {**tokens, "--oo-fg-faint": "var(--oo-role-rule-soft)", "--oo-acc-600": "var(--oo-role-primary)"}
    caught = _derivation_differences(planted, palettes, _DERIVATION_TABLE)
    assert any("--oo-fg-faint" in c for c in caught) and any("--oo-acc-600" in c for c in caught), (
        f"the comparison catches a text level mapped to a quiet rule and a hover that is the fill itself: {caught}"
    )
    differences = _derivation_differences(tokens, palettes, _DERIVATION_TABLE)
    assert not differences, "the derivation layer differs from the design's table:\n  " + "\n  ".join(differences)

    levels = sorted(name for name in tokens if _TEXT_LEVEL.match(name))
    assert len(levels) >= 8, f"the census finds the names of the text levels: {levels}"
    off = []
    for pid, (roles, scheme) in palettes.items():
        resolver = _colour_resolver(tokens, roles, scheme, pid)
        allowed = [resolver.colour("var(--oo-role-text)"), resolver.colour("var(--oo-role-text-muted)")]
        for name in levels:
            if resolver.colour(name) not in allowed:
                off.append(f"{pid}: {name}")
    assert not off, f"every name of a text level is the text or the muted text, two levels: {off}"


# ===========================================================================
# DS26 -- no colour literal but the ledgered rgb debt; no named colour
# ===========================================================================
_TW_DEFAULT_PALETTE = re.compile(
    r"(?<![\w-])(?:[\w-]+:)*(?:bg|text|border(?:-[trblxyse])?|ring|ring-offset|fill|stroke|outline|divide|from|via"
    r"|to|placeholder|decoration|caret|accent|shadow)-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow"
    r"|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}(?:/\d+)?(?![\w-])"
)
_SYSTEM_COLOURS = frozenset(
    "canvas canvastext linktext visitedtext activetext buttonface buttontext buttonborder field fieldtext "
    "highlight highlighttext selecteditem selecteditemtext mark marktext graytext accentcolor accentcolortext".split()
)


def _named_colour_words(value):
    """The CSS named colours a value spells (a fallback inside ``var()`` included)."""
    return [w for w in _WORD.findall(_VAR.sub("var(", value)) if w.lower() in _NAMED_COLOURS]


def _literal_findings(sources):
    """``{path: [finding]}``: colour functions other than rgb (and rgb outside
    components, where UR5 holds it), named colours in a colour value
    (system colours inside a forced-colours block aside), and Tailwind's
    own palette in a class."""
    out = {}
    for path, text in sources.items():
        found = []
        body = _CSS_COMMENT.sub(" ", text)
        for match in _COLOUR_FUNCTION.finditer(body):
            name = match.group(0)[:-1].lower()
            if name in ("rgb", "rgba") and path.endswith(".svelte"):
                continue
            found.append(match.group(0))
        for selector, block, context in _rules(_css_of(path, text)):
            forced = any(re.fullmatch(r"@media\s*\(\s*forced-colors\s*:\s*active\s*\)", c) for c in context)
            for name, value in _declarations(block):
                if not (_COLOUR_PROPERTY.match(name) or name.startswith("--")):
                    continue
                for word in _named_colour_words(value):
                    if word.lower() in ("white", "black"):
                        continue
                    if forced and word.lower() in _SYSTEM_COLOURS:
                        continue
                    found.append(f"{name}: {word}")
        if path.endswith(".svelte"):
            markup = _STYLE_BLOCK.sub("", text)
            for style in re.findall(r"""\sstyle\s*=\s*(["'])(.*?)\1""", markup, re.S):
                for name, value in _declarations(style[1]):
                    if _COLOUR_PROPERTY.match(name):
                        found += [f"style {name}: {w}" for w in _named_colour_words(value)
                                  if w.lower() not in ("white", "black")]
        found += _TW_DEFAULT_PALETTE.findall(text)
        if found:
            out[path] = found
    return out


def test_ds26_no_colour_literal_but_the_ledgered_rgb_debt_and_no_named_colour():
    sample = {
        f"{_SRC}/lib/sample/a.css": (
            ".a { color: crimson; background: oklch(0.5 0.1 30); border-color: hsl(20 40% 50%); }\n"
            ".b { color: var(--oo-fg-muted, gray); box-shadow: 0 0 0 1px rgba(0, 0, 0, .2); }\n"
            "@media (forced-colors: active) { .c { border-color: CanvasText; } }\n"
        ),
        f"{_SRC}/lib/sample/B.svelte": (
            "<div class=\"bg-gray-400 text-rose-600/50\" style=\"color: steelblue\"></div>\n"
            "<style>.d { background: rgba(0, 0, 0, .1); color: white; }</style>\n"
        ),
    }
    caught = _literal_findings(sample)
    assert sorted(len(v) for v in caught.values()) == [3, 5], (
        "the census reads colour functions, named colours (a fallback included), rgb outside components "
        f"and Tailwind's own palette, and not a system colour kept for forced colours nor white or black: {caught}"
    )
    sources = _sources(_EVERY, exclude=tuple(_PALETTE_FILES.values()))
    found = _literal_findings(sources)
    assert not found, (
        f"colour literals outside the palette files and the ledgered rgb debt: "
        f"{sum(len(v) for v in found.values())} in {len(found)} files: {json.dumps(found)}"
    )


# ===========================================================================
# DS27 -- the base layer Tailwind generates draws in tokens
# ===========================================================================
_BASE_LAYER_DRIVER = r"""
import { createRequire } from 'node:module';
const clause = process.argv[2];
const require = createRequire(process.env.OO_TAILWIND);
const postcss = require('postcss');
const tailwind = require('tailwindcss');
const config = (await import(process.env.OO_TAILWIND)).default;
const out = await postcss([tailwind({ ...config, content: [{ raw: '', extension: 'html' }] })])
    .process('@tailwind base;', { from: undefined });
console.log('RESULT ' + JSON.stringify(out.css));
console.log('PASS ' + clause);
"""
_NO_HUE = re.compile(r"^(?:#0000|#00000000|transparent)$", re.I)


def _base_layer_literals(css):
    """``[(property, literal)]`` of the colours the base layer writes that no
    role gives, a fully transparent colour aside."""
    found = []
    for _, body, _ in _rules(css):
        for name, value in _declarations(body):
            for literal in _colour_literals(value):
                if not _NO_HUE.match(literal):
                    found.append((name, literal))
    return found


def test_ds27_the_base_layer_tailwind_generates_draws_in_tokens():
    sample = "*, ::before { border-color: #e5e7eb; --tw-shadow: 0 0 #0000; } input::placeholder { color: rgb(1 2 3); }"
    assert _base_layer_literals(sample) == [("border-color", "#e5e7eb"), ("color", "rgb(")], (
        f"the census reads a literal colour and passes a fully transparent one: {_base_layer_literals(sample)}"
    )
    if not (REPO / "frontend/node_modules/tailwindcss").exists():
        pytest.skip("OWED: frontend/node_modules absent, the base layer cannot be generated")
    out = run_ts({"OO_TAILWIND": _TAILWIND}, _BASE_LAYER_DRIVER, "base")
    css = json.loads([line for line in out.splitlines() if line.startswith("RESULT ")][0][len("RESULT "):])
    assert "::placeholder" in css and "border-color" in css, "the base layer was generated"
    found = _base_layer_literals(css)
    assert not found, f"the base layer Tailwind generates writes colours no role gives: {found}"
    placeholder = [dict(_declarations(body)).get("color") for selector, body, _ in _rules(css)
                   if "input::placeholder" in selector]
    assert placeholder and all(v and "var(--oo-fg-muted)" in v for v in placeholder), (
        f"a field's placeholder is the muted text: {placeholder}"
    )


# ===========================================================================
# DS10's census over every name of the accent's inks, and DS19's over every spelling
# ===========================================================================
# The tokens whose value is an ink but whose job is to fill a mark that
# carries no text: the switch's track when on (the knob is under 3:1 on the
# fill in day), and the marks.
_SANCTIONED_INK_FILLS = ("--oo-switch-on",) + _MARK_TOKENS
_FILL_PROPS = frozenset({"background", "background-color", "background-image", "fill", "stop-color", "flood-color"})


def _resolves_to(tokens, name, roles, seen=()):
    ref = re.fullmatch(r"var\(\s*(--[\w-]+)\s*\)", tokens.get(name, "").strip())
    if not ref or ref.group(1) in seen:
        return False
    if ref.group(1).startswith(_ROLE_PREFIX):
        return ref.group(1)[len(_ROLE_PREFIX):] in roles
    return _resolves_to(tokens, ref.group(1), roles, seen + (ref.group(1),))


def _accent_inks(tokens):
    """Every name the derivation layer gives the accent's inks (the primary
    and the secondary ink), but the sanctioned mark fills and the chart,
    category and pipeline colours, which fill marks by design."""
    return sorted(
        name for name in tokens
        if _resolves_to(tokens, name, ("primary-ink", "secondary-ink"))
        and name not in _SANCTIONED_INK_FILLS and not name.startswith(_CHART_PREFIXES)
    )


def _ink_attribution(sources, paints, inks):
    """``(fills, unattributable)``: every read of an ink name placed in the
    property it lands in -- through the paints' expansions (a conditional, a
    function, a constant, an import), a custom property followed to where it
    is read, or the text around it -- the fills, and the reads no property
    was found for."""
    read = re.compile(
        r"var\(\s*(" + "|".join(re.escape(i) for i in sorted(inks, key=len, reverse=True)) + r")(?![\w-])"
    )
    traced, custom = {}, {}
    for paint in paints:
        for alternative in paint.alternatives:
            for _, prop, value, origin in alternative:
                for ref in _reads(value):
                    custom.setdefault((paint.path, ref), set()).add(prop)
                if origin is None:
                    continue
                pieces, start, end = origin
                for match in read.finditer(_joined(pieces), start, end):
                    where = _origin(pieces, match.start())
                    if where:
                        traced.setdefault(where, set()).add(prop)
    fills, unattributable = [], []
    for path, text in sources.items():
        for match in read.finditer(text):
            props = set(traced.get((path, match.start()), ()))
            if not props:
                kind, name = _context(text, match.start())
                if kind == "utility":
                    props = {"background-color" if name in _FILL_UTILITIES else "color"}
                elif kind in ("property", "attribute"):
                    props = {name}
            landed = set()
            for prop in props:
                if prop.startswith("--") and not prop.startswith("--oo-"):
                    landed |= custom.get((path, prop), set()) or {prop}
                else:
                    landed.add(prop)
            label = f"{path}:{text.count(chr(10), 0, match.start()) + 1} {match.group(1)}"
            if landed & _FILL_PROPS:
                fills.append(f"{label} fills ({', '.join(sorted(landed))})")
            elif not landed or not all(_INK_PROPERTY.match(p) or p in inks for p in landed):
                unattributable.append(f"{label} lands in {sorted(landed) or 'nothing the census can place'}")
    return fills, unattributable


def _on_accent_misplaced(paints, fills, inks):
    """Every place the on-accent ink fills something, or is the text on a
    ground (its own, or its variant's) other than the accent fill. Text that
    sets no ground of its own reads its parent's, which the census does not
    follow."""
    out = []
    for paint in paints:
        for alternative in paint.alternatives:
            slots = _grounds_of(alternative)
            base = slots.get("", {})
            for variant, slot in slots.items():
                if _reads(slot.get("bg", "")) & inks:
                    out.append(f"{paint.where} {variant}: the on-accent ink fills {slot['bg']}")
                if _reads(slot.get("fg", "")) & inks:
                    ground = slot.get("bg", base.get("bg", ""))
                    if ground and not (_reads(ground) & fills):
                        out.append(f"{paint.where} {variant}: the on-accent ink on {ground}")
    return sorted(set(out))


def _accent_fill_pairs(paints, fills, inks):
    """``(fills found, [unpaired])``: every opaque fill with the accent fill
    (a wash mixed with transparent is a tint, held by the pair census) on
    something that may hold text, and those whose text is not the on-accent
    ink; a mark (an element holding nothing) is the mark census's."""
    found, unpaired = 0, []
    for paint in paints:
        if paint.element is not None and paint.element.empty:
            continue
        for alternative in paint.alternatives:
            slots = _grounds_of(alternative)
            base = slots.get("", {})
            for variant, slot in slots.items():
                ground = slot.get("bg", "")
                if not (_reads(ground) & fills) or "transparent" in ground:
                    continue
                found += 1
                ink = slot.get("fg", base.get("fg", ""))
                if not (_reads(ink) & inks):
                    unpaired.append(f"{paint.where} {variant}: {ground} with {ink or 'no text colour'}")
    return found, sorted(set(unpaired))



# ===========================================================================
# DS28 -- the browser's computed colours are recorded, or owed
# ===========================================================================
def test_ds28_the_browser_computed_colours_are_recorded_or_owed():
    computed = REPO / _COMPUTED
    if not computed.is_file():
        pytest.skip(
            f"OWED: browser-computed colours ({_COMPUTED}) are recorded on the machine; "
            "the evaluator is compared with them once they are"
        )
    recorded = json.loads(computed.read_text(encoding="utf-8"))
    covered = {entry.get("palette") for entry in recorded}
    assert covered == set(_PALETTE_IDS), f"the recorded browser values cover each palette: {sorted(covered)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
