#!/usr/bin/env python3
"""Contracts for the surface rules the rebuilt files are held to.

The rebuilt interface separates by tone and space, not by lines; it is
matte; its labels are in sentence case; and a selected thing never says so by
its colour alone. The rules hold the files ``HELD`` names
(``tests/_frontend.py``): a file joins when it is rebuilt, the list only
grows, and a directory entry holds every file written under it later. The
day and night palettes draw the edge token transparent, so the edge costs
nothing there; high contrast and forced colours draw it, and a tone that
separates nothing on a dark ground is then still bounded.

Every census reads sources as text (a component's style block, its markup's
class and style attributes, a stylesheet) and carries a standing positive
fixture: a sample it must find fault with, so a census gone blind turns red
instead of reading a clean zero. Every finding of one census is reported in
one failure.

  * TN1 -- every element whose ground is a tone (the surface, the second
    surface that sheets and popovers sit on, the sunken ground, a tint, the
    code ground, the accent fill and its hover) carries the edge,
    ``border: 1px solid var(--oo-edge)``: by its own rule, by a rule it
    refines (the same element less qualified: ``.oo-btn`` under
    ``.oo-btn[data-variant='primary']:hover``), or by the ``oo-tone`` class,
    which a stylesheet declares with that border. A rule inside an at-rule
    refines the rules outside it, never the other way, so an edge drawn only
    under a media query does not bound the ground outside it; a more
    qualified rule that resets the border is read as the state it makes. A
    ground is read in a background colour, a background image, a Tailwind
    class (an arbitrary value that mixes a ground included) and a class
    list a script holds. A field listed here carries its field edge, the
    required boundary, instead. The switch's knob is not a ground: it is a
    mark on its track.
  * TN2 -- the border tokens (``--oo-bd-*``, and every alias that carries
    one) are read only by the selectors listed here: the composer's outline,
    a field, a focus ring. A Tailwind border or divider drawn in them, or in
    Tailwind's default border colour, is a read.
  * TN3 -- no capitals and no wide tracking: no ``uppercase``, title case
    (``capitalize``) or small capitals, no ``tracking-wide*`` class, no
    positive ``letter-spacing``.
  * TN4 -- matte: no gradient but the ``currentColor 0 0`` drawing idiom, no
    inset shadow, and every shadow one of the shadow tokens (or a focus ring:
    a spread in a token colour, with no offset and no blur); no text shadow
    and no drop shadow.
  * TN5 -- selection is never shown by fill alone: every element that
    carries ``aria-pressed`` draws a check in its pressed branch, inside
    the element itself, and one held file does; an ``aria-pressed`` given
    through a spread is refused, since its check cannot be read; every rule
    on ``aria-current`` sets a weight of 600; every rule that fills an
    ``aria-selected`` element sets a weight of 600 too.
  * TN6 -- ``HELD`` only grows, compared with its value at ``HEAD``, and
    every entry on it names files of the tree.
  * TN7 -- focus stays visible: no rule removes an element's outline (nor
    a class the ``outline-none`` utility) without drawing a ring in its
    place; and where a list keeps its focus on the field and points
    ``aria-activedescendant`` at an item, the class that marks the active
    item draws an outline or a ring in the focus ink, not a wash alone.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _frontend  # noqa: E402
from _frontend import REPO, files, read  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder from the junit file.
BUDGET_S = {
    "test_tn1_every_toned_ground_carries_the_edge": 1.0,
    "test_tn2_border_tokens_are_read_only_by_the_listed_selectors": 1.0,
    "test_tn3_no_capitals_and_no_wide_tracking": 1.0,
    "test_tn4_matte_no_gradient_no_inset_and_only_the_shadow_tokens": 1.0,
    "test_tn5_selection_is_never_shown_by_fill_alone": 1.0,
    "test_tn6_held_only_grows_and_names_files_of_the_tree": 1.0,
    "test_tn7_focus_stays_visible[outline]": 1.0,
    "test_tn7_focus_stays_visible[active]": 1.0,
}

_SRC = "frontend/src"
_SAMPLE_PATH = f"{_SRC}/lib/sample/Sample.svelte"
_THEME = f"{_SRC}/styles/theme.css"
_GLOBAL_SHEETS = (f"{_SRC}/app.css",)
_STYLES_DIR = f"{_SRC}/styles"

# The fields whose boundary is the field edge (the required rule), not the
# tone edge: {path: selectors}. Each must stand in its file.
_FIELD_EDGES = {
    f"{_SRC}/lib/ds/Input.svelte": (".oo-field-control",),
    f"{_SRC}/lib/ds/Select.svelte": (".oo-field-control",),
}
_FIELD_EDGE = "var(--oo-input-bd)"
# A field's boundary in a state: in focus it is drawn in the focus ink (or
# the accent the field primitives draw it in), when invalid in the error
# ink. Only a rule qualified by that state may draw it; at rest the field
# edge holds.
_FIELD_STATE_EDGES = (
    (":focus", ("var(--oo-acc-500)", "var(--oo-focus-ink)")),
    ("[aria-invalid", ("var(--oo-error)",)),
)

# The selectors that may read a border token: the composer's outline, a
# field, a focus ring. {path: selectors}. Each must stand in its file.
_BORDER_READERS = {}

# The roles a toned ground is drawn in: the surface, the second surface
# (sheets and popovers), the sunken ground, the tints, the code ground and
# the accent fill.
_GROUND_ROLES = frozenset({
    "surface", "surface-2", "sunken", "tint-1", "tint-2", "tint-3", "code-bg", "primary",
})
# Tokens drawn in a ground role that are not grounds, each with its reason.
_NOT_GROUNDS = {
    "--oo-toggle-knob": "the switch's knob is a mark on its track (the switch contract holds it)",
    "--oo-fg-on-semantic": "the ink on a status fill, never a ground",
    "--oo-acc-50": "the ink on the accent fill, never a ground",
    "--oo-scrim": "the veil behind a dialog covers the page and bounds nothing",
}


# ---------------------------------------------------------------------------
# Reading CSS
# ---------------------------------------------------------------------------
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>(.*?)</style>", re.S)
_SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>(.*?)</script>", re.S)


def _css_of(path, text):
    """The CSS a source holds: the file itself, or a component's style blocks."""
    if path.endswith((".css", ".scss")):
        return text
    if path.endswith(".svelte"):
        return "\n".join(_STYLE_BLOCK.findall(text))
    return ""


def _markup_of(path, text):
    """A component's markup: the file without its script and style blocks
    and its comments."""
    if not path.endswith(".svelte"):
        return ""
    return _HTML_COMMENT.sub(" ", _STYLE_BLOCK.sub(" ", _SCRIPT_BLOCK.sub(" ", text)))


def _skip_string(text, index):
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
    """``[(selector, body, context)]`` for every rule of ``css``; an
    at-block's rules carry its prelude in ``context``."""
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
    """``[(property, value)]`` of a rule body, split at top-level semicolons."""
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
                out.append((name.strip().lower(), " ".join(value.split())))
            start = index + 1
        index += 1
    return out


def _split_top(value, separator=","):
    """``value`` split at ``separator`` outside parentheses."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return [part.strip() for part in parts]


def _words(value):
    """``value`` split at whitespace outside parentheses."""
    words, depth, current = [], 0, ""
    for char in value:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char.isspace() and depth == 0:
            if current:
                words.append(current)
            current = ""
        else:
            current += char
    if current:
        words.append(current)
    return words


# ---------------------------------------------------------------------------
# The grounds, from the derivation layer
# ---------------------------------------------------------------------------
def _token_table(css):
    table = {}
    for _, body, _ in _rules(css):
        for name, value in _declarations(body):
            if name.startswith("--oo-"):
                table.setdefault(name, []).append(value)
    return table


def _value_roles(value, table, seen):
    """The roles a token's value is drawn in: a role, a token followed, or
    the operand of a ``color-mix`` that holds the larger share."""
    value = value.strip()
    match = re.fullmatch(r"var\(\s*(--[\w-]+)\s*\)", value)
    if match:
        ref = match.group(1)
        if ref.startswith("--oo-role-"):
            return {ref[len("--oo-role-"):]}
        if ref in seen:
            return set()
        roles = set()
        for inner in table.get(ref, ()):
            roles |= _value_roles(inner, table, seen | {ref})
        return roles
    match = re.fullmatch(r"color-mix\(\s*in\s+[\w-]+\s*,(.*)\)", value, re.S)
    if not match:
        return set()
    operands = []
    for part in _split_top(match.group(1)):
        share = re.fullmatch(r"(.*?)\s+(\d+(?:\.\d+)?)%", part, re.S)
        operands.append((share.group(1), float(share.group(2))) if share else (part, None))
    if len(operands) != 2:
        return set()
    (a, pa), (b, pb) = operands
    if pa is None and pb is None:
        pa = pb = 50.0
    elif pa is None:
        pa = 100.0 - pb
    elif pb is None:
        pb = 100.0 - pa
    if pa == pb:
        return set()
    return _value_roles(a if pa > pb else b, table, seen)


def _grounds(css):
    """The app tokens drawn in a ground role (the derivation layer's
    names), without the ones that are not grounds."""
    table = _token_table(css)
    return {
        name for name in table
        if name not in _NOT_GROUNDS
        and set().union(*(_value_roles(v, table, {name}) for v in table[name])) & _GROUND_ROLES
    }


# ---------------------------------------------------------------------------
# Reading markup: tags, classes, styles
# ---------------------------------------------------------------------------
_STRING = re.compile(r"""'([^'\\\n]*)'|"([^"\\\n]*)"|`([^`\\$\n]*)`""")


def _tag_end(text, start):
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


def _attributes(text):
    """``[(name, value)]`` of a tag's attribute text; a value keeps its
    quotes or braces, a bare attribute's value is None."""
    out, index = [], 0
    while index < len(text):
        char = text[index]
        if char.isspace() or char == "/":
            index += 1
            continue
        if char == "{":
            depth, cursor = 0, index
            while cursor < len(text):
                depth += text[cursor] == "{"
                depth -= text[cursor] == "}"
                cursor += 1
                if depth == 0:
                    break
            out.append(("{spread}", text[index:cursor]))
            index = cursor
            continue
        match = re.match(r"[^\s=/>{}\"']+", text[index:])
        if not match:
            index += 1
            continue
        name = match.group(0)
        index += len(name)
        rest = text[index:]
        eq = re.match(r"\s*=\s*", rest)
        if not eq:
            out.append((name, None))
            continue
        index += eq.end()
        if index >= len(text):
            out.append((name, ""))
            break
        opener = text[index]
        if opener in "\"'":
            cursor, depth = index + 1, 0
            while cursor < len(text):
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                elif text[cursor] == opener and depth == 0:
                    break
                cursor += 1
            out.append((name, text[index:cursor + 1]))
            index = cursor + 1
        elif opener == "{":
            depth, cursor = 0, index
            while cursor < len(text):
                depth += text[cursor] == "{"
                depth -= text[cursor] == "}"
                cursor += 1
                if depth == 0:
                    break
            out.append((name, text[index:cursor]))
            index = cursor
        else:
            bare = re.match(r"[^\s>]+", text[index:])
            out.append((name, bare.group(0)))
            index += bare.end()
    return out


def _tags(markup):
    """``(name, attributes)`` of every element tag in a component's markup."""
    index = 0
    while True:
        start = markup.find("<", index)
        if start < 0:
            return
        match = re.match(r"<([A-Za-z][\w.:-]*)", markup[start:])
        if not match:
            index = start + 1
            continue
        end = _tag_end(markup, start)
        yield match.group(1), _attributes(markup[start + len(match.group(0)):end - 1])
        index = end


def _unquote(value):
    if value and value[0] in "\"'" and value[-1:] == value[0]:
        return value[1:-1]
    return value or ""


def _script_strings(scripts, expression):
    """The string literals of the script values an expression names: the
    initializer of ``const|let|var <name> = ...`` for each identifier in it."""
    found = []
    for ident in set(re.findall(r"(?<![\w$.])([A-Za-z_$][\w$]*)", _STRING.sub(" ", expression))):
        match = re.search(r"(?:const|let|var)\s+" + re.escape(ident) + r"\b[^=;]*=", scripts)
        if not match:
            continue
        depth, index = 0, match.end()
        while index < len(scripts):
            char = scripts[index]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char in ";\n" and depth <= 0:
                break
            index += 1
        found += [a or b or c for a, b, c in _STRING.findall(scripts[match.end():index])]
    return found


def _class_tokens(attributes, scripts=""):
    """Every class an element may carry: the static tokens, the strings of
    an expression in the class value (and of a script value it names), and
    each ``class:`` directive."""
    tokens = set()
    for name, value in attributes:
        if name == "class":
            raw = _unquote(value)
            if raw.startswith("{") and value and value[0] == "{":
                expressions = [raw[1:-1]]
                raw = ""
            else:
                expressions = re.findall(r"\{[^{}]*\}", raw)
            for expression in expressions:
                tokens.update(t for parts in _STRING.findall(expression) for p in parts for t in p.split())
                tokens.update(t for s in _script_strings(scripts, expression) for t in s.split())
            tokens.update(re.sub(r"\{[^{}]*\}", " ", raw).split())
        elif name.startswith("class:"):
            tokens.add(name[len("class:"):])
    return tokens


def _scripts_of(path, text):
    if path.endswith(".svelte"):
        return "\n".join(_SCRIPT_BLOCK.findall(text))
    return ""


def _surfaces(path, text):
    """What a file draws with, as one text: its CSS (comments removed), each
    element's classes and inline style, and the string literals of its
    scripts (a class list a script builds). Prose in a comment is not a
    surface."""
    parts = [_CSS_COMMENT.sub(" ", _css_of(path, text))]
    for _, attributes in _tags(_markup_of(path, text)):
        parts.append(" ".join(sorted(_class_tokens(attributes))))
        parts.append(_style_text(attributes))
    if path.endswith(".svelte"):
        scripts = "\n".join(_SCRIPT_BLOCK.findall(text))
    elif path.endswith((".ts", ".js", ".mjs", ".cjs")):
        scripts = text
    else:
        scripts = ""
    parts += [a or b or c for a, b, c in _STRING.findall(scripts)]
    return "\n".join(part for part in parts if part)


def _style_text(attributes):
    """An element's inline style as CSS text: the ``style`` value and each
    ``style:`` directive written as a declaration."""
    parts = []
    for name, value in attributes:
        if name == "style":
            parts.append(_unquote(value))
        elif name.startswith("style:"):
            prop = name[len("style:"):].split("|")[0]
            parts.append(f"{prop}: {_unquote(value) if value else 'var(--' + prop + ')'}")
    return "; ".join(parts)


# ===========================================================================
# TN1 -- every toned ground carries the edge
# ===========================================================================
_TOKEN_REF = re.compile(r"var\(\s*(--[\w-]+)")
_TAILWIND_GROUND = re.compile(r"^(?:[\w-]+:)*bg-(?:surface-(?:50|[1-8]00)|accent-\d+)(?:/\d+)?$")
_ARBITRARY_BG = re.compile(r"^(?:[\w-]+:)*bg-\[(.*)\]$")
_GROUND_PROPERTIES = ("background", "background-color", "background-image")
_EDGE_SHORTHAND = re.compile(r"(?<![\w-])border\s*:\s*1px\s+solid\s+var\(--oo-edge\)")
_LENGTH = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:px|rem|em)?$|^(?:thin|medium|thick)$")
_STYLES = frozenset({
    "none", "hidden", "dotted", "dashed", "solid", "double", "groove", "ridge", "inset", "outset",
})


def _normal_selector(part):
    part = re.sub(r":global\(([^()]*)\)", r"\1", part.replace('"', "'"))
    part = re.sub(r"\s*([>+~])\s*", r" \1 ", part)
    return " ".join(part.split())


_SIMPLE = re.compile(
    r"\.[\w-]+|#[\w-]+|\[[^\]]*\]|::[\w-]+(?:\([^()]*\))?"
    r"|:[\w-]+(?:\((?:[^()]|\([^()]*\))*\))?|\*|[A-Za-z][\w-]*"
)


def _selector_parts(part):
    """``(prefix, identity, qualifiers)``: the ancestors, the last compound's
    type, classes, id and pseudo-elements, and its attributes and
    pseudo-classes."""
    normal = _normal_selector(part)
    words = normal.split(" ")
    prefix, compound = " ".join(words[:-1]), words[-1]
    identity, qualifiers = [], []
    for simple in _SIMPLE.findall(compound):
        if simple.startswith("[") or (simple.startswith(":") and not simple.startswith("::")):
            qualifiers.append(simple)
        else:
            identity.append(simple)
    return prefix, tuple(sorted(identity)), frozenset(qualifiers)


def _border_state(declared, state):
    """``state`` (width, style, colour) after one rule's border
    declarations, in their order; a side's own border is not the edge."""
    width, style, colour = state
    for name, value in declared:
        value = re.sub(r"\s*!\s*important$", "", value)
        if name == "border":
            width, style, colour = "medium", "none", "currentColor"
            for word in _words(value):
                if _LENGTH.match(word):
                    width = word
                elif word in _STYLES:
                    style = word
                else:
                    colour = word
        elif name == "border-width":
            width = value
        elif name == "border-style":
            style = value
        elif name == "border-color":
            colour = value
    return width, style, colour


def _grounded(value, grounds):
    return any(ref in grounds for ref in _TOKEN_REF.findall(value))


def _edge_findings(sources, grounds, tone_declared, field_edges=None):
    """What TN1 finds in ``sources`` ({path: text}); ``tone_declared`` says
    whether a stylesheet declares ``.oo-tone`` with the edge.

    Each rule that sets a ground or a border is read as the state it makes:
    the rules it refines (the same element, fewer qualifiers, in the same
    at-rule or outside every at-rule it sits in) applied first, from the
    least qualified, in source order, then itself. A rule inside an at-rule
    never refines a rule outside it."""
    findings = []
    toned = False
    field_edges = _FIELD_EDGES if field_edges is None else field_edges
    for path, text in sorted(sources.items()):
        rules = []
        for selector, body, context in _rules(_css_of(path, text)):
            declared = _declarations(body)
            for part in _split_top(selector):
                rules.append((part, declared, context))
        indexed = [(_selector_parts(part), part, declared, context) for part, declared, context in rules]
        fields = {_normal_selector(s) for s in field_edges.get(path, ())}
        for (prefix, identity, qualifiers), part, declared, context in indexed:
            if not any(n in _GROUND_PROPERTIES or n.startswith("border") for n, _ in declared):
                continue
            chain = sorted(
                (
                    (len(q), index, d) for index, ((p, i, q), _, d, c) in enumerate(indexed)
                    if p == prefix and i == identity and q <= qualifiers and c == context[:len(c)]
                ),
                key=lambda item: (item[0], item[1]),
            )
            state, colour_ground, image_ground = (None, None, None), None, None
            for _, _, declared_here in chain:
                state = _border_state(declared_here, state)
                for name, value in declared_here:
                    grounded = value if _grounded(value, grounds) else None
                    if name in ("background", "background-color"):
                        colour_ground = grounded
                    if name in ("background", "background-image"):
                        image_ground = grounded
            ground = colour_ground or image_ground
            if ground is None:
                continue
            width, style, colour = state
            wanted = ("var(--oo-edge)",)
            base = _normal_selector(part)
            if any(base == field or base.startswith(field) for field in fields):
                wanted += (_FIELD_EDGE,)
                for state, inks in _FIELD_STATE_EDGES:
                    if any(q.startswith(state) for q in qualifiers):
                        wanted += inks
            if not (width == "1px" and style == "solid" and colour in wanted):
                findings.append(
                    f"{path}: {part.strip()} fills with {ground} and its border is "
                    f"{width or 'unset'} {style or 'unset'} {colour or 'unset'}"
                )
        scripts = _scripts_of(path, text)
        for tag, attributes in _tags(_markup_of(path, text)):
            if not tag[0].islower():
                continue
            classes = _class_tokens(attributes, scripts)
            style = _style_text(attributes)
            filled = [c for c in classes if _TAILWIND_GROUND.match(c)] + [
                c for c in classes
                if _ARBITRARY_BG.match(c) and _grounded(_ARBITRARY_BG.match(c).group(1).replace("_", " "), grounds)
            ]
            filled += [
                value for name, value in _declarations(style)
                if name in _GROUND_PROPERTIES and _grounded(value, grounds)
            ]
            if not filled:
                continue
            if "oo-tone" in classes:
                toned = True
                continue
            if _EDGE_SHORTHAND.search(style) or (
                "border" in classes and "border-[var(--oo-edge)]" in classes
            ):
                continue
            findings.append(f"{path}: <{tag}> filled with {filled[0]} carries no edge")
        if re.search(r"(?<![\w-])oo-tone(?![\w-])", _markup_of(path, text)):
            toned = True
    if toned and not tone_declared:
        findings.append("oo-tone is used and no stylesheet declares .oo-tone with the edge")
    return findings


def _tone_declared(sheets):
    for text in sheets:
        for selector, body, _ in _rules(text):
            if ".oo-tone" in [s.strip() for s in _split_top(selector)]:
                state = _border_state(_declarations(body), (None, None, None))
                if state == ("1px", "solid", "var(--oo-edge)"):
                    return True
    return False


def _global_sheets():
    return [read(path) for path in (*_GLOBAL_SHEETS, *files(".css", within=_STYLES_DIR))]


_TN1_SAMPLE = {
    f"{_SRC}/lib/sample/Tone.svelte": (
        '<div class="oo-tone bg-surface-800"></div>\n'
        '<div class="bg-surface-800 rounded"></div>\n'
        '<div class="hover:bg-[var(--oo-bg-tint-1)]"></div>\n'
        '<div style="background: var(--oo-bg-code)"></div>\n'
        '<div style="background: var(--oo-bg-code); border: 1px solid var(--oo-edge)"></div>\n'
        '<div class="bg-surface-900 bg-[var(--oo-bg-hover)]"></div>\n'
        "<script>const tone = 'bg-surface-800 rounded';</script>\n"
        "<div class={tone}></div>\n"
        '<div class="bg-[color-mix(in_srgb,var(--oo-bg-surface)_80%,transparent)]"></div>\n'
        "<style>\n"
        ".card { background-color: var(--oo-bg-surface); border: 1px solid var(--oo-bd-subtle); }\n"
        ".ok { background: var(--oo-bg-elevated); border: 1px solid var(--oo-edge); }\n"
        ".btn { border: 1px solid transparent; }\n"
        ".btn[data-v='fill'] { background-color: var(--oo-acc-fill); border-color: var(--oo-edge); }\n"
        ".btn[data-v='fill']:hover:not(:disabled) { background-color: var(--oo-acc-fill-hover); }\n"
        ".btn[data-v='sheet'] { background-color: var(--oo-bg-overlay); }\n"
        ".wash { background: var(--oo-bg-hover); }\n"
        ".field { background-color: var(--oo-bg-input); border: 1px solid var(--oo-input-bd); }\n"
        ".knob { background-color: var(--oo-toggle-knob); }\n"
        "@media (max-width: 767px) { .sunk { background-color: var(--oo-bg-subtle); border: none; } }\n"
        ".lit { background-image: linear-gradient(var(--oo-bg-surface), var(--oo-bg-surface)); }\n"
        ".forced { background-color: var(--oo-bg-surface); }\n"
        "@media (forced-colors: active) { .forced { border: 1px solid var(--oo-edge); } }\n"
        ".reset { background-color: var(--oo-bg-tint-1); border: 1px solid var(--oo-edge); }\n"
        ".reset[data-quiet] { border-color: transparent; }\n"
        ".box { background-color: var(--oo-bg-input); border: 1px solid var(--oo-input-bd); }\n"
        ".box:focus { border-color: var(--oo-acc-500); }\n"
        ".box[aria-invalid='true'] { border-color: var(--oo-error); }\n"
        ".box:hover { border-color: var(--oo-acc-500); }\n"
        "</style>\n"
    ),
}


def test_tn1_every_toned_ground_carries_the_edge():
    grounds = _grounds(read(_THEME))
    expected = {
        "--oo-bg-surface", "--oo-bg-elevated", "--oo-bg-overlay", "--oo-bg-subtle",
        "--oo-bg-tint-1", "--oo-bg-tint-2", "--oo-bg-tint-3", "--oo-bg-code",
        "--oo-acc-fill", "--oo-acc-fill-hover", "--oo-btn-secondary-hover", "--oo-bg-input",
    }
    assert expected <= grounds, f"the grounds are read from the derivation layer: {sorted(expected - grounds)}"
    quiet = {"--oo-bg-hover", "--oo-bg-base", "--oo-success-bg", "--oo-scrim", "--oo-toggle-knob"}
    assert not quiet & grounds, f"a wash, the page ground and the knob are not grounds: {sorted(quiet & grounds)}"

    sample = _edge_findings(
        _TN1_SAMPLE, grounds, tone_declared=False, field_edges={f"{_SRC}/lib/sample/Tone.svelte": (".box",)},
    )
    named = sorted(re.sub(r"^.*?: ", "", finding).split(" ")[0] for finding in sample)
    assert named == sorted([
        "<div>", "<div>", "<div>", "<div>", "<div>", ".card", ".btn[data-v='sheet']", ".field", ".sunk",
        ".lit", ".forced", ".reset[data-quiet]", ".box:hover", "oo-tone",
    ]), f"the census finds each ground without its edge, and nothing else: {sample}"

    for path, selectors in _FIELD_EDGES.items():
        css = {_normal_selector(p) for s, _, _ in _rules(_css_of(path, read(path))) for p in _split_top(s)}
        stale = [s for s in selectors if _normal_selector(s) not in css]
        assert not stale, f"a listed field stands in its file: {path} {stale}"

    held = {path: read(path) for path in _frontend.held_files()}
    findings = _edge_findings(held, grounds, _tone_declared(_global_sheets()))
    assert not findings, (
        f"{len(findings)} toned grounds without the edge in held files:\n  " + "\n  ".join(findings)
    )


# ===========================================================================
# TN2 -- border tokens are read only by the listed selectors
# ===========================================================================
_BORDER_PREFIX = "--oo-bd-"
_TAILWIND_LINE_COLOUR = re.compile(r"^(?:[\w-]+:)*(?:border|divide|outline|ring)-surface-\d+(?:/\d+)?$")
_TAILWIND_WIDTH = re.compile(r"^(?:[\w-]+:)*(?:border(?:-[xytblrse])?(?:-\d+)?|border-\[\d+px\])$")
_TAILWIND_DIVIDE = re.compile(r"^(?:[\w-]+:)*divide-[xy](?:-\d+)?$")
_TAILWIND_COLOURED = re.compile(
    r"^(?:[\w-]+:)*(border|divide)-(?:surface-\d+|accent-\d+|transparent|current|inherit|\[.*\])(?:/\d+)?$"
)


def _border_aliases(texts):
    """Custom properties whose value reads a border token or another alias."""
    declared = {}
    for text in texts:
        for _, body, _ in _rules(text):
            for name, value in _declarations(body):
                if name.startswith("--"):
                    declared.setdefault(name, []).append(value)
    aliases = set()
    grew = True
    while grew:
        grew = False
        for name, values in declared.items():
            if name in aliases:
                continue
            refs = {ref for value in values for ref in _TOKEN_REF.findall(value)}
            if any(ref.startswith(_BORDER_PREFIX) or ref in aliases for ref in refs):
                aliases.add(name)
                grew = True
    return aliases


def _reads_border(value, aliases):
    return [
        ref for ref in _TOKEN_REF.findall(value)
        if ref.startswith(_BORDER_PREFIX) or ref in aliases
    ]


def _border_findings(sources, aliases, readers):
    findings = []
    for path, text in sorted(sources.items()):
        allowed = {_normal_selector(s) for s in readers.get(path, ())}
        for selector, body, _ in _rules(_css_of(path, text)):
            refs = [ref for _, value in _declarations(body) for ref in _reads_border(value, aliases)]
            if not refs:
                continue
            outside = [p for p in _split_top(selector) if _normal_selector(p) not in allowed]
            if outside:
                findings.append(f"{path}: {', '.join(outside)} reads {', '.join(sorted(set(refs)))}")
        for tag, attributes in _tags(_markup_of(path, text)):
            classes = _class_tokens(attributes)
            refs = sorted(c for c in classes if _TAILWIND_LINE_COLOUR.match(c) or _reads_border(c, aliases))
            coloured = {m.group(1) for c in classes for m in [_TAILWIND_COLOURED.match(c)] if m}
            if any(_TAILWIND_WIDTH.match(c) for c in classes) and "border" not in coloured:
                refs.append("the default border colour")
            if any(_TAILWIND_DIVIDE.match(c) for c in classes) and "divide" not in coloured:
                refs.append("the default divider colour")
            refs += _reads_border(_style_text(attributes), aliases)
            if refs:
                findings.append(f"{path}: <{tag}> reads {', '.join(refs)}")
    return findings


_TN2_SAMPLE = {
    f"{_SRC}/lib/sample/Lines.svelte": (
        '<div class="border rounded"></div>\n'
        '<div class="border border-[var(--oo-edge)]"></div>\n'
        '<ul class="divide-y"></ul>\n'
        '<p class="border-surface-700 border-l"></p>\n'
        '<p style="outline: 1px solid var(--oo-border)"></p>\n'
        "<style>\n"
        ".row { border-bottom: 1px solid var(--oo-bd-subtle); }\n"
        ".field { border: 1px solid var(--oo-bd-strong); }\n"
        ".edge { border: 1px solid var(--oo-edge); }\n"
        "</style>\n"
    ),
}


def test_tn2_border_tokens_are_read_only_by_the_listed_selectors():
    aliases = _border_aliases(_global_sheets())
    assert {"--oo-border", "--oo-border-default"} <= aliases, (
        f"the aliases of a border token are derived: {sorted(aliases)}"
    )
    sample = _border_findings(
        _TN2_SAMPLE, aliases, {f"{_SRC}/lib/sample/Lines.svelte": (".field",)},
    )
    assert len(sample) == 5 and not any(".field" in f or "border-[var(--oo-edge)]" in f for f in sample), (
        f"the census finds each read outside the listed selectors, and nothing else: {sample}"
    )

    for path, selectors in _BORDER_READERS.items():
        css = {_normal_selector(p) for s, _, _ in _rules(_css_of(path, read(path))) for p in _split_top(s)}
        stale = [s for s in selectors if _normal_selector(s) not in css]
        assert not stale, f"a listed reader stands in its file: {path} {stale}"

    held = {path: read(path) for path in _frontend.held_files()}
    findings = _border_findings(held, aliases, _BORDER_READERS)
    assert not findings, (
        f"{len(findings)} reads of a border token outside the listed selectors:\n  " + "\n  ".join(findings)
    )


# ===========================================================================
# TN3 -- no capitals and no wide tracking
# ===========================================================================
_UPPERCASE = re.compile(r"(?<![\w-])(?:uppercase|capitalize|(?:all-)?small-caps)(?![\w-])")
_TRACKING_CLASS = re.compile(
    r"(?<![\w-])tracking-(?:wide|wider|widest|\[(\d*\.?\d+)[a-z%]*\]"
    r"|\[var\(\s*--oo-tracking-wide[\w-]*\s*\)\])(?![\w-])"
)
_LETTER_SPACING = re.compile(
    r"(?:letter-spacing\s*:|(?<![\w-])style:letter-spacing\s*=\s*[{\"'`]*)"
    r"\s*([^;\"'`}\n]*)"
)
_LEADING_NUMBER = re.compile(r"\+?(\d*\.?\d+)")
_WIDE_TOKEN = re.compile(r"var\(\s*--oo-tracking-wide")


def _capitals(text):
    found = [m.group(0) for m in _UPPERCASE.finditer(text)]
    for match in _TRACKING_CLASS.finditer(text):
        if match.group(1) is None or float(match.group(1)) > 0:
            found.append(match.group(0))
    for match in _LETTER_SPACING.finditer(text):
        value = match.group(1).strip()
        number = _LEADING_NUMBER.match(value)
        if (number and float(number.group(1)) > 0) or _WIDE_TOKEN.match(value):
            found.append(f"letter-spacing: {value}")
    return found


def test_tn3_no_capitals_and_no_wide_tracking():
    sample = _capitals(_surfaces(_SAMPLE_PATH, (
        '<p class="uppercase tracking-widest tracking-[0.1em]" style:letter-spacing="0.1em"></p>\n'
        "<script>const label = 'x small-caps';</script>\n"
        "<style>a { letter-spacing: 0.05em; } b { letter-spacing: var(--oo-tracking-wide); } "
        "c { font-variant: small-caps; }</style>\n"
        '<span class="capitalize"></span><style>d { text-transform: capitalize; }</style>\n'
    )))
    assert len(sample) == 10, f"the census reads each form of capitals and wide tracking: {sample}"
    level = _capitals(_surfaces(_SAMPLE_PATH, (
        "<!-- no uppercase here, and no tracking-widest -->\n"
        '<p class="normal-case tracking-tight"></p>\n'
        "<style>/* uppercase */ a { letter-spacing: -0.01em; } b { letter-spacing: 0; } "
        "c { letter-spacing: var(--oo-tracking-tight); }</style>\n"
    )))
    assert not level, f"level or tight spacing, and prose in a comment, are not counted: {level}"

    findings = [
        f"{path}: {', '.join(found)}"
        for path in _frontend.held_files() for found in [_capitals(_surfaces(path, read(path)))] if found
    ]
    assert not findings, "capitals or wide tracking in held files:\n  " + "\n  ".join(findings)


# ===========================================================================
# TN4 -- matte: no gradient, no inset shadow, only the shadow tokens
# ===========================================================================
_GRADIENT = re.compile(
    r"(?<![\w-])(?!linear-gradient\(\s*currentColor\s+0\s+0)[\w-]*gradient\("
)
_GRADIENT_CLASS = re.compile(r"(?<![\w-])bg-(?:gradient|linear|radial|conic)(?![a-z])")
_SHADOW_TOKEN = re.compile(r"^var\(\s*--oo-shadow-(?:xs|sm|md|lg)\s*\)$")
_RING = re.compile(r"^0(?:px)?\s+0(?:px)?\s+0(?:px)?\s+-?\d*\.?\d+(?:px|rem)\s+var\(\s*--oo-[\w-]+\s*\)$")
_SHADOW_CLASS = re.compile(r"(?<![\w-])(?:[\w-]+:)*(shadow(?:-(?:sm|md|lg|xl|2xl|inner|\[[^\]]*\]))?|drop-shadow(?:-[\w\[\]-]+)?)(?![\w(-])")


def _shadow_findings(text):
    """Gradients but the drawing idiom, inset shadows, shadows not drawn
    from the shadow tokens (a focus ring excepted), text and drop shadows."""
    found = [m.group(0) for m in _GRADIENT.finditer(text)]
    found += [m.group(0) for m in _GRADIENT_CLASS.finditer(text)]
    for match in re.finditer(r"(?<![\w-])(box-shadow|text-shadow)\s*:\s*([^;}\"'\n]*)", text):
        name, value = match.group(1), " ".join(match.group(2).split())
        value = re.sub(r"\s*!\s*important$", "", value)
        if value in ("none", "", "0"):
            continue
        if name == "text-shadow":
            found.append(f"text-shadow: {value}")
            continue
        for part in _split_top(value):
            if re.search(r"(?<![\w-])inset(?![\w-])", part):
                found.append(f"inset: {part}")
            elif not (_SHADOW_TOKEN.match(part) or _RING.match(part)):
                found.append(f"box-shadow: {part}")
    found += [m.group(0) for m in re.finditer(r"drop-shadow\(", text)]
    for match in _SHADOW_CLASS.finditer(text):
        if match.group(0).endswith(("shadow-none",)):
            continue
        found.append(match.group(0))
    return found


def test_tn4_matte_no_gradient_no_inset_and_only_the_shadow_tokens():
    sample = _shadow_findings(_surfaces(_SAMPLE_PATH, (
        '<div class="bg-gradient-to-r shadow-inner shadow-md drop-shadow-sm"></div>\n'
        '<div style="box-shadow: 0 1px 2px var(--oo-shadow-c1)"></div>\n'
        "<style>a { background: radial-gradient(circle, var(--oo-acc-500), transparent); }\n"
        "b { box-shadow: inset 0 1px 0 var(--oo-shadow-c1); }\n"
        "c { box-shadow: 0 2px 8px var(--oo-shadow-c2); }\n"
        "d { text-shadow: 0 1px 0 var(--oo-shadow-c1); filter: drop-shadow(0 1px 2px var(--oo-shadow-c1)); }\n"
        "</style>\n"
    )))
    assert len(sample) == 10, f"the census reads each form of depth: {sample}"
    matte = _shadow_findings(_surfaces(_SAMPLE_PATH, (
        "<!-- raised (border + shadow-sm), and no gradient( here -->\n"
        '<div class="shadow-none"></div>\n'
        "<style>a { box-shadow: var(--oo-shadow-sm); } b { box-shadow: none; } "
        "c { box-shadow: 0 0 0 3px var(--oo-input-focus); } "
        "d { background-image: linear-gradient(currentColor 0 0); }</style>\n"
    )))
    assert not matte, (
        f"the shadow tokens, a focus ring, the drawing idiom and prose in a comment are matte: {matte}"
    )

    findings = [
        f"{path}: {', '.join(found)}"
        for path in _frontend.held_files() for found in [_shadow_findings(_surfaces(path, read(path)))] if found
    ]
    assert not findings, "depth that is not matte in held files:\n  " + "\n  ".join(findings)


# ===========================================================================
# TN5 -- selection is never shown by fill alone
# ===========================================================================
_WEIGHT = re.compile(r"^(?:[6-9]00|bold|bolder)$")


def _if_blocks(markup):
    """``(condition, then-branch)`` of every ``{#if}`` block, nesting kept."""
    blocks = []
    for match in re.finditer(r"\{#if\s+", markup):
        depth, cursor = 0, match.start()
        while cursor < len(markup):
            if markup[cursor] == "{":
                depth += 1
            elif markup[cursor] == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        condition = " ".join(markup[match.end():cursor].split())
        level, index, end = 1, cursor + 1, len(markup)
        token = re.compile(r"\{#if\b|\{/if\}|\{:else\b")
        for found in token.finditer(markup, index):
            if found.group(0).startswith("{#if"):
                level += 1
            elif found.group(0) == "{/if}":
                level -= 1
                if level == 0:
                    end = found.start()
                    break
            elif level == 1:
                end = found.start()
                break
        blocks.append((condition, markup[cursor + 1:end]))
    return blocks


def _inner(markup, name, end):
    """The markup inside the element whose opening tag, named ``name``,
    ends at ``end``; empty for a tag that closes itself."""
    if markup[:end].rstrip().endswith("/>"):
        return ""
    depth = 1
    for found in re.finditer(r"<(/?)" + re.escape(name) + r"(?=[\s/>])", markup[end:]):
        start = end + found.start()
        if found.group(1):
            depth -= 1
            if depth == 0:
                return markup[end:start]
        elif not markup[start:_tag_end(markup, start)].rstrip().endswith("/>"):
            depth += 1
    return markup[end:]


def _tags_at(markup):
    """``(name, attributes, end)`` of every element tag, ``end`` just past
    its opening tag."""
    index = 0
    while True:
        start = markup.find("<", index)
        if start < 0:
            return
        match = re.match(r"<([A-Za-z][\w.:-]*)", markup[start:])
        if not match:
            index = start + 1
            continue
        end = _tag_end(markup, start)
        yield match.group(1), _attributes(markup[start + len(match.group(0)):end - 1]), end
        index = end


def _draws_check(branch):
    return bool(re.search(r"""<Icon\b[^>]*\bname\s*=\s*(?:["']check["']|\{\s*['"]check['"]\s*\})""", branch))


def _selection_findings(sources):
    """What TN5 finds, and how many elements carry ``aria-pressed``."""
    findings, pressed = [], 0
    for path, text in sorted(sources.items()):
        markup = _markup_of(path, text)
        current_rules = False
        rules = _rules(_css_of(path, text))
        weights = [
            (_normal_selector(selector), context, dict(_declarations(body)).get("font-weight", ""))
            for selector, body, context in rules
        ]
        for selector, body, context in rules:
            declared = dict(_declarations(body))
            # The weight the element has here: this rule's, or that of the same
            # selector in an enclosing context (a media query refines the rule
            # outside it, and keeps its weight).
            weight = declared.get("font-weight", "") or next((
                w for s, c, w in reversed(weights)
                if w and s == _normal_selector(selector) and c == context[:len(c)] and len(c) < len(context)
            ), "")
            if re.search(r"\[aria-current\b", selector):
                current_rules = True
                if not _WEIGHT.match(weight):
                    findings.append(f"{path}: {selector.strip()} marks the current item without a weight of 600")
            if re.search(r"""\[aria-selected(?:=['"]?true['"]?)?\]""", selector):
                fill = [
                    v for n, v in declared.items()
                    if n in ("background", "background-color") and v not in ("transparent", "none")
                ]
                if fill and not _WEIGHT.match(weight):
                    findings.append(f"{path}: {selector.strip()} fills the selected item and sets no weight of 600")
        for tag, attributes, end in _tags_at(markup):
            values = dict(attributes)
            spread = [value for name, value in attributes if name == "{spread}" and "aria-pressed" in (value or "")]
            if spread:
                pressed += 1
                findings.append(f"{path}: <{tag} {spread[0]}> gives aria-pressed through a spread, whose check cannot be read")
            if "aria-pressed" in values:
                pressed += 1
                raw = values["aria-pressed"] or ""
                inside = _inner(markup, tag, end)
                if raw.startswith("{"):
                    expression = " ".join(raw[1:-1].split())
                    wanted = {expression, f"{expression} === true", f"!!{expression}"}
                    if not any(cond in wanted and _draws_check(branch) for cond, branch in _if_blocks(inside)):
                        findings.append(f"{path}: <{tag} aria-pressed={raw}> draws no check under {{#if {expression}}}")
                elif _unquote(raw) == "true" and not _draws_check(inside):
                    findings.append(f"{path}: <{tag} aria-pressed=\"true\"> draws no check")
            if "aria-current" in values and not current_rules:
                classes = _class_tokens(attributes)
                if not any(re.search(r"(?:^|:)font-(?:semibold|bold)$", c) for c in classes):
                    findings.append(f"{path}: <{tag} aria-current> is not set at a weight of 600")
    return findings, pressed


_TN5_SAMPLE = {
    f"{_SRC}/lib/sample/Chips.svelte": (
        '<button aria-pressed={on}>{#if on}<Icon name="check" />{/if}Web</button>\n'
        '<button aria-pressed={active}>{#if other}<Icon name="check" />{/if}Code</button>\n'
        '<button aria-pressed={lit}>{#if lit}<Icon name="dot" />{:else}<Icon name="check" />{/if}Lit</button>\n'
        '<a href="/" aria-current="page">Home</a>\n'
        '<button aria-pressed={shared}>{#if shared}<Icon name="check" />{/if}One</button>\n'
        '<button aria-pressed={shared}>Two</button>\n'
        '<button {...{ "aria-pressed": spread }}>Spread</button>\n'
        "<style>\n"
        "a[aria-current='page'] { font-weight: 500; }\n"
        ".tab[aria-selected='true'] { background-color: var(--oo-acc-fill); }\n"
        ".opt[aria-selected='true'] { background-color: var(--oo-bg-tint-1); font-weight: 600; }\n"
        "@media (forced-colors: active) { .opt[aria-selected='true'] { background-color: SelectedItem; } "
        ".alone[aria-selected='true'] { background-color: SelectedItem; } }\n"
        "</style>\n"
    ),
}


def test_tn5_selection_is_never_shown_by_fill_alone():
    sample, pressed = _selection_findings(_TN5_SAMPLE)
    assert pressed == 6 and len(sample) == 7 and not any("aria-pressed={on}" in f for f in sample) and (
        sum(".alone" in f for f in sample) == 1 and not any(".opt" in f for f in sample)
    ) and (
        sum("aria-pressed={shared}" in f for f in sample) == 1 and any("spread" in f for f in sample)
    ), (
        f"the census finds a pressed state without its check inside it, one given through a "
        f"spread, a current item and a filled selection without their weight, and nothing else: {sample}"
    )

    held = {path: read(path) for path in _frontend.held_files()}
    findings, pressed = _selection_findings(held)
    assert pressed, "a held file carries aria-pressed: the pressed rule holds over something"
    assert not findings, "selection shown by fill alone in held files:\n  " + "\n  ".join(findings)


# ===========================================================================
# TN6 -- HELD only grows, and names files of the tree
# ===========================================================================
def _held_at_head():
    """``HELD`` as ``tests/_frontend.py`` holds it at HEAD."""
    source = subprocess.run(
        ["git", "-C", str(REPO), "show", "HEAD:tests/_frontend.py"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout
    for node in ast.parse(source).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
        )
        if isinstance(target, ast.Name) and target.id == "HELD" and node.value is not None:
            return list(ast.literal_eval(node.value))
    raise AssertionError("HEAD's tests/_frontend.py assigns no HELD")


def _growth_findings(now, at_head, root=REPO):
    findings = [f"{entry} left HELD" for entry in at_head if entry not in now]
    findings += [f"{entry} is on HELD twice" for entry in set(now) if now.count(entry) > 1]
    for entry in now:
        try:
            _frontend.held_files([entry], root=root)
        except AssertionError as exc:
            findings.append(f"{entry}: {str(exc).splitlines()[0]}")
    return findings


def test_tn6_held_only_grows_and_names_files_of_the_tree():
    sample = _growth_findings(
        [f"{_SRC}/lib/ds/", f"{_SRC}/lib/ds/", f"{_SRC}/lib/absent.svelte", f"{_SRC}/lib/nowhere/"],
        [f"{_SRC}/lib/ds/", f"{_SRC}/app.html"],
    )
    assert len(sample) == 4 and any("app.html left" in f for f in sample), (
        f"the census finds an entry that left, one written twice and two that name nothing: {sample}"
    )
    assert isinstance(_frontend.HELD, list) and all(isinstance(e, str) for e in _frontend.HELD)
    findings = _growth_findings(list(_frontend.HELD), _held_at_head())
    assert not findings, "HELD shrank or names nothing:\n  " + "\n  ".join(findings)



# ===========================================================================
# TN7 -- focus stays visible
# ===========================================================================
_OUTLINE_REMOVED = re.compile(r"^(?:none|0(?:px)?)(?:\s|$)|(?:^|\s)none(?:\s|$)")
_OUTLINE_CLASS = re.compile(r"^(?:[\w-]+:)*outline-(?:none|0)$")
_RING_CLASS = re.compile(r"^(?:[\w-]+:)*ring(?:-\d+|-\[[^\]]*\])?$")
_FOCUS_INK = "var(--oo-focus-ink)"


def _removes_outline(declared):
    for name, value in declared:
        value = re.sub(r"\s*!\s*important$", "", value).strip()
        if name == "outline" and _OUTLINE_REMOVED.search(value):
            return True
        if name == "outline-style" and value == "none":
            return True
        if name == "outline-width" and re.fullmatch(r"0(?:px)?", value):
            return True
    return False


def _draws_ring(declared, ink=None):
    """Whether a rule draws a ring: a box-shadow ring (a spread, no offset,
    no blur) or a drawn outline, in ``ink`` when given."""
    for name, value in declared:
        value = re.sub(r"\s*!\s*important$", "", value).strip()
        if name == "box-shadow":
            if any(_RING.match(part) and (ink is None or ink in part) for part in _split_top(value)):
                return True
        elif name == "outline" and not _OUTLINE_REMOVED.search(value):
            if ink is None or ink in value:
                return True
        elif name == "outline-color" and ink is not None and ink in value:
            return True
    return False


def _outline_findings(sources):
    """A rule, or a class, that removes the outline and draws no ring in its place."""
    findings = []
    for path, text in sorted(sources.items()):
        for selector, body, context in _rules(_css_of(path, text)):
            if any("forced-colors" in c for c in context):
                continue
            declared = _declarations(body)
            if _removes_outline(declared) and not _draws_ring(declared):
                findings.append(f"{path}: {selector.strip()} removes the outline and draws no ring")
        scripts = _scripts_of(path, text)
        for tag, attributes in _tags(_markup_of(path, text)):
            classes = _class_tokens(attributes, scripts)
            removed = sorted(c for c in classes if _OUTLINE_CLASS.match(c))
            if removed and not any(_RING_CLASS.match(c) for c in classes):
                findings.append(f"{path}: <{tag}> {removed[0]} and no ring")
    return findings


def _active_findings(sources):
    """``(files that point aria-activedescendant, findings)``: in each, a
    class toggled on an element whose rule sets a ground draws an outline
    or a ring in the focus ink, not a wash alone."""
    pointing, findings = [], []
    for path, text in sorted(sources.items()):
        markup = _markup_of(path, text)
        if not re.search(r"(?<![\w-])aria-activedescendant\s*=", markup):
            continue
        pointing.append(path)
        toggled = {
            name[len("class:"):] for _, attributes in _tags(markup) for name, _ in attributes
            if name.startswith("class:")
        }
        for selector, body, context in _rules(_css_of(path, text)):
            if any("forced-colors" in c for c in context):
                continue
            declared = _declarations(body)
            if not any(name in _GROUND_PROPERTIES for name, _ in declared):
                continue
            marks = [
                name for name in toggled
                if re.search(r"\." + re.escape(name) + r"(?![\w-])", selector)
            ]
            if marks and not _draws_ring(declared, _FOCUS_INK):
                findings.append(
                    f"{path}: {selector.strip()} marks the active item ({marks[0]}) with a ground alone"
                )
    return pointing, findings


_TN7_SAMPLE = {
    f"{_SRC}/lib/sample/Focus.svelte": (
        '<input class="focus:outline-none" />\n'
        '<input class="focus:outline-none focus:ring-2" />\n'
        '<input aria-activedescendant={at} />\n'
        '<li class="opt" class:current={i === at} class:gone={i < 0}></li>\n'
        "<style>\n"
        ".a { outline: none; }\n"
        ".b:focus { outline: none; box-shadow: 0 0 0 3px var(--oo-input-focus); }\n"
        ".c { outline: 0; }\n"
        "@media (forced-colors: active) { .d { outline: none; } }\n"
        ".opt.current { background-color: var(--oo-bg-hover); }\n"
        ".opt.gone { opacity: 0.5; }\n"
        ".opt.current:hover { background-color: var(--oo-bg-hover); outline: 2px solid var(--oo-acc-mark); }\n"
        "</style>\n"
    ),
    f"{_SRC}/lib/sample/Kept.svelte": (
        '<input aria-activedescendant={at} />\n'
        '<li class="opt" class:current={i === at}></li>\n'
        "<style>.opt.current { background-color: var(--oo-bg-hover); outline: 2px solid var(--oo-focus-ink); "
        "outline-offset: -2px; }</style>\n"
    ),
}


@pytest.mark.parametrize("half", ("outline", "active"))
def test_tn7_focus_stays_visible(half):
    held = {path: read(path) for path in _frontend.held_files()}
    if half == "outline":
        sample = _outline_findings(_TN7_SAMPLE)
        assert sorted(f.split(": ", 1)[1].split(" ")[0] for f in sample) == sorted(["<input>", ".a", ".c"]), (
            f"the census finds an outline removed with no ring in its place, and nothing else: {sample}"
        )
        findings = _outline_findings(held)
        assert not findings, "focus hidden in held files:\n  " + "\n  ".join(findings)
        return

    pointing, sample = _active_findings(_TN7_SAMPLE)
    assert len(pointing) == 2 and [f.split(": ", 1)[1].split(" ")[0] for f in sample] == [
        ".opt.current", ".opt.current:hover",
    ], f"the census finds an active item marked by a ground alone, or off the focus ink, and nothing else: {sample}"
    pointing, findings = _active_findings(held)
    assert pointing, "a held file points aria-activedescendant: the rule holds over something"
    assert not findings, "an active item shown by a wash alone in held files:\n  " + "\n  ".join(findings)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
