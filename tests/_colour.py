#!/usr/bin/env python3
"""A colour evaluator for the design-token contracts: tokens, mixes, compositing, contrast.

A helper module, not a suite: it holds no contract of its own. The design
tokens are CSS custom properties whose values are expressions of palette
roles; this module evaluates those expressions the way a browser computes
them, so a contrast ratio can be read without one:

  * ``parse(text)`` -- a colour literal: hex (3, 4, 6 or 8 digits),
    ``rgb()``/``rgba()`` in the comma or the space syntax with an optional
    alpha, numbers or percentages, and a few named colours (``transparent``,
    ``black``, ``white``, ``red``, ``lime``, ``blue``). Returns
    ``(r, g, b, a)`` with channels on 0-255 and alpha on 0-1, unrounded.
  * ``mix(first, p, second, q)`` -- ``color-mix(in srgb, ...)`` as CSS Color
    5 defines it: omitted percentages completed to 100, a sum under 100
    kept as an alpha multiplier, and the channels interpolated
    premultiplied by alpha.
  * ``over(top, ground)`` -- source-over compositing of a colour on another.
  * ``contrast(a, b)`` -- the WCAG 2 contrast ratio of two opaque colours.
  * ``Resolver(tokens, roles, scheme)`` -- evaluates a token name or an
    expression: ``var(--name[, fallback])`` (a role ``--oo-role-<name>``
    read from ``roles``, any other name from ``tokens``), ``color-mix()``,
    ``light-dark()`` chosen by ``scheme`` ("light" or "dark"), and literals.
    An undeclared name without a fallback raises ``KeyError``; a chain that
    loops raises ``ValueError``.

Browsers keep 8-bit channels at some steps; this evaluator does not round,
so a ratio can differ from a browser's in the second decimal. What a browser
computes is recorded on the machine and compared by DS17.

Local-only (the public distribution ships no tests).
"""

from __future__ import annotations

import re

ROLE_PREFIX = "--oo-role-"

_NAMED = {
    "transparent": (0.0, 0.0, 0.0, 0.0),
    "black": (0.0, 0.0, 0.0, 1.0),
    "white": (255.0, 255.0, 255.0, 1.0),
    "red": (255.0, 0.0, 0.0, 1.0),
    "lime": (0.0, 255.0, 0.0, 1.0),
    "blue": (0.0, 0.0, 255.0, 1.0),
}
_HEX = re.compile(r"#([0-9a-fA-F]{3,8})")
_FUNCTION = re.compile(r"(rgba?)\((.*)\)", re.S)


def _channel(part):
    part = part.strip()
    if part.endswith("%"):
        return float(part[:-1]) * 255.0 / 100.0
    return float(part)


def _alpha(part):
    part = part.strip()
    if part.endswith("%"):
        return float(part[:-1]) / 100.0
    return float(part)


def parse(text):
    """``(r, g, b, a)`` of a colour literal; ``ValueError`` for anything else."""
    text = text.strip()
    lowered = text.lower()
    if lowered in _NAMED:
        return _NAMED[lowered]
    match = _HEX.fullmatch(text)
    if match:
        digits = match.group(1)
        if len(digits) in (3, 4):
            digits = "".join(c * 2 for c in digits)
        if len(digits) not in (6, 8):
            raise ValueError(f"not a hex colour: {text}")
        values = [int(digits[i:i + 2], 16) for i in range(0, len(digits), 2)]
        alpha = values[3] / 255.0 if len(values) == 4 else 1.0
        return (float(values[0]), float(values[1]), float(values[2]), alpha)
    match = _FUNCTION.fullmatch(lowered)
    if match:
        body = match.group(2).strip()
        if "," in body:
            parts = [p for p in body.split(",")]
            alpha = parts[3] if len(parts) == 4 else None
            channels = parts[:3]
        else:
            main, _, alpha = body.partition("/")
            channels = main.split()
            alpha = alpha or None
        if len(channels) != 3:
            raise ValueError(f"not an rgb colour: {text}")
        r, g, b = (_channel(c) for c in channels)
        a = 1.0 if alpha is None else _alpha(alpha)
        return (r, g, b, a)
    raise ValueError(f"not a colour literal: {text}")


def mix(first, p, second, q):
    """``color-mix(in srgb, first p%, second q%)``; ``p`` or ``q`` None when
    omitted, both as percentages (0-100)."""
    if p is None and q is None:
        p = q = 50.0
    elif p is None:
        p = 100.0 - q
    elif q is None:
        q = 100.0 - p
    total = p + q
    if total <= 0:
        raise ValueError("color-mix percentages sum to zero")
    multiplier = min(total, 100.0) / 100.0
    p, q = p / total, q / total
    alpha = first[3] * p + second[3] * q
    if alpha == 0:
        channels = (0.0, 0.0, 0.0)
    else:
        channels = tuple(
            (first[i] * first[3] * p + second[i] * second[3] * q) / alpha for i in range(3)
        )
    return (*channels, alpha * multiplier)


def over(top, ground):
    """``top`` composited over ``ground`` (source-over)."""
    alpha = top[3] + ground[3] * (1.0 - top[3])
    if alpha == 0:
        return (0.0, 0.0, 0.0, 0.0)
    channels = tuple(
        (top[i] * top[3] + ground[i] * ground[3] * (1.0 - top[3])) / alpha for i in range(3)
    )
    return (*channels, alpha)


def _luminance(colour):
    out = []
    for value in colour[:3]:
        value = value / 255.0
        out.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]


def contrast(a, b):
    """The WCAG 2 contrast ratio of two opaque colours."""
    for colour in (a, b):
        if abs(colour[3] - 1.0) > 1e-9:
            raise ValueError(f"contrast is read between opaque colours, not {colour}")
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _split_arguments(text):
    """The top-level comma-separated arguments of a function's body."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _function(text):
    """``(name, body)`` when ``text`` is one whole function call, else None."""
    match = re.fullmatch(r"([a-zA-Z-]+)\((.*)\)", text.strip(), re.S)
    if not match:
        return None
    depth = 0
    for index, char in enumerate(match.group(2)):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
    return match.group(1).lower(), match.group(2)


_OPERAND = re.compile(r"^(.*?)(?:\s+(\d+(?:\.\d+)?)%)?$", re.S)


class Resolver:
    """Evaluates tokens and expressions against one palette."""

    def __init__(self, tokens, roles, scheme):
        self.tokens = dict(tokens)
        self.roles = dict(roles)
        self.scheme = scheme

    def colour(self, expression, _seen=()):
        """``(r, g, b, a)`` of a token name (``--oo-...``) or an expression."""
        expression = expression.strip()
        if expression.startswith("--"):
            return self._name(expression, None, _seen)
        call = _function(expression)
        if call is None:
            return parse(expression)
        name, body = call
        if name == "var":
            arguments = _split_arguments(body)
            fallback = ",".join(arguments[1:]).strip() if len(arguments) > 1 else None
            return self._name(arguments[0], fallback, _seen)
        if name == "light-dark":
            first, second = _split_arguments(body)
            chosen = first if self.scheme == "light" else second
            return self.colour(chosen, _seen)
        if name == "color-mix":
            arguments = _split_arguments(body)
            if len(arguments) != 3 or not re.fullmatch(r"in\s+srgb", arguments[0]):
                raise ValueError(f"only color-mix in srgb of two colours is evaluated: {expression}")
            operands = []
            for argument in arguments[1:]:
                match = _OPERAND.match(argument)
                percentage = float(match.group(2)) if match.group(2) is not None else None
                operands.append((self.colour(match.group(1), _seen), percentage))
            (first, p), (second, q) = operands
            return mix(first, p, second, q)
        return parse(expression)

    def _name(self, name, fallback, seen):
        if name in seen:
            raise ValueError(f"the chain loops: {' -> '.join(seen + (name,))}")
        seen = seen + (name,)
        if name.startswith(ROLE_PREFIX):
            role = name[len(ROLE_PREFIX):]
            if role in self.roles:
                return self.colour(self.roles[role], seen)
        elif name in self.tokens:
            return self.colour(self.tokens[name], seen)
        if fallback is not None:
            return self.colour(fallback, seen)
        raise KeyError(f"{name} is declared nowhere")
