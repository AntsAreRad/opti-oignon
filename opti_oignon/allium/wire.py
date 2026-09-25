"""Onion canonical JSON (OCJ), wire v1: the one encoding every hashed value travels in.

The reference codec. Its Rust twin (``rust/allium/src/ocj.rs``) must accept,
refuse and emit exactly as it does, byte for byte, refusal codes included.

The domain is small on purpose, so that two engines can agree on it:

* bytes: printable ASCII 0x20-0x7E only, and no whitespace outside strings;
* values: object, array, string, integer, true, false, null -- no float;
* integers: within +-(2^53 - 1); wider values travel as hex strings;
* strings: printable ASCII, with the escapes ``\\"`` and ``\\\\`` only;
* objects: keys strictly increasing in byte order (sorted, unique);
* nesting depth at most 16.

On that domain, ``emit(v)`` equals ``json.dumps(v, sort_keys=True,
separators=(",", ":"), ensure_ascii=True)``, and a canonical input is its own
re-emission. Law and table files on disk are the one exception to the
whitespace rule: ``parse(data, lenient=True)`` reads them indented, and their
digest is taken over the canonical re-emission.

The first defect met, scanning left to right after a byte pre-pass, decides
the refusal; both engines scan in the same order, so they refuse with the
same code and the same detail.
"""

import re

checkpoint_before_apply = True

MAX_INT = (1 << 53) - 1
MAX_DEPTH = 16

# The closed set of refusal codes. Adding one is a change to the wire.
REFUSALS = (
    "non_canonical",
    "float",
    "non_ascii",
    "limit",
    "unknown_op",
    "bad_request",
    "unknown_law",
    "chain",
    "bad_fact",
    "budget",
    "engine_panic",
)

_WS_LENIENT = (0x20, 0x09, 0x0A, 0x0D)
# Inside a string, the bytes that end a plain run: the quote, the backslash, and the
# whitespace only a lenient file lets through. Everything else already passed the pre-pass.
_STRING_SPECIAL = re.compile(rb'["\\\t\n\r]')
# A string that emits as itself: printable ASCII with neither the quote nor the backslash.
_STRING_PLAIN = re.compile(r'[ -!#-\[\]-~]*')

# Compact integer arrays: "<type>:<lowercase hex>", little-endian two's complement.
BULK_TYPES = {
    "u8": (1, False),
    "i8": (1, True),
    "u16": (2, False),
    "i16": (2, True),
    "u32": (4, False),
    "i32": (4, True),
    "u64": (8, False),
}
_HEX = "0123456789abcdef"


def _digit(byte):
    return 0x30 <= byte <= 0x39


class Refused(Exception):
    """A request the engine will not take, named by a code from ``REFUSALS``."""

    def __init__(self, code, detail):
        if code not in REFUSALS:
            raise ValueError(f"unknown refusal code: {code}")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

    def as_value(self):
        return {"refused": self.code, "detail": self.detail}


class _Parser:
    __slots__ = ("data", "pos", "lenient", "max_depth")

    def __init__(self, data, lenient, max_depth):
        self.data = data
        self.pos = 0
        self.lenient = lenient
        self.max_depth = max_depth

    def peek(self):
        if self.pos < len(self.data):
            return self.data[self.pos]
        return -1

    def gap(self):
        """Between two tokens: skip whitespace in a file, refuse it on the wire."""
        while self.pos < len(self.data):
            byte = self.data[self.pos]
            if byte not in _WS_LENIENT:
                return
            if not self.lenient:
                raise Refused("non_canonical", f"whitespace at {self.pos}")
            self.pos += 1

    def expect(self, byte):
        if self.peek() != byte:
            raise Refused("non_canonical", f"expected {chr(byte)} at {self.pos}")
        self.pos += 1

    def value(self, depth):
        self.gap()
        byte = self.peek()
        if byte == 0x7B:  # {
            return self.obj(depth + 1)
        if byte == 0x5B:  # [
            return self.arr(depth + 1)
        if byte == 0x22:  # "
            return self.string()
        if byte == 0x2D or _digit(byte):  # - or digit
            return self.number()
        if byte in (0x4E, 0x49):  # N or I: NaN, Infinity
            raise Refused("float", f"constant at {self.pos}")
        for word, val in ((b"true", True), (b"false", False), (b"null", None)):
            if self.data.startswith(word, self.pos):
                self.pos += len(word)
                return val
        if byte == -1:
            raise Refused("non_canonical", f"end of input at {self.pos}")
        raise Refused("non_canonical", f"unexpected byte at {self.pos}")

    def obj(self, depth):
        if depth > self.max_depth:
            raise Refused("limit", f"depth at {self.pos}")
        self.pos += 1
        out = {}
        previous = None
        self.gap()
        if self.peek() == 0x7D:  # }
            self.pos += 1
            return out
        while True:
            self.gap()
            if self.peek() != 0x22:
                raise Refused("non_canonical", f"expected key at {self.pos}")
            at = self.pos
            key = self.string()
            raw = key.encode("ascii")
            if previous is not None and raw <= previous:
                raise Refused("non_canonical", f"key order at {at}")
            previous = raw
            self.gap()
            self.expect(0x3A)  # :
            out[key] = self.value(depth)
            self.gap()
            byte = self.peek()
            if byte == 0x2C:  # ,
                self.pos += 1
                continue
            if byte == 0x7D:  # }
                self.pos += 1
                return out
            raise Refused("non_canonical", f"expected , or }} at {self.pos}")

    def arr(self, depth):
        if depth > self.max_depth:
            raise Refused("limit", f"depth at {self.pos}")
        self.pos += 1
        out = []
        self.gap()
        if self.peek() == 0x5D:  # ]
            self.pos += 1
            return out
        while True:
            out.append(self.value(depth))
            self.gap()
            byte = self.peek()
            if byte == 0x2C:
                self.pos += 1
                continue
            if byte == 0x5D:
                self.pos += 1
                return out
            raise Refused("non_canonical", f"expected , or ] at {self.pos}")

    def string(self):
        self.pos += 1
        chars = []
        while True:
            found = _STRING_SPECIAL.search(self.data, self.pos)
            end = found.start() if found else len(self.data)
            if end > self.pos:
                chars.append(self.data[self.pos:end].decode("ascii"))
                self.pos = end
            byte = self.peek()
            if byte == -1:
                raise Refused("non_canonical", f"unterminated string at {self.pos}")
            if byte == 0x22:
                self.pos += 1
                return "".join(chars)
            if byte == 0x5C:  # backslash
                follow = self.data[self.pos + 1] if self.pos + 1 < len(self.data) else -1
                if follow in (0x22, 0x5C):
                    chars.append(chr(follow))
                    self.pos += 2
                    continue
                if follow == 0x75:  # u
                    raise Refused("non_ascii", f"unicode escape at {self.pos}")
                raise Refused("non_canonical", f"escape at {self.pos}")
            if byte in (0x09, 0x0A, 0x0D):  # only a lenient file lets these reach a string
                raise Refused("non_ascii", f"byte at {self.pos}")
            chars.append(chr(byte))
            self.pos += 1

    def number(self):
        start = self.pos
        negative = self.peek() == 0x2D
        if negative:
            self.pos += 1
            if self.peek() == 0x49:  # -Infinity
                raise Refused("float", f"constant at {start}")
        if not _digit(self.peek()):
            raise Refused("non_canonical", f"number at {start}")
        if self.peek() == 0x30:  # 0
            self.pos += 1
            if _digit(self.peek()):
                raise Refused("non_canonical", f"leading zero at {start}")
        else:
            while _digit(self.peek()):
                self.pos += 1
        if self.peek() in (0x2E, 0x65, 0x45):  # . e E
            raise Refused("float", f"number at {start}")
        text = self.data[start:self.pos]
        if text == b"-0":
            raise Refused("non_canonical", f"negative zero at {start}")
        if len(text) - (1 if negative else 0) > 16:
            raise Refused("limit", f"integer range at {start}")
        value = int(text)
        if value > MAX_INT or value < -MAX_INT:
            raise Refused("limit", f"integer range at {start}")
        return value


def parse(data, *, lenient=False, max_depth=MAX_DEPTH):
    """Decode OCJ bytes into Python values, refusing anything outside the domain."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("OCJ is parsed from bytes")
    data = bytes(data)
    for index, byte in enumerate(data):
        if 0x20 <= byte <= 0x7E:
            continue
        if lenient and byte in _WS_LENIENT:
            continue
        raise Refused("non_ascii", f"byte at {index}")
    if not data:
        raise Refused("non_canonical", "empty input")
    parser = _Parser(data, lenient, max_depth)
    value = parser.value(0)
    parser.gap()
    if parser.pos != len(data):
        raise Refused("non_canonical", f"trailing bytes at {parser.pos}")
    return value


def _emit(value, depth, out):
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        if value > MAX_INT or value < -MAX_INT:
            raise Refused("limit", "integer range")
        out.append(str(value))
    elif isinstance(value, float):
        raise Refused("float", "a float has no canonical form")
    elif isinstance(value, str) and _STRING_PLAIN.fullmatch(value):
        out.append('"')
        out.append(value)
        out.append('"')
    elif isinstance(value, str):
        out.append('"')
        for char in value:
            code = ord(char)
            if code == 0x22:
                out.append('\\"')
            elif code == 0x5C:
                out.append("\\\\")
            elif 0x20 <= code <= 0x7E:
                out.append(char)
            else:
                raise Refused("non_ascii", "character outside printable ASCII")
        out.append('"')
    elif isinstance(value, (list, tuple)):
        if depth + 1 > MAX_DEPTH:
            raise Refused("limit", "depth")
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _emit(item, depth + 1, out)
        out.append("]")
    elif isinstance(value, dict):
        if depth + 1 > MAX_DEPTH:
            raise Refused("limit", "depth")
        for key in value:
            if not isinstance(key, str):
                raise Refused("non_canonical", "object key is not a string")
        keys = sorted(value, key=lambda k: k.encode("ascii", "backslashreplace"))
        out.append("{")
        for index, key in enumerate(keys):
            if index:
                out.append(",")
            _emit(key, depth + 1, out)
            out.append(":")
            _emit(value[key], depth + 1, out)
        out.append("}")
    else:
        raise Refused("non_canonical", f"no canonical form for {type(value).__name__}")


def emit(value):
    """Encode a value as OCJ bytes; refuse what has no canonical form."""
    out = []
    _emit(value, 0, out)
    return "".join(out).encode("ascii")


def pack_bulk(kind, values):
    """Encode integers as a compact ``"<type>:<hex>"`` string, little-endian two's complement."""
    if kind not in BULK_TYPES:
        raise Refused("bad_request", "unknown bulk type")
    width, signed = BULK_TYPES[kind]
    bits = 8 * width
    low = -(1 << (bits - 1)) if signed else 0
    high = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    parts = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < low or value > high:
            raise Refused("bad_request", "bulk value out of range")
        raw = value & ((1 << bits) - 1)
        for _ in range(width):
            byte = raw & 0xFF
            parts.append(_HEX[byte >> 4] + _HEX[byte & 0x0F])
            raw >>= 8
    return kind + ":" + "".join(parts)


def unpack_bulk(text):
    """Decode a ``"<type>:<hex>"`` string; returns ``(kind, values)``."""
    if not isinstance(text, str) or ":" not in text:
        raise Refused("bad_request", "not a bulk string")
    kind, digits = text.split(":", 1)
    if kind not in BULK_TYPES:
        raise Refused("bad_request", "unknown bulk type")
    width, signed = BULK_TYPES[kind]
    if len(digits) % (2 * width):
        raise Refused("bad_request", "bulk length")
    for char in digits:
        if char not in _HEX:
            raise Refused("bad_request", "bulk hex")
    bits = 8 * width
    values = []
    for start in range(0, len(digits), 2 * width):
        raw = 0
        for offset in range(width - 1, -1, -1):
            pair = digits[start + 2 * offset: start + 2 * offset + 2]
            raw = (raw << 8) | int(pair, 16)
        if signed and raw >> (bits - 1):
            raw -= 1 << bits
        values.append(raw)
    return kind, values
