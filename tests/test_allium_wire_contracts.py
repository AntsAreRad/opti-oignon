#!/usr/bin/env python3
"""Contracts for the componion engine's wire and randomness: one encoding, two engines, one answer.

Everything the being hashes travels in onion canonical JSON (OCJ), and every
random draw is addressed by content. The Python reference and the Rust twin
in the native core must accept, refuse and emit the same bytes; the standard
library's ``json`` is the third, independent opinion on what is canonical.

  * AW1 -- over 20 000 seeded documents both engines re-emit exactly what
    ``json.dumps`` emits, and over a hostile corpus generated at run time
    they answer with the same bytes and accept exactly the canonical
    requests: everything else is refused.
  * AW2 -- a float, a negative zero, a leading zero, a unicode escape and a
    byte outside printable ASCII are each refused by their own name.
  * AW3 -- integers reach +-(2^53 - 1) and stop there: one past, or a
    number too long to read, is refused as a limit, never raised.
  * AW4 -- compact integer arrays round-trip with their type in both
    engines, as lowercase hex.
  * AW5 -- the keyed streams draw the committed golden words in both
    engines, and ``below(n)`` rejects exactly the words that would bias it.
  * AW6 -- every limit (input, depth, body, state, items, steps) is refused
    by its name, at its boundary.
  * AW7 -- the refusal codes form a closed set, the same in both engines,
    and every refusal met above carries one of them.

Local-only (the public distribution ships no tests). The modules load
through the shared isolation window; the native side needs the artefact that
``scripts/build_oo_core.sh`` builds.
"""

import json
import random
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_window import native_module, open_allium  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "allium_golden" / "v1" / "chassis.json"
MAX_INT = (1 << 53) - 1
CODES = (
    "non_canonical", "float", "non_ascii", "limit", "unknown_op", "bad_request",
    "unknown_law", "chain", "bad_fact", "budget", "engine_panic",
)

# Seconds each contract may take on this machine, read back by the ladder from the junit file.
BUDGET_S = {
    "test_aw1_both_engines_emit_the_standard_encoding_and_refuse_every_non_canonical_request": 2.0,
    "test_aw2_each_defect_is_refused_by_its_own_name": 2.0,
    "test_aw3_integers_stop_at_two_to_the_fifty_three_minus_one": 2.0,
    "test_aw4_compact_arrays_round_trip_with_their_type_as_lowercase_hex": 2.0,
    "test_aw5_streams_draw_the_golden_words_and_below_rejects_exactly_the_biasing_words": 2.0,
    "test_aw6_every_limit_is_refused_by_name_at_its_boundary": 2.0,
    "test_aw7_the_refusal_codes_are_a_closed_set_shared_by_both_engines": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def engines():
    loaded, restore = open_allium()
    try:
        yield loaded, native_module(loaded)
    finally:
        restore()


def _pair(loaded, native, request):
    """Both engines' answers to one request; they must be the same bytes."""
    ref = loaded["opti_oignon.allium.ref.protocol"].call(request)
    nat = bytes(native.allium_call(request))
    assert ref == nat, (request[:120], ref[:200], nat[:200])
    return json.loads(ref)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _documents(seed, count):
    rnd = random.Random(seed)
    alphabet = [chr(c) for c in range(0x20, 0x7F)]
    edges = (0, 1, -1, MAX_INT, -MAX_INT, 1 << 31, -(1 << 31))

    def text():
        return "".join(rnd.choice(alphabet) for _ in range(rnd.randrange(8)))

    def value(depth):
        roll = rnd.randrange(7 if depth < 4 else 5)
        if roll == 0:
            return None
        if roll == 1:
            return rnd.randrange(2) == 1
        if roll == 2:
            return rnd.choice(edges) if rnd.randrange(3) == 0 else rnd.randrange(-MAX_INT, MAX_INT + 1)
        if roll in (3, 4):
            return text()
        if roll == 5:
            return [value(depth + 1) for _ in range(rnd.randrange(4))]
        return {text(): value(depth + 1) for _ in range(rnd.randrange(4))}

    return [value(0) for _ in range(count)]


def _depth(value):
    if isinstance(value, dict):
        return 1 + max((_depth(v) for v in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_depth(v) for v in value), default=0)
    return 0


def _in_domain(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return -MAX_INT <= value <= MAX_INT
    if isinstance(value, list):
        return all(_in_domain(v) for v in value)
    if isinstance(value, dict):
        return all(_in_domain(v) for v in value.values())
    return False


def _stdlib_accepts_echo(data):
    """The standard library's opinion: is ``data`` a canonical, valid echo request?"""
    if any(not 0x20 <= byte <= 0x7E for byte in data):
        return False

    def refuse(_text):
        raise ValueError("no float")

    def pairs(items):
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)

    try:
        value = json.loads(data.decode("ascii"), parse_float=refuse, parse_constant=refuse, object_pairs_hook=pairs)
    except ValueError:
        return False
    if not _in_domain(value) or _depth(value) > 16 or _canonical(value) != data:
        return False
    return (
        isinstance(value, dict)
        and sorted(value) == ["doc", "op", "v"]
        and value["op"] == "echo"
        and type(value["v"]) is int
        and value["v"] == 1
    )


# ---------------------------------------------------------------------------
# AW1 -- the standard encoding, and nothing else
# ---------------------------------------------------------------------------
def test_aw1_both_engines_emit_the_standard_encoding_and_refuse_every_non_canonical_request(engines):
    loaded, native = engines
    wire = loaded["opti_oignon.allium.wire"]
    ref = loaded["opti_oignon.allium.ref.protocol"]
    documents = _documents(20260925, 20000)
    for doc in documents:
        canonical = _canonical(doc)
        assert wire.emit(doc) == canonical, doc
        assert wire.parse(canonical) == doc
    requests = [b'{"doc":' + _canonical(doc) + b',"op":"echo","v":1}' for doc in documents[:2000]]
    for request in requests:
        assert bytes(native.allium_call(request)) == ref.call(request) == request[:-19] + b"}"
    rnd = random.Random(7)
    hostile = b' "\\{}[]:,-0123456789.eEnNtfIu\x7f\x00\t\xc3\xa9'
    accepted = refused = 0
    for request in requests[:1500]:
        mutated = bytearray(request)
        for _ in range(rnd.randint(1, 3)):
            at = rnd.randrange(len(mutated) + 1)
            kind = rnd.randrange(3)
            if kind == 0 and len(mutated) > 1:
                del mutated[min(at, len(mutated) - 1)]
            elif kind == 1:
                mutated.insert(at, rnd.choice(hostile))
            else:
                mutated[min(at, len(mutated) - 1)] = rnd.choice(hostile)
        mutated = bytes(mutated)
        answer = _pair(loaded, native, mutated)
        if _stdlib_accepts_echo(mutated):
            accepted += 1
            assert "refused" not in answer, (mutated, answer)
        else:
            refused += 1
            assert answer.get("refused") in CODES, (mutated, answer)
    assert accepted >= 1 and refused >= 1000, (accepted, refused)
    structured = 0
    for doc in documents:
        if not isinstance(doc, dict) or len(doc) < 2:
            continue
        members = [_canonical(k) + b":" + _canonical(doc[k]) for k in sorted(doc)]
        for broken in (members[:1] + members, members[1:2] + members[:1] + members[2:]):
            request = b'{"doc":{' + b",".join(broken) + b'},"op":"echo","v":1}'
            answer = _pair(loaded, native, request)
            assert answer.get("refused") == "non_canonical" and "key order" in answer["detail"], (request, answer)
            structured += 1
        if structured >= 400:
            break
    assert structured >= 400, "duplicated and swapped keys must be met"


# ---------------------------------------------------------------------------
# AW2 -- each defect by its own name
# ---------------------------------------------------------------------------
def test_aw2_each_defect_is_refused_by_its_own_name(engines):
    loaded, native = engines
    cases = [
        (b"1.5", "float", "number"),
        (b"1e3", "float", "number"),
        (b"-1E3", "float", "number"),
        (b"NaN", "float", "constant"),
        (b"-Infinity", "float", "constant"),
        (b"-0", "non_canonical", "negative zero"),
        (b"01", "non_canonical", "leading zero"),
        (b'"\\u0041"', "non_ascii", "unicode escape"),
        (b'"\\n"', "non_canonical", "escape"),
        (b'"\xc3\xa9"', "non_ascii", "byte"),
        (b'"\x7f"', "non_ascii", "byte"),
        (b"1 ", "non_canonical", "whitespace"),
    ]
    for doc, code, detail in cases:
        answer = _pair(loaded, native, b'{"doc":' + doc + b',"op":"echo","v":1}')
        assert answer["refused"] == code and detail in answer["detail"], (doc, answer)


# ---------------------------------------------------------------------------
# AW3 -- the integer range
# ---------------------------------------------------------------------------
def test_aw3_integers_stop_at_two_to_the_fifty_three_minus_one(engines):
    loaded, native = engines
    for accepted in (MAX_INT, -MAX_INT, 0):
        doc = str(accepted).encode()
        assert _pair(loaded, native, b'{"doc":' + doc + b',"op":"echo","v":1}') == {"doc": accepted}
    for doc in (str(MAX_INT + 1).encode(), str(-MAX_INT - 1).encode(), b"1" * 17, b"9" * 5000):
        answer = _pair(loaded, native, b'{"doc":' + doc + b',"op":"echo","v":1}')
        assert answer["refused"] == "limit" and "integer range" in answer["detail"], (doc[:20], answer)
    too_wide = _pair(loaded, native, b'{"op":"bulk","unpack":"u64:ffffffffffffffff","v":1}')
    assert too_wide == {"detail": "integer range", "refused": "limit"}, "a wider value travels as hex, never as an integer"


# ---------------------------------------------------------------------------
# AW4 -- compact arrays
# ---------------------------------------------------------------------------
def test_aw4_compact_arrays_round_trip_with_their_type_as_lowercase_hex(engines):
    loaded, native = engines
    samples = {
        "u8": [0, 1, 255], "i8": [-128, -1, 0, 127], "u16": [0, 65535], "i16": [-32768, 32767, -2],
        "u32": [0, (1 << 32) - 1], "i32": [-(1 << 31), (1 << 31) - 1, -7], "u64": [0, 1, MAX_INT],
    }
    for kind, values in samples.items():
        packed = _pair(loaded, native, _canonical({"op": "bulk", "pack": {"type": kind, "values": values}, "v": 1}))
        text = packed["text"]
        prefix, digits = text.split(":", 1)
        assert prefix == kind and all(c in "0123456789abcdef" for c in digits), text
        assert len(digits) == len(values) * 2 * {"u8": 1, "i8": 1, "u16": 2, "i16": 2, "u32": 4, "i32": 4, "u64": 8}[kind]
        unpacked = _pair(loaded, native, _canonical({"op": "bulk", "unpack": text, "v": 1}))
        assert unpacked == {"type": kind, "values": values}, (kind, unpacked)
    for bad in ({"type": "u8", "values": [256]}, {"type": "f32", "values": [1]}):
        answer = _pair(loaded, native, _canonical({"op": "bulk", "pack": bad, "v": 1}))
        assert answer["refused"] == "bad_request", answer


# ---------------------------------------------------------------------------
# AW5 -- golden words, and an unbiased below
# ---------------------------------------------------------------------------
def test_aw5_streams_draw_the_golden_words_and_below_rejects_exactly_the_biasing_words(engines):
    loaded, native = engines
    rng = loaded["opti_oignon.allium.rng"]
    golden = json.loads(GOLDEN.read_text(encoding="ascii"))["rng"]
    seed = golden["seed"]
    asks = {
        "stream": {"domain": "golden", "index": 0, "kind": "stream", "n": 8},
        "unit": {"domain": "golden", "index": 3, "kind": "unit", "n": 8},
        "noise": {"domain": "golden", "index": 1, "kind": "noise", "n": 4},
        "key": {"domain": "golden", "index": 0, "kind": "key", "n": 1},
        "below": {"bound": 1000, "domain": "golden", "index": 2, "kind": "below", "n": 8},
    }
    for name, ask in asks.items():
        answer = _pair(loaded, native, _canonical({**ask, "op": "rng", "seed": seed, "v": 1}))
        assert answer["out"] == golden[name], (name, answer)
    for n in range(1, 4097):
        threshold = rng.below_threshold(n)
        assert threshold == (1 << 64) % n and ((1 << 64) - threshold) % n == 0, n
    bound = 9002803354665472  # 2^64 mod bound is close to bound: a word in about 2049 is rejected
    threshold = rng.below_threshold(bound)
    stream = rng.Stream(bytes.fromhex(seed), "bias", 0)
    rejected = sum(1 for _ in range(60000) if stream.next_u64() < threshold)
    assert rejected >= 1, "the corpus must meet at least one rejected word"
    ask = {"bound": bound, "domain": "bias", "index": 0, "kind": "below", "n": 50000, "op": "rng", "seed": seed, "v": 1}
    _pair(loaded, native, _canonical(ask))


# ---------------------------------------------------------------------------
# AW6 -- limits by name
# ---------------------------------------------------------------------------
def test_aw6_every_limit_is_refused_by_name_at_its_boundary(engines):
    loaded, native = engines
    nested = lambda n: "[" * n + "1" + "]" * n  # noqa: E731
    fact = {"being": "0f" * 16, "kind": "sow", "laws": 0, "origin": "a1" * 8, "oseq": 0, "t": 0}
    over_body = {"pad": "x" * 4087}
    at_body = {"pad": "x" * 4086}
    assert len(_canonical(at_body)) == 4096 and len(_canonical(over_body)) == 4097
    seed = "00" * 32
    cases = [
        (b'{"doc":"' + b"x" * (1 << 20) + b'","op":"echo","v":1}', "limit", "input size"),
        (('{"doc":' + nested(16) + ',"op":"echo","v":1}').encode(), "limit", "depth"),
        (('{"doc":"' + "x" * ((1 << 19) - 1) + '","op":"echo","v":1}').encode(), "limit", "state size"),
        (_canonical({"fact": {**fact, "body": over_body}, "op": "fact_id", "v": 1}), "limit", "body size"),
        (_canonical({"args": [[1, 2]] * 100001, "fn": "mul", "op": "fx", "v": 1}), "limit", "items"),
        (_canonical({"args": [[1, 2, 1 << 20], [1, 2, 1]], "fn": "decay_iter", "op": "fx", "v": 1}), "limit", "steps"),
        (_canonical({"domain": "d", "index": 0, "kind": "stream", "n": 100001, "op": "rng", "seed": seed, "v": 1}), "limit", "items"),
    ]
    for request, code, detail in cases:
        answer = _pair(loaded, native, request)
        assert answer.get("refused") == code and detail in answer.get("detail", ""), (request[:60], answer)
    fitting = [
        ('{"doc":' + nested(15) + ',"op":"echo","v":1}').encode(),
        _canonical({"fact": {**fact, "body": at_body}, "op": "fact_id", "v": 1}),
        _canonical({"args": [[1, 2, 1 << 20]], "fn": "decay_iter", "op": "fx", "v": 1}),
    ]
    for request in fitting:
        assert "refused" not in _pair(loaded, native, request), request[:60]


# ---------------------------------------------------------------------------
# AW7 -- a closed set of refusal codes
# ---------------------------------------------------------------------------
def test_aw7_the_refusal_codes_are_a_closed_set_shared_by_both_engines(engines):
    loaded, native = engines
    wire = loaded["opti_oignon.allium.wire"]
    assert tuple(wire.REFUSALS) == CODES, wire.REFUSALS
    info = json.loads(bytes(native.allium_engine()))
    assert tuple(info["refusals"]) == CODES, info["refusals"]
    seed = "00" * 32
    probes = [
        b"", b"[1]", b'{"op":1,"v":1}', b'{"op":"engine","v":2}', b'{"op":"nothing","v":1}',
        b'{"name":"elsewhere","op":"law","v":1}', b'{"fact":1,"op":"fact_id","v":1}',
        b'{"args":[[1,2]],"budget":0,"fn":"mul","op":"fx","v":1}', b'{"args":[],"fn":"cube","op":"fx","v":1}',
        _canonical({"domain": "d", "index": 0, "kind": "dice", "n": 1, "op": "rng", "seed": seed, "v": 1}),
    ]
    seen = set()
    for request in probes:
        answer = _pair(loaded, native, request)
        assert answer.get("refused") in CODES, (request, answer)
        seen.add(answer["refused"])
    expected = {"non_canonical", "bad_request", "unknown_op", "unknown_law", "bad_fact", "budget"}
    assert expected <= seen, seen
    with pytest.raises(ValueError):
        wire.Refused("brand_new", "not in the closed set")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
