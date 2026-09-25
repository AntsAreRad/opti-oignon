#!/usr/bin/env python3
"""Contracts for the companion engine's fixed point: integers only, the same in both engines.

Every quantity the being carries is Q16.16 in an i32, computed wide and
saturated. A float would answer differently on different platforms; an
overflow would wrap in a release build and not in Python. So the primitives
are total, written once in each engine, and compared on the release
artefact the native core ships.

  * FX1 -- every primitive gives the same results, the same alarms and the
    same work in both engines, over an exhaustive grid of edge values and
    100 000 seeded values, against the built artefact.
  * FX2 -- the softsign is exactly symmetric: ``sig(-a) = ONE - sig(a)``.
  * FX3 -- every primitive is total: an input, a parameter or a result
    outside its range saturates identically in both engines and raises the
    alarm; none raises and none refuses.
  * FX4 -- the sine table has one source: the file the reference reads is
    the file the native core embedded, digest for digest, and the one the
    fixture law pins.
  * FX5 -- each decaying field is declared ``iter`` or ``lazy`` in the law
    file and read only its own way: a ``lazy`` field is read through the
    decay table, an ``iter`` field is stepped, and the two forms really
    differ (7, tau 2, one step: 6 stepped, 5 by the table).

Local-only (the public distribution ships no tests). The modules load
through the shared isolation window; the native side needs the artefact that
``scripts/build_oo_core.sh`` builds.
"""

import hashlib
import json
import random
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_window import native_module, open_allium  # noqa: E402

ONE = 1 << 16
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
CMAX = 8 * ONE
EDGES = (
    I32_MIN - 1, I32_MIN, I32_MIN + 1, -(1 << 40), -CMAX, -ONE - 1, -ONE, -2, -1, 0, 1, 2,
    255, 256, ONE - 1, ONE, ONE + 1, CMAX - 1, CMAX, CMAX + 1, I32_MAX - 1, I32_MAX, I32_MAX + 1, 1 << 40,
)
SMALL = (-1, 0, 1, 2, 3, 4, 5, 16, 17)

BUDGET_S = {
    "test_fx1_every_primitive_answers_the_same_in_both_engines_on_edges_and_seeded_values": 2.0,
    "test_fx2_the_softsign_is_exactly_symmetric": 2.0,
    "test_fx3_every_primitive_is_total_and_saturates_the_same_way_with_an_alarm": 2.0,
    "test_fx4_the_sine_table_has_one_source_digest_for_digest": 2.0,
    "test_fx5_each_decaying_field_is_read_only_its_declared_way": 2.0,
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


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _batch(loaded, native, name, args, table=None):
    """One primitive over a batch, in both engines; the answers must be the same bytes."""
    request = {"args": args, "fn": name, "op": "fx", "v": 1}
    if table is not None:
        request["table"] = table
    data = _canonical(request)
    ref = loaded["opti_oignon.allium.ref.protocol"].call(data)
    nat = bytes(native.allium_call(data))
    assert ref == nat, (name, ref[:200], nat[:200])
    answer = json.loads(ref)
    assert "refused" not in answer, (name, answer)
    return answer


def _grid(arity):
    if arity == 1:
        return [[a] for a in EDGES]
    if arity == 2:
        return [[a, b] for a in EDGES for b in EDGES]
    return [[a, b, c] for a in EDGES for b in EDGES for c in SMALL]


def _lut_table():
    return [(i * i * 3) - 40000 for i in range(257)]


# ---------------------------------------------------------------------------
# FX1 -- the same answer, on edges and seeded values
# ---------------------------------------------------------------------------
def test_fx1_every_primitive_answers_the_same_in_both_engines_on_edges_and_seeded_values(engines):
    loaded, native = engines
    fx = loaded["opti_oignon.allium.fx"]
    alarms = 0
    for name in ("mul", "div", "pow", "mm", "sig", "sat", "isqrt", "sin_b", "cos_b", "hill_up", "hill_down"):
        arity = fx.PRIMITIVES[name][1]
        args = _grid(arity)
        if name == "pow":
            args = [[a, n] for a in EDGES for n in SMALL]
        if name == "sat":
            args = [[a, b, c] for a in EDGES for b in EDGES[::3] for c in EDGES[::3]]
        answer = _batch(loaded, native, name, args)
        assert len(answer["out"]) == len(args)
        alarms += answer["alarm"]
    answer = _batch(loaded, native, "lut", [[x] for x in EDGES], table=_lut_table())
    alarms += answer["alarm"]
    lazy = [[v, tau, age, length] for v in EDGES[::2] for tau in SMALL for age in (0, 1, 7, 300) for length in (1, 256)]
    alarms += _batch(loaded, native, "decay_lazy", lazy)["alarm"]
    stepped = [[v, tau, steps] for v in EDGES[::2] for tau in SMALL for steps in (0, 1, 7)]
    alarms += _batch(loaded, native, "decay_iter", stepped)["alarm"]
    assert alarms >= 1, "the edge grid must reach the corrections"
    rnd = random.Random(20260925)
    count = 0
    for name in ("mul", "div", "hill_up", "mm", "sig", "sat", "isqrt"):
        arity = fx.PRIMITIVES[name][1]
        args = []
        for _ in range(100000 // 7 + 1):
            row = [rnd.randrange(I32_MIN, I32_MAX + 1) for _ in range(arity)]
            if name == "hill_up":
                row = [rnd.randrange(0, 4 * CMAX), rnd.randrange(1, CMAX + 1), rnd.randrange(1, 5)]
            args.append(row)
        _batch(loaded, native, name, args)
        count += len(args)
    assert count >= 100000


# ---------------------------------------------------------------------------
# FX2 -- the softsign's symmetry
# ---------------------------------------------------------------------------
def test_fx2_the_softsign_is_exactly_symmetric(engines):
    loaded, native = engines
    rnd = random.Random(2)
    values = [0, 1, 2, ONE, CMAX, I32_MAX] + [rnd.randrange(1, I32_MAX + 1) for _ in range(20000)]
    out = _batch(loaded, native, "sig", [[a] for a in values] + [[-a] for a in values])["out"]
    half = len(values)
    assert out[0] == ONE // 2, "sig(0) is one half"
    for positive, negative in zip(out[:half], out[half:]):
        assert negative == ONE - positive, (positive, negative)


# ---------------------------------------------------------------------------
# FX3 -- totality, saturation and the alarm
# ---------------------------------------------------------------------------
def test_fx3_every_primitive_is_total_and_saturates_the_same_way_with_an_alarm(engines):
    loaded, native = engines
    fx = loaded["opti_oignon.allium.fx"]
    wild = (I32_MIN - 1, I32_MAX + 1, -(1 << 52), 1 << 52)
    table = _lut_table()
    for name, (function, arity) in fx.PRIMITIVES.items():
        rows = [[w] * arity for w in wild]
        if name in ("pow", "hill_up", "hill_down"):
            rows += [[ONE] * (arity - 1) + [9], [ONE] * (arity - 1) + [-3]]
        if name in ("decay_iter", "decay_lazy"):
            rows = [[I32_MAX + 1, 99, 1] + ([256] if arity == 4 else []), [5, -4, 2] + ([-9] if arity == 4 else [])]
        for row in rows:
            work = fx.Work()
            if name in ("sin_b", "cos_b"):
                value = function(row[0], [0] * 1024, work)
            elif name == "lut":
                value = function(row[0], table, work)
            else:
                value = function(*row, work)
            assert I32_MIN <= value <= I32_MAX, (name, row, value)
            if name not in ("sin_b", "cos_b"):
                assert work.alarm >= 1, (name, row, "a correction raises the alarm")
        answer = _batch(loaded, native, name, rows, table=table if name == "lut" else None)
        if name not in ("sin_b", "cos_b"):
            assert answer["alarm"] >= len(rows), (name, answer)
    wide = _batch(loaded, native, "mul", [[I32_MAX, I32_MAX], [I32_MIN, I32_MAX], [I32_MIN, I32_MIN]])
    assert wide["out"] == [I32_MAX, I32_MIN, I32_MAX] and wide["alarm"] == 3, wide


# ---------------------------------------------------------------------------
# FX4 -- one sine table
# ---------------------------------------------------------------------------
def test_fx4_the_sine_table_has_one_source_digest_for_digest(engines):
    loaded, native = engines
    lawfiles = loaded["opti_oignon.allium.lawfiles"]
    reference = lawfiles.digest(lawfiles.table("sine_q15_v1"))
    embedded = json.loads(bytes(native.allium_engine()))["tables"]["sine_q15_v1"]
    pinned = lawfiles.law("fixture")["tables"]["sine_q15"]
    assert reference == embedded == pinned, (reference, embedded, pinned)
    entries = lawfiles.sine()
    assert len(entries) == 1024 and entries[0] == 0 and entries[256] == 32767 and entries[768] == -32767
    through = _batch(loaded, native, "sin_b", [[i] for i in range(1024)])["out"]
    assert through == entries
    assert hashlib.sha256(json.dumps(lawfiles.table("sine_q15_v1"), sort_keys=True, separators=(",", ":")).encode()).hexdigest() == reference


# ---------------------------------------------------------------------------
# FX5 -- iter or lazy, never both
# ---------------------------------------------------------------------------
def test_fx5_each_decaying_field_is_read_only_its_declared_way(engines):
    loaded, native = engines
    fx = loaded["opti_oignon.allium.fx"]
    decays = loaded["opti_oignon.allium.lawfiles"].law("fixture")["decays"]
    kinds = {field["kind"] for field in decays.values()}
    assert kinds == {"iter", "lazy"}, "the fixture law declares one field of each kind"
    for name, field in sorted(decays.items()):
        tau = field["tau"]
        if field["kind"] == "lazy":
            length = field["length"]
            table = fx.decay_table(tau, length)
            rows = [[v, tau, age, length] for v in (7, 1000, ONE, -ONE) for age in range(length + 2)]
            out = _batch(loaded, native, "decay_lazy", rows)["out"]
            for (v, _tau, age, _length), got in zip(rows, out):
                want = (v * table[age]) >> 16 if age < length else 0
                assert got == want, (name, v, age, got, want)
        else:
            rows = [[v, tau, steps] for v in (7, 1000, ONE, -ONE) for steps in range(12)]
            out = _batch(loaded, native, "decay_iter", rows)["out"]
            for (v, _tau, steps), got in zip(rows, out):
                x = v
                for _ in range(steps):
                    x -= x >> tau
                assert got == x, (name, v, steps, got, x)
    witness = _batch(loaded, native, "decay_iter", [[7, 2, 1]])["out"] + _batch(loaded, native, "decay_lazy", [[7, 2, 1, 256]])["out"]
    assert witness == [6, 5], "a stepped field and a table field differ: they must never be mixed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
