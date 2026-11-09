"""Contracts of the Mycelium's value head (``opti_oignon/mycelium/value.py``) and of its native twin in the core
(``rust/oo_core/src/value.rs``).

The head orders a search's candidate files by their traits: an L2 logistic regression learned by Newton's method
on the traits centred and scaled, its logits the order. The core learns and scores it when it is built and declares
the head's shape; the reference in numpy does otherwise; both are held to the same head.

  * VH1 -- THE HEAD IS THE OPTIMUM: at the head learned, the gradient of the regularised cost (each trait's, and the
    bias's) is nought, by the native core and by the reference.
  * VH2 -- THE ORDER: a trait that separates puts the positives first; a trait multiplied by 1000 or moved leaves
    the logits where they were; equal logits keep the rows' order.
  * VH3 -- WHAT THE HEAD REFUSES, BY NAME: no row; rows of unequal lengths, traits not named one by one, labels not
    one per row; a value that is not finite or lies beyond the bound (an integer too large for a float too); a label
    neither 0 nor 1; one class alone; a strength outside its bounds; too many traits; names empty or twice; rows
    scored against another number of traits or beyond the bound; and, in the same words for either engine, a head
    that does not settle in its steps or that an engine cannot learn (review 1).
  * VH4 -- THE TWIN IS THE REFERENCE: with the core built, its weights, bias and logits are the reference's within
    1e-9, its order the same and its steps within one, on fixed draws of 1, 2 and 18 traits, a constant trait, rare
    positives and strengths other than 1; a trait that never moves, whatever its value (0.01 and 0.1 included, which
    sum inexactly), or that moves by a rounding only, a few rows a float step away, alone or beside others, weighs
    nothing in either engine and moves no logit where it reads otherwise (reviews 1 and 2); one that moves by a
    millionth of its size is a trait.
  * VH5 -- WITHOUT THE CORE, THE REFERENCE: a core absent or of another shape leaves the reference serving, and the
    head and ``engine`` say so; the module imports neither numpy nor the core when it is loaded.
  * VH6 -- A HEAD'S RECORD: written and read back, the same head; a record of another shape, of lengths that do not
    agree, with a value that is not finite or a name twice, of counts that are not counts (a flag, more positives than
    rows, more steps than the bound) or a strength beyond its bounds, refused by name.
  * VH7 -- THE CHANGELOG SAYS IT.
  * VH8 -- THE CORE LETS OTHER THREADS RUN WHILE IT LEARNS (review 1): it holds no interpreter lock in a fit.
  * VH9 -- THE CORE REFUSES WHAT DOES NOT HOLD TOGETHER, BY ITSELF (review 1): handed straight, rows that are not whole
    rows, labels not one per row, a label neither 0 nor 1, a value not finite, a strength not positive, a head and
    rows of other traits: a ValueError, never a panic.
"""
import ast
import json
import math
import random
from pathlib import Path

import pytest

from opti_oignon.mycelium import value

BUDGET_S = {
    "test_vh1_the_head_learned_is_the_optimum_of_its_cost": 2.0,
    "test_vh2_the_head_orders_by_its_traits_whatever_their_scale": 2.0,
    "test_vh3_the_head_refuses_by_name_what_it_cannot_learn_or_score": 2.0,
    "test_vh4_the_native_twin_learns_and_scores_the_references_head": 3.0,
    "test_vh5_without_the_core_the_reference_serves_and_says_so": 2.0,
    "test_vh6_a_heads_record_reads_back_the_same_head_and_refuses_the_rest": 2.0,
    "test_vh7_the_changelog_says_the_value_head": 1.0,
    "test_vh8_the_core_lets_other_threads_run_while_it_learns": 4.0,
    "test_vh9_the_core_refuses_by_itself_what_does_not_hold_together": 1.0,
}

ROOT = Path(__file__).resolve().parents[1]


def _draw(seed, n, d, *, positives=0.2, constant=None):
    rs = random.Random(seed)
    rows, labels = [], []
    for i in range(n):
        y = 1 if rs.random() < positives else 0
        rows.append([rs.gauss(0.9 * y if j % 2 == 0 else 0.0, 1.0) * (1 + j) for j in range(d)])
        labels.append(y)
    labels[0], labels[1] = 1, 0
    if constant is not None:
        for row in rows:
            row[constant] = 3.5
    return rows, labels


def _names(d):
    return [f"trait_{j}" for j in range(d)]


def _absent(monkeypatch):
    monkeypatch.setattr(value, "_core", lambda: (None, "absent"))


def _native():
    core, why = value._core()
    assert core is not None, f"oo_core cannot serve the head ({why}): run scripts/build_oo_core.sh on this machine"
    return core


def _gradient(head, rows, labels):
    """The regularised cost's gradient at ``head``: per trait, then the bias's."""
    grad, bias_grad = [0.0] * len(head.traits), 0.0
    for row, y in zip(rows, labels):
        z = [(x - m) / s for x, m, s in zip(row, head.mean, head.scale)]
        p = 1.0 / (1.0 + math.exp(-(sum(a * w for a, w in zip(z, head.weights)) + head.bias)))
        for j, a in enumerate(z):
            grad[j] += a * (p - y)
        bias_grad += p - y
    return [g + w / head.c for g, w in zip(grad, head.weights)], bias_grad


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_vh1_the_head_learned_is_the_optimum_of_its_cost(monkeypatch):
    rows, labels = _draw(1, 300, 6)
    _native()
    for engine in ("native", "reference"):
        if engine == "reference":
            _absent(monkeypatch)
        head = value.fit(rows, labels, _names(6))
        assert head.engine == engine
        grad, bias_grad = _gradient(head, rows, labels)
        assert max(abs(g) for g in grad) < 1e-6, (engine, grad)
        assert abs(bias_grad) < 1e-6, (engine, bias_grad)
        assert head.rows == 300 and head.positives == sum(labels) and head.c == 1.0
        assert any(abs(w) > 0.05 for w in head.weights), "a probe of the optimum that cannot move proves nothing"
    other = value.fit(rows, labels, _names(6), c=0.01)
    assert sum(abs(w) for w in other.weights) < sum(abs(w) for w in head.weights), "a smaller strength holds it in"


def test_vh2_the_head_orders_by_its_traits_whatever_their_scale(monkeypatch):
    rows, labels = _draw(2, 200, 4)
    rows = [row + [5.0 * y + random.Random(i).random()] for i, (row, y) in enumerate(zip(rows, labels))]
    for engine in ("native", "reference"):
        if engine == "reference":
            _absent(monkeypatch)
        head = value.fit(rows, labels, _names(5))
        first = value.order(head, rows)[:sum(labels)]
        assert sorted(first) == sorted(i for i, y in enumerate(labels) if y), engine
        moved = [row[:4] + [row[4] * 1000.0 - 17.0] for row in rows]
        head2 = value.fit(moved, labels, _names(5))
        for a, b in zip(value.scores(head, rows), value.scores(head2, moved)):
            assert _close(a, b), (engine, a, b)
        same = [rows[3], rows[3], rows[3]]
        assert value.order(head, same) == [0, 1, 2]
        assert value.scores(head, []) == [] and value.order(head, []) == []


def _refused(call, word):
    with pytest.raises(ValueError) as caught:
        call()
    assert word in str(caught.value), (word, str(caught.value))


def test_vh3_the_head_refuses_by_name_what_it_cannot_learn_or_score(monkeypatch):
    rows, labels = _draw(3, 40, 3)
    names = _names(3)
    for engine in ("native", "reference"):
        if engine == "reference":
            _absent(monkeypatch)
        _refused(lambda: value.fit([], [], names), "no row")
        _refused(lambda: value.fit(rows[:-1] + [rows[-1][:2]], labels, names), "traits")
        _refused(lambda: value.fit(rows, labels, names[:2]), "traits")
        _refused(lambda: value.fit(rows, labels[:-1], names), "label")
        _refused(lambda: value.fit(rows[:-1] + [[1.0, float("nan"), 2.0]], labels, names), "finite")
        _refused(lambda: value.fit(rows[:-1] + [[1.0, float("inf"), 2.0]], labels, names), "finite")
        _refused(lambda: value.fit(rows, labels[:-1] + [2], names), "0 or 1")
        _refused(lambda: value.fit(rows, [1] * len(rows), names), "one class")
        _refused(lambda: value.fit(rows, [0] * len(rows), names), "one class")
        for c in (0.0, -1.0, float("nan"), float("inf")):
            _refused(lambda c=c: value.fit(rows, labels, names, c=c), "strength")
        wide = value.MAX_TRAITS + 1
        _refused(lambda: value.fit([[0.0] * wide, [1.0] * wide], [0, 1], _names(wide)), "traits")
        _refused(lambda: value.fit(rows, labels, ["a", "", "c"]), "name")
        _refused(lambda: value.fit(rows, labels, ["a", "b", "a"]), "name")
        _refused(lambda: value.fit(rows[:-1] + [[1.0, 1e16, 2.0]], labels, names), "beyond")
        _refused(lambda: value.fit(rows[:-1] + [[1.0, 10 ** 400, 2.0]], labels, names), "beyond")
        for c in (value.MIN_C / 2, value.MAX_C * 2, 1e-310, 10 ** 400):
            _refused(lambda c=c: value.fit(rows, labels, names, c=c), "strength")
        apart = [[float(x)] for x in range(-40, 41) if x]
        monkeypatch.setattr(value, "MAX_STEPS", 2)
        _refused(lambda: value.fit(apart, [1 if r[0] > 0 else 0 for r in apart], ["x"], c=value.MAX_C), "settle")
        monkeypatch.setattr(value, "MAX_STEPS", 300)
        head = value.fit(rows, labels, names)
        _refused(lambda: value.scores(head, [[1.0, 2.0]]), "traits")
        _refused(lambda: value.scores(head, [[1.0, 2.0, -1e16]]), "beyond")
        _refused(lambda: value.scores(head, [[1.0, 2.0, float("nan")]]), "finite")
        _refused(lambda: value.order(head, [[1.0, 2.0, 3.0, 4.0]]), "traits")
    import numpy as np

    def singular(*_args):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(value, "_reference_fit", singular)
    _refused(lambda: value.fit(rows, labels, names), "could not be learned")


def test_vh4_the_native_twin_learns_and_scores_the_references_head(monkeypatch):
    _native()
    cases = [(*_draw(10, 120, 1), 1.0), (*_draw(11, 150, 2), 0.1), (*_draw(12, 400, 18), 1.0),
             (*_draw(13, 200, 5, constant=2), 1.0), (*_draw(14, 500, 7, positives=0.02), 1.0),
             (*_draw(15, 300, 4), 0.01)]
    natives = [value.fit(rows, labels, _names(len(rows[0])), c=c) for rows, labels, c in cases]
    native_scores = [value.scores(h, rows) for h, (rows, _labels, _c) in zip(natives, cases)]
    native_orders = [value.order(h, rows) for h, (rows, _labels, _c) in zip(natives, cases)]
    _absent(monkeypatch)
    for n, (rows, labels, c) in enumerate(cases):
        ref = value.fit(rows, labels, _names(len(rows[0])), c=c)
        twin = natives[n]
        assert twin.engine == "native" and ref.engine == "reference"
        assert len(twin.weights) == len(ref.weights) == len(rows[0]), n
        for a, b in zip([*twin.mean, *twin.scale, *twin.weights, twin.bias], [*ref.mean, *ref.scale, *ref.weights,
                                                                               ref.bias]):
            assert _close(a, b), (n, a, b)
        for a, b in zip(native_scores[n], value.scores(ref, rows)):
            assert _close(a, b), (n, a, b)
        assert native_orders[n] == value.order(ref, rows), n
    assert natives[3].scale[2] == 1.0 and natives[3].weights[2] == 0.0, "a constant trait weighs nothing"
    for n in range(len(cases)):
        assert abs(natives[n].steps - value.fit(*cases[n][:2], _names(len(cases[n][0][0])), c=cases[n][2]).steps) <= 1
    monkeypatch.undo()
    for constant in (0.01, 0.1, 3.5, -7.3):
        rows, labels = _draw(15, 300, 3)
        rows = [row + [constant] for row in rows]
        far = [rows[0][:3] + [constant + 100.0], rows[1][:3] + [constant - 100.0]]
        twin = value.fit(rows, labels, _names(4))
        _absent(monkeypatch)
        ref = value.fit(rows, labels, _names(4))
        monkeypatch.undo()
        for head in (twin, ref):
            assert head.mean[3] == constant and head.scale[3] == 1.0 and head.weights[3] == 0.0, (constant, head)
        assert value.scores(twin, far) == value.scores(twin, [rows[0], rows[1]]), constant
        for a, b in zip(value.scores(twin, far), value.scores(ref, far)):
            assert _close(a, b), (constant, a, b)
    # A trait that moves by a rounding only (a few rows one float step away, review 2) is a trait that never moves,
    # alone or beside others, in both engines; one that moves by a millionth of its size is a trait.
    step = lambda b: math.nextafter(b, 1e300)  # noqa: E731
    for base, other, off, width in ((0.01, step(0.01), 1, 1), (0.1, step(0.1), 150, 1), (0.3, step(0.3), 3, 1),
                                    (3.5, step(3.5), 1, 1), (0.01, step(0.01), 1, 4), (1e10 / 3, step(1e10 / 3), 7, 2),
                                    (1.0, 1.0 + 1e-13, 150, 3)):
        rows, labels = _draw(16, 300, width)
        moved = {1 + k * (299 // off) for k in range(off)}
        rows = [row[:-1] + [other if i in moved else base] for i, row in enumerate(rows)]
        twin = value.fit(rows, labels, _names(width))
        _absent(monkeypatch)
        ref = value.fit(rows, labels, _names(width))
        monkeypatch.undo()
        for head in (twin, ref):
            assert head.mean[-1] == base and head.scale[-1] == 1.0 and head.weights[-1] == 0.0, (base, off, head)
        far = [rows[0][:-1] + [base * 2.0], rows[1][:-1] + [-base]]
        for a, b in zip(value.scores(twin, far), value.scores(ref, far)):
            assert _close(a, b), (base, off, a, b)
    rows, labels = _draw(17, 300, 2)
    rows = [[row[0], 0.01 * (1 + 1e-6 * random.Random(i).gauss(0.0, 1.0))] for i, row in enumerate(rows)]
    twin = value.fit(rows, labels, _names(2))
    _absent(monkeypatch)
    ref = value.fit(rows, labels, _names(2))
    monkeypatch.undo()
    assert twin.scale[1] != 1.0 and ref.scale[1] != 1.0, "a trait that moves by a millionth of its size is a trait"
    for a, b in zip([*twin.mean, *twin.scale, *twin.weights, twin.bias], [*ref.mean, *ref.scale, *ref.weights, ref.bias]):
        assert _close(a, b), (a, b)


def test_vh5_without_the_core_the_reference_serves_and_says_so(monkeypatch):
    rows, labels = _draw(5, 60, 3)
    _absent(monkeypatch)
    assert value.engine() == "reference"
    assert value.fit(rows, labels, _names(3)).engine == "reference"
    from opti_oignon import native

    class Other:
        VALUE_FORMAT = value.FORMAT + 1

    monkeypatch.undo()
    monkeypatch.setattr(native, "load", lambda: Other())
    assert value._core() == (None, "another shape")
    assert value.engine() == "reference" and value.fit(rows, labels, _names(3)).engine == "reference"
    monkeypatch.setattr(native, "load", lambda: None)
    assert value._core() == (None, "absent")
    monkeypatch.undo()
    assert value.engine() == "native", "with the core built here, the twin serves"
    tree = ast.parse((ROOT / "opti_oignon" / "mycelium" / "value.py").read_text(encoding="utf-8"))
    loaded = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            loaded += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            loaded += [f"{'.' * node.level}{node.module or ''}"] + [a.name for a in node.names]
    assert not [m for m in loaded if "numpy" in m or "native" in m], loaded


def test_vh6_a_heads_record_reads_back_the_same_head_and_refuses_the_rest():
    rows, labels = _draw(6, 80, 4)
    head = value.fit(rows, labels, _names(4))
    record = json.loads(json.dumps(head.as_record()))
    assert record["format"] == value.FORMAT
    assert value.Head.from_record(record) == head
    for change, word in (({"format": value.FORMAT + 1}, "shape"), ({"weights": list(head.weights[:3])}, "lengths"),
                         ({"mean": [*head.mean[:3], float("nan")]}, "finite"), ({"bias": float("inf")}, "finite"),
                         ({"traits": ["a", "b", "a", "c"]}, "name"), ({"engine": "elsewhere"}, "engine"),
                         ({"format": True}, "shape"), ({"rows": True}, "counts"), ({"positives": 81}, "counts"),
                         ({"positives": 0}, "counts"), ({"steps": value.MAX_STEPS + 1}, "counts"),
                         ({"bias": 10 ** 400}, "finite"), ({"c": value.MAX_C * 2}, "strength")):
        _refused(lambda change=change: value.Head.from_record({**record, **change}), word)
    _refused(lambda: value.Head.from_record({k: v for k, v in record.items() if k != "scale"}), "shape")


def test_vh7_the_changelog_says_the_value_head():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    added = changelog.split("### Added\n", 1)[1].split("\n### ", 1)[0]
    entry = " ".join(next((e for e in added.split("\n- ") if "VH1-VH9" in e), "").split())
    for said in ("Mycelium", "value head", "native core", "reference"):
        assert said in entry, (said, entry[:300])


def test_vh8_the_core_lets_other_threads_run_while_it_learns():
    """The interpreter's own pace, alone, against its pace while another thread runs the core's fits one after the
    other: a core that held the lock would leave it the interpreter's switch interval (5 ms) between fits of fifty
    milliseconds or more, a tenth of its pace at most."""
    import threading
    import time

    core = _native()
    rs = random.Random(8)
    width, n = 48, 60000
    x = [rs.gauss(0.0, 1.0) for _ in range(width * n)]
    y = [1.0 if rs.random() < 0.3 else 0.0 for _ in range(n)]

    started = time.perf_counter()
    core.value_fit(width, x, y, 1.0, value.MAX_STEPS)
    took = time.perf_counter() - started
    assert took > 0.05, f"a fit lasted {took:.3f} s: too short to tell"
    count, until = 0, time.perf_counter() + 0.2
    started = time.perf_counter()
    while time.perf_counter() < until:
        count += 1
    alone = count / (time.perf_counter() - started)

    def fits():
        until = time.perf_counter() + 0.4
        while time.perf_counter() < until:
            core.value_fit(width, x, y, 1.0, value.MAX_STEPS)

    worker = threading.Thread(target=fits)
    count, started = 0, time.perf_counter()
    worker.start()
    while worker.is_alive():
        count += 1
    beside = count / (time.perf_counter() - started)
    worker.join()
    assert beside > 0.25 * alone, f"{beside:.0f} steps a second beside the fit against {alone:.0f} alone: the core held " \
        "the interpreter's lock"


def test_vh9_the_core_refuses_by_itself_what_does_not_hold_together():
    core = _native()
    good_x, good_y = [0.0, 1.0, 2.0, 3.0], [0.0, 1.0]
    for args in ((0, [], [], 1.0), (2, [0.0, 1.0, 2.0], good_y, 1.0), (2, good_x, [0.0], 1.0),
                 (2, good_x, [0.0, 2.0], 1.0), (2, [0.0, float("nan"), 2.0, 3.0], good_y, 1.0),
                 (2, good_x, good_y, 0.0), (2, good_x, good_y, float("inf"))):
        with pytest.raises(ValueError) as caught:
            core.value_fit(*args, value.MAX_STEPS)
        assert "value head" in str(caught.value), (args, str(caught.value))
    for args in (([0.0], [1.0], [1.0, 2.0], 0.0, [1.0, 2.0]), ([0.0, 0.0], [1.0, 1.0], [1.0, 2.0], 0.0, [1.0, 2.0, 3.0])):
        with pytest.raises(ValueError) as caught:
            core.value_scores(*args)
        assert "value head" in str(caught.value), (args, str(caught.value))
