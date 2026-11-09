"""The value head: the order of a search's candidate files, learned from their traits.

Each candidate file of a question is a row of traits (what the search's stages and the project's map know of it);
the head is an L2 logistic regression on those traits, centred and scaled (a trait that never moves is scaled by 1),
learned by Newton's method on the weights and the bias at once, the bias not held in; its logits order the
candidates, the higher first, rows of equal logits in the order they came. A trait that never moves in the rows
learned from -- its least and greatest values apart by no more than ``STILL`` of their size, a rounding -- is centred
on its first value and weighs nothing, whatever it reads later; the others are centred on a mean summed around their
first value, so that a sum's rounding never passes for a spread. A head is learned from rows whose label
says whether the file answers the question (1) or not (0), and stays a value: its traits' names, their means and
scales, its weights and bias, its strength, what it learned from and which engine learned it, written and read back
as a record of one shape.

The native core (``rust/oo_core/src/value.rs``) learns and scores the head when it is built and declares this
module's shape; this module does otherwise, in numpy, and stays the reference the core is held to. What the head
refuses is refused here, by name, before either engine is asked: rows, labels, a strength within its bounds, each
trait finite and within ``MAX_ABS``. A head neither engine can learn from rows it admits -- one that does not settle
within ``MAX_STEPS`` steps, a step's system an engine cannot solve -- is refused after, in the same words whichever
engine was asked (the engine's own reason after them)."""
import math
from dataclasses import asdict, dataclass

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

# The shape of a head's record, and of what the native core is handed: a core declaring another is not asked.
FORMAT = 1
ENGINES = ("native", "reference")
MAX_TRAITS = 64
MAX_ROWS = 1_000_000
MAX_STEPS = 300
# The strength's bounds, and a trait's: beyond them the solvers leave the floats they agree on.
MIN_C, MAX_C = 1e-6, 1e6
MAX_ABS = 1e15
# A trait whose values lie apart by no more than this share of their size never moves: a rounding, not a spread.
STILL = 1e-12
# Newton stops once no weight moves by more than this.
TOLERANCE = 1e-7


@dataclass(frozen=True)
class Head:
    traits: tuple
    mean: tuple
    scale: tuple
    weights: tuple
    bias: float
    c: float
    rows: int
    positives: int
    steps: int
    engine: str

    def as_record(self) -> dict:
        record = asdict(self)
        record.update({k: list(v) for k, v in record.items() if isinstance(v, tuple)})
        return {"format": FORMAT, **record}

    @classmethod
    def from_record(cls, record: dict) -> "Head":
        """The head a record holds; a record of another shape, of lengths that do not agree, with a value that is not
        finite, a name twice or an engine unknown is refused by name."""
        fields = set(cls.__dataclass_fields__)
        if (not isinstance(record, dict) or type(record.get("format")) is not int or record["format"] != FORMAT
                or set(record) != fields | {"format"}):
            raise ValueError("a value head's record of another shape")
        traits = _named(record["traits"])
        vectors = [record[k] for k in ("mean", "scale", "weights")]
        if not all(isinstance(v, list) and len(v) == len(traits) for v in vectors):
            raise ValueError("a value head's record whose lengths do not agree with its traits")
        numbers = [x for v in vectors for x in v] + [record["bias"], record["c"]]
        if not all(_finite(x) for x in numbers):
            raise ValueError("a value head's record holds a value that is not finite")
        if any(s <= 0 for s in record["scale"]):
            raise ValueError("a value head's record holds a scale that is not positive")
        if not MIN_C <= record["c"] <= MAX_C:
            raise ValueError(f"a value head's record holds a strength outside [{MIN_C}, {MAX_C}]")
        if record["engine"] not in ENGINES:
            raise ValueError(f"a value head's record names an engine unknown: {record['engine']!r}")
        counts = [record[k] for k in ("rows", "positives", "steps")]
        if (not all(type(k) is int for k in counts) or not 2 <= counts[0] <= MAX_ROWS
                or not 1 <= counts[1] < counts[0] or not 1 <= counts[2] <= MAX_STEPS):
            raise ValueError("a value head's record holds counts that are not counts")
        return cls(traits=traits, mean=tuple(float(x) for x in record["mean"]),
                   scale=tuple(float(x) for x in record["scale"]),
                   weights=tuple(float(x) for x in record["weights"]), bias=float(record["bias"]),
                   c=float(record["c"]), rows=record["rows"], positives=record["positives"], steps=record["steps"],
                   engine=record["engine"])


def _named(traits) -> tuple:
    names = tuple(traits) if isinstance(traits, (list, tuple)) else ()
    if not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError("a value head's traits each need a name")
    if len(set(names)) != len(names):
        raise ValueError("a value head's traits each need a name of their own: a name twice")
    if len(names) > MAX_TRAITS:
        raise ValueError(f"a value head takes at most {MAX_TRAITS} traits, not {len(names)}")
    return names


def _finite(x) -> bool:
    """A number (not a flag) a float holds, finite."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(float(x))
    except OverflowError:
        return False


def _core():
    """(the native core, "") when it serves this module's shape; (None, the reason) otherwise."""
    try:
        from opti_oignon.native import load
    except Exception:  # noqa: BLE001 - absence is the reference alone
        return None, "absent"
    core = load()
    if core is None:
        return None, "absent"
    if getattr(core, "VALUE_FORMAT", None) != FORMAT:
        return None, "another shape"
    return core, ""


def engine() -> str:
    """The engine that would learn and score a head now: "native" or "reference"."""
    return "native" if _core()[0] is not None else "reference"


def _matrix(rows, width: int):
    """The rows as one list of floats, row after row; rows of another width or values not finite refused."""
    flat = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError(f"a row of the value head's traits does not hold {width} traits")
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError("a value head's trait is not a number")
            try:
                flat.append(float(x))
            except OverflowError:
                raise ValueError(f"a value head's trait lies beyond {MAX_ABS:g}") from None
    if not all(math.isfinite(x) for x in flat):
        raise ValueError("a value head's trait is not finite")
    if not all(abs(x) <= MAX_ABS for x in flat):
        raise ValueError(f"a value head's trait lies beyond {MAX_ABS:g}")
    return flat


def fit(rows, labels, traits, c: float = 1.0) -> Head:
    """The head learned from ``rows`` (one per candidate, one value per trait) and ``labels`` (1: the candidate
    answers; 0: it does not), with the strength ``c`` (the larger, the less the weights are held in)."""
    names = _named(traits)
    rows = list(rows)
    labels = list(labels)
    if not rows:
        raise ValueError("a value head learns from no row")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"a value head learns from at most {MAX_ROWS} rows, not {len(rows)}")
    if len(labels) != len(rows):
        raise ValueError(f"a value head needs one label per row: {len(labels)} labels for {len(rows)} rows")
    if not _finite(c) or not MIN_C <= float(c) <= MAX_C:
        raise ValueError(f"a value head's strength must be a number within [{MIN_C}, {MAX_C}]")
    flat = _matrix(rows, len(names))
    if not all(not isinstance(y, bool) and isinstance(y, (int, float)) and y in (0, 1) for y in labels):
        raise ValueError("a value head's label is 0 or 1")
    positives = sum(1 for y in labels if y == 1)
    if positives in (0, len(labels)):
        raise ValueError("a value head cannot learn from one class alone: it needs rows labelled 1 and rows "
                         "labelled 0")
    ys = [float(y) for y in labels]
    core, _why = _core()
    try:
        if core is not None:
            mean, scale, weights, bias, steps, settled = core.value_fit(len(names), flat, ys, float(c), MAX_STEPS)
            used = "native"
        else:
            mean, scale, weights, bias, steps, settled = _reference_fit(len(names), flat, ys, float(c), MAX_STEPS)
            used = "reference"
    except ValueError as exc:
        raise ValueError(f"a value head could not be learned from these rows: {exc}") from exc
    if not settled:
        raise ValueError(f"a value head did not settle within {MAX_STEPS} steps: a smaller strength holds it in")
    return Head(traits=names, mean=tuple(mean), scale=tuple(scale), weights=tuple(weights), bias=float(bias),
                c=float(c), rows=len(rows), positives=positives, steps=int(steps), engine=used)


def _reference_fit(width: int, flat, ys, c: float, max_steps: int):
    import numpy as np

    X = np.asarray(flat, dtype=np.float64).reshape(-1, width)
    y = np.asarray(ys, dtype=np.float64)
    first = X[0].copy()
    # Summed around the first row: a sum's rounding stays below the spread it measures (reviews 1 and 2).
    mean = first + (X - first).sum(axis=0) / X.shape[0]
    scale = np.sqrt(((X - mean) ** 2).sum(axis=0) / X.shape[0])
    top, low = X.max(axis=0), X.min(axis=0)
    # A trait that never moves, or by a rounding only: centred on its first value, scaled by 1, read as nought.
    still = (top - low) <= STILL * np.maximum(np.abs(top), np.abs(low))
    mean[still] = first[still]
    scale[still] = 1.0
    scale[scale == 0] = 1.0
    centred = (X - mean) / scale
    centred[:, still] = 0.0
    Z = np.hstack([centred, np.ones((X.shape[0], 1))])
    hold = np.full(width + 1, 1.0 / c)
    hold[width] = 0.0
    theta = np.zeros(width + 1)
    steps, settled = 0, False
    for steps in range(1, max_steps + 1):
        # A logit far below zero gives exp() an overflow, and p its limit, 0: as the core computes it.
        with np.errstate(over="ignore"):
            p = 1.0 / (1.0 + np.exp(-(Z @ theta)))
        gradient = Z.T @ (p - y) + hold * theta
        r = p * (1.0 - p)
        H = (Z * r[:, None]).T @ Z + np.diag(hold)
        step = np.linalg.solve(H, gradient)
        theta = theta - step
        if np.abs(step).max() < TOLERANCE:
            settled = True
            break
    if not np.isfinite(theta).all():
        raise ValueError("value head: the head learned is not finite")
    return mean.tolist(), scale.tolist(), theta[:width].tolist(), float(theta[width]), steps, settled


def scores(head: Head, rows) -> list:
    """The logits of ``rows`` under ``head``, one per row."""
    rows = list(rows)
    if not rows:
        return []
    flat = _matrix(rows, len(head.traits))
    core, _why = _core()
    if core is not None:
        logits = list(core.value_scores(list(head.mean), list(head.scale), list(head.weights), head.bias, flat))
    else:
        import numpy as np

        X = np.asarray(flat, dtype=np.float64).reshape(-1, len(head.traits))
        Z = (X - np.asarray(head.mean)) / np.asarray(head.scale)
        logits = (Z @ np.asarray(head.weights) + head.bias).tolist()
    if not all(math.isfinite(v) for v in logits):
        raise ValueError("a value head's logit is not finite: a row reads too far from the rows it learned from")
    return logits


def order(head: Head, rows) -> list:
    """The rows' indices, the highest logit first; rows of equal logits in the order they came."""
    logits = scores(head, rows)
    return sorted(range(len(logits)), key=lambda i: -logits[i])
