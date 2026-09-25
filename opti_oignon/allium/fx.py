"""Fixed point for the companion's engine: Q16.16 in i32, intermediates wide.

The reference. Its Rust twin (``rust/allium/src/fx.rs``) computes in i128,
which none of these primitives can overflow for i32 inputs, so Python's
unbounded integers and the twin agree on every value; both then saturate to
the i32 range the same way.

Every primitive is total. An input outside i32, or outside the primitive's
domain, is brought back to the nearest legal value; a result outside i32
saturates. Each such correction increments ``alarm`` in the caller's
``Work``: a non-zero alarm is an engine bug to find, never a refusal in the
middle of a step, because a refusal would replay identically on every
replay and the being could never advance again. Division is by a positive
divisor only, and floors, as ``//`` and ``div_euclid`` both do; a shift of a
negative value floors too.

No float appears here: a transcendental function answers differently on
different platforms. The sine comes from a frozen table, one file read by
both engines.
"""

checkpoint_before_apply = True

ONE = 1 << 16
HALF = ONE >> 1
I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1
CMAX = 8 * ONE
N_MIN, N_MAX = 1, 4
TAU_MIN, TAU_MAX = 1, 16
STEPS_MAX = 1 << 20


class Work:
    """Units of work done and corrections made; both engines count the same way."""

    __slots__ = ("units", "alarm")

    def __init__(self):
        self.units = 0
        self.alarm = 0


def _in(value, work):
    """An input held to i32; an out-of-range input is corrected and counted."""
    if value > I32_MAX:
        work.alarm += 1
        return I32_MAX
    if value < I32_MIN:
        work.alarm += 1
        return I32_MIN
    return value


def _out(value, work):
    """A result saturated to i32."""
    if value > I32_MAX:
        work.alarm += 1
        return I32_MAX
    if value < I32_MIN:
        work.alarm += 1
        return I32_MIN
    return value


def _held(value, low, high, work):
    """A parameter held to its domain [low, high]; a correction is counted."""
    if value < low:
        work.alarm += 1
        return low
    if value > high:
        work.alarm += 1
        return high
    return value


def _pow_raw(c, n):
    x = c
    for _ in range(n - 1):
        x = (x * c) >> 16
    return x


def mul(a, b, work):
    work.units += 1
    a, b = _in(a, work), _in(b, work)
    return _out((a * b) >> 16, work)


def div(a, b, work):
    work.units += 1
    a, b = _in(a, work), _in(b, work)
    if b <= 0:
        work.alarm += 1
        if a > 0:
            return I32_MAX
        if a < 0:
            return I32_MIN
        return 0
    return _out((a << 16) // b, work)


def pow_(c, n, work):
    work.units += 1
    c = _in(c, work)
    n = _held(n, N_MIN, N_MAX, work)
    return _out(_pow_raw(c, n), work)


def hill_up(c, k, n, work):
    """c^n / (k^n + c^n) in Q16; c >= 0, k in (0, CMAX], n in 1..4."""
    work.units += 1
    c = _held(_in(c, work), 0, I32_MAX, work)
    k = _held(_in(k, work), 1, CMAX, work)
    n = _held(n, N_MIN, N_MAX, work)
    pc = _pow_raw(c, n)
    den = _pow_raw(k, n) + pc
    if den <= 0:
        return 0
    return _out((pc << 16) // den, work)


def hill_down(c, k, n, work):
    return ONE - hill_up(c, k, n, work)


def mm(c, k, work):
    """Michaelis-Menten c / (k + c) in Q16; c >= 0, k in (0, CMAX]."""
    work.units += 1
    c = _held(_in(c, work), 0, I32_MAX, work)
    k = _held(_in(k, work), 1, CMAX, work)
    return _out((c << 16) // (k + c), work)


def sig(a, work):
    """Softsign squashed to [0, ONE]: sig(0) = ONE/2 and sig(-a) = ONE - sig(a) exactly."""
    work.units += 1
    a = _in(a, work)
    magnitude = -a if a < 0 else a
    q = (magnitude * HALF) // (ONE + magnitude)
    if a < 0:
        return HALF - q
    return HALF + q


def sat(x, low, high, work):
    work.units += 1
    x, low, high = _in(x, work), _in(low, work), _in(high, work)
    if low > high:
        work.alarm += 1
        return low
    if x < low:
        return low
    if x > high:
        return high
    return x


def isqrt(x, work):
    import math

    work.units += 1
    x = _held(_in(x, work), 0, I32_MAX, work)
    return math.isqrt(x)


def sin_b(bam, table, work):
    """Q15 sine of a binary angle (1024 per turn), from the frozen table."""
    work.units += 1
    return table[bam & 1023]


def cos_b(bam, table, work):
    work.units += 1
    return table[(bam + 256) & 1023]


def lut(x, table, work):
    """A 257-entry frozen table over [0, ONE], interpolated in integers."""
    work.units += 1
    x = _held(_in(x, work), 0, ONE, work)
    index = x >> 8
    if index >= 256:
        return table[256]
    low = table[index]
    return _out(low + (((table[index + 1] - low) * (x & 255)) >> 8), work)


def decay_table(tau, length):
    """DECAY[0] = ONE; DECAY[k+1] = DECAY[k] - (DECAY[k] >> tau)."""
    out = [ONE]
    for _ in range(length - 1):
        out.append(out[-1] - (out[-1] >> tau))
    return out


def decay_iter(x, tau, steps, work):
    """An ``iter`` field: stepped, never jumped. ``x -= x >> tau``, ``steps`` times."""
    x = _in(x, work)
    tau = _held(tau, TAU_MIN, TAU_MAX, work)
    steps = _held(steps, 0, STEPS_MAX, work)
    for _ in range(steps):
        x -= x >> tau
    work.units += steps
    return x


def decay_lazy(v0, tau, age, length, work):
    """A ``lazy`` field, read at ``age`` steps after its contact: ``(v0 * DECAY[age]) >> 16``."""
    work.units += 1
    v0 = _in(v0, work)
    tau = _held(tau, TAU_MIN, TAU_MAX, work)
    length = _held(length, 1, STEPS_MAX, work)
    age = _held(age, 0, STEPS_MAX, work)
    if age >= length:
        return 0
    factor = ONE
    for _ in range(age):
        if factor == 0:
            break
        factor -= factor >> tau
    return _out((v0 * factor) >> 16, work)


# name -> (function, arity counting only integer arguments)
PRIMITIVES = {
    "mul": (mul, 2),
    "div": (div, 2),
    "pow": (pow_, 2),
    "hill_up": (hill_up, 3),
    "hill_down": (hill_down, 3),
    "mm": (mm, 2),
    "sig": (sig, 1),
    "sat": (sat, 3),
    "isqrt": (isqrt, 1),
    "sin_b": (sin_b, 1),
    "cos_b": (cos_b, 1),
    "lut": (lut, 1),
    "decay_iter": (decay_iter, 3),
    "decay_lazy": (decay_lazy, 4),
}
