#!/usr/bin/env python3
"""Measure what one unit of the componion's work costs on this machine, in the reference and the native core.

Machine only. The ladder never runs this script and no contract does: a
figure from a container or a loaded shared machine is an indication, not a
measurement, and none is written into the tree. Run it on the machine the
figure is claimed for, at rest, and keep its output beside the claim.

Three cases on each law (the fixture law and the full law), each timed in
both engines when the native core is built and answers for the reference's
world:

* ``awake_day`` -- one awake day of a garden being, from its state at the
  start of its third day;
* ``dormant_year`` -- a windowsill being asleep from a drought, 365 days
  from its state on day 30, on the fast path;
* ``decade`` -- 3650 days of a garden being with a water a week, from its
  genesis, in one call.

For each case and engine the call is repeated (``--repeat``, default 5) and
the best and median wall times are reported with the work and overhead the
engine counted, and the best time's nanoseconds per unit of work. The
native rows read ``owed`` when the core is absent or stale
(``scripts/build_oo_core.sh``). The output is one JSON document on standard
output, every figure under ``"source": "measured"``.

Usage: ``python3 scripts/allium_bench.py [--repeat N] [--case NAME ...]``.
"""

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opti_oignon.allium import engine as seam  # noqa: E402
from opti_oignon.allium import wire  # noqa: E402
from opti_oignon.allium.ref import protocol  # noqa: E402

DAY = 1440
MAX_INT = (1 << 53) - 1
CASES = ("awake_day", "dormant_year", "decade")
LAWS = ("fixture", "v0_1")


def _author():
    """The golden lives' author, loaded by path for its builder of beings."""
    spec = importlib.util.spec_from_file_location("_allium_author_life", ROOT / "scripts" / "allium_author_life.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native():
    """The native core's call, or the reason it is owed."""
    try:
        from opti_oignon import native
    except ImportError:
        return None, "the native loader is not importable"
    module = native.load()
    if module is None or getattr(module, "allium_call", None) is None:
        return None, "the native core is not built here (scripts/build_oo_core.sh)"
    if not seam.handshake(module):
        return None, "the native core answers for another world, a stale build (scripts/build_oo_core.sh)"
    return module.allium_call, None


def _ask(data):
    answer = wire.parse(protocol.call(data))
    if "refused" in answer:
        raise SystemExit(f"the reference refused a bench request: {answer}")
    return answer


def _request(life, state, to, facts):
    return {"budget": MAX_INT, "facts": facts, "genesis": life.genesis, "op": "advance", "state": state,
            "to": to, "v": 1}


def _case(author, law, name):
    """The request bytes a case times; its starting state is reached first, untimed, by the reference."""
    index = {"awake_day": 100, "dormant_year": 101, "decade": 102}[name] + (10 if law == "v0_1" else 0)
    if name == "dormant_year":
        life = author.Life(law, index, weather="windowsill", hemisphere="north", band="medium", tz=0, trace=False)
        start = 30 * DAY
        origin = _ask(wire.emit(_request(life, None, start, [])))["state"]
        if not origin["organs"]["stage"]["dormant"]:
            raise SystemExit(f"{law} {name}: the being is not asleep on day 30")
        return wire.emit(_request(life, origin, start + 365 * DAY, []))
    life = author.Life(law, index, weather="garden", hemisphere="north", band="medium", tz=0, trace=False)
    if name == "awake_day":
        start = 2 * DAY
        origin = _ask(wire.emit(_request(life, None, start, [])))["state"]
        if origin["organs"]["stage"]["dormant"]:
            raise SystemExit(f"{law} {name}: the being sleeps on day 2")
        return wire.emit(_request(life, origin, start + DAY, []))
    for week in range(1, 3650 // 7):
        life.act(week * 7, 8 * 60 + week % 600, "water")
    return wire.emit(_request(life, None, 3650 * DAY, life.facts))


def _time(call, data, repeat):
    times = []
    answer = None
    for _ in range(repeat):
        began = time.perf_counter_ns()
        out = bytes(call(data))
        times.append(time.perf_counter_ns() - began)
        answer = wire.parse(out)
    if "refused" in answer:
        raise SystemExit(f"a bench request was refused: {answer}")
    times.sort()
    best, median = times[0], times[len(times) // 2]
    work = answer["work"]
    return {"best_ns": best, "median_ns": median, "ns_per_unit": best // work if work else None,
            "overhead": answer["overhead"], "repeat": repeat, "request_bytes": len(data), "work": work}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=5, help="calls per case and engine (default 5)")
    parser.add_argument("--case", action="append", choices=CASES, help="a case to time (default: all three)")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat is at least 1")
    author = _author()
    native_call, owed = _native()
    rows = {}
    for law in LAWS:
        for name in args.case or CASES:
            data = _case(author, law, name)
            row = {"reference": _time(protocol.call, data, args.repeat)}
            row["native"] = {"owed": owed} if native_call is None else _time(native_call, data, args.repeat)
            rows[f"{law}/{name}"] = row
    report = {
        "cases": rows,
        "host": {"machine": platform.machine(), "node": platform.node(), "python": platform.python_version(),
                 "system": platform.system()},
        "source": "measured",
        "taken_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
