#!/usr/bin/env python3
"""CI guard: summary fidelity and needle recovery against a written floor.

The floor is an engagement, so it lives in a committed file next to this
guard rather than in a constant inside it: raising or lowering it is a
reviewed change to that file, never a silent edit to code. This guard only
READS the engagement, takes a fresh measurement with the deterministic
harness, and compares.

Exit codes:
  0  measured at or above both floors
  1  measured below a floor
  2  the engagement file or the measurement itself is unusable

The measurement is the harness's deterministic path: the shipped micro
fixture through the real tier machinery, and a needle sweep against a
throwaway conversation store under a temporary directory. No model is
consulted and no network is touched.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

DEFAULT_THRESHOLD_PATH = Path(__file__).resolve().parent / "summary_fidelity.threshold"
REQUIRED_KEYS = ("fidelity_floor", "needle_floor")


def read_threshold(path: Path) -> dict[str, float]:
    """Parse the engagement file: ``key = value`` lines, comments ignored."""
    if not path.is_file():
        raise ValueError(f"threshold file not found: {path}")
    floors: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        try:
            floors[key.strip()] = float(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"threshold file {path} has a non-numeric value on line: "
                f"{raw_line!r}"
            ) from exc
    missing = [key for key in REQUIRED_KEYS if key not in floors]
    if missing:
        raise ValueError(
            f"threshold file {path} does not state: {', '.join(missing)}"
        )
    return floors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=Path,
        default=DEFAULT_THRESHOLD_PATH,
        help="engagement file stating fidelity_floor and needle_floor",
    )
    args = parser.parse_args(argv)

    try:
        floors = read_threshold(args.threshold)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        from opti_oignon.agent_eval import fidelity

        fidelity_report = fidelity.run_fidelity()
        with tempfile.TemporaryDirectory() as tmp:
            needle_report = fidelity.run_needle_sweep(Path(tmp))
    except Exception as exc:
        print(f"ERROR: the measurement itself failed: {exc}")
        return 2

    failures = 0

    measured = fidelity_report.overall_ratio
    floor = floors["fidelity_floor"]
    print(f"fidelity: measured {measured:.4f} against floor {floor:.4f}")
    if measured < floor:
        failures += 1
        for result in fidelity_report.facts:
            if not result.retained:
                print(
                    f"  LOST {result.conversation_id}/{result.fact_id}: "
                    f"missing {result.missing_keywords}"
                )

    recovered = needle_report.recovery_ratio
    needle_floor = floors["needle_floor"]
    print(f"needle: recovered {recovered:.4f} against floor {needle_floor:.4f}")
    if recovered < needle_floor:
        failures += 1
        for case in needle_report.cases:
            if not case.recovered:
                print(
                    f"  MISSED haystack of {case.haystack_size} at depth "
                    f"{case.depth}"
                )

    if failures:
        print("FAIL: the measurement fell below the written floor")
        return 1
    print("OK: measured at or above the written floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
