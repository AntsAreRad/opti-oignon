#!/usr/bin/env python3
"""CI guard: adversarial defense hold against a written floor.

The floor is an engagement, so it lives in a committed file next to this
guard rather than in a constant inside it: raising or lowering it is a
reviewed change to that file, never a silent edit to code. This guard only
READS the engagement, takes a fresh measurement with the deterministic
harness, and compares.

Exit codes:
  0  measured at or above the floor
  1  measured below the floor
  2  the engagement file or the measurement itself is unusable

The measurement is the harness's deterministic path: the shipped
adversarial corpus, driven through the shipped obfuscation strategies at
the real defense modules, judged by rules. No model is consulted, no
network is touched, and no attack payload is ever executed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

DEFAULT_THRESHOLD_PATH = Path(__file__).resolve().parent / "red_team.threshold"
REQUIRED_KEYS = ("defense_floor",)


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
        help="engagement file stating defense_floor",
    )
    args = parser.parse_args(argv)

    try:
        floors = read_threshold(args.threshold)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        from opti_oignon.agent_eval import red_team

        report = red_team.run_red_team()
    except Exception as exc:
        print(f"ERROR: the measurement itself failed: {exc}")
        return 2

    measured = report.held_ratio
    floor = floors["defense_floor"]
    print(f"defense hold: measured {measured:.4f} against floor {floor:.4f}")
    for target in sorted(report.per_target):
        print(f"  {target}: {report.per_target[target]:.4f}")

    if measured < floor:
        for result in report.cases:
            if not result.held:
                print(f"  GAVE WAY {result.target}/{result.case_id}: "
                      f"{result.detail}")
        print("FAIL: a defense fell below the written floor")
        return 1

    print("OK: measured at or above the written floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
