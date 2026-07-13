#!/usr/bin/env python3
"""Measure the flat and native tool transcripts, and report what to allow.

The chat tool loop can replay executed tool calls to the model in two
shapes: the historical "flat" reconstruction (every result folded as text
into a rebuilt user message) and the "native" transcript (assistant
tool_calls echoes plus role "tool" messages -- the shape function-calling
models are trained on). Native is off by default and opens only for the
models named in the ``tool_transcript_models`` allowlist, so turning it on
is a deliberate, per-model decision.

This script gives that decision an evidence base. It runs both shapes over
the same scripted task and prints the measured difference, then reports
which installed models the runtime considers capable of native tool
calling. It NEVER edits the configuration: the last section prints the
exact lines to paste, and stopping there is the point.

Usage
-----
    python3 scripts/measure_tool_transcript.py
    python3 scripts/measure_tool_transcript.py --live          # probe models
    python3 scripts/measure_tool_transcript.py --json report.json

The structural measurement is fully offline and deterministic: it drives a
scripted backend, never a real model, so it works with no daemon running
and cannot be skewed by sampling. ``--live`` adds a real round trip
against each installed model to confirm the daemon actually accepts the
native tool protocol for it; without the flag no model is contacted.

Exit codes: 0 measured (and probed, if asked), 1 the structural comparison
regressed, 2 the runtime could not be imported.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from opti_oignon.agent_eval.chat_runner import run_suite
    from opti_oignon.tool_calling import model_supports_native_tools
    from opti_oignon.tool_executor import (
        TOOL_TRANSCRIPT_NATIVE,
        _allowed_transcript_models,
    )
except Exception as exc:  # pragma: no cover - host environment problem
    print(f"Cannot import the runtime from {_REPO}: {exc}", file=sys.stderr)
    print("Run this from a checkout with the backend requirements installed.",
          file=sys.stderr)
    raise SystemExit(2) from exc


# The measurement task: a three-step chain (list, read, write). Long enough
# that the flat reconstruction has to re-fold a growing history, which is
# exactly the cost the native shape avoids.
SUITE = """\
suite: transcript-measure

tasks:
  - id: chain
    title: Three-step chain
    prompt: >
      List the workspace files, read status.txt, and write a one-line
      summary to report.txt.
    fixture:
      status.txt: "All services nominal.\\nQueue depth: 3.\\n"
    checks:
      - "expect_tool:write_file"
      - "expect_file:report.txt"
      - "final_nonempty"
      - "no_misattribution"
      - "no_internal_markers"
      - "tools_before_stream"
    script:
      - tool_calls:
          - name: list_files
            arguments: {}
      - tool_calls:
          - name: read_file
            arguments: {filename: status.txt}
      - tool_calls:
          - name: write_file
            arguments:
              filename: report.txt
              content: "Services nominal; queue depth 3.\\n"
      - content: ""
      - content: "I listed the workspace, read status.txt, and wrote the summary."
"""


def _measure(base: Path) -> dict:
    """Run the same task under both shapes; return the compared numbers."""
    suite_path = base / "measure.yaml"
    suite_path.write_text(SUITE, encoding="utf-8")

    reports = {}
    for shape, kwargs in (
        ("flat", None),
        ("native", {"tool_transcript": TOOL_TRANSCRIPT_NATIVE}),
    ):
        reports[shape] = run_suite(
            str(suite_path),
            fronts=("stream",),
            trace_dir=str(base / shape),
            executor_kwargs=kwargs,
        )

    out = {}
    for shape, report in reports.items():
        record = report.records[0]
        calls = record.model_calls
        # Every model call but the last is a decision round; the last is the
        # final user-facing generation.
        decisions = calls[:-1]
        out[shape] = {
            "passed": record.passed,
            "failed_checks": [
                check.to_dict() for check in record.checks if not check.ok
            ],
            "rounds": len(calls),
            "decision_chars": [call["chars_in"] for call in decisions],
            "decision_total": sum(call["chars_in"] for call in decisions),
            "final_chars": calls[-1]["chars_in"] if calls else 0,
            "carries_trained_roles": any(
                "tool" in call["roles"] for call in calls
            ),
        }
    return out


def _print_measurement(measured: dict) -> bool:
    """Print the structural verdict; True when native is measurably better."""
    flat, native = measured["flat"], measured["native"]

    print("Transcript measurement (scripted backend, no model contacted)")
    print("-" * 62)
    for shape in ("flat", "native"):
        row = measured[shape]
        print(
            f"  {shape:<7} rounds={row['rounds']}  "
            f"decision input={row['decision_total']:>6} chars  "
            f"final={row['final_chars']:>6} chars  "
            f"checks={'pass' if row['passed'] else 'FAIL'}"
        )
    for shape in ("flat", "native"):
        for check in measured[shape]["failed_checks"]:
            print(f"    {shape}: failed check {check}")

    saved = flat["decision_total"] - native["decision_total"]
    percent = (
        (saved / flat["decision_total"] * 100)
        if flat["decision_total"] else 0.0
    )
    print()
    print(
        f"  Native sends {saved} fewer characters into the decision rounds "
        f"({percent:.1f}% less)."
    )
    print(
        "  Native rounds carry the trained assistant/tool roles: "
        f"{native['carries_trained_roles']}; flat: "
        f"{flat['carries_trained_roles']}."
    )

    ok = (
        flat["passed"]
        and native["passed"]
        and saved > 0
        and native["carries_trained_roles"]
        and not flat["carries_trained_roles"]
    )
    print()
    print(
        "  VERDICT: native is smaller and correctly shaped."
        if ok else
        "  VERDICT: REGRESSION -- the comparison did not hold. Do not enable."
    )
    return ok


def _installed_models() -> list[str] | None:
    """Model names the local daemon reports, or None when it is unreachable."""
    try:
        import ollama
        listing = ollama.list()
    except Exception:
        return None
    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, dict):
        models = listing.get("models", [])
    names = []
    for entry in models or []:
        name = (
            entry.get("model") or entry.get("name")
            if isinstance(entry, dict)
            else getattr(entry, "model", None) or getattr(entry, "name", None)
        )
        if name:
            names.append(str(name))
    return sorted(set(names))


def _live_probe(model: str) -> tuple[bool, str]:
    """One real round trip: does this model return a native tool call?"""
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Read the file notes.txt. Use the tool.",
            }],
            tools=[{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                        },
                        "required": ["filename"],
                    },
                },
            }],
            options={"temperature": 0.0},
        )
    except Exception as exc:
        return False, f"backend refused the native protocol ({exc})"
    message = (
        response.get("message") if isinstance(response, dict)
        else getattr(response, "message", None)
    )
    calls = (
        message.get("tool_calls") if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if calls:
        return True, "returned a native tool call"
    return False, "accepted the protocol but called no tool"


def _print_models(live: bool) -> list[str]:
    """Print the per-model report; return the models worth allowing."""
    print()
    print("Installed models")
    print("-" * 62)
    names = _installed_models()
    if names is None:
        print("  No local daemon reachable -- skipping the model report.")
        print("  Start it and re-run to see which models qualify.")
        return []
    if not names:
        print("  The daemon reports no installed model.")
        return []

    allowed = _allowed_transcript_models() or ()
    candidates = []
    for name in names:
        capable = model_supports_native_tools(name)
        state = "already allowed" if name in allowed else "not allowed"
        line = f"  {name:<28} native-capable={str(capable):<5} {state}"
        if live and capable:
            ok, detail = _live_probe(name)
            line += f"  probe={'ok' if ok else 'no'} ({detail})"
            capable = capable and ok
        print(line)
        if capable and name not in allowed:
            candidates.append(name)
    return candidates


def _print_next_step(candidates: list[str], measured_ok: bool) -> None:
    print()
    print("What to do next")
    print("-" * 62)
    if not measured_ok:
        print("  Nothing. The measurement regressed; leave the default alone.")
        return
    if not candidates:
        print("  Nothing to add: no new native-capable model was found.")
        print("  The flat reconstruction stays the default, which is correct.")
        return
    print("  Nothing has been changed. To allow a model, add these lines to")
    print("  the user configuration yourself (one model at a time, and")
    print("  re-run this script afterwards to confirm):")
    print()
    print("      tool_transcript: native")
    print("      tool_transcript_models:")
    for name in candidates:
        print(f"        - {name}")
    print()
    print("  A model absent from that list keeps the flat reconstruction,")
    print("  and so does an unreadable configuration.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live", action="store_true",
        help="probe each installed model with one real tool round trip",
    )
    parser.add_argument(
        "--json", metavar="PATH",
        help="also write the measurement to a JSON file",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="transcript-measure-") as tmp:
        measured = _measure(Path(tmp))

    measured_ok = _print_measurement(measured)
    candidates = _print_models(args.live)
    _print_next_step(candidates, measured_ok)

    if args.json:
        payload = {
            "measurement": measured,
            "candidates": candidates,
            "verdict": "ok" if measured_ok else "regression",
        }
        Path(args.json).write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
        print(f"\nWrote {args.json}")

    return 0 if measured_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
