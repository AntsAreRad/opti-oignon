#!/usr/bin/env python3
"""Contracts for the compression-fidelity harness and its CI floor.

What a summary keeps has to be measured, not asserted. The harness declares
facts inside fixture conversations, drives them through the REAL tier
machinery with a deterministic instrument, and scores retention by keyword
presence -- no judge model, no network, no taste. A separate probe plants a
needle deep in a real conversation store and asks the real archive
retriever to find it again after the prompt has long moved on. A CI guard
then compares the fresh measurement to a floor that lives in a committed
threshold file: the engagement is written down, and the guard only reads
it. These clauses pin all three pieces:

  * The fixture loader is loud: a fact without keywords, a probe without
    text, a conversation without messages -- each names its defect instead
    of loading half a measurement.
  * Retention over the shipped fixture with the shipped instrument is
    total, and the report names every fact it verified. This is the
    substance behind the committed floor.
  * Loss is detected and named: an instrument that drops a declared
    keyword pulls the ratio below one and the report says which fact went
    missing.
  * The needle is recovered from a real store at every declared depth and
    haystack size, through the same retriever the product uses.
  * The probe does not flatter: a nonce that was never planted is never
    reported recovered.
  * The guard reads the committed threshold file rather than embedding a
    number, passes at or above the floor, fails below a stricter floor,
    and fails loudly when the engagement file is missing -- an engagement
    that vanished is a failure, never a pass.
  * The harness is local by construction: no network-capable module and no
    model client is imported at module level.

The needle path builds its own throwaway store under a temporary
directory; nothing here touches the shipped data tree.
"""

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_PATH = _ROOT / "opti_oignon" / "agent_eval" / "fidelity.py"
_GUARD_PATH = _ROOT / ".github" / "scripts" / "summary_fidelity_guard.py"
_THRESHOLD_PATH = _ROOT / ".github" / "scripts" / "summary_fidelity.threshold"

sys.path.insert(0, str(_ROOT))

from opti_oignon.agent_eval import fidelity  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def test_f1_the_fixture_loader_is_loud_about_defects():
    conversations = fidelity.load_fixture()
    assert conversations, "the shipped fixture must load"
    for conv in conversations:
        assert conv.facts, "a fixture conversation without facts measures nothing"
        assert conv.messages, "a fixture conversation without messages is empty"
        ids = [m["id"] for m in conv.messages]
        assert ids == sorted(ids) and len(set(ids)) == len(ids), (
            "loaded messages must carry unique ascending ids"
        )

    def _broken(payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.yaml"
            path.write_text(payload, encoding="utf-8")
            with pytest.raises(ValueError) as caught:
                fidelity.load_fixture(path)
        return str(caught.value)

    message = _broken(
        "version: 1\nconversations:\n"
        "  - id: c1\n    facts:\n      - id: f1\n        probe: p\n"
        "    messages:\n      - role: user\n        content: hello\n"
    )
    assert "keywords" in message

    message = _broken(
        "version: 1\nconversations:\n"
        "  - id: c1\n    facts:\n      - id: f1\n        keywords: [k]\n"
        "    messages:\n      - role: user\n        content: hello\n"
    )
    assert "probe" in message

    message = _broken(
        "version: 1\nconversations:\n"
        "  - id: c1\n    facts:\n"
        "      - id: f1\n        probe: p\n        keywords: [k]\n"
    )
    assert "messages" in message


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_f2_retention_over_the_shipped_fixture_is_total_and_named():
    report = fidelity.run_fidelity()
    assert report.overall_ratio == 1.0, (
        "the shipped fixture with the shipped instrument is the committed "
        f"floor; measured {report.overall_ratio}"
    )
    declared = {
        (conv.id, fact.id)
        for conv in fidelity.load_fixture()
        for fact in conv.facts
    }
    verified = {(r.conversation_id, r.fact_id) for r in report.facts}
    assert verified == declared, "every declared fact must be verified by name"
    assert all(r.retained and not r.missing_keywords for r in report.facts)
    for conv_id, ratio in report.conversations.items():
        assert ratio == 1.0, f"{conv_id} lost a declared fact"


def test_f3_loss_is_detected_and_the_missing_fact_is_named():
    conversations = fidelity.load_fixture()
    victim = conversations[0].facts[0]
    doomed = victim.keywords[0]

    def lossy(messages):
        text = fidelity.extractive_instrument(messages)
        return text.replace(doomed, "") or None

    report = fidelity.run_fidelity(conversations, summarize_fn=lossy)
    assert report.overall_ratio < 1.0, "a dropped keyword must move the ratio"
    named = {
        (r.conversation_id, r.fact_id): r
        for r in report.facts
        if not r.retained
    }
    key = (conversations[0].id, victim.id)
    assert key in named, "the missing fact must be named, not averaged away"
    assert doomed in named[key].missing_keywords


# ---------------------------------------------------------------------------
# Needle
# ---------------------------------------------------------------------------


def test_f4_the_needle_is_recovered_at_every_depth_and_size():
    with tempfile.TemporaryDirectory() as tmp:
        report = fidelity.run_needle_sweep(Path(tmp))
    assert report.cases, "an empty sweep measures nothing"
    assert report.recovery_ratio == 1.0, (
        "a planted needle must be recoverable from the archive; "
        + ", ".join(
            f"{c.haystack_size}/{c.depth} missed"
            for c in report.cases
            if not c.recovered
        )
    )
    depths = {c.depth for c in report.cases}
    assert {"first", "middle", "last"} <= depths, (
        "the sweep must cover the start, the middle and the end"
    )
    assert len({c.haystack_size for c in report.cases}) >= 2, (
        "the sweep must cover more than one haystack size"
    )


def test_f5_an_unplanted_needle_is_never_reported_recovered():
    with tempfile.TemporaryDirectory() as tmp:
        from opti_oignon.conversation import ConversationManager

        manager = ConversationManager(db_path=Path(tmp) / "probe.db")
        conv = manager.create_conversation(title="haystack only")
        for k in range(12):
            manager.add_message(conv.id, "user" if k % 2 == 0 else "assistant",
                                f"ordinary filler line {chr(97 + k)} without it")
        assert fidelity.probe_needle(manager, conv.id,
                                     "756431908212") is False, (
            "recovery must be earned by the archive, never granted by the "
            "probe"
        )


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


def _run_guard(threshold=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(_GUARD_PATH)]
    if threshold is not None:
        command += ["--threshold", str(threshold)]
    return subprocess.run(
        command, cwd=_ROOT, env=env, capture_output=True, text=True,
        timeout=300,
    )


def test_f6_the_guard_reads_the_written_engagement():
    committed = _THRESHOLD_PATH.read_text(encoding="utf-8")
    assert "fidelity_floor" in committed and "needle_floor" in committed, (
        "the engagement file must state both floors explicitly"
    )

    passed = _run_guard()
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert "fidelity" in passed.stdout and "needle" in passed.stdout, (
        "the guard must report what it measured against what was promised"
    )

    with tempfile.TemporaryDirectory() as tmp:
        stricter = Path(tmp) / "stricter.threshold"
        stricter.write_text(
            "fidelity_floor = 1.01\nneedle_floor = 1.01\n", encoding="utf-8"
        )
        failed = _run_guard(stricter)
        assert failed.returncode != 0, (
            "a measurement below the floor must fail the build"
        )

        missing = _run_guard(Path(tmp) / "absent.threshold")
        assert missing.returncode != 0, (
            "an engagement that vanished is a failure, never a pass"
        )


# ---------------------------------------------------------------------------
# Locality
# ---------------------------------------------------------------------------


def test_f7_the_harness_is_local_by_construction():
    tree = ast.parse(_HARNESS_PATH.read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    forbidden = {"socket", "http", "urllib", "requests", "httpx", "ollama"}
    assert not (top_level & forbidden), (
        f"network-capable or model-client imports at module level: "
        f"{sorted(top_level & forbidden)}"
    )
    everywhere = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not (everywhere & {"socket", "http", "urllib", "requests",
                              "httpx"}), (
        "the harness must not reach for the network anywhere at all"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                import traceback

                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
