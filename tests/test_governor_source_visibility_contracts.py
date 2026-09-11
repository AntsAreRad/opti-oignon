#!/usr/bin/env python3
"""Contracts that the governor names what it could not look at.

The governor's resident-set read walks every registered backend and takes the
in-process ``_loaded_models`` dict from those that expose one. A backend that
does not -- the external llama-server, which holds whatever the host loaded
into it -- was skipped with a bare ``continue``. The snapshot then said
nothing about it at all, so "this backend holds no resident models" and "the
governor never looked at this backend" produced the identical snapshot, and
an admission decision made on that snapshot could be confidently wrong about
a card that was in fact full.

  * SV1 -- a backend whose resident set cannot be read is named in the
    snapshot's sources as unread, rather than omitted.
  * SV2 -- a backend whose set can be read is not so named, and the marker
    is therefore about the backend and not always present.

Nothing here reads a real card. Every backend is a stand-in; the contracts pin
only that the snapshot distinguishes an empty answer from no answer.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the database layer is seeded and the store lives under
pytest's tmp_path.
"""

import sqlite3
import sys
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _db_stub():
    module = types.ModuleType("opti_oignon.db_utils")
    module.safe_connect = lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False),
    )
    return module


def _open():
    loaded, restore = isolate(
        targets={
            "opti_oignon.resource_governor": source("resource_governor.py"),
        },
        seeded={"opti_oignon.db_utils": _db_stub()},
    )
    return loaded["opti_oignon.resource_governor"], restore


class _Unreadable:
    """A backend that exposes no resident set, like the external server."""

    name = "llama_server"


class _Readable:
    """A backend with a readable, empty resident set."""

    name = "llama_cpp"
    _loaded_models: dict = {}


class _Registry:
    def __init__(self, *backends):
        self._backends = list(backends)

    def backends(self):
        return list(self._backends)


def _governor(rg, tmp_path, registry):
    config = tmp_path / "resource_governor.yaml"
    config.write_text("enabled: true\ntotal_vram_gb: null\n", encoding="utf-8")
    return rg.ResourceGovernor(
        config_path=config,
        db_path=tmp_path / "governor.db",
        warmup=None,
        registry=registry,
        vram_probe=lambda: -1.0,
    )


# ---------------------------------------------------------------------------
# SV1 -- an unreadable backend is named, not omitted
# ---------------------------------------------------------------------------
def test_sv1_an_unreadable_backend_is_named_in_the_sources(tmp_path):
    rg, restore = _open()
    try:
        gov = _governor(rg, tmp_path, _Registry(_Unreadable()))
        snapshot = gov._build_snapshot()
        unread = [s for s in snapshot.sources if s.startswith("S2-unread:")]
        assert unread == ["S2-unread:llama_server"], (
            "a backend the governor could not read is named as unread, so a "
            "snapshot that says nothing about it is no longer possible"
        )
        assert snapshot.backend_resident == [], (
            "and no resident view was invented for it"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# SV2 -- a readable backend is not so named
# ---------------------------------------------------------------------------
def test_sv2_a_readable_backend_is_not_marked_unread(tmp_path):
    rg, restore = _open()
    try:
        gov = _governor(rg, tmp_path, _Registry(_Readable(), _Unreadable()))
        snapshot = gov._build_snapshot()
        unread = [s for s in snapshot.sources if s.startswith("S2-unread:")]
        assert "S2-unread:llama_cpp" not in unread, (
            "a backend whose set was read is not marked unread: the marker is "
            "about the backend, not a fixture of every snapshot"
        )
        assert unread == ["S2-unread:llama_server"], (
            "while the unreadable one beside it still is"
        )
        assert "S2" in snapshot.sources, (
            "and the read that succeeded is still credited as a source"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    import tempfile

    tests = [
        ("SV1 unreadable backend is named", test_sv1_an_unreadable_backend_is_named_in_the_sources),
        ("SV2 readable backend is not marked", test_sv2_a_readable_backend_is_not_marked_unread),
    ]
    passed = 0
    for label, fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"PASS  {label}")
                passed += 1
            except Exception:  # noqa: BLE001 -- report and continue
                print(f"FAIL  {label}")
                traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
