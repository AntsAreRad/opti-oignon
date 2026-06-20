"""S185 audit fix CW-01 -- opt-in full wipe (RAM + persisted rows).

The conversation wipe was RAM-only: it zeroed in-memory buffers but never deleted
the persisted (SQLCipher-encrypted in Bulbe) conversation rows on disk, and Bulbe
does persist conversations. That is intentional defense-in-depth, but an operator
running an emergency wipe may want the on-disk history gone too.

The fix adds an opt-in ``purge_disk`` flag (default False, so behaviour is
unchanged) to ConversationWipeManager.wipe / wipe_all and to the wipe endpoints.
When True the wipe also deletes the persisted rows via the conversation manager,
best-effort (the RAM wipe still succeeds if the disk purge is unavailable).

conversation_wipe is loaded in isolation; the conversation manager is replaced by
a stub registered at opti_oignon.conversation so no real DB is touched. The
per-turn Bulbe wipe path is intentionally not changed (it must stay RAM-only).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "conversation_wipe.py"


def _load():
    spec = importlib.util.spec_from_file_location("conversation_wipe_cw01", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cw = _load()


class _StubConv:
    def __init__(self, cid: str) -> None:
        self.id = cid


class _StubMgr:
    """Stateful stand-in for the conversation manager singleton."""

    def __init__(self, ids):
        self._ids = set(ids)
        self.deleted: list[str] = []

    def delete_conversation(self, cid: str) -> bool:
        self.deleted.append(cid)
        if cid in self._ids:
            self._ids.discard(cid)
            return True
        return False

    def list_conversations(self, limit: int = 50, offset: int = 0):
        return [_StubConv(c) for c in list(self._ids)[:limit]]


def _install_mgr(monkeypatch, ids):
    stub = _StubMgr(ids)
    module = types.ModuleType("opti_oignon.conversation")
    module.conversation_manager = stub
    monkeypatch.setitem(sys.modules, "opti_oignon.conversation", module)
    return stub


def _install_no_mgr(monkeypatch):
    # A module without conversation_manager -> the lazy import raises ImportError.
    module = types.ModuleType("opti_oignon.conversation")
    monkeypatch.setitem(sys.modules, "opti_oignon.conversation", module)


# ---------------------------------------------------------------------------
# Single-conversation wipe
# ---------------------------------------------------------------------------

def test_wipe_default_is_ram_only(monkeypatch):
    stub = _install_mgr(monkeypatch, {"c1"})
    mgr = cw.ConversationWipeManager()
    result = mgr.wipe("c1")  # no purge_disk
    assert result.disk_purged is False
    assert result.rows_deleted == 0
    assert stub.deleted == [], "default wipe must not touch the disk"


def test_wipe_purge_disk_deletes_rows(monkeypatch):
    stub = _install_mgr(monkeypatch, {"c1"})
    mgr = cw.ConversationWipeManager()
    result = mgr.wipe("c1", purge_disk=True)
    assert result.disk_purged is True
    assert result.rows_deleted == 1
    assert stub.deleted == ["c1"]


def test_wipe_purge_disk_best_effort_when_unavailable(monkeypatch):
    _install_no_mgr(monkeypatch)
    mgr = cw.ConversationWipeManager()
    result = mgr.wipe("c1", purge_disk=True)
    # RAM wipe still succeeds; disk purge silently degrades.
    assert result.success is True
    assert result.disk_purged is False


# ---------------------------------------------------------------------------
# Emergency wipe-all
# ---------------------------------------------------------------------------

def test_wipe_all_default_ram_only(monkeypatch):
    stub = _install_mgr(monkeypatch, {"c1", "c2"})
    mgr = cw.ConversationWipeManager()
    results = mgr.wipe_all()  # no purge_disk
    assert stub.deleted == []
    assert sum(r.rows_deleted for r in results) == 0


def test_wipe_all_purge_disk_purges_all_persisted(monkeypatch):
    stub = _install_mgr(monkeypatch, {"c1", "c2", "c3"})
    mgr = cw.ConversationWipeManager()
    held = ["msg-a", "msg-b"]  # keep a strong ref so the buffer survives
    mgr.register_buffer("c1", held)
    results = mgr.wipe_all(purge_disk=True)
    # All three persisted conversations are deleted, even those with no buffer.
    assert set(stub.deleted) == {"c1", "c2", "c3"}
    assert stub._ids == set()
    assert sum(r.rows_deleted for r in results) == 3
    by_id = {r.conversation_id: r for r in results}
    assert by_id["c1"].disk_purged is True
    assert by_id["c2"].buffers_wiped == 0 and by_id["c2"].disk_purged is True


# ---------------------------------------------------------------------------
# Endpoints + docs expose the opt-in flag, default off
# ---------------------------------------------------------------------------

def _src(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def test_endpoints_expose_optin_purge_disk():
    src = _src("opti_oignon/api/routes_security.py")
    assert "async def conversation_wipe_all(purge_disk: bool = False)" in src
    assert "purge_disk: bool = False" in src.split("conversation_wipe_single", 1)[1]
    assert "mgr.wipe_all(purge_disk=purge_disk)" in src
    assert "mgr.wipe(conversation_id, purge_disk=purge_disk)" in src


def test_module_documents_ram_vs_disk():
    doc = _src("opti_oignon/conversation_wipe.py").lower()
    assert "cw-01" in doc
    assert "ram-only" in doc
    assert "purge_disk" in doc
