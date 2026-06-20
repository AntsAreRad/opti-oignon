"""S191 F4e -- coding agent end to end.

No code change this lot (record + verify). These tests pin the recorded CA-04
(the coding-history store has no per-user scoping) behaviourally, and pin the
verified invariants by source: the coding path has no direct-ollama bypass, the
apply phase is per-path contained with a diff-integrity check, and the
agentic-executor approval hook is bound per call (EX-02).

`coding_history.py` is stdlib-only at module scope (the db_utils import is
guarded with a plaintext fallback, exactly the path used here for a temp DB), so
it loads in isolation with the parent package stubbed. The heavier
`coding_agent` / `chat_coding_agent` / `agentic_executor` modules pull the
sandbox + executor stack, so they are checked at the source level.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

OO_DIR = Path(__file__).resolve().parent.parent / "opti_oignon"
HISTORY = OO_DIR / "coding_history.py"
CODING_AGENT = OO_DIR / "coding_agent.py"
CHAT_CODING = OO_DIR / "chat_coding_agent.py"
EXECUTOR = OO_DIR / "agentic_executor.py"


def _load_history():
    if "opti_oignon" not in sys.modules:
        sys.modules["opti_oignon"] = types.ModuleType("opti_oignon")
    spec = importlib.util.spec_from_file_location("opti_oignon.coding_history", HISTORY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# CA-04 (behavioural): coding history has no per-user scoping
# --------------------------------------------------------------------------

def test_ca04_history_schema_has_no_user_id_column():
    hist = _load_history()
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "coding_history.db")
        store = hist.CodingHistoryStore(db_path=db)
        store.record_task_start("t1", "do a thing", project_path="/p", model="m")
        conn = store._get_conn()
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
            chk_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(checkpoints)").fetchall()
            }
        finally:
            conn.close()
    # The store is task-keyed only; unlike the memory canonical store it carries
    # no user_id column (pins CA-04: no per-user isolation in multi-user mode).
    assert "task_id" in cols
    assert "user_id" not in cols
    assert "user_id" not in chk_cols


def test_ca04_history_uses_safe_connect():
    hist = _load_history()
    assert hasattr(hist, "_safe_connect")


# --------------------------------------------------------------------------
# Verified invariants (source)
# --------------------------------------------------------------------------

def test_f4e_no_direct_ollama_in_coding_modules():
    for p in (CODING_AGENT, CHAT_CODING, EXECUTOR):
        src = p.read_text(encoding="utf-8")
        assert "ollama.chat(" not in src, f"{p.name} must not call ollama.chat directly"
        assert 'response["message"]["content"]' not in src


def test_f4e_apply_is_path_contained_and_integrity_checked():
    src = CODING_AGENT.read_text(encoding="utf-8")
    # The S184 hardening: per-path containment + diff-integrity hash + hardcoded
    # checkpoint-before-apply.
    assert "_is_within_target(" in src
    assert "Diff integrity check failed" in src
    assert "self._config.checkpoint_before_apply = True" in src


def test_f4e_executor_approval_hook_is_per_call():
    src = EXECUTOR.read_text(encoding="utf-8")
    assert "approval_fn" in src
    # EX-02 (S185): the gate is bound to this call, not a shared global.
    assert "bound to this call" in src
