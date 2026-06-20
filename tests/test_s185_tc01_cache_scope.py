"""S185 audit fix TC-01 -- semantic cache scope defaults to conversation.

The semantic cache defaulted to ``DEFAULT_SCOPE = "global"`` (and the shipped
config/cache.yaml set ``scope: "global"``). Under global scope a fuzzy semantic
hit is not confined to its originating conversation, so a response cached for
conversation A (which may embed A's private RAG/project context) can be served
to a merely-similar query in conversation B -- a confidentiality bleed across
the user's own projects, and an approximate answer served as exact.

The fix flips the default to "conversation" (constant + shipped cache.yaml) and,
in Bulbe mode, forces conversation scope regardless of config and fails closed:
when no conversation is in scope it neither serves nor stores via the shared
bucket.

The module is loaded in isolation (opti_oignon stubbed, so db_utils falls back
to plain sqlite3 and the Bulbe import in _is_bulbe returns False by default).
Tests drive the exact-match tier, so no Ollama embedder is needed. The DB is a
temp file; the scope is set per test via the constructor override.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest

# Mark opti_oignon as a package (so the relative ``from .config import DATA_DIR``
# resolves) and stub opti_oignon.config with a DATA_DIR. db_utils stays absent so
# semantic_cache falls back to plain sqlite3; security_mode stays absent so the
# default _is_bulbe() returns False (Bulbe tests monkeypatch it).
_PKG = sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))
if not hasattr(_PKG, "__path__"):
    _PKG.__path__ = []  # type: ignore[attr-defined]
if "opti_oignon.config" not in sys.modules:
    _cfg = types.ModuleType("opti_oignon.config")
    _cfg.DATA_DIR = Path(tempfile.mkdtemp())
    sys.modules["opti_oignon.config"] = _cfg

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "semantic_cache.py"


def _load():
    spec = importlib.util.spec_from_file_location("opti_oignon.semantic_cache", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12 dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


sc = _load()


def _make_cache(tmp_path, scope):
    cache = sc.SemanticCache(
        db_path=tmp_path / "cache.db",
        config_path=tmp_path / "no-such-config.yaml",  # force pure defaults
        scope=scope,
    )
    cache._config["enabled"] = True
    cache._config["semantic_match_enabled"] = False  # exact tier only
    return cache


# ---------------------------------------------------------------------------
# The shipped default is conversation
# ---------------------------------------------------------------------------

def test_default_scope_constant_is_conversation():
    assert sc.DEFAULT_SCOPE == "conversation"


def test_shipped_cache_yaml_scope_is_conversation():
    import yaml

    cfg = yaml.safe_load(
        (_REPO_ROOT / "opti_oignon" / "config" / "cache.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert cfg.get("scope") == "conversation"


# ---------------------------------------------------------------------------
# Conversation scope confines hits to the originating conversation
# ---------------------------------------------------------------------------

def test_conversation_scope_blocks_cross_conversation(tmp_path):
    cache = _make_cache(tmp_path, scope="conversation")
    q = "What is Shannon diversity?"
    cache.put(q, "answer-A", model="m", conversation_id="conv-A")

    # Same query, different conversation -> miss under conversation scope.
    assert cache.get(q, conversation_id="conv-B", model="m") is None
    # Same conversation -> hit.
    hit = cache.get(q, conversation_id="conv-A", model="m")
    assert hit is not None
    assert hit.response == "answer-A"


def test_global_scope_bleeds_cross_conversation(tmp_path):
    # Contrast: under the old global default the same lookup hits across convs.
    cache = _make_cache(tmp_path, scope="global")
    q = "What is Shannon diversity?"
    cache.put(q, "answer-A", model="m", conversation_id="conv-A")
    hit = cache.get(q, conversation_id="conv-B", model="m")
    assert hit is not None
    assert hit.response == "answer-A"


# ---------------------------------------------------------------------------
# Bulbe forces conversation scope and fails closed
# ---------------------------------------------------------------------------

def test_bulbe_forces_conversation_scope_over_global(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_is_bulbe", lambda: True)
    # Even with an explicitly-global config, Bulbe scopes per conversation.
    cache = _make_cache(tmp_path, scope="global")
    q = "What is Shannon diversity?"
    cache.put(q, "answer-A", model="m", conversation_id="conv-A")
    assert cache.get(q, conversation_id="conv-B", model="m") is None
    assert cache.get(q, conversation_id="conv-A", model="m") is not None


def test_bulbe_fails_closed_without_conversation_id(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_is_bulbe", lambda: True)
    cache = _make_cache(tmp_path, scope="global")
    q = "What is Shannon diversity?"
    # No conversation id in Bulbe: store is skipped, lookup is a miss.
    assert cache.put(q, "answer", model="m", conversation_id="") == ""
    assert cache.get(q, conversation_id="", model="m") is None

    # Nothing landed in the shared bucket: a Daily lookup with no conv id (which
    # would otherwise read the "" bucket) still finds nothing.
    monkeypatch.setattr(sc, "_is_bulbe", lambda: False)
    assert cache.get(q, conversation_id="", model="m") is None
