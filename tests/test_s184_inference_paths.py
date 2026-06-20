"""S184 audit fix IB-01 -- llama.cpp GGUF path containment + gitignore hardening.

LlamaCppBackend._resolve_model_path maps a model name to a .gguf file under the
configured model_dirs. The S136 fix rejects absolute paths and "..", then verified
containment with ``str(resolved).startswith(str(dir_resolved))``. startswith has a
sibling-prefix pitfall: a symlink inside ``models/main`` that points at a sibling
``models/main-evil/x.gguf`` resolves to a path that *startswith* ``models/main`` and
was therefore accepted, escaping the model directory. The fix replaces both
containment checks with a commonpath-based ``_is_within_dir`` helper (the same
construction as the coding-agent apply guard ``_is_within_target``).

Also asserts the gitignore now globs the data-dir runtime JSON and pickled model
artifacts (e.g. learned_router.pkl), a follow-up to the S183 P-01 packaging lot.

The module is loaded in isolation: ``ollama`` is stubbed and ``opti_oignon`` is a
bare module, so the optional engine imports are absent. _resolve_model_path is pure
path logic and needs neither ollama nor llama-cpp-python.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "inference_backend.py"


def _load():
    spec = importlib.util.spec_from_file_location("inference_backend_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12 dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


ib = _load()


# ---------------------------------------------------------------------------
# _is_within_dir unit behaviour
# ---------------------------------------------------------------------------

def test_within_dir_child_and_self():
    assert ib._is_within_dir(Path("/models/main"), Path("/models/main/a.gguf")) is True
    assert ib._is_within_dir(Path("/models/main"), Path("/models/main")) is True
    assert ib._is_within_dir(
        Path("/models/main"), Path("/models/main/sub/b.gguf")
    ) is True


def test_within_dir_sibling_prefix_refused():
    # The startswith pitfall: /models/main-evil is NOT inside /models/main.
    assert ib._is_within_dir(
        Path("/models/main"), Path("/models/main-evil/x.gguf")
    ) is False


def test_within_dir_dotdot_escape_refused():
    assert ib._is_within_dir(
        Path("/models/main"), Path("/models/main/../../etc/passwd")
    ) is False


# ---------------------------------------------------------------------------
# _resolve_model_path end-to-end on a real temp tree
# ---------------------------------------------------------------------------

def test_resolve_rejects_absolute_path(tmp_path):
    base = tmp_path / "models"
    base.mkdir()
    (base / "real.gguf").write_bytes(b"GGUF")
    backend = ib.LlamaCppBackend(model_dirs=[str(base)])
    assert backend._resolve_model_path("/etc/passwd") is None
    assert backend._resolve_model_path(str(base / "real.gguf")) is None  # absolute


def test_resolve_rejects_dotdot(tmp_path):
    base = tmp_path / "models"
    base.mkdir()
    backend = ib.LlamaCppBackend(model_dirs=[str(base)])
    assert backend._resolve_model_path("../secret.gguf") is None
    assert backend._resolve_model_path("a/../../b.gguf") is None


def test_resolve_returns_file_within_dir(tmp_path):
    base = tmp_path / "models"
    base.mkdir()
    target = base / "model.gguf"
    target.write_bytes(b"GGUF")
    backend = ib.LlamaCppBackend(model_dirs=[str(base)])
    resolved = backend._resolve_model_path("model.gguf")
    assert resolved is not None
    assert resolved == target.resolve()
    # Name without the .gguf suffix should also resolve.
    assert backend._resolve_model_path("model") == target.resolve()


def test_resolve_blocks_sibling_prefix_symlink_escape(tmp_path):
    # base = .../models/main ; sibling = .../models/main-evil (shares the prefix).
    main = tmp_path / "models" / "main"
    main.mkdir(parents=True)
    evil_dir = tmp_path / "models" / "main-evil"
    evil_dir.mkdir(parents=True)
    evil_file = evil_dir / "x.gguf"
    evil_file.write_bytes(b"GGUF")

    link = main / "link.gguf"
    link.symlink_to(evil_file)

    backend = ib.LlamaCppBackend(model_dirs=[str(main)])
    # The old startswith check would have accepted this (main-evil startswith main);
    # commonpath containment refuses it.
    assert backend._resolve_model_path("link.gguf") is None
    assert ib._is_within_dir(main, evil_file) is False


def test_resolve_allows_symlink_within_dir(tmp_path):
    # A symlink that stays inside the model dir must still resolve (no regression).
    base = tmp_path / "models"
    base.mkdir()
    real = base / "real.gguf"
    real.write_bytes(b"GGUF")
    alias = base / "alias.gguf"
    alias.symlink_to(real)

    backend = ib.LlamaCppBackend(model_dirs=[str(base)])
    assert backend._resolve_model_path("alias.gguf") == real.resolve()


# ---------------------------------------------------------------------------
# Source assertion: containment no longer uses str.startswith
# ---------------------------------------------------------------------------

def test_resolve_model_path_uses_commonpath_not_startswith():
    src = _PATH.read_text(encoding="utf-8")
    start = src.index("def _resolve_model_path")
    end = src.index("def _get_or_load", start)
    body = src[start:end]
    assert "_is_within_dir(" in body
    assert "startswith" not in body


# ---------------------------------------------------------------------------
# gitignore hardening (P-01 follow-up)
# ---------------------------------------------------------------------------

def test_gitignore_globs_runtime_json_and_pickles():
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = {ln.strip() for ln in gitignore.splitlines()}
    for pattern in ("*.pkl", "*.joblib", "data/*.json", "opti_oignon/data/*.json"):
        assert pattern in lines, f"gitignore missing pattern: {pattern}"
