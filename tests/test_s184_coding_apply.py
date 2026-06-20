"""S184 audit fix CA-01 -- apply-phase containment (sandbox escape on apply).

The coding agent's apply phase is the only exit from the sandbox. ``apply_changes``
builds ``dest = os.path.join(target, diff.path)`` and previously wrote/deleted it
with no containment check, so a diff path containing ".." (or resolving outside the
project via a symlink) escaped ``target`` onto the host filesystem. These tests pin
the ``_is_within_target`` containment helper and assert the guard precedes the host
write.

The module is loaded in isolation: ``opti_oignon`` is stubbed as a bare (non-package)
module so coding_agent's ``from opti_oignon.X import Y`` lines fail with ImportError
(caught), exercising only the pure path helpers without the sandbox stack.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Bare stubs: make the optional opti_oignon submodule imports fail gracefully.
sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

_PATH = Path(__file__).resolve().parents[1] / "opti_oignon" / "coding_agent.py"


def _load():
    spec = importlib.util.spec_from_file_location("coding_agent_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12 dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


ca = _load()


def test_within_target_normal_paths():
    assert ca._is_within_target("/home/u/proj", "/home/u/proj/src/main.py") is True
    assert ca._is_within_target("/home/u/proj", "/home/u/proj") is True


def test_within_target_dotdot_escape_refused():
    assert ca._is_within_target(
        "/home/u/proj", "/home/u/proj/../etc/cron.d/x"
    ) is False
    assert ca._is_within_target(
        "/home/u/proj", "/home/u/proj/a/b/../../../../etc/passwd"
    ) is False


def test_within_target_sibling_prefix_refused():
    # The startswith pitfall: /home/u/projevil is NOT inside /home/u/proj.
    assert ca._is_within_target("/home/u/proj", "/home/u/projevil/x") is False


def test_within_target_symlink_escape_refused(tmp_path):
    target = tmp_path / "proj"
    target.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = target / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    # A write under the symlink resolves outside the project -> refused.
    assert ca._is_within_target(str(target), str(link / "evil.txt")) is False
    # A normal file inside the project stays inside.
    assert ca._is_within_target(str(target), str(target / "ok.txt")) is True


def test_validate_apply_target_blocks_system_roots():
    for bad in ("/", "/etc", "/root", "/usr", "/home", "/mnt", "/var"):
        with pytest.raises(ValueError):
            ca._validate_apply_target(bad)


def test_apply_guard_precedes_host_write():
    # Source-level: the containment guard must run before any host write/delete.
    src = _PATH.read_text(encoding="utf-8")
    assert "_is_within_target(target, dest)" in src
    i_guard = src.index("_is_within_target(target, dest)")
    i_write = src.index('open(dest, "w"')
    i_delete = src.index("os.remove(dest)")
    assert i_guard < i_write, "guard must precede the host write"
    assert i_guard < i_delete, "guard must precede the host delete"
