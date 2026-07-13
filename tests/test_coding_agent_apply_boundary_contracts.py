#!/usr/bin/env python3
"""Apply-boundary contracts: the only exit from the coding sandbox is guarded.

The coding agent runs every filesystem step inside a disposable sandbox;
the apply phase is the single path by which files reach the real
filesystem. That boundary carries three independent guards -- a
forbidden-target validator, a per-file containment check, and a review
integrity hash -- plus a non-overridable human-checkpoint switch. This
suite pins them:

  * AB1 -- the target validator refuses system roots and bare top-level
    paths and accepts a nested project directory;
  * AB2 -- the containment check accepts the target itself and its
    descendants, and refuses parent traversal, sibling-prefix lookalikes,
    and symlink escapes;
  * AB3 -- apply refuses a per-file path that escapes the target (recorded
    as an error, nothing written outside) while still applying the
    contained files;
  * AB4 -- the checkpoint-before-apply switch is forced on at both the
    constructor and the configuration loader, whatever the input says;
  * AB5 -- sandbox contents that changed after review fail the integrity
    hash and apply refuses before writing anything.

Loads the coding module in isolation under a stand-in package; every
``opti_oignon.*`` entry plus the model client entry is snapshotted and
evicted first, so all optional integrations stay absent and the boundary
logic is exercised bare. A meta-path guard refuses any project submodule
that was not seeded, so the load behaves identically whether or not the
project is installed (an editable install resolves submodules by name and
would otherwise bypass the stand-in package, pulling live code and real
databases into the window). Local-only. Runs under pytest or the __main__
runner.
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the test's
    back -- silently importing live code and reopening real databases. This
    guard sits ahead of every finder and refuses the names that were not
    seeded, so a load behaves identically whether the project is installed
    or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


def _load():
    """Load coding_agent.py under a stand-in package."""
    keys = ["ollama"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # any client import fails deterministically

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    sys.modules["opti_oignon"] = root

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        for k, v in saved.items():
            sys.modules[k] = v

    full = "opti_oignon.coding_agent"
    spec = importlib.util.spec_from_file_location(full, _OO / "coding_agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    root.coding_agent = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        restore()
        raise

    return mod, restore


def _agent(mod):
    """A bare agent: explicit config, dummy session, no integrations."""
    return mod.CodingAgent(
        sandbox_session=SimpleNamespace(active=False),
        config=mod.CodingAgentConfig(),
    )


def _seed_diffs(mod, agent, diffs):
    """Install diffs on the agent and stamp the review integrity hash."""
    agent._diffs = diffs
    agent._diffs_hash = agent._compute_diffs_hash()


# ---------------------------------------------------------------------------
# AB1 -- the target validator refuses system roots and top-level paths
# ---------------------------------------------------------------------------
def test_ab1_validator_refuses_system_roots_and_accepts_projects():
    mod, restore = _load()
    try:
        # Single-segment system roots and a bare top-level path (the
        # top-level-parts mechanism refuses these).
        for forbidden in ("/", "/etc", "/usr", "/tmp", "/home"):
            try:
                mod._validate_apply_target(forbidden)
                raise AssertionError(
                    f"the validator must refuse the system root {forbidden}"
                )
            except ValueError:
                pass
        try:
            mod._validate_apply_target("/nonexistent-top-level")
            raise AssertionError(
                "the validator must refuse a bare top-level path"
            )
        except ValueError:
            pass
        # A multi-segment forbidden root: only the forbidden-set mechanism
        # refuses this one, so the set itself is pinned distinctly from the
        # top-level-parts check above.
        for nested_forbidden in ("/usr/bin", "/usr/sbin"):
            try:
                mod._validate_apply_target(nested_forbidden)
                raise AssertionError(
                    "the validator must refuse the nested system directory "
                    f"{nested_forbidden}"
                )
            except ValueError:
                pass
        project = tempfile.mkdtemp(prefix="apply-target-")
        try:
            mod._validate_apply_target(project)  # must not raise
        finally:
            os.rmdir(project)
    finally:
        restore()


# ---------------------------------------------------------------------------
# AB2 -- containment: descendants in, traversal/lookalikes/symlinks out
# ---------------------------------------------------------------------------
def test_ab2_containment_check_resolves_and_confines():
    mod, restore = _load()
    try:
        base = tempfile.mkdtemp(prefix="contain-")
        target = os.path.join(base, "proj")
        lookalike = os.path.join(base, "projX")
        outside = os.path.join(base, "outside")
        os.makedirs(target)
        os.makedirs(lookalike)
        os.makedirs(outside)

        assert mod._is_within_target(target, target) is True, (
            "the target itself must count as contained"
        )
        inside = os.path.join(target, "sub", "file.txt")
        assert mod._is_within_target(target, inside) is True, (
            "a descendant path must count as contained"
        )
        escape = os.path.join(target, "..", "outside", "evil.txt")
        assert mod._is_within_target(target, escape) is False, (
            "a parent-traversal path must be refused"
        )
        sibling = os.path.join(lookalike, "file.txt")
        assert mod._is_within_target(target, sibling) is False, (
            "a sibling-prefix lookalike must be refused"
        )
        link = os.path.join(target, "ln")
        os.symlink(outside, link)
        via_link = os.path.join(link, "evil.txt")
        assert mod._is_within_target(target, via_link) is False, (
            "a path resolving outside through a symlink must be refused"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AB3 -- apply refuses the escaping file and applies the contained one
# ---------------------------------------------------------------------------
def test_ab3_apply_refuses_escapes_and_writes_contained_files():
    mod, restore = _load()
    try:
        base = tempfile.mkdtemp(prefix="apply-")
        target = os.path.join(base, "proj")
        os.makedirs(target)
        agent = _agent(mod)
        good = mod.FileDiff(
            path="inside/ok.txt", is_new=True, modified_content="fine\n",
        )
        evil = mod.FileDiff(
            path="../escaped.txt", is_new=True, modified_content="leak\n",
        )
        _seed_diffs(mod, agent, [good, evil])

        result = agent.apply_changes(target_path=target)

        assert result["applied"] == 1, (
            f"exactly the contained file must apply, got {result}"
        )
        written = os.path.join(target, "inside", "ok.txt")
        assert os.path.isfile(written), "the contained file must be written"
        assert not os.path.exists(os.path.join(base, "escaped.txt")), (
            "the escaping path must never be written outside the target"
        )
        assert result["errors"] and "escape" in result["errors"][0]["error"], (
            f"the escape must be recorded as a refusal, got {result['errors']}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AB4 -- checkpoint-before-apply is forced on at both layers
# ---------------------------------------------------------------------------
def test_ab4_checkpoint_before_apply_cannot_be_disabled():
    mod, restore = _load()
    try:
        cfg = mod.CodingAgentConfig()
        cfg.checkpoint_before_apply = False
        agent = mod.CodingAgent(
            sandbox_session=SimpleNamespace(active=False), config=cfg,
        )
        assert agent.config.checkpoint_before_apply is True, (
            "the constructor must force the checkpoint switch back on"
        )

        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False,
        ) as fh:
            fh.write("checkpoint_before_apply: false\nmax_iterations: 3\n")
            crafted = fh.name
        original_path = mod._CONFIG_PATH
        try:
            mod._CONFIG_PATH = crafted
            loaded = mod._load_config()
        finally:
            mod._CONFIG_PATH = original_path
            os.unlink(crafted)
        assert loaded.max_iterations == 3, (
            "the crafted configuration must actually be read"
        )
        assert loaded.checkpoint_before_apply is True, (
            "the loader must force the checkpoint switch back on"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AB5 -- contents changed after review fail the integrity hash
# ---------------------------------------------------------------------------
def test_ab5_apply_refuses_when_review_hash_no_longer_matches():
    mod, restore = _load()
    try:
        base = tempfile.mkdtemp(prefix="apply-hash-")
        target = os.path.join(base, "proj")
        os.makedirs(target)
        agent = _agent(mod)
        diff = mod.FileDiff(
            path="inside/ok.txt", is_new=True, modified_content="reviewed\n",
        )
        _seed_diffs(mod, agent, [diff])
        diff.modified_content = "tampered\n"  # drift after review

        try:
            agent.apply_changes(target_path=target)
            raise AssertionError(
                "apply must refuse when the review hash no longer matches"
            )
        except RuntimeError as exc:
            assert "integrity" in str(exc).lower(), (
                f"the refusal must name the integrity check, got {exc}"
            )
        assert not os.path.exists(os.path.join(target, "inside")), (
            "nothing may be written when the integrity check fails"
        )
    finally:
        restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
