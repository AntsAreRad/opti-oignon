#!/usr/bin/env python3
"""
S183 / P-01: the repository and the release zip must not ship runtime state.

Note on method: importing the opti_oignon package has side effects that create
runtime databases inside the source tree (auth.db, conversations.db, ...), so an
on-disk absence assertion would be order-dependent and flaky in any session that
imports the package. The durable, non-flaky guarantees are asserted instead:

- a .gitignore exists and covers every runtime class, including the K-01 keyfile,
  so git-based packaging never tracks runtime state (the *.db catch-all ignores
  every database regardless of where the app writes it);
- the data tree is excluded from setuptools packaging and is not itself a package,
  so wheels/sdists never carry it;
- the empty seed configuration files were not removed by mistake.

The release zip is cleaned at packaging time (the data dirs are emptied of runtime
state and the zip step excludes the runtime classes) before signing.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_GITIGNORE_PATTERNS = [
    "*.db",
    "data/*.db",
    "opti_oignon/data/*.db",
    ".audit_chain_anchor",
    "data/plugin_logs/",
    "data/.keyfile",
    "fingerprint.db",
    "opti_oignon/data/project_chroma/",
]


def _gitignore_lines():
    gi = ROOT / ".gitignore"
    if not gi.is_file():
        return []
    return [ln.strip() for ln in gi.read_text(encoding="utf-8").splitlines()]


class TestGitignoreCoversRuntimeState:
    def test_gitignore_exists(self):
        assert (ROOT / ".gitignore").is_file()

    def test_required_patterns_present(self):
        lines = _gitignore_lines()
        missing = [p for p in REQUIRED_GITIGNORE_PATTERNS if p not in lines]
        assert not missing, f".gitignore missing patterns: {missing}"

    def test_db_catchall_ignores_every_database(self):
        # The global *.db rule guarantees no database is ever tracked, wherever
        # the app chooses to write it.
        assert "*.db" in _gitignore_lines()

    def test_keyfile_ignored(self):
        # Pairs with K-01: the keyfile must never be committed or backed up.
        assert "data/.keyfile" in _gitignore_lines()


class TestPackagingExcludesDataTree:
    def test_pyproject_excludes_data_package(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "opti_oignon.data*" in text

    def test_data_dir_is_not_a_package(self):
        # No __init__.py means setuptools never auto-discovers it as a package,
        # so its contents cannot be bundled even without the explicit exclude.
        assert not (ROOT / "opti_oignon" / "data" / "__init__.py").exists()


class TestSeedConfigPreserved:
    """Guard against over-deletion: the empty seed config must remain."""

    def test_seed_yaml_kept(self):
        seeds = [
            ROOT / "opti_oignon" / "data" / "system_presets.yaml",
            ROOT / "opti_oignon" / "data" / "user_presets.yaml",
            ROOT / "opti_oignon" / "data" / "pipelines_custom.yaml",
        ]
        missing = [str(p) for p in seeds if not p.is_file()]
        assert not missing, f"seed config removed by mistake: {missing}"
