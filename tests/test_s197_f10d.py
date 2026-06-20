"""S197 F10d -- packaging (pyproject), Docker, docs build.

Per-fix and verification locks:
- PKG-01: .dockerignore restored (native-first build context).
- PKG-02: the compose frontend service is disabled (Dockerfile.frontend was
  never written); compose stays valid with backend + ollama.
- s106 install-posture supersessions (deselect-plus-reassert): version is
  hardcoded by design; launch.sh / install-desktop.sh / setup.py are gone;
  the install path is pyproject + `python -m opti_oignon` + the `oo` script.
- Verification: optional-group isolation, F-05, requirements/pyproject sync,
  mkdocs nav existence.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "docker-compose.yml"
REQS = ROOT / "requirements-backend.txt"
MKDOCS = ROOT / "mkdocs.yml"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# -- Verification: pyproject optional-group isolation + F-05 --


def test_optional_group_isolation():
    src = _read(PYPROJECT)
    all_grp = re.search(r"\nall = \[(.*?)\]", src, re.S).group(1)
    assert "auth,dev,docs" in all_grp.replace(" ", "").replace("\n", "")
    # Platform-specific extras stay OUT of [all].
    for excluded in ("llama", "sqlcipher", "veilid"):
        assert excluded not in all_grp, f"[all] must not pull {excluded}"
    # Each excluded group still exists as its own extra.
    for grp in ("llama", "sqlcipher", "veilid", "auth", "dev", "docs"):
        assert re.search(rf"\n{grp} = \[", src), f"missing optional group {grp}"


def test_version_hardcoded_not_dynamic():
    # Supersedes s106 TestVersionSync::test_pyproject_toml_dynamic_version:
    # the version is hardcoded by design (avoids the import chain).
    src = _read(PYPROJECT)
    assert 'version = "3.6.0"' in src
    assert 'dynamic = ["version"]' not in src


def test_f05_multipart_in_core_deps():
    src = _read(PYPROJECT)
    deps = re.search(r"\ndependencies = \[(.*?)\]", src, re.S).group(1)
    assert "python-multipart" in deps


# -- s106 supersession: install posture --


def test_s106_install_posture_superseded():
    # launch.sh / install-desktop.sh / setup.py were removed in favour of
    # pyproject + `python -m opti_oignon` + the `oo` console script.
    for gone in ("launch.sh", "install-desktop.sh", "setup.py"):
        assert not (ROOT / gone).exists(), f"{gone} should be absent"
    assert (ROOT / "opti_oignon" / "__main__.py").exists()
    src = _read(PYPROJECT)
    assert 'oo = "opti_oignon.cli.main:cli"' in src


# -- PKG-01: .dockerignore restored --


def test_pkg01_dockerignore_present_and_complete():
    assert DOCKERIGNORE.is_file(), ".dockerignore missing"
    lines = [
        l.strip()
        for l in _read(DOCKERIGNORE).splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    for needed in ("__pycache__/", "*.pyc", "tests/", "docs/", "node_modules/",
                   ".git/", ".github/", "frontend/", "*.db", "*.zip"):
        assert needed in lines, f".dockerignore missing {needed}"
    assert "*.key" in lines or "*.pem" in lines
    assert any("SESSION_TRACKING" in l for l in lines)


# -- PKG-02: frontend compose service disabled, compose still valid --


def test_pkg02_compose_frontend_disabled():
    import yaml
    data = yaml.safe_load(_read(COMPOSE))
    services = set(data.get("services", {}).keys())
    assert services == {"backend", "ollama"}, f"unexpected services: {services}"
    # Backend + ollama keep their healthchecks.
    assert "healthcheck" in data["services"]["backend"]
    assert "healthcheck" in data["services"]["ollama"]
    # No ACTIVE (uncommented) Dockerfile.frontend reference remains.
    active = re.sub(r"(?m)^\s*#.*$", "", _read(COMPOSE))
    assert "Dockerfile.frontend" not in active
    # The disable is documented.
    assert "PKG-02" in _read(COMPOSE)


# -- Verification: requirements-backend.txt in sync with pyproject core deps --


def _pkg_names(block: str):
    names = set()
    for m in re.finditer(r'([A-Za-z0-9_.\-]+)\s*>=', block):
        names.add(m.group(1).lower().replace("_", "-"))
    return names


def test_requirements_backend_in_sync():
    deps = re.search(r"\ndependencies = \[(.*?)\]", _read(PYPROJECT), re.S).group(1)
    pyproject_core = _pkg_names(deps)
    reqs = _pkg_names(_read(REQS))
    assert pyproject_core == reqs, (
        f"requirements-backend.txt drifted from pyproject core deps: "
        f"pyproject-only={pyproject_core - reqs}, reqs-only={reqs - pyproject_core}"
    )


# -- Verification: mkdocs nav files all exist (no removed-module references) --


def test_mkdocs_nav_files_exist():
    text = _read(MKDOCS)
    docs_dir = ROOT / (re.search(r"docs_dir:\s*(\S+)", text).group(1)
                       if "docs_dir:" in text else "docs")
    navpart = text[text.index("\nnav:"):]
    files = re.findall(r":\s*([A-Za-z0-9_\-/]+\.md)\s*$", navpart, re.M)
    missing = [f for f in files if not (docs_dir / f).exists()]
    assert files, "no nav entries parsed"
    assert missing == [], f"mkdocs nav references missing files: {missing}"
