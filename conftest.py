"""
Root pytest configuration for Opti-Oignon.

This file is auto-loaded by pytest before collecting tests. It configures
asyncio mode for pytest-asyncio so async tests using @pytest.mark.asyncio
run correctly without per-test marker configuration.

It also adds the project root to sys.path so that 'opti_oignon' imports
resolve when tests are run from the project root or from a venv install.
"""

import os
import sys

# Make the project root importable when tests are run with pytest from
# the project root directory (in addition to the editable/regular install).
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def pytest_configure(config):
    """Register custom markers and configure asyncio mode."""
    # Register the 'asyncio' marker explicitly so that environments without
    # pytest-asyncio installed do not emit PytestUnknownMarkWarning.
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as an async coroutine to run via pytest-asyncio.",
    )

    # If pytest-asyncio is installed, set its mode to 'auto' so that async
    # test functions are detected without explicit markers. We do this here
    # rather than via pyproject.toml so that test runs without the plugin
    # do not fail (they simply skip async tests).
    try:
        import pytest_asyncio  # noqa: F401
        # asyncio_mode is read from the [tool.pytest.ini_options] table or
        # from this dynamic configuration. Setting it here ensures it applies
        # regardless of where pytest is invoked.
        if not config.getini("asyncio_mode"):
            config._inicache["asyncio_mode"] = "auto"
    except ImportError:
        pass
    except (ValueError, KeyError):
        # asyncio_mode ini option not registered (older pytest-asyncio versions)
        pass


# ---------------------------------------------------------------------------
# Public-repo guard: skip per-session meta-tests that read development
# bookkeeping documents (roadmaps, specs, inventories) which are intentionally
# not shipped in the public repository. Such a test is ignored only when the
# bookkeeping file it references is genuinely absent, so it still runs in a
# full development checkout where those files exist.
# ---------------------------------------------------------------------------
import pathlib as _pl
import re as _re

_REPO_ROOT = _pl.Path(__file__).parent.resolve()
_TESTS_DIR = _REPO_ROOT / "tests"
_MD_REF = _re.compile(r"""["']([A-Za-z0-9_./-]+\.md)["']""")
_BOOKKEEPING_HINTS = (
    "ROADMAP", "_SPEC", "INVENTORY", "AUDIT", "SHAKEDOWN", "LIVE_WALK",
    "REDESIGN", "ODYSSEUS", "VEILID", "MOBILE_", "PROMPT_S",
    "SESSION_TRACKING", "BASELINE", "DEBT_LOT", "FIX_REPORT", "_E2E_",
    "SIDE_QUESTS", "THREAT_MODEL",
)

collect_ignore = []
if _TESTS_DIR.is_dir():
    for _tf in sorted(_TESTS_DIR.glob("test_*.py")):
        try:
            _src = _tf.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for _ref in _MD_REF.findall(_src):
            _name = _pl.Path(_ref).name
            if any(_h in _name for _h in _BOOKKEEPING_HINTS) and not (_REPO_ROOT / _name).exists():
                collect_ignore.append(str(_tf.relative_to(_REPO_ROOT)))
                break
