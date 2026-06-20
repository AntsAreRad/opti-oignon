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
