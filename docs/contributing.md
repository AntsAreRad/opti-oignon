# Contributing

## Code style

- **Language:** all code comments, docstrings, and UI text must be in
  English. No French in code or UI.
- **No emojis** in code (comments, strings, variable names).
- **No hardcoded hex colors** in Svelte components. All colors use
  `--oo-*` CSS variables.
- **Python formatting:** follow ruff defaults. Run `scripts/lint.sh`
  before committing.
- **Type annotations:** all new functions should include type hints.
  Run `scripts/run_typecheck.sh` to verify.


## Module conventions

Every new Python module should include:

- A `FEATURE_AVAILABLE` flag for graceful degradation when optional
  dependencies are missing
- A `checkpoint_before_apply = True` sentinel (hardcoded, never
  overridable) for any module that modifies state
- Conditional imports with try/except for optional dependencies
- A module-level docstring describing purpose and session of origin


## Testing patterns

Tests use the **importlib isolation pattern** to avoid triggering the
full `opti_oignon` import chain (which requires Ollama):

```python
import importlib.util
import sys
from unittest.mock import MagicMock

# Pre-seed heavy dependencies
sys.modules.setdefault("ollama", MagicMock())
sys.modules.setdefault("chromadb", MagicMock())

# Load the module under test in isolation
spec = importlib.util.spec_from_file_location(
    "my_module",
    "opti_oignon/my_module.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
```

This pattern is used consistently across all 148+ test files. New
tests should follow it.

### Test expectations

- Each session typically produces 60-120 tests
- AST verification runs after every Python file creation or
  modification
- Stale version assertions in older test files are handled with
  `--deselect` rather than modifying them


## Database conventions

- **WAL mode** for all SQLite databases
- **Per-feature separate databases** (not one monolithic DB)
- **Module-level singletons** with `reset_*()` functions for test
  isolation
- **`safe_connect` wrapper** for all SQLCipher connections
- **Parameterized queries only** -- no f-string SQL


## Session workflow

Development follows a session-based workflow (S65, S66, ... S161+).
Each session:

1. Starts with reconnaissance of existing code
2. Implements features in sequential phases with confirmation gates
3. AST-verifies all modified Python files
4. Runs the test suite
5. Bumps the version in `__version__.py`
6. Produces four deliverables:
   - Project zip (all directories)
   - `PROMPT_S{N}.md` for the current session
   - `SESSION_TRACKING_S65_S{N}.md`
   - `PROMPT_S{N+1}.md` for the next session


## FastAPI conventions

- **Literal path segments** must be registered before parametric
  catch-alls (`/{id}`, `{model:path}`)
- **Deny-by-default** auth middleware on all new endpoints
- **Consistent error format:** `{"detail": "message"}`


## Zip packaging

Project zips must include all top-level directories:

- `opti_oignon/`, `frontend/`, `tests/`, `docs/`, `scripts/`,
  `assets/`, `data/`
- Plus root files: `README.md`, `CHANGELOG.md`, `mkdocs.yml`,
  `requirements-docs.txt`, etc.

Exclude: `__pycache__`, `.pyc`, `node_modules`, `site/`, build
artifacts, runtime databases.
