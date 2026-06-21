#!/usr/bin/env python3
"""S141 — Type Annotation Pass (Public APIs) tests.

Verifies:
- py.typed marker exists (PEP 561)
- mypy configuration in pyproject.toml
- run_typecheck.sh exists and is executable
- CONTRIBUTING.md exists
- mypy_baseline.json exists and is valid
- Type annotations on key public functions (spot-checks)
- Version bump to 3.1.1
"""

import ast
import importlib.util
import json
import os
import stat
import unittest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG = os.path.join(_PROJECT_ROOT, "opti_oignon")


def _load_module(name: str, path: str):
    """Load a module without triggering __init__ import chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_public_functions(filepath: str) -> dict:
    """Return {name: has_return_annotation} for public functions in a file."""
    with open(filepath) as fh:
        tree = ast.parse(fh.read())
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                result[node.name] = node.returns is not None
    return result


def _annotation_coverage(filepath: str) -> float:
    """Return annotation coverage percentage for public functions."""
    funcs = _get_public_functions(filepath)
    if not funcs:
        return 100.0
    annotated = sum(1 for v in funcs.values() if v)
    return annotated / len(funcs) * 100


# ===========================================================================
# Goal 1 — Return type annotations on public APIs
# ===========================================================================

class TestAnnotationCoverage(unittest.TestCase):
    """Verify type annotation coverage on priority modules."""

    def test_encryption_annotations(self):
        """encryption.py public functions should all be annotated."""
        path = os.path.join(_PKG, "encryption.py")
        if not os.path.exists(path):
            self.skipTest("encryption.py not found")
        cov = _annotation_coverage(path)
        self.assertGreaterEqual(cov, 90.0, f"encryption.py coverage {cov:.1f}% < 90%")

    def test_executor_annotations(self):
        """executor.py public functions should all be annotated."""
        path = os.path.join(_PKG, "executor.py")
        if not os.path.exists(path):
            self.skipTest("executor.py not found")
        cov = _annotation_coverage(path)
        self.assertGreaterEqual(cov, 90.0, f"executor.py coverage {cov:.1f}% < 90%")

    def test_auth_annotations(self):
        """auth.py public functions should all be annotated."""
        path = os.path.join(_PKG, "auth.py")
        if not os.path.exists(path):
            self.skipTest("auth.py not found")
        cov = _annotation_coverage(path)
        self.assertGreaterEqual(cov, 90.0, f"auth.py coverage {cov:.1f}% < 90%")

    def test_conversation_annotations(self):
        """conversation.py public functions should all be annotated."""
        path = os.path.join(_PKG, "conversation.py")
        if not os.path.exists(path):
            self.skipTest("conversation.py not found")
        cov = _annotation_coverage(path)
        self.assertGreaterEqual(cov, 90.0, f"conversation.py coverage {cov:.1f}% < 90%")

    def test_memory_annotations(self):
        """memory.py public functions should all be annotated."""
        path = os.path.join(_PKG, "memory.py")
        if not os.path.exists(path):
            self.skipTest("memory.py not found")
        cov = _annotation_coverage(path)
        self.assertGreaterEqual(cov, 90.0, f"memory.py coverage {cov:.1f}% < 90%")

    def test_router_annotations(self):
        """router.py public functions should all be annotated."""
        path = os.path.join(_PKG, "router.py")
        if not os.path.exists(path):
            self.skipTest("router.py not found")
        cov = _annotation_coverage(path)
        self.assertGreaterEqual(cov, 90.0, f"router.py coverage {cov:.1f}% < 90%")

    def test_sandbox_manager_annotations(self):
        """sandbox_manager.py public functions should all be annotated."""
        path = os.path.join(_PKG, "sandbox_manager.py")
        if not os.path.exists(path):
            self.skipTest("sandbox_manager.py not found")
        cov = _annotation_coverage(path)
        self.assertGreaterEqual(cov, 90.0, f"sandbox_manager.py coverage {cov:.1f}% < 90%")

    def test_security_mode_annotations(self):
        """security_mode.py public functions should all be annotated."""
        path = os.path.join(_PKG, "security_mode.py")
        if not os.path.exists(path):
            self.skipTest("security_mode.py not found")
        cov = _annotation_coverage(path)
        self.assertGreaterEqual(cov, 90.0, f"security_mode.py coverage {cov:.1f}% < 90%")

    def test_all_route_files_above_90_percent(self):
        """Every routes_*.py file should have >=90% annotation coverage."""
        api_dir = os.path.join(_PKG, "api")
        if not os.path.isdir(api_dir):
            self.skipTest("api/ directory not found")
        failures = []
        count = 0
        for fname in sorted(os.listdir(api_dir)):
            if not fname.startswith("routes_") or not fname.endswith(".py"):
                continue
            path = os.path.join(api_dir, fname)
            cov = _annotation_coverage(path)
            count += 1
            if cov < 90.0:
                failures.append(f"{fname}: {cov:.1f}%")
        self.assertGreater(count, 0, "No route files found")
        self.assertEqual(failures, [], f"Route files below 90%: {failures}")

    def test_overall_priority_coverage_above_90(self):
        """Overall annotation coverage on priority modules must be >=90%."""
        priority = [
            "auth.py", "auth_2fa.py", "encryption.py", "db_encryption.py",
            "conversation.py", "memory.py", "router.py", "executor.py",
            "sandbox_manager.py", "security_mode.py", "pqc_signatures.py",
        ]
        api_dir = os.path.join(_PKG, "api")
        if os.path.isdir(api_dir):
            for f in os.listdir(api_dir):
                if f.startswith("routes_") and f.endswith(".py"):
                    priority.append(os.path.join("api", f))

        total_pub = 0
        total_ann = 0
        for mod in priority:
            path = os.path.join(_PKG, mod)
            if not os.path.exists(path):
                continue
            funcs = _get_public_functions(path)
            total_pub += len(funcs)
            total_ann += sum(1 for v in funcs.values() if v)

        self.assertGreater(total_pub, 0)
        pct = total_ann / total_pub * 100
        self.assertGreaterEqual(
            pct, 90.0,
            f"Overall priority annotation coverage {pct:.1f}% < 90%",
        )


# ===========================================================================
# Goal 1 — Spot-check specific key functions
# ===========================================================================

class TestAnnotationSpotChecks(unittest.TestCase):
    """Spot-check that specific high-value public functions are annotated."""

    def _assert_annotated(self, filepath: str, func_name: str):
        """Assert that a specific function has a return type annotation."""
        funcs = _get_public_functions(filepath)
        self.assertIn(func_name, funcs, f"{func_name} not found in {filepath}")
        self.assertTrue(
            funcs[func_name],
            f"{func_name} in {filepath} has no return type annotation",
        )

    def test_encryption_secure_key_from_bytes(self):
        self._assert_annotated(os.path.join(_PKG, "encryption.py"), "secure_key_from_bytes")

    def test_encryption_as_bytes(self):
        self._assert_annotated(os.path.join(_PKG, "encryption.py"), "as_bytes")

    def test_encryption_wipe(self):
        self._assert_annotated(os.path.join(_PKG, "encryption.py"), "wipe")

    def test_encryption_is_wiped(self):
        self._assert_annotated(os.path.join(_PKG, "encryption.py"), "is_wiped")

    def test_executor_execute_cascade(self):
        self._assert_annotated(os.path.join(_PKG, "executor.py"), "execute_cascade")

    def test_executor_execute_speculative(self):
        self._assert_annotated(os.path.join(_PKG, "executor.py"), "execute_speculative")

    def test_executor_last_prompt_budget(self):
        self._assert_annotated(os.path.join(_PKG, "executor.py"), "last_prompt_budget")

    def test_executor_last_compression_result(self):
        self._assert_annotated(os.path.join(_PKG, "executor.py"), "last_compression_result")

    def test_routes_auth_login(self):
        path = os.path.join(_PKG, "api", "routes_auth.py")
        if os.path.exists(path):
            self._assert_annotated(path, "login")

    def test_routes_auth_register(self):
        path = os.path.join(_PKG, "api", "routes_auth.py")
        if os.path.exists(path):
            self._assert_annotated(path, "register")

    def test_routes_chat_stream(self):
        path = os.path.join(_PKG, "api", "routes_chat.py")
        if os.path.exists(path):
            self._assert_annotated(path, "chat_stream")

    def test_routes_conversations_list(self):
        path = os.path.join(_PKG, "api", "routes_conversations.py")
        if os.path.exists(path):
            self._assert_annotated(path, "list_conversations")

    def test_routes_security_get_status(self):
        path = os.path.join(_PKG, "api", "routes_security.py")
        if os.path.exists(path):
            self._assert_annotated(path, "get_security_status")

    def test_routes_rag_ingest(self):
        path = os.path.join(_PKG, "api", "routes_rag.py")
        if os.path.exists(path):
            self._assert_annotated(path, "ingest_document")

    def test_routes_sandbox_create(self):
        path = os.path.join(_PKG, "api", "routes_sandbox.py")
        if os.path.exists(path):
            self._assert_annotated(path, "create_sandbox")

    def test_routes_memory_list_facts(self):
        path = os.path.join(_PKG, "api", "routes_memory.py")
        if os.path.exists(path):
            self._assert_annotated(path, "list_facts")

    def test_routes_plugins_list(self):
        path = os.path.join(_PKG, "api", "routes_plugins.py")
        if os.path.exists(path):
            self._assert_annotated(path, "list_plugins")


# ===========================================================================
# Goal 2 — PEP 561 compliance
# ===========================================================================

class TestPEP561(unittest.TestCase):
    """Verify PEP 561 py.typed marker."""

    def test_py_typed_exists(self):
        """py.typed marker must exist in opti_oignon/."""
        path = os.path.join(_PKG, "py.typed")
        self.assertTrue(os.path.exists(path), "opti_oignon/py.typed not found")

    def test_py_typed_in_package_data(self):
        """pyproject.toml package-data must include py.typed."""
        toml_path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(toml_path) as fh:
            content = fh.read()
        self.assertIn("py.typed", content, "py.typed not in pyproject.toml package-data")


# ===========================================================================
# Goal 3 — mypy configuration
# ===========================================================================

class TestMypyConfig(unittest.TestCase):
    """Verify mypy configuration in pyproject.toml."""

    @classmethod
    def setUpClass(cls):
        toml_path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(toml_path) as fh:
            cls.content = fh.read()

    def test_mypy_section_exists(self):
        """[tool.mypy] section must exist."""
        self.assertIn("[tool.mypy]", self.content)

    def test_mypy_python_version(self):
        """mypy must target Python 3.10."""
        self.assertIn('python_version = "3.10"', self.content)

    def test_mypy_ignore_missing_imports(self):
        """mypy must ignore missing imports."""
        self.assertIn("ignore_missing_imports = true", self.content)

    def test_mypy_show_error_codes(self):
        """mypy must show error codes."""
        self.assertIn("show_error_codes = true", self.content)

    def test_mypy_no_implicit_optional(self):
        """mypy must enforce no implicit optional."""
        self.assertIn("no_implicit_optional = true", self.content)

    def test_mypy_excludes_plugins(self):
        """mypy must exclude plugins directory."""
        self.assertIn("opti_oignon/plugins/", self.content)

    def test_mypy_baseline_file_exists(self):
        """mypy_baseline.json must exist."""
        path = os.path.join(_PROJECT_ROOT, "mypy_baseline.json")
        self.assertTrue(os.path.exists(path), "mypy_baseline.json not found")

    def test_mypy_baseline_valid_json(self):
        """mypy_baseline.json must be valid JSON with required keys."""
        path = os.path.join(_PROJECT_ROOT, "mypy_baseline.json")
        with open(path) as fh:
            data = json.load(fh)
        self.assertIn("mypy_baseline_errors", data)
        self.assertIsInstance(data["mypy_baseline_errors"], int)
        self.assertGreaterEqual(data["mypy_baseline_errors"], 0)

    def test_mypy_baseline_has_metadata(self):
        """mypy_baseline.json should have version and date."""
        path = os.path.join(_PROJECT_ROOT, "mypy_baseline.json")
        with open(path) as fh:
            data = json.load(fh)
        self.assertIn("version", data)
        self.assertIn("date", data)


# ===========================================================================
# Goal 4 — CI-ready typecheck script
# ===========================================================================

class TestTypecheckScript(unittest.TestCase):
    """Verify run_typecheck.sh exists and is properly configured."""

    def test_script_exists(self):
        """scripts/run_typecheck.sh must exist."""
        path = os.path.join(_PROJECT_ROOT, "scripts", "run_typecheck.sh")
        self.assertTrue(os.path.exists(path), "run_typecheck.sh not found")

    def test_script_is_executable(self):
        """scripts/run_typecheck.sh must be executable."""
        path = os.path.join(_PROJECT_ROOT, "scripts", "run_typecheck.sh")
        mode = os.stat(path).st_mode
        self.assertTrue(mode & stat.S_IXUSR, "run_typecheck.sh not executable")

    def test_script_has_shebang(self):
        """scripts/run_typecheck.sh must have bash shebang."""
        path = os.path.join(_PROJECT_ROOT, "scripts", "run_typecheck.sh")
        with open(path) as fh:
            first_line = fh.readline().strip()
        self.assertIn("bash", first_line)

    def test_script_references_mypy(self):
        """scripts/run_typecheck.sh must call mypy."""
        path = os.path.join(_PROJECT_ROOT, "scripts", "run_typecheck.sh")
        with open(path) as fh:
            content = fh.read()
        self.assertIn("mypy", content)

    def test_script_references_baseline(self):
        """scripts/run_typecheck.sh must reference the baseline."""
        path = os.path.join(_PROJECT_ROOT, "scripts", "run_typecheck.sh")
        with open(path) as fh:
            content = fh.read()
        self.assertIn("baseline", content.lower())

    def test_script_has_update_mode(self):
        """scripts/run_typecheck.sh must support --update flag."""
        path = os.path.join(_PROJECT_ROOT, "scripts", "run_typecheck.sh")
        with open(path) as fh:
            content = fh.read()
        self.assertIn("--update", content)


# ===========================================================================
# Goal 5 — CONTRIBUTING.md
# ===========================================================================

class TestContributing(unittest.TestCase):
    """Verify CONTRIBUTING.md exists and covers key topics."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(_PROJECT_ROOT, "CONTRIBUTING.md")
        if os.path.exists(path):
            with open(path) as fh:
                cls.content = fh.read()
        else:
            cls.content = ""

    def test_file_exists(self):
        """CONTRIBUTING.md must exist."""
        path = os.path.join(_PROJECT_ROOT, "CONTRIBUTING.md")
        self.assertTrue(os.path.exists(path), "CONTRIBUTING.md not found")

    def test_has_code_style_section(self):
        """CONTRIBUTING.md must cover code style."""
        self.assertIn("Code Style", self.content)

    def test_has_type_annotation_section(self):
        """CONTRIBUTING.md must cover type annotations."""
        self.assertIn("Type Annotation", self.content)

    def test_has_testing_section(self):
        """CONTRIBUTING.md must cover testing."""
        self.assertIn("Testing", self.content)

    def test_has_importlib_pattern(self):
        """CONTRIBUTING.md must document the importlib test isolation pattern."""
        self.assertIn("importlib", self.content)

    def test_has_pr_checklist(self):
        """CONTRIBUTING.md must include a PR checklist."""
        self.assertIn("PR Checklist", self.content)

    def test_has_css_variables(self):
        """CONTRIBUTING.md must document --oo-* CSS variables."""
        self.assertIn("--oo-", self.content)

    def test_has_security_section(self):
        """CONTRIBUTING.md must cover security considerations."""
        self.assertIn("Security", self.content)

    def test_mentions_safe_connect(self):
        """CONTRIBUTING.md must mention safe_connect requirement."""
        self.assertIn("safe_connect", self.content)

    def test_mentions_ast_verify(self):
        """CONTRIBUTING.md must mention AST verification."""
        self.assertIn("ast", self.content.lower())


# ===========================================================================
# Goal 6 — Version bump
# ===========================================================================

class TestVersionBump(unittest.TestCase):
    """Verify version is 3.1.1."""

    def test_version_file(self):
        """__version__.py must contain 3.1.1."""
        path = os.path.join(_PKG, "__version__.py")
        with open(path) as fh:
            content = fh.read()
        self.assertIn("3.1.1", content)

    def test_version_value(self):
        """Loaded __version__ must be 3.1.1."""
        mod = _load_module(
            "opti_oignon.__version__",
            os.path.join(_PKG, "__version__.py"),
        )
        self.assertEqual(mod.__version__, "3.1.1")


# ===========================================================================
# Bonus — AST validity of all modified files
# ===========================================================================

class TestASTValidity(unittest.TestCase):
    """Verify all Python files in opti_oignon/ parse without errors."""

    def test_all_core_files_parse(self):
        """All .py files in opti_oignon/ must be valid Python."""
        errors = []
        for root, dirs, files in os.walk(_PKG):
            # Skip plugins (third-party entry points)
            if "plugins" in root:
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path) as fh:
                        ast.parse(fh.read())
                except SyntaxError as e:
                    errors.append(f"{path}: {e}")
        self.assertEqual(errors, [], "AST errors:\n" + "\n".join(errors))

    def test_route_files_parse(self):
        """All routes_*.py files must be valid Python."""
        api_dir = os.path.join(_PKG, "api")
        if not os.path.isdir(api_dir):
            self.skipTest("api/ not found")
        errors = []
        for fname in sorted(os.listdir(api_dir)):
            if not fname.startswith("routes_") or not fname.endswith(".py"):
                continue
            path = os.path.join(api_dir, fname)
            try:
                with open(path) as fh:
                    ast.parse(fh.read())
            except SyntaxError as e:
                errors.append(f"{fname}: {e}")
        self.assertEqual(errors, [], "Route AST errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
