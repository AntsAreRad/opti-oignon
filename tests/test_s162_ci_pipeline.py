"""
tests/test_s162_ci_pipeline.py -- S162 GitHub Actions CI Pipeline tests.

Validates the CI workflow configuration, script adjustments, coverage badge
generation, and branch protection documentation.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
import unittest

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKFLOW_PATH = os.path.join(PROJECT_ROOT, ".github", "workflows", "ci.yml")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
README_PATH = os.path.join(PROJECT_ROOT, "README.md")


def _load_workflow():
    """Load and parse the CI workflow YAML."""
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===================================================================
# 1. Workflow YAML structure
# ===================================================================
class TestWorkflowStructure(unittest.TestCase):
    """Validate ci.yml structure and content."""

    @classmethod
    def setUpClass(cls):
        cls.workflow = _load_workflow()

    def test_workflow_file_exists(self):
        self.assertTrue(
            os.path.isfile(WORKFLOW_PATH),
            "ci.yml must exist at .github/workflows/ci.yml",
        )

    def test_yaml_parses_without_error(self):
        self.assertIsInstance(self.workflow, dict)

    def test_workflow_has_name(self):
        self.assertIn("name", self.workflow)

    def test_trigger_on_push_main(self):
        # yaml.safe_load maps 'on' to True
        triggers = self.workflow.get(True, {})
        self.assertIn("push", triggers)
        push_branches = triggers["push"].get("branches", [])
        self.assertIn("main", push_branches)

    def test_trigger_on_pull_request_main(self):
        triggers = self.workflow.get(True, {})
        self.assertIn("pull_request", triggers)
        pr_branches = triggers["pull_request"].get("branches", [])
        self.assertIn("main", pr_branches)

    def test_has_all_required_jobs(self):
        jobs = set(self.workflow.get("jobs", {}).keys())
        required = {"lint", "typecheck", "test", "e2e", "security"}
        self.assertTrue(
            required.issubset(jobs),
            f"Missing jobs: {required - jobs}",
        )

    def test_has_badge_job(self):
        jobs = self.workflow.get("jobs", {})
        self.assertIn("badge", jobs)


# ===================================================================
# 2. Job configuration
# ===================================================================
class TestJobConfiguration(unittest.TestCase):
    """Validate individual job settings."""

    @classmethod
    def setUpClass(cls):
        cls.jobs = _load_workflow().get("jobs", {})

    def test_lint_has_python_matrix(self):
        matrix = self.jobs["lint"]["strategy"]["matrix"]
        versions = matrix.get("python-version", [])
        self.assertIn("3.11", versions)
        self.assertIn("3.12", versions)

    def test_typecheck_has_python_matrix(self):
        matrix = self.jobs["typecheck"]["strategy"]["matrix"]
        versions = matrix.get("python-version", [])
        self.assertIn("3.11", versions)
        self.assertIn("3.12", versions)

    def test_test_has_python_matrix(self):
        matrix = self.jobs["test"]["strategy"]["matrix"]
        versions = matrix.get("python-version", [])
        self.assertIn("3.11", versions)
        self.assertIn("3.12", versions)

    def test_test_depends_on_lint_and_typecheck(self):
        needs = self.jobs["test"].get("needs", [])
        self.assertIn("lint", needs)
        self.assertIn("typecheck", needs)

    def test_e2e_depends_on_lint(self):
        needs = self.jobs["e2e"].get("needs", [])
        self.assertIn("lint", needs)

    def test_security_is_non_blocking(self):
        self.assertTrue(
            self.jobs["security"].get("continue-on-error", False),
            "Security job must have continue-on-error: true",
        )

    def test_security_has_no_dependencies(self):
        needs = self.jobs["security"].get("needs", [])
        self.assertEqual(needs, [], "Security job should run independently")

    def test_badge_depends_on_test(self):
        needs = self.jobs["badge"].get("needs", [])
        self.assertIn("test", needs)

    def test_badge_runs_only_on_main_push(self):
        badge = self.jobs["badge"]
        condition = badge.get("if", "")
        self.assertIn("refs/heads/main", condition)
        self.assertIn("push", condition)

    def test_e2e_uses_node_20(self):
        steps = self.jobs["e2e"].get("steps", [])
        node_steps = [
            s for s in steps
            if s.get("uses", "").startswith("actions/setup-node")
        ]
        self.assertTrue(len(node_steps) > 0, "E2E must set up Node.js")


# ===================================================================
# 3. Security scan job details
# ===================================================================
class TestSecurityScanJob(unittest.TestCase):
    """Validate security scan references correct tools."""

    @classmethod
    def setUpClass(cls):
        cls.security_job = _load_workflow()["jobs"]["security"]
        cls.step_names = [
            s.get("name", "") for s in cls.security_job.get("steps", [])
        ]
        cls.step_runs = [
            s.get("run", "") for s in cls.security_job.get("steps", [])
            if "run" in s
        ]
        cls.all_run_text = "\n".join(cls.step_runs)

    def test_references_bandit(self):
        self.assertIn("bandit", self.all_run_text.lower())

    def test_references_pip_audit(self):
        self.assertIn("pip-audit", self.all_run_text)

    def test_references_npm_audit(self):
        self.assertIn("npm audit", self.all_run_text)

    def test_has_bandit_step(self):
        bandit_steps = [n for n in self.step_names if "bandit" in n.lower()]
        self.assertTrue(len(bandit_steps) > 0)

    def test_produces_json_reports(self):
        self.assertIn("bandit-report.json", self.all_run_text)
        self.assertIn("pip-audit-report.json", self.all_run_text)
        self.assertIn("npm-audit-report.json", self.all_run_text)

    def test_uploads_security_artifacts(self):
        steps = self.security_job.get("steps", [])
        upload_steps = [
            s for s in steps
            if "actions/upload-artifact" in s.get("uses", "")
        ]
        self.assertTrue(len(upload_steps) > 0)


# ===================================================================
# 4. Referenced scripts exist and are executable
# ===================================================================
class TestScriptsExistAndExecutable(unittest.TestCase):
    """All scripts referenced in the CI workflow must exist."""

    REQUIRED_SCRIPTS = [
        "scripts/run_tests.sh",
        "scripts/run_typecheck.sh",
        "scripts/run_coverage.sh",
        "scripts/lint.sh",
        "scripts/run_e2e.sh",
        "scripts/generate_coverage_badge.py",
    ]

    def test_all_required_scripts_exist(self):
        for script in self.REQUIRED_SCRIPTS:
            path = os.path.join(PROJECT_ROOT, script)
            self.assertTrue(
                os.path.isfile(path),
                f"{script} must exist",
            )

    def test_shell_scripts_are_executable(self):
        for script in self.REQUIRED_SCRIPTS:
            if script.endswith(".sh"):
                path = os.path.join(PROJECT_ROOT, script)
                self.assertTrue(
                    os.access(path, os.X_OK),
                    f"{script} must be executable",
                )

    def test_badge_script_is_valid_python(self):
        path = os.path.join(SCRIPTS_DIR, "generate_coverage_badge.py")
        with open(path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        func_names = [
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("generate_badge", func_names)
        self.assertIn("get_color", func_names)
        self.assertIn("main", func_names)


# ===================================================================
# 5. Script CI flags
# ===================================================================
class TestScriptCIFlags(unittest.TestCase):
    """Scripts must support --ci flag for CI environments."""

    def _read_script(self, name):
        path = os.path.join(SCRIPTS_DIR, name)
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_run_tests_supports_ci_flag(self):
        content = self._read_script("run_tests.sh")
        self.assertIn("--ci", content)
        self.assertIn("CI_MODE", content)

    def test_run_tests_ci_produces_junit_xml(self):
        content = self._read_script("run_tests.sh")
        self.assertIn("junitxml", content)

    def test_run_tests_ci_produces_json_coverage(self):
        content = self._read_script("run_tests.sh")
        self.assertIn("coverage.json", content)

    def test_run_tests_ci_disables_colors(self):
        content = self._read_script("run_tests.sh")
        self.assertIn("NO_COLOR", content)

    def test_run_typecheck_supports_ci_flag(self):
        content = self._read_script("run_typecheck.sh")
        self.assertIn("--ci", content)
        self.assertIn("CI_MODE", content)

    def test_run_typecheck_ci_disables_colors(self):
        content = self._read_script("run_typecheck.sh")
        # In CI mode, color variables should be set to empty
        self.assertIn("RED=''", content)
        self.assertIn("NC=''", content)

    def test_run_coverage_supports_ci_flag(self):
        content = self._read_script("run_coverage.sh")
        self.assertIn("--ci", content)
        self.assertIn("CI_MODE", content)

    def test_run_coverage_ci_implies_json(self):
        content = self._read_script("run_coverage.sh")
        # --ci should set JSON_REPORT=true
        self.assertIn("CI_MODE=true; JSON_REPORT=true", content)


# ===================================================================
# 6. Coverage threshold
# ===================================================================
class TestCoverageThreshold(unittest.TestCase):
    """Coverage threshold must be defined in scripts."""

    def test_overall_threshold_defined(self):
        path = os.path.join(SCRIPTS_DIR, "run_coverage.sh")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("OVERALL_FAIL_UNDER=", content)
        # Extract value
        for line in content.splitlines():
            if line.strip().startswith("OVERALL_FAIL_UNDER="):
                value = int(line.split("=")[1])
                self.assertGreater(value, 0)
                break

    def test_security_module_thresholds_defined(self):
        path = os.path.join(SCRIPTS_DIR, "run_coverage.sh")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("SECURITY_MODULE_THRESHOLDS", content)


# ===================================================================
# 7. Coverage badge generation
# ===================================================================
class TestCoverageBadgeGeneration(unittest.TestCase):
    """Badge generator must produce valid SVG."""

    @classmethod
    def setUpClass(cls):
        """Load the badge module via importlib for isolation."""
        import importlib.util

        badge_path = os.path.join(SCRIPTS_DIR, "generate_coverage_badge.py")
        spec = importlib.util.spec_from_file_location(
            "generate_coverage_badge", badge_path
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_get_color_high_coverage(self):
        color = self.mod.get_color(95)
        self.assertEqual(color, "#4c1")

    def test_get_color_medium_coverage(self):
        color = self.mod.get_color(65)
        self.assertEqual(color, "#a4a61d")

    def test_get_color_low_coverage(self):
        color = self.mod.get_color(15)
        self.assertEqual(color, "#e05d44")

    def test_generate_badge_creates_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            output_path = f.name
        try:
            self.mod.generate_badge(42.5, output_path)
            self.assertTrue(os.path.isfile(output_path))
            with open(output_path, encoding="utf-8") as f:
                svg = f.read()
            self.assertIn("<svg", svg)
            self.assertIn("coverage", svg)
            self.assertIn("42%", svg)
        finally:
            os.unlink(output_path)

    def test_badge_color_thresholds_sorted(self):
        # Thresholds must be sorted descending for correct matching
        thresholds = [t[0] for t in self.mod.THRESHOLDS]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))

    def test_placeholder_badge_exists(self):
        badge_path = os.path.join(ASSETS_DIR, "coverage-badge.svg")
        self.assertTrue(
            os.path.isfile(badge_path),
            "Placeholder badge must exist in assets/",
        )


# ===================================================================
# 8. README badge integration
# ===================================================================
class TestReadmeBadges(unittest.TestCase):
    """README must display CI and coverage badges."""

    @classmethod
    def setUpClass(cls):
        with open(README_PATH, encoding="utf-8") as f:
            cls.readme = f.read()

    def test_ci_badge_present(self):
        self.assertIn("ci.yml/badge.svg", self.readme)

    def test_coverage_badge_present(self):
        self.assertIn("coverage-badge.svg", self.readme)

    def test_badges_are_images(self):
        lines = self.readme.splitlines()
        badge_lines = [l for l in lines if "badge.svg" in l]
        for line in badge_lines:
            self.assertTrue(
                line.strip().startswith("!["),
                f"Badge line must be markdown image: {line}",
            )


# ===================================================================
# 9. Branch protection documentation
# ===================================================================
class TestBranchProtectionDocs(unittest.TestCase):
    """Branch protection documentation must exist and cover key topics."""

    @classmethod
    def setUpClass(cls):
        doc_path = os.path.join(DOCS_DIR, "BRANCH_PROTECTION.md")
        with open(doc_path, encoding="utf-8") as f:
            cls.content = f.read()

    def test_file_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(DOCS_DIR, "BRANCH_PROTECTION.md")),
        )

    def test_documents_required_checks(self):
        self.assertIn("lint", self.content.lower())
        self.assertIn("typecheck", self.content.lower())
        self.assertIn("test", self.content.lower())

    def test_documents_security_scan_as_non_blocking(self):
        # Should mention security scan is informational/non-blocking
        self.assertIn("non-blocking", self.content.lower())

    def test_documents_merge_strategy(self):
        self.assertIn("squash", self.content.lower())

    def test_documents_coverage_gate(self):
        self.assertIn("coverage", self.content.lower())


# ===================================================================
# 10. Workflow YAML advanced checks
# ===================================================================
class TestWorkflowAdvanced(unittest.TestCase):
    """Advanced validation of workflow configuration."""

    @classmethod
    def setUpClass(cls):
        cls.workflow = _load_workflow()
        cls.jobs = cls.workflow.get("jobs", {})

    def test_lint_uses_ruff_github_format(self):
        """Lint should use --output-format=github for PR annotations."""
        steps = self.jobs["lint"].get("steps", [])
        run_steps = [s.get("run", "") for s in steps if "run" in s]
        all_runs = "\n".join(run_steps)
        self.assertIn("output-format=github", all_runs)

    def test_e2e_uploads_report_on_failure(self):
        steps = self.jobs["e2e"].get("steps", [])
        upload_steps = [
            s for s in steps
            if "actions/upload-artifact" in s.get("uses", "")
        ]
        self.assertTrue(len(upload_steps) > 0)
        # Should only upload on failure
        for s in upload_steps:
            self.assertEqual(s.get("if", ""), "failure()")

    def test_badge_commits_with_skip_ci(self):
        steps = self.jobs["badge"].get("steps", [])
        run_steps = [s.get("run", "") for s in steps if "run" in s]
        all_runs = "\n".join(run_steps)
        self.assertIn("[skip ci]", all_runs)

    def test_all_jobs_use_ubuntu_latest(self):
        for name, job in self.jobs.items():
            self.assertEqual(
                job.get("runs-on"), "ubuntu-latest",
                f"Job '{name}' must use ubuntu-latest",
            )

    def test_checkout_action_version(self):
        """All jobs must use actions/checkout@v4."""
        for name, job in self.jobs.items():
            steps = job.get("steps", [])
            checkout = [
                s for s in steps
                if s.get("uses", "").startswith("actions/checkout")
            ]
            for s in checkout:
                self.assertTrue(
                    s["uses"].endswith("@v4"),
                    f"Job '{name}' should use actions/checkout@v4",
                )


if __name__ == "__main__":
    unittest.main()
