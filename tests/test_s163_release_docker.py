#!/usr/bin/env python3
"""
Tests for S163 -- Release Automation + Docker.

Covers:
- Release workflow YAML structure and jobs
- Dockerfile.backend directives and security
- docker-compose.yml services, ports, health checks
- .dockerignore exclusions
- Changelog generation script
- requirements-backend.txt completeness
"""

import ast
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DOCKERFILE = ROOT / "Dockerfile.backend"
DOCKERFILE_FRONTEND = ROOT / "frontend" / "Dockerfile.frontend"
DOCKER_COMPOSE = ROOT / "docker-compose.yml"
DOCKERIGNORE = ROOT / ".dockerignore"
CHANGELOG_SCRIPT = ROOT / "scripts" / "generate_changelog.py"
REQUIREMENTS = ROOT / "requirements-backend.txt"
VERSION_FILE = ROOT / "opti_oignon" / "__version__.py"


# ===================================================================
# Test class: Release workflow YAML
# ===================================================================

class TestReleaseWorkflow:
    """Validate .github/workflows/release.yml structure."""

    @pytest.fixture(autouse=True)
    def load_workflow(self):
        assert RELEASE_WORKFLOW.exists(), "release.yml not found"
        with open(RELEASE_WORKFLOW) as f:
            self.data = yaml.safe_load(f)
        self.raw = RELEASE_WORKFLOW.read_text()

    def test_workflow_file_exists(self):
        assert RELEASE_WORKFLOW.is_file()

    def test_workflow_name(self):
        assert self.data.get("name") == "Release"

    def test_trigger_on_tag_push(self):
        # YAML parses 'on' as True (boolean)
        trigger = self.data.get(True) or self.data.get("on")
        assert trigger is not None, "No trigger defined"
        push = trigger.get("push", {})
        tags = push.get("tags", [])
        assert "v*" in tags, f"Expected v* tag trigger, got {tags}"

    def test_no_pull_request_trigger(self):
        trigger = self.data.get(True) or self.data.get("on")
        assert "pull_request" not in trigger, (
            "Release workflow should not trigger on pull requests"
        )

    def test_has_three_jobs(self):
        jobs = self.data.get("jobs", {})
        assert set(jobs.keys()) == {"validate", "build", "release"}

    def test_validate_job_outputs_version(self):
        validate = self.data["jobs"]["validate"]
        outputs = validate.get("outputs", {})
        assert "version" in outputs
        assert "tag" in outputs

    def test_validate_checks_version_file(self):
        assert "__version__.py" in self.raw, (
            "Validate job should reference __version__.py"
        )

    def test_validate_runs_lint(self):
        steps = self.data["jobs"]["validate"].get("steps", [])
        step_texts = [
            s.get("run", "") + s.get("name", "") for s in steps
        ]
        combined = " ".join(step_texts)
        assert "ruff" in combined, "Validate should run ruff lint"

    def test_build_depends_on_validate(self):
        build = self.data["jobs"]["build"]
        assert "validate" in build.get("needs", [])

    def test_build_has_changelog_step(self):
        steps = self.data["jobs"]["build"].get("steps", [])
        step_runs = [s.get("run", "") for s in steps]
        combined = " ".join(step_runs)
        assert "generate_changelog" in combined

    def test_build_creates_archive(self):
        steps = self.data["jobs"]["build"].get("steps", [])
        step_runs = [s.get("run", "") for s in steps]
        combined = " ".join(step_runs)
        assert "zip" in combined

    def test_build_generates_checksums(self):
        steps = self.data["jobs"]["build"].get("steps", [])
        step_runs = [s.get("run", "") for s in steps]
        combined = " ".join(step_runs)
        assert "sha256sum" in combined

    def test_build_gpg_signing_is_conditional(self):
        steps = self.data["jobs"]["build"].get("steps", [])
        gpg_steps = [
            s for s in steps
            if "gpg" in s.get("name", "").lower()
            or "sign" in s.get("name", "").lower()
        ]
        assert len(gpg_steps) >= 1, "Should have a GPG signing step"
        for step in gpg_steps:
            cond = step.get("if", "")
            assert "GPG_PRIVATE_KEY" in cond, (
                "GPG step should be conditional on secret availability"
            )

    def test_release_depends_on_build(self):
        release = self.data["jobs"]["release"]
        needs = release.get("needs", [])
        assert "build" in needs
        assert "validate" in needs

    def test_release_uses_gh_release_action(self):
        steps = self.data["jobs"]["release"].get("steps", [])
        uses_values = [s.get("uses", "") for s in steps]
        assert any(
            "softprops/action-gh-release" in u for u in uses_values
        ), "Should use softprops/action-gh-release"

    def test_release_handles_prerelease_tags(self):
        assert "prerelease" in self.raw, (
            "Release job should handle prerelease detection"
        )
        # Should detect rc, beta, alpha
        for marker in ("rc", "beta", "alpha"):
            assert marker in self.raw, (
                f"Prerelease detection should include '{marker}'"
            )

    def test_release_fail_on_unmatched_false(self):
        assert "fail_on_unmatched_files: false" in self.raw, (
            "Should not fail when .sig is missing (no GPG key)"
        )

    def test_contents_write_permission(self):
        perms = self.data.get("permissions", {})
        assert perms.get("contents") == "write"

    def test_fetch_depth_zero(self):
        # Both validate and build need full history
        for job_name in ("validate", "build"):
            steps = self.data["jobs"][job_name].get("steps", [])
            checkout_steps = [
                s for s in steps if "checkout" in s.get("uses", "")
            ]
            for step in checkout_steps:
                with_block = step.get("with", {})
                assert with_block.get("fetch-depth") == 0, (
                    f"{job_name} checkout should use fetch-depth: 0"
                )

    def test_upload_artifact_step(self):
        steps = self.data["jobs"]["build"].get("steps", [])
        upload_steps = [
            s for s in steps if "upload-artifact" in s.get("uses", "")
        ]
        assert len(upload_steps) >= 1


# ===================================================================
# Test class: Dockerfile.backend
# ===================================================================

class TestDockerfileBackend:
    """Validate Dockerfile.backend directives."""

    @pytest.fixture(autouse=True)
    def load_dockerfile(self):
        assert DOCKERFILE.exists(), "Dockerfile.backend not found"
        self.content = DOCKERFILE.read_text()
        self.lines = self.content.splitlines()

    def test_file_exists(self):
        assert DOCKERFILE.is_file()

    def test_has_security_notice(self):
        assert "SECURITY NOTICE" in self.content

    def test_recommends_native_deployment(self):
        lower = self.content.lower()
        assert "native" in lower and "recommended" in lower, (
            "Should recommend native deployment"
        )

    def test_multi_stage_build(self):
        from_lines = [l for l in self.lines if l.strip().startswith("FROM")]
        assert len(from_lines) >= 2, (
            f"Expected multi-stage build (>=2 FROM), got {len(from_lines)}"
        )

    def test_builder_stage_exists(self):
        assert "AS builder" in self.content

    def test_runtime_stage_exists(self):
        assert "AS runtime" in self.content

    def test_uses_slim_base(self):
        assert "python:3.12-slim" in self.content

    def test_requirements_copied_before_code(self):
        # Layer caching: requirements should be COPY'd before app code
        req_pos = self.content.find("requirements-backend.txt")
        code_pos = self.content.find("COPY opti_oignon/")
        assert req_pos < code_pos, (
            "requirements should be copied before application code"
        )

    def test_non_root_user(self):
        assert "USER opti" in self.content or "USER 1000" in self.content

    def test_user_created_with_groupadd(self):
        assert "groupadd" in self.content
        assert "useradd" in self.content

    def test_healthcheck_directive(self):
        healthcheck_lines = [
            l for l in self.lines if "HEALTHCHECK" in l
        ]
        assert len(healthcheck_lines) >= 1

    def test_healthcheck_targets_api_health(self):
        assert "/api/health" in self.content

    def test_healthcheck_has_start_period(self):
        assert "start-period" in self.content.lower() or "start_period" in self.content.lower()

    def test_exposes_port_8001(self):
        assert "EXPOSE 8001" in self.content

    def test_python_unbuffered(self):
        assert "PYTHONUNBUFFERED=1" in self.content

    def test_no_write_bytecode(self):
        assert "PYTHONDONTWRITEBYTECODE=1" in self.content

    def test_run_example_uses_localhost(self):
        assert "127.0.0.1:8001:8001" in self.content, (
            "Run example should bind to localhost only"
        )

    def test_mentions_docker_daemon_root(self):
        lower = self.content.lower()
        assert "root" in lower and "daemon" in lower, (
            "Should warn about Docker daemon running as root"
        )

    def test_mentions_bulbe_mode(self):
        lower = self.content.lower()
        assert "bulbe" in lower, (
            "Should reference Bulbe mode as the secure alternative"
        )

    def test_uvicorn_entrypoint(self):
        assert "uvicorn" in self.content
        assert "opti_oignon.api.app:app" in self.content


# ===================================================================
# Test class: docker-compose.yml
# ===================================================================

class TestDockerCompose:
    """Validate docker-compose.yml structure and security."""

    @pytest.fixture(autouse=True)
    def load_compose(self):
        assert DOCKER_COMPOSE.exists(), "docker-compose.yml not found"
        with open(DOCKER_COMPOSE) as f:
            self.data = yaml.safe_load(f)
        self.raw = DOCKER_COMPOSE.read_text()

    def test_file_exists(self):
        assert DOCKER_COMPOSE.is_file()

    def test_has_security_notice(self):
        assert "SECURITY NOTICE" in self.raw

    def test_recommends_native_deployment(self):
        lower = self.raw.lower()
        assert "native" in lower and "recommended" in lower

    def test_has_three_services(self):
        services = self.data.get("services", {})
        assert set(services.keys()) == {"backend", "frontend", "ollama"}

    def test_backend_port_localhost_only(self):
        ports = self.data["services"]["backend"].get("ports", [])
        for port in ports:
            assert port.startswith("127.0.0.1:"), (
                f"Backend port {port} not bound to localhost"
            )

    def test_frontend_port_localhost_only(self):
        ports = self.data["services"]["frontend"].get("ports", [])
        for port in ports:
            assert port.startswith("127.0.0.1:"), (
                f"Frontend port {port} not bound to localhost"
            )

    def test_ollama_no_exposed_ports(self):
        ports = self.data["services"]["ollama"].get("ports", [])
        assert len(ports) == 0, (
            "Ollama should not expose ports to host by default"
        )

    def test_backend_depends_on_ollama(self):
        deps = self.data["services"]["backend"].get("depends_on", {})
        assert "ollama" in deps

    def test_frontend_depends_on_backend(self):
        deps = self.data["services"]["frontend"].get("depends_on", {})
        assert "backend" in deps

    def test_backend_healthcheck(self):
        hc = self.data["services"]["backend"].get("healthcheck", {})
        assert "test" in hc
        test_cmd = " ".join(hc["test"]) if isinstance(hc["test"], list) else hc["test"]
        assert "/api/health" in test_cmd

    def test_frontend_healthcheck(self):
        hc = self.data["services"]["frontend"].get("healthcheck", {})
        assert "test" in hc

    def test_ollama_healthcheck(self):
        hc = self.data["services"]["ollama"].get("healthcheck", {})
        assert "test" in hc

    def test_health_dependency_condition(self):
        # Backend should wait for ollama to be healthy
        deps = self.data["services"]["backend"].get("depends_on", {})
        ollama_dep = deps.get("ollama", {})
        assert ollama_dep.get("condition") == "service_healthy"

    def test_named_volumes(self):
        volumes = self.data.get("volumes", {})
        assert "backend_data" in volumes
        assert "ollama_data" in volumes

    def test_ollama_image(self):
        image = self.data["services"]["ollama"].get("image", "")
        assert "ollama/ollama" in image

    def test_backend_ollama_host_env(self):
        env = self.data["services"]["backend"].get("environment", [])
        env_str = " ".join(str(e) for e in env)
        assert "OLLAMA_HOST" in env_str
        assert "ollama:11434" in env_str

    def test_restart_policy(self):
        for svc_name, svc in self.data["services"].items():
            assert svc.get("restart") == "unless-stopped", (
                f"{svc_name} should have restart: unless-stopped"
            )

    def test_mentions_podman_alternative(self):
        lower = self.raw.lower()
        assert "podman" in lower, (
            "Should mention Podman as a rootless alternative"
        )

    def test_warns_against_0000_binding(self):
        assert "0.0.0.0" in self.raw, (
            "Should warn against changing to 0.0.0.0"
        )


# ===================================================================
# Test class: .dockerignore
# ===================================================================

class TestDockerignore:
    """Validate .dockerignore exclusions."""

    @pytest.fixture(autouse=True)
    def load_dockerignore(self):
        assert DOCKERIGNORE.exists(), ".dockerignore not found"
        self.content = DOCKERIGNORE.read_text()
        self.lines = [
            l.strip() for l in self.content.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]

    def test_file_exists(self):
        assert DOCKERIGNORE.is_file()

    def test_excludes_pycache(self):
        assert "__pycache__/" in self.lines

    def test_excludes_pyc(self):
        assert "*.pyc" in self.lines

    def test_excludes_tests(self):
        assert "tests/" in self.lines

    def test_excludes_docs(self):
        assert "docs/" in self.lines

    def test_excludes_node_modules(self):
        assert "node_modules/" in self.lines

    def test_excludes_git(self):
        assert ".git/" in self.lines

    def test_excludes_github(self):
        assert ".github/" in self.lines

    def test_excludes_frontend(self):
        assert "frontend/" in self.lines

    def test_excludes_databases(self):
        assert "*.db" in self.lines

    def test_excludes_secrets(self):
        assert "*.key" in self.lines or "*.pem" in self.lines

    def test_excludes_session_tracking(self):
        assert any("SESSION_TRACKING" in l for l in self.lines)

    def test_excludes_archives(self):
        assert "*.zip" in self.lines


# ===================================================================
# Test class: Changelog generation script
# ===================================================================

class TestChangelogScript:
    """Validate scripts/generate_changelog.py."""

    def test_script_exists(self):
        assert CHANGELOG_SCRIPT.is_file()

    def test_script_executable(self):
        assert os.access(CHANGELOG_SCRIPT, os.X_OK)

    def test_ast_valid(self):
        source = CHANGELOG_SCRIPT.read_text()
        tree = ast.parse(source)
        assert tree is not None

    def test_has_main_function(self):
        source = CHANGELOG_SCRIPT.read_text()
        tree = ast.parse(source)
        func_names = [
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
        ]
        assert "main" in func_names

    def test_has_categorize_function(self):
        source = CHANGELOG_SCRIPT.read_text()
        tree = ast.parse(source)
        func_names = [
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
        ]
        assert "categorize_commits" in func_names

    def test_has_conventional_categories(self):
        source = CHANGELOG_SCRIPT.read_text()
        for category in ("feat", "fix", "security", "docs", "test", "ci"):
            assert category in source, (
                f"Missing conventional commit category: {category}"
            )

    def test_supports_markdown_format(self):
        source = CHANGELOG_SCRIPT.read_text()
        assert "format_markdown" in source

    def test_supports_text_format(self):
        source = CHANGELOG_SCRIPT.read_text()
        assert "format_text" in source

    def test_module_import(self):
        """Test that the script can be imported as a module."""
        spec = __import__("importlib").util.spec_from_file_location(
            "generate_changelog", str(CHANGELOG_SCRIPT)
        )
        mod = __import__("importlib").util.module_from_spec(spec)
        # Patch sys.argv to avoid argparse issues
        old_argv = sys.argv
        sys.argv = ["generate_changelog"]
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.argv = old_argv
        assert hasattr(mod, "CATEGORIES")
        assert hasattr(mod, "categorize_commits")
        assert hasattr(mod, "format_markdown")

    def test_categorize_commits_logic(self):
        """Test commit categorization with mock data."""
        spec = __import__("importlib").util.spec_from_file_location(
            "generate_changelog", str(CHANGELOG_SCRIPT)
        )
        mod = __import__("importlib").util.module_from_spec(spec)
        old_argv = sys.argv
        sys.argv = ["generate_changelog"]
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.argv = old_argv

        commits = [
            {"hash": "abc1234", "subject": "feat: add login", "body": ""},
            {"hash": "def5678", "subject": "fix: resolve crash", "body": ""},
            {"hash": "ghi9012", "subject": "security: patch XSS", "body": ""},
            {"hash": "jkl3456", "subject": "plain commit", "body": ""},
            {"hash": "mno7890", "subject": "docs(api): update ref", "body": ""},
        ]

        categorized, uncategorized = mod.categorize_commits(commits)

        assert len(categorized["feat"]) == 1
        assert len(categorized["fix"]) == 1
        assert len(categorized["security"]) == 1
        assert len(categorized["docs"]) == 1
        assert len(uncategorized) == 1
        assert uncategorized[0]["hash"] == "jkl3456"

    def test_format_markdown_output(self):
        """Test markdown formatting produces valid output."""
        spec = __import__("importlib").util.spec_from_file_location(
            "generate_changelog", str(CHANGELOG_SCRIPT)
        )
        mod = __import__("importlib").util.module_from_spec(spec)
        old_argv = sys.argv
        sys.argv = ["generate_changelog"]
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.argv = old_argv

        from collections import OrderedDict
        categorized = OrderedDict()
        for key in mod.CATEGORIES:
            categorized[key] = []
        categorized["feat"].append({
            "hash": "abc1234",
            "clean_message": "add login",
        })

        output = mod.format_markdown("v1.0.0", "v0.9.0", categorized, [])
        assert "## v1.0.0" in output
        assert "### Features" in output
        assert "add login (abc1234)" in output
        assert "v0.9.0...v1.0.0" in output

    def test_conventional_regex_with_scope(self):
        """Test that scoped prefixes like feat(auth): are parsed."""
        spec = __import__("importlib").util.spec_from_file_location(
            "generate_changelog", str(CHANGELOG_SCRIPT)
        )
        mod = __import__("importlib").util.module_from_spec(spec)
        old_argv = sys.argv
        sys.argv = ["generate_changelog"]
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.argv = old_argv

        match = mod.CONVENTIONAL_RE.match("feat(auth): add OAuth2")
        assert match is not None
        assert match.group("type") == "feat"
        assert match.group("message") == "add OAuth2"

    def test_end_to_end_with_git(self, tmp_path):
        """Test full changelog generation with a temporary git repo."""
        repo = tmp_path / "repo"
        repo.mkdir()

        def git(*args):
            return subprocess.run(
                ["git", *args],
                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
            )

        git("init", "-q")
        git("config", "user.email", "test@test.com")
        git("config", "user.name", "Test")

        (repo / "README.md").write_text("init")
        git("add", ".")
        git("commit", "-q", "-m", "chore: initial")
        git("tag", "v0.1.0")

        (repo / "a.txt").write_text("a")
        git("add", ".")
        git("commit", "-q", "-m", "feat: add feature A")

        (repo / "b.txt").write_text("b")
        git("add", ".")
        git("commit", "-q", "-m", "fix: resolve bug B")

        git("tag", "v0.2.0")

        result = subprocess.run(
            [sys.executable, str(CHANGELOG_SCRIPT)],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout
        assert "v0.2.0" in output
        assert "feature A" in output
        assert "bug B" in output


# ===================================================================
# Test class: requirements-backend.txt
# ===================================================================

class TestRequirementsBackend:
    """Validate requirements-backend.txt."""

    @pytest.fixture(autouse=True)
    def load_requirements(self):
        assert REQUIREMENTS.exists(), "requirements-backend.txt not found"
        self.content = REQUIREMENTS.read_text()
        self.lines = [
            l.strip() for l in self.content.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        # Extract package names (before >=, ==, etc.)
        self.packages = [
            re.split(r"[><=!]", l)[0].strip().lower()
            for l in self.lines
        ]

    def test_file_exists(self):
        assert REQUIREMENTS.is_file()

    def test_has_fastapi(self):
        assert "fastapi" in self.packages

    def test_has_uvicorn(self):
        assert "uvicorn" in self.packages

    def test_has_pydantic(self):
        assert "pydantic" in self.packages

    def test_has_ollama(self):
        assert "ollama" in self.packages

    def test_has_pyyaml(self):
        assert "pyyaml" in self.packages

    def test_has_httpx(self):
        assert "httpx" in self.packages

    def test_has_cryptography(self):
        assert "cryptography" in self.packages

    def test_has_bcrypt(self):
        assert "bcrypt" in self.packages

    def test_all_have_version_constraint(self):
        for line in self.lines:
            assert ">=" in line or "==" in line or "<=" in line, (
                f"Package line missing version constraint: {line}"
            )

    def test_referenced_in_dockerfile(self):
        dockerfile = DOCKERFILE.read_text()
        assert "requirements-backend.txt" in dockerfile


# ===================================================================
# Test class: Cross-file consistency
# ===================================================================

class TestCrossFileConsistency:
    """Validate consistency across release and Docker files."""

    def test_release_workflow_references_sign_script(self):
        raw = RELEASE_WORKFLOW.read_text()
        # The workflow uses gpg directly but should reference
        # the signing pattern from sign_release.sh
        assert "gpg" in raw

    def test_release_workflow_references_changelog_script(self):
        raw = RELEASE_WORKFLOW.read_text()
        assert "generate_changelog" in raw

    def test_dockerfile_port_matches_compose(self):
        compose = yaml.safe_load(DOCKER_COMPOSE.read_text())
        dockerfile = DOCKERFILE.read_text()

        backend_ports = compose["services"]["backend"]["ports"]
        # Extract container port from compose (e.g., 127.0.0.1:8000:8000)
        for port_mapping in backend_ports:
            container_port = port_mapping.split(":")[-1]
            assert f"EXPOSE {container_port}" in dockerfile

    def test_compose_backend_dockerfile_reference(self):
        compose = yaml.safe_load(DOCKER_COMPOSE.read_text())
        build = compose["services"]["backend"].get("build", {})
        assert build.get("dockerfile") == "Dockerfile.backend"

    def test_compose_frontend_dockerfile_reference(self):
        compose = yaml.safe_load(DOCKER_COMPOSE.read_text())
        build = compose["services"]["frontend"].get("build", {})
        assert build.get("dockerfile") == "Dockerfile.frontend"

    def test_both_workflows_exist(self):
        assert CI_WORKFLOW.is_file(), "ci.yml should exist"
        assert RELEASE_WORKFLOW.is_file(), "release.yml should exist"

    def test_no_french_in_new_files(self):
        """Verify no French text in new S163 files."""
        french_words = [
            "serveur", "fichier", "securite", "connexion",
            "utilisateur", "deploiement", "disponible",
        ]
        files_to_check = [
            RELEASE_WORKFLOW, DOCKERFILE, DOCKER_COMPOSE,
            DOCKERIGNORE, CHANGELOG_SCRIPT, REQUIREMENTS,
        ]
        for filepath in files_to_check:
            content = filepath.read_text().lower()
            for word in french_words:
                assert word not in content, (
                    f"French word '{word}' found in {filepath.name}"
                )

    def test_no_emojis_in_new_files(self):
        """Verify no emojis in new S163 files."""
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251]+",
            flags=re.UNICODE,
        )
        files_to_check = [
            RELEASE_WORKFLOW, DOCKERFILE, DOCKER_COMPOSE,
            DOCKERIGNORE, CHANGELOG_SCRIPT, REQUIREMENTS,
        ]
        for filepath in files_to_check:
            content = filepath.read_text()
            matches = emoji_pattern.findall(content)
            assert not matches, (
                f"Emoji found in {filepath.name}: {matches}"
            )
