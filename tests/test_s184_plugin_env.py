#!/usr/bin/env python3
"""
S184 / PI-01: a plugin subprocess must not inherit the host environment.

Plugin code is loaded and executed in-process inside the worker
(``plugin_worker.load_plugin_module`` -> ``exec_module``), with isolation
provided only by the process boundary and resource limits -- there is no
sandbox. Before this fix both ``PluginSubprocessManager.start_plugin`` and
``AsyncPluginSubprocessManager.start_plugin`` passed ``os.environ.copy()`` to
the subprocess, so an env-provided ``OPTI_ENCRYPTION_KEY`` (the database master
key), an SQLCipher passphrase, or search API keys reached untrusted plugin
code. This is the plugin-path twin of the bubblewrap env leak (S-01/C-01) that
Track 1 closed only for the sandbox.

The fix forwards a minimal, secret-free allowlist of variables plus the
explicit OO_* / env_extra values. These tests verify the pure ``_build_plugin_env``
builder on both managers without launching a subprocess.
"""

import importlib.util
import os
import sys
import types

# Guarded stub: in CI ollama is installed and this is a no-op; locally it lets
# the isolated module load resolve the (guarded) opti_oignon import chain
# without the heavy dependency.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

_HERE = os.path.dirname(__file__)


def _load(mod_filename, mod_alias):
    path = os.path.join(_HERE, os.pardir, "opti_oignon", mod_filename)
    spec = importlib.util.spec_from_file_location(mod_alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_alias] = mod  # register before exec (3.12 dataclass order)
    spec.loader.exec_module(mod)
    return mod


_sync = _load("plugin_subprocess.py", "plugin_subprocess_s184")
_async = _load("async_plugin_subprocess.py", "async_plugin_subprocess_s184")

# Secrets that must never reach a plugin subprocess.
_SECRET_VARS = {
    "OPTI_ENCRYPTION_KEY": "BASE64_MASTER_KEY_DO_NOT_LEAK",
    "OPTI_SQLCIPHER_PASSPHRASE": "sqlcipher-secret",
    "SEARCH_API_KEY": "tavily-secret",
    "OPENAI_API_KEY": "sk-should-not-leak",
}


def _with_secrets_in_env(fn):
    saved = {k: os.environ.get(k) for k in _SECRET_VARS}
    try:
        os.environ.update(_SECRET_VARS)
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestBuildPluginEnvSync:
    def test_no_secret_leaks(self):
        def check():
            env = _sync._build_plugin_env({"OO_PLUGIN_NAME": "demo"})
            for k in _SECRET_VARS:
                assert k not in env, f"{k} leaked into plugin env"
        _with_secrets_in_env(check)

    def test_explicit_oo_vars_present(self):
        env = _sync._build_plugin_env({
            "OO_PLUGIN_NAME": "demo",
            "OO_HMAC_KEY": "deadbeef",
        })
        assert env["OO_PLUGIN_NAME"] == "demo"
        assert env["OO_HMAC_KEY"] == "deadbeef"

    def test_path_always_present(self):
        saved = os.environ.pop("PATH", None)
        try:
            env = _sync._build_plugin_env({})
            assert env["PATH"]  # backfilled to a safe default
        finally:
            if saved is not None:
                os.environ["PATH"] = saved

    def test_only_allowlisted_host_vars_forwarded(self):
        def check():
            os.environ["A_RANDOM_HOST_VAR"] = "should-not-forward"
            try:
                env = _sync._build_plugin_env({"OO_PLUGIN_NAME": "demo"})
                assert "A_RANDOM_HOST_VAR" not in env
            finally:
                os.environ.pop("A_RANDOM_HOST_VAR", None)
        _with_secrets_in_env(check)


class TestBuildPluginEnvAsync:
    def test_no_secret_leaks(self):
        def check():
            env = _async._build_plugin_env({"OO_PLUGIN_NAME": "demo"})
            for k in _SECRET_VARS:
                assert k not in env, f"{k} leaked into async plugin env"
        _with_secrets_in_env(check)

    def test_explicit_oo_vars_present(self):
        env = _async._build_plugin_env({"OO_PLUGIN_NAME": "demo"})
        assert env["OO_PLUGIN_NAME"] == "demo"

    def test_env_extra_overrides_and_merges(self):
        env = _async._build_plugin_env({"OO_PLUGIN_NAME": "demo"})
        env.update({"EXTRA": "1"})  # caller env_extra path
        assert env["EXTRA"] == "1"
        assert env["OO_PLUGIN_NAME"] == "demo"


class TestStartPluginNoLongerCopiesEnviron:
    """Source-level guard: neither manager should use os.environ.copy()."""

    def test_sync_source_has_no_environ_copy(self):
        src = open(
            os.path.join(_HERE, os.pardir, "opti_oignon", "plugin_subprocess.py"),
            encoding="utf-8",
        ).read()
        assert "os.environ.copy()" not in src

    def test_async_source_has_no_environ_copy(self):
        src = open(
            os.path.join(
                _HERE, os.pardir, "opti_oignon", "async_plugin_subprocess.py"
            ),
            encoding="utf-8",
        ).read()
        assert "os.environ.copy()" not in src
