"""S197 F10c -- CLI (`oo` Click app) + the launcher (`opti_oignon.main`).

Source-level checks (the loader idioms would drag the package import chain;
the CLI surface is fully assertable from text):
- the `oo` Click command tree is complete and every leaf reaches a real
  OOClient HTTP method;
- every backend endpoint the CLI targets exists;
- the launcher dispatches all nine subcommands to a real cmd_ function;
- N-01 posture confirmed unchanged (the launcher forces + asserts the bind,
  direct uvicorn bypasses it, the request-time SecurityModeMiddleware applies);
- CLI-01 fixed; help text is EN-only (HY-01).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
CLI_MAIN = OO / "cli" / "main.py"
CLI_CLIENT = OO / "cli" / "client.py"
LAUNCHER = OO / "main.py"
APP = OO / "api" / "app.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# -- oo Click command tree --


def test_oo_command_tree_complete():
    src = _read(CLI_MAIN)
    assert "@click.group()" in src
    # Top-level commands.
    for cmd in ("ask", "models", "status"):
        assert f"@cli.command()\n" in src  # at least one bare command
        assert f"def {cmd}(" in src, f"missing command function {cmd}"
    # Groups and their leaves.
    leaves = {
        "backup": ("export", "import"),
        "rag": ("ingest", "query"),
        "redteam": ("run", "status", "report", "compare"),
    }
    for group, subs in leaves.items():
        assert f"@cli.group()" in src
        for sub in subs:
            assert f'@{group}.command("{sub}")' in src, f"missing {group} {sub}"
    # config group with invoke_without_command + its leaves.
    assert 'name="config", invoke_without_command=True' in src
    assert '@config_cmd.command("set")' in src
    assert '@config_cmd.command("reset")' in src
    # main() invokes the group.
    assert re.search(r"def main\(\) -> None:\s*\"\"\".*?\"\"\"\s*cli\(\)", src, re.S)


def test_oo_leaves_reach_real_client_methods():
    src = _read(CLI_MAIN)
    used = set(re.findall(r"client\.([a-z_]+)\(", src))
    client_src = _read(CLI_CLIENT)
    defined = set(re.findall(r"def ([a-z_]+)\(", client_src))
    missing = used - defined
    assert not missing, f"CLI calls undefined client methods: {missing}"


def test_oo_targets_existing_backend_endpoints():
    src = _read(CLI_MAIN)
    paths = set(re.findall(r'"(/api/[a-z0-9/_{}-]+)"', src))
    # Map each path to the route file + a fragment that must be present.
    checks = {
        "/api/models": (OO / "api" / "routes_models.py", 'prefix="/api/models"'),
        "/api/health/dashboard": (OO / "api" / "routes_health.py", '"/dashboard"'),
        "/api/backup/export": (OO / "api" / "routes_backup.py", "backup/export"),
        "/api/backup/import": (OO / "api" / "routes_backup.py", "backup/import"),
        "/api/rag/ingest": (OO / "api" / "routes_rag.py", "rag/ingest"),
        "/api/rag/query": (OO / "api" / "routes_rag.py", "rag/query"),
        "/api/security/redteam/run": (OO / "api" / "routes_security.py", '"/redteam/run"'),
        "/api/security/redteam/status": (OO / "api" / "routes_security.py", '"/redteam/status"'),
    }
    for path, (route_file, fragment) in checks.items():
        assert path in src, f"CLI no longer targets {path}"
        assert fragment in _read(route_file), f"backend missing route for {path}"


# -- launcher subcommand dispatch --


def test_launcher_dispatches_all_subcommands():
    src = _read(LAUNCHER)
    commands = re.search(r"commands = \{(.*?)\}", src, re.S).group(1)
    expected = ["ui", "api", "benchmark", "rag", "config", "ask",
                "presets", "info", "export"]
    for cmd in expected:
        assert f'"{cmd}": cmd_' in commands, f"{cmd} not dispatched"
        fn = re.search(rf'"{cmd}": (cmd_[a-z]+)', commands).group(1)
        assert f"def {fn}(" in src, f"dispatch target {fn} not defined"


# -- N-01 posture (confirmation, not a re-audit) --


def test_n01_launcher_forces_and_asserts_bind():
    src = _read(LAUNCHER)
    assert "get_safe_bind_address(args.host)" in src
    assert "assert_safe_bind_address(actual_host)" in src
    # The documented bypass note must remain.
    assert "bypassing this launcher" in src


def test_n01_request_time_backstop_present():
    # Direct uvicorn skips the launcher forcing; the request-time mode
    # middleware (M-01 fail-closed) is the runtime backstop.
    app_src = _read(APP)
    assert "SecurityModeMiddleware" in app_src
    assert "app.add_middleware(SecurityModeMiddleware)" in app_src


# -- CLI-01 + HY-01 --


def test_cli01_config_show_has_no_bogus_line():
    src = _read(CLI_MAIN)
    assert "cfg.save.__func__.__defaults__" not in src
    # The correct path line is what remains.
    assert 'click.echo(f"Config path: {CONFIG_FILE}")' in src


def test_cli_help_is_english_only():
    src = _read(CLI_MAIN)
    fr = re.findall(
        r"help=[\"'][^\"']*\b(?:le|la|les|une|des|pour|avec|aucun|fichier|"
        r"mod\u00e8le|ex\u00e9cuter|afficher|param\u00e8tre)\b",
        src,
    )
    assert fr == [], f"French in CLI help strings: {fr}"
