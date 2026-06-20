#!/usr/bin/env python3
"""Tests for S229 -- AGT Lot 2: loop hardening (AGT_SPEC Section 6).

Container-provable coverage (AGT_SPEC Section 11, Lot 2):

- 6.1 caps and spill: the per-observation byte/line caps with their hard
  floors, head-by-default and head-plus-tail-for-bash truncation around the
  elision marker, the per-round budget whose over-budget observations
  truncate harder and are never dropped, the spill of the full text through
  the session's ordinary create_file under .agent/spill/ with the run-global
  counter, the stub naming the path (and omitting it on handler-only runs or
  failed writes), DispatchResult immutability (events and tool_results keep
  the full observation), and byte-identity with the legacy
  _observations_message on small outputs.
- The .agent/ manifest rule: the copy-out diff walk excludes the
  workspace-root .agent/ subtree (cross-cycle with S211/S212); non-root
  .agent directories still walk (the rule is the declared root prefix only).
- 6.2 prune: chars-based estimate, oldest-first stubbing outside the protect
  window, the verbatim stub format with spill path-or-none, protection by
  construction of the system prompt / original task / last protect_rounds
  rounds, the compaction event, and the flag-gated summarize stage's
  STRUCTURE with a fake client (off by default: no extra model call; on: one
  untrusted "[compaction summary]" message at the prune boundary).
- 6.3 doom loop: the rolling (tool, canonical args) window over EXECUTED
  calls, the one corrective untrusted observation at the threshold without
  approval_fn, the pre-dispatch abort on a further identical call
  (stop_reason "doom_loop", AgentEvent kind "aborted"), the window reset on
  a different call, refusals neither counting nor resetting, the synthetic
  approval asked once per signature with approval_fn (approve continues and
  exempts; deny or an exception aborts fail-secure), and the ARBITRATED
  child exemption (the bounded, debited task child keeps its S228 pins).
- 6.4 recovery: the three conservative replacers (line-trimmed,
  whitespace-normalized, indentation-flexible with uniform-delta
  re-indentation) on the exact-miss path only, the one-candidate rule, the
  K-regions whole-call failure, the strategy-naming success message, the
  difflib miss hint appended to the UNCHANGED not-found message, the
  untouched multi-exact and exact-success paths, and the composition with
  the S228 diagnostics-after-write at the session layer.
- 6.5 reminder: the verbatim trusted message, once per run, when (counting
  the current round) two rounds remain, including after a task debit.
- 6.6 governor branch: _derive_budgets static / fed / floor paths, the
  thread-local ticket read with honest provenance (admit/downsize only),
  the run() admitted_num_ctx keyword, and the behavioural pin that a fed
  budget suppresses a prune the static budget triggers.

Host-assured, NAMED here and never simulated in the container (AGT_SPEC
Section 11, Lot 2): the summarize-stage live behaviour -- summary quality
and token economics -- stays host-assured with the flag OFF at landing (the
Lot 3 harness measures it before anyone turns it on); the real token
economics of the caps on live models; the real in-bwrap spill round-trip
(the container exercises the spill through injected sessions and the
tempdir backend only).

Supersessions: ZERO. The Section 11 conditional family (suites pinning the
exact str_replace match-failure message) resolved EMPTY at the S229 read
gate: the miss hint is append-only on the unchanged not-found message, the
multi-exact path is untouched, and the strategy-naming successes are new
strings on a previously nonexistent path; the live pins (substring
"not found", "2 times", "successful", startswith "Error") all hold and are
re-asserted below.

Loaded in isolation via ``spec_from_file_location`` with the package stubs
and the ollama setdefault, so the runtime collects without the backend.
"""

import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"


def _ensure_pkg(name: str, path: Path) -> None:
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg


_ensure_pkg("opti_oignon", OO)
_ensure_pkg("opti_oignon.agent", AGENT)


def _load(register: str, path: Path):
    if register in sys.modules:
        return sys.modules[register]
    spec = importlib.util.spec_from_file_location(register, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[register] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load("opti_oignon.sandbox_manager", OO / "sandbox_manager.py")
tr = _load("opti_oignon.tool_registry", OO / "tool_registry.py")
ft = _load("opti_oignon.file_tools", OO / "file_tools.py")
st = _load("opti_oignon.sandbox_tools", OO / "sandbox_tools.py")
ws = _load("opti_oignon.sandbox_workspace", OO / "sandbox_workspace.py")
_load("opti_oignon.agent.tool_parsing", AGENT / "tool_parsing.py")
al = _load("opti_oignon.agent.allowlists", AGENT / "allowlists.py")
d = _load("opti_oignon.agent.dispatch", AGENT / "dispatch.py")
t = _load("opti_oignon.agent.tools", AGENT / "tools.py")
uc = _load("opti_oignon.agent.untrusted_context", AGENT / "untrusted_context.py")
L = _load("opti_oignon.agent.loop", AGENT / "loop.py")
cfg = _load("opti_oignon.agent.config_loader", AGENT / "config_loader.py")


@pytest.fixture(autouse=True)
def _reset_registry():
    t.reset_tool_registry()
    yield
    t.reset_tool_registry()


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class ScriptedClient:
    """A fake model client replaying scripted Ollama-shaped rounds."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def stream(self, messages, tools=None):
        self.calls += 1
        step = self.script.pop(0) if self.script else {"content": "done", "tool_calls": None}
        return {"message": step}


def _native(name, args):
    return [{"function": {"name": name, "arguments": args}}]


def _native_many(*pairs):
    return [{"function": {"name": n, "arguments": a}} for n, a in pairs]


class LoopSession:
    """A small-output recording sandbox session (bwrap-shaped by default)."""

    def __init__(self, bwrap: bool = True, active: bool = True):
        self.sandbox_manager = types.SimpleNamespace(bwrap_available=bwrap)
        self.active = active
        self.calls: list[tuple] = []

    def bash(self, command, timeout=30):
        self.calls.append(("bash", command))
        return f"out: {command}"

    def view(self, path, start_line=0, end_line=0):
        self.calls.append(("view", path))
        return f"view {path}"

    def create_file(self, path, content):
        self.calls.append(("create_file", path))
        return f"File created: {path} (0 bytes)"

    def str_replace(self, path, old_str, new_str=""):
        self.calls.append(("str_replace", path))
        return "ok"


class SpillSession(LoopSession):
    """A session whose reads are oversized and whose create_file is real-shaped."""

    def __init__(self, big_bytes: int = 120_000, bwrap: bool = True):
        super().__init__(bwrap=bwrap)
        self.big_bytes = big_bytes
        self.files: dict[str, str] = {}
        self.create_result: str | None = None
        self.create_raises = False

    def view(self, path, start_line=0, end_line=0):
        self.calls.append(("view", path))
        return "V" * self.big_bytes

    def bash(self, command, timeout=30):
        self.calls.append(("bash", command))
        return "HEAD_SENTINEL\n" + ("middle\n" * 4000) + "TAIL_SENTINEL"

    def create_file(self, path, content):
        self.calls.append(("create_file", path))
        if self.create_raises:
            raise RuntimeError("disk full")
        if self.create_result is not None:
            return self.create_result
        self.files[path] = content
        return f"File created: {path} ({len(content.encode('utf-8'))} bytes)"


def _joined(res) -> str:
    return "\n@@@\n".join(m.get("content", "") or "" for m in res.messages)


def _events_collector():
    events = []
    return events, events.append


# ---------------------------------------------------------------------------
# Constants and floors
# ---------------------------------------------------------------------------


class TestHardeningConstants:
    def test_stop_doom_loop_constant(self):
        assert L.STOP_DOOM_LOOP == "doom_loop"

    def test_per_observation_caps_are_spec_defaults(self):
        assert L.AGENT_OBS_MAX_BYTES == 16384
        assert L.AGENT_OBS_MAX_LINES == 256

    def test_round_budget_default(self):
        assert L.AGENT_ROUND_OBS_BUDGET == 49152

    def test_prune_defaults(self):
        assert L.PRUNE_TRIGGER_CHARS == 98304
        assert L.PRUNE_TARGET_CHARS == 65536
        assert L.PRUNE_PROTECT_ROUNDS == 3

    def test_doom_threshold_default(self):
        assert L.DOOM_LOOP_THRESHOLD == 3

    def test_hard_floors(self):
        assert L._OBS_BYTES_FLOOR == 4096
        assert L._OBS_LINES_FLOOR == 64

    def test_reminder_text_verbatim(self):
        assert L.MAX_STEPS_REMINDER == (
            "[2 rounds remain; converge: finish the task or state what is "
            "done and what remains]"
        )

    def test_spill_dir_constant(self):
        assert L._SPILL_DIR == ".agent/spill"

    def test_s222_loop_source_pins_still_literal(self):
        src = (AGENT / "loop.py").read_text(encoding="utf-8")
        for needle in (
            "MAX_AGENT_ROUNDS = 20",
            "_VERIFIER_MAX_ROUNDS = 2",
            "def _run_verifier",
            "class AgentEvent",
            "tool_output_message",
        ):
            assert needle in src, needle

    def test_round_caps_unchanged(self):
        assert L.MAX_AGENT_ROUNDS == 20
        assert L._VERIFIER_MAX_ROUNDS == 2


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _patched_yaml(monkeypatch, payload):
    monkeypatch.setattr(
        L, "_yaml", types.SimpleNamespace(safe_load=lambda _text: payload)
    )


class TestHardeningConfig:
    def test_defaults_dict_shape(self):
        assert set(L.HARDENING_DEFAULTS) == {
            "obs_max_bytes",
            "obs_max_lines",
            "round_obs_budget",
            "prune_trigger_chars",
            "prune_target_chars",
            "prune_protect_rounds",
            "summarize_compaction",
            "doom_loop_threshold",
            "obs_fraction",
        }

    def test_yaml_block_matches_defaults_today(self):
        assert L.load_hardening_config() == L.HARDENING_DEFAULTS

    def test_obs_bytes_floor_enforced(self, monkeypatch):
        _patched_yaml(monkeypatch, {"hardening": {"obs_max_bytes": 100}})
        assert L.load_hardening_config()["obs_max_bytes"] == 4096

    def test_obs_lines_floor_enforced(self, monkeypatch):
        _patched_yaml(monkeypatch, {"hardening": {"obs_max_lines": 10}})
        assert L.load_hardening_config()["obs_max_lines"] == 64

    def test_doom_threshold_floor_is_two(self, monkeypatch):
        _patched_yaml(monkeypatch, {"hardening": {"doom_loop_threshold": 1}})
        assert L.load_hardening_config()["doom_loop_threshold"] == 2

    def test_trigger_kept_above_target(self, monkeypatch):
        _patched_yaml(
            monkeypatch,
            {"hardening": {"prune_trigger_chars": 5000, "prune_target_chars": 60000}},
        )
        loaded = L.load_hardening_config()
        assert loaded["prune_target_chars"] == 60000
        assert loaded["prune_trigger_chars"] == 60000 + 4096

    def test_missing_block_yields_defaults(self, monkeypatch):
        _patched_yaml(monkeypatch, {"loop": {"round_cap": 20}})
        assert L.load_hardening_config() == L.HARDENING_DEFAULTS

    def test_missing_yaml_module_yields_defaults(self, monkeypatch):
        monkeypatch.setattr(L, "_yaml", None)
        assert L.load_hardening_config() == L.HARDENING_DEFAULTS

    def test_obs_fraction_clamped(self, monkeypatch):
        _patched_yaml(monkeypatch, {"hardening": {"obs_fraction": 5.0}})
        assert L.load_hardening_config()["obs_fraction"] == 1.0
        _patched_yaml(monkeypatch, {"hardening": {"obs_fraction": 0.0001}})
        assert L.load_hardening_config()["obs_fraction"] == 0.05

    def test_summarize_flag_off_by_default(self):
        assert L.HARDENING_DEFAULTS["summarize_compaction"] is False


# ---------------------------------------------------------------------------
# Truncation unit behaviour (6.1)
# ---------------------------------------------------------------------------


class TestTruncationUnit:
    def test_under_caps_unchanged(self):
        clipped, truncated = L._truncate_observation("abc\ndef", 1000, 100, False)
        assert clipped == "abc\ndef" and truncated is False

    def test_byte_cap_keeps_head(self):
        text = "HEAD" + "x" * 10_000
        clipped, truncated = L._truncate_observation(text, 64, 1000, False)
        assert truncated is True
        assert clipped.startswith("HEAD")
        assert len(clipped.encode("utf-8")) <= 64

    def test_line_cap_keeps_head_lines(self):
        text = "\n".join(f"L{i}" for i in range(300))
        clipped, truncated = L._truncate_observation(text, 1_000_000, 10, False)
        assert truncated is True
        kept = clipped.split("\n")
        assert kept == [f"L{i}" for i in range(10)]

    def test_utf8_boundary_safe(self):
        text = "\u00e9" * 5000  # two bytes each
        clipped, truncated = L._truncate_observation(text, 101, 9999, False)
        assert truncated is True
        raw = clipped.encode("utf-8")
        assert len(raw) <= 101
        assert clipped == "\u00e9" * (len(raw) // 2)

    def test_bash_head_and_tail_around_marker(self):
        text = "FIRST_LINE\n" + ("mid\n" * 500) + "LAST_LINE"
        clipped, truncated = L._truncate_observation(text, 200, 64, True)
        assert truncated is True
        assert L._BASH_ELISION_MARKER in clipped
        assert clipped.startswith("FIRST_LINE")
        assert clipped.endswith("LAST_LINE")

    def test_clip_tail_helper(self):
        assert L._utf8_clip_tail("abcdef", 3) == "def"
        assert L._utf8_clip("abcdef", 3) == "abc"


# ---------------------------------------------------------------------------
# Caps inside the loop (6.1)
# ---------------------------------------------------------------------------


def _dr(name, text, executed=True):
    return d.DispatchResult(
        tool_name=name,
        executed=executed,
        observation=text,
        reason="executed" if executed else "refused",
        source="native",
        mode="daily",
    )


class TestCapsInLoop:
    def test_small_outputs_byte_identical_to_legacy_builder(self):
        results = [_dr("view", "small output"), _dr("bash", "ok\nfine")]
        msg_new, spills = L._capped_observations_message(
            results,
            cfg=L.HARDENING_DEFAULTS,
            round_budget=L.AGENT_ROUND_OBS_BUDGET,
            rnd=1,
            sandbox=None,
            spill_counter={"k": 0},
        )
        msg_old = L._observations_message(results)
        assert msg_new == msg_old
        assert spills == []

    def test_oversized_view_clipped_in_transcript(self):
        sess = SpillSession()
        script = [
            {"content": "x", "tool_calls": _native("view", {"path": "/big"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess,
                    mode="daily", include_memory=False)
        joined = _joined(res)
        assert "[truncated:" in joined
        obs_msgs = [m for m in res.messages if "[truncated:" in (m.get("content") or "")]
        assert obs_msgs, "the clipped observation message must exist"
        assert len(obs_msgs[0]["content"].encode("utf-8")) < 40_000

    def test_dispatch_results_not_mutated(self):
        sess = SpillSession(big_bytes=100_000)
        script = [
            {"content": "x", "tool_calls": _native("view", {"path": "/big"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess,
                    mode="daily", include_memory=False)
        assert len(res.tool_results[0].observation) == 100_000

    def test_tool_result_event_carries_full_observation(self):
        sess = SpillSession(big_bytes=90_000)
        events, on_event = _events_collector()
        script = [
            {"content": "x", "tool_calls": _native("view", {"path": "/big"})},
            {"content": "done", "tool_calls": None},
        ]
        L.run("q", model_client=ScriptedClient(script), sandbox=sess,
              mode="daily", include_memory=False, on_event=on_event)
        tool_events = [e for e in events if e.kind == "tool_result"]
        assert len(tool_events[0].data["observation"]) == 90_000

    def test_stub_names_original_bytes_and_lines(self):
        sess = SpillSession()
        script = [
            {"content": "x", "tool_calls": _native("view", {"path": "/big"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess,
                    mode="daily", include_memory=False)
        assert re.search(r"\[truncated: \d+ bytes, \d+ lines; full output: ", _joined(res))

    def test_handler_only_run_stub_omits_path(self):
        handlers = {"web_search": lambda args: "W" * 60_000}
        script = [
            {"content": "x", "tool_calls": _native("web_search", {"query": "q"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=None,
                    mode="daily", include_memory=False, tool_handlers=handlers)
        joined = _joined(res)
        assert "[truncated:" in joined
        assert "full output:" not in joined


# ---------------------------------------------------------------------------
# Round budget (6.1)
# ---------------------------------------------------------------------------


class TestRoundBudget:
    def _build(self, round_budget):
        results = [
            _dr("t1", "~" * 6000),
            _dr("t2", "^" * 6000),
            _dr("t3", "@" * 6000),
        ]
        msg, _spills = L._capped_observations_message(
            results,
            cfg=L.HARDENING_DEFAULTS,
            round_budget=round_budget,
            rnd=1,
            sandbox=None,
            spill_counter={"k": 0},
        )
        return msg["content"]

    def test_overbudget_later_observations_truncate_harder(self):
        content = self._build(8192)
        # First fits whole (6000 <= 8192); later ones clip to the remainder
        # then the floor; nothing is dropped.
        assert content.count("~") == 6000
        assert content.count("[truncated:") == 2
        assert 0 < content.count("@") <= L._OBS_OVERBUDGET_BYTES

    def test_overbudget_floor_is_never_zero(self):
        content = self._build(1)
        for ch in ("~", "^", "@"):
            assert 0 < content.count(ch) <= L._OBS_OVERBUDGET_BYTES

    def test_no_observation_dropped(self):
        content = self._build(1)
        for name in ("t1", "t2", "t3"):
            assert name in content


# ---------------------------------------------------------------------------
# Spill (6.1)
# ---------------------------------------------------------------------------


class TestSpill:
    def _run_one_big_view(self, sess):
        script = [
            {"content": "x", "tool_calls": _native("view", {"path": "/big"})},
            {"content": "done", "tool_calls": None},
        ]
        return L.run("q", model_client=ScriptedClient(script), sandbox=sess,
                     mode="daily", include_memory=False)

    def test_spill_path_in_stub_and_full_content_captured(self):
        sess = SpillSession(big_bytes=100_000)
        res = self._run_one_big_view(sess)
        assert ".agent/spill/obs_1_1.txt" in _joined(res)
        assert sess.files[".agent/spill/obs_1_1.txt"] == "V" * 100_000

    def test_spill_counter_is_run_global_across_rounds(self):
        sess = SpillSession(big_bytes=80_000)
        script = [
            {"content": "x", "tool_calls": _native("view", {"path": "/one"})},
            {"content": "y", "tool_calls": _native("view", {"path": "/two"})},
            {"content": "done", "tool_calls": None},
        ]
        L.run("q", model_client=ScriptedClient(script), sandbox=sess,
              mode="daily", include_memory=False)
        assert ".agent/spill/obs_1_1.txt" in sess.files
        assert ".agent/spill/obs_2_2.txt" in sess.files

    def test_two_spills_in_one_round(self):
        sess = SpillSession(big_bytes=80_000)
        script = [
            {
                "content": "x",
                "tool_calls": _native_many(
                    ("view", {"path": "/one"}), ("view", {"path": "/two"})
                ),
            },
            {"content": "done", "tool_calls": None},
        ]
        L.run("q", model_client=ScriptedClient(script), sandbox=sess,
              mode="daily", include_memory=False)
        assert ".agent/spill/obs_1_1.txt" in sess.files
        assert ".agent/spill/obs_1_2.txt" in sess.files

    def test_error_result_from_create_file_omits_path(self):
        sess = SpillSession()
        sess.create_result = "Error: refused"
        res = self._run_one_big_view(sess)
        joined = _joined(res)
        assert "[truncated:" in joined
        assert "full output:" not in joined

    def test_create_file_raising_omits_path_and_run_continues(self):
        sess = SpillSession()
        sess.create_raises = True
        res = self._run_one_big_view(sess)
        joined = _joined(res)
        assert "[truncated:" in joined
        assert "full output:" not in joined
        assert res.stop_reason == L.STOP_DONE

    def test_txt_suffix_outside_diagnostics_map_source_pin(self):
        src = (OO / "sandbox_tools.py").read_text(encoding="utf-8")
        assert '(".py", ".svelte")' in src  # spill .txt writes never lint


# ---------------------------------------------------------------------------
# The .agent/ manifest rule in the copy-out diff walk (6.1 cross-cycle)
# ---------------------------------------------------------------------------


def _make_manager(tmp_path):
    return sm.SandboxManager(config=sm.SandboxConfig(
        workspace_base=str(tmp_path / "sbx"),
        audit_db_path=str(tmp_path / "audit.db"),
        isolation_backend="tempdir",
        require_degraded_confirmation=False,
        strict_mode=False,
        idle_ttl_seconds=0,
    ))


@pytest.fixture()
def diff_env(tmp_path):
    ws.reset_workspace_manifests()
    manager = _make_manager(tmp_path)
    sid = "s229-diff"
    manager.create_sandbox(sid)
    yield manager, sid, manager.get_active_workspace_path(sid)
    try:
        manager.destroy_sandbox(sid)
    except Exception:
        pass
    ws.reset_workspace_manifests()


def _write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


class TestDiffWalkAgentExclusion:
    def test_root_agent_subtree_never_classifies(self, diff_env):
        manager, sid, root = diff_env
        _write(root, "real.txt", "payload")
        _write(root, ".agent/spill/obs_1_1.txt", "V" * 100)
        diff = ws.generate_workspace_diff(sid, manager=manager)
        paths = [e.path for e in diff.entries]
        assert paths == ["real.txt"]
        assert all(".agent" not in p for p in paths)

    def test_agent_files_do_not_disturb_baseline_diff(self, diff_env):
        manager, sid, root = diff_env
        _write(root, "kept.txt", "stable")
        ws.get_workspace_manifests().record(
            sid, {"kept.txt": ws.manifest_hash_file(os.path.join(root, "kept.txt"))}
        )
        _write(root, ".agent/spill/obs_3_7.txt", "spilled")
        diff = ws.generate_workspace_diff(sid, manager=manager)
        assert diff.entries == []
        assert diff.unchanged == 1

    def test_nested_agent_subdirs_excluded_with_root(self, diff_env):
        manager, sid, root = diff_env
        _write(root, ".agent/spill/deep/nested.txt", "x")
        diff = ws.generate_workspace_diff(sid, manager=manager)
        assert diff.entries == []

    def test_non_root_agent_directory_still_walks(self, diff_env):
        manager, sid, root = diff_env
        _write(root, "src/.agent/z.txt", "tracked")
        diff = ws.generate_workspace_diff(sid, manager=manager)
        assert [e.path for e in diff.entries] == ["src/.agent/z.txt"]


# ---------------------------------------------------------------------------
# Prune (6.2)
# ---------------------------------------------------------------------------


PRUNE_CFG = dict(L.HARDENING_DEFAULTS)
PRUNE_CFG.update({"prune_trigger_chars": 2600, "prune_target_chars": 1200})


def _six_view_rounds():
    script = []
    for i in range(1, 7):
        script.append({
            "content": f"c{i}",
            "tool_calls": _native("view", {"path": f"/p{i}"}),
        })
    script.append({"content": "done", "tool_calls": None})
    return script


class _PruneSession(LoopSession):
    def view(self, path, start_line=0, end_line=0):
        self.calls.append(("view", path))
        idx = path.strip("/p")
        return f"PRUNEOBS{idx} " + "z" * 600


class TestPrune:
    def _run(self, monkeypatch, extra=None, sess=None, on_event=None):
        merged = dict(PRUNE_CFG)
        if extra:
            merged.update(extra)
        monkeypatch.setattr(L, "load_hardening_config", lambda: merged)
        return L.run(
            "TASK_SENTINEL",
            model_client=ScriptedClient(_six_view_rounds()),
            sandbox=sess or _PruneSession(),
            mode="daily",
            system_prompt="SYSTEM_SENTINEL",
            include_memory=False,
            on_event=on_event,
        )

    def test_oldest_rounds_stubbed_protected_rounds_verbatim(self, monkeypatch):
        res = self._run(monkeypatch)
        joined = _joined(res)
        for i in (1, 2, 3):
            assert f"PRUNEOBS{i}" not in joined
        for i in (4, 5, 6):
            assert f"PRUNEOBS{i}" in joined
        assert "[pruned observation, round 1, " in joined

    def test_stub_is_the_entire_message_and_verbatim(self, monkeypatch):
        res = self._run(monkeypatch)
        stubs = [
            m["content"] for m in res.messages
            if (m.get("content") or "").startswith("[pruned observation")
        ]
        assert stubs
        for stub in stubs:
            assert re.fullmatch(
                r"\[pruned observation, round \d+, \d+ bytes; spill: none\]", stub
            ), stub

    def test_system_and_task_protected_by_construction(self, monkeypatch):
        res = self._run(monkeypatch)
        assert res.messages[0] == {"role": "system", "content": "SYSTEM_SENTINEL"}
        assert {"role": "user", "content": "TASK_SENTINEL"} in res.messages

    def test_compaction_event_emitted(self, monkeypatch):
        events, on_event = _events_collector()
        self._run(monkeypatch, on_event=on_event)
        compactions = [e for e in events if e.kind == "compaction"]
        assert compactions
        data = compactions[0].data
        assert data["pruned"] >= 1
        assert data["before"] > data["after"]

    def test_pruned_spilled_stub_names_the_path(self, monkeypatch):
        sess = SpillSession(big_bytes=60_000)
        merged = dict(L.HARDENING_DEFAULTS)
        merged.update({"prune_trigger_chars": 30_000, "prune_target_chars": 8_000})
        monkeypatch.setattr(L, "load_hardening_config", lambda: merged)
        script = []
        for i in range(1, 6):
            script.append({
                "content": f"c{i}",
                "tool_calls": _native("view", {"path": f"/p{i}"}),
            })
        script.append({"content": "done", "tool_calls": None})
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess,
                    mode="daily", include_memory=False)
        assert re.search(
            r"\[pruned observation, round 1, \d+ bytes; "
            r"spill: \.agent/spill/obs_1_1\.txt\]",
            _joined(res),
        )

    def test_no_prune_under_default_thresholds(self):
        res = L.run(
            "q",
            model_client=ScriptedClient(_six_view_rounds()),
            sandbox=_PruneSession(),
            mode="daily",
            include_memory=False,
        )
        assert "[pruned observation" not in _joined(res)


# ---------------------------------------------------------------------------
# Summarize stage structure (6.2, flag-gated; live behaviour host-assured)
# ---------------------------------------------------------------------------


SUMM_CFG = dict(L.HARDENING_DEFAULTS)
SUMM_CFG.update({"prune_trigger_chars": 3000, "prune_target_chars": 800})


def _four_view_rounds(tail):
    script = []
    for i in range(1, 5):
        script.append({
            "content": f"c{i}",
            "tool_calls": _native("view", {"path": f"/p{i}"}),
        })
    script.extend(tail)
    return script


class TestSummarizeStage:
    def test_flag_off_means_no_extra_model_call(self, monkeypatch):
        merged = dict(SUMM_CFG)
        monkeypatch.setattr(L, "load_hardening_config", lambda: merged)
        client = ScriptedClient(
            _four_view_rounds([{"content": "done", "tool_calls": None}])
        )
        res = L.run("q", model_client=client, sandbox=_PruneSession(),
                    mode="daily", include_memory=False)
        assert client.calls == 5
        assert "[compaction summary]" not in _joined(res)

    def test_flag_on_inserts_untrusted_summary_once(self, monkeypatch):
        merged = dict(SUMM_CFG)
        merged["summarize_compaction"] = True
        monkeypatch.setattr(L, "load_hardening_config", lambda: merged)
        client = ScriptedClient(_four_view_rounds([
            {"content": "SUMMARY_TEXT_SENTINEL", "tool_calls": None},
            {"content": "done", "tool_calls": None},
        ]))
        res = L.run("q", model_client=client, sandbox=_PruneSession(),
                    mode="daily", include_memory=False)
        assert client.calls == 6
        summaries = [
            m for m in res.messages
            if "[compaction summary] SUMMARY_TEXT_SENTINEL" in (m.get("content") or "")
        ]
        assert len(summaries) == 1
        assert summaries[0]["role"] == "user"
        assert uc.UNTRUSTED_POLICY in summaries[0]["content"]


# ---------------------------------------------------------------------------
# Doom-loop window (6.3)
# ---------------------------------------------------------------------------


def _identical(n, tool="view", args=None, contents=None):
    args = args if args is not None else {"path": "/x"}
    script = []
    for i in range(1, n + 1):
        content = contents[i - 1] if contents else f"c{i}"
        script.append({"content": content, "tool_calls": _native(tool, dict(args))})
    return script


CORRECTIVE = (
    "[doom-loop detected: view repeated 3 times with identical arguments; "
    "vary the approach or conclude]"
)


class TestDoomLoop:
    def test_corrective_at_three_then_run_continues(self):
        script = _identical(3) + [
            {"content": "vary", "tool_calls": _native("view", {"path": "/other"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=LoopSession(),
                    mode="daily", include_memory=False)
        joined = _joined(res)
        assert joined.count(CORRECTIVE) == 1
        assert res.stop_reason == L.STOP_DONE

    def test_fourth_identical_aborts_before_execution(self):
        sess = LoopSession()
        events, on_event = _events_collector()
        script = _identical(4, contents=["c1", "c2", "c3", "c4"]) + [
            {"content": "never", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess,
                    mode="daily", include_memory=False, on_event=on_event)
        assert res.stop_reason == L.STOP_DOOM_LOOP
        assert res.rounds == 4
        assert sess.calls.count(("view", "/x")) == 3
        aborted = [e for e in events if e.kind == "aborted"]
        assert aborted and aborted[0].data == {"reason": "doom_loop", "tool": "view"}
        assert res.final_text == "c4"

    def test_window_resets_on_different_call(self):
        script = (
            _identical(2)
            + [{"content": "b", "tool_calls": _native("view", {"path": "/other"})}]
            + _identical(3)
            + [{"content": "done", "tool_calls": None}]
        )
        res = L.run("q", model_client=ScriptedClient(script), sandbox=LoopSession(),
                    mode="daily", include_memory=False)
        assert _joined(res).count(CORRECTIVE) == 1
        assert res.stop_reason == L.STOP_DONE

    def test_different_arguments_are_different_signatures(self):
        script = [
            {"content": f"c{i}", "tool_calls": _native("view", {"path": f"/v{i}"})}
            for i in range(5)
        ] + [{"content": "done", "tool_calls": None}]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=LoopSession(),
                    mode="daily", include_memory=False)
        assert "[doom-loop detected" not in _joined(res)
        assert res.stop_reason == L.STOP_DONE

    def test_refusals_neither_count_nor_reset(self):
        sess = LoopSession(bwrap=False)
        script = _identical(5, tool="bash", args={"command": "x"})
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess,
                    mode="daily", include_memory=False, max_rounds=5)
        assert "[doom-loop detected" not in _joined(res)
        assert res.stop_reason == L.STOP_MAX_ROUNDS
        assert sess.calls == []

    def test_approval_asked_once_with_spec_message_and_approve_continues(self):
        asks = []

        def approve(cid, label, details):
            asks.append((cid, label, dict(details)))
            return True

        script = _identical(6) + [{"content": "done", "tool_calls": None}]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=LoopSession(),
                    mode="daily", include_memory=False, approval_fn=approve,
                    conversation_id="conv-9")
        assert len(asks) == 1
        cid, label, details = asks[0]
        assert cid == "conv-9" and label == "doom_loop"
        assert details["tool"] == "view"
        assert details["message"] == (
            "doom_loop: the agent repeated view with identical arguments 3 "
            "times; continue?"
        )
        assert res.stop_reason == L.STOP_DONE
        assert res.rounds == 7
        assert "[doom-loop detected" not in _joined(res)

    def test_approval_deny_aborts_at_the_trip(self):
        sess = LoopSession()
        events, on_event = _events_collector()
        script = _identical(4)
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess,
                    mode="daily", include_memory=False,
                    approval_fn=lambda *a: False, on_event=on_event)
        assert res.stop_reason == L.STOP_DOOM_LOOP
        assert res.rounds == 3
        assert sess.calls.count(("view", "/x")) == 3
        assert any(e.kind == "aborted" for e in events)

    def test_approval_exception_is_fail_secure_deny(self):
        def boom(*_a):
            raise RuntimeError("approval channel down")

        res = L.run("q", model_client=ScriptedClient(_identical(4)),
                    sandbox=LoopSession(), mode="daily", include_memory=False,
                    approval_fn=boom)
        assert res.stop_reason == L.STOP_DOOM_LOOP

    def test_child_task_loop_is_exempt_by_arbitration(self):
        # The S229 read-gate decision: the bounded, debited child keeps no
        # doom window; the S228 cap pin must hold verbatim.
        script = [{"content": "p", "tool_calls": _native(
            "task", {"description": "d", "prompt": "go", "max_rounds": 50})}]
        script += [
            {"content": f"c{i}", "tool_calls": _native("bash", {"command": "x"})}
            for i in range(1, 7)
        ]
        script += [{"content": "parent done", "tool_calls": None}]
        res = L.run("q", model_client=ScriptedClient(script),
                    sandbox=LoopSession(), mode="daily", include_memory=False,
                    max_rounds=20)
        task_result = [r for r in res.tool_results if r.tool_name == "task"][0]
        assert task_result.observation.endswith("task used 6 rounds of 6")

    def test_signature_uses_canonical_json(self):
        call_a = d.ToolCall(name="view", arguments={"b": 1, "a": 2}, source="native")
        call_b = d.ToolCall(name="view", arguments={"a": 2, "b": 1}, source="native")
        assert L._doom_signature(call_a) == L._doom_signature(call_b)
        assert json.dumps({"a": 2, "b": 1}, sort_keys=True) in L._doom_signature(call_a)


# ---------------------------------------------------------------------------
# Recovery chain (6.4) -- handler level on a real tempdir workspace
# ---------------------------------------------------------------------------


@pytest.fixture()
def recovery_env(tmp_path):
    config = sm.SandboxConfig(
        enabled=True,
        isolation_backend="tempdir",
        require_degraded_confirmation=False,
        workspace_base=str(tmp_path / "sandboxes"),
        audit_db_path=str(tmp_path / "audit.db"),
    )
    manager = sm.SandboxManager(config)
    sess = manager.create_sandbox("s229-rec")
    yield manager, sess.session_id, sess.workspace_path
    try:
        manager.destroy_sandbox("s229-rec")
    except Exception:
        pass


def _put(workspace, name, text):
    with open(os.path.join(workspace, name), "w", encoding="utf-8") as fh:
        fh.write(text)


def _get(workspace, name):
    with open(os.path.join(workspace, name), encoding="utf-8") as fh:
        return fh.read()


class TestRecoveryChain:
    def test_line_trimmed_trailing_whitespace(self, recovery_env):
        manager, sid, workspace = recovery_env
        _put(workspace, "a.py", "def f():   \n    return 1\n")
        out = ft._handle_sandbox_str_replace(
            sid, "a.py", "def f():\n    return 1", "def f():\n    return 2",
            _sandbox_manager=manager)
        assert out == "Replaced (matched via line-trimmed normalization)."
        assert "return 2" in _get(workspace, "a.py")

    def test_whitespace_normalized_internal_runs(self, recovery_env):
        manager, sid, workspace = recovery_env
        _put(workspace, "b.py", "x  =  1\n")
        out = ft._handle_sandbox_str_replace(
            sid, "b.py", "x = 1", "x = 9", _sandbox_manager=manager)
        assert out == "Replaced (matched via whitespace-normalized normalization)."
        assert _get(workspace, "b.py") == "x = 9\n"

    def test_indentation_flexible_reindents_new_str(self, recovery_env):
        manager, sid, workspace = recovery_env
        _put(workspace, "c.py", "class A:\n    def g(self):\n        return 1\n")
        out = ft._handle_sandbox_str_replace(
            sid, "c.py", "def g(self):\n    return 1",
            "def g(self):\n    return 2", _sandbox_manager=manager)
        assert out == "Replaced (matched via indentation-flexible normalization)."
        assert _get(workspace, "c.py") == (
            "class A:\n    def g(self):\n        return 2\n"
        )

    def test_indentation_flexible_negative_delta(self, recovery_env):
        manager, sid, workspace = recovery_env
        _put(workspace, "d.py", "return 5\n")
        out = ft._handle_sandbox_str_replace(
            sid, "d.py", "    return 5", "    return 6", _sandbox_manager=manager)
        assert out == "Replaced (matched via indentation-flexible normalization)."
        assert _get(workspace, "d.py") == "return 6\n"

    def test_two_normalized_candidates_fail_the_whole_call(self, recovery_env):
        manager, sid, workspace = recovery_env
        original = "x  = 1\nmid\nx   = 1\n"
        _put(workspace, "e.py", original)
        out = ft._handle_sandbox_str_replace(
            sid, "e.py", "x = 1", "x = 9", _sandbox_manager=manager)
        assert out == (
            "Error: old_str matched 2 regions after normalization; "
            "make it unique"
        )
        assert _get(workspace, "e.py") == original

    def test_two_trimmed_multiline_candidates_fail(self, recovery_env):
        manager, sid, workspace = recovery_env
        original = "q1 \nq2\nmid\nq1\t\nq2\n"
        _put(workspace, "f.py", original)
        out = ft._handle_sandbox_str_replace(
            sid, "f.py", "q1\nq2", "z", _sandbox_manager=manager)
        assert out == (
            "Error: old_str matched 2 regions after normalization; "
            "make it unique"
        )
        assert _get(workspace, "f.py") == original

    def test_clean_miss_keeps_message_and_appends_hint(self, recovery_env):
        manager, sid, workspace = recovery_env
        _put(workspace, "g.py", "alpha = 1\nbeta = 2\ngamma = 3\n")
        out = ft._handle_sandbox_str_replace(
            sid, "g.py", "delta = 4", "x", _sandbox_manager=manager)
        assert out.startswith("Error: String not found in g.py. ")
        assert "not found" in out.lower()
        assert "matches exactly" in out
        assert "\nclosest lines: " in out
        assert re.search(r"closest lines: \d+: ", out)

    def test_hint_lists_at_most_three_lines(self, recovery_env):
        manager, sid, workspace = recovery_env
        _put(workspace, "h.py", "\n".join(f"line{i} = {i}" for i in range(20)) + "\n")
        out = ft._handle_sandbox_str_replace(
            sid, "h.py", "zzz_missing", "x", _sandbox_manager=manager)
        hint = out.split("closest lines: ", 1)[1]
        assert len(hint.split("; ")) <= 3

    def test_multi_exact_message_untouched(self, recovery_env):
        manager, sid, workspace = recovery_env
        original = "foo\nbar\nfoo\n"
        _put(workspace, "i.py", original)
        out = ft._handle_sandbox_str_replace(
            sid, "i.py", "foo", "baz", _sandbox_manager=manager)
        assert "2 times" in out
        assert "closest lines" not in out
        assert _get(workspace, "i.py") == original

    def test_exact_success_untouched(self, recovery_env):
        manager, sid, workspace = recovery_env
        _put(workspace, "j.py", "x = 1\ny = 2\n")
        out = ft._handle_sandbox_str_replace(
            sid, "j.py", "y = 2", "y = 42", _sandbox_manager=manager)
        assert "successful" in out.lower()

    def test_deletion_via_recovery_removes_lines(self, recovery_env):
        manager, sid, workspace = recovery_env
        _put(workspace, "k.py", "keep\nthe  end\nkeep2\n")
        out = ft._handle_sandbox_str_replace(
            sid, "k.py", "the end", "", _sandbox_manager=manager)
        assert out == "Replaced (matched via whitespace-normalized normalization)."
        assert _get(workspace, "k.py") == "keep\nkeep2\n"

    def test_blank_only_old_str_skips_recovery(self, recovery_env):
        manager, sid, workspace = recovery_env
        _put(workspace, "l.py", "a\nb\n")
        out = ft._handle_sandbox_str_replace(
            sid, "l.py", "\t\t", "x", _sandbox_manager=manager)
        assert out.startswith("Error: String not found in l.py.")

    def test_error_paths_untouched(self, recovery_env):
        manager, sid, workspace = recovery_env
        assert "Error" in ft._handle_sandbox_str_replace(
            sid, "ghost.py", "a", "b", _sandbox_manager=manager)
        _put(workspace, "m.py", "content")
        assert "Error" in ft._handle_sandbox_str_replace(
            sid, "m.py", "", "x", _sandbox_manager=manager)


# ---------------------------------------------------------------------------
# Recovery x diagnostics composition (6.4 + S228 5.2, session layer)
# ---------------------------------------------------------------------------


class WorkspaceMgr:
    def __init__(self, workspace: str):
        self.workspace = workspace

    def get_workspace_path(self, session_id):
        return self.workspace


class DiagMgr(WorkspaceMgr):
    def __init__(self, workspace: str, responder):
        super().__init__(workspace)
        self.responder = responder
        self.commands: list[str] = []

    def execute_command(self, session_id, command, timeout=None):
        self.commands.append(command)
        return self.responder(command)


class CmdResult:
    def __init__(self, rc, stdout="", stderr="", blocked=False, timed_out=False):
        self.return_code = rc
        self.stdout = stdout
        self.stderr = stderr
        self.blocked = blocked
        self.timed_out = timed_out


def _session(workspace: str, backend: str = "bwrap", mgr=None):
    s = st.SandboxToolSession(sandbox_mgr=mgr or WorkspaceMgr(workspace), tool_registry=None)
    s._session = types.SimpleNamespace(active=True, isolation_backend=backend)
    s._session_id = "s229-diag"
    return s


def _responder_with_findings(command):
    if command.startswith("command -v ruff"):
        return CmdResult(0, "/usr/bin/ruff")
    if command.startswith("ruff check"):
        return CmdResult(1, "r.py:1:5: E999 SyntaxError")
    return CmdResult(1)


class TestRecoveryDiagnosticsComposition:
    def test_recovered_success_runs_diagnostics(self, tmp_path):
        mgr = DiagMgr(str(tmp_path), _responder_with_findings)
        sess = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        # Trailing spaces before the newline: the exact match (old_str ends
        # with "\n") misses, the line-trimmed stage recovers.
        (tmp_path / "r.py").write_text("x = (   \n", encoding="utf-8")
        out = sess.str_replace("r.py", "x = (\n", "y = (\n")
        assert out.startswith("Replaced (matched via line-trimmed normalization).")
        assert "[diagnostics]" in out
        assert any(c.startswith("ruff check") for c in mgr.commands)
        assert (tmp_path / "r.py").read_text(encoding="utf-8") == "y = (\n"

    def test_k_regions_failure_skips_diagnostics(self, tmp_path):
        mgr = DiagMgr(str(tmp_path), _responder_with_findings)
        sess = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        (tmp_path / "s.py").write_text("x  = 1\nmid\nx   = 1\n", encoding="utf-8")
        out = sess.str_replace("s.py", "x = 1", "x = 9")
        assert out.startswith("Error: old_str matched 2 regions")
        assert mgr.commands == []


# ---------------------------------------------------------------------------
# Max-steps reminder (6.5)
# ---------------------------------------------------------------------------


class TestMaxStepsReminder:
    def _varied(self, n):
        return [
            {"content": f"c{i}", "tool_calls": _native("view", {"path": f"/r{i}"})}
            for i in range(1, n + 1)
        ]

    def test_fires_once_before_the_penultimate_round(self):
        script = self._varied(3) + [{"content": "done", "tool_calls": None}]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=LoopSession(),
                    mode="daily", include_memory=False, max_rounds=4)
        reminders = [
            i for i, m in enumerate(res.messages)
            if m.get("content") == L.MAX_STEPS_REMINDER
        ]
        assert len(reminders) == 1
        assert res.messages[reminders[0]]["role"] == "user"
        c3_index = next(
            i for i, m in enumerate(res.messages)
            if m.get("role") == "assistant" and m.get("content") == "c3"
        )
        assert reminders[0] < c3_index

    def test_absent_under_a_distant_cap(self):
        script = self._varied(2) + [{"content": "done", "tool_calls": None}]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=LoopSession(),
                    mode="daily", include_memory=False)
        assert all(m.get("content") != L.MAX_STEPS_REMINDER for m in res.messages)

    def test_fires_at_round_one_when_cap_is_two(self):
        script = self._varied(2)
        res = L.run("q", model_client=ScriptedClient(script), sandbox=LoopSession(),
                    mode="daily", include_memory=False, max_rounds=2)
        count = sum(
            1 for m in res.messages if m.get("content") == L.MAX_STEPS_REMINDER
        )
        assert count == 1

    def test_recomputes_after_task_debit(self):
        script = [
            {"content": "p1", "tool_calls": _native(
                "task", {"description": "d", "prompt": "go", "max_rounds": 2})},
            {"content": "c1", "tool_calls": _native("bash", {"command": "a"})},
            {"content": "c2", "tool_calls": _native("bash", {"command": "b"})},
            {"content": "p2", "tool_calls": _native("bash", {"command": "c"})},
            {"content": "p3", "tool_calls": _native("bash", {"command": "e"})},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=LoopSession(),
                    mode="daily", include_memory=False, max_rounds=5)
        count = sum(
            1 for m in res.messages if m.get("content") == L.MAX_STEPS_REMINDER
        )
        assert count == 1
        assert res.stop_reason == L.STOP_MAX_ROUNDS


# ---------------------------------------------------------------------------
# Governor branch (6.6)
# ---------------------------------------------------------------------------


class TestGovernorBranch:
    def test_static_when_no_ticket(self):
        assert L._derive_budgets(L.HARDENING_DEFAULTS, None) == (49152, 98304, 65536)

    def test_fed_raises_with_static_ratios(self):
        assert L._derive_budgets(L.HARDENING_DEFAULTS, 65536) == (
            91750, 183500, 122333,
        )

    def test_small_ctx_keeps_static_floors(self):
        assert L._derive_budgets(L.HARDENING_DEFAULTS, 8192) == (49152, 98304, 65536)

    def test_ticket_num_ctx_admit_and_downsize(self, monkeypatch):
        monkeypatch.setattr(
            L, "_get_active_ticket",
            lambda: types.SimpleNamespace(action="admit", num_ctx=32768))
        assert L._ticket_num_ctx() == 32768
        monkeypatch.setattr(
            L, "_get_active_ticket",
            lambda: types.SimpleNamespace(action="downsize", num_ctx=16384))
        assert L._ticket_num_ctx() == 16384

    def test_ticket_refusal_or_empty_yields_none(self, monkeypatch):
        monkeypatch.setattr(
            L, "_get_active_ticket",
            lambda: types.SimpleNamespace(action="refuse", num_ctx=32768))
        assert L._ticket_num_ctx() is None
        monkeypatch.setattr(
            L, "_get_active_ticket",
            lambda: types.SimpleNamespace(action="admit", num_ctx=None))
        assert L._ticket_num_ctx() is None
        monkeypatch.setattr(L, "_get_active_ticket", lambda: None)
        assert L._ticket_num_ctx() is None

    def test_absent_governor_module_yields_none(self, monkeypatch):
        monkeypatch.setattr(L, "_get_active_ticket", None)
        assert L._ticket_num_ctx() is None

    def test_run_accepts_admitted_num_ctx_keyword(self):
        res = L.run("q",
                    model_client=ScriptedClient([{"content": "done", "tool_calls": None}]),
                    sandbox=LoopSession(), mode="daily", include_memory=False,
                    admitted_num_ctx=65536)
        assert res.stop_reason == L.STOP_DONE

    def test_fed_budget_suppresses_prune_static_triggers(self, monkeypatch):
        merged = dict(PRUNE_CFG)
        monkeypatch.setattr(L, "load_hardening_config", lambda: merged)
        static_res = L.run(
            "q", model_client=ScriptedClient(_six_view_rounds()),
            sandbox=_PruneSession(), mode="daily", include_memory=False)
        fed_res = L.run(
            "q", model_client=ScriptedClient(_six_view_rounds()),
            sandbox=_PruneSession(), mode="daily", include_memory=False,
            admitted_num_ctx=65536)
        assert "[pruned observation" in _joined(static_res)
        assert "[pruned observation" not in _joined(fed_res)

    def test_thread_local_ticket_feeds_run(self, monkeypatch):
        merged = dict(PRUNE_CFG)
        monkeypatch.setattr(L, "load_hardening_config", lambda: merged)
        monkeypatch.setattr(
            L, "_get_active_ticket",
            lambda: types.SimpleNamespace(action="admit", num_ctx=65536))
        res = L.run(
            "q", model_client=ScriptedClient(_six_view_rounds()),
            sandbox=_PruneSession(), mode="daily", include_memory=False)
        assert "[pruned observation" not in _joined(res)


# ---------------------------------------------------------------------------
# Facade and yaml surface
# ---------------------------------------------------------------------------


class TestFacadeAndYaml:
    def test_facade_source_exports_s229_names(self):
        src = (AGENT / "__init__.py").read_text(encoding="utf-8")
        for needle in (
            "STOP_DOOM_LOOP",
            "MAX_STEPS_REMINDER",
            "HARDENING_DEFAULTS",
            "load_hardening_config",
            "AGENT_OBS_MAX_BYTES",
            "DOOM_LOOP_THRESHOLD",
        ):
            assert needle in src, needle

    def test_yaml_hardening_block_matches_defaults(self):
        import yaml as real_yaml

        data = real_yaml.safe_load(
            (AGENT / "config.yaml").read_text(encoding="utf-8")
        )
        assert data["hardening"] == L.HARDENING_DEFAULTS

    def test_yaml_top_level_sections_intact(self):
        import yaml as real_yaml

        data = real_yaml.safe_load(
            (AGENT / "config.yaml").read_text(encoding="utf-8")
        )
        for key in ("loop", "teacher", "tools", "diagnostics", "hardening", "presets"):
            assert key in data, key

    def test_legacy_observations_builder_still_present(self):
        # The verifier path keeps the uncapped builder (S229 leaves
        # _run_verifier unchanged); the seam itself must stay.
        results = [_dr("view", "tiny")]
        msg = L._observations_message(results)
        assert msg is not None and msg["role"] == "user"
