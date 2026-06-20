#!/usr/bin/env python3
"""S227 -- Resource Governor cycle, Bloc 4: surfaces and close.

Per-fix suite for RESOURCE_GOVERNOR_SPEC Section 9: the /api/governor route
surface (status, admissions, eviction, config read/write) with auth parity, the
Performance & Telemetry status card and its api-client module registered in
FRONTEND_REDESIGN_SPEC, the docs close (SECURITY.md advisory paragraph, README
note, the two ROADMAP residual entries), and the v3.9.0 cycle-close release.

The route layer imports the governor and edits nothing in it; the status seat
(pressure_state, queue_depth, ollama_limits_advisory, get_snapshot_fast,
recent_decisions, evict_model) is consumed as landed at S223-S226.

Harness: the web-free payload helpers are tested directly against a fake
governor (no fastapi); the FastAPI routes are exercised through the s213
_load_fresh idiom with routes_auth bound to None (the ImportError fallback makes
_auth_dep empty) so the TestClient needs no auth. The config-write test drives a
tmp copy of the shipped YAML so the comment-preservation contract is observable.
"""

import importlib.util
import os
import re
import sys
import types

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_OO = os.path.join(_ROOT, "opti_oignon")
_API = os.path.join(_OO, "api")
_CONFIG_YAML = os.path.join(_OO, "config", "resource_governor.yaml")
_FRONT = os.path.join(_ROOT, "frontend", "src")


def _read(*parts: str) -> str:
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _ensure_pkg(name: str, path: str) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod


_ensure_pkg("opti_oignon", _OO)
_ensure_pkg("opti_oignon.api", _API)


def _load_fresh(relpath: str, register: str, bind: dict | None = None):
    """Exec this file's own copy of a module with a temporary sys.modules bind."""
    bind = dict(bind or {})
    touched = list(bind.keys()) + [register]
    saved = {name: sys.modules.get(name) for name in touched}
    try:
        for name, mod in bind.items():
            sys.modules[name] = mod
        path = os.path.join(_ROOT, relpath)
        spec = importlib.util.spec_from_file_location(register, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[register] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


# ---------------------------------------------------------------------------
# A fake governor mirroring the landed status seat
# ---------------------------------------------------------------------------


class _FakeSnapshot:
    def __init__(self):
        self._d = {
            "taken_at": 1.0,
            "ttl_s": 2.0,
            "loaded": [{"name": "qwen3:32b", "size_vram_bytes": 21000000000}],
            "backend_resident": [],
            "capacity_gb": 24.0,
            "capacity_source": "ollama_ps",
            "vram_in_use_gb": 20.0,
            "vram_available_gb": 4.0,
            "vram_status": "ok",
            "ram_available_mb": 8000.0,
            "sources": ["ollama_ps", "proc_meminfo"],
        }

    def to_dict(self):
        return dict(self._d)


class _FakeStore:
    def __init__(self, decisions=None, ceiling=22.0):
        self._decisions = decisions or [
            {
                "id": 2,
                "ts": 2.0,
                "caller": "chat",
                "model": "llama3:70b",
                "requested_ctx": 8192,
                "admitted_ctx": None,
                "decision": "refuse",
                "reason": "insufficient vram",
            },
            {
                "id": 1,
                "ts": 1.0,
                "caller": "chat",
                "model": "qwen3:32b",
                "requested_ctx": 8192,
                "admitted_ctx": 8192,
                "decision": "admit",
                "reason": "",
            },
        ]
        self._ceiling = ceiling

    def recent_decisions(self, limit=20):
        return self._decisions[: max(1, int(limit))]

    def get_learned_ceiling(self):
        return self._ceiling


class _FakeConfig:
    """A GovernorConfig-shaped object (the route reads attributes only)."""

    enabled = True
    total_vram_gb = None
    safety_margin_gb = 1.5
    snapshot_ttl_s = 2.0
    kv_coefficient = 0.5
    ceiling_floor_gb = 4.0
    decisions_ring_size = 200
    ctx_ladder = [32768, 16384, 8192, 4096]
    ctx_floor = {"chat": 4096, "pipeline": 4096}
    idle_evict_threshold_s = 600.0
    pressure_soft_threshold = 0.85
    pressure_hard_threshold = 0.95
    pressure_sustain_s = 60.0
    pressure_refusal_window_s = 60.0
    pressure_keep_alive = "5m"
    queue_enabled_per_caller: dict = {}
    queue_depth = 2
    queue_wait_s = 30.0
    rlimits_enabled = False
    rlimits_as_gb = None
    rlimits_data_gb = None
    ollama_max_loaded_models = None
    ollama_num_parallel = None
    ollama_max_queue = None
    ollama_spawn_applies = True
    ollama_external_advisory = True


class _FakeGovernor:
    def __init__(self, evict_result=True):
        self.config = _FakeConfig()
        self.store = _FakeStore()
        self._evict_result = evict_result
        self.evict_calls = []
        self._snapshot_calls = 0
        self._pressure_calls = 0

    def get_snapshot_fast(self):
        self._snapshot_calls += 1
        return _FakeSnapshot()

    def pressure_state(self):
        self._pressure_calls += 1
        return {
            "level": "soft",
            "ratio": 0.833,
            "effective_capacity_gb": 24.0,
            "in_use_gb": 20.0,
            "soft_threshold": 0.85,
            "hard_threshold": 0.95,
            "refusal_rate": 0.5,
            "refusals_in_window": 1,
            "decisions_in_window": 2,
            "refusal_window_s": 60.0,
            "keep_alive_overridden": False,
        }

    @property
    def queue_depth(self):
        return 0

    def ollama_limits_advisory(self):
        return {"status": "not_configured"}

    def evict_model(self, model, trigger="manual"):
        self.evict_calls.append((model, trigger))
        return self._evict_result


def _load_routes(fake_module):
    """Load routes_governor fresh with the fake governor and no auth."""
    return _load_fresh(
        os.path.join("opti_oignon", "api", "routes_governor.py"),
        register="opti_oignon.api.routes_governor",
        bind={
            "opti_oignon.resource_governor": fake_module,
            "opti_oignon.api.routes_auth": None,
        },
    )


def _fake_rg_module(governor, config_path, reset_capture):
    mod = types.ModuleType("opti_oignon.resource_governor")
    mod.get_resource_governor = lambda *a, **k: governor
    mod.reset_resource_governor = reset_capture
    mod._DEFAULT_CONFIG_PATH = config_path
    return mod


# ---------------------------------------------------------------------------
# Web-free payload helpers (no fastapi)
# ---------------------------------------------------------------------------

# routes_governor's helpers are importable with fastapi present; load the module
# once for the helper tests (auth bound to None, a fake governor module bound).
# Guarded so a pristine tree (no routes_governor.py yet) collects and shows
# per-test reds rather than a collection error.
_RG_FAKE = _fake_rg_module(_FakeGovernor(), _CONFIG_YAML, lambda: None)
try:
    RG = _load_routes(_RG_FAKE)
except Exception:  # pragma: no cover - pristine tree only
    RG = None


class TestStatusPayload:
    def test_keys_and_shape(self):
        gov = _FakeGovernor()
        payload = RG.status_payload(gov)
        assert set(payload) == {
            "enabled",
            "snapshot",
            "learned_ceiling_gb",
            "pressure",
            "queue_depth",
            "ollama_limits",
        }
        assert payload["enabled"] is True
        assert payload["learned_ceiling_gb"] == 22.0
        assert payload["queue_depth"] == 0
        assert payload["ollama_limits"]["status"] == "not_configured"

    def test_snapshot_carries_provenance_and_capacity(self):
        snap = RG.status_payload(_FakeGovernor())["snapshot"]
        assert snap["sources"] == ["ollama_ps", "proc_meminfo"]
        assert snap["capacity_gb"] == 24.0
        assert snap["capacity_source"] == "ollama_ps"

    def test_pressure_surfaced_verbatim(self):
        pressure = RG.status_payload(_FakeGovernor())["pressure"]
        assert pressure["level"] == "soft"
        assert pressure["keep_alive_overridden"] is False

    def test_one_call_per_surface(self):
        gov = _FakeGovernor()
        RG.status_payload(gov)
        # get_snapshot_fast: one direct + one inside the fake pressure_state's
        # caller is the route, not re-derived; the helper itself calls it once.
        assert gov._snapshot_calls == 1
        assert gov._pressure_calls == 1


class TestAdmissionsPayload:
    def test_shape_and_default(self):
        payload = RG.admissions_payload(_FakeGovernor())
        assert set(payload) == {"admissions", "count", "limit", "ring_size"}
        assert payload["count"] == len(payload["admissions"])
        assert payload["limit"] == 20
        assert payload["ring_size"] == 200

    def test_limit_clamped_to_ring_size(self):
        payload = RG.admissions_payload(_FakeGovernor(), limit=100000)
        assert payload["limit"] == 200  # decisions_ring_size

    def test_limit_floor_is_one(self):
        payload = RG.admissions_payload(_FakeGovernor(), limit=0)
        assert payload["limit"] == 1


class TestEvictPayload:
    def test_true(self):
        gov = _FakeGovernor(evict_result=True)
        payload = RG.evict_payload(gov, "qwen3:32b")
        assert payload["evicted"] is True
        assert payload["model"] == "qwen3:32b"
        assert gov.evict_calls == [("qwen3:32b", "api")]

    def test_false_is_fail_open_not_error(self):
        gov = _FakeGovernor(evict_result=False)
        payload = RG.evict_payload(gov, "absent")
        assert payload["evicted"] is False
        assert "LRU" in payload["note"]


class TestConfigReadPayload:
    def test_nested_mirror_and_key_sets(self):
        payload = RG.config_read_payload(_FakeGovernor())
        cfg = payload["config"]
        assert cfg["pressure"]["soft_threshold"] == 0.85
        assert cfg["queue"]["depth"] == 2
        assert cfg["ollama_limits"]["spawn_applies"] is True
        assert cfg["rlimits"]["enabled"] is False
        assert "safety_margin_gb" in payload["writable_keys"]
        assert "pressure.soft_threshold" in payload["writable_keys"]
        # rlimits and the structured keys are read-only
        assert "rlimits.enabled" in payload["read_only_keys"]
        assert "ctx_ladder" in payload["read_only_keys"]
        assert "queue.enabled_per_caller" in payload["read_only_keys"]


class TestConfigWrite:
    def _tmp_yaml(self, tmp_path):
        text = open(_CONFIG_YAML, encoding="utf-8").read()
        p = tmp_path / "resource_governor.yaml"
        p.write_text(text, encoding="utf-8")
        return p

    def test_top_level_and_nested_apply_with_comments_preserved(self, tmp_path):
        p = self._tmp_yaml(tmp_path)
        reset_calls = []
        audit_calls = []
        result = RG.config_write_payload(
            _FakeConfig(),
            {"safety_margin_gb": 2.0, "pressure.soft_threshold": 0.8},
            config_path=str(p),
            reset_fn=lambda: reset_calls.append(True),
            audit_fn=lambda changes: audit_calls.append(changes),
        )
        text = p.read_text(encoding="utf-8")
        assert "safety_margin_gb: 2.0" in text
        assert re.search(r"^  soft_threshold: 0\.8$", text, re.M)
        # The spec caveat comments survive the write (the contract).
        assert "setrlimit is process-wide" in text
        assert "speculative_decoding precedent value" in text
        assert result["persisted"] is True
        assert "next governor access" in result["effective"]
        assert result["applied"]["safety_margin_gb"] == {"old": 1.5, "new": 2.0}
        assert reset_calls == [True]
        assert audit_calls and "safety_margin_gb" in audit_calls[0]

    def test_string_key_quoted(self, tmp_path):
        p = self._tmp_yaml(tmp_path)
        RG.config_write_payload(
            _FakeConfig(),
            {"pressure_keep_alive": "10m"},
            config_path=str(p),
            reset_fn=lambda: None,
            audit_fn=lambda c: None,
        )
        assert 'pressure_keep_alive: "10m"' in p.read_text(encoding="utf-8")

    def test_unknown_key_400_no_write(self, tmp_path):
        p = self._tmp_yaml(tmp_path)
        before = p.read_text(encoding="utf-8")
        with pytest.raises(RG.ConfigWriteError) as ei:
            RG.config_write_payload(
                _FakeConfig(),
                {"nope": 1},
                config_path=str(p),
                reset_fn=lambda: None,
                audit_fn=lambda c: None,
            )
        assert ei.value.status_code == 400
        assert p.read_text(encoding="utf-8") == before

    def test_rlimits_read_only_400(self, tmp_path):
        p = self._tmp_yaml(tmp_path)
        with pytest.raises(RG.ConfigWriteError) as ei:
            RG.config_write_payload(
                _FakeConfig(),
                {"rlimits.enabled": True},
                config_path=str(p),
                reset_fn=lambda: None,
                audit_fn=lambda c: None,
            )
        assert ei.value.status_code == 400
        assert "read-only" in ei.value.detail

    def test_bad_type_400(self, tmp_path):
        p = self._tmp_yaml(tmp_path)
        with pytest.raises(RG.ConfigWriteError) as ei:
            RG.config_write_payload(
                _FakeConfig(),
                {"safety_margin_gb": "lots"},
                config_path=str(p),
                reset_fn=lambda: None,
                audit_fn=lambda c: None,
            )
        assert ei.value.status_code == 400

    def test_bool_rejected_for_numeric(self, tmp_path):
        p = self._tmp_yaml(tmp_path)
        with pytest.raises(RG.ConfigWriteError):
            RG.config_write_payload(
                _FakeConfig(),
                {"decisions_ring_size": True},
                config_path=str(p),
                reset_fn=lambda: None,
                audit_fn=lambda c: None,
            )

    def test_missing_line_409(self, tmp_path):
        p = tmp_path / "resource_governor.yaml"
        p.write_text("enabled: true\n", encoding="utf-8")  # hand-edited, minimal
        with pytest.raises(RG.ConfigWriteError) as ei:
            RG.config_write_payload(
                _FakeConfig(),
                {"safety_margin_gb": 2.0},
                config_path=str(p),
                reset_fn=lambda: None,
                audit_fn=lambda c: None,
            )
        assert ei.value.status_code == 409

    def test_all_or_nothing_aborts_before_write(self, tmp_path):
        p = self._tmp_yaml(tmp_path)
        before = p.read_text(encoding="utf-8")
        with pytest.raises(RG.ConfigWriteError):
            RG.config_write_payload(
                _FakeConfig(),
                {"safety_margin_gb": 2.0, "rlimits.enabled": True},
                config_path=str(p),
                reset_fn=lambda: None,
                audit_fn=lambda c: None,
            )
        assert p.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# FastAPI routes through TestClient (auth bound out)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    text = open(_CONFIG_YAML, encoding="utf-8").read()
    cfg = tmp_path / "resource_governor.yaml"
    cfg.write_text(text, encoding="utf-8")
    gov = _FakeGovernor()
    reset_calls = []
    fake = _fake_rg_module(gov, str(cfg), lambda: reset_calls.append(True))
    rs = _load_routes(fake)
    app = fastapi.FastAPI()
    app.include_router(rs.router)
    return rs, TestClient(app), gov, cfg, reset_calls


class TestRoutes:
    def test_status_200(self, client):
        _, c, _, _, _ = client
        resp = c.get("/api/governor/status")
        assert resp.status_code == 200
        assert set(resp.json()) == {
            "enabled", "snapshot", "learned_ceiling_gb", "pressure",
            "queue_depth", "ollama_limits",
        }

    def test_admissions_200_and_clamp(self, client):
        _, c, _, _, _ = client
        resp = c.get("/api/governor/admissions", params={"limit": 100000})
        assert resp.status_code == 200
        assert resp.json()["limit"] == 200

    def test_evict_200(self, client):
        _, c, gov, _, _ = client
        resp = c.post("/api/governor/evict", json={"model": "qwen3:32b"})
        assert resp.status_code == 200
        assert resp.json()["evicted"] is True
        assert gov.evict_calls == [("qwen3:32b", "api")]

    def test_evict_400_missing_model(self, client):
        _, c, _, _, _ = client
        resp = c.post("/api/governor/evict", json={})
        assert resp.status_code == 400

    def test_config_get_200(self, client):
        _, c, _, _, _ = client
        resp = c.get("/api/governor/config")
        assert resp.status_code == 200
        assert "writable_keys" in resp.json()

    def test_config_post_200_persists_and_resets(self, client):
        _, c, _, cfg, reset_calls = client
        resp = c.post(
            "/api/governor/config", json={"safety_margin_gb": 2.5}
        )
        assert resp.status_code == 200
        assert resp.json()["applied"]["safety_margin_gb"]["new"] == 2.5
        assert "safety_margin_gb: 2.5" in cfg.read_text(encoding="utf-8")
        assert reset_calls == [True]

    def test_config_post_400_unknown(self, client):
        _, c, _, _, _ = client
        resp = c.post("/api/governor/config", json={"nope": 1})
        assert resp.status_code == 400

    def test_config_post_400_empty(self, client):
        _, c, _, _, _ = client
        resp = c.post("/api/governor/config", json={})
        assert resp.status_code == 400

    def test_status_503_when_unavailable(self, client):
        rs, c, _, _, _ = client
        rs._GOVERNOR_OK = False
        try:
            resp = c.get("/api/governor/status")
            assert resp.status_code == 503
        finally:
            rs._GOVERNOR_OK = True


class TestConfigWriteAuditThroughSeededChain:
    def test_default_audit_calls_chain_log(self, tmp_path, monkeypatch):
        # Seed a fake signed_audit_log and assert _default_audit appends to it.
        captured = {}

        def _chain_log(**kwargs):
            captured.update(kwargs)
            return 1

        fake_sal = types.ModuleType("opti_oignon.signed_audit_log")
        fake_sal.chain_log = _chain_log
        monkeypatch.setitem(
            sys.modules, "opti_oignon.signed_audit_log", fake_sal
        )
        text = open(_CONFIG_YAML, encoding="utf-8").read()
        p = tmp_path / "resource_governor.yaml"
        p.write_text(text, encoding="utf-8")
        RG.config_write_payload(
            _FakeConfig(),
            {"snapshot_ttl_s": 3.0},
            config_path=str(p),
            reset_fn=lambda: None,
        )
        assert captured.get("event_type") == "resource_governor"
        assert captured.get("action") == "config_change"
        assert "snapshot_ttl_s" in captured.get("changes", {})


# ---------------------------------------------------------------------------
# Auth parity and app registration (source pins, the SYN-06 idiom)
# ---------------------------------------------------------------------------


class TestAuthParity:
    SRC = None

    def setup_method(self):
        self.SRC = _read("opti_oignon", "api", "routes_governor.py")

    def test_router_declares_auth_dependency(self):
        assert "dependencies=_auth_dep" in self.SRC

    def test_guarded_auth_import(self):
        assert "from .routes_auth import _get_current_user" in self.SRC
        assert "_auth_dep = [Depends(_get_current_user)]" in self.SRC
        assert "_auth_dep = []" in self.SRC

    def test_prefix_and_router_fallback(self):
        assert 'prefix="/api/governor"' in self.SRC
        assert "router = None" in self.SRC

    def test_sentinels_and_ascii(self):
        assert "checkpoint_before_apply = True" in self.SRC
        assert "FEATURE_AVAILABLE = True" in self.SRC
        self.SRC.encode("ascii")


class TestAppRegistration:
    def setup_method(self):
        self.SRC = _read("opti_oignon", "api", "app.py")

    def test_import(self):
        assert "from .routes_governor import router as governor_router" in self.SRC

    def test_guarded_include(self):
        assert "if governor_router is not None:" in self.SRC
        assert "app.include_router(governor_router)" in self.SRC


# ---------------------------------------------------------------------------
# Frontend: FRD registration, the component, the api module, the mount
# ---------------------------------------------------------------------------


class TestFrdRegistration:
    def test_new_row(self):
        frd = _read("FRONTEND_REDESIGN_SPEC.md")
        assert "`GovernorPanel.svelte` | NEW | S227" in frd
        assert "Resource Governor status card" in frd
        assert "Settings > Performance & Telemetry" in frd


class TestComponent:
    def setup_method(self):
        self.path = os.path.join(
            "frontend", "src", "lib", "components", "panels",
            "GovernorPanel.svelte",
        )
        self.src = _read(self.path)

    def test_exists_balanced_and_token_only(self):
        src = self.src
        body = re.sub(r"<!--.*?-->", "", src, flags=re.S)
        body = re.sub(r"(<script[^>]*>).*?(</script>)", r"\1\2", body, flags=re.S)
        markup = re.sub(r"(<style[^>]*>).*?(</style>)", r"\1\2", body, flags=re.S)
        from collections import Counter

        opens = Counter(re.findall(r"<([A-Za-z][A-Za-z0-9.-]*)\b[^>]*?(?<!/)>", markup))
        closes = Counter(re.findall(r"</([A-Za-z][A-Za-z0-9.-]*)>", markup))
        for tag in set(opens) | set(closes):
            if tag.lower() in ("input", "br", "hr", "img"):
                continue
            assert opens[tag] == closes[tag], f"unbalanced <{tag}>"
        for kind in ("if", "each"):
            assert len(re.findall(r"\{#" + kind + r"\b", markup)) == len(
                re.findall(r"\{/" + kind + r"\}", markup)
            )
        for m in re.finditer(r"#[0-9a-fA-F]{3,8}\b", src):
            ctx = src[max(0, m.start() - 60):m.start()]
            assert re.search(r"var\(\s*--oo-[a-z0-9-]+\s*,\s*$", ctx), (
                f"raw hex at offset {m.start()}"
            )

    def test_honest_surfaces_and_registration(self):
        assert "mode-free" in self.src
        assert "provenance" in self.src
        assert "fail-open" in self.src
        assert "Registered in FRONTEND_REDESIGN_SPEC.md" in self.src

    def test_ds_primitives_and_governor_api(self):
        assert "from '$lib/ds'" in self.src
        assert "from '$lib/api/governor'" in self.src
        assert "onMount(load)" in self.src


class TestApiModule:
    def test_exists_with_functions(self):
        src = _read("frontend", "src", "lib", "api", "governor.ts")
        for fn in (
            "getGovernorStatus",
            "getGovernorAdmissions",
            "evictGovernorModel",
            "getGovernorConfig",
            "setGovernorConfig",
        ):
            assert f"export function {fn}" in src, fn
        assert "'/api/governor'" in src


class TestSettingsMount:
    def setup_method(self):
        self.src = _read("frontend", "src", "routes", "settings", "+page.svelte")

    def test_loader_entry(self):
        assert (
            "GovernorPanel: () => import('$lib/components/panels/GovernorPanel.svelte')"
            in self.src
        )

    def test_group_row_in_performance_section(self):
        assert "id: 'resource-governor'" in self.src
        assert "panel: 'GovernorPanel'" in self.src


# ---------------------------------------------------------------------------
# Docs close
# ---------------------------------------------------------------------------


class TestSecurityMd:
    def setup_method(self):
        self.src = _read("SECURITY.md")

    def test_governor_paragraph(self):
        assert "Resource Governor" in self.src
        assert "availability control" in self.src
        assert "can never block" in self.src

    def test_rlimit_caveat_and_cgroup_pointer(self):
        assert "setrlimit is process-wide" in self.src
        assert "ollama_cgroup_limits.sh" in self.src
        assert "print-only" in self.src


class TestReadme:
    def setup_method(self):
        self.src = _read("README.md")

    def test_api_row(self):
        assert "| `/api/governor` |" in self.src

    def test_feature_section_and_intro(self):
        assert "## Features Added in v3.9.0 (Resource Governor Cycle)" in self.src
        assert "Opti-Oignon v3.9.0 sits between" in self.src
        assert "the resource governor cycle" in self.src


class TestRoadmap:
    def setup_method(self):
        self.src = _read("ROADMAP_POST_AUDIT.md")

    def test_cycle_rolled(self):
        assert "LANDED and RELEASED at S227 (v3.9.0)" in self.src

    def test_residual_entries(self):
        assert "GOV-W1 :: governor direct-caller residual wiring" in self.src
        assert "GOV-W2 :: Ollama spawn-path consumption" in self.src


# ---------------------------------------------------------------------------
# Release (v3.9.0): the cycle-close bump + the five s214 supersessions
# ---------------------------------------------------------------------------


class TestVersionRelease:
    FINAL = "3.9.0"
    PREVIOUS = "3.8.0"

    def test_version_file_is_390(self):
        src = _read("opti_oignon", "__version__.py")
        assert f'"{self.FINAL}"' in src
        assert f'"{self.PREVIOUS}"' not in src

    def test_version_bare_no_rc(self):
        m = re.search(
            r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py")
        )
        assert m and m.group(1) == self.FINAL

    def test_pyproject_version_is_390_and_hardcoded(self):
        src = _read("pyproject.toml")
        assert f'version = "{self.FINAL}"' in src
        import tomllib

        data = tomllib.loads(src)
        assert "dynamic" not in data["project"]
        assert data["project"]["version"] == self.FINAL

    def test_addopts_carries_the_s214_supersessions(self):
        src = _read("pyproject.toml")
        for node in (
            "test_s214_release.py::TestVersionRelease::test_version_file_is_380",
            "test_s214_release.py::TestVersionRelease::test_version_bare_no_rc",
            "test_s214_release.py::TestVersionRelease::test_pyproject_version_is_380_and_hardcoded",
            "test_s214_release.py::TestChangelogRelease::test_top_entry_is_380",
            "test_s214_release.py::TestReadmeRelease::test_sits_between_refreshed_to_380",
        ):
            assert f"--deselect=tests/{node}" in src, node


class TestChangelogRelease:
    def setup_method(self):
        self.c = _read("CHANGELOG.md")

    def test_top_entry_is_390(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == "3.9.0"

    def test_previous_entries_retained(self):
        assert "## v3.8.0 -- 2026-06-06 (S214)" in self.c
        assert "## v3.7.0 -- 2026-06-05 (S208)" in self.c

    def test_entry_tells_the_cycle_bloc_by_bloc(self):
        entry = self.c.split("## v3.9.0")[1].split("## v3.8.0")[0]
        for term in ("S223", "S224", "S225", "S226", "S227", "[SECURITY]"):
            assert term in entry, term


# ---------------------------------------------------------------------------
# Module conventions on the new route file
# ---------------------------------------------------------------------------


class TestModuleConventions:
    def setup_method(self):
        self.src = _read("opti_oignon", "api", "routes_governor.py")

    def test_helpers_defined_once(self):
        for fn in (
            "def status_payload(",
            "def admissions_payload(",
            "def evict_payload(",
            "def config_read_payload(",
            "def config_write_payload(",
            "def _set_yaml_scalar(",
        ):
            assert self.src.count(fn) == 1, fn

    def test_pure_ascii(self):
        self.src.encode("ascii")

    def test_ast_valid(self):
        import ast

        ast.parse(self.src)
