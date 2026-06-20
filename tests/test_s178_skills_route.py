#!/usr/bin/env python3
"""Tests for S178 Goal 0 -- the skills HTTP route (closes the S177 carry-over).

Covers opti_oignon/api/routes_agent.py: the SKILL.md registry surface the
skills-manager panel consumes (frontend api/skills.ts):

- ``GET    /api/agent/skills``                            list, with include_drafts
- ``GET    /api/agent/skills/{category}/{name}``          one skill, with its body
- ``POST   /api/agent/skills/{category}/{name}/publish``  publish a draft
- ``DELETE /api/agent/skills/{category}/{name}``          delete

The route logic is web-free (it takes a resolved registry and returns plain
payloads), so the contract is exercised in isolation against a real
SkillRegistry rooted at a temp directory: payload shapes, the include_drafts
filter, the published-first / else-draft resolution on view and delete, publish
promoting and removing a draft, and the never-raises behaviour on a traversal
payload or a miss. A separate, fastapi-guarded class drives the live FastAPI
surface end to end with a TestClient; it skips cleanly where fastapi is absent
(the sandbox case), which keeps it out of the regression baseline.

Loaded via spec_from_file_location with opti_oignon stubbed; rooted at a
temporary directory, so the runtime collects without the backend and leaves
nothing on disk.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"
API = OO / "api"


def _ensure_pkgs():
    for name, sub in (
        ("opti_oignon", OO),
        ("opti_oignon.agent", AGENT),
        ("opti_oignon.api", API),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod


def _load_agent(name: str):
    full = f"opti_oignon.agent.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(AGENT / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


_ensure_pkgs()
for _m in (
    "tool_parsing",
    "allowlists",
    "dispatch",
    "untrusted_context",
    "loop",
    "tools",
    "teacher",
):
    _load_agent(_m)
sk = _load_agent("skills")

_ra_spec = importlib.util.spec_from_file_location(
    "opti_oignon.api.routes_agent", str(API / "routes_agent.py")
)
ra = importlib.util.module_from_spec(_ra_spec)
sys.modules["opti_oignon.api.routes_agent"] = ra
_ra_spec.loader.exec_module(ra)


# A standard structured body for tests.
def _body(when="When deploying a service to the cluster.", proc="run the deploy script"):
    return (
        f"## When to Use\n{when}\n\n"
        f"## Procedure\n{proc}\n\n"
        "## Pitfalls\nDo not skip the health check.\n\n"
        "## Verification\nConfirm the service responds on its port.\n"
    )


@pytest.fixture
def reg(tmp_path):
    return sk.SkillRegistry(root=tmp_path)


@pytest.fixture(autouse=True)
def _reset():
    sk.reset_skill_registry()
    yield
    sk.reset_skill_registry()


# The wire shape skills.ts declares for a Skill (minus the optional body).
_SKILL_KEYS = {
    "name",
    "category",
    "status",
    "version",
    "source",
    "created_at",
    "updated_at",
}


# Module surface


class TestModuleSurface:
    """The web-free logic and the error type are importable without FastAPI."""

    def test_sentinels(self):
        assert ra.checkpoint_before_apply is True
        assert ra.FEATURE_AVAILABLE is True

    def test_logic_functions_present(self):
        for fn in (
            "skills_list_payload",
            "skill_view_payload",
            "skill_publish_payload",
            "skill_delete_payload",
        ):
            assert callable(getattr(ra, fn)), fn

    def test_not_found_is_exception(self):
        assert issubclass(ra.SkillNotFound, Exception)


# List payload


class TestListPayload:
    def test_empty_registry(self, reg):
        assert ra.skills_list_payload(reg) == {"skills": []}

    def test_lists_published(self, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        out = ra.skills_list_payload(reg, include_drafts=False)
        assert [s["name"] for s in out["skills"]] == ["deploy"]
        assert out["skills"][0]["status"] == "published"

    def test_includes_drafts_by_default(self, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        reg.add("rollback", "ops", _body(), status=sk.STATUS_DRAFT)
        names = {s["name"] for s in ra.skills_list_payload(reg)["skills"]}
        assert names == {"deploy", "rollback"}

    def test_exclude_drafts(self, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        reg.add("rollback", "ops", _body(), status=sk.STATUS_DRAFT)
        out = ra.skills_list_payload(reg, include_drafts=False)
        names = {s["name"] for s in out["skills"]}
        assert names == {"deploy"}

    def test_draft_marked_as_draft(self, reg):
        reg.add("rollback", "ops", _body(), status=sk.STATUS_DRAFT)
        entry = ra.skills_list_payload(reg)["skills"][0]
        assert entry["status"] == "draft"

    def test_list_entries_have_contract_keys_no_body(self, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        entry = ra.skills_list_payload(reg)["skills"][0]
        assert _SKILL_KEYS.issubset(entry.keys())
        assert "body" not in entry


# View payload


class TestViewPayload:
    def test_view_published_has_body(self, reg):
        reg.add("deploy", "ops", _body(proc="run deploy.sh"), status=sk.STATUS_PUBLISHED)
        out = ra.skill_view_payload(reg, "ops", "deploy")
        assert out["name"] == "deploy"
        assert out["status"] == "published"
        assert "run deploy.sh" in out["body"]

    def test_view_falls_back_to_draft(self, reg):
        reg.add("rollback", "ops", _body(proc="undo it"), status=sk.STATUS_DRAFT)
        out = ra.skill_view_payload(reg, "ops", "rollback")
        assert out["status"] == "draft"
        assert "undo it" in out["body"]

    def test_published_wins_over_draft(self, reg):
        # A draft proposed over an already-published skill: view returns published.
        reg.add("deploy", "ops", _body(proc="published body"), status=sk.STATUS_PUBLISHED)
        reg.add("deploy", "ops", _body(proc="draft body"), status=sk.STATUS_DRAFT)
        out = ra.skill_view_payload(reg, "ops", "deploy")
        assert out["status"] == "published"
        assert "published body" in out["body"]

    def test_missing_raises_not_found(self, reg):
        with pytest.raises(ra.SkillNotFound):
            ra.skill_view_payload(reg, "ops", "ghost")


# Publish payload


class TestPublishPayload:
    def test_publish_promotes_and_removes_draft(self, reg):
        reg.add("rollback", "ops", _body(), status=sk.STATUS_DRAFT)
        out = ra.skill_publish_payload(reg, "ops", "rollback")
        assert out["status"] == "published"
        assert "body" in out
        # The draft is gone; the published skill exists.
        assert reg.get("rollback", "ops", draft=True) is None
        assert reg.get("rollback", "ops", draft=False) is not None

    def test_publish_without_draft_raises(self, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        with pytest.raises(ra.SkillNotFound):
            ra.skill_publish_payload(reg, "ops", "deploy")

    def test_publish_unknown_raises(self, reg):
        with pytest.raises(ra.SkillNotFound):
            ra.skill_publish_payload(reg, "ops", "ghost")


# Delete payload


class TestDeletePayload:
    def test_delete_published(self, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        assert ra.skill_delete_payload(reg, "ops", "deploy") == {"deleted": True}
        assert reg.get("deploy", "ops", draft=False) is None

    def test_delete_draft_only(self, reg):
        reg.add("rollback", "ops", _body(), status=sk.STATUS_DRAFT)
        assert ra.skill_delete_payload(reg, "ops", "rollback") == {"deleted": True}
        assert reg.get("rollback", "ops", draft=True) is None

    def test_delete_missing_is_false(self, reg):
        assert ra.skill_delete_payload(reg, "ops", "ghost") == {"deleted": False}

    def test_delete_published_first_keeps_draft(self, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        reg.add("deploy", "ops", _body(), status=sk.STATUS_DRAFT)
        assert ra.skill_delete_payload(reg, "ops", "deploy") == {"deleted": True}
        assert reg.get("deploy", "ops", draft=False) is None
        # The pending draft is untouched.
        assert reg.get("deploy", "ops", draft=True) is not None


# Never raises into the response path


class TestNeverRaises:
    def test_traversal_category_is_a_clean_miss(self, reg):
        with pytest.raises(ra.SkillNotFound):
            ra.skill_view_payload(reg, "../../etc", "passwd")

    def test_traversal_delete_is_false(self, reg):
        assert ra.skill_delete_payload(reg, "../../etc", "passwd") == {"deleted": False}

    def test_traversal_does_not_escape_root(self, reg, tmp_path):
        # A traversal payload through add must stay inside the registry root.
        reg.add("../../escape", "../../ops", _body(), status=sk.STATUS_PUBLISHED)
        for p in tmp_path.rglob("SKILL.md"):
            assert tmp_path.resolve() in p.resolve().parents

    def test_list_on_fresh_root_does_not_raise(self, tmp_path):
        fresh = sk.SkillRegistry(root=tmp_path / "nonexistent")
        assert ra.skills_list_payload(fresh) == {"skills": []}


# Live FastAPI surface (skips cleanly where fastapi is absent)


class TestFastApiWiring:
    """End-to-end through the real router with a TestClient; fastapi-guarded."""

    @pytest.fixture
    def client(self, reg):
        fastapi = pytest.importorskip("fastapi")
        pytest.importorskip("httpx")  # TestClient transport
        from fastapi.testclient import TestClient

        if ra.router is None:  # pragma: no cover - defensive
            pytest.skip("agent router unavailable")
        app = fastapi.FastAPI()
        assert ra.register(app) is True
        sk.set_skill_registry(reg)
        return TestClient(app)

    def test_list_route(self, client, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        reg.add("rollback", "ops", _body(), status=sk.STATUS_DRAFT)
        r = client.get("/api/agent/skills")
        assert r.status_code == 200
        names = {s["name"] for s in r.json()["skills"]}
        assert names == {"deploy", "rollback"}

    def test_list_exclude_drafts_query(self, client, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        reg.add("rollback", "ops", _body(), status=sk.STATUS_DRAFT)
        r = client.get("/api/agent/skills", params={"include_drafts": "false"})
        assert r.status_code == 200
        names = {s["name"] for s in r.json()["skills"]}
        assert names == {"deploy"}

    def test_view_route_returns_body(self, client, reg):
        reg.add("deploy", "ops", _body(proc="run deploy.sh"), status=sk.STATUS_PUBLISHED)
        r = client.get("/api/agent/skills/ops/deploy")
        assert r.status_code == 200
        assert "run deploy.sh" in r.json()["body"]

    def test_view_missing_is_404(self, client):
        r = client.get("/api/agent/skills/ops/ghost")
        assert r.status_code == 404

    def test_publish_route(self, client, reg):
        reg.add("rollback", "ops", _body(), status=sk.STATUS_DRAFT)
        r = client.post("/api/agent/skills/ops/rollback/publish")
        assert r.status_code == 200
        assert r.json()["status"] == "published"
        assert reg.get("rollback", "ops", draft=True) is None

    def test_publish_without_draft_is_404(self, client, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        r = client.post("/api/agent/skills/ops/deploy/publish")
        assert r.status_code == 404

    def test_delete_route(self, client, reg):
        reg.add("deploy", "ops", _body(), status=sk.STATUS_PUBLISHED)
        r = client.delete("/api/agent/skills/ops/deploy")
        assert r.status_code == 200
        assert r.json() == {"deleted": True}

    def test_delete_missing_is_false(self, client):
        r = client.delete("/api/agent/skills/ops/ghost")
        assert r.status_code == 200
        assert r.json() == {"deleted": False}
