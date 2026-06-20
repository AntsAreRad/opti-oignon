"""
S219 per-fix suite: UD-04 wipe-completeness insertable scope + REV-2.

Covers:
- plugin_reviews: the user_id owner column (fresh schema and the guarded
  additive migration, veilid/peers idiom), identity-carrying add_review,
  the per-user query and cascade delete, and the legacy-NULL semantics
  (unattributable rows are never matched by the per-user predicate).
- routes_plugin_marketplace: AddReviewRequest loses the client-supplied
  author field (REV-2; supersedes the s102 schema pin, deselected in
  pyproject and reasserted here), the endpoint binds author and owner to
  the authenticated identity, ReviewResponse exposes user_id.
- user_data_manager: the honest surface completed against
  ATREST_INVENTORY.md (14 not-covered entries incl. the four added at
  S219), the new retained-by-design class, export format 1.1 (supersedes
  the s142 format pin, deselected in pyproject and reasserted here), and
  the reviews wired into export and cascade delete.
- routes_users: DeleteDataResponse carries the new keys.
- Frontend: the author input, state, and payload removed; type parity.
- Docs: ATREST_INVENTORY.md present and consistent; the roadmap F7
  statuses rolled.

Idiom: importlib isolation with sys.modules pre-seeding (register before
exec_module), per the project convention.
"""

import ast
import importlib.util
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    """Load a module by path with register-before-exec pre-seeding."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


reviews_mod = _load(
    "opti_oignon.plugin_reviews", "opti_oignon/plugin_reviews.py"
)
udm_mod = _load(
    "opti_oignon.user_data_manager", "opti_oignon/user_data_manager.py"
)

REVIEWS_SRC = (ROOT / "opti_oignon" / "plugin_reviews.py").read_text()
MKT_SRC = (
    ROOT / "opti_oignon" / "api" / "routes_plugin_marketplace.py"
).read_text()
USERS_SRC = (ROOT / "opti_oignon" / "api" / "routes_users.py").read_text()
UDM_SRC = (ROOT / "opti_oignon" / "user_data_manager.py").read_text()
SVELTE_PATH = (
    ROOT
    / "frontend"
    / "src"
    / "lib"
    / "components"
    / "settings"
    / "PluginMarketplace.svelte"
)
SVELTE_SRC = SVELTE_PATH.read_text()
TS_API_SRC = (
    ROOT / "frontend" / "src" / "lib" / "api" / "pluginMarketplace.ts"
).read_text()
TS_TYPES_SRC = (ROOT / "frontend" / "src" / "lib" / "types.ts").read_text()
ROADMAP_SRC = (ROOT / "ROADMAP_POST_AUDIT.md").read_text()
INVENTORY_PATH = ROOT / "ATREST_INVENTORY.md"
INVENTORY_SRC = (
    INVENTORY_PATH.read_text() if INVENTORY_PATH.exists() else ""
)

LEGACY_SCHEMA = """
    CREATE TABLE plugin_reviews (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        plugin_name TEXT NOT NULL,
        rating      INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
        title       TEXT NOT NULL DEFAULT '',
        text        TEXT NOT NULL DEFAULT '',
        author      TEXT NOT NULL DEFAULT 'anonymous',
        created_at  REAL NOT NULL DEFAULT 0
    )
"""


def _columns(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [
            r[1]
            for r in conn.execute(
                "PRAGMA table_info(plugin_reviews)"
            ).fetchall()
        ]
    finally:
        conn.close()


def _class_block(src: str, name: str) -> str:
    i = src.index("class " + name)
    j = src.find("\nclass ", i + 1)
    return src[i: j if j != -1 else len(src)]


@pytest.fixture()
def store(tmp_path):
    return reviews_mod.PluginReviewStore(db_path=tmp_path / "reviews.db")


@pytest.fixture()
def legacy_db(tmp_path):
    """A pre-S219 database: no user_id column, one legacy row."""
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    conn.execute(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO plugin_reviews "
        "(plugin_name, rating, author, created_at) "
        "VALUES ('demo', 4, 'old guy', 1.0)"
    )
    conn.commit()
    conn.close()
    return p


# ---------------------------------------------------------------------------
# Store schema and migration
# ---------------------------------------------------------------------------


class TestStoreSchema:
    def test_fresh_schema_has_user_id(self, store, tmp_path):
        assert "user_id" in _columns(tmp_path / "reviews.db")

    def test_user_index_created(self, store, tmp_path):
        conn = sqlite3.connect(tmp_path / "reviews.db")
        try:
            names = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
        finally:
            conn.close()
        assert "idx_reviews_user" in names

    def test_legacy_migration_adds_user_id(self, legacy_db):
        reviews_mod.PluginReviewStore(db_path=legacy_db)
        assert "user_id" in _columns(legacy_db)

    def test_migration_idempotent(self, legacy_db):
        s = reviews_mod.PluginReviewStore(db_path=legacy_db)
        s._init_db()
        assert _columns(legacy_db).count("user_id") == 1

    def test_migration_preserves_legacy_rows(self, legacy_db):
        s = reviews_mod.PluginReviewStore(db_path=legacy_db)
        rows = s.get_reviews("demo", limit=10)
        assert len(rows) == 1
        assert rows[0].author == "old guy"
        assert rows[0].user_id is None


# ---------------------------------------------------------------------------
# Store identity carriage
# ---------------------------------------------------------------------------


class TestStoreIdentity:
    def test_add_review_accepts_user_id(self, store):
        r = store.add_review("p", 5, author="alice", user_id="alice")
        assert r.user_id == "alice"

    def test_add_review_without_user_id_backward_compatible(self, store):
        r = store.add_review("p", 4, author="bob")
        assert r.rating == 4
        assert r.author == "bob"

    def test_add_review_default_user_id_none(self, store):
        r = store.add_review("p", 4, author="bob")
        assert r.user_id is None

    def test_to_dict_carries_user_id(self, store):
        r = store.add_review("p", 5, author="alice", user_id="alice")
        d = r.to_dict()
        assert d["user_id"] == "alice"

    def test_row_mapper_carries_user_id(self, store):
        store.add_review("p", 5, author="alice", user_id="alice")
        rows = store.get_reviews("p", limit=5)
        assert rows[0].user_id == "alice"


# ---------------------------------------------------------------------------
# Per-user query and cascade delete
# ---------------------------------------------------------------------------


class TestPerUserQueryAndCascade:
    def test_get_reviews_for_user_filters(self, store):
        store.add_review("p", 5, author="alice", user_id="alice")
        store.add_review("q", 3, author="alice", user_id="alice")
        store.add_review("p", 2, author="bob", user_id="bob")
        mine = store.get_reviews_for_user("alice")
        assert len(mine) == 2
        assert {r.user_id for r in mine} == {"alice"}

    def test_get_reviews_for_user_excludes_legacy_null(self, legacy_db):
        s = reviews_mod.PluginReviewStore(db_path=legacy_db)
        s.add_review("demo", 5, author="alice", user_id="alice")
        mine = s.get_reviews_for_user("alice")
        assert len(mine) == 1
        assert mine[0].user_id == "alice"

    def test_delete_reviews_for_user_scoped(self, store):
        store.add_review("p", 5, author="alice", user_id="alice")
        store.add_review("q", 4, author="alice", user_id="alice")
        store.add_review("p", 2, author="bob", user_id="bob")
        assert store.delete_reviews_for_user("alice") == 2
        assert store.total_reviews == 1
        assert store.get_reviews("p", limit=5)[0].user_id == "bob"

    def test_delete_leaves_legacy_null_rows(self, legacy_db):
        s = reviews_mod.PluginReviewStore(db_path=legacy_db)
        s.add_review("demo", 5, author="alice", user_id="alice")
        assert s.delete_reviews_for_user("alice") == 1
        assert s.total_reviews == 1
        assert s.get_reviews("demo", limit=5)[0].user_id is None

    def test_delete_returns_zero_for_unknown_user(self, store):
        assert store.delete_reviews_for_user("nobody") == 0


# ---------------------------------------------------------------------------
# SQL discipline on the new store code
# ---------------------------------------------------------------------------


class TestStoreSqlDiscipline:
    def test_migration_uses_literal_sql(self):
        assert "PRAGMA table_info(plugin_reviews)" in REVIEWS_SRC
        assert (
            "ALTER TABLE plugin_reviews ADD COLUMN user_id TEXT"
            in REVIEWS_SRC
        )

    def test_no_fstring_sql_in_reviews(self):
        pattern = re.compile(
            r'f"""?[^"]*(SELECT|INSERT|DELETE|UPDATE|ALTER|CREATE)',
            re.IGNORECASE,
        )
        assert pattern.search(REVIEWS_SRC) is None


# ---------------------------------------------------------------------------
# Request and response models (REV-2)
# ---------------------------------------------------------------------------


class TestRequestModel:
    def test_add_review_request_source_has_no_author(self):
        block = _class_block(MKT_SRC, "AddReviewRequest")
        assert "author" not in block.replace(
            "no client-supplied author field. The author is", ""
        ).replace(
            "derived server-side from the authenticated identity; an author",
            "",
        ).replace("key sent by an older client is ignored by the model.", "")

    def test_add_review_request_schema_reassert(self):
        """Reasserts the s102 schema pin under the REV-2 contract.

        Supersedes tests/test_plugin_marketplace_s102.py::
        TestRoutesSchemas::test_add_review_request_schema (deselected in
        pyproject): the model carries rating/title/text only; the author
        field is gone.
        """
        mkt = pytest.importorskip(
            "opti_oignon.api.routes_plugin_marketplace"
        )
        req = mkt.AddReviewRequest(rating=4, title="Nice")
        assert req.rating == 4
        assert req.title == "Nice"
        assert req.text == ""
        assert not hasattr(req, "author")

    def test_request_ignores_extra_author_key(self):
        mkt = pytest.importorskip(
            "opti_oignon.api.routes_plugin_marketplace"
        )
        req = mkt.AddReviewRequest(rating=5, author="impersonator")
        assert not hasattr(req, "author")

    def test_review_response_exposes_user_id(self):
        mkt = pytest.importorskip(
            "opti_oignon.api.routes_plugin_marketplace"
        )
        assert "user_id" in mkt.ReviewResponse.model_fields
        assert (
            mkt.ReviewResponse.model_fields["user_id"].default is None
        )


# ---------------------------------------------------------------------------
# Route binding (REV-2)
# ---------------------------------------------------------------------------


def _client_with_store(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import opti_oignon.api.deps as deps
    from opti_oignon.api import routes_plugin_marketplace as mkt

    store = reviews_mod.PluginReviewStore(
        db_path=tmp_path / "route_reviews.db"
    )
    sys.modules["opti_oignon.plugin_reviews"].plugin_review_store = store
    proxy = getattr(deps, "plugin_review_store_instance", None)
    if proxy is not None:
        try:
            object.__setattr__(proxy, "_resolved", None)
            object.__setattr__(proxy, "_error", None)
        except Exception:
            pass
    app = FastAPI()
    app.include_router(mkt.router)
    return TestClient(app), store


class TestRouteBinding:
    def test_endpoint_signature_binds_identity(self):
        i = MKT_SRC.index("def add_plugin_review(")
        sig = MKT_SRC[i: MKT_SRC.index(")", i) + 1]
        assert "current_user: dict = Depends(_get_current_user)" in sig

    def test_fallback_identity_defined(self):
        assert "def _get_current_user() -> dict:" in MKT_SRC
        assert '"sub": "local"' in MKT_SRC

    def test_post_binds_author_to_identity(self, tmp_path):
        client, _ = _client_with_store(tmp_path)
        resp = client.post(
            "/api/plugins/demo/reviews",
            json={
                "rating": 5,
                "title": "t",
                "text": "x",
                "author": "Impersonator",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("success") is True
        review = body.get("review") or {}
        assert review.get("author") == "local"

    def test_post_response_carries_user_id(self, tmp_path):
        client, _ = _client_with_store(tmp_path)
        body = client.post(
            "/api/plugins/demo/reviews", json={"rating": 4}
        ).json()
        review = body.get("review") or {}
        assert review.get("user_id") == "local"

    def test_listing_carries_user_id(self, tmp_path):
        client, _ = _client_with_store(tmp_path)
        client.post("/api/plugins/demo/reviews", json={"rating": 4})
        listing = client.get("/api/plugins/demo/reviews").json()
        assert listing["reviews"], "expected one listed review"
        assert listing["reviews"][0].get("user_id") == "local"


# ---------------------------------------------------------------------------
# Wipe surface (UD-04)
# ---------------------------------------------------------------------------


class TestWipeSurface:
    def test_not_covered_has_14_entries(self):
        assert len(udm_mod.WIPE_NOT_COVERED) == 14

    def test_not_covered_contains_new_entries(self):
        joined = " | ".join(udm_mod.WIPE_NOT_COVERED)
        for needle in (
            "projects store",
            "conversation branches store",
            "learned routing store",
            "plugin-owned data stores",
        ):
            assert needle in joined

    def test_not_covered_keeps_s194_compat(self):
        assert (
            "conversations (store not user-scoped)"
            in udm_mod.WIPE_NOT_COVERED
        )
        assert len(udm_mod.WIPE_NOT_COVERED) >= 8

    def test_retained_by_design_tuple(self):
        retained = udm_mod.WIPE_RETAINED_BY_DESIGN
        assert len(retained) == 3
        joined = " | ".join(retained)
        for needle in ("admin audit", "audit chain", "signed audit"):
            assert needle in joined

    def test_reviews_not_listed_as_not_covered(self):
        assert not any(
            "review" in e for e in udm_mod.WIPE_NOT_COVERED
        )


# ---------------------------------------------------------------------------
# Export surface (UD-04 + REV-2)
# ---------------------------------------------------------------------------


def _wire_review_store(tmp_path):
    store = reviews_mod.PluginReviewStore(db_path=tmp_path / "udm.db")
    sys.modules["opti_oignon.plugin_reviews"].plugin_review_store = store
    return store


class TestExportSurface:
    def test_format_version_is_1_1_reassert(self):
        """Reasserts the export format pin at 1.1.

        Supersedes tests/test_s142_multiuser_isolation.py::
        TestUserDataExporter::test_export_format_version (deselected in
        pyproject): 1.1 adds plugin_reviews and retained_by_design.
        """
        data = udm_mod.UserDataExporter().export("u1")
        assert data["export_metadata"]["format_version"] == "1.1"

    def test_export_metadata_carries_retained(self):
        data = udm_mod.UserDataExporter().export("u1")
        assert data["export_metadata"]["retained_by_design"] == list(
            udm_mod.WIPE_RETAINED_BY_DESIGN
        )

    def test_export_has_plugin_reviews_key(self):
        data = udm_mod.UserDataExporter().export("u1")
        assert isinstance(data["plugin_reviews"], list)

    def test_export_includes_user_reviews_end_to_end(self, tmp_path):
        store = _wire_review_store(tmp_path)
        store.add_review("p", 5, author="alice", user_id="alice")
        store.add_review("p", 2, author="bob", user_id="bob")
        data = udm_mod.UserDataExporter().export("alice")
        assert len(data["plugin_reviews"]) == 1
        assert data["plugin_reviews"][0]["user_id"] == "alice"

    def test_export_not_covered_still_present(self):
        data = udm_mod.UserDataExporter().export("u1")
        assert len(data["export_metadata"]["not_covered"]) >= 8


# ---------------------------------------------------------------------------
# Delete surface (UD-04 + REV-2)
# ---------------------------------------------------------------------------


class TestDeleteSurface:
    def test_delete_results_carry_plugin_reviews_count(self, tmp_path):
        _wire_review_store(tmp_path)
        results = udm_mod.UserDataDeleter().delete_all("u1")
        assert results["plugin_reviews"] == 0

    def test_delete_results_carry_retained(self):
        results = udm_mod.UserDataDeleter().delete_all("u1")
        assert results["retained_by_design"] == list(
            udm_mod.WIPE_RETAINED_BY_DESIGN
        )

    def test_delete_cascade_end_to_end(self, tmp_path):
        store = _wire_review_store(tmp_path)
        store.add_review("p", 5, author="alice", user_id="alice")
        store.add_review("q", 4, author="alice", user_id="alice")
        store.add_review("p", 2, author="bob", user_id="bob")
        results = udm_mod.UserDataDeleter().delete_all("alice")
        assert results["plugin_reviews"] == 2
        assert store.total_reviews == 1
        assert store.get_reviews("p", limit=5)[0].user_id == "bob"

    def test_delete_not_covered_parity(self):
        results = udm_mod.UserDataDeleter().delete_all("u1")
        assert (
            "conversations (store not user-scoped)"
            in results["not_covered"]
        )


# ---------------------------------------------------------------------------
# routes_users response model
# ---------------------------------------------------------------------------


class TestRoutesUsersModel:
    def test_delete_response_has_plugin_reviews_field(self):
        block = _class_block(USERS_SRC, "DeleteDataResponse")
        assert "plugin_reviews: int = 0" in block

    def test_delete_response_has_retained_field(self):
        block = _class_block(USERS_SRC, "DeleteDataResponse")
        assert "retained_by_design: list[str] = []" in block


# ---------------------------------------------------------------------------
# Frontend (REV-2)
# ---------------------------------------------------------------------------


class TestFrontend:
    def test_component_has_no_author_state(self):
        assert "newAuthor" not in SVELTE_SRC

    def test_component_payload_has_no_author(self):
        i = SVELTE_SRC.index("async function handleSubmitReview")
        j = SVELTE_SRC.index("function", i + 30)
        assert "author" not in SVELTE_SRC[i:j]

    def test_api_client_drops_author(self):
        i = TS_API_SRC.index("export async function addPluginReview")
        j = TS_API_SRC.index("export async function", i + 10)
        assert "author" not in TS_API_SRC[i:j]

    def test_types_review_carries_user_id(self):
        i = TS_TYPES_SRC.index("export interface PluginReview {")
        j = TS_TYPES_SRC.index("}", i)
        assert "user_id?: string | null;" in TS_TYPES_SRC[i:j]

    def test_component_tag_block_balance(self):
        for tag in ("script", "style"):
            opened = len(re.findall(rf"<{tag}[ >]", SVELTE_SRC))
            closed = SVELTE_SRC.count(f"</{tag}>")
            assert opened == closed, tag
        for blk in ("if", "each", "await", "key"):
            opened = len(re.findall(r"\{#" + blk + r"[ }]", SVELTE_SRC))
            closed = len(re.findall(r"\{/" + blk + r"\}", SVELTE_SRC))
            assert opened == closed, blk


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class TestDocs:
    def test_inventory_exists(self):
        assert INVENTORY_PATH.exists()

    def test_inventory_declares_classes(self):
        for needle in (
            "pending-scoping",
            "retained-by-design",
            "device-local",
            "plugin-owned",
        ):
            assert needle in INVENTORY_SRC

    def test_inventory_lists_new_stores(self):
        for needle in (
            "plugin reviews",
            "projects",
            "conversation branches",
            "learned routing",
        ):
            assert needle in INVENTORY_SRC

    def test_inventory_bk06_candidates(self):
        for needle in (
            "pipelines_custom.yaml",
            "cache.yaml",
            "humanizer config",
            "fine_tune config",
            "memory settings",
            "projects settings",
        ):
            assert needle in INVENTORY_SRC

    def test_inventory_changefeed_note(self):
        assert "tombstones" in INVENTORY_SRC
        assert "journal" in INVENTORY_SRC

    def test_roadmap_f7_rolled(self):
        assert "ADVANCED at S219" in ROADMAP_SRC
        assert "STAGED for S220 against ATREST_INVENTORY.md" in ROADMAP_SRC


# ---------------------------------------------------------------------------
# AST validity
# ---------------------------------------------------------------------------


TOUCHED_PY = [
    "opti_oignon/plugin_reviews.py",
    "opti_oignon/api/routes_plugin_marketplace.py",
    "opti_oignon/api/routes_users.py",
    "opti_oignon/user_data_manager.py",
]


class TestAstValidity:
    @pytest.mark.parametrize("rel", TOUCHED_PY)
    def test_touched_python_parses(self, rel):
        ast.parse((ROOT / rel).read_text())

    def test_suite_parses_itself(self):
        ast.parse(Path(__file__).read_text())
