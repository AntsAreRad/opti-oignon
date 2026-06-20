#!/usr/bin/env python3
"""S248 -- the Notes feature's N.2 core UI + the N.3 selection-action panel
(the SvelteKit lot), the container-provable half of a labelled Playwright/runbook
lot (the s238..s241 precedent).

S248 lands the frontend the user drives over the already-built Notes backend
(N.1 data layer S243, N.4 manage_notes S244, N.2 route S245, N.3 surface S246,
N.3 route S247): two typed API clients (api/notes.ts over /api/notes,
api/noteActions.ts over /api/notes/actions/run), a notes store (stores/notes.ts),
the NotesPanel core UI (master-detail: list, search, tags, editor), the
NoteActionPanel selection-action panel (insert/append), and the /notes route
(page + AppShell layout + a sidebar nav entry), each new Svelte component
recorded in FRONTEND_REDESIGN_SPEC.md. The live Playwright run is host-assured
(NOTES_UI_E2E_S248.md); this suite proves the container-provable artifacts.

node_modules is absent here, so the frontend is checked by file content, a
structural Svelte block-balance pass, and the --oo-* token discipline (no raw
hex) -- the s174 idiom -- not by a browser. Seven families:

 1. The TS clients and the store -- existence, the base-client wiring, the
    interfaces, the endpoints, the store's writables / derived / actions.
 2. The components -- NotesPanel and NoteActionPanel: the ds-primitive imports,
    the client/store/model wiring, the behaviour surface, the --oo-* tokens (no
    raw hex), and Svelte block balance.
 3. The route and nav -- the /notes page renders NotesPanel, the layout wraps
    AppShell, the sidebar carries the /notes entry.
 4. The spec registration -- the NotesPanel.svelte and NoteActionPanel.svelte
    NEW/S248 rows (the s174 registration regex), which also keeps the Theme 1
    cartography invariant green.
 5. The runbook -- NOTES_UI_E2E_S248.md: existence, status (host-assured,
    findings-not-fixes, never-simulated), the container-vs-host split, the
    required sections, the companions, the version held, and pure ASCII.
 6. The seams the UI is the client of -- source pins on the premises (the N.2
    route and its five endpoints, the N.3 route and its single POST, the Note
    schemas, the base HTTP client, the ds primitives), so a later edit that
    removes a premise turns this suite red instead of letting the UI rot. Green
    on the pristine S247 tree by design.
 7. ASCII / structure -- the new TS and Svelte files are pure ASCII and
    block-balanced, and this suite parses.

Red-before discipline: on the pristine S247 tree (no frontend source, no spec
rows, no runbook) every family-1..5 and family-7 structural pin FAILS -- the read
helpers return empty strings so absence is a failure, never a collection error --
while every family-6 seam pin passes (it pins pre-existing premises this lot
relies on). Document pins read through a whitespace-flattening helper (the
s221/s238 lesson) so reflow cannot break them; source pins stay raw. This suite
imports no package code, so no ollama chain is touched.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend" / "src"

# New frontend source landed by S248.
NOTES_CLIENT = FE / "lib" / "api" / "notes.ts"
ACTIONS_CLIENT = FE / "lib" / "api" / "noteActions.ts"
NOTES_STORE = FE / "lib" / "stores" / "notes.ts"
NOTES_PANEL = FE / "lib" / "components" / "panels" / "NotesPanel.svelte"
ACTION_PANEL = FE / "lib" / "components" / "panels" / "NoteActionPanel.svelte"
NOTES_PAGE = FE / "routes" / "notes" / "+page.svelte"
NOTES_LAYOUT = FE / "routes" / "notes" / "+layout.svelte"
SIDEBAR = FE / "lib" / "components" / "layout" / "Sidebar.svelte"

NEW_TS = (NOTES_CLIENT, ACTIONS_CLIENT, NOTES_STORE)
NEW_SVELTE = (NOTES_PANEL, ACTION_PANEL, NOTES_PAGE, NOTES_LAYOUT)

# Docs.
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
RUNBOOK = ROOT / "NOTES_UI_E2E_S248.md"

# Seam premises (pre-existing; green on the pristine S247 tree by design).
ROUTES_NOTES = ROOT / "opti_oignon" / "api" / "routes_notes.py"
ROUTES_NOTE_ACTIONS = ROOT / "opti_oignon" / "api" / "routes_note_actions.py"
SCHEMAS = ROOT / "opti_oignon" / "api" / "schemas.py"
BASE_CLIENT = FE / "lib" / "api" / "client.ts"
DS_INDEX = FE / "lib" / "ds" / "index.ts"


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flat(text: str) -> str:
    """Collapse whitespace runs to single spaces (reflow-immune doc pins)."""
    return re.sub(r"\s+", " ", text)


def _strip_token_fallbacks(text: str) -> str:
    """Remove var(--oo-..., #fallback) occurrences so only RAW hex remains."""
    return re.sub(r"var\(--oo-[^)]*\)", "", text)


def _has_raw_hex(text: str) -> bool:
    """True if a hardcoded hex colour survives outside a --oo-* fallback."""
    return re.search(r"#[0-9a-fA-F]{3,8}\b", _strip_token_fallbacks(text)) is not None


def _block_balanced(text: str) -> bool:
    """Svelte block + script/style balance (a lightweight structural pass)."""
    for blk in ("if", "each", "await", "key"):
        if text.count("{#" + blk) != text.count("{/" + blk + "}"):
            return False
    if text.count("<script") != text.count("</script>"):
        return False
    if text.count("<style") != text.count("</style>"):
        return False
    return True


# ---------------------------------------------------------------------------
# Family 1 -- the TS clients and the store
# ---------------------------------------------------------------------------


class TestNotesClient:
    def test_exists(self):
        assert NOTES_CLIENT.exists()

    def test_imports_base_client(self):
        s = _read(NOTES_CLIENT)
        assert "from './client'" in s
        for fn in ("apiGet", "apiPost", "apiPatch", "apiDelete"):
            assert fn in s, fn

    def test_interfaces(self):
        s = _read(NOTES_CLIENT)
        for i in ("NoteRecord", "NoteCreate", "NoteUpdate"):
            assert f"interface {i}" in s, i

    def test_functions_over_api_notes(self):
        s = _read(NOTES_CLIENT)
        for fn in ("listNotes", "getNote", "createNote", "updateNote", "deleteNote"):
            assert fn in s, fn
        assert "'/api/notes'" in s

    def test_body_helpers(self):
        s = _read(NOTES_CLIENT)
        assert "encodeNoteBody" in s
        assert "decodeNoteBody" in s


class TestActionsClient:
    def test_exists(self):
        assert ACTIONS_CLIENT.exists()

    def test_endpoint_and_runner(self):
        s = _read(ACTIONS_CLIENT)
        assert "runNoteAction" in s
        assert "'/api/notes/actions/run'" in s

    def test_six_actions_and_table(self):
        s = _read(ACTIONS_CLIENT)
        for kind in (
            "fact_check",
            "fact_check_web",
            "develop",
            "summarize",
            "rewrite",
            "make_checklist",
        ):
            assert f"'{kind}'" in s, kind
        assert "NOTE_ACTIONS" in s

    def test_result_interface(self):
        s = _read(ACTIONS_CLIENT)
        assert "interface NoteActionResult" in s
        for field in ("action", "ok", "text", "refused", "reason"):
            assert field in s, field


class TestNotesStore:
    def test_exists(self):
        assert NOTES_STORE.exists()

    def test_imports_client(self):
        assert "from '$lib/api/notes'" in _read(NOTES_STORE)

    def test_stores_and_derived(self):
        s = _read(NOTES_STORE)
        for store in ("notes", "activeNote", "filteredNotes", "loading", "error", "search"):
            assert store in s, store
        assert "writable" in s
        assert "derived" in s

    def test_actions(self):
        s = _read(NOTES_STORE)
        for action in (
            "loadNotes",
            "createNote",
            "saveNote",
            "togglePin",
            "removeNote",
            "selectNote",
        ):
            assert action in s, action


# ---------------------------------------------------------------------------
# Family 2 -- the components
# ---------------------------------------------------------------------------


class TestNotesPanel:
    def test_exists(self):
        assert NOTES_PANEL.exists()

    def test_imports_ds_primitives(self):
        s = _read(NOTES_PANEL)
        assert "from '$lib/ds'" in s
        for prim in ("Button", "Card", "Input", "Icon", "EmptyState", "InlineError", "Modal"):
            assert prim in s, prim

    def test_wires_store_and_body_helpers(self):
        s = _read(NOTES_PANEL)
        assert "from '$lib/stores/notes'" in s
        for fn in ("loadNotes", "createNote", "saveNote", "removeNote", "selectNote"):
            assert fn in s, fn
        assert "from '$lib/api/notes'" in s
        assert "encodeNoteBody" in s
        assert "decodeNoteBody" in s

    def test_hosts_action_panel(self):
        s = _read(NOTES_PANEL)
        assert "import NoteActionPanel from './NoteActionPanel.svelte'" in s
        assert "<NoteActionPanel" in s

    def test_behaviour_surface(self):
        s = _read(NOTES_PANEL)
        for needle in (
            "function newNote",
            "async function save",
            "function askDelete",
            "bind:this={bodyEl}",
            "function updateSelection",
            "insertIntoBody",
            "appendToBody",
        ):
            assert needle in s, needle

    def test_no_raw_hex(self):
        raw = _read(NOTES_PANEL)
        assert raw, "NotesPanel.svelte absent"
        assert not _has_raw_hex(raw)

    def test_block_balanced(self):
        raw = _read(NOTES_PANEL)
        assert raw, "NotesPanel.svelte absent"
        assert _block_balanced(raw)


class TestNoteActionPanel:
    def test_exists(self):
        assert ACTION_PANEL.exists()

    def test_imports_ds_primitives(self):
        s = _read(ACTION_PANEL)
        assert "from '$lib/ds'" in s
        for prim in ("Button", "Card", "Icon", "InlineError"):
            assert prim in s, prim

    def test_wires_actions_client(self):
        s = _read(ACTION_PANEL)
        assert "from '$lib/api/noteActions'" in s
        assert "runNoteAction" in s
        assert "NOTE_ACTIONS" in s

    def test_wires_model(self):
        s = _read(ACTION_PANEL)
        assert "from '$lib/stores/chatOptions'" in s
        for store in ("selectedModel", "effectiveModel", "loadOptions"):
            assert store in s, store

    def test_behaviour_surface(self):
        s = _read(ACTION_PANEL)
        for needle in (
            "async function run",
            "function insert",
            "function append",
            "onInsert",
            "onAppend",
            "result.refused",
            "result.ok",
        ):
            assert needle in s, needle

    def test_no_raw_hex(self):
        raw = _read(ACTION_PANEL)
        assert raw, "NoteActionPanel.svelte absent"
        assert not _has_raw_hex(raw)

    def test_block_balanced(self):
        raw = _read(ACTION_PANEL)
        assert raw, "NoteActionPanel.svelte absent"
        assert _block_balanced(raw)


# ---------------------------------------------------------------------------
# Family 3 -- the route and the nav entry
# ---------------------------------------------------------------------------


class TestRouteAndNav:
    def test_page_renders_panel(self):
        s = _read(NOTES_PAGE)
        assert NOTES_PAGE.exists()
        assert "NotesPanel" in s

    def test_layout_wraps_appshell(self):
        s = _read(NOTES_LAYOUT)
        assert NOTES_LAYOUT.exists()
        assert "AppShell" in s

    def test_sidebar_nav_entry(self):
        s = _read(SIDEBAR)
        assert "'/notes'" in s
        assert "label: 'Notes'" in s


# ---------------------------------------------------------------------------
# Family 4 -- the spec registration (cartography invariant)
# ---------------------------------------------------------------------------


class TestSpecRegistration:
    def test_notes_panel_registered(self):
        spec = _read(SPEC)
        assert re.search(r"NotesPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S248", spec) is not None

    def test_action_panel_registered(self):
        spec = _read(SPEC)
        assert re.search(r"NoteActionPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S248", spec) is not None


# ---------------------------------------------------------------------------
# Family 5 -- the host-assured Playwright runbook
# ---------------------------------------------------------------------------


class TestRunbook:
    def test_exists_and_titled(self):
        text = _read(RUNBOOK)
        assert text.startswith("# NOTES_UI_E2E_S248")
        assert "Playwright E2E runbook" in text
        assert len(text) > 4000

    def test_status_and_discipline(self):
        text = _flat(_read(RUNBOOK))
        assert "written at S248" in text
        assert "host-assured" in text
        assert "produces findings, not fixes" in text
        assert "never simulated in the container" in text

    def test_container_vs_host_split(self):
        text = _flat(_read(RUNBOOK))
        assert "Container-provable" in text
        assert "Host-assured" in text
        assert "held at 3.11.0" in text

    def test_required_sections(self):
        text = _flat(_read(RUNBOOK))
        for needle in (
            "Preflight",
            "Notes CRUD over /api/notes",
            "selection-action panel",
            "Daily-only web gate",
            "Untrusted-context",
            "Findings register",
            "Routing",
        ):
            assert needle in text, needle

    def test_companions_named(self):
        text = _flat(_read(RUNBOOK))
        for needle in (
            "NOTES_FEATURE_ROADMAP.md",
            "FRONTEND_REDESIGN_SPEC.md",
            "routes_notes.py",
            "routes_note_actions.py",
        ):
            assert needle in text, needle

    def test_auth_core_edit_free(self):
        text = _flat(_read(RUNBOOK))
        assert "auth.py, auth_2fa.py" in text
        assert "emergency_stop.py" in text
        assert "edit-free" in text

    def test_pure_ascii_no_decoration(self):
        raw = _read(RUNBOOK)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


# ---------------------------------------------------------------------------
# Family 6 -- the seams the UI is the client of (green on pristine by design)
# ---------------------------------------------------------------------------


class TestSeamN2Route:
    def test_router_and_endpoints(self):
        src = _read(ROUTES_NOTES)
        assert "notes_router" in src
        assert 'prefix="/api/notes"' in src
        assert "@notes_router.get" in src
        assert "@notes_router.post" in src
        assert "@notes_router.patch" in src
        assert "@notes_router.delete" in src
        assert "/{note_id}" in src

    def test_request_schemas_used(self):
        src = _read(ROUTES_NOTES)
        for name in ("NoteCreateRequest", "NoteSchema", "NoteUpdateRequest"):
            assert name in src, name


class TestSeamN3Route:
    def test_router_and_run(self):
        src = _read(ROUTES_NOTE_ACTIONS)
        assert "note_actions_router" in src
        assert 'prefix="/api/notes/actions"' in src
        assert '"/run"' in src
        assert "def run_note_action" in src
        assert "NoteActionResultSchema" in src

    def test_not_a_model_tool(self):
        src = _read(ROUTES_NOTE_ACTIONS)
        assert "ToolSchema(" not in src
        assert "register_tool" not in src


class TestSeamSchemas:
    def test_note_schema_fields(self):
        src = _read(SCHEMAS)
        assert "class NoteSchema" in src
        for field in ("body_crdt_b64", "tags", "pinned"):
            assert field in src, field

    def test_action_result_fields(self):
        src = _read(SCHEMAS)
        assert "class NoteActionResultSchema" in src
        for field in ("ok", "refused", "reason"):
            assert field in src, field


class TestSeamBaseClient:
    def test_http_methods(self):
        src = _read(BASE_CLIENT)
        for fn in ("apiGet", "apiPost", "apiPatch", "apiDelete"):
            assert f"function {fn}" in src, fn


class TestSeamDsPrimitives:
    def test_ds_exports(self):
        src = _read(DS_INDEX)
        for prim in ("Button", "Card", "Input", "Icon", "Modal", "EmptyState", "InlineError"):
            assert prim in src, prim


# ---------------------------------------------------------------------------
# Family 7 -- ASCII / structure
# ---------------------------------------------------------------------------


class TestAsciiAndStructure:
    def test_new_ts_pure_ascii(self):
        for path in NEW_TS:
            raw = _read(path)
            assert raw != "", path.name
            assert all(ord(c) < 128 for c in raw), path.name

    def test_new_svelte_pure_ascii_and_balanced(self):
        for path in NEW_SVELTE:
            raw = _read(path)
            assert raw != "", path.name
            assert all(ord(c) < 128 for c in raw), path.name
            assert _block_balanced(raw), path.name

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)
