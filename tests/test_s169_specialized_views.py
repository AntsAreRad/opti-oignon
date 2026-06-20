"""
S169 -- Specialized Views test suite.

Validates the four S169 goals (spec 9.3 / 9.4 / 9.5 / 11.5 / 11.7 / 12.4):

1. Projects routes redesign: /projects on AppShell, ProjectDetail promoted to
   /projects/[id], ProjectList + ProjectDetail on ds primitives, sidebar
   projects context (search + Starred/All/Archived).
2. Benchmark redesign: /benchmark on AppShell, BenchmarkV2Panel decomposed into
   seven self-contained section components + shared format helpers + globalized
   styles, a per-run drawer-right, sidebar runs grouped by recency.
3. System Status: /health rebranded, six module groups, cache + alerts on
   primitives, sidebar/nav label.
4. Chat surface: the three orphan indicators reintegrated, a standalone
   tool-call approval drawer-right, French stripped from the touched files.

These are file-content assertions (the repo convention for frontend checks).
The TestBenchmarkSupersede class deliberately re-asserts the *intent* of
test_benchmark_v2_s88::test_benchmarkv2_panel_structure (deselected because the
decomposition removed the monolith's <style>/bv2-tabs/bv2-radar) against the new
decomposed structure.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend" / "src"

# --- Projects (Phase 1) ---
PROJ_LAYOUT = FE / "routes" / "projects" / "+layout.svelte"
PROJ_PAGE = FE / "routes" / "projects" / "+page.svelte"
PROJ_ID_PAGE = FE / "routes" / "projects" / "[id]" / "+page.svelte"
PROJECT_LIST = FE / "lib" / "components" / "panels" / "ProjectList.svelte"
PROJECT_DETAIL = FE / "lib" / "components" / "panels" / "ProjectDetail.svelte"

# --- System Status (Phase 2) ---
HEALTH_PAGE = FE / "routes" / "health" / "+page.svelte"
HEALTH_LAYOUT = FE / "routes" / "health" / "+layout.svelte"
HEALTH_DASHBOARD = FE / "lib" / "components" / "health" / "HealthDashboard.svelte"
CACHE_MANAGER = FE / "lib" / "components" / "health" / "CacheManager.svelte"
SIDEBAR_NAV = FE / "lib" / "components" / "layout" / "Sidebar.svelte"

# --- Benchmark (Phase 3) ---
BENCH_DIR = FE / "lib" / "components" / "panels" / "benchmark"
BENCH_PANEL = FE / "lib" / "components" / "panels" / "BenchmarkV2Panel.svelte"
BENCH_LAYOUT = FE / "routes" / "benchmark" / "+layout.svelte"
BENCH_PAGE = FE / "routes" / "benchmark" / "+page.svelte"
BENCH_FORMAT = BENCH_DIR / "format.ts"
BENCH_CSS = BENCH_DIR / "benchmark.css"
BENCH_DRAWER = BENCH_DIR / "BenchmarkRunDrawer.svelte"

BENCH_SECTIONS = [
    "BenchmarkRunSection",
    "BenchmarkLeaderboard",
    "BenchmarkHeadToHead",
    "BenchmarkTrends",
    "BenchmarkCompareSection",
    "BenchmarkHistorySection",
    "BenchmarkProfiles",
]
TAB_IDS = ["run", "leaderboard", "h2h", "trends", "compare", "history", "profiles"]
SHARED_HELPERS = [
    "scoreColor", "pct", "formatDuration", "formatDate", "radarPoints",
    "radarLabels", "radarColors", "radarLabelPos", "trendPath", "roleLabel",
    "winnerClass", "formatCooldown", "formatEventTime",
]

# --- Chat surface (Phase 4) ---
CHAT_DIR = FE / "lib" / "components" / "chat"
CHAT_MESSAGE = CHAT_DIR / "ChatMessage.svelte"
TOOLCALL_DISPLAY = CHAT_DIR / "ToolCallDisplay.svelte"
CORRECTION = CHAT_DIR / "CorrectionIndicator.svelte"
ROUTING = CHAT_DIR / "RoutingIndicator.svelte"
APPROVAL_DRAWER = CHAT_DIR / "ToolCallApprovalDrawer.svelte"
CHAT_LAYOUT = FE / "routes" / "chat" / "+layout.svelte"

# --- Shared (sidebar context list, spec) ---
SIDEBAR_CTX = FE / "lib" / "components" / "sidebar" / "SectionContextList.svelte"
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"

# Accented Latin letters that flag French text leaking into code/UI.
_FRENCH = re.compile(r"[\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef"
                     r"\u00f4\u00f9\u00fb\u00fc\u00e7\u00c0\u00c9\u00c8\u00ca\u00c7]")
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
_VAR_FALLBACK = re.compile(r"var\(\s*--oo-[a-z0-9-]+\s*,\s*#[0-9a-fA-F]{3,6}\s*\)")

# Accent-free French comment words to confirm removed from the reintegrated
# indicators (the accented regex would not catch these).
_FRENCH_WORDS = re.compile(
    r"\b(Indicateur|Affiche|Cliquable|expander|generique|formate|montrant|"
    r"selectionne|modele|Ligne compacte|Details expandable)\b"
)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _raw_hex_violations(text: str) -> list[str]:
    """6-digit hex outside a var(--oo-*, #fallback) and outside comments."""
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        st = line.strip()
        if st.startswith("//") or st.startswith("<!--") or st.startswith("*"):
            continue
        stripped = re.sub(r"&#\d+;", "", st)
        if _HEX.search(stripped) and "var(--oo-" not in stripped:
            out.append(f"line {i}: {st[:70]}")
    return out


# =====================================================================
# 1. Projects routes redesign (Phase 1 / spec 9.3)
# =====================================================================

class TestProjectsRoutes:
    def test_projects_layout_exists_on_appshell(self):
        assert PROJ_LAYOUT.exists()
        assert "AppShell" in _read(PROJ_LAYOUT)

    def test_projects_index_renders_list(self):
        src = _read(PROJ_PAGE)
        assert "ProjectList" in src

    def test_projects_index_reads_new_query(self):
        # ?new=1 opens the create flow.
        assert "new" in _read(PROJ_PAGE)

    def test_project_detail_route_exists(self):
        assert PROJ_ID_PAGE.exists()

    def test_project_detail_route_promotes_detail(self):
        src = _read(PROJ_ID_PAGE)
        assert "ProjectDetail" in src
        assert "selectProject" in src

    def test_project_detail_route_back_to_projects(self):
        assert "/projects" in _read(PROJ_ID_PAGE)

    def test_project_list_on_primitives(self):
        src = _read(PROJECT_LIST)
        assert "$lib/ds" in src
        for prim in ("Button", "Select", "Modal"):
            assert prim in src, f"ProjectList missing {prim}"

    def test_project_list_action_menu(self):
        # Secondary per-item action menu.
        assert "openMenuId" in _read(PROJECT_LIST)

    def test_project_list_navigates_by_href(self):
        # Cards/rows navigate via real links, not a select dispatch.
        assert '/projects/' in _read(PROJECT_LIST)

    def test_project_detail_uses_tabs(self):
        src = _read(PROJECT_DETAIL)
        assert "Tabs" in src and "TabItem" in src

    def test_project_detail_uses_form_primitives(self):
        src = _read(PROJECT_DETAIL)
        for prim in ("Button", "Input", "Switch"):
            assert prim in src, f"ProjectDetail missing {prim}"

    def test_sidebar_projects_search_and_groups(self):
        src = _read(SIDEBAR_CTX)
        assert "projectQuery" in src
        assert "isStarred" in src and "isArchived" in src
        assert "/projects?new=1" in src


# =====================================================================
# 2. System Status (Phase 2 / spec 9.4)
# =====================================================================

class TestSystemStatus:
    def test_health_page_titled_system_status(self):
        assert "System Status" in _read(HEALTH_PAGE)

    def test_health_route_id_unchanged(self):
        # The route stays /health for compatibility.
        assert HEALTH_PAGE.exists() and HEALTH_LAYOUT.exists()

    def test_health_layout_header_renamed(self):
        assert "System Status" in _read(HEALTH_LAYOUT)

    def test_sidebar_nav_label_renamed(self):
        assert "System Status" in _read(SIDEBAR_NAV)

    def test_sidebar_context_label_renamed(self):
        assert "System Status" in _read(SIDEBAR_CTX)

    def test_dashboard_six_module_groups(self):
        src = _read(HEALTH_DASHBOARD)
        assert "MODULE_GROUPS" in src
        for label in ("Backend", "Inference & models", "RAG & memory",
                      "Plugins & tools", "Network", "Security"):
            assert label in src, f"missing module group: {label}"

    def test_dashboard_uses_health_and_network_apis(self):
        src = _read(HEALTH_DASHBOARD)
        assert "getHealthDashboard" in src
        assert "getNetworkStatus" in src

    def test_dashboard_recent_alerts(self):
        assert "Recent alerts" in _read(HEALTH_DASHBOARD)

    def test_dashboard_on_primitives(self):
        src = _read(HEALTH_DASHBOARD)
        assert "$lib/ds" in src
        assert "Card" in src

    def test_dashboard_no_legacy_btn_token(self):
        # The old --oo-btn-primary-fg legacy token is gone.
        assert "--oo-btn-primary" not in _read(HEALTH_DASHBOARD)

    def test_cache_manager_on_primitives(self):
        src = _read(CACHE_MANAGER)
        assert "$lib/ds" in src
        for prim in ("Card", "Button", "Select"):
            assert prim in src, f"CacheManager missing {prim}"

    def test_health_page_includes_cache(self):
        assert "CacheManager" in _read(HEALTH_PAGE)


# =====================================================================
# 3. Benchmark decomposition + redesign (Phase 3 / spec 9.5)
# =====================================================================

class TestBenchmarkDecomposition:
    def test_section_files_exist(self):
        for name in BENCH_SECTIONS:
            assert (BENCH_DIR / f"{name}.svelte").exists(), f"missing {name}"

    def test_shared_format_module_exists(self):
        assert BENCH_FORMAT.exists()

    def test_shared_format_exports_helpers(self):
        src = _read(BENCH_FORMAT)
        for h in SHARED_HELPERS:
            assert f"export function {h}" in src or f"export const {h}" in src, h

    def test_globalized_css_exists(self):
        assert BENCH_CSS.exists()
        assert "bv2-" in _read(BENCH_CSS)

    def test_panel_is_thin_orchestrator(self):
        # The 2101-line monolith is reduced to a small orchestrator.
        assert BENCH_PANEL.stat().st_size < 4000
        assert len(_read(BENCH_PANEL).split("\n")) < 120

    def test_panel_imports_all_sections(self):
        src = _read(BENCH_PANEL)
        for name in BENCH_SECTIONS:
            assert name in src, f"panel does not import {name}"

    def test_panel_imports_global_css(self):
        assert "benchmark.css" in _read(BENCH_PANEL)

    def test_panel_uses_tabs_primitive(self):
        assert "Tabs" in _read(BENCH_PANEL)

    def test_panel_preserves_tab_ids(self):
        src = _read(BENCH_PANEL)
        for tab in TAB_IDS:
            assert f"'{tab}'" in src, f"missing tab id {tab}"

    def test_sections_self_load(self):
        # Each section owns its data loading (onMount).
        for name in BENCH_SECTIONS:
            assert "onMount" in _read(BENCH_DIR / f"{name}.svelte"), name

    def test_sections_import_shared_helpers(self):
        # The sections reuse the shared module rather than duplicating helpers.
        users = [
            "BenchmarkRunSection", "BenchmarkLeaderboard", "BenchmarkHeadToHead",
            "BenchmarkTrends", "BenchmarkCompareSection", "BenchmarkHistorySection",
        ]
        for name in users:
            assert "./format" in _read(BENCH_DIR / f"{name}.svelte"), name

    def test_api_surface_unchanged(self):
        # The decomposition still drives the same benchmark API.
        run = _read(BENCH_DIR / "BenchmarkRunSection.svelte")
        assert "$lib/api/benchmarkV2" in run
        assert "startRun" in run and "pollUntilDone" in run

    def test_benchmark_route_on_appshell(self):
        assert BENCH_LAYOUT.exists()
        src = _read(BENCH_LAYOUT)
        assert "AppShell" in src
        assert "Benchmark" in src

    def test_benchmark_page_renders_dashboard_and_drawer(self):
        src = _read(BENCH_PAGE)
        assert "BenchmarkPage" in src
        assert "BenchmarkRunDrawer" in src

    def test_per_run_detail_is_drawer_right(self):
        src = _read(BENCH_DRAWER)
        assert "drawer-right" in src
        assert "Modal" in src
        # Sourced from existing history, no new endpoint.
        assert "getHistory" in src

    def test_benchmark_page_reads_run_query(self):
        assert "run" in _read(BENCH_PAGE)

    def test_sidebar_runs_grouped_by_recency(self):
        src = _read(SIDEBAR_CTX)
        for grp in ("Today", "This week", "All time"):
            assert grp in src, f"missing run group {grp}"
        assert "getHistory" in src

    def test_sidebar_run_links_to_query(self):
        assert "/benchmark?run=" in _read(SIDEBAR_CTX)


class TestBenchmarkSupersede:
    """Re-assert the intent of the deselected monolith structure test."""

    def test_panel_no_longer_has_scoped_style(self):
        # bv2 styles now live in the global stylesheet, not the panel.
        assert "<style>" not in _read(BENCH_PANEL)

    def test_bv2_radar_moved_to_run_section(self):
        # The radar markup moved out of the panel into the Run section.
        assert "bv2-radar" not in _read(BENCH_PANEL)
        assert "bv2-radar" in _read(BENCH_DIR / "BenchmarkRunSection.svelte")

    def test_bv2_panel_class_retained(self):
        assert "bv2-panel" in _read(BENCH_PANEL)


# =====================================================================
# 4. Chat surface (Phase 4 / spec 11.5)
# =====================================================================

class TestChatSurface:
    def test_correction_indicator_reintegrated(self):
        assert "CorrectionIndicator" in _read(CHAT_MESSAGE)

    def test_routing_indicator_reintegrated(self):
        assert "RoutingIndicator" in _read(CHAT_MESSAGE)

    def test_correction_bound_to_message(self):
        # Driven by the typed message.correction field.
        assert "message.correction" in _read(CHAT_MESSAGE)

    def test_routing_has_fallback_source(self):
        # Prop wins, else the untyped routing_reason field on the message.
        src = _read(CHAT_MESSAGE)
        assert "routingReason" in src
        assert "routing_reason" in src

    def test_plugin_badge_reintegrated_on_tool_rows(self):
        assert "PluginPermissionBadge" in _read(TOOLCALL_DISPLAY)

    def test_plugin_badge_conditional_on_permissions(self):
        src = _read(TOOLCALL_DISPLAY)
        assert "pluginPerms" in src

    def test_approval_drawer_exists(self):
        assert APPROVAL_DRAWER.exists()

    def test_approval_drawer_is_drawer_right(self):
        src = _read(APPROVAL_DRAWER)
        assert "Modal" in src
        assert "drawer-right" in src

    def test_approval_drawer_uses_api(self):
        src = _read(APPROVAL_DRAWER)
        assert "getPendingApprovals" in src
        assert "approveToolCall" in src and "denyToolCall" in src

    def test_approval_drawer_mounted_in_chat(self):
        assert "ToolCallApprovalDrawer" in _read(CHAT_LAYOUT)

    def test_inline_approval_card_preserved(self):
        # The original inline card is kept (its s128 tests still apply).
        assert (CHAT_DIR / "ToolCallApproval.svelte").exists()

    def test_indicators_french_removed(self):
        for p in (CORRECTION, ROUTING):
            src = _read(p)
            assert not _FRENCH.search(src), f"accented French in {p.name}"
            assert not _FRENCH_WORDS.search(src), f"French words in {p.name}"


# =====================================================================
# 5. Cross-cutting conventions on new/touched files
# =====================================================================

class TestConventions:
    NEW_PURE_TOKEN = [APPROVAL_DRAWER, BENCH_DRAWER, HEALTH_DASHBOARD, CACHE_MANAGER]
    ALL_NEW = [
        APPROVAL_DRAWER, BENCH_DRAWER, BENCH_PANEL, BENCH_LAYOUT, BENCH_PAGE,
        HEALTH_DASHBOARD, CACHE_MANAGER, HEALTH_PAGE,
        PROJ_LAYOUT, PROJ_PAGE, PROJ_ID_PAGE,
    ] + [BENCH_DIR / f"{n}.svelte" for n in BENCH_SECTIONS]

    def test_no_accented_french_in_new_files(self):
        for p in self.ALL_NEW:
            assert not _FRENCH.search(_read(p)), f"accented French in {p.name}"

    def test_new_pure_token_files_have_no_raw_hex(self):
        for p in self.NEW_PURE_TOKEN:
            violations = _raw_hex_violations(_read(p))
            assert not violations, f"raw hex in {p.name}:\n" + "\n".join(violations)

    def test_no_legacy_tokens_in_new_files(self):
        for p in self.ALL_NEW:
            src = _read(p)
            assert "--oo-btn-primary-" not in src, p.name
            assert "--oo-input-" not in src, p.name

    def test_sections_have_no_scoped_style(self):
        # Section components rely on the shared global stylesheet.
        for name in BENCH_SECTIONS:
            assert "<style>" not in _read(BENCH_DIR / f"{name}.svelte"), name

    def test_format_module_is_pure(self):
        # No Svelte / DOM in the shared helper module.
        src = _read(BENCH_FORMAT)
        assert "<script" not in src
        assert "onMount" not in src

    def test_new_components_registered_in_spec(self):
        text = _read(SPEC)
        for name in BENCH_SECTIONS + ["BenchmarkRunDrawer", "ToolCallApprovalDrawer"]:
            assert name in text, f"{name} not registered in spec"

    def test_drawer_components_use_modal(self):
        for p in (APPROVAL_DRAWER, BENCH_DRAWER):
            assert "from '$lib/ds'" in _read(p)
            assert "Modal" in _read(p)
