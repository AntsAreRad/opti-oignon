"""S244 -- N.4: the gated ``manage_notes`` state-mutation tool over the N.1
NotesStore.

The data layer landed at S243 (the ``opti_oignon/notes/`` subpackage). This lot
adds the LLM-from-chat write surface: a ``manage_notes`` tool that the agent can
call to create notes and mutate their metadata, wired over the N.1 ``NotesStore``
and gated like ``manage_memory`` / ``manage_skills`` -- it joins
``STATE_MUTATION_TOOLS``, so it is auto in Daily and forbidden in Bulbe by the
structural ``BULBE_ALLOWLIST = DAILY - NETWORK - STATE_MUTATION`` derivation (no
new gating path). Per-user isolation is the store's (``effective_user_id``); the
tool resolves the active user through the store exactly as ``manage_memory`` does.

Scope decision (the S244 read gate): the note body is an OPAQUE, client-owned
CRDT, so a server-side tool cannot honestly merge text into a body. ``make``
seeds a new note with the model's markdown stored as opaque body bytes (the
backend never interprets them), and the metadata mutations (title / tags /
pinned) and the tombstone delete are honest server-side operations. The in-body
"add-to-note" insertion is deferred to N.8 (the CRDT merge story); it is NOT a
``manage_notes`` action here.

Five actions, mirroring ``manage_memory`` (list / get / add / update / delete):
``list``, ``get``, ``make``, ``update``, ``delete``.

Six families:

 1. Source / structure -- ``manage_notes`` in ``STATE_MUTATION_TOOLS``, the tool
    constant, the schema, the handler builder, the schema in ``ALL_SCHEMAS``, the
    name in ``HANDLER_TOOL_NAMES``, the registry registration.
 2. Gating -- ``allowlists`` loaded in isolation: ``manage_notes`` in Daily, not
    in Bulbe; ``is_tool_allowed`` / ``evaluate`` Daily-allows and Bulbe-refuses
    (``not_in_allowlist``) before any approval.
 3. Behavioural -- the handler built over a tmp ``NotesStore`` (loaded in
    isolation): make / list / get / update / delete round-trips, and the handler
    returns observations and never raises (bad action, missing id, missing
    title).
 4. Count reassertions -- the supersession of the LIVE S228 schema/handler-count
    pins: ``len(ALL_SCHEMAS) == 13``, ``HANDLER_TOOL_NAMES`` now the five, Daily
    exposes ``manage_notes`` with its handler attached and Bulbe excludes it,
    Daily thirteen / Bulbe nine.
 5. Premise guards -- green before and after: the NotesStore module loads, the
    allowlists primitives exist, the ``make_manage_memory_handler`` pattern
    exists, the Bulbe derivation is structural, and the base state-mutation set
    (manage_memory + manage_skills) is intact.
 6. AST / ASCII -- the touched sources parse and this suite parses; pure ASCII,
    no decoration.

Red-before discipline: on the pristine S243 tree (no ``manage_notes`` anywhere)
every family-1/2/3/4 pin FAILS (the source text lacks the symbols; the handler
builder is absent so the behavioural family raises AttributeError INSIDE the
test, never at collection; the counts read 12 / four), while every family-5
guard and the family-6 AST/ASCII pins PASS by design.

Isolation (the S243 lesson): the behavioural family loads ``notes_store`` under a
FLAT name via ``spec_from_file_location`` so it is robust to an earlier suite
having replaced ``sys.modules['opti_oignon']`` with a non-package stub; under a
flat name the store's guarded relative imports fall back to a plaintext sqlite
connection (the documented in-container posture). ``allowlists`` and ``tools``
load under their dotted names into package-like stubs (the S228 idiom, hardened
so a pre-existing non-package stub is made package-like rather than replaced);
the handlers in ``tools`` import their backends lazily, so the module loads with
no fastapi / ollama chain, and the ``manage_notes`` handler is built directly
with an injected store (never via the registry's lazy default).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

# Defensive: never pull the real ollama during collection.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"
ALLOWLISTS_PATH = PKG / "agent" / "allowlists.py"
TOOLS_PATH = PKG / "agent" / "tools.py"
NOTES_STORE_PATH = PKG / "notes" / "notes_store.py"

EXPECTED_HANDLER_NAMES = frozenset(
    {"web_search", "manage_memory", "manage_skills", "todo", "manage_notes"}
)
EXPECTED_ALL_SCHEMAS = 13
EXPECTED_DAILY_NAMES = 13
EXPECTED_BULBE_NAMES = 9


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Isolation harness
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
    """Ensure ``name`` exists in sys.modules and is package-like.

    Non-destructive: keeps any pre-existing stub object (an earlier suite's),
    only granting it a ``__path__`` so a dotted ``spec_from_file_location`` load
    of a submodule resolves. This is the S228 ``_ensure_pkg`` hardened against
    the S243 non-package-stub hazard.
    """
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if not hasattr(mod, "__path__"):
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]


_ensure_pkg("opti_oignon", PKG)
_ensure_pkg("opti_oignon.agent", PKG / "agent")


def _load_dotted(name: str, path: Path):
    """Load a module under its real dotted name, reusing an existing load."""
    existing = sys.modules.get(name)
    if existing is not None and hasattr(existing, "__file__"):
        return existing
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# allowlists and tools load with no backend (the handlers import lazily).
_al = _load_dotted("opti_oignon.agent.allowlists", ALLOWLISTS_PATH)
_tools = _load_dotted("opti_oignon.agent.tools", TOOLS_PATH)


_ISO: dict = {}


def _isolated_flat(name: str, rel: str):
    """Load a module under a FLAT name (the S243 lesson: robust in the sweep)."""
    if name not in _ISO:
        spec = importlib.util.spec_from_file_location(name, str(PKG / rel))
        if spec is None or spec.loader is None:
            raise ImportError(name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _ISO[name] = mod
    return _ISO[name]


def _notes_store_module():
    return _isolated_flat("s244_notes_store_iso", "notes/notes_store.py")


def _build_handler(tmp_path):
    """Build the manage_notes handler over a fresh single-user NotesStore."""
    ns = _notes_store_module()
    store = ns.NotesStore(root=tmp_path)
    builder = getattr(_tools, "make_manage_notes_handler")  # absent -> red
    return builder(store), store


def _injected_registry():
    """A ToolRegistry with all backends injected, so no skills/memory/notes
    import fires at construction (the count family probes the schema set only)."""
    return _tools.ToolRegistry(
        web_search_fn=lambda **kw: "",
        memory_store=object(),
        skills_handler=lambda a: "",
    )


# ---------------------------------------------------------------------------
# Family 1 -- source / structure (raw text; red before the implementation)
# ---------------------------------------------------------------------------


class TestSourceStructure:
    def test_state_mutation_includes_manage_notes(self):
        src = _read(ALLOWLISTS_PATH)
        assert "STATE_MUTATION_TOOLS = frozenset(" in src
        # the literal carries all three names
        start = src.index("STATE_MUTATION_TOOLS = frozenset(")
        literal = src[start : start + 160]
        assert "manage_memory" in literal
        assert "manage_skills" in literal
        assert "manage_notes" in literal

    def test_tool_constant_and_schema_present(self):
        src = _read(TOOLS_PATH)
        assert 'TOOL_MANAGE_NOTES = "manage_notes"' in src
        assert "MANAGE_NOTES_SCHEMA = ToolSchema(" in src
        assert "def make_manage_notes_handler(" in src

    def test_schema_in_all_schemas(self):
        src = _read(TOOLS_PATH)
        start = src.index("ALL_SCHEMAS: tuple[ToolSchema, ...] = (")
        tup = src[start : src.index(")", start)]
        assert "MANAGE_NOTES_SCHEMA," in tup

    def test_name_in_handler_tool_names(self):
        src = _read(TOOLS_PATH)
        start = src.index("HANDLER_TOOL_NAMES = frozenset(")
        literal = src[start : src.index(")", start)]
        assert "TOOL_MANAGE_NOTES" in literal

    def test_registry_registers_notes_handler(self):
        src = _read(TOOLS_PATH)
        assert "make_manage_notes_handler(" in src
        assert "TOOL_MANAGE_NOTES: make_manage_notes_handler(" in src

    def test_no_raw_sql_in_tools(self):
        # The tool delegates to NotesStore; no SQL (and no f-string SQL) lives
        # in the agent tool layer.
        src = _read(TOOLS_PATH)
        assert "sqlite3" not in src
        assert "INSERT INTO" not in src


# ---------------------------------------------------------------------------
# Family 2 -- gating (allowlists in isolation; red before the implementation)
# ---------------------------------------------------------------------------


class TestGating:
    def test_in_state_mutation_frozenset(self):
        assert "manage_notes" in _al.STATE_MUTATION_TOOLS

    def test_daily_includes_bulbe_excludes(self):
        assert "manage_notes" in _al.DAILY_ALLOWLIST
        assert "manage_notes" not in _al.BULBE_ALLOWLIST

    def test_is_tool_allowed_per_mode(self):
        assert _al.is_tool_allowed("manage_notes", "daily") is True
        assert _al.is_tool_allowed("manage_notes", "bulbe") is False
        # unknown mode is fail-secure Bulbe
        assert _al.is_tool_allowed("manage_notes", "nonsense") is False

    def test_evaluate_daily_allows(self):
        d = _al.evaluate("manage_notes", {"action": "list"}, mode="daily")
        assert d.allowed is True
        assert d.reason == _al.REASON_ALLOWED

    def test_evaluate_bulbe_refuses_before_approval(self):
        d = _al.evaluate("manage_notes", {"action": "make"}, mode="bulbe")
        assert d.allowed is False
        assert d.reason == _al.REASON_NOT_ALLOWED


# ---------------------------------------------------------------------------
# Family 3 -- behavioural (isolated tools + NotesStore; red before)
# ---------------------------------------------------------------------------


class TestHandlerBehaviour:
    def test_make_creates_note(self, tmp_path):
        handler, store = _build_handler(tmp_path)
        out = handler(
            {
                "action": "make",
                "title": "Groceries",
                "body": "- milk\n- eggs",
                "tags": ["shopping"],
            }
        )
        assert isinstance(out, str) and out
        assert store.count_notes() == 1
        notes = store.list_notes()
        assert notes[0].title == "Groceries"
        assert notes[0].body_crdt.decode("utf-8") == "- milk\n- eggs"
        assert "shopping" in notes[0].tags

    def test_list_shows_notes(self, tmp_path):
        handler, store = _build_handler(tmp_path)
        handler({"action": "make", "title": "Alpha"})
        handler({"action": "make", "title": "Beta"})
        out = handler({"action": "list"})
        assert "Alpha" in out and "Beta" in out

    def test_get_returns_metadata_and_body(self, tmp_path):
        handler, store = _build_handler(tmp_path)
        handler({"action": "make", "title": "Recipe", "body": "step one"})
        nid = store.list_notes()[0].id
        out = handler({"action": "get", "note_id": nid})
        assert "Recipe" in out
        assert "step one" in out

    def test_update_changes_fields(self, tmp_path):
        handler, store = _build_handler(tmp_path)
        handler({"action": "make", "title": "Old"})
        nid = store.list_notes()[0].id
        handler(
            {
                "action": "update",
                "note_id": nid,
                "title": "New",
                "tags": ["a", "b"],
                "pinned": True,
            }
        )
        rec = store.get_note(nid)
        assert rec.title == "New"
        assert rec.pinned is True
        assert "a" in rec.tags and "b" in rec.tags

    def test_delete_tombstones(self, tmp_path):
        handler, store = _build_handler(tmp_path)
        handler({"action": "make", "title": "Doomed"})
        nid = store.list_notes()[0].id
        out = handler({"action": "delete", "note_id": nid})
        assert isinstance(out, str) and out
        assert store.count_notes() == 0
        # the row survives as a tombstone (sync-correct deletion)
        assert len(store.list_notes(include_deleted=True)) == 1

    def test_unknown_action_is_observation_not_raise(self, tmp_path):
        handler, _ = _build_handler(tmp_path)
        out = handler({"action": "obliterate"})
        assert isinstance(out, str) and out

    def test_make_requires_title(self, tmp_path):
        handler, store = _build_handler(tmp_path)
        out = handler({"action": "make"})
        assert isinstance(out, str) and out
        assert store.count_notes() == 0

    def test_get_requires_note_id(self, tmp_path):
        handler, _ = _build_handler(tmp_path)
        out = handler({"action": "get"})
        assert isinstance(out, str) and out


# ---------------------------------------------------------------------------
# Family 4 -- count reassertions (the supersession; red before)
# ---------------------------------------------------------------------------


class TestCountReassertions:
    def test_all_schemas_is_thirteen(self):
        assert len(_tools.ALL_SCHEMAS) == EXPECTED_ALL_SCHEMAS

    def test_handler_tool_names_is_the_five(self):
        assert _tools.HANDLER_TOOL_NAMES == EXPECTED_HANDLER_NAMES

    def test_daily_exposes_notes_with_handler(self):
        reg = _injected_registry()
        ts = reg.build("daily")
        assert "manage_notes" in ts.names
        assert "manage_notes" in ts.tool_handlers

    def test_bulbe_excludes_notes(self):
        reg = _injected_registry()
        ts = reg.build("bulbe")
        assert "manage_notes" not in ts.names
        assert "manage_notes" not in ts.tool_handlers

    def test_daily_thirteen_bulbe_nine(self):
        reg = _injected_registry()
        assert len(reg.build("daily").names) == EXPECTED_DAILY_NAMES
        assert len(reg.build("bulbe").names) == EXPECTED_BULBE_NAMES


# ---------------------------------------------------------------------------
# Family 5 -- premise guards (green before AND after)
# ---------------------------------------------------------------------------


class TestPremiseGuards:
    def test_notes_store_module_loads(self, tmp_path):
        ns = _notes_store_module()
        store = ns.NotesStore(root=tmp_path)
        try:
            assert store.count_notes() == 0
        finally:
            store.close()

    def test_allowlists_primitives_exist(self):
        for attr in (
            "STATE_MUTATION_TOOLS",
            "DAILY_ALLOWLIST",
            "BULBE_ALLOWLIST",
            "NETWORK_TOOLS",
            "evaluate",
            "is_tool_allowed",
        ):
            assert hasattr(_al, attr), attr

    def test_base_state_mutation_set_intact(self):
        # the two we extend are present (stable across this lot)
        assert "manage_memory" in _al.STATE_MUTATION_TOOLS
        assert "manage_skills" in _al.STATE_MUTATION_TOOLS

    def test_memory_handler_pattern_exists(self):
        assert hasattr(_tools, "make_manage_memory_handler")

    def test_bulbe_derivation_is_structural(self):
        # adding to STATE_MUTATION auto-excludes from Bulbe by construction
        src = _read(ALLOWLISTS_PATH)
        assert (
            "BULBE_ALLOWLIST = frozenset(DAILY_ALLOWLIST - NETWORK_TOOLS "
            "- STATE_MUTATION_TOOLS)" in src
        )


# ---------------------------------------------------------------------------
# Family 4b -- superseded invariants reasserted (deselect-plus-reassert)
# ---------------------------------------------------------------------------


class TestSupersededInvariantsReasserted:
    """The corrected forms of the LIVE S228 schema/order/membership pins this
    lot supersedes, plus the config.yaml consistency the fix preserves."""

    def test_schema_order_is_stable(self):
        names = [s.name for s in _tools.ALL_SCHEMAS]
        assert names == [
            "bash", "view", "create_file", "str_replace",
            "grep", "glob", "ls",
            "web_search", "manage_memory", "manage_skills", "manage_notes",
            "todo", "task",
        ]

    def test_non_sandbox_set_is_the_six(self):
        non = {s.name for s in _tools.ALL_SCHEMAS if not s.sandboxed}
        assert non == {
            _tools.TOOL_WEB_SEARCH,
            _tools.TOOL_MANAGE_MEMORY,
            _tools.TOOL_MANAGE_SKILLS,
            _tools.TOOL_MANAGE_NOTES,
            _tools.TOOL_TODO,
            _tools.TOOL_TASK,
        }

    def test_state_mutation_frozenset_exact(self):
        assert _al.STATE_MUTATION_TOOLS == frozenset(
            {"manage_memory", "manage_skills", "manage_notes"}
        )

    def test_daily_tool_handlers_is_the_five(self):
        reg = _injected_registry()
        ts = reg.build("daily")
        assert set(ts.tool_handlers) == {
            "web_search",
            "manage_memory",
            "manage_skills",
            "manage_notes",
            "todo",
        }

    def test_config_yaml_declares_notes_in_daily_only(self):
        text = _read(PKG / "agent" / "config.yaml")
        daily_block = text[text.index("daily:") : text.index("bulbe:")]
        bulbe_block = text[text.index("bulbe:") : text.index("diagnostics:")]
        assert "manage_notes" in daily_block
        assert "manage_notes" not in bulbe_block

    def test_config_yaml_daily_thirteen_bulbe_nine(self):
        text = _read(PKG / "agent" / "config.yaml")
        daily_block = text[text.index("daily:") : text.index("bulbe:")]
        bulbe_block = text[text.index("bulbe:") : text.index("diagnostics:")]
        daily_n = sum(
            1 for ln in daily_block.splitlines() if ln.strip().startswith("- ")
        )
        bulbe_n = sum(
            1 for ln in bulbe_block.splitlines() if ln.strip().startswith("- ")
        )
        assert daily_n == 13
        assert bulbe_n == 9


# ---------------------------------------------------------------------------
# Family 7 -- addopts lineage (the deselect-plus-reassert bookkeeping)
# ---------------------------------------------------------------------------

# The twelve LIVE pins this lot supersedes, deselected in pyproject addopts so
# the normal run stays green; each corrected truth is reasserted above. Paths
# are relative to tests/ (the f-string prepends it), matching the s232/s236 form.
S244_DESELECTS = (
    "test_s228_agt_lot1.py::TestFacade::test_config_yaml_lists_grown",
    "test_s228_agt_lot1.py::TestRegistryGrowth::test_daily_exposes_all_twelve",
    "test_s228_agt_lot1.py::TestRegistryGrowth::test_daily_handlers_include_todo",
    "test_s228_agt_lot1.py::TestRegistryGrowth::test_schema_order_is_stable",
    "test_s228_agt_lot1.py::TestSupersessionReassertions::test_all_schemas_is_twelve_today",
    "test_s228_agt_lot1.py::TestSupersessionReassertions::test_daily_handler_set_gains_todo",
    "test_s228_agt_lot1.py::TestSupersessionReassertions::test_frozensets_exact_post_s228",
    "test_s228_agt_lot1.py::TestSupersessionReassertions::test_handler_names_are_the_four",
    "test_s228_agt_lot1.py::TestSupersessionReassertions::test_non_sandbox_set_gains_todo_and_task",
    "test_s228_agt_lot1.py::TestSupersessionReassertions::test_twelve_schemas",
    "test_s243_notes_data_layer.py::TestPremiseGuards::test_manage_notes_not_yet_a_state_mutation_tool",
    # the prior session's LIVE addopts-count pin, superseded by this lot's growth
    "test_s242_atrest_consistency.py::TestAddoptsLineageS242::test_count_grew_by_exactly_two_to_208",
)


class TestAddoptsLineage:
    def setup_method(self):
        self.src = _read(REPO / "pyproject.toml")

    def test_carries_the_twelve_s244_supersessions(self):
        for node in S244_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_count_grew_by_exactly_twelve(self):
        # 208 deselects at the S243 close; the twelve S244 supersessions join
        # (the ten LIVE s228 schema / order / membership / config-count pins,
        # the s243 boundary guard, and the s242 addopts-count pin this lot's
        # own growth supersedes); nothing is ever removed.
        assert self.src.count("--deselect=") == 220


# ---------------------------------------------------------------------------
# Family 6 -- AST / ASCII (green before AND after)
# ---------------------------------------------------------------------------


class TestAstAndAscii:
    def test_touched_sources_parse(self):
        for path in (ALLOWLISTS_PATH, TOOLS_PATH, NOTES_STORE_PATH):
            src = _read(path)
            assert src != "", str(path)
            ast.parse(src, filename=str(path))

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)

    def test_pure_ascii_no_decoration(self):
        raw = _read(Path(__file__))
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        # built at runtime so the literal four-equals does not appear in source
        assert ("=" * 4) not in raw
