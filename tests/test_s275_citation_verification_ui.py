#!/usr/bin/env python3
"""S275 -- the citation-verify UI half (the SvelteKit lot): the standalone
surface the user drives over the already-built S274 route, the
container-provable half of a labelled host-assured browser lot (the s248 /
s253 / s269 precedent).

S274 landed the citation-verify route
(``POST /api/claims/verify-citations``, returning
``CitationVerificationResultSchema``): a single per-user POST that runs
``citation_extraction.extract_pairs(answer, sources)`` (S273, fail-closed by
omission) to derive the (claim, source) pairs, hands them to
``claim_aggregation.make_answer_verifier(client) -> verify_answer(pairs)``
(S271, fail-secure), and returns the aggregate verdict, the per-pair results,
AND the extracted (claim, source) pairs, the pairs positionally aligned with
the results. The UI half was deferred there. S275 lands that front: one typed
API client (``api/citationVerification.ts`` over
``/api/claims/verify-citations``), one standalone component
(``CitationVerifier.svelte``: an answer textarea, an ordered-sources textarea,
a model Select, a Verify action, and the structured aggregate shown alongside
the per-pair verdicts and the extracted (claim, source) pairs), and one
standalone page (``routes/verify-citations/+page.svelte``) that renders it.

The component is added ALONGSIDE -- it does not edit ClaimVerifier.svelte (the
S269 UI), routes/claims/+page.svelte (the S269 page), Sidebar.svelte (the S270
nav), or any other pinned UI surface, and it adds NO cartography row (the S274
no-row precedent: the s217 no-ghosts invariant constrains spec -> disk, never
disk -> spec, so a new unmentioned component is no ghost), so the supersession
surface is a proven ZERO -- the ninth twin of the S266 register. The live
browser run is host-assured (CITATION_VERIFICATION_UI_E2E_S275.md); this suite
proves the container-provable artifacts.

node_modules is absent here, so the frontend is checked by file content, a
structural Svelte block-balance pass, and the --oo-* token discipline (no raw
hex) -- the s174 / s248 / s253 / s269 idiom -- not by a browser; esbuild
validation of the new TS / Svelte is an Apply-phase step, not a pytest concern.
Seven families:

 1. The TS client -- existence, the base-client wiring (apiPost), the
    CitationVerificationResult interface mirroring the backend aggregate schema
    (verdict, ok, reason, the per-pair results, and the extracted pairs), the
    per-pair and pair interfaces, the verdict union mirroring the role
    taxonomy, the endpoint, the verify function, and the answer / ordered
    sources / optional model.
 2. The component -- CitationVerifier: the ds-primitive imports, the client /
    model-source wiring, the behaviour surface (an answer textarea, an ordered
    sources input, the Verify action, the in-flight disable, the CONFIRMED
    server-truth posture), and the structured display (the aggregate verdict /
    reason, the per-pair verdicts, and the extracted (claim, source) pairs
    aligned with the per-pair results), the --oo-* tokens (no raw hex), and
    Svelte block balance.
 3. The page route -- routes/verify-citations/+page.svelte: the component
    imported and rendered. A new standalone page, NOT an edit of a pinned
    surface.
 4. The runbook -- CITATION_VERIFICATION_UI_E2E_S275.md: existence, status
    (host-assured, findings-not-fixes, never-simulated), the
    container-vs-host split, the required sections, the companions, the
    version held, the auth core edit-free note, and pure ASCII.
 5. The seams the UI is the client of -- source pins on the premises (the
    S274 route and its single endpoint / aggregate schema / no-mode-gate, the
    S273 extractor and its extract_pairs, the S271 aggregator and its
    make_answer_verifier, the S267 role taxonomy, the base HTTP client's
    apiPost, the ds primitives, the selectedModel store, the models client),
    so a later edit that removes a premise turns this suite red instead of
    letting the UI rot. Green on the pristine S274 tree by design.
 6. The not-a-tool negatives -- the client and the component define no tool
    schema and register nothing in the tool registry (the CV-D1 / N.3
    posture).
 7. ASCII / structure -- the new TS and Svelte files are pure ASCII and
    block-balanced, this suite parses, and this suite avoids the canonical
    selection literal (built only in split form) so the selection grep is not
    perturbed.

Red-before discipline: on the pristine S274 tree (no UI client, no component,
no page, no runbook) every family-1..4 and family-7 file pin FAILS -- the read
helpers return empty strings so absence is a bare AssertionError, never a
collection error -- while every family-5 seam pin and the family-6 not-a-tool /
the family-7 avoids-literal / suite-parses negatives pass (they pin
pre-existing premises and empty-string-true negatives). Document pins read
through a whitespace-flattening helper (the s221 / s238 lesson) so reflow
cannot break them; source pins stay raw. This suite imports no package code,
so no ollama chain is touched.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend" / "src"

# New frontend source landed by S275.
CLIENT = FE / "lib" / "api" / "citationVerification.ts"
COMPONENT = FE / "lib" / "components" / "panels" / "CitationVerifier.svelte"
PAGE = FE / "routes" / "verify-citations" / "+page.svelte"

# Docs.
RUNBOOK = ROOT / "CITATION_VERIFICATION_UI_E2E_S275.md"

# Seam premises (pre-existing; green on the pristine S274 tree by design).
ROUTE = ROOT / "opti_oignon" / "api" / "routes_citation_verification.py"
EXTRACTOR = ROOT / "opti_oignon" / "agent" / "citation_extraction.py"
AGGREGATOR = ROOT / "opti_oignon" / "agent" / "claim_aggregation.py"
ROLE = ROOT / "opti_oignon" / "agent" / "claim_verification.py"
BASE_CLIENT = FE / "lib" / "api" / "client.ts"
DS_INDEX = FE / "lib" / "ds" / "index.ts"
CHAT_OPTIONS = FE / "lib" / "stores" / "chatOptions.ts"
MODELS_CLIENT = FE / "lib" / "api" / "models.ts"

NEW_TS = (CLIENT,)
NEW_SVELTE = (COMPONENT, PAGE)


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
# Family 1 -- the TS client
# ---------------------------------------------------------------------------


class TestCitationVerificationClient:
    def test_exists(self):
        assert CLIENT.exists()

    def test_imports_base_client(self):
        s = _read(CLIENT)
        assert "from './client'" in s
        assert "apiPost" in s

    def test_aggregate_interface_mirrors_schema(self):
        s = _read(CLIENT)
        assert "export interface CitationVerificationResult" in s
        # Mirrors CitationVerificationResultSchema: the aggregate plus the
        # per-pair results plus the extracted (claim, source) pairs.
        for field in ("verdict", "ok", "reason", "results", "pairs"):
            assert field in s, field

    def test_per_pair_interface_mirrors_schema(self):
        s = _read(CLIENT)
        # Mirrors the per-pair ClaimVerificationResultSchema.
        assert "raw_text" in s

    def test_pair_interface_present(self):
        s = _read(CLIENT)
        # The extracted (claim, source) pair shape.
        assert "claim" in s
        assert "source" in s

    def test_verdict_union_mirrors_role_taxonomy(self):
        s = _read(CLIENT)
        for verdict in ("'supported'", "'unsupported'", "'uncertain'"):
            assert verdict in s, verdict

    def test_verify_function_and_endpoint(self):
        s = _read(CLIENT)
        assert "export async function verifyCitations" in s
        assert "apiPost" in s
        assert "/api/claims/verify-citations" in s

    def test_carries_answer_sources_and_optional_model(self):
        s = _read(CLIENT)
        assert "answer" in s
        assert "sources" in s
        # The model is optional: it rides the request but is not required.
        assert "model" in s
        assert "?" in s


# ---------------------------------------------------------------------------
# Family 2 -- the component
# ---------------------------------------------------------------------------


class TestCitationVerifierComponent:
    def test_exists(self):
        assert COMPONENT.exists()

    def test_imports_ds_primitives(self):
        s = _read(COMPONENT)
        assert "from '$lib/ds'" in s
        for prim in ("Button", "Card", "Icon", "EmptyState", "InlineError", "Select"):
            assert prim in s, prim

    def test_wires_client(self):
        s = _read(COMPONENT)
        assert "from '$lib/api/citationVerification'" in s
        assert "verifyCitations" in s

    def test_wires_model_source(self):
        # The user's selected model: the component reads the existing models
        # client and the selectedModel store (read-only); it edits neither.
        s = _read(COMPONENT)
        assert "from '$lib/api/models'" in s
        assert "listModels" in s
        assert "from '$lib/stores/chatOptions'" in s
        assert "selectedModel" in s

    def test_behaviour_surface(self):
        s = _read(COMPONENT)
        assert "<textarea" in s
        assert "async function verify" in s
        # CONFIRMED posture: the result is taken only from the server response.
        assert "await verifyCitations(" in s
        # The control is disabled while a request is in flight.
        assert "disabled" in s

    def test_displays_aggregate(self):
        s = _read(COMPONENT)
        # The mapped aggregate verdict and the fail-secure reason.
        for verdict in ("supported", "unsupported", "uncertain"):
            assert verdict in s, verdict
        assert "reason" in s

    def test_displays_per_pair_and_extracted_pairs(self):
        s = _read(COMPONENT)
        # The distinguishing S274 surface: the extracted (claim, source) pairs
        # rendered alongside the per-pair results. The component iterates the
        # results / pairs the route returns.
        assert "results" in s
        assert "pairs" in s
        assert "{#each" in s

    def test_no_raw_hex(self):
        raw = _read(COMPONENT)
        assert raw, "CitationVerifier.svelte absent"
        assert not _has_raw_hex(raw)

    def test_block_balanced(self):
        raw = _read(COMPONENT)
        assert raw, "CitationVerifier.svelte absent"
        assert _block_balanced(raw)


# ---------------------------------------------------------------------------
# Family 3 -- the page route (a new standalone page, not a pinned-surface edit)
# ---------------------------------------------------------------------------


class TestVerifyCitationsPageRoute:
    def test_exists(self):
        assert PAGE.exists()

    def test_imports_and_renders_component(self):
        s = _read(PAGE)
        assert (
            "import CitationVerifier from "
            "'$lib/components/panels/CitationVerifier.svelte'" in s
        )
        assert "<CitationVerifier" in s


# ---------------------------------------------------------------------------
# Family 4 -- the host-assured browser runbook
# ---------------------------------------------------------------------------


class TestRunbook:
    def test_exists_and_titled(self):
        text = _read(RUNBOOK)
        assert text.startswith("# CITATION_VERIFICATION_UI_E2E_S275")
        assert len(text) > 4000

    def test_status_and_discipline(self):
        text = _flat(_read(RUNBOOK))
        assert "written at S275" in text
        assert "host-assured" in text
        assert "produces findings, not fixes" in text
        assert "never simulated in the container" in text

    def test_container_vs_host_split(self):
        text = _flat(_read(RUNBOOK))
        assert "Container-provable" in text
        assert "Host-assured" in text
        assert "held at 3.13.0" in text

    def test_required_sections(self):
        text = _flat(_read(RUNBOOK))
        for needle in (
            "Preflight",
            "Answer and sources",
            "verdict",
            "fail-secure",
            "Extracted pairs",
            "Untrusted-context",
            "Findings register",
            "Routing",
        ):
            assert needle in text, needle

    def test_availability_guard_named(self):
        text = _flat(_read(RUNBOOK))
        assert "503" in text

    def test_companions_named(self):
        text = _flat(_read(RUNBOOK))
        for needle in (
            "routes_citation_verification.py",
            "citation_extraction.py",
            "claim_aggregation.py",
            "CLAIM_VERIFICATION_UI_E2E_S269.md",
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
# Family 5 -- the seams the UI is the client of (green on pristine by design)
# ---------------------------------------------------------------------------


class TestSeamRouteAndPipeline:
    def test_route_single_endpoint_and_schema(self):
        src = _read(ROUTE)
        assert "citation_verification_router" in src
        assert 'prefix="/api/claims"' in src
        assert "@citation_verification_router.post(" in src
        assert '"/verify-citations"' in src
        assert "CitationVerificationResultSchema" in src

    def test_route_aggregate_schema_fields(self):
        src = _read(ROUTE)
        assert "class CitationVerificationResultSchema" in src
        for field in ("verdict", "ok", "reason", "results", "pairs"):
            assert field in src, field

    def test_route_has_no_mode_gate(self):
        # CV-D4: the route carries no mode seam and no mode provider; a future
        # edit that adds one turns this red.
        src = _read(ROUTE)
        for forbidden in (
            "get_current_mode",
            "security_mode",
            "MODE_DAILY",
            "MODE_BULBE",
        ):
            assert forbidden not in src, forbidden

    def test_extractor_extract_pairs(self):
        src = _read(EXTRACTOR)
        assert "def extract_pairs" in src

    def test_aggregator_factory(self):
        src = _read(AGGREGATOR)
        assert "def make_answer_verifier" in src

    def test_role_taxonomy(self):
        src = _read(ROLE)
        assert "make_claim_verifier" in src
        for verdict in ('"supported"', '"unsupported"', '"uncertain"'):
            assert verdict in src, verdict


class TestSeamFrontend:
    def test_base_client_apipost(self):
        src = _read(BASE_CLIENT)
        assert "function apiPost" in src

    def test_ds_primitives(self):
        src = _read(DS_INDEX)
        for prim in ("Button", "Card", "Icon", "EmptyState", "InlineError", "Select"):
            assert prim in src, prim

    def test_selected_model_store(self):
        src = _read(CHAT_OPTIONS)
        assert "selectedModel" in src

    def test_models_client(self):
        src = _read(MODELS_CLIENT)
        assert "listModels" in src
        assert "/api/models" in src


# ---------------------------------------------------------------------------
# Family 6 -- the not-a-tool negatives
# ---------------------------------------------------------------------------


class TestNotATool:
    def test_client_defines_no_tool(self):
        # The UI surface is caller-driven (CV-D1): the client posts to the
        # route; it is not a model-reachable tool and registers nothing.
        s = _read(CLIENT)
        assert "ToolSchema" not in s
        assert "register_tool" not in s

    def test_component_defines_no_tool(self):
        s = _read(COMPONENT)
        assert "ToolSchema" not in s
        assert "register_tool" not in s


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

    def test_this_suite_pure_ascii(self):
        src = _read(Path(__file__))
        assert src != ""
        assert all(ord(c) < 128 for c in src)

    def test_this_suite_avoids_selection_literal(self):
        # The canonical selection grep matches a literal this suite must not
        # contain contiguously; build it only in split form here so the raw
        # grep count is not perturbed.
        src = _read(Path(__file__))
        assert ("sandbox" + "_" + "manager") not in src
