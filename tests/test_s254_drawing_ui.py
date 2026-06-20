#!/usr/bin/env python3
"""S254 -- the Notes feature's N.7 drawing canvas: frontend-first SVG (the
SvelteKit lot), the container-provable half of a labelled browser-runbook lot
(the s248 / s253 precedent).

S254 lands the drawing surface the user drives over the already-built media
backend and the S253 frontend client: a pure-TS drawing model module
(lib/drawing/svgDrawing.ts -- a minimal stroke/shape model serialized to a
standalone SVG document and parsed back for re-editing our own format), the
NotesDrawingCanvas component (pointer-event editor, named-colour palette so
the no-raw-hex discipline holds by construction, save through the S253 upload
client as an attachment of kind "drawing", thumbnails through in-memory
object URLs, re-edit through fetchAttachmentBlob + parseDrawing, and a
fail-safe replace: the new version uploads first and the old one is removed
only after success), one ADDITIVE derived view in stores/attachments.ts
(drawingAttachments, the audio/image siblings' idiom), an ADDITIVE
NotesPanel embed, the NEW/S254 spec-registry row, and the host-assured
runbook NOTES_DRAWING_E2E_S254.md. NO backend change; the version is HELD at
3.11.0. The optional vision describe is settled OUT of this slice at the read
gate (a later deliberate kind-widening if ever wanted; never
rasterize-as-image).

node_modules is absent here, so the frontend is checked by file content, a
structural Svelte block-balance pass, and the --oo-* token discipline (no raw
hex) -- the s174/s248/s253 idiom -- not by a browser. Eight families:

 1. The drawing model module -- existence, the exported types and constants
    (DRAWING_MIME, the named-colour palette), the serializer contract (the
    standalone SVG envelope, the data-oo-* round-trip markers, the per-tool
    elements, the colour guard), the parser contract (DOMParser, the
    foreign-SVG null, the marker check), and drawingToBlob.
 2. The store edit -- the ADDITIVE drawingAttachments derived view (the
    audio/image siblings untouched, re-asserted in family 7).
 3. The NotesDrawingCanvas component -- ds-primitive imports, store / client
    / module wiring, the behaviour surface (pointer capture, undo / clear,
    the kind-"drawing" save, the upload-before-remove replace, the re-edit
    parse with its not-editable refusal surfaced, object-URL hygiene), the
    --oo-* tokens (no raw hex), and Svelte block balance.
 4. The NotesPanel embed -- the component imported and rendered in the
    notes-media block after the S253 pair, fed the active note id (an
    ADDITIVE edit: every S248 and S253 presence pin stays green by
    construction).
 5. The spec registration -- the NotesDrawingCanvas.svelte NEW/S254 row (the
    s174 registration regex); the two S253 rows are re-asserted in family 7.
 6. The runbook -- NOTES_DRAWING_E2E_S254.md: existence, status
    (host-assured, never simulated), the live draw / save / reload / re-edit
    walk, the replace discipline, the ciphertext spot check, the findings
    register, the version held, and pure ASCII.
 7. The seams the canvas is the client of -- source pins on the premises
    (the notes_store three-kind allowlist incl. "drawing", the S249 route's
    five legs and its kind validation, the S253 typed client incl. the
    'drawing' union member and the authed blob fetch, the S253 store actions
    this component reuses, the ds exports, the S248/S253 NotesPanel surface
    this lot embeds into, and the two S253 spec rows), so a later edit that
    removes a premise turns this suite red instead of letting the canvas
    rot. Green on the pristine S253 tree by design.
 8. ASCII / structure -- the new TS, Svelte and runbook files are pure ASCII
    (existence-guarded so absence fails), the pristine NotesPanel stays
    block-balanced, the version file is held, and this suite parses.

Red-before discipline: on the pristine S253 tree (no drawing module, no
canvas component, no derived view, no embed, no spec row, no runbook) every
family-1..6 pin and the existence-guarded family-8 pins FAIL -- the read
helpers return empty strings and the guards assert existence so absence is a
failure, never a collection error -- while every family-7 seam pin and the
three pristine-true family-8 pins (panel balance, version held, suite parse)
pass. Document pins read through a whitespace-flattening helper (the
s221/s238 lesson) so reflow cannot break them; source pins stay raw. This
suite imports no package code, so no ollama chain is touched.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend" / "src"

# New frontend source landed by S254.
DRAWING_MODULE = FE / "lib" / "drawing" / "svgDrawing.ts"
CANVAS = FE / "lib" / "components" / "panels" / "NotesDrawingCanvas.svelte"

# Edited additively by S254.
ATTACH_STORE = FE / "lib" / "stores" / "attachments.ts"
NOTES_PANEL = FE / "lib" / "components" / "panels" / "NotesPanel.svelte"

# Docs.
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
RUNBOOK = ROOT / "NOTES_DRAWING_E2E_S254.md"

# Seam premises (pre-existing; green on the pristine S253 tree by design).
ROUTES_ATTACHMENTS = ROOT / "opti_oignon" / "api" / "routes_notes_attachments.py"
NOTES_STORE_PY = ROOT / "opti_oignon" / "notes" / "notes_store.py"
ATTACH_CLIENT = FE / "lib" / "api" / "attachments.ts"
BASE_CLIENT = FE / "lib" / "api" / "client.ts"
DS_INDEX = FE / "lib" / "ds" / "index.ts"
VOICE_CAPTURE = FE / "lib" / "components" / "panels" / "NotesVoiceCapture.svelte"
MEDIA_GALLERY = FE / "lib" / "components" / "panels" / "NotesMediaGallery.svelte"
VERSION_FILE = ROOT / "opti_oignon" / "__version__.py"


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
# Family 1 -- the drawing model module (lib/drawing/svgDrawing.ts)
# ---------------------------------------------------------------------------


class TestDrawingModule:
    def test_exists(self):
        assert DRAWING_MODULE.exists()

    def test_exported_types(self):
        s = _read(DRAWING_MODULE)
        assert "export type DrawingTool" in s
        for tool in ("'pen'", "'line'", "'rect'", "'ellipse'"):
            assert tool in s, tool
        assert "export interface DrawingStroke" in s
        for field in ("tool", "color", "width", "points"):
            assert field in s, field
        assert "export interface DrawingModel" in s
        assert "strokes" in s

    def test_mime_constant(self):
        s = _read(DRAWING_MODULE)
        assert "export const DRAWING_MIME" in s
        assert "'image/svg+xml'" in s

    def test_palette_is_named_colours_only(self):
        s = _read(DRAWING_MODULE)
        assert "export const DRAWING_COLORS" in s
        for colour in (
            "'black'",
            "'crimson'",
            "'steelblue'",
            "'seagreen'",
            "'darkorange'",
            "'rebeccapurple'",
        ):
            assert colour in s, colour
        assert not _has_raw_hex(s)

    def test_colour_guard(self):
        s = _read(DRAWING_MODULE)
        assert "SAFE_COLOR" in s
        assert "/^[a-z]{3,30}$/" in s
        assert "SAFE_COLOR.test(" in s
        assert "'black'" in s  # the guarded fallback

    def test_serializer_exported(self):
        s = _read(DRAWING_MODULE)
        assert "export function serializeDrawing" in s

    def test_serializer_standalone_svg_envelope(self):
        s = _read(DRAWING_MODULE)
        assert '<?xml version="1.0" encoding="UTF-8"?>' in s
        assert 'xmlns="http://www.w3.org/2000/svg"' in s
        assert "viewBox=" in s

    def test_serializer_round_trip_markers(self):
        s = _read(DRAWING_MODULE)
        assert 'data-oo-drawing="1"' in s
        assert "data-oo-tool=" in s

    def test_serializer_per_tool_elements(self):
        s = _read(DRAWING_MODULE)
        for el in ("<polyline", "<line", "<rect", "<ellipse"):
            assert el in s, el
        assert 'fill="none"' in s
        assert 'stroke-linecap="round"' in s
        assert 'stroke-linejoin="round"' in s

    def test_serializer_rounds_coordinates(self):
        s = _read(DRAWING_MODULE)
        assert "function round2(" in s
        assert "Math.round(" in s

    def test_parser_exported(self):
        s = _read(DRAWING_MODULE)
        assert "export function parseDrawing" in s

    def test_parser_uses_domparser_with_marker_check(self):
        s = _read(DRAWING_MODULE)
        assert "DOMParser" in s
        assert "parseFromString" in s
        assert "'image/svg+xml'" in s
        assert "parsererror" in s
        assert "data-oo-drawing" in s
        assert "getAttribute" in s

    def test_parser_returns_null_on_foreign_svg(self):
        s = _read(DRAWING_MODULE)
        assert "return null" in s

    def test_to_blob_helper(self):
        s = _read(DRAWING_MODULE)
        assert "export function drawingToBlob" in s
        assert "new Blob(" in s
        assert "DRAWING_MIME" in s


# ---------------------------------------------------------------------------
# Family 2 -- the ADDITIVE store edit (drawingAttachments derived view)
# ---------------------------------------------------------------------------


class TestStoreDrawingView:
    def test_drawing_derived_exported(self):
        s = _read(ATTACH_STORE)
        assert "export const drawingAttachments" in s

    def test_drawing_derived_filters_on_kind(self):
        s = _read(ATTACH_STORE)
        m = re.search(
            r"export const drawingAttachments = derived\([^;]*'drawing'", s, re.S
        )
        assert m is not None


# ---------------------------------------------------------------------------
# Family 3 -- the NotesDrawingCanvas component
# ---------------------------------------------------------------------------


class TestDrawingCanvasComponent:
    def test_exists(self):
        assert CANVAS.exists()

    def test_header_names_the_lot(self):
        s = _flat(_read(CANVAS))
        assert "NotesDrawingCanvas.svelte (S254, Notes feature N.7)" in s

    def test_note_id_prop(self):
        s = _read(CANVAS)
        assert "export let noteId: string;" in s

    def test_ds_imports(self):
        s = _read(CANVAS)
        assert "from '$lib/ds'" in s
        for sym in ("Button", "Card", "Icon", "EmptyState", "InlineError"):
            assert sym in s, sym

    def test_store_wiring(self):
        s = _read(CANVAS)
        assert "from '$lib/stores/attachments'" in s
        for sym in (
            "drawingAttachments",
            "mediaLoading",
            "mediaError",
            "loadAttachments",
            "uploadNoteAttachment",
            "removeAttachment",
        ):
            assert sym in s, sym

    def test_client_wiring(self):
        s = _read(CANVAS)
        assert "from '$lib/api/attachments'" in s
        assert "fetchAttachmentBlob" in s
        assert "AttachmentRecord" in s

    def test_drawing_module_wiring(self):
        s = _read(CANVAS)
        assert "from '$lib/drawing/svgDrawing'" in s
        for sym in ("parseDrawing", "drawingToBlob", "DRAWING_COLORS"):
            assert sym in s, sym

    def test_toast_wiring(self):
        s = _read(CANVAS)
        assert "from '$lib/stores/notifications'" in s
        assert "toastSuccess" in s
        assert "toastError" in s

    def test_pointer_event_editor(self):
        s = _read(CANVAS)
        for ev in ("on:pointerdown", "on:pointermove", "on:pointerup"):
            assert ev in s, ev
        assert "setPointerCapture" in s
        assert "getBoundingClientRect" in s

    def test_editor_viewbox_and_a11y(self):
        s = _read(CANVAS)
        assert '"0 0 800 600"' in s
        assert 'role="img"' in s
        assert "aria-label=" in s

    def test_undo_and_clear(self):
        s = _read(CANVAS)
        assert "function undoStroke(" in s
        assert "function clearStrokes(" in s

    def test_save_uploads_kind_drawing(self):
        s = _read(CANVAS)
        assert "async function saveDrawing(" in s
        assert "uploadNoteAttachment(noteId, 'drawing'," in s
        assert ".svg" in s

    def test_replace_is_upload_before_remove(self):
        s = _read(CANVAS)
        assert "upload first; remove the old version only after success" in s
        up = s.find("uploadNoteAttachment(")
        rm = s.find("removeAttachment(")
        assert up != -1 and rm != -1
        assert up < rm

    def test_reedit_parses_own_format(self):
        s = _read(CANVAS)
        assert "async function editAttachment(" in s
        assert "fetchAttachmentBlob(" in s
        assert ".text()" in s
        assert "parseDrawing(" in s

    def test_not_editable_refusal_surfaced(self):
        s = _read(CANVAS)
        assert "not an editable drawing" in s

    def test_object_url_hygiene(self):
        s = _read(CANVAS)
        assert "URL.createObjectURL" in s
        assert "URL.revokeObjectURL" in s
        assert "onDestroy(" in s

    def test_error_and_empty_surfaces(self):
        s = _read(CANVAS)
        assert "$mediaError" in s
        assert "<InlineError" in s
        assert "<EmptyState" in s

    def test_token_discipline_no_raw_hex(self):
        assert CANVAS.exists()
        s = _read(CANVAS)
        assert "var(--oo-" in s
        assert not _has_raw_hex(s)

    def test_block_balanced(self):
        assert CANVAS.exists()
        assert _block_balanced(_read(CANVAS))


# ---------------------------------------------------------------------------
# Family 4 -- the ADDITIVE NotesPanel embed
# ---------------------------------------------------------------------------


class TestNotesPanelEmbed:
    def test_import_added(self):
        s = _read(NOTES_PANEL)
        assert "import NotesDrawingCanvas from './NotesDrawingCanvas.svelte';" in s

    def test_rendered_with_active_note(self):
        s = _read(NOTES_PANEL)
        assert "<NotesDrawingCanvas noteId={$activeNote.id} />" in s

    def test_rendered_after_the_s253_pair(self):
        s = _read(NOTES_PANEL)
        gallery = s.find("<NotesMediaGallery")
        canvas = s.find("<NotesDrawingCanvas")
        assert gallery != -1 and canvas != -1
        assert gallery < canvas


# ---------------------------------------------------------------------------
# Family 5 -- the spec registration
# ---------------------------------------------------------------------------


class TestSpecRegistration:
    def test_drawing_canvas_registered(self):
        spec = _read(SPEC)
        assert (
            re.search(r"NotesDrawingCanvas\.svelte`?\s*\|\s*NEW\s*\|\s*S254", spec)
            is not None
        )


# ---------------------------------------------------------------------------
# Family 6 -- the host-assured runbook
# ---------------------------------------------------------------------------


class TestRunbook:
    def test_exists(self):
        assert RUNBOOK.exists()

    def test_title_and_status(self):
        f = _flat(_read(RUNBOOK))
        assert "NOTES_DRAWING_E2E_S254" in f
        assert "host-assured" in f
        assert "never" in f and "simulated" in f

    def test_extends_the_s253_runbook(self):
        f = _flat(_read(RUNBOOK))
        assert "NOTES_MEDIA_UI_E2E_S253.md" in f

    def test_live_walk_sections(self):
        f = _flat(_read(RUNBOOK))
        assert "draw / save / reload" in f
        assert "re-edit" in f

    def test_replace_discipline_stated(self):
        f = _flat(_read(RUNBOOK))
        assert "upload-before-remove" in f

    def test_ciphertext_spot_check(self):
        f = _flat(_read(RUNBOOK))
        assert "ciphertext" in f

    def test_findings_register_and_routing(self):
        f = _flat(_read(RUNBOOK))
        assert "Findings register" in f
        assert "Routing" in f

    def test_version_held(self):
        f = _flat(_read(RUNBOOK))
        assert "3.11.0" in f


# ---------------------------------------------------------------------------
# Family 7 -- the seams the canvas is the client of (green on pristine S253)
# ---------------------------------------------------------------------------


class TestSeams:
    def test_backend_allowlist_carries_drawing(self):
        s = _read(NOTES_STORE_PY)
        assert "ATTACHMENT_KINDS" in s
        assert '"drawing"' in s

    def test_route_validates_kind(self):
        s = _read(ROUTES_ATTACHMENTS)
        assert "kind not in ATTACHMENT_KINDS" in s

    def test_route_five_legs(self):
        s = _read(ROUTES_ATTACHMENTS)
        for leg in (
            "def upload_attachment",
            "def list_attachments",
            "def get_attachment_meta",
            "def download_attachment",
            "def delete_attachment",
        ):
            assert leg in s, leg

    def test_client_kind_union_carries_drawing(self):
        s = _read(ATTACH_CLIENT)
        assert "export type AttachmentKind" in s
        assert "'drawing'" in s

    def test_client_upload_and_blob_fetch(self):
        s = _read(ATTACH_CLIENT)
        assert "export async function uploadAttachment" in s
        assert "export async function fetchAttachmentBlob" in s
        assert "credentials: 'include'" in s

    def test_base_client_ships_api_upload(self):
        s = _read(BASE_CLIENT)
        assert "apiUpload" in s

    def test_store_premises(self):
        s = _read(ATTACH_STORE)
        for sym in (
            "export const attachments",
            "export async function loadAttachments",
            "export async function uploadNoteAttachment",
            "export async function removeAttachment",
            "derived",
        ):
            assert sym in s, sym

    def test_store_sibling_views_untouched(self):
        s = _read(ATTACH_STORE)
        assert "export const audioAttachments" in s
        assert "export const imageAttachments" in s

    def test_ds_exports(self):
        s = _read(DS_INDEX)
        for sym in ("Button", "Card", "Icon", "EmptyState", "InlineError"):
            assert sym in s, sym

    def test_panel_hosts_the_s253_pair(self):
        s = _read(NOTES_PANEL)
        assert "import NotesVoiceCapture from './NotesVoiceCapture.svelte';" in s
        assert "import NotesMediaGallery from './NotesMediaGallery.svelte';" in s
        assert "<NotesVoiceCapture noteId={$activeNote.id} />" in s
        assert "<NotesMediaGallery noteId={$activeNote.id} />" in s

    def test_panel_single_media_block(self):
        s = _read(NOTES_PANEL)
        assert s.count('class="notes-media"') == 1

    def test_s253_components_present(self):
        assert VOICE_CAPTURE.exists()
        assert MEDIA_GALLERY.exists()

    def test_s253_spec_rows_reasserted(self):
        spec = _read(SPEC)
        assert (
            re.search(r"NotesVoiceCapture\.svelte`?\s*\|\s*NEW\s*\|\s*S253", spec)
            is not None
        )
        assert (
            re.search(r"NotesMediaGallery\.svelte`?\s*\|\s*NEW\s*\|\s*S253", spec)
            is not None
        )

    def test_describe_settlement_recorded(self):
        # The read-gate settlement: caption stays image-gated; no drawing
        # widening lands in this slice (the orchestration guard is untouched).
        s = _read(ROOT / "opti_oignon" / "notes" / "caption.py")
        assert '!= "image"' in s


# ---------------------------------------------------------------------------
# Family 8 -- ASCII / structure / invariants
# ---------------------------------------------------------------------------


class TestStructure:
    def test_drawing_module_ascii(self):
        assert DRAWING_MODULE.exists()
        assert _read(DRAWING_MODULE).isascii()

    def test_canvas_ascii(self):
        assert CANVAS.exists()
        assert _read(CANVAS).isascii()

    def test_runbook_ascii(self):
        assert RUNBOOK.exists()
        assert _read(RUNBOOK).isascii()

    def test_pristine_panel_stays_balanced(self):
        assert _block_balanced(_read(NOTES_PANEL))

    def test_version_held(self):
        m = re.search(r'__version__\s*=\s*"([^"]+)"', _read(VERSION_FILE))
        assert m is not None
        assert m.group(1) == "3.11.0"

    def test_suite_parses(self):
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
