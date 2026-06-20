#!/usr/bin/env python3
"""S253 -- the Notes feature's N.5 / N.6 UI halves: the first attachment /
transcription / caption frontend client (the SvelteKit lot), the
container-provable half of a labelled Playwright/runbook lot (the s248
precedent).

S253 lands the frontend the user drives over the already-built media backend
(S249 attachments route, S250 transcription bloc, S251 caption bloc): three
typed API clients (api/attachments.ts over /api/notes/attachments,
api/transcription.ts over /api/notes/transcription, api/caption.ts over
/api/notes/caption), an attachments store (stores/attachments.ts), the
NotesVoiceCapture component (MediaRecorder capture, encrypted upload, opt-in
transcription with preview-then-approve), the NotesMediaGallery component
(image picker, encrypted upload, in-memory thumbnails, opt-in caption / OCR
with preview-then-approve), both embedded additively in NotesPanel's editor,
each new Svelte component recorded in FRONTEND_REDESIGN_SPEC.md. The live
browser run is host-assured (NOTES_MEDIA_UI_E2E_S253.md); this suite proves
the container-provable artifacts.

node_modules is absent here, so the frontend is checked by file content, a
structural Svelte block-balance pass, and the --oo-* token discipline (no raw
hex) -- the s174/s248 idiom -- not by a browser. Seven families:

 1. The TS clients and the store -- existence, the base-client wiring, the
    interfaces mirroring the backend schemas, the endpoints, the store's
    writables / derived / actions.
 2. The components -- NotesVoiceCapture and NotesMediaGallery: the
    ds-primitive imports, the client/store wiring, the behaviour surface
    (capture, upload, preview-then-approve, refusal display, object-URL
    hygiene), the --oo-* tokens (no raw hex), and Svelte block balance.
 3. The NotesPanel embed -- the two components imported and rendered in the
    editor pane, fed the active note id (an ADDITIVE edit: every S248
    presence pin on NotesPanel stays green by construction).
 4. The spec registration -- the NotesMediaGallery.svelte and
    NotesVoiceCapture.svelte NEW/S253 rows (the s174 registration regex),
    which also keeps the Theme 1 cartography invariant green.
 5. The runbook -- NOTES_MEDIA_UI_E2E_S253.md: existence, status
    (host-assured, findings-not-fixes, never-simulated), the
    container-vs-host split, the required sections, the companions, the
    version held, and pure ASCII.
 6. The seams the UI is the client of -- source pins on the premises (the
    S249 attachments route and its five legs, the S250 / S251 trigger routes
    and their approve gate, the attachment / result schemas, the base HTTP
    client incl. the S211 apiUpload multipart helper, the ds primitives, and
    the S248 NotesPanel surface this lot embeds into), so a later edit that
    removes a premise turns this suite red instead of letting the UI rot.
    Green on the pristine S252 tree by design.
 7. ASCII / structure -- the new TS and Svelte files are pure ASCII and
    block-balanced, and this suite parses.

Red-before discipline: on the pristine S252 tree (no media frontend source, no
spec rows, no runbook) every family-1..5 and family-7 structural pin FAILS --
the read helpers return empty strings so absence is a failure, never a
collection error -- while every family-6 seam pin passes (it pins pre-existing
premises this lot relies on). Document pins read through a
whitespace-flattening helper (the s221/s238 lesson) so reflow cannot break
them; source pins stay raw. This suite imports no package code, so no ollama
chain is touched.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend" / "src"

# New frontend source landed by S253.
ATTACH_CLIENT = FE / "lib" / "api" / "attachments.ts"
TRANSCRIBE_CLIENT = FE / "lib" / "api" / "transcription.ts"
CAPTION_CLIENT = FE / "lib" / "api" / "caption.ts"
ATTACH_STORE = FE / "lib" / "stores" / "attachments.ts"
VOICE_CAPTURE = FE / "lib" / "components" / "panels" / "NotesVoiceCapture.svelte"
MEDIA_GALLERY = FE / "lib" / "components" / "panels" / "NotesMediaGallery.svelte"

NEW_TS = (ATTACH_CLIENT, TRANSCRIBE_CLIENT, CAPTION_CLIENT, ATTACH_STORE)
NEW_SVELTE = (VOICE_CAPTURE, MEDIA_GALLERY)

# Edited additively by S253.
NOTES_PANEL = FE / "lib" / "components" / "panels" / "NotesPanel.svelte"

# Docs.
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
RUNBOOK = ROOT / "NOTES_MEDIA_UI_E2E_S253.md"

# Seam premises (pre-existing; green on the pristine S252 tree by design).
ROUTES_ATTACHMENTS = ROOT / "opti_oignon" / "api" / "routes_notes_attachments.py"
ROUTES_TRANSCRIPTION = ROOT / "opti_oignon" / "api" / "routes_notes_transcription.py"
ROUTES_CAPTION = ROOT / "opti_oignon" / "api" / "routes_notes_caption.py"
SCHEMAS = ROOT / "opti_oignon" / "api" / "schemas.py"
NOTES_STORE_PY = ROOT / "opti_oignon" / "notes" / "notes_store.py"
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


class TestAttachmentsClient:
    def test_exists(self):
        assert ATTACH_CLIENT.exists()

    def test_imports_base_client(self):
        s = _read(ATTACH_CLIENT)
        assert "from './client'" in s
        for sym in ("apiGet", "apiDelete", "apiUpload", "getAccessToken", "ApiError"):
            assert sym in s, sym

    def test_kind_union_matches_backend_allowlist(self):
        s = _read(ATTACH_CLIENT)
        assert "export type AttachmentKind" in s
        for kind in ("'audio'", "'image'", "'drawing'"):
            assert kind in s, kind

    def test_record_interface_mirrors_schema(self):
        s = _read(ATTACH_CLIENT)
        assert "export interface AttachmentRecord" in s
        for field in (
            "id",
            "note_id",
            "kind",
            "mime",
            "byte_size",
            "nonce",
            "created_at",
            "transcript_text",
            "caption_text",
            "ocr_text",
        ):
            assert field in s, field

    def test_endpoints_and_functions(self):
        s = _read(ATTACH_CLIENT)
        for fn in (
            "uploadAttachment",
            "listAttachments",
            "getAttachmentMeta",
            "fetchAttachmentBlob",
            "deleteAttachment",
        ):
            assert fn in s, fn
        assert "/api/notes/attachments/note/" in s
        assert "/api/notes/attachments/" in s
        assert "/blob" in s

    def test_upload_is_multipart_formdata(self):
        s = _read(ATTACH_CLIENT)
        assert "new FormData()" in s
        assert "append('kind'" in s
        assert "append('file'" in s
        assert "apiUpload" in s

    def test_blob_fetch_is_authed_and_binary(self):
        s = _read(ATTACH_CLIENT)
        assert "getAccessToken()" in s
        assert "Authorization" in s
        assert "Bearer" in s
        assert ".blob()" in s
        assert "credentials: 'include'" in s


class TestTranscriptionClient:
    def test_exists(self):
        assert TRANSCRIBE_CLIENT.exists()

    def test_result_interface_mirrors_schema(self):
        s = _read(TRANSCRIBE_CLIENT)
        assert "export interface TranscriptionResult" in s
        for field in (
            "attachment_id",
            "ok",
            "transcript_text",
            "written_back",
            "refused",
            "reason",
        ):
            assert field in s, field

    def test_trigger_function_and_endpoint(self):
        s = _read(TRANSCRIBE_CLIENT)
        assert "export async function requestTranscription" in s
        assert "from './client'" in s
        assert "apiPost" in s
        assert "/api/notes/transcription/" in s
        assert "approve" in s

    def test_approve_defaults_safe(self):
        s = _read(TRANSCRIBE_CLIENT)
        assert "approve: boolean = false" in s


class TestCaptionClient:
    def test_exists(self):
        assert CAPTION_CLIENT.exists()

    def test_result_interface_mirrors_schema(self):
        s = _read(CAPTION_CLIENT)
        assert "export interface CaptionResult" in s
        for field in (
            "attachment_id",
            "ok",
            "caption_text",
            "ocr_text",
            "written_back",
            "refused",
            "reason",
        ):
            assert field in s, field

    def test_trigger_function_and_endpoint(self):
        s = _read(CAPTION_CLIENT)
        assert "export async function requestCaption" in s
        assert "from './client'" in s
        assert "apiPost" in s
        assert "/api/notes/caption/" in s
        assert "approve" in s

    def test_approve_defaults_safe(self):
        s = _read(CAPTION_CLIENT)
        assert "approve: boolean = false" in s


class TestAttachmentsStore:
    def test_exists(self):
        assert ATTACH_STORE.exists()

    def test_imports_clients(self):
        s = _read(ATTACH_STORE)
        assert "from '$lib/api/attachments'" in s
        assert "from '$lib/api/transcription'" in s
        assert "from '$lib/api/caption'" in s

    def test_stores_and_derived(self):
        s = _read(ATTACH_STORE)
        for store in ("attachments", "attachmentsNoteId", "mediaLoading", "mediaError"):
            assert store in s, store
        for view in ("audioAttachments", "imageAttachments"):
            assert view in s, view
        assert "writable" in s
        assert "derived" in s

    def test_actions(self):
        s = _read(ATTACH_STORE)
        for action in (
            "loadAttachments",
            "uploadNoteAttachment",
            "removeAttachment",
            "transcribeAttachment",
            "captionAttachment",
            "clearAttachments",
        ):
            assert action in s, action

    def test_writeback_updates_manifest_row(self):
        # An approved transcription / caption updates the record in the list,
        # so the UI reflects the persisted text without a reload.
        s = _read(ATTACH_STORE)
        assert "written_back" in s
        assert "transcript_text" in s
        assert "caption_text" in s


# ---------------------------------------------------------------------------
# Family 2 -- the components
# ---------------------------------------------------------------------------


class TestVoiceCapture:
    def test_exists(self):
        assert VOICE_CAPTURE.exists()

    def test_imports_ds_primitives(self):
        s = _read(VOICE_CAPTURE)
        assert "from '$lib/ds'" in s
        for prim in ("Button", "Card", "Icon", "EmptyState", "InlineError"):
            assert prim in s, prim

    def test_wires_store_and_client(self):
        s = _read(VOICE_CAPTURE)
        assert "from '$lib/stores/attachments'" in s
        for sym in (
            "audioAttachments",
            "loadAttachments",
            "uploadNoteAttachment",
            "removeAttachment",
            "transcribeAttachment",
        ):
            assert sym in s, sym
        assert "from '$lib/api/attachments'" in s
        assert "fetchAttachmentBlob" in s

    def test_behaviour_surface(self):
        s = _read(VOICE_CAPTURE)
        assert "export let noteId" in s
        for needle in (
            "MediaRecorder",
            "getUserMedia",
            "async function startRecording",
            "function stopRecording",
            "async function transcribe",
            "async function approveTranscript",
            "result.refused",
        ):
            assert needle in s, needle

    def test_object_url_hygiene(self):
        s = _read(VOICE_CAPTURE)
        assert "createObjectURL" in s
        assert "revokeObjectURL" in s

    def test_no_raw_hex(self):
        raw = _read(VOICE_CAPTURE)
        assert raw, "NotesVoiceCapture.svelte absent"
        assert not _has_raw_hex(raw)

    def test_block_balanced(self):
        raw = _read(VOICE_CAPTURE)
        assert raw, "NotesVoiceCapture.svelte absent"
        assert _block_balanced(raw)


class TestMediaGallery:
    def test_exists(self):
        assert MEDIA_GALLERY.exists()

    def test_imports_ds_primitives(self):
        s = _read(MEDIA_GALLERY)
        assert "from '$lib/ds'" in s
        for prim in ("Button", "Card", "Icon", "EmptyState", "InlineError"):
            assert prim in s, prim

    def test_wires_store_and_client(self):
        s = _read(MEDIA_GALLERY)
        assert "from '$lib/stores/attachments'" in s
        for sym in (
            "imageAttachments",
            "loadAttachments",
            "uploadNoteAttachment",
            "removeAttachment",
            "captionAttachment",
        ):
            assert sym in s, sym
        assert "from '$lib/api/attachments'" in s
        assert "fetchAttachmentBlob" in s

    def test_behaviour_surface(self):
        s = _read(MEDIA_GALLERY)
        assert "export let noteId" in s
        for needle in (
            'accept="image/*"',
            "async function uploadImages",
            "async function describe",
            "async function approveCaption",
            "result.refused",
        ):
            assert needle in s, needle

    def test_object_url_hygiene(self):
        s = _read(MEDIA_GALLERY)
        assert "createObjectURL" in s
        assert "revokeObjectURL" in s

    def test_no_raw_hex(self):
        raw = _read(MEDIA_GALLERY)
        assert raw, "NotesMediaGallery.svelte absent"
        assert not _has_raw_hex(raw)

    def test_block_balanced(self):
        raw = _read(MEDIA_GALLERY)
        assert raw, "NotesMediaGallery.svelte absent"
        assert _block_balanced(raw)


# ---------------------------------------------------------------------------
# Family 3 -- the NotesPanel embed (additive)
# ---------------------------------------------------------------------------


class TestPanelEmbed:
    def test_imports_both_components(self):
        s = _read(NOTES_PANEL)
        assert "import NotesVoiceCapture from './NotesVoiceCapture.svelte'" in s
        assert "import NotesMediaGallery from './NotesMediaGallery.svelte'" in s

    def test_renders_both_with_note_id(self):
        s = _read(NOTES_PANEL)
        assert "<NotesVoiceCapture" in s
        assert "<NotesMediaGallery" in s
        assert "noteId={$activeNote.id}" in s


# ---------------------------------------------------------------------------
# Family 4 -- the spec registration (cartography invariant)
# ---------------------------------------------------------------------------


class TestSpecRegistration:
    def test_voice_capture_registered(self):
        spec = _read(SPEC)
        assert (
            re.search(r"NotesVoiceCapture\.svelte`?\s*\|\s*NEW\s*\|\s*S253", spec)
            is not None
        )

    def test_media_gallery_registered(self):
        spec = _read(SPEC)
        assert (
            re.search(r"NotesMediaGallery\.svelte`?\s*\|\s*NEW\s*\|\s*S253", spec)
            is not None
        )


# ---------------------------------------------------------------------------
# Family 5 -- the host-assured browser runbook
# ---------------------------------------------------------------------------


class TestRunbook:
    def test_exists_and_titled(self):
        text = _read(RUNBOOK)
        assert text.startswith("# NOTES_MEDIA_UI_E2E_S253")
        assert len(text) > 4000

    def test_status_and_discipline(self):
        text = _flat(_read(RUNBOOK))
        assert "written at S253" in text
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
            "Voice capture",
            "Image gallery",
            "preview-then-approve",
            "MediaRecorder",
            "opt-in",
            "Findings register",
            "Routing",
        ):
            assert needle in text, needle

    def test_companions_named(self):
        text = _flat(_read(RUNBOOK))
        for needle in (
            "NOTES_UI_E2E_S248.md",
            "NOTES_FEATURE_ROADMAP.md",
            "FRONTEND_REDESIGN_SPEC.md",
            "routes_notes_attachments.py",
            "routes_notes_transcription.py",
            "routes_notes_caption.py",
        ):
            assert needle in text, needle

    def test_sandbox_floor_named(self):
        # The live transcription / caption runs stay on the disposable
        # bubblewrap floor; the runbook must say so.
        text = _flat(_read(RUNBOOK))
        assert "disposable" in text
        assert "bubblewrap" in text

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


class TestSeamAttachmentsRoute:
    def test_router_and_five_legs(self):
        src = _read(ROUTES_ATTACHMENTS)
        assert "notes_attachments_router" in src
        assert 'prefix="/api/notes/attachments"' in src
        assert '@notes_attachments_router.post("/note/{note_id}"' in src
        assert '@notes_attachments_router.get("/note/{note_id}"' in src
        assert '@notes_attachments_router.get("/{attachment_id}"' in src
        assert '@notes_attachments_router.get("/{attachment_id}/blob")' in src
        assert "@notes_attachments_router.delete(" in src

    def test_upload_contract(self):
        src = _read(ROUTES_ATTACHMENTS)
        assert "UploadFile" in src
        assert "Form(...)" in src
        assert "ATTACHMENT_KINDS" in src

    def test_blob_decrypts_in_memory_only(self):
        src = _read(ROUTES_ATTACHMENTS)
        assert "in memory" in src
        assert "no" in src and "plaintext temp file" in src


class TestSeamTranscriptionRoute:
    def test_router_and_trigger(self):
        src = _read(ROUTES_TRANSCRIPTION)
        assert "notes_transcription_router" in src
        assert 'prefix="/api/notes/transcription"' in src
        assert "@notes_transcription_router.post(" in src
        assert "TranscriptionResultSchema" in src
        assert "approve" in src


class TestSeamCaptionRoute:
    def test_router_and_trigger(self):
        src = _read(ROUTES_CAPTION)
        assert "notes_caption_router" in src
        assert 'prefix="/api/notes/caption"' in src
        assert "@notes_caption_router.post(" in src
        assert "CaptionResultSchema" in src
        assert "approve" in src


class TestSeamSchemas:
    def test_attachment_schema_fields(self):
        src = _read(SCHEMAS)
        assert "class AttachmentSchema" in src
        for field in (
            "note_id",
            "kind",
            "byte_size",
            "transcript_text",
            "caption_text",
            "ocr_text",
        ):
            assert field in src, field

    def test_result_schemas_fields(self):
        src = _read(SCHEMAS)
        assert "class TranscriptionResultSchema" in src
        assert "class CaptionResultSchema" in src
        for field in ("written_back", "refused", "reason"):
            assert field in src, field

    def test_kind_allowlist_is_three(self):
        src = _read(NOTES_STORE_PY)
        assert (
            'ATTACHMENT_KINDS: frozenset[str] = frozenset({"audio", "image", "drawing"})'
            in src
        )


class TestSeamBaseClient:
    def test_http_and_upload_helpers(self):
        src = _read(BASE_CLIENT)
        for fn in ("apiGet", "apiPost", "apiDelete", "apiUpload"):
            assert f"function {fn}" in src, fn
        assert "FormData" in src
        assert "function getAccessToken" in src
        assert "class ApiError" in src


class TestSeamDsPrimitives:
    def test_ds_exports(self):
        src = _read(DS_INDEX)
        for prim in ("Button", "Card", "Icon", "EmptyState", "InlineError"):
            assert prim in src, prim


class TestSeamNotesPanelS248:
    def test_s248_surface_intact(self):
        # The surface this lot embeds into; an additive edit must keep it.
        s = _read(NOTES_PANEL)
        assert "import NoteActionPanel from './NoteActionPanel.svelte'" in s
        assert "<NoteActionPanel" in s
        assert "bind:this={bodyEl}" in s
        assert "from '$lib/stores/notes'" in s

    def test_s248_spec_rows_intact(self):
        spec = _read(SPEC)
        assert re.search(r"NotesPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S248", spec) is not None
        assert (
            re.search(r"NoteActionPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S248", spec)
            is not None
        )


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

    def test_edited_panel_still_balanced(self):
        raw = _read(NOTES_PANEL)
        assert raw != ""
        assert _block_balanced(raw)
        assert not _has_raw_hex(raw)

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)
