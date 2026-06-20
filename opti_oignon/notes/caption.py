#!/usr/bin/env python3
"""Notes media post-processing (N.6 picture caption / OCR): the opt-in, sandboxed
vision caption / OCR orchestration.

N.1 (S243) built the media data layer for all three kinds and S249 landed the
shared notes-attachment route over it; the ``attachment`` manifest carries
``caption_text`` and ``ocr_text`` columns that stay NULL until a post-processing
bloc fills them. S250 landed the audio sibling (``transcription.py``) and the
additive ``NotesStore.update_attachment`` write-back, whose caption / ocr legs
this module reuses unchanged. This module is the post-processing bloc for images:
it turns an encrypted image attachment into a caption and/or OCR text and writes
the produced legs back through ``NotesStore.update_attachment``, but only after
the human approves, and only ever inside a fully isolated, disposable bubblewrap
sandbox.

It is a sibling of ``transcription.py`` (the same orchestration shape), NOT a
second function bolted onto it: a separate module mirrors the separate trigger
router, keeping each media concern independent.

The disposable-bubblewrap floor (S73 / S74) is non-negotiable here, because
caption / OCR is file-touching post-processing of user content:

- FAIL-SECURE. If a real bubblewrap is not available (``sandbox.bwrap_available``
  is false), the orchestration REFUSES; it never falls back to a degraded
  tempdir-only mode for this work. An undeterminable isolation posture is a
  refusal, never a host-side run.
- DECRYPT IN MEMORY. The blob is decrypted via ``NotesBlobStore.open`` into a
  bytes value in memory; it is never written to a host temp file in transit.
- COPY THE BYTES INTO THE SANDBOX. The decrypted bytes are written into the
  disposable sandbox workspace (the sanctioned isolated location), not anywhere
  else on the host filesystem.
- RUN THE TOOL IN THE SANDBOX. The vision/OCR tool runs inside the sandbox, with
  zero host filesystem and zero network access (the sandbox's --unshare-net /
  workspace confinement). The tool run is an injected seam: the live builder
  wires the vision/OCR binding; tests inject a fake. The vision tooling is absent
  in-container, so the live run is host-assured and never simulated here.
- HUMAN APPROVAL BEFORE THE DURABLE WRITE-BACK. The derived text is written to
  the manifest only when the caller passes ``approve=True``. Without approval the
  result is returned for review (a preview) but NOT persisted; the S116 copy-out
  / approval discipline governs the durable result.
- DISPOSABLE TEARDOWN. The sandbox (and the plaintext image inside it) is
  destroyed in a ``finally``, on every path after creation.

The captioner returns a ``(caption_text, ocr_text)`` pair (either leg may be
``None`` when the tool did not produce it). On approval ONLY the produced legs
are written, so a leg the tool did not produce never blanks an existing value
(the ``update_attachment`` no-blank property). The image content is the user's
own content; the OCR/vision tool is a transcriber-of-pixels, not an
instruction-following model, so there is no untrusted-context wrapping here (that
floor is N.3's, for the LLM-from-note actions). If a later optional model-driven
"describe" step ever hands media text to a model, it goes through
``agent.untrusted_context`` then.

``checkpoint_before_apply`` is hardcoded True and never overridable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# Hardcoded checkpoint discipline for every new module; never overridable.
checkpoint_before_apply = True

# The module imported; the route's availability guard reads this.
FEATURE_AVAILABLE = True

try:
    from .blob_store import NotesBlobUnavailable
except Exception:  # pragma: no cover - constrained environments

    class NotesBlobUnavailable(RuntimeError):  # type: ignore[no-redef]
        """Fallback when blob_store is unavailable at import."""


# Whether the opt-in vision/OCR Python binding is importable. Off by default:
# the ``vision`` extra is not pulled by the base install, so this is False
# in-container, which is exactly why the live run is host-assured.
try:  # pragma: no cover - the binding is absent in-container by design
    import pytesseract  # type: ignore  # noqa: F401

    VISION_AVAILABLE = True
except Exception:
    VISION_AVAILABLE = False

# The filename the decrypted image takes inside the disposable workspace.
DEFAULT_INPUT_NAME = "input.image"

# Structured refusal / failure reasons (carried in the result body; the
# orchestration never raises -- ok / refused / a clean failure all cross the
# wire, the transcription / note_actions runner posture).
REASON_SANDBOX_UNAVAILABLE = "sandbox_unavailable"
REASON_CAPTIONER_UNAVAILABLE = "captioner_unavailable"
REASON_NOT_FOUND = "not_found"
REASON_NOT_IMAGE = "not_image"
REASON_BLOB_UNAVAILABLE = "blob_unavailable"
REASON_CAPTION_FAILED = "caption_failed"

# A captioner is a callable over the sandbox handle: (sandbox, session_id,
# input_filename) -> (caption_text, ocr_text). It runs the tool INSIDE the
# sandbox. Either leg may be None when the tool did not produce it.
Captioner = Callable[[Any, str, str], Tuple[Optional[str], Optional[str]]]


@dataclass
class CaptionResult:
    """The structured outcome of a caption / OCR request.

    ``ok`` True carries ``caption_text`` and/or ``ocr_text`` (whichever the tool
    produced); ``written_back`` records whether any leg was persisted to the
    manifest (only on approval, and only the produced legs). ``refused`` True
    marks a structured refusal (the fail-secure sandbox gate, a missing or
    non-image attachment, an unavailable blob, an absent captioner); any other
    failure is ``ok`` False with a ``reason`` and ``refused`` False. The
    orchestration never raises.
    """

    ok: bool
    attachment_id: str
    caption_text: Optional[str] = None
    ocr_text: Optional[str] = None
    written_back: bool = False
    refused: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "attachment_id": self.attachment_id,
            "caption_text": self.caption_text,
            "ocr_text": self.ocr_text,
            "written_back": self.written_back,
            "refused": self.refused,
            "reason": self.reason,
        }


def _refused(attachment_id: str, reason: str) -> CaptionResult:
    return CaptionResult(
        ok=False, attachment_id=attachment_id, refused=True, reason=reason
    )


def _coerce(raw: Any) -> Tuple[Optional[str], Optional[str]]:
    """Coerce the captioner's return into a ``(caption, ocr)`` pair of str|None.

    A 2-tuple/list maps directly (each leg str|None); a bare string is taken as
    the caption with no OCR; None is both-None. This keeps the orchestration
    leg-agnostic so a caption-only, OCR-only, or both tool all fit.
    """
    if raw is None:
        return None, None
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        cap, ocr = raw
        cap = None if cap is None else str(cap)
        ocr = None if ocr is None else str(ocr)
        return cap, ocr
    return str(raw), None


def caption_attachment(
    attachment_id: str,
    *,
    user_id: Optional[str],
    store: Any,
    blobs: Any,
    sandbox: Any,
    captioner: Optional[Captioner],
    approve: bool = False,
    input_filename: str = DEFAULT_INPUT_NAME,
) -> CaptionResult:
    """Caption / OCR one image attachment inside a disposable bubblewrap sandbox.

    Args:
        attachment_id: The image attachment to caption / OCR.
        user_id: The owning user (per-user isolation; a cross-user id resolves to
            a structured ``not_found``, never a served result).
        store: A ``NotesStore`` (``get_attachment`` / ``update_attachment``).
        blobs: A ``NotesBlobStore`` (``open`` decrypts in memory).
        sandbox: A ``SandboxManager``-like seam (``bwrap_available``,
            ``create_sandbox`` / ``get_active_workspace_path`` /
            ``destroy_sandbox``). The live one is the real disposable sandbox.
        captioner: The injected tool seam; ``None`` is a structured refusal (the
            opt-in binding is absent).
        approve: Human approval for the durable write-back. False returns the
            result as a preview without persisting it.
        input_filename: The filename the decrypted bytes take inside the sandbox.

    Returns:
        A :class:`CaptionResult`; the function never raises.
    """
    # Fail-secure on the disposable-bubblewrap floor: with no real bwrap, refuse.
    if sandbox is None or not getattr(sandbox, "bwrap_available", False):
        return _refused(attachment_id, REASON_SANDBOX_UNAVAILABLE)
    if captioner is None:
        return _refused(attachment_id, REASON_CAPTIONER_UNAVAILABLE)

    # Per-user manifest fetch; the kind must be image.
    record = store.get_attachment(attachment_id, user_id=user_id)
    if record is None:
        return _refused(attachment_id, REASON_NOT_FOUND)
    if getattr(record, "kind", "") != "image":
        return _refused(attachment_id, REASON_NOT_IMAGE)

    # Decrypt the blob in memory; a no-key / primitive-less blob store refuses
    # rather than yielding plaintext, and that surfaces as a structured refusal.
    try:
        image = blobs.open(attachment_id)
    except NotesBlobUnavailable:
        return _refused(attachment_id, REASON_BLOB_UNAVAILABLE)

    # Create the disposable sandbox; everything after is torn down in finally.
    session = sandbox.create_sandbox(
        session_id=None,
        label="notes-caption",
        owner_user_id=user_id or "local",
    )
    session_id = getattr(session, "session_id", "") or ""
    try:
        workspace = Path(sandbox.get_active_workspace_path(session_id))
        # The decrypted bytes go INTO the disposable workspace, never a host temp.
        (workspace / input_filename).write_bytes(image)
        try:
            raw = captioner(sandbox, session_id, input_filename)
        except Exception:
            # Never log the image; record only that the tool run failed.
            logger.warning("Caption/OCR tool failed", exc_info=True)
            return CaptionResult(
                ok=False,
                attachment_id=attachment_id,
                refused=False,
                reason=REASON_CAPTION_FAILED,
            )
        caption, ocr = _coerce(raw)
        if not approve:
            # Preview only: the legs are returned for review, NOT persisted.
            return CaptionResult(
                ok=True,
                attachment_id=attachment_id,
                caption_text=caption,
                ocr_text=ocr,
                written_back=False,
            )
        # Human approved: durable write-back of ONLY the produced legs (per-user).
        # An omitted (None) leg is never passed, so it cannot blank an existing
        # value (the update_attachment no-blank property).
        legs: dict[str, Any] = {}
        if caption is not None:
            legs["caption_text"] = caption
        if ocr is not None:
            legs["ocr_text"] = ocr
        written = False
        if legs:
            store.update_attachment(attachment_id, user_id=user_id, **legs)
            written = True
        return CaptionResult(
            ok=True,
            attachment_id=attachment_id,
            caption_text=caption,
            ocr_text=ocr,
            written_back=written,
        )
    finally:
        # Dispose the sandbox (and the plaintext image inside it).
        try:
            sandbox.destroy_sandbox(session_id)
        except Exception:  # pragma: no cover - defensive teardown
            logger.warning("Sandbox teardown failed for %s", session_id)


def build_live_captioner(
    *,
    binary: str = "tesseract",
    extra_args: Optional[list[str]] = None,
) -> Optional[Captioner]:
    """Build the live vision/OCR captioner, or None when the opt-in dep is off.

    HOST-ASSURED. The vision/OCR tooling is absent in-container, so this returns
    None here; the live captioner runs the tool INSIDE the disposable sandbox
    (zero host filesystem, zero network) over the injected image, then reads the
    OCR text out under the S116 approve / copy-out discipline. The sandboxed core
    produces the OCR leg; the optional model-driven "describe" caption leg (via
    the existing local vision pipeline) is a host-assured refinement and is not
    the sandboxed file-touching step. The exact tool invocation is settled on the
    host (NOTES_CAPTION_E2E_S251.md); it is never simulated in the container.
    """
    if not VISION_AVAILABLE:
        return None

    args = list(extra_args or [])

    def _run(
        sandbox: Any, session_id: str, input_filename: str
    ) -> Tuple[Optional[str], Optional[str]]:
        # Run the OCR tool inside the sandbox; the output text file is produced in
        # the workspace, approved, and copied out. The command is assembled from
        # the fixed binary name and the in-workspace input/output names (no host
        # path ever leaves the sandbox boundary).
        out_stem = "ocr"
        cmd = " ".join([binary, input_filename, out_stem, *args])
        result = sandbox.execute_command(session_id, cmd)
        if getattr(result, "return_code", 1) != 0:
            raise RuntimeError("vision/OCR tool returned non-zero")
        out_name = out_stem + ".txt"
        sandbox.approve_files(session_id, [out_name])
        workspace = Path(sandbox.get_active_workspace_path(session_id))
        ocr_text = (workspace / out_name).read_text(
            encoding="utf-8", errors="replace"
        )
        # The caption (describe) leg is host-assured / wired separately; the
        # sandboxed core produces the OCR leg.
        return (None, ocr_text)

    return _run
