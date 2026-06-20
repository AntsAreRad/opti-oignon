#!/usr/bin/env python3
"""Notes media post-processing (N.5 voice): the opt-in, sandboxed whisper.cpp
transcription orchestration.

N.1 (S243) built the media data layer for all three kinds and S249 landed the
shared notes-attachment route over it; the ``attachment`` manifest carries a
``transcript_text`` column that stays NULL until a post-processing bloc fills it.
This module is that bloc for audio: it turns an encrypted audio attachment into a
transcript and writes it back through ``NotesStore.update_attachment``, but only
after the human approves, and only ever inside a fully isolated, disposable
bubblewrap sandbox.

The disposable-bubblewrap floor (S73 / S74) is non-negotiable here, because
transcription is file-touching post-processing of user content:

- FAIL-SECURE. If a real bubblewrap is not available (``sandbox.bwrap_available``
  is false), the orchestration REFUSES; it never falls back to a degraded
  tempdir-only mode for this work. An undeterminable isolation posture is a
  refusal, never a host-side run.
- DECRYPT IN MEMORY. The blob is decrypted via ``NotesBlobStore.open`` into a
  bytes value in memory; it is never written to a host temp file in transit.
- COPY THE BYTES INTO THE SANDBOX. The decrypted bytes are written into the
  disposable sandbox workspace (the sanctioned isolated location), not anywhere
  else on the host filesystem.
- RUN THE TOOL IN THE SANDBOX. whisper.cpp runs inside the sandbox, with zero
  host filesystem and zero network access (the sandbox's --unshare-net /
  workspace confinement). The tool run is an injected seam: the live builder
  wires whisper.cpp; tests inject a fake. whisper.cpp is absent in-container, so
  the live run is host-assured and never simulated here.
- HUMAN APPROVAL BEFORE THE DURABLE WRITE-BACK. The transcript is written to the
  manifest only when the caller passes ``approve=True``. Without approval the
  transcript is returned for review (a preview) but NOT persisted; the S116
  copy-out / approval discipline governs the durable result.
- DISPOSABLE TEARDOWN. The sandbox (and the plaintext audio inside it) is
  destroyed in a ``finally``, on every path after creation.

The note / transcript text is the user's own content; whisper.cpp is a
transcriber, not an instruction-following model, so there is no untrusted-context
wrapping here (that floor is N.3's, for the LLM-from-note actions). If a later
optional "describe" step ever hands media text to a model, it goes through
``agent.untrusted_context`` then.

``checkpoint_before_apply`` is hardcoded True and never overridable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

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


# Whether the opt-in whisper.cpp Python binding is importable. Off by default:
# the ``transcribe`` extra is not pulled by the base install, so this is False
# in-container, which is exactly why the live run is host-assured.
try:  # pragma: no cover - the binding is absent in-container by design
    import pywhispercpp  # type: ignore  # noqa: F401

    TRANSCRIBE_AVAILABLE = True
except Exception:
    TRANSCRIBE_AVAILABLE = False

# The filename the decrypted audio takes inside the disposable workspace.
DEFAULT_INPUT_NAME = "input.audio"

# Structured refusal / failure reasons (carried in the result body; the
# orchestration never raises -- ok / refused / a clean failure all cross the
# wire, the note_actions runner posture).
REASON_SANDBOX_UNAVAILABLE = "sandbox_unavailable"
REASON_TRANSCRIBER_UNAVAILABLE = "transcriber_unavailable"
REASON_NOT_FOUND = "not_found"
REASON_NOT_AUDIO = "not_audio"
REASON_BLOB_UNAVAILABLE = "blob_unavailable"
REASON_TRANSCRIPTION_FAILED = "transcription_failed"

# A transcriber is a callable over the sandbox handle: (sandbox, session_id,
# input_filename) -> transcript text. It runs the tool INSIDE the sandbox.
Transcriber = Callable[[Any, str, str], str]


@dataclass
class TranscriptionResult:
    """The structured outcome of a transcription request.

    ``ok`` True carries the ``transcript_text``; ``written_back`` records whether
    the transcript was persisted to the manifest (only on approval). ``refused``
    True marks a structured refusal (the fail-secure sandbox gate, a missing or
    non-audio attachment, an unavailable blob); any other failure is ``ok`` False
    with a ``reason`` and ``refused`` False. The orchestration never raises.
    """

    ok: bool
    attachment_id: str
    transcript_text: Optional[str] = None
    written_back: bool = False
    refused: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "attachment_id": self.attachment_id,
            "transcript_text": self.transcript_text,
            "written_back": self.written_back,
            "refused": self.refused,
            "reason": self.reason,
        }


def _refused(attachment_id: str, reason: str) -> TranscriptionResult:
    return TranscriptionResult(
        ok=False, attachment_id=attachment_id, refused=True, reason=reason
    )


def transcribe_attachment(
    attachment_id: str,
    *,
    user_id: Optional[str],
    store: Any,
    blobs: Any,
    sandbox: Any,
    transcriber: Optional[Transcriber],
    approve: bool = False,
    input_filename: str = DEFAULT_INPUT_NAME,
) -> TranscriptionResult:
    """Transcribe one audio attachment inside a disposable bubblewrap sandbox.

    Args:
        attachment_id: The audio attachment to transcribe.
        user_id: The owning user (per-user isolation; a cross-user id resolves to
            a structured ``not_found``, never a served transcript).
        store: A ``NotesStore`` (``get_attachment`` / ``update_attachment``).
        blobs: A ``NotesBlobStore`` (``open`` decrypts in memory).
        sandbox: A ``SandboxManager``-like seam (``bwrap_available``,
            ``create_sandbox`` / ``get_active_workspace_path`` /
            ``destroy_sandbox``). The live one is the real disposable sandbox.
        transcriber: The injected tool seam; ``None`` is a structured refusal
            (the opt-in binding is absent).
        approve: Human approval for the durable write-back. False returns the
            transcript as a preview without persisting it.
        input_filename: The filename the decrypted bytes take inside the sandbox.

    Returns:
        A :class:`TranscriptionResult`; the function never raises.
    """
    # Fail-secure on the disposable-bubblewrap floor: with no real bwrap, refuse.
    if sandbox is None or not getattr(sandbox, "bwrap_available", False):
        return _refused(attachment_id, REASON_SANDBOX_UNAVAILABLE)
    if transcriber is None:
        return _refused(attachment_id, REASON_TRANSCRIBER_UNAVAILABLE)

    # Per-user manifest fetch; the kind must be audio.
    record = store.get_attachment(attachment_id, user_id=user_id)
    if record is None:
        return _refused(attachment_id, REASON_NOT_FOUND)
    if getattr(record, "kind", "") != "audio":
        return _refused(attachment_id, REASON_NOT_AUDIO)

    # Decrypt the blob in memory; a no-key / primitive-less blob store refuses
    # rather than yielding plaintext, and that surfaces as a structured refusal.
    try:
        audio = blobs.open(attachment_id)
    except NotesBlobUnavailable:
        return _refused(attachment_id, REASON_BLOB_UNAVAILABLE)

    # Create the disposable sandbox; everything after is torn down in finally.
    session = sandbox.create_sandbox(
        session_id=None,
        label="notes-transcription",
        owner_user_id=user_id or "local",
    )
    session_id = getattr(session, "session_id", "") or ""
    try:
        workspace = Path(sandbox.get_active_workspace_path(session_id))
        # The decrypted bytes go INTO the disposable workspace, never a host temp.
        (workspace / input_filename).write_bytes(audio)
        try:
            raw = transcriber(sandbox, session_id, input_filename)
        except Exception:
            # Never log the audio; record only that the tool run failed.
            logger.warning("Transcription tool failed", exc_info=True)
            return TranscriptionResult(
                ok=False,
                attachment_id=attachment_id,
                refused=False,
                reason=REASON_TRANSCRIPTION_FAILED,
            )
        transcript = "" if raw is None else str(raw)
        if not approve:
            # Preview only: the transcript is returned for review, NOT persisted.
            return TranscriptionResult(
                ok=True,
                attachment_id=attachment_id,
                transcript_text=transcript,
                written_back=False,
            )
        # Human approved: durable write-back onto the manifest row (per-user).
        store.update_attachment(
            attachment_id, transcript_text=transcript, user_id=user_id
        )
        return TranscriptionResult(
            ok=True,
            attachment_id=attachment_id,
            transcript_text=transcript,
            written_back=True,
        )
    finally:
        # Dispose the sandbox (and the plaintext audio inside it).
        try:
            sandbox.destroy_sandbox(session_id)
        except Exception:  # pragma: no cover - defensive teardown
            logger.warning("Sandbox teardown failed for %s", session_id)


def build_live_transcriber(
    *,
    binary: str = "whisper-cli",
    extra_args: Optional[list[str]] = None,
) -> Optional[Transcriber]:
    """Build the live whisper.cpp transcriber, or None when the opt-in dep is off.

    HOST-ASSURED. whisper.cpp is absent in-container, so this returns None here;
    the live transcriber runs the binary INSIDE the disposable sandbox (zero host
    filesystem, zero network) over the injected audio, then reads the transcript
    out under the S116 approve / copy-out discipline. The exact whisper.cpp
    invocation is settled on the host (NOTES_TRANSCRIPTION_E2E_S250.md); it is
    never simulated in the container.
    """
    if not TRANSCRIBE_AVAILABLE:
        return None

    args = list(extra_args or [])

    def _run(sandbox: Any, session_id: str, input_filename: str) -> str:
        # Run whisper.cpp inside the sandbox; the output text file is produced in
        # the workspace, approved, and copied out. The command is assembled from
        # the fixed binary name and the in-workspace input filename (no host path
        # ever leaves the sandbox boundary).
        out_stem = "transcript"
        cmd = " ".join(
            [binary, "-f", input_filename, "-otxt", "-of", out_stem, *args]
        )
        result = sandbox.execute_command(session_id, cmd)
        if getattr(result, "return_code", 1) != 0:
            raise RuntimeError("whisper.cpp returned non-zero")
        out_name = out_stem + ".txt"
        sandbox.approve_files(session_id, [out_name])
        workspace = Path(sandbox.get_active_workspace_path(session_id))
        return (workspace / out_name).read_text(encoding="utf-8", errors="replace")

    return _run
