/**
 * Typed API client for the opt-in voice transcription trigger (N.5, S253).
 *
 * One per-user endpoint, POST /api/notes/transcription/{attachment_id} (the
 * S250 route). The run is opt-in and sandboxed server-side (the disposable
 * bubblewrap floor); `approve` gates the durable write-back and its default
 * is the safe one (false: preview, no persist). Every outcome -- ok, a
 * structured refusal (the fail-secure sandbox gate, the absent opt-in
 * transcribe extra, a missing or non-audio attachment), or a clean failure --
 * crosses the wire as a structured result with HTTP 200; the only HTTP error
 * is the 503 availability guard, which the store turns into mediaError.
 */

import { apiPost } from './client';

/** The structured outcome (mirrors TranscriptionResultSchema). ok true
 * carries transcript_text; written_back records whether it was persisted
 * (only on approval); refused true marks a structured refusal with a reason. */
export interface TranscriptionResult {
	attachment_id: string;
	ok: boolean;
	transcript_text: string | null;
	written_back: boolean;
	refused: boolean;
	reason: string;
}

/**
 * Trigger the transcription of one audio attachment.
 *
 * approve=false (the default) previews the transcript without persisting;
 * approve=true is the human approval that writes it back to the manifest.
 */
export async function requestTranscription(
	attachmentId: string,
	approve: boolean = false
): Promise<TranscriptionResult> {
	return apiPost<TranscriptionResult>(
		`/api/notes/transcription/${encodeURIComponent(attachmentId)}`,
		{ approve }
	);
}
