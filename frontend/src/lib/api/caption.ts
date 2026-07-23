/**
 * Typed API client for the opt-in image caption / OCR trigger (N.6).
 *
 * One per-user endpoint, POST /api/notes/caption/{attachment_id} (the
 * route). The run is opt-in and sandboxed server-side (the disposable
 * bubblewrap floor); `approve` gates the durable write-back and its default
 * is the safe one (false: preview, no persist). Every outcome -- ok, a
 * structured refusal (the fail-secure sandbox gate, the absent opt-in vision
 * extra, a missing or non-image attachment), or a clean failure -- crosses
 * the wire as a structured result with HTTP 200; the only HTTP error is the
 * 503 availability guard, which the store turns into mediaError.
 */

import { apiPost } from './client';

/** The structured outcome (mirrors CaptionResultSchema). ok true carries
 * caption_text and/or ocr_text (whichever the tool produced); written_back
 * records whether any produced leg was persisted (only on approval); refused
 * true marks a structured refusal with a reason. */
export interface CaptionResult {
	attachment_id: string;
	ok: boolean;
	caption_text: string | null;
	ocr_text: string | null;
	written_back: boolean;
	refused: boolean;
	reason: string;
}

/**
 * Trigger the caption / OCR of one image attachment.
 *
 * approve=false (the default) previews the text without persisting;
 * approve=true is the human approval that writes it back to the manifest.
 */
export async function requestCaption(
	attachmentId: string,
	approve: boolean = false
): Promise<CaptionResult> {
	return apiPost<CaptionResult>(
		`/api/notes/caption/${encodeURIComponent(attachmentId)}`,
		{ approve }
	);
}
