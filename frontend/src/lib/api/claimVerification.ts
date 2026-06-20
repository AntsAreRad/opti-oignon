/**
 * Typed API client for the claim-vs-source verification role (S269 UI half).
 *
 * One per-user endpoint, POST /api/claims/verify (the S268 route over the
 * S267 verification role). The caller submits a model-generated claim and its
 * cited source; the role wraps both as untrusted data server-side under one
 * policy header and returns a fail-secure verdict (supported / unsupported /
 * uncertain, defaulting to uncertain on an ambiguous reply, never to supported).
 * The model is the user's selected model; an absent model yields a clean
 * fail-secure result rather than a guess. Every outcome -- ok or a clean
 * failure -- crosses the wire as a structured result with HTTP 200; the only
 * HTTP error is the 503 availability guard, which surfaces as an ApiError.
 */

import { apiPost } from './client';

/** The fail-secure verdict taxonomy (mirrors the S267 role). */
export type ClaimVerdict = 'supported' | 'unsupported' | 'uncertain';

/**
 * The structured verification result (mirrors ClaimVerificationResultSchema).
 *
 * verdict is the mapped taxonomy value; ok is false on a fail-secure failure;
 * reason carries the failure cause (empty on success); raw_text carries the
 * model's raw reply on success.
 */
export interface ClaimVerificationResult {
	verdict: ClaimVerdict;
	ok: boolean;
	reason: string;
	raw_text: string;
}

/**
 * Verify one claim against its cited source.
 *
 * model is optional: pass the user's selected model. An absent model returns a
 * clean fail-secure result (the server builds no client and reports the
 * failure) rather than guessing a model.
 */
export async function verifyClaim(
	claim: string,
	source: string,
	model?: string | null
): Promise<ClaimVerificationResult> {
	return apiPost<ClaimVerificationResult>('/api/claims/verify', {
		claim,
		source,
		model: model ?? null
	});
}
