/**
 * Typed API client for the per-answer verification surface (UI half).
 *
 * One per-user endpoint, POST /api/claims/verify-answer (the route over
 * the per-answer aggregation). The caller submits a batch of
 * (claim, source) pairs directly; the route runs each pair through the
 * verification role wrapped as untrusted data server-side under one policy
 * header, and aggregates the per-pair verdicts fail-secure into a single
 * per-answer verdict. Unlike the citation-verify route, this route
 * performs no extraction: the caller already holds the pairs, so the result
 * carries the aggregate plus the per-pair results but no echoed pairs (the
 * caller aligns the per-pair results with the pairs it submitted). The model is
 * the user's selected model; an absent model yields a clean fail-secure result
 * rather than a guess. An empty pairs list is a clean fail-secure failure (the
 * aggregate defaults to uncertain, ok false), not an error. Every outcome -- ok
 * or a clean failure -- crosses the wire as a structured result with HTTP 200;
 * the only HTTP error is the 503 availability guard, which surfaces as an
 * ApiError.
 */

import { apiPost } from './client';

/** The fail-secure verdict taxonomy (mirrors the role). */
export type ClaimVerdict = 'supported' | 'unsupported' | 'uncertain';

/**
 * One per-pair verification result (mirrors the per-pair
 * ClaimVerificationResultSchema).
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
 * One submitted (claim, source) pair (mirrors ClaimSourcePair).
 *
 * claim is the cited claim to check; source is the source the claim is checked
 * against. The caller supplies the batch; the per-pair results come back
 * positionally aligned with the submitted pairs.
 */
export interface ClaimSourcePair {
	claim: string;
	source: string;
}

/**
 * The structured per-answer aggregate (mirrors AnswerVerificationResultSchema).
 *
 * verdict is the aggregate taxonomy value; ok is true only when at least one
 * pair was supplied and every pair verified cleanly; reason carries the cause
 * on a not-ok aggregate; results are the per-pair verdicts, aligned with the
 * submitted pairs. There is deliberately no echoed pairs field: the caller
 * already holds the pairs it submitted.
 */
export interface AnswerVerificationResult {
	verdict: ClaimVerdict;
	ok: boolean;
	reason: string;
	results: ClaimVerificationResult[];
}

/**
 * Verify a batch of (claim, source) pairs and aggregate the verdicts fail-secure.
 *
 * pairs is the batch the caller holds. model is optional: pass the user's
 * selected model. An absent model returns a clean fail-secure result (the server
 * builds no client and reports the failure) rather than guessing a model.
 */
export async function verifyAnswer(
	pairs: ClaimSourcePair[],
	model?: string | null
): Promise<AnswerVerificationResult> {
	return apiPost<AnswerVerificationResult>('/api/claims/verify-answer', {
		pairs,
		model: model ?? null
	});
}
