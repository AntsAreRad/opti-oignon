/**
 * Typed API client for the citation-verify surface (UI half).
 *
 * One per-user endpoint, POST /api/claims/verify-citations (the route,
 * the join of the citation extractor and the per-answer
 * aggregation). The caller submits a produced answer carrying inline numeric
 * citation markers [n] (1-based) plus the ordered sources those markers index;
 * the route extracts the (claim, source) pairs server-side (fail-closed by
 * omission), runs each pair through the verification role wrapped as
 * untrusted data, and aggregates the per-pair verdicts fail-secure into a
 * single per-answer verdict. The result carries the aggregate, the per-pair
 * results, and the extracted (claim, source) pairs, the pairs positionally
 * aligned with the results. The model is the user's selected model; an absent
 * model yields a clean fail-secure result rather than a guess. An answer with
 * no citations, an empty answer, or an empty sources list extracts no pairs
 * and is a clean fail-secure failure (the aggregate defaults to uncertain, ok
 * false), not an error. Every outcome -- ok or a clean failure -- crosses the
 * wire as a structured result with HTTP 200; the only HTTP error is the 503
 * availability guard, which surfaces as an ApiError.
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
 * One extracted (claim, source) pair (mirrors ClaimSourcePair).
 *
 * claim is the cited sentence with its markers stripped; source is the cited
 * source the claim is checked against. The pairs are positionally aligned with
 * the per-pair results.
 */
export interface ClaimSourcePair {
	claim: string;
	source: string;
}

/**
 * The structured per-answer aggregate (mirrors CitationVerificationResultSchema).
 *
 * verdict is the aggregate taxonomy value; ok is true only when at least one
 * pair was extracted and every pair verified cleanly; reason carries the cause
 * on a not-ok aggregate; results are the per-pair verdicts; pairs are the
 * (claim, source) pairs the parser extracted, aligned with results.
 */
export interface CitationVerificationResult {
	verdict: ClaimVerdict;
	ok: boolean;
	reason: string;
	results: ClaimVerificationResult[];
	pairs: ClaimSourcePair[];
}

/**
 * Verify the cited claims in one produced answer against their ordered sources.
 *
 * answer carries inline numeric markers [n] (1-based) indexing sources by
 * position. model is optional: pass the user's selected model. An absent model
 * returns a clean fail-secure result (the server builds no client and reports
 * the failure) rather than guessing a model.
 */
export async function verifyCitations(
	answer: string,
	sources: string[],
	model?: string | null
): Promise<CitationVerificationResult> {
	return apiPost<CitationVerificationResult>('/api/claims/verify-citations', {
		answer,
		sources,
		model: model ?? null
	});
}
