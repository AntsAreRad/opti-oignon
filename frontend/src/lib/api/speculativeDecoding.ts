/**
 * Speculative Decoding API client
 *
 * Typed API for speculative decoding status, config update,
 * compatible draft model listing, and VRAM budget checks.
 * Only available when using the llama.cpp backend.
 */

import type {
	SpeculativeDecodingStatus,
	SpeculativeDecodingConfig,
	CompatibleDraftsResponse,
	VRAMBudgetResult,
} from '../types';

const BASE = '/api/speculative-decoding';

/** Get speculative decoding status, config, and stats. */
export async function getSpeculativeDecodingStatus(): Promise<SpeculativeDecodingStatus> {
	const resp = await fetch(`${BASE}/status`);
	if (!resp.ok) throw new Error(`Failed to fetch speculative decoding status: ${resp.status}`);
	return resp.json();
}

/** Update speculative decoding configuration (partial update). */
export async function updateSpeculativeDecodingConfig(
	update: Partial<SpeculativeDecodingConfig>
): Promise<SpeculativeDecodingConfig> {
	const resp = await fetch(`${BASE}/config`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(update),
	});
	if (!resp.ok) throw new Error(`Failed to update speculative decoding config: ${resp.status}`);
	return resp.json();
}

/** List compatible draft models for a given main model. */
export async function getCompatibleDrafts(
	mainModel: string,
	mainFamily?: string,
	mainParamsB?: number,
	mainQuant?: string,
): Promise<CompatibleDraftsResponse> {
	const params = new URLSearchParams({ main_model: mainModel });
	if (mainFamily) params.set('main_family', mainFamily);
	if (mainParamsB !== undefined) params.set('main_params_b', String(mainParamsB));
	if (mainQuant) params.set('main_quant', mainQuant);

	const resp = await fetch(`${BASE}/compatible-drafts?${params}`);
	if (!resp.ok) throw new Error(`Failed to fetch compatible drafts: ${resp.status}`);
	return resp.json();
}

/** Check VRAM budget for main + draft model pair. */
export async function checkVRAMBudget(
	mainParamsB: number,
	mainQuant: string,
	draftParamsB: number,
	draftQuant: string,
): Promise<VRAMBudgetResult> {
	const params = new URLSearchParams({
		main_params_b: String(mainParamsB),
		main_quant: mainQuant,
		draft_params_b: String(draftParamsB),
		draft_quant: draftQuant,
	});

	const resp = await fetch(`${BASE}/vram-budget?${params}`);
	if (!resp.ok) throw new Error(`Failed to check VRAM budget: ${resp.status}`);
	return resp.json();
}

/** Clear acceptance rate statistics. */
export async function resetSpeculativeStats(): Promise<void> {
	const resp = await fetch(`${BASE}/reset-stats`, { method: 'POST' });
	if (!resp.ok) throw new Error(`Failed to reset stats: ${resp.status}`);
}
