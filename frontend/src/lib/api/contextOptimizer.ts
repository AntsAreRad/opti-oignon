/**
 * Typed API functions for the Context Optimizer (S123).
 *
 * Provides optimizer configuration, priority presets, budget calculation
 * with preset support, and optimization report retrieval.
 */

import { apiGet, apiPut, apiPost } from './client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ZoneReport {
	zone: string;
	budgeted_tokens: number;
	actual_tokens: number;
	trimmed_tokens: number;
	strategy: string;
	detail: string;
	within_budget: boolean;
}

export interface OptimizationReport {
	model: string;
	total_window: number;
	zones: ZoneReport[];
	total_budgeted: number;
	total_actual: number;
	total_trimmed: number;
	overflow: boolean;
	preset_used: string;
	duration_ms: number;
	timestamp: number;
}

export interface OptimizerConfigResponse {
	available: boolean;
	enabled: boolean;
	active_preset?: string;
	priority_presets?: Record<string, Record<string, number>>;
	config?: Record<string, unknown>;
	message?: string;
}

export interface OptimizerConfigUpdate {
	enabled?: boolean;
	active_preset?: string;
	priority_presets?: Record<string, Record<string, number>>;
	custom_ratios?: Record<string, number>;
}

export interface OptimizerConfigUpdateResponse {
	status: string;
	enabled: boolean;
	active_preset: string;
	config: Record<string, unknown>;
}

export interface OptimizerReportResponse {
	available: boolean;
	count: number;
	total_retained: number;
	reports: OptimizationReport[];
}

export interface OptimizerPresetsResponse {
	available: boolean;
	active_preset: string;
	presets: Record<string, Record<string, number>>;
}

export interface BudgetWithPresetRequest {
	model: string;
	preset?: string;
	custom_ratios?: Record<string, number>;
	project_active?: boolean;
	context_window_override?: number;
}

export interface BudgetWithPresetResponse {
	model: string;
	preset: string;
	budget: Record<string, number>;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/** Get current optimizer configuration and presets. */
export async function getOptimizerConfig(): Promise<OptimizerConfigResponse> {
	return apiGet<OptimizerConfigResponse>('/api/context/optimizer/config');
}

/** Update optimizer configuration (enabled, preset, ratios). */
export async function updateOptimizerConfig(
	body: OptimizerConfigUpdate
): Promise<OptimizerConfigUpdateResponse> {
	return apiPut<OptimizerConfigUpdateResponse>('/api/context/optimizer/config', body);
}

/** Get last N optimization reports. */
export async function getOptimizerReports(
	lastN: number = 1
): Promise<OptimizerReportResponse> {
	return apiGet<OptimizerReportResponse>('/api/context/optimizer/report', {
		last_n: String(lastN)
	});
}

/** List all priority presets. */
export async function getOptimizerPresets(): Promise<OptimizerPresetsResponse> {
	return apiGet<OptimizerPresetsResponse>('/api/context/optimizer/presets');
}

/** Calculate budget with preset or custom ratios. */
export async function calculateBudgetWithPreset(
	body: BudgetWithPresetRequest
): Promise<BudgetWithPresetResponse> {
	return apiPost<BudgetWithPresetResponse>('/api/context/optimizer/budget', body);
}
